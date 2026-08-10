/**
 *  @file mmff94_typer_device.cuh
 *  @brief `mmff94_type_and_charge_kernel` runs the native MMFF atom typer and charge computation
 *  on the GPU.
 *
 *  Each block = one warp = one molecule. Inside the block, atoms are
 *  assigned lane-strided (lane `l` handles atoms `l, l+32, ...`) for the
 *  heavy-atom and hydrogen-typer passes. The BCI flood fill (needed for
 *  carboxylate / imidazolium / guanidinium charge sharing) has cross-atom
 *  dependencies; it runs lane-0-only with the rest of the warp idle.
 *  On drug-like ChEMBL that's ≤ 3 atoms per molecule — amortised cost is
 *  negligible vs the typer + partial-charge passes, which @b are parallel.
 *
 *  Inputs:
 *   - `descriptors[mol_index]` — device-resident `molecule_descriptor_t`
 *     (22 KB per mol). Uploaded by the pipeline caller.
 *   - `tables_view` — `mmff94_tables_view_t` with device-pointer spans
 *     (built by `upload_mmff94_tables_to_device`, see
 *     `mmff94_tables_device.cuh`).
 *
 *  Outputs:
 *   - `atom_types[mol_atom_offset ...]` — u8 per atom.
 *   - `atom_charges[mol_atom_offset ...]` — f32 per atom.
 *   - `typing_status[mol_index]` — 0 on full success, non-zero if any atom
 *     failed to type. Non-zero molecules route to the CPU fallback.
 */

#pragma once

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <cuda_runtime.h>

#include "smallfield/module_fwd.hpp"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/mmff94/mmff94_typer.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/cuda/warp_primitives.cuh"

namespace smallfield::mmff94 {
using smallfield::cuda::warp_size_k;
using smallfield::parameters::mmff94_tables_view_t;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::molecule_descriptor_t;

/** Kernel: one block per molecule, 32 threads per block. Reads the
 *  molecule's descriptor from global memory (not shared — typer only
 *  makes one pass per atom, so the caching doesn't amortise). Writes
 *  atom types to `atom_types_out + atom_offsets[mol_index]` and charges
 *  to `atom_charges_out + atom_offsets[mol_index]`. */
static __launch_bounds__(32, 8) __global__
    void mmff94_type_and_charge_kernel(molecule_descriptor_t const *__restrict__ descriptors,
                                       std::uint32_t const *__restrict__ atom_offsets, std::uint32_t num_molecules,
                                       mmff94_tables_view_t tables_view, std::uint8_t *__restrict__ atom_types_out,
                                       float *__restrict__ atom_charges_out, std::uint8_t *__restrict__ typing_status) {

    std::uint32_t const mol = blockIdx.x;
    if (mol >= num_molecules) return;
    std::uint32_t const lane = threadIdx.x;

    molecule_descriptor_t const &descriptor = descriptors[mol];
    std::uint32_t const atom_off = atom_offsets[mol];
    std::uint32_t const num_atoms = descriptor.num_atoms;
    std::uint8_t *types = atom_types_out + atom_off;
    float *charges = atom_charges_out + atom_off;

    // Heavy atoms, lane-strided.
    for (std::uint32_t a = lane; a < num_atoms; a += warp_size_k) {
        if (descriptor.element[a] == 1) {
            types[a] = 0;
            continue;
        }
        types[a] = native_set_mmff_heavy_atom_type(descriptor, a);
    }
    __syncwarp();

    // Hydrogens, keyed on heavy-atom types, so must run after the heavy-atom pass.
    std::span<std::uint8_t const> types_span(types, num_atoms);
    for (std::uint32_t a = lane; a < num_atoms; a += warp_size_k) {
        if (descriptor.element[a] != 1) continue;
        types[a] = native_set_mmff_hydrogen_type(descriptor, a, types_span);
    }
    __syncwarp();

    // Lane-0-only charge computation; the flood fill is serial.
    // Reserve per-molecule scratch for formal charges in a shared buffer
    // owned by this block — fits within max_atoms_per_molecule_k = 128 ≤ 512 B.
    __shared__ float formal_charges_shared[max_atoms_per_molecule_k];
    if (lane == 0) {
        // Status: if any atom type is 0, this molecule failed typing.
        std::uint8_t status = 0;
        for (std::uint32_t a = 0; a < num_atoms; ++a)
            if (types[a] == 0) {
                status = 1;
                break;
            }
        typing_status[mol] = status;
        if (status == 0) {
            std::span<std::uint8_t const> at_const(types, num_atoms);
            std::span<float> fc_span(formal_charges_shared, num_atoms);
            native_compute_mmff_formal_charges(descriptor, at_const, fc_span);
            std::span<float const> fc_const(formal_charges_shared, num_atoms);
            std::span<float> pc_span(charges, num_atoms);
            native_compute_mmff_partial_charges(descriptor, tables_view, at_const, fc_const, pc_span);
        }
        else {
            // Failed — zero charges as a defensive default. CPU fallback
            // will overwrite with the full native pipeline.
            for (std::uint32_t a = 0; a < num_atoms; ++a) charges[a] = 0.0f;
        }
    }
}

/** Host launcher — thin wrapper so the .cu definition is isolated. */
inline void launch_mmff94_type_and_charge(molecule_descriptor_t const *descriptors_dev,
                                          std::uint32_t const *atom_offsets_dev, std::uint32_t num_molecules,
                                          mmff94_tables_view_t tables_view, std::uint8_t *atom_types_out_dev,
                                          float *atom_charges_out_dev, std::uint8_t *typing_status_dev,
                                          cudaStream_t stream = nullptr) {
    if (num_molecules == 0) return;
    mmff94_type_and_charge_kernel<<<num_molecules, 32, 0, stream>>>(descriptors_dev, atom_offsets_dev, num_molecules,
                                                                    tables_view, atom_types_out_dev,
                                                                    atom_charges_out_dev, typing_status_dev);
}

} // namespace smallfield::mmff94
