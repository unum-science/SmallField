/**
 *  @file src/smallfield/mmff94/mmff94_typer_device.cuh
 *  @brief `mmff94_type_and_charge_kernel` runs the native MMFF atom typer and charge computation on the GPU.
 *
 *  One warp per molecule, atoms lane-strided through the heavy-atom and hydrogen passes. The BCI flood fill
 *  that carboxylate, imidazolium and guanidinium charge sharing needs has cross-atom dependencies, so it
 *  runs on lane 0 alone; on drug-like ChEMBL that reaches at most 3 atoms per molecule.
 */

#pragma once

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <cuda_runtime.h>

#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/mmff94/mmff94_typer.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/views.hpp"
#include "smallfield/cuda/warp_primitives.cuh"

namespace smallfield::mmff94 {
using smallfield::cuda::warp_size_k;
using smallfield::parameters::mmff94_tables_view_t;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::molecule_descriptor_t;

/**
 *  @brief Type and charge one molecule per block.
 *
 *  The descriptor stays in global memory: the typer makes one pass per atom, so staging it into shared
 *  never amortises. `typing_status` is zero when every atom typed and non-zero otherwise, which is what
 *  routes a molecule to the CPU fallback.
 */
static __launch_bounds__(32, 8) __global__
    void mmff94_type_and_charge_kernel(molecule_descriptor_t const *__restrict__ descriptors,
                                       std::uint32_t const *__restrict__ atom_offsets, std::uint32_t num_molecules,
                                       mmff94_tables_view_t tables_view, std::uint8_t *__restrict__ atom_types_out,
                                       float *__restrict__ atom_charges_out, std::uint8_t *__restrict__ typing_status) {

    std::uint32_t const molecule = blockIdx.x;
    if (molecule >= num_molecules) return;
    std::uint32_t const lane = threadIdx.x;

    molecule_descriptor_t const &descriptor = descriptors[molecule];
    std::uint32_t const atom_off = atom_offsets[molecule];
    std::uint32_t const num_atoms = descriptor.num_atoms;
    std::uint8_t *types = atom_types_out + atom_off;
    float *charges = atom_charges_out + atom_off;

    // Heavy atoms, lane-strided.
    for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k) {
        if (descriptor.element[atom] == 1) {
            types[atom] = 0;
            continue;
        }
        types[atom] = native_set_mmff_heavy_atom_type(descriptor, atom);
    }
    __syncwarp();

    // Hydrogens, keyed on heavy-atom types, so must run after the heavy-atom pass.
    custd::span<std::uint8_t const> types_span(types, num_atoms);
    for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k) {
        if (descriptor.element[atom] != 1) continue;
        types[atom] = native_set_mmff_hydrogen_type(descriptor, atom, types_span);
    }
    __syncwarp();

    // Lane-0-only charge computation; the flood fill is serial. The formal-charge scratch is block-scoped
    // so it costs one array per warp rather than one per thread.
    __shared__ float formal_charges_shared[max_atoms_per_molecule_k];
    if (lane == 0) {
        std::uint8_t status = 0;
        for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
            if (types[atom] == 0) {
                status = 1;
                break;
            }
        typing_status[molecule] = status;
        if (status == 0) {
            custd::span<std::uint8_t const> at_const(types, num_atoms);
            custd::span<float> fc_span(formal_charges_shared, num_atoms);
            native_compute_mmff_formal_charges(descriptor, at_const, fc_span);
            custd::span<float const> fc_const(formal_charges_shared, num_atoms);
            custd::span<float> pc_span(charges, num_atoms);
            native_compute_mmff_partial_charges(descriptor, tables_view, at_const, fc_const, pc_span);
        }
        else {
            // Zeroed rather than left stale; the CPU fallback overwrites them.
            for (std::uint32_t atom = 0; atom < num_atoms; ++atom) charges[atom] = 0.0f;
        }
    }
}

} // namespace smallfield::mmff94
