/**
 *  @file src/smallfield/cuda/preprocess_kernel.cuh
 *  @brief The two device passes that close the gap between the preprocessing kernels and the pipeline that
 *  consumes their output.
 *
 *  `mmff94_type_and_charge_kernel` leaves a per-molecule status behind, and the molecules it flags need
 *  atom types and charges from somewhere else before the extractor runs; `mmff94_apply_typing_patch_kernel`
 *  writes the host's answer for exactly those molecules.
 *
 *  `csd_torsion_match_kernel` emits a torsion list per molecule, and `etkdg_pipeline_warp_kernel` reads
 *  those torsions through a per-atom CSR; `etk_build_csr_kernel` builds that CSR on the device from the
 *  torsions the matcher wrote and the 1-2 / 1-3 constraints the host emitted, mirroring
 *  `build_etk_interaction_lists`.
 */

#pragma once

#if defined(__CUDACC__)

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <cuda_runtime.h>

#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::cuda {

using smallfield::types::atom_role_first_k;
using smallfield::types::atom_role_fourth_k;
using smallfield::types::atom_role_second_k;
using smallfield::types::atom_role_third_k;
using smallfield::types::distance_constraint_t;
using smallfield::types::etk_atom_interaction_t;
using smallfield::types::etk_atom_offsets_t;
using smallfield::types::etk_distance_constraint_k;
using smallfield::types::etk_torsion_descriptor_t;
using smallfield::types::etk_torsion_k;
using smallfield::types::etkdg_molecule_metadata_t;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::mmff94_typing_patch_t;

/**
 *  @brief Overwrite the typer's output for the molecules the host already knows it cannot type.
 *
 *  One warp per patch record, lane-strided over the molecule's atoms. Clearing the typing status
 *  is what routes the molecule back into the normal extractor path.
 *
 *  @param[in] patches One record per flagged molecule.
 *  @param[in] num_patches Records in @p patches.
 *  @param[in] staged_atom_types Host-computed atom types, concatenated in record order.
 *  @param[in] staged_atom_charges Host-computed partial charges, concatenated in record order.
 *  @param[out] atom_types Batch-global per-atom type array the extractor reads.
 *  @param[out] atom_charges Batch-global per-atom charge array the extractor reads.
 *  @param[out] typing_status Per-molecule typer status; the patched molecules are cleared to zero.
 */
static __global__ __launch_bounds__(32, 8) void mmff94_apply_typing_patch_kernel(
    mmff94_typing_patch_t const *__restrict__ patches, std::uint32_t num_patches,
    std::uint8_t const *__restrict__ staged_atom_types, float const *__restrict__ staged_atom_charges,
    std::uint8_t *__restrict__ atom_types, float *__restrict__ atom_charges, std::uint8_t *__restrict__ typing_status) {

    std::uint32_t const record = blockIdx.x;
    if (record >= num_patches) return;
    mmff94_typing_patch_t const patch = patches[record];
    for (std::uint32_t atom = threadIdx.x; atom < patch.atom_count; atom += 32) {
        atom_types[patch.atom_base + atom] = staged_atom_types[patch.patch_base + atom];
        atom_charges[patch.atom_base + atom] = staged_atom_charges[patch.patch_base + atom];
    }
    if (threadIdx.x == 0) typing_status[patch.molecule_index] = 0;
}

/**
 *  @brief Build the per-atom ETK interaction CSR from the matched torsions and the host constraints.
 *
 *  One block per molecule, lane 0 only: the two counting passes carry per-atom cursors that a
 *  lane-parallel form would have to serialise anyway, and a drug-like molecule spends tens of
 *  microseconds of a millisecond-scale wave here. Term indices stay molecule-relative because
 *  `etk_torsion_energy_functor_t` indexes the molecule's own torsion and constraint slices, while
 *  the interaction starts are batch-global because it indexes one interaction array.
 *
 *  @param[in] torsions Batch-global torsion array the matcher filled at a fixed per-molecule stride.
 *  @param[in] torsion_stride Slots the matcher reserved per molecule.
 *  @param[in] torsion_counts Torsions the matcher emitted per molecule, before truncation.
 *  @param[in] constraints Batch-global 1-2 / 1-3 constraint array.
 *  @param[in] num_molecules Molecules in the batch.
 *  @param[out] interactions Batch-global ETK interaction array.
 *  @param[out] atom_offsets Batch-global per-atom CSR entries.
 *  @param[inout] metadata Per-molecule metadata; the torsion count is stamped here.
 */
static __global__ __launch_bounds__(32, 8) void etk_build_csr_kernel(
    etk_torsion_descriptor_t const *__restrict__ torsions, std::uint32_t torsion_stride,
    std::uint32_t const *__restrict__ torsion_counts, distance_constraint_t const *__restrict__ constraints,
    std::uint32_t num_molecules, etk_atom_interaction_t *__restrict__ interactions,
    etk_atom_offsets_t *__restrict__ atom_offsets, etkdg_molecule_metadata_t *__restrict__ metadata) {

    __shared__ std::uint32_t cursors[max_atoms_per_molecule_k];

    std::uint32_t const molecule_index = blockIdx.x;
    if (molecule_index >= num_molecules) return;
    if (threadIdx.x != 0) return;

    auto &meta = metadata[molecule_index];
    std::uint32_t const num_atoms = meta.num_atoms;
    std::uint32_t const num_constraints = meta.num_distance_constraints;
    // The matcher reports what it matched; the reservation is what fits. They agree for every
    // molecule whose torsion set stays inside its bond list, which is every molecule the matcher
    // can produce, and the clamp keeps a future widening of the rule from writing out of bounds.
    std::uint32_t torsion_count = torsion_counts[molecule_index];
    if (torsion_count > torsion_stride) torsion_count = torsion_stride;
    meta.num_etk_torsions = torsion_count;

    etk_torsion_descriptor_t const *mol_torsions = torsions + (std::size_t)molecule_index * torsion_stride;
    distance_constraint_t const *mol_constraints = constraints + meta.constraint_offset;
    etk_atom_offsets_t *mol_atom_offsets = atom_offsets + meta.etk_atom_csr_offset;
    etk_atom_interaction_t *mol_interactions = interactions + meta.etk_interaction_offset;

    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) mol_atom_offsets[atom] = {0u, 0u};
    for (std::uint32_t term = 0; term < torsion_count; ++term)
        for (std::uint32_t position = 0; position < 4; ++position)
            mol_atom_offsets[mol_torsions[term].atom_indices[position]].interaction_count++;
    for (std::uint32_t term = 0; term < num_constraints; ++term) {
        mol_atom_offsets[mol_constraints[term].atom_index_i].interaction_count++;
        mol_atom_offsets[mol_constraints[term].atom_index_j].interaction_count++;
    }

    std::uint32_t running = 0;
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        mol_atom_offsets[atom].interaction_offset = meta.etk_interaction_offset + running;
        cursors[atom] = running;
        running += mol_atom_offsets[atom].interaction_count;
    }

    for (std::uint32_t term = 0; term < torsion_count; ++term) {
        auto const &torsion = mol_torsions[term];
        mol_interactions[cursors[torsion.atom_indices[0]]++] = {term, etk_torsion_k, atom_role_first_k};
        mol_interactions[cursors[torsion.atom_indices[1]]++] = {term, etk_torsion_k, atom_role_second_k};
        mol_interactions[cursors[torsion.atom_indices[2]]++] = {term, etk_torsion_k, atom_role_third_k};
        mol_interactions[cursors[torsion.atom_indices[3]]++] = {term, etk_torsion_k, atom_role_fourth_k};
    }
    for (std::uint32_t term = 0; term < num_constraints; ++term) {
        auto const &constraint = mol_constraints[term];
        mol_interactions[cursors[constraint.atom_index_i]++] = {term, etk_distance_constraint_k, atom_role_first_k};
        mol_interactions[cursors[constraint.atom_index_j]++] = {term, etk_distance_constraint_k, atom_role_second_k};
    }
}

/** Launch `mmff94_apply_typing_patch_kernel`, one warp per flagged molecule. */
inline void launch_mmff94_apply_typing_patch(mmff94_typing_patch_t const *patches, std::uint32_t num_patches,
                                             std::uint8_t const *staged_atom_types, float const *staged_atom_charges,
                                             std::uint8_t *atom_types, float *atom_charges, std::uint8_t *typing_status,
                                             cudaStream_t stream = nullptr) {
    if (num_patches == 0) return;
    mmff94_apply_typing_patch_kernel<<<num_patches, 32, 0, stream>>>(
        patches, num_patches, staged_atom_types, staged_atom_charges, atom_types, atom_charges, typing_status);
}

/** Launch `etk_build_csr_kernel`, one block per molecule. */
inline void launch_etk_build_csr(etk_torsion_descriptor_t const *torsions, std::uint32_t torsion_stride,
                                 std::uint32_t const *torsion_counts, distance_constraint_t const *constraints,
                                 std::uint32_t num_molecules, etk_atom_interaction_t *interactions,
                                 etk_atom_offsets_t *atom_offsets, etkdg_molecule_metadata_t *metadata,
                                 cudaStream_t stream = nullptr) {
    if (num_molecules == 0) return;
    etk_build_csr_kernel<<<num_molecules, 32, 0, stream>>>(torsions, torsion_stride, torsion_counts, constraints,
                                                           num_molecules, interactions, atom_offsets, metadata);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
