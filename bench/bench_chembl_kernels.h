/**
 * @file bench_chembl_kernels.h
 * @brief Shared interface for GPU pipeline benchmark wrapper.
 *
 * Included by both bench_chembl.cpp (g++) and bench_chembl_kernels.cu (nvcc).
 * Uses extern "C" linkage because g++ and nvcc have different C++ name mangling.
 */

#pragma once

#include <cstdint>

#include "etkdg_types.cuh"

struct gpu_pipeline_result_t {
    float upload_time_ms;
    float kernel_time_ms;
    float download_time_ms;
    std::uint32_t num_succeeded;
    std::uint32_t grid_size;
    std::uint32_t blocks_per_sm;
};

extern "C" gpu_pipeline_result_t gpu_etkdg_mmff_pipeline(
    float const* host_bounds, std::uint32_t bounds_total_floats,
    smallfield::chiral_center_descriptor_t const* host_chirals, std::uint32_t chirals_total,
    smallfield::etk_torsion_descriptor_t const* host_torsions, std::uint32_t torsions_total,
    smallfield::distance_constraint_t const* host_constraints, std::uint32_t constraints_total,
    smallfield::mmff94_nonbonded_pair_t const* host_mmff_nb, std::uint32_t mmff_nb_total,
    smallfield::mmff94_bond_t const* host_mmff_bonds, std::uint32_t mmff_bonds_total,
    smallfield::mmff94_angle_t const* host_mmff_angles, std::uint32_t mmff_angles_total,
    smallfield::mmff94_torsion_t const* host_mmff_torsions, std::uint32_t mmff_torsions_total,
    smallfield::mmff94_oop_t const* host_mmff_oops, std::uint32_t mmff_oops_total,
    smallfield::mmff94_stretchbend_t const* host_mmff_stbn, std::uint32_t mmff_stbn_total,
    smallfield::mmff94_atom_interaction_t const* host_mmff_interactions, std::uint32_t mmff_interactions_total,
    smallfield::mmff94_atom_offsets_t const* host_mmff_offsets, std::uint32_t mmff_offsets_total,
    smallfield::etkdg_molecule_metadata_t const* host_metadata, std::uint32_t num_molecules, float* host_coords_out,
    float* host_energies_out, std::uint8_t* host_status_out, std::uint32_t num_conformers,
    std::uint32_t max_stereo_attempts, std::uint32_t max_atoms_per_molecule, std::uint64_t seed);
