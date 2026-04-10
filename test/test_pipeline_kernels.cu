/**
 * @file test_pipeline_kernels.cu
 * @brief CUDA kernel wrapper for end-to-end ETKDG pipeline test. Compiled by nvcc.
 */

#include "etkdg_pipeline.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

using namespace smallfield;

extern "C" {

void gpu_etkdg_pipeline(
    float const* host_bounds, std::uint32_t bounds_total_floats, float const* host_initial_coordinates,
    std::uint32_t initial_coordinates_total, chiral_center_descriptor_t const* host_chirals,
    std::uint32_t chirals_total, etk_torsion_descriptor_t const* host_torsions, std::uint32_t torsions_total,
    distance_constraint_t const* host_constraints, std::uint32_t constraints_total,
    mmff94_nonbonded_pair_t const* host_mmff_nb, std::uint32_t mmff_nb_total, mmff94_bond_t const* host_mmff_bonds,
    std::uint32_t mmff_bonds_total, mmff94_angle_t const* host_mmff_angles, std::uint32_t mmff_angles_total,
    mmff94_torsion_t const* host_mmff_torsions, std::uint32_t mmff_torsions_total, mmff94_oop_t const* host_mmff_oops,
    std::uint32_t mmff_oops_total, mmff94_stretchbend_t const* host_mmff_stbn, std::uint32_t mmff_stbn_total,
    mmff94_atom_interaction_t const* host_mmff_interactions, std::uint32_t mmff_interactions_total,
    mmff94_atom_offsets_t const* host_mmff_offsets, std::uint32_t mmff_offsets_total,
    etkdg_molecule_metadata_t const* host_metadata, std::uint32_t num_molecules, float* host_coords_out,
    float* host_energies_out, std::uint8_t* host_status_out, std::uint32_t num_conformers,
    std::uint32_t max_stereo_attempts, std::uint32_t max_atoms_per_molecule, std::uint64_t seed) {

    // allocate device memory
    float* dev_bounds;
    float* dev_initial_coordinates;
    chiral_center_descriptor_t* dev_chirals;
    etk_torsion_descriptor_t* dev_torsions;
    distance_constraint_t* dev_constraints;
    mmff94_nonbonded_pair_t* dev_mmff_nb;
    mmff94_bond_t* dev_mmff_bonds;
    mmff94_angle_t* dev_mmff_angles;
    mmff94_torsion_t* dev_mmff_torsions;
    mmff94_oop_t* dev_mmff_oops;
    mmff94_stretchbend_t* dev_mmff_stbn;
    mmff94_atom_interaction_t* dev_mmff_interactions;
    mmff94_atom_offsets_t* dev_mmff_offsets;
    etkdg_molecule_metadata_t* dev_metadata;
    float* dev_coords_out;
    float* dev_energies_out;
    std::uint8_t* dev_status_out;

    cudaMalloc(&dev_bounds, bounds_total_floats * sizeof(float));
    cudaMalloc(&dev_initial_coordinates, std::max(initial_coordinates_total, 1u) * sizeof(float));
    cudaMalloc(&dev_chirals, std::max(chirals_total, 1u) * sizeof(chiral_center_descriptor_t));
    cudaMalloc(&dev_torsions, std::max(torsions_total, 1u) * sizeof(etk_torsion_descriptor_t));
    cudaMalloc(&dev_constraints, std::max(constraints_total, 1u) * sizeof(distance_constraint_t));
    cudaMalloc(&dev_mmff_nb, std::max(mmff_nb_total, 1u) * sizeof(mmff94_nonbonded_pair_t));
    cudaMalloc(&dev_mmff_bonds, std::max(mmff_bonds_total, 1u) * sizeof(mmff94_bond_t));
    cudaMalloc(&dev_mmff_angles, std::max(mmff_angles_total, 1u) * sizeof(mmff94_angle_t));
    cudaMalloc(&dev_mmff_torsions, std::max(mmff_torsions_total, 1u) * sizeof(mmff94_torsion_t));
    cudaMalloc(&dev_mmff_oops, std::max(mmff_oops_total, 1u) * sizeof(mmff94_oop_t));
    cudaMalloc(&dev_mmff_stbn, std::max(mmff_stbn_total, 1u) * sizeof(mmff94_stretchbend_t));
    cudaMalloc(&dev_mmff_interactions, std::max(mmff_interactions_total, 1u) * sizeof(mmff94_atom_interaction_t));
    cudaMalloc(&dev_mmff_offsets, std::max(mmff_offsets_total, 1u) * sizeof(mmff94_atom_offsets_t));
    cudaMalloc(&dev_metadata, num_molecules * sizeof(etkdg_molecule_metadata_t));

    std::uint32_t const max_total_conformers = num_molecules * num_conformers;
    std::uint32_t const max_coords = max_total_conformers * max_atoms_per_molecule * 3;
    cudaMalloc(&dev_coords_out, max_coords * sizeof(float));
    cudaMalloc(&dev_energies_out, max_total_conformers * sizeof(float));
    cudaMalloc(&dev_status_out, max_total_conformers * sizeof(std::uint8_t));

    // upload
    cudaMemcpy(dev_bounds, host_bounds, bounds_total_floats * sizeof(float), cudaMemcpyHostToDevice);
    if (initial_coordinates_total > 0)
        cudaMemcpy(dev_initial_coordinates, host_initial_coordinates, initial_coordinates_total * sizeof(float),
                   cudaMemcpyHostToDevice);
    if (chirals_total > 0)
        cudaMemcpy(dev_chirals, host_chirals, chirals_total * sizeof(chiral_center_descriptor_t),
                   cudaMemcpyHostToDevice);
    if (torsions_total > 0)
        cudaMemcpy(dev_torsions, host_torsions, torsions_total * sizeof(etk_torsion_descriptor_t),
                   cudaMemcpyHostToDevice);
    if (constraints_total > 0)
        cudaMemcpy(dev_constraints, host_constraints, constraints_total * sizeof(distance_constraint_t),
                   cudaMemcpyHostToDevice);
    if (mmff_nb_total > 0)
        cudaMemcpy(dev_mmff_nb, host_mmff_nb, mmff_nb_total * sizeof(mmff94_nonbonded_pair_t), cudaMemcpyHostToDevice);
    if (mmff_bonds_total > 0)
        cudaMemcpy(dev_mmff_bonds, host_mmff_bonds, mmff_bonds_total * sizeof(mmff94_bond_t), cudaMemcpyHostToDevice);
    if (mmff_angles_total > 0)
        cudaMemcpy(dev_mmff_angles, host_mmff_angles, mmff_angles_total * sizeof(mmff94_angle_t),
                   cudaMemcpyHostToDevice);
    if (mmff_torsions_total > 0)
        cudaMemcpy(dev_mmff_torsions, host_mmff_torsions, mmff_torsions_total * sizeof(mmff94_torsion_t),
                   cudaMemcpyHostToDevice);
    if (mmff_oops_total > 0)
        cudaMemcpy(dev_mmff_oops, host_mmff_oops, mmff_oops_total * sizeof(mmff94_oop_t), cudaMemcpyHostToDevice);
    if (mmff_stbn_total > 0)
        cudaMemcpy(dev_mmff_stbn, host_mmff_stbn, mmff_stbn_total * sizeof(mmff94_stretchbend_t),
                   cudaMemcpyHostToDevice);
    if (mmff_interactions_total > 0)
        cudaMemcpy(dev_mmff_interactions, host_mmff_interactions,
                   mmff_interactions_total * sizeof(mmff94_atom_interaction_t), cudaMemcpyHostToDevice);
    if (mmff_offsets_total > 0)
        cudaMemcpy(dev_mmff_offsets, host_mmff_offsets, mmff_offsets_total * sizeof(mmff94_atom_offsets_t),
                   cudaMemcpyHostToDevice);
    cudaMemcpy(dev_metadata, host_metadata, num_molecules * sizeof(etkdg_molecule_metadata_t), cudaMemcpyHostToDevice);

    // compute shared memory size
    std::uint32_t const shared_bytes = compute_pipeline_shared_memory_bytes(max_atoms_per_molecule, 0);
    std::printf("  pipeline: %u molecules, %u conformers each, max_atoms=%u, shared_mem=%u bytes\n", num_molecules,
                num_conformers, max_atoms_per_molecule, shared_bytes);

    // opt-in to extended shared memory
    cudaFuncSetAttribute(etkdg_pipeline_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);

    // warp-loop launch: min(num_molecules, 752) warps, each loops with stride

    // compute optimal grid size from occupancy
    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, etkdg_pipeline_warp_kernel, 32, shared_bytes);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    std::uint32_t const grid_size = std::min(num_molecules, static_cast<std::uint32_t>(max_blocks_per_sm * num_sms));
    etkdg_pipeline_warp_kernel<<<grid_size, 32, shared_bytes>>>(
        dev_bounds, initial_coordinates_total > 0 ? dev_initial_coordinates : nullptr, dev_chirals, dev_torsions,
        dev_constraints, dev_mmff_nb, dev_mmff_bonds, dev_mmff_angles, dev_mmff_torsions, dev_mmff_oops, dev_mmff_stbn,
        dev_mmff_interactions, dev_mmff_offsets, dev_metadata, dev_coords_out, dev_energies_out, dev_status_out,
        num_molecules, num_conformers, max_stereo_attempts, max_atoms_per_molecule, max_atoms_per_molecule, seed);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "  CUDA error: %s\n", cudaGetErrorString(err));
    }

    // download results
    cudaMemcpy(host_coords_out, dev_coords_out, max_coords * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_energies_out, dev_energies_out, max_total_conformers * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_status_out, dev_status_out, max_total_conformers * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(dev_bounds);
    cudaFree(dev_initial_coordinates);
    cudaFree(dev_chirals);
    cudaFree(dev_torsions);
    cudaFree(dev_constraints);
    cudaFree(dev_mmff_nb);
    cudaFree(dev_mmff_bonds);
    cudaFree(dev_mmff_angles);
    cudaFree(dev_mmff_torsions);
    cudaFree(dev_mmff_oops);
    cudaFree(dev_mmff_stbn);
    cudaFree(dev_mmff_interactions);
    cudaFree(dev_mmff_offsets);
    cudaFree(dev_metadata);
    cudaFree(dev_coords_out);
    cudaFree(dev_energies_out);
    cudaFree(dev_status_out);
}

} // extern "C"
