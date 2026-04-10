/**
 * @file bench_chembl_kernels.cu
 * @brief CUDA kernel wrapper for the unified ChEMBL benchmark.
 *
 * Handles GPU memory allocation, upload, kernel launch with occupancy-based
 * grid sizing, CUDA event timing, and result download.
 */

#include "etkdg_pipeline.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

using namespace smallfield;

extern "C" {

struct gpu_pipeline_result_t {
    float kernel_time_ms;
    std::uint32_t num_succeeded;
};

gpu_pipeline_result_t gpu_etkdg_mmff_pipeline(
    float const* host_bounds, std::uint32_t bounds_total_floats,
    chiral_center_descriptor_t const* host_chirals, std::uint32_t chirals_total,
    etk_torsion_descriptor_t const* host_torsions, std::uint32_t torsions_total,
    distance_constraint_t const* host_constraints, std::uint32_t constraints_total,
    mmff94_nonbonded_pair_t const* host_mmff_nb, std::uint32_t mmff_nb_total,
    mmff94_bond_t const* host_mmff_bonds, std::uint32_t mmff_bonds_total,
    mmff94_angle_t const* host_mmff_angles, std::uint32_t mmff_angles_total,
    mmff94_torsion_t const* host_mmff_torsions, std::uint32_t mmff_torsions_total,
    mmff94_atom_interaction_t const* host_mmff_interactions, std::uint32_t mmff_interactions_total,
    mmff94_atom_offsets_t const* host_mmff_offsets, std::uint32_t mmff_offsets_total,
    etkdg_molecule_metadata_t const* host_metadata, std::uint32_t num_molecules,
    float* host_coords_out, float* host_energies_out, std::uint8_t* host_status_out,
    std::uint32_t num_conformers, std::uint32_t max_stereo_attempts,
    std::uint32_t max_atoms_per_molecule, std::uint64_t seed) {

    // allocate device memory
    float* dev_bounds;
    chiral_center_descriptor_t* dev_chirals;
    etk_torsion_descriptor_t* dev_torsions;
    distance_constraint_t* dev_constraints;
    mmff94_nonbonded_pair_t* dev_mmff_nb;
    mmff94_bond_t* dev_mmff_bonds;
    mmff94_angle_t* dev_mmff_angles;
    mmff94_torsion_t* dev_mmff_torsions;
    mmff94_atom_interaction_t* dev_mmff_inter;
    mmff94_atom_offsets_t* dev_mmff_offsets;
    etkdg_molecule_metadata_t* dev_metadata;

    cudaMalloc(&dev_bounds, bounds_total_floats * sizeof(float));
    cudaMalloc(&dev_chirals, std::max(chirals_total, 1u) * sizeof(chiral_center_descriptor_t));
    cudaMalloc(&dev_torsions, std::max(torsions_total, 1u) * sizeof(etk_torsion_descriptor_t));
    cudaMalloc(&dev_constraints, std::max(constraints_total, 1u) * sizeof(distance_constraint_t));
    cudaMalloc(&dev_mmff_nb, std::max(mmff_nb_total, 1u) * sizeof(mmff94_nonbonded_pair_t));
    cudaMalloc(&dev_mmff_bonds, std::max(mmff_bonds_total, 1u) * sizeof(mmff94_bond_t));
    cudaMalloc(&dev_mmff_angles, std::max(mmff_angles_total, 1u) * sizeof(mmff94_angle_t));
    cudaMalloc(&dev_mmff_torsions, std::max(mmff_torsions_total, 1u) * sizeof(mmff94_torsion_t));
    cudaMalloc(&dev_mmff_inter, std::max(mmff_interactions_total, 1u) * sizeof(mmff94_atom_interaction_t));
    cudaMalloc(&dev_mmff_offsets, std::max(mmff_offsets_total, 1u) * sizeof(mmff94_atom_offsets_t));
    cudaMalloc(&dev_metadata, num_molecules * sizeof(etkdg_molecule_metadata_t));

    std::uint32_t const total_conf = num_molecules * num_conformers;
    std::uint32_t const total_coords = total_conf * max_atoms_per_molecule * 3;
    float* dev_coords_out;
    float* dev_energies_out;
    std::uint8_t* dev_status_out;
    cudaMalloc(&dev_coords_out, total_coords * sizeof(float));
    cudaMalloc(&dev_energies_out, total_conf * sizeof(float));
    cudaMalloc(&dev_status_out, total_conf * sizeof(std::uint8_t));

    // upload
    cudaMemcpy(dev_bounds, host_bounds, bounds_total_floats * sizeof(float), cudaMemcpyHostToDevice);
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
        cudaMemcpy(dev_mmff_nb, host_mmff_nb, mmff_nb_total * sizeof(mmff94_nonbonded_pair_t),
                   cudaMemcpyHostToDevice);
    if (mmff_bonds_total > 0)
        cudaMemcpy(dev_mmff_bonds, host_mmff_bonds, mmff_bonds_total * sizeof(mmff94_bond_t),
                   cudaMemcpyHostToDevice);
    if (mmff_angles_total > 0)
        cudaMemcpy(dev_mmff_angles, host_mmff_angles, mmff_angles_total * sizeof(mmff94_angle_t),
                   cudaMemcpyHostToDevice);
    if (mmff_torsions_total > 0)
        cudaMemcpy(dev_mmff_torsions, host_mmff_torsions, mmff_torsions_total * sizeof(mmff94_torsion_t),
                   cudaMemcpyHostToDevice);
    if (mmff_interactions_total > 0)
        cudaMemcpy(dev_mmff_inter, host_mmff_interactions,
                   mmff_interactions_total * sizeof(mmff94_atom_interaction_t), cudaMemcpyHostToDevice);
    if (mmff_offsets_total > 0)
        cudaMemcpy(dev_mmff_offsets, host_mmff_offsets, mmff_offsets_total * sizeof(mmff94_atom_offsets_t),
                   cudaMemcpyHostToDevice);
    cudaMemcpy(dev_metadata, host_metadata, num_molecules * sizeof(etkdg_molecule_metadata_t),
               cudaMemcpyHostToDevice);

    // compute shared memory and grid size
    std::uint32_t const shared_bytes = compute_pipeline_shared_memory_bytes(max_atoms_per_molecule, 0);
    cudaFuncSetAttribute(etkdg_pipeline_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);

    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, etkdg_pipeline_warp_kernel, 32, shared_bytes);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    std::uint32_t const grid_size = std::min(num_molecules, static_cast<std::uint32_t>(max_blocks_per_sm * num_sms));

    std::printf("  GPU: %d SMs, %d blocks/SM, grid=%u, shared=%u bytes\n", num_sms, max_blocks_per_sm, grid_size,
                shared_bytes);

    // warm up
    etkdg_pipeline_warp_kernel<<<grid_size, 32, shared_bytes>>>(
        dev_bounds, dev_chirals, dev_torsions, dev_constraints, dev_mmff_nb, dev_mmff_bonds, dev_mmff_angles,
        dev_mmff_torsions, dev_mmff_inter, dev_mmff_offsets, dev_metadata, dev_coords_out, dev_energies_out,
        dev_status_out, num_molecules, num_conformers, max_stereo_attempts, max_atoms_per_molecule, seed);
    cudaDeviceSynchronize();

    // timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    etkdg_pipeline_warp_kernel<<<grid_size, 32, shared_bytes>>>(
        dev_bounds, dev_chirals, dev_torsions, dev_constraints, dev_mmff_nb, dev_mmff_bonds, dev_mmff_angles,
        dev_mmff_torsions, dev_mmff_inter, dev_mmff_offsets, dev_metadata, dev_coords_out, dev_energies_out,
        dev_status_out, num_molecules, num_conformers, max_stereo_attempts, max_atoms_per_molecule, seed + 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // download results
    cudaMemcpy(host_coords_out, dev_coords_out, total_coords * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_energies_out, dev_energies_out, total_conf * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_status_out, dev_status_out, total_conf * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);

    std::uint32_t num_succeeded = 0;
    for (std::uint32_t i = 0; i < total_conf; ++i)
        if (host_status_out[i] == 0)
            num_succeeded++;

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_bounds);
    cudaFree(dev_chirals);
    cudaFree(dev_torsions);
    cudaFree(dev_constraints);
    cudaFree(dev_mmff_nb);
    cudaFree(dev_mmff_bonds);
    cudaFree(dev_mmff_angles);
    cudaFree(dev_mmff_torsions);
    cudaFree(dev_mmff_inter);
    cudaFree(dev_mmff_offsets);
    cudaFree(dev_metadata);
    cudaFree(dev_coords_out);
    cudaFree(dev_energies_out);
    cudaFree(dev_status_out);

    return {kernel_ms, num_succeeded};
}

} // extern "C"
