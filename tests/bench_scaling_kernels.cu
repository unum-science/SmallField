/**
 * @file bench_pipeline_kernels.cu
 * @brief CUDA kernel wrapper for ETKDG pipeline benchmark with CUDA event timing.
 */

#include "etkdg_pipeline.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

using namespace smallfield;

extern "C" {

float gpu_etkdg_pipeline_timed(float const* host_bounds, std::uint32_t bounds_total_floats,
                               chiral_center_descriptor_t const* host_chirals, std::uint32_t chirals_total,
                               etk_torsion_descriptor_t const* host_torsions, std::uint32_t torsions_total,
                               distance_constraint_t const* host_constraints, std::uint32_t constraints_total,
                               etkdg_molecule_metadata_t const* host_metadata, std::uint32_t num_molecules,
                               float* host_coords_out, float* host_energies_out, std::uint8_t* host_status_out,
                               std::uint32_t num_conformers, std::uint32_t max_stereo_attempts,
                               std::uint32_t max_atoms_per_molecule, std::uint64_t seed) {

    float* dev_bounds;
    chiral_center_descriptor_t* dev_chirals;
    etk_torsion_descriptor_t* dev_torsions;
    distance_constraint_t* dev_constraints;
    etkdg_molecule_metadata_t* dev_metadata;
    float* dev_coords_out;
    float* dev_energies_out;
    std::uint8_t* dev_status_out;

    cudaMalloc(&dev_bounds, bounds_total_floats * sizeof(float));
    cudaMalloc(&dev_chirals, std::max(chirals_total, 1u) * sizeof(chiral_center_descriptor_t));
    cudaMalloc(&dev_torsions, std::max(torsions_total, 1u) * sizeof(etk_torsion_descriptor_t));
    cudaMalloc(&dev_constraints, std::max(constraints_total, 1u) * sizeof(distance_constraint_t));
    cudaMalloc(&dev_metadata, num_molecules * sizeof(etkdg_molecule_metadata_t));

    std::uint32_t const total_conformers = num_molecules * num_conformers;
    std::uint32_t const total_coords = total_conformers * max_atoms_per_molecule * 3;
    cudaMalloc(&dev_coords_out, total_coords * sizeof(float));
    cudaMalloc(&dev_energies_out, total_conformers * sizeof(float));
    cudaMalloc(&dev_status_out, total_conformers * sizeof(std::uint8_t));

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
    cudaMemcpy(dev_metadata, host_metadata, num_molecules * sizeof(etkdg_molecule_metadata_t), cudaMemcpyHostToDevice);

    std::uint32_t const shared_bytes = compute_pipeline_shared_memory_bytes(max_atoms_per_molecule, 0);
    cudaFuncSetAttribute(etkdg_pipeline_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);

    // compute optimal grid size from occupancy
    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, etkdg_pipeline_warp_kernel, 32, shared_bytes);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    std::uint32_t const grid_size = std::min(num_molecules, static_cast<std::uint32_t>(max_blocks_per_sm * num_sms));

    // warm up
    etkdg_pipeline_warp_kernel<<<grid_size, 32, shared_bytes>>>(
        dev_bounds, dev_chirals, dev_torsions, dev_constraints, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        dev_metadata, dev_coords_out, dev_energies_out, dev_status_out, num_molecules, num_conformers,
        max_stereo_attempts, max_atoms_per_molecule, seed);
    cudaDeviceSynchronize();

    // timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    etkdg_pipeline_warp_kernel<<<grid_size, 32, shared_bytes>>>(
        dev_bounds, dev_chirals, dev_torsions, dev_constraints, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        dev_metadata, dev_coords_out, dev_energies_out, dev_status_out, num_molecules, num_conformers,
        max_stereo_attempts, max_atoms_per_molecule, seed + 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_time_ms = 0.0f;
    cudaEventElapsedTime(&kernel_time_ms, start, stop);

    // download results
    cudaMemcpy(host_coords_out, dev_coords_out, total_coords * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_energies_out, dev_energies_out, total_conformers * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_status_out, dev_status_out, total_conformers * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_bounds);
    cudaFree(dev_chirals);
    cudaFree(dev_torsions);
    cudaFree(dev_constraints);
    cudaFree(dev_metadata);
    cudaFree(dev_coords_out);
    cudaFree(dev_energies_out);
    cudaFree(dev_status_out);

    return kernel_time_ms;
}

} // extern "C"
