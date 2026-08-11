/**
 *  @file test/kernels/test_embedding_determinism_kernels.cu
 *  @brief CUDA driver for the metric-embedding reproducibility test.
 *
 *  Compiled by nvcc. The host test logic is in test_embedding_determinism.cpp, compiled by g++.
 */

#include <cstdint> // `std::uint32_t`

#include <vector> // `std::vector`

#include <cuda_runtime.h>

#include "smallfield/cuda/metric_embedding_prepass.cuh"

#include "test_embedding_determinism_kernels.hpp"

namespace sfc = smallfield::cuda;
using smallfield::types::etkdg_molecule_metadata_t;

namespace smallfield::tests {

/**
 *  @brief Runs `embed_4d_power_iter_kernel` over @p num_molecules copies of one bounds matrix.
 *
 *  Every block sees byte-identical input, so every block owes byte-identical output, whatever
 *  streaming multiprocessor it lands on and whichever of its warps arrives at a reduction first.
 *  The kernel smooths its bounds in place, so a fresh copy is uploaded on each call.
 *
 *  @param[in] host_bounds One @p num_atoms × @p num_atoms bounds matrix, row-major.
 *  @param[in] num_atoms Atom count of that matrix.
 *  @param[in] num_molecules Number of identical copies to embed, one block each.
 *  @param[out] host_coordinates Receives @p num_molecules × @p num_atoms × 4 floats, dimension major.
 *  @return Zero on success, or the CUDA error code that stopped the run.
 */
int gpu_embed_replicated(float const *host_bounds, std::uint32_t num_atoms, std::uint32_t num_molecules,
                         float *host_coordinates) {
    std::size_t const entries_per_molecule = static_cast<std::size_t>(num_atoms) * num_atoms;
    std::size_t const coordinates_per_molecule = static_cast<std::size_t>(num_atoms) * 4;

    std::vector<float> bounds(entries_per_molecule * num_molecules);
    std::vector<etkdg_molecule_metadata_t> metadata(num_molecules);
    std::vector<std::uint32_t> indices(num_molecules);
    for (std::uint32_t molecule = 0; molecule < num_molecules; ++molecule) {
        for (std::size_t entry = 0; entry < entries_per_molecule; ++entry)
            bounds[molecule * entries_per_molecule + entry] = host_bounds[entry];
        metadata[molecule] = etkdg_molecule_metadata_t {};
        metadata[molecule].num_atoms = num_atoms;
        metadata[molecule].bounds_offset = static_cast<std::uint32_t>(molecule * entries_per_molecule);
        indices[molecule] = molecule;
    }

    float *device_bounds = nullptr;
    etkdg_molecule_metadata_t *device_metadata = nullptr;
    std::uint32_t *device_indices = nullptr;
    float *device_coordinates = nullptr;
    std::size_t const coordinate_bytes = coordinates_per_molecule * num_molecules * sizeof(float);
    cudaMalloc(&device_bounds, bounds.size() * sizeof(float));
    cudaMalloc(&device_metadata, metadata.size() * sizeof(etkdg_molecule_metadata_t));
    cudaMalloc(&device_indices, indices.size() * sizeof(std::uint32_t));
    cudaMalloc(&device_coordinates, coordinate_bytes);
    cudaMemcpy(device_bounds, bounds.data(), bounds.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_metadata, metadata.data(), metadata.size() * sizeof(etkdg_molecule_metadata_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_indices, indices.data(), indices.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice);

    std::size_t const shared_bytes =
        static_cast<std::size_t>(num_atoms * num_atoms + num_atoms + 1 + 4 * num_atoms + num_atoms + 4) * sizeof(float);
    cudaFuncSetAttribute(sfc::embed_4d_power_iter_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared_bytes));

    cudaMemset(device_coordinates, 0, coordinate_bytes);
    sfc::embed_4d_power_iter_kernel<<<num_molecules, sfc::embed_block_size_k, shared_bytes>>>(
        device_bounds, device_metadata, device_indices, device_coordinates, num_atoms);
    cudaError_t status = cudaDeviceSynchronize();
    if (status == cudaSuccess) status = cudaGetLastError();
    if (status == cudaSuccess)
        cudaMemcpy(host_coordinates, device_coordinates, coordinate_bytes, cudaMemcpyDeviceToHost);

    cudaFree(device_bounds);
    cudaFree(device_metadata);
    cudaFree(device_indices);
    cudaFree(device_coordinates);
    return static_cast<int>(status);
}

} // namespace smallfield::tests
