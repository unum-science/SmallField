/**
 *  @file test_chirality_kernels.cu
 *  @brief CUDA kernel wrappers for chirality validation test. Compiled by nvcc.
 */

#include <cstdint>

#include <cuda_runtime.h>

#include "dg_forcefield.cuh"

using namespace smallfield;

__global__ void chirality_check_kernel(float const *positions, std::uint32_t num_atoms,
                                       chiral_center_descriptor_t const *chirals, std::uint32_t num_chirals,
                                       bool *out_result) {

    std::uint32_t const lane = threadIdx.x;
    bool const result = validate_chirality_(positions, num_atoms, chirals, num_chirals, lane);
    if (lane == 0) { *out_result = result; }
}

extern "C" {

bool gpu_validate_chirality(float const *host_coords, chiral_center_descriptor_t const *host_chirals,
                            std::uint32_t num_atoms, std::uint32_t num_chirals) {

    std::uint32_t const coord_elements = 3 * num_atoms; // 3D only for chirality
    float *dev_coords;
    chiral_center_descriptor_t *dev_chirals;
    bool *dev_result;

    cudaMalloc(&dev_coords, coord_elements * sizeof(float));
    cudaMalloc(&dev_chirals, num_chirals * sizeof(chiral_center_descriptor_t));
    cudaMalloc(&dev_result, sizeof(bool));

    cudaMemcpy(dev_coords, host_coords, coord_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_chirals, host_chirals, num_chirals * sizeof(chiral_center_descriptor_t), cudaMemcpyHostToDevice);

    chirality_check_kernel<<<1, 32>>>(dev_coords, num_atoms, dev_chirals, num_chirals, dev_result);
    cudaDeviceSynchronize();

    bool result = false;
    cudaMemcpy(&result, dev_result, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(dev_coords);
    cudaFree(dev_chirals);
    cudaFree(dev_result);

    return result;
}

} // extern "C"
