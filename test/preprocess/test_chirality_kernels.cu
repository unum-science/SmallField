/**
 *  @file test/preprocess/test_chirality_kernels.cu
 *  @brief CUDA kernel wrappers for chirality validation test.
 */

#include <cstdint> // `std::uint32_t`

#include <cuda_runtime.h>

#include "smallfield/cuda/dg_kernel.cuh"

#include "test_chirality_kernels.hpp"

namespace sft = smallfield::types;
namespace sfc = smallfield::cuda;

__global__ void chirality_check_kernel(float const *positions, std::uint32_t num_atoms,
                                       sft::chiral_center_descriptor_t const *chirals, std::uint32_t num_chirals,
                                       bool *out_result) {

    std::uint32_t const lane = threadIdx.x;
    bool const result = sfc::validate_chirality_(positions, num_atoms, chirals, num_chirals, lane);
    if (lane == 0) { *out_result = result; }
}

namespace smallfield::tests {

bool gpu_validate_chirality(float const *host_coords, sft::chiral_center_descriptor_t const *host_chirals,
                            std::uint32_t num_atoms, std::uint32_t num_chirals) {

    std::uint32_t const coord_elements = 3 * num_atoms; // 3D only for chirality
    float *device_coords;
    sft::chiral_center_descriptor_t *device_chirals;
    bool *device_result;

    cudaMalloc(&device_coords, coord_elements * sizeof(float));
    cudaMalloc(&device_chirals, num_chirals * sizeof(sft::chiral_center_descriptor_t));
    cudaMalloc(&device_result, sizeof(bool));

    cudaMemcpy(device_coords, host_coords, coord_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_chirals, host_chirals, num_chirals * sizeof(sft::chiral_center_descriptor_t),
               cudaMemcpyHostToDevice);

    chirality_check_kernel<<<1, 32>>>(device_coords, num_atoms, device_chirals, num_chirals, device_result);
    cudaDeviceSynchronize();

    bool result = false;
    cudaMemcpy(&result, device_result, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(device_coords);
    cudaFree(device_chirals);
    cudaFree(device_result);

    return result;
}

} // namespace smallfield::tests
