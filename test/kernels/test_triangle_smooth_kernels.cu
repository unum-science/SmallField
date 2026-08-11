/**
 *  @file test/kernels/test_triangle_smooth_kernels.cu
 *  @brief CUDA kernel wrappers for triangle smoothing test.
 *
 *  Compiled by nvcc. The host test code (with RDKit includes) is in test_triangle_smooth.cpp, compiled by g++.
 */

#include <cstdint> // `std::uint32_t`

#include <cuda_runtime.h>

#include "smallfield/cuda/triangle_smooth.cuh"

#include "test_triangle_smooth_kernels.hpp"

namespace sfc = smallfield::cuda;

/** Kernel that copies bounds to shared memory, smooths, copies back. */
__global__ void triangle_smooth_shared_kernel(float *global_bounds, std::uint32_t num_atoms) {
    extern __shared__ float shared_bounds[];
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const total = num_atoms * num_atoms;

    // copy global → shared
    for (std::uint32_t entry = lane; entry < total; entry += 32) { shared_bounds[entry] = global_bounds[entry]; }
    __syncwarp(0xFFFFFFFF);

    // smooth in shared memory
    sfc::triangle_smooth_warp_(shared_bounds, num_atoms, lane);

    // copy shared → global
    for (std::uint32_t entry = lane; entry < total; entry += 32) { global_bounds[entry] = shared_bounds[entry]; }
}

namespace smallfield::tests {

void gpu_triangle_smooth(float *host_bounds, std::uint32_t num_atoms) {
    float *device_bounds = nullptr;
    std::size_t const bounds_bytes = num_atoms * num_atoms * sizeof(float);

    cudaMalloc(&device_bounds, bounds_bytes);
    cudaMemcpy(device_bounds, host_bounds, bounds_bytes, cudaMemcpyHostToDevice);

    std::size_t const shared_bytes = bounds_bytes;
    triangle_smooth_shared_kernel<<<1, 32, shared_bytes>>>(device_bounds, num_atoms);
    cudaDeviceSynchronize();

    cudaMemcpy(host_bounds, device_bounds, bounds_bytes, cudaMemcpyDeviceToHost);
    cudaFree(device_bounds);
}

} // namespace smallfield::tests
