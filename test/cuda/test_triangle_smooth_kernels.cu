/**
 *  @file test/cuda/test_triangle_smooth_kernels.cu
 *  @brief CUDA kernel wrappers for the triangle-smoothing test.
 *
 *  Compiled by nvcc. The host test code, with its RDKit includes, is in test_triangle_smooth.cpp, compiled
 *  by g++.
 */

#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::fprintf`

#include <cuda_runtime.h>

#include "smallfield/cuda/triangle_smooth.cuh"
#include "smallfield/types/views.hpp"

#include "test/cuda/test_triangle_smooth_kernels.hpp"

namespace smallfield::tests {

using smallfield::cuda::smoothing_t;
using smallfield::cuda::triangle_smooth_warp_;
using smallfield::cuda::warp_size_k;
using smallfield::types::packed_matrix;

/** Stage the matrix in shared memory, smooth it in one warp, and write the matrix and its verdict back. */
__global__ void triangle_smooth_kernel(float *__restrict__ global_bounds, std::uint32_t num_atoms,
                                       smoothing_t *__restrict__ verdict) {
    extern __shared__ float shared_bounds[];
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const total_cells = num_atoms * num_atoms;

    for (std::uint32_t cell = lane; cell < total_cells; cell += warp_size_k) shared_bounds[cell] = global_bounds[cell];
    __syncwarp(smallfield::cuda::full_mask_k);

    smoothing_t const smoothed = triangle_smooth_warp_(packed_matrix<float> {shared_bounds, num_atoms, num_atoms},
                                                       lane);

    for (std::uint32_t cell = lane; cell < total_cells; cell += warp_size_k) global_bounds[cell] = shared_bounds[cell];
    if (lane == 0) *verdict = smoothed;
}

/** Bytes the widest matrix this harness accepts occupies in shared memory. */
constexpr std::size_t max_shared_bytes_k = static_cast<std::size_t>(max_smoothed_atoms_k) * max_smoothed_atoms_k *
                                           sizeof(float);

smoothing_t gpu_triangle_smooth(float *host_bounds, std::uint32_t num_atoms) {
    // Past 48 KB a kernel has to ask for its dynamic shared memory, and a 128-atom matrix is 64 KB. The
    // attribute is per-kernel driver state, so it is raised once for the process.
    [[maybe_unused]] static bool const shared_limit_raised = [] {
        cudaFuncSetAttribute(triangle_smooth_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(max_shared_bytes_k));
        return true;
    }();

    std::size_t const bounds_bytes = static_cast<std::size_t>(num_atoms) * num_atoms * sizeof(float);
    float *device_bounds = nullptr;
    smoothing_t *device_verdict = nullptr;

    // The verdict starts at the mildest value, so a launch that never ran leaves the caller a matrix its own
    // comparisons reject rather than a verdict that looks decided.
    smoothing_t verdict = smoothing_t::consistent_k;
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaMallocAsync(&device_bounds, bounds_bytes, stream);
    cudaMallocAsync(&device_verdict, sizeof(smoothing_t), stream);
    cudaMemcpyAsync(device_bounds, host_bounds, bounds_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_verdict, &verdict, sizeof(smoothing_t), cudaMemcpyHostToDevice, stream);

    triangle_smooth_kernel<<<1, warp_size_k, bounds_bytes, stream>>>(device_bounds, num_atoms, device_verdict);

    cudaMemcpyAsync(host_bounds, device_bounds, bounds_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&verdict, device_verdict, sizeof(smoothing_t), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(device_bounds, stream);
    cudaFreeAsync(device_verdict, stream);

    cudaError_t const smoothed = cudaStreamSynchronize(stream);
    if (smoothed != cudaSuccess) std::fprintf(stderr, "triangle_smooth_kernel: %s\n", cudaGetErrorString(smoothed));
    cudaStreamDestroy(stream);
    return verdict;
}

} // namespace smallfield::tests
