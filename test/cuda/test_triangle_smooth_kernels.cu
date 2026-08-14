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
#include "smallfield/types/storage.hpp"
#include "smallfield/types/views.hpp"

#include "test/cuda/test_triangle_smooth_kernels.hpp"

namespace smallfield::tests {

using smallfield::cuda::smoothing_t;
using smallfield::cuda::triangle_smooth_warp_;
using smallfield::cuda::warp_size_k;
using smallfield::types::buffer;
using smallfield::types::device_allocator_t;
using smallfield::types::packed_matrix;
using smallfield::types::reserve_outcome_t;

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

smoothing_t gpu_triangle_smooth(packed_matrix<float> bounds) {
    // Past 48 KB a kernel has to ask for its dynamic shared memory, and a 128-atom matrix is 64 KB. The
    // attribute is per-kernel driver state, so it is raised once for the process.
    [[maybe_unused]] static bool const shared_limit_raised = [] {
        cudaFuncSetAttribute(triangle_smooth_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(max_shared_bytes_k));
        return true;
    }();

    std::uint32_t const num_atoms = static_cast<std::uint32_t>(bounds.extent(0));
    std::size_t const cell_count = static_cast<std::size_t>(num_atoms) * num_atoms;

    // The verdict starts at the mildest value, so a launch that never ran leaves the caller a matrix its own
    // comparisons reject rather than a verdict that looks decided.
    smoothing_t verdict = smoothing_t::consistent_k;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return verdict;

    // The buffers live inside the stream's lifetime, so every stream-ordered free runs against a live stream.
    {
        device_allocator_t const allocator {.stream = stream};
        buffer<float, device_allocator_t> device_bounds {allocator};
        buffer<smoothing_t, device_allocator_t> device_verdict {allocator};

        bool staged = device_bounds.try_reserve(cell_count) != reserve_outcome_t::failed_k &&
                      device_verdict.try_reserve(1) != reserve_outcome_t::failed_k;
        staged = staged && types::upload_into(device_bounds.view(),
                                              custd::span<float const> {bounds.data_handle(), cell_count}, stream);
        staged = staged &&
                 types::upload_into(device_verdict.view(), custd::span<smoothing_t const> {&verdict, 1}, stream);

        if (!staged) { std::fprintf(stderr, "gpu_triangle_smooth: device staging refused for %u atoms\n", num_atoms); }
        else {
            triangle_smooth_kernel<<<1, warp_size_k, cell_count * sizeof(float), stream>>>(
                device_bounds.data(), num_atoms, device_verdict.data());

            // A refused enqueue reports here and nowhere else: the drain below sees an empty stream and
            // returns success, and the download then hands back the matrix exactly as it was uploaded
            // under the `consistent_k` the caller pre-set. This suite reads that as a smoothing that
            // found nothing to tighten and compares it against RDKit, so the enqueue status is what
            // separates a measured divergence from a shared-memory request the device would not take.
            cudaError_t const enqueued = cudaGetLastError();
            if (enqueued != cudaSuccess)
                std::fprintf(stderr, "triangle_smooth_kernel: refused at %u atoms: %s\n", num_atoms,
                             cudaGetErrorString(enqueued));
            else {
                [[maybe_unused]] bool const read_back =
                    types::download_from(custd::span<float> {bounds.data_handle(), cell_count},
                                         custd::span<float const> {device_bounds.data(), cell_count}, stream) &&
                    types::download_from(custd::span<smoothing_t> {&verdict, 1},
                                         custd::span<smoothing_t const> {device_verdict.data(), 1}, stream);

                cudaError_t const smoothed = cudaStreamSynchronize(stream);
                if (smoothed != cudaSuccess)
                    std::fprintf(stderr, "triangle_smooth_kernel: %s\n", cudaGetErrorString(smoothed));
            }
        }
    }

    cudaStreamDestroy(stream);
    return verdict;
}

} // namespace smallfield::tests
