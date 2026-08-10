/**
 *  @file triangle_smooth.cuh
 *  @brief Floyd-Warshall triangle inequality smoothing on a float bounds matrix.
 *
 *  Tightens the N x N bounds matrix by enforcing:
 *
 *  @verbatim
 *  upper: u_ij <- min(u_ij, u_ik + u_kj)   for all k
 *  lower: l_ij <- max(l_ij, l_ik - u_kj, l_kj - u_ik)  for all k
 *  @endverbatim
 *
 *  Runs entirely within one warp of 32 threads. Production smoothing runs inside
 *  `metric_embedding_prepass_kernel`, and this header serves the standalone smoothing-only test,
 *  `test_triangle_smooth`.
 *
 *  Convention: bounds[i*N+j] with i < j = upper, i > j = lower, diagonal = 0.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include "smallfield/cuda/warp_primitives.cuh"

namespace smallfield::cuda {

/**
 *  @brief Float-precision Floyd-Warshall triangle smoothing.
 *
 *  @param[inout] bounds Float N x N bounds matrix (modified in place).
 *  @param[in] num_atoms Number of atoms (matrix dimension).
 *  @param[in] lane Warp lane index (0-31).
 */
__device__ __forceinline__ void triangle_smooth_warp_(float *bounds, std::uint32_t num_atoms, std::uint32_t lane) {
    std::uint32_t const total_pairs = num_atoms * num_atoms;

    for (std::uint32_t k = 0; k < num_atoms; ++k) {
        for (std::uint32_t linear_index = lane; linear_index < total_pairs; linear_index += warp_size_k) {
            std::uint32_t const i = linear_index / num_atoms;
            std::uint32_t const j = linear_index % num_atoms;
            if (i == j || i == k || j == k) continue;

            auto read_upper = [&](std::uint32_t a, std::uint32_t b) -> float {
                return bounds[min(a, b) * num_atoms + max(a, b)];
            };
            auto read_lower = [&](std::uint32_t a, std::uint32_t b) -> float {
                return bounds[max(a, b) * num_atoms + min(a, b)];
            };

            float const upper_ik = read_upper(i, k);
            float const upper_kj = read_upper(k, j);

            if (i < j) {
                float const current = bounds[i * num_atoms + j];
                float const candidate = upper_ik + upper_kj;
                if (candidate < current) bounds[i * num_atoms + j] = candidate;
            }
            else {
                float const current = bounds[i * num_atoms + j];
                float const lower_ik = read_lower(i, k);
                float const lower_kj = read_lower(k, j);
                float const upper_ik_val = read_upper(i, k);

                float candidate = current;
                float const c1 = lower_ik - upper_kj;
                float const c2 = lower_kj - upper_ik_val;
                if (c1 > candidate) candidate = c1;
                if (c2 > candidate) candidate = c2;
                if (candidate > current) bounds[i * num_atoms + j] = candidate;
            }
        }
        __syncwarp(full_mask_k);
    }
}

/**
 *  @brief Standalone kernel for testing float-precision smoothing.
 *
 *  @param[inout] bounds Float N x N bounds matrix (modified in place).
 *  @param[in] num_atoms Number of atoms (matrix dimension).
 */
__global__ void triangle_smooth_kernel(float *bounds, std::uint32_t num_atoms) {
    triangle_smooth_warp_(bounds, num_atoms, threadIdx.x);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
