/**
 *  @file src/smallfield/cuda/triangle_smooth.cuh
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
 *  Convention: bounds[row * N + column] with row < column = upper, row > column = lower, diagonal = 0.
 */

#pragma once

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

    for (std::uint32_t via_atom = 0; via_atom < num_atoms; ++via_atom) {
        for (std::uint32_t linear_index = lane; linear_index < total_pairs; linear_index += warp_size_k) {
            std::uint32_t const row = linear_index / num_atoms;
            std::uint32_t const column = linear_index % num_atoms;
            if (row == column || row == via_atom || column == via_atom) continue;

            auto read_upper = [&](std::uint32_t first_atom, std::uint32_t second_atom) -> float {
                return bounds[min(first_atom, second_atom) * num_atoms + max(first_atom, second_atom)];
            };
            auto read_lower = [&](std::uint32_t first_atom, std::uint32_t second_atom) -> float {
                return bounds[max(first_atom, second_atom) * num_atoms + min(first_atom, second_atom)];
            };

            float const upper_row_via = read_upper(row, via_atom);
            float const upper_via_column = read_upper(via_atom, column);

            if (row < column) {
                float const current = bounds[row * num_atoms + column];
                float const candidate = upper_row_via + upper_via_column;
                if (candidate < current) bounds[row * num_atoms + column] = candidate;
            }
            else {
                float const current = bounds[row * num_atoms + column];
                float const lower_row_via = read_lower(row, via_atom);
                float const lower_via_column = read_lower(via_atom, column);
                float const upper_row_via_reread = read_upper(row, via_atom);

                float candidate = current;
                float const candidate_row_leg = lower_row_via - upper_via_column;
                float const candidate_column_leg = lower_via_column - upper_row_via_reread;
                if (candidate_row_leg > candidate) candidate = candidate_row_leg;
                if (candidate_column_leg > candidate) candidate = candidate_column_leg;
                if (candidate > current) bounds[row * num_atoms + column] = candidate;
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
