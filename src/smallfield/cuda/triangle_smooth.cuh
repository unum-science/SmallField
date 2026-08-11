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
#include "smallfield/types/views.hpp"

namespace smallfield::cuda {
using smallfield::types::lower_distance_bound_at;
using smallfield::types::packed_matrix;
using smallfield::types::upper_distance_bound_at;

/**
 *  @brief Float-precision Floyd-Warshall triangle smoothing.
 *
 *  @param[inout] bounds Float N x N bounds matrix (modified in place).
 *  @param[in] num_atoms Number of atoms (matrix dimension).
 *  @param[in] lane Warp lane index (0-31).
 */
__device__ __forceinline__ void triangle_smooth_warp_(float *bounds, std::uint32_t num_atoms, std::uint32_t lane) {
    // A read view over the same storage the loop writes through `bounds`. The element type is `const`
    // because both accessors only read; it carries no aliasing promise, so the in-place update below stays
    // visible through it.
    packed_matrix<float const> const matrix {bounds, num_atoms, num_atoms};
    std::uint32_t const total_pairs = num_atoms * num_atoms;

    for (std::uint32_t via_atom = 0; via_atom < num_atoms; ++via_atom) {
        for (std::uint32_t linear_index = lane; linear_index < total_pairs; linear_index += warp_size_k) {
            std::uint32_t const row = linear_index / num_atoms;
            std::uint32_t const column = linear_index % num_atoms;
            if (row == column || row == via_atom || column == via_atom) continue;

            float const upper_row_via = upper_distance_bound_at(matrix, row, via_atom);
            float const upper_via_column = upper_distance_bound_at(matrix, via_atom, column);

            // Above the diagonal the cell is the pair's upper bound, below it the lower one, so the branch
            // picks which of the two inequalities applies rather than which cell to touch. Only the lower
            // bounds move inside the branch that consults them; both this split and the second read below
            // are what hold the kernel at its measured register count, so re-check `cuobjdump` before
            // rearranging either.
            if (row < column) {
                float const current = bounds[row * num_atoms + column];
                float const candidate = upper_row_via + upper_via_column;
                if (candidate < current) bounds[row * num_atoms + column] = candidate;
            }
            else {
                float const current = bounds[row * num_atoms + column];
                float const lower_row_via = lower_distance_bound_at(matrix, row, via_atom);
                float const lower_via_column = lower_distance_bound_at(matrix, via_atom, column);
                float const upper_row_via_reread = upper_distance_bound_at(matrix, row, via_atom);

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
