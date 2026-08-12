/**
 *  @file src/smallfield/cuda/triangle_smooth.cuh
 *  @brief Floyd-Warshall triangle inequality smoothing on a float bounds matrix, and its verdict.
 *
 *  Tightens the N x N bounds matrix by enforcing:
 *
 *  @verbatim
 *  upper: u_ij <- min(u_ij, u_ik + u_kj)   for all k
 *  lower: l_ij <- max(l_ij, l_ik - u_kj, l_kj - u_ik)  for all k
 *  @endverbatim
 *
 *  Both lower-bound candidates are taken at their maximum, so a pair pushed apart through one leg of the
 *  triangle keeps that separation when the other leg would have asked for less.
 *
 *  One pivot round is `triangle_smooth_pivot_`, which every cooperating thread calls over its own stride of
 *  the matrix; `triangle_smooth_warp_` drives a whole matrix inside one warp for the smoothing-only test,
 *  and `embed_4d_power_iter_kernel` drives the same round across a block for production. The verdict is
 *  `smoothing_t`, so a matrix no set of coordinates can satisfy is named where it is discovered rather than
 *  embedded and left to surface as a stereo failure three stages later.
 *
 *  Convention: bounds[row * N + column] with row < column = upper, row > column = lower, diagonal = 0.
 */

#pragma once

#include <cstdint> // `std::uint8_t`

#include <string_view> // `std::string_view`

namespace smallfield::cuda {

/*  The verdict and its slack sit outside the `__CUDACC__` guard below, as `warp_size_k` does in
 *  `warp_primitives.cuh`, because a host translation unit that reads what the device decided has to be able
 *  to name the same values the kernel wrote. */

/**
 *  @brief What one smoothing pass did to a bounds matrix.
 *
 *  A `bool` cannot carry this: it folds "there was nothing left to tighten" together with "the matrix is
 *  impossible", and those are the two answers a caller acts on differently. The three are ordered by
 *  severity, so combining the verdicts of two threads is taking the larger of the two.
 */
enum class smoothing_t : std::uint8_t {
    /** Every triangle inequality already held, and no bound moved. */
    consistent_k = 0,
    /** At least one bound moved, and every pair still admits a distance. */
    tightened_k = 1,
    /** Some pair's lower bound now exceeds its upper, so no set of coordinates satisfies the matrix. */
    infeasible_k = 2,
};

/** The literal a verdict prints as. */
constexpr std::string_view smoothing_name(smoothing_t verdict) noexcept {
    switch (verdict) {
    case smoothing_t::consistent_k: return "consistent";
    case smoothing_t::tightened_k: return "tightened";
    default: return "infeasible";
    }
}

/**
 *  @brief Slack a crossed pair is allowed before it counts as impossible, in angstrom.
 *
 *  A pair whose bounds meet exactly — a rigid triangle, a fixed bond — can cross by the rounding a float
 *  sweep leaves behind, and calling that impossible would refuse molecules that embed perfectly well.
 *  `test_triangle_smooth` prints the largest crossing its corpus produced, so the margin between that and
 *  this number is read rather than assumed.
 */
constexpr float bounds_crossing_tolerance_k = 1e-4f;

} // namespace smallfield::cuda

#if defined(__CUDACC__)

#include "smallfield/cuda/warp_primitives.cuh"
#include "smallfield/types/views.hpp"

namespace smallfield::cuda {
using smallfield::types::lower_distance_bound_at;
using smallfield::types::packed_matrix;
using smallfield::types::upper_distance_bound_at;

/** The more severe of two verdicts, which is the larger of the two underlying values. */
__device__ __forceinline__ smoothing_t worse_smoothing_(smoothing_t first, smoothing_t second) {
    return static_cast<std::uint8_t>(first) > static_cast<std::uint8_t>(second) ? first : second;
}

/**
 *  @brief One Floyd-Warshall pivot round over the cells this thread owns.
 *
 *  @b No barrier is needed inside a round, and that is a property of which cells it touches rather than an
 *  accepted race. Every cell it reads is incident to @p pivot — `u_row_pivot`, `u_pivot_column`,
 *  `l_row_pivot`, `l_pivot_column` — and every cell it writes is skipped by the `row == pivot ||
 *  column == pivot` guard, so the round's read set and its write set are disjoint and no thread can observe
 *  a cell another thread is tightening. The only cell a thread both reads and writes is its own, which no
 *  other thread visits because the strided walk hands each linear index to exactly one thread. The barrier
 *  that does matter is the one @b after the round, which the caller supplies at its own granularity: the
 *  next pivot reads cells this round wrote.
 *
 *  @param[inout] bounds Square bounds matrix over one molecule's atoms.
 *  @param[in] pivot Atom this round routes both bounds through.
 *  @param[in] first_cell Linear index of this thread's first cell.
 *  @param[in] cell_stride Threads cooperating on the matrix.
 *  @return `tightened_k` when this thread moved a bound, `consistent_k` when it moved none.
 */
__device__ __forceinline__ smoothing_t triangle_smooth_pivot_(packed_matrix<float> bounds, std::uint32_t pivot,
                                                              std::uint32_t first_cell, std::uint32_t cell_stride) {
    std::uint32_t const num_atoms = bounds.extent(0);
    std::uint32_t const total_cells = num_atoms * num_atoms;
    smoothing_t verdict = smoothing_t::consistent_k;

    for (std::uint32_t cell = first_cell; cell < total_cells; cell += cell_stride) {
        std::uint32_t const row = cell / num_atoms;
        std::uint32_t const column = cell % num_atoms;
        if (row == column || row == pivot || column == pivot) continue;

        float const upper_row_pivot = upper_distance_bound_at(bounds, row, pivot);
        float const upper_pivot_column = upper_distance_bound_at(bounds, pivot, column);
        float const current = bounds(row, column);

        // Above the diagonal the cell is the pair's upper bound, below it the lower one, so the branch picks
        // which of the two inequalities applies rather than which cell to touch. The lower bounds stay
        // inside the branch that consults them, where they cost no register on the other path.
        if (row < column) {
            float const candidate = upper_row_pivot + upper_pivot_column;
            if (candidate < current) {
                bounds(row, column) = candidate;
                verdict = smoothing_t::tightened_k;
            }
        }
        else {
            float const lower_row_pivot = lower_distance_bound_at(bounds, row, pivot);
            float const lower_pivot_column = lower_distance_bound_at(bounds, pivot, column);
            float const candidate = fmaxf(lower_row_pivot - upper_pivot_column, lower_pivot_column - upper_row_pivot);
            if (candidate > current) {
                bounds(row, column) = candidate;
                verdict = smoothing_t::tightened_k;
            }
        }
    }
    return verdict;
}

/**
 *  @brief Whether every pair this thread owns still admits a distance.
 *
 *  Scanning once after the sweep is exactly as strong as testing inside every pivot round and costs one pass
 *  rather than N: smoothing only raises a lower bound and only lowers an upper one, so a pair that crosses
 *  at any point in the sweep is still crossed when the sweep ends.
 *
 *  @param[in] bounds Square bounds matrix over one molecule's atoms.
 *  @param[in] first_cell Linear index of this thread's first cell.
 *  @param[in] cell_stride Threads cooperating on the matrix.
 *  @return `infeasible_k` when some pair crosses by more than `bounds_crossing_tolerance_k`,
 *          `consistent_k` otherwise; never `tightened_k`, since a scan moves nothing.
 */
__device__ __forceinline__ smoothing_t bounds_consistency_(packed_matrix<float const> bounds, std::uint32_t first_cell,
                                                           std::uint32_t cell_stride) {
    std::uint32_t const num_atoms = bounds.extent(0);
    std::uint32_t const total_cells = num_atoms * num_atoms;
    smoothing_t verdict = smoothing_t::consistent_k;

    for (std::uint32_t cell = first_cell; cell < total_cells; cell += cell_stride) {
        std::uint32_t const row = cell / num_atoms;
        std::uint32_t const column = cell % num_atoms;
        if (row >= column) continue; // one visit per unordered pair, and the diagonal is neither bound
        float const upper = upper_distance_bound_at(bounds, row, column);
        float const lower = lower_distance_bound_at(bounds, row, column);
        if (lower - upper > bounds_crossing_tolerance_k) verdict = smoothing_t::infeasible_k;
    }
    return verdict;
}

/**
 *  @brief Smooth a whole bounds matrix inside one warp and report what the sweep did.
 *
 *  @param[inout] bounds Square bounds matrix over one molecule's atoms, smoothed in place.
 *  @param[in] lane Warp lane index, 0 to `warp_size_k` - 1.
 *  @return The verdict over the whole matrix, identical in every lane.
 */
__device__ __forceinline__ smoothing_t triangle_smooth_warp_(packed_matrix<float> bounds, std::uint32_t lane) {
    smoothing_t verdict = smoothing_t::consistent_k;
    for (std::uint32_t pivot = 0; pivot < bounds.extent(0); ++pivot) {
        verdict = worse_smoothing_(verdict, triangle_smooth_pivot_(bounds, pivot, lane, warp_size_k));
        __syncwarp(full_mask_k);
    }
    verdict = worse_smoothing_(verdict, bounds_consistency_(bounds, lane, warp_size_k));
    // `sm_80` is the floor of `SMALLFIELD_CUDA_ARCH`, so the warp-wide maximum is one `REDUX` rather than
    // the five-step shuffle ladder the float reductions in `warp_primitives.cuh` are built from.
    return static_cast<smoothing_t>(__reduce_max_sync(full_mask_k, static_cast<unsigned int>(verdict)));
}

} // namespace smallfield::cuda

#endif // __CUDACC__
