/**
 *  @file test/cuda/test_dg_energy_kernels.hpp
 *  @brief Device harness for distance-geometry energy, chirality validation, and FIRE minimization.
 */

#pragma once

#include <cstdint> // `std::int32_t`, `std::uint32_t`

#include <string_view> // `std::string_view`

#include "smallfield/cuda/fire_warp.cuh" // `smallfield::cuda::fire_status_t`, outside its CUDA guard
#include "smallfield/types/etkdg_types.hpp"

namespace smallfield::tests {

/**
 *  @brief Which step of a device round trip refused, so a caller can say why it compared nothing.
 *
 *  A launch the driver declines never writes an output buffer, and none of the entry points below zeroes
 *  one first, so a caller that reads the buffer anyway scores whatever the allocation found and reports it
 *  as an energy, a gradient, or a chiral-center count.
 */
enum class dg_probe_t {
    completed_k,
    staging_failed_k,
    transfer_failed_k,
    launch_failed_k,
};

/** The sentence a `dg_probe_t` reads as, for the `NOTE` line above a skipped comparison. */
std::string_view dg_probe_reason(dg_probe_t outcome) noexcept;

/**
 *  @brief Everything a device call needs about the molecule it is scoring, except the coordinates.
 *
 *  The three penalty weights and the two counts are all scalars of the same type, so the entry
 *  points below take them as one named aggregate rather than as eleven positional arguments a call
 *  site can silently transpose.
 */
struct dg_problem_t {
    /** Flat `num_atoms` squared bounds matrix, upper bounds above the diagonal. */
    float const *bounds = nullptr;
    /** Chiral center descriptors, or `nullptr` when the molecule has none. */
    types::chiral_center_descriptor_t const *chiral_centers = nullptr;
    std::uint32_t num_atoms = 0;
    std::uint32_t num_chiral_centers = 0;
    /** Coordinate dimensionality — three or four. */
    std::uint32_t spatial_dimension = 4;
    float upper_weight = 1.0f;
    float lower_weight = 1.0f;
    float chiral_weight = 0.0f;
    float fourth_dimension_weight = 0.0f;
};

/**
 *  @brief Evaluate distance-geometry energy and its gradient on the device for one molecule.
 *
 *  @param[in] problem Bounds, chiral centers, extents, and penalty weights.
 *  @param[in] host_coords Coordinates as `(axis, atom)`, `spatial_dimension * num_atoms` entries, which is
 *      the layout the kernel indexes; an interleaved block passed here scores a different molecule.
 *  @param[out] host_gradient Receives the gradient, same layout as @p host_coords; untouched on a refusal.
 *  @param[out] out_energy Receives the total energy; untouched on a refusal.
 *  @return Whether the round trip completed, and which step refused it.
 */
[[nodiscard]] dg_probe_t gpu_dg_energy(dg_problem_t const &problem, float const *host_coords, float *host_gradient,
                                       float *out_energy);

/**
 *  @brief Count the chiral centers whose realized geometry contradicts their descriptor.
 *
 *  The count is an output rather than the return, because zero is the answer "the stereochemistry is
 *  intact" and a refused launch must not be able to spell it.
 *
 *  @param[in] problem Bounds, chiral centers, extents, and penalty weights; only the centers are read.
 *  @param[in] host_coords Coordinates as `(axis, atom)`.
 *  @param[out] out_invalid Receives how many centers failed; untouched on a refusal.
 *  @return Whether the round trip completed, and which step refused it.
 */
[[nodiscard]] dg_probe_t gpu_count_invalid_chiral_centers(dg_problem_t const &problem, float const *host_coords,
                                                          std::uint32_t &out_invalid);

/** What FIRE reported when it stopped; every field describes the positions it handed back. */
struct dg_minimize_report_t {
    float energy = 0.0f;
    float rms_gradient = 0.0f;
    std::int32_t iterations = 0;
    cuda::fire_status_t status = cuda::fire_status_t::converged_k;
};

/**
 *  @brief Run FIRE minimization of the distance-geometry energy on the device for one molecule.
 *
 *  @param[in] problem Bounds, chiral centers, extents, and penalty weights.
 *  @param[inout] host_coords Coordinates as `(axis, atom)`, overwritten with the minimized positions.
 *  @param[in] max_iterations Iteration cap for the FIRE loop.
 *  @param[in] gradient_threshold Convergence threshold on the gradient norm.
 *  @param[out] out_report Receives the final energy, RMS gradient, iteration count, and status; untouched
 *      on a refusal, as are @p host_coords.
 *  @return Whether the round trip completed, and which step refused it.
 */
[[nodiscard]] dg_probe_t gpu_dg_minimize(dg_problem_t const &problem, float *host_coords, std::int32_t max_iterations,
                                         float gradient_threshold, dg_minimize_report_t *out_report);

} // namespace smallfield::tests
