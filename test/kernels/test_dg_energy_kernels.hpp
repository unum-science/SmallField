/**
 *  @file test/kernels/test_dg_energy_kernels.hpp
 *  @brief Device harness for distance-geometry energy and FIRE minimization, for use by its unit tests.
 */

#pragma once

#include <cstdint> // `std::int32_t`, `std::uint32_t`

namespace smallfield::tests {

/**
 *  @brief Evaluate distance-geometry energy and its gradient on the device for one molecule.
 *
 *  @param[in] host_coords Flat interleaved coordinates, `spatial_dimension * num_atoms` entries.
 *  @param[out] host_gradient Receives the gradient, same layout as @p host_coords.
 *  @param[in] host_bounds Flat `num_atoms` squared bounds matrix.
 *  @param[in] num_atoms Atom count.
 *  @param[in] spatial_dimension Coordinate dimensionality.
 *  @param[in] upper_weight Penalty weight for upper-bound violations.
 *  @param[in] lower_weight Penalty weight for lower-bound violations.
 *  @param[in] fourth_dim_weight Penalty weight for the fourth-dimension term.
 *  @param[out] out_energy Receives the total energy.
 */
void gpu_dg_energy(float const *host_coords, float *host_gradient, float const *host_bounds, std::uint32_t num_atoms,
                   std::uint32_t spatial_dimension, float upper_weight, float lower_weight, float fourth_dim_weight,
                   float *out_energy);

/**
 *  @brief Run FIRE minimization of the distance-geometry energy on the device for one molecule.
 *
 *  @param[inout] host_coords Flat interleaved coordinates, overwritten with the minimized positions.
 *  @param[in] host_bounds Flat `num_atoms` squared bounds matrix.
 *  @param[in] num_atoms Atom count.
 *  @param[in] spatial_dimension Coordinate dimensionality.
 *  @param[in] upper_weight Penalty weight for upper-bound violations.
 *  @param[in] lower_weight Penalty weight for lower-bound violations.
 *  @param[in] fourth_dim_weight Penalty weight for the fourth-dimension term.
 *  @param[in] max_iterations Iteration cap for the FIRE loop.
 *  @param[in] gradient_threshold Convergence threshold on the gradient norm.
 *  @param[out] out_energy Receives the final energy.
 *  @param[out] out_iterations Receives the iteration count at convergence or the cap.
 *  @param[out] out_status Receives the FIRE termination status.
 */
void gpu_dg_minimize(float *host_coords, float const *host_bounds, std::uint32_t num_atoms,
                     std::uint32_t spatial_dimension, float upper_weight, float lower_weight, float fourth_dim_weight,
                     std::int32_t max_iterations, float gradient_threshold, float *out_energy,
                     std::int32_t *out_iterations, std::int32_t *out_status);

} // namespace smallfield::tests
