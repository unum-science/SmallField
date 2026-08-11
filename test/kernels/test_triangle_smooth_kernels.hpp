/**
 *  @file test/kernels/test_triangle_smooth_kernels.hpp
 *  @brief Device harness for the shared-memory triangle-smoothing kernel, for use by its unit tests.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

namespace smallfield::tests {

/**
 *  @brief Smooth one bounds matrix on the device, entirely in shared memory.
 *
 *  @param[inout] host_bounds Flat @p num_atoms times @p num_atoms bounds matrix, smoothed in place.
 *  @param[in] num_atoms Atom count of the matrix.
 */
void gpu_triangle_smooth(float *host_bounds, std::uint32_t num_atoms);

} // namespace smallfield::tests
