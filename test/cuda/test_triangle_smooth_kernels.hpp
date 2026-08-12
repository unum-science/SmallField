/**
 *  @file test/cuda/test_triangle_smooth_kernels.hpp
 *  @brief Device harness for the shared-memory triangle-smoothing kernel, for use by its unit tests.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

#include "smallfield/cuda/triangle_smooth.cuh"

namespace smallfield::tests {

/**
 *  @brief Atom ceiling of the warp-level kernel: the whole matrix has to fit one shared-memory tile.
 *
 *  128 atoms is 64 KB of dynamic shared memory, which every architecture in `SMALLFIELD_CUDA_ARCH` grants
 *  once the kernel asks for it. The caller skips anything wider rather than watching the launch refuse.
 */
constexpr std::uint32_t max_smoothed_atoms_k = 128;

/**
 *  @brief Smooth one bounds matrix on the device, entirely in shared memory.
 *
 *  @param[inout] host_bounds Flat @p num_atoms times @p num_atoms bounds matrix, smoothed in place.
 *  @param[in] num_atoms Atom count of the matrix.
 *  @return What the sweep did to the matrix.
 */
smallfield::cuda::smoothing_t gpu_triangle_smooth(float *host_bounds, std::uint32_t num_atoms);

} // namespace smallfield::tests
