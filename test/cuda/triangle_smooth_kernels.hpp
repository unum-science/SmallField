/**
 *  @file test/cuda/triangle_smooth_kernels.hpp
 *  @brief Device harness for the shared-memory triangle-smoothing kernel, for use by its unit tests.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

#include "smallfield/cuda/triangle_smooth.cuh"
#include "smallfield/types/views.hpp"

namespace smallfield::tests {

/**
 *  @brief Atom ceiling of the warp-level kernel: the whole matrix has to fit one shared-memory tile. 128
 *  atoms is 64 KB of dynamic shared memory, which every architecture in `SMALLFIELD_CUDA_ARCH` grants once
 *  the kernel asks. The caller skips anything wider rather than watching the launch refuse.
 */
constexpr std::uint32_t max_smoothed_atoms_k = 128;

/**
 *  @brief Smooth one bounds matrix on the device, entirely in shared memory.
 *  @param[inout] bounds Square bounds matrix over the molecule's atoms, smoothed in place.
 *  @return What the sweep did to the matrix.
 */
smallfield::cuda::smoothing_t gpu_triangle_smooth(smallfield::types::packed_matrix<float> bounds);

} // namespace smallfield::tests
