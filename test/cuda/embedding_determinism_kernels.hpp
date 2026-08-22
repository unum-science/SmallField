/**
 *  @file test/cuda/embedding_determinism_kernels.hpp
 *  @brief Device harness for the metric-embedding reproducibility test, for use by its unit tests.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

#include "smallfield/types/views.hpp"

namespace smallfield::tests {

/**
 *  @brief Run `embed_4d_power_iter_kernel` over @p num_molecules copies of one bounds matrix.
 *
 *  Every block sees byte-identical input, so every block owes byte-identical output whatever streaming
 *  multiprocessor it lands on and whichever warp reaches a reduction first. The kernel smooths its bounds
 *  in place, so a fresh copy is uploaded per call.
 *  @param[in] host_bounds One square bounds matrix, whose extent is the molecule's atom count.
 *  @param[in] num_molecules Number of identical copies to embed, one block each.
 *  @param[out] host_coordinates Receives @p num_molecules times atoms times four floats, dimension major.
 *  @return Zero on success, or the CUDA error code that stopped the run.
 */
int gpu_embed_replicated(smallfield::types::packed_matrix<float const> host_bounds, std::uint32_t num_molecules,
                         custd::span<float> host_coordinates);

} // namespace smallfield::tests
