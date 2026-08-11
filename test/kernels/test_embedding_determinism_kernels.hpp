/**
 *  @file test/kernels/test_embedding_determinism_kernels.hpp
 *  @brief Device harness for the metric-embedding reproducibility test, for use by its unit tests.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

namespace smallfield::tests {

/**
 *  @brief Run `embed_4d_power_iter_kernel` over @p num_molecules copies of one bounds matrix.
 *
 *  Every block sees byte-identical input, so every block owes byte-identical output, whatever
 *  streaming multiprocessor it lands on and whichever of its warps arrives at a reduction first.
 *  The kernel smooths its bounds in place, so a fresh copy is uploaded on each call.
 *
 *  @param[in] host_bounds One @p num_atoms times @p num_atoms bounds matrix, row-major.
 *  @param[in] num_atoms Atom count of that matrix.
 *  @param[in] num_molecules Number of identical copies to embed, one block each.
 *  @param[out] host_coordinates Receives @p num_molecules times @p num_atoms times 4 floats, dimension major.
 *  @return Zero on success, or the CUDA error code that stopped the run.
 */
int gpu_embed_replicated(float const *host_bounds, std::uint32_t num_atoms, std::uint32_t num_molecules,
                         float *host_coordinates);

} // namespace smallfield::tests
