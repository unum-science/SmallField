/**
 *  @file test/kernels/test_bond_distance_kernels.hpp
 *  @brief Device harness for the bond-distance BFS kernel, for use by its unit tests.
 */

#pragma once

#include <cstddef> // `std::byte`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include "smallfield/api/wave_slice_pool.hpp"

namespace smallfield::tests {

/**
 *  @brief Mirror a prepared batch onto the device and run the warp BFS over it.
 *
 *  Allocates one device pool of `layout.bytes_total`, uploads the same
 *  `layout.bytes_uploaded` prefix the production launch path uploads, fills the
 *  device-only hop-count region with `smallfield::cuda::launch_bond_distance_bfs`,
 *  and copies that region back.
 *
 *  @param[in] host_pool Base of the batch's pinned host buffer.
 *  @param[in] layout Byte layout of the batch's regions.
 *  @param[in] molecule_count Number of molecules in the batch.
 *  @param[out] bond_distance_out Receives `layout.num_bond_distance` bytes.
 */
void run_bond_distance_bfs_on_gpu(std::byte const *host_pool, smallfield::api::wave_slice_layout_t const &layout,
                                  std::uint32_t molecule_count, std::uint8_t *bond_distance_out);

} // namespace smallfield::tests
