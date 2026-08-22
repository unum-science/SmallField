/**
 *  @file test/cuda/bond_distance_kernels.hpp
 *  @brief Device harness for the bond-distance BFS kernel, for use by its unit tests.
 */

#pragma once

#include <cstddef> // `std::byte`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include "smallfield/preprocess/wave_slice_pool.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::tests {

/**
 *  @brief Mirror a prepared batch onto the device and run the warp BFS over it.
 *
 *  Reserves one device pool the whole slice spans, uploads the prefix the production launch path uploads,
 *  fills the device-only hop-count region with `smallfield::cuda::launch_bond_distance_bfs` and copies it
 *  back. A pool it cannot reserve leaves @p bond_distance_out untouched and says so on `stderr`.
 *  @param[in] host_uploaded The batch's uploaded prefix, straight off its pinned host view.
 *  @param[in] shape Region extents the device mirror is carved with.
 *  @param[out] bond_distance_out Receives one hop count per element of the slice's hop-count region.
 */
void run_bond_distance_bfs_on_gpu(custd::span<std::byte const> host_uploaded,
                                  smallfield::preprocess::wave_slice_shape_t const &shape,
                                  custd::span<std::uint8_t> bond_distance_out);

} // namespace smallfield::tests
