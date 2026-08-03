#pragma once

#include <cstddef>
#include <cstdint>

#include "smallfield/api/batch_pool.hpp"

/**
 *  @brief Mirror a prepared batch onto the device and run the warp BFS over it.
 *
 *  Allocates one device pool of @c layout.bytes_total, uploads the same
 *  @c layout.bytes_uploaded prefix the production launch path uploads, fills the
 *  device-only hop-count region with @ref smallfield::cuda::launch_bond_distance_bfs,
 *  and copies that region back.
 *
 *  @param[in] host_pool Base of the batch's pinned host buffer.
 *  @param[in] layout Byte layout of the batch's regions.
 *  @param[in] molecule_count Number of molecules in the batch.
 *  @param[out] bond_distance_out Receives @c layout.count_bond_distance bytes.
 */
void run_bond_distance_bfs_on_gpu(std::byte const *host_pool, smallfield::api::batch_layout_t const &layout,
                                  std::uint32_t molecule_count, std::uint8_t *bond_distance_out);
