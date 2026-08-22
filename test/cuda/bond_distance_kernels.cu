/**
 *  @file test/cuda/bond_distance_kernels.cu
 *  @brief Device-side driver for the hop-count BFS parity test.
 */

#include <cstddef> // `std::byte`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <fmt/format.h> // `fmt::println` -- host-side only; nothing below a `__global__` uses it

#include <cuda_runtime.h>

#include "smallfield/cuda/bond_distance_kernel.cuh"
#include "smallfield/types/storage.hpp"

#include "test/cuda/bond_distance_kernels.hpp"

namespace sfc = smallfield::cuda;
namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;

namespace smallfield::tests {

void run_bond_distance_bfs_on_gpu(custd::span<std::byte const> host_uploaded, sfp::wave_slice_shape_t const &shape,
                                  custd::span<std::uint8_t> bond_distance_out) {
    std::size_t const slice_bytes = sfp::wave_slice_bytes(shape);
    std::uint32_t const molecule_count = shape.totals.valid;
    if (slice_bytes == 0 || molecule_count == 0) return;

    // The BFS launch and both transfers ride the legacy default stream, so this allocation is drawn from
    // the pool and released against that same stream, needing no synchronize of its own.
    sft::device_bytes_t device_pool;
    if (device_pool.try_reserve(slice_bytes) == sft::reserve_outcome_t::failed_k) {
        fmt::println(stderr, "NOTE bond_distance_kernels field=device_pool bytes={} succeeded=0", slice_bytes);
        return;
    }
    sfp::pool_cursor<std::byte> cursor {device_pool.view()};
    sfp::wave_slice_view_t const device_view = sfp::carve_wave_slice(shape, cursor);

    // The uploaded prefix is what production ships; the hop-count region past it is poisoned first so a
    // kernel that fails to write a byte cannot pass by accident.
    if (!sft::upload_into(device_view.uploaded, host_uploaded, nullptr)) {
        fmt::println(stderr, "NOTE bond_distance_kernels field=upload bytes={} succeeded=0", host_uploaded.size());
        return;
    }
    custd::span<std::uint8_t> const device_hops = device_view.bond_distance;
    cudaMemset(device_hops.data(), 0xAB, device_hops.size_bytes());

    sfc::launch_bond_distance_bfs(device_view.descriptors.data(), device_view.molecule_metadata.data(),
                                  device_hops.data(), molecule_count);

    if (!sft::download_from(bond_distance_out, custd::span<std::uint8_t const> {device_hops}, nullptr))
        fmt::println(stderr, "NOTE bond_distance_kernels field=download hops={} succeeded=0", device_hops.size());
    cudaStreamSynchronize(nullptr);
}

} // namespace smallfield::tests
