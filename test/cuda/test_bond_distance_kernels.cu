/**
 *  @file test/cuda/test_bond_distance_kernels.cu
 *  @brief Device-side driver for the hop-count BFS parity test.
 */

#include <cstddef> // `std::byte`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <fmt/format.h> // `fmt::println` -- host-side only; nothing below a `__global__` uses it

#include <cuda_runtime.h>

#include "smallfield/cuda/bond_distance_kernel.cuh"
#include "smallfield/types/storage.hpp"

#include "test/cuda/test_bond_distance_kernels.hpp"

namespace sfa = smallfield::api;
namespace sfc = smallfield::cuda;
namespace sft = smallfield::types;

namespace smallfield::tests {

void run_bond_distance_bfs_on_gpu(custd::span<std::byte const> host_pool, sfa::wave_slice_layout_t const &layout,
                                  std::uint32_t molecule_count, custd::span<std::uint8_t> bond_distance_out) {
    if (layout.bytes_total == 0 || molecule_count == 0) return;

    // The BFS launch and both transfers ride the legacy default stream, so this allocation is drawn from
    // the device memory pool and released against that same stream, and needs no synchronize of its own.
    sft::device_bytes_t device_pool;
    if (device_pool.try_reserve(layout.bytes_total) == sft::reserve_outcome_t::failed_k) {
        fmt::println(stderr, "NOTE bond_distance_kernels field=device_pool bytes={} succeeded=0", layout.bytes_total);
        return;
    }
    // The uploaded prefix is exactly what production ships; the hop-count region past it is
    // poisoned first so a kernel that fails to write a byte cannot pass by accident.
    if (!sft::upload_into(device_pool.view(), host_pool.first(layout.bytes_uploaded), nullptr)) {
        fmt::println(stderr, "NOTE bond_distance_kernels field=upload bytes={} succeeded=0", layout.bytes_uploaded);
        return;
    }
    custd::span<std::uint8_t> const device_hops = layout.bond_distance.over<std::uint8_t>(device_pool.data());
    cudaMemset(device_hops.data(), 0xAB, device_hops.size_bytes());

    sfa::batch_pool_view_t const device_view {device_pool.data(), &layout};
    sfc::launch_bond_distance_bfs(device_view.descriptors().data(), device_view.molecule_metadata().data(),
                                  device_hops.data(), molecule_count);

    if (!sft::download_from(bond_distance_out, custd::span<std::uint8_t const> {device_hops}, nullptr))
        fmt::println(stderr, "NOTE bond_distance_kernels field=download hops={} succeeded=0", device_hops.size());
    cudaStreamSynchronize(nullptr);
}

} // namespace smallfield::tests
