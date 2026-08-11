/**
 *  @file test/cuda/test_bond_distance_kernels.cu
 *  @brief Device-side driver for the hop-count BFS parity test.
 */

#include <fmt/format.h> // `fmt::println` -- host-side only; nothing below a `__global__` uses it

#include <cuda_runtime.h>

#include "smallfield/cuda/bond_distance_kernel.cuh"

#include "test/cuda/test_bond_distance_kernels.hpp"

namespace sfa = smallfield::api;
namespace sfc = smallfield::cuda;

namespace smallfield::tests {

void run_bond_distance_bfs_on_gpu(std::byte const *host_pool, sfa::wave_slice_layout_t const &layout,
                                  std::uint32_t molecule_count, std::uint8_t *bond_distance_out) {
    if (layout.bytes_total == 0 || molecule_count == 0) return;

    std::byte *device_pool = nullptr;
    if (cudaMalloc(&device_pool, layout.bytes_total) != cudaSuccess) {
        fmt::println(stderr, "NOTE bond_distance_kernels field=cuda_malloc bytes={} succeeded=0", layout.bytes_total);
        return;
    }
    // The uploaded prefix is exactly what production ships; the hop-count region past it is
    // poisoned first so a kernel that fails to write a byte cannot pass by accident.
    cudaMemcpy(device_pool, host_pool, layout.bytes_uploaded, cudaMemcpyHostToDevice);
    cudaMemset(device_pool + layout.bond_distance_byte_offset, 0xAB, layout.num_bond_distance);

    sfa::batch_pool_view_t device_view {device_pool, &layout};
    sfc::launch_bond_distance_bfs(device_view.descriptors().data(), device_view.molecule_metadata().data(),
                                  device_view.bond_distance().data(), molecule_count);
    cudaDeviceSynchronize();
    cudaMemcpy(bond_distance_out, device_pool + layout.bond_distance_byte_offset, layout.num_bond_distance,
               cudaMemcpyDeviceToHost);
    cudaFree(device_pool);
}

} // namespace smallfield::tests
