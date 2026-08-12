/**
 *  @file test/cuda/test_csd_torsion_device_kernels.cu
 *  @brief Device-side half of the CSD torsion parity test.
 *
 *  Uploads the pattern table, runs the warp-per-molecule matcher over a batch of descriptors, and hands the
 *  torsions back to the host .cpp that compares them against `extract_etk_torsions_native`.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <fmt/format.h> // `fmt::println` -- host-side only; nothing below a `__global__` uses it

#include <cuda_runtime.h>

#include "smallfield/cuda/csd_torsion_kernel.cuh"
#include "smallfield/types/storage.hpp"

#include "test/cuda/test_csd_torsion_device_kernels.hpp"

namespace sfc = smallfield::cuda;
namespace sft = smallfield::types;

namespace smallfield::tests {

namespace {

/** Owns the staged and uploaded copies of the table for the lifetime of the test. */
struct csd_torsion_test_table_t {
    sfc::csd_torsion_table_host_t host;
    sfc::csd_torsion_table_device_t device;
};

} // namespace

/** Grow-only device storage over the legacy default stream, which is the one the matcher launches on. */
template <typename value_type_>
using device_array = sft::buffer<value_type_, sft::device_allocator_t>;

/** Every device array one batch stages, released in stream order on every path out of the launch. */
struct csd_torsion_arena_t {
    device_array<sft::molecule_descriptor_t> descriptors;
    device_array<sft::etk_torsion_descriptor_t> torsions;
    device_array<std::uint32_t> counts;

    /** Reserve one descriptor and one count per molecule, and @p torsion_stride output slots each. */
    [[nodiscard]] bool resize_for(std::uint32_t num_molecules, std::uint32_t torsion_stride) noexcept {
        bool reserved = true;
        auto reserve_ = [&](auto &array, std::size_t count) {
            reserved &= array.try_reserve(count) != sft::reserve_outcome_t::failed_k;
        };
        reserve_(descriptors, num_molecules);
        reserve_(torsions, std::size_t {num_molecules} * torsion_stride);
        reserve_(counts, num_molecules);
        return reserved;
    }
};

void *create_csd_torsion_device_table(std::size_t &table_bytes) {
    auto *table = new csd_torsion_test_table_t;
    sfc::build_csd_torsion_table(table->host);
    // The harness launches on the legacy default stream, so the upload rides it too and the matcher
    // is already ordered behind the copies.
    sfc::upload_csd_torsion_table_to_device(table->host, table->device, nullptr);
    if (table->device.view.patterns == nullptr) {
        delete table;
        return nullptr;
    }
    table_bytes = table->host.byte_size();
    return table;
}

void destroy_csd_torsion_device_table(void *handle) { delete static_cast<csd_torsion_test_table_t *>(handle); }

bool run_csd_torsion_device_batch(void *handle, custd::span<sft::molecule_descriptor_t const> descriptors,
                                  std::uint32_t torsion_stride, custd::span<sft::etk_torsion_descriptor_t> torsions_out,
                                  custd::span<std::uint32_t> counts_out) {
    auto *table = static_cast<csd_torsion_test_table_t *>(handle);
    std::uint32_t const num_molecules = static_cast<std::uint32_t>(descriptors.size());
    if (table == nullptr || num_molecules == 0) return false;

    csd_torsion_arena_t arena;
    if (!arena.resize_for(num_molecules, torsion_stride)) {
        fmt::println(stderr, "NOTE csd_torsion_kernels field=device_arena molecules={} succeeded=0", num_molecules);
        return false;
    }
    if (!sft::upload_into(arena.descriptors.view(), descriptors, nullptr)) {
        fmt::println(stderr, "NOTE csd_torsion_kernels field=upload molecules={} succeeded=0", num_molecules);
        return false;
    }
    cudaMemset(arena.counts.data(), 0, arena.counts.view().size_bytes());

    sfc::csd_torsion_match_kernel<<<num_molecules, sfc::warp_size_k, 0, nullptr>>>(
        arena.descriptors.data(), num_molecules, table->device.view, arena.torsions.data(), torsion_stride,
        arena.counts.data());
    cudaDeviceSynchronize();
    cudaError_t const error = cudaGetLastError();
    if (error != cudaSuccess)
        fmt::println(stderr, "NOTE csd_torsion_kernels field=launch cuda_error={}", cudaGetErrorString(error));

    custd::span<sft::etk_torsion_descriptor_t const> const device_torsions = arena.torsions.view();
    custd::span<std::uint32_t const> const device_counts = arena.counts.view();
    bool const copied = sft::download_from(torsions_out, device_torsions, nullptr) &&
                        sft::download_from(counts_out, device_counts, nullptr);
    cudaStreamSynchronize(nullptr);
    return error == cudaSuccess && copied;
}

} // namespace smallfield::tests
