/**
 *  @file test/cuda/test_csd_torsion_device_kernels.cu
 *  @brief Device-side half of the CSD torsion parity test.
 *
 *  Uploads the pattern table, runs the warp-per-molecule matcher over a batch of descriptors, and hands the
 *  torsions back to the host .cpp that compares them against `extract_etk_torsions_native`.
 */

#include <cstdint> // `std::uint32_t`

#include <fmt/format.h> // `fmt::println` -- host-side only; nothing below a `__global__` uses it

#include <cuda_runtime.h>

#include "smallfield/cuda/csd_torsion_kernel.cuh"

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

bool run_csd_torsion_device_batch(void *handle, sft::molecule_descriptor_t const *descriptors,
                                  std::uint32_t num_molecules, std::uint32_t torsion_stride,
                                  sft::etk_torsion_descriptor_t *torsions_out, std::uint32_t *counts_out) {
    auto *table = static_cast<csd_torsion_test_table_t *>(handle);
    if (table == nullptr || num_molecules == 0) return false;

    sft::molecule_descriptor_t *descriptors_device = nullptr;
    sft::etk_torsion_descriptor_t *torsions_device = nullptr;
    std::uint32_t *counts_device = nullptr;
    std::size_t const torsion_slots = static_cast<std::size_t>(num_molecules) * torsion_stride;
    cudaMalloc(&descriptors_device, num_molecules * sizeof(sft::molecule_descriptor_t));
    cudaMalloc(&torsions_device, torsion_slots * sizeof(sft::etk_torsion_descriptor_t));
    cudaMalloc(&counts_device, num_molecules * sizeof(std::uint32_t));
    if (descriptors_device == nullptr || torsions_device == nullptr || counts_device == nullptr) return false;

    cudaMemcpy(descriptors_device, descriptors, num_molecules * sizeof(sft::molecule_descriptor_t),
               cudaMemcpyHostToDevice);
    cudaMemset(counts_device, 0, num_molecules * sizeof(std::uint32_t));

    sfc::launch_csd_torsion_match(descriptors_device, num_molecules, table->device.view, torsions_device,
                                  torsion_stride, counts_device);
    cudaDeviceSynchronize();
    cudaError_t const error = cudaGetLastError();
    if (error != cudaSuccess)
        fmt::println(stderr, "NOTE csd_torsion_kernels field=launch cuda_error={}", cudaGetErrorString(error));

    cudaMemcpy(torsions_out, torsions_device, torsion_slots * sizeof(sft::etk_torsion_descriptor_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(counts_out, counts_device, num_molecules * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(descriptors_device);
    cudaFree(torsions_device);
    cudaFree(counts_device);
    return error == cudaSuccess;
}

} // namespace smallfield::tests
