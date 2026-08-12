/**
 *  @file test/cuda/test_embedding_determinism_kernels.cu
 *  @brief CUDA driver for the metric-embedding reproducibility test.
 *
 *  Compiled by nvcc. The host test logic is in test_embedding_determinism.cpp, compiled by g++.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <vector> // `std::vector`

#include <cuda_runtime.h>

#include "smallfield/api/conformer_status.hpp"
#include "smallfield/cuda/metric_embedding_prepass.cuh"
#include "smallfield/types/storage.hpp"

#include "test/cuda/test_embedding_determinism_kernels.hpp"

namespace sfc = smallfield::cuda;
namespace sft = smallfield::types;
using smallfield::types::etkdg_molecule_metadata_t;

namespace smallfield::tests {

/** Grow-only device storage over the legacy default stream, which is the one the embedding launches on. */
template <typename value_type_>
using device_array = sft::buffer<value_type_, sft::device_allocator_t>;

/** Every device array the replicated embedding stages, released in stream order however the call leaves. */
struct embedding_arena_t {
    device_array<float> bounds;
    device_array<etkdg_molecule_metadata_t> metadata;
    device_array<std::uint32_t> indices;
    device_array<float> coordinates;
    /** One verdict per copy, which the kernel writes only where a matrix turns out impossible. */
    device_array<smallfield::api::conformer_status_t> verdicts;

    /** Reserve one slot per replicated copy; false when the device refused any of the five arrays. */
    [[nodiscard]] bool resize_for(std::uint32_t num_molecules, std::size_t entries_per_molecule,
                                  std::size_t coordinates_per_molecule) noexcept {
        bool reserved = true;
        auto reserve_ = [&](auto &array, std::size_t count) {
            reserved &= array.try_reserve(count) != sft::reserve_outcome_t::failed_k;
        };
        reserve_(bounds, entries_per_molecule * num_molecules);
        reserve_(metadata, num_molecules);
        reserve_(indices, num_molecules);
        reserve_(coordinates, coordinates_per_molecule * num_molecules);
        reserve_(verdicts, num_molecules);
        return reserved;
    }
};

int gpu_embed_replicated(sft::packed_matrix<float const> host_bounds, std::uint32_t num_molecules,
                         custd::span<float> host_coordinates) {
    std::uint32_t const num_atoms = host_bounds.extent(0);
    std::size_t const entries_per_molecule = std::size_t {num_atoms} * num_atoms;
    std::size_t const coordinates_per_molecule = std::size_t {num_atoms} * sft::embedding_dimensions_k;

    std::vector<float> bounds(entries_per_molecule * num_molecules);
    std::vector<etkdg_molecule_metadata_t> metadata(num_molecules);
    std::vector<std::uint32_t> indices(num_molecules);
    for (std::uint32_t molecule = 0; molecule < num_molecules; ++molecule) {
        for (std::size_t entry = 0; entry < entries_per_molecule; ++entry)
            bounds[molecule * entries_per_molecule + entry] = host_bounds.data_handle()[entry];
        metadata[molecule] = etkdg_molecule_metadata_t {};
        metadata[molecule].num_atoms = num_atoms;
        metadata[molecule].bounds_offset = static_cast<std::uint32_t>(molecule * entries_per_molecule);
        indices[molecule] = molecule;
    }

    embedding_arena_t arena;
    if (!arena.resize_for(num_molecules, entries_per_molecule, coordinates_per_molecule))
        return static_cast<int>(cudaErrorMemoryAllocation);

    if (!sft::upload_into(arena.bounds.view(), custd::span<float const> {bounds}, nullptr) ||
        !sft::upload_into(arena.metadata.view(), custd::span<etkdg_molecule_metadata_t const> {metadata}, nullptr) ||
        !sft::upload_into(arena.indices.view(), custd::span<std::uint32_t const> {indices}, nullptr))
        return static_cast<int>(cudaErrorInvalidValue);

    std::size_t const shared_bytes = sfc::embed_shared_bytes(num_atoms);
    cudaFuncSetAttribute(sfc::embed_4d_power_iter_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared_bytes));

    cudaMemset(arena.coordinates.data(), 0, arena.coordinates.view().size_bytes());
    cudaMemset(arena.verdicts.data(), 0, arena.verdicts.view().size_bytes());
    sfc::embed_4d_power_iter_kernel<<<num_molecules, sfc::embed_block_size_k, shared_bytes>>>(
        arena.bounds.data(), arena.metadata.data(), arena.indices.data(), arena.coordinates.data(), num_atoms,
        arena.verdicts.data());
    cudaError_t status = cudaDeviceSynchronize();
    if (status == cudaSuccess) status = cudaGetLastError();

    custd::span<float const> const device_coordinates = arena.coordinates.view();
    if (status == cudaSuccess && !sft::download_from(host_coordinates, device_coordinates, nullptr))
        status = cudaErrorInvalidValue;
    cudaStreamSynchronize(nullptr);
    return static_cast<int>(status);
}

} // namespace smallfield::tests
