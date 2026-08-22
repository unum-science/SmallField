/**
 *  @file test/cuda/test_bounds_infeasible_kernels.cu
 *  @brief CUDA driver for the bounds-infeasibility verdict test.
 *
 *  Compiled by nvcc. The host test logic is in test_bounds_infeasible.cpp, compiled by g++.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <array>  // `std::array`
#include <vector> // `std::vector`

#include <cuda_runtime.h>

#include "smallfield/cuda/etkdg_pipeline.cuh"
#include "smallfield/cuda/metric_embedding_prepass.cuh"
#include "smallfield/types/atom_limited_bins.hpp"
#include "smallfield/types/storage.hpp"

#include "test/cuda/test_bounds_infeasible_kernels.hpp"

namespace sfc = smallfield::cuda;
namespace sft = smallfield::types;

namespace smallfield::tests {

using smallfield::types::conformer_status_t;
using smallfield::types::etkdg_molecule_metadata_t;

/** Stream-ordered device storage, matching how the pipeline itself allocates. */
template <typename value_type_>
using device_array = sft::buffer<value_type_, sft::device_allocator_t>;

/** What `build_metric_initial_coordinates_on_gpu` binds its molecules into, one wave's worth. */
struct prepass_scratch_t {
    std::array<std::vector<std::uint32_t>, sft::atom_limited_bin_count_k> bin_groups;
    sft::device_indices_t group_indices;
    std::uint32_t max_atoms = 0;

    explicit prepass_scratch_t(cudaStream_t stream) noexcept : group_indices(sft::device_allocator_t {stream}) {}
};

/**
 *  @brief Every device array one molecule's pass through the three launches needs.
 *
 *  The five term arrays hold one unused element each: the molecule declares no chiral centers, torsions,
 *  constraints or interactions, so the kernel only offsets into them and a null would make that undefined.
 */
struct pipeline_arena_t {
    device_array<float> bounds;
    device_array<etkdg_molecule_metadata_t> metadata;
    device_array<std::uint32_t> input_indices;
    device_array<float> embedding;
    device_array<float> coordinates;
    device_array<float> energies;
    device_array<std::uint8_t> status;
    device_array<conformer_status_t> verdicts;
    device_array<sft::chiral_center_descriptor_t> chiral_centers;
    device_array<sft::etk_torsion_descriptor_t> torsions;
    device_array<sft::distance_constraint_t> constraints;
    device_array<sft::etk_atom_interaction_t> etk_interactions;
    device_array<sft::etk_atom_offsets_t> etk_atom_offsets;

    explicit pipeline_arena_t(cudaStream_t stream) noexcept
        : bounds(sft::device_allocator_t {stream}), metadata(sft::device_allocator_t {stream}),
          input_indices(sft::device_allocator_t {stream}), embedding(sft::device_allocator_t {stream}),
          coordinates(sft::device_allocator_t {stream}), energies(sft::device_allocator_t {stream}),
          status(sft::device_allocator_t {stream}), verdicts(sft::device_allocator_t {stream}),
          chiral_centers(sft::device_allocator_t {stream}), torsions(sft::device_allocator_t {stream}),
          constraints(sft::device_allocator_t {stream}), etk_interactions(sft::device_allocator_t {stream}),
          etk_atom_offsets(sft::device_allocator_t {stream}) {}

    /** Reserve every array for one molecule of @p num_atoms atoms; false when the device refused any. */
    [[nodiscard]] bool resize_for(std::uint32_t num_atoms) noexcept {
        bool reserved = true;
        auto reserve_ = [&](auto &array, std::size_t count) {
            reserved &= array.try_reserve(count) != sft::reserve_outcome_t::failed_k;
        };
        reserve_(bounds, std::size_t {num_atoms} * num_atoms);
        reserve_(metadata, 1);
        reserve_(input_indices, 1);
        reserve_(embedding, std::size_t {num_atoms} * sft::embedding_dimensions_k);
        reserve_(coordinates, std::size_t {num_atoms} * sft::conformer_dimensions_k);
        reserve_(energies, 1);
        reserve_(status, 1);
        reserve_(verdicts, 1);
        reserve_(chiral_centers, 1);
        reserve_(torsions, 1);
        reserve_(constraints, 1);
        reserve_(etk_interactions, 1);
        reserve_(etk_atom_offsets, 1);
        return reserved;
    }
};

conformer_status_t gpu_pipeline_status_of_bounds(sft::packed_matrix<float> bounds) {
    std::uint32_t const num_atoms = bounds.extent(0);
    std::size_t const cell_count = std::size_t {num_atoms} * num_atoms;
    // The prepass pads every molecule to its bin's cap and refuses a scratch sized for the molecule
    // instead, which is what a wave's own reservation carries.
    std::uint32_t const bin_atoms = sft::atom_limited_bin_caps_k[sft::atom_limited_bin_for(num_atoms)];

    // Neither launch ran, which is exactly what the caller should be told when staging refuses below.
    conformer_status_t verdict = conformer_status_t::mmff_unconverged_k;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return verdict;

    // The arenas live inside the stream's lifetime, so every stream-ordered free runs against a live stream.
    {
        pipeline_arena_t arena {stream};
        prepass_scratch_t scratch {stream};
        bool staged = arena.resize_for(num_atoms) &&
                      scratch.group_indices.try_reserve(1) != sft::reserve_outcome_t::failed_k;
        scratch.max_atoms = bin_atoms;

        etkdg_molecule_metadata_t metadata {};
        metadata.num_atoms = num_atoms;
        std::uint32_t const input_index = 0;
        // The status slot starts where no kernel would leave it, so a launch that never ran is legible;
        // the verdict starts saying nothing, the state a wave memsets it to.
        std::uint8_t const unreached = static_cast<std::uint8_t>(conformer_status_t::mmff_unconverged_k);
        conformer_status_t const no_complaint = conformer_status_t::success_k;

        staged =
            staged &&
            sft::upload_into(arena.bounds.view(), custd::span<float const> {bounds.data_handle(), cell_count},
                             stream) &&
            sft::upload_into(arena.metadata.view(), custd::span<etkdg_molecule_metadata_t const> {&metadata, 1},
                             stream) &&
            sft::upload_into(arena.input_indices.view(), custd::span<std::uint32_t const> {&input_index, 1}, stream) &&
            sft::upload_into(arena.status.view(), custd::span<std::uint8_t const> {&unreached, 1}, stream) &&
            sft::upload_into(arena.verdicts.view(), custd::span<conformer_status_t const> {&no_complaint, 1}, stream);
        staged = staged && cudaMemsetAsync(arena.coordinates.data(), 0, arena.coordinates.view().size_bytes(),
                                           stream) == cudaSuccess;

        if (staged) {
            // The wave raises this once per device at construction; a harness driving the kernel must too.
            cudaFuncSetAttribute(sfc::embed_4d_power_iter_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(sfc::embed_shared_bytes(bin_atoms)));

            [[maybe_unused]] bool const embedded = sfc::build_metric_initial_coordinates_on_gpu(
                arena.bounds.data(), arena.metadata.data(), custd::span<etkdg_molecule_metadata_t const> {&metadata, 1},
                num_atoms, arena.embedding.data(), arena.verdicts.data(), scratch, {}, stream);

            sfc::etkdg_pipeline_warp_kernel<<<1, sfc::warp_size_k, sfc::compute_pipeline_shared_memory_bytes(num_atoms),
                                              stream>>>(
                arena.bounds.data(), arena.embedding.data(), arena.chiral_centers.data(), arena.torsions.data(),
                arena.constraints.data(), arena.etk_interactions.data(), arena.etk_atom_offsets.data(),
                arena.metadata.data(), arena.input_indices.data(), arena.coordinates.data(), arena.energies.data(),
                arena.status.data(), 1u, 1u, 4u, num_atoms, num_atoms, num_atoms, 0x5eedull);

            // Shared memory scales with atom count, so this is the launch a device can refuse. Read before
            // the stamp is enqueued: the sticky slot holds one error and the caller needs to know whose.
            cudaError_t const pipeline_enqueued = cudaGetLastError();

            sfc::stamp_refused_molecules_kernel<<<1, 1, 0, stream>>>(arena.verdicts.data(), arena.status.data(), 1u,
                                                                     1u);
            cudaError_t const stamp_enqueued = cudaGetLastError();

            std::uint8_t status_byte = unreached;
            [[maybe_unused]] bool const read_back =
                sft::download_from(custd::span<float> {bounds.data_handle(), cell_count},
                                   custd::span<float const> {arena.bounds.view()}, stream) &&
                sft::download_from(custd::span<std::uint8_t> {&status_byte, 1},
                                   custd::span<std::uint8_t const> {arena.status.view()}, stream);

            cudaError_t const finished = cudaStreamSynchronize(stream);
            // A refused enqueue leaves the status at `unreached`, `mmff_unconverged_k`, the same byte a
            // molecule the kernels reached and failed to converge carries. The drain returns success
            // either way, so without these two a declined launch would read as a verdict.
            if (pipeline_enqueued != cudaSuccess)
                std::fprintf(stderr, "etkdg_pipeline_warp_kernel: refused at %u atoms: %s\n", num_atoms,
                             cudaGetErrorString(pipeline_enqueued));
            else if (stamp_enqueued != cudaSuccess)
                std::fprintf(stderr, "stamp_refused_molecules_kernel: %s\n", cudaGetErrorString(stamp_enqueued));
            else if (finished != cudaSuccess)
                std::fprintf(stderr, "gpu_pipeline_status_of_bounds: %s\n", cudaGetErrorString(finished));
            else verdict = static_cast<conformer_status_t>(status_byte);
        }
        else { std::fprintf(stderr, "gpu_pipeline_status_of_bounds: device staging refused %u atoms\n", num_atoms); }
    }

    cudaStreamDestroy(stream);
    return verdict;
}

} // namespace smallfield::tests
