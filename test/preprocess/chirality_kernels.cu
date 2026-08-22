/**
 *  @file test/preprocess/chirality_kernels.cu
 *  @brief CUDA kernel wrappers for chirality validation test.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <cuda_runtime.h>

#include "smallfield/cuda/dg_kernel.cuh"
#include "smallfield/types/storage.hpp"

#include "test/preprocess/chirality_kernels.hpp"

namespace sft = smallfield::types;
namespace sfc = smallfield::cuda;

__global__ void chirality_check_kernel(float const *positions, std::uint32_t num_atoms,
                                       sft::chiral_center_descriptor_t const *chirals, std::uint32_t num_chirals,
                                       bool *out_result) {

    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const invalid = sfc::count_invalid_chiral_centers_(positions, num_atoms, chirals, num_chirals, lane);
    if (lane == 0) { *out_result = invalid == 0; }
}

namespace smallfield::tests {

/** Grow-only device storage over the stream this check creates for itself. */
template <typename value_type_>
using device_array = sft::buffer<value_type_, sft::device_allocator_t>;

/** Every device array one chirality check stages, released in stream order on every path out of the call. */
struct chirality_arena_t {
    device_array<float> coordinates;
    device_array<sft::chiral_center_descriptor_t> centers;
    device_array<bool> verdict;

    /** Bind every array to the stream the check's transfers and launch ride, so each free reaches it. */
    explicit chirality_arena_t(sft::device_allocator_t allocator) noexcept
        : coordinates(allocator), centers(allocator), verdict(allocator) {}

    /** Reserve one molecule's coordinates and centers; false when the device refused any of them. */
    [[nodiscard]] bool resize_for(std::size_t num_components, std::size_t num_centers) noexcept {
        bool reserved = true;
        auto reserve_ = [&](auto &array, std::size_t count) {
            reserved &= array.try_reserve(count) != sft::reserve_outcome_t::failed_k;
        };
        reserve_(coordinates, num_components);
        reserve_(centers, num_centers);
        reserve_(verdict, 1);
        return reserved;
    }
};

chirality_verdict_t gpu_validate_chirality(conformer_soa_reader_t host_coords,
                                           custd::span<sft::chiral_center_descriptor_t const> host_chirals) {

    std::uint32_t const num_atoms = host_coords.extent(0);
    std::size_t const num_components = sft::conformer_dimensions_k * std::size_t {num_atoms};

    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
        return chirality_verdict_t::unavailable_k;

    // Only an answer that came back off the stream decides this, so a staging or launch failure leaves the
    // caller a refusal rather than a violation the device never reported.
    chirality_verdict_t verdict = chirality_verdict_t::unavailable_k;

    // The arrays live inside the stream's lifetime, so every stream-ordered free runs against a live stream.
    {
        chirality_arena_t arena {sft::device_allocator_t {.stream = stream}};
        custd::span<float const> const coordinates {host_coords.data_handle(), num_components};

        bool staged = arena.resize_for(num_components, host_chirals.size());
        staged = staged && sft::upload_into(arena.coordinates.view(), coordinates, stream);
        staged = staged && sft::upload_into(arena.centers.view(), host_chirals, stream);

        if (staged) {
            chirality_check_kernel<<<1, sfc::warp_size_k, 0, stream>>>(
                arena.coordinates.data(), num_atoms, arena.centers.data(),
                static_cast<std::uint32_t>(host_chirals.size()), arena.verdict.data());

            // A refused enqueue reports here and nowhere else: the drain below succeeds over an empty stream and
            // the download reads whatever the allocation held, and an unwritten zero byte reads as `violated_k`,
            // the very answer these tests exist to detect.
            cudaError_t const enqueued = cudaGetLastError();

            bool satisfied = false;
            custd::span<bool const> const device_verdict = arena.verdict.view();
            bool const read_back = sft::download_from(custd::span<bool> {&satisfied, 1}, device_verdict, stream);
            cudaError_t const drained = cudaStreamSynchronize(stream);
            if (enqueued == cudaSuccess && read_back && drained == cudaSuccess)
                verdict = satisfied ? chirality_verdict_t::satisfied_k : chirality_verdict_t::violated_k;
        }
    }

    cudaStreamDestroy(stream);
    return verdict;
}

} // namespace smallfield::tests
