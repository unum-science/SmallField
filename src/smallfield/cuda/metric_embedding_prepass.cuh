/**
 *  @file src/smallfield/cuda/metric_embedding_prepass.cuh
 *  @brief GPU prepass for bounds smoothing and native 4D metric embedding.
 *
 *  Implements the first stage of ETKDG conformer generation on the GPU:
 *  1. Triangle-inequality bounds smoothing, Floyd-Warshall style, on the upper and lower distance bounds matrix.
 *  2. Gram matrix construction from midpoint squared distances with double-centering.
 *  3. Deflated power iteration to extract the top-4 eigenvectors of the Gram matrix.
 *  4. Projection into 4D metric coordinates as the initial embedding for subsequent ETKDG refinement stages.
 *
 *  Molecules are grouped by atom count into fixed bins so each kernel launch operates on identically-sized
 *  matrices.
 */

#pragma once

#if defined(__CUDACC__)

#include <array>   // `std::array`
#include <cstdint> // `std::uint32_t`
#include <vector>  // `std::vector`

#include <cuda_runtime.h>

#include "smallfield/cuda/warp_primitives.cuh"
#include "smallfield/cuda_error.hpp"
#include "smallfield/types/atom_limited_bins.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::cuda {
using smallfield::types::etkdg_molecule_metadata_t;
using smallfield::types::lower_distance_bound_at;
using smallfield::types::packed_matrix;
using smallfield::types::upper_distance_bound_at;

/** Thread count of every `embed_4d_power_iter_kernel` block; one block handles one molecule. */
constexpr std::uint32_t embed_block_size_k = 128;

/** One staging slot per warp for the block-wide sum in `embed_4d_power_iter_kernel`. */
constexpr std::uint32_t embed_reduction_slots_k = embed_block_size_k / warp_size_k;

/**
 *  @brief Dynamic shared floats `embed_4d_power_iter_kernel` needs for a molecule of @p num_atoms atoms.
 *
 *  States the layout the kernel carves, term for term: the Gram matrix, the row means widened to hold the
 *  block reduction's staging slots, the grand mean, four eigenvector estimates, the mat-vec scratch, and
 *  four eigenvalues. Every block derives its own layout from its own atom count, so a launch covering
 *  several molecules sizes its request from the largest.
 *
 *  @param[in] num_atoms Atoms the launch is sized against.
 *  @return Floats of dynamic shared memory to request.
 */
__host__ __device__ __forceinline__ std::size_t embed_shared_floats(std::uint32_t num_atoms) noexcept {
    std::size_t const atoms = num_atoms;
    std::size_t const row_means_extent = atoms > embed_reduction_slots_k ? atoms : embed_reduction_slots_k;
    return atoms * atoms + row_means_extent + 1u + 4u * atoms + atoms + 4u;
}

/** Bytes of the same request. */
__host__ __device__ __forceinline__ std::size_t embed_shared_bytes(std::uint32_t num_atoms) noexcept {
    return embed_shared_floats(num_atoms) * sizeof(float);
}

/**
 *  @brief Returns ((upper + lower) / 2)² for Gram-matrix double-centering.
 *
 *  Computes the squared midpoint distance between two atoms from their upper and lower bounds.
 *  Returns zero when atom_i equals atom_j.
 *
 *  @param[in] bounds Square bounds matrix over the molecule's atoms.
 *  @param[in] atom_i First atom index.
 *  @param[in] atom_j Second atom index.
 *  @return Squared midpoint distance between atom_i and atom_j.
 *  @see embed_4d_power_iter_kernel
 */
__device__ __forceinline__ float midpoint_distance_squared_(packed_matrix<float const> bounds, std::uint32_t atom_i,
                                                            std::uint32_t atom_j) {
    if (atom_i == atom_j) return 0.0f;
    float const upper = upper_distance_bound_at(bounds, atom_i, atom_j);
    float const lower = lower_distance_bound_at(bounds, atom_i, atom_j);
    float const distance = 0.5f * (fmaxf(lower, 0.0f) + fmaxf(upper, lower));
    return distance * distance;
}

/**
 *  @brief Deterministic block-wide float sum.
 *
 *  Reduces inside each warp with butterfly shuffles, stages one partial per warp in
 *  @p warp_partials, then has every thread re-add the partials in warp order. The
 *  summation tree depends only on lane and warp index, so the result is bitwise
 *  identical on every run and in every thread — the property a shared-memory float
 *  `atomicAdd` cannot offer, since it commits contributions in whatever order the
 *  scheduler grants and float addition does not associate.
 *
 *  Both barriers live inside, so a caller may invoke this straight out of a loop that
 *  produced @p value and consume the return value immediately. Every thread of the
 *  block must reach the call.
 *
 *  @param[in] value This thread's contribution to the sum.
 *  @param[inout] warp_partials Shared scratch of at least `embed_reduction_slots_k` floats.
 *  @return The sum across the whole block, identical in every thread.
 */
__device__ __forceinline__ float embed_block_reduce_sum_(float value, float *warp_partials) {
    sf_assert_(blockDim.x == embed_block_size_k);
    std::uint32_t const lane_in_warp = threadIdx.x % warp_size_k;
    std::uint32_t const warp_index = threadIdx.x / warp_size_k;
    float const warp_sum = warp_reduce_sum_f32_(value);
    __syncthreads(); // Readers of the previous reduction must finish before the restage
    if (lane_in_warp == 0) warp_partials[warp_index] = warp_sum;
    __syncthreads();
    float total = 0.0f;
    for (std::uint32_t warp = 0; warp < embed_reduction_slots_k; ++warp) total += warp_partials[warp];
    return total;
}

/**
 *  @brief Fused bounds-smooth → Gram → top-4 eigen → 4D embedding kernel.
 *
 *  One block of `embed_block_size_k` threads per molecule. Smooths the bounds with
 *  triangle inequalities, builds the double-centered Gram matrix, runs deflated power
 *  iteration to extract the four largest eigenvalues + eigenvectors, and writes the 4D
 *  embedding directly into `initial_coordinates` — one launch, no cusolver, no host
 *  workspace, no inter-kernel Gram-matrix global memory traffic.
 *
 *  Every block-wide sum goes through `embed_block_reduce_sum_`, so the embedding is a
 *  pure function of the bounds matrix: bitwise equal across launches and across blocks,
 *  which is what lets a shipped shard be regenerated. test_embedding_determinism pins it.
 *
 *  Shared memory layout (all floats):
 *
 *  @verbatim
 *    gram[N x N]                (overwrites bounds buffer after smoothing)
 *    row_means[max(N, 4)]       (Gram build; reduction staging afterwards)
 *    grand_mean[1]              (transient)
 *    v[4][N]                    (eigenvector estimates)
 *    gram_product[N]            (mat-vec scratch)
 *    lambda[4]                  (eigenvalue estimates)
 *  @endverbatim
 *
 *  `embed_shared_floats` states this same layout as an expression, and every launch site sizes its request
 *  from it rather than restating the sum.
 *
 *  @param[inout] bounds_matrices Flat per-molecule bounds matrices, smoothed in place.
 *  @param[in] molecule_metadata Per-molecule atom counts and bounds offsets.
 *  @param[in] molecule_indices Molecule index handled by each block.
 *  @param[out] initial_coordinates Destination for the 4D metric embedding.
 *  @param[in] max_atoms_per_molecule Stride between coordinate dimensions.
 */
__global__ void embed_4d_power_iter_kernel(float *__restrict__ bounds_matrices,
                                           etkdg_molecule_metadata_t const *__restrict__ molecule_metadata,
                                           std::uint32_t const *__restrict__ molecule_indices,
                                           float *__restrict__ initial_coordinates,
                                           std::uint32_t max_atoms_per_molecule) {
    std::uint32_t const group_index = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const block_size = blockDim.x;
    std::uint32_t const molecule_index = molecule_indices[group_index];
    auto const meta = molecule_metadata[molecule_index];
    std::uint32_t const num_atoms = meta.num_atoms;
    if (num_atoms == 0) return;

    extern __shared__ float embed_shared_raw[];
    // The row-mean region is at least one slot per warp so it can stage the block reductions
    // once the Gram build that produced the row means is done with it.
    std::uint32_t const row_means_extent = max(num_atoms, embed_reduction_slots_k);
    float *shared_gram = embed_shared_raw; // N × N, reused for bounds smoothing
    float *row_means = shared_gram + num_atoms * num_atoms;
    float *warp_partials = row_means;                     // Live only after the Gram build
    float *grand_mean_ptr = row_means + row_means_extent; // 1
    // The four eigenvector estimates are one contiguous 4 × N block, so each estimate is reached by pointer
    // arithmetic on the loop counter. An array of four pointers indexed by that counter would be a
    // runtime-indexed local array, which nvcc places on the stack rather than in registers, costing a
    // per-thread spill for no gain over the multiply.
    float *eigenvectors = grand_mean_ptr + 1; // 4 × N
    float *gram_product = eigenvectors + 4 * num_atoms;
    float *lambda_ptr = gram_product + num_atoms; // 4

    float *global_bounds = bounds_matrices + meta.bounds_offset;
    float *molecule_coordinates = initial_coordinates + molecule_index * max_atoms_per_molecule * 4;
    std::uint32_t const total_entries = num_atoms * num_atoms;

    // Load bounds into shared memory; shared_gram doubles as the bounds scratch.
    for (std::uint32_t linear_index = lane; linear_index < total_entries; linear_index += block_size)
        shared_gram[linear_index] = global_bounds[linear_index];
    __syncthreads();

    // A read view over the shared block while it still holds bounds. The element type is `const` because
    // every access through it is a read; it carries no aliasing promise, so the in-place updates below and
    // the Gram build further down stay visible through it.
    packed_matrix<float const> const shared_bounds {shared_gram, num_atoms, num_atoms};

    // Floyd-Warshall triangle smoothing on the N × N block.
    for (std::uint32_t pivot = 0; pivot < num_atoms; ++pivot) {
        for (std::uint32_t linear_index = lane; linear_index < total_entries; linear_index += block_size) {
            std::uint32_t const row = linear_index / num_atoms;
            std::uint32_t const column = linear_index % num_atoms;
            if (row == column || row == pivot || column == pivot) continue;
            float const upper_row_pivot = upper_distance_bound_at(shared_bounds, row, pivot);
            float const upper_pivot_column = upper_distance_bound_at(shared_bounds, pivot, column);
            // Above the diagonal the cell is the pair's upper bound, below it the lower one, so the branch
            // picks which of the two inequalities applies rather than which cell to touch. The lower bounds
            // stay inside the branch that consults them, where they cost no register on the other path.
            if (row < column) {
                float const candidate = upper_row_pivot + upper_pivot_column;
                float &current = shared_gram[row * num_atoms + column];
                if (candidate < current) current = candidate;
            }
            else {
                float const lower_row_pivot = lower_distance_bound_at(shared_bounds, row, pivot);
                float const lower_pivot_column = lower_distance_bound_at(shared_bounds, pivot, column);
                float &current = shared_gram[row * num_atoms + column];
                current = fmaxf(current,
                                fmaxf(lower_row_pivot - upper_pivot_column, lower_pivot_column - upper_row_pivot));
            }
        }
        __syncthreads();
    }

    // Persist smoothed bounds back to global memory.
    for (std::uint32_t linear_index = lane; linear_index < total_entries; linear_index += block_size)
        global_bounds[linear_index] = shared_gram[linear_index];

    // Row means and grand mean for double-centering.
    for (std::uint32_t atom = lane; atom < num_atoms; atom += block_size) {
        float row_sum = 0.0f;
        for (std::uint32_t other = 0; other < num_atoms; ++other)
            row_sum += midpoint_distance_squared_(shared_bounds, atom, other);
        row_means[atom] = row_sum / static_cast<float>(num_atoms);
    }
    __syncthreads();
    if (lane == 0) {
        float grand_mean = 0.0f;
        for (std::uint32_t atom = 0; atom < num_atoms; ++atom) grand_mean += row_means[atom];
        grand_mean_ptr[0] = grand_mean / static_cast<float>(num_atoms);
    }
    __syncthreads();
    float const grand_mean = grand_mean_ptr[0];

    // Build the Gram matrix in place, overwriting the smoothed bounds. midpoint_distance_squared_ reads both
    // the upper and the lower bound of a pair, so a cell must not be overwritten while another thread still
    // needs it. Walking the upper triangle makes each unordered atom pair the property of exactly one thread:
    // that thread reads both of the pair's cells and, since the double-centered Gram matrix is symmetric,
    // writes the one value it computed into both. No cell is ever touched by two threads, so no staging across
    // a barrier is needed and the kernel keeps every intermediate in registers.
    for (std::uint32_t linear_index = lane; linear_index < total_entries; linear_index += block_size) {
        std::uint32_t const row = linear_index / num_atoms;
        std::uint32_t const column = linear_index % num_atoms;
        if (row > column) continue;
        float const squared_distance = midpoint_distance_squared_(shared_bounds, row, column);
        float const centered = -0.5f * (squared_distance - row_means[row] - row_means[column] + grand_mean);
        shared_gram[row * num_atoms + column] = centered;
        shared_gram[column * num_atoms + row] = centered;
    }
    __syncthreads();

    // Deflated power iteration for the top-4 eigenvectors. For each estimate in turn, iterate
    // v_k ← (G - Σ_{j<k} λ_j v_j v_j^T) v_k / ‖·‖. With positive-semidefinite Gram matrices the top
    // eigenvalues are all non-negative and power iteration converges.
    constexpr std::uint32_t max_power_iterations_k = 60;
    constexpr float convergence_tolerance_k = 1e-4f;
    for (std::uint32_t eigenvector_index = 0; eigenvector_index < 4; ++eigenvector_index) {
        float *current_eigenvector = eigenvectors + eigenvector_index * num_atoms;
        // Initialize with a non-trivial vector orthogonal to the estimates already found, for a better
        // deflation start: the matching basis vector plus a small spread.
        for (std::uint32_t atom = lane; atom < num_atoms; atom += block_size)
            current_eigenvector[atom] = (atom == eigenvector_index) ? 1.0f : 0.5f / static_cast<float>(num_atoms);
        __syncthreads();
        // Normalize the initial estimate to unit length.
        float local_square_sum = 0.0f;
        for (std::uint32_t atom = lane; atom < num_atoms; atom += block_size)
            local_square_sum += current_eigenvector[atom] * current_eigenvector[atom];
        float const current_eigenvector_norm = sqrtf(
            fmaxf(embed_block_reduce_sum_(local_square_sum, warp_partials), 1e-20f));
        for (std::uint32_t atom = lane; atom < num_atoms; atom += block_size)
            current_eigenvector[atom] /= current_eigenvector_norm;
        __syncthreads();

        float lambda_prev = 0.0f;
        for (std::uint32_t iteration = 0; iteration < max_power_iterations_k; ++iteration) {
            // gram_product = G · current_eigenvector
            for (std::uint32_t row = lane; row < num_atoms; row += block_size) {
                float sum = 0.0f;
                for (std::uint32_t column = 0; column < num_atoms; ++column)
                    sum += shared_gram[row * num_atoms + column] * current_eigenvector[column];
                gram_product[row] = sum;
            }
            __syncthreads();

            // Deflation: gram_product -= Σ_{j<k} λ_j (v_j · gram_product) v_j. Since G is symmetric PSD,
            // removing projections onto the already-found eigenvectors lets power iteration converge onto
            // the next-largest eigenvalue.
            for (std::uint32_t previous_index = 0; previous_index < eigenvector_index; ++previous_index) {
                float const *previous_eigenvector = eigenvectors + previous_index * num_atoms;
                float local_projection = 0.0f;
                for (std::uint32_t atom = lane; atom < num_atoms; atom += block_size)
                    local_projection += previous_eigenvector[atom] * gram_product[atom];
                float const projection = embed_block_reduce_sum_(local_projection, warp_partials);
                for (std::uint32_t atom = lane; atom < num_atoms; atom += block_size)
                    gram_product[atom] -= projection * previous_eigenvector[atom];
                __syncthreads();
            }

            // λ = ‖gram_product‖
            float local_gram_product_square_sum = 0.0f;
            for (std::uint32_t atom = lane; atom < num_atoms; atom += block_size)
                local_gram_product_square_sum += gram_product[atom] * gram_product[atom];
            float const lambda = sqrtf(
                fmaxf(embed_block_reduce_sum_(local_gram_product_square_sum, warp_partials), 1e-20f));

            // Normalize and check convergence.
            for (std::uint32_t atom = lane; atom < num_atoms; atom += block_size)
                current_eigenvector[atom] = gram_product[atom] / lambda;
            __syncthreads();

            float const delta = fabsf(lambda - lambda_prev);
            if (delta < convergence_tolerance_k * fmaxf(lambda, 1e-6f)) {
                lambda_prev = lambda;
                break;
            }
            lambda_prev = lambda;
        }
        if (lane == 0) lambda_ptr[eigenvector_index] = lambda_prev;
        __syncthreads();
    }

    // Write the 4D metric coordinates.
    for (std::uint32_t atom = lane; atom < num_atoms; atom += block_size) {
        for (std::uint32_t dimension = 0; dimension < 4; ++dimension) {
            float const eigenvalue = fmaxf(lambda_ptr[dimension], 0.0f);
            float const value = (eigenvalue > 1e-8f) ? sqrtf(eigenvalue) * eigenvectors[dimension * num_atoms + atom]
                                                     : 0.0f;
            molecule_coordinates[dimension * max_atoms_per_molecule + atom] = value;
        }
    }
}

/**
 *  @brief Async-only metric prepass driver that takes a caller-owned solver handle +
 *  scratch pool.
 *
 *  @p scratch must be pre-sized for @p num_molecules × @p max_atoms_per_molecule by
 *  the caller (see `ensure_metric_scratch` in smallfield.cu). This function performs
 *  no device allocations and no CPU-side synchronization; all CUDA calls go on
 *  @p stream, so the caller can issue kernel launches against @p stream
 *  immediately afterward and the driver orders them correctly.
 */
template <typename scratch_>
inline bool build_metric_initial_coordinates_on_gpu(float *device_bounds,
                                                    etkdg_molecule_metadata_t const *device_metadata,
                                                    custd::span<etkdg_molecule_metadata_t const> host_metadata,
                                                    std::uint32_t max_atoms_per_molecule,
                                                    float *device_initial_coordinates, scratch_ &scratch,
                                                    cuda_error_sink_t sink, cudaStream_t stream = nullptr) {
    std::uint32_t const num_molecules = static_cast<std::uint32_t>(host_metadata.size());
    if (num_molecules == 0 || max_atoms_per_molecule == 0) return true;

    if (!sink.check(cudaMemsetAsync(
                        device_initial_coordinates, 0,
                        static_cast<std::size_t>(num_molecules) * max_atoms_per_molecule * 4 * sizeof(float), stream),
                    "cudaMemsetAsync(metric coordinates)"))
        return false;

    // Bin molecules into 8 fixed wave-dispatch buckets {16,32,48,64,80,96,112,128}.
    // Everything in a bin pads up to the bin's N for the eigendecomposition, so the
    // kernel launches ONCE per bin instead of once per exact atom count.
    auto const &bin_atom_counts_k = smallfield::types::atom_limited_bin_caps_k;
    constexpr std::uint32_t num_bins_k = static_cast<std::uint32_t>(smallfield::types::atom_limited_bin_count_k);

    auto bin_for = [&](std::uint32_t num_atoms) -> int {
        std::size_t const bin_index = smallfield::types::atom_limited_bin_for(num_atoms);
        return bin_index < smallfield::types::atom_limited_bin_count_k ? static_cast<int>(bin_index) : -1;
    };

    // Bins are reused across calls via a thread_local store; no per-call allocation.
    thread_local std::array<std::vector<std::uint32_t>, num_bins_k> bin_groups;
    for (auto &bin_group : bin_groups) bin_group.clear();
    for (std::uint32_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        std::uint32_t const num_atoms = host_metadata[molecule_index].num_atoms;
        if (num_atoms == 0 || num_atoms > max_atoms_per_molecule) continue;
        int const bin = bin_for(num_atoms);
        if (bin < 0) continue;
        bin_groups[bin].push_back(molecule_index);
    }

    bool success = true;

    // The `cudaFuncAttributeMaxDynamicSharedMemorySize` attribute for
    // `embed_4d_power_iter_kernel` is set once per device in
    // `context_t::context_t` at construction and must not be re-asserted here:
    // this function runs concurrently from OMP per-device worker threads, and
    // the attribute mutates driver-global per-kernel state on the same symbol.

    for (std::uint32_t bin = 0; bin < num_bins_k && success; ++bin) {
        auto const &group = bin_groups[bin];
        if (group.empty()) continue;

        std::uint32_t const bin_n = bin_atom_counts_k[bin];
        std::size_t const group_size = group.size();
        if (group_size > scratch.max_group_size || bin_n > scratch.max_atoms) {
            if (sink.log != nullptr)
                sink.log->error(
                    "metric_embedding_prepass: scratch too small (need group={} bin_n={}, have group={} atoms={})",
                    group_size, bin_n, scratch.max_group_size, scratch.max_atoms);
            return false;
        }

        std::uint32_t *const device_group_indices = scratch.group_indices.data();

        // Shared memory for embed_4d_power_iter_kernel:
        std::size_t const shared_bytes = embed_shared_bytes(bin_n);

        success &= sink.check(cudaMemcpyAsync(device_group_indices, group.data(), group_size * sizeof(std::uint32_t),
                                              cudaMemcpyHostToDevice, stream),
                              "cudaMemcpyAsync(bin group indices)");
        if (!success) continue;

        // One kernel launch performs the smooth-and-build-gram + eigendecomposition + project
        // chain entirely on-device: no host-side workspace, no global Gram/eigenvalue buffers.
        embed_4d_power_iter_kernel<<<static_cast<unsigned int>(group_size), embed_block_size_k, shared_bytes, stream>>>(
            device_bounds, device_metadata, device_group_indices, device_initial_coordinates, max_atoms_per_molecule);
        success &= sink.check(cudaGetLastError(), "embed_4d_power_iter_kernel launch");
    }

    // No final stream sync: the prepass is stream-ordered with whatever work the
    // caller launches next on the same stream. Caller's `finalize_gpu_pipeline`
    // syncs the stream once per wave, which is enough.
    return success;
}

} // namespace smallfield::cuda

#endif // __CUDACC__
