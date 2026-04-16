/**
 *  @file metric_embedding_prepass.cuh
 *  @brief GPU prepass for bounds smoothing and native 4D metric embedding.
 *
 *  Implements the first stage of ETKDG conformer generation on the GPU:
 *  1. Triangle-inequality bounds smoothing (Floyd-Warshall style) on the
 *     upper/lower distance bounds matrix.
 *  2. Gram matrix construction from midpoint squared distances with
 *     double-centering.
 *  3. Batched eigendecomposition via cuSOLVER (syevjBatched) to extract
 *     the top-4 eigenvectors.
 *  4. Projection into 4D metric coordinates as the initial embedding
 *     for subsequent ETKDG refinement stages.
 *
 *  Molecules are grouped by atom count so that cuSOLVER batched calls
 *  operate on identically-sized matrices for maximum throughput.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "smallfield/api/smallfield.hpp" // for cuda_error_counter
#include "smallfield/types/etkdg_types.hpp"

namespace smallfield::cuda {
using smallfield::api::cuda_error_counter;
using smallfield::types::etkdg_molecule_metadata_t;

/** @brief Log and count a CUDA runtime error if status is not cudaSuccess. */
inline bool check_cuda_status_(cudaError_t status, char const *operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
    smallfield::api::cuda_error_counter.fetch_add(1, std::memory_order_relaxed);
    return false;
}

/** @brief Log and count a cuSOLVER error if status is not CUSOLVER_STATUS_SUCCESS. */
inline bool check_cusolver_status_(cusolverStatus_t status, char const *operation) {
    if (status == CUSOLVER_STATUS_SUCCESS) return true;
    std::fprintf(stderr, "cuSOLVER error in %s: %d\n", operation, static_cast<int>(status));
    smallfield::api::cuda_error_counter.fetch_add(1, std::memory_order_relaxed);
    return false;
}

/**
 * @brief Reads the upper distance bound for atom pair (i, j) from the float bounds matrix.
 *
 * The upper bound is stored at bounds[min(i, j) * N + max(i, j)].
 *
 * @param[in] bounds Flat N x N bounds matrix in row-major order.
 * @param[in] num_atoms Number of atoms N in the molecule.
 * @param[in] atom_a First atom index.
 * @param[in] atom_b Second atom index.
 * @return Upper distance bound between atom_a and atom_b.
 */
__device__ __forceinline__ float read_upper_bound_float_(float const *bounds, std::uint32_t num_atoms,
                                                         std::uint32_t atom_a, std::uint32_t atom_b) {
    std::uint32_t const row = min(atom_a, atom_b);
    std::uint32_t const col = max(atom_a, atom_b);
    return bounds[row * num_atoms + col];
}

/**
 * @brief Reads the lower distance bound for atom pair (i, j) from the float bounds matrix.
 *
 * The lower bound is stored at bounds[max(i, j) * N + min(i, j)].
 *
 * @param[in] bounds Flat N x N bounds matrix in row-major order.
 * @param[in] num_atoms Number of atoms N in the molecule.
 * @param[in] atom_a First atom index.
 * @param[in] atom_b Second atom index.
 * @return Lower distance bound between atom_a and atom_b.
 */
__device__ __forceinline__ float read_lower_bound_float_(float const *bounds, std::uint32_t num_atoms,
                                                         std::uint32_t atom_a, std::uint32_t atom_b) {
    std::uint32_t const row = max(atom_a, atom_b);
    std::uint32_t const col = min(atom_a, atom_b);
    return bounds[row * num_atoms + col];
}

/**
 * @brief Returns ((upper + lower) / 2)^2 for Gram-matrix double-centering.
 *
 * Computes the squared midpoint distance between two atoms from their upper and lower bounds.
 * Returns zero when atom_i equals atom_j.
 *
 * @param[in] bounds Flat N x N bounds matrix in row-major order.
 * @param[in] num_atoms Number of atoms N in the molecule.
 * @param[in] atom_i First atom index.
 * @param[in] atom_j Second atom index.
 * @return Squared midpoint distance between atom_i and atom_j.
 * @see smooth_and_build_gram_kernel
 */
__device__ __forceinline__ float midpoint_distance_squared_(float const *bounds, std::uint32_t num_atoms,
                                                            std::uint32_t atom_i, std::uint32_t atom_j) {
    if (atom_i == atom_j) return 0.0f;
    float const upper = read_upper_bound_float_(bounds, num_atoms, atom_i, atom_j);
    float const lower = read_lower_bound_float_(bounds, num_atoms, atom_i, atom_j);
    float const distance = 0.5f * (fmaxf(lower, 0.0f) + fmaxf(upper, lower));
    return distance * distance;
}

/**
 *  @brief Triangle-smooth the per-molecule bounds, build a zero-padded Gram
 *  matrix at a uniform @p bin_n × @p bin_n stride, and write it to @p gram_matrices.
 *
 *  The actual per-molecule size is @c meta.num_atoms, always ≤ @p bin_n.  The
 *  kernel smooths only the real N×N sub-block, zeros the Gram padding region,
 *  and writes into the bin-sized slot.  cusolverDnSsyevjBatched can then
 *  operate on a single uniform N across all molecules in the bin.
 *
 *  @see embed_4d_power_iter_kernel
 */
__global__ void smooth_and_build_gram_kernel(float *__restrict__ bounds_matrices,
                                             etkdg_molecule_metadata_t const *__restrict__ molecule_metadata,
                                             std::uint32_t const *__restrict__ molecule_indices,
                                             float *__restrict__ gram_matrices, std::uint32_t bin_n) {
    std::uint32_t const group_index = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const molecule_index = molecule_indices[group_index];
    auto const meta = molecule_metadata[molecule_index];
    std::uint32_t const num_atoms = meta.num_atoms;

    // Shared layout: shared_bounds[num_atoms × num_atoms] + row_means[num_atoms] + grand_mean.
    // Sized by the host launcher for bin_n (the worst case in this bin).
    extern __shared__ float gram_shared_raw[];
    float *shared_bounds = gram_shared_raw;
    float *row_means = shared_bounds + num_atoms * num_atoms;
    float *grand_mean_ptr = row_means + num_atoms;

    float *global_bounds = bounds_matrices + meta.bounds_offset;
    float *gram = gram_matrices + group_index * bin_n * bin_n;
    std::uint32_t const total_entries = num_atoms * num_atoms;
    std::uint32_t const gram_total = bin_n * bin_n;

    // Load per-molecule bounds (at actual stride num_atoms) into shared memory.
    for (std::uint32_t linear_index = lane; linear_index < total_entries; linear_index += blockDim.x)
        shared_bounds[linear_index] = global_bounds[linear_index];
    __syncthreads();

    // Floyd-Warshall on the real N×N block only.
    for (std::uint32_t pivot = 0; pivot < num_atoms; ++pivot) {
        for (std::uint32_t linear_index = lane; linear_index < total_entries; linear_index += blockDim.x) {
            std::uint32_t const i = linear_index / num_atoms;
            std::uint32_t const j = linear_index % num_atoms;
            if (i == j || i == pivot || j == pivot) continue;

            float const upper_ip = read_upper_bound_float_(shared_bounds, num_atoms, i, pivot);
            float const upper_pj = read_upper_bound_float_(shared_bounds, num_atoms, pivot, j);

            if (i < j) {
                float const candidate = upper_ip + upper_pj;
                float &current = shared_bounds[i * num_atoms + j];
                if (candidate < current) current = candidate;
            }
            else {
                float const lower_ip = read_lower_bound_float_(shared_bounds, num_atoms, i, pivot);
                float const lower_pj = read_lower_bound_float_(shared_bounds, num_atoms, pivot, j);
                float &current = shared_bounds[i * num_atoms + j];
                current = fmaxf(current, fmaxf(lower_ip - upper_pj, lower_pj - upper_ip));
            }
        }
        __syncthreads();
    }

    // Persist the smoothed bounds back to global memory (at the molecule's own stride).
    for (std::uint32_t linear_index = lane; linear_index < total_entries; linear_index += blockDim.x)
        global_bounds[linear_index] = shared_bounds[linear_index];

    for (std::uint32_t atom = lane; atom < num_atoms; atom += blockDim.x) {
        float row_sum = 0.0f;
        for (std::uint32_t other = 0; other < num_atoms; ++other)
            row_sum += midpoint_distance_squared_(shared_bounds, num_atoms, atom, other);
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
    // Fill the whole bin-sized Gram block: real values inside [0, num_atoms)², zeros
    // outside. The zero-pad rows/cols contribute zero eigenvalues that sort below the
    // real ones, so the top-4 eigenvectors reconstruct the same 4D embedding.
    for (std::uint32_t linear_index = lane; linear_index < gram_total; linear_index += blockDim.x) {
        std::uint32_t const i = linear_index / bin_n;
        std::uint32_t const j = linear_index % bin_n;
        float value = 0.0f;
        if (i < num_atoms && j < num_atoms) {
            float const squared_distance = midpoint_distance_squared_(shared_bounds, num_atoms, i, j);
            value = -0.5f * (squared_distance - row_means[i] - row_means[j] + grand_mean);
        }
        gram[j * bin_n + i] = value;
    }
}

/**
 *  @brief Fused bounds-smooth → Gram → top-4 eigen → 4D embedding kernel.
 *
 *  One block per molecule.  Reuses the existing triangle-smooth + Gram build
 *  code, then runs deflated power iteration to extract the four largest
 *  eigenvalues + eigenvectors, and writes the 4D embedding directly into
 *  @c initial_coordinates.  Replaces the three-kernel chain
 *  (smooth_and_build_gram_kernel → cusolverDnXsyevBatched →
 *  project_metric_coordinates_kernel) with a single launch — no cusolver,
 *  no host workspace, no inter-kernel Gram-matrix global memory traffic.
 *
 *  Shared memory layout (all floats):
 *    gram[N × N]                (overwrites bounds buffer after smoothing)
 *    row_means[N]               (transient during Gram build)
 *    grand_mean[1]              (transient)
 *    v[4][N]                    (eigenvector estimates)
 *    w[N]                       (mat-vec scratch)
 *    lambda[4]                  (eigenvalue estimates)
 *    dot[1]                     (reduction scratch)
 *  At N=128: 16384 + 128 + 1 + 512 + 128 + 4 + 1 = 17158 floats ≈ 67 KB.
 *
 *  @see smooth_and_build_gram_kernel
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
    float *shared_gram = embed_shared_raw; // N × N, reused for bounds smoothing
    float *row_means = shared_gram + num_atoms * num_atoms;
    float *grand_mean_ptr = row_means + num_atoms; // 1
    float *v0 = grand_mean_ptr + 1;                // v[0]  ... v[3]
    float *v1 = v0 + num_atoms;
    float *v2 = v1 + num_atoms;
    float *v3 = v2 + num_atoms;
    float *w = v3 + num_atoms;
    float *lambda_ptr = w + num_atoms;       // 4
    float *dot_accumulator = lambda_ptr + 4; // 1 (reduction scratch)

    float *global_bounds = bounds_matrices + meta.bounds_offset;
    float *molecule_coordinates = initial_coordinates + molecule_index * max_atoms_per_molecule * 4;
    std::uint32_t const total_entries = num_atoms * num_atoms;

    // Load bounds into shared memory; shared_gram doubles as the bounds scratch.
    for (std::uint32_t linear_index = lane; linear_index < total_entries; linear_index += block_size)
        shared_gram[linear_index] = global_bounds[linear_index];
    __syncthreads();

    // Floyd-Warshall triangle smoothing on the N × N block.
    for (std::uint32_t pivot = 0; pivot < num_atoms; ++pivot) {
        for (std::uint32_t linear_index = lane; linear_index < total_entries; linear_index += block_size) {
            std::uint32_t const i = linear_index / num_atoms;
            std::uint32_t const j = linear_index % num_atoms;
            if (i == j || i == pivot || j == pivot) continue;
            float const upper_ip = read_upper_bound_float_(shared_gram, num_atoms, i, pivot);
            float const upper_pj = read_upper_bound_float_(shared_gram, num_atoms, pivot, j);
            if (i < j) {
                float const candidate = upper_ip + upper_pj;
                float &current = shared_gram[i * num_atoms + j];
                if (candidate < current) current = candidate;
            }
            else {
                float const lower_ip = read_lower_bound_float_(shared_gram, num_atoms, i, pivot);
                float const lower_pj = read_lower_bound_float_(shared_gram, num_atoms, pivot, j);
                float &current = shared_gram[i * num_atoms + j];
                current = fmaxf(current, fmaxf(lower_ip - upper_pj, lower_pj - upper_ip));
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
            row_sum += midpoint_distance_squared_(shared_gram, num_atoms, atom, other);
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

    // Build the Gram matrix in place, overwriting the smoothed bounds. Every thread computes
    // its Gram entries into registers first, then writes them back after a full-block barrier;
    // midpoint_distance_squared_ reads both upper and lower bound entries, so all reads must complete
    // before any writes. Each thread holds at most ⌈N²/block_size⌉ entries — at N=128 with 128 threads
    // that's 128 floats or 512 B per thread, comfortable in local memory.
    constexpr std::uint32_t max_entries_per_thread_k = 128; // supports up to N=128 at 128 threads
    float register_buffer[max_entries_per_thread_k];
    std::uint32_t entries_per_thread = 0;
    for (std::uint32_t linear_index = lane; linear_index < total_entries; linear_index += block_size) {
        std::uint32_t const i = linear_index / num_atoms;
        std::uint32_t const j = linear_index % num_atoms;
        float const sq = midpoint_distance_squared_(shared_gram, num_atoms, i, j);
        register_buffer[entries_per_thread++] = -0.5f * (sq - row_means[i] - row_means[j] + grand_mean);
    }
    __syncthreads();
    entries_per_thread = 0;
    for (std::uint32_t linear_index = lane; linear_index < total_entries; linear_index += block_size)
        shared_gram[linear_index] = register_buffer[entries_per_thread++];
    __syncthreads();

    // Deflated power iteration for the top-4 eigenvectors.
    // For each k in 0..3, iterate v_k ← (G - Σ_{j<k} λ_j v_j v_j^T) v_k / ‖·‖.
    // With positive-semidefinite Gram matrices the top eigenvalues are all
    // non-negative and power iteration converges.
    float *eigenvector_ptrs[4] = {v0, v1, v2, v3};
    constexpr std::uint32_t max_power_iterations_k = 60;
    constexpr float convergence_tolerance_k = 1e-4f;
    for (std::uint32_t k = 0; k < 4; ++k) {
        float *eigenvector_k = eigenvector_ptrs[k];
        // Initialize with a non-trivial vector orthogonal to e_0..e_{k-1} for better
        // deflation start (simple: e_k plus a small spread).
        for (std::uint32_t i = lane; i < num_atoms; i += block_size)
            eigenvector_k[i] = (i == k) ? 1.0f : 0.5f / static_cast<float>(num_atoms);
        __syncthreads();
        // Normalize initial eigenvector_k to unit length.
        if (lane == 0) dot_accumulator[0] = 0.0f;
        __syncthreads();
        float local_sq = 0.0f;
        for (std::uint32_t i = lane; i < num_atoms; i += block_size) local_sq += eigenvector_k[i] * eigenvector_k[i];
        atomicAdd(dot_accumulator, local_sq);
        __syncthreads();
        float eigenvector_k_norm = sqrtf(fmaxf(dot_accumulator[0], 1e-20f));
        for (std::uint32_t i = lane; i < num_atoms; i += block_size) eigenvector_k[i] /= eigenvector_k_norm;
        __syncthreads();

        float lambda_prev = 0.0f;
        for (std::uint32_t iter = 0; iter < max_power_iterations_k; ++iter) {
            // w = G · eigenvector_k
            for (std::uint32_t i = lane; i < num_atoms; i += block_size) {
                float sum = 0.0f;
                for (std::uint32_t j = 0; j < num_atoms; ++j) sum += shared_gram[i * num_atoms + j] * eigenvector_k[j];
                w[i] = sum;
            }
            __syncthreads();

            // Deflation: w -= Σ_{j<k} λ_j (v_j · w) v_j  (since G is symmetric PSD, removing
            // projections onto previously-found eigenvectors lets power iteration converge
            // onto the next-largest eigenvalue).
            for (std::uint32_t j = 0; j < k; ++j) {
                if (lane == 0) dot_accumulator[0] = 0.0f;
                __syncthreads();
                float local = 0.0f;
                for (std::uint32_t i = lane; i < num_atoms; i += block_size) local += eigenvector_ptrs[j][i] * w[i];
                atomicAdd(dot_accumulator, local);
                __syncthreads();
                float const proj = dot_accumulator[0];
                for (std::uint32_t i = lane; i < num_atoms; i += block_size) w[i] -= proj * eigenvector_ptrs[j][i];
                __syncthreads();
            }

            // λ = ‖w‖
            if (lane == 0) dot_accumulator[0] = 0.0f;
            __syncthreads();
            float local_w = 0.0f;
            for (std::uint32_t i = lane; i < num_atoms; i += block_size) local_w += w[i] * w[i];
            atomicAdd(dot_accumulator, local_w);
            __syncthreads();
            float const lambda = sqrtf(fmaxf(dot_accumulator[0], 1e-20f));

            // Normalize and check convergence.
            for (std::uint32_t i = lane; i < num_atoms; i += block_size) eigenvector_k[i] = w[i] / lambda;
            __syncthreads();

            float const delta = fabsf(lambda - lambda_prev);
            if (delta < convergence_tolerance_k * fmaxf(lambda, 1e-6f)) {
                lambda_prev = lambda;
                break;
            }
            lambda_prev = lambda;
        }
        if (lane == 0) lambda_ptr[k] = lambda_prev;
        __syncthreads();
    }

    // Write the 4D metric coordinates.
    for (std::uint32_t atom = lane; atom < num_atoms; atom += block_size) {
        for (std::uint32_t dim = 0; dim < 4; ++dim) {
            float const lam = fmaxf(lambda_ptr[dim], 0.0f);
            float const value = (lam > 1e-8f) ? sqrtf(lam) * eigenvector_ptrs[dim][atom] : 0.0f;
            molecule_coordinates[dim * max_atoms_per_molecule + atom] = value;
        }
    }
}

/**
 *  @brief Project the top-4 eigenvectors into 4D coordinates.
 *
 *  The eigenproblem was solved at a uniform @p bin_n; each molecule's actual size is
 *  @c meta.num_atoms ≤ @p bin_n.  Eigenvectors are stored with stride @p bin_n; we read
 *  only the first @c meta.num_atoms components.  The last 4 eigenvalues (largest under
 *  cusolver's ascending sort) carry the real 4D embedding — zero-padded rows produced
 *  trailing zero eigenvalues that sort below.
 */
__global__ void project_metric_coordinates_kernel(float const *__restrict__ eigenvectors,
                                                  float const *__restrict__ eigenvalues,
                                                  etkdg_molecule_metadata_t const *__restrict__ molecule_metadata,
                                                  std::uint32_t const *__restrict__ molecule_indices,
                                                  float *__restrict__ initial_coordinates,
                                                  std::uint32_t max_atoms_per_molecule, std::uint32_t bin_n) {
    std::uint32_t const group_index = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const molecule_index = molecule_indices[group_index];
    std::uint32_t const num_atoms = molecule_metadata[molecule_index].num_atoms;

    float const *molecule_eigenvectors = eigenvectors + group_index * bin_n * bin_n;
    float const *molecule_eigenvalues = eigenvalues + group_index * bin_n;
    float *molecule_coordinates = initial_coordinates + molecule_index * max_atoms_per_molecule * 4;

    for (std::uint32_t atom = lane; atom < num_atoms; atom += blockDim.x) {
        for (std::uint32_t dim = 0; dim < 4; ++dim) {
            std::int32_t const eigen_index = static_cast<std::int32_t>(bin_n) - 1 - static_cast<std::int32_t>(dim);
            float value = 0.0f;
            if (eigen_index >= 0) {
                float const eigenvalue = fmaxf(molecule_eigenvalues[eigen_index], 0.0f);
                if (eigenvalue > 1.0e-8f) value = sqrtf(eigenvalue) * molecule_eigenvectors[atom + eigen_index * bin_n];
            }
            molecule_coordinates[dim * max_atoms_per_molecule + atom] = value;
        }
    }
}

/**
 *  @brief Async-only metric prepass driver that takes a caller-owned solver handle +
 *  scratch pool.
 *
 *  @p scratch must be pre-sized for @p num_molecules × @p max_atoms_per_molecule by
 *  the caller (see `ensure_metric_scratch` in smallfield.cu).  This function performs
 *  no device allocations and no CPU-side synchronization; all CUDA/cusolver calls go
 *  on @p stream, so the caller can issue kernel launches against @p stream
 *  immediately afterward and the driver orders them correctly.
 *
 *  The previous entry point allocated scratch per atom-count group via cudaMalloc and
 *  queried cusolver workspace size per group, both of which are synchronous device
 *  operations that force the CPU to wait.  At 30-ish atom-count groups per chunk ×
 *  8 GPUs × N chunks, that overhead adds up to ~800 ms per chunk per GPU and was the
 *  reason the pipelined dispatcher needed `std::thread` to parallelize across GPUs.
 *  With pre-allocation this function is truly non-blocking.
 */
template <typename scratch_>
inline bool build_metric_initial_coordinates_on_gpu(float *device_bounds,
                                                    etkdg_molecule_metadata_t const *device_metadata,
                                                    std::span<etkdg_molecule_metadata_t const> host_metadata,
                                                    std::uint32_t max_atoms_per_molecule,
                                                    float *device_initial_coordinates, cusolverDnHandle_t solver_handle,
                                                    scratch_ &scratch, cudaStream_t stream = nullptr) {
    std::uint32_t const num_molecules = static_cast<std::uint32_t>(host_metadata.size());
    if (num_molecules == 0 || max_atoms_per_molecule == 0) return true;

    if (!check_cuda_status_(
            cudaMemsetAsync(device_initial_coordinates, 0,
                            static_cast<std::size_t>(num_molecules) * max_atoms_per_molecule * 4 * sizeof(float),
                            stream),
            "cudaMemsetAsync(metric coordinates)"))
        return false;

    // Bin molecules into 8 fixed wave-dispatch buckets {16,32,48,64,80,96,112,128}.
    // Everything in a bin padds up to the bin's N for the eigendecomposition, so
    // cusolver is called ONCE per bin instead of once per exact atom count (~30 calls
    // today). Per-call CPU overhead inside cusolver dominated the per-launch wall.
    constexpr std::uint32_t bin_atom_counts_k[] = {16, 32, 48, 64, 80, 96, 112, 128};
    constexpr std::uint32_t num_bins_k = sizeof(bin_atom_counts_k) / sizeof(bin_atom_counts_k[0]);

    auto bin_for = [&](std::uint32_t num_atoms) -> int {
        for (std::uint32_t b = 0; b < num_bins_k; ++b)
            if (num_atoms <= bin_atom_counts_k[b]) return static_cast<int>(b);
        return -1;
    };

    // Bins are reused across calls via a thread_local store; no per-call allocation.
    thread_local std::array<std::vector<std::uint32_t>, num_bins_k> bin_groups;
    for (auto &g : bin_groups) g.clear();
    for (std::uint32_t mol = 0; mol < num_molecules; ++mol) {
        std::uint32_t const num_atoms = host_metadata[mol].num_atoms;
        if (num_atoms == 0 || num_atoms > max_atoms_per_molecule) continue;
        int const bin = bin_for(num_atoms);
        if (bin < 0) continue;
        bin_groups[bin].push_back(mol);
    }

    // The caller already called cusolverDnSetStream(solver_handle, stream) at context
    // creation, but the stream may have been re-set since; cheap to re-set here.
    if (!check_cusolver_status_(cusolverDnSetStream(solver_handle, stream), "cusolverDnSetStream")) return false;

    constexpr std::uint32_t block_size_k = 128;
    bool success = true;

    for (std::uint32_t b = 0; b < num_bins_k && success; ++b) {
        auto const &group = bin_groups[b];
        if (group.empty()) continue;

        std::uint32_t const bin_n = bin_atom_counts_k[b];
        std::size_t const group_size = group.size();
        if (group_size > scratch.max_group_size || bin_n > scratch.max_atoms) {
            std::fprintf(
                stderr,
                "metric_embedding_prepass: scratch too small (need group=%zu bin_n=%u, have group=%zu atoms=%u)\n",
                group_size, bin_n, scratch.max_group_size, scratch.max_atoms);
            return false;
        }

        std::uint32_t *const device_group_indices = scratch.group_indices.data();

        // Shared memory for embed_4d_power_iter_kernel:
        //   gram[N²] + row_means[N] + grand_mean[1] + v[4][N] + w[N] + lambda[4] + dot[1]
        std::size_t const shared_bytes =
            static_cast<std::size_t>(bin_n * bin_n + bin_n + 1 + 4 * bin_n + bin_n + 4 + 1) * sizeof(float);

        success &= check_cuda_status_(
            cudaMemcpyAsync(device_group_indices, group.data(), group_size * sizeof(std::uint32_t),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync(bin group indices)");
        success &= check_cuda_status_(
            cudaFuncSetAttribute(embed_4d_power_iter_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(shared_bytes)),
            "cudaFuncSetAttribute(embed_4d_power_iter_kernel)");
        if (!success) continue;

        // One kernel launch replaces the smooth-and-build-gram + cusolverDnXsyevBatched +
        // project chain. No cusolver state updates, no global Gram/eigenvalue buffers.
        embed_4d_power_iter_kernel<<<static_cast<unsigned int>(group_size), block_size_k, shared_bytes, stream>>>(
            device_bounds, device_metadata, device_group_indices, device_initial_coordinates, max_atoms_per_molecule);
        success &= check_cuda_status_(cudaGetLastError(), "embed_4d_power_iter_kernel launch");
    }

    // The solver handle is no longer used inside this function but we keep the
    // parameter for API compatibility; caller still owns the handle.
    (void)solver_handle;

    // Solver handle is owned by the caller (context_t) — do NOT destroy it here.
    // No final stream sync: the prepass is stream-ordered with whatever work the
    // caller launches next on the same stream. Caller's `finalize_gpu_pipeline`
    // syncs the stream once per chunk, which is enough.
    return success;
}

} // namespace smallfield::cuda

#endif // __CUDACC__
