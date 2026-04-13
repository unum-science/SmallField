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

#if defined(__CUDACC__)

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "etkdg_types.cuh"

namespace smallfield {

inline bool check_cuda_status_(cudaError_t status, char const *operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

inline bool check_cusolver_status_(cusolverStatus_t status, char const *operation) {
    if (status == CUSOLVER_STATUS_SUCCESS) return true;
    std::fprintf(stderr, "cuSOLVER error in %s: %d\n", operation, static_cast<int>(status));
    return false;
}

__device__ __forceinline__ float read_upper_bound_float_(float const *bounds, std::uint32_t num_atoms,
                                                         std::uint32_t atom_a, std::uint32_t atom_b) {
    std::uint32_t const row = min(atom_a, atom_b);
    std::uint32_t const col = max(atom_a, atom_b);
    return bounds[row * num_atoms + col];
}

__device__ __forceinline__ float read_lower_bound_float_(float const *bounds, std::uint32_t num_atoms,
                                                         std::uint32_t atom_a, std::uint32_t atom_b) {
    std::uint32_t const row = max(atom_a, atom_b);
    std::uint32_t const col = min(atom_a, atom_b);
    return bounds[row * num_atoms + col];
}

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
    for (std::uint32_t idx = lane; idx < total_entries; idx += blockDim.x) shared_bounds[idx] = global_bounds[idx];
    __syncthreads();

    // Floyd-Warshall on the real N×N block only.
    for (std::uint32_t pivot = 0; pivot < num_atoms; ++pivot) {
        for (std::uint32_t idx = lane; idx < total_entries; idx += blockDim.x) {
            std::uint32_t const i = idx / num_atoms;
            std::uint32_t const j = idx % num_atoms;
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
    for (std::uint32_t idx = lane; idx < total_entries; idx += blockDim.x) global_bounds[idx] = shared_bounds[idx];

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
    // outside.  The zero-pad rows/cols contribute zero eigenvalues that sort below the
    // real ones, so the top-4 eigenvectors reconstruct the same 4D embedding.
    for (std::uint32_t idx = lane; idx < gram_total; idx += blockDim.x) {
        std::uint32_t const i = idx / bin_n;
        std::uint32_t const j = idx % bin_n;
        float value = 0.0f;
        if (i < num_atoms && j < num_atoms) {
            float const squared_distance = midpoint_distance_squared_(shared_bounds, num_atoms, i, j);
            value = -0.5f * (squared_distance - row_means[i] - row_means[j] + grand_mean);
        }
        gram[j * bin_n + i] = value;
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
template <typename ScratchT>
inline bool build_metric_initial_coordinates_on_gpu(float *dev_bounds, etkdg_molecule_metadata_t const *dev_metadata,
                                                    std::span<etkdg_molecule_metadata_t const> host_metadata,
                                                    std::uint32_t max_atoms_per_molecule,
                                                    float *dev_initial_coordinates, cusolverDnHandle_t solver_handle,
                                                    ScratchT &scratch, cudaStream_t stream = nullptr) {
    std::uint32_t const num_molecules = static_cast<std::uint32_t>(host_metadata.size());
    if (num_molecules == 0 || max_atoms_per_molecule == 0) return true;

    if (!check_cuda_status_(
            cudaMemsetAsync(dev_initial_coordinates, 0,
                            static_cast<std::size_t>(num_molecules) * max_atoms_per_molecule * 4 * sizeof(float),
                            stream),
            "cudaMemsetAsync(metric coordinates)"))
        return false;

    // Bin molecules into 8 fixed wave-dispatch buckets {16,32,48,64,80,96,112,128}.
    // Everything in a bin padds up to the bin's N for the eigendecomposition, so
    // cusolver is called ONCE per bin instead of once per exact atom count (~30 calls
    // today).  Per-call CPU overhead inside cusolver dominated the per-launch wall.
    constexpr std::uint32_t bin_ns[] = {16, 32, 48, 64, 80, 96, 112, 128};
    constexpr std::uint32_t num_bins = sizeof(bin_ns) / sizeof(bin_ns[0]);

    auto bin_for = [&](std::uint32_t num_atoms) -> int {
        for (std::uint32_t b = 0; b < num_bins; ++b)
            if (num_atoms <= bin_ns[b]) return static_cast<int>(b);
        return -1;
    };

    // Bins are reused across calls via a thread_local store; no per-call allocation.
    thread_local std::array<std::vector<std::uint32_t>, num_bins> bin_groups;
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

    constexpr std::uint32_t block_size = 128;
    bool success = true;

    for (std::uint32_t b = 0; b < num_bins && success; ++b) {
        auto const &group = bin_groups[b];
        if (group.empty()) continue;

        std::uint32_t const bin_n = bin_ns[b];
        std::size_t const group_size = group.size();
        if (group_size > scratch.max_group_size || bin_n > scratch.max_atoms) {
            std::fprintf(
                stderr,
                "metric_embedding_prepass: scratch too small (need group=%zu bin_n=%u, have group=%zu atoms=%u)\n",
                group_size, bin_n, scratch.max_group_size, scratch.max_atoms);
            return false;
        }

        std::uint32_t *const dev_group_indices = scratch.group_indices.data();
        float *const dev_gram_matrices = scratch.gram_matrices.data();
        float *const dev_eigenvalues = scratch.eigenvalues.data();
        void *const dev_workspace = scratch.workspace.data();
        int *const dev_info = scratch.info.data();
        std::size_t const dev_ws_bytes = scratch.max_device_workspace_bytes;
        void *const host_workspace = scratch.host_workspace.data();
        std::size_t const host_ws_bytes = scratch.max_host_workspace_bytes;

        // Shared memory sized for the bin's worst-case N.
        std::size_t const shared_bytes = static_cast<std::size_t>(bin_n * bin_n + bin_n + 1) * sizeof(float);

        success &= check_cuda_status_(
            cudaMemcpyAsync(dev_group_indices, group.data(), group_size * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                            stream),
            "cudaMemcpyAsync(bin group indices)");
        success &= check_cuda_status_(
            cudaFuncSetAttribute(smooth_and_build_gram_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(shared_bytes)),
            "cudaFuncSetAttribute(smooth_and_build_gram_kernel)");
        if (!success) continue;

        smooth_and_build_gram_kernel<<<static_cast<unsigned int>(group_size), block_size, shared_bytes, stream>>>(
            dev_bounds, dev_metadata, dev_group_indices, dev_gram_matrices, bin_n);
        success &= check_cuda_status_(cudaGetLastError(), "smooth_and_build_gram_kernel launch");
        if (!success) continue;

        // Generic 64-bit batched eigen API.  Lower per-call CPU overhead than the
        // legacy cusolverDnSsyevjBatched — no Jacobi state bookkeeping.
        success &= check_cusolver_status_(
            cusolverDnXsyevBatched(solver_handle, scratch.xsyev_params, CUSOLVER_EIG_MODE_VECTOR,
                                   CUBLAS_FILL_MODE_UPPER, static_cast<int64_t>(bin_n), CUDA_R_32F, dev_gram_matrices,
                                   static_cast<int64_t>(bin_n), CUDA_R_32F, dev_eigenvalues, CUDA_R_32F, dev_workspace,
                                   dev_ws_bytes, host_workspace, host_ws_bytes, dev_info,
                                   static_cast<int64_t>(group_size)),
            "cusolverDnXsyevBatched");
        if (!success) continue;

        project_metric_coordinates_kernel<<<static_cast<unsigned int>(group_size), block_size, 0, stream>>>(
            dev_gram_matrices, dev_eigenvalues, dev_metadata, dev_group_indices, dev_initial_coordinates,
            max_atoms_per_molecule, bin_n);
        success &= check_cuda_status_(cudaGetLastError(), "project_metric_coordinates_kernel launch");
    }

    // Solver handle is owned by the caller (context_t) — do NOT destroy it here.
    // No final stream sync: the prepass is stream-ordered with whatever work the
    // caller launches next on the same stream.  Caller's `finalize_gpu_pipeline`
    // syncs the stream once per chunk, which is enough.
    return success;
}

} // namespace smallfield

#endif // __CUDACC__
