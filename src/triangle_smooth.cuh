/**
 * @file triangle_smooth.cuh
 * @brief Floyd-Warshall triangle inequality smoothing for half-precision bounds.
 *
 * Tightens the N x N bounds matrix by enforcing:
 *   upper: uᵢⱼ ← min(uᵢⱼ, uᵢₖ + uₖⱼ)   for all k
 *   lower: lᵢⱼ ← max(lᵢⱼ, lᵢₖ − uₖⱼ, lₖⱼ − uᵢₖ)  for all k
 *
 * Runs entirely within one warp (32 threads).
 * Bounds stored as __half to halve shared memory usage.
 *
 * Convention: bounds[i*N+j] with i < j = upper, i > j = lower, diagonal = 0.
 */

#pragma once

#if defined(__CUDACC__)

#include "lbfgs_warp.cuh"
#include <cuda_fp16.h>

namespace smallfield {

/** Read upper bound between atoms a and b (a ≠ b). */
__device__ __forceinline__ float read_upper_bound_(__half const* bounds, std::uint32_t num_atoms, std::uint32_t atom_a,
                                                   std::uint32_t atom_b) {
    std::uint32_t const row = min(atom_a, atom_b);
    std::uint32_t const col = max(atom_a, atom_b);
    return __half2float(bounds[row * num_atoms + col]);
}

/** Read lower bound between atoms a and b (a ≠ b). */
__device__ __forceinline__ float read_lower_bound_(__half const* bounds, std::uint32_t num_atoms, std::uint32_t atom_a,
                                                   std::uint32_t atom_b) {
    std::uint32_t const row = max(atom_a, atom_b);
    std::uint32_t const col = min(atom_a, atom_b);
    return __half2float(bounds[row * num_atoms + col]);
}

/**
 * In-place Floyd-Warshall triangle smoothing on half-precision bounds.
 * Operates in float arithmetic, writes results back as half.
 */
__device__ __forceinline__ void triangle_smooth_warp_half_(__half* bounds, std::uint32_t num_atoms,
                                                           std::uint32_t lane) {
    std::uint32_t const total_pairs = num_atoms * num_atoms;

    for (std::uint32_t k = 0; k < num_atoms; ++k) {
        for (std::uint32_t linear_index = lane; linear_index < total_pairs; linear_index += warp_size) {
            std::uint32_t const i = linear_index / num_atoms;
            std::uint32_t const j = linear_index % num_atoms;
            if (i == j || i == k || j == k)
                continue;

            float const upper_ik = read_upper_bound_(bounds, num_atoms, i, k);
            float const upper_kj = read_upper_bound_(bounds, num_atoms, k, j);

            if (i < j) {
                // upper bound: uᵢⱼ ← min(uᵢⱼ, uᵢₖ + uₖⱼ)
                float const current = __half2float(bounds[i * num_atoms + j]);
                float const candidate = upper_ik + upper_kj;
                if (candidate < current)
                    bounds[i * num_atoms + j] = __float2half(candidate);
            } else {
                // lower bound: lᵢⱼ ← max(lᵢⱼ, lᵢₖ − uₖⱼ, lₖⱼ − uᵢₖ)
                float const current = __half2float(bounds[i * num_atoms + j]);
                float const lower_ik = read_lower_bound_(bounds, num_atoms, i, k);
                float const lower_kj = read_lower_bound_(bounds, num_atoms, k, j);
                float const upper_ik_val = read_upper_bound_(bounds, num_atoms, i, k);

                float candidate = current;
                float const c1 = lower_ik - upper_kj;
                float const c2 = lower_kj - upper_ik_val;
                if (c1 > candidate)
                    candidate = c1;
                if (c2 > candidate)
                    candidate = c2;

                if (candidate > current)
                    bounds[i * num_atoms + j] = __float2half(candidate);
            }
        }
        __syncwarp(full_mask);
    }
}

/**
 * Float-precision version (kept for backward compatibility with tests).
 */
__device__ __forceinline__ void triangle_smooth_warp_(float* bounds, std::uint32_t num_atoms, std::uint32_t lane) {
    std::uint32_t const total_pairs = num_atoms * num_atoms;

    for (std::uint32_t k = 0; k < num_atoms; ++k) {
        for (std::uint32_t linear_index = lane; linear_index < total_pairs; linear_index += warp_size) {
            std::uint32_t const i = linear_index / num_atoms;
            std::uint32_t const j = linear_index % num_atoms;
            if (i == j || i == k || j == k)
                continue;

            auto read_upper = [&](std::uint32_t a, std::uint32_t b) -> float {
                return bounds[min(a, b) * num_atoms + max(a, b)];
            };
            auto read_lower = [&](std::uint32_t a, std::uint32_t b) -> float {
                return bounds[max(a, b) * num_atoms + min(a, b)];
            };

            float const upper_ik = read_upper(i, k);
            float const upper_kj = read_upper(k, j);

            if (i < j) {
                float const current = bounds[i * num_atoms + j];
                float const candidate = upper_ik + upper_kj;
                if (candidate < current)
                    bounds[i * num_atoms + j] = candidate;
            } else {
                float const current = bounds[i * num_atoms + j];
                float const lower_ik = read_lower(i, k);
                float const lower_kj = read_lower(k, j);
                float const upper_ik_val = read_upper(i, k);

                float candidate = current;
                float const c1 = lower_ik - upper_kj;
                float const c2 = lower_kj - upper_ik_val;
                if (c1 > candidate)
                    candidate = c1;
                if (c2 > candidate)
                    candidate = c2;
                if (candidate > current)
                    bounds[i * num_atoms + j] = candidate;
            }
        }
        __syncwarp(full_mask);
    }
}

/** Standalone kernel for testing float-precision smoothing. */
__global__ void triangle_smooth_kernel(float* bounds, std::uint32_t num_atoms) {
    triangle_smooth_warp_(bounds, num_atoms, threadIdx.x);
}

} // namespace smallfield

#endif // __CUDACC__
