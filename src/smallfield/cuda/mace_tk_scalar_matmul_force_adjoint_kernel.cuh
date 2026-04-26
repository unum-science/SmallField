/**
 *  @file mace_tk_scalar_matmul_force_adjoint_kernel.cuh
 *  @brief Backward sibling of `mace_tk_scalar_matmul_kernel` — the
 *         L=0-only per-irrep linear force adjoint on tensor cores.
 *
 *  Forward (per `mace_tk_scalar_matmul_kernel`):
 *      output[a, c_out] = (1 / sqrt(c_in)) * Σ_{c_in} weight[c_in, c_out] * input[a, c_in]
 *
 *  Adjoint (this file):
 *      grad_input[a, c_in] += (1 / sqrt(c_in)) * Σ_{c_out} weight[c_in, c_out] * upstream[a, c_out]
 *
 *  Replaces the brute-force `mace_linear_per_irrep_irrep1_force_adjoint_kernel`
 *  (one thread per (atom, c_in), serialised inner reduction over c_out) — NCU
 *  measured that fp32 reference at 5.72 % of total kernel time on the post-
 *  multi-warp baseline, the largest remaining fp32 fallback in the chain.
 *
 *  Same per-atom WGMMA tile shape as the forward (M = 64 atoms, K = c_out,
 *  N = c_in for the small model, both = 96).  Uses `mm_ABt` because the
 *  weight blob is stored as `[c_in, c_out]` row-major: viewing it as B,
 *  TK's mm_ABt computes `D[M, N] = A[M, K] @ B[N, K]^T`, so the (c_in, c_out)
 *  axes of `weight` line up correctly without an explicit transpose.
 *
 *  The += accumulation semantics of the fp32 reference are preserved: the
 *  kernel reads `grad_input[a, c_in]` at the store epilogue, adds the scaled
 *  register accumulator, and writes back.  Caller doesn't need to pre-zero.
 *
 *  Per-CTA shared budget:
 *      upstream tile      : 64 × 96 × 2 = 12 KB
 *      weight tile        : 96 × 96 × 2 = 18 KB
 *      delta staging fp32 : 64 × 96 × 4 = 24 KB
 *      total                              ≈ 54 KB     (fits default 100 KB)
 *
 *  Compile-guarded on `SMALLFIELD_USE_THUNDERKITTENS`.
 */

#pragma once

#include "smallfield/cuda/mace_tk_tile.cuh"

#if defined(SMALLFIELD_USE_THUNDERKITTENS) && defined(__CUDACC__)

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace smallfield::cuda {

namespace mace_tk_scalar_matmul_force_adjoint_detail_ {

constexpr int channels_k               = mace_tk::mace_channels_small_k;
constexpr int wgmma_m_rows_k           = 64;
constexpr int warpgroup_thread_count_k = kittens::WARPGROUP_THREADS;

using shared_upstream_tile_t = mace_tk::shared_tile_bf16<wgmma_m_rows_k, channels_k>;   // [64, c_out]
using shared_weight_tile_t   = mace_tk::shared_tile_bf16<channels_k, channels_k>;        // [c_in, c_out]
using shared_delta_tile_t    = mace_tk::shared_tile_fp32<wgmma_m_rows_k, channels_k>;    // [64, c_in] fp32 staging

using register_accumulator_t = kittens::rt_fl<16, channels_k>;                            // [64, c_in] across warpgroup

/// Stage 64 atoms' upstream gradient `[c_out]` into bf16 shared.
__device__ __forceinline__ void load_upstream_slab_(
    shared_upstream_tile_t &dst_tile,
    float const *upstream_global,
    std::uint32_t n_atoms,
    std::uint32_t base_atom_index
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = wgmma_m_rows_k * channels_k;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const atom_in_cta = linear_index / channels_k;
        int const c_out       = linear_index % channels_k;
        std::uint32_t const atom_index = base_atom_index + static_cast<std::uint32_t>(atom_in_cta);
        float value = 0.0f;
        if (atom_index < n_atoms) {
            value = upstream_global[static_cast<std::uint64_t>(atom_index) * channels_k + c_out];
        }
        dst_tile[{atom_in_cta, c_out}] = __float2bfloat16_rn(value);
    }
}

/// Stage `[c_in, c_out]` weight slab fp32 → bf16 in shared memory.  Same as
/// forward's helper; duplicated here so this header has no cross-file
/// dependency on the forward TU.
__device__ __forceinline__ void load_weight_slab_(
    shared_weight_tile_t &dst_tile,
    float const *weight_global
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = channels_k * channels_k;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const c_in_row  = linear_index / channels_k;
        int const c_out_col = linear_index % channels_k;
        float const value = weight_global[
            static_cast<std::uint64_t>(c_in_row) * channels_k + c_out_col];
        dst_tile[{c_in_row, c_out_col}] = __float2bfloat16_rn(value);
    }
}

/// Apply `1 / sqrt(c_in)` (× optional pre-scale) to every fp32 register
/// accumulator lane in place.
__device__ __forceinline__ void scale_accumulator_in_place_(
    register_accumulator_t &accumulator,
    float scaled_path_weight
) {
    kittens::warpgroup::apply(
        accumulator,
        accumulator,
        [scaled_path_weight] __device__ (int /*row*/, int /*col*/, float pre) {
            return pre * scaled_path_weight;
        });
}

/// Read existing `grad_input[a, c_in]` from global, add the scaled register
/// accumulator's value, and write back — mirrors the `+=` semantics of the
/// fp32 reference adjoint kernel.  We stage the register tile through a fp32
/// shared tile first (cooperative fp32 store), then per-thread RMW global.
__device__ __forceinline__ void accumulate_into_grad_input_(
    float *grad_input_global,
    shared_delta_tile_t &delta_staging,
    register_accumulator_t const &accumulator,
    std::uint32_t n_atoms,
    std::uint32_t base_atom_index
) {
    kittens::warpgroup::store(delta_staging, accumulator);
    kittens::warpgroup::sync(0);

    int const thread_id = threadIdx.x;
    constexpr int tile_elements = wgmma_m_rows_k * channels_k;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const atom_in_cta = linear_index / channels_k;
        int const c_in        = linear_index % channels_k;
        std::uint32_t const atom_index = base_atom_index + static_cast<std::uint32_t>(atom_in_cta);
        if (atom_index >= n_atoms) continue;
        std::uint64_t const offset = static_cast<std::uint64_t>(atom_index) * channels_k + c_in;
        float const existing = grad_input_global[offset];
        float const delta    = delta_staging[{atom_in_cta, c_in}];
        grad_input_global[offset] = existing + delta;
    }
}

/**
 *  @brief Persistent backward kernel body.
 *
 *  Grid: one CTA per 64-atom tile.  Block: 128 threads (one warpgroup).
 *  Per CTA:
 *    - load 64 atoms' upstream gradient `[c_out]` slab,
 *    - load `[c_in, c_out]` weight (shared with the forward call),
 *    - one WGMMA `mm_ABt` produces `[64, c_in]` register accumulator,
 *    - scale by path-weight, accumulate (+= ) into `grad_input` global.
 */
__global__ void mace_tk_scalar_matmul_force_adjoint_kernel(
    std::uint32_t n_atoms,
    float const *__restrict__ upstream_grad_global,    // [n_atoms, c_out]
    float const *__restrict__ weight_global,           // [c_in, c_out]
    float scaled_path_weight,                           // 1 / sqrt(c_in) [× pre_scale]
    float *__restrict__ grad_input_global               // [n_atoms, c_in] (RMW: += scaled @)
) {
    extern __shared__ kittens::alignment_dummy raw_shared_memory[];
    kittens::shared_allocator shared_alloc(reinterpret_cast<int *>(&raw_shared_memory[0]));

    shared_upstream_tile_t &shared_upstream_tile = shared_alloc.allocate<shared_upstream_tile_t>();
    shared_weight_tile_t   &shared_weight_tile   = shared_alloc.allocate<shared_weight_tile_t>();
    shared_delta_tile_t    &shared_delta_tile    = shared_alloc.allocate<shared_delta_tile_t>();

    std::uint32_t const base_atom_index = blockIdx.x * static_cast<std::uint32_t>(wgmma_m_rows_k);
    if (base_atom_index >= n_atoms) return;

    load_upstream_slab_(shared_upstream_tile, upstream_grad_global, n_atoms, base_atom_index);
    load_weight_slab_(shared_weight_tile, weight_global);
    kittens::warpgroup::sync(0);

    register_accumulator_t accumulator;
    kittens::warp::zero(accumulator);
    // D[M=64, N=c_in] = A[M=64, K=c_out] @ B[N=c_in, K=c_out]^T
    kittens::warpgroup::mm_ABt(accumulator, shared_upstream_tile, shared_weight_tile);
    kittens::warpgroup::mma_async_wait();

    scale_accumulator_in_place_(accumulator, scaled_path_weight);

    accumulate_into_grad_input_(grad_input_global,
                                 shared_delta_tile,
                                 accumulator,
                                 n_atoms, base_atom_index);
}

}  // namespace mace_tk_scalar_matmul_force_adjoint_detail_

/**
 *  @brief Launch the TK backward scalar matmul (L=0 per-irrep linear adjoint).
 *
 *  Inputs / outputs:
 *    - `upstream_grad_fp32[n_atoms, c_out]` fp32 row-major,
 *    - `weight_fp32[c_in, c_out]`           fp32 row-major (same buffer as forward),
 *    - `grad_input_fp32[n_atoms, c_in]`     fp32 row-major, accumulated `+=`.
 *
 *  Path-weight `1 / sqrt(c_in)` is folded internally; an optional uniform
 *  `pre_scale_multiplier` (typically `1 / avg_num_neighbors`) is multiplied
 *  in alongside.
 *
 *  Returns `false` when the channel-count tuple isn't handled; caller must
 *  fall back to `launch_mace_linear_per_irrep_force_adjoint`.
 */
inline bool launch_mace_tk_scalar_matmul_force_adjoint(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_channels_out,
    float const *weight_fp32,
    float const *upstream_grad_fp32,
    float *grad_input_fp32,
    cudaStream_t stream,
    float pre_scale_multiplier = 1.0f
) {
    if (n_atoms == 0u) return true;

    using namespace mace_tk_scalar_matmul_force_adjoint_detail_;

    if (n_channels_in  != static_cast<std::uint32_t>(channels_k) ||
        n_channels_out != static_cast<std::uint32_t>(channels_k)) {
        return false;
    }

    float const scaled_path_weight = pre_scale_multiplier
        / sqrtf(static_cast<float>(n_channels_in));

    constexpr std::size_t shared_memory_bytes_k = 64 * 1024;

    cudaFuncSetAttribute(
        mace_tk_scalar_matmul_force_adjoint_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_memory_bytes_k));

    std::uint32_t const n_atom_tiles = (n_atoms + static_cast<std::uint32_t>(wgmma_m_rows_k) - 1u)
                                       / static_cast<std::uint32_t>(wgmma_m_rows_k);
    dim3 const grid(n_atom_tiles, 1, 1);
    dim3 const block(static_cast<std::uint32_t>(warpgroup_thread_count_k), 1, 1);

    mace_tk_scalar_matmul_force_adjoint_kernel<<<grid, block, shared_memory_bytes_k, stream>>>(
        n_atoms,
        upstream_grad_fp32,
        weight_fp32,
        scaled_path_weight,
        grad_input_fp32);
    return true;
}

}  // namespace smallfield::cuda

#endif  // SMALLFIELD_USE_THUNDERKITTENS && __CUDACC__
