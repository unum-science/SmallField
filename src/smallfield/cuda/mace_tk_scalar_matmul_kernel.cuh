/**
 *  @file mace_tk_scalar_matmul_kernel.cuh
 *  @brief ThunderKittens scalar-feature matmul for MACE-OFF.
 *
 *  Replaces the `irrep_dim=1, active_irrep_block_count=1` fast path inside
 *  `launch_mace_linear_per_irrep` (which dispatches to the brute-force
 *  `mace_matmul_rowmajor_kernel`) for the L=0-only channel mixes:
 *
 *    - `linear_up`        — input scalar features → linear in channel,
 *    - `product_linear`   — post-symcontract scalar invariants → channel mix.
 *
 *  Algebraic shape per call:
 *
 *      output[atom, c_out] = (1 / sqrt(c_in))
 *                          * Σ_{c_in} weight[c_in, c_out] * input[atom, c_in]
 *
 *  with `c_in == c_out == 96` for MACE-OFF-small.
 *
 *  Why this exists:
 *
 *    The fp32 reference `mace_matmul_rowmajor_kernel` is a brute-force
 *    one-thread-per-output kernel: each thread accumulates `dim_k` (= c_in)
 *    fp32 multiplies sequentially and emits one fp32 store.  At the
 *    production saturating batch (n_atoms = 61 440 × n_channels = 96) NCU
 *    measured this kernel at ~377 GFLOPS — under 1 % of H100's fp32 peak,
 *    let alone bf16-WGMMA peak.  The kernel is L1-friendly but pipeline-
 *    serialised; with no ILP across outputs and no tile-level cooperation,
 *    it leaves the WGMMA path entirely unused.
 *
 *    The existing `mace_tk_per_irrep_linear_kernel` *can* serve the
 *    `active_irrep_block_count = 1` case, but it bakes M = 4 atoms × 16
 *    m-rows into the WGMMA tile; for an L=0-only call only m = 0 has data
 *    and the other 15 m-rows are zero-fill.  WGMMA still computes all 16
 *    rows — wasting 15/16 of the FMA throughput.
 *
 *    This kernel reinterprets the WGMMA M-axis as 64 ATOMS instead of
 *    4 atoms × 16 m-rows.  Every output row is a real atom-channel pair;
 *    no zero-fill, no waste.  Output stays fp32 (the order-3 ACE accumulator
 *    that consumes this output downstream is precision-load-bearing per
 *    the symmetric-contraction kernel's design).
 *
 *  Tile layout (small model, c = 96):
 *
 *      A = [M = 64 atoms, K = c_in = 96]             bf16 in shared
 *      B = [K = c_in = 96, N = c_out = 96]            bf16 in shared
 *      D = [M = 64,        N = c_out = 96]            fp32 register accumulator
 *      Output                                          fp32 stored direct
 *
 *  Per-CTA shared budget:
 *      input tile         : 64 × 96 × 2 = 12 KB
 *      weight tile        : 96 × 96 × 2 = 18 KB
 *      output staging fp32: 64 × 96 × 4 = 24 KB
 *      total                              ≈ 54 KB     (fits default 100 KB)
 *
 *  Grid: `dim3(ceil(n_atoms / 64), 1, 1)`.  Block: 128 threads (one warpgroup).
 *
 *  Output is fp32 to match `S.node_features_l0_only`'s buffer type at the
 *  call site.  The `1 / sqrt(c_in)` path-weight scale is folded into the
 *  fp32 accumulator before the store.  An optional `pre_scale_multiplier`
 *  gets multiplied into the same path_weight at no extra cost (mirroring
 *  the per_irrep_linear convention added in commit 683ed04 / Phase β-2).
 *
 *  Compile-guarded on `SMALLFIELD_USE_THUNDERKITTENS`.
 */

#pragma once

#include "smallfield/cuda/mace_tk_tile.cuh"
#include "smallfield/cuda/mace_tk_blackwell.cuh"

#if defined(SMALLFIELD_USE_THUNDERKITTENS) && defined(__CUDACC__)

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace smallfield::cuda {

namespace mace_tk_scalar_matmul_detail_ {

constexpr int channels_k = mace_tk::mace_channels_small_k;           // 96
constexpr int wgmma_m_rows_k = 64;                                   // atoms per CTA
constexpr int warpgroup_thread_count_k = kittens::WARPGROUP_THREADS; // 128

using shared_input_tile_t = mace_tk::shared_tile_bf16<wgmma_m_rows_k, channels_k>;
using shared_weight_tile_t = mace_tk::shared_tile_bf16<channels_k, channels_k>;
using shared_output_tile_t = mace_tk::shared_tile_fp32<wgmma_m_rows_k, channels_k>;

using register_accumulator_t = kittens::rt_fl<16, channels_k>;

/**
 *  @brief Stage 64 atoms' `[c_in]` input slabs into one shared `[64, c_in]` bf16 tile.
 *
 *  Each thread writes a strided subset.  Out-of-range atoms zero-fill so the
 *  WGMMA produces zero contributions on those rows; the per-thread store
 *  guard at the epilogue then drops them.
 */
__device__ __forceinline__ void load_input_slab_(shared_input_tile_t &dst_tile, float const *input_global,
                                                 std::uint32_t n_atoms, std::uint32_t base_atom_index) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = wgmma_m_rows_k * channels_k;
#pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const atom_in_cta = linear_index / channels_k;
        int const channel = linear_index % channels_k;
        std::uint32_t const atom_index = base_atom_index + static_cast<std::uint32_t>(atom_in_cta);
        float value = 0.0f;
        if (atom_index < n_atoms) {
            value = input_global[static_cast<std::uint64_t>(atom_index) * channels_k + channel];
        }
        dst_tile[{atom_in_cta, channel}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Stage the `[c_in, c_out]` weight slab fp32 → bf16 in shared memory.
 *  e3nn `o3.Linear` weights for an `irrep_dim=1, active_block=1` linear are
 *  exactly `[c_in, c_out]` row-major (no per-block axis to skip).
 */
__device__ __forceinline__ void load_weight_slab_(shared_weight_tile_t &dst_tile, float const *weight_global) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = channels_k * channels_k;
#pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const c_in_row = linear_index / channels_k;
        int const c_out_col = linear_index % channels_k;
        float const value = weight_global[static_cast<std::uint64_t>(c_in_row) * channels_k + c_out_col];
        dst_tile[{c_in_row, c_out_col}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Apply the e3nn `1 / sqrt(c_in)` (× optional pre-scale) path
 *         weight to every fp32 register-tile lane in place.
 */
__device__ __forceinline__ void scale_accumulator_in_place_(register_accumulator_t &accumulator,
                                                            float scaled_path_weight) {
    kittens::warpgroup::apply(
        accumulator, accumulator,
        [scaled_path_weight] __device__(int /*row*/, int /*col*/, float pre) { return pre * scaled_path_weight; });
}

/**
 *  @brief Stage the fp32 register accumulator into a `[64, c_out]` fp32
 *         shared tile, then cooperatively store to global with atom-bounds
 *         guards.
 *
 *  Output layout: `[n_atoms, c_out]` row-major fp32.  Out-of-range atoms
 *  (`base_atom + atom_in_cta >= n_atoms`) skip their row.
 */
__device__ __forceinline__ void store_output_atoms_fp32_(float *output_global, shared_output_tile_t &staging_tile,
                                                         register_accumulator_t const &accumulator,
                                                         std::uint32_t n_atoms, std::uint32_t base_atom_index) {
    kittens::warpgroup::store(staging_tile, accumulator);
    kittens::warpgroup::sync(0);

    int const thread_id = threadIdx.x;
    constexpr int tile_elements = wgmma_m_rows_k * channels_k;
#pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const atom_in_cta = linear_index / channels_k;
        int const channel = linear_index % channels_k;
        std::uint32_t const atom_index = base_atom_index + static_cast<std::uint32_t>(atom_in_cta);
        if (atom_index >= n_atoms) continue;
        output_global[static_cast<std::uint64_t>(atom_index) * channels_k + channel] =
            staging_tile[{atom_in_cta, channel}];
    }
}

/**
 *  @brief Persistent scalar-matmul kernel body.
 *
 *  Grid: one CTA per 64-atom tile.  Block: 128 threads (one warpgroup).
 *  Per CTA: load 64 atoms' input slab + the full c_in × c_out weight, run
 *  one WGMMA, scale, store fp32 directly.
 */
inline __global__ void mace_tk_scalar_matmul_kernel(std::uint32_t n_atoms,
                                                    float const *__restrict__ input_global,  // [n_atoms, c_in]
                                                    float const *__restrict__ weight_global, // [c_in, c_out]
                                                    float scaled_path_weight,         // 1 / sqrt(c_in) [× pre_scale]
                                                    float *__restrict__ output_global // [n_atoms, c_out]
) {
    extern __shared__ kittens::alignment_dummy raw_shared_memory[];
    kittens::shared_allocator shared_alloc(reinterpret_cast<int *>(&raw_shared_memory[0]));

    shared_input_tile_t &shared_input_tile = shared_alloc.allocate<shared_input_tile_t>();
    shared_weight_tile_t &shared_weight_tile = shared_alloc.allocate<shared_weight_tile_t>();
    shared_output_tile_t &shared_output_tile = shared_alloc.allocate<shared_output_tile_t>();

    mace_tk::mma_ctx_t<register_accumulator_t> mma_ctx;
    mma_ctx.init(shared_alloc);

    std::uint32_t const base_atom_index = blockIdx.x * static_cast<std::uint32_t>(wgmma_m_rows_k);
    if (base_atom_index >= n_atoms) return;

    load_input_slab_(shared_input_tile, input_global, n_atoms, base_atom_index);
    load_weight_slab_(shared_weight_tile, weight_global);
    kittens::warpgroup::sync(0);

    register_accumulator_t accumulator;
    mace_tk::mma_AB_overwrite_sync(accumulator, shared_input_tile, shared_weight_tile, mma_ctx);

    scale_accumulator_in_place_(accumulator, scaled_path_weight);

    store_output_atoms_fp32_(output_global, shared_output_tile, accumulator, n_atoms, base_atom_index);
}

} // namespace mace_tk_scalar_matmul_detail_

/**
 *  @brief Launch the TK-fused scalar-feature matmul.
 *
 *  Inputs / outputs:
 *    - `input_fp32[n_atoms, c_in]`     fp32 row-major,
 *    - `weight_fp32[c_in, c_out]`      fp32 row-major (e3nn `o3.Linear` weight blob;
 *                                       only the single block is consumed for `irrep_dim=1`),
 *    - `output_fp32[n_atoms, c_out]`   fp32 row-major (overwritten).
 *
 *  Path-weight `1/sqrt(c_in)` is folded internally; an optional uniform
 *  `pre_scale_multiplier` (typically `1 / avg_num_neighbors`) is multiplied
 *  in alongside, mirroring `launch_mace_tk_per_irrep_linear`'s convention.
 *
 *  Returns `false` when the channel-count tuple isn't handled (the kernel
 *  templates the small-model 96-channel shape into the WGMMA tile); caller
 *  must fall back to `launch_mace_linear_per_irrep` (which dispatches to
 *  `mace_matmul_rowmajor_kernel` on the irrep_dim=1 fast path).
 */
inline bool launch_mace_tk_scalar_matmul(std::uint32_t n_atoms, std::uint32_t n_channels_in,
                                         std::uint32_t n_channels_out, float const *weight_fp32,
                                         float const *input_fp32, float *output_fp32, cudaStream_t stream,
                                         float pre_scale_multiplier = 1.0f) {
    if (n_atoms == 0u) return true;

    using namespace mace_tk_scalar_matmul_detail_;

    if (n_channels_in != static_cast<std::uint32_t>(channels_k) ||
        n_channels_out != static_cast<std::uint32_t>(channels_k)) {
        return false;
    }

    float const scaled_path_weight = pre_scale_multiplier / sqrtf(static_cast<float>(n_channels_in));

    constexpr std::size_t shared_memory_bytes_k = 64 * 1024; // generous; well under H100's 228 KB cap

    cudaFuncSetAttribute(mace_tk_scalar_matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared_memory_bytes_k));

    std::uint32_t const n_atom_tiles = (n_atoms + static_cast<std::uint32_t>(wgmma_m_rows_k) - 1u) /
                                       static_cast<std::uint32_t>(wgmma_m_rows_k);
    dim3 const grid(n_atom_tiles, 1, 1);
    dim3 const block(static_cast<std::uint32_t>(warpgroup_thread_count_k), 1, 1);

    mace_tk_scalar_matmul_kernel<<<grid, block, shared_memory_bytes_k, stream>>>(n_atoms, input_fp32, weight_fp32,
                                                                                 scaled_path_weight, output_fp32);
    return true;
}

} // namespace smallfield::cuda

#endif // SMALLFIELD_USE_THUNDERKITTENS && __CUDACC__
