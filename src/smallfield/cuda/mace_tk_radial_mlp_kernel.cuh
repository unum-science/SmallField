/**
 *  @file mace_tk_radial_mlp_kernel.cuh
 *  @brief ThunderKittens-fused 4-layer radial MLP for MACE-OFF.
 *
 *  The MACE-OFF radial MLP is the per-edge feed-forward network that turns the
 *  Bessel × cutoff `radial_basis` into the per-path tensor-product weights:
 *
 *      hidden_0 = silu_normalize2mom( radial_basis @ W_0 / sqrt( 8 ) )
 *      hidden_1 = silu_normalize2mom( hidden_0     @ W_1 / sqrt(64) )
 *      hidden_2 = silu_normalize2mom( hidden_1     @ W_2 / sqrt(64) )
 *      tp_w     =                     hidden_2     @ W_3 / sqrt(64)
 *
 *  with shapes [n_edges, 8] -> [n_edges, 64] -> [n_edges, 64] ->
 *              [n_edges, 64] -> [n_edges, 384].  Weights are stored row-major
 *  as `(in_dim, out_dim)` per the e3nn `_Layer.weight` convention, and each
 *  layer applies a `1/sqrt(in_dim)` scale to the matmul output.  The
 *  activation between layers is `silu(x) * normalize2mom_constant` (e3nn's
 *  `normalize2mom(silu)`); see `mace_tk_tile.cuh::silu_normalize2mom_epilogue_t`.
 *
 *  The fp32 reference path in `mace_linear_kernel.cuh` runs each layer as a
 *  separate launch and bounces hidden activations through DRAM.  This file
 *  fuses all four layers into one persistent ThunderKittens kernel:
 *
 *    - bf16 weights for all four layers are converted from the fp32 weights
 *      blob once at kernel entry and pinned in shared memory across layers,
 *    - bf16 hidden activations live in shared memory between layers (no DRAM
 *      round-trip),
 *    - fp32 register accumulators are folded directly into bf16 store + silu
 *      epilogue without an intermediate global write,
 *    - each CTA owns a 64-edge tile and emits its 64 x 384 bf16 output slab
 *      in one pass over six 64-wide column tiles of the final layer.
 *
 *  This header is gated on `SMALLFIELD_USE_THUNDERKITTENS`; when the macro is
 *  off the file is empty and the fp32 reference path in
 *  `mace_linear_kernel.cuh` remains the only radial-MLP code path.
 *
 *  Validation: the smoke-test driver compares per-edge tp_weights against the
 *  fp32 reference (4 calls to `launch_mace_matmul_rowmajor` + 3 calls to
 *  `launch_mace_silu_norm2mom`) on a synthetic 128-edge batch, expected
 *  max-abs <= 1e-2 (bf16 noise).
 */

#pragma once

#include "smallfield/cuda/mace_tk_tile.cuh"
#include "smallfield/cuda/mace_tk_blackwell.cuh"

#if defined(SMALLFIELD_USE_THUNDERKITTENS) && defined(__CUDACC__)

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace smallfield::cuda {

namespace mace_tk_radial_mlp_detail_ {

// Compile-time shape constants for the MACE-OFF radial MLP.  Edge tile rows
// (64) is also the WGMMA M tile; `mace_radial_input_dim_k` = 8 is padded to
// `padded_input_dim_k` = 16 to match the bf16 WGMMA K granularity.  The
// output dim 384 is emitted as `output_n_tile_count_k` = 6 column tiles of
// width `mace_radial_hidden_dim_k` = 64 each.
constexpr int edge_tile_rows_k = 64;
constexpr int hidden_dim_k = mace_tk::mace_radial_hidden_dim_k;      // 64
constexpr int radial_input_dim_k = mace_tk::mace_radial_input_dim_k; // 8
constexpr int padded_input_dim_k = 16;                               // bf16 WGMMA K granularity
constexpr int output_dim_k = mace_tk::mace_radial_output_dim_tk_max_k; // 512 (covers 384 via runtime guards)
constexpr int output_n_tile_count_k = output_dim_k / hidden_dim_k;   // 6
constexpr int warpgroup_thread_count_k = kittens::WARPGROUP_THREADS; // 128

// Shared-tile aliases.  All activations and weights live as bf16 in shared
// memory; the WGMMA accumulator is fp32 in registers (rt_fl).
using shared_tile_input_t = mace_tk::shared_tile_bf16<edge_tile_rows_k, padded_input_dim_k>;
using shared_tile_hidden_t = mace_tk::shared_tile_bf16<edge_tile_rows_k, hidden_dim_k>;
using shared_tile_w0_t = mace_tk::shared_tile_bf16<padded_input_dim_k, hidden_dim_k>;
using shared_tile_w_hh_t = mace_tk::shared_tile_bf16<hidden_dim_k, hidden_dim_k>;
using shared_tile_w3_col_t = mace_tk::shared_tile_bf16<hidden_dim_k, hidden_dim_k>;

// Per-warpgroup register accumulator.  The warpgroup of 4 warps owns 64 rows
// (16 each) by `hidden_dim_k` columns of fp32.  rt_bf is used to stage bf16
// register data when copying weights from global into shared memory.
using register_accumulator_t = kittens::rt_fl<16, hidden_dim_k>;
using register_weight_w0_t = kittens::rt_bf<16, hidden_dim_k>;   // 64x64 across warpgroup, K-padded
using register_weight_hh_t = kittens::rt_bf<16, hidden_dim_k>;   // 64x64 across warpgroup
using register_input_t = kittens::rt_bf<16, padded_input_dim_k>; // 64x16 across warpgroup

// Global layouts.  Inputs and weights arrive in fp32; output is bf16.  We
// expose the "row count" axis as dynamic so the same kernel template handles
// any edge count; column sizes are static to fold into the swizzle.
using gl_input_fp32_t = kittens::gl<float, 1, 1, -1, radial_input_dim_k>;
using gl_w0_fp32_t = kittens::gl<float, 1, 1, radial_input_dim_k, hidden_dim_k>;
using gl_w_hh_fp32_t = kittens::gl<float, 1, 1, hidden_dim_k, hidden_dim_k>;
using gl_w3_fp32_t = kittens::gl<float, 1, 1, hidden_dim_k, output_dim_k>;
using gl_output_bf16_t = kittens::gl<__nv_bfloat16, 1, 1, -1, output_dim_k>;

/**
 *  @brief Layer-N residual scale factor — `1.0f / sqrtf(in_dim)`.
 *
 *  Matches the e3nn `_Layer` path-normalisation, computed once per layer at
 *  kernel-launch time on the host and passed as a kernel argument so we
 *  avoid a per-thread sqrt in the device path.  Folded into the silu
 *  epilogue for activated layers and into the final-layer store epilogue.
 */
struct layer_scale_factors_t {
    float layer_0_scale;
    float layer_1_scale;
    float layer_2_scale;
    float layer_3_scale;
};

/**
 *  @brief Stage the radial-basis input tile into shared memory, padded.
 *
 *  Reads `[edge_tile_rows_k, radial_input_dim_k = 8]` from global fp32 and
 *  writes `[edge_tile_rows_k, padded_input_dim_k = 16]` bf16 into shared,
 *  zero-filling cols 8..15.  Out-of-range edge rows (`base_edge + r >=
 *  n_edges`) are also zeroed so the matmul produces zeros that the caller
 *  can ignore at the global-store boundary.
 */
__device__ __forceinline__ void load_input_tile_padded_(shared_tile_input_t &dst_input_tile,
                                                        float const *radial_basis_input, std::uint32_t n_edges,
                                                        std::uint32_t base_edge_index) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = edge_tile_rows_k * padded_input_dim_k;
#pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / padded_input_dim_k;
        int const col_in_tile = linear_index % padded_input_dim_k;
        std::uint32_t const edge_index = base_edge_index + static_cast<std::uint32_t>(row_in_tile);
        float value = 0.0f;
        if (edge_index < n_edges && col_in_tile < radial_input_dim_k) {
            value = radial_basis_input[static_cast<std::uint64_t>(edge_index) * radial_input_dim_k + col_in_tile];
        }
        dst_input_tile[{row_in_tile, col_in_tile}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Stage layer-0 weight `[8, 64]` into a `[16, 64]` bf16 shared tile.
 *
 *  Pads rows 8..15 with zeros to match `padded_input_dim_k` so the WGMMA
 *  K dimension is a multiple of 16.  Run once per CTA at kernel entry.
 */
__device__ __forceinline__ void load_layer_0_weight_padded_(shared_tile_w0_t &dst_weight_tile,
                                                            float const *layer_0_weight) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = padded_input_dim_k * hidden_dim_k;
#pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / hidden_dim_k;
        int const col_in_tile = linear_index % hidden_dim_k;
        float value = 0.0f;
        if (row_in_tile < radial_input_dim_k) { value = layer_0_weight[row_in_tile * hidden_dim_k + col_in_tile]; }
        dst_weight_tile[{row_in_tile, col_in_tile}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Stage a hidden-to-hidden weight `[64, 64]` from fp32 global into
 *         bf16 shared.  Used for layers 1 and 2.
 */
__device__ __forceinline__ void load_hidden_to_hidden_weight_(shared_tile_w_hh_t &dst_weight_tile,
                                                              float const *hidden_weight) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = hidden_dim_k * hidden_dim_k;
#pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / hidden_dim_k;
        int const col_in_tile = linear_index % hidden_dim_k;
        float const value = hidden_weight[row_in_tile * hidden_dim_k + col_in_tile];
        dst_weight_tile[{row_in_tile, col_in_tile}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Stage the final layer-3 weight column tile `[64, 64]` from the
 *         full `[64, 384]` fp32 weight into bf16 shared.
 *
 *  `output_column_tile_index` selects the 64-wide column slab of the 384-wide
 *  output dimension to load.  Called six times per CTA, once per output tile.
 */
__device__ __forceinline__ void load_layer_3_weight_column_tile_(shared_tile_w3_col_t &dst_weight_tile,
                                                                 float const *layer_3_weight,
                                                                 std::uint32_t runtime_output_dim,
                                                                 int output_column_tile_index) {
    int const thread_id = threadIdx.x;
    int const base_output_col = output_column_tile_index * hidden_dim_k;
    constexpr int tile_elements = hidden_dim_k * hidden_dim_k;
#pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / hidden_dim_k;
        int const col_in_tile = linear_index % hidden_dim_k;
        float value = 0.0f;
        int const output_col = base_output_col + col_in_tile;
        if (output_col < static_cast<int>(runtime_output_dim)) {
            value = layer_3_weight[row_in_tile * runtime_output_dim + output_col];
        }
        dst_weight_tile[{row_in_tile, col_in_tile}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Apply `silu(x) * normalize2mom_constant` to a fp32 register tile.
 *
 *  Uses `kittens::warpgroup::apply` to walk every (row, col) lane in the
 *  rt_fl<16, 64> accumulator with the silu-normalize2mom epilogue functor
 *  declared in `mace_tk_tile.cuh`.  Pre-activation values are passed in;
 *  post-activation values are written back into the same register tile.
 */
__device__ __forceinline__ void apply_silu_normalize2mom_in_place_(register_accumulator_t &accumulator) {
    mace_tk::silu_normalize2mom_epilogue_t silu_op;
    kittens::warpgroup::apply(
        accumulator, accumulator,
        [silu_op] __device__(int /*row*/, int /*col*/, float pre_activation) { return silu_op(pre_activation); });
}

/**
 *  @brief Scale a fp32 register tile by a host-supplied `1/sqrt(in_dim)`.
 *
 *  Folded into the post-MMA path so we apply the e3nn `_Layer.path_weight`
 *  before silu (matches the fp32 reference path in `mace_linear_kernel.cuh`).
 */
__device__ __forceinline__ void scale_accumulator_in_place_(register_accumulator_t &accumulator, float scale) {
    kittens::warpgroup::apply(accumulator, accumulator,
                              [scale] __device__(int /*row*/, int /*col*/, float pre) { return pre * scale; });
}

/**
 *  @brief Write a `rt_fl<16, 64>` accumulator to an `[n_edges, 384]` bf16
 *         output, masking rows past `n_edges` and converting fp32 -> bf16.
 *
 *  We can't use the canonical TK `warpgroup::store(gl, rt, coord)` directly
 *  because it would write past the end of the output array on partial edge
 *  tiles.  Instead we stage to a shared bf16 tile and then iterate manually
 *  over rows with an `edge_index < n_edges` guard.
 */
__device__ __forceinline__ void store_output_tile_with_guard_(__nv_bfloat16 *output_global,
                                                              shared_tile_hidden_t &staging_shared_tile,
                                                              register_accumulator_t const &accumulator,
                                                              std::uint32_t n_edges, std::uint32_t base_edge_index,
                                                              std::uint32_t runtime_output_dim,
                                                              int output_column_tile_index) {
    // Convert fp32 register tile to bf16 in shared memory (with conversion).
    kittens::warpgroup::store(staging_shared_tile, accumulator);
    kittens::warpgroup::sync(0);

    int const thread_id = threadIdx.x;
    constexpr int tile_elements = edge_tile_rows_k * hidden_dim_k;
    int const base_output_col = output_column_tile_index * hidden_dim_k;
#pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / hidden_dim_k;
        int const col_in_tile = linear_index % hidden_dim_k;
        std::uint32_t const edge_index = base_edge_index + static_cast<std::uint32_t>(row_in_tile);
        if (edge_index >= n_edges) continue;
        int const output_col = base_output_col + col_in_tile;
        if (output_col >= static_cast<int>(runtime_output_dim)) continue;
        __nv_bfloat16 const value = staging_shared_tile[{row_in_tile, col_in_tile}];
        output_global[static_cast<std::uint64_t>(edge_index) * runtime_output_dim + output_col] = value;
    }
}

/**
 *  @brief Stage accumulator to bf16 via the supplied shared tile, then write
 *  bf16 row-major to a `[n_edges, 64]` global pre-activation buffer with the
 *  partial-tile edge guard.  The staging tile is reused across the per-layer
 *  pre-act dumps; in this kernel we use `shared_layer_3_weight_column` as
 *  the staging slot (it's loaded only inside the layer-3 column loop).
 *
 *  Used to expose layer 0/1/2 pre-activation values for the radial-MLP
 *  backward chain (Phase B6) — saves the fp32 forward rerun previously
 *  needed in `mace_inference_run_force_adjoint` (commit `7fc5b11`).
 */
__device__ __forceinline__ void store_pre_activation_bf16_(__nv_bfloat16 *output_global,
                                                           shared_tile_w3_col_t &staging_shared_tile,
                                                           register_accumulator_t const &accumulator,
                                                           std::uint32_t n_edges, std::uint32_t base_edge_index) {
    kittens::warpgroup::store(staging_shared_tile, accumulator);
    kittens::warpgroup::sync(0);

    int const thread_id = threadIdx.x;
    constexpr int tile_elements = edge_tile_rows_k * hidden_dim_k;
#pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / hidden_dim_k;
        int const col_in_tile = linear_index % hidden_dim_k;
        std::uint32_t const edge_index = base_edge_index + static_cast<std::uint32_t>(row_in_tile);
        if (edge_index >= n_edges) continue;
        __nv_bfloat16 const value = staging_shared_tile[{row_in_tile, col_in_tile}];
        output_global[static_cast<std::uint64_t>(edge_index) * hidden_dim_k + col_in_tile] = value;
    }
}

/**
 *  @brief Persistent radial-MLP kernel body: 64 edges per CTA, 4 layers fused.
 *
 *  Grid: `dim3(n_edge_tiles, 1, 1)` where `n_edge_tiles = ceil(n_edges/64)`.
 *  Block: 128 threads (one warpgroup, 4 warps).  No producer/consumer split —
 *  the per-CTA workload is small enough that a single warpgroup keeps the
 *  shared memory and register footprint compact while removing 3 of 4 kernel
 *  launches relative to the fp32 reference.
 *
 *  When `pre_act_layer_*_bf16` are non-null, the kernel stores the per-layer
 *  pre-activation values (post-scale, pre-silu·n2m) to bf16 global buffers
 *  shaped `[n_edges, 64]`.  These are consumed by the radial-MLP backward
 *  to skip the fp32 forward rerun (Phase B6 of MACE_OPTIMIZATION_ROADMAP.md).
 */
inline __global__ void mace_tk_radial_mlp_kernel(
    std::uint32_t n_edges, float const *__restrict__ radial_basis_input, float const *__restrict__ layer_0_weight,
    float const *__restrict__ layer_1_weight, float const *__restrict__ layer_2_weight,
    float const *__restrict__ layer_3_weight, __nv_bfloat16 *__restrict__ tensor_product_weights_per_edge,
    std::uint32_t runtime_output_dim,
    __nv_bfloat16 *__restrict__ pre_act_layer_0_bf16, // [n_edges, 64] or nullptr
    __nv_bfloat16 *__restrict__ pre_act_layer_1_bf16, // [n_edges, 64] or nullptr
    __nv_bfloat16 *__restrict__ pre_act_layer_2_bf16, // [n_edges, 64] or nullptr
    layer_scale_factors_t scales) {
    extern __shared__ kittens::alignment_dummy raw_shared_memory[];
    kittens::shared_allocator shared_alloc(reinterpret_cast<int *>(&raw_shared_memory[0]));

    // Persistent per-CTA shared memory: weights for layers 0,1,2,3-column-tile,
    // plus an input tile and a single hidden tile that ping-pongs across layers.
    shared_tile_w0_t &shared_layer_0_weight = shared_alloc.allocate<shared_tile_w0_t>();
    shared_tile_w_hh_t &shared_layer_1_weight = shared_alloc.allocate<shared_tile_w_hh_t>();
    shared_tile_w_hh_t &shared_layer_2_weight = shared_alloc.allocate<shared_tile_w_hh_t>();
    shared_tile_w3_col_t &shared_layer_3_weight_column = shared_alloc.allocate<shared_tile_w3_col_t>();
    shared_tile_input_t &shared_input_tile = shared_alloc.allocate<shared_tile_input_t>();
    shared_tile_hidden_t &shared_hidden_tile = shared_alloc.allocate<shared_tile_hidden_t>();

    // All 4 layer MMAs share the same `register_accumulator_t` shape, so
    // a single context (one TT slot + one pair of semaphores) suffices.
    mace_tk::mma_ctx_t<register_accumulator_t> mma_ctx;
    mma_ctx.init(shared_alloc);

    std::uint32_t const base_edge_index = blockIdx.x * static_cast<std::uint32_t>(edge_tile_rows_k);
    if (base_edge_index >= n_edges) return;

    // Stage weights for layers 0, 1, 2 into shared memory once.  Layer 3
    // weights are 6x larger (64x384) than the per-tile budget, so we load
    // each 64-wide output column tile just-in-time inside the layer-3 loop.
    load_layer_0_weight_padded_(shared_layer_0_weight, layer_0_weight);
    load_hidden_to_hidden_weight_(shared_layer_1_weight, layer_1_weight);
    load_hidden_to_hidden_weight_(shared_layer_2_weight, layer_2_weight);

    // Stage the per-CTA input slab.
    load_input_tile_padded_(shared_input_tile, radial_basis_input, n_edges, base_edge_index);

    kittens::warpgroup::sync(0);

    // ---- Layer 0: [64x16] @ [16x64] -> [64x64] + silu*normalize2mom ----
    register_accumulator_t accumulator;
    mace_tk::mma_AB_overwrite_sync(accumulator, shared_input_tile, shared_layer_0_weight, mma_ctx);

    scale_accumulator_in_place_(accumulator, scales.layer_0_scale);
    if (pre_act_layer_0_bf16 != nullptr) {
        store_pre_activation_bf16_(pre_act_layer_0_bf16, shared_layer_3_weight_column, accumulator, n_edges,
                                   base_edge_index);
    }
    apply_silu_normalize2mom_in_place_(accumulator);

    kittens::warpgroup::store(shared_hidden_tile, accumulator);
    kittens::warpgroup::sync(0);

    // ---- Layer 1: [64x64] @ [64x64] -> [64x64] + silu*normalize2mom ----
    mace_tk::mma_AB_overwrite_sync(accumulator, shared_hidden_tile, shared_layer_1_weight, mma_ctx);

    scale_accumulator_in_place_(accumulator, scales.layer_1_scale);
    if (pre_act_layer_1_bf16 != nullptr) {
        store_pre_activation_bf16_(pre_act_layer_1_bf16, shared_layer_3_weight_column, accumulator, n_edges,
                                   base_edge_index);
    }
    apply_silu_normalize2mom_in_place_(accumulator);

    kittens::warpgroup::store(shared_hidden_tile, accumulator);
    kittens::warpgroup::sync(0);

    // ---- Layer 2: [64x64] @ [64x64] -> [64x64] + silu*normalize2mom ----
    mace_tk::mma_AB_overwrite_sync(accumulator, shared_hidden_tile, shared_layer_2_weight, mma_ctx);

    scale_accumulator_in_place_(accumulator, scales.layer_2_scale);
    if (pre_act_layer_2_bf16 != nullptr) {
        store_pre_activation_bf16_(pre_act_layer_2_bf16, shared_layer_3_weight_column, accumulator, n_edges,
                                   base_edge_index);
    }
    apply_silu_normalize2mom_in_place_(accumulator);

    kittens::warpgroup::store(shared_hidden_tile, accumulator);
    kittens::warpgroup::sync(0);

// ---- Layer 3: [64x64] @ [64x64] -> [64x64] per output column tile,
//               6 column tiles total, no activation.  Final store is bf16.
#pragma unroll
    for (int output_column_tile_index = 0; output_column_tile_index < output_n_tile_count_k;
         ++output_column_tile_index) {
        load_layer_3_weight_column_tile_(shared_layer_3_weight_column, layer_3_weight, runtime_output_dim,
                                         output_column_tile_index);
        kittens::warpgroup::sync(0);

        mace_tk::mma_AB_overwrite_sync(accumulator, shared_hidden_tile, shared_layer_3_weight_column, mma_ctx);

        scale_accumulator_in_place_(accumulator, scales.layer_3_scale);

        // We reuse `shared_layer_3_weight_column` as a staging tile for the
        // bf16 conversion — its lifetime is bounded by the next iteration.
        // Re-use of `shared_hidden_tile` (the layer-3 input) would corrupt
        // the input on the next column-tile iteration.
        store_output_tile_with_guard_(tensor_product_weights_per_edge, shared_layer_3_weight_column, accumulator,
                                      n_edges, base_edge_index, runtime_output_dim, output_column_tile_index);
        kittens::warpgroup::sync(0);
    }
}

} // namespace mace_tk_radial_mlp_detail_

/**
 *  @brief Launch the TK-fused radial MLP for MACE-OFF.
 *
 *  Reads `radial_basis_input[n_edges, n_radial=8]` in fp32, applies the four
 *  MACE-OFF radial-MLP layers (8 -> 64 -> 64 -> 64 -> {384,512}) with
 *  `normalize2mom(silu)` between activated layers and a `1/sqrt(in_dim)` scale
 *  per layer (e3nn `_Layer` convention), and writes
 *  `tensor_product_weights_per_edge[n_edges, 384]` in bf16.
 *
 *  Weights are read in fp32 from the existing weights blob — the kernel
 *  converts them to bf16 in shared memory once at kernel entry, so the
 *  hidden activations live in bf16 throughout but the WGMMA accumulators
 *  are fp32.
 *
 *  Requires `n_radial == 8` (the MACE-OFF23 setting); other values are not
 *  supported by this fused kernel and should fall back to the fp32 reference.
 *
 *  Gated on `SMALLFIELD_USE_THUNDERKITTENS`; only compiled when the macro is
 *  defined.  If the macro is off, callers should use the four-launch fp32
 *  reference path in `mace_linear_kernel.cuh`.
 */
/// Returns true on successful launch, false if the (n_radial, output_dim)
/// tuple isn't handled (caller must fall back to the fp32 reference path).
inline bool launch_mace_tk_radial_mlp(std::uint32_t n_edges, std::uint32_t n_radial,
                                      std::uint32_t runtime_output_dim, float const *radial_basis_input,
                                      float const *radial_mlp_weight_layer_0, float const *radial_mlp_weight_layer_1,
                                      float const *radial_mlp_weight_layer_2, float const *radial_mlp_weight_layer_3,
                                      __nv_bfloat16 *tensor_product_weights_per_edge, cudaStream_t stream,
                                      __nv_bfloat16 *pre_act_layer_0_bf16 = nullptr,
                                      __nv_bfloat16 *pre_act_layer_1_bf16 = nullptr,
                                      __nv_bfloat16 *pre_act_layer_2_bf16 = nullptr) {
    if (n_edges == 0u) return true;
    // The fused kernel template-bakes the small-or-medium max shape
    // (n_radial=8, hidden=64, output_dim up to 4·128=512).  Larger output
    // widths still need the fp32 reference path.
    if (n_radial != static_cast<std::uint32_t>(mace_tk_radial_mlp_detail_::radial_input_dim_k)) { return false; }
    if (runtime_output_dim != static_cast<std::uint32_t>(mace_tk::mace_radial_output_dim_k) &&
        runtime_output_dim != static_cast<std::uint32_t>(mace_tk::mace_radial_output_dim_medium_k)) {
        return false;
    }

    using namespace mace_tk_radial_mlp_detail_;

    layer_scale_factors_t scales;
    scales.layer_0_scale = 1.0f / sqrtf(static_cast<float>(radial_input_dim_k));
    scales.layer_1_scale = 1.0f / sqrtf(static_cast<float>(hidden_dim_k));
    scales.layer_2_scale = 1.0f / sqrtf(static_cast<float>(hidden_dim_k));
    scales.layer_3_scale = 1.0f / sqrtf(static_cast<float>(hidden_dim_k));

    // Shared-mem footprint, one of each tile type:
    //   layer_0_weight       16 x 64 bf16 =  2 KB
    //   layer_1_weight       64 x 64 bf16 =  8 KB
    //   layer_2_weight       64 x 64 bf16 =  8 KB
    //   layer_3_weight_col   64 x 64 bf16 =  8 KB (one column tile at a time)
    //   input_tile           64 x 16 bf16 =  2 KB
    //   hidden_tile          64 x 64 bf16 =  8 KB
    //   ----                                ------
    //                                       ~36 KB + alignment slack
    constexpr std::size_t shared_memory_bytes_k = 64 * 1024; // generous, well under H100's 228 KB cap

    cudaFuncSetAttribute(mace_tk_radial_mlp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared_memory_bytes_k));

    std::uint32_t const n_edge_tiles = (n_edges + static_cast<std::uint32_t>(edge_tile_rows_k) - 1u) /
                                       static_cast<std::uint32_t>(edge_tile_rows_k);
    dim3 const grid(n_edge_tiles, 1, 1);
    dim3 const block(static_cast<std::uint32_t>(warpgroup_thread_count_k), 1, 1);

    mace_tk_radial_mlp_kernel<<<grid, block, shared_memory_bytes_k, stream>>>(
        n_edges, radial_basis_input, radial_mlp_weight_layer_0, radial_mlp_weight_layer_1, radial_mlp_weight_layer_2,
        radial_mlp_weight_layer_3, tensor_product_weights_per_edge, runtime_output_dim, pre_act_layer_0_bf16,
        pre_act_layer_1_bf16, pre_act_layer_2_bf16, scales);
    return true;
}

} // namespace smallfield::cuda

#endif // SMALLFIELD_USE_THUNDERKITTENS && __CUDACC__
