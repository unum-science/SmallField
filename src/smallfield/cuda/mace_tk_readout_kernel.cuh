/**
 *  @file mace_tk_readout_kernel.cuh
 *  @brief ThunderKittens-fused readout kernels for MACE-OFF.
 *
 *  Each MACE interaction layer feeds a per-atom scalar invariant tensor
 *  `node_scalar_invariants[n_atoms, n_channels]` (the L=0 component of the
 *  post-product node features) into a readout that emits a per-atom energy
 *  contribution.  MACE-OFF uses two flavours:
 *
 *    - Layer 0 (linear readout):
 *        e[a] += (1/sqrt(c)) * Σ_c W[c] * inv[a, c]
 *
 *    - Layer N-1 (nonlinear MLP readout, hidden_dim = 16):
 *        h[a, k] = silu_normalize2mom( (1/sqrt(c)) * Σ_c W1[c, k] * inv[a, c] )
 *        e[a]   += (1/sqrt(hidden))   * Σ_k W2[k]   * h[a, k]
 *
 *  The 1/sqrt(in_dim) factors are e3nn `o3.Linear` path-weight normalisations
 *  (irreps_in = `c x 0e` to `1 x 0e` gives alpha = c).  The silu epilogue is
 *  `silu(x) * normalize2mom_constant`, the same activation that wraps every
 *  layer of the radial MLP — see
 *  `mace_tk_tile.cuh::silu_normalize2mom_epilogue_t`.
 *
 *  Both readouts are tiny per-atom matvec / two-stage MLP operations.  The
 *  fp32 reference path (see `mace_inference.cuh::mace_detail_::
 *  mace_readout_linear_kernel` and `mace_readout_nonlinear_mlp_kernel`)
 *  runs one CUDA block per atom and accumulates with `atomicAdd`.  The TK
 *  kernels here promote each launch to a single-warpgroup persistent kernel
 *  that processes 64 atoms at a time using bf16 WGMMA with fp32 accumulators:
 *
 *    - linear readout: one [64, 96] @ [96, 16]_padded -> [64, 16] WGMMA, the
 *      padded W column 0 carries the readout weight, columns 1..15 are zero,
 *      and only column 0 of the accumulator is consumed,
 *
 *    - nonlinear MLP: stage 1 is [64, 96] @ [96, 16] -> [64, 16] with the
 *      silu_normalize2mom epilogue (and 1/sqrt(c) scale) folded into the
 *      `apply` lambda; stage 2 is [64, 16] @ [16, 16]_padded -> [64, 16]
 *      with W2 placed in column 0 of the [16, 16] padded weight tile.  The
 *      hidden activation lives in shared memory between the two stages, so
 *      there is one shared-memory hand-off and zero DRAM round-trip on the
 *      hidden vector.
 *
 *  bf16 storage and compute, fp32 register accumulator, fp32 `+=` into the
 *  caller's `node_energy_out` matches the existing reference contract — the
 *  per-atom energy tensor accumulates contributions across both readouts and
 *  the energy reduce kernel.
 *
 *  This header is gated on `SMALLFIELD_USE_THUNDERKITTENS`; when the macro
 *  is off the file is empty and the fp32 reference path in
 *  `mace_inference.cuh` remains the only readout code path.
 */

#pragma once

#include "smallfield/cuda/mace_tk_tile.cuh"

#if defined(SMALLFIELD_USE_THUNDERKITTENS) && defined(__CUDACC__)

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace smallfield::cuda {

namespace mace_tk_readout_detail_ {

// Compile-time shape constants for the MACE-OFF readouts.  The atom tile
// height (64) is the WGMMA M tile minimum.  The hidden / output column tile
// width (16) is the WGMMA N tile minimum; for the linear readout we pad the
// 1-wide output dim to 16, and for the nonlinear readout we use the natural
// hidden_dim = 16.  bf16 WGMMA also requires K to be a multiple of 16 — the
// MACE-OFF n_channels = 96 satisfies this directly with no padding.
constexpr int atom_tile_rows_k         = 64;
constexpr int output_tile_cols_k       = 16;     // 1 (linear) or hidden=16 (nonlinear) padded to 16
constexpr int nonlinear_hidden_dim_k   = 16;     // MACE-OFF hidden dim of the nonlinear MLP readout
constexpr int warpgroup_thread_count_k = kittens::WARPGROUP_THREADS; // 128

// Maximum supported channel count.  MACE-OFF small/medium/large = {96, 128,
// 192}; the templates below are baked at compile time, so the launcher
// dispatches to the matching specialisation.  We hard-wire the small tier
// here (n_channels = 96) — that is the variant Track C5 validates against.
// Extending to 128 / 192 amounts to instantiating the kernel template with
// the new constant; the body is shape-agnostic past the WGMMA static checks.
constexpr int n_channels_small_k       = mace_tk::mace_channels_small_k; // 96

// Shared-tile aliases.  All activations and weights live as bf16 in shared
// memory; accumulators are fp32 in registers.
using shared_tile_invariants_t = mace_tk::shared_tile_bf16<atom_tile_rows_k, n_channels_small_k>;
using shared_tile_w_linear_t   = mace_tk::shared_tile_bf16<n_channels_small_k, output_tile_cols_k>;
using shared_tile_w_mlp_1_t    = mace_tk::shared_tile_bf16<n_channels_small_k, nonlinear_hidden_dim_k>;
using shared_tile_hidden_t     = mace_tk::shared_tile_bf16<atom_tile_rows_k, nonlinear_hidden_dim_k>;
using shared_tile_w_mlp_2_t    = mace_tk::shared_tile_bf16<nonlinear_hidden_dim_k, output_tile_cols_k>;

// Per-warpgroup register accumulator.  rt_fl<16, 16> is the canonical
// 64x16 warpgroup output (each of 4 warps owns 16 rows).
using register_accumulator_t = kittens::rt_fl<16, output_tile_cols_k>;

/**
 *  @brief Stage `[atom_tile_rows_k, n_channels_small_k]` invariants from
 *         fp32 global into bf16 shared, masking rows past `n_atoms` with 0.
 *
 *  Rows past the end of the input get zero so the WGMMA accumulator's tail
 *  rows hold a defined value (zero) we can ignore at the global-store
 *  boundary, mirroring the radial-MLP kernel's approach.
 */
__device__ __forceinline__ void load_invariants_tile_(
    shared_tile_invariants_t &dst_invariants_tile,
    float const *node_scalar_invariants,
    std::uint32_t n_atoms,
    std::uint32_t base_atom_index
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = atom_tile_rows_k * n_channels_small_k;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / n_channels_small_k;
        int const col_in_tile = linear_index % n_channels_small_k;
        std::uint32_t const atom_index = base_atom_index + static_cast<std::uint32_t>(row_in_tile);
        float value = 0.0f;
        if (atom_index < n_atoms) {
            value = node_scalar_invariants[static_cast<std::uint64_t>(atom_index) * n_channels_small_k + col_in_tile];
        }
        dst_invariants_tile[{row_in_tile, col_in_tile}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Stage the linear-readout weight `[n_channels]` into a
 *         `[n_channels, 16]` bf16 shared tile, with column 0 carrying the
 *         weight and columns 1..15 zero-padded.
 *
 *  This packs the matvec into a `[64, 96] @ [96, 16] -> [64, 16]` WGMMA;
 *  only column 0 of the resulting accumulator is consumed at the output
 *  store.  Wasting 15/16 of the WGMMA columns is acceptable because the
 *  readout is nano-tiny (one per atom-tile per launch) and lets us reuse
 *  the canonical warpgroup matmul plumbing without a hand-rolled matvec.
 */
__device__ __forceinline__ void load_linear_readout_weight_padded_(
    shared_tile_w_linear_t &dst_weight_tile,
    float const *readout_weight
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = n_channels_small_k * output_tile_cols_k;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / output_tile_cols_k;
        int const col_in_tile = linear_index % output_tile_cols_k;
        float value = 0.0f;
        if (col_in_tile == 0) value = readout_weight[row_in_tile];
        dst_weight_tile[{row_in_tile, col_in_tile}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Stage the nonlinear-readout layer-1 weight `[n_channels, hidden]`
 *         (full bf16, no padding — hidden_dim = 16 already matches the WGMMA
 *         tile column granularity).
 */
__device__ __forceinline__ void load_nonlinear_w1_weight_(
    shared_tile_w_mlp_1_t &dst_weight_tile,
    float const *mlp_w1_weight
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = n_channels_small_k * nonlinear_hidden_dim_k;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / nonlinear_hidden_dim_k;
        int const col_in_tile = linear_index % nonlinear_hidden_dim_k;
        float const value = mlp_w1_weight[row_in_tile * nonlinear_hidden_dim_k + col_in_tile];
        dst_weight_tile[{row_in_tile, col_in_tile}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Stage the nonlinear-readout layer-2 weight `[hidden]` into a
 *         `[hidden, 16]` bf16 shared tile, column 0 carries the weight and
 *         columns 1..15 are zero (analogous to the linear readout's W).
 *
 *  The second stage of the MLP is a `[hidden] x [hidden]` -> `scalar`
 *  matvec; we pack it into `[64, 16] @ [16, 16] -> [64, 16]` for the same
 *  reason: column 0 carries the answer, the rest are wasted but cheap.
 */
__device__ __forceinline__ void load_nonlinear_w2_weight_padded_(
    shared_tile_w_mlp_2_t &dst_weight_tile,
    float const *mlp_w2_weight
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = nonlinear_hidden_dim_k * output_tile_cols_k;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / output_tile_cols_k;
        int const col_in_tile = linear_index % output_tile_cols_k;
        float value = 0.0f;
        if (col_in_tile == 0) value = mlp_w2_weight[row_in_tile];
        dst_weight_tile[{row_in_tile, col_in_tile}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Apply `silu_normalize2mom( pre * scale )` in-place to the
 *         accumulator, folding the path-weight scale and the activation
 *         into one register-resident `apply` pass.
 *
 *  Mirrors the radial-MLP kernel's epilogue but takes the scale as an
 *  argument so the readout's 1/sqrt(c) does not need a separate apply pass.
 */
__device__ __forceinline__ void apply_silu_normalize2mom_with_scale_in_place_(
    register_accumulator_t &accumulator,
    float pre_activation_scale
) {
    mace_tk::silu_normalize2mom_epilogue_t silu_op;
    kittens::warpgroup::apply(
        accumulator,
        accumulator,
        [silu_op, pre_activation_scale] __device__ (int /*row*/, int /*col*/, float pre) {
            return silu_op(pre * pre_activation_scale);
        });
}

/**
 *  @brief Scale a fp32 register tile by a host-supplied factor in place.
 *
 *  Used to fold the linear readout's 1/sqrt(c) and the nonlinear readout's
 *  outer 1/sqrt(hidden) into the post-MMA path before the final atomic-add.
 */
__device__ __forceinline__ void scale_accumulator_in_place_(
    register_accumulator_t &accumulator,
    float scale
) {
    kittens::warpgroup::apply(
        accumulator,
        accumulator,
        [scale] __device__ (int /*row*/, int /*col*/, float pre) { return pre * scale; });
}

/**
 *  @brief Accumulate column 0 of a `[64, 16]` fp32 register tile into the
 *         `[n_atoms]` global energy buffer with row-wise atomicAdd.
 *
 *  Stages the accumulator to a fp32 shared-mem tile so we can pick out
 *  column 0 row by row with one `atomicAdd` per atom.  Out-of-range tail
 *  rows (`atom_index >= n_atoms`) are skipped.
 *
 *  We use a fp32 staging tile rather than the existing bf16 hidden tile
 *  because the per-atom energy must round only at the final accumulation —
 *  a bf16 staging step would clip ~0.4% of the per-atom value before the
 *  add, which compounds across both readouts and the energy reduce.
 */
__device__ __forceinline__ void accumulate_column_zero_into_global_(
    float *node_energy_out,
    mace_tk::shared_tile_fp32<atom_tile_rows_k, output_tile_cols_k> &staging_shared_tile,
    register_accumulator_t const &accumulator,
    std::uint32_t n_atoms,
    std::uint32_t base_atom_index
) {
    kittens::warpgroup::store(staging_shared_tile, accumulator);
    kittens::warpgroup::sync(0);

    int const thread_id = threadIdx.x;
    if (thread_id < atom_tile_rows_k) {
        std::uint32_t const atom_index = base_atom_index + static_cast<std::uint32_t>(thread_id);
        if (atom_index < n_atoms) {
            float const energy_contribution = staging_shared_tile[{thread_id, 0}];
            atomicAdd(node_energy_out + atom_index, energy_contribution);
        }
    }
}

/**
 *  @brief Persistent linear-readout kernel: 64 atoms per CTA, one WGMMA.
 *
 *  Grid: `dim3(n_atom_tiles, 1, 1)` where `n_atom_tiles = ceil(n_atoms/64)`.
 *  Block: 128 threads (one warpgroup, 4 warps).  Mirrors the radial-MLP
 *  kernel's single-warpgroup persistent shape.
 */
inline __global__ void mace_tk_readout_linear_kernel(
    std::uint32_t n_atoms,
    float const *__restrict__ node_scalar_invariants,
    float const *__restrict__ readout_weight,
    float *__restrict__ node_energy_out,
    float readout_path_scale
) {
    extern __shared__ kittens::alignment_dummy raw_shared_memory[];
    kittens::shared_allocator shared_alloc(reinterpret_cast<int *>(&raw_shared_memory[0]));

    shared_tile_invariants_t &shared_invariants_tile = shared_alloc.allocate<shared_tile_invariants_t>();
    shared_tile_w_linear_t   &shared_weight_tile    = shared_alloc.allocate<shared_tile_w_linear_t>();
    auto                     &shared_staging_tile_fp32 = shared_alloc.allocate<
        mace_tk::shared_tile_fp32<atom_tile_rows_k, output_tile_cols_k>>();

    std::uint32_t const base_atom_index = blockIdx.x * static_cast<std::uint32_t>(atom_tile_rows_k);
    if (base_atom_index >= n_atoms) return;

    load_invariants_tile_(shared_invariants_tile, node_scalar_invariants, n_atoms, base_atom_index);
    load_linear_readout_weight_padded_(shared_weight_tile, readout_weight);
    kittens::warpgroup::sync(0);

    register_accumulator_t accumulator;
    kittens::warp::zero(accumulator);
    kittens::warpgroup::mm_AB(accumulator, shared_invariants_tile, shared_weight_tile);
    kittens::warpgroup::mma_async_wait();

    scale_accumulator_in_place_(accumulator, readout_path_scale);

    accumulate_column_zero_into_global_(
        node_energy_out, shared_staging_tile_fp32, accumulator, n_atoms, base_atom_index);
}

/**
 *  @brief Persistent nonlinear-MLP-readout kernel: 64 atoms per CTA, two
 *         fused WGMMA stages with a shared-memory hand-off in between.
 *
 *  Stage 1: `[64, 96] @ [96, 16] -> [64, 16]` then `silu_normalize2mom * sc1`.
 *  Stage 2: `[64, 16] @ [16, 16]_padded -> [64, 16]` then `* sc2` and
 *           atomicAdd column 0 into `node_energy_out`.
 *
 *  The hidden activation is converted fp32 -> bf16 via
 *  `kittens::warpgroup::store(shared_hidden, accumulator)` between stages,
 *  then re-fed as the A tile of the second matmul.  No DRAM round-trip on
 *  the hidden vector.
 */
inline __global__ void mace_tk_readout_nonlinear_mlp_kernel(
    std::uint32_t n_atoms,
    float const *__restrict__ node_scalar_invariants,
    float const *__restrict__ mlp_w1_weight,
    float const *__restrict__ mlp_w2_weight,
    float *__restrict__ node_energy_out,
    float pre_activation_scale,    // 1 / sqrt(n_channels)
    float output_scale             // 1 / sqrt(hidden_dim)
) {
    extern __shared__ kittens::alignment_dummy raw_shared_memory[];
    kittens::shared_allocator shared_alloc(reinterpret_cast<int *>(&raw_shared_memory[0]));

    shared_tile_invariants_t &shared_invariants_tile = shared_alloc.allocate<shared_tile_invariants_t>();
    shared_tile_w_mlp_1_t    &shared_w1_tile         = shared_alloc.allocate<shared_tile_w_mlp_1_t>();
    shared_tile_hidden_t     &shared_hidden_tile     = shared_alloc.allocate<shared_tile_hidden_t>();
    shared_tile_w_mlp_2_t    &shared_w2_tile         = shared_alloc.allocate<shared_tile_w_mlp_2_t>();
    auto                     &shared_staging_tile_fp32 = shared_alloc.allocate<
        mace_tk::shared_tile_fp32<atom_tile_rows_k, output_tile_cols_k>>();

    std::uint32_t const base_atom_index = blockIdx.x * static_cast<std::uint32_t>(atom_tile_rows_k);
    if (base_atom_index >= n_atoms) return;

    load_invariants_tile_(shared_invariants_tile, node_scalar_invariants, n_atoms, base_atom_index);
    load_nonlinear_w1_weight_(shared_w1_tile, mlp_w1_weight);
    load_nonlinear_w2_weight_padded_(shared_w2_tile, mlp_w2_weight);
    kittens::warpgroup::sync(0);

    // Stage 1: invariants @ W1 -> hidden, with silu_normalize2mom(pre * sc1).
    register_accumulator_t accumulator;
    kittens::warp::zero(accumulator);
    kittens::warpgroup::mm_AB(accumulator, shared_invariants_tile, shared_w1_tile);
    kittens::warpgroup::mma_async_wait();

    apply_silu_normalize2mom_with_scale_in_place_(accumulator, pre_activation_scale);

    // Hand-off: convert fp32 register tile to bf16 shared, feed into stage 2.
    kittens::warpgroup::store(shared_hidden_tile, accumulator);
    kittens::warpgroup::sync(0);

    // Stage 2: hidden @ W2 -> energy contribution, with outer 1/sqrt(hidden).
    kittens::warp::zero(accumulator);
    kittens::warpgroup::mm_AB(accumulator, shared_hidden_tile, shared_w2_tile);
    kittens::warpgroup::mma_async_wait();

    scale_accumulator_in_place_(accumulator, output_scale);

    accumulate_column_zero_into_global_(
        node_energy_out, shared_staging_tile_fp32, accumulator, n_atoms, base_atom_index);
}

} // namespace mace_tk_readout_detail_

/**
 *  @brief Launch the TK-fused linear readout.
 *
 *  Reads `node_scalar_invariants[n_atoms, n_channels]` and `readout_weight
 *  [n_channels]` (both fp32); accumulates `(1/sqrt(n_channels)) * Σ_c
 *  W[c] * inv[a, c]` into `node_energy_out[n_atoms]` via atomicAdd, matching
 *  the existing fp32 reference contract.
 *
 *  This launcher is currently specialised to MACE-OFF small / medium /
 *  large by the channel count baked into the kernel template; the small
 *  variant (n_channels = 96) is the only one exercised by the smoke test
 *  here.  Extending to other channel counts means adding a dispatch
 *  switch in this function and an extra template instantiation.
 *
 *  Gated on `SMALLFIELD_USE_THUNDERKITTENS`.  When the macro is off,
 *  callers must use the fp32 reference path in `mace_inference.cuh`.
 */
/// Returns true on successful launch, false if the channel count isn't
/// handled (caller must fall back to the fp32 reference path).
inline bool launch_mace_tk_readout_linear(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    float const *node_scalar_invariants_fp32,
    float const *readout_weight_fp32,
    float *node_energy_out_fp32,
    cudaStream_t stream
) {
    if (n_atoms == 0u) return true;
    if (n_channels != static_cast<std::uint32_t>(mace_tk_readout_detail_::n_channels_small_k)) {
        return false;
    }

    using namespace mace_tk_readout_detail_;

    float const readout_path_scale = 1.0f / sqrtf(static_cast<float>(n_channels));

    // Shared-mem footprint:
    //   invariants_tile     64 x 96 bf16 = 12 KB
    //   w_linear_padded     96 x 16 bf16 =  3 KB
    //   staging_tile_fp32   64 x 16 fp32 =  4 KB
    //   ----                                ------
    //                                       ~19 KB + alignment slack
    constexpr std::size_t shared_memory_bytes_k = 32 * 1024;

    cudaFuncSetAttribute(
        mace_tk_readout_linear_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_memory_bytes_k));

    std::uint32_t const n_atom_tiles = (n_atoms + static_cast<std::uint32_t>(atom_tile_rows_k) - 1u)
                                       / static_cast<std::uint32_t>(atom_tile_rows_k);
    dim3 const grid(n_atom_tiles, 1, 1);
    dim3 const block(static_cast<std::uint32_t>(warpgroup_thread_count_k), 1, 1);

    mace_tk_readout_linear_kernel<<<grid, block, shared_memory_bytes_k, stream>>>(
        n_atoms,
        node_scalar_invariants_fp32,
        readout_weight_fp32,
        node_energy_out_fp32,
        readout_path_scale);
    return true;
}

/**
 *  @brief Launch the TK-fused nonlinear MLP readout.
 *
 *  Reads `node_scalar_invariants[n_atoms, n_channels]`, `mlp_w1_fp32
 *  [n_channels, hidden_dim]`, and `mlp_w2_fp32[hidden_dim]` (all fp32);
 *  accumulates the MLP output `(1/sqrt(hidden)) * Σ_k W2[k] *
 *  silu_normalize2mom( (1/sqrt(c)) * Σ_c W1[c, k] * inv[a, c] )` into
 *  `node_energy_out[n_atoms]` via atomicAdd.
 *
 *  Requires `n_channels = 96` (MACE-OFF small) and `hidden_dim = 16`; other
 *  values fall through with no launch.  See the linear launcher for the
 *  channel-count extension story.
 *
 *  Gated on `SMALLFIELD_USE_THUNDERKITTENS`.
 */
/// Returns true on successful launch, false if the channel/hidden tuple
/// isn't handled (caller must fall back to the fp32 reference path).
inline bool launch_mace_tk_readout_nonlinear_mlp(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t hidden_dim,
    float const *node_scalar_invariants_fp32,
    float const *mlp_w1_fp32,
    float const *mlp_w2_fp32,
    float *node_energy_out_fp32,
    cudaStream_t stream
) {
    if (n_atoms == 0u) return true;
    if (n_channels != static_cast<std::uint32_t>(mace_tk_readout_detail_::n_channels_small_k)) return false;
    if (hidden_dim != static_cast<std::uint32_t>(mace_tk_readout_detail_::nonlinear_hidden_dim_k)) return false;

    using namespace mace_tk_readout_detail_;

    float const pre_activation_scale = 1.0f / sqrtf(static_cast<float>(n_channels));
    float const output_scale         = 1.0f / sqrtf(static_cast<float>(hidden_dim));

    // Shared-mem footprint:
    //   invariants_tile     64 x 96  bf16 = 12 KB
    //   w1_tile             96 x 16  bf16 =  3 KB
    //   hidden_tile         64 x 16  bf16 =  2 KB
    //   w2_tile_padded      16 x 16  bf16 =  ~0.5 KB
    //   staging_tile_fp32   64 x 16  fp32 =  4 KB
    //   ----                                  ------
    //                                         ~22 KB + alignment slack
    constexpr std::size_t shared_memory_bytes_k = 48 * 1024;

    cudaFuncSetAttribute(
        mace_tk_readout_nonlinear_mlp_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_memory_bytes_k));

    std::uint32_t const n_atom_tiles = (n_atoms + static_cast<std::uint32_t>(atom_tile_rows_k) - 1u)
                                       / static_cast<std::uint32_t>(atom_tile_rows_k);
    dim3 const grid(n_atom_tiles, 1, 1);
    dim3 const block(static_cast<std::uint32_t>(warpgroup_thread_count_k), 1, 1);

    mace_tk_readout_nonlinear_mlp_kernel<<<grid, block, shared_memory_bytes_k, stream>>>(
        n_atoms,
        node_scalar_invariants_fp32,
        mlp_w1_fp32,
        mlp_w2_fp32,
        node_energy_out_fp32,
        pre_activation_scale,
        output_scale);
    return true;
}

} // namespace smallfield::cuda

#endif // SMALLFIELD_USE_THUNDERKITTENS && __CUDACC__
