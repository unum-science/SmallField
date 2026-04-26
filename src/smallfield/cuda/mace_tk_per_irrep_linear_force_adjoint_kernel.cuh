/**
 *  @file mace_tk_per_irrep_linear_force_adjoint_kernel.cuh
 *  @brief ThunderKittens-fused per-irrep linear force adjoint (D1) for MACE-OFF.
 *
 *  Adjoint of `mace_linear_per_irrep_kernel` / `mace_tk_per_irrep_linear_kernel`.
 *
 *      Forward: out[a, c_out, m] = (1 / sqrt(c_in)) * Σ_{c_in} W[L_b, c_in, c_out] * in[a, c_in, m]
 *      Reverse: din[a, c_in, m]  = (1 / sqrt(c_in)) * Σ_{c_out} W[L_b, c_in, c_out] * dout[a, c_out, m]
 *
 *  This is the same matmul as the forward but with the c_in / c_out tile
 *  roles swapped — the same `[c_in, c_out]` weight slab is read and the inner
 *  reduction is over c_out instead of c_in.  In WGMMA terms we keep loading
 *  the weight tile in its natural `[c_in, c_out]` layout and use `mm_ABt`
 *  (A times B-transposed) instead of `mm_AB` (A times B):
 *
 *    - Forward `mm_AB(D, A, B)`:
 *        A = `[64 = 4 atoms × 16 m, c_in]` (transposed per-atom input slab)
 *        B = `[c_in, c_out]`
 *        D = `[64, c_out]`  (atom-major m × c_out)
 *
 *    - Adjoint `mm_ABt(D, A, B)`:
 *        A = `[64 = 4 atoms × 16 m, c_out]` (transposed per-atom upstream slab)
 *        B = `[c_in, c_out]`           (same memory layout as forward)
 *        D = `[64, c_in]`  (atom-major m × c_in)
 *
 *  TK's `mm_ABt` interprets `B` with shape `[N, K]` and computes
 *  `D[m, n] = Σ_k A[m, k] * B[n, k]` — so feeding the natural `[c_in, c_out]`
 *  weight slab as `B` produces `D[m, c_in] = Σ_{c_out} A[m, c_out] * W[c_in, c_out]`,
 *  exactly matching the closed-form adjoint.  No new weight transpose / staging
 *  pass is needed; we reuse the forward's weight load helper unchanged.
 *
 *  Per the public ABI (matches the fp32 reference signature in
 *  `mace_force_adjoints.cuh::launch_mace_linear_per_irrep_force_adjoint`), the
 *  output is written, NOT accumulated — the caller is responsible for any
 *  upstream zeroing or sum-with-other-adjoint composition.  This deliberately
 *  differs from the fp32 reference's `+=` convention because bf16 stores at
 *  this precision tier do not reliably preserve a previously written gradient
 *  contribution; downstream consumers that need accumulation should cast to
 *  fp32 first or use the fp32 reference path.
 *
 *  Shape constants are baked in for MACE-OFF-small/medium (c_in = c_out = 96).
 *  Larger configurations (c = 192) need a separate specialization.
 *
 *  Validation: a standalone smoke test in `/tmp/test_tk_force_adjoints.cu`
 *  compares this kernel against `launch_mace_linear_per_irrep_force_adjoint`
 *  (fp32) on a 64-atom synthetic batch with `active_irrep_block_count = 4`.
 *  Target: relative max-abs ≤ 1 bf16 ulp at the output magnitude.
 *
 *  Gated on `SMALLFIELD_USE_THUNDERKITTENS`; when the macro is off this file
 *  is empty and the fp32 reference path stays the only D1 adjoint.
 */

#pragma once

#include "smallfield/cuda/mace_tk_tile.cuh"

#if defined(SMALLFIELD_USE_THUNDERKITTENS) && defined(__CUDACC__)

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace smallfield::cuda {

namespace mace_tk_per_irrep_linear_force_adjoint_detail_ {

// Compile-time shape constants — mirror the forward TK kernel.  WGMMA M = 64
// (= 4 atoms × 16 m); K (= c_out for the adjoint matmul) and N (= c_in for the
// adjoint matmul) are 96 for MACE-OFF-small.  We bake the small/medium shapes
// in here because the production deployment only uses these widths; larger
// configurations (c = 192) need a separate specialization.
constexpr int atoms_per_cta_k          = 4;
constexpr int irrep_dim_total_k        = mace_tk::mace_irrep_dim_total_k;     // 16
constexpr int wgmma_m_rows_k           = atoms_per_cta_k * irrep_dim_total_k; // 64
constexpr int channels_in_k            = mace_tk::mace_channels_small_k;      // 96
constexpr int channels_out_k           = mace_tk::mace_channels_small_k;      // 96
constexpr int irrep_block_count_k      = 4;                                    // L = 0, 1, 2, 3
constexpr int warpgroup_thread_count_k = kittens::WARPGROUP_THREADS;           // 128

// Per-block m-range.  Indices into the irrep_dim_total = 16 axis that
// correspond to L = `irrep_block_index`.  Same canonical L mapping as the
// forward kernel: L=0 → m=0; L=1 → m=1..3; L=2 → m=4..8; L=3 → m=9..15.
__device__ __forceinline__ int irrep_block_m_start_(int irrep_block_index) {
    constexpr int starts[irrep_block_count_k] = {0, 1, 4, 9};
    return starts[irrep_block_index];
}
__device__ __forceinline__ int irrep_block_m_count_(int irrep_block_index) {
    constexpr int counts[irrep_block_count_k] = {1, 3, 5, 7};
    return counts[irrep_block_index];
}

// Shared-tile aliases.  `shared_upstream_tile_t` holds 4 atoms' upstream
// gradient slabs after transposing each `[c_out, 16]` slab to `[16, c_out]`,
// packed contiguously along the M axis of the WGMMA.  Weights are
// `[c_in, c_out]` per block (same as the forward kernel).  The output
// staging tile holds the per-block adjoint accumulator after fp32 → bf16
// conversion, before we transpose-write back to global.
using shared_upstream_tile_t      = mace_tk::shared_tile_bf16<wgmma_m_rows_k, channels_out_k>;
using shared_weight_tile_t        = mace_tk::shared_tile_bf16<channels_in_k, channels_out_k>;
using shared_grad_input_stage_t   = mace_tk::shared_tile_bf16<wgmma_m_rows_k, channels_in_k>;

// Per-warpgroup register accumulator for the adjoint.  D = `[64, c_in]` fp32,
// distributed across 4 warps × 16 rows each → `rt_fl<16, c_in>`.
using register_accumulator_t = kittens::rt_fl<16, channels_in_k>;

/**
 *  @brief Layer-wide path scale `1 / sqrt(c_in)` (e3nn `o3.Linear`).
 *
 *  Computed once on the host and folded into the per-block accumulator
 *  before the bf16 store epilogue.  Matches the forward kernel's scale
 *  exactly so the same weight slab can be reused without rescaling.
 */
struct per_irrep_linear_force_adjoint_scales_t {
    float path_weight;
};

/**
 *  @brief Stage 4 atoms' upstream-gradient slabs into one shared `[64, c_out]` bf16 tile.
 *
 *  Reads `[c_out, irrep_dim]` per atom from global fp32 (the upstream gradient
 *  on the forward output) and writes a transposed `[16, c_out]` bf16 sub-tile
 *  starting at M-row = `atom_in_cta * 16`.  Out-of-range atoms zero-fill so
 *  the WGMMA produces zero contributions which the per-atom store path then
 *  ignores via the same atom-index guard.
 *
 *  m-rows that fall outside the active L-blocks are also zero-filled — the
 *  forward kernel writes 0 there so the upstream gradient is meaningfully
 *  defined only on active m-rows; defensively zeroing the tile keeps any
 *  caller-supplied garbage out of the WGMMA accumulator.
 */
__device__ __forceinline__ void load_upstream_tile_transposed_(
    shared_upstream_tile_t &dst_upstream_tile,
    float const *upstream_grad_output_global,
    std::uint32_t n_atoms,
    std::uint32_t base_atom_index,
    std::uint32_t n_channels_out_runtime,
    std::uint32_t output_irrep_dim_runtime,
    std::uint32_t active_irrep_block_count
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = wgmma_m_rows_k * channels_out_k;
    int const m_active_max = irrep_block_m_start_(static_cast<int>(active_irrep_block_count) - 1)
                             + irrep_block_m_count_(static_cast<int>(active_irrep_block_count) - 1);

    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const m_row_in_tile  = linear_index / channels_out_k;
        int const channel_out_col = linear_index % channels_out_k;
        int const atom_in_cta    = m_row_in_tile / irrep_dim_total_k;
        int const m_in_atom      = m_row_in_tile % irrep_dim_total_k;
        std::uint32_t const atom_index = base_atom_index + static_cast<std::uint32_t>(atom_in_cta);

        float value = 0.0f;
        if (atom_index < n_atoms &&
            static_cast<std::uint32_t>(channel_out_col) < n_channels_out_runtime &&
            static_cast<std::uint32_t>(m_in_atom)       < output_irrep_dim_runtime &&
            m_in_atom < m_active_max) {
            std::uint64_t const offset =
                (static_cast<std::uint64_t>(atom_index) * n_channels_out_runtime
                 + static_cast<std::uint32_t>(channel_out_col)) * output_irrep_dim_runtime
                + static_cast<std::uint32_t>(m_in_atom);
            value = upstream_grad_output_global[offset];
        }
        dst_upstream_tile[{m_row_in_tile, channel_out_col}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Stage one block's `[c_in, c_out]` weight from fp32 global into bf16 shared.
 *
 *  Identical to the forward kernel's weight-load helper — same memory layout,
 *  same `[n_blocks, c_in, c_out]` blob.  Mirrored here as a private helper to
 *  avoid a cross-include of the forward kernel's anonymous detail namespace.
 */
__device__ __forceinline__ void load_block_weight_(
    shared_weight_tile_t &dst_weight_tile,
    float const *weight_blocks_global,
    std::uint32_t irrep_block_index,
    std::uint32_t n_channels_in_runtime,
    std::uint32_t n_channels_out_runtime
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = channels_in_k * channels_out_k;
    std::uint64_t const block_base = static_cast<std::uint64_t>(irrep_block_index)
                                     * n_channels_in_runtime * n_channels_out_runtime;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const channel_in_row  = linear_index / channels_out_k;
        int const channel_out_col = linear_index % channels_out_k;
        float value = 0.0f;
        if (static_cast<std::uint32_t>(channel_in_row)  < n_channels_in_runtime &&
            static_cast<std::uint32_t>(channel_out_col) < n_channels_out_runtime) {
            std::uint64_t const offset = block_base
                + static_cast<std::uint64_t>(channel_in_row) * n_channels_out_runtime
                + static_cast<std::uint32_t>(channel_out_col);
            value = weight_blocks_global[offset];
        }
        dst_weight_tile[{channel_in_row, channel_out_col}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Apply the e3nn `1 / sqrt(c_in)` path-weight scale to the accumulator.
 *
 *  Fused via `kittens::warpgroup::apply` on every (row, col) lane in the
 *  fp32 register tile.  Same scale as the forward — comes from the e3nn
 *  one-instruction-per-output `o3.Linear` path-weight convention.
 */
__device__ __forceinline__ void scale_accumulator_in_place_(
    register_accumulator_t &accumulator,
    float path_weight
) {
    kittens::warpgroup::apply(
        accumulator,
        accumulator,
        [path_weight] __device__ (int /*row*/, int /*col*/, float pre) {
            return pre * path_weight;
        });
}

/**
 *  @brief Transpose-and-write one block's `[64, c_in]` adjoint slab to
 *         global `[n_atoms, c_in, irrep_dim]`, masking m-rows outside the block.
 *
 *  The shared bf16 staging tile already holds the post-scale post-conversion
 *  accumulator (from `kittens::warpgroup::store(st, rt)`).  We then walk
 *  `(atom_in_cta, c_in, m_in_block)` triples cooperatively across the
 *  warpgroup and emit one bf16 element each — `m_in_block` maps back to
 *  the absolute m-position via `irrep_block_m_start_(block)`.
 *
 *  Output is overwritten (not accumulated) to match the public ABI.
 */
__device__ __forceinline__ void store_block_grad_input_transposed_(
    __nv_bfloat16 *grad_input_global,
    shared_grad_input_stage_t const &staged_grad_input_tile,
    std::uint32_t n_atoms,
    std::uint32_t base_atom_index,
    std::uint32_t n_channels_in_runtime,
    std::uint32_t irrep_dim_runtime,
    int irrep_block_index
) {
    int const thread_id = threadIdx.x;
    int const m_start   = irrep_block_m_start_(irrep_block_index);
    int const m_count   = irrep_block_m_count_(irrep_block_index);
    int const total_elements = atoms_per_cta_k * static_cast<int>(n_channels_in_runtime) * m_count;

    #pragma unroll
    for (int linear_index = thread_id; linear_index < total_elements; linear_index += warpgroup_thread_count_k) {
        int const atom_in_cta = linear_index / (static_cast<int>(n_channels_in_runtime) * m_count);
        int const within_atom = linear_index % (static_cast<int>(n_channels_in_runtime) * m_count);
        int const channel_in  = within_atom / m_count;
        int const m_in_block  = within_atom % m_count;
        int const m_in_atom   = m_start + m_in_block;
        std::uint32_t const atom_index = base_atom_index + static_cast<std::uint32_t>(atom_in_cta);
        if (atom_index >= n_atoms) continue;

        int const m_row_in_tile = atom_in_cta * irrep_dim_total_k + m_in_atom;
        __nv_bfloat16 const value = staged_grad_input_tile[{m_row_in_tile, channel_in}];

        std::uint64_t const out_offset =
            (static_cast<std::uint64_t>(atom_index) * n_channels_in_runtime
             + static_cast<std::uint32_t>(channel_in)) * irrep_dim_runtime
            + static_cast<std::uint32_t>(m_in_atom);
        grad_input_global[out_offset] = value;
    }
}

/**
 *  @brief Zero the m-rows that belong to inactive L blocks in the grad input.
 *
 *  When `active_irrep_block_count < 4` the higher-L m-rows of the grad input
 *  must be zero so downstream consumers see a clean tensor, mirroring the
 *  forward kernel's `output[...] = 0.0f` early-return for inactive blocks.
 */
__device__ __forceinline__ void zero_inactive_block_m_rows_(
    __nv_bfloat16 *grad_input_global,
    std::uint32_t n_atoms,
    std::uint32_t base_atom_index,
    std::uint32_t n_channels_in_runtime,
    std::uint32_t irrep_dim_runtime,
    std::uint32_t active_irrep_block_count
) {
    int const thread_id = threadIdx.x;
    int const inactive_block_index = static_cast<int>(active_irrep_block_count);
    if (inactive_block_index >= irrep_block_count_k) return;

    int const m_inactive_start = irrep_block_m_start_(inactive_block_index);
    int const m_inactive_count = static_cast<int>(irrep_dim_runtime) - m_inactive_start;
    if (m_inactive_count <= 0) return;

    int const total_elements = atoms_per_cta_k * static_cast<int>(n_channels_in_runtime) * m_inactive_count;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < total_elements; linear_index += warpgroup_thread_count_k) {
        int const atom_in_cta = linear_index / (static_cast<int>(n_channels_in_runtime) * m_inactive_count);
        int const within_atom = linear_index % (static_cast<int>(n_channels_in_runtime) * m_inactive_count);
        int const channel_in  = within_atom / m_inactive_count;
        int const m_offset    = within_atom % m_inactive_count;
        int const m_in_atom   = m_inactive_start + m_offset;
        std::uint32_t const atom_index = base_atom_index + static_cast<std::uint32_t>(atom_in_cta);
        if (atom_index >= n_atoms) continue;
        std::uint64_t const out_offset =
            (static_cast<std::uint64_t>(atom_index) * n_channels_in_runtime
             + static_cast<std::uint32_t>(channel_in)) * irrep_dim_runtime
            + static_cast<std::uint32_t>(m_in_atom);
        grad_input_global[out_offset] = __float2bfloat16_rn(0.0f);
    }
}

/**
 *  @brief Persistent per-irrep linear adjoint kernel: 4 atoms / CTA, 4 blocks fused.
 *
 *  Grid: `dim3(n_atom_tiles, 1, 1)` with `n_atom_tiles = ceil(n_atoms / 4)`.
 *  Block: 128 threads (one warpgroup, 4 warps).  Within a CTA we run one
 *  WGMMA-with-B-transposed per active L block, sharing the upstream-gradient
 *  shared tile across all blocks.  Inactive blocks (when
 *  `active_irrep_block_count < 4`) have their m-rows zeroed in the output
 *  rather than computed.
 */
__global__ void mace_tk_per_irrep_linear_force_adjoint_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_channels_out,
    std::uint32_t output_irrep_dim_runtime,
    std::uint32_t grad_input_irrep_dim_runtime,
    std::uint32_t active_irrep_block_count,
    float const *__restrict__ weight_blocks_global,
    float const *__restrict__ upstream_grad_output_global,
    __nv_bfloat16 *__restrict__ grad_input_global,
    per_irrep_linear_force_adjoint_scales_t scales
) {
    extern __shared__ kittens::alignment_dummy raw_shared_memory[];
    kittens::shared_allocator shared_alloc(reinterpret_cast<int *>(&raw_shared_memory[0]));

    // Persistent shared tiles.  Upstream gradient tile is shared across all 4
    // blocks (loaded once per CTA, reused per block); weight tile is refilled
    // per block; grad-input staging tile is reused per block as the bf16
    // conversion target before the transposed store.
    shared_upstream_tile_t    &shared_upstream_tile = shared_alloc.allocate<shared_upstream_tile_t>();
    shared_weight_tile_t      &shared_weight_tile   = shared_alloc.allocate<shared_weight_tile_t>();
    shared_grad_input_stage_t &shared_stage_tile    = shared_alloc.allocate<shared_grad_input_stage_t>();

    std::uint32_t const base_atom_index = blockIdx.x * static_cast<std::uint32_t>(atoms_per_cta_k);
    if (base_atom_index >= n_atoms) return;

    // Stage the per-CTA upstream gradient slab once; reused across all L blocks.
    load_upstream_tile_transposed_(
        shared_upstream_tile,
        upstream_grad_output_global,
        n_atoms, base_atom_index,
        n_channels_out, output_irrep_dim_runtime,
        active_irrep_block_count);
    kittens::warpgroup::sync(0);

    // Zero inactive-block m-rows up front so subsequent active-block stores
    // overwrite cleanly without a dependency on the previous block's launch.
    zero_inactive_block_m_rows_(
        grad_input_global,
        n_atoms, base_atom_index,
        n_channels_in, grad_input_irrep_dim_runtime,
        active_irrep_block_count);

    register_accumulator_t accumulator;

    // ---- One WGMMA-with-B-transposed per active L block ----
    #pragma unroll
    for (int irrep_block_index = 0; irrep_block_index < irrep_block_count_k; ++irrep_block_index) {
        if (static_cast<std::uint32_t>(irrep_block_index) >= active_irrep_block_count) break;

        // Stage this block's `[c_in, c_out]` weight tile (same layout as forward).
        load_block_weight_(
            shared_weight_tile,
            weight_blocks_global,
            static_cast<std::uint32_t>(irrep_block_index),
            n_channels_in, n_channels_out);
        kittens::warpgroup::sync(0);

        // D = A * B^T with A = `[64, c_out]` (transposed upstream), B =
        // `[c_in, c_out]` (weights), D = `[64, c_in]` (atom-major m × c_in).
        // `mm_ABt` interprets B as `[N=c_in, K=c_out]`, computing
        // `D[m, c_in] = Σ_{c_out} A[m, c_out] * B[c_in, c_out]` — exactly the
        // closed-form D1 adjoint sum.
        kittens::warp::zero(accumulator);
        kittens::warpgroup::mm_ABt(accumulator, shared_upstream_tile, shared_weight_tile);
        kittens::warpgroup::mma_async_wait();

        scale_accumulator_in_place_(accumulator, scales.path_weight);

        // fp32 → bf16 staging in shared memory; then transpose-store the
        // m-rows of this block to global `[n_atoms, c_in, irrep_dim]`.
        kittens::warpgroup::store(shared_stage_tile, accumulator);
        kittens::warpgroup::sync(0);

        store_block_grad_input_transposed_(
            grad_input_global,
            shared_stage_tile,
            n_atoms, base_atom_index,
            n_channels_in, grad_input_irrep_dim_runtime,
            irrep_block_index);
        kittens::warpgroup::sync(0);
    }
}

} // namespace mace_tk_per_irrep_linear_force_adjoint_detail_

/**
 *  @brief Launch the TK-fused per-irrep linear force adjoint (D1) for MACE-OFF.
 *
 *  Reads `weight_blocks_fp32[n_blocks, c_in, c_out]` (the same e3nn `o3.Linear`
 *  per-instruction weight blob the forward consumes) and
 *  `upstream_grad_output_fp32[n_atoms, c_out, 16]` (the upstream gradient on
 *  the forward output, typically already cast from bf16); writes
 *  `grad_input_bf16[n_atoms, c_in, 16]`.  Applies `path_weight = 1 / sqrt(c_in)`
 *  uniformly per block.  When `active_irrep_block_count < 4` the higher-L
 *  m-rows of the grad input are zeroed so downstream consumers see a clean
 *  tensor.  Output is written, NOT accumulated — caller must zero-init or
 *  cast-then-add if downstream needs accumulation across multiple adjoint paths.
 *
 *  Threading: `n_atom_tiles = ceil(n_atoms / 4)` CTAs, 128 threads each
 *  (one warpgroup).  WGMMA shape: M = 64 (4 atoms × 16 m), K = c_out, N = c_in.
 *
 *  Gated on `SMALLFIELD_USE_THUNDERKITTENS`.  When the macro is off the
 *  driver falls back to `launch_mace_linear_per_irrep_force_adjoint` in
 *  `mace_force_adjoints.cuh`.
 */
inline void launch_mace_tk_per_irrep_linear_force_adjoint(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_channels_out,
    std::uint32_t active_irrep_block_count,
    float const *weight_blocks_fp32,
    float const *upstream_grad_output_fp32,
    __nv_bfloat16 *grad_input_bf16,
    cudaStream_t stream
) {
    if (n_atoms == 0u) return;

    using namespace mace_tk_per_irrep_linear_force_adjoint_detail_;

    // The fused adjoint templates the small/medium MACE-OFF shape
    // (c_in = c_out = 96) into the WGMMA tile.  Other configurations need a
    // separate specialization; refuse rather than silently producing
    // truncated output.
    if (n_channels_in  != static_cast<std::uint32_t>(channels_in_k) ||
        n_channels_out != static_cast<std::uint32_t>(channels_out_k)) {
        return;
    }
    if (active_irrep_block_count == 0u ||
        active_irrep_block_count > static_cast<std::uint32_t>(irrep_block_count_k)) {
        return;
    }

    per_irrep_linear_force_adjoint_scales_t scales;
    scales.path_weight = 1.0f / sqrtf(static_cast<float>(n_channels_in));

    // Shared-mem footprint, one of each tile type:
    //   shared_upstream_tile   64 x 96 bf16 = 12 KB
    //   shared_weight_tile     96 x 96 bf16 = 18 KB
    //   shared_stage_tile      64 x 96 bf16 = 12 KB
    //   ----                                 ------
    //                                        ~42 KB + alignment slack
    constexpr std::size_t shared_memory_bytes_k = 64 * 1024;  // matches forward's budget

    cudaFuncSetAttribute(
        mace_tk_per_irrep_linear_force_adjoint_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_memory_bytes_k));

    // The upstream gradient and grad-input both live on the full
    // `[n_atoms, c, 16]` layout — the per-irrep adjoint never operates on a
    // truncated `[n_atoms, c, 1]` slab the way the forward `linear_up` /
    // `product_linear` callers do, because the upstream gradient flowing into
    // `linear_up` from the symmetric contraction is already 16-wide and the
    // grad-input on the input side likewise lives in the 16-wide layout.
    constexpr std::uint32_t output_irrep_dim_runtime     = static_cast<std::uint32_t>(irrep_dim_total_k);
    constexpr std::uint32_t grad_input_irrep_dim_runtime = static_cast<std::uint32_t>(irrep_dim_total_k);

    std::uint32_t const n_atom_tiles = (n_atoms + static_cast<std::uint32_t>(atoms_per_cta_k) - 1u)
                                       / static_cast<std::uint32_t>(atoms_per_cta_k);
    dim3 const grid(n_atom_tiles, 1, 1);
    dim3 const block(static_cast<std::uint32_t>(warpgroup_thread_count_k), 1, 1);

    mace_tk_per_irrep_linear_force_adjoint_kernel<<<grid, block, shared_memory_bytes_k, stream>>>(
        n_atoms,
        n_channels_in,
        n_channels_out,
        output_irrep_dim_runtime,
        grad_input_irrep_dim_runtime,
        active_irrep_block_count,
        weight_blocks_fp32,
        upstream_grad_output_fp32,
        grad_input_bf16,
        scales);
}

} // namespace smallfield::cuda

#endif // SMALLFIELD_USE_THUNDERKITTENS && __CUDACC__
