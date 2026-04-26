/**
 *  @file mace_tk_element_conditioned_residual_force_adjoint_kernel.cuh
 *  @brief ThunderKittens-fused element-conditioned residual force adjoint (D3) for MACE-OFF.
 *
 *  Adjoint of `mace_tk_element_conditioned_residual_kernel` /
 *  `mace_element_conditioned_residual_kernel` (a.k.a. skip_tp).
 *
 *      Forward: out[a, c_out, m] = (1 / sqrt(c_in * n_elements))
 *                                * Σ_{c_in} W[L_b, c_in, Z[a], c_out] * in[a, c_in, m]
 *      Reverse: din[a, c_in, m] += (1 / sqrt(c_in * n_elements))
 *                                * Σ_{c_out} W[L_b, c_in, Z[a], c_out] * dout[a, c_out, m]
 *
 *  Same Z-indirect weight slab read as the forward — `Z[a] = atom_element[a]`
 *  picks the `(c_in, c_out)` slice from the larger
 *  `[active_irrep_block_count, c_in, n_elements, c_out]` weights blob, and each
 *  active irrep block contributes only into m-components inside its block.
 *  Inactive m-components are left untouched (the caller pre-zeros
 *  `grad_input_fp32` and we accumulate with `+=` so this kernel composes with
 *  upstream adjoint contributions).
 *
 *  Backward WGMMA shape (mirrors the forward but with c_in / c_out roles
 *  swapped — the weight slab is the same memory, just read with a different
 *  M / K interpretation):
 *
 *    - Forward `mm_AtB(D, A, B)`:
 *        A = `[c_in = 96, c_out_tile = 64]` (weight slab, transposed in WGMMA)
 *        B = `[c_in = 96, m = 16]`         (input slab)
 *        D = `[c_out_tile = 64, m = 16]`   (output sub-slab)
 *
 *    - Adjoint `mm_AB(D, A, B)`:
 *        A = `[c_in_tile = 64, c_out = 96]` (weight slab in natural layout)
 *        B = `[c_out = 96, m = 16]`         (upstream gradient slab)
 *        D = `[c_in_tile = 64, m = 16]`     (grad-input sub-slab)
 *
 *  The forward tiles `c_out = 96` into two 64-wide column tiles (with a
 *  zero-padded 32-col tail in the second tile) because WGMMA M = 64.  The
 *  adjoint mirror-images this: `c_in = 96` is the WGMMA M axis here, so we
 *  tile it the same way (two 64-wide row tiles, second tile drops the bottom
 *  32 rows on store).  The K axis of WGMMA is then `c_out = 96` which is
 *  6 × 16 — a clean K, no padding.  The N axis is `m = 16` — exactly one
 *  TK column tile.
 *
 *  Per the public ABI (matches the fp32 reference signature in
 *  `mace_force_adjoints.cuh::launch_mace_element_conditioned_residual_force_adjoint`),
 *  the output is accumulated with `+=` — the caller is responsible for
 *  pre-zeroing `grad_input_fp32` once per backward pass and for any
 *  composition with other adjoint contributions.  We keep the fp32
 *  accumulator on the global side rather than writing bf16 because the
 *  surrounding D2 / D5-7 reverse chain feeds fp32 buffers; a bf16 store
 *  here would force the caller to insert a cast bridge that the rest of
 *  the chain doesn't need.  The matmul itself stays bf16-in / fp32-accum
 *  internally, so the precision tier matches the rest of the TK adjoints.
 *
 *  Shape constants are baked in for MACE-OFF-small/medium (c_in = c_out = 96).
 *  Larger configurations (c = 192) silently fall through and the caller must
 *  use the fp32 reference.
 *
 *  Validation: a standalone smoke test in `/tmp/test_tk_d3_force_adjoint.cu`
 *  compares this kernel against `launch_mace_element_conditioned_residual_force_adjoint`
 *  (fp32) on a 64-atom synthetic batch with `active_irrep_block_count` ∈ {1, 4}.
 *  Target: relative max-abs ≤ 1 bf16 ulp at the output magnitude.
 *
 *  Gated on `SMALLFIELD_USE_THUNDERKITTENS`; when the macro is off this file
 *  is empty and the fp32 reference path stays the only D3 adjoint.
 */

#pragma once

#include "smallfield/cuda/mace_tk_tile.cuh"

#if defined(SMALLFIELD_USE_THUNDERKITTENS) && defined(__CUDACC__)

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace smallfield::cuda {

namespace mace_tk_element_conditioned_residual_force_adjoint_detail_ {

// Compile-time shapes — mirror the forward kernel.  `c_in == c_out == 96` and
// the irrep axis packs as 1 + 3 + 5 + 7 = 16 m-components.  WGMMA on Hopper
// requires the output `M` axis to be exactly 64; we tile `c_in = 96` as two
// 64-wide row tiles (the second tile zero-pads the bottom 32 rows and the
// per-thread global store drops them).
constexpr int channels_in_k             = mace_tk::mace_channels_small_k;        // 96
constexpr int channels_out_k            = mace_tk::mace_channels_small_k;        // 96
constexpr int irrep_dim_total_k         = mace_tk::mace_irrep_dim_total_k;       // 16
constexpr int channels_in_tile_size_k   = 64;                                    // WGMMA M
constexpr int channels_in_tile_count_k  = (channels_in_k + channels_in_tile_size_k - 1)
                                          / channels_in_tile_size_k;             // 2
constexpr int max_active_irrep_block_count_k = 4;
constexpr int warpgroup_thread_count_k  = kittens::WARPGROUP_THREADS;             // 128

// Shared-tile aliases.  Upstream gradient and weight slab live as bf16 in
// shared memory; the WGMMA accumulator stays fp32 in registers and is staged
// to a fp32 shared tile before the masked global `+=` epilogue (we cannot use
// `warpgroup::store` to a fp32 global tile because we need an additive
// epilogue with per-m-component irrep-block masking).
using shared_tile_upstream_t       = mace_tk::shared_tile_bf16<channels_out_k, irrep_dim_total_k>;
using shared_tile_weight_slab_t    = mace_tk::shared_tile_bf16<channels_in_tile_size_k, channels_out_k>;
using shared_tile_grad_input_stage_t = mace_tk::shared_tile_fp32<channels_in_tile_size_k, irrep_dim_total_k>;

// Per-warpgroup register accumulator: 64 rows × 16 cols (one warp per 16
// rows in the M direction).  TK expresses register-tile dims in tile units
// of 16; 16/16 = 1, 16/16 = 1 → `rt_fl<16, 16>` is `rt_fl<height=1, width=1>`
// per warp = 64 rows × 16 cols across the warpgroup of four warps.
using register_accumulator_t = kittens::rt_fl<16, irrep_dim_total_k>;

/**
 *  @brief Map an m-component index `[0, 16)` to its irrep block index.
 *
 *  Mirrors the helper in the forward kernel: L=0 → m=0, L=1 → m=1..3,
 *  L=2 → m=4..8, L=3 → m=9..15.  Local copy keeps this header includable
 *  from host translation units without pulling in another kernel header.
 */
__device__ __forceinline__ unsigned int irrep_block_index_for_component_(unsigned int irrep_component_index) {
    if (irrep_component_index == 0u) return 0u;
    if (irrep_component_index <= 3u) return 1u;
    if (irrep_component_index <= 8u) return 2u;
    return 3u;
}

/**
 *  @brief Stage the per-atom upstream-gradient slab `[c_out, irrep_dim] = [96, 16]`
 *         from fp32 global into bf16 shared memory.
 *
 *  Out-of-bounds atoms (`atom_index >= n_atoms`) are caught at kernel entry
 *  so this loader assumes a valid `atom_index`.  m-components outside the
 *  active irrep blocks are zero-filled defensively — the forward kernel's
 *  output is meaningfully defined only inside active blocks; downstream
 *  consumers may have written zeros there but we don't rely on it.
 */
__device__ __forceinline__ void load_atom_upstream_slab_(
    shared_tile_upstream_t &dst_upstream_tile,
    float const *upstream_grad_output_global,
    std::uint32_t atom_index,
    std::uint32_t active_irrep_block_count
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = channels_out_k * irrep_dim_total_k;
    std::uint64_t const atom_upstream_offset =
        static_cast<std::uint64_t>(atom_index) * channels_out_k * irrep_dim_total_k;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const channel_out_index    = linear_index / irrep_dim_total_k;  // c_out
        int const irrep_component_index = linear_index % irrep_dim_total_k;  // m
        unsigned int const irrep_block_index = irrep_block_index_for_component_(
            static_cast<unsigned int>(irrep_component_index));
        float value = 0.0f;
        if (irrep_block_index < active_irrep_block_count) {
            value = upstream_grad_output_global[atom_upstream_offset
                                                + static_cast<std::uint64_t>(channel_out_index) * irrep_dim_total_k
                                                + irrep_component_index];
        }
        dst_upstream_tile[{channel_out_index, irrep_component_index}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Stage one Z-indirect weight slab `[c_in_tile = 64, c_out = 96]`
 *         for `(irrep_block_index, atom_element_index)` from the global
 *         fp32 weights blob into bf16 shared memory.
 *
 *  Source layout is `weights[L_b, c_in, Z, c_out]` row-major; we read the
 *  64-row sub-slice
 *  `weights[L_b, c_in_tile_base + 0..63, Z, :]`.  Rows past `c_in` (the
 *  bottom 32 of the second tile) are zero-padded so the WGMMA picks up
 *  zeros there and the per-thread global `+=` skips those rows entirely.
 */
__device__ __forceinline__ void load_z_indirect_weight_slab_(
    shared_tile_weight_slab_t &dst_weight_tile,
    float const *weights_global,
    std::uint32_t n_elements,
    std::uint32_t n_channels_out,
    unsigned int irrep_block_index,
    int atom_element_index,
    int channels_in_tile_index
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = channels_in_tile_size_k * channels_out_k;
    std::uint64_t const block_stride =
        static_cast<std::uint64_t>(channels_in_k) * n_elements * n_channels_out;
    std::uint64_t const channel_in_stride = static_cast<std::uint64_t>(n_elements) * n_channels_out;
    std::uint64_t const block_base =
        static_cast<std::uint64_t>(irrep_block_index) * block_stride
        + static_cast<std::uint64_t>(atom_element_index) * n_channels_out;
    int const channels_in_tile_base = channels_in_tile_index * channels_in_tile_size_k;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile  = linear_index / channels_out_k;  // local c_in
        int const col_in_tile  = linear_index % channels_out_k;  // c_out
        int const channel_in_global = channels_in_tile_base + row_in_tile;
        float value = 0.0f;
        if (channel_in_global < channels_in_k &&
            static_cast<std::uint32_t>(col_in_tile) < n_channels_out) {
            value = weights_global[block_base
                                    + static_cast<std::uint64_t>(channel_in_global) * channel_in_stride
                                    + static_cast<std::uint64_t>(col_in_tile)];
        }
        dst_weight_tile[{row_in_tile, col_in_tile}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Apply the e3nn `path_weight = 1/sqrt(c_in * n_elements)` scale
 *         to a fp32 register tile in place.
 */
__device__ __forceinline__ void scale_accumulator_in_place_(
    register_accumulator_t &accumulator,
    float scale
) {
    kittens::warpgroup::apply(
        accumulator,
        accumulator,
        [scale] __device__ (int /*row*/, int /*col*/, float pre_value) { return pre_value * scale; });
}

/**
 *  @brief Add the fp32 register accumulator into the global grad-input slab
 *         with per-m-component irrep-block masking.
 *
 *  We can't use `warpgroup::store(gl, rt, coord)` because (a) the output is
 *  additive (`+=`, the public ABI of the D3 adjoint), (b) the second
 *  `c_in` tile is partial (32 valid rows of 64) and (c) we must add only
 *  into m-components that belong to the current irrep block — the other
 *  m-components are owned by a different block iteration (or are outside
 *  the active set entirely).  We stage the register tile to a fp32 shared
 *  tile and then do guarded per-thread global `+=`.
 *
 *  The path-weight scale is folded into the register accumulator before
 *  this epilogue, so the value written is the post-scale gradient.
 */
__device__ __forceinline__ void add_irrep_block_into_grad_input_(
    float *grad_input_global,
    shared_tile_grad_input_stage_t &staging_shared_tile,
    register_accumulator_t const &accumulator,
    std::uint32_t atom_index,
    std::uint32_t n_channels_in,
    unsigned int irrep_block_index,
    int channels_in_tile_index
) {
    // Convert fp32 register tile to fp32 shared.
    kittens::warpgroup::store(staging_shared_tile, accumulator);
    kittens::warpgroup::sync(0);

    int const thread_id = threadIdx.x;
    constexpr int tile_elements = channels_in_tile_size_k * irrep_dim_total_k;
    int const channels_in_tile_base = channels_in_tile_index * channels_in_tile_size_k;
    std::uint64_t const atom_grad_input_offset =
        static_cast<std::uint64_t>(atom_index) * n_channels_in * irrep_dim_total_k;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile           = linear_index / irrep_dim_total_k;  // local c_in
        int const irrep_component_index = linear_index % irrep_dim_total_k;  // m
        int const channel_in_global     = channels_in_tile_base + row_in_tile;
        if (channel_in_global >= static_cast<int>(n_channels_in)) continue;

        unsigned int const m_block_index = irrep_block_index_for_component_(
            static_cast<unsigned int>(irrep_component_index));
        if (m_block_index != irrep_block_index) continue;

        std::uint64_t const grad_input_addr =
            atom_grad_input_offset
            + static_cast<std::uint64_t>(channel_in_global) * irrep_dim_total_k
            + static_cast<std::uint64_t>(irrep_component_index);
        grad_input_global[grad_input_addr] += staging_shared_tile[{row_in_tile, irrep_component_index}];
    }
}

/**
 *  @brief Persistent kernel body: one CTA per atom, one warpgroup per CTA.
 *
 *  Grid: `dim3(n_atoms, 1, 1)`.  Block: 128 threads (one warpgroup).
 *  Per atom:
 *    - read `Z[atom]` once,
 *    - stage the upstream-gradient slab `[c_out, m]` once (zero-fill outside
 *      active irrep blocks),
 *    - for each active irrep block, for each of two `c_in` tiles of 64:
 *        load the Z-indirect weight slab → mm_AB → scale → masked global
 *        `+=` epilogue.
 */
__global__ void mace_tk_element_conditioned_residual_force_adjoint_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,                // == 96 for MACE-OFF-small
    std::uint32_t n_elements,                   // == 10 for MACE-OFF23
    std::uint32_t n_channels_out,               // == 96 for MACE-OFF-small
    std::uint32_t active_irrep_block_count,     // 1 (L=0 only) or 4
    float path_weight_scale,                    // 1 / sqrt(c_in * n_elements)
    std::int32_t const *__restrict__ atom_element,
    float const *__restrict__ weights_fp32,
    float const *__restrict__ upstream_grad_output_fp32,
    float *__restrict__ grad_input_fp32
) {
    extern __shared__ kittens::alignment_dummy raw_shared_memory[];
    kittens::shared_allocator shared_alloc(reinterpret_cast<int *>(&raw_shared_memory[0]));

    shared_tile_upstream_t        &shared_upstream_tile     = shared_alloc.allocate<shared_tile_upstream_t>();
    shared_tile_weight_slab_t     &shared_weight_slab       = shared_alloc.allocate<shared_tile_weight_slab_t>();
    shared_tile_grad_input_stage_t &shared_grad_input_stage = shared_alloc.allocate<shared_tile_grad_input_stage_t>();

    std::uint32_t const atom_index = blockIdx.x;
    if (atom_index >= n_atoms) return;
    if (n_channels_in  != static_cast<std::uint32_t>(channels_in_k))  return;
    if (n_channels_out != static_cast<std::uint32_t>(channels_out_k)) return;

    int const atom_element_index = atom_element[atom_index];

    // Stage the per-atom upstream gradient slab once; reused across all blocks.
    load_atom_upstream_slab_(shared_upstream_tile, upstream_grad_output_fp32,
                              atom_index, active_irrep_block_count);
    kittens::warpgroup::sync(0);

    register_accumulator_t accumulator;

    for (unsigned int irrep_block_index = 0u; irrep_block_index < active_irrep_block_count; ++irrep_block_index) {
        #pragma unroll
        for (int channels_in_tile_index = 0;
             channels_in_tile_index < channels_in_tile_count_k;
             ++channels_in_tile_index) {

            load_z_indirect_weight_slab_(shared_weight_slab,
                                         weights_fp32,
                                         n_elements,
                                         n_channels_out,
                                         irrep_block_index,
                                         atom_element_index,
                                         channels_in_tile_index);
            kittens::warpgroup::sync(0);

            // D[c_in_tile=64, m=16] = W[c_in_tile=64, c_out=96]
            //                        @ dout[c_out=96, m=16]
            kittens::warp::zero(accumulator);
            kittens::warpgroup::mm_AB(accumulator, shared_weight_slab, shared_upstream_tile);
            kittens::warpgroup::mma_async_wait();

            scale_accumulator_in_place_(accumulator, path_weight_scale);

            add_irrep_block_into_grad_input_(grad_input_fp32,
                                              shared_grad_input_stage,
                                              accumulator,
                                              atom_index,
                                              n_channels_in,
                                              irrep_block_index,
                                              channels_in_tile_index);
            kittens::warpgroup::sync(0);
        }
    }
}

} // namespace mace_tk_element_conditioned_residual_force_adjoint_detail_

/**
 *  @brief Launch the TK-fused element-conditioned residual force adjoint (D3) for MACE-OFF.
 *
 *  Replaces `launch_mace_element_conditioned_residual_force_adjoint`
 *  (`mace_force_adjoints.cuh`) for the MACE-OFF-small/medium shape
 *  (`c_in == c_out == 96`).  Two forms:
 *
 *    - `active_irrep_block_count = 4`: layer-0 `RealAgnosticInteractionBlock`
 *      skip path — adjoint contributes to all four irrep blocks of the input
 *      gradient.
 *    - `active_irrep_block_count = 1`: layer-N `RealAgnosticResidualInteractionBlock`
 *      residual path — adjoint contributes to the L=0 m-component only.
 *
 *  Inputs / outputs:
 *    - `atom_element[n_atoms]`                                        : int32 species indices `Z[a] ∈ [0, n_elements)`,
 *    - `weights_fp32[n_blocks, c_in, n_elements, c_out]`              : fp32 row-major weights,
 *    - `upstream_grad_output_fp32[n_atoms, c_out, irrep_dim_total]`   : fp32 upstream gradient,
 *    - `grad_input_fp32[n_atoms, c_in, irrep_dim_total]`              : fp32 grad-input, accumulated `+=`.
 *
 *  Output is accumulated with `+=` (matches the fp32 reference's public ABI);
 *  the caller is responsible for pre-zeroing on the first adjoint contribution
 *  per backward pass.  Inactive m-components (those whose irrep-block index
 *  exceeds `active_irrep_block_count - 1`) are NOT touched, so the caller's
 *  zero-init survives.
 *
 *  This kernel template-bakes the MACE-OFF-small shape (`c_in == c_out == 96`,
 *  `irrep_dim_total == 16`).  Other shapes silently fall through and emit no
 *  work — callers must use the fp32 reference for those.
 *
 *  Gated on `SMALLFIELD_USE_THUNDERKITTENS`; only compiled when the macro is
 *  defined.
 */
inline void launch_mace_tk_element_conditioned_residual_force_adjoint(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_elements,
    std::uint32_t n_channels_out,
    std::uint32_t active_irrep_block_count,
    std::int32_t const *atom_element,
    float const *weights_fp32,
    float const *upstream_grad_output_fp32,
    float *grad_input_fp32,
    cudaStream_t stream
) {
    if (n_atoms == 0u) return;
    if (active_irrep_block_count == 0u) return;
    if (active_irrep_block_count > 4u) return;
    if (n_channels_in != static_cast<std::uint32_t>(
            mace_tk_element_conditioned_residual_force_adjoint_detail_::channels_in_k)) return;
    if (n_channels_out != static_cast<std::uint32_t>(
            mace_tk_element_conditioned_residual_force_adjoint_detail_::channels_out_k)) return;

    using namespace mace_tk_element_conditioned_residual_force_adjoint_detail_;

    float const path_weight_scale = 1.0f
        / sqrtf(static_cast<float>(n_channels_in) * static_cast<float>(n_elements));

    // Shared-mem footprint:
    //   upstream_tile        96 x 16 bf16 = 3 KB
    //   weight_slab          64 x 96 bf16 = 12 KB
    //   grad_input_stage     64 x 16 fp32 = 4 KB
    //   ----                                 ------
    //                                        ~19 KB + alignment slack
    constexpr std::size_t shared_memory_bytes_k = 32 * 1024;

    cudaFuncSetAttribute(
        mace_tk_element_conditioned_residual_force_adjoint_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_memory_bytes_k));

    dim3 const grid(n_atoms, 1, 1);
    dim3 const block(static_cast<std::uint32_t>(warpgroup_thread_count_k), 1, 1);

    mace_tk_element_conditioned_residual_force_adjoint_kernel<<<grid, block, shared_memory_bytes_k, stream>>>(
        n_atoms,
        n_channels_in,
        n_elements,
        n_channels_out,
        active_irrep_block_count,
        path_weight_scale,
        atom_element,
        weights_fp32,
        upstream_grad_output_fp32,
        grad_input_fp32);
}

} // namespace smallfield::cuda

#endif // SMALLFIELD_USE_THUNDERKITTENS && __CUDACC__
