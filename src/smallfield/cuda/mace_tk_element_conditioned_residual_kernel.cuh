/**
 *  @file mace_tk_element_conditioned_residual_kernel.cuh
 *  @brief ThunderKittens-fused element-conditioned per-irrep linear for MACE-OFF.
 *
 *  In MACE-OFF source this is the `skip_tp` of `RealAgnosticInteractionBlock`
 *  (layer-0, 4 active irrep blocks) and `RealAgnosticResidualInteractionBlock`
 *  (layer-N, 1 active block, applied to L=0 only).  The hand-rolled fp32
 *  reference lives in `mace_skip_tp_kernel.cuh::mace_element_conditioned_residual_kernel`.
 *
 *  The op is, for each active irrep block `L_b`:
 *
 *      output[a, c_out, m]  =  (1 / sqrt(c_in * n_elements))
 *                            * Σ_{c_in} W[L_b, c_in, Z[a], c_out] * input[a, c_in, m]
 *
 *  where `Z[a] = atom_element[a]` and `m` ranges over the (2 L_b + 1)
 *  m-components of block `L_b` inside the canonical 16-wide irrep axis
 *  `[L=0 (1 m), L=1 (3 m), L=2 (5 m), L=3 (7 m)]`.  Higher-L output
 *  m-components are zeroed when `active_irrep_block_count < 4`.
 *
 *  Kernel shape:
 *    - one CTA per atom,
 *    - one warpgroup (128 threads, 4 warps) per CTA,
 *    - per atom we load the input slab `[c_in = 96, m = 16]` once into bf16
 *      shared memory, then iterate over the active irrep blocks and over
 *      the two `c_out`-tile-of-64 columns (96 = 64 + 32 padded).  The
 *      `(L_b, Z[a])` weight slab `[c_in, c_out_tile = 64]` is hand-loaded
 *      from global fp32 into bf16 shared per (block, c_out tile),
 *    - WGMMA is `mma_AtB`: `D[c_out=64, m=16] = W[K=c_in, M=c_out_tile=64]^T
 *      @ input[K=c_in, N=m=16]`, fp32 accumulator,
 *    - the e3nn `path_weight = 1/sqrt(c_in * n_elements)` is folded into the
 *      register accumulator via `warpgroup::apply` before the bf16 store,
 *    - the final bf16 store guards against the partial second `c_out` tile
 *      and zero-fills the `m`-components that fall outside the current
 *      irrep block.
 *
 *  The Z indirection — picking which `[c_in, c_out]` slab of the larger
 *  `[active_irrep_block_count, c_in, n_elements, c_out]` weights blob to
 *  pull — is implemented as a per-thread strided fp32 load with bf16
 *  conversion into a swizzled `st_bf<c_in, c_out_tile>` shared tile.  This
 *  is the simplest correct shape; a TMA gather over the `Z` axis is a
 *  natural future optimisation but is not needed for correctness on the
 *  current MACE-OFF23 shape (`n_elements = 10`, weight slab per (block, Z)
 *  = 96 × 96 × 2 bytes = 18 KB, four blocks fit easily inside H100 shmem
 *  budget though we hold only one slab live at a time).
 *
 *  Compile-guarded on `SMALLFIELD_USE_THUNDERKITTENS`; when the macro is off
 *  this header is empty and the fp32 reference path stays the only impl.
 *
 *  Validated against the fp32 reference on a 64-atom synthetic batch with
 *  random element indices in `[0, n_elements)` and random
 *  `[n_atoms, c_in, 16]` features; max-abs target ≤ 5e-3 (bf16 noise).
 */

#pragma once

#include "smallfield/cuda/mace_tk_tile.cuh"

#if defined(SMALLFIELD_USE_THUNDERKITTENS) && defined(__CUDACC__)

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace smallfield::cuda {

namespace mace_tk_element_conditioned_residual_detail_ {

// Compile-time shapes for MACE-OFF-small.  `c_in == c_out == 96` and the
// irrep axis packs as 1 + 3 + 5 + 7 = 16 m-components.  WGMMA on Hopper
// requires the output `M` axis to be exactly 64, so we tile `c_out = 96`
// as two 64-wide column tiles of 64 (the second tile zero-pads cols
// 32..63 and we drop those rows during the bf16 store).
constexpr int channels_in_k             = mace_tk::mace_channels_small_k;        // 96
constexpr int channels_out_k            = mace_tk::mace_channels_small_k;        // 96
constexpr int irrep_dim_total_k         = mace_tk::mace_irrep_dim_total_k;       // 16
constexpr int channels_out_tile_size_k  = 64;                                    // WGMMA M
constexpr int channels_out_tile_count_k = (channels_out_k + channels_out_tile_size_k - 1)
                                          / channels_out_tile_size_k;            // 2
constexpr int max_active_irrep_block_count_k = 4;
constexpr int warpgroup_thread_count_k  = kittens::WARPGROUP_THREADS;             // 128

// Shared-tile aliases.  Activations and weights live as bf16 in shared
// memory; the WGMMA accumulator stays fp32 in registers.
using shared_tile_input_t        = mace_tk::shared_tile_bf16<channels_in_k, irrep_dim_total_k>;
using shared_tile_weight_slab_t  = mace_tk::shared_tile_bf16<channels_in_k, channels_out_tile_size_k>;
using shared_tile_output_stage_t = mace_tk::shared_tile_bf16<channels_out_tile_size_k, irrep_dim_total_k>;

// Per-warpgroup register accumulator: 64 rows × 16 cols (one warp per 16
// rows in the M direction).  TK expresses register-tile dims in tile
// units of 16; 16/16 = 1, 16/16 = 1 → `rt_fl<16, 16>` equals
// `rt_fl<height=1, width=1>` per warp = 64 rows × 16 cols across the
// warpgroup of four warps.
using register_accumulator_t = kittens::rt_fl<16, irrep_dim_total_k>;

/**
 *  @brief Map an m-component index `[0, 16)` to its irrep block index.
 *
 *  Mirrors `mace_irrep_block_index_for_component_` from
 *  `mace_linear_kernel.cuh` (which is `__device__`-only, hence the local
 *  copy here so this header is includable from host translation units).
 *  The m-component partition is `L=0 → m=0`, `L=1 → m=1..3`,
 *  `L=2 → m=4..8`, `L=3 → m=9..15`.
 */
__device__ __forceinline__ unsigned int irrep_block_index_for_component_(unsigned int irrep_component_index) {
    if (irrep_component_index == 0u) return 0u;
    if (irrep_component_index <= 3u) return 1u;
    if (irrep_component_index <= 8u) return 2u;
    return 3u;
}

/// Identity / type-converting bf16 cast — used inside the templated input
/// loader so the bf16 path skips the `__float2bfloat16_rn` round-trip.
__device__ __forceinline__ __nv_bfloat16 input_to_bf16_(float v) {
    return __float2bfloat16_rn(v);
}
__device__ __forceinline__ __nv_bfloat16 input_to_bf16_(__nv_bfloat16 v) {
    return v;
}

/**
 *  @brief Stage the per-atom input slab `[c_in, irrep_dim] = [96, 16]`
 *         into bf16 shared memory.  Templated on the input element type
 *         (`float` or `__nv_bfloat16`); the bf16 specialisation skips the
 *         `__float2bfloat16_rn` conversion.
 *
 *  Out-of-bounds atoms (`atom_index >= n_atoms`) are caught at kernel entry
 *  so this loader assumes a valid `atom_index`.
 */
template <typename input_t>
__device__ __forceinline__ void load_atom_input_slab_(
    shared_tile_input_t &dst_input_tile,
    input_t const *input_global,
    std::uint32_t atom_index
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = channels_in_k * irrep_dim_total_k;
    std::uint64_t const atom_input_offset =
        static_cast<std::uint64_t>(atom_index) * channels_in_k * irrep_dim_total_k;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / irrep_dim_total_k;  // c_in
        int const col_in_tile = linear_index % irrep_dim_total_k;  // m
        input_t const value = input_global[atom_input_offset
                                          + static_cast<std::uint64_t>(row_in_tile) * irrep_dim_total_k
                                          + col_in_tile];
        dst_input_tile[{row_in_tile, col_in_tile}] = input_to_bf16_(value);
    }
}

/**
 *  @brief Stage one Z-indirect weight slab `[c_in, c_out_tile = 64]`
 *         for `(irrep_block_index, atom_element_index)` from the global
 *         fp32 weights blob into bf16 shared memory.
 *
 *  Source layout is `weights[L_b, c_in, Z, c_out]` row-major; we read
 *  the slice `weights[L_b, :, Z, c_out_tile_base + 0..63]`.  Columns
 *  past `c_out` (e.g. cols 32..63 of the second tile) are zero-padded
 *  so the WGMMA picks up zeros there and the bf16 store path drops the
 *  corresponding output rows.
 */
__device__ __forceinline__ void load_z_indirect_weight_slab_(
    shared_tile_weight_slab_t &dst_weight_tile,
    float const *weights_global,
    std::uint32_t n_elements,
    std::uint32_t n_channels_out,
    unsigned int irrep_block_index,
    int atom_element_index,
    int channels_out_tile_index
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = channels_in_k * channels_out_tile_size_k;
    std::uint64_t const block_stride =
        static_cast<std::uint64_t>(channels_in_k) * n_elements * n_channels_out;
    std::uint64_t const channel_in_stride = static_cast<std::uint64_t>(n_elements) * n_channels_out;
    std::uint64_t const block_base =
        static_cast<std::uint64_t>(irrep_block_index) * block_stride
        + static_cast<std::uint64_t>(atom_element_index) * n_channels_out;
    int const channels_out_tile_base = channels_out_tile_index * channels_out_tile_size_k;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / channels_out_tile_size_k;  // c_in
        int const col_in_tile = linear_index % channels_out_tile_size_k;  // local c_out
        int const channel_out_global = channels_out_tile_base + col_in_tile;
        float value = 0.0f;
        if (channel_out_global < static_cast<int>(n_channels_out)) {
            value = weights_global[block_base
                                    + static_cast<std::uint64_t>(row_in_tile) * channel_in_stride
                                    + static_cast<std::uint64_t>(channel_out_global)];
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
 *  @brief Convert the fp32 accumulator to bf16, mask out m-components
 *         that fall outside the current irrep block, and write into the
 *         per-atom global output `[c_out, irrep_dim]` at the correct
 *         `c_out` tile rows.
 *
 *  We can't use the canonical `warpgroup::store(gl, rt, coord)` because
 *  (a) the second `c_out` tile is partial (32 valid rows of 64) and
 *  (b) we need to drop m's that don't belong to the current irrep block.
 *  We stage to a `[c_out_tile_size = 64, irrep_dim = 16]` shared tile and
 *  then do guarded per-thread global stores.
 *
 *  Output rows that don't belong to the current irrep block are not
 *  touched, so the caller must zero-init the global output once per
 *  atom (across the full `[c_out, irrep_dim]` slab) before the first
 *  irrep-block invocation.  The zero-init is folded into the first call
 *  via `is_first_irrep_block` below.
 */
__device__ __forceinline__ void store_irrep_block_with_guard_(
    __nv_bfloat16 *output_global,
    shared_tile_output_stage_t &staging_shared_tile,
    register_accumulator_t const &accumulator,
    std::uint32_t atom_index,
    std::uint32_t n_channels_out,
    unsigned int irrep_block_index,
    int channels_out_tile_index,
    bool is_first_irrep_block
) {
    // Convert fp32 register tile to bf16 in shared memory.
    kittens::warpgroup::store(staging_shared_tile, accumulator);
    kittens::warpgroup::sync(0);

    int const thread_id = threadIdx.x;
    constexpr int tile_elements = channels_out_tile_size_k * irrep_dim_total_k;
    int const channels_out_tile_base = channels_out_tile_index * channels_out_tile_size_k;
    std::uint64_t const atom_output_offset =
        static_cast<std::uint64_t>(atom_index) * n_channels_out * irrep_dim_total_k;
    __nv_bfloat16 const zero_bf16 = __float2bfloat16_rn(0.0f);
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / irrep_dim_total_k;       // local c_out
        int const col_in_tile = linear_index % irrep_dim_total_k;       // m
        int const channel_out_global = channels_out_tile_base + row_in_tile;
        if (channel_out_global >= static_cast<int>(n_channels_out)) continue;

        unsigned int const m_block_index = irrep_block_index_for_component_(static_cast<unsigned int>(col_in_tile));
        std::uint64_t const out_addr = atom_output_offset
                                       + static_cast<std::uint64_t>(channel_out_global) * irrep_dim_total_k
                                       + static_cast<std::uint64_t>(col_in_tile);
        if (m_block_index == irrep_block_index) {
            output_global[out_addr] = staging_shared_tile[{row_in_tile, col_in_tile}];
        } else if (is_first_irrep_block) {
            // Zero-init lanes that this irrep block doesn't write — only on
            // the first irrep-block pass per atom so subsequent blocks don't
            // clobber each other's m components.
            output_global[out_addr] = zero_bf16;
        }
    }
}

/**
 *  @brief Persistent kernel body: one CTA per atom, one warpgroup per CTA.
 *
 *  Grid: `dim3(n_atoms, 1, 1)`.  Block: 128 threads (one warpgroup).
 *  Per atom:
 *    - read `Z[atom]` once via a shared scalar (one thread reads, all wait on
 *      the post-load `warpgroup::sync(0)`),
 *    - stage the input slab `[c_in, m]` once,
 *    - for each active irrep block, for each of two `c_out` tiles of 64:
 *        load the Z-indirect weight slab → mma_AtB → scale → guarded bf16
 *        store with m-block masking.
 */
template <typename input_t>
__global__ void mace_tk_element_conditioned_residual_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,                // == 96 for MACE-OFF-small
    std::uint32_t n_elements,                   // == 10 for MACE-OFF23
    std::uint32_t n_channels_out,               // == 96 for MACE-OFF-small
    std::uint32_t active_irrep_block_count,     // 1 (L=0 only) or 4
    float path_weight_scale,                    // 1 / sqrt(c_in * n_elements) [optionally premultiplied
                                                //                              by 1/avg_num_neighbors]
    std::int32_t const *__restrict__ atom_element,
    float const *__restrict__ weights_fp32,
    input_t const *__restrict__ input_global,   // float OR __nv_bfloat16
    __nv_bfloat16 *__restrict__ output_bf16
) {
    extern __shared__ kittens::alignment_dummy raw_shared_memory[];
    kittens::shared_allocator shared_alloc(reinterpret_cast<int *>(&raw_shared_memory[0]));

    shared_tile_input_t        &shared_input_tile        = shared_alloc.allocate<shared_tile_input_t>();
    shared_tile_weight_slab_t  &shared_weight_slab       = shared_alloc.allocate<shared_tile_weight_slab_t>();
    shared_tile_output_stage_t &shared_output_stage_tile = shared_alloc.allocate<shared_tile_output_stage_t>();

    std::uint32_t const atom_index = blockIdx.x;
    if (atom_index >= n_atoms) return;
    if (n_channels_in  != static_cast<std::uint32_t>(channels_in_k))  return;
    if (n_channels_out != static_cast<std::uint32_t>(channels_out_k)) return;

    int const atom_element_index = atom_element[atom_index];

    // Stage the per-atom input slab once.
    load_atom_input_slab_<input_t>(shared_input_tile, input_global, atom_index);
    kittens::warpgroup::sync(0);

    register_accumulator_t accumulator;

    for (unsigned int irrep_block_index = 0u; irrep_block_index < active_irrep_block_count; ++irrep_block_index) {
        bool const is_first_irrep_block = (irrep_block_index == 0u);

        #pragma unroll
        for (int channels_out_tile_index = 0;
             channels_out_tile_index < channels_out_tile_count_k;
             ++channels_out_tile_index) {

            load_z_indirect_weight_slab_(shared_weight_slab,
                                         weights_fp32,
                                         n_elements,
                                         n_channels_out,
                                         irrep_block_index,
                                         atom_element_index,
                                         channels_out_tile_index);
            kittens::warpgroup::sync(0);

            // D[c_out=64, m=16] = W[K=c_in, M=c_out_tile=64]^T @ In[K=c_in, N=m=16]
            kittens::warp::zero(accumulator);
            kittens::warpgroup::mm_AtB(accumulator, shared_weight_slab, shared_input_tile);
            kittens::warpgroup::mma_async_wait();

            scale_accumulator_in_place_(accumulator, path_weight_scale);

            store_irrep_block_with_guard_(output_bf16,
                                          shared_output_stage_tile,
                                          accumulator,
                                          atom_index,
                                          n_channels_out,
                                          irrep_block_index,
                                          channels_out_tile_index,
                                          is_first_irrep_block);
            kittens::warpgroup::sync(0);
        }
    }

    // For active_irrep_block_count < 4, zero the m-components of higher
    // irrep blocks across the entire `c_out` axis.  The per-block loop
    // already zero-initialises lanes outside the current block on its
    // first invocation, but only for `m_block_index < active_irrep_block_count`.
    // We still need to clear any m for which the block index >= active count.
    if (active_irrep_block_count < static_cast<std::uint32_t>(max_active_irrep_block_count_k)) {
        int const thread_id = threadIdx.x;
        std::uint64_t const atom_output_offset =
            static_cast<std::uint64_t>(atom_index) * n_channels_out * irrep_dim_total_k;
        int const total_lanes = static_cast<int>(n_channels_out) * irrep_dim_total_k;
        __nv_bfloat16 const zero_bf16 = __float2bfloat16_rn(0.0f);
        for (int linear_index = thread_id; linear_index < total_lanes; linear_index += warpgroup_thread_count_k) {
            int const channel_out_index = linear_index / irrep_dim_total_k;
            int const irrep_component_index = linear_index % irrep_dim_total_k;
            unsigned int const m_block_index = irrep_block_index_for_component_(
                static_cast<unsigned int>(irrep_component_index));
            if (m_block_index >= active_irrep_block_count) {
                output_bf16[atom_output_offset
                            + static_cast<std::uint64_t>(channel_out_index) * irrep_dim_total_k
                            + static_cast<std::uint64_t>(irrep_component_index)] = zero_bf16;
            }
        }
    }
}

} // namespace mace_tk_element_conditioned_residual_detail_

/**
 *  @brief Launch the TK-fused element-conditioned residual for MACE-OFF.
 *
 *  Replaces `launch_mace_skip_tp` (`mace_skip_tp_kernel.cuh`) for the MACE-OFF
 *  layer-0 `RealAgnosticInteractionBlock` skip path (4 active irrep blocks)
 *  and the layer-N `RealAgnosticResidualInteractionBlock` skip path (1 active
 *  block, applied to the L=0 component only).
 *
 *  Inputs / outputs:
 *    - `atom_element[n_atoms]`              : int32 species indices `Z[a] ∈ [0, n_elements)`,
 *    - `weights_fp32[n_blocks, c_in, n_elements, c_out]` row-major fp32 weights,
 *    - `input_fp32[n_atoms, c_in, 16]`      : fp32 per-atom irrep features,
 *    - `output_bf16[n_atoms, c_out, 16]`    : bf16 per-atom output features
 *      (caller is expected to consume bf16; if fp32 is needed, follow with a
 *       bf16 → fp32 cast).
 *
 *  This kernel template-bakes the MACE-OFF-small shape (`c_in == c_out == 96`,
 *  `irrep_dim_total == 16`).  Other shapes silently fall through and emit
 *  no work — callers should fall back to the fp32 reference path for those.
 *
 *  Gated on `SMALLFIELD_USE_THUNDERKITTENS`; only compiled when the macro is
 *  defined.
 */
/// Returns true on successful launch, false if the channel-count tuple
/// isn't handled (caller must fall back to the fp32 reference path).
inline bool launch_mace_tk_element_conditioned_residual(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_elements,
    std::uint32_t n_channels_out,
    std::uint32_t active_irrep_block_count,
    std::int32_t const *atom_element,
    float const *weights_fp32,
    float const *input_fp32,
    __nv_bfloat16 *output_bf16,
    cudaStream_t stream
) {
    if (n_atoms == 0u) return true;
    if (active_irrep_block_count == 0u) return false;
    if (active_irrep_block_count > 4u) return false;
    if (n_channels_in != static_cast<std::uint32_t>(
            mace_tk_element_conditioned_residual_detail_::channels_in_k)) return false;
    if (n_channels_out != static_cast<std::uint32_t>(
            mace_tk_element_conditioned_residual_detail_::channels_out_k)) return false;

    using namespace mace_tk_element_conditioned_residual_detail_;

    float const path_weight_scale = 1.0f
        / sqrtf(static_cast<float>(n_channels_in) * static_cast<float>(n_elements));

    // Shared-mem footprint:
    //   input_tile          96 x 16 bf16 = 3 KB
    //   weight_slab         96 x 64 bf16 = 12 KB
    //   output_stage_tile   64 x 16 bf16 = 2 KB
    //   ----                                ------
    //                                      ~17 KB + alignment slack
    constexpr std::size_t shared_memory_bytes_k = 32 * 1024;

    cudaFuncSetAttribute(
        mace_tk_element_conditioned_residual_kernel<float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_memory_bytes_k));

    dim3 const grid(n_atoms, 1, 1);
    dim3 const block(static_cast<std::uint32_t>(warpgroup_thread_count_k), 1, 1);

    mace_tk_element_conditioned_residual_kernel<float><<<grid, block, shared_memory_bytes_k, stream>>>(
        n_atoms,
        n_channels_in,
        n_elements,
        n_channels_out,
        active_irrep_block_count,
        path_weight_scale,
        atom_element,
        weights_fp32,
        input_fp32,
        output_bf16);
    return true;
}

/**
 *  @brief Bf16-input variant of `launch_mace_tk_element_conditioned_residual` —
 *         Megakernel B Phase β-1.
 *
 *  Reads input directly from a bf16 buffer (typically the output of
 *  `launch_mace_tk_per_irrep_linear`, which already emits bf16) instead of
 *  going through an fp32 cast bridge.  Folds an additional uniform
 *  `pre_scale_multiplier` into the path-weight so the upstream
 *  `1 / avg_num_neighbors` scale step can also be skipped.  The fused
 *  effective scale is `pre_scale_multiplier / sqrt(c_in * n_elements)`.
 *
 *  Wins relative to the fp32 path:
 *    - no `bf16 → fp32` cast on the input (`n_atoms × c_in × 16` element
 *      bandwidth = ~96 MB at the small/medium production shape),
 *    - no `scale_in_place` on the fp32 buffer (~192 MB read+write),
 *    - the bf16 input slab still gets converted to bf16 in the shared
 *      tile, but the conversion is a no-op identity at the hardware
 *      level — the per-thread `__float2bfloat16_rn` becomes a register
 *      move.
 *
 *  Same shape constraints as the fp32 launcher: `c_in == c_out == 96`
 *  for MACE-OFF-small (returns false otherwise so the caller falls back
 *  to the fp32 chain).
 */
inline bool launch_mace_tk_element_conditioned_residual_bf16_input(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_elements,
    std::uint32_t n_channels_out,
    std::uint32_t active_irrep_block_count,
    std::int32_t const *atom_element,
    float const *weights_fp32,
    __nv_bfloat16 const *input_bf16,
    float pre_scale_multiplier,        // typically 1 / avg_num_neighbors
    __nv_bfloat16 *output_bf16,
    cudaStream_t stream
) {
    if (n_atoms == 0u) return true;
    if (active_irrep_block_count == 0u) return false;
    if (active_irrep_block_count > 4u) return false;
    if (n_channels_in != static_cast<std::uint32_t>(
            mace_tk_element_conditioned_residual_detail_::channels_in_k)) return false;
    if (n_channels_out != static_cast<std::uint32_t>(
            mace_tk_element_conditioned_residual_detail_::channels_out_k)) return false;

    using namespace mace_tk_element_conditioned_residual_detail_;

    float const path_weight_scale = pre_scale_multiplier
        / sqrtf(static_cast<float>(n_channels_in) * static_cast<float>(n_elements));

    constexpr std::size_t shared_memory_bytes_k = 32 * 1024;

    cudaFuncSetAttribute(
        mace_tk_element_conditioned_residual_kernel<__nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_memory_bytes_k));

    dim3 const grid(n_atoms, 1, 1);
    dim3 const block(static_cast<std::uint32_t>(warpgroup_thread_count_k), 1, 1);

    mace_tk_element_conditioned_residual_kernel<__nv_bfloat16><<<grid, block, shared_memory_bytes_k, stream>>>(
        n_atoms,
        n_channels_in,
        n_elements,
        n_channels_out,
        active_irrep_block_count,
        path_weight_scale,
        atom_element,
        weights_fp32,
        input_bf16,
        output_bf16);
    return true;
}

} // namespace smallfield::cuda

#endif // SMALLFIELD_USE_THUNDERKITTENS && __CUDACC__
