/**
 *  @file mace_tk_conv_tp_scatter_fused_force_adjoint_kernel.cuh
 *  @brief ThunderKittens-flavoured fused conv_tp + scatter force adjoint (D2) for MACE-OFF.
 *
 *  Adjoint of `mace_tk_conv_tp_scatter_segmented_kernel` / the unfused
 *  `launch_mace_tensor_product` + `launch_mace_scatter_atomic` pair.
 *
 *  Forward (per edge `e`, channel `c`, m-component `m`):
 *      msg[e, c, m]            = path_weight[e, L(m), c]
 *                              * sph_harm[e, m]
 *                              * node_l0_feature[edge_source_atom[e], c]
 *  followed by a scatter-sum into the receiver's row of `node_features_full`.
 *
 *  Receiving the post-scatter upstream gradient `du[a, c, m]`, the adjoint
 *  emits three independent gradient buffers per edge `e` (with
 *  `s = edge_source_atom[e]`, `t = edge_target_atom[e]`):
 *
 *      d_node_l0[s, c]
 *          += Σ_m path_weight[e, L(m), c] * sph_harm[e, m] * du[t, c, m]
 *          (atomicAdd: multi-edge fan-in to the same source atom)
 *
 *      d_path_weights[e, p, c]
 *          = Σ_{m ∈ p} sph_harm[e, m] * node_l0_feature[s, c] * du[t, c, m]
 *          (non-atomic: per-edge per-path-per-channel is single-owner)
 *
 *      d_sph_harm[e, m]
 *          += Σ_c path_weight[e, L(m), c] * node_l0_feature[s, c] * du[t, c, m]
 *          (atomicAdd: per-edge but channel-tiled across CTAs)
 *
 *  bf16 atomicAdds on Hopper are NOT commutative — non-determinism would leak
 *  into the per-source-atom and per-edge-m gather sums.  The output buffers
 *  therefore stay fp32; only the per-edge tile loads (path weights, upstream)
 *  go through bf16 staging in shared memory to match the forward's storage
 *  precision.  fp32 register accumulators dominate per-thread state; the bf16
 *  surface area is the per-edge load, not the gather.
 *
 *  Tile shape mirrors the forward kernel:
 *    - One CTA per `(edge_index, channel_tile_index)` with 16 channels per
 *      tile and `irrep_dim_total_k = 16` m-components.
 *    - 256 threads per CTA laid out as `(channel_local, irrep_component_index)`.
 *    - bf16 shared tiles for `node_l0`, `tp_weights`, `sph_harm`, `upstream`
 *      (matching the forward's tile shapes; same WGMMA-tile-class footprint).
 *    - fp32 scratchpad for two per-CTA reductions (path-weight m-fold and
 *      d_node_l0 m-fold).
 *    - Per-thread fp32 register fold computes its (channel_local,
 *      irrep_component_index) cell's three gradient contributions.
 *
 *  WGMMA per se is not used in this kernel — the per-edge backward inner
 *  contraction has axis `m = 16` (for d_node_l0) or axis `c = 16` per CTA
 *  (for d_sph_harm), neither of which justifies a `mm_AB`/`mm_ABt` issue.
 *  The "matmul tile shape" parity with the forward is in the bf16 shared-
 *  memory tile geometry, not in a tensor-core invocation.
 *
 *  The receiver-edge segment pointers in the public ABI are accepted but
 *  unused by this kernel — the natural CTA dimension is per-edge.  They are
 *  in the signature so the launcher matches the forward kernel's parameter
 *  shape and a future fused-with-segments variant can drop in without touching
 *  callers.
 *
 *  Compile-guarded on `SMALLFIELD_USE_THUNDERKITTENS`; when the macro is off
 *  the file is empty and the fp32 reference path
 *  (`launch_mace_tensor_product_and_scatter_force_adjoint` in
 *  `mace_force_adjoints.cuh`) remains the only D2 adjoint.
 */

#pragma once

#include "smallfield/cuda/mace_tk_tile.cuh"

#if defined(SMALLFIELD_USE_THUNDERKITTENS) && defined(__CUDACC__)

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace smallfield::cuda {

namespace mace_tk_conv_tp_scatter_force_adjoint_detail_ {

/// Number of irrep components in MACE-OFF (L=0..3 → 1+3+5+7 = 16).
constexpr int irrep_dim_total_k = mace_tk::mace_irrep_dim_total_k;

/// Number of (L_in=0 × L_attr → L_out=L_attr) paths emitted by conv_tp.
constexpr int tensor_product_path_count_k = mace_tk::mace_n_paths_k;

/// Channel tile width per CTA.  Matches the forward kernel's tile so the same
/// CTA shape and shared-memory budget cover both passes; one warp covers the
/// 16 channels × 16 m-components plane, with extra warps used for cooperative
/// loads.
constexpr int channel_tile_width_k = 16;

/// Threads per CTA: one thread per (channel_local, irrep_component_index)
/// pair.  Mirrors the forward kernel.
constexpr int channel_threads_per_warp_k = 16;
constexpr int irrep_threads_per_warp_k = 16;
constexpr int threads_per_cta_k = channel_threads_per_warp_k * irrep_threads_per_warp_k; // 256
static_assert(threads_per_cta_k == channel_tile_width_k * irrep_dim_total_k, "thread layout matches tile shape");

/**
 *  @brief Map an irrep-component index `m \in [0, 16)` to its irrep-block
 *         index `L \in {0, 1, 2, 3}`.  Device counterpart of
 *         `mace_irrep_block_index_for_component_` in `mace_linear_kernel.cuh`;
 *         identical to the forward kernel's helper.
 */
__device__ __forceinline__ int irrep_block_index_for_component_(int irrep_component_index) {
    if (irrep_component_index == 0) return 0;
    if (irrep_component_index <= 3) return 1;
    if (irrep_component_index <= 8) return 2;
    return 3;
}

/**
 *  @brief Per-edge fused conv_tp + scatter force adjoint kernel.
 *
 *  Grid: `dim3(n_edges, n_channel_tiles, 1)` with
 *  `n_channel_tiles = ceil_div(n_channels, channel_tile_width_k)`.
 *
 *  Block: 256 threads laid out as `(channel_local, irrep_component_index)`:
 *      channel_local         = threadIdx.x        (0 .. 15)
 *      irrep_component_index = threadIdx.y        (0 .. 15)
 *
 *  Per CTA:
 *    1. Cooperatively stage the per-edge data into shared bf16 tiles:
 *         - `node_l0_tile`   : C entries (source-atom L=0 row, channel slice)
 *         - `tp_weights_tile`: 4*C entries (path-major, channel-minor)
 *         - `sph_harm_tile`  : 16 entries (per-edge spherical harmonics)
 *         - `upstream_tile`  : C*16 entries (upstream gradient slab from receiver,
 *                              channel-major × m-minor)
 *      All loads are fp32 → bf16 conversions on the way in (matching forward's
 *      storage precision).  Per-thread fp32 register fold consumes them.
 *    2. Per-thread fold over its (channel_local, irrep_component_index) cell:
 *         - Compute three contributions:
 *             dnode_l0_contribution  = scale * tp_w * sph_harm * du
 *             dtp_weight_contribution = scale * sph_harm * node_l0 * du
 *             dsph_harm_contribution = scale * tp_w * node_l0 * du
 *    3. Reduce `d_path_weights` along the m lane via shared-memory scratch
 *       (rows summing per L block: m=0; m=1..3; m=4..8; m=9..15) and emit
 *       single-owner non-atomic writes to `[edge, path, channel]`.
 *    4. Reduce `d_sph_harm` along the channel lane via `__shfl_xor_sync`
 *       butterfly and atomicAdd into `[edge, m]` (channel-tiled CTAs may
 *       collide on the same slot).
 *    5. Reduce `d_node_l0` along the m lane via shared-memory scratch and
 *       atomicAdd into `[source_atom, channel]` (multi-edge fan-in).
 *
 *  Note: per-thread, the `irrep_component_index` is fixed, so only one of the
 *  four path-block accumulators ever receives a nonzero contribution from
 *  that thread.  The shared-memory reduction collects the per-(channel,
 *  L-block) sum back together for the per-edge per-path per-channel write.
 */
inline __global__ void mace_tk_conv_tp_scatter_force_adjoint_kernel(
    std::uint32_t n_edges, std::uint32_t n_channels, float scale, std::int32_t const *__restrict__ edge_source_atom,
    std::int32_t const *__restrict__ edge_target_atom, float const *__restrict__ node_l0_feature_global,
    float const *__restrict__ sph_harm_global, float const *__restrict__ tensor_product_weights_per_edge_global,
    float const *__restrict__ upstream_grad_node_messages_full_global, float *__restrict__ grad_node_l0_feature_global,
    float *__restrict__ grad_tensor_product_weights_per_edge_global, float *__restrict__ grad_sph_harm_global) {
    std::uint32_t const edge_index = blockIdx.x;
    std::uint32_t const channel_tile_index = blockIdx.y;
    if (edge_index >= n_edges) return;

    int const channel_local = static_cast<int>(threadIdx.x);
    int const irrep_component_index = static_cast<int>(threadIdx.y);
    int const linear_thread_index = irrep_component_index * channel_tile_width_k + channel_local;

    std::uint32_t const channel_index = channel_tile_index * static_cast<std::uint32_t>(channel_tile_width_k) +
                                        static_cast<std::uint32_t>(channel_local);
    bool const channel_in_range = (channel_index < n_channels);

    int const source_atom_index = edge_source_atom[edge_index];
    int const target_atom_index = edge_target_atom[edge_index];

    // -- Shared-memory layout.
    //    Two regions: the bf16 tile slab and the fp32 reduction scratch.
    //    The driver passes a single `extern __shared__` allocation sized
    //    `bf16_tile_bytes_k + fp32_scratch_bytes_k`; we cast it to the
    //    per-region pointer types here.  The fp32 region is placed AFTER
    //    the bf16 tiles to ensure 4-byte alignment regardless of bf16 tile
    //    parity.
    constexpr int node_l0_tile_size_k = channel_tile_width_k;
    constexpr int tp_weights_tile_size_k = tensor_product_path_count_k * channel_tile_width_k;
    constexpr int sph_harm_tile_size_k = irrep_dim_total_k;
    constexpr int upstream_tile_size_k = channel_tile_width_k * irrep_dim_total_k;
    constexpr int bf16_total_size_k = node_l0_tile_size_k + tp_weights_tile_size_k + sph_harm_tile_size_k +
                                      upstream_tile_size_k;
    // bf16_total_size_k = 16 + 64 + 16 + 256 = 352 (even ⇒ 4-byte aligned tail).

    extern __shared__ __nv_bfloat16 shared_buffer_[];
    __nv_bfloat16 *node_l0_tile = shared_buffer_;
    __nv_bfloat16 *tp_weights_tile = node_l0_tile + node_l0_tile_size_k;
    __nv_bfloat16 *sph_harm_tile = tp_weights_tile + tp_weights_tile_size_k;
    __nv_bfloat16 *upstream_tile = sph_harm_tile + sph_harm_tile_size_k;
    float *fp32_reduction_scratch = reinterpret_cast<float *>(shared_buffer_ + bf16_total_size_k);
    // fp32_reduction_scratch holds 16 m × 16 channels = 256 fp32 entries
    // (1 KB), reused across the path-weight reduction and the d_node_l0
    // reduction (the two are sequenced by `__syncthreads()`).

    std::uint32_t const channel_tile_base = channel_tile_index * static_cast<std::uint32_t>(channel_tile_width_k);

    // -- 1. Cooperative loads.  Total elements to load per CTA:
    //       C + 4*C + 16 + C*16 = 16 + 64 + 16 + 256 = 352 elements across 256
    //       threads.  The upstream tile (256 elements) covers exactly the
    //       linear-thread range so it overlaps every other load.

    // 1a. node_l0_feature[source, channel_tile_base .. + C)
    if (linear_thread_index < node_l0_tile_size_k) {
        std::uint32_t const channel_to_load = channel_tile_base + static_cast<std::uint32_t>(linear_thread_index);
        float value = 0.0f;
        if (channel_to_load < n_channels) {
            value =
                node_l0_feature_global[static_cast<std::uint64_t>(source_atom_index) * n_channels + channel_to_load];
        }
        node_l0_tile[linear_thread_index] = __float2bfloat16_rn(value);
    }

    // 1b. tp_weights[edge, path, channel_tile_base .. + C) for path 0..3.
    //     Source layout: [n_edges, 4 paths, n_channels] (path outer, channel inner).
    if (linear_thread_index < tp_weights_tile_size_k) {
        int const path_index = linear_thread_index / channel_tile_width_k;
        int const channel_local_inner = linear_thread_index % channel_tile_width_k;
        std::uint32_t const channel_to_load = channel_tile_base + static_cast<std::uint32_t>(channel_local_inner);
        float value = 0.0f;
        if (channel_to_load < n_channels) {
            std::uint64_t const tp_offset = static_cast<std::uint64_t>(edge_index) *
                                                (tensor_product_path_count_k * n_channels) +
                                            static_cast<std::uint64_t>(path_index) * n_channels + channel_to_load;
            value = tensor_product_weights_per_edge_global[tp_offset];
        }
        tp_weights_tile[linear_thread_index] = __float2bfloat16_rn(value);
    }

    // 1c. sph_harm[edge, 0..16)
    {
        int const sph_load_index = linear_thread_index - (node_l0_tile_size_k + tp_weights_tile_size_k);
        if (sph_load_index >= 0 && sph_load_index < sph_harm_tile_size_k) {
            sph_harm_tile[sph_load_index] = __float2bfloat16_rn(
                sph_harm_global[static_cast<std::uint64_t>(edge_index) * irrep_dim_total_k + sph_load_index]);
        }
    }

    // 1d. upstream[target_atom, channel_tile_base .. + C, 0..16) — load all 256 cells.
    //     Source layout: [n_atoms, n_channels, 16].  Tile is `[channel_local, m]`
    //     stored row-major (channel-major, m-minor).
    {
        int const upstream_thread_index = linear_thread_index;
        if (upstream_thread_index < upstream_tile_size_k) {
            int const tile_channel_local = upstream_thread_index / irrep_dim_total_k;
            int const tile_m_index = upstream_thread_index % irrep_dim_total_k;
            std::uint32_t const channel_to_load = channel_tile_base + static_cast<std::uint32_t>(tile_channel_local);
            float value = 0.0f;
            if (channel_to_load < n_channels) {
                std::uint64_t const upstream_offset =
                    (static_cast<std::uint64_t>(target_atom_index) * n_channels + channel_to_load) * irrep_dim_total_k +
                    static_cast<std::uint32_t>(tile_m_index);
                value = upstream_grad_node_messages_full_global[upstream_offset];
            }
            upstream_tile[upstream_thread_index] = __float2bfloat16_rn(value);
        }
    }

    __syncthreads();

    // -- 2. Per-thread fold.  Each thread owns one (channel_local, irrep_component_index)
    //       cell.  The path index for this thread is fixed (determined by m).
    int const path_index_for_thread = irrep_block_index_for_component_(irrep_component_index);

    float const node_l0_value = __bfloat162float(node_l0_tile[channel_local]);
    float const sph_harm_value = __bfloat162float(sph_harm_tile[irrep_component_index]);
    float const tp_weight_value = __bfloat162float(
        tp_weights_tile[path_index_for_thread * channel_tile_width_k + channel_local]);
    float const upstream_value = __bfloat162float(
        upstream_tile[channel_local * irrep_dim_total_k + irrep_component_index]);

    // d_node_l0[s, c] += scale * tp_w[L(m), c] * sph_harm[m] * du[c, m]
    float const dnode_l0_contribution = scale * tp_weight_value * sph_harm_value * upstream_value;

    // d_path_weights[e, L(m), c] += scale * sph_harm[m] * node_l0[c] * du[c, m]
    float const dtp_weight_contribution = scale * sph_harm_value * node_l0_value * upstream_value;

    // d_sph_harm[e, m] += scale * Σ_c tp_w[L(m), c] * node_l0[c] * du[c, m]
    float const dsph_harm_contribution = scale * tp_weight_value * node_l0_value * upstream_value;

    // -- 3. Reduce `d_path_weights` along the m lane.  Each thread writes its
    //       contribution into the [16 m][16 channels] scratchpad; one writer
    //       per (path-block, channel) sums the rows in that block and emits
    //       the per-edge per-path per-channel store (single-owner, non-atomic).
    fp32_reduction_scratch[irrep_component_index * channel_tile_width_k + channel_local] = dtp_weight_contribution;
    __syncthreads();

    if (channel_in_range && grad_tensor_product_weights_per_edge_global != nullptr) {
        struct path_block_t {
            int m_start;
            int m_count;
            int path_index;
        };
        path_block_t const blocks[tensor_product_path_count_k] = {{0, 1, 0}, {1, 3, 1}, {4, 5, 2}, {9, 7, 3}};
        for (int block_index = 0; block_index < tensor_product_path_count_k; ++block_index) {
            path_block_t const blk = blocks[block_index];
            // Designate the first m-row of each block as the writer for that
            // (path, channel) slot — single-owner avoids any cross-thread
            // contention.
            if (irrep_component_index != blk.m_start) continue;
            float accumulator = 0.0f;
            for (int m_within = 0; m_within < blk.m_count; ++m_within) {
                accumulator += fp32_reduction_scratch[(blk.m_start + m_within) * channel_tile_width_k + channel_local];
            }
            std::uint64_t const out_offset = static_cast<std::uint64_t>(edge_index) *
                                                 (tensor_product_path_count_k * n_channels) +
                                             static_cast<std::uint64_t>(blk.path_index) * n_channels + channel_index;
            grad_tensor_product_weights_per_edge_global[out_offset] = accumulator;
        }
    }

    // -- 4. Reduce `d_sph_harm` along the channel lane.  With block layout
    //       (blockDim.x = 16, blockDim.y = 16) the linear thread index is
    //       `m * 16 + channel_local`, so two consecutive m-rows share a
    //       32-lane warp.  A `__shfl_xor_sync` butterfly with offsets 8, 4,
    //       2, 1 reduces within each 16-lane half-warp, leaving the per-row
    //       sum in `channel_local == 0`.
    {
        unsigned const full_mask = 0xFFFFFFFFu;
        float channel_reduced = dsph_harm_contribution;
        if (!channel_in_range) channel_reduced = 0.0f;
#pragma unroll
        for (int xor_offset = 8; xor_offset > 0; xor_offset >>= 1) {
            channel_reduced += __shfl_xor_sync(full_mask, channel_reduced, xor_offset);
        }
        if (channel_local == 0 && grad_sph_harm_global != nullptr) {
            atomicAdd(grad_sph_harm_global + static_cast<std::uint64_t>(edge_index) * irrep_dim_total_k +
                          static_cast<std::uint32_t>(irrep_component_index),
                      channel_reduced);
        }
    }

    // -- 5. Reduce `d_node_l0` along the m lane and atomicAdd into the source
    //       atom's row.  Reuse the same shared scratchpad — its previous
    //       contents (path-weight contributions) were already consumed in
    //       step 3.
    __syncthreads();
    fp32_reduction_scratch[irrep_component_index * channel_tile_width_k + channel_local] = dnode_l0_contribution;
    __syncthreads();

    if (irrep_component_index == 0 && channel_in_range && grad_node_l0_feature_global != nullptr) {
        float accumulator = 0.0f;
#pragma unroll
        for (int m_index = 0; m_index < irrep_dim_total_k; ++m_index) {
            accumulator += fp32_reduction_scratch[m_index * channel_tile_width_k + channel_local];
        }
        atomicAdd(
            grad_node_l0_feature_global + static_cast<std::uint64_t>(source_atom_index) * n_channels + channel_index,
            accumulator);
    }
}

} // namespace mace_tk_conv_tp_scatter_force_adjoint_detail_

/**
 *  @brief Launch the TK-flavoured fused conv_tp + scatter force adjoint (D2).
 *
 *  Computes three gradient buffers in one kernel launch — adjoint of the
 *  forward fused conv_tp + scatter:
 *
 *    - `grad_node_l0_feature_fp32[n_atoms, n_channels]` — atomicAdd
 *      accumulator (multi-edge fan-in to source atom).  Caller must
 *      pre-zero (or pre-populate from a parallel adjoint path).
 *    - `grad_tensor_product_weights_per_edge_fp32[n_edges, 4 * n_channels]` —
 *      non-atomic write (single-owner per (edge, path, channel)).  Output is
 *      OVERWRITTEN, not accumulated.
 *    - `grad_sph_harm_fp32[n_edges, 16]` — atomicAdd accumulator (channel-
 *      tiled CTAs collide on the same (edge, m) slot).  Caller must pre-zero.
 *
 *  Pass `nullptr` for any of the three output pointers to skip its
 *  computation; the per-thread compute path is unchanged but the global
 *  store is elided.
 *
 *  The receiver-edge segment pointers are accepted for ABI parity with the
 *  forward TK kernel but are unused — the natural CTA dimension here is
 *  per-edge (the fan-in is on the source-atom side, not the target).
 *
 *  Output buffers stay fp32 because bf16 atomicAdds are NOT commutative on
 *  Hopper — non-determinism would leak into the per-source-atom and per-edge
 *  m gather sums.  bf16 only appears on the per-edge tile loads in shared
 *  memory.
 *
 *  Threading: `dim3(n_edges, ceil(n_channels / 16), 1)` grid; 256 threads per
 *  CTA (`16 channels × 16 m`).  Shared-memory footprint ≈ 0.7 KB bf16 tiles
 *  + 1 KB fp32 reduction scratch ≈ 1.7 KB per CTA.
 *
 *  Gated on `SMALLFIELD_USE_THUNDERKITTENS`; when the macro is off the driver
 *  falls back to `launch_mace_tensor_product_and_scatter_force_adjoint` in
 *  `mace_force_adjoints.cuh`.
 */
inline void launch_mace_tk_tensor_product_and_scatter_force_adjoint(
    std::uint32_t n_atoms, std::uint32_t n_edges, std::uint32_t n_channels, std::int32_t const *edge_source_atom,
    std::int32_t const *edge_target_atom,
    std::int32_t const *receiver_edge_segment_start,    // [n_atoms] or nullptr
    std::int32_t const *receiver_edge_segment_end,      // [n_atoms] or nullptr
    float const *node_l0_feature_fp32,                  // [n_atoms, n_channels]
    float const *sph_harm_fp32,                         // [n_edges, 16]
    float const *tensor_product_weights_per_edge_fp32,  // [n_edges, 4 * n_channels]
    float const *upstream_grad_node_messages_full_fp32, // [n_atoms, n_channels, 16]
    float *grad_node_l0_feature_fp32,                   // [n_atoms, n_channels]      atomicAdd
    float *grad_tensor_product_weights_per_edge_fp32,   // [n_edges, 4 * n_channels]   non-atomic
    float *grad_sph_harm_fp32,                          // [n_edges, 16]              atomicAdd
    cudaStream_t stream) {
    using namespace mace_tk_conv_tp_scatter_force_adjoint_detail_;
    (void)n_atoms;
    (void)receiver_edge_segment_start;
    (void)receiver_edge_segment_end;
    if (n_edges == 0u) return;

    // Match the fp32 reference's scale convention: the radial-MLP weights
    // already absorb the e3nn TP path-weight factor, so scale = 1.
    float const scale = 1.0f;

    std::uint32_t const channel_tile_count = (n_channels + static_cast<std::uint32_t>(channel_tile_width_k) - 1u) /
                                             static_cast<std::uint32_t>(channel_tile_width_k);

    dim3 const grid(n_edges, channel_tile_count, 1);
    dim3 const block(static_cast<std::uint32_t>(channel_tile_width_k), static_cast<std::uint32_t>(irrep_dim_total_k),
                     1);

    // Shared-memory budget:
    //   bf16 tiles    : (C + 4*C + 16 + C*16) bf16 = 352 bf16 = 704 bytes
    //   fp32 scratch  : 16 m × 16 channels = 256 fp32 = 1024 bytes
    constexpr std::size_t bf16_total_size_k = channel_tile_width_k +
                                              tensor_product_path_count_k * channel_tile_width_k + irrep_dim_total_k +
                                              channel_tile_width_k * irrep_dim_total_k;
    constexpr std::size_t bf16_tile_bytes_k = sizeof(__nv_bfloat16) * bf16_total_size_k;
    constexpr std::size_t fp32_scratch_bytes_k = sizeof(float) * channel_tile_width_k * irrep_dim_total_k;
    constexpr std::size_t shared_memory_bytes_k = bf16_tile_bytes_k + fp32_scratch_bytes_k;

    mace_tk_conv_tp_scatter_force_adjoint_kernel<<<grid, block, shared_memory_bytes_k, stream>>>(
        n_edges, n_channels, scale, edge_source_atom, edge_target_atom, node_l0_feature_fp32, sph_harm_fp32,
        tensor_product_weights_per_edge_fp32, upstream_grad_node_messages_full_fp32, grad_node_l0_feature_fp32,
        grad_tensor_product_weights_per_edge_fp32, grad_sph_harm_fp32);
}

} // namespace smallfield::cuda

#endif // SMALLFIELD_USE_THUNDERKITTENS && __CUDACC__
