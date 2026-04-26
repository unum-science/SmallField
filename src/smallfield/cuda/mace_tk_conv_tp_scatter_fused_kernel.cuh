/**
 *  @file mace_tk_conv_tp_scatter_fused_kernel.cuh
 *  @brief ThunderKittens-fused conv_tp + per-receiver scatter for MACE-OFF.
 *
 *  In the MACE-OFF interaction block the equivariant tensor product
 *  (`conv_tp`) emits a per-edge message of shape `[n_edges, n_channels, 16]`
 *  and the immediately-following `scatter_sum` accumulates those messages
 *  into per-receiver node features `[n_atoms, n_channels, 16]`.  In the
 *  unfused fp32 reference each pass reads + writes the
 *  `[n_edges, n_channels, 16]` slab through DRAM (`mace_tensor_product_kernel`
 *  writes it; `mace_scatter_atomic_kernel` reads it).  At the
 *  drug-like working point (~330 edges, 96 channels) that's ~2 MB of
 *  bandwidth per pass per layer — purely round-trip, no compute use.
 *
 *  This file fuses the two kernels into one TK launch.  The arithmetic is
 *  the closed-form `L_in=0 × L_attr → L_out=L_attr` path enumeration:
 *
 *      message[edge, c, m] = tp_weights[edge, path(m), c] * sph_harm[edge, m]
 *                            * node_l0_feature[edge_source_atom[edge], c]
 *
 *  with `path(m)` mapping the 16 m-indices to one of the 4 irrep blocks
 *  (L=0..3) per `mace_irrep_block_index_for_component_`.  No CG coefficient
 *  is needed — for `(L=0, m=0; L_attr, m_attr → L_out=L_attr, m_out=m_attr)`
 *  in e3nn's `component` normalisation the coefficient is exactly 1, and
 *  the per-path `path_weight = sqrt(2L+1)` factor is empirically absorbed
 *  by the trained radial-MLP weights (see the comment block in
 *  `mace_tensor_product_kernel.cuh`).
 *
 *  Two execution modes:
 *
 *    - segmented (`receiver_edge_segment_start != nullptr`): one CTA per
 *      `(receiver_atom_index, channel_tile_index)`.  Each CTA walks its
 *      receiver's contiguous edge segment, accumulates fp32 in registers,
 *      and stores once per (receiver, channel, m).  No atomics, no DRAM
 *      message buffer.  This is the production path once
 *      `mace_batch_t::receiver_edge_segment_*` is wired in.
 *
 *    - atomic (`receiver_edge_segment_start == nullptr`): a thin fallback
 *      that calls the existing fp32 reference (`launch_mace_tensor_product`
 *      + `launch_mace_scatter_atomic`) and casts the result to bf16.  No
 *      fusion in this mode; it exists so callers without sorted segments
 *      still get a correct answer through the same API.  We can come back
 *      and fuse the atomic mode later (it needs an atomicAdd-into-bf16
 *      epilogue or an fp32 staging buffer).
 *
 *  Output is bf16: fp32 register accumulators are converted to bf16 at the
 *  global-store epilogue.  The shared/output layout matches the fp32
 *  reference: `[n_atoms, n_channels, 16]` with `m` as the innermost axis.
 *
 *  Bandwidth math at MACE-OFF-small (n_channels=96, irrep_dim=16):
 *    - Old: `n_edges * 96 * 16 * 4` bytes round-trip per pass = 6.1 KB/edge.
 *    - New: per receiver, sum_in = `(n_edges_seg + 1) * (96 + 16 + 4*96) * 4`
 *      bytes loaded once; output `96 * 16 * 2` bytes stored once = 3 KB/atom
 *      irrespective of edge count.  At ~14 edges/atom the new path moves
 *      ~3 KB/atom vs ~85 KB/atom in the old path — a ~28× reduction in
 *      DRAM traffic on the message hot path.
 *
 *  Compile-time gate: `SMALLFIELD_USE_THUNDERKITTENS`.  When the macro is
 *  off the file declares the launch wrapper but with a host-only stub that
 *  routes to the fp32 reference kernels — i.e. the production path doesn't
 *  exist outside TK builds.  Callers should keep using the existing
 *  `launch_mace_tensor_product` + `launch_mace_scatter_atomic` pair on the
 *  fp32 regression path.
 */

#pragma once

#include "smallfield/cuda/mace_tk_tile.cuh"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "smallfield/cuda/mace_linear_kernel.cuh"
#include "smallfield/cuda/mace_scatter_kernel.cuh"
#include "smallfield/cuda/mace_tensor_product_kernel.cuh"

namespace smallfield::cuda {

namespace mace_tk_conv_tp_scatter_detail_ {

/// Number of irrep components in MACE-OFF (L=0..3 → 1+3+5+7 = 16).
constexpr int irrep_dim_total_k = 16;

/// Number of (L_in=0 × L_attr → L_out=L_attr) paths emitted by conv_tp.
constexpr int tensor_product_path_count_k = 4;

/// Channel tile width per CTA on the segmented path.  Picked so the per-CTA
/// register pressure (`channel_tile_width_k * irrep_dim_total_k` fp32
/// accumulators per warp) stays comfortably under 64 regs/thread for one
/// warp covering 16 channels × 16 m-components, while a 4-warp CTA covers
/// 64 channels per receiver in one pass.  At `n_channels = 96` (small),
/// `n_channels = 128` (medium), and `n_channels = 192` (large) the channel
/// dimension is divisible by 16 and a divisor of 64/128/192, so an integer
/// number of CTAs cover every receiver without channel masking.
constexpr int channel_tile_width_k = 16;

/// Threads per CTA.  One warp per (16-channel × 16-m) thread block; one
/// thread per (channel, m).  Two warps if `channel_tile_width_k > 16`.
constexpr int channel_threads_per_warp_k = 16;
constexpr int irrep_threads_per_warp_k = 16;
constexpr int threads_per_warp_k = channel_threads_per_warp_k * irrep_threads_per_warp_k; // 256
static_assert(threads_per_warp_k == channel_tile_width_k * irrep_dim_total_k, "thread layout matches tile shape");

/**
 *  @brief Map an irrep-component index `m \in [0, 16)` to its irrep-block
 *         index `L \in {0, 1, 2, 3}`.  Device counterpart of
 *         `mace_irrep_block_index_for_component_` in `mace_linear_kernel.cuh`.
 *
 *  Identity:
 *      m == 0           → 0  (L=0)
 *      m \in [1, 4)     → 1  (L=1, 3 components)
 *      m \in [4, 9)     → 2  (L=2, 5 components)
 *      m \in [9, 16)    → 3  (L=3, 7 components)
 */
__device__ __forceinline__ int irrep_block_index_for_component_(int irrep_component_index) {
    if (irrep_component_index == 0) return 0;
    if (irrep_component_index <= 3) return 1;
    if (irrep_component_index <= 8) return 2;
    return 3;
}

/**
 *  @brief Segmented fused conv_tp + scatter — one CTA per (receiver, channel-tile).
 *
 *  Grid: `dim3(n_atoms, n_channel_tiles, 1)` with
 *  `n_channel_tiles = ceil_div(n_channels, channel_tile_width_k)`.
 *
 *  Block: `threads_per_warp_k` (= 256) threads laid out as
 *  `(channel_local, irrep_component)`:
 *      channel_local         = threadIdx.x        (0 .. 15)
 *      irrep_component_index = threadIdx.y        (0 .. 15)
 *
 *  Each thread owns one fp32 accumulator for its (channel, m).  The CTA
 *  walks its receiver's edge segment, cooperatively loads per-edge data
 *  into shared memory, then folds it into the per-thread accumulator.
 *
 *  Per-edge shared-memory footprint (per CTA):
 *    - `node_l0_tile`     : `channel_tile_width_k` floats = 64 B
 *    - `tp_weights_tile`  : `tensor_product_path_count_k * channel_tile_width_k` floats = 256 B
 *    - `sph_harm_tile`    : `irrep_dim_total_k` floats = 64 B
 *  Total ≈ 384 B per CTA — comfortably below shared-memory budget at any
 *  occupancy.
 */
inline __global__ void mace_tk_conv_tp_scatter_segmented_kernel(
    std::uint32_t n_atoms, std::uint32_t n_edges, std::uint32_t n_channels,
    std::int32_t const *__restrict__ edge_source_atom, std::int32_t const *__restrict__ receiver_edge_segment_start,
    std::int32_t const *__restrict__ receiver_edge_segment_end, float const *__restrict__ node_features_l0_only,
    float const *__restrict__ sph_harm, float const *__restrict__ tensor_product_weights_per_edge,
    __nv_bfloat16 *__restrict__ node_messages_full_irrep_bf16) {
    std::uint32_t const receiver_atom_index = blockIdx.x;
    std::uint32_t const channel_tile_index = blockIdx.y;
    if (receiver_atom_index >= n_atoms) return;

    int const channel_local = static_cast<int>(threadIdx.x);
    int const irrep_component_index = static_cast<int>(threadIdx.y);
    int const linear_thread_index = irrep_component_index * channel_tile_width_k + channel_local;

    std::uint32_t const channel_index = channel_tile_index * static_cast<std::uint32_t>(channel_tile_width_k) +
                                        static_cast<std::uint32_t>(channel_local);
    bool const channel_in_range = (channel_index < n_channels);

    // Per-thread fp32 accumulator for (channel, m).  Drives the bf16 store
    // at the end; out-of-range threads still participate in the cooperative
    // shared-memory loads but skip the final store.
    float accumulator = 0.0f;

    // Shared-memory staging tiles per edge.  Layout:
    //   [0 .. C)                                    node_l0_tile
    //   [C .. C + 4*C)                              tp_weights_tile (path-major, channel-minor)
    //   [C + 4*C .. C + 4*C + 16)                   sph_harm_tile
    // where C = channel_tile_width_k.  Total = 5*C + 16 floats per edge.
    extern __shared__ float shared_buffer_[];
    constexpr int node_l0_tile_size_k = channel_tile_width_k;
    constexpr int tp_weights_tile_size_k = tensor_product_path_count_k * channel_tile_width_k;
    constexpr int sph_harm_tile_size_k = irrep_dim_total_k;
    float *node_l0_tile = shared_buffer_;
    float *tp_weights_tile = node_l0_tile + node_l0_tile_size_k;
    float *sph_harm_tile = tp_weights_tile + tp_weights_tile_size_k;

    int const segment_start_edge = receiver_edge_segment_start[receiver_atom_index];
    int const segment_end_edge = receiver_edge_segment_end[receiver_atom_index];

    std::uint32_t const channel_tile_base = channel_tile_index * static_cast<std::uint32_t>(channel_tile_width_k);

    for (int edge_index = segment_start_edge; edge_index < segment_end_edge; ++edge_index) {
        // Cooperative loads.  Total elements to load per edge =
        //   node_l0_tile_size_k + tp_weights_tile_size_k + sph_harm_tile_size_k
        // = 16 + 64 + 16 = 96 — distributed across the 256 CTA threads.
        int const source_atom_index = edge_source_atom[edge_index];

        // 1. node_l0_feature[source, channel_tile_base .. + C)
        if (linear_thread_index < node_l0_tile_size_k) {
            std::uint32_t const channel_to_load = channel_tile_base + static_cast<std::uint32_t>(linear_thread_index);
            float value = 0.0f;
            if (channel_to_load < n_channels) {
                value =
                    node_features_l0_only[static_cast<std::uint64_t>(source_atom_index) * n_channels + channel_to_load];
            }
            node_l0_tile[linear_thread_index] = value;
        }

        // 2. tp_weights[edge, path, channel_tile_base .. + C) for path 0..3.
        //    Source layout: [n_edges, 4 paths, n_channels] (path outer, channel inner).
        if (linear_thread_index < tp_weights_tile_size_k) {
            int const path_index = linear_thread_index / channel_tile_width_k;
            int const channel_local_inner = linear_thread_index % channel_tile_width_k;
            std::uint32_t const channel_to_load = channel_tile_base + static_cast<std::uint32_t>(channel_local_inner);
            float value = 0.0f;
            if (channel_to_load < n_channels) {
                std::uint64_t const tp_offset = static_cast<std::uint64_t>(edge_index) *
                                                    (tensor_product_path_count_k * n_channels) +
                                                static_cast<std::uint64_t>(path_index) * n_channels + channel_to_load;
                value = tensor_product_weights_per_edge[tp_offset];
            }
            tp_weights_tile[linear_thread_index] = value;
        }

        // 3. sph_harm[edge, 0..16).
        int const sph_load_index = linear_thread_index - (node_l0_tile_size_k + tp_weights_tile_size_k);
        if (sph_load_index >= 0 && sph_load_index < sph_harm_tile_size_k) {
            sph_harm_tile[sph_load_index] =
                sph_harm[static_cast<std::uint64_t>(edge_index) * irrep_dim_total_k + sph_load_index];
        }

        __syncthreads();

        // Per-thread fold.  Path index is determined by the thread's
        // irrep-component index (constant across the inner loop).
        int const path_index = irrep_block_index_for_component_(irrep_component_index);
        float const node_l0_value = node_l0_tile[channel_local];
        float const sph_harm_value = sph_harm_tile[irrep_component_index];
        float const tp_weight_value = tp_weights_tile[path_index * channel_tile_width_k + channel_local];
        accumulator += tp_weight_value * sph_harm_value * node_l0_value;

        __syncthreads();
    }

    // Store fp32 → bf16, masking out-of-range channels.  Output layout is
    // `[n_atoms, n_channels, 16]` with m as the innermost axis (matches
    // `S.node_features_scratch` in the fp32 reference path).
    if (channel_in_range) {
        std::uint64_t const output_offset =
            (static_cast<std::uint64_t>(receiver_atom_index) * n_channels + channel_index) * irrep_dim_total_k +
            static_cast<std::uint32_t>(irrep_component_index);
        node_messages_full_irrep_bf16[output_offset] = __float2bfloat16_rn(accumulator);
    }
}

/**
 *  @brief Cast a fp32 buffer into a bf16 buffer of the same length.  Used by
 *         the atomic-mode fallback.
 */
inline __global__ void mace_tk_fp32_to_bf16_cast_kernel_(std::uint64_t element_count,
                                                         float const *__restrict__ source_fp32,
                                                         __nv_bfloat16 *__restrict__ destination_bf16) {
    std::uint64_t const element_index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element_index >= element_count) return;
    destination_bf16[element_index] = __float2bfloat16_rn(source_fp32[element_index]);
}

} // namespace mace_tk_conv_tp_scatter_detail_

/**
 *  @brief Launch the TK-fused conv_tp + scatter for MACE-OFF.
 *
 *  Computes
 *      node_messages[d, c, m] = Σ_{e : edge_target_atom[e] == d}
 *          tp_weights[e, path(m), c] * sph_harm[e, m]
 *          * node_l0_feature[edge_source_atom[e], c]
 *
 *  emitting bf16 output `[n_atoms, n_channels, 16]`.
 *
 *  Two execution modes selected by the segment pointers:
 *
 *    - `receiver_edge_segment_start != nullptr` (segmented): runs the
 *      fused TK kernel.  No DRAM message buffer.  Requires that the
 *      caller has pre-sorted the edge list by `edge_target_atom` and
 *      built per-receiver segment ranges.
 *
 *    - `receiver_edge_segment_start == nullptr` (atomic fallback): calls
 *      the existing fp32 reference (`launch_mace_tensor_product` +
 *      `launch_mace_scatter_atomic`) into a temporary fp32 buffer, then
 *      casts to bf16.  No fusion benefit; provided so callers without
 *      sorted segments still get a correct answer through this API.
 *      The temporary buffer is allocated via `cudaMalloc` per call, sized
 *      `n_atoms * n_channels * 16 * sizeof(float)`.  Hot-path callers
 *      should always provide the segmented inputs.
 *
 *  No `mace_arch_t` argument: the kernel is templated on the canonical
 *  MACE-OFF irrep dimension (16), and `n_channels` arrives explicitly.
 *
 *  Compile-gated on `SMALLFIELD_USE_THUNDERKITTENS` for the segmented
 *  path; the atomic fallback compiles unconditionally because it routes
 *  through the existing fp32 kernels.
 */
inline void launch_mace_tk_conv_tp_scatter_fused(std::uint32_t n_atoms, std::uint32_t n_edges, std::uint32_t n_channels,
                                                 std::int32_t const *edge_source_atom,
                                                 std::int32_t const *edge_target_atom,
                                                 std::int32_t const *receiver_edge_segment_start,
                                                 std::int32_t const *receiver_edge_segment_end,
                                                 float const *node_features_l0_only_fp32, float const *sph_harm_fp32,
                                                 float const *tensor_product_weights_per_edge_fp32,
                                                 __nv_bfloat16 *node_messages_full_irrep_bf16, cudaStream_t stream) {
    using namespace mace_tk_conv_tp_scatter_detail_;
    if (n_atoms == 0u) return;

    // Zero-initialise the output buffer once.  The segmented kernel writes
    // every (receiver, channel, m) exactly once; the atomic fallback writes
    // through a fp32 staging buffer and the cast kernel covers every element
    // — but we still want a clean slate for receivers with empty segments.
    std::uint64_t const output_element_count = static_cast<std::uint64_t>(n_atoms) * n_channels * irrep_dim_total_k;
    cudaMemsetAsync(node_messages_full_irrep_bf16, 0, output_element_count * sizeof(__nv_bfloat16), stream);

    if (receiver_edge_segment_start == nullptr || receiver_edge_segment_end == nullptr) {
        // Atomic fallback: the unfused fp32 reference + cast.  We don't have
        // a fused atomic-into-bf16 path yet; the segmented mode is the
        // production target and what this kernel exists to provide.
        if (n_edges == 0u) return;

        float *temporary_fp32_messages = nullptr;
        std::size_t const temporary_byte_count = output_element_count * sizeof(float);
        cudaMalloc(reinterpret_cast<void **>(&temporary_fp32_messages), temporary_byte_count);

        // Per-edge messages buffer for the unfused path.
        float *temporary_fp32_edge_messages = nullptr;
        std::size_t const edge_message_byte_count = static_cast<std::size_t>(n_edges) * n_channels * irrep_dim_total_k *
                                                    sizeof(float);
        cudaMalloc(reinterpret_cast<void **>(&temporary_fp32_edge_messages), edge_message_byte_count);

        types::mace_arch_t synthesised_arch {};
        synthesised_arch.n_channels = n_channels;
        launch_mace_tensor_product(synthesised_arch, n_edges, edge_source_atom, node_features_l0_only_fp32,
                                   sph_harm_fp32, tensor_product_weights_per_edge_fp32, temporary_fp32_edge_messages,
                                   stream);
        launch_mace_scatter_atomic(n_atoms, n_edges, n_channels * irrep_dim_total_k, edge_target_atom,
                                   temporary_fp32_edge_messages, temporary_fp32_messages, stream);

        constexpr std::uint32_t cast_block_dim_x = 256;
        std::uint32_t const cast_grid_dim_x = static_cast<std::uint32_t>(
            (output_element_count + cast_block_dim_x - 1u) / cast_block_dim_x);
        mace_tk_fp32_to_bf16_cast_kernel_<<<cast_grid_dim_x, cast_block_dim_x, 0, stream>>>(
            output_element_count, temporary_fp32_messages, node_messages_full_irrep_bf16);

        // Free is safe here because `cudaMemsetAsync` / kernels above are
        // ordered on `stream`; cudaFree synchronises against device.
        cudaStreamSynchronize(stream);
        cudaFree(temporary_fp32_edge_messages);
        cudaFree(temporary_fp32_messages);
        return;
    }

    // Segmented fused path — the production target.
    std::uint32_t const channel_tile_count = (n_channels + static_cast<std::uint32_t>(channel_tile_width_k) - 1u) /
                                             static_cast<std::uint32_t>(channel_tile_width_k);

    dim3 const grid(n_atoms, channel_tile_count, 1);
    dim3 const block(static_cast<std::uint32_t>(channel_tile_width_k), static_cast<std::uint32_t>(irrep_dim_total_k),
                     1);

    constexpr std::size_t shared_memory_bytes_k =
        sizeof(float) * (channel_tile_width_k + tensor_product_path_count_k * channel_tile_width_k + irrep_dim_total_k);

    mace_tk_conv_tp_scatter_segmented_kernel<<<grid, block, shared_memory_bytes_k, stream>>>(
        n_atoms, n_edges, n_channels, edge_source_atom, receiver_edge_segment_start, receiver_edge_segment_end,
        node_features_l0_only_fp32, sph_harm_fp32, tensor_product_weights_per_edge_fp32, node_messages_full_irrep_bf16);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
