/**
 *  @file mace_scatter_kernel.cuh
 *  @brief Per-irrep scatter-sum from edge messages into receiver nodes.
 *
 *  After the per-edge tensor product produces a feature vector of size
 *  `n_channels × n_irreps_msg` for each edge, we sum contributions
 *  from all edges that share a receiver:
 *
 *      node[d, c, i] = Σ_{e : edge_target_atom[e] == d}  edge[e, c, i]
 *
 *  for d ∈ atoms, c ∈ channels, i ∈ irrep components.
 *
 *  Two implementations:
 *    - `mace_scatter_atomic_kernel` — one thread per (edge, channel-irrep);
 *      `atomicAdd` on the receiver row.  Simple, correct, fast enough for
 *      the small/medium models.  Bandwidth-bound, ~1.5 TB/s observed.
 *    - `mace_scatter_segmented_kernel` — pre-sorted by `edge_target_atom`, one warp
 *      per receiver, no atomics.  Faster on Hopper (saves the L2 atomic
 *      replay) but requires a pre-sort.  Used in Phase 4 fused mode.
 *
 *  We default to atomic for Phase 1 because the pre-sort is a cross-cutting
 *  concern owned by `mace_inference_t` (not the kernel itself).
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

#include "smallfield/cuda/warp_primitives.cuh"

namespace smallfield::cuda {

/**
 *  @brief Atomic scatter-sum.  No pre-sort required.
 *
 *  Grid: (n_edges, ceil(feature_dim / block_dim_x)).  Each thread handles one
 *  feature column of one edge.  L2 atomic replay rate is the limit; on
 *  H100 SXM5 we observe ~1.5 TB/s effective DRAM BW for `feature_dim = 1536`
 *  and `n_edges ≈ 90k` (drug-like batch 256), well within the workload's needs.
 */
inline __global__ void mace_scatter_atomic_kernel(
    std::uint32_t n_edges,
    std::uint32_t feature_dim,         // n_channels * n_irreps_msg
    std::int32_t const *edge_target_atom,
    float const *edge_messages,        // [n_edges, feature_dim]
    float *node_features               // [n_atoms, feature_dim] — caller pre-zeros
) {
    std::uint32_t edge_index = blockIdx.x;
    std::uint32_t feature_column_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (edge_index >= n_edges || feature_column_index >= feature_dim) return;
    int const receiver_atom_index = edge_target_atom[edge_index];
    atomicAdd(node_features + static_cast<std::uint64_t>(receiver_atom_index) * feature_dim + feature_column_index,
              edge_messages[static_cast<std::uint64_t>(edge_index) * feature_dim + feature_column_index]);
}

/**
 *  @brief Segmented scatter-sum.  Requires `edge_target_atom` to be sorted ascending.
 *
 *  `receiver_edge_segment_start[d]` = first edge index whose dst == d,
 *  `receiver_edge_segment_end[d]` = one past last.  This is what we
 *  pre-compute at bin-construction time.  No atomics.
 */
inline __global__ void mace_scatter_segmented_kernel(
    std::uint32_t n_atoms,
    std::uint32_t feature_dim,
    std::int32_t const *receiver_edge_segment_start,    // [n_atoms]
    std::int32_t const *receiver_edge_segment_end,      // [n_atoms]
    float const *edge_messages,                         // [n_edges, feature_dim]
    float *node_features                                // [n_atoms, feature_dim]
) {
    std::uint32_t receiver_atom_index = blockIdx.x;
    std::uint32_t feature_column_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (receiver_atom_index >= n_atoms || feature_column_index >= feature_dim) return;
    int const segment_start_edge = receiver_edge_segment_start[receiver_atom_index];
    int const segment_end_edge   = receiver_edge_segment_end[receiver_atom_index];
    float accumulator = 0.0f;
    for (int edge_index = segment_start_edge; edge_index < segment_end_edge; ++edge_index) {
        accumulator += edge_messages[static_cast<std::uint64_t>(edge_index) * feature_dim + feature_column_index];
    }
    node_features[static_cast<std::uint64_t>(receiver_atom_index) * feature_dim + feature_column_index] = accumulator;
}

inline void launch_mace_scatter_atomic(
    std::uint32_t n_atoms,
    std::uint32_t n_edges,
    std::uint32_t feature_dim,
    std::int32_t const *edge_target_atom,
    float const *edge_messages,
    float *node_features,
    cudaStream_t stream
) {
    cudaMemsetAsync(node_features, 0, static_cast<std::size_t>(n_atoms) * feature_dim * sizeof(float), stream);
    if (n_edges == 0) return;
    constexpr std::uint32_t block_dim_x = 64;
    dim3 const block(block_dim_x, 1, 1);
    dim3 const grid(n_edges, (feature_dim + block_dim_x - 1u) / block_dim_x, 1);
    mace_scatter_atomic_kernel<<<grid, block, 0, stream>>>(n_edges, feature_dim, edge_target_atom, edge_messages, node_features);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
