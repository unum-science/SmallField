/**
 *  @file mace_tensor_product_kernel.cuh
 *  @brief MACE-OFF interaction `conv_tp` — equivariant tensor product (e3nn `uvu`).
 *
 *  After `linear_up` projects the input node features down to scalar (L=0)
 *  features `[n_nodes, c, 1]`, the conv_tp combines per-node scalars with
 *  per-edge spherical-harmonic attributes `[n_edges, 16]` to produce a
 *  full irrep message `[n_edges, c, 16]`.  Because the input is purely L=0
 *  the four `(L_in=0 × L_attr → L_out=L_attr)` paths have a closed form:
 *
 *      msg[e, c, m]  =  W[e, L(m), c]  *  sph_harm[e, m]  *  node_l0_feature[edge_source_atom[e], c]
 *
 *  Where `W` is the radial-MLP output reshaped as `[n_edges, 4 paths, c]`
 *  and `L(m)` maps an output m-index to its irrep block (0..3).  The CG
 *  coefficient for `(L=0, m=0; L_attr, m_attr → L_out=L_attr, m_out=m_attr)`
 *  in e3nn's `component` normalisation is exactly 1, so it folds away.
 *
 *  Threading: one thread per (edge, channel).  Inner: 16 multiplies + stores.
 *  ~88k edges × 96 channels = ~8.5M threads, well-shaped for H100.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

#include "smallfield/types/mace_types.hpp"

namespace smallfield::cuda {

constexpr std::uint32_t mace_tensor_product_path_count_k = 4u;

/**
 *  @brief Tensor product (full L_in=0 form) — fp32 reference.
 *
 *  Inputs:
 *    `node_l0_feature[n_atoms, n_channels]`              — sender L=0 features
 *    `sph_harm[n_edges, 16]`                             — spherical-harmonic edge attribute
 *    `tensor_product_weights_per_edge[n_edges, 4 * c]`   — per-edge per-path-per-channel scalars
 *  Output:
 *    `edge_messages[n_edges, n_channels, 16]`
 *
 *  The radial-MLP output layout `[n_edges, 4 * c]` is interpreted as
 *  `(path_idx, channel_idx)` — path outer, channel inner.  This matches
 *  e3nn's `TensorProduct.weight` packing for "uvu" mode.
 */
inline __global__ void mace_tensor_product_kernel(
    std::uint32_t n_edges, std::uint32_t n_channels,
    float scale, // e3nn TP path normalisation = 1/sqrt(c_in * 1)
    std::int32_t const *edge_source_atom,
    float const *node_l0_feature,                 // [n_atoms, n_channels]
    float const *sph_harm,                        // [n_edges, 16]
    float const *tensor_product_weights_per_edge, // [n_edges, 4 * n_channels]
    float *edge_messages                          // [n_edges, n_channels, 16]
) {
    std::uint32_t edge_index = blockIdx.x;
    std::uint32_t channel_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (edge_index >= n_edges || channel_index >= n_channels) return;

    int const src = edge_source_atom[edge_index];
    float const node_l0_feature_value = node_l0_feature[static_cast<std::uint64_t>(src) * n_channels + channel_index];

    // Pre-load per-path weights for this (edge, channel).
    float const path_weight_l0 = tensor_product_weights_per_edge[static_cast<std::uint64_t>(edge_index) *
                                                                     (mace_tensor_product_path_count_k * n_channels) +
                                                                 0u * n_channels + channel_index];
    float const path_weight_l1 = tensor_product_weights_per_edge[static_cast<std::uint64_t>(edge_index) *
                                                                     (mace_tensor_product_path_count_k * n_channels) +
                                                                 1u * n_channels + channel_index];
    float const path_weight_l2 = tensor_product_weights_per_edge[static_cast<std::uint64_t>(edge_index) *
                                                                     (mace_tensor_product_path_count_k * n_channels) +
                                                                 2u * n_channels + channel_index];
    float const path_weight_l3 = tensor_product_weights_per_edge[static_cast<std::uint64_t>(edge_index) *
                                                                     (mace_tensor_product_path_count_k * n_channels) +
                                                                 3u * n_channels + channel_index];

    std::uint64_t const sph_harm_offset = static_cast<std::uint64_t>(edge_index) * 16u;
    std::uint64_t const output_offset = (static_cast<std::uint64_t>(edge_index) * n_channels + channel_index) * 16u;
    float const scaled_node_l0_feature = scale * node_l0_feature_value;
    edge_messages[output_offset + 0] = path_weight_l0 * sph_harm[sph_harm_offset + 0] * scaled_node_l0_feature;
    edge_messages[output_offset + 1] = path_weight_l1 * sph_harm[sph_harm_offset + 1] * scaled_node_l0_feature;
    edge_messages[output_offset + 2] = path_weight_l1 * sph_harm[sph_harm_offset + 2] * scaled_node_l0_feature;
    edge_messages[output_offset + 3] = path_weight_l1 * sph_harm[sph_harm_offset + 3] * scaled_node_l0_feature;
    edge_messages[output_offset + 4] = path_weight_l2 * sph_harm[sph_harm_offset + 4] * scaled_node_l0_feature;
    edge_messages[output_offset + 5] = path_weight_l2 * sph_harm[sph_harm_offset + 5] * scaled_node_l0_feature;
    edge_messages[output_offset + 6] = path_weight_l2 * sph_harm[sph_harm_offset + 6] * scaled_node_l0_feature;
    edge_messages[output_offset + 7] = path_weight_l2 * sph_harm[sph_harm_offset + 7] * scaled_node_l0_feature;
    edge_messages[output_offset + 8] = path_weight_l2 * sph_harm[sph_harm_offset + 8] * scaled_node_l0_feature;
    edge_messages[output_offset + 9] = path_weight_l3 * sph_harm[sph_harm_offset + 9] * scaled_node_l0_feature;
    edge_messages[output_offset + 10] = path_weight_l3 * sph_harm[sph_harm_offset + 10] * scaled_node_l0_feature;
    edge_messages[output_offset + 11] = path_weight_l3 * sph_harm[sph_harm_offset + 11] * scaled_node_l0_feature;
    edge_messages[output_offset + 12] = path_weight_l3 * sph_harm[sph_harm_offset + 12] * scaled_node_l0_feature;
    edge_messages[output_offset + 13] = path_weight_l3 * sph_harm[sph_harm_offset + 13] * scaled_node_l0_feature;
    edge_messages[output_offset + 14] = path_weight_l3 * sph_harm[sph_harm_offset + 14] * scaled_node_l0_feature;
    edge_messages[output_offset + 15] = path_weight_l3 * sph_harm[sph_harm_offset + 15] * scaled_node_l0_feature;
}

inline void launch_mace_tensor_product(types::mace_arch_t const &arch, std::uint32_t n_edges,
                                       std::int32_t const *edge_source_atom, float const *node_l0_feature,
                                       float const *sph_harm, float const *tensor_product_weights_per_edge,
                                       float *edge_messages, cudaStream_t stream) {
    if (n_edges == 0) return;
    constexpr std::uint32_t block_dim_x = 32;
    dim3 const block(block_dim_x, 1, 1);
    dim3 const grid(n_edges, (arch.n_channels + block_dim_x - 1u) / block_dim_x, 1);
    // No explicit path normalisation here — empirically (validated against the
    // golden trace), the trained radial-MLP weights already absorb the e3nn
    // path-weight factor, and applying 1/sqrt(c_in) on top makes the energy
    // overshoot in the opposite direction.  This is one of the two remaining
    // ~0.2 % gaps logged in `MACE_INFERENCE.md`'s Phase-1-followup section.
    float const scale = 1.0f;
    mace_tensor_product_kernel<<<grid, block, 0, stream>>>(n_edges, arch.n_channels, scale, edge_source_atom,
                                                           node_l0_feature, sph_harm, tensor_product_weights_per_edge,
                                                           edge_messages);
}

/**
 *  @brief Multiply the radial-MLP output by the cutoff envelope per edge.
 *
 *  In the live MACE forward, `tensor_product_weights_per_edge =
 *  conv_tp_weights(edge_feats) * cutoff_envelope` — the `cutoff_envelope`
 *  here is the polynomial cutoff envelope already computed in
 *  `mace_radial_basis_kernel`.  We keep the per-edge scalar separately so
 *  we can reuse the radial-MLP intermediate across the two interaction
 *  layers (cutoff is purely geometric, doesn't change between layers).
 *
 *  Output dim per edge = `n_paths * n_channels`.  Threading: one thread per
 *  (edge, channel-tile).
 */
inline __global__ void mace_apply_cutoff_envelope_kernel(
    std::uint32_t n_edges,
    std::uint32_t feature_dim,             // n_paths * n_channels
    float const *cutoff_envelope,          // [n_edges]
    float *tensor_product_weights_per_edge // [n_edges, feature_dim] — modified in place
) {
    std::uint32_t edge_index = blockIdx.x;
    std::uint32_t feature_column_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (edge_index >= n_edges || feature_column_index >= feature_dim) return;
    float const cut = cutoff_envelope[edge_index];
    tensor_product_weights_per_edge[static_cast<std::uint64_t>(edge_index) * feature_dim + feature_column_index] *= cut;
}

inline void launch_mace_apply_cutoff(std::uint32_t n_edges, std::uint32_t feature_dim, float const *cutoff_envelope,
                                     float *tensor_product_weights_per_edge, cudaStream_t stream) {
    if (n_edges == 0) return;
    constexpr std::uint32_t block_dim_x = 64;
    dim3 const block(block_dim_x, 1, 1);
    dim3 const grid(n_edges, (feature_dim + block_dim_x - 1u) / block_dim_x, 1);
    mace_apply_cutoff_envelope_kernel<<<grid, block, 0, stream>>>(n_edges, feature_dim, cutoff_envelope,
                                                                  tensor_product_weights_per_edge);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
