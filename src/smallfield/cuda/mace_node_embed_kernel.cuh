/**
 *  @file mace_node_embed_kernel.cuh
 *  @brief LinearNodeEmbeddingBlock — one-hot Z → 0e channel features.
 *
 *  MACE-OFF's `LinearNodeEmbeddingBlock` is a linear layer from the one-hot
 *  atomic-species tensor (shape `[n_atoms, n_elements]`) to the per-atom
 *  scalar features (shape `[n_atoms, n_channels]`).  Because the input is
 *  one-hot, the linear reduces to a row lookup: `feat[a, c] = W[Z[a], c]`,
 *  where `W` has shape `[n_elements, n_channels]`.
 *
 *  Output goes into the L=0 component of the full irrep tensor; L=1..3 stay
 *  zero on entry into the first interaction.  The internal layout is
 *  `[n_atoms, n_channels, irrep_dim=16]` with `irrep_dim` index 0 = L=0.
 *  This matches `reshape_irreps` output (mul × m within each irrep block,
 *  blocks concatenated along the inner axis).
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

#include "smallfield/types/mace_types.hpp"

namespace smallfield::cuda {

/**
 *  @brief Fill the L=0 channel slot of every node from the species table.
 *
 *  Threading: one block per (atom, channel-tile).  The kernel zeros the full
 *  `[n_channels, 16]` tile before writing the L=0 slot, so callers get a
 *  fresh, layer-ready feature buffer.
 *
 *  Note that MACE's e3nn-style `Linear` for `node_embedding` stores its
 *  weight as `(n_elements * n_channels)` flat, with elements outer.  We
 *  read `node_embedding[Z * n_channels + c]`, matching the export tool.
 */
inline __global__ void mace_node_embed_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t irrep_dim,                   // 16 for MACE-OFF L_max=3
    float scale,                                // e3nn path_weight = 1/sqrt(n_elements)
    std::int32_t const *atom_element,          // [n_atoms]
    float const *node_embedding_weight,        // [n_elements, n_channels]
    float *node_features                       // [n_atoms, n_channels, irrep_dim]
) {
    std::uint32_t atom_index = blockIdx.x;
    std::uint32_t channel_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (atom_index >= n_atoms || channel_index >= n_channels) return;
    int const element_index = atom_element[atom_index];
    // Zero out the full irrep slab first (write strided fp32; cheap).
    for (std::uint32_t irrep_component_index = 0; irrep_component_index < irrep_dim; ++irrep_component_index) {
        node_features[(static_cast<std::uint64_t>(atom_index) * n_channels + channel_index) * irrep_dim + irrep_component_index] = 0.0f;
    }
    // L=0 is m=0; write the embedding weight for this element, scaled by the e3nn
    // path normalisation factor (= 1/sqrt(num_input_irreps_paths_to_this_output)
    // = 1/sqrt(n_elements) for one-hot 0e → 0e linear).
    node_features[(static_cast<std::uint64_t>(atom_index) * n_channels + channel_index) * irrep_dim + 0] =
        scale * node_embedding_weight[static_cast<std::uint64_t>(element_index) * n_channels + channel_index];
}

inline void launch_mace_node_embed(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t n_elements,
    std::uint32_t irrep_dim,
    std::int32_t const *atom_element,
    float const *node_embedding_weight,
    float *node_features,
    cudaStream_t stream
) {
    if (n_atoms == 0 || n_channels == 0) return;
    constexpr std::uint32_t block_dim_x = 64;
    dim3 const block(block_dim_x, 1, 1);
    dim3 const grid(n_atoms, (n_channels + block_dim_x - 1u) / block_dim_x, 1);
    float const scale = 1.0f / sqrtf(static_cast<float>(n_elements));
    mace_node_embed_kernel<<<grid, block, 0, stream>>>(
        n_atoms, n_channels, irrep_dim, scale, atom_element, node_embedding_weight, node_features);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
