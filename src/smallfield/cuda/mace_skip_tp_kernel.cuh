/**
 *  @file mace_skip_tp_kernel.cuh
 *  @brief Element-conditioned residual: FullyConnectedTensorProduct of irrep
 *         features × one-hot atomic species.
 *
 *  In MACE-OFF source this appears as `skip_tp` inside each interaction
 *  block.  Mathematically it is `e3nn.o3.FullyConnectedTensorProduct`
 *  between an irrep tensor (channel × m_components) and a one-hot
 *  atom-species feature.  The one-hot input is L=0, so the only valid TP
 *  paths are `(L_in × L=0 → L_in)` for each `L_in` shared between input
 *  and output.  Because the species-side input is one-hot, the layer
 *  reduces to an element-conditioned channel-mix per irrep block — hence
 *  the rename from the e3nn jargon `skip_tp` to
 *  `mace_element_conditioned_residual_kernel`.
 *
 *  For each such path the tensor product reduces to:
 *
 *      output[a, c_out, m]  =  Σ_{c_in}  W[L_block, c_in, Z[a], c_out]  *  input[a, c_in, m]
 *
 *  where `irrep_block_index` is the irrep index of `m` (0 for L=0, 1 for L=1, ..., 3
 *  for L=3) and Z[a] is the atomic species of atom a.  If a `irrep_block_index`
 *  is in input but not in output (or vice versa), it is skipped — the layer
 *  drops or zero-pads accordingly.
 *
 *  Two forms are exposed:
 *    - Layer-0 form: input AND output have the same irrep set
 *      (e.g. `96x0e + 96x1o + 96x2e + 96x3o` → same).  All 4 L-blocks present.
 *    - Layer-N form: input has all 4 L-blocks, output has only L=0
 *      (`96x0e`).  Only the L=0 block matches; other m's are dropped.
 *
 *  Threading: one thread per (atom, c_out, m).  Inner: 96 multiplies +
 *  accumulate.  Cheap.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

#include "smallfield/cuda/mace_linear_kernel.cuh" // mace_irrep_block_index_for_component_

namespace smallfield::cuda {

/**
 *  @brief Element-conditioned per-irrep linear (the "skip_tp" in MACE source).
 *
 *  Weight blob layout: `[active_irrep_block_count, c_in, n_elements, c_out]`
 *  flat.  `active_irrep_block_count` is 1 (only L=0 path) for the
 *  residual-style layer and 4 for the layer-0 selector form.
 */
inline __global__ void mace_element_conditioned_residual_kernel(
    std::uint32_t n_atoms, std::uint32_t n_channels_in, std::uint32_t n_elements, std::uint32_t n_channels_out,
    std::uint32_t irrep_dim,                // 16
    std::uint32_t active_irrep_block_count, // 1 or 4
    float scale,                      // e3nn path_weight = 1/sqrt(c_in * n_elements) for one-instruction-per-output
                                      // FullyConnectedTensorProduct
    std::int32_t const *atom_element, // [n_atoms]
    float const *weights,             // [active_irrep_block_count, c_in, n_elements, c_out]
    float const *input,               // [n_atoms, c_in, irrep_dim]
    float *output                     // [n_atoms, c_out, irrep_dim]
) {
    std::uint32_t atom_index = blockIdx.x;
    std::uint32_t channel_out_index = blockIdx.y;
    std::uint32_t irrep_component_index = threadIdx.x;
    if (atom_index >= n_atoms || channel_out_index >= n_channels_out || irrep_component_index >= irrep_dim) return;

    std::uint32_t const irrep_block_index = mace_irrep_block_index_for_component_(irrep_component_index);
    if (irrep_block_index >= active_irrep_block_count) {
        output[(static_cast<std::uint64_t>(atom_index) * n_channels_out + channel_out_index) * irrep_dim +
               irrep_component_index] = 0.0f;
        return;
    }
    int const element_index = atom_element[atom_index];
    std::uint64_t const weight_block_offset = static_cast<std::uint64_t>(irrep_block_index) * n_channels_in *
                                              n_elements * n_channels_out;
    std::uint64_t const input_offset = static_cast<std::uint64_t>(atom_index) * n_channels_in * irrep_dim;
    float accumulator = 0.0f;
    for (std::uint32_t channel_in_index = 0; channel_in_index < n_channels_in; ++channel_in_index) {
        float const w =
            weights[weight_block_offset + static_cast<std::uint64_t>(channel_in_index) * n_elements * n_channels_out +
                    static_cast<std::uint64_t>(element_index) * n_channels_out + channel_out_index];
        float const x =
            input[input_offset + static_cast<std::uint64_t>(channel_in_index) * irrep_dim + irrep_component_index];
        accumulator += w * x;
    }
    output[(static_cast<std::uint64_t>(atom_index) * n_channels_out + channel_out_index) * irrep_dim +
           irrep_component_index] = scale * accumulator;
}

/**
 *  @brief Specialized irrep_dim=1 (scalar-only L=0) skip-tp / element-conditioned residual.
 *
 *  Equivalent to the general kernel for the (irrep_dim=1, active=1) case.
 *  Avoids the 1-thread-per-CTA shape that the general launcher picks when
 *  irrep_dim=1 (block(1,1,1) × grid(n_atoms, n_channels_out, 1) =
 *  5.9 M one-thread CTAs).
 *
 *  `out[a, c_out] = scale * Σ_{c_in} W[Z[a], c_in, c_out] * in[a, c_in]`
 *
 *  Weight layout: [n_in, n_elements, n_out] (matching the general kernel's
 *  per-block layout, but irrep_block_index=0 collapses out the leading dim).
 */
inline __global__ void mace_skip_tp_irrep1_kernel(std::uint32_t n_atoms, std::uint32_t n_channels_in,
                                                  std::uint32_t n_elements, std::uint32_t n_channels_out, float scale,
                                                  std::int32_t const *__restrict__ atom_element, // [n_atoms]
                                                  float const *__restrict__ weights, // [n_in, n_elements, n_out]
                                                  float const *__restrict__ input,   // [n_atoms, n_in]
                                                  float *__restrict__ output         // [n_atoms, n_out]
) {
    std::uint32_t const atom_index = blockIdx.x * blockDim.y + static_cast<std::uint32_t>(threadIdx.y);
    std::uint32_t const channel_out = blockIdx.y * blockDim.x + static_cast<std::uint32_t>(threadIdx.x);
    if (atom_index >= n_atoms || channel_out >= n_channels_out) return;

    int const element_index = atom_element[atom_index];
    float const *input_row = input + static_cast<std::uint64_t>(atom_index) * n_channels_in;
    float accumulator = 0.0f;
    for (std::uint32_t channel_in_index = 0; channel_in_index < n_channels_in; ++channel_in_index) {
        std::uint64_t const weight_offset = (static_cast<std::uint64_t>(channel_in_index) * n_elements +
                                             static_cast<std::uint64_t>(element_index)) *
                                                n_channels_out +
                                            channel_out;
        accumulator += weights[weight_offset] * input_row[channel_in_index];
    }
    output[static_cast<std::uint64_t>(atom_index) * n_channels_out + channel_out] = scale * accumulator;
}

inline void launch_mace_skip_tp(std::uint32_t n_atoms, std::uint32_t n_channels_in, std::uint32_t n_elements,
                                std::uint32_t n_channels_out, std::uint32_t irrep_dim,
                                std::uint32_t active_irrep_block_count, std::int32_t const *atom_element,
                                float const *weights, float const *input, float *output, cudaStream_t stream) {
    if (n_atoms == 0) return;
    // For FullyConnectedTensorProduct(irreps_in_1, "n_elements x 0e", irreps_out)
    // with one instruction per matched (L_in, L_out) pair (input × 0e → output L = input L),
    // path normalisation 'element' gives alpha = mul_in_1 * mul_in_2 = c_in * n_elements
    // and weight scale = 1/sqrt(alpha).
    float const scale = 1.0f / sqrtf(static_cast<float>(n_channels_in) * static_cast<float>(n_elements));

    // Fast path: scalar-only (L=0) skip-tp, e.g. layer 0's first-layer
    // residual.  The general kernel's block(irrep_dim,1,1) collapses to a
    // 1-thread CTA when irrep_dim=1.
    if (irrep_dim == 1u && active_irrep_block_count == 1u) {
        constexpr std::uint32_t block_dim_x = 16, block_dim_y = 16;
        dim3 const block(block_dim_x, block_dim_y, 1);
        dim3 const grid((n_atoms + block_dim_y - 1u) / block_dim_y, (n_channels_out + block_dim_x - 1u) / block_dim_x,
                        1);
        mace_skip_tp_irrep1_kernel<<<grid, block, 0, stream>>>(n_atoms, n_channels_in, n_elements, n_channels_out,
                                                               scale, atom_element, weights, input, output);
        return;
    }

    dim3 const block(irrep_dim, 1, 1);
    dim3 const grid(n_atoms, n_channels_out, 1);
    mace_element_conditioned_residual_kernel<<<grid, block, 0, stream>>>(
        n_atoms, n_channels_in, n_elements, n_channels_out, irrep_dim, active_irrep_block_count, scale, atom_element,
        weights, input, output);
}

/** Element-wise add: `dst += src` for the residual sum. */
inline __global__ void mace_add_in_place_kernel(std::uint32_t n, float const *src, float *dst) {
    std::uint32_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index >= n) return;
    dst[element_index] += src[element_index];
}

inline void launch_mace_add_inplace(std::uint32_t n, float const *src, float *dst, cudaStream_t stream) {
    if (n == 0) return;
    constexpr std::uint32_t block = 256;
    std::uint32_t const grid = (n + block - 1u) / block;
    mace_add_in_place_kernel<<<grid, block, 0, stream>>>(n, src, dst);
}

/** Element-wise scale `dst = src * alpha`. */
inline __global__ void mace_scale_in_place_kernel(std::uint32_t n, float const *src, float alpha, float *dst) {
    std::uint32_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index >= n) return;
    dst[element_index] = src[element_index] * alpha;
}

inline void launch_mace_scale(std::uint32_t n, float const *src, float alpha, float *dst, cudaStream_t stream) {
    if (n == 0) return;
    constexpr std::uint32_t block = 256;
    std::uint32_t const grid = (n + block - 1u) / block;
    mace_scale_in_place_kernel<<<grid, block, 0, stream>>>(n, src, alpha, dst);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
