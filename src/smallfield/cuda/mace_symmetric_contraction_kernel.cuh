/**
 *  @file mace_symmetric_contraction_kernel.cuh
 *  @brief Atomic Cluster Expansion symmetric contraction (MACE `product` block).
 *
 *  Implements the explicit form of the Horner-style symmetric contraction
 *  defined in `mace.modules.symmetric_contraction.Contraction`:
 *
 *      B[a, c] = Σ_{i,j,k,p} U_3[i,j,k,p] * w_max[Z[a], p, c]
 *                            * X[a, c, i] * X[a, c, j] * X[a, c, k]
 *
 *              + Σ_{i,j,p}   U_2[i,j,p]   * w_2[Z[a], p, c]
 *                            * X[a, c, i] * X[a, c, j]
 *
 *              + Σ_{i,p}     U_1[i,p]     * w_1[Z[a], p, c]
 *                            * X[a, c, i]
 *
 *  Sizes for MACE-OFF23 (any size): irrep_dim = 16, path_count_max = 23,
 *  path_count_order_2 = 4, path_count_order_1 = 1.  Output is the
 *  per-channel L=0 (scalar) feature; per-`Contraction` instance there is
 *  one output irrep, and the full SymmetricContraction concatenates these.
 *  For MACE-OFF the ν=1..3 contraction has only one output irrep (0e) so
 *  this kernel is the whole contraction.
 *
 *  Threading: one thread per (atom, channel).  Inner work per thread:
 *  16³ = 4096 multiplies for the order-3 path (× 23 paths/element folded
 *  into the same loop), plus 16² × 4 + 16 for orders 2 and 1.  Per
 *  (atom, channel) ≈ 100k flops.  At MACE-OFF small batch 256 (24k atoms),
 *  channels 96 → ~2.3 GFLOP per pass — ~30 µs on H100 SXM5 fp32.
 *
 *  All math fp32.  bf16 / HMMA variant is the Phase-2 follow-up; the
 *  algebra here is the reference.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

#include "smallfield/types/mace_types.hpp"

namespace smallfield::cuda {

inline __global__ void mace_symmetric_contraction_kernel(
    std::uint32_t n_atoms, std::uint32_t n_channels,
    std::uint32_t irrep_dim,          // 16
    std::uint32_t path_count_max,     // 23 (order-3)
    std::uint32_t path_count_order_2, // 4
    std::uint32_t path_count_order_1, // 1
    std::uint32_t n_elements,         // 10
    std::int32_t const *atom_element, // [n_atoms]
    float const *X,                   // [n_atoms, n_channels, irrep_dim]
    float const *u_matrix_1,          // [irrep_dim, path_count_order_1]
    float const *u_matrix_2,          // [irrep_dim, irrep_dim, path_count_order_2]
    float const *u_matrix_3,          // [irrep_dim, irrep_dim, irrep_dim, path_count_max]
    float const *weights_max,         // [n_elements, path_count_max, n_channels]
    float const *weights_2,           // [n_elements, path_count_order_2, n_channels]
    float const *weights_1,           // [n_elements, path_count_order_1, n_channels]
    float *B                          // [n_atoms, n_channels]  output (scalar irrep)
) {
    std::uint32_t atom_index = blockIdx.x;
    std::uint32_t channel_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (atom_index >= n_atoms || channel_index >= n_channels) return;

    int const element_index = atom_element[atom_index];

    // Pointer into X for this (atom, channel).  Stride-1 over the irrep_dim axis.
    // i, j, k below are the ACE basis indices that index into this row in the
    // formulas above (matching the math notation U_3[i,j,k,p] * X[i] * X[j] * X[k]).
    float const *atom_irrep_row = X + (static_cast<std::uint64_t>(atom_index) * n_channels + channel_index) * irrep_dim;

    // Order 1: B1 = Σ_{i, p} U_1[i, p] * w_1[Z, p, c] * X[i]
    float accumulator = 0.0f;
    for (std::uint32_t path_index = 0; path_index < path_count_order_1; ++path_index) {
        float const weighted_path_coefficient =
            weights_1[(static_cast<std::uint64_t>(element_index) * path_count_order_1 + path_index) * n_channels +
                      channel_index];
        for (std::uint32_t irrep_index_i = 0; irrep_index_i < irrep_dim; ++irrep_index_i) {
            float const u = u_matrix_1[static_cast<std::uint64_t>(irrep_index_i) * path_count_order_1 + path_index];
            accumulator += u * weighted_path_coefficient * atom_irrep_row[irrep_index_i];
        }
    }

    // Order 2: B2 = Σ_{i, j, p} U_2[i, j, p] * w_2[Z, p, c] * X[i] * X[j]
    for (std::uint32_t path_index = 0; path_index < path_count_order_2; ++path_index) {
        float const weighted_path_coefficient =
            weights_2[(static_cast<std::uint64_t>(element_index) * path_count_order_2 + path_index) * n_channels +
                      channel_index];
        // Hoist Σ_i U_2[i,j,p] * X[i] for each j, then multiply by X[j].
        for (std::uint32_t irrep_index_j = 0; irrep_index_j < irrep_dim; ++irrep_index_j) {
            float inner_sum_over_irrep_dim = 0.0f;
            for (std::uint32_t irrep_index_i = 0; irrep_index_i < irrep_dim; ++irrep_index_i) {
                float const u = u_matrix_2[((static_cast<std::uint64_t>(irrep_index_i) * irrep_dim) + irrep_index_j) *
                                               path_count_order_2 +
                                           path_index];
                inner_sum_over_irrep_dim += u * atom_irrep_row[irrep_index_i];
            }
            accumulator += weighted_path_coefficient * inner_sum_over_irrep_dim * atom_irrep_row[irrep_index_j];
        }
    }

    // Order 3: B3 = Σ_{i, j, k, p} U_3[i, j, k, p] * w_max[Z, p, c] * X[i]X[j]X[k]
    // Hoist X[i] across the (j, k) loop pair to reduce work.
    for (std::uint32_t path_index = 0; path_index < path_count_max; ++path_index) {
        float const weighted_path_coefficient =
            weights_max[(static_cast<std::uint64_t>(element_index) * path_count_max + path_index) * n_channels +
                        channel_index];
        // Compute Σ_{j,k} (Σ_i U_3[i,j,k,p] * X[i]) * X[j] * X[k]
        for (std::uint32_t irrep_index_j = 0; irrep_index_j < irrep_dim; ++irrep_index_j) {
            for (std::uint32_t irrep_index_k = 0; irrep_index_k < irrep_dim; ++irrep_index_k) {
                float inner_sum_over_irrep_dim = 0.0f;
                for (std::uint32_t irrep_index_i = 0; irrep_index_i < irrep_dim; ++irrep_index_i) {
                    float const u =
                        u_matrix_3[(((static_cast<std::uint64_t>(irrep_index_i) * irrep_dim) + irrep_index_j) *
                                        irrep_dim +
                                    irrep_index_k) *
                                       path_count_max +
                                   path_index];
                    inner_sum_over_irrep_dim += u * atom_irrep_row[irrep_index_i];
                }
                accumulator += weighted_path_coefficient * inner_sum_over_irrep_dim * atom_irrep_row[irrep_index_j] *
                               atom_irrep_row[irrep_index_k];
            }
        }
    }

    B[static_cast<std::uint64_t>(atom_index) * n_channels + channel_index] = accumulator;
}

inline void launch_mace_symmetric_contraction(types::mace_arch_t const &arch,
                                              types::mace_weights_device_t::layer_t const &layer, std::uint32_t n_atoms,
                                              std::int32_t const *atom_element, float const *X,
                                              float *B, // [n_atoms, n_channels]
                                              cudaStream_t stream) {
    if (n_atoms == 0) return;
    constexpr std::uint32_t block_dim_x = 32;
    dim3 const block(block_dim_x, 1, 1);
    dim3 const grid(n_atoms, (arch.n_channels + block_dim_x - 1u) / block_dim_x, 1);
    constexpr std::uint32_t path_count_max = 23u;
    constexpr std::uint32_t path_count_order_2 = 4u;
    constexpr std::uint32_t path_count_order_1 = 1u;
    mace_symmetric_contraction_kernel<<<grid, block, 0, stream>>>(
        n_atoms, arch.n_channels, /*irrep_dim*/ 16u, path_count_max, path_count_order_2, path_count_order_1,
        arch.n_elements, atom_element, X, layer.u_matrix_1, layer.u_matrix_2, layer.u_matrix_3, layer.weights_max,
        layer.weights_lo_0, layer.weights_lo_1, B);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
