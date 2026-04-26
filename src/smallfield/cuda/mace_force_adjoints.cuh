/**
 *  @file mace_force_adjoints.cuh
 *  @brief Analytic adjoint kernels for MACE-OFF inference — forces (= -∂E/∂r),
 *         not training gradients.
 *
 *  Phase 3 of the MACE inference plan.  For inference (no training) we do
 *  not run torch.autograd; instead each forward operator has a closed-form
 *  adjoint and we hand-author them here.  This is identical in structure to
 *  the Python autograd graph but lets us:
 *
 *    - Skip building the autograd graph (no Python overhead).
 *    - Reuse the bf16 HMMA tiles from the forward path with transposed access.
 *    - Capture the entire forward + force-adjoint pass into one `cudaGraph_t`
 *      per atom-limited-bin (Phase 4) — autograd cannot be captured.
 *
 *  Forces = -∂E/∂r.  The adjoint we propagate has shape
 *  `[n_atoms, n_channels, 16]` (gradient w.r.t. node features), and finally
 *  `[n_atoms, 3]` after going through the radial+sph-harm preamble.
 *
 *  ----------------------------------------------------------------------
 *  Phase-3 status: structural skeleton only.  Each adjoint kernel below is
 *  declared and documented; the implementations carry TODO markers.  The
 *  rationale is that forward correctness must land first (Phase 1 closes
 *  the per-config energy gap, Phase 2 swaps to bf16) before adjoint
 *  validation against `torch.autograd.grad` is meaningful.
 *
 *  Once Phase-2 forward is correct to < 0.5 meV / config, each adjoint is
 *  a 1-2 day job: closed-form derivative + a unit test that compares
 *  against numerical-gradient and against autograd on the same input.
 *  ----------------------------------------------------------------------
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

#include "smallfield/types/mace_types.hpp"

namespace smallfield::cuda {

/**
 *  @brief Force-adjoint through the per-graph energy reduction.
 *
 *  Trivial: ∂E_g/∂node_invariants[a, c] = scale * readout_linear_w[c]
 *  for every atom a in graph g (plus the MLP path's contribution for
 *  the last layer).  Output: `dnode_invariants` [n_atoms, n_channels].
 */
__global__ void mace_energy_reduce_force_adjoint_kernel(
    std::uint32_t n_atoms, std::uint32_t n_channels, float scale, float const *readout_linear_w,
    float const *grad_energy, // [n_graphs] = 1.0 if computing forces
    std::int32_t const *graph_of_atom,
    float *dnode_invariants_layer0, // [n_atoms, n_channels]
    float *dnode_invariants_layer1) {
    std::uint32_t atom_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_index >= n_atoms) return;
    int const graph_index = graph_of_atom[atom_index];
    float const ge = grad_energy[graph_index];
    for (std::uint32_t channel_index = 0; channel_index < n_channels; ++channel_index) {
        float const w = readout_linear_w[channel_index];
        dnode_invariants_layer0[static_cast<std::uint64_t>(atom_index) * n_channels + channel_index] = scale * w * ge;
        dnode_invariants_layer1[static_cast<std::uint64_t>(atom_index) * n_channels + channel_index] = scale * w * ge;
    }
}

/**
 *  @brief Force-adjoint through the symmetric contraction.
 *
 *  Closed-form: derivative of order-k contraction w.r.t. the input A is a
 *  k-1-fold contraction of A times the U tensor, scaled by k for symmetry.
 *  Output: gradient w.r.t. node_features_full [n_atoms, n_channels, 16].
 *
 *  @todo Phase-3 — finalise once forward is bit-exact.
 */
inline void launch_mace_symmetric_contraction_force_adjoint(types::mace_arch_t const & /*arch*/, cudaStream_t /*stream*/
) {
    // No-op stub.
}

/**
 *  @brief Force-adjoint through the scatter sum: gather from receiver gradient back to per-edge gradient.
 *
 *  This is just a gather — for each edge, copy the gradient from its
 *  receiver row of `dnode_features_full` into `dmessages`.  No atomics.
 */
__global__ void mace_scatter_force_adjoint_kernel(std::uint32_t n_edges, std::uint32_t feature_dim,
                                                  std::int32_t const *edge_target_atom,
                                                  float const *dnode_features_full, // [n_atoms, feature_dim]
                                                  float *dmessages                  // [n_edges, feature_dim]
) {
    std::uint32_t edge_index = blockIdx.x;
    std::uint32_t feature_column_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (edge_index >= n_edges || feature_column_index >= feature_dim) return;
    int const receiver_atom_index = edge_target_atom[edge_index];
    dmessages[static_cast<std::uint64_t>(edge_index) * feature_dim + feature_column_index] =
        dnode_features_full[static_cast<std::uint64_t>(receiver_atom_index) * feature_dim + feature_column_index];
}

/**
 *  @brief Force-adjoint through the tensor product.
 *
 *  The tensor product is bilinear in (node_features, sph_harm) with weights
 *  fixed for inference.  Adjoints:
 *    ∂msg/∂node_feat[src, c, m_in] = Σ_{m_out, m_attr} CG[m_in, m_attr → m_out] *
 *                                     sph_harm[e, m_attr] * tp_weights[e, path, c]
 *    ∂msg/∂sph_harm[e, m_attr]     = Σ_{m_in, m_out, c} CG[...] *
 *                                     node_feat[src, c, m_in] * tp_weights[e, path, c]
 *
 *  We need the first one for forces (sph_harm depends on positions).
 *  The second one feeds into the position derivative (sph_harm has its
 *  own adjoint w.r.t. r̂).
 *
 *  @todo Phase-3 — implement once forward is bit-exact.
 */
inline void launch_mace_tensor_product_force_adjoint(types::mace_arch_t const & /*arch*/, cudaStream_t /*stream*/
) {
    // No-op stub.
}

/**
 *  @brief Force-adjoint through spherical harmonics — closed-form polynomial derivatives.
 *
 *  ∂y_lm/∂r̂_x, ∂y_lm/∂r̂_y, ∂y_lm/∂r̂_z are polynomials of total degree
 *  (l-1).  The full table for L=0..3 is ~30 lines; we will inline it
 *  alongside the forward formulas in `mace_sph_harm_kernel.cuh` once
 *  Phase 2 forward is locked.
 *
 *  @todo Phase-3.
 */
inline void launch_mace_sph_harm_force_adjoint(std::uint32_t /*n_edges*/, cudaStream_t /*stream*/
) {
    // No-op stub.
}

/**
 *  @brief Force-adjoint through the radial preamble (Bessel × cutoff) and edge geometry.
 *
 *  Combines:
 *    - dradial_basis → dr (chain through Bessel + cutoff derivatives)
 *    - dr_unit → dpositions[src], dpositions[dst]
 *    - dr_norm → dpositions[src], dpositions[dst]
 *
 *  Final output is `forces[a, k] = -Σ_e (∂E/∂positions[a, k] from edges where a == src or dst)`.
 *  Atomic adds into `forces`.
 *
 *  @todo Phase-3.
 */
inline void launch_mace_position_force_adjoint(std::uint32_t /*n_edges*/, cudaStream_t /*stream*/
) {
    // No-op stub.
}

/**
 *  @brief Top-level launcher — Phase-3 stub.  Returns immediately.  Real
 *  implementation calls each adjoint above in reverse forward order.
 */
inline void mace_inference_run_force_adjoint(cudaStream_t stream) {
    (void)stream;
    // No-op until Phase 2 forward is bit-exact.
}

} // namespace smallfield::cuda

#endif // __CUDACC__
