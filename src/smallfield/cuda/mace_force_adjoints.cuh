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
 *  Phase-3 status: structural skeleton + closed-form leaves (D5, D6, D7).
 *  The middle of the chain (D3, D4) — symmetric contraction and tensor
 *  product adjoints — still carries TODO markers; the rationale is that
 *  forward correctness must land first (Phase 1 closes the per-config
 *  energy gap, Phase 2 swaps to bf16) before adjoint validation against
 *  `torch.autograd.grad` is meaningful for those middle stages.
 *  ----------------------------------------------------------------------
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

#include "smallfield/types/mace_types.hpp"
#include "smallfield/cuda/mace_linear_kernel.cuh"  // mace_silu_normalize_to_unit_second_moment_constant_k

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

// =========================================================================
// D5.  Spherical harmonics adjoint — closed-form polynomial derivatives.
// =========================================================================

namespace mace_sph_adjoint_detail_ {

/**
 *  @brief Component-wise derivatives of the L=0..3 e3nn `component`-normalised
 *         real spherical harmonics y_lm(r̂) w.r.t. (r̂_x, r̂_y, r̂_z).
 *
 *  Each output row is written as three fp32 columns (dx, dy, dz).  The
 *  derivatives are the symbolic derivatives of the polynomials in
 *  `mace_sph_detail_::sph_harm_l0123_`, multiplied by the same per-L
 *  `component` factor sqrt(2L+1).
 *
 *  Layout matches the forward `sph_harm` ordering — index 0..15 over (l, m).
 *  Output `dy_dxyz_out` has shape [16][3] and the caller multiplies by the
 *  upstream gradient `dsph_harm[edge, lm]` to accumulate `dr_unit[edge, k]`.
 */
__device__ __forceinline__ void sph_harm_l0123_derivative_(
    float x, float y, float z,
    float *dy_dxyz_out  // [16, 3] flattened: dy_dxyz_out[3*lm + k]
) {
    constexpr float kSqrt3   = 1.7320508075688772f;
    constexpr float kSqrt5o6 = 0.9128709291752769f;   // sqrt(5/6)
    constexpr float kSqrt15  = 3.872983346207417f;    // sqrt(15)
    constexpr float kSqrt3o8 = 0.6123724356957945f;   // sqrt(3/8)

    // Component normalisation, per L block (matches forward).
    // L=0 factor is sqrt(1) = 1 — its derivatives are all zero so we elide it.
    constexpr float kC1 = 1.7320508075688772f;        // sqrt(3)
    constexpr float kC2 = 2.2360679774997896f;        // sqrt(5)
    constexpr float kC3 = 2.6457513110645907f;        // sqrt(7)

    auto write = [&](int lm, float dx, float dy, float dz) {
        dy_dxyz_out[3 * lm + 0] = dx;
        dy_dxyz_out[3 * lm + 1] = dy;
        dy_dxyz_out[3 * lm + 2] = dz;
    };

    // ------- L = 0 -------
    // sh_0_0 = 1
    write(0, 0.0f, 0.0f, 0.0f);

    // ------- L = 1 -------
    // sh_1_0 = x; sh_1_1 = y; sh_1_2 = z
    write(1, kC1 * 1.0f, 0.0f,        0.0f);
    write(2, 0.0f,        kC1 * 1.0f, 0.0f);
    write(3, 0.0f,        0.0f,        kC1 * 1.0f);

    // ------- L = 2 -------
    // sh_2_0 = sqrt3 * x * z
    write(4, kC2 * kSqrt3 * z, 0.0f,                kC2 * kSqrt3 * x);
    // sh_2_1 = sqrt3 * x * y
    write(5, kC2 * kSqrt3 * y, kC2 * kSqrt3 * x,  0.0f);
    // sh_2_2 = y^2 - 0.5 * (x^2 + z^2)  =>  d/dx = -x, d/dy = 2y, d/dz = -z
    write(6, kC2 * (-x),        kC2 * (2.0f * y),  kC2 * (-z));
    // sh_2_3 = sqrt3 * y * z
    write(7, 0.0f,               kC2 * kSqrt3 * z, kC2 * kSqrt3 * y);
    // sh_2_4 = (sqrt3/2) * (z^2 - x^2)  =>  d/dx = -sqrt3 * x, d/dz = sqrt3 * z
    write(8, kC2 * (-kSqrt3 * x), 0.0f,            kC2 * (kSqrt3 * z));

    // ------- L = 3 -------
    // sh_3_0 = sqrt(5/6) * (sh_2_0 * z + sh_2_4 * x)
    //        = sqrt(5/6) * sqrt3 * x * (1.5*z^2 - 0.5*x^2)
    //   d/dx = sqrt(5/6) * (sqrt3 * z^2 + (sqrt3/2)*(z^2 - 3x^2))
    //   d/dy = 0
    //   d/dz = sqrt(5/6) * 3 * sqrt3 * x * z
    {
        float const dx = kSqrt5o6 * (kSqrt3 * z * z + (kSqrt3 * 0.5f) * (z * z - 3.0f * x * x));
        float const dz = kSqrt5o6 * (3.0f * kSqrt3 * x * z);
        write(9, kC3 * dx, 0.0f, kC3 * dz);
    }
    // sh_3_1 = sqrt5 * sh_2_0 * y = sqrt15 * x * y * z
    write(10, kC3 * kSqrt15 * y * z, kC3 * kSqrt15 * x * z, kC3 * kSqrt15 * x * y);
    // sh_3_2 = sqrt(3/8) * (4 y^2 - x^2 - z^2) * x
    //   d/dx = sqrt(3/8) * (4 y^2 - 3 x^2 - z^2)
    //   d/dy = sqrt(3/8) * 8 x y
    //   d/dz = sqrt(3/8) * (-2 x z)
    {
        float const dx = kSqrt3o8 * (4.0f * y * y - 3.0f * x * x - z * z);
        float const dy_ = kSqrt3o8 * (8.0f * x * y);
        float const dz = kSqrt3o8 * (-2.0f * x * z);
        write(11, kC3 * dx, kC3 * dy_, kC3 * dz);
    }
    // sh_3_3 = 0.5 * y * (2 y^2 - 3 (x^2 + z^2)) = y^3 - 1.5 * y * (x^2 + z^2)
    //   d/dx = -3 x y
    //   d/dy = 3 y^2 - 1.5 (x^2 + z^2)
    //   d/dz = -3 y z
    {
        float const dx = -3.0f * x * y;
        float const dy_ = 3.0f * y * y - 1.5f * (x * x + z * z);
        float const dz = -3.0f * y * z;
        write(12, kC3 * dx, kC3 * dy_, kC3 * dz);
    }
    // sh_3_4 = sqrt(3/8) * z * (4 y^2 - x^2 - z^2)
    //   d/dx = sqrt(3/8) * (-2 x z)
    //   d/dy = sqrt(3/8) * 8 y z
    //   d/dz = sqrt(3/8) * (4 y^2 - x^2 - 3 z^2)
    {
        float const dx = kSqrt3o8 * (-2.0f * x * z);
        float const dy_ = kSqrt3o8 * (8.0f * y * z);
        float const dz = kSqrt3o8 * (4.0f * y * y - x * x - 3.0f * z * z);
        write(13, kC3 * dx, kC3 * dy_, kC3 * dz);
    }
    // sh_3_5 = sqrt5 * sh_2_4 * y = (sqrt15 / 2) * y * (z^2 - x^2)
    //   d/dx = -sqrt15 * x * y
    //   d/dy = (sqrt15 / 2) * (z^2 - x^2)
    //   d/dz =  sqrt15 * y * z
    {
        float const dx = -kSqrt15 * x * y;
        float const dy_ = (kSqrt15 * 0.5f) * (z * z - x * x);
        float const dz = kSqrt15 * y * z;
        write(14, kC3 * dx, kC3 * dy_, kC3 * dz);
    }
    // sh_3_6 = sqrt(5/6) * (sh_2_4 * z - sh_2_0 * x)
    //        = sqrt(5/6) * sqrt3 * (0.5 z^3 - 1.5 x^2 z)
    //   d/dx = sqrt(5/6) * (-3 sqrt3 * x * z)
    //   d/dy = 0
    //   d/dz = sqrt(5/6) * (sqrt3 / 2) * (3 z^2 - 3 x^2)
    {
        float const dx = kSqrt5o6 * (-3.0f * kSqrt3 * x * z);
        float const dz = kSqrt5o6 * (kSqrt3 * 0.5f) * (3.0f * z * z - 3.0f * x * x);
        write(15, kC3 * dx, 0.0f, kC3 * dz);
    }
}

} // namespace mace_sph_adjoint_detail_

/**
 *  @brief Adjoint of `mace_sph_harm_kernel`.
 *
 *  Forward: y_lm = polynomial in (x, y, z) with `component` normalisation
 *  baked in.  Backward: dr_unit_k = Σ_lm (dy_dxyz[lm, k]) * dsph_harm[lm].
 *
 *  Inputs:  `r_unit` [n_edges, 3]  — same input the forward used.
 *           `dsph_harm` [n_edges, 16] — upstream gradient of the loss w.r.t.
 *                                       sph_harm.
 *  Output:  `dr_unit` [n_edges, 3] — written, not accumulated.  Caller
 *           composes with the geometry adjoint (D6).
 *
 *  Threading: one thread per edge; 128 threads per block.  No shared memory.
 */
__global__ void mace_sph_harm_force_adjoint_kernel(
    std::uint32_t n_edges,
    float const *r_unit,        // [n_edges, 3]
    float const *dsph_harm,     // [n_edges, 16]
    float *dr_unit              // [n_edges, 3]   — overwritten
) {
    std::uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_index >= n_edges) return;

    float const x = r_unit[3 * edge_index + 0];
    float const y = r_unit[3 * edge_index + 1];
    float const z = r_unit[3 * edge_index + 2];

    float dy_dxyz[16 * 3];
    mace_sph_adjoint_detail_::sph_harm_l0123_derivative_(x, y, z, dy_dxyz);

    float accumulator_x = 0.0f, accumulator_y = 0.0f, accumulator_z = 0.0f;
    for (std::uint32_t irrep_component_index = 0; irrep_component_index < 16u; ++irrep_component_index) {
        float const upstream = dsph_harm[16u * edge_index + irrep_component_index];
        accumulator_x += upstream * dy_dxyz[3 * irrep_component_index + 0];
        accumulator_y += upstream * dy_dxyz[3 * irrep_component_index + 1];
        accumulator_z += upstream * dy_dxyz[3 * irrep_component_index + 2];
    }
    dr_unit[3 * edge_index + 0] = accumulator_x;
    dr_unit[3 * edge_index + 1] = accumulator_y;
    dr_unit[3 * edge_index + 2] = accumulator_z;
}

inline void launch_mace_sph_harm_force_adjoint(
    std::uint32_t n_edges,
    float const *r_unit,
    float const *dsph_harm,
    float *dr_unit,
    cudaStream_t stream
) {
    if (n_edges == 0) return;
    constexpr std::uint32_t block = 128;
    std::uint32_t const grid = (n_edges + block - 1u) / block;
    mace_sph_harm_force_adjoint_kernel<<<grid, block, 0, stream>>>(n_edges, r_unit, dsph_harm, dr_unit);
}

// =========================================================================
// D6.  Radial basis × cutoff envelope + edge geometry adjoint.
// =========================================================================

/**
 *  @brief Adjoint of (radial-basis × cutoff) and edge geometry, fused.
 *
 *  Forward decomposition:
 *      r_vec      = positions[target] - positions[source]      (1)
 *      r          = ||r_vec||                                  (2)
 *      r_unit     = r_vec / r                                  (3)
 *      env(r)     = 1 + a u^p + b u^(p+1) + c u^(p+2)          (4)   u = r / r_max
 *      b_n(r)     = prefactor * sin(w_n r) / r                  (5)
 *      rb[n]      = b_n(r) * env(r)                            (6)
 *
 *  Reverse:
 *      d_rb -> d_r:        Σ_n d rb[n] * d rb[n] / d r
 *                           = env * Σ_n d rb[n] * (prefactor / r^2) * (w_n r cos(w_n r) - sin(w_n r))
 *                             + d env / d r * Σ_n d rb[n] * b_n(r)
 *      d_r_unit -> d_r_vec, d_r: comes purely from the sph-harm/tensor-product
 *                                chain (D5) — d r / d r_unit = 0 by construction
 *                                (r_unit is normalised), so we propagate d r_unit
 *                                only into d r_vec (not into d r).
 *      d r / d r_vec:        r_unit
 *      d r_unit_k / d r_vec_j = (delta_kj - r_unit_k * r_unit_j) / r
 *
 *  Then d r_vec contributes:
 *      dpositions[target] +=  d_r_vec
 *      dpositions[source] += -d_r_vec
 *
 *  Threading: one thread per edge; 128 threads per block; atomic adds into
 *  the shared `dpositions` buffer (which is later negated to produce forces).
 *
 *  All math fp32; envelope clamped to zero outside r_max (matches forward).
 */
__global__ void mace_radial_basis_and_edge_geometry_force_adjoint_kernel(
    std::uint32_t n_edges,
    std::uint32_t n_radial,
    float r_max,
    float bessel_prefactor,
    float cutoff_p,
    float const *bessel_weights,    // [n_radial]
    float const *r_norm,            // [n_edges]
    float const *r_unit,            // [n_edges, 3]
    std::int32_t const *edge_source_atom,
    std::int32_t const *edge_target_atom,
    float const *dradial_basis,     // [n_edges, n_radial]
    float const *dr_unit,           // [n_edges, 3]      (from D5)
    float *dpositions               // [n_atoms, 3]      (atomic-accumulated)
) {
    std::uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_index >= n_edges) return;

    float const r = r_norm[edge_index];
    float const ux = r_unit[3 * edge_index + 0];
    float const uy = r_unit[3 * edge_index + 1];
    float const uz = r_unit[3 * edge_index + 2];

    // -- Recompute env(r) and d env / d r (clamp matches forward).
    float env = 0.0f;
    float denv_dr = 0.0f;
    if (r < r_max) {
        float const u   = r / r_max;
        float const p   = cutoff_p;
        float const a   = -(p + 1.0f) * (p + 2.0f) * 0.5f;
        float const b_  =  p * (p + 2.0f);
        float const c   = -p * (p + 1.0f) * 0.5f;
        float const up_minus1 = powf(u, p - 1.0f);    // u^(p-1)
        float const up        = up_minus1 * u;        // u^p
        float const up1       = up * u;               // u^(p+1)
        float const up2       = up1 * u;              // u^(p+2)
        env = 1.0f + a * up + b_ * up1 + c * up2;
        // d/dr (u^k) = (k / r_max) * u^(k-1)
        float const inv_r_max = 1.0f / r_max;
        denv_dr = inv_r_max * (a * p * up_minus1
                              + b_ * (p + 1.0f) * up
                              + c   * (p + 2.0f) * up1);
    }

    // -- Bessel chain: accumulate dr_from_basis = Σ_n d rb[n] * d rb[n] / d r.
    float const inv_r  = 1.0f / r;
    float const inv_r2 = inv_r * inv_r;
    float dr_from_basis = 0.0f;
    for (std::uint32_t n = 0; n < n_radial; ++n) {
        float const w_n  = bessel_weights[n];
        float const arg  = w_n * r;
        float       sin_arg, cos_arg;
        __sincosf(arg, &sin_arg, &cos_arg);
        float const b_n  = bessel_prefactor * sin_arg * inv_r;
        // d b_n / d r = prefactor * (w_n * r * cos(w_n r) - sin(w_n r)) / r^2
        float const dbn_dr = bessel_prefactor * (w_n * r * cos_arg - sin_arg) * inv_r2;
        // d rb[n] / d r = dbn_dr * env + b_n * denv_dr
        float const drb_dr = dbn_dr * env + b_n * denv_dr;
        float const upstream = dradial_basis[edge_index * n_radial + n];
        dr_from_basis += upstream * drb_dr;
    }

    // -- Geometry chain.
    //   r       = ||r_vec||
    //   d r / d r_vec_k = r_unit_k
    //   r_unit  = r_vec / r
    //   d r_unit_k / d r_vec_j = (delta_kj - r_unit_k * r_unit_j) / r
    //
    // Total d r_vec = dr_from_basis * r_unit  +  (dr_unit - r_unit (r_unit · dr_unit)) / r
    float const dr_unit_x = dr_unit[3 * edge_index + 0];
    float const dr_unit_y = dr_unit[3 * edge_index + 1];
    float const dr_unit_z = dr_unit[3 * edge_index + 2];
    float const u_dot_dru = ux * dr_unit_x + uy * dr_unit_y + uz * dr_unit_z;
    float const inv_r_for_unit = inv_r;

    float const drvec_x = dr_from_basis * ux + (dr_unit_x - ux * u_dot_dru) * inv_r_for_unit;
    float const drvec_y = dr_from_basis * uy + (dr_unit_y - uy * u_dot_dru) * inv_r_for_unit;
    float const drvec_z = dr_from_basis * uz + (dr_unit_z - uz * u_dot_dru) * inv_r_for_unit;

    int const src = edge_source_atom[edge_index];
    int const dst = edge_target_atom[edge_index];

    atomicAdd(dpositions + 3 * dst + 0,  drvec_x);
    atomicAdd(dpositions + 3 * dst + 1,  drvec_y);
    atomicAdd(dpositions + 3 * dst + 2,  drvec_z);
    atomicAdd(dpositions + 3 * src + 0, -drvec_x);
    atomicAdd(dpositions + 3 * src + 1, -drvec_y);
    atomicAdd(dpositions + 3 * src + 2, -drvec_z);
}

inline void launch_mace_radial_basis_and_edge_geometry_force_adjoint(
    types::mace_arch_t const &arch,
    std::uint32_t n_edges,
    float const *bessel_weights,
    float const *r_norm,
    float const *r_unit,
    std::int32_t const *edge_source_atom,
    std::int32_t const *edge_target_atom,
    float const *dradial_basis,
    float const *dr_unit,
    float *dpositions,
    cudaStream_t stream
) {
    if (n_edges == 0) return;
    constexpr std::uint32_t block = 128;
    std::uint32_t const grid = (n_edges + block - 1u) / block;
    mace_radial_basis_and_edge_geometry_force_adjoint_kernel<<<grid, block, 0, stream>>>(
        n_edges, arch.n_radial, arch.r_max, arch.bessel_prefactor, arch.cutoff_p,
        bessel_weights, r_norm, r_unit, edge_source_atom, edge_target_atom,
        dradial_basis, dr_unit, dpositions);
}

/**
 *  @brief Final negate: forces[a, k] = -dpositions[a, k].
 *
 *  Lives as its own kernel because multiple adjoint passes (sph-harm path,
 *  radial path, future stages) all atomic-add into `dpositions` before the
 *  final negation.  Threading: 256 threads per block, one thread per
 *  scalar component (n_atoms * 3).
 */
__global__ void mace_force_negate_kernel(
    std::uint32_t n_atoms,
    float const *dpositions,    // [n_atoms, 3]
    float *forces               // [n_atoms, 3]
) {
    std::uint32_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t total = n_atoms * 3u;
    if (element_index >= total) return;
    forces[element_index] = -dpositions[element_index];
}

inline void launch_mace_force_negate(std::uint32_t n_atoms, float const *dpositions, float *forces, cudaStream_t stream) {
    if (n_atoms == 0) return;
    constexpr std::uint32_t block = 256;
    std::uint32_t const grid = (n_atoms * 3u + block - 1u) / block;
    mace_force_negate_kernel<<<grid, block, 0, stream>>>(n_atoms, dpositions, forces);
}

inline void launch_mace_position_force_adjoint(std::uint32_t /*n_edges*/, cudaStream_t /*stream*/) {
    // Convenience wrapper kept for the existing top-level launcher signature.
    // The two real launchers are launch_mace_radial_basis_and_edge_geometry_force_adjoint
    // and launch_mace_force_negate; the driver calls them directly.
}

// =========================================================================
// D7.  Readout adjoints — linear and nonlinear MLP.
// =========================================================================

/**
 *  @brief Adjoint of `mace_readout_linear_kernel`.
 *
 *  Forward:    e[a] = scale * Σ_c W[c] * inv[a, c]
 *  Backward:   d inv[a, c] = scale * W[c] * d e[a]
 *
 *  No reduction, no atomics — pure elementwise.  Threading mirrors the
 *  scatter adjoint: one block per atom, each thread handles a strided
 *  channel.  Output is overwritten (caller chooses to add via a separate
 *  accumulator if multiple readouts feed the same atom — which they do in
 *  the layer-0 + layer-1 sum, handled by `mace_energy_reduce_force_adjoint_kernel`
 *  pre-broadcast).  Here we accumulate (atomicAdd-free since each thread
 *  owns its own (atom, channel)).
 */
__global__ void mace_readout_linear_force_adjoint_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    float scale,                            // 1/sqrt(n_channels)
    float const *readout_w,                 // [n_channels]
    float const *grad_node_energy,          // [n_atoms]   d e[a]
    float *dnode_scalar_invariants          // [n_atoms, n_channels]   accumulated
) {
    std::uint32_t atom_index = blockIdx.x;
    if (atom_index >= n_atoms) return;
    float const upstream = grad_node_energy[atom_index];
    for (std::uint32_t channel_index = threadIdx.x; channel_index < n_channels; channel_index += blockDim.x) {
        float const w = readout_w[channel_index];
        std::uint64_t const out_offset = static_cast<std::uint64_t>(atom_index) * n_channels + channel_index;
        // Accumulate so a downstream pass that already populated dnode_scalar_invariants
        // (e.g. the MLP readout on the same layer) is preserved.
        dnode_scalar_invariants[out_offset] += scale * w * upstream;
    }
}

inline void launch_mace_readout_linear_force_adjoint(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    float scale,
    float const *readout_w,
    float const *grad_node_energy,
    float *dnode_scalar_invariants,
    cudaStream_t stream
) {
    if (n_atoms == 0) return;
    constexpr std::uint32_t block = 32;
    mace_readout_linear_force_adjoint_kernel<<<n_atoms, block, 0, stream>>>(
        n_atoms, n_channels, scale, readout_w, grad_node_energy, dnode_scalar_invariants);
}

/**
 *  @brief Adjoint of `mace_readout_nonlinear_mlp_kernel`.
 *
 *  Forward:
 *      pre_act[a, k] = scale1 * Σ_c W1[c, k] * inv[a, c]
 *      h[a, k]       = silu(pre_act[a, k]) * normalize2mom_constant
 *      e[a]          = scale2 * Σ_k W2[k] * h[a, k]
 *
 *  We did not save `pre_act[a, k]` in the forward (cheap to recompute).
 *
 *  Reverse:
 *      d h[a, k]      = scale2 * W2[k] * d e[a]
 *      d pre_act[a, k] = d h[a, k] * normalize2mom_constant *
 *                        ( sigmoid(p) + p * sigmoid(p) * (1 - sigmoid(p)) )
 *                      = d h[a, k] * normalize2mom_constant *
 *                          sigmoid(p) * (1 + p * (1 - sigmoid(p)))
 *      d inv[a, c]    = scale1 * Σ_k W1[c, k] * d pre_act[a, k]
 *
 *  Threading: one block per atom; the block first cooperatively recomputes
 *  pre_act[*, k] for k in [0, hidden_dim) and the corresponding d pre_act,
 *  storing them in shared memory; then each thread strides over channels
 *  and accumulates into d inv[a, c].
 *
 *  Atomic adds: none — we accumulate (`+=`) into d inv[a, c] non-atomically
 *  because each (atom, channel) is owned by exactly one thread within the
 *  block.  Caller is expected to have either zero-initialised the buffer
 *  or already populated it from another readout (the linear path); we add
 *  on top.
 */
__global__ void mace_readout_nonlinear_mlp_force_adjoint_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t hidden_dim,
    float scale1,                           // 1/sqrt(n_channels)
    float scale2,                           // 1/sqrt(hidden_dim)
    float const *node_scalar_invariants,    // [n_atoms, n_channels]    forward input
    float const *w1,                        // [n_channels, hidden_dim]
    float const *w2,                        // [hidden_dim]
    float const *grad_node_energy,          // [n_atoms]   d e[a]
    float *dnode_scalar_invariants          // [n_atoms, n_channels]    accumulated
) {
    std::uint32_t atom_index = blockIdx.x;
    if (atom_index >= n_atoms) return;
    extern __shared__ float dpre_act_shared[];   // [hidden_dim]
    if (threadIdx.x < hidden_dim) {
        std::uint32_t hidden_index = threadIdx.x;
        // Recompute pre_act[a, k].
        float accumulator = 0.0f;
        for (std::uint32_t channel_index = 0; channel_index < n_channels; ++channel_index) {
            accumulator += node_scalar_invariants[static_cast<std::uint64_t>(atom_index) * n_channels + channel_index]
                 * w1[static_cast<std::uint64_t>(channel_index) * hidden_dim + hidden_index];
        }
        float const pre_act = scale1 * accumulator;
        float const sigmoid_value = 1.0f / (1.0f + __expf(-pre_act));
        // d silu(p)/dp = sigmoid(p) + p * sigmoid(p) * (1 - sigmoid(p))
        //              = sigmoid(p) * (1 + p * (1 - sigmoid(p)))
        float const dsilu_dp = sigmoid_value * (1.0f + pre_act * (1.0f - sigmoid_value));
        // d h[a, k] = scale2 * W2[k] * d e[a]
        float const dh = scale2 * w2[hidden_index] * grad_node_energy[atom_index];
        // d pre_act[a, k] = d h[a, k] * normalize2mom_constant * dsilu_dp
        dpre_act_shared[hidden_index] = dh * mace_silu_normalize_to_unit_second_moment_constant_k * dsilu_dp;
    }
    __syncthreads();

    // d inv[a, c] += scale1 * Σ_k W1[c, k] * d pre_act[a, k]
    for (std::uint32_t channel_index = threadIdx.x; channel_index < n_channels; channel_index += blockDim.x) {
        float accumulator = 0.0f;
        for (std::uint32_t hidden_index = 0; hidden_index < hidden_dim; ++hidden_index) {
            accumulator += w1[static_cast<std::uint64_t>(channel_index) * hidden_dim + hidden_index]
                 * dpre_act_shared[hidden_index];
        }
        dnode_scalar_invariants[static_cast<std::uint64_t>(atom_index) * n_channels + channel_index] += scale1 * accumulator;
    }
}

inline void launch_mace_readout_nonlinear_mlp_force_adjoint(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t hidden_dim,
    float scale1,
    float scale2,
    float const *node_scalar_invariants,
    float const *w1,
    float const *w2,
    float const *grad_node_energy,
    float *dnode_scalar_invariants,
    cudaStream_t stream
) {
    if (n_atoms == 0) return;
    constexpr std::uint32_t block = 32;
    std::uint32_t const shared_bytes = hidden_dim * sizeof(float);
    mace_readout_nonlinear_mlp_force_adjoint_kernel<<<n_atoms, block, shared_bytes, stream>>>(
        n_atoms, n_channels, hidden_dim, scale1, scale2,
        node_scalar_invariants, w1, w2, grad_node_energy, dnode_scalar_invariants);
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
