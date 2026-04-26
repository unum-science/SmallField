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
 *  Phase-3 status: closed-form leaves (D5, D6, D7) plus matmul-shaped
 *  middle stages (D1 per-irrep linear, D2 tensor product + scatter,
 *  D3 element-conditioned residual, D4 symmetric contraction) — all
 *  fp32 reference adjoints validated against numerical-gradient on
 *  synthetic inputs to ≤ 1e-3 relative error.  bf16 / TK variants are
 *  a follow-up; driver-side wiring of the full backward chain is its
 *  own follow-up commit.
 *  ----------------------------------------------------------------------
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

#include "smallfield/types/mace_types.hpp"
#include "smallfield/cuda/mace_linear_kernel.cuh"  // mace_silu_normalize_to_unit_second_moment_constant_k, mace_irrep_block_index_for_component_
#include "smallfield/cuda/mace_tensor_product_kernel.cuh"  // mace_tensor_product_path_count_k

namespace smallfield::cuda {

/**
 *  @brief Force-adjoint through the per-graph energy reduction.
 *
 *  Trivial: ∂E_g/∂node_invariants[a, c] = scale * readout_linear_w[c]
 *  for every atom a in graph g (plus the MLP path's contribution for
 *  the last layer).  Output: `dnode_invariants` [n_atoms, n_channels].
 */
inline __global__ void mace_energy_reduce_force_adjoint_kernel(
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
 *  @brief Force-adjoint through the scatter sum: gather from receiver gradient back to per-edge gradient.
 *
 *  This is just a gather — for each edge, copy the gradient from its
 *  receiver row of `dnode_features_full` into `dmessages`.  No atomics.
 */
inline __global__ void mace_scatter_force_adjoint_kernel(std::uint32_t n_edges, std::uint32_t feature_dim,
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
inline __global__ void mace_sph_harm_force_adjoint_kernel(
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
inline __global__ void mace_radial_basis_and_edge_geometry_force_adjoint_kernel(
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
inline __global__ void mace_force_negate_kernel(
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
inline __global__ void mace_readout_linear_force_adjoint_kernel(
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
inline __global__ void mace_readout_nonlinear_mlp_force_adjoint_kernel(
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

// =========================================================================
// D1.  Per-irrep linear adjoint — transposed channel mix.
// =========================================================================

/**
 *  @brief Adjoint of `mace_linear_per_irrep_kernel`.
 *
 *  Forward: out[a, c_out, m] = scale * Σ_{c_in} W[L(m), c_in, c_out] * in[a, c_in, m]
 *  Reverse: din[a, c_in, m] += scale * Σ_{c_out} W[L(m), c_in, c_out] * dout[a, c_out, m]
 *
 *  Same matmul as the forward but with the c_in / c_out tile roles swapped —
 *  the same weight slab is read with the inner index over c_out instead of c_in.
 *  Output is accumulated with `+=` so the caller can chain into a buffer that
 *  already holds gradient from a parallel adjoint path; caller must pre-zero
 *  if no other adjoint feeds the buffer.
 *
 *  Threading: one thread per (atom, c_in, m).  Inner: n_channels_out multiplies.
 */
inline __global__ void mace_linear_per_irrep_force_adjoint_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_channels_out,
    std::uint32_t irrep_dim,                   // 16
    std::uint32_t active_irrep_block_count,
    float scale,                               // 1/sqrt(c_in)
    float const *weight_blocks,                // [n_blocks, c_in, c_out] — same as forward
    float const *upstream_grad_output,         // [n_atoms, c_out, irrep_dim]
    float *grad_input                          // [n_atoms, c_in, irrep_dim]   accumulated +=
) {
    std::uint32_t atom_index = blockIdx.x;
    std::uint32_t channel_in_index = blockIdx.y;
    std::uint32_t irrep_component_index = threadIdx.x;
    if (atom_index >= n_atoms || channel_in_index >= n_channels_in || irrep_component_index >= irrep_dim) return;

    std::uint32_t const irrep_block_index = mace_irrep_block_index_for_component_(irrep_component_index);
    if (irrep_block_index >= active_irrep_block_count) {
        // Inactive m-block contributes zero to the input gradient.
        return;
    }
    std::uint64_t const weight_block_offset = static_cast<std::uint64_t>(irrep_block_index) * n_channels_in * n_channels_out;
    std::uint64_t const weight_row_offset   = weight_block_offset + static_cast<std::uint64_t>(channel_in_index) * n_channels_out;
    float const *upstream_atom_row = upstream_grad_output + static_cast<std::uint64_t>(atom_index) * n_channels_out * irrep_dim;
    float accumulator = 0.0f;
    for (std::uint32_t channel_out_index = 0; channel_out_index < n_channels_out; ++channel_out_index) {
        float const w  = weight_blocks[weight_row_offset + channel_out_index];
        float const du = upstream_atom_row[channel_out_index * irrep_dim + irrep_component_index];
        accumulator += w * du;
    }
    std::uint64_t const grad_input_offset =
        (static_cast<std::uint64_t>(atom_index) * n_channels_in + channel_in_index) * irrep_dim + irrep_component_index;
    grad_input[grad_input_offset] += scale * accumulator;
}

/**
 *  @brief Specialized irrep_dim=1 (scalar-only L=0) per-irrep linear adjoint.
 *
 *  Equivalent to the general kernel above for the (irrep_dim=1, active=1)
 *  case but launched as a 16×16 = 256-thread CTA covering 16 atoms × 16 c_in
 *  per CTA.  The general launcher would pick `block(1,1,1)` = one thread per
 *  CTA, leaving warps mostly empty and burning scheduler overhead on
 *  n_atoms × n_in CTAs.
 *
 *  `din[a, c_in] += scale * Σ_{c_out} W[c_in, c_out] * dout[a, c_out]`
 *
 *  Output is accumulated with `+=`, matching the general kernel's contract.
 */
inline __global__ void mace_linear_per_irrep_irrep1_force_adjoint_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_channels_out,
    float scale,
    float const *__restrict__ weight,           // [c_in, c_out]
    float const *__restrict__ upstream_grad,    // [n_atoms, c_out]
    float *__restrict__ grad_input              // [n_atoms, c_in]   accumulated +=
) {
    std::uint32_t const atom_index   = blockIdx.x * blockDim.y + static_cast<std::uint32_t>(threadIdx.y);
    std::uint32_t const channel_in   = blockIdx.y * blockDim.x + static_cast<std::uint32_t>(threadIdx.x);
    if (atom_index >= n_atoms || channel_in >= n_channels_in) return;

    float const *weight_row = weight + static_cast<std::uint64_t>(channel_in) * n_channels_out;
    float const *upstream_row = upstream_grad + static_cast<std::uint64_t>(atom_index) * n_channels_out;
    float accumulator = 0.0f;
    for (std::uint32_t channel_out_index = 0; channel_out_index < n_channels_out; ++channel_out_index) {
        accumulator += upstream_row[channel_out_index] * weight_row[channel_out_index];
    }
    std::uint64_t const out_offset = static_cast<std::uint64_t>(atom_index) * n_channels_in + channel_in;
    grad_input[out_offset] += scale * accumulator;
}

inline void launch_mace_linear_per_irrep_force_adjoint(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_channels_out,
    std::uint32_t irrep_dim,
    std::uint32_t active_irrep_block_count,
    float const *weight_blocks_fp32,
    float const *upstream_grad_output_fp32,
    float *grad_input_fp32,
    cudaStream_t stream
) {
    if (n_atoms == 0) return;
    float const scale = 1.0f / sqrtf(static_cast<float>(n_channels_in));

    // Fast path: scalar-only (L=0) backward avoids the 1-thread-per-CTA shape
    // of the general kernel for irrep_dim=1.
    if (irrep_dim == 1u && active_irrep_block_count == 1u) {
        constexpr std::uint32_t block_dim_x = 16, block_dim_y = 16;
        dim3 const block(block_dim_x, block_dim_y, 1);
        dim3 const grid((n_atoms + block_dim_y - 1u) / block_dim_y,
                        (n_channels_in + block_dim_x - 1u) / block_dim_x, 1);
        mace_linear_per_irrep_irrep1_force_adjoint_kernel<<<grid, block, 0, stream>>>(
            n_atoms, n_channels_in, n_channels_out, scale,
            weight_blocks_fp32, upstream_grad_output_fp32, grad_input_fp32);
        return;
    }

    dim3 const block(irrep_dim, 1, 1);
    dim3 const grid(n_atoms, n_channels_in, 1);
    mace_linear_per_irrep_force_adjoint_kernel<<<grid, block, 0, stream>>>(
        n_atoms, n_channels_in, n_channels_out, irrep_dim, active_irrep_block_count, scale,
        weight_blocks_fp32, upstream_grad_output_fp32, grad_input_fp32);
}

// =========================================================================
// D2.  Tensor product + scatter adjoint — three gradient buffers.
// =========================================================================

/**
 *  @brief Force-adjoint through the fused (tensor product → scatter) chain.
 *
 *  Forward (per edge `e`, channel `c`, m-component `m`):
 *      msg[e, c, m]  =  scale * path_weight[e, L(m), c] * sph_harm[e, m]
 *                                * node_l0[edge_source_atom[e], c]
 *  followed by a scatter-add into the receiver's row of `node_features_full`.
 *
 *  Receiving the post-scatter upstream gradient `dnode_features_full[a, c, m]`,
 *  the adjoint emits three independent buffers:
 *
 *    dnode_l0[s, c]
 *      += scale * Σ_m path_weight[e, L(m), c] * sph_harm[e, m]
 *                       * dnode_features_full[edge_target_atom[e], c, m]
 *      where s = edge_source_atom[e].  Multiple edges can share a source atom,
 *      so this requires `atomicAdd`.
 *
 *    dpath_weights[e, p, c]
 *      = scale * Σ_{m ∈ path p} sph_harm[e, m] * node_l0[s, c]
 *                       * dnode_features_full[edge_target_atom[e], c, m]
 *      One edge owns each (e, p, c) tuple — no atomic needed.
 *
 *    dsph_harm[e, m]
 *      += scale * Σ_c path_weight[e, L(m), c] * node_l0[s, c]
 *                       * dnode_features_full[edge_target_atom[e], c, m]
 *      One edge owns each (e, m), but the work is sliced across channel-tiles
 *      below so we use `atomicAdd` for safety; the per-edge accumulator could
 *      be promoted to per-thread + warp reduce in a follow-up if profiling
 *      shows this is the hot path.
 *
 *  Threading: one block per edge × channel-tile (32 channels per block); each
 *  thread handles one channel.  Inside the block:
 *    - 16-wide loop over `m` accumulates the three gradient contributions
 *      simultaneously (one source-atom load + one weight-tuple per (e, c) is
 *      reused across all 16 m's, mirroring the forward kernel's organisation).
 *    - `dpath_weights` is written non-atomically (per-edge, per-channel
 *      ownership).  `dnode_l0` is `atomicAdd`-accumulated.  `dsph_harm` is
 *      `atomicAdd`-accumulated across channels.
 *  Pass `nullptr` for any of the three output buffers to skip its computation.
 */
inline __global__ void mace_tensor_product_and_scatter_force_adjoint_kernel(
    std::uint32_t n_edges,
    std::uint32_t n_channels,
    float scale,
    std::int32_t const *edge_source_atom,
    std::int32_t const *edge_target_atom,
    float const *node_l0_feature,                         // [n_atoms, n_channels]
    float const *sph_harm,                                // [n_edges, 16]
    float const *tensor_product_weights_per_edge,         // [n_edges, 4 * n_channels]
    float const *upstream_grad_node_messages_full,        // [n_atoms, n_channels, 16]
    float *grad_node_l0_feature,                          // [n_atoms, n_channels]   — atomicAdd, may be nullptr
    float *grad_tensor_product_weights_per_edge,          // [n_edges, 4 * n_channels] — non-atomic, may be nullptr
    float *grad_sph_harm                                  // [n_edges, 16]            — atomicAdd, may be nullptr
) {
    std::uint32_t edge_index = blockIdx.x;
    std::uint32_t channel_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (edge_index >= n_edges || channel_index >= n_channels) return;

    int const source_atom_index = edge_source_atom[edge_index];
    int const target_atom_index = edge_target_atom[edge_index];

    float const node_l0_feature_value =
        node_l0_feature[static_cast<std::uint64_t>(source_atom_index) * n_channels + channel_index];

    // Pre-load this (edge, channel)'s 4 path weights.
    std::uint64_t const path_weight_base = static_cast<std::uint64_t>(edge_index) * (mace_tensor_product_path_count_k * n_channels);
    float const path_weight_l0 = tensor_product_weights_per_edge[path_weight_base + 0u * n_channels + channel_index];
    float const path_weight_l1 = tensor_product_weights_per_edge[path_weight_base + 1u * n_channels + channel_index];
    float const path_weight_l2 = tensor_product_weights_per_edge[path_weight_base + 2u * n_channels + channel_index];
    float const path_weight_l3 = tensor_product_weights_per_edge[path_weight_base + 3u * n_channels + channel_index];

    std::uint64_t const sph_harm_offset = static_cast<std::uint64_t>(edge_index) * 16u;
    std::uint64_t const grad_msg_offset = (static_cast<std::uint64_t>(target_atom_index) * n_channels + channel_index) * 16u;

    // Per-path partial sums for `dpath_weights` (shape: 4) and `dnode_l0` (scalar).
    float dpath_weight_accumulator_l0 = 0.0f;
    float dpath_weight_accumulator_l1 = 0.0f;
    float dpath_weight_accumulator_l2 = 0.0f;
    float dpath_weight_accumulator_l3 = 0.0f;
    float dnode_l0_accumulator        = 0.0f;

    // Single 16-wide pass: m=0 (L=0), m=1..3 (L=1), m=4..8 (L=2), m=9..15 (L=3).
    // For each m we read the upstream gradient and the spherical-harmonic
    // value, then accumulate into the three output buffers.
    auto accumulate_m = [&](std::uint32_t m_index, float path_weight_for_block,
                            float &dpath_weight_block_accumulator) {
        float const sh = sph_harm[sph_harm_offset + m_index];
        float const du = upstream_grad_node_messages_full[grad_msg_offset + m_index];

        // dnode_l0[s, c] += scale * path_weight * sph_harm * du
        dnode_l0_accumulator += path_weight_for_block * sh * du;
        // dpath_weights[e, L, c] += scale * sph_harm * node_l0 * du   (summed over m's in L block)
        dpath_weight_block_accumulator += sh * node_l0_feature_value * du;
        // dsph_harm[e, m] += scale * path_weight * node_l0 * du   (summed over channels)
        if (grad_sph_harm != nullptr) {
            float const contribution = scale * path_weight_for_block * node_l0_feature_value * du;
            atomicAdd(grad_sph_harm + sph_harm_offset + m_index, contribution);
        }
    };

    accumulate_m(0u, path_weight_l0, dpath_weight_accumulator_l0);
    accumulate_m(1u, path_weight_l1, dpath_weight_accumulator_l1);
    accumulate_m(2u, path_weight_l1, dpath_weight_accumulator_l1);
    accumulate_m(3u, path_weight_l1, dpath_weight_accumulator_l1);
    accumulate_m(4u, path_weight_l2, dpath_weight_accumulator_l2);
    accumulate_m(5u, path_weight_l2, dpath_weight_accumulator_l2);
    accumulate_m(6u, path_weight_l2, dpath_weight_accumulator_l2);
    accumulate_m(7u, path_weight_l2, dpath_weight_accumulator_l2);
    accumulate_m(8u, path_weight_l2, dpath_weight_accumulator_l2);
    accumulate_m(9u,  path_weight_l3, dpath_weight_accumulator_l3);
    accumulate_m(10u, path_weight_l3, dpath_weight_accumulator_l3);
    accumulate_m(11u, path_weight_l3, dpath_weight_accumulator_l3);
    accumulate_m(12u, path_weight_l3, dpath_weight_accumulator_l3);
    accumulate_m(13u, path_weight_l3, dpath_weight_accumulator_l3);
    accumulate_m(14u, path_weight_l3, dpath_weight_accumulator_l3);
    accumulate_m(15u, path_weight_l3, dpath_weight_accumulator_l3);

    if (grad_node_l0_feature != nullptr) {
        atomicAdd(grad_node_l0_feature + static_cast<std::uint64_t>(source_atom_index) * n_channels + channel_index,
                  scale * dnode_l0_accumulator);
    }
    if (grad_tensor_product_weights_per_edge != nullptr) {
        grad_tensor_product_weights_per_edge[path_weight_base + 0u * n_channels + channel_index] = scale * dpath_weight_accumulator_l0;
        grad_tensor_product_weights_per_edge[path_weight_base + 1u * n_channels + channel_index] = scale * dpath_weight_accumulator_l1;
        grad_tensor_product_weights_per_edge[path_weight_base + 2u * n_channels + channel_index] = scale * dpath_weight_accumulator_l2;
        grad_tensor_product_weights_per_edge[path_weight_base + 3u * n_channels + channel_index] = scale * dpath_weight_accumulator_l3;
    }
}

inline void launch_mace_tensor_product_and_scatter_force_adjoint(
    std::uint32_t n_atoms,
    std::uint32_t n_edges,
    std::uint32_t n_channels,
    std::int32_t const *edge_source_atom,
    std::int32_t const *edge_target_atom,
    float const *node_l0_feature_fp32,
    float const *sph_harm_fp32,
    float const *tensor_product_weights_per_edge_fp32,
    float const *upstream_grad_node_messages_full_fp32,
    float *grad_node_l0_feature_fp32,
    float *grad_tensor_product_weights_per_edge_fp32,
    float *grad_sph_harm_fp32,
    cudaStream_t stream
) {
    (void)n_atoms;
    if (n_edges == 0) return;
    constexpr std::uint32_t block_dim_x = 32;
    dim3 const block(block_dim_x, 1, 1);
    dim3 const grid(n_edges, (n_channels + block_dim_x - 1u) / block_dim_x, 1);
    // Match the forward — empirically the radial-MLP weights already absorb
    // the e3nn TP path-weight factor, so scale = 1.
    float const scale = 1.0f;
    mace_tensor_product_and_scatter_force_adjoint_kernel<<<grid, block, 0, stream>>>(
        n_edges, n_channels, scale, edge_source_atom, edge_target_atom,
        node_l0_feature_fp32, sph_harm_fp32, tensor_product_weights_per_edge_fp32,
        upstream_grad_node_messages_full_fp32,
        grad_node_l0_feature_fp32, grad_tensor_product_weights_per_edge_fp32, grad_sph_harm_fp32);
}

// =========================================================================
// D3.  Element-conditioned residual adjoint — Z-indirect transposed channel mix.
// =========================================================================

/**
 *  @brief Adjoint of `mace_element_conditioned_residual_kernel` (a.k.a. skip_tp).
 *
 *  Forward: out[a, c_out, m] = scale * Σ_{c_in} W[L_b, c_in, Z[a], c_out] * in[a, c_in, m]
 *  Reverse: din[a, c_in, m] += scale * Σ_{c_out} W[L_b, c_in, Z[a], c_out] * dout[a, c_out, m]
 *
 *  Same shape as the per-irrep linear adjoint, with one extra Z-indirect index
 *  on the weight load.  Output accumulates with `+=` so the caller can chain
 *  into a buffer that already holds the per-irrep linear adjoint contribution.
 *
 *  Threading: one thread per (atom, c_in, m).  Inner: n_channels_out multiplies.
 */
inline __global__ void mace_element_conditioned_residual_force_adjoint_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_elements,
    std::uint32_t n_channels_out,
    std::uint32_t irrep_dim,
    std::uint32_t active_irrep_block_count,
    float scale,                                // 1/sqrt(c_in * n_elements)
    std::int32_t const *atom_element,           // [n_atoms]
    float const *weights,                       // [active_irrep_block_count, c_in, n_elements, c_out]
    float const *upstream_grad_output,          // [n_atoms, c_out, irrep_dim]
    float *grad_input                           // [n_atoms, c_in, irrep_dim]    accumulated +=
) {
    std::uint32_t atom_index = blockIdx.x;
    std::uint32_t channel_in_index = blockIdx.y;
    std::uint32_t irrep_component_index = threadIdx.x;
    if (atom_index >= n_atoms || channel_in_index >= n_channels_in || irrep_component_index >= irrep_dim) return;

    std::uint32_t const irrep_block_index = mace_irrep_block_index_for_component_(irrep_component_index);
    if (irrep_block_index >= active_irrep_block_count) return;

    int const element_index = atom_element[atom_index];
    std::uint64_t const weight_block_offset = static_cast<std::uint64_t>(irrep_block_index) * n_channels_in * n_elements * n_channels_out;
    std::uint64_t const weight_row_offset   = weight_block_offset
                                            + static_cast<std::uint64_t>(channel_in_index) * n_elements * n_channels_out
                                            + static_cast<std::uint64_t>(element_index)   * n_channels_out;
    float const *upstream_atom_row = upstream_grad_output + static_cast<std::uint64_t>(atom_index) * n_channels_out * irrep_dim;
    float accumulator = 0.0f;
    for (std::uint32_t channel_out_index = 0; channel_out_index < n_channels_out; ++channel_out_index) {
        float const w  = weights[weight_row_offset + channel_out_index];
        float const du = upstream_atom_row[channel_out_index * irrep_dim + irrep_component_index];
        accumulator += w * du;
    }
    std::uint64_t const grad_input_offset =
        (static_cast<std::uint64_t>(atom_index) * n_channels_in + channel_in_index) * irrep_dim + irrep_component_index;
    grad_input[grad_input_offset] += scale * accumulator;
}

/**
 *  @brief Specialized irrep_dim=1 skip-tp adjoint, mirroring the irrep_dim=1
 *         per-irrep linear adjoint fast path.  Avoids the 1-thread-per-CTA
 *         shape of the general kernel for the (irrep_dim=1, active=1) case.
 */
inline __global__ void mace_skip_tp_irrep1_force_adjoint_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_elements,
    std::uint32_t n_channels_out,
    float scale,
    std::int32_t const *__restrict__ atom_element,
    float const *__restrict__ weights,                 // [n_in, n_elements, n_out]
    float const *__restrict__ upstream_grad,           // [n_atoms, n_out]
    float *__restrict__ grad_input                     // [n_atoms, n_in]    accumulated +=
) {
    std::uint32_t const atom_index   = blockIdx.x * blockDim.y + static_cast<std::uint32_t>(threadIdx.y);
    std::uint32_t const channel_in   = blockIdx.y * blockDim.x + static_cast<std::uint32_t>(threadIdx.x);
    if (atom_index >= n_atoms || channel_in >= n_channels_in) return;

    int const element_index = atom_element[atom_index];
    float const *upstream_row = upstream_grad + static_cast<std::uint64_t>(atom_index) * n_channels_out;
    std::uint64_t const weight_row_offset =
        (static_cast<std::uint64_t>(channel_in) * n_elements + static_cast<std::uint64_t>(element_index))
            * n_channels_out;
    float accumulator = 0.0f;
    for (std::uint32_t channel_out_index = 0; channel_out_index < n_channels_out; ++channel_out_index) {
        accumulator += weights[weight_row_offset + channel_out_index] * upstream_row[channel_out_index];
    }
    std::uint64_t const out_offset = static_cast<std::uint64_t>(atom_index) * n_channels_in + channel_in;
    grad_input[out_offset] += scale * accumulator;
}

inline void launch_mace_element_conditioned_residual_force_adjoint(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_elements,
    std::uint32_t n_channels_out,
    std::uint32_t irrep_dim,
    std::uint32_t active_irrep_block_count,
    std::int32_t const *atom_element,
    float const *weights_fp32,
    float const *upstream_grad_output_fp32,
    float *grad_input_fp32,
    cudaStream_t stream
) {
    if (n_atoms == 0) return;
    float const scale = 1.0f / sqrtf(static_cast<float>(n_channels_in) * static_cast<float>(n_elements));

    if (irrep_dim == 1u && active_irrep_block_count == 1u) {
        constexpr std::uint32_t block_dim_x = 16, block_dim_y = 16;
        dim3 const block(block_dim_x, block_dim_y, 1);
        dim3 const grid((n_atoms + block_dim_y - 1u) / block_dim_y,
                        (n_channels_in + block_dim_x - 1u) / block_dim_x, 1);
        mace_skip_tp_irrep1_force_adjoint_kernel<<<grid, block, 0, stream>>>(
            n_atoms, n_channels_in, n_elements, n_channels_out, scale,
            atom_element, weights_fp32, upstream_grad_output_fp32, grad_input_fp32);
        return;
    }

    dim3 const block(irrep_dim, 1, 1);
    dim3 const grid(n_atoms, n_channels_in, 1);
    mace_element_conditioned_residual_force_adjoint_kernel<<<grid, block, 0, stream>>>(
        n_atoms, n_channels_in, n_elements, n_channels_out, irrep_dim, active_irrep_block_count, scale,
        atom_element, weights_fp32, upstream_grad_output_fp32, grad_input_fp32);
}

// =========================================================================
// D4.  Symmetric contraction adjoint — closed-form ν=1, 2, 3 derivatives.
// =========================================================================

/**
 *  @brief Adjoint of `mace_symmetric_contraction_kernel`.
 *
 *  Forward (B[a, c] is the scalar L=0 output):
 *      B[a, c] = Σ_{i, p}                w_1[Z, p, c]   * U_1[i, p]       * X[i]
 *              + Σ_{i, j, p}             w_2[Z, p, c]   * U_2[i, j, p]    * X[i] * X[j]
 *              + Σ_{i, j, k, p}          w_max[Z, p, c] * U_3[i, j, k, p] * X[i] * X[j] * X[k]
 *
 *  Each U_k is fully symmetric under permutation of its X-index slots, so for
 *  output ν the derivative ∂B/∂X[a, c, i] picks up a leading factor of ν:
 *      ∂B/∂X[a, c, i] += Σ_{j_2, ..., j_ν, p}  ν * U_ν[i, j_2, ..., j_ν, p]
 *                                                * w_ν[Z, p, c] * Π_l X[a, c, j_l]
 *
 *  Receiving upstream `grad_B[a, c]` (shape `[n_atoms, n_channels]`), the
 *  per-input gradient is `dX[a, c, i] += grad_B[a, c] * (∂B/∂X[a, c, i])`.
 *  We sum the three ν contributions into the same `dX` buffer; the same
 *  `(j_2, j_3)` outer loop structure as the forward is reused, with each X
 *  factor in turn replaced by the gradient slot.  Output is `+=` accumulator.
 *
 *  Threading: one thread per (atom, channel).  Each thread loops over its
 *  16-wide X row and writes a 16-wide gradient row.  For each forward
 *  iteration `(i, j_2, ..., j_k, p)` the term `U_k[...] * Π X[...] * w`
 *  contributes to k separate gradient slots — we emit those k additive
 *  writes inline rather than relying on U-permutation symmetry.  The
 *  MACE-OFF23 U_3 tensor is NOT fully symmetric in the (i, j, k) index
 *  triple (verified empirically), so the "pin first index, multiply by ν"
 *  shortcut would be incorrect for ν=3.  U_2 is symmetric in (i, j), but
 *  we use the same emit-both-slots form for uniformity.
 */
inline __global__ void mace_symmetric_contraction_force_adjoint_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t irrep_dim,            // 16
    std::uint32_t path_count_max,       // 23
    std::uint32_t path_count_order_2,   // 4
    std::uint32_t path_count_order_1,   // 1
    std::uint32_t n_elements,           // 10
    std::int32_t const *atom_element,
    float const *X,                     // [n_atoms, n_channels, irrep_dim]
    float const *u_matrix_1,
    float const *u_matrix_2,
    float const *u_matrix_3,
    float const *weights_max,
    float const *weights_2,
    float const *weights_1,
    float const *upstream_grad_output,  // [n_atoms, n_channels]
    float *grad_X                       // [n_atoms, n_channels, irrep_dim]   accumulated +=
) {
    std::uint32_t atom_index = blockIdx.x;
    std::uint32_t channel_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (atom_index >= n_atoms || channel_index >= n_channels) return;

    int const element_index = atom_element[atom_index];

    float const *atom_irrep_row = X + (static_cast<std::uint64_t>(atom_index) * n_channels + channel_index) * irrep_dim;
    float *grad_atom_irrep_row = grad_X + (static_cast<std::uint64_t>(atom_index) * n_channels + channel_index) * irrep_dim;
    float const upstream = upstream_grad_output[static_cast<std::uint64_t>(atom_index) * n_channels + channel_index];

    // Local accumulator for this (atom, channel)'s 16 gradient slots.
    float dx_local[16] = {0.0f};

    // ν = 1: forward term is `U_1[i, p] * w_1 * X[i]`, single X factor → simple write.
    //   ∂B/∂X[i] = Σ_p w_1[Z, p, c] * U_1[i, p]
    for (std::uint32_t path_index = 0; path_index < path_count_order_1; ++path_index) {
        float const weighted_path_coefficient = weights_1[(static_cast<std::uint64_t>(element_index) * path_count_order_1 + path_index) * n_channels + channel_index];
        for (std::uint32_t irrep_index_i = 0; irrep_index_i < irrep_dim; ++irrep_index_i) {
            float const u = u_matrix_1[static_cast<std::uint64_t>(irrep_index_i) * path_count_order_1 + path_index];
            dx_local[irrep_index_i] += weighted_path_coefficient * u;
        }
    }

    // ν = 2: forward term is `U_2[i, j, p] * w_2 * X[i] * X[j]`.
    //   Differentiate: contributes `u_w * X[j]` to dx[i], `u_w * X[i]` to dx[j].
    //   Walk all ordered (i, j) pairs; emit both writes inside the loop so we
    //   are correct even if U_2 is not symmetric.
    for (std::uint32_t path_index = 0; path_index < path_count_order_2; ++path_index) {
        float const weighted_path_coefficient = weights_2[(static_cast<std::uint64_t>(element_index) * path_count_order_2 + path_index) * n_channels + channel_index];
        for (std::uint32_t irrep_index_i = 0; irrep_index_i < irrep_dim; ++irrep_index_i) {
            float const x_i = atom_irrep_row[irrep_index_i];
            for (std::uint32_t irrep_index_j = 0; irrep_index_j < irrep_dim; ++irrep_index_j) {
                float const u = u_matrix_2[((static_cast<std::uint64_t>(irrep_index_i) * irrep_dim) + irrep_index_j) * path_count_order_2 + path_index];
                float const u_w = weighted_path_coefficient * u;
                float const x_j = atom_irrep_row[irrep_index_j];
                dx_local[irrep_index_i] += u_w * x_j;
                dx_local[irrep_index_j] += u_w * x_i;
            }
        }
    }

    // ν = 3: forward term is `U_3[i, j, k, p] * w_max * X[i] * X[j] * X[k]`.
    //   Differentiate and emit three writes per inner iteration:
    //     dx[i] += u_w * X[j] * X[k]
    //     dx[j] += u_w * X[i] * X[k]
    //     dx[k] += u_w * X[i] * X[j]
    //   The MACE-OFF23 U_3 is not fully symmetric across (i, j, k), so the
    //   pinned-first-index trick is invalid; emitting all three permutation-
    //   conjugate slots inside the loop makes us correct without assuming
    //   any U_3 symmetry.
    for (std::uint32_t path_index = 0; path_index < path_count_max; ++path_index) {
        float const weighted_path_coefficient = weights_max[(static_cast<std::uint64_t>(element_index) * path_count_max + path_index) * n_channels + channel_index];
        for (std::uint32_t irrep_index_i = 0; irrep_index_i < irrep_dim; ++irrep_index_i) {
            float const x_i = atom_irrep_row[irrep_index_i];
            for (std::uint32_t irrep_index_j = 0; irrep_index_j < irrep_dim; ++irrep_index_j) {
                float const x_j = atom_irrep_row[irrep_index_j];
                float const x_ij = x_i * x_j;
                for (std::uint32_t irrep_index_k = 0; irrep_index_k < irrep_dim; ++irrep_index_k) {
                    float const u = u_matrix_3[(((static_cast<std::uint64_t>(irrep_index_i) * irrep_dim) + irrep_index_j) * irrep_dim + irrep_index_k) * path_count_max + path_index];
                    float const u_w = weighted_path_coefficient * u;
                    float const x_k = atom_irrep_row[irrep_index_k];
                    dx_local[irrep_index_i] += u_w * x_j * x_k;
                    dx_local[irrep_index_j] += u_w * x_i * x_k;
                    dx_local[irrep_index_k] += u_w * x_ij;
                }
            }
        }
    }

    // Multiply by upstream and write back with `+=`.
    for (std::uint32_t irrep_index_i = 0; irrep_index_i < irrep_dim; ++irrep_index_i) {
        grad_atom_irrep_row[irrep_index_i] += upstream * dx_local[irrep_index_i];
    }
}

inline void launch_mace_symmetric_contraction_force_adjoint(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t irrep_dim,
    std::uint32_t n_elements,
    std::int32_t const *atom_element,
    float const *node_features_full_fp32,
    float const *u_matrix_1_fp32,
    float const *u_matrix_2_fp32,
    float const *u_matrix_3_fp32,
    float const *weights_max_fp32,
    float const *weights_2_fp32,
    float const *weights_1_fp32,
    float const *upstream_grad_output_fp32,
    float *grad_node_features_full_fp32,
    cudaStream_t stream
) {
    if (n_atoms == 0) return;
    constexpr std::uint32_t block_dim_x = 32;
    dim3 const block(block_dim_x, 1, 1);
    dim3 const grid(n_atoms, (n_channels + block_dim_x - 1u) / block_dim_x, 1);
    constexpr std::uint32_t path_count_max     = 23u;
    constexpr std::uint32_t path_count_order_2 = 4u;
    constexpr std::uint32_t path_count_order_1 = 1u;
    mace_symmetric_contraction_force_adjoint_kernel<<<grid, block, 0, stream>>>(
        n_atoms, n_channels, irrep_dim, path_count_max, path_count_order_2, path_count_order_1, n_elements,
        atom_element, node_features_full_fp32,
        u_matrix_1_fp32, u_matrix_2_fp32, u_matrix_3_fp32,
        weights_max_fp32, weights_2_fp32, weights_1_fp32,
        upstream_grad_output_fp32, grad_node_features_full_fp32);
}

// =========================================================================
// Driver-side helper adjoints (small structural ops with closed-form backwards).
// =========================================================================

/**
 *  @brief Fill a fp32 buffer with a constant value (graph-capture safe).
 *
 *  Used by `mace_inference_run_force_adjoint` to seed `grad_per_graph_energy`
 *  with 1.0 — host->device memcpy is unsafe inside captured graphs because
 *  the source buffer must survive until graph replay.
 */
inline __global__ void mace_fill_constant_kernel(float *dst, std::uint32_t n_elements, float value) {
    std::uint32_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index >= n_elements) return;
    dst[element_index] = value;
}

inline void launch_mace_fill_constant(float *dst, std::uint32_t n_elements, float value, cudaStream_t stream) {
    if (n_elements == 0) return;
    constexpr std::uint32_t block = 128;
    std::uint32_t const grid = (n_elements + block - 1u) / block;
    mace_fill_constant_kernel<<<grid, block, 0, stream>>>(dst, n_elements, value);
}


/**
 *  @brief Adjoint of `mace_per_graph_energy_reduce_kernel`.
 *
 *  Forward: `e_g = Σ_{a ∈ g} (atomic_energies[Z_a] + scale * node_energy[a] + shift)`.
 *  Reverse: `dnode_energy[a] = scale * grad_per_graph_energy[graph_of_atom[a]]`.
 *  The atomic-energy and shift terms have no dependence on positions.
 */
inline __global__ void mace_per_graph_energy_force_adjoint_kernel(
    std::uint32_t n_atoms,
    float scale,
    std::int32_t const *graph_of_atom,
    float const *grad_per_graph_energy,
    float *grad_node_energy
) {
    std::uint32_t atom_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_index >= n_atoms) return;
    int const graph_index = graph_of_atom[atom_index];
    grad_node_energy[atom_index] = scale * grad_per_graph_energy[graph_index];
}

inline void launch_mace_per_graph_energy_force_adjoint(
    std::uint32_t n_atoms,
    float scale,
    std::int32_t const *graph_of_atom,
    float const *grad_per_graph_energy,
    float *grad_node_energy,
    cudaStream_t stream
) {
    if (n_atoms == 0) return;
    constexpr std::uint32_t block = 128;
    std::uint32_t const grid = (n_atoms + block - 1u) / block;
    mace_per_graph_energy_force_adjoint_kernel<<<grid, block, 0, stream>>>(
        n_atoms, scale, graph_of_atom, grad_per_graph_energy, grad_node_energy);
}

/** Launcher for `mace_scatter_force_adjoint_kernel` declared above. */
inline void launch_mace_scatter_force_adjoint(
    std::uint32_t n_edges,
    std::uint32_t feature_dim,
    std::int32_t const *edge_target_atom,
    float const *grad_node_features_full,
    float *grad_edge_messages,
    cudaStream_t stream
) {
    if (n_edges == 0) return;
    constexpr std::uint32_t block_dim_x = 128;
    dim3 const block(block_dim_x, 1, 1);
    dim3 const grid(n_edges, (feature_dim + block_dim_x - 1u) / block_dim_x, 1);
    mace_scatter_force_adjoint_kernel<<<grid, block, 0, stream>>>(
        n_edges, feature_dim, edge_target_atom, grad_node_features_full, grad_edge_messages);
}

/**
 *  @brief Adjoint of in-place scalar scale (`x *= s`).
 *
 *  Reverse: `grad_x = grad_y * s`.  In-place; we just multiply the gradient
 *  buffer.  Used for the `/avg_num_neighbors` rescale in the forward.
 */
inline __global__ void mace_scale_force_adjoint_kernel(
    std::uint32_t n_elements,
    float scale,
    float *grad_x_inplace
) {
    std::uint32_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index >= n_elements) return;
    grad_x_inplace[element_index] *= scale;
}

inline void launch_mace_scale_force_adjoint(
    std::uint32_t n_elements,
    float scale,
    float *grad_x_inplace,
    cudaStream_t stream
) {
    if (n_elements == 0) return;
    constexpr std::uint32_t block = 256;
    std::uint32_t const grid = (n_elements + block - 1u) / block;
    mace_scale_force_adjoint_kernel<<<grid, block, 0, stream>>>(n_elements, scale, grad_x_inplace);
}

/**
 *  @brief Adjoint of `apply_cutoff` (per-edge multiply by env(r)).
 *
 *  Forward: `tp_post[e, k] = tp_pre[e, k] * env(r[e])`.
 *  Reverse:
 *    - `grad_tp_pre[e, k] = grad_tp_post[e, k] * env(r[e])`  (in-place rescale)
 *    - `grad_env[e] = Σ_k grad_tp_post[e, k] * tp_pre[e, k]` (saved for the
 *       env-derivative path back to dr; folded into the radial-basis +
 *       edge-geometry adjoint via a separate dr accumulator).
 *
 *  We split into two kernels: one rescales `grad_tp` in-place; the other
 *  reduces `grad_env` per edge.  The second kernel uses a 32-thread block-
 *  per-edge pattern with warp shuffle reduction, identical to the readout
 *  linear kernel.
 */
inline __global__ void mace_apply_cutoff_force_adjoint_grad_tp_kernel(
    std::uint32_t n_edges,
    std::uint32_t feature_dim,                // 4 * n_channels
    float const *cutoff_envelope,             // [n_edges]
    float *grad_tp_inplace                    // [n_edges, feature_dim]
) {
    std::uint32_t edge_index = blockIdx.x;
    std::uint32_t feature_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (edge_index >= n_edges || feature_index >= feature_dim) return;
    float const env = cutoff_envelope[edge_index];
    std::uint64_t const offset = static_cast<std::uint64_t>(edge_index) * feature_dim + feature_index;
    grad_tp_inplace[offset] = grad_tp_inplace[offset] * env;
}

inline __global__ void mace_apply_cutoff_force_adjoint_grad_env_kernel(
    std::uint32_t n_edges,
    std::uint32_t feature_dim,
    float const *grad_tp_post,                // [n_edges, feature_dim] — caller's grad_tp BEFORE the in-place rescale
    float const *tp_pre,                      // [n_edges, feature_dim] — saved pre-cutoff tp_weights
    float *grad_env                           // [n_edges] — written
) {
    std::uint32_t edge_index = blockIdx.x;
    if (edge_index >= n_edges) return;
    std::uint64_t const row_offset = static_cast<std::uint64_t>(edge_index) * feature_dim;
    float accumulator = 0.0f;
    for (std::uint32_t feature_index = threadIdx.x; feature_index < feature_dim; feature_index += blockDim.x) {
        accumulator += grad_tp_post[row_offset + feature_index] * tp_pre[row_offset + feature_index];
    }
    for (int shfl_offset = 16; shfl_offset > 0; shfl_offset /= 2) {
        accumulator += __shfl_xor_sync(0xFFFFFFFF, accumulator, shfl_offset);
    }
    if (threadIdx.x == 0) grad_env[edge_index] = accumulator;
}

inline void launch_mace_apply_cutoff_force_adjoint(
    std::uint32_t n_edges,
    std::uint32_t feature_dim,
    float const *cutoff_envelope,
    float const *tp_pre,
    float *grad_tp_inplace,
    float *grad_env,
    cudaStream_t stream
) {
    if (n_edges == 0) return;
    // First reduce grad_env using grad_tp_post BEFORE we rescale grad_tp.
    if (grad_env != nullptr) {
        constexpr std::uint32_t block_env = 32;
        mace_apply_cutoff_force_adjoint_grad_env_kernel<<<n_edges, block_env, 0, stream>>>(
            n_edges, feature_dim, grad_tp_inplace, tp_pre, grad_env);
    }
    constexpr std::uint32_t block_dim_x = 128;
    dim3 const block(block_dim_x, 1, 1);
    dim3 const grid(n_edges, (feature_dim + block_dim_x - 1u) / block_dim_x, 1);
    mace_apply_cutoff_force_adjoint_grad_tp_kernel<<<grid, block, 0, stream>>>(
        n_edges, feature_dim, cutoff_envelope, grad_tp_inplace);
}

/**
 *  @brief env(r) derivative folded into dpositions.
 *
 *  `grad_env[e]` is the cutoff envelope's accumulated gradient from the
 *  apply_cutoff adjoint.  d env / d r is already computed inside the radial
 *  geometry adjoint as `denv_dr`; here we replicate that math standalone so
 *  the apply_cutoff path can route into dpositions independently.
 *
 *  d r / d positions[target/source] = ± r_unit.  Atomic-add into dpositions.
 */
inline __global__ void mace_cutoff_grad_env_to_positions_kernel(
    std::uint32_t n_edges,
    float r_max,
    float cutoff_p,
    float const *r_norm,
    float const *r_unit,
    std::int32_t const *edge_source_atom,
    std::int32_t const *edge_target_atom,
    float const *grad_env,
    float *dpositions
) {
    std::uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_index >= n_edges) return;
    float const r = r_norm[edge_index];
    if (r >= r_max) return;
    float const u = r / r_max;
    float const p = cutoff_p;
    float const a = -(p + 1.0f) * (p + 2.0f) * 0.5f;
    float const b = p * (p + 2.0f);
    float const c = -p * (p + 1.0f) * 0.5f;
    float const up_minus1 = powf(u, p - 1.0f);
    float const up        = up_minus1 * u;
    float const up1       = up * u;
    float const inv_r_max = 1.0f / r_max;
    float const denv_dr = inv_r_max * (a * p * up_minus1 + b * (p + 1.0f) * up + c * (p + 2.0f) * up1);
    float const ge = grad_env[edge_index];
    float const dr_contribution = ge * denv_dr;
    float const ux = r_unit[3 * edge_index + 0];
    float const uy = r_unit[3 * edge_index + 1];
    float const uz = r_unit[3 * edge_index + 2];
    int const src = edge_source_atom[edge_index];
    int const dst = edge_target_atom[edge_index];
    atomicAdd(dpositions + 3 * dst + 0,  dr_contribution * ux);
    atomicAdd(dpositions + 3 * dst + 1,  dr_contribution * uy);
    atomicAdd(dpositions + 3 * dst + 2,  dr_contribution * uz);
    atomicAdd(dpositions + 3 * src + 0, -dr_contribution * ux);
    atomicAdd(dpositions + 3 * src + 1, -dr_contribution * uy);
    atomicAdd(dpositions + 3 * src + 2, -dr_contribution * uz);
}

inline void launch_mace_cutoff_grad_env_to_positions(
    types::mace_arch_t const &arch,
    std::uint32_t n_edges,
    float const *r_norm,
    float const *r_unit,
    std::int32_t const *edge_source_atom,
    std::int32_t const *edge_target_atom,
    float const *grad_env,
    float *dpositions,
    cudaStream_t stream
) {
    if (n_edges == 0) return;
    constexpr std::uint32_t block = 128;
    std::uint32_t const grid = (n_edges + block - 1u) / block;
    mace_cutoff_grad_env_to_positions_kernel<<<grid, block, 0, stream>>>(
        n_edges, arch.r_max, arch.cutoff_p, r_norm, r_unit,
        edge_source_atom, edge_target_atom, grad_env, dpositions);
}

/**
 *  @brief Adjoint of `mace_extract_l0_irrep_block_kernel` and
 *         `mace_inject_l0_irrep_block_kernel`.
 *
 *  These two ops are duals of each other.  `extract_l0` reads the L=0
 *  component of `[n_atoms, c, 16]` into `[n_atoms, c]`.  `inject_l0` does
 *  the reverse but ZEROS the upper L slots.
 *
 *  Adjoint of `extract_l0` (forward L=0 → scalar): the upstream is shape
 *  `[n_atoms, c]`; we add it to the L=0 slot of an irrep tensor (no zeroing
 *  the rest, since they may already hold gradient from another path).
 *
 *  Adjoint of `inject_l0` (forward scalar → L=0; zeros upper Ls): the
 *  upstream is shape `[n_atoms, c, 16]`; only the L=0 slot has non-zero
 *  forward sensitivity, so we extract that into a `[n_atoms, c]` accumulator.
 */
inline __global__ void mace_extract_l0_irrep_block_force_adjoint_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t irrep_dim,
    float const *grad_l0_only,                // [n_atoms, n_channels]
    float *grad_node_features_full            // [n_atoms, n_channels, irrep_dim]   accumulated +=
) {
    std::uint32_t atom_index = blockIdx.x;
    std::uint32_t channel_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (atom_index >= n_atoms || channel_index >= n_channels) return;
    grad_node_features_full[(static_cast<std::uint64_t>(atom_index) * n_channels + channel_index) * irrep_dim + 0]
        += grad_l0_only[static_cast<std::uint64_t>(atom_index) * n_channels + channel_index];
}

inline void launch_mace_extract_l0_irrep_block_force_adjoint(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t irrep_dim,
    float const *grad_l0_only,
    float *grad_node_features_full,
    cudaStream_t stream
) {
    if (n_atoms == 0) return;
    constexpr std::uint32_t block_dim_x = 64;
    dim3 const block(block_dim_x, 1, 1);
    dim3 const grid(n_atoms, (n_channels + block_dim_x - 1u) / block_dim_x, 1);
    mace_extract_l0_irrep_block_force_adjoint_kernel<<<grid, block, 0, stream>>>(
        n_atoms, n_channels, irrep_dim, grad_l0_only, grad_node_features_full);
}

inline __global__ void mace_inject_l0_irrep_block_force_adjoint_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t irrep_dim,
    float const *grad_node_features_full,     // [n_atoms, n_channels, irrep_dim]
    float *grad_l0_only                       // [n_atoms, n_channels]   accumulated +=
) {
    std::uint32_t atom_index = blockIdx.x;
    std::uint32_t channel_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (atom_index >= n_atoms || channel_index >= n_channels) return;
    grad_l0_only[static_cast<std::uint64_t>(atom_index) * n_channels + channel_index]
        += grad_node_features_full[(static_cast<std::uint64_t>(atom_index) * n_channels + channel_index) * irrep_dim + 0];
}

inline void launch_mace_inject_l0_irrep_block_force_adjoint(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t irrep_dim,
    float const *grad_node_features_full,
    float *grad_l0_only,
    cudaStream_t stream
) {
    if (n_atoms == 0) return;
    constexpr std::uint32_t block_dim_x = 64;
    dim3 const block(block_dim_x, 1, 1);
    dim3 const grid(n_atoms, (n_channels + block_dim_x - 1u) / block_dim_x, 1);
    mace_inject_l0_irrep_block_force_adjoint_kernel<<<grid, block, 0, stream>>>(
        n_atoms, n_channels, irrep_dim, grad_node_features_full, grad_l0_only);
}

/**
 *  @brief Adjoint of the radial MLP — 4 fp32 fully-connected layers with
 *         normalize2mom(silu) between layers 0..2.
 *
 *  Forward (per edge):
 *    h0 = silu·n2m(scale0 * radial_basis @ W0)              ; W0:[n_radial, h]
 *    h1 = silu·n2m(scale1 * h0 @ W1)                        ; W1:[h, h]
 *    h2 = silu·n2m(scale2 * h1 @ W2)                        ; W2:[h, h]
 *    out = scale3 * h2 @ W3                                 ; W3:[h, 4*c]
 *  where each scale_i = 1/sqrt(in_dim_i).
 *
 *  We need the saved pre-activations h0_pre, h1_pre, h2_pre (= scale * (in @ W))
 *  to recompute d silu·n2m / dx.  The forward currently saves only the
 *  POST-activation buffers `S.radial_mlp_hidden_{0,1,2}` (the silu output
 *  is written in-place into the same buffer that held the linear output).
 *  For the backward we re-derive pre_act from post_act:  silu(x) = x * sigmoid(x),
 *  invert numerically?  Cheaper: re-run the four forward matmuls and stage
 *  pre_act buffers in the kernel via a fp32 scratch; or save pre_act in the
 *  forward (small, ~edges * h floats * 3).
 *
 *  We choose the scratch-recompute path: the backward kernel re-runs the
 *  three pre-act linears on the fly using the saved post-activations from
 *  the previous step.  Simpler than allocating new save buffers, and the
 *  cost is dominated by the matmul backward anyway.
 *
 *  Since saving post-activations means the silu input is lost, we instead
 *  recover pre_act by undoing silu·n2m.  silu·n2m(x) = N * x * sigmoid(x);
 *  this is monotonic and well-conditioned outside the saturation regime, so
 *  Newton iteration converges in ~3 steps.  Implemented inline.
 *
 *  --- Simpler approach taken below: the forward saves the post-silu·n2m
 *      activations, and the backward iteratively inverts silu·n2m to
 *      recover pre_act.  6 Newton iterations gives ~1e-7 precision.  Then
 *      d silu·n2m / d pre_act is computed analytically.
 */
__device__ __forceinline__ float mace_invert_silu_norm2mom_(float y) {
    // Forward: y = N * x * sigmoid(x)  with N = mace_silu_normalize_to_unit_second_moment_constant_k.
    // f(x) = N * x * sigmoid(x) - y, f'(x) = N * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))).
    // Newton's method, starting from x0 = y / N (good for small y, since silu(x) ~ x/2 + x^2/4 - ...
    // and N ~ 1.679 — y/N is a slight overestimate around zero but stable).
    constexpr float N = mace_silu_normalize_to_unit_second_moment_constant_k;
    float x = y / N;
    // For very negative y, silu·n2m is bounded below ≈ -0.557 * N ~ -0.935; clamp the
    // initial guess to avoid divergent iterates.
    if (x < -3.0f) x = -3.0f;
    if (x >  6.0f) x =  6.0f;
    for (int iter = 0; iter < 8; ++iter) {
        float const sig = 1.0f / (1.0f + __expf(-x));
        float const f   = N * x * sig - y;
        float const fp  = N * (sig + x * sig * (1.0f - sig));
        // Guard against fp ~ 0 (shouldn't happen in our regime).
        float const step = (fabsf(fp) > 1e-8f) ? (f / fp) : 0.0f;
        x -= step;
    }
    return x;
}

/** d silu·n2m / d pre_act, given pre_act. */
__device__ __forceinline__ float mace_silu_norm2mom_derivative_(float pre_act) {
    constexpr float N = mace_silu_normalize_to_unit_second_moment_constant_k;
    float const sig = 1.0f / (1.0f + __expf(-pre_act));
    return N * sig * (1.0f + pre_act * (1.0f - sig));
}

/**
 *  Backward of one fully-connected layer with optional silu·n2m epilogue.
 *
 *    forward:  out = silu_n2m(scale * (in @ W))   (or just scale * (in @ W) for the final layer)
 *    backward: d_in[a, c_in]  = scale * Σ_{c_out} d_pre[a, c_out] * W[c_in, c_out]
 *  where d_pre = d_out * silu_n2m'(pre_act) (or just d_out if no epilogue).
 *
 *  Inputs:
 *    `post_activation`: forward output (post-silu_n2m if applicable) — shape [N, c_out]
 *    `weight`:          [c_in, c_out]
 *    `grad_out`:        [N, c_out]
 *    `scale`:           1/sqrt(c_in)
 *    `apply_silu_n2m_inverse`: true for layers 0..2, false for layer 3 (final, no activation)
 *  Output:
 *    `grad_in`:         [N, c_in]   (overwritten)
 *
 *  Threading: one thread per (row, c_in).  Inner loop over c_out.
 */
inline __global__ void mace_radial_mlp_layer_force_adjoint_kernel(
    std::uint32_t n_rows,
    std::uint32_t n_in,
    std::uint32_t n_out,
    float scale,
    bool apply_silu_n2m_inverse,
    bool accumulate,                 // true: grad_in += result; false: grad_in = result
    bool activation_is_pre_act,      // true: `activation` holds pre-activation directly (skip Newton inversion)
    float const *activation,         // [n_rows, n_out]   (pre- OR post-silu_n2m, see flag)
    float const *weight,             // [n_in, n_out]
    float const *grad_out,           // [n_rows, n_out]
    float *grad_in                   // [n_rows, n_in]
) {
    std::uint32_t row_index = blockIdx.x;
    std::uint32_t input_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (row_index >= n_rows || input_index >= n_in) return;
    float accumulator = 0.0f;
    for (std::uint32_t out_index = 0; out_index < n_out; ++out_index) {
        float du = grad_out[static_cast<std::uint64_t>(row_index) * n_out + out_index];
        if (apply_silu_n2m_inverse) {
            float const a = activation[static_cast<std::uint64_t>(row_index) * n_out + out_index];
            // silu_n2m is non-monotonic (has a min near x = -1.278 with min value ≈ -0.467 * N).
            // Inverting from the post-activation by Newton picks the WRONG branch when the
            // forward pre_act sat below the critical point — sign of the derivative flips,
            // ruining the chain rule.  Prefer the pre-activation direct path when the caller
            // can supply it (re-run forward + skip the in-place silu).
            float const pre = activation_is_pre_act ? a : mace_invert_silu_norm2mom_(a);
            float const dact = mace_silu_norm2mom_derivative_(pre);
            du *= dact;
        }
        accumulator += du * weight[static_cast<std::uint64_t>(input_index) * n_out + out_index];
    }
    std::uint64_t const out_offset = static_cast<std::uint64_t>(row_index) * n_in + input_index;
    float const result = scale * accumulator;
    grad_in[out_offset] = accumulate ? (grad_in[out_offset] + result) : result;
}

inline void launch_mace_radial_mlp_layer_force_adjoint(
    std::uint32_t n_rows,
    std::uint32_t n_in,
    std::uint32_t n_out,
    float scale,
    bool apply_silu_n2m_inverse,
    bool accumulate,
    float const *activation,
    float const *weight,
    float const *grad_out,
    float *grad_in,
    cudaStream_t stream,
    bool activation_is_pre_act = false
) {
    if (n_rows == 0) return;
    constexpr std::uint32_t block_dim_x = 64;
    dim3 const block(block_dim_x, 1, 1);
    dim3 const grid(n_rows, (n_in + block_dim_x - 1u) / block_dim_x, 1);
    mace_radial_mlp_layer_force_adjoint_kernel<<<grid, block, 0, stream>>>(
        n_rows, n_in, n_out, scale, apply_silu_n2m_inverse, accumulate, activation_is_pre_act,
        activation, weight, grad_out, grad_in);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
