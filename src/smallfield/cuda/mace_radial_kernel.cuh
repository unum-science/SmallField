/**
 *  @file mace_radial_kernel.cuh
 *  @brief Per-edge edge geometry, Bessel basis × polynomial cutoff, and SiLU.
 *
 *  Mirrors `mace.modules.radial.BesselBasis` and `PolynomialCutoff`:
 *
 *    bessel_weights[n] = π * (n + 1) / r_max,   n = 0..n_radial-1   (stored in the weight blob)
 *    prefactor         = sqrt(2 / r_max)                            (stored as `bessel_prefactor`)
 *    b_n(r)            = prefactor * sin(bessel_weights[n] * r) / r
 *    e(r)              = 1 - (p+1)(p+2)/2 * (r/r_max)^p
 *                            + p(p+2)        * (r/r_max)^(p+1)
 *                            - p(p+1)/2      * (r/r_max)^(p+2)         (clamped to 0 for r ≥ r_max)
 *    radial_basis[n]   = b_n(r) * e(r)
 *
 *  Verified bit-for-bit against the golden trace produced by
 *  `python/smallfield/mace_export.py` (golden is the output of
 *  `RadialEmbeddingBlock.forward(...)`, which equals `radial * cutoff`).
 *
 *  Also defines the SiLU activation kernel used by the radial MLP and the
 *  nonlinear readout (`f(x) = x * sigmoid(x)`).
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

#include "smallfield/types/mace_types.hpp"

namespace smallfield::cuda {

using smallfield::types::mace_arch_t;

/**
 *  @brief Compute per-edge `r_norm` (distance) and `r_unit` (direction).
 *
 *  Block: 1 warp per edge.  Trivially memory-bound.  Kept in its own kernel
 *  because the LBFGS step writes new positions and we want to recompute only
 *  this; weights and edge index do not change.
 */
inline __global__ void mace_edge_geometry_kernel(
    std::uint32_t n_edges,
    float const *positions,            // [n_atoms, 3] (x, y, z) per atom
    std::int32_t const *edge_source_atom,
    std::int32_t const *edge_target_atom,
    float *r_norm,                     // [n_edges]
    float *r_unit                      // [n_edges, 3]
) {
    std::uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_index >= n_edges) return;
    int const src = edge_source_atom[edge_index];
    int const dst = edge_target_atom[edge_index];
    float const sx = positions[3 * src + 0];
    float const sy = positions[3 * src + 1];
    float const sz = positions[3 * src + 2];
    float const dx = positions[3 * dst + 0] - sx;
    float const dy = positions[3 * dst + 1] - sy;
    float const dz = positions[3 * dst + 2] - sz;
    float const r2 = dx * dx + dy * dy + dz * dz;
    float const r  = sqrtf(fmaxf(r2, 1e-20f));
    float const inv = 1.0f / r;
    r_norm[edge_index] = r;
    r_unit[3 * edge_index + 0] = dx * inv;
    r_unit[3 * edge_index + 1] = dy * inv;
    r_unit[3 * edge_index + 2] = dz * inv;
}

/**
 *  @brief Bessel basis × polynomial cutoff in one kernel.
 *
 *  All math is fp32 (well within precision for an envelope function).
 *  `bessel_weights[n]` already encodes `π * (n+1) / r_max` so we multiply
 *  raw `r` (not the normalised distance) by the weight.
 */
inline __global__ void mace_radial_basis_kernel(
    std::uint32_t n_edges,
    std::uint32_t n_radial,
    float r_max,
    float bessel_prefactor,
    float cutoff_p,
    float const *bessel_weights,       // [n_radial], = π * (n+1) / r_max
    float const *r_norm,               // [n_edges]
    float *radial_basis,               // [n_edges, n_radial]
    float *cutoff_envelope             // [n_edges]
) {
    std::uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_index >= n_edges) return;

    float const r = r_norm[edge_index];

    // Polynomial cutoff envelope, mirrors PolynomialCutoff.calculate_envelope.
    float env = 0.0f;
    if (r < r_max) {
        float const u   = r / r_max;
        float const p   = cutoff_p;
        float const a   = -(p + 1.0f) * (p + 2.0f) * 0.5f;
        float const b   =  p * (p + 2.0f);
        float const c   = -p * (p + 1.0f) * 0.5f;
        float const up  = powf(u, p);
        float const up1 = up * u;
        float const up2 = up1 * u;
        env = 1.0f + a * up + b * up1 + c * up2;
    }
    cutoff_envelope[edge_index] = env;

    float const inv_r = 1.0f / r;
    for (std::uint32_t n = 0; n < n_radial; ++n) {
        float const w   = bessel_weights[n];           // π (n+1) / r_max
        float const arg = w * r;                       // raw r, not normalised
        float const b_n = bessel_prefactor * sinf(arg) * inv_r;
        radial_basis[edge_index * n_radial + n] = b_n * env;
    }
}

/** SiLU activation, x * sigmoid(x). One thread per element. */
inline __global__ void mace_silu_activation_kernel(std::uint32_t n, float const *src, float *dst) {
    std::uint32_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index >= n) return;
    float const x = src[element_index];
    float const s = 1.0f / (1.0f + __expf(-x));
    dst[element_index] = x * s;
}

inline void launch_mace_edge_geometry(
    std::uint32_t n_edges,
    float const *positions,
    std::int32_t const *edge_source_atom,
    std::int32_t const *edge_target_atom,
    float *r_norm,
    float *r_unit,
    cudaStream_t stream
) {
    if (n_edges == 0) return;
    constexpr std::uint32_t block = 128;
    std::uint32_t const grid = (n_edges + block - 1u) / block;
    mace_edge_geometry_kernel<<<grid, block, 0, stream>>>(n_edges, positions, edge_source_atom, edge_target_atom, r_norm, r_unit);
}

inline void launch_mace_radial_basis(
    mace_arch_t const &arch,
    std::uint32_t n_edges,
    float const *bessel_weights,
    float const *r_norm,
    float *radial_basis,
    float *cutoff_envelope,
    cudaStream_t stream
) {
    if (n_edges == 0) return;
    constexpr std::uint32_t block = 128;
    std::uint32_t const grid = (n_edges + block - 1u) / block;
    mace_radial_basis_kernel<<<grid, block, 0, stream>>>(
        n_edges, arch.n_radial, arch.r_max, arch.bessel_prefactor, arch.cutoff_p,
        bessel_weights, r_norm, radial_basis, cutoff_envelope);
}

inline void launch_mace_silu(std::uint32_t n, float const *src, float *dst, cudaStream_t stream) {
    if (n == 0) return;
    constexpr std::uint32_t block = 256;
    std::uint32_t const grid = (n + block - 1u) / block;
    mace_silu_activation_kernel<<<grid, block, 0, stream>>>(n, src, dst);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
