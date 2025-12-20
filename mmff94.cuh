/**
 * MMFF94 Energy Kernels for NVIDIA H100 GPU
 *
 * Architecture:
 * - One warp (32 threads) per molecule
 * - FP16 and FP32 compute variants
 * - Register-only design with warp shuffles
 * - No shared memory required
 *
 * Supported molecule sizes:
 * - Configurable via pairs per molecule
 * - Strided iteration: thread k processes pairs k, k+32, k+64...
 *
 * Target: SM 9.0 (H100)
 * Compile: nvcc -gencode=arch=compute_90,code=sm_90
 */

#pragma once

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace smallfield {

#pragma region Constants

constexpr std::uint32_t warp_size = 32;
constexpr unsigned int full_mask = 0xFFFFFFFF;

constexpr float const coulomb_const_f32 = 332.0716f;
constexpr float const delta_f32 = 0.05f;
constexpr float const scale_1_4_f32 = 0.75f;
constexpr float const scale_1_4_diff_f32 = scale_1_4_f32 - 1.0f; // For branchless scaling

#pragma endregion

#pragma region Atom-Centric Data Structures

/**
 * TODO: Consider a dense per-atom coordinate layout for molecules under 256 atoms.
 * For small molecules, storing coordinates inline with AoS layout and using
 * register-based accumulation could further reduce memory traffic.
 * Current implementation uses shuffle-based coordinate access which works well
 * for molecules up to 32 atoms per warp.
 */

/**
 * Neighbor entry in symmetric adjacency list.
 *
 * Each pair (A, B) is stored twice:
 * - In A's neighbor list with j=B and sign=-1 (force on A)
 * - In B's neighbor list with j=A and sign=+1 (force on B)
 *
 * Memory layout (16 bytes aligned for coalesced loads):
 * ┌───────┬───────┬────────┬─────────┬────────┬───────────┐
 * │ j (1) │sign(1)│flags(2)│ eps (4) │rstar(4)│q_prod (4) │
 * └───────┴───────┴────────┴─────────┴────────┴───────────┘
 */
struct __align__(16) neighbor_t {
    std::uint8_t j;      // Neighbor atom index within molecule (0-255)
    std::int8_t sign;    // Force direction: -1 or +1
    std::uint16_t flags; // Bit 0 = is_1_4
    float epsilon;       // VdW epsilon
    float r_star;        // VdW equilibrium distance
    float q_product;     // qi * qj (precomputed for electrostatics)
};

static_assert(sizeof(neighbor_t) == 16, "neighbor_t must be 16 bytes");

/**
 * Per-molecule metadata for atom-centric kernel.
 *
 * Layout:
 * ┌─────────────────────────────────────────────────────────────┐
 * │  atom_start: First atom index in global coordinate array   │
 * │  n_atoms:    Number of atoms in this molecule              │
 * │  neighbor_start: First neighbor in global neighbor array   │
 * │  n_neighbors: Total neighbors (2x pairs due to symmetry)   │
 * └─────────────────────────────────────────────────────────────┘
 */
struct mol_meta_t {
    std::uint32_t atom_start;     // First atom index in global array
    std::uint32_t n_atoms;        // Number of atoms in molecule
    std::uint32_t neighbor_start; // First neighbor in neighbor array
    std::uint32_t n_neighbors;    // Total neighbors (2x pairs)
};

#pragma endregion

#pragma region FP16 Helper Functions

/**
 * Fast reciprocal square root for FP16.
 */
__device__ __forceinline__ half h_rsqrt_(half x) { return hrsqrt(x); }

/**
 * Fast reciprocal for FP16.
 */
__device__ __forceinline__ half h_rcp_(half x) { return hrcp(x); }

/**
 * FP16 square root.
 */
__device__ __forceinline__ half h_sqrt_(half x) { return hsqrt(x); }

/**
 * Fused multiply-add: a*b + c.
 */
__device__ __forceinline__ half h_fma_(half a, half b, half c) { return __hfma(a, b, c); }

#pragma endregion

#pragma region FP16 half2 Packed Helper Functions

/**
 * Fast reciprocal square root for half2 (packed FP16).
 */
__device__ __forceinline__ half2 h2_rsqrt_(half2 x) { return h2rsqrt(x); }

/**
 * Fast reciprocal for half2 (packed FP16).
 */
__device__ __forceinline__ half2 h2_rcp_(half2 x) { return h2rcp(x); }

/**
 * FP16 square root for half2 (packed FP16).
 */
__device__ __forceinline__ half2 h2_sqrt_(half2 x) { return h2sqrt(x); }

/**
 * Fused multiply-add for half2: a*b + c.
 */
__device__ __forceinline__ half2 h2_fma_(half2 a, half2 b, half2 c) { return __hfma2(a, b, c); }

#pragma endregion

#pragma region Warp Reduction

/**
 * Warp reduction using butterfly XOR shuffles.
 * Uses 5 shuffle operations to reduce 32 lanes.
 * Result is available in all lanes.
 *
 * Note: __reduce_add_sync only supports integer types on sm_80+.
 * For float reduction, we use the shuffle-based approach.
 */
__device__ __forceinline__ float warp_reduce_sum_f32_(float val) {
    val += __shfl_xor_sync(full_mask, val, 16);
    val += __shfl_xor_sync(full_mask, val, 8);
    val += __shfl_xor_sync(full_mask, val, 4);
    val += __shfl_xor_sync(full_mask, val, 2);
    val += __shfl_xor_sync(full_mask, val, 1);
    return val;
}

#pragma endregion

#pragma region Distance Computation

/**
 * Euclidean distance in FP16.
 * Uses FMA for better precision: dx² + dy² + dz².
 */
__device__ __forceinline__ half compute_distance_f16_(half const xi, half const yi, half const zi, half const xj,
                                                      half const yj, half const zj) {
    half const dx = __hsub(xi, xj);
    half const dy = __hsub(yi, yj);
    half const dz = __hsub(zi, zj);

    half r2 = __hmul(dx, dx);
    r2 = h_fma_(dy, dy, r2);
    r2 = h_fma_(dz, dz, r2);

    return h_sqrt_(r2);
}

/**
 * Euclidean distance in FP32.
 */
__device__ __forceinline__ float compute_distance_f32_(float const xi, float const yi, float const zi, float const xj,
                                                       float const yj, float const zj) {
    float const dx = xi - xj;
    float const dy = yi - yj;
    float const dz = zi - zj;

    float r2 = fmaf(dx, dx, 0.0f);
    r2 = fmaf(dy, dy, r2);
    r2 = fmaf(dz, dz, r2);

    return sqrtf(r2);
}

/**
 * Paired Euclidean distance computation using half2.
 * Computes distances for two pairs simultaneously.
 *
 * Input: Two sets of coordinates packed as half2.
 * Output: Two distances packed as half2.
 */
__device__ __forceinline__ half2 compute_distance_f16_pair_(half2 const xi, half2 const yi, half2 const zi,
                                                            half2 const xj, half2 const yj, half2 const zj) {
    half2 const dx = __hsub2(xi, xj);
    half2 const dy = __hsub2(yi, yj);
    half2 const dz = __hsub2(zi, zj);

    half2 r2 = __hmul2(dx, dx);
    r2 = h2_fma_(dy, dy, r2);
    r2 = h2_fma_(dz, dz, r2);

    return h2_sqrt_(r2);
}

#pragma endregion

#pragma region VdW Energy Computation

/**
 * MMFF94 Buffered 14-7 van der Waals energy (FP16 compute, FP32 accumulation).
 *
 * E = ε * (1.07*Rstar/(r+0.07*Rstar))^7 * ((1.12*Rstar^7/(r^7+0.12*Rstar^7)) - 2)
 *
 * Computation in FP16, final multiply in FP32 for accumulation precision.
 */
__device__ __forceinline__ float compute_vdw_energy_f16_(half const r, half const epsilon, half const r_star) {
    half const r_safe = __hgt(r, __float2half(0.001f)) ? r : __float2half(0.001f);

    half const inv_rstar = h_rcp_(r_star);
    half const rho = __hmul(r_safe, inv_rstar);

    half const rho2 = __hmul(rho, rho);
    half const rho4 = __hmul(rho2, rho2);
    half const rho6 = __hmul(rho4, rho2);
    half const rho7 = __hmul(rho6, rho);

    half const c_107 = __float2half(1.07f);
    half const c_007 = __float2half(0.07f);
    half const t1 = __hmul(c_107, h_rcp_(__hadd(rho, c_007)));
    half const t1_2 = __hmul(t1, t1);
    half const t1_4 = __hmul(t1_2, t1_2);
    half const t1_6 = __hmul(t1_4, t1_2);
    half const t1_7 = __hmul(t1_6, t1);

    half const c_112 = __float2half(1.12f);
    half const c_012 = __float2half(0.12f);
    half const c_2 = __float2half(2.0f);
    half const t2 = __hsub(__hmul(c_112, h_rcp_(__hadd(rho7, c_012))), c_2);

    return __half2float(epsilon) * __half2float(t1_7) * __half2float(t2);
}

/**
 * MMFF94 Buffered 14-7 van der Waals energy (FP32 compute and accumulation).
 */
__device__ __forceinline__ float compute_vdw_energy_f32_(float const r, float const epsilon, float const r_star) {
    float const r_safe = (r > 0.001f) ? r : 0.001f;

    float const inv_rstar = __frcp_rn(r_star);
    float const rho = r_safe * inv_rstar;

    float const rho2 = rho * rho;
    float const rho4 = rho2 * rho2;
    float const rho6 = rho4 * rho2;
    float const rho7 = rho6 * rho;

    float const c_107 = 1.07f;
    float const c_007 = 0.07f;
    float const t1 = c_107 * __frcp_rn(rho + c_007);
    float const t1_2 = t1 * t1;
    float const t1_4 = t1_2 * t1_2;
    float const t1_6 = t1_4 * t1_2;
    float const t1_7 = t1_6 * t1;

    float const c_112 = 1.12f;
    float const c_012 = 0.12f;
    float const c_2 = 2.0f;
    float const t2 = c_112 * __frcp_rn(rho7 + c_012) - c_2;

    return epsilon * t1_7 * t2;
}

/**
 * MMFF94 Buffered 14-7 van der Waals energy using half2 packed operations.
 * Processes 2 pairs simultaneously for 2x throughput.
 *
 * Returns: Two energies packed as float (low + high).
 *
 * Note: Computation in FP16 with packed half2, final conversion to FP32 for accumulation.
 */
__device__ __forceinline__ float compute_vdw_energy_f16_pair_(half2 const r, half2 const epsilon, half2 const r_star) {
    // Safety clamp: r >= 0.001
    half2 const safety = __float2half2_rn(0.001f);
    half2 const r_safe = __hmul2(__hgt2(r, safety), r) + __hmul2(__hle2(r, safety), safety);

    half2 const inv_rstar = h2_rcp_(r_star);
    half2 const rho = __hmul2(r_safe, inv_rstar);

    half2 const rho2 = __hmul2(rho, rho);
    half2 const rho4 = __hmul2(rho2, rho2);
    half2 const rho6 = __hmul2(rho4, rho2);
    half2 const rho7 = __hmul2(rho6, rho);

    half2 const c_107 = __float2half2_rn(1.07f);
    half2 const c_007 = __float2half2_rn(0.07f);
    half2 const t1 = __hmul2(c_107, h2_rcp_(__hadd2(rho, c_007)));
    half2 const t1_2 = __hmul2(t1, t1);
    half2 const t1_4 = __hmul2(t1_2, t1_2);
    half2 const t1_6 = __hmul2(t1_4, t1_2);
    half2 const t1_7 = __hmul2(t1_6, t1);

    half2 const c_112 = __float2half2_rn(1.12f);
    half2 const c_012 = __float2half2_rn(0.12f);
    half2 const c_2 = __float2half2_rn(2.0f);
    half2 const t2 = __hsub2(__hmul2(c_112, h2_rcp_(__hadd2(rho7, c_012))), c_2);

    // Final energy: epsilon * t1_7 * t2 (per pair)
    half2 const e_pair = __hmul2(__hmul2(epsilon, t1_7), t2);

    // Convert both halves to FP32 and return sum
    float const e_low = __half2float(__low2half(e_pair));
    float const e_high = __half2float(__high2half(e_pair));

    return e_low + e_high;
}

#pragma endregion

#pragma region Electrostatic Energy Computation

/**
 * MMFF94 Buffered Coulomb electrostatic energy (FP16 compute, FP32 accumulation).
 *
 * E = 332.0716 * qi * qj / (r + 0.05)
 *
 * For 1-4 interactions: scale by 0.75.
 * Uses branchless scaling: scale = 1.0 + is_1_4 * (0.75 - 1.0)
 */
__device__ __forceinline__ float compute_elec_energy_f16_(half const r, half const qi, half const qj,
                                                          std::int32_t const is_1_4) {
    half const coulomb_const = __float2half(332.0716f);
    half const delta = __float2half(0.05f);

    half const r_buffered = __hadd(r, delta);
    half const inv_r = h_rcp_(r_buffered);

    half e = __hmul(__hmul(__hmul(coulomb_const, qi), qj), inv_r);

    // Branchless 1-4 scaling: scale = 1.0 + is_1_4 * (0.75 - 1.0)
    float const scale = 1.0f + static_cast<float>(is_1_4) * scale_1_4_diff_f32;
    return __half2float(e) * scale;
}

/**
 * MMFF94 Buffered Coulomb electrostatic energy (FP32 compute and accumulation).
 * Uses branchless scaling: scale = 1.0 + is_1_4 * (0.75 - 1.0)
 */
__device__ __forceinline__ float compute_elec_energy_f32_(float const r, float const qi, float const qj,
                                                          std::int32_t const is_1_4) {
    float const r_buffered = r + delta_f32;
    float const inv_r = __frcp_rn(r_buffered);

    float e = coulomb_const_f32 * qi * qj * inv_r;

    // Branchless 1-4 scaling
    float const scale = 1.0f + static_cast<float>(is_1_4) * scale_1_4_diff_f32;
    return e * scale;
}

/**
 * MMFF94 Buffered Coulomb electrostatic energy using half2 packed operations.
 * Processes 2 pairs simultaneously for 2x throughput.
 *
 * Returns: Two energies (with 1-4 scaling applied) as float (low + high).
 *
 * Note: Computation in FP16 with packed half2, 1-4 scaling in FP32.
 */
__device__ __forceinline__ float compute_elec_energy_f16_pair_(half2 const r, half2 const qi, half2 const qj,
                                                               std::int32_t const is_1_4_low,
                                                               std::int32_t const is_1_4_high) {
    half2 const coulomb_const = __float2half2_rn(332.0716f);
    half2 const delta = __float2half2_rn(0.05f);

    half2 const r_buffered = __hadd2(r, delta);
    half2 const inv_r = h2_rcp_(r_buffered);

    half2 const e = __hmul2(__hmul2(__hmul2(coulomb_const, qi), qj), inv_r);

    // Extract and apply 1-4 scaling in FP32
    float const e_low = __half2float(__low2half(e));
    float const e_high = __half2float(__high2half(e));

    float const scale_low = 1.0f + static_cast<float>(is_1_4_low) * scale_1_4_diff_f32;
    float const scale_high = 1.0f + static_cast<float>(is_1_4_high) * scale_1_4_diff_f32;

    return e_low * scale_low + e_high * scale_high;
}

#pragma endregion

#pragma region VdW Gradient Computation

/**
 * MMFF94 Buffered 14-7 van der Waals gradient (FP32 compute).
 * Returns dE/dr (not force, just the energy gradient).
 */
__device__ __forceinline__ float compute_vdw_gradient_f32_(float const r, float const epsilon, float const r_star) {
    float const r_safe = (r > 0.001f) ? r : 0.001f;

    float const inv_rstar = __frcp_rn(r_star);
    float const rho = r_safe * inv_rstar;

    float const rho2 = rho * rho;
    float const rho4 = rho2 * rho2;
    float const rho6 = rho4 * rho2;
    float const rho7 = rho6 * rho;

    float const c_107 = 1.07f;
    float const c_007 = 0.07f;
    float const c_112 = 1.12f;
    float const c_012 = 0.12f;
    float const c_2 = 2.0f;

    // Repulsive term: (1.07/(rho+0.07))^7
    float const rep_denom = rho + c_007;
    float const rep_ratio = c_107 * __frcp_rn(rep_denom);
    float const rep_2 = rep_ratio * rep_ratio;
    float const rep_4 = rep_2 * rep_2;
    float const rep_6 = rep_4 * rep_2;
    float const rep_7 = rep_6 * rep_ratio;

    // Attractive term: (1.12/(rho^7+0.12)) - 2
    float const att_denom = rho7 + c_012;
    float const att_term = c_112 * __frcp_rn(att_denom) - c_2;

    // Gradient of repulsive: d/drho[(1.07/(rho+0.07))^7] = 7*(1.07/(rho+0.07))^6 * (-1.07/(rho+0.07)^2)
    float const d_rep = -7.0f * rep_6 * c_107 / (rep_denom * rep_denom);

    // Gradient of attractive: d/drho[(1.12/(rho^7+0.12)) - 2] = -1.12 * 7*rho^6 / (rho^7+0.12)^2
    float const d_att = -c_112 * 7.0f * rho6 / (att_denom * att_denom);

    // Product rule: d/drho[rep * att] = d_rep * att + rep * d_att
    float const dE_drho = epsilon * (d_rep * att_term + rep_7 * d_att);

    // Chain rule: dE/dr = dE/drho * drho/dr = dE/drho * (1/r_star)
    float const dE_dr = dE_drho * inv_rstar;

    return dE_dr;
}

/**
 * MMFF94 Buffered 14-7 van der Waals gradient (FP16 compute, FP32 return).
 * Returns dE/dr (not force, just the energy gradient).
 */
__device__ __forceinline__ float compute_vdw_gradient_f16_(half const r, half const epsilon, half const r_star) {
    half const r_safe = __hgt(r, __float2half(0.001f)) ? r : __float2half(0.001f);

    half const inv_rstar = h_rcp_(r_star);
    half const rho = __hmul(r_safe, inv_rstar);

    half const rho2 = __hmul(rho, rho);
    half const rho4 = __hmul(rho2, rho2);
    half const rho6 = __hmul(rho4, rho2);
    half const rho7 = __hmul(rho6, rho);

    half const c_107 = __float2half(1.07f);
    half const c_007 = __float2half(0.07f);
    half const c_112 = __float2half(1.12f);
    half const c_012 = __float2half(0.12f);
    half const c_2 = __float2half(2.0f);

    // Repulsive term: (1.07/(rho+0.07))^7
    half const rep_denom = __hadd(rho, c_007);
    half const rep_ratio = __hmul(c_107, h_rcp_(rep_denom));
    half const rep_2 = __hmul(rep_ratio, rep_ratio);
    half const rep_4 = __hmul(rep_2, rep_2);
    half const rep_6 = __hmul(rep_4, rep_2);
    half const rep_7 = __hmul(rep_6, rep_ratio);

    // Attractive term: (1.12/(rho^7+0.12)) - 2
    half const att_denom = __hadd(rho7, c_012);
    half const att_term = __hsub(__hmul(c_112, h_rcp_(att_denom)), c_2);

    // Gradient of repulsive
    half const rep_denom_sq = __hmul(rep_denom, rep_denom);
    half const d_rep = __hmul(__hmul(__hmul(__float2half(-7.0f), rep_6), c_107), h_rcp_(rep_denom_sq));

    // Gradient of attractive
    half const att_denom_sq = __hmul(att_denom, att_denom);
    half const d_att = __hmul(__hmul(__float2half(-7.84f), rho6), h_rcp_(att_denom_sq));

    // Product rule
    half const dE_drho = __hmul(epsilon, __hadd(__hmul(d_rep, att_term), __hmul(rep_7, d_att)));

    // Chain rule
    half const dE_dr = __hmul(dE_drho, inv_rstar);

    return __half2float(dE_dr);
}

#pragma endregion

#pragma region VdW Energy Kernels

/**
 * MMFF94 van der Waals energy kernel (FP16 compute with half2 packed operations, FP32 accumulation).
 *
 * One warp (32 threads) per molecule.
 * Thread k processes pairs k, k+32, k+64... via strided iteration.
 * Uses half2 to process 2 pairs at once when possible (2x stride).
 * Falls back to single half for odd remaining pairs.
 * Warp reduction sums energies across lanes.
 * Lane 0 writes final result.
 */
__global__ void mmff94_vdw_energy_kernel_f16(half const* coord_x, half const* coord_y, half const* coord_z,
                                             std::int32_t const* nb_i, std::int32_t const* nb_j, half const* nb_epsilon,
                                             half const* nb_r_star, std::int32_t const* mol_nb_start,
                                             std::int32_t const* mol_nb_end, float* out_vdw,
                                             std::uint32_t const n_molecules) {
    std::uint32_t const mol_idx = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;

    if (mol_idx >= n_molecules) {
        return;
    }

    std::uint32_t const nb_start = static_cast<std::uint32_t>(mol_nb_start[mol_idx]);
    std::uint32_t const nb_end = static_cast<std::uint32_t>(mol_nb_end[mol_idx]);
    std::uint32_t const n_pairs = nb_end - nb_start;
    float vdw_accum = 0.0f;

    // Process pairs in groups of 2 using half2 (2x warp_size stride)
    std::uint32_t const stride_2x = warp_size * 2;
    for (std::uint32_t pair_idx = nb_start + lane * 2; pair_idx + 1 < nb_end; pair_idx += stride_2x) {
        // Load pair 0
        std::int32_t const i0 = nb_i[pair_idx];
        std::int32_t const j0 = nb_j[pair_idx];

        // Load pair 1
        std::int32_t const i1 = nb_i[pair_idx + 1];
        std::int32_t const j1 = nb_j[pair_idx + 1];

        // Pack coordinates as half2
        half2 const xi = make_half2(coord_x[i0], coord_x[i1]);
        half2 const yi = make_half2(coord_y[i0], coord_y[i1]);
        half2 const zi = make_half2(coord_z[i0], coord_z[i1]);
        half2 const xj = make_half2(coord_x[j0], coord_x[j1]);
        half2 const yj = make_half2(coord_y[j0], coord_y[j1]);
        half2 const zj = make_half2(coord_z[j0], coord_z[j1]);

        half2 const r = compute_distance_f16_pair_(xi, yi, zi, xj, yj, zj);

        half2 const eps = make_half2(nb_epsilon[pair_idx], nb_epsilon[pair_idx + 1]);
        half2 const rstar = make_half2(nb_r_star[pair_idx], nb_r_star[pair_idx + 1]);

        vdw_accum += compute_vdw_energy_f16_pair_(r, eps, rstar);
    }

    // Handle remaining single pair if n_pairs is odd
    if ((n_pairs & 1) && (lane == 0)) {
        std::uint32_t const pair_idx = nb_end - 1;
        std::int32_t const i = nb_i[pair_idx];
        std::int32_t const j = nb_j[pair_idx];

        half const xi = coord_x[i];
        half const yi = coord_y[i];
        half const zi = coord_z[i];
        half const xj = coord_x[j];
        half const yj = coord_y[j];
        half const zj = coord_z[j];

        half const r = compute_distance_f16_(xi, yi, zi, xj, yj, zj);

        half const eps = nb_epsilon[pair_idx];
        half const rstar = nb_r_star[pair_idx];

        vdw_accum += compute_vdw_energy_f16_(r, eps, rstar);
    }

    float const vdw_total = warp_reduce_sum_f32_(vdw_accum);

    if (lane == 0) {
        out_vdw[mol_idx] = vdw_total;
    }
}

/**
 * MMFF94 van der Waals energy kernel (FP32 compute and accumulation).
 */
__global__ void mmff94_vdw_energy_kernel_f32(float const* coord_x, float const* coord_y, float const* coord_z,
                                             std::int32_t const* nb_i, std::int32_t const* nb_j,
                                             float const* nb_epsilon, float const* nb_r_star,
                                             std::int32_t const* mol_nb_start, std::int32_t const* mol_nb_end,
                                             float* out_vdw, std::uint32_t const n_molecules) {
    std::uint32_t const mol_idx = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;

    if (mol_idx >= n_molecules) {
        return;
    }

    std::uint32_t const nb_start = static_cast<std::uint32_t>(mol_nb_start[mol_idx]);
    std::uint32_t const nb_end = static_cast<std::uint32_t>(mol_nb_end[mol_idx]);
    float vdw_accum = 0.0f;

    for (std::uint32_t pair_idx = nb_start + lane; pair_idx < nb_end; pair_idx += warp_size) {
        std::int32_t const i = nb_i[pair_idx];
        std::int32_t const j = nb_j[pair_idx];

        float const xi = coord_x[i];
        float const yi = coord_y[i];
        float const zi = coord_z[i];
        float const xj = coord_x[j];
        float const yj = coord_y[j];
        float const zj = coord_z[j];

        float const r = compute_distance_f32_(xi, yi, zi, xj, yj, zj);

        float const eps = nb_epsilon[pair_idx];
        float const rstar = nb_r_star[pair_idx];

        vdw_accum += compute_vdw_energy_f32_(r, eps, rstar);
    }

    float const vdw_total = warp_reduce_sum_f32_(vdw_accum);

    if (lane == 0) {
        out_vdw[mol_idx] = vdw_total;
    }
}

#pragma endregion

#pragma region Electrostatic Energy Kernels

/**
 * MMFF94 electrostatic energy kernel (FP16 compute with half2 packed operations, FP32 accumulation).
 *
 * One warp (32 threads) per molecule.
 * Thread k processes pairs k, k+32, k+64... via strided iteration.
 * Uses half2 to process 2 pairs at once when possible (2x stride).
 * Falls back to single half for odd remaining pairs.
 * Warp reduction sums energies across lanes.
 * Lane 0 writes final result.
 */
__global__ void mmff94_elec_energy_kernel_f16(half const* coord_x, half const* coord_y, half const* coord_z,
                                              std::int32_t const* nb_i, std::int32_t const* nb_j, half const* nb_qi,
                                              half const* nb_qj, std::int32_t const* nb_is_1_4,
                                              std::int32_t const* mol_nb_start, std::int32_t const* mol_nb_end,
                                              float* out_elec, std::uint32_t const n_molecules) {
    std::uint32_t const mol_idx = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;

    if (mol_idx >= n_molecules) {
        return;
    }

    std::uint32_t const nb_start = static_cast<std::uint32_t>(mol_nb_start[mol_idx]);
    std::uint32_t const nb_end = static_cast<std::uint32_t>(mol_nb_end[mol_idx]);
    std::uint32_t const n_pairs = nb_end - nb_start;
    float elec_accum = 0.0f;

    // Process pairs in groups of 2 using half2 (2x warp_size stride)
    std::uint32_t const stride_2x = warp_size * 2;
    for (std::uint32_t pair_idx = nb_start + lane * 2; pair_idx + 1 < nb_end; pair_idx += stride_2x) {
        // Load pair 0
        std::int32_t const i0 = nb_i[pair_idx];
        std::int32_t const j0 = nb_j[pair_idx];

        // Load pair 1
        std::int32_t const i1 = nb_i[pair_idx + 1];
        std::int32_t const j1 = nb_j[pair_idx + 1];

        // Pack coordinates as half2
        half2 const xi = make_half2(coord_x[i0], coord_x[i1]);
        half2 const yi = make_half2(coord_y[i0], coord_y[i1]);
        half2 const zi = make_half2(coord_z[i0], coord_z[i1]);
        half2 const xj = make_half2(coord_x[j0], coord_x[j1]);
        half2 const yj = make_half2(coord_y[j0], coord_y[j1]);
        half2 const zj = make_half2(coord_z[j0], coord_z[j1]);

        half2 const r = compute_distance_f16_pair_(xi, yi, zi, xj, yj, zj);

        half2 const qi = make_half2(nb_qi[pair_idx], nb_qi[pair_idx + 1]);
        half2 const qj = make_half2(nb_qj[pair_idx], nb_qj[pair_idx + 1]);
        std::int32_t const is_1_4_low = nb_is_1_4[pair_idx];
        std::int32_t const is_1_4_high = nb_is_1_4[pair_idx + 1];

        elec_accum += compute_elec_energy_f16_pair_(r, qi, qj, is_1_4_low, is_1_4_high);
    }

    // Handle remaining single pair if n_pairs is odd
    if ((n_pairs & 1) && (lane == 0)) {
        std::uint32_t const pair_idx = nb_end - 1;
        std::int32_t const i = nb_i[pair_idx];
        std::int32_t const j = nb_j[pair_idx];

        half const xi = coord_x[i];
        half const yi = coord_y[i];
        half const zi = coord_z[i];
        half const xj = coord_x[j];
        half const yj = coord_y[j];
        half const zj = coord_z[j];

        half const r = compute_distance_f16_(xi, yi, zi, xj, yj, zj);

        half const qi = nb_qi[pair_idx];
        half const qj = nb_qj[pair_idx];
        std::int32_t const is_1_4 = nb_is_1_4[pair_idx]; // Branchless: use int directly

        elec_accum += compute_elec_energy_f16_(r, qi, qj, is_1_4);
    }

    float const elec_total = warp_reduce_sum_f32_(elec_accum);

    if (lane == 0) {
        out_elec[mol_idx] = elec_total;
    }
}

/**
 * MMFF94 electrostatic energy kernel (FP32 compute and accumulation).
 */
__global__ void mmff94_elec_energy_kernel_f32(float const* coord_x, float const* coord_y, float const* coord_z,
                                              std::int32_t const* nb_i, std::int32_t const* nb_j, float const* nb_qi,
                                              float const* nb_qj, std::int32_t const* nb_is_1_4,
                                              std::int32_t const* mol_nb_start, std::int32_t const* mol_nb_end,
                                              float* out_elec, std::uint32_t const n_molecules) {
    std::uint32_t const mol_idx = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;

    if (mol_idx >= n_molecules) {
        return;
    }

    std::uint32_t const nb_start = static_cast<std::uint32_t>(mol_nb_start[mol_idx]);
    std::uint32_t const nb_end = static_cast<std::uint32_t>(mol_nb_end[mol_idx]);
    float elec_accum = 0.0f;

    for (std::uint32_t pair_idx = nb_start + lane; pair_idx < nb_end; pair_idx += warp_size) {
        std::int32_t const i = nb_i[pair_idx];
        std::int32_t const j = nb_j[pair_idx];

        float const xi = coord_x[i];
        float const yi = coord_y[i];
        float const zi = coord_z[i];
        float const xj = coord_x[j];
        float const yj = coord_y[j];
        float const zj = coord_z[j];

        float const r = compute_distance_f32_(xi, yi, zi, xj, yj, zj);

        float const qi = nb_qi[pair_idx];
        float const qj = nb_qj[pair_idx];
        std::int32_t const is_1_4 = nb_is_1_4[pair_idx]; // Branchless: use int directly

        elec_accum += compute_elec_energy_f32_(r, qi, qj, is_1_4);
    }

    float const elec_total = warp_reduce_sum_f32_(elec_accum);

    if (lane == 0) {
        out_elec[mol_idx] = elec_total;
    }
}

#pragma endregion

#pragma region Electrostatic Gradient Computation

/**
 * MMFF94 Buffered Coulomb electrostatic gradient (FP32 compute).
 *
 * dE/dr = -332.0716 * qi * qj / (r + 0.05)^2
 * Force on atom j: F_j = -(dE/dr) * (r_vec / |r_vec|)
 * Force on atom i: F_i = -F_j (Newton's 3rd law)
 *
 * For 1-4 interactions: scale gradient by 0.75.
 * Uses branchless scaling: scale = 1.0 + is_1_4 * (0.75 - 1.0)
 */
__device__ __forceinline__ void compute_elec_gradient_f32_(float const r, float const qi, float const qj,
                                                           std::int32_t const is_1_4, float const dx, float const dy,
                                                           float const dz, float* fx, float* fy, float* fz) {
    float const r_buffered = r + delta_f32;
    float const r_buffered_sq = r_buffered * r_buffered;

    // dE/dr = -332.0716 * qi * qj / (r + 0.05)^2
    float const grad_magnitude = -coulomb_const_f32 * qi * qj * __frcp_rn(r_buffered_sq);

    // Branchless 1-4 scaling
    float const scale = 1.0f + static_cast<float>(is_1_4) * scale_1_4_diff_f32;
    float const scaled_grad = grad_magnitude * scale;

    // Force direction: F_j = -(dE/dr) * (r_vec / |r_vec|)
    float const inv_r = __frcp_rn(r);
    *fx = -scaled_grad * dx * inv_r;
    *fy = -scaled_grad * dy * inv_r;
    *fz = -scaled_grad * dz * inv_r;
}

/**
 * MMFF94 Buffered Coulomb electrostatic gradient (FP16 compute, FP32 output).
 *
 * Uses FP16 for intermediate computations, returns forces in FP32.
 */
__device__ __forceinline__ void compute_elec_gradient_f16_(half const r, half const qi, half const qj,
                                                           std::int32_t const is_1_4, half const dx, half const dy,
                                                           half const dz, float* fx, float* fy, float* fz) {
    half const coulomb_const = __float2half(332.0716f);
    half const delta = __float2half(0.05f);

    half const r_buffered = __hadd(r, delta);
    half const r_buffered_sq = __hmul(r_buffered, r_buffered);

    // dE/dr = -332.0716 * qi * qj / (r + 0.05)^2
    half const grad_magnitude_abs = __hmul(__hmul(__hmul(coulomb_const, qi), qj), h_rcp_(r_buffered_sq));
    half const grad_magnitude = __hneg(grad_magnitude_abs);

    // Branchless 1-4 scaling (in FP32 for precision)
    float const scale = 1.0f + static_cast<float>(is_1_4) * scale_1_4_diff_f32;
    float const scaled_grad = __half2float(grad_magnitude) * scale;

    // Force direction: F_j = -(dE/dr) * (r_vec / |r_vec|)
    half const inv_r = h_rcp_(r);
    *fx = -scaled_grad * __half2float(dx) * __half2float(inv_r);
    *fy = -scaled_grad * __half2float(dy) * __half2float(inv_r);
    *fz = -scaled_grad * __half2float(dz) * __half2float(inv_r);
}

#pragma endregion

#pragma region Electrostatic Gradient Kernels

/**
 * MMFF94 electrostatic gradient kernel (FP32 compute and accumulation).
 *
 * One warp (32 threads) per molecule.
 * Thread k processes pairs k, k+32, k+64... via strided iteration.
 * Atomic operations accumulate forces to handle race conditions.
 */
__global__ void mmff94_elec_gradient_kernel_f32(float const* coord_x, float const* coord_y, float const* coord_z,
                                                std::int32_t const* nb_i, std::int32_t const* nb_j, float const* nb_qi,
                                                float const* nb_qj, std::int32_t const* nb_is_1_4,
                                                std::int32_t const* mol_nb_start, std::int32_t const* mol_nb_end,
                                                float* grad_x, float* grad_y, float* grad_z,
                                                std::uint32_t const n_molecules) {
    std::uint32_t const mol_idx = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;

    if (mol_idx >= n_molecules) {
        return;
    }

    std::uint32_t const nb_start = static_cast<std::uint32_t>(mol_nb_start[mol_idx]);
    std::uint32_t const nb_end = static_cast<std::uint32_t>(mol_nb_end[mol_idx]);

    for (std::uint32_t pair_idx = nb_start + lane; pair_idx < nb_end; pair_idx += warp_size) {
        std::int32_t const i = nb_i[pair_idx];
        std::int32_t const j = nb_j[pair_idx];

        float const xi = coord_x[i];
        float const yi = coord_y[i];
        float const zi = coord_z[i];
        float const xj = coord_x[j];
        float const yj = coord_y[j];
        float const zj = coord_z[j];

        float const dx = xj - xi;
        float const dy = yj - yi;
        float const dz = zj - zi;

        float const r = compute_distance_f32_(xi, yi, zi, xj, yj, zj);

        float const qi = nb_qi[pair_idx];
        float const qj = nb_qj[pair_idx];
        std::int32_t const is_1_4 = nb_is_1_4[pair_idx];

        float fx, fy, fz;
        compute_elec_gradient_f32_(r, qi, qj, is_1_4, dx, dy, dz, &fx, &fy, &fz);

        // Accumulate forces (Newton's 3rd law: F_i = -F_j)
        atomicAdd(&grad_x[i], -fx);
        atomicAdd(&grad_y[i], -fy);
        atomicAdd(&grad_z[i], -fz);
        atomicAdd(&grad_x[j], fx);
        atomicAdd(&grad_y[j], fy);
        atomicAdd(&grad_z[j], fz);
    }
}

/**
 * MMFF94 electrostatic gradient kernel (FP16 compute, FP32 accumulation).
 *
 * One warp (32 threads) per molecule.
 * Thread k processes pairs k, k+32, k+64... via strided iteration.
 * Atomic operations accumulate forces to handle race conditions.
 */
__global__ void mmff94_elec_gradient_kernel_f16(half const* coord_x, half const* coord_y, half const* coord_z,
                                                std::int32_t const* nb_i, std::int32_t const* nb_j, half const* nb_qi,
                                                half const* nb_qj, std::int32_t const* nb_is_1_4,
                                                std::int32_t const* mol_nb_start, std::int32_t const* mol_nb_end,
                                                float* grad_x, float* grad_y, float* grad_z,
                                                std::uint32_t const n_molecules) {
    std::uint32_t const mol_idx = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;

    if (mol_idx >= n_molecules) {
        return;
    }

    std::uint32_t const nb_start = static_cast<std::uint32_t>(mol_nb_start[mol_idx]);
    std::uint32_t const nb_end = static_cast<std::uint32_t>(mol_nb_end[mol_idx]);

    for (std::uint32_t pair_idx = nb_start + lane; pair_idx < nb_end; pair_idx += warp_size) {
        std::int32_t const i = nb_i[pair_idx];
        std::int32_t const j = nb_j[pair_idx];

        half const xi = coord_x[i];
        half const yi = coord_y[i];
        half const zi = coord_z[i];
        half const xj = coord_x[j];
        half const yj = coord_y[j];
        half const zj = coord_z[j];

        half const dx = __hsub(xj, xi);
        half const dy = __hsub(yj, yi);
        half const dz = __hsub(zj, zi);

        half const r = compute_distance_f16_(xi, yi, zi, xj, yj, zj);

        half const qi = nb_qi[pair_idx];
        half const qj = nb_qj[pair_idx];
        std::int32_t const is_1_4 = nb_is_1_4[pair_idx];

        float fx, fy, fz;
        compute_elec_gradient_f16_(r, qi, qj, is_1_4, dx, dy, dz, &fx, &fy, &fz);

        // Accumulate forces (Newton's 3rd law: F_i = -F_j)
        atomicAdd(&grad_x[i], -fx);
        atomicAdd(&grad_y[i], -fy);
        atomicAdd(&grad_z[i], -fz);
        atomicAdd(&grad_x[j], fx);
        atomicAdd(&grad_y[j], fy);
        atomicAdd(&grad_z[j], fz);
    }
}

#pragma endregion

#pragma region VdW Gradient Kernels

/**
 * MMFF94 van der Waals gradient kernel (FP32 compute and accumulation).
 *
 * One warp (32 threads) per molecule.
 * Thread k processes pairs k, k+32, k+64... via strided iteration.
 * Atomic operations accumulate forces to handle race conditions.
 */
__global__ void mmff94_vdw_gradient_kernel_f32(float const* coord_x, float const* coord_y, float const* coord_z,
                                               std::int32_t const* nb_i, std::int32_t const* nb_j,
                                               float const* nb_epsilon, float const* nb_r_star,
                                               std::int32_t const* mol_nb_start, std::int32_t const* mol_nb_end,
                                               float* grad_x, float* grad_y, float* grad_z,
                                               std::uint32_t const n_molecules) {
    std::uint32_t const mol_idx = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;

    if (mol_idx >= n_molecules) {
        return;
    }

    std::uint32_t const nb_start = static_cast<std::uint32_t>(mol_nb_start[mol_idx]);
    std::uint32_t const nb_end = static_cast<std::uint32_t>(mol_nb_end[mol_idx]);

    for (std::uint32_t pair_idx = nb_start + lane; pair_idx < nb_end; pair_idx += warp_size) {
        std::int32_t const i = nb_i[pair_idx];
        std::int32_t const j = nb_j[pair_idx];

        float const xi = coord_x[i];
        float const yi = coord_y[i];
        float const zi = coord_z[i];
        float const xj = coord_x[j];
        float const yj = coord_y[j];
        float const zj = coord_z[j];

        float const dx = xj - xi;
        float const dy = yj - yi;
        float const dz = zj - zi;

        float const r = compute_distance_f32_(xi, yi, zi, xj, yj, zj);

        float const eps = nb_epsilon[pair_idx];
        float const rstar = nb_r_star[pair_idx];

        // Get dE/dr
        float const dE_dr = compute_vdw_gradient_f32_(r, eps, rstar);

        // Force: F = -(dE/dr) * (r_vec / |r_vec|)
        float const inv_r = __frcp_rn(r);
        float const fx = -dE_dr * dx * inv_r;
        float const fy = -dE_dr * dy * inv_r;
        float const fz = -dE_dr * dz * inv_r;

        // Accumulate forces (Newton's 3rd law: F_i = -F_j)
        atomicAdd(&grad_x[i], -fx);
        atomicAdd(&grad_y[i], -fy);
        atomicAdd(&grad_z[i], -fz);
        atomicAdd(&grad_x[j], fx);
        atomicAdd(&grad_y[j], fy);
        atomicAdd(&grad_z[j], fz);
    }
}

/**
 * MMFF94 van der Waals gradient kernel (FP16 compute, FP32 accumulation).
 *
 * One warp (32 threads) per molecule.
 * Thread k processes pairs k, k+32, k+64... via strided iteration.
 * Atomic operations accumulate forces to handle race conditions.
 */
__global__ void mmff94_vdw_gradient_kernel_f16(half const* coord_x, half const* coord_y, half const* coord_z,
                                               std::int32_t const* nb_i, std::int32_t const* nb_j,
                                               half const* nb_epsilon, half const* nb_r_star,
                                               std::int32_t const* mol_nb_start, std::int32_t const* mol_nb_end,
                                               float* grad_x, float* grad_y, float* grad_z,
                                               std::uint32_t const n_molecules) {
    std::uint32_t const mol_idx = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;

    if (mol_idx >= n_molecules) {
        return;
    }

    std::uint32_t const nb_start = static_cast<std::uint32_t>(mol_nb_start[mol_idx]);
    std::uint32_t const nb_end = static_cast<std::uint32_t>(mol_nb_end[mol_idx]);

    for (std::uint32_t pair_idx = nb_start + lane; pair_idx < nb_end; pair_idx += warp_size) {
        std::int32_t const i = nb_i[pair_idx];
        std::int32_t const j = nb_j[pair_idx];

        half const xi = coord_x[i];
        half const yi = coord_y[i];
        half const zi = coord_z[i];
        half const xj = coord_x[j];
        half const yj = coord_y[j];
        half const zj = coord_z[j];

        half const dx = __hsub(xj, xi);
        half const dy = __hsub(yj, yi);
        half const dz = __hsub(zj, zi);

        half const r = compute_distance_f16_(xi, yi, zi, xj, yj, zj);

        half const eps = nb_epsilon[pair_idx];
        half const rstar = nb_r_star[pair_idx];

        // Get dE/dr (returns FP32)
        float const dE_dr = compute_vdw_gradient_f16_(r, eps, rstar);

        // Force: F = -(dE/dr) * (r_vec / |r_vec|)
        half const inv_r = h_rcp_(r);
        float const fx = -dE_dr * __half2float(dx) * __half2float(inv_r);
        float const fy = -dE_dr * __half2float(dy) * __half2float(inv_r);
        float const fz = -dE_dr * __half2float(dz) * __half2float(inv_r);

        // Accumulate forces (Newton's 3rd law: F_i = -F_j)
        atomicAdd(&grad_x[i], -fx);
        atomicAdd(&grad_y[i], -fy);
        atomicAdd(&grad_z[i], -fz);
        atomicAdd(&grad_x[j], fx);
        atomicAdd(&grad_y[j], fy);
        atomicAdd(&grad_z[j], fz);
    }
}

#pragma endregion

#pragma region L-BFGS Optimizer

/**
 * L-BFGS workspace size per molecule.
 *
 * Memory layout per molecule (for memory_depth m and n_atoms atoms):
 * - s_history: m * n_atoms * 3 (position deltas)
 * - y_history: m * n_atoms * 3 (gradient deltas)
 * - rho: m (scaling factors)
 * - alpha: m (line search coefficients)
 * - gradient_prev: n_atoms * 3 (previous gradient)
 */
__device__ __forceinline__ std::uint32_t lbfgs_workspace_size_(std::uint32_t const n_atoms,
                                                               std::uint32_t const memory_depth) {
    return memory_depth * n_atoms * 3 * 2 // s_history + y_history
           + memory_depth                 // rho
           + memory_depth                 // alpha
           + n_atoms * 3;                 // gradient_prev
}

/**
 * Compute dot product using warp reduction.
 * Each thread processes elements in strided fashion, then reduces across warp.
 */
__device__ __forceinline__ float warp_dot_product_f32_(float const* a, float const* b, std::uint32_t const n,
                                                       std::uint32_t const lane) {
    float dot = 0.0f;
    for (std::uint32_t i = lane; i < n; i += warp_size) {
        dot = fmaf(a[i], b[i], dot);
    }
    return warp_reduce_sum_f32_(dot);
}

/**
 * Compute RMS (root mean square) using warp reduction.
 * RMS = sqrt(sum(x[i]^2) / n)
 */
__device__ __forceinline__ float warp_rms_f32_(float const* x, std::uint32_t const n, std::uint32_t const lane) {
    float sum_sq = 0.0f;
    for (std::uint32_t i = lane; i < n; i += warp_size) {
        float const val = x[i];
        sum_sq = fmaf(val, val, sum_sq);
    }
    float const total_sum_sq = warp_reduce_sum_f32_(sum_sq);
    return sqrtf(total_sum_sq * __frcp_rn(static_cast<float>(n)));
}

/**
 * L-BFGS two-loop recursion for computing search direction.
 *
 * Given gradient g and history (s, y, rho), computes search direction d = -H * g
 * where H is the inverse Hessian approximation.
 *
 * One warp processes this sequentially. Lane 0 does scalar ops, all lanes do vector ops.
 */
__device__ __forceinline__ void lbfgs_two_loop_recursion_(float const* gradient, float* search_dir, float* s_history,
                                                          float* y_history, float* rho, float* alpha,
                                                          std::uint32_t const n_elements, std::uint32_t const k,
                                                          std::uint32_t const memory_depth, std::uint32_t const lane) {
    // Copy gradient to search_dir: q = g
    for (std::uint32_t i = lane; i < n_elements; i += warp_size) {
        search_dir[i] = gradient[i];
    }
    __syncwarp(full_mask);

    // First loop (backward through history)
    for (std::uint32_t i = 0; i < k; ++i) {
        std::uint32_t const idx = (k - 1 - i) % memory_depth;
        float* s = s_history + idx * n_elements;
        float* y = y_history + idx * n_elements;

        float const rho_val = rho[idx];
        float const dot = warp_dot_product_f32_(s, search_dir, n_elements, lane);

        float alpha_val = 0.0f;
        if (lane == 0) {
            alpha_val = rho_val * dot;
            alpha[idx] = alpha_val;
        }
        alpha_val = __shfl_sync(full_mask, alpha_val, 0);

        // q = q - alpha * y
        for (std::uint32_t j = lane; j < n_elements; j += warp_size) {
            search_dir[j] -= alpha_val * y[j];
        }
        __syncwarp(full_mask);
    }

    // Initial Hessian scaling: H0 = (s_k · y_k) / (y_k · y_k)
    if (k > 0) {
        std::uint32_t const last_idx = (k - 1) % memory_depth;
        float* s_last = s_history + last_idx * n_elements;
        float* y_last = y_history + last_idx * n_elements;

        float const s_dot_y = warp_dot_product_f32_(s_last, y_last, n_elements, lane);
        float const y_dot_y = warp_dot_product_f32_(y_last, y_last, n_elements, lane);

        float gamma = 1.0f;
        if (lane == 0) {
            gamma = s_dot_y * __frcp_rn(y_dot_y);
        }
        gamma = __shfl_sync(full_mask, gamma, 0);

        // r = gamma * q
        for (std::uint32_t i = lane; i < n_elements; i += warp_size) {
            search_dir[i] *= gamma;
        }
        __syncwarp(full_mask);
    }

    // Second loop (forward through history)
    for (std::uint32_t i = 0; i < k; ++i) {
        std::uint32_t const idx = i % memory_depth;
        float* s = s_history + idx * n_elements;
        float* y = y_history + idx * n_elements;

        float const rho_val = rho[idx];
        float const alpha_val = alpha[idx];
        float const dot = warp_dot_product_f32_(y, search_dir, n_elements, lane);

        float beta = 0.0f;
        if (lane == 0) {
            beta = rho_val * dot;
        }
        beta = __shfl_sync(full_mask, beta, 0);

        // r = r + s * (alpha - beta)
        float const coeff = alpha_val - beta;
        for (std::uint32_t j = lane; j < n_elements; j += warp_size) {
            search_dir[j] = fmaf(s[j], coeff, search_dir[j]);
        }
        __syncwarp(full_mask);
    }

    // Negate to get descent direction: d = -r
    for (std::uint32_t i = lane; i < n_elements; i += warp_size) {
        search_dir[i] = -search_dir[i];
    }
    __syncwarp(full_mask);
}

/**
 * Backtracking line search with Armijo condition.
 *
 * Finds step size alpha that satisfies:
 * E(x + alpha*d) <= E(x) + c1 * alpha * (grad · d)
 *
 * Returns: step size alpha
 */
__device__ __forceinline__ float
lbfgs_line_search_f32_(float const* coord_x, float const* coord_y, float const* coord_z, float const* search_dir,
                       float const* gradient, std::int32_t const* nb_i, std::int32_t const* nb_j,
                       float const* nb_epsilon, float const* nb_r_star, float const* nb_qi, float const* nb_qj,
                       std::int32_t const* nb_is_1_4, std::uint32_t const nb_start, std::uint32_t const nb_end,
                       std::uint32_t const atom_start, std::uint32_t const n_atoms, float const energy_current,
                       std::uint32_t const lane, float* trial_coords) {
    constexpr float c1 = 1e-4f; // Armijo parameter
    constexpr float alpha_init = 1.0f;
    constexpr float rho_decay = 0.5f;
    constexpr int max_line_search = 20;

    // Compute grad · d (directional derivative)
    std::uint32_t const n_elements = n_atoms * 3;
    float const grad_dot_d = warp_dot_product_f32_(gradient, search_dir, n_elements, lane);

    float alpha = alpha_init;

    // Trial coordinates (passed from caller to avoid overwriting gradient/search_dir)
    float* trial_x = trial_coords;
    float* trial_y = trial_x + n_atoms;
    float* trial_z = trial_y + n_atoms;

    for (int iter = 0; iter < max_line_search; ++iter) {
        // Compute trial position: x_trial = x + alpha * d
        for (std::uint32_t i = lane; i < n_atoms; i += warp_size) {
            std::uint32_t const atom_idx = atom_start + i;
            trial_x[i] = coord_x[atom_idx] + alpha * search_dir[i];
            trial_y[i] = coord_y[atom_idx] + alpha * search_dir[n_atoms + i];
            trial_z[i] = coord_z[atom_idx] + alpha * search_dir[2 * n_atoms + i];
        }
        __syncwarp(full_mask);

        // Evaluate energy at trial position
        float energy_trial = 0.0f;

        for (std::uint32_t pair_idx = nb_start + lane; pair_idx < nb_end; pair_idx += warp_size) {
            std::int32_t const i = nb_i[pair_idx] - static_cast<std::int32_t>(atom_start);
            std::int32_t const j = nb_j[pair_idx] - static_cast<std::int32_t>(atom_start);

            float const xi = trial_x[i];
            float const yi = trial_y[i];
            float const zi = trial_z[i];
            float const xj = trial_x[j];
            float const yj = trial_y[j];
            float const zj = trial_z[j];

            float const r = compute_distance_f32_(xi, yi, zi, xj, yj, zj);

            // VdW energy
            float const eps = nb_epsilon[pair_idx];
            float const rstar = nb_r_star[pair_idx];
            energy_trial += compute_vdw_energy_f32_(r, eps, rstar);

            // Electrostatic energy
            float const qi = nb_qi[pair_idx];
            float const qj = nb_qj[pair_idx];
            std::int32_t const is_1_4 = nb_is_1_4[pair_idx];
            energy_trial += compute_elec_energy_f32_(r, qi, qj, is_1_4);
        }

        energy_trial = warp_reduce_sum_f32_(energy_trial);

        // Check Armijo condition
        float armijo_threshold = 0.0f;
        if (lane == 0) {
            armijo_threshold = energy_current + c1 * alpha * grad_dot_d;
        }
        armijo_threshold = __shfl_sync(full_mask, armijo_threshold, 0);

        if (energy_trial <= armijo_threshold) {
            return alpha;
        }

        // Reduce step size
        alpha *= rho_decay;
    }

    return alpha;
}

/**
 * L-BFGS optimization kernel (FP32 compute).
 *
 * One warp (32 threads) per molecule.
 * Each warp independently optimizes its assigned molecule.
 *
 * Status codes:
 * 0: Success (gradient threshold reached)
 * 1: Max iterations reached
 * 2: Line search failed
 */
__global__ void mmff94_lbfgs_optimize_kernel_f32(
    float* coord_x, float* coord_y, float* coord_z, std::int32_t const* nb_i, std::int32_t const* nb_j,
    float const* nb_epsilon, float const* nb_r_star, float const* nb_qi, float const* nb_qj,
    std::int32_t const* nb_is_1_4, std::int32_t const* mol_atom_start, std::int32_t const* mol_atom_end,
    std::int32_t const* mol_nb_start, std::int32_t const* mol_nb_end, float* lbfgs_workspace, std::int32_t* out_status,
    std::int32_t* out_iterations, float* out_energy, float* out_rms_grad, std::uint32_t const n_molecules,
    std::int32_t const max_iterations, float const grad_threshold, std::uint32_t const memory_depth,
    std::uint32_t const max_atoms_per_mol) {
    std::uint32_t const mol_idx = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;

    if (mol_idx >= n_molecules) {
        return;
    }

    std::uint32_t const atom_start = static_cast<std::uint32_t>(mol_atom_start[mol_idx]);
    std::uint32_t const atom_end = static_cast<std::uint32_t>(mol_atom_end[mol_idx]);
    std::uint32_t const n_atoms = atom_end - atom_start;
    std::uint32_t const n_elements = n_atoms * 3;

    std::uint32_t const nb_start = static_cast<std::uint32_t>(mol_nb_start[mol_idx]);
    std::uint32_t const nb_end = static_cast<std::uint32_t>(mol_nb_end[mol_idx]);

    // Setup workspace pointers - use max_atoms_per_mol for consistent offset calculation
    std::uint32_t const max_n_elements = max_atoms_per_mol * 3;
    std::uint32_t const workspace_offset = mol_idx * lbfgs_workspace_size_(max_atoms_per_mol, memory_depth);
    float* s_history = lbfgs_workspace + workspace_offset;
    float* y_history = s_history + memory_depth * max_n_elements;
    float* rho = y_history + memory_depth * max_n_elements;
    float* alpha = rho + memory_depth;
    float* gradient_prev = alpha + memory_depth;

    // Temporary arrays (allocated in shared memory - each block gets its own)
    extern __shared__ float shared_mem[];
    float* gradient = shared_mem;
    float* search_dir = gradient + n_elements;
    float* trial_coords = search_dir + n_elements;

    // Initialize gradient arrays to zero
    for (std::uint32_t i = lane; i < n_elements; i += warp_size) {
        gradient[i] = 0.0f;
        gradient_prev[i] = 0.0f;
    }
    __syncwarp(full_mask);

    std::uint32_t k = 0; // History size

    for (std::int32_t iter = 0; iter < max_iterations; ++iter) {
        // Zero out gradient
        for (std::uint32_t i = lane; i < n_elements; i += warp_size) {
            gradient[i] = 0.0f;
        }
        __syncwarp(full_mask);

        // Compute energy and gradient
        float energy = 0.0f;

        for (std::uint32_t pair_idx = nb_start + lane; pair_idx < nb_end; pair_idx += warp_size) {
            std::int32_t const i_global = nb_i[pair_idx];
            std::int32_t const j_global = nb_j[pair_idx];
            std::int32_t const i = i_global - static_cast<std::int32_t>(atom_start);
            std::int32_t const j = j_global - static_cast<std::int32_t>(atom_start);

            float const xi = coord_x[i_global];
            float const yi = coord_y[i_global];
            float const zi = coord_z[i_global];
            float const xj = coord_x[j_global];
            float const yj = coord_y[j_global];
            float const zj = coord_z[j_global];

            float const dx = xj - xi;
            float const dy = yj - yi;
            float const dz = zj - zi;

            float const r = compute_distance_f32_(xi, yi, zi, xj, yj, zj);

            // VdW energy and gradient
            float const eps = nb_epsilon[pair_idx];
            float const rstar = nb_r_star[pair_idx];
            energy += compute_vdw_energy_f32_(r, eps, rstar);

            float const dE_dr_vdw = compute_vdw_gradient_f32_(r, eps, rstar);
            float const inv_r = __frcp_rn(r);
            // Force components (negative gradient): F = -dE/dr * (r_vec / |r_vec|)
            float const fx_vdw = -dE_dr_vdw * dx * inv_r;
            float const fy_vdw = -dE_dr_vdw * dy * inv_r;
            float const fz_vdw = -dE_dr_vdw * dz * inv_r;

            // Accumulate negative gradients (forces) - matches CPU convention
            // grad[i] stores -∂E/∂ri (force on i)
            atomicAdd(&gradient[i], -fx_vdw);
            atomicAdd(&gradient[n_atoms + i], -fy_vdw);
            atomicAdd(&gradient[2 * n_atoms + i], -fz_vdw);
            atomicAdd(&gradient[j], fx_vdw);
            atomicAdd(&gradient[n_atoms + j], fy_vdw);
            atomicAdd(&gradient[2 * n_atoms + j], fz_vdw);

            // Electrostatic energy and gradient
            float const qi = nb_qi[pair_idx];
            float const qj = nb_qj[pair_idx];
            std::int32_t const is_1_4 = nb_is_1_4[pair_idx];
            energy += compute_elec_energy_f32_(r, qi, qj, is_1_4);

            float fx_elec, fy_elec, fz_elec;
            compute_elec_gradient_f32_(r, qi, qj, is_1_4, dx, dy, dz, &fx_elec, &fy_elec, &fz_elec);

            // Accumulate negative gradients (forces) - matches CPU convention
            atomicAdd(&gradient[i], -fx_elec);
            atomicAdd(&gradient[n_atoms + i], -fy_elec);
            atomicAdd(&gradient[2 * n_atoms + i], -fz_elec);
            atomicAdd(&gradient[j], fx_elec);
            atomicAdd(&gradient[n_atoms + j], fy_elec);
            atomicAdd(&gradient[2 * n_atoms + j], fz_elec);
        }

        energy = warp_reduce_sum_f32_(energy);
        __syncwarp(full_mask);

        // Compute gradient RMS
        float const rms_grad = warp_rms_f32_(gradient, n_elements, lane);

        // Check convergence
        if (rms_grad < grad_threshold) {
            if (lane == 0) {
                out_status[mol_idx] = 0;
                out_iterations[mol_idx] = iter;
                out_energy[mol_idx] = energy;
                out_rms_grad[mol_idx] = rms_grad;
            }
            return;
        }

        // Compute search direction using L-BFGS two-loop recursion
        lbfgs_two_loop_recursion_(gradient, search_dir, s_history, y_history, rho, alpha, n_elements, k, memory_depth,
                                  lane);

        // Line search
        float const step_size = lbfgs_line_search_f32_(coord_x, coord_y, coord_z, search_dir, gradient, nb_i, nb_j,
                                                       nb_epsilon, nb_r_star, nb_qi, nb_qj, nb_is_1_4, nb_start, nb_end,
                                                       atom_start, n_atoms, energy, lane, trial_coords);

        // Update positions: x = x + step_size * search_dir
        for (std::uint32_t i = lane; i < n_atoms; i += warp_size) {
            std::uint32_t const atom_idx = atom_start + i;
            coord_x[atom_idx] += step_size * search_dir[i];
            coord_y[atom_idx] += step_size * search_dir[n_atoms + i];
            coord_z[atom_idx] += step_size * search_dir[2 * n_atoms + i];
        }
        __syncwarp(full_mask);

        // Update L-BFGS history
        if (iter > 0) {
            std::uint32_t const hist_idx = k % memory_depth;
            float* s = s_history + hist_idx * n_elements;
            float* y = y_history + hist_idx * n_elements;

            float s_dot_y = 0.0f;

            for (std::uint32_t i = lane; i < n_elements; i += warp_size) {
                s[i] = step_size * search_dir[i];
                y[i] = gradient[i] - gradient_prev[i];
                s_dot_y = fmaf(s[i], y[i], s_dot_y);
            }

            s_dot_y = warp_reduce_sum_f32_(s_dot_y);

            if (lane == 0) {
                rho[hist_idx] = __frcp_rn(s_dot_y);
            }

            k = (k < memory_depth) ? k + 1 : memory_depth;
        }

        // Save current gradient for next iteration
        for (std::uint32_t i = lane; i < n_elements; i += warp_size) {
            gradient_prev[i] = gradient[i];
        }
        __syncwarp(full_mask);
    }

    // Max iterations reached
    float const energy_final = 0.0f; // Could recompute if needed
    float const rms_grad_final = warp_rms_f32_(gradient, n_elements, lane);

    if (lane == 0) {
        out_status[mol_idx] = 1;
        out_iterations[mol_idx] = max_iterations;
        out_energy[mol_idx] = energy_final;
        out_rms_grad[mol_idx] = rms_grad_final;
    }
}

/**
 * L-BFGS optimization kernel (FP16 compute, FP32 accumulation).
 *
 * One warp (32 threads) per molecule.
 * Each warp independently optimizes its assigned molecule.
 */
__global__ void mmff94_lbfgs_optimize_kernel_f16(
    half* coord_x, half* coord_y, half* coord_z, std::int32_t const* nb_i, std::int32_t const* nb_j,
    half const* nb_epsilon, half const* nb_r_star, half const* nb_qi, half const* nb_qj, std::int32_t const* nb_is_1_4,
    std::int32_t const* mol_atom_start, std::int32_t const* mol_atom_end, std::int32_t const* mol_nb_start,
    std::int32_t const* mol_nb_end, float* lbfgs_workspace, std::int32_t* out_status, std::int32_t* out_iterations,
    float* out_energy, float* out_rms_grad, std::uint32_t const n_molecules, std::int32_t const max_iterations,
    float const grad_threshold, std::uint32_t const memory_depth, std::uint32_t const max_atoms_per_mol) {
    std::uint32_t const mol_idx = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;

    if (mol_idx >= n_molecules) {
        return;
    }

    std::uint32_t const atom_start = static_cast<std::uint32_t>(mol_atom_start[mol_idx]);
    std::uint32_t const atom_end = static_cast<std::uint32_t>(mol_atom_end[mol_idx]);
    std::uint32_t const n_atoms = atom_end - atom_start;
    std::uint32_t const n_elements = n_atoms * 3;

    std::uint32_t const nb_start = static_cast<std::uint32_t>(mol_nb_start[mol_idx]);
    std::uint32_t const nb_end = static_cast<std::uint32_t>(mol_nb_end[mol_idx]);

    // Setup workspace pointers - use max_atoms_per_mol for consistent offset calculation
    std::uint32_t const max_n_elements = max_atoms_per_mol * 3;
    std::uint32_t const workspace_offset = mol_idx * lbfgs_workspace_size_(max_atoms_per_mol, memory_depth);
    float* s_history = lbfgs_workspace + workspace_offset;
    float* y_history = s_history + memory_depth * max_n_elements;
    float* rho = y_history + memory_depth * max_n_elements;
    float* alpha = rho + memory_depth;
    float* gradient_prev = alpha + memory_depth;

    // Temporary arrays (allocated in shared memory - each block gets its own)
    extern __shared__ float shared_mem[];
    float* gradient = shared_mem;
    float* search_dir = gradient + n_elements;
    float* trial_coords = search_dir + n_elements;

    // Initialize gradient arrays to zero
    for (std::uint32_t i = lane; i < n_elements; i += warp_size) {
        gradient[i] = 0.0f;
        gradient_prev[i] = 0.0f;
    }
    __syncwarp(full_mask);

    std::uint32_t k = 0; // History size

    for (std::int32_t iter = 0; iter < max_iterations; ++iter) {
        // Zero out gradient
        for (std::uint32_t i = lane; i < n_elements; i += warp_size) {
            gradient[i] = 0.0f;
        }
        __syncwarp(full_mask);

        // Compute energy and gradient
        float energy = 0.0f;

        for (std::uint32_t pair_idx = nb_start + lane; pair_idx < nb_end; pair_idx += warp_size) {
            std::int32_t const i_global = nb_i[pair_idx];
            std::int32_t const j_global = nb_j[pair_idx];
            std::int32_t const i = i_global - static_cast<std::int32_t>(atom_start);
            std::int32_t const j = j_global - static_cast<std::int32_t>(atom_start);

            half const xi = coord_x[i_global];
            half const yi = coord_y[i_global];
            half const zi = coord_z[i_global];
            half const xj = coord_x[j_global];
            half const yj = coord_y[j_global];
            half const zj = coord_z[j_global];

            half const dx = __hsub(xj, xi);
            half const dy = __hsub(yj, yi);
            half const dz = __hsub(zj, zi);

            half const r = compute_distance_f16_(xi, yi, zi, xj, yj, zj);

            // VdW energy and gradient
            half const eps = nb_epsilon[pair_idx];
            half const rstar = nb_r_star[pair_idx];
            energy += compute_vdw_energy_f16_(r, eps, rstar);

            float const dE_dr_vdw = compute_vdw_gradient_f16_(r, eps, rstar);
            half const inv_r = h_rcp_(r);
            float const fx_vdw = -dE_dr_vdw * __half2float(dx) * __half2float(inv_r);
            float const fy_vdw = -dE_dr_vdw * __half2float(dy) * __half2float(inv_r);
            float const fz_vdw = -dE_dr_vdw * __half2float(dz) * __half2float(inv_r);

            atomicAdd(&gradient[i], -fx_vdw);
            atomicAdd(&gradient[n_atoms + i], -fy_vdw);
            atomicAdd(&gradient[2 * n_atoms + i], -fz_vdw);
            atomicAdd(&gradient[j], fx_vdw);
            atomicAdd(&gradient[n_atoms + j], fy_vdw);
            atomicAdd(&gradient[2 * n_atoms + j], fz_vdw);

            // Electrostatic energy and gradient
            half const qi = nb_qi[pair_idx];
            half const qj = nb_qj[pair_idx];
            std::int32_t const is_1_4 = nb_is_1_4[pair_idx];
            energy += compute_elec_energy_f16_(r, qi, qj, is_1_4);

            float fx_elec, fy_elec, fz_elec;
            compute_elec_gradient_f16_(r, qi, qj, is_1_4, dx, dy, dz, &fx_elec, &fy_elec, &fz_elec);

            atomicAdd(&gradient[i], -fx_elec);
            atomicAdd(&gradient[n_atoms + i], -fy_elec);
            atomicAdd(&gradient[2 * n_atoms + i], -fz_elec);
            atomicAdd(&gradient[j], fx_elec);
            atomicAdd(&gradient[n_atoms + j], fy_elec);
            atomicAdd(&gradient[2 * n_atoms + j], fz_elec);
        }

        energy = warp_reduce_sum_f32_(energy);
        __syncwarp(full_mask);

        // Compute gradient RMS
        float const rms_grad = warp_rms_f32_(gradient, n_elements, lane);

        // Check convergence
        if (rms_grad < grad_threshold) {
            if (lane == 0) {
                out_status[mol_idx] = 0;
                out_iterations[mol_idx] = iter;
                out_energy[mol_idx] = energy;
                out_rms_grad[mol_idx] = rms_grad;
            }
            return;
        }

        // Compute search direction using L-BFGS two-loop recursion
        lbfgs_two_loop_recursion_(gradient, search_dir, s_history, y_history, rho, alpha, n_elements, k, memory_depth,
                                  lane);

        // Line search (convert coords to FP32 for line search)
        float* trial_x = search_dir + n_elements; // Reuse shared memory
        float* trial_y = trial_x + n_atoms;
        float* trial_z = trial_y + n_atoms;

        for (std::uint32_t i = lane; i < n_atoms; i += warp_size) {
            std::uint32_t const atom_idx = atom_start + i;
            trial_x[i] = __half2float(coord_x[atom_idx]);
            trial_y[i] = __half2float(coord_y[atom_idx]);
            trial_z[i] = __half2float(coord_z[atom_idx]);
        }
        __syncwarp(full_mask);

        // Simplified line search for FP16 (fixed step)
        float const step_size = 0.01f; // Conservative for FP16

        // Update positions: x = x + step_size * search_dir
        for (std::uint32_t i = lane; i < n_atoms; i += warp_size) {
            std::uint32_t const atom_idx = atom_start + i;
            float const new_x = trial_x[i] + step_size * search_dir[i];
            float const new_y = trial_y[i] + step_size * search_dir[n_atoms + i];
            float const new_z = trial_z[i] + step_size * search_dir[2 * n_atoms + i];
            coord_x[atom_idx] = __float2half(new_x);
            coord_y[atom_idx] = __float2half(new_y);
            coord_z[atom_idx] = __float2half(new_z);
        }
        __syncwarp(full_mask);

        // Update L-BFGS history
        if (iter > 0) {
            std::uint32_t const hist_idx = k % memory_depth;
            float* s = s_history + hist_idx * n_elements;
            float* y = y_history + hist_idx * n_elements;

            float s_dot_y = 0.0f;

            for (std::uint32_t i = lane; i < n_elements; i += warp_size) {
                s[i] = step_size * search_dir[i];
                y[i] = gradient[i] - gradient_prev[i];
                s_dot_y = fmaf(s[i], y[i], s_dot_y);
            }

            s_dot_y = warp_reduce_sum_f32_(s_dot_y);

            if (lane == 0) {
                rho[hist_idx] = __frcp_rn(s_dot_y);
            }

            k = (k < memory_depth) ? k + 1 : memory_depth;
        }

        // Save current gradient for next iteration
        for (std::uint32_t i = lane; i < n_elements; i += warp_size) {
            gradient_prev[i] = gradient[i];
        }
        __syncwarp(full_mask);
    }

    // Max iterations reached
    float const energy_final = 0.0f; // Could recompute if needed
    float const rms_grad_final = warp_rms_f32_(gradient, n_elements, lane);

    if (lane == 0) {
        out_status[mol_idx] = 1;
        out_iterations[mol_idx] = max_iterations;
        out_energy[mol_idx] = energy_final;
        out_rms_grad[mol_idx] = rms_grad_final;
    }
}

#pragma endregion

#pragma region Atom-Centric Kernels

/**
 * Atom-centric MMFF94 energy and gradient kernel (FP32).
 *
 * This kernel eliminates atomics and gathers by using:
 * 1. Register-based coordinate storage (each lane owns one atom)
 * 2. Shuffle-based coordinate access (~5 cycles vs ~100 for gather)
 * 3. Local gradient accumulation (no race conditions)
 * 4. Symmetric adjacency list (each pair stored twice)
 *
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────┐
 * │  REGISTER OWNERSHIP (molecule with N ≤ 32 atoms):          │
 * │                                                            │
 * │  Lane 0 owns atom 0: holds (x0, y0, z0) in registers       │
 * │  Lane 1 owns atom 1: holds (x1, y1, z1) in registers       │
 * │  ...                                                       │
 * │  Lane 31 owns atom 31 (or idle if N < 32)                  │
 * │                                                            │
 * │  Each lane processes atoms: lane, lane+32, lane+64, ...    │
 * │  ► NO CONFLICTS: each lane writes only to its own atoms    │
 * │  ► NO ATOMICS: pure local accumulation                     │
 * │  ► NO SYNC: lanes work independently                       │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Data layout:
 * - coords: [mol0: x0 y0 z0 x1 y1 z1 ...] [mol1: ...] ... (AoS per molecule)
 * - neighbor_offsets: [n_atoms+1] entries per molecule giving neighbor ranges
 * - neighbors: symmetric adjacency list with neighbor_t entries
 *
 * One warp (32 threads) per molecule.
 */
__global__ void mmff94_energy_gradient_atom_centric_f32(
    float* __restrict__ coords,                         // [sum(n_atoms) * 3] AoS: x0,y0,z0,x1,y1,z1,...
    neighbor_t const* __restrict__ neighbors,           // [sum(n_neighbors)] symmetric adjacency list
    std::uint32_t const* __restrict__ neighbor_offsets, // Per-atom neighbor ranges
    mol_meta_t const* __restrict__ metas,               // Per-molecule metadata
    float* __restrict__ grad_out,                       // [sum(n_atoms) * 3] output gradients
    float* __restrict__ energy_out,                     // [n_molecules] output energies
    std::uint32_t const n_molecules) {

    std::uint32_t const mol = blockIdx.x;
    std::uint32_t const lane = threadIdx.x & 31;

    if (mol >= n_molecules)
        return;

    mol_meta_t const meta = metas[mol];
    std::uint32_t const n_atoms = meta.n_atoms;
    std::uint32_t const atom_base = meta.atom_start;
    std::uint32_t const neighbor_base = meta.neighbor_start;

    // Load MY atom's coordinates into registers
    // Each lane loads atom (lane % n_atoms) if lane < n_atoms
    float my_x = 0.0f, my_y = 0.0f, my_z = 0.0f;
    if (lane < n_atoms) {
        std::uint32_t const coord_idx = (atom_base + lane) * 3;
        my_x = coords[coord_idx + 0];
        my_y = coords[coord_idx + 1];
        my_z = coords[coord_idx + 2];
    }

    float total_energy = 0.0f;

    // Each lane processes atoms it owns: lane, lane+32, lane+64, ...
    // For molecules ≤32 atoms, each lane processes at most 1 atom
    for (std::uint32_t atom = lane; atom < n_atoms; atom += warp_size) {
        // Get this atom's coordinates via shuffle
        // (If atom == lane, we use my own values; otherwise shuffle)
        float const ax = __shfl_sync(full_mask, my_x, atom % warp_size);
        float const ay = __shfl_sync(full_mask, my_y, atom % warp_size);
        float const az = __shfl_sync(full_mask, my_z, atom % warp_size);

        float gx = 0.0f, gy = 0.0f, gz = 0.0f;

        // Get this atom's neighbor range
        std::uint32_t const start = neighbor_offsets[atom_base + atom];
        std::uint32_t const end = neighbor_offsets[atom_base + atom + 1];

        // Iterate this atom's neighbors
        for (std::uint32_t n = start; n < end; ++n) {
            neighbor_t const nb = neighbors[neighbor_base + n];

            // Get neighbor coordinates via shuffle
            float const bx = __shfl_sync(full_mask, my_x, nb.j % warp_size);
            float const by = __shfl_sync(full_mask, my_y, nb.j % warp_size);
            float const bz = __shfl_sync(full_mask, my_z, nb.j % warp_size);

            // Compute distance
            float const dx = bx - ax;
            float const dy = by - ay;
            float const dz = bz - az;
            float const r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
            float const r = sqrtf(r2);
            float const r_safe = (r > 0.001f) ? r : 0.001f;

            // ── VdW Energy ──────────────────────────────────────────────
            float const inv_rstar = __frcp_rn(nb.r_star);
            float const rho = r_safe * inv_rstar;

            float const rho2 = rho * rho;
            float const rho4 = rho2 * rho2;
            float const rho6 = rho4 * rho2;
            float const rho7 = rho6 * rho;

            float const rep_denom = rho + 0.07f;
            float const rep_ratio = 1.07f * __frcp_rn(rep_denom);
            float const rep_2 = rep_ratio * rep_ratio;
            float const rep_4 = rep_2 * rep_2;
            float const rep_6 = rep_4 * rep_2;
            float const rep_7 = rep_6 * rep_ratio;

            float const att_denom = rho7 + 0.12f;
            float const att_term = 1.12f * __frcp_rn(att_denom) - 2.0f;

            float const e_vdw = nb.epsilon * rep_7 * att_term;

            // ── VdW Gradient ────────────────────────────────────────────
            float const d_rep = -7.0f * rep_6 * 1.07f / (rep_denom * rep_denom);
            float const d_att = -7.84f * rho6 / (att_denom * att_denom);
            float const dE_drho = nb.epsilon * (d_rep * att_term + rep_7 * d_att);
            float const dE_dr_vdw = dE_drho * inv_rstar;

            // ── Electrostatic Energy ────────────────────────────────────
            float const r_buffered = r_safe + delta_f32;
            float const inv_r_buf = __frcp_rn(r_buffered);
            float const is_1_4 = static_cast<float>(nb.flags & 1);
            float const scale = 1.0f + is_1_4 * scale_1_4_diff_f32;
            float const e_elec = coulomb_const_f32 * nb.q_product * inv_r_buf * scale;

            // ── Electrostatic Gradient ──────────────────────────────────
            float const dE_dr_elec = -coulomb_const_f32 * nb.q_product * __frcp_rn(r_buffered * r_buffered) * scale;

            // ── Combined Force ──────────────────────────────────────────
            float const dE_dr = dE_dr_vdw + dE_dr_elec;
            float const inv_r = __frcp_rn(r_safe);
            float const fx = -dE_dr * dx * inv_r;
            float const fy = -dE_dr * dy * inv_r;
            float const fz = -dE_dr * dz * inv_r;

            // Apply sign for force direction
            // sign=-1: force points TO this atom (we're atom i in original pair)
            // sign=+1: force points AWAY from this atom (we're atom j in original pair)
            float const s = static_cast<float>(nb.sign);
            gx += s * fx;
            gy += s * fy;
            gz += s * fz;

            // Energy: count once per pair (only when this atom < neighbor)
            // Since pairs are symmetric, we only count when atom < nb.j
            if (atom < nb.j) {
                total_energy += e_vdw + e_elec;
            }
        }

        // Write gradient for this atom
        std::uint32_t const out_idx = (atom_base + atom) * 3;
        grad_out[out_idx + 0] = gx;
        grad_out[out_idx + 1] = gy;
        grad_out[out_idx + 2] = gz;
    }

    // Reduce energy across warp
    total_energy = warp_reduce_sum_f32_(total_energy);
    if (lane == 0) {
        energy_out[mol] = total_energy;
    }
}

/**
 * Atom-centric MMFF94 energy-only kernel (FP32).
 *
 * Simplified version without gradient computation for energy evaluation.
 */
__global__ void mmff94_energy_atom_centric_f32(float const* __restrict__ coords,         // [sum(n_atoms) * 3] AoS
                                               neighbor_t const* __restrict__ neighbors, // [sum(n_neighbors)]
                                               std::uint32_t const* __restrict__ neighbor_offsets,
                                               mol_meta_t const* __restrict__ metas, float* __restrict__ energy_out,
                                               std::uint32_t const n_molecules) {

    std::uint32_t const mol = blockIdx.x;
    std::uint32_t const lane = threadIdx.x & 31;

    if (mol >= n_molecules)
        return;

    mol_meta_t const meta = metas[mol];
    std::uint32_t const n_atoms = meta.n_atoms;
    std::uint32_t const atom_base = meta.atom_start;
    std::uint32_t const neighbor_base = meta.neighbor_start;

    // Load coordinates into registers
    float my_x = 0.0f, my_y = 0.0f, my_z = 0.0f;
    if (lane < n_atoms) {
        std::uint32_t const coord_idx = (atom_base + lane) * 3;
        my_x = coords[coord_idx + 0];
        my_y = coords[coord_idx + 1];
        my_z = coords[coord_idx + 2];
    }

    float total_energy = 0.0f;

    for (std::uint32_t atom = lane; atom < n_atoms; atom += warp_size) {
        float const ax = __shfl_sync(full_mask, my_x, atom % warp_size);
        float const ay = __shfl_sync(full_mask, my_y, atom % warp_size);
        float const az = __shfl_sync(full_mask, my_z, atom % warp_size);

        std::uint32_t const start = neighbor_offsets[atom_base + atom];
        std::uint32_t const end = neighbor_offsets[atom_base + atom + 1];

        for (std::uint32_t n = start; n < end; ++n) {
            neighbor_t const nb = neighbors[neighbor_base + n];

            float const bx = __shfl_sync(full_mask, my_x, nb.j % warp_size);
            float const by = __shfl_sync(full_mask, my_y, nb.j % warp_size);
            float const bz = __shfl_sync(full_mask, my_z, nb.j % warp_size);

            float const dx = bx - ax;
            float const dy = by - ay;
            float const dz = bz - az;
            float const r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
            float const r = sqrtf(r2);
            float const r_safe = (r > 0.001f) ? r : 0.001f;

            // VdW energy
            float const inv_rstar = __frcp_rn(nb.r_star);
            float const rho = r_safe * inv_rstar;
            float const rho2 = rho * rho;
            float const rho4 = rho2 * rho2;
            float const rho6 = rho4 * rho2;
            float const rho7 = rho6 * rho;

            float const rep_ratio = 1.07f * __frcp_rn(rho + 0.07f);
            float const rep_7 = rep_ratio * rep_ratio * rep_ratio * rep_ratio * rep_ratio * rep_ratio * rep_ratio;
            float const att_term = 1.12f * __frcp_rn(rho7 + 0.12f) - 2.0f;
            float const e_vdw = nb.epsilon * rep_7 * att_term;

            // Electrostatic energy
            float const r_buffered = r_safe + delta_f32;
            float const is_1_4 = static_cast<float>(nb.flags & 1);
            float const scale = 1.0f + is_1_4 * scale_1_4_diff_f32;
            float const e_elec = coulomb_const_f32 * nb.q_product * __frcp_rn(r_buffered) * scale;

            // Only count energy when atom < neighbor (avoid double counting)
            if (atom < nb.j) {
                total_energy += e_vdw + e_elec;
            }
        }
    }

    total_energy = warp_reduce_sum_f32_(total_energy);
    if (lane == 0) {
        energy_out[mol] = total_energy;
    }
}

#pragma endregion

} // namespace smallfield

#endif // __CUDACC__
