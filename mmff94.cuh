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

} // namespace smallfield

#endif // __CUDACC__
