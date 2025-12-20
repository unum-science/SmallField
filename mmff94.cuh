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

constexpr std::size_t warp_size = 32;
constexpr unsigned int full_mask = 0xFFFFFFFF;

constexpr float const coulomb_const_f32 = 332.0716f;
constexpr float const delta_f32 = 0.05f;
constexpr float const scale_1_4_f32 = 0.75f;

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

#pragma region Warp Reduction

/**
 * Butterfly reduction: sum across all 32 warp lanes using XOR shuffle.
 * After reduction, ALL lanes have the total sum.
 *
 * Uses 5 shuffle operations (log2(32) = 5).
 */
__device__ __forceinline__ float warp_reduce_sum_f32_(float val) {
    float sum = val;
    sum += __shfl_xor_sync(full_mask, sum, 16);
    sum += __shfl_xor_sync(full_mask, sum, 8);
    sum += __shfl_xor_sync(full_mask, sum, 4);
    sum += __shfl_xor_sync(full_mask, sum, 2);
    sum += __shfl_xor_sync(full_mask, sum, 1);
    return sum;
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

#pragma endregion

#pragma region VdW Energy Computation

/**
 * MMFF94 Buffered 14-7 van der Waals energy (FP16 compute, FP32 accumulation).
 *
 * E = ε * (1.07*Rstar/(r+0.07*Rstar))^7 * ((1.12*Rstar^7/(r^7+0.12*Rstar^7)) - 2)
 *
 * Computation in FP16, final multiply in FP32 for accumulation precision.
 */
__device__ __forceinline__ float compute_vdw_energy_f16_(
    half const r,
    half const epsilon,
    half const r_star
) {
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

#pragma endregion

#pragma region Electrostatic Energy Computation

/**
 * MMFF94 Buffered Coulomb electrostatic energy (FP16 compute, FP32 accumulation).
 *
 * E = 332.0716 * qi * qj / (r + 0.05)
 *
 * For 1-4 interactions: scale by 0.75.
 */
__device__ __forceinline__ float compute_elec_energy_f16_(half const r, half const qi, half const qj,
                                                          bool const is_1_4) {
    half const coulomb_const = __float2half(332.0716f);
    half const delta = __float2half(0.05f);
    half const scale_1_4 = __float2half(0.75f);

    half const r_buffered = __hadd(r, delta);
    half const inv_r = h_rcp_(r_buffered);

    half e = __hmul(__hmul(__hmul(coulomb_const, qi), qj), inv_r);

    if (is_1_4) {
        e = __hmul(e, scale_1_4);
    }

    return __half2float(e);
}

/**
 * MMFF94 Buffered Coulomb electrostatic energy (FP32 compute and accumulation).
 */
__device__ __forceinline__ float compute_elec_energy_f32_(float const r, float const qi, float const qj,
                                                          bool const is_1_4) {
    float const r_buffered = r + delta_f32;
    float const inv_r = __frcp_rn(r_buffered);

    float e = coulomb_const_f32 * qi * qj * inv_r;

    if (is_1_4) {
        e *= scale_1_4_f32;
    }

    return e;
}

#pragma endregion

#pragma region VdW Energy Kernels

/**
 * MMFF94 van der Waals energy kernel (FP16 compute, FP32 accumulation).
 *
 * One warp (32 threads) per molecule.
 * Thread k processes pairs k, k+32, k+64... via strided iteration.
 * Warp reduction sums energies across lanes.
 * Lane 0 writes final result.
 */
__global__ void mmff94_vdw_energy_kernel_f16(half const* coord_x, half const* coord_y, half const* coord_z,
                                             std::int32_t const* nb_i, std::int32_t const* nb_j, half const* nb_epsilon,
                                             half const* nb_r_star, std::int32_t const* mol_nb_start,
                                             std::int32_t const* mol_nb_end, float* out_vdw,
                                             std::int32_t const n_molecules) {
    std::size_t const mol_idx = blockIdx.x;
    std::size_t const lane = threadIdx.x;

    if (mol_idx >= static_cast<std::size_t>(n_molecules)) {
        return;
    }

    std::int32_t const nb_start = mol_nb_start[mol_idx];
    std::int32_t const nb_end = mol_nb_end[mol_idx];
    float vdw_accum = 0.0f;

    for (std::size_t pair_idx = static_cast<std::size_t>(nb_start) + lane; pair_idx < static_cast<std::size_t>(nb_end);
         pair_idx += warp_size) {
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
                                             float* out_vdw, std::int32_t const n_molecules) {
    std::size_t const mol_idx = blockIdx.x;
    std::size_t const lane = threadIdx.x;

    if (mol_idx >= static_cast<std::size_t>(n_molecules)) {
        return;
    }

    std::int32_t const nb_start = mol_nb_start[mol_idx];
    std::int32_t const nb_end = mol_nb_end[mol_idx];
    float vdw_accum = 0.0f;

    for (std::size_t pair_idx = static_cast<std::size_t>(nb_start) + lane; pair_idx < static_cast<std::size_t>(nb_end);
         pair_idx += warp_size) {
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
 * MMFF94 electrostatic energy kernel (FP16 compute, FP32 accumulation).
 */
__global__ void mmff94_elec_energy_kernel_f16(half const* coord_x, half const* coord_y, half const* coord_z,
                                              std::int32_t const* nb_i, std::int32_t const* nb_j, half const* nb_qi,
                                              half const* nb_qj, std::int32_t const* nb_is_1_4,
                                              std::int32_t const* mol_nb_start, std::int32_t const* mol_nb_end,
                                              float* out_elec, std::int32_t const n_molecules) {
    std::size_t const mol_idx = blockIdx.x;
    std::size_t const lane = threadIdx.x;

    if (mol_idx >= static_cast<std::size_t>(n_molecules)) {
        return;
    }

    std::int32_t const nb_start = mol_nb_start[mol_idx];
    std::int32_t const nb_end = mol_nb_end[mol_idx];
    float elec_accum = 0.0f;

    for (std::size_t pair_idx = static_cast<std::size_t>(nb_start) + lane; pair_idx < static_cast<std::size_t>(nb_end);
         pair_idx += warp_size) {
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
        bool const is_1_4 = (nb_is_1_4[pair_idx] != 0);

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
                                              float* out_elec, std::int32_t const n_molecules) {
    std::size_t const mol_idx = blockIdx.x;
    std::size_t const lane = threadIdx.x;

    if (mol_idx >= static_cast<std::size_t>(n_molecules)) {
        return;
    }

    std::int32_t const nb_start = mol_nb_start[mol_idx];
    std::int32_t const nb_end = mol_nb_end[mol_idx];
    float elec_accum = 0.0f;

    for (std::size_t pair_idx = static_cast<std::size_t>(nb_start) + lane; pair_idx < static_cast<std::size_t>(nb_end);
         pair_idx += warp_size) {
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
        bool const is_1_4 = (nb_is_1_4[pair_idx] != 0);

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
