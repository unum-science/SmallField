/**
 * @file mmff94.hpp
 * @brief MMFF94 force field energy kernels for CPU (serial and AVX-512)
 * @author Ash Vardanian
 *
 * Implements van der Waals (buffered 14-7) and electrostatic energy calculations.
 * All data structures are padded to SIMD width, so no tail handling is needed.
 *
 * Supported backends:
 * - Serial templated (f32/f64)
 * - Skylake: AVX-512 F32
 * - Genoa: AVX-512 BF16
 * - Turin: AVX-512 FP16
 *
 * Usage:
 *     uv run pytest --benchmark-enable bench.py
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace smallfield {

#pragma region Constants

/** Van der Waals delta parameter for buffered 14-7 potential. */
float constexpr vdw_delta = 0.07f;

/** Van der Waals gamma parameter for buffered 14-7 potential. */
float constexpr vdw_gamma = 0.12f;

/** Electrostatic conversion constant (kcal*A/mol/e^2). */
float constexpr elec_constant = 332.0716f;

/** Electrostatic buffer to prevent singularities (Angstroms). */
float constexpr elec_buffer = 0.05f;

/** Default 1-4 scaling factor for electrostatic interactions. */
float constexpr scale_1_4_default = 0.75f;

/** Minimum distance squared to avoid numerical instabilities. */
float constexpr min_distance_squared = 0.01f;

#pragma endregion

#pragma region Helpers

/** Compute x^7 efficiently as x * x^2 * x^4. */
template <typename scalar_t> inline scalar_t pow7_(scalar_t x) {
    scalar_t const x2 = x * x;
    scalar_t const x4 = x2 * x2;
    return x * x2 * x4;
}

#pragma endregion

#pragma region Serial Kernels

/**
 * Compute van der Waals energy using MMFF94 buffered 14-7 potential.
 * Data is assumed to be padded to SIMD width (no tail handling).
 */
template <typename scalar_t>
scalar_t compute_vdw_energy(scalar_t const* coord_x, scalar_t const* coord_y, scalar_t const* coord_z,
                            std::int32_t const* pair_atom_i, std::int32_t const* pair_atom_j,
                            scalar_t const* pair_epsilon, scalar_t const* pair_equilibrium_distance,
                            std::size_t pair_start, std::size_t pair_count) {
    scalar_t constexpr delta = static_cast<scalar_t>(vdw_delta);
    scalar_t constexpr gamma = static_cast<scalar_t>(vdw_gamma);
    scalar_t constexpr one_plus_delta = static_cast<scalar_t>(1.0 + vdw_delta);
    scalar_t constexpr one_plus_gamma = static_cast<scalar_t>(1.0 + vdw_gamma);
    scalar_t constexpr two = static_cast<scalar_t>(2.0);

    scalar_t accum = 0;

    std::size_t const pair_end = pair_start + pair_count;
    for (std::size_t idx = pair_start; idx < pair_end; ++idx) {
        std::int32_t const i = pair_atom_i[idx];
        std::int32_t const j = pair_atom_j[idx];

        scalar_t const dx = coord_x[j] - coord_x[i];
        scalar_t const dy = coord_y[j] - coord_y[i];
        scalar_t const dz = coord_z[j] - coord_z[i];
        scalar_t const distance_sq = dx * dx + dy * dy + dz * dz;
        scalar_t const distance = std::sqrt(distance_sq);

        scalar_t const eps = pair_epsilon[idx];
        scalar_t const eq_dist = pair_equilibrium_distance[idx];

        scalar_t const eq_dist_7 = pow7_(eq_dist);
        scalar_t const dist_7 = pow7_(distance);

        scalar_t const repulsive_denom = distance + delta * eq_dist;
        scalar_t const repulsive_ratio = one_plus_delta * eq_dist / repulsive_denom;
        scalar_t const repulsive_term = pow7_(repulsive_ratio);

        scalar_t const attractive_denom = dist_7 + gamma * eq_dist_7;
        scalar_t const attractive_term = one_plus_gamma * eq_dist_7 / attractive_denom - two;

        accum += eps * repulsive_term * attractive_term;
    }

    return accum;
}

/**
 * Compute electrostatic energy using MMFF94 Coulomb potential with buffer.
 * Data is assumed to be padded to SIMD width (no tail handling).
 */
template <typename scalar_t>
scalar_t compute_electrostatic_energy(scalar_t const* coord_x, scalar_t const* coord_y, scalar_t const* coord_z,
                                      std::int32_t const* pair_atom_i, std::int32_t const* pair_atom_j,
                                      scalar_t const* pair_charge_i, scalar_t const* pair_charge_j,
                                      std::int32_t const* pair_is_1_4, std::size_t pair_start, std::size_t pair_count,
                                      scalar_t scaling_1_4 = static_cast<scalar_t>(scale_1_4_default)) {
    scalar_t constexpr coulomb = static_cast<scalar_t>(elec_constant);
    scalar_t constexpr buffer = static_cast<scalar_t>(elec_buffer);
    scalar_t constexpr one = static_cast<scalar_t>(1.0);

    scalar_t const scale_diff = scaling_1_4 - one;
    scalar_t accum = 0;

    std::size_t const pair_end = pair_start + pair_count;
    for (std::size_t idx = pair_start; idx < pair_end; ++idx) {
        std::int32_t const i = pair_atom_i[idx];
        std::int32_t const j = pair_atom_j[idx];

        scalar_t const qi = pair_charge_i[idx];
        scalar_t const qj = pair_charge_j[idx];

        scalar_t const dx = coord_x[j] - coord_x[i];
        scalar_t const dy = coord_y[j] - coord_y[i];
        scalar_t const dz = coord_z[j] - coord_z[i];
        scalar_t const distance_sq = dx * dx + dy * dy + dz * dz;
        scalar_t const distance = std::sqrt(distance_sq);

        scalar_t const effective_distance = distance + buffer;
        scalar_t const charge_product = qi * qj;

        scalar_t energy = coulomb * charge_product / effective_distance;

        // Branchless 1-4 scaling
        scalar_t const is_1_4_float = (pair_is_1_4[idx] != 0) ? one : static_cast<scalar_t>(0.0);
        scalar_t const scale = one + is_1_4_float * scale_diff;
        energy *= scale;

        accum += energy;
    }

    return accum;
}

#pragma endregion

#pragma region AVX-512 F32 Kernels

#if defined(__AVX512F__)

/** Compute x^7 for 16 floats using AVX-512. */
inline __m512 pow7_(__m512 x) {
    __m512 const x2 = _mm512_mul_ps(x, x);
    __m512 const x4 = _mm512_mul_ps(x2, x2);
    return _mm512_mul_ps(x, _mm512_mul_ps(x2, x4));
}

/**
 * @brief Compute van der Waals energy using AVX-512 with 16-way parallelism.
 *
 * Uses fast reciprocal (rcp14) and reciprocal sqrt (rsqrt14) approximations.
 * Data must be padded to multiple of 16 pairs.
 *
 * @param coord_x X coordinates of all atoms
 * @param coord_y Y coordinates of all atoms
 * @param coord_z Z coordinates of all atoms
 * @param pair_atom_i First atom index for each pair
 * @param pair_atom_j Second atom index for each pair
 * @param pair_epsilon Well depth parameter for each pair
 * @param pair_equilibrium_distance Equilibrium distance for each pair
 * @param pair_start Starting pair index
 * @param pair_count Number of pairs (must be multiple of 16)
 * @return Total van der Waals energy in kcal/mol
 */
inline float compute_vdw_energy_skylake(float const* coord_x, float const* coord_y, float const* coord_z,
                                        std::int32_t const* pair_atom_i, std::int32_t const* pair_atom_j,
                                        float const* pair_epsilon, float const* pair_equilibrium_distance,
                                        std::size_t pair_start, std::size_t pair_count) {
    __m512 const delta_vec = _mm512_set1_ps(vdw_delta);
    __m512 const gamma_vec = _mm512_set1_ps(vdw_gamma);
    __m512 const one_plus_delta_vec = _mm512_set1_ps(1.0f + vdw_delta);
    __m512 const one_plus_gamma_vec = _mm512_set1_ps(1.0f + vdw_gamma);
    __m512 const two_vec = _mm512_set1_ps(2.0f);

    __m512 accum = _mm512_setzero_ps();

    std::size_t const simd_iterations = pair_count / 16;

    for (std::size_t iter = 0; iter < simd_iterations; ++iter) {
        std::size_t const base = pair_start + iter * 16;

        // Load consecutive pair indices
        __m512i const idx_i = _mm512_loadu_si512(&pair_atom_i[base]);
        __m512i const idx_j = _mm512_loadu_si512(&pair_atom_j[base]);

        // Gather scattered coordinates using atom indices
        __m512 const xi = _mm512_i32gather_ps(idx_i, coord_x, 4);
        __m512 const yi = _mm512_i32gather_ps(idx_i, coord_y, 4);
        __m512 const zi = _mm512_i32gather_ps(idx_i, coord_z, 4);
        __m512 const xj = _mm512_i32gather_ps(idx_j, coord_x, 4);
        __m512 const yj = _mm512_i32gather_ps(idx_j, coord_y, 4);
        __m512 const zj = _mm512_i32gather_ps(idx_j, coord_z, 4);

        // Load consecutive pair parameters
        __m512 const eps = _mm512_loadu_ps(&pair_epsilon[base]);
        __m512 const eq_dist = _mm512_loadu_ps(&pair_equilibrium_distance[base]);

        // Displacement and distance
        __m512 const dx = _mm512_sub_ps(xj, xi);
        __m512 const dy = _mm512_sub_ps(yj, yi);
        __m512 const dz = _mm512_sub_ps(zj, zi);
        __m512 const dz_sq = _mm512_mul_ps(dz, dz);
        __m512 const dist_sq = _mm512_fmadd_ps(dx, dx, _mm512_fmadd_ps(dy, dy, dz_sq));
        __m512 const inv_dist = _mm512_rsqrt14_ps(dist_sq);
        __m512 const dist = _mm512_mul_ps(dist_sq, inv_dist);

        // Buffered 14-7 potential
        __m512 const eq_dist_7 = pow7_(eq_dist);
        __m512 const dist_7 = pow7_(dist);

        __m512 const rep_denom = _mm512_fmadd_ps(delta_vec, eq_dist, dist);
        __m512 const rep_ratio = _mm512_mul_ps(one_plus_delta_vec, _mm512_mul_ps(eq_dist, _mm512_rcp14_ps(rep_denom)));
        __m512 const rep_term = pow7_(rep_ratio);

        __m512 const att_denom = _mm512_fmadd_ps(gamma_vec, eq_dist_7, dist_7);
        __m512 const att_term = _mm512_sub_ps(
            _mm512_mul_ps(one_plus_gamma_vec, _mm512_mul_ps(eq_dist_7, _mm512_rcp14_ps(att_denom))), two_vec);

        __m512 const energy = _mm512_mul_ps(eps, _mm512_mul_ps(rep_term, att_term));

        accum = _mm512_add_ps(accum, energy);
    }

    return _mm512_reduce_add_ps(accum);
}

/**
 * @brief Compute electrostatic energy using AVX-512 with 16-way parallelism.
 *
 * Uses branchless 1-4 scaling via arithmetic operations.
 * Data must be padded to multiple of 16 pairs.
 */
inline float compute_electrostatic_energy_skylake(float const* coord_x, float const* coord_y, float const* coord_z,
                                                  std::int32_t const* pair_atom_i, std::int32_t const* pair_atom_j,
                                                  float const* pair_charge_i, float const* pair_charge_j,
                                                  std::int32_t const* pair_is_1_4, std::size_t pair_start,
                                                  std::size_t pair_count, float scaling_1_4 = scale_1_4_default) {
    __m512 const coulomb_vec = _mm512_set1_ps(elec_constant);
    __m512 const buffer_vec = _mm512_set1_ps(elec_buffer);
    __m512 const one_vec = _mm512_set1_ps(1.0f);
    __m512 const scale_1_4_vec = _mm512_set1_ps(scaling_1_4);
    __m512 const scale_diff_vec = _mm512_sub_ps(scale_1_4_vec, one_vec);

    __m512 accum = _mm512_setzero_ps();

    std::size_t const simd_iterations = pair_count / 16;

    for (std::size_t iter = 0; iter < simd_iterations; ++iter) {
        std::size_t const base = pair_start + iter * 16;

        // Load consecutive pair indices
        __m512i const idx_i = _mm512_loadu_si512(&pair_atom_i[base]);
        __m512i const idx_j = _mm512_loadu_si512(&pair_atom_j[base]);

        // Gather scattered coordinates using atom indices
        __m512 const xi = _mm512_i32gather_ps(idx_i, coord_x, 4);
        __m512 const yi = _mm512_i32gather_ps(idx_i, coord_y, 4);
        __m512 const zi = _mm512_i32gather_ps(idx_i, coord_z, 4);
        __m512 const xj = _mm512_i32gather_ps(idx_j, coord_x, 4);
        __m512 const yj = _mm512_i32gather_ps(idx_j, coord_y, 4);
        __m512 const zj = _mm512_i32gather_ps(idx_j, coord_z, 4);

        // Load consecutive pair parameters
        __m512 const qi = _mm512_loadu_ps(&pair_charge_i[base]);
        __m512 const qj = _mm512_loadu_ps(&pair_charge_j[base]);
        __m512i const is_1_4_i32 = _mm512_loadu_si512(&pair_is_1_4[base]);
        __m512 const is_1_4 = _mm512_cvtepi32_ps(is_1_4_i32);

        __m512 const dx = _mm512_sub_ps(xj, xi);
        __m512 const dy = _mm512_sub_ps(yj, yi);
        __m512 const dz = _mm512_sub_ps(zj, zi);
        __m512 const dz_sq = _mm512_mul_ps(dz, dz);
        __m512 const dist_sq = _mm512_fmadd_ps(dx, dx, _mm512_fmadd_ps(dy, dy, dz_sq));
        __m512 const inv_dist = _mm512_rsqrt14_ps(dist_sq);
        __m512 const dist = _mm512_mul_ps(dist_sq, inv_dist);

        __m512 const effective_dist = _mm512_add_ps(dist, buffer_vec);
        __m512 const charge_product = _mm512_mul_ps(qi, qj);
        __m512 energy = _mm512_mul_ps(coulomb_vec, _mm512_mul_ps(charge_product, _mm512_rcp14_ps(effective_dist)));

        // Branchless 1-4 scaling
        __m512 const scale = _mm512_fmadd_ps(is_1_4, scale_diff_vec, one_vec);
        energy = _mm512_mul_ps(energy, scale);

        accum = _mm512_add_ps(accum, energy);
    }

    return _mm512_reduce_add_ps(accum);
}

#endif // __AVX512F__

#pragma endregion

#pragma region AVX-512 BF16 Kernels

#if defined(__AVX512BF16__)

/** Convert 16 BF16 values to F32. */
inline __m512 bf16_to_f32_(std::uint16_t const* bf16_data) {
    // Load raw bytes as __m256i, then reinterpret as __m256bh for cvtpbh
    __m256i const bf16_vec = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(bf16_data));
    return _mm512_cvtpbh_ps(*reinterpret_cast<__m256bh const*>(&bf16_vec));
}

/**
 * @brief Compute van der Waals energy with BF16 storage and F32 compute.
 *
 * Saves 2x memory bandwidth while maintaining F32 compute precision.
 */
inline float compute_vdw_energy_genoa(std::uint16_t const* coord_x, std::uint16_t const* coord_y,
                                      std::uint16_t const* coord_z, std::int32_t const* pair_atom_i,
                                      std::int32_t const* pair_atom_j, std::uint16_t const* pair_epsilon,
                                      std::uint16_t const* pair_equilibrium_distance, std::size_t pair_start,
                                      std::size_t pair_count) {
    __m512 const delta_vec = _mm512_set1_ps(vdw_delta);
    __m512 const gamma_vec = _mm512_set1_ps(vdw_gamma);
    __m512 const one_plus_delta_vec = _mm512_set1_ps(1.0f + vdw_delta);
    __m512 const one_plus_gamma_vec = _mm512_set1_ps(1.0f + vdw_gamma);
    __m512 const two_vec = _mm512_set1_ps(2.0f);

    __m512 accum = _mm512_setzero_ps();

    std::size_t const simd_iterations = pair_count / 16;

    for (std::size_t iter = 0; iter < simd_iterations; ++iter) {
        std::size_t const base = pair_start + iter * 16;

        alignas(64) std::uint16_t xi_bf16[16], yi_bf16[16], zi_bf16[16];
        alignas(64) std::uint16_t xj_bf16[16], yj_bf16[16], zj_bf16[16];
        alignas(64) std::uint16_t eps_bf16[16], eq_dist_bf16[16];

        for (std::size_t k = 0; k < 16; ++k) {
            std::size_t const idx = base + k;
            std::int32_t const i = pair_atom_i[idx];
            std::int32_t const j = pair_atom_j[idx];
            xi_bf16[k] = coord_x[i];
            yi_bf16[k] = coord_y[i];
            zi_bf16[k] = coord_z[i];
            xj_bf16[k] = coord_x[j];
            yj_bf16[k] = coord_y[j];
            zj_bf16[k] = coord_z[j];
            eps_bf16[k] = pair_epsilon[idx];
            eq_dist_bf16[k] = pair_equilibrium_distance[idx];
        }

        __m512 const xi = bf16_to_f32_(xi_bf16);
        __m512 const yi = bf16_to_f32_(yi_bf16);
        __m512 const zi = bf16_to_f32_(zi_bf16);
        __m512 const xj = bf16_to_f32_(xj_bf16);
        __m512 const yj = bf16_to_f32_(yj_bf16);
        __m512 const zj = bf16_to_f32_(zj_bf16);
        __m512 const eps = bf16_to_f32_(eps_bf16);
        __m512 const eq_dist = bf16_to_f32_(eq_dist_bf16);

        __m512 const dx = _mm512_sub_ps(xj, xi);
        __m512 const dy = _mm512_sub_ps(yj, yi);
        __m512 const dz = _mm512_sub_ps(zj, zi);
        __m512 const dz_sq = _mm512_mul_ps(dz, dz);
        __m512 const dist_sq = _mm512_fmadd_ps(dx, dx, _mm512_fmadd_ps(dy, dy, dz_sq));
        __m512 const inv_dist = _mm512_rsqrt14_ps(dist_sq);
        __m512 const dist = _mm512_mul_ps(dist_sq, inv_dist);

        __m512 const eq_dist_7 = pow7_(eq_dist);
        __m512 const dist_7 = pow7_(dist);

        __m512 const rep_denom = _mm512_fmadd_ps(delta_vec, eq_dist, dist);
        __m512 const rep_ratio = _mm512_mul_ps(one_plus_delta_vec, _mm512_mul_ps(eq_dist, _mm512_rcp14_ps(rep_denom)));
        __m512 const rep_term = pow7_(rep_ratio);

        __m512 const att_denom = _mm512_fmadd_ps(gamma_vec, eq_dist_7, dist_7);
        __m512 const att_term = _mm512_sub_ps(
            _mm512_mul_ps(one_plus_gamma_vec, _mm512_mul_ps(eq_dist_7, _mm512_rcp14_ps(att_denom))), two_vec);

        __m512 const energy = _mm512_mul_ps(eps, _mm512_mul_ps(rep_term, att_term));

        accum = _mm512_add_ps(accum, energy);
    }

    return _mm512_reduce_add_ps(accum);
}

#endif // __AVX512BF16__

#pragma endregion

#pragma region AVX-512 FP16 Kernels

#if defined(__AVX512FP16__)

/**
 * @brief Compute van der Waals energy with FP16 storage and mixed precision.
 *
 * Distances computed in native FP16 (32-wide), then widened to F32 for pow7.
 */
inline float compute_vdw_energy_turin(_Float16 const* coord_x, _Float16 const* coord_y, _Float16 const* coord_z,
                                      std::int32_t const* pair_atom_i, std::int32_t const* pair_atom_j,
                                      _Float16 const* pair_epsilon, _Float16 const* pair_equilibrium_distance,
                                      std::size_t pair_start, std::size_t pair_count) {
    __m512 const delta_vec = _mm512_set1_ps(vdw_delta);
    __m512 const gamma_vec = _mm512_set1_ps(vdw_gamma);
    __m512 const one_plus_delta_vec = _mm512_set1_ps(1.0f + vdw_delta);
    __m512 const one_plus_gamma_vec = _mm512_set1_ps(1.0f + vdw_gamma);
    __m512 const two_vec = _mm512_set1_ps(2.0f);

    __m512 accum = _mm512_setzero_ps();

    // Process 32 pairs at a time (native FP16 width)
    std::size_t const simd_iterations = pair_count / 32;

    for (std::size_t iter = 0; iter < simd_iterations; ++iter) {
        std::size_t const base = pair_start + iter * 32;

        alignas(64) _Float16 xi_f16[32], yi_f16[32], zi_f16[32];
        alignas(64) _Float16 xj_f16[32], yj_f16[32], zj_f16[32];
        alignas(64) _Float16 eps_f16[32], eq_dist_f16[32];

        for (std::size_t k = 0; k < 32; ++k) {
            std::size_t const idx = base + k;
            std::int32_t const i = pair_atom_i[idx];
            std::int32_t const j = pair_atom_j[idx];
            xi_f16[k] = coord_x[i];
            yi_f16[k] = coord_y[i];
            zi_f16[k] = coord_z[i];
            xj_f16[k] = coord_x[j];
            yj_f16[k] = coord_y[j];
            zj_f16[k] = coord_z[j];
            eps_f16[k] = pair_epsilon[idx];
            eq_dist_f16[k] = pair_equilibrium_distance[idx];
        }

        // Native FP16 distance computation (32-wide)
        __m512h const xi_h = _mm512_loadu_ph(xi_f16);
        __m512h const yi_h = _mm512_loadu_ph(yi_f16);
        __m512h const zi_h = _mm512_loadu_ph(zi_f16);
        __m512h const xj_h = _mm512_loadu_ph(xj_f16);
        __m512h const yj_h = _mm512_loadu_ph(yj_f16);
        __m512h const zj_h = _mm512_loadu_ph(zj_f16);

        __m512h const dx_h = _mm512_sub_ph(xj_h, xi_h);
        __m512h const dy_h = _mm512_sub_ph(yj_h, yi_h);
        __m512h const dz_h = _mm512_sub_ph(zj_h, zi_h);
        __m512h const dz_sq_h = _mm512_mul_ph(dz_h, dz_h);
        __m512h const dist_sq_h = _mm512_fmadd_ph(dx_h, dx_h, _mm512_fmadd_ph(dy_h, dy_h, dz_sq_h));
        __m512h const inv_dist_h = _mm512_rsqrt_ph(dist_sq_h);
        __m512h const dist_h = _mm512_mul_ph(dist_sq_h, inv_dist_h);

        // Widen to F32 for pow7 computation (split 32 -> 2x16)
        // Cast through integer types for compatibility with _mm512_cvtph_ps
        __m512i const dist_i = _mm512_castph_si512(dist_h);
        __m256i const dist_lo_i = _mm512_castsi512_si256(dist_i);
        __m256i const dist_hi_i = _mm512_extracti64x4_epi64(dist_i, 1);
        __m512 const dist_lo = _mm512_cvtph_ps(dist_lo_i);
        __m512 const dist_hi = _mm512_cvtph_ps(dist_hi_i);

        __m512h const eps_h = _mm512_loadu_ph(eps_f16);
        __m512h const eq_dist_h = _mm512_loadu_ph(eq_dist_f16);
        __m512i const eps_i = _mm512_castph_si512(eps_h);
        __m512i const eq_dist_i = _mm512_castph_si512(eq_dist_h);
        __m256i const eps_lo_i = _mm512_castsi512_si256(eps_i);
        __m256i const eps_hi_i = _mm512_extracti64x4_epi64(eps_i, 1);
        __m256i const eq_dist_lo_i = _mm512_castsi512_si256(eq_dist_i);
        __m256i const eq_dist_hi_i = _mm512_extracti64x4_epi64(eq_dist_i, 1);
        __m512 const eps_lo = _mm512_cvtph_ps(eps_lo_i);
        __m512 const eps_hi = _mm512_cvtph_ps(eps_hi_i);
        __m512 const eq_dist_lo = _mm512_cvtph_ps(eq_dist_lo_i);
        __m512 const eq_dist_hi = _mm512_cvtph_ps(eq_dist_hi_i);

        // Compute energy for lower half
        __m512 const eq_dist_7_lo = pow7_(eq_dist_lo);
        __m512 const dist_7_lo = pow7_(dist_lo);
        __m512 const rep_denom_lo = _mm512_fmadd_ps(delta_vec, eq_dist_lo, dist_lo);
        __m512 const rep_ratio_lo =
            _mm512_mul_ps(one_plus_delta_vec, _mm512_mul_ps(eq_dist_lo, _mm512_rcp14_ps(rep_denom_lo)));
        __m512 const rep_term_lo = pow7_(rep_ratio_lo);
        __m512 const att_denom_lo = _mm512_fmadd_ps(gamma_vec, eq_dist_7_lo, dist_7_lo);
        __m512 const att_term_lo = _mm512_sub_ps(
            _mm512_mul_ps(one_plus_gamma_vec, _mm512_mul_ps(eq_dist_7_lo, _mm512_rcp14_ps(att_denom_lo))), two_vec);
        __m512 const energy_lo = _mm512_mul_ps(eps_lo, _mm512_mul_ps(rep_term_lo, att_term_lo));

        // Compute energy for upper half
        __m512 const eq_dist_7_hi = pow7_(eq_dist_hi);
        __m512 const dist_7_hi = pow7_(dist_hi);
        __m512 const rep_denom_hi = _mm512_fmadd_ps(delta_vec, eq_dist_hi, dist_hi);
        __m512 const rep_ratio_hi =
            _mm512_mul_ps(one_plus_delta_vec, _mm512_mul_ps(eq_dist_hi, _mm512_rcp14_ps(rep_denom_hi)));
        __m512 const rep_term_hi = pow7_(rep_ratio_hi);
        __m512 const att_denom_hi = _mm512_fmadd_ps(gamma_vec, eq_dist_7_hi, dist_7_hi);
        __m512 const att_term_hi = _mm512_sub_ps(
            _mm512_mul_ps(one_plus_gamma_vec, _mm512_mul_ps(eq_dist_7_hi, _mm512_rcp14_ps(att_denom_hi))), two_vec);
        __m512 const energy_hi = _mm512_mul_ps(eps_hi, _mm512_mul_ps(rep_term_hi, att_term_hi));

        accum = _mm512_add_ps(accum, _mm512_add_ps(energy_lo, energy_hi));
    }

    return _mm512_reduce_add_ps(accum);
}

#endif // __AVX512FP16__

#pragma endregion

} // namespace smallfield
