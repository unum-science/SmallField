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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace smallfield {

#pragma region Constants

// =============================================================================
// Van der Waals Constants
// =============================================================================

/** Van der Waals delta parameter for buffered 14-7 potential. */
float constexpr vdw_delta = 0.07f;

/** Van der Waals gamma parameter for buffered 14-7 potential. */
float constexpr vdw_gamma = 0.12f;

// =============================================================================
// Electrostatic Constants
// =============================================================================

/** Electrostatic conversion constant (kcal*A/mol/e^2). */
float constexpr elec_constant = 332.0716f;

/** Electrostatic buffer to prevent singularities (Angstroms). */
float constexpr elec_buffer = 0.05f;

/** Default 1-4 scaling factor for electrostatic interactions. */
float constexpr scale_1_4_default = 0.75f;

// =============================================================================
// Bond Stretching Constants (Quartic Potential)
// =============================================================================

/** Bond prefactor: converts mdyn*A to kcal/mol. */
float constexpr bond_prefactor = 143.9325f;

/** Cubic stretch constant (A^-1). */
float constexpr bond_cubic_stretch = -2.0f;

// =============================================================================
// Angle Bending Constants (Cubic Polynomial)
// =============================================================================

/** Angle prefactor: 143.9325 * (pi/180)^2 = 0.043844 (mdyn*A/rad^2 -> kcal/mol/deg^2). */
float constexpr angle_prefactor = 0.043844f;

/** Cubic bend constant (deg^-1). */
float constexpr angle_cubic_bend = -0.006981317f;

// =============================================================================
// Torsion Constants (Cosine Series)
// =============================================================================

/** Torsion prefactor. */
float constexpr torsion_prefactor = 0.5f;

// =============================================================================
// Stretch-Bend Coupling Constants
// =============================================================================

/** Stretch-bend prefactor: 143.9325 * (pi/180) = 2.51210 */
float constexpr stretch_bend_prefactor = 2.51210f;

// =============================================================================
// Out-of-Plane Bending Constants
// =============================================================================

/** OOP prefactor (same as angle prefactor). */
float constexpr oop_prefactor = 0.043844f;

// =============================================================================
// Numerical Constants
// =============================================================================

/** Minimum distance squared to avoid numerical instabilities. */
float constexpr min_distance_squared = 0.01f;

/** Pi constant. */
double constexpr pi = 3.14159265358979323846;

/** Degrees to radians conversion. */
double constexpr deg_to_rad = pi / 180.0;

/** Radians to degrees conversion. */
double constexpr rad_to_deg = 180.0 / pi;

#pragma endregion

#pragma region Helpers

/** Compute x^7 efficiently as x * x^2 * x^4. */
template <typename scalar_t> inline scalar_t pow7_(scalar_t x) {
    scalar_t const x2 = x * x;
    scalar_t const x4 = x2 * x2;
    return x * x2 * x4;
}

/** Clamp value to [lo, hi] range. */
template <typename scalar_t> inline scalar_t clamp_(scalar_t x, scalar_t lo, scalar_t hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

template <typename scalar_t> struct vec3_t {
    scalar_t x;
    scalar_t y;
    scalar_t z;
};

template <typename scalar_t> inline vec3_t<scalar_t> make_vec3_(scalar_t x, scalar_t y, scalar_t z) {
    return {x, y, z};
}

template <typename scalar_t> inline vec3_t<scalar_t> sub_(vec3_t<scalar_t> const& a, vec3_t<scalar_t> const& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template <typename scalar_t> inline vec3_t<scalar_t> add_(vec3_t<scalar_t> const& a, vec3_t<scalar_t> const& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template <typename scalar_t> inline vec3_t<scalar_t> scale_(vec3_t<scalar_t> const& a, scalar_t s) {
    return {a.x * s, a.y * s, a.z * s};
}

template <typename scalar_t> inline scalar_t dot_(vec3_t<scalar_t> const& a, vec3_t<scalar_t> const& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename scalar_t> inline vec3_t<scalar_t> cross_(vec3_t<scalar_t> const& a, vec3_t<scalar_t> const& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

template <typename scalar_t> inline scalar_t norm_(vec3_t<scalar_t> const& a) { return std::sqrt(dot_(a, a)); }

#pragma endregion

#pragma region L-BFGS Data Structures

/**
 * @brief State container for L-BFGS optimizer.
 *
 * Stores history vectors (s, y), working buffers, and iteration state.
 * Memory layout: all arrays are flat and contiguous for cache efficiency.
 */
template <typename scalar_t> struct lbfgs_state_t {
    scalar_t* s_history;    // [m * 3 * n_atoms] position differences
    scalar_t* y_history;    // [m * 3 * n_atoms] gradient differences
    scalar_t* alpha;        // [m] two-loop recursion coefficients
    scalar_t* rho;          // [m] inverse dot products 1/(y·s)
    scalar_t* q;            // [3 * n_atoms] work buffer for two-loop
    scalar_t* grad;         // [3 * n_atoms] current gradient
    scalar_t* grad_prev;    // [3 * n_atoms] previous gradient
    scalar_t* direction;    // [3 * n_atoms] search direction
    scalar_t* coord_backup; // [3 * n_atoms] line search backup
    std::size_t n_atoms;    // number of atoms
    std::size_t m;          // history size (default 10)
    std::size_t k;          // iteration count
};

/**
 * @brief Result of optimization run.
 */
struct optimize_result_t {
    int status;          // 0=converged, 1=max_iter, -1=error
    int iterations;      // number of iterations performed
    double final_energy; // final energy value
    double rms_gradient; // RMS gradient magnitude
};

#pragma endregion

#pragma region Precomputed VdW Pair Constants

/**
 * @brief Precomputed VdW pair constants for eliminating redundant calculations.
 *
 * These values are constant for each pair and can be computed once at setup.
 * Eliminates ~8 multiplications per 16 pairs in every energy/gradient call.
 */
template <typename scalar_t> struct vdw_precomputed_t {
    scalar_t* eq_dist_7;                // pow7(equilibrium_distance)
    scalar_t* delta_eq_dist;            // delta * equilibrium_distance
    scalar_t* one_plus_delta_eq_dist;   // (1 + delta) * equilibrium_distance
    scalar_t* gamma_eq_dist_7;          // gamma * eq_dist_7
    scalar_t* one_plus_gamma_eq_dist_7; // (1 + gamma) * eq_dist_7
};

/**
 * @brief Precompute VdW pair constants from epsilon and equilibrium distance arrays.
 *
 * Call once at setup before optimization. The output arrays must be pre-allocated
 * with size >= n_pairs.
 */
template <typename scalar_t>
void precompute_vdw_constants(scalar_t const* pair_equilibrium_distance, std::size_t n_pairs,
                              vdw_precomputed_t<scalar_t>& out) {
    scalar_t constexpr delta = static_cast<scalar_t>(vdw_delta);
    scalar_t constexpr gamma = static_cast<scalar_t>(vdw_gamma);
    scalar_t constexpr one_plus_delta = static_cast<scalar_t>(1.0 + vdw_delta);
    scalar_t constexpr one_plus_gamma = static_cast<scalar_t>(1.0 + vdw_gamma);

    for (std::size_t i = 0; i < n_pairs; ++i) {
        scalar_t const eq_dist = pair_equilibrium_distance[i];
        scalar_t const eq_dist_7 = pow7_(eq_dist);

        out.eq_dist_7[i] = eq_dist_7;
        out.delta_eq_dist[i] = delta * eq_dist;
        out.one_plus_delta_eq_dist[i] = one_plus_delta * eq_dist;
        out.gamma_eq_dist_7[i] = gamma * eq_dist_7;
        out.one_plus_gamma_eq_dist_7[i] = one_plus_gamma * eq_dist_7;
    }
}

#pragma endregion

#pragma region MMFF94 Force Field Parameters

/**
 * @brief All MMFF94 force field parameters for a molecule.
 *
 * Contains arrays for all energy terms: VdW, electrostatic, bond, angle,
 * torsion, stretch-bend, and out-of-plane bending.
 */
template <typename scalar_t> struct mmff94_params_t {
    // VdW pairs (buffered 14-7)
    std::int32_t const* vdw_pair_i;
    std::int32_t const* vdw_pair_j;
    scalar_t const* vdw_epsilon;
    scalar_t const* vdw_r_star;
    std::size_t vdw_count;

    // Electrostatic pairs
    std::int32_t const* elec_pair_i;
    std::int32_t const* elec_pair_j;
    scalar_t const* elec_charge_i;
    scalar_t const* elec_charge_j;
    std::int32_t const* elec_is_1_4;
    std::size_t elec_count;

    // Bonds (quartic stretching)
    std::int32_t const* bond_atom_i;
    std::int32_t const* bond_atom_j;
    scalar_t const* bond_kb;
    scalar_t const* bond_r0;
    std::size_t bond_count;

    // Angles (cubic bending)
    std::int32_t const* angle_atom_i;
    std::int32_t const* angle_atom_j;
    std::int32_t const* angle_atom_k;
    scalar_t const* angle_ka;
    scalar_t const* angle_theta0;
    std::size_t angle_count;

    // Torsions (3-term Fourier)
    std::int32_t const* torsion_atom_i;
    std::int32_t const* torsion_atom_j;
    std::int32_t const* torsion_atom_k;
    std::int32_t const* torsion_atom_l;
    scalar_t const* torsion_v1;
    scalar_t const* torsion_v2;
    scalar_t const* torsion_v3;
    std::size_t torsion_count;

    // Stretch-bend coupling
    std::int32_t const* stbn_atom_i;
    std::int32_t const* stbn_atom_j;
    std::int32_t const* stbn_atom_k;
    scalar_t const* stbn_kba_ijk;
    scalar_t const* stbn_kba_kji;
    scalar_t const* stbn_r0_ij;
    scalar_t const* stbn_r0_kj;
    scalar_t const* stbn_theta0;
    std::size_t stbn_count;

    // Out-of-plane bending (Wilson angle)
    std::int32_t const* oop_atom_i;
    std::int32_t const* oop_atom_j;
    std::int32_t const* oop_atom_k;
    std::int32_t const* oop_atom_l;
    scalar_t const* oop_koop;
    std::size_t oop_count;
};

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
 * Compute van der Waals gradient using MMFF94 buffered 14-7 potential.
 * Accumulates forces into grad_x, grad_y, grad_z (caller must zero arrays first).
 */
template <typename scalar_t>
void compute_vdw_gradient(scalar_t const* coord_x, scalar_t const* coord_y, scalar_t const* coord_z,
                          std::int32_t const* pair_atom_i, std::int32_t const* pair_atom_j,
                          scalar_t const* pair_epsilon, scalar_t const* pair_equilibrium_distance,
                          std::size_t pair_start, std::size_t pair_count, scalar_t* grad_x, scalar_t* grad_y,
                          scalar_t* grad_z) {
    scalar_t constexpr delta = static_cast<scalar_t>(vdw_delta);
    scalar_t constexpr gamma = static_cast<scalar_t>(vdw_gamma);
    scalar_t constexpr one_plus_delta = static_cast<scalar_t>(1.0 + vdw_delta);
    scalar_t constexpr one_plus_gamma = static_cast<scalar_t>(1.0 + vdw_gamma);
    scalar_t constexpr two = static_cast<scalar_t>(2.0);
    scalar_t constexpr seven = static_cast<scalar_t>(7.0);

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

        // Repulsive term: (1.07*R/(r+0.07*R))^7
        scalar_t const repulsive_denom = distance + delta * eq_dist;
        scalar_t const repulsive_ratio = one_plus_delta * eq_dist / repulsive_denom;
        scalar_t const repulsive_term = pow7_(repulsive_ratio);

        // Attractive term: (1.12*R^7/(r^7+0.12*R^7)) - 2
        scalar_t const attractive_denom = dist_7 + gamma * eq_dist_7;
        scalar_t const attractive_term = one_plus_gamma * eq_dist_7 / attractive_denom - two;

        // Gradient of repulsive: 7*u^6 * (-1.07*R/(r+0.07*R)^2)
        scalar_t const repulsive_6 = repulsive_term / repulsive_ratio;
        scalar_t const d_repulsive =
            -seven * repulsive_6 * one_plus_delta * eq_dist / (repulsive_denom * repulsive_denom);

        // Gradient of attractive: -1.12*R^7 * 7*r^6 / (r^7+0.12*R^7)^2
        scalar_t const dist_6 = dist_7 / distance;
        scalar_t const d_attractive =
            -one_plus_gamma * eq_dist_7 * seven * dist_6 / (attractive_denom * attractive_denom);

        // Total gradient: dE/dr = ε * (d_repulsive * attractive + repulsive * d_attractive)
        scalar_t const dE_dr = eps * (d_repulsive * attractive_term + repulsive_term * d_attractive);

        // Gradient contribution: dE/dx_i = dE/dr * dr/dx_i = dE/dr * (-dx/r)
        // and dE/dx_j = dE/dr * dr/dx_j = dE/dr * (+dx/r)
        scalar_t const inv_distance = static_cast<scalar_t>(1.0) / distance;
        scalar_t const dE_dx = dE_dr * dx * inv_distance;
        scalar_t const dE_dy = dE_dr * dy * inv_distance;
        scalar_t const dE_dz = dE_dr * dz * inv_distance;

        // dr/dx_i = -dx/r, dr/dx_j = +dx/r
        grad_x[i] -= dE_dx;
        grad_y[i] -= dE_dy;
        grad_z[i] -= dE_dz;
        grad_x[j] += dE_dx;
        grad_y[j] += dE_dy;
        grad_z[j] += dE_dz;
    }
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

/**
 * Compute electrostatic gradient using MMFF94 Coulomb potential with buffer.
 * Accumulates forces into grad_x/y/z arrays (does not zero them).
 * Data is assumed to be padded to SIMD width (no tail handling).
 */
template <typename scalar_t>
void compute_electrostatic_gradient(scalar_t const* coord_x, scalar_t const* coord_y, scalar_t const* coord_z,
                                    std::int32_t const* pair_atom_i, std::int32_t const* pair_atom_j,
                                    scalar_t const* pair_charge_i, scalar_t const* pair_charge_j,
                                    std::int32_t const* pair_is_1_4, std::size_t pair_start, std::size_t pair_count,
                                    scalar_t scaling_1_4, scalar_t* grad_x, scalar_t* grad_y, scalar_t* grad_z) {
    scalar_t constexpr coulomb = static_cast<scalar_t>(elec_constant);
    scalar_t constexpr buffer = static_cast<scalar_t>(elec_buffer);
    scalar_t constexpr one = static_cast<scalar_t>(1.0);

    scalar_t const scale_diff = scaling_1_4 - one;

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

        // dE/dr = -332.0716 * qi * qj / (r + 0.05)^2
        scalar_t const effective_distance_sq = effective_distance * effective_distance;
        scalar_t const dE_dr = -coulomb * charge_product / effective_distance_sq;

        // Branchless 1-4 scaling
        scalar_t const is_1_4_float = (pair_is_1_4[idx] != 0) ? one : static_cast<scalar_t>(0.0);
        scalar_t const scale = one + is_1_4_float * scale_diff;
        scalar_t const scaled_dE_dr = dE_dr * scale;

        // Gradient contribution: dE/dx_i = dE/dr * dr/dx_i = dE/dr * (-dx/r)
        // and dE/dx_j = dE/dr * dr/dx_j = dE/dr * (+dx/r)
        scalar_t const inv_distance = one / distance;
        scalar_t const dE_dx = scaled_dE_dr * dx * inv_distance;
        scalar_t const dE_dy = scaled_dE_dr * dy * inv_distance;
        scalar_t const dE_dz = scaled_dE_dr * dz * inv_distance;

        // dr/dx_i = -dx/r, dr/dx_j = +dx/r
        grad_x[i] -= dE_dx;
        grad_y[i] -= dE_dy;
        grad_z[i] -= dE_dz;
        grad_x[j] += dE_dx;
        grad_y[j] += dE_dy;
        grad_z[j] += dE_dz;
    }
}

// =============================================================================
// Bond Stretching Energy and Gradient (Quartic Potential)
// =============================================================================

/**
 * Compute bond stretching energy using MMFF94 quartic potential.
 * E = 0.5 * 143.9325 * kb * dr^2 * (1 + cs*dr + 7/12*cs^2*dr^2)
 */
template <typename scalar_t>
scalar_t compute_bond_energy(scalar_t const* coord_x, scalar_t const* coord_y, scalar_t const* coord_z,
                             std::int32_t const* bond_atom_i, std::int32_t const* bond_atom_j, scalar_t const* bond_kb,
                             scalar_t const* bond_r0, std::size_t bond_start, std::size_t bond_count) {
    scalar_t constexpr prefactor = static_cast<scalar_t>(bond_prefactor);
    scalar_t constexpr cs = static_cast<scalar_t>(bond_cubic_stretch);
    scalar_t constexpr cs2 = cs * cs;
    scalar_t constexpr seven_twelfths = static_cast<scalar_t>(7.0 / 12.0);

    scalar_t accum = 0;

    std::size_t const bond_end = bond_start + bond_count;
    for (std::size_t idx = bond_start; idx < bond_end; ++idx) {
        std::int32_t const i = bond_atom_i[idx];
        std::int32_t const j = bond_atom_j[idx];

        scalar_t const dx = coord_x[j] - coord_x[i];
        scalar_t const dy = coord_y[j] - coord_y[i];
        scalar_t const dz = coord_z[j] - coord_z[i];
        scalar_t const r = std::sqrt(dx * dx + dy * dy + dz * dz);

        scalar_t const kb = bond_kb[idx];
        scalar_t const r0 = bond_r0[idx];
        scalar_t const dr = r - r0;
        scalar_t const dr2 = dr * dr;

        // Quartic potential: 0.5 * 143.9325 * kb * dr^2 * (1 + cs*dr + 7/12*cs^2*dr^2)
        scalar_t const term = static_cast<scalar_t>(1.0) + cs * dr + seven_twelfths * cs2 * dr2;
        accum += static_cast<scalar_t>(0.5) * prefactor * kb * dr2 * term;
    }

    return accum;
}

/**
 * Compute bond stretching gradient using MMFF94 quartic potential.
 * dE/dr = 143.9325 * kb * dr * (1 + 1.5*cs*dr + 7/6*cs^2*dr^2)
 */
template <typename scalar_t>
void compute_bond_gradient(scalar_t const* coord_x, scalar_t const* coord_y, scalar_t const* coord_z,
                           std::int32_t const* bond_atom_i, std::int32_t const* bond_atom_j, scalar_t const* bond_kb,
                           scalar_t const* bond_r0, std::size_t bond_start, std::size_t bond_count, scalar_t* grad_x,
                           scalar_t* grad_y, scalar_t* grad_z) {
    scalar_t constexpr prefactor = static_cast<scalar_t>(bond_prefactor);
    scalar_t constexpr cs = static_cast<scalar_t>(bond_cubic_stretch);
    scalar_t constexpr cs2 = cs * cs;
    scalar_t constexpr one_point_five = static_cast<scalar_t>(1.5);
    scalar_t constexpr seven_sixths = static_cast<scalar_t>(7.0 / 6.0);

    std::size_t const bond_end = bond_start + bond_count;
    for (std::size_t idx = bond_start; idx < bond_end; ++idx) {
        std::int32_t const i = bond_atom_i[idx];
        std::int32_t const j = bond_atom_j[idx];

        scalar_t const dx = coord_x[j] - coord_x[i];
        scalar_t const dy = coord_y[j] - coord_y[i];
        scalar_t const dz = coord_z[j] - coord_z[i];
        scalar_t const r = std::sqrt(dx * dx + dy * dy + dz * dz);

        scalar_t const kb = bond_kb[idx];
        scalar_t const r0 = bond_r0[idx];
        scalar_t const dr = r - r0;

        // dE/dr = 143.9325 * kb * dr * (1 + 1.5*cs*dr + 7/6*cs^2*dr^2)
        scalar_t const term = static_cast<scalar_t>(1.0) + one_point_five * cs * dr + seven_sixths * cs2 * dr * dr;
        scalar_t const dE_dr = prefactor * kb * dr * term;

        // Convert to force: F = -dE/dr * (r_vec / |r_vec|)
        scalar_t const inv_r = static_cast<scalar_t>(1.0) / r;
        scalar_t const fx = dE_dr * dx * inv_r;
        scalar_t const fy = dE_dr * dy * inv_r;
        scalar_t const fz = dE_dr * dz * inv_r;

        // Gradient (not force): grad_i = -F_i, grad_j = +F_i
        grad_x[i] -= fx;
        grad_y[i] -= fy;
        grad_z[i] -= fz;
        grad_x[j] += fx;
        grad_y[j] += fy;
        grad_z[j] += fz;
    }
}

// =============================================================================
// Angle Bending Energy and Gradient (Cubic Polynomial)
// =============================================================================

/**
 * Compute angle bending energy using MMFF94 cubic polynomial.
 * E = 0.5 * 0.043844 * ka * dtheta^2 * (1 + cb*dtheta)
 * where dtheta is in degrees.
 */
template <typename scalar_t>
scalar_t compute_angle_energy(scalar_t const* coord_x, scalar_t const* coord_y, scalar_t const* coord_z,
                              std::int32_t const* angle_atom_i, std::int32_t const* angle_atom_j,
                              std::int32_t const* angle_atom_k, scalar_t const* angle_ka, scalar_t const* angle_theta0,
                              std::size_t angle_start, std::size_t angle_count) {
    scalar_t constexpr prefactor = static_cast<scalar_t>(angle_prefactor);
    scalar_t constexpr cb = static_cast<scalar_t>(angle_cubic_bend);
    scalar_t constexpr to_deg = static_cast<scalar_t>(rad_to_deg);

    scalar_t accum = 0;

    std::size_t const angle_end = angle_start + angle_count;
    for (std::size_t idx = angle_start; idx < angle_end; ++idx) {
        std::int32_t const i = angle_atom_i[idx];
        std::int32_t const j = angle_atom_j[idx]; // central atom
        std::int32_t const k = angle_atom_k[idx];

        // Vectors from central atom j to i and k
        scalar_t const rji_x = coord_x[i] - coord_x[j];
        scalar_t const rji_y = coord_y[i] - coord_y[j];
        scalar_t const rji_z = coord_z[i] - coord_z[j];
        scalar_t const rjk_x = coord_x[k] - coord_x[j];
        scalar_t const rjk_y = coord_y[k] - coord_y[j];
        scalar_t const rjk_z = coord_z[k] - coord_z[j];

        scalar_t const rji_len = std::sqrt(rji_x * rji_x + rji_y * rji_y + rji_z * rji_z);
        scalar_t const rjk_len = std::sqrt(rjk_x * rjk_x + rjk_y * rjk_y + rjk_z * rjk_z);

        // cos(theta) = (rji . rjk) / (|rji| * |rjk|)
        scalar_t const dot = rji_x * rjk_x + rji_y * rjk_y + rji_z * rjk_z;
        scalar_t const cos_theta =
            clamp_(dot / (rji_len * rjk_len), static_cast<scalar_t>(-1.0), static_cast<scalar_t>(1.0));
        scalar_t const theta_rad = std::acos(cos_theta);
        scalar_t const theta_deg = theta_rad * to_deg;

        scalar_t const ka = angle_ka[idx];
        scalar_t const theta0 = angle_theta0[idx]; // already in degrees
        scalar_t const dtheta = theta_deg - theta0;
        scalar_t const dtheta2 = dtheta * dtheta;

        // E = 0.5 * 0.043844 * ka * dtheta^2 * (1 + cb*dtheta)
        scalar_t const term = static_cast<scalar_t>(1.0) + cb * dtheta;
        accum += static_cast<scalar_t>(0.5) * prefactor * ka * dtheta2 * term;
    }

    return accum;
}

/**
 * Compute angle bending gradient using MMFF94 cubic polynomial.
 * Uses chain rule: dE/dx = dE/dtheta * dtheta/dx
 */
template <typename scalar_t>
void compute_angle_gradient(scalar_t const* coord_x, scalar_t const* coord_y, scalar_t const* coord_z,
                            std::int32_t const* angle_atom_i, std::int32_t const* angle_atom_j,
                            std::int32_t const* angle_atom_k, scalar_t const* angle_ka, scalar_t const* angle_theta0,
                            std::size_t angle_start, std::size_t angle_count, scalar_t* grad_x, scalar_t* grad_y,
                            scalar_t* grad_z) {
    std::size_t const angle_end = angle_start + angle_count;
    for (std::size_t idx = angle_start; idx < angle_end; ++idx) {
        std::int32_t const i = angle_atom_i[idx];
        std::int32_t const j = angle_atom_j[idx]; // central atom
        std::int32_t const k = angle_atom_k[idx];

        vec3_t<scalar_t> const p1 = make_vec3_(coord_x[i], coord_y[i], coord_z[i]);
        vec3_t<scalar_t> const p2 = make_vec3_(coord_x[j], coord_y[j], coord_z[j]);
        vec3_t<scalar_t> const p3 = make_vec3_(coord_x[k], coord_y[k], coord_z[k]);

        vec3_t<scalar_t> const r12 = sub_(p1, p2);
        vec3_t<scalar_t> const r32 = sub_(p3, p2);
        scalar_t const dist1 = norm_(r12);
        scalar_t const dist2 = norm_(r32);
        if (dist1 < static_cast<scalar_t>(1e-8) || dist2 < static_cast<scalar_t>(1e-8)) {
            continue;
        }

        vec3_t<scalar_t> const p12 = scale_(r12, static_cast<scalar_t>(1.0) / dist1);
        vec3_t<scalar_t> const p32 = scale_(r32, static_cast<scalar_t>(1.0) / dist2);
        scalar_t const cos_theta = clamp_(dot_(p12, p32), static_cast<scalar_t>(-1.0), static_cast<scalar_t>(1.0));
        scalar_t const sin_theta_sq = static_cast<scalar_t>(1.0) - cos_theta * cos_theta;
        scalar_t const sin_theta =
            std::max((sin_theta_sq > static_cast<scalar_t>(0.0) ? std::sqrt(sin_theta_sq) : static_cast<scalar_t>(0.0)),
                     static_cast<scalar_t>(1e-8));

        scalar_t constexpr cb = static_cast<scalar_t>(angle_cubic_bend);
        scalar_t constexpr to_deg = static_cast<scalar_t>(rad_to_deg);
        scalar_t constexpr c2 = static_cast<scalar_t>(angle_prefactor);
        scalar_t const theta0 = angle_theta0[idx];
        scalar_t const angle_term = to_deg * std::acos(cos_theta) - theta0;
        scalar_t const ka = angle_ka[idx];

        // dE/dTheta in degrees (matches RDKit form).
        scalar_t const dE_dtheta =
            to_deg * c2 * ka * angle_term * (static_cast<scalar_t>(1.0) + static_cast<scalar_t>(1.5) * cb * angle_term);

        scalar_t const dCos_dS1 = static_cast<scalar_t>(1.0) / dist1 * (p32.x - cos_theta * p12.x);
        scalar_t const dCos_dS2 = static_cast<scalar_t>(1.0) / dist1 * (p32.y - cos_theta * p12.y);
        scalar_t const dCos_dS3 = static_cast<scalar_t>(1.0) / dist1 * (p32.z - cos_theta * p12.z);
        scalar_t const dCos_dS4 = static_cast<scalar_t>(1.0) / dist2 * (p12.x - cos_theta * p32.x);
        scalar_t const dCos_dS5 = static_cast<scalar_t>(1.0) / dist2 * (p12.y - cos_theta * p32.y);
        scalar_t const dCos_dS6 = static_cast<scalar_t>(1.0) / dist2 * (p12.z - cos_theta * p32.z);

        scalar_t const inv_sin = static_cast<scalar_t>(-1.0) / sin_theta;
        grad_x[i] += dE_dtheta * dCos_dS1 * inv_sin;
        grad_y[i] += dE_dtheta * dCos_dS2 * inv_sin;
        grad_z[i] += dE_dtheta * dCos_dS3 * inv_sin;
        grad_x[j] += dE_dtheta * (-dCos_dS1 - dCos_dS4) * inv_sin;
        grad_y[j] += dE_dtheta * (-dCos_dS2 - dCos_dS5) * inv_sin;
        grad_z[j] += dE_dtheta * (-dCos_dS3 - dCos_dS6) * inv_sin;
        grad_x[k] += dE_dtheta * dCos_dS4 * inv_sin;
        grad_y[k] += dE_dtheta * dCos_dS5 * inv_sin;
        grad_z[k] += dE_dtheta * dCos_dS6 * inv_sin;
    }
}

// =============================================================================
// Torsion Energy and Gradient (Cosine Series)
// =============================================================================

/**
 * Compute torsion energy using MMFF94 3-term cosine series.
 * E = 0.5 * (V1*(1+cos(phi)) + V2*(1-cos(2*phi)) + V3*(1+cos(3*phi)))
 */
template <typename scalar_t>
scalar_t compute_torsion_energy(scalar_t const* coord_x, scalar_t const* coord_y, scalar_t const* coord_z,
                                std::int32_t const* torsion_atom_i, std::int32_t const* torsion_atom_j,
                                std::int32_t const* torsion_atom_k, std::int32_t const* torsion_atom_l,
                                scalar_t const* torsion_v1, scalar_t const* torsion_v2, scalar_t const* torsion_v3,
                                std::size_t torsion_start, std::size_t torsion_count) {
    scalar_t constexpr prefactor = static_cast<scalar_t>(torsion_prefactor);

    scalar_t accum = 0;

    std::size_t const torsion_end = torsion_start + torsion_count;
    for (std::size_t idx = torsion_start; idx < torsion_end; ++idx) {
        std::int32_t const i = torsion_atom_i[idx];
        std::int32_t const j = torsion_atom_j[idx];
        std::int32_t const k = torsion_atom_k[idx];
        std::int32_t const l = torsion_atom_l[idx];

        // Vectors: b1 = j-i, b2 = k-j, b3 = l-k
        scalar_t const b1_x = coord_x[j] - coord_x[i];
        scalar_t const b1_y = coord_y[j] - coord_y[i];
        scalar_t const b1_z = coord_z[j] - coord_z[i];
        scalar_t const b2_x = coord_x[k] - coord_x[j];
        scalar_t const b2_y = coord_y[k] - coord_y[j];
        scalar_t const b2_z = coord_z[k] - coord_z[j];
        scalar_t const b3_x = coord_x[l] - coord_x[k];
        scalar_t const b3_y = coord_y[l] - coord_y[k];
        scalar_t const b3_z = coord_z[l] - coord_z[k];

        // Normal vectors: n1 = b1 x b2, n2 = b2 x b3
        scalar_t const n1_x = b1_y * b2_z - b1_z * b2_y;
        scalar_t const n1_y = b1_z * b2_x - b1_x * b2_z;
        scalar_t const n1_z = b1_x * b2_y - b1_y * b2_x;
        scalar_t const n2_x = b2_y * b3_z - b2_z * b3_y;
        scalar_t const n2_y = b2_z * b3_x - b2_x * b3_z;
        scalar_t const n2_z = b2_x * b3_y - b2_y * b3_x;

        scalar_t const n1_len = std::sqrt(n1_x * n1_x + n1_y * n1_y + n1_z * n1_z);
        scalar_t const n2_len = std::sqrt(n2_x * n2_x + n2_y * n2_y + n2_z * n2_z);

        if (n1_len < static_cast<scalar_t>(1e-8) || n2_len < static_cast<scalar_t>(1e-8))
            continue;

        // cos(phi) = (n1 . n2) / (|n1| * |n2|)
        scalar_t const dot_n = n1_x * n2_x + n1_y * n2_y + n1_z * n2_z;
        scalar_t const cos_phi =
            clamp_(dot_n / (n1_len * n2_len), static_cast<scalar_t>(-1.0), static_cast<scalar_t>(1.0));

        // Sign of phi from (n1 x n2) . b2
        scalar_t const cross_x = n1_y * n2_z - n1_z * n2_y;
        scalar_t const cross_y = n1_z * n2_x - n1_x * n2_z;
        scalar_t const cross_z = n1_x * n2_y - n1_y * n2_x;
        scalar_t const sign_dot = cross_x * b2_x + cross_y * b2_y + cross_z * b2_z;
        scalar_t const sign = (sign_dot >= 0) ? static_cast<scalar_t>(1.0) : static_cast<scalar_t>(-1.0);
        scalar_t const phi = sign * std::acos(cos_phi);

        scalar_t const v1 = torsion_v1[idx];
        scalar_t const v2 = torsion_v2[idx];
        scalar_t const v3 = torsion_v3[idx];

        // E = 0.5 * (V1*(1+cos(phi)) + V2*(1-cos(2*phi)) + V3*(1+cos(3*phi)))
        scalar_t const cos_2phi = static_cast<scalar_t>(2.0) * cos_phi * cos_phi - static_cast<scalar_t>(1.0);
        scalar_t const cos_3phi =
            cos_phi * (static_cast<scalar_t>(4.0) * cos_phi * cos_phi - static_cast<scalar_t>(3.0));

        accum +=
            prefactor * (v1 * (static_cast<scalar_t>(1.0) + cos_phi) + v2 * (static_cast<scalar_t>(1.0) - cos_2phi) +
                         v3 * (static_cast<scalar_t>(1.0) + cos_3phi));
    }

    return accum;
}

/**
 * Compute torsion gradient using MMFF94 3-term cosine series.
 * dE/dphi = 0.5 * (-V1*sin(phi) + 2*V2*sin(2*phi) - 3*V3*sin(3*phi))
 */
template <typename scalar_t>
void compute_torsion_gradient(scalar_t const* coord_x, scalar_t const* coord_y, scalar_t const* coord_z,
                              std::int32_t const* torsion_atom_i, std::int32_t const* torsion_atom_j,
                              std::int32_t const* torsion_atom_k, std::int32_t const* torsion_atom_l,
                              scalar_t const* torsion_v1, scalar_t const* torsion_v2, scalar_t const* torsion_v3,
                              std::size_t torsion_start, std::size_t torsion_count, scalar_t* grad_x, scalar_t* grad_y,
                              scalar_t* grad_z) {
    std::size_t const torsion_end = torsion_start + torsion_count;
    for (std::size_t idx = torsion_start; idx < torsion_end; ++idx) {
        std::int32_t const i = torsion_atom_i[idx];
        std::int32_t const j = torsion_atom_j[idx];
        std::int32_t const k = torsion_atom_k[idx];
        std::int32_t const l = torsion_atom_l[idx];

        vec3_t<scalar_t> const p1 = make_vec3_(coord_x[i], coord_y[i], coord_z[i]);
        vec3_t<scalar_t> const p2 = make_vec3_(coord_x[j], coord_y[j], coord_z[j]);
        vec3_t<scalar_t> const p3 = make_vec3_(coord_x[k], coord_y[k], coord_z[k]);
        vec3_t<scalar_t> const p4 = make_vec3_(coord_x[l], coord_y[l], coord_z[l]);

        vec3_t<scalar_t> r[4];
        vec3_t<scalar_t> t[2];
        scalar_t d[2];
        r[0] = sub_(p1, p2);
        r[1] = sub_(p3, p2);
        r[2] = scale_(r[1], static_cast<scalar_t>(-1.0));
        r[3] = sub_(p4, p3);

        t[0] = cross_(r[0], r[1]);
        d[0] = std::max(norm_(t[0]), static_cast<scalar_t>(1e-5));
        t[0] = scale_(t[0], static_cast<scalar_t>(1.0) / d[0]);
        t[1] = cross_(r[2], r[3]);
        d[1] = std::max(norm_(t[1]), static_cast<scalar_t>(1e-5));
        t[1] = scale_(t[1], static_cast<scalar_t>(1.0) / d[1]);

        scalar_t const cos_phi = clamp_(dot_(t[0], t[1]), static_cast<scalar_t>(-1.0), static_cast<scalar_t>(1.0));
        scalar_t const sin_phi_sq = static_cast<scalar_t>(1.0) - cos_phi * cos_phi;
        scalar_t const sin_phi =
            (sin_phi_sq > static_cast<scalar_t>(0.0)) ? std::sqrt(sin_phi_sq) : static_cast<scalar_t>(0.0);

        scalar_t const v1 = torsion_v1[idx];
        scalar_t const v2 = torsion_v2[idx];
        scalar_t const v3 = torsion_v3[idx];

        // dE/dphi = 0.5 * (-V1*sin(phi) + 2*V2*sin(2*phi) - 3*V3*sin(3*phi))
        scalar_t const sin_2phi = static_cast<scalar_t>(2.0) * sin_phi * cos_phi;
        scalar_t const sin_3phi =
            static_cast<scalar_t>(3.0) * sin_phi - static_cast<scalar_t>(4.0) * sin_phi * sin_phi * sin_phi;
        scalar_t const dE_dphi =
            static_cast<scalar_t>(0.5) *
            (-v1 * sin_phi + static_cast<scalar_t>(2.0) * v2 * sin_2phi - static_cast<scalar_t>(3.0) * v3 * sin_3phi);

        if (std::abs(cos_phi) < static_cast<scalar_t>(1e-8) && std::abs(sin_phi) < static_cast<scalar_t>(1e-8)) {
            continue;
        }

        scalar_t const sin_term =
            (std::abs(sin_phi) < static_cast<scalar_t>(1e-8)) ? (-dE_dphi / cos_phi) : (-dE_dphi / sin_phi);

        scalar_t const dCos_dT[6] = {
            static_cast<scalar_t>(1.0) / d[0] * (t[1].x - cos_phi * t[0].x),
            static_cast<scalar_t>(1.0) / d[0] * (t[1].y - cos_phi * t[0].y),
            static_cast<scalar_t>(1.0) / d[0] * (t[1].z - cos_phi * t[0].z),
            static_cast<scalar_t>(1.0) / d[1] * (t[0].x - cos_phi * t[1].x),
            static_cast<scalar_t>(1.0) / d[1] * (t[0].y - cos_phi * t[1].y),
            static_cast<scalar_t>(1.0) / d[1] * (t[0].z - cos_phi * t[1].z),
        };

        grad_x[i] += sin_term * (dCos_dT[2] * r[1].y - dCos_dT[1] * r[1].z);
        grad_y[i] += sin_term * (dCos_dT[0] * r[1].z - dCos_dT[2] * r[1].x);
        grad_z[i] += sin_term * (dCos_dT[1] * r[1].x - dCos_dT[0] * r[1].y);

        grad_x[j] += sin_term * (dCos_dT[1] * (r[1].z - r[0].z) + dCos_dT[2] * (r[0].y - r[1].y) +
                                 dCos_dT[4] * (-r[3].z) + dCos_dT[5] * (r[3].y));
        grad_y[j] += sin_term * (dCos_dT[0] * (r[0].z - r[1].z) + dCos_dT[2] * (r[1].x - r[0].x) +
                                 dCos_dT[3] * (r[3].z) + dCos_dT[5] * (-r[3].x));
        grad_z[j] += sin_term * (dCos_dT[0] * (r[1].y - r[0].y) + dCos_dT[1] * (r[0].x - r[1].x) +
                                 dCos_dT[3] * (-r[3].y) + dCos_dT[4] * (r[3].x));

        grad_x[k] += sin_term * (dCos_dT[1] * (r[0].z) + dCos_dT[2] * (-r[0].y) + dCos_dT[4] * (r[3].z - r[2].z) +
                                 dCos_dT[5] * (r[2].y - r[3].y));
        grad_y[k] += sin_term * (dCos_dT[0] * (-r[0].z) + dCos_dT[2] * (r[0].x) + dCos_dT[3] * (r[2].z - r[3].z) +
                                 dCos_dT[5] * (r[3].x - r[2].x));
        grad_z[k] += sin_term * (dCos_dT[0] * (r[0].y) + dCos_dT[1] * (-r[0].x) + dCos_dT[3] * (r[3].y - r[2].y) +
                                 dCos_dT[4] * (r[2].x - r[3].x));

        grad_x[l] += sin_term * (dCos_dT[4] * r[2].z - dCos_dT[5] * r[2].y);
        grad_y[l] += sin_term * (dCos_dT[5] * r[2].x - dCos_dT[3] * r[2].z);
        grad_z[l] += sin_term * (dCos_dT[3] * r[2].y - dCos_dT[4] * r[2].x);
    }
}

// =============================================================================
// Stretch-Bend Coupling Energy and Gradient
// =============================================================================

/**
 * Compute stretch-bend coupling energy.
 * E = 2.51210 * (kba_ijk * dr_ij + kba_kji * dr_kj) * dtheta
 */
template <typename scalar_t>
scalar_t compute_stretch_bend_energy(scalar_t const* coord_x, scalar_t const* coord_y, scalar_t const* coord_z,
                                     std::int32_t const* stbn_atom_i, std::int32_t const* stbn_atom_j,
                                     std::int32_t const* stbn_atom_k, scalar_t const* stbn_kba_ijk,
                                     scalar_t const* stbn_kba_kji, scalar_t const* stbn_r0_ij,
                                     scalar_t const* stbn_r0_kj, scalar_t const* stbn_theta0, std::size_t stbn_start,
                                     std::size_t stbn_count) {
    scalar_t constexpr prefactor = static_cast<scalar_t>(stretch_bend_prefactor);
    scalar_t constexpr to_deg = static_cast<scalar_t>(rad_to_deg);

    scalar_t accum = 0;

    std::size_t const stbn_end = stbn_start + stbn_count;
    for (std::size_t idx = stbn_start; idx < stbn_end; ++idx) {
        std::int32_t const i = stbn_atom_i[idx];
        std::int32_t const j = stbn_atom_j[idx];
        std::int32_t const k = stbn_atom_k[idx];

        // Compute bond lengths and angle
        scalar_t const rji_x = coord_x[i] - coord_x[j];
        scalar_t const rji_y = coord_y[i] - coord_y[j];
        scalar_t const rji_z = coord_z[i] - coord_z[j];
        scalar_t const rjk_x = coord_x[k] - coord_x[j];
        scalar_t const rjk_y = coord_y[k] - coord_y[j];
        scalar_t const rjk_z = coord_z[k] - coord_z[j];

        scalar_t const r_ij = std::sqrt(rji_x * rji_x + rji_y * rji_y + rji_z * rji_z);
        scalar_t const r_kj = std::sqrt(rjk_x * rjk_x + rjk_y * rjk_y + rjk_z * rjk_z);

        scalar_t const dot = rji_x * rjk_x + rji_y * rjk_y + rji_z * rjk_z;
        scalar_t const cos_theta = clamp_(dot / (r_ij * r_kj), static_cast<scalar_t>(-1.0), static_cast<scalar_t>(1.0));
        scalar_t const theta_deg = std::acos(cos_theta) * to_deg;

        scalar_t const dr_ij = r_ij - stbn_r0_ij[idx];
        scalar_t const dr_kj = r_kj - stbn_r0_kj[idx];
        scalar_t const dtheta = theta_deg - stbn_theta0[idx];

        // E = 2.51210 * (kba_ijk * dr_ij + kba_kji * dr_kj) * dtheta
        accum += prefactor * (stbn_kba_ijk[idx] * dr_ij + stbn_kba_kji[idx] * dr_kj) * dtheta;
    }

    return accum;
}

/**
 * Compute stretch-bend coupling gradient.
 * Cross-term coupling between bond stretches and angle bend.
 */
template <typename scalar_t>
void compute_stretch_bend_gradient(scalar_t const* coord_x, scalar_t const* coord_y, scalar_t const* coord_z,
                                   std::int32_t const* stbn_atom_i, std::int32_t const* stbn_atom_j,
                                   std::int32_t const* stbn_atom_k, scalar_t const* stbn_kba_ijk,
                                   scalar_t const* stbn_kba_kji, scalar_t const* stbn_r0_ij, scalar_t const* stbn_r0_kj,
                                   scalar_t const* stbn_theta0, std::size_t stbn_start, std::size_t stbn_count,
                                   scalar_t* grad_x, scalar_t* grad_y, scalar_t* grad_z) {
    std::size_t const stbn_end = stbn_start + stbn_count;
    for (std::size_t idx = stbn_start; idx < stbn_end; ++idx) {
        std::int32_t const i = stbn_atom_i[idx];
        std::int32_t const j = stbn_atom_j[idx];
        std::int32_t const k = stbn_atom_k[idx];

        vec3_t<scalar_t> const p1 = make_vec3_(coord_x[i], coord_y[i], coord_z[i]);
        vec3_t<scalar_t> const p2 = make_vec3_(coord_x[j], coord_y[j], coord_z[j]);
        vec3_t<scalar_t> const p3 = make_vec3_(coord_x[k], coord_y[k], coord_z[k]);

        vec3_t<scalar_t> const r12 = sub_(p1, p2);
        vec3_t<scalar_t> const r32 = sub_(p3, p2);
        scalar_t const dist1 = norm_(r12);
        scalar_t const dist2 = norm_(r32);
        if (dist1 < static_cast<scalar_t>(1e-8) || dist2 < static_cast<scalar_t>(1e-8)) {
            continue;
        }

        vec3_t<scalar_t> const p12 = scale_(r12, static_cast<scalar_t>(1.0) / dist1);
        vec3_t<scalar_t> const p32 = scale_(r32, static_cast<scalar_t>(1.0) / dist2);
        scalar_t const cos_theta = clamp_(dot_(p12, p32), static_cast<scalar_t>(-1.0), static_cast<scalar_t>(1.0));
        scalar_t const sin_theta_sq = static_cast<scalar_t>(1.0) - cos_theta * cos_theta;
        scalar_t const sin_theta =
            std::max((sin_theta_sq > static_cast<scalar_t>(0.0) ? std::sqrt(sin_theta_sq) : static_cast<scalar_t>(0.0)),
                     static_cast<scalar_t>(1e-8));

        scalar_t constexpr to_deg = static_cast<scalar_t>(rad_to_deg);
        scalar_t constexpr c5 = static_cast<scalar_t>(stretch_bend_prefactor);
        scalar_t const theta0 = stbn_theta0[idx];
        scalar_t const angle_term = to_deg * std::acos(cos_theta) - theta0; // degrees
        // dist_term is sum of (force_constant * bond_deviation) - NO to_deg factor here!
        scalar_t const dist_term =
            stbn_kba_ijk[idx] * (dist1 - stbn_r0_ij[idx]) + stbn_kba_kji[idx] * (dist2 - stbn_r0_kj[idx]);

        scalar_t const dCos_dS1 = static_cast<scalar_t>(1.0) / dist1 * (p32.x - cos_theta * p12.x);
        scalar_t const dCos_dS2 = static_cast<scalar_t>(1.0) / dist1 * (p32.y - cos_theta * p12.y);
        scalar_t const dCos_dS3 = static_cast<scalar_t>(1.0) / dist1 * (p32.z - cos_theta * p12.z);
        scalar_t const dCos_dS4 = static_cast<scalar_t>(1.0) / dist2 * (p12.x - cos_theta * p32.x);
        scalar_t const dCos_dS5 = static_cast<scalar_t>(1.0) / dist2 * (p12.y - cos_theta * p32.y);
        scalar_t const dCos_dS6 = static_cast<scalar_t>(1.0) / dist2 * (p12.z - cos_theta * p32.z);

        // Chain rule: ∂θ_deg/∂cos(θ) = -to_deg/sin(θ) because θ_deg = acos(cos(θ)) * to_deg
        scalar_t const dTheta_dCos = -to_deg / sin_theta;
        grad_x[i] += c5 * (p12.x * stbn_kba_ijk[idx] * angle_term + dCos_dS1 * dTheta_dCos * dist_term);
        grad_y[i] += c5 * (p12.y * stbn_kba_ijk[idx] * angle_term + dCos_dS2 * dTheta_dCos * dist_term);
        grad_z[i] += c5 * (p12.z * stbn_kba_ijk[idx] * angle_term + dCos_dS3 * dTheta_dCos * dist_term);
        grad_x[j] += c5 * ((-p12.x * stbn_kba_ijk[idx] - p32.x * stbn_kba_kji[idx]) * angle_term +
                           (-dCos_dS1 - dCos_dS4) * dTheta_dCos * dist_term);
        grad_y[j] += c5 * ((-p12.y * stbn_kba_ijk[idx] - p32.y * stbn_kba_kji[idx]) * angle_term +
                           (-dCos_dS2 - dCos_dS5) * dTheta_dCos * dist_term);
        grad_z[j] += c5 * ((-p12.z * stbn_kba_ijk[idx] - p32.z * stbn_kba_kji[idx]) * angle_term +
                           (-dCos_dS3 - dCos_dS6) * dTheta_dCos * dist_term);
        grad_x[k] += c5 * (p32.x * stbn_kba_kji[idx] * angle_term + dCos_dS4 * dTheta_dCos * dist_term);
        grad_y[k] += c5 * (p32.y * stbn_kba_kji[idx] * angle_term + dCos_dS5 * dTheta_dCos * dist_term);
        grad_z[k] += c5 * (p32.z * stbn_kba_kji[idx] * angle_term + dCos_dS6 * dTheta_dCos * dist_term);
    }
}

// =============================================================================
// Out-of-Plane Bending Energy and Gradient (Wilson Angle)
// =============================================================================

/**
 * Compute out-of-plane bending energy for sp2 centers.
 * E = 0.5 * 0.043844 * koop * chi^2
 * where chi is the Wilson out-of-plane angle in degrees.
 * Atom j is the central sp2 atom with neighbors i, k, l.
 */
template <typename scalar_t>
scalar_t compute_oop_energy(scalar_t const* coord_x, scalar_t const* coord_y, scalar_t const* coord_z,
                            std::int32_t const* oop_atom_i, std::int32_t const* oop_atom_j,
                            std::int32_t const* oop_atom_k, std::int32_t const* oop_atom_l, scalar_t const* oop_koop,
                            std::size_t oop_start, std::size_t oop_count) {
    scalar_t constexpr prefactor = static_cast<scalar_t>(oop_prefactor);
    scalar_t constexpr to_deg = static_cast<scalar_t>(rad_to_deg);

    scalar_t accum = 0;

    std::size_t const oop_end = oop_start + oop_count;
    for (std::size_t idx = oop_start; idx < oop_end; ++idx) {
        std::int32_t const i = oop_atom_i[idx];
        std::int32_t const j = oop_atom_j[idx]; // central sp2 atom
        std::int32_t const k = oop_atom_k[idx];
        std::int32_t const l = oop_atom_l[idx];

        // Vectors from central atom j
        scalar_t const rji_x = coord_x[i] - coord_x[j];
        scalar_t const rji_y = coord_y[i] - coord_y[j];
        scalar_t const rji_z = coord_z[i] - coord_z[j];
        scalar_t const rjk_x = coord_x[k] - coord_x[j];
        scalar_t const rjk_y = coord_y[k] - coord_y[j];
        scalar_t const rjk_z = coord_z[k] - coord_z[j];
        scalar_t const rjl_x = coord_x[l] - coord_x[j];
        scalar_t const rjl_y = coord_y[l] - coord_y[j];
        scalar_t const rjl_z = coord_z[l] - coord_z[j];

        // Normal to plane i-j-k: n = rji x rjk
        scalar_t const n_x = rji_y * rjk_z - rji_z * rjk_y;
        scalar_t const n_y = rji_z * rjk_x - rji_x * rjk_z;
        scalar_t const n_z = rji_x * rjk_y - rji_y * rjk_x;
        scalar_t const n_len = std::sqrt(n_x * n_x + n_y * n_y + n_z * n_z);
        scalar_t const rjl_len = std::sqrt(rjl_x * rjl_x + rjl_y * rjl_y + rjl_z * rjl_z);

        if (n_len < static_cast<scalar_t>(1e-8) || rjl_len < static_cast<scalar_t>(1e-8))
            continue;

        // Wilson angle: sin(chi) = (rjl . n) / (|rjl| * |n|)
        scalar_t const dot = rjl_x * n_x + rjl_y * n_y + rjl_z * n_z;
        scalar_t const sin_chi =
            clamp_(dot / (rjl_len * n_len), static_cast<scalar_t>(-1.0), static_cast<scalar_t>(1.0));
        scalar_t const chi_rad = std::asin(sin_chi);
        scalar_t const chi_deg = chi_rad * to_deg;

        scalar_t const koop = oop_koop[idx];

        // E = 0.5 * 0.043844 * koop * chi^2
        accum += static_cast<scalar_t>(0.5) * prefactor * koop * chi_deg * chi_deg;
    }

    return accum;
}

/**
 * Compute out-of-plane bending gradient for sp2 centers.
 */
template <typename scalar_t>
void compute_oop_gradient(scalar_t const* coord_x, scalar_t const* coord_y, scalar_t const* coord_z,
                          std::int32_t const* oop_atom_i, std::int32_t const* oop_atom_j,
                          std::int32_t const* oop_atom_k, std::int32_t const* oop_atom_l, scalar_t const* oop_koop,
                          std::size_t oop_start, std::size_t oop_count, scalar_t* grad_x, scalar_t* grad_y,
                          scalar_t* grad_z) {
    std::size_t const oop_end = oop_start + oop_count;
    for (std::size_t idx = oop_start; idx < oop_end; ++idx) {
        std::int32_t const i = oop_atom_i[idx];
        std::int32_t const j = oop_atom_j[idx];
        std::int32_t const k = oop_atom_k[idx];
        std::int32_t const l = oop_atom_l[idx];

        vec3_t<scalar_t> rji = make_vec3_(coord_x[i] - coord_x[j], coord_y[i] - coord_y[j], coord_z[i] - coord_z[j]);
        vec3_t<scalar_t> rjk = make_vec3_(coord_x[k] - coord_x[j], coord_y[k] - coord_y[j], coord_z[k] - coord_z[j]);
        vec3_t<scalar_t> rjl = make_vec3_(coord_x[l] - coord_x[j], coord_y[l] - coord_y[j], coord_z[l] - coord_z[j]);

        scalar_t const dji = norm_(rji);
        scalar_t const djk = norm_(rjk);
        scalar_t const djl = norm_(rjl);
        if (dji < static_cast<scalar_t>(1e-8) || djk < static_cast<scalar_t>(1e-8) ||
            djl < static_cast<scalar_t>(1e-8)) {
            continue;
        }

        rji = scale_(rji, static_cast<scalar_t>(1.0) / dji);
        rjk = scale_(rjk, static_cast<scalar_t>(1.0) / djk);
        rjl = scale_(rjl, static_cast<scalar_t>(1.0) / djl);

        vec3_t<scalar_t> n = cross_(scale_(rji, static_cast<scalar_t>(-1.0)), rjk);
        scalar_t const n_len = std::max(norm_(n), static_cast<scalar_t>(1e-8));
        n = scale_(n, static_cast<scalar_t>(1.0) / n_len);

        scalar_t const c2 = static_cast<scalar_t>(oop_prefactor);
        scalar_t const to_deg = static_cast<scalar_t>(rad_to_deg);
        scalar_t const sin_chi = clamp_(dot_(rjl, n), static_cast<scalar_t>(-1.0), static_cast<scalar_t>(1.0));
        scalar_t const cos_chi_sq = static_cast<scalar_t>(1.0) - sin_chi * sin_chi;
        scalar_t const cos_chi =
            std::max((cos_chi_sq > static_cast<scalar_t>(0.0) ? std::sqrt(cos_chi_sq) : static_cast<scalar_t>(0.0)),
                     static_cast<scalar_t>(1e-8));
        scalar_t const chi = to_deg * std::asin(sin_chi);
        scalar_t const cos_theta = clamp_(dot_(rji, rjk), static_cast<scalar_t>(-1.0), static_cast<scalar_t>(1.0));
        scalar_t const sin_theta_sq =
            std::max(static_cast<scalar_t>(1.0) - cos_theta * cos_theta, static_cast<scalar_t>(1e-8));
        scalar_t const sin_theta =
            std::max((sin_theta_sq > static_cast<scalar_t>(0.0) ? std::sqrt(sin_theta_sq) : static_cast<scalar_t>(0.0)),
                     static_cast<scalar_t>(1e-8));

        scalar_t const dE_dchi = to_deg * c2 * oop_koop[idx] * chi;
        vec3_t<scalar_t> const t1 = cross_(rjl, rjk);
        vec3_t<scalar_t> const t2 = cross_(rji, rjl);
        vec3_t<scalar_t> const t3 = cross_(rjk, rji);
        scalar_t const term1 = cos_chi * sin_theta;
        scalar_t const term2 = sin_chi / (cos_chi * sin_theta_sq);

        vec3_t<scalar_t> const tg1 = scale_(
            sub_(scale_(t1, static_cast<scalar_t>(1.0) / term1), scale_(sub_(rji, scale_(rjk, cos_theta)), term2)),
            static_cast<scalar_t>(1.0) / dji);
        vec3_t<scalar_t> const tg3 = scale_(
            sub_(scale_(t2, static_cast<scalar_t>(1.0) / term1), scale_(sub_(rjk, scale_(rji, cos_theta)), term2)),
            static_cast<scalar_t>(1.0) / djk);
        vec3_t<scalar_t> const tg4 =
            scale_(sub_(scale_(t3, static_cast<scalar_t>(1.0) / term1), scale_(rjl, sin_chi / cos_chi)),
                   static_cast<scalar_t>(1.0) / djl);

        grad_x[i] += dE_dchi * tg1.x;
        grad_y[i] += dE_dchi * tg1.y;
        grad_z[i] += dE_dchi * tg1.z;
        grad_x[j] += -dE_dchi * (tg1.x + tg3.x + tg4.x);
        grad_y[j] += -dE_dchi * (tg1.y + tg3.y + tg4.y);
        grad_z[j] += -dE_dchi * (tg1.z + tg3.z + tg4.z);
        grad_x[k] += dE_dchi * tg3.x;
        grad_y[k] += dE_dchi * tg3.y;
        grad_z[k] += dE_dchi * tg3.z;
        grad_x[l] += dE_dchi * tg4.x;
        grad_y[l] += dE_dchi * tg4.y;
        grad_z[l] += dE_dchi * tg4.z;
    }
}

/**
 * @brief L-BFGS optimizer for MMFF94 force field.
 *
 * Minimizes energy using Limited-memory BFGS with two-loop recursion.
 * Combines VdW and electrostatic gradients for total force calculation.
 *
 * @param coord_x X coordinates (will be modified)
 * @param coord_y Y coordinates (will be modified)
 * @param coord_z Z coordinates (will be modified)
 * @param vdw_pair_i VdW pair first atom indices
 * @param vdw_pair_j VdW pair second atom indices
 * @param vdw_epsilon Well depth parameters
 * @param vdw_r_star Equilibrium distances
 * @param vdw_pair_count Number of VdW pairs
 * @param elec_pair_i Electrostatic pair first atom indices
 * @param elec_pair_j Electrostatic pair second atom indices
 * @param elec_charge_i First atom charges
 * @param elec_charge_j Second atom charges
 * @param elec_is_1_4 1-4 interaction flags
 * @param elec_pair_count Number of electrostatic pairs
 * @param max_iterations Maximum optimization iterations
 * @param grad_threshold RMS gradient convergence threshold
 * @param state L-BFGS state (pre-allocated)
 * @return Optimization result with status and final metrics
 */
template <typename scalar_t>
optimize_result_t lbfgs_optimize(scalar_t* coord_x, scalar_t* coord_y, scalar_t* coord_z,
                                 std::int32_t const* vdw_pair_i, std::int32_t const* vdw_pair_j,
                                 scalar_t const* vdw_epsilon, scalar_t const* vdw_r_star, std::size_t vdw_pair_count,
                                 std::int32_t const* elec_pair_i, std::int32_t const* elec_pair_j,
                                 scalar_t const* elec_charge_i, scalar_t const* elec_charge_j,
                                 std::int32_t const* elec_is_1_4, std::size_t elec_pair_count, int max_iterations,
                                 scalar_t grad_threshold, lbfgs_state_t<scalar_t>& state) {
    scalar_t constexpr zero = static_cast<scalar_t>(0.0);
    scalar_t constexpr one = static_cast<scalar_t>(1.0);
    scalar_t constexpr alpha_init = static_cast<scalar_t>(1.0); // Initial step size (L-BFGS direction is pre-scaled)
    scalar_t constexpr c1 = static_cast<scalar_t>(1e-4);        // Armijo condition constant
    scalar_t constexpr rho_ls = static_cast<scalar_t>(0.5);     // Line search backtrack factor

    std::size_t const n_coords = 3 * state.n_atoms;
    optimize_result_t result{-1, 0, 0.0, 0.0};

    // Helper: compute RMS gradient
    auto compute_rms_gradient_ = [&]() -> scalar_t {
        scalar_t sum_sq = zero;
        for (std::size_t i = 0; i < n_coords; ++i) {
            sum_sq += state.grad[i] * state.grad[i];
        }
        return std::sqrt(sum_sq / static_cast<scalar_t>(n_coords));
    };

    // Helper: compute total energy
    auto compute_total_energy_ = [&]() -> scalar_t {
        scalar_t vdw_energy = compute_vdw_energy(coord_x, coord_y, coord_z, vdw_pair_i, vdw_pair_j, vdw_epsilon,
                                                 vdw_r_star, 0, vdw_pair_count);
        scalar_t elec_energy =
            compute_electrostatic_energy(coord_x, coord_y, coord_z, elec_pair_i, elec_pair_j, elec_charge_i,
                                         elec_charge_j, elec_is_1_4, 0, elec_pair_count);
        return vdw_energy + elec_energy;
    };

    // Helper: compute total gradient (zeros arrays first)
    auto compute_total_gradient_ = [&]() -> void {
        for (std::size_t i = 0; i < state.n_atoms; ++i) {
            state.grad[i] = zero;
            state.grad[state.n_atoms + i] = zero;
            state.grad[2 * state.n_atoms + i] = zero;
        }
        compute_vdw_gradient(coord_x, coord_y, coord_z, vdw_pair_i, vdw_pair_j, vdw_epsilon, vdw_r_star, 0,
                             vdw_pair_count, state.grad, state.grad + state.n_atoms, state.grad + 2 * state.n_atoms);
        compute_electrostatic_gradient(coord_x, coord_y, coord_z, elec_pair_i, elec_pair_j, elec_charge_i,
                                       elec_charge_j, elec_is_1_4, 0, elec_pair_count,
                                       static_cast<scalar_t>(scale_1_4_default), state.grad, state.grad + state.n_atoms,
                                       state.grad + 2 * state.n_atoms);
    };

    // Helper: dot product
    auto dot_product_ = [](scalar_t const* a, scalar_t const* b, std::size_t n) -> scalar_t {
        scalar_t sum = static_cast<scalar_t>(0.0);
        for (std::size_t i = 0; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    };

    // Initialize gradient
    compute_total_gradient_();
    scalar_t rms_grad = compute_rms_gradient_();

    // Check initial convergence
    if (rms_grad < grad_threshold) {
        result.status = 0;
        result.iterations = 0;
        result.final_energy = static_cast<double>(compute_total_energy_());
        result.rms_gradient = static_cast<double>(rms_grad);
        return result;
    }

    // Main optimization loop
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Two-loop recursion to compute search direction
        // Copy gradient to q
        for (std::size_t i = 0; i < n_coords; ++i) {
            state.q[i] = state.grad[i];
        }

        // First loop (backward)
        std::size_t const history_size = (state.k < state.m) ? state.k : state.m;
        for (std::size_t i = 0; i < history_size; ++i) {
            std::size_t const idx = (state.k - 1 - i) % state.m;
            scalar_t const* s_i = state.s_history + idx * n_coords;
            scalar_t const* y_i = state.y_history + idx * n_coords;

            state.alpha[idx] = state.rho[idx] * dot_product_(s_i, state.q, n_coords);
            for (std::size_t j = 0; j < n_coords; ++j) {
                state.q[j] -= state.alpha[idx] * y_i[j];
            }
        }

        // Initial Hessian approximation: H_0 = (s·y)/(y·y) * I
        scalar_t gamma = one;
        if (history_size > 0) {
            std::size_t const newest = (state.k - 1) % state.m;
            scalar_t const* s_newest = state.s_history + newest * n_coords;
            scalar_t const* y_newest = state.y_history + newest * n_coords;
            scalar_t const s_dot_y = dot_product_(s_newest, y_newest, n_coords);
            scalar_t const y_dot_y = dot_product_(y_newest, y_newest, n_coords);
            gamma = s_dot_y / y_dot_y;
        }

        for (std::size_t i = 0; i < n_coords; ++i) {
            state.direction[i] = gamma * state.q[i];
        }

        // Second loop (forward)
        for (std::size_t i = history_size; i > 0; --i) {
            std::size_t const idx = (state.k - i) % state.m;
            scalar_t const* s_i = state.s_history + idx * n_coords;
            scalar_t const* y_i = state.y_history + idx * n_coords;

            scalar_t const beta = state.rho[idx] * dot_product_(y_i, state.direction, n_coords);
            for (std::size_t j = 0; j < n_coords; ++j) {
                state.direction[j] += s_i[j] * (state.alpha[idx] - beta);
            }
        }

        // Negate direction (L-BFGS gives ascent direction, we want descent)
        for (std::size_t i = 0; i < n_coords; ++i) {
            state.direction[i] = -state.direction[i];
        }

        // Backtracking line search
        scalar_t const energy_old = compute_total_energy_();
        scalar_t grad_dot_dir = dot_product_(state.grad, state.direction, n_coords);

        // If direction is not descent, reset history and use steepest descent
        if (grad_dot_dir >= zero) {
            // Reset L-BFGS history
            state.k = 0;
            // Use steepest descent: direction = -gradient
            for (std::size_t i = 0; i < n_coords; ++i) {
                state.direction[i] = -state.grad[i];
            }
            grad_dot_dir = dot_product_(state.grad, state.direction, n_coords);
            // If still not descent (shouldn't happen unless gradient is zero), exit
            if (grad_dot_dir >= zero) {
                result.status = 0; // Converged (gradient is zero or nearly zero)
                result.iterations = iter;
                result.final_energy = static_cast<double>(energy_old);
                result.rms_gradient = static_cast<double>(rms_grad);
                return result;
            }
        }

        scalar_t alpha = alpha_init;
        scalar_t energy_new = zero;
        int ls_iter = 0;
        int const max_ls_iter = 20;

        // Save current coordinates to pre-allocated backup buffer
        scalar_t* backup_x = state.coord_backup;
        scalar_t* backup_y = state.coord_backup + state.n_atoms;
        scalar_t* backup_z = state.coord_backup + 2 * state.n_atoms;
        for (std::size_t i = 0; i < state.n_atoms; ++i) {
            backup_x[i] = coord_x[i];
            backup_y[i] = coord_y[i];
            backup_z[i] = coord_z[i];
        }

        for (; ls_iter < max_ls_iter; ++ls_iter) {
            for (std::size_t i = 0; i < state.n_atoms; ++i) {
                coord_x[i] = backup_x[i] + alpha * state.direction[i];
                coord_y[i] = backup_y[i] + alpha * state.direction[state.n_atoms + i];
                coord_z[i] = backup_z[i] + alpha * state.direction[2 * state.n_atoms + i];
            }
            energy_new = compute_total_energy_();
            if (energy_new <= energy_old + c1 * alpha * grad_dot_dir)
                break;
            alpha *= rho_ls;
        }

        if (ls_iter >= max_ls_iter) {
            // Line search failed - restore coordinates and reset L-BFGS history
            for (std::size_t i = 0; i < state.n_atoms; ++i) {
                coord_x[i] = backup_x[i];
                coord_y[i] = backup_y[i];
                coord_z[i] = backup_z[i];
            }
            state.k = 0; // Reset history for next iteration
            // Continue with steepest descent from current position
            continue;
        }

        // Save old gradient
        for (std::size_t i = 0; i < n_coords; ++i) {
            state.grad_prev[i] = state.grad[i];
        }

        // Compute new gradient
        compute_total_gradient_();
        rms_grad = compute_rms_gradient_();

        // Update history
        std::size_t const hist_idx = state.k % state.m;
        scalar_t* s_k = state.s_history + hist_idx * n_coords;
        scalar_t* y_k = state.y_history + hist_idx * n_coords;

        // s = x_new - x_old = alpha * direction
        // y = grad_new - grad_old
        for (std::size_t i = 0; i < n_coords; ++i) {
            s_k[i] = alpha * state.direction[i];
            y_k[i] = state.grad[i] - state.grad_prev[i];
        }

        // Compute rho = 1/(y·s) - only update history if curvature condition satisfied
        scalar_t const y_dot_s = dot_product_(y_k, s_k, n_coords);
        if (y_dot_s > static_cast<scalar_t>(1e-10)) {
            state.rho[hist_idx] = one / y_dot_s;
            state.k++;
        }
        result.iterations = iter + 1;

        // Check convergence
        if (rms_grad < grad_threshold) {
            result.status = 0; // Converged
            result.final_energy = static_cast<double>(energy_new);
            result.rms_gradient = static_cast<double>(rms_grad);
            return result;
        }
    }

    // Max iterations reached
    result.status = 1;
    result.final_energy = static_cast<double>(compute_total_energy_());
    result.rms_gradient = static_cast<double>(rms_grad);
    return result;
}

/**
 * @brief L-BFGS optimization using complete MMFF94 force field.
 *
 * Includes all energy terms: VdW, electrostatic, bond stretching, angle bending,
 * torsion, stretch-bend coupling, and out-of-plane bending.
 *
 * @param coord_x X coordinates (modified in-place)
 * @param coord_y Y coordinates (modified in-place)
 * @param coord_z Z coordinates (modified in-place)
 * @param params MMFF94 force field parameters
 * @param max_iterations Maximum optimization iterations
 * @param grad_threshold RMS gradient convergence threshold
 * @param state L-BFGS state (pre-allocated)
 * @return Optimization result with status and final metrics
 */
template <typename scalar_t>
optimize_result_t lbfgs_optimize_mmff94(scalar_t* coord_x, scalar_t* coord_y, scalar_t* coord_z,
                                        mmff94_params_t<scalar_t> const& params, int max_iterations,
                                        scalar_t grad_threshold, lbfgs_state_t<scalar_t>& state) {
    scalar_t constexpr zero = static_cast<scalar_t>(0.0);
    scalar_t constexpr one = static_cast<scalar_t>(1.0);
    scalar_t constexpr alpha_init = static_cast<scalar_t>(1.0); // L-BFGS direction is pre-scaled
    scalar_t constexpr c1 = static_cast<scalar_t>(1e-4);
    scalar_t constexpr rho_ls = static_cast<scalar_t>(0.5);

    std::size_t const n_coords = 3 * state.n_atoms;
    optimize_result_t result{-1, 0, 0.0, 0.0};

    auto compute_rms_gradient_ = [&]() -> scalar_t {
        scalar_t sum_sq = zero;
        for (std::size_t i = 0; i < n_coords; ++i) {
            sum_sq += state.grad[i] * state.grad[i];
        }
        return std::sqrt(sum_sq / static_cast<scalar_t>(n_coords));
    };

    auto compute_total_energy_ = [&]() -> scalar_t {
        scalar_t energy = zero;

        // VdW
        if (params.vdw_count > 0) {
            energy += compute_vdw_energy(coord_x, coord_y, coord_z, params.vdw_pair_i, params.vdw_pair_j,
                                         params.vdw_epsilon, params.vdw_r_star, 0, params.vdw_count);
        }

        // Electrostatic
        if (params.elec_count > 0) {
            energy += compute_electrostatic_energy(coord_x, coord_y, coord_z, params.elec_pair_i, params.elec_pair_j,
                                                   params.elec_charge_i, params.elec_charge_j, params.elec_is_1_4, 0,
                                                   params.elec_count);
        }

        // Bond stretching
        if (params.bond_count > 0) {
            energy += compute_bond_energy(coord_x, coord_y, coord_z, params.bond_atom_i, params.bond_atom_j,
                                          params.bond_kb, params.bond_r0, 0, params.bond_count);
        }

        // Angle bending
        if (params.angle_count > 0) {
            energy +=
                compute_angle_energy(coord_x, coord_y, coord_z, params.angle_atom_i, params.angle_atom_j,
                                     params.angle_atom_k, params.angle_ka, params.angle_theta0, 0, params.angle_count);
        }

        // Torsion
        if (params.torsion_count > 0) {
            energy += compute_torsion_energy(coord_x, coord_y, coord_z, params.torsion_atom_i, params.torsion_atom_j,
                                             params.torsion_atom_k, params.torsion_atom_l, params.torsion_v1,
                                             params.torsion_v2, params.torsion_v3, 0, params.torsion_count);
        }

        // Stretch-bend coupling
        if (params.stbn_count > 0) {
            energy += compute_stretch_bend_energy(coord_x, coord_y, coord_z, params.stbn_atom_i, params.stbn_atom_j,
                                                  params.stbn_atom_k, params.stbn_kba_ijk, params.stbn_kba_kji,
                                                  params.stbn_r0_ij, params.stbn_r0_kj, params.stbn_theta0, 0,
                                                  params.stbn_count);
        }

        // Out-of-plane bending
        if (params.oop_count > 0) {
            energy += compute_oop_energy(coord_x, coord_y, coord_z, params.oop_atom_i, params.oop_atom_j,
                                         params.oop_atom_k, params.oop_atom_l, params.oop_koop, 0, params.oop_count);
        }

        return energy;
    };

    auto compute_total_gradient_ = [&]() -> void {
        scalar_t* grad_x = state.grad;
        scalar_t* grad_y = state.grad + state.n_atoms;
        scalar_t* grad_z = state.grad + 2 * state.n_atoms;

        // Zero gradients
        for (std::size_t i = 0; i < state.n_atoms; ++i) {
            grad_x[i] = zero;
            grad_y[i] = zero;
            grad_z[i] = zero;
        }

        // VdW
        if (params.vdw_count > 0) {
            compute_vdw_gradient(coord_x, coord_y, coord_z, params.vdw_pair_i, params.vdw_pair_j, params.vdw_epsilon,
                                 params.vdw_r_star, 0, params.vdw_count, grad_x, grad_y, grad_z);
        }

        // Electrostatic
        if (params.elec_count > 0) {
            compute_electrostatic_gradient(coord_x, coord_y, coord_z, params.elec_pair_i, params.elec_pair_j,
                                           params.elec_charge_i, params.elec_charge_j, params.elec_is_1_4, 0,
                                           params.elec_count, static_cast<scalar_t>(scale_1_4_default), grad_x, grad_y,
                                           grad_z);
        }

        // Bond stretching
        if (params.bond_count > 0) {
            compute_bond_gradient(coord_x, coord_y, coord_z, params.bond_atom_i, params.bond_atom_j, params.bond_kb,
                                  params.bond_r0, 0, params.bond_count, grad_x, grad_y, grad_z);
        }

        // Angle bending
        if (params.angle_count > 0) {
            compute_angle_gradient(coord_x, coord_y, coord_z, params.angle_atom_i, params.angle_atom_j,
                                   params.angle_atom_k, params.angle_ka, params.angle_theta0, 0, params.angle_count,
                                   grad_x, grad_y, grad_z);
        }

        // Torsion
        if (params.torsion_count > 0) {
            compute_torsion_gradient(coord_x, coord_y, coord_z, params.torsion_atom_i, params.torsion_atom_j,
                                     params.torsion_atom_k, params.torsion_atom_l, params.torsion_v1, params.torsion_v2,
                                     params.torsion_v3, 0, params.torsion_count, grad_x, grad_y, grad_z);
        }

        // Stretch-bend coupling
        if (params.stbn_count > 0) {
            compute_stretch_bend_gradient(coord_x, coord_y, coord_z, params.stbn_atom_i, params.stbn_atom_j,
                                          params.stbn_atom_k, params.stbn_kba_ijk, params.stbn_kba_kji,
                                          params.stbn_r0_ij, params.stbn_r0_kj, params.stbn_theta0, 0,
                                          params.stbn_count, grad_x, grad_y, grad_z);
        }

        // Out-of-plane bending
        if (params.oop_count > 0) {
            compute_oop_gradient(coord_x, coord_y, coord_z, params.oop_atom_i, params.oop_atom_j, params.oop_atom_k,
                                 params.oop_atom_l, params.oop_koop, 0, params.oop_count, grad_x, grad_y, grad_z);
        }
    };

    auto dot_product_ = [](scalar_t const* a, scalar_t const* b, std::size_t n) -> scalar_t {
        scalar_t sum = static_cast<scalar_t>(0.0);
        for (std::size_t i = 0; i < n; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    };

    // Initialize gradient
    compute_total_gradient_();
    scalar_t rms_grad = compute_rms_gradient_();

    // Check initial convergence
    if (rms_grad < grad_threshold) {
        result.status = 0;
        result.iterations = 0;
        result.final_energy = static_cast<double>(compute_total_energy_());
        result.rms_gradient = static_cast<double>(rms_grad);
        return result;
    }

    // Main optimization loop
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Two-loop recursion
        for (std::size_t i = 0; i < n_coords; ++i) {
            state.q[i] = state.grad[i];
        }

        std::size_t const history_size = (state.k < state.m) ? state.k : state.m;

        // First loop (backward)
        for (std::size_t i = 0; i < history_size; ++i) {
            std::size_t const idx = (state.k - 1 - i) % state.m;
            scalar_t const* s_i = state.s_history + idx * n_coords;
            scalar_t const* y_i = state.y_history + idx * n_coords;
            state.alpha[idx] = state.rho[idx] * dot_product_(s_i, state.q, n_coords);
            for (std::size_t j = 0; j < n_coords; ++j) {
                state.q[j] -= state.alpha[idx] * y_i[j];
            }
        }

        // Initial Hessian approximation
        scalar_t gamma = one;
        if (history_size > 0) {
            std::size_t const newest = (state.k - 1) % state.m;
            scalar_t const* s_newest = state.s_history + newest * n_coords;
            scalar_t const* y_newest = state.y_history + newest * n_coords;
            gamma = dot_product_(s_newest, y_newest, n_coords) / dot_product_(y_newest, y_newest, n_coords);
        }

        for (std::size_t i = 0; i < n_coords; ++i) {
            state.direction[i] = gamma * state.q[i];
        }

        // Second loop (forward)
        for (std::size_t i = history_size; i > 0; --i) {
            std::size_t const idx = (state.k - i) % state.m;
            scalar_t const* s_i = state.s_history + idx * n_coords;
            scalar_t const* y_i = state.y_history + idx * n_coords;
            scalar_t const beta = state.rho[idx] * dot_product_(y_i, state.direction, n_coords);
            for (std::size_t j = 0; j < n_coords; ++j) {
                state.direction[j] += s_i[j] * (state.alpha[idx] - beta);
            }
        }

        // Negate for descent direction
        for (std::size_t i = 0; i < n_coords; ++i) {
            state.direction[i] = -state.direction[i];
        }

        // Backtracking line search
        scalar_t const energy_old = compute_total_energy_();
        scalar_t grad_dot_dir = dot_product_(state.grad, state.direction, n_coords);

        // If direction is not descent, reset history and use steepest descent
        if (grad_dot_dir >= zero) {
            state.k = 0;
            for (std::size_t i = 0; i < n_coords; ++i) {
                state.direction[i] = -state.grad[i];
            }
            grad_dot_dir = dot_product_(state.grad, state.direction, n_coords);
            if (grad_dot_dir >= zero) {
                result.status = 0; // Converged
                result.iterations = iter;
                result.final_energy = static_cast<double>(energy_old);
                result.rms_gradient = static_cast<double>(rms_grad);
                return result;
            }
        }

        scalar_t alpha = alpha_init;
        scalar_t energy_new = zero;
        int ls_iter = 0;
        int const max_ls_iter = 20;

        scalar_t* backup_x = state.coord_backup;
        scalar_t* backup_y = state.coord_backup + state.n_atoms;
        scalar_t* backup_z = state.coord_backup + 2 * state.n_atoms;

        for (std::size_t i = 0; i < state.n_atoms; ++i) {
            backup_x[i] = coord_x[i];
            backup_y[i] = coord_y[i];
            backup_z[i] = coord_z[i];
        }

        for (; ls_iter < max_ls_iter; ++ls_iter) {
            for (std::size_t i = 0; i < state.n_atoms; ++i) {
                coord_x[i] = backup_x[i] + alpha * state.direction[i];
                coord_y[i] = backup_y[i] + alpha * state.direction[state.n_atoms + i];
                coord_z[i] = backup_z[i] + alpha * state.direction[2 * state.n_atoms + i];
            }
            energy_new = compute_total_energy_();
            if (energy_new <= energy_old + c1 * alpha * grad_dot_dir)
                break;
            alpha *= rho_ls;
        }

        if (ls_iter >= max_ls_iter) {
            // Line search failed - restore coordinates and reset L-BFGS history
            for (std::size_t i = 0; i < state.n_atoms; ++i) {
                coord_x[i] = backup_x[i];
                coord_y[i] = backup_y[i];
                coord_z[i] = backup_z[i];
            }
            state.k = 0;
            continue;
        }

        // Save old gradient
        for (std::size_t i = 0; i < n_coords; ++i) {
            state.grad_prev[i] = state.grad[i];
        }

        // Compute new gradient
        compute_total_gradient_();
        rms_grad = compute_rms_gradient_();

        // Update history (only if curvature condition y·s > 0 is satisfied)
        std::size_t const hist_idx = state.k % state.m;
        scalar_t* s_k = state.s_history + hist_idx * n_coords;
        scalar_t* y_k = state.y_history + hist_idx * n_coords;

        for (std::size_t i = 0; i < n_coords; ++i) {
            s_k[i] = alpha * state.direction[i];
            y_k[i] = state.grad[i] - state.grad_prev[i];
        }

        scalar_t const ys = dot_product_(y_k, s_k, n_coords);
        if (ys > static_cast<scalar_t>(1e-10)) {
            // Curvature condition satisfied - update history
            state.rho[hist_idx] = one / ys;
            state.k++;
        }
        // If curvature condition not satisfied, skip history update (L-BFGS will use steepest descent)
        result.iterations = iter + 1;

        if (rms_grad < grad_threshold) {
            result.status = 0;
            result.final_energy = static_cast<double>(energy_new);
            result.rms_gradient = static_cast<double>(rms_grad);
            return result;
        }
    }

    result.status = 1;
    result.final_energy = static_cast<double>(compute_total_energy_());
    result.rms_gradient = static_cast<double>(rms_grad);
    return result;
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
 * @brief Compute van der Waals gradient using AVX-512 with 16-way parallelism.
 *
 * Accumulates forces into grad arrays using scatter operations.
 * Data must be padded to multiple of 16 pairs.
 */
inline void compute_vdw_gradient_skylake(float const* coord_x, float const* coord_y, float const* coord_z,
                                         std::int32_t const* pair_atom_i, std::int32_t const* pair_atom_j,
                                         float const* pair_epsilon, float const* pair_equilibrium_distance,
                                         std::size_t pair_start, std::size_t pair_count, float* grad_x, float* grad_y,
                                         float* grad_z) {
    __m512 const delta_vec = _mm512_set1_ps(vdw_delta);
    __m512 const gamma_vec = _mm512_set1_ps(vdw_gamma);
    __m512 const one_plus_delta_vec = _mm512_set1_ps(1.0f + vdw_delta);
    __m512 const one_plus_gamma_vec = _mm512_set1_ps(1.0f + vdw_gamma);
    __m512 const two_vec = _mm512_set1_ps(2.0f);
    __m512 const seven_vec = _mm512_set1_ps(7.0f);

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

        // Buffered 14-7 potential terms
        __m512 const eq_dist_7 = pow7_(eq_dist);
        __m512 const dist_7 = pow7_(dist);

        __m512 const rep_denom = _mm512_fmadd_ps(delta_vec, eq_dist, dist);
        __m512 const rep_ratio = _mm512_mul_ps(one_plus_delta_vec, _mm512_mul_ps(eq_dist, _mm512_rcp14_ps(rep_denom)));
        __m512 const rep_term = pow7_(rep_ratio);

        __m512 const att_denom = _mm512_fmadd_ps(gamma_vec, eq_dist_7, dist_7);
        __m512 const att_term = _mm512_sub_ps(
            _mm512_mul_ps(one_plus_gamma_vec, _mm512_mul_ps(eq_dist_7, _mm512_rcp14_ps(att_denom))), two_vec);

        // Gradient computation
        __m512 const rep_6 = _mm512_mul_ps(rep_term, _mm512_rcp14_ps(rep_ratio));
        __m512 const rep_denom_sq = _mm512_mul_ps(rep_denom, rep_denom);
        __m512 const d_repulsive = _mm512_mul_ps(
            _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(-7.0f), rep_6), _mm512_mul_ps(one_plus_delta_vec, eq_dist)),
            _mm512_rcp14_ps(rep_denom_sq));

        __m512 const dist_6 = _mm512_mul_ps(dist_7, _mm512_rcp14_ps(dist));
        __m512 const att_denom_sq = _mm512_mul_ps(att_denom, att_denom);
        __m512 const d_attractive = _mm512_mul_ps(
            _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(-7.0f), one_plus_gamma_vec), _mm512_mul_ps(eq_dist_7, dist_6)),
            _mm512_rcp14_ps(att_denom_sq));

        __m512 const dE_dr =
            _mm512_mul_ps(eps, _mm512_fmadd_ps(d_repulsive, att_term, _mm512_mul_ps(rep_term, d_attractive)));

        // Gradient components: dE/dx = dE/dr * dx/r
        __m512 const fx = _mm512_mul_ps(_mm512_mul_ps(dE_dr, dx), inv_dist);
        __m512 const fy = _mm512_mul_ps(_mm512_mul_ps(dE_dr, dy), inv_dist);
        __m512 const fz = _mm512_mul_ps(_mm512_mul_ps(dE_dr, dz), inv_dist);

        // Accumulate forces via scalar loop (scatter is slow)
        alignas(64) float fx_arr[16], fy_arr[16], fz_arr[16];
        alignas(64) std::int32_t idx_i_arr[16], idx_j_arr[16];
        _mm512_store_ps(fx_arr, fx);
        _mm512_store_ps(fy_arr, fy);
        _mm512_store_ps(fz_arr, fz);
        _mm512_store_si512(idx_i_arr, idx_i);
        _mm512_store_si512(idx_j_arr, idx_j);

        for (std::size_t k = 0; k < 16; ++k) {
            std::int32_t const i = idx_i_arr[k];
            std::int32_t const j = idx_j_arr[k];
            grad_x[i] -= fx_arr[k];
            grad_y[i] -= fy_arr[k];
            grad_z[i] -= fz_arr[k];
            grad_x[j] += fx_arr[k];
            grad_y[j] += fy_arr[k];
            grad_z[j] += fz_arr[k];
        }
    }
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

/**
 * @brief Compute electrostatic gradient using AVX-512 with 16-way parallelism.
 *
 * Accumulates forces into grad_x/y/z arrays using scatter operations.
 * Data must be padded to multiple of 16 pairs.
 */
inline void compute_electrostatic_gradient_skylake(float const* coord_x, float const* coord_y, float const* coord_z,
                                                   std::int32_t const* pair_atom_i, std::int32_t const* pair_atom_j,
                                                   float const* pair_charge_i, float const* pair_charge_j,
                                                   std::int32_t const* pair_is_1_4, std::size_t pair_start,
                                                   std::size_t pair_count, float scaling_1_4, float* grad_x,
                                                   float* grad_y, float* grad_z) {
    __m512 const coulomb_vec = _mm512_set1_ps(elec_constant);
    __m512 const buffer_vec = _mm512_set1_ps(elec_buffer);
    __m512 const one_vec = _mm512_set1_ps(1.0f);
    __m512 const scale_1_4_vec = _mm512_set1_ps(scaling_1_4);
    __m512 const scale_diff_vec = _mm512_sub_ps(scale_1_4_vec, one_vec);

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

        // dE/dr = -332.0716 * qi * qj / (r + 0.05)^2
        __m512 const effective_dist_sq = _mm512_mul_ps(effective_dist, effective_dist);
        __m512 const grad_magnitude =
            _mm512_mul_ps(_mm512_mul_ps(coulomb_vec, charge_product), _mm512_rcp14_ps(effective_dist_sq));

        // Branchless 1-4 scaling
        __m512 const scale = _mm512_fmadd_ps(is_1_4, scale_diff_vec, one_vec);
        __m512 const scaled_grad = _mm512_mul_ps(grad_magnitude, scale);

        // Gradient components: dE/dx = dE/dr * dx/r
        __m512 const fx = _mm512_mul_ps(_mm512_mul_ps(scaled_grad, dx), inv_dist);
        __m512 const fy = _mm512_mul_ps(_mm512_mul_ps(scaled_grad, dy), inv_dist);
        __m512 const fz = _mm512_mul_ps(_mm512_mul_ps(scaled_grad, dz), inv_dist);

        // Accumulate gradients (Newton's 3rd law: grad_i = -grad_j)
        // Note: Using serial accumulation to avoid race conditions
        alignas(64) std::int32_t atom_i_arr[16];
        alignas(64) std::int32_t atom_j_arr[16];
        alignas(64) float fx_arr[16], fy_arr[16], fz_arr[16];

        _mm512_store_si512(atom_i_arr, idx_i);
        _mm512_store_si512(atom_j_arr, idx_j);
        _mm512_store_ps(fx_arr, fx);
        _mm512_store_ps(fy_arr, fy);
        _mm512_store_ps(fz_arr, fz);

        for (std::size_t k = 0; k < 16; ++k) {
            grad_x[atom_i_arr[k]] -= fx_arr[k];
            grad_y[atom_i_arr[k]] -= fy_arr[k];
            grad_z[atom_i_arr[k]] -= fz_arr[k];
            grad_x[atom_j_arr[k]] += fx_arr[k];
            grad_y[atom_j_arr[k]] += fy_arr[k];
            grad_z[atom_j_arr[k]] += fz_arr[k];
        }
    }
}

/**
 * @brief L-BFGS optimizer using AVX-512 for vectorized operations.
 *
 * Vectorizes dot products and array operations for improved performance.
 * Falls back to serial gradient computation for force accumulation.
 */
inline optimize_result_t lbfgs_optimize_skylake(float* coord_x, float* coord_y, float* coord_z,
                                                std::int32_t const* vdw_pair_i, std::int32_t const* vdw_pair_j,
                                                float const* vdw_epsilon, float const* vdw_r_star,
                                                std::size_t vdw_pair_count, std::int32_t const* elec_pair_i,
                                                std::int32_t const* elec_pair_j, float const* elec_charge_i,
                                                float const* elec_charge_j, std::int32_t const* elec_is_1_4,
                                                std::size_t elec_pair_count, int max_iterations, float grad_threshold,
                                                lbfgs_state_t<float>& state) {
    // All arrays are padded to SIMD width (16 floats) - no tail handling needed
    std::size_t const n_atoms = state.n_atoms;
    std::size_t const n_coords = 3 * n_atoms;
    std::size_t const atom_chunks = n_atoms / 16;
    std::size_t const coord_chunks = n_coords / 16;
    optimize_result_t result{-1, 0, 0.0, 0.0};

    // Backup pointers into pre-allocated state buffer
    float* backup_x = state.coord_backup;
    float* backup_y = state.coord_backup + n_atoms;
    float* backup_z = state.coord_backup + 2 * n_atoms;

    // Vectorized dot product (no tail - arrays are padded)
    auto dot_ = [](float const* a, float const* b, std::size_t chunks) -> float {
        __m512 acc = _mm512_setzero_ps();
        for (std::size_t i = 0; i < chunks; ++i)
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i * 16), _mm512_loadu_ps(b + i * 16), acc);
        return _mm512_reduce_add_ps(acc);
    };

    // RMS gradient magnitude
    auto rms_grad_ = [&]() -> float {
        return std::sqrt(dot_(state.grad, state.grad, coord_chunks) / static_cast<float>(n_coords));
    };

    // Total energy (data is padded - call AVX-512 kernels directly)
    auto energy_ = [&]() -> float {
        return compute_vdw_energy_skylake(coord_x, coord_y, coord_z, vdw_pair_i, vdw_pair_j, vdw_epsilon, vdw_r_star, 0,
                                          vdw_pair_count) +
               compute_electrostatic_energy_skylake(coord_x, coord_y, coord_z, elec_pair_i, elec_pair_j, elec_charge_i,
                                                    elec_charge_j, elec_is_1_4, 0, elec_pair_count);
    };

    // Compute gradient (zeros then accumulates)
    auto gradient_ = [&]() -> void {
        __m512 const zero = _mm512_setzero_ps();
        for (std::size_t i = 0; i < atom_chunks; ++i) {
            _mm512_storeu_ps(state.grad + i * 16, zero);
            _mm512_storeu_ps(state.grad + n_atoms + i * 16, zero);
            _mm512_storeu_ps(state.grad + 2 * n_atoms + i * 16, zero);
        }
        compute_vdw_gradient_skylake(coord_x, coord_y, coord_z, vdw_pair_i, vdw_pair_j, vdw_epsilon, vdw_r_star, 0,
                                     vdw_pair_count, state.grad, state.grad + n_atoms, state.grad + 2 * n_atoms);
        compute_electrostatic_gradient_skylake(coord_x, coord_y, coord_z, elec_pair_i, elec_pair_j, elec_charge_i,
                                               elec_charge_j, elec_is_1_4, 0, elec_pair_count, scale_1_4_default,
                                               state.grad, state.grad + n_atoms, state.grad + 2 * n_atoms);
    };

    // Initialize
    gradient_();
    float rms = rms_grad_();
    if (rms < grad_threshold) {
        result.status = 0;
        result.iterations = 0;
        result.final_energy = static_cast<double>(energy_());
        result.rms_gradient = static_cast<double>(rms);
        return result;
    }

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Copy gradient to q
        for (std::size_t i = 0; i < coord_chunks; ++i)
            _mm512_storeu_ps(state.q + i * 16, _mm512_loadu_ps(state.grad + i * 16));

        // Two-loop recursion: first loop (backward)
        std::size_t const hist_size = (state.k < state.m) ? state.k : state.m;
        for (std::size_t h = 0; h < hist_size; ++h) {
            std::size_t const idx = (state.k - 1 - h) % state.m;
            float const* s = state.s_history + idx * n_coords;
            float const* y = state.y_history + idx * n_coords;
            state.alpha[idx] = state.rho[idx] * dot_(s, state.q, coord_chunks);
            __m512 const a = _mm512_set1_ps(state.alpha[idx]);
            for (std::size_t i = 0; i < coord_chunks; ++i)
                _mm512_storeu_ps(state.q + i * 16,
                                 _mm512_fnmadd_ps(a, _mm512_loadu_ps(y + i * 16), _mm512_loadu_ps(state.q + i * 16)));
        }

        // Initial Hessian scaling
        float gamma = 1.0f;
        if (hist_size > 0) {
            std::size_t const idx = (state.k - 1) % state.m;
            float const* s = state.s_history + idx * n_coords;
            float const* y = state.y_history + idx * n_coords;
            gamma = dot_(s, y, coord_chunks) / dot_(y, y, coord_chunks);
        }
        __m512 const g = _mm512_set1_ps(gamma);
        for (std::size_t i = 0; i < coord_chunks; ++i)
            _mm512_storeu_ps(state.direction + i * 16, _mm512_mul_ps(g, _mm512_loadu_ps(state.q + i * 16)));

        // Second loop (forward)
        for (std::size_t h = hist_size; h > 0; --h) {
            std::size_t const idx = (state.k - h) % state.m;
            float const* s = state.s_history + idx * n_coords;
            float const* y = state.y_history + idx * n_coords;
            float const beta = state.rho[idx] * dot_(y, state.direction, coord_chunks);
            __m512 const diff = _mm512_set1_ps(state.alpha[idx] - beta);
            for (std::size_t i = 0; i < coord_chunks; ++i)
                _mm512_storeu_ps(state.direction + i * 16, _mm512_fmadd_ps(_mm512_loadu_ps(s + i * 16), diff,
                                                                           _mm512_loadu_ps(state.direction + i * 16)));
        }

        // Negate direction (descent)
        __m512 const zero = _mm512_setzero_ps();
        for (std::size_t i = 0; i < coord_chunks; ++i)
            _mm512_storeu_ps(state.direction + i * 16, _mm512_sub_ps(zero, _mm512_loadu_ps(state.direction + i * 16)));

        // Line search
        float const e_old = energy_();
        float g_dot_d = dot_(state.grad, state.direction, coord_chunks);

        // If direction is not descent, reset history and use steepest descent
        if (g_dot_d >= 0.0f) {
            state.k = 0;
            // direction = -gradient
            for (std::size_t i = 0; i < coord_chunks; ++i)
                _mm512_storeu_ps(state.direction + i * 16, _mm512_sub_ps(zero, _mm512_loadu_ps(state.grad + i * 16)));
            g_dot_d = dot_(state.grad, state.direction, coord_chunks);
            if (g_dot_d >= 0.0f) {
                result.status = 0; // Converged
                result.iterations = iter;
                result.final_energy = static_cast<double>(e_old);
                result.rms_gradient = static_cast<double>(rms);
                return result;
            }
        }

        // Backup coordinates
        for (std::size_t i = 0; i < atom_chunks; ++i) {
            _mm512_storeu_ps(backup_x + i * 16, _mm512_loadu_ps(coord_x + i * 16));
            _mm512_storeu_ps(backup_y + i * 16, _mm512_loadu_ps(coord_y + i * 16));
            _mm512_storeu_ps(backup_z + i * 16, _mm512_loadu_ps(coord_z + i * 16));
        }

        float alpha = 1.0f; // L-BFGS direction is pre-scaled
        float e_new = 0.0f;
        int ls;
        for (ls = 0; ls < 20; ++ls) {
            __m512 const av = _mm512_set1_ps(alpha);
            for (std::size_t i = 0; i < atom_chunks; ++i) {
                _mm512_storeu_ps(coord_x + i * 16, _mm512_fmadd_ps(av, _mm512_loadu_ps(state.direction + i * 16),
                                                                   _mm512_loadu_ps(backup_x + i * 16)));
                _mm512_storeu_ps(coord_y + i * 16,
                                 _mm512_fmadd_ps(av, _mm512_loadu_ps(state.direction + n_atoms + i * 16),
                                                 _mm512_loadu_ps(backup_y + i * 16)));
                _mm512_storeu_ps(coord_z + i * 16,
                                 _mm512_fmadd_ps(av, _mm512_loadu_ps(state.direction + 2 * n_atoms + i * 16),
                                                 _mm512_loadu_ps(backup_z + i * 16)));
            }
            e_new = energy_();
            if (e_new <= e_old + 1e-4f * alpha * g_dot_d)
                break;
            alpha *= 0.5f;
        }

        if (ls >= 20) {
            result.status = -1;
            result.iterations = iter;
            result.final_energy = static_cast<double>(e_new);
            result.rms_gradient = static_cast<double>(rms);
            return result;
        }

        // Save gradient, compute new gradient
        for (std::size_t i = 0; i < coord_chunks; ++i)
            _mm512_storeu_ps(state.grad_prev + i * 16, _mm512_loadu_ps(state.grad + i * 16));
        gradient_();
        rms = rms_grad_();

        // Update history
        std::size_t const hist_idx = state.k % state.m;
        float* s_k = state.s_history + hist_idx * n_coords;
        float* y_k = state.y_history + hist_idx * n_coords;
        __m512 const av = _mm512_set1_ps(alpha);
        for (std::size_t i = 0; i < coord_chunks; ++i) {
            _mm512_storeu_ps(s_k + i * 16, _mm512_mul_ps(av, _mm512_loadu_ps(state.direction + i * 16)));
            _mm512_storeu_ps(y_k + i * 16, _mm512_sub_ps(_mm512_loadu_ps(state.grad + i * 16),
                                                         _mm512_loadu_ps(state.grad_prev + i * 16)));
        }
        state.rho[hist_idx] = 1.0f / dot_(y_k, s_k, coord_chunks);
        state.k++;
        result.iterations = iter + 1;

        if (rms < grad_threshold) {
            result.status = 0;
            result.final_energy = static_cast<double>(e_new);
            result.rms_gradient = static_cast<double>(rms);
            return result;
        }
    }

    result.status = 1;
    result.final_energy = static_cast<double>(energy_());
    result.rms_gradient = static_cast<double>(rms);
    return result;
}

/**
 * @brief Compute VdW energy using precomputed constants (fast path).
 *
 * Eliminates redundant pow7 and constant multiplications by using
 * pre-calculated values. ~2x fewer FLOPs than compute_vdw_energy_skylake.
 */
inline float compute_vdw_energy_skylake_fast(float const* coord_x, float const* coord_y, float const* coord_z,
                                             std::int32_t const* pair_atom_i, std::int32_t const* pair_atom_j,
                                             float const* pair_epsilon, vdw_precomputed_t<float> const& pre,
                                             std::size_t pair_start, std::size_t pair_count) {
    __m512 const two_vec = _mm512_set1_ps(2.0f);
    __m512 accum = _mm512_setzero_ps();

    std::size_t const simd_iterations = pair_count / 16;

    for (std::size_t iter = 0; iter < simd_iterations; ++iter) {
        std::size_t const base = pair_start + iter * 16;

        __m512i const idx_i = _mm512_loadu_si512(&pair_atom_i[base]);
        __m512i const idx_j = _mm512_loadu_si512(&pair_atom_j[base]);

        __m512 const xi = _mm512_i32gather_ps(idx_i, coord_x, 4);
        __m512 const yi = _mm512_i32gather_ps(idx_i, coord_y, 4);
        __m512 const zi = _mm512_i32gather_ps(idx_i, coord_z, 4);
        __m512 const xj = _mm512_i32gather_ps(idx_j, coord_x, 4);
        __m512 const yj = _mm512_i32gather_ps(idx_j, coord_y, 4);
        __m512 const zj = _mm512_i32gather_ps(idx_j, coord_z, 4);

        __m512 const eps = _mm512_loadu_ps(&pair_epsilon[base]);

        // Load precomputed constants (no pow7, no multiplications!)
        __m512 const eq_dist_7 = _mm512_loadu_ps(&pre.eq_dist_7[base]);
        __m512 const delta_eq_dist = _mm512_loadu_ps(&pre.delta_eq_dist[base]);
        __m512 const one_plus_delta_eq_dist = _mm512_loadu_ps(&pre.one_plus_delta_eq_dist[base]);
        __m512 const gamma_eq_dist_7 = _mm512_loadu_ps(&pre.gamma_eq_dist_7[base]);
        __m512 const one_plus_gamma_eq_dist_7 = _mm512_loadu_ps(&pre.one_plus_gamma_eq_dist_7[base]);

        __m512 const dx = _mm512_sub_ps(xj, xi);
        __m512 const dy = _mm512_sub_ps(yj, yi);
        __m512 const dz = _mm512_sub_ps(zj, zi);
        __m512 const dz_sq = _mm512_mul_ps(dz, dz);
        __m512 const dist_sq = _mm512_fmadd_ps(dx, dx, _mm512_fmadd_ps(dy, dy, dz_sq));
        __m512 const inv_dist = _mm512_rsqrt14_ps(dist_sq);
        __m512 const dist = _mm512_mul_ps(dist_sq, inv_dist);

        // Only dist_7 needs pow7 (distance-dependent)
        __m512 const dist_7 = pow7_(dist);

        // rep_denom = dist + delta*eq_dist (precomputed)
        __m512 const rep_denom = _mm512_add_ps(dist, delta_eq_dist);
        // rep_ratio = (1+delta)*eq_dist / rep_denom (precomputed numerator)
        __m512 const rep_ratio = _mm512_mul_ps(one_plus_delta_eq_dist, _mm512_rcp14_ps(rep_denom));
        __m512 const rep_term = pow7_(rep_ratio);

        // att_denom = dist_7 + gamma*eq_dist_7 (precomputed)
        __m512 const att_denom = _mm512_add_ps(dist_7, gamma_eq_dist_7);
        // att_term = (1+gamma)*eq_dist_7 / att_denom - 2 (precomputed numerator)
        __m512 const att_term =
            _mm512_sub_ps(_mm512_mul_ps(one_plus_gamma_eq_dist_7, _mm512_rcp14_ps(att_denom)), two_vec);

        __m512 const energy = _mm512_mul_ps(eps, _mm512_mul_ps(rep_term, att_term));
        accum = _mm512_add_ps(accum, energy);
    }

    return _mm512_reduce_add_ps(accum);
}

/**
 * @brief Compute VdW gradient using precomputed constants (fast path).
 */
inline void compute_vdw_gradient_skylake_fast(float const* coord_x, float const* coord_y, float const* coord_z,
                                              std::int32_t const* pair_atom_i, std::int32_t const* pair_atom_j,
                                              float const* pair_epsilon, vdw_precomputed_t<float> const& pre,
                                              std::size_t pair_start, std::size_t pair_count, float* grad_x,
                                              float* grad_y, float* grad_z) {
    __m512 const two_vec = _mm512_set1_ps(2.0f);
    __m512 const seven_vec = _mm512_set1_ps(7.0f);

    std::size_t const simd_iterations = pair_count / 16;

    for (std::size_t iter = 0; iter < simd_iterations; ++iter) {
        std::size_t const base = pair_start + iter * 16;

        __m512i const idx_i = _mm512_loadu_si512(&pair_atom_i[base]);
        __m512i const idx_j = _mm512_loadu_si512(&pair_atom_j[base]);

        __m512 const xi = _mm512_i32gather_ps(idx_i, coord_x, 4);
        __m512 const yi = _mm512_i32gather_ps(idx_i, coord_y, 4);
        __m512 const zi = _mm512_i32gather_ps(idx_i, coord_z, 4);
        __m512 const xj = _mm512_i32gather_ps(idx_j, coord_x, 4);
        __m512 const yj = _mm512_i32gather_ps(idx_j, coord_y, 4);
        __m512 const zj = _mm512_i32gather_ps(idx_j, coord_z, 4);

        __m512 const eps = _mm512_loadu_ps(&pair_epsilon[base]);

        // Load precomputed constants
        __m512 const eq_dist_7 = _mm512_loadu_ps(&pre.eq_dist_7[base]);
        __m512 const delta_eq_dist = _mm512_loadu_ps(&pre.delta_eq_dist[base]);
        __m512 const one_plus_delta_eq_dist = _mm512_loadu_ps(&pre.one_plus_delta_eq_dist[base]);
        __m512 const gamma_eq_dist_7 = _mm512_loadu_ps(&pre.gamma_eq_dist_7[base]);
        __m512 const one_plus_gamma_eq_dist_7 = _mm512_loadu_ps(&pre.one_plus_gamma_eq_dist_7[base]);

        __m512 const dx = _mm512_sub_ps(xj, xi);
        __m512 const dy = _mm512_sub_ps(yj, yi);
        __m512 const dz = _mm512_sub_ps(zj, zi);
        __m512 const dz_sq = _mm512_mul_ps(dz, dz);
        __m512 const dist_sq = _mm512_fmadd_ps(dx, dx, _mm512_fmadd_ps(dy, dy, dz_sq));
        __m512 const inv_dist = _mm512_rsqrt14_ps(dist_sq);
        __m512 const dist = _mm512_mul_ps(dist_sq, inv_dist);

        __m512 const dist_7 = pow7_(dist);

        __m512 const rep_denom = _mm512_add_ps(dist, delta_eq_dist);
        __m512 const rep_ratio = _mm512_mul_ps(one_plus_delta_eq_dist, _mm512_rcp14_ps(rep_denom));
        __m512 const rep_term = pow7_(rep_ratio);

        __m512 const att_denom = _mm512_add_ps(dist_7, gamma_eq_dist_7);
        __m512 const att_term =
            _mm512_sub_ps(_mm512_mul_ps(one_plus_gamma_eq_dist_7, _mm512_rcp14_ps(att_denom)), two_vec);

        // Gradient computation
        __m512 const rep_6 = _mm512_mul_ps(rep_term, _mm512_rcp14_ps(rep_ratio));
        __m512 const rep_denom_sq = _mm512_mul_ps(rep_denom, rep_denom);
        __m512 const d_repulsive =
            _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(-7.0f), rep_6), one_plus_delta_eq_dist),
                          _mm512_rcp14_ps(rep_denom_sq));

        __m512 const dist_6 = _mm512_mul_ps(dist_7, _mm512_rcp14_ps(dist));
        __m512 const att_denom_sq = _mm512_mul_ps(att_denom, att_denom);
        __m512 const d_attractive =
            _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(-7.0f), one_plus_gamma_eq_dist_7), dist_6),
                          _mm512_rcp14_ps(att_denom_sq));

        __m512 const dE_dr =
            _mm512_mul_ps(eps, _mm512_fmadd_ps(d_repulsive, att_term, _mm512_mul_ps(rep_term, d_attractive)));

        // Gradient components: dE/dx = dE/dr * dx/r
        __m512 const fx = _mm512_mul_ps(_mm512_mul_ps(dE_dr, dx), inv_dist);
        __m512 const fy = _mm512_mul_ps(_mm512_mul_ps(dE_dr, dy), inv_dist);
        __m512 const fz = _mm512_mul_ps(_mm512_mul_ps(dE_dr, dz), inv_dist);

        // Accumulate via scalar loop
        alignas(64) float fx_arr[16], fy_arr[16], fz_arr[16];
        alignas(64) std::int32_t idx_i_arr[16], idx_j_arr[16];
        _mm512_store_ps(fx_arr, fx);
        _mm512_store_ps(fy_arr, fy);
        _mm512_store_ps(fz_arr, fz);
        _mm512_store_si512(idx_i_arr, idx_i);
        _mm512_store_si512(idx_j_arr, idx_j);

        for (std::size_t k = 0; k < 16; ++k) {
            std::int32_t const i = idx_i_arr[k];
            std::int32_t const j = idx_j_arr[k];
            grad_x[i] -= fx_arr[k];
            grad_y[i] -= fy_arr[k];
            grad_z[i] -= fz_arr[k];
            grad_x[j] += fx_arr[k];
            grad_y[j] += fy_arr[k];
            grad_z[j] += fz_arr[k];
        }
    }
}

/**
 * @brief L-BFGS optimizer using precomputed VdW constants (fast path).
 *
 * Same algorithm as lbfgs_optimize_skylake but uses precomputed constants
 * to eliminate redundant calculations in energy/gradient evaluations.
 */
inline optimize_result_t lbfgs_optimize_skylake_fast(float* coord_x, float* coord_y, float* coord_z,
                                                     std::int32_t const* vdw_pair_i, std::int32_t const* vdw_pair_j,
                                                     float const* vdw_epsilon, vdw_precomputed_t<float> const& vdw_pre,
                                                     std::size_t vdw_pair_count, std::int32_t const* elec_pair_i,
                                                     std::int32_t const* elec_pair_j, float const* elec_charge_i,
                                                     float const* elec_charge_j, std::int32_t const* elec_is_1_4,
                                                     std::size_t elec_pair_count, int max_iterations,
                                                     float grad_threshold, lbfgs_state_t<float>& state) {
    std::size_t const n_atoms = state.n_atoms;
    std::size_t const n_coords = 3 * n_atoms;
    std::size_t const atom_chunks = n_atoms / 16;
    std::size_t const coord_chunks = n_coords / 16;
    optimize_result_t result{-1, 0, 0.0, 0.0};

    float* backup_x = state.coord_backup;
    float* backup_y = state.coord_backup + n_atoms;
    float* backup_z = state.coord_backup + 2 * n_atoms;

    auto dot_ = [](float const* a, float const* b, std::size_t chunks) -> float {
        __m512 acc = _mm512_setzero_ps();
        for (std::size_t i = 0; i < chunks; ++i)
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i * 16), _mm512_loadu_ps(b + i * 16), acc);
        return _mm512_reduce_add_ps(acc);
    };

    auto rms_grad_ = [&]() -> float {
        return std::sqrt(dot_(state.grad, state.grad, coord_chunks) / static_cast<float>(n_coords));
    };

    // Fast energy using precomputed constants
    auto energy_ = [&]() -> float {
        float vdw = compute_vdw_energy_skylake_fast(coord_x, coord_y, coord_z, vdw_pair_i, vdw_pair_j, vdw_epsilon,
                                                    vdw_pre, 0, vdw_pair_count);
        if (elec_pair_count > 0) {
            vdw += compute_electrostatic_energy_skylake(coord_x, coord_y, coord_z, elec_pair_i, elec_pair_j,
                                                        elec_charge_i, elec_charge_j, elec_is_1_4, 0, elec_pair_count);
        }
        return vdw;
    };

    // Fast gradient using precomputed constants
    auto gradient_ = [&]() -> void {
        __m512 const zero = _mm512_setzero_ps();
        for (std::size_t i = 0; i < atom_chunks; ++i) {
            _mm512_storeu_ps(state.grad + i * 16, zero);
            _mm512_storeu_ps(state.grad + n_atoms + i * 16, zero);
            _mm512_storeu_ps(state.grad + 2 * n_atoms + i * 16, zero);
        }
        compute_vdw_gradient_skylake_fast(coord_x, coord_y, coord_z, vdw_pair_i, vdw_pair_j, vdw_epsilon, vdw_pre, 0,
                                          vdw_pair_count, state.grad, state.grad + n_atoms, state.grad + 2 * n_atoms);
        if (elec_pair_count > 0) {
            compute_electrostatic_gradient_skylake(coord_x, coord_y, coord_z, elec_pair_i, elec_pair_j, elec_charge_i,
                                                   elec_charge_j, elec_is_1_4, 0, elec_pair_count, scale_1_4_default,
                                                   state.grad, state.grad + n_atoms, state.grad + 2 * n_atoms);
        }
    };

    // Initialize
    gradient_();
    float rms = rms_grad_();
    if (rms < grad_threshold) {
        result.status = 0;
        result.iterations = 0;
        result.final_energy = static_cast<double>(energy_());
        result.rms_gradient = static_cast<double>(rms);
        return result;
    }

    for (int iter = 0; iter < max_iterations; ++iter) {
        for (std::size_t i = 0; i < coord_chunks; ++i)
            _mm512_storeu_ps(state.q + i * 16, _mm512_loadu_ps(state.grad + i * 16));

        std::size_t const hist_size = (state.k < state.m) ? state.k : state.m;
        for (std::size_t h = 0; h < hist_size; ++h) {
            std::size_t const idx = (state.k - 1 - h) % state.m;
            float const* s = state.s_history + idx * n_coords;
            float const* y = state.y_history + idx * n_coords;
            state.alpha[idx] = state.rho[idx] * dot_(s, state.q, coord_chunks);
            __m512 const a = _mm512_set1_ps(state.alpha[idx]);
            for (std::size_t i = 0; i < coord_chunks; ++i)
                _mm512_storeu_ps(state.q + i * 16,
                                 _mm512_fnmadd_ps(a, _mm512_loadu_ps(y + i * 16), _mm512_loadu_ps(state.q + i * 16)));
        }

        float gamma = 1.0f;
        if (hist_size > 0) {
            std::size_t const idx = (state.k - 1) % state.m;
            float const* s = state.s_history + idx * n_coords;
            float const* y = state.y_history + idx * n_coords;
            gamma = dot_(s, y, coord_chunks) / dot_(y, y, coord_chunks);
        }
        __m512 const g = _mm512_set1_ps(gamma);
        for (std::size_t i = 0; i < coord_chunks; ++i)
            _mm512_storeu_ps(state.direction + i * 16, _mm512_mul_ps(g, _mm512_loadu_ps(state.q + i * 16)));

        for (std::size_t h = hist_size; h > 0; --h) {
            std::size_t const idx = (state.k - h) % state.m;
            float const* s = state.s_history + idx * n_coords;
            float const* y = state.y_history + idx * n_coords;
            float const beta = state.rho[idx] * dot_(y, state.direction, coord_chunks);
            __m512 const diff = _mm512_set1_ps(state.alpha[idx] - beta);
            for (std::size_t i = 0; i < coord_chunks; ++i)
                _mm512_storeu_ps(state.direction + i * 16, _mm512_fmadd_ps(_mm512_loadu_ps(s + i * 16), diff,
                                                                           _mm512_loadu_ps(state.direction + i * 16)));
        }

        __m512 const zero = _mm512_setzero_ps();
        for (std::size_t i = 0; i < coord_chunks; ++i)
            _mm512_storeu_ps(state.direction + i * 16, _mm512_sub_ps(zero, _mm512_loadu_ps(state.direction + i * 16)));

        float const e_old = energy_();
        float g_dot_d = dot_(state.grad, state.direction, coord_chunks);

        // If direction is not descent, reset history and use steepest descent
        if (g_dot_d >= 0.0f) {
            state.k = 0;
            for (std::size_t i = 0; i < coord_chunks; ++i)
                _mm512_storeu_ps(state.direction + i * 16, _mm512_sub_ps(zero, _mm512_loadu_ps(state.grad + i * 16)));
            g_dot_d = dot_(state.grad, state.direction, coord_chunks);
            if (g_dot_d >= 0.0f) {
                result.status = 0; // Converged
                result.iterations = iter;
                result.final_energy = static_cast<double>(e_old);
                result.rms_gradient = static_cast<double>(rms);
                return result;
            }
        }

        for (std::size_t i = 0; i < atom_chunks; ++i) {
            _mm512_storeu_ps(backup_x + i * 16, _mm512_loadu_ps(coord_x + i * 16));
            _mm512_storeu_ps(backup_y + i * 16, _mm512_loadu_ps(coord_y + i * 16));
            _mm512_storeu_ps(backup_z + i * 16, _mm512_loadu_ps(coord_z + i * 16));
        }

        float alpha = 1.0f; // L-BFGS direction is pre-scaled
        float e_new = 0.0f;
        int ls;
        for (ls = 0; ls < 20; ++ls) {
            __m512 const av = _mm512_set1_ps(alpha);
            for (std::size_t i = 0; i < atom_chunks; ++i) {
                _mm512_storeu_ps(coord_x + i * 16, _mm512_fmadd_ps(av, _mm512_loadu_ps(state.direction + i * 16),
                                                                   _mm512_loadu_ps(backup_x + i * 16)));
                _mm512_storeu_ps(coord_y + i * 16,
                                 _mm512_fmadd_ps(av, _mm512_loadu_ps(state.direction + n_atoms + i * 16),
                                                 _mm512_loadu_ps(backup_y + i * 16)));
                _mm512_storeu_ps(coord_z + i * 16,
                                 _mm512_fmadd_ps(av, _mm512_loadu_ps(state.direction + 2 * n_atoms + i * 16),
                                                 _mm512_loadu_ps(backup_z + i * 16)));
            }
            e_new = energy_();
            if (e_new <= e_old + 1e-4f * alpha * g_dot_d)
                break;
            alpha *= 0.5f;
        }

        if (ls >= 20) {
            result.status = -1;
            result.iterations = iter;
            result.final_energy = static_cast<double>(e_new);
            result.rms_gradient = static_cast<double>(rms);
            return result;
        }

        for (std::size_t i = 0; i < coord_chunks; ++i)
            _mm512_storeu_ps(state.grad_prev + i * 16, _mm512_loadu_ps(state.grad + i * 16));
        gradient_();
        rms = rms_grad_();

        std::size_t const hist_idx = state.k % state.m;
        float* s_k = state.s_history + hist_idx * n_coords;
        float* y_k = state.y_history + hist_idx * n_coords;
        __m512 const av = _mm512_set1_ps(alpha);
        for (std::size_t i = 0; i < coord_chunks; ++i) {
            _mm512_storeu_ps(s_k + i * 16, _mm512_mul_ps(av, _mm512_loadu_ps(state.direction + i * 16)));
            _mm512_storeu_ps(y_k + i * 16, _mm512_sub_ps(_mm512_loadu_ps(state.grad + i * 16),
                                                         _mm512_loadu_ps(state.grad_prev + i * 16)));
        }
        state.rho[hist_idx] = 1.0f / dot_(y_k, s_k, coord_chunks);
        state.k++;
        result.iterations = iter + 1;

        if (rms < grad_threshold) {
            result.status = 0;
            result.final_energy = static_cast<double>(e_new);
            result.rms_gradient = static_cast<double>(rms);
            return result;
        }
    }

    result.status = 1;
    result.final_energy = static_cast<double>(energy_());
    result.rms_gradient = static_cast<double>(rms);
    return result;
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
