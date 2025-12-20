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
    int status;           // 0=converged, 1=max_iter, -1=error
    int iterations;       // number of iterations performed
    double final_energy;  // final energy value
    double rms_gradient;  // RMS gradient magnitude
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

        // Force magnitude (negative gradient): F = -dE/dr
        scalar_t const force_magnitude = -dE_dr;

        // Unit vector: r_vec / |r_vec|
        scalar_t const inv_distance = static_cast<scalar_t>(1.0) / distance;
        scalar_t const fx = force_magnitude * dx * inv_distance;
        scalar_t const fy = force_magnitude * dy * inv_distance;
        scalar_t const fz = force_magnitude * dz * inv_distance;

        // Accumulate forces (Newton's 3rd law)
        grad_x[i] -= fx;
        grad_y[i] -= fy;
        grad_z[i] -= fz;
        grad_x[j] += fx;
        grad_y[j] += fy;
        grad_z[j] += fz;
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
        scalar_t const grad_magnitude = -coulomb * charge_product / effective_distance_sq;

        // Branchless 1-4 scaling
        scalar_t const is_1_4_float = (pair_is_1_4[idx] != 0) ? one : static_cast<scalar_t>(0.0);
        scalar_t const scale = one + is_1_4_float * scale_diff;
        scalar_t const scaled_grad = grad_magnitude * scale;

        // Force direction: F_j = -(dE/dr) * (r_vec / |r_vec|)
        scalar_t const inv_distance = one / distance;
        scalar_t const fx = -scaled_grad * dx * inv_distance;
        scalar_t const fy = -scaled_grad * dy * inv_distance;
        scalar_t const fz = -scaled_grad * dz * inv_distance;

        // Accumulate forces (Newton's 3rd law: F_i = -F_j)
        grad_x[i] -= fx;
        grad_y[i] -= fy;
        grad_z[i] -= fz;
        grad_x[j] += fx;
        grad_y[j] += fy;
        grad_z[j] += fz;
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
    scalar_t constexpr alpha_init = static_cast<scalar_t>(0.001);  // Initial step size
    scalar_t constexpr c1 = static_cast<scalar_t>(1e-4);           // Armijo condition constant
    scalar_t constexpr rho_ls = static_cast<scalar_t>(0.5);        // Line search backtrack factor

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
        scalar_t vdw_energy =
            compute_vdw_energy(coord_x, coord_y, coord_z, vdw_pair_i, vdw_pair_j, vdw_epsilon, vdw_r_star, 0, vdw_pair_count);
        scalar_t elec_energy = compute_electrostatic_energy(coord_x, coord_y, coord_z, elec_pair_i, elec_pair_j,
                                                             elec_charge_i, elec_charge_j, elec_is_1_4, 0, elec_pair_count);
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
        compute_electrostatic_gradient(coord_x, coord_y, coord_z, elec_pair_i, elec_pair_j, elec_charge_i, elec_charge_j,
                                        elec_is_1_4, 0, elec_pair_count, static_cast<scalar_t>(scale_1_4_default),
                                        state.grad, state.grad + state.n_atoms, state.grad + 2 * state.n_atoms);
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
        scalar_t const grad_dot_dir = dot_product_(state.grad, state.direction, n_coords);

        // Early exit if direction is not descent
        if (grad_dot_dir >= zero) {
            result.status = -1;  // Error: not a descent direction
            result.iterations = iter;
            result.final_energy = static_cast<double>(energy_old);
            result.rms_gradient = static_cast<double>(rms_grad);
            return result;
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
            if (energy_new <= energy_old + c1 * alpha * grad_dot_dir) break;
            alpha *= rho_ls;
        }

        if (ls_iter >= max_ls_iter) {
            result.status = -1;  // Error: line search failed
            result.iterations = iter;
            result.final_energy = static_cast<double>(energy_new);
            result.rms_gradient = static_cast<double>(rms_grad);
            return result;
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

        // Compute rho = 1/(y·s)
        scalar_t const y_dot_s = dot_product_(y_k, s_k, n_coords);
        state.rho[hist_idx] = one / y_dot_s;

        state.k++;
        result.iterations = iter + 1;

        // Check convergence
        if (rms_grad < grad_threshold) {
            result.status = 0;  // Converged
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
        __m512 const d_repulsive = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(-7.0f), rep_6),
                                                                _mm512_mul_ps(one_plus_delta_vec, eq_dist)),
                                                 _mm512_rcp14_ps(rep_denom_sq));

        __m512 const dist_6 = _mm512_mul_ps(dist_7, _mm512_rcp14_ps(dist));
        __m512 const att_denom_sq = _mm512_mul_ps(att_denom, att_denom);
        __m512 const d_attractive =
            _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(-7.0f), one_plus_gamma_vec),
                                        _mm512_mul_ps(eq_dist_7, dist_6)),
                          _mm512_rcp14_ps(att_denom_sq));

        __m512 const dE_dr = _mm512_mul_ps(eps, _mm512_fmadd_ps(d_repulsive, att_term, _mm512_mul_ps(rep_term, d_attractive)));

        // Force magnitude: F = -dE/dr
        __m512 const force_mag = _mm512_sub_ps(_mm512_setzero_ps(), dE_dr);

        // Force components: F_vec = force_mag * (r_vec / |r_vec|)
        __m512 const fx = _mm512_mul_ps(_mm512_mul_ps(force_mag, dx), inv_dist);
        __m512 const fy = _mm512_mul_ps(_mm512_mul_ps(force_mag, dy), inv_dist);
        __m512 const fz = _mm512_mul_ps(_mm512_mul_ps(force_mag, dz), inv_dist);

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
        __m512 const grad_magnitude_neg = _mm512_sub_ps(_mm512_setzero_ps(), grad_magnitude);

        // Branchless 1-4 scaling
        __m512 const scale = _mm512_fmadd_ps(is_1_4, scale_diff_vec, one_vec);
        __m512 const scaled_grad = _mm512_mul_ps(grad_magnitude_neg, scale);

        // Force direction: F_j = -(dE/dr) * (r_vec / |r_vec|)
        __m512 const fx = _mm512_mul_ps(_mm512_mul_ps(scaled_grad, dx), inv_dist);
        __m512 const fy = _mm512_mul_ps(_mm512_mul_ps(scaled_grad, dy), inv_dist);
        __m512 const fz = _mm512_mul_ps(_mm512_mul_ps(scaled_grad, dz), inv_dist);

        __m512 const fx_neg = _mm512_sub_ps(_mm512_setzero_ps(), fx);
        __m512 const fy_neg = _mm512_sub_ps(_mm512_setzero_ps(), fy);
        __m512 const fz_neg = _mm512_sub_ps(_mm512_setzero_ps(), fz);

        // Accumulate forces (Newton's 3rd law: F_i = -F_j)
        // Note: Using serial accumulation to avoid race conditions
        alignas(64) std::int32_t atom_i_arr[16];
        alignas(64) std::int32_t atom_j_arr[16];
        alignas(64) float fx_arr[16], fy_arr[16], fz_arr[16];

        _mm512_store_si512(atom_i_arr, idx_i);
        _mm512_store_si512(atom_j_arr, idx_j);
        _mm512_store_ps(fx_arr, fx_neg);
        _mm512_store_ps(fy_arr, fy_neg);
        _mm512_store_ps(fz_arr, fz_neg);

        for (std::size_t k = 0; k < 16; ++k) {
            grad_x[atom_i_arr[k]] += fx_arr[k];
            grad_y[atom_i_arr[k]] += fy_arr[k];
            grad_z[atom_i_arr[k]] += fz_arr[k];
        }

        _mm512_store_ps(fx_arr, fx);
        _mm512_store_ps(fy_arr, fy);
        _mm512_store_ps(fz_arr, fz);

        for (std::size_t k = 0; k < 16; ++k) {
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
        return compute_vdw_energy_skylake(coord_x, coord_y, coord_z, vdw_pair_i, vdw_pair_j,
                                          vdw_epsilon, vdw_r_star, 0, vdw_pair_count) +
               compute_electrostatic_energy_skylake(coord_x, coord_y, coord_z, elec_pair_i, elec_pair_j,
                                                    elec_charge_i, elec_charge_j, elec_is_1_4, 0, elec_pair_count);
    };

    // Compute gradient (zeros then accumulates)
    auto gradient_ = [&]() -> void {
        __m512 const zero = _mm512_setzero_ps();
        for (std::size_t i = 0; i < atom_chunks; ++i) {
            _mm512_storeu_ps(state.grad + i * 16, zero);
            _mm512_storeu_ps(state.grad + n_atoms + i * 16, zero);
            _mm512_storeu_ps(state.grad + 2 * n_atoms + i * 16, zero);
        }
        compute_vdw_gradient_skylake(coord_x, coord_y, coord_z, vdw_pair_i, vdw_pair_j, vdw_epsilon, vdw_r_star,
                                     0, vdw_pair_count, state.grad, state.grad + n_atoms, state.grad + 2 * n_atoms);
        compute_electrostatic_gradient_skylake(coord_x, coord_y, coord_z, elec_pair_i, elec_pair_j,
                                               elec_charge_i, elec_charge_j, elec_is_1_4, 0, elec_pair_count,
                                               scale_1_4_default, state.grad, state.grad + n_atoms, state.grad + 2 * n_atoms);
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
                _mm512_storeu_ps(state.direction + i * 16,
                                 _mm512_fmadd_ps(_mm512_loadu_ps(s + i * 16), diff, _mm512_loadu_ps(state.direction + i * 16)));
        }

        // Negate direction (descent)
        __m512 const zero = _mm512_setzero_ps();
        for (std::size_t i = 0; i < coord_chunks; ++i)
            _mm512_storeu_ps(state.direction + i * 16, _mm512_sub_ps(zero, _mm512_loadu_ps(state.direction + i * 16)));

        // Line search
        float const e_old = energy_();
        float const g_dot_d = dot_(state.grad, state.direction, coord_chunks);
        if (g_dot_d >= 0.0f) {
            result.status = -1;
            result.iterations = iter;
            result.final_energy = static_cast<double>(e_old);
            result.rms_gradient = static_cast<double>(rms);
            return result;
        }

        // Backup coordinates
        for (std::size_t i = 0; i < atom_chunks; ++i) {
            _mm512_storeu_ps(backup_x + i * 16, _mm512_loadu_ps(coord_x + i * 16));
            _mm512_storeu_ps(backup_y + i * 16, _mm512_loadu_ps(coord_y + i * 16));
            _mm512_storeu_ps(backup_z + i * 16, _mm512_loadu_ps(coord_z + i * 16));
        }

        float alpha = 0.001f;
        float e_new = 0.0f;
        int ls;
        for (ls = 0; ls < 20; ++ls) {
            __m512 const av = _mm512_set1_ps(alpha);
            for (std::size_t i = 0; i < atom_chunks; ++i) {
                _mm512_storeu_ps(coord_x + i * 16, _mm512_fmadd_ps(av, _mm512_loadu_ps(state.direction + i * 16),
                                                                   _mm512_loadu_ps(backup_x + i * 16)));
                _mm512_storeu_ps(coord_y + i * 16, _mm512_fmadd_ps(av, _mm512_loadu_ps(state.direction + n_atoms + i * 16),
                                                                   _mm512_loadu_ps(backup_y + i * 16)));
                _mm512_storeu_ps(coord_z + i * 16, _mm512_fmadd_ps(av, _mm512_loadu_ps(state.direction + 2 * n_atoms + i * 16),
                                                                   _mm512_loadu_ps(backup_z + i * 16)));
            }
            e_new = energy_();
            if (e_new <= e_old + 1e-4f * alpha * g_dot_d) break;
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
