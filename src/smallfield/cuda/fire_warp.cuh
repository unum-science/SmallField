/**
 *  @file src/smallfield/cuda/fire_warp.cuh
 *  @brief FIRE, Fast Inertial Relaxation Engine, warp-level optimizer.
 *
 *  Bitzek et al., PRL 97, 170201 (2006). Gradient descent with adaptive momentum, much lighter than L-BFGS:
 *  positions, velocities and gradient, three arrays, no history. The functor interface is
 *  @code
 *    float functor(positions, gradient, num_elements, lane)
 *  @endcode
 */

#pragma once

#include <cstdint> // `std::int32_t`

namespace smallfield::cuda {

/**
 *  @brief Why a FIRE stage stopped.
 *
 *  Outside the CUDA guard below so a host caller staging or reporting an outcome can name the two states.
 */
enum class fire_status_t : std::int32_t {
    /** The RMS gradient fell below the stage's threshold. */
    converged_k = 0,
    /** The iteration budget ran out first. */
    exhausted_k = 1,
};

} // namespace smallfield::cuda

#if defined(__CUDACC__)

#include "smallfield/cuda/warp_primitives.cuh"
#include "smallfield/types/views.hpp"

namespace smallfield::cuda {
using smallfield::types::coordinate_role_t;
using smallfield::types::coordinates_soa;

/** The positions a FIRE stage moves, at @p dimensions_ rows per atom. */
template <std::size_t dimensions_>
using fire_positions = coordinates_soa<coordinate_role_t::position_k, float, dimensions_>;

/** The momentum FIRE carries between steps, in the same shape and layout as the positions. */
template <std::size_t dimensions_>
using fire_velocities = coordinates_soa<coordinate_role_t::velocity_k, float, dimensions_>;

/** Where the energy functor writes what it computed, in the same shape and layout again. */
template <std::size_t dimensions_>
using fire_gradient = coordinates_soa<coordinate_role_t::gradient_k, float, dimensions_>;

/**
 *  @brief The three same-shaped arrays a FIRE stage walks together.
 *
 *  Positions, velocities and gradient share shape, layout and element type, so as three bare pointers under one
 *  count only their order tells them apart, and that order is restated at every call. Swapping the last two
 *  compiles and minimizes something else: the driver zeroes what it thinks are velocities and the functor
 *  overwrites what it thinks is the gradient. `coordinate_role_t` is an accessor, so the names cost no bytes.
 *
 *  @tparam dimensions_ Rows per atom: four while the fourth dimension is live, three afterwards.
 */
template <std::size_t dimensions_>
struct fire_scratch {
    fire_positions<dimensions_> positions;
    fire_velocities<dimensions_> velocities;
    fire_gradient<dimensions_> gradient;
};

/**
 *  @brief Outcome of a FIRE minimization stage, taken at the positions it hands back.
 *
 *  All fields are warp-uniform: `energy` and `rms_gradient` are warp-reduced inside the loop and the rest come
 *  from the uniform loop counter. The energy and the gradient norm describe the @b same geometry, the one in
 *  `positions` on return, which is why the loop stops one step short rather than taking a move it has not
 *  measured.
 *
 *  `rms_gradient` is a root mean square over @b elements, so one threshold is a different absolute norm per
 *  stage: distance geometry and the fourth-dimension crush run at four elements per atom against three for ETK
 *  and MMFF, a factor of sqrt(4/3) for one molecule. `select_refinement_policy_` sets the ETK and MMFF
 *  thresholds separately and the distance-geometry stages carry their own, so the four are calibrated apart.
 *  `final_timestep` is the step length the driver last used, in its own update's units, reported because a
 *  stalled stage and a descending one are indistinguishable from the iteration count alone, and that decides
 *  whether more budget would buy anything.
 */
struct fire_result_t {
    float energy;
    float rms_gradient;
    float final_timestep;
    std::int32_t iterations;
    fire_status_t status;
};

/**
 *  Largest root-mean-square displacement one step may apply, in angstrom.
 *
 *  It bounds the step's length and leaves its direction alone. Bounding each component instead bounds a box
 *  rather than a ball, so a step whose components differ in magnitude comes out rotated — the clipped ones
 *  shrank, the rest kept their length — and the minimizer walks somewhere the energy never pointed.
 */
constexpr float max_root_mean_square_step_k = 0.5f;

/**
 *  @brief FIRE minimization driver.
 *
 *  @tparam dimensions_ Rows per atom the stage works in.
 *  @tparam energy_gradient_functor_ Callable: float(positions, gradient, num_elements, lane)
 *
 *  @param[inout] scratch The stage's positions, velocities and gradient, each named by its role.
 *  @param[in] max_iterations Maximum energy evaluations; one fewer step is taken, so the state
 *      reported is one the driver measured rather than one it moved to and never looked at.
 *  @param[in] gradient_threshold Convergence when rms(gradient) < threshold.
 *  @param[in] lane Warp lane index (threadIdx.x % 32).
 *  @param[in] compute_energy_gradient The energy/gradient functor.
 *  @return Warp-uniform `fire_result_t` with energy, RMS gradient, iteration count, and status.
 *  @see distance_geometry_energy_functor_t, etk_torsion_energy_functor_t, mmff94_energy_functor_t
 */
template <std::size_t dimensions_, typename energy_gradient_functor_>
__device__ __forceinline__ fire_result_t fire_minimize_(fire_scratch<dimensions_> scratch, std::int32_t max_iterations,
                                                        float gradient_threshold, std::uint32_t lane,
                                                        energy_gradient_functor_ &&compute_energy_gradient) {
    // The evaluators take the flat arrays a kernel parameter can carry `__restrict__` on, so the roles stop
    // here; they exist to keep the three apart at the call sites where a transposition would compile.
    float *const positions = scratch.positions.data_handle();
    float *const velocities = scratch.velocities.data_handle();
    float *const gradient = scratch.gradient.data_handle();
    std::uint32_t const num_elements = scratch.positions.extent(0) * static_cast<std::uint32_t>(dimensions_);

    // FIRE parameters (Bitzek et al., PRL 2006)
    constexpr float initial_timestep_k = 0.05f;
    constexpr float max_timestep_k = 0.5f;
    // Below this a step is shorter than single precision resolves against a coordinate of a few angstrom, so
    // the geometry stops moving while the budget drains. Nothing in the captured corpus reaches it — the
    // measured floor of the distribution is 1e-3 — so it is a guard rail on an unbounded halving, not a lever.
    constexpr float min_timestep_k = 1e-5f;
    constexpr float timestep_increase_factor_k = 1.1f;
    constexpr float timestep_decrease_factor_k = 0.5f;
    constexpr float initial_alpha_k = 0.1f;
    constexpr float alpha_decay_factor_k = 0.99f;
    constexpr int min_steps_before_acceleration_k = 5;

    float timestep = initial_timestep_k;
    float alpha = initial_alpha_k;
    int steps_since_reset = 0;

    // The state the driver last measured, and the state it returns.
    float energy = 0.0f;
    float rms_gradient = 0.0f;

    // zero velocities
    for (std::uint32_t element = lane; element < num_elements; element += warp_size_k) velocities[element] = 0.0f;
    __syncwarp(full_mask_k);

    for (std::int32_t iteration = 0; iteration < max_iterations; ++iteration) {
        // compute energy and gradient (gradient = -force, so force = -gradient)
        energy = compute_energy_gradient(positions, gradient, num_elements, lane);
        __syncwarp(full_mask_k);

        // fused computation: rms_gradient, power, velocity_norm, force_norm in a single pass
        float gradient_dot_gradient = 0.0f;
        float gradient_dot_velocity = 0.0f;
        float velocity_dot_velocity = 0.0f;
        for (std::uint32_t element = lane; element < num_elements; element += warp_size_k) {
            float const gradient_value = gradient[element];
            float const velocity_value = velocities[element];
            gradient_dot_gradient = fmaf(gradient_value, gradient_value, gradient_dot_gradient);
            gradient_dot_velocity = fmaf(gradient_value, velocity_value, gradient_dot_velocity);
            velocity_dot_velocity = fmaf(velocity_value, velocity_value, velocity_dot_velocity);
        }
        // interleaved warp reduction for instruction-level parallelism
        for (int offset = 16; offset > 0; offset >>= 1) {
            gradient_dot_gradient += __shfl_xor_sync(full_mask_k, gradient_dot_gradient, offset);
            gradient_dot_velocity += __shfl_xor_sync(full_mask_k, gradient_dot_velocity, offset);
            velocity_dot_velocity += __shfl_xor_sync(full_mask_k, velocity_dot_velocity, offset);
        }

        rms_gradient = sqrtf(gradient_dot_gradient / static_cast<float>(num_elements));
        if (rms_gradient < gradient_threshold)
            return {.energy = energy,
                    .rms_gradient = rms_gradient,
                    .final_timestep = timestep,
                    .iterations = iteration,
                    .status = fire_status_t::converged_k};

        // The last budgeted evaluation has happened, so stepping now would move past the only state this call
        // measured, and the caller would read an energy and a gradient norm from two different geometries.
        if (iteration + 1 == max_iterations) break;

        // P = F·v = -∇E·v. Equality is admitted because entry zeroes the velocities, so every stage's first
        // iteration has `power == -0.0f`, and a strict test would take the decelerating branch before a single
        // step — halving the initial timestep to 0.025 on every stage, restart and molecule, unconditionally.
        float const power = -gradient_dot_velocity;
        if (power >= 0.0f) {
            float const inverse_force_norm = rsqrtf(fmaxf(gradient_dot_gradient, 1e-20f));
            float const velocity_norm = sqrtf(fmaxf(velocity_dot_velocity, 1e-20f));
            float const velocity_over_force_ratio = velocity_norm * inverse_force_norm;

            // v = (1-α)·v + α·|v|·(F / |F|)  where F = -gradient
            for (std::uint32_t element = lane; element < num_elements; element += warp_size_k) {
                velocities[element] = (1.0f - alpha) * velocities[element] -
                                      alpha * velocity_over_force_ratio * gradient[element];
            }

            steps_since_reset++;
            if (steps_since_reset > min_steps_before_acceleration_k) {
                timestep = fminf(timestep * timestep_increase_factor_k, max_timestep_k);
                alpha *= alpha_decay_factor_k;
            }
        }
        else {
            // Decelerating: momentum carried the geometry past where the force still agreed with it, so undo
            // half the last step before dropping the velocity. Zeroing alone leaves the overshoot in the
            // positions for the next downhill stretch to walk back — FIRE 2.0's correction, Guénolé et al. 2020.
            for (std::uint32_t element = lane; element < num_elements; element += warp_size_k) {
                positions[element] -= 0.5f * timestep * velocities[element];
                velocities[element] = 0.0f;
            }
            timestep = fmaxf(timestep * timestep_decrease_factor_k, min_timestep_k);
            alpha = initial_alpha_k;
            steps_since_reset = 0;
        }
        __syncwarp(full_mask_k);

        // Velocity Verlet: x += v·dt − ½·g·dt², then v -= g·dt, with the whole step held to a trust region on
        // its length. That length is unknown until the step is built, so the sweep runs twice, once to measure
        // and once to apply; the second rebuilds the step because a fourth `num_elements` array costs more.
        float const half_timestep_squared = 0.5f * timestep * timestep;
        float step_dot_step = 0.0f;
        for (std::uint32_t element = lane; element < num_elements; element += warp_size_k) {
            float const step = fmaf(velocities[element], timestep, -gradient[element] * half_timestep_squared);
            step_dot_step = fmaf(step, step, step_dot_step);
        }
        step_dot_step = warp_reduce_sum_f32_(step_dot_step);

        float const root_mean_square_step = sqrtf(step_dot_step / static_cast<float>(num_elements));
        float const step_scale = root_mean_square_step > max_root_mean_square_step_k
                                     ? max_root_mean_square_step_k / root_mean_square_step
                                     : 1.0f;

        // The scale rides the velocity increment too: the momentum must agree with how far the step went.
        for (std::uint32_t element = lane; element < num_elements; element += warp_size_k) {
            float const step = fmaf(velocities[element], timestep, -gradient[element] * half_timestep_squared);
            positions[element] += step * step_scale;
            velocities[element] = fmaf(gradient[element], -timestep * step_scale, velocities[element]);
        }
        __syncwarp(full_mask_k);
    }

    return {.energy = energy,
            .rms_gradient = rms_gradient,
            .final_timestep = timestep,
            .iterations = max_iterations,
            .status = fire_status_t::exhausted_k};
}

/**
 *  @brief Barzilai-Borwein gradient descent, warp-level, same three arrays and same functor.
 *
 *  FIRE carries one timestep for every degree of freedom, so a molecular force field holds it at the stiffest
 *  bond stretch's stability limit while the soft torsions relax at that rate — measured as a residual RMS
 *  gradient three decades above any threshold after the whole budget. Barzilai-Borwein reads curvature out of
 *  the last step instead: with @f$ s = x_k - x_{k-1} @f$ and @f$ y = g_k - g_{k-1} @f$ the step is
 *  @f$ \langle s,y \rangle / \langle y,y \rangle @f$, a secant estimate of inverse curvature along the
 *  direction actually travelled.
 *
 *  It costs nothing extra: the previous gradient lives in FIRE's velocity array, so @f$ s @f$ is
 *  @f$ -\text{step} \cdot g_{k-1} @f$ and both inner products fall out of the three reductions FIRE already
 *  performs, and the position update is one sweep since @f$ -\alpha g @f$ has RMS length
 *  @f$ \alpha \cdot \mathrm{rms}(g) @f$ analytically.
 *
 *  The method is not monotone — a step may raise the energy on the way to a far better point, which is where
 *  its speed comes from — so `mmff_polish_warp_kernel` keeps the staged geometry and declines a worse write.
 *
 *  @param[inout] scratch Positions, the previous gradient in the velocity slot, and the gradient.
 *  @param[in] max_iterations Maximum energy evaluations; the last one is measured, not stepped from.
 *  @param[in] gradient_threshold Convergence when rms(gradient) < threshold.
 *  @param[in] lane Warp lane index (threadIdx.x % 32).
 *  @param[in] compute_energy_gradient The energy/gradient functor.
 *  @return Warp-uniform `fire_result_t`, its `final_timestep` the last step length in Å per unit force.
 */
template <std::size_t dimensions_, typename energy_gradient_functor_>
__device__ __forceinline__ fire_result_t
barzilai_borwein_minimize_(fire_scratch<dimensions_> scratch, std::int32_t max_iterations, float gradient_threshold,
                           std::uint32_t lane, energy_gradient_functor_ &&compute_energy_gradient) {

    float *const positions = scratch.positions.data_handle();
    float *const previous_gradient = scratch.velocities.data_handle();
    float *const gradient = scratch.gradient.data_handle();
    std::uint32_t const num_elements = scratch.positions.extent(0) * static_cast<std::uint32_t>(dimensions_);

    // RMS displacement the opening step is allowed, before any curvature has been observed.
    constexpr float opening_root_mean_square_step_k = 0.01f;
    // Widest and narrowest inverse curvature the secant estimate may report, in Å² per kcal/mol.
    constexpr float min_step_k = 1e-6f;
    constexpr float max_step_k = 1.0f;

    // Zero rather than uninitialized: the opening iteration reduces against this array before anything
    // has been written to it, and `applied_step` is what tells it there is no history to read.
    for (std::uint32_t element = lane; element < num_elements; element += warp_size_k)
        previous_gradient[element] = 0.0f;
    __syncwarp(full_mask_k);

    float applied_step = 0.0f;
    float energy = 0.0f;
    float rms_gradient = 0.0f;

    for (std::int32_t iteration = 0; iteration < max_iterations; ++iteration) {
        energy = compute_energy_gradient(positions, gradient, num_elements, lane);
        __syncwarp(full_mask_k);

        float gradient_dot_gradient = 0.0f;
        // The secant needs `<p, g - p>` differenced per element, not formed as `<p,p> - <g,p>` from two
        // separately reduced sums: near a stationary point those sums approach each other while each keeps the
        // rounding of its own magnitude, and under about 1e-7 of them the difference is noise — the step it
        // denominates then comes back arbitrarily large or with its convexity test flipped.
        float previous_dot_difference = 0.0f;
        float previous_dot_previous = 0.0f;
        for (std::uint32_t element = lane; element < num_elements; element += warp_size_k) {
            float const gradient_value = gradient[element];
            float const previous_value = previous_gradient[element];
            float const difference_value = gradient_value - previous_value;
            gradient_dot_gradient = fmaf(gradient_value, gradient_value, gradient_dot_gradient);
            previous_dot_difference = fmaf(previous_value, difference_value, previous_dot_difference);
            previous_dot_previous = fmaf(previous_value, previous_value, previous_dot_previous);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            gradient_dot_gradient += __shfl_xor_sync(full_mask_k, gradient_dot_gradient, offset);
            previous_dot_difference += __shfl_xor_sync(full_mask_k, previous_dot_difference, offset);
            previous_dot_previous += __shfl_xor_sync(full_mask_k, previous_dot_previous, offset);
        }

        rms_gradient = sqrtf(gradient_dot_gradient / static_cast<float>(num_elements));
        if (rms_gradient < gradient_threshold)
            return {.energy = energy,
                    .rms_gradient = rms_gradient,
                    .final_timestep = applied_step,
                    .iterations = iteration,
                    .status = fire_status_t::converged_k};

        // The last budgeted evaluation has happened, so a step now would move the structure past the
        // only state this call has measured.
        if (iteration + 1 == max_iterations) break;

        // `s = -applied_step * p`, so `<s,y> = -applied_step * <p, g-p>` and `<s,s>` is a sum of squares with
        // no cancellation — hence only the first needed the elementwise pass, and `<y,y>` is reconstructed.
        float const step_dot_difference = -applied_step * previous_dot_difference;
        float const step_dot_step = applied_step * applied_step * previous_dot_previous;
        bool const secant_is_convex = step_dot_difference > 0.0f && step_dot_step > 0.0f;

        // `<s,y>/<y,y>`, the short Barzilai-Borwein step, near `1/lambda_max`. Its sibling `<s,s>/<s,y>` sits
        // near `1/lambda_min` and looks better on paper, but unguarded it needs a line search this kernel has
        // no registers for and diverges outright on part of the corpus; the long step also flatters the
        // convergence counter on small well-conditioned molecules while degrading the geometry, so judge a
        // change here by the energy gap and never by the converged share. Measured over 36 drug-like
        // molecules at the 800-evaluation cap: median RMS gradient 0.49 against 0.049, worst case 8.8e4
        // against 0.58, a median energy gap to RDKit's basin of 3.0 kcal/mol against 1.5, and 97 % landing
        // within 10 kcal/mol of it against 69 %. Without history,
        // or along a
        // direction the secant reports as non-convex, the fallback is the trust region alone — a fixed RMS
        // displacement, the one step length that needs no curvature to be safe.
        float const difference_dot_difference = gradient_dot_gradient - previous_dot_previous -
                                                2.0f * previous_dot_difference;
        float step = secant_is_convex ? step_dot_difference / difference_dot_difference
                                      : opening_root_mean_square_step_k / rms_gradient;
        step = fminf(fmaxf(step, min_step_k), max_step_k);
        step = fminf(step, max_root_mean_square_step_k / rms_gradient);

        for (std::uint32_t element = lane; element < num_elements; element += warp_size_k) {
            float const gradient_value = gradient[element];
            previous_gradient[element] = gradient_value;
            positions[element] -= step * gradient_value;
        }
        applied_step = step;
        __syncwarp(full_mask_k);
    }

    return {.energy = energy,
            .rms_gradient = rms_gradient,
            .final_timestep = applied_step,
            .iterations = max_iterations,
            .status = fire_status_t::exhausted_k};
}

} // namespace smallfield::cuda

#endif // __CUDACC__
