/**
 *  @file src/smallfield/cuda/fire_warp.cuh
 *  @brief FIRE, Fast Inertial Relaxation Engine, warp-level optimizer.
 *
 *  Bitzek et al., PRL 97, 170201 (2006). Gradient descent with adaptive momentum. Much lighter than
 *  L-BFGS: needs only positions, velocities and gradient, three arrays, no history.
 *
 *  Functor interface:
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
 *  Outside the CUDA guard below because a host caller staging or reporting a stage's outcome needs
 *  to name the two states, and the enumerator is the name.
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
 *  Positions, velocities and a gradient have identical shape, layout and element type, so as three
 *  bare pointers under one count nothing but their order tells them apart — and the order is
 *  restated at every call. Swapping the last two compiles and minimizes something else: the driver
 *  zeroes what it thinks are velocities and the functor overwrites what it thinks is the gradient.
 *  `coordinate_role_t` is an accessor rather than a wrapper, so naming them costs no bytes.
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
 *  All fields are warp-uniform: every lane returns the same struct value because energy and
 *  `rms_gradient` are warp-reduced inside the loop and the other two are derived from the uniform
 *  loop counter.
 *
 *  The energy and the gradient norm describe the @b same geometry, which is the geometry in
 *  `positions` on return. That is not free: it is why the loop stops one step short rather than
 *  taking a move it has not measured.
 */
struct fire_result_t {
    float energy;
    /**
     *  Root-mean-square gradient over @b elements, so the same threshold is a different absolute
     *  norm per stage: distance geometry and the fourth-dimension crush run at four elements per
     *  atom and the ETK and MMFF stages at three, which is a factor of sqrt(4/3) between them for
     *  one molecule. `select_refinement_policy_` sets the ETK and MMFF thresholds separately and the
     *  distance-geometry stages carry their own, so the four are calibrated apart rather than shared.
     */
    float rms_gradient;
    std::int32_t iterations;
    fire_status_t status;
};

/**
 *  Largest root-mean-square displacement one step may apply, in angstrom.
 *
 *  It bounds the step's length and leaves its direction alone. Bounding each component instead
 *  bounds a box rather than a ball, so a step whose components differ in magnitude comes out
 *  rotated: the components that clipped shrank and the ones that did not kept their length, and the
 *  minimizer then walks somewhere the energy never pointed.
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
    // The evaluators take the flat arrays a kernel formal parameter can carry `__restrict__` on, so
    // the roles stop here: they exist to keep the three apart at every call site, which is where a
    // transposition would otherwise compile.
    float *const positions = scratch.positions.data_handle();
    float *const velocities = scratch.velocities.data_handle();
    float *const gradient = scratch.gradient.data_handle();
    std::uint32_t const num_elements = scratch.positions.extent(0) * static_cast<std::uint32_t>(dimensions_);

    // FIRE parameters (Bitzek et al., PRL 2006)
    constexpr float initial_timestep_k = 0.05f;
    constexpr float max_timestep_k = 0.5f;
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
                    .iterations = iteration,
                    .status = fire_status_t::converged_k};

        // The last budgeted evaluation has happened, so a step now would move the structure past the
        // only state this call has measured, and the caller would read an energy and a gradient norm
        // belonging to two different geometries.
        if (iteration + 1 == max_iterations) break;

        float const power = -gradient_dot_velocity; // P = F·v = -∇E·v
        if (power > 0.0f) {
            float const inverse_force_norm = rsqrtf(fmaxf(gradient_dot_gradient, 1e-20f));
            float const velocity_norm = sqrtf(fmaxf(velocity_dot_velocity, 1e-20f));
            float const velocity_over_force_ratio = velocity_norm * inverse_force_norm;

            // v = (1-α)·v + α·|v|·(F / |F|)  where F = -gradient
            for (std::uint32_t element = lane; element < num_elements; element += warp_size_k) {
                velocities[element] = (1.0f - alpha) * velocities[element] -
                                      alpha * velocity_over_force_ratio * gradient[element];
            }

            if (steps_since_reset > min_steps_before_acceleration_k) {
                timestep = fminf(timestep * timestep_increase_factor_k, max_timestep_k);
                alpha *= alpha_decay_factor_k;
            }
            steps_since_reset++;
        }
        else {
            // decelerating: reset velocity, shrink step
            for (std::uint32_t element = lane; element < num_elements; element += warp_size_k)
                velocities[element] = 0.0f;
            timestep *= timestep_decrease_factor_k;
            alpha = initial_alpha_k;
            steps_since_reset = 0;
        }
        __syncwarp(full_mask_k);

        // velocity Verlet: x += v·dt − ½·g·dt², then v -= g·dt, with the whole step held to a
        // trust region on its length. Its length is not known until the step is built, so the sweep
        // runs twice: once to measure, once to apply. The second pass rebuilds the step rather than
        // parking it, because a fourth shared array of `num_elements` floats would cost more than
        // the arithmetic does.
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

        // The scale rides the velocity increment too: a bounded step is a shorter step, and the
        // momentum it feeds into the next iteration has to agree with how far it actually went.
        for (std::uint32_t element = lane; element < num_elements; element += warp_size_k) {
            float const step = fmaf(velocities[element], timestep, -gradient[element] * half_timestep_squared);
            positions[element] += step * step_scale;
            velocities[element] = fmaf(gradient[element], -timestep * step_scale, velocities[element]);
        }
        __syncwarp(full_mask_k);
    }

    return {.energy = energy,
            .rms_gradient = rms_gradient,
            .iterations = max_iterations,
            .status = fire_status_t::exhausted_k};
}

} // namespace smallfield::cuda

#endif // __CUDACC__
