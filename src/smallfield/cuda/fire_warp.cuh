/**
 *  @file fire_warp.cuh
 *  @brief FIRE (Fast Inertial Relaxation Engine) warp-level optimizer.
 *
 *  Bitzek et al., PRL 97, 170201 (2006).
 *  Gradient descent with adaptive momentum. Much lighter than L-BFGS:
 *  needs only positions + velocities + gradient (3 arrays, no history).
 *
 *  Functor interface:
 *   float functor(positions, gradient, num_elements, lane)
 *   float functor.energy_only(positions, num_elements, lane)
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include "smallfield/cuda/warp_primitives.cuh"

namespace smallfield::cuda {

/**
 *  @brief Outcome of a FIRE minimization stage.
 *
 *  All fields are warp-uniform: every lane returns the same struct value
 *  because energy / rms_gradient are warp-reduced inside the loop and
 *  iterations / status are derived from the (uniform) loop counter.
 */
struct fire_result_t {
    float energy;
    float rms_gradient;
    std::int32_t iterations;
    std::int32_t status; ///< 0 = converged below @p gradient_threshold, 1 = hit max_iterations.
};

/**
 *  @brief FIRE minimization driver.
 *
 *  @tparam energy_gradient_functor_ Callable: float(positions, gradient, num_elements, lane)
 *
 *  @param[inout] positions Coordinate array (modified in place). Length = @p num_elements.
 *  @param[inout] velocities Shared memory for FIRE velocities. Length = @p num_elements.
 *  @param[out] gradient Shared memory for gradient. Length = @p num_elements.
 *  @param[in] num_elements Total coordinate dimensions (e.g., 3*N or 4*N).
 *  @param[in] max_iterations Maximum FIRE iterations.
 *  @param[in] gradient_threshold Convergence when rms(gradient) < threshold.
 *  @param[in] lane Warp lane index (threadIdx.x % 32).
 *  @param[in] compute_energy_gradient The energy/gradient functor.
 *  @return Warp-uniform @ref fire_result_t with energy, RMS gradient, iteration count, and status.
 *  @see distance_geometry_energy_functor_t, etk_torsion_energy_functor_t, mmff94_energy_functor_t
 */
template <typename energy_gradient_functor_>
__device__ __forceinline__ fire_result_t fire_minimize_(float *positions, float *velocities, float *gradient,
                                                        std::uint32_t num_elements, std::int32_t max_iterations,
                                                        float gradient_threshold, std::uint32_t lane,
                                                        energy_gradient_functor_ &&compute_energy_gradient) {
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
    float last_rms_gradient = 0.0f;

    // zero velocities
    for (std::uint32_t i = lane; i < num_elements; i += warp_size_k) velocities[i] = 0.0f;
    __syncwarp(full_mask_k);

    for (std::int32_t iteration = 0; iteration < max_iterations; ++iteration) {
        // compute energy and gradient (gradient = -force, so force = -gradient)
        float const energy = compute_energy_gradient(positions, gradient, num_elements, lane);
        __syncwarp(full_mask_k);

        // fused computation: rms_gradient, power, velocity_norm, force_norm in a single pass
        float gradient_dot_gradient = 0.0f;
        float gradient_dot_velocity = 0.0f;
        float velocity_dot_velocity = 0.0f;
        for (std::uint32_t i = lane; i < num_elements; i += warp_size_k) {
            float const gradient_value = gradient[i];
            float const velocity_value = velocities[i];
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

        float const rms_gradient = sqrtf(gradient_dot_gradient / static_cast<float>(num_elements));
        last_rms_gradient = rms_gradient;
        if (rms_gradient < gradient_threshold) {
            return {energy, rms_gradient, iteration, 0};
        }

        float const power = -gradient_dot_velocity; // P = F·v = -∇E·v
        if (power > 0.0f) {
            float const inverse_force_norm = rsqrtf(fmaxf(gradient_dot_gradient, 1e-20f));
            float const velocity_norm = sqrtf(fmaxf(velocity_dot_velocity, 1e-20f));
            float const velocity_over_force_ratio = velocity_norm * inverse_force_norm;

            // v = (1-α)·v + α·|v|·F̂  where F = -gradient
            for (std::uint32_t i = lane; i < num_elements; i += warp_size_k) {
                velocities[i] = (1.0f - alpha) * velocities[i] - alpha * velocity_over_force_ratio * gradient[i];
            }

            if (steps_since_reset > min_steps_before_acceleration_k) {
                timestep = fminf(timestep * timestep_increase_factor_k, max_timestep_k);
                alpha *= alpha_decay_factor_k;
            }
            steps_since_reset++;
        }
        else {
            // decelerating: reset velocity, shrink step
            for (std::uint32_t i = lane; i < num_elements; i += warp_size_k) velocities[i] = 0.0f;
            timestep *= timestep_decrease_factor_k;
            alpha = initial_alpha_k;
            steps_since_reset = 0;
        }
        __syncwarp(full_mask_k);

        // velocity Verlet: x += v·dt − ½·g·dt², then v -= g·dt
        float const half_timestep_squared = 0.5f * timestep * timestep;
        for (std::uint32_t i = lane; i < num_elements; i += warp_size_k) {
            float displacement = fmaf(velocities[i], timestep, -gradient[i] * half_timestep_squared);
            displacement = fmaxf(-0.5f, fminf(0.5f, displacement));
            positions[i] += displacement;
            velocities[i] = fmaf(gradient[i], -timestep, velocities[i]);
        }
        __syncwarp(full_mask_k);
    }

    // max iterations reached — report the last computed energy
    float const final_energy = compute_energy_gradient.energy_only(positions, num_elements, lane);
    return {final_energy, last_rms_gradient, max_iterations, 1};
}

} // namespace smallfield::cuda

#endif // __CUDACC__
