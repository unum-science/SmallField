/**
 * @file fire_warp.cuh
 * @brief FIRE (Fast Inertial Relaxation Engine) warp-level optimizer.
 *
 * Bitzek et al., PRL 97, 170201 (2006).
 * Gradient descent with adaptive momentum. Much lighter than L-BFGS:
 * needs only positions + velocities + gradient (3 arrays, no history).
 *
 * Functor interface:
 *   float functor(positions, gradient, num_elements, lane)
 *   float functor.energy_only(positions, num_elements, lane)
 */

#pragma once

#if defined(__CUDACC__)

#include "warp_primitives.cuh"

namespace smallfield {

/**
 * FIRE minimization driver.
 *
 * @tparam EnergyGradientFunctor  Callable: float(positions, gradient, num_elements, lane)
 *
 * @param positions       Coordinate array (modified in place). Length = num_elements.
 * @param velocities      Shared memory for FIRE velocities. Length = num_elements.
 * @param gradient        Shared memory for gradient. Length = num_elements.
 * @param num_elements    Total coordinate dimensions (e.g., 3*N or 4*N).
 * @param max_iterations  Maximum FIRE iterations.
 * @param gradient_threshold  Convergence when rms(gradient) < threshold.
 * @param lane            Warp lane index (threadIdx.x % 32).
 * @param compute_energy_gradient  The energy/gradient functor.
 * @param[out] out_final_energy    Final energy (lane 0).
 * @param[out] out_iterations      Iteration count (lane 0).
 * @param[out] out_status          0=converged, 1=max_iterations (lane 0).
 * @param[out] out_final_rms_gradient  Final RMS gradient (lane 0, optional).
 */
template <typename EnergyGradientFunctor>
__device__ __forceinline__ void fire_minimize_(float* positions, float* velocities, float* gradient,
                                               std::uint32_t num_elements, std::int32_t max_iterations,
                                               float gradient_threshold, std::uint32_t lane,
                                               EnergyGradientFunctor&& compute_energy_gradient, float* out_final_energy,
                                               std::int32_t* out_iterations, std::int32_t* out_status,
                                               float* out_final_rms_gradient = nullptr) {
    // FIRE parameters (Bitzek et al., PRL 2006)
    constexpr float dt_start = 0.05f;
    constexpr float dt_max = 0.5f;
    constexpr float f_inc = 1.1f;
    constexpr float f_dec = 0.5f;
    constexpr float alpha_start = 0.1f;
    constexpr float f_alpha = 0.99f;
    constexpr int n_min = 5; // minimum steps before accelerating

    float dt = dt_start;
    float alpha = alpha_start;
    int steps_since_reset = 0;
    float last_rms_gradient = 0.0f;

    // zero velocities
    for (std::uint32_t i = lane; i < num_elements; i += warp_size)
        velocities[i] = 0.0f;
    __syncwarp(full_mask);

    for (std::int32_t iteration = 0; iteration < max_iterations; ++iteration) {
        // compute energy and gradient (gradient = -force, so force = -gradient)
        float const energy = compute_energy_gradient(positions, gradient, num_elements, lane);
        __syncwarp(full_mask);

        // fused computation: rms_gradient, power, v_norm, f_norm in a single pass
        float sum_gg = 0.0f, sum_gv = 0.0f, sum_vv = 0.0f;
        for (std::uint32_t i = lane; i < num_elements; i += warp_size) {
            float const g = gradient[i], v = velocities[i];
            sum_gg = fmaf(g, g, sum_gg);
            sum_gv = fmaf(g, v, sum_gv);
            sum_vv = fmaf(v, v, sum_vv);
        }
        // interleaved warp reduction for instruction-level parallelism
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_gg += __shfl_xor_sync(full_mask, sum_gg, offset);
            sum_gv += __shfl_xor_sync(full_mask, sum_gv, offset);
            sum_vv += __shfl_xor_sync(full_mask, sum_vv, offset);
        }

        float const rms_gradient = sqrtf(sum_gg * __frcp_rn(static_cast<float>(num_elements)));
        last_rms_gradient = rms_gradient;
        if (rms_gradient < gradient_threshold) {
            if (lane == 0) {
                *out_status = 0;
                *out_iterations = iteration;
                *out_final_energy = energy;
                if (out_final_rms_gradient != nullptr)
                    *out_final_rms_gradient = rms_gradient;
            }
            return;
        }

        float const power = -sum_gv; // P = F·v = -∇E·v
        if (power > 0.0f) {
            float const inverse_f_norm = rsqrtf(fmaxf(sum_gg, 1e-20f));
            float const v_norm = sqrtf(fmaxf(sum_vv, 1e-20f));
            float const v_over_f = v_norm * inverse_f_norm;

            // v = (1-α)·v + α·|v|·F̂  where F = -gradient
            for (std::uint32_t i = lane; i < num_elements; i += warp_size) {
                velocities[i] = (1.0f - alpha) * velocities[i] - alpha * v_over_f * gradient[i];
            }

            if (steps_since_reset > n_min) {
                dt = fminf(dt * f_inc, dt_max);
                alpha *= f_alpha;
            }
            steps_since_reset++;
        } else {
            // decelerating: reset velocity, shrink step
            for (std::uint32_t i = lane; i < num_elements; i += warp_size)
                velocities[i] = 0.0f;
            dt *= f_dec;
            alpha = alpha_start;
            steps_since_reset = 0;
        }
        __syncwarp(full_mask);

        // velocity Verlet: x += v·dt − ½·g·dt², then v -= g·dt
        float const half_dt_squared = 0.5f * dt * dt;
        for (std::uint32_t i = lane; i < num_elements; i += warp_size) {
            float displacement = fmaf(velocities[i], dt, -gradient[i] * half_dt_squared);
            displacement = fmaxf(-0.5f, fminf(0.5f, displacement));
            positions[i] += displacement;
            velocities[i] = fmaf(gradient[i], -dt, velocities[i]);
        }
        __syncwarp(full_mask);
    }

    // max iterations reached — report the last computed energy
    float const final_energy = compute_energy_gradient.energy_only(positions, num_elements, lane);
    if (lane == 0) {
        *out_status = 1;
        *out_iterations = max_iterations;
        *out_final_energy = final_energy;
        if (out_final_rms_gradient != nullptr)
            *out_final_rms_gradient = last_rms_gradient;
    }
}

} // namespace smallfield

#endif // __CUDACC__
