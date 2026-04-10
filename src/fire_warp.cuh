/**
 * @file fire_warp.cuh
 * @brief FIRE (Fast Inertial Relaxation Engine) warp-level optimizer.
 *
 * Bitzek et al., PRL 97, 170201 (2006).
 * Gradient descent with adaptive momentum. Much lighter than L-BFGS:
 * needs only positions + velocities + gradient (3 arrays, no history).
 *
 * Same functor interface as lbfgs_minimize_:
 *   float functor(positions, gradient, num_elements, lane)
 *   float functor.energy_only(positions, num_elements, lane)
 */

#pragma once

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

namespace smallfield {

constexpr std::uint32_t fire_warp_size = 32;
constexpr unsigned int fire_full_mask = 0xFFFFFFFF;

/** Warp reduction via butterfly XOR shuffles. All lanes get the result. */
__device__ __forceinline__ float fire_warp_reduce_sum_(float value) {
    value += __shfl_xor_sync(fire_full_mask, value, 16);
    value += __shfl_xor_sync(fire_full_mask, value, 8);
    value += __shfl_xor_sync(fire_full_mask, value, 4);
    value += __shfl_xor_sync(fire_full_mask, value, 2);
    value += __shfl_xor_sync(fire_full_mask, value, 1);
    return value;
}

/** Warp dot product. */
__device__ __forceinline__ float fire_warp_dot_(float const* a, float const* b,
                                                 std::uint32_t num_elements, std::uint32_t lane) {
    float dot = 0.0f;
    for (std::uint32_t i = lane; i < num_elements; i += fire_warp_size)
        dot = fmaf(a[i], b[i], dot);
    return fire_warp_reduce_sum_(dot);
}

/** Warp RMS: √(Σxᵢ²/n) */
__device__ __forceinline__ float fire_warp_rms_(float const* x, std::uint32_t num_elements,
                                                 std::uint32_t lane) {
    float sum_sq = 0.0f;
    for (std::uint32_t i = lane; i < num_elements; i += fire_warp_size) {
        float const val = x[i];
        sum_sq = fmaf(val, val, sum_sq);
    }
    return sqrtf(fire_warp_reduce_sum_(sum_sq) * __frcp_rn(static_cast<float>(num_elements)));
}

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
 */
template <typename EnergyGradientFunctor>
__device__ __forceinline__ void fire_minimize_(float* positions, float* velocities, float* gradient,
                                               std::uint32_t num_elements, std::int32_t max_iterations,
                                               float gradient_threshold, std::uint32_t lane,
                                               EnergyGradientFunctor&& compute_energy_gradient,
                                               float* out_final_energy, std::int32_t* out_iterations,
                                               std::int32_t* out_status) {
    // FIRE parameters (Bitzek et al., PRL 2006)
    constexpr float dt_start = 0.1f;
    constexpr float dt_max = 1.0f;
    constexpr float f_inc = 1.1f;
    constexpr float f_dec = 0.5f;
    constexpr float alpha_start = 0.1f;
    constexpr float f_alpha = 0.99f;
    constexpr int n_min = 5; // minimum steps before accelerating

    float dt = dt_start;
    float alpha = alpha_start;
    int steps_since_reset = 0;

    // zero velocities
    for (std::uint32_t i = lane; i < num_elements; i += fire_warp_size)
        velocities[i] = 0.0f;
    __syncwarp(fire_full_mask);

    for (std::int32_t iteration = 0; iteration < max_iterations; ++iteration) {
        // compute energy and gradient (gradient = -force, so force = -gradient)
        float const energy = compute_energy_gradient(positions, gradient, num_elements, lane);
        __syncwarp(fire_full_mask);

        // convergence check
        float const rms_gradient = fire_warp_rms_(gradient, num_elements, lane);
        if (rms_gradient < gradient_threshold) {
            if (lane == 0) {
                *out_status = 0;
                *out_iterations = iteration;
                *out_final_energy = energy;
            }
            return;
        }

        // P = F · v = -gradient · velocity
        float const power = -fire_warp_dot_(gradient, velocities, num_elements, lane);

        // broadcast power to all lanes (already done by warp_reduce)
        if (power > 0.0f) {
            // accelerating: mix velocity toward force direction
            // |v| and |F| for normalization
            float const v_norm_sq = fire_warp_dot_(velocities, velocities, num_elements, lane);
            float const f_norm_sq = fire_warp_dot_(gradient, gradient, num_elements, lane);
            float const v_norm = sqrtf(fmaxf(v_norm_sq, 1e-20f));
            float const f_norm = sqrtf(fmaxf(f_norm_sq, 1e-20f));
            float const v_over_f = v_norm * __frcp_rn(f_norm);

            // v = (1-α)·v + α·|v|·F̂  where F = -gradient
            for (std::uint32_t i = lane; i < num_elements; i += fire_warp_size) {
                velocities[i] = (1.0f - alpha) * velocities[i] - alpha * v_over_f * gradient[i];
            }

            if (steps_since_reset > n_min) {
                dt = fminf(dt * f_inc, dt_max);
                alpha *= f_alpha;
            }
            steps_since_reset++;
        } else {
            // decelerating: reset velocity, shrink step
            for (std::uint32_t i = lane; i < num_elements; i += fire_warp_size)
                velocities[i] = 0.0f;
            dt *= f_dec;
            alpha = alpha_start;
            steps_since_reset = 0;
        }
        __syncwarp(fire_full_mask);

        // velocity Verlet integration: v += F·dt, x += v·dt
        // (F = -gradient)
        for (std::uint32_t i = lane; i < num_elements; i += fire_warp_size) {
            velocities[i] -= gradient[i] * dt;
            positions[i] += velocities[i] * dt;
        }
        __syncwarp(fire_full_mask);
    }

    // max iterations reached
    if (lane == 0) {
        *out_status = 1;
        *out_iterations = max_iterations;
        *out_final_energy = 0.0f;
    }
}

} // namespace smallfield

#endif // __CUDACC__
