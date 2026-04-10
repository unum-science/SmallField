/**
 * @file lbfgs_warp.cuh
 * @brief Warp-level L-BFGS optimizer, energy-functor-generic.
 *
 * Extracted from mmff94.cuh to be reusable across force fields (MMFF94, DG, ETK).
 * All operations are warp-level (32 threads), using only __syncwarp and __shfl_*_sync.
 *
 * The optimizer is templatized on an energy/gradient functor:
 *   float functor(float const* positions, float* gradient,
 *                 std::uint32_t num_elements, std::uint32_t lane) const;
 *   Returns total energy. Writes ∂E/∂xᵢ into gradient array.
 *
 * Features over the original mmff94.cuh version:
 *   - Energy-functor-generic (template parameter, not hard-coded MMFF94)
 *   - Cautious update (Li & Fukushima, 2001): skips history when yᵀs < ε·‖s‖²
 *   - H₀ scaling: γ = sᵀy / yᵀy (already present in original, preserved here)
 *
 * Target: SM 9.0+ (H100, Blackwell). Requires CUDA 12+.
 */

#pragma once

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

namespace smallfield {

constexpr std::uint32_t warp_size = 32;
constexpr unsigned int full_mask = 0xFFFFFFFF;

/**
 * Warp reduction via butterfly XOR shuffles.
 * Five shuffles to reduce 32 lanes. Result available in all lanes.
 */
__device__ __forceinline__ float warp_reduce_sum_f32_(float value) {
    value += __shfl_xor_sync(full_mask, value, 16);
    value += __shfl_xor_sync(full_mask, value, 8);
    value += __shfl_xor_sync(full_mask, value, 4);
    value += __shfl_xor_sync(full_mask, value, 2);
    value += __shfl_xor_sync(full_mask, value, 1);
    return value;
}

/**
 * Compute workspace size (in floats) for L-BFGS history and scratch.
 *
 * Layout:
 *   s_history:     memory_depth * num_elements
 *   y_history:     memory_depth * num_elements
 *   rho:           memory_depth
 *   alpha:         memory_depth
 *   gradient_prev: num_elements
 */
__device__ __forceinline__ std::uint32_t lbfgs_workspace_size_(std::uint32_t num_atoms, std::uint32_t memory_depth) {
    return memory_depth * num_atoms * 3 * 2 // s_history + y_history
           + memory_depth                   // rho
           + memory_depth                   // alpha
           + num_atoms * 3;                 // gradient_prev
}

/**
 * Dot product via warp reduction.
 * Each thread accumulates strided elements, then butterfly-reduces.
 * All lanes receive the same result.
 */
__device__ __forceinline__ float warp_dot_product_f32_(float const* a, float const* b, std::uint32_t num_elements,
                                                       std::uint32_t lane) {
    float dot = 0.0f;
    for (std::uint32_t i = lane; i < num_elements; i += warp_size) {
        dot = fmaf(a[i], b[i], dot);
    }
    return warp_reduce_sum_f32_(dot);
}

/**
 * Root mean square via warp reduction.
 * rms = √(Σᵢ xᵢ² / n)
 */
__device__ __forceinline__ float warp_rms_f32_(float const* x, std::uint32_t num_elements, std::uint32_t lane) {
    float sum_squared = 0.0f;
    for (std::uint32_t i = lane; i < num_elements; i += warp_size) {
        float const value = x[i];
        sum_squared = fmaf(value, value, sum_squared);
    }
    float const total = warp_reduce_sum_f32_(sum_squared);
    return sqrtf(total * __frcp_rn(static_cast<float>(num_elements)));
}

/**
 * L-BFGS two-loop recursion for computing the search direction.
 *
 * Given gradient g and history pairs (sₖ, yₖ, ρₖ), computes d = -H·g
 * where H is the inverse Hessian approximation from Nocedal & Wright, Algorithm 7.4.
 *
 * H₀ scaling (Equation 7.20): γ = sₖ₋₁ᵀyₖ₋₁ / yₖ₋₁ᵀyₖ₋₁
 * This matches the Hessian's scale and typically reduces iteration count by 30-50%.
 */
__device__ __forceinline__ void lbfgs_two_loop_recursion_(float const* gradient, float* search_direction,
                                                          float* s_history, float* y_history, float* rho, float* alpha,
                                                          std::uint32_t num_elements, std::uint32_t history_count,
                                                          std::uint32_t memory_depth, std::uint32_t lane) {

    // q = gradient
    for (std::uint32_t i = lane; i < num_elements; i += warp_size) {
        search_direction[i] = gradient[i];
    }
    __syncwarp(full_mask);

    // first loop (backward through history)
    for (std::uint32_t i = 0; i < history_count; ++i) {
        std::uint32_t const index = (history_count - 1 - i) % memory_depth;
        float* s = s_history + index * num_elements;
        float* y = y_history + index * num_elements;

        float const rho_value = rho[index];
        float const dot = warp_dot_product_f32_(s, search_direction, num_elements, lane);

        float alpha_value = 0.0f;
        if (lane == 0) {
            alpha_value = rho_value * dot;
            alpha[index] = alpha_value;
        }
        alpha_value = __shfl_sync(full_mask, alpha_value, 0);

        // q = q - αᵢ·yᵢ
        for (std::uint32_t j = lane; j < num_elements; j += warp_size) {
            search_direction[j] -= alpha_value * y[j];
        }
        __syncwarp(full_mask);
    }

    // H₀ scaling: r = γ·q  where γ = sₖ₋₁ᵀyₖ₋₁ / yₖ₋₁ᵀyₖ₋₁
    if (history_count > 0) {
        std::uint32_t const last_index = (history_count - 1) % memory_depth;
        float* s_last = s_history + last_index * num_elements;
        float* y_last = y_history + last_index * num_elements;

        float const s_dot_y = warp_dot_product_f32_(s_last, y_last, num_elements, lane);
        float const y_dot_y = warp_dot_product_f32_(y_last, y_last, num_elements, lane);

        float gamma = 1.0f;
        if (lane == 0) {
            gamma = s_dot_y * __frcp_rn(y_dot_y);
        }
        gamma = __shfl_sync(full_mask, gamma, 0);

        for (std::uint32_t i = lane; i < num_elements; i += warp_size) {
            search_direction[i] *= gamma;
        }
        __syncwarp(full_mask);
    }

    // second loop (forward through history)
    for (std::uint32_t i = 0; i < history_count; ++i) {
        std::uint32_t const index = i % memory_depth;
        float* s = s_history + index * num_elements;
        float* y = y_history + index * num_elements;

        float const rho_value = rho[index];
        float const alpha_value = alpha[index];
        float const dot = warp_dot_product_f32_(y, search_direction, num_elements, lane);

        float beta = 0.0f;
        if (lane == 0) {
            beta = rho_value * dot;
        }
        beta = __shfl_sync(full_mask, beta, 0);

        float const coefficient = alpha_value - beta;
        for (std::uint32_t j = lane; j < num_elements; j += warp_size) {
            search_direction[j] = fmaf(s[j], coefficient, search_direction[j]);
        }
        __syncwarp(full_mask);
    }

    // negate: d = -H·g (descent direction)
    for (std::uint32_t i = lane; i < num_elements; i += warp_size) {
        search_direction[i] = -search_direction[i];
    }
    __syncwarp(full_mask);
}

/**
 * Backtracking line search with Armijo condition.
 *
 * Finds step size α satisfying:
 *   E(x + α·d) ≤ E(x) + c₁·α·(∇E·d)
 *
 * The energy evaluation is delegated to the functor, making this
 * reusable across force fields (MMFF94, DG, ETK).
 *
 * @tparam EnergyEvaluator  Callable: float(float const* trial_positions, std::uint32_t lane)
 */
template <typename EnergyEvaluator>
__device__ __forceinline__ float lbfgs_line_search_(float const* positions, float const* search_direction,
                                                    float const* gradient, std::uint32_t num_elements,
                                                    float current_energy, std::uint32_t lane, float* trial_positions,
                                                    EnergyEvaluator&& evaluate_energy) {

    constexpr float armijo_parameter = 1e-4f;
    constexpr float initial_step_size = 1.0f;
    constexpr float backtrack_factor = 0.5f;
    constexpr int max_line_search_iterations = 20;

    float const directional_derivative = warp_dot_product_f32_(gradient, search_direction, num_elements, lane);

    float step_size = initial_step_size;

    for (int iteration = 0; iteration < max_line_search_iterations; ++iteration) {
        // trial position: x_trial = x + α·d
        for (std::uint32_t i = lane; i < num_elements; i += warp_size) {
            trial_positions[i] = positions[i] + step_size * search_direction[i];
        }
        __syncwarp(full_mask);

        float const trial_energy = evaluate_energy(trial_positions, lane);

        // Armijo condition: E(x + α·d) ≤ E(x) + c₁·α·(∇E·d)
        float armijo_threshold = 0.0f;
        if (lane == 0) {
            armijo_threshold = current_energy + armijo_parameter * step_size * directional_derivative;
        }
        armijo_threshold = __shfl_sync(full_mask, armijo_threshold, 0);

        if (trial_energy <= armijo_threshold) {
            return step_size;
        }

        step_size *= backtrack_factor;
    }

    return step_size;
}

/**
 * Generic L-BFGS minimization driver.
 *
 * Runs the full L-BFGS loop: energy/gradient evaluation, convergence check,
 * two-loop recursion, line search, position and history update.
 *
 * Includes cautious update (Li & Fukushima, 2001): skips the history pair
 * when the curvature condition yᵀs < ε·‖s‖² is poorly satisfied,
 * preventing bad search directions near saddle points.
 *
 * @tparam EnergyGradientFunctor  Callable with signature:
 *   float operator()(float const* positions, float* gradient,
 *                    std::uint32_t num_elements, std::uint32_t lane) const;
 *
 * @param positions          Coordinate array (modified in place). Length = num_elements.
 * @param gradient           Shared memory for gradient. Length = num_elements.
 * @param search_direction   Shared memory for search direction. Length = num_elements.
 * @param trial_positions    Shared memory for line search trial. Length = num_elements.
 * @param workspace          Global or shared memory for L-BFGS history.
 *                           Size: lbfgs_workspace_size_(num_elements/3, memory_depth) floats.
 * @param num_elements       Total coordinate dimensions (e.g., 3*N or 4*N).
 * @param memory_depth       Number of history pairs (m). Typically 5-10.
 * @param max_iterations     Maximum L-BFGS iterations.
 * @param gradient_threshold Convergence when rms(gradient) < threshold.
 * @param lane               Warp lane index (threadIdx.x % 32).
 * @param compute_energy_gradient  The energy/gradient functor.
 * @param[out] out_final_energy    Final energy (written by lane 0).
 * @param[out] out_iterations      Iteration count (written by lane 0).
 * @param[out] out_status          0=converged, 1=max_iterations, 2=line_search_failed (lane 0).
 */
template <typename EnergyGradientFunctor>
__device__ __forceinline__ void
lbfgs_minimize_(float* positions, float* gradient, float* search_direction, float* trial_positions, float* workspace,
                std::uint32_t num_elements, std::uint32_t memory_depth, std::int32_t max_iterations,
                float gradient_threshold, std::uint32_t lane, EnergyGradientFunctor&& compute_energy_gradient,
                float* out_final_energy, std::int32_t* out_iterations, std::int32_t* out_status) {

    // workspace layout (all in global or shared memory)
    float* s_history = workspace;
    float* y_history = s_history + memory_depth * num_elements;
    float* rho = y_history + memory_depth * num_elements;
    float* alpha_scratch = rho + memory_depth;
    float* gradient_prev = alpha_scratch + memory_depth;

    // zero gradient arrays
    for (std::uint32_t i = lane; i < num_elements; i += warp_size) {
        gradient[i] = 0.0f;
        gradient_prev[i] = 0.0f;
    }
    __syncwarp(full_mask);

    std::uint32_t history_count = 0;
    constexpr float cautious_update_threshold = 1e-6f;

    for (std::int32_t iteration = 0; iteration < max_iterations; ++iteration) {

        // evaluate energy and gradient
        float const energy = compute_energy_gradient(positions, gradient, num_elements, lane);
        __syncwarp(full_mask);

        // convergence check: rms(∇E) < threshold
        float const rms_gradient = warp_rms_f32_(gradient, num_elements, lane);
        if (rms_gradient < gradient_threshold) {
            if (lane == 0) {
                *out_status = 0; // converged
                *out_iterations = iteration;
                *out_final_energy = energy;
            }
            return;
        }

        // compute search direction via L-BFGS two-loop recursion
        lbfgs_two_loop_recursion_(gradient, search_direction, s_history, y_history, rho, alpha_scratch, num_elements,
                                  history_count, memory_depth, lane);

        // line search: energy-only evaluation (no gradient needed, saves ~40% per step)
        auto energy_evaluator = [&](float const* trial_pos, std::uint32_t ln) -> float {
            return compute_energy_gradient.energy_only(trial_pos, num_elements, ln);
        };
        float const step_size = lbfgs_line_search_(positions, search_direction, gradient, num_elements, energy, lane,
                                                   trial_positions, energy_evaluator);

        // update positions: x = x + α·d
        for (std::uint32_t i = lane; i < num_elements; i += warp_size) {
            positions[i] += step_size * search_direction[i];
        }
        __syncwarp(full_mask);

        // update L-BFGS history (cautious update)
        if (iteration > 0) {
            std::uint32_t const history_index = history_count % memory_depth;
            float* s_vector = s_history + history_index * num_elements;
            float* y_vector = y_history + history_index * num_elements;

            float step_gradient_dot = 0.0f;
            float step_norm_squared = 0.0f;

            for (std::uint32_t i = lane; i < num_elements; i += warp_size) {
                float const s_value = step_size * search_direction[i];
                float const y_value = gradient[i] - gradient_prev[i];
                s_vector[i] = s_value;
                y_vector[i] = y_value;
                step_gradient_dot = fmaf(s_value, y_value, step_gradient_dot);
                step_norm_squared = fmaf(s_value, s_value, step_norm_squared);
            }

            step_gradient_dot = warp_reduce_sum_f32_(step_gradient_dot);
            step_norm_squared = warp_reduce_sum_f32_(step_norm_squared);

            // cautious update (Li & Fukushima, 2001):
            // skip when curvature yᵀs is too weak relative to ‖s‖²
            bool const curvature_sufficient = step_gradient_dot > cautious_update_threshold * step_norm_squared;

            if (curvature_sufficient) {
                if (lane == 0) {
                    rho[history_index] = __frcp_rn(step_gradient_dot);
                }
                history_count = (history_count < memory_depth) ? history_count + 1 : memory_depth;
            }
            // if curvature insufficient, s/y vectors are abandoned (overwritten next time)
        }

        // save current gradient for next iteration's y = g_{k+1} - g_k
        for (std::uint32_t i = lane; i < num_elements; i += warp_size) {
            gradient_prev[i] = gradient[i];
        }
        __syncwarp(full_mask);
    }

    // max iterations reached
    float const final_rms = warp_rms_f32_(gradient, num_elements, lane);
    if (lane == 0) {
        *out_status = 1; // max iterations
        *out_iterations = max_iterations;
        *out_final_energy = 0.0f; // caller can re-evaluate if needed
    }
}

} // namespace smallfield

#endif // __CUDACC__
