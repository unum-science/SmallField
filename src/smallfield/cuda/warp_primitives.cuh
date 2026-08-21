/**
 *  @file src/smallfield/cuda/warp_primitives.cuh
 *  @brief Shared warp-level constants, reduction primitives, and float3 vector helpers.
 *
 *  Multi-dimensional views live in `smallfield/types/views.hpp`, which host code can include too, and
 *  `sf_assert_` lives in `smallfield/utilities.hpp` for the same reason.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

namespace smallfield::cuda {

/*  These two sit outside the `__CUDACC__` guard, unlike everything else here, because they are hardware facts
 *  rather than device code: a host translation unit sizing a launch or a stride has to name the same constant
 *  the kernel does, and a guard `g++` never takes would push it back to a bare `32`. */

/** Threads a warp carries on every architecture this project targets. */
constexpr std::uint32_t warp_size_k = 32;

/** Every lane of a full warp, for the `__shfl_*_sync` and `__ballot_sync` mask argument. */
constexpr unsigned int full_mask_k = 0xFFFFFFFF;

/**
 *  @brief `|sin φ|` below which a torsion's chain-rule factor takes its L'Hopital limit instead.
 *
 *  Both torsion terms divide their first derivative by `sin φ`, and numerator and denominator both vanish at
 *  φ = 0 and φ = pi. One threshold serves both kernels, so the limit behind it is written once.
 */
constexpr float torsion_planarity_threshold_k = 1e-6f;

} // namespace smallfield::cuda

#if defined(__CUDACC__)

#include <cuda_runtime.h>

#include "smallfield/utilities.hpp"

namespace smallfield::cuda {

__device__ __forceinline__ float dot_float3_(float3 const &lhs, float3 const &rhs) {
    return fmaf(lhs.x, rhs.x, fmaf(lhs.y, rhs.y, lhs.z * rhs.z));
}

__device__ __forceinline__ float3 cross_float3_(float3 const &lhs, float3 const &rhs) {
    return {lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x};
}

__device__ __forceinline__ float3 add_float3_(float3 const &lhs, float3 const &rhs) {
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

__device__ __forceinline__ float3 sub_float3_(float3 const &lhs, float3 const &rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

__device__ __forceinline__ float3 scale_float3_(float3 const &value, float scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

/** Load atom @p atom from a SoA position array [x0..xN, y0..yN, z0..zN] of length 3*num_atoms. */
__device__ __forceinline__ float3 load_position_(float const *positions, std::uint32_t num_atoms, std::uint32_t atom) {
    return {positions[atom], positions[num_atoms + atom], positions[2u * num_atoms + atom]};
}

/** Store @p value into a SoA gradient array [x0..xN, y0..yN, z0..zN] at @p atom (mirrors load_position_). */
__device__ __forceinline__ void store_gradient_(float *gradient, std::uint32_t num_atoms, std::uint32_t atom,
                                                float3 const &value) {
    gradient[atom] = value.x;
    gradient[num_atoms + atom] = value.y;
    gradient[2u * num_atoms + atom] = value.z;
}

/** Add @p value into a SoA gradient array [x0..xN, y0..yN, z0..zN] at @p atom (mirrors store_gradient_). */
__device__ __forceinline__ void accumulate_gradient_into_(float *gradient, std::uint32_t num_atoms, std::uint32_t atom,
                                                          float3 const &value) {
    gradient[atom] += value.x;
    gradient[num_atoms + atom] += value.y;
    gradient[2u * num_atoms + atom] += value.z;
}

/** Add @p delta into a per-atom float3 accumulator held in registers. */
__device__ __forceinline__ void accumulate_gradient_(float3 *gradient, float3 const &delta) {
    *gradient = add_float3_(*gradient, delta);
}

/**
 *  @brief Partial derivative of cos(θ) with respect to the @p this_atom position.
 *
 *  For angle θ at center j between neighbors @p this_atom and @p other_atom, with `this_unit`/`other_unit` the
 *  bond unit vectors *from* j and `inverse_distance_this` = 1 / |r_this - r_j|:
 *
 *      ∂cosθ/∂r_this = (other_unit − cosθ · this_unit) / |r_this - r_j|
 *
 *  The center-atom gradient is `-(grad_first + grad_third)`. Used by the angle and stretch-bend terms.
 */
__device__ __forceinline__ float3 angle_cosine_gradient_(float3 const &this_unit, float3 const &other_unit,
                                                         float cos_theta, float inverse_distance_this) {
    return scale_float3_(sub_float3_(other_unit, scale_float3_(this_unit, cos_theta)), inverse_distance_this);
}

/** Warp reduction via butterfly XOR shuffles. All lanes get the result. */
__device__ __forceinline__ float warp_reduce_sum_f32_(float value) {
    value += __shfl_xor_sync(full_mask_k, value, 16);
    value += __shfl_xor_sync(full_mask_k, value, 8);
    value += __shfl_xor_sync(full_mask_k, value, 4);
    value += __shfl_xor_sync(full_mask_k, value, 2);
    value += __shfl_xor_sync(full_mask_k, value, 1);
    return value;
}

/** Warp min reduction via butterfly XOR shuffles. All lanes get the result. */
__device__ __forceinline__ float warp_reduce_min_f32_(float value) {
    value = fminf(value, __shfl_xor_sync(full_mask_k, value, 16));
    value = fminf(value, __shfl_xor_sync(full_mask_k, value, 8));
    value = fminf(value, __shfl_xor_sync(full_mask_k, value, 4));
    value = fminf(value, __shfl_xor_sync(full_mask_k, value, 2));
    value = fminf(value, __shfl_xor_sync(full_mask_k, value, 1));
    return value;
}

/** Warp max reduction via butterfly XOR shuffles. All lanes get the result. */
__device__ __forceinline__ float warp_reduce_max_f32_(float value) {
    value = fmaxf(value, __shfl_xor_sync(full_mask_k, value, 16));
    value = fmaxf(value, __shfl_xor_sync(full_mask_k, value, 8));
    value = fmaxf(value, __shfl_xor_sync(full_mask_k, value, 4));
    value = fmaxf(value, __shfl_xor_sync(full_mask_k, value, 2));
    value = fmaxf(value, __shfl_xor_sync(full_mask_k, value, 1));
    return value;
}

} // namespace smallfield::cuda

#endif // __CUDACC__
