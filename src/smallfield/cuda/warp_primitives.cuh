/**
 *  @file warp_primitives.cuh
 *  @brief Shared warp-level constants, reduction primitives, float3 vector helpers,
 *         and tiny tensor-view types with debug-mode bounds checking.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

namespace smallfield::cuda {

constexpr std::uint32_t warp_size_k = 32;
constexpr unsigned int full_mask_k = 0xFFFFFFFF;

// Debug-only assertion that compiles to nothing in Release.  Use this to
// guard pointer-math accesses (e.g. matrix_view_2d::operator()) so OOB
// derefs trap immediately under compute-sanitizer / a debugger instead of
// silently corrupting memory and crashing later somewhere unrelated.
//
//  - Device path: __trap() so compute-sanitizer captures the kernel
//    offender with full thread/block coordinates.
//  - Host path: __builtin_trap() so gdb / a core dump points at the bad
//    caller.
//  - Release (NDEBUG): no-op; same SASS as raw pointer math.
#if defined(NDEBUG)
#define sf_assert_(cond) ((void)0)
#elif defined(__CUDA_ARCH__)
#define sf_assert_(cond)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) { __trap(); }                                                                                     \
    } while (0)
#else
#define sf_assert_(cond)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) { __builtin_trap(); }                                                                             \
    } while (0)
#endif

/**
 *  @brief Type-safe view of a row-major N×N matrix.
 *
 *  12 bytes (pointer + uint32 stride).  Trivially copyable so it passes
 *  through `__global__` kernel arguments without a dance.  In Release
 *  the @ref operator() compiles to the same SASS as raw `data[i*n+j]`.
 *  In Debug (`!NDEBUG`) it asserts `i < n && j < n` and traps on OOB —
 *  this is the guardrail that catches stride / offset bugs before they
 *  silently corrupt neighbouring memory.
 */
template <typename value_type_>
struct matrix_view_2d {
    value_type_ *data = nullptr;
    std::uint32_t n = 0;

    constexpr matrix_view_2d() noexcept = default;
    __host__ __device__ constexpr matrix_view_2d(value_type_ *data_, std::uint32_t n_) noexcept
        : data(data_), n(n_) {}

    __host__ __device__ __forceinline__ value_type_ &operator()(std::uint32_t i, std::uint32_t j) const noexcept {
        sf_assert_(i < n && j < n);
        return data[i * n + j];
    }
};

/**
 *  @brief Type-safe view of a SoA `[x0..xN, y0..yN, z0..zN]` 3-component
 *  buffer.  Mirrors @ref load_position_ / @ref store_gradient_ semantics
 *  with the same Debug-mode bounds check as @ref matrix_view_2d.
 */
template <typename value_type_>
struct soa_view_3d {
    value_type_ *data = nullptr;
    std::uint32_t num_atoms = 0;

    constexpr soa_view_3d() noexcept = default;
    __host__ __device__ constexpr soa_view_3d(value_type_ *data_, std::uint32_t num_atoms_) noexcept
        : data(data_), num_atoms(num_atoms_) {}

    __host__ __device__ __forceinline__ value_type_ &component(std::uint32_t dim, std::uint32_t atom) const noexcept {
        sf_assert_(dim < 3u && atom < num_atoms);
        return data[dim * num_atoms + atom];
    }
};

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
__device__ __forceinline__ float3 load_position_(float const *positions, std::uint32_t num_atoms,
                                                 std::uint32_t atom) {
    return {positions[atom], positions[num_atoms + atom], positions[2u * num_atoms + atom]};
}

/** Store @p value into a SoA gradient array [x0..xN, y0..yN, z0..zN] at @p atom (mirrors load_position_). */
__device__ __forceinline__ void store_gradient_(float *gradient, std::uint32_t num_atoms, std::uint32_t atom,
                                                float3 const &value) {
    gradient[atom] = value.x;
    gradient[num_atoms + atom] = value.y;
    gradient[2u * num_atoms + atom] = value.z;
}

/** Add @p delta into a per-atom float3 accumulator held in registers. */
__device__ __forceinline__ void accumulate_gradient_(float3 *gradient, float3 const &delta) {
    *gradient = add_float3_(*gradient, delta);
}

/**
 *  @brief Partial derivative of cos(θ) with respect to the @p this_atom position.
 *
 *  For an angle θ formed at a center j by neighbors @p this_atom and @p other_atom, where
 *  `this_unit` and `other_unit` are the corresponding bond unit vectors *from* j and
 *  `inverse_distance_this` is `1 / |r_{this,j}|`:
 *
 *      ∂cosθ/∂r_this = (other_unit − cosθ · this_unit) / |r_{this,j}|
 *
 *  Used by mmff_angle_contribution_ and mmff_stretchbend_contribution_; the
 *  center-atom gradient is just `−(grad_first + grad_third)` from this formula.
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
