/**
 *  @file warp_primitives.cuh
 *  @brief Shared warp-level constants, reduction primitives, and float3 vector helpers.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

namespace smallfield::cuda {

constexpr std::uint32_t warp_size_k = 32;
constexpr unsigned int full_mask_k = 0xFFFFFFFF;

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
