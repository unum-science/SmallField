/**
 *  @file warp_primitives.cuh
 *  @brief Shared warp-level constants and reduction primitives.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

namespace smallfield::cuda {

constexpr std::uint32_t warp_size_k = 32;
constexpr unsigned int full_mask_k = 0xFFFFFFFF;

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
