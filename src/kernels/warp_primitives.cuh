/**
 *  @file warp_primitives.cuh
 *  @brief Shared warp-level constants and reduction primitives.
 */

#pragma once

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

namespace smallfield {

constexpr std::uint32_t warp_size = 32;
constexpr unsigned int full_mask = 0xFFFFFFFF;

/** Warp reduction via butterfly XOR shuffles. All lanes get the result. */
__device__ __forceinline__ float warp_reduce_sum_f32_(float value) {
    value += __shfl_xor_sync(full_mask, value, 16);
    value += __shfl_xor_sync(full_mask, value, 8);
    value += __shfl_xor_sync(full_mask, value, 4);
    value += __shfl_xor_sync(full_mask, value, 2);
    value += __shfl_xor_sync(full_mask, value, 1);
    return value;
}

} // namespace smallfield

#endif // __CUDACC__
