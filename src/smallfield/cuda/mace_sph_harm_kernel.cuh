/**
 *  @file mace_sph_harm_kernel.cuh
 *  @brief Real spherical harmonics y_lm(r̂) for L = 0..3, e3nn convention.
 *
 *  Direct port of e3nn's `_spherical_harmonics(lmax=3, x, y, z)` (see
 *  `e3nn/o3/_spherical_harmonics.py:184`) plus `component` normalisation
 *  (each L block multiplied by sqrt(2L+1)).
 *
 *  Output ordering and signs match e3nn exactly; verified bit-for-bit
 *  against the golden trace (`golden_*.bin`, tensor key `sph_harm`).
 *
 *  Memory: 64 bytes per edge (16 × fp32).  Pure compute, no global-memory
 *  pressure beyond the 16 stores and 3 loads of `r_unit`.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

namespace smallfield::cuda {

namespace mace_sph_detail_ {

// e3nn `component`-normalised real spherical harmonics, L=0..3.
// Layout: 16 floats per edge in the order
//   L=0: [0]
//   L=1: [1, 2, 3]
//   L=2: [4, 5, 6, 7, 8]
//   L=3: [9, 10, 11, 12, 13, 14, 15]
__device__ __forceinline__ void sph_harm_l0123_(
    float x, float y, float z,
    float *y_out  // [16]
) {
    // Unscaled e3nn values, exactly as in `_spherical_harmonics(lmax=3, ...)`.
    float const sh_0_0 = 1.0f;
    float const sh_1_0 = x;
    float const sh_1_1 = y;
    float const sh_1_2 = z;

    constexpr float kSqrt3 = 1.7320508075688772f;
    float const y2   = y * y;
    float const x2z2 = x * x + z * z;
    float const sh_2_0 = kSqrt3 * x * z;
    float const sh_2_1 = kSqrt3 * x * y;
    float const sh_2_2 = y2 - 0.5f * x2z2;
    float const sh_2_3 = kSqrt3 * y * z;
    float const sh_2_4 = (kSqrt3 * 0.5f) * (z * z - x * x);

    constexpr float kSqrt5o6 = 0.9128709291752769f;   // sqrt(5/6)
    constexpr float kSqrt5   = 2.2360679774997896f;   // sqrt(5)
    constexpr float kSqrt3o8 = 0.6123724356957945f;   // sqrt(3/8)
    float const sh_3_0 = kSqrt5o6 * (sh_2_0 * z + sh_2_4 * x);
    float const sh_3_1 = kSqrt5   * sh_2_0 * y;
    float const sh_3_2 = kSqrt3o8 * (4.0f * y2 - x2z2) * x;
    float const sh_3_3 = 0.5f * y * (2.0f * y2 - 3.0f * x2z2);
    float const sh_3_4 = kSqrt3o8 * z * (4.0f * y2 - x2z2);
    float const sh_3_5 = kSqrt5   * sh_2_4 * y;
    float const sh_3_6 = kSqrt5o6 * (sh_2_4 * z - sh_2_0 * x);

    // Component normalisation: multiply each L block by sqrt(2L+1).
    constexpr float kC0 = 1.0f;                       // sqrt(1)
    constexpr float kC1 = 1.7320508075688772f;        // sqrt(3)
    constexpr float kC2 = 2.2360679774997896f;        // sqrt(5)
    constexpr float kC3 = 2.6457513110645907f;        // sqrt(7)

    y_out[ 0] = kC0 * sh_0_0;
    y_out[ 1] = kC1 * sh_1_0;
    y_out[ 2] = kC1 * sh_1_1;
    y_out[ 3] = kC1 * sh_1_2;
    y_out[ 4] = kC2 * sh_2_0;
    y_out[ 5] = kC2 * sh_2_1;
    y_out[ 6] = kC2 * sh_2_2;
    y_out[ 7] = kC2 * sh_2_3;
    y_out[ 8] = kC2 * sh_2_4;
    y_out[ 9] = kC3 * sh_3_0;
    y_out[10] = kC3 * sh_3_1;
    y_out[11] = kC3 * sh_3_2;
    y_out[12] = kC3 * sh_3_3;
    y_out[13] = kC3 * sh_3_4;
    y_out[14] = kC3 * sh_3_5;
    y_out[15] = kC3 * sh_3_6;
}

} // namespace mace_sph_detail_

inline __global__ void mace_sph_harm_kernel(
    std::uint32_t n_edges,
    float const *r_unit,
    float *sph_harm
) {
    std::uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_index >= n_edges) return;
    float const x = r_unit[3 * edge_index + 0];
    float const y = r_unit[3 * edge_index + 1];
    float const z = r_unit[3 * edge_index + 2];
    mace_sph_detail_::sph_harm_l0123_(x, y, z, sph_harm + 16 * edge_index);
}

inline void launch_mace_sph_harm(
    std::uint32_t n_edges,
    float const *r_unit,
    float *sph_harm,
    cudaStream_t stream
) {
    if (n_edges == 0) return;
    constexpr std::uint32_t block = 128;
    std::uint32_t const grid = (n_edges + block - 1u) / block;
    mace_sph_harm_kernel<<<grid, block, 0, stream>>>(n_edges, r_unit, sph_harm);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
