/**
 *  @file mace_fp16_io_kernel.cuh
 *  @brief IO cast layer between fp16 geometry storage and fp32 MACE working precision.
 *
 *  SmallField's coprocessed-batch upstream stores atomic positions in fp16 to
 *  halve the geometry-buffer footprint in the LBFGS / ETKDG pipeline; fp16
 *  dynamic range is ample for atomic-Å-scale coordinates (a typical drug-like
 *  molecule fits in a 30 Å box and fp16 resolves ~1e-3 Å there).  The MACE
 *  module's per-edge geometry kernels (`mace_edge_geometry_kernel`,
 *  `mace_radial_basis_kernel`, `mace_sph_harm_kernel`) all run in fp32, so we
 *  expand fp16 → fp32 at batch ingest and quantise the produced fp32 forces
 *  back to fp16 at egress for the downstream geometry update.  Both casts
 *  are trivial elementwise kernels — one thread per element.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace smallfield::cuda {

/**
 *  @brief Expand a flat fp16 position tensor into fp32.
 *
 *  `n` is `total_atom_count * 3` — the kernel does not care about per-atom
 *  layout, just the flat element count.  One thread per element.
 */
inline __global__ void mace_positions_fp16_to_fp32_kernel(
    std::uint32_t n,
    __half const *positions_fp16,
    float *positions_fp32
) {
    std::uint32_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index >= n) return;
    positions_fp32[element_index] = __half2float(positions_fp16[element_index]);
}

/**
 *  @brief Quantise a flat fp32 force tensor to fp16 for downstream storage.
 *
 *  `n` is `total_atom_count * 3`.  One thread per element.
 */
inline __global__ void mace_forces_fp32_to_fp16_kernel(
    std::uint32_t n,
    float const *forces_fp32,
    __half *forces_fp16
) {
    std::uint32_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index >= n) return;
    forces_fp16[element_index] = __float2half(forces_fp32[element_index]);
}

inline void launch_mace_positions_fp16_to_fp32(
    std::uint32_t n,
    __half const *positions_fp16,
    float *positions_fp32,
    cudaStream_t stream
) {
    if (n == 0) return;
    constexpr std::uint32_t block = 256;
    std::uint32_t const grid = (n + block - 1u) / block;
    mace_positions_fp16_to_fp32_kernel<<<grid, block, 0, stream>>>(n, positions_fp16, positions_fp32);
}

inline void launch_mace_forces_fp32_to_fp16(
    std::uint32_t n,
    float const *forces_fp32,
    __half *forces_fp16,
    cudaStream_t stream
) {
    if (n == 0) return;
    constexpr std::uint32_t block = 256;
    std::uint32_t const grid = (n + block - 1u) / block;
    mace_forces_fp32_to_fp16_kernel<<<grid, block, 0, stream>>>(n, forces_fp32, forces_fp16);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
