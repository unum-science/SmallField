/**
 *  @file mace_linear_kernel.cuh
 *  @brief Hand-rolled per-irrep linear (channel mixing) for MACE-OFF.
 *
 *  e3nn `Linear(c_in × Σ_L 1xL, c_out × Σ_L 1xL)` is per-L-block independent:
 *  for each L block both in input and output, there is a `(c_out, c_in)`
 *  weight matrix, applied identically across the (2L+1) m-components of that
 *  block.  The same applies to the in/out channel multiplicities.
 *
 *  The internal layout we use throughout the MACE inference module is
 *  `[n_atoms, n_channels, irrep_dim_total = 16]` with the irrep-dim index
 *  laid out as `[L=0 (1 m), L=1 (3 m), L=2 (5 m), L=3 (7 m)]`.  That matches
 *  the `reshape_irreps` output of MACE's e3nn-format flat layout
 *  `[n_atoms, sum_L mul_L * (2L+1) = 1536]`.
 *
 *  The kernel:
 *
 *      out[a, c_out, m]  =  Σ_{c_in}  W[L(m), c_out, c_in] * in[a, c_in, m]
 *
 *  Threading: one thread per (atom, c_out, m).  ~24k threads × 96 inner
 *  multiplies × 16 m × 4 L-blocks-per-pass = ~150 MFLOPs per layer.  Well
 *  under H100's per-second budget; cheap to do hand-rolled before swapping
 *  to a cuBLAS gemm-batched call later.
 *
 *  `weight_blocks` points to `[n_blocks, c_out, c_in]` where the L→block
 *  mapping is given by the canonical irrep ordering for L_max=3:
 *      L=0 → block 0 (m=0)
 *      L=1 → block 1 (m=1..3)
 *      L=2 → block 2 (m=4..8)
 *      L=3 → block 3 (m=9..15)
 *
 *  If only some blocks are populated (e.g. `linear_up` has only the L=0
 *  block in the weights blob), pass `active_irrep_block_count = 1` and the
 *  kernel zeros the higher blocks.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

#include "smallfield/types/mace_types.hpp"

namespace smallfield::cuda {

/** Block index for a given m-component index, irrep_dim_total = 16. */
__device__ __forceinline__ std::uint32_t mace_irrep_block_index_for_component_(std::uint32_t irrep_component_index) {
    if (irrep_component_index == 0) return 0;
    if (irrep_component_index <= 3) return 1;
    if (irrep_component_index <= 8) return 2;
    return 3;
}

/**
 *  @brief Per-irrep linear, hand-rolled.
 *
 *  Writes `out[a, c_out, m] = Σ_{c_in} W[L(m), c_out, c_in] * in[a, c_in, m]`
 *  for every active L-block.  m indices in inactive blocks (m >= active
 *  blocks' m_total) get 0.
 */
__global__ void mace_linear_per_irrep_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_channels_out,
    std::uint32_t irrep_dim,                  // 16
    std::uint32_t active_irrep_block_count,   // 1 (0e only) or 4 (all)
    float scale,                              // e3nn path_weight = 1/sqrt(c_in) for one-instruction-per-output Linear
    float const *weight_blocks,               // [n_blocks, c_out, c_in]
    float const *input,                       // [n_atoms, c_in, irrep_dim]
    float *output                             // [n_atoms, c_out, irrep_dim]
) {
    std::uint32_t atom_index = blockIdx.x;
    std::uint32_t channel_out_index = blockIdx.y;
    std::uint32_t irrep_component_index = threadIdx.x;
    if (atom_index >= n_atoms || channel_out_index >= n_channels_out || irrep_component_index >= irrep_dim) return;

    std::uint32_t const irrep_block_index = mace_irrep_block_index_for_component_(irrep_component_index);
    if (irrep_block_index >= active_irrep_block_count) {
        output[(static_cast<std::uint64_t>(atom_index) * n_channels_out + channel_out_index) * irrep_dim + irrep_component_index] = 0.0f;
        return;
    }
    float accumulator = 0.0f;
    // e3nn `o3.Linear` stores per-block weights as `(c_in, c_out)` row-major:
    //   weight_blocks[irrep_block_index, c_in, c_out].  The element offset is
    //   irrep_block_index*c_in*c_out + c_in*c_out + c_out.
    std::uint64_t const weight_block_offset = static_cast<std::uint64_t>(irrep_block_index) * n_channels_in * n_channels_out;
    float const *input_atom_row = input + static_cast<std::uint64_t>(atom_index) * n_channels_in * irrep_dim;
    for (std::uint32_t channel_in_index = 0; channel_in_index < n_channels_in; ++channel_in_index) {
        float const w = weight_blocks[weight_block_offset + static_cast<std::uint64_t>(channel_in_index) * n_channels_out + channel_out_index];
        accumulator += w * input_atom_row[channel_in_index * irrep_dim + irrep_component_index];
    }
    output[(static_cast<std::uint64_t>(atom_index) * n_channels_out + channel_out_index) * irrep_dim + irrep_component_index] = scale * accumulator;
}

/// Forward-declare the row-major matmul launcher so the irrep_dim=1 fast path
/// below can dispatch to it without reordering the file.
__global__ void mace_matmul_rowmajor_kernel(
    std::uint32_t dim_m, std::uint32_t dim_n, std::uint32_t dim_k,
    float const *A, float const *B, float scale, float *C);

inline void launch_mace_linear_per_irrep(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_channels_out,
    std::uint32_t irrep_dim,
    std::uint32_t active_irrep_block_count,
    float const *weight_blocks,
    float const *input,
    float *output,
    cudaStream_t stream
) {
    if (n_atoms == 0) return;
    float const scale = 1.0f / sqrtf(static_cast<float>(n_channels_in));

    // Fast path: scalar-only (L=0) per-irrep linear is a plain row-major matmul
    // `[n_atoms, n_in] @ [n_in, n_out] -> [n_atoms, n_out]`.  The general
    // kernel launches `dim_block = irrep_dim` threads per CTA, which collapses
    // to a 1-thread CTA when irrep_dim=1 — n_atoms*n_channels_out CTAs each
    // with one active warp lane wastes ~97% of warp slots and burns scheduler
    // overhead.  The matmul kernel uses 16×16=256 threads per CTA and gets
    // good occupancy.
    if (irrep_dim == 1u && active_irrep_block_count == 1u) {
        constexpr std::uint32_t block_dim_x = 16, block_dim_y = 16;
        dim3 const mm_block(block_dim_x, block_dim_y, 1);
        dim3 const mm_grid((n_atoms + block_dim_x - 1u) / block_dim_x,
                           (n_channels_out + block_dim_y - 1u) / block_dim_y, 1);
        mace_matmul_rowmajor_kernel<<<mm_grid, mm_block, 0, stream>>>(
            n_atoms, n_channels_out, n_channels_in,
            input, weight_blocks, scale, output);
        return;
    }

    dim3 const block(irrep_dim, 1, 1);
    dim3 const grid(n_atoms, n_channels_out, 1);
    mace_linear_per_irrep_kernel<<<grid, block, 0, stream>>>(
        n_atoms, n_channels_in, n_channels_out, irrep_dim, active_irrep_block_count, scale,
        weight_blocks, input, output);
}

/**
 *  @brief Bare matmul `C = A * B` for the radial MLP, fp32.
 *
 *  Shapes: A = [dim_m, dim_k] row-major, B = [dim_k, dim_n] row-major
 *  (e3nn `_Layer.weight`), C = [dim_m, dim_n] row-major.  Computes
 *  `C[i, j] = Σ_l A[i, l] * B[l, j]`.
 *
 *  A hand-rolled square-tile kernel rather than cuBLAS: weights for the
 *  radial MLP are tiny (8→64, 64→64×3, 64→384) and live entirely in L1; the
 *  cuBLAS launch overhead dominates on these shapes.  Threading: one thread
 *  per (i, j_tile) accumulating across `dim_k`.
 */
__global__ void mace_matmul_rowmajor_kernel(
    std::uint32_t dim_m, std::uint32_t dim_n, std::uint32_t dim_k,
    float const *A,        // [dim_m, dim_k]
    float const *B,        // [dim_k, dim_n]
    float scale,           // multiplied into C; typically 1/sqrt(dim_k) for e3nn _Layer
    float *C               // [dim_m, dim_n]
) {
    std::uint32_t row_index = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t col_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_index >= dim_m || col_index >= dim_n) return;
    float accumulator = 0.0f;
    for (std::uint32_t inner_index = 0; inner_index < dim_k; ++inner_index) {
        accumulator += A[static_cast<std::uint64_t>(row_index) * dim_k + inner_index] * B[static_cast<std::uint64_t>(inner_index) * dim_n + col_index];
    }
    C[static_cast<std::uint64_t>(row_index) * dim_n + col_index] = accumulator * scale;
}

inline void launch_mace_matmul_rowmajor(
    std::uint32_t dim_m, std::uint32_t dim_n, std::uint32_t dim_k,
    float const *A, float const *B, float scale, float *C,
    cudaStream_t stream
) {
    if (dim_m == 0 || dim_n == 0) return;
    constexpr std::uint32_t block_dim_x = 16, block_dim_y = 16;
    dim3 const block(block_dim_x, block_dim_y, 1);
    dim3 const grid((dim_m + block_dim_x - 1u) / block_dim_x, (dim_n + block_dim_y - 1u) / block_dim_y, 1);
    mace_matmul_rowmajor_kernel<<<grid, block, 0, stream>>>(dim_m, dim_n, dim_k, A, B, scale, C);
}

/**
 *  @brief In-place `x = silu(x) * cst` for a flat fp32 buffer.
 *
 *  The constant for e3nn `normalize2mom(silu)` evaluated with seed=0 over
 *  1M N(0,1) samples (the same procedure e3nn uses) is ≈ 1.6788235567 — we
 *  hard-code this as `mace_silu_normalize_to_unit_second_moment_constant_k`.
 *  If that constant diverges from the live e3nn value (e.g. e3nn changes
 *  seed or sampling), the radial MLP output will scale uniformly off — the
 *  comparison test flags this.
 */
constexpr float mace_silu_normalize_to_unit_second_moment_constant_k = 1.6787818690495806f;

__global__ void mace_silu_normalize_to_unit_second_moment_kernel(std::uint32_t n, float const *src, float *dst) {
    std::uint32_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index >= n) return;
    float const x = src[element_index];
    float const s = 1.0f / (1.0f + __expf(-x));
    dst[element_index] = x * s * mace_silu_normalize_to_unit_second_moment_constant_k;
}

inline void launch_mace_silu_norm2mom(std::uint32_t n, float const *src, float *dst, cudaStream_t stream) {
    if (n == 0) return;
    constexpr std::uint32_t block = 256;
    std::uint32_t const grid = (n + block - 1u) / block;
    mace_silu_normalize_to_unit_second_moment_kernel<<<grid, block, 0, stream>>>(n, src, dst);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
