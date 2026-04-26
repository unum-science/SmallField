/**
 *  @file mace_bf16.cuh
 *  @brief Phase 2 helpers — bf16 conversion + cublasLt bf16 GEMM wrappers.
 *
 *  Once Phase 1 forward is correctness-locked, switching to bf16 is a
 *  surgical change:
 *
 *    1. At weight load time, downcast every tensor blob from fp32 → bf16
 *       (or, for the symmetric-contraction U-matrices, keep fp32 — those
 *       are dense low-rank tensors with high condition number and not
 *       worth quantising).
 *    2. Replace every `cublasSgemm` call in `mace_inference.cuh` with
 *       `cublasLtMatmul` configured for bf16 input/output and fp32
 *       accumulate.  Same M/N/K math.
 *    3. The two compute-bound non-cublas kernels (tensor product and
 *       symmetric contraction) get bf16 input loads and fp32 accumulators
 *       inside the kernel; outputs stored as bf16.
 *    4. The radial/sph-harm/scatter kernels stay fp32 — they are tiny.
 *
 *  Speedup target: 1.6–2× on the cueq-equivalent portion (which we
 *  measured at ~25 % of pass time on H100 SXM5).  End-to-end, plus the
 *  Phase-4 fusion and graph capture, the target is 3–5× over the
 *  measured 0.169 ms/config of MACE-OFF-large.
 *
 *  This header is intentionally a *plan*: it defines the helper signatures
 *  but does not yet implement them.  Phase 2 is gated on Phase 1 closing
 *  the energy-error gap (see `MACE_INFERENCE.md`).
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "smallfield/types/mace_types.hpp"

namespace smallfield::cuda {

/**
 *  @brief Downcast an fp32 tensor to bf16 in-place on device.
 *
 *  Uses CUDA's `__float2bfloat16` intrinsic; truncates rather than
 *  rounds-to-nearest-even (the difference is < 1 ULP and irrelevant for
 *  inference).  After the cast, the bf16 view of the buffer is valid for
 *  the first `n` elements.
 */
__global__ void mace_fp32_to_bf16_kernel(std::uint32_t n, float const *src, __nv_bfloat16 *dst) {
    std::uint32_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index >= n) return;
    dst[element_index] = __float2bfloat16(src[element_index]);
}

inline void launch_mace_fp32_to_bf16(std::uint32_t n, float const *src, __nv_bfloat16 *dst, cudaStream_t stream) {
    if (n == 0) return;
    constexpr std::uint32_t block = 256;
    std::uint32_t const grid = (n + block - 1u) / block;
    mace_fp32_to_bf16_kernel<<<grid, block, 0, stream>>>(n, src, dst);
}

/**
 *  @brief bf16 cublasLt linear apply (Phase 2 stub).
 *
 *  Phase 2 implementation:
 *    - Build a `cublasLtMatmulDesc_t` once at engine construction.
 *    - Configure `CUBLASLT_MATMUL_DESC_EPILOGUE` to fold any per-block
 *      bias / activation in the same kernel where applicable.
 *    - Reuse the descriptor across all GEMMs of the same shape (the radial
 *      MLP layers 0..3 have 4 distinct shapes total).
 *
 *  @todo Phase-2 — fill in once forward fp32 reference is bit-exact.
 */
inline void mace_linear_per_irrep_apply_bf16_stub(cudaStream_t stream) {
    (void)stream;
    // TODO(phase2): cublasLtMatmul setup as described in the file header.
}

} // namespace smallfield::cuda

#endif // __CUDACC__
