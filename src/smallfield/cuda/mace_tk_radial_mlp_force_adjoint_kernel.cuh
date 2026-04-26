/**
 *  @file mace_tk_radial_mlp_force_adjoint_kernel.cuh
 *  @brief ThunderKittens-WGMMA per-layer radial MLP force adjoint (E1).
 *
 *  Adjoint of one fully-connected layer of the radial MLP with optional
 *  silu·n2m epilogue.  Replaces `mace_radial_mlp_layer_force_adjoint_kernel`
 *  for the (n_in, n_out) shapes that appear in MACE-OFF-small/medium:
 *
 *    Layer 1, 2 backward:  n_in = 64, n_out = 64,   with silu epilogue
 *    Layer 3   backward:   n_in = 64, n_out = 384,  no  silu epilogue
 *
 *  Layer 0 backward (n_in = 8, accumulating, with silu) stays on the fp32
 *  reference path — its shape doesn't fit the WGMMA N=64 tile well and the
 *  per-replay cost is small (~2% of radial backward).
 *
 *  Forward:
 *      pre = scale * (in @ W)               in : [N, n_in]   W : [n_in, n_out]
 *      out = silu_n2m(pre)                  (if epilogue)
 *
 *  Backward:
 *      grad_pre = grad_out * silu_n2m'(pre_act)   (or = grad_out if no epilogue)
 *      grad_in  = scale * grad_pre @ W^T          [N, n_in]
 *
 *  In WGMMA terms (per CTA, M = 64 edges):
 *      A = `[64, n_out]` bf16        (folded silu' x grad_out, or pure grad_out)
 *      B = `[n_in, n_out]` bf16      (weight, same layout as forward)
 *      D = `[64, n_in]` fp32         (grad_in, written non-atomically)
 *      mm_ABt(D, A, B)
 *
 *  The "fold silu' into A" step happens during the bf16 stage of grad_out:
 *  grad_out is fp32 in global, pre-act is bf16 in global (saved by Phase B6),
 *  per-element we compute `du = grad_out * silu_n2m_deriv(pre_act)` in fp32
 *  registers and write the bf16 result to shared.  This eliminates the three
 *  `bf16_to_fp32` cast launches that previously fed the fp32 reference path.
 *
 *  Output is fp32 (overwritten, not accumulated) to match the public ABI for
 *  layers 1, 2, 3 — the chained next-layer backward consumes it directly.
 *
 *  Gated on `SMALLFIELD_USE_THUNDERKITTENS`; fp32 reference remains the
 *  fallback for the off case and for layer-0 backward.
 */

#pragma once

#include "smallfield/cuda/mace_tk_tile.cuh"

#if defined(SMALLFIELD_USE_THUNDERKITTENS) && defined(__CUDACC__)

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace smallfield::cuda {

namespace mace_tk_radial_mlp_force_adjoint_detail_ {

// Edge tile rows = WGMMA M.  Picked as 64 to match the radial-MLP forward
// kernel's tiling so a downstream fused 4-layer adjoint can reuse the same
// shared layout.
constexpr int edge_tile_rows_k         = 64;
constexpr int hidden_dim_k             = mace_tk::mace_radial_hidden_dim_k;   // 64
constexpr int output_dim_layer3_k      = mace_tk::mace_radial_output_dim_k;   // 384
constexpr int warpgroup_thread_count_k = kittens::WARPGROUP_THREADS;          // 128

// silu·n2m derivative — same closed form as the fp32 reference, evaluated
// inline so we can fold it into the bf16 stage of grad_out without a
// separate elementwise kernel.
__device__ __forceinline__ float silu_norm2mom_derivative_(float pre_act) {
    constexpr float N = mace_tk::silu_normalize2mom_epilogue_t::normalize_constant;
    float const sig = 1.0f / (1.0f + __expf(-pre_act));
    return N * sig * (1.0f + pre_act * (1.0f - sig));
}

/**
 *  @brief Stage `[edge_tile_rows, K]` bf16 grad_pre tile, optionally folding
 *         the silu·n2m derivative across the bf16 conversion.
 *
 *  - `apply_silu`: when true, multiplies grad_out by silu·n2m'(pre_act) in fp32
 *    registers before bf16 round.  `pre_act_bf16` must be non-null and point
 *    to the per-edge pre-activation slab saved by the forward.
 *  - `apply_silu == false`: pure cast of grad_out fp32 → bf16.  `pre_act_bf16`
 *    is unused.
 *
 *  Out-of-range edges (`edge_index >= n_edges`) are zeroed so the matmul
 *  produces zeros that the caller masks out at the global-store boundary.
 */
template <int K_dim, bool apply_silu>
__device__ __forceinline__ void load_grad_pre_tile_with_silu_fold_(
    kittens::st_bf<edge_tile_rows_k, K_dim> &dst_grad_pre_tile,
    float const *grad_out_global,                    // [n_edges, K]  fp32
    __nv_bfloat16 const *pre_act_bf16_global,        // [n_edges, K]  bf16, only read when apply_silu
    std::uint32_t n_edges,
    std::uint32_t base_edge_index
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = edge_tile_rows_k * K_dim;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / K_dim;
        int const col_in_tile = linear_index % K_dim;
        std::uint32_t const edge_index = base_edge_index + static_cast<std::uint32_t>(row_in_tile);
        float value = 0.0f;
        if (edge_index < n_edges) {
            std::uint64_t const offset = static_cast<std::uint64_t>(edge_index) * K_dim + col_in_tile;
            value = grad_out_global[offset];
            if constexpr (apply_silu) {
                float const pre = __bfloat162float(pre_act_bf16_global[offset]);
                value *= silu_norm2mom_derivative_(pre);
            }
        }
        dst_grad_pre_tile[{row_in_tile, col_in_tile}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Stage the layer weight `[n_in, n_out]` from fp32 global into bf16
 *         shared `[N_dim = n_in, K_dim = n_out]`.  Layout is identical to
 *         the forward kernel's so the same weight blob is reused.
 */
template <int N_dim, int K_dim>
__device__ __forceinline__ void load_weight_tile_(
    kittens::st_bf<N_dim, K_dim> &dst_weight_tile,
    float const *weight_global
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = N_dim * K_dim;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / K_dim;
        int const col_in_tile = linear_index % K_dim;
        float const value = weight_global[row_in_tile * K_dim + col_in_tile];
        dst_weight_tile[{row_in_tile, col_in_tile}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Apply per-element scale `1/sqrt(n_in)` to the fp32 register accumulator.
 */
template <int N_dim>
__device__ __forceinline__ void scale_accumulator_in_place_(
    kittens::rt_fl<16, N_dim> &accumulator,
    float scale
) {
    kittens::warpgroup::apply(
        accumulator,
        accumulator,
        [scale] __device__ (int /*row*/, int /*col*/, float pre) { return pre * scale; });
}

/**
 *  @brief Write the `[edge_tile_rows, N_dim]` accumulator to global fp32
 *         `grad_in[n_edges, N_dim]`, masking edges past `n_edges`.
 *
 *  TK's `warpgroup::store(gl, rt, coord)` would write past array end on the
 *  trailing partial tile, so we stage to a fp32 shared tile then iterate
 *  manually with an `edge_index < n_edges` guard.
 */
template <int N_dim>
__device__ __forceinline__ void store_grad_in_with_guard_(
    float *grad_in_global,
    kittens::st_fl<edge_tile_rows_k, N_dim> &staging_shared_tile,
    kittens::rt_fl<16, N_dim> const &accumulator,
    std::uint32_t n_edges,
    std::uint32_t base_edge_index
) {
    kittens::warpgroup::store(staging_shared_tile, accumulator);
    kittens::warpgroup::sync(0);

    int const thread_id = threadIdx.x;
    constexpr int tile_elements = edge_tile_rows_k * N_dim;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const row_in_tile = linear_index / N_dim;
        int const col_in_tile = linear_index % N_dim;
        std::uint32_t const edge_index = base_edge_index + static_cast<std::uint32_t>(row_in_tile);
        if (edge_index >= n_edges) continue;
        float const value = staging_shared_tile[{row_in_tile, col_in_tile}];
        grad_in_global[static_cast<std::uint64_t>(edge_index) * N_dim + col_in_tile] = value;
    }
}

/**
 *  @brief Per-CTA radial-MLP layer backward: 64 edges per CTA, one WGMMA.
 *
 *  Grid: `dim3(ceil_div(n_edges, 64), 1, 1)`.
 *  Block: 128 threads (one warpgroup).
 *
 *  Per-CTA flow:
 *    1. Stage `[64, K]` bf16 grad_pre tile, folding silu·n2m'(pre_act) on the way in.
 *    2. Stage `[N, K]` bf16 weight tile from fp32 global.
 *    3. WGMMA mm_ABt: D[edge, c_in] = Σ_{c_out} grad_pre[edge, c_out] * W[c_in, c_out].
 *    4. Scale by `1/sqrt(n_in)` (e3nn `_Layer.path_weight`).
 *    5. Stage to fp32 shared tile, write to global `[n_edges, N]` with mask.
 */
template <int N_dim, int K_dim, bool apply_silu>
__global__ void mace_tk_radial_mlp_layer_force_adjoint_kernel(
    std::uint32_t n_edges,
    float scale,
    float const *__restrict__ grad_out_fp32,         // [n_edges, K_dim]
    __nv_bfloat16 const *__restrict__ pre_act_bf16,  // [n_edges, K_dim] (only read when apply_silu)
    float const *__restrict__ weight_fp32,           // [N_dim, K_dim]
    float *__restrict__ grad_in_fp32                 // [n_edges, N_dim]
) {
    extern __shared__ kittens::alignment_dummy raw_shared_memory[];
    kittens::shared_allocator shared_alloc(reinterpret_cast<int *>(&raw_shared_memory[0]));

    auto &shared_grad_pre_tile = shared_alloc.allocate<kittens::st_bf<edge_tile_rows_k, K_dim>>();
    auto &shared_weight_tile   = shared_alloc.allocate<kittens::st_bf<N_dim, K_dim>>();
    auto &shared_staging_tile  = shared_alloc.allocate<kittens::st_fl<edge_tile_rows_k, N_dim>>();

    std::uint32_t const base_edge_index = blockIdx.x * static_cast<std::uint32_t>(edge_tile_rows_k);
    if (base_edge_index >= n_edges) return;

    load_grad_pre_tile_with_silu_fold_<K_dim, apply_silu>(
        shared_grad_pre_tile, grad_out_fp32, pre_act_bf16, n_edges, base_edge_index);
    load_weight_tile_<N_dim, K_dim>(shared_weight_tile, weight_fp32);
    kittens::warpgroup::sync(0);

    kittens::rt_fl<16, N_dim> accumulator;
    kittens::warp::zero(accumulator);
    kittens::warpgroup::mm_ABt(accumulator, shared_grad_pre_tile, shared_weight_tile);
    kittens::warpgroup::mma_async_wait();

    scale_accumulator_in_place_<N_dim>(accumulator, scale);

    store_grad_in_with_guard_<N_dim>(
        grad_in_fp32, shared_staging_tile, accumulator, n_edges, base_edge_index);
}

} // namespace mace_tk_radial_mlp_force_adjoint_detail_

/**
 *  @brief Launch the TK-WGMMA radial-MLP layer backward.
 *
 *  Specialized for two MACE-OFF shapes:
 *    - (n_in=64, n_out=64,  apply_silu=true)  : layers 1, 2 backward
 *    - (n_in=64, n_out=384, apply_silu=false) : layer 3 backward
 *
 *  Caller must ensure `pre_act_bf16` is non-null when `apply_silu = true`.
 *  Output is overwritten (not accumulated) — caller is responsible for any
 *  composition with parallel adjoint paths.
 *
 *  Returns false if the (n_in, n_out, apply_silu) tuple is not handled by
 *  this TK kernel; caller should fall back to `launch_mace_radial_mlp_layer_force_adjoint`
 *  in `mace_force_adjoints.cuh`.
 */
inline bool launch_mace_tk_radial_mlp_layer_force_adjoint(
    std::uint32_t n_edges,
    std::uint32_t n_in,
    std::uint32_t n_out,
    float scale,
    bool apply_silu_n2m_inverse,
    float const *grad_out_fp32,
    __nv_bfloat16 const *pre_act_bf16,
    float const *weight_fp32,
    float *grad_in_fp32,
    cudaStream_t stream
) {
    using namespace mace_tk_radial_mlp_force_adjoint_detail_;
    if (n_edges == 0) return true;

    auto launch = [&](auto kernel_ptr, std::size_t shared_bytes) {
        // Opt-in to >48 KB shared memory for the K=384 path.  ~63 KB per CTA
        // (48 KB grad_pre + 48 KB weight + 16 KB staging) so we need to
        // request the dynamic-allocation cap.
        if (shared_bytes > 48u * 1024u) {
            cudaFuncSetAttribute(kernel_ptr,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(shared_bytes));
        }
        std::uint32_t const grid_dim = (n_edges + edge_tile_rows_k - 1u) / edge_tile_rows_k;
        dim3 const grid(grid_dim, 1, 1);
        dim3 const block(warpgroup_thread_count_k, 1, 1);
        kernel_ptr<<<grid, block, shared_bytes, stream>>>(
            n_edges, scale, grad_out_fp32, pre_act_bf16, weight_fp32, grad_in_fp32);
    };

    // Compute shared-memory size from the template tile shapes.
    auto shared_bytes_for = [](int N_dim, int K_dim) {
        return static_cast<std::size_t>(edge_tile_rows_k * K_dim * sizeof(__nv_bfloat16))   // grad_pre
             + static_cast<std::size_t>(N_dim * K_dim * sizeof(__nv_bfloat16))              // weight
             + static_cast<std::size_t>(edge_tile_rows_k * N_dim * sizeof(float))           // staging
             + 1024u;                                                                       // alignment slack
    };

    if (n_in == 64u && n_out == 64u && apply_silu_n2m_inverse) {
        launch(&mace_tk_radial_mlp_layer_force_adjoint_kernel<64, 64, true>,
               shared_bytes_for(64, 64));
        return true;
    }
    if (n_in == 64u && n_out == 384u && !apply_silu_n2m_inverse) {
        // MACE-OFF-small: n_channels = 96 → tp_dim = 4 * 96 = 384.
        launch(&mace_tk_radial_mlp_layer_force_adjoint_kernel<64, 384, false>,
               shared_bytes_for(64, 384));
        return true;
    }
    if (n_in == 64u && n_out == 512u && !apply_silu_n2m_inverse) {
        // MACE-OFF-medium: n_channels = 128 → tp_dim = 4 * 128 = 512.
        launch(&mace_tk_radial_mlp_layer_force_adjoint_kernel<64, 512, false>,
               shared_bytes_for(64, 512));
        return true;
    }
    if (n_in == 64u && n_out == 768u && !apply_silu_n2m_inverse) {
        // MACE-OFF (hypothetical 192-channel variant): tp_dim = 4 * 192 = 768.
        launch(&mace_tk_radial_mlp_layer_force_adjoint_kernel<64, 768, false>,
               shared_bytes_for(64, 768));
        return true;
    }
    if (n_in == 64u && n_out == 896u && !apply_silu_n2m_inverse) {
        // MACE-OFF23-large: n_channels = 224 → tp_dim = 4 * 224 = 896.
        launch(&mace_tk_radial_mlp_layer_force_adjoint_kernel<64, 896, false>,
               shared_bytes_for(64, 896));
        return true;
    }
    return false;
}

} // namespace smallfield::cuda

#endif // SMALLFIELD_USE_THUNDERKITTENS && __CUDACC__
