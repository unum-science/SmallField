/**
 *  @file mace_tk_symmetric_contraction_force_adjoint_kernel.cuh
 *  @brief ThunderKittens-flavoured symmetric contraction force adjoint (D4) for MACE-OFF.
 *
 *  Adjoint of `mace_symmetric_contraction_kernel` /
 *  `mace_tk_symmetric_contraction_kernel`.  For the order-3 ACE expansion
 *
 *      B[a, c] = Σ_{i, p}        w_1[Z, p, c]   * U_1[i, p]       * X[a, c, i]
 *              + Σ_{i, j, p}     w_2[Z, p, c]   * U_2[i, j, p]    * X[a, c, i] * X[a, c, j]
 *              + Σ_{i, j, k, p}  w_max[Z, p, c] * U_3[i, j, k, p] * X[a, c, i] * X[a, c, j] * X[a, c, k]
 *
 *  the reverse mode emits three additive writes per (i, j, k) inner iteration
 *  for the order-3 term — the MACE-OFF23 U_3 tensor is NOT fully symmetric in
 *  the (i, j, k) index triple (max asymmetry ≈ 0.577 in MACE-OFF-small
 *  weights), so the "pin first index, multiply by ν" shortcut would be
 *  incorrect.  Mirroring the fp32 reference's D4 implementation:
 *
 *      dx[i] += u_w * X[j] * X[k]
 *      dx[j] += u_w * X[i] * X[k]
 *      dx[k] += u_w * X[i] * X[j]
 *
 *  where `u_w = W3_eff[Z, i, j, k, c]` is the precomputed slab the forward
 *  kernel already consumes (path_index folded against `weights_max`):
 *
 *      W3_eff[Z, i, j, k, c] = Σ_p U_3[i, j, k, p] * w_max[Z, p, c]
 *
 *  Reusing the same `W3_eff` buffer is load-bearing — the fp32 reference's
 *  per-launch path-loop over 23 paths × 16³ cells × n_channels would otherwise
 *  dominate the adjoint cost; the precomputed slab folds those 23 paths away
 *  at the cost of one host-side reduction at engine init (which the forward
 *  already paid).  No new precompute is needed at the adjoint call site —
 *  the caller passes the same buffer the forward consumed.
 *
 *  Order-2 and order-1 are computed inline against the original `U_*` and
 *  `weights_*` tensors — the same low-flops paths the forward kernel handles
 *  inline; folding their `path_index` into a precomputed slab would only save
 *  ~4 × 16² FFMA per (atom, channel), which is invisible relative to the
 *  order-3 cost.
 *
 *  Output stays fp32 — order-3 accumulator precision is load-bearing per the
 *  inference work plan's risk register (bf16 here costs ≥ 4 meV / config on
 *  the goldens, against a ≤ 1 meV diff target).  The output is `+=`-
 *  accumulated, matching the fp32 reference's contract: the caller-supplied
 *  `grad_node_features_full_fp32` already holds gradient contributions from
 *  parallel adjoint paths (e.g. the per-irrep linear), and we add on top.
 *
 *  Threading: one CTA per atom, one warp (32 threads) per CTA, each thread
 *  owning a contiguous tile of channels.  Per-atom slab of `W3_eff` is too
 *  big for shared memory (`16³ × n_channels × 4 B ≈ 1.5 MB` for
 *  n_channels = 96), so we read it cell-by-cell from HBM and let the L2 +
 *  texture cache absorb the reuse across atoms of the same element.  The
 *  per-atom `X[atom, :, :]` slab and the per-atom local gradient row both
 *  live in shared memory.
 *
 *  Compile-guarded on `SMALLFIELD_USE_THUNDERKITTENS`; when the macro is off
 *  the file is empty and the fp32 reference path remains the only D4 adjoint.
 */

#pragma once

#include "smallfield/cuda/mace_tk_tile.cuh"

#if defined(SMALLFIELD_USE_THUNDERKITTENS) && defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

namespace smallfield::cuda {

namespace mace_tk_symmetric_contraction_force_adjoint_detail_ {

// Compile-time shape constants, matching the forward TK kernel.
constexpr int irrep_dim_k             = mace_tk::mace_irrep_dim_total_k; // 16
constexpr int path_count_order_2_k    = 4;
constexpr int path_count_order_1_k    = 1;
constexpr int warp_thread_count_k     = 32;

/**
 *  @brief Stage the per-atom feature slab `X[atom, :, :]` into shared memory.
 *
 *  Identical layout to the forward kernel's helper: `n_channels × irrep_dim`
 *  fp32 values with the irrep axis as the inner stride-1 dimension.  We load
 *  X here so the (i, j, k) triple loop (and the order-1 / order-2 paths)
 *  read it from shared, not from HBM, and so the order-3 (i, j, k) loop can
 *  read X[i], X[j], X[k] without re-issuing global loads per inner iteration.
 */
__device__ __forceinline__ void load_atom_feature_slab_(
    float *shared_features_atom,
    float const *node_features_full_fp32,
    std::uint32_t atom_index,
    std::uint32_t n_channels
) {
    int const thread_id = threadIdx.x;
    int const tile_elements = static_cast<int>(n_channels) * irrep_dim_k;
    float const *atom_slab = node_features_full_fp32
                             + static_cast<std::uint64_t>(atom_index) * static_cast<std::uint64_t>(tile_elements);
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warp_thread_count_k) {
        shared_features_atom[linear_index] = atom_slab[linear_index];
    }
}

/**
 *  @brief Per-atom symmetric contraction adjoint kernel using a precomputed `W3_eff`.
 *
 *  Grid: `dim3(n_atoms, 1, 1)` — one CTA per atom.
 *  Block: 32 threads (one warp).  Each thread owns a strided set of channels
 *  (`channel_index = thread_id, thread_id + 32, thread_id + 64, ...`).
 *
 *  Per-CTA flow:
 *    1. Stage `X[atom, :, :]` into shared.
 *    2. For each (channel-tile-of-1) the thread owns:
 *       a. Zero a local 16-wide register accumulator `dx_local`.
 *       b. Order-1 contribution: `dx[i] += w_1[Z, p, c] * U_1[i, p]`.
 *       c. Order-2 contribution: emit both (i, j) writes inside the inner
 *          loop because we don't rely on U_2 symmetry.
 *       d. Order-3 contribution: emit all three (i, j, k) permutation-conjugate
 *          writes inside the inner loop because U_3 is not symmetric.
 *       e. Multiply each `dx_local[i]` by `upstream_grad_output[a, c]` and
 *          `+=`-accumulate into `grad_node_features_full_fp32[a, c, :]`.
 */
__global__ void mace_tk_symmetric_contraction_force_adjoint_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t n_elements,
    std::int32_t const *__restrict__ atom_element,
    float const *__restrict__ node_features_full_fp32,
    float const *__restrict__ u_matrix_1_fp32,
    float const *__restrict__ u_matrix_2_fp32,
    float const *__restrict__ weights_2_fp32,
    float const *__restrict__ weights_1_fp32,
    float const *__restrict__ precomputed_w3_effective_fp32,    // unused under Transform #1; kept for ABI compat
    float const *__restrict__ precomputed_w3_partsym_fp32,      // Transform #1: order-3 partsym-symmetric tensor
    float const *__restrict__ upstream_grad_output_fp32,
    float *__restrict__ grad_node_features_full_fp32
) {
    (void)precomputed_w3_effective_fp32;
    extern __shared__ float raw_shared_memory_for_features[];
    float *shared_features_atom = raw_shared_memory_for_features;

    std::uint32_t const atom_index = blockIdx.x;
    if (atom_index >= n_atoms) return;

    int const element_index = atom_element[atom_index];

    // Stage `X[atom, :, :]` into shared memory once.
    load_atom_feature_slab_(shared_features_atom, node_features_full_fp32, atom_index, n_channels);
    __syncwarp();

    int const thread_id = threadIdx.x;
    std::uint64_t const w3_per_element_stride = static_cast<std::uint64_t>(irrep_dim_k)
                                                * static_cast<std::uint64_t>(irrep_dim_k)
                                                * static_cast<std::uint64_t>(irrep_dim_k)
                                                * static_cast<std::uint64_t>(n_channels);
    // Transform #1: read from W3_partsym_sym (canonical α ≤ β cells) instead
    // of W3_eff.  The partsym tensor encodes the closed-form quadratic adjoint
    // already symmetrised over the (α, β) input axes.
    float const *w3_partsym_for_element = precomputed_w3_partsym_fp32
                                          + static_cast<std::uint64_t>(element_index) * w3_per_element_stride;

    // Walk the channel axis in strided chunks of `warp_thread_count_k = 32`.
    // Each thread owns one channel per chunk and keeps the 16-wide gradient
    // accumulator in registers.
    for (std::uint32_t channel_base = 0u;
         channel_base < n_channels;
         channel_base += static_cast<std::uint32_t>(warp_thread_count_k)) {

        std::uint32_t const channel_index = channel_base + static_cast<std::uint32_t>(thread_id);
        if (channel_index >= n_channels) break;

        // Pointer into shared memory at `X[atom, channel_index, :]`.
        float const *atom_irrep_row = shared_features_atom + static_cast<std::uint64_t>(channel_index) * irrep_dim_k;

        // 16-wide local gradient accumulator for this (atom, channel).
        float dx_local[irrep_dim_k] = {0.0f};

        // ---- Order 1 ----
        // Forward: B += w_1[Z, p, c] * U_1[i, p] * X[i].
        // Reverse: dx[i] += w_1[Z, p, c] * U_1[i, p].
        #pragma unroll
        for (int path_index = 0; path_index < path_count_order_1_k; ++path_index) {
            float const weighted_path_coefficient = weights_1_fp32[
                (static_cast<std::uint64_t>(element_index) * path_count_order_1_k + path_index) * n_channels
                + channel_index];
            #pragma unroll
            for (int irrep_index_i = 0; irrep_index_i < irrep_dim_k; ++irrep_index_i) {
                float const u = u_matrix_1_fp32[
                    static_cast<std::uint64_t>(irrep_index_i) * path_count_order_1_k + path_index];
                dx_local[irrep_index_i] += weighted_path_coefficient * u;
            }
        }

        // ---- Order 2 ----
        // Forward: B += w_2[Z, p, c] * U_2[i, j, p] * X[i] * X[j].
        // Reverse: dx[i] += u_w * X[j];  dx[j] += u_w * X[i].
        // We emit both writes inside the loop so we are correct without
        // assuming any U_2 symmetry — even though U_2 is in fact symmetric
        // in MACE-OFF23 the cost difference is negligible against the
        // order-3 path.
        #pragma unroll
        for (int path_index = 0; path_index < path_count_order_2_k; ++path_index) {
            float const weighted_path_coefficient = weights_2_fp32[
                (static_cast<std::uint64_t>(element_index) * path_count_order_2_k + path_index) * n_channels
                + channel_index];
            #pragma unroll
            for (int irrep_index_i = 0; irrep_index_i < irrep_dim_k; ++irrep_index_i) {
                float const x_i = atom_irrep_row[irrep_index_i];
                #pragma unroll
                for (int irrep_index_j = 0; irrep_index_j < irrep_dim_k; ++irrep_index_j) {
                    float const u = u_matrix_2_fp32[
                        ((static_cast<std::uint64_t>(irrep_index_i) * irrep_dim_k) + irrep_index_j)
                            * path_count_order_2_k + path_index];
                    float const u_w = weighted_path_coefficient * u;
                    float const x_j = atom_irrep_row[irrep_index_j];
                    dx_local[irrep_index_i] += u_w * x_j;
                    dx_local[irrep_index_j] += u_w * x_i;
                }
            }
        }

        // ---- Order 3 (Transform #1: partsym + X·X cache) ----
        // Closed-form derivative: dx[m] = Σ_{α,β} W3_partsym[m, α, β, c] · X[α] · X[β]
        // with the α-β symmetry folded into W3_partsym_sym at engine-init time:
        //     dx[m] = Σ_{α=β}        W3_partsym_sym[m,α,α,c] · X[α]²
        //           + Σ_{α<β}        W3_partsym_sym[m,α,β,c] · X[α] · X[β]
        // Total ops per (a, c): 136 muls (X·X cache) + 16 × 136 = 2 176 FMAs
        // vs original 16³ × 3 + 16³ × 2 = 20 480 ops — ~5.6× FMA reduction
        // and 16 INDEPENDENT length-136 FMA chains (one per dx[m]) instead of
        // a single ~4096-deep chain on three accumulators.
        //
        // Precision: exact identity (folds permutations of W3, exploits X·X
        // symmetry); no precision loss vs the original cubic-write form.
        //
        // Loop ordering: (α, β) outer, m inner unrolled.  This gives 16
        // INDEPENDENT FMAs per (α, β) iteration (one per dx_local[m]) — the
        // compiler can keep all 16 in flight, exposing massive ILP and shorter
        // per-register dependency chains than the old single-accumulator form.
        for (int alpha = 0; alpha < irrep_dim_k; ++alpha) {
            float const x_alpha = atom_irrep_row[alpha];
            #pragma unroll
            for (int beta = alpha; beta < irrep_dim_k; ++beta) {
                float const x_alpha_beta = x_alpha * atom_irrep_row[beta];
                std::uint64_t const w_base =
                    (static_cast<std::uint64_t>(alpha) * irrep_dim_k + beta)
                    * static_cast<std::uint64_t>(n_channels) + channel_index;
                std::uint64_t const w_per_m_stride = static_cast<std::uint64_t>(irrep_dim_k)
                                                     * static_cast<std::uint64_t>(irrep_dim_k)
                                                     * static_cast<std::uint64_t>(n_channels);
                #pragma unroll
                for (int m = 0; m < irrep_dim_k; ++m) {
                    float const w = w3_partsym_for_element[
                        static_cast<std::uint64_t>(m) * w_per_m_stride + w_base];
                    dx_local[m] += w * x_alpha_beta;
                }
            }
        }

        // Multiply by upstream and `+=`-accumulate into the global grad buffer.
        float const upstream = upstream_grad_output_fp32[
            static_cast<std::uint64_t>(atom_index) * n_channels + channel_index];
        float *grad_atom_irrep_row = grad_node_features_full_fp32
            + (static_cast<std::uint64_t>(atom_index) * n_channels + channel_index) * irrep_dim_k;
        #pragma unroll
        for (int irrep_index_i = 0; irrep_index_i < irrep_dim_k; ++irrep_index_i) {
            grad_atom_irrep_row[irrep_index_i] += upstream * dx_local[irrep_index_i];
        }
    }
}

} // namespace mace_tk_symmetric_contraction_force_adjoint_detail_

/**
 *  @brief Launch the TK-flavoured symmetric contraction force adjoint (D4).
 *
 *  Reuses the same `precomputed_w3_effective_fp32` buffer the forward kernel
 *  consumes — no new precompute is needed at the adjoint call site.  Order-2
 *  and order-1 read the original `u_matrix_*` and `weights_*` tensors inline.
 *
 *  Inputs:
 *    - `node_features_full_fp32` is `[n_atoms, n_channels, irrep_dim_k=16]`,
 *      the same X buffer the forward consumed.  We re-stage it from HBM into
 *      shared memory inside the kernel; no caller-side caching needed.
 *    - `precomputed_w3_effective_fp32` is `[n_elements, 16, 16, 16, n_channels]`
 *      (channel axis stride-1), produced once per engine init by
 *      `mace_tk_precompute_w3_effective` in the forward kernel header.
 *    - `upstream_grad_output_fp32` is `[n_atoms, n_channels]` — the gradient
 *      flowing back from downstream onto the per-atom scalar invariants `B`.
 *
 *  Output:
 *    - `grad_node_features_full_fp32` is `[n_atoms, n_channels, 16]` and is
 *      `+=`-accumulated.  Caller is responsible for any pre-zeroing or
 *      composition with parallel adjoint paths.
 *
 *  Threading: `n_atoms` CTAs of one warp each; per-channel work is independent
 *  inside a thread.  Output stays fp32 — order-3 accumulator precision is
 *  load-bearing per the work plan's risk register.
 */
inline void launch_mace_tk_symmetric_contraction_force_adjoint(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t n_elements,
    std::int32_t const *atom_element,
    float const *node_features_full_fp32,
    float const *u_matrix_1_fp32,
    float const *u_matrix_2_fp32,
    float const *weights_2_fp32,
    float const *weights_1_fp32,
    float const *precomputed_w3_effective_fp32,           // unused under Transform #1; kept for ABI parity
    float const *precomputed_w3_partsym_fp32,             // Transform #1: order-3 partsym tensor
    float const *upstream_grad_output_fp32,
    float *grad_node_features_full_fp32,
    cudaStream_t stream
) {
    if (n_atoms == 0u) return;
    if (precomputed_w3_partsym_fp32 == nullptr) return;

    using namespace mace_tk_symmetric_contraction_force_adjoint_detail_;

    // Shared-memory budget: per-atom feature slab `n_channels × 16` fp32.  For
    // the largest MACE-OFF setting (n_channels = 192) this is 12 KB; under
    // the 48 KB default per-block limit so no opt-in attribute is required.
    std::size_t const shared_memory_bytes =
        static_cast<std::size_t>(n_channels) * static_cast<std::size_t>(irrep_dim_k) * sizeof(float);

    dim3 const grid(n_atoms, 1, 1);
    dim3 const block(static_cast<std::uint32_t>(warp_thread_count_k), 1, 1);

    mace_tk_symmetric_contraction_force_adjoint_kernel<<<grid, block, shared_memory_bytes, stream>>>(
        n_atoms, n_channels, n_elements,
        atom_element, node_features_full_fp32,
        u_matrix_1_fp32, u_matrix_2_fp32,
        weights_2_fp32, weights_1_fp32,
        precomputed_w3_effective_fp32,
        precomputed_w3_partsym_fp32,
        upstream_grad_output_fp32,
        grad_node_features_full_fp32);
}

} // namespace smallfield::cuda

#endif // SMALLFIELD_USE_THUNDERKITTENS && __CUDACC__
