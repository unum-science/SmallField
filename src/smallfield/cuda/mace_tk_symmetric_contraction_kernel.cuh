/**
 *  @file mace_tk_symmetric_contraction_kernel.cuh
 *  @brief ThunderKittens-flavoured symmetric contraction (MACE `product` block).
 *
 *  The MACE-OFF symmetric contraction implements the order-3 ACE expansion
 *  (per `mace.modules.symmetric_contraction.Contraction`):
 *
 *      B[atom, c] = Σ_{i,j,k,p} U_3[i,j,k,p] * w_max[Z[atom], p, c]
 *                              * X[atom, c, i] * X[atom, c, j] * X[atom, c, k]
 *
 *                 + Σ_{i,j,p}   U_2[i,j,p]   * w_2[Z[atom], p, c]
 *                              * X[atom, c, i] * X[atom, c, j]
 *
 *                 + Σ_{i,p}     U_1[i,p]     * w_1[Z[atom], p, c]
 *                              * X[atom, c, i]
 *
 *  with `irrep_dim = 16`, `path_count_max = 23`, `path_count_order_2 = 4`,
 *  `path_count_order_1 = 1`, `n_elements = 10` for MACE-OFF23.
 *
 *  The order-3 path dominates the flops budget by ~50× (16³ × 23 paths per
 *  (atom, channel) ≈ 100 k FLOPs vs ~1 k for orders 1+2).  Two structural
 *  observations drive the kernel design:
 *
 *  1.  `U_3` is sparse and `path_count_max = 23` is small enough that the
 *      reduction over `path_index` against `weights_max[Z, path, c]` can be
 *      precomputed once per element / channel / (i, j, k) cell:
 *
 *          W3_eff[Z, i, j, k, c] = Σ_path U_3[i, j, k, path] * w_max[Z, path, c]
 *
 *      For MACE-OFF-small this slab is `10 × 16 × 16 × 16 × 96 × 4 B ≈ 1.5 MB`
 *      total, fits comfortably in HBM and the L2 cache, and folds away the
 *      `path_index` axis at the cost of one host-side reduction per engine
 *      construction.  The forward then reads each `(Z, i, j, k, c)` cell
 *      exactly once per atom of element `Z`.
 *
 *  2.  After the `path_index` collapse the order-3 contraction becomes a
 *      pointwise-in-`c` triple sum:
 *
 *          B3[atom, c] = Σ_{i, j, k} W3_eff[Z, i, j, k, c]
 *                                    * X[atom, c, i] * X[atom, c, j] * X[atom, c, k]
 *
 *      The reduction structure is per-channel — there is no shared `c` that
 *      a WGMMA could amortise, because the three `X` factors are all indexed
 *      by the same `c`.  WGMMA is the wrong abstraction here; we still gain
 *      from the TK shared-tile / synchronisation primitives and from the
 *      W3_eff precomputation, but the inner accumulator is FFMA in registers.
 *
 *  Order-2 and order-1 are computed inline against the original `U_*` and
 *  `weights_*` tensors — a fold against `path` for those would only save
 *  4 × (16² × n_channels) and 1 × (16 × n_channels) elements of work per
 *  atom, which is invisible relative to the order-3 cost.
 *
 *  Threading:
 *   - One CTA per atom (no atom-batching: each atom needs its own X slab and
 *     its own `Z`-indexed `W3_eff` slice).  The per-atom slab of W3_eff is
 *     `16³ × n_channels × 4 B ≈ 1.5 MB` for n_channels=96, which is too big
 *     to stage in shared memory; we read it from HBM cell-by-cell instead
 *     and let the L2 + texture cache absorb the reuse across atoms of the
 *     same element.
 *   - One warp (32 threads) per CTA, each thread owning a contiguous tile of
 *     channels.  A 4-warp warpgroup would oversubscribe the 96-channel
 *     small-model row; a single warp keeps the per-thread channel tile dense
 *     (3 channels per thread for n_channels = 96) while leaving headroom for
 *     a 32-channel-wide register staging of `X[atom, c, :]`.
 *   - The `X[atom, :, :]` slab (`n_channels × 16`) lives in shared memory and
 *     is shared across the (i, j, k) triple loop.  For n_channels = 96 that
 *     is 6 KB, fits trivially.
 *
 *  Output: `B[atom, c]` in fp32.  The order-3 accumulator is high-precision-
 *  sensitive (16³ × 23 = 94 k FLOPs per (atom, channel)) and the existing
 *  fp32 reference is the regression target, so we deliberately keep the
 *  accumulator and output in fp32 — bf16 here loses ≥ 4 meV / config on the
 *  goldens.
 *
 *  Compile-guarded on `SMALLFIELD_USE_THUNDERKITTENS`; when the macro is off
 *  the file is empty and the fp32 reference path remains the only code path.
 */

#pragma once

#include "smallfield/cuda/mace_tk_tile.cuh"

#if defined(SMALLFIELD_USE_THUNDERKITTENS) && defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

namespace smallfield::cuda {

namespace mace_tk_symmetric_contraction_detail_ {

// Compile-time shape constants for the MACE-OFF symmetric contraction.
constexpr int irrep_dim_k             = mace_tk::mace_irrep_dim_total_k; // 16
constexpr int path_count_max_k        = 23;
constexpr int path_count_order_2_k    = 4;
constexpr int path_count_order_1_k    = 1;
constexpr int warp_thread_count_k     = 32;

/**
 *  @brief Stage the per-atom feature slab `X[atom, :, :]` into shared.
 *
 *  Reads `n_channels × irrep_dim` fp32 values from
 *  `node_features_full_fp32 + atom_index * (n_channels * irrep_dim)` and
 *  writes them into a shared-memory buffer with the same layout
 *  (`shared[c, i] = X[atom, c, i]`).
 *
 *  The 16-element `i` axis is the inner stride-1 dimension to match the
 *  reference kernel's pointer arithmetic; this preserves coalesced loads on
 *  the 32 threads × 4-byte fp32 boundary because consecutive threads handle
 *  consecutive linear offsets and the slab is fp32.
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
 *  @brief Per-atom symmetric-contraction kernel using a precomputed `W3_eff`.
 *
 *  Grid: `dim3(n_atoms, 1, 1)` — one CTA per atom.
 *  Block: 32 threads (a single warp).  Each thread covers a strided set of
 *  channels (`channel_index = thread_id, thread_id + 32, ...`).
 *
 *  Per-CTA flow:
 *    1. Stage `X[atom, :, :]` into shared.
 *    2. Compute order-3 contribution against `W3_eff[Z, i, j, k, c]` with a
 *       fused (i, j, k) loop and per-channel fp32 register accumulators.
 *    3. Add order-2 contribution (path-ful form, 4 paths × 16² FFMA / channel).
 *    4. Add order-1 contribution (1 path × 16 FFMA / channel).
 *    5. Store the per-channel accumulator out to `output_node_scalar_invariants_fp32`.
 */
__global__ void mace_tk_symmetric_contraction_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t n_elements,
    std::int32_t const *__restrict__ atom_element,
    float const *__restrict__ node_features_full_fp32,
    float const *__restrict__ u_matrix_1_fp32,
    float const *__restrict__ u_matrix_2_fp32,
    float const *__restrict__ weights_2_fp32,
    float const *__restrict__ weights_1_fp32,
    float const *__restrict__ precomputed_w3_effective_fp32,
    float *__restrict__ output_node_scalar_invariants_fp32
) {
    extern __shared__ float raw_shared_memory_for_features[];
    float *shared_features_atom = raw_shared_memory_for_features;

    std::uint32_t const atom_index = blockIdx.x;
    if (atom_index >= n_atoms) return;

    int const element_index = atom_element[atom_index];

    // Stage the per-atom feature slab `X[atom, :, :]` into shared memory once.
    load_atom_feature_slab_(shared_features_atom, node_features_full_fp32, atom_index, n_channels);
    __syncwarp();

    int const thread_id = threadIdx.x;
    std::uint64_t const w3_per_element_stride = static_cast<std::uint64_t>(irrep_dim_k)
                                                * static_cast<std::uint64_t>(irrep_dim_k)
                                                * static_cast<std::uint64_t>(irrep_dim_k)
                                                * static_cast<std::uint64_t>(n_channels);
    float const *w3_for_element = precomputed_w3_effective_fp32
                                  + static_cast<std::uint64_t>(element_index) * w3_per_element_stride;

    // Walk the channel axis in strided chunks of `warp_thread_count_k = 32`.
    // Per-thread channel work is independent — we can hold the per-channel
    // accumulator in a single register and emit one global store at the end.
    for (std::uint32_t channel_base = 0u;
         channel_base < n_channels;
         channel_base += static_cast<std::uint32_t>(warp_thread_count_k)) {

        std::uint32_t const channel_index = channel_base + static_cast<std::uint32_t>(thread_id);
        if (channel_index >= n_channels) break;

        // Pointer into shared memory at `X[atom, channel_index, :]`.
        float const *atom_irrep_row = shared_features_atom + static_cast<std::uint64_t>(channel_index) * irrep_dim_k;

        float accumulator = 0.0f;

        // ---- Order 1 ----
        // Σ_{i, p} U_1[i, p] * w_1[Z, p, c] * X[i]
        for (std::uint32_t path_index = 0u;
             path_index < static_cast<std::uint32_t>(path_count_order_1_k);
             ++path_index) {
            float const weighted_path_coefficient = weights_1_fp32[
                (static_cast<std::uint64_t>(element_index) * path_count_order_1_k + path_index) * n_channels
                + channel_index];
            #pragma unroll
            for (int irrep_index_i = 0; irrep_index_i < irrep_dim_k; ++irrep_index_i) {
                float const u = u_matrix_1_fp32[
                    static_cast<std::uint64_t>(irrep_index_i) * path_count_order_1_k + path_index];
                accumulator += u * weighted_path_coefficient * atom_irrep_row[irrep_index_i];
            }
        }

        // ---- Order 2 ----
        // Σ_{i, j, p} U_2[i, j, p] * w_2[Z, p, c] * X[i] * X[j]
        // Hoist the inner-i sum for each (j, p), then multiply by X[j].
        for (std::uint32_t path_index = 0u;
             path_index < static_cast<std::uint32_t>(path_count_order_2_k);
             ++path_index) {
            float const weighted_path_coefficient = weights_2_fp32[
                (static_cast<std::uint64_t>(element_index) * path_count_order_2_k + path_index) * n_channels
                + channel_index];
            #pragma unroll
            for (int irrep_index_j = 0; irrep_index_j < irrep_dim_k; ++irrep_index_j) {
                float inner_sum_over_irrep_dim = 0.0f;
                #pragma unroll
                for (int irrep_index_i = 0; irrep_index_i < irrep_dim_k; ++irrep_index_i) {
                    float const u = u_matrix_2_fp32[
                        ((static_cast<std::uint64_t>(irrep_index_i) * irrep_dim_k) + irrep_index_j)
                            * path_count_order_2_k + path_index];
                    inner_sum_over_irrep_dim += u * atom_irrep_row[irrep_index_i];
                }
                accumulator += weighted_path_coefficient * inner_sum_over_irrep_dim * atom_irrep_row[irrep_index_j];
            }
        }

        // ---- Order 3 (path-folded via precomputed W3_eff) ----
        // Σ_{i, j, k} W3_eff[Z, i, j, k, c] * X[i] * X[j] * X[k]
        // Hoist X[i] across (j, k); inside the j-loop hoist X[j] across k.
        // i × 4 / j × 16 unroll (commit `e347639`) gives 4 independent FMA chains.
        //
        // NOTE: Transform #1 (X-commutativity → canonical triples) does NOT help
        // this forward kernel — it has only one fp32 accumulator, the dependency
        // chain on `accumulator` was already broken into 4 chains by the i unroll,
        // and the kernel runs at ~99% SM / ~7% FMA-pipe with the existing layout.
        // The backward kernel (16 dx_local accumulators, much longer chain) is
        // where Transform #1 lands a real gain.
        #pragma unroll 4
        for (int irrep_index_i = 0; irrep_index_i < irrep_dim_k; ++irrep_index_i) {
            float const x_i = atom_irrep_row[irrep_index_i];
            #pragma unroll 16
            for (int irrep_index_j = 0; irrep_index_j < irrep_dim_k; ++irrep_index_j) {
                float const x_j = atom_irrep_row[irrep_index_j];
                float const x_ij = x_i * x_j;
                float const *w3_at_ij_base = w3_for_element
                    + ((static_cast<std::uint64_t>(irrep_index_i) * irrep_dim_k + irrep_index_j) * irrep_dim_k)
                          * static_cast<std::uint64_t>(n_channels);
                #pragma unroll
                for (int irrep_index_k = 0; irrep_index_k < irrep_dim_k; ++irrep_index_k) {
                    float const x_k = atom_irrep_row[irrep_index_k];
                    float const w3_value = w3_at_ij_base[
                        static_cast<std::uint64_t>(irrep_index_k) * n_channels + channel_index];
                    accumulator += w3_value * x_ij * x_k;
                }
            }
        }

        output_node_scalar_invariants_fp32[
            static_cast<std::uint64_t>(atom_index) * n_channels + channel_index] = accumulator;
    }
}

/**
 *  @brief Transform #1 — symmetrise W3_eff by summing all 6 (i, j, k) permutations
 *         into the CANONICAL `(i ≤ j ≤ k)` representative cell.
 *
 *  Mathematical identity (X-commutativity):
 *      Σ_{i,j,k all} W3[i,j,k,c] · X[i] X[j] X[k]
 *    = Σ_{(i ≤ j ≤ k)} W3_sym_total[i,j,k,c] · X[i] X[j] X[k]
 *  where
 *      W3_sym_total[i,j,k,c] = Σ_{σ ∈ distinct perms of (i,j,k)} W3[σ(i,j,k), c]
 *
 *  Distinct-perm count is 1 (i=j=k), 3 (i=j<k or i<j=k), or 6 (i<j<k).
 *  Output is exactly equivalent to the original cubic sum but evaluated over
 *  C(n+2, 3) = 816 canonical triples instead of n³ = 4096 ordered triples
 *  (n = irrep_dim_k = 16) — a ~5× FMA-count reduction on the order-3 path.
 *
 *  Non-canonical entries of `w3_sym_total` (cells where NOT i ≤ j ≤ k) are
 *  left undefined and should never be read by the consumer kernel.
 *
 *  Grid: `dim3(n_elements, irrep_dim³, 1)` — one CTA per (Z, ordered-triple).
 *  Block: same as `mace_tk_precompute_w3_effective_kernel` so we can share
 *  the launch shape without further tuning.
 */
__global__ void mace_tk_precompute_w3_sym_total_kernel(
    std::uint32_t n_channels,
    float const *__restrict__ w3_effective_fp32,    // [n_elements, 16, 16, 16, n_channels]
    float *__restrict__ w3_sym_total_fp32           // [n_elements, 16, 16, 16, n_channels] canonical-only
) {
    int const element_index = blockIdx.x;
    int const linear_ijk    = blockIdx.y;

    int const i = linear_ijk / (irrep_dim_k * irrep_dim_k);
    int const j = (linear_ijk / irrep_dim_k) % irrep_dim_k;
    int const k = linear_ijk % irrep_dim_k;
    if (!(i <= j && j <= k)) return;  // only canonical cells

    std::uint64_t const stride_per_element = static_cast<std::uint64_t>(irrep_dim_k)
                                              * static_cast<std::uint64_t>(irrep_dim_k)
                                              * static_cast<std::uint64_t>(irrep_dim_k)
                                              * static_cast<std::uint64_t>(n_channels);
    float const *w3_for_element = w3_effective_fp32
        + static_cast<std::uint64_t>(element_index) * stride_per_element;

    auto offset = [&](int a, int b, int c) {
        return ((static_cast<std::uint64_t>(a) * irrep_dim_k + b) * irrep_dim_k + c)
               * static_cast<std::uint64_t>(n_channels);
    };

    // Permutation set depends on equality structure.
    int perm_count = (i == k) ? 1 : (i == j || j == k) ? 3 : 6;
    int perms[6][3];
    int n_perms;
    if (i == j && j == k) {
        perms[0][0] = i; perms[0][1] = j; perms[0][2] = k;
        n_perms = 1;
    } else if (i == j) {  // i = j < k
        perms[0][0] = i; perms[0][1] = j; perms[0][2] = k;
        perms[1][0] = i; perms[1][1] = k; perms[1][2] = j;
        perms[2][0] = k; perms[2][1] = i; perms[2][2] = j;
        n_perms = 3;
    } else if (j == k) {  // i < j = k
        perms[0][0] = i; perms[0][1] = j; perms[0][2] = k;
        perms[1][0] = j; perms[1][1] = i; perms[1][2] = k;
        perms[2][0] = j; perms[2][1] = k; perms[2][2] = i;
        n_perms = 3;
    } else {              // i < j < k, all distinct → 6 perms
        perms[0][0] = i; perms[0][1] = j; perms[0][2] = k;
        perms[1][0] = i; perms[1][1] = k; perms[1][2] = j;
        perms[2][0] = j; perms[2][1] = i; perms[2][2] = k;
        perms[3][0] = j; perms[3][1] = k; perms[3][2] = i;
        perms[4][0] = k; perms[4][1] = i; perms[4][2] = j;
        perms[5][0] = k; perms[5][1] = j; perms[5][2] = i;
        n_perms = 6;
    }
    (void)perm_count;

    float *w3_sym_out = w3_sym_total_fp32 + static_cast<std::uint64_t>(element_index) * stride_per_element
                                          + offset(i, j, k);
    for (std::uint32_t channel = static_cast<std::uint32_t>(threadIdx.x);
         channel < n_channels;
         channel += static_cast<std::uint32_t>(blockDim.x)) {
        float accumulator = 0.0f;
        #pragma unroll
        for (int p = 0; p < 6; ++p) {
            if (p >= n_perms) break;
            float const v = w3_for_element[offset(perms[p][0], perms[p][1], perms[p][2]) + channel];
            accumulator += v;
        }
        w3_sym_out[channel] = accumulator;
    }
}

/**
 *  @brief Transform #1 — precompute `W3_partsym_sym[Z, m, a, b, c]` for the
 *         BACKWARD's quadratic form.
 *
 *  Closed-form derivative of the order-3 forward:
 *      ∂B[a, c] / ∂X[m] = Σ_{α,β} (W3[m,α,β,c] + W3[α,m,β,c] + W3[α,β,m,c]) · X[α] · X[β]
 *                       = Σ_{α,β} W3_partsym[m, α, β, c] · X[α] · X[β]
 *
 *  Since X[α]·X[β] is symmetric in (α, β) and W3_partsym is not, we further
 *  fold:
 *      W3_partsym_sym[m, α, β, c] = W3_partsym[m, α, β, c] + W3_partsym[m, β, α, c]   (α < β)
 *                                 = W3_partsym[m, α, α, c]                            (α = β)
 *
 *  The backward then runs as 16 *independent* length-136 quadratic forms
 *  (one per m), with a shared 136-entry X·X pair cache.  Net: ~5.6× FMA-count
 *  reduction with no precision loss.
 *
 *  Output buffer has the same `[n_elements, 16, 16, 16, n_channels]` shape
 *  as W3_eff for layout simplicity; only the canonical `(α ≤ β)` cells of
 *  the inner two axes are populated.
 */
__global__ void mace_tk_precompute_w3_partsym_kernel(
    std::uint32_t n_channels,
    float const *__restrict__ w3_effective_fp32,
    float *__restrict__ w3_partsym_sym_fp32
) {
    int const element_index = blockIdx.x;
    int const linear_mab    = blockIdx.y;
    int const m = linear_mab / (irrep_dim_k * irrep_dim_k);
    int const a = (linear_mab / irrep_dim_k) % irrep_dim_k;
    int const b = linear_mab % irrep_dim_k;
    if (a > b) return;  // only canonical (α ≤ β) cells

    std::uint64_t const stride_per_element = static_cast<std::uint64_t>(irrep_dim_k)
                                              * static_cast<std::uint64_t>(irrep_dim_k)
                                              * static_cast<std::uint64_t>(irrep_dim_k)
                                              * static_cast<std::uint64_t>(n_channels);
    float const *w3_for_element = w3_effective_fp32
        + static_cast<std::uint64_t>(element_index) * stride_per_element;
    auto offset = [&](int x, int y, int z) {
        return ((static_cast<std::uint64_t>(x) * irrep_dim_k + y) * irrep_dim_k + z)
               * static_cast<std::uint64_t>(n_channels);
    };
    float *out = w3_partsym_sym_fp32 + static_cast<std::uint64_t>(element_index) * stride_per_element
                                     + offset(m, a, b);

    for (std::uint32_t channel = static_cast<std::uint32_t>(threadIdx.x);
         channel < n_channels;
         channel += static_cast<std::uint32_t>(blockDim.x)) {
        // 3 partsym permutations for (m, a, b):
        float partsym_ab = w3_for_element[offset(m, a, b) + channel]
                         + w3_for_element[offset(a, m, b) + channel]
                         + w3_for_element[offset(a, b, m) + channel];
        if (a == b) {
            out[channel] = partsym_ab;
        } else {
            // (α, β)-symmetry fold: also add the partsym of the (m, b, a) swap.
            float partsym_ba = w3_for_element[offset(m, b, a) + channel]
                             + w3_for_element[offset(b, m, a) + channel]
                             + w3_for_element[offset(b, a, m) + channel];
            out[channel] = partsym_ab + partsym_ba;
        }
    }
}

/**
 *  @brief Helper kernel — fold `weights_max` into `U_3` once per engine init.
 *
 *  Computes
 *      W3_eff[Z, i, j, k, c] = Σ_{path} U_3[i, j, k, path] * w_max[Z, path, c]
 *
 *  Grid: `dim3(n_elements, irrep_dim³)` — one CTA per (Z, i, j, k) cell.
 *  Block: `n_channels` threads (or strided if n_channels > 1024).  Each thread
 *  owns one channel and reduces 23 paths sequentially.
 *
 *  The path tensor `U_3` is shared across all `Z`; this kernel is launched
 *  once per engine init so the simple-and-correct form here trumps an MMA
 *  fusion.
 */
__global__ void mace_tk_precompute_w3_effective_kernel(
    std::uint32_t n_channels,
    std::uint32_t path_count_max,
    float const *__restrict__ u_matrix_3_fp32,        // [16, 16, 16, path_count_max]
    float const *__restrict__ weights_max_fp32,       // [n_elements, path_count_max, n_channels]
    float *__restrict__ w3_effective_fp32             // [n_elements, 16, 16, 16, n_channels]
) {
    int const element_index = blockIdx.x;
    int const linear_ijk    = blockIdx.y;

    int const irrep_index_i = linear_ijk / (irrep_dim_k * irrep_dim_k);
    int const irrep_index_j = (linear_ijk / irrep_dim_k) % irrep_dim_k;
    int const irrep_index_k = linear_ijk % irrep_dim_k;

    float const *u3_at_ijk = u_matrix_3_fp32 + static_cast<std::uint64_t>(linear_ijk) * path_count_max;
    float const *weights_for_element = weights_max_fp32
        + static_cast<std::uint64_t>(element_index) * path_count_max * n_channels;
    float *w3_out_at_ijk = w3_effective_fp32
        + ((static_cast<std::uint64_t>(element_index) * irrep_dim_k + irrep_index_i) * irrep_dim_k + irrep_index_j)
              * irrep_dim_k * n_channels
        + static_cast<std::uint64_t>(irrep_index_k) * n_channels;

    for (std::uint32_t channel_index = static_cast<std::uint32_t>(threadIdx.x);
         channel_index < n_channels;
         channel_index += static_cast<std::uint32_t>(blockDim.x)) {
        float accumulator = 0.0f;
        #pragma unroll 8
        for (std::uint32_t path_index = 0u; path_index < path_count_max; ++path_index) {
            float const u_value = u3_at_ijk[path_index];
            float const w_value = weights_for_element[
                static_cast<std::uint64_t>(path_index) * n_channels + channel_index];
            accumulator += u_value * w_value;
        }
        w3_out_at_ijk[channel_index] = accumulator;
    }
}

} // namespace mace_tk_symmetric_contraction_detail_

/**
 *  @brief Host-side helper to precompute `W3_eff` once at engine construction.
 *
 *  Allocates a fresh device buffer of shape
 *  `[n_elements, 16, 16, 16, n_channels]` fp32 and runs the reduction kernel
 *  to fill it.  Caller owns the returned buffer and must release it with
 *  `cudaFree`.
 *
 *  Returns `nullptr` on allocation failure (e.g. the slab does not fit) or
 *  on a launch error; in that case the caller should fall back to the fp32
 *  reference path.
 */
inline float *mace_tk_precompute_w3_effective(
    std::uint32_t n_elements,
    std::uint32_t n_channels,
    std::uint32_t path_count_max,
    float const *u_matrix_3_fp32,
    float const *weights_max_fp32,
    cudaStream_t stream
) {
    using namespace mace_tk_symmetric_contraction_detail_;

    if (n_elements == 0u || n_channels == 0u || path_count_max == 0u) return nullptr;

    std::size_t const n_floats = static_cast<std::size_t>(n_elements)
                                 * static_cast<std::size_t>(irrep_dim_k)
                                 * static_cast<std::size_t>(irrep_dim_k)
                                 * static_cast<std::size_t>(irrep_dim_k)
                                 * static_cast<std::size_t>(n_channels);

    float *device_buffer = nullptr;
    cudaError_t alloc_err = cudaMalloc(reinterpret_cast<void **>(&device_buffer), n_floats * sizeof(float));
    if (alloc_err != cudaSuccess) return nullptr;

    // One CTA per (Z, i, j, k); threads = min(n_channels, 256) walk the
    // channel axis cooperatively.
    int const block_dim_x = (n_channels >= 256u) ? 256 : static_cast<int>(n_channels);
    dim3 const grid(n_elements,
                    static_cast<std::uint32_t>(irrep_dim_k * irrep_dim_k * irrep_dim_k),
                    1);
    dim3 const block(block_dim_x, 1, 1);

    mace_tk_precompute_w3_effective_kernel<<<grid, block, 0, stream>>>(
        n_channels, path_count_max,
        u_matrix_3_fp32, weights_max_fp32, device_buffer);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        cudaFree(device_buffer);
        return nullptr;
    }
    return device_buffer;
}

/**
 *  @brief Transform #1 host-side helper — produce `W3_partsym_sym` for the
 *  symmetric-contraction backward.  See the kernel's docstring for the
 *  precise algebraic identity.  Caller owns the buffer; release with `cudaFree`.
 */
inline float *mace_tk_precompute_w3_partsym(
    std::uint32_t n_elements,
    std::uint32_t n_channels,
    float const *w3_effective_fp32,
    cudaStream_t stream
) {
    using namespace mace_tk_symmetric_contraction_detail_;
    if (n_elements == 0u || n_channels == 0u || w3_effective_fp32 == nullptr) return nullptr;

    std::size_t const n_floats = static_cast<std::size_t>(n_elements)
                                 * static_cast<std::size_t>(irrep_dim_k)
                                 * static_cast<std::size_t>(irrep_dim_k)
                                 * static_cast<std::size_t>(irrep_dim_k)
                                 * static_cast<std::size_t>(n_channels);
    float *device_buffer = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&device_buffer), n_floats * sizeof(float)) != cudaSuccess) {
        return nullptr;
    }
    int const block_dim_x = (n_channels >= 256u) ? 256 : static_cast<int>(n_channels);
    dim3 const grid(n_elements,
                    static_cast<std::uint32_t>(irrep_dim_k * irrep_dim_k * irrep_dim_k),
                    1);
    dim3 const block(block_dim_x, 1, 1);
    mace_tk_precompute_w3_partsym_kernel<<<grid, block, 0, stream>>>(
        n_channels, w3_effective_fp32, device_buffer);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(device_buffer);
        return nullptr;
    }
    return device_buffer;
}

/**
 *  @brief Transform #1 host-side helper — produce the canonically-folded W3
 *  buffer that the v2 forward kernel iterates over.  Allocates a fresh device
 *  buffer of the same shape as W3_eff (`[n_elements, 16, 16, 16, n_channels]`)
 *  and fills the canonical `(i ≤ j ≤ k)` cells with the all-permutations sum.
 *
 *  Caller owns the buffer; release with `cudaFree`.
 */
inline float *mace_tk_precompute_w3_sym_total(
    std::uint32_t n_elements,
    std::uint32_t n_channels,
    float const *w3_effective_fp32,
    cudaStream_t stream
) {
    using namespace mace_tk_symmetric_contraction_detail_;
    if (n_elements == 0u || n_channels == 0u || w3_effective_fp32 == nullptr) return nullptr;

    std::size_t const n_floats = static_cast<std::size_t>(n_elements)
                                 * static_cast<std::size_t>(irrep_dim_k)
                                 * static_cast<std::size_t>(irrep_dim_k)
                                 * static_cast<std::size_t>(irrep_dim_k)
                                 * static_cast<std::size_t>(n_channels);
    float *device_buffer = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&device_buffer), n_floats * sizeof(float)) != cudaSuccess) {
        return nullptr;
    }
    // Non-canonical cells are never read; we don't bother zeroing them.

    int const block_dim_x = (n_channels >= 256u) ? 256 : static_cast<int>(n_channels);
    dim3 const grid(n_elements,
                    static_cast<std::uint32_t>(irrep_dim_k * irrep_dim_k * irrep_dim_k),
                    1);
    dim3 const block(block_dim_x, 1, 1);
    mace_tk_precompute_w3_sym_total_kernel<<<grid, block, 0, stream>>>(
        n_channels, w3_effective_fp32, device_buffer);

    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(device_buffer);
        return nullptr;
    }
    return device_buffer;
}

/**
 *  @brief Launch the TK-flavoured symmetric contraction.
 *
 *  Computes per-atom per-channel scalar invariants via the order-1 + order-2
 *  + order-3 ACE expansion, with the order-3 path's `path_index` reduction
 *  pre-folded into `precomputed_w3_effective_fp32` (see
 *  @ref mace_tk_precompute_w3_effective).
 *
 *  The output is fp32 — the order-3 accumulator's precision is load-bearing
 *  for the per-config energy diff target (≤ 1 meV), and bf16 here costs
 *  more than the saved bandwidth.
 *
 *  Layout assumptions (must match `launch_mace_symmetric_contraction`):
 *    - `node_features_full_fp32` is `[n_atoms, n_channels, irrep_dim_k=16]`,
 *      with the irrep axis as the inner stride-1 dimension.
 *    - `precomputed_w3_effective_fp32` is `[n_elements, 16, 16, 16, n_channels]`,
 *      with the channel axis as the inner stride-1 dimension (the layout the
 *      precompute helper emits).
 *    - `weights_1_fp32` is `[n_elements, 1, n_channels]`,
 *      `weights_2_fp32` is `[n_elements, 4, n_channels]`,
 *      `u_matrix_1_fp32` is `[16, 1]`, `u_matrix_2_fp32` is `[16, 16, 4]`.
 */
inline void launch_mace_tk_symmetric_contraction(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t n_elements,
    std::int32_t const *atom_element,
    float const *node_features_full_fp32,
    float const *u_matrix_1_fp32,
    float const *u_matrix_2_fp32,
    float const *weights_2_fp32,
    float const *weights_1_fp32,
    float const *precomputed_w3_effective_fp32,
    float *output_node_scalar_invariants_fp32,
    cudaStream_t stream
) {
    if (n_atoms == 0u) return;
    if (precomputed_w3_effective_fp32 == nullptr) return;

    using namespace mace_tk_symmetric_contraction_detail_;

    // Shared-memory budget: the per-atom feature slab is `n_channels × 16`
    // fp32 values.  For the largest MACE-OFF setting (n_channels = 192) this
    // is 12 KB; comfortably under the 48 KB default per-block limit.
    std::size_t const shared_memory_bytes =
        static_cast<std::size_t>(n_channels) * static_cast<std::size_t>(irrep_dim_k) * sizeof(float);

    dim3 const grid(n_atoms, 1, 1);
    dim3 const block(static_cast<std::uint32_t>(warp_thread_count_k), 1, 1);

    mace_tk_symmetric_contraction_kernel<<<grid, block, shared_memory_bytes, stream>>>(
        n_atoms, n_channels, n_elements,
        atom_element, node_features_full_fp32,
        u_matrix_1_fp32, u_matrix_2_fp32,
        weights_2_fp32, weights_1_fp32,
        precomputed_w3_effective_fp32,
        output_node_scalar_invariants_fp32);
}

} // namespace smallfield::cuda

#endif // SMALLFIELD_USE_THUNDERKITTENS && __CUDACC__
