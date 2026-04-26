/**
 *  @file mace_tk_per_irrep_linear_kernel.cuh
 *  @brief ThunderKittens-fused per-irrep linear (4-block channel mix) for MACE-OFF.
 *
 *  e3nn `o3.Linear(c_in × Σ_L 1xL, c_out × Σ_L 1xL)` factorises into one
 *  independent `(c_in, c_out)` matmul per L block: for L=0,1,2,3 the
 *  `(2L+1)` m-components share the same channel-mix weight.  In MACE-OFF23
 *  this op appears in three places:
 *
 *    - `linear_up`        (`active_irrep_block_count = 1`, L=0 only),
 *    - `linear_post_tp`   (`active_irrep_block_count = 4`, all L blocks),
 *    - `product_linear`   (`active_irrep_block_count = 1`, L=0 only).
 *
 *  The fp32 reference path in `mace_linear_kernel.cuh` launches all 4 blocks
 *  as a single hand-rolled kernel with one thread per (atom, c_out, m).  The
 *  natural fit on Hopper is a bf16 WGMMA on a 96 × 96 tile per block — but
 *  per-atom WGMMAs would each waste M=64 - 16 = 48 rows on the WGMMA M axis.
 *  This kernel batches 4 atoms per CTA so the M=64 WGMMA tile maps to
 *  `4 atoms × 16 m-positions`.  All 4 L blocks share the same shared-memory
 *  input tile (loaded once) and the same fp32 register accumulator type
 *  (zeroed and refilled per block); only the weight tile differs across
 *  blocks.  After the four WGMMAs we transpose each per-block accumulator
 *  back to the `[c_out, m]` per-atom layout via a shared bf16 staging tile.
 *
 *  Layout notes that differ between the input/output tensor and the WGMMA:
 *
 *    - The MACE feature tensor is `[n_atoms, c, irrep_dim_total = 16]` row-
 *      major; per atom that is a `[c, 16]` slab with m as the fastest axis.
 *    - The WGMMA `mm_AB(D, A, B)` wants A as `[M, K]` (m-rows, c_in cols)
 *      and B as `[K, N]` (c_in rows, c_out cols).  We therefore transpose
 *      each atom's input slab from `[c_in, 16]` into `[16, c_in]` while
 *      staging into shared memory.  The weight blob is already
 *      `[c_in, c_out]` per block, so it copies straight in.
 *    - The accumulator is `[64, c_out] = [4 atoms × 16 m, c_out]`.  The
 *      output expects `[c_out, 16]` per atom, so we transpose again at the
 *      bf16 store boundary, masking m-rows that don't belong to the current
 *      L block.  Inactive blocks (when `active_irrep_block_count < 4`) have
 *      their m-rows zeroed in the output, matching the fp32 reference.
 *
 *  The L block ↔ m mapping (L_max = 3, irrep_dim_total = 16) is canonical:
 *      L=0 → m=0          (1 m-component)
 *      L=1 → m=1..3       (3 m-components)
 *      L=2 → m=4..8       (5 m-components)
 *      L=3 → m=9..15      (7 m-components)
 *
 *  Path-weight: e3nn applies `1 / sqrt(c_in)` uniformly across blocks for
 *  the one-instruction-per-output `o3.Linear` we use here; the kernel folds
 *  this scale into the per-block accumulator before the bf16 store.
 *
 *  Validation: a standalone smoke test in `/tmp/test_tk_per_irrep_linear.cu`
 *  compares this kernel against `launch_mace_linear_per_irrep` (fp32) on a
 *  64-atom synthetic batch.  bf16 noise on a single matmul should keep the
 *  max-abs diff at ~5e-3 or better.
 *
 *  Gated on `SMALLFIELD_USE_THUNDERKITTENS`; when the macro is off this file
 *  is empty and the fp32 reference path stays the only per-irrep linear.
 */

#pragma once

#include "smallfield/cuda/mace_tk_tile.cuh"

#if defined(SMALLFIELD_USE_THUNDERKITTENS) && defined(__CUDACC__)

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace smallfield::cuda {

namespace mace_tk_per_irrep_linear_detail_ {

// Compile-time shape constants for MACE-OFF23-small/medium/large.  WGMMA
// requires M = 64 (= 4 m-blocks of 16) on Hopper; we map the M axis to
// `atoms_per_cta_k × irrep_dim_total_k = 4 × 16 = 64`.  K (= c_in) and
// N (= c_out) are 96 for MACE-OFF-small.  We bake the small shapes here
// because the production deployment is small/medium and the medium model
// also uses c = 96 channels.  Larger configurations (c = 192) would need
// a second specialization or a dynamic K/N tile loop.
constexpr int atoms_per_cta_k         = 4;
constexpr int irrep_dim_total_k       = mace_tk::mace_irrep_dim_total_k;     // 16
constexpr int wgmma_m_rows_k          = atoms_per_cta_k * irrep_dim_total_k; // 64
constexpr int channels_in_k           = mace_tk::mace_channels_small_k;      // 96
constexpr int channels_out_k          = mace_tk::mace_channels_small_k;      // 96
constexpr int irrep_block_count_k     = 4;                                    // L = 0, 1, 2, 3
constexpr int warpgroup_thread_count_k = kittens::WARPGROUP_THREADS;          // 128

// Per-block m-range.  Indices into the irrep_dim_total = 16 axis that
// correspond to L = `irrep_block_index`.  Used to mask non-block m-rows
// at output-store time.  Spelled out as `__device__ __forceinline__`
// helpers rather than `constexpr` arrays so the array storage lives in
// device-visible memory regardless of host/device translation-unit split
// (NVCC otherwise rejects array-form constexpr access from `__global__`).
__device__ __forceinline__ int irrep_block_m_start_(int irrep_block_index) {
    constexpr int starts[irrep_block_count_k] = {0, 1, 4, 9};
    return starts[irrep_block_index];
}
__device__ __forceinline__ int irrep_block_m_count_(int irrep_block_index) {
    constexpr int counts[irrep_block_count_k] = {1, 3, 5, 7};
    return counts[irrep_block_index];
}

// Shared-tile aliases.  `shared_input_tile_t` holds 4 atoms' inputs after
// transposing each `[c_in, 16]` slab to `[16, c_in]`, packed contiguously
// along the M axis.  Weights are `[c_in, c_out]` per block.  The output
// staging tile holds the per-block accumulator after fp32 → bf16
// conversion before we transpose-write back to global.
using shared_input_tile_t  = mace_tk::shared_tile_bf16<wgmma_m_rows_k, channels_in_k>;
using shared_weight_tile_t = mace_tk::shared_tile_bf16<channels_in_k, channels_out_k>;
using shared_output_tile_t = mace_tk::shared_tile_bf16<wgmma_m_rows_k, channels_out_k>;

// Per-warpgroup register accumulator.  Hopper warpgroup WGMMA produces a
// `(64, N)` fp32 register tile distributed across 4 warps × 16 rows each;
// in TK's `rt_fl<H_in_tiles, W_in_elements>` that is `rt_fl<1, channels_out_k>`.
using register_accumulator_t = kittens::rt_fl<16, channels_out_k>;

/**
 *  @brief Layer-wide path scale `1 / sqrt(c_in)` (e3nn `o3.Linear`).
 *
 *  Computed once on the host so the device path skips the per-thread sqrt.
 *  Folded into each per-block accumulator before the bf16 store epilogue.
 */
struct per_irrep_linear_scales_t {
    float path_weight;
};

/**
 *  @brief Stage 4 atoms' input slabs into one shared `[64, c_in]` bf16 tile.
 *
 *  Reads `[c_in, 16]` per atom from global fp32 and writes a transposed
 *  `[16, c_in]` bf16 sub-tile starting at M-row = `atom_in_cta * 16`.
 *  Out-of-range atoms (`base_atom + atom_in_cta >= n_atoms`) zero-fill
 *  their 16 rows so downstream WGMMA produces zero contributions which
 *  the per-atom store path then ignores via the same atom-index guard.
 */
/// Identity / type-converting bf16 cast — used inside the templated input
/// loader so the bf16 path skips the `__float2bfloat16_rn` round-trip.
__device__ __forceinline__ __nv_bfloat16 input_to_bf16_(float v) {
    return __float2bfloat16_rn(v);
}
__device__ __forceinline__ __nv_bfloat16 input_to_bf16_(__nv_bfloat16 v) {
    return v;
}

template <typename input_t>
__device__ __forceinline__ void load_input_tile_transposed_(
    shared_input_tile_t &dst_input_tile,
    input_t const *input_global,
    std::uint32_t n_atoms,
    std::uint32_t base_atom_index,
    std::uint32_t n_channels_in_runtime,
    std::uint32_t input_irrep_dim_runtime
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = wgmma_m_rows_k * channels_in_k;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const m_row_in_tile  = linear_index / channels_in_k;
        int const channel_in_col = linear_index % channels_in_k;
        int const atom_in_cta    = m_row_in_tile / irrep_dim_total_k;
        int const m_in_atom      = m_row_in_tile % irrep_dim_total_k;
        std::uint32_t const atom_index = base_atom_index + static_cast<std::uint32_t>(atom_in_cta);

        input_t value = input_t{};  // default zero (fp32 0.0 or bf16 0)
        // The caller's input is `[n_atoms, c_in, input_irrep_dim_runtime]`:
        //   - linear_up / product_linear pass `input_irrep_dim_runtime = 1`
        //     (a scalar feature stack — only m = 0 lives in DRAM),
        //   - linear_post_tp passes `input_irrep_dim_runtime = 16`
        //     (the full L = 0..3 stack).
        // Either way, our `[64, c_in]` shared tile has 16 m-rows per atom
        // and we zero-fill the m-rows beyond `input_irrep_dim_runtime`.
        if (atom_index < n_atoms &&
            static_cast<std::uint32_t>(channel_in_col) < n_channels_in_runtime &&
            static_cast<std::uint32_t>(m_in_atom)      < input_irrep_dim_runtime) {
            std::uint64_t const offset =
                (static_cast<std::uint64_t>(atom_index) * n_channels_in_runtime
                 + static_cast<std::uint32_t>(channel_in_col)) * input_irrep_dim_runtime
                + static_cast<std::uint32_t>(m_in_atom);
            value = input_global[offset];
        }
        dst_input_tile[{m_row_in_tile, channel_in_col}] = input_to_bf16_(value);
    }
}

/**
 *  @brief Stage one block's `[c_in, c_out]` weight from fp32 global into bf16 shared.
 *
 *  e3nn stores per-instruction `o3.Linear` weights as a single flat blob of
 *  `[n_blocks, c_in, c_out]`; this helper copies one block's slab and
 *  converts fp32 → bf16 in flight.  Called once per L block per CTA.
 */
__device__ __forceinline__ void load_block_weight_(
    shared_weight_tile_t &dst_weight_tile,
    float const *weight_blocks_global,
    std::uint32_t irrep_block_index,
    std::uint32_t n_channels_in_runtime,
    std::uint32_t n_channels_out_runtime
) {
    int const thread_id = threadIdx.x;
    constexpr int tile_elements = channels_in_k * channels_out_k;
    std::uint64_t const block_base = static_cast<std::uint64_t>(irrep_block_index)
                                     * n_channels_in_runtime * n_channels_out_runtime;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < tile_elements; linear_index += warpgroup_thread_count_k) {
        int const channel_in_row  = linear_index / channels_out_k;
        int const channel_out_col = linear_index % channels_out_k;
        float value = 0.0f;
        if (static_cast<std::uint32_t>(channel_in_row)  < n_channels_in_runtime &&
            static_cast<std::uint32_t>(channel_out_col) < n_channels_out_runtime) {
            std::uint64_t const offset = block_base
                + static_cast<std::uint64_t>(channel_in_row) * n_channels_out_runtime
                + static_cast<std::uint32_t>(channel_out_col);
            value = weight_blocks_global[offset];
        }
        dst_weight_tile[{channel_in_row, channel_out_col}] = __float2bfloat16_rn(value);
    }
}

/**
 *  @brief Apply the e3nn `1 / sqrt(c_in)` path-weight scale to the accumulator.
 *
 *  Fused via `kittens::warpgroup::apply` on every (row, col) lane in the
 *  fp32 register tile.  Matches the fp32 reference's `scale * accumulator`
 *  step in `mace_linear_per_irrep_kernel`.
 */
__device__ __forceinline__ void scale_accumulator_in_place_(
    register_accumulator_t &accumulator,
    float path_weight
) {
    kittens::warpgroup::apply(
        accumulator,
        accumulator,
        [path_weight] __device__ (int /*row*/, int /*col*/, float pre) {
            return pre * path_weight;
        });
}

/**
 *  @brief Transpose-and-write one block's `[64, c_out]` accumulator slab
 *         to global `[n_atoms, c_out, 16]`, masking m-rows outside the block.
 *
 *  The shared bf16 staging tile already holds the post-scale post-conversion
 *  accumulator (from `kittens::warpgroup::store(st, rt)`).  We then walk
 *  `(atom_in_cta, c_out, m_in_block)` triples cooperatively across the
 *  warpgroup and emit one bf16 element each — the m_in_block walk maps
 *  back to the absolute m-position via `irrep_block_m_start_k[block]`.
 *  Out-of-range atoms (`base_atom + atom_in_cta >= n_atoms`) are skipped.
 */
__device__ __forceinline__ void store_block_output_transposed_(
    __nv_bfloat16 *output_global,
    shared_output_tile_t const &staged_output_tile,
    std::uint32_t n_atoms,
    std::uint32_t base_atom_index,
    std::uint32_t n_channels_out_runtime,
    std::uint32_t irrep_dim_runtime,
    int irrep_block_index
) {
    int const thread_id = threadIdx.x;
    int const m_start   = irrep_block_m_start_(irrep_block_index);
    int const m_count   = irrep_block_m_count_(irrep_block_index);
    int const total_elements = atoms_per_cta_k * static_cast<int>(n_channels_out_runtime) * m_count;

    #pragma unroll
    for (int linear_index = thread_id; linear_index < total_elements; linear_index += warpgroup_thread_count_k) {
        int const atom_in_cta    = linear_index / (static_cast<int>(n_channels_out_runtime) * m_count);
        int const within_atom    = linear_index % (static_cast<int>(n_channels_out_runtime) * m_count);
        int const channel_out    = within_atom / m_count;
        int const m_in_block     = within_atom % m_count;
        int const m_in_atom      = m_start + m_in_block;
        std::uint32_t const atom_index = base_atom_index + static_cast<std::uint32_t>(atom_in_cta);
        if (atom_index >= n_atoms) continue;

        int const m_row_in_tile = atom_in_cta * irrep_dim_total_k + m_in_atom;
        __nv_bfloat16 const value = staged_output_tile[{m_row_in_tile, channel_out}];

        std::uint64_t const out_offset =
            (static_cast<std::uint64_t>(atom_index) * n_channels_out_runtime
             + static_cast<std::uint32_t>(channel_out)) * irrep_dim_runtime
            + static_cast<std::uint32_t>(m_in_atom);
        output_global[out_offset] = value;
    }
}

/**
 *  @brief Zero the m-rows that belong to inactive L blocks in the output.
 *
 *  When `active_irrep_block_count < 4` (e.g. `linear_up` and `product_linear`,
 *  which are L=0-only), the higher-L m-rows of the output are required to
 *  be zero so downstream consumers (`mace_tensor_product_kernel`,
 *  symmetric contraction) see a clean tensor.  Mirrors the
 *  `output[...] = 0.0f` early-return in the fp32 reference.
 */
__device__ __forceinline__ void zero_inactive_block_m_rows_(
    __nv_bfloat16 *output_global,
    std::uint32_t n_atoms,
    std::uint32_t base_atom_index,
    std::uint32_t n_channels_out_runtime,
    std::uint32_t irrep_dim_runtime,
    std::uint32_t active_irrep_block_count
) {
    int const thread_id = threadIdx.x;
    int const inactive_block_index = static_cast<int>(active_irrep_block_count);
    if (inactive_block_index >= irrep_block_count_k) return;

    int const m_inactive_start = irrep_block_m_start_(inactive_block_index);
    int const m_inactive_count = static_cast<int>(irrep_dim_runtime) - m_inactive_start;
    if (m_inactive_count <= 0) return;

    int const total_elements = atoms_per_cta_k * static_cast<int>(n_channels_out_runtime) * m_inactive_count;
    #pragma unroll
    for (int linear_index = thread_id; linear_index < total_elements; linear_index += warpgroup_thread_count_k) {
        int const atom_in_cta = linear_index / (static_cast<int>(n_channels_out_runtime) * m_inactive_count);
        int const within_atom = linear_index % (static_cast<int>(n_channels_out_runtime) * m_inactive_count);
        int const channel_out = within_atom / m_inactive_count;
        int const m_offset    = within_atom % m_inactive_count;
        int const m_in_atom   = m_inactive_start + m_offset;
        std::uint32_t const atom_index = base_atom_index + static_cast<std::uint32_t>(atom_in_cta);
        if (atom_index >= n_atoms) continue;
        std::uint64_t const out_offset =
            (static_cast<std::uint64_t>(atom_index) * n_channels_out_runtime
             + static_cast<std::uint32_t>(channel_out)) * irrep_dim_runtime
            + static_cast<std::uint32_t>(m_in_atom);
        output_global[out_offset] = __float2bfloat16_rn(0.0f);
    }
}

/**
 *  @brief Persistent per-irrep linear kernel body: 4 atoms / CTA, 4 blocks fused.
 *
 *  Grid: `dim3(n_atom_tiles, 1, 1)` where `n_atom_tiles = ceil(n_atoms / 4)`.
 *  Block: 128 threads (one warpgroup, 4 warps).  Within a CTA we run one
 *  WGMMA per active L block, sharing the input shared tile across all
 *  blocks.  Inactive blocks (when `active_irrep_block_count < 4`) have
 *  their m-rows zeroed in the output instead.
 */
template <typename input_t>
inline __global__ void mace_tk_per_irrep_linear_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_channels_out,
    std::uint32_t input_irrep_dim_runtime,
    std::uint32_t output_irrep_dim_runtime,
    std::uint32_t active_irrep_block_count,
    float const *__restrict__ weight_blocks_global,
    input_t const *__restrict__ input_global,        // float OR __nv_bfloat16
    __nv_bfloat16 *__restrict__ output_global,
    per_irrep_linear_scales_t scales
) {
    extern __shared__ kittens::alignment_dummy raw_shared_memory[];
    kittens::shared_allocator shared_alloc(reinterpret_cast<int *>(&raw_shared_memory[0]));

    // Persistent shared tiles: one input tile (shared across all 4 blocks),
    // one weight tile (refilled per block), one output staging tile (re-used
    // per block as the bf16 conversion target before the transposed store).
    shared_input_tile_t  &shared_input_tile  = shared_alloc.allocate<shared_input_tile_t>();
    shared_weight_tile_t &shared_weight_tile = shared_alloc.allocate<shared_weight_tile_t>();
    shared_output_tile_t &shared_output_tile = shared_alloc.allocate<shared_output_tile_t>();

    std::uint32_t const base_atom_index = blockIdx.x * static_cast<std::uint32_t>(atoms_per_cta_k);
    if (base_atom_index >= n_atoms) return;

    // Stage the per-CTA input slab once; reused across all L blocks below.
    load_input_tile_transposed_<input_t>(
        shared_input_tile,
        input_global,
        n_atoms, base_atom_index,
        n_channels_in, input_irrep_dim_runtime);
    kittens::warpgroup::sync(0);

    // Zero inactive-block m-rows up front so subsequent active-block stores
    // overwrite cleanly without a dependency on the previous block's launch.
    zero_inactive_block_m_rows_(
        output_global,
        n_atoms, base_atom_index,
        n_channels_out, output_irrep_dim_runtime,
        active_irrep_block_count);

    register_accumulator_t accumulator;

    // ---- One WGMMA per active L block ----
    #pragma unroll
    for (int irrep_block_index = 0; irrep_block_index < irrep_block_count_k; ++irrep_block_index) {
        if (static_cast<std::uint32_t>(irrep_block_index) >= active_irrep_block_count) break;

        // Stage this block's `[c_in, c_out]` weight tile.
        load_block_weight_(
            shared_weight_tile,
            weight_blocks_global,
            static_cast<std::uint32_t>(irrep_block_index),
            n_channels_in, n_channels_out);
        kittens::warpgroup::sync(0);

        // C = A @ B with A = `[64, c_in]` (transposed input), B = `[c_in,
        // c_out]` (weights), C = `[64, c_out]` accumulator (atom-major
        // m-rows × c_out cols).  All M=64 rows are computed; the per-block
        // store path below masks the m-rows that don't belong to L = block.
        kittens::warp::zero(accumulator);
        kittens::warpgroup::mm_AB(accumulator, shared_input_tile, shared_weight_tile);
        kittens::warpgroup::mma_async_wait();

        scale_accumulator_in_place_(accumulator, scales.path_weight);

        // fp32 -> bf16 staging in shared memory; then transpose-store the
        // m-rows of this block to global `[n_atoms, c_out, 16]`.  We re-use
        // the staging tile across blocks, which is safe because each block
        // walks the staging tile to completion before the next block's
        // WGMMA writes back into it.
        kittens::warpgroup::store(shared_output_tile, accumulator);
        kittens::warpgroup::sync(0);

        store_block_output_transposed_(
            output_global,
            shared_output_tile,
            n_atoms, base_atom_index,
            n_channels_out, output_irrep_dim_runtime,
            irrep_block_index);
        kittens::warpgroup::sync(0);
    }
}

} // namespace mace_tk_per_irrep_linear_detail_

/**
 *  @brief Launch the TK-fused per-irrep linear for MACE-OFF.
 *
 *  Reads `weight_blocks_fp32[active_irrep_block_count, c_in, c_out]` (e3nn
 *  `o3.Linear` per-instruction layout) and `input_fp32[n_atoms, c_in, 16]`
 *  (the project-wide MACE feature layout); writes
 *  `output_bf16[n_atoms, c_out, 16]`.  Applies `path_weight = 1 / sqrt(c_in)`
 *  uniformly per block.  When `active_irrep_block_count < 4` the higher-L
 *  m-rows of the output are zeroed (matching the fp32 reference's early
 *  return for inactive blocks).
 *
 *  Threading: `n_atom_tiles = ceil(n_atoms / 4)` CTAs, 128 threads each
 *  (one warpgroup).  The kernel templates the small/medium MACE-OFF shapes
 *  (c_in = c_out = 96) into the shared tile sizes; runtime mismatches
 *  (e.g. larger c_in) are caught by the runtime-vs-compile-time channel
 *  guards inside the staging helpers.
 *
 *  Gated on `SMALLFIELD_USE_THUNDERKITTENS`.  When the macro is off the
 *  driver should fall back to `launch_mace_linear_per_irrep` in
 *  `mace_linear_kernel.cuh`.
 */
/// Returns true on successful launch, false if the (n_channels_in,
/// n_channels_out, active_irrep_block_count) tuple is not handled by this
/// kernel.  The caller MUST check the return and fall back to the fp32
/// reference path otherwise — the previous version silently `return`-ed
/// and left the output buffer with stale data, which silently broke the
/// medium (128 ch) and large (224 ch) MACE-OFF models under the TK build.
inline bool launch_mace_tk_per_irrep_linear(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_channels_out,
    std::uint32_t active_irrep_block_count,
    float const *weight_blocks_fp32,
    float const *input_fp32,
    __nv_bfloat16 *output_bf16,
    cudaStream_t stream,
    float pre_scale_multiplier = 1.0f                    // Megakernel B β-2: fold an extra
                                                          // uniform scale (typically 1/avg_num_neighbors)
                                                          // into path_weight so the caller can skip
                                                          // a downstream scale_in_place launch.
) {
    if (n_atoms == 0u) return true;

    using namespace mace_tk_per_irrep_linear_detail_;

    // The fused kernel templates the MACE-OFF-small shape (c_in = c_out = 96)
    // into the WGMMA tile.  Other configurations (medium 128, large 224)
    // need a separate specialization (Transform #4 in the algebraic-transforms
    // roadmap); we reject explicitly here so the caller can route to the fp32
    // reference path instead of silently producing zeros.
    if (n_channels_in  != static_cast<std::uint32_t>(channels_in_k) ||
        n_channels_out != static_cast<std::uint32_t>(channels_out_k)) {
        return false;
    }
    if (active_irrep_block_count == 0u || active_irrep_block_count > static_cast<std::uint32_t>(irrep_block_count_k)) {
        return false;
    }

    per_irrep_linear_scales_t scales;
    scales.path_weight = pre_scale_multiplier / sqrtf(static_cast<float>(n_channels_in));

    // Shared-mem footprint, one of each tile type:
    //   shared_input_tile     64 x 96 bf16 = 12 KB
    //   shared_weight_tile    96 x 96 bf16 = 18 KB
    //   shared_output_tile    64 x 96 bf16 = 12 KB
    //   ----                                ------
    //                                       ~42 KB + alignment slack
    constexpr std::size_t shared_memory_bytes_k = 64 * 1024;  // generous, well under H100's 228 KB cap

    cudaFuncSetAttribute(
        mace_tk_per_irrep_linear_kernel<float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_memory_bytes_k));

    // Mirror the fp32 reference's caller convention: `linear_up` and
    // `product_linear` pass a scalar `[n_atoms, c_in]` input (conceptually
    // `[n_atoms, c_in, 1]`), while `linear_post_tp` passes the full
    // `[n_atoms, c_in, 16]` stack.  The output is always `[n_atoms, c_out, 16]`
    // — the caller relies on the higher-L m-rows being zeroed when
    // `active_irrep_block_count == 1`.
    std::uint32_t const input_irrep_dim_runtime  =
        (active_irrep_block_count == 1u) ? 1u : static_cast<std::uint32_t>(irrep_dim_total_k);
    std::uint32_t const output_irrep_dim_runtime = static_cast<std::uint32_t>(irrep_dim_total_k);

    std::uint32_t const n_atom_tiles = (n_atoms + static_cast<std::uint32_t>(atoms_per_cta_k) - 1u)
                                       / static_cast<std::uint32_t>(atoms_per_cta_k);
    dim3 const grid(n_atom_tiles, 1, 1);
    dim3 const block(static_cast<std::uint32_t>(warpgroup_thread_count_k), 1, 1);

    mace_tk_per_irrep_linear_kernel<float><<<grid, block, shared_memory_bytes_k, stream>>>(
        n_atoms,
        n_channels_in,
        n_channels_out,
        input_irrep_dim_runtime,
        output_irrep_dim_runtime,
        active_irrep_block_count,
        weight_blocks_fp32,
        input_fp32,
        output_bf16,
        scales);
    return true;
}

/**
 *  @brief Bf16-input variant of `launch_mace_tk_per_irrep_linear` —
 *         Megakernel B Phase β-3.  Reads input directly from a bf16 buffer
 *         (typically the `S.tk_node_messages_full_irrep_bf16` produced by
 *         the conv_tp_scatter fused kernel) instead of going through an
 *         fp32 cast bridge.
 *
 *  Same `pre_scale_multiplier` semantics as the fp32 launcher (Phase β-2).
 *  Same shape constraints (c_in == c_out == 96; non-96 returns false so
 *  the caller falls back to the fp32 chain).
 */
inline bool launch_mace_tk_per_irrep_linear_bf16_input(
    std::uint32_t n_atoms,
    std::uint32_t n_channels_in,
    std::uint32_t n_channels_out,
    std::uint32_t active_irrep_block_count,
    float const *weight_blocks_fp32,
    __nv_bfloat16 const *input_bf16,
    __nv_bfloat16 *output_bf16,
    cudaStream_t stream,
    float pre_scale_multiplier = 1.0f
) {
    if (n_atoms == 0u) return true;

    using namespace mace_tk_per_irrep_linear_detail_;

    if (n_channels_in  != static_cast<std::uint32_t>(channels_in_k) ||
        n_channels_out != static_cast<std::uint32_t>(channels_out_k)) {
        return false;
    }
    if (active_irrep_block_count == 0u || active_irrep_block_count > static_cast<std::uint32_t>(irrep_block_count_k)) {
        return false;
    }

    per_irrep_linear_scales_t scales;
    scales.path_weight = pre_scale_multiplier / sqrtf(static_cast<float>(n_channels_in));

    constexpr std::size_t shared_memory_bytes_k = 64 * 1024;

    cudaFuncSetAttribute(
        mace_tk_per_irrep_linear_kernel<__nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_memory_bytes_k));

    std::uint32_t const input_irrep_dim_runtime  =
        (active_irrep_block_count == 1u) ? 1u : static_cast<std::uint32_t>(irrep_dim_total_k);
    std::uint32_t const output_irrep_dim_runtime = static_cast<std::uint32_t>(irrep_dim_total_k);

    std::uint32_t const n_atom_tiles = (n_atoms + static_cast<std::uint32_t>(atoms_per_cta_k) - 1u)
                                       / static_cast<std::uint32_t>(atoms_per_cta_k);
    dim3 const grid(n_atom_tiles, 1, 1);
    dim3 const block(static_cast<std::uint32_t>(warpgroup_thread_count_k), 1, 1);

    mace_tk_per_irrep_linear_kernel<__nv_bfloat16><<<grid, block, shared_memory_bytes_k, stream>>>(
        n_atoms,
        n_channels_in,
        n_channels_out,
        input_irrep_dim_runtime,
        output_irrep_dim_runtime,
        active_irrep_block_count,
        weight_blocks_fp32,
        input_bf16,
        output_bf16,
        scales);
    return true;
}

} // namespace smallfield::cuda

#endif // SMALLFIELD_USE_THUNDERKITTENS && __CUDACC__
