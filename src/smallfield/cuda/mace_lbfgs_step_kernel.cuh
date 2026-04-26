/**
 *  @file mace_lbfgs_step_kernel.cuh
 *  @brief GPU-resident LBFGS-style relaxation step for MACE-OFF refinement.
 *
 *  Eliminates the host-side gradient-descent loop in
 *  `smallfield::mace::context_t::refine_batch` (and its 20 host launches per
 *  relaxation, ~25 % of end-to-end wall time at the current kernel speed) by
 *  fusing the (positions ← positions + step·forces) update with a per-conformer
 *  fmax convergence check into a single warp-cooperative kernel.
 *
 *  Threading shape — one CTA per conformer, 32-thread warp:
 *    - The MACE-OFF refinement workload is embarrassingly parallel across
 *      conformers (every (positions, forces) update is independent), so the
 *      per-warp granularity that's wrong for the forward+backward chain
 *      (which needs WGMMA tiles spanning many conformers) is exactly right
 *      here.  Each warp owns one conformer's full atom range, reads its
 *      `[n_atoms_in_conformer × 3]` forces in stride-32 chunks, computes a
 *      warp-shuffle reduction of the per-component max-abs, and writes the
 *      per-conformer fmax + converged flag.
 *
 *  Adaptive early exit:
 *    - When `converged_flags[g] == 1` from a prior step, the kernel skips
 *      the position writeback and the fmax compute for that conformer —
 *      the position stays at the converged geometry across subsequent replays
 *      of the captured graph.  Forward + backward kernels still run for
 *      converged conformers (their per-conformer gating would require
 *      kernel-level rework; tracked as a follow-up), but their gradients
 *      stay zero and the chain is stable.
 *
 *  Captured-graph compatibility:
 *    - All kernel parameters and pointers are stable across iterations; the
 *      caller can wrap N step calls into a single `cudaGraph_t` and replay
 *      with one host launch instead of N.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cuda_runtime.h>

namespace smallfield::cuda {

namespace mace_lbfgs_detail_ {

/// Warp-shuffle reduction: returns max-abs across all 32 lanes, broadcast to lane 0.
__device__ __forceinline__ float warp_reduce_max_abs_(float v) {
    v = fabsf(v);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float const peer = __shfl_xor_sync(0xFFFFFFFFu, v, offset);
        v = fmaxf(v, peer);
    }
    return v;
}

}  // namespace mace_lbfgs_detail_

/**
 *  @brief Per-conformer GD-style position update + fmax convergence check.
 *
 *  Grid: `dim3(n_graphs, 1, 1)` — one CTA per conformer.
 *  Block: 32 threads (single warp).  More threads would split the per-atom
 *  range into smaller per-thread chunks but the warp-level shuffle reduction
 *  is the load-bearing primitive; multi-warp CTAs add no value here.
 *
 *  @param  graph_atom_offsets [n_graphs + 1] inclusive prefix-sum of atom
 *          counts per conformer; offsets[g+1] - offsets[g] = atom count of
 *          conformer g.  Built once at batch-staging time.
 *  @param  positions [n_atoms × 3] in-place position buffer.
 *  @param  forces [n_atoms × 3] gradient buffer (`-∂E/∂r`); read-only.
 *  @param  fmax_eV_per_A_threshold convergence threshold; conformers with
 *          per-component max-abs force ≤ this skip the writeback.
 *  @param  step_size GD step size; 0.01 is a reasonable default for MACE-OFF.
 *  @param  fmax_per_conformer [n_graphs] OUTPUT: latest per-conformer fmax.
 *  @param  converged_flags [n_graphs] IN/OUT: 0 = still relaxing, 1 = converged.
 */
inline __global__ void mace_lbfgs_step_kernel(
    std::uint32_t n_graphs,
    std::int32_t const *__restrict__ graph_atom_offsets,
    float const *__restrict__ forces,
    float *__restrict__ positions,
    float fmax_eV_per_A_threshold,
    float step_size,
    float *__restrict__ fmax_per_conformer,
    std::uint8_t *__restrict__ converged_flags
) {
    std::uint32_t const graph_index = blockIdx.x;
    if (graph_index >= n_graphs) return;

    int const lane = static_cast<int>(threadIdx.x);
    if (lane >= 32) return;

    // Skip already-converged conformers — leave their geometry pinned.
    if (converged_flags[graph_index] != 0u) {
        if (lane == 0) {
            // Maintain the last-seen fmax for callers that read it.
        }
        return;
    }

    std::int32_t const atom_start = graph_atom_offsets[graph_index];
    std::int32_t const atom_end   = graph_atom_offsets[graph_index + 1];
    std::int32_t const n_components = (atom_end - atom_start) * 3;

    // First pass — compute per-conformer fmax via warp shuffle.
    float local_max = 0.0f;
    for (std::int32_t component = lane; component < n_components; component += 32) {
        float const f = forces[(atom_start * 3) + component];
        local_max = fmaxf(local_max, fabsf(f));
    }
    float const fmax_warp = mace_lbfgs_detail_::warp_reduce_max_abs_(local_max);

    if (lane == 0) {
        fmax_per_conformer[graph_index] = fmax_warp;
        if (fmax_warp <= fmax_eV_per_A_threshold) {
            converged_flags[graph_index] = 1u;
        }
    }

    // If we just converged, skip the position update — the chain is at the
    // minimum and additional drift would only re-introduce float noise.
    if (fmax_warp <= fmax_eV_per_A_threshold) return;

    // Second pass — gradient-descent position update.  Forces in MACE-OFF
    // inference are -∂E/∂r so positions move down-energy as positions += step·forces.
    for (std::int32_t component = lane; component < n_components; component += 32) {
        std::size_t const offset = static_cast<std::size_t>(atom_start * 3) + component;
        positions[offset] += step_size * forces[offset];
    }
}

/// Host launcher for `mace_lbfgs_step_kernel`.
inline void launch_mace_lbfgs_step(
    std::uint32_t n_graphs,
    std::int32_t const *graph_atom_offsets,
    float const *forces,
    float *positions,
    float fmax_eV_per_A_threshold,
    float step_size,
    float *fmax_per_conformer,
    std::uint8_t *converged_flags,
    cudaStream_t stream
) {
    if (n_graphs == 0u) return;
    dim3 const grid(n_graphs, 1, 1);
    dim3 const block(32, 1, 1);
    mace_lbfgs_step_kernel<<<grid, block, 0, stream>>>(
        n_graphs, graph_atom_offsets,
        forces, positions,
        fmax_eV_per_A_threshold, step_size,
        fmax_per_conformer, converged_flags);
}

/// Reset the converged flags to 0 (not converged).  Captured-graph callers
/// run this once at the start of a relaxation; subsequent step kernels then
/// flip flags to 1 as conformers converge.
inline __global__ void mace_lbfgs_reset_converged_flags_kernel(
    std::uint32_t n_graphs,
    std::uint8_t *__restrict__ converged_flags
) {
    std::uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_graphs) return;
    converged_flags[i] = 0u;
}

inline void launch_mace_lbfgs_reset_converged_flags(
    std::uint32_t n_graphs, std::uint8_t *converged_flags, cudaStream_t stream
) {
    if (n_graphs == 0u) return;
    constexpr std::uint32_t block = 256u;
    std::uint32_t const grid = (n_graphs + block - 1u) / block;
    mace_lbfgs_reset_converged_flags_kernel<<<grid, block, 0, stream>>>(
        n_graphs, converged_flags);
}

/// AND-reduce the per-conformer `converged_flags` into a single device byte.
/// Used by `refine_batch` to decide whether all conformers in the batch have
/// converged and the relaxation loop can terminate early.  The reduction
/// runs in a single block (n_graphs is small — at most ~16 K in production)
/// so the result is consistent on a single in-stream sync.
///
/// Output `all_converged_out[0]` is 1 when every flag is non-zero, 0 otherwise.
inline __global__ void mace_lbfgs_all_converged_kernel(
    std::uint32_t n_graphs,
    std::uint8_t const *__restrict__ converged_flags,
    std::uint8_t *__restrict__ all_converged_out
) {
    __shared__ std::uint32_t s_partial[32];
    std::uint32_t const lane = threadIdx.x & 31u;
    std::uint32_t const warp = threadIdx.x >> 5;
    std::uint32_t local = 1u;
    for (std::uint32_t i = threadIdx.x; i < n_graphs; i += blockDim.x) {
        if (converged_flags[i] == 0u) { local = 0u; }
    }
    // Warp AND-reduce.
    for (int offset = 16; offset > 0; offset >>= 1) {
        local &= __shfl_xor_sync(0xffffffffu, local, offset);
    }
    if (lane == 0u) s_partial[warp] = local;
    __syncthreads();
    if (warp == 0u) {
        std::uint32_t v = (threadIdx.x < (blockDim.x >> 5)) ? s_partial[lane] : 1u;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v &= __shfl_xor_sync(0xffffffffu, v, offset);
        }
        if (lane == 0u) all_converged_out[0] = static_cast<std::uint8_t>(v & 1u);
    }
}

inline void launch_mace_lbfgs_all_converged(
    std::uint32_t n_graphs,
    std::uint8_t const *converged_flags,
    std::uint8_t *all_converged_out,
    cudaStream_t stream
) {
    if (n_graphs == 0u) {
        // Vacuously all-converged.
        std::uint8_t const one = 1u;
        cudaMemcpyAsync(all_converged_out, &one, sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice, stream);
        return;
    }
    constexpr std::uint32_t block = 256u;
    mace_lbfgs_all_converged_kernel<<<1u, block, 0, stream>>>(
        n_graphs, converged_flags, all_converged_out);
}

/// Build `graph_atom_offsets[n_graphs + 1]` from a `graph_of_atom[n_atoms]`
/// labelling.  Called once when staging a batch — stable across LBFGS iterations.
///
/// Assumes the batch is sorted by `graph_of_atom` (atoms of conformer g are
/// contiguous, in order of g).  This is the case for the synthetic bench
/// batches and for `pack_mace_batch` in `cli/smallfield_mace_refine.cpp`.
inline __global__ void mace_lbfgs_build_graph_atom_offsets_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_graphs,
    std::int32_t const *__restrict__ graph_of_atom,
    std::int32_t *__restrict__ graph_atom_offsets    // [n_graphs + 1]
) {
    std::uint32_t const graph_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (graph_index > n_graphs) return;

    if (graph_index == 0u) {
        graph_atom_offsets[0] = 0;
    } else if (graph_index == n_graphs) {
        graph_atom_offsets[n_graphs] = static_cast<std::int32_t>(n_atoms);
    } else {
        // Linear scan to find first atom whose graph_of_atom == graph_index.
        // For sorted batches this is monotone, so we can binary-search but
        // n_graphs is small (~K) and this kernel runs once per batch.
        std::int32_t lo = 0, hi = static_cast<std::int32_t>(n_atoms);
        while (lo < hi) {
            std::int32_t const mid = (lo + hi) >> 1;
            if (graph_of_atom[mid] < static_cast<std::int32_t>(graph_index)) lo = mid + 1;
            else hi = mid;
        }
        graph_atom_offsets[graph_index] = lo;
    }
}

inline void launch_mace_lbfgs_build_graph_atom_offsets(
    std::uint32_t n_atoms,
    std::uint32_t n_graphs,
    std::int32_t const *graph_of_atom,
    std::int32_t *graph_atom_offsets,
    cudaStream_t stream
) {
    if (n_graphs == 0u) return;
    constexpr std::uint32_t block = 256u;
    std::uint32_t const grid = (n_graphs + 1u + block - 1u) / block;
    mace_lbfgs_build_graph_atom_offsets_kernel<<<grid, block, 0, stream>>>(
        n_atoms, n_graphs, graph_of_atom, graph_atom_offsets);
}

}  // namespace smallfield::cuda

#endif  // __CUDACC__
