/**
 *  @file mace_tk_blackwell.cuh
 *  @brief Hopper-vs-consumer-Blackwell portability layer for MACE-TK MMA kernels.
 *
 *  ThunderKittens exposes one warpgroup-scope MMA API per arch:
 *
 *    - __Hopper__ (`KITTENS_HOPPER`, sm_90a):
 *      `kittens::warpgroup::mm_AB(D_register_tile, A_smem, B_smem)` — 4
 *      warps cooperate via the WGMMA hardware on a single m=64×N tile.
 *
 *    - __Datacenter Blackwell__ (`KITTENS_BLACKWELL` + sm_100a):
 *      `kittens::warpgroup::mm_AB(D_TT_tile, A_smem, B_smem, semaphore)`
 *      — async tcgen05 MMA into tensor memory.  Replaces WGMMA wholesale.
 *
 *    - __Consumer Blackwell__ (sm_120a / sm_121a, e.g. RTX PRO 6000):
 *      Neither WGMMA nor tcgen05 exists on this silicon (the tensor-memory
 *      interconnect is physically absent from the GB202 die).  The only
 *      tensor-core path is warp-scope `mma.sync.aligned.m16n8k16.*`, the
 *      synchronous register-resident family that goes back to Hopper /
 *      Ampere / Volta.  TK exposes this via `kittens::warp::mma_AB`, but
 *      every operand must live in registers, not shared memory — so the
 *      helper has to load A (per-warp slice via `warpgroup::load`) and
 *      B (full, replicated across warps via `warp::load`) into register
 *      tiles per call.  Each of the four warps in a CTA computes its own
 *      16-row slice of what was the warpgroup-level matmul; aggregate
 *      output lines up byte-for-byte with the WGMMA / tcgen05 versions.
 *
 *  Each MACE TK kernel was originally written against the Hopper API.
 *  This header gives every kernel one helper with identical signature on
 *  all three arches:
 *
 *      mace_tk::mma_AB_overwrite_sync(D_register_acc, A_smem, B_smem, ctx);
 *
 *  - Hopper:   inlines to `kittens::warpgroup::mm_AB` + `mma_async_wait`.
 *  - Datacenter Blackwell (sm_100a): NOT enabled in this build (TK's
 *    `KITTENS_BLACKWELL` would route to tcgen05 but ptxas rejects it on
 *    sm_120a).  Reachable when a B100 / B200 host appears.
 *  - Consumer Blackwell (sm_120a / sm_121a): per-warp warp-scope MMA.
 *
 *  v1 scope: synchronous single-CTA MMA per call, no async pipelining
 *  with the next iteration.  Numerical parity with the Hopper reference.
 */

#pragma once

#include "smallfield/module_fwd.hpp"
#include "smallfield/cuda/mace_tk_tile.cuh"

#if defined(__CUDACC__)

#include <kittens.cuh>

namespace smallfield::cuda::mace_tk {

#if defined(KITTENS_BLACKWELL)

/** Empty per-kernel context on consumer Blackwell — no allocator,
 *  no semaphore, no TT slot.  `init` accepts the caller's
 *  `kittens::shared_allocator` for signature parity with Hopper /
 *  datacenter-Blackwell variants; the arg is unused. */
template <typename register_tile_>
struct mma_ctx_t {
    template <typename shared_alloc_t_>
    __device__ inline void init(shared_alloc_t_ & /*unused*/) {}
};

/**
 *  @brief D = A * B (overwrite) — per-warp synchronous warp-scope MMA.
 *
 *  Loads:
 *    - A into a per-warp register tile via `kittens::warpgroup::load` —
 *      this collaborative load auto-distributes the M axis across the 4
 *      warps in the CTA, so each warp's `A_reg` holds its 16-row slice.
 *    - B into a per-warp register tile via `kittens::warp::load` — each
 *      warp loads the FULL B (4× redundant shared traffic, but each
 *      warp needs its own register copy for the warp-scope MMA).
 *  Computes:
 *    `kittens::warp::mma_AB(D_per_warp, A_reg, B_reg, D_per_warp)` after
 *    zeroing D, giving overwrite (not accumulate) semantics matching the
 *    Hopper `mm_AB` it replaces.
 *
 *  Cost relative to Hopper WGMMA: 4× shared-mem reads on B, no async
 *  pipelining, plus extra `ldmatrix` issues.  Acceptable trade because
 *  consumer Blackwell has no warpgroup-async path at all.
 */
template <typename register_tile_, typename a_smem_t_, typename b_smem_t_>
__device__ inline void mma_AB_overwrite_sync(register_tile_ &D_per_warp, a_smem_t_ const &A_full,
                                             b_smem_t_ const &B_full, mma_ctx_t<register_tile_> & /*unused*/) {
    using a_reg_t = kittens::rt_bf<register_tile_::rows, a_smem_t_::cols>;
    using b_reg_t = kittens::rt_bf<b_smem_t_::rows, b_smem_t_::cols, kittens::ducks::rt_layout::col>;
    a_reg_t A_reg;
    b_reg_t B_reg;
    kittens::warpgroup::load(A_reg, A_full);
    kittens::warp::load(B_reg, B_full);
    kittens::warp::zero(D_per_warp);
    kittens::warp::mma_AB(D_per_warp, A_reg, B_reg, D_per_warp);
}

/** D = A * Bᵀ (overwrite) — same per-warp pattern, transposed-B variant. */
template <typename register_tile_, typename a_smem_t_, typename b_smem_t_>
__device__ inline void mma_ABt_overwrite_sync(register_tile_ &D_per_warp, a_smem_t_ const &A_full,
                                              b_smem_t_ const &B_full, mma_ctx_t<register_tile_> & /*unused*/) {
    using a_reg_t = kittens::rt_bf<register_tile_::rows, a_smem_t_::cols>;
    using b_reg_t = kittens::rt_bf<b_smem_t_::rows, b_smem_t_::cols, kittens::ducks::rt_layout::row>;
    a_reg_t A_reg;
    b_reg_t B_reg;
    kittens::warpgroup::load(A_reg, A_full);
    kittens::warp::load(B_reg, B_full);
    kittens::warp::zero(D_per_warp);
    kittens::warp::mma_ABt(D_per_warp, A_reg, B_reg, D_per_warp);
}

/** D = Aᵀ * B (overwrite) — transposed-A variant.
 *
 *  A holds `[K, M]` in shared memory.  Aᵀ·B writes the M axis to D's
 *  rows, so distributing M across warps means each warp takes a COLUMN
 *  slice of A (cols `[warp_id*M_per_warp, (warp_id+1)*M_per_warp)`).
 *  TK's collaborative `warpgroup::load` distributes by ROWS only, so
 *  this case can't use it — each warp pulls its column slice via the
 *  shared-tile `subtile` API and a warp-scope load.  B (shape `[K, N]`)
 *  is needed in full by every warp, so each loads it independently.
 */
template <typename register_tile_, typename a_smem_t_, typename b_smem_t_>
__device__ inline void mma_AtB_overwrite_sync(register_tile_ &D_per_warp, a_smem_t_ const &A_full,
                                              b_smem_t_ const &B_full, mma_ctx_t<register_tile_> & /*unused*/) {
    using a_reg_t = kittens::rt_bf<a_smem_t_::rows, register_tile_::rows, kittens::ducks::rt_layout::col>;
    using b_reg_t = kittens::rt_bf<b_smem_t_::rows, b_smem_t_::cols, kittens::ducks::rt_layout::col>;
    a_reg_t A_reg;
    b_reg_t B_reg;
    int const warp_id_in_wg = kittens::warpgroup::warpid();
    auto const &A_subtile = const_cast<a_smem_t_ &>(A_full).template subtile<a_smem_t_::rows, register_tile_::rows>(
        {0, warp_id_in_wg * static_cast<int>(register_tile_::rows)});
    kittens::warp::load(A_reg, A_subtile);
    kittens::warp::load(B_reg, B_full);
    kittens::warp::zero(D_per_warp);
    kittens::warp::mma_AtB(D_per_warp, A_reg, B_reg, D_per_warp);
}

#else // KITTENS_HOPPER (or no MMA arch defined — fall back to the same Hopper path)

/** Empty per-kernel context — Hopper needs no allocator or semaphore. */
template <typename register_tile_>
struct mma_ctx_t {
    template <typename shared_alloc_t_>
    __device__ inline void init(shared_alloc_t_ & /*unused*/) {}
};

/**
 *  @brief D = A * B (overwrite) — Hopper warpgroup MMA + sync.  Direct
 *         translation of the call sites the MACE TK kernels were written
 *         against; the unused `ctx` parameter keeps the call signature
 *         portable across arches.
 */
template <typename register_tile_, typename a_smem_t_, typename b_smem_t_>
__device__ inline void mma_AB_overwrite_sync(register_tile_ &D_reg, a_smem_t_ const &A, b_smem_t_ const &B,
                                             mma_ctx_t<register_tile_> & /*unused*/) {
    kittens::warpgroup::mm_AB(D_reg, A, B);
    kittens::warpgroup::mma_async_wait();
}

/** D = A * Bᵀ (overwrite) — Hopper warpgroup MMA, B transposed. */
template <typename register_tile_, typename a_smem_t_, typename b_smem_t_>
__device__ inline void mma_ABt_overwrite_sync(register_tile_ &D_reg, a_smem_t_ const &A, b_smem_t_ const &B,
                                              mma_ctx_t<register_tile_> & /*unused*/) {
    kittens::warpgroup::mm_ABt(D_reg, A, B);
    kittens::warpgroup::mma_async_wait();
}

/** D = Aᵀ * B (overwrite) — Hopper warpgroup MMA, A transposed. */
template <typename register_tile_, typename a_smem_t_, typename b_smem_t_>
__device__ inline void mma_AtB_overwrite_sync(register_tile_ &D_reg, a_smem_t_ const &A, b_smem_t_ const &B,
                                              mma_ctx_t<register_tile_> & /*unused*/) {
    kittens::warpgroup::mm_AtB(D_reg, A, B);
    kittens::warpgroup::mma_async_wait();
}

#endif // KITTENS_BLACKWELL

} // namespace smallfield::cuda::mace_tk

#endif // __CUDACC__
