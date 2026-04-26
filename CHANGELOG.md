# SmallField Changelog

Landed work, newest first.
The road still ahead lives in [`ROADMAP.md`](ROADMAP.md); experiments that did not survive review live in the same file's "Tried and rejected" section.

Throughput numbers throughout this file are measured on the same 8x RTX PRO 6000 Blackwell / 2x Xeon 6776P host as [`BENCH.md`](BENCH.md), 10M USearch K=1 unless otherwise noted.
The CPU cumulative gain on that workload from the original 2026-04-19 baseline (20 283 mol/s) to the current head (23 735 mol/s) is __+16.9 %__.

## MACE-OFF Megakernel B α–β-3 — 2026-04-26 — Full Chain Landed

Megakernel B is the per-atom downstream-chain fusion sketched in the file header of `mace_tk_atom_chain_fused_kernel.cuh`.
Six landed phases plus the scaffolding committed in α, in increasing order of kernel-time impact:

| Phase | Commit       | What                                                          |
| :---: | :----------- | :------------------------------------------------------------ |
| α     | `9d1b53b`    | Scaffolding + design doc + no-op stub launcher                |
| β-1   | `1e78b30`    | Bf16-input element_residual + fold scale into path_weight    |
| β-2   | `683ed04`    | Pre-scale param on linear_post_tp; skip `launch_mace_scale` |
| __γ__ | `f3b6611`    | __TK scalar matmul (M = 64 atoms) for L=0 linears (linear_up + product_linear) — replaces matmul_rowmajor brute force__ |
| __δ__ | `3fca275`    | __TK backward scalar matmul (mm_ABt) — replaces fp32-ref `linear_per_irrep_irrep1_force_adjoint`__ |
| ε     | `9648bf3`    | Wire existing TK element_residual backward at active=1 for layer-N residual chain |
| β-3   | `814dbc2`    | Bf16-input linear_post_tp; skip cast bridge between conv_tp_scatter and linear |

Cumulative measurement at batch=2048 × n_atoms=30 + forces + capture:

| Stage             | ms/config | confs/sec | Δ vs prior | Δ vs multi-warp |
| :---------------- | --------: | --------: | ---------: | --------------: |
| Multi-warp (`15a04c2`)|   0.0473 |       891 |       —    |          —     |
| + β-1             |   0.0471 |       896 |     +0.5 % |        +0.5 %  |
| + β-2             |   0.0470 |       898 |     +0.2 % |        +0.8 %  |
| + γ               |   0.0448 |       939 |     +4.6 % |        +5.4 %  |
| + δ               |   0.0425 |       978 |     +4.2 % |        +9.8 %  |
| + ε               |   0.0424 |       980 |     +0.2 % |       +10.0 %  |
| __+ β-3__         |  __0.0421__ |   __~982__ |   +0.2 %   |     __+10.2 %__|

The two load-bearing commits are γ and δ (the TK L=0 linear forward and backward) — together about 9 % of the cumulative gain.
The cubic fp32 reference (`mace_matmul_rowmajor` + `linear_per_irrep_irrep1_force_adjoint`) that they replace was running at ~0.75 % of fp32 peak; a properly tiled WGMMA over `M = 64 atoms × K = c_in × N = c_out` reaches the expected 5-10× per-call speedup, even though each call only needs 1/16 of the m-rows (the L=0-only output) — the kernel reinterprets M as 64 atoms (not 4 atoms × 16 m-rows like the older `mace_tk_per_irrep_linear_kernel`), so every output row is real.

### Symmetric-Contraction Stage-1 Segmented Polynomial — Investigated, Not Viable In-Session

The roadmap's "mega-fusion / cueq parity" line names a Stage-1 segmented-polynomial decomposition as the path to closing the remaining gap.
Spent time on this in-session and concluded it isn't buildable within a single working session.
The blockers:

- The order-3 ACE contraction `B[a, c] = Σ_{i,j,k} W3[Z, i, j, k, c] · X[a, c, i] · X[a, c, j] · X[a, c, k]` has the channel axis `c` as the __outer__ axis of every operand — all three X factors and W3 index it.
  There is no shared K / N dimension that a WGMMA can amortise across; each (a, c) pair is an independent cubic form.
- The natural decomposition into a polynomial monomial vector `m[a, c, idx ∈ [0, 816)]` × coefficient `W_poly[Z, c, idx]` → `B[a, c]` is a per-(a, c) inner product.
  Tensor-core gains would require batching over an axis where the operand is shared — atoms share W3 only when they share `Z`, which means atom-binning by element first, which is multi-day infrastructure work.
- cueq's actual implementation uses an internal `SegmentedPolynomial` primitive in `cuequivariance`; we can't replicate it directly without porting that primitive's runtime (which has its own dependencies on cuTe / WGMMA layout boilerplate).

Short version: closing the cueq parity gap on symcontract requires either (a) atom-binning by element + per-Z batched WGMMA on grouped atoms, or (b) a port of the SegmentedPolynomial primitive.
Both are multi-week, not single-session.

### Where Time Goes Now — Post-β-3 NCU Breakdown

| Kernel                                          | %      | Bottleneck |
| ----------------------------------------------- | -----: | ---------- |
| `mace_tk_symmetric_contraction` (forward)       | 24.4 % | FMA pipe (compute-saturated) |
| `mace_tk_symmetric_contraction_force_adjoint`   | 14.8 % | TK multi-warp |
| `mace_tk_element_conditioned_residual_force_adjoint` | 11.3 % | TK |
| `mace_tk_conv_tp_scatter_force_adjoint`         |  8.9 % | 18 % barrier stalls |
| `mace_tk_element_conditioned_residual` (forward, bf16-in) | 7.6 % | TK |
| `mace_tk_conv_tp_scatter_segmented` (forward)   |  5.9 % | TK |
| `mace_tk_per_irrep_linear` (forward)            |  5.1 % | TK |
| `mace_tk_per_irrep_linear_force_adjoint`        |  4.6 % | TK |
| `mace_radial_mlp_layer_force_adjoint` (n_in=8)  |  3.0 % | fp32 ref — tractable but needs a custom WGMMA-padded kernel |
| `mace_inject_l0_irrep_block`                    |  2.6 % | bandwidth-bound utility |

The top 8 are all on TK already.
Remaining levers are:
- __Symcontract restructure__ (~39 % of total): multi-week algorithmic rewrite as above.
- __conv_tp_scatter barrier reduction__ (~9 %): the 18 % barrier stall on the backward is internal sync between the segment-reduction phases.
  Restructuring is multi-day.
- __n_in=8 radial_mlp backward TK port__ (3 %): the fp32 reference's one-thread-per-(row, c_in) launcher with `block_dim_x = 64` only populates 8 of 64 threads at n_in=8.
  A WGMMA port with N_dim padded to 16 (with zero-fill for rows 8-15) would still beat the brute-force reference even with half the FMA throughput wasted on zeros.
  Tractable as ~1 day of work.

End of in-session optimisation cycle.
Trajectory from session start (pre-multi-warp 724 c/s, 1.36 ms/relax) to end (982 c/s, 0.0421 ms/config kernel-only): __+35.6 % throughput__.
Gap to cueq's 0.0307 ms/config is now __1.37×__ (was 1.95× pre-session).

## MACE-OFF Saturation Sweep at Large Batch — 2026-04-26

After multi-warp landed and brought single-H100 throughput to ~890 c/s at the canonical batch=2048 × n_atoms=30 shape, the obvious follow-up was "do larger batches push higher".
They do not.

End-to-end refine_batch (20 LBFGS steps, MACE-OFF23-small, fp32 cast-bridge fallbacks, captured graph + Phase K + early-exit), single H100:

|         | bs=512 / 1024 | bs=2048 | bs=4096 | bs=8192 |
| ------: | :------------ | :------ | :------ | :------ |
| n_atoms=25 |       1068 c/s |  1076 c/s |  1078 c/s |  1081 c/s |
| n_atoms=30 |        884 c/s |   891 c/s |   892 c/s |   895 c/s |
| n_atoms=50 |        532 c/s |   533 c/s |   533 c/s |   (skip)  |
| n_atoms=80 |        329 c/s |   331 c/s |   332 c/s |   (skip)  |

Throughput is essentially flat across bs=1024 → bs=8192 (0.5–1 % gain across the entire range).
NCU at bs=8192 confirms the hot kernels (symcontract, conv_tp_scatter, element_residual) sit at 99 %+ SM activity — they're already at the SM ceiling.
The handful of small bookkeeping kernels (`fill_constant`, `per_graph_energy_*`, `edge_geometry`) remain sub-saturating (0.03–4 waves, 12–80 % SM activity) but contribute <1 % of total time at every batch size, so their saturation profile doesn't matter.

__The saturation knee is at ~25–80 K atoms total per batch.__
Once the GPU sees that many atoms, every kernel's work is fully spread across all 132 SMs and adding more atoms just stretches wall time proportionally.

__Practical batch-size guidance__:
- n_atoms ≤ 30 (median drug-like): bs=1024-2048 saturates; bs=4096 fits comfortably in 80 GB and gives a tiny edge.
- n_atoms 50: bs=1024 saturates; bs=2048 still fits; bs=4096 is the largest tested.
- n_atoms 80: bs=512-1024 saturates.
- bs=16384 × n_atoms=50 OOMs (allocates ~10 M edges of fp32 scratch before launch); bs=8192 × n_atoms=50 is the practical headroom limit.

The way to push past 891 c/s on a single H100 is __not__ larger batches (saturated) but kernel-level work (TK ports of the residual fp32 fallbacks, conv_tp_scatter barrier reduction, etc.) or multi-GPU fan-out (8× cards estimated 5.1× of one H100 on the production hardware).

## MACE-OFF Symcontract Multi-Warp CTAs — 2026-04-26 — `15a04c2`, Follow-Up Smem-W3 Prefetch Reverted

NCU's "73% local speedup from fixing warp-eligibility starvation" estimate on the symcontract backward (84% L1TEX scoreboard stalls, 0.31 eligible warps/scheduler with single-warp CTAs) translated almost 1:1 to wall-time:

| Change                                                  | Backward  | Forward   | ms/config | refine_batch |
| ------------------------------------------------------- | --------: | --------: | --------: | -----------: |
| Multi-warp baseline (1 warp/CTA, 32 threads)            | 19.69 ms  | 11.14 ms  | 0.0586    | 1.36 ms      |
| Multi-warp 96 threads (3 warps/CTA, 1 ch/thread)        |  7.19 ms  | 11.07 ms  | 0.0473    | 1.12 ms      |
| Δ                                                       | __-63.5 %__ | -0.6 %  | __-19.3 %__ | __-17.6 %__ |

Long-scoreboard stalls on the backward dropped 33.89 % → 7.33 %; the forward was never warp-eligibility-starved (its 13.05 % long_sb is from fp32-ALU-pipe latency, not memory).
Bit-identical to the single-warp baseline on all 5 reference goldens.

End-to-end refine_batch throughput at batch=2048 × n_atoms=30 × 20 LBFGS steps:  724 → __891 confs/sec/GPU__ = __+23 %__.

### Smem-Resident W3_partsym Tile — Investigated, Reverted

Followed the multi-warp commit with a per-(α, β) cooperative prefetch of the 16-element W3 m-row into a shared-memory tile, hoping to claw back the residual 7.33 % long_sb stalls on the backward.
The plan: each thread reads its 16 m-values from smem (zero latency) instead of issuing 16 strided global reads per (α, β) iteration.

__Result: 0.0473 → 0.0627 ms / config = +33 % regression.
Reverted.__

Why it loses: the 7.33 % residual stall budget was already small.
The prefetch added 136 `__syncthreads()` per CTA (one per (α, β) iteration) plus the cooperative-load instruction count.
Together those exceeded the L1-latency savings the smem reads bought.

The user-requested "move W3_partsym_sym fully into shared memory" plan also doesn't fit:
- Full slab per element: 16 m × 136 (α ≤ β) × 96 ch × 4 B = __786 KB per element__, ~5× the 100 KB default per-CTA smem and 3.5× the 228 KB opt-in maximum.
- Channel-partitioned variant (8 CTAs per atom × 12 ch each → 105 KB smem/CTA) would fit with `cudaFuncSetAttribute(MaxDynamicSharedMemSize)`, but smem-bound occupancy drops from 8 CTAs/SM (post-multi-warp) to 1 CTA/SM — the very issue NCU said multi-warp fixed.
- Per-row tile (the variant we benched) avoids the occupancy cliff but loses to sync overhead.

The post-multi-warp NCU shows 99.76 % SM activity on the backward at 46 % issue rate — the backward is now compute-throughput bound, not memory-stall-bound.
Smem caching helps when load latency dominates the issue queue; here multi-warp already filled the queue.

### Transform #1 Forward, Canonical-Triple Iteration — Investigated, Reverted

Followed the multi-warp commit with a Transform #1-style canonical-triple forward: iterate only the 816 canonical (i ≤ j ≤ k) cells against the already-precomputed `tk_precomputed_w3_sym_total_layer` slab, instead of the cubic 16³ = 4 096 ordered triples against `W3_eff`.
The W3_sym_total tensor was added in commit `895bfa4` precisely for this and noted in the CHANGELOG as "staged but not yet wired".

__Result: 0.0473 → 0.0473 ms/config — at best a wash on the bench, and the kernel itself ran 11.07 → 13.5 ms/call (slower).
Reverted.__

NCU on the canonical kernel:
| Metric                              | cubic forward | canonical forward |
| ----------------------------------- | ------------: | ----------------: |
| FMA pipe utilisation                |      18.87 %  |          6.45 %   |
| Long-scoreboard stalls              |      13.05 %  |         46.17 %   |
| Per-call duration                   |      11.07 ms |         13.50 ms  |

The variable inner-loop bounds (`j ≥ i`, `k ≥ j`) prevent the compiler from applying the fixed unroll factors that the cubic kernel relies on (`#pragma unroll 4` on i, `#pragma unroll 16` on j, full unroll on k).
The W3_sym_total reads also follow a non-uniform stride pattern that defeats the L1 prefetcher's linear-stride heuristic.
Net: the 5× FMA- count reduction is more than offset by 3.5× higher long-scoreboard stalls and a collapsed FMA pipe utilisation.

The original CHANGELOG note for Transform #1 ("does NOT help this forward kernel — it has only one fp32 accumulator, the dependency chain on `accumulator` was already broken into 4 chains by the i unroll, and the kernel runs at ~99% SM / ~7% FMA-pipe with the existing layout") was correct; this experiment confirmed it on the post-multi-warp baseline.
The forward symcontract on H100 is L1-throughput-bound on linear stride access, not FMA-count-bound.

`mace_tk_precompute_w3_sym_total` and the kernel + launcher remain available in-tree (commit `895bfa4`) for future Transform #2 work.

### What the Post-Phase-1 NCU Now Flags

Kernel time breakdown (288 ms total over 30 launches at batch=2048×30):

| Kernel                                        | Was   | Now   | %    | Bottleneck                      |
| --------------------------------------------- | ----: | ----: | ---: | ------------------------------- |
| `mace_tk_symmetric_contraction` (forward)     | 11.14 ms | 11.09 ms | __19.3 %__ | fp32 FMA pipe (compute) |
| `mace_tk_symmetric_contraction_force_adjoint` | 19.69 ms |  7.20 ms | 15.0 % | balanced (post-multi-warp) |
| `mace_tk_conv_tp_scatter_force_adjoint`       |  4.36 ms |  4.35 ms |  9.1 % | 18.5 % barrier stalls |
| `mace_tk_element_conditioned_residual_force_adjoint` | 8.58 ms | 8.55 ms |  8.9 % | 32 % FMA pipe |
| Phase-E TK-able fp32 fallbacks (4 kernels)    | 16.0 ms | 16.0 ms | ~14 % | mechanical TK port |

The new biggest hotspot is the symcontract __forward__, which the multi-warp change didn't move because it's compute-bound.
The forward's inner loop iterates over all 16³ = 4 096 (i, j, k) cells; the __X-commutativity Transform #1 W3_sym_total tensor__ is already precomputed and would let the forward iterate over canonical (i ≤ j ≤ k) only — 816 cells, __5× FMA reduction__.
The CHANGELOG note on Transform #1 noted W3_sym_total was "staged but not yet wired" — it is the next lever for the forward.

## MACE-OFF NCU Profiling — 2026-04-26 — TMA Pipelining Is Not the Right Next Lever

End-to-end refine_batch throughput on H100 SXM5 (one card, MACE-OFF23-small, fp32 reference fallback for the medium model's non-96-channel kernels — i.e. the post-`444c379` honest baseline).
Captured-graph + Phase K GPU-resident LBFGS + 20-step early-exit:

| n_atoms | batch=1024 | batch=2048 | batch=4096 |
| ------: | ---------: | ---------: | ---------: |
|      25 |    883 c/s |    895 c/s |    901 c/s |
|      50 |    443 c/s |    446 c/s |    447 c/s |
|      80 |    277 c/s |    278 c/s |    279 c/s |

Saturates at batch ≥ 1024.
At n_atoms=30 (drug-like median) and batch=2048, kernel-only forward+backward = __0.0586 ms / config__.

NCU 2025.2.1 hot-kernel breakdown (200-launch sample, captured graph replay) — share of __total kernel time__ at batch=2048×30:

| Kernel                                                    | Share | TC %  | DRAM % | Long-SB stall | Notes |
| --------------------------------------------------------- | ----: | ----: | -----: | ------------: | --------------------------------- |
| `mace_tk_symmetric_contraction_force_adjoint`             | __32.5 %__ |  0.0 % |  1.95 % |     __33.9 %__ | single-warp CTA, L1-bound |
| `mace_tk_symmetric_contraction` (fwd)                     | __15.3 %__ |  0.0 % |  1.69 % |       13.1 %  | single-warp CTA, L1-bound |
| `mace_tk_conv_tp_scatter_force_adjoint`                   |   7.2 % |  0.1 % |  9.00 % |        7.1 %  | 18.5 % barrier stalls |
| `mace_tk_element_conditioned_residual_force_adjoint`      |   7.1 % |  1.5 % |  3.92 % |        0.9 %  | 32 % FMA pipe |
| `mace_linear_per_irrep_irrep1_force_adjoint` (fp32 ref)   |   5.1 % | —     | —      | —             | TK-able (Phase E) |
| `mace_matmul_rowmajor` (fp32 ref)                          |   4.1 % | —     | —      | —             | TK-able (Phase E) |
| `mace_tk_per_irrep_linear_force_adjoint`                  |   3.7 % |  4.3 % |  7.59 % |        0.6 %  | 28 % FMA pipe |
| `mace_tk_element_conditioned_residual` (fwd)              |   3.6 % |  1.9 % |  2.58 % |        0.8 %  |       |
| `mace_tk_conv_tp_scatter_segmented` (fwd)                 |   3.5 % |  0.0 % | 11.30 % |        6.8 %  | 6.5 % barrier |
| `mace_tk_per_irrep_linear` (fwd)                          |   3.1 % |  4.3 % |  7.56 % |        0.5 %  |       |
| `mace_skip_tp_irrep1_force_adjoint` (fp32 ref)            |   2.3 % | —     | —      | —             | TK-able (Phase E) |
| `mace_radial_mlp_layer_force_adjoint` (fp32 ref)          |   2.3 % | —     | —      | —             | TK-able (Phase E) |
| `mace_bf16_to_fp32` (cast bridges)                        |   2.1 % | —     | —      | —             | bridges TK↔fp32 |
| `mace_tk_radial_mlp_layer_force_adjoint<64,384,0>`        |   1.3 % |  ~6 % | —      | —             |       |
| `mace_tk_radial_mlp` (fwd)                                |   1.3 % |  2.7 % | 10.21 % |        0.7 %  |       |
| Other ~25 kernels                                          |  ~5 %  |       |        |              |       |

### What TMA Can Yield

TMA pipelining (Phase A) overlaps `tma::load_async(W_{i+1})` with `WGMMA(W_i)`.
It addresses __DRAM→smem load latency__.
But across every kernel where TMA could even apply (the 5 WGMMA kernels at the bottom of the table — radial_mlp, per_irrep_linear×2, element_conditioned_residual×2):

- __DRAM utilisation: 1.7-11.3 %__ of peak.
  We are not bandwidth-bound on any kernel.
  HBM has substantial headroom.
- __Long-scoreboard stalls: 0.5-1.0 %__ on those WGMMA kernels.
  TMA pipelining can save at most that fraction of stall cycles per kernel.

Combined share of those WGMMA kernels: ~24 % of total kernel time.
Maximum theoretical TMA gain: 0.5-1.0 % of 24 % = __~0.15-0.25 % global speedup__.
Phase A is __not the right next lever__.

### What the NCU Data Actually Says

The 47.8 % time share owned by symcontract forward + backward is:
- __L1-throughput-bound__, not DRAM-bound (`L1/TEX Throughput = 44 %` on the forward; `Compute SM Throughput = 17 %` on the backward).
- __Warp-eligibility-starved__: the kernels use __one warp per CTA__ (block size = 32 threads, grid = n_atoms).
  Schedulers see 6.84 active warps and 0.31 eligible warps on average; 83 % of cycles have no eligible warp.
  NCU estimates __73 % local speedup__ is on the table for these two kernels alone.
- Stall composition: __84.5 % of warp-stall cycles are L1TEX scoreboard waits__ — the cubic loop's repeated reads of `W3_partsym_sym` are hitting L1 (which itself hits L2 at 87 %), but the single-warp CTA has no other warp to issue while one waits.

The two structural fixes the data points to:
1. __Multi-warp CTAs__ (4-8 warps/CTA) so scheduler can hide L1 stalls with other warps' independent FMAs.  ~3.7× theoretical kernel speedup per NCU's "73 % local speedup" estimate.
2. __Bring `W3_partsym_sym` fully into shared memory__ to convert the L1TEX waits into ~10× faster smem reads.

The TK-able fp32-reference fallbacks (`linear_per_irrep_irrep1`, `matmul_rowmajor`, `skip_tp_irrep1`, `radial_mlp_layer` = __13.7 % combined__) are an orthogonal Phase E lever — mechanical TK port, expected ~30 % reduction on that share = ~4 % global gain.
Smaller than the symcontract restructure but much smaller scope (1-2 days).

### Tooling Notes

NCU 2026.1.0 fails on driver 570 ("Cuda driver is not compatible"); 2025.2.1 works.
Source-line attribution requires `-lineinfo` at compile time and a separate run without `--capture` (cudaGraph replay isn't compatible with NCU kernel-replay).
PC-sampling (`smsp__pcsamp_sample_count`) was unavailable in this run — would identify the exact instruction line where the L1TEX waits land.
nsys timeline confirms no host idle gaps in the captured-graph chain (cudaGraphLaunch ≈ 14 µs/call, vs ~50 eager kernel launches saved).



The Phase K converged-flags infrastructure provided the per-conformer "this conformer is done" signal but `refine_batch` still ran every iteration to completion.
This change adds a single-block AND-reduction of the converged flags into a device byte, copied to pinned host memory every 4 LBFGS iterations.
When the result is 1 (all conformers converged) the loop terminates early — saving the remaining forward+backward+step rounds for that batch.

Why this is the right level: kernel-level per-CTA gating (the previous attempt) inflates register pressure and tanks occupancy.
All-converged batch-level gating skips work _between_ kernels at the loop driver — no register cost, no kernel surgery.
It only fires when the _whole batch_ is done, but production drug-like batches' tail iterations happen with most conformers already converged, so the check stops the slowest remaining conformer from dragging the entire batch.

| Scenario (batch=2048×30, 20 steps)         | ms / relax | Speedup |
| ------------------------------------------ | ---------: | ------: |
| Baseline (no early exit)                   |       1.37 |     1.0× |
| All immediately converged (fmax = ∞)       |       0.30 |     4.6× |
| Probe overhead when nothing converges      |     +<1 %  |       —  |

The 4-iter check cadence was chosen so the per-probe sync cost (~30-50 µs) stays under 1 % of the per-iter kernel cost (~70 µs at batch=2048).
At smaller batches the probe is also cheap; bench shows no measurable regression at batch ∈ {16, 64, 256}.

This is the "Option (2)" path from the per-conformer-gating CHANGELOG note from earlier today, applied at the loop level.
Kernel-level re-batching of _partially_ converged batches (the bigger optimisation where unconverged conformers get compacted into a smaller batch) is the next lever.

## MACE-OFF cudaGraph Caching for `refine_batch` — 2026-04-26 — Investigated, Reverted

Hypothesis: the per-iteration kernel launches inside `refine_batch` (one forward + one backward + one LBFGS step ≈ ~50 small kernel launches per LBFGS iteration × 20 iterations × N relaxations) might be launch-bound at small batch sizes, and a captured-once / replayed-many cudaGraph would amortise the launch cost away.

Implementation cached `cudaGraphExec_t fwd_bwd_graph` on `smallfield::mace::context_t::impl_t`, gated rebuild on `(n_atoms, n_edges, n_graphs)` shape match, replayed via `cudaGraphLaunch` per iter.
Capture+instantiate (~50-100 ms) was a one-time cost amortised over many relaxations.

__Result at the production batch (2048 conformers × 30 atoms): 1.38 ms / relaxation, statistically indistinguishable from the 1.36 ms eager baseline.
Net throughput change: 0 %.
Reverted.__

| Batch size | Eager (baseline) | Cached cudaGraph | Δ        |
| ---------- | ---------------- | ---------------- | -------- |
| 16         | 6.22 ms          | 6.22 ms          |  0.0 %   |
| 64         | 2.12 ms          | 2.12 ms          |  0.0 %   |
| 256        | 1.53 ms          | 1.53 ms          |  0.0 %   |
| 2048       | 1.36 ms          | 1.38 ms          | +1.5 %   |

Why the null result: at batch=2048×30=61 440 atoms each kernel does hundreds of microseconds of work — the eager launch overhead (~5-10 µs per launch × ~50 launches/iter × 20 iters = ~5-10 ms total over 2.83 s of wall time) is already <0.5 % of total time.
The cudaGraph replay path saves at most that fraction.
At batch=16 (the launch- bound regime where cudaGraph would matter most) Phase K already eliminated the per-iter host roundtrip — so what remains is kernel work, not launch overhead.
At every batch size we tested, launch overhead is no longer the bottleneck.

__Decision: cudaGraph caching is shelved.__
We can re-add it if the re-batching work below produces small enough rebuilt batches that launch overhead returns as a meaningful share.

## MACE-OFF Per-Conformer Kernel Gating — 2026-04-26 — Investigated, Reverted

Attempted to land per-conformer gating on the symcontract forward + backward kernels (the largest share of E+F time, ~62 % combined per the post-#1 NCU profile) so that converged conformers' atoms become no-ops on subsequent LBFGS iterations.
The plan was: add `(graph_of_atom, converged_flags)` as optional kernel parameters and a CTA-entry early-return on `converged_flags[graph_of_atom[atom_index]] != 0`.

__Result: -1.7× regression (1.36 → 1.83 ms / relaxation in the best-case all-converged synthetic test, 1.36 → 3.02 ms in the steady-state no-convergence case).
Reverted.__

Root cause via ptxas inspection: adding 2 pointer parameters and the early- return guard inflated the symcontract backward register usage from __48 → 72 registers__ even though the per-CTA latency cost of the new memory loads was ~60 cycles (negligible vs the kernel's 390 µs runtime).
The 50 % register bump dropped CTAs/SM from 42 → 27 — a ~36 % occupancy loss → the kernel runs in 1.6× more waves.
The compiler's register schedule across the body got materially worse despite the new code paths being tiny — likely because the constant-pointer-as-runtime-conditional in the no-convergence case prevents some of the same scheduling decisions the simpler kernel allowed.

Approaches that would actually work:
1. __Compile-time-templated kernels__ — separate `<bool gated>` instantiations (one with the early-return, one without).
   Caller picks at launch.
   Avoids forcing the compiler to handle both paths in a single function body.
2. __Re-batching unconverged conformers periodically__ (the simpler structural approach) — every K=10 LBFGS steps, gather only the unconverged conformers into a smaller batch and continue with no kernel changes.
   Saves all the work for converged conformers exactly, no kernel surgery.
   Handles its own correctness issues via clean batch-rebuild (no stale accumulator buffers to worry about).
3. __Conditional-graph nodes__ (CUDA 12.3+) wrapping a per-iteration check that re-launches a smaller batch.

Option (2) is the practical right answer for this code base — the existing batch-staging plumbing in `cli/smallfield_mace_refine.cpp` already does shard-by-shard rebuilds and can be extended to do iter-by-iter rebuilds within a single relaxation.
Option (1) is feasible if we want to keep the single-batch model but is a substantial refactor of every gated kernel.

This finding is now in the kernel comments as a TODO note alongside the Phase K converged-flags infrastructure (commit `5183ebb`), which still provides the per-conformer fmax + flag tracking that re-batching would read at K-step boundaries.

## MACE-OFF Phase K — 2026-04-26 — `5183ebb`

GPU-resident LBFGS / GD-with-early-exit, eliminating the host-side relaxation loop in `smallfield::mace::context_t::refine_batch` (previously 20 host launches per relaxation, ~25 % of end-to-end wall time at the current kernel speed).

New kernel `mace_lbfgs_step_kernel`:
- One CTA per conformer, __single warp__ — a deliberate departure from the atom/edge-tiled threading shape of the MACE forward+backward chain.
  The forward+backward needs WGMMA tiles spanning many conformers (`mace_tk_per_irrep_linear` is 4 atoms × 96 channels per CTA = mixes conformers; `mace_tk_radial_mlp` is 64 edges per CTA, etc.) to fill the WGMMA tile and keep SM utilisation at ~99 %.
  The LBFGS update is embarrassingly parallel across conformers and the per-conformer fmax reduction is a natural 32-thread warp-shuffle — that's the natural threading shape for THIS kernel even though it doesn't fit the rest of the chain.
- Per-conformer fmax via `__shfl_xor_sync` butterfly.
- Adaptive early exit: when `fmax ≤ config.fmax_eV_per_A` the kernel sets a converged flag and skips position writeback on subsequent iterations, pinning the conformer at its converged geometry.

`mace_scratch_t` gains `lbfgs_graph_atom_offsets`, `lbfgs_fmax_per_conformer`, `lbfgs_converged_flags` (sized for max_graphs).

The forward + backward kernels still re-run for converged conformers (per-kernel gating against `converged_flags` would require touching every TK kernel — substantial; deferred).
Net wins this commit:
- Eliminate D2H/H2D ping-pong on positions across the N steps.
- Eliminate one host-side kernel launch (the per-element GD step) per iteration → N-1 host launches saved.
- Per-conformer convergence detection so callers can rebatch only unconverged conformers after a fixed K iterations.

Bench: forward+backward path unchanged at 0.0593 ms/config (the bench times the captured fwd+bwd graph, not refine_batch).
The end-to-end LBFGS-relaxation gain lands at `refine_batch` callers; the standalone CLI `smallfield_mace_refine` and the Python `MaceContext.refine` pick this up automatically.

Closes Phase K of the optimisation roadmap.

## MACE-OFF Algebraic Transforms — 2026-04-26

A planned sequence of algebraic transformations that move work onto tensor cores or reduce FMA count via exact identities.
Findings from the experimental campaign:

| Transform | Status | Realised gain |
|-----------|:------:|--------------:|
| #1 X-commutativity partsym fold (commit `895bfa4`) | __landed__ | -1.3 % ms/config |
| #2 Two-stage decomposition with Stage 1 on TC | not pursued | NCU shows post-#1 kernel is memory-latency-bound (7 % FMA pipe, 99.97 % SM); TC compute-density win lands on a small share of remaining time.  Substantial implementation effort (new TK WGMMA kernel + Stage 2 quadratic-form launcher) with unclear net win at the current bottleneck mix. |
| #3 bf16x2 packed FFMA on Stage 2 | not pursued | precision risk on the order-3 accumulator (already at the fp32 noise floor at 10–15 meV/Å); needs a golden regression suite to gate, which would also surface the medium-model latent issue noted below before any precision-trade landed. |
| #4 Extend TK kernels to 224 channels | not pursued | Refactor across `mace_tk_per_irrep_linear_*`, `mace_tk_element_conditioned_residual_*`, `mace_tk_readout_*` to template the channel width, then dispatch by `arch.n_channels`.  Mechanical but extensive (~5–7 days).  Worth landing only if the production decision picks the OFF23-large model. |

__Latent issue surfaced and fixed during this campaign__ (commits `40c0937` and `444c379`): six TK launchers were silently `return`-ing on `n_channels != 96` instead of signalling failure to the caller.
The bf16 output buffers kept their prior content and the chain produced numerically meaningless energies under the TK build for MACE-OFF-medium (128 ch) and MACE-OFF-large (224 ch).
All six are now `bool`-returning and the inference driver falls back to the fp32 reference path on channel-count mismatch.
Affected:

- `launch_mace_tk_per_irrep_linear` (forward)
- `launch_mace_tk_per_irrep_linear_force_adjoint` (backward)
- `launch_mace_tk_element_conditioned_residual` (forward)
- `launch_mace_tk_element_conditioned_residual_force_adjoint` (backward)
- `launch_mace_tk_readout_linear`
- `launch_mace_tk_readout_nonlinear_mlp`
- `launch_mace_tk_radial_mlp` (forward)

Pre-fix medium bench numbers in earlier reports never measured a working pipeline — the "0.0807 ms/config" headline was running on stale buffers across every TK kernel.
Post-fix medium runs correctly at 0.6097 ms/config; closing that perf regression is Transform #4 (templating each TK kernel over channel width).
Small model (96 ch) is unaffected — all TK paths still run, the bool-return wiring is no-op for it.

### Transform #1 — X-Commutativity Partsym Fold, Commit `895bfa4`

Backward kernel rewritten from a cubic-write structure (`dx[i]+=W3·X[j]·X[k]` plus two siblings) into a quadratic form whose two input axes are pre-symmetrised at engine init: ∂B/∂X[m] = Σ_{α≤β} W3_partsym_sym[m, α, β, c] · X[α]·X[β] where W3_partsym_sym absorbs the 3-permutation partsym sum AND the (α, β) symmetry of X·X.
Inner loop now does 16 INDEPENDENT FMAs per (α, β) iteration into 16 dx_local accumulators, vs the old 3 FMAs.

Modest wall-time gain (kernel was already near its memory ceiling) but exact identity, no precision risk.
Lays the algebraic foundation for Transform #2 — the partsym tensor is what Stage 1 of the two-step matmul decomposition consumes.

| metric            |  prev   |  after  | delta  | cumulative vs B6 |
| :---------------- | ------: | ------: | -----: | ---------------: |
| ms/config (E+F)   | 0.0601  | 0.0593  | -1.3 % |          -78.2 % |
| ms/iter           |  123.2  |  121.5  | -1.4 % |          -78.2 % |

Force RMS on the 5 reference goldens unchanged from the pre-transform binary (1.94 / 1.34 / 1.26 / 1.08 / 1.76 meV/Å).

Also lands `mace_tk_precompute_w3_sym_total` (the canonical-triple W3 fold for the FORWARD) — staged but not yet wired, since the forward kernel's single accumulator is already saturated.
Will be consumed by Transform #2.

## MACE-OFF Refinement Scaffolding — 2026-04-25 — `e05fcc7`, `6bf94a0`

End-to-end MACE-OFF refinement now has a stable C++ / Python / CLI surface alongside the existing ETKDG + MMFF94 path.

- __Python binding__ (`python/smallfield/_binding.cpp`, `api.py`): `MaceArch`, `MaceWeights.load(path)`, `MaceContext` with `infer_energy`, `infer_energy_and_forces`, and `refine`.
  High-level wrapper `sf.refine_with_mace(coords, smiles_list, weights_path)` derives elements via RDKit and builds edges from `arch.r_max`.
  All numpy I/O; H2D + D2H copies handled inside the binding.
- __Standalone CLI__ `smallfield_mace_refine` (`cli/smallfield_mace_refine.cpp`, `cli/parquet_3d_reader.{hpp,cpp}`): Reads SmallField output parquet shards (smiles + 3D coords + num_atoms), refines per-conformer via `mace::context_t::refine_batch`, writes refined parquet shards.
  Decoupled from `smallfield_batch` so refinement can scale independently of generation.
  Single-GPU today; multi-GPU is a thread-per- device fan-out plus the existing `shard_inventory` queue.
- __Doc cleanup__: retired `MACE.md`, `MACE_INFERENCE.md`, `MACE_INTEGRATION_PLAN.md`, `MACE_OPTIMIZATION_ROADMAP.md`, `MACE_BENCH_REPORT.md` — planning + measurement docs whose load-bearing content has been absorbed into `README.md`, `BENCH.md`, `GLOSSARY.md`, `ROADMAP.md`, and the entries below.

## MACE-OFF Kernel Optimisation Pass — 2026-04-25

Kernel-only ms/config at the saturating shape (batch=2048, n_atoms=30, energy+forces, captured-graph replay, MACE-OFF23-small, H100 SXM5): __0.272 → 0.0600__ = __4.5× speedup__, closing the with-forces gap to cueq small from 9.83× → 1.95×.
At this rate one H100 relaxes ~835 confs/sec end-to-end LBFGS @ 20 steps.

Cumulative progression (each row = one commit):

| Step                                                                                     | ms/config |     Δ |
| :--------------------------------------------------------------------------------------- | --------: | ----: |
| Initial fp32 reference adjoint chain                                                     |     2.103 |    —  |
| TK D2/D3/D4 force-adjoint wiring (`012de17`)                                             |     0.334 | -84 % |
| Chain numerical fixes — `apply_cutoff` + silu·n2m inversion (`7fc5b11`, `3b887ac`)       |     0.305 |  -9 % |
| __B5__ TK D1 per-irrep linear adjoint (`5087cf0`)                                        |     0.293 |  -4 % |
| __B6__ bf16 radial-MLP hidden saves, skip backward rerun (`90595a7`)                     |     0.272 |  -7 % |
| Drop over-unroll on symcontract adjoint — eliminate 30 KB/thread spill (`a4a75e9`)       |     0.205 | -25 % |
| `irrep_dim=1` fast path: linear-per-irrep fwd + bwd (`27177f6`)                          |     0.178 | -13 % |
| ILP unroll i×4/j×16 on symcontract adjoint (`d494800`)                                   |     0.112 | -37 % |
| `irrep_dim=1` fast path: skip-tp fwd + bwd (`a89ae08`)                                   |     0.106 |  -6 % |
| ILP unroll on symcontract forward (`e347639`)                                            |     0.101 |  -5 % |
| __E1__ TK-WGMMA radial-MLP layer backward (`2112b9d`)                                    |     0.065 | -36 % |
| Skip dead fp32 reference radial-MLP fwd under TK build (`b1e0781`)                       |     0.060 |  -8 % |
| Extend TK layer-3 backward to MACE-OFF-medium / -large (`8b26c49`)                       |     0.060 |    —  |

Top kernels by NCU-measured share of E+F time, post-pass:

| Kernel                                                | %    |
| ----------------------------------------------------- | ---: |
| `mace_tk_symmetric_contraction_force_adjoint_kernel`  | 46 % |
| `mace_tk_symmetric_contraction_kernel` (forward)      | 16 % |
| `mace_tk_element_conditioned_residual_force_adjoint`  | 11 % |
| `mace_tk_conv_tp_scatter_force_adjoint`               |  9 % |
| Everything else (~30 kernels)                         | 18 % |

__F1 not pursued__: the projected savings (50% off E+F) were premised on the symcontract backward being 140 M NCU units (28 % of E+F).
After `a4a75e9` + `d494800` it dropped to 39 M units, capturing those savings via different mechanism.
True fwd+bwd fusion is also structurally infeasible — B (forward output) and the upstream gradient (backward input) are needed at different points in the execution chain, so a fused kernel would have to run twice with different input modes.

Saturating-batch sweep at this commit (energy + forces, captured replay):

| Model      | n_atoms=15 | n_atoms=30  | n_atoms=50 | n_atoms=80 |
| :--------- | ---------: | ----------: | ---------: | ---------: |
| small (96 ch)  |  0.0299  |  __0.0599__ |   0.1005   |   0.1627   |
| medium (128 ch)|  0.0386  |  __0.0781__ |   0.1322   |   0.2143   |

Energy-only saturates at batch ≈ 256 (~0.024 ms/config small, ~0.032 medium); energy+forces saturates at batch ≈ 1024–2048 (slight slowdown past 4096 from L2 pressure on the W3_eff slab).

## MACE-OFF Native CUDA Module Bring-Up — 2026-04-25 — `b50d186` … `90595a7`

Initial native MACE-OFF inference module covering the full forward + analytic- adjoint chain.
Exists because the cueq + PyTorch reference path is host-pipeline-bound (only ~14.6 % of production wall time is the actual GPU model — the rest is `AtomicData.from_config` + DataLoader collate); a fully GPU-resident path can win end-to-end even at slower per-kernel speed.

- __Phase 0__ — weights export from `mace_off` HuggingFace checkpoint to a flat `.bin` consumable from C++ (`python/smallfield/mace_export.py`).
- __Forward path, fp32 reference__ — implements every block of MACE-OFF: radial basis + Bessel + cutoff, spherical harmonics, conv_tp + scatter, per-irrep linear, element-conditioned residual, symmetric contraction (orders 1, 2, 3), readouts, energy reduce.
  Validated against `torch.autograd` golden traces on 5 reference molecules.
- __TK forward path__ — every heavy kernel ported to ThunderKittens / WGMMA (radial-MLP fused 4-layer, conv_tp + scatter segmented, element-conditioned residual, per-irrep linear, symmetric contraction, both readouts).
  Energy-only, batch 256, n_atoms=30: TK build is __22× faster__ than the fp32 reference build.
- __Force-adjoint chain (D1–D7)__ — wired in `mace_inference_run_force_adjoint`; forward + backward run inside one `cudaGraphLaunch` via `mace_capture_forward_with_force_adjoint`.

Force-adjoint validation, per-atom force RMS vs `-torch.autograd.grad` on the 5 reference goldens:

| Molecule    | n_atoms | Initial force RMS | After validation pass | Reduction |
| ----------- | ------: | ----------------: | --------------------: | --------: |
| caffeine    |      24 |          1.96 eV/Å |            0.0148 eV/Å |     132×  |
| paracetamol |      20 |          ~2.16    |            0.0144     |     ~150× |
| ibuprofen   |      33 |          ~1.48    |            0.0102     |     ~145× |
| propranolol |      40 |          ~2.01    |            0.0083     |     ~240× |
| aspirin     |      21 |          ~2.32    |            0.0140     |     ~165× |

Principal bugs that landed in this phase (each with a 5–80× force-RMS reduction on its own):

1. `b50d186` — symmetric-contraction-backward weight argument-order swap (was passing `(weights_max, weights_lo_1, weights_lo_0)` instead of `(weights_max, weights_lo_0, weights_lo_1)`).
2. `012de17` — removed the duplicate `apply_cutoff` multiply on tp_weights (MACE-OFF's `RadialEmbeddingBlock` returns `radial * cutoff` already, so the second multiply double-counted `env(r)`).
3. `3b887ac` — per-layer save of `tensor_product_weights` (was being overwritten by the next forward layer), per-layer rerun of the radial MLP forward, and several missing `cudaMemsetAsync(0)` before adjoint kernels with `+=` semantics.
4. `7fc5b11` — `silu·n2m(x) = N · x · sigmoid(x)` is __non-monotonic__ (min at x ≈ -1.278), so the Newton inversion `silu_n2m^{-1}(y)` could pick the wrong branch and flip the chain-rule derivative sign (caffeine's `mlp_h2` lands in the ambiguous region [-0.45, -0.467]).
   Fix: save pre-activations directly during the backward forward-rerun.

Residual ~10–15 meV/Å gap to the 1 meV/Å plan target sits at the fp32 noise floor of 4 chained radial-MLP layers; reaching ≤ 1 meV/Å in fp32 likely needs an fp64 accumulator path.
The chain is production-usable as-is — 0.01 eV/Å is well inside the convergence tolerance of every common LBFGS / FIRE optimiser (`fmax = 0.05 eV/Å`).

- __Phase H__ — public C++ API `smallfield::mace::context_t` (commit `11f691b`): RAII `weights_t::load(path)` + persistent context with `infer_energy`, `infer_energy_and_forces`, `refine_batch`.

## RDKit 2025.09.6 Bump, Per-Molecule Preprocess Deadline and `preprocess_timeout` Status — Pending

Under the multi-`parquet-shard` invocation wrapper, a single pathological drug-like `molecule` could stall the entire 4-role pipeline for 15–20 minutes.
`cuda-gdb` on the stuck thread placed the hang inside `boost::detail::VF2SubState::MatchAll`, reachable from `RDKit::SubstructMatch` on the CSD-torsion fallback path.
RDKit 2024.03.6's `SubstructMatchParameters` exposed only `extraFinalCheck` — fires after a candidate match is found, useless for searches that never find one.

- __RDKit upgraded__ 2024.03.6 → 2025.09.6 (conda-forge `librdkit-dev` / `libboost-devel=1.86`), installed into a dedicated conda-forge environment.
  Build wiring unchanged — the `RDKIT_PREFIX` hand-rolled imported-target mechanism in `CMakeLists.txt:96-148` takes the new env as-is.
  Two test translation units needed an explicit `#include <iostream>` because 2025.09.6's `ForceFieldHelpers/MMFF/AtomTyper.h` references `std::cout` without the backing include.
- __Per-`molecule` soft deadline__.
  New header `src/smallfield/preprocess/preprocess_deadline.hpp` defines `preprocess_deadline_t { deadline, tripped }` — a stack-owned struct, no `thread_local` or process-wide state.
  Each OMP-for iteration in `prepare_per_thread_contribs_` stack-allocates one, calls `reset(2 s)` before each `molecule`, and passes a pointer through `prepare_molecule_into` → `extract_etk_torsions_native` → the `extraAtomCheck` lambda in `SubstructMatchParameters` (new in RDKit 2025.09.6).
  The lambda returns `!exceeded_or_null(deadline)` on every VF2 atom comparison; once past the deadline, VF2 unwinds in O(candidates) without throwing through RDKit internals.
  No singleton, no pimpl, no signal — a pointer threaded through 3 function signatures.
- __New `conformer_status_t::preprocess_timeout = 5`__.
  Stamped in `generate_conformers` for every conformer slot whose `molecule` was recorded in the new `timed_out_input_indices` list (on `batch_prep_thread_contribution_t` and on `batch_pool_t`).
  Grow-only vectors across calls — zero hot-path allocation after warmup.
  The `fallback-role` picks up timed-out molecules on its existing "any non-success status" retry path and re-embeds them via ETKDGv3, which does not use `SubstructMatch` and so cannot re-trigger the same VF2 worst case.
- __Adversarial axis `vf2_hang_candidates`__.
  Twenty polycyclic-aromatic / heavily-halogenated templates appended to `test/preprocess/adversarial_molecules.py` as a regression gate for the `extraAtomCheck` hook.
- __Two pybind enum bindings__ (`src/smallfield/api/smallfield.cu:1018`, `python/smallfield/_binding.cpp:27`) gain `.value("preprocess_timeout", ...)`.
- __New `num_atoms_out` output span on `generate_conformers`__.
  Library now copies each `molecule`'s atom count (heavy + explicit H, post-`addHs`) from preprocess metadata into a caller-provided `std::span<std::uint8_t>`.
  Default-empty span keeps the old API valid; the CLI passes it through to the writer to enable tight per-row coordinate emission.

All 13 ctest targets green on the 2025.09.6 build.
Validated by encoding all 978 GDB13 shards + 4000+ Enamine REAL shards on the new build with zero correctness issues.

## Tight Per-Row Coordinate Parquet Output, Zstd Compression and Auto-Park CUDA-Error Guardrail — Pending

Three independent CLI / writer changes that together drop output-shard size ~3× and make multi-shard runs self-recovering on a poisoned CUDA context:

- __Tight `List<float16>` coordinates per row__.
  `pack_coordinates_as_list` in `cli/parquet_writer.cpp` replaces the old `FixedSizeList<float16, max_atoms × 3>` with a variable-length `List<float16>` whose row length is exactly `num_atoms × 3` halves.
  Drug-like Enamine REAL averages ~50 atoms per molecule against a 128-atom kernel stride, so 60 % of every row was trailing zero padding; tight rows cut the coordinate column by ~60 %.
- __ZSTD-3 compression on every column__.
  Parquet writer now passes `parquet::Compression::ZSTD` with level 3 (`cli/parquet_writer.cpp`).
  Replaces the silent default of `UNCOMPRESSED` that had been inflating outputs ~2-3 ×; combined with the tight-coords fix, full-Enamine output projects from ~11 TB → ~2.7 TB.
- __Always-populated `num_atoms` column__.
  CLI now passes `buf->num_atoms_per_molecule.data()` into `generate_conformers`'s new `num_atoms_out` parameter (instead of zero-filling and only setting the column for fallback-rescued slots).
  Writer treats the column as non-null going forward.
  Old shards converted by a one-shot Python tool — `/tmp/convert_old_shards.py` regenerates `num_atoms` from re-parsed SMILES via RDKit.
- __CUDA-error auto-park + abort__.
  `gpu_runner_loop` in `cli/smallfield_batch.cpp` now checks `gpu_context.cuda_error_count` after every `generate_conformers` call.
  On any CUDA error during a batch, the contributing input shards are atomically `mv`'d to `{input_dir}_skip/` and the process exits with status 2.
  The wrapper-loop's `for attempt in $(seq 1 999)` catches the non-zero exit, sleeps 8 s, and relaunches with a fresh CUDA context — the now-skipped shard never re-poisons the new invocation.
  Replaces the prior behaviour where one bad-data CUDA-OOB shard cascaded into thousands of garbage-output shards before the wrapper finally noticed.

Per-shard size win measured on Enamine REAL shard 0 (3 M conformer rows): __1785 MB → 225 MB (12.6 %)__ via the bulk Python converter; __~656 MB__ for fresh-encoded drug-like shards including the writer-side compression.
Combined the converter and the new writer reclaimed ~1.9 TB on the running disk while encoding continued without interruption.

## Complexity-Aware Stripe and Partitioned CPU Budget — Pending

Two changes target throughput on drug-like `coprocessed-batches` where the dominant pathologies are the per-device wave straggler and concurrent-role CPU over-subscription.

- __Complexity-descending stripe__.
  `distribute_batch_to_per_device_pools_` now sorts molecules by `molecule_complexity_score_` (atom-count² + 4·torsion-count + per-term cost, doubled on heavy-rescue cases — see `src/smallfield/preprocess/molecule_complexity.hpp`) descending and stripes sorted rank `k` onto device `k % device_count`.
  Replaces the old `num_atoms`-only sort.
  The sort permutation and per-device assignment are packed into a single CSR layout (one flat `std::vector<std::uint32_t>` + one `std::array<std::uint32_t, max_devices_k + 1>` offset array), both held on a new caller-owned `distribute_scratch_t` sibling of the existing `static etkdg_molecule_batch_t shared_scratch`.
  No vector-of-vectors, no per-call allocation on the distribute hot path; the previous code built a `std::vector<std::uint32_t>` the size of the whole batch plus a 32-wide vector-of-vectors every `embedding-wave`.

- __Multi-`parquet-shard` invocation__.
  The per-`parquet-shard` wrapper (`/tmp/pubchem_per_shard_loop.sh`) was replaced with a multi-`parquet-shard` wrapper (`/tmp/pubchem_multi_shard_loop.sh`) that points `smallfield_batch` at the entire input directory in one invocation.
  `--resume` (CLI default) auto-skips `parquet-shards` already produced, so crash-retry picks up cleanly.
  The 4-role ring (reader → gpu-runner → fallback → writer) only overlaps when multiple `coprocessed-batches` are in flight, which per-shard invocation never provided (1 M ÷ 800 K ≈ 1.25 `coprocessed-batches` per process).
  An earlier attempt to partition the `compute_thread_budget` 5 : 3 between `fallback-role` and `gpu-runner-role` was reverted — it reduced single-shard throughput ~5 % in the per-shard-invocation mode, and the overlap window it depended on never existed.
  Both roles default again to `num_procs - 4`; new flags `--gpu-runner-threads N` and `--rdkit-fallback-threads N` stay for manual tuning.

Baseline measurements pre-change (PubChem shard 8, 8 GPUs, K=3, seed 42): `351 s wall, 2 845 mol/s, 0 CUDA errors`.
Post-stripe-only single-shard: `349 s wall, 2 867 mol/s, 0 CUDA errors` — within noise.
Post-multi-shard-invocation throughput: pending measurement on full PubChem tail.

## 3-Stream Chain Reverted — `db25c9d`

The 3-stream chain from `a791e54` was kept as an architectural improvement assuming the copy-engine overlap would show at production scale.
Measured under PubChem drug-like K=3 it lost on both axes:

- __Throughput.__
  Shard 47 ran 17 % slower than the single-stream shard-46 baseline at the same input shape and fallback rate (387 s vs 331 s, 2 581 mol/s vs 3 019 mol/s).
  On 10M USearch K=3 the single-stream build runs 587.9 s (17 007 mol/s / 51 022 conformers/s) vs 3-stream's 661.9 s — a 13 % single-stream advantage on a workload where 3-stream had been claimed to help.
  The overlap gain is a loss because kernel phases are short enough that the extra `cudaEvent` hand-off overhead dominates the saved copy/compute concurrency.
- __Stability.__
  A second `illegal memory access` race surfaced on the 3-stream build at `cudaEventSynchronize(finalize)` on PubChem shard 50, +1:26 into the wave.
  The per-wave `cudaFuncSetAttribute` fix (`1b786f1`) closed the root cause; the 3-stream chain re-opens a separate race on the event hand-off that was not pinned down.

Reverted `a791e54`'s `smallfield.cu` + `smallfield.hpp` changes (100 lines, 2 files).
Per-device pinned pools, ctor pre-allocation, and the launch-config cache from the same refactor survive unchanged.

## Multi-GPU Illegal-Memory-Access Fix — `1b786f1`

Dropped a per-wave `cudaFuncSetAttribute(embed_4d_power_iter_kernel, MaxDynSM, 67 KB)` from `build_metric_initial_coordinates_on_gpu` that was reached concurrently from every OMP per-device worker thread inside `launch_wave`.
The call mutates driver-global per-kernel attribute state on a shared kernel symbol; at ≥ 2 GPUs the 8 worker threads raced on that state, producing a window during which `embed_4d_power_iter_kernel` launched with the 48 KB hardware-default dynamic shared instead of the 67 KB the kernel needs at bin 128.
The out-of-bounds shared-memory access surfaced on a later kernel launch as `an illegal memory access was encountered`.
The attribute is already set once per device in `context_t::context_t`; the per-wave re-assert was defensive overkill left over from `9cb82be`, where the ctor hoist landed but the per-wave copy was not removed.

Symptom matching:

- Fired only at ≥ 2 GPUs — a single OMP thread cannot race with itself.
- Timing-random 30–120 s into a run — a small torn-state window that depends on thread interleaving, not on molecule shape.
- `CUDA_LAUNCH_BLOCKING=1` masked it — blocking launches serialize the attribute update far enough that no kernel launches within the torn window.
- `compute-sanitizer memcheck` never reproduced — the instrumentation overhead widens each `setAttribute`'s critical section enough to close the race.

Verification — 10 M USearch K=3, 8 GPUs, post-fix:

| Variant                          |    Wall |  mol/s | conformers/s |           Crashes |
| :------------------------------- | ------: | -----: | -----------: | ----------------: |
| Pre-fix non-blocking             |       — |      — |            — | crashed @ +9:49 s |
| Pre-fix `CUDA_LAUNCH_BLOCKING=1` | 1 063 s |  9 404 |       28 211 |                 0 |
| Post-fix non-blocking            |   662 s | 15 106 |       45 319 |                 0 |

## Multi-GPU Pipeline Refactor — `a791e54` and `9cb82be`

Four architectural changes to the per-wave GPU dispatch path; together they land the host-scheduling items from the roadmap without altering any kernel code.

- __Per-device pinned pools.__
  `prepare_per_device_batches_into` produces one pinned batch pool per device (atom-count-descending stripe partition, each device's mols ending in ascending-atom-count order for the kernel bin dispatch).
  Each GPU's `cudaMemcpyAsync H2D` reads from a buffer only that GPU touches.
  Replaces the shared wave-wide pinned source whose 8-way concurrent DMA was the prime suspect behind the illegal-memory-access crash until the root cause was pinned down as the `cudaFuncSetAttribute` race above.
- __3-stream chain per device.__
  H2D / compute / D2H run on three streams chained by `cudaEvent` hand-offs so Blackwell's two copy engines can overlap H2D of wave N+1 with compute of wave N and D2H of wave N-1.
- __Ctor pre-allocation.__
  `context_t::context_t` pre-grows every per-device pool (`launch_scratch.device_pool`, `launch_scratch.host_staging`, `metric_scratch.pool`, `batch_slots_per_device`) to worst case using `cudaMemGetInfo` plus the cusolver workspace cap.
  Hot-path `cudaMalloc` / `cudaHostAlloc` never fire.
- __Launch-config cache.__
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` probes run once per device at ctor (128 probes, ≈ 13 ms) and cache into a `device_launch_config_t` table.
  Per-bin dispatch in `launch_gpu_pipeline` reads the table — no host API call on the hot path.

Drive-by cleanup: dropped the dead `context_t::batch_ring` field, the unreferenced `run_gpu_pipeline` backward-compat wrapper, and legacy PascalCase constants `kNoCharge` / `kMaxIters` / `kNumShards` (now `no_charge_k` / `max_iterations_k` / `scaffold_cache_shard_count_k`).

Scope: 2 files in `src/smallfield/api/`, ≈ 100 lines of pipeline wiring; preprocess adds the new per-device distribution helper; three renames in the preprocess headers.
No kernel code changed.

## CSD Torsion Round 2 — `bc4e1af`

Native pattern coverage 391/470 → __408/470__ (83.2 % → 86.8 %); throughput 23 371 → 23 735 mol/s (+1.55 % over round 1, +16.9 % cumulative).
The diminishing-returns shape is expected: round 1 already removed the biggest VF2 hot-spots, leaving non-linear topology patterns that resist a strictly linear matcher.

Three additive matcher extensions in `csd_torsion_native.hpp` plus four new recursive predicates and a hybridization-based per-molecule prefilter:

- __`R<n>` ring-count constraint__ — an `atom_disjunct_t::ring_count` field matched via `__builtin_popcount(desc.atom_ring_bitmap[atom_idx])`, reusing the descriptor's existing 32-ring bitmap.
  `R2,R3` decomposes through the existing DNF parser into two disjuncts. +5 patterns.
- __Chain-in-branch via predicate lookup__ — bodies like `(cO)`, `(C=O)`, `([cH0]Cl)`, `([cH0]F)` lift to an already-compiled recursive predicate via a new `lookup_chain_branch` table instead of growing a generic 2-hop matcher.
  Adds a `predicate` field on `simple_neighbor_constraint_t` checked after the atom-disjunct pass. +8 patterns.
- __Four new recursive predicate bodies__ — `[cH0](=O)`, `c!@c`, `[n,N](-a)`, `[a]([nH0,o])`.
  Standard one-hop walks of element + aromaticity + bond-type constraints. +4 patterns.

The hybridization prefilter precomputes a per-molecule `molecule_hybridizations` mask alongside the existing `molecule_elements` mask; the per-pattern requirement is the intersection of disjuncts per atom then union across the four labelled atoms (so `[^2,^3]`-style OR-disjuncts correctly yield no constraint).
Patterns whose required mask is not a subset of the molecule's mask are skipped before the matcher runs.

Validation: 40.8 M tier-1 (100 K ChEMBL) and 404.5 M tier-2 (1 M USearch) byte-equivalence checks against `RDKit::SubstructMatch`, zero disagreements; full ctest passes.

## CSD Torsion Round 1 — `1f57e83`

Native pattern coverage 314/470 → __391/470__ (66.8 % → 83.2 %); wall 493 s → 428 s (-13.2 %); throughput 20 283 → 23 371 mol/s (+15.2 %); `boost::VF2SubState::match` self-time 10.38 % → 6.85 %.

The CSD torsion matcher's RDKit `SubstructMatch` fallback was the dominant CPU cost at 10M-scale diversity (27–30 % of CPU at scaffold-cache hit rate 33.4 %).
This commit widened `match_linear_pattern`'s SMARTS coverage so the bulk of patterns parse natively, leaving the RDKit fallback only for genuinely non-linear patterns (ring-closure digits, chain continuation inside a branch, multi-hop recursive predicates).

Parser and matcher changes in `csd_torsion_native.hpp`:

- Replaced the flat `simple_atom_constraint_t` with an `atom_disjunct_t[4]` disjunct list plus shared predicate / neighbour-branch fields, unlocking atom-level OR (`[cH,nX2H0]`, `[NH1,NH2]`, `[A,a;r{5-8}]`).
- Replaced the flat `simple_bond_constraint_t` with a `bond_disjunct_t[4]` list, unlocking bond-level OR (`@;=,:`) and encoding the SMARTS default bond as two disjuncts (single, aromatic).
- Fixed SMARTS logical-operator precedence — `,` (OR) binds tighter than `;` (AND), so `A,a;r{5-8}` correctly means `(A OR a) AND r{5-8}`.
  Both `parse_atom_spec` and `parse_bond_op` compile to disjunctive normal form by cross-producting `,`-alternatives across `;`-separated conjuncts.
- Treats bare `[H]` as the hydrogen ATOM (element 1) when no other element primitive appears in the term; treats it as the H-count modifier otherwise.
- Maps `D<n>` to total explicit degree (matching RDKit on `addHs`-ed input).
- Adds an `element_mask_exclusive` mode so `!#1` accepts elements > 63 (heavy metals in the corpus) while a 64-bit mask still covers the common first-63 case.
- Enforces all four torsion atoms distinct (`a1 != a4`).
- Rejects patterns with trailing unbracketed content after the 4th labeled atom (e.g. `...[C:4]=O`).

14 new recursive predicates in `csd_torsion_predicates.hpp` covering the top SMARTS bodies in the fallback set: sulfonyl `S(=O)=O` / `S(=O)(=O)`, `[CX3](=[!O])`, `C([F])([F])[F]`, `[CX3]([C])([H])`, `C([NX3])N` / `C([NX3])n`, the `[N][CX4]` family, `[cH0]F` / `[cH0][OH1]` / `[cH0][OH0]`, `[cH0]([CX3])`, `cO`, `a[OH1]`, `a[NH1,NH2]`, `c([NH1,NH2])`.

Validation: env-var-gated tiered byte-equivalence test in `test_csd_torsion_predicates.cpp` covering a 100 K ChEMBL sample and a 1 M USearch sample, zero disagreements on either tier.

## Earlier Landed Work

Each item references a CHANGELOG-worthy ship event prior to the round-1 / round-2 work above.

- __Int8 bounds quantization__ — N×N DG bounds matrix as `uint8_t` with per-molecule linear dequantization.
  Halved shared-memory at N=128 (32 KB → 16 KB).
  Single-GPU USearch 1M K=1 throughput: +21 %.
- __DG+ETK / MMFF kernel split__ — separated the monolithic pipeline kernel.
  Register pressure 145 → 95–96 per kernel; MMFF theoretical occupancy at bin 128 rose 8.33 % → 37.5 %.
  Single-GPU USearch 1M K=1: +26 % on top of quantization, +53 % cumulative.
- __CSD torsion native predicates (initial 13)__ — replaced the three heaviest RDKit `SubstructMatch` calls (covering 75–85 % of match CPU across six tested datasets) with hand-written C++ predicates.
- __Native Murcko decomposition__ — BFS peel + Weisfeiler–Lehman canonical hash replacing `MurckoDecompose`.
  Eliminated 15 % of CPU prep time.
- __MMFF parameter table extraction__ — full MMFF94 typing, charge, term extraction ported to native flat-array code.
  Byte-identical on 20 K ChEMBL: atom-type 99.91 %, partial-charge 99.97 %, term failure 22 ppm.
- __GPU MMFF atom typer + charge kernel__ — 1 045 580 atoms byte-identical vs CPU; 439 / 20 K molecules fall back.
- __GPU term extraction + CSR build__ — 941 / 1 009 test molecules match; remainder dispatched to CPU.
- __SMILES parse with `removeHs=false`__ — skipped the `removeHs` + re-`addHs` round-trip.
  Saved ~6 % of CPU prep wall time.
- __Scaffold cache__ — caches torsion-pattern match results by Bemis-Murcko scaffold. +44 % throughput on curated ChEMBL (4 698 unique scaffolds across 237 600 cyclic molecules, 98 % reuse).
  Lower hit rate on diverse corpora.
- __Per-wave per-device `cudaEvent_t` instrumentation__ — enables the per-chunk breakdown in `BENCH.md`.
- __Molecule sort by atom count + output permutation__ — dispatches smaller molecules first, permutes back to input order at finalize.
  Improves `atom-limited-bin` fill.
- __Pre-warm cusolver + metric scratch in `context_t` ctor__ — avoids first-wave allocation stall.
- __Pre-grow 256 MB device / host pools per device__ — arena claimed at construction.
- __Four redundant `__syncwarp` removals__ on lane-owned access paths (2026-04-18).
