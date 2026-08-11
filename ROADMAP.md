# SmallField — Roadmap

Future improvements for the library, spanning both software optimisation and dataset / output-completeness work.
For shipped work see [`CHANGELOG.md`](CHANGELOG.md); for measured throughput see [`BENCH.md`](BENCH.md); for the canonical vocabulary (`parquet-shard`, `coprocessed-batch`, `embedding-wave`, `atom-limited-bin`, `pipeline-role`, `wave-step`) see [`GLOSSARY.md`](GLOSSARY.md).

## How to Read This Document

Every number below carries a scale tag — for example `10M USearch, K=1, 8-GPU` or `100K ChEMBL, K=1, 1-GPU`.
This matters because:

- Corpus shape decides which bins exist at all.
  PubChem spreads across all eight atom-limited bins with 1 % above 112 atoms; Enamine REAL tops out at 72 atoms, so three bins are empty and 98 % of molecules land in bins 48 and 64; GDB13 is 96 % in bin 32.
  A number measured on one of those does not transfer to another — the warp-per-pair change is 2.27 × on PubChem and 1.58 × on REAL for exactly this reason.
  Sample with `bench/sample_corpora.py`, which draws across shards; the head of one shard is not a sample of the corpus.
- Kernel occupancy is dominated by launch parallelism, not per-kernel resource pressure.
  At bins with ≤ 100 molecules, achieved occupancy is 2 %; at bins with 2 000+ molecules it rises to 23–32 %.
  Extrapolating "the kernel is idle" from small-dataset measurements is misleading.
- Pipeline steady state takes several waves to reach.
  Runs below ~300 K molecules spend a meaningful fraction of wall time in bootstrap / drain, biasing the perceived CPU / GPU share.

Rule of thumb: do not trust any number in this document that is not paired with a scale tag.
When re-profiling to drive a decision, aim for ≥ 1 M molecules and at least 10 `embedding-waves` of steady state — and separately note whether the workload is drug-like (ChEMBL, PubChem) or diverse combinatorial (USearch, Enamine, ZINC).

## Baseline as of 2026-04-19

__This table is superseded and kept for provenance only.__
Every VF2 symbol in it — `VF2SubState::match`, `MatchSubqueries`, `SubstructMatch`, `QueryAtom::Match`, `AndQuery`, `EqualityQuery`, `atomCompat`, together about 23 % of host CPU — no longer exists on any code path.
The scaffold cache it references is deleted, so its hit rate no longer means anything.
Re-profile before using any share below to justify work.

CPU hotspots — 10 M USearch, K=1, 8-GPU, scaffold-cache hit 33.4 %, captured before the round-1 / round-2 CSD-torsion work landed.
`perf record -F 199 --call-graph dwarf,8192 bench_profile /tmp/usearch_10M.smi 1`.
493 s wall, 20 283 mol/s, 98.6 % success, 2.9 M samples.
For the post-CSD-work numbers see `CHANGELOG.md` § round 1 / round 2.

| Symbol (self-time)                                     |   Share | Notes                                                                                                      |
| :----------------------------------------------------- | ------: | :--------------------------------------------------------------------------------------------------------- |
| `boost::detail::VF2SubState<…>::match`                 | 10.38 % | RDKit VF2 subgraph-isomorphism backend, invoked from `SubstructMatch` fallback in the CSD-torsion matcher. |
| `match_linear_pattern<extract_etk_torsions_native::…>` |  6.86 % | Our native CSD torsion matcher — first-path for ~49 % of patterns.                                         |
| `RDKit::detail::MatchSubqueries`                       |  6.20 % | Part of the VF2 subquery walk.                                                                             |
| `extract_mmff94_params_via_rdkit`                      |  4.67 % | MMFF term extraction (native).                                                                             |
| libc malloc / mutex hotspot                            |  2.82 % | Allocator + mutex contention.                                                                              |
| `gomp_barrier_wait_end` + team variant                 |  5.17 % | OpenMP load-imbalance at `prepare_per_thread_contribs_` barrier.                                           |
| `RDKit::SubstructMatch`                                |  1.83 % | Fallback entry.                                                                                            |
| `extract_etk_torsions_native`                          |  1.79 % | ETK torsion extraction outside the matcher.                                                                |
| `RDKit::QueryAtom::Match`                              |  1.78 % | Query-chain predicate.                                                                                     |
| `Queries::AndQuery<…>::Match`                          |  1.75 % | Query combinator.                                                                                          |
| `Queries::EqualityQuery<…>::Match`                     |  1.48 % | Atom-spec equality check.                                                                                  |
| `build_descriptor`                                     |  1.45 % | Molecule descriptor build.                                                                                 |
| `RDKit::atomCompat`                                    |  1.39 % | Atom compare in VF2.                                                                                       |
| MMFF force-field helpers (RDKit)                       |  1.35 % | Per-term lookup tail.                                                                                      |
| `operator new`                                         |  1.08 % | Allocation.                                                                                                |
| `RDKit::Chirality::iterateCIPRanks`                    |  1.00 % | Chirality ranker.                                                                                          |

GPU per-bin occupancy — 100 K ChEMBL, K=1, 1-GPU, captured with `ncu --set basic --launch-skip 32 --launch-count 16` so each row is a steady-state launch (not bootstrap).

| Atom-limited-bin | Theoretical occupancy | Achieved | Stall reason (top)              |
| ---------------: | --------------------: | -------: | :------------------------------ |
|             ≤ 16 |                 100 % |     32 % | Launch latency (small grid)     |
|               32 |                 100 % |     28 % | Launch latency                  |
|               48 |                  75 % |     25 % | Long scoreboard (positions[])   |
|               64 |                  50 % |     23 % | Long scoreboard                 |
|               80 |                  40 % |     20 % | Long scoreboard, register cap   |
|               96 |                  30 % |     17 % | Register cap (95 regs / thread) |
|              112 |                  25 % |     14 % | Register cap                    |
|              128 |                  20 % |     12 % | Register cap, shared-memory cap |

What this means for a fresh planner: at 8-GPU K=1, this workload is host-bound — the CPU `preprocess-step` and the RDKit fallback path together dominate.
The CSD-torsion native matcher (round-1 + round-2) addressed the biggest single CPU symbol, but `extract_mmff94_params_via_rdkit`, the OpenMP barrier wait, and SMILES parse remain as CPU-side opportunities.
On the GPU side, occupancy at production bin sizes (~ 23–32 % on the medium bins) is healthy; the residual is launch parallelism rather than per-kernel resource pressure, so kernel restructuring has limited headroom unless it widens the launch grid.

## Software Roadmap

Ranked by measured-gain × risk-adjusted probability of delivery.
Items that have already shipped are recorded in `CHANGELOG.md`.

### 1. Inner-Pipeline Ring-Buffer Depth 3 + Async Finalize

The 2-slot `batch_pool_t` ring in `generate_conformers` idles the GPU 20–30 ms per wave transition while the CPU preprocess of wave N+1 is still running and wave N+1's `launch_wave` has not fired.
`finalize_gpu_pipeline` calls `cudaStreamSynchronize` and blocks for 5–10 ms before next wave's `launch_wave` can start.

Promote the 2-slot ring to 3 slots; issue the D2H + reorder of wave N on a separate stream so wave N+1's launch can overlap with wave N's finalize; pre-allocate the third slot's pinned buffer at `context_t` ctor so no `cudaHostAlloc` fires mid-run.
Expected gain 4–6 % from the depth-3 ring plus 1–2 % from async finalize.
Independent of kernel-level work; both are host-scheduling fixes.
Effort medium — the ring state-machine needs careful invariants so slot `(i % 3)` is not overwritten while wave `(i - 3)` is still in flight.
Risk medium — a 3-stream split (H2D / compute / D2H) was landed in `a791e54` and reverted in `db25c9d` because it introduced a second `illegal memory access` race at the `cudaEventSynchronize(finalize)` hand-off under PubChem drug-like load AND regressed 10M K=3 throughput by ~13 %.
Any future attempt must re-validate at drug-like scale, not just on USearch fragment-like where the race window is smaller and kernel phases are long enough to benefit from copy/compute overlap.

### 2. OpenMP Load-Imbalance Fix at `preprocess-step` Barrier

5.17 % of CPU at 10M scale lives in `gomp_barrier_wait_end` at the end of the `#pragma omp parallel for` in `prepare_per_thread_contribs_`.
Tail threads still process the biggest molecules while short-molecule threads idle.

Switch `schedule(dynamic, 64)` to `schedule(guided)`, or pre-sort molecules by estimated atom count so the longest ones start first.
The SMILES-length sort inside `generate_conformers` already approximates this at the outer level; applying the same sort per thread-block inside the prep loop closes the imbalance.
Expected gain 4–6 % host prep.
Effort small — one pragma change plus an optional sort.
Risk low.

The 5.17 % is measured against a `prepare_per_thread_contribs_` that still typed and extracted MMFF terms on the host.
Those stages are on the device now, so the loop this item targets is a fraction of what was profiled and the imbalance has to be re-measured before the fix is sized.

### 2b. Multi-`parquet-shard` Invocation for Full 4-Role Overlap

The `smallfield_batch` CLI runs a persistent 4-role pipeline — `reader-role → gpu-runner-role → fallback-role → writer-role`, with depth-3 ring handoffs between them — but the historical wrapper (`/tmp/pubchem_per_shard_loop.sh`) invoked `smallfield_batch` once per `parquet-shard`.
At 1 M molecules per `parquet-shard` and ~800 K target `coprocessed-batch` size on 8 GPUs, that means each invocation saw ~1.25 `coprocessed-batches` in flight, so the ring stayed near-empty and the role overlap never materialised.
GPU aggregate utilisation measured 12.5 % across a wave's wall; `top -bn1 -H -p <pid>` showed one role burning 1–27 cores while the others sat in `futex_wait`.

The current wrapper (`/tmp/pubchem_multi_shard_loop.sh`) invokes `smallfield_batch` _once_ on the entire `parquet-shard` input directory.
`--resume` (CLI default) auto-skips any `parquet-shard` whose `*_3D.parquet` output already exists, so relaunching on crash picks up exactly where the previous process left off.
One process sees all pending `parquet-shards` as a stream, the ring genuinely fills, and `fallback-role` on `coprocessed-batch` N overlaps `gpu-runner-role` on `coprocessed-batch` N+1 as the glossary diagram always promised.

A previous attempt to solve the overlap by splitting the `compute_thread_budget` between `gpu-runner-role` and `fallback-role` (5 : 3 in favour of fallback) reduced single-invocation throughput by ~5 % — each role got fewer cores for its solo bursts and the "overlap" window that was supposed to compensate didn't exist in the per-shard wrapper mode.
That split is reverted; both roles again default to `num_procs - 4` (the pre-existing design).
The `--rdkit-fallback-threads N` and `--gpu-runner-threads N` flags remain for workloads where an operator wants to tune the mix by hand.

### 3. SMILES Parse Overlap Across Waves

`libRDKitSmilesParse` accounts for 4.87 % of CPU self-time at 10 M scale.
Today every wave's SMILES parsing runs inside that wave's `preprocess-step` on the critical path.

Parse wave N+1's SMILES on the host CPU during wave N's `embedding-step` (which is GPU-bound and leaves CPU cores idle).
The `reader-role` already pulls the next batch from disk in parallel; extending the parse into the async phase closes the remaining CPU-bound share.
Expected gain 4–7 % host prep, upper end if combined with item 2 above.
Effort medium — requires decoupling SMILES parse from the rest of `prepare_molecule_batch_into`.
Risk low to medium — RDKit's SMILES parser is thread-safe for independent molecules.

### 4. Kernel Occupancy — Closed, Not Worth Pursuing

Retired by measurement rather than delivered.
`etkdg_pipeline_warp_kernel` went from 80 to 72 registers per thread, lifting it from 25 to 28 blocks per SM, and PubChem 5K at K=3 moved from 4351 to 4359 conf/s on an idle box — 0.2 %, inside a 0.6 % spread.

A 12 % occupancy gain buying nothing measurable says the kernel is not occupancy-limited at production bin sizes, so neither a 3-way split nor the remaining eight registers to 64 would pay.

Two further constraints found while establishing that.
A 32-thread block can occupy at most 32 of an SM's 64 warp slots, measured via `maxBlocksPerMultiProcessor`, so 50 % is this shape's ceiling at any register count — the occupancy table in the baseline above quotes figures at and beyond 100 % that are unreachable on any GPU this project targets.
And shared memory binds only at N=128, which is 1 % of PubChem and __0 % of Enamine REAL__, whose molecules top out at 72 atoms.

### 5. Allocator Pressure Reduction

jemalloc 4.10 % + libc malloc 2.82 % = ~6.9 % of CPU in allocation.
Scattered across small-vector warmups in the hot loop (`build_descriptor`, `extract_mmff94_params_via_rdkit`, per-thread contribs).

Audit `prepare_molecule_batch_into` for temp-alloc sites; promote short-lived `std::vector` to per-thread scratch with grow-only capacity; pre-size the known-bounded containers.
Expected gain 3–5 % host prep.
Effort medium.
Risk low — RAII scratch is a well-understood pattern and the project already uses it for batch pools.

`extract_mmff94_params_via_rdkit` and the per-thread contribs are gone from the host path, which is where most of this allocation sat, so the 6.9 % is stale and needs a fresh profile before the audit is worth running.

### 6. Attribute the Fixed Context-Construction Cost

`context_t` construction costs about 35 seconds before a single molecule is embedded, measured as the gap between total process time and the timed `generate_conformers` window on a 20-molecule input.
It is a fixed cost, independent of corpus size and conformer count.

Excluded so far, each measured directly: the MMFF94 table extraction from RDKit and the compiled CSD torsion table, since `test_mmff94_tables`, `test_mmff_gpu_typer` and `test_csd_torsion_device` each build their tables and finish in under half a second; CUDA module loading, since `CUDA_MODULE_LOADING=LAZY` changes nothing; and PTX JIT, since `cuobjdump` reports an sm_90 cubin in the binary.
Neither the per-device scratch struct nor the MMFF staging layout is large enough to explain it — the struct holds only buffer handles and the staging window is a few hundred megabytes.

The CLI amortizes this over a whole shard, so it does not touch dataset throughput, but it dominates short runs and the Python binding, and it silently distorts any whole-process utilization measurement.
Attribute it before optimizing anything else in the startup path.

### 7. Widen the ETK CSR Build Past One Lane

`etk_build_csr_kernel` in `preprocess_kernel.cuh` launches a warp per molecule and then returns every lane but the first, so the counting pass, the prefix sum over the per-atom cursors and the scatter all run serially on lane 0.
The work is a single pass over the molecule's torsions and constraints, so it is small next to the ETKDG kernel and was not worth blocking the device-preprocessing merge on.
A lane-strided count with a warp-scan over the cursors is the natural fix.
Measure the kernel in isolation first — if it is under a percent of the wave it stays a note.

## Dataset Roadmap

Separate axis from raw throughput.
These items close output-completeness gaps that ETKDG + MMFF do not address, not CPU / GPU cycles.

### 1. Detect-and-Tag Axial / Helical Chirality

Axial chirality (atropisomerism in ortho-substituted biaryls) and helical chirality (helicenes and similar fused-aromatic helices) are not encoded in a SMILES string at all — two molecules with opposite handedness share one SMILES.
ETKDG embeds one handedness arbitrarily from the random seed.
Sampling 100 K of each of USearch and ChEMBL shows axial chirality in 0.28–0.44 % of molecules and helical chirality in 0.11–0.18 %.

Detection is cheap once `build_descriptor` has run.
The biaryl axial check is a single SMARTS-style walk over the heavy atoms; the helical check is a connected-component count over `atom_ring_bitmap`.
Both reuse the descriptor's existing aromaticity and ring-info fields and add < 2 µs per molecule on top of the current ~10 ms preprocess budget.

Scope: add `axial_chirality_detected` and `helical_chirality_detected` bits to `molecule_descriptor_t`, propagate them through `finalize_gpu_pipeline` into `output_buffers_t`, and emit them as two new boolean columns in the Parquet writer.
Consumers who care about enantiopure series can filter to the rare flagged molecules and re-embed them with explicitly constrained stereo; consumers who do not care simply ignore the columns.
Effort small (~50 LOC additive).
Risk low.

This is explicitly NOT enumeration.
The library does not generate a second handedness on the user's behalf.
Per the README's `Project Goals & FAQ`, SmallField respects the input SMILES and treats configuration-space coverage as an upstream concern.

### 2. Scaffold-Aware Input Sharding

Today `smi_to_parquet.py` writes offset-based shards.
Grouping input molecules by Bemis-Murcko scaffold before sharding would place structurally similar molecules in the same `coprocessed-batch`.
Effort small — compute a scaffold hash in `smi_to_parquet.py` and cluster on it.
Risk low.

__The original justification for this item is void.__
It rested on a 33 % scaffold-cache hit rate measured at 10 M USearch, and that cache was deleted along with the VF2 fallback, so there is no longer a cache to keep warm.
Any remaining gain would have to come from somewhere else — warp divergence in the native torsion matcher, or `atom-limited-bin` fill — and none of that has been measured.
Re-derive the mechanism and measure it before costing this work.

### 3. Direct Cloud / S3 Input

Today the `reader-role` reads local `parquet-shard` files.
For billion-scale workloads that do not fit on local disk, an S3 / GCS streaming reader skips the intermediate copy.
Effort medium (Arrow's `S3FileSystem` is already available; plug it into the reader).
Risk low.

## Optimizations Tried and Rejected

- __Size-aware distance-geometry budget__ — the DG stage runs a flat 100 FIRE iterations for every molecule while `select_refinement_policy_` already scales the ETK and MMFF stages, so it looked like the one stage whose cost grows with the bounds matrix had simply been missed.
  Giving it the same three tiers — 60 iterations for light molecules, 160 for heavy — cost 3 102 conformers on stratified 500 K PubChem at K=3 and pushed `stereo_failure` from 0.87 % to 1.07 %, for no wall-time change at all: 198.7 s against 199.8 s.
  Cutting the budget yields a worse starting geometry, and chirality validation is what notices.
  The stage is not oversized for small molecules; it is what makes their stereocentres survive.
  Reverted.
- __Gradient register tile__ — moved FIRE gradient array from shared to per-lane registers to attack `long_scoreboard` stalls.
  ncu showed stalls came from positions/velocities reads, not gradient.
  Registers rose 116 → 146 with no occupancy win.
  Reverted.
- __`__launch_bounds__` bump from 32,4 to 32,6__ — inert because nvcc already fit 8 blocks/SM from the shared-memory cap.
- __Register cap on the monolithic kernel__ — `--maxrregcount 128` caused local-memory spills that cancelled the occupancy gain.
  Only viable after the kernel split (now landed).
- __Wave-bin retune__ — merging bins 97–112 into 128 would cost ~30 % more compute on those molecules; dropping bin ≤ 16 would save ~40 ms (covers only 0.6 % of ChEMBL).
  Not worth the churn.
- __Tensor cores, FP16 coordinates, larger chunks__ — rejected by ncu data.
  Math pipes idle at 0.2 %; grid already fills the device in production.
- __Native CIP chirality ranker__ — two attempts reached 98.94 % and 99.71 % agreement with RDKit, short of the ≥ 99.95 % threshold.
  Second attempt regressed P50 energy (−6.3 vs −5.4 kcal/mol) and −2.8 % throughput.
  Shelved until edge cases around embedded double-bond systems are resolved upstream.
- __Branchless `match_linear_pattern` lookup__ — no viable optimization target after profiling.
  Closed.
- __`__noinline__` on `fire_minimize_`__ — increased register pressure to 207, dropping theoretical occupancy to 16.67 %.
  nvcc re-inlined when the hint was removed; compiler default was already optimal.
- __`cudaEventSynchronize` in place of the finalize stream drain__ — swapping the `cudaStreamSynchronize` in `finalize_gpu_pipeline` for a sync on the recorded end event surfaced `cudaErrorNotSupportedOnDeviceAddressSpace` on several hundred waves per shard under PubChem drug-like load.
  The stream drain is correct whether or not the end event was recorded and costs about 5 ms per wave per device.
  Reverted.
- __`cudaMallocManaged` + `cudaMemPrefetchAsync` for `wave_slice_pool_t`__ — considered for a truly unified pointer in place of the pinned-host-plus-explicit-copy design.
  HMM overhead and prefetch bookkeeping ran 2x slower than pinned+explicit at the batch sizes this project ships at.
  The layout and span surface are identical, so the backing store can still be swapped if unified-memory hardware makes managed competitive.

Delivered work is recorded in [`CHANGELOG.md`](CHANGELOG.md), including the fixed-seed reproducibility fix,
the retirement of the VF2 fallback, and the move of preprocessing onto the device.
