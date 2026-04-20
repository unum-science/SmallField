# SmallField — Roadmap

Future improvements for the library, spanning both software optimisation and dataset / output-completeness work.
For shipped work see [`CHANGELOG.md`](CHANGELOG.md); for measured throughput see [`BENCH.md`](BENCH.md); for the canonical vocabulary (`parquet-shard`, `coprocessed-batch`, `embedding-wave`, `atom-limited-bin`, `pipeline-role`, `wave-step`) see [`GLOSSARY.md`](GLOSSARY.md).

## How to Read This Document

Every number below carries a scale tag — for example `10M USearch, K=1, 8-GPU` or `100K ChEMBL, K=1, 1-GPU`.
This matters because:

- Scaffold-cache hit rate is dataset-diversity-bound.
  On 50 K ChEMBL it is 85 %; on 10 M USearch it is 33 %; on Enamine REAL it is expected lower.
  Any CPU optimisation whose wall-time share depends on cache-hit rate changes by several × across scales.
- Kernel occupancy is dominated by launch parallelism, not per-kernel resource pressure.
  At bins with ≤ 100 molecules, achieved occupancy is 2 %; at bins with 2 000+ molecules it rises to 23–32 %.
  Extrapolating "the kernel is idle" from small-dataset measurements is misleading.
- Pipeline steady state takes several waves to reach.
  Runs below ~300 K molecules spend a meaningful fraction of wall time in bootstrap / drain, biasing the perceived CPU / GPU share.

Rule of thumb: do not trust any number in this document that is not paired with a scale tag.
When re-profiling to drive a decision, aim for ≥ 1 M molecules and at least 10 `embedding-waves` of steady state — and separately note whether the workload is drug-like (ChEMBL, PubChem) or diverse combinatorial (USearch, Enamine, ZINC).

## Baseline (2026-04-19)

CPU hotspots — 10 M USearch, K=1, 8-GPU, scaffold-cache hit 33.4 %, captured before the round-1 / round-2 CSD-torsion work landed.
`perf record -F 199 --call-graph dwarf,8192 bench_profile /tmp/usearch_10M.smi 1`.
493 s wall, 20 283 mol/s, 98.6 % success, 2.9 M samples.
For the post-CSD-work numbers see `CHANGELOG.md` § round 1 / round 2.

| Symbol (self-time)                                     |   Share | Notes                                                                                                      |
| :----------------------------------------------------- | ------: | :--------------------------------------------------------------------------------------------------------- |
| `boost::detail::VF2SubState<…>::match`                 | 10.38 % | RDKit VF2 subgraph-isomorphism backend, invoked from `SubstructMatch` fallback in the CSD-torsion matcher. |
| `match_linear_pattern<extract_etk_torsions_native::…>` |  6.86 % | Our native CSD torsion matcher — first-path for ~49 % of patterns.                                         |
| `RDKit::detail::MatchSubqueries`                       |  6.20 % | Part of the VF2 subquery walk.                                                                             |
| `extract_mmff94_params`                                |  4.67 % | MMFF term extraction (native).                                                                             |
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

| Atom-limited-bin | Theoretical occupancy | Achieved | Stall reason (top)               |
| ---------------: | --------------------: | -------: | :------------------------------- |
|             ≤ 16 |                 100 % |     32 % | Launch latency (small grid)      |
|               32 |                 100 % |     28 % | Launch latency                   |
|               48 |                  75 % |     25 % | Long scoreboard (positions[])    |
|               64 |                  50 % |     23 % | Long scoreboard                  |
|               80 |                  40 % |     20 % | Long scoreboard, register cap    |
|               96 |                  30 % |     17 % | Register cap (95 regs / thread)  |
|              112 |                  25 % |     14 % | Register cap                     |
|              128 |                  20 % |     12 % | Register cap, shared-memory cap  |

What this means for a fresh planner: at 8-GPU K=1, this workload is host-bound — the CPU `preprocess-step` and the RDKit fallback path together dominate.
The CSD-torsion native matcher (round-1 + round-2) addressed the biggest single CPU symbol, but `extract_mmff94_params`, the OpenMP barrier wait, and SMILES parse remain as CPU-side opportunities.
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

### 3. SMILES Parse Overlap Across Waves

`libRDKitSmilesParse` accounts for 4.87 % of CPU self-time at 10 M scale.
Today every wave's SMILES parsing runs inside that wave's `preprocess-step` on the critical path.

Parse wave N+1's SMILES on the host CPU during wave N's `embedding-step` (which is GPU-bound and leaves CPU cores idle).
The `reader-role` already pulls the next batch from disk in parallel; extending the parse into the async phase closes the remaining CPU-bound share.
Expected gain 4–7 % host prep, upper end if combined with item 2 above.
Effort medium — requires decoupling SMILES parse from the rest of `prepare_molecule_batch_into`.
Risk low to medium — RDKit's SMILES parser is thread-safe for independent molecules.

### 4. 3-way kernel split (DG + ETK-torsion + MMFF)

`etkdg_pipeline_warp_kernel` achieved 10–15 % occupancy vs 25–37 % theoretical at medium bins (see baseline table).
Register pressure (95 / thread) is the secondary constraint after shared-memory.
`mmff_polish_warp_kernel` is near-ceiling already (32 % achieved vs 41.7 % theoretical at bin 128); no meaningful headroom there.

Split `etkdg_pipeline_warp_kernel` into `etkdg_metric_kernel` (DG + stereo retry + fourth-dim crush) and `etkdg_torsion_kernel` (ETK torsion refinement).
Each sub-kernel should fit under ≤ 80 regs/thread and raise its per-block occupancy ceiling.
Existing `mmff_polish_warp_kernel` unchanged.
Expected gain 3–6 % single-GPU K=3 (revised down from earlier estimates: at measured occupancy numbers the binding constraint is launch-parallelism, not register spill; the split only helps if we also end up with more concurrent blocks per SM).
Gated on measuring the global-memory round-trip cost of `positions[]` between kernels — at chunk = 100 K, K = 3, that is roughly 150 MB of extra DMA per bin boundary.
Effort medium.
Risk medium — kernel structure change; numeric parity must stay within the existing P95 energy-delta tolerance on `validate_parquet.py`.

### 5. Allocator Pressure Reduction

jemalloc 4.10 % + libc malloc 2.82 % = ~6.9 % of CPU in allocation.
Scattered across small-vector warmups in the hot loop (`build_descriptor`, `extract_mmff94_params`, per-thread contribs).

Audit `prepare_molecule_batch_into` for temp-alloc sites; promote short-lived `std::vector` to per-thread scratch with grow-only capacity; pre-size the known-bounded containers.
Expected gain 3–5 % host prep.
Effort medium.
Risk low (RAII scratch is a well-understood pattern; the project already uses it for batch pools).

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
At 10 M USearch the scaffold-cache hit rate is 33 %; at Enamine REAL (1.2 B, much longer-tail scaffold distribution) it will drop further.
Grouping input molecules by Bemis-Murcko scaffold before sharding raises the intra-shard cache hit rate and amortises torsion-pattern matching.
Effort small (modify `smi_to_parquet.py` to compute scaffold hash and cluster).
Risk low.
Expected gain is scaffold-hit-rate-dependent: on ChEMBL-like corpora maybe negligible; on Enamine REAL potentially +20–40 % throughput by keeping the cache warm.

### 3. Direct Cloud / S3 Input

Today the `reader-role` reads local `parquet-shard` files.
For billion-scale workloads that do not fit on local disk, an S3 / GCS streaming reader skips the intermediate copy.
Effort medium (Arrow's `S3FileSystem` is already available; plug it into the reader).
Risk low.

## Optimizations Tried and Rejected

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

## Parked (not on roadmap)

- __MMFF parameter static-LUT replacement__ — at 10 M scale MMFF-related CPU is ~8 % total; a 66 % reduction on per-term lookup cost gives ≤ 3 % wall.
  The existing `thread_local std::unordered_map` cache is already lock-free; the proposed static-array replacement would add a dependency (`tsl::robin_map` or similar) and complexity for modest gain.
  The old 23 % MMFF-lookup figure in earlier drafts reflected pre-cache measurement; the cache already captured the bulk of the win.
- __Further `__syncwarp` removals__ — four clearly-redundant sites landed on 2026-04-18.
  The remaining 13 sites are load-bearing (they guard cross-lane reads in shared memory).
- __Bounds matrix int4 nibble-packing__ — marginal gain per author's prior analysis; precision risk on bonded-pair windows.

## Billion-scale workloads

The CLI already streams shard-by-shard: `reader-role` reads one `parquet-shard` at a time, accumulates into a `coprocessed-batch`, `gpu-runner-role` runs its internal `embedding-wave` pipeline, `fallback-role` handles unresolved conformers, `writer-role` flushes an output `parquet-shard`.
ROMols are released per `embedding-wave`, output per `parquet-shard` — nothing scales with total input size.

Remaining gaps at 1–10 B molecule scale are captured in the Dataset roadmap above (scaffold-aware sharding, adaptive cache eviction, direct cloud input).

Projected wall time at the current 10K mol/s host-prep ceiling, no further optimisations applied:

| Library           | Molecules | Wall estimate |
| :---------------- | --------: | ------------: |
| Enamine REAL      |     1.2 B |        ~33 hr |
| GDB-13            |     1.0 B |        ~27 hr |
| REAL Space tier 2 |     7.0 B |       ~195 hr |
| ZINC22            |    37.0 B |         weeks |

Scaffold cache internals: a 64-shard `tsl::robin_map<uint64_t, scaffold_cache_entry_t>` at ~5 KB per entry, keyed by Bemis-Murcko hash.
On curated ChEMBL (4 698 unique scaffolds across 237 600 cyclic molecules, 98 % reuse) it adds +44 % throughput.
Long-tail libraries show lower hit rates — that is what the adaptive-eviction + scaffold-aware-sharding gaps in the Dataset roadmap address.
