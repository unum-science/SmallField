# SmallField — Roadmap

Possible future improvements for the library, spanning both software optimisation and dataset / output-completeness work.
See [`BENCH.md`](BENCH.md) for hardware + measured throughput; [`GLOSSARY.md`](GLOSSARY.md) for the canonical vocabulary (`parquet-shard`, `coprocessed-batch`, `embedding-wave`, `atom-limited-bin`, `pipeline-role`, `wave-step`).

## How to Read This Document

Every number below carries a __scale tag__ — e.g. `10M USearch, K=1, 8-GPU` or `100K ChEMBL, K=1, 1-GPU`.
This matters because:

- __Scaffold-cache hit rate is dataset-diversity-bound__.
  On 50 K ChEMBL it's 85 %; on 10 M USearch it's 33 %; on Enamine REAL it's expected lower.
  Any CPU optimisation whose wall-time share depends on cache-hit rate changes by several × across scales.
- __Kernel occupancy is dominated by launch parallelism, not per-kernel resource pressure__.
  At bins with ≤100 molecules, achieved occupancy is 2 %; at bins with 2 000+ molecules it rises to 23–32 %.
  Extrapolating "the kernel is idle" from small-dataset measurements is misleading.
- __Pipeline steady state takes several waves to reach__.
  Runs below ~300 K molecules spend a meaningful fraction of wall time in bootstrap / drain, biasing the perceived CPU / GPU share.

__Rule of thumb__: do not trust any number in this document that is not paired with a scale tag.
When re-profiling to drive a decision, aim for ≥ 1 M molecules and at least 10 `embedding-waves` of steady state — and separately note whether the workload is drug-like (ChEMBL, PubChem) or diverse combinatorial (USearch, Enamine, ZINC).

## Baseline (2026-04-19)

### CPU hotspots — 10M USearch, K=1, 8-GPU, scaffold-cache hit 33.4 %

`perf record -F 199 --call-graph dwarf,8192 bench_profile /tmp/usearch_10M.smi 1`.
493 s wall, 20 283 mol/s, 98.6 % success, 2.9 M samples.

| Symbol (self-time) | Share | Notes |
| :--- | ---: | :--- |
| `boost::detail::VF2SubState<…>::match` | 10.38 % | RDKit VF2 subgraph-isomorphism backend, invoked from `SubstructMatch` fallback in the CSD-torsion matcher. |
| `match_linear_pattern<extract_etk_torsions_native::…>` | 6.86 % | Our native CSD torsion matcher — first-path for ~49 % of patterns. |
| `RDKit::detail::MatchSubqueries` | 6.20 % | Part of the VF2 subquery walk. |
| `extract_mmff94_params` | 4.67 % | MMFF term extraction (native). |
| libc malloc / mutex hotspot | 2.82 % | Allocator + mutex contention. |
| `gomp_barrier_wait_end` + team variant | 5.17 % | OpenMP load-imbalance at `prepare_per_thread_contribs_` barrier. |
| `RDKit::SubstructMatch` | 1.83 % | Fallback entry. |
| `extract_etk_torsions_native` | 1.79 % | ETK torsion extraction outside the matcher. |
| `RDKit::QueryAtom::Match` | 1.78 % | Query-chain predicate. |
| `Queries::AndQuery<…>::Match` | 1.75 % | Query combinator. |
| `Queries::EqualityQuery<…>::Match` | 1.48 % | Atom-spec equality check. |
| `build_descriptor` | 1.45 % | Molecule descriptor build. |
| `RDKit::atomCompat` | 1.39 % | Atom compare in VF2. |
| MMFF force-field helpers (RDKit) | 1.35 % | Per-term lookup tail. |
| `operator new` | 1.08 % | Allocation. |
| `RDKit::Chirality::iterateCIPRanks` | 1.00 % | Chirality ranker. |

__DSO share__ — where self-time lives by shared library:

| DSO | Share |
| :--- | ---: |
| `libRDKitGraphMol` | 21.6 % |
| `libRDKitSubstructMatch` | 21.5 % |
| `bench_profile` (native code) | 18.6 % |
| `[kernel / unmapped]` | 12.9 % |
| `libc` | 5.9 % |
| `libgomp` | 5.2 % |
| `libRDKitSmilesParse` | 4.9 % |
| `libjemalloc` | 4.1 % |
| `libRDKitForceFieldHelpers` | 4.0 % |

__Bucket ranking__ (sum of related symbols):

| Concern | Combined % | Commentary |
| :--- | ---: | :--- |
| __RDKit VF2 + `SubstructMatch` query-chain__ | __27–30 %__ | VF2 10.38 + MatchSubqueries 6.20 + SubstructMatch 1.83 + QueryAtom::Match 1.78 + AndQuery 1.75 + EqualityQuery 1.48 + atomCompat 1.39 + bondCompat 0.80 + RecursiveMatcher 0.60 + query-chain tail.  #1 CPU opportunity. |
| __CSD torsion native matcher__ (our code) | ~8.6 % | `match_linear_pattern` 6.86 + `extract_etk_torsions_native` 1.79 — already first-path for ~49 % of patterns. |
| __MMFF (all)__ | ~8 % | `extract_mmff94_params` 4.67 + RDKit MMFF helpers ~3.3. |
| __Allocator pressure__ | ~6.9 % | libc 2.82 + libjemalloc 4.10 — temp-alloc churn in the hot loop. |
| __SMILES parse + descriptor__ | ~6.3 % | libRDKitSmilesParse 4.87 + `build_descriptor` 1.45. |
| __OpenMP barrier wait__ | 5.17 % | Load imbalance at the prep barrier. |
| __Chirality (RDKit CIP)__ | ~1 % | Much smaller than previous drafts claimed. |

### GPU per-bin occupancy — 100K ChEMBL, K=1, 1-GPU

`ncu --set basic --launch-skip 32 --launch-count 16 bench_profile /tmp/chembl_100k.smi 1` with `CUDA_VISIBLE_DEVICES=0`.

| Kernel | Grid | shmem (B) | Theoretical Occ | __Achieved Occ__ | SM % | Duration (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `etkdg_pipeline_warp_kernel` | 328 | 13 824 | 12.5 % | 3.6 % | 1.6 % | 59 |
| `etkdg_pipeline_warp_kernel` | __2 392__ | 4 608 | 37.5 % | __13.0 %__ | 7.1 % | 59 |
| `etkdg_pipeline_warp_kernel` | __2 256__ | 7 168 | 25.0 % | __15.2 %__ | 11.8 % | 93 |
| `etkdg_pipeline_warp_kernel` | 1 216 | 10 240 | 18.8 % | 10.9 % | 5.9 % | 55 |
| `mmff_polish_warp_kernel` | 328 | 3 456 | 41.7 % | 3.6 % | 5.1 % | 409 |
| `mmff_polish_warp_kernel` | __2 392__ | 1 728 | 41.7 % | __23.3 %__ | __34.9 %__ | 181 |
| `mmff_polish_warp_kernel` | __3 760__ | 2 304 | 41.7 % | __32.3 %__ | __37.6 %__ | 543 |
| `mmff_polish_warp_kernel` | 1 216 | 2 880 | 41.7 % | 12.2 % | 16.5 % | 383 |

Per-bin theoretical occupancy from the register/shared-memory cap analysis (stable across runs):

| Kernel | Bin | Regs | Dyn Shmem | Block Limit (shmem / regs) | Theoretical Occ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `etkdg_pipeline_warp_kernel` | 64 | 95 | 7.17 KB | 12 / 20 | 25.0 % |
| `etkdg_pipeline_warp_kernel` | 128 | 95 | 22.53 KB | 4 / 20 | 8.3 % |
| `mmff_polish_warp_kernel` | 64 | 96 | 2.30 KB | 30 / 20 | 41.7 % |
| `mmff_polish_warp_kernel` | 128 | 96 | 4.61 KB | 18 / 20 | 37.5 % |

The DG+ETK kernel is shared-memory-bound at large molecules (int8-quantized bounds matrix, 16 KB at N=128).
The MMFF kernel carries no bounds matrix and hits its ceiling.

### What this means

At __8-GPU K=1 on diverse corpora__, host preprocess is the ceiling.
The single biggest CPU bucket is the RDKit VF2 subgraph-isomorphism backend dragged in by the ~200 CSD torsion patterns that our native matcher doesn't yet parse — 27–30 % of CPU at 10M scale.

GPU kernels at production bin size are __near-optimal, not idle__.
MMFF polish reaches 32 % achieved occupancy, 37 % SM throughput on the largest bins — the earlier "2 % occupancy" claim was a small-dataset artefact.
The ETKDG pipeline kernel has a 2× gap from its theoretical ceiling (15 % achieved vs 37 % theoretical at medium bins) — that's where the remaining kernel-level headroom lives.

The third axis — __CPU-GPU ping-pong__ — is a real cost not attributable to either CPU or GPU in isolation.
The 2-slot `embedding-wave` ring buffer leaves 20–30 ms of GPU idle per wave transition; finalize-to-launch sequencing loses another 5–10 ms per wave.
Together ≈ 5–8 % of end-to-end throughput that can be clawed back without touching either the CPU code or the kernels.

## Software Roadmap

Ranked by measured-gain × risk-adjusted probability of delivery.

### 1. Extend native CSD torsion predicates to cover the VF2 fallback

__Measured basis__: 27–30 % of CPU at 10M USearch goes through RDKit VF2 + query-chain via one single direct call site — `src/smallfield/preprocess/csd_torsion_dispatcher.hpp:357`, the `RDKit::SubstructMatch` fallback inside the CSD-torsion pattern loop.
Not buried; we own the dispatch code above and below the call.

__Scope__: the CSD torsion pattern library ships as ~400 fixed SMARTS patterns with RDKit.
`src/smallfield/preprocess/csd_torsion_native.hpp::match_linear_pattern` already covers ~49 % of them via a purpose-built linear-SMARTS parser plus 13 hardcoded recursive-subquery predicates.
The remaining ~51 % fall back to VF2 because they use SMARTS features the linear parser doesn't handle (atom-level `OR`, recursive `$(…)` outside the 13 hardcoded ones, more than two inline branches per atom).

__Engineering__: ~1 000 LOC of additive hand-coded predicates — same pattern as the existing 13.
Correctness validated against the existing `test/preprocess/test_csd_torsion_predicates.cpp` harness (byte-identical match lists).
No architectural risk; the dispatcher already selects per-pattern between native and fallback.

__Expected gain__: 20–25 % of host CPU at 10M-class diversity (scaffold-cache hit rate 33 %).
On 50K-ChEMBL-class diversity (scaffold-cache hit rate 85 %), the combined VF2 + query-chain is only ~5 % of CPU — so the same code wins a much smaller fraction.
__This optimisation's value scales with input diversity__.

__Effort__: Medium-High (1 000 LOC, repetitive per-pattern, each new predicate is ~50 LOC).
__Risk__: Low — all predicates are byte-equivalence-tested.

### 2. Inner-pipeline ring-buffer depth 3 + async finalize

__Measured basis__: the 2-slot `batch_pool_t` ring in `generate_conformers` idles the GPU 20–30 ms per wave transition while the CPU preprocess of wave N+1 is still running and wave N+1's `launch_wave` hasn't fired.
Additionally, `finalize_gpu_pipeline` at `src/smallfield/api/smallfield.cu:623` calls `cudaStreamSynchronize` and blocks for 5–10 ms before next wave's `launch_wave` can start.

__Scope__: promote the 2-slot ring to 3 slots; issue the D2H + reorder of wave N on a separate stream so wave N+1's launch can overlap with wave N's finalize; pre-allocate the third slot's pinned buffer at `context_t` ctor so no `cudaHostAlloc` mid-run.

__Expected gain__: 4–6 % throughput from depth-3 ring + 1–2 % from async finalize.
Independent of kernel-level work; both are host-scheduling fixes.

__Effort__: Medium (ring state-machine needs careful invariants so slot `(i % 3)` isn't overwritten while wave `(i - 3)` is still in flight).
__Risk__: Medium — the 2-slot ring has proven stable after last cycle's multi-wave illegal-memory-access debugging; changing depth re-opens the invariant carefully.

### 3. OpenMP load-imbalance fix at `preprocess-step` barrier

__Measured basis__: 5.17 % of CPU at 10M scale in `gomp_barrier_wait_end` at the end of the `#pragma omp parallel for` in `prepare_per_thread_contribs_`.
Tail threads still processing the biggest molecules while short-molecule threads idle.

__Scope__: switch `schedule(dynamic, 64)` to `schedule(guided)`, or pre-sort molecules by estimated atom count so the longest ones start first.
The SMILES-length sort inside `generate_conformers` already approximates this at the outer level; applying the same sort per thread-block inside the prep loop closes the imbalance.

__Expected gain__: 4–6 % host prep.
__Effort__: Small — one pragma change + optional sort.
__Risk__: Low.

### 4. SMILES parse overlap across waves

__Measured basis__: `libRDKitSmilesParse` accounts for 4.87 % of CPU self-time at 10M scale.
Today every wave's SMILES parsing runs inside that wave's `preprocess-step` on the critical path.

__Scope__: parse wave N+1's SMILES on the host CPU _during_ wave N's `embedding-step` (which is GPU-bound and leaves CPU cores idle).
The `reader-role` already pulls the next batch from disk in parallel; extending the parse into the async phase closes the remaining CPU-bound share.

__Expected gain__: 4–7 % host prep (upper end if combined with the load-imbalance fix above).
__Effort__: Medium — requires decoupling SMILES parse from the rest of `prepare_molecule_batch_into`.
__Risk__: Low–Medium — RDKit's SMILES parser is thread-safe for independent molecules.

### 5. 3-way kernel split (DG + ETK-torsion + MMFF)

__Measured basis__: `etkdg_pipeline_warp_kernel` achieved 10–15 % occupancy vs 25–37 % theoretical at medium bins (see baseline).
Register pressure (95/thread) is the secondary constraint after shared-memory.
`mmff_polish_warp_kernel` is near-ceiling already (32 % achieved vs 41.7 % theoretical at bin 128); no meaningful headroom there.

__Scope__: split `etkdg_pipeline_warp_kernel` into `etkdg_metric_kernel` (DG + stereo retry + fourth-dim crush) and `etkdg_torsion_kernel` (ETK torsion refinement).
Each sub-kernel should fit under ≤80 regs/thread and raise its per-block occupancy ceiling.
Existing `mmff_polish_warp_kernel` unchanged.

__Expected gain__: __3–6 % single-GPU K=3__ (revised down from the previous 5–15 % draft — at measured occupancy numbers the binding constraint is launch-parallelism, not register spill; the split only helps if we also end up with more concurrent blocks / SM).
__Gated__: measure the global-memory round-trip cost of `positions[]` between kernels first — at chunk=100K, K=3, that's ~150 MB extra DMA per bin boundary.
__Effort__: Medium.
__Risk__: Medium — kernel structure change; numeric parity must stay within the existing P95 energy-delta tolerance on `validate_parquet.py`.

### 6. Allocator pressure reduction

__Measured basis__: jemalloc 4.10 % + libc malloc 2.82 % = ~6.9 % of CPU in allocation.
Scattered across small-vector warmups in the hot loop (`build_descriptor`, `extract_mmff94_params`, per-thread contribs).

__Scope__: audit `prepare_molecule_batch_into` for temp-alloc sites; promote short-lived `std::vector` to per-thread scratch with grow-only capacity; pre-size the known-bounded containers.

__Expected gain__: 3–5 % host prep.
__Effort__: Medium.
__Risk__: Low (RAII scratch is a well-understood pattern; we already do it for batch pools).

## Dataset Roadmap

Separate axis from raw throughput.
These items close __output-completeness__ gaps that ETKDG + MMFF do not address, not CPU / GPU cycles.

### 1. Symmetry-class enumeration upstream of conformer generation

__Gap__: ETKDG samples _locally_ within a conformational basin; MMFF energy-minimises toward the nearest local minimum.
Neither enumerates disconnected regions of configuration space.
For inputs with unspecified point chirality, axial chirality (biphenyls), or helicity (helicenes):

- ETKDG picks an arbitrary embedding → one stereoisomer, one helicity per call.
- MMFF collapses shallow variants into the nearest basin.
- Multiple `generate_conformers` calls on the same SMILES do not systematically cover both enantiomers or both helicities.

For production consumers that care about chirality completeness, this is a __correctness-level concern__, not a throughput one.

__Proposal__: detect symmetry classes upstream, produce one "virtual input" per class, run the existing conformer generator per class, and tag output rows with class ID.

Pipeline sketch:

```
Input SMILES
   → symmetry-class detector   (CPU, upstream of embedding-wave dispatch)
   → fan out to N virtual inputs, one per class
   → existing wave pipeline runs per class independently
   → output tagged with (input_idx, class_id, class_kind, helicity_sign, …)
```

__Three tiers__:

1. __Point chirality__ — stereocenters inherited from SMILES `@` / `@@` markers and inferred via CIP.
   RDKit's `EnumerateStereoisomers` does this today; integrate as the first tier.
   Low-risk, covers the most common case.
2. __Axial / helical chirality__ — biphenyls with hindered rotation, helicenes, atropisomers.
   RDKit does not enumerate these by default.
   Needs native detection + explicit seed geometry for each handedness.
3. __Topological variants__ — knots, trefoils, linked rings.
   Rare in drug-like libraries; deferred.

__Effort__: Medium–Large (new preprocess-step before the SMILES parse).
__Risk__: Medium (tier 1 straightforward; tiers 2–3 require pattern detection + constrained embedding).

### 2. Output tagging for class completeness

When the above lands, output parquet-shards gain per-conformer columns: `class_id`, `class_kind` (enum: point / axial / helical / topological), `helicity_sign`, `stereocenter_count`, `completeness_flag`.
Existing users see no change (single class per input, backwards-compatible default).

### 3. Scaffold-aware input sharding

Today `smi_to_parquet.py` writes offset-based shards.
At 10M USearch the scaffold-cache hit rate is 33 %; at Enamine REAL (1.2 B, much longer-tail scaffold distribution) it will drop further.
Grouping input molecules by Bemis-Murcko scaffold before sharding raises the intra-shard cache hit rate and amortises torsion-pattern matching.

__Effort__: Small (modify `smi_to_parquet.py` to compute scaffold hash + cluster).
__Risk__: Low.
__Expected gain__: scaffold-hit-rate-dependent.
On ChEMBL-like corpora maybe negligible; on Enamine REAL potentially +20–40 % throughput by keeping cache warm.

### 4. Direct cloud / S3 input

Today the `reader-role` reads local `parquet-shard` files.
For billion-scale workloads that don't fit on local disk, an S3 / GCS streaming reader skips the intermediate copy.

__Effort__: Medium (Arrow's `S3FileSystem` is already available; plug it into the reader).
__Risk__: Low.

## Optimizations that landed (historical reference)

- __Int8 bounds quantization__ — N×N DG bounds matrix as `uint8_t` with per-molecule linear dequantization.
  Halved shared-memory at N=128 (32 KB → 16 KB).
  Single-GPU USearch 1M K=1: +21 %.
- __DG+ETK / MMFF kernel split__ — separated the monolithic pipeline kernel.
  Register pressure 145 → 95–96 per kernel; MMFF theoretical occupancy at bin 128 rose 8.33 % → 37.5 %.
  Single-GPU USearch 1M K=1: +26 % on top of quantization, +53 % cumulative.
- __CSD torsion native predicates__ — replaced the three heaviest RDKit `SubstructMatch` calls (covering 75–85 % of match CPU across six tested datasets) with hand-written C++ predicates.
- __Native Murcko decomposition__ — BFS peel + WL canonical hash replacing `MurckoDecompose`.
  Eliminated 15 % of CPU prep time.
- __MMFF parameter table extraction__ — full MMFF94 typing, charge, term-extraction ported to native flat-array code.
  Byte-identical on 20K ChEMBL (atom-type 99.91 %, partial-charge 99.97 %, term failure 22 ppm).
- __GPU MMFF atom typer + charge kernel__ — 1 045 580 atoms byte-identical vs CPU; 439 / 20 K molecules fall back.
- __GPU term extraction + CSR build__ — 941 / 1 009 test molecules match; remainder dispatched to CPU.
- __SMILES parse with `removeHs=false`__ — skipped the `removeHs` + re-`addHs` round-trip.
  Saved ~6 % of CPU prep wall time.
- __Scaffold cache__ — caches torsion-pattern match results by Bemis-Murcko scaffold.  +44 % throughput on curated ChEMBL (4 698 unique scaffolds across 237 600 cyclic molecules, 98 % reuse).
  Lower hit on diverse corpora.
- __Per-wave per-device `cudaEvent_t` instrumentation__ — enables the per-chunk breakdown in `BENCH.md`.
- __Molecule-sort by atom count + output permutation__ — dispatches smaller molecules first, permutes back to input order at finalize.
  Improves `atom-limited-bin` fill.
- __Pre-warm cusolver + metric scratch in `context_t` ctor__ — avoids first-wave allocation stall.
- __Pre-grow 256 MB device / host pools per device__ — arena claimed at construction.
- __Four redundant `__syncwarp` removals__ on lane-owned access paths (2026-04-18).

## Optimizations Tried and Rejected

- __Gradient register tile__ — moved FIRE gradient array from shared to per-lane registers to attack `long_scoreboard` stalls.  ncu showed stalls came from positions/velocities reads, not gradient.
  Registers rose 116 → 146 with no occupancy win.
  Reverted.
- __`__launch_bounds__` bump from 32,4 to 32,6__ — inert because nvcc already fit 8 blocks/SM from the shared-memory cap.
- __Register cap on the monolithic kernel__ — `--maxrregcount 128` caused local-memory spills that cancelled the occupancy gain.
  Only viable after the kernel split, which is now landed.
- __Wave-bin retune__ — merging bins 97–112 into 128 would cost ~30 % more compute on those molecules; dropping bin ≤16 would save ~40 ms (covers only 0.6 % of ChEMBL).
  Not worth churn.
- __Tensor cores, FP16 coordinates, larger chunks__ — rejected by ncu data.
  Math pipes idle at 0.2 %; grid already fills the device in production.
- __Native CIP chirality ranker__ — two attempts reached 98.94 % and 99.71 % agreement with RDKit, short of the ≥ 99.95 % threshold.
  Second attempt regressed P50 energy (−6.3 vs −5.4 kcal/mol) and −2.8 % throughput.
  Shelved until edge cases around embedded double-bond systems are resolved upstream.
- __Branchless `match_linear_pattern` lookup__ — no viable optimization target after profiling.
  Closed.
- __`__noinline__` on `fire_minimize_`__ — increased register pressure to 207, dropping theoretical occupancy to 16.67 %.  nvcc re-inlined when the hint was removed; compiler default was already optimal.

## Parked (not on roadmap)

- __MMFF parameter static-LUT replacement__ — at 10M scale MMFF-related CPU is ~8 % total; a 66 % reduction on per-term lookup cost gives ≤ 3 % wall.
  The existing `thread_local std::unordered_map` cache is already lock-free; the proposed static-array replacement would add a dep (`tsl::robin_map` or similar) and complexity for modest gain.
  The old 23 % MMFF-lookup figure in earlier drafts reflected pre-cache measurement; the cache already captured the bulk of the win.
- __Further `__syncwarp` removals__ — four clearly-redundant sites landed on 2026-04-18.
  The remaining 13 sites are load-bearing (guard cross-lane reads in shared memory).
- __Bounds matrix int4 nibble-packing__ — marginal gain per author's prior analysis; precision risk on bonded-pair windows.

## Billion-scale workloads

The CLI already streams shard-by-shard: `reader-role` reads one `parquet-shard` at a time, accumulates into a `coprocessed-batch`, `gpu-runner-role` runs its internal `embedding-wave` pipeline, `fallback-role` handles unresolved conformers, `writer-role` flushes an output `parquet-shard`.
ROMols are released per `embedding-wave`, output per `parquet-shard` — nothing scales with total input size.

Remaining gaps at 1–10 B molecule scale are captured in the __Dataset roadmap__ above (scaffold-aware sharding, adaptive cache eviction, direct cloud input).

Projected wall time at 10K mol/s host-prep ceiling (no optimisations applied beyond what's landed):

| Library | Molecules | Wall estimate |
| :--- | ---: | ---: |
| Enamine REAL | 1.2 B | ~33 hr |
| GDB-13 | 1.0 B | ~27 hr |
| REAL Space tier 2 | 7.0 B | ~195 hr |
| ZINC22 | 37.0 B | weeks |

Scaffold Cache internals (for reference): 64-shard `tsl::robin_map<uint64_t, scaffold_cache_entry_t>` at ~5 KB per entry, keyed by Bemis-Murcko hash.
On curated ChEMBL (4 698 unique scaffolds across 237 600 cyclic molecules, 98 % reuse) adds +44 % throughput.
Long-tail libraries show lower hit rates — that's what the adaptive-eviction + scaffold-aware-sharding gaps in the Dataset roadmap address.
