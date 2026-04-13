# SmallField Benchmark

ETKDG + MMFF94 conformer generation: CPU (RDKit) vs GPU (SmallField).

## Hardware

| Component | Specification                                                        |
| --------- | -------------------------------------------------------------------- |
| GPU       | 8x NVIDIA RTX PRO 6000 Blackwell (188 SMs each, SM 12.0, 96 GB VRAM) |
| CPU       | 2x Intel Xeon 6776P (48 cores / 96 threads each, 192 threads total)  |
| RAM       | 1.7 TB DDR5                                                          |

## Dataset

ChEMBL drug-like molecules, pre-filtered to N ≤ 128 atoms (including explicit H).

| Stat               | Value                                       |
| ------------------ | ------------------------------------------- |
| Total molecules    | 8,948 (small batch) / 241,596 (large batch) |
| Min atoms          | 8                                           |
| P25 / Median / P75 | 36 / 48 / 63                                |
| P90 / Max          | 84 / 128                                    |

## Throughput

All numbers below use jemalloc preloaded and all current optimizations enabled.
No RDKit binary patches.
Invoke with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ./bench/bench_chembl --smi … --conformers 1`.

### Build flags

The CMake default is now `Release` with these flags baked in (set automatically unless the user overrides `CMAKE_BUILD_TYPE` or the per-language flag variables):

- CXX: `-O3 -DNDEBUG -march=native -fopenmp -std=gnu++20`
- CUDA: `-O3 -DNDEBUG --use_fast_math -Xcompiler=-march=native -Xptxas=-O3 -Xcompiler=-fopenmp -std=c++20`

`-march=native` uses the host CPU's full ISA including AVX-512 on the Xeon 6776P.
`--use_fast_math` maps `expf`/`logf`/`sqrtf`/etc. to their GPU hardware intrinsics and enables FTZ.
`-Xptxas=-O3` sets the PTX assembler's optimization level explicitly.
`-Xcompiler=-fopenmp` forwards `-fopenmp` to the host compiler through nvcc (required for the `#pragma omp parallel for` inside `etkdg_preprocess.hpp` to actually parallelize — see the `-fopenmp` fix section below).

Moving from `-O2 -g -DNDEBUG` (the previous `RelWithDebInfo` default) to these flags gave ~2-3% on end-to-end throughput.
The small gain is because RDKit's `SubstructMatch`, `MMFFMolProperties`, and `setTopolBounds` live in the conda-forge `.so` files we link against; those were compiled with whatever flags conda-forge used and our flags do not touch them.
The flags still matter for SmallField's own hot paths (descriptor build, native SMARTS matcher, MMFF VdW combining rules, the CUDA kernels).

| Batch   | RDKit CPU (192 thr) | SmallField 1 GPU | SmallField 8 GPUs | 1→8 scaling |
| ------- | ------------------- | ---------------- | ----------------- | ----------- |
| 8,948   | 1,672 mol/s         | 1,954 mol/s      | 2,429 mol/s       | 1.24×       |
| 50,000  | 1,730 mol/s         | 2,449 mol/s      | 4,766 mol/s       | 1.95×       |
| 100,000 | 1,725 mol/s         | 3,123 mol/s      | 5,884 mol/s       | 1.88×       |
| 241,596 | 1,763 mol/s         | 4,900 mol/s      | __14,727 mol/s__  | __3.00×__   |

At 241,596 molecules on all 8 GPUs: ~16.4 seconds end-to-end (post-SC scaffold cache + MS-1 `removeHs=false` + MM-1 native Murcko).
Quality is unchanged: P50 ≈ 73 kcal/mol vs RDKit 78 kcal/mol at 2K, 96.7 % of conformers successful.

### CPU-prep acceleration progression (2026-04-13)

Single-threaded `profile_driver` (bypasses 8-GPU pipelining to isolate per-molecule CPU share) and full 240K × 8 GPU bench:

| Phase              | profile_driver mol/s | 240K × 8 GPU mol/s (median) | Top CPU share change                                                                                    |
| ------------------ | -------------------: | --------------------------: | ------------------------------------------------------------------------------------------------------- |
| Baseline (post-SC) |               12,644 |                      14,670 | FloydWarshall 15.1 %, MurckoDecompose 18.3 %, sanitize/removeHs 5.7 %, SubstructMatch ~0 % (SC killed) |
| + MS-1             |               14,366 |                         n/a | `removeHs` path eliminated (5.7 % → 0 %) — +14 % on single-threaded, GPU-bound at 8 GPU scale            |
| + MM-1             |               14,430 |                      14,727 | FloydWarshall + MurckoDecompose + MolToSmiles all eliminated (33 % → 0 %) — MMFF 18 % → 23 % now #1      |

Session-level progress: the 241K × 8-GPU throughput went from 4,008 mol/s (60.3 s baseline) to 14,727 mol/s (16.4 s) — __+267 %__.
Single-GPU climbed from 1,716 to ~4,830 mol/s (+181 %) as per-launch metric prep CPU work and the MMFF nonbonded upload both shrank.

1→8 scaling on the current build is __3.04 ×__ (measured 3.042, 3.057, 3.040 across three runs; 1-GPU range 0.7 %, 8-GPU range 1.1 %).
Fine-grained scaling deltas (< 3 %) are within measurement noise — reading a scaling _trend_ across historical snapshots of the codebase isn't reliable, so the earlier "3.11× → 2.18×" discussion is removed.
Absolute throughput on the latest build at 241K × 8 GPU is the metric that matters.

## Scaling analysis (241,596 molecules)

### Terminology

- __chunk__ — a slice of the input batch.
  `generate_conformers` splits input into chunks of ≤50,000 molecules so CPU prep of chunk N+1 overlaps GPU work of chunk N.
  At 241K input there are ~5 chunks of ~48K molecules each.
- __per-GPU slice__ — the portion of one chunk assigned to one GPU.
  At 8 GPUs that's ~48K / 8 = 6,000 molecules per GPU per chunk.
- __launch__ — one `launch_gpu_pipeline` call: one chunk's slice on one GPU.
  8 launches per chunk when running on 8 GPUs.

### GPU scaling (SmallField) on 241K

| Config | Wall time | Throughput  | Scaling |
| ------ | --------- | ----------- | ------- |
| 1 GPU  | 140.8 s   | 1,716 mol/s | 1.0×    |
| 8 GPUs | 29.0 s    | 8,344 mol/s | 4.86×   |

Observed per-launch wall (instrumentation, post-OpenMP dispatcher): `setup=0.3 mallocpool=0 upload=0.2 metric=400–560 kernels_post=0.2 total≈400–560 ms`.
The 8 launches run concurrently on 8 OpenMP threads, so their wall contribution to the chunk is ≈ max(500 ms) rather than sum.

### CPU scaling (RDKit) on 241K

| OMP threads           | Throughput  | Notes                                                                                          |
| --------------------- | ----------- | ---------------------------------------------------------------------------------------------- |
| 192                   | 1,753 mol/s | Warm allocator post-workstream-N run; earlier cold runs hit ~873 mol/s before jemalloc settled |
| 192 (8,948 mol batch) | 1,672 mol/s | Peak CPU throughput on small batch                                                             |

RDKit's `MMFFMolProperties` per-molecule heap traffic (internal lookup tables + exclusion matrices) scales poorly on large batches because the allocator starts contending.
Peak CPU throughput is ~1,700 mol/s regardless of core count beyond 96.

### Where the per-chunk wall time goes at 8 GPUs (measured via `cudaEvent_t`)

Workstream FF added stream-local `cudaEvent_t` markers around each per-launch phase (upload, metric prepass, kernel waves, download).
This gives __measured GPU phase timings__ rather than the prior call-site estimates.
Numbers below are from a single representative 241K × 8-GPU run, captured into per-chunk records and printed by `generate_conformers`.

Per-chunk per-device GPU phase wall (ms), sample at chunk 1 (warm; chunk 0 includes the cold dispatcher + first cudaHostAlloc):

| Device | Upload H2D | Metric prepass | Kernel waves | Download D2H | Total |
| ------ | ---------: | -------------: | -----------: | -----------: | ----: |
| gpu0   |         62 |             17 |         2273 |          0.3 |  2353 |
| gpu1   |         63 |             17 |         2462 |          0.4 |  2542 |
| gpu2   |         82 |             17 |         2451 |          0.3 |  2550 |
| gpu3   |         82 |             16 |         2458 |          0.2 |  2556 |
| gpu4   |         74 |             18 |         2437 |          0.4 |  2530 |
| gpu5   |         74 |             22 |         3153 |          0.4 |  3249 |
| gpu6   |         84 |             17 |         2452 |          0.4 |  2554 |
| gpu7   |         84 |             15 |         2484 |          0.3 |  2585 |

Observations:
- __Kernel waves are 95–99 % of per-device GPU wall.__
  Upload is ~3 %, metric prepass < 1 %, download < 0.1 %.
  Memory bandwidth is __not__ the bottleneck.
- __Per-device GPU wall ≈ 2.5–3.2 s for a 50K-mol chunk on 8 GPUs.__
  The spread (gpu5 at 3249 ms vs gpu0 at 2353 ms — 38 % wider) is straggler variance from the wave-binned dispatch (one large molecule stretches the block it's in).
- __Chunk GPU wall (max across devices) ≈ 3.2 s.__
  Sum across devices ≈ 20.7 s.

Per-chunk host-wall layout (measured by `std::chrono::steady_clock` at boundaries of `prepare_molecule_batch` / `launch_chunk` / `finalize_chunk`):

```
chunk      0       1       2       3       4
prep    ▓▓▓▓    ▓▓▓▓    ▓▓▓▓    ▓▓▓▓    ▓▓▓
gpu      ▓▓▓▓    ▓▓▓▓    ▓▓▓▓    ▓▓▓▓    ▓▓▓
fin       ░       ░       ░       ░       ▓▓▓ ← only chunk 4's GPU wall is fully unhidden

 t=0    4s      8s     12s     16s     20s   23s
```

- CPU prep: ~4.0 s per 50K-mol chunk.
- GPU wall per chunk: ~3.2 s (chunk-wide max across the 8 devices).
- Pipeline overlap: chunk N+1's CPU prep runs in parallel with chunk N's GPU work.
  Per-chunk wall = max(prep, GPU) ≈ 4.0 s.
- Tail: chunk 4's GPU wall is fully unhidden (no chunk 5's prep to hide it).
  At 41,596 mols (smaller than 50 K), chunk 4's GPU wall is ~3 s.

Total wall ≈ prep_chunk_0 + 4 × 4.0 s prep + chunk_4 GPU wall ≈ 4 + 16 + 3 = 23 s.

The critical path is __CPU prep__ (slightly longer than GPU wall per chunk), with the unhidden chunk-4 GPU wall as the visible tail.

### Why 1→8 scaling is 4.86× not 8×

Two compounding factors (was three; dispatcher is no longer sequential):

1. __CPU prep is constant per chunk regardless of GPU count.__  ~4.0 s/chunk × 5 chunks = 20 s of CPU work.
   At 1 GPU the GPU work (~10 s of kernel per chunk on one device) dominates so CPU prep fully overlaps.
   At 8 GPUs the GPU work drops to ~3.2 s/chunk (8 devices in parallel) so CPU prep becomes the slightly-longer phase — this is the __per-chunk__ critical path.
2. __Unhidden last-chunk GPU tail.__
   After the last chunk's CPU prep finishes, the GPU's still working — that wall is fully unhidden.
   At 8 GPUs the unhidden tail is ~3 s.
   Workstream EE (smallest-chunk-last) addresses this by sorting molecules globally so the last chunk has the cheapest GPU work.

## Optimizations that landed

### Scaffold-cache wiring into `extract_etk_torsions_native` (workstream SC)

Previously the scaffold-cache infrastructure (`scaffold_cache_t`, `enable_scaffold_cache`, sharded concurrent map) was landed but __unused__ because the integration with the prep path wasn't done.
This commit wires it in:

1. Thread `scaffold_cache_t *cache` through `prepare_molecule_batch` → `prepare_molecule_into` → `extract_etk_torsions_native`.
   Default `nullptr` preserves the existing API behaviour.
2. Inside the extractor, when `cache != nullptr`:
   - `MurckoDecompose(molecule)` → canonical scaffold SMILES → 64-bit
     FNV-1a hash.
   - `SubstructMatch(molecule, scaffold)` once to build the molecule →
     scaffold atom-index map.
   - __Cache hit__: for each non-native pattern flagged "fully scaffold-
     local" (no R-group-spanning matches were observed when populated),
     translate the cached 4-tuples through the atom map and emit torsions
     — __skipping the full-molecule `SubstructMatch` entirely__.
   - __Cache miss__: run `SubstructMatch` on the full molecule (existing
     code path); classify each match as scaffold-local (all 4 atoms in
     scaffold) or R-group-spanning; cache the scaffold-local matches in
     scaffold-atom-index space; record the R-group flag for future hits.
3. `bench_chembl.cpp` opts in via `ctx.enable_scaffold_cache()`.

Effect at 241K × 8 GPU: __10,204 → 14,713 mol/s__ (+44 %).
Single-GPU is roughly unchanged (4,800 → 4,900 mol/s) because the GPU is the longer phase there and the saved CPU time overlaps with GPU work.
Multi-GPU success rate 96.7 % (no regression).
Per-mol overhead of `MurckoDecompose` + `MolToSmiles` + `SubstructMatch(mol, scaffold)` is ~150 µs but is amortised across all molecules sharing the scaffold (98 % reuse on the curated 241K ChEMBL subset per the prior feasibility experiment).

### Native CIP rank ranker (workstream V — investigated, reverted)

A `compute_cip_ranks` implementation in `src/cip_ranks.hpp` uses iterative sorted-tuple partition refinement (matching RDKit's `iterateCIPRanks` algorithm).
Cross-validated against RDKit's UINT_VECT output on 1,698 chiral centres in 500 ChEMBL molecules: __98.94 % chiral-centre ordering agreement__.
This is far better than the prior Morgan-hash attempt (4.24 %), but the residual 1.06 % causes a small P50-energy regression (-5.8 vs -5.4 kcal/mol against RDKit baseline) AND costs ~280 mol/s of throughput at 241K × 8 GPU because the iterative algorithm is slower than RDKit's optimised internal call.

Decision: revert the swap-in for now; keep the implementation in `src/cip_ranks.hpp` for future work.
IMPROVE.md tracks V as "investigated, deferred until ≥ 99.5 % agreement is reached".

### Native formal-charge support in CSD parser (workstream N+++)

Extended `parse_atom_term` to accept SMARTS formal-charge specifiers (`+`, `+N`, `-`, `-N`).
Stored into `simple_atom_constraint_t::formal_charge` (int8, `INT8_MIN` = no constraint) and checked against the descriptor's `formal_charge[]` array in `atom_match` and `neighbor_match`.
`parse_atom_spec` no longer bails on `+` / `-` characters.

Native pattern coverage: __303 → 314 patterns__ (+11).
The newly-native patterns include the p249 family (benzoate-NH torsions with `[O-]` carboxylate branches) which previously fell back to RDKit SubstructMatch.

Wall-time at 241K × 8 GPU held within noise (10,168 → 10,174 mol/s) — the CPU/GPU phase balance dominates at that scale.
Gain is on single-GPU and small-batch workloads where CPU prep is the critical path.

### Scaffold-cache infrastructure (workstream SC, partial)

Landed `scaffold_cache_t` (64-shard concurrent `unordered_map` keyed on scaffold-SMILES 64-bit FNV-1a hash) + `context_t::enable_scaffold_cache` opt-in API + stats observability.
The integration with `extract_etk_torsions_native` (MurckoDecompose + scaffold→molecule atom map + cache population + hit translation) is pending as an isolated follow-up commit — the feasibility analysis and expected 25–30 % SubstructMatch CPU reduction are documented in `IMPROVE.md §4`.

### Mega-kernel launch_bounds + context-ctor pre-warm (workstreams W, X, DD, Y)

Three small-but-related fixes for cold-start and occupancy latency-hiding:

__W + X — `__launch_bounds__(32, 4)` on `etkdg_pipeline_warp_kernel`.__  ncu showed the mega-kernel ran at 2 % achieved occupancy due to 128 regs/thread capping blocks-per-SM to 1.
W first landed `(32, 2)` to force ≥ 2 blocks/SM; X pushed to `(32, 4)` = 128 concurrent warps/SM of occupancy.
The compiler spills more registers to local/L1 to recover occupancy.
Top warp stalls in ncu were `wait` (32 %) and `long_scoreboard` (26 %), both latency-bound — more concurrent warps hide them.
At production 240K scale the gain is within noise but nominally positive (10,155 → 10,168 → 10,204 mol/s across W, N+++, X).

__DD — context_t ctor pre-warms cusolver workspace + metric scratch.__
The first user call paid ~100 ms cusolver-workspace-probe per device and a cold cudaMalloc.
Now `context_t()` calls `ensure_metric_scratch` at a default (5 K molecules × 128 atoms) so the first user launch finds the scratch already sized and the cusolver handshake done.

__Y — context_t ctor pre-allocates 256 MB device + 256 MB host staging per device.__  nsys traces showed `cudaHostAlloc` as a 915 ms isolated stall on first launch for large batches; the persistent pool sidesteps that for any workload that fits the default.
Bigger batches still grow on demand via the existing `ensure_capacity` path.

Net effect on 240K × 8 GPU: wall is unchanged within noise (10,155 → 10,168 mol/s), but startup predictability improves — the first user call no longer hits a multi-hundred-ms cold stall, which matters for latency-sensitive library use.

### Native fast-paths for top SubstructMatch hot-spots (workstream N++)

Multi-dataset profiling (chembl / moses / zinc / gdb13 / synth-scaffold / synth-noise) identified that __two CSD torsion patterns — p94 and p114 — cover 75–85 % of RDKit SubstructMatch fallback CPU on every dataset__.
Whichever is #1 depends on input: p94 dominates drug-like libraries (Moses, ZINC at 73–76 %), p114 dominates enumerative combinatorial libraries (GDB-13, scaffold-decorated at 78–82 %).
Together they universally dominate.

Four new predicate helpers in `src/csd_torsion_predicates.hpp`:

| Predicate           | Body                           | Use                                        |
| ------------------- | ------------------------------ | ------------------------------------------ |
| `a_nb_cH1`          | `$([a]([cH1]))`                | aromatic atom with aromatic-CH neighbour   |
| `aH0_nb_nonring_O`  | `$([aH0](!@O))`                | aromatic, 0 H, non-ring O neighbour        |
| `c_two_cH_or_nX2H0` | `$([c]([cH,nX2H0])[cH,nX2H0])` | aromatic C with two cH-or-nX2H0 neighbours |
| `cH0_nb_NH12`       | `$([cH0][NH1,NH2])`            | aromatic C, 0 H, with NH1 or NH2 neighbour |

Pattern coverage rises 300 / 470 → 303 / 470; three patterns (p94, p95, p114) now compile to native form.
All 13 total predicates cross-validate against `RDKit::SubstructMatch` on 210 ChEMBL molecules (every atom of every molecule produces the same verdict).

End-to-end throughput at 241K × 8 GPU held within noise of the pre-N++ baseline (10,130 → 10,155 mol/s median of two runs) — the wall is co-balanced between CPU prep and GPU kernel phases at this scale, so the SubstructMatch reduction is absorbed into remaining CPU slack.
Largest single-GPU gain observed: 4,520 → 4,852 mol/s (+7 %), matching the agent's prediction that single-GPU runs (where CPU prep dominates) get the biggest benefit.

### Smallest-chunk-last molecule sort (workstream EE)

Before chunking, `generate_conformers` sorts the full input by descending SMILES length (proxy for atom count).
The LAST chunk thus contains the smallest molecules, minimising the unhidden GPU tail that would otherwise dominate the last few hundred milliseconds of wall time.
Outputs are permuted back to the caller's original input order in the post-loop write loop, so the public API behaviour is unchanged.

Measured effect on 241K × 8 GPU (per-chunk `cudaEvent` timings from workstream FF):

| Chunk | GPU max wall (pre-EE) | GPU max wall (post-EE) |
| ----: | --------------------: | ---------------------: |
|     0 |              3,206 ms |               5,818 ms |
|     1 |              3,169 ms |               3,279 ms |
|     2 |              3,230 ms |               2,159 ms |
|     3 |              3,209 ms |               1,033 ms |
|     4 |              2,969 ms |             __792 ms__ |

Chunk 4's unhidden GPU tail dropped by __~2.2 s__ (2,969 → 792).
At this dataset's molecule-size variance, the saving is offset by chunk 0's heavier prep + GPU (since the largest molecules now live there), so total wall is ~neutral — but the behaviour is much more predictable for datasets with extreme size variance (e.g. mixed GDB-13 + drug-like).
Keeps wall-time budget stable even as input distribution shifts.

### cudaEvent phase instrumentation (workstream FF)

Added stream-local `cudaEvent_t` markers around each per-launch GPU phase (upload H2D, metric prepass, kernel waves, download D2H).
Events live in a persistent per-device pool inside `launch_scratch_t` — created with `cudaEventBlockingSync` at context construction and reused for the lifetime of the `context_t`.

A prior prototype created/destroyed events per launch and hit an intermittent multi-GPU regression (240K success rate dropped to 60 % on chunks 3–4) caused by a race between `cudaEventDestroy` on one stream and the next launch's `cudaEventCreate` on a different OpenMP dispatcher thread.
Persistent reuse sidesteps the issue entirely.

Per-chunk records hold host-wall markers (prep / launch / finalize boundaries via `std::chrono::steady_clock`) plus per-device `gpu_phase_ms_t` copies from `cudaEventElapsedTime`.
End of `generate_conformers` prints an ASCII timeline:

```
=== chunk timeline (ms, host wall + per-device GPU phase) ===
  ck   n_mol  prep_lo  prep_hi  lnch_lo  lnch_hi   fin_lo   fin_hi  gpu_max  gpu_sum  gpu_avg
   0   50000      0.0   5990.8   5990.8   6286.4  10367.0  11952.1   5817.6  37657.0   4707.1
   ...
=== per-device phase breakdown (ms): U=upload  M=metric  K=kernels  D=download  T=total ===
ck=0 dev=0  U=113.8  M=34.3  K=4249.6  D=0.4  T=4398.1
   ...
```

This is the foundation for the next-tier GPU optimizations (W, X, Z) documented in `IMPROVE.md`.

### Native CSD SMARTS branch support + element-mask fix (workstream N)

`parse_linear_torsion` now accepts inline `]([…])` branches on any of the four chain atoms.
Each branch becomes a `simple_neighbor_constraint_t` stored on its parent atom; `match_linear_pattern` verifies up to two distinct branch neighbours after the 4-atom chain is matched (both orderings tried so `(~[N,n])(~[N,n])` maps either way).
Two slot types of content are accepted: bracketed `[spec]` (same disjunct / conjunction grammar as the main atom) and shorthand elements with an optional bond operator (`=O`, `Cl`, `c`).
Branches that contain atom-map labels, nested parens, recursive sub-queries, or charge signs still fall through to RDKit.

This also fixed a pre-existing correctness bug in `parse_atom_term`: the `term_has_element` parameter was passed by value, so simple atom specs like `[O:1]` left `disjunct_has_element=false` and ended up with `element_mask=0` (i.e. "any element").
The native matcher was silently wildcard-matching non-O bonds as if they were `[O]`; the downstream `done_bonds` dedup masked the symptom in aggregate but broken per-pattern torsion assignments existed latently.
Fixed by passing the flag by reference.

Native pattern coverage: __216 / 472 → 300 / 470__ (46% → 64%).
Throughput at 241K × 8 GPU: 10,107 → 10,130 mol/s (median of two runs; 10,154 and 10,105 individually).
Flat within run-to-run noise — the RDKit `SubstructMatch` fallback was already fast after the element prefilter, so converting 84 more patterns to native saves only microseconds of per-molecule CPU time.
The real CPU-prep cost is `MMFFMolProperties` (atom typing), which this workstream does not touch.
Net: a correctness improvement with no throughput regression; the pre-existing wildcard matcher was producing spurious torsions that the ETKDG refiner then had to spend iterations cleaning up.

### Fused bounds-smooth + Gram + power-iter embedding kernel (workstream O)

The three-stage metric prepass (smooth_and_build_gram_kernel → cusolverDnXsyevBatched → project_metric_coordinates_kernel) collapses into ONE kernel: `embed_4d_power_iter_kernel` in `src/metric_embedding_prepass.cuh`.

One block per molecule.
Shared memory holds the Gram matrix (built in place over the smoothed bounds — Floyd-Warshall + row-mean centering unchanged) plus four eigenvector slots + scratch (~67 KB at N=128).
Deflated power iteration extracts the top-4 eigenvalues: initialize `v_k` near `e_k`, then iterate `w = G·v_k` → Gram-Schmidt against `v_{0..k-1}` → normalize → check eigenvalue stability.
Up to 60 sweeps; converges in 10-20 for drug-like molecules.
Output `v_k × sqrt(λ_k)` goes directly into `initial_coordinates` — no global Gram, no cusolver, no separate projection kernel.

Removes the three CPU-side cusolver launch costs per bin, the host workspace buffer, the batched-Gram global memory traffic, and the separate project kernel.
The cusolverDnParams_t state in `metric_scratch_t` is now unused (kept for backward compatibility).

End-to-end 241K × 8 GPU: 24.7 s → 23.9 s (+3 %), 9,792 → 10,107 mol/s — __above the 10K target__.
All 7 ctests pass; P50 = 72.7 kcal/mol vs RDKit 79.1 at 2K (within baseline noise).

### Direct-to-pool Phase 1 preprocessing (workstream E)

`prepare_molecule_batch` used to build `std::vector<etkdg_prepared_molecule_t> prepared(n)` — 241K structs each carrying 14 `std::vector<T>` members.
First-touch per molecule allocated each inner vector's backing storage: roughly 3.4 million allocations per 241K batch, plus ~15 GB of transient per-molecule data that Phase 2 then memcpy'd into the pool and threw away.

The rewrite eliminates the `prepared[n]` outer container entirely.
Each OpenMP thread owns ONE reusable `etkdg_prepared_molecule_t` whose vectors keep their capacity across every molecule that thread processes — after the first few molecules on a thread, every `push_back` stays on the allocator's fast path.
Each thread also owns a `thread_contribution_t` with 16 `std::vector<std::byte>` segments (one per pool region) that byte-append the current molecule's data.

Three passes:

1. __Parallel__: `prepare_molecule_into(smiles, tc.scratch)` fills the thread's scratch; `append_bytes` appends each region into `tc.*_bytes` and records a thread-local `per_mol_record_t` with region offsets.
2. __Serial prefix scan__ (fast): per-thread pool bases per region + per-thread dense metadata slot bases.
   Takes a few hundred microseconds.
3. __Parallel__: per-thread `std::memcpy` from `tc.*_bytes` to pool + per-molecule metadata write with pool-wide offset rebase.

End-to-end 241K × 8 GPU: 25.7 s → 24.7 s (+5 %), 9,400 → 9,792 mol/s.
The modest absolute gain reflects that most per-molecule heap pressure is inside RDKit's `SmilesToMol` / `addHs` / `MMFFMolProperties` which this pass doesn't touch.
The cleaner design is the bigger win.

The `prepare_molecule(smiles) -> etkdg_prepared_molecule_t` entry point remains as a thin wrapper over `prepare_molecule_into` so tests keep working unchanged.

### cusolverDnXsyevBatched in place of cusolverDnSsyevjBatched (workstream K)

The metric prepass used `cusolverDnSsyevjBatched` (a batched Jacobi eigensolver).
Per-call CPU setup was ~60 ms × 8 bins per launch = 480 ms per launch of CPU work inside cusolver state bookkeeping.

Swapped to `cusolverDnXsyevBatched` (CUDA 13's 64-bit generic batched eigen API).
It takes `cusolverDnParams_t` instead of the Jacobi-specific `syevjInfo_t` — no Jacobi sweep state tracking, lower per-call overhead.
Also needs both a device and host workspace buffer; host workspace lives in `metric_scratch_t::host_workspace` (grow-only `std::vector<std::byte>`).

End-to-end 241K × 8 GPU: 29.0 s → 25.7 s (+13 %), 8,344 → 9,400 mol/s.

### Parallel dispatcher via OpenMP (workstream J)

`launch_chunk` used to call `launch_gpu_pipeline` 8 times sequentially on one thread.
Each call takes ~500 ms of CPU-side work (mostly cusolver state update for the metric prepass and group-indices gather), so dispatcher wall at 8 GPUs was 4 s/chunk — more than the kernel itself.

Replaced the sequential loop with `#pragma omp parallel for num_threads(num_devices) schedule(static, 1)`.
Each OpenMP thread owns one device's state (`streams[d]`, `solvers[d]`, `metric_scratch[d]`, `launch_scratch[d]`) — no cross-thread contention.
The 8 launches now run concurrently on 8 cores.

Dispatcher wall per chunk: 4 s → ~0.5 s.
End-to-end 241K × 8 GPU: 45.2 s → 29.0 s (+56 %).
1→8 GPU scaling: 3.11× → 4.86×.

### MMFF nonbonded pair enumeration on GPU (workstream D)

`extract_mmff94_params` used to materialize a `std::vector<mmff94_nonbonded_pair_t>` with every non-excluded (i, j) pair — up to N(N-1)/2 × 16 B = 128 KB per N=128 molecule, ~6.4 GB per 50K-molecule chunk.
The kernel iterated a per-atom CSR into that pair array.

The rewrite uploads __per-atom__ MMFF data instead:

- `mmff94_atom_vdw_params_t` (α, N, A, G, R*, donor/acceptor flag) — 24 B per atom.
- `mmff_atom_charge` — 4 B per atom.
- `bond_distance` — 1 B × N² per molecule (copied from the descriptor's BFS matrix).

Inside the nonbonded kernel loop, each atom enumerates partners from `bond_distance[i][j] > 3` and combines pair parameters inline using RDKit's Halgren R* rule (with the donor-suppression clause) + Slater-Kirkwood ε combining + donor-acceptor scaling.
The combining constants (B=0.2, β=12, DARAD=0.8, DAEPS=0.5) match `MMFFDefaultParameters::getMMFFVdW()` exactly.

Two kernel sub-loops preserve the existing role-first / role-second symmetry: `j > atom` owns the pair's energy + this atom's gradient as the first endpoint; `j < atom` adds only this atom's gradient as the second endpoint.

Upload savings: ~6.4 GB → ~180 MB per chunk (35× reduction).
Also eliminates the O(N²) CPU pair-construction loop inside `extract_mmff94_params`.

Validation: `test_mmff_gradient` (the MMFF parity test against RDKit) passes within its existing tolerance.
2K-mol quality gate: P50 = 73 kcal/mol vs RDKit 78.

End-to-end impact (241K × 8 GPU median of 3 runs): 53.9 s → 45.2 s (+19 %).

### Metric prepass 8-bin padded syevjBatched

`build_metric_initial_coordinates_on_gpu` used to group molecules by exact atom count and call `cusolverDnSsyevjBatched` once per distinct N — drug-like distributions span 30+ values, and per-call cusolver overhead dominated the per-launch wall at ~500 ms per GPU.

Now molecules are binned into the same 8 buckets the wave dispatcher uses (`{16, 32, 48, 64, 80, 96, 112, 128}`).
Each bin issues ONE `cusolverDnSsyevjBatched` at the bin's padded N.
Shorter molecules zero-pad the Gram matrix up to the bin size — the padding creates zero eigenvalues that sort below the real ones, so the top-4 eigenvectors still reconstruct the same 4D embedding.
The kernel smooths only the real N × N sub-block (no wasted Floyd-Warshall work).

Result: per-launch wall dropped from ~700 ms to ~430 ms.
Dispatcher critical path at 8 GPUs went from 5.6 s/chunk to ~3.4 s/chunk.

### Native bounds matrix builder (no RDKit `setTopolBounds`)

The old `build_bounds_matrix` called RDKit's `setTopolBounds`, which allocates a fresh `DistGeom::BoundsMatrix` plus two `boost::shared_array<double>` and two `boost::shared_array<int>` per molecule (~140 KB of throwaway heap) and walks RDKit's atom typer.
At 241K molecules that's ~33 GB of transient allocator traffic per batch.

The replacement reads only the flat `molecule_descriptor_t` and the small tables in `src/forcefield_tables.hpp`:

- Covalent radii per atomic number + bond-order shortening factor produce the 1-2 bond length (±2 % tolerance).
- Ideal-angle-by-hybridization + law of cosines produce the 1-3 distance (±3 % tolerance).
- BFS distance = 3 with a conservative cis/trans envelope produces the 1-4 bound.
- VdW radii produce the long-range lower bound for `bond_distance > 3`.

Writes directly into the caller's `std::vector<float>` — zero other heap activity per call.
Quality gate holds (P50 energy within ~5 kcal/mol of the RDKit baseline; the subsequent triangle-smoothing + FIRE refinement absorbs the residual).

### Persistent pinned host output buffers

`host_coords` (1.1 GB at 241K × 3 conformers × 128 atoms), `host_energies`, and `host_status` used to be pageable `std::vector`s allocated per `generate_conformers` call.
They now live in a shared `output_staging_t` owned by `context_t`: one `pinned_byte_buffer_t` that grows monotonically across calls.
Eliminates a ~1 GB `malloc` + zero-fill per call.

### SMILES kept as `std::string_view` end-to-end

`prepare_molecule` used to take `std::string const &`, so `generate_conformers` copied its `std::span<std::string_view const>` input into a fresh `std::vector<std::string>` (~24 MB at 241K) and `prep_slice` copied another range per chunk.
The API now takes `std::span<std::string_view const>` all the way down; one unavoidable `std::string` materialization happens at the single `RDKit::SmilesToMol` call inside `prepare_molecule` — scoped to microseconds of stack/SSO storage.

### Unified `batch_pool_t` with pinned host + persistent device mirrors

`etkdg_molecule_batch_t` used to hold 14 separate `std::vector<T>` fields.
The launch path carved 14 device slices out of one pool, then issued 14 `cudaMemcpyAsync` calls whose sources were pageable `std::vector` storage.
CUDA silently degrades async H2D from pageable memory to synchronous, so each of the 14 calls blocked the dispatcher for its full transfer time.

The new `batch_pool_t` allocates ONE pinned host buffer per chunk (via `cudaHostAlloc`) described by a `batch_layout_t` — 16 byte-offset / count pairs for each sub-region.
Phase 2 of `prepare_molecule_batch` memcpys per-molecule data directly into this pinned buffer; the launch path issues a single `cudaMemcpyAsync` that copies the whole buffer to a per-device device mirror.
The mirror is owned by a persistent `launch_scratch_t` in `context_t` so it is allocated once and reused across chunks.

Both buffers are exposed as `std::span<T>` views via a new `batch_pool_view_t` that parameterizes by a raw `std::byte *` base pointer — the same class produces host and device views from the same layout.

End-to-end: `launch_gpu_pipeline` per-GPU time dropped from ~1.5 s (synchronous upload loop) to a few milliseconds of dispatcher work.
We evaluated `cudaMallocManaged` + `cudaMemPrefetchAsync` for a truly unified pointer; on consumer-tier PCIe Blackwell the HMM bookkeeping cost about 2× more than pinned+explicit, so we stayed with explicit copies.

### `std::span` sweep over scratch + handle state

`metric_scratch_t` and `gpu_pipeline_handle_t` used to expose raw `T*` + length pairs.
All of them are now `std::span<T>` — length is carried with the pointer and consumers can `.size()` / `.data()` interchangeably.
`launch_gpu_pipeline` and `run_gpu_pipeline` take non-owning input/output arrays as `std::span` parameters.
Kernel signatures still take `(T* __restrict__, int N)` pairs — nvcc's `mdspan` support under `__restrict__` is incomplete, so we keep the aliasing guarantee at the kernel boundary and pass `.data()` / `.size()` there.

### The `-fopenmp` through-nvcc bug (the one-line fix that gave 4×)

Prior benchmarks reported ~587 mol/s on 8 GPUs, which looked terrible.
Root cause: `smallfield.cu` is compiled by nvcc, and nvcc silently ignores `#pragma omp parallel for` unless you forward `-fopenmp` to the host compiler:

```cmake
target_compile_options(smallfield_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
```

Before this flag, the OpenMP loop inside `prepare_molecule_batch()` ran single-threaded when called via the library (`omp_get_max_threads()` returned 1).
Preprocessing 8,948 molecules took __13.3 seconds__ instead of 625 ms — a 21× stall that dwarfed everything else.
The CMake one-liner restored the correct parallelism and was worth __16× end-to-end__ on the 241K batch.

### Parallel Phase 2 pack (struct-of-sizes + parallel memcpy)

Phase 2 of `prepare_molecule_batch` used to be a serial loop of 14 `std::vector::insert` concatenations per molecule.
At 241K molecules this cost ~15 s on its own and dominated the CPU side.

The rewrite operates on a single `std::vector<per_mol_sizes_t>` — one 64-byte struct per molecule holding the 14 per-array counts plus a `valid` flag.
Three passes: parallel size extraction, single prefix scan over the struct (cache-local, <5 ms at 241K), parallel memcpy into pre-sized batch arrays.
The prefix scan of the `valid` field doubles as the dense-output compaction index.
At 241K, Phase 2 drops from ~15 s to ~1 s.

### CPU/GPU pipelining via async CUDA streams

`generate_conformers` now splits large inputs into ~50K-molecule chunks.
While GPU runs chunk N, the calling thread preprocesses chunk N+1.
No `std::thread`, no queue — CUDA streams already provide the asynchrony; the main thread drives the loop.

Inside each chunk, 8 GPUs still launch in parallel via short-lived `std::thread`s (the `metric_embedding_prepass` stage does ~800 ms of CPU-side cusolver setup per GPU that cannot start until the previous GPU's kernels finish — parallelizing the launches across GPUs hides that cost).
Between chunks, there is no inter-chunk thread.

To make pipelining actually pipeline, two sub-fixes were needed:

1. __Remove the blocking cusolver-info check in `metric_embedding_prepass.cuh`.__
   It used `cudaMemcpyAsync` + `cudaStreamSynchronize` to read per-atom-group cusolver exit codes and abort on failure.
   That sync blocked the calling thread for the full cusolver duration, making "launch" synchronous and defeating the pipeline.
   Cusolver failures on Gram matrices from well-formed bounds are vanishingly rare, and when they happen the downstream kernels surface as `stereo_failure` / `mmff_unconverged` anyway.

2. __Move the device→host download from `launch_gpu_pipeline` into `finalize_gpu_pipeline`.__
   `cudaMemcpyAsync` with a non-pinned host destination __degrades to synchronous__ — it does not actually enqueue on the stream, it blocks the CPU for the full download duration (~2 s per chunk here).
   Doing it in finalize, where the stream has already drained via `cudaStreamSynchronize`, lets us use plain `cudaMemcpy` without the async/non-pinned pitfall.

With both fixes, `launch_gpu_pipeline` returns in ~1.6 s (down from 3.8 s), and the pipelined path shaves ~10 % off 241K end-to-end.

### Native CSD torsion extraction with SMARTS predicates

The CSD torsion SMARTS library (470 patterns) is matched via a stack-only native matcher.
46% of patterns (216 of 470) parse into a compact constraint struct and run with zero heap allocations — no VF2, no `std::vector` per match, no RDKit functor objects.
The remaining 54% fall back to RDKit's `SubstructMatch`.

9 hand-written predicate functions (`csd_torsion_predicates.hpp`) cover the most common recursive sub-queries: `$(C=O)`, `$(S(=O))`, `$([CX3]=O)`, `$([C](=O))`, `$([NX3H1])`, `$([NX3H2])`, `$(C([NX3]))`, `$([CX3]([C]))`, `$([cH0](Cl))`.
Each is 5-10 stack-only lines reading the descriptor.
All 9 cross-validated against RDKit's `SubstructMatch` on 210 molecules with zero divergences.

Without this work the torsion step called `SubstructMatch` for every pattern × every molecule, and `SubstructMatch` allocates several `std::vector` objects plus a VF2 state per call.
At 192 threads that heap churn caps preprocessing at ~13K mol/s no matter how many threads you throw at it.
With the native matcher + jemalloc, peak CPU throughput reaches ~17K mol/s.

### Flat molecule descriptor

One pass through RDKit's slow accessor API fills a 22 KB stack-allocated SoA struct (atoms, bonds, CSR adjacency, BFS distance matrix, SSSR rings, atom/bond flags).
All downstream stages (chiral centers, distance constraints, nonbonded exclusion) read this flat data with no RDKit accessors in the hot path.
Build cost: 0.026 ms/mol.

### MMFF VdW pointer caching

RDKit's `MMFFMolProperties::getMMFFVdWParams(i, j)` does two singleton-map lookups per call.
We hoist those lookups to an N-element per-atom `MMFFVdW const *` array, then call Halgren's combining rules directly.
Eliminates ~N²/2 binary searches per molecule in the O(N²) nonbonded pair construction loop.

### BFS shortest-path injection

The descriptor's BFS distance matrix is cast to `double[N²]` and injected into RDKit's molecule property cache under the key `DistanceMatrix`.
`setTopolBounds()` picks it up instead of running Floyd-Warshall O(N³).
Saves ~7 % of preprocessing time, no correctness change.

### MMFF pre-sanitization

Set the `_MMFFSanitized` property on the molecule before constructing `MMFFMolProperties` so the constructor skips its redundant Kekulize step.
1.6× on the MMFF constructor, small absolute win.

### GPU-side optimizations (pre-existing)

- `rsqrtf` replaces 17 `sqrtf` + `__frcp_rn` pairs.
- Interleaved triple warp reduction in the FIRE optimizer.
- FMA in the FIRE velocity-Verlet step.
- ETK per-atom interaction lists (was O(N·T) brute scan).
- MMFF94 per-type sub-loops (was 7-case switch with warp divergence).
- `if constexpr` template functors (compile-time gradient/energy-only branching).
- `__restrict__` on all functor parameter pointers.
- Pool allocator (was 19 `cudaMalloc` per chunk, now 1).
- `__half2` vectorized bounds loading.
- `__half` coordinate output.
- 8-GPU round-robin dispatch with atom-count-sorted waves.

## Where the time still goes

At 241K molecules × 8 GPUs, after everything above.
Numbers from the pipelined path with 5 chunks of ~48K molecules each.

| Phase per chunk                 | Wall time | Notes                                                              |
| ------------------------------- | --------- | ------------------------------------------------------------------ |
| `prepare_molecule_batch`        | 5.5-6.0 s | OpenMP parallel, 192 threads; Phase 2 pack now ~1 s                |
| `launch_gpu_pipeline` (all 8)   | 1.5-1.8 s | __dominated by metric prepass cusolver setup__, 8 GPUs in parallel |
| GPU kernel (background)         | ~5 s      | overlaps with next chunk's CPU prep; download becomes nearly free  |
| `finalize_gpu_pipeline` (all 8) | 20-30 ms  | stream has usually drained by the time we get here                 |

The remaining bottlenecks, ordered by impact:

### 1. Per-GPU fixed overhead (800 ms of cusolver CPU setup per chunk per GPU)

`metric_embedding_prepass.cuh` does group-by-atom-count setup that involves cusolver workspace sizing, per-group cudaMalloc×4, and buffer preparation.
The cusolver API calls themselves are not purely stream-async — some query the device synchronously.
This currently dominates `launch_gpu_pipeline` wall time.

Two fixes worth trying:

- Cache the cusolver workspace in `context_t` across calls instead of re-sizing per chunk.
- Consolidate the per-atom-count groups so we make fewer cusolver calls per chunk.

Expected: launch time drops from 1.6 s to a few hundred ms, end-to-end 241K drops toward ~30 s.

### 2. Bounds matrix construction on GPU

`build_bounds_matrix` is ~13 % of CPU preprocessing today.
With pipelining in place, freeing 13 % of CPU time lets the overlap reach further (or lets us shrink each chunk for better per-GPU balance).
Port `set12`/`set13`/`set14` and VdW lower bounds to four embarrassingly parallel GPU kernels (one thread per bond, angle, 1-4 path, atom pair).

### 3. Avoid materializing N²/2 nonbonded pairs on CPU

Per molecule we currently build `std::vector<mmff94_nonbonded_pair_t>` with all non-excluded pairs (up to 128 KB per molecule), then upload it.
At 50K mols/chunk that's ~6 GB of host-side construction per chunk.
Alternative: upload only atom types + charges (≈ 2 KB/mol) + the bond_distance matrix (16 KB/mol), and enumerate pairs on the fly in the MMFF nonbonded force pass from `bond_distance[i,j] > 3`.
Saves ~50× on upload bandwidth and removes the O(N²) CPU pair construction.
Requires changes to `mmff94_functor.cuh`.

## NUMA notes

Machine topology: 2 NUMA nodes × 48 physical cores × 2 HT = 192 logical CPUs, 20× remote-vs-local memory distance penalty.

Pure CPU preprocessing (8,948 mols, `bench_threaded`):

| Threads | Default affinity | `OMP_PROC_BIND=close OMP_PLACES=cores` |
| ------- | ---------------- | -------------------------------------- |
| 64      | 17,640 mol/s     | 17,126 mol/s                           |
| 96      | 17,014 mol/s     | 17,055 mol/s                           |
| 192     | 15,263 mol/s     | __17,042 mol/s__                       |

Binding OpenMP threads to physical cores stops the 192-thread collapse.
For single-socket isolation, `numactl --cpunodebind=0 --membind=0` hits the highest peak — 18,760 mol/s at 96 threads — but wastes the other socket.

End-to-end (CPU + 8-GPU pipeline) disagrees:

| Configuration                          | End-to-end time | Throughput |
| -------------------------------------- | --------------- | ---------- |
| default                                | 16,204 ms       | 552 mol/s  |
| `OMP_PROC_BIND=close OMP_PLACES=cores` | 20,488 ms       | 437 mol/s  |
| `numactl --interleave=all`             | 16,842 ms       | 531 mol/s  |

Pinning OpenMP threads to cores starves the CUDA worker and upload threads, which run outside the OpenMP pool.
Do not bake NUMA defaults into the library.
Users running CPU-only preprocessing can opt in with `OMP_PROC_BIND=close OMP_PLACES=cores`; everyone else should leave the default.

## Quality

Cross-validation: random conformer sample evaluated by RDKit's `MMFF::calcEnergy()`.

| Metric                   | Value                                   |
| ------------------------ | --------------------------------------- |
| Kernel success rate      | 95.5% (230,735 / 241,596 on 241K batch) |
| Median RDKit energy      | 59 kcal/mol                             |
| Median SmallField energy | 57 kcal/mol                             |

RDKit and SmallField energies agree within ~2 kcal/mol at median.
Half-precision coordinate output introduces ~0.1% quantization error.

### Per-atom-count breakdown (8,948-molecule subset)

| Atoms   | Count | RDKit P50 | SmallField P50 |
| ------- | ----- | --------- | -------------- |
| 0-20    | 189   | 1.2       | 1.2            |
| 20-40   | 3,391 | 36.1      | 36.1           |
| 40-60   | 3,751 | 67.3      | 67.2           |
| 60-80   | 1,530 | 82.6      | 82.4           |
| 80-100  | 619   | 122.9     | 121.9          |
| 100-128 | 260   | 136.5     | 136.4          |
