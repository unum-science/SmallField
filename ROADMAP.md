# SmallField — Roadmap

Work that is not done yet, spanning both software optimisation and dataset / output-completeness gaps.
For shipped work see [`CHANGELOG.md`](CHANGELOG.md); for measured throughput see [`BENCH.md`](BENCH.md); for the canonical vocabulary — `parquet-shard`, `coprocessed-batch`, `embedding-wave`, `atom-limited-bin`, `pipeline-role`, `wave-step` — see [`GLOSSARY.md`](GLOSSARY.md).

Nothing here is in the past tense.
Every experiment that has been tried and rejected is a dated entry in [`CHANGELOG.md`](CHANGELOG.md), including the ones that never produced a commit.
Read those before proposing work here, so a closed question is not reopened.

## Current State

The pipeline is GPU-bound on a single Hopper card as of 2026-08-03, after preprocessing moved to the device.
Sampling `utilization.gpu` at 100 ms across a 20 000-molecule stratified PubChem run at K=3 on one H100 gives 7.71 s of GPU-busy time against an 8.05 s `generate_conformers` window, about 96 %.

`BENCH.md` still reports host-bound at 10M USearch, K=1, 8× Blackwell, refreshed 2026-07-31, which predates that move.
Both readings are in this document's terms: different corpus, different conformer count, different hardware, and one of them older than the change that matters.
Neither generalises to the other, and neither is the answer for a multi-node tier nobody has measured.
Re-measure per tier before treating either as current.

## How to Read This Document

Every number below carries a scale tag naming corpus, conformer count and hardware — for example `10M USearch, K=1, 8× Blackwell` or `100K ChEMBL, K=1, 1× H100`.
This matters because:

- Hardware decides whether a number transfers at all.
  Hopper and Blackwell differ in copy engines, shared-memory budget and register file, and this project targets both, at every tier from a single GPU to a multi-node cluster.
  A figure measured on one is evidence about that one; the box a measurement happened to run on is never the design target.

- Corpus shape decides which bins exist at all.
  PubChem spreads across all nine atom-limited bins with 1 % above 112 atoms; Enamine REAL tops out at 72 atoms, so four bins are empty and 98 % of molecules land in bins 48 and 64; GDB13 is 96 % in bin 32.
  A number measured on one of those does not transfer to another — the warp-per-pair change is 2.27 × on PubChem and 1.58 × on REAL for exactly this reason.
  Sample with `bench/sample_corpora.py`, which draws across shards; the head of one shard is not a sample of the corpus.
- Kernel occupancy is dominated by launch parallelism, not per-kernel resource pressure.
  At bins with ≤ 100 molecules, achieved occupancy is 2 %; at bins with 2 000+ molecules it rises to 23–32 %.
  Extrapolating "the kernel is idle" from small-dataset measurements is misleading.
- Pipeline steady state takes several waves to reach.
  Runs below ~300 K molecules spend a meaningful fraction of wall time in bootstrap / drain, biasing the perceived CPU / GPU share.

Rule of thumb: do not trust any number in this document that is not paired with a scale tag.
When re-profiling to drive a decision, aim for ≥ 1 M molecules and at least 10 `embedding-waves` of steady state — and separately note whether the workload is drug-like, meaning ChEMBL or PubChem, or diverse combinatorial, meaning USearch, Enamine or ZINC.

Every item below carries the same six fields, so two items can be compared without reading both in full.
__Evidence__ names its scale tag and its date, and says so outright when the measurement predates a change that invalidates it.

## Software

### Depth-3 Ring Buffer and Async Finalize

__Mechanism:__ Promote the 2-slot `batch_pool_t` ring in `generate_conformers` to 3 slots, issue wave N's D2H and reorder on a separate stream so wave N+1's `launch_wave` overlaps wave N's finalize, and pre-allocate the third slot's pinned buffer in the `context_t` constructor so no `cudaHostAlloc` fires mid-run.
__Evidence:__ 10M USearch, K=1, 8× Blackwell, 2026-04-19 — the ring idles the GPU 20–30 ms per wave transition, and `finalize_gpu_pipeline` blocks 5–10 ms in `cudaStreamSynchronize` before the next `launch_wave` can start; _stale_, predates device preprocessing.
__Gain:__ 4–6 % from the depth-3 ring plus 1–2 % from async finalize.
__Effort:__ Medium — the ring state machine needs careful invariants so slot `i % 3` is not overwritten while wave `i - 3` is still in flight.
__Risk:__ Medium — a 3-stream split landed in `abe4d03` and was reverted in `d19fdad` for an illegal-memory-access race at the `cudaEventSynchronize(finalize)` hand-off under PubChem drug-like load, together with a ~13 % throughput regression at 10M K=3.
__Gate:__ Re-validate at drug-like scale, not on USearch fragment-like, where the race window is smaller and kernel phases are long enough to flatter copy/compute overlap.

### OpenMP Load Imbalance at the Preprocess Barrier

__Mechanism:__ Switch `schedule(dynamic, 64)` in `prepare_per_thread_contribs_` to `schedule(guided)`, or pre-sort molecules by estimated atom count so the longest ones start first.
The SMILES-length sort inside `generate_conformers` already approximates this at the outer level; applying it per thread-block inside the prep loop closes the imbalance.
__Evidence:__ 10M USearch, K=1, 8× Blackwell, 2026-04-19 — 5.17 % of CPU in `gomp_barrier_wait_end` while tail threads finish the biggest molecules; _stale_, measured against a loop that still typed and extracted MMFF terms on the host, so the loop this targets is now a fraction of what was profiled.
__Gain:__ 4–6 % host prep, unverified since the device move.
__Effort:__ Small — one pragma plus an optional sort.
__Risk:__ Low.
__Gate:__ Re-measure the barrier share before sizing the fix.

### SMILES Parse Overlap Across Waves

__Mechanism:__ Parse wave N+1's SMILES on the host during wave N's `embedding-step`, which is GPU-bound and leaves cores idle.
The `reader-role` already pulls the next batch from disk in parallel; this extends the same idea to the parse.
__Evidence:__ 10M USearch, K=1, 8× Blackwell, 2026-04-19 — `libRDKitSmilesParse` at 4.87 % of CPU self-time.
Unlike the other host items this one did not shrink: the RDKit front-end is all that stays on the host now, so parse is the largest remaining host cost rather than a smaller share of a smaller loop.
__Gain:__ 4–7 % host prep, upper end if combined with the barrier fix above.
__Effort:__ Medium — requires decoupling the parse from `prepare_molecule_batch_into`.
__Risk:__ Low to medium — RDKit's SMILES parser is thread-safe for independent molecules.
__Gate:__ Confirm the embedding step is long enough to hide a parse at the wave sizes auto-sizing now picks, roughly 263 K molecules on an 80 GB card.

### Allocator Pressure Reduction

__Mechanism:__ Audit `prepare_molecule_batch_into` for temp-alloc sites, promote short-lived `std::vector` to per-thread scratch with grow-only capacity, and pre-size the known-bounded containers.
__Evidence:__ jemalloc 4.10 % plus libc malloc 2.82 % at 10M USearch, K=1, 8× Blackwell, 2026-04-19; _stale_, because `extract_mmff94_params_via_rdkit` is off the production batch path, which is where most of this allocation sat.
That function still exists and is still called by `prepare_molecule_into`, the non-pipelined path the tests use, so a profile taken through that entry point will still show it.
__Gain:__ 3–5 % host prep.
__Effort:__ Medium.
__Risk:__ Low — RAII scratch is well understood and the project already uses it for batch pools.
__Gate:__ Fresh profile through the batch path, not the single-molecule path, before the audit is worth running.

### Attribute the Fixed Context-Construction Cost

__Mechanism:__ Find where roughly 35 seconds go before a single molecule is embedded.
__Evidence:__ Measured as the gap between total process time and the timed `generate_conformers` window on a 20-molecule input; fixed, independent of corpus size and conformer count.
Excluded so far, each measured directly: MMFF94 table extraction from RDKit and the compiled CSD torsion table, since `test_mmff94_tables`, `test_mmff_gpu_typer` and `test_csd_torsion_device` each build their tables and finish in under half a second; CUDA module loading, since `CUDA_MODULE_LOADING=LAZY` changes nothing; and PTX JIT, since `cuobjdump` reports an sm_90 cubin in the binary.
Neither the per-device scratch struct nor the MMFF staging layout is large enough to explain it.
__Gain:__ None on dataset throughput — the CLI amortises this over a whole shard.
It dominates short runs and the Python binding, and it silently distorts any whole-process utilisation measurement.
__Effort:__ Unknown until attributed.
__Risk:__ Low.
__Gate:__ Attribute it before optimising anything else in the startup path.

### Widen the ETK CSR Build Past One Lane

__Mechanism:__ `etk_build_csr_kernel` in `preprocess_kernel.cuh` launches a warp per molecule and then returns every lane but the first, so the counting pass, the prefix sum over the per-atom cursors and the scatter all run serially on lane 0.
A lane-strided count with a warp scan over the cursors is the natural fix.
__Evidence:__ None measured — the work is a single pass over the molecule's torsions and constraints, small next to the ETKDG kernel, and was not worth blocking the device-preprocessing merge on.
__Gain:__ Unknown.
__Effort:__ Small.
__Risk:__ Low.
__Gate:__ Measure the kernel in isolation first; under a percent of the wave and it stays a note.

### Reduced-Precision Bounds in the Smoothing Prepass

__Mechanism:__ `embed_4d_power_iter_kernel` in `metric_embedding_prepass.cuh` holds the whole N × N bounds matrix in shared memory as `float` — `shared_gram`, which doubles as the smoothing scratch before being overwritten with the Gram matrix.
That is about 144 KB at the 192-atom bin, large enough to need the `cudaFuncAttributeMaxDynamicSharedMemorySize` opt-in to launch at all, and the largest shared-memory consumer anywhere in the pipeline.
__Evidence:__ None — `CHANGELOG.md` records the int8 quantization of the shared bounds inside the ETKDG kernel and says nothing about the prepass, so the precision floor for Floyd-Warshall itself is unmeasured.
The warp kernel downstream does demonstrate the bounds survive int8 with a per-molecule linear scale, which is what makes this worth trying rather than what makes it safe.
__Gain:__ Occupancy on bins 128 and 192 only, which is 1 % of PubChem and none of Enamine REAL.
__Effort:__ Medium.
__Risk:__ Medium — smoothing is a transitive closure over sums of bounds, so quantization error compounds across pivots, where the downstream dequantization reads each entry once.
A geometry regression surfaces as chirality failures downstream, not as a smoothing bug.
__Gate:__ Measure the bin population first and drop this if the corpus does not reach it.
Then measure `__half` before int8, and gate on `stereo_failure` rate and paired energies rather than on wall time — the size-aware distance-geometry budget rejected on 2026-08-03 looked free on wall time and cost 3 102 conformers.

## Datasets

Separate axis from raw throughput.
These close output-completeness gaps that ETKDG and MMFF94 do not address, not CPU / GPU cycles.

### Detect and Tag Axial / Helical Chirality

__Mechanism:__ Add `axial_chirality_detected` and `helical_chirality_detected` bits to `molecule_descriptor_t`, propagate them through `finalize_gpu_pipeline` into `output_buffers_t`, and emit them as two boolean columns in the Parquet writer.
The biaryl axial check is a single SMARTS-style walk over the heavy atoms; the helical check is a connected-component count over `atom_ring_bitmap`.
Both reuse the descriptor's existing aromaticity and ring-info fields.
This is explicitly not enumeration — the library does not generate a second handedness on the user's behalf, and per the README's `Project Goals & FAQ` treats configuration-space coverage as an upstream concern.
__Evidence:__ 100K samples of each of USearch and ChEMBL — axial chirality in 0.28–0.44 % of molecules, helical in 0.11–0.18 %.
Neither is encoded in a SMILES string at all, so ETKDG embeds one handedness arbitrarily from the random seed and a user counting on both enantiomers of an atropisomer gets one.
__Gain:__ Output completeness, not throughput; detection adds under 2 µs per molecule against a preprocess budget of roughly 10 ms.
__Effort:__ Small, about 50 additive lines.
__Risk:__ Low.
__Gate:__ None — the cost is bounded and the columns are ignorable by consumers who do not want them.

### Scaffold-Aware Input Sharding

__Mechanism:__ Compute a Bemis-Murcko scaffold hash in `smi_to_parquet.py` and cluster on it, so structurally similar molecules land in the same `coprocessed-batch` instead of the current offset-based shards.
__Evidence:__ __The original justification is void.__
It rested on a 33 % scaffold-cache hit rate at 10M USearch, and that cache was deleted along with the VF2 fallback, so there is no cache left to keep warm.
__Gain:__ Unknown.
Any remaining gain would have to come from warp divergence in the native torsion matcher or from `atom-limited-bin` fill, neither of which has been measured.
__Effort:__ Small.
__Risk:__ Low.
__Gate:__ Re-derive the mechanism and measure it before costing this work.

### Direct Cloud / S3 Input

__Mechanism:__ Plug Arrow's `S3FileSystem` into the `reader-role`, which reads local `parquet-shard` files today, so billion-scale workloads that do not fit on local disk skip the intermediate copy.
__Evidence:__ None needed — this is a capability gap, not an optimisation.
__Gain:__ Removes a full staging copy at billion scale.
__Effort:__ Medium.
__Risk:__ Low.
__Gate:__ None.
