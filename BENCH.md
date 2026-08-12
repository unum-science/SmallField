# SmallField Benchmarks

How to measure this project, and what the current numbers are.

For where the time goes and what to do about it see [`ROADMAP.md`](ROADMAP.md); for the canonical vocabulary — `parquet-shard`, `coprocessed-batch`, `embedding-wave`, `atom-limited-bin`, `pipeline-role`, `wave-step` — see [`GLOSSARY.md`](GLOSSARY.md).

## Harnesses

Four binaries measure four different things, and quoting one as another is the most common mistake made with this document.

| Binary                     | Measures                                            | Includes the Pipeline Roles                |
| :------------------------- | :-------------------------------------------------- | :----------------------------------------- |
| `smallfield_batch`         | End-to-end corpus throughput, Parquet in and out    | Yes — reader, gpu-runner, fallback, writer |
| `bench_profile`            | One `generate_conformers` pass, GPU only            | No                                         |
| `bench_conformer_backends` | SmallField against RDKit CPU, plus the quality gate | No                                         |
| `bench_mmff_kernels`       | The MMFF typing, extraction and CSR kernel trio     | No                                         |

__Only `smallfield_batch` measures the shipping system.__
The library harnesses skip the Parquet reader, the writer, and the RDKit fallback that re-embeds what the GPU refused.
That fallback is 42 % of a shard's serial wall clock on Enamine REAL and roughly 85 % on PubChem, so a library number quoted as pipeline throughput overstates the pipeline by up to 7×.

## Running a Benchmark

```bash
# Build. `dev` is the inner loop; `release` is the shipped fatbin.
cmake --preset dev -DSMALLFIELD_BUILD_CLI=ON
cmake --build build --parallel

# End-to-end pipeline throughput — the number that describes the shipping system.
# Stage at least six shards into a folder, or the depth-5 ring never fills.
LD_PRELOAD=$PWD/.pixi/envs/default/lib/libjemalloc.so.2 \
  ./build/cli/smallfield_batch \
  --input <folder of parquet-shards> --output <folder on local disk> \
  --dataset real --num-conformers 3 --verbose

# GPU stage alone, no reader, writer or fallback
./build/bench/bench_profile --smi <corpus>.smi --conformers 3

# RDKit comparison and the paired quality gate
./build/bench/bench_conformer_backends --smi <corpus>.smi --conformers 3 --quality-samples 512
```

Three conformers per molecule is the production mode; K changes which resource binds, so a number at another K answers a different question.

`--verbose` prints the per-`embedding-wave` table, which is where host preprocess, launch, and per-stage GPU time become visible.
`--verbose=2` adds the per-device breakdown.

Write output to local NVMe.
The shared filesystem matches local sequential throughput but is 20 to 200× slower on small-file metadata operations, which lands entirely on the writer-role.

## Corpora

Datasets are reached through `data/` symlinks, which each machine makes for itself; never copy a dataset into the repository.

`bench/sample_corpora.py` draws a stratified SMILES sample when a harness wants a flat file rather than shards.
It walks every shard and takes a random row group from each, because rows within a shard share a deposition batch or a reagent series — the head of one shard is not a sample of the corpus.

```bash
python bench/sample_corpora.py --corpus real --count 100000 --out real-100k.smi
```

Corpus shape decides which `atom-limited-bins` exist at all, so a result on one does not transfer to another.

| Corpus  | Heavy Atoms p50 / max | With Hydrogens p50 / p99 / max | Over the 192-Atom Cap |
| :------ | :-------------------- | :----------------------------- | --------------------: |
| PubChem | 24 / 202              | 46 / 168 / 412                 |                0.68 % |
| GDB-13  | 13 / 13               | 27 / 34 / 39                   |                0.00 % |
| REAL    | 27 / 34               | 52 / 64 / 70                   |                0.00 % |

The atom cap applies __after__ explicit hydrogens, which is where the two counts diverge most: PubChem's p99 is 79 heavy atoms and 168 total.
That 0.68 % predicts the measured `too_many_atoms_k` rate of 0.69 % from an independent path.
REAL never reaches the 96, 112, 128 or 192 bins.

Shards are not ordered by molecule size — across REAL's 8 shards the mean with-hydrogen count spans 1.6 %, non-monotone, and PubChem gives a Spearman correlation of −0.01 against shard index.
Consuming shards in filename order does not bias a run.

## Reading the Output

__The progress line.__
`fallback 25,999→6,823 rec` names molecules attempted and molecules __recovered__, not molecules remaining, and both are molecules rather than conformer slots.

__Per-shard against cumulative.__
The per-shard `mol/s` charges one shard for its own serial GPU-then-fallback path.
The cumulative rate is what the pipeline delivers once shard N's fallback overlaps shard N+1's GPU work, and it climbs for the first several shards as the ring fills.
__Quote the cumulative figure__ from the closing `complete` line.

__The quality gate__ compares RDKit's MMFF94 evaluated on SmallField's own output geometry, so both columns describe identical coordinates and the paired per-conformer difference is the quantity that measures agreement.
Percentiles of the two distributions taken separately do not — a uniform offset, a heavy one-sided tail and partial cancellation all look the same that way.
Read the tail, not the median.

__Exit codes__ are `0` success, `1` a check failed, `2` the command line was refused.

## Reference Numbers

One H100 80 GB HBM3, Intel Xeon Platinum 8468 with 16 threads, driver 570.195.03 and CUDA 12.8, at `60feb403` on 2026-08-12.
Sixteen host threads is this box's defining constraint: `compute_thread_budget_` sizes the gpu-runner-role and fallback-role at `num_procs - 4` each, so they are 12 wide and contend, where a many-threaded server would not.

__Pipeline, `smallfield_batch`, K=3, one GPU.__

| Corpus               |   End-to-End |     GPU Stage | Molecule Success |
| :------------------- | -----------: | ------------: | ---------------: |
| REAL, 8M molecules   | 9,291 conf/s | 10,070 conf/s |          99.99 % |
| PubChem, first shard | 1,326 conf/s |  9,090 conf/s |                — |

The GPU stage differs by 10 % between the two corpora and the end-to-end figure by 7×.
The difference is the fallback-role: PubChem sends 2.60 % of molecules to it and recovers 26 % of them, REAL sends 0.68 % and recovers 98.5 %.

__Library, `bench_conformer_backends`, PubChem 20,000 molecules, K=1.__

| System                | conf/s |
| :-------------------- | -----: |
| RDKit CPU, 16 threads |   77.5 |
| SmallField, 1 GPU     |  4,027 |

That multiplier is against a 16-thread baseline and says as much about this host's CPU as about either system.

__Kernel resources__, from `cuobjdump --dump-resource-usage` on the `sm_90` build.

| Kernel                       | Registers | Stack | Note       |
| :--------------------------- | --------: | ----: | :--------- |
| `etkdg_pipeline_warp_kernel` |        69 |     0 | no spills  |
| `mmff_polish_warp_kernel`    |        72 |     0 | no spills  |
| `csd_torsion_match_kernel`   |       255 |   192 | __spills__ |

A 32-thread block occupies at most 32 of a multiprocessor's 64 warp slots, so 50 % is the ceiling this shape reaches at any register count.
Registers bind first, at 28 blocks per multiprocessor, which is 43.75 % theoretical.

__Quality__, PubChem 20,000 molecules with 10,000 conformers sampled.

| Metric                          |                   Value |
| :------------------------------ | ----------------------: |
| Finite RDKit MMFF94 evaluations | 9,364 / 10,000 = 93.6 % |
| Paired P50                      |         −0.013 kcal/mol |
| Paired P5 / P95                 |  −62.5 / +0.67 kcal/mol |
| Paired minimum                  |      −5.14e+10 kcal/mol |
| Heavy-bin gate                  |                    FAIL |

The median conformer agrees with RDKit to within rounding.
The paired minimum is an open regression rather than a better conformer: the MMFF94 cubic bend factor turns negative past a 143° deviation and the FIRE minimiser has no energy-decrease check to reject the step.

## Profiling

`ncu` and `nsys` crash any binary linked against jemalloc, with SIGSEGV inside `cudaGetDeviceCount` from unbounded recursion between jemalloc's malloc interposition and the profiler's `dlopen` hooks.
Preloading or not preloading it proves nothing, because `SMALLFIELD_USE_JEMALLOC=ON` puts it in `DT_NEEDED` regardless.

```bash
cmake -B build-nojemalloc -DCMAKE_BUILD_TYPE=Release -DSMALLFIELD_CUDA_ARCH="90-real" \
      -DSMALLFIELD_CUDA_LINEINFO=ON -DSMALLFIELD_USE_JEMALLOC=OFF
```

`ncu` needs `sudo` because `RmProfilingAdminOnly` is 1, and needs Nsight Compute __2025.1.1__ — version 2026.1.1 reports a driver incompatibility against 570.195.03 and fails multi-pass replay.
The binary still exits 11 during teardown under the profiler, after printing its full result and after every launch is captured, so a profiler-driven check must ignore the exit code.

The `profile` preset adds CUDA line info and NVTX ranges, so an `nsys -t cuda,nvtx` capture attributes host time to the four pipeline-roles rather than showing an anonymous CUDA timeline.

Device codegen is unaffected by the allocator, which is what makes a jemalloc-free measurement valid: `cuobjdump --dump-resource-usage` is byte-identical between builds.

## What Invalidates a Number

__Too few shards.__
The reader flushes a `coprocessed-batch` as soon as it reaches the target, which a 1,000,000-row shard does alone, so one shard is one batch and the handoffs are depth-5.
A three-shard run is all ramp and drain and measures roughly half what eight shards measure.

__The wrong harness.__
See the table at the top.

__A cold first shard.__
Wave 0 preprocess costs about 20× wave 12's as caches warm.

__A corpus that does not fill the machine.__
Achieved occupancy is grid-bound below roughly 50,000 conformers per launch — 1.6 % at 200 molecules against 40.9 % at 100,000 — so small-corpus occupancy says nothing about production.

__A profiled run's wall clock.__
Replay and clock locking inflate `gpu__time_duration.sum` well past the unprofiled time.
Never read it as throughput.

__A microbenchmark of one access pattern.__
This project once measured 6.9× on the bounds read and quoted it as a headline.
It was real at 128 atoms, which is 0.1 % of PubChem, and worth under 3 % end to end.
Weight anything distribution-sensitive by the measured atom-count distribution before quoting it.
