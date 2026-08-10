# SmallField Benchmarks

Throughput, scaling, and quality measurements on the current build.
Last refreshed 2026-07-31.

Covers ETKDG + MMFF94 conformer generation.

See [`GLOSSARY.md`](GLOSSARY.md) for the canonical vocabulary used below (`parquet-shard`, `coprocessed-batch`, `embedding-wave`, `atom-limited-bin`, `pipeline-role`, `wave-step`).

## Hardware

| Component | Specification                                   |
| :-------- | :---------------------------------------------- |
| GPU       | 8× NVIDIA RTX PRO 6000 Blackwell Server Edition |
| CPU       | 2× Intel Xeon 6776P, 192 hardware threads total |
| RAM       | 1.7 TB DDR5                                     |

## Method

- Build: `cmake -S . -B build -DRDKIT_PREFIX=$CONDA_PREFIX -DSMALLFIELD_BUILD_BATCH_BINARY=ON`
- Benchmark binary: `build/bench/bench_conformers`
- Allocator: `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`
- Primary dataset: USearch Molecules — 9,998,572 SMILES at `/tmp/usearch_10M.smi`, 1 M clean subset at `/tmp/usearch_1M_clean.smi`.
- Multi-GPU runs use `--multi-gpu-only` from a fresh process.

## End-to-End Throughput

All numbers are conformers per second — succeeded conformers divided by wall time.

### Single H100, PubChem 5K, K=3

Measured on this box, three interleaved runs per arm, `--multi-gpu-only`, spread under 1 %.

| Build | conf/s | Note |
| :---- | -----: | :--- |
Both corpora, three runs per arm on an idle box, comparing `1a0bcda` against `31a1e85` — conformer parallelism, the device bond-hop BFS, and the register cut together, of which parallelism dominates.

| Corpus | before | after | |
| :----- | -----: | ----: | :-- |
| PubChem, stratified across 116 shards | 1 924 | 4 359 | __2.27×__ |
| Enamine REAL, stratified across 8 shards | 4 612 | 7 262 | __1.58×__ |

The gap between the two is the point.
The old shape capped its grid at the molecule count, so an atom-limited bin holding fewer molecules than the grid cap could not fill the device.
PubChem spreads across all eight bins and starves in four of them; REAL tops out at 72 atoms, so bins 96, 112 and 128 are empty and 98 % of its molecules land in bins 48 and 64 — which were already large enough to fill the grid.
REAL was less starved to begin with, and gains correspondingly less.

REAL is also faster in absolute terms on both builds, and succeeds at 99.2 % against PubChem's ~91 %: narrower molecules, fewer stereo retries.

| Same PubChem measurement, at 72 regs/thread instead of 80 | 4 359 | +0.2 %, inside the noise |

That second row is a negative result worth keeping.
Dropping `etkdg_pipeline_warp_kernel` from 80 to 72 registers lifts it from 25 to 28 blocks per SM, a 12 % occupancy gain, and moves throughput by nothing measurable — so the kernel is not occupancy-limited at production bin sizes, and the remaining eight registers to 64 would not pay either.

On Enamine REAL, the acceptance corpus, the same build runs at __7 262 conf/s__ (7275 / 7237 / 7275, 0.5 % spread) with a 99.2 % success rate.
REAL tops out at 72 atoms, so bins 96, 112 and 128 are empty and 98 % of molecules land in bins 48 and 64.

The old shape also swings with machine load — an interleaved set taken while other work shared the box read 1 371–1 937 against 4 097–4 270 — which is the signature of a latency-bound launch.
The new shape holds under 1 % either way.

### SmallField — 8 GPUs, USearch 10M

| System            | Input |   K |    c/s |               Succeeded | Success |    Wall |
| :---------------- | ----: | --: | -----: | ----------------------: | ------: | ------: |
| SmallField 8 GPUs |   10M |   1 | 20,372 |   9,859,890 / 9,998,572 |  98.6 % |   484 s |
| SmallField 8 GPUs |   10M |   3 | 29,302 | 29,580,111 / 29,995,716 |  98.6 % | 1,010 s |
| SmallField 8 GPUs |   10M |  10 | 27,143 | 98,601,092 / 99,985,720 |  98.6 % | 3,202 s |

Success rate is 98.6 % across all three K values.
Conformer generation quality is independent of K.

### SmallField vs RDKit — 1 GPU / CPU 192T, USearch 1M

| System             | Input |   K |   c/s |             Succeeded | Success |      Wall |
| :----------------- | ----: | --: | ----: | --------------------: | ------: | --------: |
| RDKit CPU 192T ¹   |    1M |   1 | 2,458 |           956,961 / — |  95.7 % |   389.3 s |
| RDKit CPU 192T ¹   |    1M |   3 | 2,671 |         2,870,796 / — |  95.6 % | 1,074.6 s |
|                    |       |     |       |                       |         |           |
| SmallField 1 GPU   |    1M |   1 | 5,948 |   986,091 / 1,000,000 |  98.6 % |   163.6 s |
| SmallField 1 GPU ¹ |    1M |   1 | 8,541 |   986,017 / 1,000,000 |  98.6 % |   115.4 s |
| SmallField 1 GPU ¹ |    1M |   3 | 9,066 | 2,958,411 / 3,000,000 |  98.6 % |   326.3 s |

¹ Archived measurement from an earlier build — same hardware, same dataset.
The 1-GPU numbers will be refreshed on the current build.

Speedups vs RDKit CPU 192T on USearch 1M:

- K=1 — SmallField 1 GPU is 3.5× faster.
- K=3 — SmallField 1 GPU is 3.4× faster.

### SmallField vs nvMolKit 0.4.0

[nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) is NVIDIA's open-source (Apache 2.0) GPU conformer generator.
ETKDGv3 embedding + MMFF94 optimization, CUDA/C++ with Python bindings.
Installed via `conda install -c conda-forge nvmolkit`.

nvMolKit was benchmarked in 5 000-molecule chunks because it throws a hard `RuntimeError` on any batch containing a molecule it cannot prepare.
No per-molecule graceful degradation.

| System             | Input |   K |    c/s |        Succeeded | Success |      Wall |
| :----------------- | ----: | --: | -----: | ---------------: | ------: | --------: |
| nvMolKit 1 GPU     |    1M |   3 |    510 |   2,902,811 / 3M |  96.8 % | 5,691.8 s |
| nvMolKit "8 GPUs"² |    1M |   3 |    755 |   2,903,363 / 3M |  96.8 % | 3,847.1 s |
|                    |       |     |        |                  |         |           |
| SmallField 1 GPU   |    1M |   3 |  9,226 |   2,958,359 / 3M |  98.6 % |   320.6 s |
| SmallField 8 GPUs  |    1M |   3 | 17,715 |   2,958,295 / 3M |  98.6 % |   167.0 s |
| SmallField 8 GPUs  |   10M |   3 | 29,302 | 29,580,111 / 30M |  98.6 % | 1,010.0 s |

² `nvidia-smi` shows only GPU 0 active during the embed phase (91 % of wall time).
The MMFF optimization phase (9 %) does distribute across devices.

SmallField throughput grows with batch size: 11K → 18K → 29K conf/s at 100K → 1M → 10M molecules.
The pipelined chunk scheduler overlaps CPU prep with GPU execution across devices, so larger batches amortize host overhead better.

- SmallField is __18× faster__ on 1 GPU and __39× faster__ on 8 GPUs at production scale.
- nvMolKit's embed phase does not scale across GPUs — multi-GPU speedup is only 1.48×, entirely from the MMFF phase.
  SmallField achieves 1.92× at 1M and 2.9× at 10M.
- nvMolKit crashes on 1 of 200 chunks, losing 5 000 molecules per crash.
  SmallField marks individual failures and continues (98.6 % vs 96.8 %).
- nvMolKit would need ~55 hours to process 10M at its observed 1-GPU rate.

### SmallField vs RDKit — ChEMBL, 8× RTX PRO 6000 Blackwell

Molecules per second rather than conformers per second, against RDKit on the same host's 192 CPU threads.

| Dataset        | Workload                | RDKit CPU (192 thr) | SmallField 1 GPU | SmallField 8 GPU | Notes                                                     |
| :------------- | :---------------------- | ------------------: | ---------------: | ---------------: | :-------------------------------------------------------- |
| ChEMBL 8,948   | 3 conformers / molecule |         588.3 mol/s |    1,032.9 mol/s |    1,342.6 mol/s | 2.28x faster than RDKit at 8 GPU; 96.7% conformer success |
| ChEMBL 241,596 | 1 conformer / molecule  |       1,734.8 mol/s |    4,900.7 mol/s |   16,826.6 mol/s | 9.70x faster than RDKit at 8 GPU; 96.6% conformer success |

ChEMBL 8,948 is small enough that the pipeline never reaches steady state, which is why its 8-GPU multiplier is the weaker of the two.

## Bottleneck Summary

At 8 GPUs × K=1 on USearch 10M the pipeline is __host-bound__.
The per-chunk GPU slot — ~1 220 ms across 8 devices — sits well below the per-chunk host prep — ~2 360 ms for RDKit SMILES parse, MMFF parameter extraction, and distance-constraint build on 50 K-molecule chunks.
GPU kernel improvements translate to wall time only when the GPU is the bottleneck: on a single GPU, or at K≥3 where each chunk's GPU work grows proportionally.

Per-chunk breakdown — 10M × K=1, 8 GPUs, 1 600 chunk-device samples:

| Metric, per chunk per device |      Value |
| :--------------------------- | ---------: |
| Upload                       |     9.4 ms |
| Kernel                       | 1 134.4 ms |
| Total GPU                    | 1 220.1 ms |
| End-to-end wall, 200 chunks  |      484 s |

## Kernel Profile

Nsight Compute metrics for the two pipeline kernels at wave bins 64 and 128.
Captures under `bench/profiling/ncu/2026-04-16-phase3/`.

| Kernel                       | Bin | Regs | Dyn Shmem | Block Limit Shmem / Regs | Theoretical Occ |
| :--------------------------- | --: | ---: | --------: | -----------------------: | --------------: |
| `etkdg_pipeline_warp_kernel` |  64 |   95 |   7.17 KB |                  12 / 20 |         25.00 % |
| `etkdg_pipeline_warp_kernel` | 128 |   95 |  22.53 KB |                   4 / 20 |          8.33 % |
| `mmff_polish_warp_kernel`    |  64 |   96 |   2.30 KB |                  30 / 20 |         41.67 % |
| `mmff_polish_warp_kernel`    | 128 |   96 |   4.61 KB |                  18 / 20 |         37.50 % |

Those figures are sm_120.
On this H100, `cuobjdump --dump-resource-usage` reports `etkdg_pipeline_warp_kernel` at __80 regs__ and `mmff_polish_warp_kernel` at 72, no spills.

__Registers bind, not shared memory.__
The shared-memory reading above holds only at N=128, which is 0.1 % of PubChem.
At the median molecule of 45 atoms the DG kernel uses 4.5 KB of the SM's 228 KB, while 80 regs/thread allows 25 blocks/SM.

There is also a ceiling independent of both: a 32-thread block can occupy at most 32 of the SM's 64 warp slots, so __50 % is the most this kernel shape can reach at any register count__.
`maxBlocksPerMultiProcessor` is 32, measured.
Cutting to 64 regs/thread reaches that ceiling and is worth about 1.28×; nothing below 32 regs/thread buys more.

## Quality Snapshot

RDKit's MMFF94 is evaluated on SmallField's __own output geometry__, so the two columns describe identical coordinates and the per-conformer difference is the quantity that measures agreement.
Percentiles of the two energy distributions taken separately do not: a uniform offset, a heavy one-sided tail, and partial cancellation all produce the same apparent shift.

Quality check from PubChem 5,000 molecules / 3 conformers each, sampled on 512 successful conformers:

| Metric                             |                Value |
| ---------------------------------- | -------------------: |
| Finite RDKit MMFF94 evaluations    |    509 / 512 = 99.4% |
| SmallField − RDKit, paired P50     |      −0.0 kcal/mol   |
| SmallField − RDKit, paired P5 / P95 | −55.8 / +0.8 kcal/mol |
| SmallField − RDKit, paired minimum  |    −137.2 kcal/mol   |
| Heavy-bin gate, 100–128 atoms      |                 FAIL |

Paired per-bin deltas from the same quality sample:

| Atom bin | Count | delta P50 | delta P95 | delta min |
| -------: | ----: | --------: | --------: | --------: |
|     0–20 |    10 |      −0.0 |      −0.0 |     −56.8 |
|    20–40 |   180 |      −0.0 |      +1.5 |    −108.9 |
|    40–60 |   227 |      −0.0 |      +0.4 |    −129.6 |
|    60–80 |    67 |      −0.0 |      +0.2 |     −69.0 |
|   80–100 |    16 |      −1.3 |      +0.1 |    −137.2 |
|  100–128 |     9 |      −1.7 |     +46.2 |     −75.3 |

Two things the paired view shows that separately-sorted percentiles hid:

- __The median conformer agrees with RDKit to within rounding.__
  Every bin below 80 atoms has a paired median of −0.0 kcal/mol.
  Earlier revisions of this table reported a several-kcal/mol gap in SmallField's favour across those same bins; that gap was an artifact of sorting the two arrays independently, not a real energy difference.
- __A minority of conformers sit far below RDKit.__
  The paired P5 is −55.8 and the minimum −137.2 kcal/mol, so roughly 5 % of conformers land in a region the MMFF94 functional form does not bound from below — the cubic bend factor `(1 + cb · Δθ)` turns negative past Δθ > 143°, and the FIRE minimizer has no energy-decrease check to reject the step.
  Tracking this tail is the purpose of the `delta min` column.

The 100–128 bin carries only 9 samples here, so its `FAIL` reflects one outlier rather than a bin-wide regression; re-run at a larger `--quality-samples` before reading anything into it.

## Reproduction Commands

```bash
# USearch 10M production-scale sweep — 8 GPUs
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
  ./build/bench/bench_conformers \
  --smi /tmp/usearch_10M.smi \
  --conformers 3 \
  --multi-gpu-only

# USearch 1M single-GPU
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
  ./build/bench/bench_conformers \
  --smi /tmp/usearch_1M_clean.smi \
  --conformers 1 \
  --multi-gpu-only

# Quality check — ChEMBL 9K, 3 conformers, 256 samples
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
  ./build/bench/bench_conformers \
  --smi /tmp/chembl_valid.smi \
  --conformers 3 \
  --quality-samples 256
```
