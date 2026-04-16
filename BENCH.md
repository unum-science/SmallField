# SmallField Benchmarks

Fresh measurements collected on 2026-04-15 from the current tree after the namespace/path refactor, Python packaging cleanup, and repo-wide formatting pass.
Phase 2 (int8 bounds quantization, 2026-04-15) and Phase 3a (2-way DG+ETK / MMFF kernel split, 2026-04-16) each contribute to the single-GPU wins documented in the dedicated sections at the bottom.
The ChEMBL 8,948 and ChEMBL 241,596 tables immediately below predate Phase 2; they are preserved as a historical baseline so the phase-by-phase progression remains auditable.

## Hardware

| Component | Specification                                    |
| :-------- | :----------------------------------------------- |
| GPU       | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition  |
| CPU       | 2x Intel Xeon 6776P (192 hardware threads total) |
| RAM       | 1.7 TB DDR5                                      |

## Datasets

| File                    | Molecules | Workload                                        |
| ----------------------- | --------: | ----------------------------------------------- |
| `/tmp/chembl_valid.smi` |     8,948 | README-scale run, 3 conformers / molecule       |
| `/tmp/chembl_240k.smi`  |   241,596 | Large-batch scaling run, 1 conformer / molecule |

## Method

- Build: `cmake -S . -B build -DRDKIT_PREFIX=$CONDA_PREFIX -DSMALLFIELD_BUILD_BATCH_BINARY=ON`
- Benchmark binary: `build/benchmark/bench_pipeline_throughput`
- Allocator: `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`
- Default mode reports RDKit CPU vs SmallField 1-GPU plus an RDKit MMFF94 quality check.
- Multi-GPU scaling uses dedicated `--multi-gpu-only` runs so each GPU-count measurement comes from a fresh process.

## End-to-End Throughput

| Dataset        | Workload                |     RDKit CPU | SmallField 1 GPU | SmallField 8 GPU |                   Success |
| -------------- | ----------------------- | ------------: | ---------------: | ---------------: | ------------------------: |
| ChEMBL 8,948   | 3 conformers / molecule |   588.3 mol/s |    1,032.9 mol/s |    1,342.6 mol/s |   25,955 / 26,844 = 96.7% |
| ChEMBL 241,596 | 1 conformer / molecule  | 1,734.8 mol/s |    4,900.7 mol/s |   16,826.6 mol/s | 233,436 / 241,596 = 96.6% |

Speedups on the current hardware:

- 8,948-molecule run: 1 GPU is 1.76x faster than RDKit; 8 GPUs are 2.28x faster than RDKit.
- 241,596-molecule run: 1 GPU is 2.82x faster than RDKit; 8 GPUs are 9.70x faster than RDKit.

## Batch-Size Scaling

Single-process `bench_pipeline_throughput --smi /tmp/chembl_240k.smi --conformers 1`:

| Batch size |     RDKit CPU | SmallField 1 GPU |
| ---------: | ------------: | ---------------: |
|        100 |    74.3 mol/s |       56.7 mol/s |
|        500 |   304.6 mol/s |      241.6 mol/s |
|      1,000 |   594.5 mol/s |      445.9 mol/s |
|      2,000 |   681.4 mol/s |      863.9 mol/s |
|      5,000 | 1,443.0 mol/s |    1,807.0 mol/s |
|    241,596 | 1,734.8 mol/s |    4,900.7 mol/s |

The crossover on this host is between 1,000 and 2,000 molecules.
Once the batch is large enough to amortize preprocessing and launch overhead, the 1-GPU path pulls away quickly.

## GPU-Count Scaling

Dedicated fresh-process runs with `--multi-gpu-only` on `/tmp/chembl_240k.smi`, 1 conformer / molecule:

| Visible GPUs |      Time |     Throughput | Speedup vs 1 GPU |
| -----------: | --------: | -------------: | ---------------: |
|            1 | 46,902 ms |  5,151.1 mol/s |            1.00x |
|            2 | 28,197 ms |  8,568.3 mol/s |            1.66x |
|            4 | 18,275 ms | 13,220.2 mol/s |            2.57x |
|            8 | 14,358 ms | 16,826.6 mol/s |            3.27x |

1->8 scaling is materially sublinear.
The limiting factor at this point is no longer the warp kernel alone; host-side preparation and launch orchestration become visible once the per-GPU slice gets small enough.

## CPU-Thread Scaling

Dedicated fresh-process 8-GPU runs with `OMP_NUM_THREADS` varied on `/tmp/chembl_240k.smi`, 1 conformer / molecule:

| OMP threads |       Time |     Throughput | Speedup vs 1 thread |
| ----------: | ---------: | -------------: | ------------------: |
|           1 | 101,846 ms |  2,372.2 mol/s |               1.00x |
|           2 |  54,041 ms |  4,470.6 mol/s |               1.88x |
|           4 |  31,252 ms |  7,730.5 mol/s |               3.26x |
|           8 |  19,224 ms | 12,567.1 mol/s |               5.30x |
|          16 |  16,085 ms | 15,020.2 mol/s |               6.33x |
|          32 |  14,691 ms | 16,445.4 mol/s |               6.93x |
|          64 |  14,076 ms | 17,163.1 mol/s |               7.24x |
|          96 |  14,030 ms | 17,220.1 mol/s |               7.26x |
|         192 |  14,255 ms | 16,948.4 mol/s |               7.14x |

The scaling plateau is effectively reached by 64-96 threads.
Pushing from 96 to 192 threads adds scheduling noise but no meaningful throughput.

## Quality Snapshot

Quality check from the 8,948-molecule / 3-conformer run, sampled on 256 successful conformers:

| Metric                          |             Value |
| ------------------------------- | ----------------: |
| Finite RDKit MMFF94 evaluations | 250 / 256 = 97.7% |
| RDKit MMFF94 P50                |     68.8 kcal/mol |
| SmallField P50                  |     62.7 kcal/mol |
| Heavy-bin gate (100-128 atoms)  |              PASS |

Per-bin medians from the same quality sample:

| Atom bin | Count | RDKit P50 | SmallField P50 |
| -------: | ----: | --------: | -------------: |
|     0-20 |     4 |      13.8 |           13.8 |
|    20-40 |    76 |      39.3 |           29.4 |
|    40-60 |   104 |      68.0 |           62.8 |
|    60-80 |    34 |     103.1 |           94.0 |
|   80-100 |    19 |     106.2 |          103.1 |
|  100-128 |    13 |     157.9 |          158.0 |

## Phase 2 A/B — int8 bounds quantization (2026-04-15)

Change: the N×N distance-geometry bounds matrix is now int8-quantized with a per-molecule `(mol_min, inv_scale)` pair instead of stored as `__half`.
The min/max scan runs once at kernel stage 2 via warp-reduce; the DG evaluator dequantizes at read with a single FMA.
Halves the bounds footprint in shared memory (32 KB → 16 KB at N=128) and directly attacks the Block-Limit-Shared-Mem cap that previously pinned bin 128 to two blocks per SM.

Single-GPU production A/B on USearch 1M at 1 conformer / molecule, same binary, same dataset, same SMILES order, same seed:

| Build                    |    Time |     Throughput | Success             | vs baseline |
| :----------------------- | ------: | -------------: | :------------------ | ----------: |
| Pre-Phase-2 (`__half`)   | 253.3 s | 3,947.9 mol/s  | 986,103 / 1,000,000 |       1.00× |
| __Phase 2 (int8 packed)__ | 209.4 s | __4,774.7 mol/s__ | 986,046 / 1,000,000 |   __1.209×__ |

Success rate moved 98.61 % → 98.60 %, a 57-molecule delta out of 1 M — within run-to-run noise, so int8 quantization does not cost finite conformer success.
`test_pipeline` parity held: max |ΔE| ≤ 0.15 kcal/mol against the half-precision baseline on its 10-molecule reference set.

ncu per-kernel confirmations (captures under `benchmark/profiling/ncu/2026-04-16-phase2/`):

| Signal                        | Pre-Phase-2 bin 128 | Phase 2 bin 128 | Pre-Phase-2 bin 64 | Phase 2 bin 64 |
| :---------------------------- | ------------------: | --------------: | -----------------: | -------------: |
| Dynamic Shared Memory / block |             38.9 KB |     __22.5 KB__ |            11.3 KB |     __7.2 KB__ |
| Block Limit Shared Mem        |                   2 |           __4__ |                  9 |             12 |
| Theoretical Occupancy         |              4.17 % |      __8.33 %__ |             16.7 % |       __25 %__ |
| Registers Per Thread          |                 116 |             117 |                116 |            117 |
| Compute (SM) Throughput       |              0.95 % |          0.98 % |             1.07 % |         1.17 % |

The 2× jump in theoretical occupancy at bin 128 is the mechanical source of the 1.209× end-to-end win: shared-memory per block was the occupancy-binding cap at long molecules, and int8 packing cuts it in half.

Reproduction (single GPU, 1M sample):

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
  ./build/benchmark/bench_pipeline_throughput \
  --smi /tmp/usearch_1M_clean.smi \
  --conformers 1 \
  --multi-gpu-only
```

## Phase 3a A/B — 2-way kernel split (2026-04-16)

Change: the monolithic `etkdg_pipeline_warp_kernel` held all DG / ETK / MMFF state live across the whole iteration budget, which pinned register pressure at 145 regs / thread.
Phase 3a splits the pipeline into two sibling kernels on the same CUDA stream:

1. `etkdg_pipeline_warp_kernel` — stages 2, 4-8 (bounds int8 quantize, DG FIRE, stereo retry, fourth-dim crush, ETK torsion refinement, final stereo validation).
   Writes 3D ETK-refined coordinates into `conformer_coordinates_out`.
2. `mmff_polish_warp_kernel` — stage 9 (MMFF94 FIRE with optional heavy rescue).
   Reads positions from the first kernel's output, overwrites them with the MMFF-polished coordinates, replaces the ETK energy with the MMFF energy.
   Skips slots with non-zero status (failed stereo) or zero MMFF terms.

Register-attribution probe on the pre-split code showed the 145-reg monolithic kernel spent ~53 regs on stage-overlap state (all three functors' pointer fields + RNG + stereo retry loop alive across branch boundaries), vs ~72-92 regs for any single stage measured in isolation.
After the split nvcc resolves the pipeline kernel to 95 regs / thread and the MMFF polish kernel to 96 regs — a ~1.5× headroom for reg-cap-bound occupancy.

### Single-GPU production A/B on USearch 1M × 1 conformer

| Build                        |    Time |     Throughput |             Success | vs baseline |
| :--------------------------- | ------: | -------------: | ------------------: | ----------: |
| Pre-Phase-2 (`__half` bounds) | 253.3 s | 3,947.9 mol/s  | 986,103 / 1,000,000 |       1.00× |
| Phase 2 (int8 packed bounds) | 209.4 s | 4,774.7 mol/s  | 986,046 / 1,000,000 |      1.209× |
| __Phase 3a (2-way split)__    | __163.6 s__ | __6,031.6 mol/s__ | __986,091 / 1,000,000__ |  __1.528×__ |

Success rate runs within ±0.01 pp across all three builds — precision of both structural changes is invisible at the mol/s-scale accuracy gate.

### 8-GPU production A/B on USearch 10M, 1 and 3 conformer / molecule

| Workload                                       | Build       |     Wall | Throughput (partial) | Per-chunk-per-device K | vs baseline |
| :--------------------------------------------- | :---------- | -------: | -------------------: | ---------------------: | ----------: |
| 10M × 1 conformer                              | Pre-Phase-2 |  478.9 s |       20,880.0 mol/s |             1 291.4 ms |       1.00× |
| 10M × 1 conformer                              | __Phase 3a__ | __472.4 s__ |  __21,166.8 mol/s__ |         __1 134.4 ms__ |  __1.014×__ |
| 10M × 3 conformers (partial, chunk 0 – 191) ¹ | Pre-Phase-2 |  1 030 s |        9,320.4 mol/s |             4 588.4 ms |       1.00× |
| 10M × 3 conformers (partial, chunk 0 – 191) ¹ | __Phase 3a__ | __934 s__ |   __10,278.5 mol/s__ |         __4 120.6 ms__ |  __1.103×__ |

¹ Both K=3 runs hit the same tail-of-run segfault at chunk 192 (known bug on the final-chunk cleanup path; unrelated to the structural work in Phases 2/3a and pre-dates this session).
Wall times are log-file birth → modify on the identical truncation point, so the partial-throughput comparison is apples-to-apples: 192 × ~50 K-molecule chunks, 9.6 M molecules, 28.8 M conformers.
The `--multi-gpu-only` driver does not emit a final Mol/s line when the process crashes — the numbers are derived from chunks-completed / wall-to-crash.

K=1 moves only 1.4 % at 8-GPU scale despite the 52.8 % single-GPU gain.
K=3 picks up 10.3 % because the GPU work per chunk is now ~3.6× bigger, which flips the critical path back to GPU-bound:

- __K=1 per chunk:__ GPU parallel slot ~1.22 s (across 8 devices), host prep ~2.36 s / chunk.
  Host is the critical path, so kernel speedups shrink GPU idle gaps but not wall.
- __K=3 per chunk:__ GPU parallel slot ~4.12 s, host prep still ~2.3 s / chunk.
  GPU is the critical path, so Phase 3a's 10 % kernel contraction bubbles straight up into wall.

The takeaway: Phase 3a's kernel-path win translates to throughput wherever GPU is the bottleneck.
For 1-conformer bulk scans at 8-GPU scale, host prep is now the ceiling — see `IMPROVE.md` for the post-session lever ranking (MMFF parameter LUT caching, SMILES parse overlap, scaffold-cache extension).

### Per-chunk kernel-time breakdown (10M × 1 conformer, 8 GPU, 1 600 chunk-device samples)

| Metric (per chunk per device) | Pre-Phase-2 | Phase 3a |       Δ |
| :---------------------------- | ----------: | -------: | ------: |
| Upload (U)                    |    10.1 ms  |   9.4 ms |  −6.9 % |
| Kernel (K)                    |  1 291.4 ms | 1 134.4 ms | __−12.2 %__ |
| Total per chunk (T)           |  1 381.5 ms | 1 220.1 ms |  −11.7 % |
| End-to-end wall (200 chunks)  |    478.9 s  |   472.4 s |  __−1.4 %__ |

The kernel phase contracted 12 % — consistent with the Phase 3a reg-cap lift on `mmff_polish_warp_kernel` — but barely any of that contraction bubbles up into end-to-end wall.
Per-chunk GPU parallel slot (~1 220 ms across 8 devices) is already smaller than the per-chunk host prep (~2 360 ms for RDKit parse + MMFF parameter extraction + distance-constraint build on a 50 K-molecule chunk), so the 8-GPU pipeline is __host-bound__.
Cutting GPU kernel time only shrinks the GPU's idle gap between N / N+1 uploads; it does not shrink the host prep critical path.
On single-GPU runs the same 50 K-molecule chunk has 8× the GPU wall time per device but the same host prep, flipping the bottleneck back to GPU — which is why the 1 M × 1-GPU A/B sees the full kernel-path 52.8 % gain while the 8-GPU × 10 M number stays near baseline.

### ncu per-kernel confirmations at bin 64 / bin 128

Captures under `benchmark/profiling/ncu/2026-04-16-phase3/` (regenerate with `benchmark/profiling/run_ncu_phase3.sh`).

| Kernel                        | Bin |   Regs | Dyn Shmem | BlockLim(Shmem / Regs) |  Theoretical Occ |
| :---------------------------- | --: | -----: | --------: | :--------------------: | ---------------: |
| Monolithic (Phase 2)          |  64 |    117 |   7.17 KB |           12 / 16      |         25.00 %  |
| Monolithic (Phase 2)          | 128 |    117 |  22.53 KB |            4 / 16      |          8.33 %  |
| `etkdg_pipeline_warp_kernel`  |  64 | __95__ |   7.17 KB |           12 / 20      |         25.00 %  |
| `etkdg_pipeline_warp_kernel`  | 128 | __95__ |  22.53 KB |            4 / 20      |          8.33 %  |
| `mmff_polish_warp_kernel`     |  64 |     96 | __2.30 KB__ |         30 / 20      |   __41.67 %__   |
| `mmff_polish_warp_kernel`     | 128 |     96 | __4.61 KB__ |         18 / 20      |   __37.50 %__   |

The MMFF polish kernel is the big win: its per-block shmem footprint drops from the monolithic kernel's 22.5 KB (bounds-dominated) to 4.6 KB (FIRE scratch only — MMFF evaluation uses explicit term lists, not a pair-distance matrix), unlocking ~4.5× theoretical occupancy vs running MMFF inside the monolithic kernel.
The DG+ETK kernel cuts registers by 22 / thread but still sits at the same shmem-cap theoretical occupancy (the bounds matrix is unchanged there).

### Reproduction

```bash
# Single GPU 1M A/B — reproduce the 6,031.6 mol/s number
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
  ./build/benchmark/bench_pipeline_throughput \
  --smi /tmp/usearch_1M_clean.smi --conformers 1 --multi-gpu-only

# 8 GPU 10M — reproduce the 21,166.8 mol/s number
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
  ./build/benchmark/bench_pipeline_throughput \
  --smi /tmp/usearch_10M.smi --conformers 1 --multi-gpu-only
```

## Reproduction Commands

```bash
# README-scale run
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
  ./build/benchmark/bench_pipeline_throughput \
  --smi /tmp/chembl_valid.smi \
  --conformers 3 \
  --quality-samples 256

# Large-batch CPU vs 1-GPU run
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
  ./build/benchmark/bench_pipeline_throughput \
  --smi /tmp/chembl_240k.smi \
  --conformers 1

# Large-batch multi-GPU scaling
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
  ./build/benchmark/bench_pipeline_throughput \
  --smi /tmp/chembl_240k.smi \
  --conformers 1 \
  --multi-gpu-only
```
