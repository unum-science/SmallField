# SmallField Benchmarks

Fresh measurements collected on 2026-04-15 from the current tree after the namespace/path refactor, Python packaging cleanup, and repo-wide formatting pass.

The ChEMBL 8,948 and ChEMBL 241,596 tables below are the __pre-quantization baseline__ (half-precision bounds in shared memory).
Phase 2 (int8 bounds packing, landed 2026-04-15) adds a +20.9 % single-GPU win on the same code path — see the dedicated USearch 1M A/B at the bottom of this file.
Those ChEMBL tables will be re-measured under Phase 2 in a follow-up pass; the mechanics of the change are `src/smallfield/cuda/dg_kernel.cuh` (int8 read + single-FMA dequantize) and `src/smallfield/cuda/etkdg_pipeline.cuh` (per-molecule min/max warp reduction, quantize to shared on stage 2).

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
