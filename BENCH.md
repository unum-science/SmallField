# SmallField Benchmarks

Fresh measurements collected on 2026-04-15 from the current tree after the namespace/path refactor, Python packaging cleanup, and repo-wide formatting pass.

## Hardware

| Component | Specification |
| --------- | ------------- |
| GPU | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition |
| CPU | 2x Intel Xeon 6776P (192 hardware threads total) |
| RAM | 1.7 TB DDR5 |

## Datasets

| File | Molecules | Workload |
| ---- | --------: | -------- |
| `/tmp/chembl_valid.smi` | 8,948 | README-scale run, 3 conformers / molecule |
| `/tmp/chembl_240k.smi` | 241,596 | Large-batch scaling run, 1 conformer / molecule |

## Method

- Build: `cmake -S . -B build -DRDKIT_PREFIX=$CONDA_PREFIX -DSMALLFIELD_BUILD_BATCH_BINARY=ON`
- Benchmark binary: `build/benchmark/bench_pipeline_throughput`
- Allocator: `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`
- Default mode reports RDKit CPU vs SmallField 1-GPU plus an RDKit MMFF94 quality check.
- Multi-GPU scaling uses dedicated `--multi-gpu-only` runs so each GPU-count measurement comes from a fresh process.

## End-to-End Throughput

| Dataset | Workload | RDKit CPU | SmallField 1 GPU | SmallField 8 GPU | Success |
| ------- | -------- | --------: | ----------------: | ----------------: | ------: |
| ChEMBL 8,948 | 3 conformers / molecule | 588.3 mol/s | 1,032.9 mol/s | 1,342.6 mol/s | 25,955 / 26,844 = 96.7% |
| ChEMBL 241,596 | 1 conformer / molecule | 1,734.8 mol/s | 4,900.7 mol/s | 16,826.6 mol/s | 233,436 / 241,596 = 96.6% |

Speedups on the current hardware:

- 8,948-molecule run: 1 GPU is 1.76x faster than RDKit; 8 GPUs are 2.28x faster than RDKit.
- 241,596-molecule run: 1 GPU is 2.82x faster than RDKit; 8 GPUs are 9.70x faster than RDKit.

## Batch-Size Scaling

Single-process `bench_pipeline_throughput --smi /tmp/chembl_240k.smi --conformers 1`:

| Batch size | RDKit CPU | SmallField 1 GPU |
| ---------: | --------: | ----------------: |
| 100 | 74.3 mol/s | 56.7 mol/s |
| 500 | 304.6 mol/s | 241.6 mol/s |
| 1,000 | 594.5 mol/s | 445.9 mol/s |
| 2,000 | 681.4 mol/s | 863.9 mol/s |
| 5,000 | 1,443.0 mol/s | 1,807.0 mol/s |
| 241,596 | 1,734.8 mol/s | 4,900.7 mol/s |

The crossover on this host is between 1,000 and 2,000 molecules.
Once the batch is large enough to amortize preprocessing and launch overhead, the 1-GPU path pulls away quickly.

## GPU-Count Scaling

Dedicated fresh-process runs with `--multi-gpu-only` on `/tmp/chembl_240k.smi`, 1 conformer / molecule:

| Visible GPUs | Time | Throughput | Speedup vs 1 GPU |
| -----------: | ---: | ---------: | ---------------: |
| 1 | 46,902 ms | 5,151.1 mol/s | 1.00x |
| 2 | 28,197 ms | 8,568.3 mol/s | 1.66x |
| 4 | 18,275 ms | 13,220.2 mol/s | 2.57x |
| 8 | 14,358 ms | 16,826.6 mol/s | 3.27x |

1->8 scaling is materially sublinear.
The limiting factor at this point is no longer the warp kernel alone; host-side preparation and launch orchestration become visible once the per-GPU slice gets small enough.

## CPU-Thread Scaling

Dedicated fresh-process 8-GPU runs with `OMP_NUM_THREADS` varied on `/tmp/chembl_240k.smi`, 1 conformer / molecule:

| OMP threads | Time | Throughput | Speedup vs 1 thread |
| ----------: | ---: | ---------: | ------------------: |
| 1 | 101,846 ms | 2,372.2 mol/s | 1.00x |
| 2 | 54,041 ms | 4,470.6 mol/s | 1.88x |
| 4 | 31,252 ms | 7,730.5 mol/s | 3.26x |
| 8 | 19,224 ms | 12,567.1 mol/s | 5.30x |
| 16 | 16,085 ms | 15,020.2 mol/s | 6.33x |
| 32 | 14,691 ms | 16,445.4 mol/s | 6.93x |
| 64 | 14,076 ms | 17,163.1 mol/s | 7.24x |
| 96 | 14,030 ms | 17,220.1 mol/s | 7.26x |
| 192 | 14,255 ms | 16,948.4 mol/s | 7.14x |

The scaling plateau is effectively reached by 64-96 threads.
Pushing from 96 to 192 threads adds scheduling noise but no meaningful throughput.

## Quality Snapshot

Quality check from the 8,948-molecule / 3-conformer run, sampled on 256 successful conformers:

| Metric | Value |
| ------ | ----: |
| Finite RDKit MMFF94 evaluations | 250 / 256 = 97.7% |
| RDKit MMFF94 P50 | 68.8 kcal/mol |
| SmallField P50 | 62.7 kcal/mol |
| Heavy-bin gate (100-128 atoms) | PASS |

Per-bin medians from the same quality sample:

| Atom bin | Count | RDKit P50 | SmallField P50 |
| -------: | ----: | --------: | -------------: |
| 0-20 | 4 | 13.8 | 13.8 |
| 20-40 | 76 | 39.3 | 29.4 |
| 40-60 | 104 | 68.0 | 62.8 |
| 60-80 | 34 | 103.1 | 94.0 |
| 80-100 | 19 | 106.2 | 103.1 |
| 100-128 | 13 | 157.9 | 158.0 |

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
