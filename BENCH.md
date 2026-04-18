# SmallField Benchmarks

Throughput, scaling, and quality measurements on the current build.
Last refreshed 2026-04-16.

## Hardware

| Component | Specification                                   |
| :-------- | :---------------------------------------------- |
| GPU       | 8× NVIDIA RTX PRO 6000 Blackwell Server Edition |
| CPU       | 2× Intel Xeon 6776P, 192 hardware threads total |
| RAM       | 1.7 TB DDR5                                     |

## Method

- Build: `cmake -S . -B build -DRDKIT_PREFIX=$CONDA_PREFIX -DSMALLFIELD_BUILD_BATCH_BINARY=ON`
- Benchmark binary: `build/bench/bench_pipeline_throughput`
- Allocator: `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`
- Primary dataset: USearch Molecules — 9,998,572 SMILES at `/tmp/usearch_10M.smi`, 1 M clean subset at `/tmp/usearch_1M_clean.smi`.
- Multi-GPU runs use `--multi-gpu-only` from a fresh process.

## End-to-End Throughput

All numbers are conformers per second — succeeded conformers divided by wall time.

### SmallField — 8 GPUs, USearch 10M

| System            | Input |    K |    c/s |               Succeeded | Success |    Wall |
| :---------------- | ----: | ---: | -----: | ----------------------: | ------: | ------: |
| SmallField 8 GPUs |   10M |    1 | 20,372 |  9,859,890 /  9,998,572 |  98.6 % |   484 s |
| SmallField 8 GPUs |   10M |    3 | 29,302 | 29,580,111 / 29,995,716 |  98.6 % | 1,010 s |
| SmallField 8 GPUs |   10M |   10 | 27,143 | 98,601,092 / 99,985,720 |  98.6 % | 3,202 s |

Success rate is 98.6 % across all three K values.
Conformer generation quality is independent of K.

### SmallField vs RDKit — 1 GPU / CPU 192T, USearch 1M

| System             | Input |    K |   c/s |             Succeeded | Success |      Wall |
| :----------------- | ----: | ---: | ----: | --------------------: | ------: | --------: |
| RDKit CPU 192T ¹   |    1M |    1 | 2,458 |           956,961 / — |  95.7 % |   389.3 s |
| RDKit CPU 192T ¹   |    1M |    3 | 2,671 |         2,870,796 / — |  95.6 % | 1,074.6 s |
|                    |       |      |       |                       |         |           |
| SmallField 1 GPU   |    1M |    1 | 5,948 |   986,091 / 1,000,000 |  98.6 % |   163.6 s |
| SmallField 1 GPU ¹ |    1M |    1 | 8,541 |   986,017 / 1,000,000 |  98.6 % |   115.4 s |
| SmallField 1 GPU ¹ |    1M |    3 | 9,066 | 2,958,411 / 3,000,000 |  98.6 % |   326.3 s |

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

| System             | Input |    K |    c/s |        Succeeded | Success |      Wall |
| :----------------- | ----: | ---: | -----: | ---------------: | ------: | --------: |
| nvMolKit 1 GPU     |    1M |    3 |    510 |   2,902,811 / 3M |  96.8 % | 5,691.8 s |
| nvMolKit "8 GPUs"² |    1M |    3 |    755 |   2,903,363 / 3M |  96.8 % | 3,847.1 s |
|                    |       |      |        |                  |         |           |
| SmallField 1 GPU   |    1M |    3 |  9,226 |   2,958,359 / 3M |  98.6 % |   320.6 s |
| SmallField 8 GPUs  |    1M |    3 | 17,715 |   2,958,295 / 3M |  98.6 % |   167.0 s |
| SmallField 8 GPUs  |   10M |    3 | 29,302 | 29,580,111 / 30M |  98.6 % | 1,010.0 s |

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

| Kernel                       |  Bin | Regs | Dyn Shmem | Block Limit Shmem / Regs | Theoretical Occ |
| :--------------------------- | ---: | ---: | --------: | -----------------------: | --------------: |
| `etkdg_pipeline_warp_kernel` |   64 |   95 |   7.17 KB |                  12 / 20 |         25.00 % |
| `etkdg_pipeline_warp_kernel` |  128 |   95 |  22.53 KB |                   4 / 20 |          8.33 % |
| `mmff_polish_warp_kernel`    |   64 |   96 |   2.30 KB |                  30 / 20 |         41.67 % |
| `mmff_polish_warp_kernel`    |  128 |   96 |   4.61 KB |                  18 / 20 |         37.50 % |

The MMFF polish kernel runs with 4.5× higher theoretical occupancy than the DG+ETK kernel at bin 128 because it carries no bounds matrix in shared memory — only the 3D FIRE scratch arrays.
The DG+ETK kernel is shared-memory-bound at large molecules; its int8-quantized bounds matrix — 16 KB at N=128 — is the binding constraint.

## Quality Snapshot

Quality check from ChEMBL 8,948 molecules / 3 conformers each, sampled on 256 successful conformers:

| Metric                          |             Value |
| ------------------------------- | ----------------: |
| Finite RDKit MMFF94 evaluations | 250 / 256 = 97.7% |
| RDKit MMFF94 P50                |     68.8 kcal/mol |
| SmallField P50                  |     62.7 kcal/mol |
| Heavy-bin gate, 100–128 atoms   |              PASS |

Per-bin medians from the same quality sample:

| Atom bin | Count | RDKit P50 | SmallField P50 |
| -------: | ----: | --------: | -------------: |
|     0–20 |     4 |      13.8 |           13.8 |
|    20–40 |    76 |      39.3 |           29.4 |
|    40–60 |   104 |      68.0 |           62.8 |
|    60–80 |    34 |     103.1 |           94.0 |
|   80–100 |    19 |     106.2 |          103.1 |
|  100–128 |    13 |     157.9 |          158.0 |

## Reproduction Commands

```bash
# USearch 10M production-scale sweep — 8 GPUs
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
  ./build/bench/bench_pipeline_throughput \
  --smi /tmp/usearch_10M.smi \
  --conformers 3 \
  --multi-gpu-only

# USearch 1M single-GPU
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
  ./build/bench/bench_pipeline_throughput \
  --smi /tmp/usearch_1M_clean.smi \
  --conformers 1 \
  --multi-gpu-only

# Quality check — ChEMBL 9K, 3 conformers, 256 samples
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
  ./build/bench/bench_pipeline_throughput \
  --smi /tmp/chembl_valid.smi \
  --conformers 3 \
  --quality-samples 256
```
