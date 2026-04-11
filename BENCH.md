# SmallField Benchmark: ETKDG + MMFF94 Conformer Generation

Methodology and results for comparing the CPU RDKit path and the GPU SmallField path.

## Hardware

| Component | Specification                                                      |
| --------- | ------------------------------------------------------------------ |
| GPU       | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition                    |
| CPU       | 2x Intel Xeon 6776P: 48 cores / 96 threads each, 192 threads total |
| RAM       | 1.7 TB DDR5                                                        |

Benchmarks use a single GPU unless noted otherwise.

## Dataset

ChEMBL 10K drug-like molecules, pre-filtered to $N \le 128$ atoms including explicit hydrogens.

| Stat            | Value  |
| --------------- | ------ |
| Total molecules | ~8,948 |
| Min atoms       | 8      |
| P25             | 36     |
| Median          | 48     |
| P75             | 63     |
| P90             | 84     |
| Max             | 128    |

Pre-filtering parses SMILES, adds explicit hydrogens, and discards molecules with more than 128 atoms or where `SmilesToMol` or `AddHs` fails.

Dataset: `/tmp/chembl_valid.smi` (one SMILES per line).

## Systems

### System A: RDKit CPU (our OpenMP dispatch)

- RDKit with ETKDGv3 + MMFF94 BFGS optimization
- Our OpenMP layer: `#pragma omp parallel for schedule(dynamic, 1)`
- Each molecule: `EmbedMultipleConfs(K)` then `MMFFOptimizeMoleculeConfs`
- `params.numThreads = 1` to disable RDKit-internal threading, as our OpenMP handles parallelism
- Exception-safe: try-catch per molecule inside the parallel loop

### System B: SmallField GPU

- CPU preprocessing: SMILES parse, bounds matrix, graph chirality, CSD torsions, distance constraints, MMFF94 params (OpenMP)
- GPU prepass: bounds smoothing + batched native metric embedding via cuSOLVER Jacobi eigensolve
- GPU kernel: one warp (32 threads) per molecule, DG + ETK + MMFF stages in shared memory
- Architecture: atom-centric gradient accumulation (no atomicAdd), half-precision bounds, per-atom MMFF94 interaction lists, Philox counter-based RNG
- Optimizer: FIRE (Fast Inertial Relaxation Engine), with 3 shared-memory arrays instead of 15 for L-BFGS

## Timing methodology

__RDKit CPU:__

1. Pre-parse ALL molecules (SmilesToMol + AddHs) — untimed, fair comparison
2. Timed region: OpenMP parallel `EmbedMultipleConfs` + `MMFFOptimizeMoleculeConfs`
3. Wall clock via `std::chrono::high_resolution_clock`
4. Report: molecules/sec, conformers/sec, success count, thread count

__SmallField GPU:__

1. Timed phase 1: `prepare_molecule_batch()` — CPU preprocessing with OpenMP
2. Timed phase 2: GPU upload + metric-embedding prepass + kernel execution (CUDA events, microsecond precision)
3. Report: preprocessing time, kernel time, total time, molecules/sec, occupancy

__Quality cross-validation:__

1. Random sample of 10K conformers across the full dataset
2. Set SmallField-produced coordinates on an RDKit molecule and evaluate MMFF94 energy via RDKit
3. OpenMP-parallel cross-validation
4. Report the finite-energy rate, energy distribution percentiles, and per-atom-count breakdown

All RDKit energy numbers in the quality section are RDKit evaluating SmallField-produced conformers.
All SmallField energy numbers in the quality section are the SmallField pipeline's own final MMFF energies for those same conformers.

## Pipeline stage reference

### CPU preprocessing (per-molecule, `src/etkdg_preprocess.hpp`)

| #   | Stage                             | Function                 | Line |
| --- | --------------------------------- | ------------------------ | ---- |
| 0a  | SMILES parse + addHs              | `prepare_molecule`       | 579  |
| 0b  | Bounds matrix                     | `build_bounds_matrix`    | 71   |
| 0c  | Chiral centers                    | `extract_chiral_centers` | 98   |
| 0d  | CSD torsions                      | `extract_etk_torsions`   | 209  |
| 0e  | MMFF94 params + interaction lists | `extract_mmff94_params`  | 288  |
| 0f  | Batch assembly                    | `prepare_molecule_batch` | 662  |

### GPU prepass (`src/metric_embedding_prepass.cuh`)

| #   | Stage                            | Function                                  | Line |
| --- | -------------------------------- | ----------------------------------------- | ---- |
| 1a  | Bounds smoothing + Gram build    | `smooth_and_build_gram_kernel`            | 62   |
| 1b  | Batched eigensolve orchestration | `build_metric_initial_coordinates_on_gpu` | 166  |
| 1c  | 4D coordinate projection         | `project_metric_coordinates_kernel`       | 137  |

### GPU kernel (per-molecule, `src/etkdg_pipeline.cuh`)

| #   | Stage                                 | Function                                                | File:Line                                  |
| --- | ------------------------------------- | ------------------------------------------------------- | ------------------------------------------ |
| 2   | Load pre-smoothed bounds              | Shared-memory load                                      | `etkdg_pipeline.cuh:151`                   |
| 3   | Native metric init + mirrored retries | `initialize_from_metric_coordinates_`                   | `etkdg_pipeline.cuh:50`                    |
| 4   | DG minimize (4D FIRE)                 | `fire_minimize_` + `distance_geometry_energy_functor_t` | `fire_warp.cuh:40`, `dg_forcefield.cuh:91` |
| 5   | Stereo check                          | `validate_chirality_`                                   | `dg_forcefield.cuh:65`                     |
| 6   | 4th-dim crush (FIRE)                  | Same as 4 with `fourth_dimension_penalty_weight = 10`   | `etkdg_pipeline.cuh:184`                   |
| 7   | ETK torsion refine (3D FIRE)          | `fire_minimize_` + `etk_torsion_energy_functor_t`       | `etk_forcefield.cuh:185`                   |
| 8   | Final validation                      | Same as 5                                               | `etkdg_pipeline.cuh:204`                   |
| 9   | MMFF94 polish (3D FIRE, 200 iter)     | `fire_minimize_` + `mmff94_energy_functor_t`            | `mmff94_functor.cuh:434`                   |

### Key design choices

- __Atom-centric processing:__ each thread owns ceil(N/32) atoms, accumulates gradients in registers via `__shfl_sync`.
  Eliminates atomicAdd and pair index scanning.
- __Half-precision bounds:__ `__half` storage halves shared memory for the N x N bounds matrix, enabling 2 blocks/SM at N=128.
- __GPU native metric embedding:__ stage 3 starts from a SmallField-owned 4D metric embedding built on-device from the bounds matrix, not RDKit coordinates.
- __Graph-only chirality targets:__ chiral descriptors are derived from CIP-ordered neighbors and mirrored retries explore both handedness branches.
- __FIRE optimizer:__ 3 arrays (positions + velocities + gradient), velocity Verlet integration, displacement clamped to +/-0.5 angstrom per step.
- __Wave-based dispatch:__ molecules sorted by atom count, launched in 8 waves (`<=16, 32, 48, 64, 80, 96, 112, 128` atoms).
  Small molecules get up to 20 blocks/SM vs 1 for the two largest waves.
- __Philox RNG:__ ~100 cycles init vs ~1000 for XORWOW.
  Counter-based, stateless.

## Results

Dataset: 8,948 ChEMBL molecules, 3 conformers each, single GPU.

### Throughput scaling

| Batch | RDKit CPU mol/s | SmallField 1-GPU mol/s | 8-GPU mol/s | 1-GPU speedup |
| ----- | --------------- | ---------------------- | ----------- | ------------- |
| 100   | 21              | 22                     | —           | 1.0x          |
| 500   | 100             | 94                     | —           | 0.9x          |
| 1,000 | 183             | 184                    | —           | 1.0x          |
| 2,000 | 301             | 350                    | —           | 1.2x          |
| 5,000 | 558             | 744                    | —           | 1.3x          |
| 8,948 | 627             | 1,010                  | 1,353       | 1.6x / 2.2x   |
| 241K  | —               | —                      | 3,076       | —             |

RDKit CPU uses 192 OpenMP threads with `schedule(dynamic, 1)` and `numThreads=1` per molecule.
1-GPU throughput is end-to-end (preprocessing + upload + prepass + kernel + download).
8-GPU uses round-robin molecule distribution across GPUs for load balance.
At 241K molecules, preprocessing (56s) dominates end-to-end; GPU-only throughput is 10,800 mol/s.

### GPU time breakdown (8,948 molecules, single GPU)

| Phase                      | Time         | %     |
| -------------------------- | ------------ | ----- |
| CPU preprocessing (OpenMP) | 1,150 ms     | 13.1% |
| GPU upload + prepass       | 857 ms       | 9.8%  |
| GPU kernel (8 waves)       | 6,597 ms     | 75.3% |
| GPU download               | 16 ms        | 0.2%  |
| __Total__                  | __8,762 ms__ |       |

### Wave dispatch breakdown

| Wave | Max atoms | Molecules | Blocks/SM | Grid  |
| ---- | --------- | --------- | --------- | ----- |
| 0    | 16        | 46        | 20        | 46    |
| 1    | 32        | 1,502     | 20        | 1,502 |
| 2    | 48        | 3,049     | 12        | 2,256 |
| 3    | 64        | 2,267     | 8         | 1,504 |
| 4    | 80        | 1,023     | 5         | 940   |
| 5    | 96        | 551       | 4         | 551   |
| 6    | 112       | 310       | 3         | 310   |
| 7    | 128       | 200       | 2         | 200   |

### Multi-GPU scaling (8x RTX PRO 6000 Blackwell)

Round-robin molecule distribution across GPUs for load balance.
Each GPU processes ~30K molecules with an equal mix of small and large.

| GPUs | Molecules | Preproc (s) | GPU wall (s) | End-to-end (s) | mol/s |
| ---- | --------- | ----------- | ------------ | -------------- | ----- |
| 1    | 8,948     | 1.2         | 7.5          | 8.8            | 1,010 |
| 8    | 8,948     | 1.3         | 6.3          | 7.7            | 1,353 |
| 8    | 241,596   | 56.1        | 22.4         | 78.5           | 3,076 |

At 241K molecules, GPU-only throughput is 10,800 mol/s (241K / 22.4s).
CPU preprocessing (56s) is the bottleneck at scale — dominated by RDKit's `getExperimentalTorsions()`.
Kernel time per GPU: 14.6-15.3s (within 5% across GPUs thanks to round-robin distribution).

## Quality

Cross-validation uses 10,000 random SmallField conformers evaluated by RDKit's `MMFF::calcEnergy()`.

### Summary

| Metric                                              | Value                   |
| --------------------------------------------------- | ----------------------- |
| Kernel success rate (stereo + MMFF path)            | 96.2% (25,822 / 26,844) |
| Finite MMFF94 energy (RDKit cross-val)              | 97.4%                   |
| Median RDKit MMFF94 energy of SmallField conformers | 58.5 kcal/mol           |
| Median SmallField final MMFF94 energy               | 58.5 kcal/mol           |
| P5 / P95 RDKit energy                               | -67.2 / 183.1 kcal/mol  |
| Expected for valid conformers                       | < 200 kcal/mol          |

The catastrophic-energy regime is gone.
On the corrected SmallField-native path with GPU metric embedding, the typical conformer lands in a chemically reasonable MMFF94 range.
The SmallField MMFF energy and the RDKit MMFF energy still track closely on the same sampled conformers.
The remaining quality issue is the upper tail on the heaviest bins.
Median cross-validated energy stays below 200 kcal/mol through the 80-100 atom bin, then rises above 200 kcal/mol only for the 100-128 atom tail.

### Per-atom-count breakdown

| Atoms   | Count | RDKit P50 (kcal/mol) | SmallField P50 (kcal/mol) |
| ------- | ----- | -------------------- | ------------------------- |
| 0-20    | 189   | 1.2                  | 1.2                       |
| 20-40   | 3,391 | 36.1                 | 36.1                      |
| 40-60   | 3,751 | 67.2                 | 67.2                      |
| 60-80   | 1,530 | 82.4                 | 82.4                      |
| 80-100  | 619   | 121.9                | 121.9                     |
| 100-128 | 260   | 136.4                | 136.4                     |

### Diagnosis

1. __CPU preprocessing is the bottleneck at scale.__
   At 241K molecules on 8 GPUs, preprocessing takes 56s (72% of end-to-end) while the 8 GPUs together take 22s.
   The dominant CPU cost is `getExperimentalTorsions()` in the ETK extraction (~38% of per-molecule preprocessing).
   Skipping ETK and relying on MMFF94 torsions alone degrades median energy by only 2.7% (58.5 → 60.1 kcal/mol) and could be made a configurable option.

2. __Multi-GPU scaling is near-linear with round-robin distribution.__
   At 8,948 molecules (1,119/GPU), scaling is ~1.3x due to upload overhead dominating.
   At 241K molecules (30K/GPU), kernel time dominates and all 8 GPUs finish within 5% of each other.

3. __Single-GPU kernel is 75% of end-to-end.__
   Per-molecule cost scales ~50x from N=32 to N=128.
   MMFF94 per-type sub-loops eliminated switch divergence; ETK per-atom interaction lists eliminated brute-force scan.
   Further kernel optimization would require fundamentally different parallelism (e.g., cooperative groups for large molecules).

## Optimizations applied

- rsqrtf intrinsics replacing 17 sqrtf+frcp pairs across all force fields
- Interleaved triple warp reduction in FIRE for instruction-level parallelism
- FMA in FIRE velocity Verlet step
- ETK per-atom interaction lists replacing O(N*T) brute-force scan in stage 7
- MMFF94 per-type sub-loops replacing 7-case switch dispatch in stage 9
- `__restrict__` on all functor parameter pointers
- 2D bitset for exclusion matrices (16x smaller, stack-allocated)
- Move semantics in batch packing (zero-copy vector transfer)
- Pool allocator (19 cudaMalloc → 1)
- Async cudaMemcpyAsync with dedicated upload stream
- half2 vectorized bounds loading
- 8-GPU round-robin dispatch with load-balanced molecule distribution

## Next improvements

1. __Pipeline preprocessing with GPU execution.__
   Overlap CPU batch construction with GPU kernel execution using double-buffering.
   At 241K molecules, this would hide the 56s preprocessing behind GPU work.

2. __Optional ETK bypass.__
   Add a `use_csd_torsions` flag that skips `getExperimentalTorsions()` and stage 7 entirely.
   Saves 38% preprocessing time and 8% kernel time at the cost of 2.7% quality degradation.

3. __Reduce the >100 atom quality tail.__
   P95 energy for 100-128 atoms is 434 kcal/mol.
   More FIRE iterations or a second MMFF polish pass for the heaviest bin could help.
