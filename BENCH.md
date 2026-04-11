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

| Batch | RDKit CPU mol/s | SmallField GPU mol/s | GPU speedup |
| ----- | --------------- | -------------------- | ----------- |
| 100   | 18              | 17                   | 0.9x        |
| 500   | 88              | 76                   | 0.9x        |
| 1,000 | 185             | 148                  | 0.8x        |
| 2,000 | 296             | 275                  | 0.9x        |
| 5,000 | 561             | 605                  | 1.1x        |
| 8,948 | 628             | 823                  | 1.3x        |

RDKit CPU uses 192 OpenMP threads with `schedule(dynamic, 1)` and `numThreads=1` per molecule.
GPU throughput is end-to-end (preprocessing + upload + prepass + kernel + download).
GPU overtakes CPU at ~5,000 molecules; at full dataset the GPU is 1.3x faster.
For smaller batches the GPU prepass and upload overhead dominate, making it slower than 192 CPU cores.

### GPU time breakdown (8,948 molecules)

| Phase                      | Time          | %     |
| -------------------------- | ------------- | ----- |
| CPU preprocessing (OpenMP) | 1,249 ms      | 11.5% |
| GPU upload + prepass       | 994 ms        | 9.1%  |
| GPU kernel (8 waves)       | 8,596 ms      | 79.1% |
| GPU download               | 31 ms         | 0.3%  |
| __Total__                  | __10,870 ms__ |       |

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

### Per-wave kernel profiling (nsys, 8,948 molecules)

Profiled with `nsys profile` (CUDA 13.0, Blackwell SM 12.0).
Pipeline kernel accounts for 97.5% of total GPU time; metric embedding prepass is 2.5%.

| Wave | Atoms | Mols  | Grid  | Kernel ms | ms/mol | us/conf | % of kernel |
| ---- | ----- | ----- | ----- | --------- | ------ | ------- | ----------- |
| 0    | 16    | 46    | 46    | 59        | 1.28   | 425     | 0.4%        |
| 1    | 32    | 1,502 | 1,502 | 465       | 0.31   | 103     | 3.0%        |
| 2    | 48    | 3,049 | 2,256 | 1,669     | 0.55   | 182     | 10.7%       |
| 3    | 64    | 2,267 | 1,504 | 2,077     | 0.92   | 305     | 13.4%       |
| 4    | 80    | 1,023 | 940   | 2,715     | 2.65   | 885     | 17.5%       |
| 5    | 96    | 551   | 551   | 2,376     | 4.31   | 1,437   | 15.3%       |
| 6    | 112   | 310   | 310   | 2,865     | 9.24   | 3,081   | 18.4%       |
| 7    | 128   | 200   | 200   | 3,302     | 16.51  | 5,503   | 21.3%       |

Per-molecule cost scales ~50x from N=32 to N=128.
Waves 4-7 (atoms > 64) contain only 23% of molecules but consume 72% of kernel time.
The dominant cost is the O(N²) pairwise MMFF94 interaction computation in stage 9.

## Quality

Cross-validation uses 10,000 random SmallField conformers evaluated by RDKit's `MMFF::calcEnergy()`.

### Summary

| Metric                                              | Value                    |
| --------------------------------------------------- | ------------------------ |
| Kernel success rate (stereo + MMFF path)            | 96.2% (25,822 / 26,844) |
| Finite MMFF94 energy (RDKit cross-val)              | 97.4%                    |
| Median RDKit MMFF94 energy of SmallField conformers | 58.5 kcal/mol            |
| Median SmallField final MMFF94 energy               | 58.5 kcal/mol            |
| P5 / P95 RDKit energy                               | -67.2 / 183.1 kcal/mol  |
| Expected for valid conformers                       | < 200 kcal/mol           |

The catastrophic-energy regime is gone.
On the corrected SmallField-native path with GPU metric embedding, the typical conformer lands in a chemically reasonable MMFF94 range.
The SmallField MMFF energy and the RDKit MMFF energy still track closely on the same sampled conformers.
The remaining quality issue is the upper tail on the heaviest bins.
Median cross-validated energy stays below 200 kcal/mol through the 80-100 atom bin, then rises above 200 kcal/mol only for the 100-128 atom tail.

### Per-atom-count breakdown

| Atoms   | Count | RDKit P50 (kcal/mol) | SmallField P50 (kcal/mol) |
| ------- | ----- | -------------------- | ------------------------- |
| 0-20    | 191   | 0.6                  | 0.6                       |
| 20-40   | 3,366 | 36.0                 | 36.0                      |
| 40-60   | 3,737 | 65.1                 | 65.2                      |
| 60-80   | 1,584 | 83.8                 | 83.8                      |
| 80-100  | 603   | 121.2                | 121.2                     |
| 100-128 | 251   | 149.3                | 149.3                     |

### Diagnosis

The current bottlenecks are different from the original report.

1. __Large-molecule tail.__
   The GPU native metric embedding fixes the startup geometry, but molecules above roughly 100 atoms still carry a noticeably heavier MMFF tail after stage 9.

2. __GPU prepass cost.__
   Moving metric embedding onto the GPU removed the old CPU eigendecomposition cost, but the upload-plus-prepass phase is now about 15.2% of end-to-end time on the full dataset.
   That is acceptable, but it is now large enough to deserve its own tuning pass.

3. __Kernel remains dominant.__
   Even with 8 wave bins, the GPU kernel is still about 68.6% of end-to-end runtime on the full dataset.
   Any further throughput work now needs to focus on reducing stage-4, stage-7, and stage-9 iteration cost instead of fixing startup geometry.

## Next improvements

1. __Reduce the >80 atom quality tail.__
   The accuracy floor is now acceptable, but large molecules still drift above the `<200 kcal/mol` target too often.
   The next quality pass should focus on stronger ETK and MMFF cleanup for the two largest bins.

2. __Trim the prepass overhead.__
   The architecture is now end-to-end on the GPU after batch construction, but the upload-plus-prepass phase is still over a second on the full dataset.
   The next pass should focus on reducing workspace traffic, grouping overhead, and global-memory writes in the prepass.

3. __Parallelize ETK interactions like MMFF.__
   Stage 7 still scans torsions and constraints less efficiently than stage 9's per-atom adjacency layout.

4. __Overlap preprocessing and execution.__
   Double-buffering batches would hide part of the 1.46 s CPU preprocessing cost behind the GPU waves.

5. __Multi-GPU dispatch.__
   The current benchmark still uses a single GPU on an 8-GPU machine.
