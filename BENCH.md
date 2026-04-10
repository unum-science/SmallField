# SmallField Benchmark: ETKDG + MMFF94 Conformer Generation

Methodology and results for comparing CPU (RDKit) and GPU (SmallField) conformer generation pipelines.

## Hardware

| Component | Specification                                                                  |
| --------- | ------------------------------------------------------------------------------ |
| GPU       | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (188 SMs, SM 12.0, 96 GB each) |
| CPU       | 2x Intel Xeon 6776P (48 cores / 96 threads each, 192 threads total)            |
| RAM       | 1.7 TB DDR5                                                                    |

Benchmarks use a single GPU unless noted otherwise.

## Dataset

ChEMBL 10K drug-like molecules, pre-filtered to N <= 128 atoms (including explicit H).

| Stat            | Value  |
| --------------- | ------ |
| Total molecules | ~8,948 |
| Min atoms       | 8      |
| P25             | 36     |
| Median          | 48     |
| P75             | 63     |
| P90             | 84     |
| Max             | 128    |

Pre-filtering: parse SMILES, add explicit H, discard molecules with > 128 atoms or where `SmilesToMol` / `AddHs` fails.

Dataset: `/tmp/chembl_valid.smi` (one SMILES per line).

## Systems

### System A: RDKit CPU (our OpenMP dispatch)

- RDKit with ETKDGv3 + MMFF94 BFGS optimization
- Our OpenMP layer: `#pragma omp parallel for schedule(dynamic, 1)`
- Each molecule: `EmbedMultipleConfs(K)` then `MMFFOptimizeMoleculeConfs`
- `params.numThreads = 1` (no RDKit-internal threading; our OpenMP handles parallelism)
- Exception-safe: try-catch per molecule inside the parallel loop

### System B: SmallField GPU

- CPU preprocessing: SMILES parse, bounds matrix, CSD torsions, MMFF94 params (OpenMP)
- GPU kernel: one warp (32 threads) per molecule, all stages in shared memory
- Architecture: atom-centric gradient accumulation (no atomicAdd), half-precision bounds, per-atom MMFF94 interaction lists, Philox counter-based RNG
- Optimizer: FIRE (Fast Inertial Relaxation Engine) — 3 shared memory arrays vs 15 for L-BFGS

## Timing methodology

__RDKit CPU:__
1. Pre-parse ALL molecules (SmilesToMol + AddHs) — untimed, fair comparison
2. Timed region: OpenMP parallel `EmbedMultipleConfs` + `MMFFOptimizeMoleculeConfs`
3. Wall clock via `std::chrono::high_resolution_clock`
4. Report: molecules/sec, conformers/sec, success count, thread count

__SmallField GPU:__
1. Timed phase 1: `prepare_molecule_batch()` — CPU preprocessing with OpenMP
2. Timed phase 2: GPU memory upload + kernel execution (CUDA events, microsecond precision)
3. Report: preprocessing time, kernel time, total time, molecules/sec, occupancy

__Quality cross-validation:__
1. Random sample of 10K conformers across the full dataset
2. Set SmallField coordinates on RDKit mol, evaluate MMFF94 energy via RDKit
3. OpenMP-parallel cross-validation
4. Report: finite energy %, energy distribution percentiles, per-atom-count breakdown

## Pipeline stage reference

### CPU preprocessing (per-molecule, `src/etkdg_preprocess.hpp`)

| #   | Stage                             | Function                 | Line |
| --- | --------------------------------- | ------------------------ | ---- |
| 0a  | SMILES parse + addHs              | `prepare_molecule`       | 472  |
| 0b  | Bounds matrix                     | `build_bounds_matrix`    | 68   |
| 0c  | Chiral centers                    | `extract_chiral_centers` | 96   |
| 0d  | CSD torsions                      | `extract_etk_torsions`   | 186  |
| 0e  | MMFF94 params + interaction lists | `extract_mmff94_params`  | 265  |
| 0f  | Batch assembly                    | `prepare_molecule_batch` | 550  |

### GPU kernel (per-molecule, `src/etkdg_pipeline.cuh`)

| #   | Stage                             | Function                                                | File:Line                                  |
| --- | --------------------------------- | ------------------------------------------------------- | ------------------------------------------ |
| 2   | Triangle smoothing                | `triangle_smooth_warp_half_`                            | `triangle_smooth.cuh:44`                   |
| 3   | Random 4D init                    | `generate_random_coordinates_`                          | `etkdg_pipeline.cuh:50`                    |
| 4   | DG minimize (4D FIRE)             | `fire_minimize_` + `distance_geometry_energy_functor_t` | `fire_warp.cuh:40`, `dg_forcefield.cuh:91` |
| 5   | Stereo check                      | `validate_chirality_`                                   | `dg_forcefield.cuh:65`                     |
| 6   | 4th-dim crush (FIRE)              | Same as 4 with `fourth_dimension_penalty_weight = 10`   | `etkdg_pipeline.cuh:187`                   |
| 7   | ETK torsion refine (3D FIRE)      | `fire_minimize_` + `etk_torsion_energy_functor_t`       | `etk_forcefield.cuh:185`                   |
| 8   | Final validation                  | Same as 5                                               | `etkdg_pipeline.cuh:209`                   |
| 9   | MMFF94 polish (3D FIRE, 200 iter) | `fire_minimize_` + `mmff94_energy_functor_t`            | `mmff94_functor.cuh:273`                   |

### Key design choices

- __Atom-centric processing:__ each thread owns ceil(N/32) atoms, accumulates gradients in registers via `__shfl_sync`.
  Eliminates atomicAdd and pair index scanning.
- __Half-precision bounds:__ `__half` storage halves shared memory for the N x N bounds matrix, enabling 2 blocks/SM at N=128.
- __FIRE optimizer:__ 3 arrays (positions + velocities + gradient), velocity Verlet integration, displacement clamped to +/-0.5 angstrom per step.
- __Wave-based dispatch:__ molecules sorted by atom count, launched in 4 waves (<=32, <=64,
  <=96, <=128 atoms). Small molecules get up to 16 blocks/SM vs 2 for N=128.
- __Philox RNG:__ ~100 cycles init vs ~1000 for XORWOW.
  Counter-based, stateless.

## Results

[To be filled after benchmark runs]

## Quality

[To be filled after benchmark runs]
