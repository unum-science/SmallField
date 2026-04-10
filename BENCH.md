# SmallField Benchmark: ETKDG + MMFF94 Conformer Generation

Methodology and results for comparing CPU (RDKit) and GPU (SmallField) conformer generation pipelines.

## Hardware

| Component | Specification                                                      |
| --------- | ------------------------------------------------------------------ |
| GPU       | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition                    |
| CPU       | 2x Intel Xeon 6776P: 48 cores / 96 threads each, 192 threads total |
| RAM       | 1.7 TB DDR5                                                        |

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
- `params.numThreads = 1` to disable RDKit-internal threading, as our OpenMP handles parallelism
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
- __Wave-based dispatch:__ molecules sorted by atom count, launched in 4 waves (<=32, <=64, <=96, <=128 atoms).
  Small molecules get up to 16 blocks/SM vs 2 for N=128.
- __Philox RNG:__ ~100 cycles init vs ~1000 for XORWOW.
  Counter-based, stateless.

## Results

Dataset: 8,948 ChEMBL molecules, 3 conformers each, single GPU.

### Throughput scaling

| Batch | RDKit CPU mol/s | SmallField GPU mol/s | GPU speedup |
| ----- | --------------- | -------------------- | ----------- |
| 100   | 20              | 37                   | 1.9x        |
| 500   | 95              | 161                  | 1.7x        |
| 1,000 | 125             | 286                  | 2.3x        |
| 2,000 | 300             | 536                  | 1.8x        |
| 5,000 | 558             | 877                  | 1.6x        |
| 8,948 | 620             | 1,266                | 2.0x        |

RDKit CPU uses 192 OpenMP threads with `schedule(dynamic, 1)`, `numThreads=1` per molecule.
GPU throughput is end-to-end (preprocessing + upload + kernel + download).

### GPU time breakdown (8,948 molecules)

| Phase                      | Time         | %     |
| -------------------------- | ------------ | ----- |
| CPU preprocessing (OpenMP) | 1,119 ms     | 15.8% |
| GPU upload (PCIe)          | 67 ms        | 0.9%  |
| GPU kernel (4 waves)       | 5,913 ms     | 83.1% |
| GPU download               | 9 ms         | 0.1%  |
| __Total__                  | __7,069 ms__ |       |

### Wave dispatch breakdown

| Wave | Max atoms | Molecules | Blocks/SM | Grid  |
| ---- | --------- | --------- | --------- | ----- |
| 0    | 32        | 1,548     | 16        | 1,548 |
| 1    | 64        | 5,316     | 8         | 1,504 |
| 2    | 96        | 1,574     | 4         | 752   |
| 3    | 128       | 510       | 2         | 376   |

## Quality

Cross-validation: 10,000 random conformers evaluated by RDKit's `MMFF::calcEnergy()`.

### Summary

| Metric                                 | Value                   |
| -------------------------------------- | ----------------------- |
| Kernel success rate (stereo check)     | 97.0% (26,049 / 26,844) |
| Finite MMFF94 energy (RDKit cross-val) | 97.3%                   |
| Median RDKit MMFF94 energy             | 9.7M kcal/mol           |
| Expected for valid conformers          | < 200 kcal/mol          |

97% of conformers pass the pipeline's internal stereo check and produce finite energies when cross-validated by RDKit.
However, the median energy is ~50,000x higher than a valid conformer, indicating the 3D geometry is not chemically reasonable.

### Per-atom-count breakdown

| Atoms   | Count | RDKit P50 (kcal/mol) | SmallField P50 (kcal/mol) |
| ------- | ----- | -------------------- | ------------------------- |
| 0-20    | 198   | 6.2M                 | 168M                      |
| 20-40   | 3,343 | 9.5M                 | 487M                      |
| 40-60   | 3,761 | 9.8M                 | 1.2B                      |
| 60-80   | 1,539 | 9.9M                 | 1.8B                      |
| 80-100  | 601   | 8.7M                 | 2.5B                      |
| 100-128 | 292   | 10.5M                | 3.3B                      |

### Diagnosis

The pipeline produces correctly-laid-out 3D coordinates (97% finite), but the geometry is far from any MMFF94 local minimum.

1. __Random 4D box initialization (stage 3):__ coordinates are uniform random in `[-max_upper/2, max_upper/2]`.
   For a 60-atom molecule this means atoms start at random positions in a ~15 angstrom cube, with severe VdW clashes and wrong bond lengths.
   RDKit uses metric matrix embedding (eigendecomposition of the squared distance matrix) which produces a geometry that approximately satisfies the distance bounds from the start.

2. __Insufficient DG minimization (stage 4):__ 100 FIRE iterations with displacement clamping at +/-0.5 angstrom cannot traverse the >10 angstrom distances needed to reach the correct basin from a random start.

3. __Missing MMFF94 terms:__ the GPU functor implements VdW, electrostatic, bond stretching, angle bending, and torsion -- but is missing out-of-plane bending and stretch-bend coupling, which contribute to planarity of sp2 centers and coupling between bond/angle deformations.

## Next improvements

### Quality (highest priority)

1. __Metric matrix embedding for stage 3.__
   Replace random box with eigendecomposition of the squared distance matrix derived from the smoothed bounds.
   This produces a starting geometry that approximately satisfies all distance constraints, eliminating the need for aggressive DG minimization.
   Can be implemented on GPU using warp-level Jacobi eigendecomposition for small matrices (N <= 128).

2. __Add out-of-plane bending term.__
   MMFF94 requires `E_oop = 0.5 * 0.043844 * koop * chi^2` for sp2 centers.
   Without this, aromatic rings and carbonyl groups are not penalized for non-planarity.
   Requires extracting OOP parameters during preprocessing and adding a new interaction type to the per-atom interaction list.

3. __Add stretch-bend coupling.__
   `E_sb = 2.5121 * (kba_ijk * dr_ij + kba_kji * dr_kj) * dtheta`.
   Couples bond stretching with angle bending for more accurate geometries near equilibrium.

### Performance

4. __Multi-GPU dispatch.__
   8 GPUs available but only 1 used.
   Partition the molecule batch across GPUs with separate CUDA streams.
   Expected ~8x throughput for large batches (kernel-bound).

5. __Overlap preprocessing with kernel execution.__
   Currently sequential.
   With double-buffering, wave N+1's preprocessing can overlap with wave N's kernel execution on the GPU.

6. __Reduce CSD torsion extraction cost.__
   `extract_etk_torsions` calls RDKit's `getExperimentalTorsions()` which takes ~88% of preprocessing time.
   Caching torsion parameters by SMARTS pattern or bond environment hash could eliminate redundant lookups.

7. __Finer wave bins.__
   Current 4 bins (32/64/96/128) leave occupancy gaps.
   8 bins at 16-atom increments would better match the molecule size distribution and increase average occupancy.
