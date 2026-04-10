![SmallField conformer generation library](https://github.com/ashvardanian/ashvardanian/raw/master/repositories/SmallField.jpg?raw=true)

__SmallField__ generates 3D conformers from SMILES strings using [ETKDG](https://doi.org/10.1021/acs.jcim.5b00654) distance geometry with experimental torsion preferences followed by [MMFF94](https://doi.org/10.1006/jcat.1996.0027) force field optimization — entirely on the GPU.
One CUDA warp processes one molecule, running all pipeline stages in shared memory with no global memory atomics.
CPU preprocessing uses [RDKit](https://github.com/rdkit/rdkit) for SMILES parsing, bounds matrices, CSD torsion lookup, and MMFF94 parameter extraction.

## Pipeline

__CPU (OpenMP, per-molecule):__ SMILES parse + addHs → bounds matrix → chiral centers → CSD torsion Fourier coefficients → MMFF94 params + per-atom interaction lists.

__GPU (one warp per molecule, shared memory):__

| #    | Stage              | Method                                                    |
| :--- | :----------------- | :-------------------------------------------------------- |
| 2    | Triangle smoothing | Floyd-Warshall on half-precision bounds                   |
| 3    | Random 4D init     | Philox counter-based RNG                                  |
| 4    | DG minimize        | FIRE optimizer, 4D, atom-centric                          |
| 5    | Stereo check       | Chiral volume sign validation                             |
| 6    | 4th-dim crush      | FIRE with w&#8324; penalty                                |
| 7    | ETK torsion refine | FIRE, 3D, 6-term Fourier + distance constraints           |
| 8    | Final validation   | Repeat stereo check                                       |
| 9    | MMFF94 polish      | FIRE, 3D, full force field (VdW + electrostatic + bonded) |

## Architecture

- __Atom-centric:__ each thread owns ceil(N/32) atoms, accumulates gradients in registers via `__shfl_sync`.
  No `atomicAdd`, no pair index scanning.
- __Half-precision bounds:__ `__half` storage halves shared memory for the N×N bounds matrix, enabling up to 16 blocks/SM for small molecules.
- __FIRE optimizer:__ 3 shared arrays (positions + velocities + gradient) vs 15 for L-BFGS.
  Velocity Verlet integration with displacement clamping.
- __Wave dispatch:__ molecules sorted by atom count, launched in waves (≤32, ≤64, ≤96, ≤128 atoms).
  Small molecules get 16 blocks/SM, large get 2.
- __Per-atom interaction lists:__ two-pass build (count, then fill) during preprocessing.
  GPU iterates only relevant terms per atom.

## Performance

Measured on ChEMBL 8,948 drug-like molecules (≤128 atoms with H), 3 conformers each.
See [BENCH.md](BENCH.md) for full methodology.

| System         | Hardware                    |     Mol/s | Conformers/s |
| :------------- | :-------------------------- | --------: | -----------: |
| __SmallField__ | 1x RTX PRO 6000 Blackwell   | __1,245__ |        3,626 |
| RDKit CPU      | 2x Xeon 6776P (192 threads) |       627 |        1,717 |

NVIDIA's [nvMolKit](https://docs.nvidia.com/bionemo/) also targets GPU conformer generation but requires the BioNeMo proprietary stack.
SmallField is open-source, runs on any CUDA 12+ GPU, and uses a single persistent warp-loop kernel.

## Python

```python
import smallfield

coords = smallfield.generate_conformers("c1ccc(CC(=O)O)cc1", num_conformers=10)
print(coords.shape)  # (K, N, 3) where K = successful conformers
```

Requires RDKit for preprocessing - `pip install rdkit`.
The GPU pipeline context is also available directly:

```python
ctx = smallfield.EtkdgContext(max_atoms=128, max_molecules=1000, max_conformers=10)
coords, energies, status = ctx.generate_conformers(
    bounds_flat, chirals, torsions, constraints, metadata,
    num_conformers=10, max_stereo_attempts=20, random_seed=42)
```

## C++ Benchmark

```bash
cd build
cmake .. -DRDKIT_PREFIX=/path/to/conda/env -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) bench_chembl

LD_LIBRARY_PATH=/path/to/conda/env/lib:$LD_LIBRARY_PATH \
    ./bench/bench_chembl --smi /path/to/chembl.smi
```

Runs both RDKit CPU (OpenMP, `numThreads=1` per molecule) and SmallField GPU with per-stage timing, wave dispatch breakdown, and quality cross-validation against RDKit's MMFF94 energy evaluator.

## Installation

__Python:__

```bash
pip install -e .[gpu] # builds CUDA extension via nvcc
```

__C++ tests and benchmarks (requires RDKit from conda-forge):__

```bash
micromamba install -c conda-forge rdkit-dev
cd build && cmake .. -DRDKIT_PREFIX=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
```

## Project Layout

```
src/
  warp_primitives.cuh     warp_size, full_mask, warp_reduce_sum_f32_
  fire_warp.cuh           FIRE optimizer (warp-level, functor-generic)
  triangle_smooth.cuh     Floyd-Warshall on half-precision bounds
  etkdg_types.cuh         GPU data structures (metadata, descriptors, MMFF94 terms)
  dg_forcefield.cuh       distance geometry energy + gradient (atom-centric)
  etk_forcefield.cuh      ETK torsion Fourier energy + gradient (atom-centric)
  mmff94_functor.cuh      MMFF94 energy + gradient (per-atom interaction lists)
  etkdg_pipeline.cuh      persistent warp-loop kernel (stages 2-9)
  etkdg_preprocess.hpp    CPU preprocessing (RDKit -> GPU-ready arrays)
  smallfield.cu           pybind11 bindings
test/                     C++ tests cross-validated against RDKit
bench/                    C++ benchmark (RDKit CPU vs SmallField GPU)
python/smallfield/        Python package
```

~2,500 lines of CUDA/C++.
No dependencies beyond CUDA 12+ and RDKit for preprocessing only.

## License

Apache 2.0
