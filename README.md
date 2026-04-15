![SmallField conformer generation library](https://github.com/ashvardanian/ashvardanian/raw/master/repositories/SmallField.jpg?raw=true)

__SmallField__ generates 3D conformers from SMILES strings using [ETKDG](https://doi.org/10.1021/acs.jcim.5b00654) distance geometry with experimental torsion preferences followed by [MMFF94](https://doi.org/10.1006/jcat.1996.0027) force field optimization, entirely on the GPU.
One CUDA warp processes one molecule and runs all pipeline stages in shared memory with no global-memory atomics.
CPU preprocessing uses [RDKit](https://github.com/rdkit/rdkit) for SMILES parsing, bounds matrices, CSD torsion lookup, and MMFF94 parameter extraction.

## Pipeline

__CPU (OpenMP, per-molecule):__

SMILES parse + addHs → bounds matrix → chiral centers → CSD torsion Fourier coefficients → MMFF94 params + per-atom interaction lists.

__GPU (one warp per molecule, shared memory):__

| #    | Stage              | Method                                                    |
| :--- | :----------------- | :-------------------------------------------------------- |
| 2    | Triangle smoothing | Floyd-Warshall on half-precision bounds                   |
| 3    | Native metric init | SmallField classical MDS embedding + mirrored retries     |
| 4    | DG minimize        | FIRE optimizer, 4D, atom-centric                          |
| 5    | Stereo check       | Chiral volume sign validation                             |
| 6    | 4th-dim crush      | FIRE with w&#8324; penalty                                |
| 7    | ETK torsion refine | FIRE, 3D, 6-term Fourier + distance constraints           |
| 8    | Final validation   | Repeat stereo check                                       |
| 9    | MMFF94 polish      | FIRE, 3D, full force field (VdW + electrostatic + bonded) |

## Architecture

- __Atom-centric:__ each thread owns ceil(N/32) atoms, accumulates gradients in registers via `__shfl_sync`.
  No `atomicAdd` is needed, and there is no pair-index scanning.
- __Half-precision bounds:__ `__half` storage halves shared memory for the N×N bounds matrix, enabling up to 16 blocks/SM for small molecules.
- __Native metric embedding:__ stage 3 starts from a SmallField-owned bounds-matrix embedding instead of RDKit coordinates.
- __FIRE optimizer:__ 3 shared arrays (positions + velocities + gradient) vs 15 for L-BFGS.
  Velocity Verlet integration is used with displacement clamping.
- __Wave dispatch:__ molecules are sorted by atom count and launched in 8 waves (`≤16, 32, 48, 64, 80, 96, 112, 128` atoms).
  Small molecules get up to 20 blocks/SM, and the two largest waves get 1.
- __Per-atom interaction lists:__ two-pass build (count, then fill) during preprocessing.
  The GPU iterates only the relevant terms per atom.

## Performance

Measured on ChEMBL 8,948 drug-like molecules with $N \le 128$ atoms including explicit hydrogens, with 3 conformers per molecule.
See [BENCH.md](BENCH.md) for full methodology.

| System         | Hardware                    |   Mol/s | Conformers/s |
| :------------- | :-------------------------- | ------: | -----------: |
| __SmallField__ | 1x RTX PRO 6000 Blackwell   | __797__ |        2,312 |
| RDKit CPU      | 2x Xeon 6776P (192 threads) |     410 |        1,122 |

NVIDIA's [nvMolKit](https://docs.nvidia.com/bionemo/) also targets GPU conformer generation, but it requires the BioNeMo proprietary stack.
SmallField is open-source, runs on any CUDA 12+ GPU, and uses a single persistent warp-loop kernel.

## Python

```python
import smallfield

coords = smallfield.generate_conformers("c1ccc(CC(=O)O)cc1", num_conformers=10)
print(coords.shape)  # (K, N, 3) where K = successful conformers
```

Requires RDKit for preprocessing.
Install it with `pip install rdkit`.
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
make -j$(nproc) bench_pipeline_throughput

LD_LIBRARY_PATH=/path/to/conda/env/lib:$LD_LIBRARY_PATH \
    ./bench/bench_pipeline_throughput --smi /path/to/chembl.smi
```

This runs both the RDKit CPU path with OpenMP and `numThreads=1` per molecule, and the SmallField GPU path.
It reports per-stage timing, the wave-dispatch breakdown, and quality cross-validation against RDKit's MMFF94 energy evaluator.

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
src/api/                            public C++ API (smallfield.hpp, batch_pool.hpp)
src/types/                          plain data structures shared host + device
src/parameters/                     MMFF94 parameter tables (host extraction, device upload)
src/preprocess/                     CPU preprocessing: SMILES parse, bounds, CIP, CSD torsions, scaffold cache
src/mmff94/                         native MMFF94 typer, charge, term extractor (CPU + GPU)
src/kernels/                        CUDA kernels: metric embedding, DG, ETK, MMFF, FIRE, pipeline driver
binary/                             smallfield_batch — restartable CLI over Parquet shards
test/                               C++ tests cross-validated against RDKit
bench/                              C++ benchmarks: pipeline throughput + isolated MMFF kernels
python/smallfield/                  Python package including tools/smi_to_parquet.py
```

There are no dependencies beyond CUDA 12+ and RDKit for preprocessing.
Apache Arrow + Parquet are optional and only needed for the batch CLI binary.

## License

Apache 2.0
