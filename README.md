![SmallField conformer generation library](https://github.com/ashvardanian/ashvardanian/raw/master/repositories/SmallField.jpg?raw=true)

__SmallField__ generates 3D conformers from SMILES strings using [ETKDG](https://doi.org/10.1021/acs.jcim.5b00654) distance geometry with experimental torsion preferences followed by [MMFF94](https://doi.org/10.1006/jcat.1996.0027) force field optimization, entirely on the GPU.
One CUDA warp processes one molecule and runs all pipeline stages in shared memory with no global-memory atomics.
CPU preprocessing uses [RDKit](https://github.com/rdkit/rdkit) for SMILES parsing, bounds matrices, CSD torsion lookup, and MMFF94 parameter extraction.

## Project Goals & FAQ

SmallField is a 3D conformer generator, not a chemical-identity toolkit.
Given one SMILES string, SmallField produces K 3D geometries consistent with that SMILES — nothing more.
Questions that interconvert distinct SMILES (tautomer canonicalisation, stereo enumeration, protonation-state prediction) are upstream of this library and belong in RDKit or a dedicated standardiser.

__Why are tautomers not enumerated?__
A tautomer pair has different SMILES — different bond positions, different mobile-H placement — so two tautomers of the same compound are two distinct inputs to SmallField.
Each input gets its own conformer pool and its own 3D geometry, which is what the user asked for when they handed us that SMILES.
If you want tautomer coverage, do it before calling SmallField: run `RDKit::MolStandardize::TautomerEnumerator::Enumerate(mol)`, write the N tautomer SMILES as N separate rows, and feed the enlarged list to `generate_conformers`.
SmallField is deterministic on SMILES input, so downstream consumers can still group by a canonical-tautomer hash if they want compound-level dedup.
Having SmallField silently substitute a different tautomer than the one you provided would be wrong; we respect the SMILES.

__Why is point (tetrahedral) chirality not enumerated?__
Same reason: distinct stereoisomers are distinct SMILES.
If an input SMILES has all stereocentres assigned, we use that assignment; if a centre is unspecified, ETKDG picks one handedness deterministically from the random seed.
For bulk corpora that ship every centre specified (ChEMBL, USearch, PubChem canonical), this is zero information loss — a 100 K sample of USearch and ChEMBL both measured 0.00 % of molecules with undefined stereocentres.

__What about axial and helical chirality?__
These are geometric chirality modes (atropisomerism of ortho-substituted biaryls, helicenes) that are not captured by a SMILES string at all — two molecules with opposite helicity have the same SMILES.
ETKDG embeds one handedness arbitrarily from the random seed.
Sampling shows axial chirality in 0.28–0.44 % of molecules and helical chirality in 0.11–0.18 %.
These are corner cases but real: a user counting on both enantiomers of a given atropisomer will get only one.
Flagged future work is to detect and tag such molecules in the output so downstream consumers can re-embed them with constrained stereo, not to silently double the row count.

__What is in scope?__
Parsing any valid SMILES that RDKit accepts.
Producing K 3D conformers per input, filtered by stereo validity (for defined stereocentres) and MMFF94 convergence.
Deterministic behaviour under a fixed random seed.
Throughput at scale — millions of molecules per hour on an 8-GPU host.

__What is not in scope?__
Canonicalising tautomers, protonation states, or kekulisations.
Enumerating stereoisomers, tautomers, or protomers.
SMILES-equivalence testing or cross-input dedup.
Energy ranking beyond what MMFF94 polish computes.
Solvent effects, pH corrections, or ensemble averaging over interconverting forms.

## What's New

The seven things SmallField does that a baseline RDKit pipeline does not.

__One CUDA warp per molecule, all pipeline stages in shared memory, no global-memory atomics.__
Each thread of a warp owns `ceil(N/32)` atoms and accumulates gradients in registers via `__shfl_sync`; there is no `atomicAdd` and no pair-index scanning anywhere on the device.

__Int8 distance-bounds matrix.__
The N×N bounds live in shared memory as `uint8_t` with a per-molecule linear dequantization, halving the footprint against `__half` — 32 KB to 16 KB at 128 atoms — so more molecules stay resident per multiprocessor.

__Native classical-MDS metric embedding warm start.__
The metric-embedding stage starts from a SmallField-owned bounds-matrix embedding instead of RDKit's coordinate generator, which removes the largest preprocess step from the host critical path.

__FIRE optimizer instead of L-BFGS.__
Three shared arrays (positions, velocities, gradient) carry the entire optimiser state vs the fifteen L-BFGS would need; the saved shared memory pays for the warp residency above.

__Native CSD torsion matcher.__
All 470 SMARTS patterns shipped with RDKit's experimental-torsion library parse natively and run via a stack-only matcher, byte-equivalent to `RDKit::SubstructMatch` over a 1 M USearch sample.
There is no VF2 fallback: a pattern the compiler cannot read is a load-time error rather than a silent runtime detour.
This removed the VF2 + query-chain CPU hot-spot at 10 M scale entirely.

__Per-atom interaction lists.__
Built in two passes (count, then fill) on the CPU; the GPU iterates only the relevant terms per atom rather than scanning the full pair list.

## Pipeline

> See [`GLOSSARY.md`](GLOSSARY.md) for the canonical vocabulary — `parquet-shard` / `coprocessed-batch` / `embedding-wave` / `atom-limited-bin` at the data-granularity level, and `pipeline-role` (CLI outer pipeline) vs `wave-step` (library inner pipeline) for the pipelining layers.

CPU side, parallelised over molecules with OpenMP: SMILES parse with `addHs` → bounds matrix → chiral centres → CSD torsion Fourier coefficients → MMFF94 params + per-atom interaction lists.

GPU side, one warp per molecule, all data resident in shared memory:

| #   | Stage                | Method                                                    |
| :-- | :------------------- | :-------------------------------------------------------- |
| 1   | Bounds quantisation  | fp32 → `__half` distance bounds upload                    |
| 2   | Triangle smoothing   | Floyd-Warshall on half-precision bounds                   |
| 3   | Native metric init   | SmallField classical MDS embedding + mirrored retries     |
| 4   | DG minimize          | FIRE optimizer, 4D, atom-centric                          |
| 5   | Stereo check         | Chiral volume sign validation                             |
| 6   | 4th-dim crush        | FIRE with w&#8324; penalty                                |
| 7   | ETK torsion refine   | FIRE, 3D, 6-term Fourier + distance constraints           |
| 8   | Final validation     | Repeat stereo check                                       |
| 9   | MMFF94 polish        | FIRE, 3D, full force field (VdW + electrostatic + bonded) |

## Architecture

Wave dispatch sorts molecules by atom count and launches them in eight waves capped at 16, 32, 48, 64, 80, 96, 112, and 128 atoms.
Small-molecule waves run up to twenty blocks per SM; the two largest get one each.
The persistent warp loop keeps the same kernel resident across all stages, avoiding launch overhead between FIRE iterations and between phases.

The CPU pipeline runs in two layers.
The outer layer is CLI-level: a reader role streams Parquet shards into coprocessed batches, GPU-runner roles consume them, a writer role emits per-shard output, and a fallback role handles whatever the GPU could not converge.
The inner layer is library-level inside `generate_conformers`: a preprocess step on the host, an embedding step on the GPU, and a finalize step that drains the stream into the caller's buffer.
Wave N+1's preprocess overlaps wave N's embedding via a depth-2 batch-pool ring.

## Performance

On an 8× RTX PRO 6000 Blackwell / 2× Xeon 6776P host, SmallField generates ETKDG + MMFF94 conformers up to __9.7× faster than RDKit__ across 192 CPU threads, at 16,826 mol/s on ChEMBL 241,596 with one conformer per molecule.
Every number, with its corpus, its hardware and its methodology, lives in [`BENCH.md`](BENCH.md) — including single-H100 figures and the nvMolKit comparison.

NVIDIA's [nvMolKit](https://docs.nvidia.com/bionemo/) also targets GPU conformer generation, but it requires the BioNeMo proprietary stack.
SmallField is open-source and runs on any CUDA 12+ GPU.

## Python

ETKDG + MMFF94 conformer generation:

```python
import smallfield.api as sf

coordinates = sf.generate_conformers("c1ccc(CC(=O)O)cc1", num_conformers=10)
print(coordinates.shape)  # (K, N, 3) where K = successful conformers
```

The GPU pipeline context is also available directly:

```python
context = sf.Context()
coordinates_flat, energies, status = context.generate_conformers(
    ["CCO", "c1ccccc1"], num_conformers=3, max_atoms_per_molecule=128,
    max_stereo_attempts=10, random_seed=42,
)
```

Requires RDKit for preprocessing — `pip install rdkit`.

## C++ API

```cpp
#include "smallfield/api/smallfield.hpp"        // ETKDG + MMFF94

smallfield::api::context_t etkdg_ctx;
etkdg_ctx.generate_conformers(smiles, config, max_atoms,
                              coords_half, energies, status);
```

## CLI

Restartable parquet → parquet binary that streams shards through pipeline-roles (reader → runner → writer) and resumes from a partially-completed output folder.

```bash
smallfield_batch --input chembl_smi/ --output chembl_3d_mmff/ \
    --num-conformers 1 --gpus 0,1,2,3,4,5,6,7
```

## Installation

There are no wheels: SmallField compiles CUDA and links RDKit, so every install builds from source and needs `nvcc`, CMake, and the RDKit development headers already present.

```bash
pip install --no-build-isolation "git+https://github.com/unum-science/SmallField.git"
```

`--no-build-isolation` matters — under build isolation `pip` cannot see the RDKit and CUDA installation the build shells out to.
From a clone, `pip install --no-build-isolation -e .` does the same thing.

`pixi.toml` pins the environment this is developed and tested in — Python 3.14, RDKit 2025.09, Arrow 24 — and `pixi install` reproduces it.

C++ tests and benchmarks need RDKit from conda-forge:

```bash
micromamba install -c conda-forge rdkit-dev
cmake -S . -B build -DRDKIT_PREFIX=$CONDA_PREFIX -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

## C++ Benchmarks

```bash
cmake --build build -j$(nproc) --target bench_conformers

# ETKDG + MMFF94 throughput (RDKit CPU vs SmallField GPU)
./build/bench/bench_conformers --smi /path/to/chembl.smi
```

`--multi-gpu-only` on `bench_conformers` reports isolated 1/2/4/8-GPU scaling.

## Project Layout

```
src/smallfield/api/         public C++ API (smallfield.hpp)
src/smallfield/types/       plain data structures shared host + device
src/smallfield/parameters/  MMFF94 parameter tables
src/smallfield/preprocess/  CPU preprocessing: SMILES parse, bounds, CIP, CSD torsions
src/smallfield/mmff94/      native MMFF94 typer, charge, term extractor (CPU + GPU)
src/smallfield/cuda/        CUDA kernels: metric embedding, DG, ETK, MMFF, FIRE
cli/                        smallfield_batch (gen) — Parquet → Parquet
test/                       C++ and Python tests cross-validated against RDKit
bench/                      bench_conformers + bench_mmff_kernels + bench_profile
python/smallfield/          Python package: api.py (generate)
```

There are no dependencies beyond CUDA 12+ and RDKit for preprocessing.
Apache Arrow + Parquet are optional and only needed for the batch CLI binary.

## License

Apache 2.0
