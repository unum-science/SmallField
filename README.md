# SmallField

![SmallField conformer generation library](https://github.com/ashvardanian/ashvardanian/raw/master/repositories/SmallField.jpg?raw=true)

__SmallField__ generates 3D conformers from _"Simplified Molecular-Input Line-Entry System"_ SMILES strings using _"Experimental-Torsion Basic-Knowledge Distance Geometry"_ [ETKDG](https://doi.org/10.1021/acs.jcim.5b00654), followed by _"Merck Molecular Force Field"_ [MMFF94](https://doi.org/10.1002/%28SICI%291096-987X%28199604%2917%3A5/6%3C490%3A%3AAID-JCC1%3E3.0.CO%3B2-P) optimization, entirely on the GPU.
One CUDA warp processes one molecule and runs all pipeline stages in shared memory with no global-memory atomics.
CPU preprocessing uses [RDKit](https://github.com/rdkit/rdkit) for SMILES parsing, bounds matrices, _"Cambridge Structural Database"_ CSD torsion lookup, and MMFF94 parameter extraction.

> [!WARNING]
> Much of this project is AI-generated, and its correctness was defined by agreement with RDKit.
> RDKit carries its own mistakes and numerical instabilities, so exact agreement is neither always achievable nor always desirable.
> Validate on your own corpus before you depend on it.

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
For bulk corpora that ship every centre specified — ChEMBL, USearch, PubChem canonical — this is zero information loss, and a 100 K sample of USearch and ChEMBL both measured 0.00 % of molecules with undefined stereocentres.

One caveat that measurement does not cover: __CXSMILES enhanced-stereo groups are read as absolute configurations__.
A block such as `|&1:2,4|` declares those centres racemic, but `extract_chiral_centers` accepts any atom whose chiral tag survives `assignStereochemistry` and never consults `getStereoGroups`, so the written enantiomer is embedded and enforced as though it were asserted.
This affects 19.6 % of Enamine REAL rows.
Since chiral-volume validation is the only check that fails a conformer on the GPU, those centres are also enforced during embedding, which costs retries the input never asked for.

__What about axial and helical chirality?__
These are geometric chirality modes (atropisomerism of ortho-substituted biaryls, helicenes) that are not captured by a SMILES string at all — two molecules with opposite helicity have the same SMILES.
ETKDG embeds one handedness arbitrarily from the random seed.
Sampling shows axial chirality in 0.28–0.44 % of molecules and helical chirality in 0.11–0.18 %.
These are corner cases but real: a user counting on both enantiomers of a given atropisomer will get only one.
Flagged future work is to detect and tag such molecules in the output so downstream consumers can re-embed them with constrained stereo, not to silently double the row count.

__What happens to salts and multi-fragment SMILES?__
A SMILES naming several disconnected fragments is reduced to its largest one by default, and the counterions are dropped.
This is the one place SmallField does not hand the whole input to the embedder, and it is not a canonicalisation preference: a counterion has no meaningful conformer, and MMFF94 cannot type an isolated `HCl`, so keeping the fragment would reject the parent along with it.
`salt_policy_t::keep_all_fragments_k` restores the literal reading for callers who want it.
Conformers of a genuine multi-fragment system are laid out along the +x axis by rigid translation, so components do not interpenetrate and shape descriptors stay meaningful.

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
The N×N bounds live in shared memory as `uint8_t` with a per-molecule linear dequantization, halving the footprint against `__half` — 74 KB to 37 KB at 192 atoms — so more molecules stay resident per multiprocessor.
Quantization runs inside the pipeline kernel rather than on the wire, because the scale has to be derived from the post-smoothed values the metric prepass writes.

__Native classical _"multidimensional scaling"_ MDS metric embedding warm start.__
The metric-embedding stage starts from a SmallField-owned bounds-matrix embedding instead of RDKit's coordinate generator, which removes the largest preprocess step from the host critical path.

__The _"Fast Inertial Relaxation Engine"_ FIRE optimizer instead of _"limited-memory Broyden–Fletcher–Goldfarb–Shanno"_ L-BFGS.__
Three shared arrays (positions, velocities, gradient) carry the entire optimiser state vs the fifteen L-BFGS would need; the saved shared memory pays for the warp residency above.

__Native CSD torsion matcher.__
All 470 _"SMILES Arbitrary Target Specification"_ SMARTS patterns shipped with RDKit's experimental-torsion library parse natively and run via a stack-only matcher, byte-equivalent to `RDKit::SubstructMatch` over a 1 M USearch sample.
There is no VF2 fallback: a pattern the compiler cannot read is a load-time error rather than a silent runtime detour.
This removed the VF2 + query-chain CPU hot-spot at 10 M scale entirely.

__Preprocessing on the device, not just embedding.__
Everything downstream of `build_descriptor` runs on the GPU: the bond-hop BFS, MMFF94 typing, MMFF94 term extraction with compaction, CSD torsion matching, and the ETK per-atom CSR build.
The host keeps only the RDKit front-end — parse, sanitize, `addHs`, stereo — plus the bounds matrix and chiral centres, all of which are pure over `molecule_descriptor_t`.
CSD torsion matching alone dropped by more than an order of magnitude per molecule, and the pipeline is GPU-bound for the first time.
Moving the extractor also closed a silent quality hole: the host path dropped any term whose RDKit parameter lookup failed and shipped the molecule anyway, where the device path falls back to the empirical rules MMFF94 defines for that case.

__Per-atom interaction lists.__
Built in two passes — count, then fill — on the device alongside the term extraction that feeds them, so the GPU iterates only the relevant terms per atom rather than scanning the full pair list.

## Pipeline

> See [`GLOSSARY.md`](GLOSSARY.md) for the canonical vocabulary — `parquet-shard` / `coprocessed-batch` / `embedding-wave` / `atom-limited-bin` at the data-granularity level, and `pipeline-role` (CLI outer pipeline) vs `wave-step` (library inner pipeline) for the pipelining layers.

CPU side, parallelised over molecules with OpenMP: SMILES parse with `addHs` → molecule descriptor → bounds matrix → chiral centres → 1-2 and 1-3 distance constraints.
MMFF94 typing runs on the host only as a check, and only the molecules the native typer rejects pay for an RDKit patch.

Device side, everything downstream of the descriptor: bond-hop BFS → MMFF94 typing → MMFF94 term extraction with compaction → CSD torsion matching → ETK per-atom CSR build.
Nothing in that chain returns to the CPU; the host uploads a descriptor, a bounds matrix, chiral centres, the constraints and the per-molecule bases, and reads back coordinates.

Then the embedding itself, one warp per molecule, all data resident in shared memory:

| #    | Stage                       | Method                                                                |
| :--- | :-------------------------- | :-------------------------------------------------------------------- |
| 1    | Bounds upload               | fp32 host → device, unquantized on the wire                           |
| 2    | Triangle smoothing          | Floyd-Warshall on float bounds, fused into the prepass                |
| 3    | Native metric init          | SmallField classical MDS embedding + mirrored retries                 |
| 4    | Bounds quantization         | int8 into shared memory, per-molecule linear scale                    |
| 5    | Distance-geometry minimize  | FIRE optimizer, 4D, atom-centric                                      |
| 6    | Stereo check                | Chiral volume sign validation                                         |
| 7    | 4th-dim crush               | FIRE with w&#8324; penalty                                            |
| 8    | ETK torsion refine          | FIRE, 3D, 6-term Fourier + distance constraints                       |
| 9    | Final validation            | Repeat stereo check                                                   |
| 10   | MMFF94 polish               | FIRE, 3D, full force field: van der Waals + electrostatic + bonded    |
| 11   | Shipped-geometry validation | Chiral volumes and declared E/Z re-read from the polished coordinates |

Stage 11 exists because stage 10 rewrites every coordinate after stage 9 has already stamped its verdict, and carries no stereochemistry term of its own — so a centre or bond the polish inverts would otherwise leave no trace.
It reads single precision before the half-precision narrowing, and demotes to `stereo_failure`, which the CPU fallback already knows how to recover.

## Architecture

Wave dispatch sorts molecules by atom count and launches them in nine waves capped at 16, 32, 48, 64, 80, 96, 112, 128, and 192 atoms.
The ladder is even to 128 and then jumps, because molecules past 128 atoms are rare enough that a tighter step would buy empty bins.
A 32-thread block occupies at most 32 of a _"streaming multiprocessor"_'s 64 warp slots, so that is the ceiling this shape can reach at any register count; `__launch_bounds__(32, 4)` sets a floor and nvcc fits well past it on the small bins.
The persistent warp loop keeps the same kernel resident across all stages, avoiding launch overhead between FIRE iterations and between phases.

The CPU pipeline runs in two layers.
The outer layer is CLI-level: a reader role streams Parquet shards into coprocessed batches, GPU-runner roles consume them, a writer role emits per-shard output, and a fallback role handles whatever the GPU could not converge.
The inner layer is library-level inside `generate_conformers`: a preprocess step on the host, an embedding step on the GPU, and a finalize step that drains the stream into the caller's buffer.
Wave N+1's preprocess overlaps wave N's embedding via a depth-2 batch-pool ring.

## Performance

Conformers per second, at three conformers per molecule, normalised per compute unit.

| System         | Throughput Per Core | Core Type   | Per Device | Per DGX Node |
| :------------- | ------------------: | :---------- | ---------: | -----------: |
| __SmallField__ |      76 molecules/s | GPU SM core |       ~10⁴ |     ~10⁴–10⁵ |
| nvMolKit       |     3.9 molecules/s | GPU SM core |   ~10²–10³ |         ~10³ |
| RDKit          |     4.8 molecules/s | CPU core    |       ~10² |         ~10³ |

A single DGX-H100 node is eight GPUs or roughly two hundred CPU cores.
An H100 has 132 GPU _"streaming multiprocessor"_ SM cores.
On one H100 the GPU stage runs at 10,070 conf/s and the full batch pipeline delivers 9,291 conf/s end to end, which is 10 million Enamine REAL molecules in under an hour.

Python and C++ share the GPU path.
`sf.generate_conformers` and `smallfield::api::context_t` enter the same CUDA pipeline, so neither is the faster interface.
The `smallfield_batch` binary adds the Parquet reader, writer and CPU fallback roles around it for corpus-scale runs.

Sustained throughput depends on how often the GPU refuses a molecule, since every refusal falls back to CPU RDKit.
Clean, lead-like corpora stay near the ceiling; heterogeneous archives with large molecules and unsupported elements pay more.
Every number, with its corpus, its hardware and its methodology, including that corpus spread, lives in [`BENCH.md`](BENCH.md).

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
batch = context.generate_conformers(
    ["CCO", "c1ccccc1"], num_conformers=3, max_atoms_per_molecule=192,
    max_stereo_attempts=10, random_seed=42,
)
batch.coordinates  # float16, flat, padded to max_atoms_per_molecule
batch.energies     # float32, kcal/mol, one per conformer slot
batch.status       # uint8 ConformerStatus codes
batch.num_atoms    # one per molecule, with hydrogens, 0 if preprocess rejected it
```

Reuse one `Context` across calls: constructing it initialises every GPU and reserves every pinned pool, so building one per call pays that twice.

Coordinates cross as `float16`, which is the precision the pipeline produces — widening them at the boundary would cost a second buffer three times the size for accuracy the slots never held.
Call `.astype(numpy.float32)` if you want it, and the cost is then yours to see.

`max_atoms_per_molecule` is the output stride as well as the cap, so lowering it to the largest molecule you actually feed in saves proportional host memory.
Read the ceiling from `sf.MAX_ATOMS_PER_MOLECULE` rather than restating it; it is the stride two layers have to agree on.

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

`smallfield_descriptors` augments shards that already hold 3D conformers, computing USR and USRCAT shape descriptors from the stored fp16 coordinates instead of re-running the GPU.
It rounds through fp16 exactly as the producer does, so a re-generated shard and an augmented one agree byte for byte.

```bash
smallfield_descriptors --input chembl_3d_mmff/ --output chembl_3d_usrcat/
```

## Installation

There are no wheels: SmallField compiles CUDA and links RDKit, so every install builds from source and needs `nvcc`, CMake, and the RDKit development headers already present.

```bash
pip install --no-build-isolation "git+https://github.com/unum-science/SmallField.git"
```

`--no-build-isolation` matters — under build isolation `pip` cannot see the RDKit and CUDA installation the build shells out to.
From a clone, `pip install --no-build-isolation -e .` does the same thing.

`pixi.toml` pins the environment this is developed and tested in — Python 3.14, RDKit 2026.03, Arrow 24 — and `pixi install` reproduces it.

C++ tests and benchmarks need RDKit, which `pixi install` already provides — the build finds the in-tree environment on its own, and `-DRDKIT_PREFIX=` is only needed to point at some other one.

```bash
cmake --preset dev                     # one architecture, detected from the local GPU
cmake --build build --parallel
ctest --preset dev
```

`--preset release` builds the full `80/89/90/100/120` fatbin into `build-release/`, which is what ships and what CI compiles.
It is several times slower, so keep it for release checks rather than the inner loop.

Every knob is `SMALLFIELD_`-prefixed and listed at the top of `CMakeLists.txt`; `cmake -LH build | grep SMALLFIELD` prints them with their descriptions.
`SMALLFIELD_BUILD_TESTS` and `SMALLFIELD_BUILD_BENCHMARKS` default to on only when this is the top-level project, so vendoring the tree gets the library alone.

## Using SmallField From Another Project

The same target spelling works whether the tree is vendored or installed:

```cmake
find_package(smallfield REQUIRED)   # after `cmake --install build --prefix ...`
add_subdirectory(smallfield)        # or vendored in place

target_link_libraries(app PRIVATE smallfield::smallfield)
```

The installed package records the RDKit environment it was built against, so a consumer on the same machine needs no `-DRDKIT_PREFIX=`.
Because the library is header-only apart from one translation unit, the headers install alongside the archive and both are needed.

## C++ Benchmarks

```bash
cmake --build build --parallel --target bench_conformer_backends

# ETKDG + MMFF94 throughput (RDKit CPU vs SmallField GPU)
./build/bench/bench_conformer_backends --smi /path/to/chembl.smi
```

`--multi-gpu-only` on `bench_conformer_backends` reports isolated 1/2/4/8-GPU scaling.

## Project Layout

```
src/smallfield/api/         public C++ API (smallfield.hpp)
src/smallfield/types/       plain data structures shared host + device
src/smallfield/parameters/  MMFF94 parameter tables
src/smallfield/preprocess/  host preprocessing: SMILES parse, bounds, Cahn-Ingold-Prelog, torsion tables
src/smallfield/mmff94/      native MMFF94 typer, charge, term extractor (CPU + GPU)
src/smallfield/cuda/        CUDA kernels: preprocess, metric embedding, DG, ETK, MMFF, FIRE
src/smallfield/postprocess/ USR / USRCAT shape descriptors, fragment separation
cli/                        smallfield_batch (generate) + smallfield_descriptors (augment)
test/                       C++ and Python tests cross-validated against RDKit
bench/                      bench_conformer_backends + bench_mmff_kernels + bench_profile
python/smallfield/          Python package: api.py (generate), tools/ (shard preparation)
```

There are no dependencies beyond CUDA 12+ and RDKit for preprocessing.
Apache Arrow + Parquet are optional and only needed for the batch CLI binary.

## Acknowledgements

The methods are not ours.
[ETKDG](https://doi.org/10.1021/acs.jcim.5b00654) is Riniker and Landrum's, and [MMFF94](https://doi.org/10.1002/%28SICI%291096-987X%28199604%2917%3A5/6%3C490%3A%3AAID-JCC1%3E3.0.CO%3B2-P) is Halgren's.
SmallField changes neither; it moves them onto the GPU.

[RDKit](https://github.com/rdkit/rdkit) did the bulk of this work, over two decades, before any of it was fast.
Every correctness claim here is a claim about agreeing with them, and a reimplementation is only ever as good as what it reimplements.

[Nebius](https://nebius.com) provided the compute this was built and measured on.

[NVIDIA](https://www.nvidia.com) deserves acknowledgement well beyond the GPUs: their sustained work on accelerating science is what makes a project like this reasonable to attempt at all.
[nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) is their own GPU conformer generator, and the closest comparable work — it is the baseline in the table above.

## License

Apache 2.0
