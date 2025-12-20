# SmallField

> Hardware-accelerated Small Molecule conformer generation on CPUs and GPUs.

SmallField is a high-performance library for generating 3D conformers of small molecules using mixed precision numerics.
It implements [Merck Molecular Force Field](https://en.wikipedia.org/wiki/Merck_molecular_force_field) optimization (MMFF94) via the [Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS) (L-BFGS), comparing both throughput and accuracy against [RDKit](https://github.com/rdkit/rdkit)'s baseline.
It works around the fact that most modern CPUs and GPUs lack `f64` FMA units or Tensor Cores, focusing on `f16` and `bf16` representations with `f32` accumulation instead.
All of its core data-structures are padded proportionally to SIMD and SIMT widths to maximize hardware utilization and avoid code duplication for loop tails.
It calculates all 7 energy terms:

1. VdW (van der Waals) computed for $O(n²)$ all atom pairs
2. Electrostatic computed for $O(n²)$ all atom pairs
3. Bond stretching computed for $O(n)$ bonds
4. Angle bending computed for $O(n)$ angles
5. Torsion computed for $O(n)$ torsions
6. Stretch-bend coupling computed for $O(n)$ stretch-bend terms
7. Out-of-plane bending computed for $O(n)$ out-of-plane terms

## Backends

CPUs:

- __Serial__: Templated C++ kernels for `f32` and `f64` arithmetic
- __Skylake__: AVX-512 kernels for `f32` arithmetic leveraging `rsqrt14`, `rcp14`, and FMA units
- __Turin__: AVX-512 kernels for `f16` coordinates leveraging `f32` for accumulation

GPUs:

- __CUDA F32__: NVIDIA H100 kernels using warp reductions
- __CUDA F16__: NVIDIA H100 kernels for `half2` coordinates and Tensor Cores

## Installation

```bash
git submodule update --init --recursive     # fetch ForkUnion
uv pip install -e .[gpu]                    # or `.[cpu]` for non-CUDA devices
uv run pytest test.py -s -x                 # build and test
```

Or using a conventional Python environment:

```bash
git submodule update --init --recursive
pip install -r requirements-gpu.txt         # or requirements-cpu.txt
pip install -e .
pytest test.py -s -x
```

## Usage

```python
import smallfield as sf
import numpy as np

# Check available backends
print(f"Skylake AVX-512: {sf.has_skylake()}")
print(f"CUDA: {sf.has_cuda()}")

# Create context with pre-allocated buffers
ctx = sf.Context(
    total_atoms=10000,      # Max atoms (rounded up for SIMD)
    total_pairs=1000000,    # Max VdW pairs
    max_molecules=1000      # Max molecules in batch
)

# Compute VdW energy for single molecule
energy = ctx.vdw_skylake(
    coords_x, coords_y, coords_z,  # f64 coordinate arrays
    pair_i, pair_j,                 # i32 atom pair indices
    epsilon, r_star                 # f64 VdW parameters
)

# Compute VdW energy for batch of molecules
batch_energy = ctx.vdw_batch_skylake(
    coords_x, coords_y, coords_z,
    pair_i, pair_j, epsilon, r_star,
    mol_start, mol_end,             # i32 pair index ranges per molecule
    n_molecules
)
```

## Benchmarking

SmallField includes comprehensive benchmarking tools to compare performance across all implementations.
To run the benchmark on a 1000 randomly sampled molecules for 100 iterations:

```bash
uv run --with rdkit python bench.py --n-molecules 1000 --iterations 100 --smiles-dir /path/to/smiles/
```
