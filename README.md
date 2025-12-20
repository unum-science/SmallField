# SmallField

> Hardware-accelerated Small Molecule conformer generation on CPUs and GPUs.

SmallField is a high-performance library for generating 3D conformers of small molecules using mixed precision numerics.
It works around the fact that most modern CPUs and GPUs lack `f64` FMA units or Tensor Cores, focusing on `f16` and `bf16` representations with `f32` accumulation instead.
It implements ETKDG and MMFF force field based optimization, comparing against RDKit's implementations.
All of its core data-structures are padded proportionally to SIMD and SIMT widths to maximize hardware utilization and avoid code duplication for loop tails.

## CPU Backends

- __Serial__: Templated f32/f64 with 4/8-way accumulators (auto-vectorizable)
- __AVX-512 F32__: Skylake-X and newer (rsqrt14, rcp14, FMA)
- __AVX-512 F16__: Intel SPR, AMD Turin (vcvtph2ps/vcvtps2ph)
- __AVX-512 BF16__: AMD Genoa (vdpbf16ps)

## GPU Backends

- __CUDA F16__: NVIDIA H100 (warp reductions, tensor cores)
- __CUDA F32__: NVIDIA H100 (warp reductions)

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
import smallfield

# Compute MMFF94 energy
energy = smallfield.compute_energy(mol, backend=smallfield.Backend.AVX512_F32)

# Optimize conformer
optimized_mol = smallfield.optimize_conformer(mol, max_iters=200)
```
