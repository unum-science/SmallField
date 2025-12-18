# SmallField

> Hardware-accelerated Small Molecule conformer generation on CPUs and GPUs using Mojo.

SmallField is a high-performance library for generating 3D conformers of small molecules using mixed precision numerics implemented in Mojo.
It works around the fact, that most modern CPUs and GPUs lack `f64` FMA units or Tensor Cores, focusing on `f16` and `bf16` representations with `f32` accumulation instead.
It implements ETKDG and MMFF force field based optimization, comparing against RDKit's implementations.

The CPU version is targeting mainly AVX-512 capable processors with optimizatinos for:

- Skylake: supporting F32 dot-products
- Genoa: supporting BF16 dot-products into F32 accumulators
- Turin: supporting F16 dot-products into F32 accumulators
- Sapphire Rapids: supporting AMX-BF16 matrix operations

The GPU version:

- Ada: supporting F16/BF16 matrix multiplcations via Tensor Cores
