# SmallField

> Hardware-accelerated Small Molecule conformer generation on CPUs and GPUs using Mojo.

SmallField is a high-performance library for generating 3D conformers of small molecules using mixed precision numerics implemented in Mojo.
It works around the fact, that most modern CPUs and GPUs lack `f64` FMA units or Tensor Cores, focusing on `f16` and `bf16` representations with `f32` accumulation instead.
It implements ETKDG and MMFF force field based optimization, comparing against RDKit's implementations.
