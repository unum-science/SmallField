# SmallField Benchmark

ETKDG + MMFF94 conformer generation: CPU (RDKit) vs GPU (SmallField).

## Hardware

| Component | Specification                                                        |
| --------- | -------------------------------------------------------------------- |
| GPU       | 8x NVIDIA RTX PRO 6000 Blackwell (188 SMs each, SM 12.0, 96 GB VRAM) |
| CPU       | 2x Intel Xeon 6776P (48 cores / 96 threads each, 192 threads total)  |
| RAM       | 1.7 TB DDR5                                                          |

## Dataset

ChEMBL drug-like molecules, pre-filtered to N ≤ 128 atoms (including explicit H).

| Stat               | Value                                       |
| ------------------ | ------------------------------------------- |
| Total molecules    | 8,948 (small batch) / 241,596 (large batch) |
| Min atoms          | 8                                           |
| P25 / Median / P75 | 36 / 48 / 63                                |
| P90 / Max          | 84 / 128                                    |

## Throughput

| Batch | RDKit CPU (192 threads) | SmallField 1 GPU | SmallField 8 GPUs |
| ----- | ----------------------- | ---------------- | ----------------- |
| 8,948 | 626 mol/s               | 266 mol/s        | 287 mol/s         |
| 241K  | —                       | —                | 269 mol/s         |

GPU kernel-only throughput at 241K molecules: __15,895 mol/s__ (8 GPUs, 15.2s for 241K molecules).
End-to-end throughput is bottlenecked by CPU preprocessing, not GPU compute.

## Where time goes

### At scale: 241K molecules, 8 GPUs (nsys profiled)

| Phase                        | Code location                                                 | Time     | %         |
| ---------------------------- | ------------------------------------------------------------- | -------- | --------- |
| __CPU preprocessing__        | `etkdg_preprocess.hpp` `prepare_molecule_batch()`             | __883s__ | __98.3%__ |
| GPU upload + metric prepass  | `smallfield.cu` async memcpy + `metric_embedding_prepass.cuh` | ~2s      | 0.2%      |
| GPU kernel (8 GPUs parallel) | `etkdg_pipeline.cuh` `etkdg_pipeline_warp_kernel()`           | 15.2s    | 1.7%      |
| Total                        |                                                               | __898s__ |           |

__The GPU finishes 59x faster than the CPU can feed it data.__

### CPU preprocessing breakdown (per molecule, ~3.7 ms avg on 192 threads)

| Stage                                     | Code location                                          | % of preproc |
| ----------------------------------------- | ------------------------------------------------------ | ------------ |
| CSD torsion extraction                    | `etkdg_preprocess.hpp:217` `getExperimentalTorsions()` | __37%__      |
| MMFF94 parameter lookup                   | `etkdg_preprocess.hpp:355` `extract_mmff94_params()`   | __30%__      |
| Bounds matrix construction                | `etkdg_preprocess.hpp:71` `build_bounds_matrix()`      | __20%__      |
| Chiral + distance constraints + ETK lists | Various                                                | 13%          |

The dominant cost is RDKit's internal SMARTS pattern matching in `getExperimentalTorsions()`.
Skipping ETK torsions (relying on MMFF94 torsions only) degrades median energy by 2.7% (58.5 → 60.1 kcal/mol) but saves 37% of preprocessing time.

### GPU kernel breakdown (estimated from iteration counts, N=128 heavy molecules)

| Stage                                         | Code location                                                   | % of kernel |
| --------------------------------------------- | --------------------------------------------------------------- | ----------- |
| Stage 4: DG minimize (4D, 100 FIRE iter)      | `dg_forcefield.cuh` `evaluate_distance_geometry_energy<true>()` | __36%__     |
| Stage 9: MMFF94 polish (3D, 200-480 iter)     | `mmff94_functor.cuh` `evaluate_mmff94_energy<true>()`           | __25%__     |
| Stage 7: ETK torsion refine (3D, 60-240 iter) | `etk_forcefield.cuh` `evaluate_etk_torsion_energy<true>()`      | __21%__     |
| Stage 6: 4th-dim crush (50 FIRE iter)         | Same DG functor, `fourth_dimension_penalty_weight = 10`         | __18%__     |

Stages 4+6 (distance geometry) dominate at 54% — the O(N²) all-pairs distance computation in the tiled `__shfl_sync` loop.
Stages 7+9 are O(interactions_per_atom) thanks to per-atom interaction lists.

## Quality

Cross-validation: 10,000 random conformers evaluated by RDKit's `MMFF::calcEnergy()`.

| Metric                                 | Value                   |
| -------------------------------------- | ----------------------- |
| Kernel success rate                    | 96.2% (25,822 / 26,844) |
| Finite MMFF94 energy (RDKit cross-val) | 97.4%                   |
| Median RDKit energy                    | 58.6 kcal/mol           |
| Median SmallField energy               | 58.5 kcal/mol           |
| P5 / P95 RDKit energy                  | -67.2 / 183.9 kcal/mol  |

RDKit and SmallField energies agree within 0.1 kcal/mol at median.
Half-precision coordinate output introduces ~0.1% quantization error.

### Per-atom-count breakdown

| Atoms   | Count | RDKit P50 | SmallField P50 |
| ------- | ----- | --------- | -------------- |
| 0-20    | 189   | 1.2       | 1.2            |
| 20-40   | 3,391 | 36.1      | 36.1           |
| 40-60   | 3,751 | 67.3      | 67.2           |
| 60-80   | 1,530 | 82.6      | 82.4           |
| 80-100  | 619   | 122.9     | 121.9          |
| 100-128 | 260   | 136.5     | 136.4          |

## Optimizations applied

__GPU kernel:__
- `rsqrtf` replacing 17 `sqrtf` + `__frcp_rn` pairs
- Interleaved triple warp reduction in FIRE
- FMA in FIRE velocity Verlet step
- ETK per-atom interaction lists (was O(N·T) brute scan)
- MMFF94 per-type sub-loops (was 7-case switch with warp divergence)
- `if constexpr` template functors (compile-time gradient/energy-only branching)
- `__restrict__` on all functor parameter pointers

__GPU infrastructure:__
- Pool allocator (19 `cudaMalloc` → 1)
- Async `cudaMemcpyAsync` with dedicated upload stream
- `__half2` vectorized bounds loading
- 8-GPU round-robin dispatch with load-balanced molecule distribution
- Half-precision `__half` coordinate output

__CPU preprocessing:__
- Stack-allocated 2D bitset for exclusion matrices (2 KB vs 32 KB heap)
- Move semantics in batch packing

## Next improvements

### 1. Pipeline preprocessing with GPU execution

The CPU takes 883s to preprocess 241K molecules while the GPU finishes in 15s.
Double-buffering: preprocess batch N+1 on CPU while GPU runs batch N.
With 16 sub-batches of 15K molecules each, preprocessing and GPU can overlap almost completely.
Expected end-to-end: ~60s (CPU-bound) instead of 898s (sequential).

### 2. Optional ETK bypass

Add `config_t::use_csd_torsions = false` to skip `getExperimentalTorsions()` and stage 7.
Saves 37% preprocessing time and 21% kernel time.
Quality cost: median energy 58.5 → 60.1 kcal/mol (+2.7%).

### 3. Move bounds matrix construction to GPU

`build_bounds_matrix()` is 20% of preprocessing.
RDKit's `setTopolBounds` does graph traversal + VdW radii lookup — potentially parallelizable on GPU if atom types and bond graph are uploaded.

### 4. Cache RDKit MMFF94 properties

`extract_mmff94_params()` calls `MMFFGetMoleculeProperties` per molecule (30% of preprocessing).
For repeated molecular scaffolds (common in combinatorial libraries), cache atom types + parameters by canonical SMILES hash.

### 5. Reduce large-molecule quality tail

P95 energy for 100-128 atoms is 437 kcal/mol (target: < 200).
The adaptive refinement policy already gives heavy molecules more iterations (320+160 MMFF).
A second MMFF polish pass or tighter convergence threshold for the heaviest bin could help.
