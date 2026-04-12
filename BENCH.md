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

All numbers below use jemalloc preloaded and all current optimizations enabled.
No RDKit binary patches.
Invoke with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ./bench/bench_chembl --smi … --conformers 1`.

### Build flags

The CMake default is now `Release` with these flags baked in (set automatically unless the user overrides `CMAKE_BUILD_TYPE` or the per-language flag variables):

- CXX: `-O3 -DNDEBUG -march=native -fopenmp -std=gnu++20`
- CUDA: `-O3 -DNDEBUG --use_fast_math -Xcompiler=-march=native -Xptxas=-O3 -Xcompiler=-fopenmp -std=c++20`

`-march=native` uses the host CPU's full ISA including AVX-512 on the Xeon 6776P.
`--use_fast_math` maps `expf`/`logf`/`sqrtf`/etc. to their GPU hardware intrinsics and enables FTZ.
`-Xptxas=-O3` sets the PTX assembler's optimization level explicitly.
`-Xcompiler=-fopenmp` forwards `-fopenmp` to the host compiler through nvcc (required for the `#pragma omp parallel for` inside `etkdg_preprocess.hpp` to actually parallelize — see the `-fopenmp` fix section below).

Moving from `-O2 -g -DNDEBUG` (the previous `RelWithDebInfo` default) to these flags gave ~2-3% on end-to-end throughput.
The small gain is because RDKit's `SubstructMatch`, `MMFFMolProperties`, and `setTopolBounds` live in the conda-forge `.so` files we link against; those were compiled with whatever flags conda-forge used and our flags do not touch them.
The flags still matter for SmallField's own hot paths (descriptor build, native SMARTS matcher, MMFF VdW combining rules, the CUDA kernels).

| Batch   | RDKit CPU (192 thr) | SmallField 1 GPU | SmallField 8 GPUs | 1→8 scaling |
| ------- | ------------------- | ---------------- | ----------------- | ----------- |
| 8,948   | 1,672 mol/s         | 1,954 mol/s      | 2,429 mol/s       | 1.24×       |
| 50,000  | 1,730 mol/s         | 2,449 mol/s      | 4,766 mol/s       | 1.95×       |
| 100,000 | 1,725 mol/s         | 3,123 mol/s      | __5,884 mol/s__   | __1.88×__   |
| 241,596 | —                   | —                | __4,475 mol/s__   | —           |

At 241,596 molecules on all 8 GPUs: 53.9 seconds end-to-end (median of 3 runs).
Quality is unchanged: P50 ≈ 56 kcal/mol vs RDKit 61 kcal/mol, 96 % of conformers successful.

Two observations about the 1→8 scaling:

1. __1→8 GPU scaling is 1.88×, not 8×.__
   The ceiling is not the GPU kernel itself — kernel-only throughput at 241K is 15,895 mol/s on 8 GPUs.
   The remaining wall is (a) per-GPU fixed overhead in the metric embedding prepass (~800 ms of cusolver CPU-side setup per chunk), and (b) CPU preprocessing of chunk 0 that cannot yet be overlapped with anything.

2. __Multi-GPU throughput is now stable at 5,884 mol/s from 100K upward.__
   The earlier regression at 241K (4,356 mol/s) came from the serial Phase 2 pack, which is now parallelized.

## Optimizations that landed

### Metric prepass 8-bin padded syevjBatched

`build_metric_initial_coordinates_on_gpu` used to group molecules by exact atom count and call `cusolverDnSsyevjBatched` once per distinct N — drug-like distributions span 30+ values, and per-call cusolver overhead dominated the per-launch wall at ~500 ms per GPU.

Now molecules are binned into the same 8 buckets the wave dispatcher uses (`{16, 32, 48, 64, 80, 96, 112, 128}`).
Each bin issues ONE `cusolverDnSsyevjBatched` at the bin's padded N.
Shorter molecules zero-pad the Gram matrix up to the bin size — the padding creates zero eigenvalues that sort below the real ones, so the top-4 eigenvectors still reconstruct the same 4D embedding.
The kernel smooths only the real N × N sub-block (no wasted Floyd-Warshall work).

Result: per-launch wall dropped from ~700 ms to ~430 ms.
Dispatcher critical path at 8 GPUs went from 5.6 s/chunk to ~3.4 s/chunk.

### Native bounds matrix builder (no RDKit `setTopolBounds`)

The old `build_bounds_matrix` called RDKit's `setTopolBounds`, which allocates a fresh `DistGeom::BoundsMatrix` plus two `boost::shared_array<double>` and two `boost::shared_array<int>` per molecule (~140 KB of throwaway heap) and walks RDKit's atom typer.
At 241K molecules that's ~33 GB of transient allocator traffic per batch.

The replacement reads only the flat `molecule_descriptor_t` and the small tables in `src/forcefield_tables.hpp`:

- Covalent radii per atomic number + bond-order shortening factor produce the 1-2 bond length (±2 % tolerance).
- Ideal-angle-by-hybridization + law of cosines produce the 1-3 distance (±3 % tolerance).
- BFS distance = 3 with a conservative cis/trans envelope produces the 1-4 bound.
- VdW radii produce the long-range lower bound for `bond_distance > 3`.

Writes directly into the caller's `std::vector<float>` — zero other heap activity per call.
Quality gate holds (P50 energy within ~5 kcal/mol of the RDKit baseline; the subsequent triangle-smoothing + FIRE refinement absorbs the residual).

### Persistent pinned host output buffers

`host_coords` (1.1 GB at 241K × 3 conformers × 128 atoms), `host_energies`, and `host_status` used to be pageable `std::vector`s allocated per `generate_conformers` call.
They now live in a shared `output_staging_t` owned by `context_t`: one `pinned_byte_buffer_t` that grows monotonically across calls.
Eliminates a ~1 GB `malloc` + zero-fill per call.

### SMILES kept as `std::string_view` end-to-end

`prepare_molecule` used to take `std::string const &`, so `generate_conformers` copied its `std::span<std::string_view const>` input into a fresh `std::vector<std::string>` (~24 MB at 241K) and `prep_slice` copied another range per chunk.
The API now takes `std::span<std::string_view const>` all the way down; one unavoidable `std::string` materialization happens at the single `RDKit::SmilesToMol` call inside `prepare_molecule` — scoped to microseconds of stack/SSO storage.

### Unified `batch_pool_t` with pinned host + persistent device mirrors

`etkdg_molecule_batch_t` used to hold 14 separate `std::vector<T>` fields.
The launch path carved 14 device slices out of one pool, then issued 14 `cudaMemcpyAsync` calls whose sources were pageable `std::vector` storage.
CUDA silently degrades async H2D from pageable memory to synchronous, so each of the 14 calls blocked the dispatcher for its full transfer time.

The new `batch_pool_t` allocates ONE pinned host buffer per chunk (via `cudaHostAlloc`) described by a `batch_layout_t` — 16 byte-offset / count pairs for each sub-region.
Phase 2 of `prepare_molecule_batch` memcpys per-molecule data directly into this pinned buffer; the launch path issues a single `cudaMemcpyAsync` that copies the whole buffer to a per-device device mirror.
The mirror is owned by a persistent `launch_scratch_t` in `context_t` so it is allocated once and reused across chunks.

Both buffers are exposed as `std::span<T>` views via a new `batch_pool_view_t` that parameterizes by a raw `std::byte *` base pointer — the same class produces host and device views from the same layout.

End-to-end: `launch_gpu_pipeline` per-GPU time dropped from ~1.5 s (synchronous upload loop) to a few milliseconds of dispatcher work.
We evaluated `cudaMallocManaged` + `cudaMemPrefetchAsync` for a truly unified pointer; on consumer-tier PCIe Blackwell the HMM bookkeeping cost about 2× more than pinned+explicit, so we stayed with explicit copies.

### `std::span` sweep over scratch + handle state

`metric_scratch_t` and `gpu_pipeline_handle_t` used to expose raw `T*` + length pairs.
All of them are now `std::span<T>` — length is carried with the pointer and consumers can `.size()` / `.data()` interchangeably.
`launch_gpu_pipeline` and `run_gpu_pipeline` take non-owning input/output arrays as `std::span` parameters.
Kernel signatures still take `(T* __restrict__, int N)` pairs — nvcc's `mdspan` support under `__restrict__` is incomplete, so we keep the aliasing guarantee at the kernel boundary and pass `.data()` / `.size()` there.

### The `-fopenmp` through-nvcc bug (the one-line fix that gave 4×)

Prior benchmarks reported ~587 mol/s on 8 GPUs, which looked terrible.
Root cause: `smallfield.cu` is compiled by nvcc, and nvcc silently ignores `#pragma omp parallel for` unless you forward `-fopenmp` to the host compiler:

```cmake
target_compile_options(smallfield_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
```

Before this flag, the OpenMP loop inside `prepare_molecule_batch()` ran single-threaded when called via the library (`omp_get_max_threads()` returned 1).
Preprocessing 8,948 molecules took __13.3 seconds__ instead of 625 ms — a 21× stall that dwarfed everything else.
The CMake one-liner restored the correct parallelism and was worth __16× end-to-end__ on the 241K batch.

### Parallel Phase 2 pack (struct-of-sizes + parallel memcpy)

Phase 2 of `prepare_molecule_batch` used to be a serial loop of 14 `std::vector::insert` concatenations per molecule.
At 241K molecules this cost ~15 s on its own and dominated the CPU side.

The rewrite operates on a single `std::vector<per_mol_sizes_t>` — one 64-byte struct per molecule holding the 14 per-array counts plus a `valid` flag.
Three passes: parallel size extraction, single prefix scan over the struct (cache-local, <5 ms at 241K), parallel memcpy into pre-sized batch arrays.
The prefix scan of the `valid` field doubles as the dense-output compaction index.
At 241K, Phase 2 drops from ~15 s to ~1 s.

### CPU/GPU pipelining via async CUDA streams

`generate_conformers` now splits large inputs into ~50K-molecule chunks.
While GPU runs chunk N, the calling thread preprocesses chunk N+1.
No `std::thread`, no queue — CUDA streams already provide the asynchrony; the main thread drives the loop.

Inside each chunk, 8 GPUs still launch in parallel via short-lived `std::thread`s (the `metric_embedding_prepass` stage does ~800 ms of CPU-side cusolver setup per GPU that cannot start until the previous GPU's kernels finish — parallelizing the launches across GPUs hides that cost).
Between chunks, there is no inter-chunk thread.

To make pipelining actually pipeline, two sub-fixes were needed:

1. __Remove the blocking cusolver-info check in `metric_embedding_prepass.cuh`.__
   It used `cudaMemcpyAsync` + `cudaStreamSynchronize` to read per-atom-group cusolver exit codes and abort on failure.
   That sync blocked the calling thread for the full cusolver duration, making "launch" synchronous and defeating the pipeline.
   Cusolver failures on Gram matrices from well-formed bounds are vanishingly rare, and when they happen the downstream kernels surface as `stereo_failure` / `mmff_unconverged` anyway.

2. __Move the device→host download from `launch_gpu_pipeline` into `finalize_gpu_pipeline`.__
   `cudaMemcpyAsync` with a non-pinned host destination __degrades to synchronous__ — it does not actually enqueue on the stream, it blocks the CPU for the full download duration (~2 s per chunk here).
   Doing it in finalize, where the stream has already drained via `cudaStreamSynchronize`, lets us use plain `cudaMemcpy` without the async/non-pinned pitfall.

With both fixes, `launch_gpu_pipeline` returns in ~1.6 s (down from 3.8 s), and the pipelined path shaves ~10 % off 241K end-to-end.

### Native CSD torsion extraction with SMARTS predicates

The CSD torsion SMARTS library (470 patterns) is matched via a stack-only native matcher.
46% of patterns (216 of 470) parse into a compact constraint struct and run with zero heap allocations — no VF2, no `std::vector` per match, no RDKit functor objects.
The remaining 54% fall back to RDKit's `SubstructMatch`.

9 hand-written predicate functions (`csd_torsion_predicates.hpp`) cover the most common recursive sub-queries: `$(C=O)`, `$(S(=O))`, `$([CX3]=O)`, `$([C](=O))`, `$([NX3H1])`, `$([NX3H2])`, `$(C([NX3]))`, `$([CX3]([C]))`, `$([cH0](Cl))`.
Each is 5-10 stack-only lines reading the descriptor.
All 9 cross-validated against RDKit's `SubstructMatch` on 210 molecules with zero divergences.

Without this work the torsion step called `SubstructMatch` for every pattern × every molecule, and `SubstructMatch` allocates several `std::vector` objects plus a VF2 state per call.
At 192 threads that heap churn caps preprocessing at ~13K mol/s no matter how many threads you throw at it.
With the native matcher + jemalloc, peak CPU throughput reaches ~17K mol/s.

### Flat molecule descriptor

One pass through RDKit's slow accessor API fills a 22 KB stack-allocated SoA struct (atoms, bonds, CSR adjacency, BFS distance matrix, SSSR rings, atom/bond flags).
All downstream stages (chiral centers, distance constraints, nonbonded exclusion) read this flat data with no RDKit accessors in the hot path.
Build cost: 0.026 ms/mol.

### MMFF VdW pointer caching

RDKit's `MMFFMolProperties::getMMFFVdWParams(i, j)` does two singleton-map lookups per call.
We hoist those lookups to an N-element per-atom `MMFFVdW const *` array, then call Halgren's combining rules directly.
Eliminates ~N²/2 binary searches per molecule in the O(N²) nonbonded pair construction loop.

### BFS shortest-path injection

The descriptor's BFS distance matrix is cast to `double[N²]` and injected into RDKit's molecule property cache under the key `DistanceMatrix`.
`setTopolBounds()` picks it up instead of running Floyd-Warshall O(N³).
Saves ~7 % of preprocessing time, no correctness change.

### MMFF pre-sanitization

Set the `_MMFFSanitized` property on the molecule before constructing `MMFFMolProperties` so the constructor skips its redundant Kekulize step.
1.6× on the MMFF constructor, small absolute win.

### GPU-side optimizations (pre-existing)

- `rsqrtf` replaces 17 `sqrtf` + `__frcp_rn` pairs.
- Interleaved triple warp reduction in the FIRE optimizer.
- FMA in the FIRE velocity-Verlet step.
- ETK per-atom interaction lists (was O(N·T) brute scan).
- MMFF94 per-type sub-loops (was 7-case switch with warp divergence).
- `if constexpr` template functors (compile-time gradient/energy-only branching).
- `__restrict__` on all functor parameter pointers.
- Pool allocator (was 19 `cudaMalloc` per chunk, now 1).
- `__half2` vectorized bounds loading.
- `__half` coordinate output.
- 8-GPU round-robin dispatch with atom-count-sorted waves.

## Where the time still goes

At 241K molecules × 8 GPUs, after everything above.
Numbers from the pipelined path with 5 chunks of ~48K molecules each.

| Phase per chunk                 | Wall time | Notes                                                              |
| ------------------------------- | --------- | ------------------------------------------------------------------ |
| `prepare_molecule_batch`        | 5.5-6.0 s | OpenMP parallel, 192 threads; Phase 2 pack now ~1 s                |
| `launch_gpu_pipeline` (all 8)   | 1.5-1.8 s | __dominated by metric prepass cusolver setup__, 8 GPUs in parallel |
| GPU kernel (background)         | ~5 s      | overlaps with next chunk's CPU prep; download becomes nearly free  |
| `finalize_gpu_pipeline` (all 8) | 20-30 ms  | stream has usually drained by the time we get here                 |

The remaining bottlenecks, ordered by impact:

### 1. Per-GPU fixed overhead (800 ms of cusolver CPU setup per chunk per GPU)

`metric_embedding_prepass.cuh` does group-by-atom-count setup that involves cusolver workspace sizing, per-group cudaMalloc×4, and buffer preparation.
The cusolver API calls themselves are not purely stream-async — some query the device synchronously.
This currently dominates `launch_gpu_pipeline` wall time.

Two fixes worth trying:

- Cache the cusolver workspace in `context_t` across calls instead of re-sizing per chunk.
- Consolidate the per-atom-count groups so we make fewer cusolver calls per chunk.

Expected: launch time drops from 1.6 s to a few hundred ms, end-to-end 241K drops toward ~30 s.

### 2. Bounds matrix construction on GPU

`build_bounds_matrix` is ~13 % of CPU preprocessing today.
With pipelining in place, freeing 13 % of CPU time lets the overlap reach further (or lets us shrink each chunk for better per-GPU balance).
Port `set12`/`set13`/`set14` and VdW lower bounds to four embarrassingly parallel GPU kernels (one thread per bond, angle, 1-4 path, atom pair).

### 3. Avoid materializing N²/2 nonbonded pairs on CPU

Per molecule we currently build `std::vector<mmff94_nonbonded_pair_t>` with all non-excluded pairs (up to 128 KB per molecule), then upload it.
At 50K mols/chunk that's ~6 GB of host-side construction per chunk.
Alternative: upload only atom types + charges (≈ 2 KB/mol) + the bond_distance matrix (16 KB/mol), and enumerate pairs on the fly in the MMFF nonbonded force pass from `bond_distance[i,j] > 3`.
Saves ~50× on upload bandwidth and removes the O(N²) CPU pair construction.
Requires changes to `mmff94_functor.cuh`.

## NUMA notes

Machine topology: 2 NUMA nodes × 48 physical cores × 2 HT = 192 logical CPUs, 20× remote-vs-local memory distance penalty.

Pure CPU preprocessing (8,948 mols, `bench_threaded`):

| Threads | Default affinity | `OMP_PROC_BIND=close OMP_PLACES=cores` |
| ------- | ---------------- | -------------------------------------- |
| 64      | 17,640 mol/s     | 17,126 mol/s                           |
| 96      | 17,014 mol/s     | 17,055 mol/s                           |
| 192     | 15,263 mol/s     | __17,042 mol/s__                       |

Binding OpenMP threads to physical cores stops the 192-thread collapse.
For single-socket isolation, `numactl --cpunodebind=0 --membind=0` hits the highest peak — 18,760 mol/s at 96 threads — but wastes the other socket.

End-to-end (CPU + 8-GPU pipeline) disagrees:

| Configuration                          | End-to-end time | Throughput |
| -------------------------------------- | --------------- | ---------- |
| default                                | 16,204 ms       | 552 mol/s  |
| `OMP_PROC_BIND=close OMP_PLACES=cores` | 20,488 ms       | 437 mol/s  |
| `numactl --interleave=all`             | 16,842 ms       | 531 mol/s  |

Pinning OpenMP threads to cores starves the CUDA worker and upload threads, which run outside the OpenMP pool.
Do not bake NUMA defaults into the library.
Users running CPU-only preprocessing can opt in with `OMP_PROC_BIND=close OMP_PLACES=cores`; everyone else should leave the default.

## Quality

Cross-validation: random conformer sample evaluated by RDKit's `MMFF::calcEnergy()`.

| Metric                   | Value                                   |
| ------------------------ | --------------------------------------- |
| Kernel success rate      | 95.5% (230,735 / 241,596 on 241K batch) |
| Median RDKit energy      | 59 kcal/mol                             |
| Median SmallField energy | 57 kcal/mol                             |

RDKit and SmallField energies agree within ~2 kcal/mol at median.
Half-precision coordinate output introduces ~0.1% quantization error.

### Per-atom-count breakdown (8,948-molecule subset)

| Atoms   | Count | RDKit P50 | SmallField P50 |
| ------- | ----- | --------- | -------------- |
| 0-20    | 189   | 1.2       | 1.2            |
| 20-40   | 3,391 | 36.1      | 36.1           |
| 40-60   | 3,751 | 67.3      | 67.2           |
| 60-80   | 1,530 | 82.6      | 82.4           |
| 80-100  | 619   | 122.9     | 121.9          |
| 100-128 | 260   | 136.5     | 136.4          |
