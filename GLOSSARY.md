# SmallField Glossary

Canonical vocabulary for data granularity and pipeline structure.
Every doc, code comment, CLI log line, and struct/function name in this repo uses exactly these terms — no aliases, no drift.

## Data Granularity

From outer to inner.
A `parquet-shard` carries roughly a million molecules; a `coprocessed-batch` aggregates enough molecules to saturate every visible GPU; an `embedding-wave` is the unit of CPU/GPU pipelining; an `atom-limited-bin` is what one kernel launch sees.

| Term                    | What it is                                                                                                                                                                                                                   | Typical size                      |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| __`parquet-shard`__     | One on-disk Parquet file. Inputs and outputs both use this unit.                                                                                                                                                             | ~1 M molecules                    |
| __`coprocessed-batch`__ | One call to `context_t::generate_conformers`. Aggregates molecules from one or more `parquet-shards` into a batch large enough to saturate all visible GPUs.                                                                 | `num_gpus × 100 K` molecules      |
| __`embedding-wave`__    | Subdivision of a `coprocessed-batch`. The unit of CPU/GPU pipelined work: wave N+1's CPU step overlaps wave N's GPU step.                                                                                                    | ~50 K molecules (VRAM-aware auto) |
| __`wave-slice`__        | One `embedding-wave`'s share for a single GPU. The unit a `wave_slice_pool_t` actually holds; the per-device stripe below produces one per device per wave.                                                       | wave ÷ `num_gpus`                 |
| __`atom-limited-bin`__  | Inside one `wave-slice`, molecules are grouped into bins whose member atom counts fall under a fixed cap (16, 32, 48, 64, 80, 96, 112, 128). Each bin dispatches one kernel launch with shared memory sized for the cap. | 6–8 K molecules/bin               |
| __`molecule`__          | One input SMILES → K output conformers.                                                                                                                                                                                      | —                                 |

Within one `embedding-wave` the per-device stripe sorts molecules by `molecule_complexity_score_` (a pure function of `etkdg_molecule_metadata_t`; see `src/smallfield/preprocess/molecule_complexity.hpp`) descending and assigns sorted rank `k` to device `k % device_count`.
The score combines atom-count² (bounds-matrix + nonbonded-pair scan), MMFF+ETK torsion count (FIRE iteration cost), and the `is_heavy_refinement_case` multiplier — the same predicate `src/smallfield/cuda/etkdg_pipeline.cuh` dispatches its heavy-rescue branch on — so the stripe balances estimated GPU wall time rather than atom count alone.

Old names that have been retired: `gpus-batch` → `coprocessed-batch`, `chunk` → `embedding-wave`, `wave bin` → `atom-limited-bin`.
The CLI flag `--gpus-batch-size` stays as an alias for `--batch-size`.

## Pipeline Roles, CLI Outer

Four persistent worker threads, each an OpenMP section, with depth-3 ring-buffer handoffs between them.
A _role_ is a persistent thread actor; never substitute "section" or "phase".

| Role                  | Binary                  | Responsibility                                                                                       |
| --------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------- |
| __`reader-role`__     | `smallfield_batch`      | Reads a `parquet-shard` off disk, extracts SMILES into a `coprocessed-batch`.                        |
| __`gpu-runner-role`__ | `smallfield_batch`      | Calls `context_t::generate_conformers` on each `coprocessed-batch` (inner pipeline runs here).       |
| __`fallback-role`__   | `smallfield_batch`      | Runs RDKit CPU for the conformer slots the GPU could not fill (≤5 % on drug-like inputs).            |
| __`writer-role`__     | `smallfield_batch`      | Writes one output `parquet-shard` per consumed input `parquet-shard`.                                |

## Wave Steps, Library Inner

Three steps per `embedding-wave`, all inside one `generate_conformers` call.
A _step_ is a discrete transformation on one `embedding-wave`; never substitute "section" or "phase".

| Step                  | Runs on                       | What happens                                                                                                             |
| --------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| __`preprocess-step`__ | CPU (OpenMP across molecules) | SMILES parse + RDKit / native typing + MMFF atom / charge + pack into pinned host memory.                                |
| __`embedding-step`__  | GPU (async on stream)         | H2D upload → metric-embedding prepass → ETKDG kernels → MMFF polish → fp32→fp16 convert → D2H. Host returns immediately. |
| __`finalize-step`__   | CPU + stream sync             | Block on stream drain → permute pinned outputs into caller's buffer.                                                     |

## Overlap Semantics

Three views of the same pipeline, time evolving left to right.

The inner pipeline at three `embedding-waves` of steady state, best-overlap case where CPU preprocess time roughly matches GPU embed time:

```
          0ms              50ms             100ms            150ms            200ms
          ├────────────────┼────────────────┼────────────────┼────────────────┤

Wave 0:   │ preprocess-CPU │ embedding-GPU  │ finalize-CPU   │
Wave 1:                    │ preprocess-CPU │ embedding-GPU  │ finalize-CPU   │
Wave 2:                                     │ preprocess-CPU │ embedding-GPU  │ finalize-CPU

CPU busy: ████████████████████████████████████████████████████░░░░░░░░░░░░░░░  (back-to-back wave preps)
GPU busy: ░░░░░░░░░░░░░░░░████████████████████████████████████████████░░░░░░░  (bootstrap idle + drain idle)
```

Per-wave wall ≈ `max(CPU preprocess, GPU embed)` rather than their sum — that is the throughput win.
Two idle intervals bracket the run: bootstrap (GPU waits on wave 0's prep) and drain (CPU idles during the final wave's embed + finalize).

The outer pipeline at four `pipeline-roles` streaming `parquet-shards`, with vertical arrows as ring-buffer handoffs:

```
Time →

reader-role      ──► read parquet-shard ──► read parquet-shard ──► read parquet-shard ──► …
                             │                       │                       │
                             ▼ batch queue           ▼                       ▼
gpu-runner-role              ──► generate_conformers ──► generate_conformers ──► …
                             │  (inner wave pipeline   (inner wave pipeline
                             │   runs here)             runs here)
                             ▼ result queue          ▼                       ▼
fallback-role                ──► RDKit fill-in       ──► RDKit fill-in       ──► …
                             │                       │                       │
                             ▼ write queue           ▼                       ▼
writer-role                  ──► write parquet-shard ──► write parquet-shard ──► …
```

At steady state all four rows fire simultaneously on different `coprocessed-batches`; steady-state throughput is one batch per tick of the slowest role.

A run-wide single-line view of the same evolution:

```
CPU: [prep0][prep1][prep2][prep3]…[prep_{N-1}][prep_N]│░░░░░░│
GPU: ░░░░░░│[emb0 ][emb1 ][emb2 ]…[emb_{N-1} ][emb_N ][fin_N]
           ↑                                          ↑
           bootstrap idle                             drain idle
```

Across an entire run the only idle intervals are startup (one `preprocess-step` before first GPU work) and teardown (CPU idle during the final `embedding-step` + `finalize-step`).
Everything in between is full overlap.

## Related Utilities

- `logger_t` (`src/smallfield/log.hpp`) — owned per-`context_t` or per-tool; emits timestamp-prefixed log lines.
- `cuda_error_sink_t` (`src/smallfield/cuda_error.hpp`) — routes CUDA / cuSOLVER errors through a `logger_t` and bumps an owner-supplied counter.
- `wave_slice_pool_t` (`src/smallfield/preprocess/wave_slice_pool.hpp`) — pinned host buffer packed with all per-molecule preprocess output for one `wave-slice`, plus the `wave_slice_shape_t` that carves it.
- `pool_cursor` (`src/smallfield/types/views.hpp`) — a byte span and a running offset, from which `carve<element_>` hands out one aligned subspan at a time. A cursor holding no pool hands back empty spans and totals what the allocation must span, so `carve_wave_slice` in `src/smallfield/preprocess/wave_slice_pool.hpp` is one sequence that sizes a pool and fills a `wave_slice_view` over it; no byte offset is ever stored beside an extent.
- `molecule_complexity_score_` (`src/smallfield/preprocess/molecule_complexity.hpp`) — pure per-molecule work-estimate function used to balance the per-device stripe inside one `embedding-wave`.
- `distribute_scratch_t` (`src/smallfield/preprocess/etkdg_prepare_batch.hpp`) — grow-only CSR scratch (sort permutation + per-device flat buffer + per-device offsets) reused across every `embedding-wave` so the stripe partition never allocates on the hot path.
- `metric_scratch_t` (`src/smallfield/api/smallfield.hpp`) — per-GPU device scratch for the metric-embedding prepass within an `embedding-step`.
- `launch_scratch_t` (`src/smallfield/api/smallfield.cu`) — per-GPU persistent device pool + pinned host staging; one per GPU, reused across `embedding-waves`.

## Conformer Status Codes

`conformer_status_t` (`src/smallfield/types/conformer_status.hpp`, re-exported by `src/smallfield/api/smallfield.hpp`) is written per conformer slot into the caller's `status_out` span.
The Parquet writer persists the raw byte.

| Value | Name                 | Meaning                                                                                                     |
| ----: | :------------------- | :---------------------------------------------------------------------------------------------------------- |
|     0 | `success`            | Conformer produced by the GPU `embedding-step` (or by the `fallback-role` re-embed).                        |
|     1 | `stereo_failure`     | Stereochemistry assignment fell through all attempts.                                                       |
|     2 | `mmff_unconverged`   | MMFF94 polish did not converge inside the iteration budget.                                                 |
|     3 | `invalid_smiles`     | SMILES failed to parse or produced an empty `molecule`.                                                     |
|     4 | `too_many_atoms`     | Atom count exceeds the compile-time kernel bound.                                                           |
|     5 | `preprocess_timeout` | Reserved. Nothing produces it; the value is kept so shipped shard bytes keep their meaning.                  |
|     6 | `unsupported_element` | An atom has no MMFF94 type — boron, transition metals, lanthanides. Neither GPU nor `fallback-role` recovers it. |
|     7 | `too_many_bonds`     | More bonds than the flat descriptor's `max_bonds_k`, which only a molecule under the atom cap carrying an unusual number of ring closures reaches. |
|     8 | `too_many_rings`     | More SSSR rings than `atom_ring_bitmap` has bits, or more ring-membership entries than `ring_atoms` holds, so a genuinely in-ring atom would carry no ring label. |
|     9 | `bounds_infeasible`  | Triangle smoothing drove some atom pair's lower bound above its upper, so no coordinates satisfy the molecule's bounds matrix. |

## Fallback Disposition

`fallback_disposition_t` in `cli/rdkit_fallback.hpp` is the `fallback-role`'s reading of a `conformer_status_t`, and the two states are the whole vocabulary for what a refused slot is worth dispatching.

| Term            | What it means                                                                                                                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| __`retry_k`__   | RDKit embeds by a different route and may fill the slot, so the `fallback-role` spends an embedding on it. Every failure the GPU path can reach through its own numerics is one of these.        |
| __`terminal_k`__ | `admit_molecule_` re-derives the same refusal from the same caps and the same parser, so dispatching buys one `SmilesToMol` and the verdict the slot already carries. `success` is terminal too. |

The mapping is exhaustive over `conformer_status_t` and carries no `default:`, so a status appended to that append-only enum is a compiler warning rather than a policy inherited in silence.

## Views

A matrix crossing into a kernel is one of two views, and which one is a claim about the row stride rather than a preference.
Both are `custd::mdspan` addressed `(row, column)`, so no call site decodes an index expression; the aliases are in `src/smallfield/types/views.hpp`.

| Term                  | What it is                                                                                                                                                                                                    |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| __`packed_matrix`__   | Rows packed at the column extent, the host layout. `build_bounds_matrix` and `fill_bond_distances` both write it, and `hop_matrix_t` is its `std::uint8_t` specialization. 16 bytes per kernel argument.     |
| __`strided_matrix`__  | Rows padded to a stride the caller states. Two layouts need exactly that: the device bounds matrix, whose rows `compute_bounds_row_stride` pads so all 32 lanes land on distinct shared-memory banks, and a conformer slot, whose component planes pad to `conformer_slot_layout_t::max_atoms`. 24 bytes per kernel argument. |

## Code Map

Glossary terms are vocabulary; the table below points at where each one is grounded in source so a new reader can grep from doc to code.

| Glossary term       | Canonical location in source                                                                                                                  |
| :------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| `parquet-shard`     | `cli/smallfield_batch.cpp` — path-based iteration, no dedicated struct                                                                        |
| `coprocessed-batch` | `coprocessed_batch_accumulator_t` in `cli/smallfield_batch.cpp` — one `generate_conformers` call        |
| `wave-slice`        | `wave_slice_pool_t` in `src/smallfield/preprocess/wave_slice_pool.hpp` |
| `embedding-wave`    | `wave_record_t` in `src/smallfield/api/smallfield.cu`                                                                                         |
| `atom-limited-bin`  | `atom_limited_bin_caps_k[]` in `src/smallfield/api/smallfield.hpp`                                                                            |
| pipeline roles      | `cli/smallfield_batch.cpp` — architectural comment blocks, no per-role struct                                                                 |
| `preprocess-step`   | `prepare_molecule_batch_into` in `src/smallfield/preprocess/etkdg_prepare_batch.hpp`                                                          |
| `embedding-step`    | `launch_gpu_pipeline` in `src/smallfield/api/smallfield.cu`                                                                                   |
| `finalize-step`     | `finalize_gpu_pipeline` in `src/smallfield/api/smallfield.cu`                                                                                 |
