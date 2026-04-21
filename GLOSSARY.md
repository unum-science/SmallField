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
| __`atom-limited-bin`__  | Inside one `embedding-wave`, molecules are grouped into bins whose member atom counts fall under a fixed cap (16, 32, 48, 64, 80, 96, 112, 128). Each bin dispatches one kernel launch with shared memory sized for the cap. | 6–8 K molecules/bin               |
| __`molecule`__          | One input SMILES → K output conformers.                                                                                                                                                                                      | —                                 |

Within one `embedding-wave` the per-device stripe sorts molecules by `molecule_complexity_score_` (a pure function of `etkdg_molecule_metadata_t`; see `src/smallfield/preprocess/molecule_complexity.hpp`) descending and assigns sorted rank `k` to device `k % device_count`.
The score combines atom-count² (bounds-matrix + nonbonded-pair scan), MMFF+ETK torsion count (FIRE iteration cost), and the `is_heavy_refinement_case_` multiplier, so the stripe balances estimated GPU wall time rather than atom count alone.

Old names that have been retired: `gpus-batch` → `coprocessed-batch`, `chunk` → `embedding-wave`, `wave bin` → `atom-limited-bin`.
The CLI flag `--gpus-batch-size` stays as an alias for `--batch-size`.

## Pipeline Roles, CLI Outer

Four persistent worker threads, each an OpenMP section, with depth-3 ring-buffer handoffs between them.
A _role_ is a persistent thread actor; never substitute "section" or "phase".

| Role                  | Responsibility                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| __`reader-role`__     | Reads a `parquet-shard` off disk, extracts SMILES views, accumulates into a `coprocessed-batch`. |
| __`gpu-runner-role`__ | Calls `context_t::generate_conformers` on each `coprocessed-batch` (inner pipeline runs here).   |
| __`fallback-role`__   | Runs RDKit CPU for the conformer slots the GPU could not fill (≤5 % on drug-like inputs).        |
| __`writer-role`__     | Writes one output `parquet-shard` per consumed input `parquet-shard`.                            |

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
- `batch_pool_t` (`src/smallfield/api/batch_pool.hpp`) — pinned host buffer packed with all per-molecule preprocess output for one `embedding-wave`.
- `molecule_complexity_score_` (`src/smallfield/preprocess/molecule_complexity.hpp`) — pure per-molecule work-estimate function used to balance the per-device stripe inside one `embedding-wave`.
- `distribute_scratch_t` (`src/smallfield/preprocess/etkdg_prepare_batch.hpp`) — grow-only CSR scratch (sort permutation + per-device flat buffer + per-device offsets) reused across every `embedding-wave` so the stripe partition never allocates on the hot path.
- `metric_scratch_t` (`src/smallfield/api/smallfield.hpp`) — per-GPU device scratch for the metric-embedding prepass within an `embedding-step`.
- `launch_scratch_t` (`src/smallfield/api/smallfield.cu`) — per-GPU persistent device pool + pinned host staging; one per GPU, reused across `embedding-waves`.

## Code Map

Glossary terms are vocabulary; the table below points at where each one is grounded in source so a new reader can grep from doc to code.

| Glossary term       | Canonical location in source                                                                  |
| :------------------ | :-------------------------------------------------------------------------------------------- |
| `parquet-shard`     | `cli/smallfield_batch.cpp` — path-based iteration, no dedicated struct                        |
| `coprocessed-batch` | `batch_pool_t` in `src/smallfield/api/batch_pool.hpp` (alias `etkdg_molecule_batch_t` in `src/smallfield/preprocess/etkdg_prepare_batch.hpp`) |
| `embedding-wave`    | `wave_record_t` in `src/smallfield/api/smallfield.cu`                                         |
| `atom-limited-bin`  | `atom_limited_bin_caps[]` in `src/smallfield/api/smallfield.cu`                               |
| pipeline roles      | `cli/smallfield_batch.cpp` — architectural comment blocks, no per-role struct                 |
| `preprocess-step`   | `prepare_molecule_batch_into` in `src/smallfield/preprocess/etkdg_prepare_batch.hpp`          |
| `embedding-step`    | `launch_gpu_pipeline` in `src/smallfield/api/smallfield.cu`                                   |
| `finalize-step`     | `finalize_gpu_pipeline` in `src/smallfield/api/smallfield.cu`                                 |
