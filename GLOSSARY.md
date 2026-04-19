# SmallField Glossary

Canonical vocabulary for data granularity and pipeline structure.
Every doc, code comment, CLI log line, and struct/function name in this repo uses exactly these terms — no aliases, no drift.

## Data granularity — outer to inner

| Term                    | What it is                                                                                                                                                                                                                   | Typical size                      |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| __`parquet-shard`__     | One on-disk Parquet file. Inputs and outputs both use this unit.                                                                                                                                                             | ~1 M molecules                    |
| __`coprocessed-batch`__ | One call to `context_t::generate_conformers`. Aggregates molecules from one or more `parquet-shards` into a batch large enough to saturate all visible GPUs.                                                                 | `num_gpus × 100 K` molecules      |
| __`embedding-wave`__    | Subdivision of a `coprocessed-batch`. The unit of CPU/GPU pipelined work: wave N+1's CPU step overlaps wave N's GPU step.                                                                                                    | ~50 K molecules (VRAM-aware auto) |
| __`atom-limited-bin`__  | Inside one `embedding-wave`, molecules are grouped into bins whose member atom counts fall under a fixed cap (16, 32, 48, 64, 80, 96, 112, 128). Each bin dispatches one kernel launch with shared memory sized for the cap. | 6–8 K molecules/bin               |
| __`molecule`__          | One input SMILES → K output conformers.                                                                                                                                                                                      | —                                 |

Old names that have been retired: `gpus-batch` → `coprocessed-batch`, `chunk` → `embedding-wave`, `wave bin` → `atom-limited-bin`.
CLI flag `--gpus-batch-size` stays as an alias for `--batch-size`.

## Pipeline structure — two distinct layers

### Outer layer — `pipeline-role` (CLI-level)

Four persistent worker threads, each an OpenMP section, with depth-3 ring-buffer handoffs between them:

| Role                  | Responsibility                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| __`reader-role`__     | Reads a `parquet-shard` off disk, extracts SMILES views, accumulates into a `coprocessed-batch`. |
| __`gpu-runner-role`__ | Calls `context_t::generate_conformers` on each `coprocessed-batch` (inner pipeline runs here).   |
| __`fallback-role`__   | Runs RDKit CPU for the conformer slots the GPU could not fill (≤5 % on drug-like inputs).        |
| __`writer-role`__     | Writes one output `parquet-shard` per consumed input `parquet-shard`.                            |

### Inner layer — `wave-step` (library-level)

Three steps per `embedding-wave`, all inside one `generate_conformers` call:

| Step                  | Runs on                       | What happens                                                                                                             |
| --------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| __`preprocess-step`__ | CPU (OpenMP across molecules) | SMILES parse + RDKit / native typing + MMFF atom / charge + pack into pinned host memory.                                |
| __`embedding-step`__  | GPU (async on stream)         | H2D upload → metric-embedding prepass → ETKDG kernels → MMFF polish → fp32→fp16 convert → D2H. Host returns immediately. |
| __`finalize-step`__   | CPU + stream sync             | Block on stream drain → permute pinned outputs into caller's buffer.                                                     |

__Vocabulary rule:__ _role_ = persistent thread actor (CLI layer).
_step_ = discrete transformation on one `embedding-wave` (library layer).
Never substitute "section" or "phase" for either — those words drift back toward each other in prose and lose the distinction.

## Overlap semantics — ASCII time evolution

### Inner pipeline: three `embedding-waves` in steady state

Best-overlap case, CPU preprocess time ≈ GPU embed time:

```
          0ms              50ms             100ms            150ms            200ms
          ├────────────────┼────────────────┼────────────────┼────────────────┤

Wave 0:   │ preprocess-CPU │ embedding-GPU  │ finalize-CPU   │
Wave 1:                    │ preprocess-CPU │ embedding-GPU  │ finalize-CPU   │
Wave 2:                                     │ preprocess-CPU │ embedding-GPU  │ finalize-CPU

CPU busy: ████████████████████████████████████████████████████░░░░░░░░░░░░░░░  (back-to-back wave preps)
GPU busy: ░░░░░░░░░░░░░░░░████████████████████████████████████████████░░░░░░░  (bootstrap idle + drain idle)
```

One row per `embedding-wave`; segments are `wave-steps` on the labeled resource.
At steady state, wave N+1's `preprocess-step` runs on CPU while wave N's `embedding-step` runs on GPU — both resources saturated.
Two idle intervals bracket the run: __bootstrap__ (GPU waits on wave 0's prep) and __drain__ (CPU idles during the final wave's embed + finalize).

Per-wave wall ≈ `max(CPU preprocess, GPU embed)` rather than their sum — that's why the pipelining is the throughput win.

### Outer pipeline: four `pipeline-roles` streaming `parquet-shards`

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

Each row is a `pipeline-role` on its own thread.
Vertical arrows are ring-buffer handoffs.
At steady state all four rows fire simultaneously on different `coprocessed-batches`; steady-state throughput is one batch per tick of the slowest role.

### Run-wide occupancy — single line per resource

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
- `metric_scratch_t` (`src/smallfield/api/smallfield.hpp`) — per-GPU device scratch for the metric-embedding prepass within an `embedding-step`.
- `launch_scratch_t` (`src/smallfield/api/smallfield.cu`) — per-GPU persistent device pool + pinned host staging; one per GPU, reused across `embedding-waves`.
