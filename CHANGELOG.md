# SmallField Changelog

Landed work, newest first.
The road still ahead lives in [`ROADMAP.md`](ROADMAP.md); experiments that did not survive review live in the same file's "Tried and rejected" section.

Throughput numbers throughout this file are measured on the same 8x RTX PRO 6000 Blackwell / 2x Xeon 6776P host as [`BENCH.md`](BENCH.md), 10M USearch K=1 unless otherwise noted.
The CPU cumulative gain on that workload from the original 2026-04-19 baseline (20 283 mol/s) to the current head (23 735 mol/s) is __+16.9 %__.

## Complexity-Aware Stripe and Partitioned CPU Budget — Pending

Two changes target throughput on drug-like `coprocessed-batches` where the dominant pathologies are the per-device wave straggler and concurrent-role CPU over-subscription.

- __Complexity-descending stripe__.
  `distribute_batch_to_per_device_pools_` now sorts molecules by `molecule_complexity_score_` (atom-count² + 4·torsion-count + per-term cost, doubled on heavy-rescue cases — see `src/smallfield/preprocess/molecule_complexity.hpp`) descending and stripes sorted rank `k` onto device `k % device_count`.
  Replaces the old `num_atoms`-only sort.
  The sort permutation and per-device assignment are packed into a single CSR layout (one flat `std::vector<std::uint32_t>` + one `std::array<std::uint32_t, max_devices_k + 1>` offset array), both held on a new caller-owned `distribute_scratch_t` sibling of the existing `static etkdg_molecule_batch_t shared_scratch`.
  No vector-of-vectors, no per-call allocation on the distribute hot path; the previous code built a `std::vector<std::uint32_t>` the size of the whole batch plus a 32-wide vector-of-vectors every `embedding-wave`.

- __Partitioned CPU budget__.
  `compute_thread_budget` divides the non-reserved cores 5 : 3 between `fallback-role` and `gpu-runner-role` by default (fallback is the pacer on drug-like input).
  On 192 cores the default goes from `gpu_runner=188, fallback=188` (over-subscribed, OS context-switch thrash) to `gpu_runner=69, fallback=113` (sum fits within `num_procs − reader − writer − os_reserve`).
  New CLI flag `--gpu-runner-threads N` alongside the existing `--rdkit-fallback-threads N`.

Baseline measurements pre-change (PubChem shard 8, 8 GPUs, K=3, seed 42): `351 s wall, 2 845 mol/s, 0 CUDA errors`.
Post-change first single-shard run: `349 s wall, 2 867 mol/s, 0 CUDA errors` — within noise on a single shard.
The benefit is expected at steady state across many shards where `fallback-role` and `gpu-runner-role` previously stalled each other; follow-up campaign measurements will land here.

## 3-Stream Chain Reverted — `db25c9d`

The 3-stream chain from `a791e54` was kept as an architectural improvement assuming the copy-engine overlap would show at production scale.
Measured under PubChem drug-like K=3 it lost on both axes:

- __Throughput.__
  Shard 47 ran 17 % slower than the single-stream shard-46 baseline at the same input shape and fallback rate (387 s vs 331 s, 2 581 mol/s vs 3 019 mol/s).
  On 10M USearch K=3 the single-stream build runs 587.9 s (17 007 mol/s / 51 022 conformers/s) vs 3-stream's 661.9 s — a 13 % single-stream advantage on a workload where 3-stream had been claimed to help.
  The overlap gain is a loss because kernel phases are short enough that the extra `cudaEvent` hand-off overhead dominates the saved copy/compute concurrency.
- __Stability.__
  A second `illegal memory access` race surfaced on the 3-stream build at `cudaEventSynchronize(finalize)` on PubChem shard 50, +1:26 into the wave.
  The per-wave `cudaFuncSetAttribute` fix (`1b786f1`) closed the root cause; the 3-stream chain re-opens a separate race on the event hand-off that was not pinned down.

Reverted `a791e54`'s `smallfield.cu` + `smallfield.hpp` changes (100 lines, 2 files).
Per-device pinned pools, ctor pre-allocation, and the launch-config cache from the same refactor survive unchanged.

## Multi-GPU Illegal-Memory-Access Fix — `1b786f1`

Dropped a per-wave `cudaFuncSetAttribute(embed_4d_power_iter_kernel, MaxDynSM, 67 KB)` from `build_metric_initial_coordinates_on_gpu` that was reached concurrently from every OMP per-device worker thread inside `launch_wave`.
The call mutates driver-global per-kernel attribute state on a shared kernel symbol; at ≥ 2 GPUs the 8 worker threads raced on that state, producing a window during which `embed_4d_power_iter_kernel` launched with the 48 KB hardware-default dynamic shared instead of the 67 KB the kernel needs at bin 128.
The out-of-bounds shared-memory access surfaced on a later kernel launch as `an illegal memory access was encountered`.
The attribute is already set once per device in `context_t::context_t`; the per-wave re-assert was defensive overkill left over from `9cb82be`, where the ctor hoist landed but the per-wave copy was not removed.

Symptom matching:

- Fired only at ≥ 2 GPUs — a single OMP thread cannot race with itself.
- Timing-random 30–120 s into a run — a small torn-state window that depends on thread interleaving, not on molecule shape.
- `CUDA_LAUNCH_BLOCKING=1` masked it — blocking launches serialize the attribute update far enough that no kernel launches within the torn window.
- `compute-sanitizer memcheck` never reproduced — the instrumentation overhead widens each `setAttribute`'s critical section enough to close the race.

Verification — 10 M USearch K=3, 8 GPUs, post-fix:

| Variant                             |    Wall |  mol/s | conformers/s | Crashes |
| :---------------------------------- | ------: | -----: | -----------: | ------: |
| Pre-fix non-blocking                |       — |      — |            — | crashed @ +9:49 s |
| Pre-fix `CUDA_LAUNCH_BLOCKING=1`    | 1 063 s | 9 404 |      28 211 | 0 |
| Post-fix non-blocking               |   662 s | 15 106 |      45 319 | 0 |

## Multi-GPU Pipeline Refactor — `a791e54` and `9cb82be`

Four architectural changes to the per-wave GPU dispatch path; together they land the host-scheduling items from the roadmap without altering any kernel code.

- __Per-device pinned pools.__
  `prepare_per_device_batches_into` produces one pinned batch pool per device (atom-count-descending stripe partition, each device's mols ending in ascending-atom-count order for the kernel bin dispatch).
  Each GPU's `cudaMemcpyAsync H2D` reads from a buffer only that GPU touches.
  Replaces the shared wave-wide pinned source whose 8-way concurrent DMA was the prime suspect behind the illegal-memory-access crash until the root cause was pinned down as the `cudaFuncSetAttribute` race above.
- __3-stream chain per device.__
  H2D / compute / D2H run on three streams chained by `cudaEvent` hand-offs so Blackwell's two copy engines can overlap H2D of wave N+1 with compute of wave N and D2H of wave N-1.
- __Ctor pre-allocation.__
  `context_t::context_t` pre-grows every per-device pool (`launch_scratch.device_pool`, `launch_scratch.host_staging`, `metric_scratch.pool`, `batch_slots_per_device`) to worst case using `cudaMemGetInfo` plus the cusolver workspace cap.
  Hot-path `cudaMalloc` / `cudaHostAlloc` never fire.
- __Launch-config cache.__
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` probes run once per device at ctor (128 probes, ≈ 13 ms) and cache into a `device_launch_config_t` table.
  Per-bin dispatch in `launch_gpu_pipeline` reads the table — no host API call on the hot path.

Drive-by cleanup: dropped the dead `context_t::batch_ring` field, the unreferenced `run_gpu_pipeline` backward-compat wrapper, and legacy PascalCase constants `kNoCharge` / `kMaxIters` / `kNumShards` (now `no_charge_k` / `max_iterations_k` / `scaffold_cache_shard_count_k`).

Scope: 2 files in `src/smallfield/api/`, ≈ 100 lines of pipeline wiring; preprocess adds the new per-device distribution helper; three renames in the preprocess headers.
No kernel code changed.

## CSD Torsion Round 2 — `bc4e1af`

Native pattern coverage 391/470 → __408/470__ (83.2 % → 86.8 %); throughput 23 371 → 23 735 mol/s (+1.55 % over round 1, +16.9 % cumulative).
The diminishing-returns shape is expected: round 1 already removed the biggest VF2 hot-spots, leaving non-linear topology patterns that resist a strictly linear matcher.

Three additive matcher extensions in `csd_torsion_native.hpp` plus four new recursive predicates and a hybridization-based per-molecule prefilter:

- __`R<n>` ring-count constraint__ — an `atom_disjunct_t::ring_count` field matched via `__builtin_popcount(desc.atom_ring_bitmap[atom_idx])`, reusing the descriptor's existing 32-ring bitmap.
  `R2,R3` decomposes through the existing DNF parser into two disjuncts. +5 patterns.
- __Chain-in-branch via predicate lookup__ — bodies like `(cO)`, `(C=O)`, `([cH0]Cl)`, `([cH0]F)` lift to an already-compiled recursive predicate via a new `lookup_chain_branch` table instead of growing a generic 2-hop matcher.
  Adds a `predicate` field on `simple_neighbor_constraint_t` checked after the atom-disjunct pass. +8 patterns.
- __Four new recursive predicate bodies__ — `[cH0](=O)`, `c!@c`, `[n,N](-a)`, `[a]([nH0,o])`.
  Standard one-hop walks of element + aromaticity + bond-type constraints. +4 patterns.

The hybridization prefilter precomputes a per-molecule `molecule_hybridizations` mask alongside the existing `molecule_elements` mask; the per-pattern requirement is the intersection of disjuncts per atom then union across the four labelled atoms (so `[^2,^3]`-style OR-disjuncts correctly yield no constraint).
Patterns whose required mask is not a subset of the molecule's mask are skipped before the matcher runs.

Validation: 40.8 M tier-1 (100 K ChEMBL) and 404.5 M tier-2 (1 M USearch) byte-equivalence checks against `RDKit::SubstructMatch`, zero disagreements; full ctest passes.

## CSD Torsion Round 1 — `1f57e83`

Native pattern coverage 314/470 → __391/470__ (66.8 % → 83.2 %); wall 493 s → 428 s (-13.2 %); throughput 20 283 → 23 371 mol/s (+15.2 %); `boost::VF2SubState::match` self-time 10.38 % → 6.85 %.

The CSD torsion matcher's RDKit `SubstructMatch` fallback was the dominant CPU cost at 10M-scale diversity (27–30 % of CPU at scaffold-cache hit rate 33.4 %).
This commit widened `match_linear_pattern`'s SMARTS coverage so the bulk of patterns parse natively, leaving the RDKit fallback only for genuinely non-linear patterns (ring-closure digits, chain continuation inside a branch, multi-hop recursive predicates).

Parser and matcher changes in `csd_torsion_native.hpp`:

- Replaced the flat `simple_atom_constraint_t` with an `atom_disjunct_t[4]` disjunct list plus shared predicate / neighbour-branch fields, unlocking atom-level OR (`[cH,nX2H0]`, `[NH1,NH2]`, `[A,a;r{5-8}]`).
- Replaced the flat `simple_bond_constraint_t` with a `bond_disjunct_t[4]` list, unlocking bond-level OR (`@;=,:`) and encoding the SMARTS default bond as two disjuncts (single, aromatic).
- Fixed SMARTS logical-operator precedence — `,` (OR) binds tighter than `;` (AND), so `A,a;r{5-8}` correctly means `(A OR a) AND r{5-8}`.
  Both `parse_atom_spec` and `parse_bond_op` compile to disjunctive normal form by cross-producting `,`-alternatives across `;`-separated conjuncts.
- Treats bare `[H]` as the hydrogen ATOM (element 1) when no other element primitive appears in the term; treats it as the H-count modifier otherwise.
- Maps `D<n>` to total explicit degree (matching RDKit on `addHs`-ed input).
- Adds an `element_mask_exclusive` mode so `!#1` accepts elements > 63 (heavy metals in the corpus) while a 64-bit mask still covers the common first-63 case.
- Enforces all four torsion atoms distinct (`a1 != a4`).
- Rejects patterns with trailing unbracketed content after the 4th labeled atom (e.g. `...[C:4]=O`).

14 new recursive predicates in `csd_torsion_predicates.hpp` covering the top SMARTS bodies in the fallback set: sulfonyl `S(=O)=O` / `S(=O)(=O)`, `[CX3](=[!O])`, `C([F])([F])[F]`, `[CX3]([C])([H])`, `C([NX3])N` / `C([NX3])n`, the `[N][CX4]` family, `[cH0]F` / `[cH0][OH1]` / `[cH0][OH0]`, `[cH0]([CX3])`, `cO`, `a[OH1]`, `a[NH1,NH2]`, `c([NH1,NH2])`.

Validation: env-var-gated tiered byte-equivalence test in `test_csd_torsion_predicates.cpp` covering a 100 K ChEMBL sample and a 1 M USearch sample, zero disagreements on either tier.

## Earlier Landed Work

Each item references a CHANGELOG-worthy ship event prior to the round-1 / round-2 work above.

- __Int8 bounds quantization__ — N×N DG bounds matrix as `uint8_t` with per-molecule linear dequantization.
  Halved shared-memory at N=128 (32 KB → 16 KB).
  Single-GPU USearch 1M K=1 throughput: +21 %.
- __DG+ETK / MMFF kernel split__ — separated the monolithic pipeline kernel.
  Register pressure 145 → 95–96 per kernel; MMFF theoretical occupancy at bin 128 rose 8.33 % → 37.5 %.
  Single-GPU USearch 1M K=1: +26 % on top of quantization, +53 % cumulative.
- __CSD torsion native predicates (initial 13)__ — replaced the three heaviest RDKit `SubstructMatch` calls (covering 75–85 % of match CPU across six tested datasets) with hand-written C++ predicates.
- __Native Murcko decomposition__ — BFS peel + Weisfeiler–Lehman canonical hash replacing `MurckoDecompose`.
  Eliminated 15 % of CPU prep time.
- __MMFF parameter table extraction__ — full MMFF94 typing, charge, term extraction ported to native flat-array code.
  Byte-identical on 20 K ChEMBL: atom-type 99.91 %, partial-charge 99.97 %, term failure 22 ppm.
- __GPU MMFF atom typer + charge kernel__ — 1 045 580 atoms byte-identical vs CPU; 439 / 20 K molecules fall back.
- __GPU term extraction + CSR build__ — 941 / 1 009 test molecules match; remainder dispatched to CPU.
- __SMILES parse with `removeHs=false`__ — skipped the `removeHs` + re-`addHs` round-trip.
  Saved ~6 % of CPU prep wall time.
- __Scaffold cache__ — caches torsion-pattern match results by Bemis-Murcko scaffold. +44 % throughput on curated ChEMBL (4 698 unique scaffolds across 237 600 cyclic molecules, 98 % reuse).
  Lower hit rate on diverse corpora.
- __Per-wave per-device `cudaEvent_t` instrumentation__ — enables the per-chunk breakdown in `BENCH.md`.
- __Molecule sort by atom count + output permutation__ — dispatches smaller molecules first, permutes back to input order at finalize.
  Improves `atom-limited-bin` fill.
- __Pre-warm cusolver + metric scratch in `context_t` ctor__ — avoids first-wave allocation stall.
- __Pre-grow 256 MB device / host pools per device__ — arena claimed at construction.
- __Four redundant `__syncwarp` removals__ on lane-owned access paths (2026-04-18).
