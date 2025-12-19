# SmallField Benchmark Suite

Unified benchmark for measuring MMFF94 force field accuracy and throughput across storage types.

## Quick Start

```bash
pixi run benchmark          # Full benchmark (1000 molecules)
pixi run benchmark-quick    # Quick test (100 molecules)
```

## What It Measures

__Accuracy__: Per-term RMSE% against Python f64 reference (bond, angle, torsion, vdW, electrostatic)

__Throughput__: Molecules per second vs RDKit MMFF94 baseline

## Storage Types

| Type | Bits | Use Case                       |
| ---- | ---- | ------------------------------ |
| f32  | 32   | Baseline accuracy              |
| bf16 | 16   | Training-friendly, wider range |
| f16  | 16   | Inference-optimized            |

## Architecture

```
USearchMolecules SMILES
        │
        ▼
 molecule_loader.py     Extract MMFF94 params via RDKit
        │
   ┌────┴────┬──────────────┐
   ▼         ▼              ▼
baseline  data_exporter  benchmark_rdkit
 (f64)    (.bin format)   (throughput)
   │         │              │
   │         ▼              │
   │    runner.mojo ────────┤
   │    (f16/bf16/f32)      │
   │         │              │
   └─────────┴──────────────┘
             │
             ▼
      report_generator.py
      (formatted tables)
```

## Files

| File                  | Purpose                                           |
| --------------------- | ------------------------------------------------- |
| `orchestrator.py`     | Main entry point, coordinates all steps           |
| `molecule_loader.py`  | Loads SMILES, extracts MMFF94 parameters          |
| `baseline_compute.py` | Python f64 reference implementation               |
| `data_exporter.py`    | Binary format for Python→Mojo data exchange       |
| `runner.mojo`         | Native Mojo benchmark (compiled, not interpreted) |
| `report_generator.py` | Accuracy/throughput tables                        |

## Binary Format

Custom CSR-style format for efficient data transfer:

```
Header: magic (0x4C4D4653), num_molecules, array counts
Offsets: atom_starts, bond_starts, angle_starts, ...
Data: positions, bond_params, angle_params, ... (all f64)
```

## Output Example

```
================================================================================
SmallField Unified Benchmark
================================================================================
Dataset: 1000 molecules | Baseline: Python f64 | Reference: RDKit MMFF94

ACCURACY (RMSE% vs Python f64)
--------------------------------------------------------------------------------
Storage     Bond      Angle     Torsion   vdW       Elec      Total
--------------------------------------------------------------------------------
f32         0.0001    0.0002    0.0003    0.0010    0.0005    0.0004
bf16        0.12      0.15      0.08      0.25      0.18      0.16
f16         0.05      0.08      0.04      0.15      0.10      0.08

THROUGHPUT
--------------------------------------------------------------------------------
Implementation          mol/sec         Speedup vs RDKit
--------------------------------------------------------------------------------
RDKit MMFF94            45,000          1.00x
SmallField f32          850,000         18.89x
SmallField bf16         920,000         20.44x
SmallField f16          980,000         21.78x

SIMD: AVX-512 (512-bit)
================================================================================
```
