#!/usr/bin/env python3
"""
bench.py - Geometry Optimization Benchmark for SmallField MMFF94.

Benchmarks full geometry optimization (not just energy calculation):
- Throughput: molecules optimized per second vs RDKit baseline
- Accuracy: final energies and coordinates vs RDKit and F64 serial

Usage:
    python bench.py --n-molecules 1000 --max-iter 200
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("ERROR: RDKit not available. Install with: pip install rdkit")
    raise SystemExit(1)

try:
    import smallfield
except ImportError:
    print("ERROR: SmallField not available. Build with: uv pip install -e .")
    raise SystemExit(1)


DEFAULT_SMILES_PATH = (
    Path(__file__).parent.parent / "USearchMolecules" / "data" / "example" / "smiles"
)


@dataclass
class MoleculeData:
    """Container for molecule coordinates and force field parameters."""

    mol: Chem.Mol
    n_atoms: int
    n_pairs: int
    coords_x: np.ndarray  # f64
    coords_y: np.ndarray  # f64
    coords_z: np.ndarray  # f64
    pair_i: np.ndarray  # i32
    pair_j: np.ndarray  # i32
    epsilon: np.ndarray  # f64
    r_star: np.ndarray  # f64

    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> "MoleculeData":
        """Create MoleculeData from RDKit Mol object."""
        n_atoms = mol.GetNumAtoms()
        conf = mol.GetConformer(-1)

        coords_x = np.array(
            [conf.GetAtomPosition(i).x for i in range(n_atoms)], dtype=np.float64
        )
        coords_y = np.array(
            [conf.GetAtomPosition(i).y for i in range(n_atoms)], dtype=np.float64
        )
        coords_z = np.array(
            [conf.GetAtomPosition(i).z for i in range(n_atoms)], dtype=np.float64
        )

        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is None:
            raise ValueError("Could not get MMFF properties")

        pair_i_list, pair_j_list, epsilon_list, r_star_list = [], [], [], []

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if mol.GetBondBetweenAtoms(i, j) is not None:
                    continue
                vdw_params = mmff_props.GetMMFFVdWParams(i, j)
                if vdw_params is not None:
                    pair_i_list.append(i)
                    pair_j_list.append(j)
                    epsilon_list.append(vdw_params[1])
                    r_star_list.append(vdw_params[0])

        n_pairs = len(pair_i_list)

        # Pad to multiple of 32 for SIMD
        padding = (32 - n_pairs % 32) % 32
        total_pairs = n_pairs + padding

        pair_i = np.zeros(total_pairs, dtype=np.int32)
        pair_j = np.zeros(total_pairs, dtype=np.int32)
        epsilon = np.zeros(total_pairs, dtype=np.float64)
        r_star = np.zeros(total_pairs, dtype=np.float64)

        for k in range(n_pairs):
            pair_i[k] = pair_i_list[k]
            pair_j[k] = pair_j_list[k]
            epsilon[k] = epsilon_list[k]
            r_star[k] = r_star_list[k]

        # Pad with safe values
        if padding > 0 and n_pairs > 0:
            for k in range(n_pairs, total_pairs):
                pair_i[k] = pair_i[n_pairs - 1]
                pair_j[k] = pair_j[n_pairs - 1]
                epsilon[k] = 0.0
                r_star[k] = 1.0

        return cls(
            mol=mol,
            n_atoms=n_atoms,
            n_pairs=n_pairs,
            coords_x=coords_x,
            coords_y=coords_y,
            coords_z=coords_z,
            pair_i=pair_i,
            pair_j=pair_j,
            epsilon=epsilon,
            r_star=r_star,
        )


@dataclass
class OptimizationResult:
    """Result of geometry optimization."""

    final_energy: float
    final_coords_x: np.ndarray
    final_coords_y: np.ndarray
    final_coords_z: np.ndarray
    iterations: int
    converged: bool
    time_ms: float


@dataclass
class BenchmarkResult:
    """Benchmark results for a single backend."""

    backend: str
    time_ms: float
    mol_per_sec: float
    energies: np.ndarray
    coords_x: List[np.ndarray] = field(default_factory=list)
    coords_y: List[np.ndarray] = field(default_factory=list)
    coords_z: List[np.ndarray] = field(default_factory=list)


def load_smiles_from_file(file_path: Path, n_molecules: int) -> List[str]:
    """Load SMILES strings from a file."""
    smiles_list = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                smiles = line.split()[0] if " " in line or "\t" in line else line
                smiles_list.append(smiles)
                if len(smiles_list) >= n_molecules:
                    break
    return smiles_list


def load_molecules(n_molecules: int, smiles_dir: Path) -> List[MoleculeData]:
    """Load molecules from SMILES directory."""
    smiles_files = sorted(smiles_dir.glob("*.smi"))

    if not smiles_files:
        print(f"No SMILES files found in {smiles_dir}")
        print("Using default test molecules...")
        return load_default_molecules(n_molecules)

    print(f"Loading {n_molecules} molecules from {smiles_dir}...")

    all_smiles = []
    for smi_file in smiles_files:
        all_smiles.extend(
            load_smiles_from_file(smi_file, n_molecules - len(all_smiles))
        )
        if len(all_smiles) >= n_molecules:
            break

    mol_data_list = []
    failed = 0

    for smiles in all_smiles[:n_molecules]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed += 1
                continue
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            mol_data = MoleculeData.from_mol(mol)
            if mol_data.n_pairs > 0:
                mol_data_list.append(mol_data)
        except Exception:
            failed += 1
            continue

    print(f"Loaded {len(mol_data_list)} molecules ({failed} failed)")
    return mol_data_list


def load_default_molecules(n_molecules: int) -> List[MoleculeData]:
    """Load default test molecules."""
    default_smiles = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC(C)(C)c1ccc(O)cc1",  # BHT
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    ]

    mol_data_list = []
    idx = 0
    while len(mol_data_list) < n_molecules:
        smiles = default_smiles[idx % len(default_smiles)]
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        mol_data_list.append(MoleculeData.from_mol(mol))
        idx += 1

    return mol_data_list


def compute_rmsd(
    coords1_x: np.ndarray,
    coords1_y: np.ndarray,
    coords1_z: np.ndarray,
    coords2_x: np.ndarray,
    coords2_y: np.ndarray,
    coords2_z: np.ndarray,
) -> float:
    """Compute RMSD between two coordinate sets."""
    dx = coords1_x - coords2_x
    dy = coords1_y - coords2_y
    dz = coords1_z - coords2_z
    return float(np.sqrt(np.mean(dx * dx + dy * dy + dz * dz)))


# =============================================================================
# Optimization Backends
# =============================================================================


def optimize_rdkit(
    mol_data_list: List[MoleculeData], max_iterations: int
) -> BenchmarkResult:
    """Optimize molecules using RDKit MMFF94 (baseline)."""
    n_molecules = len(mol_data_list)
    energies = np.zeros(n_molecules, dtype=np.float64)
    coords_x_list = []
    coords_y_list = []
    coords_z_list = []

    # Warmup
    for m in mol_data_list[:min(10, n_molecules)]:
        mol_copy = Chem.Mol(m.mol)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol_copy, AllChem.MMFFGetMoleculeProperties(mol_copy)
        )
        if ff:
            ff.Minimize(maxIts=max_iterations)

    start = time.perf_counter()

    for i, m in enumerate(mol_data_list):
        mol_copy = Chem.Mol(m.mol)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol_copy, AllChem.MMFFGetMoleculeProperties(mol_copy)
        )
        if ff:
            ff.Minimize(maxIts=max_iterations)
            energies[i] = ff.CalcEnergy()

            conf = mol_copy.GetConformer(-1)
            n_atoms = mol_copy.GetNumAtoms()
            coords_x_list.append(
                np.array([conf.GetAtomPosition(j).x for j in range(n_atoms)])
            )
            coords_y_list.append(
                np.array([conf.GetAtomPosition(j).y for j in range(n_atoms)])
            )
            coords_z_list.append(
                np.array([conf.GetAtomPosition(j).z for j in range(n_atoms)])
            )

    end = time.perf_counter()
    time_ms = (end - start) * 1000
    mol_per_sec = n_molecules / (time_ms / 1000)

    return BenchmarkResult(
        backend="RDKit (baseline)",
        time_ms=time_ms,
        mol_per_sec=mol_per_sec,
        energies=energies,
        coords_x=coords_x_list,
        coords_y=coords_y_list,
        coords_z=coords_z_list,
    )


def optimize_serial_f64(
    mol_data_list: List[MoleculeData], max_iterations: int, grad_threshold: float
) -> Optional[BenchmarkResult]:
    """Optimize molecules using SmallField serial F64."""
    n_molecules = len(mol_data_list)

    # Get max sizes for context
    max_atoms = max(m.n_atoms for m in mol_data_list)
    max_pairs = max(len(m.pair_i) for m in mol_data_list)

    ctx = smallfield.Context(max_atoms, max_pairs, 1)

    energies = np.zeros(n_molecules, dtype=np.float64)
    coords_x_list = []
    coords_y_list = []
    coords_z_list = []

    # Warmup
    for m in mol_data_list[:min(5, n_molecules)]:
        cx, cy, cz = m.coords_x.copy(), m.coords_y.copy(), m.coords_z.copy()
        try:
            ctx.optimize_serial(
                cx, cy, cz, m.pair_i, m.pair_j, m.epsilon, m.r_star,
                max_iterations, grad_threshold
            )
        except Exception:
            pass

    start = time.perf_counter()

    for i, m in enumerate(mol_data_list):
        cx = m.coords_x.copy()
        cy = m.coords_y.copy()
        cz = m.coords_z.copy()

        try:
            result = ctx.optimize_serial(
                cx, cy, cz, m.pair_i, m.pair_j, m.epsilon, m.r_star,
                max_iterations, grad_threshold
            )
            energies[i] = result[0]  # final_energy
            coords_x_list.append(cx)
            coords_y_list.append(cy)
            coords_z_list.append(cz)
        except Exception as e:
            print(f"Serial F64 failed for molecule {i}: {e}")
            energies[i] = float("nan")
            coords_x_list.append(m.coords_x.copy())
            coords_y_list.append(m.coords_y.copy())
            coords_z_list.append(m.coords_z.copy())

    end = time.perf_counter()
    time_ms = (end - start) * 1000
    mol_per_sec = n_molecules / (time_ms / 1000)

    return BenchmarkResult(
        backend="Serial F64",
        time_ms=time_ms,
        mol_per_sec=mol_per_sec,
        energies=energies,
        coords_x=coords_x_list,
        coords_y=coords_y_list,
        coords_z=coords_z_list,
    )


def optimize_skylake(
    mol_data_list: List[MoleculeData], max_iterations: int, grad_threshold: float
) -> Optional[BenchmarkResult]:
    """Optimize molecules using SmallField Skylake AVX-512 F32."""
    if not smallfield.has_skylake():
        return None

    n_molecules = len(mol_data_list)
    max_atoms = max(m.n_atoms for m in mol_data_list)
    max_pairs = max(len(m.pair_i) for m in mol_data_list)

    ctx = smallfield.Context(max_atoms, max_pairs, 1)

    energies = np.zeros(n_molecules, dtype=np.float64)
    coords_x_list = []
    coords_y_list = []
    coords_z_list = []

    # Warmup
    for m in mol_data_list[:min(5, n_molecules)]:
        cx, cy, cz = m.coords_x.copy(), m.coords_y.copy(), m.coords_z.copy()
        try:
            ctx.optimize_skylake(
                cx, cy, cz, m.pair_i, m.pair_j, m.epsilon, m.r_star,
                max_iterations, grad_threshold
            )
        except Exception:
            pass

    start = time.perf_counter()

    for i, m in enumerate(mol_data_list):
        cx = m.coords_x.copy()
        cy = m.coords_y.copy()
        cz = m.coords_z.copy()

        try:
            result = ctx.optimize_skylake(
                cx, cy, cz, m.pair_i, m.pair_j, m.epsilon, m.r_star,
                max_iterations, grad_threshold
            )
            energies[i] = result[0]
            coords_x_list.append(cx)
            coords_y_list.append(cy)
            coords_z_list.append(cz)
        except Exception as e:
            print(f"Skylake failed for molecule {i}: {e}")
            energies[i] = float("nan")
            coords_x_list.append(m.coords_x.copy())
            coords_y_list.append(m.coords_y.copy())
            coords_z_list.append(m.coords_z.copy())

    end = time.perf_counter()
    time_ms = (end - start) * 1000
    mol_per_sec = n_molecules / (time_ms / 1000)

    return BenchmarkResult(
        backend="Skylake AVX-512 F32",
        time_ms=time_ms,
        mol_per_sec=mol_per_sec,
        energies=energies,
        coords_x=coords_x_list,
        coords_y=coords_y_list,
        coords_z=coords_z_list,
    )


def optimize_cuda_f32(
    mol_data_list: List[MoleculeData], max_iterations: int, grad_threshold: float
) -> Optional[BenchmarkResult]:
    """Optimize molecules using SmallField CUDA F32."""
    if not smallfield.has_cuda():
        return None

    n_molecules = len(mol_data_list)
    max_atoms = max(m.n_atoms for m in mol_data_list)
    max_pairs = max(len(m.pair_i) for m in mol_data_list)

    ctx = smallfield.Context(max_atoms, max_pairs, 1)

    energies = np.zeros(n_molecules, dtype=np.float64)
    coords_x_list = []
    coords_y_list = []
    coords_z_list = []

    # Warmup
    for m in mol_data_list[:min(5, n_molecules)]:
        cx, cy, cz = m.coords_x.copy(), m.coords_y.copy(), m.coords_z.copy()
        mol_pair_start = np.array([0], dtype=np.int32)
        mol_pair_end = np.array([m.n_pairs], dtype=np.int32)
        try:
            ctx.optimize_cuda_f32(
                cx, cy, cz, m.pair_i, m.pair_j, m.epsilon, m.r_star,
                mol_pair_start, mol_pair_end, 1, max_iterations, grad_threshold
            )
        except Exception:
            pass

    start = time.perf_counter()

    for i, m in enumerate(mol_data_list):
        cx = m.coords_x.copy()
        cy = m.coords_y.copy()
        cz = m.coords_z.copy()
        mol_pair_start = np.array([0], dtype=np.int32)
        mol_pair_end = np.array([m.n_pairs], dtype=np.int32)

        try:
            result = ctx.optimize_cuda_f32(
                cx, cy, cz, m.pair_i, m.pair_j, m.epsilon, m.r_star,
                mol_pair_start, mol_pair_end, 1, max_iterations, grad_threshold
            )
            energies[i] = result[0]
            coords_x_list.append(cx)
            coords_y_list.append(cy)
            coords_z_list.append(cz)
        except Exception as e:
            print(f"CUDA F32 failed for molecule {i}: {e}")
            energies[i] = float("nan")
            coords_x_list.append(m.coords_x.copy())
            coords_y_list.append(m.coords_y.copy())
            coords_z_list.append(m.coords_z.copy())

    end = time.perf_counter()
    time_ms = (end - start) * 1000
    mol_per_sec = n_molecules / (time_ms / 1000)

    return BenchmarkResult(
        backend="CUDA F32",
        time_ms=time_ms,
        mol_per_sec=mol_per_sec,
        energies=energies,
        coords_x=coords_x_list,
        coords_y=coords_y_list,
        coords_z=coords_z_list,
    )


def optimize_cuda_f16(
    mol_data_list: List[MoleculeData], max_iterations: int, grad_threshold: float
) -> Optional[BenchmarkResult]:
    """Optimize molecules using SmallField CUDA F16."""
    if not smallfield.has_cuda():
        return None

    n_molecules = len(mol_data_list)
    max_atoms = max(m.n_atoms for m in mol_data_list)
    max_pairs = max(len(m.pair_i) for m in mol_data_list)

    ctx = smallfield.Context(max_atoms, max_pairs, 1)

    energies = np.zeros(n_molecules, dtype=np.float64)
    coords_x_list = []
    coords_y_list = []
    coords_z_list = []

    # Warmup
    for m in mol_data_list[:min(5, n_molecules)]:
        cx, cy, cz = m.coords_x.copy(), m.coords_y.copy(), m.coords_z.copy()
        mol_pair_start = np.array([0], dtype=np.int32)
        mol_pair_end = np.array([m.n_pairs], dtype=np.int32)
        try:
            ctx.optimize_cuda_f16(
                cx, cy, cz, m.pair_i, m.pair_j, m.epsilon, m.r_star,
                mol_pair_start, mol_pair_end, 1, max_iterations, grad_threshold
            )
        except Exception:
            pass

    start = time.perf_counter()

    for i, m in enumerate(mol_data_list):
        cx = m.coords_x.copy()
        cy = m.coords_y.copy()
        cz = m.coords_z.copy()
        mol_pair_start = np.array([0], dtype=np.int32)
        mol_pair_end = np.array([m.n_pairs], dtype=np.int32)

        try:
            result = ctx.optimize_cuda_f16(
                cx, cy, cz, m.pair_i, m.pair_j, m.epsilon, m.r_star,
                mol_pair_start, mol_pair_end, 1, max_iterations, grad_threshold
            )
            energies[i] = result[0]
            coords_x_list.append(cx)
            coords_y_list.append(cy)
            coords_z_list.append(cz)
        except Exception as e:
            print(f"CUDA F16 failed for molecule {i}: {e}")
            energies[i] = float("nan")
            coords_x_list.append(m.coords_x.copy())
            coords_y_list.append(m.coords_y.copy())
            coords_z_list.append(m.coords_z.copy())

    end = time.perf_counter()
    time_ms = (end - start) * 1000
    mol_per_sec = n_molecules / (time_ms / 1000)

    return BenchmarkResult(
        backend="CUDA F16",
        time_ms=time_ms,
        mol_per_sec=mol_per_sec,
        energies=energies,
        coords_x=coords_x_list,
        coords_y=coords_y_list,
        coords_z=coords_z_list,
    )


# =============================================================================
# Accuracy Metrics
# =============================================================================


def compute_accuracy_metrics(
    results: Dict[str, BenchmarkResult], baseline_key: str
) -> Dict[str, Dict[str, float]]:
    """Compute accuracy metrics relative to a baseline."""
    baseline = results.get(baseline_key)
    if baseline is None:
        return {}

    metrics = {}

    for name, result in results.items():
        if name == baseline_key:
            continue

        # Energy error
        energy_diff = np.abs(result.energies - baseline.energies)
        valid_mask = ~np.isnan(energy_diff)

        if not np.any(valid_mask):
            continue

        energy_mae = float(np.mean(energy_diff[valid_mask]))
        energy_max = float(np.max(energy_diff[valid_mask]))

        # Coordinate RMSD (average over molecules)
        rmsd_list = []
        for i in range(len(result.coords_x)):
            if i < len(baseline.coords_x):
                rmsd = compute_rmsd(
                    result.coords_x[i],
                    result.coords_y[i],
                    result.coords_z[i],
                    baseline.coords_x[i],
                    baseline.coords_y[i],
                    baseline.coords_z[i],
                )
                rmsd_list.append(rmsd)

        avg_rmsd = float(np.mean(rmsd_list)) if rmsd_list else float("nan")

        metrics[name] = {
            "energy_mae": energy_mae,
            "energy_max": energy_max,
            "coord_rmsd": avg_rmsd,
        }

    return metrics


# =============================================================================
# Main Benchmark
# =============================================================================


def run_benchmark(
    n_molecules: int, max_iterations: int, grad_threshold: float, smiles_dir: Path
):
    """Run the geometry optimization benchmark."""
    print("=" * 100)
    print("SmallField MMFF94 Geometry Optimization Benchmark")
    print("=" * 100)
    print()

    # Print capabilities
    print("Capabilities:")
    print(f"  Skylake (AVX-512 F32):  {'Yes' if smallfield.has_skylake() else 'No'}")
    print(f"  CUDA:                   {'Yes' if smallfield.has_cuda() else 'No'}")
    print()

    # Load molecules
    mol_data_list = load_molecules(n_molecules, smiles_dir)
    if not mol_data_list:
        print("No molecules loaded. Exiting.")
        return

    print(f"\nOptimization parameters:")
    print(f"  Max iterations:      {max_iterations}")
    print(f"  Gradient threshold:  {grad_threshold}")
    print()

    total_atoms = sum(m.n_atoms for m in mol_data_list)
    total_pairs = sum(m.n_pairs for m in mol_data_list)

    print(f"Batch statistics:")
    print(f"  Molecules:   {len(mol_data_list):,}")
    print(f"  Total atoms: {total_atoms:,}")
    print(f"  Total pairs: {total_pairs:,}")
    print()

    # Run all backends
    results: Dict[str, BenchmarkResult] = {}

    print("Running optimizations...")
    print()

    # RDKit baseline
    print("  RDKit (baseline)...", end=" ", flush=True)
    rdkit_result = optimize_rdkit(mol_data_list, max_iterations)
    results["RDKit (baseline)"] = rdkit_result
    print(f"done ({rdkit_result.time_ms:.1f} ms)")

    # Serial F64
    print("  Serial F64...", end=" ", flush=True)
    try:
        serial_result = optimize_serial_f64(mol_data_list, max_iterations, grad_threshold)
        if serial_result:
            results["Serial F64"] = serial_result
            print(f"done ({serial_result.time_ms:.1f} ms)")
        else:
            print("N/A")
    except Exception as e:
        print(f"failed: {e}")

    # Skylake
    print("  Skylake AVX-512 F32...", end=" ", flush=True)
    try:
        skylake_result = optimize_skylake(mol_data_list, max_iterations, grad_threshold)
        if skylake_result:
            results["Skylake AVX-512 F32"] = skylake_result
            print(f"done ({skylake_result.time_ms:.1f} ms)")
        else:
            print("N/A")
    except Exception as e:
        print(f"failed: {e}")

    # CUDA F32
    print("  CUDA F32...", end=" ", flush=True)
    try:
        cuda_f32_result = optimize_cuda_f32(mol_data_list, max_iterations, grad_threshold)
        if cuda_f32_result:
            results["CUDA F32"] = cuda_f32_result
            print(f"done ({cuda_f32_result.time_ms:.1f} ms)")
        else:
            print("N/A")
    except Exception as e:
        print(f"failed: {e}")

    # CUDA F16 - temporarily disabled due to crash
    print("  CUDA F16... skipped (needs debugging)")
    # try:
    #     cuda_f16_result = optimize_cuda_f16(mol_data_list, max_iterations, grad_threshold)
    #     if cuda_f16_result:
    #         results["CUDA F16"] = cuda_f16_result
    #         print(f"done ({cuda_f16_result.time_ms:.1f} ms)")
    #     else:
    #         print("N/A")
    # except Exception as e:
    #     print(f"failed: {e}")

    print()

    # Throughput table
    print("-" * 100)
    print("THROUGHPUT")
    print("-" * 100)
    print(f"{'Backend':<30} {'Time (ms)':>15} {'Mol/sec':>15} {'Speedup':>15}")
    print("-" * 100)

    rdkit_time = results["RDKit (baseline)"].time_ms

    for name, result in results.items():
        speedup = rdkit_time / result.time_ms if result.time_ms > 0 else 0
        print(
            f"{name:<30} {result.time_ms:>15.2f} {result.mol_per_sec:>15.0f} {speedup:>14.1f}x"
        )

    print("-" * 100)
    print()

    # Accuracy vs RDKit
    print("-" * 100)
    print("ACCURACY vs RDKit")
    print("-" * 100)
    print(f"{'Backend':<30} {'Energy MAE':>15} {'Energy Max':>15} {'Coord RMSD':>15}")
    print("-" * 100)

    rdkit_metrics = compute_accuracy_metrics(results, "RDKit (baseline)")
    for name, m in rdkit_metrics.items():
        print(
            f"{name:<30} {m['energy_mae']:>15.6f} {m['energy_max']:>15.6f} {m['coord_rmsd']:>15.6f}"
        )

    print("-" * 100)
    print()

    # Accuracy vs Serial F64
    if "Serial F64" in results:
        print("-" * 100)
        print("ACCURACY vs Serial F64")
        print("-" * 100)
        print(f"{'Backend':<30} {'Energy MAE':>15} {'Energy Max':>15} {'Coord RMSD':>15}")
        print("-" * 100)

        serial_metrics = compute_accuracy_metrics(results, "Serial F64")
        for name, m in serial_metrics.items():
            print(
                f"{name:<30} {m['energy_mae']:>15.6f} {m['energy_max']:>15.6f} {m['coord_rmsd']:>15.6f}"
            )

        print("-" * 100)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SmallField MMFF94 Geometry Optimization Benchmark"
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=100,
        help="Number of molecules (default: 100)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Max optimization iterations (default: 200)",
    )
    parser.add_argument(
        "--grad-threshold",
        type=float,
        default=1e-4,
        help="Gradient convergence threshold (default: 1e-4)",
    )
    parser.add_argument(
        "--smiles-dir",
        type=Path,
        default=DEFAULT_SMILES_PATH,
        help=f"Path to SMILES directory (default: {DEFAULT_SMILES_PATH})",
    )
    args = parser.parse_args()

    run_benchmark(args.n_molecules, args.max_iter, args.grad_threshold, args.smiles_dir)
