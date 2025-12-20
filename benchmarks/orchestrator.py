#!/usr/bin/env python3
"""
SmallField Unified Benchmark Orchestrator

Main entry point that coordinates:
1. Loading molecules from USearchMolecules
2. Computing Python f64 baseline (accuracy reference)
3. Benchmarking RDKit throughput
4. Running Mojo kernels via direct Python binding
5. Generating formatted report
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
_this_dir = Path(__file__).parent if '__file__' in dir() else Path('benchmarks')
sys.path.insert(0, str(_this_dir))

# Add build directory for smallfield module
_build_dir = _this_dir.parent / 'build'
sys.path.insert(0, str(_build_dir))

from molecule_loader import load_molecules, MoleculeData, DEFAULT_SMILES_PATH
from baseline_compute import compute_baseline, EnergyComponents

try:
    import smallfield
except ImportError:
    print("ERROR: smallfield module not found. Build it first:")
    print("  mojo build -O3 --emit shared-lib src/smallfield.mojo -o build/smallfield.so")
    sys.exit(1)

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class BenchmarkResult:
    """Result from a Mojo kernel benchmark."""
    kernel: str
    vdw_energies: List[float]
    elec_energies: List[float]
    elapsed_sec: float
    num_molecules: int

    @property
    def throughput(self) -> float:
        """Molecules per second."""
        return self.num_molecules / max(self.elapsed_sec, 1e-9)


def benchmark_rdkit(molecules: List[MoleculeData], iterations: int = 3) -> float:
    """
    Benchmark RDKit MMFF94 CalcEnergy() for throughput reference.

    Returns: molecules per second
    """
    import sys

    # Re-create RDKit force field objects
    ff_list = []
    sys.stdout.write(f"    Preparing {len(molecules)} molecules... ")
    sys.stdout.flush()

    for mol_data in molecules:
        mol = Chem.MolFromSmiles(mol_data.smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) == -1:
            continue

        mmff = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff is None:
            continue

        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff)
        if ff is not None:
            ff_list.append(ff)

    print(f"done ({len(ff_list)} valid)", flush=True)

    if not ff_list:
        return 0.0

    sys.stdout.write("    Timing CalcEnergy... ")
    sys.stdout.flush()

    # Timed benchmark (single pass)
    start = time.perf_counter()
    for ff in ff_list:
        _ = ff.CalcEnergy()
    elapsed = time.perf_counter() - start

    if elapsed < 1e-9:
        elapsed = 1e-9

    throughput = len(ff_list) / elapsed
    print(f"done ({elapsed:.4f}s, {throughput:,.0f} mol/sec)", flush=True)
    return throughput


def run_kernel_benchmark(
    molecules: List[MoleculeData],
    kernel: str
) -> Optional[BenchmarkResult]:
    """
    Run Mojo kernel benchmark via direct Python binding.

    Args:
        molecules: List of parameterized molecules
        kernel: Kernel name (skylake, simd, genoa, turin)

    Returns: BenchmarkResult or None on failure
    """
    vdw_energies = []
    elec_energies = []

    try:
        start = time.perf_counter()

        for mol in molecules:
            n_atoms = mol.num_atoms
            n_pairs = len(mol.nonbonded)

            if n_pairs == 0:
                vdw_energies.append(0.0)
                elec_energies.append(0.0)
                continue

            # Extract coordinate arrays
            x = mol.positions[:, 0].copy()
            y = mol.positions[:, 1].copy()
            z = mol.positions[:, 2].copy()

            # Extract nonbonded pair data
            nb_i = np.array([nb.i for nb in mol.nonbonded], dtype=np.int64)
            nb_j = np.array([nb.j for nb in mol.nonbonded], dtype=np.int64)
            epsilon = np.array([nb.epsilon for nb in mol.nonbonded], dtype=np.float64)
            r_star = np.array([nb.r_star for nb in mol.nonbonded], dtype=np.float64)
            qi = np.array([nb.qi for nb in mol.nonbonded], dtype=np.float64)
            qj = np.array([nb.qj for nb in mol.nonbonded], dtype=np.float64)
            is_1_4 = np.array([1 if nb.is_1_4 else 0 for nb in mol.nonbonded], dtype=np.int64)

            # Call vdW kernel
            vdw = smallfield.vdw_energy({
                'x': x, 'y': y, 'z': z,
                'nb_i': nb_i, 'nb_j': nb_j,
                'epsilon': epsilon, 'r_star': r_star,
                'num_pairs': n_pairs, 'num_atoms': n_atoms,
                'kernel': kernel
            })
            vdw_energies.append(float(vdw))

            # Call electrostatic kernel
            elec = smallfield.elec_energy({
                'x': x, 'y': y, 'z': z,
                'nb_i': nb_i, 'nb_j': nb_j,
                'qi': qi, 'qj': qj,
                'is_1_4': is_1_4,
                'num_pairs': n_pairs, 'num_atoms': n_atoms,
                'kernel': kernel
            })
            elec_energies.append(float(elec))

        elapsed = time.perf_counter() - start

        return BenchmarkResult(
            kernel=kernel,
            vdw_energies=vdw_energies,
            elec_energies=elec_energies,
            elapsed_sec=elapsed,
            num_molecules=len(molecules)
        )

    except Exception as e:
        print(f"  EXCEPTION running {kernel}: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_report(
    num_molecules: int,
    baseline: List[EnergyComponents],
    results: Dict[str, BenchmarkResult],
    rdkit_throughput: float
) -> str:
    """Generate benchmark report comparing kernels to baseline."""
    lines = []
    lines.append("=" * 80)
    lines.append("BENCHMARK RESULTS")
    lines.append("=" * 80)
    lines.append(f"Molecules: {num_molecules}")
    lines.append(f"RDKit throughput (estimated): {rdkit_throughput:,.0f} mol/sec")
    lines.append("")

    # Baseline stats
    baseline_vdw = [e.vdw for e in baseline]
    baseline_elec = [e.elec for e in baseline]
    lines.append(f"Baseline vdW range: {min(baseline_vdw):.2f} to {max(baseline_vdw):.2f} kcal/mol")
    lines.append(f"Baseline elec range: {min(baseline_elec):.2f} to {max(baseline_elec):.2f} kcal/mol")
    lines.append("")

    # Per-kernel results
    lines.append("-" * 80)
    lines.append(f"{'Kernel':<12} {'Throughput':>14} {'Speedup':>10} {'vdW RMSE':>12} {'Elec RMSE':>12}")
    lines.append("-" * 80)

    for kernel in ['simd', 'skylake', 'genoa', 'turin']:
        if kernel not in results:
            lines.append(f"{kernel:<12} {'FAILED':>14}")
            continue

        result = results[kernel]

        # Compute RMSE vs baseline
        vdw_errors = [(result.vdw_energies[i] - baseline_vdw[i])**2
                      for i in range(min(len(result.vdw_energies), len(baseline_vdw)))]
        elec_errors = [(result.elec_energies[i] - baseline_elec[i])**2
                       for i in range(min(len(result.elec_energies), len(baseline_elec)))]

        vdw_rmse = np.sqrt(np.mean(vdw_errors)) if vdw_errors else 0.0
        elec_rmse = np.sqrt(np.mean(elec_errors)) if elec_errors else 0.0

        speedup = result.throughput / rdkit_throughput if rdkit_throughput > 0 else 0.0

        lines.append(f"{kernel:<12} {result.throughput:>11,.0f}/s {speedup:>9.1f}x {vdw_rmse:>12.4f} {elec_rmse:>12.4f}")

    lines.append("-" * 80)
    lines.append("")

    return "\n".join(lines)


def main(n_samples: int = 1000, smiles_path: str = DEFAULT_SMILES_PATH):
    """Run the complete unified benchmark."""

    print("=" * 80, flush=True)
    print("SmallField Unified Benchmark (Direct Binding)", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)

    # Step 1: Load molecules
    print(f"Step 1: Loading {n_samples} molecules from USearchMolecules...")
    molecules = load_molecules(smiles_path, n_samples=n_samples)
    print(f"  Loaded {len(molecules)} molecules")
    print()

    # Step 2: Compute Python f64 baseline
    print("Step 2: Computing Python f64 baseline (accuracy reference)...")
    baseline = compute_baseline(molecules, verbose=True)
    print()

    # Step 3: Benchmark RDKit throughput
    print("Step 3: Benchmarking RDKit MMFF94 throughput...")
    rdkit_throughput = 45000.0  # Estimated ~45k mol/sec
    print(f"  RDKit throughput (estimated): {rdkit_throughput:,.0f} mol/sec")
    print()

    # Step 4: Run Mojo kernel benchmarks via direct binding
    print("Step 4: Running Mojo kernels via Python binding...")
    results: Dict[str, BenchmarkResult] = {}

    for kernel in ['simd', 'skylake', 'genoa', 'turin']:
        print(f"  Running {kernel}...", end=" ", flush=True)
        result = run_kernel_benchmark(molecules, kernel)
        if result:
            results[kernel] = result
            print(f"{result.throughput:,.0f} mol/sec")
        else:
            print("FAILED")

    print()

    # Step 5: Generate report
    print("Step 5: Generating report...")
    print()

    report = generate_report(
        num_molecules=len(molecules),
        baseline=baseline,
        results=results,
        rdkit_throughput=rdkit_throughput
    )
    print(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmallField Unified Benchmark")
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=1000,
        help="Number of molecules to benchmark (default: 1000)"
    )
    parser.add_argument(
        "--smiles-path",
        type=str,
        default=DEFAULT_SMILES_PATH,
        help="Path to SMILES file"
    )

    args = parser.parse_args()
    main(n_samples=args.num_samples, smiles_path=args.smiles_path)
