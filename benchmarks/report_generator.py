"""
Report generator for unified SmallField benchmark.

Generates formatted tables comparing accuracy and throughput
across storage types.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from baseline_compute import EnergyComponents


@dataclass
class BenchmarkResult:
    """Results from a single storage type benchmark."""

    storage_type: str
    energies: List[EnergyComponents]
    elapsed_ns: int
    num_molecules: int
    simd_info: Dict[str, str]

    @property
    def throughput(self) -> float:
        """Molecules per second."""
        return self.num_molecules / (self.elapsed_ns / 1e9)


def compute_rmse_percent(
    baseline: List[EnergyComponents], test: List[EnergyComponents], term: str
) -> float:
    """Compute RMSE as percentage of mean baseline value."""
    baseline_values = np.array([getattr(e, term) for e in baseline])
    test_values = np.array([getattr(e, term) for e in test])

    # Handle near-zero baseline
    mean_abs_baseline = np.mean(np.abs(baseline_values))
    if mean_abs_baseline < 1e-10:
        return 0.0

    mse = np.mean((test_values - baseline_values) ** 2)
    rmse = np.sqrt(mse)
    return (rmse / mean_abs_baseline) * 100


def compute_mae(
    baseline: List[EnergyComponents], test: List[EnergyComponents], term: str
) -> float:
    """Compute Mean Absolute Error in kcal/mol."""
    baseline_values = np.array([getattr(e, term) for e in baseline])
    test_values = np.array([getattr(e, term) for e in test])
    return np.mean(np.abs(test_values - baseline_values))


def compute_max_error(
    baseline: List[EnergyComponents], test: List[EnergyComponents], term: str
) -> float:
    """Compute maximum absolute error in kcal/mol."""
    baseline_values = np.array([getattr(e, term) for e in baseline])
    test_values = np.array([getattr(e, term) for e in test])
    return np.max(np.abs(test_values - baseline_values))


def generate_report(
    num_molecules: int,
    baseline: List[EnergyComponents],
    results: Dict[str, BenchmarkResult],
    rdkit_throughput: float,
) -> str:
    """
    Generate formatted benchmark report.

    Args:
        num_molecules: Number of molecules tested
        baseline: Python f64 baseline energies
        results: Dict mapping storage type to BenchmarkResult
        rdkit_throughput: RDKit molecules/second

    Returns:
        Formatted report string
    """
    terms = ["bond", "angle", "torsion", "vdw", "elec", "total"]
    storage_types = ["f32", "bf16", "f16"]

    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("SmallField Unified Benchmark")
    lines.append("=" * 80)
    lines.append(
        f"Dataset: {num_molecules} molecules | Baseline: Python f64 | Reference: RDKit MMFF94"
    )
    lines.append("")

    # SIMD info from first result
    first_result = list(results.values())[0] if results else None
    if first_result and first_result.simd_info:
        lines.append("SIMD Configuration:")
        for key, value in first_result.simd_info.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

    # Accuracy table
    lines.append("ACCURACY (RMSE% vs Python f64)")
    lines.append("-" * 80)
    header = f"{'Storage':<10}"
    for term in terms:
        header += f"{term.capitalize():>12}"
    lines.append(header)
    lines.append("-" * 80)

    for stype in storage_types:
        if stype not in results:
            continue
        row = f"{stype:<10}"
        for term in terms:
            rmse = compute_rmse_percent(baseline, results[stype].energies, term)
            if rmse < 0.01:
                row += f"{rmse:>12.4f}"
            elif rmse < 1.0:
                row += f"{rmse:>12.3f}"
            else:
                row += f"{rmse:>12.2f}"
        lines.append(row)

    lines.append("")

    # MAE table (kcal/mol)
    lines.append("ACCURACY (MAE in kcal/mol)")
    lines.append("-" * 80)
    header = f"{'Storage':<10}"
    for term in terms:
        header += f"{term.capitalize():>12}"
    lines.append(header)
    lines.append("-" * 80)

    for stype in storage_types:
        if stype not in results:
            continue
        row = f"{stype:<10}"
        for term in terms:
            mae = compute_mae(baseline, results[stype].energies, term)
            if mae < 0.001:
                row += f"{mae:>12.6f}"
            elif mae < 0.1:
                row += f"{mae:>12.4f}"
            else:
                row += f"{mae:>12.2f}"
        lines.append(row)

    lines.append("")

    # Throughput table
    lines.append("THROUGHPUT")
    lines.append("-" * 80)
    lines.append(f"{'Implementation':<30} {'mol/sec':>15} {'Speedup':>15}")
    lines.append("-" * 80)
    lines.append(f"{'RDKit MMFF94 (C++)':<30} {rdkit_throughput:>15,.0f} {'1.00x':>15}")

    for stype in storage_types:
        if stype not in results:
            continue
        throughput = results[stype].throughput
        speedup = throughput / rdkit_throughput if rdkit_throughput > 0 else 0
        lines.append(
            f"{'SmallField ' + stype:<30} {throughput:>15,.0f} {speedup:>14.2f}x"
        )

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def parse_mojo_output(
    output: str,
) -> Tuple[List[EnergyComponents], int, int, Dict[str, str]]:
    """
    Parse output from Mojo runner.

    Returns:
        (energies, elapsed_ns, num_molecules, simd_info)
    """
    energies = []
    elapsed_ns = 0
    num_molecules = 0
    simd_info = {}

    for line in output.strip().split("\n"):
        line = line.strip()

        if line.startswith("#"):
            # Parse SIMD info comments
            if ":" in line:
                key_val = line[1:].strip()
                if ":" in key_val:
                    key, val = key_val.split(":", 1)
                    simd_info[key.strip()] = val.strip()
            continue

        if line.startswith("TIMING"):
            parts = line.split()
            if len(parts) >= 3:
                elapsed_ns = int(parts[1])
                num_molecules = int(parts[2])
            continue

        # Parse energy line: mol_idx bond angle torsion vdw elec
        parts = line.split()
        if len(parts) >= 6:
            try:
                energy = EnergyComponents(
                    bond=float(parts[1]),
                    angle=float(parts[2]),
                    torsion=float(parts[3]),
                    vdw=float(parts[4]),
                    elec=float(parts[5]),
                )
                energies.append(energy)
            except (ValueError, IndexError):
                pass

    return energies, elapsed_ns, num_molecules, simd_info


if __name__ == "__main__":
    # Demo with synthetic data
    from baseline_compute import EnergyComponents

    baseline = [
        EnergyComponents(bond=1.0, angle=2.0, torsion=0.5, vdw=-1.0, elec=0.3)
        for _ in range(100)
    ]

    # Simulate f32 results (very close)
    f32_energies = [
        EnergyComponents(
            bond=1.0001, angle=2.0002, torsion=0.5001, vdw=-1.0001, elec=0.3001
        )
        for _ in range(100)
    ]

    # Simulate f16 results (more error)
    f16_energies = [
        EnergyComponents(bond=1.005, angle=2.01, torsion=0.51, vdw=-1.01, elec=0.31)
        for _ in range(100)
    ]

    results = {
        "f32": BenchmarkResult(
            "f32", f32_energies, 1_000_000, 100, {"AVX-512": "True"}
        ),
        "f16": BenchmarkResult("f16", f16_energies, 900_000, 100, {"AVX-512": "True"}),
    }

    report = generate_report(100, baseline, results, 40000)
    print(report)
