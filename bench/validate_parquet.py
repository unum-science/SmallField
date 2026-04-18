#!/usr/bin/env python3
"""Validate SmallField output Parquet shards: schema, stats, and RDKit cross-check.

Run as a script::

    python validate_parquet.py --input /tmp/usearch_10M_3D \\
        --num-samples 500 --shard-limit 100

Reports:
  - Schema check on the first shard (required columns present).
  - Aggregate stats across shards (total rows, success / failure counts,
    energy percentiles).
  - Cross-validation against RDKit MMFF94: for a random sample of
    successful conformers, re-embed via RDKit, overlay SmallField
    coordinates, recompute MMFF energy, and compare.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
from rich.table import Table

from smallfield.log import Logger, log as _default_log, thousands

log = Logger()

REQUIRED_COLUMNS: Sequence[str] = (
    "input_shard",
    "input_row",
    "smiles",
    "conformer_index",
    "status",
    "solver",
    "energy_kcal_per_mol",
    "coordinates_xyz",
)


@dataclass
class ShardStats:
    """Aggregate statistics across all shards in an output directory."""

    num_shards: int = 0
    total_rows: int = 0
    total_success: int = 0
    total_failed: int = 0
    energies: np.ndarray = field(default_factory=lambda: np.array([]))
    missing_columns: List[str] = field(default_factory=list)


@dataclass
class CrossValidation:
    """Per-sample comparison of SmallField coordinates against RDKit MMFF94."""

    num_samples: int = 0
    validated: int = 0
    nan_coords: int = 0
    rdkit_failures: int = 0
    energy_deltas: np.ndarray = field(default_factory=lambda: np.array([]))
    skipped: bool = False  # true when RDKit is not importable
    ok: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SmallField output parquet-shards.")
    parser.add_argument(
        "--input",
        default="/tmp/usearch_10M_3D",
        help="Directory containing output parquet-shards (*.parquet).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="RDKit cross-validation sample count (drawn from the first shard).",
    )
    parser.add_argument(
        "--shard-limit",
        type=int,
        default=0,
        help="Cap on shards scanned for aggregate stats (0 = all).",
    )
    return parser.parse_args()


def find_shards(input_dir: str, shard_limit: int = 0) -> List[str]:
    """Return the list of parquet-shards inside ``input_dir``, sorted and optionally capped."""
    shards = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if shard_limit > 0:
        shards = shards[:shard_limit]
    return shards


def check_schema(first_shard: str) -> List[str]:
    """Return the subset of ``REQUIRED_COLUMNS`` absent from ``first_shard``'s schema."""
    table = pq.read_table(first_shard)
    return [c for c in REQUIRED_COLUMNS if c not in table.column_names]


def aggregate_shard_stats(shards: Sequence[str]) -> ShardStats:
    """Scan every shard for status/solver/energy and collect aggregate stats."""
    stats = ShardStats(num_shards=len(shards))
    energies_per_shard: List[np.ndarray] = []
    for path in shards:
        table = pq.read_table(path, columns=["status", "solver", "energy_kcal_per_mol"])
        stats.total_rows += len(table)
        status = table.column("status").to_numpy()
        solver = table.column("solver").to_numpy()
        stats.total_success += int(np.sum(status == 0))
        stats.total_failed += int(np.sum(solver == 2))
        energy = table.column("energy_kcal_per_mol").to_numpy().astype(float)
        energies_per_shard.append(energy[~np.isnan(energy)])
    if energies_per_shard:
        stats.energies = np.concatenate(energies_per_shard)
    return stats


def sample_successful_indices(first_shard: str, num_samples: int, rng_seed: int = 42) -> np.ndarray:
    """Choose up to ``num_samples`` random indices of successful conformers from the first shard."""
    table = pq.read_table(first_shard)
    status = table.column("status").to_numpy()
    solver = table.column("solver").to_numpy()
    candidates = np.where((status == 0) & (solver != 2))[0]
    if len(candidates) <= num_samples:
        return candidates
    rng = np.random.default_rng(rng_seed)
    return rng.choice(candidates, num_samples, replace=False)


def cross_validate_with_rdkit(first_shard: str, sample_indices: np.ndarray) -> CrossValidation:
    """Recompute RDKit MMFF94 energy on SmallField coords and compare."""
    result = CrossValidation(num_samples=len(sample_indices))
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDistGeom
        from rdkit.Geometry import Point3D
    except ImportError:
        result.skipped = True
        result.ok = False
        return result

    table = pq.read_table(first_shard)
    smiles_col = table.column("smiles")
    energy_col = table.column("energy_kcal_per_mol").to_numpy()
    coord_col = table.column("coordinates_xyz")

    deltas: List[float] = []
    for idx in sample_indices:
        smi = smiles_col[int(idx)].as_py()
        stored_energy = float(energy_col[idx])
        raw_coords = coord_col[int(idx)].as_py()

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            result.rdkit_failures += 1
            continue
        mol = Chem.AddHs(mol)
        n_atoms = mol.GetNumAtoms()

        fp16 = np.array(raw_coords, dtype=np.float16).astype(np.float32)
        max_atoms = len(fp16) // 3
        xs = fp16[0:max_atoms][:n_atoms]
        ys = fp16[max_atoms : 2 * max_atoms][:n_atoms]
        zs = fp16[2 * max_atoms : 3 * max_atoms][:n_atoms]
        if np.any(np.isnan(xs)) or np.any(np.isnan(ys)) or np.any(np.isnan(zs)):
            result.nan_coords += 1
            continue

        AllChem.EmbedMolecule(mol, rdDistGeom.ETKDGv3())
        if mol.GetNumConformers() == 0:
            result.rdkit_failures += 1
            continue

        conf = mol.GetConformer(0)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, Point3D(float(xs[i]), float(ys[i]), float(zs[i])))

        try:
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
            if mmff_props is None:
                result.rdkit_failures += 1
                continue
            ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=0)
            if ff is None:
                result.rdkit_failures += 1
                continue
            rdkit_energy = ff.CalcEnergy()
            deltas.append(abs(rdkit_energy - stored_energy))
            result.validated += 1
        except Exception:
            result.rdkit_failures += 1

    if deltas:
        result.energy_deltas = np.array(deltas)
        result.energy_deltas = result.energy_deltas[np.isfinite(result.energy_deltas)]
    result.ok = result.validated > 0 and (
        not deltas or float(np.percentile(result.energy_deltas, 95)) < 10.0
    )
    return result


def render_aggregate_table(stats: ShardStats) -> None:
    table = Table(title=f"Aggregate stats — {thousands(stats.num_shards)} shards", show_header=False)
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("total rows", thousands(stats.total_rows))
    success_pct = (100 * stats.total_success / max(stats.total_rows, 1)) if stats.total_rows else 0.0
    table.add_row("success", f"{thousands(stats.total_success)} ({success_pct:.1f}%)")
    table.add_row("failed", thousands(stats.total_failed))
    if stats.energies.size:
        table.add_row("energy P50 kcal/mol", f"{np.median(stats.energies):.1f}")
        table.add_row("energy P95 kcal/mol", f"{np.percentile(stats.energies, 95):.1f}")
        table.add_row(
            "energy range kcal/mol",
            f"[{np.min(stats.energies):.1f}, {np.max(stats.energies):.1f}]",
        )
    _default_log.console.print(table)


def render_xval_table(xval: CrossValidation) -> None:
    if xval.skipped:
        log.warn("RDKit not available; cross-validation skipped")
        return
    table = Table(title=f"RDKit cross-validation — {thousands(xval.num_samples)} samples", show_header=False)
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("validated", f"{xval.validated} / {xval.num_samples}")
    table.add_row("NaN coords", str(xval.nan_coords))
    table.add_row("RDKit fails", str(xval.rdkit_failures))
    if xval.energy_deltas.size:
        deltas = xval.energy_deltas
        table.add_row("energy Δ P50 kcal/mol", f"{np.median(deltas):.2f}")
        table.add_row("energy Δ P95 kcal/mol", f"{np.percentile(deltas, 95):.2f}")
        table.add_row("energy Δ max kcal/mol", f"{np.max(deltas):.2f}")
        table.add_row(
            "within 1 kcal",
            f"{int(np.sum(deltas < 1.0))} / {len(deltas)} ({100 * np.mean(deltas < 1.0):.1f}%)",
        )
        table.add_row(
            "within 5 kcal",
            f"{int(np.sum(deltas < 5.0))} / {len(deltas)} ({100 * np.mean(deltas < 5.0):.1f}%)",
        )
    _default_log.console.print(table)


def main() -> int:
    args = parse_args()

    shards = find_shards(args.input, args.shard_limit)
    if not shards:
        log.error("no parquet files in {}", args.input)
        return 1
    log.info("found {} output shards", thousands(len(shards)))

    missing = check_schema(shards[0])
    for col in missing:
        log.error("MISSING column: {}", col)
    if missing:
        return 2

    stats = aggregate_shard_stats(shards)
    render_aggregate_table(stats)

    log.info("RDKit cross-validation on up to {} random successful conformers", thousands(args.num_samples))
    sample_indices = sample_successful_indices(shards[0], args.num_samples)
    xval = cross_validate_with_rdkit(shards[0], sample_indices)
    render_xval_table(xval)

    return 0 if xval.ok else 3


if __name__ == "__main__":
    sys.exit(main())
