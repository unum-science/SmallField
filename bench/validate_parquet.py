#!/usr/bin/env python3
"""Validate SmallField output Parquet shards: schema, aggregate stats, and an RDKit MMFF94 cross-check.

Exit code 1 is a check that failed and 2 is a refused command line or an input directory holding
nothing to validate, so a broken shard and a mistyped path do not read alike.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import pyarrow.parquet as pq
from rich.table import Table

from smallfield.log import Logger, log as _default_log, thousands

log = Logger()

# The canonical schema, in the order `cli/parquet_writer.hpp` documents it, which is the authority for
# what a shard carries. A shard missing any of these was written by something other than our writer.
REQUIRED_COLUMNS: Sequence[str] = (
    "input_shard",
    "input_row",
    "smiles",
    "conformer_index",
    "status",
    "n_heavy_atoms",
    "n_atoms",
    "n_bonds",
    "molecular_weight",
    "n_conformers",
    "conformer_coords",
    "conformer_energy",
    "usrcat",
)

#: `status` value a surviving conformer carries; mirrors `conformer_status_t::success_k`.
SUCCESS_STATUS = 0

#: Largest paired energy difference, at P95, a shard may show before the run is called a failure.
ENERGY_DELTA_P95_LIMIT = 10.0


@dataclass
class ShardStats:
    """Aggregate statistics across all shards in an output directory."""

    num_shards: int = 0
    total_rows: int = 0
    total_success: int = 0
    total_failed: int = 0
    energies: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class CrossValidation:
    """Per-sample comparison of SmallField coordinates against RDKit MMFF94."""

    num_samples: int = 0
    compared: int = 0
    nan_coordinates: int = 0
    rdkit_failures: int = 0
    energy_deltas: np.ndarray = field(default_factory=lambda: np.array([]))
    #: True when RDKit is not importable, which is a skip rather than a verdict.
    skipped: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SmallField output parquet-shards.")
    parser.add_argument(
        "--input",
        required=True,
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
    """Return the parquet-shards inside ``input_dir``, sorted and optionally capped."""
    shards = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if shard_limit > 0:
        shards = shards[:shard_limit]
    return shards


def missing_columns(first_shard: str) -> List[str]:
    """Return the subset of ``REQUIRED_COLUMNS`` absent from ``first_shard``'s schema."""
    column_names = pq.read_schema(first_shard).names
    return [column for column in REQUIRED_COLUMNS if column not in column_names]


def aggregate_shard_stats(shards: Sequence[str]) -> ShardStats:
    """Scan every shard for status and energy, and collect aggregate stats."""
    stats = ShardStats(num_shards=len(shards))
    energies_per_shard: List[np.ndarray] = []
    for path in shards:
        table = pq.read_table(path, columns=["status", "conformer_energy"])
        stats.total_rows += len(table)
        status = table.column("status").to_numpy()
        stats.total_success += int(np.sum(status == SUCCESS_STATUS))
        stats.total_failed += int(np.sum(status != SUCCESS_STATUS))
        energy = table.column("conformer_energy").to_numpy(zero_copy_only=False).astype(float)
        energies_per_shard.append(energy[~np.isnan(energy)])
    if energies_per_shard:
        stats.energies = np.concatenate(energies_per_shard)
    return stats


def sample_successful_indices(first_shard: str, num_samples: int, seed: int = 42) -> np.ndarray:
    """Choose up to ``num_samples`` random row indices of successful conformers in the first shard.

    A shuffle rather than a prefix: rows are grouped by molecule and molecules follow corpus order,
    so the first N rows sample one neighbourhood of the corpus rather than the corpus.
    """
    status = pq.read_table(first_shard, columns=["status"]).column("status").to_numpy()
    candidates = np.where(status == SUCCESS_STATUS)[0]
    if len(candidates) <= num_samples:
        return candidates
    random_generator = np.random.default_rng(seed)
    return random_generator.choice(candidates, num_samples, replace=False)


def cross_validate_with_rdkit(first_shard: str, sample_indices: np.ndarray) -> CrossValidation:
    """Recompute the RDKit MMFF94 energy on the stored SmallField coordinates and compare.

    The conformer attached below is empty rather than embedded: it only has to exist at the right
    size, and embedding one per sample would silently drop the molecules ETKDG refuses, which are
    disproportionately the large ones this comparison is about.
    """
    result = CrossValidation(num_samples=len(sample_indices))
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Geometry import Point3D
    except ImportError:
        result.skipped = True
        return result

    table = pq.read_table(first_shard, columns=["smiles", "conformer_energy", "conformer_coords"])
    smiles_column = table.column("smiles")
    energy_column = table.column("conformer_energy").to_numpy(zero_copy_only=False)
    coordinates_column = table.column("conformer_coords")

    deltas: List[float] = []
    for row in sample_indices:
        row = int(row)
        smiles = smiles_column[row].as_py()
        stored_energy = float(energy_column[row])
        stored_coordinates = coordinates_column[row].as_py()
        if stored_coordinates is None:
            result.rdkit_failures += 1
            continue

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            result.rdkit_failures += 1
            continue
        molecule = Chem.AddHs(molecule)
        num_atoms = molecule.GetNumAtoms()

        coordinates = np.asarray(stored_coordinates, dtype=np.float16).astype(np.float32)
        # The parse must reproduce the atom order the writer stored, and a length mismatch is exactly
        # where it did not.
        if coordinates.size != num_atoms * 3:
            result.rdkit_failures += 1
            continue
        coordinates = coordinates.reshape(num_atoms, 3)
        if np.any(np.isnan(coordinates)):
            result.nan_coordinates += 1
            continue

        conformer = Chem.Conformer(num_atoms)
        for atom in range(num_atoms):
            conformer.SetAtomPosition(atom, Point3D(*(float(value) for value in coordinates[atom])))
        molecule.AddConformer(conformer, assignId=True)

        try:
            properties = AllChem.MMFFGetMoleculeProperties(molecule)
            if properties is None:
                result.rdkit_failures += 1
                continue
            force_field = AllChem.MMFFGetMoleculeForceField(molecule, properties, confId=0)
            if force_field is None:
                result.rdkit_failures += 1
                continue
            # Signed and paired: an absolute value hides which engine sits above the other.
            deltas.append(stored_energy - force_field.CalcEnergy())
            result.compared += 1
        except Exception:
            result.rdkit_failures += 1

    if deltas:
        finite = np.array(deltas)
        result.energy_deltas = finite[np.isfinite(finite)]
    return result


def cross_validation_passed(result: CrossValidation) -> bool:
    """A comparison that compared nothing is a failure, not a pass."""
    if result.skipped:
        return True
    if result.compared == 0 or result.energy_deltas.size == 0:
        return False
    return float(np.percentile(np.abs(result.energy_deltas), 95)) < ENERGY_DELTA_P95_LIMIT


def render_aggregate_table(stats: ShardStats) -> None:
    table = Table(title=f"Aggregate stats — {thousands(stats.num_shards)} shards", show_header=False)
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("total rows", thousands(stats.total_rows))
    success_share = (100 * stats.total_success / max(stats.total_rows, 1)) if stats.total_rows else 0.0
    table.add_row("success", f"{thousands(stats.total_success)} ({success_share:.3f} %)")
    table.add_row("failed", thousands(stats.total_failed))
    if stats.energies.size:
        table.add_row("energy P50 kcal/mol", f"{np.median(stats.energies):.1f}")
        table.add_row("energy P95 kcal/mol", f"{np.percentile(stats.energies, 95):.1f}")
        table.add_row(
            "energy range kcal/mol",
            f"[{np.min(stats.energies):.1f}, {np.max(stats.energies):.1f}]",
        )
    _default_log.console.print(table)


def render_cross_validation_table(result: CrossValidation) -> None:
    if result.skipped:
        log.warn("RDKit not available; cross-validation skipped")
        return
    table = Table(title=f"RDKit cross-validation — {thousands(result.num_samples)} samples", show_header=False)
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    # `compared` is printed whatever it is, so a validator that compares nothing cannot look like a pass.
    table.add_row("compared", f"{result.compared} / {result.num_samples}")
    table.add_row("NaN coordinates", str(result.nan_coordinates))
    table.add_row("RDKit failures", str(result.rdkit_failures))
    if result.energy_deltas.size:
        deltas = result.energy_deltas
        absolute = np.abs(deltas)
        table.add_row("SmallField − RDKit P50 kcal/mol", f"{np.median(deltas):+.3g}")
        table.add_row("SmallField − RDKit P95 kcal/mol", f"{np.percentile(deltas, 95):+.3g}")
        table.add_row("|Δ| max kcal/mol", f"{np.max(absolute):.3g}")
        table.add_row(
            "within 1 kcal",
            f"{int(np.sum(absolute < 1.0))} / {len(deltas)} ({100 * np.mean(absolute < 1.0):.3f} %)",
        )
        table.add_row(
            "within 5 kcal",
            f"{int(np.sum(absolute < 5.0))} / {len(deltas)} ({100 * np.mean(absolute < 5.0):.3f} %)",
        )
    _default_log.console.print(table)


def main() -> int:
    args = parse_args()

    shards = find_shards(args.input, args.shard_limit)
    if not shards:
        log.error("no parquet files in {}", args.input)
        return 2

    log.info("found {} output shards", thousands(len(shards)))
    missing = missing_columns(shards[0])
    for column in missing:
        log.error("MISSING column: {}", column)
    if missing:
        return 1

    render_aggregate_table(aggregate_shard_stats(shards))

    log.info("RDKit cross-validation on up to {} random successful conformers", thousands(args.num_samples))
    result = cross_validate_with_rdkit(shards[0], sample_successful_indices(shards[0], args.num_samples))
    render_cross_validation_table(result)
    return 0 if cross_validation_passed(result) else 1


if __name__ == "__main__":
    sys.exit(main())
