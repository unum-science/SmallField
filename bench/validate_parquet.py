#!/usr/bin/env python3
"""Validate SmallField output Parquet: schema, coordinates, and MMFF94 energies.

Usage: python validate_parquet.py /tmp/usearch_10M_3D [num_samples]
"""

import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from rich.table import Table

from smallfield.log import Logger, log as _default_log, thousands

log = Logger()

output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/usearch_10M_3D"
num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 500

shard_files = sorted(glob.glob(os.path.join(output_dir, "*.parquet")))
if not shard_files:
    log.error("no parquet files in {}", output_dir)
    sys.exit(1)

log.info("found {} output shards", thousands(len(shard_files)))

# Schema check on the first shard.
table = pq.read_table(shard_files[0])
log.info("schema of {}: rows {}", os.path.basename(shard_files[0]), thousands(len(table)))

required = [
    "input_shard",
    "input_row",
    "smiles",
    "conformer_index",
    "status",
    "solver",
    "energy_kcal_per_mol",
    "coordinates_xyz",
]
missing = [col for col in required if col not in table.column_names]
for col in missing:
    log.error("MISSING column: {}", col)

# Aggregate stats across all shards.
total_rows = 0
total_success = 0
total_failed = 0
energies_all = []
for path in shard_files:
    t = pq.read_table(path, columns=["status", "solver", "energy_kcal_per_mol"])
    total_rows += len(t)
    status = t.column("status").to_numpy()
    solver = t.column("solver").to_numpy()
    total_success += int(np.sum(status == 0))
    total_failed += int(np.sum(solver == 2))
    e = t.column("energy_kcal_per_mol").to_numpy()
    valid_e = e[~np.isnan(e.astype(float))]
    energies_all.append(valid_e)

energies_all = np.concatenate(energies_all) if energies_all else np.array([])

aggregate = Table(title=f"Aggregate stats — {len(shard_files)} shards", show_header=False)
aggregate.add_column("metric", style="bold")
aggregate.add_column("value", justify="right")
aggregate.add_row("total rows", thousands(total_rows))
aggregate.add_row("success", f"{thousands(total_success)} ({100 * total_success / max(total_rows, 1):.1f}%)")
aggregate.add_row("failed", thousands(total_failed))
if len(energies_all) > 0:
    aggregate.add_row("energy P50 kcal/mol", f"{np.median(energies_all):.1f}")
    aggregate.add_row("energy P95 kcal/mol", f"{np.percentile(energies_all, 95):.1f}")
    aggregate.add_row(
        "energy range kcal/mol", f"[{np.min(energies_all):.1f}, {np.max(energies_all):.1f}]"
    )
_default_log.console.print(aggregate)

# RDKit cross-validation on a sample of successful conformers.
log.info("RDKit cross-validation on {} random successful conformers", thousands(num_samples))

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom
    from rdkit.Geometry import Point3D
except ImportError:
    log.warn("RDKit not available, skipping cross-validation")
    sys.exit(0)

table = pq.read_table(shard_files[0])
status = table.column("status").to_numpy()
solver = table.column("solver").to_numpy()
success_mask = (status == 0) & (solver != 2)
success_indices = np.where(success_mask)[0]
if len(success_indices) > num_samples:
    rng = np.random.default_rng(42)
    success_indices = rng.choice(success_indices, num_samples, replace=False)

smiles_col = table.column("smiles")
energy_col = table.column("energy_kcal_per_mol").to_numpy()
coord_col = table.column("coordinates_xyz")

valid_count = 0
energy_deltas = []
nan_coords = 0
rdkit_failures = 0

for idx in success_indices:
    smi = smiles_col[int(idx)].as_py()
    stored_energy = float(energy_col[idx])
    raw_coords = coord_col[int(idx)].as_py()

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        rdkit_failures += 1
        continue
    mol = Chem.AddHs(mol)
    n_atoms = mol.GetNumAtoms()

    fp16_arr = np.array(raw_coords, dtype=np.float16).astype(np.float32)
    max_atoms = len(fp16_arr) // 3
    xs = fp16_arr[0:max_atoms][:n_atoms]
    ys = fp16_arr[max_atoms : 2 * max_atoms][:n_atoms]
    zs = fp16_arr[2 * max_atoms : 3 * max_atoms][:n_atoms]

    if np.any(np.isnan(xs)) or np.any(np.isnan(ys)) or np.any(np.isnan(zs)):
        nan_coords += 1
        continue

    AllChem.EmbedMolecule(mol, rdDistGeom.ETKDGv3())
    if mol.GetNumConformers() == 0:
        rdkit_failures += 1
        continue

    conf = mol.GetConformer(0)
    for i in range(n_atoms):
        conf.SetAtomPosition(i, Point3D(float(xs[i]), float(ys[i]), float(zs[i])))

    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is None:
            rdkit_failures += 1
            continue
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=0)
        if ff is None:
            rdkit_failures += 1
            continue
        rdkit_energy = ff.CalcEnergy()
        delta = abs(rdkit_energy - stored_energy)
        energy_deltas.append(delta)
        valid_count += 1
    except Exception:
        rdkit_failures += 1

xval = Table(title=f"RDKit cross-validation — {len(success_indices)} samples", show_header=False)
xval.add_column("metric", style="bold")
xval.add_column("value", justify="right")
xval.add_row("validated", f"{valid_count} / {len(success_indices)}")
xval.add_row("NaN coords", str(nan_coords))
xval.add_row("RDKit fails", str(rdkit_failures))
if energy_deltas:
    deltas = np.array(energy_deltas)
    deltas = deltas[np.isfinite(deltas)]
    xval.add_row("energy Δ P50 kcal/mol", f"{np.median(deltas):.2f}")
    xval.add_row("energy Δ P95 kcal/mol", f"{np.percentile(deltas, 95):.2f}")
    xval.add_row("energy Δ max kcal/mol", f"{np.max(deltas):.2f}")
    xval.add_row(
        "within 1 kcal", f"{int(np.sum(deltas < 1.0))} / {len(deltas)} ({100 * np.mean(deltas < 1.0):.1f}%)"
    )
    xval.add_row(
        "within 5 kcal", f"{int(np.sum(deltas < 5.0))} / {len(deltas)} ({100 * np.mean(deltas < 5.0):.1f}%)"
    )
_default_log.console.print(xval)
