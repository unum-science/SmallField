#!/usr/bin/env python3
"""Validate SmallField output parquet: check schema, coordinates, and MMFF94 energies.

Usage: python validate_output_parquet.py /tmp/usearch_10M_3D [num_samples]
"""

import sys
import numpy as np
import pyarrow.parquet as pq

output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/usearch_10M_3D"
num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 500

import glob
import os

shard_files = sorted(glob.glob(os.path.join(output_dir, "*.parquet")))
if not shard_files:
    print(f"no parquet files in {output_dir}")
    sys.exit(1)

print(f"found {len(shard_files)} output shards")

# read first shard to check schema
table = pq.read_table(shard_files[0])
print(f"\nschema of {os.path.basename(shard_files[0])}:")
print(table.schema)
print(f"rows: {len(table):,}")

# check required columns
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
for col in required:
    if col not in table.column_names:
        print(f"  MISSING column: {col}")

# basic stats across all available shards
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
print(f"\naggregate stats across {len(shard_files)} shards:")
print(f"  total rows:    {total_rows:,}")
print(f"  success:       {total_success:,} ({100*total_success/max(total_rows,1):.1f}%)")
print(f"  failed:        {total_failed:,}")
if len(energies_all) > 0:
    print(f"  energy P50:    {np.median(energies_all):.1f} kcal/mol")
    print(f"  energy P95:    {np.percentile(energies_all, 95):.1f} kcal/mol")
    print(f"  energy range:  [{np.min(energies_all):.1f}, {np.max(energies_all):.1f}]")

# RDKit cross-validation on a sample
print(f"\n--- RDKit cross-validation on {num_samples} random successful conformers ---")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom
    from rdkit.Geometry import Point3D
except ImportError:
    print("RDKit not available, skipping cross-validation")
    sys.exit(0)

# sample from the first available shard
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

print(f"  validated:     {valid_count} / {len(success_indices)}")
print(f"  NaN coords:   {nan_coords}")
print(f"  RDKit fails:  {rdkit_failures}")
if energy_deltas:
    deltas = np.array(energy_deltas)
    deltas = deltas[np.isfinite(deltas)]
    print(f"  energy delta P50:  {np.median(deltas):.2f} kcal/mol")
    print(f"  energy delta P95:  {np.percentile(deltas, 95):.2f} kcal/mol")
    print(f"  energy delta max:  {np.max(deltas):.2f} kcal/mol")
    print(f"  within 1 kcal:     {int(np.sum(deltas < 1.0))} / {len(deltas)} ({100*np.mean(deltas < 1.0):.1f}%)")
    print(f"  within 5 kcal:     {int(np.sum(deltas < 5.0))} / {len(deltas)} ({100*np.mean(deltas < 5.0):.1f}%)")
