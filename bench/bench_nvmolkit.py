#!/usr/bin/env python3
"""nvMolKit GPU conformer generation benchmark.

Usage: python bench_nvmolkit.py [/path/to/smiles.smi] [max_molecules]

Processes molecules in chunks to survive nvMolKit's hard crash on
unsupported molecules. Reports aggregate throughput.
"""

import sys
import time

from rdkit import Chem
from rdkit.Chem import rdDistGeom

smi_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/usearch_1M_clean.smi"
max_mol = int(sys.argv[2]) if len(sys.argv) > 2 else 100000

raw_smiles = []
with open(smi_path) as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            raw_smiles.append(parts[0])
            if max_mol and len(raw_smiles) >= max_mol:
                break

print(f"parsing {len(raw_smiles)} SMILES...")
molecules = []
for smi in raw_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    mol = Chem.AddHs(mol)
    molecules.append(mol)
print(f"parsed {len(molecules)} valid molecules")

num_molecules = len(molecules)
num_conformers = 3
chunk_size = 5000
print(f"generating {num_conformers} conformers each, chunk_size={chunk_size}")
print()

from nvmolkit.embedMolecules import EmbedMolecules
from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
from nvmolkit.types import HardwareOptions

params = rdDistGeom.ETKDGv3()
params.randomSeed = 42
params.useRandomCoords = True


def bench_nvmolkit(gpu_ids, label):
    hw = HardwareOptions()
    hw.gpuIds = gpu_ids

    total_confs = 0
    total_embed_time = 0.0
    total_mmff_time = 0.0
    failed_chunks = 0

    for start in range(0, num_molecules, chunk_size):
        end = min(start + chunk_size, num_molecules)
        mols = [Chem.RWMol(m) for m in molecules[start:end]]

        t0 = time.perf_counter()
        try:
            EmbedMolecules(mols, params, confsPerMolecule=num_conformers, hardwareOptions=hw)
        except RuntimeError:
            failed_chunks += 1
            continue
        total_embed_time += time.perf_counter() - t0

        t1 = time.perf_counter()
        try:
            MMFFOptimizeMoleculesConfs(mols, hardwareOptions=hw)
        except RuntimeError:
            pass
        total_mmff_time += time.perf_counter() - t1

        total_confs += sum(m.GetNumConformers() for m in mols)

    total = total_embed_time + total_mmff_time
    print(f"nvMolKit {label}:")
    print(f"  embed:    {total_embed_time:.2f}s")
    print(f"  MMFF:     {total_mmff_time:.2f}s")
    print(f"  total:    {total:.2f}s")
    print(f"  failed chunks: {failed_chunks}/{(num_molecules + chunk_size - 1) // chunk_size}")
    print(f"  conformers generated: {total_confs:,}")
    if total > 0:
        print(f"  mol/s:    {num_molecules / total:.0f}")
        print(f"  conf/s:   {total_confs / total:.0f}")
    print()


bench_nvmolkit([0], "1 GPU")
bench_nvmolkit([], "all GPUs")
