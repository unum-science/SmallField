#!/usr/bin/env python3
"""nvMolKit GPU conformer generation benchmark.

Usage: python bench_nvmolkit.py [/path/to/smiles.smi] [max_molecules]

Processes molecules in chunks to survive nvMolKit's hard crash on
unsupported molecules.  Reports aggregate throughput.
"""

import sys
import time

from rdkit import Chem
from rdkit.Chem import rdDistGeom

from smallfield.log import Logger, thousands

log = Logger()

smi_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/usearch_1M_clean.smi"
max_mol = int(sys.argv[2]) if len(sys.argv) > 2 else 100000

raw_smiles = []
with open(smi_path) as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue
        raw_smiles.append(parts[0])
        if max_mol and len(raw_smiles) >= max_mol:
            break

log.info("parsing {} SMILES", thousands(len(raw_smiles)))
molecules = []
for smi in raw_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    mol = Chem.AddHs(mol)
    molecules.append(mol)
log.info("parsed {} valid molecules", thousands(len(molecules)))

num_molecules = len(molecules)
num_conformers = 3
chunk_size = 5000
log.info("generating {} conformers each, chunk_size {}", num_conformers, thousands(chunk_size))

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
    num_chunks = (num_molecules + chunk_size - 1) // chunk_size

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
    log.info("nvMolKit {}:", label)
    log.info(
        "  embed {:.2f}s, MMFF {:.2f}s, total {:.2f}s",
        total_embed_time,
        total_mmff_time,
        total,
    )
    log.info("  failed chunks: {}/{}", failed_chunks, num_chunks)
    log.info("  conformers generated: {}", thousands(total_confs))
    if total > 0:
        log.info(
            "  [rate]{} mol/s / {} conf/s[/]",
            thousands(int(num_molecules / total)),
            thousands(int(total_confs / total)),
        )


bench_nvmolkit([0], "1 GPU")
bench_nvmolkit([], "all GPUs")
