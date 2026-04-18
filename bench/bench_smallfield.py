#!/usr/bin/env python3
"""SmallField GPU conformer generation benchmark.

Usage: python bench_smallfield_py.py [/path/to/smiles.smi] [max_molecules]
"""

import sys
import time

smi_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/usearch_1M_clean.smi"
max_mol = int(sys.argv[2]) if len(sys.argv) > 2 else 100000

smiles = []
with open(smi_path) as f:
    for line in f:
        s = line.strip().split()[0] if line.strip() else ""
        if s:
            smiles.append(s)
            if max_mol and len(smiles) >= max_mol:
                break

num_molecules = len(smiles)
num_conformers = 3
print(f"loaded {num_molecules} SMILES, generating {num_conformers} conformers each")
print(f"total conformers target: {num_molecules * num_conformers:,}")
print()

import smallfield.api as sf


def bench_smallfield(gpu_ids, label):
    ctx = sf.Context(gpu_ids)
    sf.generate_conformers(smiles[:100], num_conformers=num_conformers, device_ids=gpu_ids)

    t0 = time.perf_counter()
    coords, energies, status = sf.generate_conformers(smiles, num_conformers=num_conformers, device_ids=gpu_ids)
    total = time.perf_counter() - t0

    import numpy as np

    status_arr = np.array(status, dtype=np.uint8)
    num_success = int(np.sum(status_arr == 0))

    print(f"SmallField {label}:")
    print(f"  total:    {total:.2f}s")
    print(f"  conformers succeeded: {num_success:,} / {num_molecules * num_conformers:,}")
    print(f"  mol/s:    {num_molecules / total:.0f}")
    print(f"  conf/s:   {num_success / total:.0f}")
    print()


bench_smallfield([0], "1 GPU")
bench_smallfield([], "all GPUs")
