#!/usr/bin/env python3
"""SmallField GPU conformer generation benchmark.

Usage: python bench_smallfield.py [/path/to/smiles.smi] [max_molecules]
"""

import sys
import time

import numpy as np

from smallfield.log import Logger, thousands

log = Logger()

smi_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/usearch_1M_clean.smi"
max_mol = int(sys.argv[2]) if len(sys.argv) > 2 else 100000

smiles = []
with open(smi_path) as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue
        smiles.append(parts[0])
        if max_mol and len(smiles) >= max_mol:
            break

num_molecules = len(smiles)
num_conformers = 3
log.info(
    "loaded {} SMILES, generating {} conformers each (target {} conformers)",
    thousands(num_molecules),
    num_conformers,
    thousands(num_molecules * num_conformers),
)

import smallfield.api as sf


def bench_smallfield(gpu_ids, label):
    sf.Context(gpu_ids)  # warm-up
    sf.generate_conformers(smiles[:100], num_conformers=num_conformers, device_ids=gpu_ids)

    t0 = time.perf_counter()
    _coords, _energies, status = sf.generate_conformers(
        smiles, num_conformers=num_conformers, device_ids=gpu_ids
    )
    total = time.perf_counter() - t0

    status_arr = np.array(status, dtype=np.uint8)
    num_success = int(np.sum(status_arr == 0))

    log.info(
        "SmallField {}: total {:.2f}s  succeeded {}/{}  [rate]{} mol/s / {} conf/s[/]",
        label,
        total,
        thousands(num_success),
        thousands(num_molecules * num_conformers),
        thousands(int(num_molecules / total)),
        thousands(int(num_success / total)),
    )


bench_smallfield([0], "1 GPU")
bench_smallfield([], "all GPUs")
