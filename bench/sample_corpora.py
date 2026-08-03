#!/usr/bin/env python3
"""Draw a stratified SMILES sample across the shards of a molecule corpus.

Taking the first N rows of one shard is not a sample of the corpus. Shards are ordered by
deposition or by enumeration index, so neighbouring rows share a submission, a scaffold series,
or a reagent set. This walks every shard, takes a random row group from each, and draws evenly,
so the result reflects the corpus rather than one neighbourhood of it.

    ./sample_corpora.py --corpus pubchem --count 20000 --out /tmp/pubchem-20k.smi

Corpora are resolved through `../USearchMolecules/data`, except `real`, whose regeneration tree
is passed explicitly with --root.
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq

CORPUS_ROOTS = {
    "pubchem": "data/usearch-molecules/pubchem/parquet",
    "gdb13": "data/usearch-molecules/gdb13/parquet",
    "real": "data/usearch-molecules/real/parquet",
}


def shard_paths(root: str) -> list:
    """Input shards only — the `*.3D.parquet` outputs carry one row per conformer."""
    every = sorted(glob.glob(os.path.join(root, "*.parquet")))
    return [path for path in every if ".3D." not in os.path.basename(path)]


def draw(paths: list, total: int, seed: int) -> list:
    rng = random.Random(seed)
    per_shard = max(1, total // max(1, len(paths)))
    out = []
    for path in paths:
        if len(out) >= total:
            break
        try:
            handle = pq.ParquetFile(path)
            group = rng.randrange(handle.metadata.num_row_groups)
            column = handle.read_row_group(group, columns=["smiles"])["smiles"].to_pylist()
        except Exception as error:  # noqa: BLE001 - a damaged shard should not abort the draw
            print(f"  skipping {os.path.basename(path)}: {error}", file=sys.stderr)
            continue
        column = [entry for entry in column if entry]
        if not column:
            continue
        take = min(per_shard, len(column), total - len(out))
        out.extend(rng.sample(column, take))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=sorted(CORPUS_ROOTS), required=True)
    parser.add_argument("--root", help="override the corpus root")
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    root = arguments.root or CORPUS_ROOTS[arguments.corpus]
    paths = shard_paths(root)
    if not paths:
        print(f"no input shards under {root}", file=sys.stderr)
        return 1

    print(f"{arguments.corpus}: {len(paths)} shards under {root}")
    smiles = draw(paths, arguments.count, arguments.seed)
    with open(arguments.out, "w") as handle:
        handle.write("\n".join(smiles) + "\n")

    lengths = np.array([len(entry) for entry in smiles])
    print(f"  wrote {len(smiles)} SMILES to {arguments.out}")
    print(f"  length mean {lengths.mean():.1f}  p50 {np.percentile(lengths, 50):.0f}  max {lengths.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
