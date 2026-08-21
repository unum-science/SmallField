#!/usr/bin/env python3
"""Draw a SMILES sample across the shards of a molecule corpus, uniformly or stratified by rarity.

Taking the first N rows of one shard is not a sample of the corpus: shards are ordered by deposition
or by enumeration index, so neighbouring rows share a submission, a scaffold series, or a reagent
set. Both modes below walk every shard instead.

`--mode uniform` reproduces the corpus distribution, which is what throughput numbers want and what
makes it close to useless for finding bugs. The gate split measured over PubChem is roughly 99 % of
molecules reaching the GPU, 0.9 % refused for an element MMFF94 cannot type and 0.15 % refused for
exceeding the atom cap, so twenty thousand uniform rows carry a couple of dozen of each. Every
interesting branch arrives as a rounding error.

`--mode stratified` buckets by feature and draws equally per bucket, so the rare tail arrives with the
weight of the common middle. A molecule lands in every bucket it qualifies for, because the
combinations are where defects live: a charged macrocycle exercises more than either alone. Reservoir
sampling keeps each bucket unbiased however many rows were scanned to fill it, and a bucket that stays
short is reported rather than padded. An empty bucket is the useful output on its own — it names a
code path the source corpus cannot reach.

Corpora are resolved under `data/`, whose entries are symlinks each machine makes for itself, so
nothing here names a path outside the repository. `--root` overrides the lookup entirely.
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
import pyarrow.parquet as pq

# Relative to the repository root, which is this script's parent directory.
CORPUS_ROOTS = {
    "pubchem": "data/usearch-molecules/pubchem/parquet",
    "gdb13": "data/usearch-molecules/gdb13/parquet",
    "real": "data/usearch-molecules/real/parquet",
    # Two shards against PubChem's hundreds, and self-contained, which is what makes it the source a
    # test fixture can be rebuilt from without the full corpus mounted.
    "example": "data/usearch-molecules/example/parquet",
}

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The dispatch ladder counts atoms with hydrogens, so every count below does too. Mirrors
# `atom_limited_bin_caps_k` in `src/smallfield/types/atom_limited_bins.hpp`.
ATOM_BIN_CAPS = (16, 32, 48, 64, 80, 96, 112, 128, 192)
MAX_ATOMS_PER_MOLECULE = ATOM_BIN_CAPS[-1]

BUCKET_NAMES = (
    "atoms_small",
    "atoms_bin_edge",
    "atoms_over_cap",
    "off_mmff_element",
    "multi_fragment",
    "isotope",
    "radical",
    "formal_charge",
    "macrocycle",
    "many_stereo_centres",
    "declared_bond_stereo",
    "acyclic",
)


def shard_paths(root: str) -> list:
    """Input shards only — the `*.3D.parquet` outputs carry one row per conformer."""
    every = sorted(glob.glob(os.path.join(root, "*.parquet")))
    return [path for path in every if ".3D." not in os.path.basename(path)]


def draw(paths: list, total: int, seed: int) -> list:
    """Draw evenly from a random row group of every shard, up to ``total`` SMILES."""
    random_generator = random.Random(seed)
    per_shard = max(1, total // max(1, len(paths)))
    drawn = []
    for path in paths:
        if len(drawn) >= total:
            break
        try:
            handle = pq.ParquetFile(path)
            group = random_generator.randrange(handle.metadata.num_row_groups)
            column = handle.read_row_group(group, columns=["smiles"])["smiles"].to_pylist()
        except Exception as error:  # noqa: BLE001 - a damaged shard should not abort the draw
            print(f"  skipping {os.path.basename(path)}: {error}", file=sys.stderr)
            continue
        column = [entry for entry in column if entry]
        if not column:
            continue
        take = min(per_shard, len(column), total - len(drawn))
        drawn.extend(random_generator.sample(column, take))
    return drawn


def stream_smiles(paths: list, seed: int):
    """Yield every SMILES in the shards at ``paths``, shard by shard.

    ``seed`` shuffles the row groups within each shard, so a caller that stops early still holds a
    sample of the whole file rather than its opening rows.
    """
    random_generator = random.Random(seed)
    for path in paths:
        try:
            handle = pq.ParquetFile(path)
            groups = list(range(handle.metadata.num_row_groups))
        except Exception as error:  # noqa: BLE001 - a damaged shard must not abort the draw
            print(f"  skipping {os.path.basename(path)}: {error}", file=sys.stderr)
            continue
        random_generator.shuffle(groups)
        for group in groups:
            try:
                column = handle.read_row_group(group, columns=["smiles"])["smiles"].to_pylist()
            except Exception as error:  # noqa: BLE001
                print(f"  skipping a row group of {os.path.basename(path)}: {error}", file=sys.stderr)
                continue
            for entry in column:
                if entry:
                    yield entry


def buckets_for(molecule, parent, chemistry) -> list:
    """Every bucket this molecule qualifies for, which is usually more than one.

    ``molecule`` is the whole parse and answers only the fragment count. ``parent`` is the largest
    fragment with hydrogens, which is what `generate_conformers` embeds, and answers everything else:
    counting the whole parse instead puts a salt in the wrong atom bin, since counter-ions carry it
    past a cap the pipeline never sees it near, and refuses it for an element sitting in a fragment
    that is discarded before typing. ``chemistry`` is the `(Chem, AllChem)` pair, passed in rather
    than imported here so that a uniform draw needs neither.
    """
    Chem, AllChem = chemistry
    atom_count = parent.GetNumAtoms()
    ring_sizes = [len(ring) for ring in parent.GetRingInfo().AtomRings()]

    found = []
    if atom_count <= ATOM_BIN_CAPS[0]:
        found.append("atoms_small")
    if any(abs(atom_count - cap) <= 1 for cap in ATOM_BIN_CAPS):
        found.append("atoms_bin_edge")
    if atom_count > MAX_ATOMS_PER_MOLECULE:
        found.append("atoms_over_cap")
    # The pipeline's own question asked the pipeline's way: not "is this element in a list" but "does
    # MMFF94 carry parameters for this molecule", which is what the host gate refuses on.
    if not AllChem.MMFFHasAllMoleculeParams(parent):
        found.append("off_mmff_element")
    # The one test that wants the whole parse: the fragments are the subject, not a distraction.
    if len(Chem.GetMolFrags(molecule)) > 1:
        found.append("multi_fragment")
    if any(atom.GetIsotope() for atom in parent.GetAtoms()):
        found.append("isotope")
    # Over-counts: RDKit gives a bare `[Gd+3]` one unpaired electron and `[Sb+3]` or `[Bi+3]` two, so a
    # corpus carrying those salts lands them here. They are still worth drawing, since the pipeline sees
    # the same thing, but this bucket is not a count of genuine radicals.
    if any(atom.GetNumRadicalElectrons() for atom in parent.GetAtoms()):
        found.append("radical")
    if any(atom.GetFormalCharge() for atom in parent.GetAtoms()):
        found.append("formal_charge")
    # Under-counts, and the miss is silent: SSSR decomposes a cucurbituril cavity into eight- and
    # five-membered rings, so a genuinely macrocyclic host reports a largest ring of 8 and never lands
    # here. Any rule keyed on ring size shares the blind spot.
    if any(size > 8 for size in ring_sizes):
        found.append("macrocycle")
    if len(Chem.FindMolChiralCenters(parent, includeUnassigned=True)) >= 4:
        found.append("many_stereo_centres")
    if any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in parent.GetBonds()):
        found.append("declared_bond_stereo")
    if not ring_sizes:
        found.append("acyclic")
    return found


def stratify(paths: list, per_bucket: int, max_scanned: int, seed: int, chemistry, parent_of) -> tuple:
    """Reservoir-sample each bucket from the shards at ``paths`` to a depth of ``per_bucket`` rows.

    Stops once every bucket is full or ``max_scanned`` rows have been read, whichever comes first, so
    a bucket the corpus cannot fill bounds the run rather than exhausting the whole source. ``seed``
    fixes both the row-group order and the reservoir's replacement choices, so a rerun draws the same
    sample. ``chemistry`` is the `(Chem, AllChem)` pair and ``parent_of`` resolves a SMILES to the
    fragment the pipeline embeds; both are passed in for the reason `buckets_for` gives.

    Returns the reservoirs, how many rows each bucket saw, how many were scanned, and how many failed
    to parse — the seen counts being what turns a draw into a measured rate.
    """
    Chem, _ = chemistry
    random_generator = random.Random(seed)
    reservoirs = {name: [] for name in BUCKET_NAMES}
    seen_counts = {name: 0 for name in BUCKET_NAMES}
    scanned = 0
    unparseable = 0

    for smiles in stream_smiles(paths, seed):
        if scanned >= max_scanned:
            break
        scanned += 1
        molecule = Chem.MolFromSmiles(smiles)
        parent = parent_of(smiles) if molecule is not None else None
        if molecule is None or parent is None:
            unparseable += 1
            continue
        for name in buckets_for(molecule, parent, chemistry):
            seen_counts[name] += 1
            reservoir = reservoirs[name]
            if len(reservoir) < per_bucket:
                reservoir.append(smiles)
            else:
                # The draw stays uniform over everything seen, so a bucket filled early still speaks
                # for the whole scan rather than for its first rows.
                index = random_generator.randrange(seen_counts[name])
                if index < per_bucket:
                    reservoir[index] = smiles
        if scanned % 25000 == 0:
            short = sum(1 for name in reservoirs if len(reservoirs[name]) < per_bucket)
            print(f"  scanned {scanned}, {short} bucket(s) still short", file=sys.stderr)
        if all(len(reservoir) >= per_bucket for reservoir in reservoirs.values()):
            break

    return reservoirs, seen_counts, scanned, unparseable


def report_uniform(smiles: list, out: str) -> None:
    """Report the ``smiles`` written to ``out``, with the length spread that describes the draw."""
    lengths = np.array([len(entry) for entry in smiles])
    print(
        "\n".join(
            [
                f"  wrote {len(smiles)} SMILES to {out}",
                f"  length mean {lengths.mean():.1f}  p50 {np.percentile(lengths, 50):.0f}  max {lengths.max()}",
            ]
        )
    )


def report_stratified(
    reservoirs: dict, seen_counts: dict, scanned: int, unparseable: int, per_bucket: int, unique: list, out: str
) -> None:
    """Report what each bucket drew and what it cost to draw it.

    ``reservoirs`` and ``seen_counts`` come straight from `stratify`, and printing both is the point:
    the drawn count says what the file holds while the seen count over ``scanned`` says how rare the
    feature was, which is the number that decides whether a uniform draw could have found it at all.
    ``unparseable`` is reported rather than folded into the scan, since a corpus that will not parse
    is a different problem from one that lacks a feature. ``unique`` is what was written to ``out``,
    smaller than the reservoirs combined because a molecule in three buckets is written once. Any
    bucket below ``per_bucket`` is named at the end rather than padded from elsewhere.
    """
    rows = [
        f"  {name:<24}{len(reservoirs[name]):>7}{seen_counts[name]:>10}"
        f"{f'{100 * seen_counts[name] / scanned:.3f} %' if scanned else '-':>9}"
        for name in BUCKET_NAMES
    ]
    print(
        "\n".join(
            [
                f"  scanned {scanned} rows, {unparseable} unparseable",
                f"  wrote {len(unique)} unique SMILES to {out}",
                f"  {'bucket':<24}{'drawn':>7}{'seen':>10}{'rate':>9}",
                *rows,
            ]
        )
    )
    for name in BUCKET_NAMES:
        if len(reservoirs[name]) < per_bucket:
            reached = "nothing in the scan reached it" if not reservoirs[name] else "the scan ran out first"
            print(f"  [short] {name}: {len(reservoirs[name])} of {per_bucket}, {reached}")


def main() -> int:
    """Resolve the corpus, draw it in the requested mode, and report what the draw covered."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=sorted(CORPUS_ROOTS), required=True)
    parser.add_argument("--mode", choices=("uniform", "stratified"), default="uniform")
    parser.add_argument("--root", help="override the corpus root")
    parser.add_argument("--count", type=int, default=20000, help="uniform mode: total SMILES to draw")
    parser.add_argument("--per-bucket", type=int, default=500, help="stratified mode: SMILES per bucket")
    parser.add_argument(
        "--max-scanned",
        type=int,
        default=400000,
        help="stratified mode: rows to read before giving up on the buckets still short",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    # A `--root` is taken as typed; a named corpus resolves under the repository, so any cwd works.
    root = arguments.root or os.path.join(REPOSITORY_ROOT, CORPUS_ROOTS[arguments.corpus])
    paths = shard_paths(root)
    if not paths:
        print(f"no input shards under {root}", file=sys.stderr)
        return 1
    print(f"{arguments.corpus}: {len(paths)} shards under {root}")

    if arguments.mode == "uniform":
        smiles = draw(paths, arguments.count, arguments.seed)
        with open(arguments.out, "w") as handle:
            handle.write("\n".join(smiles) + "\n")
        report_uniform(smiles, arguments.out)
        return 0

    # Both imports land here rather than at the top: every bucket test needs a parsed molecule and a
    # uniform draw needs none, so the cheaper mode keeps working on a machine carrying neither. The
    # `python/` entry is `pyproject.toml`'s `pythonpath`, which only pytest reads, so a script run
    # directly has to put it there itself.
    sys.path.insert(0, os.path.join(REPOSITORY_ROOT, "python"))
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem

        from smallfield.chemistry import parent_with_hydrogens
    except ImportError:
        print(
            "[error] stratified mode needs RDKit and `python/smallfield`, since every bucket test "
            "parses the molecule and measures the fragment the pipeline embeds.",
            file=sys.stderr,
        )
        raise
    RDLogger.DisableLog("rdApp.*")

    reservoirs, seen_counts, scanned, unparseable = stratify(
        paths, arguments.per_bucket, arguments.max_scanned, arguments.seed, (Chem, AllChem), parent_with_hydrogens
    )
    # A molecule in three buckets is written once; the counts still report it thrice, because what each
    # bucket covers is the question, not how many rows the file holds.
    unique = sorted({smiles for reservoir in reservoirs.values() for smiles in reservoir})
    with open(arguments.out, "w") as handle:
        handle.write("\n".join(unique) + "\n")
    report_stratified(reservoirs, seen_counts, scanned, unparseable, arguments.per_bucket, unique, arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
