#!/usr/bin/env python3
"""Measure the stereochemical fidelity of SmallField output shards: tetrahedral CIP codes and E/Z bonds.

The control decides whether the number means anything. RDKit's own ETKDGv3 geometries are scored
through the identical path first, and a control above `CONTROL_INVERSION_LIMIT` means the instrument
is broken rather than the pipeline — the failure mode that produced, and then hid, a withdrawn energy
table in this project's own history.

Exit code 1 is a check that failed and 2 is a refused command line or an input directory holding
nothing to validate, so a broken shard and a mistyped path do not read alike.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
from rich.table import Table

from smallfield.chemistry import parent_with_hydrogens
from smallfield.log import Logger, log as _default_log, thousands

log = Logger()

#: `status` value a surviving conformer carries; mirrors `conformer_status_t::success_k`.
SUCCESS_STATUS = 0

#: Inversion share, in percent, the RDKit control may show before the instrument is called broken.
#: RDKit scoring its own geometry has no mechanism to fail, so anything above zero is a harness bug.
CONTROL_INVERSION_LIMIT = 0.5

#: Inversion share, in percent, the subject may show before the run fails. These are regression
#: ratchets set at twice the worst rate current code is recorded at, not targets: tetrahedral stereo
#: sits at 6.1 % and 5.5 % on the two standing corpora, so 12 % catches a regression toward the 20 %
#: and 23.3 % those corpora scored before the chiral-volume target landed, and toward the 22.9 % the
#: shipped corpus still carries. Override per corpus rather than loosening these, because the two
#: differ in stereo density and one ceiling for both would be the looser of the two.
SHARD_CHIRAL_INVERSION_LIMIT = 12.0

#: E/Z rests on a smaller draw — 1 of 584 declared bonds, 0.171 %, with the post-polish check in
#: place — and `BENCH.md` cautions that the sample wants a larger one. Set at 1 %, about six bonds of
#: that draw, so counting noise cannot fail a run while a regression toward the 6.441 % of the
#: pre-check configuration or the 52.08 % of the shipped corpus still does.
SHARD_EZ_INVERSION_LIMIT = 1.0

#: Share of scored rows, in percent, that may fail the bond-length window. Measured at 0.140 % over
#: 1,432 rows of the shipped PubChem corpus. This ceiling is what makes the two above meaningful: an
#: unusable row leaves the inversion ratio entirely, so without it a geometry that degrades until
#: nothing is scorable reports a perfect rate.
GEOMETRY_UNUSABLE_LIMIT = 1.0

#: Declared centers or bonds the subject's score must rest on before it is a measurement rather than
#: an accident. `StereoScore.percent` divides by `max(declared, 1)`, so zero declared renders as a
#: clean `0.000 %` — the same shape a validator that compared nothing has, and the same failure.
MINIMUM_DECLARED = 100

#: Bond-length window, in angstrom, every bond must land inside before a geometry is trusted.
#: A wrong atom-order guess fails this instantly, which is what it exists for.
BOND_LENGTH_WINDOW = (0.8, 2.2)

#: Seconds `EmbedMolecule` may spend on one control geometry. A handful of PubChem molecules take
#: minutes each and would otherwise set the runtime of the whole harness; a control that skips them
#: measures fewer molecules, which the report prints, rather than measuring them slowly.
CONTROL_EMBED_TIMEOUT_SECONDS = 10


@dataclass
class StereoScore:
    """Centers or bonds declared by the input, and how many the geometry contradicts."""

    molecules: int = 0
    declared: int = 0
    inverted: int = 0
    #: Molecules carrying at least one declared center or bond.
    carrying: int = 0
    #: Of molecules with two or more declared centers: every one inverted together.
    mirror: int = 0
    #: Of molecules with two or more declared centers: some inverted, some not.
    partial: int = 0
    #: Of molecules with two or more declared centers: none inverted.
    clean: int = 0

    @property
    def percent(self) -> float:
        return 100.0 * self.inverted / max(self.declared, 1)


@dataclass
class FidelityReport:
    """Everything one run measured, control beside subject so neither can be read alone."""

    control_chiral: StereoScore = field(default_factory=StereoScore)
    control_ez: StereoScore = field(default_factory=StereoScore)
    shard_chiral: StereoScore = field(default_factory=StereoScore)
    shard_ez: StereoScore = field(default_factory=StereoScore)
    #: Rows whose reference molecule and stored slot describe different atom counts.
    atom_count_mismatch: int = 0
    #: Rows whose geometry failed the bond-length window, so no verdict was taken.
    geometry_unusable: int = 0
    #: Rows that reached the scorer at all, which is what `geometry_unusable` is a share of. An
    #: unusable row leaves both sides of the inversion ratio, so without this denominator a geometry
    #: that degrades until nothing is scorable reports a perfect score.
    rows_scored: int = 0
    unparsable: int = 0
    #: True when RDKit is not importable, which is a skip rather than a verdict.
    skipped: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure stereochemical fidelity of SmallField output shards.")
    parser.add_argument("--input", required=True, help="Directory containing output parquet-shards (*.parquet).")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=400,
        help="Molecules scored per shard. Each one also pays for an RDKit ETKDGv3 control embedding.",
    )
    parser.add_argument(
        "--shard-limit",
        type=int,
        default=1,
        help="Cap on shards scanned (0 = all). One shard samples one neighbourhood of a corpus.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the control embedding.")
    parser.add_argument(
        "--max-chiral-inversion",
        type=float,
        default=SHARD_CHIRAL_INVERSION_LIMIT,
        help="Percent of declared tetrahedral centers the shard may invert before the run fails.",
    )
    parser.add_argument(
        "--max-ez-inversion",
        type=float,
        default=SHARD_EZ_INVERSION_LIMIT,
        help="Percent of declared E/Z bonds the shard may invert before the run fails.",
    )
    parser.add_argument(
        "--max-unusable",
        type=float,
        default=GEOMETRY_UNUSABLE_LIMIT,
        help="Percent of scored rows that may fail the bond-length window before the run fails.",
    )
    return parser.parse_args()


def find_shards(input_dir: str, shard_limit: int = 0) -> List[str]:
    """Return the parquet-shards inside ``input_dir``, sorted and optionally capped."""
    shards = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if shard_limit > 0:
        shards = shards[:shard_limit]
    return shards


def chiral_codes(molecule) -> Dict[int, str]:
    """CIP codes keyed by atom index, from whatever stereochemistry the molecule currently carries.

    CIP rather than the raw tetrahedral tag, because a tag is defined against a bond order that a
    re-perception need not reproduce, and comparing tags reports nearly everything as wrong.
    """
    from rdkit import Chem

    work = Chem.Mol(molecule)
    Chem.AssignStereochemistry(work, cleanIt=True, force=True)
    return {atom.GetIdx(): atom.GetProp("_CIPCode") for atom in work.GetAtoms() if atom.HasProp("_CIPCode")}


def bond_codes(molecule) -> Dict[Tuple[int, int], str]:
    """E/Z codes keyed by the bond's atom pair, normalized through the CIP labeller.

    `AssignStereochemistryFrom3D` spells bond stereo as CIS/TRANS relative to whichever two reference
    atoms it picked, while a SMILES parse spells it E/Z. Both sides go through `AssignCIPLabels`,
    which writes a reference-atom independent `_CIPCode` onto the bond itself.
    """
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    work = Chem.Mol(molecule)
    try:
        rdCIPLabeler.AssignCIPLabels(work)
    except Exception:
        return {}
    codes: Dict[Tuple[int, int], str] = {}
    for bond in work.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        if bond.HasProp("_CIPCode"):
            codes[tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))] = bond.GetProp("_CIPCode")
    return codes


def perceived_from_geometry(molecule, coordinates: np.ndarray):
    """Attach ``coordinates``, clear the declared stereochemistry, and re-perceive it from 3D."""
    from rdkit import Chem

    work = Chem.Mol(molecule)
    conformer = Chem.Conformer(work.GetNumAtoms())
    for atom in range(work.GetNumAtoms()):
        conformer.SetAtomPosition(atom, [float(value) for value in coordinates[atom]])
    work.RemoveAllConformers()
    work.AddConformer(conformer, assignId=True)
    Chem.RemoveStereochemistry(work)
    Chem.AssignStereochemistryFrom3D(work)
    return work


def geometry_is_plausible(molecule, coordinates: np.ndarray) -> bool:
    """Every bond must land inside ``BOND_LENGTH_WINDOW``, which a wrong atom order never does."""
    low, high = BOND_LENGTH_WINDOW
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        length = float(np.linalg.norm(coordinates[begin] - coordinates[end]))
        if not low <= length <= high:
            return False
    return True


def fold_into_(score: StereoScore, declared: int, inverted: int) -> None:
    """Accumulate one molecule's verdict, keeping mirror and partial apart.

    They are different defects with different fixes: a mirror is one global handedness failure that a
    per-center volume floor cannot touch, a partial is a per-constraint embedding failure that it can.
    """
    score.molecules += 1
    score.declared += declared
    score.inverted += inverted
    if declared >= 1:
        score.carrying += 1
    if declared >= 2:
        if inverted == declared:
            score.mirror += 1
        elif inverted == 0:
            score.clean += 1
        else:
            score.partial += 1


def score_geometry(molecule, coordinates: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return ``(chiral_declared, chiral_inverted, bond_declared, bond_inverted)``, or None if unusable."""
    if not geometry_is_plausible(molecule, coordinates):
        return None
    declared_chiral = chiral_codes(molecule)
    declared_bonds = bond_codes(molecule)
    if not declared_chiral and not declared_bonds:
        return (0, 0, 0, 0)
    perceived = perceived_from_geometry(molecule, coordinates)
    perceived_chiral = chiral_codes(perceived)
    perceived_bonds = bond_codes(perceived)
    chiral_inverted = sum(1 for atom, code in declared_chiral.items() if perceived_chiral.get(atom) != code)
    bond_inverted = sum(1 for pair, code in declared_bonds.items() if perceived_bonds.get(pair) != code)
    return (len(declared_chiral), chiral_inverted, len(declared_bonds), bond_inverted)


def control_coordinates(molecule, seed: int) -> Optional[np.ndarray]:
    """RDKit's own ETKDGv3 geometry for the same molecule, or None when its embedder refuses."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    work = Chem.Mol(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.timeout = CONTROL_EMBED_TIMEOUT_SECONDS
    if AllChem.EmbedMolecule(work, parameters) != 0:
        return None
    return work.GetConformer().GetPositions()


def measure_shard(path: str, num_samples: int, seed: int, report: FidelityReport) -> None:
    """Score up to ``num_samples`` molecules of one shard, control first and subject second."""
    table = pq.read_table(
        path, columns=["smiles", "status", "conformer_index", "n_atoms", "conformer_coords", "input_row"]
    )
    smiles_column = table.column("smiles").to_pylist()
    status = table.column("status").to_numpy(zero_copy_only=False)
    conformer_index = table.column("conformer_index").to_numpy(zero_copy_only=False)
    atom_counts = table.column("n_atoms").to_numpy(zero_copy_only=False)
    coordinates_column = table.column("conformer_coords").to_pylist()
    input_rows = table.column("input_row").to_numpy(zero_copy_only=False)

    # Selected by `input_row` at an even stride rather than by a random draw over surviving rows.
    # Which molecules survive depends on the code under test, so a draw over survivors samples a
    # different molecule set per run and two runs cannot be compared. An even stride over the shared
    # input ordering picks the same molecules on both sides, up to the few whose status differs.
    candidates = np.where((status == SUCCESS_STATUS) & (conformer_index == 0))[0]
    candidates = candidates[np.argsort(input_rows[candidates], kind="stable")]
    if candidates.size > num_samples:
        stride = candidates.size / num_samples
        candidates = candidates[(np.arange(num_samples) * stride).astype(np.int64)]

    for row in candidates:
        row = int(row)
        molecule = parent_with_hydrogens(smiles_column[row])
        if molecule is None:
            report.unparsable += 1
            continue
        stored = coordinates_column[row]
        atom_count = int(atom_counts[row])
        if stored is None or atom_count == 0 or molecule.GetNumAtoms() != atom_count:
            report.atom_count_mismatch += 1
            continue
        flat = np.asarray(stored, dtype=np.float16).astype(np.float64)
        if flat.size < 3 * atom_count:
            report.atom_count_mismatch += 1
            continue
        coordinates = flat[: 3 * atom_count].reshape(atom_count, 3)
        report.rows_scored += 1

        # Control first, and every time. A subject number taken without one is not a measurement.
        reference = control_coordinates(molecule, seed)
        if reference is not None:
            scored = score_geometry(molecule, reference)
            if scored is not None:
                fold_into_(report.control_chiral, scored[0], scored[1])
                fold_into_(report.control_ez, scored[2], scored[3])

        scored = score_geometry(molecule, coordinates)
        if scored is None:
            report.geometry_unusable += 1
            continue
        fold_into_(report.shard_chiral, scored[0], scored[1])
        fold_into_(report.shard_ez, scored[2], scored[3])


def render_report(report: FidelityReport, corpus: str, shards: int) -> None:
    if report.skipped:
        log.warn("RDKit not available; stereochemical fidelity not measured")
        return

    table = Table(title=f"Stereochemical fidelity — {corpus}, {thousands(shards)} shards", show_header=True)
    table.add_column("quantity", style="bold")
    table.add_column("control", justify="right")
    table.add_column("shard", justify="right")
    for label, control, subject in (
        ("tetrahedral centers", report.control_chiral, report.shard_chiral),
        ("E/Z bonds", report.control_ez, report.shard_ez),
    ):
        table.add_row(f"{label} declared", thousands(control.declared), thousands(subject.declared))
        table.add_row(f"{label} inverted", f"{control.percent:.3f} %", f"{subject.percent:.3f} %")
        table.add_row(
            f"{label} mirror / partial",
            f"{control.mirror} / {control.partial}",
            f"{subject.mirror} / {subject.partial}",
        )
    table.add_row(
        "molecules scored", thousands(report.control_chiral.molecules), thousands(report.shard_chiral.molecules)
    )
    _default_log.console.print(table)

    # Printed whatever they are, so a harness that scored almost nothing cannot look like a pass.
    log.info(
        "skipped: {} unparsable, {} atom-count mismatch, {} geometry unusable of {} scored ({:.3f} %)",
        thousands(report.unparsable),
        thousands(report.atom_count_mismatch),
        thousands(report.geometry_unusable),
        thousands(report.rows_scored),
        100.0 * report.geometry_unusable / max(report.rows_scored, 1),
    )


def control_is_sound(report: FidelityReport) -> bool:
    """A non-zero control means the harness is wrong, and the subject number must not be quoted."""
    if report.skipped:
        return True
    if report.control_chiral.molecules == 0:
        log.error("control scored no molecules — the instrument proved nothing")
        return False
    for label, score in (("tetrahedral", report.control_chiral), ("E/Z", report.control_ez)):
        if score.declared and score.percent > CONTROL_INVERSION_LIMIT:
            log.error(
                "{} control at {:.3f} % exceeds {:.3f} % — the instrument is broken, not the pipeline",
                label,
                score.percent,
                CONTROL_INVERSION_LIMIT,
            )
            return False
    return True


def shard_is_within_budget(report: FidelityReport, max_chiral: float, max_ez: float, max_unusable: float) -> bool:
    """The subject's own verdict, which `control_is_sound` deliberately does not take.

    Every refusal names the measured percentage beside the ceiling it was measured against, at the
    precision the ceiling is written at, so a failing run is reproducible from its own log.
    """
    if report.skipped:
        return True

    within_budget = True
    for label, score, ceiling in (
        ("tetrahedral", report.shard_chiral, max_chiral),
        ("E/Z", report.shard_ez, max_ez),
    ):
        if score.declared < MINIMUM_DECLARED:
            log.error(
                "{} shard rests on {} declared, under {} — the run measured too little to answer",
                label,
                thousands(score.declared),
                thousands(MINIMUM_DECLARED),
            )
            within_budget = False
        elif score.percent > ceiling:
            log.error("{} shard at {:.3f} % exceeds {:.3f} %", label, score.percent, ceiling)
            within_budget = False

    if report.rows_scored == 0:
        log.error("no row reached the scorer — the run proved nothing")
        return False
    unusable = 100.0 * report.geometry_unusable / report.rows_scored
    if unusable > max_unusable:
        log.error(
            "{:.3f} % of scored rows were geometrically unusable, over {:.3f} % — an unusable row "
            "leaves the inversion ratio entirely, so this is what stops a worse geometry scoring better",
            unusable,
            max_unusable,
        )
        within_budget = False
    return within_budget


def main() -> int:
    args = parse_args()

    shards = find_shards(args.input, args.shard_limit)
    if not shards:
        log.error("no parquet files in {}", args.input)
        return 2

    report = FidelityReport()
    try:
        import rdkit  # noqa: F401
    except ImportError:
        report.skipped = True
        render_report(report, os.path.basename(os.path.normpath(args.input)), len(shards))
        return 0

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    log.info("scoring up to {} molecules per shard across {} shards", thousands(args.num_samples), len(shards))
    for path in shards:
        measure_shard(path, args.num_samples, args.seed, report)

    render_report(report, os.path.basename(os.path.normpath(args.input)), len(shards))
    # Both verdicts are taken rather than short-circuited: a broken instrument and an out-of-budget
    # shard are different failures, and a run carrying both should say so once for each.
    sound = control_is_sound(report)
    within_budget = shard_is_within_budget(report, args.max_chiral_inversion, args.max_ez_inversion, args.max_unusable)
    return 0 if sound and within_budget else 1


if __name__ == "__main__":
    sys.exit(main())
