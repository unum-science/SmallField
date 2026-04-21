"""
adversarial_molecules.py — generate a small dataset that exercises every
boundary condition in SmallField's preprocess + GPU pipeline, for use as a
fast regression signal (~2 min on 1 GPU, ~30 s on 8 GPUs).

The set is grouped by axis; each axis targets a specific code path that has
historically been the source of crashes or correctness bugs:

  1. atom-count bin edges  — the 8 atom-limited-bin caps {16,32,48,64,80,96,112,128}
     plus their immediate neighbors.  Exercises kernel grid sizing and per-bin
     shared-memory paths.
  2. per-atom interaction density — cage and fused-ring templates pushing any
     single atom into a large number of bonds/angles/torsions/oops/stretchbends,
     the class of molecule that overflowed our original uint8 per-atom counters.
  3. MMFF typing corners — hypervalent, organocharged, and boron/phosphorus
     species that exercise the MMFF94 fallback-type paths.
  4. stereochemistry boundaries — 0/1/2/many chiral centers, E/Z, atropisomers.
  5. degenerate graphs — long chains and highly-branched structures that
     stress OpenMP load balance and atom-count distribution.
  6. RDKit parse corners — explicit H, long SMILES, nested ring closures,
     kekulized vs aromatic notation.
  7. DG convergence corners — molecules whose FIRE minimization reaches max
     iterations without convergence.
  8. scaffold-cache stress — many decorations of one Bemis-Murcko scaffold
     exercising the cache hit path + the insert-if-missing contention window.

Usage:
    python adversarial_molecules.py \\
        --output test/preprocess/adversarial.smi \\
        --meta   test/preprocess/adversarial.meta.json

Output format: `adversarial.smi` is one SMILES per line, no IDs — fed into
smi_to_parquet.py for consumption by smallfield_batch.  The meta JSON
records which template each row came from, so diagnostic messages can point
at the boundary a failing molecule exercises.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable


@dataclass
class template_t:
    """A single adversarial molecule template."""

    axis: str
    name: str
    smiles_generator: Callable[[], Iterable[str]]
    target_code_path: str
    # Expansion multiplier.  Used by the scaffold-cache-stress axis to emit
    # many similar decorations of a single scaffold.
    multiplicity: int = 1


def _repeat_c(n: int) -> str:
    """Linear alkane of n carbons: ``CCC...C``."""

    return "C" * n


def _star_alkane(branch_count: int, branch_length: int) -> str:
    """Star alkane: central carbon with `branch_count` branches, each
    `branch_length` carbons long.  Valid only for branch_count ≤ 4 (carbon
    valence); for larger branch counts we build a dendrimer-style tree."""

    if branch_count <= 4:
        branches = [_repeat_c(branch_length) for _ in range(branch_count)]
        return "C(" + ")(".join(branches) + ")"
    # Dendrimer: each branch ends in a quaternary carbon that spawns two more.
    def _leaf(depth: int) -> str:
        if depth == 0:
            return _repeat_c(branch_length)
        return "C(" + _leaf(depth - 1) + ")(" + _leaf(depth - 1) + ")"

    depth = max(1, branch_count // 2)
    return "C(" + _leaf(depth) + ")(" + _leaf(depth) + ")C"


def _substituted_aniline(n: int) -> Iterable[str]:
    """Yield `n` distinct simple decorations of aniline's scaffold (c1ccc(cc1)N).
    Shares one Bemis-Murcko scaffold, so the scaffold cache sees repeated
    hash hits on the first call and steady state hits on the rest."""

    decorations = [
        "C", "CC", "CCC", "O", "OC", "OCC", "N(C)C", "F", "Cl", "Br",
        "C#N", "C(=O)C", "C(=O)O", "S", "SC", "C(F)(F)F", "C(C)(C)C",
        "OC(C)=O", "N", "NC(=O)C",
    ]
    # Generate n distinct combinations of decorations at the 2, 3, 4 positions
    # of the aniline ring.  Use integer encoding so results are deterministic
    # and reproducible across platforms.
    count = 0
    for i in range(len(decorations)):
        for j in range(len(decorations)):
            for k in range(len(decorations)):
                if count >= n:
                    return
                # Build as: c1cc(R3)c(R4)c(R2)c1N
                sub_i, sub_j, sub_k = decorations[i], decorations[j], decorations[k]
                smi = f"c1cc({sub_k})c({sub_j})c({sub_i})c1N"
                yield smi
                count += 1


def build_templates() -> list[template_t]:
    """Assemble the full list of adversarial templates, grouped by axis."""

    templates: list[template_t] = []

    # Axis 1: atom-count bin edges — C_N for N at and around every bin cap.
    bin_edges = [1, 2, 3, 4, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65,
                 79, 80, 81, 95, 96, 97, 111, 112, 113, 126, 127, 128]
    for n in bin_edges:
        templates.append(template_t(
            axis="atom_count_bin_edges",
            name=f"linear_alkane_C{n}",
            smiles_generator=lambda m=n: [_repeat_c(m)],
            target_code_path="kernel grid sizing / per-bin shared-memory",
        ))

    # Axis 2: per-atom interaction density — cages, fused systems, high-coord atoms.
    high_density = [
        ("adamantane", "C1C2CC3CC1CC(C2)C3"),
        ("bridged_bicyclic", "C1CC2CCC1C2"),
        ("pentaerythritol", "OCC(CO)(CO)CO"),
        ("hexamethylbenzene", "Cc1c(C)c(C)c(C)c(C)c1C"),
        ("cubane", "C12C3C4C1C5C2C3C45"),
        ("norbornane", "C1CC2CCC1C2"),
        ("cyclodecane", "C1CCCCCCCCC1"),
        ("benzo_fused_tetracene", "c1ccc2cc3cc4ccccc4cc3cc2c1"),
        ("porphine_like", "c1cc2cc3ccc(cc4ccc(cc5ccc(cc1[nH]2)[nH]5)n4)[nH]3"),
    ]
    for name, smi in high_density:
        templates.append(template_t(
            axis="per_atom_interaction_density",
            name=name,
            smiles_generator=lambda s=smi: [s],
            target_code_path="MMFF term enumeration / per-atom offset struct",
        ))

    # Axis 3: MMFF typing corners.
    mmff_corners = [
        ("dimethyl_sulfone", "CS(=O)(=O)C"),
        ("phosphine_oxide", "CP(=O)(C)C"),
        ("hypervalent_sulfur", "OS(=O)(=O)O"),
        ("ammonium", "C[N+](C)(C)C"),
        ("carboxylate", "CC(=O)[O-]"),
        ("boronic_ester", "CB(OC)OC"),
        ("guanidinium", "NC(=N)N"),
        ("sulfonate", "CS(=O)(=O)[O-]"),
        ("nitro", "C[N+](=O)[O-]"),
        ("thiocarbonyl", "CC(=S)C"),
    ]
    for name, smi in mmff_corners:
        templates.append(template_t(
            axis="mmff_typing_corners",
            name=name,
            smiles_generator=lambda s=smi: [s],
            target_code_path="MMFF typer fallback / charged-atom parameters",
        ))

    # Axis 4: stereochemistry boundaries.
    stereo = [
        ("achiral", "CCC"),
        ("single_chiral", "C[C@H](O)C"),
        ("two_chiral", "C[C@H](O)[C@H](O)C"),
        ("ez_double_bond", "C/C=C/C"),
        ("atropisomer", "Cc1ccc(-c2ccccc2Cl)cc1"),
        ("quad_chiral_steroid_lite", "C[C@H]1CC[C@@H]2CC[C@H]3[C@@H]4CC[C@H]4CC3CC12"),
        ("cis_trans_ring", "C1CC[C@H]2CC[C@@H]1C2"),
    ]
    for name, smi in stereo:
        templates.append(template_t(
            axis="stereo_boundaries",
            name=name,
            smiles_generator=lambda s=smi: [s],
            target_code_path="CSD-torsion stereo replay / chirality extraction",
        ))

    # Axis 5: degenerate graphs.
    degenerates = [
        ("C127_alkane", _repeat_c(127)),
        ("star_quaternary", _star_alkane(4, 30)),
        ("wide_star", _star_alkane(4, 10)),
        ("very_branched", "CC(C)(C)C(C)(C)CC(C)(C)C(C)(C)C"),
    ]
    for name, smi in degenerates:
        templates.append(template_t(
            axis="degenerate_graphs",
            name=name,
            smiles_generator=lambda s=smi: [s],
            target_code_path="atom-count sort / CPU prep load balance",
        ))

    # Axis 6: RDKit parse corners.
    parse = [
        ("explicit_h_methane", "[H]C([H])([H])[H]"),
        ("nested_ring_closure", "C1CCC2C(C1)CCC2"),
        ("kekulized_aromatic", "C1=CC=CC=C1"),
        ("aromatic_lowercase", "c1ccccc1"),
        ("long_polypeptide_motif", "C" * 100 + "(=O)N" + "C" * 100),
    ]
    for name, smi in parse:
        templates.append(template_t(
            axis="rdkit_parse_corners",
            name=name,
            smiles_generator=lambda s=smi: [s],
            target_code_path="SMILES parse latency / kekule fallback",
        ))

    # Axis 7: DG convergence corners — molecules that tend to hit FIRE max iter.
    dg_corners = [
        ("isocyanide", "[N+]#[C-]c1ccccc1"),
        ("many_rotatable", "CCCCCCCCCC(C)(C)CCCCCCCCCC"),
        ("highly_strained_spiro", "C1CC2(CC1)CCC(CC2)"),
        ("bicyclic_near_planar", "C1CC2CCC1CC2"),
    ]
    for name, smi in dg_corners:
        templates.append(template_t(
            axis="dg_convergence_corners",
            name=name,
            smiles_generator=lambda s=smi: [s],
            target_code_path="FIRE max-iter exit / heavy-rescue branch",
        ))

    # Axis 8: scaffold-cache stress — many decorations of one aniline scaffold.
    templates.append(template_t(
        axis="scaffold_cache_stress",
        name="aniline_decorated",
        smiles_generator=lambda: _substituted_aniline(1000),
        target_code_path="scaffold lookup hot path + insert-if-missing",
        multiplicity=1,  # generator yields all rows directly
    ))

    return templates


def emit(templates: list[template_t], *, output_smi: Path, meta_json: Path) -> None:
    """Instantiate templates, dedupe by SMILES string, and write SMI + meta JSON."""

    try:
        from rdkit import Chem
        canonicalise = lambda smi: Chem.CanonSmiles(smi) if smi else smi
    except ImportError:
        print("[warn] RDKit not importable; skipping canonicalization.", file=sys.stderr)
        canonicalise = lambda smi: smi

    seen: set[str] = set()
    rows: list[tuple[str, str]] = []  # (canonical_smi, template_name)

    for t in templates:
        for raw in t.smiles_generator():
            try:
                canon = canonicalise(raw)
            except Exception:
                canon = raw  # keep original if canonicalise chokes
            if not canon:
                continue
            if canon in seen:
                continue
            seen.add(canon)
            rows.append((canon, f"{t.axis}:{t.name}"))

    output_smi.parent.mkdir(parents=True, exist_ok=True)
    with output_smi.open("w") as f:
        for canon, _ in rows:
            f.write(f"{canon}\n")

    meta: dict = {
        "total_molecules": len(rows),
        "templates": [
            {
                "axis": t.axis,
                "name": t.name,
                "target_code_path": t.target_code_path,
                "multiplicity": t.multiplicity,
            }
            for t in templates
        ],
        "rows": [
            {"line_number": i, "smiles": smi, "template": tag}
            for i, (smi, tag) in enumerate(rows)
        ],
    }
    with meta_json.open("w") as f:
        json.dump(meta, f, indent=2)

    # Summary to stderr so the caller sees the shape of what landed.
    print(f"[ok] wrote {len(rows)} unique SMILES to {output_smi}", file=sys.stderr)
    print(f"[ok] wrote provenance to {meta_json}", file=sys.stderr)
    by_axis: dict[str, int] = {}
    for _, tag in rows:
        axis = tag.split(":", 1)[0]
        by_axis[axis] = by_axis.get(axis, 0) + 1
    for axis in sorted(by_axis):
        print(f"       {axis:<30s} {by_axis[axis]:>6d}", file=sys.stderr)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=here / "adversarial.smi",
                        help="Output SMILES file, one per line.")
    parser.add_argument("--meta", type=Path, default=here / "adversarial.meta.json",
                        help="Output provenance JSON.")
    args = parser.parse_args()

    templates = build_templates()
    emit(templates, output_smi=args.output, meta_json=args.meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
