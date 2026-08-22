"""
adversarial_molecules.py — generate a small dataset exercising every boundary condition in SmallField's
preprocess + GPU pipeline, as a fast regression signal: ~2 min on 1 GPU, ~30 s on 8 GPUs.

Grouped by axis, each targeting a code path where crashes and correctness bugs concentrate:

  1. atom-count bin edges — linear alkanes straddling every cap in `atom_limited_bin_caps_k`, counted in
     atoms with hydrogens as the ladder counts them, exercising kernel grid sizing, per-bin shared-memory
     paths, and the `too_many_atoms_k` edge past the last cap.
  2. per-atom interaction density — cage and fused-ring templates pushing one atom into a large number of
     bonds/angles/torsions/oops/stretchbends, the class a `uint8` per-atom counter overflows.
  3. MMFF typing corners — hypervalent, organocharged and boron/phosphorus species reaching the MMFF94
     fallback-type paths, plus elements MMFF94 has no type for, which the host gate refuses.
  4. stereochemistry boundaries — 0/1/2/many chiral centers, E/Z, atropisomers.
  5. degenerate graphs — long chains and highly-branched structures stressing OpenMP load balance and
     atom-count distribution.
  6. RDKit parse corners — explicit H, long SMILES, nested ring closures, kekulized vs aromatic notation.
  7. DG convergence corners — molecules whose FIRE minimization reaches max iterations without converging.
  8. scaffold-cache stress — many decorations of one Bemis-Murcko scaffold, exercising the cache hit path
     and the insert-if-missing contention window.
  9. symmetric polycyclic matching — highly-substituted polycyclic aromatics and heavily fused cages,
     whose symmetry is what makes subgraph matching blow up in recursion depth.
 10. salt and multi-fragment — where the largest-fragment choice is load-bearing: the discarded fragment
     is the bigger one, the two are the same size, or the raw parse is past the atom cap and the parent
     is not.
 11. isotopes and radicals — labelled atoms and unpaired electrons, both of which reach the parser and
     the MMFF typer by routes nothing else here takes.
 12. ring size — cycloalkanes from 12 to 64 carbons, the last landing on the atom cap from inside a ring
     rather than along a chain.
 13. declared E/Z bonds — cis and trans across chain length and substitution, polyenes carrying several
     declared bonds, and ring-constrained double bonds where the declaration is redundant.
 14. near-degenerate geometry — strained cages and crowded centres, where an embedding can start two
     atoms nearly coincident and the minimizer descends into the singularity.

Usage:
    python adversarial_molecules.py \\
        --output test/preprocess/adversarial.smi \\
        --meta   test/preprocess/adversarial.meta.json

Output: `adversarial.smi` is one SMILES per line, no IDs, fed into smi_to_parquet.py for smallfield_generate.
The meta JSON records which template each row came from, so a diagnostic can point at the boundary a
failing molecule exercises.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass
class template_t:
    """A single adversarial molecule template."""

    axis: str
    name: str
    smiles_generator: Callable[[], Iterable[str]]
    target_code_path: str
    # Expansion multiplier, used by the scaffold-cache-stress axis for many decorations of one scaffold.
    multiplicity: int = 1


def _repeat_c(carbon_count: int) -> str:
    """Linear alkane of `carbon_count` carbons: ``CCC...C``."""

    return "C" * carbon_count


def _star_alkane(branch_count: int, branch_length: int) -> str:
    """Star alkane: central carbon with `branch_count` branches of `branch_length` carbons. Carbon valence
    allows this only up to branch_count 4; beyond that it builds a dendrimer-style tree."""

    if branch_count <= 4:
        branches = [_repeat_c(branch_length) for _ in range(branch_count)]
        return "C(" + ")(".join(branches) + ")"

    def _leaf(depth: int) -> str:
        """One dendrimer branch, ending at ``depth`` zero in a chain and otherwise in two more leaves.

        Each level doubles the branch count, so the atom count grows exponentially in ``depth`` and a
        small number reaches the per-atom interaction ceiling this axis is built to test.
        """
        if depth == 0:
            return _repeat_c(branch_length)
        return "C(" + _leaf(depth - 1) + ")(" + _leaf(depth - 1) + ")"

    depth = max(1, branch_count // 2)
    return "C(" + _leaf(depth) + ")(" + _leaf(depth) + ")C"


def _substituted_aniline(sample_count: int) -> Iterable[str]:
    """Yield `sample_count` distinct simple decorations of aniline's scaffold (c1ccc(cc1)N). One shared
    Bemis-Murcko scaffold, so the cache sees repeated hash hits on the first call, steady state on the rest."""

    decorations = [
        "C",
        "CC",
        "CCC",
        "O",
        "OC",
        "OCC",
        "N(C)C",
        "F",
        "Cl",
        "Br",
        "C#N",
        "C(=O)C",
        "C(=O)O",
        "S",
        "SC",
        "C(F)(F)F",
        "C(C)(C)C",
        "OC(C)=O",
        "N",
        "NC(=O)C",
    ]
    # `sample_count` distinct decoration combinations at the aniline ring's 2, 3, 4 positions, integer-encoded
    # so results are deterministic and reproducible across platforms.
    count = 0
    for first_index in range(len(decorations)):
        for second_index in range(len(decorations)):
            for third_index in range(len(decorations)):
                if count >= sample_count:
                    return
                # Build as: c1cc(R3)c(R4)c(R2)c1N
                first_decoration = decorations[first_index]
                second_decoration = decorations[second_index]
                third_decoration = decorations[third_index]
                smiles = f"c1cc({third_decoration})c({second_decoration})c({first_decoration})c1N"
                yield smiles
                count += 1


def build_templates() -> list[template_t]:
    """Assemble the full list of adversarial templates, grouped by axis."""

    templates: list[template_t] = []

    # Axis 1: atom-count bin edges — C_N straddling every cap in `atom_limited_bin_caps_k`.
    #
    # The ladder counts atoms with hydrogens, and a linear alkane of N carbons carries 3N+2 of them, so the
    # carbon count landing on a cap is (cap - 2) // 3. Derived from the caps rather than written out, because
    # a literal list drifts the moment the ladder gains a rung. The top entry overshoots 192 on purpose: that
    # is the `too_many_atoms_k` edge.
    bin_caps = [16, 32, 48, 64, 80, 96, 112, 128, 192]
    bin_edges = sorted(
        {1, 2}
        | {
            carbon_count
            for cap in bin_caps
            for carbon_count in ((cap - 2) // 3 - 1, (cap - 2) // 3, (cap - 2) // 3 + 1)
            if carbon_count >= 1
        }
    )
    for carbon_count in bin_edges:
        templates.append(
            template_t(
                axis="atom_count_bin_edges",
                name=f"linear_alkane_C{carbon_count}",
                smiles_generator=lambda bound_count=carbon_count: [_repeat_c(bound_count)],
                target_code_path="kernel grid sizing / per-bin shared-memory",
            )
        )

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
        # Free-base porphine carries two `[nH]` and two `n`, on opposite pyrroles.
        ("porphine", "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[nH]3"),
    ]
    for name, smiles in high_density:
        templates.append(
            template_t(
                axis="per_atom_interaction_density",
                name=name,
                smiles_generator=lambda bound_smiles=smiles: [bound_smiles],
                target_code_path="MMFF term enumeration / per-atom offset struct",
            )
        )

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
    for name, smiles in mmff_corners:
        templates.append(
            template_t(
                axis="mmff_typing_corners",
                name=name,
                smiles_generator=lambda bound_smiles=smiles: [bound_smiles],
                target_code_path="MMFF typer fallback / charged-atom parameters",
            )
        )

    # Elements MMFF94 carries no type for, which the host gate refuses with `unsupported_element_k` before
    # any geometry exists. PubChem reaches this on roughly one molecule in a hundred and the corpus reached
    # it on none, so the gate shipped untested by the default draw.
    # Bare trivalent and divalent cations are avoided here: RDKit gives `[Gd+3]` one radical electron
    # and `[Sb+3]`, `[Bi+3]` two, so they would read as radicals to anything counting unpaired spins.
    # `[Fe+2]`, `[Pt+2]` and `[Zn+2]` are clean, and the rest are complexes rather than bare ions.
    off_mmff_elements = [
        ("technetium_99", "[99Tc]"),
        ("ferrocene", "[Fe+2].[cH-]1cccc1.[cH-]1cccc1"),
        ("cisplatin", "[NH3][Pt]([NH3])(Cl)Cl"),
        ("selenophene", "c1ccc[se]1"),
        ("arsenic_acid", "O[As](=O)(O)O"),
        ("auranofin", "CCP(CC)(CC)=[Au]S[C@@H]1O[C@H](COC(C)=O)[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@H]1OC(C)=O"),
        (
            "hemin",
            "CC1=C(CCC(O)=O)C2=Cc3c(C)c(C=C)c4C=C5C(C)=C(C=C)C6=[N]5[Fe]5(Cl)([N]2=C1C=c1c(C)c(CCC(O)=O)c(n15)=C6)n34",
        ),
        ("bortezomib", "CC(C)C[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)c1cnccn1)B(O)O"),
        ("sodium_tetraphenylborate", "[Na+].[B-](c1ccccc1)(c1ccccc1)(c1ccccc1)c1ccccc1"),
        ("ebselen", "O=C1N(c2ccccc2)[Se]c2ccccc21"),
        ("diphenyl_telluride", "c1ccccc1[Te]c1ccccc1"),
        ("roxarsone", "O[As](O)(=O)c1ccc(O)c(c1)[N+](=O)[O-]"),
        ("triphenylstibine", "c1ccccc1[Sb](c1ccccc1)c1ccccc1"),
    ]
    for name, smiles in off_mmff_elements:
        templates.append(
            template_t(
                axis="off_mmff_elements",
                name=name,
                smiles_generator=lambda bound_smiles=smiles: [bound_smiles],
                target_code_path="element gate / unsupported_element_k",
            )
        )

    # Axis 4: stereochemistry boundaries.
    # An atropisomer is deliberately absent. Plain SMILES carries no axial stereochemistry, so
    # `Cc1ccc(-c2ccccc2Cl)cc1` declares nothing at all, and the CXSMILES wedge form RDKit does accept
    # gets no `_CIPCode`, so `bench/validate_stereo.py` skips the bond. Colchicine and vancomycin carry
    # real axes below, and only their tetrahedral centres are scoreable.
    stereo = [
        ("achiral", "CCC"),
        ("single_chiral", "C[C@H](O)C"),
        ("two_chiral", "C[C@H](O)[C@H](O)C"),
        ("ez_double_bond", "C/C=C/C"),
        ("quad_chiral_steroid_lite", "C[C@H]1CC[C@@H]2CC[C@H]3[C@@H]4CC[C@H]4CC3CC12"),
        ("cis_trans_ring", "C1CC[C@H]2CC[C@@H]1C2"),
        # Many centres on one molecule, which is what makes the chiral draw large enough to measure.
        (
            "digoxin",
            "C[C@H]1O[C@@H](O[C@H]2CC[C@]3(C)[C@H](CC[C@H]4[C@@H]3C[C@@H](O)[C@]3(C)[C@H](CC[C@]43O)C3=CC(=O)OC3)C2)C[C@H](O)[C@H]1O[C@H]1C[C@H](O)[C@H](O[C@H]2C[C@H](O)[C@H](O)[C@@H](C)O2)[C@@H](C)O1",
        ),
        (
            "vancomycin",
            "CN[C@H](CC(C)C)C(=O)N[C@@H]1[C@H](O)c2ccc(Oc3cc4cc(Oc5ccc(cc5Cl)[C@@H](O)[C@@H]5NC(=O)[C@H](NC(=O)[C@@H]4NC(=O)[C@H](CC(N)=O)NC1=O)c1ccc(O)c(c1)-c1c(O)cc(O)cc1[C@H](NC5=O)C(=O)O)c3O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O[C@H]1C[C@](C)(N)[C@H](O)[C@H](C)O1)c(Cl)c2",
        ),
        (
            "erythromycin_a",
            "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H]([C@H]2O)N(C)C)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
        ),
        (
            "tobramycin",
            "N[C@H]1C[C@H](O)[C@@H](CN)O[C@H]1O[C@@H]1[C@@H](O)[C@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](N)[C@H]2O)[C@@H](N)C[C@@H]1N",
        ),
        (
            "paclitaxel",
            "CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c2ccccc2)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c1ccccc1)c1ccccc1)O)(C)C",
        ),
        (
            "cholic_acid",
            "C[C@H](CCC(=O)O)[C@H]1CC[C@H]2[C@@H]3[C@@H](O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3C[C@H](O)[C@]12C",
        ),
        ("trehalose", "OC[C@H]1O[C@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"),
        ("sucrose", "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O"),
        ("cholesterol", "C[C@H](CCCC(C)C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"),
        ("dexamethasone", "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO"),
        ("artemisinin", "C[C@@H]1CC[C@H]2[C@@H](C)C(=O)O[C@@H]3O[C@]4(C)CC[C@@H]1[C@]23OO4"),
        ("strychnine", "O=C1C[C@H]2OCC=C3CN4CC[C@]56c7ccccc7N1[C@H]5[C@H]4C[C@H]3[C@H]26"),
        (
            "reserpine",
            "CO[C@@H]1C[C@H]2CN3CCc4c([nH]c5cc(OC)ccc45)[C@@H]3C[C@H]2[C@@H](C(=O)OC)[C@@H]1OC(=O)c1cc(OC)c(OC)c(OC)c1",
        ),
        ("scopolamine", "CN1[C@H]2C[C@@H](OC(=O)[C@H](CO)c3ccccc3)C[C@@H]1[C@@H]1O[C@@H]21"),
        ("morphine", "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@H](O)C=C[C@H]3[C@H]1C5"),
        ("cocaine", "COC(=O)[C@H]1[C@@H](OC(=O)c2ccccc2)C[C@@H]2CC[C@H]1N2C"),
        ("quinine", "COc1ccc2nccc([C@@H](O)[C@@H]3C[C@@H]4CCN3C[C@@H]4C=C)c2c1"),
        # Bridgehead centres, ring-fused systems where the ring already fixes the relationship, and a
        # centre whose fourth substituent is a lone pair rather than an atom.
        ("camphor", "CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2"),
        ("borneol", "C[C@@]12CC[C@@H](C1(C)C)C[C@@H]2O"),
        ("alpha_pinene", "CC1=CC[C@@H]2C[C@H]1C2(C)C"),
        ("norbornane_diol", "O[C@@H]1[C@@H](O)[C@@H]2CC[C@H]1C2"),
        ("biotin", "O=C1N[C@@H]2CS[C@@H](CCCCC(=O)O)[C@@H]2N1"),
        ("penicillin_g", "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O"),
        ("griseofulvin", "C[C@@H]1CC(=O)C=C(OC)[C@]12Oc1c(Cl)c(OC)cc(OC)c1C2=O"),
        ("esomeprazole", "COc1ccc2[nH]c(nc2c1)[S@](=O)Cc1ncc(C)c(OC)c1C"),
        ("armodafinil", "NC(=O)C[S@](=O)C(c1ccccc1)c1ccccc1"),
        ("sofosbuvir", "C[C@@H](N[P@](=O)(OC[C@H]1O[C@@H](N2C=CC(=O)NC2=O)[C@](C)(F)[C@@H]1O)Oc1ccccc1)C(=O)OC(C)C"),
        ("colchicine", "COc1cc2c(c(OC)c1OC)-c1ccc(OC)c(=O)cc1[C@@H](NC(C)=O)CC2"),
        # Pseudo-asymmetric centres, where the CIP ranks tie and the label is r/s rather than R/S.
        ("pentanetriol_2R3s4S", "C[C@@H](O)[C@H](O)[C@@H](O)C"),
        ("trichloropentane_2R3s4S", "C[C@@H](Cl)[C@H](Cl)[C@@H](Cl)C"),
        ("trihydroxyglutaric_2R3s4S", "OC(=O)[C@H](O)[C@@H](O)[C@H](O)C(=O)O"),
    ]
    for name, smiles in stereo:
        templates.append(
            template_t(
                axis="stereo_boundaries",
                name=name,
                smiles_generator=lambda bound_smiles=smiles: [bound_smiles],
                target_code_path="CSD-torsion stereo replay / chirality extraction",
            )
        )

    # Axis 5: degenerate graphs.
    degenerates = [
        ("C127_alkane", _repeat_c(127)),
        ("star_quaternary", _star_alkane(4, 30)),
        ("wide_star", _star_alkane(4, 10)),
        ("very_branched", "CC(C)(C)C(C)(C)CC(C)(C)C(C)(C)C"),
    ]
    for name, smiles in degenerates:
        templates.append(
            template_t(
                axis="degenerate_graphs",
                name=name,
                smiles_generator=lambda bound_smiles=smiles: [bound_smiles],
                target_code_path="atom-count sort / CPU prep load balance",
            )
        )

    # Axis 6: RDKit parse corners.
    parse = [
        ("explicit_h_methane", "[H]C([H])([H])[H]"),
        ("nested_ring_closure", "C1CCC2C(C1)CCC2"),
        ("kekulized_aromatic", "C1=CC=CC=C1"),
        ("aromatic_lowercase", "c1ccccc1"),
        ("long_polypeptide_motif", "C" * 100 + "(=O)N" + "C" * 100),
    ]
    for name, smiles in parse:
        templates.append(
            template_t(
                axis="rdkit_parse_corners",
                name=name,
                smiles_generator=lambda bound_smiles=smiles: [bound_smiles],
                target_code_path="SMILES parse latency / kekule fallback",
            )
        )

    # Axis 7: DG convergence corners — molecules that tend to hit FIRE max iter.
    dg_corners = [
        ("isocyanide", "[C-]#[N+]c1ccccc1"),
        ("many_rotatable", "CCCCCCCCCC(C)(C)CCCCCCCCCC"),
        ("highly_strained_spiro", "C1CC2(CC1)CCC(CC2)"),
        ("bicyclic_near_planar", "C1CC2CCC1CC2"),
    ]
    for name, smiles in dg_corners:
        templates.append(
            template_t(
                axis="dg_convergence_corners",
                name=name,
                smiles_generator=lambda bound_smiles=smiles: [bound_smiles],
                target_code_path="FIRE max-iter exit / heavy-rescue branch",
            )
        )

    # Axis 8: scaffold-cache stress — many decorations of one aniline scaffold.
    #
    # Held to roughly a third of the corpus. This axis wants volume, but eight test binaries take this file
    # as their corpus fallback, and at a thousand rows it was 94 % of what they drew: every other axis was
    # noise against one scaffold. A run that genuinely wants cache contention passes its own corpus.
    templates.append(
        template_t(
            axis="scaffold_cache_stress",
            name="aniline_decorated",
            smiles_generator=lambda: _substituted_aniline(160),
            target_code_path="scaffold lookup hot path + insert-if-missing",
            multiplicity=1,  # generator yields all rows directly
        )
    )

    # Axis 9: symmetric polycyclic matching — highly-substituted polycyclic aromatics and heavily fused
    # cages, whose symmetry is what makes subgraph matching blow up in recursion depth. Preprocessing walks
    # them with the stack-only CSD torsion matcher, and `postprocess/shape_descriptors.hpp` is where a
    # `RDKit::SubstructMatch` still runs, over the CREDO environments.
    # Every benzenoid below was generated by placing hexagons on a hex lattice and letting RDKit
    # kekulize the result, so the topology follows from the geometry rather than from SMILES
    # bookkeeping, and each was confirmed by molecular formula, ring-size multiset and automorphism
    # count. Every string here parses, and each canonicalises to a structure no other entry shares.
    symmetric_polycyclics = [
        # Linear acenes, whose fusion bonds all sit [2,2] apart.
        ("naphthalene", "c1ccc2ccccc2c1"),
        ("anthracene", "c1ccc2cc3ccccc3cc2c1"),
        ("tetracene", "c1ccc2cc3cc4ccccc4cc3cc2c1"),
        ("pentacene", "c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1"),
        ("hexacene", "c1ccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc2c1"),
        ("heptacene", "c1ccc2cc3cc4cc5cc6cc7ccccc7cc6cc5cc4cc3cc2c1"),
        # Angular fusions and phenacenes.
        ("phenanthrene", "c1ccc2c(c1)ccc1ccccc12"),
        ("chrysene", "c1ccc2c(c1)ccc1c3ccccc3ccc21"),
        ("picene", "c1ccc2c(c1)ccc1c2ccc2c3ccccc3ccc21"),
        ("benz_a_anthracene", "c1ccc2cc3c(ccc4ccccc43)cc2c1"),
        ("triphenylene", "c1ccc2c(c1)c1ccccc1c1ccccc21"),
        # Helicenes: every fusion angular, so the molecule cannot be planar.
        ("tetrahelicene", "c1ccc2c(c1)ccc1ccc3ccccc3c12"),
        ("pentahelicene", "c1ccc2c(c1)ccc1ccc3ccc4ccccc4c3c12"),
        ("hexahelicene", "c1ccc2c(c1)ccc1ccc3ccc4ccc5ccccc5c4c3c12"),
        ("heptahelicene", "c1ccc2c(c1)ccc1ccc3ccc4ccc5ccc6ccccc6c5c4c3c12"),
        ("nonahelicene", "c1ccc2c(c1)ccc1ccc3ccc4ccc5ccc6ccc7ccc8ccccc8c7c6c5c4c3c12"),
        ("undecahelicene", "c1ccc2c(c1)ccc1ccc3ccc4ccc5ccc6ccc7ccc8ccc9ccc%10ccccc%10c9c8c7c6c5c4c3c12"),
        # Pericondensed sheets, where an interior atom belongs to three rings at once.
        ("pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34"),
        ("perylene", "c1cc2cccc3c4cccc5cccc(c(c1)c23)c54"),
        ("benzo_ghi_perylene", "c1cc2ccc3ccc4ccc5cccc6c(c1)c2c3c4c56"),
        ("anthanthrene", "c1cc2ccc3cc4cccc5ccc6cc(c1)c2c3c6c54"),
        ("coronene", "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61"),
        ("ovalene", "c1cc2ccc3cc4ccc5ccc6ccc7cc8ccc1c1c2c3c2c4c5c6c7c2c81"),
        ("terrylene", "c1cc2cccc3c4ccc5c6cccc7cccc(c8ccc(c(c1)c23)c4c85)c76"),
        ("quaterrylene", "c1cc2cccc3c4ccc5c6ccc7c8cccc9cccc(c%10ccc(c%11ccc(c(c1)c23)c4c%115)c6c%107)c98"),
        ("circumanthracene", "c1cc2ccc3cc4cc5ccc6ccc7ccc8cc9cc%10ccc1c1c2c3c2c4c3c5c6c7c8c3c9c2c%101"),
        ("circumpyrene", "c1cc2cc3ccc4cc5ccc6ccc7cc8ccc9cc%10ccc1c1c2c2c3c4c3c5c6c7c4c8c9c(c%101)c2c34"),
        (
            "hexabenzocoronene",
            "c1cc2c3cccc4c5cccc6c7cccc8c9cccc%10c%11cccc%12c(c1)c2c1c(c34)c(c56)c(c78)c(c%109)c1c%12%11",
        ),
        (
            "circumcoronene",
            "c1cc2cc3ccc4cc5ccc6cc7ccc8cc9ccc%10cc%11ccc%12cc1c1c2c2c3c4c3c5c6c4c7c8c5c9c%10c6c%11c%12c1c1c2c3c4c5c61",
        ),
        (
            "circumcircumcoronene",
            "c1cc2cc3cc4ccc5cc6cc7ccc8cc9cc%10ccc%11cc%12cc%13ccc%14cc%15cc%16ccc%17cc%18cc1c1c2c2c3c3c4c5c4c6c5c7c8c6c9c7c%10c%11c8c%12c9c%13c%14c%10c%15c%11c%16c%17c%12c%18c1c1c2c2c3c4c3c5c6c4c7c8c5c9c%10c6c%11c%12c1c1c2c3c4c5c61",
        ),
        # Kekulene's twelve rings enclose an eighteen-membered rim carrying inward-pointing hydrogens.
        ("kekulene", "c1cc2cc3ccc4cc5ccc6cc7ccc8cc9ccc%10cc%11ccc%12cc1c1cc%12c%11cc%10c9cc8c7cc6c5cc4c3cc21"),
        # Five-membered rings, bowls and a closed cage, none of which are planar.
        ("acenaphthylene", "C1=Cc2cccc3cccc1c23"),
        ("fluorene", "c1ccc2c(c1)Cc1ccccc1-2"),
        ("biphenylene", "c1ccc2c(c1)-c1ccccc1-2"),
        ("fluoranthene", "c1ccc2c(c1)-c1cccc3cccc-2c13"),
        ("indeno_1_2_3_cd_pyrene", "c1ccc2c(c1)-c1ccc3ccc4cccc5cc-2c1c3c45"),
        ("corannulene", "c1cc2ccc3ccc4ccc5ccc1c1c2c3c4c51"),
        ("sumanene", "c1cc2c3c4c1Cc1ccc5c(c14)c1c(ccc(c31)C2)C5"),
        ("heptacirculene", "c1cc2ccc3ccc4ccc5ccc6ccc7ccc1c1c2c3c4c5c6c71"),
        ("decacyclene", "c1cc2cccc3c2c(c1)c1c2c4cccc5cccc(c54)c2c2c4cccc5cccc(c54)c2c31"),
        (
            "fullerene_c60",
            "c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c7c2c2c7c8c3c3c8c9c4c4c9c%10c5c5c1c1c6c6c%11c2c2c7c3c3c8c4c4c9c5c1c1c6c2c3c41",
        ),
        # Bridged cages and iptycenes, where the rings are not coplanar at all.
        ("triptycene", "c1ccc2c(c1)C1c3ccccc3C2c2ccccc21"),
        ("iptycene_fragment", "c1ccc2c(c1)C1c3ccccc3C2c2ccc3ccccc3c21"),
        ("diamantane", "C1C2CC3CC1C1CC2CC3C1"),
        ("dibenzothiophene_fused", "c1ccc2c(c1)sc1c2cc2ccccc2c1"),
        # Perhalogenated, where every hydrogen is replaced and the periphery is crowded.
        ("hexachlorobenzene", "Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl"),
        ("hexabromobenzene", "Brc1c(Br)c(Br)c(Br)c(Br)c1Br"),
        ("hexaiodobenzene", "Ic1c(I)c(I)c(I)c(I)c1I"),
        ("hexabromonaphthalene", "Brc1c(Br)c(Br)c2c(Br)c(Br)c(Br)cc2c1"),
        ("hexachloronaphthalene", "Clc1c(Cl)c(Cl)c2c(Cl)c(Cl)c(Cl)cc2c1"),
        ("octachloronaphthalene", "Clc1c(Cl)c(Cl)c2c(Cl)c(Cl)c(Cl)c(Cl)c2c1Cl"),
        ("decafluorobiphenyl", "Fc1c(F)c(F)c(-c2c(F)c(F)c(F)c(F)c2F)c(F)c1F"),
        ("decachloroanthracene", "Clc1c(Cl)c(Cl)c2c(Cl)c3c(Cl)c(Cl)c(Cl)c(Cl)c3c(Cl)c2c1Cl"),
        ("perchloropentacene", "Clc1c(Cl)c(Cl)c2c(Cl)c3c(Cl)c4c(Cl)c5c(Cl)c(Cl)c(Cl)c(Cl)c5c(Cl)c4c(Cl)c3c(Cl)c2c1Cl"),
        ("decabromopyrene", "Brc1c(Br)c2c(Br)c(Br)c3c(Br)c(Br)c(Br)c4c(Br)c(Br)c(c1Br)c2c34"),
        ("pyrene_1_3_6_8_tetrabromo", "Brc1cc(Br)c2ccc3c(Br)cc(Br)c4ccc1c2c43"),
        ("dodecachlorotriphenylene", "Clc1c(Cl)c(Cl)c2c(c1Cl)c1c(Cl)c(Cl)c(Cl)c(Cl)c1c1c(Cl)c(Cl)c(Cl)c(Cl)c21"),
        (
            "dodecachlorocoronene",
            "Clc1c(Cl)c2c(Cl)c(Cl)c3c(Cl)c(Cl)c4c(Cl)c(Cl)c5c(Cl)c(Cl)c6c(Cl)c(Cl)c1c1c2c3c4c5c61",
        ),
        ("dodecafluorocoronene", "Fc1c(F)c2c(F)c(F)c3c(F)c(F)c4c(F)c(F)c5c(F)c(F)c6c(F)c(F)c1c1c2c3c4c5c61"),
        ("decachlorocorannulene", "Clc1c(Cl)c2c(Cl)c(Cl)c3c(Cl)c(Cl)c4c(Cl)c(Cl)c5c(Cl)c(Cl)c1c1c2c3c4c51"),
        # Porphyrin-type macrocycles, four rings around an eighteen-atom conjugated circuit.
        (
            "tetraphenylporphyrin",
            "C1=Cc2nc1c(-c1ccccc1)c1ccc([nH]1)c(-c1ccccc1)c1nc(c(-c3ccccc3)c3ccc([nH]3)c2-c2ccccc2)C=C1",
        ),
        ("tetramethylporphyrin", "Cc1c2nc(c(C)c3ccc([nH]3)c(C)c3nc(c(C)c4ccc1[nH]4)C=C3)C=C2"),
        ("octaethylporphyrin", "CCC1=C(CC)c2cc3[nH]c(cc4nc(cc5[nH]c(cc1n2)c(CC)c5CC)C(CC)=C4CC)c(CC)c3CC"),
        ("octabromoporphyrin", "BrC1=C(Br)c2cc3[nH]c(cc4nc(cc5[nH]c(cc1n2)c(Br)c5Br)C(Br)=C4Br)c(Br)c3Br"),
        ("phthalocyanine", "c1ccc2c(c1)-c1nc-2nc2[nH]c(nc3nc(nc4[nH]c(n1)c1ccccc41)-c1ccccc1-3)c1ccccc21"),
        # Aryl crowding, where every ring wants a torsion the neighbours will not allow.
        ("hexaphenylbenzene", "c1ccc(-c2c(-c3ccccc3)c(-c3ccccc3)c(-c3ccccc3)c(-c3ccccc3)c2-c2ccccc2)cc1"),
        ("pyrene_1_3_6_8_tetraphenyl", "c1ccc(-c2cc(-c3ccccc3)c3ccc4c(-c5ccccc5)cc(-c5ccccc5)c5ccc2c3c54)cc1"),
        (
            "hexaphenyltriphenylene",
            "c1ccc(-c2c(-c3ccccc3)c(-c3ccccc3)c3c(c2-c2ccccc2)c2c(-c4ccccc4)c(-c4ccccc4)c(-c4ccccc4)c(-c4ccccc4)c2c2c(-c4ccccc4)c(-c4ccccc4)c(-c4ccccc4)c(-c4ccccc4)c32)cc1",
        ),
        # A 61-ring sheet at 180 atoms, twelve short of the cap, for ring perception at the boundary.
        (
            "circum3coronene",
            "c1cc2cc3cc4cc5ccc6cc7cc8cc9ccc%10cc%11cc%12cc%13ccc%14cc%15cc%16cc%17ccc%18cc%19cc%20cc%21ccc%22cc%23cc%24cc1c1c2c2c3c3c4c4c5c6c5c7c6c8c7c9c%10c8c%11c9c%12c%10c%13c%14c%11c%15c%12c%16c%13c%17c%18c%14c%19c%15c%20c%16c%21c%22c%17c%23c%18c%24c1c1c2c2c3c3c4c5c4c6c5c7c8c6c9c7c%10c%11c8c%12c9c%13c%14c%10c%15c%11c%16c%17c%12c%18c1c1c2c2c3c4c3c5c6c4c7c8c5c9c%10c6c%11c%12c1c1c2c3c4c5c61",
        ),
    ]
    for name, smiles in symmetric_polycyclics:
        templates.append(
            template_t(
                axis="symmetric_polycyclic_matching",
                name=name,
                smiles_generator=lambda bound_smiles=smiles: [bound_smiles],
                target_code_path="native CSD torsion matcher / CREDO environment SubstructMatch",
            )
        )

    # Axis 10: salt and multi-fragment — 8.9 % of PubChem, and `keep_largest_fragment_k` is on the critical
    # path with nothing exercising it. The molecules that matter are the ones where the choice is
    # load-bearing: where the discarded fragment is the bigger one, where the two are the same size, and
    # where the raw parse is past the atom cap but the retained parent is not.
    salts = [
        ("sodium_salt", "CC(=O)[O-].[Na+]"),
        ("counter_ion_larger", "C.CC(=O)Oc1ccccc1C(=O)O"),
        ("two_equal_halves", "c1ccccc1C(=O)O.c1ccccc1C(=O)O"),
        ("hydrate_chain", "CCO.O.O.O.O"),
        ("hydrochloride", "CCN.Cl"),
        ("gate_crossing", "C" * 63 + "." + "C" * 63),
        # Real salt forms, where the fragment the chooser keeps is the question rather than a detail.
        ("fluoxetine_hcl", "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1.Cl"),
        ("imatinib_mesylate", "Cc1ccc(cc1Nc1nccc(n1)-c1cccnc1)NC(=O)c1ccc(CN2CCN(C)CC2)cc1.CS(=O)(=O)O"),
        ("amlodipine_besylate", "CCOC(=O)C1=C(COCCN)NC(C)=C(C1c1ccccc1Cl)C(=O)OC.OS(=O)(=O)c1ccccc1"),
        # Two copies of the parent around one counter-ion, so the chooser sees a tie between equals.
        (
            "metoprolol_tartrate",
            "COCCc1ccc(OCC(O)CNC(C)C)cc1.COCCc1ccc(OCC(O)CNC(C)C)cc1.O[C@H]([C@@H](O)C(=O)O)C(=O)O",
        ),
        # Many small fragments around one parent, which is the shape a hydrate takes.
        ("ampicillin_trihydrate", "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccccc3)C(=O)N2[C@H]1C(=O)O.O.O.O"),
        (
            "atropine_sulfate_hydrate",
            "CN1[C@H]2CC[C@@H]1C[C@@H](C2)OC(=O)C(CO)c1ccccc1.CN1[C@H]2CC[C@@H]1C[C@@H](C2)OC(=O)C(CO)c1ccccc1.OS(=O)(=O)O.O",
        ),
        (
            "quinine_sulfate_dihydrate",
            "C=C[C@H]1CN2CC[C@H]1C[C@H]2[C@H](O)c1ccnc2ccc(OC)cc12.C=C[C@H]1CN2CC[C@H]1C[C@H]2[C@H](O)c1ccnc2ccc(OC)cc12.OS(=O)(=O)O.O.O",
        ),
        (
            "morphine_sulfate_pentahydrate",
            "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5.CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5.OS(=O)(=O)O.O.O.O.O.O",
        ),
        # The counter-ion is larger than the parent, so keeping the biggest fragment keeps the wrong one.
        ("pyrantel_pamoate", "CN1CCCN=C1/C=C/c1cccs1.OC(=O)c1cc2ccccc2c(Cc2c(O)c3ccccc3cc2C(=O)O)c1O"),
        ("oxtriphylline", "C[N+](C)(C)CCO.Cn1c(=O)c2[n-]cnc2n(C)c1=O"),
        (
            "diatrizoate_meglumine",
            "CC(=O)Nc1c(I)c(C(=O)[O-])c(I)c(NC(C)=O)c1I.C[NH2+]C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",
        ),
        # The retained fragment is itself ionic, or a zwitterion, rather than neutral.
        (
            "betamethasone_phosphate",
            "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)COP(=O)([O-])[O-].[Na+].[Na+]",
        ),
        ("hexamethonium_bromide", "C[N+](C)(C)CCCCCC[N+](C)(C)C.[Br-].[Br-]"),
        ("cefepime_hcl_hydrate", "CO/N=C(\\C(=O)N[C@H]1[C@H]2SCC(=C(N2C1=O)C(=O)[O-])C[N+]1(C)CCCC1)c1csc(N)n1.Cl.O"),
        # Exact ties: identical on every metric a chooser could break the tie with.
        ("butanol_isomer_tie", "CCCCO.CC(C)CO"),
        ("quinhydrone", "Oc1ccc(O)cc1.O=C1C=CC(=O)C=C1"),
    ]
    for name, smiles in salts:
        templates.append(
            template_t(
                axis="salt_multi_fragment",
                name=name,
                smiles_generator=lambda bound_smiles=smiles: [bound_smiles],
                target_code_path="largest-fragment chooser / fragment separation",
            )
        )

    # Axis 11: isotopes and radicals — 3.2 % and 1.4 % of PubChem, neither reaching the parser here.
    isotopes = [
        ("deuterated_methanol", "[2H]C([2H])([2H])O"),
        ("carbon_13_methane", "[13CH4]"),
        ("nitrogen_15_amine", "[15NH2]CC"),
        ("benzene_d6", "[2H]c1c([2H])c([2H])c([2H])c([2H])c1[2H]"),
        # Perdeuteration, where every hydrogen is an explicit graph atom rather than implicit.
        (
            "palmitic_acid_d31",
            "[2H]C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])C(=O)O",
        ),
        ("deutetrabenazine", "[2H]C([2H])([2H])Oc1cc2c(cc1OC([2H])([2H])[2H])[C@H]1CC(=O)[C@H](CC(C)C)CN1CC2"),
        ("dextromethorphan_d3", "[2H]C([2H])([2H])Oc1ccc2c(c1)[C@]13CCCC[C@@H]1[C@H](C2)N(C)CC3.Br.O"),
        ("acetic_acid_13c", "C[13C](=O)O"),
        ("glucose_13c6", "O[13CH2][13C@H]1O[13C@H](O)[13C@H](O)[13C@@H](O)[13C@@H]1O"),
        ("glycine_13c2_15n", "[15NH2][13CH2][13C](=O)O"),
        ("fludeoxyglucose_f18", "OC[C@H]1O[C@H](O)[C@H]([18F])[C@@H](O)[C@@H]1O"),
        ("methionine_11c", "[11CH3]SCC[C@H](N)C(=O)O"),
        ("iobenguane_i131", "NC(=N)NCc1cccc([131I])c1"),
    ]
    for name, smiles in isotopes:
        templates.append(
            template_t(
                axis="isotopes",
                name=name,
                smiles_generator=lambda bound_smiles=smiles: [bound_smiles],
                target_code_path="SMILES parse of an isotope label",
            )
        )

    radicals = [
        ("carbene", "[CH2]"),
        ("benzyl_radical", "[CH2]c1ccccc1"),
        # Persistent radicals, which are real compounds rather than transient intermediates.
        ("tempo", "CC1(C)CCCC(C)(C)N1[O]"),
        ("carbamoyl_proxyl", "CC1(C)CC(C(N)=O)C(C)(C)N1[O]"),
        ("ptio_nitronyl", "CC1(C)N([O])C(=[N+]([O-])C1(C)C)c1ccccc1"),
        ("dpph", "[O-][N+](=O)c1cc([N+](=O)[O-])c(c(c1)[N+](=O)[O-])[N](N(c1ccccc1)c1ccccc1)"),
        ("blatter_radical", "[N]1c2ccccc2N(c2ccccc2)N=C1c1ccccc1"),
        ("trityl_radical", "[C](c1ccccc1)(c1ccccc1)c1ccccc1"),
        ("ttm_radical", "[C](c1c(Cl)cc(Cl)cc1Cl)(c1c(Cl)cc(Cl)cc1Cl)c1c(Cl)cc(Cl)cc1Cl"),
        # Two unpaired electrons on one molecule, and one that is charged as well as radical.
        ("totapol_biradical", "CC1(C)CC(OCC(O)CNC2CC(C)(C)N([O])C(C)(C)C2)CC(C)(C)N1[O]"),
        ("imes_carbene", "Cc1cc(C)c(c(C)c1)N1C=CN([C]1)c1c(C)cc(C)cc1C"),
        ("benzosemiquinone", "[O-]c1ccc([O])cc1"),
    ]
    for name, smiles in radicals:
        templates.append(
            template_t(
                axis="radicals",
                name=name,
                smiles_generator=lambda bound_smiles=smiles: [bound_smiles],
                target_code_path="MMFF typing of an unpaired electron",
            )
        )

    # Axis 12: ring size — PubChem carries rings past eighty atoms and nothing here exceeded ten. A
    # cycloalkane of N carbons is 3N atoms with hydrogens, so the 64-ring lands on exactly
    # `max_atoms_per_molecule_k` and the cap is reached from inside a ring rather than along a chain.
    for ring_carbons in (12, 18, 40, 64):
        templates.append(
            template_t(
                axis="ring_size",
                name=f"cycloalkane_C{ring_carbons}",
                smiles_generator=lambda bound_count=ring_carbons: ["C1" + "C" * (bound_count - 1) + "1"],
                target_code_path="ring perception / macrocycle torsion preferences",
            )
        )

    # Real macrocycles across the size range, rather than plain carbon rings. Cucurbituril is here for
    # a specific reason: RDKit's SSSR decomposes its cavity into eight- and five-membered rings, so it
    # reports a largest ring of 8 and any rule keyed on ring size will not see it as macrocyclic at all.
    macrocycles = [
        ("cyclononanone", "O=C1CCCCCCCC1"),
        ("alpha_humulene", "CC1=CC/C=C(C)/CCC(C)(C)/C=C/C1"),
        ("crown_ether_12c4", "C1COCCOCCOCCO1"),
        ("crown_ether_18c6", "C1COCCOCCOCCOCCOCCO1"),
        ("muscone", "CC1CCCCCCCCCCCCCC1=O"),
        ("calix_4_arene", "C9c(c(O)1)cccc1Cc(c(O)1)cccc1Cc(c(O)1)cccc1Cc(c(O)1)cccc19"),
        (
            "cucurbit_6_uril",
            "N91C2N8C(=O)N(C3)C2N(C1=O)CN1C2N3C(=O)N(C4)C2N(C1=O)CN1C2N4C(=O)N(C3)C2N(C1=O)CN1C2N3C(=O)N(C4)C2N(C1=O)CN1C2N4C(=O)N(C3)C2N(C1=O)CN1C2N3C(=O)N(C8)C2N(C1=O)C9",
        ),
        (
            "gramicidin_s",
            "N9[C@@H](C(C)C)C(=O)N[C@@H](CCCN)C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCCN)C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](Cc1ccccc1)C(=O)N1CCC[C@H]1C(=O)9",
        ),
        (
            "valinomycin",
            "N9[C@H](C(C)C)C(=O)O[C@@H](C)C(=O)N[C@@H](C(C)C)C(=O)O[C@H](C(C)C)C(=O)N[C@H](C(C)C)C(=O)O[C@@H](C)C(=O)N[C@@H](C(C)C)C(=O)O[C@H](C(C)C)C(=O)N[C@H](C(C)C)C(=O)O[C@@H](C)C(=O)N[C@@H](C(C)C)C(=O)O[C@H](C(C)C)C(=O)9",
        ),
        (
            "alpha_cyclodextrin",
            "[C@@H]%99%10[C@H](O)[C@@H](O)[C@@H](O[C@H]%11[C@H](O)[C@@H](O)[C@@H](O[C@H]%12[C@H](O)[C@@H](O)[C@@H](O[C@H]%13[C@H](O)[C@@H](O)[C@@H](O[C@H]%14[C@H](O)[C@@H](O)[C@@H](O[C@H]%15[C@H](O)[C@@H](O)[C@@H](O%99)O[C@@H]%15CO)O[C@@H]%14CO)O[C@@H]%13CO)O[C@@H]%12CO)O[C@@H]%11CO)O[C@@H]%10CO",
        ),
    ]
    for name, smiles in macrocycles:
        templates.append(
            template_t(
                axis="ring_size",
                name=name,
                smiles_generator=lambda bound_smiles=smiles: [bound_smiles],
                target_code_path="ring perception / macrocycle torsion preferences",
            )
        )

    # Axis 13: declared E/Z bonds — 8.6 % of PubChem against one molecule here, which is the thinnest
    # coverage in the corpus and the one the post-polish stereo work most needs. `BENCH.md` records that
    # the E/Z rate rests on a draw too small to carry weight; this is what widens it.
    bond_stereo = [
        # Synthetic minima, so the axis still covers the shortest declarations.
        ("trans_hexene", "CC/C=C/CC"),
        ("cis_butene", "C/C=C\\C"),
        ("trisubstituted_ez", "C/C(Cl)=C/C"),
        ("mixed_ez_diene", "C/C=C/C=C\\C"),
        ("ez_with_heteroatom", "C/C=C/O"),
        # Polyenes carrying many declarations at once, which is what makes the draw large enough to
        # measure: one carotenoid contributes nine bonds where a butene contributes one.
        ("beta_carotene", "CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C=C/C=C(\\C)/C=C/C=C(\\C)/C=C/C2=C(CCCC2(C)C)C)/C)/C"),
        ("lycopene", "CC(=CCC/C(=C/C=C/C(=C/C=C/C(=C/C=C/C=C(\\C)/C=C/C=C(\\C)/C=C/C=C(\\C)/CCC=C(C)C)/C)/C)/C)C"),
        ("canthaxanthin", "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)C(=O)CCC2(C)C)C(=O)CCC1(C)C"),
        ("crocetin", "C/C(=C\\C=C\\C(C)=C\\C=C\\C=C(C)\\C=C\\C=C(C)\\C(=O)O)C(=O)O"),
        ("docosahexaenoic_acid", "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCC(=O)O"),
        ("eicosapentaenoic_acid", "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)O"),
        ("arachidonic_acid", "CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)O"),
        ("alpha_parinaric_acid", "CC/C=C\\C=C\\C=C\\C=C/CCCCCCCC(=O)O"),
        ("all_trans_retinol", "OC/C=C(C)/C=C/C=C(C)/C=C/C1=C(C)CCCC1(C)C"),
        ("tretinoin", "OC(=O)/C=C(C)/C=C/C=C(C)/C=C/C1=C(C)CCCC1(C)C"),
        ("isotretinoin", "CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C\\C(=O)O)/C)/C"),
        ("alitretinoin", "CC1=C(/C=C/C(C)=C\\C=C\\C(C)=C\\C(=O)O)C(C)(C)CCC1"),
        ("eleven_cis_retinal", "CC1=C(/C=C/C(C)=C/C=C\\C(C)=C\\C=O)C(C)(C)CCC1"),
        ("acitretin", "Cc1cc(OC)c(C)c(C)c1/C=C/C(C)=C/C=C/C(C)=C/C(=O)O"),
        ("squalene", "CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C=C(C)/CC/C=C(C)/CCC=C(C)C"),
        ("deca_tetraenedioic", "OC(=O)/C=C/C=C/C=C/C=C/C(=O)O"),
        ("alpha_linolenic_acid", "CC/C=C\\C/C=C\\C/C=C\\CCCCCCCC(=O)O"),
        ("punicic_acid", "CCCC/C=C\\C=C\\C=C/CCCCCCCC(=O)O"),
        ("geranylgeraniol", "CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CO"),
        ("rumenic_acid", "CCCCCC/C=C/C=C\\CCCCCCCC(=O)O"),
        ("linoleic_acid", "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)O"),
        ("piperine", "O=C(/C=C/C=C/c1ccc2OCOc2c1)N1CCCCC1"),
        ("curcumin", "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O"),
        ("sorbic_acid", "C/C=C/C=C/C(=O)O"),
        ("muconic_acid", "OC(=O)/C=C/C=C/C(=O)O"),
        ("oleic_acid", "CCCCCCCC/C=C\\CCCCCCCC(=O)O"),
        ("elaidic_acid", "CCCCCCCC/C=C/CCCCCCCC(=O)O"),
        ("fumaric_acid", "OC(=O)/C=C/C(=O)O"),
        ("maleic_acid", "OC(=O)/C=C\\C(=O)O"),
        # Where CIP priority rather than substituent size decides the label, which is the assignment a
        # re-read can get backwards without the geometry being wrong.
        ("tamoxifen_z", "CC/C(=C(\\c1ccccc1)/c1ccc(OCCN(C)C)cc1)/c1ccccc1"),
        ("diethylstilbestrol_e", "CC/C(=C(/CC)\\c1ccc(O)cc1)/c1ccc(O)cc1"),
        ("enclomifene", "CCN(CC)CCOc1ccc(cc1)/C(=C(/Cl)\\c1ccccc1)/c1ccccc1"),
        ("dichlorobutene_e", "C/C(Cl)=C(/C)Cl"),
        ("methylpentene_e", "CC/C(C)=C/C"),
        ("bromochloropropene_e", "C/C=C(\\Cl)Br"),
        ("stilbene_e", "c1ccc(/C=C/c2ccccc2)cc1"),
        ("stilbene_z", "c1ccc(/C=C\\c2ccccc2)cc1"),
        ("resveratrol", "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1"),
        ("combretastatin_a4", "Oc1cc(ccc1OC)/C=C\\c1cc(OC)c(OC)c(OC)c1"),
        ("azobenzene_e", "c1ccc(/N=N/c2ccccc2)cc1"),
        ("fluvoxamine", "COCCCC/C(=N\\OCCN)c1ccc(C(F)(F)F)cc1"),
        # A double bond inside a ring large enough for both senses to exist.
        ("cyclooctene_z", "C1CC/C=C\\CCC1"),
        ("cyclooctene_e", "C1CC/C=C/CCC1"),
        ("cyclodecene_e", "C1CCC/C=C/CCCC1"),
        ("cyclododecene_z", "C1CCCC/C=C\\CCCCC1"),
        ("civetone", "O=C1CCCCCCC/C=C\\CCCCCCC1"),
        ("ambrettolide", "O=C1CCCCC/C=C\\CCCCCCCCO1"),
        # Declared chirality and declared E/Z together, which is where a polish that fixes one and
        # breaks the other shows up.
        (
            "rapamycin",
            "CO[C@@H]1C[C@H](C[C@@H](C)[C@@H]2CC(=O)[C@H](C)/C=C(C)/[C@@H](O)[C@@H](OC)C(=O)[C@H](C)C[C@H](C)/C=C/C=C/C=C(C)/[C@H](OC)C[C@H]3CC[C@H](C)[C@](O)(O3)C(=O)C(=O)N3CCCC[C@H]3C(=O)O2)CC[C@H]1O",
        ),
        (
            "avermectin_b1a",
            "CC[C@H](C)[C@@H]1[C@H](C=C[C@@]2(O1)C[C@@H]3C[C@@H](O2)C/C=C(/[C@H]([C@H](/C=C/C=C/4\\CO[C@H]5[C@@]4([C@@H](C=C([C@H]5O)C)C(=O)O3)O)C)O[C@H]6C[C@@H]([C@H]([C@@H](O6)C)O[C@H]7C[C@@H]([C@H]([C@@H](O7)C)O)OC)OC)\\C)C",
        ),
        (
            "rifampicin",
            "CO[C@H]1/C=C/O[C@@]2(C)Oc3c(C)c(O)c4c(O)c(/C=N/N5CCN(C)CC5)c(NC(=O)/C(C)=C\\C=C\\[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C)c(O)c4c3C2=O",
        ),
        ("mupirocin", "C[C@H]1O[C@H](C[C@H](O)[C@@H]1O)C[C@@H]1O[C@@H]1C[C@H](C)[C@@H](O)/C(C)=C/C(=O)OCCCCCCCCC(=O)O"),
        (
            "epothilone_b",
            "C[C@H]1CCC[C@@]2(C)O[C@H]2C[C@H](OC(=O)C[C@@H](O)C(C)(C)C(=O)[C@@H](C)[C@@H]1O)/C(C)=C/c1csc(C)n1",
        ),
        ("calcipotriol", "C[C@H](/C=C/[C@H](O)C1CC1)[C@H]1CC[C@H]2/C(=C\\C=C3/C[C@@H](O)C[C@H](O)C3=C)CCC[C@]12C"),
        ("ergocalciferol", "C[C@@H](/C=C/[C@H](C)C(C)C)[C@H]1CC[C@H]2/C(=C\\C=C3/C[C@@H](O)CCC3=C)CCC[C@]12C"),
        ("calcitriol", "C[C@H](CCCC(C)(C)O)[C@H]1CC[C@H]2/C(=C\\C=C3/C[C@@H](O)C[C@H](O)C3=C)CCC[C@]12C"),
        ("cholecalciferol", "C[C@H](CCCC(C)C)[C@H]1CC[C@H]2/C(=C\\C=C3/C[C@@H](O)CCC3=C)CCC[C@]12C"),
        ("brefeldin_a", "C[C@@H]1CC[C@H]2C[C@@H](O)C[C@H]2/C=C/[C@H](O)C/C=C/C(=O)O1"),
        ("dinoprost", "CCCCC[C@@H](O)/C=C/[C@H]1[C@@H](C/C=C\\CCCC(=O)O)[C@@H](O)C[C@@H]1O"),
        ("dinoprostone", "CCCCC[C@@H](O)/C=C/[C@H]1[C@@H](C/C=C\\CCCC(=O)O)C(=O)C[C@@H]1O"),
        ("latanoprost", "CC(C)OC(=O)CCC/C=C\\C[C@H]1[C@@H](O)C[C@@H](O)[C@@H]1CC[C@@H](O)CCc1ccccc1"),
        ("misoprostol", "CCCC[C@](C)(O)C/C=C/[C@H]1[C@H](O)CC(=O)[C@@H]1CCCCCCC(=O)OC"),
        ("leukotriene_b4", "CCCCC/C=C\\C[C@@H](O)/C=C/C=C/C=C\\[C@@H](O)CCCC(=O)O"),
        (
            "lutein",
            "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/[C@H]2C(C)=C[C@H](O)CC2(C)C)C(C)(C)C[C@H](O)C1",
        ),
        (
            "zeaxanthin",
            "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)C[C@@H](O)CC2(C)C)C(C)(C)C[C@H](O)C1",
        ),
        (
            "astaxanthin",
            "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)C(=O)[C@@H](O)CC2(C)C)C(=O)[C@@H](O)CC1(C)C",
        ),
        ("alpha_carotene", "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/[C@H]2C(C)=CCCC2(C)C)C(C)(C)CCC1"),
        ("chlorogenic_acid", "C1[C@H]([C@H]([C@@H](C[C@@]1(C(=O)O)O)OC(=O)/C=C/C2=CC(=C(C=C2)O)O)O)O"),
        ("cefotaxime", "CO/N=C(/c1csc(N)n1)C(=O)N[C@H]1[C@H]2SCC(COC(C)=O)=C(C(=O)O)N2C1=O"),
        ("clavulanic_acid", "OC/C=C1\\O[C@@H](C(=O)O)[C@H]2CC(=O)N12"),
        ("sphingosine", "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@@H](N)CO"),
        ("methyl_jasmonate", "CC/C=C\\C[C@@H]1[C@@H](CC(=O)OC)CCC1=O"),
        ("fluvastatin", "CC(C)n1c(/C=C/[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccc(F)cc2)c2ccccc21"),
        ("phylloquinone", "CC1=C(C/C=C(C)/CCC[C@H](C)CCC[C@H](C)CCCC(C)C)C(=O)c2ccccc2C1=O"),
        ("phytol", "CC(C)CCC[C@@H](C)CCC[C@@H](C)CCC/C(C)=C/CO"),
        ("abscisic_acid", "CC1=CC(=O)CC(C)(C)[C@]1(O)/C=C/C(C)=C\\C(=O)O"),
        ("zearalenone", "C[C@@H]1CCCC(=O)CCC/C=C/c2cc(O)cc(O)c2C(=O)O1"),
        ("rosmarinic_acid", "O=C(O)[C@@H](Cc1ccc(O)c(O)c1)OC(=O)/C=C/c1ccc(O)c(O)c1"),
    ]
    for name, smiles in bond_stereo:
        templates.append(
            template_t(
                axis="declared_bond_stereo",
                name=name,
                smiles_generator=lambda bound_smiles=smiles: [bound_smiles],
                target_code_path="bond-stereo extraction / post-polish E/Z re-read",
            )
        )

    # Axis 14: near-degenerate geometry — strained cages and crowded centres, where the embedding can start
    # two atoms nearly coincident and the minimizer walks into the singularity rather than out of it. The
    # refusal that catches it measures a separation rather than an energy, because a collapse drives the
    # buffered 14-7 term to an enormous but finite value.
    crowded = [
        ("bicyclobutane", "C1C2CC12"),
        ("spiropentane", "C1CC12CC2"),
        ("prismane", "C1C2C3C1C1C2C31"),
        ("tetrahedrane", "C12C3C1C23"),
        ("hexa_tert_butylbenzene", "CC(C)(C)c1c(C(C)(C)C)c(C(C)(C)C)c(C(C)(C)C)c(C(C)(C)C)c1C(C)(C)C"),
        ("ortho_crowded_biphenyl", "CC(C)(C)c1ccccc1-c1ccccc1C(C)(C)C"),
        ("propellane", "C1CC2(C1)CC2"),
    ]
    for name, smiles in crowded:
        templates.append(
            template_t(
                axis="near_degenerate_geometry",
                name=name,
                smiles_generator=lambda bound_smiles=smiles: [bound_smiles],
                target_code_path="post-polish atom-separation floor",
            )
        )

    return templates


def emit(templates: list[template_t], *, output_smi: Path, meta_json: Path) -> None:
    """Instantiate templates, dedupe by SMILES string, and write SMI + meta JSON."""

    # RDKit is required rather than optional. Eight test binaries read this corpus, and without
    # canonicalisation the rows are neither deduplicated nor checked, so a silent fallback ships a file
    # that looks like the real one and is not.
    try:
        from rdkit import Chem
    except ImportError:
        print("[error] RDKit is not importable, and an unvalidated corpus is worse than none.", file=sys.stderr)
        raise

    seen: set[str] = set()
    rows: list[tuple[str, str]] = []  # (canonical_smiles, template_name)
    refused: list[tuple[str, str]] = []  # (template_name, raw_smiles)

    for template in templates:
        for raw in template.smiles_generator():
            if not raw:
                continue
            # A template RDKit cannot parse is refused and named, never written through. Keeping the raw
            # string on a canonicalisation failure is how six unparseable rows reached the shipped corpus.
            try:
                canon = Chem.CanonSmiles(raw)
            except Exception:
                refused.append((f"{template.axis}:{template.name}", raw))
                continue
            if not canon or canon in seen:
                continue
            seen.add(canon)
            rows.append((canon, f"{template.axis}:{template.name}"))

    output_smi.parent.mkdir(parents=True, exist_ok=True)
    with output_smi.open("w") as stream:
        for canon, _ in rows:
            stream.write(f"{canon}\n")

    meta: dict = {
        "total_molecules": len(rows),
        "templates": [
            {
                "axis": template.axis,
                "name": template.name,
                "target_code_path": template.target_code_path,
                "multiplicity": template.multiplicity,
            }
            for template in templates
        ],
        "rows": [
            {"line_number": line_number, "smiles": smiles, "template": tag}
            for line_number, (smiles, tag) in enumerate(rows)
        ],
    }
    with meta_json.open("w") as stream:
        json.dump(meta, stream, indent=2)

    by_axis: dict[str, int] = {}
    for _, tag in rows:
        axis = tag.split(":", 1)[0]
        by_axis[axis] = by_axis.get(axis, 0) + 1
    spread = [f"       {axis:<30s} {by_axis[axis]:>6d}" for axis in sorted(by_axis)]

    # Summary to stderr so the caller sees the shape of what landed.
    print(
        "\n".join(
            [
                f"[ok] wrote {len(rows)} unique SMILES to {output_smi}",
                f"[ok] wrote provenance to {meta_json}",
                *spread,
            ]
        ),
        file=sys.stderr,
    )

    # Loud, because a template whose SMILES fails to parse otherwise contributes nothing at all and
    # its axis quietly shrinks.
    if refused:
        named = [f"[refused] {name}: RDKit cannot parse {raw[:70]}" for name, raw in refused]
        print(
            "\n".join([*named, f"[refused] {len(refused)} template(s) contributed nothing"]),
            file=sys.stderr,
        )


def main() -> int:
    """Write the corpus and its provenance, defaulting both paths beside this script."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=here / "adversarial.smi", help="Output SMILES file, one per line."
    )
    parser.add_argument("--meta", type=Path, default=here / "adversarial.meta.json", help="Output provenance JSON.")
    args = parser.parse_args()

    templates = build_templates()
    emit(templates, output_smi=args.output, meta_json=args.meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
