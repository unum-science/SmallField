"""RDKit helpers shared by the validators, so one fact is encoded once.

Imports RDKit lazily: a validator that only checks a schema must still run without it.
"""

from __future__ import annotations

from typing import Optional


def parent_with_hydrogens(smiles: str):
    """The molecule the pipeline actually embedded: largest fragment, then explicit hydrogens.

    `generate_conformers` embeds only the parent fragment, so a validator parsing the whole SMILES describes a
    different molecule — more atoms than the slot holds, the extra indices reading slot padding at the origin — and
    the defects it then reports are its own rather than the pipeline's.
    """
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:
        return None
    if len(Chem.GetMolFrags(molecule)) > 1:
        molecule = rdMolStandardize.LargestFragmentChooser().choose(molecule)
        Chem.SanitizeMol(molecule)
    return Chem.AddHs(molecule)
