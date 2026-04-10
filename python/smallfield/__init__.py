"""
SmallField: GPU-accelerated ETKDG + MMFF94 conformer generation.

Usage:
    >>> import smallfield
    >>> coords = smallfield.generate_conformers("CCO", num_conformers=10)
    >>> print(coords.shape)  # (K, N, 3)
"""

from typing import Optional
import numpy as np

__version__ = "0.2.0"

try:
    import _smallfield

    supports_cuda = _smallfield.supports_cuda

    EtkdgContext: Optional[type] = None
    try:
        EtkdgContext = _smallfield.EtkdgContext
    except AttributeError:
        pass

except ImportError as e:
    raise ImportError(
        "Failed to import _smallfield extension module. "
        "Please build the extension with: pip install -e ."
    ) from e


def has_cuda() -> bool:
    """Check if CUDA support is available."""
    return supports_cuda()


def generate_conformers(smiles: str, num_conformers: int = 10,
                        max_stereo_attempts: int = 20,
                        seed: int = 42) -> Optional[np.ndarray]:
    """Generate 3D conformers for a molecule using GPU ETKDG pipeline.

    Returns numpy array of shape (K, N, 3) or None if GPU unavailable.
    Requires RDKit for molecular preprocessing.
    """
    if EtkdgContext is None:
        return None

    try:
        from rdkit import Chem
        from rdkit.Chem import rdDistGeom
    except ImportError:
        raise ImportError("RDKit is required for ETKDG conformer generation")

    import struct

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    num_atoms = mol.GetNumAtoms()

    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    bounds_flat = bounds.astype(np.float32).flatten()

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)

    chirals = []
    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag not in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                       Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            continue
        if not atom.HasProp("_CIPCode"):
            continue
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 3:
            continue
        neighbors.sort(key=lambda idx: (-int(mol.GetAtomWithIdx(idx).GetProp("_CIPRank")), idx))
        target_sign = 1 if atom.GetProp("_CIPCode") == "S" else -1
        descriptor = struct.pack("BBBBBbBB",
            atom.GetIdx(),
            neighbors[0], neighbors[1] if len(neighbors) > 1 else 0,
            neighbors[2] if len(neighbors) > 2 else 0,
            neighbors[3] if len(neighbors) > 3 else 0,
            target_sign, 0, 0)
        chirals.append(descriptor)

    chirals_bytes = b"".join(chirals) if chirals else b""

    torsions = []
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1.5:
            continue
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        i_atom = next((n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors()
                       if n.GetIdx() != k), None)
        l_atom = next((n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors()
                       if n.GetIdx() != j), None)
        if i_atom is None or l_atom is None:
            continue
        descriptor = struct.pack("BBBBBBBB6f",
            i_atom, j, k, l_atom, 0, 0, 0, 0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        torsions.append(descriptor)

    torsions_bytes = b"".join(torsions) if torsions else b""

    constraints = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        descriptor = struct.pack("BBBBfff", i, j, 0, 0, 1.425, 1.575, 10.0)
        constraints.append(descriptor)

    constraints_bytes = b"".join(constraints) if constraints else b""

    metadata = struct.pack("IIIIIIII",
        num_atoms, len(chirals), len(torsions), len(constraints),
        0, 0, 0, 0)

    ctx = EtkdgContext(num_atoms, 1, num_conformers)
    coords_flat, energies, status = ctx.generate_conformers(
        bounds_flat,
        np.frombuffer(chirals_bytes, dtype=np.uint8) if chirals_bytes else np.array([], dtype=np.uint8),
        np.frombuffer(torsions_bytes, dtype=np.uint8) if torsions_bytes else np.array([], dtype=np.uint8),
        np.frombuffer(constraints_bytes, dtype=np.uint8) if constraints_bytes else np.array([], dtype=np.uint8),
        np.frombuffer(metadata, dtype=np.uint8),
        num_conformers, max_stereo_attempts, seed)

    coords_array = np.array(coords_flat).reshape(num_conformers, num_atoms * 3)
    status_array = np.array(status)

    successful = []
    for k_idx in range(num_conformers):
        if status_array[k_idx] == 0:
            soa = coords_array[k_idx]
            xyz = np.zeros((num_atoms, 3), dtype=np.float32)
            xyz[:, 0] = soa[0:num_atoms]
            xyz[:, 1] = soa[num_atoms:2*num_atoms]
            xyz[:, 2] = soa[2*num_atoms:3*num_atoms]
            successful.append(xyz)

    return np.stack(successful) if successful else None


__all__ = ["supports_cuda", "has_cuda", "EtkdgContext", "generate_conformers"]
