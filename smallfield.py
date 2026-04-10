"""
@file smallfield.py
@brief Python wrapper for SmallField MMFF94 force field implementation.
@author Ash Vardanian
@date 2025-12-20

This module provides a Python interface to the SmallField force field
implementation, allowing it to work with RDKit molecules.
"""

import struct
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple, List, Any, Union
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True
    Mol = Chem.Mol
except ImportError:
    RDKIT_AVAILABLE = False
    Mol = Any


# MMFF94 Constants (matching Mojo implementation)
MMFF_BOND_PREFACTOR = 143.9325  # mdyn*A/rad^2 -> kcal/mol
MMFF_CUBIC_STRETCH = -2.0  # A^-1


def bond_energy_single(r: float, r0: float, kb: float) -> float:
    """
    MMFF94 bond stretching energy for a single bond.

    E = 0.5 * 143.9325 * kb * dr^2 * (1 + cs*dr + 7/12*cs^2*dr^2)

    Args:
        r: Current bond length (Angstroms)
        r0: Equilibrium bond length (Angstroms)
        kb: Force constant (mdyn/A)

    Returns:
        Energy in kcal/mol
    """
    dr = r - r0
    dr2 = dr * dr
    cs = MMFF_CUBIC_STRETCH
    cs2 = cs * cs
    return (
        0.5
        * MMFF_BOND_PREFACTOR
        * kb
        * dr2
        * (1.0 + cs * dr + (7.0 / 12.0) * cs2 * dr2)
    )


def compute_mmff_bond_energy(mol, conf_id: int = -1) -> Tuple[float, List[float]]:
    """
    Compute MMFF94 bond stretching energy for an RDKit molecule.

    Args:
        mol: RDKit molecule with conformer
        conf_id: Conformer ID (-1 for first conformer)

    Returns:
        Tuple of (total_energy, list of per-bond energies)
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecular calculations")

    # Get MMFF properties
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    if mmff_props is None:
        raise ValueError("Could not get MMFF properties for molecule")

    conf = mol.GetConformer(conf_id)
    total_energy = 0.0
    bond_energies = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Get positions
        pos_i = conf.GetAtomPosition(i)
        pos_j = conf.GetAtomPosition(j)

        # Compute distance
        dx = pos_j.x - pos_i.x
        dy = pos_j.y - pos_i.y
        dz = pos_j.z - pos_i.z
        r = np.sqrt(dx * dx + dy * dy + dz * dz)

        # Get MMFF parameters
        stretch_params = mmff_props.GetMMFFBondStretchParams(mol, i, j)
        if stretch_params is not None:
            kb = stretch_params[1]
            r0 = stretch_params[2]
        else:
            # Default values
            kb = 4.0
            r0 = 1.5

        # Compute energy
        e = bond_energy_single(r, r0, kb)
        bond_energies.append(e)
        total_energy += e

    return total_energy, bond_energies


def compare_with_rdkit(mol, conf_id: int = -1) -> dict:
    """
    Compare SmallField bond energy calculation with RDKit.

    Args:
        mol: RDKit molecule with conformer
        conf_id: Conformer ID

    Returns:
        Dictionary with comparison results
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required")

    # Get RDKit total MMFF energy
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
    rdkit_total_energy = ff.CalcEnergy()

    # Get SmallField bond energy
    smallfield_bond_energy, _ = compute_mmff_bond_energy(mol, conf_id)

    return {
        "rdkit_total_energy": rdkit_total_energy,
        "smallfield_bond_energy": smallfield_bond_energy,
        "note": "RDKit total includes all MMFF terms; SmallField is bond-only",
    }


def is_mojo_available() -> bool:
    """Check if Mojo runtime is available."""
    try:
        result = subprocess.run(["mojo", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


class SmallFieldForceField:
    """
    SmallField MMFF94 force field wrapper.

    Provides an interface similar to RDKit's force field objects.
    """

    def __init__(self, mol, mmff_variant: str = "MMFF94"):
        """
        Initialize force field for a molecule.

        Args:
            mol: RDKit molecule
            mmff_variant: MMFF variant ("MMFF94" or "MMFF94s")
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required")

        self.mol = mol
        self.mmff_variant = mmff_variant
        self.mmff_props = AllChem.MMFFGetMoleculeProperties(mol)

        if self.mmff_props is None:
            raise ValueError("Molecule is not compatible with MMFF")

    def calc_energy(self, conf_id: int = -1) -> float:
        """
        Calculate total MMFF energy.

        Currently only implements bond stretching.
        """
        bond_energy, _ = compute_mmff_bond_energy(self.mol, conf_id)
        # TODO: Add angle, torsion, vdW, electrostatic terms
        return bond_energy

    def calc_bond_energy(self, conf_id: int = -1) -> float:
        """Calculate bond stretching energy only."""
        energy, _ = compute_mmff_bond_energy(self.mol, conf_id)
        return energy


def process_molecule(smiles: str, num_conformers: int = 10) -> Optional[dict]:
    """
    Generate conformers for a molecule using SmallField.

    Args:
        smiles: SMILES string
        num_conformers: Number of conformers to generate

    Returns:
        Dictionary with molecule data or None if failed
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Use RDKit for embedding (ETKDG)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1

    conf_ids = AllChem.EmbedMultipleConfs(mol, num_conformers, params)

    if len(conf_ids) == 0:
        return None

    # Compute energies with SmallField
    energies = []
    for conf_id in conf_ids:
        try:
            bond_energy = compute_mmff_bond_energy(mol, conf_id)[0]
            energies.append(bond_energy)
        except Exception as e:
            energies.append(float("inf"))

    return {
        "smiles": smiles,
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "num_conformers": len(conf_ids),
        "bond_energies": energies,
        "min_energy": min(energies) if energies else None,
    }


if __name__ == "__main__":
    print("SmallField Python Module")
    print("========================")

    if not RDKIT_AVAILABLE:
        print("RDKit not available. Install with: pip install rdkit")
        exit(1)

    # Test with ethanol
    smiles = "CCO"
    print(f"\nTesting with: {smiles}")

    result = process_molecule(smiles, num_conformers=5)
    if result:
        print(f"  Atoms: {result['num_atoms']}")
        print(f"  Bonds: {result['num_bonds']}")
        print(f"  Conformers: {result['num_conformers']}")
        print(f"  Bond energies: {result['bond_energies']}")
        print(f"  Min energy: {result['min_energy']:.4f} kcal/mol")

    # Compare with RDKit
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

    comparison = compare_with_rdkit(mol)
    print(f"\nComparison with RDKit:")
    print(f"  RDKit total MMFF energy: {comparison['rdkit_total_energy']:.4f} kcal/mol")
    print(
        f"  SmallField bond energy:  {comparison['smallfield_bond_energy']:.4f} kcal/mol"
    )
    print(f"  Note: {comparison['note']}")


def _prepare_molecule_for_gpu(smiles: str):
    """Prepare a single molecule for GPU ETKDG pipeline.

    Returns bounds matrix, chiral descriptors, torsion descriptors,
    distance constraints, and metadata as numpy arrays.
    Requires RDKit.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for ETKDG conformer generation")

    from rdkit.Chem import rdDistGeom

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    num_atoms = mol.GetNumAtoms()

    # bounds matrix (already smoothed by RDKit)
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    # RDKit returns upper triangle as upper bounds, lower triangle as lower bounds
    bounds_flat = bounds.astype(np.float32).flatten()

    # embed to get 3D geometry for chiral sign computation
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

    # chiral centers: 8 bytes each (center, 4 neighbors, sign, 2 padding)
    chirals = []
    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag not in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                       Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            continue
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 3:
            continue
        # compute signed volume from embedded geometry to get target sign
        conf = mol.GetConformer()
        ra = conf.GetAtomPosition(atom.GetIdx())
        rb = conf.GetAtomPosition(neighbors[0])
        rc = conf.GetAtomPosition(neighbors[1])
        rd = conf.GetAtomPosition(neighbors[2])
        bx, by, bz = rb.x - ra.x, rb.y - ra.y, rb.z - ra.z
        cx, cy, cz = rc.x - ra.x, rc.y - ra.y, rc.z - ra.z
        dx, dy, dz = rd.x - ra.x, rd.y - ra.y, rd.z - ra.z
        volume = bx * (cy*dz - cz*dy) + by * (cz*dx - cx*dz) + bz * (cx*dy - cy*dx)
        target_sign = 1 if volume > 0 else -1

        descriptor = struct.pack("BBBBBbBB",
            atom.GetIdx(),
            neighbors[0] if len(neighbors) > 0 else 0,
            neighbors[1] if len(neighbors) > 1 else 0,
            neighbors[2] if len(neighbors) > 2 else 0,
            neighbors[3] if len(neighbors) > 3 else 0,
            target_sign, 0, 0)
        chirals.append(descriptor)

    chirals_bytes = b"".join(chirals)

    # torsion terms: 32 bytes each (4 atom indices, 4 padding, 6 floats)
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
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0)  # default V3=1.0
        torsions.append(descriptor)

    torsions_bytes = b"".join(torsions)

    # distance constraints: 16 bytes each
    constraints = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        descriptor = struct.pack("BBBBfff",
            i, j, 0, 0, 1.425, 1.575, 10.0)  # 1-2 bond, ~1.5A
        constraints.append(descriptor)

    constraints_bytes = b"".join(constraints)

    # metadata: 32 bytes
    metadata = struct.pack("IIIIIIII",
        num_atoms,
        len(chirals),
        len(torsions),
        len(constraints),
        0,  # bounds_offset (single molecule)
        0,  # chiral_offset
        0,  # torsion_offset
        0)  # constraint_offset

    return {
        "num_atoms": num_atoms,
        "bounds": bounds_flat,
        "chirals": np.frombuffer(chirals_bytes, dtype=np.uint8) if chirals_bytes else np.array([], dtype=np.uint8),
        "torsions": np.frombuffer(torsions_bytes, dtype=np.uint8) if torsions_bytes else np.array([], dtype=np.uint8),
        "constraints": np.frombuffer(constraints_bytes, dtype=np.uint8) if constraints_bytes else np.array([], dtype=np.uint8),
        "metadata": np.frombuffer(metadata, dtype=np.uint8),
    }


def generate_conformers(smiles: str, num_conformers: int = 10,
                        max_stereo_attempts: int = 20,
                        seed: int = 42) -> Optional[np.ndarray]:
    """Generate 3D conformers for a molecule using GPU ETKDG pipeline.

    Args:
        smiles: SMILES string for the molecule.
        num_conformers: Number of conformers to generate.
        max_stereo_attempts: Maximum retry attempts per conformer on stereo failure.
        seed: Random seed for reproducibility.

    Returns:
        numpy array of shape (K, N, 3) with 3D coordinates for successful conformers,
        or None if the GPU pipeline is not available.
    """
    import smallfield as sf
    if sf.EtkdgContext is None:
        return None

    prepared = _prepare_molecule_for_gpu(smiles)
    num_atoms = prepared["num_atoms"]

    ctx = sf.EtkdgContext(num_atoms, 1, num_conformers)
    coords, energies, status = ctx.generate_conformers(
        prepared["bounds"],
        prepared["chirals"],
        prepared["torsions"],
        prepared["constraints"],
        prepared["metadata"],
        num_conformers,
        max_stereo_attempts,
        seed)

    # extract successful conformers, convert from SoA to (K, N, 3)
    coords_array = np.array(coords).reshape(num_conformers, num_atoms * 3)
    status_array = np.array(status)

    successful = []
    for k in range(num_conformers):
        if status_array[k] == 0:
            soa = coords_array[k]  # [x0..xN, y0..yN, z0..zN]
            xyz = np.zeros((num_atoms, 3), dtype=np.float32)
            xyz[:, 0] = soa[0:num_atoms]
            xyz[:, 1] = soa[num_atoms:2*num_atoms]
            xyz[:, 2] = soa[2*num_atoms:3*num_atoms]
            successful.append(xyz)

    if not successful:
        return None

    return np.stack(successful)
