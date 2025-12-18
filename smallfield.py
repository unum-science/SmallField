"""
SmallField: Python wrapper for Mojo MMFF94 implementation.

This module provides a Python interface to the SmallField force field
implementation, allowing it to work with RDKit molecules.
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# MMFF94 Constants (matching Mojo implementation)
MMFF_BOND_PREFACTOR = 143.9325  # mdyn*A/rad^2 -> kcal/mol
MMFF_CUBIC_STRETCH = -2.0       # A^-1


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
    return 0.5 * MMFF_BOND_PREFACTOR * kb * dr2 * (1.0 + cs * dr + (7.0 / 12.0) * cs2 * dr2)


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
        r = np.sqrt(dx*dx + dy*dy + dz*dz)

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
        'rdkit_total_energy': rdkit_total_energy,
        'smallfield_bond_energy': smallfield_bond_energy,
        'note': 'RDKit total includes all MMFF terms; SmallField is bond-only'
    }


def is_mojo_available() -> bool:
    """Check if Mojo runtime is available."""
    try:
        result = subprocess.run(['mojo', '--version'], capture_output=True, text=True)
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
            energies.append(float('inf'))

    return {
        'smiles': smiles,
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_conformers': len(conf_ids),
        'bond_energies': energies,
        'min_energy': min(energies) if energies else None
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
    print(f"  SmallField bond energy:  {comparison['smallfield_bond_energy']:.4f} kcal/mol")
    print(f"  Note: {comparison['note']}")
