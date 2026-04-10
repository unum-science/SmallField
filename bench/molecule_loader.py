"""
Molecule loader for unified SmallField benchmark.

Loads molecules from USearchMolecules SMILES file and extracts all MMFF94 parameters
using RDKit. Returns pre-parameterized MoleculeData objects ready for benchmarking.
"""

import os
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class BondData:
    """MMFF94 bond parameters."""

    i: int
    j: int
    kb: float  # Force constant (mdyn/A)
    r0: float  # Equilibrium distance (A)


@dataclass
class AngleData:
    """MMFF94 angle parameters."""

    i: int
    j: int  # Central atom
    k: int
    ka: float  # Force constant (mdyn*A/rad^2)
    theta0: float  # Equilibrium angle (radians)


@dataclass
class TorsionData:
    """MMFF94 torsion parameters."""

    i: int
    j: int
    k: int
    l: int
    v1: float  # Barrier heights (kcal/mol)
    v2: float
    v3: float


@dataclass
class NonbondedPair:
    """MMFF94 nonbonded pair (vdW + electrostatic)."""

    i: int
    j: int
    is_1_4: bool  # True if 1-4 interaction (electrostatic scaling)
    epsilon: float  # vdW well depth (kcal/mol)
    r_star: float  # vdW minimum distance (A)
    qi: float  # Partial charge on atom i
    qj: float  # Partial charge on atom j


@dataclass
class MoleculeData:
    """Complete MMFF94 parameterized molecule."""

    smiles: str
    num_atoms: int
    positions: np.ndarray  # Shape (num_atoms, 3), dtype float64
    charges: np.ndarray  # Shape (num_atoms,), dtype float64
    bonds: List[BondData] = field(default_factory=list)
    angles: List[AngleData] = field(default_factory=list)
    torsions: List[TorsionData] = field(default_factory=list)
    nonbonded: List[NonbondedPair] = field(default_factory=list)


def load_smiles_file(path: str, max_read: int = 20000) -> List[str]:
    """Load SMILES strings from file."""
    smiles_list = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                # Take first component (before any '.')
                smi = parts[0].split(".")[0]
                if smi and 5 <= len(smi) <= 80:
                    smiles_list.append(smi)
            if len(smiles_list) >= max_read:
                break
    return smiles_list


def parameterize_molecule(smiles: str, seed: int = 42) -> Optional[MoleculeData]:
    """
    Convert SMILES to fully parameterized MoleculeData.

    Returns None if molecule cannot be parameterized (invalid SMILES,
    embedding failure, or MMFF94 unsupported).
    """
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add hydrogens
    mol = Chem.AddHs(mol)
    num_atoms = mol.GetNumAtoms()

    # Skip very large molecules
    if num_atoms > 100:
        return None

    # Generate 3D coordinates
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) == -1:
        return None

    # Get MMFF94 properties
    mmff = AllChem.MMFFGetMoleculeProperties(mol)
    if mmff is None:
        return None

    # Verify force field can be constructed
    ff = AllChem.MMFFGetMoleculeForceField(mol, mmff)
    if ff is None:
        return None

    # Extract positions
    conf = mol.GetConformer()
    positions = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(num_atoms)
        ],
        dtype=np.float64,
    )

    # Extract charges
    charges = np.array(
        [mmff.GetMMFFPartialCharge(i) for i in range(num_atoms)], dtype=np.float64
    )

    # Extract bond parameters
    bonds = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        params = mmff.GetMMFFBondStretchParams(mol, i, j)
        if params:
            _, kb, r0 = params
            bonds.append(BondData(i=i, j=j, kb=kb, r0=r0))

    # Extract angle parameters
    angles = []
    angles_seen = set()
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        for idx1, i in enumerate(neighbors):
            for k in neighbors[idx1 + 1 :]:
                key = (min(i, k), j, max(i, k))
                if key not in angles_seen:
                    angles_seen.add(key)
                    params = mmff.GetMMFFAngleBendParams(mol, i, j, k)
                    if params:
                        _, ka, theta0_deg = params
                        theta0_rad = theta0_deg * np.pi / 180.0
                        angles.append(
                            AngleData(i=i, j=j, k=k, ka=ka, theta0=theta0_rad)
                        )

    # Extract torsion parameters
    torsions = []
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        neighbors_j = [
            n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k
        ]
        neighbors_k = [
            n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j
        ]

        for i in neighbors_j:
            for l in neighbors_k:
                if i != l:
                    params = mmff.GetMMFFTorsionParams(mol, i, j, k, l)
                    if params:
                        _, v1, v2, v3 = params
                        torsions.append(
                            TorsionData(i=i, j=j, k=k, l=l, v1=v1, v2=v2, v3=v3)
                        )

    # Build exclusion sets for nonbonded
    bonded_1_2 = {i: set() for i in range(num_atoms)}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded_1_2[i].add(j)
        bonded_1_2[j].add(i)

    bonded_1_3 = {i: set() for i in range(num_atoms)}
    for i in range(num_atoms):
        for j in bonded_1_2[i]:
            for k in bonded_1_2[j]:
                if k != i:
                    bonded_1_3[i].add(k)

    bonded_1_4 = {i: set() for i in range(num_atoms)}
    for i in range(num_atoms):
        for j in bonded_1_3[i]:
            for k in bonded_1_2[j]:
                if k != i and k not in bonded_1_2[i] and k not in bonded_1_3[i]:
                    bonded_1_4[i].add(k)

    # Extract nonbonded pairs
    nonbonded = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            # Skip 1-2 and 1-3 interactions
            if j in bonded_1_2[i] or j in bonded_1_3[i]:
                continue

            # Get vdW parameters
            vdw_params = mmff.GetMMFFVdWParams(i, j)
            if vdw_params:
                r_star, epsilon = vdw_params[0], vdw_params[1]
                is_1_4 = j in bonded_1_4[i]

                nonbonded.append(
                    NonbondedPair(
                        i=i,
                        j=j,
                        is_1_4=is_1_4,
                        epsilon=epsilon,
                        r_star=r_star,
                        qi=charges[i],
                        qj=charges[j],
                    )
                )

    return MoleculeData(
        smiles=smiles,
        num_atoms=num_atoms,
        positions=positions,
        charges=charges,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        nonbonded=nonbonded,
    )


def load_molecules(
    smiles_path: str, n_samples: int = 1000, seed: int = 42, verbose: bool = True
) -> List[MoleculeData]:
    """
    Load and parameterize molecules from USearchMolecules SMILES file.

    Args:
        smiles_path: Path to SMILES file
        n_samples: Number of molecules to load
        seed: Random seed for sampling and conformer generation
        verbose: Print progress messages

    Returns:
        List of parameterized MoleculeData objects
    """
    random.seed(seed)
    np.random.seed(seed)

    if verbose:
        print(f"Loading SMILES from {smiles_path}...")

    all_smiles = load_smiles_file(smiles_path)
    if verbose:
        print(f"  Found {len(all_smiles)} candidate SMILES")

    # Sample more than needed since some will fail
    sample_size = min(n_samples * 3, len(all_smiles))
    sample = random.sample(all_smiles, sample_size)

    molecules = []
    failed = 0

    for smi in sample:
        if len(molecules) >= n_samples:
            break

        mol_data = parameterize_molecule(smi, seed=seed)
        if mol_data is not None:
            molecules.append(mol_data)
        else:
            failed += 1

    if verbose:
        print(f"  Successfully parameterized {len(molecules)} molecules")
        print(f"  Failed: {failed}")

        # Statistics
        atom_counts = [m.num_atoms for m in molecules]
        bond_counts = [len(m.bonds) for m in molecules]
        nb_counts = [len(m.nonbonded) for m in molecules]

        print(
            f"  Atom counts: min={min(atom_counts)}, max={max(atom_counts)}, avg={np.mean(atom_counts):.1f}"
        )
        print(
            f"  Bonds per mol: min={min(bond_counts)}, max={max(bond_counts)}, avg={np.mean(bond_counts):.1f}"
        )
        print(f"  NB pairs per mol: avg={np.mean(nb_counts):.1f}")

    return molecules


# Default data path
DEFAULT_SMILES_PATH = os.environ.get("SMALLFIELD_SMI_CORPUS", "")


if __name__ == "__main__":
    # Quick test
    molecules = load_molecules(DEFAULT_SMILES_PATH, n_samples=10)

    print("\nSample molecule details:")
    mol = molecules[0]
    print(f"  SMILES: {mol.smiles}")
    print(f"  Atoms: {mol.num_atoms}")
    print(f"  Bonds: {len(mol.bonds)}")
    print(f"  Angles: {len(mol.angles)}")
    print(f"  Torsions: {len(mol.torsions)}")
    print(f"  Nonbonded pairs: {len(mol.nonbonded)}")
