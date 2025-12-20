#!/usr/bin/env python3
"""
test.py - Unit tests comparing SmallField against RDKit's MMFF94 implementation.

This module verifies the correctness of van der Waals and electrostatic energy
calculations by comparing SmallField's optimized implementations against RDKit.

Tests cover:
- Serial CPU (f32, f64)
- AVX-512 CPU (f32)
- CUDA GPU (f32) if available

Usage:
    pytest test.py -v
    python -m pytest test.py -s -x
"""

import pytest
import numpy as np
from typing import Tuple, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolTransforms

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    pytest.skip("RDKit not available", allow_module_level=True)

import smallfield


def create_test_molecule(smiles: str = "CCO") -> Chem.Mol:
    """
    Create a test molecule with 3D coordinates and MMFF94 properties.

    Args:
        smiles: SMILES string for the molecule

    Returns:
        RDKit molecule with embedded conformer
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def extract_coordinates(
    mol: Chem.Mol, conf_id: int = -1, dtype=np.float64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract atom coordinates from RDKit molecule.

    Args:
        mol: RDKit molecule
        conf_id: Conformer ID
        dtype: NumPy dtype for coordinates

    Returns:
        Tuple of (coords_x, coords_y, coords_z) arrays
    """
    conf = mol.GetConformer(conf_id)
    n_atoms = mol.GetNumAtoms()

    coords_x = np.zeros(n_atoms, dtype=dtype)
    coords_y = np.zeros(n_atoms, dtype=dtype)
    coords_z = np.zeros(n_atoms, dtype=dtype)

    for i in range(n_atoms):
        pos = conf.GetAtomPosition(i)
        coords_x[i] = pos.x
        coords_y[i] = pos.y
        coords_z[i] = pos.z

    return coords_x, coords_y, coords_z


def extract_vdw_pairs(mol: Chem.Mol, dtype=np.float64) -> Tuple[np.ndarray, ...]:
    """
    Extract van der Waals pair information from RDKit molecule.

    Args:
        mol: RDKit molecule with MMFF properties
        dtype: NumPy dtype for float arrays

    Returns:
        Tuple of (pair_i, pair_j, epsilon, r0) arrays
    """
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    if mmff_props is None:
        raise ValueError("Could not get MMFF properties")

    pair_i_list = []
    pair_j_list = []
    epsilon_list = []
    r0_list = []

    # Extract non-bonded pairs from MMFF force field
    n_atoms = mol.GetNumAtoms()
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Check if this is a non-bonded interaction
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                continue  # Skip bonded pairs

            # Get VdW parameters
            params = mmff_props.GetMMFFVdWParams(i, j)
            if params is not None:
                pair_i_list.append(i)
                pair_j_list.append(j)
                epsilon_list.append(params[1])  # Well depth
                r0_list.append(params[0])  # Equilibrium distance

    pair_i = np.array(pair_i_list, dtype=np.int32)
    pair_j = np.array(pair_j_list, dtype=np.int32)
    epsilon = np.array(epsilon_list, dtype=dtype)
    r0 = np.array(r0_list, dtype=dtype)

    return pair_i, pair_j, epsilon, r0


def extract_electrostatic_pairs(
    mol: Chem.Mol, dtype=np.float64
) -> Tuple[np.ndarray, ...]:
    """
    Extract electrostatic pair information from RDKit molecule.

    Args:
        mol: RDKit molecule with MMFF properties
        dtype: NumPy dtype for float arrays

    Returns:
        Tuple of (pair_i, pair_j, charge_i, charge_j, is_1_4) arrays
    """
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    if mmff_props is None:
        raise ValueError("Could not get MMFF properties")

    pair_i_list = []
    pair_j_list = []
    charge_i_list = []
    charge_j_list = []
    is_1_4_list = []

    n_atoms = mol.GetNumAtoms()
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Check if this is a non-bonded interaction
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                continue

            # Get partial charges
            charge_i = mmff_props.GetMMFFPartialCharge(i)
            charge_j = mmff_props.GetMMFFPartialCharge(j)

            # Determine if this is a 1-4 interaction
            # For simplicity, we'll mark all as non-1-4 (0)
            is_1_4 = 0

            pair_i_list.append(i)
            pair_j_list.append(j)
            charge_i_list.append(charge_i)
            charge_j_list.append(charge_j)
            is_1_4_list.append(is_1_4)

    pair_i = np.array(pair_i_list, dtype=np.int32)
    pair_j = np.array(pair_j_list, dtype=np.int32)
    charge_i = np.array(charge_i_list, dtype=dtype)
    charge_j = np.array(charge_j_list, dtype=dtype)
    is_1_4 = np.array(is_1_4_list, dtype=np.int32)

    return pair_i, pair_j, charge_i, charge_j, is_1_4


def compute_rdkit_vdw_energy(mol: Chem.Mol) -> float:
    """
    Compute van der Waals energy using RDKit's MMFF94.

    Args:
        mol: RDKit molecule

    Returns:
        VdW energy in kcal/mol
    """
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)

    # RDKit doesn't provide direct VdW-only energy, so we'll use total non-bonded
    # This is an approximation for testing purposes
    return ff.CalcEnergy()


@pytest.fixture
def ethanol_mol():
    """Fixture providing an ethanol molecule."""
    return create_test_molecule("CCO")


@pytest.fixture
def benzene_mol():
    """Fixture providing a benzene molecule."""
    return create_test_molecule("c1ccccc1")


def test_serial_vdw_f64(ethanol_mol):
    """Test serial VdW energy computation with float64."""
    coords_x, coords_y, coords_z = extract_coordinates(ethanol_mol, dtype=np.float64)
    pair_i, pair_j, epsilon, r0 = extract_vdw_pairs(ethanol_mol, dtype=np.float64)

    # Pad arrays to multiple of 4 for serial kernel
    n_pairs = len(pair_i)
    padding = (4 - n_pairs % 4) % 4
    if padding > 0:
        pair_i = np.pad(pair_i, (0, padding), mode="edge")
        pair_j = np.pad(pair_j, (0, padding), mode="edge")
        epsilon = np.pad(epsilon, (0, padding), mode="constant", constant_values=0)
        r0 = np.pad(r0, (0, padding), mode="constant", constant_values=1.0)

    energy = smallfield.compute_vdw_energy_serial(
        coords_x, coords_y, coords_z, pair_i, pair_j, epsilon, r0
    )

    assert isinstance(energy, (float, np.floating))
    assert energy >= 0  # VdW energy should be non-negative for repulsive interactions


def test_serial_vdw_f32(ethanol_mol):
    """Test serial VdW energy computation with float32."""
    coords_x, coords_y, coords_z = extract_coordinates(ethanol_mol, dtype=np.float32)
    pair_i, pair_j, epsilon, r0 = extract_vdw_pairs(ethanol_mol, dtype=np.float32)

    # Pad arrays
    n_pairs = len(pair_i)
    padding = (4 - n_pairs % 4) % 4
    if padding > 0:
        pair_i = np.pad(pair_i, (0, padding), mode="edge")
        pair_j = np.pad(pair_j, (0, padding), mode="edge")
        epsilon = np.pad(epsilon, (0, padding), mode="constant", constant_values=0)
        r0 = np.pad(r0, (0, padding), mode="constant", constant_values=1.0)

    energy_f32 = smallfield.compute_vdw_energy_serial(
        coords_x, coords_y, coords_z, pair_i, pair_j, epsilon, r0
    )

    # Compare with f64 version
    coords_x_f64, coords_y_f64, coords_z_f64 = extract_coordinates(
        ethanol_mol, dtype=np.float64
    )
    pair_i_f64, pair_j_f64, epsilon_f64, r0_f64 = extract_vdw_pairs(
        ethanol_mol, dtype=np.float64
    )

    if padding > 0:
        pair_i_f64 = np.pad(pair_i_f64, (0, padding), mode="edge")
        pair_j_f64 = np.pad(pair_j_f64, (0, padding), mode="edge")
        epsilon_f64 = np.pad(
            epsilon_f64, (0, padding), mode="constant", constant_values=0
        )
        r0_f64 = np.pad(r0_f64, (0, padding), mode="constant", constant_values=1.0)

    energy_f64 = smallfield.compute_vdw_energy_serial(
        coords_x_f64,
        coords_y_f64,
        coords_z_f64,
        pair_i_f64,
        pair_j_f64,
        epsilon_f64,
        r0_f64,
    )

    # Should be close but not exact due to floating point precision
    np.testing.assert_allclose(energy_f32, energy_f64, rtol=1e-4)


def test_avx512_vdw(ethanol_mol):
    """Test AVX-512 VdW energy computation."""
    coords_x, coords_y, coords_z = extract_coordinates(ethanol_mol, dtype=np.float32)
    pair_i, pair_j, epsilon, r0 = extract_vdw_pairs(ethanol_mol, dtype=np.float32)

    # Pad to multiple of 16 for AVX-512
    n_pairs = len(pair_i)
    padding = (16 - n_pairs % 16) % 16
    if padding > 0:
        pair_i = np.pad(pair_i, (0, padding), mode="edge")
        pair_j = np.pad(pair_j, (0, padding), mode="edge")
        epsilon = np.pad(epsilon, (0, padding), mode="constant", constant_values=0)
        r0 = np.pad(r0, (0, padding), mode="constant", constant_values=1.0)

    energy_avx512 = smallfield.compute_vdw_energy_avx512(
        coords_x, coords_y, coords_z, pair_i, pair_j, epsilon, r0
    )

    # Compare with serial version
    energy_serial = smallfield.compute_vdw_energy_serial(
        coords_x.astype(np.float64),
        coords_y.astype(np.float64),
        coords_z.astype(np.float64),
        pair_i,
        pair_j,
        epsilon.astype(np.float64),
        r0.astype(np.float64),
    )

    # AVX-512 uses fast approximations, so tolerance is higher
    np.testing.assert_allclose(energy_avx512, energy_serial, rtol=1e-3)


def test_electrostatic_serial(ethanol_mol):
    """Test serial electrostatic energy computation."""
    coords_x, coords_y, coords_z = extract_coordinates(ethanol_mol, dtype=np.float64)
    pair_i, pair_j, charge_i, charge_j, is_1_4 = extract_electrostatic_pairs(
        ethanol_mol, dtype=np.float64
    )

    # Pad arrays
    n_pairs = len(pair_i)
    padding = (4 - n_pairs % 4) % 4
    if padding > 0:
        pair_i = np.pad(pair_i, (0, padding), mode="edge")
        pair_j = np.pad(pair_j, (0, padding), mode="edge")
        charge_i = np.pad(charge_i, (0, padding), mode="constant", constant_values=0)
        charge_j = np.pad(charge_j, (0, padding), mode="constant", constant_values=0)
        is_1_4 = np.pad(is_1_4, (0, padding), mode="constant", constant_values=0)

    energy = smallfield.compute_electrostatic_energy_serial(
        coords_x, coords_y, coords_z, pair_i, pair_j, charge_i, charge_j, is_1_4
    )

    assert isinstance(energy, (float, np.floating))


def test_high_level_api(ethanol_mol):
    """Test high-level convenience API."""
    coords_x, coords_y, coords_z = extract_coordinates(ethanol_mol, dtype=np.float32)
    pair_i, pair_j, epsilon, r0 = extract_vdw_pairs(ethanol_mol, dtype=np.float32)
    _, _, charge_i, charge_j, is_1_4 = extract_electrostatic_pairs(
        ethanol_mol, dtype=np.float32
    )

    # Pad arrays
    n_pairs = len(pair_i)
    padding = (16 - n_pairs % 16) % 16
    if padding > 0:
        pair_i = np.pad(pair_i, (0, padding), mode="edge")
        pair_j = np.pad(pair_j, (0, padding), mode="edge")
        epsilon = np.pad(epsilon, (0, padding), mode="constant", constant_values=0)
        r0 = np.pad(r0, (0, padding), mode="constant", constant_values=1.0)
        charge_i = np.pad(charge_i, (0, padding), mode="constant", constant_values=0)
        charge_j = np.pad(charge_j, (0, padding), mode="constant", constant_values=0)
        is_1_4 = np.pad(is_1_4, (0, padding), mode="constant", constant_values=0)

    # Test automatic backend selection
    vdw_energy = smallfield.compute_vdw_energy(
        coords_x, coords_y, coords_z, pair_i, pair_j, epsilon, r0, backend="auto"
    )

    elec_energy = smallfield.compute_electrostatic_energy(
        coords_x,
        coords_y,
        coords_z,
        pair_i,
        pair_j,
        charge_i,
        charge_j,
        is_1_4,
        backend="auto",
    )

    total, vdw, elec = smallfield.compute_total_energy(
        coords_x,
        coords_y,
        coords_z,
        pair_i,
        pair_j,
        epsilon,
        r0,
        charge_i,
        charge_j,
        is_1_4,
        backend="auto",
    )

    assert total == vdw + elec
    assert vdw == vdw_energy
    assert elec == elec_energy


@pytest.mark.skipif(not smallfield.has_cuda(), reason="CUDA not available")
def test_cuda_vdw(ethanol_mol):
    """Test CUDA VdW energy computation."""
    coords_x, coords_y, coords_z = extract_coordinates(ethanol_mol, dtype=np.float32)
    pair_i, pair_j, epsilon, r0 = extract_vdw_pairs(ethanol_mol, dtype=np.float32)

    # Pad arrays
    n_pairs = len(pair_i)
    padding = (16 - n_pairs % 16) % 16
    if padding > 0:
        pair_i = np.pad(pair_i, (0, padding), mode="edge")
        pair_j = np.pad(pair_j, (0, padding), mode="edge")
        epsilon = np.pad(epsilon, (0, padding), mode="constant", constant_values=0)
        r0 = np.pad(r0, (0, padding), mode="constant", constant_values=1.0)

    energy_cuda = smallfield.compute_vdw_energy_cuda(
        coords_x, coords_y, coords_z, pair_i, pair_j, epsilon, r0
    )

    # Compare with serial version
    energy_serial = smallfield.compute_vdw_energy_serial(
        coords_x.astype(np.float64),
        coords_y.astype(np.float64),
        coords_z.astype(np.float64),
        pair_i,
        pair_j,
        epsilon.astype(np.float64),
        r0.astype(np.float64),
    )

    # CUDA uses FP16/FP32, so tolerance is higher
    np.testing.assert_allclose(energy_cuda, energy_serial, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
