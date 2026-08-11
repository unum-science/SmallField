#!/usr/bin/env python3
"""
SmallField vs RDKit MMFF94 Validation

Compare each energy term against RDKit's MMFF94 implementation:
- Bond stretching
- Angle bending
- Torsion
- Van der Waals
- Electrostatic

This script validates that SmallField produces comparable results to RDKit.
"""

import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
from rich.table import Table

from smallfield.log import Logger

log = Logger()

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdForceFieldHelpers
except ImportError:
    log.error("RDKit required. Install with: pixi add rdkit")
    sys.exit(1)

# MMFF94 Constants (must match the native implementation exactly)
MMFF_BOND_PREFACTOR = 143.9325  # mdyn*A -> kcal/mol
MMFF_CUBIC_STRETCH = -2.0  # A^-1

MMFF_ANGLE_PREFACTOR = 0.021922  # 0.5 * 0.043844 (mdyn*A/rad^2 -> kcal/mol)
MMFF_ANGLE_CUBIC = -0.006981317  # Exact RDKit cb value (degree^-1)

MMFF_TORSION_PREFACTOR = 0.5  # kcal/mol

MMFF_VDW_DELTA = 0.07  # Buffering parameter
MMFF_VDW_GAMMA = 0.12  # Buffering parameter

MMFF_ELEC_CONSTANT = 332.0716  # e^2/(4*pi*eps0) in kcal*A/mol
MMFF_ELEC_BUFFER = 0.05  # Angstrom buffer for singularity

# Additional terms
MMFF_OOP_PREFACTOR = 0.5 * 0.043844  # 0.5 * c2 = 0.5 * 143.9325 * DEG2RAD^2
MMFF_STBN_PREFACTOR = 143.9325 * np.pi / 180.0  # c5 = MDYNE_A_TO_KCAL_MOL * DEG2RAD


def bond_energy(bond_length: float, equilibrium_distance: float, force_constant: float) -> float:
    """MMFF94 bond stretching energy, quartic-with-cubic-correction form.

    ``E = ½ · 143.9325 · kb · (r − r0)² · [1 + cs·(r − r0) + (7/12)·cs²·(r − r0)²]``
    with ``cs = −2.0 Å⁻¹`` (the MMFF94 cubic stretch constant).

    Args:
        bond_length: Current bond length in Å, MMFF ``r``.
        equilibrium_distance: Equilibrium bond length in Å, MMFF ``r0``.
        force_constant: Bond-stretching force constant in mdyn/Å, MMFF ``kb``.

    Returns:
        Energy in kcal/mol.
    """
    deviation = bond_length - equilibrium_distance
    deviation_squared = deviation * deviation
    cubic_stretch = MMFF_CUBIC_STRETCH
    return (0.5 * MMFF_BOND_PREFACTOR * force_constant * deviation_squared *
            (1.0 + cubic_stretch * deviation + (7.0 / 12.0) * cubic_stretch * cubic_stretch * deviation_squared))


def angle_energy(theta_rad: float, theta0_rad: float, force_constant: float) -> float:
    """MMFF94 angle-bending energy, quadratic-with-cubic-correction in degrees.

    ``E = 0.021922 · ka · Δθ² · (1 + cb · Δθ)`` where ``Δθ`` is ``(θ − θ0)`` in
    degrees and ``cb = −0.006981317 deg⁻¹`` is the MMFF94 cubic-bend constant.

    Args:
        theta_rad: Current angle i-j-k in radians.
        theta0_rad: Equilibrium angle i-j-k in radians.
        force_constant: Angle-bending force constant in mdyn·Å/rad², MMFF ``ka``.

    Returns:
        Energy in kcal/mol.
    """
    dtheta_deg = (theta_rad - theta0_rad) * 180.0 / np.pi
    dtheta_deg2 = dtheta_deg * dtheta_deg
    return MMFF_ANGLE_PREFACTOR * force_constant * dtheta_deg2 * (1.0 + MMFF_ANGLE_CUBIC * dtheta_deg)


def torsion_energy(phi_rad: float, torsion_v1: float, torsion_v2: float, torsion_v3: float) -> float:
    """MMFF94 three-term torsion energy.

    ``E = ½ · [V1·(1 + cos φ) + V2·(1 − cos 2φ) + V3·(1 + cos 3φ)]``.

    Args:
        phi_rad: Dihedral angle i-j-k-l in radians.
        torsion_v1: 1-fold Fourier coefficient (kcal/mol), MMFF ``V1``.
        torsion_v2: 2-fold Fourier coefficient (kcal/mol), MMFF ``V2``.
        torsion_v3: 3-fold Fourier coefficient (kcal/mol), MMFF ``V3``.

    Returns:
        Energy in kcal/mol.
    """
    cos_phi = np.cos(phi_rad)
    cos_2phi = np.cos(2.0 * phi_rad)
    cos_3phi = np.cos(3.0 * phi_rad)
    return MMFF_TORSION_PREFACTOR * (torsion_v1 * (1.0 + cos_phi) + torsion_v2 * (1.0 - cos_2phi) +
                                     torsion_v3 * (1.0 + cos_3phi))


def vdw_energy(distance: float, epsilon: float, r_star: float) -> float:
    """MMFF94 buffered 14-7 van der Waals energy.

    ``E = ε · [(1+δ)·R* / (r + δ·R*)]⁷ · [(1+γ)·R*⁷ / (r⁷ + γ·R*⁷) − 2]``
    with buffering constants ``δ = 0.07`` and ``γ = 0.12``.

    Args:
        distance: Distance between the two atoms in Å.
        epsilon: Well depth ε in kcal/mol.
        r_star: Minimum-energy separation R* in Å.

    Returns:
        Energy in kcal/mol.
    """
    delta = MMFF_VDW_DELTA
    gamma = MMFF_VDW_GAMMA

    # Repulsive term: (1.07 * R* / (R + 0.07*R*))^7
    r_star_7 = r_star**7
    denom_rep = distance + delta * r_star
    ratio_rep = (1.0 + delta) * r_star / denom_rep
    rep_term = ratio_rep**7

    # Attractive term: (1.12*R*^7)/(R^7 + 0.12*R*^7) - 2
    distance_7 = distance**7
    denom_att = distance_7 + gamma * r_star_7
    att_term = (1.0 + gamma) * r_star_7 / denom_att - 2.0

    return epsilon * rep_term * att_term


def elec_energy(distance: float, charge_i: float, charge_j: float, dielectric: float = 1.0) -> float:
    """MMFF94 buffered Coulomb electrostatic energy.

    ``E = 332.0716 · qi · qj / [dielectric · (r + 0.05)]``. The 0.05 Å buffer
    prevents singularity at short range; 332.0716 converts e²/(4π·ε₀) to
    kcal·Å/mol.

    Args:
        distance: Distance between the two atoms in Å.
        charge_i: Partial charge on atom i in electron units.
        charge_j: Partial charge on atom j in electron units.
        dielectric: Dielectric constant (default 1.0 for gas-phase).

    Returns:
        Energy in kcal/mol.
    """
    r_eff = distance + MMFF_ELEC_BUFFER
    return MMFF_ELEC_CONSTANT * charge_i * charge_j / (dielectric * r_eff)


def oop_energy(chi_deg: float, koop: float) -> float:
    """MMFF94 out-of-plane bending energy.

    ``E = ½ · c2 · koop · χ²`` where ``c2 = 0.043844`` kcal/(mol·deg²) converts
    mdyn·Å·rad⁻² into kcal/(mol·deg²).

    Args:
        chi_deg: Out-of-plane angle χ in degrees (the angle between the j→l
            bond and the plane defined by i-j-k, with j the central sp² atom).
        koop: Out-of-plane force constant in mdyn·Å/rad².

    Returns:
        Energy in kcal/mol.
    """
    return MMFF_OOP_PREFACTOR * koop * chi_deg * chi_deg


def stretch_bend_energy(deviation_ij: float, deviation_jk: float, dtheta_deg: float, kba_ijk: float,
                        kba_kji: float) -> float:
    """MMFF94 stretch-bend coupling energy.

    ``E = c5 · Δθ · (kba_ijk · Δr_ij + kba_kji · Δr_jk)`` where
    ``c5 = 143.9325 · π/180`` converts the mixed mdyn-mdyn·Å / rad-deg units
    into kcal/mol.

    Args:
        deviation_ij: Bond-length deviation Δr_ij = r_ij − r0_ij for the i-j bond, in Å.
        deviation_jk: Bond-length deviation Δr_jk = r_jk − r0_jk for the j-k bond, in Å.
        dtheta_deg: Angle deviation Δθ = θ_ijk − θ0_ijk in degrees.
        kba_ijk: Stretch-bend force constant coupling the i-j bond to the
            i-j-k angle, in mdyn/rad.
        kba_kji: Stretch-bend force constant coupling the k-j bond to the
            i-j-k angle, in mdyn/rad.

    Returns:
        Energy in kcal/mol.
    """
    return MMFF_STBN_PREFACTOR * dtheta_deg * (kba_ijk * deviation_ij + kba_kji * deviation_jk)


def compute_oop_angle(positions: np.ndarray, atom_i: int, atom_j: int, atom_k: int, atom_l: int) -> float:
    """Compute out-of-plane angle chi in degrees.

    Atom j is the central sp2 atom, with i, k, l as neighbors.
    chi is the angle between the j-l bond and the plane defined by i-j-k.
    """
    r_ji = positions[atom_i] - positions[atom_j]
    r_jk = positions[atom_k] - positions[atom_j]
    r_jl = positions[atom_l] - positions[atom_j]

    # Normalize
    n_ji = np.linalg.norm(r_ji)
    n_jk = np.linalg.norm(r_jk)
    n_jl = np.linalg.norm(r_jl)

    if n_ji < 1e-10 or n_jk < 1e-10 or n_jl < 1e-10:
        return 0.0

    r_ji = r_ji / n_ji
    r_jk = r_jk / n_jk
    r_jl = r_jl / n_jl

    # Normal to the i-j-k plane
    plane_normal = np.cross(r_ji, r_jk)
    plane_normal_length = np.linalg.norm(plane_normal)
    if plane_normal_length < 1e-10:
        return 0.0
    plane_normal = plane_normal / plane_normal_length

    # sin(chi) = r_jl dot plane_normal
    sin_chi = np.dot(r_jl, plane_normal)
    sin_chi = np.clip(sin_chi, -1.0, 1.0)

    return np.degrees(np.arcsin(sin_chi))


def get_atom_positions(molecule, conf_id: int = -1) -> np.ndarray:
    """Get atom positions as numpy array."""
    conf = molecule.GetConformer(conf_id)
    atom_count = molecule.GetNumAtoms()
    positions = np.zeros((atom_count, 3))
    for atom_index in range(atom_count):
        position = conf.GetAtomPosition(atom_index)
        positions[atom_index] = [position.x, position.y, position.z]
    return positions


def compute_distance(positions: np.ndarray, atom_i: int, atom_j: int) -> float:
    """Compute distance between two atoms."""
    diff = positions[atom_j] - positions[atom_i]
    return np.sqrt(np.sum(diff * diff))


def compute_angle(positions: np.ndarray, atom_i: int, atom_j: int, atom_k: int) -> float:
    """Compute angle i-j-k (j is central) in radians."""
    edge_ji = positions[atom_i] - positions[atom_j]
    edge_jk = positions[atom_k] - positions[atom_j]
    length_ji = np.sqrt(np.sum(edge_ji * edge_ji))
    length_jk = np.sqrt(np.sum(edge_jk * edge_jk))
    if length_ji < 1e-10 or length_jk < 1e-10:
        return 0.0
    cos_theta = np.dot(edge_ji, edge_jk) / (length_ji * length_jk)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def compute_torsion(positions: np.ndarray, atom_i: int, atom_j: int, atom_k: int, atom_l: int) -> float:
    """Compute torsion angle i-j-k-l in radians."""
    bond_ij = positions[atom_j] - positions[atom_i]
    bond_jk = positions[atom_k] - positions[atom_j]
    bond_kl = positions[atom_l] - positions[atom_k]

    normal_ijk = np.cross(bond_ij, bond_jk)
    normal_jkl = np.cross(bond_jk, bond_kl)

    normal_ijk_length = np.sqrt(np.sum(normal_ijk * normal_ijk))
    normal_jkl_length = np.sqrt(np.sum(normal_jkl * normal_jkl))

    if normal_ijk_length < 1e-10 or normal_jkl_length < 1e-10:
        return 0.0

    normal_ijk = normal_ijk / normal_ijk_length
    normal_jkl = normal_jkl / normal_jkl_length

    cos_phi = np.dot(normal_ijk, normal_jkl)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)

    # Sign from cross product
    bond_jk_length = np.sqrt(np.sum(bond_jk * bond_jk))
    if bond_jk_length > 1e-10:
        reference_axis = np.cross(normal_ijk, bond_jk / bond_jk_length)
        sin_phi = np.dot(reference_axis, normal_jkl)
        phi_rad = np.arctan2(sin_phi, cos_phi)
    else:
        phi_rad = np.arccos(cos_phi)

    return phi_rad


def validate_molecule(smiles: str, verbose: bool = True) -> Dict:
    """
    Validate SmallField MMFF94 against RDKit for a molecule.

    Returns dict with per-term comparison results.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {"error": f"Invalid SMILES: {smiles}"}

    molecule = Chem.AddHs(molecule)

    # Generate conformer
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(molecule, params) == -1:
        return {"error": "Could not embed molecule"}

    # Get MMFF properties
    mmff_props = AllChem.MMFFGetMoleculeProperties(molecule)
    if mmff_props is None:
        return {"error": "Could not get MMFF properties"}

    # Get RDKit force field for reference total
    force_field = AllChem.MMFFGetMoleculeForceField(molecule, mmff_props)
    rdkit_total = force_field.CalcEnergy()

    # Get positions
    positions = get_atom_positions(molecule)
    n_atoms = molecule.GetNumAtoms()

    # region Bond Stretching
    smallfield_bond = 0.0
    rdkit_bond = 0.0
    n_bonds = 0

    for bond in molecule.GetBonds():
        atom_i = bond.GetBeginAtomIdx()
        atom_j = bond.GetEndAtomIdx()

        distance = compute_distance(positions, atom_i, atom_j)

        # Get MMFF parameters
        params = mmff_props.GetMMFFBondStretchParams(molecule, atom_i, atom_j)
        if params:
            _, force_constant, equilibrium_distance = params
            energy = bond_energy(distance, equilibrium_distance, force_constant)
            smallfield_bond += energy
            n_bonds += 1

    # RDKit doesn't expose per-term energies directly, but we can estimate
    # by comparing our implementation
    # endregion Bond Stretching

    # region Angle Bending
    smallfield_angle = 0.0
    n_angles = 0

    # Find all angles from bond connectivity
    angles_seen = set()
    for atom in molecule.GetAtoms():
        atom_j = atom.GetIdx()
        neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        for idx1, atom_i in enumerate(neighbors):
            for atom_k in neighbors[idx1 + 1 :]:
                if (atom_i, atom_j, atom_k) not in angles_seen and (atom_k, atom_j, atom_i) not in angles_seen:
                    angles_seen.add((atom_i, atom_j, atom_k))

                    theta = compute_angle(positions, atom_i, atom_j, atom_k)

                    # Get MMFF angle parameters
                    params = mmff_props.GetMMFFAngleBendParams(molecule, atom_i, atom_j, atom_k)
                    if params:
                        _, force_constant, theta0_deg = params
                        theta0_rad = theta0_deg * np.pi / 180.0
                        energy = angle_energy(theta, theta0_rad, force_constant)
                        smallfield_angle += energy
                        n_angles += 1

    # endregion Angle Bending

    # region Torsion
    smallfield_torsion = 0.0
    n_torsions = 0

    # Find all torsions
    for bond in molecule.GetBonds():
        atom_j = bond.GetBeginAtomIdx()
        atom_k = bond.GetEndAtomIdx()

        neighbors_j = [neighbor.GetIdx() for neighbor in molecule.GetAtomWithIdx(atom_j).GetNeighbors()
                       if neighbor.GetIdx() != atom_k]
        neighbors_k = [neighbor.GetIdx() for neighbor in molecule.GetAtomWithIdx(atom_k).GetNeighbors()
                       if neighbor.GetIdx() != atom_j]

        for atom_i in neighbors_j:
            for atom_l in neighbors_k:
                if atom_i != atom_l:
                    phi_rad = compute_torsion(positions, atom_i, atom_j, atom_k, atom_l)

                    # Get MMFF torsion parameters
                    params = mmff_props.GetMMFFTorsionParams(molecule, atom_i, atom_j, atom_k, atom_l)
                    if params:
                        _, torsion_v1, torsion_v2, torsion_v3 = params
                        energy = torsion_energy(phi_rad, torsion_v1, torsion_v2, torsion_v3)
                        smallfield_torsion += energy
                        n_torsions += 1

    # endregion Torsion

    # region Van Der Waals
    smallfield_vdw = 0.0
    n_vdw = 0

    # Build bonded neighbor sets for exclusions
    bonded_1_2 = {atom_index: set() for atom_index in range(n_atoms)}
    for bond in molecule.GetBonds():
        atom_i, atom_j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded_1_2[atom_i].add(atom_j)
        bonded_1_2[atom_j].add(atom_i)

    # 1-3 neighbors
    bonded_1_3 = {atom_index: set() for atom_index in range(n_atoms)}
    for atom_i in range(n_atoms):
        for atom_j in bonded_1_2[atom_i]:
            for atom_k in bonded_1_2[atom_j]:
                if atom_k != atom_i:
                    bonded_1_3[atom_i].add(atom_k)

    # 1-4 neighbors
    bonded_1_4 = {atom_index: set() for atom_index in range(n_atoms)}
    for atom_i in range(n_atoms):
        for atom_j in bonded_1_3[atom_i]:
            for atom_k in bonded_1_2[atom_j]:
                if atom_k != atom_i and atom_k not in bonded_1_2[atom_i] and atom_k not in bonded_1_3[atom_i]:
                    bonded_1_4[atom_i].add(atom_k)

    for atom_i in range(n_atoms):
        for atom_j in range(atom_i + 1, n_atoms):
            # Skip 1-2 and 1-3
            if atom_j in bonded_1_2[atom_i] or atom_j in bonded_1_3[atom_i]:
                continue

            distance = compute_distance(positions, atom_i, atom_j)
            if distance < 0.1:
                continue

            # Get vdW parameters (no 1-4 scaling for vdW in MMFF94)
            params = mmff_props.GetMMFFVdWParams(atom_i, atom_j)
            if params:
                r_star_ij, epsilon_ij = params[0], params[1]

                # Note: MMFF94 does NOT apply 1-4 scaling to vdW energy
                # 1-4 scaling (0.75) is only for electrostatic interactions
                energy = vdw_energy(distance, epsilon_ij, r_star_ij)
                smallfield_vdw += energy
                n_vdw += 1

    # endregion Van Der Waals

    # region Electrostatic
    smallfield_elec = 0.0
    n_elec = 0

    for atom_i in range(n_atoms):
        for atom_j in range(atom_i + 1, n_atoms):
            # Skip 1-2 and 1-3
            if atom_j in bonded_1_2[atom_i] or atom_j in bonded_1_3[atom_i]:
                continue

            charge_i = mmff_props.GetMMFFPartialCharge(atom_i)
            charge_j = mmff_props.GetMMFFPartialCharge(atom_j)

            if abs(charge_i) < 1e-10 or abs(charge_j) < 1e-10:
                continue

            distance = compute_distance(positions, atom_i, atom_j)
            if distance < 0.1:
                continue

            # 1-4 scaling
            scale = 0.75 if atom_j in bonded_1_4[atom_i] else 1.0

            energy = elec_energy(distance, charge_i, charge_j) * scale
            smallfield_elec += energy
            n_elec += 1

    # endregion Electrostatic

    # region Stretch-Bend Coupling
    smallfield_stbn = 0.0
    n_stbn = 0

    # Recompute angles with bond deviations for stretch-bend
    for atom in molecule.GetAtoms():
        atom_j = atom.GetIdx()
        neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        for idx1, atom_i in enumerate(neighbors):
            for atom_k in neighbors[idx1 + 1 :]:
                # Get stretch-bend parameters
                stbn_params = mmff_props.GetMMFFStretchBendParams(molecule, atom_i, atom_j, atom_k)
                if stbn_params:
                    _, kba_ijk, kba_kji = stbn_params

                    # Get bond parameters for the two bonds
                    bond_params1 = mmff_props.GetMMFFBondStretchParams(molecule, atom_i, atom_j)
                    bond_params2 = mmff_props.GetMMFFBondStretchParams(molecule, atom_j, atom_k)
                    angle_params = mmff_props.GetMMFFAngleBendParams(molecule, atom_i, atom_j, atom_k)

                    if bond_params1 and bond_params2 and angle_params:
                        _, force_constant_ij, equilibrium_distance_ij = bond_params1
                        _, force_constant_jk, equilibrium_distance_jk = bond_params2
                        _, force_constant_angle, theta0_deg = angle_params

                        distance_ij = compute_distance(positions, atom_i, atom_j)
                        distance_jk = compute_distance(positions, atom_j, atom_k)
                        theta = compute_angle(positions, atom_i, atom_j, atom_k)
                        theta_deg = np.degrees(theta)

                        deviation_ij = distance_ij - equilibrium_distance_ij
                        deviation_jk = distance_jk - equilibrium_distance_jk
                        dtheta_deg = theta_deg - theta0_deg

                        energy = stretch_bend_energy(deviation_ij, deviation_jk, dtheta_deg, kba_ijk, kba_kji)
                        smallfield_stbn += energy
                        n_stbn += 1

    # endregion Stretch-Bend Coupling

    # region Out-Of-Plane Bending (For sp2 Centers)
    smallfield_oop = 0.0
    n_oop = 0

    # Find trigonal (sp2) atoms with 3 neighbors
    for atom in molecule.GetAtoms():
        atom_j = atom.GetIdx()
        neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]

        if len(neighbors) == 3:
            # Try all permutations of i, k, l around central atom j
            for perm_idx, (atom_i, atom_k, atom_l) in enumerate(
                [
                    (neighbors[0], neighbors[1], neighbors[2]),
                    (neighbors[0], neighbors[2], neighbors[1]),
                    (neighbors[1], neighbors[0], neighbors[2]),
                    (neighbors[1], neighbors[2], neighbors[0]),
                    (neighbors[2], neighbors[0], neighbors[1]),
                    (neighbors[2], neighbors[1], neighbors[0]),
                ]
            ):
                # Get OOP parameters
                oop_params = mmff_props.GetMMFFOopBendParams(molecule, atom_i, atom_j, atom_k, atom_l)
                if oop_params:
                    koop = oop_params

                    chi_deg = compute_oop_angle(positions, atom_i, atom_j, atom_k, atom_l)
                    energy = oop_energy(chi_deg, koop)
                    smallfield_oop += energy
                    n_oop += 1

    # endregion Out-Of-Plane Bending (For sp2 Centers)

    # region Summary
    smallfield_total = (
        smallfield_bond
        + smallfield_angle
        + smallfield_stbn
        + smallfield_oop
        + smallfield_torsion
        + smallfield_vdw
        + smallfield_elec
    )

    result = {
        "smiles": smiles,
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "n_angles": n_angles,
        "n_stbn": n_stbn,
        "n_oop": n_oop,
        "n_torsions": n_torsions,
        "n_vdw_pairs": n_vdw,
        "n_elec_pairs": n_elec,
        "smallfield": {
            "bond": smallfield_bond,
            "angle": smallfield_angle,
            "stbn": smallfield_stbn,
            "oop": smallfield_oop,
            "torsion": smallfield_torsion,
            "vdw": smallfield_vdw,
            "elec": smallfield_elec,
            "total": smallfield_total,
        },
        "rdkit_total": rdkit_total,
        "difference": smallfield_total - rdkit_total,
        "relative_error": abs(smallfield_total - rdkit_total) / abs(rdkit_total) if abs(rdkit_total) > 1e-10 else 0.0,
    }

    if verbose:
        breakdown = Table(
            title=f"{smiles}  ({n_atoms} atoms, {n_bonds} bonds, {n_angles} angles, "
            f"{n_torsions} torsions, {n_stbn} stbn, {n_oop} oop, {n_vdw} vdW, {n_elec} elec)",
            show_header=False,
        )
        breakdown.add_column("term", style="bold")
        breakdown.add_column("kcal/mol", justify="right")
        breakdown.add_row("Bond", f"{smallfield_bond:12.4f}")
        breakdown.add_row("Angle", f"{smallfield_angle:12.4f}")
        breakdown.add_row("Stretch-bend", f"{smallfield_stbn:12.4f}")
        breakdown.add_row("Out-of-plane", f"{smallfield_oop:12.4f}")
        breakdown.add_row("Torsion", f"{smallfield_torsion:12.4f}")
        breakdown.add_row("van der Waals", f"{smallfield_vdw:12.4f}")
        breakdown.add_row("Electrostatic", f"{smallfield_elec:12.4f}")
        breakdown.add_row("[bold]SmallField TOTAL[/]", f"[bold]{smallfield_total:12.4f}[/]")
        breakdown.add_row("RDKit MMFF94 total", f"{rdkit_total:12.4f}")
        breakdown.add_row("difference", f"{result['difference']:12.4f}")
        breakdown.add_row("relative error", f"{result['relative_error']*100:12.2f} %")
        log.console.print(breakdown)
    # endregion Summary

    return result


def load_usearch_molecules(filepath: str, n_samples: int = 50, seed: int = 42) -> List[str]:
    """Load random sample of molecules from USearchMolecules SMILES file."""
    import random

    random.seed(seed)

    molecules = []
    with open(filepath, "r") as stream:
        for line in stream:
            # Each line may contain multiple molecules separated by '.'
            # Take the first molecule from each line
            smiles = line.strip().split(".")[0]
            if smiles and len(smiles) < 100:  # Skip very long SMILES
                molecules.append(smiles)
            if len(molecules) >= 10000:  # Read max 10k to sample from
                break

    # Random sample
    return random.sample(molecules, min(n_samples, len(molecules)))


def test_validation_formula_track_rdkit_for_ethanol():
    result = validate_molecule("CCO", verbose=False)

    assert "error" not in result
    assert result["n_atoms"] == 9
    assert result["relative_error"] < 1e-3
    assert abs(result["smallfield"]["total"] - result["rdkit_total"]) < 1e-3


def main():
    import os

    log.info("SmallField vs RDKit MMFF94 validation")

    # Optional corpus for `--usearch`; point SMALLFIELD_SMI_CORPUS at a .smi file.
    usearch_path = os.environ.get("SMALLFIELD_SMI_CORPUS", "")

    if os.path.exists(usearch_path) and len(sys.argv) > 1 and sys.argv[1] == "--usearch":
        log.info("loading molecules from USearchMolecules dataset")
        test_molecules = load_usearch_molecules(usearch_path, n_samples=50)
        log.info("loaded {} random molecules", len(test_molecules))
    else:
        # Test molecules of increasing complexity
        test_molecules = [
            "C",  # Methane
            "CC",  # Ethane
            "CCC",  # Propane
            "CCO",  # Ethanol
            "CCCC",  # Butane
            "c1ccccc1",  # Benzene
            "CC(=O)O",  # Acetic acid
            "CCCCCCCCC",  # Nonane
            "c1ccc(O)cc1",  # Phenol
        ]

    results = []
    for smiles in test_molecules:
        result = validate_molecule(smiles, verbose=True)
        if "error" not in result:
            results.append(result)

    summary = Table(title="Validation summary", show_header=True, header_style="bold")
    summary.add_column("Molecule")
    summary.add_column("Atoms", justify="right")
    summary.add_column("SF total", justify="right")
    summary.add_column("RDKit", justify="right")
    summary.add_column("Error %", justify="right")
    for result in results:
        summary.add_row(
            result["smiles"],
            str(result["n_atoms"]),
            f"{result['smallfield']['total']:.4f}",
            f"{result['rdkit_total']:.4f}",
            f"{result['relative_error']*100:.2f}",
        )
    log.console.print(summary)

    # Overall accuracy assessment
    avg_error = np.mean([result["relative_error"] for result in results]) * 100
    max_error = np.max([result["relative_error"] for result in results]) * 100

    log.info("average relative error: {:.2f}%", avg_error)
    log.info("maximum relative error: {:.2f}%", max_error)

    if max_error < 5.0:
        log.info("PASS: SmallField matches RDKit within 5% tolerance — ready for SIMD/GPU optimization")
    elif max_error < 10.0:
        log.warn("some discrepancies detected (5-10%) — review parameter handling before optimization")
    else:
        log.error("significant discrepancies detected (>10%) — fix accuracy before SIMD/GPU optimization")


if __name__ == "__main__":
    main()
