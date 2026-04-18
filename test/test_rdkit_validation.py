#!/usr/bin/env python3
"""
SmallField vs RDKit MMFF94 Validation

Compare each energy term against RDKit's MMFF94 implementation:
- Bond stretching
- Angle bending
- Torsion
- Van der Waals
- Electrostatic

This script validates that SmallField produces comparable results to RDKit
before we proceed with SIMD/GPU optimization.
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

# MMFF94 Constants (must match Mojo implementation exactly)
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


def bond_energy(r: float, r0: float, kb: float) -> float:
    """MMFF94 bond stretching energy, quartic-with-cubic-correction form.

    ``E = ½ · 143.9325 · kb · (r − r0)² · [1 + cs·(r − r0) + (7/12)·cs²·(r − r0)²]``
    with ``cs = −2.0 Å⁻¹`` (the MMFF94 cubic stretch constant).

    Args:
        r: Current bond length in Å.
        r0: Equilibrium bond length in Å.
        kb: Bond-stretching force constant in mdyn/Å.

    Returns:
        Energy in kcal/mol.
    """
    dr = r - r0
    dr2 = dr * dr
    cs = MMFF_CUBIC_STRETCH
    return 0.5 * MMFF_BOND_PREFACTOR * kb * dr2 * (1.0 + cs * dr + (7.0 / 12.0) * cs * cs * dr2)


def angle_energy(theta_rad: float, theta0_rad: float, ka: float) -> float:
    """MMFF94 angle-bending energy, quadratic-with-cubic-correction in degrees.

    ``E = 0.021922 · ka · Δθ² · (1 + cb · Δθ)`` where ``Δθ`` is ``(θ − θ0)`` in
    degrees and ``cb = −0.006981317 deg⁻¹`` is the MMFF94 cubic-bend constant.

    Args:
        theta_rad: Current angle i-j-k in radians.
        theta0_rad: Equilibrium angle i-j-k in radians.
        ka: Angle-bending force constant in mdyn·Å/rad².

    Returns:
        Energy in kcal/mol.
    """
    dtheta_deg = (theta_rad - theta0_rad) * 180.0 / np.pi
    dtheta_deg2 = dtheta_deg * dtheta_deg
    return MMFF_ANGLE_PREFACTOR * ka * dtheta_deg2 * (1.0 + MMFF_ANGLE_CUBIC * dtheta_deg)


def torsion_energy(phi_rad: float, v1: float, v2: float, v3: float) -> float:
    """MMFF94 three-term torsion energy.

    ``E = ½ · [V1·(1 + cos φ) + V2·(1 − cos 2φ) + V3·(1 + cos 3φ)]``.

    Args:
        phi_rad: Dihedral angle i-j-k-l in radians.
        v1: 1-fold Fourier coefficient (kcal/mol).
        v2: 2-fold Fourier coefficient (kcal/mol).
        v3: 3-fold Fourier coefficient (kcal/mol).

    Returns:
        Energy in kcal/mol.
    """
    cos_phi = np.cos(phi_rad)
    cos_2phi = np.cos(2.0 * phi_rad)
    cos_3phi = np.cos(3.0 * phi_rad)
    return MMFF_TORSION_PREFACTOR * (v1 * (1.0 + cos_phi) + v2 * (1.0 - cos_2phi) + v3 * (1.0 + cos_3phi))


def vdw_energy(r: float, epsilon: float, r_star: float) -> float:
    """MMFF94 buffered 14-7 van der Waals energy.

    ``E = ε · [(1+δ)·R* / (r + δ·R*)]⁷ · [(1+γ)·R*⁷ / (r⁷ + γ·R*⁷) − 2]``
    with buffering constants ``δ = 0.07`` and ``γ = 0.12``.

    Args:
        r: Distance between the two atoms in Å.
        epsilon: Well depth ε in kcal/mol.
        r_star: Minimum-energy separation R* in Å.

    Returns:
        Energy in kcal/mol.
    """
    delta = MMFF_VDW_DELTA
    gamma = MMFF_VDW_GAMMA

    # Repulsive term: (1.07 * R* / (R + 0.07*R*))^7
    r_star_7 = r_star**7
    denom_rep = r + delta * r_star
    ratio_rep = (1.0 + delta) * r_star / denom_rep
    rep_term = ratio_rep**7

    # Attractive term: (1.12*R*^7)/(R^7 + 0.12*R*^7) - 2
    r_7 = r**7
    denom_att = r_7 + gamma * r_star_7
    att_term = (1.0 + gamma) * r_star_7 / denom_att - 2.0

    return epsilon * rep_term * att_term


def elec_energy(r: float, qi: float, qj: float, dielectric: float = 1.0) -> float:
    """MMFF94 buffered Coulomb electrostatic energy.

    ``E = 332.0716 · qi · qj / [dielectric · (r + 0.05)]``.  The 0.05 Å buffer
    prevents singularity at short range; 332.0716 converts e²/(4π·ε₀) to
    kcal·Å/mol.

    Args:
        r: Distance between the two atoms in Å.
        qi: Partial charge on atom i in electron units.
        qj: Partial charge on atom j in electron units.
        dielectric: Dielectric constant (default 1.0 for gas-phase).

    Returns:
        Energy in kcal/mol.
    """
    r_eff = r + MMFF_ELEC_BUFFER
    return MMFF_ELEC_CONSTANT * qi * qj / (dielectric * r_eff)


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


def stretch_bend_energy(dr1: float, dr2: float, dtheta_deg: float, kba_ijk: float, kba_kji: float) -> float:
    """MMFF94 stretch-bend coupling energy.

    ``E = c5 · Δθ · (kba_ijk · Δr_ij + kba_kji · Δr_jk)`` where
    ``c5 = 143.9325 · π/180`` converts the mixed mdyn-mdyn·Å / rad-deg units
    into kcal/mol.

    Args:
        dr1: Bond-length deviation Δr_ij = r_ij − r0_ij for the i-j bond, in Å.
        dr2: Bond-length deviation Δr_jk = r_jk − r0_jk for the j-k bond, in Å.
        dtheta_deg: Angle deviation Δθ = θ_ijk − θ0_ijk in degrees.
        kba_ijk: Stretch-bend force constant coupling the i-j bond to the
            i-j-k angle, in mdyn/rad.
        kba_kji: Stretch-bend force constant coupling the k-j bond to the
            i-j-k angle, in mdyn/rad.

    Returns:
        Energy in kcal/mol.
    """
    return MMFF_STBN_PREFACTOR * dtheta_deg * (kba_ijk * dr1 + kba_kji * dr2)


def compute_oop_angle(pos: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Compute out-of-plane angle chi in degrees.

    Atom j is the central sp2 atom, with i, k, l as neighbors.
    chi is the angle between the j-l bond and the plane defined by i-j-k.
    """
    r_ji = pos[i] - pos[j]
    r_jk = pos[k] - pos[j]
    r_jl = pos[l] - pos[j]

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


def get_atom_positions(mol, conf_id: int = -1) -> np.ndarray:
    """Get atom positions as numpy array."""
    conf = mol.GetConformer(conf_id)
    n = mol.GetNumAtoms()
    pos = np.zeros((n, 3))
    for i in range(n):
        p = conf.GetAtomPosition(i)
        pos[i] = [p.x, p.y, p.z]
    return pos


def compute_distance(pos: np.ndarray, i: int, j: int) -> float:
    """Compute distance between two atoms."""
    diff = pos[j] - pos[i]
    return np.sqrt(np.sum(diff * diff))


def compute_angle(pos: np.ndarray, i: int, j: int, k: int) -> float:
    """Compute angle i-j-k (j is central) in radians."""
    v1 = pos[i] - pos[j]
    v2 = pos[k] - pos[j]
    r1 = np.sqrt(np.sum(v1 * v1))
    r2 = np.sqrt(np.sum(v2 * v2))
    if r1 < 1e-10 or r2 < 1e-10:
        return 0.0
    cos_theta = np.dot(v1, v2) / (r1 * r2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def compute_torsion(pos: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Compute torsion angle i-j-k-l in radians."""
    b1 = pos[j] - pos[i]
    b2 = pos[k] - pos[j]
    b3 = pos[l] - pos[k]

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.sqrt(np.sum(n1 * n1))
    n2_norm = np.sqrt(np.sum(n2 * n2))

    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    cos_phi = np.dot(n1, n2)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)

    # Sign from cross product
    b2_norm = np.sqrt(np.sum(b2 * b2))
    if b2_norm > 1e-10:
        m1 = np.cross(n1, b2 / b2_norm)
        sin_phi = np.dot(m1, n2)
        phi = np.arctan2(sin_phi, cos_phi)
    else:
        phi = np.arccos(cos_phi)

    return phi


def validate_molecule(smiles: str, verbose: bool = True) -> Dict:
    """
    Validate SmallField MMFF94 against RDKit for a molecule.

    Returns dict with per-term comparison results.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}

    mol = Chem.AddHs(mol)

    # Generate conformer
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) == -1:
        return {"error": "Could not embed molecule"}

    # Get MMFF properties
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    if mmff_props is None:
        return {"error": "Could not get MMFF properties"}

    # Get RDKit force field for reference total
    ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)
    rdkit_total = ff.CalcEnergy()

    # Get positions
    pos = get_atom_positions(mol)
    n_atoms = mol.GetNumAtoms()

    # =========================================================================
    # BOND STRETCHING
    # =========================================================================
    smallfield_bond = 0.0
    rdkit_bond = 0.0
    n_bonds = 0

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        r = compute_distance(pos, i, j)

        # Get MMFF parameters
        params = mmff_props.GetMMFFBondStretchParams(mol, i, j)
        if params:
            _, kb, r0 = params
            e = bond_energy(r, r0, kb)
            smallfield_bond += e
            n_bonds += 1

    # RDKit doesn't expose per-term energies directly, but we can estimate
    # by comparing our implementation

    # =========================================================================
    # ANGLE BENDING
    # =========================================================================
    smallfield_angle = 0.0
    n_angles = 0

    # Find all angles from bond connectivity
    angles_seen = set()
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        for idx1, i in enumerate(neighbors):
            for k in neighbors[idx1 + 1 :]:
                if (i, j, k) not in angles_seen and (k, j, i) not in angles_seen:
                    angles_seen.add((i, j, k))

                    theta = compute_angle(pos, i, j, k)

                    # Get MMFF angle parameters
                    params = mmff_props.GetMMFFAngleBendParams(mol, i, j, k)
                    if params:
                        _, ka, theta0_deg = params
                        theta0_rad = theta0_deg * np.pi / 180.0
                        e = angle_energy(theta, theta0_rad, ka)
                        smallfield_angle += e
                        n_angles += 1

    # =========================================================================
    # TORSION
    # =========================================================================
    smallfield_torsion = 0.0
    n_torsions = 0

    # Find all torsions
    for bond in mol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()

        neighbors_j = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
        neighbors_k = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]

        for i in neighbors_j:
            for l in neighbors_k:
                if i != l:
                    phi = compute_torsion(pos, i, j, k, l)

                    # Get MMFF torsion parameters
                    params = mmff_props.GetMMFFTorsionParams(mol, i, j, k, l)
                    if params:
                        _, v1, v2, v3 = params
                        e = torsion_energy(phi, v1, v2, v3)
                        smallfield_torsion += e
                        n_torsions += 1

    # =========================================================================
    # VAN DER WAALS
    # =========================================================================
    smallfield_vdw = 0.0
    n_vdw = 0

    # Build bonded neighbor sets for exclusions
    bonded_1_2 = {i: set() for i in range(n_atoms)}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded_1_2[i].add(j)
        bonded_1_2[j].add(i)

    # 1-3 neighbors
    bonded_1_3 = {i: set() for i in range(n_atoms)}
    for i in range(n_atoms):
        for j in bonded_1_2[i]:
            for k in bonded_1_2[j]:
                if k != i:
                    bonded_1_3[i].add(k)

    # 1-4 neighbors
    bonded_1_4 = {i: set() for i in range(n_atoms)}
    for i in range(n_atoms):
        for j in bonded_1_3[i]:
            for k in bonded_1_2[j]:
                if k != i and k not in bonded_1_2[i] and k not in bonded_1_3[i]:
                    bonded_1_4[i].add(k)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Skip 1-2 and 1-3
            if j in bonded_1_2[i] or j in bonded_1_3[i]:
                continue

            r = compute_distance(pos, i, j)
            if r < 0.1:
                continue

            # Get vdW parameters (no 1-4 scaling for vdW in MMFF94)
            params = mmff_props.GetMMFFVdWParams(i, j)
            if params:
                r_star_ij, epsilon_ij = params[0], params[1]

                # Note: MMFF94 does NOT apply 1-4 scaling to vdW energy
                # 1-4 scaling (0.75) is only for electrostatic interactions
                e = vdw_energy(r, epsilon_ij, r_star_ij)
                smallfield_vdw += e
                n_vdw += 1

    # =========================================================================
    # ELECTROSTATIC
    # =========================================================================
    smallfield_elec = 0.0
    n_elec = 0

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Skip 1-2 and 1-3
            if j in bonded_1_2[i] or j in bonded_1_3[i]:
                continue

            qi = mmff_props.GetMMFFPartialCharge(i)
            qj = mmff_props.GetMMFFPartialCharge(j)

            if abs(qi) < 1e-10 or abs(qj) < 1e-10:
                continue

            r = compute_distance(pos, i, j)
            if r < 0.1:
                continue

            # 1-4 scaling
            scale = 0.75 if j in bonded_1_4[i] else 1.0

            e = elec_energy(r, qi, qj) * scale
            smallfield_elec += e
            n_elec += 1

    # =========================================================================
    # STRETCH-BEND COUPLING
    # =========================================================================
    smallfield_stbn = 0.0
    n_stbn = 0

    # Recompute angles with bond deviations for stretch-bend
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        for idx1, i in enumerate(neighbors):
            for k in neighbors[idx1 + 1 :]:
                # Get stretch-bend parameters
                stbn_params = mmff_props.GetMMFFStretchBendParams(mol, i, j, k)
                if stbn_params:
                    _, kba_ijk, kba_kji = stbn_params

                    # Get bond parameters for the two bonds
                    bond_params1 = mmff_props.GetMMFFBondStretchParams(mol, i, j)
                    bond_params2 = mmff_props.GetMMFFBondStretchParams(mol, j, k)
                    angle_params = mmff_props.GetMMFFAngleBendParams(mol, i, j, k)

                    if bond_params1 and bond_params2 and angle_params:
                        _, kb1, r0_1 = bond_params1
                        _, kb2, r0_2 = bond_params2
                        _, ka, theta0_deg = angle_params

                        r1 = compute_distance(pos, i, j)
                        r2 = compute_distance(pos, j, k)
                        theta = compute_angle(pos, i, j, k)
                        theta_deg = np.degrees(theta)

                        dr1 = r1 - r0_1
                        dr2 = r2 - r0_2
                        dtheta_deg = theta_deg - theta0_deg

                        e = stretch_bend_energy(dr1, dr2, dtheta_deg, kba_ijk, kba_kji)
                        smallfield_stbn += e
                        n_stbn += 1

    # =========================================================================
    # OUT-OF-PLANE BENDING (for sp2 centers)
    # =========================================================================
    smallfield_oop = 0.0
    n_oop = 0

    # Find trigonal (sp2) atoms with 3 neighbors
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]

        if len(neighbors) == 3:
            # Try all permutations of i, k, l around central atom j
            for perm_idx, (i, k, l) in enumerate(
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
                oop_params = mmff_props.GetMMFFOopBendParams(mol, i, j, k, l)
                if oop_params:
                    koop = oop_params

                    chi = compute_oop_angle(pos, i, j, k, l)
                    e = oop_energy(chi, koop)
                    smallfield_oop += e
                    n_oop += 1

    # =========================================================================
    # SUMMARY
    # =========================================================================
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

    return result


def load_usearch_molecules(filepath: str, n_samples: int = 50, seed: int = 42) -> List[str]:
    """Load random sample of molecules from USearchMolecules SMILES file."""
    import random

    random.seed(seed)

    molecules = []
    with open(filepath, "r") as f:
        for line in f:
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
    for r in results:
        summary.add_row(
            r["smiles"],
            str(r["n_atoms"]),
            f"{r['smallfield']['total']:.4f}",
            f"{r['rdkit_total']:.4f}",
            f"{r['relative_error']*100:.2f}",
        )
    log.console.print(summary)

    # Overall accuracy assessment
    avg_error = np.mean([r["relative_error"] for r in results]) * 100
    max_error = np.max([r["relative_error"] for r in results]) * 100

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
