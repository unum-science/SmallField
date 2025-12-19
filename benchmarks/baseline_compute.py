"""
Python f64 baseline computation for unified SmallField benchmark.

Computes MMFF94 energy in double precision (float64) for each molecule.
Results serve as the accuracy reference for comparing f16/bf16/f32 storage types.
"""

import numpy as np
from dataclasses import dataclass
from typing import List

from molecule_loader import MoleculeData


# MMFF94 Constants (must match Mojo implementation exactly)
MMFF_BOND_PREFACTOR = 143.9325  # mdyn*A -> kcal/mol
MMFF_CUBIC_STRETCH = -2.0  # A^-1

MMFF_ANGLE_PREFACTOR = 0.021922  # 0.5 * 0.043844 (mdyn*A/rad^2 -> kcal/mol)
MMFF_ANGLE_CUBIC = -0.006981317  # Exact RDKit cb value (degree^-1)

MMFF_TORSION_PREFACTOR = 0.5  # kcal/mol

MMFF_VDW_DELTA = 0.07  # Buffering parameter
MMFF_VDW_GAMMA = 0.12  # Buffering parameter

MMFF_ELEC_CONSTANT = 332.0716  # e^2/(4*pi*eps0) in kcal*A/mol
MMFF_ELEC_BUFFER = 0.05  # Angstrom buffer

RAD_TO_DEG = 180.0 / np.pi


@dataclass
class EnergyComponents:
    """Per-term MMFF94 energy breakdown."""
    bond: float = 0.0
    angle: float = 0.0
    torsion: float = 0.0
    vdw: float = 0.0
    elec: float = 0.0

    @property
    def total(self) -> float:
        return self.bond + self.angle + self.torsion + self.vdw + self.elec


def bond_energy(r: float, r0: float, kb: float) -> float:
    """MMFF94 bond stretching energy."""
    dr = r - r0
    dr2 = dr * dr
    cs = MMFF_CUBIC_STRETCH
    return 0.5 * MMFF_BOND_PREFACTOR * kb * dr2 * (1.0 + cs * dr + (7.0 / 12.0) * cs * cs * dr2)


def angle_energy(theta_rad: float, theta0_rad: float, ka: float) -> float:
    """MMFF94 angle bending energy."""
    dtheta_deg = (theta_rad - theta0_rad) * RAD_TO_DEG
    dtheta_deg2 = dtheta_deg * dtheta_deg
    return MMFF_ANGLE_PREFACTOR * ka * dtheta_deg2 * (1.0 + MMFF_ANGLE_CUBIC * dtheta_deg)


def torsion_energy(phi_rad: float, v1: float, v2: float, v3: float) -> float:
    """MMFF94 torsion energy."""
    cos_phi = np.cos(phi_rad)
    cos_2phi = np.cos(2.0 * phi_rad)
    cos_3phi = np.cos(3.0 * phi_rad)
    return MMFF_TORSION_PREFACTOR * (
        v1 * (1.0 + cos_phi) +
        v2 * (1.0 - cos_2phi) +
        v3 * (1.0 + cos_3phi)
    )


def pow7(x: float) -> float:
    """Efficient x^7 = x * x^2 * x^4."""
    x2 = x * x
    x4 = x2 * x2
    return x * x2 * x4


def vdw_energy(r: float, epsilon: float, r_star: float) -> float:
    """MMFF94 buffered 14-7 van der Waals energy."""
    delta = MMFF_VDW_DELTA
    gamma = MMFF_VDW_GAMMA

    # Repulsive term
    r_star_7 = pow7(r_star)
    denom_rep = r + delta * r_star
    ratio_rep = (1.0 + delta) * r_star / denom_rep
    rep_term = pow7(ratio_rep)

    # Attractive term
    r_7 = pow7(r)
    denom_att = r_7 + gamma * r_star_7
    att_term = (1.0 + gamma) * r_star_7 / denom_att - 2.0

    return epsilon * rep_term * att_term


def elec_energy(r: float, qi: float, qj: float, scale: float = 1.0) -> float:
    """MMFF94 electrostatic energy."""
    r_eff = r + MMFF_ELEC_BUFFER
    return scale * MMFF_ELEC_CONSTANT * qi * qj / r_eff


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


def compute_energy_f64(mol: MoleculeData) -> EnergyComponents:
    """
    Compute MMFF94 energy in f64 precision.

    Args:
        mol: Pre-parameterized molecule data

    Returns:
        EnergyComponents with per-term breakdown
    """
    pos = mol.positions  # Already float64

    result = EnergyComponents()

    # Bond stretching
    for bond in mol.bonds:
        r = compute_distance(pos, bond.i, bond.j)
        result.bond += bond_energy(r, bond.r0, bond.kb)

    # Angle bending
    for angle in mol.angles:
        theta = compute_angle(pos, angle.i, angle.j, angle.k)
        result.angle += angle_energy(theta, angle.theta0, angle.ka)

    # Torsion
    for tors in mol.torsions:
        phi = compute_torsion(pos, tors.i, tors.j, tors.k, tors.l)
        result.torsion += torsion_energy(phi, tors.v1, tors.v2, tors.v3)

    # Nonbonded (vdW + electrostatic)
    for nb in mol.nonbonded:
        r = compute_distance(pos, nb.i, nb.j)
        if r < 0.1:
            continue

        # vdW (no 1-4 scaling)
        result.vdw += vdw_energy(r, nb.epsilon, nb.r_star)

        # Electrostatic (0.75 scaling for 1-4)
        if abs(nb.qi) > 1e-10 and abs(nb.qj) > 1e-10:
            scale = 0.75 if nb.is_1_4 else 1.0
            result.elec += elec_energy(r, nb.qi, nb.qj, scale)

    return result


def compute_baseline(molecules: List[MoleculeData], verbose: bool = True) -> List[EnergyComponents]:
    """
    Compute f64 baseline for all molecules.

    Args:
        molecules: List of parameterized molecules
        verbose: Print progress

    Returns:
        List of EnergyComponents, one per molecule
    """
    if verbose:
        print(f"Computing f64 baseline for {len(molecules)} molecules...")

    results = []
    for mol in molecules:
        energy = compute_energy_f64(mol)
        results.append(energy)

    if verbose:
        # Statistics
        totals = [e.total for e in results]
        print(f"  Energy range: {min(totals):.2f} to {max(totals):.2f} kcal/mol")
        print(f"  Mean energy: {np.mean(totals):.2f} kcal/mol")

    return results


if __name__ == "__main__":
    from molecule_loader import load_molecules, DEFAULT_SMILES_PATH

    # Quick test
    molecules = load_molecules(DEFAULT_SMILES_PATH, n_samples=10)
    baseline = compute_baseline(molecules)

    print("\nSample energy breakdown:")
    for i, (mol, energy) in enumerate(zip(molecules[:3], baseline[:3])):
        print(f"\n{mol.smiles}:")
        print(f"  Bond:    {energy.bond:10.4f}")
        print(f"  Angle:   {energy.angle:10.4f}")
        print(f"  Torsion: {energy.torsion:10.4f}")
        print(f"  vdW:     {energy.vdw:10.4f}")
        print(f"  Elec:    {energy.elec:10.4f}")
        print(f"  Total:   {energy.total:10.4f}")
