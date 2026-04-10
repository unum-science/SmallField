"""
Binary data exporter for unified SmallField benchmark.

Exports pre-parameterized molecules to a binary format that can be
efficiently read by the Mojo runner.
"""

import struct
import numpy as np
from typing import List
from pathlib import Path

from molecule_loader import MoleculeData


# Binary format magic number: "SFML" = SmallField Molecule List
MAGIC = 0x4C4D4653  # "SFML" in little-endian
VERSION = 1


def export_binary(molecules: List[MoleculeData], output_path: str) -> dict:
    """
    Export molecules to binary format for Mojo runner.

    Binary Format:
    ==============
    Header (64 bytes):
        magic: u32 = 0x4C4D4653 ("SFML")
        version: u32 = 1
        num_molecules: u32
        num_atoms_total: u32
        num_bonds_total: u32
        num_angles_total: u32
        num_torsions_total: u32
        num_nb_total: u32
        padding: 36 bytes

    Molecule offsets (CSR-style, 5 arrays of (num_molecules + 1) u32 each):
        atom_starts[]
        bond_starts[]
        angle_starts[]
        torsion_starts[]
        nb_starts[]

    Atom data (contiguous, all f64):
        x[num_atoms_total]
        y[num_atoms_total]
        z[num_atoms_total]
        charge[num_atoms_total]

    Bond data:
        bond_i[num_bonds_total] (u32)
        bond_j[num_bonds_total] (u32)
        bond_kb[num_bonds_total] (f64)
        bond_r0[num_bonds_total] (f64)

    Angle data:
        angle_i[num_angles_total] (u32)
        angle_j[num_angles_total] (u32)
        angle_k[num_angles_total] (u32)
        angle_ka[num_angles_total] (f64)
        angle_theta0[num_angles_total] (f64)

    Torsion data:
        torsion_i[num_torsions_total] (u32)
        torsion_j[num_torsions_total] (u32)
        torsion_k[num_torsions_total] (u32)
        torsion_l[num_torsions_total] (u32)
        torsion_v1[num_torsions_total] (f64)
        torsion_v2[num_torsions_total] (f64)
        torsion_v3[num_torsions_total] (f64)

    Nonbonded data:
        nb_i[num_nb_total] (u32)
        nb_j[num_nb_total] (u32)
        nb_is_1_4[num_nb_total] (u8)
        nb_epsilon[num_nb_total] (f64)
        nb_r_star[num_nb_total] (f64)
        nb_qi[num_nb_total] (f64)
        nb_qj[num_nb_total] (f64)

    Returns:
        Dict with counts for verification
    """
    num_molecules = len(molecules)

    # Count totals
    num_atoms_total = sum(m.num_atoms for m in molecules)
    num_bonds_total = sum(len(m.bonds) for m in molecules)
    num_angles_total = sum(len(m.angles) for m in molecules)
    num_torsions_total = sum(len(m.torsions) for m in molecules)
    num_nb_total = sum(len(m.nonbonded) for m in molecules)

    # Build offset arrays (CSR-style)
    atom_starts = np.zeros(num_molecules + 1, dtype=np.uint32)
    bond_starts = np.zeros(num_molecules + 1, dtype=np.uint32)
    angle_starts = np.zeros(num_molecules + 1, dtype=np.uint32)
    torsion_starts = np.zeros(num_molecules + 1, dtype=np.uint32)
    nb_starts = np.zeros(num_molecules + 1, dtype=np.uint32)

    for i, mol in enumerate(molecules):
        atom_starts[i + 1] = atom_starts[i] + mol.num_atoms
        bond_starts[i + 1] = bond_starts[i] + len(mol.bonds)
        angle_starts[i + 1] = angle_starts[i] + len(mol.angles)
        torsion_starts[i + 1] = torsion_starts[i] + len(mol.torsions)
        nb_starts[i + 1] = nb_starts[i] + len(mol.nonbonded)

    # Allocate data arrays
    x = np.zeros(num_atoms_total, dtype=np.float64)
    y = np.zeros(num_atoms_total, dtype=np.float64)
    z = np.zeros(num_atoms_total, dtype=np.float64)
    charge = np.zeros(num_atoms_total, dtype=np.float64)

    bond_i = np.zeros(num_bonds_total, dtype=np.uint32)
    bond_j = np.zeros(num_bonds_total, dtype=np.uint32)
    bond_kb = np.zeros(num_bonds_total, dtype=np.float64)
    bond_r0 = np.zeros(num_bonds_total, dtype=np.float64)

    angle_i = np.zeros(num_angles_total, dtype=np.uint32)
    angle_j = np.zeros(num_angles_total, dtype=np.uint32)
    angle_k = np.zeros(num_angles_total, dtype=np.uint32)
    angle_ka = np.zeros(num_angles_total, dtype=np.float64)
    angle_theta0 = np.zeros(num_angles_total, dtype=np.float64)

    torsion_i = np.zeros(num_torsions_total, dtype=np.uint32)
    torsion_j = np.zeros(num_torsions_total, dtype=np.uint32)
    torsion_k = np.zeros(num_torsions_total, dtype=np.uint32)
    torsion_l = np.zeros(num_torsions_total, dtype=np.uint32)
    torsion_v1 = np.zeros(num_torsions_total, dtype=np.float64)
    torsion_v2 = np.zeros(num_torsions_total, dtype=np.float64)
    torsion_v3 = np.zeros(num_torsions_total, dtype=np.float64)

    nb_i = np.zeros(num_nb_total, dtype=np.uint32)
    nb_j = np.zeros(num_nb_total, dtype=np.uint32)
    nb_is_1_4 = np.zeros(num_nb_total, dtype=np.uint8)
    nb_epsilon = np.zeros(num_nb_total, dtype=np.float64)
    nb_r_star = np.zeros(num_nb_total, dtype=np.float64)
    nb_qi = np.zeros(num_nb_total, dtype=np.float64)
    nb_qj = np.zeros(num_nb_total, dtype=np.float64)

    # Fill data arrays
    for mol_idx, mol in enumerate(molecules):
        # Atoms
        atom_off = atom_starts[mol_idx]
        for a in range(mol.num_atoms):
            x[atom_off + a] = mol.positions[a, 0]
            y[atom_off + a] = mol.positions[a, 1]
            z[atom_off + a] = mol.positions[a, 2]
            charge[atom_off + a] = mol.charges[a]

        # Bonds
        bond_off = bond_starts[mol_idx]
        for b_idx, bond in enumerate(mol.bonds):
            bond_i[bond_off + b_idx] = bond.i
            bond_j[bond_off + b_idx] = bond.j
            bond_kb[bond_off + b_idx] = bond.kb
            bond_r0[bond_off + b_idx] = bond.r0

        # Angles
        angle_off = angle_starts[mol_idx]
        for a_idx, angle in enumerate(mol.angles):
            angle_i[angle_off + a_idx] = angle.i
            angle_j[angle_off + a_idx] = angle.j
            angle_k[angle_off + a_idx] = angle.k
            angle_ka[angle_off + a_idx] = angle.ka
            angle_theta0[angle_off + a_idx] = angle.theta0

        # Torsions
        torsion_off = torsion_starts[mol_idx]
        for t_idx, tors in enumerate(mol.torsions):
            torsion_i[torsion_off + t_idx] = tors.i
            torsion_j[torsion_off + t_idx] = tors.j
            torsion_k[torsion_off + t_idx] = tors.k
            torsion_l[torsion_off + t_idx] = tors.l
            torsion_v1[torsion_off + t_idx] = tors.v1
            torsion_v2[torsion_off + t_idx] = tors.v2
            torsion_v3[torsion_off + t_idx] = tors.v3

        # Nonbonded
        nb_off = nb_starts[mol_idx]
        for nb_idx, nb in enumerate(mol.nonbonded):
            nb_i[nb_off + nb_idx] = nb.i
            nb_j[nb_off + nb_idx] = nb.j
            nb_is_1_4[nb_off + nb_idx] = 1 if nb.is_1_4 else 0
            nb_epsilon[nb_off + nb_idx] = nb.epsilon
            nb_r_star[nb_off + nb_idx] = nb.r_star
            nb_qi[nb_off + nb_idx] = nb.qi
            nb_qj[nb_off + nb_idx] = nb.qj

    # Write binary file
    with open(output_path, "wb") as f:
        # Header (64 bytes)
        header = struct.pack(
            "<IIIIIIII",  # 8 x u32 = 32 bytes
            MAGIC,
            VERSION,
            num_molecules,
            num_atoms_total,
            num_bonds_total,
            num_angles_total,
            num_torsions_total,
            num_nb_total,
        )
        f.write(header)
        f.write(b"\x00" * 32)  # Padding to 64 bytes

        # Offset arrays
        f.write(atom_starts.tobytes())
        f.write(bond_starts.tobytes())
        f.write(angle_starts.tobytes())
        f.write(torsion_starts.tobytes())
        f.write(nb_starts.tobytes())

        # Atom data
        f.write(x.tobytes())
        f.write(y.tobytes())
        f.write(z.tobytes())
        f.write(charge.tobytes())

        # Bond data
        f.write(bond_i.tobytes())
        f.write(bond_j.tobytes())
        f.write(bond_kb.tobytes())
        f.write(bond_r0.tobytes())

        # Angle data
        f.write(angle_i.tobytes())
        f.write(angle_j.tobytes())
        f.write(angle_k.tobytes())
        f.write(angle_ka.tobytes())
        f.write(angle_theta0.tobytes())

        # Torsion data
        f.write(torsion_i.tobytes())
        f.write(torsion_j.tobytes())
        f.write(torsion_k.tobytes())
        f.write(torsion_l.tobytes())
        f.write(torsion_v1.tobytes())
        f.write(torsion_v2.tobytes())
        f.write(torsion_v3.tobytes())

        # Nonbonded data
        f.write(nb_i.tobytes())
        f.write(nb_j.tobytes())
        f.write(nb_is_1_4.tobytes())
        f.write(nb_epsilon.tobytes())
        f.write(nb_r_star.tobytes())
        f.write(nb_qi.tobytes())
        f.write(nb_qj.tobytes())

    file_size = Path(output_path).stat().st_size

    return {
        "num_molecules": num_molecules,
        "num_atoms": num_atoms_total,
        "num_bonds": num_bonds_total,
        "num_angles": num_angles_total,
        "num_torsions": num_torsions_total,
        "num_nb": num_nb_total,
        "file_size_bytes": file_size,
        "file_size_mb": file_size / (1024 * 1024),
    }


if __name__ == "__main__":
    from molecule_loader import load_molecules, DEFAULT_SMILES_PATH
    import tempfile

    # Quick test
    molecules = load_molecules(DEFAULT_SMILES_PATH, n_samples=100)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        output_path = f.name

    stats = export_binary(molecules, output_path)

    print("\nBinary export statistics:")
    print(f"  Molecules: {stats['num_molecules']}")
    print(f"  Atoms: {stats['num_atoms']}")
    print(f"  Bonds: {stats['num_bonds']}")
    print(f"  Angles: {stats['num_angles']}")
    print(f"  Torsions: {stats['num_torsions']}")
    print(f"  Nonbonded pairs: {stats['num_nb']}")
    print(f"  File size: {stats['file_size_mb']:.2f} MB")
    print(f"  Output: {output_path}")
