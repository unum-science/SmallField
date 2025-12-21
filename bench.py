#!/usr/bin/env python3
"""
bench.py - Geometry Optimization Benchmark for SmallField MMFF94.

Benchmarks full geometry optimization (not just energy calculation):
- Throughput: molecules optimized per second vs RDKit baseline
- Accuracy: final energies and coordinates vs RDKit and F64 serial

Usage:
    python bench.py --n-molecules 1000 --max-iter 200
"""

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Get CPU core count for theoretical multi-core throughput
CPU_CORES = os.cpu_count() or 1

def get_omp_threads() -> int:
    """Get OpenMP thread count (used by RDKit)."""
    # Check environment variable first
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_threads:
        return int(omp_threads)
    # Default to CPU count if not set
    return CPU_CORES

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("ERROR: RDKit not available. Install with: pip install rdkit")
    raise SystemExit(1)

try:
    import smallfield
except ImportError:
    print("ERROR: SmallField not available. Build with: uv pip install -e .")
    raise SystemExit(1)


DEFAULT_SMILES_PATH = (
    Path(__file__).parent.parent / "USearchMolecules" / "data" / "example" / "smiles"
)


@dataclass
class MoleculeData:
    """Container for molecule coordinates and force field parameters."""

    mol: Chem.Mol
    n_atoms: int
    n_pairs: int
    coords_x: np.ndarray  # f64
    coords_y: np.ndarray  # f64
    coords_z: np.ndarray  # f64
    pair_i: np.ndarray  # i32
    pair_j: np.ndarray  # i32
    epsilon: np.ndarray  # f64
    r_star: np.ndarray  # f64

    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> "MoleculeData":
        """Create MoleculeData from RDKit Mol object."""
        n_atoms = mol.GetNumAtoms()
        conf = mol.GetConformer(-1)

        coords_x = np.array(
            [conf.GetAtomPosition(i).x for i in range(n_atoms)], dtype=np.float64
        )
        coords_y = np.array(
            [conf.GetAtomPosition(i).y for i in range(n_atoms)], dtype=np.float64
        )
        coords_z = np.array(
            [conf.GetAtomPosition(i).z for i in range(n_atoms)], dtype=np.float64
        )

        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is None:
            raise ValueError("Could not get MMFF properties")

        pair_i_list, pair_j_list, epsilon_list, r_star_list = [], [], [], []

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if mol.GetBondBetweenAtoms(i, j) is not None:
                    continue
                vdw_params = mmff_props.GetMMFFVdWParams(i, j)
                if vdw_params is not None:
                    pair_i_list.append(i)
                    pair_j_list.append(j)
                    epsilon_list.append(vdw_params[1])
                    r_star_list.append(vdw_params[0])

        n_pairs = len(pair_i_list)

        # Pad to multiple of 32 for SIMD
        padding = (32 - n_pairs % 32) % 32
        total_pairs = n_pairs + padding

        pair_i = np.zeros(total_pairs, dtype=np.int32)
        pair_j = np.zeros(total_pairs, dtype=np.int32)
        epsilon = np.zeros(total_pairs, dtype=np.float64)
        r_star = np.zeros(total_pairs, dtype=np.float64)

        for k in range(n_pairs):
            pair_i[k] = pair_i_list[k]
            pair_j[k] = pair_j_list[k]
            epsilon[k] = epsilon_list[k]
            r_star[k] = r_star_list[k]

        # Pad with safe values
        if padding > 0 and n_pairs > 0:
            for k in range(n_pairs, total_pairs):
                pair_i[k] = pair_i[n_pairs - 1]
                pair_j[k] = pair_j[n_pairs - 1]
                epsilon[k] = 0.0
                r_star[k] = 1.0

        return cls(
            mol=mol,
            n_atoms=n_atoms,
            n_pairs=n_pairs,
            coords_x=coords_x,
            coords_y=coords_y,
            coords_z=coords_z,
            pair_i=pair_i,
            pair_j=pair_j,
            epsilon=epsilon,
            r_star=r_star,
        )


@dataclass
class MMFF94MolData:
    """Full MMFF94 force field parameters for a molecule."""

    mol: Chem.Mol
    n_atoms: int
    coords_x: np.ndarray
    coords_y: np.ndarray
    coords_z: np.ndarray
    # VdW pairs
    vdw_pair_i: np.ndarray
    vdw_pair_j: np.ndarray
    vdw_epsilon: np.ndarray
    vdw_r_star: np.ndarray
    # Electrostatic pairs
    elec_pair_i: np.ndarray
    elec_pair_j: np.ndarray
    elec_charge_i: np.ndarray
    elec_charge_j: np.ndarray
    elec_is_1_4: np.ndarray
    # Bonds
    bond_atom_i: np.ndarray
    bond_atom_j: np.ndarray
    bond_kb: np.ndarray
    bond_r0: np.ndarray
    # Angles
    angle_atom_i: np.ndarray
    angle_atom_j: np.ndarray
    angle_atom_k: np.ndarray
    angle_ka: np.ndarray
    angle_theta0: np.ndarray
    # Torsions
    torsion_atom_i: np.ndarray
    torsion_atom_j: np.ndarray
    torsion_atom_k: np.ndarray
    torsion_atom_l: np.ndarray
    torsion_v1: np.ndarray
    torsion_v2: np.ndarray
    torsion_v3: np.ndarray
    # Stretch-bend
    stbn_atom_i: np.ndarray
    stbn_atom_j: np.ndarray
    stbn_atom_k: np.ndarray
    stbn_kba_ijk: np.ndarray
    stbn_kba_kji: np.ndarray
    stbn_r0_ij: np.ndarray
    stbn_r0_kj: np.ndarray
    stbn_theta0: np.ndarray
    # Out-of-plane
    oop_atom_i: np.ndarray
    oop_atom_j: np.ndarray
    oop_atom_k: np.ndarray
    oop_atom_l: np.ndarray
    oop_koop: np.ndarray

    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> "MMFF94MolData":
        """Extract all MMFF94 parameters from RDKit Mol object."""
        n_atoms = mol.GetNumAtoms()
        conf = mol.GetConformer(-1)

        coords_x = np.array(
            [conf.GetAtomPosition(i).x for i in range(n_atoms)], dtype=np.float64
        )
        coords_y = np.array(
            [conf.GetAtomPosition(i).y for i in range(n_atoms)], dtype=np.float64
        )
        coords_z = np.array(
            [conf.GetAtomPosition(i).z for i in range(n_atoms)], dtype=np.float64
        )

        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is None:
            raise ValueError("Could not get MMFF properties")

        # Build exclusion sets:
        # - 1-2 pairs (bonded): excluded from VdW/elec
        # - 1-3 pairs (share an angle): excluded from VdW/elec
        # - 1-4 pairs (share a torsion): included with scaling
        excluded_pairs = set()  # 1-2 and 1-3 pairs
        is_1_4_pair = set()  # 1-4 pairs

        # 1-2 pairs (directly bonded)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            excluded_pairs.add((min(i, j), max(i, j)))

        # 1-3 pairs (share an angle - neighbors of neighbors)
        for atom in mol.GetAtoms():
            j = atom.GetIdx()
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            for idx_i in range(len(neighbors)):
                for idx_k in range(idx_i + 1, len(neighbors)):
                    i, k = neighbors[idx_i], neighbors[idx_k]
                    excluded_pairs.add((min(i, k), max(i, k)))

        # 1-4 pairs (share a torsion - atoms at ends of torsions)
        for bond in mol.GetBonds():
            j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            j_neighbors = [
                n.GetIdx()
                for n in mol.GetAtomWithIdx(j).GetNeighbors()
                if n.GetIdx() != k
            ]
            k_neighbors = [
                n.GetIdx()
                for n in mol.GetAtomWithIdx(k).GetNeighbors()
                if n.GetIdx() != j
            ]
            for i in j_neighbors:
                for l in k_neighbors:
                    if i != l:  # Avoid same atom
                        pair = (min(i, l), max(i, l))
                        if pair not in excluded_pairs:
                            is_1_4_pair.add(pair)

        # VdW pairs (exclude 1-2 and 1-3 pairs)
        vdw_i, vdw_j, vdw_eps, vdw_rs = [], [], [], []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                pair = (i, j)
                if pair in excluded_pairs:
                    continue
                params = mmff_props.GetMMFFVdWParams(i, j)
                if params:
                    vdw_i.append(i)
                    vdw_j.append(j)
                    vdw_eps.append(params[1])
                    vdw_rs.append(params[0])

        # Electrostatic pairs (exclude 1-2 and 1-3 pairs, mark 1-4 for scaling)
        elec_i, elec_j, elec_qi, elec_qj, elec_14 = [], [], [], [], []
        for i in range(n_atoms):
            qi = mmff_props.GetMMFFPartialCharge(i)
            for j in range(i + 1, n_atoms):
                pair = (i, j)
                if pair in excluded_pairs:
                    continue
                qj = mmff_props.GetMMFFPartialCharge(j)
                elec_i.append(i)
                elec_j.append(j)
                elec_qi.append(qi)
                elec_qj.append(qj)
                elec_14.append(1 if pair in is_1_4_pair else 0)

        # Bonds
        bond_i, bond_j, bond_kb_list, bond_r0_list = [], [], [], []
        bond_params_cache = {}
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            params = mmff_props.GetMMFFBondStretchParams(mol, i, j)
            if params:
                bond_i.append(i)
                bond_j.append(j)
                bond_kb_list.append(params[1])
                bond_r0_list.append(params[2])
                bond_params_cache[(i, j)] = params[2]
                bond_params_cache[(j, i)] = params[2]

        # Angles
        ang_i, ang_j, ang_k, ang_ka_list, ang_theta0_list = [], [], [], [], []
        angle_params_cache = {}
        for atom in mol.GetAtoms():
            j = atom.GetIdx()
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            for idx_i in range(len(neighbors)):
                for idx_k in range(idx_i + 1, len(neighbors)):
                    i, k = neighbors[idx_i], neighbors[idx_k]
                    params = mmff_props.GetMMFFAngleBendParams(mol, i, j, k)
                    if params:
                        ang_i.append(i)
                        ang_j.append(j)
                        ang_k.append(k)
                        ang_ka_list.append(params[1])
                        ang_theta0_list.append(params[2])
                        angle_params_cache[(i, j, k)] = params[2]
                        angle_params_cache[(k, j, i)] = params[2]

        # Torsions
        tor_i, tor_j, tor_k, tor_l = [], [], [], []
        tor_v1, tor_v2, tor_v3 = [], [], []
        for bond in mol.GetBonds():
            j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            j_neighbors = [
                n.GetIdx()
                for n in mol.GetAtomWithIdx(j).GetNeighbors()
                if n.GetIdx() != k
            ]
            k_neighbors = [
                n.GetIdx()
                for n in mol.GetAtomWithIdx(k).GetNeighbors()
                if n.GetIdx() != j
            ]
            for i in j_neighbors:
                for l in k_neighbors:
                    params = mmff_props.GetMMFFTorsionParams(mol, i, j, k, l)
                    if params:
                        tor_i.append(i)
                        tor_j.append(j)
                        tor_k.append(k)
                        tor_l.append(l)
                        tor_v1.append(params[1])
                        tor_v2.append(params[2])
                        tor_v3.append(params[3])

        # Stretch-bend
        stbn_i, stbn_j, stbn_k = [], [], []
        stbn_kij, stbn_kji, stbn_rij, stbn_rkj, stbn_t0 = [], [], [], [], []
        for atom in mol.GetAtoms():
            j = atom.GetIdx()
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            for idx_i in range(len(neighbors)):
                for idx_k in range(idx_i + 1, len(neighbors)):
                    i, k = neighbors[idx_i], neighbors[idx_k]
                    params = mmff_props.GetMMFFStretchBendParams(mol, i, j, k)
                    if params:
                        r0_ij = bond_params_cache.get((i, j), 1.5)
                        r0_kj = bond_params_cache.get((k, j), 1.5)
                        theta0 = angle_params_cache.get((i, j, k), 109.5)
                        stbn_i.append(i)
                        stbn_j.append(j)
                        stbn_k.append(k)
                        stbn_kij.append(params[1])
                        stbn_kji.append(params[2])
                        stbn_rij.append(r0_ij)
                        stbn_rkj.append(r0_kj)
                        stbn_t0.append(theta0)

        # Out-of-plane bending (sp2 centers)
        oop_i, oop_j, oop_k, oop_l, oop_ko = [], [], [], [], []
        for atom in mol.GetAtoms():
            if atom.GetHybridization() == Chem.HybridizationType.SP2:
                j = atom.GetIdx()
                neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
                if len(neighbors) == 3:
                    i, k, l = neighbors
                    params = mmff_props.GetMMFFOopBendParams(mol, i, j, k, l)
                    if params:
                        oop_i.append(i)
                        oop_j.append(j)
                        oop_k.append(k)
                        oop_l.append(l)
                        oop_ko.append(params)

        return cls(
            mol=mol,
            n_atoms=n_atoms,
            coords_x=coords_x,
            coords_y=coords_y,
            coords_z=coords_z,
            vdw_pair_i=np.array(vdw_i, dtype=np.int32),
            vdw_pair_j=np.array(vdw_j, dtype=np.int32),
            vdw_epsilon=np.array(vdw_eps, dtype=np.float64),
            vdw_r_star=np.array(vdw_rs, dtype=np.float64),
            elec_pair_i=np.array(elec_i, dtype=np.int32),
            elec_pair_j=np.array(elec_j, dtype=np.int32),
            elec_charge_i=np.array(elec_qi, dtype=np.float64),
            elec_charge_j=np.array(elec_qj, dtype=np.float64),
            elec_is_1_4=np.array(elec_14, dtype=np.int32),
            bond_atom_i=np.array(bond_i, dtype=np.int32),
            bond_atom_j=np.array(bond_j, dtype=np.int32),
            bond_kb=np.array(bond_kb_list, dtype=np.float64),
            bond_r0=np.array(bond_r0_list, dtype=np.float64),
            angle_atom_i=np.array(ang_i, dtype=np.int32),
            angle_atom_j=np.array(ang_j, dtype=np.int32),
            angle_atom_k=np.array(ang_k, dtype=np.int32),
            angle_ka=np.array(ang_ka_list, dtype=np.float64),
            angle_theta0=np.array(ang_theta0_list, dtype=np.float64),
            torsion_atom_i=np.array(tor_i, dtype=np.int32),
            torsion_atom_j=np.array(tor_j, dtype=np.int32),
            torsion_atom_k=np.array(tor_k, dtype=np.int32),
            torsion_atom_l=np.array(tor_l, dtype=np.int32),
            torsion_v1=np.array(tor_v1, dtype=np.float64),
            torsion_v2=np.array(tor_v2, dtype=np.float64),
            torsion_v3=np.array(tor_v3, dtype=np.float64),
            stbn_atom_i=np.array(stbn_i, dtype=np.int32),
            stbn_atom_j=np.array(stbn_j, dtype=np.int32),
            stbn_atom_k=np.array(stbn_k, dtype=np.int32),
            stbn_kba_ijk=np.array(stbn_kij, dtype=np.float64),
            stbn_kba_kji=np.array(stbn_kji, dtype=np.float64),
            stbn_r0_ij=np.array(stbn_rij, dtype=np.float64),
            stbn_r0_kj=np.array(stbn_rkj, dtype=np.float64),
            stbn_theta0=np.array(stbn_t0, dtype=np.float64),
            oop_atom_i=np.array(oop_i, dtype=np.int32),
            oop_atom_j=np.array(oop_j, dtype=np.int32),
            oop_atom_k=np.array(oop_k, dtype=np.int32),
            oop_atom_l=np.array(oop_l, dtype=np.int32),
            oop_koop=np.array(oop_ko, dtype=np.float64),
        )


@dataclass
class OptimizationResult:
    """Result of geometry optimization."""

    final_energy: float
    final_coords_x: np.ndarray
    final_coords_y: np.ndarray
    final_coords_z: np.ndarray
    iterations: int
    converged: bool
    time_ms: float


@dataclass
class BenchmarkResult:
    """Benchmark results for a single backend."""

    backend: str
    time_ms: float
    mol_per_sec: float
    energies: np.ndarray
    coords_x: List[np.ndarray] = field(default_factory=list)
    coords_y: List[np.ndarray] = field(default_factory=list)
    coords_z: List[np.ndarray] = field(default_factory=list)
    iter_counts: List[int] = field(default_factory=list)


def load_smiles_from_file(file_path: Path, n_molecules: int) -> List[str]:
    """Load SMILES strings from a file."""
    smiles_list = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                smiles = line.split()[0] if " " in line or "\t" in line else line
                smiles_list.append(smiles)
                if len(smiles_list) >= n_molecules:
                    break
    return smiles_list


def load_molecules(n_molecules: int, smiles_dir: Path) -> List[MoleculeData]:
    """Load molecules from SMILES directory."""
    smiles_files = sorted(smiles_dir.glob("*.smi"))

    if not smiles_files:
        print(f"No SMILES files found in {smiles_dir}")
        print("Using default test molecules...")
        return load_default_molecules(n_molecules)

    print(f"Loading {n_molecules} molecules from {smiles_dir}...")

    all_smiles = []
    for smi_file in smiles_files:
        all_smiles.extend(
            load_smiles_from_file(smi_file, n_molecules - len(all_smiles))
        )
        if len(all_smiles) >= n_molecules:
            break

    mol_data_list = []
    failed = 0

    for smiles in all_smiles[:n_molecules]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed += 1
                continue
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            mol_data = MoleculeData.from_mol(mol)
            if mol_data.n_pairs > 0:
                mol_data_list.append(mol_data)
        except Exception:
            failed += 1
            continue

    print(f"Loaded {len(mol_data_list)} molecules ({failed} failed)")
    return mol_data_list


def load_default_molecules(n_molecules: int) -> List[MoleculeData]:
    """Load default test molecules."""
    default_smiles = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC(C)(C)c1ccc(O)cc1",  # BHT
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    ]

    mol_data_list = []
    idx = 0
    while len(mol_data_list) < n_molecules:
        smiles = default_smiles[idx % len(default_smiles)]
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        mol_data_list.append(MoleculeData.from_mol(mol))
        idx += 1

    return mol_data_list


def compute_rmsd(
    coords1_x: np.ndarray,
    coords1_y: np.ndarray,
    coords1_z: np.ndarray,
    coords2_x: np.ndarray,
    coords2_y: np.ndarray,
    coords2_z: np.ndarray,
) -> float:
    """Compute RMSD between two coordinate sets."""
    dx = coords1_x - coords2_x
    dy = coords1_y - coords2_y
    dz = coords1_z - coords2_z
    return float(np.sqrt(np.mean(dx * dx + dy * dy + dz * dz)))


# =============================================================================
# Optimization Backends
# =============================================================================


def optimize_rdkit(
    mol_data_list: List[MoleculeData], max_iterations: int
) -> BenchmarkResult:
    """Optimize molecules using RDKit MMFF94 (baseline)."""
    n_molecules = len(mol_data_list)
    energies = np.zeros(n_molecules, dtype=np.float64)
    coords_x_list = []
    coords_y_list = []
    coords_z_list = []

    # Warmup
    for m in mol_data_list[: min(10, n_molecules)]:
        mol_copy = Chem.Mol(m.mol)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol_copy, AllChem.MMFFGetMoleculeProperties(mol_copy)
        )
        if ff:
            ff.Minimize(maxIts=max_iterations)

    start = time.perf_counter()

    for i, m in enumerate(mol_data_list):
        mol_copy = Chem.Mol(m.mol)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol_copy, AllChem.MMFFGetMoleculeProperties(mol_copy)
        )
        if ff:
            ff.Minimize(maxIts=max_iterations)
            energies[i] = ff.CalcEnergy()

            conf = mol_copy.GetConformer(-1)
            n_atoms = mol_copy.GetNumAtoms()
            coords_x_list.append(
                np.array([conf.GetAtomPosition(j).x for j in range(n_atoms)])
            )
            coords_y_list.append(
                np.array([conf.GetAtomPosition(j).y for j in range(n_atoms)])
            )
            coords_z_list.append(
                np.array([conf.GetAtomPosition(j).z for j in range(n_atoms)])
            )

    end = time.perf_counter()
    time_ms = (end - start) * 1000
    mol_per_sec = n_molecules / (time_ms / 1000)

    return BenchmarkResult(
        backend="RDKit (baseline)",
        time_ms=time_ms,
        mol_per_sec=mol_per_sec,
        energies=energies,
        coords_x=coords_x_list,
        coords_y=coords_y_list,
        coords_z=coords_z_list,
    )


def optimize_serial_f64(
    mol_data_list: List[MMFF94MolData], max_iterations: int, grad_threshold: float
) -> Optional[BenchmarkResult]:
    """Optimize molecules using SmallField serial F64 with full MMFF94."""
    n_molecules = len(mol_data_list)
    max_atoms = max(m.n_atoms for m in mol_data_list)
    max_pairs = max(len(m.vdw_pair_i) for m in mol_data_list)

    ctx = smallfield.Context(max_atoms, max_pairs, 1)

    energies = np.zeros(n_molecules, dtype=np.float64)
    coords_x_list = []
    coords_y_list = []
    coords_z_list = []

    # Warmup
    for m in mol_data_list[: min(5, n_molecules)]:
        cx, cy, cz = m.coords_x.copy(), m.coords_y.copy(), m.coords_z.copy()
        try:
            ctx.optimize_serial_f64(
                cx,
                cy,
                cz,
                m.vdw_pair_i,
                m.vdw_pair_j,
                m.vdw_epsilon,
                m.vdw_r_star,
                m.elec_pair_i,
                m.elec_pair_j,
                m.elec_charge_i,
                m.elec_charge_j,
                m.elec_is_1_4,
                m.bond_atom_i,
                m.bond_atom_j,
                m.bond_kb,
                m.bond_r0,
                m.angle_atom_i,
                m.angle_atom_j,
                m.angle_atom_k,
                m.angle_ka,
                m.angle_theta0,
                m.torsion_atom_i,
                m.torsion_atom_j,
                m.torsion_atom_k,
                m.torsion_atom_l,
                m.torsion_v1,
                m.torsion_v2,
                m.torsion_v3,
                m.stbn_atom_i,
                m.stbn_atom_j,
                m.stbn_atom_k,
                m.stbn_kba_ijk,
                m.stbn_kba_kji,
                m.stbn_r0_ij,
                m.stbn_r0_kj,
                m.stbn_theta0,
                m.oop_atom_i,
                m.oop_atom_j,
                m.oop_atom_k,
                m.oop_atom_l,
                m.oop_koop,
                max_iterations,
                grad_threshold,
            )
        except Exception:
            pass

    start = time.perf_counter()

    iter_counts = []
    for i, m in enumerate(mol_data_list):
        cx, cy, cz = m.coords_x.copy(), m.coords_y.copy(), m.coords_z.copy()
        try:
            result = ctx.optimize_serial_f64(
                cx,
                cy,
                cz,
                m.vdw_pair_i,
                m.vdw_pair_j,
                m.vdw_epsilon,
                m.vdw_r_star,
                m.elec_pair_i,
                m.elec_pair_j,
                m.elec_charge_i,
                m.elec_charge_j,
                m.elec_is_1_4,
                m.bond_atom_i,
                m.bond_atom_j,
                m.bond_kb,
                m.bond_r0,
                m.angle_atom_i,
                m.angle_atom_j,
                m.angle_atom_k,
                m.angle_ka,
                m.angle_theta0,
                m.torsion_atom_i,
                m.torsion_atom_j,
                m.torsion_atom_k,
                m.torsion_atom_l,
                m.torsion_v1,
                m.torsion_v2,
                m.torsion_v3,
                m.stbn_atom_i,
                m.stbn_atom_j,
                m.stbn_atom_k,
                m.stbn_kba_ijk,
                m.stbn_kba_kji,
                m.stbn_r0_ij,
                m.stbn_r0_kj,
                m.stbn_theta0,
                m.oop_atom_i,
                m.oop_atom_j,
                m.oop_atom_k,
                m.oop_atom_l,
                m.oop_koop,
                max_iterations,
                grad_threshold,
            )
            energies[i] = result[0]
            iter_counts.append(result[1])
            coords_x_list.append(cx)
            coords_y_list.append(cy)
            coords_z_list.append(cz)
        except Exception as e:
            print(f"Serial F64 failed for molecule {i}: {e}")
            energies[i] = float("nan")
            iter_counts.append(max_iterations)
            coords_x_list.append(m.coords_x.copy())
            coords_y_list.append(m.coords_y.copy())
            coords_z_list.append(m.coords_z.copy())

    end = time.perf_counter()
    time_ms = (end - start) * 1000
    mol_per_sec = n_molecules / (time_ms / 1000)

    return BenchmarkResult(
        backend="Serial F64",
        time_ms=time_ms,
        mol_per_sec=mol_per_sec,
        energies=energies,
        coords_x=coords_x_list,
        coords_y=coords_y_list,
        coords_z=coords_z_list,
        iter_counts=iter_counts,
    )


def optimize_serial_f32(
    mol_data_list: List[MMFF94MolData], max_iterations: int, grad_threshold: float
) -> Optional[BenchmarkResult]:
    """Optimize molecules using SmallField serial F32 with full MMFF94."""
    n_molecules = len(mol_data_list)
    max_atoms = max(m.n_atoms for m in mol_data_list)
    max_pairs = max(len(m.vdw_pair_i) for m in mol_data_list)

    ctx = smallfield.Context(max_atoms, max_pairs, 1)

    energies = np.zeros(n_molecules, dtype=np.float64)
    coords_x_list = []
    coords_y_list = []
    coords_z_list = []

    # Warmup
    for m in mol_data_list[: min(5, n_molecules)]:
        cx, cy, cz = m.coords_x.copy(), m.coords_y.copy(), m.coords_z.copy()
        try:
            ctx.optimize_serial_f32(
                cx,
                cy,
                cz,
                m.vdw_pair_i,
                m.vdw_pair_j,
                m.vdw_epsilon,
                m.vdw_r_star,
                m.elec_pair_i,
                m.elec_pair_j,
                m.elec_charge_i,
                m.elec_charge_j,
                m.elec_is_1_4,
                m.bond_atom_i,
                m.bond_atom_j,
                m.bond_kb,
                m.bond_r0,
                m.angle_atom_i,
                m.angle_atom_j,
                m.angle_atom_k,
                m.angle_ka,
                m.angle_theta0,
                m.torsion_atom_i,
                m.torsion_atom_j,
                m.torsion_atom_k,
                m.torsion_atom_l,
                m.torsion_v1,
                m.torsion_v2,
                m.torsion_v3,
                m.stbn_atom_i,
                m.stbn_atom_j,
                m.stbn_atom_k,
                m.stbn_kba_ijk,
                m.stbn_kba_kji,
                m.stbn_r0_ij,
                m.stbn_r0_kj,
                m.stbn_theta0,
                m.oop_atom_i,
                m.oop_atom_j,
                m.oop_atom_k,
                m.oop_atom_l,
                m.oop_koop,
                max_iterations,
                grad_threshold,
            )
        except Exception:
            pass

    start = time.perf_counter()

    for i, m in enumerate(mol_data_list):
        cx, cy, cz = m.coords_x.copy(), m.coords_y.copy(), m.coords_z.copy()
        try:
            result = ctx.optimize_serial_f32(
                cx,
                cy,
                cz,
                m.vdw_pair_i,
                m.vdw_pair_j,
                m.vdw_epsilon,
                m.vdw_r_star,
                m.elec_pair_i,
                m.elec_pair_j,
                m.elec_charge_i,
                m.elec_charge_j,
                m.elec_is_1_4,
                m.bond_atom_i,
                m.bond_atom_j,
                m.bond_kb,
                m.bond_r0,
                m.angle_atom_i,
                m.angle_atom_j,
                m.angle_atom_k,
                m.angle_ka,
                m.angle_theta0,
                m.torsion_atom_i,
                m.torsion_atom_j,
                m.torsion_atom_k,
                m.torsion_atom_l,
                m.torsion_v1,
                m.torsion_v2,
                m.torsion_v3,
                m.stbn_atom_i,
                m.stbn_atom_j,
                m.stbn_atom_k,
                m.stbn_kba_ijk,
                m.stbn_kba_kji,
                m.stbn_r0_ij,
                m.stbn_r0_kj,
                m.stbn_theta0,
                m.oop_atom_i,
                m.oop_atom_j,
                m.oop_atom_k,
                m.oop_atom_l,
                m.oop_koop,
                max_iterations,
                grad_threshold,
            )
            energies[i] = result[0]
            coords_x_list.append(cx)
            coords_y_list.append(cy)
            coords_z_list.append(cz)
        except Exception as e:
            print(f"Serial F32 failed for molecule {i}: {e}")
            energies[i] = float("nan")
            coords_x_list.append(m.coords_x.copy())
            coords_y_list.append(m.coords_y.copy())
            coords_z_list.append(m.coords_z.copy())

    end = time.perf_counter()
    time_ms = (end - start) * 1000
    mol_per_sec = n_molecules / (time_ms / 1000)

    return BenchmarkResult(
        backend="Serial F32",
        time_ms=time_ms,
        mol_per_sec=mol_per_sec,
        energies=energies,
        coords_x=coords_x_list,
        coords_y=coords_y_list,
        coords_z=coords_z_list,
    )


def optimize_skylake(
    mol_data_list: List[MoleculeData], max_iterations: int, grad_threshold: float
) -> Optional[BenchmarkResult]:
    """Optimize molecules using SmallField Skylake AVX-512 F32."""
    if not smallfield.has_skylake():
        return None

    n_molecules = len(mol_data_list)
    max_atoms = max(m.n_atoms for m in mol_data_list)
    max_pairs = max(len(m.pair_i) for m in mol_data_list)

    ctx = smallfield.Context(max_atoms, max_pairs, 1)

    energies = np.zeros(n_molecules, dtype=np.float64)
    coords_x_list = []
    coords_y_list = []
    coords_z_list = []

    # Warmup
    for m in mol_data_list[: min(5, n_molecules)]:
        cx, cy, cz = m.coords_x.copy(), m.coords_y.copy(), m.coords_z.copy()
        try:
            ctx.optimize_skylake(
                cx,
                cy,
                cz,
                m.pair_i,
                m.pair_j,
                m.epsilon,
                m.r_star,
                max_iterations,
                grad_threshold,
            )
        except Exception:
            pass

    start = time.perf_counter()

    for i, m in enumerate(mol_data_list):
        cx = m.coords_x.copy()
        cy = m.coords_y.copy()
        cz = m.coords_z.copy()

        try:
            result = ctx.optimize_skylake(
                cx,
                cy,
                cz,
                m.pair_i,
                m.pair_j,
                m.epsilon,
                m.r_star,
                max_iterations,
                grad_threshold,
            )
            energies[i] = result[0]
            coords_x_list.append(cx)
            coords_y_list.append(cy)
            coords_z_list.append(cz)
        except Exception as e:
            print(f"Skylake failed for molecule {i}: {e}")
            energies[i] = float("nan")
            coords_x_list.append(m.coords_x.copy())
            coords_y_list.append(m.coords_y.copy())
            coords_z_list.append(m.coords_z.copy())

    end = time.perf_counter()
    time_ms = (end - start) * 1000
    mol_per_sec = n_molecules / (time_ms / 1000)

    return BenchmarkResult(
        backend="Skylake AVX-512 F32",
        time_ms=time_ms,
        mol_per_sec=mol_per_sec,
        energies=energies,
        coords_x=coords_x_list,
        coords_y=coords_y_list,
        coords_z=coords_z_list,
    )


def optimize_cuda_f32(
    mol_data_list: List[MoleculeData], max_iterations: int, grad_threshold: float
) -> Optional[BenchmarkResult]:
    """Optimize molecules using SmallField CUDA F32 (batched)."""
    if not smallfield.has_cuda():
        return None

    n_molecules = len(mol_data_list)

    # Build batched arrays for all molecules
    total_atoms = sum(m.n_atoms for m in mol_data_list)
    total_pairs = sum(len(m.pair_i) for m in mol_data_list)

    # Concatenated coordinate arrays
    all_coords_x = np.zeros(total_atoms, dtype=np.float64)
    all_coords_y = np.zeros(total_atoms, dtype=np.float64)
    all_coords_z = np.zeros(total_atoms, dtype=np.float64)

    # Concatenated pair arrays
    all_pair_i = np.zeros(total_pairs, dtype=np.int32)
    all_pair_j = np.zeros(total_pairs, dtype=np.int32)
    all_epsilon = np.zeros(total_pairs, dtype=np.float64)
    all_r_star = np.zeros(total_pairs, dtype=np.float64)

    # Per-molecule pair ranges
    mol_pair_start = np.zeros(n_molecules, dtype=np.int32)
    mol_pair_end = np.zeros(n_molecules, dtype=np.int32)

    # Track atom offsets for index adjustment
    mol_atom_start = np.zeros(n_molecules, dtype=np.int32)

    atom_offset = 0
    pair_offset = 0

    for i, m in enumerate(mol_data_list):
        n_atoms = m.n_atoms
        n_pairs = len(m.pair_i)

        mol_atom_start[i] = atom_offset
        mol_pair_start[i] = pair_offset
        mol_pair_end[i] = pair_offset + m.n_pairs  # Use actual pairs, not padded

        # Copy coordinates
        all_coords_x[atom_offset : atom_offset + n_atoms] = m.coords_x[:n_atoms]
        all_coords_y[atom_offset : atom_offset + n_atoms] = m.coords_y[:n_atoms]
        all_coords_z[atom_offset : atom_offset + n_atoms] = m.coords_z[:n_atoms]

        # Copy pairs with adjusted indices (add atom offset)
        all_pair_i[pair_offset : pair_offset + n_pairs] = (
            m.pair_i[:n_pairs] + atom_offset
        )
        all_pair_j[pair_offset : pair_offset + n_pairs] = (
            m.pair_j[:n_pairs] + atom_offset
        )
        all_epsilon[pair_offset : pair_offset + n_pairs] = m.epsilon[:n_pairs]
        all_r_star[pair_offset : pair_offset + n_pairs] = m.r_star[:n_pairs]

        atom_offset += n_atoms
        pair_offset += n_pairs

    ctx = smallfield.Context(total_atoms, total_pairs, n_molecules)

    # Warmup with small batch
    try:
        warmup_coords_x = all_coords_x.copy()
        warmup_coords_y = all_coords_y.copy()
        warmup_coords_z = all_coords_z.copy()
        ctx.optimize_cuda_f32(
            warmup_coords_x,
            warmup_coords_y,
            warmup_coords_z,
            all_pair_i,
            all_pair_j,
            all_epsilon,
            all_r_star,
            mol_pair_start[: min(5, n_molecules)],
            mol_pair_end[: min(5, n_molecules)],
            min(5, n_molecules),
            max_iterations,
            grad_threshold,
        )
    except Exception:
        pass

    # Timed batch run
    batch_coords_x = all_coords_x.copy()
    batch_coords_y = all_coords_y.copy()
    batch_coords_z = all_coords_z.copy()

    start = time.perf_counter()

    try:
        result = ctx.optimize_cuda_f32(
            batch_coords_x,
            batch_coords_y,
            batch_coords_z,
            all_pair_i,
            all_pair_j,
            all_epsilon,
            all_r_star,
            mol_pair_start,
            mol_pair_end,
            n_molecules,
            max_iterations,
            grad_threshold,
        )
    except Exception as e:
        print(f"CUDA F32 batch optimization failed: {e}")
        return None

    end = time.perf_counter()
    time_ms = (end - start) * 1000
    mol_per_sec = n_molecules / (time_ms / 1000)

    # Extract per-molecule results
    energies = np.zeros(n_molecules, dtype=np.float64)
    coords_x_list = []
    coords_y_list = []
    coords_z_list = []

    # result[0] is the energy of the first molecule (or batch energy)
    # For now, extract individual energies by computing VdW for each molecule
    for i, m in enumerate(mol_data_list):
        n_atoms = m.n_atoms
        a_start = mol_atom_start[i]

        coords_x_list.append(batch_coords_x[a_start : a_start + n_atoms].copy())
        coords_y_list.append(batch_coords_y[a_start : a_start + n_atoms].copy())
        coords_z_list.append(batch_coords_z[a_start : a_start + n_atoms].copy())

        # Compute final VdW energy for this molecule
        p_start = mol_pair_start[i]
        p_end = mol_pair_end[i]
        energy = 0.0
        for p in range(p_start, p_end):
            ii = all_pair_i[p] - a_start
            jj = all_pair_j[p] - a_start
            dx = batch_coords_x[a_start + ii] - batch_coords_x[a_start + jj]
            dy = batch_coords_y[a_start + ii] - batch_coords_y[a_start + jj]
            dz = batch_coords_z[a_start + ii] - batch_coords_z[a_start + jj]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            rho = r / all_r_star[p]
            rho7 = rho**7
            energy += (
                all_epsilon[p]
                * (1.07 / (rho + 0.07)) ** 7
                * ((1.12 / (rho7 + 0.12)) - 2)
            )
        energies[i] = energy

    return BenchmarkResult(
        backend="CUDA F32",
        time_ms=time_ms,
        mol_per_sec=mol_per_sec,
        energies=energies,
        coords_x=coords_x_list,
        coords_y=coords_y_list,
        coords_z=coords_z_list,
    )


def optimize_cuda_f16(
    mol_data_list: List[MoleculeData], max_iterations: int, grad_threshold: float
) -> Optional[BenchmarkResult]:
    """Optimize molecules using SmallField CUDA F16 (batched)."""
    if not smallfield.has_cuda():
        return None

    n_molecules = len(mol_data_list)

    # Build batched arrays for all molecules (same as F32)
    total_atoms = sum(m.n_atoms for m in mol_data_list)
    total_pairs = sum(len(m.pair_i) for m in mol_data_list)

    all_coords_x = np.zeros(total_atoms, dtype=np.float64)
    all_coords_y = np.zeros(total_atoms, dtype=np.float64)
    all_coords_z = np.zeros(total_atoms, dtype=np.float64)
    all_pair_i = np.zeros(total_pairs, dtype=np.int32)
    all_pair_j = np.zeros(total_pairs, dtype=np.int32)
    all_epsilon = np.zeros(total_pairs, dtype=np.float64)
    all_r_star = np.zeros(total_pairs, dtype=np.float64)
    mol_pair_start = np.zeros(n_molecules, dtype=np.int32)
    mol_pair_end = np.zeros(n_molecules, dtype=np.int32)
    mol_atom_start = np.zeros(n_molecules, dtype=np.int32)

    atom_offset = 0
    pair_offset = 0

    for i, m in enumerate(mol_data_list):
        n_atoms = m.n_atoms
        n_pairs = len(m.pair_i)

        mol_atom_start[i] = atom_offset
        mol_pair_start[i] = pair_offset
        mol_pair_end[i] = pair_offset + m.n_pairs

        all_coords_x[atom_offset : atom_offset + n_atoms] = m.coords_x[:n_atoms]
        all_coords_y[atom_offset : atom_offset + n_atoms] = m.coords_y[:n_atoms]
        all_coords_z[atom_offset : atom_offset + n_atoms] = m.coords_z[:n_atoms]

        all_pair_i[pair_offset : pair_offset + n_pairs] = (
            m.pair_i[:n_pairs] + atom_offset
        )
        all_pair_j[pair_offset : pair_offset + n_pairs] = (
            m.pair_j[:n_pairs] + atom_offset
        )
        all_epsilon[pair_offset : pair_offset + n_pairs] = m.epsilon[:n_pairs]
        all_r_star[pair_offset : pair_offset + n_pairs] = m.r_star[:n_pairs]

        atom_offset += n_atoms
        pair_offset += n_pairs

    ctx = smallfield.Context(total_atoms, total_pairs, n_molecules)

    # Warmup
    try:
        warmup_coords_x = all_coords_x.copy()
        warmup_coords_y = all_coords_y.copy()
        warmup_coords_z = all_coords_z.copy()
        ctx.optimize_cuda_f16(
            warmup_coords_x,
            warmup_coords_y,
            warmup_coords_z,
            all_pair_i,
            all_pair_j,
            all_epsilon,
            all_r_star,
            mol_pair_start[: min(5, n_molecules)],
            mol_pair_end[: min(5, n_molecules)],
            min(5, n_molecules),
            max_iterations,
            grad_threshold,
        )
    except Exception:
        pass

    batch_coords_x = all_coords_x.copy()
    batch_coords_y = all_coords_y.copy()
    batch_coords_z = all_coords_z.copy()

    start = time.perf_counter()

    try:
        result = ctx.optimize_cuda_f16(
            batch_coords_x,
            batch_coords_y,
            batch_coords_z,
            all_pair_i,
            all_pair_j,
            all_epsilon,
            all_r_star,
            mol_pair_start,
            mol_pair_end,
            n_molecules,
            max_iterations,
            grad_threshold,
        )
    except Exception as e:
        print(f"CUDA F16 batch optimization failed: {e}")
        return None

    end = time.perf_counter()
    time_ms = (end - start) * 1000
    mol_per_sec = n_molecules / (time_ms / 1000)

    # Extract per-molecule results
    energies = np.zeros(n_molecules, dtype=np.float64)
    coords_x_list = []
    coords_y_list = []
    coords_z_list = []

    for i, m in enumerate(mol_data_list):
        n_atoms = m.n_atoms
        a_start = mol_atom_start[i]

        coords_x_list.append(batch_coords_x[a_start : a_start + n_atoms].copy())
        coords_y_list.append(batch_coords_y[a_start : a_start + n_atoms].copy())
        coords_z_list.append(batch_coords_z[a_start : a_start + n_atoms].copy())

        # Compute final VdW energy
        p_start = mol_pair_start[i]
        p_end = mol_pair_end[i]
        energy = 0.0
        for p in range(p_start, p_end):
            ii = all_pair_i[p] - a_start
            jj = all_pair_j[p] - a_start
            dx = batch_coords_x[a_start + ii] - batch_coords_x[a_start + jj]
            dy = batch_coords_y[a_start + ii] - batch_coords_y[a_start + jj]
            dz = batch_coords_z[a_start + ii] - batch_coords_z[a_start + jj]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            rho = r / all_r_star[p]
            rho7 = rho**7
            energy += (
                all_epsilon[p]
                * (1.07 / (rho + 0.07)) ** 7
                * ((1.12 / (rho7 + 0.12)) - 2)
            )
        energies[i] = energy

    return BenchmarkResult(
        backend="CUDA F16",
        time_ms=time_ms,
        mol_per_sec=mol_per_sec,
        energies=energies,
        coords_x=coords_x_list,
        coords_y=coords_y_list,
        coords_z=coords_z_list,
    )


# =============================================================================
# Accuracy Metrics
# =============================================================================


def compute_accuracy_metrics(
    results: Dict[str, BenchmarkResult], baseline_key: str
) -> Dict[str, Dict[str, float]]:
    """Compute accuracy metrics relative to a baseline."""
    baseline = results.get(baseline_key)
    if baseline is None:
        return {}

    metrics = {}

    for name, result in results.items():
        if name == baseline_key:
            continue

        # Energy error
        energy_diff = np.abs(result.energies - baseline.energies)
        valid_mask = ~np.isnan(energy_diff)

        if not np.any(valid_mask):
            continue

        energy_mae = float(np.mean(energy_diff[valid_mask]))
        energy_max = float(np.max(energy_diff[valid_mask]))

        # Coordinate RMSD (average over molecules)
        rmsd_list = []
        for i in range(len(result.coords_x)):
            if i < len(baseline.coords_x):
                rmsd = compute_rmsd(
                    result.coords_x[i],
                    result.coords_y[i],
                    result.coords_z[i],
                    baseline.coords_x[i],
                    baseline.coords_y[i],
                    baseline.coords_z[i],
                )
                rmsd_list.append(rmsd)

        avg_rmsd = float(np.mean(rmsd_list)) if rmsd_list else float("nan")

        metrics[name] = {
            "energy_mae": energy_mae,
            "energy_max": energy_max,
            "coord_rmsd": avg_rmsd,
        }

    return metrics


# =============================================================================
# Main Benchmark
# =============================================================================


def run_benchmark(
    n_molecules: int, max_iterations: int, grad_threshold: float, smiles_dir: Path
):
    """Run the geometry optimization benchmark."""
    print("=" * 100)
    print("SmallField MMFF94 Geometry Optimization Benchmark")
    print("=" * 100)
    print()

    # Thread configuration
    omp_threads = get_omp_threads()
    print("Thread Configuration:")
    print(f"  CPU cores available:    {CPU_CORES}")
    print(f"  OMP_NUM_THREADS:        {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"  RDKit threads (OMP):    {omp_threads}")
    print(f"  SmallField Serial:      1 (single-threaded)")
    print()

    # Print capabilities
    print("Capabilities:")
    print(f"  Skylake (AVX-512 F32):  {'Yes' if smallfield.has_skylake() else 'No'}")
    print(f"  CUDA:                   {'Yes' if smallfield.has_cuda() else 'No'}")
    print()

    # Load molecules
    mol_data_list = load_molecules(n_molecules, smiles_dir)
    if not mol_data_list:
        print("No molecules loaded. Exiting.")
        return

    # Create MMFF94 data for serial optimizers (full force field)
    print("Extracting full MMFF94 parameters for serial optimizers...")
    mmff94_mol_data_list: List[MMFF94MolData] = []
    for m in mol_data_list:
        try:
            mmff94_data = MMFF94MolData.from_mol(m.mol)
            mmff94_mol_data_list.append(mmff94_data)
        except Exception as e:
            print(f"  Warning: Failed to extract MMFF94 params: {e}")
    print(f"  Extracted MMFF94 params for {len(mmff94_mol_data_list)} molecules")

    print(f"\nOptimization parameters:")
    print(f"  Max iterations:      {max_iterations}")
    print(f"  Gradient threshold:  {grad_threshold}")
    print()

    total_atoms = sum(m.n_atoms for m in mol_data_list)
    total_pairs = sum(m.n_pairs for m in mol_data_list)

    print(f"Batch statistics:")
    print(f"  Molecules:   {len(mol_data_list):,}")
    print(f"  Total atoms: {total_atoms:,}")
    print(f"  Total pairs: {total_pairs:,}")
    print()

    # Run all backends
    results: Dict[str, BenchmarkResult] = {}

    print("Running optimizations...")
    print()

    # RDKit baseline
    print("  RDKit (baseline)...", end=" ", flush=True)
    rdkit_result = optimize_rdkit(mol_data_list, max_iterations)
    results["RDKit (baseline)"] = rdkit_result
    print(f"done ({rdkit_result.time_ms:.1f} ms)")

    # Serial F64 (uses full MMFF94)
    print("  Serial F64 (MMFF94)...", end=" ", flush=True)
    try:
        serial_result = optimize_serial_f64(
            mmff94_mol_data_list, max_iterations, grad_threshold
        )
        if serial_result:
            results["Serial F64"] = serial_result
            print(f"done ({serial_result.time_ms:.1f} ms)")
        else:
            print("N/A")
    except Exception as e:
        print(f"failed: {e}")

    # Serial F32 (uses full MMFF94)
    print("  Serial F32 (MMFF94)...", end=" ", flush=True)
    try:
        serial_f32_result = optimize_serial_f32(
            mmff94_mol_data_list, max_iterations, grad_threshold
        )
        if serial_f32_result:
            results["Serial F32"] = serial_f32_result
            print(f"done ({serial_f32_result.time_ms:.1f} ms)")
        else:
            print("N/A")
    except Exception as e:
        print(f"failed: {e}")

    # Skylake (VdW-only for now, pending kernel update)
    print("  Skylake AVX-512 F32 (VdW)...", end=" ", flush=True)
    try:
        skylake_result = optimize_skylake(mol_data_list, max_iterations, grad_threshold)
        if skylake_result:
            results["Skylake AVX-512 F32"] = skylake_result
            print(f"done ({skylake_result.time_ms:.1f} ms)")
        else:
            print("N/A")
    except Exception as e:
        print(f"failed: {e}")

    # CUDA F32 - disabled for kernel rewrite
    print("  CUDA F32... disabled (kernel rewrite pending)")

    # CUDA F16 - disabled for kernel rewrite
    print("  CUDA F16... disabled (kernel rewrite pending)")

    print()

    # Throughput table
    print("-" * 100)
    print(f"THROUGHPUT  (CPU cores: {CPU_CORES})")
    print("-" * 100)
    print(f"{'Backend':<35} {'Time (ms)':>15} {'Mol/sec':>15} {'Speedup':>15}")
    print("-" * 100)

    rdkit_result = results["RDKit (baseline)"]
    rdkit_time = rdkit_result.time_ms
    rdkit_mol_per_sec = rdkit_result.mol_per_sec

    for name, result in results.items():
        speedup = rdkit_time / result.time_ms if result.time_ms > 0 else 0
        print(
            f"{name:<35} {result.time_ms:>15.2f} {result.mol_per_sec:>15.0f} {speedup:>14.1f}x"
        )

    # Add theoretical multi-core RDKit throughput
    theoretical_mol_per_sec = rdkit_mol_per_sec * CPU_CORES
    theoretical_time = rdkit_time / CPU_CORES
    print(
        f"{'RDKit (theoretical ' + str(CPU_CORES) + ' cores)':<35} {theoretical_time:>15.2f} {theoretical_mol_per_sec:>15.0f} {'1.0x (upper bound)':>15}"
    )

    print("-" * 100)
    print()

    # Iteration statistics for SmallField
    for name, result in results.items():
        if result.iter_counts:
            iters = np.array(result.iter_counts)
            hit_max = np.sum(iters >= max_iterations)
            print(f"Iteration stats ({name}):")
            print(f"  Mean: {np.mean(iters):.1f}, Median: {np.median(iters):.0f}, Max: {np.max(iters)}")
            print(f"  Hit max ({max_iterations}): {hit_max}/{len(iters)} ({100*hit_max/len(iters):.1f}%)")
            # Check for diverged molecules (energy > 1000 kcal/mol or very negative)
            diverged = np.sum((result.energies > 1000) | (result.energies < -1000))
            if diverged > 0:
                print(f"  Diverged (|E| > 1000): {diverged}/{len(result.energies)}")
            print()

    # Accuracy vs RDKit
    print("-" * 100)
    print("ACCURACY vs RDKit")
    print("-" * 100)
    print(f"{'Backend':<30} {'Energy MAE':>15} {'Energy Max':>15} {'Coord RMSD':>15}")
    print("-" * 100)

    rdkit_metrics = compute_accuracy_metrics(results, "RDKit (baseline)")
    for name, m in rdkit_metrics.items():
        print(
            f"{name:<30} {m['energy_mae']:>15.6f} {m['energy_max']:>15.6f} {m['coord_rmsd']:>15.6f}"
        )

    print("-" * 100)
    print()

    # Accuracy vs Serial F64
    if "Serial F64" in results:
        print("-" * 100)
        print("ACCURACY vs Serial F64")
        print("-" * 100)
        print(
            f"{'Backend':<30} {'Energy MAE':>15} {'Energy Max':>15} {'Coord RMSD':>15}"
        )
        print("-" * 100)

        serial_metrics = compute_accuracy_metrics(results, "Serial F64")
        for name, m in serial_metrics.items():
            print(
                f"{name:<30} {m['energy_mae']:>15.6f} {m['energy_max']:>15.6f} {m['coord_rmsd']:>15.6f}"
            )

        print("-" * 100)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SmallField MMFF94 Geometry Optimization Benchmark"
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=100,
        help="Number of molecules (default: 100)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Max optimization iterations (default: 200)",
    )
    parser.add_argument(
        "--grad-threshold",
        type=float,
        default=0.1,
        help="Gradient convergence threshold (default: 0.1)",
    )
    parser.add_argument(
        "--smiles-dir",
        type=Path,
        default=DEFAULT_SMILES_PATH,
        help=f"Path to SMILES directory (default: {DEFAULT_SMILES_PATH})",
    )
    args = parser.parse_args()

    run_benchmark(args.n_molecules, args.max_iter, args.grad_threshold, args.smiles_dir)
