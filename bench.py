#!/usr/bin/env python3
"""
bench.py - Batch benchmark for SmallField MMFF94 implementations.

Loads molecules from USearchMolecules and benchmarks batch processing.

Usage:
    python bench.py --n-molecules 1000 --iterations 100
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("ERROR: RDKit not available. Install with: pip install rdkit")
    raise SystemExit(1)

try:
    import smallfield
except ImportError:
    print("ERROR: SmallField not available. Build with: pip install -e .")
    raise SystemExit(1)


DEFAULT_SMILES_PATH = (
    Path(__file__).parent.parent / "USearchMolecules" / "data" / "example" / "smiles"
)


class MoleculeData:
    """Container for molecule coordinates and force field parameters."""

    def __init__(self, mol: Chem.Mol):
        self.mol = mol
        self.n_atoms = mol.GetNumAtoms()

        # Extract coordinates
        conf = mol.GetConformer(-1)
        self.coords_x_f64 = np.array(
            [conf.GetAtomPosition(i).x for i in range(self.n_atoms)], dtype=np.float64
        )
        self.coords_y_f64 = np.array(
            [conf.GetAtomPosition(i).y for i in range(self.n_atoms)], dtype=np.float64
        )
        self.coords_z_f64 = np.array(
            [conf.GetAtomPosition(i).z for i in range(self.n_atoms)], dtype=np.float64
        )

        # Extract VdW parameters
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is None:
            raise ValueError("Could not get MMFF properties")

        pair_i_list, pair_j_list, epsilon_list, r0_list = [], [], [], []

        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                if mol.GetBondBetweenAtoms(i, j) is not None:
                    continue
                vdw_params = mmff_props.GetMMFFVdWParams(i, j)
                if vdw_params is not None:
                    pair_i_list.append(i)
                    pair_j_list.append(j)
                    epsilon_list.append(vdw_params[1])
                    r0_list.append(vdw_params[0])

        self.n_pairs = len(pair_i_list)

        # Pad to multiple of 32 for SIMD
        padding = (32 - self.n_pairs % 32) % 32
        total_pairs = self.n_pairs + padding

        self.pair_i = np.zeros(total_pairs, dtype=np.int32)
        self.pair_j = np.zeros(total_pairs, dtype=np.int32)
        self.epsilon_f64 = np.zeros(total_pairs, dtype=np.float64)
        self.r0_f64 = np.zeros(total_pairs, dtype=np.float64)

        for k in range(self.n_pairs):
            self.pair_i[k] = pair_i_list[k]
            self.pair_j[k] = pair_j_list[k]
            self.epsilon_f64[k] = epsilon_list[k]
            self.r0_f64[k] = r0_list[k]

        # Pad with safe values
        if padding > 0 and self.n_pairs > 0:
            last_idx = self.n_pairs - 1
            for k in range(self.n_pairs, total_pairs):
                self.pair_i[k] = self.pair_i[last_idx]
                self.pair_j[k] = self.pair_j[last_idx]
                self.epsilon_f64[k] = 0.0
                self.r0_f64[k] = 1.0


class BatchData:
    """Container for batched molecule data."""

    def __init__(self, mol_data_list: List[MoleculeData]):
        self.n_molecules = len(mol_data_list)

        # Concatenate coordinates
        self.coords_x_f64 = np.concatenate([m.coords_x_f64 for m in mol_data_list])
        self.coords_y_f64 = np.concatenate([m.coords_y_f64 for m in mol_data_list])
        self.coords_z_f64 = np.concatenate([m.coords_z_f64 for m in mol_data_list])

        # Adjust pair indices and concatenate
        all_pair_i, all_pair_j = [], []
        all_epsilon, all_r0 = [], []
        atom_offset = 0

        for m in mol_data_list:
            all_pair_i.append(m.pair_i + atom_offset)
            all_pair_j.append(m.pair_j + atom_offset)
            all_epsilon.append(m.epsilon_f64)
            all_r0.append(m.r0_f64)
            atom_offset += m.n_atoms

        self.pair_i = np.concatenate(all_pair_i)
        self.pair_j = np.concatenate(all_pair_j)
        self.epsilon_f64 = np.concatenate(all_epsilon)
        self.r0_f64 = np.concatenate(all_r0)

        # Create molecule indexing arrays
        mol_nb_start, mol_nb_end = [], []
        pair_offset = 0

        for m in mol_data_list:
            mol_nb_start.append(pair_offset)
            pair_offset += len(m.pair_i)
            mol_nb_end.append(pair_offset)

        self.mol_nb_start = np.array(mol_nb_start, dtype=np.int32)
        self.mol_nb_end = np.array(mol_nb_end, dtype=np.int32)

        self.total_atoms = sum(m.n_atoms for m in mol_data_list)
        self.total_pairs = len(self.pair_i)


def load_smiles_from_file(file_path: Path, n_molecules: int) -> List[str]:
    """Load SMILES strings from a file."""
    smiles_list = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                # SMILES files often have SMILES as first column
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
        print("Falling back to default test molecules...")
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
            mol_data = MoleculeData(mol)
            if mol_data.n_pairs > 0:
                mol_data_list.append(mol_data)
        except Exception:
            failed += 1
            continue

    print(f"Loaded {len(mol_data_list)} molecules ({failed} failed)")
    return mol_data_list


def load_default_molecules(n_molecules: int) -> List[MoleculeData]:
    """Load default test molecules when USearchMolecules is not available."""
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
        mol_data_list.append(MoleculeData(mol))
        idx += 1

    return mol_data_list


def benchmark_batch_cuda_f32(
    batch: BatchData, n_iterations: int
) -> Optional[Tuple[float, float]]:
    """Benchmark CUDA F32 batch processing."""
    if not smallfield.has_cuda():
        return None

    # Warmup
    for _ in range(10):
        smallfield.compute_vdw_energy_cuda_f32(
            batch.coords_x_f64,
            batch.coords_y_f64,
            batch.coords_z_f64,
            batch.pair_i,
            batch.pair_j,
            batch.epsilon_f64,
            batch.r0_f64,
            batch.mol_nb_start,
            batch.mol_nb_end,
            batch.n_molecules,
        )

    start = time.perf_counter()
    for _ in range(n_iterations):
        energy = smallfield.compute_vdw_energy_cuda_f32(
            batch.coords_x_f64,
            batch.coords_y_f64,
            batch.coords_z_f64,
            batch.pair_i,
            batch.pair_j,
            batch.epsilon_f64,
            batch.r0_f64,
            batch.mol_nb_start,
            batch.mol_nb_end,
            batch.n_molecules,
        )
    end = time.perf_counter()

    return energy, (end - start) / n_iterations * 1000


def benchmark_batch_cuda_f16(
    batch: BatchData, n_iterations: int
) -> Optional[Tuple[float, float]]:
    """Benchmark CUDA F16 batch processing."""
    if not smallfield.has_cuda():
        return None

    # Warmup
    for _ in range(10):
        smallfield.compute_vdw_energy_cuda_f16(
            batch.coords_x_f64,
            batch.coords_y_f64,
            batch.coords_z_f64,
            batch.pair_i,
            batch.pair_j,
            batch.epsilon_f64,
            batch.r0_f64,
            batch.mol_nb_start,
            batch.mol_nb_end,
            batch.n_molecules,
        )

    start = time.perf_counter()
    for _ in range(n_iterations):
        energy = smallfield.compute_vdw_energy_cuda_f16(
            batch.coords_x_f64,
            batch.coords_y_f64,
            batch.coords_z_f64,
            batch.pair_i,
            batch.pair_j,
            batch.epsilon_f64,
            batch.r0_f64,
            batch.mol_nb_start,
            batch.mol_nb_end,
            batch.n_molecules,
        )
    end = time.perf_counter()

    return energy, (end - start) / n_iterations * 1000


def benchmark_batch_skylake(
    batch: BatchData, n_iterations: int
) -> Optional[Tuple[float, float]]:
    """Benchmark Skylake AVX-512 batch processing (sequential per-molecule)."""
    if not smallfield.has_skylake():
        return None

    ctx = smallfield.Context(batch.total_atoms, batch.total_pairs, batch.n_molecules)

    # Warmup
    for _ in range(10):
        ctx.compute_vdw_energy_skylake_batch(
            batch.coords_x_f64,
            batch.coords_y_f64,
            batch.coords_z_f64,
            batch.pair_i,
            batch.pair_j,
            batch.epsilon_f64,
            batch.r0_f64,
            batch.mol_nb_start,
            batch.mol_nb_end,
            batch.n_molecules,
        )

    start = time.perf_counter()
    for _ in range(n_iterations):
        energy = ctx.compute_vdw_energy_skylake_batch(
            batch.coords_x_f64,
            batch.coords_y_f64,
            batch.coords_z_f64,
            batch.pair_i,
            batch.pair_j,
            batch.epsilon_f64,
            batch.r0_f64,
            batch.mol_nb_start,
            batch.mol_nb_end,
            batch.n_molecules,
        )
    end = time.perf_counter()

    return energy, (end - start) / n_iterations * 1000


def run_benchmark(n_molecules: int, n_iterations: int):
    """Run the batch benchmark."""
    print("=" * 100)
    print("SmallField MMFF94 Batch Benchmark")
    print("=" * 100)
    print()

    # Print capabilities
    print("Capabilities:")
    print(f"  Skylake (AVX-512 F32):  {'Yes' if smallfield.has_skylake() else 'No'}")
    print(f"  CUDA:                   {'Yes' if smallfield.has_cuda() else 'No'}")
    print()

    # Load molecules
    mol_data_list = load_molecules_from_usearch(n_molecules)
    if not mol_data_list:
        print("No molecules loaded. Exiting.")
        return

    batch = BatchData(mol_data_list)
    print(f"\nBatch statistics:")
    print(f"  Molecules:   {batch.n_molecules:,}")
    print(f"  Total atoms: {batch.total_atoms:,}")
    print(f"  Total pairs: {batch.total_pairs:,}")
    print()

    print("-" * 100)
    print(
        f"{'Backend':<30} {'Energy (kcal/mol)':>20} {'Time (ms)':>15} {'Mol/sec':>15}"
    )
    print("-" * 100)

    # Skylake batch
    skylake_result = benchmark_batch_skylake(batch, n_iterations)
    if skylake_result:
        energy, time_ms = skylake_result
        mol_per_sec = batch.n_molecules / (time_ms / 1000)
        print(
            f"{'Skylake (AVX-512 F32)':<30} {energy:>20.4f} {time_ms:>15.4f} {mol_per_sec:>15.0f}"
        )
    else:
        print(f"{'Skylake (AVX-512 F32)':<30} {'N/A':>20} {'N/A':>15} {'N/A':>15}")

    # CUDA F32 batch
    cuda_f32_result = benchmark_batch_cuda_f32(batch, n_iterations)
    if cuda_f32_result:
        energy, time_ms = cuda_f32_result
        mol_per_sec = batch.n_molecules / (time_ms / 1000)
        print(
            f"{'CUDA (F32)':<30} {energy:>20.4f} {time_ms:>15.4f} {mol_per_sec:>15.0f}"
        )
    else:
        print(f"{'CUDA (F32)':<30} {'N/A':>20} {'N/A':>15} {'N/A':>15}")

    # CUDA F16 batch
    cuda_f16_result = benchmark_batch_cuda_f16(batch, n_iterations)
    if cuda_f16_result:
        energy, time_ms = cuda_f16_result
        mol_per_sec = batch.n_molecules / (time_ms / 1000)
        print(
            f"{'CUDA (F16)':<30} {energy:>20.4f} {time_ms:>15.4f} {mol_per_sec:>15.0f}"
        )
    else:
        print(f"{'CUDA (F16)':<30} {'N/A':>20} {'N/A':>15} {'N/A':>15}")

    print("-" * 100)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmallField MMFF94 Batch Benchmark")
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=1000,
        help="Number of molecules (default: 1000)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Benchmark iterations (default: 100)",
    )
    args = parser.parse_args()

    run_benchmark(args.n_molecules, args.iterations)
