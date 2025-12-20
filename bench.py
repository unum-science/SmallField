#!/usr/bin/env python3
"""
bench.py - Comprehensive benchmark comparing ALL SmallField MMFF94 implementations.

This benchmark compares:
- RDKit (f32) - baseline
- SmallField Serial (f32, f64)
- SmallField Skylake (AVX-512 F32)
- SmallField Genoa (AVX-512 BF16)
- SmallField Turin (AVX-512 FP16)
- SmallField CUDA (f16, f32) - H100 and newer

Reports:
- Accuracy: Error vs RDKit baseline
- Throughput: Speedup vs RDKit baseline
- Absolute timing for each implementation

Usage:
    uv run python bench.py
    uv run pytest --benchmark-enable bench.py
"""

import time
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("ERROR: RDKit not available. Install with: pip install rdkit")
    sys.exit(1)

try:
    import smallfield

    SMALLFIELD_AVAILABLE = True
except ImportError:
    SMALLFIELD_AVAILABLE = False
    print("ERROR: SmallField not available. Build with: pip install -e .")
    sys.exit(1)

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


class MoleculeData:
    """Container for molecule coordinates and force field parameters."""

    def __init__(self, mol: Chem.Mol):
        self.mol = mol
        self.n_atoms = mol.GetNumAtoms()

        # Extract coordinates
        conf = mol.GetConformer(-1)
        self.coords_x_f64 = np.zeros(self.n_atoms, dtype=np.float64)
        self.coords_y_f64 = np.zeros(self.n_atoms, dtype=np.float64)
        self.coords_z_f64 = np.zeros(self.n_atoms, dtype=np.float64)

        for i in range(self.n_atoms):
            pos = conf.GetAtomPosition(i)
            self.coords_x_f64[i] = pos.x
            self.coords_y_f64[i] = pos.y
            self.coords_z_f64[i] = pos.z

        self.coords_x_f32 = self.coords_x_f64.astype(np.float32)
        self.coords_y_f32 = self.coords_y_f64.astype(np.float32)
        self.coords_z_f32 = self.coords_z_f64.astype(np.float32)

        # Extract VdW parameters
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is None:
            raise ValueError("Could not get MMFF properties")

        pair_i_list = []
        pair_j_list = []
        epsilon_list = []
        r0_list = []
        charge_i_list = []
        charge_j_list = []

        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                # Skip bonded pairs
                if mol.GetBondBetweenAtoms(i, j) is not None:
                    continue

                # Get VdW parameters
                vdw_params = mmff_props.GetMMFFVdWParams(i, j)
                if vdw_params is not None:
                    pair_i_list.append(i)
                    pair_j_list.append(j)
                    epsilon_list.append(vdw_params[1])
                    r0_list.append(vdw_params[0])
                    charge_i_list.append(mmff_props.GetMMFFPartialCharge(i))
                    charge_j_list.append(mmff_props.GetMMFFPartialCharge(j))

        self.n_pairs = len(pair_i_list)

        # Pad to multiple of 16 for SIMD
        padding = (16 - self.n_pairs % 16) % 16
        total_pairs = self.n_pairs + padding

        self.pair_i = np.zeros(total_pairs, dtype=np.int32)
        self.pair_j = np.zeros(total_pairs, dtype=np.int32)
        self.epsilon_f64 = np.zeros(total_pairs, dtype=np.float64)
        self.r0_f64 = np.zeros(total_pairs, dtype=np.float64)
        self.charge_i_f64 = np.zeros(total_pairs, dtype=np.float64)
        self.charge_j_f64 = np.zeros(total_pairs, dtype=np.float64)
        self.is_1_4 = np.zeros(total_pairs, dtype=np.int32)

        # Fill with actual data
        for k in range(self.n_pairs):
            self.pair_i[k] = pair_i_list[k]
            self.pair_j[k] = pair_j_list[k]
            self.epsilon_f64[k] = epsilon_list[k]
            self.r0_f64[k] = r0_list[k]
            self.charge_i_f64[k] = charge_i_list[k]
            self.charge_j_f64[k] = charge_j_list[k]

        # Pad with safe values
        if padding > 0:
            last_idx = self.n_pairs - 1
            for k in range(self.n_pairs, total_pairs):
                self.pair_i[k] = self.pair_i[last_idx]
                self.pair_j[k] = self.pair_j[last_idx]
                self.epsilon_f64[k] = 0.0
                self.r0_f64[k] = 1.0

        # F32 versions
        self.epsilon_f32 = self.epsilon_f64.astype(np.float32)
        self.r0_f32 = self.r0_f64.astype(np.float32)
        self.charge_i_f32 = self.charge_i_f64.astype(np.float32)
        self.charge_j_f32 = self.charge_j_f64.astype(np.float32)

        # F16 versions (stored as uint16 for CUDA half / Turin FP16 compatibility)
        # NumPy float16 has same IEEE 754 binary format as CUDA half
        self.coords_x_f16 = self.coords_x_f32.astype(np.float16).view(np.uint16)
        self.coords_y_f16 = self.coords_y_f32.astype(np.float16).view(np.uint16)
        self.coords_z_f16 = self.coords_z_f32.astype(np.float16).view(np.uint16)
        self.epsilon_f16 = self.epsilon_f32.astype(np.float16).view(np.uint16)
        self.r0_f16 = self.r0_f32.astype(np.float16).view(np.uint16)
        self.charge_i_f16 = self.charge_i_f32.astype(np.float16).view(np.uint16)
        self.charge_j_f16 = self.charge_j_f32.astype(np.float16).view(np.uint16)

        # BF16 versions (for Genoa AVX-512 BF16)
        # BF16 = truncate f32 mantissa: keep sign(1) + exp(8) + mantissa(7)
        # This is equivalent to: (f32_bits >> 16) with rounding
        def f32_to_bf16(arr_f32):
            """Convert float32 array to bfloat16 stored as uint16."""
            arr_u32 = arr_f32.view(np.uint32)
            # Round to nearest even (add 0x7FFF + LSB of result for tie-breaking)
            rounding = 0x7FFF + ((arr_u32 >> 16) & 1)
            bf16 = ((arr_u32 + rounding) >> 16).astype(np.uint16)
            return bf16

        self.coords_x_bf16 = f32_to_bf16(self.coords_x_f32)
        self.coords_y_bf16 = f32_to_bf16(self.coords_y_f32)
        self.coords_z_bf16 = f32_to_bf16(self.coords_z_f32)
        self.epsilon_bf16 = f32_to_bf16(self.epsilon_f32)
        self.r0_bf16 = f32_to_bf16(self.r0_f32)


def create_molecule(smiles: str, optimize: bool = True) -> Chem.Mol:
    """Create and prepare a molecule for benchmarking."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

    if optimize:
        AllChem.MMFFOptimizeMolecule(mol)

    return mol


def benchmark_rdkit(
    mol_data: MoleculeData, n_iterations: int = 100
) -> Tuple[float, float]:
    """
    Benchmark RDKit MMFF94 energy calculation.

    Returns:
        (energy, time_per_iteration_ms)
    """
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol_data.mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol_data.mol, mmff_props)

    # Warmup
    for _ in range(10):
        ff.CalcEnergy()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        energy = ff.CalcEnergy()
    end = time.perf_counter()

    time_ms = (end - start) / n_iterations * 1000
    return energy, time_ms


def benchmark_smallfield_serial_f64(
    mol_data: MoleculeData, n_iterations: int = 100
) -> Tuple[float, float]:
    """Benchmark SmallField serial f64 implementation."""
    # Warmup
    for _ in range(10):
        smallfield.compute_vdw_energy_serial_f64(
            mol_data.coords_x_f64,
            mol_data.coords_y_f64,
            mol_data.coords_z_f64,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f64,
            mol_data.r0_f64,
        )

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        energy = smallfield.compute_vdw_energy_serial_f64(
            mol_data.coords_x_f64,
            mol_data.coords_y_f64,
            mol_data.coords_z_f64,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f64,
            mol_data.r0_f64,
        )
    end = time.perf_counter()

    time_ms = (end - start) / n_iterations * 1000
    return energy, time_ms


def benchmark_smallfield_serial_f32(
    mol_data: MoleculeData, n_iterations: int = 100
) -> Tuple[float, float]:
    """Benchmark SmallField serial f32 implementation."""
    # Warmup
    for _ in range(10):
        smallfield.compute_vdw_energy_serial_f32(
            mol_data.coords_x_f32,
            mol_data.coords_y_f32,
            mol_data.coords_z_f32,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f32,
            mol_data.r0_f32,
        )

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        energy = smallfield.compute_vdw_energy_serial_f32(
            mol_data.coords_x_f32,
            mol_data.coords_y_f32,
            mol_data.coords_z_f32,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f32,
            mol_data.r0_f32,
        )
    end = time.perf_counter()

    time_ms = (end - start) / n_iterations * 1000
    return energy, time_ms


def benchmark_smallfield_skylake(
    mol_data: MoleculeData, n_iterations: int = 100
) -> Optional[Tuple[float, float]]:
    """Benchmark SmallField AVX-512 F32 (Skylake) implementation."""
    if not smallfield.has_skylake():
        return None

    # Warmup
    for _ in range(10):
        smallfield.compute_vdw_energy_skylake_f32(
            mol_data.coords_x_f32,
            mol_data.coords_y_f32,
            mol_data.coords_z_f32,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f32,
            mol_data.r0_f32,
        )

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        energy = smallfield.compute_vdw_energy_skylake_f32(
            mol_data.coords_x_f32,
            mol_data.coords_y_f32,
            mol_data.coords_z_f32,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f32,
            mol_data.r0_f32,
        )
    end = time.perf_counter()

    time_ms = (end - start) / n_iterations * 1000
    return energy, time_ms


def benchmark_smallfield_genoa(
    mol_data: MoleculeData, n_iterations: int = 100
) -> Optional[Tuple[float, float]]:
    """Benchmark SmallField AVX-512 BF16 (Genoa) implementation."""
    if not smallfield.has_genoa():
        return None

    # Warmup (use BF16 arrays - stored as uint16 with bfloat16 format)
    for _ in range(10):
        smallfield.compute_vdw_energy_genoa_bf16(
            mol_data.coords_x_bf16,
            mol_data.coords_y_bf16,
            mol_data.coords_z_bf16,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_bf16,
            mol_data.r0_bf16,
        )

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        energy = smallfield.compute_vdw_energy_genoa_bf16(
            mol_data.coords_x_bf16,
            mol_data.coords_y_bf16,
            mol_data.coords_z_bf16,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_bf16,
            mol_data.r0_bf16,
        )
    end = time.perf_counter()

    time_ms = (end - start) / n_iterations * 1000
    return energy, time_ms


def benchmark_smallfield_turin(
    mol_data: MoleculeData, n_iterations: int = 100
) -> Optional[Tuple[float, float]]:
    """Benchmark SmallField AVX-512 FP16 (Turin) implementation."""
    if not smallfield.has_turin():
        return None

    # Warmup (use FP16 arrays - stored as uint16 with IEEE 754 half format)
    for _ in range(10):
        smallfield.compute_vdw_energy_turin_fp16(
            mol_data.coords_x_f16,
            mol_data.coords_y_f16,
            mol_data.coords_z_f16,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f16,
            mol_data.r0_f16,
        )

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        energy = smallfield.compute_vdw_energy_turin_fp16(
            mol_data.coords_x_f16,
            mol_data.coords_y_f16,
            mol_data.coords_z_f16,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f16,
            mol_data.r0_f16,
        )
    end = time.perf_counter()

    time_ms = (end - start) / n_iterations * 1000
    return energy, time_ms


def benchmark_smallfield_cuda_f32(
    mol_data: MoleculeData, n_iterations: int = 100
) -> Optional[Tuple[float, float]]:
    """Benchmark SmallField CUDA f32 implementation."""
    if not smallfield.has_cuda():
        return None

    # For single molecule batch
    mol_nb_start = np.array([0], dtype=np.int32)
    mol_nb_end = np.array([len(mol_data.pair_i)], dtype=np.int32)
    n_molecules = 1

    # Warmup
    for _ in range(10):
        smallfield.compute_vdw_energy_cuda_f32(
            mol_data.coords_x_f32,
            mol_data.coords_y_f32,
            mol_data.coords_z_f32,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f32,
            mol_data.r0_f32,
            mol_nb_start,
            mol_nb_end,
            n_molecules,
        )

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        energy = smallfield.compute_vdw_energy_cuda_f32(
            mol_data.coords_x_f32,
            mol_data.coords_y_f32,
            mol_data.coords_z_f32,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f32,
            mol_data.r0_f32,
            mol_nb_start,
            mol_nb_end,
            n_molecules,
        )
    end = time.perf_counter()

    time_ms = (end - start) / n_iterations * 1000
    return energy, time_ms


def benchmark_smallfield_cuda_f16(
    mol_data: MoleculeData, n_iterations: int = 100
) -> Optional[Tuple[float, float]]:
    """Benchmark SmallField CUDA f16 implementation."""
    if not smallfield.has_cuda():
        return None

    # For single molecule batch
    mol_nb_start = np.array([0], dtype=np.int32)
    mol_nb_end = np.array([len(mol_data.pair_i)], dtype=np.int32)
    n_molecules = 1

    # Warmup (use FP16 arrays - stored as uint16 with IEEE 754 half format)
    for _ in range(10):
        smallfield.compute_vdw_energy_cuda_f16(
            mol_data.coords_x_f16,
            mol_data.coords_y_f16,
            mol_data.coords_z_f16,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f16,
            mol_data.r0_f16,
            mol_nb_start,
            mol_nb_end,
            n_molecules,
        )

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        energy = smallfield.compute_vdw_energy_cuda_f16(
            mol_data.coords_x_f16,
            mol_data.coords_y_f16,
            mol_data.coords_z_f16,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f16,
            mol_data.r0_f16,
            mol_nb_start,
            mol_nb_end,
            n_molecules,
        )
    end = time.perf_counter()

    time_ms = (end - start) / n_iterations * 1000
    return energy, time_ms


def run_comprehensive_benchmark(smiles_list: List[str], n_iterations: int = 100):
    """
    Run comprehensive benchmark comparing all implementations.

    Args:
        smiles_list: List of SMILES strings to benchmark
        n_iterations: Number of iterations per benchmark
    """
    print("=" * 120)
    print("SmallField MMFF94 Comprehensive Benchmark")
    print("=" * 120)
    print()

    # Print capabilities
    print("Capabilities:")
    print(f"  Skylake (AVX-512 F32):  {'Yes' if smallfield.has_skylake() else 'No'}")
    print(f"  Genoa (AVX-512 BF16):   {'Yes' if smallfield.has_genoa() else 'No'}")
    print(f"  Turin (AVX-512 FP16):   {'Yes' if smallfield.has_turin() else 'No'}")
    print(f"  CUDA:                   {'Yes' if smallfield.has_cuda() else 'No'}")
    print()

    results = []

    for smiles in smiles_list:
        print(f"Molecule: {smiles}")
        mol = create_molecule(smiles)
        mol_data = MoleculeData(mol)
        print(f"  Atoms: {mol_data.n_atoms}, Non-bonded pairs: {mol_data.n_pairs}")

        # Benchmark all implementations
        rdkit_energy, rdkit_time = benchmark_rdkit(mol_data, n_iterations)
        print(
            f"  RDKit (baseline):      {rdkit_energy:10.4f} kcal/mol  {rdkit_time:8.4f} ms  (1.00x)"
        )

        serial_f64_energy, serial_f64_time = benchmark_smallfield_serial_f64(
            mol_data, n_iterations
        )
        serial_f64_speedup = rdkit_time / serial_f64_time
        serial_f64_error = abs(serial_f64_energy - rdkit_energy)
        print(
            f"  Serial (f64):          {serial_f64_energy:10.4f} kcal/mol  {serial_f64_time:8.4f} ms  ({serial_f64_speedup:5.2f}x)  error: {serial_f64_error:.2e}"
        )

        serial_f32_energy, serial_f32_time = benchmark_smallfield_serial_f32(
            mol_data, n_iterations
        )
        serial_f32_speedup = rdkit_time / serial_f32_time
        serial_f32_error = abs(serial_f32_energy - rdkit_energy)
        print(
            f"  Serial (f32):          {serial_f32_energy:10.4f} kcal/mol  {serial_f32_time:8.4f} ms  ({serial_f32_speedup:5.2f}x)  error: {serial_f32_error:.2e}"
        )

        result = {
            "smiles": smiles,
            "n_atoms": mol_data.n_atoms,
            "n_pairs": mol_data.n_pairs,
            "rdkit_energy": rdkit_energy,
            "rdkit_time": rdkit_time,
            "serial_f64_energy": serial_f64_energy,
            "serial_f64_time": serial_f64_time,
            "serial_f64_speedup": serial_f64_speedup,
            "serial_f64_error": serial_f64_error,
            "serial_f32_energy": serial_f32_energy,
            "serial_f32_time": serial_f32_time,
            "serial_f32_speedup": serial_f32_speedup,
            "serial_f32_error": serial_f32_error,
        }

        # Skylake
        skylake_result = benchmark_smallfield_skylake(mol_data, n_iterations)
        if skylake_result is not None:
            skylake_energy, skylake_time = skylake_result
            skylake_speedup = rdkit_time / skylake_time
            skylake_error = abs(skylake_energy - rdkit_energy)
            print(
                f"  Skylake (AVX-512 F32): {skylake_energy:10.4f} kcal/mol  {skylake_time:8.4f} ms  ({skylake_speedup:5.2f}x)  error: {skylake_error:.2e}"
            )
            result["skylake_energy"] = skylake_energy
            result["skylake_time"] = skylake_time
            result["skylake_speedup"] = skylake_speedup
            result["skylake_error"] = skylake_error
        else:
            print(f"  Skylake (AVX-512 F32): N/A")

        # Genoa
        genoa_result = benchmark_smallfield_genoa(mol_data, n_iterations)
        if genoa_result is not None:
            genoa_energy, genoa_time = genoa_result
            genoa_speedup = rdkit_time / genoa_time
            genoa_error = abs(genoa_energy - rdkit_energy)
            print(
                f"  Genoa (AVX-512 BF16):  {genoa_energy:10.4f} kcal/mol  {genoa_time:8.4f} ms  ({genoa_speedup:5.2f}x)  error: {genoa_error:.2e}"
            )
            result["genoa_energy"] = genoa_energy
            result["genoa_time"] = genoa_time
            result["genoa_speedup"] = genoa_speedup
            result["genoa_error"] = genoa_error
        else:
            print(f"  Genoa (AVX-512 BF16):  N/A")

        # Turin
        turin_result = benchmark_smallfield_turin(mol_data, n_iterations)
        if turin_result is not None:
            turin_energy, turin_time = turin_result
            turin_speedup = rdkit_time / turin_time
            turin_error = abs(turin_energy - rdkit_energy)
            print(
                f"  Turin (AVX-512 FP16):  {turin_energy:10.4f} kcal/mol  {turin_time:8.4f} ms  ({turin_speedup:5.2f}x)  error: {turin_error:.2e}"
            )
            result["turin_energy"] = turin_energy
            result["turin_time"] = turin_time
            result["turin_speedup"] = turin_speedup
            result["turin_error"] = turin_error
        else:
            print(f"  Turin (AVX-512 FP16):  N/A")

        # CUDA F32
        cuda_f32_result = benchmark_smallfield_cuda_f32(mol_data, n_iterations)
        if cuda_f32_result is not None:
            cuda_f32_energy, cuda_f32_time = cuda_f32_result
            cuda_f32_speedup = rdkit_time / cuda_f32_time
            cuda_f32_error = abs(cuda_f32_energy - rdkit_energy)
            print(
                f"  CUDA (f32):            {cuda_f32_energy:10.4f} kcal/mol  {cuda_f32_time:8.4f} ms  ({cuda_f32_speedup:5.2f}x)  error: {cuda_f32_error:.2e}"
            )
            result["cuda_f32_energy"] = cuda_f32_energy
            result["cuda_f32_time"] = cuda_f32_time
            result["cuda_f32_speedup"] = cuda_f32_speedup
            result["cuda_f32_error"] = cuda_f32_error
        else:
            print(f"  CUDA (f32):            N/A")

        # CUDA F16
        cuda_f16_result = benchmark_smallfield_cuda_f16(mol_data, n_iterations)
        if cuda_f16_result is not None:
            cuda_f16_energy, cuda_f16_time = cuda_f16_result
            cuda_f16_speedup = rdkit_time / cuda_f16_time
            cuda_f16_error = abs(cuda_f16_energy - rdkit_energy)
            print(
                f"  CUDA (f16):            {cuda_f16_energy:10.4f} kcal/mol  {cuda_f16_time:8.4f} ms  ({cuda_f16_speedup:5.2f}x)  error: {cuda_f16_error:.2e}"
            )
            result["cuda_f16_energy"] = cuda_f16_energy
            result["cuda_f16_time"] = cuda_f16_time
            result["cuda_f16_speedup"] = cuda_f16_speedup
            result["cuda_f16_error"] = cuda_f16_error
        else:
            print(f"  CUDA (f16):            N/A")

        print()
        results.append(result)

    # Summary table
    print("=" * 120)
    print("SUMMARY: Speedup vs RDKit")
    print("=" * 120)
    header = f"{'Molecule':<30} {'Serial-f64':>12} {'Serial-f32':>12} {'Skylake':>12} {'Genoa':>12} {'Turin':>12} {'CUDA-f32':>12} {'CUDA-f16':>12}"
    print(header)
    print("-" * 120)
    for r in results:
        skylake_str = (
            f"{r.get('skylake_speedup', 0):11.2f}x"
            if "skylake_speedup" in r
            else "         N/A"
        )
        genoa_str = (
            f"{r.get('genoa_speedup', 0):11.2f}x"
            if "genoa_speedup" in r
            else "         N/A"
        )
        turin_str = (
            f"{r.get('turin_speedup', 0):11.2f}x"
            if "turin_speedup" in r
            else "         N/A"
        )
        cuda_f32_str = (
            f"{r.get('cuda_f32_speedup', 0):11.2f}x"
            if "cuda_f32_speedup" in r
            else "         N/A"
        )
        cuda_f16_str = (
            f"{r.get('cuda_f16_speedup', 0):11.2f}x"
            if "cuda_f16_speedup" in r
            else "         N/A"
        )
        print(
            f"{r['smiles']:<30} {r['serial_f64_speedup']:11.2f}x {r['serial_f32_speedup']:11.2f}x {skylake_str} {genoa_str} {turin_str} {cuda_f32_str} {cuda_f16_str}"
        )
    print()

    print("=" * 120)
    print("SUMMARY: Error vs RDKit (kcal/mol)")
    print("=" * 120)
    header = f"{'Molecule':<30} {'Serial-f64':>12} {'Serial-f32':>12} {'Skylake':>12} {'Genoa':>12} {'Turin':>12} {'CUDA-f32':>12} {'CUDA-f16':>12}"
    print(header)
    print("-" * 120)
    for r in results:
        skylake_str = (
            f"{r.get('skylake_error', 0):11.2e}"
            if "skylake_error" in r
            else "         N/A"
        )
        genoa_str = (
            f"{r.get('genoa_error', 0):11.2e}" if "genoa_error" in r else "         N/A"
        )
        turin_str = (
            f"{r.get('turin_error', 0):11.2e}" if "turin_error" in r else "         N/A"
        )
        cuda_f32_str = (
            f"{r.get('cuda_f32_error', 0):11.2e}"
            if "cuda_f32_error" in r
            else "         N/A"
        )
        cuda_f16_str = (
            f"{r.get('cuda_f16_error', 0):11.2e}"
            if "cuda_f16_error" in r
            else "         N/A"
        )
        print(
            f"{r['smiles']:<30} {r['serial_f64_error']:11.2e} {r['serial_f32_error']:11.2e} {skylake_str} {genoa_str} {turin_str} {cuda_f32_str} {cuda_f16_str}"
        )
    print()


# Pytest fixtures and tests for pytest-benchmark integration
if PYTEST_AVAILABLE:
    import pytest

    @pytest.fixture(scope="module")
    def test_molecules():
        """Fixture providing test molecules for benchmarking."""
        smiles_list = [
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        ]
        return [create_molecule(smiles) for smiles in smiles_list]

    @pytest.fixture(scope="module")
    def mol_data_list(test_molecules):
        """Fixture providing molecule data containers."""
        return [MoleculeData(mol) for mol in test_molecules]

    @pytest.mark.parametrize("mol_idx", [0, 1, 2])
    def test_bench_rdkit(benchmark, mol_data_list, mol_idx):
        """Benchmark RDKit MMFF94."""
        mol_data = mol_data_list[mol_idx]
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_data.mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol_data.mol, mmff_props)
        benchmark(ff.CalcEnergy)

    @pytest.mark.parametrize("mol_idx", [0, 1, 2])
    def test_bench_serial_f64(benchmark, mol_data_list, mol_idx):
        """Benchmark SmallField serial f64."""
        mol_data = mol_data_list[mol_idx]
        benchmark(
            smallfield.compute_vdw_energy_serial_f64,
            mol_data.coords_x_f64,
            mol_data.coords_y_f64,
            mol_data.coords_z_f64,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f64,
            mol_data.r0_f64,
        )

    @pytest.mark.parametrize("mol_idx", [0, 1, 2])
    def test_bench_serial_f32(benchmark, mol_data_list, mol_idx):
        """Benchmark SmallField serial f32."""
        mol_data = mol_data_list[mol_idx]
        benchmark(
            smallfield.compute_vdw_energy_serial_f32,
            mol_data.coords_x_f32,
            mol_data.coords_y_f32,
            mol_data.coords_z_f32,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f32,
            mol_data.r0_f32,
        )

    @pytest.mark.skipif(not smallfield.has_skylake(), reason="Skylake not available")
    @pytest.mark.parametrize("mol_idx", [0, 1, 2])
    def test_bench_skylake(benchmark, mol_data_list, mol_idx):
        """Benchmark SmallField AVX-512 F32 (Skylake)."""
        mol_data = mol_data_list[mol_idx]
        benchmark(
            smallfield.compute_vdw_energy_skylake_f32,
            mol_data.coords_x_f32,
            mol_data.coords_y_f32,
            mol_data.coords_z_f32,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f32,
            mol_data.r0_f32,
        )

    @pytest.mark.skipif(not smallfield.has_cuda(), reason="CUDA not available")
    @pytest.mark.parametrize("mol_idx", [0, 1, 2])
    def test_bench_cuda_f32(benchmark, mol_data_list, mol_idx):
        """Benchmark SmallField CUDA f32."""
        mol_data = mol_data_list[mol_idx]
        mol_nb_start = np.array([0], dtype=np.int32)
        mol_nb_end = np.array([len(mol_data.pair_i)], dtype=np.int32)
        benchmark(
            smallfield.compute_vdw_energy_cuda_f32,
            mol_data.coords_x_f32,
            mol_data.coords_y_f32,
            mol_data.coords_z_f32,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f32,
            mol_data.r0_f32,
            mol_nb_start,
            mol_nb_end,
            1,
        )

    @pytest.mark.skipif(not smallfield.has_cuda(), reason="CUDA not available")
    @pytest.mark.parametrize("mol_idx", [0, 1, 2])
    def test_bench_cuda_f16(benchmark, mol_data_list, mol_idx):
        """Benchmark SmallField CUDA f16."""
        mol_data = mol_data_list[mol_idx]
        mol_nb_start = np.array([0], dtype=np.int32)
        mol_nb_end = np.array([len(mol_data.pair_i)], dtype=np.int32)
        benchmark(
            smallfield.compute_vdw_energy_cuda_f16,
            mol_data.coords_x_f16,
            mol_data.coords_y_f16,
            mol_data.coords_z_f16,
            mol_data.pair_i,
            mol_data.pair_j,
            mol_data.epsilon_f16,
            mol_data.r0_f16,
            mol_nb_start,
            mol_nb_end,
            1,
        )


if __name__ == "__main__":
    # Default molecules for benchmarking
    test_smiles = [
        "CCO",  # Ethanol - small molecule
        "c1ccccc1",  # Benzene - aromatic ring
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen - drug-like
        "CC(C)(C)c1ccc(O)cc1",  # BHT - larger molecule
    ]

    run_comprehensive_benchmark(test_smiles, n_iterations=1000)
