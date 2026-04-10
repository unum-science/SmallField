"""
SmallField: Fast MMFF94 force field implementation with CPU (AVX-512) and GPU (CUDA) support.

This module provides Python bindings to optimized MMFF94 energy calculations for:
- Van der Waals energy (buffered 14-7 potential)
- Electrostatic energy (Coulomb with buffer)

Supported backends:
- Serial CPU (f32, f64)
- AVX-512 CPU (Skylake F32, Genoa BF16, Turin FP16)
- CUDA GPU (f16, f32) - H100 and newer

Usage with Context (recommended for repeated calculations):
    >>> import smallfield
    >>> import numpy as np
    >>>
    >>> # Create a context with pre-allocated buffers
    >>> ctx = smallfield.Context(
    ...     total_atoms=1000,
    ...     total_pairs=5000,
    ...     max_molecules=10
    ... )
    >>>
    >>> # Prepare data (f64 arrays - conversion happens internally)
    >>> coords_x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    >>> coords_y = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    >>> coords_z = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    >>> pair_i = np.array([0, 1], dtype=np.int32)
    >>> pair_j = np.array([1, 2], dtype=np.int32)
    >>> epsilon = np.array([0.1, 0.1], dtype=np.float64)
    >>> r0 = np.array([1.5, 1.5], dtype=np.float64)
    >>>
    >>> # Compute energy using different backends
    >>> energy_f32 = ctx.compute_vdw_energy_serial_f32(
    ...     coords_x, coords_y, coords_z,
    ...     pair_i, pair_j, epsilon, r0
    ... )
    >>>
    >>> # For CUDA:
    >>> # energy_cuda = ctx.compute_vdw_energy_cuda_f32(
    >>> #     coords_x, coords_y, coords_z,
    >>> #     nb_i, nb_j, epsilon, r_star,
    >>> #     mol_nb_start, mol_nb_end, n_molecules
    >>> # )

Usage without Context (one-off calculations):
    >>> import smallfield
    >>> import numpy as np
    >>>
    >>> # Check capabilities
    >>> has_cuda = smallfield.has_cuda()
    >>> has_skylake = smallfield.has_skylake()
    >>>
    >>> # Compute van der Waals energy
    >>> coords_x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    >>> coords_y = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    >>> coords_z = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    >>> pair_i = np.array([0, 1], dtype=np.int32)
    >>> pair_j = np.array([1, 2], dtype=np.int32)
    >>> epsilon = np.array([0.1, 0.1], dtype=np.float32)
    >>> r0 = np.array([1.5, 1.5], dtype=np.float32)
    >>>
    >>> energy = smallfield.compute_vdw_energy_serial_f32(
    ...     coords_x, coords_y, coords_z,
    ...     pair_i, pair_j, epsilon, r0
    ... )
"""

from typing import Union, Tuple, Optional
import numpy as np
import numpy.typing as npt

__version__ = "0.1.0"

# Type aliases for clarity
ArrayF32 = npt.NDArray[np.float32]
ArrayF64 = npt.NDArray[np.float64]
ArrayI32 = npt.NDArray[np.int32]
Scalar = Union[float, np.floating]

# Import the C++ extension module
try:
    import _smallfield

    # Core capabilities
    supports_cuda = _smallfield.supports_cuda
    supports_avx512_f32 = _smallfield.supports_avx512_f32
    supports_avx512_bf16 = _smallfield.supports_avx512_bf16
    supports_avx512_fp16 = _smallfield.supports_avx512_fp16

    # Serial kernels (always available)
    compute_vdw_energy_serial_f32 = _smallfield.compute_vdw_energy_serial_f32
    compute_vdw_energy_serial_f64 = _smallfield.compute_vdw_energy_serial_f64
    compute_electrostatic_energy_serial_f32 = (
        _smallfield.compute_electrostatic_energy_serial_f32
    )
    compute_electrostatic_energy_serial_f64 = (
        _smallfield.compute_electrostatic_energy_serial_f64
    )

    # AVX-512 Skylake kernels (optional)
    compute_vdw_energy_skylake_f32: Optional[callable] = None
    compute_electrostatic_energy_skylake_f32: Optional[callable] = None
    try:
        compute_vdw_energy_skylake_f32 = _smallfield.compute_vdw_energy_skylake_f32
        compute_electrostatic_energy_skylake_f32 = (
            _smallfield.compute_electrostatic_energy_skylake_f32
        )
    except AttributeError:
        pass

    # AVX-512 Genoa kernels (optional)
    compute_vdw_energy_genoa_bf16: Optional[callable] = None
    try:
        compute_vdw_energy_genoa_bf16 = _smallfield.compute_vdw_energy_genoa_bf16
    except AttributeError:
        pass

    # AVX-512 Turin kernels (optional)
    compute_vdw_energy_turin_fp16: Optional[callable] = None
    try:
        compute_vdw_energy_turin_fp16 = _smallfield.compute_vdw_energy_turin_fp16
    except AttributeError:
        pass

    # CUDA kernels (optional)
    compute_vdw_energy_cuda_f16: Optional[callable] = None
    compute_vdw_energy_cuda_f32: Optional[callable] = None
    compute_electrostatic_energy_cuda_f16: Optional[callable] = None
    compute_electrostatic_energy_cuda_f32: Optional[callable] = None
    try:
        compute_vdw_energy_cuda_f16 = _smallfield.compute_vdw_energy_cuda_f16
        compute_vdw_energy_cuda_f32 = _smallfield.compute_vdw_energy_cuda_f32
        compute_electrostatic_energy_cuda_f16 = (
            _smallfield.compute_electrostatic_energy_cuda_f16
        )
        compute_electrostatic_energy_cuda_f32 = (
            _smallfield.compute_electrostatic_energy_cuda_f32
        )
    except AttributeError:
        pass

    # SmallFieldContext class
    Context: Optional[type] = None
    try:
        Context = _smallfield.Context
    except AttributeError:
        pass

    # ETKDG conformer generation pipeline (CUDA only)
    EtkdgContext: Optional[type] = None
    try:
        EtkdgContext = _smallfield.EtkdgContext
    except AttributeError:
        pass

except ImportError as e:
    raise ImportError(
        "Failed to import _smallfield extension module. "
        "Please build the extension with: pip install -e ."
    ) from e


def has_cuda() -> bool:
    """Check if CUDA support is available and compiled."""
    return compute_vdw_energy_cuda_f32 is not None and supports_cuda()


def has_skylake() -> bool:
    """Check if AVX-512 F32 (Skylake) support is available."""
    return compute_vdw_energy_skylake_f32 is not None and supports_avx512_f32()


def has_genoa() -> bool:
    """Check if AVX-512 BF16 (Genoa) support is available."""
    return compute_vdw_energy_genoa_bf16 is not None and supports_avx512_bf16()


def has_turin() -> bool:
    """Check if AVX-512 FP16 (Turin) support is available."""
    return compute_vdw_energy_turin_fp16 is not None and supports_avx512_fp16()


__all__ = [
    # Capabilities
    "supports_cuda",
    "supports_avx512_f32",
    "supports_avx512_bf16",
    "supports_avx512_fp16",
    "has_cuda",
    "has_skylake",
    "has_genoa",
    "has_turin",
    # Serial kernels
    "compute_vdw_energy_serial_f32",
    "compute_vdw_energy_serial_f64",
    "compute_electrostatic_energy_serial_f32",
    "compute_electrostatic_energy_serial_f64",
    # AVX-512 kernels
    "compute_vdw_energy_skylake_f32",
    "compute_electrostatic_energy_skylake_f32",
    "compute_vdw_energy_genoa_bf16",
    "compute_vdw_energy_turin_fp16",
    # CUDA kernels
    "compute_vdw_energy_cuda_f16",
    "compute_vdw_energy_cuda_f32",
    "compute_electrostatic_energy_cuda_f16",
    "compute_electrostatic_energy_cuda_f32",
    # Context classes
    "Context",
    "EtkdgContext",
    # High-level conformer generation
    "generate_conformers",
]


def generate_conformers(smiles: str, num_conformers: int = 10,
                        max_stereo_attempts: int = 20,
                        seed: int = 42) -> Optional[np.ndarray]:
    """Generate 3D conformers for a molecule using GPU ETKDG pipeline.

    Args:
        smiles: SMILES string for the molecule.
        num_conformers: Number of conformers to generate.
        max_stereo_attempts: Maximum retry attempts per conformer on stereo failure.
        seed: Random seed for reproducibility.

    Returns:
        numpy array of shape (K, N, 3) with 3D coordinates for successful conformers,
        or None if the GPU pipeline is not available.

    Requires RDKit for molecular preprocessing.
    """
    if EtkdgContext is None:
        return None

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDistGeom
    except ImportError:
        raise ImportError("RDKit is required for ETKDG conformer generation")

    import struct

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    num_atoms = mol.GetNumAtoms()

    # bounds matrix (already smoothed by RDKit)
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    bounds_flat = bounds.astype(np.float32).flatten()

    # embed to get 3D geometry for chiral sign computation
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

    # chiral centers: 8 bytes each
    chirals = []
    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag not in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                       Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            continue
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 3:
            continue
        conf = mol.GetConformer()
        ra = conf.GetAtomPosition(atom.GetIdx())
        rb = conf.GetAtomPosition(neighbors[0])
        rc = conf.GetAtomPosition(neighbors[1])
        rd = conf.GetAtomPosition(neighbors[2])
        bv = (rb.x - ra.x, rb.y - ra.y, rb.z - ra.z)
        cv = (rc.x - ra.x, rc.y - ra.y, rc.z - ra.z)
        dv = (rd.x - ra.x, rd.y - ra.y, rd.z - ra.z)
        volume = (bv[0] * (cv[1]*dv[2] - cv[2]*dv[1])
                + bv[1] * (cv[2]*dv[0] - cv[0]*dv[2])
                + bv[2] * (cv[0]*dv[1] - cv[1]*dv[0]))
        target_sign = 1 if volume > 0 else -1
        descriptor = struct.pack("BBBBBbBB",
            atom.GetIdx(),
            neighbors[0], neighbors[1] if len(neighbors) > 1 else 0,
            neighbors[2] if len(neighbors) > 2 else 0,
            neighbors[3] if len(neighbors) > 3 else 0,
            target_sign, 0, 0)
        chirals.append(descriptor)

    chirals_bytes = b"".join(chirals) if chirals else b""

    # torsion terms: 32 bytes each
    torsions = []
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() > 1.5:
            continue
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        i_atom = next((n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors()
                       if n.GetIdx() != k), None)
        l_atom = next((n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors()
                       if n.GetIdx() != j), None)
        if i_atom is None or l_atom is None:
            continue
        descriptor = struct.pack("BBBBBBBB6f",
            i_atom, j, k, l_atom, 0, 0, 0, 0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        torsions.append(descriptor)

    torsions_bytes = b"".join(torsions) if torsions else b""

    # distance constraints: 16 bytes each
    constraints = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        descriptor = struct.pack("BBBBfff", i, j, 0, 0, 1.425, 1.575, 10.0)
        constraints.append(descriptor)

    constraints_bytes = b"".join(constraints) if constraints else b""

    # metadata: 32 bytes
    metadata = struct.pack("IIIIIIII",
        num_atoms, len(chirals), len(torsions), len(constraints),
        0, 0, 0, 0)

    ctx = EtkdgContext(num_atoms, 1, num_conformers)
    coords_flat, energies, status = ctx.generate_conformers(
        bounds_flat,
        np.frombuffer(chirals_bytes, dtype=np.uint8) if chirals_bytes else np.array([], dtype=np.uint8),
        np.frombuffer(torsions_bytes, dtype=np.uint8) if torsions_bytes else np.array([], dtype=np.uint8),
        np.frombuffer(constraints_bytes, dtype=np.uint8) if constraints_bytes else np.array([], dtype=np.uint8),
        np.frombuffer(metadata, dtype=np.uint8),
        num_conformers, max_stereo_attempts, seed)

    coords_array = np.array(coords_flat).reshape(num_conformers, num_atoms * 3)
    status_array = np.array(status)

    successful = []
    for k_idx in range(num_conformers):
        if status_array[k_idx] == 0:
            soa = coords_array[k_idx]
            xyz = np.zeros((num_atoms, 3), dtype=np.float32)
            xyz[:, 0] = soa[0:num_atoms]
            xyz[:, 1] = soa[num_atoms:2*num_atoms]
            xyz[:, 2] = soa[2*num_atoms:3*num_atoms]
            successful.append(xyz)

    return np.stack(successful) if successful else None
