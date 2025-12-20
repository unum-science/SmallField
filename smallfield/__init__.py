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
    # Context class
    "Context",
]
