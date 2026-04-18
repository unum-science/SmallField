"""Python bindings for the SmallField API surface."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

try:
    import _smallfield
except ImportError as exc:  # pragma: no cover - import failure is the signal
    raise ImportError(
        "Failed to import the _smallfield extension module. Build it with `pip install -e .`."
    ) from exc


Context = _smallfield.Context
ConformerStatus = _smallfield.ConformerStatus
supports_cuda = _smallfield.supports_cuda


def has_cuda() -> bool:
    """Return whether the compiled extension can use CUDA."""

    return supports_cuda()


def _rdkit_num_atoms(smiles: str) -> int:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError("RDKit is required for `smallfield.api.generate_conformers`.") from exc

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    molecule = Chem.AddHs(molecule)
    return molecule.GetNumAtoms()


def _reshape_single_molecule(
    coordinates_flat: np.ndarray,
    status_codes: np.ndarray,
    num_conformers: int,
    max_atoms_per_molecule: int,
    num_atoms: int,
) -> Optional[np.ndarray]:
    successful_conformers = []
    for conformer_index in range(num_conformers):
        if int(status_codes[conformer_index]) != int(ConformerStatus.success):
            continue
        conformer_offset = conformer_index * max_atoms_per_molecule * 3
        soa_coordinates = coordinates_flat[
            conformer_offset : conformer_offset + max_atoms_per_molecule * 3
        ]
        interleaved = np.zeros((num_atoms, 3), dtype=np.float32)
        interleaved[:, 0] = soa_coordinates[0:max_atoms_per_molecule][:num_atoms]
        interleaved[:, 1] = soa_coordinates[max_atoms_per_molecule : 2 * max_atoms_per_molecule][:num_atoms]
        interleaved[:, 2] = soa_coordinates[2 * max_atoms_per_molecule : 3 * max_atoms_per_molecule][:num_atoms]
        successful_conformers.append(interleaved)
    if not successful_conformers:
        return None
    return np.stack(successful_conformers)


def generate_conformers(
    smiles: str | Sequence[str],
    num_conformers: int = 3,
    max_atoms_per_molecule: int = 128,
    max_stereo_attempts: int = 10,
    random_seed: int = 42,
    device_ids: Optional[Iterable[int]] = None,
):
    """Generate conformers through the SmallField Python API.

    For a single SMILES string, returns a `(K, N, 3)` float32 array of successful conformers
    or `None` when none succeed.

    For a sequence of SMILES strings, returns the raw `(coordinates, energies, status)` tuple
    from `smallfield.api.Context.generate_conformers`.
    """

    gpu_context = Context(list(device_ids or []))
    if isinstance(smiles, str):
        num_atoms = _rdkit_num_atoms(smiles)
        coordinates, _energies, status = gpu_context.generate_conformers(
            [smiles],
            num_conformers,
            max_atoms_per_molecule,
            max_stereo_attempts,
            random_seed,
        )
        return _reshape_single_molecule(
            np.asarray(coordinates, dtype=np.float32),
            np.asarray(status, dtype=np.uint8),
            num_conformers,
            max_atoms_per_molecule,
            num_atoms,
        )

    smiles_list = list(smiles)
    return gpu_context.generate_conformers(
        smiles_list,
        num_conformers,
        max_atoms_per_molecule,
        max_stereo_attempts,
        random_seed,
    )


__all__ = [
    "ConformerStatus",
    "Context",
    "generate_conformers",
    "has_cuda",
    "supports_cuda",
]
