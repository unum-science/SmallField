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
    or `None` when none succeed. Conformers are sorted by MMFF energy ascending; failed
    conformers are dropped.

    For a sequence of SMILES strings, returns the raw
    `(coordinates, energies, status, num_atoms, num_heavy_atoms, num_bonds, molecular_weight)`
    tuple from `smallfield.api.Context.generate_conformers`. Conformers within each molecule
    are pre-sorted by energy ascending (failed conformers last); `num_atoms[i]` is the
    post-preprocess atom count (with H) for molecule `i`, or `0` if preprocess rejected it.
    The remaining three readback arrays (`num_heavy_atoms`, `num_bonds`, `molecular_weight`)
    come from the same post-AddHs RDKit mol used by SmallField's preprocess, so callers can
    populate the canonical parquet schema columns without a duplicate parse.
    """

    gpu_context = Context(list(device_ids or []))
    if isinstance(smiles, str):
        result = gpu_context.generate_conformers(
            [smiles],
            num_conformers,
            max_atoms_per_molecule,
            max_stereo_attempts,
            random_seed,
        )
        coordinates, _energies, status, num_atoms_arr = result[0], result[1], result[2], result[3]
        num_atoms = int(np.asarray(num_atoms_arr, dtype=np.uint8)[0])
        if num_atoms == 0:
            return None
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


# MACE-OFF inference + refinement surface (Phase I).  These re-export the
# pybind11-wrapped C++ classes from `_smallfield` and add a high-level
# `refine_with_mace` helper that builds the edge list / element vector
# from RDKit so a user can hand it just (coords, smiles).
MaceArch = _smallfield.MaceArch
MaceWeights = _smallfield.MaceWeights
MaceContext = _smallfield.MaceContext
MaceStatus = _smallfield.MaceStatus


# MACE-OFF supports these 10 atomic numbers, in this order.  Mapping atomic
# number → element index must match the order baked into the weight blob
# (see `python/smallfield/mace_export.py::SUPPORTED_Z`).
_MACE_OFF_ATOMIC_NUMBERS = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)
_ATOMIC_NUMBER_TO_INDEX = {z: i for i, z in enumerate(_MACE_OFF_ATOMIC_NUMBERS)}


def _build_mace_batch_from_geometry(
    coords_per_molecule: list[np.ndarray],
    elements_per_molecule: list[np.ndarray],
    r_max: float,
):
    """Pack a list of per-molecule (coords, atomic_numbers) into the flat
    MACE batch layout: positions [N_total, 3], element index [N_total],
    edges (src, dst) [E_total], graph_of_atom [N_total].

    Edges are O(n²) intra-molecule pair checks against `r_max`.
    """
    flat_positions: list[np.ndarray] = []
    flat_elements: list[np.ndarray] = []
    edge_src: list[np.ndarray] = []
    edge_dst: list[np.ndarray] = []
    graph_of_atom: list[np.ndarray] = []
    atom_offset = 0
    for graph_index, (coords, elements) in enumerate(zip(coords_per_molecule, elements_per_molecule)):
        n = coords.shape[0]
        flat_positions.append(coords.astype(np.float32, copy=False))
        flat_elements.append(elements.astype(np.int32, copy=False))
        graph_of_atom.append(np.full((n,), graph_index, dtype=np.int32))
        if n > 1:
            d = coords[:, None, :] - coords[None, :, :]
            distances = np.sqrt((d * d).sum(axis=-1))
            mask = (distances < r_max) & (distances > 0.0)
            src_local, dst_local = np.where(mask)
            edge_src.append((src_local + atom_offset).astype(np.int32))
            edge_dst.append((dst_local + atom_offset).astype(np.int32))
        atom_offset += n
    positions = np.concatenate(flat_positions, axis=0) if flat_positions else np.empty((0, 3), dtype=np.float32)
    element_index = np.concatenate(flat_elements, axis=0) if flat_elements else np.empty((0,), dtype=np.int32)
    src = np.concatenate(edge_src, axis=0) if edge_src else np.empty((0,), dtype=np.int32)
    dst = np.concatenate(edge_dst, axis=0) if edge_dst else np.empty((0,), dtype=np.int32)
    graph = np.concatenate(graph_of_atom, axis=0) if graph_of_atom else np.empty((0,), dtype=np.int32)
    return positions, element_index, src, dst, graph


def _atomic_numbers_from_smiles(smiles: str) -> np.ndarray:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover
        raise ImportError("RDKit is required for `refine_with_mace`.") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    molecule = Chem.AddHs(molecule)
    z = np.asarray([atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.int32)
    try:
        return np.asarray([_ATOMIC_NUMBER_TO_INDEX[int(value)] for value in z], dtype=np.int32)
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Atomic number {exc.args[0]} is not in MACE-OFF's supported set "
                         f"{_MACE_OFF_ATOMIC_NUMBERS}") from exc


def refine_with_mace(
    coordinates: np.ndarray,
    smiles_list: Sequence[str],
    weights_path: str,
    num_steps: int = 20,
    fmax: float = 0.05,
    step_size: float = 0.01,
    device_id: int = 0,
):
    """High-level MACE-OFF refinement.

    Parameters
    ----------
    coordinates : np.ndarray
        Either `(N, 3)` for a single molecule, or `(M, N, 3)` for M conformers
        of the same molecule.  Float32 (will be cast).
    smiles_list : Sequence[str]
        One SMILES per molecule.  When `coordinates` has shape `(M, N, 3)`
        and `smiles_list` has length 1, the same SMILES applies to all M
        conformers.
    weights_path : str
        Path to the SmallField .bin weights blob (output of `mace_export.py`).
    num_steps : int
        Maximum LBFGS-style relaxation steps.
    fmax : float
        Convergence threshold on per-atom force magnitude (eV/Å).

    Returns
    -------
    refined : np.ndarray
        Same shape as input `coordinates`.
    energies : np.ndarray
        Per-conformer energies in eV, shape `(M,)` (or scalar for single input).

    Notes
    -----
    Edges are computed from positions with `r_max` from the weight blob's
    architecture metadata.  The mapping atomic-number → element-index uses
    MACE-OFF23's 10-element table {H, C, N, O, F, P, S, Cl, Br, I}.
    """
    coords = np.asarray(coordinates, dtype=np.float32)
    single_input = (coords.ndim == 2)
    if single_input:
        coords = coords[None, ...]

    weights = MaceWeights.load(weights_path)
    arch = weights.arch()

    smiles_seq = list(smiles_list)
    if len(smiles_seq) == 1 and coords.shape[0] != 1:
        smiles_seq = smiles_seq * coords.shape[0]
    if len(smiles_seq) != coords.shape[0]:
        raise ValueError(
            f"smiles_list ({len(smiles_seq)}) must match number of conformers ({coords.shape[0]})")

    coords_per_molecule = [coords[m] for m in range(coords.shape[0])]
    elements_per_molecule = [_atomic_numbers_from_smiles(smi) for smi in smiles_seq]
    for m, (c, e) in enumerate(zip(coords_per_molecule, elements_per_molecule)):
        if c.shape[0] != e.shape[0]:
            raise ValueError(f"conformer {m}: coords have {c.shape[0]} atoms, smiles has {e.shape[0]}")

    positions, element_index, edge_src, edge_dst, graph_of_atom = \
        _build_mace_batch_from_geometry(coords_per_molecule, elements_per_molecule, arch.r_max)

    n_atoms = positions.shape[0]
    n_edges = edge_src.shape[0]
    n_graphs = coords.shape[0]
    context = MaceContext(
        device_id=device_id,
        weights=weights,
        max_atoms=max(n_atoms, 1),
        max_edges=max(n_edges, 1),
        max_graphs=max(n_graphs, 1),
    )
    refined_positions, energies = context.refine(
        weights, positions, element_index, edge_src, edge_dst, graph_of_atom, n_graphs,
        num_steps=num_steps, fmax=fmax, step_size=step_size)

    refined_per_mol = []
    offset = 0
    for c in coords_per_molecule:
        n = c.shape[0]
        refined_per_mol.append(refined_positions[offset:offset + n])
        offset += n
    if single_input:
        return refined_per_mol[0], float(energies[0])
    return np.stack(refined_per_mol, axis=0), energies


__all__ = [
    "ConformerStatus",
    "Context",
    "MaceArch",
    "MaceContext",
    "MaceStatus",
    "MaceWeights",
    "generate_conformers",
    "has_cuda",
    "refine_with_mace",
    "supports_cuda",
]
