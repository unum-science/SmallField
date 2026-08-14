"""Pytest suite for the `smallfield.api` Python surface: package layout, single- and
batch-SMILES conformer generation, and error handling on invalid input.

Run with ``pytest test/test_mmff94_energy.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from rdkit import Chem

import smallfield
import smallfield.api as sfa


def _num_atoms_with_hydrogens(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    molecule = Chem.AddHs(molecule)
    return molecule.GetNumAtoms()


def test_package_surface_moves_api_under_smallfield_api():
    assert hasattr(smallfield, "api")
    assert not hasattr(smallfield, "Context")
    assert not hasattr(smallfield, "ConformerStatus")
    assert sfa.Context is smallfield.api.Context
    assert sfa.ConformerStatus is smallfield.api.ConformerStatus


def test_single_smiles_generation_returns_interleaved_coordinates():
    smiles = "CCO"
    coordinates = sfa.generate_conformers(smiles, num_conformers=2, random_seed=42)

    assert coordinates is not None
    assert coordinates.dtype == np.float32
    assert coordinates.ndim == 3
    assert coordinates.shape[1] == _num_atoms_with_hydrogens(smiles)
    assert coordinates.shape[2] == 3
    assert coordinates.shape[0] >= 1
    assert np.isfinite(coordinates).all()


def test_batch_generation_returns_flat_buffers_and_status_codes():
    smiles_batch = ["CCO", "c1ccccc1"]
    num_conformers = 2
    atom_stride = 64
    gpu_context = sfa.Context([])
    batch = gpu_context.generate_conformers(
        smiles_batch,
        num_conformers=num_conformers,
        max_atoms_per_molecule=atom_stride,
        random_seed=42,
    )

    expected_conformer_count = len(smiles_batch) * num_conformers
    # Coordinates cross as float16, which is the storage the pipeline produces; widening them here
    # would assert a precision the slots never held.
    assert batch.coordinates.dtype == np.float16
    assert batch.coordinates.shape == (expected_conformer_count * atom_stride * 3,)
    assert batch.energies.shape == (expected_conformer_count,)
    assert batch.status.shape == (expected_conformer_count,)
    assert np.isfinite(batch.energies).all()
    assert any(int(code) == int(sfa.ConformerStatus.SUCCESS) for code in batch.status)

    # The per-molecule descriptors come from the molecule SmallField embedded, so they are one per
    # input rather than one per conformer.
    assert batch.num_atoms.shape == (len(smiles_batch),)
    assert batch.num_heavy_atoms.shape == (len(smiles_batch),)
    assert batch.num_bonds.shape == (len(smiles_batch),)
    assert batch.molecular_weight.shape == (len(smiles_batch),)


def test_atom_stride_is_published_rather_than_restated():
    """The stride is the output layout, so a caller reads it rather than hard-coding 128 or 192.

    It stays under ``smallfield.api`` rather than the package root, which holds only submodules —
    the same reason ``Context`` is not re-exported there.
    """
    assert sfa.MAX_ATOMS_PER_MOLECULE > 0
    assert not hasattr(smallfield, "MAX_ATOMS_PER_MOLECULE")


def test_invalid_smiles_raises_value_error():
    with pytest.raises(ValueError, match="Invalid SMILES"):
        sfa.generate_conformers("not_a_smiles")


def test_empty_smiles_is_invalid_rather_than_a_successful_no_atom_molecule():
    """RDKit parses an empty string into a valid molecule of no atoms, and a null Parquet cell arrives
    as exactly that. Every slot must carry INVALID_SMILES, since a SUCCESS there is a molecule counted
    against the shipped success rate that no conformer was ever produced for.
    """
    gpu_context = sfa.Context([])
    batch = gpu_context.generate_conformers(["", "   ", "CCO"], num_conformers=2, random_seed=42)

    assert all(int(code) == int(sfa.ConformerStatus.INVALID_SMILES) for code in batch.status[:4])
    assert int(batch.num_atoms[0]) == 0 and int(batch.num_atoms[1]) == 0
    assert any(int(code) == int(sfa.ConformerStatus.SUCCESS) for code in batch.status[4:])
