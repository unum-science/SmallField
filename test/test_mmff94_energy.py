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
    gpu_context = sfa.Context([])
    coordinates, energies, status_codes = gpu_context.generate_conformers(
        smiles_batch,
        2,
        128,
        10,
        True,
        42,
    )

    expected_conformer_count = len(smiles_batch) * 2
    assert np.asarray(coordinates).shape == (expected_conformer_count * 128 * 3,)
    assert np.asarray(energies).shape == (expected_conformer_count,)
    assert np.asarray(status_codes).shape == (expected_conformer_count,)
    assert np.isfinite(np.asarray(energies)).all()
    assert any(int(code) == int(sfa.ConformerStatus.success) for code in status_codes)


def test_invalid_smiles_raises_value_error():
    with pytest.raises(ValueError, match="Invalid SMILES"):
        sfa.generate_conformers("not_a_smiles")
