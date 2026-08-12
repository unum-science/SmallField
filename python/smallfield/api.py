"""Python bindings for the SmallField API surface."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

try:
    import _smallfield
except ImportError as exc:  # pragma: no cover - import failure is the signal
    raise ImportError("Failed to import the _smallfield extension module. Build it with `pip install -e .`.") from exc


Context = _smallfield.Context
ConformerBatch = _smallfield.ConformerBatch
ConformerStatus = _smallfield.ConformerStatus
supports_cuda = _smallfield.supports_cuda

#: Atom ceiling the warp pipeline supports, and the stride every conformer slot is padded to.
#: Read from the extension rather than restated, because the two layers disagreeing about it means
#: the same flat buffer is addressed with two different strides.
MAX_ATOMS_PER_MOLECULE = _smallfield.MAX_ATOMS_PER_MOLECULE


def _widen_successful_conformers(
    batch: ConformerBatch,
    num_conformers: int,
    max_atoms_per_molecule: int,
    num_atoms: int,
) -> Optional[np.ndarray]:
    """Interleave one molecule's successful slots into a contiguous ``(K, N, 3)`` float32 array.

    A slot is structure-of-arrays — ``[x_0..x_{max-1}][y_0..][z_0..]`` — so the reshape names that
    layout and the transpose is what makes the result atom-major. Failed slots are dropped, so ``K``
    is at most ``num_conformers`` and ``None`` means every slot failed.
    """

    succeeded = batch.status[:num_conformers] == int(ConformerStatus.SUCCESS)
    if not succeeded.any():
        return None
    slots = batch.coordinates.reshape(num_conformers, 3, max_atoms_per_molecule)
    return np.ascontiguousarray(slots[succeeded, :, :num_atoms].transpose(0, 2, 1), dtype=np.float32)


def generate_conformers(
    smiles: str | Sequence[str],
    num_conformers: int = 3,
    max_atoms_per_molecule: int = MAX_ATOMS_PER_MOLECULE,
    max_stereo_attempts: int = 10,
    random_seed: int = 42,
    device_ids: Optional[Iterable[int]] = None,
    context: Optional[Context] = None,
) -> ConformerBatch | np.ndarray | None:
    """Generate conformers through the SmallField Python API.

    For a single SMILES string, returns a ``(K, N, 3)`` float32 array of the conformers that
    succeeded, sorted by ascending MMFF94 energy, or ``None`` when every conformer failed. A SMILES
    that preprocessing rejected outright raises ``ValueError`` instead, so "unusable input" and
    "nothing converged" do not both arrive as ``None``.

    For a sequence of SMILES strings, returns the ``ConformerBatch`` from
    ``smallfield.api.Context.generate_conformers`` unchanged. Its ``coordinates`` are ``float16`` —
    the precision the pipeline actually produces — and flat, padded to ``max_atoms_per_molecule``;
    ``num_atoms[i]`` is the post-preprocess atom count (with H) for molecule ``i``, or ``0`` if
    preprocess rejected it. ``num_heavy_atoms``, ``num_bonds`` and ``molecular_weight`` come from the
    same molecule SmallField embedded, so a caller populates the canonical parquet schema columns
    without a duplicate parse. Preprocessing keeps only the largest fragment, so for a salt these
    describe the parent rather than the input string.

    Building a ``Context`` initializes every visible GPU, so pass one in through ``context`` to reuse
    it across calls; ``device_ids`` builds a throwaway one for a single call.
    """

    if context is not None and device_ids is not None:
        raise ValueError("pass either `context` or the `device_ids` to build one from, not both")
    gpu_context = context if context is not None else Context(list(device_ids or []))

    single_molecule = isinstance(smiles, str)
    batch = gpu_context.generate_conformers(
        [smiles] if single_molecule else list(smiles),
        num_conformers,
        max_atoms_per_molecule,
        max_stereo_attempts,
        random_seed,
    )
    if not single_molecule:
        return batch

    num_atoms = int(batch.num_atoms[0])
    if num_atoms == 0:
        status = ConformerStatus(int(batch.status[0]))
        raise ValueError(f"Invalid SMILES or unsupported molecule {smiles!r}: {status}")
    return _widen_successful_conformers(batch, num_conformers, max_atoms_per_molecule, num_atoms)


__all__ = [
    "MAX_ATOMS_PER_MOLECULE",
    "ConformerBatch",
    "ConformerStatus",
    "Context",
    "generate_conformers",
    "supports_cuda",
]
