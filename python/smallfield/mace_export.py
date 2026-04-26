"""Export MACE-OFF weights and golden traces for the SmallField CUDA C++ inference module.

Phase 0 deliverable: takes a MACE-OFF checkpoint (small/medium/large), produces
two artefacts the C++ side reads as the source of truth:

  1) <out>/MACE-OFF23-<size>.weights.bin   — flat little-endian binary, one
     contiguous blob with a JSON manifest at the end describing every tensor's
     name, dtype, shape, and offset.  C++ loads via mmap + json.

  2) <out>/golden_<molname>.bin            — for each of five reference
     molecules we run one fp64 forward through the Python model and dump every
     intermediate tensor (post-radial, post-sph-harm, post-interaction[k],
     post-product[k], readouts, final energy, final forces).  Same flat-blob +
     JSON-manifest layout.  C++ unit tests use this as the regression target.

Run:
    python3 -m smallfield.mace_export --model small --out /tmp/mace_export

The script is intentionally conservative: every tensor is stored fp64 in the
weights blob so the C++ side decides its own working dtype (fp32 reference,
bf16 production).  Golden traces are stored fp64 too — the C++ test does the
fp32 / bf16 cast on its side and reports max-abs and rms error.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import torch
from ase import Atoms

from mace.calculators import mace_off
from mace.data import AtomicData, config_from_atoms
from mace.tools import AtomicNumberTable, torch_geometric


# ----------------------------------------------------------------------------
# Reference molecules — picked for size variety and chemical diversity.  Each
# is a small drug-like SMILES that contains H/C/N/O so we exercise multiple
# rows of the atomic-energy table.  Coordinates are MMFF-optimised so the
# inputs are in-distribution for MACE-OFF.
# ----------------------------------------------------------------------------
REFERENCE_MOLECULES = [
    ("paracetamol", "CC(=O)NC1=CC=C(O)C=C1"),
    ("ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("propranolol", "CC(C)NCC(O)COC1=CC=CC2=CC=CC=C12"),
    ("aspirin", "O=C(C)Oc1ccccc1C(=O)O"),
]


# ----------------------------------------------------------------------------
# Binary-blob format: tiny by design.
#
#   magic (8 bytes):      b"SF_MACE\0"
#   version (u32):        currently 1
#   manifest_offset (u64): byte offset where the JSON manifest starts
#   manifest_length (u64): JSON manifest byte length
#   <data>                contiguous tensor data, fp64 little-endian
#   <manifest JSON>       manifest = {"tensors": [{"name", "dtype", "shape", "offset", "nbytes"}, ...], "header": {...}}
#
# C++ side: read the 28-byte preamble, seek to manifest, parse JSON, then mmap
# offsets into the data region.
# ----------------------------------------------------------------------------

MAGIC = b"SF_MACE\0"
FORMAT_VERSION = 1
PREAMBLE_FMT = "<8sIQQ"  # magic, version, manifest_offset, manifest_length
PREAMBLE_SIZE = struct.calcsize(PREAMBLE_FMT)


def _to_numpy_fp64(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(torch.float64).contiguous().numpy()


def _to_numpy_int64(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(torch.int64).contiguous().numpy()


class BlobWriter:
    """Writes tensors to a flat binary, building a manifest as it goes."""

    def __init__(self, header: Dict):
        self._buf = bytearray()
        self._tensors: List[Dict] = []
        self._header = dict(header)

    def add(self, name: str, arr: np.ndarray) -> None:
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        offset = len(self._buf)
        nbytes = arr.nbytes
        self._buf.extend(arr.tobytes())
        self._tensors.append(
            {
                "name": name,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "offset": offset,
                "nbytes": nbytes,
            }
        )

    def write(self, path: Path) -> None:
        manifest = {"header": self._header, "tensors": self._tensors}
        manifest_bytes = json.dumps(manifest, indent=0).encode("utf-8")
        manifest_offset = PREAMBLE_SIZE + len(self._buf)
        manifest_length = len(manifest_bytes)
        preamble = struct.pack(
            PREAMBLE_FMT,
            MAGIC,
            FORMAT_VERSION,
            manifest_offset,
            manifest_length,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(preamble)
            f.write(self._buf)
            f.write(manifest_bytes)

    def n_tensors(self) -> int:
        return len(self._tensors)

    def n_bytes(self) -> int:
        return len(self._buf)


# ----------------------------------------------------------------------------
# Weight extraction — we walk model.state_dict() and dump everything.
# Architectural metadata (channels, L_max, irreps, correlation order) goes
# into the JSON header so the C++ side does not have to introspect names.
# ----------------------------------------------------------------------------


def _infer_architecture(model: torch.nn.Module) -> Dict:
    """Pull the architectural shape parameters out of the loaded model.

    These values are constants for a given MACE-OFF size, but reading them off
    the live model is more robust than hard-coding."""
    sd = model.state_dict()
    n_elements = int(sd["atomic_numbers"].numel())
    n_radial = int(sd["radial_embedding.bessel_fn.bessel_weights"].numel())
    # channels = output dim of node_embedding linear / n_elements (one-hot in)
    embed_w = sd["node_embedding.linear.weight"]
    n_channels = embed_w.numel() // n_elements
    # MACE-OFF uses L_max = 3 — read from any irrep-shaped output_mask.
    # The interaction.0.linear.output_mask has size n_channels * n_irreps.
    out_mask_dim = int(sd["interactions.0.linear.output_mask"].numel())
    n_irreps_msg = out_mask_dim // n_channels  # 16 for L_max=3 (1+3+5+7)
    L_max = {1: 0, 4: 1, 9: 2, 16: 3, 25: 4}[n_irreps_msg]
    # correlation_order: number of U_matrix_k tensors in product 0
    correlation_order = 0
    while f"products.0.symmetric_contractions.contractions.0.U_matrix_{correlation_order + 1}" in sd:
        correlation_order += 1
    # num_interactions
    num_interactions = int(sd["num_interactions"].item())
    # scale / shift
    scale = float(sd["scale_shift.scale"].item())
    shift = float(sd["scale_shift.shift"].item())
    # cutoff polynomial order
    cutoff_p = float(sd["radial_embedding.cutoff_fn.p"].item())
    # r_max
    r_max = float(sd["r_max"].item())
    # bessel_prefactor
    bessel_prefactor = float(sd["radial_embedding.bessel_fn.prefactor"].item())
    # avg_num_neighbors — hyperparameter divided into messages post-scatter
    avg_num_neighbors = float(model.interactions[0].avg_num_neighbors)
    # conv_tp path-normalisation factor — e3nn applies 1/sqrt(num_paths) per
    # output irrep inside its TensorProduct.  We pull the *actual* multiplier
    # out of the first instruction's `path_weight`; for MACE-OFF the values
    # are uniform across the 4 paths so first-element is sufficient.  If the
    # attribute layout shifts in a future e3nn we fall back to 1.0 (matches
    # the current hard-coded value in launch_mace_tensor_product).
    try:
        conv_tp = model.interactions[0].conv_tp
        conv_tp_path_weight = float(conv_tp.instructions[0].path_weight)
    except Exception as exc:
        print(f"warning: could not extract conv_tp path_weight ({exc!r}); defaulting to 1.0")
        conv_tp_path_weight = 1.0
    return {
        "n_elements": n_elements,
        "n_radial": n_radial,
        "n_channels": n_channels,
        "n_irreps_msg": n_irreps_msg,  # full message irrep count (16 for L=3)
        "L_max": L_max,
        "correlation_order": correlation_order,
        "num_interactions": num_interactions,
        "scale": scale,
        "shift": shift,
        "cutoff_p": cutoff_p,
        "r_max": r_max,
        "bessel_prefactor": bessel_prefactor,
        "avg_num_neighbors": avg_num_neighbors,
        "conv_tp_path_weight": conv_tp_path_weight,
    }


def export_weights(model: torch.nn.Module, out_path: Path, model_size: str) -> Dict:
    arch = _infer_architecture(model)
    arch["model_size"] = model_size
    arch["mace_version"] = getattr(__import__("mace"), "__version__", "unknown")

    writer = BlobWriter(arch)
    sd = model.state_dict()

    # Atomic numbers + atomic energies first (small, frequently accessed)
    writer.add("atomic_numbers", _to_numpy_int64(sd["atomic_numbers"]))
    writer.add("atomic_energies", _to_numpy_fp64(sd["atomic_energies_fn.atomic_energies"]))
    writer.add("bessel_weights", _to_numpy_fp64(sd["radial_embedding.bessel_fn.bessel_weights"]))

    # Node embedding: maps one-hot element index to channels.
    writer.add("node_embedding.weight", _to_numpy_fp64(sd["node_embedding.linear.weight"]))

    # Per interaction layer
    for k in range(arch["num_interactions"]):
        writer.add(f"interactions.{k}.linear_up.weight", _to_numpy_fp64(sd[f"interactions.{k}.linear_up.weight"]))
        # conv_tp_weights MLP (4 layers, fixed shapes)
        for li in range(4):
            writer.add(
                f"interactions.{k}.conv_tp_weights.layer{li}.weight",
                _to_numpy_fp64(sd[f"interactions.{k}.conv_tp_weights.layer{li}.weight"]),
            )
        writer.add(f"interactions.{k}.linear.weight", _to_numpy_fp64(sd[f"interactions.{k}.linear.weight"]))
        writer.add(f"interactions.{k}.skip_tp.weight", _to_numpy_fp64(sd[f"interactions.{k}.skip_tp.weight"]))

    # Per product (symmetric contraction)
    for k in range(arch["num_interactions"]):
        for u in range(1, arch["correlation_order"] + 1):
            writer.add(
                f"products.{k}.U_matrix_{u}",
                _to_numpy_fp64(sd[f"products.{k}.symmetric_contractions.contractions.0.U_matrix_{u}"]),
            )
        # weights_max for max correlation order
        writer.add(
            f"products.{k}.weights_max",
            _to_numpy_fp64(sd[f"products.{k}.symmetric_contractions.contractions.0.weights_max"]),
        )
        # lower-order weights, indexed 0 .. correlation_order-2
        for w_idx in range(arch["correlation_order"] - 1):
            writer.add(
                f"products.{k}.weights.{w_idx}",
                _to_numpy_fp64(sd[f"products.{k}.symmetric_contractions.contractions.0.weights.{w_idx}"]),
            )
        writer.add(f"products.{k}.linear.weight", _to_numpy_fp64(sd[f"products.{k}.linear.weight"]))

    # Readouts.  readouts[0] is a LinearReadout; readouts[-1] is NonLinearReadout (MLP).
    writer.add("readouts.0.linear.weight", _to_numpy_fp64(sd["readouts.0.linear.weight"]))
    last = arch["num_interactions"] - 1
    if last >= 1 and f"readouts.{last}.linear_1.weight" in sd:
        writer.add(f"readouts.{last}.linear_1.weight", _to_numpy_fp64(sd[f"readouts.{last}.linear_1.weight"]))
        writer.add(f"readouts.{last}.linear_2.weight", _to_numpy_fp64(sd[f"readouts.{last}.linear_2.weight"]))

    writer.write(out_path)
    print(f"  wrote {writer.n_tensors()} tensors / {writer.n_bytes()/1e6:.2f} MB to {out_path}")
    return arch


# ----------------------------------------------------------------------------
# Golden trace — run one fp64 forward on a fixed deterministic batch of 1
# molecule and snapshot every layer-boundary tensor that the C++ side will
# need to validate against.
# ----------------------------------------------------------------------------


def _build_atoms_from_smiles(smi: str) -> Atoms:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.MolFromSmiles(smi)
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(m)
    conf = m.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(j)) for j in range(m.GetNumAtoms())], dtype=np.float64)
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    return Atoms(symbols=syms, positions=pos)


def _run_with_trace(model: torch.nn.Module, calc, atoms: Atoms) -> Dict[str, np.ndarray]:
    """Run one fp64 forward pass, attach hooks to capture every intermediate."""
    z_table = AtomicNumberTable([int(z) for z in calc.models[0].atomic_numbers])
    data_list = [AtomicData.from_config(config_from_atoms(atoms), z_table=z_table, cutoff=float(calc.r_max))]
    loader = torch_geometric.dataloader.DataLoader(dataset=data_list, batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(next(model.parameters()).device)
    bd = batch.to_dict()
    bd["positions"].requires_grad_(True)

    captured: Dict[str, torch.Tensor] = {}

    def grab(name):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                captured[name] = t.detach().clone()

        return hook

    handles = []
    handles.append(model.radial_embedding.register_forward_hook(grab("radial")))
    handles.append(model.spherical_harmonics.register_forward_hook(grab("sph_harm")))
    handles.append(model.node_embedding.register_forward_hook(grab("node_embed")))
    for k in range(model.num_interactions.item()):
        handles.append(model.interactions[k].register_forward_hook(grab(f"inter_{k}_out")))
        handles.append(model.products[k].register_forward_hook(grab(f"prod_{k}_out")))
        if k < len(model.readouts):
            handles.append(model.readouts[k].register_forward_hook(grab(f"readout_{k}")))

    try:
        out = model(bd, compute_force=True, training=False)
    finally:
        for h in handles:
            h.remove()

    captured["positions"] = bd["positions"].detach().clone()
    captured["edge_index"] = bd["edge_index"].detach().clone()
    captured["shifts"] = (
        bd.get("shifts", torch.zeros((bd["edge_index"].shape[1], 3), device=bd["positions"].device)).detach().clone()
    )
    captured["node_attrs"] = bd["node_attrs"].detach().clone()
    captured["energy"] = out["energy"].detach().clone()
    captured["forces"] = out["forces"].detach().clone()
    captured["node_energy"] = out["node_energy"].detach().clone()
    return {k: v.cpu().double().numpy() for k, v in captured.items()}


def export_golden(model: torch.nn.Module, calc, out_dir: Path, arch: Dict) -> None:
    """Run + dump golden tensors for each reference molecule."""
    for name, smi in REFERENCE_MOLECULES:
        atoms = _build_atoms_from_smiles(smi)
        captured = _run_with_trace(model, calc, atoms)
        writer = BlobWriter(
            {
                "molecule": name,
                "smiles": smi,
                "n_atoms": int(captured["positions"].shape[0]),
                "n_edges": int(captured["edge_index"].shape[1]),
                "model_size": arch["model_size"],
            }
        )
        for tname, arr in captured.items():
            writer.add(tname, arr)
        path = out_dir / f"golden_{arch['model_size']}_{name}.bin"
        writer.write(path)
        e = float(captured["energy"].item() if captured["energy"].size == 1 else captured["energy"].sum())
        print(
            f"  {name:<14s} natoms={captured['positions'].shape[0]:3d}  nedges={captured['edge_index'].shape[1]:4d}  E={e:.6f} eV  -> {path.name}"
        )


# ----------------------------------------------------------------------------
# Sample fp16 batch — concatenate the 5 reference molecules into one CSR-style
# graph batch and dump fp16 positions alongside the original fp64 positions
# so downstream batched / fp16 work can validate round-trip error.
# ----------------------------------------------------------------------------


def export_sample_fp16_batch(model: torch.nn.Module, out_path: Path) -> None:
    atomic_numbers = model.state_dict()["atomic_numbers"].cpu().to(torch.int64).numpy()
    z_to_index = {int(z): i for i, z in enumerate(atomic_numbers)}
    from ase.data import atomic_numbers as ase_z

    positions_list: List[np.ndarray] = []
    elements_list: List[np.ndarray] = []
    offsets: List[int] = [0]
    for name, smi in REFERENCE_MOLECULES:
        atoms = _build_atoms_from_smiles(smi)
        pos = np.asarray(atoms.get_positions(), dtype=np.float64)
        zs = [int(ase_z[s]) for s in atoms.get_chemical_symbols()]
        idx = np.asarray([z_to_index[z] for z in zs], dtype=np.int64)
        positions_list.append(pos)
        elements_list.append(idx)
        offsets.append(offsets[-1] + pos.shape[0])
        print(f"  {name:<14s} natoms={pos.shape[0]:3d}  offset={offsets[-2]}")

    positions_fp64 = np.concatenate(positions_list, axis=0).astype(np.float64)
    positions_fp16 = positions_fp64.astype(np.float16)
    atom_element = np.concatenate(elements_list, axis=0).astype(np.int64)
    graph_atom_offset = np.asarray(offsets, dtype=np.int64)

    header = {
        "n_atoms_total": int(positions_fp64.shape[0]),
        "n_graphs": len(REFERENCE_MOLECULES),
        "graph_atom_offset": [int(x) for x in offsets],
    }
    writer = BlobWriter(header)
    writer.add("positions_fp16", positions_fp16)
    writer.add("positions_fp32_reference", positions_fp64)
    writer.add("atom_element", atom_element)
    writer.add("graph_atom_offset", graph_atom_offset)
    writer.write(out_path)
    print(
        f"  wrote sample fp16 batch: {writer.n_tensors()} tensors / "
        f"{writer.n_bytes()/1e3:.2f} KB / n_atoms_total={header['n_atoms_total']} -> {out_path.name}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", choices=["small", "medium", "large"])
    ap.add_argument("--out", default="/tmp/mace_export")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-golden", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading MACE-OFF-{args.model} on {args.device} (fp64)...")
    calc = mace_off(model=args.model, device=args.device, default_dtype="float64", enable_cueq=False)
    model = calc.models[0]
    model.eval()

    weights_path = out_dir / f"MACE-OFF23-{args.model}.weights.bin"
    print(f"\nexporting weights to {weights_path}")
    arch = export_weights(model, weights_path, args.model)

    print(f"\narchitecture: {json.dumps(arch, indent=2)}")

    if not args.skip_golden:
        print("\ndumping golden traces:")
        export_golden(model, calc, out_dir, arch)

    fp16_batch_path = out_dir / "sample_fp16_batch.bin"
    print(f"\nbuilding sample fp16 batch -> {fp16_batch_path}")
    export_sample_fp16_batch(model, fp16_batch_path)


if __name__ == "__main__":
    main()
