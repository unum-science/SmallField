#!/usr/bin/env python3
"""Side-by-side conformer-generation benchmark across three systems.

Runs each selected system on the same input SMILES and reports wall time,
mol/s, conf/s, and success rate in a single rich.table.Table.

Systems:
  1. Pure RDKit Python — ``EmbedMultipleConfs`` + ``MMFFOptimizeMoleculeConfs``,
     OpenMP across molecules via RDKit's built-in ``numThreads``.
  2. SmallField Python API — ``smallfield.api.generate_conformers`` with
     internal RDKit preprocessing + CUDA ETKDG/MMFF kernels.
  3. RDKit + nvMolKit — ``nvmolkit.EmbedMolecules`` + ``MMFFOptimizeMoleculesConfs``,
     processed in small chunks to survive nvMolKit's hard runtime crashes.

Usage::

    python bench_conformers.py --input /tmp/usearch_1M_clean.smi \\
        --max-mol 100000 --num-conformers 3 --gpus 0,1,2,3

    python bench_conformers.py --input ... --skip-nvmolkit --skip-rdkit
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, Iterable, List, Optional

import numpy as np
from rich.table import Table

from smallfield.log import Logger, log as _default_log, thousands

log = Logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Side-by-side conformer-generation benchmark (RDKit / SmallField / nvMolKit)."
    )
    parser.add_argument("--input", required=True, help="Path to a .smi file (one SMILES per line).")
    parser.add_argument("--max-mol", type=int, default=100_000, help="Cap on molecules read from --input.")
    parser.add_argument("--num-conformers", type=int, default=3, help="Conformers per molecule.")
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="Comma-separated GPU IDs (e.g. '0,1,2,3'). Empty = all visible.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--skip-rdkit", action="store_true", help="Skip the pure-RDKit system.")
    parser.add_argument("--skip-smallfield", action="store_true", help="Skip the SmallField system.")
    parser.add_argument("--skip-nvmolkit", action="store_true", help="Skip the RDKit+nvMolKit system.")
    parser.add_argument(
        "--nvmolkit-chunk-size",
        type=int,
        default=5_000,
        help="nvMolKit chunk size (mitigates hard crashes on unsupported molecules).",
    )
    return parser.parse_args()


def load_smiles(path: str, max_mol: int) -> List[str]:
    smiles: List[str] = []
    with open(path) as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            smiles.append(parts[0])
            if max_mol and len(smiles) >= max_mol:
                break
    return smiles


class Result:
    """One system's benchmark result."""

    def __init__(self, name: str):
        self.name = name
        self.wall_s: Optional[float] = None
        self.mol_per_s: Optional[float] = None
        self.conf_per_s: Optional[float] = None
        self.num_success: Optional[int] = None
        self.num_target: Optional[int] = None
        self.note: str = ""

    def record(self, wall_s: float, num_success: int, num_target: int, num_mol: int, note: str = "") -> None:
        self.wall_s = wall_s
        self.mol_per_s = num_mol / wall_s if wall_s > 0 else 0.0
        self.conf_per_s = num_success / wall_s if wall_s > 0 else 0.0
        self.num_success = num_success
        self.num_target = num_target
        self.note = note


def bench_rdkit(smiles: List[str], num_conformers: int, random_seed: int) -> Result:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom

    result = Result("RDKit (CPU, OpenMP)")
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0  # auto, use all cores

    molecules = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        molecules.append(Chem.AddHs(mol))

    t0 = time.perf_counter()
    num_success = 0
    for mol in molecules:
        try:
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
            if conf_ids:
                AllChem.MMFFOptimizeMoleculeConfs(mol)
                num_success += len(conf_ids)
        except Exception:
            continue
    wall = time.perf_counter() - t0
    result.record(wall, num_success, len(smiles) * num_conformers, len(smiles))
    return result


def bench_smallfield(smiles: List[str], num_conformers: int, random_seed: int, gpu_ids: List[int]) -> Result:
    import smallfield.api as sf

    result = Result("SmallField (GPU)")
    # Warm up: build context + short generate_conformers to pay first-call costs.
    sf.Context(gpu_ids)
    sf.generate_conformers(
        smiles[: min(100, len(smiles))],
        num_conformers=num_conformers,
        random_seed=random_seed,
        device_ids=gpu_ids,
    )

    t0 = time.perf_counter()
    _coords, _energies, status = sf.generate_conformers(
        smiles,
        num_conformers=num_conformers,
        random_seed=random_seed,
        device_ids=gpu_ids,
    )
    wall = time.perf_counter() - t0

    status_arr = np.array(status, dtype=np.uint8)
    num_success = int(np.sum(status_arr == 0))
    result.record(wall, num_success, len(smiles) * num_conformers, len(smiles))
    return result


def bench_nvmolkit(
    smiles: List[str],
    num_conformers: int,
    random_seed: int,
    gpu_ids: List[int],
    chunk_size: int,
) -> Result:
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom
    from nvmolkit.embedMolecules import EmbedMolecules
    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
    from nvmolkit.types import HardwareOptions

    result = Result("RDKit + nvMolKit (GPU)")
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = random_seed
    params.useRandomCoords = True

    molecules = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        molecules.append(Chem.AddHs(mol))

    hw = HardwareOptions()
    hw.gpuIds = gpu_ids

    num_mol = len(molecules)
    total_confs = 0
    failed_chunks = 0
    t0 = time.perf_counter()
    for start in range(0, num_mol, chunk_size):
        mols = [Chem.RWMol(m) for m in molecules[start : start + chunk_size]]
        try:
            EmbedMolecules(mols, params, confsPerMolecule=num_conformers, hardwareOptions=hw)
        except RuntimeError:
            failed_chunks += 1
            continue
        try:
            MMFFOptimizeMoleculesConfs(mols, hardwareOptions=hw)
        except RuntimeError:
            pass
        total_confs += sum(m.GetNumConformers() for m in mols)
    wall = time.perf_counter() - t0

    num_chunks = (num_mol + chunk_size - 1) // chunk_size
    note = f"{failed_chunks}/{num_chunks} chunks crashed" if failed_chunks else ""
    result.record(wall, total_confs, num_mol * num_conformers, num_mol, note=note)
    return result


def render_results_table(results: Iterable[Result]) -> None:
    table = Table(title="Conformer-generation benchmark", show_header=True, header_style="bold")
    table.add_column("System")
    table.add_column("Wall (s)", justify="right")
    table.add_column("mol/s", justify="right", style="green")
    table.add_column("conf/s", justify="right", style="green")
    table.add_column("Success", justify="right")
    table.add_column("Notes")
    for r in results:
        if r.wall_s is None:
            table.add_row(r.name, "—", "—", "—", "—", "skipped")
            continue
        success_pct = (100.0 * r.num_success / r.num_target) if r.num_target else 0.0
        table.add_row(
            r.name,
            f"{r.wall_s:.2f}",
            thousands(int(r.mol_per_s)),
            thousands(int(r.conf_per_s)),
            f"{thousands(r.num_success)} / {thousands(r.num_target)} ({success_pct:.1f} %)",
            r.note or "",
        )
    _default_log.console.print(table)


def main() -> int:
    args = parse_args()
    gpu_ids: List[int] = [int(x) for x in args.gpus.split(",") if x.strip()]

    smiles = load_smiles(args.input, args.max_mol)
    log.info(
        "loaded {} SMILES from {}, generating {} conformers each",
        thousands(len(smiles)),
        args.input,
        args.num_conformers,
    )

    tasks: List[tuple[str, Callable[[], Result]]] = []
    if not args.skip_rdkit:
        tasks.append(("RDKit", lambda: bench_rdkit(smiles, args.num_conformers, args.random_seed)))
    if not args.skip_smallfield:
        tasks.append(
            ("SmallField", lambda: bench_smallfield(smiles, args.num_conformers, args.random_seed, gpu_ids))
        )
    if not args.skip_nvmolkit:
        tasks.append(
            (
                "nvMolKit",
                lambda: bench_nvmolkit(
                    smiles,
                    args.num_conformers,
                    args.random_seed,
                    gpu_ids,
                    args.nvmolkit_chunk_size,
                ),
            )
        )

    results: List[Result] = []
    for label, runner in tasks:
        log.info("running {}", label)
        try:
            results.append(runner())
        except Exception as exc:
            log.error("{} failed: {}", label, exc)
            r = Result(label)
            r.note = f"error: {exc}"
            results.append(r)

    render_results_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
