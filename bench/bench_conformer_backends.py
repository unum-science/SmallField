#!/usr/bin/env python3
"""Side-by-side conformer-generation benchmark across pure RDKit, SmallField, and RDKit + nvMolKit.

nvMolKit is driven in small chunks because it aborts the whole call on a molecule it does not support,
so a chunk is the unit that can be lost rather than the run.
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


def load_smiles(path: str, max_molecules: int) -> List[str]:
    """Read one SMILES per line, first whitespace field only, up to ``max_molecules``."""
    smiles: List[str] = []
    with open(path) as handle:
        for line in handle:
            fields = line.strip().split()
            if not fields:
                continue
            smiles.append(fields[0])
            if max_molecules and len(smiles) >= max_molecules:
                break
    return smiles


def parse_molecules(smiles: List[str]) -> List["object"]:
    """Parse and hydrogenate every SMILES that RDKit accepts, outside any timed region.

    Shared by the two systems that hand RDKit molecules rather than strings, so both start from the
    same set of molecules.
    """
    from rdkit import Chem

    molecules = []
    for entry in smiles:
        molecule = Chem.MolFromSmiles(entry)
        if molecule is None:
            continue
        molecules.append(Chem.AddHs(molecule))
    return molecules


class Result:
    """One system's benchmark result."""

    def __init__(self, name: str):
        self.name = name
        self.wall_seconds: Optional[float] = None
        self.molecules_per_second: Optional[float] = None
        self.conformers_per_second: Optional[float] = None
        self.num_success: Optional[int] = None
        self.num_target: Optional[int] = None
        self.note: str = ""

    def record(
        self, wall_seconds: float, num_success: int, num_target: int, num_molecules: int, note: str = ""
    ) -> None:
        self.wall_seconds = wall_seconds
        self.molecules_per_second = num_molecules / wall_seconds if wall_seconds > 0 else 0.0
        self.conformers_per_second = num_success / wall_seconds if wall_seconds > 0 else 0.0
        self.num_success = num_success
        self.num_target = num_target
        self.note = note


def bench_rdkit(smiles: List[str], num_conformers: int, random_seed: int) -> Result:
    from rdkit.Chem import AllChem, rdDistGeom

    result = Result("RDKit (CPU, OpenMP)")
    parameters = rdDistGeom.ETKDGv3()
    parameters.randomSeed = random_seed
    parameters.numThreads = 0  # auto, use all cores

    molecules = parse_molecules(smiles)

    started = time.perf_counter()
    num_success = 0
    for molecule in molecules:
        try:
            conformer_ids = AllChem.EmbedMultipleConfs(molecule, numConfs=num_conformers, params=parameters)
            if conformer_ids:
                AllChem.MMFFOptimizeMoleculeConfs(molecule)
                num_success += len(conformer_ids)
        except Exception:
            continue
    wall = time.perf_counter() - started
    result.record(wall, num_success, len(smiles) * num_conformers, len(smiles))
    return result


def bench_smallfield(smiles: List[str], num_conformers: int, random_seed: int, gpu_ids: List[int]) -> Result:
    import smallfield.api as smallfield

    result = Result("SmallField (GPU)")
    # One reused context plus the warm-up below keeps device init and the first-call kernel load out of the timing.
    context = smallfield.Context(gpu_ids)
    smallfield.generate_conformers(
        smiles[: min(100, len(smiles))],
        num_conformers=num_conformers,
        random_seed=random_seed,
        context=context,
    )

    started = time.perf_counter()
    batch = smallfield.generate_conformers(
        smiles,
        num_conformers=num_conformers,
        random_seed=random_seed,
        context=context,
    )
    wall = time.perf_counter() - started

    status = np.asarray(batch.status, dtype=np.uint8)
    num_success = int(np.sum(status == int(smallfield.ConformerStatus.SUCCESS)))
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
    parameters = rdDistGeom.ETKDGv3()
    parameters.randomSeed = random_seed
    parameters.useRandomCoords = True

    molecules = parse_molecules(smiles)

    hardware = HardwareOptions()
    hardware.gpuIds = gpu_ids

    num_molecules = len(molecules)
    total_conformers = 0
    failed_chunks = 0
    started = time.perf_counter()
    for chunk_start in range(0, num_molecules, chunk_size):
        chunk = [Chem.RWMol(molecule) for molecule in molecules[chunk_start : chunk_start + chunk_size]]
        try:
            EmbedMolecules(chunk, parameters, confsPerMolecule=num_conformers, hardwareOptions=hardware)
        except RuntimeError:
            # The count is reported beside the throughput it depressed.
            failed_chunks += 1
            continue
        try:
            MMFFOptimizeMoleculesConfs(chunk, hardwareOptions=hardware)
        except RuntimeError:
            pass
        total_conformers += sum(molecule.GetNumConformers() for molecule in chunk)
    wall = time.perf_counter() - started

    num_chunks = (num_molecules + chunk_size - 1) // chunk_size
    note = f"{failed_chunks}/{num_chunks} chunks crashed" if failed_chunks else ""
    result.record(wall, total_conformers, num_molecules * num_conformers, num_molecules, note=note)
    return result


def render_results_table(results: Iterable[Result]) -> None:
    table = Table(title="Conformer-generation benchmark", show_header=True, header_style="bold")
    table.add_column("System")
    table.add_column("Wall (s)", justify="right")
    table.add_column("mol/s", justify="right", style="green")
    table.add_column("conf/s", justify="right", style="green")
    table.add_column("Success", justify="right")
    table.add_column("Notes")
    for result in results:
        if result.wall_seconds is None:
            table.add_row(result.name, "—", "—", "—", "—", result.note or "skipped")
            continue
        success_share = (100.0 * result.num_success / result.num_target) if result.num_target else 0.0
        table.add_row(
            result.name,
            f"{result.wall_seconds:.2f}",
            thousands(int(result.molecules_per_second)),
            thousands(int(result.conformers_per_second)),
            f"{thousands(result.num_success)} / {thousands(result.num_target)} ({success_share:.3f} %)",
            result.note or "",
        )
    _default_log.console.print(table)


def selected_systems(args: argparse.Namespace, smiles: List[str], gpu_ids: List[int]) -> List[tuple]:
    """Pair each system the flags left enabled with the call that runs it."""
    systems: List[tuple[str, Callable[[], Result]]] = []
    if not args.skip_rdkit:
        systems.append(("RDKit", lambda: bench_rdkit(smiles, args.num_conformers, args.random_seed)))
    if not args.skip_smallfield:
        systems.append(("SmallField", lambda: bench_smallfield(smiles, args.num_conformers, args.random_seed, gpu_ids)))
    if not args.skip_nvmolkit:
        systems.append(
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
    return systems


def main() -> int:
    args = parse_args()
    gpu_ids: List[int] = [int(entry) for entry in args.gpus.split(",") if entry.strip()]

    smiles = load_smiles(args.input, args.max_mol)
    if not smiles:
        log.error("no SMILES read from {}", args.input)
        return 1
    log.info(
        "loaded {} SMILES from {}, generating {} conformers each",
        thousands(len(smiles)),
        args.input,
        args.num_conformers,
    )

    results: List[Result] = []
    failures = 0
    for label, runner in selected_systems(args, smiles, gpu_ids):
        log.info("running {}", label)
        try:
            results.append(runner())
        except Exception as error:
            # One system failing is a result about that system, so the rest still run and the table still prints.
            log.error("{} failed: {}", label, error)
            failed = Result(label)
            failed.note = f"error: {error}"
            results.append(failed)
            failures += 1

    render_results_table(results)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
