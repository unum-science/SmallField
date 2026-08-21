"""
Convert a SMILES file (one SMILES per line, optionally with whitespace-separated name)
into a folder of Parquet shards that ``smallfield_batch`` can consume.

Output shards have two columns: ``id`` (string) and ``smiles`` (string). Shard
filenames are deterministic given the input path and ``--rows-per-shard``::

    shard_0000000000_0000999999.parquet   # rows [0, 1_000_000)
    shard_0001000000_0001999999.parquet   # rows [1_000_000, 2_000_000)

Re-running with the same arguments overwrites. Pass ``--workers N`` to parallelise
shard writes across N processes (I/O bound above ~4 workers).

Example::

    python -m smallfield.tools.smi_to_parquet \\
        --input data/usearch-molecules/pubchem/smiles \\
        --output data/usearch-molecules/pubchem/parquet \\
        --rows-per-shard 1000000 \\
        --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Iterator, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from smallfield.log import Logger

_log = Logger()


SHARD_NAME_FMT = "shard_{start:010d}_{end:010d}.parquet"


def _iter_smiles_file(path: Path) -> Iterator[Tuple[str, str]]:
    """Yield (id, smiles) pairs from a SMILES file.

    Each line is split on whitespace. Column 1 is SMILES; column 2 (if present)
    is treated as the id. Lines starting with ``#`` or empty lines are skipped.
    If no id is supplied, the 1-based line number is used.
    """
    with open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line_number, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(None, 1)
            smiles = parts[0]
            identifier = parts[1].strip() if len(parts) > 1 else str(line_number)
            yield identifier, smiles


def _write_shard_batch(
    output_dir: Path,
    start_row: int,
    end_row: int,
    identifiers: List[str],
    smiles: List[str],
    compression: str,
) -> Path:
    """Write one shard containing rows [start_row, end_row)."""
    assert len(identifiers) == len(smiles)
    table = pa.table(
        {
            "id": pa.array(identifiers, type=pa.string()),
            "smiles": pa.array(smiles, type=pa.string()),
        }
    )
    output_path = output_dir / SHARD_NAME_FMT.format(start=start_row, end=end_row - 1)
    pq.write_table(table, output_path, compression=compression)
    return output_path


def _worker_write_shard(task: Tuple[Path, int, int, List[str], List[str], str]) -> str:
    """Subprocess wrapper: unpacks the tuple and writes one shard."""
    output_dir, start_row, end_row, identifiers, smiles, compression = task
    path = _write_shard_batch(output_dir, start_row, end_row, identifiers, smiles, compression)
    return str(path)


def convert_smi_to_parquet(
    input_path: Path,
    output_dir: Path,
    rows_per_shard: int,
    workers: int = 1,
    compression: str = "zstd",
) -> int:
    """Convert a SMILES file into Parquet shards.

    Returns the total number of rows written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_identifiers: List[str] = []
    batch_smiles: List[str] = []
    batch_start_row = 0
    global_row = 0

    pending_tasks: List[Tuple[Path, int, int, List[str], List[str], str]] = []
    pool = mp.Pool(workers) if workers > 1 else None

    def flush_batch(start_row: int, identifiers: List[str], smiles: List[str]):
        """Write the shard covering ``identifiers`` and ``smiles`` from ``start_row``, or queue it for the pool.

        A queued task keeps its rows resident until the pool is drained after the last batch.
        """
        end_row = start_row + len(identifiers)
        task = (output_dir, start_row, end_row, identifiers, smiles, compression)
        if pool is None:
            _worker_write_shard(task)
        else:
            pending_tasks.append(task)

    for identifier, smiles in _iter_smiles_file(input_path):
        batch_identifiers.append(identifier)
        batch_smiles.append(smiles)
        global_row += 1
        if len(batch_identifiers) >= rows_per_shard:
            flush_batch(batch_start_row, batch_identifiers, batch_smiles)
            batch_start_row += len(batch_identifiers)
            batch_identifiers = []
            batch_smiles = []

    if batch_identifiers:
        flush_batch(batch_start_row, batch_identifiers, batch_smiles)

    if pool is not None:
        for out_path in pool.imap_unordered(_worker_write_shard, pending_tasks):
            _log.info("wrote {}", out_path)
        pool.close()
        pool.join()

    return global_row


def main() -> int:
    """Parse the command line, run `convert_smi_to_parquet`, and return the process exit code.

    ``--compression none`` becomes ``None`` rather than the string, which is how PyArrow spells uncompressed.
    """
    parser = argparse.ArgumentParser(
        description="Convert a SMILES file into Parquet shards for smallfield_batch.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .smi file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=1_000_000,
        help="Rows per output Parquet shard (default: 1_000_000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel shard writes (default: 1)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["snappy", "gzip", "brotli", "zstd", "none"],
        help="Parquet compression codec (default: zstd)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")

    compression = args.compression if args.compression != "none" else None
    num_rows = convert_smi_to_parquet(
        input_path=args.input,
        output_dir=args.output,
        rows_per_shard=args.rows_per_shard,
        workers=max(1, args.workers),
        compression=compression,
    )
    _log.info("wrote {} rows to {}", f"{num_rows:,}", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
