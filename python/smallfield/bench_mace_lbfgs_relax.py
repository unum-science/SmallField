"""End-to-end LBFGS relaxation throughput on a batch of drug-like molecules.

Times the full relaxation loop, not just one forward pass.  ASE LBFGS is
single-molecule, so to get batched relaxation throughput we step every
conformer's optimiser using the model's batched energy+forces output.

Reports:
  - ms per LBFGS step per conformer (the load-bearing metric)
  - total relaxation time for K steps × N conformers
  - extrapolated single-card throughput in mols/sec for K=20 step relaxation
"""

import argparse, time, warnings, json
warnings.filterwarnings("ignore")
import numpy as np
import torch
from ase import Atoms
from mace.calculators import mace_off
from mace.data import AtomicData, config_from_atoms
from mace.tools import AtomicNumberTable, torch_geometric


def make_drug_like(rng, n_atoms):
    nC = max(1, n_atoms // 3)
    nN = max(1, n_atoms // 8)
    nO = max(1, n_atoms // 8)
    nH = n_atoms - nC - nN - nO
    syms = ["C"] * nC + ["N"] * nN + ["O"] * nO + ["H"] * nH
    pos = rng.standard_normal((n_atoms, 3)) * 0.4 + np.arange(n_atoms)[:, None] * 0.4
    return Atoms(symbols=syms, positions=pos)


def build_batch(calc, atoms_list, device):
    z_table = AtomicNumberTable([int(z) for z in calc.models[0].atomic_numbers])
    r_max = float(calc.r_max)
    data_list = [
        AtomicData.from_config(config_from_atoms(a), z_table=z_table, cutoff=r_max)
        for a in atoms_list
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=data_list, batch_size=len(data_list), shuffle=False, drop_last=False
    )
    batch = next(iter(loader)).to(device)
    return batch


def gradient_descent_step(positions, forces, lr=0.05):
    """Plain GD step — surrogate for one LBFGS step's GPU work.  Real LBFGS
    has a tiny CPU-side line search but its GPU work is dominated by the
    energy+forces evaluation, which is what we measure here.
    """
    return positions + lr * forces


def bench_relax(calc, batch_size, n_atoms, n_steps=20, n_warmup=3):
    rng = np.random.default_rng(0)
    atoms_list = [make_drug_like(rng, n_atoms) for _ in range(batch_size)]
    batch = build_batch(calc, atoms_list, "cuda")
    model = calc.models[0]
    model.eval()

    # Warmup
    for _ in range(n_warmup):
        out = model(batch.to_dict(), compute_force=True, training=False)
        torch.cuda.synchronize()

    # Timed N-step relaxation loop.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(n_steps):
        out = model(batch.to_dict(), compute_force=True, training=False)
        forces = out["forces"]
        # GD step (stand-in for LBFGS's GPU work).  No line search; that
        # would add a few % overhead on the host but no extra GPU work.
        batch.positions = (batch.positions + 0.05 * forces).detach()
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "batch": batch_size,
        "n_atoms": n_atoms,
        "n_steps": n_steps,
        "total_ms": total_ms,
        "ms_per_step_per_config": total_ms / n_steps / batch_size,
        "ms_per_step_total_batch": total_ms / n_steps,
        "configs_per_sec_in_relax": batch_size * n_steps / (total_ms / 1000.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--n-atoms", default="15,30,50")
    ap.add_argument("--batches", default="256,512,1024")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--out", default="/tmp/mace_cueq_lbfgs.jsonl")
    args = ap.parse_args()

    n_atoms_list = [int(x) for x in args.n_atoms.split(",")]
    batch_list = [int(x) for x in args.batches.split(",")]

    calc = mace_off(model=args.model, device="cuda", default_dtype="float32", enable_cueq=True)

    results = []
    for n_atoms in n_atoms_list:
        for bs in batch_list:
            try:
                r = bench_relax(calc, bs, n_atoms, n_steps=args.n_steps)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                r = {"batch": bs, "n_atoms": n_atoms, "error": "oom"}
            r["model"] = args.model
            r["cueq"] = True
            print(json.dumps(r))
            results.append(r)

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
