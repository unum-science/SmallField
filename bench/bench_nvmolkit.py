#!/usr/bin/env python3
"""
Config B benchmark: RDKit ETKDG (CPU) + nvMolKit MMFF (GPU).

Run from the USearchMolecules pixi environment:
  pixi run python tests/bench_nvmolkit.py
"""

import os
import sys
import time

# standard test molecules (same as C++ bench)
SMILES = [
    "CCO", "CC(=O)O", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "CC(=O)Nc1ccc(O)cc1", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
]


def main():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        print("RDKit not available")
        return

    num_conformers = 3

    for batch_n in [100, 500, 1000]:
        smiles_list = []
        while len(smiles_list) < batch_n:
            for s in SMILES:
                smiles_list.append(s)
                if len(smiles_list) >= batch_n:
                    break

        # step 1: RDKit ETKDG on CPU
        t0 = time.perf_counter()
        mols = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.numThreads = 1
            AllChem.EmbedMultipleConfs(mol, num_conformers, params)
            mols.append(mol)
        t_etkdg = time.perf_counter() - t0

        # step 2: nvMolKit MMFF on GPU
        try:
            nvmolkit_site_packages = os.environ.get("NVMOLKIT_SITE_PACKAGES")
            if nvmolkit_site_packages:
                sys.path.insert(0, nvmolkit_site_packages)
            from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

            t0 = time.perf_counter()
            MMFFOptimizeMoleculesConfs(mols, maxIters=200)
            t_mmff = time.perf_counter() - t0
        except Exception as e:
            t_mmff = float("nan")
            print(f"  nvMolKit error: {e}")

        total_ms = (t_etkdg + t_mmff) * 1000
        mol_per_sec = batch_n / (t_etkdg + t_mmff) if t_mmff == t_mmff else 0
        print(
            f"  Config B (RDKit+nvMolKit) batch={batch_n:5d}: "
            f"etkdg={t_etkdg*1000:.0f}ms mmff={t_mmff*1000:.0f}ms "
            f"total={total_ms:.0f}ms {mol_per_sec:.1f} mol/s"
        )


if __name__ == "__main__":
    main()
