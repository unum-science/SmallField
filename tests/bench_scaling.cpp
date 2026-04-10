/**
 * @file bench_scaling.cpp
 * @brief Measures actual per-molecule GPU kernel time across batch sizes.
 * Uses a single ~100-atom molecule replicated to eliminate load imbalance.
 */

#include "etkdg_preprocess.hpp"
#include "test_helpers.hpp"
#include <cstdio>
#include <vector>

using namespace smallfield;
using namespace smallfield::test;

extern "C" float gpu_etkdg_pipeline_timed(float const*, std::uint32_t, chiral_center_descriptor_t const*, std::uint32_t,
                                          etk_torsion_descriptor_t const*, std::uint32_t, distance_constraint_t const*,
                                          std::uint32_t, etkdg_molecule_metadata_t const*, std::uint32_t, float*,
                                          float*, std::uint8_t*, std::uint32_t, std::uint32_t, std::uint32_t,
                                          std::uint64_t);

int main() {
    // find a molecule close to 100 atoms (with H) from known-good SMILES
    std::vector<std::pair<std::string, std::string>> candidates = {
        {"testosterone", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"},
        {"dexamethasone", "CC1CC2C3CCC4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)CO)O)C)O)F)C"},
        {"erythromycin",
         "CCC1C(C(C(C(=O)C(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)O)(C)O"},
        {"celecoxib", "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"},
        {"diazepam", "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3"},
    };

    std::printf("finding a molecule with ~100 atoms (with H)...\n");
    std::string chosen_smiles;
    std::string chosen_name;
    std::uint32_t chosen_atoms = 0;

    for (auto const& [name, smi] : candidates) {
        auto mol = mol_from_smiles(smi);
        if (!mol) {
            std::printf("  %s: FAILED to parse\n", name.c_str());
            continue;
        }
        std::uint32_t n = mol->getNumAtoms();
        std::printf("  %s: %u atoms (with H)\n", name.c_str(), n);
        if (n <= 128 && n > chosen_atoms) {
            chosen_atoms = n;
            chosen_smiles = smi;
            chosen_name = name;
        }
    }

    if (chosen_atoms == 0) {
        std::printf("no suitable molecule found\n");
        return 1;
    }
    std::printf("\nusing: %s (%u atoms)\n\n", chosen_name.c_str(), chosen_atoms);

    constexpr std::uint32_t num_conformers = 3;
    constexpr std::uint32_t max_stereo_attempts = 10;

    std::printf("%-10s  %9s  %10s  %11s  %12s\n", "batch", "kernel_ms", "ms/mol", "ms/conf", "conf/sec");
    std::printf("%-10s  %9s  %10s  %11s  %12s\n", "-----", "---------", "------", "-------", "--------");

    for (int batch_n : {1, 4, 16, 50, 100, 200, 400, 750, 1500, 3000}) {
        // replicate the same molecule batch_n times
        std::vector<std::string> smiles(batch_n, chosen_smiles);

        auto batch = prepare_molecule_batch(smiles);
        if (batch.num_molecules == 0) {
            std::printf("  prep failed for batch %d\n", batch_n);
            continue;
        }

        std::uint32_t total_conf = batch.num_molecules * num_conformers;
        std::uint32_t max_coords = total_conf * batch.max_atoms_per_molecule * 3;
        std::vector<float> coords(max_coords), energies(total_conf);
        std::vector<std::uint8_t> status(total_conf);

        float best_ms = 1e9f;
        for (int r = 0; r < 3; ++r) {
            float ms = gpu_etkdg_pipeline_timed(
                batch.bounds_matrices.data(), (std::uint32_t)batch.bounds_matrices.size(),
                batch.chiral_centers.empty() ? nullptr : batch.chiral_centers.data(),
                (std::uint32_t)batch.chiral_centers.size(),
                batch.etk_torsions.empty() ? nullptr : batch.etk_torsions.data(),
                (std::uint32_t)batch.etk_torsions.size(),
                batch.distance_constraints.empty() ? nullptr : batch.distance_constraints.data(),
                (std::uint32_t)batch.distance_constraints.size(), batch.molecule_metadata.data(), batch.num_molecules,
                coords.data(), energies.data(), status.data(), num_conformers, max_stereo_attempts,
                batch.max_atoms_per_molecule, 42 + r);
            if (ms < best_ms)
                best_ms = ms;
        }

        double ms_per_mol = best_ms / batch.num_molecules;
        double ms_per_conf = best_ms / total_conf;
        double conf_per_sec = total_conf / (best_ms / 1000.0);

        std::printf("%-10d  %9.1f  %10.3f  %11.3f  %12.1f\n", batch_n, best_ms, ms_per_mol, ms_per_conf, conf_per_sec);
        std::fflush(stdout);
    }
    return 0;
}
