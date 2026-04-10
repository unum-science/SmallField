/**
 * @file bench_comparison.cpp
 * @brief 3-way benchmark: OpenMP RDKit vs SmallField GPU (ETKDG+MMFF).
 *
 * Config A: RDKit ETKDG + MMFF on CPU (OpenMP, 1 core/molecule)
 * Config C: SmallField ETKDG + MMFF on GPU (FIRE optimizer, stages 2-9)
 *
 * Config B (RDKit + nvMolKit) runs separately via tests/bench_nvmolkit.py.
 */

#include "etkdg_preprocess.hpp"
#include "test_helpers.hpp"

#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <omp.h>
#include <vector>

using namespace smallfield;
using namespace smallfield::test;
using clock_type = std::chrono::high_resolution_clock;

extern "C" {
struct gpu_pipeline_result_t {
    float kernel_time_ms;
    std::uint32_t num_succeeded;
};

gpu_pipeline_result_t gpu_etkdg_mmff_pipeline(
    float const*, std::uint32_t, chiral_center_descriptor_t const*, std::uint32_t,
    etk_torsion_descriptor_t const*, std::uint32_t, distance_constraint_t const*, std::uint32_t,
    mmff94_nonbonded_pair_t const*, std::uint32_t, mmff94_bond_t const*, std::uint32_t,
    mmff94_angle_t const*, std::uint32_t, mmff94_torsion_t const*, std::uint32_t,
    mmff94_atom_interaction_t const*, std::uint32_t, mmff94_atom_offsets_t const*, std::uint32_t,
    etkdg_molecule_metadata_t const*, std::uint32_t, float*, float*, std::uint8_t*,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t);
}

int main() {
    std::printf("=== 3-way benchmark: ETKDG + MMFF94 ===\n\n");

    auto base_smiles = standard_test_smiles();
    constexpr std::uint32_t num_conformers = 3;
    constexpr std::uint32_t max_stereo_attempts = 10;

    int num_threads = omp_get_max_threads();
    std::printf("CPU: %d OpenMP threads\n\n", num_threads);

    for (int batch_n : {100, 500, 1000, 2000}) {
        std::vector<std::string> smiles;
        smiles.reserve(batch_n);
        while (static_cast<int>(smiles.size()) < batch_n)
            for (auto const& s : base_smiles) {
                smiles.push_back(s);
                if (static_cast<int>(smiles.size()) >= batch_n) break;
            }

        std::printf("--- batch_size = %d, %u conformers each ---\n", batch_n, num_conformers);

        // config A: OpenMP RDKit (ETKDG + MMFF on CPU)
        {
            auto t0 = clock_type::now();
            std::uint32_t succeeded = 0;
            #pragma omp parallel for schedule(dynamic, 4) reduction(+:succeeded)
            for (int i = 0; i < batch_n; ++i) {
                auto mol = mol_from_smiles(smiles[i]);
                if (!mol) continue;
                RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
                params.randomSeed = 42 + i;
                params.numThreads = 1;
                auto conf_ids = RDKit::DGeomHelpers::EmbedMultipleConfs(*mol, num_conformers, params);
                if (!conf_ids.empty()) {
                    std::vector<std::pair<int, double>> mmff_results;
                    RDKit::MMFF::MMFFOptimizeMoleculeConfs(*mol, mmff_results);
                    succeeded += conf_ids.size();
                }
            }
            auto t1 = clock_type::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double mol_per_sec = batch_n / (ms / 1000.0);
            std::printf("  Config A (OpenMP RDKit %d cores):  %7.0f ms  %7.1f mol/s  %u conf ok\n",
                        num_threads, ms, mol_per_sec, succeeded);
        }

        // config C: SmallField GPU (ETKDG + MMFF)
        {
            auto t0 = clock_type::now();
            auto batch = prepare_molecule_batch(smiles);
            auto t1_preproc = clock_type::now();

            std::uint32_t const total_conf = batch.num_molecules * num_conformers;
            std::uint32_t const max_coords = total_conf * batch.max_atoms_per_molecule * 3;
            std::vector<float> coords(max_coords), energies(total_conf);
            std::vector<std::uint8_t> status(total_conf);

            auto result = gpu_etkdg_mmff_pipeline(
                batch.bounds_matrices.data(), static_cast<std::uint32_t>(batch.bounds_matrices.size()),
                batch.chiral_centers.empty() ? nullptr : batch.chiral_centers.data(),
                static_cast<std::uint32_t>(batch.chiral_centers.size()),
                batch.etk_torsions.empty() ? nullptr : batch.etk_torsions.data(),
                static_cast<std::uint32_t>(batch.etk_torsions.size()),
                batch.distance_constraints.empty() ? nullptr : batch.distance_constraints.data(),
                static_cast<std::uint32_t>(batch.distance_constraints.size()),
                batch.mmff_nonbonded.empty() ? nullptr : batch.mmff_nonbonded.data(),
                static_cast<std::uint32_t>(batch.mmff_nonbonded.size()),
                batch.mmff_bonds.empty() ? nullptr : batch.mmff_bonds.data(),
                static_cast<std::uint32_t>(batch.mmff_bonds.size()),
                batch.mmff_angles.empty() ? nullptr : batch.mmff_angles.data(),
                static_cast<std::uint32_t>(batch.mmff_angles.size()),
                batch.mmff_torsions.empty() ? nullptr : batch.mmff_torsions.data(),
                static_cast<std::uint32_t>(batch.mmff_torsions.size()),
                batch.mmff_interactions.empty() ? nullptr : batch.mmff_interactions.data(),
                static_cast<std::uint32_t>(batch.mmff_interactions.size()),
                batch.mmff_atom_offsets.empty() ? nullptr : batch.mmff_atom_offsets.data(),
                static_cast<std::uint32_t>(batch.mmff_atom_offsets.size()),
                batch.molecule_metadata.data(), batch.num_molecules,
                coords.data(), energies.data(), status.data(),
                num_conformers, max_stereo_attempts, batch.max_atoms_per_molecule, 42);

            auto t1 = clock_type::now();
            double preproc_ms = std::chrono::duration<double, std::milli>(t1_preproc - t0).count();
            double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double mol_per_sec = batch.num_molecules / (total_ms / 1000.0);

            // cross-validate: check MMFF94 energy via RDKit
            std::uint32_t finite_count = 0, reasonable_count = 0;
            for (std::uint32_t m = 0; m < std::min(batch.num_molecules, 20u); ++m) {
                auto mol = mol_from_smiles(smiles[m]);
                if (!mol) continue;
                bool embedded = embed_molecule(*mol);
                if (!embedded) continue;
                std::uint32_t const na = batch.molecule_metadata[m].num_atoms;
                std::uint32_t const max_n = batch.max_atoms_per_molecule;
                for (std::uint32_t c = 0; c < num_conformers; ++c) {
                    std::uint32_t const slot = m * num_conformers + c;
                    if (status[slot] != 0) continue;
                    std::vector<float> interleaved(na * 3);
                    std::uint32_t const off = slot * max_n * 3;
                    for (std::uint32_t i = 0; i < na; ++i) {
                        interleaved[i * 3 + 0] = coords[off + 0 * max_n + i];
                        interleaved[i * 3 + 1] = coords[off + 1 * max_n + i];
                        interleaved[i * 3 + 2] = coords[off + 2 * max_n + i];
                    }
                    set_coords_interleaved(*mol, interleaved.data());
                    double mmff = rdkit_mmff94_energy(*mol);
                    if (std::isfinite(mmff)) {
                        finite_count++;
                        if (mmff < 1000.0) reasonable_count++;
                    }
                }
            }

            std::printf("  Config C (SmallField GPU):         %7.0f ms  %7.1f mol/s  %u conf ok",
                        total_ms, mol_per_sec, result.num_succeeded);
            std::printf("  (preproc %.0fms, kernel %.1fms)\n", preproc_ms, result.kernel_time_ms);
            std::printf("           quality (first 20 mols): %u/%u finite MMFF, %u reasonable\n",
                        finite_count, std::min(batch.num_molecules, 20u) * num_conformers, reasonable_count);
        }

        std::printf("\n");
    }

    return 0;
}
