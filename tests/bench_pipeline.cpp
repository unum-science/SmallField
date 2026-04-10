/**
 * @file bench_pipeline.cpp
 * @brief Throughput benchmark: SmallField GPU ETKDG pipeline vs RDKit CPU EmbedMultipleConfs.
 *
 * Measures:
 *   1. CPU preprocessing time (SMILES → bounds matrix + torsions + chirals)
 *   2. GPU kernel time (stages 2-8, via CUDA events)
 *   3. End-to-end time (preprocessing + upload + kernel + download)
 *   4. RDKit EmbedMultipleConfs baseline
 *
 * Reports conformers/second and speedup ratio.
 */

#include "etkdg_preprocess.hpp"
#include "test_helpers.hpp"

#include <GraphMol/DistGeomHelpers/Embedder.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

using namespace smallfield;
using namespace smallfield::test;
using clock_type = std::chrono::high_resolution_clock;

extern "C" float gpu_etkdg_pipeline_timed(
    float const* host_bounds, std::uint32_t bounds_total_floats, chiral_center_descriptor_t const* host_chirals,
    std::uint32_t chirals_total, etk_torsion_descriptor_t const* host_torsions, std::uint32_t torsions_total,
    distance_constraint_t const* host_constraints, std::uint32_t constraints_total,
    etkdg_molecule_metadata_t const* host_metadata, std::uint32_t num_molecules, float* host_coords_out,
    float* host_energies_out, std::uint8_t* host_status_out, std::uint32_t num_conformers,
    std::uint32_t max_stereo_attempts, std::uint32_t max_atoms_per_molecule, std::uint64_t seed);

static double mean(std::vector<double> const& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

static double stddev(std::vector<double> const& values) {
    double const m = mean(values);
    double sum_sq = 0.0;
    for (auto v : values)
        sum_sq += (v - m) * (v - m);
    return std::sqrt(sum_sq / values.size());
}

int main() {
    std::printf("bench_pipeline: SmallField GPU ETKDG vs RDKit CPU\n\n");

    auto base_smiles = standard_test_smiles();
    constexpr std::uint32_t num_conformers = 3;
    constexpr std::uint32_t max_stereo_attempts = 10;
    constexpr int num_benchmark_runs = 2;

    // test scaling from small batches to GPU-saturating sizes
    // 188 SMs × 4 blocks/SM = 752 concurrent warps for small molecules
    // with 3 conformers each, need ~750+ to keep all SMs busy for the full run
    std::vector<std::uint32_t> batch_sizes = {100, 750, 2000, 5000, 10000};
    for (auto target_size : batch_sizes) {
        std::vector<std::string> test_smiles;
        test_smiles.reserve(target_size);
        while (test_smiles.size() < target_size) {
            for (auto const& s : base_smiles) {
                test_smiles.push_back(s);
                if (test_smiles.size() >= target_size)
                    break;
            }
        }

        std::printf("\n========== batch_size = %u ==========\n", target_size);

        // step 1: benchmark CPU preprocessing
        std::printf("step 1: CPU preprocessing (%zu molecules)\n", test_smiles.size());
        std::vector<double> preprocess_times;
        etkdg_molecule_batch_t batch;
        for (int run = 0; run < num_benchmark_runs; ++run) {
            auto t0 = clock_type::now();
            batch = prepare_molecule_batch(test_smiles);
            auto t1 = clock_type::now();
            double const elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            preprocess_times.push_back(elapsed_ms);
        }
        std::printf("  %u molecules, max_atoms=%u\n", batch.num_molecules, batch.max_atoms_per_molecule);
        std::printf("  preprocessing: %.2f +/- %.2f ms\n", mean(preprocess_times), stddev(preprocess_times));

        // step 2: benchmark GPU kernel
        std::printf("\nstep 2: GPU pipeline kernel (%u molecules x %u conformers)\n", batch.num_molecules,
                    num_conformers);

        std::uint32_t const total_conformers = batch.num_molecules * num_conformers;
        std::uint32_t const max_coords = total_conformers * batch.max_atoms_per_molecule * 3;
        std::vector<float> coords_out(max_coords);
        std::vector<float> energies_out(total_conformers);
        std::vector<std::uint8_t> status_out(total_conformers);

        std::vector<double> kernel_times;
        for (int run = 0; run < num_benchmark_runs; ++run) {
            float kernel_ms = gpu_etkdg_pipeline_timed(
                batch.bounds_matrices.data(), static_cast<std::uint32_t>(batch.bounds_matrices.size()),
                batch.chiral_centers.empty() ? nullptr : batch.chiral_centers.data(),
                static_cast<std::uint32_t>(batch.chiral_centers.size()),
                batch.etk_torsions.empty() ? nullptr : batch.etk_torsions.data(),
                static_cast<std::uint32_t>(batch.etk_torsions.size()),
                batch.distance_constraints.empty() ? nullptr : batch.distance_constraints.data(),
                static_cast<std::uint32_t>(batch.distance_constraints.size()), batch.molecule_metadata.data(),
                batch.num_molecules, coords_out.data(), energies_out.data(), status_out.data(), num_conformers,
                max_stereo_attempts, batch.max_atoms_per_molecule, 42 + run);
            kernel_times.push_back(static_cast<double>(kernel_ms));
        }

        std::uint32_t num_successful = 0;
        for (auto s : status_out)
            if (s == 0)
                num_successful++;

        double const kernel_mean_ms = mean(kernel_times);
        double const gpu_conformers_per_sec = total_conformers / (kernel_mean_ms / 1000.0);
        double const gpu_molecules_per_sec = batch.num_molecules / (kernel_mean_ms / 1000.0);

        std::printf("  kernel time:     %.2f +/- %.2f ms\n", kernel_mean_ms, stddev(kernel_times));
        std::printf("  success rate:    %u/%u (%.1f%%)\n", num_successful, total_conformers,
                    100.0 * num_successful / total_conformers);
        std::printf("  conformers/sec:  %.1f (GPU kernel only)\n", gpu_conformers_per_sec);
        std::printf("  molecules/sec:   %.1f (GPU kernel only, %u conf each)\n", gpu_molecules_per_sec, num_conformers);

        // step 3: benchmark end-to-end (preprocessing + kernel)
        std::printf("\nstep 3: end-to-end (preprocessing + GPU)\n");
        std::vector<double> e2e_times;
        for (int run = 0; run < num_benchmark_runs; ++run) {
            auto t0 = clock_type::now();
            auto run_batch = prepare_molecule_batch(test_smiles);
            float kernel_ms = gpu_etkdg_pipeline_timed(
                run_batch.bounds_matrices.data(), static_cast<std::uint32_t>(run_batch.bounds_matrices.size()),
                run_batch.chiral_centers.empty() ? nullptr : run_batch.chiral_centers.data(),
                static_cast<std::uint32_t>(run_batch.chiral_centers.size()),
                run_batch.etk_torsions.empty() ? nullptr : run_batch.etk_torsions.data(),
                static_cast<std::uint32_t>(run_batch.etk_torsions.size()),
                run_batch.distance_constraints.empty() ? nullptr : run_batch.distance_constraints.data(),
                static_cast<std::uint32_t>(run_batch.distance_constraints.size()), run_batch.molecule_metadata.data(),
                run_batch.num_molecules, coords_out.data(), energies_out.data(), status_out.data(), num_conformers,
                max_stereo_attempts, run_batch.max_atoms_per_molecule, 42 + run);
            auto t1 = clock_type::now();
            double const elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            e2e_times.push_back(elapsed_ms);
        }
        double const e2e_mean_ms = mean(e2e_times);
        double const e2e_molecules_per_sec = batch.num_molecules / (e2e_mean_ms / 1000.0);
        std::printf("  end-to-end:      %.2f +/- %.2f ms\n", e2e_mean_ms, stddev(e2e_times));
        std::printf("  molecules/sec:   %.1f (including preprocessing)\n", e2e_molecules_per_sec);

        // step 4: RDKit baseline
        std::printf("\nstep 4: RDKit EmbedMultipleConfs baseline\n");
        std::vector<double> rdkit_times;
        for (int run = 0; run < num_benchmark_runs; ++run) {
            auto t0 = clock_type::now();
            for (auto const& smiles : test_smiles) {
                auto mol = mol_from_smiles(smiles);
                if (!mol)
                    continue;
                RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
                params.randomSeed = 42 + run;
                params.numThreads = 1;
                RDKit::DGeomHelpers::EmbedMultipleConfs(*mol, num_conformers, params);
            }
            auto t1 = clock_type::now();
            double const elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            rdkit_times.push_back(elapsed_ms);
        }
        double const rdkit_mean_ms = mean(rdkit_times);
        double const rdkit_molecules_per_sec = test_smiles.size() / (rdkit_mean_ms / 1000.0);
        std::printf("  RDKit time:      %.2f +/- %.2f ms\n", rdkit_mean_ms, stddev(rdkit_times));
        std::printf("  molecules/sec:   %.1f (single-threaded)\n", rdkit_molecules_per_sec);

        // summary
        double const speedup_kernel = rdkit_mean_ms / kernel_mean_ms;
        double const speedup_e2e = rdkit_mean_ms / e2e_mean_ms;

        std::printf("\n  --- batch_size=%u summary ---\n", target_size);
        std::printf("  GPU kernel speedup vs RDKit:     %.2fx\n", speedup_kernel);
        std::printf("  End-to-end speedup vs RDKit:     %.2fx\n", speedup_e2e);
        std::printf("  GPU kernel:    %.1f conf/sec   (%.1f mol/sec)\n", gpu_conformers_per_sec, gpu_molecules_per_sec);
        std::printf("  RDKit:         %.1f conf/sec   (%.1f mol/sec)\n",
                    test_smiles.size() * num_conformers / (rdkit_mean_ms / 1000.0), rdkit_molecules_per_sec);

    } // end batch_sizes loop

    return 0;
}
