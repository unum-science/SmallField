/**
 *  @file bench_chembl.cpp
 *  @brief Unified benchmark: RDKit CPU vs SmallField GPU on ChEMBL dataset.
 *
 *  Measures:
 *   1. RDKit CPU: OpenMP parallel EmbedMultipleConfs + MMFFOptimize
 *   2. SmallField GPU: single-GPU and multi-GPU conformer generation
 *   3. Quality cross-validation: SmallField coords evaluated by RDKit MMFF94
 *   4. Scaling analysis across batch sizes
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <omp.h>

#include "smallfield/api/smallfield.hpp"
#include "test_helpers.hpp"

namespace sf = smallfield;
namespace sfapi = smallfield::api;
namespace sftest = smallfield::tests;
using clock_type = std::chrono::high_resolution_clock;

// Load SMILES from a file, one per line.
static std::vector<std::string> load_smiles(std::string const &path, std::uint32_t max_count = 0) {
    std::vector<std::string> result;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        // Take the first whitespace-delimited token because the line may also contain a molecule name.
        auto end = line.find_first_of(" \t");
        if (end != std::string::npos) line = line.substr(0, end);
        result.push_back(line);
        if (max_count > 0 && result.size() >= max_count) break;
    }
    return result;
}

static double percentile(std::vector<double> &sorted_values, double p) {
    if (sorted_values.empty()) return 0.0;
    double idx = p / 100.0 * static_cast<double>(sorted_values.size() - 1);
    std::size_t lo = static_cast<std::size_t>(idx);
    std::size_t hi = std::min(lo + 1, sorted_values.size() - 1);
    double frac = idx - static_cast<double>(lo);
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac;
}

static int report_cuda_errors(char const *context, std::uint64_t cuda_errors) {
    std::fprintf(
        stderr,
        "\nFAIL: %llu CUDA errors reported by the library during %s; " "throughput numbers above are unreliable. "
                                                                       "Clean-rebuild and retry.\n",
        static_cast<unsigned long long>(cuda_errors), context);
    return 2;
}

static void bench_rdkit_cpu(std::vector<std::string> const &smiles, std::uint32_t num_conformers,
                            std::uint32_t max_stereo_attempts) {
    int num_threads = omp_get_max_threads();
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    std::printf("\n=== System A: RDKit CPU (%d threads) ===\n", num_threads);

    // Pre-parse all molecules outside the timed region.
    std::printf("  Pre-parsing %u molecules...\n", num_smiles);
    std::vector<std::unique_ptr<RDKit::RWMol>> molecules(num_smiles);
    std::uint32_t num_parsed = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : num_parsed)
    for (std::uint32_t molecule_index = 0; molecule_index < num_smiles; ++molecule_index) {
        try {
            molecules[molecule_index] = sftest::mol_from_smiles(smiles[molecule_index]);
            if (molecules[molecule_index]) num_parsed++;
        }
        catch (...) {
            molecules[molecule_index] = nullptr;
        }
    }
    std::printf("  Parsed: %u / %u\n", num_parsed, num_smiles);

    // Time EmbedMultipleConfs followed by MMFFOptimize.
    std::printf("  Running ETKDG + MMFF94 (%u conformers each)...\n", num_conformers);
    auto const start_time = clock_type::now();
    std::uint32_t num_successful_conformers = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : num_successful_conformers)
    for (std::uint32_t molecule_index = 0; molecule_index < num_smiles; ++molecule_index) {
        if (!molecules[molecule_index]) continue;
        try {
            RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
            params.randomSeed = 42 + static_cast<int>(molecule_index);
            params.numThreads = 1;
            params.maxIterations = static_cast<int>(max_stereo_attempts);
            auto const conformer_ids = RDKit::DGeomHelpers::EmbedMultipleConfs(*molecules[molecule_index],
                                                                               num_conformers, params);
            if (!conformer_ids.empty()) {
                std::vector<std::pair<int, double>> mmff_results;
                RDKit::MMFF::MMFFOptimizeMoleculeConfs(*molecules[molecule_index], mmff_results);
                num_successful_conformers += static_cast<std::uint32_t>(conformer_ids.size());
            }
        }
        catch (...) {
            // Skip molecules that fail RDKit processing.
        }
    }
    auto const end_time = clock_type::now();
    double const elapsed_ms_value = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double const mol_per_sec = num_parsed / (elapsed_ms_value / 1000.0);
    double const conf_per_sec = num_successful_conformers / (elapsed_ms_value / 1000.0);
    std::printf("  Time:      %.0f ms\n", elapsed_ms_value);
    std::printf("  Mol/s:     %.1f\n", mol_per_sec);
    std::printf("  Conf/s:    %.1f\n", conf_per_sec);
    std::printf("  Succeeded: %u conformers\n", num_successful_conformers);
}

static void bench_smallfield_gpu(std::vector<std::string> const &smiles, std::uint32_t num_conformers,
                                 std::uint32_t max_stereo_attempts, std::vector<__half> &coords_out,
                                 std::vector<float> &energies_out, std::vector<sfapi::conformer_status_t> &status_out) {
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    std::uint32_t const max_atoms = 128;
    std::printf("\n=== System B: SmallField GPU ===\n");

    // Build string_view span for the library.
    std::vector<std::string_view> smiles_views(smiles.begin(), smiles.end());

    sfapi::config_t generation_config;
    generation_config.num_conformers = num_conformers;
    generation_config.max_stereo_attempts = max_stereo_attempts;

    // Allocate output buffers.
    std::uint32_t const total_conformers = num_smiles * num_conformers;
    coords_out.resize(total_conformers * max_atoms * 3);
    energies_out.resize(total_conformers);
    status_out.resize(total_conformers);

    // Single GPU.
    int const device_id = 0;
    sfapi::context_t gpu_context(std::span<int const>(&device_id, 1));

    auto const start_time = clock_type::now();
    gpu_context.generate_conformers(smiles_views, generation_config, max_atoms, coords_out, energies_out, status_out);
    auto const end_time = clock_type::now();

    double const elapsed_ms_value = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Count successes.
    std::uint32_t num_successful_conformers = 0;
    for (auto status_code : status_out)
        if (status_code == sfapi::conformer_status_t::success) num_successful_conformers++;

    double const mol_per_sec = num_smiles / (elapsed_ms_value / 1000.0);
    double const conf_per_sec = num_successful_conformers / (elapsed_ms_value / 1000.0);
    std::printf("  Time:      %.0f ms\n", elapsed_ms_value);
    std::printf("  Mol/s:     %.1f\n", mol_per_sec);
    std::printf("  Conf/s:    %.1f\n", conf_per_sec);
    std::printf("  Succeeded: %u / %u conformers\n", num_successful_conformers, total_conformers);
}

static void bench_smallfield_multi_gpu(std::vector<std::string> const &smiles, std::uint32_t num_conformers,
                                       std::uint32_t max_stereo_attempts) {
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    std::uint32_t const max_atoms = 128;
    std::printf("\n=== System B: SmallField multi-GPU (all GPUs) ===\n");

    // Build string_view span for the library.
    std::vector<std::string_view> smiles_views(smiles.begin(), smiles.end());

    sfapi::config_t generation_config;
    generation_config.num_conformers = num_conformers;
    generation_config.max_stereo_attempts = max_stereo_attempts;

    // Allocate output buffers.
    std::uint32_t const total_conformers = num_smiles * num_conformers;
    std::vector<__half> coords(total_conformers * max_atoms * 3);
    std::vector<float> energies(total_conformers);
    std::vector<sfapi::conformer_status_t> status(total_conformers);

    // All GPUs (empty device span).
    sfapi::context_t gpu_context;
    // Opt in to scaffold-based SubstructMatch caching.
    // Amortises RDKit fallback cost across molecules sharing the same Murcko
    // scaffold; safe no-op if disabled or for acyclic molecules.
    gpu_context.enable_scaffold_cache();

    // Allow 1-GPU runs via --multi-gpu-only (used by scaling sweeps that
    // control GPU count via CUDA_VISIBLE_DEVICES). The "multi" in the
    // flag name is historical; the path still runs correctly on 1 GPU.
    if (gpu_context.device_ids.empty()) {
        std::printf("  Skipped (no GPU available)\n");
        return;
    }
    std::printf("  Using %zu GPUs\n", gpu_context.device_ids.size());

    auto const start_time = clock_type::now();
    gpu_context.generate_conformers(smiles_views, generation_config, max_atoms, coords, energies, status);
    auto const end_time = clock_type::now();

    double const elapsed_ms_value = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Count successes.
    std::uint32_t num_successful_conformers = 0;
    for (auto status_code : status)
        if (status_code == sfapi::conformer_status_t::success) num_successful_conformers++;

    double const mol_per_sec = num_smiles / (elapsed_ms_value / 1000.0);
    double const conf_per_sec = num_successful_conformers / (elapsed_ms_value / 1000.0);
    std::printf("  Time:      %.0f ms\n", elapsed_ms_value);
    std::printf("  Mol/s:     %.1f\n", mol_per_sec);
    std::printf("  Conf/s:    %.1f\n", conf_per_sec);
    std::printf("  Succeeded: %u / %u conformers\n", num_successful_conformers, total_conformers);
}

static void cross_validate(std::vector<std::string> const &smiles, std::vector<__half> const &coords,
                           std::vector<float> const &energies, std::vector<sfapi::conformer_status_t> const &status,
                           std::uint32_t num_conformers, std::uint32_t max_atoms, std::uint32_t max_samples) {
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    std::printf("\n=== Quality cross-validation ===\n");

    // Build the list of succeeded conformers.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> successful_conformers;
    for (std::uint32_t molecule_index = 0; molecule_index < num_smiles; ++molecule_index) {
        for (std::uint32_t conformer_index = 0; conformer_index < num_conformers; ++conformer_index) {
            std::uint32_t const flat_index = molecule_index * num_conformers + conformer_index;
            if (status[flat_index] == sfapi::conformer_status_t::success)
                successful_conformers.push_back({molecule_index, conformer_index});
        }
    }

    // Select a random sample.
    std::mt19937 rng(42);
    if (successful_conformers.size() > max_samples) {
        std::shuffle(successful_conformers.begin(), successful_conformers.end(), rng);
        successful_conformers.resize(max_samples);
    }
    std::printf("  Evaluating %zu conformers via RDKit MMFF94...\n", successful_conformers.size());

    // Run cross-validation in parallel.
    std::vector<double> rdkit_reference_energies(successful_conformers.size(),
                                                 std::numeric_limits<double>::quiet_NaN());
    std::vector<double> smallfield_energies(successful_conformers.size());
    std::vector<std::uint32_t> atom_counts(successful_conformers.size());

#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t sample_index = 0; sample_index < successful_conformers.size(); ++sample_index) {
        auto [molecule_index, conformer_index] = successful_conformers[sample_index];
        try {
            auto rdkit_molecule = sftest::mol_from_smiles(smiles[molecule_index]);
            if (!rdkit_molecule) continue;
            bool did_embed = sftest::embed_molecule(*rdkit_molecule);
            if (!did_embed) continue;

            std::uint32_t const num_atoms = rdkit_molecule->getNumAtoms();
            std::uint32_t const flat_index = molecule_index * num_conformers + conformer_index;
            std::uint32_t const coordinate_offset = flat_index * max_atoms * 3;

            // Convert SoA half-precision coordinates to interleaved float layout.
            std::vector<float> interleaved_coordinates(num_atoms * 3);
            for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
                interleaved_coordinates[atom_index * 3 + 0] = __half2float(
                    coords[coordinate_offset + 0 * max_atoms + atom_index]);
                interleaved_coordinates[atom_index * 3 + 1] = __half2float(
                    coords[coordinate_offset + 1 * max_atoms + atom_index]);
                interleaved_coordinates[atom_index * 3 + 2] = __half2float(
                    coords[coordinate_offset + 2 * max_atoms + atom_index]);
            }
            sftest::set_coords_interleaved(*rdkit_molecule, interleaved_coordinates.data());
            rdkit_reference_energies[sample_index] = sftest::rdkit_mmff94_energy(*rdkit_molecule);
            smallfield_energies[sample_index] = static_cast<double>(energies[flat_index]);
            atom_counts[sample_index] = num_atoms;
        }
        catch (...) {
            // Skip failures.
        }
    }

    // Analyze the resulting energy distributions.
    std::uint32_t num_finite_energies = 0;
    std::vector<double> finite_rdkit_energies;
    std::vector<double> finite_smallfield_energies;
    for (std::size_t sample_index = 0; sample_index < rdkit_reference_energies.size(); ++sample_index) {
        if (std::isfinite(rdkit_reference_energies[sample_index])) {
            num_finite_energies++;
            finite_rdkit_energies.push_back(rdkit_reference_energies[sample_index]);
            finite_smallfield_energies.push_back(smallfield_energies[sample_index]);
        }
    }

    std::printf("  Finite:    %u / %zu (%.1f%%)\n", num_finite_energies, successful_conformers.size(),
                100.0 * num_finite_energies / successful_conformers.size());

    if (!finite_rdkit_energies.empty()) {
        std::sort(finite_rdkit_energies.begin(), finite_rdkit_energies.end());
        std::sort(finite_smallfield_energies.begin(), finite_smallfield_energies.end());

        std::printf("  RDKit MMFF94 energy (kcal/mol):\n");
        std::printf("    P5=%.1f  P25=%.1f  P50=%.1f  P75=%.1f  P95=%.1f\n", percentile(finite_rdkit_energies, 5),
                    percentile(finite_rdkit_energies, 25), percentile(finite_rdkit_energies, 50),
                    percentile(finite_rdkit_energies, 75), percentile(finite_rdkit_energies, 95));
        std::printf("  SmallField energy (kcal/mol):\n");
        std::printf("    P5=%.1f  P25=%.1f  P50=%.1f  P75=%.1f  P95=%.1f\n", percentile(finite_smallfield_energies, 5),
                    percentile(finite_smallfield_energies, 25), percentile(finite_smallfield_energies, 50),
                    percentile(finite_smallfield_energies, 75), percentile(finite_smallfield_energies, 95));

        // Print the per-atom-count breakdown.
        std::printf("  Per-atom-count breakdown:\n");
        std::printf("    %6s  %6s  %8s  %8s\n", "atoms", "count", "rdkit_P50", "sf_P50");
        std::vector<std::uint32_t> atom_count_bins = {20, 40, 60, 80, 100, 128};
        std::vector<double> heavy_atom_rdkit_energies;
        for (std::size_t bin_index = 0; bin_index < atom_count_bins.size(); ++bin_index) {
            std::uint32_t lower_bound = (bin_index == 0) ? 0 : atom_count_bins[bin_index - 1];
            std::uint32_t upper_bound = atom_count_bins[bin_index];
            std::vector<double> bin_rdkit_energies, bin_smallfield_energies;
            for (std::size_t sample_index = 0; sample_index < successful_conformers.size(); ++sample_index) {
                if (std::isfinite(rdkit_reference_energies[sample_index]) && atom_counts[sample_index] > lower_bound &&
                    atom_counts[sample_index] <= upper_bound) {
                    bin_rdkit_energies.push_back(rdkit_reference_energies[sample_index]);
                    bin_smallfield_energies.push_back(smallfield_energies[sample_index]);
                }
            }
            if (!bin_rdkit_energies.empty()) {
                std::sort(bin_rdkit_energies.begin(), bin_rdkit_energies.end());
                std::sort(bin_smallfield_energies.begin(), bin_smallfield_energies.end());
                std::printf("    %3u-%-3u  %5zu  %8.1f  %8.1f\n", lower_bound, upper_bound, bin_rdkit_energies.size(),
                            percentile(bin_rdkit_energies, 50), percentile(bin_smallfield_energies, 50));
                if (lower_bound == 100 && upper_bound == 128) heavy_atom_rdkit_energies = bin_rdkit_energies;
            }
        }

        if (!heavy_atom_rdkit_energies.empty()) {
            double const heavy_p50 = percentile(heavy_atom_rdkit_energies, 50);
            double const heavy_p95 = percentile(heavy_atom_rdkit_energies, 95);
            bool const passes_gate = heavy_p50 < 180.0 && heavy_p95 < 300.0;
            std::printf("  Heavy-bin gate (100-128 atoms): P50=%.1f  P95=%.1f  [%s]\n", heavy_p50, heavy_p95,
                        passes_gate ? "PASS" : "FAIL");
        }
    }
}

int main(int argc, char **argv) {
    std::printf("bench_chembl: SmallField GPU vs RDKit CPU\n");
    std::printf("==========================================\n");

    // Set defaults.
    std::string smi_path = "/tmp/chembl_valid.smi";
    std::uint32_t num_conformers = 3;
    std::uint32_t max_stereo_attempts = 10;
    std::uint32_t max_atoms = 128;
    std::uint32_t quality_samples = 10000;

    // Parse the command line.
    bool multi_gpu_only = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--smi" && i + 1 < argc) smi_path = argv[++i];
        else if (arg == "--conformers" && i + 1 < argc)
            num_conformers = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        else if (arg == "--quality-samples" && i + 1 < argc)
            quality_samples = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        else if (arg == "--multi-gpu-only") multi_gpu_only = true;
    }

    std::printf("Dataset:     %s\n", smi_path.c_str());
    std::printf("Conformers:  %u per molecule\n", num_conformers);
    std::printf("CPU threads: %d\n", omp_get_max_threads());

    auto all_smiles = load_smiles(smi_path);
    std::printf("Loaded:      %zu SMILES\n", all_smiles.size());
    if (all_smiles.empty()) {
        std::fprintf(stderr, "No SMILES loaded from %s\n", smi_path.c_str());
        return 1;
    }

    sfapi::reset_cuda_error_count();

    if (multi_gpu_only) {
        bench_smallfield_multi_gpu(all_smiles, num_conformers, max_stereo_attempts);
        std::uint64_t const cuda_errors = sfapi::cuda_error_count();
        if (cuda_errors > 0) return report_cuda_errors("generate_conformers", cuda_errors);
        std::printf("\nDone.\n");
        return 0;
    }

    // Run the scaling analysis at multiple batch sizes.
    std::vector<std::uint32_t> batch_sizes = {100, 500, 1000, 2000, 5000};
    if (all_smiles.size() > 5000) batch_sizes.push_back(static_cast<std::uint32_t>(all_smiles.size()));

    for (auto batch_size : batch_sizes) {
        if (batch_size > all_smiles.size()) continue;

        std::vector<std::string> smiles(all_smiles.begin(), all_smiles.begin() + batch_size);
        std::printf("\n\n############## batch_size = %u ##############\n", batch_size);

        // Run system A: RDKit CPU.
        bench_rdkit_cpu(smiles, num_conformers, max_stereo_attempts);

        // Run system B: SmallField single GPU.
        std::vector<__half> coords;
        std::vector<float> energies;
        std::vector<sfapi::conformer_status_t> status;
        bench_smallfield_gpu(smiles, num_conformers, max_stereo_attempts, coords, energies, status);

        // Multi-GPU scaling is measured through a dedicated fresh process via
        // --multi-gpu-only. Keeping the mixed CPU + 1-GPU default path in a
        // single process avoids state bleed between the 1-GPU and 8-GPU runs
        // on the largest batches.
        if (batch_size == batch_sizes.back()) {
            std::printf("\n=== System B: SmallField multi-GPU (all GPUs) ===\n");
            std::printf("  Skipped in mixed mode. Re-run with --multi-gpu-only for isolated multi-GPU numbers.\n");
        }

        // Run quality cross-validation only on the full dataset.
        if (batch_size == static_cast<std::uint32_t>(all_smiles.size()) || batch_size == batch_sizes.back()) {
            cross_validate(smiles, coords, energies, status, num_conformers, max_atoms, quality_samples);
        }
    }

    std::uint64_t const cuda_errors = sfapi::cuda_error_count();
    if (cuda_errors > 0) return report_cuda_errors("one of the GPU runs", cuda_errors);
    std::printf("\nDone.\n");
    return 0;
}
