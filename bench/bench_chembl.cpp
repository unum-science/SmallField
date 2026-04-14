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

#include "smallfield.hpp"
#include "test_helpers.hpp"

using namespace smallfield;
using namespace smallfield::test;
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

static void bench_rdkit_cpu(std::vector<std::string> const &smiles, std::uint32_t num_conformers,
                            std::uint32_t max_stereo_attempts) {
    int num_threads = omp_get_max_threads();
    std::uint32_t const n = static_cast<std::uint32_t>(smiles.size());
    std::printf("\n=== System A: RDKit CPU (%d threads) ===\n", num_threads);

    // Pre-parse all molecules outside the timed region.
    std::printf("  Pre-parsing %u molecules...\n", n);
    std::vector<std::unique_ptr<RDKit::RWMol>> mols(n);
    std::uint32_t parse_ok = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : parse_ok)
    for (std::uint32_t i = 0; i < n; ++i) {
        try {
            mols[i] = mol_from_smiles(smiles[i]);
            if (mols[i]) parse_ok++;
        }
        catch (...) {
            mols[i] = nullptr;
        }
    }
    std::printf("  Parsed: %u / %u\n", parse_ok, n);

    // Time EmbedMultipleConfs followed by MMFFOptimize.
    std::printf("  Running ETKDG + MMFF94 (%u conformers each)...\n", num_conformers);
    auto t0 = clock_type::now();
    std::uint32_t total_conf_ok = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : total_conf_ok)
    for (std::uint32_t i = 0; i < n; ++i) {
        if (!mols[i]) continue;
        try {
            RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
            params.randomSeed = 42 + static_cast<int>(i);
            params.numThreads = 1;
            params.maxIterations = static_cast<int>(max_stereo_attempts);
            auto conf_ids = RDKit::DGeomHelpers::EmbedMultipleConfs(*mols[i], num_conformers, params);
            if (!conf_ids.empty()) {
                std::vector<std::pair<int, double>> mmff_results;
                RDKit::MMFF::MMFFOptimizeMoleculeConfs(*mols[i], mmff_results);
                total_conf_ok += static_cast<std::uint32_t>(conf_ids.size());
            }
        }
        catch (...) {
            // Skip molecules that fail RDKit processing.
        }
    }
    auto t1 = clock_type::now();
    double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double const mol_per_sec = parse_ok / (ms / 1000.0);
    double const conf_per_sec = total_conf_ok / (ms / 1000.0);
    std::printf("  Time:      %.0f ms\n", ms);
    std::printf("  Mol/s:     %.1f\n", mol_per_sec);
    std::printf("  Conf/s:    %.1f\n", conf_per_sec);
    std::printf("  Succeeded: %u conformers\n", total_conf_ok);
}

static void bench_smallfield_gpu(std::vector<std::string> const &smiles, std::uint32_t num_conformers,
                                 std::uint32_t max_stereo_attempts, std::vector<__half> &coords_out,
                                 std::vector<float> &energies_out, std::vector<conformer_status_t> &status_out) {
    std::uint32_t const n = static_cast<std::uint32_t>(smiles.size());
    std::uint32_t const max_atoms = 128;
    std::printf("\n=== System B: SmallField GPU ===\n");

    // Build string_view span for the library.
    std::vector<std::string_view> smiles_views(smiles.begin(), smiles.end());

    config_t config;
    config.num_conformers = num_conformers;
    config.max_stereo_attempts = max_stereo_attempts;

    // Allocate output buffers.
    std::uint32_t const total_conf = n * num_conformers;
    coords_out.resize(total_conf * max_atoms * 3);
    energies_out.resize(total_conf);
    status_out.resize(total_conf);

    // Single GPU.
    int const device = 0;
    context_t ctx(std::span<int const>(&device, 1));

    auto t0 = clock_type::now();
    ctx.generate_conformers(smiles_views, config, max_atoms, coords_out, energies_out, status_out);
    auto t1 = clock_type::now();

    double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Count successes.
    std::uint32_t num_succeeded = 0;
    for (auto s : status_out)
        if (s == conformer_status_t::success) num_succeeded++;

    double const mol_per_sec = n / (ms / 1000.0);
    double const conf_per_sec = num_succeeded / (ms / 1000.0);
    std::printf("  Time:      %.0f ms\n", ms);
    std::printf("  Mol/s:     %.1f\n", mol_per_sec);
    std::printf("  Conf/s:    %.1f\n", conf_per_sec);
    std::printf("  Succeeded: %u / %u conformers\n", num_succeeded, total_conf);
}

static void bench_smallfield_multi_gpu(std::vector<std::string> const &smiles, std::uint32_t num_conformers,
                                       std::uint32_t max_stereo_attempts) {
    std::uint32_t const n = static_cast<std::uint32_t>(smiles.size());
    std::uint32_t const max_atoms = 128;
    std::printf("\n=== System B: SmallField multi-GPU (all GPUs) ===\n");

    // Build string_view span for the library.
    std::vector<std::string_view> smiles_views(smiles.begin(), smiles.end());

    config_t config;
    config.num_conformers = num_conformers;
    config.max_stereo_attempts = max_stereo_attempts;

    // Allocate output buffers.
    std::uint32_t const total_conf = n * num_conformers;
    std::vector<__half> coords(total_conf * max_atoms * 3);
    std::vector<float> energies(total_conf);
    std::vector<conformer_status_t> status(total_conf);

    // All GPUs (empty device span).
    context_t ctx;
    // Workstream SC-2 — opt in to scaffold-based SubstructMatch caching.
    // Amortises RDKit fallback cost across molecules sharing the same Murcko
    // scaffold; safe no-op if disabled or for acyclic molecules.
    ctx.enable_scaffold_cache();

    // Allow 1-GPU runs via --multi-gpu-only (used by scaling sweeps that
    // control GPU count via CUDA_VISIBLE_DEVICES).  The "multi" in the
    // flag name is historical; the path still runs correctly on 1 GPU.
    if (ctx.device_ids.empty()) {
        std::printf("  Skipped (no GPU available)\n");
        return;
    }
    std::printf("  Using %zu GPUs\n", ctx.device_ids.size());

    auto t0 = clock_type::now();
    ctx.generate_conformers(smiles_views, config, max_atoms, coords, energies, status);
    auto t1 = clock_type::now();

    double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Count successes.
    std::uint32_t num_succeeded = 0;
    for (auto s : status)
        if (s == conformer_status_t::success) num_succeeded++;

    double const mol_per_sec = n / (ms / 1000.0);
    double const conf_per_sec = num_succeeded / (ms / 1000.0);
    std::printf("  Time:      %.0f ms\n", ms);
    std::printf("  Mol/s:     %.1f\n", mol_per_sec);
    std::printf("  Conf/s:    %.1f\n", conf_per_sec);
    std::printf("  Succeeded: %u / %u conformers\n", num_succeeded, total_conf);
}

static void cross_validate(std::vector<std::string> const &smiles, std::vector<__half> const &coords,
                           std::vector<float> const &energies, std::vector<conformer_status_t> const &status,
                           std::uint32_t num_conformers, std::uint32_t max_atoms, std::uint32_t max_samples) {
    std::uint32_t const n = static_cast<std::uint32_t>(smiles.size());
    std::printf("\n=== Quality cross-validation ===\n");

    // Build the list of succeeded conformers.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> succeeded;
    for (std::uint32_t m = 0; m < n; ++m) {
        for (std::uint32_t c = 0; c < num_conformers; ++c) {
            std::uint32_t const slot = m * num_conformers + c;
            if (status[slot] == conformer_status_t::success) succeeded.push_back({m, c});
        }
    }

    // Select a random sample.
    std::mt19937 rng(42);
    if (succeeded.size() > max_samples) {
        std::shuffle(succeeded.begin(), succeeded.end(), rng);
        succeeded.resize(max_samples);
    }
    std::printf("  Evaluating %zu conformers via RDKit MMFF94...\n", succeeded.size());

    // Run cross-validation in parallel.
    std::vector<double> rdkit_energies(succeeded.size(), std::numeric_limits<double>::quiet_NaN());
    std::vector<double> sf_energies(succeeded.size());
    std::vector<std::uint32_t> atom_counts(succeeded.size());

#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t idx = 0; idx < succeeded.size(); ++idx) {
        auto [m, c] = succeeded[idx];
        try {
            auto mol = mol_from_smiles(smiles[m]);
            if (!mol) continue;
            bool embedded = embed_molecule(*mol);
            if (!embedded) continue;

            std::uint32_t const na = mol->getNumAtoms();
            std::uint32_t const slot = m * num_conformers + c;
            std::uint32_t const coord_offset = slot * max_atoms * 3;

            // Convert SoA half-precision coordinates to interleaved float layout.
            std::vector<float> interleaved(na * 3);
            for (std::uint32_t i = 0; i < na; ++i) {
                interleaved[i * 3 + 0] = __half2float(coords[coord_offset + 0 * max_atoms + i]);
                interleaved[i * 3 + 1] = __half2float(coords[coord_offset + 1 * max_atoms + i]);
                interleaved[i * 3 + 2] = __half2float(coords[coord_offset + 2 * max_atoms + i]);
            }
            set_coords_interleaved(*mol, interleaved.data());
            rdkit_energies[idx] = rdkit_mmff94_energy(*mol);
            sf_energies[idx] = static_cast<double>(energies[slot]);
            atom_counts[idx] = na;
        }
        catch (...) {
            // Skip failures.
        }
    }

    // Analyze the resulting energy distributions.
    std::uint32_t finite_count = 0;
    std::vector<double> valid_rdkit, valid_sf;
    for (std::size_t i = 0; i < rdkit_energies.size(); ++i) {
        if (std::isfinite(rdkit_energies[i])) {
            finite_count++;
            valid_rdkit.push_back(rdkit_energies[i]);
            valid_sf.push_back(sf_energies[i]);
        }
    }

    std::printf("  Finite:    %u / %zu (%.1f%%)\n", finite_count, succeeded.size(),
                100.0 * finite_count / succeeded.size());

    if (!valid_rdkit.empty()) {
        std::sort(valid_rdkit.begin(), valid_rdkit.end());
        std::sort(valid_sf.begin(), valid_sf.end());

        std::printf("  RDKit MMFF94 energy (kcal/mol):\n");
        std::printf("    P5=%.1f  P25=%.1f  P50=%.1f  P75=%.1f  P95=%.1f\n", percentile(valid_rdkit, 5),
                    percentile(valid_rdkit, 25), percentile(valid_rdkit, 50), percentile(valid_rdkit, 75),
                    percentile(valid_rdkit, 95));
        std::printf("  SmallField energy (kcal/mol):\n");
        std::printf("    P5=%.1f  P25=%.1f  P50=%.1f  P75=%.1f  P95=%.1f\n", percentile(valid_sf, 5),
                    percentile(valid_sf, 25), percentile(valid_sf, 50), percentile(valid_sf, 75),
                    percentile(valid_sf, 95));

        // Print the per-atom-count breakdown.
        std::printf("  Per-atom-count breakdown:\n");
        std::printf("    %6s  %6s  %8s  %8s\n", "atoms", "count", "rdkit_P50", "sf_P50");
        std::vector<std::uint32_t> bins = {20, 40, 60, 80, 100, 128};
        std::vector<double> heavy_bin_rdkit;
        for (std::size_t b = 0; b < bins.size(); ++b) {
            std::uint32_t lo = (b == 0) ? 0 : bins[b - 1];
            std::uint32_t hi = bins[b];
            std::vector<double> bin_rdkit, bin_sf;
            for (std::size_t i = 0; i < succeeded.size(); ++i) {
                if (std::isfinite(rdkit_energies[i]) && atom_counts[i] > lo && atom_counts[i] <= hi) {
                    bin_rdkit.push_back(rdkit_energies[i]);
                    bin_sf.push_back(sf_energies[i]);
                }
            }
            if (!bin_rdkit.empty()) {
                std::sort(bin_rdkit.begin(), bin_rdkit.end());
                std::sort(bin_sf.begin(), bin_sf.end());
                std::printf("    %3u-%-3u  %5zu  %8.1f  %8.1f\n", lo, hi, bin_rdkit.size(), percentile(bin_rdkit, 50),
                            percentile(bin_sf, 50));
                if (lo == 100 && hi == 128) heavy_bin_rdkit = bin_rdkit;
            }
        }

        if (!heavy_bin_rdkit.empty()) {
            double const heavy_p50 = percentile(heavy_bin_rdkit, 50);
            double const heavy_p95 = percentile(heavy_bin_rdkit, 95);
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

    if (multi_gpu_only) {
        bench_smallfield_multi_gpu(all_smiles, num_conformers, max_stereo_attempts);
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
        std::vector<conformer_status_t> status;
        bench_smallfield_gpu(smiles, num_conformers, max_stereo_attempts, coords, energies, status);

        // Run system B (multi-GPU) on the largest batch size.
        if (batch_size == batch_sizes.back()) {
            bench_smallfield_multi_gpu(smiles, num_conformers, max_stereo_attempts);
        }

        // Run quality cross-validation only on the full dataset.
        if (batch_size == static_cast<std::uint32_t>(all_smiles.size()) || batch_size == batch_sizes.back()) {
            cross_validate(smiles, coords, energies, status, num_conformers, max_atoms, quality_samples);
        }
    }

    std::printf("\nDone.\n");
    return 0;
}
