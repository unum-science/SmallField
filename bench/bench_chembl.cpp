/**
 * @file bench_chembl.cpp
 * @brief Unified benchmark: RDKit CPU vs SmallField GPU on ChEMBL dataset.
 *
 * Measures:
 *   1. RDKit CPU: OpenMP parallel EmbedMultipleConfs + MMFFOptimize
 *   2. SmallField GPU: CPU preprocessing + GPU kernel (stages 2-9)
 *   3. Quality cross-validation: SmallField coords evaluated by RDKit MMFF94
 *   4. Scaling analysis across batch sizes
 */

#include "bench_chembl_kernels.h"
#include "etkdg_preprocess.hpp"
#include "test_helpers.hpp"

#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <vector>

using namespace smallfield;
using namespace smallfield::test;
using clock_type = std::chrono::high_resolution_clock;

// load SMILES from file (one per line)
static std::vector<std::string> load_smiles(std::string const& path, std::uint32_t max_count = 0) {
    std::vector<std::string> result;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        // take first whitespace-delimited token (SMILES may be followed by name)
        auto end = line.find_first_of(" \t");
        if (end != std::string::npos)
            line = line.substr(0, end);
        result.push_back(line);
        if (max_count > 0 && result.size() >= max_count)
            break;
    }
    return result;
}

static double percentile(std::vector<double>& sorted_values, double p) {
    if (sorted_values.empty())
        return 0.0;
    double idx = p / 100.0 * static_cast<double>(sorted_values.size() - 1);
    std::size_t lo = static_cast<std::size_t>(idx);
    std::size_t hi = std::min(lo + 1, sorted_values.size() - 1);
    double frac = idx - static_cast<double>(lo);
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac;
}

// section 1: RDKit CPU benchmark
static void bench_rdkit_cpu(std::vector<std::string> const& smiles, std::uint32_t num_conformers,
                            std::uint32_t max_stereo_attempts) {
    int num_threads = omp_get_max_threads();
    std::uint32_t const n = static_cast<std::uint32_t>(smiles.size());
    std::printf("\n=== System A: RDKit CPU (%d threads) ===\n", num_threads);

    // pre-parse all molecules (untimed)
    std::printf("  Pre-parsing %u molecules...\n", n);
    std::vector<std::unique_ptr<RDKit::RWMol>> mols(n);
    std::uint32_t parse_ok = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : parse_ok)
    for (std::uint32_t i = 0; i < n; ++i) {
        try {
            mols[i] = mol_from_smiles(smiles[i]);
            if (mols[i])
                parse_ok++;
        } catch (...) {
            mols[i] = nullptr;
        }
    }
    std::printf("  Parsed: %u / %u\n", parse_ok, n);

    // timed: EmbedMultipleConfs + MMFFOptimize
    std::printf("  Running ETKDG + MMFF94 (%u conformers each)...\n", num_conformers);
    auto t0 = clock_type::now();
    std::uint32_t total_conf_ok = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : total_conf_ok)
    for (std::uint32_t i = 0; i < n; ++i) {
        if (!mols[i])
            continue;
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
        } catch (...) {
            // bad molecule, skip
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

// section 2: SmallField GPU benchmark
static void bench_smallfield_gpu(std::vector<std::string> const& smiles, std::uint32_t num_conformers,
                                 std::uint32_t max_stereo_attempts, std::vector<float>& coords_out,
                                 std::vector<float>& energies_out, std::vector<std::uint8_t>& status_out,
                                 etkdg_molecule_batch_t& batch_out) {
    std::uint32_t const n = static_cast<std::uint32_t>(smiles.size());
    std::printf("\n=== System B: SmallField GPU ===\n");

    // timed phase 1: CPU preprocessing
    std::printf("  Preprocessing %u molecules (OpenMP)...\n", n);
    auto t0 = clock_type::now();
    batch_out = prepare_molecule_batch(smiles);
    auto t1_preproc = clock_type::now();
    double const preproc_ms = std::chrono::duration<double, std::milli>(t1_preproc - t0).count();
    std::printf("  Preprocessed: %u molecules, max_atoms=%u\n", batch_out.num_molecules,
                batch_out.max_atoms_per_molecule);
    std::printf("  Preprocessing: %.0f ms\n", preproc_ms);

    // allocate output
    std::uint32_t const total_conf = batch_out.num_molecules * num_conformers;
    std::uint32_t const max_coords = total_conf * batch_out.max_atoms_per_molecule * 3;
    coords_out.resize(max_coords);
    energies_out.resize(total_conf);
    status_out.resize(total_conf);

    // timed phase 2: GPU upload + kernel
    auto result = gpu_etkdg_mmff_pipeline(
        batch_out.bounds_matrices.data(), static_cast<std::uint32_t>(batch_out.bounds_matrices.size()),
        batch_out.chiral_centers.empty() ? nullptr : batch_out.chiral_centers.data(),
        static_cast<std::uint32_t>(batch_out.chiral_centers.size()),
        batch_out.etk_torsions.empty() ? nullptr : batch_out.etk_torsions.data(),
        static_cast<std::uint32_t>(batch_out.etk_torsions.size()),
        batch_out.distance_constraints.empty() ? nullptr : batch_out.distance_constraints.data(),
        static_cast<std::uint32_t>(batch_out.distance_constraints.size()),
        batch_out.mmff_nonbonded.empty() ? nullptr : batch_out.mmff_nonbonded.data(),
        static_cast<std::uint32_t>(batch_out.mmff_nonbonded.size()),
        batch_out.mmff_bonds.empty() ? nullptr : batch_out.mmff_bonds.data(),
        static_cast<std::uint32_t>(batch_out.mmff_bonds.size()),
        batch_out.mmff_angles.empty() ? nullptr : batch_out.mmff_angles.data(),
        static_cast<std::uint32_t>(batch_out.mmff_angles.size()),
        batch_out.mmff_torsions.empty() ? nullptr : batch_out.mmff_torsions.data(),
        static_cast<std::uint32_t>(batch_out.mmff_torsions.size()),
        batch_out.mmff_oops.empty() ? nullptr : batch_out.mmff_oops.data(),
        static_cast<std::uint32_t>(batch_out.mmff_oops.size()),
        batch_out.mmff_stretchbends.empty() ? nullptr : batch_out.mmff_stretchbends.data(),
        static_cast<std::uint32_t>(batch_out.mmff_stretchbends.size()),
        batch_out.mmff_interactions.empty() ? nullptr : batch_out.mmff_interactions.data(),
        static_cast<std::uint32_t>(batch_out.mmff_interactions.size()),
        batch_out.mmff_atom_offsets.empty() ? nullptr : batch_out.mmff_atom_offsets.data(),
        static_cast<std::uint32_t>(batch_out.mmff_atom_offsets.size()), batch_out.molecule_metadata.data(),
        batch_out.num_molecules, coords_out.data(), energies_out.data(), status_out.data(), num_conformers,
        max_stereo_attempts, batch_out.max_atoms_per_molecule, 42);

    double const gpu_total_ms = result.upload_time_ms + result.kernel_time_ms + result.download_time_ms;
    double const e2e_ms = preproc_ms + gpu_total_ms;
    double const mol_per_sec = batch_out.num_molecules / (e2e_ms / 1000.0);
    double const conf_per_sec = result.num_succeeded / (e2e_ms / 1000.0);

    std::printf("  GPU: grid=%u (%u blocks/SM)\n", result.grid_size, result.blocks_per_sm);
    std::printf("  Upload:      %.2f ms\n", result.upload_time_ms);
    std::printf("  Kernel:      %.2f ms\n", result.kernel_time_ms);
    std::printf("  Download:    %.2f ms\n", result.download_time_ms);
    std::printf("  GPU total:   %.2f ms (upload + kernel + download)\n", gpu_total_ms);
    std::printf("  End-to-end:  %.0f ms (preproc %.0f + GPU %.0f)\n", e2e_ms, preproc_ms, gpu_total_ms);
    std::printf("  Mol/s:       %.1f\n", mol_per_sec);
    std::printf("  Conf/s:      %.1f\n", conf_per_sec);
    std::printf("  Succeeded:   %u / %u conformers\n", result.num_succeeded, total_conf);

    // GPU memory usage estimate
    std::size_t bounds_bytes = batch_out.bounds_matrices.size() * sizeof(float);
    std::size_t params_bytes = batch_out.mmff_nonbonded.size() * sizeof(mmff94_nonbonded_pair_t) +
                               batch_out.mmff_bonds.size() * sizeof(mmff94_bond_t) +
                               batch_out.mmff_angles.size() * sizeof(mmff94_angle_t) +
                               batch_out.mmff_torsions.size() * sizeof(mmff94_torsion_t) +
                               batch_out.mmff_interactions.size() * sizeof(mmff94_atom_interaction_t) +
                               batch_out.mmff_atom_offsets.size() * sizeof(mmff94_atom_offsets_t);
    std::printf("  GPU upload:  %.1f MB (bounds %.1f + params %.1f)\n", (bounds_bytes + params_bytes) / 1e6,
                bounds_bytes / 1e6, params_bytes / 1e6);
}

// section 3: quality cross-validation
static void cross_validate(std::vector<std::string> const& smiles, etkdg_molecule_batch_t const& batch,
                           std::vector<float> const& coords, std::vector<float> const& energies,
                           std::vector<std::uint8_t> const& status, std::uint32_t num_conformers,
                           std::uint32_t max_samples) {
    std::printf("\n=== Quality cross-validation ===\n");

    // build list of (mol_index, conf_index) for succeeded conformers
    std::vector<std::pair<std::uint32_t, std::uint32_t>> succeeded;
    for (std::uint32_t m = 0; m < batch.num_molecules; ++m) {
        for (std::uint32_t c = 0; c < num_conformers; ++c) {
            std::uint32_t const slot = m * num_conformers + c;
            if (status[slot] == 0)
                succeeded.push_back({m, c});
        }
    }

    // random sample
    std::mt19937 rng(42);
    if (succeeded.size() > max_samples) {
        std::shuffle(succeeded.begin(), succeeded.end(), rng);
        succeeded.resize(max_samples);
    }
    std::printf("  Evaluating %zu conformers via RDKit MMFF94...\n", succeeded.size());

    // parallel cross-validation
    std::vector<double> rdkit_energies(succeeded.size(), std::numeric_limits<double>::quiet_NaN());
    std::vector<double> sf_energies(succeeded.size());
    std::vector<std::uint32_t> atom_counts(succeeded.size());

#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t idx = 0; idx < succeeded.size(); ++idx) {
        auto [m, c] = succeeded[idx];
        try {
            auto mol = mol_from_smiles(smiles[m]);
            if (!mol)
                continue;
            bool embedded = embed_molecule(*mol);
            if (!embedded)
                continue;

            std::uint32_t const na = batch.molecule_metadata[m].num_atoms;
            std::uint32_t const max_n = batch.max_atoms_per_molecule;
            std::uint32_t const slot = m * num_conformers + c;
            std::uint32_t const off = slot * max_n * 3;

            // convert SoA → interleaved
            std::vector<float> interleaved(na * 3);
            for (std::uint32_t i = 0; i < na; ++i) {
                interleaved[i * 3 + 0] = coords[off + 0 * max_n + i];
                interleaved[i * 3 + 1] = coords[off + 1 * max_n + i];
                interleaved[i * 3 + 2] = coords[off + 2 * max_n + i];
            }
            set_coords_interleaved(*mol, interleaved.data());
            rdkit_energies[idx] = rdkit_mmff94_energy(*mol);
            sf_energies[idx] = static_cast<double>(energies[slot]);
            atom_counts[idx] = na;
        } catch (...) {
            // skip
        }
    }

    // analyze
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

        // per-atom-count breakdown
        std::printf("  Per-atom-count breakdown:\n");
        std::printf("    %6s  %6s  %8s  %8s\n", "atoms", "count", "rdkit_P50", "sf_P50");
        std::vector<std::uint32_t> bins = {20, 40, 60, 80, 100, 128};
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
            }
        }
    }
}

int main(int argc, char** argv) {
    std::printf("bench_chembl: SmallField GPU vs RDKit CPU\n");
    std::printf("==========================================\n");

    // defaults
    std::string smi_path = "/tmp/chembl_valid.smi";
    std::uint32_t num_conformers = 3;
    std::uint32_t max_stereo_attempts = 10;
    std::uint32_t quality_samples = 10000;

    // simple arg parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--smi" && i + 1 < argc)
            smi_path = argv[++i];
        else if (arg == "--conformers" && i + 1 < argc)
            num_conformers = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        else if (arg == "--quality-samples" && i + 1 < argc)
            quality_samples = static_cast<std::uint32_t>(std::atoi(argv[++i]));
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

    // scaling analysis: run at multiple batch sizes
    std::vector<std::uint32_t> batch_sizes = {100, 500, 1000, 2000, 5000};
    if (all_smiles.size() > 5000)
        batch_sizes.push_back(static_cast<std::uint32_t>(all_smiles.size()));

    for (auto batch_size : batch_sizes) {
        if (batch_size > all_smiles.size())
            continue;

        std::vector<std::string> smiles(all_smiles.begin(), all_smiles.begin() + batch_size);
        std::printf("\n\n############## batch_size = %u ##############\n", batch_size);

        // system A: RDKit CPU
        bench_rdkit_cpu(smiles, num_conformers, max_stereo_attempts);

        // system B: SmallField GPU
        std::vector<float> coords, energies_f;
        std::vector<std::uint8_t> status;
        etkdg_molecule_batch_t batch;
        bench_smallfield_gpu(smiles, num_conformers, max_stereo_attempts, coords, energies_f, status, batch);

        // quality cross-validation (only on full dataset)
        if (batch_size == static_cast<std::uint32_t>(all_smiles.size()) || batch_size == batch_sizes.back()) {
            cross_validate(smiles, batch, coords, energies_f, status, num_conformers, quality_samples);
        }
    }

    std::printf("\nDone.\n");
    return 0;
}
