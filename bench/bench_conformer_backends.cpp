/**
 *  @file bench/bench_conformer_backends.cpp
 *  @brief Unified benchmark: RDKit CPU vs SmallField GPU on ChEMBL dataset.
 *
 *  Measures:
 *   1. RDKit CPU: OpenMP parallel EmbedMultipleConfs + MMFFOptimize
 *   2. SmallField GPU: single-GPU and multi-GPU conformer generation
 *   3. Quality cross-validation: SmallField coords evaluated by RDKit MMFF94
 *   4. Scaling analysis across batch sizes
 */

#include <cmath>  // `std::isfinite`, `std::fabs`
#include <cstdio> // `stderr`

#include <algorithm>   // `std::min`, `std::sort`
#include <array>       // `std::array`
#include <chrono>      // `std::chrono::high_resolution_clock`, `std::chrono::duration`
#include <limits>      // `std::numeric_limits`
#include <random>      // `std::mt19937`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <cuda_fp16.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <omp.h>

#include "smallfield/api/smallfield.hpp"

#include "bench/bench_arguments.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfapi = smallfield::api;
namespace sfbench = smallfield::bench;
namespace sftest = smallfield::tests;
using clock_type_t = std::chrono::high_resolution_clock;

/** Name printed on every throughput line so a concatenated CI log stays attributable. */
static constexpr char const *executable_name_k = "bench_conformer_backends";

/**
 *  @brief Ceilings on the value-taking flags, so a mistyped count is a diagnostic and not a `std::bad_alloc`.
 *
 *  The coordinate buffer is `molecules × conformers × max_atoms × 3` halves, which is where an unbounded
 *  `--conformers` lands first.
 */
static constexpr std::uint32_t max_conformers_per_molecule_k = 1024;
static constexpr std::uint32_t max_quality_samples_k = 10'000'000;

static double percentile(std::vector<double> &sorted_values, double percent) {
    if (sorted_values.empty()) return 0.0;
    double index = percent / 100.0 * static_cast<double>(sorted_values.size() - 1);
    std::size_t lower_index = static_cast<std::size_t>(index);
    std::size_t upper_index = std::min(lower_index + 1, sorted_values.size() - 1);
    double fraction = index - static_cast<double>(lower_index);
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction;
}

/** Fill @p delta with sorted per-conformer differences, smallfield minus rdkit.
 *  Inputs must be in matching sample order; the subtraction happens before any sort so
 *  the pairing survives. @p delta is caller-owned and reused across bins — it keeps its
 *  capacity, so the per-bin loop allocates once rather than once per bin. */
static void paired_energy_delta_into(std::vector<double> const &rdkit_energies,
                                     std::vector<double> const &smallfield_energies, std::vector<double> &delta) {
    delta.clear();
    for (std::size_t sample_index = 0; sample_index < rdkit_energies.size(); ++sample_index)
        delta.push_back(smallfield_energies[sample_index] - rdkit_energies[sample_index]);
    std::sort(delta.begin(), delta.end());
}

static void bench_rdkit_cpu(std::vector<std::string> const &smiles, std::uint32_t num_conformers,
                            std::uint32_t max_stereo_attempts) {
    int num_threads = omp_get_max_threads();
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    fmt::println("\n=== System A: RDKit CPU ({} threads) ===", num_threads);

    // Pre-parse all molecules outside the timed region.
    fmt::println("  Pre-parsing {} molecules...", fmt::group_digits(num_smiles));
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
    fmt::println("  Parsed: {} / {}", fmt::group_digits(num_parsed), fmt::group_digits(num_smiles));

    // Time EmbedMultipleConfs followed by MMFF94 optimization.
    fmt::println("  Running ETKDG + MMFF94 ({} conformers each)...", num_conformers);
    auto const start_time = clock_type_t::now();
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
    auto const end_time = clock_type_t::now();
    double const elapsed_ms_value = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double const mol_per_sec = num_parsed / (elapsed_ms_value / 1000.0);
    double const conf_per_sec = num_successful_conformers / (elapsed_ms_value / 1000.0);
    fmt::println("  Time:      {:.1f} ms", elapsed_ms_value);
    fmt::println("  {} mol/s:  {:.1f}", executable_name_k, mol_per_sec);
    fmt::println("  {} conf/s: {:.1f}", executable_name_k, conf_per_sec);
    fmt::println("  Succeeded: {} conformers", fmt::group_digits(num_successful_conformers));
}

static void bench_smallfield_gpu(std::vector<std::string> const &smiles, std::uint32_t num_conformers,
                                 std::uint32_t max_stereo_attempts, std::vector<__half> &coords_out,
                                 std::vector<float> &energies_out, std::vector<sfapi::conformer_status_t> &status_out) {
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    std::uint32_t const max_atoms = smallfield::types::max_atoms_per_molecule_k;
    fmt::println("\n=== System B: SmallField GPU ===");

    // Build string_view span for the library.
    std::vector<std::string_view> smiles_views(smiles.begin(), smiles.end());

    sfapi::config_t generation_config;
    generation_config.num_conformers = num_conformers;
    generation_config.max_stereo_attempts = max_stereo_attempts;

    // Allocate output buffers. Use size_t arithmetic — the coordinate
    // buffer for 10M × K=3 × 128 atoms × 3 dims exceeds UINT32_MAX elements.
    std::size_t const total_conformers = static_cast<std::size_t>(num_smiles) * num_conformers;
    coords_out.resize(total_conformers * max_atoms * 3);
    energies_out.resize(total_conformers);
    status_out.resize(total_conformers);

    // Single GPU.
    int const device_id = 0;
    sfapi::context_t gpu_context(std::span<int const>(&device_id, 1));

    auto const start_time = clock_type_t::now();
    [[maybe_unused]] auto const batch_status = gpu_context.generate_conformers(
        smiles_views, generation_config, max_atoms, coords_out, energies_out, status_out);
    auto const end_time = clock_type_t::now();

    double const elapsed_ms_value = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // `conf/s` already divides by produced conformers, so it is comparable across corpora with
    // different rejection rates. The breakdown is what makes a rejection class visible: the
    // 128-atom cap alone drops about 4 % of PubChem before any molecule reaches the GPU.
    std::uint32_t num_successful_conformers = 0;
    std::array<std::uint64_t, 8> conformers_by_status {};
    for (auto status_code : status_out) {
        if (status_code == sfapi::conformer_status_t::success_k) num_successful_conformers++;
        auto const code = static_cast<std::size_t>(status_code);
        if (code < conformers_by_status.size()) ++conformers_by_status[code];
    }

    double const mol_per_sec = num_smiles / (elapsed_ms_value / 1000.0);
    double const conf_per_sec = num_successful_conformers / (elapsed_ms_value / 1000.0);
    fmt::println("  Time:      {:.1f} ms", elapsed_ms_value);
    fmt::println("  {} mol/s:  {:.1f}", executable_name_k, mol_per_sec);
    fmt::println("  {} conf/s: {:.1f}", executable_name_k, conf_per_sec);
    fmt::println("  Succeeded: {} / {} conformers", fmt::group_digits(num_successful_conformers),
                 fmt::group_digits(total_conformers));
    constexpr std::array<char const *, 8> status_names {
        "success",        "stereo_failure",     "mmff_unconverged",    "invalid_smiles",
        "too_many_atoms", "preprocess_timeout", "unsupported_element", "other"};
    for (std::size_t code = 1; code < conformers_by_status.size(); ++code)
        if (conformers_by_status[code] > 0)
            fmt::println(
                "    {:<20} {:>12}  {:.3f} %", status_names[code], fmt::group_digits(conformers_by_status[code]),
                100.0 * static_cast<double>(conformers_by_status[code]) / static_cast<double>(total_conformers));
    std::uint64_t const cuda_errors = gpu_context.cuda_error_count.load();
    if (cuda_errors > 0)
        fmt::println(stderr, "  cuda errors during single-GPU bench: {}", fmt::group_digits(cuda_errors));
}

static void bench_smallfield_multi_gpu(std::vector<std::string> const &smiles, std::uint32_t num_conformers,
                                       std::uint32_t max_stereo_attempts) {
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    std::uint32_t const max_atoms = smallfield::types::max_atoms_per_molecule_k;
    fmt::println("\n=== System B: SmallField multi-GPU (all GPUs) ===");

    // Build string_view span for the library.
    std::vector<std::string_view> smiles_views(smiles.begin(), smiles.end());

    sfapi::config_t generation_config;
    generation_config.num_conformers = num_conformers;
    generation_config.max_stereo_attempts = max_stereo_attempts;

    // Allocate output buffers. Use size_t arithmetic — the coordinate
    // buffer for 10M × K=3 × 128 atoms × 3 dims exceeds UINT32_MAX elements.
    std::size_t const total_conformers = static_cast<std::size_t>(num_smiles) * num_conformers;
    std::vector<__half> coords(total_conformers * max_atoms * 3);
    std::vector<float> energies(total_conformers);
    std::vector<sfapi::conformer_status_t> status(total_conformers);

    // All GPUs — an empty device span.
    sfapi::context_t gpu_context;
    // Opt in to scaffold-based SubstructMatch caching. Amortises RDKit fallback cost across molecules sharing
    // the same Murcko scaffold; safe no-op if disabled or for acyclic molecules.

    // Allow single-GPU runs via `--multi-gpu-only`, which scaling sweeps use when they control the
    // GPU count through `CUDA_VISIBLE_DEVICES`. The flag says "multi" but the path runs correctly
    // on one GPU.
    if (gpu_context.device_ids.empty()) {
        fmt::println("  Skipped (no GPU available)");
        return;
    }
    fmt::println("  Using {} GPUs: [{}]", gpu_context.device_ids.size(), fmt::join(gpu_context.device_ids, ", "));

    auto const start_time = clock_type_t::now();
    [[maybe_unused]] auto const batch_status = gpu_context.generate_conformers(smiles_views, generation_config,
                                                                               max_atoms, coords, energies, status);
    auto const end_time = clock_type_t::now();

    double const elapsed_ms_value = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Count successes.
    std::uint32_t num_successful_conformers = 0;
    for (auto status_code : status)
        if (status_code == sfapi::conformer_status_t::success_k) num_successful_conformers++;

    double const mol_per_sec = num_smiles / (elapsed_ms_value / 1000.0);
    double const conf_per_sec = num_successful_conformers / (elapsed_ms_value / 1000.0);
    fmt::println("  Time:      {:.1f} ms", elapsed_ms_value);
    fmt::println("  {} mol/s:  {:.1f}", executable_name_k, mol_per_sec);
    fmt::println("  {} conf/s: {:.1f}", executable_name_k, conf_per_sec);
    fmt::println("  Succeeded: {} / {} conformers", fmt::group_digits(num_successful_conformers),
                 fmt::group_digits(total_conformers));
    std::uint64_t const cuda_errors = gpu_context.cuda_error_count.load();
    if (cuda_errors > 0)
        fmt::println(stderr, "  cuda errors during multi-GPU bench: {}", fmt::group_digits(cuda_errors));
}

static void cross_validate(std::vector<std::string> const &smiles, std::vector<__half> const &coords,
                           std::vector<float> const &energies, std::vector<sfapi::conformer_status_t> const &status,
                           std::uint32_t num_conformers, std::uint32_t max_atoms, std::uint32_t max_samples) {
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    fmt::println("\n=== Quality cross-validation ===");

    // Build the list of succeeded conformers.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> successful_conformers;
    for (std::uint32_t molecule_index = 0; molecule_index < num_smiles; ++molecule_index) {
        for (std::uint32_t conformer_index = 0; conformer_index < num_conformers; ++conformer_index) {
            std::uint32_t const flat_index = molecule_index * num_conformers + conformer_index;
            if (status[flat_index] == sfapi::conformer_status_t::success_k)
                successful_conformers.push_back({molecule_index, conformer_index});
        }
    }

    // Select a random sample.
    std::mt19937 random_engine(42);
    if (successful_conformers.size() > max_samples) {
        std::shuffle(successful_conformers.begin(), successful_conformers.end(), random_engine);
        successful_conformers.resize(max_samples);
    }
    fmt::println("  Evaluating {} conformers via RDKit MMFF94...", fmt::group_digits(successful_conformers.size()));

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

    fmt::println("  Finite:    {} / {} ({:.3f} %)", fmt::group_digits(num_finite_energies),
                 fmt::group_digits(successful_conformers.size()),
                 100.0 * num_finite_energies / static_cast<double>(successful_conformers.size()));

    if (!finite_rdkit_energies.empty()) {
        // Both columns describe the same geometry — RDKit is evaluated on SmallField's
        // own output coordinates — so the per-conformer difference is the quantity that
        // measures agreement. Independently sorted marginals do not: a uniform offset,
        // a heavy negative tail, and partial cancellation all produce the same shift.
        // Scratch reused by every delta computation below, including the per-bin loop.
        std::vector<double> paired_delta;
        paired_delta.reserve(finite_rdkit_energies.size());
        paired_energy_delta_into(finite_rdkit_energies, finite_smallfield_energies, paired_delta);

        // Deltas carry `{:+.3g}` per the diagnostics table: the sign is always visible and three digits
        // show the magnitude, which `.1f` hid whenever a bin agreed to better than 0.05 kcal/mol.
        fmt::println("  SmallField − RDKit, paired per conformer (kcal/mol):");
        fmt::println("    P5={:+.3g}  P25={:+.3g}  P50={:+.3g}  P75={:+.3g}  P95={:+.3g}  min={:+.3g}  max={:+.3g}",
                     percentile(paired_delta, 5), percentile(paired_delta, 25), percentile(paired_delta, 50),
                     percentile(paired_delta, 75), percentile(paired_delta, 95), paired_delta.front(),
                     paired_delta.back());

        // Marginals are retained for context only; they do not measure agreement.
        std::sort(finite_rdkit_energies.begin(), finite_rdkit_energies.end());
        std::sort(finite_smallfield_energies.begin(), finite_smallfield_energies.end());

        fmt::println("  RDKit MMFF94 energy (kcal/mol):");
        fmt::println("    P5={:.1f}  P25={:.1f}  P50={:.1f}  P75={:.1f}  P95={:.1f}",
                     percentile(finite_rdkit_energies, 5), percentile(finite_rdkit_energies, 25),
                     percentile(finite_rdkit_energies, 50), percentile(finite_rdkit_energies, 75),
                     percentile(finite_rdkit_energies, 95));
        fmt::println("  SmallField energy (kcal/mol):");
        fmt::println("    P5={:.1f}  P25={:.1f}  P50={:.1f}  P75={:.1f}  P95={:.1f}",
                     percentile(finite_smallfield_energies, 5), percentile(finite_smallfield_energies, 25),
                     percentile(finite_smallfield_energies, 50), percentile(finite_smallfield_energies, 75),
                     percentile(finite_smallfield_energies, 95));

        // Print the per-atom-count breakdown. `delta_min` surfaces the negative tail
        // that a shifted median hides — the MMFF cubic bend term is unbounded below
        // past Δθ > 143°, and only the minimum exposes a conformer that fell into it.
        fmt::println("  Per-atom-count breakdown:");
        fmt::println("    {:>7}  {:>8}  {:>10}  {:>10}  {:>10}", "atoms", "count", "delta_P50", "delta_P95",
                     "delta_min");
        constexpr std::uint32_t atom_count_bins[] = {20, 40, 60, 80, 100, 128};
        // Four scratch buffers hoisted out of the bin loop; each is cleared per bin and
        // keeps its capacity, so the whole breakdown costs one growth phase.
        std::vector<double> bin_rdkit_energies, bin_smallfield_energies, bin_delta;
        std::vector<double> heavy_atom_rdkit_energies, heavy_atom_delta;
        for (std::size_t bin_index = 0; bin_index < std::size(atom_count_bins); ++bin_index) {
            std::uint32_t lower_bound = (bin_index == 0) ? 0 : atom_count_bins[bin_index - 1];
            std::uint32_t upper_bound = atom_count_bins[bin_index];
            bin_rdkit_energies.clear();
            bin_smallfield_energies.clear();
            for (std::size_t sample_index = 0; sample_index < successful_conformers.size(); ++sample_index) {
                if (std::isfinite(rdkit_reference_energies[sample_index]) && atom_counts[sample_index] > lower_bound &&
                    atom_counts[sample_index] <= upper_bound) {
                    bin_rdkit_energies.push_back(rdkit_reference_energies[sample_index]);
                    bin_smallfield_energies.push_back(smallfield_energies[sample_index]);
                }
            }
            if (bin_rdkit_energies.empty()) continue;
            paired_energy_delta_into(bin_rdkit_energies, bin_smallfield_energies, bin_delta);
            fmt::println("    {:>3}-{:<3}  {:>8}  {:>+10.3g}  {:>+10.3g}  {:>+10.3g}", lower_bound, upper_bound,
                         fmt::group_digits(bin_delta.size()), percentile(bin_delta, 50), percentile(bin_delta, 95),
                         bin_delta.front());
            if (lower_bound == 100 && upper_bound == 128) {
                std::sort(bin_rdkit_energies.begin(), bin_rdkit_energies.end());
                heavy_atom_rdkit_energies = bin_rdkit_energies;
                heavy_atom_delta = bin_delta;
            }
        }

        if (!heavy_atom_rdkit_energies.empty()) {
            double const heavy_p50 = percentile(heavy_atom_rdkit_energies, 50);
            double const heavy_p95 = percentile(heavy_atom_rdkit_energies, 95);
            double const heavy_delta_p50 = percentile(heavy_atom_delta, 50);
            // The absolute bounds catch a blown-up geometry; the paired median catches a
            // force-field divergence that leaves the absolute energies looking plausible.
            char const *failed_threshold = nullptr;
            if (heavy_p50 >= 180.0) failed_threshold = "P50 >= 180";
            else if (heavy_p95 >= 300.0) failed_threshold = "P95 >= 300";
            else if (std::fabs(heavy_delta_p50) >= 10.0) failed_threshold = "|delta_P50| >= 10";
            // Verdict grammar shared with `test/test_helpers.hpp`: verb at column zero, one space after it,
            // `key=value` one space apart, so `grep '^FAIL heavy_bin_gate'` answers over a concatenated CI log.
            // The three numbers decide the verdict, so they print at `{}` — shortest round-trip — and a value
            // that only just cleared its threshold can never render as one that just missed it.
            if (failed_threshold == nullptr)
                fmt::println("PASS heavy_bin_gate atoms=100-128 P50={} P95={} delta_P50={}", heavy_p50, heavy_p95,
                             heavy_delta_p50);
            else
                fmt::println(stderr, "FAIL heavy_bin_gate: {} atoms=100-128 P50={} P95={} delta_P50={}",
                             failed_threshold, heavy_p50, heavy_p95, heavy_delta_p50);
        }
    }
}

/**
 *  @brief Read the value that follows @p flag into @p number, or explain on `stderr` why it could not be.
 *
 *  Three silent failures the loop used to have are refusals here: a token that is not a number, a token
 *  outside `[1, maximum]`, and a value-taking flag that appears last with nothing after it.
 */
static bool take_number_argument_(std::string_view flag, int argc, char **argv, int &argument_index,
                                  std::uint32_t maximum, std::uint32_t &number) {
    if (argument_index + 1 >= argc) {
        fmt::println(stderr, "{}: {} takes a value, and none followed it", executable_name_k, flag);
        return false;
    }
    std::string_view const text = argv[++argument_index];
    auto const status = sfbench::parse_number(text, std::uint32_t {1}, maximum, number);
    if (status == sfbench::number_parse_t::parsed_k) return true;
    fmt::println(stderr, "{}: {} value '{}' {} (accepted range 1..{})", executable_name_k, flag, text,
                 sfbench::describe_number_parse(status), fmt::group_digits(maximum));
    return false;
}

int main(int argc, char **argv) {
    fmt::println("bench_chembl: SmallField GPU vs RDKit CPU");
    fmt::println("==========================================");

    // Set defaults.
    std::string smiles_path = "/tmp/chembl_valid.smi";
    std::uint32_t num_conformers = 3;
    std::uint32_t max_stereo_attempts = 10;
    std::uint32_t max_atoms = 128;
    std::uint32_t quality_samples = 10000;

    // Parse the command line. Every branch either consumes its token or refuses the whole command line: an
    // unrecognised flag and a missing value both used to be dropped without a word, so a benchmark run under
    // a mistyped flag reported the defaults as though they were what was asked for.
    bool multi_gpu_only = false;
    for (int argument_index = 1; argument_index < argc; ++argument_index) {
        std::string_view const flag = argv[argument_index];
        if (flag == "--smi") {
            if (argument_index + 1 >= argc) {
                fmt::println(stderr, "{}: --smi takes a value, and none followed it", executable_name_k);
                return 2;
            }
            smiles_path = argv[++argument_index];
        }
        else if (flag == "--conformers") {
            if (!take_number_argument_(flag, argc, argv, argument_index, max_conformers_per_molecule_k,
                                       num_conformers))
                return 2;
        }
        else if (flag == "--quality-samples") {
            if (!take_number_argument_(flag, argc, argv, argument_index, max_quality_samples_k, quality_samples))
                return 2;
        }
        else if (flag == "--multi-gpu-only") { multi_gpu_only = true; }
        else {
            fmt::println(stderr, "{}: unknown argument '{}'", executable_name_k, flag);
            fmt::println(stderr, "usage: {} [--smi PATH] [--conformers N] [--quality-samples N] [--multi-gpu-only]",
                         argv[0]);
            return 2;
        }
    }

    fmt::println("Dataset:     {}", smiles_path);
    fmt::println("Conformers:  {} per molecule", num_conformers);
    fmt::println("CPU threads: {}", omp_get_max_threads());

    auto all_smiles = sftest::load_smiles(smiles_path, std::numeric_limits<std::size_t>::max());
    fmt::println("Loaded:      {} SMILES", fmt::group_digits(all_smiles.size()));
    if (all_smiles.empty()) {
        fmt::println(stderr, "{}: no SMILES loaded from '{}'", executable_name_k, smiles_path);
        return 1;
    }

    if (multi_gpu_only) {
        bench_smallfield_multi_gpu(all_smiles, num_conformers, max_stereo_attempts);
        fmt::println("\nDone.");
        return 0;
    }

    // Run the scaling analysis at multiple batch sizes.
    std::vector<std::uint32_t> batch_sizes = {100, 500, 1000, 2000, 5000};
    if (all_smiles.size() > 5000) batch_sizes.push_back(static_cast<std::uint32_t>(all_smiles.size()));
    fmt::println("Batch sizes: [{}]", fmt::join(batch_sizes, ", "));

    for (auto batch_size : batch_sizes) {
        if (batch_size > all_smiles.size()) continue;

        std::vector<std::string> smiles(all_smiles.begin(), all_smiles.begin() + batch_size);
        fmt::println("\n\n############## batch_size = {} ##############", fmt::group_digits(batch_size));

        // Run system A: RDKit CPU.
        bench_rdkit_cpu(smiles, num_conformers, max_stereo_attempts);

        // Run system B: SmallField single GPU.
        std::vector<__half> coords;
        std::vector<float> energies;
        std::vector<sfapi::conformer_status_t> status;
        bench_smallfield_gpu(smiles, num_conformers, max_stereo_attempts, coords, energies, status);

        // Multi-GPU scaling is measured through a dedicated fresh process via `--multi-gpu-only`. Keeping the
        // mixed CPU + 1-GPU default path in a single process avoids state bleed between the 1-GPU and 8-GPU
        // runs on the largest batches.
        if (batch_size == batch_sizes.back()) {
            fmt::println("\n=== System B: SmallField multi-GPU (all GPUs) ===");
            fmt::println("  Skipped in mixed mode. Re-run with --multi-gpu-only for isolated multi-GPU numbers.");
        }

        // Run quality cross-validation only on the full dataset.
        if (batch_size == static_cast<std::uint32_t>(all_smiles.size()) || batch_size == batch_sizes.back()) {
            cross_validate(smiles, coords, energies, status, num_conformers, max_atoms, quality_samples);
        }
    }

    fmt::println("\nDone.");
    return 0;
}
