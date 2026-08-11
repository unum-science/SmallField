/**
 *  @file bench/bench_conformer_backends.cpp
 *  @brief Unified benchmark: RDKit CPU vs SmallField GPU over a SMILES corpus.
 *
 *  Measures:
 *   1. RDKit CPU: OpenMP parallel EmbedMultipleConfs + MMFFOptimize
 *   2. SmallField GPU: single-GPU and multi-GPU conformer generation
 *   3. Quality cross-validation: SmallField coords evaluated by RDKit MMFF94
 *   4. Scaling analysis across batch sizes
 *
 *  The corpus arrives through `--smi`, or through `SMALLFIELD_TEST_CORPUS`, and a path that does not open is a
 *  refusal rather than an empty run. Exit codes are `0` for a clean run, `1` when the quality gate fails, and
 *  `2` when the command line or the corpus is refused.
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
#include "smallfield/types/views.hpp"

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

/**
 *  @brief Fewest heavy-bin samples the gate will render a verdict on.
 *
 *  Drug-like corpora put around 2 % of their molecules in the 100-128 atom bin — 97 of 4,788 on the corpus
 *  behind `BENCH.md` — so `--quality-samples 512` reaches the gate with about ten. A P95 over ten values is
 *  its second-largest element, and a verdict on that is the silent-validator shape this project has removed
 *  twice already. Clearing this floor takes roughly 1,500 quality samples.
 */
static constexpr std::size_t min_heavy_bin_samples_k = 30;

/**
 *  @brief Atom stride of every conformer slot in the coordinate pool.
 *
 *  The generator is handed this value and the cross-validator reads back with it, so both must name the same
 *  constant. Reading a pool written at one stride with another silently returns a neighbouring conformer's
 *  coordinates, which scores as a plausible-looking geometry with an implausible energy.
 */
static constexpr std::uint32_t coordinate_stride_atoms_k = smallfield::types::max_atoms_per_molecule_k;

#pragma region Statistics

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

/** Upper bounds of the reported atom-count bins; bin `bin` covers atom counts in `(lower, bounds[bin]]`. */
static constexpr std::uint32_t atom_count_bin_bounds_k[] = {20, 40, 60, 80, 100, 128};
static constexpr std::size_t num_atom_count_bins_k = std::size(atom_count_bin_bounds_k);

/** Exclusive lower bound of bin @p bin. */
static constexpr std::uint32_t atom_count_bin_lower_bound_(std::size_t bin) {
    return bin == 0 ? 0u : atom_count_bin_bounds_k[bin - 1];
}

/**
 *  @brief Index of the bin covering @p num_atoms, or `num_atom_count_bins_k` when no bin does.
 *
 *  Past-the-end is the not-found answer, as in `std::find`. A zero atom count marks a sample RDKit never
 *  scored, and a count above the last bound is a molecule larger than the breakdown tabulates.
 */
static std::size_t atom_count_bin_of_(std::uint32_t num_atoms) {
    for (std::size_t bin = 0; bin < num_atom_count_bins_k; ++bin)
        if (num_atoms > atom_count_bin_lower_bound_(bin) && num_atoms <= atom_count_bin_bounds_k[bin]) return bin;
    return num_atom_count_bins_k;
}

#pragma endregion Statistics

#pragma region Benchmarked Systems

static void bench_rdkit_cpu(std::vector<std::string> const &smiles, std::uint32_t num_conformers,
                            std::uint32_t max_stereo_attempts) {
    int num_threads = omp_get_max_threads();
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    fmt::println("\nSystem A: RDKit CPU, {} threads", num_threads);

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

/** Print every non-empty failure class in @p conformers_by_status as a count and a share of the total. */
static void print_status_histogram_(std::array<std::uint64_t, 8> const &conformers_by_status,
                                    std::size_t total_conformers) {
    constexpr std::array<char const *, 8> status_names {
        "success",        "stereo_failure",     "mmff_unconverged",    "invalid_smiles",
        "too_many_atoms", "preprocess_timeout", "unsupported_element", "other"};
    for (std::size_t code = 1; code < conformers_by_status.size(); ++code)
        if (conformers_by_status[code] > 0)
            fmt::println(
                "    {:<20} {:>12}  {:.3f} %", status_names[code], fmt::group_digits(conformers_by_status[code]),
                100.0 * static_cast<double>(conformers_by_status[code]) / static_cast<double>(total_conformers));
}

static void bench_smallfield_gpu(std::vector<std::string> const &smiles, std::uint32_t num_conformers,
                                 std::uint32_t max_stereo_attempts, std::vector<__half> &coords_out,
                                 std::vector<float> &energies_out, std::vector<sfapi::conformer_status_t> &status_out) {
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    fmt::println("\nSystem B: SmallField GPU");

    // Build string_view span for the library.
    std::vector<std::string_view> smiles_views(smiles.begin(), smiles.end());

    sfapi::config_t generation_config;
    generation_config.num_conformers = num_conformers;
    generation_config.max_stereo_attempts = max_stereo_attempts;

    // Allocate output buffers. Use size_t arithmetic — the coordinate
    // buffer for 10M × K=3 × 128 atoms × 3 dims exceeds UINT32_MAX elements.
    std::size_t const total_conformers = static_cast<std::size_t>(num_smiles) * num_conformers;
    coords_out.resize(total_conformers * coordinate_stride_atoms_k * 3);
    energies_out.resize(total_conformers);
    status_out.resize(total_conformers);

    // Single GPU.
    int const device_id = 0;
    sfapi::context_t gpu_context(custd::span<int const>(&device_id, 1));

    auto const start_time = clock_type_t::now();
    [[maybe_unused]] auto const batch_status = gpu_context.generate_conformers(
        smiles_views, generation_config, coordinate_stride_atoms_k, coords_out, energies_out, status_out);
    auto const end_time = clock_type_t::now();

    double const elapsed_ms_value = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // `conf/s` already divides by produced conformers, so it is comparable across corpora with
    // different rejection rates. The breakdown is what makes a rejection class visible: the
    // atom-count cap alone drops about 4 % of PubChem before any molecule reaches the GPU.
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
    print_status_histogram_(conformers_by_status, total_conformers);
    std::uint64_t const cuda_errors = gpu_context.cuda_error_count.load();
    if (cuda_errors > 0)
        fmt::println(stderr, "  cuda errors during single-GPU bench: {}", fmt::group_digits(cuda_errors));
}

static void bench_smallfield_multi_gpu(std::vector<std::string> const &smiles, std::uint32_t num_conformers,
                                       std::uint32_t max_stereo_attempts) {
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    fmt::println("\nSystem B: SmallField multi-GPU, all visible devices");

    // Build string_view span for the library.
    std::vector<std::string_view> smiles_views(smiles.begin(), smiles.end());

    sfapi::config_t generation_config;
    generation_config.num_conformers = num_conformers;
    generation_config.max_stereo_attempts = max_stereo_attempts;

    // Allocate output buffers. Use size_t arithmetic — the coordinate
    // buffer for 10M × K=3 × 128 atoms × 3 dims exceeds UINT32_MAX elements.
    std::size_t const total_conformers = static_cast<std::size_t>(num_smiles) * num_conformers;
    std::vector<__half> coords(total_conformers * coordinate_stride_atoms_k * 3);
    std::vector<float> energies(total_conformers);
    std::vector<sfapi::conformer_status_t> status(total_conformers);

    // All GPUs — an empty device span.
    sfapi::context_t gpu_context;

    // Allow single-GPU runs via `--multi-gpu-only`, which scaling sweeps use when they control the
    // GPU count through `CUDA_VISIBLE_DEVICES`. The flag says "multi" but the path runs correctly
    // on one GPU.
    if (gpu_context.device_ids.empty()) {
        fmt::println("  Skipped (no GPU available)");
        return;
    }
    fmt::println("  Using {} GPUs: [{}]", gpu_context.device_ids.size(), fmt::join(gpu_context.device_ids, ", "));

    auto const start_time = clock_type_t::now();
    [[maybe_unused]] auto const batch_status = gpu_context.generate_conformers(
        smiles_views, generation_config, coordinate_stride_atoms_k, coords, energies, status);
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

#pragma endregion Benchmarked Systems

#pragma region Quality Gate

/** Verdict of the 100-128 atom energy gate, including the case where too few samples reached it. */
enum class gate_verdict_t {
    /** The heavy bin was scored and cleared every threshold. */
    passed_k,
    /** The heavy bin was scored and breached a threshold. */
    failed_k,
    /** Too few conformers reached the heavy bin for a percentile to mean anything, so no verdict was formed. */
    not_evaluated_k,
};

static gate_verdict_t cross_validate(std::vector<std::string> const &smiles, std::vector<__half> const &coords,
                                     std::vector<float> const &energies,
                                     std::vector<sfapi::conformer_status_t> const &status, std::uint32_t num_conformers,
                                     std::uint32_t max_samples) {
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    sfapi::conformer_slot_layout_t const slot_layout {coordinate_stride_atoms_k, num_conformers};
    fmt::println("\nQuality cross-validation");

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
            if (!rdkit_molecule) continue; // the corpus row does not parse
            // The coordinates written below are SmallField's, so the conformer only has to exist at the
            // right size. Embedding one to obtain the slot spends an ETKDG run per sample and drops the
            // molecules ETKDG refuses, which are disproportionately the large ones this gate judges.
            sftest::attach_empty_conformer(*rdkit_molecule);

            std::uint32_t const num_atoms = rdkit_molecule->getNumAtoms();
            std::uint32_t const flat_index = molecule_index * num_conformers + conformer_index;
            // The pool is indexed in `std::size_t`: a 10 M molecule run at K=3 passes UINT32_MAX
            // elements around conformer 11.2 M, and the wrapped offset reads another molecule.
            std::size_t const coordinate_offset = slot_layout.slot_offset(molecule_index, conformer_index);

            std::vector<float> interleaved_coordinates(static_cast<std::size_t>(num_atoms) * 3u);
            sfapi::gather_slot_into_interleaved(coords.data() + coordinate_offset, slot_layout, num_atoms,
                                                interleaved_coordinates.data());
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

    if (finite_rdkit_energies.empty()) {
        fmt::println("NOTE heavy_bin_gate atoms=100-128 compared=0 required={}", min_heavy_bin_samples_k);
        return gate_verdict_t::not_evaluated_k;
    }

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
                 percentile(paired_delta, 75), percentile(paired_delta, 95), paired_delta.front(), paired_delta.back());

    // Marginals are retained for context only; they do not measure agreement.
    std::sort(finite_rdkit_energies.begin(), finite_rdkit_energies.end());
    std::sort(finite_smallfield_energies.begin(), finite_smallfield_energies.end());

    fmt::println("  RDKit MMFF94 energy (kcal/mol):");
    fmt::println("    P5={:.1f}  P25={:.1f}  P50={:.1f}  P75={:.1f}  P95={:.1f}", percentile(finite_rdkit_energies, 5),
                 percentile(finite_rdkit_energies, 25), percentile(finite_rdkit_energies, 50),
                 percentile(finite_rdkit_energies, 75), percentile(finite_rdkit_energies, 95));
    fmt::println("  SmallField energy (kcal/mol):");
    fmt::println("    P5={:.1f}  P25={:.1f}  P50={:.1f}  P75={:.1f}  P95={:.1f}",
                 percentile(finite_smallfield_energies, 5), percentile(finite_smallfield_energies, 25),
                 percentile(finite_smallfield_energies, 50), percentile(finite_smallfield_energies, 75),
                 percentile(finite_smallfield_energies, 95));

    // Bucket every sample once, then report. Scanning the samples again per bin is quadratic in a
    // job that is linear, and the outer loop hid the gather it wrapped.
    std::array<std::vector<double>, num_atom_count_bins_k> bin_rdkit_energies;
    std::array<std::vector<double>, num_atom_count_bins_k> bin_smallfield_energies;
    for (std::size_t sample_index = 0; sample_index < successful_conformers.size(); ++sample_index) {
        if (!std::isfinite(rdkit_reference_energies[sample_index])) continue; // RDKit never scored this geometry
        std::size_t const bin = atom_count_bin_of_(atom_counts[sample_index]);
        if (bin == num_atom_count_bins_k) continue; // atom count outside the tabulated bins
        bin_rdkit_energies[bin].push_back(rdkit_reference_energies[sample_index]);
        bin_smallfield_energies[bin].push_back(smallfield_energies[sample_index]);
    }

    // Print the per-atom-count breakdown. `delta_min` surfaces the negative tail
    // that a shifted median hides — the MMFF cubic bend term is unbounded below
    // past Δθ > 143°, and only the minimum exposes a conformer that fell into it.
    fmt::println("  Per-atom-count breakdown:");
    fmt::println("    {:>7}  {:>8}  {:>10}  {:>10}  {:>10}", "atoms", "count", "delta_P50", "delta_P95", "delta_min");
    std::vector<double> bin_delta;
    std::vector<double> heavy_atom_rdkit_energies, heavy_atom_delta;
    for (std::size_t bin = 0; bin < num_atom_count_bins_k; ++bin) {
        if (bin_rdkit_energies[bin].empty()) continue; // no sample landed in this bin
        std::uint32_t const lower_bound = atom_count_bin_lower_bound_(bin);
        std::uint32_t const upper_bound = atom_count_bin_bounds_k[bin];
        paired_energy_delta_into(bin_rdkit_energies[bin], bin_smallfield_energies[bin], bin_delta);
        fmt::println("    {:>3}-{:<3}  {:>8}  {:>+10.3g}  {:>+10.3g}  {:>+10.3g}", lower_bound, upper_bound,
                     fmt::group_digits(bin_delta.size()), percentile(bin_delta, 50), percentile(bin_delta, 95),
                     bin_delta.front());
        if (lower_bound == 100 && upper_bound == 128) {
            std::sort(bin_rdkit_energies[bin].begin(), bin_rdkit_energies[bin].end());
            heavy_atom_rdkit_energies = bin_rdkit_energies[bin];
            heavy_atom_delta = bin_delta;
        }
    }

    if (heavy_atom_rdkit_energies.size() < min_heavy_bin_samples_k) {
        fmt::println("NOTE heavy_bin_gate atoms=100-128 compared={} required={}", heavy_atom_rdkit_energies.size(),
                     min_heavy_bin_samples_k);
        return gate_verdict_t::not_evaluated_k;
    }

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
    if (failed_threshold == nullptr) {
        fmt::println("PASS heavy_bin_gate atoms=100-128 compared={} P50={} P95={} delta_P50={}",
                     heavy_atom_rdkit_energies.size(), heavy_p50, heavy_p95, heavy_delta_p50);
        return gate_verdict_t::passed_k;
    }
    fmt::println("FAIL heavy_bin_gate: {} atoms=100-128 compared={} P50={} P95={} delta_P50={}", failed_threshold,
                 heavy_atom_rdkit_energies.size(), heavy_p50, heavy_p95, heavy_delta_p50);
    return gate_verdict_t::failed_k;
}

#pragma endregion Quality Gate

#pragma region Command Line

/** Which systems a run measures. */
enum class bench_scope_t {
    /** RDKit CPU, one GPU, and the quality gate, swept over batch sizes. */
    cpu_and_single_gpu_k,
    /** Every visible GPU, once, in a process that ran nothing else. */
    multi_gpu_only_k,
};

/** Everything this driver takes from the command line. */
struct backend_bench_arguments_t {
    /** Corpus to benchmark over, from `--smi` or from `SMALLFIELD_TEST_CORPUS`. There is no default. */
    std::string_view smiles_path;
    /** Conformers requested per molecule. */
    std::uint32_t num_conformers = 3;
    /** Embedding attempts RDKit is allowed per molecule, matched by the GPU config. */
    std::uint32_t max_stereo_attempts = 10;
    /** Ceiling on conformers drawn for the RDKit cross-validation. */
    std::uint32_t quality_samples = 10000;
    /** Which systems to measure. */
    bench_scope_t scope = bench_scope_t::cpu_and_single_gpu_k;
};

/** Whether the command line was accepted or refused. */
enum class arguments_parse_t { accepted_k, refused_k };

/** Print the one usage line, naming every flag this driver understands. */
static void print_usage_(char const *invocation) {
    fmt::println(stderr, "usage: {} --smi PATH [--conformers N] [--quality-samples N] [--multi-gpu-only]", invocation);
}

/**
 *  @brief Read the value that follows @p flag into @p number, or explain on `stderr` why it could not be.
 *
 *  Three things are refusals rather than silent defaults: a token that is not a number, a token outside
 *  `[1, maximum]`, and a value-taking flag that appears last with nothing after it.
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

/**
 *  @brief Fill @p arguments from `argv`, or explain on `stderr` why the command line was refused.
 *
 *  Every branch either consumes its token or refuses the whole command line, so a run under an unrecognised
 *  flag or a flag with no value never reports the defaults as though they were what was asked for.
 */
static arguments_parse_t parse_arguments_(int argc, char **argv, backend_bench_arguments_t &arguments) {
    for (int argument_index = 1; argument_index < argc; ++argument_index) {
        std::string_view const flag = argv[argument_index];
        if (flag == "--smi") {
            if (argument_index + 1 >= argc) {
                fmt::println(stderr, "{}: --smi takes a value, and none followed it", executable_name_k);
                return arguments_parse_t::refused_k;
            }
            arguments.smiles_path = argv[++argument_index];
        }
        else if (flag == "--conformers") {
            if (!take_number_argument_(flag, argc, argv, argument_index, max_conformers_per_molecule_k,
                                       arguments.num_conformers))
                return arguments_parse_t::refused_k;
        }
        else if (flag == "--quality-samples") {
            if (!take_number_argument_(flag, argc, argv, argument_index, max_quality_samples_k,
                                       arguments.quality_samples))
                return arguments_parse_t::refused_k;
        }
        else if (flag == "--multi-gpu-only") { arguments.scope = bench_scope_t::multi_gpu_only_k; }
        else {
            fmt::println(stderr, "{}: unknown argument '{}'", executable_name_k, flag);
            print_usage_(argv[0]);
            return arguments_parse_t::refused_k;
        }
    }
    if (arguments.smiles_path.empty()) arguments.smiles_path = sfbench::corpus_path_from_environment();
    return arguments_parse_t::accepted_k;
}

#pragma endregion Command Line

/**
 *  @brief Run both systems at every batch size the corpus supports, returning the quality gate's verdict.
 *
 *  The sweep is deliberately non-short-circuiting: a gate that fails on one rung still leaves the remaining
 *  rungs' throughput to report, and the verdict keeps the failure.
 */
static gate_verdict_t run_scaling_sweep_(std::vector<std::string> const &all_smiles,
                                         backend_bench_arguments_t const &arguments) {
    // Rungs the corpus can fill, strictly increasing, with the whole corpus always last. Selecting a fixed
    // ladder and skipping the rungs that overshoot leaves two corpus sizes measuring nothing: one under 100
    // runs no rung at all, and one between 1,000 and 5,000 runs its rungs but never reaches the gate, because
    // the gate fired on a rung the corpus was too small to hold. Both exited clean having compared nothing.
    std::vector<std::uint32_t> batch_sizes;
    for (std::uint32_t rung : {100u, 500u, 1000u, 2000u, 5000u})
        if (rung < all_smiles.size()) batch_sizes.push_back(rung);
    batch_sizes.push_back(static_cast<std::uint32_t>(all_smiles.size()));
    fmt::println("Batch sizes: [{}]", fmt::join(batch_sizes, ", "));

    gate_verdict_t verdict = gate_verdict_t::not_evaluated_k;
    for (auto batch_size : batch_sizes) {
        std::vector<std::string> smiles(all_smiles.begin(), all_smiles.begin() + batch_size);
        fmt::println("\n\nbatch_size={}", batch_size);

        // Run system A: RDKit CPU.
        bench_rdkit_cpu(smiles, arguments.num_conformers, arguments.max_stereo_attempts);

        // Run system B: SmallField single GPU.
        std::vector<__half> coords;
        std::vector<float> energies;
        std::vector<sfapi::conformer_status_t> status;
        bench_smallfield_gpu(smiles, arguments.num_conformers, arguments.max_stereo_attempts, coords, energies, status);

        // Multi-GPU scaling is measured through a dedicated fresh process via `--multi-gpu-only`. Keeping the
        // mixed CPU + 1-GPU default path in a single process avoids state bleed between the 1-GPU and 8-GPU
        // runs on the largest batches.
        if (batch_size == batch_sizes.back())
            fmt::println("\nSystem B: SmallField multi-GPU skipped in mixed mode. "       //
                         "Re-run with --multi-gpu-only for isolated multi-GPU numbers."); //

        // Cross-validate on the whole corpus only, which the last rung now always is.
        if (batch_size == batch_sizes.back()) {
            gate_verdict_t const rung_verdict = cross_validate(smiles, coords, energies, status,
                                                               arguments.num_conformers, arguments.quality_samples);
            // A failure anywhere in the sweep is the answer; a later pass does not clear it.
            if (verdict != gate_verdict_t::failed_k) verdict = rung_verdict;
        }
    }
    return verdict;
}

int main(int argc, char **argv) {
    backend_bench_arguments_t arguments;
    if (parse_arguments_(argc, argv, arguments) != arguments_parse_t::accepted_k) return 2;

    fmt::println("{}: SmallField GPU conformer generation against the RDKit CPU baseline", executable_name_k);
    fmt::println("Dataset:     {}", arguments.smiles_path);
    fmt::println("Conformers:  {} per molecule", arguments.num_conformers);
    fmt::println("CPU threads: {}", omp_get_max_threads());

    // A corpus that never opened must not read as an empty corpus: a validator over zero molecules
    // compares nothing and still exits clean.
    std::vector<std::string> all_smiles;
    auto const load_status = sfbench::load_corpus_into(arguments.smiles_path, std::numeric_limits<std::size_t>::max(),
                                                       all_smiles);
    if (load_status != sfbench::corpus_load_t::loaded_k) {
        fmt::println(stderr, "{}: corpus '{}' {}", executable_name_k, arguments.smiles_path,
                     sfbench::describe_corpus_load(load_status));
        print_usage_(argv[0]);
        return 2;
    }
    fmt::println("Loaded:      {} SMILES", fmt::group_digits(all_smiles.size()));

    if (arguments.scope == bench_scope_t::multi_gpu_only_k) {
        bench_smallfield_multi_gpu(all_smiles, arguments.num_conformers, arguments.max_stereo_attempts);
        fmt::println("\nDone.");
        return 0;
    }

    gate_verdict_t const verdict = run_scaling_sweep_(all_smiles, arguments);
    fmt::println("\nDone.");
    return verdict == gate_verdict_t::failed_k ? 1 : 0;
}
