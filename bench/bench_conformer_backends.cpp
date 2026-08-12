/**
 *  @file bench/bench_conformer_backends.cpp
 *  @brief RDKit CPU against SmallField GPU over one SMILES corpus, swept across batch sizes, with the
 *  SmallField geometries cross-validated by RDKit's own MMFF94.
 *
 *  Exit code `1` is reserved for the quality gate failing, so a throughput regression and a correctness
 *  regression do not read alike.
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
#include "bench/bench_reporting.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfapi = smallfield::api;
namespace sfbench = smallfield::bench;
namespace sftest = smallfield::tests;
using clock_type_t = std::chrono::high_resolution_clock;

/** Name printed on every throughput line so a concatenated CI log stays attributable. */
static constexpr char const *executable_name_k = "bench_conformer_backends";

/* Ceilings on the value-taking flags, so a mistyped count is a diagnostic and not a `std::bad_alloc`. */
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
 *  The generator is handed this and the cross-validator reads back with it, so both must name the same
 *  constant: reading a pool at the wrong stride silently returns a neighbouring conformer's coordinates.
 */
static constexpr std::uint32_t coordinate_stride_atoms_k = smallfield::types::max_atoms_per_molecule_k;

#pragma region Statistics

/** Linear-interpolated percentile of an @b already sorted sample; zero for an empty one. */
static double percentile(custd::span<double const> sorted_values, double percent) {
    if (sorted_values.empty()) return 0.0;
    double const index = percent / 100.0 * static_cast<double>(sorted_values.size() - 1);
    std::size_t const lower_index = static_cast<std::size_t>(index);
    std::size_t const upper_index = std::min(lower_index + 1, sorted_values.size() - 1);
    double const fraction = index - static_cast<double>(lower_index);
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction;
}

/**
 *  @brief Upper bounds of the reported bins; bin `bin` covers atom counts in `(lower, bounds[bin]]`.
 *
 *  The last bound is the library's own atom cap rather than a number of its own, because a table stopping
 *  short of it drops every molecule above the shortfall out of the breakdown with nothing said.
 */
static constexpr std::uint32_t atom_count_bin_bounds_k[] = {20, 40, 60, 80, 100, 128, coordinate_stride_atoms_k};
static constexpr std::size_t num_atom_count_bins_k = std::size(atom_count_bin_bounds_k);
static_assert(atom_count_bin_bounds_k[num_atom_count_bins_k - 1] == coordinate_stride_atoms_k,
              "the last bin reaches the atom cap");

/** Exclusive lower bound of bin @p bin. */
static constexpr std::uint32_t atom_count_bin_lower_bound_(std::size_t bin) {
    return bin == 0 ? 0u : atom_count_bin_bounds_k[bin - 1];
}

/**
 *  @brief The bin the gate renders its verdict on, and the thresholds it renders it against.
 *
 *  100-128 atoms, which is where the distance-geometry embedding and the MMFF relaxation are both under the
 *  most strain and where a regression shows first. The bin above it is reported but not gated: no threshold
 *  has been calibrated against a 129-192 atom sample, and a gate on an uncalibrated threshold is a coin toss
 *  dressed as a check.
 */
static constexpr std::size_t gate_bin_k = 5;
static_assert(atom_count_bin_bounds_k[gate_bin_k - 1] == 100 && atom_count_bin_bounds_k[gate_bin_k] == 128,
              "the gate's thresholds were measured on the 100-128 atom bin");

/** Index of the bin covering @p num_atoms, or `num_atom_count_bins_k` when no bin does, as in `std::find`. */
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
            params.maxIterations = static_cast<unsigned int>(max_stereo_attempts);
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

/** The throughput block both SmallField systems print, and the rejection breakdown under it. */
static void report_generation_throughput_(std::uint32_t num_smiles, std::size_t total_conformers,
                                          double elapsed_milliseconds, sfbench::status_histogram_t const &histogram) {
    std::uint64_t const succeeded = sfbench::successful_conformers_of(histogram);
    fmt::println("  Time:      {:.1f} ms", elapsed_milliseconds);
    fmt::println("  {} mol/s:  {:.1f}", executable_name_k, num_smiles / (elapsed_milliseconds / 1000.0));
    fmt::println("  {} conf/s: {:.1f}", executable_name_k,
                 static_cast<double>(succeeded) / (elapsed_milliseconds / 1000.0));
    fmt::println("  Succeeded: {} / {} conformers", fmt::group_digits(succeeded), fmt::group_digits(total_conformers));
    sfbench::print_status_histogram(histogram, total_conformers);
}

static void bench_smallfield_gpu(std::vector<std::string> const &smiles, std::uint32_t num_conformers,
                                 std::uint32_t max_stereo_attempts, std::vector<__half> &coords_out,
                                 std::vector<float> &energies_out, std::vector<sfapi::conformer_status_t> &status_out) {
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    fmt::println("\nSystem B: SmallField GPU");

    std::vector<std::string_view> smiles_views(smiles.begin(), smiles.end());

    sfapi::config_t generation_config;
    generation_config.num_conformers = num_conformers;
    generation_config.max_stereo_attempts = max_stereo_attempts;

    // `std::size_t` throughout: at 10 M molecules and three conformers the coordinate buffer passes
    // `UINT32_MAX` elements, and a 32-bit offset would wrap into another molecule's coordinates.
    std::size_t const total_conformers = static_cast<std::size_t>(num_smiles) * num_conformers;
    coords_out.resize(total_conformers * coordinate_stride_atoms_k * 3);
    energies_out.resize(total_conformers);
    status_out.resize(total_conformers);

    int const device_id = 0;
    sfapi::context_t gpu_context(custd::span<int const>(&device_id, 1));

    auto const start_time = clock_type_t::now();
    [[maybe_unused]] auto const batch_status = gpu_context.generate_conformers(
        smiles_views, generation_config, coordinate_stride_atoms_k, coords_out, energies_out, status_out);
    auto const end_time = clock_type_t::now();

    double const elapsed_ms_value = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    auto const histogram = sfbench::status_histogram_of(status_out);
    report_generation_throughput_(num_smiles, total_conformers, elapsed_ms_value, histogram);
    std::uint64_t const cuda_errors = gpu_context.cuda_error_count.load();
    if (cuda_errors > 0)
        fmt::println(stderr, "  cuda errors during single-GPU bench: {}", fmt::group_digits(cuda_errors));
}

static void bench_smallfield_multi_gpu(std::vector<std::string> const &smiles, std::uint32_t num_conformers,
                                       std::uint32_t max_stereo_attempts) {
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    fmt::println("\nSystem B: SmallField multi-GPU, all visible devices");

    std::vector<std::string_view> smiles_views(smiles.begin(), smiles.end());

    sfapi::config_t generation_config;
    generation_config.num_conformers = num_conformers;
    generation_config.max_stereo_attempts = max_stereo_attempts;

    std::size_t const total_conformers = static_cast<std::size_t>(num_smiles) * num_conformers;
    std::vector<__half> coords(total_conformers * coordinate_stride_atoms_k * 3);
    std::vector<float> energies(total_conformers);
    std::vector<sfapi::conformer_status_t> status(total_conformers);

    // An empty device span is every visible GPU, so a scaling sweep controlling the count through
    // `CUDA_VISIBLE_DEVICES` reaches this path with one device and still runs.
    sfapi::context_t gpu_context;

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

    report_generation_throughput_(num_smiles, total_conformers, elapsed_ms_value, sfbench::status_histogram_of(status));
    std::uint64_t const cuda_errors = gpu_context.cuda_error_count.load();
    if (cuda_errors > 0)
        fmt::println(stderr, "  cuda errors during multi-GPU bench: {}", fmt::group_digits(cuda_errors));
}

#pragma endregion Benchmarked Systems

#pragma region Quality Gate

/** Verdict of the 100-128 atom energy gate, including the case where too few samples reached it. */
enum class gate_verdict_t {
    passed_k,
    failed_k,
    /** Too few conformers reached the heavy bin for a percentile to mean anything. */
    not_evaluated_k,
};

/** One conformer drawn for cross-validation: where it came from, and what both engines scored it. */
struct sampled_conformer_t {
    std::uint32_t molecule_index = 0;
    std::uint32_t conformer_index = 0;
    std::uint32_t num_atoms = 0;
    /** Non-finite @b is the answer "RDKit never scored this geometry", so no validity flag can disagree. */
    double rdkit_energy = std::numeric_limits<double>::quiet_NaN();
    double smallfield_energy = 0.0;

    [[nodiscard]] bool scored() const noexcept { return std::isfinite(rdkit_energy); }
    /** SmallField minus RDKit on the same geometry, which is the quantity that measures agreement. */
    [[nodiscard]] double paired_delta() const noexcept { return smallfield_energy - rdkit_energy; }
};

/**
 *  @brief Draw up to @p max_samples successful conformer slots into @p sample, which is cleared first.
 *
 *  A shuffle rather than a prefix, because slot order follows corpus order and a corpus is ordered by
 *  deposition, so a prefix samples one neighbourhood of it.
 */
static void fill_success_sample_into_(custd::span<sfapi::conformer_status_t const> status,
                                      sfapi::conformer_slot_layout_t const &slot_layout, std::uint32_t max_samples,
                                      std::vector<sampled_conformer_t> &sample) {
    sample.clear();
    std::uint32_t const num_molecules = static_cast<std::uint32_t>(status.size() / slot_layout.num_conformers);
    for (std::uint32_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index)
        for (std::uint32_t conformer_index = 0; conformer_index < slot_layout.num_conformers; ++conformer_index)
            if (status[slot_layout.flat_slot(molecule_index, conformer_index)] == sfapi::conformer_status_t::success_k)
                sample.push_back({.molecule_index = molecule_index, .conformer_index = conformer_index});

    if (sample.size() <= max_samples) return;
    std::mt19937 random_engine(42);
    std::shuffle(sample.begin(), sample.end(), random_engine);
    sample.resize(max_samples);
}

/** Score every drawn conformer with RDKit's own MMFF94, on SmallField's coordinates, in place. */
static void score_sample_with_rdkit_(std::vector<std::string> const &smiles, std::vector<__half> const &coords,
                                     std::vector<float> const &energies,
                                     sfapi::conformer_slot_layout_t const &slot_layout,
                                     custd::span<sampled_conformer_t> sample) {
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t sample_index = 0; sample_index < sample.size(); ++sample_index) {
        sampled_conformer_t &drawn = sample[sample_index];
        try {
            auto rdkit_molecule = sftest::mol_from_smiles(smiles[drawn.molecule_index]);
            if (!rdkit_molecule) continue; // the corpus row does not parse, so the sample stays unscored
            // The coordinates written below are SmallField's, so the conformer only has to exist at the
            // right size. Embedding one to obtain the slot spends an ETKDG run per sample and drops the
            // molecules ETKDG refuses, which are disproportionately the large ones this gate judges.
            sftest::attach_empty_conformer(*rdkit_molecule);

            std::uint32_t const num_atoms = rdkit_molecule->getNumAtoms();
            std::size_t const coordinate_offset = slot_layout.slot_offset(drawn.molecule_index, drawn.conformer_index);

            std::vector<float> interleaved_coordinates(static_cast<std::size_t>(num_atoms) * 3u);
            sfapi::gather_slot_into_interleaved(coords.data() + coordinate_offset, slot_layout, num_atoms,
                                                interleaved_coordinates.data());
            sftest::set_coords_interleaved(*rdkit_molecule, interleaved_coordinates.data());
            drawn.num_atoms = num_atoms;
            drawn.smallfield_energy = static_cast<double>(
                energies[slot_layout.flat_slot(drawn.molecule_index, drawn.conformer_index)]);
            // Written last: a finite value here is what marks the sample scored, so it must not be set
            // before the two fields a reader will pair it with.
            drawn.rdkit_energy = sftest::rdkit_mmff94_energy(*rdkit_molecule);
        }
        catch (...) {
            // An unscored sample keeps its NaN and is excluded from every statistic below.
        }
    }
}

/**
 *  @brief Report the paired delta and both marginal distributions over every scored sample.
 *
 *  Both columns describe the same geometry, since RDKit is evaluated on SmallField's own coordinates, so
 *  the per-conformer difference is what measures agreement. Independently sorted marginals do not: a
 *  uniform offset, a heavy negative tail, and partial cancellation all shift them alike.
 *
 *  @return Samples RDKit actually scored, which every `compared=` count downstream is measured against.
 */
static std::size_t report_energy_distributions_(custd::span<sampled_conformer_t const> sample,
                                                std::vector<double> &scratch) {
    scratch.clear();
    for (sampled_conformer_t const &drawn : sample)
        if (drawn.scored()) scratch.push_back(drawn.paired_delta());
    std::size_t const num_scored = scratch.size();
    fmt::println("  Finite:    {} / {} ({:.3f} %)", fmt::group_digits(num_scored), fmt::group_digits(sample.size()),
                 100.0 * static_cast<double>(num_scored) / static_cast<double>(sample.size()));
    if (num_scored == 0) return 0;
    std::sort(scratch.begin(), scratch.end());

    fmt::println("  SmallField − RDKit, paired per conformer (kcal/mol):");
    fmt::println("    P5={:+.3g}  P25={:+.3g}  P50={:+.3g}  P75={:+.3g}  P95={:+.3g}  min={:+.3g}  max={:+.3g}",
                 percentile(scratch, 5), percentile(scratch, 25), percentile(scratch, 50), percentile(scratch, 75),
                 percentile(scratch, 95), scratch.front(), scratch.back());

    // Marginals are retained for context only; they do not measure agreement.
    auto report_marginal_ = [&](char const *label, auto energy_of) {
        scratch.clear();
        for (sampled_conformer_t const &drawn : sample)
            if (drawn.scored()) scratch.push_back(energy_of(drawn));
        std::sort(scratch.begin(), scratch.end());
        fmt::println("  {} energy (kcal/mol):", label);
        fmt::println("    P5={:.1f}  P25={:.1f}  P50={:.1f}  P75={:.1f}  P95={:.1f}", percentile(scratch, 5),
                     percentile(scratch, 25), percentile(scratch, 50), percentile(scratch, 75),
                     percentile(scratch, 95));
    };
    report_marginal_("RDKit MMFF94", [](sampled_conformer_t const &drawn) { return drawn.rdkit_energy; });
    report_marginal_("SmallField", [](sampled_conformer_t const &drawn) { return drawn.smallfield_energy; });
    return num_scored;
}

/** The scored samples of one atom-count bin, sorted along each axis the gate reads. */
struct bin_sample_t {
    std::vector<double> rdkit_energies;
    std::vector<double> paired_deltas;
};

/**
 *  @brief Bucket every scored sample by atom count once, then print the breakdown.
 *
 *  `delta_min` surfaces the negative tail a shifted median hides: the MMFF cubic bend term is unbounded
 *  below past a 143° bend, and only the minimum exposes a conformer that fell into it.
 */
static void report_atom_count_bins_(custd::span<sampled_conformer_t const> sample,
                                    std::array<bin_sample_t, num_atom_count_bins_k> &bins) {
    for (bin_sample_t &bin : bins) {
        bin.rdkit_energies.clear();
        bin.paired_deltas.clear();
    }
    for (sampled_conformer_t const &drawn : sample) {
        if (!drawn.scored()) continue; // RDKit never scored this geometry
        std::size_t const bin = atom_count_bin_of_(drawn.num_atoms);
        if (bin == num_atom_count_bins_k) continue; // atom count outside the tabulated bins
        bins[bin].rdkit_energies.push_back(drawn.rdkit_energy);
        bins[bin].paired_deltas.push_back(drawn.paired_delta());
    }

    fmt::println("  Per-atom-count breakdown:");
    fmt::println("    {:>7}  {:>8}  {:>10}  {:>10}  {:>10}", "atoms", "count", "delta_P50", "delta_P95", "delta_min");
    for (std::size_t bin = 0; bin < num_atom_count_bins_k; ++bin) {
        std::sort(bins[bin].rdkit_energies.begin(), bins[bin].rdkit_energies.end());
        std::sort(bins[bin].paired_deltas.begin(), bins[bin].paired_deltas.end());
        std::vector<double> const &deltas = bins[bin].paired_deltas;
        if (deltas.empty()) continue; // no sample landed in this bin
        fmt::println("    {:>3}-{:<3}  {:>8}  {:>+10.3g}  {:>+10.3g}  {:>+10.3g}", atom_count_bin_lower_bound_(bin),
                     atom_count_bin_bounds_k[bin], fmt::group_digits(deltas.size()), percentile(deltas, 50),
                     percentile(deltas, 95), deltas.front());
    }
}

/**
 *  @brief Render the verdict on the heavy bin, or decline to when too few samples reached it.
 *
 *  The absolute bounds catch a blown-up geometry; the paired median catches a force-field divergence that
 *  leaves the absolute energies looking plausible. The three deciding numbers print at `{}`, shortest
 *  round-trip, so a value that only just cleared its threshold can never render as one that just missed it.
 */
static gate_verdict_t judge_heavy_bin_(bin_sample_t const &heavy_bin) {
    std::uint32_t const lower = atom_count_bin_lower_bound_(gate_bin_k);
    std::uint32_t const upper = atom_count_bin_bounds_k[gate_bin_k];
    if (heavy_bin.rdkit_energies.size() < min_heavy_bin_samples_k) {
        fmt::println("NOTE heavy_bin_gate atoms={}-{} compared={} required={}", lower, upper,
                     heavy_bin.rdkit_energies.size(), min_heavy_bin_samples_k);
        return gate_verdict_t::not_evaluated_k;
    }

    double const heavy_p50 = percentile(heavy_bin.rdkit_energies, 50);
    double const heavy_p95 = percentile(heavy_bin.rdkit_energies, 95);
    double const heavy_delta_p50 = percentile(heavy_bin.paired_deltas, 50);
    char const *failed_threshold = nullptr;
    if (heavy_p50 >= 180.0) failed_threshold = "P50 >= 180";
    else if (heavy_p95 >= 300.0) failed_threshold = "P95 >= 300";
    else if (std::fabs(heavy_delta_p50) >= 10.0) failed_threshold = "|delta_P50| >= 10";

    if (failed_threshold == nullptr) {
        fmt::println("PASS heavy_bin_gate atoms={}-{} compared={} P50={} P95={} delta_P50={}", lower, upper,
                     heavy_bin.rdkit_energies.size(), heavy_p50, heavy_p95, heavy_delta_p50);
        return gate_verdict_t::passed_k;
    }
    fmt::println("FAIL heavy_bin_gate: {} atoms={}-{} compared={} P50={} P95={} delta_P50={}", failed_threshold, lower,
                 upper, heavy_bin.rdkit_energies.size(), heavy_p50, heavy_p95, heavy_delta_p50);
    return gate_verdict_t::failed_k;
}

static gate_verdict_t cross_validate(std::vector<std::string> const &smiles, std::vector<__half> const &coords,
                                     std::vector<float> const &energies,
                                     std::vector<sfapi::conformer_status_t> const &status, std::uint32_t num_conformers,
                                     std::uint32_t max_samples) {
    sfapi::conformer_slot_layout_t const slot_layout {coordinate_stride_atoms_k, num_conformers};
    fmt::println("\nQuality cross-validation");

    std::vector<sampled_conformer_t> sample;
    fill_success_sample_into_(status, slot_layout, max_samples, sample);
    fmt::println("  Evaluating {} conformers via RDKit MMFF94...", fmt::group_digits(sample.size()));
    if (sample.empty()) {
        fmt::println("NOTE heavy_bin_gate atoms={}-{} compared=0 required={}", atom_count_bin_lower_bound_(gate_bin_k),
                     atom_count_bin_bounds_k[gate_bin_k], min_heavy_bin_samples_k);
        return gate_verdict_t::not_evaluated_k;
    }
    score_sample_with_rdkit_(smiles, coords, energies, slot_layout, sample);

    std::vector<double> scratch;
    scratch.reserve(sample.size());
    if (report_energy_distributions_(sample, scratch) == 0) {
        fmt::println("NOTE heavy_bin_gate atoms={}-{} compared=0 required={}", atom_count_bin_lower_bound_(gate_bin_k),
                     atom_count_bin_bounds_k[gate_bin_k], min_heavy_bin_samples_k);
        return gate_verdict_t::not_evaluated_k;
    }

    std::array<bin_sample_t, num_atom_count_bins_k> bins;
    report_atom_count_bins_(sample, bins);
    return judge_heavy_bin_(bins[gate_bin_k]);
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
    /** From `--smi` or from `SMALLFIELD_TEST_CORPUS`; there is no default. */
    std::string_view smiles_path;
    std::uint32_t num_conformers = 3;
    /** Embedding attempts RDKit is allowed per molecule, matched by the GPU config. */
    std::uint32_t max_stereo_attempts = 10;
    /** Ceiling on conformers drawn for the RDKit cross-validation. */
    std::uint32_t quality_samples = 10000;
    bench_scope_t scope = bench_scope_t::cpu_and_single_gpu_k;
};

using sfbench::arguments_parse_t;

static void print_usage_(char const *invocation) {
    fmt::println(stderr, "usage: {} --smi PATH [--conformers N] [--quality-samples N] [--multi-gpu-only]", invocation);
}

/** Fill @p arguments from `argv`, or explain on `stderr` why the command line was refused. */
static arguments_parse_t parse_arguments_(int argc, char **argv, backend_bench_arguments_t &arguments) {
    for (sfbench::command_line_t line {executable_name_k, argc, argv}; line.advance();) {
        arguments_parse_t status = arguments_parse_t::accepted_k;
        if (line.flag() == "--smi") status = line.take_text(arguments.smiles_path);
        else if (line.flag() == "--conformers")
            status = line.take_number(max_conformers_per_molecule_k, arguments.num_conformers);
        else if (line.flag() == "--quality-samples")
            status = line.take_number(max_quality_samples_k, arguments.quality_samples);
        else if (line.flag() == "--multi-gpu-only") arguments.scope = bench_scope_t::multi_gpu_only_k;
        else {
            print_usage_(argv[0]);
            status = line.refuse_unknown();
        }
        if (status != arguments_parse_t::accepted_k) return arguments_parse_t::refused_k;
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
    // Rungs the corpus can fill, strictly increasing, with the whole corpus always last — which is what
    // guarantees a corpus of any size runs at least one rung and reaches the gate.
    std::vector<std::uint32_t> batch_sizes;
    for (std::uint32_t rung : {100u, 500u, 1000u, 2000u, 5000u})
        if (rung < all_smiles.size()) batch_sizes.push_back(rung);
    batch_sizes.push_back(static_cast<std::uint32_t>(all_smiles.size()));
    fmt::println("Batch sizes: [{}]", fmt::join(batch_sizes, ", "));

    gate_verdict_t verdict = gate_verdict_t::not_evaluated_k;
    for (auto batch_size : batch_sizes) {
        std::vector<std::string> smiles(all_smiles.begin(), all_smiles.begin() + batch_size);
        fmt::println("\n\nbatch_size={}", batch_size);

        bench_rdkit_cpu(smiles, arguments.num_conformers, arguments.max_stereo_attempts);

        std::vector<__half> coords;
        std::vector<float> energies;
        std::vector<sfapi::conformer_status_t> status;
        bench_smallfield_gpu(smiles, arguments.num_conformers, arguments.max_stereo_attempts, coords, energies, status);

        // Multi-GPU runs in a fresh process under `--multi-gpu-only`, so no state bleeds from the 1-GPU run
        // into the 8-GPU one on the largest batches.
        if (batch_size == batch_sizes.back())
            fmt::println("\nSystem B: SmallField multi-GPU skipped in mixed mode. "       //
                         "Re-run with --multi-gpu-only for isolated multi-GPU numbers."); //

        // Cross-validate on the whole corpus only, which the last rung always is.
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
