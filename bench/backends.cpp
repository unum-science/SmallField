/**
 *  @file bench/backends.cpp
 *  @brief RDKit CPU against SmallField GPU over one SMILES corpus, swept across batch sizes, with the
 *  SmallField geometries cross-validated by RDKit's own MMFF94.
 *
 *  Exit code `1` is the quality gate failing, so throughput and correctness regressions do not read alike.
 */

#include <cmath>   // `std::isfinite`, `std::fabs`
#include <cstdint> // `std::uint8_t`, `std::uint32_t`
#include <cstdio>  // `stderr`

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

#include "bench/arguments.hpp"
#include "bench/reporting.hpp"
#include "test/rdkit_reference.hpp"

namespace sfapi = smallfield::api;
namespace sfbench = smallfield::bench;
namespace sftest = smallfield::tests;
namespace sftypes = smallfield::types;
using clock_type_t = std::chrono::high_resolution_clock;

/** Name printed on every throughput line so a concatenated CI log stays attributable. */
static constexpr char const *executable_name_k = "smallfield_bench_backends";

/* Ceilings on the value-taking flags, so a mistyped count is a diagnostic and not a `std::bad_alloc`. */
static constexpr std::uint32_t max_conformers_per_molecule_k = 1024;
static constexpr std::uint32_t max_quality_samples_k = 10'000'000;

/** Fewest heavy-bin samples the gate will render a verdict on; ~1,500 quality samples clear it. Drug-like corpora
 *  put ~2 % of molecules in the 100-128 atom bin — 97 of 4,788 on the corpus behind `BENCH.md` — so
 *  `--quality-samples 512` reaches the gate with about ten, whose P95 is just their second-largest element. */
static constexpr std::size_t min_heavy_bin_samples_k = 30;

/** Below this an atom pair is closer than any bond MMFF94 parameterises, so the geometry is not one. A geometry
 *  check, not an energy one: it catches a readback of the wrong atoms, which energy alone only inflates. */
static constexpr double clash_distance_angstrom_k = 0.5;

/** Atom stride of every conformer slot in the coordinate pool. Generator and cross-validator must share it: a wrong
 *  stride silently reads a neighbour conformer's coordinates. */
static constexpr std::uint32_t coordinate_stride_atoms_k = smallfield::types::max_atoms_per_molecule_k;

/** What a conformer slot carries before `generate_conformers` has written it. A value-initialized
 *  `conformer_status_t` is `success_k`, so a slot the pipeline never reached would be drawn into the RDKit
 *  cross-validation sample, scored on uninitialized coordinates, and decide this binary's exit code. */
static constexpr sfapi::conformer_status_t unwritten_status_k = sfapi::conformer_status_t::invalid_smiles_k;

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

/** Upper bounds of the reported bins; bin `bin` covers atom counts in `(lower, bounds[bin]]`. The last bound is the
 *  library's atom cap; stopping short drops every heavier molecule from the breakdown silently. */
static constexpr std::uint32_t atom_count_bin_bounds_k[] = {20, 40, 60, 80, 100, 128, coordinate_stride_atoms_k};
static constexpr std::size_t num_atom_count_bins_k = std::size(atom_count_bin_bounds_k);
static_assert(atom_count_bin_bounds_k[num_atom_count_bins_k - 1] == coordinate_stride_atoms_k,
              "the last bin reaches the atom cap");

/** Exclusive lower bound of bin @p bin. */
static constexpr std::uint32_t atom_count_bin_lower_bound_(std::size_t bin) {
    return bin == 0 ? 0u : atom_count_bin_bounds_k[bin - 1];
}

/** The bin the gate renders its verdict on, and the thresholds it renders it against: 100-128 atoms, where the
 *  embedding and the MMFF relaxation are both under the most strain and a regression shows first. The bin above is
 *  reported but not gated, no threshold being calibrated against a 129-192 atom sample. */
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
                                 std::vector<float> &energies_out, std::vector<sfapi::conformer_status_t> &status_out,
                                 std::vector<std::uint8_t> &num_atoms_out) {
    std::uint32_t const num_smiles = static_cast<std::uint32_t>(smiles.size());
    fmt::println("\nSystem B: SmallField GPU");

    std::vector<std::string_view> smiles_views(smiles.begin(), smiles.end());

    sfapi::config_t generation_config;
    generation_config.num_conformers = num_conformers;
    generation_config.max_stereo_attempts = max_stereo_attempts;

    // `std::size_t` throughout: ten million molecules at three conformers each passes `UINT32_MAX`
    // elements, and a 32-bit offset wraps.
    std::size_t const total_conformers = static_cast<std::size_t>(num_smiles) * num_conformers;
    coords_out.resize(total_conformers * coordinate_stride_atoms_k * 3);
    energies_out.resize(total_conformers);
    status_out.assign(total_conformers, unwritten_status_k);
    // Zero is what `generate_conformers` writes for a refused molecule, so an unreached one reads refused, not stale.
    num_atoms_out.assign(num_smiles, std::uint8_t {0});

    int const device_id = 0;
    sfapi::context_t gpu_context(custd::span<int const>(&device_id, 1));

    auto const start_time = clock_type_t::now();
    [[maybe_unused]] auto const batch_status = gpu_context.generate_conformers(smiles_views, generation_config,
                                                                               coordinate_stride_atoms_k, coords_out,
                                                                               energies_out, status_out, num_atoms_out);
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
    std::vector<sfapi::conformer_status_t> status(total_conformers, unwritten_status_k);

    // An empty device span is every visible GPU, so a `CUDA_VISIBLE_DEVICES`-controlled sweep still runs at one.
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

/** Verdict of the 100-128 atom quality gate; `not_evaluated_k` is too few samples for a percentile to mean anything. */
enum class gate_verdict_t {
    passed_k,
    failed_k,
    not_evaluated_k,
};

/** One conformer drawn for cross-validation: where it came from, and what both engines scored it. `num_atoms` is
 *  SmallField's own post-preprocess count, naming the fragment the pipeline embedded; each field's initializer is
 *  its "unwritten" sentinel, so no validity flag can disagree with the data; and `scored()` demands both sides
 *  finite, one NaN inside the `std::sort` below being an undefined ordering over the whole sample. */
struct sampled_conformer_t {
    std::uint32_t molecule_index = 0;
    std::uint32_t conformer_index = 0;
    std::uint32_t num_atoms = 0;
    double closest_atom_pair = std::numeric_limits<double>::infinity();
    double rdkit_energy = std::numeric_limits<double>::quiet_NaN();
    double smallfield_energy = 0.0;

    [[nodiscard]] bool gathered() const noexcept { return num_atoms != 0; }
    [[nodiscard]] bool scored() const noexcept {
        return std::isfinite(rdkit_energy) && std::isfinite(smallfield_energy);
    }
    [[nodiscard]] bool clashing() const noexcept { return closest_atom_pair < clash_distance_angstrom_k; }
    [[nodiscard]] double paired_delta() const noexcept { return smallfield_energy - rdkit_energy; }
};

/** Conformers the cross-validation drew but could not compare, and why — a shrinking `compared=` says nothing about
 *  which reason moved. `unparsed` is a corpus row the reference parser refuses; `atom_count_mismatch` is a re-parse
 *  keeping every fragment where `generate_conformers` embedded only the largest, so the two counts describe
 *  different molecules — a corpus property, which must not read as a regression. */
struct sample_skips_t {
    std::size_t unparsed = 0;
    std::size_t atom_count_mismatch = 0;
};

/** Draw up to @p max_samples successful conformer slots into @p sample, which is cleared first. A shuffle, not a
 *  prefix: slot order follows the corpus's deposition order, so a prefix samples one neighbourhood. */
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

/**
 *  @brief Score every drawn conformer with RDKit's own MMFF94, on SmallField's coordinates, in place.
 *
 *  `generate_conformers` embeds only a molecule's largest fragment, so `num_atoms_per_molecule`, not the corpus
 *  string nor the reference molecule re-derived from it, says how many atoms the slot holds. At the re-parsed count a
 *  salt form gathers the counterion's atoms out of slot padding onto the parent's at the origin, and RDKit's van der
 *  Waals term answers nine orders of magnitude past any conformer energy — a different molecule, so it is skipped.
 */
static sample_skips_t score_sample_with_rdkit_(std::vector<std::string> const &smiles,
                                               std::vector<__half> const &coords, std::vector<float> const &energies,
                                               custd::span<std::uint8_t const> num_atoms_per_molecule,
                                               sfapi::conformer_slot_layout_t const &slot_layout,
                                               custd::span<sampled_conformer_t> sample) {
    std::size_t unparsed = 0, mismatched = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : unparsed, mismatched)
    for (std::size_t sample_index = 0; sample_index < sample.size(); ++sample_index) {
        sampled_conformer_t &drawn = sample[sample_index];
        try {
            auto rdkit_molecule = sftest::mol_from_smiles(smiles[drawn.molecule_index]);
            if (!rdkit_molecule) { // the corpus row does not parse, so there is no reference to compare against
                ++unparsed;
                continue;
            }
            std::uint32_t const num_atoms = num_atoms_per_molecule[drawn.molecule_index];
            if (rdkit_molecule->getNumAtoms() != num_atoms) { // gathering past our own count reads padding
                ++mismatched;
                continue;
            }
            // The coordinates written below are SmallField's, so the conformer need only exist at the right size;
            // embedding one costs an ETKDG run per sample and drops the large molecules ETKDG refuses — the gated ones.
            sftest::attach_empty_conformer(*rdkit_molecule);

            std::size_t const coordinate_offset = slot_layout.slot_offset(drawn.molecule_index, drawn.conformer_index);
            std::vector<float> interleaved_storage;
            sftypes::conformer_aos_t const geometry =
                sftypes::reset_coordinates_aos<sftypes::coordinate_role_t::position_k, float>(interleaved_storage,
                                                                                              num_atoms);
            sfapi::gather_slot_into_interleaved(coords.data() + coordinate_offset, slot_layout, num_atoms,
                                                geometry.data_handle());
            sftest::set_coords_interleaved(*rdkit_molecule, geometry.data_handle());
            drawn.closest_atom_pair = sftest::closest_atom_pair(geometry);
            drawn.smallfield_energy = static_cast<double>(
                energies[slot_layout.flat_slot(drawn.molecule_index, drawn.conformer_index)]);
            // Written after the geometry it describes: a non-zero count is what marks the slot read back.
            drawn.num_atoms = num_atoms;
            // Written last, for the same reason: a finite energy is what marks the sample scored.
            drawn.rdkit_energy = sftest::rdkit_mmff94_energy(*rdkit_molecule);
        }
        catch (...) {
            // An unscored sample keeps its NaN and is excluded from every energy statistic below.
        }
    }
    return {.unparsed = unparsed, .atom_count_mismatch = mismatched};
}

/** Report the paired delta and both marginal distributions over every scored sample. RDKit is evaluated on
 *  SmallField's own coordinates, so the per-conformer difference measures agreement, while independently sorted
 *  marginals do not: a uniform offset, a heavy negative tail and partial cancellation shift them alike. */
static void report_energy_distributions_(custd::span<sampled_conformer_t const> sample, sample_skips_t const &skips,
                                         std::vector<double> &scratch) {
    if (sample.empty()) return; // the counts the gate declines on are printed by `judge_heavy_bin_`
    std::size_t num_gathered = 0, num_clashing = 0;
    for (sampled_conformer_t const &drawn : sample) {
        if (!drawn.gathered()) continue;
        ++num_gathered;
        if (drawn.clashing()) ++num_clashing;
    }
    // Three denominators, printed apart: a draw the harness could not read back is an instrument limit, a read-back
    // one RDKit refused to type is a corpus property, a clash is a geometry defect; one percentage names none.
    auto const percent_of_gathered_ = [&](std::size_t part) {
        return num_gathered == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(num_gathered);
    };
    fmt::println("  Gathered:  {} / {} ({:.3f} %), skipped unparsed={} atom_count_mismatch={}",
                 fmt::group_digits(num_gathered), fmt::group_digits(sample.size()),
                 100.0 * static_cast<double>(num_gathered) / static_cast<double>(sample.size()),
                 fmt::group_digits(skips.unparsed), fmt::group_digits(skips.atom_count_mismatch));

    scratch.clear();
    for (sampled_conformer_t const &drawn : sample)
        if (drawn.scored()) scratch.push_back(drawn.paired_delta());
    std::size_t const num_scored = scratch.size();
    fmt::println("  Finite:    {} / {} gathered ({:.3f} %)", fmt::group_digits(num_scored),
                 fmt::group_digits(num_gathered), percent_of_gathered_(num_scored));
    fmt::println("  Clashing:  {} / {} gathered ({:.3f} %) under {:.3g} A", fmt::group_digits(num_clashing),
                 fmt::group_digits(num_gathered), percent_of_gathered_(num_clashing), clash_distance_angstrom_k);
    if (num_scored == 0) return;
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
}

/** One atom-count bin. `gathered` counts slots read back at our own atom count — the clash rate's denominator, and
 *  not `rdkit_energies.size()`, since a geometry RDKit declines to type still has a closest pair. */
struct bin_sample_t {
    std::size_t gathered = 0;
    std::size_t clashing = 0;
    std::vector<double> rdkit_energies;
    std::vector<double> paired_deltas;
};

/** Bucket every read-back sample by atom count once, then print the breakdown. The atom count is SmallField's own,
 *  so a salt form bins by the embedded fragment; the re-parsed count would migrate every counterion-carrying
 *  molecule one bin up, loading the heaviest bin with the rows the harness cannot read back. `delta_min` is reported
 *  and never gated — one order statistic whose expected magnitude grows with the sample, so a threshold on it would
 *  answer a question about `--quality-samples`. */
static void report_atom_count_bins_(custd::span<sampled_conformer_t const> sample,
                                    std::array<bin_sample_t, num_atom_count_bins_k> &bins) {
    for (bin_sample_t &bin : bins) bin = {};
    for (sampled_conformer_t const &drawn : sample) {
        if (!drawn.gathered()) continue; // no geometry was read back for this slot
        std::size_t const bin = atom_count_bin_of_(drawn.num_atoms);
        if (bin == num_atom_count_bins_k) continue; // atom count outside the tabulated bins
        ++bins[bin].gathered;
        if (drawn.clashing()) ++bins[bin].clashing;
        if (!drawn.scored()) continue; // RDKit never scored this geometry
        bins[bin].rdkit_energies.push_back(drawn.rdkit_energy);
        bins[bin].paired_deltas.push_back(drawn.paired_delta());
    }

    fmt::println("  Per-atom-count breakdown:");
    fmt::println("    {:>7}  {:>9}  {:>9}  {:>8}  {:>10}  {:>10}  {:>10}", "atoms", "gathered", "clash_pct", "scored",
                 "delta_P50", "delta_P95", "delta_min");
    for (std::size_t bin = 0; bin < num_atom_count_bins_k; ++bin) {
        std::sort(bins[bin].rdkit_energies.begin(), bins[bin].rdkit_energies.end());
        std::sort(bins[bin].paired_deltas.begin(), bins[bin].paired_deltas.end());
        std::vector<double> const &deltas = bins[bin].paired_deltas;
        if (bins[bin].gathered == 0) continue; // no sample landed in this bin
        double const clash_percent = 100.0 * static_cast<double>(bins[bin].clashing) /
                                     static_cast<double>(bins[bin].gathered);
        fmt::println("    {:>3}-{:<3}  {:>9}  {:>9.3f}  {:>8}  {:>+10.3g}  {:>+10.3g}  {:>+10.3g}",
                     atom_count_bin_lower_bound_(bin), atom_count_bin_bounds_k[bin],
                     fmt::group_digits(bins[bin].gathered), clash_percent, fmt::group_digits(deltas.size()),
                     percentile(deltas, 50), percentile(deltas, 95), deltas.empty() ? 0.0 : deltas.front());
    }
}

/*
 *  The four numbers the heavy-bin verdict is rendered against — each a rate or a percentile, never an extremum,
 *  whose expected magnitude would grow with `--quality-samples` and answer a question about the flag. The three
 *  energy bounds were measured on the 100-128 atom bin; the clash bound needs no calibration, since a pair under
 *  half the shortest MMFF94 bond is not a geometry at any size and only its frequency is in question.
 */
static constexpr double max_heavy_bin_clash_percent_k = 1.0;
static constexpr double max_heavy_bin_energy_p50_k = 180.0;
static constexpr double max_heavy_bin_energy_p95_k = 300.0;
static constexpr double max_heavy_bin_delta_p50_k = 10.0;

/** Render the verdict on the heavy bin, or decline to when too few samples reached it. The clash rate is read first
 *  and reads no energy, so a geometry defect names itself; the absolute bounds catch a blown-up geometry it misses,
 *  the paired median a force-field divergence leaving absolute energies plausible. Deciding numbers print at `{}`,
 *  shortest round-trip, so a near-pass never renders as a near-miss. */
static gate_verdict_t judge_heavy_bin_(bin_sample_t const &heavy_bin) {
    std::uint32_t const lower = atom_count_bin_lower_bound_(gate_bin_k);
    std::uint32_t const upper = atom_count_bin_bounds_k[gate_bin_k];
    if (heavy_bin.rdkit_energies.size() < min_heavy_bin_samples_k) {
        fmt::println("NOTE heavy_bin_gate atoms={}-{} gathered={} compared={} required={}", lower, upper,
                     heavy_bin.gathered, heavy_bin.rdkit_energies.size(), min_heavy_bin_samples_k);
        return gate_verdict_t::not_evaluated_k;
    }

    double const heavy_clash_percent = 100.0 * static_cast<double>(heavy_bin.clashing) /
                                       static_cast<double>(heavy_bin.gathered);
    double const heavy_p50 = percentile(heavy_bin.rdkit_energies, 50);
    double const heavy_p95 = percentile(heavy_bin.rdkit_energies, 95);
    double const heavy_delta_p50 = percentile(heavy_bin.paired_deltas, 50);
    char const *failed_threshold = nullptr;
    if (heavy_clash_percent >= max_heavy_bin_clash_percent_k) failed_threshold = "clash_percent";
    else if (heavy_p50 >= max_heavy_bin_energy_p50_k) failed_threshold = "P50";
    else if (heavy_p95 >= max_heavy_bin_energy_p95_k) failed_threshold = "P95";
    else if (std::fabs(heavy_delta_p50) >= max_heavy_bin_delta_p50_k) failed_threshold = "delta_P50";

    // One line for both verdicts, so a recorded PASS and FAIL carry the same fields, `failed_threshold=` included.
    fmt::println("{} heavy_bin_gate atoms={}-{} gathered={} compared={} " //
                 "clash_percent={:.3f} P50={} P95={} delta_P50={} "       //
                 "failed_threshold={}",                                   //
                 failed_threshold == nullptr ? "PASS" : "FAIL", lower, upper, heavy_bin.gathered,
                 heavy_bin.rdkit_energies.size(), heavy_clash_percent, heavy_p50, heavy_p95, heavy_delta_p50,
                 failed_threshold == nullptr ? "none" : failed_threshold);
    return failed_threshold == nullptr ? gate_verdict_t::passed_k : gate_verdict_t::failed_k;
}

static gate_verdict_t cross_validate(std::vector<std::string> const &smiles, std::vector<__half> const &coords,
                                     std::vector<float> const &energies,
                                     std::vector<sfapi::conformer_status_t> const &status,
                                     std::vector<std::uint8_t> const &num_atoms_per_molecule,
                                     std::uint32_t num_conformers, std::uint32_t max_samples) {
    sfapi::conformer_slot_layout_t const slot_layout {coordinate_stride_atoms_k, num_conformers};
    fmt::println("\nQuality cross-validation");

    std::vector<sampled_conformer_t> sample;
    fill_success_sample_into_(status, slot_layout, max_samples, sample);
    fmt::println("  Evaluating {} conformers via RDKit MMFF94...", fmt::group_digits(sample.size()));
    sample_skips_t const skips = score_sample_with_rdkit_(smiles, coords, energies, num_atoms_per_molecule, slot_layout,
                                                          sample);

    // An empty or unscored sample is not short-circuited: `judge_heavy_bin_` prints the `NOTE` naming the shortfall.
    std::vector<double> scratch;
    scratch.reserve(sample.size());
    report_energy_distributions_(sample, skips, scratch);

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

/** Everything this driver takes from the command line: `smiles_path` from `--smi` or `SMALLFIELD_TEST_CORPUS` with
 *  no default, `max_stereo_attempts` per molecule matched by the GPU config, `quality_samples` a ceiling on the
 *  conformers drawn for the RDKit cross-validation. */
struct backend_bench_arguments_t {
    std::string_view smiles_path;
    std::uint32_t num_conformers = 3;
    std::uint32_t max_stereo_attempts = 10;
    std::uint32_t quality_samples = 10000;
    bench_scope_t scope = bench_scope_t::cpu_and_single_gpu_k;
};

using sfbench::arguments_parse_t;

static void print_usage_(char const *invocation) {
    fmt::println(stderr, "usage: {} --smi PATH [--conformers N] [--quality-samples N] [--multi-gpu-only]", invocation);
}

/** Fill @p arguments from `argv`, answer a help request, or explain on `stderr` why the line was refused. */
static arguments_parse_t parse_arguments_(int argc, char **argv, backend_bench_arguments_t &arguments) {
    for (sfbench::command_line_t line {executable_name_k, argc, argv}; line.advance();) {
        arguments_parse_t status = arguments_parse_t::accepted_k;
        if (line.flag() == "--smi") status = line.take_text(arguments.smiles_path);
        else if (line.flag() == "--conformers")
            status = line.take_number(max_conformers_per_molecule_k, arguments.num_conformers);
        else if (line.flag() == "--quality-samples")
            status = line.take_number(max_quality_samples_k, arguments.quality_samples);
        else if (line.flag() == "--multi-gpu-only") arguments.scope = bench_scope_t::multi_gpu_only_k;
        else if (line.flag() == "--help") {
            print_usage_(argv[0]);
            return arguments_parse_t::help_requested_k;
        }
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

/** Run both systems at every batch size the corpus supports, returning the quality gate's verdict. Deliberately
 *  non-short-circuiting: a rung that fails the gate still leaves later rungs' throughput to report. */
static gate_verdict_t run_scaling_sweep_(std::vector<std::string> const &all_smiles,
                                         backend_bench_arguments_t const &arguments) {
    // Rungs the corpus can fill, increasing, whole corpus last — so a corpus of any size runs a rung and is gated.
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
        // Post-preprocess atom counts; without them the cross-validation cannot tell which salt fragment these are.
        std::vector<std::uint8_t> num_atoms_per_molecule;
        bench_smallfield_gpu(smiles, arguments.num_conformers, arguments.max_stereo_attempts, coords, energies, status,
                             num_atoms_per_molecule);

        // Multi-GPU runs in a fresh process under `--multi-gpu-only`, so no 1-GPU state bleeds into the 8-GPU run.
        if (batch_size == batch_sizes.back())
            fmt::println("\nSystem B: SmallField multi-GPU skipped in mixed mode. "       //
                         "Re-run with --multi-gpu-only for isolated multi-GPU numbers."); //

        // Cross-validate on the whole corpus only, which the last rung always is.
        if (batch_size == batch_sizes.back()) {
            gate_verdict_t const rung_verdict = cross_validate(smiles, coords, energies, status, num_atoms_per_molecule,
                                                               arguments.num_conformers, arguments.quality_samples);
            // A failure anywhere in the sweep is the answer; a later pass does not clear it.
            if (verdict != gate_verdict_t::failed_k) verdict = rung_verdict;
        }
    }
    return verdict;
}

int main(int argc, char **argv) {
    backend_bench_arguments_t arguments;
    switch (parse_arguments_(argc, argv, arguments)) {
    case arguments_parse_t::accepted_k: break;
    case arguments_parse_t::help_requested_k: return 0;
    case arguments_parse_t::refused_k: return 2;
    }

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
