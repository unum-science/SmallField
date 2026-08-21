/**
 *  @file bench/bench_bounds_embedding.cpp
 *  @brief Score `build_bounds_matrix` by handing it to RDKit's own embedder and reading what comes back.
 *
 *  Our refusal counter cannot answer this: `bounds_infeasible_k` is emitted at one site and has read zero across
 *  every corpus measured, a null control in which a matrix admitting no good embedding is indistinguishable from one
 *  admitting a perfect one. A wrong bounds matrix produces a worse geometry inside the population already counted
 *  successful, which only an embedder driven by the matrix under test can see. So this drives
 *  `RDKit::DGeomHelpers::EmbedMolecule` three times per molecule at one seed, changing nothing but where the distance
 *  bounds came from, and scores each embedding by its MMFF94 single-point energy. The two control arms are the design
 *  rather than decoration: an embedder run with no custom matrix and one run with RDKit's own `setTopolBounds` matrix
 *  pushed back through `params.boundsMat` are one measurement through two channels, so any gap is the instrument's
 *  own error and bounds how much of the third arm's gap can be believed. Triangle smoothing is applied here rather
 *  than left to the embedder because its verdict is the one refusal our own pipeline cannot report: a matrix no
 *  coordinates satisfy, caught by the smoother on the host and counted beside the embedding failures.
 */

#include <cmath>   // `std::isfinite`, `std::fabs`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `stderr`

#include <algorithm>   // `std::sort`, `std::nth_element`
#include <memory>      // `std::unique_ptr`, `std::make_shared`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h>

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/TriangleSmooth.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>

#include "smallfield/preprocess/bounds_matrix_builder.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/views.hpp"

#include "bench/bench_arguments.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfbench = smallfield::bench;
namespace sfp = smallfield::preprocess;
namespace sftest = smallfield::tests;
namespace sftypes = smallfield::types;

/** Name printed on every line so a concatenated CI log stays attributable. */
static constexpr char const *executable_name_k = "bench_bounds_embedding";

/** Molecules read from the corpus when `--molecules` is not given; the size the recorded experiment used. */
static constexpr std::size_t default_molecule_count_k = 300;

/** Embedding seed when `--seed` is not given; changing it changes which molecules ETKDG happens to refuse. */
static constexpr std::uint32_t default_seed_k = 42;

/** Ceilings on the value-taking flags, so a mistyped count is a diagnostic and not a `std::bad_alloc`. */
static constexpr std::size_t max_molecule_count_k = 1'000'000;
static constexpr std::uint32_t max_seed_k = 2'000'000'000;

#pragma region Arms

/** Where one arm's distance bounds came from: the first two are controls and the third the measurement, one
 *  enumeration rather than a flag on a shared routine because the whole comparison is between them and a run that
 *  quietly dropped one would still print a table. */
enum class bounds_source_t : std::uint8_t {
    /** No custom matrix: the embedder calls `setTopolBounds` itself. The reference every delta is against. */
    embedder_own_k,
    /** RDKit's `setTopolBounds`, smoothed and pushed back through `params.boundsMat`. */
    rdkit_through_channel_k,
    /** Our `build_bounds_matrix`, smoothed and pushed through the same channel. */
    smallfield_k,
};

/** Column heading for an arm. */
static constexpr char const *bounds_source_name(bounds_source_t source) noexcept {
    switch (source) {
    case bounds_source_t::embedder_own_k: return "embedder-own";
    case bounds_source_t::rdkit_through_channel_k: return "rdkit-through-channel";
    case bounds_source_t::smallfield_k: return "smallfield";
    }
    return "unnamed-source";
}

/** What one arm did to one molecule: `scored_k` embedded and MMFF94 scored the raw coordinates,
 *  `descriptor_refused_k` our builder refused it so the arm has nothing to say, `smoothing_refused_k` triangle
 *  smoothing found the matrix inconsistent so no embedding was attempted, `embedding_failed_k` `EmbedMolecule`
 *  returned no conformer, `energy_unavailable_k` the embedding exists but MMFF94 could not type or score it. */
enum class arm_outcome_t : std::uint8_t {
    scored_k,
    descriptor_refused_k,
    smoothing_refused_k,
    embedding_failed_k,
    energy_unavailable_k,
};

/** Running counts and the paired energies for one arm across the corpus; `energy_deltas` holds one entry per
 *  molecule this arm and the reference arm both scored, in kcal/mol. */
struct arm_tally_t {
    std::size_t attempted = 0;
    std::size_t scored = 0;
    std::size_t descriptor_refused = 0;
    std::size_t smoothing_refused = 0;
    std::size_t embedding_failed = 0;
    std::size_t energy_unavailable = 0;
    std::vector<double> energy_deltas;
};

static void record_outcome(arm_tally_t &tally, arm_outcome_t outcome) {
    ++tally.attempted;
    switch (outcome) {
    case arm_outcome_t::scored_k: ++tally.scored; return;
    case arm_outcome_t::descriptor_refused_k: ++tally.descriptor_refused; return;
    case arm_outcome_t::smoothing_refused_k: ++tally.smoothing_refused; return;
    case arm_outcome_t::embedding_failed_k: ++tally.embedding_failed; return;
    case arm_outcome_t::energy_unavailable_k: ++tally.energy_unavailable; return;
    }
}

#pragma endregion Arms

#pragma region Measurement

/** Buffers every molecule reuses, so a corpus pass grows them only at first use. */
struct sweep_scratch_t {
    std::vector<float> our_bounds;
    sftypes::molecule_descriptor_t descriptor;
};

/** Copy our packed bounds into a fresh `DistGeom::BoundsMatrix`. Our layout keeps the upper bound above the diagonal
 *  and the lower below, the convention `types/views.hpp` states and `upper_distance_bound_at` reads; RDKit's is
 *  addressed by an ordered pair, so this is where the two meet. */
static void fill_rdkit_matrix_from_ours_(sftypes::packed_matrix<float> ours, std::uint32_t num_atoms,
                                         DistGeom::BoundsMatrix &destination) {
    for (std::uint32_t row = 0; row < num_atoms; ++row) {
        destination.setUpperBound(row, row, 0.0);
        destination.setLowerBound(row, row, 0.0);
        for (std::uint32_t column = row + 1; column < num_atoms; ++column) {
            destination.setUpperBound(row, column,
                                      static_cast<double>(sftypes::upper_distance_bound_at(ours, row, column)));
            destination.setLowerBound(row, column,
                                      static_cast<double>(sftypes::lower_distance_bound_at(ours, row, column)));
        }
    }
}

/** RDKit's own `setTopolBounds` matrix for @p molecule, initialized the way the embedder initializes it. */
static DistGeom::BoundsMatPtr rdkit_bounds_for_(RDKit::ROMol const &molecule) {
    auto *raw = new DistGeom::BoundsMatrix(molecule.getNumAtoms());
    DistGeom::BoundsMatPtr owned(raw);
    RDKit::DGeomHelpers::initBoundsMat(raw);
    RDKit::DGeomHelpers::setTopolBounds(molecule, owned);
    return owned;
}

/** Our `build_bounds_matrix` output for @p molecule, or empty when the descriptor builder refused it. */
static DistGeom::BoundsMatPtr our_bounds_for_(RDKit::ROMol const &molecule, sweep_scratch_t &scratch) {
    if (sftypes::build_descriptor(molecule, scratch.descriptor) != sftypes::conformer_status_t::success_k)
        return DistGeom::BoundsMatPtr();
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    sftypes::packed_matrix<float> const ours = sftypes::reset_square_matrix(scratch.our_bounds, num_atoms, 0.0f);
    sfp::build_bounds_matrix(scratch.descriptor, ours);

    auto *raw = new DistGeom::BoundsMatrix(num_atoms);
    DistGeom::BoundsMatPtr owned(raw);
    fill_rdkit_matrix_from_ours_(ours, num_atoms, *raw);
    return owned;
}

/**
 *  @brief Embed @p molecule once under @p source and score the raw coordinates with MMFF94.
 *
 *  @param[in] molecule Molecule with explicit hydrogens; copied, so the caller's copy keeps no conformer.
 *  @param[in] seed Embedding seed, identical across arms so only the bounds differ.
 *  @param[out] energy MMFF94 single-point energy in kcal/mol, written only on `scored_k`.
 */
static arm_outcome_t embed_and_score_(RDKit::ROMol const &molecule, bounds_source_t source, std::uint32_t seed,
                                      sweep_scratch_t &scratch, double &energy) {
    RDKit::DGeomHelpers::EmbedParameters parameters = RDKit::DGeomHelpers::ETKDGv3;
    parameters.randomSeed = static_cast<int>(seed);

    if (source != bounds_source_t::embedder_own_k) {
        DistGeom::BoundsMatPtr bounds = source == bounds_source_t::rdkit_through_channel_k
                                            ? rdkit_bounds_for_(molecule)
                                            : our_bounds_for_(molecule, scratch);
        if (!bounds) return arm_outcome_t::descriptor_refused_k;
        if (!DistGeom::triangleSmoothBounds(bounds)) return arm_outcome_t::smoothing_refused_k;
        parameters.boundsMat = bounds;
    }

    RDKit::RWMol working(molecule);
    int const conformer_id = RDKit::DGeomHelpers::EmbedMolecule(working, parameters);
    if (conformer_id < 0) return arm_outcome_t::embedding_failed_k;

    energy = sftest::rdkit_mmff94_energy(working, conformer_id);
    if (!std::isfinite(energy)) return arm_outcome_t::energy_unavailable_k;
    return arm_outcome_t::scored_k;
}

#pragma endregion Measurement

#pragma region Reporting

/** Linear-interpolated median of a sample, which is sorted in place; zero for an empty one. */
static double median_of(std::vector<double> &values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    std::size_t const last = values.size() - 1;
    double const position = 0.5 * static_cast<double>(last);
    auto const lower = static_cast<std::size_t>(position);
    std::size_t const upper = lower + 1 < values.size() ? lower + 1 : lower;
    double const fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

/** How many of the paired deltas are worse than the reference, which is the table's `worse / all` column. */
static std::size_t count_worse(std::vector<double> const &deltas) {
    std::size_t worse = 0;
    for (double const delta : deltas) worse += delta > 0.0;
    return worse;
}

#pragma endregion Reporting

#pragma region Driver

static void print_usage() {
    fmt::println("Usage: {} [--smi PATH] [--molecules N] [--seed N]", executable_name_k);
    fmt::println("  --smi PATH        SMILES corpus, one per line; defaults to ${}",
                 sfbench::corpus_environment_variable_k);
    fmt::println("  --molecules N     Molecules to read from the corpus (default {})", default_molecule_count_k);
    fmt::println("  --seed N          Embedding seed, shared by every arm (default {})", default_seed_k);
}

int main(int argc, char **argv) {
    std::string_view corpus_path = sfbench::corpus_path_from_environment();
    std::size_t molecule_count = default_molecule_count_k;
    std::uint32_t seed = default_seed_k;

    sfbench::command_line_t command_line {executable_name_k, argc, argv};
    while (command_line.advance()) {
        std::string_view const flag = command_line.flag();
        sfbench::arguments_parse_t status = sfbench::arguments_parse_t::accepted_k;
        if (flag == "--help" || flag == "-h") {
            print_usage();
            return 0;
        }
        else if (flag == "--smi") status = command_line.take_text(corpus_path);
        else if (flag == "--molecules") status = command_line.take_number(max_molecule_count_k, molecule_count);
        else if (flag == "--seed") status = command_line.take_number(max_seed_k, seed);
        else status = command_line.refuse_unknown();
        if (status != sfbench::arguments_parse_t::accepted_k) return 1;
    }

    std::vector<std::string> corpus;
    sfbench::corpus_load_t const load = sfbench::load_corpus_into(corpus_path, molecule_count, corpus);
    if (load != sfbench::corpus_load_t::loaded_k) {
        fmt::println(stderr, "{}: corpus '{}' {}", executable_name_k, corpus_path, sfbench::describe_corpus_load(load));
        return 1;
    }

    constexpr bounds_source_t arms_k[] = {bounds_source_t::embedder_own_k, bounds_source_t::rdkit_through_channel_k,
                                          bounds_source_t::smallfield_k};
    constexpr std::size_t num_arms_k = std::size(arms_k);
    arm_tally_t tallies[num_arms_k];
    sweep_scratch_t scratch;
    std::size_t unparsable = 0;

    for (std::string const &smiles : corpus) {
        std::unique_ptr<RDKit::RWMol> const molecule = sftest::mol_from_smiles(smiles);
        if (!molecule) {
            ++unparsable;
            continue;
        }

        double energies[num_arms_k] = {};
        arm_outcome_t outcomes[num_arms_k] = {};
        for (std::size_t arm_index = 0; arm_index < num_arms_k; ++arm_index) {
            outcomes[arm_index] = embed_and_score_(*molecule, arms_k[arm_index], seed, scratch, energies[arm_index]);
            record_outcome(tallies[arm_index], outcomes[arm_index]);
        }

        // Paired per molecule against the reference arm: one the reference could not score contributes to no delta.
        if (outcomes[0] != arm_outcome_t::scored_k) continue;
        for (std::size_t arm_index = 1; arm_index < num_arms_k; ++arm_index)
            if (outcomes[arm_index] == arm_outcome_t::scored_k)
                tallies[arm_index].energy_deltas.push_back(energies[arm_index] - energies[0]);
    }

    fmt::println("{} corpus={} molecules={} unparsable={} seed={}", executable_name_k, corpus_path, corpus.size(),
                 unparsable, seed);
    fmt::println("{:<24} {:>9} {:>9} {:>9} {:>9} {:>11} {:>13}", "arm", "embedded", "embedfail", "smoothfail",
                 "noenergy", "dE-median", "worse/paired");
    for (std::size_t arm_index = 0; arm_index < num_arms_k; ++arm_index) {
        arm_tally_t &tally = tallies[arm_index];
        std::size_t const worse = count_worse(tally.energy_deltas);
        std::size_t const paired = tally.energy_deltas.size();
        double const median = median_of(tally.energy_deltas);
        fmt::println("{:<24} {:>9} {:>9} {:>9} {:>9} {:>11} {:>13}", bounds_source_name(arms_k[arm_index]),
                     tally.scored, tally.embedding_failed + tally.descriptor_refused, tally.smoothing_refused,
                     tally.energy_unavailable,
                     arm_index == 0 ? std::string {"reference"} : fmt::format("{:+.2f}", median),
                     arm_index == 0 ? std::string {"—"} : fmt::format("{}/{}", worse, paired));
    }

    // Both controls drive one embedder over one topology at one seed, so a gap between them is the channel's own
    // error and bounds how much of the third arm's gap can be believed. The energy half is what is worth gating on:
    // one disagreement about embedding is one sample, but a non-zero median across the paired set means
    // `params.boundsMat` is not a transparent channel and every number above it measures the instrument.
    std::size_t const control_gap = tallies[0].scored > tallies[1].scored ? tallies[0].scored - tallies[1].scored
                                                                          : tallies[1].scored - tallies[0].scored;
    double const control_median = median_of(tallies[1].energy_deltas);
    fmt::println("CONTROL-GAP embedded={} of {} paired-dE-median={:+.4f}", control_gap, corpus.size() - unparsable,
                 control_median);

    if (tallies[0].attempted == 0) {
        fmt::println(stderr, "{}: no molecule reached any arm, so nothing was compared", executable_name_k);
        return 1;
    }
    // Widened only for a reason that is measured and named, never to make a run pass.
    constexpr double control_median_tolerance_k = 1.0e-3;
    if (std::fabs(control_median) > control_median_tolerance_k) {
        fmt::println(stderr, "{}: the two controls disagree by {:+.4f} kcal/mol at the median, past {}",
                     executable_name_k, control_median, control_median_tolerance_k);
        return 1;
    }
    return 0;
}

#pragma endregion Driver
