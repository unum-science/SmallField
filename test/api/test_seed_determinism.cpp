/**
 *  @file test/api/test_seed_determinism.cpp
 *  @brief Assert that a fixed `random_seed` reproduces conformers bit for bit, whatever surrounds them.
 *
 *  Three properties, each its own section, because each fails its own way. `repeat` runs one batch twice
 *  through one `context_t`: preprocess schedules molecules dynamically over host threads and atom-count-sorts
 *  them per device, so a molecule's offset inside the device batch tracks thread timing, while the kernel
 *  keys Philox off its position in the caller's input instead. OpenMP stays at its default width, and the
 *  corpus spans the atom-count bins widely enough that the dynamic chunks interleave differently between
 *  the two calls. `fresh_context`
 *  repeats it against a second `context_t` built after the first is destroyed, which is what a second process
 *  sees and the only section that catches a divergence carried by persistent per-device scratch or by a wave
 *  size chosen from whatever VRAM was free. `composition` runs a prefix alone, then as the head of a longer
 *  batch: appending must leave the prefix's coordinates unchanged to the bit.
 *
 *  Every section compares coordinates element by element, since a divergence that reshuffles which rows move
 *  is invisible to anything coarser, and stamps the two runs with different poison bytes so a slot neither
 *  wrote is a mismatch rather than agreement.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::fflush`, `stderr`
#include <cstring> // `std::memcmp`, `std::memset`

#include <algorithm>   // `std::min`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <cuda_fp16.h>

#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/api/smallfield.hpp"
#include "smallfield/types/views.hpp"

#include "test/test_helpers.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfapi = smallfield::api;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_seed_determinism";

/** Molecules the built-in corpus is grown to, and the cap applied to a named one. */
static constexpr std::size_t default_max_molecules_k = 768;

/** Printed mismatch lines one section may spend; the rest are counted and reported as `suppressed`. */
static constexpr std::size_t max_reported_mismatches_k = 20;

/** Conformers per molecule, and the coordinate stride every run is written through. */
static constexpr std::uint32_t num_conformers_k = 3;
static constexpr std::uint32_t max_atoms_k = 128;

/** The seed under test; the point of the file is that this number alone decides the geometry. */
static constexpr std::uint64_t random_seed_k = 4242;

/** Molecules the composition section runs alone, and how many it then appends behind them. */
static constexpr std::uint32_t composition_prefix_k = 64;
static constexpr std::uint32_t composition_extension_k = 32;

/** Pre-run stamp over each run's buffers; the two differ, so a slot neither filled cannot agree by default. */
static constexpr int first_run_poison_k = 0xa5;
static constexpr int second_run_poison_k = 0x5a;

namespace smallfield::tests {

using api::conformer_slot_layout_t;
using api::conformer_status_t;

#pragma region Corpus

/**
 *  @brief Build @p num_molecules distinct, valid SMILES into @p corpus_out, cleared first, large enough
 *  to expose scheduling jitter.
 *
 *  Cycles scaffolds and grows an alkyl tail so no two entries repeat, up to four scaffold repeats, running
 *  atom counts from a dozen to near the largest bin — each bin is its own kernel launch over its own group,
 *  so a corpus landing in one bin exercises one of them.
 */
static void fill_corpus_into(std::size_t num_molecules, std::vector<std::string> &corpus_out) {
    static char const *const scaffolds[] = {"c1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCNCC1", "c1ccsc1", "C1COCCN1"};
    static char const *const heads[] = {"CC", "CCO", "CCN", "CC(=O)", "CCC(N)", "COC"};
    constexpr std::size_t num_scaffolds = sizeof(scaffolds) / sizeof(scaffolds[0]);
    constexpr std::size_t num_heads = sizeof(heads) / sizeof(heads[0]);

    corpus_out.clear();
    corpus_out.reserve(num_molecules);
    for (std::size_t molecule = 0; molecule < num_molecules; ++molecule) {
        std::string smiles = heads[molecule % num_heads];
        std::size_t const tail_length = 1u + (molecule / (num_heads * num_scaffolds)) % 6u;
        std::size_t const num_repeats = 1u + (molecule / num_heads) % 4u;
        for (std::size_t repeat = 0; repeat < num_repeats; ++repeat) {
            for (std::size_t carbon = 0; carbon < tail_length; ++carbon) smiles += 'C';
            smiles += scaffolds[(molecule / num_heads + repeat) % num_scaffolds];
        }
        corpus_out.push_back(smiles);
    }
}

#pragma endregion Corpus

#pragma region Runs

/** One `generate_conformers` call's outputs, owned by the caller and reused across sections. */
struct generation_outputs_t {
    std::vector<__half> coordinates;
    std::vector<float> energies;
    std::vector<conformer_status_t> statuses;
    std::vector<std::uint8_t> num_atoms;
    conformer_status_t batch_status = conformer_status_t::success_k;
    std::uint64_t cuda_errors = 0;

    /**
     *  @brief Size every buffer for @p num_molecules through @p layout and stamp it with @p poison.
     *  The poison makes an unwritten slot visible: runs stamped with different bytes agree only where the
     *  pipeline actually wrote, so a section can never pass on buffers nobody filled.
     */
    void reset_for(std::size_t num_molecules, conformer_slot_layout_t layout, int poison) {
        std::size_t const num_slots = num_molecules * layout.num_conformers;
        coordinates.resize(num_slots * layout.slot_stride());
        energies.resize(num_slots);
        statuses.resize(num_slots);
        num_atoms.resize(num_molecules);
        // `__half` keeps its storage `protected`, so the byte stamp goes through a `void *`.
        std::memset(static_cast<void *>(coordinates.data()), poison, coordinates.size() * sizeof(__half));
        std::memset(energies.data(), poison, energies.size() * sizeof(float));
        std::memset(statuses.data(), poison, statuses.size() * sizeof(conformer_status_t));
        std::memset(num_atoms.data(), poison, num_atoms.size());
    }
};

/** Run one batch through GPU @p context — whose CUDA error counter is read and reset around the call — over
 *  @p corpus in the input order that keys the seed, into @p outputs_out sized for @p layout and stamped
 *  with @p poison first. */
static void generate_into(sfapi::context_t &context, custd::span<std::string_view const> corpus,
                          conformer_slot_layout_t layout, int poison, generation_outputs_t &outputs_out) {
    sfapi::config_t config;
    config.num_conformers = layout.num_conformers;
    config.max_stereo_attempts = 20;
    config.random_seed = random_seed_k;

    outputs_out.reset_for(corpus.size(), layout, poison);
    context.cuda_error_count.store(0);
    outputs_out.batch_status = context.generate_conformers(corpus, config, layout.max_atoms, outputs_out.coordinates,
                                                           outputs_out.energies, outputs_out.statuses,
                                                           outputs_out.num_atoms);
    outputs_out.cuda_errors = context.cuda_error_count.load();
}

/** Conformers a run finished, which is what tells a passing comparison from a comparison of nothing. */
static std::size_t count_successes(generation_outputs_t const &outputs, std::size_t num_slots) {
    std::size_t successes = 0;
    for (std::size_t slot = 0; slot < num_slots; ++slot)
        if (outputs.statuses[slot] == conformer_status_t::success_k) ++successes;
    return successes;
}

#pragma endregion Runs

#pragma region Comparison

/** What one section counted, in its diagnostics' units. The `unwritten_on_one_side` count is differing
 *  elements one run left at zero, separating a geometry that moved from one never written. */
struct divergence_tally_t {
    std::size_t slots_compared = 0;
    std::size_t slots_with_moved_coordinates = 0;
    std::size_t coordinate_elements_differing = 0;
    std::size_t coordinate_elements_unwritten_on_one_side = 0;
    std::size_t coordinate_elements_compared = 0;
    std::size_t successes_first = 0;
    std::size_t successes_second = 0;
};

/** How one slot of two runs differs; a `first_element` equal to the slot length means it does not. */
struct slot_difference_t {
    std::size_t first_element = 0;
    std::size_t differing_elements = 0;
    std::size_t unwritten_on_one_side = 0;
};

/**
 *  @brief Locate and count the halves that differ between slot @p first and slot @p second: where they
 *  first part company, how far apart they are, and how much of that is a zero.
 *  Raw bytes rather than floats, because bitwise reproduction is the claim: two halves that both decode
 *  to a NaN are still a divergence, and `==` would call them equal.
 */
static slot_difference_t compare_slot_(custd::span<__half const> first, custd::span<__half const> second) {
    slot_difference_t difference;
    difference.first_element = first.size();
    for (std::size_t element = 0; element < first.size(); ++element) {
        if (std::memcmp(&first[element], &second[element], sizeof(__half)) == 0) continue;
        if (difference.first_element == first.size()) difference.first_element = element;
        ++difference.differing_elements;
        bool const first_is_zero = __half2float(first[element]) == 0.0f;
        bool const second_is_zero = __half2float(second[element]) == 0.0f;
        if (first_is_zero != second_is_zero) ++difference.unwritten_on_one_side;
    }
    return difference;
}

/**
 *  @brief Compare reference @p first against @p second slot by slot, reporting every field that did not
 *  reproduce through @p reporter under @p subject and counting into @p tally_out.
 *
 *  Walks the first @p num_molecules molecules of both at the same input index, which is what lets a longer
 *  batch be compared against the prefix it contains. Coordinates are addressed through @p layout rather than
 *  by a flat read, and the differing element is named by its atom and axis so a report says where it moved.
 */
static void compare_runs_into(diagnostic_reporter_t &reporter, std::string_view subject, conformer_slot_layout_t layout,
                              std::size_t num_molecules, generation_outputs_t const &first,
                              generation_outputs_t const &second, divergence_tally_t &tally_out) {
    for (std::size_t molecule = 0; molecule < num_molecules; ++molecule) {
        for (std::uint32_t conformer = 0; conformer < layout.num_conformers; ++conformer) {
            std::size_t const slot = molecule * layout.num_conformers + conformer;
            std::size_t const base = layout.slot_offset(molecule, conformer);
            custd::span<__half const> const first_slot {first.coordinates.data() + base, layout.slot_stride()};
            custd::span<__half const> const second_slot {second.coordinates.data() + base, layout.slot_stride()};

            slot_difference_t const difference = compare_slot_(first_slot, second_slot);
            bool const coordinates_agree = difference.first_element == first_slot.size();
            std::size_t const reported = coordinates_agree ? 0 : difference.first_element;
            std::uint32_t const axis = static_cast<std::uint32_t>(reported / layout.max_atoms);
            std::uint32_t const atom = static_cast<std::uint32_t>(reported - axis * layout.max_atoms);

            reporter
                // `num_atoms` keys the line as well as being compared below, since a log is filtered on
                // keys and whether the named atom is inside the molecule or in padding is asked first.
                .mismatch(subject, "molecule={} conformer={} atom={} axis={} num_atoms={}", molecule, conformer, atom,
                          axis, static_cast<std::uint32_t>(first.num_atoms[molecule]))
                .field("status", static_cast<std::uint32_t>(first.statuses[slot]),
                       static_cast<std::uint32_t>(second.statuses[slot]), exactly_equal)
                .field("num_atoms", static_cast<std::uint32_t>(first.num_atoms[molecule]),
                       static_cast<std::uint32_t>(second.num_atoms[molecule]), exactly_equal)
                .field("energy_kcal_per_mol", first.energies[slot], second.energies[slot],
                       [](float reference, float observed) {
                           return std::memcmp(&reference, &observed, sizeof(float)) == 0;
                       })
                // The predicate is the whole-slot byte compare, so the pair printed is the first it rejected.
                .field("coordinate_angstrom", __half2float(first_slot[reported]), __half2float(second_slot[reported]),
                       [coordinates_agree](float, float) { return coordinates_agree; });

            ++tally_out.slots_compared;
            tally_out.coordinate_elements_compared += first_slot.size();
            tally_out.coordinate_elements_differing += difference.differing_elements;
            tally_out.coordinate_elements_unwritten_on_one_side += difference.unwritten_on_one_side;
            if (difference.differing_elements != 0) ++tally_out.slots_with_moved_coordinates;
            if (first.statuses[slot] == conformer_status_t::success_k) ++tally_out.successes_first;
            if (second.statuses[slot] == conformer_status_t::success_k) ++tally_out.successes_second;
        }
    }
    reporter.summarize(subject);
}

/** Print what a section walked, so a run that compared few slots cannot read like one that compared many. */
static void print_section_summary(std::string_view subject, divergence_tally_t const &tally) {
    fmt::println("  {}: {} slots, {} coordinate elements, {} succeeded first / {} second, " //
                 "{} slots moved over {} elements, {} of them zero on one side",            //
                 subject, fmt::group_digits(tally.slots_compared),
                 fmt::group_digits(tally.coordinate_elements_compared), fmt::group_digits(tally.successes_first),
                 fmt::group_digits(tally.successes_second), fmt::group_digits(tally.slots_with_moved_coordinates),
                 fmt::group_digits(tally.coordinate_elements_differing),
                 fmt::group_digits(tally.coordinate_elements_unwritten_on_one_side));
}

#pragma endregion Comparison

#pragma region Sections

/** Same @p corpus, same seed, twice through one context and once through a fresh one, all at @p layout;
 *  @p first_out keeps the first run so the caller can gate on what it produced. */
static void check_repeatability(custd::span<std::string_view const> corpus, conformer_slot_layout_t layout,
                                diagnostic_reporter_t &reporter, generation_outputs_t &first_out) {
    generation_outputs_t second, third;
    int const device_id = 0;
    {
        sfapi::context_t context(custd::span<int const>(&device_id, 1));
        generate_into(context, corpus, layout, first_run_poison_k, first_out);
        generate_into(context, corpus, layout, second_run_poison_k, second);
    }

    divergence_tally_t repeat_tally;
    compare_runs_into(reporter, "repeat", layout, corpus.size(), first_out, second, repeat_tally);
    print_section_summary("repeat", repeat_tally);

    // A fresh context is what a second process sees: new per-device scratch, wave size re-resolved from VRAM.
    {
        sfapi::context_t context(custd::span<int const>(&device_id, 1));
        generate_into(context, corpus, layout, second_run_poison_k, third);
    }

    divergence_tally_t fresh_tally;
    compare_runs_into(reporter, "fresh_context", layout, corpus.size(), first_out, third, fresh_tally);
    print_section_summary("fresh_context", fresh_tally);
}

/**
 *  @brief The same molecules of @p corpus at the same input indices, alone and then followed by more,
 *  at @p layout and counted into @p tally_out.
 *
 *  The prefix keeps its input positions, so every conformer's seed stream is unchanged and the only
 *  difference is who else rode along. Anything that moves is a batch-composition dependency.
 */
static void check_batch_composition(custd::span<std::string_view const> corpus, conformer_slot_layout_t layout,
                                    diagnostic_reporter_t &reporter, divergence_tally_t &tally_out) {
    std::size_t const prefix_size = std::min<std::size_t>(composition_prefix_k, corpus.size());
    std::size_t const extended_size = std::min<std::size_t>(prefix_size + composition_extension_k, corpus.size());

    // A context per run, so call order and carried state cannot stand in for the variable: who else rode along.
    generation_outputs_t alone, extended;
    int const device_id = 0;
    {
        sfapi::context_t context(custd::span<int const>(&device_id, 1));
        generate_into(context, corpus.first(prefix_size), layout, first_run_poison_k, alone);
    }
    {
        sfapi::context_t context(custd::span<int const>(&device_id, 1));
        generate_into(context, corpus.first(extended_size), layout, second_run_poison_k, extended);
    }

    compare_runs_into(reporter, "composition", layout, prefix_size, alone, extended, tally_out);
    print_section_summary("composition", tally_out);
    fmt::println("  composition: {} molecules alone against the head of {}", fmt::group_digits(prefix_size),
                 fmt::group_digits(extended_size));
}

#pragma endregion Sections

} // namespace smallfield::tests

int main(int argc, char **argv) {
    fmt::println("{}: one seed reproduced across two calls, a fresh context, and a longer batch", test_name_k);

    auto const selection = sftest::select_corpus(argc, argv, default_max_molecules_k);
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        std::fflush(stdout);
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", test_name_k);
        return 2;
    }

    std::vector<std::string> corpus_storage;
    if (selection.knob == sftest::corpus_knob_t::named_k) {
        auto const outcome = sftest::load_smiles_into(selection.path, selection.max_molecules, corpus_storage);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), selection.path);
        fmt::println("  corpus: {} ({} molecules)", selection.path, fmt::group_digits(corpus_storage.size()));
    }
    else {
        sftest::fill_corpus_into(selection.max_molecules, corpus_storage);
        fmt::println("  corpus: {} generated molecules; name one on the command line or in SMALLFIELD_TEST_CORPUS",
                     fmt::group_digits(corpus_storage.size()));
    }
    std::vector<std::string_view> const corpus(corpus_storage.begin(), corpus_storage.end());

    sfapi::conformer_slot_layout_t const layout {max_atoms_k, num_conformers_k};
    sftest::diagnostic_reporter_t reporter {"first", "second"};
    reporter.limit_printed_lines(max_reported_mismatches_k);

    sftest::generation_outputs_t reference;
    sftest::check_repeatability(corpus, layout, reporter, reference);

    sftest::divergence_tally_t composition_tally;
    sftest::check_batch_composition(corpus, layout, reporter, composition_tally);

    std::size_t const num_slots = corpus.size() * num_conformers_k;
    std::size_t const successes = sftest::count_successes(reference, num_slots);
    fmt::println("  {} / {} conformers succeeded on the reference run, batch status {}, {} CUDA errors",
                 fmt::group_digits(successes), fmt::group_digits(num_slots),
                 static_cast<std::uint32_t>(reference.batch_status), reference.cuda_errors);

    if (reference.cuda_errors != 0)
        return sftest::report_fail(test_name_k, "the reference run raised {} CUDA errors, so its buffers prove nothing",
                                   reference.cuda_errors);
    if (successes == 0)
        return sftest::report_fail(test_name_k, "the corpus produced no conformers, so the comparison proves nothing");
    if (composition_tally.successes_first == 0)
        return sftest::report_fail(test_name_k, "the composition prefix produced no conformers");
    if (!reporter.passed())
        return sftest::report_fail(test_name_k, "seed {} did not reproduce: {} mismatches over {} comparisons",
                                   random_seed_k, reporter.mismatch_count(), reporter.comparison_count());
    return sftest::report_pass(test_name_k);
}
