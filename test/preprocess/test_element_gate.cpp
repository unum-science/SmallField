/**
 *  @file test/preprocess/test_element_gate.cpp
 *  @brief Validate the host-side element-validation gate in `prepare_molecule_into`.
 *
 *  Molecules whose atoms fall outside the MMFF94 parameter set (Y, lanthanides, transition metals, etc.) must be
 *  rejected here: left unchecked they reach `etkdg_pipeline_warp_kernel` and go out of bounds, poisoning every
 *  subsequent CUDA call on every device. These tests assert each known rejection path returns the right
 *  `conformer_status_t` so the caller can route to CPU fallback or surface a precise error to the user.
 *
 *  A curated table pins one molecule per rejection path and runs on every `ctest` invocation. A corpus named on
 *  the command line adds the two invariants that hold for arbitrary input: the gate answers only with a status it
 *  can produce, and it leaves atoms behind exactly when it accepts the molecule.
 */

#include <cstddef> // `std::size_t`

#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include "smallfield/api/conformer_status.hpp"
#include "smallfield/preprocess/etkdg_prepare_batch.hpp"

#include "test/test_helpers.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfapi = smallfield::api;
namespace sfprep = smallfield::preprocess;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_element_gate";

/** One pathological corpus molecule cannot bury the rest of the run. */
static constexpr std::size_t max_printed_lines_k = 64;

namespace smallfield::tests {

#pragma region Curated Cases

/** One curated molecule and the status the gate owes it. */
struct element_gate_case_t {
    std::string_view label;
    std::string_view smiles;
    sfapi::conformer_status_t expected;
};

/**
 *  @brief The atom count the single-warp pipeline caps a molecule at, hydrogens included.
 *
 *  `parse_smiles` applies it after `addHs`, so the alkanes below straddle the cap by chain length:
 *  C63H128 is 191 atoms and fits, C64H130 is 194 and does not.
 */
static constexpr std::size_t atom_cap_k = sfprep::max_atoms_per_molecule_k;

/** The whole curated corpus: one entry per path the gate can take. */
static element_gate_case_t const element_gate_cases_k[] = {
    // Happy path — drug-like organic, all MMFF94-supported elements.
    {"ethanol", "CCO", sfapi::conformer_status_t::success_k},
    {"benzene", "c1ccccc1", sfapi::conformer_status_t::success_k},
    {"aspirin", "CC(=O)Oc1ccccc1C(=O)O", sfapi::conformer_status_t::success_k},

    // Bug-report repro — yttrium (Z=39) is not in the MMFF94 atom-type table.
    {"yttrium-bracketed", "CC[Y]CC", sfapi::conformer_status_t::unsupported_element_k},

    // Other transition metals and lanthanides — no MMFF94 type.
    {"lone-iron", "[Fe]", sfapi::conformer_status_t::unsupported_element_k},
    {"lanthanum", "[La]", sfapi::conformer_status_t::unsupported_element_k},

    // SMILES parse failure — clearly malformed input.
    {"garbage", "not-a-smiles-string-((", sfapi::conformer_status_t::invalid_smiles_k},
};

/** The four statuses `prepare_molecule_into` can answer with; anything else means a path went unhandled. */
static bool gate_can_answer_(sfapi::conformer_status_t status) noexcept {
    switch (status) {
    case sfapi::conformer_status_t::success_k:
    case sfapi::conformer_status_t::invalid_smiles_k:
    case sfapi::conformer_status_t::too_many_atoms_k:
    case sfapi::conformer_status_t::unsupported_element_k: return true;
    default: return false;
    }
}

/** A status code and an atoms-survived flag agree only exactly. */
static bool agree_exactly_(unsigned reference, unsigned observed) noexcept { return reference == observed; }
static bool flags_agree_(bool reference, bool observed) noexcept { return reference == observed; }

#pragma endregion Curated Cases

#pragma region Checks

/**
 *  @brief Run one molecule through the gate and report its status and whether atoms survived.
 *
 *  @param[in] probe The molecule and the status expected of it.
 *  @param[in] scratch Reusable preparation buffer, cleared by `prepare_molecule_into` on entry.
 *  @param[inout] reporter Sink that counts and prints both comparisons.
 */
static void check_curated_case_(element_gate_case_t const &probe, sfprep::etkdg_prepared_molecule_t &scratch,
                                diagnostic_reporter_t &reporter) {
    std::string const smiles {probe.smiles};
    auto const status = sfprep::prepare_molecule_into(smiles, scratch);

    // A rejected molecule must leave nothing behind, so the caller cannot mistake a refusal for an
    // empty-but-accepted preparation and hand it to the device.
    bool const expected_atoms_present = probe.expected == sfapi::conformer_status_t::success_k;
    reporter.mismatch("element_gate", "molecule={} smiles={}", probe.label, probe.smiles)
        .field("status_code", static_cast<unsigned>(probe.expected), static_cast<unsigned>(status), agree_exactly_)
        .field("atoms_present", expected_atoms_present, scratch.num_atoms != 0, flags_agree_);
}

/**
 *  @brief Check the two invariants that hold for a molecule nobody wrote an expectation for.
 *
 *  @param[in] smiles Corpus molecule, of unknown validity.
 *  @param[in] scratch Reusable preparation buffer.
 *  @param[inout] reporter Sink that counts and prints both comparisons.
 *  @return The status the gate answered with, for the census.
 */
static sfapi::conformer_status_t check_corpus_molecule_(std::string const &smiles,
                                                        sfprep::etkdg_prepared_molecule_t &scratch,
                                                        diagnostic_reporter_t &reporter) {
    auto const status = sfprep::prepare_molecule_into(smiles, scratch);
    bool const expected_atoms_present = status == sfapi::conformer_status_t::success_k;
    reporter.mismatch("element_gate", "molecule=corpus smiles={}", smiles)
        .field("status_answerable", true, gate_can_answer_(status), flags_agree_)
        .field("atoms_present", expected_atoms_present, scratch.num_atoms != 0, flags_agree_);
    return status;
}

/** Per-status census of a corpus run; the comparison counts belong to the reporter's own section. */
struct status_census_t {
    std::size_t accepted = 0;
    std::size_t unsupported_element = 0;
    std::size_t too_many_atoms = 0;
    std::size_t invalid_smiles = 0;

    void count(sfapi::conformer_status_t status) noexcept {
        switch (status) {
        case sfapi::conformer_status_t::success_k: ++accepted; break;
        case sfapi::conformer_status_t::unsupported_element_k: ++unsupported_element; break;
        case sfapi::conformer_status_t::too_many_atoms_k: ++too_many_atoms; break;
        default: ++invalid_smiles; break;
        }
    }
};

#pragma endregion Checks

} // namespace smallfield::tests

int main(int argc, char **argv) {
    fmt::println("{}: MMFF94 element gate over curated rejections, then a corpus for the gate's invariants",
                 test_name_k);

    auto const selection = sftest::select_corpus(argc, argv, 10000);
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", argv[0]);
        return 2;
    }

    sftest::diagnostic_reporter_t reporter {"expected", "observed"};
    reporter.limit_printed_lines(max_printed_lines_k);
    sfprep::etkdg_prepared_molecule_t scratch {};

    for (auto const &probe : sftest::element_gate_cases_k) sftest::check_curated_case_(probe, scratch, reporter);

    // The atom-count cap is a boundary, so it is checked as one: the longest alkane that fits and the
    // shortest that does not, both named from the library constant rather than from a magic number.
    std::size_t const carbons_that_fit = (sftest::atom_cap_k - 2) / 3;
    std::string const alkane_at_cap(carbons_that_fit, 'C');
    std::string const alkane_over_cap(carbons_that_fit + 1, 'C');
    sftest::element_gate_case_t const cap_cases[] = {
        {"alkane-at-cap", alkane_at_cap, sfapi::conformer_status_t::success_k},
        {"alkane-over-cap", alkane_over_cap, sfapi::conformer_status_t::too_many_atoms_k},
    };
    for (auto const &probe : cap_cases) sftest::check_curated_case_(probe, scratch, reporter);
    reporter.summarize("curated");

    if (!reporter.passed())
        return sftest::report_fail(test_name_k, "curated element-gate expectations diverged over {} comparisons",
                                   reporter.comparison_count());

    if (selection.knob == sftest::corpus_knob_t::named_k) {
        // Naming a corpus is a request to validate against it, so an unreadable or empty file is a
        // failure rather than a zero-molecule run that satisfies every invariant vacuously.
        std::vector<std::string> corpus;
        auto const outcome = sftest::load_smiles_into(selection.path, selection.max_molecules, corpus);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), selection.path);
        fmt::println("{}: corpus {}, {} SMILES", test_name_k, selection.path, fmt::group_digits(corpus.size()));

        sftest::status_census_t census;
        for (auto const &smiles : corpus) census.count(sftest::check_corpus_molecule_(smiles, scratch, reporter));
        reporter.summarize("corpus");
        fmt::println("  corpus verdicts: {} accepted, {} unsupported element, {} over the atom cap, {} unparsable",
                     fmt::group_digits(census.accepted), fmt::group_digits(census.unsupported_element),
                     fmt::group_digits(census.too_many_atoms), fmt::group_digits(census.invalid_smiles));

        if (!reporter.passed())
            return sftest::report_fail(test_name_k, "corpus violated the gate's invariants over {} comparisons",
                                       reporter.comparison_count());
    }

    return sftest::report_pass(test_name_k);
}
