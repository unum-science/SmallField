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

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/types/conformer_status.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

#include "test/test_helpers.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
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
    sft::conformer_status_t expected;
};

/**
 *  @brief The atom count the single-warp pipeline caps a molecule at, hydrogens included.
 *
 *  `parse_smiles` applies it after `addHs`, so the alkanes below straddle the cap by chain length:
 *  C63H128 is 191 atoms and fits, C64H130 is 194 and does not.
 */
static constexpr std::size_t atom_cap_k = sfp::max_atoms_per_molecule_k;

/** The ladder below opens one ring-bond label per rung beside the macrocycle's own, and SMILES has
 *  ninety-nine of them. */
static_assert(sft::max_rings_k + 11 < 99, "the ring cap outgrew the two-digit ring-bond labels");

/**
 *  @brief A carbon ladder RDKit perceives exactly @p rings rings in, for probing the ring cap.
 *
 *  Two parallel carbon chains walked as one macrocycle — down one side and back up the other — with
 *  every rung but the last closed by its own ring-bond label. A rung adds two carbons and three
 *  bonds, so a ladder straddling the ring cap sits under three quarters of the atom cap and the ring
 *  extent is the only one it can cross.
 */
static std::string carbon_ladder_(std::size_t rings) {
    std::string ladder = "C%99";
    for (std::size_t rung = 1; rung < rings; ++rung) ladder += fmt::format("C%{}", rung + 10);
    ladder += "CC"; // The far end, whose two carbons close the last ring against the chain itself.
    for (std::size_t rung = rings - 1; rung > 0; --rung) ladder += fmt::format("C%{}", rung + 10);
    return ladder + "C%99";
}

/** The whole curated corpus: one entry per path the gate can take. */
static element_gate_case_t const element_gate_cases_k[] = {
    // Happy path — drug-like organic, all MMFF94-supported elements.
    {"ethanol", "CCO", sft::conformer_status_t::success_k},
    {"benzene", "c1ccccc1", sft::conformer_status_t::success_k},
    {"aspirin", "CC(=O)Oc1ccccc1C(=O)O", sft::conformer_status_t::success_k},

    // Bug-report repro — yttrium (Z=39) is not in the MMFF94 atom-type table.
    {"yttrium-bracketed", "CC[Y]CC", sft::conformer_status_t::unsupported_element_k},

    // Other transition metals and lanthanides — no MMFF94 type.
    {"lone-iron", "[Fe]", sft::conformer_status_t::unsupported_element_k},
    {"lanthanum", "[La]", sft::conformer_status_t::unsupported_element_k},

    // SMILES parse failure — clearly malformed input.
    {"garbage", "not-a-smiles-string-((", sft::conformer_status_t::invalid_smiles_k},
};

/**
 *  @brief Whether `prepare_molecule_into` is entitled to answer with @p status.
 *
 *  Exhaustive over `conformer_status_t` and carrying no `default:`, so a status appended to that
 *  append-only enum is a `-Wswitch` warning here rather than a refusal this suite blames on the
 *  molecule that carried it.
 */
static bool gate_can_answer_(sft::conformer_status_t status) noexcept {
    switch (status) {
    // The three refusal points the gate walks: `parse_smiles`, then `build_descriptor` for each of
    // the flat descriptor's extents, then MMFF typing.
    case sft::conformer_status_t::success_k:
    case sft::conformer_status_t::invalid_smiles_k:
    case sft::conformer_status_t::too_many_atoms_k:
    case sft::conformer_status_t::too_many_bonds_k:
    case sft::conformer_status_t::too_many_rings_k:
    case sft::conformer_status_t::unsupported_element_k: return true;
    // Verdicts of stages downstream of the gate — the metric prepass, the warp pipeline and the MMFF
    // polish — plus the reserved value nothing produces. `batch_overflow_k` sits here too and is the one
    // that names no property of the molecule at all: it is what the wave sizer and the device pools
    // refuse, so the gate could not answer with it even in principle.
    case sft::conformer_status_t::stereo_failure_k:
    case sft::conformer_status_t::mmff_unconverged_k:
    case sft::conformer_status_t::preprocess_timeout_k:
    case sft::conformer_status_t::bounds_infeasible_k:
    case sft::conformer_status_t::nonphysical_energy_k:
    case sft::conformer_status_t::batch_overflow_k:
    // `slot_unwritten_k` names the absence of a device verdict, and the gate runs on the host before any
    // slot exists to be left unwritten.
    case sft::conformer_status_t::slot_unwritten_k: return false;
    }
    // A status byte no enumerator claims is nothing the gate answers with, so it borrows no verdict.
    return false;
}

#pragma endregion Curated Cases

#pragma region Checks

/**
 *  @brief Run one molecule through the gate and report its status and whether atoms survived.
 *
 *  @param[in] probe The molecule and the status expected of it.
 *  @param[in] scratch Reusable preparation buffer, cleared by `prepare_molecule_into` on entry.
 *  @param[inout] reporter Sink that counts and prints both comparisons.
 */
static void check_curated_case_(element_gate_case_t const &probe, sfp::etkdg_prepared_molecule_t &scratch,
                                diagnostic_reporter_t &reporter) {
    std::string const smiles {probe.smiles};
    auto const status = sfp::prepare_molecule_into(smiles, scratch);

    // A rejected molecule must leave nothing behind, so the caller cannot mistake a refusal for an
    // empty-but-accepted preparation and hand it to the device.
    bool const expected_atoms_present = probe.expected == sft::conformer_status_t::success_k;
    reporter.mismatch("element_gate", "molecule={} smiles={}", probe.label, probe.smiles)
        .field("status_code", static_cast<unsigned>(probe.expected), static_cast<unsigned>(status), exactly_equal)
        .field("atoms_present", expected_atoms_present, scratch.num_atoms != 0, exactly_equal);
}

/**
 *  @brief Check the two invariants that hold for a molecule nobody wrote an expectation for.
 *
 *  @param[in] smiles Corpus molecule, of unknown validity.
 *  @param[in] scratch Reusable preparation buffer.
 *  @param[inout] reporter Sink that counts and prints both comparisons.
 *  @return The status the gate answered with, for the census.
 */
static sft::conformer_status_t check_corpus_molecule_(std::string const &smiles,
                                                      sfp::etkdg_prepared_molecule_t &scratch,
                                                      diagnostic_reporter_t &reporter) {
    auto const status = sfp::prepare_molecule_into(smiles, scratch);
    bool const expected_atoms_present = status == sft::conformer_status_t::success_k;
    reporter.mismatch("element_gate", "molecule=corpus smiles={}", smiles)
        .field("status_answerable", true, gate_can_answer_(status), exactly_equal)
        .field("atoms_present", expected_atoms_present, scratch.num_atoms != 0, exactly_equal);
    return status;
}

/** One census slot per `conformer_status_t`, plus the trailing slot a byte no enumerator claims falls
 *  into — which `conformer_status_name` already names `unnamed_status`. */
static constexpr std::size_t census_slots_k = sft::conformer_status_count_k + 1;

/** Per-status census of a corpus run; the comparison counts belong to the reporter's own section. */
struct status_census_t {
    /** Indexed by the status itself, so a status appended to the enum counts under its own name
     *  rather than inflating a neighbour's column. */
    std::size_t counts[census_slots_k] = {};

    void count(sft::conformer_status_t status) noexcept {
        std::size_t const slot = static_cast<std::size_t>(status);
        ++counts[slot < sft::conformer_status_count_k ? slot : sft::conformer_status_count_k];
    }
};

/** One line of `status=count` for every verdict the corpus produced, ending with the molecule count so
 *  a run that classified nothing cannot read like one that classified everything. */
static void report_census_(status_census_t const &census) {
    std::size_t molecules = 0;
    fmt::print("NOTE element_gate verdicts");
    for (std::size_t slot = 0; slot < census_slots_k; ++slot) {
        molecules += census.counts[slot];
        if (census.counts[slot] == 0) continue;
        fmt::print(" {}={}", sft::conformer_status_name(static_cast<sft::conformer_status_t>(slot)),
                   census.counts[slot]);
    }
    fmt::println(" molecules={}", molecules);
}

#pragma endregion Checks

} // namespace smallfield::tests

int main(int argc, char **argv) {
    fmt::println("{}: MMFF94 element gate over curated rejections, then a corpus for the gate's invariants",
                 test_name_k);

    auto const selection = sftest::select_corpus(argc, argv, 10000, sftest::adversarial_corpus_path());
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", argv[0]);
        return 2;
    }

    sftest::diagnostic_reporter_t reporter {"expected", "observed"};
    reporter.limit_printed_lines(max_printed_lines_k);
    sfp::etkdg_prepared_molecule_t scratch {};

    for (auto const &probe : sftest::element_gate_cases_k) sftest::check_curated_case_(probe, scratch, reporter);

    // Each descriptor extent the gate refuses on is a boundary, so both are checked as one: the
    // longest alkane that fits and the shortest that does not, then the widest ladder whose SSSR
    // rings fit and the narrowest whose rings do not. Every bound is named from its library constant
    // rather than from a magic number.
    //
    // `too_many_bonds_k` gets no case: at four bonds per carbon a molecule under the atom cap tops
    // out at exactly `max_bonds_k`, so no real molecule crosses it and only a descriptor array bound
    // is being guarded.
    std::size_t const carbons_that_fit = (sftest::atom_cap_k - 2) / 3;
    std::string const alkane_at_cap(carbons_that_fit, 'C');
    std::string const alkane_over_cap(carbons_that_fit + 1, 'C');
    std::string const ladder_at_cap = sftest::carbon_ladder_(sft::max_rings_k);
    std::string const ladder_over_cap = sftest::carbon_ladder_(sft::max_rings_k + 1);
    sftest::element_gate_case_t const cap_cases[] = {
        {"alkane-at-cap", alkane_at_cap, sft::conformer_status_t::success_k},
        {"alkane-over-cap", alkane_over_cap, sft::conformer_status_t::too_many_atoms_k},
        {"ladder-at-ring-cap", ladder_at_cap, sft::conformer_status_t::success_k},
        {"ladder-over-ring-cap", ladder_over_cap, sft::conformer_status_t::too_many_rings_k},
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
        sftest::report_census_(census);

        if (!reporter.passed())
            return sftest::report_fail(test_name_k, "corpus violated the gate's invariants over {} comparisons",
                                       reporter.comparison_count());
    }

    return sftest::report_pass(test_name_k);
}
