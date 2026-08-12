/**
 *  @file test/preprocess/test_chirality.cpp
 *  @brief Police the device chirality check against conformers RDKit embedded from the same tags.
 *
 *  `validate_chirality_` answers one question per molecule: does every extracted chiral center realize the
 *  signed tetrahedral volume its target demands? Two comparisons per molecule police that answer, and the
 *  second is what keeps the first honest. An RDKit conformer embedded from the same tetrahedral tags must
 *  satisfy the targets `extract_chiral_centers` derived from them, so the upright verdict is `1`; inverting
 *  every target sign leaves the same geometry describing the opposite handedness, so the inverted verdict
 *  must be `0`. A kernel that answered `1` unconditionally passes the first and fails the second.
 *
 *  Coverage comes from a curated set of stereocenter-bearing drug molecules and an optional SMILES corpus
 *  named by `argv[1]` or by `SMALLFIELD_TEST_CORPUS`, capped by `argv[2]` or `SMALLFIELD_TEST_MAX_MOLS`.
 *  A named corpus that does not open is a failure, never an empty run.
 *
 *  Compiled with g++, C++20. CUDA kernels in test_chirality_kernels.cu.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int8_t`, `std::uint32_t`, `std::uint64_t`

#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <GraphMol/GraphMol.h>

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/types/views.hpp"

#include "test/preprocess/test_chirality_kernels.hpp"
#include "test/test_helpers.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_chirality";

/** A stereo-rich corpus can disagree at hundreds of molecules, and ten lines already name the pattern. */
static constexpr std::size_t max_printed_lines_k = 10;

#pragma region Molecule Walk

/** Buffers one molecule's check reuses, declared above the walk so a growth phase happens once. */
struct chirality_scratch_t {
    /** RDKit's conformer, one xyz triple per atom. */
    std::vector<float> atom_major;
    /** The same coordinates as three contiguous axis planes, which is what the device check reads. */
    std::vector<float> axis_major;
    std::vector<sft::chiral_center_descriptor_t> targets;
    std::vector<sft::chiral_center_descriptor_t> inverted_targets;
    sft::molecule_descriptor_t descriptor;
};

/**
 *  @brief Census of what the walk reached, and of what it had to decline.
 *
 *  Only what the reporter does not already hold: it owns every comparison and every mismatch, and
 *  `summarize` prints those, so counting them a second time here would be two schemes for one fact.
 *  The decline reasons stay apart because the walk already distinguishes them, and a corpus that
 *  declines every molecule reads very differently depending on which counter holds them.
 */
struct chirality_tally_t {
    /** Molecules whose targets reached the device. */
    std::uint64_t molecules = 0;
    /** Chiral centers those molecules carried. */
    std::uint64_t centers = 0;
    std::uint64_t unparsed = 0;
    std::uint64_t unembedded = 0;
    std::uint64_t oversized = 0;
    std::uint64_t achiral = 0;
    std::uint64_t device_refused = 0;
};

/** Molecules that never reached a comparison, however they got there. */
static std::uint64_t declined_(chirality_tally_t const &tally) {
    return tally.unparsed + tally.unembedded + tally.oversized + tally.achiral + tally.device_refused;
}

/** How far one molecule got, so the walk can bucket it without the check counting anything itself. */
enum class molecule_outcome_t {
    /** Both verdicts were compared against the device. */
    compared_k,
    /** RDKit refused the SMILES. */
    unparsed_k,
    /** ETKDG produced no conformer to read coordinates from. */
    unembedded_k,
    /** The molecule sits past the descriptor's fixed atom or bond extents. */
    oversized_k,
    /** No tetrahedral center survived `assignStereochemistry`. */
    achiral_k,
    /** The device refused the staging, so neither verdict was ever computed. */
    device_refused_k,
};

/**
 *  @brief Run one molecule's extracted chiral targets, and their inversion, through the device check.
 *
 *  @param[in] smiles Molecule to check, parsed and embedded here.
 *  @param[inout] scratch Buffers reused across molecules.
 *  @param[inout] reporter Receives the two verdict comparisons.
 *  @param[out] num_centers Chiral centers compared, zero unless the outcome is `compared_k`.
 *  @return How far the molecule got.
 */
static molecule_outcome_t check_molecule_(std::string const &smiles, chirality_scratch_t &scratch,
                                          sftest::diagnostic_reporter_t &reporter, std::uint32_t &num_centers) {

    num_centers = 0;

    // `SmilesToMol` throws on some malformed corpus input rather than returning null, so the parse is
    // guarded rather than merely null-checked.
    std::unique_ptr<RDKit::RWMol> molecule;
    try {
        molecule = sftest::mol_from_smiles(smiles);
    }
    catch (...) {
        molecule.reset();
    }
    if (!molecule) return molecule_outcome_t::unparsed_k;
    if (!sftest::embed_molecule(*molecule)) return molecule_outcome_t::unembedded_k;
    // `build_descriptor` refuses a molecule past its fixed atom and bond extents, and leaves its output
    // stale rather than empty when it does, so the refusal is read before the extraction trusts it.
    if (sft::build_descriptor(*molecule, scratch.descriptor) != sft::conformer_status_t::success_k)
        return molecule_outcome_t::oversized_k;

    sfp::extract_chiral_centers(*molecule, scratch.descriptor, scratch.targets);
    // An achiral molecule satisfies the upright and the inverted targets alike — the warp ballot over zero
    // centers is true either way — so it would report two agreements it never tested.
    if (scratch.targets.empty()) return molecule_outcome_t::achiral_k;

    std::uint32_t const num_atoms = molecule->getNumAtoms();
    auto const atom_major = sft::reset_coordinates_aos<sft::coordinate_role_t::position_k>(scratch.atom_major,
                                                                                           num_atoms);
    auto const axis_major = sft::reset_coordinates_soa<sft::coordinate_role_t::position_k, sft::conformer_dimensions_k>(
        scratch.axis_major, num_atoms);
    sftest::read_conformer_into(*molecule, atom_major);
    sft::transpose_into(atom_major, axis_major);

    scratch.inverted_targets.assign(scratch.targets.begin(), scratch.targets.end());
    for (auto &center : scratch.inverted_targets) center.target_sign = static_cast<std::int8_t>(-center.target_sign);

    auto const upright = sftest::gpu_validate_chirality(axis_major, scratch.targets);
    auto const inverted = sftest::gpu_validate_chirality(axis_major, scratch.inverted_targets);
    // A refused staging is not an inverted stereocenter, so it never reaches the comparison below.
    if (upright == sftest::chirality_verdict_t::unavailable_k || inverted == sftest::chirality_verdict_t::unavailable_k)
        return molecule_outcome_t::device_refused_k;

    num_centers = static_cast<std::uint32_t>(scratch.targets.size());
    reporter.mismatch("chirality", "smiles={} centers={}", smiles, num_centers)
        .field("satisfied", true, upright == sftest::chirality_verdict_t::satisfied_k, sftest::exactly_equal)
        .field("satisfied_inverted", false, inverted == sftest::chirality_verdict_t::satisfied_k,
               sftest::exactly_equal);

    return molecule_outcome_t::compared_k;
}

/** Check every molecule of one section, bucketing each by how far it got. */
static void walk_(custd::span<std::string const> corpus, chirality_scratch_t &scratch,
                  sftest::diagnostic_reporter_t &reporter, chirality_tally_t &tally) {
    for (auto const &smiles : corpus) {
        std::uint32_t num_centers = 0;
        switch (check_molecule_(smiles, scratch, reporter, num_centers)) {
        case molecule_outcome_t::unparsed_k: ++tally.unparsed; break;
        case molecule_outcome_t::unembedded_k: ++tally.unembedded; break;
        case molecule_outcome_t::oversized_k: ++tally.oversized; break;
        case molecule_outcome_t::achiral_k: ++tally.achiral; break;
        case molecule_outcome_t::device_refused_k: ++tally.device_refused; break;
        case molecule_outcome_t::compared_k:
            ++tally.molecules;
            tally.centers += num_centers;
            break;
        }
    }
}

/** Print one section's census beneath the section's own `PASS`/`FAIL` line. */
static void report_(std::string_view heading, chirality_tally_t const &tally) {
    fmt::println("  {}: molecules {}, centers {}, declined {} "                               //
                 "({} unparsed, {} unembedded, {} oversized, {} achiral, {} device-refused)", //
                 heading, fmt::group_digits(tally.molecules), fmt::group_digits(tally.centers),
                 fmt::group_digits(declined_(tally)), fmt::group_digits(tally.unparsed),
                 fmt::group_digits(tally.unembedded), fmt::group_digits(tally.oversized),
                 fmt::group_digits(tally.achiral), fmt::group_digits(tally.device_refused));
}

/**
 *  @brief Drug molecules whose stereocenters survive `assignStereochemistry`, built once at start-up.
 *
 *  Glucose carries four centers, so the set also exercises a molecule whose warp ballot spans several lanes.
 */
static std::vector<std::string> curated_chiral_smiles_() {
    return {
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",        // (S)-ibuprofen
        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", // glucose
        "C[C@@H](O)CC",                           // (R)-2-butanol
        "N[C@@H](CC(=O)O)C(=O)O",                 // L-aspartic acid
    };
}

#pragma endregion Molecule Walk

int main(int argc, char **argv) {
    auto const selection = sftest::select_corpus(argc, argv, 200);
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", argv[0]);
        return 2;
    }

    fmt::println("{}: device tetrahedral parity against RDKit-embedded conformers", test_name_k);

    sftest::diagnostic_reporter_t reporter("expected", "device");
    reporter.limit_printed_lines(max_printed_lines_k);

    chirality_scratch_t scratch;
    chirality_tally_t curated;
    auto const curated_smiles = curated_chiral_smiles_();
    walk_(curated_smiles, scratch, reporter, curated);
    report_("curated", curated);
    reporter.summarize("curated");
    sf_verify_(declined_(curated) == 0, "every curated molecule must parse, embed, and yield a chiral center");
    sf_verify_(curated.centers > curated.molecules, "the curated set must include a multi-center molecule");

    if (selection.knob == sftest::corpus_knob_t::named_k) {
        // Naming a corpus is a request to validate against it, so an empty result is a failure and not an
        // empty corpus: it would otherwise run zero comparisons and satisfy `mismatches == 0` vacuously.
        std::vector<std::string> corpus;
        auto const outcome = sftest::load_smiles_into(selection.path, selection.max_molecules, corpus);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), selection.path);
        fmt::println("{}: corpus {}, {} SMILES", test_name_k, selection.path, fmt::group_digits(corpus.size()));

        chirality_tally_t corpus_tally;
        walk_(corpus, scratch, reporter, corpus_tally);
        report_("corpus", corpus_tally);
        // The same reasoning one level down: every molecule may have been declined, and `summarize` is what
        // separates a section of zero comparisons from a section of zero mismatches.
        reporter.summarize("corpus");
    }

    if (!reporter.passed())
        return sftest::report_fail(test_name_k, "{} mismatches over {} comparisons", reporter.mismatch_count(),
                                   reporter.comparison_count());
    return sftest::report_pass(test_name_k);
}
