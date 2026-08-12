/**
 *  @file test/mmff94/test_mmff_typer.cpp
 *  @brief Cross-validate the native MMFF atom typer against RDKit's `MMFFMolProperties::getMMFFAtomType`.
 *
 *  On drug-like SMILES. The gate counts @b molecules, not atoms: a wrong type is a wrong molecule, and
 *  averaging six bad atoms over the ~889 000 in a 20 000-molecule sweep hides a whole structural class
 *  behind four nines.
 *
 *  The two typers do @b not automatically read the same aromaticity. `MMFFMolProperties` guards only its
 *  Kekulize behind `_MMFFSanitized` and calls `MolOps::setMMFFAromaticity` unconditionally, so marking a
 *  molecule sanitized suppresses the Kekulize and nothing else. Whichever side snapshots the molecule
 *  before that call sees a different one, and the curated set comes out exact only because none of its
 *  seventeen probes is a molecule whose perception moves.
 */

#include <cmath>   // `std::fabs`
#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::fflush`

#include <memory>      // `std::unique_ptr`
#include <sstream>     // RDKit's `AtomTyper.h` uses `std::stringstream` without including it
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <boost/shared_ptr.hpp>
#include <RDGeneral/types.h>
#include <ForceField/MMFF/Params.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>

#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_rdkit.hpp"
#include "smallfield/mmff94/mmff94_typer.hpp"

#include "test/test_helpers.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfapi = smallfield::api;
namespace sfparam = smallfield::parameters;
namespace sfmmff = smallfield::mmff94;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_mmff_typer";

namespace smallfield::tests {

/** A partial charge is summed in `float` here against RDKit's `double`, so it carries an absolute tolerance. */
static constexpr auto charge_within_tolerance_ = [](float rdkit_value, float native_value) {
    return std::fabs(rdkit_value - native_value) <= 1e-3f;
};

/** How many mismatching fields either section prints before the rest are only counted. A cap of three
 *  or ten turned a thirteen-molecule structural class into what read as a single stray molecule, so
 *  the budget is generous; lower it locally when triaging one molecule. */
static constexpr std::size_t max_reported_mismatches_k = 64;

/** One reporter per compared section, so each closes with its own `compared=` count. */
struct typer_reporters_t {
    diagnostic_reporter_t atom_type {"rdkit", "native"};
    diagnostic_reporter_t partial_charge {"rdkit", "native"};

    /** Cap what each section prints, so one pathological corpus cannot bury the log. */
    void limit_printed_lines(std::size_t maximum) noexcept {
        atom_type.limit_printed_lines(maximum);
        partial_charge.limit_printed_lines(maximum);
    }

    /** Close both sections; true when each compared something and every field agreed. */
    bool summarize_sections() {
        atom_type.summarize("atom_type");
        partial_charge.summarize("partial_charge");
        return atom_type.passed() && partial_charge.passed();
    }
};

/** Molecule counts for the sweep, threaded by reference so the probe and the corpus aggregate into one. */
struct typer_tally_t {
    std::uint64_t molecules = 0;
    std::uint64_t molecules_mismatched = 0;
};

/** Reusable per-molecule scratch for the native typer, so the sweep allocates once rather than per molecule. */
struct native_typing_scratch_t {
    std::vector<std::uint8_t> atom_types;
    std::vector<float> formal_charges;
    std::vector<float> partial_charges;

    void resize(std::uint32_t num_atoms) {
        atom_types.assign(num_atoms, std::uint8_t {0});
        formal_charges.assign(num_atoms, 0.0f);
        partial_charges.assign(num_atoms, 0.0f);
    }
};

/**
 *  @brief Type one SMILES both ways and fold the outcome into @p tally.
 *
 *  A molecule RDKit itself cannot type is not a disagreement, so it leaves no trace in the tally.
 *  The charge section compares only the atoms whose type already agrees, which is why its `compared=`
 *  count is smaller than the atom section's and cannot be derived from it.
 */
static void test_one_(std::string const &smiles, sfparam::mmff94_tables_view_t const &tables,
                      typer_reporters_t &reporters, native_typing_scratch_t &scratch, typer_tally_t &tally) {
    std::unique_ptr<RDKit::RWMol> molecule;
    try {
        RDKit::SmilesParserParams parser_params;
        parser_params.sanitize = true;
        parser_params.removeHs = false;
        parser_params.strictCXSMILES = false;
        molecule.reset(RDKit::SmilesToMol(smiles, parser_params));
    }
    catch (...) {
        return;
    }
    if (!molecule) return; // the parser refused it
    try {
        RDKit::MolOps::addHs(*molecule);
    }
    catch (...) {
        return;
    }
    if (molecule->getNumAtoms() > 128) return; // beyond the descriptor's fixed capacity

    // RDKit baseline, which uses MMFF-specific aromaticity internally.
    if (!molecule->hasProp(RDKit::common_properties::_MMFFSanitized))
        molecule->setProp(RDKit::common_properties::_MMFFSanitized, 1, true);
    RDKit::MMFF::MMFFMolProperties props(*molecule);
    if (!props.isValid()) return; // RDKit cannot type it either, so there is nothing to compare

    sft::molecule_descriptor_t descriptor;
    // an element or a bond order we do not carry
    if (sft::build_descriptor(*molecule, descriptor) != sfapi::conformer_status_t::success_k) return;
    scratch.resize(descriptor.num_atoms);
    sfmmff::native_compute_mmff_atom_types(descriptor, scratch.atom_types);
    sfmmff::native_compute_mmff_formal_charges(descriptor, scratch.atom_types, scratch.formal_charges);
    sfmmff::native_compute_mmff_partial_charges(descriptor, tables, scratch.atom_types, scratch.formal_charges,
                                                scratch.partial_charges);

    ++tally.molecules;
    std::size_t mismatching_atoms = 0;
    for (std::uint32_t atom_index = 0; atom_index < descriptor.num_atoms; ++atom_index) {
        std::uint8_t const rdkit_atom_type = props.getMMFFAtomType(atom_index);
        auto atom = reporters.atom_type.mismatch("mmff_typer", "smiles={} atom={} element={}", smiles, atom_index,
                                                 descriptor.element[atom_index]);
        atom.field("atom_type", rdkit_atom_type, scratch.atom_types[atom_index], exactly_equal);
        bool const types_agree = atom.reported_field_count() == 0;
        mismatching_atoms += !types_agree;
        if (!types_agree) continue; // a wrong type gives a wrong charge, so the charge says nothing

        auto charge = reporters.partial_charge.mismatch("mmff_typer", "smiles={} atom={} element={}", smiles,
                                                        atom_index, descriptor.element[atom_index]);
        charge.field("partial_charge", static_cast<float>(props.getMMFFPartialCharge(atom_index)),
                     scratch.partial_charges[atom_index], charge_within_tolerance_);
        mismatching_atoms += charge.reported_field_count() != 0;
    }
    if (mismatching_atoms != 0) ++tally.molecules_mismatched;
}

} // namespace smallfield::tests

int main(int argc, char **argv) {
    fmt::println("{}: native MMFF atom-typer + BCI charge gate vs RDKit", test_name_k);

    auto const corpus = sftest::select_corpus(argc, argv, 5000);
    if (corpus.knob == sftest::corpus_knob_t::refused_k) {
        std::fflush(stdout);
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", test_name_k);
        return 2;
    }

    sfparam::mmff94_tables_host_t tables_host;
    sfparam::extract_mmff94_tables_from_rdkit(tables_host);
    sfparam::mmff94_tables_view_t const tables = tables_host.view();

    std::vector<std::string> const probe = {
        "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "CC(=O)NC1=CC=C(O)C=C1", "[NH2+]=C(N)N", "CC(=O)[O-]", "c1cc[nH+]c(c1)", "C1CCOC1", "C1CC1", "c1ccc2[nH]ccc2c1",
        "O=N(=O)c1ccccc1", "CCO", "c1ccsc1",
        // Regression witnesses, one per way the typer used to read the descriptor wrong.
        "CC1=CC=CC=[N+]1[O-]",                                              // aromatic N-oxide: O is 32, not 35
        "CC1=C(C=CN=C1C2=CC(=CC(=C2)Cl)[NH+](O)[O-])CC#N",                  // hydroxylammonium O the typer refused
        "C1=CC=C2C=C(C=CC2=C1)NC3=CC#CC=C3",                                // triple bond inside an aromatic ring
        "COC(=O)C(CC1=CC=CC=C1)N=C2N[N+]#CS2",                              // the same, in a five-ring
        "CC(C)NC(=O)C1=CC(=C(C=C1)C2=C3C=CC(=O)C=C3OC4=C2C=CC(=C4)O)C(=O)O" // quinoid single bond, fused system
    };

    // A named corpus that does not open is a failure, never an empty run: the ppm gate below is
    // loosened for external structures, and loosening it while comparing nothing passes everything.
    std::vector<std::string> external;
    if (corpus.knob == sftest::corpus_knob_t::named_k) {
        auto const outcome = sftest::load_smiles_into(corpus.path, corpus.max_molecules, external);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), corpus.path);
    }
    fmt::println("probing {} curated + {} external SMILES", fmt::group_digits(probe.size()),
                 fmt::group_digits(external.size()));

    sftest::typer_reporters_t reporters;
    reporters.limit_printed_lines(sftest::max_reported_mismatches_k);
    sftest::typer_tally_t tally;
    sftest::native_typing_scratch_t scratch;
    for (auto const &smiles : probe) sftest::test_one_(smiles, tables, reporters, scratch, tally);
    for (auto const &smiles : external) sftest::test_one_(smiles, tables, reporters, scratch, tally);

    fmt::println("molecules: {} tested, {} with at least one type or charge mismatch",
                 fmt::group_digits(tally.molecules), fmt::group_digits(tally.molecules_mismatched));
    bool const sections_passed = reporters.summarize_sections();

    std::size_t const atoms_checked = reporters.atom_type.comparison_count();
    std::size_t const charges_checked = reporters.partial_charge.comparison_count();
    double const atom_match = atoms_checked
                                  ? 100.0 * (1.0 - (double)reporters.atom_type.mismatch_count() / (double)atoms_checked)
                                  : 100.0;
    double const charge_match = charges_checked ? 100.0 * (1.0 - (double)reporters.partial_charge.mismatch_count() /
                                                                     (double)charges_checked)
                                                : 100.0;
    double const molecule_match = tally.molecules
                                      ? 100.0 * (1.0 - (double)tally.molecules_mismatched / (double)tally.molecules)
                                      : 100.0;
    fmt::println("atom match: {:.3f} %, charge match: {:.3f} %, molecule match: {:.3f} %", atom_match, charge_match,
                 molecule_match);

    double const molecule_mismatch_ppm = tally.molecules
                                             ? 1.0e6 * (double)tally.molecules_mismatched / (double)tally.molecules
                                             : 0.0;
    fmt::println("molecule mismatch: {:.3f} ppm", molecule_mismatch_ppm);
    // Gate on molecules. The curated probe set carries a witness for every known typing hazard and must be
    // exact; a corpus that actually loaded gets 100 ppm of slack for structures nobody has triaged yet,
    // which is still tight enough that a single structural class — the smallest seen was five molecules in
    // 18 344 — fails it. Keying the slack on whether molecules arrived rather than on `argc` is what stops
    // a mistyped path from buying the slack while comparing no external molecule at all.
    double const max_molecule_mismatch_ppm = external.empty() ? 0.0 : 100.0;
    if (molecule_mismatch_ppm > max_molecule_mismatch_ppm)
        return sftest::report_fail(test_name_k, "{} of {} molecules mismatched, {:.3f} ppm > {:.3f} ppm",
                                   tally.molecules_mismatched, tally.molecules, molecule_mismatch_ppm,
                                   max_molecule_mismatch_ppm);
    if (!sections_passed)
        return sftest::report_fail(test_name_k, "a comparison section mismatched or compared nothing at all");
    return sftest::report_pass(test_name_k);
}
