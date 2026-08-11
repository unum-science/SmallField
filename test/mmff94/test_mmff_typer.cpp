/**
 *  @file test/mmff94/test_mmff_typer.cpp
 *  @brief Cross-validate the native MMFF atom typer against RDKit's `MMFFMolProperties::getMMFFAtomType`.
 *
 *  On drug-like SMILES. The gate counts @b molecules, not atoms: a wrong type is a wrong molecule, and
 *  averaging six bad atoms over the ~889 000 in a 20 000-molecule sweep hides a whole structural class
 *  behind four nines. Both typers read RDKit's default aromaticity perception — `MMFFMolProperties` skips
 *  `MolOps::setMMFFAromaticity` on a molecule already marked `_MMFFSanitized`, which this test sets — so
 *  there is no aromaticity model difference to excuse a mismatch, and the curated set must come out exact.
 */

#include <cmath>  // `std::fabs`
#include <cstdio> // `printf`

#include <fstream> // `std::ifstream`
#include <memory>  // `std::unique_ptr`
#include <sstream> // RDKit's `AtomTyper.h` uses `std::stringstream` without including it
#include <string>  // `std::string`, `std::to_string`
#include <vector>  // `std::vector`

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

namespace sfparam = smallfield::parameters;
namespace sfmmff = smallfield::mmff94;
namespace sft = smallfield::types;

namespace smallfield::tests {
static sfparam::mmff94_tables_host_t tables_host_;
static sfparam::mmff94_tables_view_t tables_;
static int n_atoms_checked_ = 0;
static int n_atoms_mismatch_ = 0;
static int n_charges_checked_ = 0;
static int n_charges_mismatch_ = 0;
static int n_mols_tested_ = 0;
static int n_mols_mismatch_ = 0;

/** How many mismatching atoms to describe before going silent. A cap of three or ten turned a
 *  thirteen-molecule structural class into what read as a single stray molecule, so the default is
 *  generous; lower it locally when triaging a single molecule. A shared knobs header is on its way and
 *  will own this, at which point the constant here becomes its default. */
static std::size_t max_reported_mismatches_ = 64;

struct summary_t {
    int element = 0;
    std::uint8_t expected = 0;
    std::uint8_t got = 0;
    std::string smiles;
};

static std::vector<summary_t> mismatches_;

static void test_one_(std::string const &smiles, bool verbose) {
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
    if (!molecule) return;
    try {
        RDKit::MolOps::addHs(*molecule);
    }
    catch (...) {
        return;
    }
    if (molecule->getNumAtoms() > 128) return;

    // RDKit baseline (uses MMFF-specific aromaticity internally).
    if (!molecule->hasProp(RDKit::common_properties::_MMFFSanitized))
        molecule->setProp(RDKit::common_properties::_MMFFSanitized, 1, true);
    RDKit::MMFF::MMFFMolProperties props(*molecule);
    if (!props.isValid()) return;

    sft::molecule_descriptor_t descriptor;
    if (!sft::build_descriptor(*molecule, descriptor)) return;
    std::vector<std::uint8_t> native_atom_types(descriptor.num_atoms);
    sfmmff::native_compute_mmff_atom_types(descriptor, native_atom_types);
    std::vector<float> native_formal_charges(descriptor.num_atoms), native_partial_charges(descriptor.num_atoms);
    sfmmff::native_compute_mmff_formal_charges(descriptor, native_atom_types, native_formal_charges);
    sfmmff::native_compute_mmff_partial_charges(descriptor, tables_, native_atom_types, native_formal_charges,
                                                native_partial_charges);

    ++n_mols_tested_;
    bool this_mol_ok = true;
    for (std::uint32_t atom_index = 0; atom_index < descriptor.num_atoms; ++atom_index) {
        ++n_atoms_checked_;
        std::uint8_t const rdkit_atom_type = props.getMMFFAtomType(atom_index);
        if (rdkit_atom_type != native_atom_types[atom_index]) {
            ++n_atoms_mismatch_;
            this_mol_ok = false;
            if (verbose && mismatches_.size() < max_reported_mismatches_)
                mismatches_.push_back(summary_t {(int)descriptor.element[atom_index], rdkit_atom_type,
                                                 native_atom_types[atom_index], smiles});
        }
        if (rdkit_atom_type == native_atom_types[atom_index]) {
            ++n_charges_checked_;
            float const rdkit_partial_charge = (float)props.getMMFFPartialCharge(atom_index);
            float const diff = std::fabs(rdkit_partial_charge - native_partial_charges[atom_index]);
            if (diff > 1e-3f) {
                ++n_charges_mismatch_;
                this_mol_ok = false;
                if (verbose && mismatches_.size() < max_reported_mismatches_) {
                    summary_t mismatch;
                    mismatch.element = (int)descriptor.element[atom_index];
                    mismatch.smiles = std::string("chg ") + smiles + " (rdk=" + std::to_string(rdkit_partial_charge) +
                                      " nat=" + std::to_string(native_partial_charges[atom_index]) + ")";
                    mismatches_.push_back(mismatch);
                }
            }
        }
    }
    if (!this_mol_ok) ++n_mols_mismatch_;
}
} // namespace smallfield::tests

namespace sftest = smallfield::tests;

using sftest::max_reported_mismatches_;
using sftest::mismatches_;
using sftest::n_atoms_checked_;
using sftest::n_atoms_mismatch_;
using sftest::n_charges_checked_;
using sftest::n_charges_mismatch_;
using sftest::n_mols_mismatch_;
using sftest::n_mols_tested_;
using sftest::tables_;
using sftest::tables_host_;
using sftest::test_one_;

int main(int argc, char **argv) {
    std::printf("=== test_mmff_typer: native MMFF atom-typer + BCI charge gate ===\n");
    sfparam::extract_mmff94_tables_from_rdkit(tables_host_);
    tables_ = tables_host_.view();

    std::vector<std::string> probe = {
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "CC(=O)NC1=CC=C(O)C=C1",
        "[NH2+]=C(N)N",
        "CC(=O)[O-]",
        "c1cc[nH+]c(c1)",
        "C1CCOC1",
        "C1CC1",
        "c1ccc2[nH]ccc2c1",
        "O=N(=O)c1ccccc1",
        "CCO",
        "c1ccsc1",
        // Regression witnesses, one per way the typer used to read the descriptor wrong.
        "CC1=CC=CC=[N+]1[O-]",                                              // aromatic N-oxide: O is 32, not 35
        "CC1=C(C=CN=C1C2=CC(=CC(=C2)Cl)[NH+](O)[O-])CC#N",                  // hydroxylammonium O the typer refused
        "C1=CC=C2C=C(C=CC2=C1)NC3=CC#CC=C3",                                // triple bond inside an aromatic ring
        "COC(=O)C(CC1=CC=CC=C1)N=C2N[N+]#CS2",                              // the same, in a five-ring
        "CC(C)NC(=O)C1=CC(=C(C=C1)C2=C3C=CC(=O)C=C3OC4=C2C=CC(=C4)O)C(=O)O" // quinoid single bond, fused system
    };

    std::size_t const capacity = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 5000;
    for (auto const &smiles : probe) test_one_(smiles, true);
    if (argc > 1) {
        std::ifstream corpus_file(argv[1]);
        std::string line;
        std::size_t loaded = 0;
        while (std::getline(corpus_file, line) && loaded < capacity) {
            if (line.empty() || line[0] == '#') continue;
            auto separator_position = line.find_first_of(" \t");
            if (separator_position != std::string::npos) line.resize(separator_position);
            if (!line.empty()) {
                test_one_(line, mismatches_.size() < max_reported_mismatches_);
                ++loaded;
            }
        }
    }

    std::printf("molecules: %d tested, %d with ≥ 1 type or charge mismatch\n", n_mols_tested_, n_mols_mismatch_);
    std::printf("atoms: %d checked, %d type mismatches\n", n_atoms_checked_, n_atoms_mismatch_);
    std::printf("charges: %d checked, %d mismatches (> 1e-3)\n", n_charges_checked_, n_charges_mismatch_);
    double const atom_match = n_atoms_checked_ ? 100.0 * (1.0 - (double)n_atoms_mismatch_ / n_atoms_checked_) : 100.0;
    double const molecule_match = n_mols_tested_ ? 100.0 * (1.0 - (double)n_mols_mismatch_ / n_mols_tested_) : 100.0;
    double const charge_match = n_charges_checked_ ? 100.0 * (1.0 - (double)n_charges_mismatch_ / n_charges_checked_)
                                                   : 100.0;
    std::printf("atom match: %.4f %%, charge match: %.4f %%, molecule match: %.3f %%\n", atom_match, charge_match,
                molecule_match);
    for (auto const &mismatch : mismatches_) {
        std::printf("  element=%d rdk=%u nat=%u  [%s]\n", mismatch.element, mismatch.expected, mismatch.got,
                    mismatch.smiles.c_str());
    }
    double const molecule_mismatch_ppm = n_mols_tested_ ? 1.0e6 * (double)n_mols_mismatch_ / (double)n_mols_tested_
                                                        : 0.0;
    std::printf("molecule mismatch: %.1f ppm\n", molecule_mismatch_ppm);
    // Gate on molecules. The curated probe set carries a witness for every known typing hazard and must be
    // exact; an external corpus gets 100 ppm of slack for structures nobody has triaged yet, which is still
    // tight enough that a single structural class — the smallest seen was five molecules in 18 344 — fails it.
    double const max_molecule_mismatch_ppm = (argc > 1) ? 100.0 : 0.0;
    if (molecule_mismatch_ppm > max_molecule_mismatch_ppm) {
        std::printf("FAIL (%d of %d molecules mismatched, %.1f ppm > %.1f ppm)\n", n_mols_mismatch_, n_mols_tested_,
                    molecule_mismatch_ppm, max_molecule_mismatch_ppm);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
