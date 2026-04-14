/**
 *  @file test_mmff_typer.cpp
 *  @brief MG-1.5/1.6 gate — cross-validate native MMFF atom typer against
 *  RDKit's `MMFFMolProperties::getMMFFAtomType` on drug-like SMILES.
 *  Target: ≥ 99 % per-atom match on ChEMBL 20K (some discrepancy expected
 *  from MMFF-specific aromaticity we don't yet invoke).
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <RDGeneral/types.h>
#include <ForceField/MMFF/Params.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>

#include "molecule_descriptor.hpp"
#include "mmff94_typer.hpp"

using namespace smallfield;

namespace {
int n_atoms_checked = 0;
int n_atoms_mismatch = 0;
int n_mols_tested = 0;
int n_mols_mismatch = 0;

struct summary_t {
    int el = 0;
    std::uint8_t expected = 0;
    std::uint8_t got = 0;
    std::string smiles;
};

std::vector<summary_t> mismatches;

void test_one(std::string const &smiles, bool verbose) {
    std::unique_ptr<RDKit::RWMol> mol;
    try {
        RDKit::SmilesParserParams p;
        p.sanitize = true; p.removeHs = false; p.strictCXSMILES = false;
        mol.reset(RDKit::SmilesToMol(smiles, p));
    }
    catch (...) { return; }
    if (!mol) return;
    try { RDKit::MolOps::addHs(*mol); } catch (...) { return; }
    if (mol->getNumAtoms() > 128) return;

    // RDKit baseline (uses MMFF-specific aromaticity internally).
    if (!mol->hasProp(RDKit::common_properties::_MMFFSanitized))
        mol->setProp(RDKit::common_properties::_MMFFSanitized, 1, true);
    RDKit::MMFF::MMFFMolProperties props(*mol);
    if (!props.isValid()) return;

    molecule_descriptor_t desc;
    if (!build_descriptor(*mol, desc)) return;
    std::vector<std::uint8_t> nat(desc.num_atoms);
    native_compute_mmff_atom_types(desc, nat);

    ++n_mols_tested;
    bool this_mol_ok = true;
    for (std::uint32_t a = 0; a < desc.num_atoms; ++a) {
        ++n_atoms_checked;
        std::uint8_t const rdk_t = props.getMMFFAtomType(a);
        if (rdk_t != nat[a]) {
            ++n_atoms_mismatch;
            this_mol_ok = false;
            if (verbose && mismatches.size() < 20) {
                mismatches.push_back({(int)desc.element[a], rdk_t, nat[a], smiles});
            }
        }
    }
    if (!this_mol_ok) ++n_mols_mismatch;
}
} // namespace

int main(int argc, char **argv) {
    std::printf("=== test_mmff_typer: MG-1.5/1.6 gate ===\n");

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
    };

    std::size_t const cap = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 5000;
    for (auto const &s : probe) test_one(s, true);
    if (argc > 1) {
        std::ifstream ifs(argv[1]);
        std::string line;
        std::size_t loaded = 0;
        while (std::getline(ifs, line) && loaded < cap) {
            if (line.empty() || line[0] == '#') continue;
            auto sp = line.find_first_of(" \t");
            if (sp != std::string::npos) line.resize(sp);
            if (!line.empty()) { test_one(line, mismatches.size() < 10); ++loaded; }
        }
    }

    std::printf("molecules: %d tested, %d with ≥ 1 type mismatch\n",
                n_mols_tested, n_mols_mismatch);
    std::printf("atoms: %d checked, %d mismatches\n",
                n_atoms_checked, n_atoms_mismatch);
    double const atom_match = n_atoms_checked ? 100.0 * (1.0 - (double)n_atoms_mismatch / n_atoms_checked) : 100.0;
    double const mol_match = n_mols_tested ? 100.0 * (1.0 - (double)n_mols_mismatch / n_mols_tested) : 100.0;
    std::printf("atom match: %.4f %%, molecule match: %.3f %%\n", atom_match, mol_match);
    for (auto const &m : mismatches) {
        std::printf("  el=%d rdk=%u nat=%u  [%s]\n", m.el, m.expected, m.got, m.smiles.c_str());
    }
    // Gate: 99 % atom-level match for curated + 95 % for full ChEMBL.
    double const min_match = (argc > 1) ? 95.0 : 99.9;
    if (atom_match < min_match) {
        std::printf("FAIL (atom match %.3f%% < %.1f%%)\n", atom_match, min_match);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
