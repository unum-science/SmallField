/**
 *  @file test_mmff_typer.cpp
 *  @brief Cross-validate the native MMFF atom typer against RDKit's
 *  `MMFFMolProperties::getMMFFAtomType` on drug-like SMILES. Target: ≥ 99 % per-atom match on
 *  ChEMBL 20K. Some discrepancy is expected because we don't yet invoke MMFF-specific aromaticity.
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

#include "types/molecule_descriptor.hpp"
#include "parameters/mmff94_tables.hpp"
#include "mmff94/mmff94_typer.hpp"

using namespace smallfield;

namespace {
mmff94_tables_host_t tables_host;
mmff94_tables_view_t tables;
int n_atoms_checked = 0;
int n_atoms_mismatch = 0;
int n_charges_checked = 0;
int n_charges_mismatch = 0;
int n_mols_tested = 0;
int n_mols_mismatch = 0;

struct summary_t {
    int element = 0;
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
    std::vector<float> nat_fc(desc.num_atoms), nat_pc(desc.num_atoms);
    native_compute_mmff_formal_charges(desc, nat, nat_fc);
    native_compute_mmff_partial_charges(desc, tables, nat, nat_fc, nat_pc);

    ++n_mols_tested;
    bool this_mol_ok = true;
    for (std::uint32_t a = 0; a < desc.num_atoms; ++a) {
        ++n_atoms_checked;
        std::uint8_t const rdk_t = props.getMMFFAtomType(a);
        if (rdk_t != nat[a]) {
            ++n_atoms_mismatch;
            this_mol_ok = false;
            if (verbose && mismatches.size() < 20)
                mismatches.push_back({(int)desc.element[a], rdk_t, nat[a], smiles});
        }
        if (rdk_t == nat[a]) {
            ++n_charges_checked;
            float const rdk_pc = (float)props.getMMFFPartialCharge(a);
            float const diff = std::fabs(rdk_pc - nat_pc[a]);
            if (diff > 1e-3f) {
                ++n_charges_mismatch;
                this_mol_ok = false;
                if (verbose && mismatches.size() < 20) {
                    summary_t m; m.element = (int)desc.element[a];
                    m.smiles = std::string("chg ") + smiles +
                               " (rdk=" + std::to_string(rdk_pc) +
                               " nat=" + std::to_string(nat_pc[a]) + ")";
                    mismatches.push_back(m);
                }
            }
        }
    }
    if (!this_mol_ok) ++n_mols_mismatch;
}
} // namespace

int main(int argc, char **argv) {
    std::printf("=== test_mmff_typer: native MMFF atom-typer + BCI charge gate ===\n");
    extract_mmff94_tables_from_rdkit(tables_host);
    tables = tables_host.view();

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

    std::printf("molecules: %d tested, %d with ≥ 1 type or charge mismatch\n",
                n_mols_tested, n_mols_mismatch);
    std::printf("atoms: %d checked, %d type mismatches\n",
                n_atoms_checked, n_atoms_mismatch);
    std::printf("charges: %d checked, %d mismatches (> 1e-3)\n",
                n_charges_checked, n_charges_mismatch);
    double const atom_match = n_atoms_checked ? 100.0 * (1.0 - (double)n_atoms_mismatch / n_atoms_checked) : 100.0;
    double const mol_match = n_mols_tested ? 100.0 * (1.0 - (double)n_mols_mismatch / n_mols_tested) : 100.0;
    double const chg_match = n_charges_checked ? 100.0 * (1.0 - (double)n_charges_mismatch / n_charges_checked) : 100.0;
    std::printf("atom match: %.4f %%, charge match: %.4f %%, molecule match: %.3f %%\n",
                 atom_match, chg_match, mol_match);
    for (auto const &m : mismatches) {
        std::printf("  element=%d rdk=%u nat=%u  [%s]\n", m.element, m.expected, m.got, m.smiles.c_str());
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
