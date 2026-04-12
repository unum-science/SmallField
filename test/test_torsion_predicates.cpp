/**
 *  @file test_torsion_predicates.cpp
 *  @brief Cross-validate the hand-written recursive-SMARTS predicates in
 *  @ref csd_torsion_predicates.hpp against RDKit's SubstructMatch.
 *
 *  For every predicate we compile the matching SMARTS body through
 *  `RDKit::SmartsToMol`, run `SubstructMatch` against each test molecule, and
 *  collect the set of root-atom indices that RDKit reports. We then call the
 *  native predicate on every atom and assert the boolean answer matches the
 *  RDKit truth set, atom-for-atom, across ≥200 ChEMBL-derived molecules.
 *
 *  Any divergence means the predicate has the wrong semantics — debug the
 *  predicate, never weaken the assertion.
 */

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include "csd_torsion_predicates.hpp"
#include "molecule_descriptor.hpp"
#include "test_helpers.hpp"

using namespace smallfield;
using namespace smallfield::test;

namespace {

struct predicate_case_t {
    char const *name;
    char const *smarts;
    csd_predicate_t id;
};

/**
 *  @brief Load up to @p max_count SMILES strings from @p path, one per line.
 *  Empty lines and lines starting with '#' are skipped.
 */
std::vector<std::string> load_smiles(char const *path, std::size_t max_count) {
    std::vector<std::string> out;
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "FAIL: cannot open %s\n", path);
        std::exit(1);
    }
    std::string line;
    while (out.size() < max_count && std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        // Strip trailing whitespace.
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
        // First token only (some SMILES files have a name after the SMILES).
        std::size_t const sp = line.find_first_of(" \t");
        if (sp != std::string::npos) line.resize(sp);
        if (!line.empty()) out.push_back(std::move(line));
    }
    return out;
}

/**
 *  @brief Compute the set of atom indices that are the root (first) atom of any
 *  `SubstructMatch` hit for @p query in @p mol.
 */
std::unordered_set<std::uint32_t> rdkit_root_match_set(RDKit::ROMol const &mol, RDKit::ROMol const &query) {
    std::vector<RDKit::MatchVectType> matches;
    RDKit::SubstructMatch(mol, query, matches, /*uniquify=*/false, /*recursionPossible=*/true);
    std::unordered_set<std::uint32_t> roots;
    for (auto const &m : matches) {
        if (m.empty()) continue;
        roots.insert(static_cast<std::uint32_t>(m[0].second));
    }
    return roots;
}

/**
 *  @brief Cross-validate a single predicate against RDKit for one molecule.
 *  Asserts via SF_TEST_ASSERT that the predicate's verdict matches RDKit for
 *  every atom.
 */
void validate_predicate_for_molecule(predicate_case_t const &pc, RDKit::ROMol const &mol,
                                     molecule_descriptor_t const &desc, std::string const &smiles) {
    std::unique_ptr<RDKit::ROMol const> query(RDKit::SmartsToMol(pc.smarts));
    SF_TEST_ASSERT(query != nullptr, pc.smarts);

    auto const rdkit_roots = rdkit_root_match_set(mol, *query);

    std::uint32_t const N = desc.num_atoms;
    for (std::uint32_t a = 0; a < N; ++a) {
        bool const native_hit = csd_predicate_check(desc, a, pc.id);
        bool const rdkit_hit = (rdkit_roots.count(a) != 0);
        if (native_hit != rdkit_hit) {
            std::fprintf(
                stderr,
                "Predicate mismatch: %s (SMARTS \"%s\") atom=%u native=%d rdkit=%d\n" "  element=%u " "aromatic=%d " "d" "e" "g" "r" "e" "e" "=" "%" "u nH=%u " "hyb=%u\n" "  " "SMILE" "S: " "%s\n",
                pc.name, pc.smarts, a, native_hit ? 1 : 0, rdkit_hit ? 1 : 0, desc.element[a],
                atom_has_flag(desc, a, atom_flag_aromatic_k) ? 1 : 0, desc.degree[a], desc.num_hydrogens[a],
                desc.hybridization[a], smiles.c_str());
            SF_TEST_ASSERT(false, "predicate mismatch vs RDKit");
        }
    }
}

} // namespace

int main() {
    std::printf("Cross-validating CSD torsion predicates against RDKit SubstructMatch...\n");

    predicate_case_t const cases[] = {
        {"c_eq_o", "C=O", csd_predicate_c_eq_o_k},
        {"s_db_o", "S(=O)", csd_predicate_s_db_o_k},
        {"cx3_eq_o", "[CX3]=O", csd_predicate_cx3_eq_o_k},
        {"c_eq_o_branch", "[C](=O)", csd_predicate_c_eq_o_branch_k},
        {"nx3h1", "[NX3H1]", csd_predicate_nx3h1_k},
        {"nx3h2", "[NX3H2]", csd_predicate_nx3h2_k},
        {"c_nb_nx3", "C([NX3])", csd_predicate_c_nb_nx3_k},
        {"cx3_nb_c", "[CX3]([C])", csd_predicate_cx3_nb_c_k},
        {"ch0_nb_cl", "[cH0](Cl)", csd_predicate_ch0_nb_cl_k},
    };

    // Also sanity-check that the built-in standard molecules work.
    std::vector<std::string> smiles_list = standard_test_smiles();

    // Append up to 200 molecules from the validation SMILES file.
    constexpr std::size_t chembl_target_count = 200;
    auto chembl = load_smiles("/tmp/chembl_valid.smi", chembl_target_count * 2);
    for (auto &s : chembl) smiles_list.push_back(std::move(s));

    std::size_t total_validated = 0;
    std::size_t skipped = 0;
    std::size_t validated_per_pred[std::size(cases)] = {};

    for (auto const &smi : smiles_list) {
        // Parse with sanitization; skip on failure or size overflow.
        std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smi));
        if (!mol) {
            ++skipped;
            continue;
        }
        try {
            RDKit::MolOps::addHs(*mol);
        }
        catch (...) {
            ++skipped;
            continue;
        }
        if (mol->getNumAtoms() > max_atoms_k || mol->getNumBonds() > max_bonds_k) {
            ++skipped;
            continue;
        }

        molecule_descriptor_t desc;
        if (!build_descriptor(*mol, desc)) {
            ++skipped;
            continue;
        }

        for (std::size_t i = 0; i < std::size(cases); ++i) {
            validate_predicate_for_molecule(cases[i], *mol, desc, smi);
            ++validated_per_pred[i];
        }
        ++total_validated;

        if (total_validated >= chembl_target_count + standard_test_smiles().size()) break;
    }

    std::printf("\nValidated %zu molecules (skipped %zu).\n", total_validated, skipped);
    for (std::size_t i = 0; i < std::size(cases); ++i)
        std::printf("  %-16s  ok on %zu / %zu molecules\n", cases[i].name, validated_per_pred[i], total_validated);

    SF_TEST_ASSERT(total_validated >= 200, "fewer than 200 molecules validated — bad test corpus?");
    std::printf("\nAll predicate cross-validation tests passed.\n");
    return 0;
}
