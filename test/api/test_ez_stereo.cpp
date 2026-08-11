/**
 *  @file test/api/test_ez_stereo.cpp
 *  @brief Confirm, round-trip through RDKit, that generated conformers honour defined E/Z double bonds.
 *
 *  Generates conformers for a set of E/Z isomer pairs, hands the coordinates back to RDKit, re-perceives
 *  stereochemistry from 3D, and asserts the perceived bond stereo matches what the input SMILES declared. Without
 *  a stereo-aware 1-4 bounds envelope both isomers of a pair collapse to whichever geometry the minimizer happens
 *  to reach.
 */

#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`

#include <cuda_fp16.h>
#include <GraphMol/MolOps.h>

#include "smallfield/api/smallfield.hpp"

#include "test/test_rdkit_reference_helpers.hpp"

namespace sfapi = smallfield::api;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_ez_stereo";

/** One isomer under test and the bond stereo its SMILES declares. */
struct ez_case_t {
    char const *smiles;
    /** True for Z / cis, false for E / trans. */
    bool expect_cis;
};

/** Perceive bond stereo from 3D and report whether any double bond is cis. */
static bool perceived_any_cis(RDKit::RWMol &molecule, bool &found_stereo) {
    RDKit::MolOps::assignStereochemistryFrom3D(molecule);
    found_stereo = false;
    for (auto const *bond : molecule.bonds()) {
        auto const stereo = bond->getStereo();
        if (stereo == RDKit::Bond::STEREOZ || stereo == RDKit::Bond::STEREOCIS) {
            found_stereo = true;
            return true;
        }
        if (stereo == RDKit::Bond::STEREOE || stereo == RDKit::Bond::STEREOTRANS) {
            found_stereo = true;
            return false;
        }
    }
    return false;
}

int main() {
    // Each pair differs only in double-bond configuration, so a stereo-blind bounds
    // envelope produces indistinguishable geometry for both members.
    std::vector<ez_case_t> const cases = {
        {"C/C=C/C", false},        // trans-2-butene
        {"C/C=C\\C", true},        // cis-2-butene
        {"C/C=C/CC", false},       // trans-2-pentene
        {"C/C=C\\CC", true},       // cis-2-pentene
        {"Cl/C=C/Cl", false},      // trans-1,2-dichloroethene
        {"Cl/C=C\\Cl", true},      // cis-1,2-dichloroethene
        {"C/C=C/c1ccccc1", false}, // trans-beta-methylstyrene
        {"C/C=C\\c1ccccc1", true}, // cis-beta-methylstyrene
    };

    std::vector<std::string> smiles_storage;
    for (auto const &entry : cases) smiles_storage.emplace_back(entry.smiles);
    std::vector<std::string_view> smiles(smiles_storage.begin(), smiles_storage.end());

    constexpr std::uint32_t num_conformers = 4;
    constexpr std::uint32_t max_atoms = 128;
    std::uint32_t const num_molecules = static_cast<std::uint32_t>(smiles.size());
    std::size_t const total_slots = static_cast<std::size_t>(num_molecules) * num_conformers;

    std::vector<__half> coords(total_slots * max_atoms * 3);
    std::vector<float> energies(total_slots);
    std::vector<sfapi::conformer_status_t> status(total_slots);

    sfapi::context_t context {std::span<int const> {}};
    sfapi::config_t config;
    config.num_conformers = num_conformers;
    auto const batch_status = context.generate_conformers(smiles, config, max_atoms, coords, energies, status);
    sf_verify_(batch_status == sfapi::conformer_status_t::success_k,
               "the batch must generate; otherwise every per-slot check below passes vacuously");

    std::uint32_t checked = 0, matched = 0, unperceived = 0;
    std::vector<float> interleaved;
    for (std::uint32_t mol_index = 0; mol_index < num_molecules; ++mol_index) {
        for (std::uint32_t conformer = 0; conformer < num_conformers; ++conformer) {
            std::size_t const slot = static_cast<std::size_t>(mol_index) * num_conformers + conformer;
            if (status[slot] != sfapi::conformer_status_t::success_k) continue;

            auto molecule = sftest::mol_from_smiles(smiles_storage[mol_index]);
            if (!molecule) continue;
            std::uint32_t const num_atoms = molecule->getNumAtoms();
            // `set_coords_interleaved` writes into an existing conformer, and a molecule
            // parsed from SMILES carries none.
            if (molecule->getNumConformers() == 0) molecule->addConformer(new RDKit::Conformer(num_atoms), true);

            // Slot layout is SoA at the caller's 128-atom stride.
            std::size_t const base = slot * max_atoms * 3;
            interleaved.assign(static_cast<std::size_t>(num_atoms) * 3, 0.0f);
            for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
                for (std::uint32_t axis = 0; axis < 3; ++axis)
                    interleaved[atom * 3 + axis] = __half2float(coords[base + axis * max_atoms + atom]);
            sftest::set_coords_interleaved(*molecule, interleaved.data());

            bool found_stereo = false;
            bool const is_cis = perceived_any_cis(*molecule, found_stereo);
            ++checked;
            if (!found_stereo) {
                ++unperceived;
                continue;
            }
            if (is_cis == cases[mol_index].expect_cis) ++matched;
            else {
                std::fflush(stdout);
                fmt::println(stderr, "MISMATCH ez_stereo smiles={} conformer={} field=bond_stereo declared={} perceived={}",
                             cases[mol_index].smiles, conformer, cases[mol_index].expect_cis ? "cis" : "trans",
                             is_cis ? "cis" : "trans");
            }
        }
    }

    fmt::println("E/Z round-trip: {} conformers checked, {} matched, {} without perceivable stereo", checked, matched,
                 unperceived);

    sf_verify_(checked > 0, "at least one conformer must be generated");
    // Allow a small share of conformers whose geometry RDKit cannot assign, but every
    // conformer that does carry perceivable stereo must carry the declared one.
    std::uint32_t const perceivable = checked - unperceived;
    sf_verify_(perceivable > 0, "at least one conformer must have perceivable double-bond stereo");
    sf_verify_(matched == perceivable, "every perceivable conformer must match its declared E/Z");

    return sftest::report_pass(test_name_k);
}
