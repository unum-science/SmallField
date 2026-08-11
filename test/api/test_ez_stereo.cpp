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
#include "smallfield/types/views.hpp"

#include "test/test_rdkit_reference_helpers.hpp"

namespace sfapi = smallfield::api;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_ez_stereo";

/**
 *  @brief A double bond's configuration, including the answer "the geometry carries none".
 *
 *  The third state is the point. Split across a `bool` return and a `bool` out-parameter, "not cis"
 *  and "nothing to perceive" render identically, so a caller that consults only the return classifies
 *  every unperceived conformer as trans and the suite silently stops testing what it was written for.
 */
enum class bond_stereo_t {
    /** Z / cis. */
    cis_k,
    /** E / trans. */
    trans_k,
    /** No double bond in the molecule carried assignable stereo. */
    unperceived_k,
};

/** The literal a bond stereo prints as. */
static std::string_view bond_stereo_name(bond_stereo_t stereo) noexcept {
    switch (stereo) {
    case bond_stereo_t::cis_k: return "cis";
    case bond_stereo_t::trans_k: return "trans";
    default: return "unperceived";
    }
}

/** One isomer under test and the bond stereo its SMILES declares. */
struct ez_case_t {
    char const *smiles;
    bond_stereo_t declared;
};

/** Perceive bond stereo from 3D; the first double bond RDKit can assign decides the answer. */
static bond_stereo_t perceived_bond_stereo(RDKit::RWMol &molecule) {
    RDKit::MolOps::assignStereochemistryFrom3D(molecule);
    for (auto const *bond : molecule.bonds()) {
        auto const stereo = bond->getStereo();
        if (stereo == RDKit::Bond::STEREOZ || stereo == RDKit::Bond::STEREOCIS) return bond_stereo_t::cis_k;
        if (stereo == RDKit::Bond::STEREOE || stereo == RDKit::Bond::STEREOTRANS) return bond_stereo_t::trans_k;
    }
    return bond_stereo_t::unperceived_k;
}

int main() {
    fmt::println("{}: generated conformers round-tripped through RDKit against their declared E/Z", test_name_k);

    // Each pair differs only in double-bond configuration, so a stereo-blind bounds
    // envelope produces indistinguishable geometry for both members.
    std::vector<ez_case_t> const cases = {
        {"C/C=C/C", bond_stereo_t::trans_k},        // trans-2-butene
        {"C/C=C\\C", bond_stereo_t::cis_k},         // cis-2-butene
        {"C/C=C/CC", bond_stereo_t::trans_k},       // trans-2-pentene
        {"C/C=C\\CC", bond_stereo_t::cis_k},        // cis-2-pentene
        {"Cl/C=C/Cl", bond_stereo_t::trans_k},      // trans-1,2-dichloroethene
        {"Cl/C=C\\Cl", bond_stereo_t::cis_k},       // cis-1,2-dichloroethene
        {"C/C=C/c1ccccc1", bond_stereo_t::trans_k}, // trans-beta-methylstyrene
        {"C/C=C\\c1ccccc1", bond_stereo_t::cis_k},  // cis-beta-methylstyrene
    };

    std::vector<std::string> smiles_storage;
    for (auto const &entry : cases) smiles_storage.emplace_back(entry.smiles);
    std::vector<std::string_view> smiles(smiles_storage.begin(), smiles_storage.end());

    constexpr std::uint32_t num_conformers = 4;
    constexpr std::uint32_t max_atoms = 128;
    std::uint32_t const num_molecules = static_cast<std::uint32_t>(smiles.size());
    sfapi::conformer_slot_layout_t const slot_layout {max_atoms, num_conformers};
    std::size_t const total_slots = static_cast<std::size_t>(num_molecules) * num_conformers;

    std::vector<__half> coords(total_slots * max_atoms * 3);
    std::vector<float> energies(total_slots);
    std::vector<sfapi::conformer_status_t> status(total_slots);

    sfapi::context_t context {custd::span<int const> {}};
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

            std::size_t const base = slot_layout.slot_offset(mol_index, conformer);
            interleaved.assign(static_cast<std::size_t>(num_atoms) * 3, 0.0f);
            sfapi::gather_slot_into_interleaved(coords.data() + base, slot_layout, num_atoms, interleaved.data());
            sftest::set_coords_interleaved(*molecule, interleaved.data());

            bond_stereo_t const perceived = perceived_bond_stereo(*molecule);
            ++checked;
            if (perceived == bond_stereo_t::unperceived_k) {
                ++unperceived;
                continue;
            }
            if (perceived == cases[mol_index].declared) ++matched;
            else {
                std::fflush(stdout);
                fmt::println(stderr,
                             "MISMATCH ez_stereo smiles={} conformer={} field=bond_stereo declared={} perceived={}",
                             cases[mol_index].smiles, conformer, bond_stereo_name(cases[mol_index].declared),
                             bond_stereo_name(perceived));
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
