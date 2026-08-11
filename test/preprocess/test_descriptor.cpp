/**
 *  @file test/preprocess/test_descriptor.cpp
 *  @brief Validate molecule_descriptor_t fields against RDKit's accessor API.
 *
 *  For a diverse set of drug-like molecules: build the descriptor, then for every atom/bond/neighbor/ring assert
 *  the descriptor field matches RDKit's answer exactly. Also validates BFS distance matrix against RDKit's
 *  getDistanceMat() (Floyd-Warshall).
 */

#include <cstdio> // `printf`

#include <string> // `std::string`
#include <vector> // `std::vector`

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/types/molecule_descriptor.hpp"

#include "test_helpers.hpp"

namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static void validate_molecule(std::string const &smiles) {
    auto molecule = sftest::mol_from_smiles(smiles);
    sf_verify_(molecule != nullptr, smiles.c_str());

    sft::molecule_descriptor_t descriptor;
    bool ok = sft::build_descriptor(*molecule, descriptor);
    sf_verify_(ok, ("build_descriptor failed for: " + smiles).c_str());

    std::uint32_t const num_atoms = molecule->getNumAtoms();
    sf_verify_(descriptor.num_atoms == num_atoms, "num_atoms mismatch");
    sf_verify_(descriptor.num_bonds == molecule->getNumBonds(), "num_bonds mismatch");

    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        auto const *atom = molecule->getAtomWithIdx(atom_index);
        sf_verify_(descriptor.element[atom_index] == atom->getAtomicNum(), "element mismatch");
        sf_verify_(descriptor.hybridization[atom_index] == static_cast<std::uint8_t>(atom->getHybridization()),
                   "hybridization mismatch");
        sf_verify_(descriptor.degree[atom_index] == atom->getDegree(), "degree mismatch");
        sf_verify_(descriptor.num_hydrogens[atom_index] == atom->getTotalNumHs(true), "num_hydrogens mismatch");
        sf_verify_(descriptor.formal_charge[atom_index] == static_cast<std::int8_t>(atom->getFormalCharge()),
                   "formal_charge mismatch");

        bool const expect_aromatic = atom->getIsAromatic();
        bool const actual_aromatic = sft::atom_has_flag(descriptor, atom_index, sft::atom_flag_aromatic_k);
        sf_verify_(expect_aromatic == actual_aromatic, "aromatic flag mismatch");
    }

    for (std::uint32_t bond_index = 0; bond_index < molecule->getNumBonds(); ++bond_index) {
        auto const *bond = molecule->getBondWithIdx(bond_index);
        std::uint32_t const atom_i = bond->getBeginAtomIdx();
        std::uint32_t const atom_j = bond->getEndAtomIdx();
        sf_verify_(descriptor.bond_atom_i[bond_index] == atom_i, "bond atom_i mismatch");
        sf_verify_(descriptor.bond_atom_j[bond_index] == atom_j, "bond atom_j mismatch");

        bool const expect_aromatic = bond->getIsAromatic();
        bool const actual_aromatic = sft::bond_has_flag(descriptor, bond_index, sft::bond_flag_aromatic_k);
        sf_verify_(expect_aromatic == actual_aromatic, "bond aromatic flag mismatch");
    }

    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        auto const *atom = molecule->getAtomWithIdx(atom_index);
        // RDKit neighbors
        std::vector<std::uint32_t> rdkit_neighbors;
        for (auto const *neighbor : molecule->atomNeighbors(atom)) rdkit_neighbors.push_back(neighbor->getIdx());
        std::sort(rdkit_neighbors.begin(), rdkit_neighbors.end());

        // Descriptor neighbors
        std::vector<std::uint32_t> desc_neighbors;
        std::uint16_t const start = descriptor.neighbor_offset[atom_index];
        std::uint16_t const end = descriptor.neighbor_offset[atom_index + 1];
        for (std::uint16_t entry = start; entry < end; ++entry)
            desc_neighbors.push_back(descriptor.neighbor_atom[entry]);
        std::sort(desc_neighbors.begin(), desc_neighbors.end());

        sf_verify_(rdkit_neighbors.size() == desc_neighbors.size(), "neighbor count mismatch");
        for (std::size_t neighbor_index = 0; neighbor_index < rdkit_neighbors.size(); ++neighbor_index)
            sf_verify_(rdkit_neighbors[neighbor_index] == desc_neighbors[neighbor_index], "neighbor index mismatch");
    }

    // Compare the derived hop matrix to RDKit's Floyd-Warshall.
    std::vector<std::uint8_t> bond_hops(static_cast<std::size_t>(num_atoms) * num_atoms, std::uint8_t {255});
    sft::fill_bond_distances(descriptor, bond_hops.data(), num_atoms);
    double *rdkit_distances = RDKit::MolOps::getDistanceMat(*molecule);
    for (std::uint32_t row = 0; row < num_atoms; ++row) {
        for (std::uint32_t column = 0; column < num_atoms; ++column) {
            double const reference = rdkit_distances[row * num_atoms + column];
            if (reference >= 1e7) continue; // unreachable in RDKit's LOCAL_INF encoding
            sf_verify_(static_cast<double>(bond_hops[row * num_atoms + column]) == reference,
                       "bond hop count mismatch with Floyd-Warshall");
        }
    }

    auto const *ring_info = molecule->getRingInfo();
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        bool const expect_in_ring = ring_info->numAtomRings(atom_index) > 0;
        bool const actual_in_ring = sft::atom_has_flag(descriptor, atom_index, sft::atom_flag_in_ring_k);
        sf_verify_(expect_in_ring == actual_in_ring, "atom in-ring flag mismatch");
    }
    for (std::uint32_t bond_index = 0; bond_index < molecule->getNumBonds(); ++bond_index) {
        bool const expect_ring = ring_info->numBondRings(bond_index) > 0;
        bool const actual_ring = sft::bond_has_flag(descriptor, bond_index, sft::bond_flag_in_ring_k);
        sf_verify_(expect_ring == actual_ring, "bond in-ring flag mismatch");
    }

    std::printf("  ok: %s (num_atoms=%u, bonds=%u, rings=%u)\n", smiles.c_str(), num_atoms, descriptor.num_bonds,
                descriptor.num_rings);
}

int main() {
    std::printf("Validating molecule_descriptor_t against RDKit accessors...\n");

    for (auto const &smiles : sftest::standard_test_smiles()) validate_molecule(smiles);

    // Stress cases
    validate_molecule("N#Cc1ccc(cc1)/C=C/c2ccc(C#N)cc2"); // stereo + nitriles
    validate_molecule("[H+].[H+].[Cl-].[Cl-]");           // ions
    validate_molecule("OC(=O)CCCCCCCCCCCCCCC(=O)O");      // long chain (hexadecanedioic acid)
    validate_molecule("C1CC2CC3CC1CC(C2)C3");             // bridged ring (adamantane)
    validate_molecule("c1ccc2c(c1)[nH]c3ccccc23");        // fused aromatic (carbazole)

    std::printf("\nAll descriptor validation tests passed.\n");
    return 0;
}
