/**
 *  @file test_descriptor.cpp
 *  @brief Validate molecule_descriptor_t fields against RDKit's accessor API.
 *
 *  For a diverse set of drug-like molecules: build the descriptor, then for every
 *  atom/bond/neighbor/ring assert the descriptor field matches RDKit's answer exactly.
 *  Also validates BFS distance matrix against RDKit's getDistanceMat() (Floyd-Warshall).
 */

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "types/molecule_descriptor.hpp"
#include "test_helpers.hpp"

using namespace smallfield;
using namespace smallfield::test;

static void validate_molecule(std::string const &smiles) {
    auto mol = mol_from_smiles(smiles);
    SF_TEST_ASSERT(mol != nullptr, smiles.c_str());

    molecule_descriptor_t desc;
    bool ok = build_descriptor(*mol, desc);
    SF_TEST_ASSERT(ok, ("build_descriptor failed for: " + smiles).c_str());

    std::uint32_t const N = mol->getNumAtoms();
    SF_TEST_ASSERT(desc.num_atoms == N, "num_atoms mismatch");
    SF_TEST_ASSERT(desc.num_bonds == mol->getNumBonds(), "num_bonds mismatch");

    for (std::uint32_t a = 0; a < N; ++a) {
        auto const *atom = mol->getAtomWithIdx(a);
        SF_TEST_ASSERT(desc.element[a] == atom->getAtomicNum(), "element mismatch");
        SF_TEST_ASSERT(desc.hybridization[a] == static_cast<std::uint8_t>(atom->getHybridization()),
                       "hybridization mismatch");
        SF_TEST_ASSERT(desc.degree[a] == atom->getDegree(), "degree mismatch");
        SF_TEST_ASSERT(desc.num_hydrogens[a] == atom->getTotalNumHs(true), "num_hydrogens mismatch");
        SF_TEST_ASSERT(desc.formal_charge[a] == static_cast<std::int8_t>(atom->getFormalCharge()),
                       "formal_charge mismatch");

        bool const expect_aromatic = atom->getIsAromatic();
        bool const actual_aromatic = atom_has_flag(desc, a, atom_flag_aromatic_k);
        SF_TEST_ASSERT(expect_aromatic == actual_aromatic, "aromatic flag mismatch");
    }

    for (std::uint32_t b = 0; b < mol->getNumBonds(); ++b) {
        auto const *bond = mol->getBondWithIdx(b);
        std::uint32_t const bi = bond->getBeginAtomIdx();
        std::uint32_t const bj = bond->getEndAtomIdx();
        SF_TEST_ASSERT(desc.bond_atom_i[b] == bi, "bond atom_i mismatch");
        SF_TEST_ASSERT(desc.bond_atom_j[b] == bj, "bond atom_j mismatch");

        bool const expect_aromatic = bond->getIsAromatic();
        bool const actual_aromatic = bond_has_flag(desc, b, bond_flag_aromatic_k);
        SF_TEST_ASSERT(expect_aromatic == actual_aromatic, "bond aromatic flag mismatch");
    }

    for (std::uint32_t a = 0; a < N; ++a) {
        auto const *atom = mol->getAtomWithIdx(a);
        // RDKit neighbors
        std::vector<std::uint32_t> rdkit_neighbors;
        for (auto const *nb : mol->atomNeighbors(atom)) rdkit_neighbors.push_back(nb->getIdx());
        std::sort(rdkit_neighbors.begin(), rdkit_neighbors.end());

        // Descriptor neighbors
        std::vector<std::uint32_t> desc_neighbors;
        std::uint16_t const start = desc.neighbor_offset[a];
        std::uint16_t const end = desc.neighbor_offset[a + 1];
        for (std::uint16_t n = start; n < end; ++n) desc_neighbors.push_back(desc.neighbor_atom[n]);
        std::sort(desc_neighbors.begin(), desc_neighbors.end());

        SF_TEST_ASSERT(rdkit_neighbors.size() == desc_neighbors.size(), "neighbor count mismatch");
        for (std::size_t i = 0; i < rdkit_neighbors.size(); ++i)
            SF_TEST_ASSERT(rdkit_neighbors[i] == desc_neighbors[i], "neighbor index mismatch");
    }

    // Compare to RDKit's Floyd-Warshall on a few random pairs.
    double *rdkit_dist = RDKit::MolOps::getDistanceMat(*mol);
    for (std::uint32_t i = 0; i < N; ++i) {
        for (std::uint32_t j = 0; j < N; ++j) {
            double rd = rdkit_dist[i * N + j];
            if (rd >= 1e7) continue; // unreachable in RDKit's LOCAL_INF encoding
            std::uint32_t d = bonds_between(desc, i, j);
            SF_TEST_ASSERT(static_cast<double>(d) == rd, "bond_distance mismatch with Floyd-Warshall");
        }
    }

    auto const *ring_info = mol->getRingInfo();
    for (std::uint32_t a = 0; a < N; ++a) {
        bool const expect_in_ring = ring_info->numAtomRings(a) > 0;
        bool const actual_in_ring = atom_has_flag(desc, a, atom_flag_in_ring_k);
        SF_TEST_ASSERT(expect_in_ring == actual_in_ring, "atom in-ring flag mismatch");
    }
    for (std::uint32_t b = 0; b < mol->getNumBonds(); ++b) {
        bool const expect_ring = ring_info->numBondRings(b) > 0;
        bool const actual_ring = bond_has_flag(desc, b, bond_flag_in_ring_k);
        SF_TEST_ASSERT(expect_ring == actual_ring, "bond in-ring flag mismatch");
    }

    std::printf("  ok: %s (N=%u, bonds=%u, rings=%u)\n", smiles.c_str(), N, desc.num_bonds, desc.num_rings);
}

int main() {
    std::printf("Validating molecule_descriptor_t against RDKit accessors...\n");

    for (auto const &smi : standard_test_smiles()) validate_molecule(smi);

    // Stress cases
    validate_molecule("N#Cc1ccc(cc1)/C=C/c2ccc(C#N)cc2"); // stereo + nitriles
    validate_molecule("[H+].[H+].[Cl-].[Cl-]");           // ions
    validate_molecule("OC(=O)CCCCCCCCCCCCCCC(=O)O");      // long chain (hexadecanedioic acid)
    validate_molecule("C1CC2CC3CC1CC(C2)C3");             // bridged ring (adamantane)
    validate_molecule("c1ccc2c(c1)[nH]c3ccccc23");        // fused aromatic (carbazole)

    std::printf("\nAll descriptor validation tests passed.\n");
    return 0;
}
