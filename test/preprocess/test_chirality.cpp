/**
 *  @file test_chirality.cpp
 *  @brief Tests for GPU chirality validation vs RDKit reference.
 *  Compiled with g++ (C++20). CUDA kernels in test_chirality_kernels.cu.
 */

#include <cstdio>
#include <vector>

#include "preprocess/etkdg_prepare_batch.hpp"
#include "test_helpers.hpp"

using namespace smallfield;
using namespace smallfield::test;

extern "C" bool gpu_validate_chirality(float const *host_coords, chiral_center_descriptor_t const *host_chirals,
                                       std::uint32_t num_atoms, std::uint32_t num_chirals);

static void test_chirality_correct_conformer() {
    std::printf("test_chirality_correct_conformer\n");

    // Use molecules with known chirality.
    std::vector<std::string> chiral_smiles = {
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",        // (S)-ibuprofen
        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", // glucose
        "C[C@@H](O)CC",                           // (R)-2-butanol
        "N[C@@H](CC(=O)O)C(=O)O",                 // L-aspartic acid
    };

    for (auto const &smiles : chiral_smiles) {
        std::printf("  testing: %s\n", smiles.c_str());

        auto mol = mol_from_smiles(smiles);
        SF_TEST_ASSERT(mol != nullptr, "failed to parse SMILES");

        bool embedded = embed_molecule(*mol);
        SF_TEST_ASSERT(embedded, "failed to embed molecule");
        std::uint32_t const num_atoms = mol->getNumAtoms();

        // Extract graph-only chiral targets from the molecular graph.
        std::vector<chiral_center_descriptor_t> chirals;
        molecule_descriptor_t desc;
        build_descriptor(*mol, desc);
        extract_chiral_centers(*mol, desc, chirals);
        std::printf("    found %zu chiral centers\n", chirals.size());

        if (chirals.empty()) {
            std::printf("    skipping (no chirals extracted)\n");
            continue;
        }

        // Extract coordinates in 3D SoA layout.
        auto coords_interleaved = extract_coords_interleaved(*mol);
        std::vector<float> coords_soa(3 * num_atoms);
        for (std::uint32_t i = 0; i < num_atoms; ++i) {
            coords_soa[0 * num_atoms + i] = coords_interleaved[i * 3 + 0];
            coords_soa[1 * num_atoms + i] = coords_interleaved[i * 3 + 1];
            coords_soa[2 * num_atoms + i] = coords_interleaved[i * 3 + 2];
        }

        bool result = gpu_validate_chirality(coords_soa.data(), chirals.data(), num_atoms,
                                             static_cast<std::uint32_t>(chirals.size()));
        std::printf("    graph-based check: %s\n", result ? "PASS" : "FAIL");
        SF_TEST_ASSERT(result, "embedded RDKit conformer must satisfy extracted chirality targets");
    }
}

static void test_chirality_inverted() {
    std::printf("test_chirality_inverted\n");

    auto mol = mol_from_smiles("C[C@@H](O)CC");
    SF_TEST_ASSERT(mol != nullptr, "failed to parse");

    bool embedded = embed_molecule(*mol);
    SF_TEST_ASSERT(embedded, "failed to embed");
    std::uint32_t const num_atoms = mol->getNumAtoms();

    std::vector<chiral_center_descriptor_t> chirals;
    molecule_descriptor_t desc;
    build_descriptor(*mol, desc);
    extract_chiral_centers(*mol, desc, chirals);

    if (chirals.empty()) {
        std::printf("  skipping (no chirals extracted)\n");
        return;
    }

    auto coords_interleaved = extract_coords_interleaved(*mol);
    std::vector<float> coords_soa(3 * num_atoms);
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        coords_soa[0 * num_atoms + i] = coords_interleaved[i * 3 + 0];
        coords_soa[1 * num_atoms + i] = coords_interleaved[i * 3 + 1];
        coords_soa[2 * num_atoms + i] = coords_interleaved[i * 3 + 2];
    }

    // Flip the target sign. The validation should now fail.
    std::vector<chiral_center_descriptor_t> flipped_chirals = chirals;
    for (auto &c : flipped_chirals) { c.target_sign = -c.target_sign; }

    bool result_original = gpu_validate_chirality(coords_soa.data(), chirals.data(), num_atoms,
                                                  static_cast<std::uint32_t>(chirals.size()));
    bool result_flipped = gpu_validate_chirality(coords_soa.data(), flipped_chirals.data(), num_atoms,
                                                 static_cast<std::uint32_t>(flipped_chirals.size()));
    std::printf("  original: %s, flipped: %s\n", result_original ? "PASS" : "FAIL", result_flipped ? "PASS" : "FAIL");
    // The key test is that the original and flipped targets produce opposite results.
    SF_TEST_ASSERT(result_original != result_flipped, "flipping chiral sign must change validation outcome");
}

int main() {
    test_chirality_correct_conformer();
    test_chirality_inverted();
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
