/**
 *  @file test/preprocess/test_chirality.cpp
 *  @brief Test GPU chirality validation against the RDKit reference.
 *
 *  Compiled with g++, C++20. CUDA kernels in test_chirality_kernels.cu.
 */

#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"

#include "test/preprocess/test_chirality_kernels.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

using sftest::gpu_validate_chirality;

static constexpr std::string_view test_name_k = "test_chirality";

static void test_chirality_correct_conformer() {
    fmt::println("test_chirality_correct_conformer");

    // Use molecules with known chirality.
    std::vector<std::string> chiral_smiles = {
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",        // (S)-ibuprofen
        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", // glucose
        "C[C@@H](O)CC",                           // (R)-2-butanol
        "N[C@@H](CC(=O)O)C(=O)O",                 // L-aspartic acid
    };

    for (auto const &smiles : chiral_smiles) {
        fmt::println("  testing: {}", smiles);

        auto molecule = sftest::mol_from_smiles(smiles);
        sf_verify_(molecule != nullptr, "failed to parse SMILES");

        bool embedded = sftest::embed_molecule(*molecule);
        sf_verify_(embedded, "failed to embed molecule");
        std::uint32_t const num_atoms = molecule->getNumAtoms();

        // Extract graph-only chiral targets from the molecular graph.
        std::vector<sft::chiral_center_descriptor_t> chirals;
        sft::molecule_descriptor_t descriptor;
        sft::build_descriptor(*molecule, descriptor);
        sfp::extract_chiral_centers(*molecule, descriptor, chirals);
        fmt::println("    found {} chiral centers", chirals.size());

        if (chirals.empty()) {
            fmt::println("    skipping (no chirals extracted)");
            continue;
        }

        // Extract coordinates in 3D SoA layout.
        auto coords_interleaved = sftest::extract_coords_interleaved(*molecule);
        std::vector<float> coords_soa(3 * num_atoms);
        for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
            coords_soa[0 * num_atoms + atom_index] = coords_interleaved[atom_index * 3 + 0];
            coords_soa[1 * num_atoms + atom_index] = coords_interleaved[atom_index * 3 + 1];
            coords_soa[2 * num_atoms + atom_index] = coords_interleaved[atom_index * 3 + 2];
        }

        bool result = gpu_validate_chirality(coords_soa.data(), chirals.data(), num_atoms,
                                             static_cast<std::uint32_t>(chirals.size()));
        fmt::println("    NOTE chirality smiles={} field=graph_check satisfied={}", smiles, result ? 1 : 0);
        sf_verify_(result, "embedded RDKit conformer must satisfy extracted chirality targets");
    }
}

static void test_chirality_inverted() {
    fmt::println("test_chirality_inverted");

    auto molecule = sftest::mol_from_smiles("C[C@@H](O)CC");
    sf_verify_(molecule != nullptr, "failed to parse");

    bool embedded = sftest::embed_molecule(*molecule);
    sf_verify_(embedded, "failed to embed");
    std::uint32_t const num_atoms = molecule->getNumAtoms();

    std::vector<sft::chiral_center_descriptor_t> chirals;
    sft::molecule_descriptor_t descriptor;
    sft::build_descriptor(*molecule, descriptor);
    sfp::extract_chiral_centers(*molecule, descriptor, chirals);

    if (chirals.empty()) {
        fmt::println("  skipping (no chirals extracted)");
        return;
    }

    auto coords_interleaved = sftest::extract_coords_interleaved(*molecule);
    std::vector<float> coords_soa(3 * num_atoms);
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        coords_soa[0 * num_atoms + atom_index] = coords_interleaved[atom_index * 3 + 0];
        coords_soa[1 * num_atoms + atom_index] = coords_interleaved[atom_index * 3 + 1];
        coords_soa[2 * num_atoms + atom_index] = coords_interleaved[atom_index * 3 + 2];
    }

    // Flip the target sign. The validation should now fail.
    std::vector<sft::chiral_center_descriptor_t> flipped_chirals = chirals;
    for (auto &center : flipped_chirals) { center.target_sign = -center.target_sign; }

    bool result_original = gpu_validate_chirality(coords_soa.data(), chirals.data(), num_atoms,
                                                  static_cast<std::uint32_t>(chirals.size()));
    bool result_flipped = gpu_validate_chirality(coords_soa.data(), flipped_chirals.data(), num_atoms,
                                                 static_cast<std::uint32_t>(flipped_chirals.size()));
    fmt::println("  NOTE chirality field=target_sign original_satisfied={} flipped_satisfied={}",
                 result_original ? 1 : 0, result_flipped ? 1 : 0);
    // The key test is that the original and flipped targets produce opposite results.
    sf_verify_(result_original != result_flipped, "flipping chiral sign must change validation outcome");
}

int main() {
    test_chirality_correct_conformer();
    test_chirality_inverted();
    return sftest::report_pass(test_name_k);
}
