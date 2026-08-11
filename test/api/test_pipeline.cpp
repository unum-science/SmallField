/**
 *  @file test/api/test_pipeline.cpp
 *  @brief Test the end-to-end ETKDG pipeline using the SmallField library API.
 *
 *  Tests: SMILES → context_t::generate_conformers → RDKit cross-validation.
 */

#include <cmath>  // `std::isfinite`
#include <cstdio> // `printf`

#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <cuda_fp16.h>

#include "smallfield/api/smallfield.hpp"

#include "test_helpers.hpp"

namespace sfapi = smallfield::api;
namespace sftest = smallfield::tests;

int main() {
    std::printf("test_pipeline: end-to-end ETKDG conformer generation via library API\n");

    auto test_smiles_str = sftest::standard_test_smiles();
    std::vector<std::string_view> test_smiles(test_smiles_str.begin(), test_smiles_str.end());

    constexpr std::uint32_t num_conformers = 5;
    constexpr std::uint32_t max_atoms = 128;

    sfapi::config_t config;
    config.num_conformers = num_conformers;
    config.max_stereo_attempts = 20;
    config.random_seed = 42;

    std::uint32_t const num_molecules = static_cast<std::uint32_t>(test_smiles.size());
    std::uint32_t const total_conformers = num_molecules * num_conformers;
    std::uint32_t const total_coords = total_conformers * max_atoms * 3;

    std::vector<__half> coordinates(total_coords);
    std::vector<float> energies(total_conformers);
    std::vector<sfapi::conformer_status_t> status(total_conformers, sfapi::conformer_status_t::invalid_smiles_k);

    // use a single GPU for the test
    int const device_id = 0;
    sfapi::context_t context(std::span<int const>(&device_id, 1));

    sfapi::conformer_status_t const batch_status = context.generate_conformers(test_smiles, config, max_atoms,
                                                                               coordinates, energies, status);

    std::printf("  batch status: %u\n", static_cast<unsigned>(batch_status));

    // cross-validate against RDKit
    std::uint32_t total_attempted = 0;
    std::uint32_t total_successful = 0;
    std::uint32_t total_finite_energy = 0;
    std::uint32_t total_reasonable_energy = 0;

    for (std::uint32_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        std::printf("  molecule %u (%.*s):\n", molecule_index, static_cast<int>(test_smiles[molecule_index].size()),
                    test_smiles[molecule_index].data());

        auto rdkit_mol = sftest::mol_from_smiles(std::string(test_smiles[molecule_index]));
        if (!rdkit_mol) {
            std::printf("    skipped (RDKit parse failure)\n");
            continue;
        }
        bool const embedded = sftest::embed_molecule(*rdkit_mol);
        if (!embedded) {
            std::printf("    skipped (RDKit embed failure)\n");
            continue;
        }

        std::uint32_t const num_atoms = rdkit_mol->getNumAtoms();

        for (std::uint32_t conformer = 0; conformer < num_conformers; ++conformer) {
            total_attempted++;
            std::uint32_t const slot = molecule_index * num_conformers + conformer;

            if (status[slot] != sfapi::conformer_status_t::success_k) {
                std::printf("    conformer %u: status=%u\n", conformer, static_cast<unsigned>(status[slot]));
                continue;
            }
            total_successful++;

            // convert half-precision SoA coordinates to float interleaved for RDKit
            std::uint32_t const coord_offset = slot * max_atoms * 3;
            std::vector<float> interleaved(num_atoms * 3);
            for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
                interleaved[atom * 3 + 0] = __half2float(coordinates[coord_offset + 0 * max_atoms + atom]);
                interleaved[atom * 3 + 1] = __half2float(coordinates[coord_offset + 1 * max_atoms + atom]);
                interleaved[atom * 3 + 2] = __half2float(coordinates[coord_offset + 2 * max_atoms + atom]);
            }

            sftest::set_coords_interleaved(*rdkit_mol, interleaved.data());
            double const rdkit_energy = sftest::rdkit_mmff94_energy(*rdkit_mol);

            if (std::isfinite(rdkit_energy)) {
                total_finite_energy++;
                if (rdkit_energy < 500.0) total_reasonable_energy++;
            }

            if (conformer == 0) {
                std::printf("    conformer 0: rdkit=%.2f sf=%.4f kcal/mol\n", rdkit_energy, energies[slot]);
            }
        }
    }

    double const success_rate = total_attempted > 0 ? 100.0 * total_successful / total_attempted : 0.0;
    std::printf("\n  summary:\n");
    std::printf("    conformers attempted:  %u\n", total_attempted);
    std::printf("    conformers succeeded:  %u (%.1f%%)\n", total_successful, success_rate);
    std::printf("    finite MMFF94 energy:  %u (%.1f%% of successes)\n", total_finite_energy,
                total_successful > 0 ? 100.0 * total_finite_energy / total_successful : 0.0);
    std::printf("    reasonable energy:     %u (< 500 kcal/mol)\n", total_reasonable_energy);

    sf_verify_(total_successful > 0, "at least one conformer must succeed");
    sf_verify_(success_rate >= 50.0, "success rate must be at least 50%%");

    std::printf("\ntest_pipeline: PASSED\n");
    return 0;
}
