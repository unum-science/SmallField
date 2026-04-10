/**
 * @file test_pipeline.cpp
 * @brief End-to-end ETKDG pipeline test: SMILES → GPU conformers → RDKit cross-validation.
 * Compiled with g++ (C++20). CUDA kernel wrapper in test_pipeline_kernels.cu.
 */

#include <cmath>
#include <cstdio>
#include <vector>

#include "etkdg_preprocess.hpp"
#include "test_helpers.hpp"

using namespace smallfield;
using namespace smallfield::test;

extern "C" void gpu_etkdg_pipeline(
    float const* host_bounds, std::uint32_t bounds_total_floats, float const* host_initial_coordinates,
    std::uint32_t initial_coordinates_total, chiral_center_descriptor_t const* host_chirals,
    std::uint32_t chirals_total, etk_torsion_descriptor_t const* host_torsions, std::uint32_t torsions_total,
    distance_constraint_t const* host_constraints, std::uint32_t constraints_total,
    mmff94_nonbonded_pair_t const* host_mmff_nb, std::uint32_t mmff_nb_total, mmff94_bond_t const* host_mmff_bonds,
    std::uint32_t mmff_bonds_total, mmff94_angle_t const* host_mmff_angles, std::uint32_t mmff_angles_total,
    mmff94_torsion_t const* host_mmff_torsions, std::uint32_t mmff_torsions_total, mmff94_oop_t const* host_mmff_oops,
    std::uint32_t mmff_oops_total, mmff94_stretchbend_t const* host_mmff_stbn, std::uint32_t mmff_stbn_total,
    mmff94_atom_interaction_t const* host_mmff_interactions, std::uint32_t mmff_interactions_total,
    mmff94_atom_offsets_t const* host_mmff_offsets, std::uint32_t mmff_offsets_total,
    etkdg_molecule_metadata_t const* host_metadata, std::uint32_t num_molecules, float* host_coords_out,
    float* host_energies_out, std::uint8_t* host_status_out, std::uint32_t num_conformers,
    std::uint32_t max_stereo_attempts, std::uint32_t max_atoms_per_molecule, std::uint64_t seed);

static void test_batch_skips_invalid_inputs() {
    std::printf("test_batch_skips_invalid_inputs\n");
    std::vector<std::string> smiles = {"CCO", "not_a_smiles", "CC(=O)O"};
    auto batch = prepare_molecule_batch(smiles);
    SF_TEST_ASSERT(batch.num_molecules == 2, "invalid inputs must be excluded from the packed batch");
    SF_TEST_ASSERT(batch.input_indices.size() == 2, "packed batch must retain input index mapping");
    SF_TEST_ASSERT(batch.input_indices[0] == 0 && batch.input_indices[1] == 2,
                   "packed batch must preserve source input indices");
}

int main() {
    test_batch_skips_invalid_inputs();
    std::printf("test_pipeline: end-to-end ETKDG conformer generation\n");

    auto test_smiles = standard_test_smiles();
    constexpr std::uint32_t num_conformers = 5;
    constexpr std::uint32_t max_stereo_attempts = 20;
    constexpr std::uint64_t seed = 42;

    // Run stage 0 and stage 1 CPU preprocessing.
    std::printf("  preprocessing %zu molecules...\n", test_smiles.size());
    auto batch = prepare_molecule_batch(test_smiles);
    std::printf("  batch: %u molecules, max_atoms=%u\n", batch.num_molecules, batch.max_atoms_per_molecule);

    SF_TEST_ASSERT(batch.num_molecules > 0, "batch must have at least one molecule");
    SF_TEST_ASSERT(batch.max_atoms_per_molecule <= 128, "max atoms must be <= 128 for single-warp pipeline");

    // Allocate output buffers.
    std::uint32_t const max_total_conformers = batch.num_molecules * num_conformers;
    std::uint32_t const max_coords = max_total_conformers * batch.max_atoms_per_molecule * 3;
    std::vector<float> coords_out(max_coords, 0.0f);
    std::vector<float> energies_out(max_total_conformers, 0.0f);
    std::vector<std::uint8_t> status_out(max_total_conformers, 255);

    // Run the stage 2 through stage 9 GPU pipeline.
    std::printf("  launching GPU pipeline...\n");
    gpu_etkdg_pipeline(batch.bounds_matrices.data(), static_cast<std::uint32_t>(batch.bounds_matrices.size()),
                       batch.initial_coordinates.data(), static_cast<std::uint32_t>(batch.initial_coordinates.size()),
                       batch.chiral_centers.empty() ? nullptr : batch.chiral_centers.data(),
                       static_cast<std::uint32_t>(batch.chiral_centers.size()),
                       batch.etk_torsions.empty() ? nullptr : batch.etk_torsions.data(),
                       static_cast<std::uint32_t>(batch.etk_torsions.size()),
                       batch.distance_constraints.empty() ? nullptr : batch.distance_constraints.data(),
                       static_cast<std::uint32_t>(batch.distance_constraints.size()),
                       batch.mmff_nonbonded.empty() ? nullptr : batch.mmff_nonbonded.data(),
                       static_cast<std::uint32_t>(batch.mmff_nonbonded.size()),
                       batch.mmff_bonds.empty() ? nullptr : batch.mmff_bonds.data(),
                       static_cast<std::uint32_t>(batch.mmff_bonds.size()),
                       batch.mmff_angles.empty() ? nullptr : batch.mmff_angles.data(),
                       static_cast<std::uint32_t>(batch.mmff_angles.size()),
                       batch.mmff_torsions.empty() ? nullptr : batch.mmff_torsions.data(),
                       static_cast<std::uint32_t>(batch.mmff_torsions.size()),
                       batch.mmff_oops.empty() ? nullptr : batch.mmff_oops.data(),
                       static_cast<std::uint32_t>(batch.mmff_oops.size()),
                       batch.mmff_stretchbends.empty() ? nullptr : batch.mmff_stretchbends.data(),
                       static_cast<std::uint32_t>(batch.mmff_stretchbends.size()),
                       batch.mmff_interactions.empty() ? nullptr : batch.mmff_interactions.data(),
                       static_cast<std::uint32_t>(batch.mmff_interactions.size()),
                       batch.mmff_atom_offsets.empty() ? nullptr : batch.mmff_atom_offsets.data(),
                       static_cast<std::uint32_t>(batch.mmff_atom_offsets.size()), batch.molecule_metadata.data(),
                       batch.num_molecules, coords_out.data(), energies_out.data(), status_out.data(), num_conformers,
                       max_stereo_attempts, batch.max_atoms_per_molecule, seed);

    // Cross-validate against RDKit.
    std::uint32_t total_successful = 0;
    std::uint32_t total_attempted = 0;
    std::uint32_t total_finite_energy = 0;
    std::uint32_t total_reasonable_energy = 0;

    for (std::uint32_t mol_idx = 0; mol_idx < batch.num_molecules; ++mol_idx) {
        std::uint32_t const input_index = batch.input_indices[mol_idx];
        std::uint32_t const num_atoms = batch.molecule_metadata[mol_idx].num_atoms;
        std::printf("  molecule %u (%s): %u atoms\n", mol_idx, test_smiles[input_index].c_str(), num_atoms);

        auto rdkit_mol = mol_from_smiles(test_smiles[input_index]);
        SF_TEST_ASSERT(rdkit_mol != nullptr, "failed to parse SMILES for cross-reference");
        bool embedded = embed_molecule(*rdkit_mol);
        SF_TEST_ASSERT(embedded, "failed to embed molecule for cross-reference");

        for (std::uint32_t conf = 0; conf < num_conformers; ++conf) {
            total_attempted++;
            std::uint32_t const status_index = mol_idx * num_conformers + conf;
            std::uint8_t const status = status_out[status_index];

            if (status != 0) {
                std::printf("    conformer %u: FAILED (status=%u)\n", conf, status);
                continue;
            }
            total_successful++;

            // Extract conformer coordinates and convert from SoA to interleaved layout for RDKit.
            // Use max_atoms_per_molecule as the stride within each conformer block.
            std::uint32_t const max_n = batch.max_atoms_per_molecule;
            std::uint32_t const coord_offset = (mol_idx * num_conformers + conf) * max_n * 3;
            std::vector<float> interleaved_coords(num_atoms * 3);
            for (std::uint32_t i = 0; i < num_atoms; ++i) {
                interleaved_coords[i * 3 + 0] = coords_out[coord_offset + 0 * max_n + i];
                interleaved_coords[i * 3 + 1] = coords_out[coord_offset + 1 * max_n + i];
                interleaved_coords[i * 3 + 2] = coords_out[coord_offset + 2 * max_n + i];
            }

            // Cross-reference by setting the GPU coordinates on the RDKit molecule and evaluating MMFF94 energy.
            set_coords_interleaved(*rdkit_mol, interleaved_coords.data());
            double const mmff_energy = rdkit_mmff94_energy(*rdkit_mol);

            if (std::isfinite(mmff_energy)) {
                total_finite_energy++;
                if (mmff_energy < 200.0) {
                    total_reasonable_energy++;
                }
                double const energy_delta = std::fabs(mmff_energy - energies_out[status_index]);
                SF_TEST_ASSERT(energy_delta < 1.0, "pipeline MMFF energy must match RDKit cross-check");
            }

            if (conf == 0) {
                // Print the first conformer details.
                std::printf("    conformer %u: MMFF94=%.2f kcal/mol pipeline_energy=%.4f\n", conf, mmff_energy,
                            energies_out[status_index]);
            }
        }
    }

    // Print the summary.
    double const success_rate = total_attempted > 0 ? 100.0 * total_successful / total_attempted : 0.0;
    double const finite_rate = total_successful > 0 ? 100.0 * total_finite_energy / total_successful : 0.0;

    std::printf("\n  summary:\n");
    std::printf("    conformers attempted:  %u\n", total_attempted);
    std::printf("    conformers succeeded:  %u (%.1f%%)\n", total_successful, success_rate);
    std::printf("    finite MMFF94 energy:  %u (%.1f%% of successes)\n", total_finite_energy, finite_rate);
    std::printf("    reasonable energy:     %u (< 200 kcal/mol)\n", total_reasonable_energy);

    // The pipeline should generate at least some successful conformers.
    SF_TEST_ASSERT(total_successful > 0, "at least some conformers must succeed");
    SF_TEST_ASSERT(total_finite_energy > 0, "at least some conformers must have finite MMFF94 energy");
    SF_TEST_ASSERT(total_reasonable_energy == total_successful,
                   "all successful conformers must be chemically reasonable");

    std::printf("ALL TESTS PASSED\n");
    return 0;
}
