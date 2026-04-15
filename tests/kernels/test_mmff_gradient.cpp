/**
 *  @file test_mmff_gradient.cpp
 *  @brief MMFF94 energy and gradient validation for OOP and full-stage terms.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "test_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

extern "C" void gpu_mmff_energy(float const *host_positions, std::uint32_t num_atoms,
                                sft::mmff94_atom_vdw_params_t const *host_atom_vdw, float const *host_atom_charge,
                                std::uint8_t const *host_bond_distance, sft::mmff94_bond_t const *host_bond_params,
                                std::uint32_t num_bond_params, sft::mmff94_angle_t const *host_angle_params,
                                std::uint32_t num_angle_params, sft::mmff94_torsion_t const *host_torsion_params,
                                std::uint32_t num_torsion_params, sft::mmff94_oop_t const *host_oop_params,
                                std::uint32_t num_oop_params, sft::mmff94_stretchbend_t const *host_stretchbend_params,
                                std::uint32_t num_stretchbend_params,
                                sft::mmff94_atom_interaction_t const *host_interactions, std::uint32_t num_interactions,
                                sft::mmff94_atom_offsets_t const *host_atom_offsets, std::uint32_t num_atom_offsets,
                                float *host_gradient, float *out_energy);

namespace {

std::vector<float> interleaved_to_soa(std::vector<float> const &interleaved) {
    std::uint32_t const num_atoms = static_cast<std::uint32_t>(interleaved.size() / 3);
    std::vector<float> soa(3 * num_atoms, 0.0f);
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        soa[atom] = interleaved[3 * atom + 0];
        soa[num_atoms + atom] = interleaved[3 * atom + 1];
        soa[2 * num_atoms + atom] = interleaved[3 * atom + 2];
    }
    return soa;
}

void evaluate_mmff_energy_and_gradient(sfp::etkdg_prepared_molecule_t const &prepared,
                                       std::vector<float> const &positions_soa, float *out_energy,
                                       std::vector<float> &gradient_out) {
    gradient_out.assign(positions_soa.size(), 0.0f);
    gpu_mmff_energy(positions_soa.data(), prepared.num_atoms,
                    prepared.mmff_atom_vdw.empty() ? nullptr : prepared.mmff_atom_vdw.data(),
                    prepared.mmff_atom_charge.empty() ? nullptr : prepared.mmff_atom_charge.data(),
                    prepared.bond_distance.empty() ? nullptr : prepared.bond_distance.data(),
                    prepared.mmff_bonds.empty() ? nullptr : prepared.mmff_bonds.data(),
                    static_cast<std::uint32_t>(prepared.mmff_bonds.size()),
                    prepared.mmff_angles.empty() ? nullptr : prepared.mmff_angles.data(),
                    static_cast<std::uint32_t>(prepared.mmff_angles.size()),
                    prepared.mmff_torsions.empty() ? nullptr : prepared.mmff_torsions.data(),
                    static_cast<std::uint32_t>(prepared.mmff_torsions.size()),
                    prepared.mmff_oops.empty() ? nullptr : prepared.mmff_oops.data(),
                    static_cast<std::uint32_t>(prepared.mmff_oops.size()),
                    prepared.mmff_stretchbends.empty() ? nullptr : prepared.mmff_stretchbends.data(),
                    static_cast<std::uint32_t>(prepared.mmff_stretchbends.size()),
                    prepared.mmff_interactions.empty() ? nullptr : prepared.mmff_interactions.data(),
                    static_cast<std::uint32_t>(prepared.mmff_interactions.size()),
                    prepared.mmff_atom_offsets.empty() ? nullptr : prepared.mmff_atom_offsets.data(),
                    static_cast<std::uint32_t>(prepared.mmff_atom_offsets.size()), gradient_out.data(), out_energy);
}

auto build_single_oop_prepared(sfp::etkdg_prepared_molecule_t const &prepared) {
    SF_TEST_ASSERT(!prepared.mmff_oops.empty(), "prepared molecule must contain at least one OOP term");

    sfp::etkdg_prepared_molecule_t single {};
    single.num_atoms = prepared.num_atoms;
    single.mmff_oops.push_back(prepared.mmff_oops.front());
    single.mmff_atom_offsets.assign(prepared.num_atoms, {});

    sft::mmff94_oop_t const &oop = single.mmff_oops.front();
    std::uint8_t const atoms[] = {oop.atom_i, oop.atom_j, oop.atom_k, oop.atom_l};
    for (std::uint32_t role = 0; role < 4; ++role) {
        std::uint8_t const atom = atoms[role];
        std::uint32_t const position = static_cast<std::uint16_t>(single.mmff_interactions.size());
        single.mmff_atom_offsets[atom] = {};
        single.mmff_atom_offsets[atom].oop_start = position;
        single.mmff_atom_offsets[atom].oop_count = 1;
        single.mmff_interactions.push_back({0, sft::mmff94_out_of_plane_k, static_cast<sft::atom_role_t>(role)});
    }
    return single;
}

void test_single_oop_gradient_finite_difference() {
    std::printf("test_single_oop_gradient_finite_difference\n");

    std::string const smiles = "CC(=O)Oc1ccccc1C(=O)O";
    auto prepared_full = sfp::prepare_molecule(smiles);
    SF_TEST_ASSERT(prepared_full.num_atoms > 0, "failed to prepare molecule");
    SF_TEST_ASSERT(!prepared_full.mmff_oops.empty(), "test molecule must contain OOP terms");

    auto prepared = build_single_oop_prepared(prepared_full);
    auto mol = sftest::mol_from_smiles(smiles);
    SF_TEST_ASSERT(mol != nullptr, "failed to build RDKit molecule");
    SF_TEST_ASSERT(sftest::embed_molecule(*mol), "failed to embed RDKit molecule");

    std::vector<float> positions = interleaved_to_soa(sftest::extract_coords_interleaved(*mol));
    std::vector<float> analytical_gradient;
    float analytical_energy = 0.0f;
    evaluate_mmff_energy_and_gradient(prepared, positions, &analytical_energy, analytical_gradient);

    sft::mmff94_oop_t const &oop = prepared.mmff_oops.front();
    std::uint32_t const oop_atoms[] = {oop.atom_i, oop.atom_j, oop.atom_k, oop.atom_l};
    constexpr float epsilon = 1.0e-4f;

    for (std::uint32_t role = 0; role < 4; ++role) {
        for (std::uint32_t dim = 0; dim < 3; ++dim) {
            std::uint32_t const component = dim * prepared.num_atoms + oop_atoms[role];
            std::vector<float> plus = positions;
            std::vector<float> minus = positions;
            plus[component] += epsilon;
            minus[component] -= epsilon;

            std::vector<float> scratch_gradient;
            float energy_plus = 0.0f;
            float energy_minus = 0.0f;
            evaluate_mmff_energy_and_gradient(prepared, plus, &energy_plus, scratch_gradient);
            evaluate_mmff_energy_and_gradient(prepared, minus, &energy_minus, scratch_gradient);

            float const finite_difference = (energy_plus - energy_minus) / (2.0f * epsilon);
            float const analytical = analytical_gradient[component];
            float const abs_error = std::fabs(analytical - finite_difference);
            float const rel_error = abs_error / std::max(1.0f, std::fabs(finite_difference));

            SF_TEST_ASSERT(abs_error < 2.0e-2f || rel_error < 5.0e-2f,
                           "analytical OOP gradient must match finite differences");
        }
    }
}

void test_full_mmff_gradient_and_energy() {
    std::printf("test_full_mmff_gradient_and_energy\n");

    std::string const smiles = "CC(=O)Oc1ccccc1C(=O)O";
    auto prepared = sfp::prepare_molecule(smiles);
    SF_TEST_ASSERT(prepared.num_atoms > 0, "failed to prepare molecule");
    SF_TEST_ASSERT(!prepared.mmff_oops.empty(), "test molecule must contain OOP terms");
    SF_TEST_ASSERT(!prepared.mmff_stretchbends.empty(), "test molecule must contain stretch-bend terms");

    auto mol = sftest::mol_from_smiles(smiles);
    SF_TEST_ASSERT(mol != nullptr, "failed to build RDKit molecule");
    SF_TEST_ASSERT(sftest::embed_molecule(*mol), "failed to embed RDKit molecule");

    auto interleaved = sftest::extract_coords_interleaved(*mol);
    auto positions = interleaved_to_soa(interleaved);

    float gpu_energy = 0.0f;
    std::vector<float> gradient;
    evaluate_mmff_energy_and_gradient(prepared, positions, &gpu_energy, gradient);

    double const rdkit_energy = sftest::rdkit_mmff94_energy(*mol);
    SF_TEST_ASSERT(std::isfinite(rdkit_energy), "RDKit MMFF energy must be finite");
    SF_TEST_ASSERT(std::fabs(static_cast<double>(gpu_energy) - rdkit_energy) < 1.0,
                   "MMFF stage energy must match RDKit at the same coordinates");

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;
    for (std::uint32_t atom = 0; atom < prepared.num_atoms; ++atom) {
        sum_x += gradient[atom];
        sum_y += gradient[prepared.num_atoms + atom];
        sum_z += gradient[2 * prepared.num_atoms + atom];
    }
    SF_TEST_ASSERT(std::fabs(sum_x) < 1.0e-2f && std::fabs(sum_y) < 1.0e-2f && std::fabs(sum_z) < 1.0e-2f,
                   "MMFF analytical gradients must remain translationally invariant");

    constexpr float epsilon = 1.0e-4f;
    std::vector<std::uint32_t> sample_components;
    sample_components.reserve(15);
    for (std::uint32_t component = 0; component < gradient.size() && sample_components.size() < 15; component += 4)
        sample_components.push_back(component);

    std::vector<float> scratch_gradient;
    for (std::uint32_t component : sample_components) {
        std::vector<float> plus = positions;
        std::vector<float> minus = positions;
        plus[component] += epsilon;
        minus[component] -= epsilon;

        float energy_plus = 0.0f;
        float energy_minus = 0.0f;
        evaluate_mmff_energy_and_gradient(prepared, plus, &energy_plus, scratch_gradient);
        evaluate_mmff_energy_and_gradient(prepared, minus, &energy_minus, scratch_gradient);

        float const finite_difference = (energy_plus - energy_minus) / (2.0f * epsilon);
        float const analytical = gradient[component];
        float const abs_error = std::fabs(analytical - finite_difference);
        float const rel_error = abs_error / std::max(1.0f, std::fabs(finite_difference));
        SF_TEST_ASSERT(abs_error < 2.5e-2f || rel_error < 7.5e-2f,
                       "full MMFF analytical gradients must match finite differences");
    }
}

} // namespace

int main() {
    test_single_oop_gradient_finite_difference();
    test_full_mmff_gradient_and_energy();
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
