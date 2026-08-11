/**
 *  @file test/cuda/test_mmff_gradient.cpp
 *  @brief Validate MMFF94 energy and gradient for OOP and full-stage terms.
 */

#include <cmath>   // `std::fabs`, `std::isfinite`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <algorithm> // `std::max`
#include <vector>    // `std::vector`

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"

#include "test/test_rdkit_reference_helpers.hpp"
#include "test/cuda/test_mmff_gradient_kernels.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

using sftest::gpu_mmff_energy;

namespace smallfield::tests {

static std::vector<float> interleaved_to_soa_(std::vector<float> const &interleaved) {
    std::uint32_t const num_atoms = static_cast<std::uint32_t>(interleaved.size() / 3);
    std::vector<float> soa_positions(3 * num_atoms, 0.0f);
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        soa_positions[atom] = interleaved[3 * atom + 0];
        soa_positions[num_atoms + atom] = interleaved[3 * atom + 1];
        soa_positions[2 * num_atoms + atom] = interleaved[3 * atom + 2];
    }
    return soa_positions;
}

static void evaluate_mmff_energy_and_gradient_(sfp::etkdg_prepared_molecule_t const &prepared,
                                               std::vector<float> const &positions_soa, float *out_energy,
                                               std::vector<float> &gradient_out) {
    gradient_out.assign(positions_soa.size(), 0.0f);
    // The pool ships no hop matrix: the device rebuilds it per wave from the bond list, and this
    // single-molecule harness does the same so the nonbonded sweep sees production's exclusions.
    std::vector<std::uint8_t> bond_hops;
    sftest::fill_bond_hops(prepared.mmff_bonds.data(), static_cast<std::uint32_t>(prepared.mmff_bonds.size()),
                           prepared.num_atoms, bond_hops);
    gpu_mmff_energy(positions_soa.data(), prepared.num_atoms,
                    prepared.mmff_atom_vdw.empty() ? nullptr : prepared.mmff_atom_vdw.data(),
                    prepared.mmff_atom_charge.empty() ? nullptr : prepared.mmff_atom_charge.data(),
                    bond_hops.empty() ? nullptr : bond_hops.data(),
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

static auto build_single_oop_prepared_(sfp::etkdg_prepared_molecule_t const &prepared) {
    sf_verify_(!prepared.mmff_oops.empty(), "prepared molecule must contain at least one OOP term");

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
        single.mmff_atom_offsets[atom].oop_offset = position;
        single.mmff_atom_offsets[atom].oop_count = 1;
        single.mmff_interactions.push_back({0, sft::mmff94_out_of_plane_k, static_cast<sft::atom_role_t>(role)});
    }
    return single;
}

static void test_single_oop_gradient_finite_difference_() {
    std::printf("test_single_oop_gradient_finite_difference\n");

    std::string const smiles = "CC(=O)Oc1ccccc1C(=O)O";
    auto prepared_full = sfp::prepare_molecule(smiles);
    sf_verify_(prepared_full.num_atoms > 0, "failed to prepare molecule");
    sf_verify_(!prepared_full.mmff_oops.empty(), "test molecule must contain OOP terms");

    auto prepared = build_single_oop_prepared_(prepared_full);
    auto molecule = sftest::mol_from_smiles(smiles);
    sf_verify_(molecule != nullptr, "failed to build RDKit molecule");
    sf_verify_(sftest::embed_molecule(*molecule), "failed to embed RDKit molecule");

    std::vector<float> positions = interleaved_to_soa_(sftest::extract_coords_interleaved(*molecule));
    std::vector<float> analytical_gradient;
    float analytical_energy = 0.0f;
    evaluate_mmff_energy_and_gradient_(prepared, positions, &analytical_energy, analytical_gradient);

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
            evaluate_mmff_energy_and_gradient_(prepared, plus, &energy_plus, scratch_gradient);
            evaluate_mmff_energy_and_gradient_(prepared, minus, &energy_minus, scratch_gradient);

            float const finite_difference = (energy_plus - energy_minus) / (2.0f * epsilon);
            float const analytical = analytical_gradient[component];
            float const abs_error = std::fabs(analytical - finite_difference);
            float const rel_error = abs_error / std::max(1.0f, std::fabs(finite_difference));

            sf_verify_(abs_error < 2.0e-2f || rel_error < 5.0e-2f,
                       "analytical OOP gradient must match finite differences");
        }
    }
}

static void test_full_mmff_gradient_and_energy_() {
    std::printf("test_full_mmff_gradient_and_energy\n");

    std::string const smiles = "CC(=O)Oc1ccccc1C(=O)O";
    auto prepared = sfp::prepare_molecule(smiles);
    sf_verify_(prepared.num_atoms > 0, "failed to prepare molecule");
    sf_verify_(!prepared.mmff_oops.empty(), "test molecule must contain OOP terms");
    sf_verify_(!prepared.mmff_stretchbends.empty(), "test molecule must contain stretch-bend terms");

    auto molecule = sftest::mol_from_smiles(smiles);
    sf_verify_(molecule != nullptr, "failed to build RDKit molecule");
    sf_verify_(sftest::embed_molecule(*molecule), "failed to embed RDKit molecule");

    auto interleaved = sftest::extract_coords_interleaved(*molecule);
    auto positions = interleaved_to_soa_(interleaved);

    float gpu_energy = 0.0f;
    std::vector<float> gradient;
    evaluate_mmff_energy_and_gradient_(prepared, positions, &gpu_energy, gradient);

    double const rdkit_energy = sftest::rdkit_mmff94_energy(*molecule);
    sf_verify_(std::isfinite(rdkit_energy), "RDKit MMFF energy must be finite");
    sf_verify_(std::fabs(static_cast<double>(gpu_energy) - rdkit_energy) < 1.0,
               "MMFF stage energy must match RDKit at the same coordinates");

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;
    for (std::uint32_t atom = 0; atom < prepared.num_atoms; ++atom) {
        sum_x += gradient[atom];
        sum_y += gradient[prepared.num_atoms + atom];
        sum_z += gradient[2 * prepared.num_atoms + atom];
    }
    sf_verify_(std::fabs(sum_x) < 1.0e-2f && std::fabs(sum_y) < 1.0e-2f && std::fabs(sum_z) < 1.0e-2f,
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
        evaluate_mmff_energy_and_gradient_(prepared, plus, &energy_plus, scratch_gradient);
        evaluate_mmff_energy_and_gradient_(prepared, minus, &energy_minus, scratch_gradient);

        float const finite_difference = (energy_plus - energy_minus) / (2.0f * epsilon);
        float const analytical = gradient[component];
        float const abs_error = std::fabs(analytical - finite_difference);
        float const rel_error = abs_error / std::max(1.0f, std::fabs(finite_difference));
        sf_verify_(abs_error < 2.5e-2f || rel_error < 7.5e-2f,
                   "full MMFF analytical gradients must match finite differences");
    }
}

} // namespace smallfield::tests

int main() {
    smallfield::tests::test_single_oop_gradient_finite_difference_();
    smallfield::tests::test_full_mmff_gradient_and_energy_();
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
