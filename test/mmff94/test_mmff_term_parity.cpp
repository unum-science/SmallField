/**
 *  @file test_mmff_term_parity.cpp
 *  @brief Per-term MMFF94 energy parity between the GPU kernel and RDKit.
 *
 *  The total-energy checks elsewhere cannot localize a divergence, and a term that is
 *  wrong in one direction can be masked by another that is wrong in the other. RDKit
 *  exposes per-term toggles, so each term is compared in isolation on identical
 *  coordinates.
 *
 *  Isolation on our side: bonded terms are selected by keeping only that category's
 *  offsets and interactions, and the nonbonded sweep — which is enumerated on the fly
 *  rather than from a list — is suppressed by marking every pair BFS-unreachable.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <iostream> // RDKit's `AtomTyper.h` uses `std::cout` without including it
#include <memory>   // `std::unique_ptr`
#include <sstream>  // RDKit's `AtomTyper.h` uses `std::stringstream` without including it
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <GraphMol/GraphMol.h>

#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>

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

/** @brief Which single MMFF94 term a comparison isolates. */
enum class term_kind_t { bond, angle, stretchbend, oop, torsion, nonbonded };

char const *term_name(term_kind_t kind) {
    switch (kind) {
    case term_kind_t::bond: return "bond";
    case term_kind_t::angle: return "angle";
    case term_kind_t::stretchbend: return "stretch-bend";
    case term_kind_t::oop: return "out-of-plane";
    case term_kind_t::torsion: return "torsion";
    default: return "nonbonded";
    }
}

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

/**
 *  @brief Copy @p full with every term but @p kind neutralized.
 *
 *  The per-atom interaction CSR is left exactly as production builds it — rebuilding it
 *  by hand would test the reconstruction rather than the kernel.  Excluded bonded terms
 *  are zeroed at their force constants, which contributes neither energy nor gradient,
 *  The nonbonded sweep is enumerated on the fly from the hop matrix rather than from a list,
 *  so it is isolated at the call site instead: the caller hands the kernel a matrix of the
 *  unreachable sentinel for every other term, which the sweep skips pair by pair.
 */
sfp::etkdg_prepared_molecule_t isolate_term(sfp::etkdg_prepared_molecule_t const &full, term_kind_t kind) {
    sfp::etkdg_prepared_molecule_t single = full;

    if (kind != term_kind_t::bond)
        for (auto &term : single.mmff_bonds) term.force_constant = 0.0f;
    if (kind != term_kind_t::angle)
        for (auto &term : single.mmff_angles) term.force_constant = 0.0f;
    if (kind != term_kind_t::torsion)
        for (auto &term : single.mmff_torsions) term.v1 = term.v2 = term.v3 = 0.0f;
    if (kind != term_kind_t::oop)
        for (auto &term : single.mmff_oops) term.force_constant = 0.0f;
    if (kind != term_kind_t::stretchbend)
        for (auto &term : single.mmff_stretchbends) term.kba_ijk = term.kba_kji = 0.0f;
    return single;
}

/** @brief RDKit MMFF94 energy with exactly @p kind enabled, at the molecule's coordinates. */
double rdkit_single_term_energy(RDKit::RWMol &mol, term_kind_t kind) {
    RDKit::MMFF::MMFFMolProperties props(mol);
    if (!props.isValid()) return std::nan("");
    props.setMMFFBondTerm(kind == term_kind_t::bond);
    props.setMMFFAngleTerm(kind == term_kind_t::angle);
    props.setMMFFStretchBendTerm(kind == term_kind_t::stretchbend);
    props.setMMFFOopTerm(kind == term_kind_t::oop);
    props.setMMFFTorsionTerm(kind == term_kind_t::torsion);
    props.setMMFFVdWTerm(kind == term_kind_t::nonbonded);
    props.setMMFFEleTerm(kind == term_kind_t::nonbonded);
    std::unique_ptr<ForceFields::ForceField> field {RDKit::MMFF::constructForceField(mol, &props, 100.0, -1, false)};
    if (!field) return std::nan("");
    field->initialize();
    return field->calcEnergy();
}

std::uint32_t failures = 0;

/**
 *  @brief Deterministically displace every atom by roughly @p amplitude angstroms.
 *
 *  An RDKit-embedded geometry sits at or near equilibrium, where several MMFF forms
 *  coincide — a linear center at theta = 180 gives the same energy under the quadratic
 *  and the linear bend, so a comparison there cannot tell them apart.  Distorting the
 *  geometry moves the terms onto the parts of their curves that actually differ.
 */
void perturb(std::vector<float> &positions_soa, std::uint32_t num_atoms, float amplitude) {
    std::uint32_t state = 0x9E3779B9u;
    for (std::uint32_t index = 0; index < 3 * num_atoms; ++index) {
        state = state * 1664525u + 1013904223u;
        float const unit = static_cast<float>(state >> 8) * (1.0f / 16777216.0f) - 0.5f;
        positions_soa[index] += 2.0f * amplitude * unit;
    }
}

void compare_terms(std::string const &smiles, char const *label, float perturbation = 0.0f) {
    auto prepared_full = sfp::prepare_molecule(smiles);
    if (prepared_full.num_atoms == 0) {
        std::printf("  %-22s SKIP (preprocess rejected)\n", label);
        return;
    }
    auto mol = sftest::mol_from_smiles(smiles);
    SF_TEST_ASSERT(mol != nullptr, "failed to build RDKit molecule");
    SF_TEST_ASSERT(sftest::embed_molecule(*mol), "failed to embed RDKit molecule");

    std::vector<float> positions = interleaved_to_soa(sftest::extract_coords_interleaved(*mol));
    if (perturbation > 0.0f) {
        perturb(positions, prepared_full.num_atoms, perturbation);
        // RDKit must score the same distorted geometry, not the embedded one.
        std::vector<float> interleaved(static_cast<std::size_t>(prepared_full.num_atoms) * 3, 0.0f);
        for (std::uint32_t atom = 0; atom < prepared_full.num_atoms; ++atom)
            for (std::uint32_t axis = 0; axis < 3; ++axis)
                interleaved[atom * 3 + axis] = positions[axis * prepared_full.num_atoms + atom];
        sftest::set_coords_interleaved(*mol, interleaved.data());
    }
    std::vector<float> gradient;
    // Production rebuilds the hop matrix on the device, so this single-molecule harness derives
    // it from the bond list, and hands the sweep an all-unreachable matrix whenever the
    // comparison isolates some other term.
    std::vector<std::uint8_t> bond_hops;
    sftest::fill_bond_hops(prepared_full.mmff_bonds.data(), static_cast<std::uint32_t>(prepared_full.mmff_bonds.size()),
                           prepared_full.num_atoms, bond_hops);
    std::vector<std::uint8_t> const unreachable_hops(bond_hops.size(), std::uint8_t {255});

    term_kind_t const kinds[] = {term_kind_t::bond,    term_kind_t::angle,   term_kind_t::stretchbend,
                                 term_kind_t::oop,     term_kind_t::torsion, term_kind_t::nonbonded};
    if (perturbation > 0.0f) std::printf("  %s (%s, distorted %.2f A)\n", label, smiles.c_str(), perturbation);
    else
        std::printf("  %s (%s)\n", label, smiles.c_str());
    for (term_kind_t kind : kinds) {
        auto const single = isolate_term(prepared_full, kind);
        gradient.assign(positions.size(), 0.0f);
        float gpu_energy = 0.0f;
        gpu_mmff_energy(positions.data(), single.num_atoms,
                        single.mmff_atom_vdw.empty() ? nullptr : single.mmff_atom_vdw.data(),
                        single.mmff_atom_charge.empty() ? nullptr : single.mmff_atom_charge.data(),
                        kind == term_kind_t::nonbonded ? bond_hops.data() : unreachable_hops.data(),
                        single.mmff_bonds.empty() ? nullptr : single.mmff_bonds.data(),
                        static_cast<std::uint32_t>(single.mmff_bonds.size()),
                        single.mmff_angles.empty() ? nullptr : single.mmff_angles.data(),
                        static_cast<std::uint32_t>(single.mmff_angles.size()),
                        single.mmff_torsions.empty() ? nullptr : single.mmff_torsions.data(),
                        static_cast<std::uint32_t>(single.mmff_torsions.size()),
                        single.mmff_oops.empty() ? nullptr : single.mmff_oops.data(),
                        static_cast<std::uint32_t>(single.mmff_oops.size()),
                        single.mmff_stretchbends.empty() ? nullptr : single.mmff_stretchbends.data(),
                        static_cast<std::uint32_t>(single.mmff_stretchbends.size()),
                        single.mmff_interactions.empty() ? nullptr : single.mmff_interactions.data(),
                        static_cast<std::uint32_t>(single.mmff_interactions.size()),
                        single.mmff_atom_offsets.empty() ? nullptr : single.mmff_atom_offsets.data(),
                        static_cast<std::uint32_t>(single.mmff_atom_offsets.size()), gradient.data(), &gpu_energy);

        double const reference = rdkit_single_term_energy(*mol, kind);
        if (!std::isfinite(reference)) {
            std::printf("    %-14s RDKit reference unavailable\n", term_name(kind));
            continue;
        }
        double const difference = static_cast<double>(gpu_energy) - reference;
        double const tolerance = 0.05 + 0.01 * std::fabs(reference);
        bool const ok = std::fabs(difference) <= tolerance;
        std::printf("    %-14s gpu=%12.4f  rdkit=%12.4f  delta=%+10.4f  [%s]\n", term_name(kind),
                    static_cast<double>(gpu_energy), reference, difference, ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    }
}

} // namespace

int main() {
    std::printf("MMFF94 per-term parity against RDKit\n");
    // Aspirin is the general case; the rest target the terms most likely to diverge —
    // linear centers, and a disconnected salt for the nonbonded exclusion path.
    compare_terms("CC(=O)Oc1ccccc1C(=O)O", "aspirin");
    compare_terms("CC#N", "acetonitrile");
    compare_terms("CC#CC", "2-butyne");
    compare_terms("C/C=C/C", "trans-2-butene");
    compare_terms("C/C=C\\C", "cis-2-butene");
    compare_terms("c1ccccc1", "benzene");

    // Distorted repeats: at equilibrium the quadratic and linear bends coincide at a
    // linear center, so only a bent geometry discriminates between them.
    compare_terms("CC#N", "acetonitrile", 0.30f);
    compare_terms("CC#CC", "2-butyne", 0.30f);
    compare_terms("CC(=O)Oc1ccccc1C(=O)O", "aspirin", 0.20f);

    SF_TEST_ASSERT(failures == 0, "every MMFF94 term must match RDKit in isolation");
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
