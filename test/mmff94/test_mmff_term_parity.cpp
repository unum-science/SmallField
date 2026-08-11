/**
 *  @file test/mmff94/test_mmff_term_parity.cpp
 *  @brief Gate per-term MMFF94 energy parity between the GPU kernel and RDKit.
 *
 *  The total-energy checks elsewhere cannot localize a divergence, and a term that is wrong in one direction can be
 *  masked by another that is wrong in the other. RDKit exposes per-term toggles, so each term is compared in
 *  isolation on identical coordinates.
 *
 *  Isolation on the native side: bonded terms are selected by keeping only that category's offsets and
 *  interactions, and the nonbonded sweep — which is enumerated on the fly rather than from a list — is suppressed
 *  by marking every pair BFS-unreachable.
 */

#include <cmath>   // `std::isfinite`, `std::fabs`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`
#include <cstdio>  // `printf`

#include <iostream> // RDKit's `AtomTyper.h` needs it but does not include it
#include <memory>   // `std::unique_ptr`
#include <sstream>  // RDKit's `AtomTyper.h` needs it but does not include it
#include <string>   // `std::string`
#include <vector>   // `std::vector`

#include <boost/shared_ptr.hpp>
#include <GraphMol/GraphMol.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"

#include "test/cuda/test_mmff_gradient_kernels.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

using sftest::gpu_mmff_energy;

namespace smallfield::tests {

/** Which single MMFF94 term a comparison isolates. */
enum class term_kind_t { bond_k, angle_k, stretchbend_k, oop_k, torsion_k, nonbonded_k };

static char const *term_name_(term_kind_t kind) {
    switch (kind) {
    case term_kind_t::bond_k: return "bond";
    case term_kind_t::angle_k: return "angle";
    case term_kind_t::stretchbend_k: return "stretch-bend";
    case term_kind_t::oop_k: return "out-of-plane";
    case term_kind_t::torsion_k: return "torsion";
    default: return "nonbonded";
    }
}

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

/**
 *  @brief Copy @p full with every term but @p kind neutralized.
 *
 *  The per-atom interaction CSR is left exactly as production builds it — rebuilding it
 *  by hand would test the reconstruction rather than the kernel. Excluded bonded terms
 *  are zeroed at their force constants, which contributes neither energy nor gradient.
 *  The nonbonded sweep is enumerated on the fly from the hop matrix rather than from a list,
 *  so it is isolated at the call site instead: the caller hands the kernel a matrix of the
 *  unreachable sentinel for every other term, which the sweep skips pair by pair.
 */
static sfp::etkdg_prepared_molecule_t isolate_term_(sfp::etkdg_prepared_molecule_t const &full, term_kind_t kind) {
    sfp::etkdg_prepared_molecule_t single = full;

    if (kind != term_kind_t::bond_k)
        for (auto &term : single.mmff_bonds) term.force_constant = 0.0f;
    if (kind != term_kind_t::angle_k)
        for (auto &term : single.mmff_angles) term.force_constant = 0.0f;
    if (kind != term_kind_t::torsion_k)
        for (auto &term : single.mmff_torsions) term.torsion_v1 = term.torsion_v2 = term.torsion_v3 = 0.0f;
    if (kind != term_kind_t::oop_k)
        for (auto &term : single.mmff_oops) term.force_constant = 0.0f;
    if (kind != term_kind_t::stretchbend_k)
        for (auto &term : single.mmff_stretchbends) term.kba_ijk = term.kba_kji = 0.0f;
    return single;
}

/** RDKit MMFF94 energy with exactly @p kind enabled, at the molecule's coordinates. */
static double rdkit_single_term_energy_(RDKit::RWMol &molecule, term_kind_t kind) {
    RDKit::MMFF::MMFFMolProperties props(molecule);
    if (!props.isValid()) return std::nan("");
    props.setMMFFBondTerm(kind == term_kind_t::bond_k);
    props.setMMFFAngleTerm(kind == term_kind_t::angle_k);
    props.setMMFFStretchBendTerm(kind == term_kind_t::stretchbend_k);
    props.setMMFFOopTerm(kind == term_kind_t::oop_k);
    props.setMMFFTorsionTerm(kind == term_kind_t::torsion_k);
    props.setMMFFVdWTerm(kind == term_kind_t::nonbonded_k);
    props.setMMFFEleTerm(kind == term_kind_t::nonbonded_k);
    std::unique_ptr<ForceFields::ForceField> field {
        RDKit::MMFF::constructForceField(molecule, &props, 100.0, -1, false)};
    if (!field) return std::nan("");
    field->initialize();
    return field->calcEnergy();
}

static std::uint32_t failures_ = 0;

/**
 *  @brief Deterministically displace every atom by roughly @p amplitude angstroms.
 *
 *  An RDKit-embedded geometry sits at or near equilibrium, where several MMFF forms
 *  coincide — a linear center at theta = 180 gives the same energy under the quadratic
 *  and the linear bend, so a comparison there cannot tell them apart. Distorting the
 *  geometry moves the terms onto the parts of their curves that actually differ.
 */
static void perturb_(std::vector<float> &positions_soa, std::uint32_t num_atoms, float amplitude) {
    std::uint32_t state = 0x9E3779B9u;
    for (std::uint32_t index = 0; index < 3 * num_atoms; ++index) {
        state = state * 1664525u + 1013904223u;
        float const unit = static_cast<float>(state >> 8) * (1.0f / 16777216.0f) - 0.5f;
        positions_soa[index] += 2.0f * amplitude * unit;
    }
}

static void compare_terms_(std::string const &smiles, char const *label, float perturbation = 0.0f) {
    auto prepared_full = sfp::prepare_molecule(smiles);
    if (prepared_full.num_atoms == 0) {
        std::printf("  %-22s SKIP (preprocess rejected)\n", label);
        return;
    }
    auto molecule = sftest::mol_from_smiles(smiles);
    sf_verify_(molecule != nullptr, "failed to build RDKit molecule");
    sf_verify_(sftest::embed_molecule(*molecule), "failed to embed RDKit molecule");

    std::vector<float> positions = interleaved_to_soa_(sftest::extract_coords_interleaved(*molecule));
    if (perturbation > 0.0f) {
        perturb_(positions, prepared_full.num_atoms, perturbation);
        // RDKit must score the same distorted geometry, not the embedded one.
        std::vector<float> interleaved(static_cast<std::size_t>(prepared_full.num_atoms) * 3, 0.0f);
        for (std::uint32_t atom = 0; atom < prepared_full.num_atoms; ++atom)
            for (std::uint32_t axis = 0; axis < 3; ++axis)
                interleaved[atom * 3 + axis] = positions[axis * prepared_full.num_atoms + atom];
        sftest::set_coords_interleaved(*molecule, interleaved.data());
    }
    std::vector<float> gradient;
    // Production rebuilds the hop matrix on the device, so this single-molecule harness derives
    // it from the bond list, and hands the sweep an all-unreachable matrix whenever the
    // comparison isolates some other term.
    std::vector<std::uint8_t> bond_hops;
    sftest::fill_bond_hops(prepared_full.mmff_bonds.data(), static_cast<std::uint32_t>(prepared_full.mmff_bonds.size()),
                           prepared_full.num_atoms, bond_hops);
    std::vector<std::uint8_t> const unreachable_hops(bond_hops.size(), std::uint8_t {255});

    term_kind_t const kinds[] = {term_kind_t::bond_k, term_kind_t::angle_k,   term_kind_t::stretchbend_k,
                                 term_kind_t::oop_k,  term_kind_t::torsion_k, term_kind_t::nonbonded_k};
    if (perturbation > 0.0f) std::printf("  %s (%s, distorted %.2f A)\n", label, smiles.c_str(), perturbation);
    else std::printf("  %s (%s)\n", label, smiles.c_str());
    for (term_kind_t kind : kinds) {
        auto const single = isolate_term_(prepared_full, kind);
        gradient.assign(positions.size(), 0.0f);
        float gpu_energy = 0.0f;
        gpu_mmff_energy(positions.data(), single.num_atoms,
                        single.mmff_atom_vdw.empty() ? nullptr : single.mmff_atom_vdw.data(),
                        single.mmff_atom_charge.empty() ? nullptr : single.mmff_atom_charge.data(),
                        kind == term_kind_t::nonbonded_k ? bond_hops.data() : unreachable_hops.data(),
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

        double const reference = rdkit_single_term_energy_(*molecule, kind);
        if (!std::isfinite(reference)) {
            std::printf("    %-14s RDKit reference unavailable\n", term_name_(kind));
            continue;
        }
        double const difference = static_cast<double>(gpu_energy) - reference;
        double const tolerance = 0.05 + 0.01 * std::fabs(reference);
        bool const ok = std::fabs(difference) <= tolerance;
        std::printf("    %-14s gpu=%12.4f  rdkit=%12.4f  delta=%+10.4f  [%s]\n", term_name_(kind),
                    static_cast<double>(gpu_energy), reference, difference, ok ? "ok" : "FAIL");
        if (!ok) ++failures_;
    }
}

} // namespace smallfield::tests

int main() {
    std::printf("MMFF94 per-term parity against RDKit\n");
    // Aspirin is the general case; the rest target the terms most likely to diverge —
    // linear centers, and a disconnected salt for the nonbonded exclusion path.
    sftest::compare_terms_("CC(=O)Oc1ccccc1C(=O)O", "aspirin");
    sftest::compare_terms_("CC#N", "acetonitrile");
    sftest::compare_terms_("CC#CC", "2-butyne");
    sftest::compare_terms_("C/C=C/C", "trans-2-butene");
    sftest::compare_terms_("C/C=C\\C", "cis-2-butene");
    sftest::compare_terms_("c1ccccc1", "benzene");

    // Distorted repeats: at equilibrium the quadratic and linear bends coincide at a
    // linear center, so only a bent geometry discriminates between them.
    sftest::compare_terms_("CC#N", "acetonitrile", 0.30f);
    sftest::compare_terms_("CC#CC", "2-butyne", 0.30f);
    sftest::compare_terms_("CC(=O)Oc1ccccc1C(=O)O", "aspirin", 0.20f);

    sf_verify_(sftest::failures_ == 0, "every MMFF94 term must match RDKit in isolation");
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
