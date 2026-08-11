/**
 *  @file test/mmff94/test_mmff_cpu_port.cpp
 *  @brief Cross-validate the native MMFF CPU port against the existing RDKit-backed path.
 *
 *  Both native extractors are held to the same oracle: a cache-free baseline built from RDKit's own per-term
 *  `getMMFF*Params`. `extract_mmff94_params_native` takes atom types and charges from RDKit, while
 *  `extract_mmff94_params_fully_native` derives them itself with zero runtime RDKit. Both must produce
 *  byte-identical MMFF term arrays on a curated SMILES set; small numerical deltas ≤ 1e-3 are allowed on
 *  empirical-rule fallback terms due to float/double rounding in MMFF.V eqs 18-19 and the torsion rules, and on
 *  the fully-native partial charges, which sum bond-charge increments in `float` where RDKit sums in `double`.
 *  A term count that does not match fails outright; molecules whose terms differ only in value get the ppm
 *  budget of `max_molecule_value_mismatch_ppm_k`.
 */

#include <cmath>   // `std::fabs`
#include <cstring> // `std::memcpy`

#include <sstream>     // RDKit's `AtomTyper.h` uses `std::stringstream` without including it
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::print`, `fmt::println`, `fmt::group_digits`

#include <boost/shared_ptr.hpp>
#include <RDGeneral/types.h>
#include <ForceField/MMFF/Params.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/mmff94/mmff94_extract.hpp"
#include "smallfield/mmff94/mmff94_term_equations.hpp"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_rdkit.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

#include "test/test_rdkit_reference_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sfparam = smallfield::parameters;
namespace sfmmff = smallfield::mmff94;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

using sftest::load_smiles;

static constexpr std::string_view test_name_k = "test_mmff_cpu_port";

namespace smallfield::tests {

static int n_checks_ = 0;
static int n_fails_ = 0;
static int n_mol_pass_ = 0;
static int n_mol_fail_ = 0;
static int n_mol_with_any_fail_ = 0;

/** How many failing molecules to describe before going silent. A cap of three turned a thirteen-molecule
 *  structural class into what read as a single stray molecule, so the default is generous; lower it locally
 *  when triaging one molecule. A shared knobs header is on its way and will own this, at which point the
 *  constant here becomes its default. */
static int max_reported_failing_molecules_ = 32;

/** Ceiling on molecules whose native terms differ in value from the RDKit oracle, in parts per million.
 *  Structural differences — a term count that does not match — are never tolerated, see `n_mol_fail_`. */
static double constexpr max_molecule_value_mismatch_ppm_k = 100.0;

/** Bitwise-identical float comparison (allowing NaN != NaN). */
static bool bit_eq_(float left, float right) {
    std::uint32_t left_bits, right_bits;
    std::memcpy(&left_bits, &left, 4);
    std::memcpy(&right_bits, &right, 4);
    return left_bits == right_bits;
}

/** Near-equal with absolute tolerance — used for empirical-rule terms. */
static bool near_(float left, float right, float tolerance = 1e-3f) {
    if (bit_eq_(left, right)) return true;
    return std::fabs(left - right) <= tolerance * std::max(1.0f, std::max(std::fabs(left), std::fabs(right)));
}

/** Which native extractor produced the terms under test. */
enum class native_extractor_t : std::uint8_t {
    /** `extract_mmff94_params_native` — atom types and partial charges handed in from RDKit. */
    hybrid_k,
    /** `extract_mmff94_params_fully_native` — types and charges derived with zero runtime RDKit. */
    fully_native_k,
};

static char const *name_of_(native_extractor_t extractor) {
    return extractor == native_extractor_t::hybrid_k ? "hybrid" : "fully-native";
}

static bool compare_mol_(std::string_view smiles, native_extractor_t extractor,
                         sfp::etkdg_prepared_molecule_t const &rdkit_result,
                         sfp::etkdg_prepared_molecule_t const &native_result, bool verbose) {
    char const *native_path_label = name_of_(extractor);
    bool ok = true;
    auto complain = [&](std::string_view what) {
        if (verbose && ok) fmt::print("  [{}] {} MISMATCH: ", smiles, native_path_label);
        if (verbose) fmt::print("{} ", what);
        ok = false;
    };

    // Printed once, ahead of the first per-term value mismatch, so every value line below is
    // attributable to a molecule and to the extractor that produced it.
    bool value_header_printed = false;
    auto value_header = [&]() {
        if (!verbose || value_header_printed) return;
        fmt::println("  [{}] {} value mismatches:", smiles, native_path_label);
        value_header_printed = true;
    };

    if (rdkit_result.num_atoms != native_result.num_atoms) complain("num_atoms");
    if (rdkit_result.mmff_bonds.size() != native_result.mmff_bonds.size()) complain("bond_count");
    if (rdkit_result.mmff_angles.size() != native_result.mmff_angles.size()) complain("angle_count");
    if (rdkit_result.mmff_torsions.size() != native_result.mmff_torsions.size()) {
        complain("torsion_count");
        if (verbose) {
            // Dump first few torsions present in rdk but missing from nat
            // and vice versa — index them by (i, j, k, l).
            auto ijkl_of = [](auto const &torsion) {
                return (std::uint32_t)torsion.atom_i | ((std::uint32_t)torsion.atom_j << 8) |
                       ((std::uint32_t)torsion.atom_k << 16) | ((std::uint32_t)torsion.atom_l << 24);
            };
            std::vector<std::uint32_t> rdkit_torsion_keys, native_torsion_keys;
            for (auto const &torsion : rdkit_result.mmff_torsions) rdkit_torsion_keys.push_back(ijkl_of(torsion));
            for (auto const &torsion : native_result.mmff_torsions) native_torsion_keys.push_back(ijkl_of(torsion));
            std::sort(rdkit_torsion_keys.begin(), rdkit_torsion_keys.end());
            std::sort(native_torsion_keys.begin(), native_torsion_keys.end());
            int printed = 0;
            for (auto packed_ijkl : rdkit_torsion_keys) {
                if (!std::binary_search(native_torsion_keys.begin(), native_torsion_keys.end(), packed_ijkl) &&
                    printed < 3) {
                    fmt::println("    rdk-only tor: {}-{}-{}-{}", packed_ijkl & 0xff, (packed_ijkl >> 8) & 0xff,
                                 (packed_ijkl >> 16) & 0xff, (packed_ijkl >> 24) & 0xff);
                    ++printed;
                }
            }
            printed = 0;
            for (auto packed_ijkl : native_torsion_keys) {
                if (!std::binary_search(rdkit_torsion_keys.begin(), rdkit_torsion_keys.end(), packed_ijkl) &&
                    printed < 3) {
                    fmt::println("    nat-only tor: {}-{}-{}-{}", packed_ijkl & 0xff, (packed_ijkl >> 8) & 0xff,
                                 (packed_ijkl >> 16) & 0xff, (packed_ijkl >> 24) & 0xff);
                    ++printed;
                }
            }
        }
    }
    if (rdkit_result.mmff_oops.size() != native_result.mmff_oops.size()) complain("oop_count");
    if (rdkit_result.mmff_stretchbends.size() != native_result.mmff_stretchbends.size()) complain("stbn_count");

    if (!ok) {
        if (verbose)
            fmt::println("\n    sizes: bond {}/{}, ang {}/{}, tor {}/{}, oop {}/{}, sb {}/{}",
                         rdkit_result.mmff_bonds.size(), native_result.mmff_bonds.size(),
                         rdkit_result.mmff_angles.size(), native_result.mmff_angles.size(),
                         rdkit_result.mmff_torsions.size(), native_result.mmff_torsions.size(),
                         rdkit_result.mmff_oops.size(), native_result.mmff_oops.size(),
                         rdkit_result.mmff_stretchbends.size(), native_result.mmff_stretchbends.size());
        return ok;
    }

    // Per-atom VdW + charge
    for (std::size_t atom = 0; atom < rdkit_result.mmff_atom_vdw.size(); ++atom) {
        auto const &rdkit_term = rdkit_result.mmff_atom_vdw[atom];
        auto const &native_term = native_result.mmff_atom_vdw[atom];
        if (!(bit_eq_(rdkit_term.alpha, native_term.alpha) &&
              bit_eq_(rdkit_term.n_effective, native_term.n_effective) &&
              bit_eq_(rdkit_term.a_scale, native_term.a_scale) && bit_eq_(rdkit_term.g_factor, native_term.g_factor) &&
              bit_eq_(rdkit_term.r_star, native_term.r_star) &&
              rdkit_term.donor_acceptor == native_term.donor_acceptor)) {
            if (verbose) {
                value_header();
                fmt::println("    vdw[{}] rdk α={:.4f} r*={:.4f}; nat α={:.4f} r*={:.4f}", atom, rdkit_term.alpha,
                             rdkit_term.r_star, native_term.alpha, native_term.r_star);
            }
            ++n_fails_;
        }
        // The hybrid path is handed RDKit's own `double` charges, so it must reproduce them bit for bit.
        // The fully-native path sums bond-charge increments in `float` where RDKit sums in `double`, so it
        // earns the same 1e-3 tolerance every other natively-computed term in this file already gets.
        bool const charge_matches =
            extractor == native_extractor_t::hybrid_k
                ? bit_eq_(rdkit_result.mmff_atom_charge[atom], native_result.mmff_atom_charge[atom])
                : near_(rdkit_result.mmff_atom_charge[atom], native_result.mmff_atom_charge[atom]);
        if (!charge_matches) {
            if (verbose) {
                value_header();
                fmt::println("    charge[{}] rdk={} nat={} Δ={:+.3g}", atom, rdkit_result.mmff_atom_charge[atom],
                             native_result.mmff_atom_charge[atom],
                             native_result.mmff_atom_charge[atom] - rdkit_result.mmff_atom_charge[atom]);
            }
            ++n_fails_;
        }
    }

    // Bonds
    for (std::size_t term = 0; term < rdkit_result.mmff_bonds.size(); ++term) {
        auto const &rdkit_term = rdkit_result.mmff_bonds[term];
        auto const &native_term = native_result.mmff_bonds[term];
        ++n_checks_;
        if (rdkit_term.atom_i != native_term.atom_i || rdkit_term.atom_j != native_term.atom_j) {
            if (verbose) {
                value_header();
                fmt::println("    bond[{}] ij rdk={}-{} nat={}-{}", term, rdkit_term.atom_i, rdkit_term.atom_j,
                             native_term.atom_i, native_term.atom_j);
            }
            ++n_fails_;
            continue;
        }
        if (!near_(rdkit_term.force_constant, native_term.force_constant) ||
            !near_(rdkit_term.equilibrium_distance, native_term.equilibrium_distance)) {
            if (verbose) {
                value_header();
                fmt::println("    bond[{}] kb rdk={:.4f} nat={:.4f}  r0 rdk={:.4f} nat={:.4f}", term,
                             rdkit_term.force_constant, native_term.force_constant, rdkit_term.equilibrium_distance,
                             native_term.equilibrium_distance);
            }
            ++n_fails_;
        }
    }

    // Angles
    for (std::size_t term = 0; term < rdkit_result.mmff_angles.size(); ++term) {
        auto const &rdkit_term = rdkit_result.mmff_angles[term];
        auto const &native_term = native_result.mmff_angles[term];
        ++n_checks_;
        if (rdkit_term.atom_i != native_term.atom_i || rdkit_term.atom_j != native_term.atom_j ||
            rdkit_term.atom_k != native_term.atom_k) {
            if (verbose) {
                value_header();
                fmt::println("    angle[{}] ijk rdk={}-{}-{} nat={}-{}-{}", term, rdkit_term.atom_i,
                             rdkit_term.atom_j, rdkit_term.atom_k, native_term.atom_i, native_term.atom_j,
                             native_term.atom_k);
            }
            ++n_fails_;
            continue;
        }
        if (!near_(rdkit_term.force_constant, native_term.force_constant) ||
            !near_(rdkit_term.equilibrium_angle, native_term.equilibrium_angle)) {
            if (verbose) {
                value_header();
                fmt::println("    angle[{}] ka rdk={:.4f} nat={:.4f} θ₀ rdk={:.4f} nat={:.4f}", term,
                             rdkit_term.force_constant, native_term.force_constant, rdkit_term.equilibrium_angle,
                             native_term.equilibrium_angle);
            }
            ++n_fails_;
        }
    }

    // Torsions
    for (std::size_t term = 0; term < rdkit_result.mmff_torsions.size(); ++term) {
        auto const &rdkit_term = rdkit_result.mmff_torsions[term];
        auto const &native_term = native_result.mmff_torsions[term];
        ++n_checks_;
        if (rdkit_term.atom_i != native_term.atom_i || rdkit_term.atom_j != native_term.atom_j ||
            rdkit_term.atom_k != native_term.atom_k || rdkit_term.atom_l != native_term.atom_l) {
            if (verbose) {
                value_header();
                fmt::println("    tor[{}] ijkl rdk={}-{}-{}-{} nat={}-{}-{}-{}", term, rdkit_term.atom_i,
                             rdkit_term.atom_j, rdkit_term.atom_k, rdkit_term.atom_l, native_term.atom_i,
                             native_term.atom_j, native_term.atom_k, native_term.atom_l);
            }
            ++n_fails_;
            continue;
        }
        if (!near_(rdkit_term.torsion_v1, native_term.torsion_v1) ||
            !near_(rdkit_term.torsion_v2, native_term.torsion_v2) ||
            !near_(rdkit_term.torsion_v3, native_term.torsion_v3)) {
            if (verbose) {
                value_header();
                fmt::println("    tor[{}] ijkl={}-{}-{}-{} V rdk=({:.4f}, {:.4f}, {:.4f}) nat=({:.4f}, {:.4f}, {:.4f})",
                             term, rdkit_term.atom_i, rdkit_term.atom_j, rdkit_term.atom_k, rdkit_term.atom_l,
                             rdkit_term.torsion_v1, rdkit_term.torsion_v2, rdkit_term.torsion_v3,
                             native_term.torsion_v1, native_term.torsion_v2, native_term.torsion_v3);
            }
            ++n_fails_;
        }
    }

    // OOPs
    for (std::size_t term = 0; term < rdkit_result.mmff_oops.size(); ++term) {
        auto const &rdkit_term = rdkit_result.mmff_oops[term];
        auto const &native_term = native_result.mmff_oops[term];
        ++n_checks_;
        if (rdkit_term.atom_i != native_term.atom_i || rdkit_term.atom_j != native_term.atom_j ||
            rdkit_term.atom_k != native_term.atom_k || rdkit_term.atom_l != native_term.atom_l) {
            if (verbose) {
                value_header();
                fmt::println("    oop[{}] ijkl rdk={}-{}-{}-{} nat={}-{}-{}-{}", term, rdkit_term.atom_i,
                             rdkit_term.atom_j, rdkit_term.atom_k, rdkit_term.atom_l, native_term.atom_i,
                             native_term.atom_j, native_term.atom_k, native_term.atom_l);
            }
            ++n_fails_;
            continue;
        }
        if (!near_(rdkit_term.force_constant, native_term.force_constant)) {
            if (verbose) {
                value_header();
                fmt::println("    oop[{}] ijkl={}-{}-{}-{} koop rdk={:.4f} nat={:.4f}", term, rdkit_term.atom_i,
                             rdkit_term.atom_j, rdkit_term.atom_k, rdkit_term.atom_l, rdkit_term.force_constant,
                             native_term.force_constant);
            }
            ++n_fails_;
        }
    }

    // Stretch-bends
    for (std::size_t term = 0; term < rdkit_result.mmff_stretchbends.size(); ++term) {
        auto const &rdkit_term = rdkit_result.mmff_stretchbends[term];
        auto const &native_term = native_result.mmff_stretchbends[term];
        ++n_checks_;
        if (rdkit_term.atom_i != native_term.atom_i || rdkit_term.atom_j != native_term.atom_j ||
            rdkit_term.atom_k != native_term.atom_k) {
            if (verbose) {
                value_header();
                fmt::println("    sb[{}] ijk rdk={}-{}-{} nat={}-{}-{}", term, rdkit_term.atom_i, rdkit_term.atom_j,
                             rdkit_term.atom_k, native_term.atom_i, native_term.atom_j, native_term.atom_k);
            }
            ++n_fails_;
            continue;
        }
        if (!near_(rdkit_term.kba_ijk, native_term.kba_ijk) || !near_(rdkit_term.kba_kji, native_term.kba_kji) ||
            !near_(rdkit_term.equilibrium_angle, native_term.equilibrium_angle) ||
            !near_(rdkit_term.equilibrium_distance_ij, native_term.equilibrium_distance_ij) ||
            !near_(rdkit_term.equilibrium_distance_kj, native_term.equilibrium_distance_kj)) {
            if (verbose) {
                value_header();
                fmt::println("    sb[{}] ijk={}-{}-{} rdk k=({:.4f}, {:.4f}) θ₀={:.4f} r0=({:.4f}, {:.4f}) " //
                             "nat k=({:.4f}, {:.4f}) θ₀={:.4f} r0=({:.4f}, {:.4f})",
                             term, rdkit_term.atom_i, rdkit_term.atom_j, rdkit_term.atom_k, rdkit_term.kba_ijk,
                             rdkit_term.kba_kji, rdkit_term.equilibrium_angle, rdkit_term.equilibrium_distance_ij,
                             rdkit_term.equilibrium_distance_kj, native_term.kba_ijk, native_term.kba_kji,
                             native_term.equilibrium_angle, native_term.equilibrium_distance_ij,
                             native_term.equilibrium_distance_kj);
            }
            ++n_fails_;
        }
    }

    // CSR interactions are built only by the native path in this test
    // (the rdkit_direct_extract baseline leaves them empty). Skip that
    // comparison — the native CSR is validated implicitly by the bonded
    // term counts above plus cross-checks in downstream ETKDG kernels.

    return ok;
}

/** Build a "ground truth" prepared molecule by calling RDKit's per-term
 *  `getMMFF*Params` directly — bypasses the `mmff_param_cache`, whose
 *  torsion/angle/stbn keys miss ring context. The native CPU port is
 *  validated against this, not against `extract_mmff94_params_via_rdkit`, so that
 *  any latent cache bugs don't contaminate the comparison. */
static void rdkit_direct_extract_(RDKit::ROMol &molecule, sft::molecule_descriptor_t const &descriptor,
                                  sfp::etkdg_prepared_molecule_t &out) {
    if (!molecule.hasProp(RDKit::common_properties::_MMFFSanitized)) {
        molecule.setProp(RDKit::common_properties::_MMFFSanitized, 1, true);
    }
    RDKit::MMFF::MMFFMolProperties props(molecule);
    if (!props.isValid()) return;
    std::uint32_t const num_atoms = descriptor.num_atoms;
    out.mmff_atom_vdw.resize(num_atoms);
    out.mmff_atom_charge.resize(num_atoms);
    auto const *vdw_collection = RDKit::MMFF::DefaultParameters::getMMFFVdW();
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        auto const *vdw_params = (*vdw_collection)(props.getMMFFAtomType(atom));
        sft::mmff94_atom_vdw_params_t vdw_params_out {};
        if (vdw_params) {
            vdw_params_out.alpha = (float)vdw_params->alpha_i;
            vdw_params_out.n_effective = (float)vdw_params->N_i;
            vdw_params_out.a_scale = (float)vdw_params->A_i;
            vdw_params_out.g_factor = (float)vdw_params->G_i;
            vdw_params_out.r_star = (float)vdw_params->R_star;
            vdw_params_out.donor_acceptor = (vdw_params->DA == 'D') ? 2 : (vdw_params->DA == 'A') ? 1 : 0;
        }
        out.mmff_atom_vdw[atom] = vdw_params_out;
        out.mmff_atom_charge[atom] = (float)props.getMMFFPartialCharge(atom);
    }
    auto find_bond = [&](std::uint32_t from_atom, std::uint32_t to_atom) -> std::int32_t {
        for (std::uint16_t neighbor_slot = descriptor.neighbor_offset[from_atom];
             neighbor_slot < descriptor.neighbor_offset[from_atom + 1]; ++neighbor_slot)
            if (descriptor.neighbor_atom[neighbor_slot] == to_atom) return descriptor.neighbor_bond[neighbor_slot];
        return -1;
    };
    for (std::uint32_t bond = 0; bond < descriptor.num_bonds; ++bond) {
        std::uint32_t atom_i = descriptor.bond_atom_i[bond], atom_j = descriptor.bond_atom_j[bond];
        unsigned bond_type;
        ForceFields::MMFF::MMFFBond bond_params;
        if (!props.getMMFFBondStretchParams(molecule, atom_i, atom_j, bond_type, bond_params)) continue;
        out.mmff_bonds.push_back({.atom_i = (std::uint8_t)atom_i,
                                  .atom_j = (std::uint8_t)atom_j,
                                  .padding = {0, 0},
                                  .force_constant = (float)bond_params.kb,
                                  .equilibrium_distance = (float)bond_params.r0,
                                  .padding_float = 0.0f});
    }
    for (std::uint32_t center_atom = 0; center_atom < num_atoms; ++center_atom) {
        std::uint16_t neighbor_start = descriptor.neighbor_offset[center_atom],
                      neighbor_end = descriptor.neighbor_offset[center_atom + 1];
        for (std::uint16_t first_neighbor_slot = neighbor_start; first_neighbor_slot < neighbor_end;
             ++first_neighbor_slot)
            for (std::uint16_t second_neighbor_slot = first_neighbor_slot + 1; second_neighbor_slot < neighbor_end;
                 ++second_neighbor_slot) {
                std::uint32_t atom_i = descriptor.neighbor_atom[first_neighbor_slot],
                              atom_k = descriptor.neighbor_atom[second_neighbor_slot];
                if (find_bond(atom_i, center_atom) < 0 || find_bond(center_atom, atom_k) < 0) continue;
                unsigned angle_type;
                ForceFields::MMFF::MMFFAngle angle_params;
                if (!props.getMMFFAngleBendParams(molecule, atom_i, center_atom, atom_k, angle_type, angle_params))
                    continue;
                auto const *prop_center = (*RDKit::MMFF::DefaultParameters::getMMFFProp())(
                    props.getMMFFAtomType(center_atom));
                std::uint8_t const is_linear = (prop_center && prop_center->linh) ? 1u : 0u;
                out.mmff_angles.push_back({.atom_i = (std::uint8_t)atom_i,
                                           .atom_j = (std::uint8_t)center_atom,
                                           .atom_k = (std::uint8_t)atom_k,
                                           .is_linear = is_linear,
                                           .force_constant = (float)angle_params.ka,
                                           .equilibrium_angle = (float)(angle_params.theta0 * M_PI / 180.0),
                                           .padding_float = 0.0f});
            }
    }
    for (std::uint32_t bond = 0; bond < descriptor.num_bonds; ++bond) {
        std::uint32_t atom_j = descriptor.bond_atom_i[bond], atom_k = descriptor.bond_atom_j[bond];
        std::uint16_t j_neighbor_start = descriptor.neighbor_offset[atom_j],
                      j_neighbor_end = descriptor.neighbor_offset[atom_j + 1];
        std::uint16_t k_neighbor_start = descriptor.neighbor_offset[atom_k],
                      k_neighbor_end = descriptor.neighbor_offset[atom_k + 1];
        for (std::uint16_t j_neighbor_slot = j_neighbor_start; j_neighbor_slot < j_neighbor_end; ++j_neighbor_slot) {
            std::uint32_t atom_i = descriptor.neighbor_atom[j_neighbor_slot];
            if (atom_i == atom_k) continue;
            for (std::uint16_t k_neighbor_slot = k_neighbor_start; k_neighbor_slot < k_neighbor_end;
                 ++k_neighbor_slot) {
                std::uint32_t atom_l = descriptor.neighbor_atom[k_neighbor_slot];
                if (atom_l == atom_j) continue;
                unsigned torsion_type;
                ForceFields::MMFF::MMFFTor torsion_params;
                if (!props.getMMFFTorsionParams(molecule, atom_i, atom_j, atom_k, atom_l, torsion_type, torsion_params))
                    continue;
                out.mmff_torsions.push_back({.atom_i = (std::uint8_t)atom_i,
                                             .atom_j = (std::uint8_t)atom_j,
                                             .atom_k = (std::uint8_t)atom_k,
                                             .atom_l = (std::uint8_t)atom_l,
                                             .torsion_v1 = (float)torsion_params.V1,
                                             .torsion_v2 = (float)torsion_params.V2,
                                             .torsion_v3 = (float)torsion_params.V3});
            }
        }
    }
    for (std::uint32_t center_atom = 0; center_atom < num_atoms; ++center_atom) {
        if (descriptor.degree[center_atom] < 3) continue;
        std::uint16_t neighbor_start = descriptor.neighbor_offset[center_atom];
        std::uint8_t neighbor_cache[8];
        std::uint32_t neighbor_count = 0;
        for (std::uint16_t neighbor_slot = neighbor_start;
             neighbor_slot < descriptor.neighbor_offset[center_atom + 1] && neighbor_count < 8; ++neighbor_slot)
            neighbor_cache[neighbor_count++] = descriptor.neighbor_atom[neighbor_slot];
        for (std::uint32_t apex_neighbor = 0; apex_neighbor < neighbor_count; ++apex_neighbor) {
            std::uint8_t plane_atoms[2];
            std::uint32_t plane_count = 0;
            for (std::uint32_t plane_candidate = 0; plane_candidate < neighbor_count && plane_count < 2;
                 ++plane_candidate)
                if (plane_candidate != apex_neighbor) plane_atoms[plane_count++] = neighbor_cache[plane_candidate];
            if (plane_count < 2) continue;
            ForceFields::MMFF::MMFFOop oop_params;
            if (!props.getMMFFOopBendParams(molecule, neighbor_cache[apex_neighbor], center_atom, plane_atoms[0],
                                            plane_atoms[1], oop_params))
                continue;
            out.mmff_oops.push_back({.atom_i = neighbor_cache[apex_neighbor],
                                     .atom_j = (std::uint8_t)center_atom,
                                     .atom_k = plane_atoms[0],
                                     .atom_l = plane_atoms[1],
                                     .force_constant = (float)oop_params.koop,
                                     .padding = {0.0f, 0.0f}});
        }
    }
    for (std::uint32_t center_atom = 0; center_atom < num_atoms; ++center_atom) {
        std::uint16_t neighbor_start = descriptor.neighbor_offset[center_atom],
                      neighbor_end = descriptor.neighbor_offset[center_atom + 1];
        for (std::uint16_t first_neighbor_slot = neighbor_start; first_neighbor_slot < neighbor_end;
             ++first_neighbor_slot)
            for (std::uint16_t second_neighbor_slot = first_neighbor_slot + 1; second_neighbor_slot < neighbor_end;
                 ++second_neighbor_slot) {
                std::uint32_t atom_i = descriptor.neighbor_atom[first_neighbor_slot],
                              atom_k = descriptor.neighbor_atom[second_neighbor_slot];
                if (find_bond(atom_i, center_atom) < 0 || find_bond(center_atom, atom_k) < 0) continue;
                unsigned stretch_bend_type;
                ForceFields::MMFF::MMFFStbn stretch_bend_params;
                ForceFields::MMFF::MMFFBond bond_params[2];
                ForceFields::MMFF::MMFFAngle angle_params;
                if (!props.getMMFFStretchBendParams(molecule, atom_i, center_atom, atom_k, stretch_bend_type,
                                                    stretch_bend_params, bond_params, angle_params))
                    continue;
                out.mmff_stretchbends.push_back({.atom_i = (std::uint8_t)atom_i,
                                                 .atom_j = (std::uint8_t)center_atom,
                                                 .atom_k = (std::uint8_t)atom_k,
                                                 .padding = 0,
                                                 .kba_ijk = (float)stretch_bend_params.kbaIJK,
                                                 .kba_kji = (float)stretch_bend_params.kbaKJI,
                                                 .equilibrium_angle = (float)(angle_params.theta0 * M_PI / 180.0),
                                                 .equilibrium_distance_ij = (float)bond_params[0].r0,
                                                 .equilibrium_distance_kj = (float)bond_params[1].r0,
                                                 .padding_float = 0.0f});
            }
    }
}

static void run_one_(std::string_view smiles, sfparam::mmff94_tables_view_t const &tables, bool verbose) {
    // Native + RDKit preprocessing share the same SmilesToMol/addHs/build_descriptor path.
    std::unique_ptr<RDKit::RWMol> molecule;
    try {
        std::string smiles_string {smiles};
        RDKit::SmilesParserParams params;
        params.sanitize = true;
        params.removeHs = false;
        params.strictCXSMILES = false;
        molecule.reset(RDKit::SmilesToMol(smiles_string, params));
    }
    catch (...) {
        return;
    }
    if (!molecule) return;
    try {
        RDKit::MolOps::addHs(*molecule);
    }
    catch (...) {
        return;
    }
    std::uint32_t const num_atoms = molecule->getNumAtoms();
    if (num_atoms > 128) return;

    sft::molecule_descriptor_t descriptor;
    if (!sft::build_descriptor(*molecule, descriptor)) return;

    // RDKit direct (cache-free) baseline.
    sfp::etkdg_prepared_molecule_t rdk_result;
    rdk_result.num_atoms = num_atoms;
    rdkit_direct_extract_(*molecule, descriptor, rdk_result);

    // Native path — pull atom types + charges from MMFFMolProperties (same
    // source as the RDKit path).
    if (!molecule->hasProp(RDKit::common_properties::_MMFFSanitized)) {
        molecule->setProp(RDKit::common_properties::_MMFFSanitized, 1, true);
    }
    RDKit::MMFF::MMFFMolProperties props(*molecule);
    if (!props.isValid()) return;

    std::vector<std::uint8_t> atom_types(num_atoms);
    std::vector<float> charges(num_atoms);
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        atom_types[atom] = static_cast<std::uint8_t>(props.getMMFFAtomType(atom));
        charges[atom] = static_cast<float>(props.getMMFFPartialCharge(atom));
    }

    sfp::etkdg_prepared_molecule_t nat_result;
    nat_result.num_atoms = num_atoms;
    sfmmff::extract_mmff94_params_native(descriptor, atom_types, charges, tables, nat_result);

    // Fully-native: zero runtime RDKit. Its terms are held to the same
    // RDKit-derived oracle as the hybrid path.
    sfp::etkdg_prepared_molecule_t fully_native_result;
    fully_native_result.num_atoms = num_atoms;
    bool const extracted = sfmmff::extract_mmff94_params_fully_native(descriptor, tables, fully_native_result);

    int const fails_before = n_fails_;
    bool ok = compare_mol_(smiles, sftest::native_extractor_t::hybrid_k, rdk_result, nat_result, verbose);
    if (!extracted) {
        if (verbose) fmt::println("  [{}] fully-native extraction REFUSED (typer failure)", smiles);
        ok = false;
    }
    else if (!compare_mol_(smiles, sftest::native_extractor_t::fully_native_k, rdk_result, fully_native_result,
                           verbose))
        ok = false;
    if (ok && n_fails_ == fails_before) ++n_mol_pass_;
    else if (!ok) ++n_mol_fail_;
    if (n_fails_ > fails_before) ++n_mol_with_any_fail_;
}

} // namespace smallfield::tests

using sftest::n_checks_;
using sftest::n_fails_;
using sftest::n_mol_fail_;
using sftest::n_mol_pass_;
using sftest::n_mol_with_any_fail_;
using sftest::run_one_;

int main(int argc, char **argv) {
    fmt::println("test_mmff_cpu_port: native MMFF term extractor vs the RDKit-backed baseline");

    sfparam::mmff94_tables_host_t tables_host;
    sfparam::extract_mmff94_tables_from_rdkit(tables_host);
    sfparam::mmff94_tables_view_t const tables = tables_host.view();
    fmt::println("extracted tables: vdw={} bond={} angle={} tor={} oop={} stbn={} dfsb={}",
                 fmt::group_digits(tables_host.vdw.size()), fmt::group_digits(tables_host.bond.size()),
                 fmt::group_digits(tables_host.angle.size()), fmt::group_digits(tables_host.torsion.size()),
                 fmt::group_digits(tables_host.oop.size()), fmt::group_digits(tables_host.stbn.size()),
                 fmt::group_digits(tables_host.dfsb.size()));

    // Curated handful of drug-like SMILES (aspirin, caffeine, ibuprofen,
    // paracetamol, benzodiazepine, adenine, imidazolium, carboxylate).
    std::vector<std::string> probe = {
        "CC(=O)OC1=CC=CC=C1C(=O)O",      // aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  // caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", // ibuprofen
        "CC(=O)NC1=CC=C(O)C=C1",         // paracetamol
        "O=C1NC(=O)C2=CC=CC=C2N1",       // phthalimidine-ish
        "Nc1ncnc2n(cnc12)",              // adenine
        "[NH2+]=C(N)N",                  // guanidinium
        "CC(=O)[O-]",                    // acetate
        "c1cc[nH+]c(c1)",                // pyridinium
        "C1CCOC1",                       // tetrahydrofuran
        "C1CC1",                         // cyclopropane
        "c1ccc2[nH]ccc2c1",              // indole
        // Regression witnesses, one per way the typer used to read the descriptor wrong. Terms derived from
        // a wrong atom type are wrong terms, so the fully-native extractor fails these before the fix too.
        "CC1=CC=CC=[N+]1[O-]",                                              // aromatic N-oxide: O is 32, not 35
        "CC1=C(C=CN=C1C2=CC(=CC(=C2)Cl)[NH+](O)[O-])CC#N",                  // hydroxylammonium O the typer refused
        "C1=CC=C2C=C(C=CC2=C1)NC3=CC#CC=C3",                                // triple bond inside an aromatic ring
        "COC(=O)C(CC1=CC=CC=C1)N=C2N[N+]#CS2",                              // the same, in a five-ring
        "CC(C)NC(=O)C1=CC(=C(C=C1)C2=C3C=CC(=O)C=C3OC4=C2C=CC(=C4)O)C(=O)O" // quinoid single bond, fused system
    };

    // Optional: a SMILES file for larger coverage
    std::vector<std::string> extra;
    std::size_t const max_smiles_to_load = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 5000;
    if (argc > 1) extra = load_smiles(argv[1], max_smiles_to_load);
    fmt::println("probing {} curated + {} external SMILES", fmt::group_digits(probe.size()),
                 fmt::group_digits(extra.size()));

    bool const verbose = (argc <= 1);
    for (auto const &smiles : probe) run_one_(smiles, tables, verbose);
    // Print first few failures for triage, then go silent.
    int fail_prints = 0;
    for (auto const &smiles : extra) {
        int const fail_before = n_fails_;
        int const mfail_before = n_mol_fail_;
        bool const verbose_for_molecule = (fail_prints < sftest::max_reported_failing_molecules_);
        run_one_(smiles, tables, verbose_for_molecule);
        bool const had_fail = n_fails_ > fail_before || n_mol_fail_ > mfail_before;
        if (had_fail && fail_prints < sftest::max_reported_failing_molecules_) {
            fmt::println("  (above is from SMILES: {})", smiles);
            ++fail_prints;
        }
    }

    std::size_t const total_mols = n_mol_pass_ + n_mol_fail_ + n_mol_with_any_fail_;
    fmt::println("molecules: {} size-pass / {} size-fail / {} value-fail (checks={}, term failures={})",
                 fmt::group_digits(n_mol_pass_ + n_mol_with_any_fail_), fmt::group_digits(n_mol_fail_),
                 fmt::group_digits(n_mol_with_any_fail_), fmt::group_digits(n_checks_), fmt::group_digits(n_fails_));
    if (total_mols > 0) {
        double const mol_fail_ppm = 1.0e6 * (double)(n_mol_fail_ + n_mol_with_any_fail_) / (double)total_mols;
        double const term_fail_ppm = (n_checks_ > 0) ? 1.0e6 * (double)n_fails_ / (double)n_checks_ : 0.0;
        fmt::println("ppm: molecules with any failure={:.3f}, term failures={:.3f}", mol_fail_ppm, term_fail_ppm);
    }
    // Structural mismatches are never tolerated; value differences get the ppm budget the file header
    // documents. Exotic structures — pentacoordinate P, porphyrins — can exceed it and should be triaged
    // on the full 10M USearch set rather than absorbed by loosening this.
    if (n_mol_fail_ > 0)
        return sftest::report_fail(test_name_k, "{} of {} molecules differ structurally from the RDKit baseline",
                                   n_mol_fail_, total_mols);
    double const value_mismatch_ppm = (total_mols > 0) ? 1.0e6 * (double)n_mol_with_any_fail_ / (double)total_mols
                                                       : 0.0;
    if (value_mismatch_ppm > sftest::max_molecule_value_mismatch_ppm_k)
        return sftest::report_fail(test_name_k, "{} of {} molecules differ in value, {:.3f} ppm > {:.3f} ppm",
                                   n_mol_with_any_fail_, total_mols, value_mismatch_ppm,
                                   sftest::max_molecule_value_mismatch_ppm_k);
    return sftest::report_pass(test_name_k);
}
