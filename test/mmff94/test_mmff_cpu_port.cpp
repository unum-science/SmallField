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

#include <cstdint> // `std::uint8_t`, `std::uint64_t`
#include <cstdio>  // `std::fflush`, `stderr`

#include <algorithm>   // `std::binary_search`, `std::sort`
#include <array>       // `std::array`
#include <memory>      // `std::unique_ptr`
#include <sstream>     // RDKit's `AtomTyper.h` uses `std::stringstream` without including it
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::print`, `fmt::println`

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
#include "smallfield/types/views.hpp"

#include "test/test_rdkit_reference_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sfparam = smallfield::parameters;
namespace sfmmff = smallfield::mmff94;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_mmff_cpu_port";

namespace smallfield::tests {

/** How many failing molecules to describe before going silent. A cap of three turned a thirteen-molecule
 *  structural class into what read as a single stray molecule, so the default is generous; lower it locally
 *  when triaging one molecule. */
static int constexpr max_reported_failing_molecules_k = 32;

/** Ceiling on molecules whose native terms differ in value from the RDKit oracle, in parts per million.
 *  Structural differences — a term count that does not match — are never tolerated. */
static double constexpr max_molecule_value_mismatch_ppm_k = 100.0;

/** Atom ceiling this comparison holds itself to, below `types::max_atoms_per_molecule_k`; a larger
 *  molecule is skipped rather than compared. */
static std::uint32_t constexpr max_atoms_compared_k = 128;

/**
 *  @brief How one molecule compared against the RDKit oracle.
 *
 *  The three verdicts partition the molecules tested, which is the point of naming them: counting a
 *  structural failure and a value failure in two independent buckets double-counts any molecule that is
 *  both, and every ppm below is a fraction of that total.
 */
enum class molecule_verdict_t : std::uint8_t {
    matched_k,
    value_differs_k,
    structure_differs_k,
};

/** Whether the two term sets have the same shape: the same atom count and the same number of each term. */
enum class term_structure_t : std::uint8_t {
    matches_k,
    differs_k,
};

/** Whether a comparison narrates the differences it finds, or only counts them. */
enum class narration_t : std::uint8_t {
    silent_k,
    narrated_k,
};

/** Whether a sweep narrates every molecule it compares, or only the first molecules that differ. */
enum class narration_policy_t : std::uint8_t {
    every_molecule_k,
    first_failures_k,
};

/**
 *  @brief Why a SMILES never reached the comparison, or that it did.
 *
 *  Each refusal names a molecule about which nothing was measured, so it stays out of the ppm
 *  denominator instead of counting as a match and diluting the rate.
 */
enum class molecule_intake_t : std::uint8_t {
    comparable_k,
    /** RDKit refused the SMILES, or refused to add hydrogens to it. */
    unparsable_k,
    /** More atoms than `max_atoms_compared_k`. */
    above_atom_ceiling_k,
    /** `build_descriptor` refused the molecule. */
    undescribable_k,
    /** `MMFFMolProperties` came back invalid, so there are no reference atom types or charges. */
    untypable_k,
};

/** Counts for one sweep, threaded by reference so the curated probe and the external corpus aggregate. */
struct port_tally_t {
    std::uint64_t term_checks = 0;
    std::uint64_t term_failures = 0;
    /** Molecules per `molecule_verdict_t`, so the three buckets sum to the number tested. */
    std::array<std::uint64_t, 3> molecules {};
    /** Molecules a `molecule_intake_t` refusal kept out of the comparison, and out of every ppm below. */
    std::uint64_t molecules_skipped = 0;

    std::uint64_t &molecules_with(molecule_verdict_t verdict) { return molecules[static_cast<std::size_t>(verdict)]; }
    std::uint64_t molecules_with(molecule_verdict_t verdict) const {
        return molecules[static_cast<std::size_t>(verdict)];
    }
    std::uint64_t total_molecules() const { return molecules[0] + molecules[1] + molecules[2]; }
};

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

/** How many one-sided torsions the structural diagnostic names before it only counts them. */
static int constexpr max_reported_one_sided_torsions_k = 3;

/** Pack a torsion's four atom positions into one sortable key. */
template <typename torsion_type_>
static std::uint32_t packed_ijkl_of_(torsion_type_ const &torsion) {
    return static_cast<std::uint32_t>(torsion.atom_i) | (static_cast<std::uint32_t>(torsion.atom_j) << 8) |
           (static_cast<std::uint32_t>(torsion.atom_k) << 16) | (static_cast<std::uint32_t>(torsion.atom_l) << 24);
}

/**
 *  @brief Name the torsions @p present enumerated and @p absent did not, one line each.
 *
 *  The scan runs to completion and the budget decides only what is printed, so the trailing count is
 *  the true number of one-sided torsions rather than the number that fit on screen.
 *
 *  @param[in] label Which extractor the listed torsions came from.
 *  @param[in] present Sorted packed keys from that extractor.
 *  @param[in] absent Sorted packed keys from the other one.
 */
static void report_one_sided_torsions_(std::string_view label, std::vector<std::uint32_t> const &present,
                                       std::vector<std::uint32_t> const &absent) {
    int one_sided = 0, printed = 0;
    for (auto packed_ijkl : present) {
        if (std::binary_search(absent.begin(), absent.end(), packed_ijkl)) continue;
        ++one_sided;
        if (printed >= max_reported_one_sided_torsions_k) continue;
        fmt::println("    torsion {}-only ijkl={}-{}-{}-{}", label, packed_ijkl & 0xff, (packed_ijkl >> 8) & 0xff,
                     (packed_ijkl >> 16) & 0xff, (packed_ijkl >> 24) & 0xff);
        ++printed;
    }
    fmt::println("    torsion {}-only count={} suppressed={}", label, one_sided, one_sided - printed);
}

static term_structure_t compare_mol_(std::string_view smiles, native_extractor_t extractor,
                                     sfp::etkdg_prepared_molecule_t const &rdkit_result,
                                     sfp::etkdg_prepared_molecule_t const &native_result, narration_t narration,
                                     port_tally_t &tally) {
    char const *native_path_label = name_of_(extractor);
    bool const verbose = narration == narration_t::narrated_k;
    term_structure_t structure = term_structure_t::matches_k;

    // Printed once, ahead of the first differing count, so the run of names that follows is
    // attributable to a molecule and to the extractor that produced it.
    bool structure_header_printed = false;
    auto complain = [&](std::string_view what) {
        structure = term_structure_t::differs_k;
        if (!verbose) return;
        if (!structure_header_printed) {
            fmt::print("  [{}] {} MISMATCH: ", smiles, native_path_label);
            structure_header_printed = true;
        }
        fmt::print("{} ", what);
    };

    // The same idiom for the value section, which is reached only when the shapes agree.
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
            // Which torsions one extractor enumerated and the other did not, keyed by (i, j, k, l).
            std::vector<std::uint32_t> rdkit_torsion_keys, native_torsion_keys;
            for (auto const &torsion : rdkit_result.mmff_torsions)
                rdkit_torsion_keys.push_back(packed_ijkl_of_(torsion));
            for (auto const &torsion : native_result.mmff_torsions)
                native_torsion_keys.push_back(packed_ijkl_of_(torsion));
            std::sort(rdkit_torsion_keys.begin(), rdkit_torsion_keys.end());
            std::sort(native_torsion_keys.begin(), native_torsion_keys.end());
            report_one_sided_torsions_("rdkit", rdkit_torsion_keys, native_torsion_keys);
            report_one_sided_torsions_("native", native_torsion_keys, rdkit_torsion_keys);
        }
    }
    if (rdkit_result.mmff_oops.size() != native_result.mmff_oops.size()) complain("oop_count");
    if (rdkit_result.mmff_stretchbends.size() != native_result.mmff_stretchbends.size()) complain("stbn_count");

    if (structure == term_structure_t::differs_k) {
        if (verbose)
            fmt::println("\n    sizes: bond {}/{}, ang {}/{}, tor {}/{}, oop {}/{}, sb {}/{}",
                         rdkit_result.mmff_bonds.size(), native_result.mmff_bonds.size(),
                         rdkit_result.mmff_angles.size(), native_result.mmff_angles.size(),
                         rdkit_result.mmff_torsions.size(), native_result.mmff_torsions.size(),
                         rdkit_result.mmff_oops.size(), native_result.mmff_oops.size(),
                         rdkit_result.mmff_stretchbends.size(), native_result.mmff_stretchbends.size());
        return structure;
    }

    // Per-atom VdW + charge
    for (std::size_t atom = 0; atom < rdkit_result.mmff_atom_vdw.size(); ++atom) {
        auto const &rdkit_term = rdkit_result.mmff_atom_vdw[atom];
        auto const &native_term = native_result.mmff_atom_vdw[atom];
        if (!(bitwise_equal(rdkit_term.alpha, native_term.alpha) &&
              bitwise_equal(rdkit_term.n_effective, native_term.n_effective) &&
              bitwise_equal(rdkit_term.a_scale, native_term.a_scale) &&
              bitwise_equal(rdkit_term.g_factor, native_term.g_factor) &&
              bitwise_equal(rdkit_term.r_star, native_term.r_star) &&
              rdkit_term.donor_acceptor == native_term.donor_acceptor)) {
            if (verbose) {
                value_header();
                fmt::println("    vdw[{}] rdk α={:.4f} r*={:.4f}; nat α={:.4f} r*={:.4f}", atom, rdkit_term.alpha,
                             rdkit_term.r_star, native_term.alpha, native_term.r_star);
            }
            ++tally.term_failures;
        }
        // The hybrid path is handed RDKit's own `double` charges, so it must reproduce them bit for bit.
        // The fully-native path sums bond-charge increments in `float` where RDKit sums in `double`, so it
        // earns the shared charge tolerance the two typer suites hold the same quantity to.
        bool const charge_matches =
            extractor == native_extractor_t::hybrid_k
                ? bitwise_equal(rdkit_result.mmff_atom_charge[atom], native_result.mmff_atom_charge[atom])
                : equal_within(mmff94_partial_charge_tolerance_k)(rdkit_result.mmff_atom_charge[atom],
                                                                  native_result.mmff_atom_charge[atom]);
        if (!charge_matches) {
            if (verbose) {
                value_header();
                fmt::println("    charge[{}] rdk={} nat={} Δ={:+.3g}", atom, rdkit_result.mmff_atom_charge[atom],
                             native_result.mmff_atom_charge[atom],
                             native_result.mmff_atom_charge[atom] - rdkit_result.mmff_atom_charge[atom]);
            }
            ++tally.term_failures;
        }
    }

    // Bonds
    for (std::size_t term = 0; term < rdkit_result.mmff_bonds.size(); ++term) {
        auto const &rdkit_term = rdkit_result.mmff_bonds[term];
        auto const &native_term = native_result.mmff_bonds[term];
        ++tally.term_checks;
        if (rdkit_term.atom_i != native_term.atom_i || rdkit_term.atom_j != native_term.atom_j) {
            if (verbose) {
                value_header();
                fmt::println("    bond[{}] ij rdk={}-{} nat={}-{}", term, rdkit_term.atom_i, rdkit_term.atom_j,
                             native_term.atom_i, native_term.atom_j);
            }
            ++tally.term_failures;
            continue;
        }
        if (!approximately_equal(rdkit_term.force_constant, native_term.force_constant) ||
            !approximately_equal(rdkit_term.equilibrium_distance, native_term.equilibrium_distance)) {
            if (verbose) {
                value_header();
                fmt::println("    bond[{}] kb rdk={:.4f} nat={:.4f}  r0 rdk={:.4f} nat={:.4f}", term,
                             rdkit_term.force_constant, native_term.force_constant, rdkit_term.equilibrium_distance,
                             native_term.equilibrium_distance);
            }
            ++tally.term_failures;
        }
    }

    // Angles
    for (std::size_t term = 0; term < rdkit_result.mmff_angles.size(); ++term) {
        auto const &rdkit_term = rdkit_result.mmff_angles[term];
        auto const &native_term = native_result.mmff_angles[term];
        ++tally.term_checks;
        if (rdkit_term.atom_i != native_term.atom_i || rdkit_term.atom_j != native_term.atom_j ||
            rdkit_term.atom_k != native_term.atom_k) {
            if (verbose) {
                value_header();
                fmt::println("    angle[{}] ijk rdk={}-{}-{} nat={}-{}-{}", term, rdkit_term.atom_i,
                             rdkit_term.atom_j, rdkit_term.atom_k, native_term.atom_i, native_term.atom_j,
                             native_term.atom_k);
            }
            ++tally.term_failures;
            continue;
        }
        if (!approximately_equal(rdkit_term.force_constant, native_term.force_constant) ||
            !approximately_equal(rdkit_term.equilibrium_angle, native_term.equilibrium_angle)) {
            if (verbose) {
                value_header();
                fmt::println("    angle[{}] ka rdk={:.4f} nat={:.4f} θ₀ rdk={:.4f} nat={:.4f}", term,
                             rdkit_term.force_constant, native_term.force_constant, rdkit_term.equilibrium_angle,
                             native_term.equilibrium_angle);
            }
            ++tally.term_failures;
        }
    }

    // Torsions
    for (std::size_t term = 0; term < rdkit_result.mmff_torsions.size(); ++term) {
        auto const &rdkit_term = rdkit_result.mmff_torsions[term];
        auto const &native_term = native_result.mmff_torsions[term];
        ++tally.term_checks;
        if (rdkit_term.atom_i != native_term.atom_i || rdkit_term.atom_j != native_term.atom_j ||
            rdkit_term.atom_k != native_term.atom_k || rdkit_term.atom_l != native_term.atom_l) {
            if (verbose) {
                value_header();
                fmt::println("    tor[{}] ijkl rdk={}-{}-{}-{} nat={}-{}-{}-{}", term, rdkit_term.atom_i,
                             rdkit_term.atom_j, rdkit_term.atom_k, rdkit_term.atom_l, native_term.atom_i,
                             native_term.atom_j, native_term.atom_k, native_term.atom_l);
            }
            ++tally.term_failures;
            continue;
        }
        if (!approximately_equal(rdkit_term.torsion_v1, native_term.torsion_v1) ||
            !approximately_equal(rdkit_term.torsion_v2, native_term.torsion_v2) ||
            !approximately_equal(rdkit_term.torsion_v3, native_term.torsion_v3)) {
            if (verbose) {
                value_header();
                fmt::println("    tor[{}] ijkl={}-{}-{}-{} V rdk=({:.4f}, {:.4f}, {:.4f}) nat=({:.4f}, {:.4f}, {:.4f})",
                             term, rdkit_term.atom_i, rdkit_term.atom_j, rdkit_term.atom_k, rdkit_term.atom_l,
                             rdkit_term.torsion_v1, rdkit_term.torsion_v2, rdkit_term.torsion_v3,
                             native_term.torsion_v1, native_term.torsion_v2, native_term.torsion_v3);
            }
            ++tally.term_failures;
        }
    }

    // OOPs
    for (std::size_t term = 0; term < rdkit_result.mmff_oops.size(); ++term) {
        auto const &rdkit_term = rdkit_result.mmff_oops[term];
        auto const &native_term = native_result.mmff_oops[term];
        ++tally.term_checks;
        if (rdkit_term.atom_i != native_term.atom_i || rdkit_term.atom_j != native_term.atom_j ||
            rdkit_term.atom_k != native_term.atom_k || rdkit_term.atom_l != native_term.atom_l) {
            if (verbose) {
                value_header();
                fmt::println("    oop[{}] ijkl rdk={}-{}-{}-{} nat={}-{}-{}-{}", term, rdkit_term.atom_i,
                             rdkit_term.atom_j, rdkit_term.atom_k, rdkit_term.atom_l, native_term.atom_i,
                             native_term.atom_j, native_term.atom_k, native_term.atom_l);
            }
            ++tally.term_failures;
            continue;
        }
        if (!approximately_equal(rdkit_term.force_constant, native_term.force_constant)) {
            if (verbose) {
                value_header();
                fmt::println("    oop[{}] ijkl={}-{}-{}-{} koop rdk={:.4f} nat={:.4f}", term, rdkit_term.atom_i,
                             rdkit_term.atom_j, rdkit_term.atom_k, rdkit_term.atom_l, rdkit_term.force_constant,
                             native_term.force_constant);
            }
            ++tally.term_failures;
        }
    }

    // Stretch-bends
    for (std::size_t term = 0; term < rdkit_result.mmff_stretchbends.size(); ++term) {
        auto const &rdkit_term = rdkit_result.mmff_stretchbends[term];
        auto const &native_term = native_result.mmff_stretchbends[term];
        ++tally.term_checks;
        if (rdkit_term.atom_i != native_term.atom_i || rdkit_term.atom_j != native_term.atom_j ||
            rdkit_term.atom_k != native_term.atom_k) {
            if (verbose) {
                value_header();
                fmt::println("    sb[{}] ijk rdk={}-{}-{} nat={}-{}-{}", term, rdkit_term.atom_i, rdkit_term.atom_j,
                             rdkit_term.atom_k, native_term.atom_i, native_term.atom_j, native_term.atom_k);
            }
            ++tally.term_failures;
            continue;
        }
        if (!approximately_equal(rdkit_term.kba_ijk, native_term.kba_ijk) ||
            !approximately_equal(rdkit_term.kba_kji, native_term.kba_kji) ||
            !approximately_equal(rdkit_term.equilibrium_angle, native_term.equilibrium_angle) ||
            !approximately_equal(rdkit_term.equilibrium_distance_ij, native_term.equilibrium_distance_ij) ||
            !approximately_equal(rdkit_term.equilibrium_distance_kj, native_term.equilibrium_distance_kj)) {
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
            ++tally.term_failures;
        }
    }

    // CSR interactions are built only by the native path in this test
    // (the rdkit_direct_extract baseline leaves them empty). Skip that
    // comparison — the native CSR is validated implicitly by the bonded
    // term counts above plus cross-checks in downstream ETKDG kernels.

    // Every value difference above went into the tally rather than into this answer. Reaching here
    // means the two term sets have the same shape, which is all this function decides.
    return term_structure_t::matches_k;
}

/** Build a "ground truth" prepared molecule by calling RDKit's per-term
 *  `getMMFF*Params` directly — bypasses the `mmff_param_cache`, whose
 *  torsion/angle/stbn keys miss ring context. The native CPU port is
 *  validated against this, not against `extract_mmff94_params_via_rdkit`, so that
 *  any latent cache bugs don't contaminate the comparison. */
static void rdkit_direct_extract_(RDKit::ROMol &molecule, sft::molecule_descriptor_t const &descriptor,
                                  sfp::etkdg_prepared_molecule_t &out) {
    // `out` is caller-owned scratch reused across the sweep, so every array this function appends to is
    // emptied first; the two it resizes are covered by the resize.
    out.mmff_bonds.clear();
    out.mmff_angles.clear();
    out.mmff_torsions.clear();
    out.mmff_oops.clear();
    out.mmff_stretchbends.clear();

    // `prepare_one_` already put the molecule through this constructor, so MMFF's own aromatic
    // perception is what @p descriptor was snapshotted from and a second construction changes nothing.
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

/**
 *  @brief Everything both extractors read for one molecule, plus the three term sets they are judged by.
 *
 *  Owned by the sweep and reused across molecules, so the growth phase happens once rather than eleven
 *  vectors per SMILES. Every array is refilled from scratch by the extractor that owns it.
 */
struct molecule_under_test_t {
    std::unique_ptr<RDKit::RWMol> molecule;
    sft::molecule_descriptor_t descriptor {};
    /** Atom types RDKit assigned, which the hybrid extractor is handed. */
    std::vector<std::uint8_t> atom_types;
    /** Partial charges RDKit assigned, which the hybrid extractor is handed. */
    std::vector<float> charges;
    /** The oracle: RDKit's own per-term answer, built without the parameter cache. */
    sfp::etkdg_prepared_molecule_t rdkit_result;
    /** `extract_mmff94_params_native`, handed RDKit's atom types and charges. */
    sfp::etkdg_prepared_molecule_t hybrid_result;
    /** `extract_mmff94_params_fully_native`, deriving atom types and charges itself. */
    sfp::etkdg_prepared_molecule_t fully_native_result;
};

/**
 *  @brief Parse one SMILES and fill everything the comparison consumes, or name why it cannot.
 *
 *  Native and RDKit preprocessing share this one `SmilesToMol` / `addHs` / `build_descriptor` path, so a
 *  difference the comparison reports is a difference between the extractors and never in how the
 *  molecule was read.
 *
 *  @param[in] smiles Molecule to read.
 *  @param[out] subject Overwritten with the parsed molecule, its descriptor, and RDKit's types and charges.
 *  @return Whether the molecule is comparable, and if not, which stage refused it.
 */
static molecule_intake_t prepare_one_(std::string_view smiles, molecule_under_test_t &subject) {
    try {
        std::string const smiles_string {smiles};
        RDKit::SmilesParserParams params;
        params.sanitize = true;
        params.removeHs = false;
        params.strictCXSMILES = false;
        subject.molecule.reset(RDKit::SmilesToMol(smiles_string, params));
        if (!subject.molecule) return molecule_intake_t::unparsable_k;
        RDKit::MolOps::addHs(*subject.molecule);
    }
    catch (...) {
        return molecule_intake_t::unparsable_k; // RDKit throws out of sanitization and out of `addHs`
    }

    std::uint32_t const num_atoms = subject.molecule->getNumAtoms();
    if (num_atoms > max_atoms_compared_k) return molecule_intake_t::above_atom_ceiling_k;

    // Both the oracle and the hybrid path read their atom types and charges here, so a molecule MMFF94
    // cannot type has no reference to compare the native tables against. Constructing the properties
    // rewrites the molecule into MMFF's own aromatic perception, so it precedes the descriptor: the
    // native extractors would otherwise read the parser's rings while the types describe MMFF's.
    RDKit::MMFF::MMFFMolProperties props(*subject.molecule);
    if (!props.isValid()) return molecule_intake_t::untypable_k;
    if (sft::build_descriptor(*subject.molecule, subject.descriptor) != sft::conformer_status_t::success_k)
        return molecule_intake_t::undescribable_k;

    subject.atom_types.assign(num_atoms, std::uint8_t {0});
    subject.charges.assign(num_atoms, 0.0f);
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        subject.atom_types[atom] = static_cast<std::uint8_t>(props.getMMFFAtomType(atom));
        subject.charges[atom] = static_cast<float>(props.getMMFFPartialCharge(atom));
    }
    return molecule_intake_t::comparable_k;
}

/**
 *  @brief Compare both native extractors for one prepared molecule against the RDKit oracle.
 *
 *  Both extractors always run, and every value difference lands in @p tally rather than in the verdict,
 *  so a molecule that differs in one term still contributes its remaining term checks to the totals.
 *
 *  @param[in] smiles Molecule under test, for the narration only.
 *  @param[in] tables MMFF94 parameter tables both native extractors read.
 *  @param[in] narration Whether the differences found are described or only counted.
 *  @param[inout] subject Prepared molecule; its three term sets are overwritten.
 *  @param[inout] tally Term checks and term failures, accumulated across molecules.
 *  @return How this one molecule compared, as one of three verdicts that partition the corpus.
 */
static molecule_verdict_t run_one_(std::string_view smiles, sfparam::mmff94_tables_view_t const &tables,
                                   narration_t narration, molecule_under_test_t &subject, port_tally_t &tally) {
    std::uint64_t const term_failures_before = tally.term_failures;
    std::uint32_t const num_atoms = subject.descriptor.num_atoms;

    subject.rdkit_result.num_atoms = num_atoms;
    rdkit_direct_extract_(*subject.molecule, subject.descriptor, subject.rdkit_result);

    subject.hybrid_result.num_atoms = num_atoms;
    sfmmff::extract_mmff94_params_native(subject.descriptor, subject.atom_types, subject.charges, tables,
                                         subject.hybrid_result);
    term_structure_t structure = compare_mol_(smiles, native_extractor_t::hybrid_k, subject.rdkit_result,
                                              subject.hybrid_result, narration, tally);

    // Zero runtime RDKit, held to the same oracle. A typer refusal is a structural difference rather
    // than a value one, because the terms it would have produced do not exist at all.
    subject.fully_native_result.num_atoms = num_atoms;
    if (!sfmmff::extract_mmff94_params_fully_native(subject.descriptor, tables, subject.fully_native_result)) {
        if (narration == narration_t::narrated_k)
            fmt::println("  [{}] fully-native extraction REFUSED (typer failure)", smiles);
        structure = term_structure_t::differs_k;
    }
    else if (compare_mol_(smiles, native_extractor_t::fully_native_k, subject.rdkit_result, subject.fully_native_result,
                          narration, tally) == term_structure_t::differs_k)
        structure = term_structure_t::differs_k;

    // Structure is decided first because a molecule that is both is still one molecule: the three
    // verdicts have to partition the corpus for the ppm computed over them to be a fraction of anything.
    if (structure == term_structure_t::differs_k) return molecule_verdict_t::structure_differs_k;
    if (tally.term_failures != term_failures_before) return molecule_verdict_t::value_differs_k;
    return molecule_verdict_t::matched_k;
}

/**
 *  @brief Run every SMILES in @p corpus through both extractors, accumulating into @p tally.
 *
 *  The loop is deliberately non-short-circuiting: its counters are the test's output and feed the gate,
 *  so a differing molecule is described and then walked past rather than ending the sweep.
 *
 *  @param[in] corpus SMILES to compare, in order.
 *  @param[in] tables MMFF94 parameter tables both native extractors read.
 *  @param[in] policy Whether every molecule is narrated, or only the first molecules that differ.
 *  @param[inout] tally Molecule verdicts and term counts, accumulated across sweeps.
 */
static void sweep_(custd::span<std::string const> corpus, sfparam::mmff94_tables_view_t const &tables,
                   narration_policy_t policy, port_tally_t &tally) {
    molecule_under_test_t subject;
    int narrated_failures = 0;
    for (auto const &smiles : corpus) {
        // A molecule RDKit cannot parse, size, describe or type measures nothing about the port, so it
        // stays out of the denominator instead of counting as a match.
        if (prepare_one_(smiles, subject) != molecule_intake_t::comparable_k) {
            ++tally.molecules_skipped;
            continue;
        }
        bool const within_narration_budget = narrated_failures < max_reported_failing_molecules_k;
        narration_t const narration = policy == narration_policy_t::every_molecule_k || within_narration_budget
                                          ? narration_t::narrated_k
                                          : narration_t::silent_k;
        molecule_verdict_t const verdict = run_one_(smiles, tables, narration, subject, tally);
        ++tally.molecules_with(verdict);

        if (verdict == molecule_verdict_t::matched_k) continue;       // nothing to attribute
        if (policy != narration_policy_t::first_failures_k) continue; // the probe already names itself
        if (!within_narration_budget) continue;                       // budget spent, and still counted
        fmt::println("  (above is from SMILES: {})", smiles);
        ++narrated_failures;
    }
}

/**
 *  @brief Print what the sweeps counted and answer `main` with the exit code the gate decided.
 *
 *  @param[in] tally Aggregated counts from every sweep.
 *  @return `0` when the port matches within budget, `1` otherwise.
 */
static int report_(port_tally_t const &tally) {
    std::uint64_t const compared = tally.total_molecules();
    std::uint64_t const value_differs = tally.molecules_with(molecule_verdict_t::value_differs_k);
    std::uint64_t const structure_differs = tally.molecules_with(molecule_verdict_t::structure_differs_k);
    fmt::println("molecules compared={} matched={} value_differs={} structure_differs={} skipped={}", compared,
                 tally.molecules_with(molecule_verdict_t::matched_k), value_differs, structure_differs,
                 tally.molecules_skipped);
    fmt::println("terms compared={} failures={}", tally.term_checks, tally.term_failures);

    // A validator that compared nothing passes every run, so an empty sweep is a failure and not a pass.
    if (compared == 0) return report_fail(test_name_k, "no molecule reached the comparison");

    double const molecule_ppm = 1.0e6 * static_cast<double>(value_differs + structure_differs) /
                                static_cast<double>(compared);
    double const value_ppm = 1.0e6 * static_cast<double>(value_differs) / static_cast<double>(compared);
    double const term_ppm = tally.term_checks > 0 ? 1.0e6 * static_cast<double>(tally.term_failures) /
                                                        static_cast<double>(tally.term_checks)
                                                  : 0.0;
    fmt::println("ppm molecules_differing={:.3f} molecules_value_differing={:.3f} terms_differing={:.3f} budget={:.3f}",
                 molecule_ppm, value_ppm, term_ppm, max_molecule_value_mismatch_ppm_k);

    // Structural mismatches are never tolerated; value differences get the ppm budget the file header
    // documents. Exotic structures — pentacoordinate P, porphyrins — can exceed it and should be triaged
    // on the full 10M USearch set rather than absorbed by loosening this.
    if (structure_differs > 0)
        return report_fail(test_name_k, "{} of {} molecules differ structurally from the RDKit baseline",
                           structure_differs, compared);
    if (value_ppm > max_molecule_value_mismatch_ppm_k)
        return report_fail(test_name_k, "{} of {} molecules differ in value, {:.3f} ppm > {:.3f} ppm", value_differs,
                           compared, value_ppm, max_molecule_value_mismatch_ppm_k);
    return report_pass(test_name_k);
}

} // namespace smallfield::tests

int main(int argc, char **argv) {
    fmt::println("{}: native MMFF term extractor vs the RDKit-backed baseline", test_name_k);

    auto const corpus = sftest::select_corpus(argc, argv, 5000);
    if (corpus.knob == sftest::corpus_knob_t::refused_k) {
        std::fflush(stdout);
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", test_name_k);
        return 2;
    }

    sfparam::mmff94_tables_host_t tables_host;
    sfparam::extract_mmff94_tables_from_rdkit(tables_host);
    sfparam::mmff94_tables_view_t const tables = tables_host.view();
    fmt::println("tables vdw={} bond={} angle={} torsion={} oop={} stbn={} dfsb={}", tables_host.vdw.size(),
                 tables_host.bond.size(), tables_host.angle.size(), tables_host.torsion.size(), tables_host.oop.size(),
                 tables_host.stbn.size(), tables_host.dfsb.size());

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

    // A corpus arrives by flag or by `SMALLFIELD_TEST_CORPUS`, and a path that does not open is a
    // failure rather than an empty run that passes because it compared nothing.
    std::vector<std::string> extra;
    if (corpus.knob == sftest::corpus_knob_t::named_k) {
        auto const outcome = sftest::load_smiles_into(corpus.path, corpus.max_molecules, extra);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), corpus.path);
    }
    fmt::println("smiles curated={} external={}", probe.size(), extra.size());

    sftest::port_tally_t tally;
    sftest::sweep_(probe, tables, sftest::narration_policy_t::every_molecule_k, tally);
    sftest::sweep_(extra, tables, sftest::narration_policy_t::first_failures_k, tally);
    return sftest::report_(tally);
}
