/**
 *  @file src/smallfield/mmff94/mmff94_term_equations.hpp
 *  @brief MMFF94 term classification and empirical rules, shared by the host and device extractors.
 *
 *  Reads `molecule_descriptor_t` plus caller-supplied per-atom MMFF types and answers, term by term,
 *  the questions both extractors ask:
 *
 *   1. Topology queries — bond lookup through the descriptor's neighbour CSR, plus the 3-, 4-, and
 *      5-membered ring tests that angle and torsion typing key on.
 *   2. Term type classification — ports of `MMFFMolProperties::getMMFFBondType`, `getMMFFAngleType`
 *      and `getMMFFTorsionType`, and of the free `getMMFFStretchBendType`.
 *   3. Empirical fallback rules for bond stretch, angle bend, and torsion, reimplemented from MMFF.V
 *      equations 18 and 19 and torsion rules @b a through @b h, alongside `native_is_zero`, the
 *      `isDoubleZero` threshold that decides when a table entry is empty enough to fall back.
 *
 *  Nothing here touches `etkdg_prepared_molecule_t` or the preprocessing batch header, which is what
 *  lets a CUDA translation unit include it. Both extractors do:
 *  `mmff94/mmff94_extract.hpp` on the host, `mmff94/mmff94_extract_device.cuh` on the device.
 *  Sharing this layer is what keeps the two in step — `test/mmff94/test_mmff_gpu_extract.cpp`
 *  requires byte identity across them for every bonded term.
 *
 *  All parameter resolution goes through `parameters/mmff94_tables.hpp` — the extracted RDKit tables
 *  plus the `lookup_*` cascades that mirror RDKit's `operator()` behaviour, feeding off the `covrad`,
 *  `bndk`, `hl`, and `prop` tables.
 *
 *  Atom types and partial charges are caller-provided. The native atom typer and BCI charge
 *  perception live in the GPU kernel; where it leaves a molecule untyped, the host supplies RDKit's
 *  answer through `preprocess/etkdg_prepare_batch.hpp::extract_mmff94_typing_via_rdkit_`.
 */

#pragma once

#include <cmath>   // `std::fabs`, `std::pow`
#include <cstdint> // `std::uint8_t`, `std::int32_t`

#include <array>   // `std::array`
#include <span>    // `std::span`
#include <utility> // `std::pair`

#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/mmff94_bounds.hpp"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/mmff94/mmff94_typer.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::mmff94 {
using smallfield::parameters::lookup_bndk;
using smallfield::parameters::lookup_covrad;
using smallfield::parameters::lookup_hl;
using smallfield::parameters::lookup_prop;
using smallfield::parameters::mmff94_tables_view_t;
using smallfield::types::bond_flag_aromatic_k;
using smallfield::types::molecule_descriptor_t;

#pragma region Periodic Table Rows

/** Matches `RDKit::MMFF::getPeriodicTableRow`: row 0 holds H and He; row 1 Li through Ne; row 2 Na
 *  through Ar; row 3 K through Kr; row 4 Rb through Xe. */
constexpr std::uint8_t periodic_table_row(std::uint8_t atomic_num) noexcept {
    if (atomic_num >= 3 && atomic_num <= 10) return 1;
    if (atomic_num >= 11 && atomic_num <= 18) return 2;
    if (atomic_num >= 19 && atomic_num <= 36) return 3;
    if (atomic_num >= 37 && atomic_num <= 54) return 4;
    return 0;
}

/** Matches `RDKit::MMFF::getPeriodicTableRowHL` — 1-based, transition metals ×10. Used only by the
 *  Herschbach-Laurie fallback. */
constexpr std::uint8_t periodic_table_row_hl(std::uint8_t atomic_num) noexcept {
    std::uint8_t row = 0;
    if (atomic_num == 2) row = 1;
    else if (atomic_num >= 3 && atomic_num <= 10) row = 2;
    else if (atomic_num >= 11 && atomic_num <= 18) row = 3;
    else if (atomic_num >= 19 && atomic_num <= 36) row = 4;
    else if (atomic_num >= 37 && atomic_num <= 54) row = 5;
    if ((atomic_num >= 21 && atomic_num <= 30) || (atomic_num >= 39 && atomic_num <= 48)) {
        row = static_cast<std::uint8_t>(row * 10);
    }
    return row;
}

#pragma endregion Periodic Table Rows

#pragma region Neighbor CSR Helpers

/** Find the bond index between `atom_i` and `atom_j` via the descriptor's neighbor CSR. Returns -1 on miss.
 *  Matches the `find_bond_index` lambda in `preprocess/etkdg_prepare_batch.hpp`. */
constexpr std::int32_t find_bond_index(molecule_descriptor_t const &descriptor, std::uint32_t atom_i,
                                       std::uint32_t atom_j) noexcept {
    std::uint16_t const start = descriptor.neighbor_offset[atom_i];
    std::uint16_t const end = descriptor.neighbor_offset[atom_i + 1];
    for (std::uint16_t neighbor_slot = start; neighbor_slot < end; ++neighbor_slot)
        if (descriptor.neighbor_atom[neighbor_slot] == atom_j) return descriptor.neighbor_bond[neighbor_slot];
    return -1;
}

/** Are `atom_i` and `atom_k` both adjacent to `atom_j` and to each other? That is, does the i-j-k
 *  angle close back into a 3-ring. Purely uses the descriptor's CSR — no ring info required. */
constexpr bool angle_in_ring_of_size_3(molecule_descriptor_t const &descriptor, std::uint32_t atom_i,
                                       std::uint32_t atom_k) noexcept {
    return find_bond_index(descriptor, atom_i, atom_k) >= 0;
}

/** Do `atom_i` and `atom_k` share a common neighbor that isn't `atom_j`? Matches
 *  `RDKit::MMFF::isAngleInRingOfSize3or4`'s size-4 branch exactly. */
constexpr bool angle_has_common_neighbor(molecule_descriptor_t const &descriptor, std::uint32_t atom_i,
                                         std::uint32_t atom_j, std::uint32_t atom_k) noexcept {
    std::uint16_t const atom_i_neighbor_start = descriptor.neighbor_offset[atom_i];
    std::uint16_t const atom_i_neighbor_end = descriptor.neighbor_offset[atom_i + 1];
    for (std::uint16_t neighbor_offset_i = atom_i_neighbor_start; neighbor_offset_i < atom_i_neighbor_end;
         ++neighbor_offset_i) {
        std::uint32_t const candidate_atom = descriptor.neighbor_atom[neighbor_offset_i];
        if (candidate_atom == atom_j) continue;
        if (find_bond_index(descriptor, candidate_atom, atom_k) >= 0) return true;
    }
    return false;
}

/** Ring size containing the i-j-k angle, or 0 if not 3/4. Matches `isAngleInRingOfSize3or4` exactly. */
constexpr unsigned native_is_angle_in_ring_3_or_4(molecule_descriptor_t const &descriptor, std::uint32_t atom_i,
                                                  std::uint32_t atom_j, std::uint32_t atom_k) noexcept {
    // Caller should hold that i-j and j-k are real bonds; only the ring closure checks trigger here.
    if (find_bond_index(descriptor, atom_i, atom_j) < 0 || find_bond_index(descriptor, atom_j, atom_k) < 0) return 0;
    if (angle_in_ring_of_size_3(descriptor, atom_i, atom_k)) return 3;
    if (angle_has_common_neighbor(descriptor, atom_i, atom_j, atom_k)) return 4;
    return 0;
}

/** Ring size containing the i-j-k-l torsion, or 0 if not 4/5. Matches `isTorsionInRingOfSize4or5` exactly. */
constexpr unsigned native_is_torsion_in_ring_4_or_5(molecule_descriptor_t const &descriptor, std::uint32_t atom_i,
                                                    std::uint32_t atom_j, std::uint32_t atom_k,
                                                    std::uint32_t atom_l) noexcept {
    if (find_bond_index(descriptor, atom_i, atom_j) < 0 || find_bond_index(descriptor, atom_j, atom_k) < 0 ||
        find_bond_index(descriptor, atom_k, atom_l) < 0)
        return 0;
    if (find_bond_index(descriptor, atom_l, atom_i) >= 0) return 4;
    // Size-5: do atoms i and l share a common neighbor that isn't j/k?
    std::uint16_t const atom_i_neighbor_start = descriptor.neighbor_offset[atom_i];
    std::uint16_t const atom_i_neighbor_end = descriptor.neighbor_offset[atom_i + 1];
    for (std::uint16_t neighbor_offset_i = atom_i_neighbor_start; neighbor_offset_i < atom_i_neighbor_end;
         ++neighbor_offset_i) {
        std::uint32_t const candidate_from_i = descriptor.neighbor_atom[neighbor_offset_i];
        if (candidate_from_i == atom_j) continue;
        // Is `candidate_from_i` also a neighbor of l, and not k?
        std::uint16_t const atom_l_neighbor_start = descriptor.neighbor_offset[atom_l];
        std::uint16_t const atom_l_neighbor_end = descriptor.neighbor_offset[atom_l + 1];
        for (std::uint16_t neighbor_offset_l = atom_l_neighbor_start; neighbor_offset_l < atom_l_neighbor_end;
             ++neighbor_offset_l) {
            std::uint32_t const candidate_from_l = descriptor.neighbor_atom[neighbor_offset_l];
            if (candidate_from_l == atom_k) continue;
            if (candidate_from_i == candidate_from_l) return 5;
        }
    }
    return 0;
}

#pragma endregion Neighbor CSR Helpers

#pragma region Native Type Classification

/** Native port of `MMFFMolProperties::getMMFFBondType`. Returns 1 iff the bond is SINGLE and its two
 *  endpoints either both carry the MMFFProp single-bond-multiple-bond designator or are both
 *  aromatic atoms. Otherwise returns 0. */
constexpr std::uint8_t native_get_mmff_bond_type(molecule_descriptor_t const &descriptor,
                                                 mmff94_tables_view_t const &tables,
                                                 std::span<std::uint8_t const> atom_types,
                                                 std::uint32_t bond_index) noexcept {
    std::uint8_t const bond_order = descriptor.bond_order[bond_index];
    bool const aromatic_bond = (descriptor.bond_flags[bond_index] & bond_flag_aromatic_k) != 0;
    // RDKit's rule triggers only when the bond is SINGLE. Aromatic bonds in the descriptor report order=1
    // with the aromatic flag set, so the test keys on `bond_order == 1 && !aromatic_bond` to exclude aromatic.
    if (!(bond_order == 1 && !aromatic_bond)) return 0;
    std::uint8_t const atom_i = descriptor.bond_atom_i[bond_index];
    std::uint8_t const atom_j = descriptor.bond_atom_j[bond_index];
    auto const *prop_i = lookup_prop(tables, atom_types[atom_i]);
    auto const *prop_j = lookup_prop(tables, atom_types[atom_j]);
    if (!prop_i || !prop_j) return 0;
    if ((prop_i->single_bond_multiple_bond && prop_j->single_bond_multiple_bond) ||
        (prop_i->aromatic && prop_j->aromatic))
        return 1;
    return 0;
}

/** Native port of `MMFFMolProperties::getMMFFAngleType`. */
constexpr std::uint8_t native_get_mmff_angle_type(molecule_descriptor_t const &descriptor,
                                                  std::span<std::uint8_t const> bond_types, std::uint32_t atom_i,
                                                  std::uint32_t atom_j, std::uint32_t atom_k) noexcept {
    std::int32_t const bond_index_ij = find_bond_index(descriptor, atom_i, atom_j);
    std::int32_t const bond_index_jk = find_bond_index(descriptor, atom_j, atom_k);
    if (bond_index_ij < 0 || bond_index_jk < 0) return 0;
    unsigned const bond_type_sum = bond_types[bond_index_ij] + bond_types[bond_index_jk];
    unsigned angle_type = bond_type_sum;
    unsigned const size = native_is_angle_in_ring_3_or_4(descriptor, atom_i, atom_j, atom_k);
    if (size) {
        angle_type = size;
        if (bond_type_sum) angle_type += (bond_type_sum + size - 2);
    }
    return static_cast<std::uint8_t>(angle_type);
}

/** Native port of `MMFFMolProperties::getMMFFTorsionType`, returning (primary, secondary) matching
 *  RDKit's pair. */
constexpr std::pair<std::uint8_t, std::uint8_t> native_get_mmff_torsion_type(molecule_descriptor_t const &descriptor,
                                                                             std::span<std::uint8_t const> atom_types,
                                                                             std::span<std::uint8_t const> bond_types,
                                                                             std::uint32_t atom_i, std::uint32_t atom_j,
                                                                             std::uint32_t atom_k,
                                                                             std::uint32_t atom_l) noexcept {
    std::int32_t const bond_index_jk = find_bond_index(descriptor, atom_j, atom_k);
    std::int32_t const bond_index_ij = find_bond_index(descriptor, atom_i, atom_j);
    std::int32_t const bond_index_kl = find_bond_index(descriptor, atom_k, atom_l);
    if (bond_index_jk < 0 || bond_index_ij < 0 || bond_index_kl < 0) return {0, 0};

    std::uint8_t const bond_type_ij = bond_types[bond_index_ij];
    std::uint8_t const bond_type_jk = bond_types[bond_index_jk];
    std::uint8_t const bond_type_kl = bond_types[bond_index_kl];
    bool const jk_single = (descriptor.bond_order[bond_index_jk] == 1) &&
                           !(descriptor.bond_flags[bond_index_jk] & bond_flag_aromatic_k);

    unsigned torsion_type = bond_type_jk;
    unsigned secondary = 0;
    if (bond_type_jk == 0 && jk_single && (bond_type_ij == 1 || bond_type_kl == 1)) torsion_type = 2;

    unsigned const size = native_is_torsion_in_ring_4_or_5(descriptor, atom_i, atom_j, atom_k, atom_l);
    if (size == 4) {
        // Skip the 4-ring reassignment for torsions where the "4-ring" is actually a pair of fused
        // 3-rings, with i-k or j-l already bonded.
        bool const fused_3_3 = (find_bond_index(descriptor, atom_i, atom_k) >= 0) ||
                               (find_bond_index(descriptor, atom_j, atom_l) >= 0);
        if (!fused_3_3) {
            secondary = torsion_type;
            torsion_type = 4;
        }
    }
    else if (size == 5) {
        // Only if any of the four atoms has MMFF type 1.
        if (atom_types[atom_i] == 1 || atom_types[atom_j] == 1 || atom_types[atom_k] == 1 || atom_types[atom_l] == 1) {
            secondary = torsion_type;
            torsion_type = 5;
        }
    }
    return {static_cast<std::uint8_t>(torsion_type), static_cast<std::uint8_t>(secondary)};
}

/** Native port of the free function `getMMFFStretchBendType`. */
constexpr std::uint8_t native_get_mmff_stretch_bend_type(unsigned angle_type, unsigned bond_type_1,
                                                         unsigned bond_type_2) noexcept {
    switch (angle_type) {
    case 1: return (bond_type_1 || (bond_type_1 == bond_type_2)) ? 1 : 2;
    case 2: return 3;
    case 4: return 4;
    case 3: return 5;
    case 5: return (bond_type_1 || (bond_type_1 == bond_type_2)) ? 6 : 7;
    case 6: return 8;
    case 7: return (bond_type_1 || (bond_type_1 == bond_type_2)) ? 9 : 10;
    case 8: return 11;
    default: return 0;
    }
}

#pragma endregion Native Type Classification

#pragma region Empirical Fallback Rules

/** Empirical bond-stretch rule from MMFF.V equations 18-19. Matches `getMMFFBondStretchEmpiricalRuleParams` and
 *  returns the force constant `kb` together with the equilibrium distance `r0` by value. Internally uses `double`
 *  to match RDKit's precision; inputs and outputs are float for parity with the extracted-table pipeline. */
SF_HOST_DEVICE_ inline std::pair<float, float> native_bond_empirical(mmff94_tables_view_t const &tables,
                                                                     std::uint8_t atomic_i,
                                                                     std::uint8_t atomic_j) noexcept {
    auto const *cov_i = lookup_covrad(tables, atomic_i);
    auto const *cov_j = lookup_covrad(tables, atomic_j);
    if (!cov_i || !cov_j) return {0.0f, 0.0f};
    double const electronegativity_coefficient = ((atomic_i == 1 || atomic_j == 1) ? 0.050 : 0.085);
    double const electronegativity_exponent = 1.4;
    double const delta = 0.0;
    double const chi_abs = std::fabs(static_cast<double>(cov_i->pauling_electronegativity) -
                                     cov_j->pauling_electronegativity);
    double const equilibrium_distance = static_cast<double>(cov_i->covalent_radius) + cov_j->covalent_radius -
                                        electronegativity_coefficient * std::pow(chi_abs, electronegativity_exponent) -
                                        delta;

    auto const *bndk = lookup_bndk(tables, atomic_i, atomic_j);
    double force_constant = 0.0;
    if (bndk) {
        double const coeff = static_cast<double>(bndk->equilibrium_distance) / equilibrium_distance;
        double const coeff2 = coeff * coeff;
        double const coeff6 = coeff2 * coeff2 * coeff2;
        force_constant = static_cast<double>(bndk->force_constant) * coeff6;
    }
    else {
        auto const *herschbach_laurie_entry = lookup_hl(tables, periodic_table_row_hl(atomic_i),
                                                        periodic_table_row_hl(atomic_j));
        if (herschbach_laurie_entry)
            force_constant = std::pow(
                10.0, -(equilibrium_distance - herschbach_laurie_entry->a_ij) / herschbach_laurie_entry->d_ij);
    }
    return {static_cast<float>(force_constant), static_cast<float>(equilibrium_distance)};
}

/** Empirical angle-bend rule from MMFF.V, Table VI and its equations. Takes already-computed bond params for the
 *  two flanking bonds and returns the force constant `ka` with θ₀ in degrees. Matches
 *  `getMMFFAngleBendEmpiricalRuleParams`. A `theta0_override_deg` of zero or more replaces the default θ₀ derived
 *  from `coordination_number`, `multiple_bond_designator` and `valence`; that matches RDKit's behaviour when a
 *  table entry exists with ka=0, where RDKit keeps the table's θ₀ and recomputes ka empirically. */
SF_HOST_DEVICE_ inline std::pair<float, float> native_angle_empirical(molecule_descriptor_t const &descriptor,
                                                                      mmff94_tables_view_t const &tables,
                                                                      std::span<std::uint8_t const> atom_types,
                                                                      std::uint32_t atom_i, std::uint32_t atom_j,
                                                                      std::uint32_t atom_k, float r0_ij, float r0_kj,
                                                                      float theta0_override_deg = -1.0f) noexcept {
    std::uint8_t const atomic_number_i = descriptor.element[atom_i], atomic_number_j = descriptor.element[atom_j],
                       atomic_number_k = descriptor.element[atom_k];

    // θ₀ — depends on the central atom's `coordination_number`, `multiple_bond_designator`,
    // `valence` and `linear`.
    auto const *prop_j = lookup_prop(tables, atom_types[atom_j]);
    float theta0_deg = 120.0f;
    if (prop_j) {
        switch (prop_j->coordination_number) {
        case 4: theta0_deg = 109.45f; break;
        case 2:
            if (atomic_number_j == 8) theta0_deg = 105.0f;
            else if (prop_j->linear == 1) theta0_deg = 180.0f;
            break;
        case 3:
            if (prop_j->valence == 3 && prop_j->multiple_bond_designator == 0) {
                theta0_deg = (atomic_number_j == 7) ? 107.0f : 92.0f;
            }
            break;
        }
    }
    unsigned const ring_size = native_is_angle_in_ring_3_or_4(descriptor, atom_i, atom_j, atom_k);
    if (ring_size == 3) theta0_deg = 60.0f;
    else if (ring_size == 4) theta0_deg = 90.0f;
    if (theta0_override_deg >= 0.0f) theta0_deg = theta0_override_deg;

    // Force constant `ka` — atomic_constant_i * period_constant_j * atomic_constant_k * r0_ij * r0_kj ...
    auto lookup_angle_bend_constants = [](std::uint8_t atomic_num) -> std::pair<double, double> {
        // Table VI — MMFF.V, page 628; RDKit `AtomTyper.cpp` lines 2787-2854.
        // Returns (atomic_constant "Z", period_constant "C").
        switch (atomic_num) {
        case 1: return {1.395, 0.0};
        case 6: return {2.494, 1.016};
        case 7: return {2.711, 1.113};
        case 8: return {3.045, 1.337};
        case 9: return {2.847, 0.0};
        case 14: return {2.350, 0.811};
        case 15: return {2.350, 1.068};
        case 16: return {2.980, 1.249};
        case 17: return {2.909, 1.078};
        case 35: return {3.017, 0.0};
        case 53: return {3.086, 0.0};
        default: return {0.0, 0.0};
        }
    };
    double const atomic_constant_i = lookup_angle_bend_constants(atomic_number_i).first;
    double const period_constant_j = lookup_angle_bend_constants(atomic_number_j).second;
    double const atomic_constant_k = lookup_angle_bend_constants(atomic_number_k).first;

    double beta = 1.75;
    if (ring_size == 4) beta *= 0.85;
    else if (ring_size == 3) beta *= 0.05;
    constexpr double mmff_degrees_to_radians_k = M_PI / 180.0;
    double const theta0_rad = static_cast<double>(theta0_deg) * mmff_degrees_to_radians_k;
    // Widen before the arithmetic, not after: the difference and the sum below are formed in
    // double so the squared asymmetry keeps its precision, which is what the float parameters
    // were promoted for in the first place.
    double const bond_length_ij = r0_ij;
    double const bond_length_kj = r0_kj;
    double const bond_length_sum = bond_length_ij + bond_length_kj;
    double const bond_length_asymmetry = (bond_length_ij - bond_length_kj) * (bond_length_ij - bond_length_kj) /
                                         (bond_length_sum * bond_length_sum);
    double const force_constant = beta * atomic_constant_i * period_constant_j * atomic_constant_k /
                                  (bond_length_sum * theta0_rad * theta0_rad * std::exp(2.0 * bond_length_asymmetry));
    return {static_cast<float>(force_constant), theta0_deg};
}

/** Empirical torsion rule from MMFF.V, rules a-h. Matches `getMMFFTorsionEmpiricalRuleParams`.
 *  Returns (V1, V2, V3). The central bond's type is passed in so it is not looked up twice. */
SF_HOST_DEVICE_ inline std::array<float, 3> native_torsion_empirical(molecule_descriptor_t const &descriptor,
                                                                     mmff94_tables_view_t const &tables,
                                                                     std::span<std::uint8_t const> atom_types,
                                                                     std::uint32_t atom_j, std::uint32_t atom_k,
                                                                     std::int32_t bond_jk_index) noexcept {
    auto const *properties_j = lookup_prop(tables, atom_types[atom_j]);
    auto const *properties_k = lookup_prop(tables, atom_types[atom_k]);
    std::array<float, 3> out {0.0f, 0.0f, 0.0f};
    if (!properties_j || !properties_k) return out;
    // Table VII — MMFF.V, page 631. Returns the twofold-pi constant "U", threefold constant "V",
    // and O/S lone-pair twofold constant "W".
    auto lookup_torsion_constants = [](std::uint8_t atomic_number, float &twofold_pi_constant,
                                       float &threefold_constant, float &lone_pair_constant) {
        twofold_pi_constant = threefold_constant = lone_pair_constant = 0.0f;
        switch (atomic_number) {
        case 6:
            twofold_pi_constant = 2.00f;
            threefold_constant = 2.12f;
            break;
        case 7:
            twofold_pi_constant = 2.00f;
            threefold_constant = 1.50f;
            break;
        case 8:
            twofold_pi_constant = 2.00f;
            threefold_constant = 0.20f;
            lone_pair_constant = 2.0f;
            break;
        case 14:
            twofold_pi_constant = 1.25f;
            threefold_constant = 1.22f;
            break;
        case 15:
            twofold_pi_constant = 1.25f;
            threefold_constant = 2.40f;
            break;
        case 16:
            twofold_pi_constant = 1.25f;
            threefold_constant = 0.49f;
            lone_pair_constant = 8.0f;
            break;
        }
    };
    float twofold_pi_constant_j, threefold_constant_j, lone_pair_constant_j;
    float twofold_pi_constant_k, threefold_constant_k, lone_pair_constant_k;
    lookup_torsion_constants(descriptor.element[atom_j], twofold_pi_constant_j, threefold_constant_j,
                             lone_pair_constant_j);
    lookup_torsion_constants(descriptor.element[atom_k], twofold_pi_constant_k, threefold_constant_k,
                             lone_pair_constant_k);
    float const torsions_about_jk = static_cast<float>((properties_j->coordination_number - 1) *
                                                       (properties_k->coordination_number - 1));
    bool const jk_double = descriptor.bond_order[bond_jk_index] == 2;
    bool const jk_aromatic = (descriptor.bond_flags[bond_jk_index] & bond_flag_aromatic_k) != 0;
    bool const jk_single = descriptor.bond_order[bond_jk_index] == 1 && !jk_aromatic;

    // rule (a): linear heavy atom on either side — all zero.
    if (properties_j->linear || properties_k->linear) return out;

    // rule (b): aromatic-aromatic in aromatic ring.
    // RDKit uses the `MMFFArom` set of atom types, never nullptr for non-zero types. Replicated
    // here with a small constant list — see `DefaultParameters::getMMFFArom`.
    // `MMFFArom` types: {37, 38, 39, 44, 58, 59, 63, 64, 65, 66, 69, 76, 78, 79, 80, 81, 82}.
    auto is_mmff_arom = [](std::uint8_t atom_type) {
        switch (atom_type) {
        case 37:
        case 38:
        case 39:
        case 44:
        case 58:
        case 59:
        case 63:
        case 64:
        case 65:
        case 66:
        case 69:
        case 76:
        case 78:
        case 79:
        case 80:
        case 81:
        case 82: return true;
        default: return false;
        }
    };
    if (is_mmff_arom(atom_types[atom_j]) && is_mmff_arom(atom_types[atom_k]) && jk_aromatic) {
        float const beta = (((properties_j->valence == 3 && properties_k->valence == 4) ||
                             (properties_j->valence == 4 && properties_k->valence == 3))
                                ? 3.0f
                                : 6.0f);
        float const pi_jk = ((properties_j->pi_lone_pair == 0 && properties_k->pi_lone_pair == 0) ? 0.5f : 0.3f);
        out[1] = beta * pi_jk * std::sqrt(twofold_pi_constant_j * twofold_pi_constant_k);
        return out;
    }

    // rule (c): central bond is double.
    if (jk_double) {
        float const beta = 6.0f;
        float const pi_jk =
            ((properties_j->multiple_bond_designator == 2 && properties_k->multiple_bond_designator == 2) ? 1.0f
                                                                                                          : 0.4f);
        out[1] = beta * pi_jk * std::sqrt(twofold_pi_constant_j * twofold_pi_constant_k);
        return out;
    }

    // rule (d): sp3-sp3.
    if (properties_j->coordination_number == 4 && properties_k->coordination_number == 4) {
        out[2] = std::sqrt(threefold_constant_j * threefold_constant_k) / torsions_about_jk;
        return out;
    }

    // rules (e) and (f): sp3-X (or X-sp3).
    if ((properties_j->coordination_number == 4 && properties_k->coordination_number != 4) ||
        (properties_k->coordination_number == 4 && properties_j->coordination_number != 4)) {
        auto const *neighbor_properties = (properties_j->coordination_number == 4) ? properties_k : properties_j;
        bool zero = false;
        if ((neighbor_properties->coordination_number == 3 &&
             ((neighbor_properties->valence == 4 || neighbor_properties->valence == 34) ||
              neighbor_properties->multiple_bond_designator)) ||
            (neighbor_properties->coordination_number == 2 &&
             (neighbor_properties->valence == 3 || neighbor_properties->multiple_bond_designator)))
            zero = true;
        if (!zero) out[2] = std::sqrt(threefold_constant_j * threefold_constant_k) / torsions_about_jk;
        return out;
    }

    // rule (g): multiple_bond_designator / pi_lone_pair combinations.
    bool const g_trigger = (jk_single && properties_j->multiple_bond_designator &&
                            properties_k->multiple_bond_designator) ||
                           (properties_j->multiple_bond_designator && properties_k->pi_lone_pair) ||
                           (properties_j->pi_lone_pair && properties_k->multiple_bond_designator);
    if (g_trigger) {
        if (properties_j->pi_lone_pair && properties_k->pi_lone_pair) return out;
        std::uint8_t const atomic_number_j = descriptor.element[atom_j], atomic_number_k = descriptor.element[atom_k];
        std::uint8_t const row_j = periodic_table_row(atomic_number_j);
        std::uint8_t const row_k = periodic_table_row(atomic_number_k);
        float pi_jk = 0.0f;
        float const beta = 6.0f;
        if (properties_j->pi_lone_pair && properties_k->multiple_bond_designator) {
            if (properties_j->multiple_bond_designator == 1) pi_jk = 0.5f;
            else if (row_j == 2 && row_k == 2) pi_jk = 0.3f;
            else pi_jk = 0.15f;
        }
        else if (properties_k->pi_lone_pair && properties_j->multiple_bond_designator) {
            if (properties_k->multiple_bond_designator == 1) pi_jk = 0.5f;
            else if (row_j == 2 && row_k == 2) pi_jk = 0.3f;
            else pi_jk = 0.15f;
        }
        else if ((properties_j->multiple_bond_designator == 1 || properties_k->multiple_bond_designator == 1) &&
                 (atomic_number_j != 6 || atomic_number_k != 6))
            pi_jk = 0.4f;
        else pi_jk = 0.15f;
        out[1] = beta * pi_jk * std::sqrt(twofold_pi_constant_j * twofold_pi_constant_k);
        return out;
    }

    // rule (h).
    std::uint8_t const atomic_number_j = descriptor.element[atom_j], atomic_number_k = descriptor.element[atom_k];
    if ((atomic_number_j == 8 || atomic_number_j == 16) && (atomic_number_k == 8 || atomic_number_k == 16)) {
        out[1] = -std::sqrt(lone_pair_constant_j * lone_pair_constant_k);
    }
    else { out[2] = std::sqrt(threefold_constant_j * threefold_constant_k) / torsions_about_jk; }
    return out;
}

#pragma endregion Empirical Fallback Rules

#pragma region Numeric Predicates

/** `isDoubleZero` predicate matching RDKit's 1e-10 threshold. Used by the angle-bend and torsion
 *  fallback logic. */
SF_HOST_DEVICE_ inline bool native_is_zero(float value) noexcept { return std::fabs(value) < 1.0e-10f; }

#pragma endregion Numeric Predicates

} // namespace smallfield::mmff94
