/**
 *  @file mmff94_cpu.hpp
 *  @brief Self-contained CPU port of MMFF94 term extraction.
 *
 *  Reads `molecule_descriptor_t` plus per-atom MMFF types and partial charges, and produces a fully
 *  populated `etkdg_prepared_molecule_t` without any runtime RDKit calls. Serves three roles:
 *
 *   1. GPU debugging reference — every GPU kernel must match the output of this module byte-for-byte
 *      on the same inputs.
 *   2. Production fallback — molecules the GPU atom-typer cannot classify are re-routed through this
 *      path on the host.
 *   3. Throughput ceiling — pure C++ lookups against our flat POD tables bound the best the GPU port
 *      can do before taking over.
 *
 *  All parameter resolution goes through `parameters/mmff94_tables.hpp` — the extracted RDKit tables
 *  plus the `lookup_*` cascades that mirror RDKit's `operator()` behaviour. The empirical rules for
 *  bond stretch, angle bend, and torsion are reimplemented in-file from MMFF.V equations 18, 19, (g),
 *  and (h), feeding off the `covrad`, `bndk`, `hl`, and `prop` tables.
 *
 *  Atom types and partial charges are caller-provided. The native atom typer and BCI charge
 *  perception live in the GPU kernel; the host-side fallback pulls types and charges from
 *  `MMFFMolProperties` in the caller — see
 *  `preprocess/etkdg_prepare_batch.hpp::extract_mmff94_params_via_cpu_port`.
 */

#pragma once

#include <cmath>   // `std::fabs`, `std::pow`
#include <cstdint> // `std::uint8_t`, `std::int32_t`

#include <array>   // `std::array`
#include <span>    // `std::span`
#include <utility> // `std::pair`

#include "smallfield/module_fwd.hpp"
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

/** Matches `RDKit::MMFF::getPeriodicTableRow`: row 0 holds H and He; row 1 Li through Ne;
 *  row 2 Na through Ar; row 3 K through Kr; row 4 Rb through Xe. */
SF_HOST_DEVICE_ inline std::uint8_t periodic_table_row(std::uint8_t atomic_num) noexcept {
    if (atomic_num >= 3 && atomic_num <= 10) return 1;
    if (atomic_num >= 11 && atomic_num <= 18) return 2;
    if (atomic_num >= 19 && atomic_num <= 36) return 3;
    if (atomic_num >= 37 && atomic_num <= 54) return 4;
    return 0;
}

/** Matches `RDKit::MMFF::getPeriodicTableRowHL` — 1-based, transition
 *  metals ×10. Used only by the Herschbach-Laurie fallback. */
SF_HOST_DEVICE_ inline std::uint8_t periodic_table_row_hl(std::uint8_t atomic_num) noexcept {
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

/** Find the bond index between atoms i and j via the descriptor's neighbor CSR.
 *  Returns -1 on miss. Matches `find_bond_index` in
 *  `etkdg_preprocess.hpp`. */
SF_HOST_DEVICE_ inline std::int32_t find_bond_index(molecule_descriptor_t const &descriptor, std::uint32_t i,
                                                    std::uint32_t j) noexcept {
    std::uint16_t const start = descriptor.neighbor_offset[i];
    std::uint16_t const end = descriptor.neighbor_offset[i + 1];
    for (std::uint16_t n = start; n < end; ++n)
        if (descriptor.neighbor_atom[n] == j) return descriptor.neighbor_bond[n];
    return -1;
}

/** Are `atom_i` and `atom_k` both adjacent to `atom_j` AND to each other?
 *  I.e., does the i-j-k angle close back into a 3-ring. Purely uses the
 *  descriptor's CSR — no ring info required. */
SF_HOST_DEVICE_ inline bool angle_in_ring_of_size_3(molecule_descriptor_t const &descriptor, std::uint32_t i,
                                                    std::uint32_t k) noexcept {
    return find_bond_index(descriptor, i, k) >= 0;
}

/** Do atoms i and k share a common neighbor that isn't j? Matches
 *  `RDKit::MMFF::isAngleInRingOfSize3or4`'s size-4 branch exactly. */
SF_HOST_DEVICE_ inline bool angle_has_common_neighbor(molecule_descriptor_t const &descriptor, std::uint32_t i,
                                                      std::uint32_t j, std::uint32_t k) noexcept {
    std::uint16_t const atom_i_neighbor_start = descriptor.neighbor_offset[i];
    std::uint16_t const atom_i_neighbor_end = descriptor.neighbor_offset[i + 1];
    for (std::uint16_t neighbor_offset_i = atom_i_neighbor_start; neighbor_offset_i < atom_i_neighbor_end;
         ++neighbor_offset_i) {
        std::uint32_t const candidate_atom = descriptor.neighbor_atom[neighbor_offset_i];
        if (candidate_atom == j) continue;
        if (find_bond_index(descriptor, candidate_atom, k) >= 0 && candidate_atom != j) return true;
    }
    return false;
}

/** Ring size containing the i-j-k angle, or 0 if not 3/4. Matches
 *  `isAngleInRingOfSize3or4` exactly. */
SF_HOST_DEVICE_ inline unsigned native_is_angle_in_ring_3_or_4(molecule_descriptor_t const &descriptor, std::uint32_t i,
                                                               std::uint32_t j, std::uint32_t k) noexcept {
    // Caller should hold that i-j and j-k are real bonds; we only
    // trigger on the ring closure checks here.
    if (find_bond_index(descriptor, i, j) < 0 || find_bond_index(descriptor, j, k) < 0) return 0;
    if (angle_in_ring_of_size_3(descriptor, i, k)) return 3;
    if (angle_has_common_neighbor(descriptor, i, j, k)) return 4;
    return 0;
}

/** Ring size containing the i-j-k-l torsion, or 0 if not 4/5. Matches
 *  `isTorsionInRingOfSize4or5` exactly. */
SF_HOST_DEVICE_ inline unsigned native_is_torsion_in_ring_4_or_5(molecule_descriptor_t const &descriptor,
                                                                 std::uint32_t i, std::uint32_t j, std::uint32_t k,
                                                                 std::uint32_t l) noexcept {
    if (find_bond_index(descriptor, i, j) < 0 || find_bond_index(descriptor, j, k) < 0 ||
        find_bond_index(descriptor, k, l) < 0)
        return 0;
    if (find_bond_index(descriptor, l, i) >= 0) return 4;
    // Size-5: do atoms i and l share a common neighbor that isn't j/k?
    std::uint16_t const atom_i_neighbor_start = descriptor.neighbor_offset[i];
    std::uint16_t const atom_i_neighbor_end = descriptor.neighbor_offset[i + 1];
    for (std::uint16_t neighbor_offset_i = atom_i_neighbor_start; neighbor_offset_i < atom_i_neighbor_end;
         ++neighbor_offset_i) {
        std::uint32_t const candidate_from_i = descriptor.neighbor_atom[neighbor_offset_i];
        if (candidate_from_i == j) continue;
        // Is `candidate_from_i` also a neighbor of l, and not k?
        std::uint16_t const atom_l_neighbor_start = descriptor.neighbor_offset[l];
        std::uint16_t const atom_l_neighbor_end = descriptor.neighbor_offset[l + 1];
        for (std::uint16_t neighbor_offset_l = atom_l_neighbor_start; neighbor_offset_l < atom_l_neighbor_end;
             ++neighbor_offset_l) {
            std::uint32_t const candidate_from_l = descriptor.neighbor_atom[neighbor_offset_l];
            if (candidate_from_l == k) continue;
            if (candidate_from_i == candidate_from_l) return 5;
        }
    }
    return 0;
}

#pragma endregion Neighbor CSR Helpers

#pragma region Native Type Classification

/** Native port of `MMFFMolProperties::getMMFFBondType`. Returns 1 iff the bond is SINGLE and both
 *  endpoints are sbmb atoms or both are aromatic
 *  atoms (per MMFFProp). Otherwise returns 0. */
SF_HOST_DEVICE_ inline std::uint8_t native_get_mmff_bond_type(molecule_descriptor_t const &descriptor,
                                                              mmff94_tables_view_t const &t,
                                                              std::span<std::uint8_t const> atom_types,
                                                              std::uint32_t bond_index) noexcept {
    std::uint8_t const bo = descriptor.bond_order[bond_index];
    bool const aromatic_bond = (descriptor.bond_flags[bond_index] & bond_flag_aromatic_k) != 0;
    // RDKit's rule triggers only when the bond is SINGLE. Aromatic bonds in
    // the descriptor report order=1 with the aromatic flag set, so we key on
    // `bo == 1 && !aromatic_bond` to exclude aromatic.
    if (!(bo == 1 && !aromatic_bond)) return 0;
    std::uint8_t const i = descriptor.bond_atom_i[bond_index];
    std::uint8_t const j = descriptor.bond_atom_j[bond_index];
    auto const *pi = lookup_prop(t, atom_types[i]);
    auto const *pj = lookup_prop(t, atom_types[j]);
    if (!pi || !pj) return 0;
    if ((pi->sbmb && pj->sbmb) || (pi->arom && pj->arom)) return 1;
    return 0;
}

/** Native port of `MMFFMolProperties::getMMFFAngleType`. */
SF_HOST_DEVICE_ inline std::uint8_t native_get_mmff_angle_type(molecule_descriptor_t const &descriptor,
                                                               std::span<std::uint8_t const> bond_types,
                                                               std::uint32_t i, std::uint32_t j,
                                                               std::uint32_t k) noexcept {
    std::int32_t const bond_index_ij = find_bond_index(descriptor, i, j);
    std::int32_t const bond_index_jk = find_bond_index(descriptor, j, k);
    if (bond_index_ij < 0 || bond_index_jk < 0) return 0;
    unsigned const bond_type_sum = bond_types[bond_index_ij] + bond_types[bond_index_jk];
    unsigned angle_type = bond_type_sum;
    unsigned const size = native_is_angle_in_ring_3_or_4(descriptor, i, j, k);
    if (size) {
        angle_type = size;
        if (bond_type_sum) angle_type += (bond_type_sum + size - 2);
    }
    return static_cast<std::uint8_t>(angle_type);
}

/** Native port of `MMFFMolProperties::getMMFFTorsionType`, returning
 *  (primary, secondary) matching RDKit's pair. */
SF_HOST_DEVICE_ inline std::pair<std::uint8_t, std::uint8_t> native_get_mmff_torsion_type(
    molecule_descriptor_t const &descriptor, std::span<std::uint8_t const> atom_types,
    std::span<std::uint8_t const> bond_types, std::uint32_t i, std::uint32_t j, std::uint32_t k,
    std::uint32_t l) noexcept {
    std::int32_t const bond_index_jk = find_bond_index(descriptor, j, k);
    std::int32_t const bond_index_ij = find_bond_index(descriptor, i, j);
    std::int32_t const bond_index_kl = find_bond_index(descriptor, k, l);
    if (bond_index_jk < 0 || bond_index_ij < 0 || bond_index_kl < 0) return {0, 0};

    std::uint8_t const bond_type_ij = bond_types[bond_index_ij];
    std::uint8_t const bond_type_jk = bond_types[bond_index_jk];
    std::uint8_t const bond_type_kl = bond_types[bond_index_kl];
    bool const jk_single = (descriptor.bond_order[bond_index_jk] == 1) &&
                           !(descriptor.bond_flags[bond_index_jk] & bond_flag_aromatic_k);

    unsigned torsion_type = bond_type_jk;
    unsigned secondary = 0;
    if (bond_type_jk == 0 && jk_single && (bond_type_ij == 1 || bond_type_kl == 1)) torsion_type = 2;

    unsigned const size = native_is_torsion_in_ring_4_or_5(descriptor, i, j, k, l);
    if (size == 4) {
        // Skip the 4-ring reassignment for torsions where the "4-ring" is
        // actually a pair of fused 3-rings (i-k or j-l already bonded).
        bool const fused_3_3 = (find_bond_index(descriptor, i, k) >= 0) || (find_bond_index(descriptor, j, l) >= 0);
        if (!fused_3_3) {
            secondary = torsion_type;
            torsion_type = 4;
        }
    }
    else if (size == 5) {
        // Only if any of the four atoms has MMFF type 1.
        if (atom_types[i] == 1 || atom_types[j] == 1 || atom_types[k] == 1 || atom_types[l] == 1) {
            secondary = torsion_type;
            torsion_type = 5;
        }
    }
    return {static_cast<std::uint8_t>(torsion_type), static_cast<std::uint8_t>(secondary)};
}

/** Native port of the free function `getMMFFStretchBendType`. */
SF_HOST_DEVICE_ inline std::uint8_t native_get_mmff_stretch_bend_type(unsigned angle_type, unsigned bond_type_1,
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

/** Empirical bond-stretch rule from MMFF.V equations 18-19. Matches
 *  `getMMFFBondStretchEmpiricalRuleParams` and returns (kb, r0) by value.
 *  Internally uses `double` to match RDKit's precision; inputs/outputs are
 *  float for parity with the extracted-table pipeline. */
SF_HOST_DEVICE_ inline std::pair<float, float> native_bond_empirical(mmff94_tables_view_t const &t,
                                                                     std::uint8_t atomic_i,
                                                                     std::uint8_t atomic_j) noexcept {
    auto const *cov_i = lookup_covrad(t, atomic_i);
    auto const *cov_j = lookup_covrad(t, atomic_j);
    if (!cov_i || !cov_j) return {0.0f, 0.0f};
    double const c = ((atomic_i == 1 || atomic_j == 1) ? 0.050 : 0.085);
    double const n = 1.4;
    double const delta = 0.0;
    double const chi_abs = std::fabs(static_cast<double>(cov_i->chi) - cov_j->chi);
    double const r0 = static_cast<double>(cov_i->r0) + cov_j->r0 - c * std::pow(chi_abs, n) - delta;

    auto const *bndk = lookup_bndk(t, atomic_i, atomic_j);
    double kb = 0.0;
    if (bndk) {
        double const coeff = static_cast<double>(bndk->r0) / r0;
        double const coeff2 = coeff * coeff;
        double const coeff6 = coeff2 * coeff2 * coeff2;
        kb = static_cast<double>(bndk->kb) * coeff6;
    }
    else {
        auto const *hl = lookup_hl(t, periodic_table_row_hl(atomic_i), periodic_table_row_hl(atomic_j));
        if (hl) kb = std::pow(10.0, -(r0 - hl->a_ij) / hl->d_ij);
    }
    return {static_cast<float>(kb), static_cast<float>(r0)};
}

/** Empirical angle-bend rule (MMFF.V, Table VI + eqs). Takes already-
 *  computed bond params for the two flanking bonds and returns (ka, θ₀deg).
 *  Matches `getMMFFAngleBendEmpiricalRuleParams`. If `theta0_override_deg`
 *  is finite (≥ 0), it replaces the crd/mltb/val-derived default θ₀; use
 *  this to match RDKit's behaviour when a table entry exists with ka=0
 *  (then RDKit keeps the table's θ₀ and recomputes ka empirically). */
SF_HOST_DEVICE_ inline std::pair<float, float> native_angle_empirical(molecule_descriptor_t const &descriptor,
                                                                      mmff94_tables_view_t const &t,
                                                                      std::span<std::uint8_t const> atom_types,
                                                                      std::uint32_t i, std::uint32_t j, std::uint32_t k,
                                                                      float r0_ij, float r0_jk,
                                                                      float theta0_override_deg = -1.0f) noexcept {
    std::uint8_t const a_i = descriptor.element[i], a_j = descriptor.element[j], a_k = descriptor.element[k];

    // θ₀ — depends on central-atom crd / mltb / val / linh.
    auto const *prop_j = lookup_prop(t, atom_types[j]);
    float theta0_deg = 120.0f;
    if (prop_j) {
        switch (prop_j->crd) {
        case 4: theta0_deg = 109.45f; break;
        case 2:
            if (a_j == 8) theta0_deg = 105.0f;
            else if (prop_j->linh == 1) theta0_deg = 180.0f;
            break;
        case 3:
            if (prop_j->val == 3 && prop_j->mltb == 0) { theta0_deg = (a_j == 7) ? 107.0f : 92.0f; }
            break;
        }
    }
    unsigned const ring_size = native_is_angle_in_ring_3_or_4(descriptor, i, j, k);
    if (ring_size == 3) theta0_deg = 60.0f;
    else if (ring_size == 4) theta0_deg = 90.0f;
    if (theta0_override_deg >= 0.0f) theta0_deg = theta0_override_deg;

    // ka — Z_i * C_j * Z_k * r0_ij * r0_jk ...
    auto row_Z_C = [](std::uint8_t an, double &Z, double &C) {
        // Table VI — MMFF.V, page 628 (RDKit AtomTyper.cpp lines 2787-2854).
        Z = 0.0;
        C = 0.0;
        switch (an) {
        case 1: Z = 1.395; break;
        case 6:
            Z = 2.494;
            C = 1.016;
            break;
        case 7:
            Z = 2.711;
            C = 1.113;
            break;
        case 8:
            Z = 3.045;
            C = 1.337;
            break;
        case 9: Z = 2.847; break;
        case 14:
            Z = 2.350;
            C = 0.811;
            break;
        case 15:
            Z = 2.350;
            C = 1.068;
            break;
        case 16:
            Z = 2.980;
            C = 1.249;
            break;
        case 17:
            Z = 2.909;
            C = 1.078;
            break;
        case 35: Z = 3.017; break;
        case 53: Z = 3.086; break;
        }
    };
    double Zi = 0.0, Ci = 0.0, Zj = 0.0, Cj = 0.0, Zk = 0.0, Ck = 0.0;
    row_Z_C(a_i, Zi, Ci);
    row_Z_C(a_j, Zj, Cj);
    row_Z_C(a_k, Zk, Ck);

    double beta = 1.75;
    if (ring_size == 4) beta *= 0.85;
    else if (ring_size == 3) beta *= 0.05;
    double const DEG2RAD = M_PI / 180.0;
    double const theta0_rad = static_cast<double>(theta0_deg) * DEG2RAD;
    double const rij = r0_ij, rjk = r0_jk;
    double const D = (rij - rjk) * (rij - rjk) / ((rij + rjk) * (rij + rjk));
    double const ka = beta * Zi * Cj * Zk / ((rij + rjk) * theta0_rad * theta0_rad * std::exp(2.0 * D));
    return {static_cast<float>(ka), theta0_deg};
}

/** Empirical torsion rule (MMFF.V rules a-h). Matches
 *  `getMMFFTorsionEmpiricalRuleParams`. Returns (V1, V2, V3). Bond type
 *  of the central bond is passed in so we avoid re-looking-it-up. */
SF_HOST_DEVICE_ inline std::array<float, 3> native_torsion_empirical(molecule_descriptor_t const &descriptor,
                                                                     mmff94_tables_view_t const &t,
                                                                     std::span<std::uint8_t const> atom_types,
                                                                     std::uint32_t j, std::uint32_t k,
                                                                     std::int32_t bond_jk_index) noexcept {
    auto const *jp = lookup_prop(t, atom_types[j]);
    auto const *kp = lookup_prop(t, atom_types[k]);
    std::array<float, 3> out {0.0f, 0.0f, 0.0f};
    if (!jp || !kp) return out;
    auto U_V_W = [](std::uint8_t an, float &U, float &V, float &W) {
        U = V = W = 0.0f;
        switch (an) {
        case 6:
            U = 2.00f;
            V = 2.12f;
            break;
        case 7:
            U = 2.00f;
            V = 1.50f;
            break;
        case 8:
            U = 2.00f;
            V = 0.20f;
            W = 2.0f;
            break;
        case 14:
            U = 1.25f;
            V = 1.22f;
            break;
        case 15:
            U = 1.25f;
            V = 2.40f;
            break;
        case 16:
            U = 1.25f;
            V = 0.49f;
            W = 8.0f;
            break;
        }
    };
    float U0, V0, W0, U1, V1_, W1;
    U_V_W(descriptor.element[j], U0, V0, W0);
    U_V_W(descriptor.element[k], U1, V1_, W1);
    float const N_jk = static_cast<float>((jp->crd - 1) * (kp->crd - 1));
    bool const jk_double = descriptor.bond_order[bond_jk_index] == 2;
    bool const jk_aromatic = (descriptor.bond_flags[bond_jk_index] & bond_flag_aromatic_k) != 0;
    bool const jk_single = descriptor.bond_order[bond_jk_index] == 1 && !jk_aromatic;

    // rule (a): linear heavy atom on either side — all zero.
    if (jp->linh || kp->linh) return out;

    // rule (b): aromatic-aromatic in aromatic ring.
    // RDKit uses MMFFArom set of atom types (never nullptr for non-zero types).
    // We replicate with a small constant list — see `DefaultParameters::getMMFFArom`.
    // MMFFArom types: {37, 38, 39, 44, 58, 59, 63, 64, 65, 66, 69, 76, 78, 79, 80, 81, 82}.
    auto is_mmff_arom = [](std::uint8_t at) {
        switch (at) {
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
    if (is_mmff_arom(atom_types[j]) && is_mmff_arom(atom_types[k]) && jk_aromatic) {
        float const beta = (((jp->val == 3 && kp->val == 4) || (jp->val == 4 && kp->val == 3)) ? 3.0f : 6.0f);
        float const pi_jk = ((jp->pilp == 0 && kp->pilp == 0) ? 0.5f : 0.3f);
        out[1] = beta * pi_jk * std::sqrt(U0 * U1);
        return out;
    }

    // rule (c): central bond is double.
    if (jk_double) {
        float const beta = 6.0f;
        float const pi_jk = ((jp->mltb == 2 && kp->mltb == 2) ? 1.0f : 0.4f);
        out[1] = beta * pi_jk * std::sqrt(U0 * U1);
        return out;
    }

    // rule (d): sp3-sp3.
    if (jp->crd == 4 && kp->crd == 4) {
        out[2] = std::sqrt(V0 * V1_) / N_jk;
        return out;
    }

    // rules (e) and (f): sp3-X (or X-sp3).
    if ((jp->crd == 4 && kp->crd != 4) || (kp->crd == 4 && jp->crd != 4)) {
        auto const *np = (jp->crd == 4) ? kp : jp;
        bool zero = false;
        if ((np->crd == 3 && ((np->val == 4 || np->val == 34) || np->mltb)) ||
            (np->crd == 2 && (np->val == 3 || np->mltb)))
            zero = true;
        if (!zero) out[2] = std::sqrt(V0 * V1_) / N_jk;
        return out;
    }

    // rule (g): mltb / pilp combinations.
    bool const g_trigger = (jk_single && jp->mltb && kp->mltb) || (jp->mltb && kp->pilp) || (jp->pilp && kp->mltb);
    if (g_trigger) {
        if (jp->pilp && kp->pilp) return out;
        std::uint8_t const ai = descriptor.element[j], ak = descriptor.element[k];
        std::uint8_t const row_j = periodic_table_row(ai);
        std::uint8_t const row_k = periodic_table_row(ak);
        float pi_jk = 0.0f;
        float const beta = 6.0f;
        if (jp->pilp && kp->mltb) {
            if (jp->mltb == 1) pi_jk = 0.5f;
            else if (row_j == 2 && row_k == 2) pi_jk = 0.3f;
            else pi_jk = 0.15f;
        }
        else if (kp->pilp && jp->mltb) {
            if (kp->mltb == 1) pi_jk = 0.5f;
            else if (row_j == 2 && row_k == 2) pi_jk = 0.3f;
            else pi_jk = 0.15f;
        }
        else if ((jp->mltb == 1 || kp->mltb == 1) && (ai != 6 || ak != 6)) pi_jk = 0.4f;
        else pi_jk = 0.15f;
        out[1] = beta * pi_jk * std::sqrt(U0 * U1);
        return out;
    }

    // rule (h).
    std::uint8_t const ai = descriptor.element[j], ak = descriptor.element[k];
    if ((ai == 8 || ai == 16) && (ak == 8 || ak == 16)) { out[1] = -std::sqrt(W0 * W1); }
    else { out[2] = std::sqrt(V0 * V1_) / N_jk; }
    return out;
}

#pragma endregion Empirical Fallback Rules

#pragma region Native Extractor

/** isDoubleZero predicate matching RDKit's 1e-10 threshold. Used by the
 *  angle-bend + torsion fallback logic. */
SF_HOST_DEVICE_ inline bool native_is_zero(float v) noexcept { return std::fabs(v) < 1.0e-10f; }

#pragma endregion Native Extractor

} // namespace smallfield::mmff94
