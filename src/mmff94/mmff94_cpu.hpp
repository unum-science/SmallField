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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "types/etkdg_types.hpp"
#include "types/mmff94_bounds.hpp"
#include "parameters/mmff94_tables.hpp"
#include "mmff94/mmff94_typer.hpp"
#include "types/molecule_descriptor.hpp"

// `extract_mmff94_params_native` + `extract_mmff94_params_fully_native`
// at the bottom of this file use `etkdg_prepared_molecule_t` from
// `etkdg_preprocess.hpp`. That header transitively pulls in RDKit and
// cannot be compiled by a .cu file's host pass. Gate the two extract
// functions on the presence of the full struct definition so .cu
// callers that only want the helpers (`native_*`, periodic-table rows,
// etc.) can include `mmff94_cpu.hpp` via `SMALLFIELD_MMFF_TABLES_NO_RDKIT`.
#if !defined(SMALLFIELD_MMFF_TABLES_NO_RDKIT)
#define SMALLFIELD_MMFF_CPU_HAS_PREP 1
#else
#define SMALLFIELD_MMFF_CPU_HAS_PREP 0
#endif

namespace smallfield {

// Periodic-table row utilities that match RDKit's helpers.

/** Matches `RDKit::MMFF::getPeriodicTableRow`: row 0 holds H and He; row 1 Li through Ne;
 *  row 2 Na through Ar; row 3 K through Kr; row 4 Rb through Xe. */
SMALLFIELD_HD inline std::uint8_t periodic_table_row(std::uint8_t atomic_num) noexcept {
    if (atomic_num >= 3 && atomic_num <= 10) return 1;
    if (atomic_num >= 11 && atomic_num <= 18) return 2;
    if (atomic_num >= 19 && atomic_num <= 36) return 3;
    if (atomic_num >= 37 && atomic_num <= 54) return 4;
    return 0;
}

/** Matches `RDKit::MMFF::getPeriodicTableRowHL` — 1-based, transition
 *  metals ×10.  Used only by the Herschbach-Laurie fallback. */
SMALLFIELD_HD inline std::uint8_t periodic_table_row_hl(std::uint8_t atomic_num) noexcept {
    std::uint8_t row = 0;
    if (atomic_num == 2) row = 1;
    else if (atomic_num >= 3 && atomic_num <= 10) row = 2;
    else if (atomic_num >= 11 && atomic_num <= 18) row = 3;
    else if (atomic_num >= 19 && atomic_num <= 36) row = 4;
    else if (atomic_num >= 37 && atomic_num <= 54) row = 5;
    if ((atomic_num >= 21 && atomic_num <= 30) ||
        (atomic_num >= 39 && atomic_num <= 48)) {
        row = static_cast<std::uint8_t>(row * 10);
    }
    return row;
}

// Neighbor CSR convenience helpers.

/** Find the bond index between atoms i and j via the descriptor's neighbor CSR.
 *  Returns -1 on miss. Matches `find_bond_idx` in
 *  `etkdg_preprocess.hpp`. */
SMALLFIELD_HD inline std::int32_t find_bond_idx(molecule_descriptor_t const &desc,
                                  std::uint32_t i, std::uint32_t j) noexcept {
    std::uint16_t const start = desc.neighbor_offset[i];
    std::uint16_t const end = desc.neighbor_offset[i + 1];
    for (std::uint16_t n = start; n < end; ++n)
        if (desc.neighbor_atom[n] == j) return desc.neighbor_bond[n];
    return -1;
}

/** Are `atom_i` and `atom_k` both adjacent to `atom_j` AND to each other?
 *  I.e., does the i-j-k angle close back into a 3-ring.  Purely uses the
 *  descriptor's CSR — no ring info required. */
SMALLFIELD_HD inline bool angle_in_ring_of_size_3(molecule_descriptor_t const &desc,
                                    std::uint32_t i, std::uint32_t k) noexcept {
    return find_bond_idx(desc, i, k) >= 0;
}

/** Do atoms i and k share a common neighbor that isn't j?  Matches
 *  `RDKit::MMFF::isAngleInRingOfSize3or4`'s size-4 branch exactly. */
SMALLFIELD_HD inline bool angle_has_common_neighbor(molecule_descriptor_t const &desc,
                                      std::uint32_t i, std::uint32_t j,
                                      std::uint32_t k) noexcept {
    std::uint16_t const atom_i_neighbor_start = desc.neighbor_offset[i];
    std::uint16_t const atom_i_neighbor_end = desc.neighbor_offset[i + 1];
    for (std::uint16_t neighbor_offset_i = atom_i_neighbor_start;
         neighbor_offset_i < atom_i_neighbor_end; ++neighbor_offset_i) {
        std::uint32_t const candidate_atom = desc.neighbor_atom[neighbor_offset_i];
        if (candidate_atom == j) continue;
        if (find_bond_idx(desc, candidate_atom, k) >= 0 && candidate_atom != j) return true;
    }
    return false;
}

/** Ring size containing the i-j-k angle, or 0 if not 3/4.  Matches
 *  `isAngleInRingOfSize3or4` exactly. */
SMALLFIELD_HD inline unsigned native_is_angle_in_ring_3_or_4(molecule_descriptor_t const &desc,
                                               std::uint32_t i, std::uint32_t j,
                                               std::uint32_t k) noexcept {
    // Caller should hold that i-j and j-k are real bonds; we only
    // trigger on the ring closure checks here.
    if (find_bond_idx(desc, i, j) < 0 || find_bond_idx(desc, j, k) < 0) return 0;
    if (angle_in_ring_of_size_3(desc, i, k)) return 3;
    if (angle_has_common_neighbor(desc, i, j, k)) return 4;
    return 0;
}

/** Ring size containing the i-j-k-l torsion, or 0 if not 4/5.  Matches
 *  `isTorsionInRingOfSize4or5` exactly. */
SMALLFIELD_HD inline unsigned native_is_torsion_in_ring_4_or_5(molecule_descriptor_t const &desc,
                                                 std::uint32_t i, std::uint32_t j,
                                                 std::uint32_t k, std::uint32_t l) noexcept {
    if (find_bond_idx(desc, i, j) < 0 || find_bond_idx(desc, j, k) < 0 ||
        find_bond_idx(desc, k, l) < 0) return 0;
    if (find_bond_idx(desc, l, i) >= 0) return 4;
    // Size-5: do atoms i and l share a common neighbor that isn't j/k?
    std::uint16_t const atom_i_neighbor_start = desc.neighbor_offset[i];
    std::uint16_t const atom_i_neighbor_end = desc.neighbor_offset[i + 1];
    for (std::uint16_t neighbor_offset_i = atom_i_neighbor_start;
         neighbor_offset_i < atom_i_neighbor_end; ++neighbor_offset_i) {
        std::uint32_t const candidate_from_i = desc.neighbor_atom[neighbor_offset_i];
        if (candidate_from_i == j) continue;
        // Is `candidate_from_i` also a neighbor of l, and not k?
        std::uint16_t const atom_l_neighbor_start = desc.neighbor_offset[l];
        std::uint16_t const atom_l_neighbor_end = desc.neighbor_offset[l + 1];
        for (std::uint16_t neighbor_offset_l = atom_l_neighbor_start;
             neighbor_offset_l < atom_l_neighbor_end; ++neighbor_offset_l) {
            std::uint32_t const candidate_from_l = desc.neighbor_atom[neighbor_offset_l];
            if (candidate_from_l == k) continue;
            if (candidate_from_i == candidate_from_l) return 5;
        }
    }
    return 0;
}

// Native MMFF type classification.

/** Native port of `MMFFMolProperties::getMMFFBondType`. Returns 1 iff the bond is SINGLE and both
 *  endpoints are sbmb atoms or both are aromatic
 *  atoms (per MMFFProp).  Otherwise returns 0. */
SMALLFIELD_HD inline std::uint8_t native_get_mmff_bond_type(molecule_descriptor_t const &desc,
                                              mmff94_tables_view_t const &t,
                                              std::span<std::uint8_t const> atom_types,
                                              std::uint32_t bond_idx) noexcept {
    std::uint8_t const bo = desc.bond_order[bond_idx];
    bool const aromatic_bond = (desc.bond_flags[bond_idx] & bond_flag_aromatic_k) != 0;
    // RDKit's rule triggers only when the bond is SINGLE. Aromatic bonds in
    // the descriptor report order=1 with the aromatic flag set, so we key on
    // `bo == 1 && !aromatic_bond` to exclude aromatic.
    if (!(bo == 1 && !aromatic_bond)) return 0;
    std::uint8_t const i = desc.bond_atom_i[bond_idx];
    std::uint8_t const j = desc.bond_atom_j[bond_idx];
    auto const *pi = lookup_prop(t, atom_types[i]);
    auto const *pj = lookup_prop(t, atom_types[j]);
    if (!pi || !pj) return 0;
    if ((pi->sbmb && pj->sbmb) || (pi->arom && pj->arom)) return 1;
    return 0;
}

/** Native port of `MMFFMolProperties::getMMFFAngleType`. */
SMALLFIELD_HD inline std::uint8_t native_get_mmff_angle_type(molecule_descriptor_t const &desc,
                                               std::span<std::uint8_t const> bond_types,
                                               std::uint32_t i, std::uint32_t j,
                                               std::uint32_t k) noexcept {
    std::int32_t const bond_index_ij = find_bond_idx(desc, i, j);
    std::int32_t const bond_index_jk = find_bond_idx(desc, j, k);
    if (bond_index_ij < 0 || bond_index_jk < 0) return 0;
    unsigned const bond_type_sum = bond_types[bond_index_ij] + bond_types[bond_index_jk];
    unsigned angle_type = bond_type_sum;
    unsigned const size = native_is_angle_in_ring_3_or_4(desc, i, j, k);
    if (size) {
        angle_type = size;
        if (bond_type_sum) angle_type += (bond_type_sum + size - 2);
    }
    return static_cast<std::uint8_t>(angle_type);
}

/** Native port of `MMFFMolProperties::getMMFFTorsionType`, returning
 *  (primary, secondary) matching RDKit's pair. */
SMALLFIELD_HD inline std::pair<std::uint8_t, std::uint8_t>
native_get_mmff_torsion_type(molecule_descriptor_t const &desc,
                             std::span<std::uint8_t const> atom_types,
                             std::span<std::uint8_t const> bond_types,
                             std::uint32_t i, std::uint32_t j,
                             std::uint32_t k, std::uint32_t l) noexcept {
    std::int32_t const bond_index_jk = find_bond_idx(desc, j, k);
    std::int32_t const bond_index_ij = find_bond_idx(desc, i, j);
    std::int32_t const bond_index_kl = find_bond_idx(desc, k, l);
    if (bond_index_jk < 0 || bond_index_ij < 0 || bond_index_kl < 0) return {0, 0};

    std::uint8_t const bond_type_ij = bond_types[bond_index_ij];
    std::uint8_t const bond_type_jk = bond_types[bond_index_jk];
    std::uint8_t const bond_type_kl = bond_types[bond_index_kl];
    bool const jk_single = (desc.bond_order[bond_index_jk] == 1) &&
                           !(desc.bond_flags[bond_index_jk] & bond_flag_aromatic_k);

    unsigned torsion_type = bond_type_jk;
    unsigned secondary = 0;
    if (bond_type_jk == 0 && jk_single && (bond_type_ij == 1 || bond_type_kl == 1)) torsion_type = 2;

    unsigned const size = native_is_torsion_in_ring_4_or_5(desc, i, j, k, l);
    if (size == 4) {
        // Skip the 4-ring reassignment for torsions where the "4-ring" is
        // actually a pair of fused 3-rings (i-k or j-l already bonded).
        bool const fused_3_3 = (find_bond_idx(desc, i, k) >= 0) ||
                               (find_bond_idx(desc, j, l) >= 0);
        if (!fused_3_3) {
            secondary = torsion_type;
            torsion_type = 4;
        }
    }
    else if (size == 5) {
        // Only if any of the four atoms has MMFF type 1.
        if (atom_types[i] == 1 || atom_types[j] == 1 || atom_types[k] == 1 ||
            atom_types[l] == 1) {
            secondary = torsion_type;
            torsion_type = 5;
        }
    }
    return {static_cast<std::uint8_t>(torsion_type), static_cast<std::uint8_t>(secondary)};
}

/** Native port of the free function `getMMFFStretchBendType`. */
SMALLFIELD_HD inline std::uint8_t native_get_mmff_stretch_bend_type(unsigned angle_type,
                                                      unsigned bond_type_1, unsigned bond_type_2) noexcept {
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

// Empirical fallback rules used when a parameter lookup misses.

/** Empirical bond-stretch rule from MMFF.V equations 18-19. Matches
 *  `getMMFFBondStretchEmpiricalRuleParams` and returns (kb, r0) by value.
 *  Internally uses `double` to match RDKit's precision; inputs/outputs are
 *  float for parity with the extracted-table pipeline. */
SMALLFIELD_HD inline std::pair<float, float> native_bond_empirical(mmff94_tables_view_t const &t,
                                                     std::uint8_t atomic_i,
                                                     std::uint8_t atomic_j) noexcept {
    auto const *cov_i = lookup_covrad(t, atomic_i);
    auto const *cov_j = lookup_covrad(t, atomic_j);
    if (!cov_i || !cov_j) return {0.0f, 0.0f};
    double const c = ((atomic_i == 1 || atomic_j == 1) ? 0.050 : 0.085);
    double const n = 1.4;
    double const delta = 0.0;
    double const chi_abs = std::fabs(static_cast<double>(cov_i->chi) - cov_j->chi);
    double const r0 = static_cast<double>(cov_i->r0) + cov_j->r0 -
                      c * std::pow(chi_abs, n) - delta;

    auto const *bndk = lookup_bndk(t, atomic_i, atomic_j);
    double kb = 0.0;
    if (bndk) {
        double const coeff = static_cast<double>(bndk->r0) / r0;
        double const coeff2 = coeff * coeff;
        double const coeff6 = coeff2 * coeff2 * coeff2;
        kb = static_cast<double>(bndk->kb) * coeff6;
    }
    else {
        auto const *hl = lookup_hl(t, periodic_table_row_hl(atomic_i),
                                   periodic_table_row_hl(atomic_j));
        if (hl) kb = std::pow(10.0, -(r0 - hl->a_ij) / hl->d_ij);
    }
    return {static_cast<float>(kb), static_cast<float>(r0)};
}

/** Empirical angle-bend rule (MMFF.V, Table VI + eqs).  Takes already-
 *  computed bond params for the two flanking bonds and returns (ka, θ₀deg).
 *  Matches `getMMFFAngleBendEmpiricalRuleParams`.  If `theta0_override_deg`
 *  is finite (≥ 0), it replaces the crd/mltb/val-derived default θ₀; use
 *  this to match RDKit's behaviour when a table entry exists with ka=0
 *  (then RDKit keeps the table's θ₀ and recomputes ka empirically). */
SMALLFIELD_HD inline std::pair<float, float> native_angle_empirical(
    molecule_descriptor_t const &desc, mmff94_tables_view_t const &t,
    std::span<std::uint8_t const> atom_types, std::uint32_t i, std::uint32_t j,
    std::uint32_t k, float r0_ij, float r0_jk,
    float theta0_override_deg = -1.0f) noexcept {
    std::uint8_t const a_i = desc.element[i], a_j = desc.element[j], a_k = desc.element[k];

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
                if (prop_j->val == 3 && prop_j->mltb == 0) {
                    theta0_deg = (a_j == 7) ? 107.0f : 92.0f;
                }
                break;
        }
    }
    unsigned const ring_size = native_is_angle_in_ring_3_or_4(desc, i, j, k);
    if (ring_size == 3) theta0_deg = 60.0f;
    else if (ring_size == 4) theta0_deg = 90.0f;
    if (theta0_override_deg >= 0.0f) theta0_deg = theta0_override_deg;

    // ka — Z_i * C_j * Z_k * r0_ij * r0_jk ...
    auto row_Z_C = [](std::uint8_t an, double &Z, double &C) {
        // Table VI — MMFF.V, page 628 (RDKit AtomTyper.cpp lines 2787-2854).
        Z = 0.0; C = 0.0;
        switch (an) {
            case 1:  Z = 1.395; break;
            case 6:  Z = 2.494; C = 1.016; break;
            case 7:  Z = 2.711; C = 1.113; break;
            case 8:  Z = 3.045; C = 1.337; break;
            case 9:  Z = 2.847; break;
            case 14: Z = 2.350; C = 0.811; break;
            case 15: Z = 2.350; C = 1.068; break;
            case 16: Z = 2.980; C = 1.249; break;
            case 17: Z = 2.909; C = 1.078; break;
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
    double const ka = beta * Zi * Cj * Zk /
                      ((rij + rjk) * theta0_rad * theta0_rad * std::exp(2.0 * D));
    return {static_cast<float>(ka), theta0_deg};
}

/** Empirical torsion rule (MMFF.V rules a-h).  Matches
 *  `getMMFFTorsionEmpiricalRuleParams`.  Returns (V1, V2, V3).  Bond type
 *  of the central bond is passed in so we avoid re-looking-it-up. */
SMALLFIELD_HD inline std::array<float, 3> native_torsion_empirical(
    molecule_descriptor_t const &desc, mmff94_tables_view_t const &t,
    std::span<std::uint8_t const> atom_types, std::uint32_t j, std::uint32_t k,
    std::int32_t bond_jk_idx) noexcept {
    auto const *jp = lookup_prop(t, atom_types[j]);
    auto const *kp = lookup_prop(t, atom_types[k]);
    std::array<float, 3> out {0.0f, 0.0f, 0.0f};
    if (!jp || !kp) return out;
    auto U_V_W = [](std::uint8_t an, float &U, float &V, float &W) {
        U = V = W = 0.0f;
        switch (an) {
            case 6:  U = 2.00f; V = 2.12f; break;
            case 7:  U = 2.00f; V = 1.50f; break;
            case 8:  U = 2.00f; V = 0.20f; W = 2.0f; break;
            case 14: U = 1.25f; V = 1.22f; break;
            case 15: U = 1.25f; V = 2.40f; break;
            case 16: U = 1.25f; V = 0.49f; W = 8.0f; break;
        }
    };
    float U0, V0, W0, U1, V1_, W1;
    U_V_W(desc.element[j], U0, V0, W0);
    U_V_W(desc.element[k], U1, V1_, W1);
    float const N_jk = static_cast<float>((jp->crd - 1) * (kp->crd - 1));
    bool const jk_double = desc.bond_order[bond_jk_idx] == 2;
    bool const jk_aromatic = (desc.bond_flags[bond_jk_idx] & bond_flag_aromatic_k) != 0;
    bool const jk_single = desc.bond_order[bond_jk_idx] == 1 && !jk_aromatic;

    // rule (a): linear heavy atom on either side — all zero.
    if (jp->linh || kp->linh) return out;

    // rule (b): aromatic-aromatic in aromatic ring.
    // RDKit uses MMFFArom set of atom types (never nullptr for non-zero types).
    // We replicate with a small constant list — see `DefaultParameters::getMMFFArom`.
    // MMFFArom types: {37, 38, 39, 44, 58, 59, 63, 64, 65, 66, 69, 76, 78, 79, 80, 81, 82}.
    auto is_mmff_arom = [](std::uint8_t at) {
        switch (at) {
            case 37: case 38: case 39: case 44: case 58: case 59:
            case 63: case 64: case 65: case 66: case 69: case 76:
            case 78: case 79: case 80: case 81: case 82: return true;
            default: return false;
        }
    };
    if (is_mmff_arom(atom_types[j]) && is_mmff_arom(atom_types[k]) && jk_aromatic) {
        float const beta = (((jp->val == 3 && kp->val == 4) ||
                             (jp->val == 4 && kp->val == 3)) ? 3.0f : 6.0f);
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
            (np->crd == 2 && (np->val == 3 || np->mltb))) zero = true;
        if (!zero) out[2] = std::sqrt(V0 * V1_) / N_jk;
        return out;
    }

    // rule (g): mltb / pilp combinations.
    bool const g_trigger = (jk_single && jp->mltb && kp->mltb) ||
                           (jp->mltb && kp->pilp) || (jp->pilp && kp->mltb);
    if (g_trigger) {
        if (jp->pilp && kp->pilp) return out;
        std::uint8_t const ai = desc.element[j], ak = desc.element[k];
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
    std::uint8_t const ai = desc.element[j], ak = desc.element[k];
    if ((ai == 8 || ai == 16) && (ak == 8 || ak == 16)) {
        out[1] = -std::sqrt(W0 * W1);
    }
    else {
        out[2] = std::sqrt(V0 * V1_) / N_jk;
    }
    return out;
}

// Top-level native extractor.

/** isDoubleZero predicate matching RDKit's 1e-10 threshold. Used by the
 *  angle-bend + torsion fallback logic. */
SMALLFIELD_HD inline bool native_is_zero(float v) noexcept { return std::fabs(v) < 1.0e-10f; }

/** Native MMFF94 term + CSR extractor.
 *
 *  @param[in] desc        Molecule descriptor (neighbors, ring info, elements).
 *  @param[in] atom_types  Per-atom MMFF type (size num_atoms).
 *  @param[in] charges     Per-atom MMFF partial charge (size num_atoms).
 *  @param[in] tables      Extracted MMFF parameter tables.
 *  @param[in,out] result  Prepared molecule; MMFF arrays populated in place.
 *
 *  Fills: `mmff_atom_vdw`, `mmff_atom_charge`, `bond_distance`,
 *  `mmff_bonds`, `mmff_angles`, `mmff_torsions`, `mmff_oops`,
 *  `mmff_stretchbends`, `mmff_interactions`, `mmff_atom_offsets`.
 *
 *  Matches `extract_mmff94_params` (the RDKit-backed path) byte-for-byte
 *  on well-typed drug-like inputs.
 */
#if SMALLFIELD_MMFF_CPU_HAS_PREP
inline void extract_mmff94_params_native(molecule_descriptor_t const &desc,
                                         std::span<std::uint8_t const> atom_types,
                                         std::span<float const> charges,
                                         mmff94_tables_view_t const &tables,
                                         etkdg_prepared_molecule_t &result) {
    result.mmff_atom_vdw.clear();
    result.mmff_atom_charge.clear();
    result.bond_distance.clear();
    result.mmff_bonds.clear();
    result.mmff_angles.clear();
    result.mmff_torsions.clear();
    result.mmff_oops.clear();
    result.mmff_stretchbends.clear();
    result.mmff_interactions.clear();
    result.mmff_atom_offsets.clear();

    std::uint32_t const num_atoms = desc.num_atoms;
    std::uint32_t const num_bonds = desc.num_bonds;

    // Per-atom VdW parameters and partial charge.
    result.mmff_atom_vdw.resize(num_atoms);
    result.mmff_atom_charge.resize(num_atoms);
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        auto const *vdw = lookup_vdw(tables, atom_types[a]);
        mmff94_atom_vdw_params_t params {};
        if (vdw) {
            params.alpha = vdw->alpha_i;
            params.n_effective = vdw->n_i;
            params.a_scale = vdw->a_i;
            params.g_factor = vdw->g_i;
            params.r_star = vdw->r_star;
            params.donor_acceptor = vdw->donor_acceptor;
        }
        result.mmff_atom_vdw[a] = params;
        result.mmff_atom_charge[a] = charges[a];
    }

    // Bond-distance copy: N×N, packed, no stride padding.
    result.bond_distance.resize(static_cast<std::size_t>(num_atoms) * num_atoms);
    for (std::uint32_t i = 0; i < num_atoms; ++i)
        for (std::uint32_t j = 0; j < num_atoms; ++j)
            result.bond_distance[i * num_atoms + j] = desc.bond_distance[i * max_atoms_k + j];

    // Precompute per-bond MMFF bond type.
    std::array<std::uint8_t, max_bonds_k> bond_type_arr {};
    for (std::uint32_t b = 0; b < num_bonds; ++b) {
        bond_type_arr[b] = native_get_mmff_bond_type(desc, tables, atom_types, b);
    }
    std::span<std::uint8_t const> bond_types {bond_type_arr.data(), num_bonds};

    // Bond terms. Retain r0_ij and kb per bond for downstream angle and stretchbend reuse.
    std::array<float, max_bonds_k> bond_r0 {};
    std::array<float, max_bonds_k> bond_kb {};
    for (std::uint32_t b = 0; b < num_bonds; ++b) {
        std::uint8_t const i = desc.bond_atom_i[b];
        std::uint8_t const j = desc.bond_atom_j[b];
        std::uint8_t const bt = bond_type_arr[b];
        auto const *bp = lookup_bond(tables, bt, atom_types[i], atom_types[j]);
        float kb = 0.0f, r0 = 0.0f;
        if (bp) { kb = bp->kb; r0 = bp->r0; }
        else {
            auto const emp = native_bond_empirical(tables, desc.element[i], desc.element[j]);
            kb = emp.first;
            r0 = emp.second;
        }
        bond_r0[b] = r0;
        bond_kb[b] = kb;
        result.mmff_bonds.push_back({static_cast<std::uint8_t>(i),
                                     static_cast<std::uint8_t>(j), {0, 0}, kb, r0, 0.0f});
    }

    // Angle terms. Also emit per-angle scratch — θ₀, ka, and the bond-type pair — consumed below
    // by the stretchbend pass.
    struct angle_scratch_t {
        std::uint8_t i, j, k;
        std::uint8_t angle_type;
        std::uint8_t bond_type_ij, bond_type_jk;
        std::int32_t bond_index_ij, bond_index_jk;
        float theta0_deg;
        float ka;
    };
    // Stack-allocated scratch — upper bound matches the GPU's per-molecule
    // angle slot count. Avoids a per-molecule heap allocation on the hot
    // prep path.
    std::array<angle_scratch_t, max_mmff_angles_per_mol_k> angles_scratch;
    std::uint32_t num_angles_scratch = 0;

    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint16_t const nstart = desc.neighbor_offset[center];
        std::uint16_t const nend = desc.neighbor_offset[center + 1];
        for (std::uint16_t a = nstart; a < nend; ++a) {
            for (std::uint16_t b = a + 1; b < nend; ++b) {
                std::uint32_t const ai = desc.neighbor_atom[a];
                std::uint32_t const aj = desc.neighbor_atom[b];
                std::int32_t const bond_index_ic = find_bond_idx(desc, ai, center);
                std::int32_t const bond_index_cj = find_bond_idx(desc, center, aj);
                if (bond_index_ic < 0 || bond_index_cj < 0) continue;
                std::uint8_t const at = native_get_mmff_angle_type(desc, bond_types, ai,
                                                                    center, aj);
                auto const *ap = lookup_angle(tables, at, atom_types[ai],
                                              atom_types[center], atom_types[aj]);
                float ka = 0.0f, theta0_deg = 0.0f;
                bool use_empirical = (!ap || native_is_zero(ap->ka));
                if (ap && !native_is_zero(ap->ka)) {
                    ka = ap->ka;
                    theta0_deg = ap->theta0_deg;
                }
                if (use_empirical) {
                    float const r0_ic = bond_r0[bond_index_ic];
                    float const r0_cj = bond_r0[bond_index_cj];
                    // If a table entry exists with ka=0, RDKit keeps its θ₀ and
                    // uses it inside the empirical ka computation — pass it as
                    // the override so we match byte-for-byte.
                    float const theta_override = ap ? ap->theta0_deg : -1.0f;
                    auto const emp = native_angle_empirical(desc, tables, atom_types,
                                                            ai, center, aj, r0_ic, r0_cj,
                                                            theta_override);
                    ka = emp.first;
                    theta0_deg = emp.second;
                }
                float const theta0_rad = theta0_deg * static_cast<float>(M_PI) / 180.0f;
                result.mmff_angles.push_back({static_cast<std::uint8_t>(ai),
                                              static_cast<std::uint8_t>(center),
                                              static_cast<std::uint8_t>(aj), 0, ka,
                                              theta0_rad, 0.0f});
                angles_scratch[num_angles_scratch++] = {static_cast<std::uint8_t>(ai),
                                           static_cast<std::uint8_t>(center),
                                           static_cast<std::uint8_t>(aj), at,
                                           bond_types[bond_index_ic], bond_types[bond_index_cj],
                                           bond_index_ic, bond_index_cj, theta0_deg, ka};
            }
        }
    }

    // Torsion terms.
    for (std::uint32_t b = 0; b < num_bonds; ++b) {
        std::uint32_t const j = desc.bond_atom_i[b];
        std::uint32_t const k = desc.bond_atom_j[b];
        std::uint16_t const jstart = desc.neighbor_offset[j];
        std::uint16_t const jend = desc.neighbor_offset[j + 1];
        std::uint16_t const kstart = desc.neighbor_offset[k];
        std::uint16_t const kend = desc.neighbor_offset[k + 1];
        for (std::uint16_t nj = jstart; nj < jend; ++nj) {
            std::uint32_t const i = desc.neighbor_atom[nj];
            if (i == k) continue;
            for (std::uint16_t nk = kstart; nk < kend; ++nk) {
                std::uint32_t const l = desc.neighbor_atom[nk];
                if (l == j) continue;
                auto const tt = native_get_mmff_torsion_type(desc, atom_types, bond_types,
                                                              i, j, k, l);
                auto const lookup = lookup_torsion(tables, tt.first, tt.second,
                                                    atom_types[i], atom_types[j],
                                                    atom_types[k], atom_types[l]);
                float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
                if (lookup.second) {
                    v1 = lookup.second->v1;
                    v2 = lookup.second->v2;
                    v3 = lookup.second->v3;
                }
                else {
                    auto const emp = native_torsion_empirical(desc, tables, atom_types,
                                                                j, k, static_cast<std::int32_t>(b));
                    v1 = emp[0];
                    v2 = emp[1];
                    v3 = emp[2];
                }
                bool const all_zero = native_is_zero(v1) && native_is_zero(v2) && native_is_zero(v3);
                if (all_zero) continue;
                result.mmff_torsions.push_back({static_cast<std::uint8_t>(i),
                                                static_cast<std::uint8_t>(j),
                                                static_cast<std::uint8_t>(k),
                                                static_cast<std::uint8_t>(l), v1, v2, v3});
            }
        }
    }

    // Out-of-plane bending terms.
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint32_t const degree = desc.degree[center];
        if (degree < 3) continue;
        std::uint16_t const nstart = desc.neighbor_offset[center];
        std::uint8_t neighbor_cache[8];
        std::uint32_t neighbor_count = 0;
        for (std::uint16_t n = nstart; n < desc.neighbor_offset[center + 1] && neighbor_count < 8; ++n)
            neighbor_cache[neighbor_count++] = desc.neighbor_atom[n];

        for (std::uint32_t a = 0; a < neighbor_count; ++a) {
            std::uint8_t plane[2];
            std::uint32_t plane_count = 0;
            for (std::uint32_t c = 0; c < neighbor_count && plane_count < 2; ++c)
                if (c != a) plane[plane_count++] = neighbor_cache[c];
            if (plane_count < 2) continue;
            auto const *op = lookup_oop(tables, atom_types[neighbor_cache[a]],
                                         atom_types[center], atom_types[plane[0]],
                                         atom_types[plane[1]]);
            if (!op) continue;
            result.mmff_oops.push_back({neighbor_cache[a], static_cast<std::uint8_t>(center),
                                         plane[0], plane[1], op->koop, {0.0f, 0.0f}});
        }
    }

    // Stretch-bend terms.
    for (std::uint32_t angle_index = 0; angle_index < num_angles_scratch; ++angle_index) {
        auto const &as = angles_scratch[angle_index];
        auto const *prop_j = lookup_prop(tables, atom_types[as.j]);
        if (prop_j && prop_j->linh) continue;

        std::uint8_t const sbt = native_get_mmff_stretch_bend_type(
            as.angle_type,
            (atom_types[as.i] <= atom_types[as.k]) ? as.bond_type_ij : as.bond_type_jk,
            (atom_types[as.i] <  atom_types[as.k]) ? as.bond_type_jk : as.bond_type_ij);
        auto stbn = lookup_stbn(tables, sbt, as.bond_type_ij, as.bond_type_jk,
                                atom_types[as.i], atom_types[as.j], atom_types[as.k]);
        // RDKit falls back to DFSB ONLY when the STBN pointer is null. If
        // STBN has an entry with both kba=0 (e.g., cyclopropane CR3R-CR3R-
        // CR3R), the term is dropped rather than re-looked-up in DFSB.
        float kba_ijk = 0.0f, kba_kji = 0.0f;
        bool swap_flag = false;
        if (stbn.second) {
            kba_ijk = stbn.second->kba_ijk;
            kba_kji = stbn.second->kba_kji;
            swap_flag = stbn.first;
        }
        else {
            auto dfsb = lookup_dfsb(tables, periodic_table_row(desc.element[as.i]),
                                     periodic_table_row(desc.element[as.j]),
                                     periodic_table_row(desc.element[as.k]));
            if (!dfsb.second) continue;
            kba_ijk = dfsb.second->kba_ijk;
            kba_kji = dfsb.second->kba_kji;
            swap_flag = dfsb.first;
        }
        if (native_is_zero(kba_ijk) && native_is_zero(kba_kji)) continue;
        if (swap_flag) std::swap(kba_ijk, kba_kji);
        float const theta0_rad = as.theta0_deg * static_cast<float>(M_PI) / 180.0f;
        float const r0_ij = bond_r0[as.bond_index_ij];
        float const r0_jk = bond_r0[as.bond_index_jk];
        result.mmff_stretchbends.push_back({as.i, as.j, as.k, 0, kba_ijk, kba_kji,
                                              theta0_rad, r0_ij, r0_jk, 0.0f});
    }

    // Per-atom CSR interaction list with the same layout as the RDKit path.
    auto const &bonds = result.mmff_bonds;
    auto const &ang_v = result.mmff_angles;
    auto const &tor_v = result.mmff_torsions;
    auto const &oop_v = result.mmff_oops;
    auto const &sb_v = result.mmff_stretchbends;

    struct type_counts_t {
        std::uint8_t bond, angle, torsion, oop, stretchbend;
    };
    std::array<type_counts_t, max_atoms_k> type_counts {};
    for (auto const &e : bonds) { type_counts[e.atom_i].bond++; type_counts[e.atom_j].bond++; }
    for (auto const &e : ang_v) { type_counts[e.atom_i].angle++; type_counts[e.atom_j].angle++; type_counts[e.atom_k].angle++; }
    for (auto const &e : tor_v) {
        type_counts[e.atom_i].torsion++; type_counts[e.atom_j].torsion++;
        type_counts[e.atom_k].torsion++; type_counts[e.atom_l].torsion++;
    }
    for (auto const &e : oop_v) {
        type_counts[e.atom_i].oop++; type_counts[e.atom_j].oop++; type_counts[e.atom_k].oop++; type_counts[e.atom_l].oop++;
    }
    for (auto const &e : sb_v) {
        type_counts[e.atom_i].stretchbend++; type_counts[e.atom_j].stretchbend++; type_counts[e.atom_k].stretchbend++;
    }

    result.mmff_atom_offsets.resize(num_atoms);
    std::uint32_t running = 0;
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        auto const &c = type_counts[a];
        auto &off = result.mmff_atom_offsets[a];
        off.bond_start = running;       off.bond_count = c.bond;           running += c.bond;
        off.angle_start = running;      off.angle_count = c.angle;         running += c.angle;
        off.torsion_start = running;    off.torsion_count = c.torsion;     running += c.torsion;
        off.oop_start = running;        off.oop_count = c.oop;             running += c.oop;
        off.stretchbend_start = running; off.stretchbend_count = c.stretchbend; running += c.stretchbend;
    }
    result.mmff_interactions.resize(running);

    struct cursors_t {
        std::uint32_t bond, angle, torsion, oop, stretchbend;
    };
    std::array<cursors_t, max_atoms_k> cursors {};
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        auto const &off = result.mmff_atom_offsets[a];
        cursors[a] = {off.bond_start, off.angle_start, off.torsion_start, off.oop_start,
                  off.stretchbend_start};
    }
    for (std::uint32_t t = 0; t < bonds.size(); ++t) {
        result.mmff_interactions[cursors[bonds[t].atom_i].bond++] = {t, mmff94_bond_k, atom_role_first_k};
        result.mmff_interactions[cursors[bonds[t].atom_j].bond++] = {t, mmff94_bond_k, atom_role_second_k};
    }
    for (std::uint32_t t = 0; t < ang_v.size(); ++t) {
        result.mmff_interactions[cursors[ang_v[t].atom_i].angle++] = {t, mmff94_angle_end_k, atom_role_first_k};
        result.mmff_interactions[cursors[ang_v[t].atom_j].angle++] = {t, mmff94_angle_center_k, atom_role_second_k};
        result.mmff_interactions[cursors[ang_v[t].atom_k].angle++] = {t, mmff94_angle_end_k, atom_role_third_k};
    }
    for (std::uint32_t t = 0; t < tor_v.size(); ++t) {
        result.mmff_interactions[cursors[tor_v[t].atom_i].torsion++] = {t, mmff94_torsion_k, atom_role_first_k};
        result.mmff_interactions[cursors[tor_v[t].atom_j].torsion++] = {t, mmff94_torsion_k, atom_role_second_k};
        result.mmff_interactions[cursors[tor_v[t].atom_k].torsion++] = {t, mmff94_torsion_k, atom_role_third_k};
        result.mmff_interactions[cursors[tor_v[t].atom_l].torsion++] = {t, mmff94_torsion_k, atom_role_fourth_k};
    }
    for (std::uint32_t t = 0; t < oop_v.size(); ++t) {
        result.mmff_interactions[cursors[oop_v[t].atom_i].oop++] = {t, mmff94_out_of_plane_k, atom_role_first_k};
        result.mmff_interactions[cursors[oop_v[t].atom_j].oop++] = {t, mmff94_out_of_plane_k, atom_role_second_k};
        result.mmff_interactions[cursors[oop_v[t].atom_k].oop++] = {t, mmff94_out_of_plane_k, atom_role_third_k};
        result.mmff_interactions[cursors[oop_v[t].atom_l].oop++] = {t, mmff94_out_of_plane_k, atom_role_fourth_k};
    }
    for (std::uint32_t t = 0; t < sb_v.size(); ++t) {
        result.mmff_interactions[cursors[sb_v[t].atom_i].stretchbend++] = {t, mmff94_stretch_bend_k, atom_role_first_k};
        result.mmff_interactions[cursors[sb_v[t].atom_j].stretchbend++] = {t, mmff94_stretch_bend_k, atom_role_second_k};
        result.mmff_interactions[cursors[sb_v[t].atom_k].stretchbend++] = {t, mmff94_stretch_bend_k, atom_role_third_k};
    }
}

/** Fully native end-to-end MMFF extraction: atom typing + charges +
 *  all terms + CSR.  Zero RDKit calls at runtime.  Returns true on
 *  successful typing; false means one or more atoms could not be typed
 *  and the caller should route the molecule through the RDKit fallback. */
inline bool extract_mmff94_params_fully_native(molecule_descriptor_t const &desc,
                                               mmff94_tables_view_t const &tables,
                                               etkdg_prepared_molecule_t &result) {
    std::uint32_t const n = desc.num_atoms;
    std::array<std::uint8_t, max_atoms_k> atom_types_buf {};
    std::array<float, max_atoms_k> formal_charges_buf {};
    std::array<float, max_atoms_k> partial_charges_buf {};
    std::span<std::uint8_t> atom_types {atom_types_buf.data(), n};
    std::span<float> formal_charges {formal_charges_buf.data(), n};
    std::span<float> partial_charges {partial_charges_buf.data(), n};

    if (!native_compute_mmff_atom_types(desc, atom_types)) return false;
    native_compute_mmff_formal_charges(desc, atom_types, formal_charges);
    native_compute_mmff_partial_charges(desc, tables, atom_types, formal_charges, partial_charges);

    extract_mmff94_params_native(desc, atom_types, partial_charges, tables, result);
    return true;
}
#endif // SMALLFIELD_MMFF_CPU_HAS_PREP

} // namespace smallfield
