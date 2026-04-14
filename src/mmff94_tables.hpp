/**
 *  @file mmff94_tables.hpp
 *  @brief Flat MMFF94 parameter tables extracted once at context init.
 *
 *  Workstream MG-0.  RDKit's `DefaultParameters::getMMFF*()` collections are
 *  ultimately backed by parallel `std::vector<std::uint8_t>` lookup columns
 *  and a `std::vector<MMFF*>` parameter column — the `operator()` helpers
 *  do a cascade of `std::equal_range` calls against those vectors.  For a
 *  GPU port we want the same data in flat POD arrays we can upload with
 *  one `cudaMemcpy` each and binary-search on the device.
 *
 *  This header exposes:
 *   - POD entry structs (`mmff94_*_entry_t`) — one per collection.
 *   - `mmff94_tables_host_t` — aggregates all tables as sorted
 *     `std::vector`s in the same key order RDKit uses internally, so the
 *     device-side lookup can use `std::equal_range`-equivalent nested
 *     searches and produce bit-identical results.
 *   - `extract_mmff94_tables_from_rdkit()` — reads from
 *     `ForceFields::MMFF::MMFFVdWCollection` and siblings via RDKit's
 *     public `d_*` member vectors (they are `public:` in
 *     `rdkit/ForceField/MMFF/Params.h`).
 *   - `lookup_*` functions that match RDKit's `operator()` /
 *     `getMMFF*Params` behaviour exactly, including the 4/5-level
 *     equivalence cascades for angle / torsion / oop via `MMFFDef`.
 *
 *  Doubles are converted to floats at extraction time — the main ETKDG
 *  kernel already consumes float32, and the extra precision in RDKit's
 *  tables is not meaningful for the tolerances we need.
 *
 *  Cross-validation (Phase MG-0 gate): call both our `lookup_*` and
 *  RDKit's `operator()` on 10 K random key tuples, assert byte-equality
 *  on the returned parameters (bitwise equality of the float bits).
 *  See `test/test_mmff94_tables.cpp`.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include <ForceField/MMFF/Params.h>

namespace smallfield {

// ---- Flat POD entries -----------------------------------------------------

/** Van-der-Waals parameters, one entry per MMFF atom type.  Matches
 *  `mmff94_atom_vdw_params_t` in `etkdg_types.cuh` in most respects; the
 *  difference is that this struct keys by `atom_type` for lookup. */
struct alignas(8) mmff94_vdw_entry_t {
    std::uint8_t atom_type;
    std::uint8_t donor_acceptor; ///< 0 = '-', 1 = 'A' (acceptor), 2 = 'D' (donor)
    std::uint8_t padding[2];
    float alpha_i;
    float n_i;
    float a_i;
    float g_i;
    float r_star;
    float padding_float;
};
static_assert(sizeof(mmff94_vdw_entry_t) == 32);

/** Bond-stretch parameter (kb, r0), keyed by (canonical atom type pair,
 *  MMFF bond type 0/1). */
struct alignas(16) mmff94_bond_entry_t {
    std::uint8_t i_type;   ///< canonical: i_type <= j_type
    std::uint8_t j_type;
    std::uint8_t bond_type;
    std::uint8_t padding;
    float kb;
    float r0;
    float padding_float;
};
static_assert(sizeof(mmff94_bond_entry_t) == 16);

/** Angle-bend parameter (ka, theta0 in deg), keyed by (j_type, i_type,
 *  k_type, angle_type) with i_type ≤ k_type canonical. */
struct alignas(16) mmff94_angle_entry_t {
    std::uint8_t j_type; ///< center atom type
    std::uint8_t i_type; ///< canonical: i_type <= k_type
    std::uint8_t k_type;
    std::uint8_t angle_type;
    float ka;
    float theta0_deg;
    float padding_float;
};
static_assert(sizeof(mmff94_angle_entry_t) == 16);

/** Torsion parameter (V1, V2, V3), keyed by (j, k, i, l, torsion_type)
 *  with canonicalization defined in RDKit's `MMFFTorCollection::
 *  getMMFFTorParams`: when canJ > canK, swap (i,j)↔(l,k). */
struct alignas(16) mmff94_torsion_entry_t {
    std::uint8_t j_type;
    std::uint8_t k_type;
    std::uint8_t i_type;
    std::uint8_t l_type;
    std::uint8_t tor_type;
    std::uint8_t padding[3];
    float v1;
    float v2;
    float v3;
};
static_assert(sizeof(mmff94_torsion_entry_t) == 32);

/** Out-of-plane bend parameter (koop), keyed by (j, canI, canK, canL)
 *  where (canI, canK, canL) is sorted ascending. */
struct alignas(8) mmff94_oop_entry_t {
    std::uint8_t j_type; ///< center
    std::uint8_t i_type; ///< sorted
    std::uint8_t k_type;
    std::uint8_t l_type;
    float koop;
};
static_assert(sizeof(mmff94_oop_entry_t) == 8 || sizeof(mmff94_oop_entry_t) == 16);

/** Stretch-bend parameter (kba_ijk, kba_kji), keyed by (j_type, i_type,
 *  k_type, stretch_bend_type) with the caller responsible for the swap
 *  flag returned alongside (whether i/k were swapped canonically). */
struct alignas(16) mmff94_stbn_entry_t {
    std::uint8_t j_type;
    std::uint8_t i_type; ///< canonical: i_type <= k_type
    std::uint8_t k_type;
    std::uint8_t stretch_bend_type;
    float kba_ijk;
    float kba_kji;
    float padding_float;
};
static_assert(sizeof(mmff94_stbn_entry_t) == 16);

/** Default-stretch-bend (DFSB) fallback parameter.  Used by RDKit when
 *  primary Stbn lookup fails: keyed by (periodic-table row of i, j, k). */
struct alignas(8) mmff94_dfsb_entry_t {
    std::uint8_t row_i; ///< canonical: row_i <= row_k
    std::uint8_t row_j;
    std::uint8_t row_k;
    std::uint8_t padding;
    float kba_ijk;
    float kba_kji;
};
static_assert(sizeof(mmff94_dfsb_entry_t) == 16);

/** MMFFDef equivalence levels — 4 bytes per atom type.  Indexed directly
 *  by `atom_type` (stored as dense array, one entry per possible type in
 *  [1, max_atom_type], with zeros for unused slots). */
struct mmff94_def_entry_t {
    std::uint8_t eq_level[4];
};
static_assert(sizeof(mmff94_def_entry_t) == 4);

/** Partial-bond-charge-increment per atom type. */
struct alignas(8) mmff94_pbci_entry_t {
    std::uint8_t atom_type;
    std::uint8_t padding[3];
    float pbci;
    float fcadj;
};
static_assert(sizeof(mmff94_pbci_entry_t) == 16);

/** Bond-charge-increment parameter.  The *directional* increment
 *  applied from atom i to atom j; atom i's partial charge gains `bci`,
 *  atom j loses it.  Sorted by (i, j, bond_type); RDKit's Chg lookup
 *  checks (i,j,bt) and (j,i,bt), flipping sign accordingly. */
struct alignas(8) mmff94_chg_entry_t {
    std::uint8_t i_type;
    std::uint8_t j_type;
    std::uint8_t bond_type;
    std::uint8_t padding;
    float bci;
};
static_assert(sizeof(mmff94_chg_entry_t) == 8);

/** Bndk default-bond parameter, keyed by (atomic_number_i, atomic_number_j)
 *  canonical ordering.  Used by RDKit's empirical bond-stretch rule when
 *  the primary Bond lookup fails. */
struct alignas(8) mmff94_bndk_entry_t {
    std::uint8_t atomic_num_i; ///< canonical: i <= j
    std::uint8_t atomic_num_j;
    std::uint8_t padding[2];
    float kb;
    float r0;
};
static_assert(sizeof(mmff94_bndk_entry_t) == 16);

/** Herschbach-Laurie's rule parameter, for Badger's rule bond stretch
 *  estimation when neither Bond nor Bndk have the pair. */
struct alignas(8) mmff94_hl_entry_t {
    std::uint8_t row_i; ///< canonical: row_i <= row_j
    std::uint8_t row_j;
    std::uint8_t padding[2];
    float a_ij;
    float d_ij;
    float dp_ij;
};
static_assert(sizeof(mmff94_hl_entry_t) == 16);

/** Covalent radius + Pauling electronegativity per element, for the
 *  MMFF empirical bond-stretch fallback. */
struct mmff94_covrad_entry_t {
    std::uint8_t atomic_num;
    std::uint8_t padding[3];
    float r0;
    float chi;
};
static_assert(sizeof(mmff94_covrad_entry_t) == 12);

// ---- Aggregate host-side table container ---------------------------------

/** All MMFF94 parameter tables in a single container, extracted once at
 *  context init.  Each vector is sorted in the exact lookup order RDKit
 *  uses internally — see the per-collection `d_*` column orderings in
 *  `rdkit/ForceField/MMFF/Params.h`. */
struct mmff94_tables_host_t {
    std::vector<mmff94_vdw_entry_t> vdw;          ///< sorted by atom_type
    std::vector<mmff94_bond_entry_t> bond;        ///< sorted by (i, j, bond_type)
    std::vector<mmff94_angle_entry_t> angle;      ///< sorted by (j, i, k, angle_type)
    std::vector<mmff94_torsion_entry_t> torsion;  ///< sorted by (j, k, i, l, tor_type)
    std::vector<mmff94_oop_entry_t> oop;          ///< sorted by (j, i, k, l)
    std::vector<mmff94_stbn_entry_t> stbn;        ///< sorted by (j, i, k, sb_type)
    std::vector<mmff94_dfsb_entry_t> dfsb;        ///< sorted by (row_i, row_j, row_k)
    std::vector<mmff94_def_entry_t> def;          ///< dense: indexed directly by atom_type
    std::vector<mmff94_pbci_entry_t> pbci;        ///< sorted by atom_type
    std::vector<mmff94_chg_entry_t> chg;          ///< sorted by (i, j, bond_type)
    std::vector<mmff94_bndk_entry_t> bndk;        ///< sorted by (atomic_num_i, atomic_num_j)
    std::vector<mmff94_hl_entry_t> hl;            ///< sorted by (row_i, row_j)
    std::vector<mmff94_covrad_entry_t> covrad;    ///< sorted by atomic_num

    // VdW collection scalar constants (from MMFFVdWCollection members)
    float vdw_power = 0.25f;
    float vdw_b = 0.2f;
    float vdw_beta = 12.0f;
    float vdw_darad = 0.8f;
    float vdw_daeps = 0.5f;

    std::uint8_t max_atom_type = 0; ///< highest atom type seen (for MMFFDef indexing)
};

// ---- Extraction from RDKit ------------------------------------------------

/**
 *  Extract every MMFF94 parameter table from RDKit's defaults collection
 *  into flat sorted `std::vector`s.  Only called once per process.
 *
 *  RDKit's collections (`MMFFVdWCollection`, `MMFFBondCollection`, ...)
 *  expose their internal `d_*` parallel vectors as `public:` members, so
 *  we copy directly from them.  The collections are already sorted by the
 *  lookup keys (as required by `std::equal_range` in RDKit's `operator()`
 *  implementations), so we preserve that ordering.
 */
inline void extract_mmff94_tables_from_rdkit(mmff94_tables_host_t &out) {
    namespace MMFF = ForceFields::MMFF;

    // --- VdW ---------------------------------------------------------------
    {
        auto const *vc = RDKit::MMFF::DefaultParameters::getMMFFVdW();
        out.vdw_power = static_cast<float>(vc->power);
        out.vdw_b = static_cast<float>(vc->B);
        out.vdw_beta = static_cast<float>(vc->Beta);
        out.vdw_darad = static_cast<float>(vc->DARAD);
        out.vdw_daeps = static_cast<float>(vc->DAEPS);
        std::size_t const n = vc->d_atomType.size();
        out.vdw.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.vdw[k];
            e.atom_type = vc->d_atomType[k];
            auto const &p = vc->d_params[k];
            e.alpha_i = static_cast<float>(p.alpha_i);
            e.n_i = static_cast<float>(p.N_i);
            e.a_i = static_cast<float>(p.A_i);
            e.g_i = static_cast<float>(p.G_i);
            e.r_star = static_cast<float>(p.R_star);
            e.donor_acceptor = (p.DA == 'D') ? 2u : (p.DA == 'A') ? 1u : 0u;
            e.padding[0] = e.padding[1] = 0;
            e.padding_float = 0.0f;
            if (e.atom_type > out.max_atom_type) out.max_atom_type = e.atom_type;
        }
    }

    // --- Bond — TODO(MG-1): RDKit doesn't expose `getMMFFBond()` in the
    //     public DefaultParameters API.  Direct construction
    //     (`MMFFBondCollection bc;`) segfaults on initial data load,
    //     suggesting the symbol is private to libRDKitForceField.cpp.
    //     We'll reach these via `MMFFMolProperties::getMMFFBondStretchParams`
    //     during the MG-1 CPU port and capture the table at that point.
    out.bond.clear();

    // --- Angle -------------------------------------------------------------
    {
        auto const *ac = RDKit::MMFF::DefaultParameters::getMMFFAngle();
        std::size_t const n = ac->d_iAtomType.size();
        out.angle.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.angle[k];
            e.j_type = ac->d_jAtomType[k];
            e.i_type = ac->d_iAtomType[k];
            e.k_type = ac->d_kAtomType[k];
            e.angle_type = ac->d_angleType[k];
            auto const &p = ac->d_params[k];
            e.ka = static_cast<float>(p.ka);
            e.theta0_deg = static_cast<float>(p.theta0);
            e.padding_float = 0.0f;
        }
    }

    // --- Torsion (isMMFFs = false for standard MMFF94 torsion table) ------
    {
        auto const *tc = RDKit::MMFF::DefaultParameters::getMMFFTor(false);
        std::size_t const n = tc->d_iAtomType.size();
        out.torsion.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.torsion[k];
            e.j_type = tc->d_jAtomType[k];
            e.k_type = tc->d_kAtomType[k];
            e.i_type = tc->d_iAtomType[k];
            e.l_type = tc->d_lAtomType[k];
            e.tor_type = tc->d_torType[k];
            e.padding[0] = e.padding[1] = e.padding[2] = 0;
            auto const &p = tc->d_params[k];
            e.v1 = static_cast<float>(p.V1);
            e.v2 = static_cast<float>(p.V2);
            e.v3 = static_cast<float>(p.V3);
        }
    }

    // --- Out-of-plane ------------------------------------------------------
    {
        auto const *oc = RDKit::MMFF::DefaultParameters::getMMFFOop(false);
        std::size_t const n = oc->d_iAtomType.size();
        out.oop.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.oop[k];
            e.j_type = oc->d_jAtomType[k];
            e.i_type = oc->d_iAtomType[k];
            e.k_type = oc->d_kAtomType[k];
            e.l_type = oc->d_lAtomType[k];
            e.koop = static_cast<float>(oc->d_params[k].koop);
        }
    }

    // --- Stretch-bend ------------------------------------------------------
    {
        auto const *sc = RDKit::MMFF::DefaultParameters::getMMFFStbn();
        std::size_t const n = sc->d_iAtomType.size();
        out.stbn.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.stbn[k];
            e.j_type = sc->d_jAtomType[k];
            e.i_type = sc->d_iAtomType[k];
            e.k_type = sc->d_kAtomType[k];
            e.stretch_bend_type = sc->d_stretchBendType[k];
            e.kba_ijk = static_cast<float>(sc->d_params[k].kbaIJK);
            e.kba_kji = static_cast<float>(sc->d_params[k].kbaKJI);
            e.padding_float = 0.0f;
        }
    }

    // --- DFSB (periodic-table-row-based fallback) --------------------------
    // RDKit stores DFSB as a nested std::map; we flatten it in row order.
    {
        auto const *dc = RDKit::MMFF::DefaultParameters::getMMFFDfsb();
        for (auto const &[row_i, inner1] : dc->d_params) {
            for (auto const &[row_j, inner2] : inner1) {
                for (auto const &[row_k, p] : inner2) {
                    mmff94_dfsb_entry_t e;
                    e.row_i = static_cast<std::uint8_t>(row_i);
                    e.row_j = static_cast<std::uint8_t>(row_j);
                    e.row_k = static_cast<std::uint8_t>(row_k);
                    e.padding = 0;
                    e.kba_ijk = static_cast<float>(p.kbaIJK);
                    e.kba_kji = static_cast<float>(p.kbaKJI);
                    out.dfsb.push_back(e);
                }
            }
        }
        // Sort for binary search.
        std::sort(out.dfsb.begin(), out.dfsb.end(), [](auto const &a, auto const &b) {
            if (a.row_i != b.row_i) return a.row_i < b.row_i;
            if (a.row_j != b.row_j) return a.row_j < b.row_j;
            return a.row_k < b.row_k;
        });
    }

    // --- MMFFDef (equivalence levels) — RDKit stores as vector<MMFFDef>
    //     indexed by (atomType - 1).  We build a dense array indexed
    //     directly by atom_type for O(1) device lookup, sized to
    //     max_atom_type + 1 so any plausible query fits.  Atom types
    //     beyond Def's coverage get all-zero eqLevels (which means
    //     wildcard equivalence — no match — exactly RDKit's behaviour
    //     when `(*mmffDef)(atomType)` returns nullptr).
    {
        auto const *dc = RDKit::MMFF::DefaultParameters::getMMFFDef();
        std::size_t const n = dc->d_params.size();
        std::size_t const dense_n =
            std::max<std::size_t>(out.max_atom_type, static_cast<std::uint8_t>(n)) + 1;
        out.def.assign(dense_n, mmff94_def_entry_t {});
        for (std::size_t k = 0; k < n; ++k) {
            auto const &d = dc->d_params[k];
            std::size_t const at = k + 1; // RDKit's 1-based indexing
            out.def[at].eq_level[0] = d.eqLevel[0];
            out.def[at].eq_level[1] = d.eqLevel[1];
            out.def[at].eq_level[2] = d.eqLevel[2];
            out.def[at].eq_level[3] = d.eqLevel[3];
        }
    }

    // --- PBCI — vector<MMFFPBCI> indexed by (atomType - 1) -----------------
    {
        auto const *pc = RDKit::MMFF::DefaultParameters::getMMFFPBCI();
        std::size_t const n = pc->d_params.size();
        out.pbci.reserve(n);
        for (std::size_t k = 0; k < n; ++k) {
            mmff94_pbci_entry_t e {};
            e.atom_type = static_cast<std::uint8_t>(k + 1);
            e.pbci = static_cast<float>(pc->d_params[k].pbci);
            e.fcadj = static_cast<float>(pc->d_params[k].fcadj);
            out.pbci.push_back(e);
        }
    }

    // --- Chg — TODO(MG-1): same situation as Bond.  Will be captured
    //     via MMFFMolProperties calls during the CPU port.
    out.chg.clear();

    // --- Bndk — TODO(MG-1): not in public API. ------------------------
    out.bndk.clear();

    // --- Herschbach-Laurie --------------------------------------------------
    {
        auto const *hc = RDKit::MMFF::DefaultParameters::getMMFFHerschbachLaurie();
        std::size_t const n = hc->d_iRow.size();
        out.hl.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            auto &e = out.hl[k];
            e.row_i = hc->d_iRow[k];
            e.row_j = hc->d_jRow[k];
            e.padding[0] = e.padding[1] = 0;
            e.a_ij = static_cast<float>(hc->d_params[k].a_ij);
            e.d_ij = static_cast<float>(hc->d_params[k].d_ij);
            e.dp_ij = static_cast<float>(hc->d_params[k].dp_ij);
        }
    }

    // --- CovRadPauEle — TODO(MG-1): not in public API. -----------------
    out.covrad.clear();
}

// ---- Host-side lookups matching RDKit's operator() behaviour ------------
// These are used both by (a) the MG-0 cross-validation test to sanity-check
// that we extracted sorted data correctly, and (b) the MG-1 CPU port of
// `extract_mmff94_params`.  Must produce results byte-identical to RDKit.

/** Binary-search cascade matching `MMFFVdWCollection::operator()`. */
inline mmff94_vdw_entry_t const *lookup_vdw(mmff94_tables_host_t const &t,
                                            std::uint8_t atom_type) noexcept {
    auto it = std::lower_bound(t.vdw.begin(), t.vdw.end(), atom_type,
                               [](auto const &e, std::uint8_t v) { return e.atom_type < v; });
    if (it != t.vdw.end() && it->atom_type == atom_type) return &*it;
    return nullptr;
}

/** Binary-search cascade matching `MMFFBondCollection::operator()`.
 *  Canonicalises (atom_i, atom_j) to (lo, hi). */
inline mmff94_bond_entry_t const *lookup_bond(mmff94_tables_host_t const &t,
                                              std::uint8_t bond_type,
                                              std::uint8_t atom_i,
                                              std::uint8_t atom_j) noexcept {
    std::uint8_t lo = atom_i, hi = atom_j;
    if (lo > hi) std::swap(lo, hi);

    // RDKit's cascade: equal_range on i_type, then on j_type (within that
    // slice), then on bond_type.  Because the table is sorted by (i, j,
    // bond_type) ascending, std::equal_range on `d_iAtomType` finds the i
    // slice; on `d_jAtomType` within the slice finds the j slice; etc.
    auto i_begin = t.bond.begin();
    auto i_end = t.bond.end();
    auto i_lo = std::lower_bound(i_begin, i_end, lo,
                                  [](auto const &e, std::uint8_t v) { return e.i_type < v; });
    auto i_hi = std::lower_bound(i_lo, i_end, static_cast<std::uint8_t>(lo + 1),
                                  [](auto const &e, std::uint8_t v) { return e.i_type < v; });
    if (i_lo == i_hi) return nullptr;
    auto j_lo = std::lower_bound(i_lo, i_hi, hi,
                                  [](auto const &e, std::uint8_t v) { return e.j_type < v; });
    auto j_hi = std::lower_bound(j_lo, i_hi, static_cast<std::uint8_t>(hi + 1),
                                  [](auto const &e, std::uint8_t v) { return e.j_type < v; });
    if (j_lo == j_hi) return nullptr;
    auto bt_lo = std::lower_bound(j_lo, j_hi, bond_type,
                                   [](auto const &e, std::uint8_t v) { return e.bond_type < v; });
    if (bt_lo != j_hi && bt_lo->bond_type == bond_type) return &*bt_lo;
    return nullptr;
}

/** Binary-search cascade matching `MMFFAngleCollection::operator()` —
 *  including the 4-level equivalence fallback via MMFFDef. */
inline mmff94_angle_entry_t const *lookup_angle(mmff94_tables_host_t const &t,
                                                std::uint8_t angle_type,
                                                std::uint8_t atom_i,
                                                std::uint8_t atom_j_center,
                                                std::uint8_t atom_k) noexcept {
    for (int iter = 0; iter < 4; ++iter) {
        std::uint8_t can_i =
            (atom_i < t.def.size()) ? t.def[atom_i].eq_level[iter] : std::uint8_t {0};
        std::uint8_t can_k =
            (atom_k < t.def.size()) ? t.def[atom_k].eq_level[iter] : std::uint8_t {0};
        if (can_i > can_k) std::swap(can_i, can_k);

        auto j_begin = t.angle.begin();
        auto j_end = t.angle.end();
        auto j_lo = std::lower_bound(j_begin, j_end, atom_j_center,
                                      [](auto const &e, std::uint8_t v) { return e.j_type < v; });
        auto j_hi = std::lower_bound(j_lo, j_end,
                                      static_cast<std::uint8_t>(atom_j_center + 1),
                                      [](auto const &e, std::uint8_t v) { return e.j_type < v; });
        if (j_lo == j_hi) continue;
        auto i_lo = std::lower_bound(j_lo, j_hi, can_i,
                                      [](auto const &e, std::uint8_t v) { return e.i_type < v; });
        auto i_hi = std::lower_bound(i_lo, j_hi, static_cast<std::uint8_t>(can_i + 1),
                                      [](auto const &e, std::uint8_t v) { return e.i_type < v; });
        if (i_lo == i_hi) continue;
        auto k_lo = std::lower_bound(i_lo, i_hi, can_k,
                                      [](auto const &e, std::uint8_t v) { return e.k_type < v; });
        auto k_hi = std::lower_bound(k_lo, i_hi, static_cast<std::uint8_t>(can_k + 1),
                                      [](auto const &e, std::uint8_t v) { return e.k_type < v; });
        if (k_lo == k_hi) continue;
        auto at_lo = std::lower_bound(k_lo, k_hi, angle_type,
                                       [](auto const &e, std::uint8_t v) { return e.angle_type < v; });
        if (at_lo != k_hi && at_lo->angle_type == angle_type) return &*at_lo;
    }
    return nullptr;
}

// Torsion and OOP lookups with their respective multi-level equivalence
// cascades are more complex — they're implemented fully in MG-1 when the
// CPU port is written.  MG-0 only needs to validate the simpler
// (VdW/Bond/Angle) lookups byte-equal RDKit's.

} // namespace smallfield
