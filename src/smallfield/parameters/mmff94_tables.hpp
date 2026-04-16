/**
 *  @file mmff94_tables.hpp
 *  @brief Flat MMFF94 parameter tables extracted once at context init.
 *
 *  RDKit's `DefaultParameters::getMMFF*()` collections are
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
 *   - `lookup_*` functions that match RDKit's `operator()` /
 *     `getMMFF*Params` behaviour exactly, including the 4/5-level
 *     equivalence cascades for angle / torsion / oop via `MMFFDef`.
 *
 *  Doubles are converted to floats at extraction time — the main ETKDG
 *  kernel already consumes float32, and the extra precision in RDKit's
 *  tables is not meaningful for the tolerances we need.
 *
 *  Cross-validation gate: call both our `lookup_*` and RDKit's `operator()` on 10K random key
 *  tuples, and assert byte-equality on the returned parameters. See
 *  `test/parameters/test_mmff94_tables.cpp`.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

// `SMALLFIELD_HD_` resolves to `__host__ __device__` when compiled by nvcc,
// empty otherwise, so every `lookup_*` below is callable from both host
// C++ and device kernels with the same source.
#if defined(__CUDACC__)
#define SMALLFIELD_HD_ __host__ __device__
#else
#define SMALLFIELD_HD_
#endif

namespace smallfield::parameters {

// Flat POD entry structs for every MMFF94 parameter table.

/** Van-der-Waals parameters, one entry per MMFF atom type. Matches `mmff94_atom_vdw_params_t` in
 *  `types/etkdg_types.hpp` in most respects; the difference is that this struct keys by `atom_type`
 *  for lookup. */
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
    std::uint8_t i_type; ///< canonical: i_type <= j_type
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

/** MMFFProp — per-atom-type chemical properties.  Used by
 *  `getMMFFBondType` (sbmb / arom fields) and by the empirical
 *  angle-bending rule (crd / val / mltb / linh fields).  Dense array
 *  indexed directly by atom_type (one slot per possible type, zeros
 *  for unused slots), mirroring `mmff94_def_entry_t`. */
struct alignas(8) mmff94_prop_entry_t {
    std::uint8_t atomic_num;
    std::uint8_t crd;
    std::uint8_t val;
    std::uint8_t pilp;
    std::uint8_t mltb;
    std::uint8_t arom;
    std::uint8_t linh;
    std::uint8_t sbmb;
};
static_assert(sizeof(mmff94_prop_entry_t) == 8);

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

// Read-only view over the tables, shared between host and device.

/** @brief POD read-only span — typed pointer + length, ABI-safe across host and device.
 *
 *  Replaces `std::span<T const>` to avoid a libstdc++ ABI mismatch when passing
 *  spans by value to `__global__` kernels.
 */
template <typename element_>
struct mmff94_span_t {
    element_ const *ptr_ = nullptr;
    std::size_t size_ = 0;
    SMALLFIELD_HD_ element_ const *data() const noexcept { return ptr_; }
    SMALLFIELD_HD_ std::size_t size() const noexcept { return size_; }
    SMALLFIELD_HD_ bool empty() const noexcept { return size_ == 0; }
    SMALLFIELD_HD_ element_ const *begin() const noexcept { return ptr_; }
    SMALLFIELD_HD_ element_ const *end() const noexcept { return ptr_ + size_; }
    SMALLFIELD_HD_ element_ const &operator[](std::size_t i) const noexcept { return ptr_[i]; }
};

/** POD view over every extracted MMFF table.  Consumed by all `lookup_*`
 *  functions and by host/device alike — on host `ptr_` points into the
 *  `std::vector`s inside `mmff94_tables_host_t`; on device it points
 *  into a `cudaMalloc`ed arena built by `upload_mmff94_tables_to_device`
 *  (see `src/mmff94_tables_device.cuh`). */
struct mmff94_tables_view_t {
    mmff94_span_t<mmff94_vdw_entry_t> vdw;
    mmff94_span_t<mmff94_bond_entry_t> bond;
    mmff94_span_t<mmff94_angle_entry_t> angle;
    mmff94_span_t<mmff94_torsion_entry_t> torsion;
    mmff94_span_t<mmff94_oop_entry_t> oop;
    mmff94_span_t<mmff94_stbn_entry_t> stbn;
    mmff94_span_t<mmff94_dfsb_entry_t> dfsb;
    mmff94_span_t<mmff94_def_entry_t> def;
    mmff94_span_t<mmff94_prop_entry_t> prop;
    mmff94_span_t<mmff94_pbci_entry_t> pbci;
    mmff94_span_t<mmff94_chg_entry_t> chg;
    mmff94_span_t<mmff94_bndk_entry_t> bndk;
    mmff94_span_t<mmff94_hl_entry_t> hl;
    mmff94_span_t<mmff94_covrad_entry_t> covrad;

    float vdw_power = 0.25f;
    float vdw_b = 0.2f;
    float vdw_beta = 12.0f;
    float vdw_darad = 0.8f;
    float vdw_daeps = 0.5f;

    std::uint8_t max_atom_type = 0;
};

// Aggregate host-side table container.

/** All MMFF94 parameter tables in a single container, extracted once at
 *  context init.  Each vector is sorted in the exact lookup order RDKit
 *  uses internally — see the per-collection `d_*` column orderings in
 *  `rdkit/ForceField/MMFF/Params.h`. */
struct mmff94_tables_host_t {
    std::vector<mmff94_vdw_entry_t> vdw;         ///< sorted by atom_type
    std::vector<mmff94_bond_entry_t> bond;       ///< sorted by (i, j, bond_type)
    std::vector<mmff94_angle_entry_t> angle;     ///< sorted by (j, i, k, angle_type)
    std::vector<mmff94_torsion_entry_t> torsion; ///< sorted by (j, k, i, l, tor_type)
    std::vector<mmff94_oop_entry_t> oop;         ///< sorted by (j, i, k, l)
    std::vector<mmff94_stbn_entry_t> stbn;       ///< sorted by (j, i, k, sb_type)
    std::vector<mmff94_dfsb_entry_t> dfsb;       ///< sorted by (row_i, row_j, row_k)
    std::vector<mmff94_def_entry_t> def;         ///< dense: indexed directly by atom_type
    std::vector<mmff94_prop_entry_t> prop;       ///< dense: indexed directly by atom_type
    std::vector<mmff94_pbci_entry_t> pbci;       ///< sorted by atom_type
    std::vector<mmff94_chg_entry_t> chg;         ///< sorted by (i, j, bond_type)
    std::vector<mmff94_bndk_entry_t> bndk;       ///< sorted by (atomic_num_i, atomic_num_j)
    std::vector<mmff94_hl_entry_t> hl;           ///< sorted by (row_i, row_j)
    std::vector<mmff94_covrad_entry_t> covrad;   ///< sorted by atomic_num

    // VdW collection scalar constants (from MMFFVdWCollection members)
    float vdw_power = 0.25f;
    float vdw_b = 0.2f;
    float vdw_beta = 12.0f;
    float vdw_darad = 0.8f;
    float vdw_daeps = 0.5f;

    std::uint8_t max_atom_type = 0; ///< highest atom type seen (for MMFFDef indexing)

    /** Build a read-only view that references this container's vector
     *  storage.  Cheap — just copies pointers + sizes + scalar constants. */
    mmff94_tables_view_t view() const noexcept {
        mmff94_tables_view_t v;
        v.vdw = {vdw.data(), vdw.size()};
        v.bond = {bond.data(), bond.size()};
        v.angle = {angle.data(), angle.size()};
        v.torsion = {torsion.data(), torsion.size()};
        v.oop = {oop.data(), oop.size()};
        v.stbn = {stbn.data(), stbn.size()};
        v.dfsb = {dfsb.data(), dfsb.size()};
        v.def = {def.data(), def.size()};
        v.prop = {prop.data(), prop.size()};
        v.pbci = {pbci.data(), pbci.size()};
        v.chg = {chg.data(), chg.size()};
        v.bndk = {bndk.data(), bndk.size()};
        v.hl = {hl.data(), hl.size()};
        v.covrad = {covrad.data(), covrad.size()};
        v.vdw_power = vdw_power;
        v.vdw_b = vdw_b;
        v.vdw_beta = vdw_beta;
        v.vdw_darad = vdw_darad;
        v.vdw_daeps = vdw_daeps;
        v.max_atom_type = max_atom_type;
        return v;
    }
};

// Hand-rolled lower_bound. `std::lower_bound` from libstdc++ compiles on device with
// `--expt-relaxed-constexpr` but has been seen to stop the binary search
// early on device — in practice returning the first few elements only
// and declaring the key "not found" for anything past ~8 entries in a
// 95-element sorted array. Rolling our own fixes the issue and keeps
// the code path identical between host and device.
template <typename element_, typename key_accessor_>
SMALLFIELD_HD_ inline std::size_t lower_bound_u8_(element_ const *base, std::size_t n, std::uint8_t key,
                                                  key_accessor_ &&kf) noexcept {
    std::size_t lo = 0, hi = n;
    while (lo < hi) {
        std::size_t const mid = lo + (hi - lo) / 2;
        if (kf(base[mid]) < key) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Host-side lookups that match RDKit's operator() behaviour. These serve two callers: the
// cross-validation test sanity-checks that we extracted sorted data correctly, and the native CPU port
// of `extract_mmff94_params`. Must produce results byte-identical to RDKit.

/** Binary-search cascade matching `MMFFVdWCollection::operator()`. */
SMALLFIELD_HD_ inline mmff94_vdw_entry_t const *lookup_vdw(mmff94_tables_view_t const &t,
                                                           std::uint8_t atom_type) noexcept {
    std::size_t const idx = lower_bound_u8_(t.vdw.data(), t.vdw.size(), atom_type,
                                            [](mmff94_vdw_entry_t const &e) { return e.atom_type; });
    if (idx < t.vdw.size() && t.vdw[idx].atom_type == atom_type) return &t.vdw[idx];
    return nullptr;
}

/** Binary-search cascade matching `MMFFBondCollection::operator()`.
 *  Canonicalises (atom_i, atom_j) to (lo, hi). */
SMALLFIELD_HD_ inline mmff94_bond_entry_t const *lookup_bond(mmff94_tables_view_t const &t, std::uint8_t bond_type,
                                                             std::uint8_t atom_i, std::uint8_t atom_j) noexcept {
    std::uint8_t lo = atom_i, hi = atom_j;
    if (lo > hi) {
        std::uint8_t tmp = lo;
        lo = hi;
        hi = tmp;
    }

    // Cascade on i_type → j_type → bond_type.
    mmff94_bond_entry_t const *b0 = t.bond.data();
    std::size_t const n = t.bond.size();
    auto i_key = [](mmff94_bond_entry_t const &e) { return e.i_type; };
    auto j_key = [](mmff94_bond_entry_t const &e) { return e.j_type; };
    auto bond_type_key = [](mmff94_bond_entry_t const &e) { return e.bond_type; };
    std::size_t const i_lo = lower_bound_u8_(b0, n, lo, i_key);
    std::size_t const i_hi = lower_bound_u8_(b0, n, (std::uint8_t)(lo + 1), i_key);
    if (i_lo == i_hi) return nullptr;
    std::size_t const j_span = i_hi - i_lo;
    std::size_t const j_lo = i_lo + lower_bound_u8_(b0 + i_lo, j_span, hi, j_key);
    std::size_t const j_hi = i_lo + lower_bound_u8_(b0 + i_lo, j_span, (std::uint8_t)(hi + 1), j_key);
    if (j_lo == j_hi) return nullptr;
    std::size_t const bond_type_span = j_hi - j_lo;
    std::size_t const bond_type_low = j_lo + lower_bound_u8_(b0 + j_lo, bond_type_span, bond_type, bond_type_key);
    if (bond_type_low != j_hi && b0[bond_type_low].bond_type == bond_type) return b0 + bond_type_low;
    return nullptr;
}

/** Binary-search cascade matching `MMFFAngleCollection::operator()` —
 *  including the 4-level equivalence fallback via MMFFDef. */
SMALLFIELD_HD_ inline mmff94_angle_entry_t const *lookup_angle(mmff94_tables_view_t const &t, std::uint8_t angle_type,
                                                               std::uint8_t atom_i, std::uint8_t atom_j_center,
                                                               std::uint8_t atom_k) noexcept {
    mmff94_angle_entry_t const *a0 = t.angle.data();
    std::size_t const n = t.angle.size();
    auto j_key = [](mmff94_angle_entry_t const &e) { return e.j_type; };
    auto i_key = [](mmff94_angle_entry_t const &e) { return e.i_type; };
    auto k_key = [](mmff94_angle_entry_t const &e) { return e.k_type; };
    auto at_key = [](mmff94_angle_entry_t const &e) { return e.angle_type; };
    for (int iter = 0; iter < 4; ++iter) {
        std::uint8_t can_i = (atom_i < t.def.size()) ? t.def[atom_i].eq_level[iter] : std::uint8_t {0};
        std::uint8_t can_k = (atom_k < t.def.size()) ? t.def[atom_k].eq_level[iter] : std::uint8_t {0};
        if (can_i > can_k) {
            std::uint8_t tmp = can_i;
            can_i = can_k;
            can_k = tmp;
        }

        std::size_t const j_lo = lower_bound_u8_(a0, n, atom_j_center, j_key);
        std::size_t const j_hi = lower_bound_u8_(a0, n, (std::uint8_t)(atom_j_center + 1), j_key);
        if (j_lo == j_hi) continue;
        std::size_t const i_lo = j_lo + lower_bound_u8_(a0 + j_lo, j_hi - j_lo, can_i, i_key);
        std::size_t const i_hi = j_lo + lower_bound_u8_(a0 + j_lo, j_hi - j_lo, (std::uint8_t)(can_i + 1), i_key);
        if (i_lo == i_hi) continue;
        std::size_t const k_lo = i_lo + lower_bound_u8_(a0 + i_lo, i_hi - i_lo, can_k, k_key);
        std::size_t const k_hi = i_lo + lower_bound_u8_(a0 + i_lo, i_hi - i_lo, (std::uint8_t)(can_k + 1), k_key);
        if (k_lo == k_hi) continue;
        std::size_t const at_lo = k_lo + lower_bound_u8_(a0 + k_lo, k_hi - k_lo, angle_type, at_key);
        if (at_lo != k_hi && a0[at_lo].angle_type == angle_type) return a0 + at_lo;
    }
    return nullptr;
}

/** MMFFProp dense lookup — `nullptr` for unknown atom types. */
SMALLFIELD_HD_ inline mmff94_prop_entry_t const *lookup_prop(mmff94_tables_view_t const &t,
                                                             std::uint8_t atom_type) noexcept {
    if (atom_type == 0 || atom_type >= t.prop.size()) return nullptr;
    return &t.prop[atom_type];
}

/** PBCI (partial-bond-charge-increment) lookup by atom type. */
SMALLFIELD_HD_ inline mmff94_pbci_entry_t const *lookup_pbci(mmff94_tables_view_t const &t,
                                                             std::uint8_t atom_type) noexcept {
    std::size_t const idx = lower_bound_u8_(t.pbci.data(), t.pbci.size(), atom_type,
                                            [](mmff94_pbci_entry_t const &e) { return e.atom_type; });
    if (idx < t.pbci.size() && t.pbci[idx].atom_type == atom_type) return &t.pbci[idx];
    return nullptr;
}

/** Signed bond-charge-increment lookup.  Mirrors
 *  `MMFFChgCollection::getMMFFChgParams`: returns (sign, bci) where sign
 *  is -1 if (i_type <= j_type) held in the tabulation order, +1 if the
 *  arguments had to be swapped.  Returns {0, nullptr} on miss. */
SMALLFIELD_HD_ inline std::pair<int, mmff94_chg_entry_t const *> lookup_chg(mmff94_tables_view_t const &t,
                                                                            std::uint8_t bond_type, std::uint8_t atom_i,
                                                                            std::uint8_t atom_j) noexcept {
    int sign = -1;
    std::uint8_t lo = atom_i, hi = atom_j;
    if (atom_i > atom_j) {
        lo = atom_j;
        hi = atom_i;
        sign = 1;
    }
    mmff94_chg_entry_t const *c0 = t.chg.data();
    std::size_t const n = t.chg.size();
    auto i_key = [](mmff94_chg_entry_t const &e) { return e.i_type; };
    auto j_key = [](mmff94_chg_entry_t const &e) { return e.j_type; };
    auto bond_type_key = [](mmff94_chg_entry_t const &e) { return e.bond_type; };
    std::size_t const i_lo = lower_bound_u8_(c0, n, lo, i_key);
    std::size_t const i_hi = lower_bound_u8_(c0, n, (std::uint8_t)(lo + 1), i_key);
    if (i_lo == i_hi) return {sign, nullptr};
    std::size_t const j_lo = i_lo + lower_bound_u8_(c0 + i_lo, i_hi - i_lo, hi, j_key);
    std::size_t const j_hi = i_lo + lower_bound_u8_(c0 + i_lo, i_hi - i_lo, (std::uint8_t)(hi + 1), j_key);
    if (j_lo == j_hi) return {sign, nullptr};
    std::size_t const bond_type_low = j_lo + lower_bound_u8_(c0 + j_lo, j_hi - j_lo, bond_type, bond_type_key);
    if (bond_type_low != j_hi && c0[bond_type_low].bond_type == bond_type) return {sign, c0 + bond_type_low};
    return {sign, nullptr};
}

/** Bndk (default bond by atomic-number pair) lookup. */
SMALLFIELD_HD_ inline mmff94_bndk_entry_t const *lookup_bndk(mmff94_tables_view_t const &t, std::uint8_t atomic_i,
                                                             std::uint8_t atomic_j) noexcept {
    std::uint8_t lo = atomic_i, hi = atomic_j;
    if (lo > hi) {
        std::uint8_t tmp = lo;
        lo = hi;
        hi = tmp;
    }
    mmff94_bndk_entry_t const *b0 = t.bndk.data();
    std::size_t const n = t.bndk.size();
    auto i_key = [](mmff94_bndk_entry_t const &e) { return e.atomic_num_i; };
    auto j_key = [](mmff94_bndk_entry_t const &e) { return e.atomic_num_j; };
    std::size_t const i_lo = lower_bound_u8_(b0, n, lo, i_key);
    std::size_t const i_hi = lower_bound_u8_(b0, n, (std::uint8_t)(lo + 1), i_key);
    if (i_lo == i_hi) return nullptr;
    std::size_t const j_lo = i_lo + lower_bound_u8_(b0 + i_lo, i_hi - i_lo, hi, j_key);
    if (j_lo != i_hi && b0[j_lo].atomic_num_j == hi) return b0 + j_lo;
    return nullptr;
}

/** Covalent radius + Pauling electronegativity lookup by atomic number. */
SMALLFIELD_HD_ inline mmff94_covrad_entry_t const *lookup_covrad(mmff94_tables_view_t const &t,
                                                                 std::uint8_t atomic_num) noexcept {
    std::size_t const idx = lower_bound_u8_(t.covrad.data(), t.covrad.size(), atomic_num,
                                            [](mmff94_covrad_entry_t const &e) { return e.atomic_num; });
    if (idx < t.covrad.size() && t.covrad[idx].atomic_num == atomic_num) return &t.covrad[idx];
    return nullptr;
}

/** Herschbach-Laurie lookup by periodic-table-row pair. */
SMALLFIELD_HD_ inline mmff94_hl_entry_t const *lookup_hl(mmff94_tables_view_t const &t, std::uint8_t row_i,
                                                         std::uint8_t row_j) noexcept {
    std::uint8_t lo = row_i, hi = row_j;
    if (lo > hi) {
        std::uint8_t tmp = lo;
        lo = hi;
        hi = tmp;
    }
    mmff94_hl_entry_t const *h0 = t.hl.data();
    std::size_t const n = t.hl.size();
    auto i_key = [](mmff94_hl_entry_t const &e) { return e.row_i; };
    auto j_key = [](mmff94_hl_entry_t const &e) { return e.row_j; };
    std::size_t const i_lo = lower_bound_u8_(h0, n, lo, i_key);
    std::size_t const i_hi = lower_bound_u8_(h0, n, (std::uint8_t)(lo + 1), i_key);
    if (i_lo == i_hi) return nullptr;
    std::size_t const j_lo = i_lo + lower_bound_u8_(h0 + i_lo, i_hi - i_lo, hi, j_key);
    if (j_lo != i_hi && h0[j_lo].row_j == hi) return h0 + j_lo;
    return nullptr;
}

/** OOP bend lookup with 4-stage equivalence cascade.  Matches
 *  `MMFFOopCollection::operator()` exactly. */
SMALLFIELD_HD_ inline mmff94_oop_entry_t const *lookup_oop(mmff94_tables_view_t const &t, std::uint8_t atom_i,
                                                           std::uint8_t atom_j, std::uint8_t atom_k,
                                                           std::uint8_t atom_l) noexcept {
    mmff94_oop_entry_t const *o0 = t.oop.data();
    std::size_t const n = t.oop.size();
    auto j_key = [](mmff94_oop_entry_t const &e) { return e.j_type; };
    auto i_key = [](mmff94_oop_entry_t const &e) { return e.i_type; };
    auto k_key = [](mmff94_oop_entry_t const &e) { return e.k_type; };
    auto l_key = [](mmff94_oop_entry_t const &e) { return e.l_type; };
    for (int iter = 0; iter < 4; ++iter) {
        std::uint8_t ikl[3] = {
            (atom_i < t.def.size()) ? t.def[atom_i].eq_level[iter] : std::uint8_t {0},
            (atom_k < t.def.size()) ? t.def[atom_k].eq_level[iter] : std::uint8_t {0},
            (atom_l < t.def.size()) ? t.def[atom_l].eq_level[iter] : std::uint8_t {0},
        };
        if (ikl[0] > ikl[1]) {
            std::uint8_t tmp = ikl[0];
            ikl[0] = ikl[1];
            ikl[1] = tmp;
        }
        if (ikl[1] > ikl[2]) {
            std::uint8_t tmp = ikl[1];
            ikl[1] = ikl[2];
            ikl[2] = tmp;
        }
        if (ikl[0] > ikl[1]) {
            std::uint8_t tmp = ikl[0];
            ikl[0] = ikl[1];
            ikl[1] = tmp;
        }
        std::size_t const j_lo = lower_bound_u8_(o0, n, atom_j, j_key);
        std::size_t const j_hi = lower_bound_u8_(o0, n, (std::uint8_t)(atom_j + 1), j_key);
        if (j_lo == j_hi) continue;
        std::size_t const i_lo = j_lo + lower_bound_u8_(o0 + j_lo, j_hi - j_lo, ikl[0], i_key);
        std::size_t const i_hi = j_lo + lower_bound_u8_(o0 + j_lo, j_hi - j_lo, (std::uint8_t)(ikl[0] + 1), i_key);
        if (i_lo == i_hi) continue;
        std::size_t const k_lo = i_lo + lower_bound_u8_(o0 + i_lo, i_hi - i_lo, ikl[1], k_key);
        std::size_t const k_hi = i_lo + lower_bound_u8_(o0 + i_lo, i_hi - i_lo, (std::uint8_t)(ikl[1] + 1), k_key);
        if (k_lo == k_hi) continue;
        std::size_t const l_lo = k_lo + lower_bound_u8_(o0 + k_lo, k_hi - k_lo, ikl[2], l_key);
        if (l_lo != k_hi && o0[l_lo].l_type == ikl[2]) return o0 + l_lo;
    }
    return nullptr;
}

/** Stretch-bend lookup with swap tracking.  No equivalence cascade (the
 *  STBN table is a single level).  Swap indicates whether kba_ijk and
 *  kba_kji should be exchanged — matches
 *  `MMFFStbnCollection::getMMFFStbnParams`. */
SMALLFIELD_HD_ inline std::pair<bool, mmff94_stbn_entry_t const *> lookup_stbn(
    mmff94_tables_view_t const &t, std::uint8_t stretch_bend_type, std::uint8_t bond_type_1, std::uint8_t bond_type_2,
    std::uint8_t atom_i, std::uint8_t atom_j_center, std::uint8_t atom_k) noexcept {
    bool swap = false;
    std::uint8_t lo = atom_i, hi = atom_k;
    if (atom_i > atom_k) {
        lo = atom_k;
        hi = atom_i;
        swap = true;
    }
    else if (atom_i == atom_k) swap = bond_type_1 < bond_type_2;

    mmff94_stbn_entry_t const *s0 = t.stbn.data();
    std::size_t const n = t.stbn.size();
    auto j_key = [](mmff94_stbn_entry_t const &e) { return e.j_type; };
    auto i_key = [](mmff94_stbn_entry_t const &e) { return e.i_type; };
    auto k_key = [](mmff94_stbn_entry_t const &e) { return e.k_type; };
    auto sb_key = [](mmff94_stbn_entry_t const &e) { return e.stretch_bend_type; };
    std::size_t const j_lo = lower_bound_u8_(s0, n, atom_j_center, j_key);
    std::size_t const j_hi = lower_bound_u8_(s0, n, (std::uint8_t)(atom_j_center + 1), j_key);
    if (j_lo == j_hi) return {swap, nullptr};
    std::size_t const i_lo = j_lo + lower_bound_u8_(s0 + j_lo, j_hi - j_lo, lo, i_key);
    std::size_t const i_hi = j_lo + lower_bound_u8_(s0 + j_lo, j_hi - j_lo, (std::uint8_t)(lo + 1), i_key);
    if (i_lo == i_hi) return {swap, nullptr};
    std::size_t const k_lo = i_lo + lower_bound_u8_(s0 + i_lo, i_hi - i_lo, hi, k_key);
    std::size_t const k_hi = i_lo + lower_bound_u8_(s0 + i_lo, i_hi - i_lo, (std::uint8_t)(hi + 1), k_key);
    if (k_lo == k_hi) return {swap, nullptr};
    std::size_t const stretch_bend_low = k_lo + lower_bound_u8_(s0 + k_lo, k_hi - k_lo, stretch_bend_type, sb_key);
    if (stretch_bend_low != k_hi && s0[stretch_bend_low].stretch_bend_type == stretch_bend_type)
        return {swap, s0 + stretch_bend_low};
    return {swap, nullptr};
}

/** DFSB fallback lookup by periodic-table-row triple, with swap. */
SMALLFIELD_HD_ inline std::pair<bool, mmff94_dfsb_entry_t const *> lookup_dfsb(mmff94_tables_view_t const &t,
                                                                               std::uint8_t row_i, std::uint8_t row_j,
                                                                               std::uint8_t row_k) noexcept {
    bool swap = false;
    std::uint8_t lo = row_i, hi = row_k;
    if (lo > hi) {
        std::uint8_t tmp = lo;
        lo = hi;
        hi = tmp;
        swap = true;
    }
    mmff94_dfsb_entry_t const *d0 = t.dfsb.data();
    std::size_t const n = t.dfsb.size();
    auto ri_key = [](mmff94_dfsb_entry_t const &e) { return e.row_i; };
    auto rj_key = [](mmff94_dfsb_entry_t const &e) { return e.row_j; };
    auto rk_key = [](mmff94_dfsb_entry_t const &e) { return e.row_k; };
    std::size_t const ri_lo = lower_bound_u8_(d0, n, lo, ri_key);
    std::size_t const ri_hi = lower_bound_u8_(d0, n, (std::uint8_t)(lo + 1), ri_key);
    if (ri_lo == ri_hi) return {swap, nullptr};
    std::size_t const rj_lo = ri_lo + lower_bound_u8_(d0 + ri_lo, ri_hi - ri_lo, row_j, rj_key);
    std::size_t const rj_hi = ri_lo + lower_bound_u8_(d0 + ri_lo, ri_hi - ri_lo, (std::uint8_t)(row_j + 1), rj_key);
    if (rj_lo == rj_hi) return {swap, nullptr};
    std::size_t const rk_lo = rj_lo + lower_bound_u8_(d0 + rj_lo, rj_hi - rj_lo, hi, rk_key);
    if (rk_lo != rj_hi && d0[rk_lo].row_k == hi) return {swap, d0 + rk_lo};
    return {swap, nullptr};
}

/** Torsion lookup with 5-stage equivalence cascade + fallback to the
 *  secondary torsion type.  Mirrors
 *  `MMFFTorCollection::getMMFFTorParams` exactly.  Returns the resolved
 *  (canonical torsion type, params pointer) — caller uses the canonical
 *  tor type for downstream debug/metrics, and nullptr params on miss. */
SMALLFIELD_HD_ inline std::pair<unsigned, mmff94_torsion_entry_t const *> lookup_torsion(
    mmff94_tables_view_t const &t, std::uint8_t tor_type_primary, std::uint8_t tor_type_secondary, std::uint8_t atom_i,
    std::uint8_t atom_j, std::uint8_t atom_k, std::uint8_t atom_l) noexcept {
    mmff94_torsion_entry_t const *hit = nullptr;
    unsigned canTorType = tor_type_primary;
    unsigned iter = 0;
    unsigned maxIter = 5;
    while (((iter < maxIter) && ((!hit) || (maxIter == 4))) ||
           ((iter == 4) && (tor_type_primary == 5) && tor_type_secondary)) {
        // Secondary fallback: after the 5-stage attempt with the primary
        // tor type, retry the first 4 stages with the secondary tor type.
        if ((maxIter == 5) && (iter == 4)) {
            maxIter = 4;
            iter = 0;
            canTorType = tor_type_secondary;
        }
        unsigned iWildCard = iter, lWildCard = iter;
        if (iter == 1) {
            iWildCard = 1;
            lWildCard = 3;
        }
        else if (iter == 2) {
            iWildCard = 3;
            lWildCard = 1;
        }

        std::uint8_t canI = (atom_i < t.def.size()) ? t.def[atom_i].eq_level[iWildCard] : 0;
        std::uint8_t canJ = atom_j;
        std::uint8_t canK = atom_k;
        std::uint8_t canL = (atom_l < t.def.size()) ? t.def[atom_l].eq_level[lWildCard] : 0;
        if (canJ > canK) {
            std::uint8_t tmp = canJ;
            canJ = canK;
            canK = tmp;
            tmp = canI;
            canI = canL;
            canL = tmp;
        }
        else if (canJ == canK && canI > canL) {
            std::uint8_t tmp = canI;
            canI = canL;
            canL = tmp;
        }

        // Cascade of 5 binary searches: j → k → i → l → tor_type.
        mmff94_torsion_entry_t const *t0 = t.torsion.data();
        std::size_t const n = t.torsion.size();
        auto j_key = [](mmff94_torsion_entry_t const &e) { return e.j_type; };
        auto k_key = [](mmff94_torsion_entry_t const &e) { return e.k_type; };
        auto i_key = [](mmff94_torsion_entry_t const &e) { return e.i_type; };
        auto l_key = [](mmff94_torsion_entry_t const &e) { return e.l_type; };
        auto tt_key = [](mmff94_torsion_entry_t const &e) { return e.tor_type; };
        std::size_t const j_lo = lower_bound_u8_(t0, n, canJ, j_key);
        std::size_t const j_hi = lower_bound_u8_(t0, n, (std::uint8_t)(canJ + 1), j_key);
        if (j_lo != j_hi) {
            std::size_t const k_lo = j_lo + lower_bound_u8_(t0 + j_lo, j_hi - j_lo, canK, k_key);
            std::size_t const k_hi = j_lo + lower_bound_u8_(t0 + j_lo, j_hi - j_lo, (std::uint8_t)(canK + 1), k_key);
            if (k_lo != k_hi) {
                std::size_t const i_lo = k_lo + lower_bound_u8_(t0 + k_lo, k_hi - k_lo, canI, i_key);
                std::size_t const i_hi = k_lo +
                                         lower_bound_u8_(t0 + k_lo, k_hi - k_lo, (std::uint8_t)(canI + 1), i_key);
                if (i_lo != i_hi) {
                    std::size_t const l_lo = i_lo + lower_bound_u8_(t0 + i_lo, i_hi - i_lo, canL, l_key);
                    std::size_t const l_hi = i_lo +
                                             lower_bound_u8_(t0 + i_lo, i_hi - i_lo, (std::uint8_t)(canL + 1), l_key);
                    if (l_lo != l_hi) {
                        std::size_t const tt_lo = l_lo + lower_bound_u8_(t0 + l_lo, l_hi - l_lo,
                                                                         (std::uint8_t)canTorType, tt_key);
                        if (tt_lo != l_hi && t0[tt_lo].tor_type == canTorType) {
                            hit = t0 + tt_lo;
                            if (maxIter == 4) break;
                        }
                    }
                }
            }
        }
        ++iter;
    }
    return {canTorType, hit};
}

} // namespace smallfield::parameters
