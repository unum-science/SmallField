/**
 *  @file src/smallfield/parameters/mmff94_tables.hpp
 *  @brief Flat MMFF94 parameter tables extracted once at context init.
 *
 *  RDKit's `DefaultParameters::getMMFF*()` collections are ultimately backed by parallel
 *  `std::vector<std::uint8_t>` lookup columns and a `std::vector<MMFF*>` parameter column — the
 *  `operator()` helpers do a cascade of `std::equal_range` calls against those vectors. A GPU port
 *  needs the same data in flat POD arrays that upload with one `cudaMemcpy` each and binary-search
 *  on the device.
 *
 *  This header exposes:
 *   - POD entry structs, `mmff94_*_entry_t`, one per collection.
 *   - `mmff94_tables_host_t` — aggregates all tables as sorted `std::vector`s in the same key order
 *     RDKit uses internally, so the device-side lookup can use `std::equal_range`-equivalent nested
 *     searches and produce bit-identical results.
 *   - `lookup_*` functions that match RDKit's `operator()` and `getMMFF*Params` behaviour exactly,
 *     including the 4/5-level equivalence cascades for angle, torsion and oop via `MMFFDef`.
 *
 *  Doubles are converted to floats at extraction time — the main ETKDG kernel already consumes
 *  float32, and the extra precision in RDKit's tables is not meaningful for the tolerances required.
 *
 *  Cross-validation gate: call both the `lookup_*` functions here and RDKit's `operator()` on 10K
 *  random key tuples, and assert byte-equality on the returned parameters. See
 *  `test/parameters/test_mmff94_tables.cpp`.
 */

#pragma once

#include <cstdint> // `std::uint8_t`

#include <utility> // `std::pair`
#include <vector>  // `std::vector`

#include "smallfield/types/views.hpp"
#include "smallfield/utilities.hpp"

/*  `SF_HOST_DEVICE_` resolves to `__host__ __device__` when compiled by nvcc, empty otherwise, so a
 *  function marked with it is callable from both host C++ and device kernels with the same source.
 *  The `lookup_*` helpers below reach device code as `constexpr` instead; the macro serves the MMFF94
 *  sources that include this header. */
#if defined(__CUDACC__)
#define SF_HOST_DEVICE_ __host__ __device__
#else
#define SF_HOST_DEVICE_
#endif

namespace smallfield::parameters {

#pragma region Table Entry Structs

/** Van-der-Waals parameters, one entry per MMFF atom type. Matches `mmff94_atom_vdw_params_t` in
 *  `types/etkdg_types.hpp` in most respects; the difference is that this struct keys by `atom_type`
 *  for lookup. */
struct alignas(8) mmff94_vdw_entry_t {
    std::uint8_t atom_type;
    /** 0 = '-', 1 = 'A' acceptor, 2 = 'D' donor. */
    std::uint8_t donor_acceptor;
    std::uint8_t padding[2];
    float alpha_i;
    /** Effective number of valence electrons, MMFF `N_i`. */
    float n_effective;
    /** MMFF `A_i`. */
    float a_scale;
    /** Slater-Kirkwood factor, MMFF `G_i`. */
    float g_factor;
    float r_star;
    float padding_float;
};
static_assert(sizeof(mmff94_vdw_entry_t) == 32);

/** Bond-stretch parameter, MMFF `kb` and `r0`, keyed by (canonical atom type pair, MMFF bond type 0/1). */
struct alignas(16) mmff94_bond_entry_t {
    /** Canonical: i_type <= j_type. */
    std::uint8_t i_type;
    std::uint8_t j_type;
    std::uint8_t bond_type;
    std::uint8_t padding;
    /** MMFF `kb`, in mdyn per angstrom. */
    float force_constant;
    /** MMFF `r0`, in angstroms. */
    float equilibrium_distance;
    float padding_float;
};
static_assert(sizeof(mmff94_bond_entry_t) == 16);

/** Angle-bend parameter, MMFF `ka` and `theta0` in degrees, keyed by (j_type, i_type, k_type, angle_type)
 *  with i_type ≤ k_type canonical. */
struct alignas(16) mmff94_angle_entry_t {
    /** Center atom type. */
    std::uint8_t j_type;
    /** Canonical: i_type <= k_type. */
    std::uint8_t i_type;
    std::uint8_t k_type;
    std::uint8_t angle_type;
    /** MMFF `ka`, in mdyn·angstrom per rad². */
    float force_constant;
    float theta0_deg;
    float padding_float;
};
static_assert(sizeof(mmff94_angle_entry_t) == 16);

/** Torsion parameter (V1, V2, V3), keyed by (j, k, i, l, torsion_type) with canonicalization defined
 *  in RDKit's `MMFFTorCollection::getMMFFTorParams`: when canonical_j_type > canonical_k_type, swap
 *  (i,j)↔(l,k). */
struct alignas(16) mmff94_torsion_entry_t {
    std::uint8_t j_type;
    std::uint8_t k_type;
    std::uint8_t i_type;
    std::uint8_t l_type;
    std::uint8_t tor_type;
    std::uint8_t padding[3];
    /** First barrier height, MMFF `V1`, in kcal/mol. */
    float torsion_v1;
    /** Second barrier height, MMFF `V2`, in kcal/mol. */
    float torsion_v2;
    /** Third barrier height, MMFF `V3`, in kcal/mol. */
    float torsion_v3;
};
static_assert(sizeof(mmff94_torsion_entry_t) == 32);

/** Out-of-plane bend parameter, MMFF `koop`, keyed by (j, canonical_i_type, canonical_k_type,
 *  canonical_l_type) where (canonical_i_type, canonical_k_type, canonical_l_type) is sorted ascending. */
struct alignas(8) mmff94_oop_entry_t {
    /** Center. */
    std::uint8_t j_type;
    /** Sorted. */
    std::uint8_t i_type;
    std::uint8_t k_type;
    std::uint8_t l_type;
    /** MMFF `koop`, in mdyn·angstrom per rad², against the Wilson out-of-plane angle. */
    float force_constant;
};
static_assert(sizeof(mmff94_oop_entry_t) == 8 || sizeof(mmff94_oop_entry_t) == 16);

/** Stretch-bend parameter (kba_ijk, kba_kji), keyed by (j_type, i_type, k_type, stretch_bend_type)
 *  with the caller responsible for the swap flag returned alongside — whether i/k were swapped
 *  canonically. */
struct alignas(16) mmff94_stbn_entry_t {
    std::uint8_t j_type;
    /** Canonical: i_type <= k_type. */
    std::uint8_t i_type;
    std::uint8_t k_type;
    std::uint8_t stretch_bend_type;
    float kba_ijk;
    float kba_kji;
    float padding_float;
};
static_assert(sizeof(mmff94_stbn_entry_t) == 16);

/** Default-stretch-bend (DFSB) fallback parameter. Used by RDKit when primary Stbn lookup fails:
 *  keyed by (periodic-table row of i, j, k). */
struct alignas(8) mmff94_dfsb_entry_t {
    /** Canonical: row_i <= row_k. */
    std::uint8_t row_i;
    std::uint8_t row_j;
    std::uint8_t row_k;
    std::uint8_t padding;
    float kba_ijk;
    float kba_kji;
};
static_assert(sizeof(mmff94_dfsb_entry_t) == 16);

/** MMFFProp — per-atom-type chemical properties. Used by `getMMFFBondType` through the
 *  `single_bond_multiple_bond` and `aromatic` fields, and by the empirical angle-bending rule
 *  through `coordination_number`, `valence`, `multiple_bond_designator` and `linear`. Dense array
 *  indexed directly by `atom_type`, one slot per possible type with zeros for unused slots,
 *  mirroring `mmff94_def_entry_t`. */
struct alignas(8) mmff94_prop_entry_t {
    std::uint8_t atomic_num;
    std::uint8_t coordination_number;
    std::uint8_t valence;
    std::uint8_t pi_lone_pair;
    std::uint8_t multiple_bond_designator;
    std::uint8_t aromatic;
    /** 1 when the atom requires linear bond geometry. */
    std::uint8_t linear;
    std::uint8_t single_bond_multiple_bond;
};
static_assert(sizeof(mmff94_prop_entry_t) == 8);

/** MMFFDef equivalence levels — 4 bytes per atom type. Indexed directly by `atom_type` — a dense
 *  array with one entry per possible type in [1, max_atom_type] and zeros for unused slots. */
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

/** Bond-charge-increment parameter. The *directional* increment applied from atom i to atom j; atom i's
 *  partial charge gains it, atom j loses it. Sorted by (i, j, bond_type); RDKit's Chg lookup checks (i,j,bt)
 *  and (j,i,bt), flipping sign accordingly. */
struct alignas(8) mmff94_chg_entry_t {
    std::uint8_t i_type;
    std::uint8_t j_type;
    std::uint8_t bond_type;
    std::uint8_t padding;
    /** MMFF `bci`, in electrons. */
    float bond_charge_increment;
};
static_assert(sizeof(mmff94_chg_entry_t) == 8);

/** Bndk default-bond parameter, keyed by (atomic_number_i, atomic_number_j) canonical ordering. Used
 *  by RDKit's empirical bond-stretch rule when the primary Bond lookup fails. */
struct alignas(8) mmff94_bndk_entry_t {
    /** Canonical: i <= j. */
    std::uint8_t atomic_num_i;
    std::uint8_t atomic_num_j;
    std::uint8_t padding[2];
    /** MMFF `kb`, in mdyn per angstrom. */
    float force_constant;
    /** MMFF `r0`, in angstroms. */
    float equilibrium_distance;
};
static_assert(sizeof(mmff94_bndk_entry_t) == 16);

/** Herschbach-Laurie's rule parameter, for Badger's rule bond stretch estimation when neither Bond
 *  nor Bndk have the pair. */
struct alignas(8) mmff94_hl_entry_t {
    /** Canonical: row_i <= row_j. */
    std::uint8_t row_i;
    std::uint8_t row_j;
    std::uint8_t padding[2];
    float a_ij;
    float d_ij;
    float dp_ij;
};
static_assert(sizeof(mmff94_hl_entry_t) == 16);

/** Covalent radius + Pauling electronegativity per element, for the MMFF empirical bond-stretch
 *  fallback. */
struct mmff94_covrad_entry_t {
    std::uint8_t atomic_num;
    std::uint8_t padding[3];
    /** MMFF `r0` of the CovRadPauEle table — a covalent radius in angstroms, not a bond length. */
    float covalent_radius;
    /** Pauling electronegativity, MMFF `chi`. Unrelated to the Wilson angle `chi` of `mmff94_oop_entry_t`. */
    float pauling_electronegativity;
};
static_assert(sizeof(mmff94_covrad_entry_t) == 12);

#pragma endregion Table Entry Structs

#pragma region Table View

/**
 *  @brief POD view over every extracted MMFF table, passed by value into kernels.
 *
 *  Consumed by all `lookup_*` functions and by host and device alike — on host each span points into
 *  the `std::vector`s inside `mmff94_tables_host_t`; on device into the per-table device buffers that
 *  `upload_mmff94_tables_to_device` fills, see `parameters/mmff94_tables_device.cuh`.
 *
 *  The spans are `custd::span` rather than `std::span` because `std::span`'s members are not
 *  `__host__ __device__`, so the shared `lookup_*` cascades could not index one from a kernel.
 */
struct mmff94_tables_view_t {
    custd::span<mmff94_vdw_entry_t const> vdw;
    custd::span<mmff94_bond_entry_t const> bond;
    custd::span<mmff94_angle_entry_t const> angle;
    custd::span<mmff94_torsion_entry_t const> torsion;
    custd::span<mmff94_oop_entry_t const> oop;
    custd::span<mmff94_stbn_entry_t const> stbn;
    custd::span<mmff94_dfsb_entry_t const> dfsb;
    custd::span<mmff94_def_entry_t const> def;
    custd::span<mmff94_prop_entry_t const> prop;
    custd::span<mmff94_pbci_entry_t const> pbci;
    custd::span<mmff94_chg_entry_t const> chg;
    custd::span<mmff94_bndk_entry_t const> bndk;
    custd::span<mmff94_hl_entry_t const> hl;
    custd::span<mmff94_covrad_entry_t const> covrad;

    float vdw_power = 0.25f;
    float vdw_b = 0.2f;
    float vdw_beta = 12.0f;
    float vdw_darad = 0.8f;
    float vdw_daeps = 0.5f;

    std::uint8_t max_atom_type = 0;
};

/*  Fourteen 16-byte spans, five floats and a byte, padded to the pointer alignment. The upload in
 *  `mmff94_tables_device.cuh` writes these fields and every MMFF kernel takes the struct by value, so a
 *  change in the layout silently redirects every parameter lookup on the device. */
static_assert(sizeof(mmff94_tables_view_t) == 248);

#pragma endregion Table View

#pragma region Table Container

/** All MMFF94 parameter tables in a single container, extracted once at context init. Each vector is
 *  sorted in the exact lookup order RDKit uses internally — see the per-collection `d_*` column
 *  orderings in `rdkit/ForceField/MMFF/Params.h`. */
struct mmff94_tables_host_t {
    /** Sorted by atom_type. */
    std::vector<mmff94_vdw_entry_t> vdw;
    /** Sorted by (i, j, bond_type). */
    std::vector<mmff94_bond_entry_t> bond;
    /** Sorted by (j, i, k, angle_type). */
    std::vector<mmff94_angle_entry_t> angle;
    /** Sorted by (j, k, i, l, tor_type). */
    std::vector<mmff94_torsion_entry_t> torsion;
    /** Sorted by (j, i, k, l). */
    std::vector<mmff94_oop_entry_t> oop;
    /** Sorted by (j, i, k, sb_type). */
    std::vector<mmff94_stbn_entry_t> stbn;
    /** Sorted by (row_i, row_j, row_k). */
    std::vector<mmff94_dfsb_entry_t> dfsb;
    /** Dense: indexed directly by atom_type. */
    std::vector<mmff94_def_entry_t> def;
    /** Dense: indexed directly by atom_type. */
    std::vector<mmff94_prop_entry_t> prop;
    /** Sorted by atom_type. */
    std::vector<mmff94_pbci_entry_t> pbci;
    /** Sorted by (i, j, bond_type). */
    std::vector<mmff94_chg_entry_t> chg;
    /** Sorted by (atomic_num_i, atomic_num_j). */
    std::vector<mmff94_bndk_entry_t> bndk;
    /** Sorted by (row_i, row_j). */
    std::vector<mmff94_hl_entry_t> hl;
    /** Sorted by atomic_num. */
    std::vector<mmff94_covrad_entry_t> covrad;

    /** VdW collection scalar constants, from `MMFFVdWCollection` members. */
    float vdw_power = 0.25f;
    float vdw_b = 0.2f;
    float vdw_beta = 12.0f;
    float vdw_darad = 0.8f;
    float vdw_daeps = 0.5f;

    /** Highest atom type seen, for `MMFFDef` indexing. */
    std::uint8_t max_atom_type = 0;

    /** Build a read-only view that references this container's vector storage. Cheap — just copies
     *  pointers + sizes + scalar constants. */
    mmff94_tables_view_t view() const noexcept {
        mmff94_tables_view_t result;
        result.vdw = {vdw.data(), vdw.size()};
        result.bond = {bond.data(), bond.size()};
        result.angle = {angle.data(), angle.size()};
        result.torsion = {torsion.data(), torsion.size()};
        result.oop = {oop.data(), oop.size()};
        result.stbn = {stbn.data(), stbn.size()};
        result.dfsb = {dfsb.data(), dfsb.size()};
        result.def = {def.data(), def.size()};
        result.prop = {prop.data(), prop.size()};
        result.pbci = {pbci.data(), pbci.size()};
        result.chg = {chg.data(), chg.size()};
        result.bndk = {bndk.data(), bndk.size()};
        result.hl = {hl.data(), hl.size()};
        result.covrad = {covrad.data(), covrad.size()};
        result.vdw_power = vdw_power;
        result.vdw_b = vdw_b;
        result.vdw_beta = vdw_beta;
        result.vdw_darad = vdw_darad;
        result.vdw_daeps = vdw_daeps;
        result.max_atom_type = max_atom_type;
        return result;
    }
};

#pragma endregion Table Container

/**
 *  @brief Hand-rolled `lower_bound` over one sorted table column, used by every lookup cascade below.
 *
 *  The project keeps its own binary search instead of `std::lower_bound`: libstdc++'s version reaches
 *  device code only through `--expt-relaxed-constexpr`, and along that route it has been seen to stop
 *  the search early, returning the first few elements only and declaring the key not found for
 *  anything past about 8 entries in a 95-element sorted array. A local definition keeps the code path
 *  identical between host and device.
 */
template <typename element_, typename key_, typename key_accessor_>
constexpr std::size_t lower_bound_(element_ const *base, std::size_t count, key_ key, key_accessor_ &&key_of) noexcept {
    std::size_t lower = 0, upper = count;
    while (lower < upper) {
        std::size_t const middle = lower + (upper - lower) / 2;
        if (key_of(base[middle]) < key) lower = middle + 1;
        else upper = middle;
    }
    return lower;
}

/**
 *  @brief Binary-search cascade matching `MMFFVdWCollection::operator()`.
 *
 *  This and the lookups below match RDKit's `operator()` behaviour, byte-identical to RDKit, and
 *  serve two callers: the cross-validation test, which sanity-checks that the extraction produced
 *  sorted data correctly, and the native CPU port of `extract_mmff94_params_via_rdkit`.
 */
constexpr mmff94_vdw_entry_t const *lookup_vdw(mmff94_tables_view_t const &tables, std::uint8_t atom_type) noexcept {
    std::size_t const index = lower_bound_(tables.vdw.data(), tables.vdw.size(), atom_type,
                                           [](mmff94_vdw_entry_t const &entry) { return entry.atom_type; });
    if (index < tables.vdw.size() && tables.vdw[index].atom_type == atom_type) return &tables.vdw[index];
    return nullptr;
}

/** Binary-search cascade matching `MMFFBondCollection::operator()`. Canonicalises (atom_i, atom_j) to
 *  (lower_type, upper_type). */
constexpr mmff94_bond_entry_t const *lookup_bond(mmff94_tables_view_t const &tables, std::uint8_t bond_type,
                                                 std::uint8_t atom_i, std::uint8_t atom_j) noexcept {
    std::uint8_t lower_type = atom_i, upper_type = atom_j;
    if (lower_type > upper_type) swap_(lower_type, upper_type);

    // Cascade on i_type → j_type → bond_type.
    mmff94_bond_entry_t const *bond_base = tables.bond.data();
    std::size_t const count = tables.bond.size();
    auto i_key = [](mmff94_bond_entry_t const &entry) { return entry.i_type; };
    auto j_key = [](mmff94_bond_entry_t const &entry) { return entry.j_type; };
    auto bond_type_key = [](mmff94_bond_entry_t const &entry) { return entry.bond_type; };
    std::size_t const i_lo = lower_bound_(bond_base, count, lower_type, i_key);
    std::size_t const i_hi = lower_bound_(bond_base, count, lower_type + 1, i_key);
    if (i_lo == i_hi) return nullptr;
    std::size_t const j_span = i_hi - i_lo;
    std::size_t const j_lo = i_lo + lower_bound_(bond_base + i_lo, j_span, upper_type, j_key);
    std::size_t const j_hi = i_lo + lower_bound_(bond_base + i_lo, j_span, upper_type + 1, j_key);
    if (j_lo == j_hi) return nullptr;
    std::size_t const bond_type_span = j_hi - j_lo;
    std::size_t const bond_type_low = j_lo + lower_bound_(bond_base + j_lo, bond_type_span, bond_type, bond_type_key);
    if (bond_type_low != j_hi && bond_base[bond_type_low].bond_type == bond_type) return bond_base + bond_type_low;
    return nullptr;
}

/** Binary-search cascade matching `MMFFAngleCollection::operator()` — including the 4-level
 *  equivalence fallback via `MMFFDef`. */
constexpr mmff94_angle_entry_t const *lookup_angle(mmff94_tables_view_t const &tables, std::uint8_t angle_type,
                                                   std::uint8_t atom_i, std::uint8_t atom_j_center,
                                                   std::uint8_t atom_k) noexcept {
    mmff94_angle_entry_t const *angle_base = tables.angle.data();
    std::size_t const count = tables.angle.size();
    auto j_key = [](mmff94_angle_entry_t const &entry) { return entry.j_type; };
    auto i_key = [](mmff94_angle_entry_t const &entry) { return entry.i_type; };
    auto k_key = [](mmff94_angle_entry_t const &entry) { return entry.k_type; };
    auto at_key = [](mmff94_angle_entry_t const &entry) { return entry.angle_type; };
    for (int iter = 0; iter < 4; ++iter) {
        std::uint8_t canonical_i_type = (atom_i < tables.def.size()) ? tables.def[atom_i].eq_level[iter]
                                                                     : std::uint8_t {0};
        std::uint8_t canonical_k_type = (atom_k < tables.def.size()) ? tables.def[atom_k].eq_level[iter]
                                                                     : std::uint8_t {0};
        if (canonical_i_type > canonical_k_type) swap_(canonical_i_type, canonical_k_type);

        std::size_t const j_lo = lower_bound_(angle_base, count, atom_j_center, j_key);
        std::size_t const j_hi = lower_bound_(angle_base, count, atom_j_center + 1, j_key);
        if (j_lo == j_hi) continue;
        std::size_t const i_lo = j_lo + lower_bound_(angle_base + j_lo, j_hi - j_lo, canonical_i_type, i_key);
        std::size_t const i_hi = j_lo + lower_bound_(angle_base + j_lo, j_hi - j_lo, canonical_i_type + 1, i_key);
        if (i_lo == i_hi) continue;
        std::size_t const k_lo = i_lo + lower_bound_(angle_base + i_lo, i_hi - i_lo, canonical_k_type, k_key);
        std::size_t const k_hi = i_lo + lower_bound_(angle_base + i_lo, i_hi - i_lo, canonical_k_type + 1, k_key);
        if (k_lo == k_hi) continue;
        std::size_t const at_lo = k_lo + lower_bound_(angle_base + k_lo, k_hi - k_lo, angle_type, at_key);
        if (at_lo != k_hi && angle_base[at_lo].angle_type == angle_type) return angle_base + at_lo;
    }
    return nullptr;
}

/** MMFFProp dense lookup — `nullptr` for unknown atom types. */
constexpr mmff94_prop_entry_t const *lookup_prop(mmff94_tables_view_t const &tables, std::uint8_t atom_type) noexcept {
    if (atom_type == 0 || atom_type >= tables.prop.size()) return nullptr;
    return &tables.prop[atom_type];
}

/** Partial-bond-charge-increment lookup by atom type. */
constexpr mmff94_pbci_entry_t const *lookup_pbci(mmff94_tables_view_t const &tables, std::uint8_t atom_type) noexcept {
    std::size_t const index = lower_bound_(tables.pbci.data(), tables.pbci.size(), atom_type,
                                           [](mmff94_pbci_entry_t const &entry) { return entry.atom_type; });
    if (index < tables.pbci.size() && tables.pbci[index].atom_type == atom_type) return &tables.pbci[index];
    return nullptr;
}

/** Signed bond-charge-increment lookup. Mirrors `MMFFChgCollection::getMMFFChgParams`: returns (sign, entry)
 *  where sign is -1 if `i_type <= j_type` held in the tabulation order, +1 if the arguments had to be
 *  swapped. On a miss the pointer is null and the sign still carries. */
constexpr std::pair<int, mmff94_chg_entry_t const *> lookup_chg(mmff94_tables_view_t const &tables,
                                                                std::uint8_t bond_type, std::uint8_t atom_i,
                                                                std::uint8_t atom_j) noexcept {
    int sign = -1;
    std::uint8_t lower_type = atom_i, upper_type = atom_j;
    if (atom_i > atom_j) {
        lower_type = atom_j;
        upper_type = atom_i;
        sign = 1;
    }
    mmff94_chg_entry_t const *chg_base = tables.chg.data();
    std::size_t const count = tables.chg.size();
    auto i_key = [](mmff94_chg_entry_t const &entry) { return entry.i_type; };
    auto j_key = [](mmff94_chg_entry_t const &entry) { return entry.j_type; };
    auto bond_type_key = [](mmff94_chg_entry_t const &entry) { return entry.bond_type; };
    std::size_t const i_lo = lower_bound_(chg_base, count, lower_type, i_key);
    std::size_t const i_hi = lower_bound_(chg_base, count, lower_type + 1, i_key);
    if (i_lo == i_hi) return {sign, nullptr};
    std::size_t const j_lo = i_lo + lower_bound_(chg_base + i_lo, i_hi - i_lo, upper_type, j_key);
    std::size_t const j_hi = i_lo + lower_bound_(chg_base + i_lo, i_hi - i_lo, upper_type + 1, j_key);
    if (j_lo == j_hi) return {sign, nullptr};
    std::size_t const bond_type_low = j_lo + lower_bound_(chg_base + j_lo, j_hi - j_lo, bond_type, bond_type_key);
    if (bond_type_low != j_hi && chg_base[bond_type_low].bond_type == bond_type)
        return {sign, chg_base + bond_type_low};
    return {sign, nullptr};
}

/** Bndk lookup — the default bond keyed by atomic-number pair. */
constexpr mmff94_bndk_entry_t const *lookup_bndk(mmff94_tables_view_t const &tables, std::uint8_t atomic_i,
                                                 std::uint8_t atomic_j) noexcept {
    std::uint8_t lower_atomic_num = atomic_i, upper_atomic_num = atomic_j;
    if (lower_atomic_num > upper_atomic_num) swap_(lower_atomic_num, upper_atomic_num);
    mmff94_bndk_entry_t const *bndk_base = tables.bndk.data();
    std::size_t const count = tables.bndk.size();
    auto i_key = [](mmff94_bndk_entry_t const &entry) { return entry.atomic_num_i; };
    auto j_key = [](mmff94_bndk_entry_t const &entry) { return entry.atomic_num_j; };
    std::size_t const i_lo = lower_bound_(bndk_base, count, lower_atomic_num, i_key);
    std::size_t const i_hi = lower_bound_(bndk_base, count, lower_atomic_num + 1, i_key);
    if (i_lo == i_hi) return nullptr;
    std::size_t const j_lo = i_lo + lower_bound_(bndk_base + i_lo, i_hi - i_lo, upper_atomic_num, j_key);
    if (j_lo != i_hi && bndk_base[j_lo].atomic_num_j == upper_atomic_num) return bndk_base + j_lo;
    return nullptr;
}

/** Covalent radius + Pauling electronegativity lookup by atomic number. */
constexpr mmff94_covrad_entry_t const *lookup_covrad(mmff94_tables_view_t const &tables,
                                                     std::uint8_t atomic_num) noexcept {
    std::size_t const index = lower_bound_(tables.covrad.data(), tables.covrad.size(), atomic_num,
                                           [](mmff94_covrad_entry_t const &entry) { return entry.atomic_num; });
    if (index < tables.covrad.size() && tables.covrad[index].atomic_num == atomic_num) return &tables.covrad[index];
    return nullptr;
}

/** Herschbach-Laurie lookup by periodic-table-row pair. */
constexpr mmff94_hl_entry_t const *lookup_hl(mmff94_tables_view_t const &tables, std::uint8_t row_i,
                                             std::uint8_t row_j) noexcept {
    std::uint8_t lower_row = row_i, upper_row = row_j;
    if (lower_row > upper_row) swap_(lower_row, upper_row);
    mmff94_hl_entry_t const *hl_base = tables.hl.data();
    std::size_t const count = tables.hl.size();
    auto i_key = [](mmff94_hl_entry_t const &entry) { return entry.row_i; };
    auto j_key = [](mmff94_hl_entry_t const &entry) { return entry.row_j; };
    std::size_t const i_lo = lower_bound_(hl_base, count, lower_row, i_key);
    std::size_t const i_hi = lower_bound_(hl_base, count, lower_row + 1, i_key);
    if (i_lo == i_hi) return nullptr;
    std::size_t const j_lo = i_lo + lower_bound_(hl_base + i_lo, i_hi - i_lo, upper_row, j_key);
    if (j_lo != i_hi && hl_base[j_lo].row_j == upper_row) return hl_base + j_lo;
    return nullptr;
}

/** OOP bend lookup with 4-stage equivalence cascade. Matches `MMFFOopCollection::operator()` exactly. */
constexpr mmff94_oop_entry_t const *lookup_oop(mmff94_tables_view_t const &tables, std::uint8_t atom_i,
                                               std::uint8_t atom_j, std::uint8_t atom_k, std::uint8_t atom_l) noexcept {
    mmff94_oop_entry_t const *oop_base = tables.oop.data();
    std::size_t const count = tables.oop.size();
    auto j_key = [](mmff94_oop_entry_t const &entry) { return entry.j_type; };
    auto i_key = [](mmff94_oop_entry_t const &entry) { return entry.i_type; };
    auto k_key = [](mmff94_oop_entry_t const &entry) { return entry.k_type; };
    auto l_key = [](mmff94_oop_entry_t const &entry) { return entry.l_type; };
    for (int iter = 0; iter < 4; ++iter) {
        // The MMFF out-of-plane spec letters i, k, l after the sorting network below.
        std::uint8_t canonical_ikl_types[3] = {
            (atom_i < tables.def.size()) ? tables.def[atom_i].eq_level[iter] : std::uint8_t {0},
            (atom_k < tables.def.size()) ? tables.def[atom_k].eq_level[iter] : std::uint8_t {0},
            (atom_l < tables.def.size()) ? tables.def[atom_l].eq_level[iter] : std::uint8_t {0},
        };
        if (canonical_ikl_types[0] > canonical_ikl_types[1]) swap_(canonical_ikl_types[0], canonical_ikl_types[1]);
        if (canonical_ikl_types[1] > canonical_ikl_types[2]) swap_(canonical_ikl_types[1], canonical_ikl_types[2]);
        if (canonical_ikl_types[0] > canonical_ikl_types[1]) swap_(canonical_ikl_types[0], canonical_ikl_types[1]);
        std::size_t const j_lo = lower_bound_(oop_base, count, atom_j, j_key);
        std::size_t const j_hi = lower_bound_(oop_base, count, atom_j + 1, j_key);
        if (j_lo == j_hi) continue;
        std::size_t const i_lo = j_lo + lower_bound_(oop_base + j_lo, j_hi - j_lo, canonical_ikl_types[0], i_key);
        std::size_t const i_hi = j_lo + lower_bound_(oop_base + j_lo, j_hi - j_lo, canonical_ikl_types[0] + 1, i_key);
        if (i_lo == i_hi) continue;
        std::size_t const k_lo = i_lo + lower_bound_(oop_base + i_lo, i_hi - i_lo, canonical_ikl_types[1], k_key);
        std::size_t const k_hi = i_lo + lower_bound_(oop_base + i_lo, i_hi - i_lo, canonical_ikl_types[1] + 1, k_key);
        if (k_lo == k_hi) continue;
        std::size_t const l_lo = k_lo + lower_bound_(oop_base + k_lo, k_hi - k_lo, canonical_ikl_types[2], l_key);
        if (l_lo != k_hi && oop_base[l_lo].l_type == canonical_ikl_types[2]) return oop_base + l_lo;
    }
    return nullptr;
}

/** Stretch-bend lookup with swap tracking. No equivalence cascade — the STBN table is a single level.
 *  Swap indicates whether kba_ijk and kba_kji should be exchanged, matching
 *  `MMFFStbnCollection::getMMFFStbnParams`. */
constexpr std::pair<bool, mmff94_stbn_entry_t const *> lookup_stbn(mmff94_tables_view_t const &tables,
                                                                   std::uint8_t stretch_bend_type,
                                                                   std::uint8_t bond_type_1, std::uint8_t bond_type_2,
                                                                   std::uint8_t atom_i, std::uint8_t atom_j_center,
                                                                   std::uint8_t atom_k) noexcept {
    bool swap = false;
    std::uint8_t lower_type = atom_i, upper_type = atom_k;
    if (atom_i > atom_k) {
        lower_type = atom_k;
        upper_type = atom_i;
        swap = true;
    }
    else if (atom_i == atom_k) swap = bond_type_1 < bond_type_2;

    mmff94_stbn_entry_t const *stbn_base = tables.stbn.data();
    std::size_t const count = tables.stbn.size();
    auto j_key = [](mmff94_stbn_entry_t const &entry) { return entry.j_type; };
    auto i_key = [](mmff94_stbn_entry_t const &entry) { return entry.i_type; };
    auto k_key = [](mmff94_stbn_entry_t const &entry) { return entry.k_type; };
    auto sb_key = [](mmff94_stbn_entry_t const &entry) { return entry.stretch_bend_type; };
    std::size_t const j_lo = lower_bound_(stbn_base, count, atom_j_center, j_key);
    std::size_t const j_hi = lower_bound_(stbn_base, count, atom_j_center + 1, j_key);
    if (j_lo == j_hi) return {swap, nullptr};
    std::size_t const i_lo = j_lo + lower_bound_(stbn_base + j_lo, j_hi - j_lo, lower_type, i_key);
    std::size_t const i_hi = j_lo + lower_bound_(stbn_base + j_lo, j_hi - j_lo, lower_type + 1, i_key);
    if (i_lo == i_hi) return {swap, nullptr};
    std::size_t const k_lo = i_lo + lower_bound_(stbn_base + i_lo, i_hi - i_lo, upper_type, k_key);
    std::size_t const k_hi = i_lo + lower_bound_(stbn_base + i_lo, i_hi - i_lo, upper_type + 1, k_key);
    if (k_lo == k_hi) return {swap, nullptr};
    std::size_t const stretch_bend_low = k_lo + lower_bound_(stbn_base + k_lo, k_hi - k_lo, stretch_bend_type, sb_key);
    if (stretch_bend_low != k_hi && stbn_base[stretch_bend_low].stretch_bend_type == stretch_bend_type)
        return {swap, stbn_base + stretch_bend_low};
    return {swap, nullptr};
}

/** DFSB fallback lookup by periodic-table-row triple, with swap. */
constexpr std::pair<bool, mmff94_dfsb_entry_t const *> lookup_dfsb(mmff94_tables_view_t const &tables,
                                                                   std::uint8_t row_i, std::uint8_t row_j,
                                                                   std::uint8_t row_k) noexcept {
    bool swap = false;
    std::uint8_t lower_row = row_i, upper_row = row_k;
    if (lower_row > upper_row) {
        swap_(lower_row, upper_row);
        swap = true;
    }
    mmff94_dfsb_entry_t const *dfsb_base = tables.dfsb.data();
    std::size_t const count = tables.dfsb.size();
    auto ri_key = [](mmff94_dfsb_entry_t const &entry) { return entry.row_i; };
    auto rj_key = [](mmff94_dfsb_entry_t const &entry) { return entry.row_j; };
    auto rk_key = [](mmff94_dfsb_entry_t const &entry) { return entry.row_k; };
    std::size_t const ri_lo = lower_bound_(dfsb_base, count, lower_row, ri_key);
    std::size_t const ri_hi = lower_bound_(dfsb_base, count, lower_row + 1, ri_key);
    if (ri_lo == ri_hi) return {swap, nullptr};
    std::size_t const rj_lo = ri_lo + lower_bound_(dfsb_base + ri_lo, ri_hi - ri_lo, row_j, rj_key);
    std::size_t const rj_hi = ri_lo + lower_bound_(dfsb_base + ri_lo, ri_hi - ri_lo, row_j + 1, rj_key);
    if (rj_lo == rj_hi) return {swap, nullptr};
    std::size_t const rk_lo = rj_lo + lower_bound_(dfsb_base + rj_lo, rj_hi - rj_lo, upper_row, rk_key);
    if (rk_lo != rj_hi && dfsb_base[rk_lo].row_k == upper_row) return {swap, dfsb_base + rk_lo};
    return {swap, nullptr};
}

/** Torsion lookup with 5-stage equivalence cascade + fallback to the secondary torsion type. Mirrors
 *  `MMFFTorCollection::getMMFFTorParams` exactly. Returns the resolved (canonical torsion type,
 *  params pointer) — caller uses the canonical tor type for downstream debug/metrics, and nullptr
 *  params on miss. */
constexpr std::pair<unsigned, mmff94_torsion_entry_t const *> lookup_torsion(
    mmff94_tables_view_t const &tables, std::uint8_t tor_type_primary, std::uint8_t tor_type_secondary,
    std::uint8_t atom_i, std::uint8_t atom_j, std::uint8_t atom_k, std::uint8_t atom_l) noexcept {
    mmff94_torsion_entry_t const *hit = nullptr;
    unsigned canonical_torsion_type = tor_type_primary;
    unsigned iter = 0;
    unsigned max_iterations = 5;
    while (((iter < max_iterations) && ((!hit) || (max_iterations == 4))) ||
           ((iter == 4) && (tor_type_primary == 5) && tor_type_secondary)) {
        // Secondary fallback: after the 5-stage attempt with the primary tor type, retry the first
        // 4 stages with the secondary tor type.
        if ((max_iterations == 5) && (iter == 4)) {
            max_iterations = 4;
            iter = 0;
            canonical_torsion_type = tor_type_secondary;
        }
        unsigned i_wildcard_level = iter, l_wildcard_level = iter;
        if (iter == 1) {
            i_wildcard_level = 1;
            l_wildcard_level = 3;
        }
        else if (iter == 2) {
            i_wildcard_level = 3;
            l_wildcard_level = 1;
        }

        std::uint8_t canonical_i_type = (atom_i < tables.def.size()) ? tables.def[atom_i].eq_level[i_wildcard_level]
                                                                     : 0;
        std::uint8_t canonical_j_type = atom_j;
        std::uint8_t canonical_k_type = atom_k;
        std::uint8_t canonical_l_type = (atom_l < tables.def.size()) ? tables.def[atom_l].eq_level[l_wildcard_level]
                                                                     : 0;
        if (canonical_j_type > canonical_k_type) {
            swap_(canonical_j_type, canonical_k_type);
            swap_(canonical_i_type, canonical_l_type);
        }
        else if (canonical_j_type == canonical_k_type && canonical_i_type > canonical_l_type) {
            swap_(canonical_i_type, canonical_l_type);
        }

        // Cascade of 5 binary searches: j → k → i → l → tor_type.
        mmff94_torsion_entry_t const *torsion_base = tables.torsion.data();
        std::size_t const count = tables.torsion.size();
        auto j_key = [](mmff94_torsion_entry_t const &entry) { return entry.j_type; };
        auto k_key = [](mmff94_torsion_entry_t const &entry) { return entry.k_type; };
        auto i_key = [](mmff94_torsion_entry_t const &entry) { return entry.i_type; };
        auto l_key = [](mmff94_torsion_entry_t const &entry) { return entry.l_type; };
        auto tt_key = [](mmff94_torsion_entry_t const &entry) { return entry.tor_type; };
        std::size_t const j_lo = lower_bound_(torsion_base, count, canonical_j_type, j_key);
        std::size_t const j_hi = lower_bound_(torsion_base, count, canonical_j_type + 1, j_key);
        if (j_lo != j_hi) {
            int const canonical_k_type_next = canonical_k_type + 1;
            std::size_t const k_lo = j_lo + lower_bound_(torsion_base + j_lo, j_hi - j_lo, canonical_k_type, k_key);
            std::size_t const k_hi = j_lo +
                                     lower_bound_(torsion_base + j_lo, j_hi - j_lo, canonical_k_type_next, k_key);
            if (k_lo != k_hi) {
                int const canonical_i_type_next = canonical_i_type + 1;
                std::size_t const i_lo = k_lo + lower_bound_(torsion_base + k_lo, k_hi - k_lo, canonical_i_type, i_key);
                std::size_t const i_hi = k_lo +
                                         lower_bound_(torsion_base + k_lo, k_hi - k_lo, canonical_i_type_next, i_key);
                if (i_lo != i_hi) {
                    int const canonical_l_type_next = canonical_l_type + 1;
                    std::size_t const l_lo = i_lo +
                                             lower_bound_(torsion_base + i_lo, i_hi - i_lo, canonical_l_type, l_key);
                    std::size_t const l_hi = i_lo + lower_bound_(torsion_base + i_lo, i_hi - i_lo,
                                                                 canonical_l_type_next, l_key);
                    if (l_lo != l_hi) {
                        std::size_t const tt_lo = l_lo + lower_bound_(torsion_base + l_lo, l_hi - l_lo,
                                                                      (std::uint8_t)canonical_torsion_type, tt_key);
                        if (tt_lo != l_hi && torsion_base[tt_lo].tor_type == canonical_torsion_type) {
                            hit = torsion_base + tt_lo;
                            if (max_iterations == 4) break;
                        }
                    }
                }
            }
        }
        ++iter;
    }
    return {canonical_torsion_type, hit};
}

} // namespace smallfield::parameters
