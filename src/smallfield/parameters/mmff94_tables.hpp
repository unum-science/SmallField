/**
 *  @file src/smallfield/parameters/mmff94_tables.hpp
 *  @brief Flat MMFF94 parameter tables extracted once at context init.
 *
 *  RDKit's `DefaultParameters::getMMFF*()` collections are parallel lookup columns searched by a
 *  cascade of `std::equal_range` calls. The POD entry structs and `lookup_*` cascades here hold the
 *  same data in flat arrays that upload with one copy each and binary-search on the device, and every
 *  cascade answers byte-identically to RDKit's `operator()` — including the four- and five-level
 *  `MMFFDef` equivalence fallbacks of angle, torsion and out-of-plane.
 *  `test/parameters/test_mmff94_tables.cpp` gates that against RDKit over 10K random key tuples.
 *
 *  Doubles become floats at extraction time: the ETKDG kernel consumes float32 and the extra
 *  precision sits below the tolerances required.
 */

#pragma once

#include <cstdint> // `std::uint8_t`

#include <utility> // `std::pair`
#include <vector>  // `std::vector`

#include "smallfield/types/views.hpp"
#include "smallfield/utilities.hpp"

/*  `SF_HOST_DEVICE_` is `__host__ __device__` under nvcc and empty otherwise. The `lookup_*` helpers
 *  below reach device code as `constexpr` instead; the macro serves the MMFF94 sources that include
 *  this header. */
#if defined(__CUDACC__)
#define SF_HOST_DEVICE_ __host__ __device__
#else
#define SF_HOST_DEVICE_
#endif

namespace smallfield::parameters {

#pragma region Table Entry Structs

/** Van-der-Waals parameters, one entry per MMFF atom type. Same payload as
 *  `mmff94_atom_vdw_params_t` in `types/etkdg_types.hpp`, carrying the `atom_type` it is keyed by. */
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

/** Out-of-plane bend parameter, MMFF `koop`, keyed by (j_type, i_type, k_type, l_type). */
struct alignas(8) mmff94_oop_entry_t {
    std::uint8_t j_type;
    /** Canonical: i_type <= k_type <= l_type. */
    std::uint8_t i_type;
    std::uint8_t k_type;
    std::uint8_t l_type;
    /** MMFF `koop`, in mdyn·angstrom per rad², against the Wilson out-of-plane angle. */
    float force_constant;
};
static_assert(sizeof(mmff94_oop_entry_t) == 8);

/** Stretch-bend parameter, keyed by (j_type, i_type, k_type, stretch_bend_type). */
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

/** Default-stretch-bend fallback parameter, keyed by the periodic-table rows of i, j and k. RDKit reads
 *  it when the primary Stbn lookup misses. */
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

/** MMFFProp per-atom-type chemical properties. Dense array indexed directly by `atom_type`, one slot
 *  per possible type with zeros for unused slots. */
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

/** Equivalence levels MMFFDef tabulates per atom type, and the stage count of every cascade keyed on
 *  them: level 0 is the type itself and level 3 the wildcard. */
inline constexpr unsigned mmff94_equivalence_levels_k = 4;

/** MMFFDef equivalence levels, one byte per level. Dense array indexed directly by `atom_type`. */
struct mmff94_def_entry_t {
    std::uint8_t eq_level[mmff94_equivalence_levels_k];
};
static_assert(sizeof(mmff94_def_entry_t) == mmff94_equivalence_levels_k);

/** Partial-bond-charge-increment per atom type. */
struct alignas(8) mmff94_pbci_entry_t {
    std::uint8_t atom_type;
    std::uint8_t padding[3];
    float pbci;
    float fcadj;
};
static_assert(sizeof(mmff94_pbci_entry_t) == 16);

/** Bond-charge-increment parameter, keyed by (i_type, j_type, bond_type). The increment is
 *  @b directional: atom i's partial charge gains it and atom j loses it. */
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

#pragma region VdW Collection Scalars

/*  The `MMFFVdWCollection` scalars that belong to the collection rather than to one atom type. They are
 *  the same on every molecule, so the nonbonded kernel compiles them in rather than reading them from the
 *  uploaded view; `test/parameters/test_mmff94_tables.cpp` compares each against RDKit's live collection,
 *  which is what holds a compiled-in constant to the parameter file it was read from.
 *
 *  MMFF `power`, the exponent `alpha_i` is raised to, has no consumer: `r_star` arrives already combined
 *  from `MMFFVdW::R_star`, so nothing here ever raises `alpha_i` to anything. */

/** MMFF `B`, the amplitude of the R* combination rule's asymmetry correction. */
constexpr float mmff94_vdw_asymmetry_amplitude_k = 0.2f;
/** MMFF `Beta`, how fast that correction decays with the asymmetry. */
constexpr float mmff94_vdw_asymmetry_decay_k = 12.0f;
/** MMFF `DARAD`, the R* scale a donor-acceptor pair takes. */
constexpr float mmff94_vdw_donor_acceptor_radius_scale_k = 0.8f;
/** MMFF `DAEPS`, the well-depth scale a donor-acceptor pair takes. */
constexpr float mmff94_vdw_donor_acceptor_well_depth_scale_k = 0.5f;

#pragma endregion VdW Collection Scalars

#pragma region Table View

/**
 *  @brief POD view over every extracted MMFF table, passed by value into kernels.
 *
 *  On the host each span points into the `std::vector`s of `mmff94_tables_host_t`; on the device into
 *  the per-table buffers `upload_mmff94_tables_to_device` fills.
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

    std::uint8_t max_atom_type = 0;
};

/*  The upload in `mmff94_tables_device.hpp` writes these fields and every MMFF kernel takes the struct
 *  by value, so a change in the layout silently redirects every parameter lookup on the device. */
static_assert(sizeof(mmff94_tables_view_t) == 232);

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

    /** Highest atom type seen, for `MMFFDef` indexing. */
    std::uint8_t max_atom_type = 0;

    /** A read-only view over this container's own storage, which must outlive it. Each assignment is
     *  `custd::span`'s range constructor, so the extent comes from the vector rather than restated. */
    mmff94_tables_view_t view() const noexcept {
        mmff94_tables_view_t result;
        result.vdw = vdw;
        result.bond = bond;
        result.angle = angle;
        result.torsion = torsion;
        result.oop = oop;
        result.stbn = stbn;
        result.dfsb = dfsb;
        result.def = def;
        result.prop = prop;
        result.pbci = pbci;
        result.chg = chg;
        result.bndk = bndk;
        result.hl = hl;
        result.covrad = covrad;
        result.max_atom_type = max_atom_type;
        return result;
    }
};

#pragma endregion Table Container

#pragma region Sorted Column Search

/**
 *  @brief Hand-rolled `lower_bound` over one sorted table column, used by every lookup cascade below.
 *
 *  Not `std::lower_bound`: libstdc++'s reaches device code only through `--expt-relaxed-constexpr`,
 *  and along that route it stops the search early — past about 8 entries of a 95-element sorted array
 *  it declares the key not found. A local definition keeps host and device on one code path.
 *
 *  @p key is `unsigned` so the `key + 1` the range search forms cannot wrap at atom type 255.
 */
template <typename element_, typename key_accessor_>
constexpr std::size_t lower_bound_(custd::span<element_ const> sorted, unsigned key, key_accessor_ &&key_of) noexcept {
    std::size_t lower = 0, upper = sorted.size();
    while (lower < upper) {
        std::size_t const middle = lower + (upper - lower) / 2;
        if (static_cast<unsigned>(key_of(sorted[middle])) < key) lower = middle + 1;
        else upper = middle;
    }
    return lower;
}

/** Narrow a sorted column, or an already-narrowed run of one, to the run whose key equals @p key.
 *  Empty when the key is absent. */
template <typename element_, typename key_accessor_>
constexpr custd::span<element_ const> equal_range_(custd::span<element_ const> sorted, unsigned key,
                                                   key_accessor_ &&key_of) noexcept {
    custd::span<element_ const> const tail = sorted.subspan(lower_bound_(sorted, key, key_of));
    return tail.first(lower_bound_(tail, key + 1, key_of));
}

/** The one entry of a sorted column whose key equals @p key, which every cascade ends on, or
 *  `nullptr` on a miss. */
template <typename element_, typename key_accessor_>
constexpr element_ const *find_sorted_(custd::span<element_ const> sorted, unsigned key,
                                       key_accessor_ &&key_of) noexcept {
    std::size_t const index = lower_bound_(sorted, key, key_of);
    if (index == sorted.size() || static_cast<unsigned>(key_of(sorted[index])) != key) return nullptr;
    return &sorted[index];
}

/**
 *  @brief One MMFFDef equivalence level of @p atom_type, or 0 where the dense table does not reach.
 *
 *  @p level is 0 for the type itself and 3 for the wildcard, but RDKit's torsion cascade has one stage
 *  that asks for level 4 and reads one past the entry, landing on the next atom type's level 0. This
 *  port has to answer identically, so the levels are walked as the flat dense run they physically are
 *  rather than by indexing past an array. The invariant no type carries is that
 *  `mmff94_tables_host_t::def` is dense and contiguous in atom type, which its extraction guarantees.
 */
constexpr std::uint8_t equivalent_atom_type_(mmff94_tables_view_t const &tables, std::uint8_t atom_type,
                                             unsigned level) noexcept {
    std::size_t const entry = atom_type + level / mmff94_equivalence_levels_k;
    if (entry >= tables.def.size()) return 0;
    return tables.def[entry].eq_level[level % mmff94_equivalence_levels_k];
}

#pragma endregion Sorted Column Search

/** Binary-search cascade matching `MMFFVdWCollection::operator()`. */
constexpr mmff94_vdw_entry_t const *lookup_vdw(mmff94_tables_view_t const &tables, std::uint8_t atom_type) noexcept {
    return find_sorted_(tables.vdw, atom_type, [](mmff94_vdw_entry_t const &entry) { return entry.atom_type; });
}

/** Binary-search cascade matching `MMFFBondCollection::operator()`. Canonicalises (atom_i, atom_j) to
 *  (lower_type, upper_type). */
constexpr mmff94_bond_entry_t const *lookup_bond(mmff94_tables_view_t const &tables, std::uint8_t bond_type,
                                                 std::uint8_t atom_i, std::uint8_t atom_j) noexcept {
    std::uint8_t lower_type = atom_i, upper_type = atom_j;
    if (lower_type > upper_type) swap_(lower_type, upper_type);

    auto const by_i = equal_range_(tables.bond, lower_type,
                                   [](mmff94_bond_entry_t const &entry) { return entry.i_type; });
    auto const by_j = equal_range_(by_i, upper_type, [](mmff94_bond_entry_t const &entry) { return entry.j_type; });
    return find_sorted_(by_j, bond_type, [](mmff94_bond_entry_t const &entry) { return entry.bond_type; });
}

/** Binary-search cascade matching `MMFFAngleCollection::operator()` — including the 4-level
 *  equivalence fallback via `MMFFDef`. */
constexpr mmff94_angle_entry_t const *lookup_angle(mmff94_tables_view_t const &tables, std::uint8_t angle_type,
                                                   std::uint8_t atom_i, std::uint8_t atom_j_center,
                                                   std::uint8_t atom_k) noexcept {
    auto const j_key = [](mmff94_angle_entry_t const &entry) { return entry.j_type; };
    auto const i_key = [](mmff94_angle_entry_t const &entry) { return entry.i_type; };
    auto const k_key = [](mmff94_angle_entry_t const &entry) { return entry.k_type; };
    auto const angle_type_key = [](mmff94_angle_entry_t const &entry) { return entry.angle_type; };
    for (unsigned level = 0; level < mmff94_equivalence_levels_k; ++level) {
        std::uint8_t canonical_i_type = equivalent_atom_type_(tables, atom_i, level);
        std::uint8_t canonical_k_type = equivalent_atom_type_(tables, atom_k, level);
        if (canonical_i_type > canonical_k_type) swap_(canonical_i_type, canonical_k_type);

        auto const by_j = equal_range_(tables.angle, atom_j_center, j_key);
        auto const by_i = equal_range_(by_j, canonical_i_type, i_key);
        auto const by_k = equal_range_(by_i, canonical_k_type, k_key);
        if (auto const *entry = find_sorted_(by_k, angle_type, angle_type_key); entry) return entry;
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
    return find_sorted_(tables.pbci, atom_type, [](mmff94_pbci_entry_t const &entry) { return entry.atom_type; });
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
    auto const by_i = equal_range_(tables.chg, lower_type,
                                   [](mmff94_chg_entry_t const &entry) { return entry.i_type; });
    auto const by_j = equal_range_(by_i, upper_type, [](mmff94_chg_entry_t const &entry) { return entry.j_type; });
    return {sign, find_sorted_(by_j, bond_type, [](mmff94_chg_entry_t const &entry) { return entry.bond_type; })};
}

/** Bndk lookup — the default bond keyed by atomic-number pair. */
constexpr mmff94_bndk_entry_t const *lookup_bndk(mmff94_tables_view_t const &tables, std::uint8_t atomic_i,
                                                 std::uint8_t atomic_j) noexcept {
    std::uint8_t lower_atomic_num = atomic_i, upper_atomic_num = atomic_j;
    if (lower_atomic_num > upper_atomic_num) swap_(lower_atomic_num, upper_atomic_num);
    auto const by_i = equal_range_(tables.bndk, lower_atomic_num,
                                   [](mmff94_bndk_entry_t const &entry) { return entry.atomic_num_i; });
    return find_sorted_(by_i, upper_atomic_num, [](mmff94_bndk_entry_t const &entry) { return entry.atomic_num_j; });
}

/** Covalent radius + Pauling electronegativity lookup by atomic number. */
constexpr mmff94_covrad_entry_t const *lookup_covrad(mmff94_tables_view_t const &tables,
                                                     std::uint8_t atomic_num) noexcept {
    return find_sorted_(tables.covrad, atomic_num, [](mmff94_covrad_entry_t const &entry) { return entry.atomic_num; });
}

/** Herschbach-Laurie lookup by periodic-table-row pair. */
constexpr mmff94_hl_entry_t const *lookup_hl(mmff94_tables_view_t const &tables, std::uint8_t row_i,
                                             std::uint8_t row_j) noexcept {
    std::uint8_t lower_row = row_i, upper_row = row_j;
    if (lower_row > upper_row) swap_(lower_row, upper_row);
    auto const by_i = equal_range_(tables.hl, lower_row, [](mmff94_hl_entry_t const &entry) { return entry.row_i; });
    return find_sorted_(by_i, upper_row, [](mmff94_hl_entry_t const &entry) { return entry.row_j; });
}

/** OOP bend lookup with 4-stage equivalence cascade. Matches `MMFFOopCollection::operator()` exactly. */
constexpr mmff94_oop_entry_t const *lookup_oop(mmff94_tables_view_t const &tables, std::uint8_t atom_i,
                                               std::uint8_t atom_j, std::uint8_t atom_k, std::uint8_t atom_l) noexcept {
    auto const j_key = [](mmff94_oop_entry_t const &entry) { return entry.j_type; };
    auto const i_key = [](mmff94_oop_entry_t const &entry) { return entry.i_type; };
    auto const k_key = [](mmff94_oop_entry_t const &entry) { return entry.k_type; };
    auto const l_key = [](mmff94_oop_entry_t const &entry) { return entry.l_type; };
    for (unsigned level = 0; level < mmff94_equivalence_levels_k; ++level) {
        // The MMFF out-of-plane spec letters i, k, l after the sorting network below.
        std::uint8_t canonical_ikl_types[3] = {
            equivalent_atom_type_(tables, atom_i, level),
            equivalent_atom_type_(tables, atom_k, level),
            equivalent_atom_type_(tables, atom_l, level),
        };
        if (canonical_ikl_types[0] > canonical_ikl_types[1]) swap_(canonical_ikl_types[0], canonical_ikl_types[1]);
        if (canonical_ikl_types[1] > canonical_ikl_types[2]) swap_(canonical_ikl_types[1], canonical_ikl_types[2]);
        if (canonical_ikl_types[0] > canonical_ikl_types[1]) swap_(canonical_ikl_types[0], canonical_ikl_types[1]);
        auto const by_j = equal_range_(tables.oop, atom_j, j_key);
        auto const by_i = equal_range_(by_j, canonical_ikl_types[0], i_key);
        auto const by_k = equal_range_(by_i, canonical_ikl_types[1], k_key);
        if (auto const *entry = find_sorted_(by_k, canonical_ikl_types[2], l_key); entry) return entry;
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

    auto const by_j = equal_range_(tables.stbn, atom_j_center,
                                   [](mmff94_stbn_entry_t const &entry) { return entry.j_type; });
    auto const by_i = equal_range_(by_j, lower_type, [](mmff94_stbn_entry_t const &entry) { return entry.i_type; });
    auto const by_k = equal_range_(by_i, upper_type, [](mmff94_stbn_entry_t const &entry) { return entry.k_type; });
    return {swap, find_sorted_(by_k, stretch_bend_type,
                               [](mmff94_stbn_entry_t const &entry) { return entry.stretch_bend_type; })};
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
    auto const by_row_i = equal_range_(tables.dfsb, lower_row,
                                       [](mmff94_dfsb_entry_t const &entry) { return entry.row_i; });
    auto const by_row_j = equal_range_(by_row_i, row_j, [](mmff94_dfsb_entry_t const &entry) { return entry.row_j; });
    return {swap, find_sorted_(by_row_j, upper_row, [](mmff94_dfsb_entry_t const &entry) { return entry.row_k; })};
}

/**
 *  @brief MMFFDef levels the torsion cascade reads for the i and l ends at one stage.
 *
 *  The five stages combine levels 1-1-1-1, 2-2-2-2, 3-2-2-5, 5-2-2-3 and 5-2-2-5 in MMFF.IV's numbering,
 *  so stages 1 and 2 wildcard one end while holding the other and the two levels are not simply the
 *  stage number.
 */
struct torsion_wildcard_levels_t {
    unsigned i_level;
    unsigned l_level;
};

/** The i and l equivalence levels of one torsion cascade stage. */
constexpr torsion_wildcard_levels_t torsion_wildcard_levels_(unsigned stage) noexcept {
    if (stage == 1) return {.i_level = 1, .l_level = 3};
    if (stage == 2) return {.i_level = 3, .l_level = 1};
    return {.i_level = stage, .l_level = stage};
}

/** One stage of the torsion cascade: five binary searches on j → k → i → l → tor_type. */
constexpr mmff94_torsion_entry_t const *find_torsion_entry_(
    mmff94_tables_view_t const &tables, std::uint8_t canonical_i_type, std::uint8_t canonical_j_type,
    std::uint8_t canonical_k_type, std::uint8_t canonical_l_type, unsigned canonical_torsion_type) noexcept {
    auto const by_j = equal_range_(tables.torsion, canonical_j_type,
                                   [](mmff94_torsion_entry_t const &entry) { return entry.j_type; });
    auto const by_k = equal_range_(by_j, canonical_k_type,
                                   [](mmff94_torsion_entry_t const &entry) { return entry.k_type; });
    auto const by_i = equal_range_(by_k, canonical_i_type,
                                   [](mmff94_torsion_entry_t const &entry) { return entry.i_type; });
    auto const by_l = equal_range_(by_i, canonical_l_type,
                                   [](mmff94_torsion_entry_t const &entry) { return entry.l_type; });
    return find_sorted_(by_l, canonical_torsion_type,
                        [](mmff94_torsion_entry_t const &entry) { return entry.tor_type; });
}

/** Torsion lookup with the 5-stage equivalence cascade and the fallback to the secondary torsion
 *  type, matching `MMFFTorCollection::getMMFFTorParams`. Returns the resolved canonical torsion type
 *  beside the entry, which is null on a miss. */
constexpr std::pair<unsigned, mmff94_torsion_entry_t const *> lookup_torsion(
    mmff94_tables_view_t const &tables, std::uint8_t tor_type_primary, std::uint8_t tor_type_secondary,
    std::uint8_t atom_i, std::uint8_t atom_j, std::uint8_t atom_k, std::uint8_t atom_l) noexcept {
    mmff94_torsion_entry_t const *hit = nullptr;
    unsigned canonical_torsion_type = tor_type_primary;
    unsigned stage = 0;
    unsigned max_stages = 5;
    while (((stage < max_stages) && ((!hit) || (max_stages == 4))) ||
           ((stage == 4) && (tor_type_primary == 5) && tor_type_secondary)) {
        // Secondary fallback: after the 5-stage attempt with the primary tor type, retry the first
        // 4 stages with the secondary tor type.
        if ((max_stages == 5) && (stage == 4)) {
            max_stages = 4;
            stage = 0;
            canonical_torsion_type = tor_type_secondary;
        }
        auto const [i_level, l_level] = torsion_wildcard_levels_(stage);
        std::uint8_t canonical_i_type = equivalent_atom_type_(tables, atom_i, i_level);
        std::uint8_t canonical_j_type = atom_j;
        std::uint8_t canonical_k_type = atom_k;
        std::uint8_t canonical_l_type = equivalent_atom_type_(tables, atom_l, l_level);
        if (canonical_j_type > canonical_k_type) {
            swap_(canonical_j_type, canonical_k_type);
            swap_(canonical_i_type, canonical_l_type);
        }
        else if (canonical_j_type == canonical_k_type && canonical_i_type > canonical_l_type) {
            swap_(canonical_i_type, canonical_l_type);
        }

        if (auto const *entry = find_torsion_entry_(tables, canonical_i_type, canonical_j_type, canonical_k_type,
                                                    canonical_l_type, canonical_torsion_type);
            entry) {
            hit = entry;
            if (max_stages == 4) break;
        }
        ++stage;
    }
    return {canonical_torsion_type, hit};
}

} // namespace smallfield::parameters
