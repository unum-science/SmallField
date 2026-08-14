/**
 *  @file src/smallfield/preprocess/bounds_matrix_builder.hpp
 *  @brief Native distance-bounds matrix construction, from a flat descriptor and nothing else.
 *
 *  Replaces RDKit's `setTopolBounds`, which allocates a fresh `DistGeom::BoundsMatrix` plus two
 *  `boost::shared_array<>` per molecule and walks RDKit's atom typer, roughly 140 KB of throwaway heap
 *  per call. This reads only `molecule_descriptor_t` and the element tables in the sibling
 *  `bounds_matrix_tables.hpp`, and writes straight into caller-owned storage.
 *
 *  The 1-4 half is the substance. A pair three bonds apart is bounded by the case its path falls into
 *  rather than by one envelope: a stereo-defined double bond pins it, an amide holds it trans, a ring
 *  closure pins it cis, and a disulfide sits near a right angle. Ported from RDKit's `Type14` dispatch
 *  in `Code/GraphMol/DistGeomHelpers/BoundsMatrixBuilder.cpp`, re-expressed rather than transliterated:
 *  their `aid1`..`aid4` are positions along a `path_14_t`, their `bool preferTrans` and
 *  `bool forceTransAmides` are named enumerations, and their `merge` of a heap vector is a sweep over
 *  caller-owned scratch.
 *
 *  A pair reachable by several paths collects one interval per path, so nothing depends on which path is
 *  found first. The 1-4 sweep collapses its claims through `merge_intervals_`; the 1-3 sweep widens the
 *  pair to the union of what its shared centers ask, which is the rule `_checkAndSetBounds` applies
 *  upstream and not the same rule as `merge_intervals_` — that one intersects claims that overlap.
 */

#pragma once

#include <cmath>   // `std::cos`, `std::sin`, `std::sqrt`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <algorithm> // `std::max`, `std::min`, `std::sort`, `std::swap`
#include <optional>  // `std::optional`

#include "smallfield/preprocess/bounds_matrix_tables.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::preprocess {

using smallfield::types::atom_flag_hbond_acceptor_k;
using smallfield::types::atom_flag_hbond_donor_h_k;
using smallfield::types::bond_flag_aromatic_k;
using smallfield::types::bond_flag_order_aromatic_k;
using smallfield::types::bond_flag_stereo_e_k;
using smallfield::types::bond_flag_stereo_z_k;
using smallfield::types::fill_bond_distances;
using smallfield::types::hop_matrix_t;
using smallfield::types::lower_distance_bound_at;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::max_bonds_k;
using smallfield::types::molecule_descriptor_t;
using smallfield::types::no_stereo_atom_k;
using smallfield::types::packed_matrix;
using smallfield::types::upper_distance_bound_at;

#pragma region Path Vocabulary

/**
 *  @brief Which case a 1-4 path falls into, since the case is what fixes its central torsion.
 *
 *  Ported from RDKit's `Type14` in `Code/GraphMol/DistGeomHelpers/BoundsMatrixBuilder.cpp`. The three
 *  bonds of a path are `first — second`, `second — third` and `third — fourth`, and the ring membership
 *  of those three is the whole classification.
 */
enum class path_14_kind_t : std::uint8_t {
    /** The middle bond is not a ring bond and neither end bond is either. */
    in_chain_k,
    /** All three bonds are consecutive in one ring, so the ring closure fixes the torsion. */
    all_in_ring_k,
    /** Two consecutive bonds share a ring and the third leaves it. */
    two_in_same_ring_k,
    /** The middle bond and an end bond are both ring bonds, of rings that do not coincide. */
    two_in_different_rings_k,
    /** The middle bond is a ring bond and neither end bond shares a ring with it. */
    shares_ring_bond_k,
    /** `two_in_same_ring_k` where the middle bond belongs to a macrocycle. */
    macrocycle_two_in_same_ring_k,
    /** `all_in_ring_k` where that ring is a macrocycle. */
    macrocycle_all_in_ring_k,
};

/** What a path's case asks of its central torsion. */
enum class torsion_preference_t : std::uint8_t {
    /** Torsion 0, so the two ends eclipse. */
    cis_k,
    /** Torsion pi, so the two ends are anti. */
    trans_k,
    /** Anything between cis and trans, which is an interval rather than a value. */
    flexible_k,
    /** Torsion pi/2, which only a disulfide asks for. */
    right_angle_k,
    /** The path declines to say, so the pair takes no interval from it. */
    none_k,
};

/** The torsion a path's case asks for, and the slack that case adds on top. */
struct torsion_target_t {
    torsion_preference_t preference = torsion_preference_t::flexible_k;
    /** Extra reach in Å on top of the computed distance; only a macrocyclic trans amide asks for it. */
    float extra_reach = 0.0f;
};

/** The E/Z relationship a central bond fixes between a path's two end atoms. */
enum class path_stereo_t : std::uint8_t { cis_k, trans_k, unspecified_k };

/** Whether a flat ring holds the substituent leaving it anti to the path's other end. */
enum class exocyclic_preference_t : std::uint8_t { unbiased_k, prefers_trans_k };

/** Bond order as the 1-4 classification reads it; an aromatic bond is neither single nor double. */
enum class bond_kind_t : std::uint8_t { single_k, double_k, other_k };

/** Interval bounds a single 1-4 path claims for its end-to-end distance, in Å. */
struct bound_interval_t {
    float lower = 0.0f;
    float upper = 0.0f;
};

/** The four atoms and three bonds of one 1-4 path, named by position along it. */
struct path_14_t {
    std::uint32_t first = 0;
    std::uint32_t second = 0;
    std::uint32_t third = 0;
    std::uint32_t fourth = 0;
    std::uint32_t bond_first_second = 0;
    std::uint32_t bond_second_third = 0;
    std::uint32_t bond_third_fourth = 0;
};

/** The case a path falls into, with the ring size and the exocyclic bias that case reads. */
struct path_14_case_t {
    path_14_kind_t kind = path_14_kind_t::in_chain_k;
    /** Size of the ring all three bonds lie in; 0 when they do not share one. */
    std::uint32_t ring_size = 0;
    exocyclic_preference_t exocyclic = exocyclic_preference_t::unbiased_k;
};

/** Smallest ring RDKit's `useMacrocycle14config` treats as a macrocycle, in atoms. ETKDGv3 leaves it on. */
inline constexpr std::uint32_t macrocycle_ring_size_k = 9;

/** Largest ring whose sp2 pair still pins its 1-4 path cis; past it there is room for trans. */
inline constexpr std::uint32_t largest_pinned_ring_size_k = 8;

/** Half-width around a torsion-determined 1-4 distance. RDKit's `GEN_DIST_TOL`. */
inline constexpr float one_four_tolerance_k = 0.06f;

/** Interval width below which a flexible 1-4 pair is widened instead. RDKit's `DIST12_DELTA`. */
inline constexpr float degenerate_interval_k = 0.01f;

/** Extra reach a macrocyclic trans amide takes, the most that survives triangle smoothing. */
inline constexpr float macrocycle_amide_reach_k = 0.1f;

/** Lower bound in Å for a pair that can close an intramolecular hydrogen bond. */
inline constexpr float hydrogen_bond_length_k = 1.8f;

/**
 *  @brief The most 1-4 paths one atom pair collects, past which the remaining paths are dropped.
 *
 *  A pair's path count is the product of the two end atoms' degrees, and a hypervalent centre in the
 *  elements MMFF94 parametrises carries at most six bonds, so thirty-six is the chemical ceiling and
 *  the buffer clears it. The cap exists because the claims sit on this frame rather than on the heap.
 */
inline constexpr std::uint32_t max_paths_per_pair_k = 64;

#pragma endregion Path Vocabulary

#pragma region Graph Queries

/** Index of the bond joining @p atom_i and @p atom_j, or empty when they are not bonded. */
inline std::optional<std::uint32_t> bond_between_(molecule_descriptor_t const &descriptor, std::uint32_t atom_i,
                                                  std::uint32_t atom_j) {
    std::uint32_t const first_entry = descriptor.neighbor_offset[atom_i];
    std::uint32_t const last_entry = descriptor.neighbor_offset[atom_i + 1];
    for (std::uint32_t entry = first_entry; entry < last_entry; ++entry)
        if (descriptor.neighbor_atom[entry] == atom_j) return descriptor.neighbor_bond[entry];
    return std::nullopt;
}

/** Size of the smallest SSSR ring named in @p ring_mask; 0 when the mask names none. */
inline std::uint32_t smallest_ring_size_(molecule_descriptor_t const &descriptor, std::uint32_t ring_mask) {
    std::uint32_t smallest = 0;
    for (std::uint32_t ring = 0; ring < descriptor.num_rings; ++ring) {
        if ((ring_mask & (1u << ring)) == 0) continue;
        std::uint32_t const ring_size = descriptor.ring_offset[ring + 1] - descriptor.ring_offset[ring];
        if (smallest == 0 || ring_size < smallest) smallest = ring_size;
    }
    return smallest;
}

/**
 *  @brief The angle a ring atom has left for a pair that reaches outside the ring.
 *
 *  RDKit assigns ring angles first and then divides what remains of the center's budget among the
 *  pairs that did not get one, `BoundsMatrixBuilder.cpp:629-641`. At an sp2 center the budget is a
 *  full turn, so a three-ring taking 60° leaves 150° for each of the two exocyclic pairs — not the
 *  120° the hybridization alone would say. An sp3 center is not a budget at all upstream, just two
 *  literals for the two strained ring sizes.
 *
 *  The remainder is recomputed here rather than accumulated in a per-atom table: at degree three the
 *  leftover is uniquely determined, so this is exact where it matters, and a bond-by-bond matrix at
 *  `max_bonds_k` would put half a megabyte on a frame that already carries the hop matrix.
 *
 *  @param[in] descriptor Molecule supplying adjacency, ring membership and hybridization.
 *  @param[in] center Ring atom the angle is measured at.
 *  @param[in] center_ring_size Smallest ring the center belongs to.
 *  @return Angle in radians.
 */
inline float exocyclic_angle_radians_(molecule_descriptor_t const &descriptor, std::uint32_t center,
                                      std::uint32_t center_ring_size) {
    std::uint8_t const hybridization = descriptor.hybridization[center];
    if (hybridization == hybridization_sp3_k) {
        if (center_ring_size == 3) return sp3_in_three_ring_angle_radians_k;
        if (center_ring_size == 4) return sp3_in_four_ring_angle_radians_k;
        return tetrahedral_angle_radians_k;
    }
    if (hybridization != hybridization_sp2_k) return ideal_angle_radians(hybridization);

    // Every neighbour pair sharing a ring with the center has already been answered by the closure;
    // the rest divide the turn that is left.
    std::uint16_t const neighbor_first = descriptor.neighbor_offset[center];
    std::uint16_t const neighbor_last = descriptor.neighbor_offset[center + 1];
    float spent = 0.0f;
    std::uint32_t ring_pairs = 0, total_pairs = 0;
    for (std::uint16_t left = neighbor_first; left < neighbor_last; ++left)
        for (std::uint16_t right = left + 1; right < neighbor_last; ++right) {
            ++total_pairs;
            std::uint32_t const pair_shared = descriptor.atom_ring_bitmap[descriptor.neighbor_atom[left]] &
                                              descriptor.atom_ring_bitmap[center] &
                                              descriptor.atom_ring_bitmap[descriptor.neighbor_atom[right]];
            std::uint32_t const pair_ring_size = smallest_ring_size_(descriptor, pair_shared);
            if (pair_ring_size == 0) continue;
            ++ring_pairs;
            spent += ring_interior_angle_radians(hybridization, pair_ring_size);
        }

    std::uint32_t const open_pairs = total_pairs - ring_pairs;
    if (open_pairs == 0 || spent >= full_turn_radians_k) return ideal_angle_radians(hybridization);
    return (full_turn_radians_k - spent) / static_cast<float>(open_pairs);
}

/**
 *  @brief How many of a 1-3 triple's atoms are heavy sp2 ring atoms, each doubling the window.
 *
 *  `isLargerSP2Atom` upstream: past the first two rows of the periodic table, sp2, and in a ring.
 *  A thiophene or furan 1-3 spans atoms whose geometry the polygon interior angle describes poorly,
 *  and doubling per such atom is upstream declining to assert a window it cannot justify — dropping
 *  it while adopting `DIST13_TOL` would manufacture infeasibility exactly where RDKit loosens.
 *
 *  @param[in] descriptor Molecule supplying element, hybridization and ring membership.
 *  @param[in] first First atom of the 1-3 triple.
 *  @param[in] center Shared neighbour.
 *  @param[in] second Third atom of the triple.
 *  @return Count in [0, 3]; the caller shifts the tolerance left by it.
 */
inline std::uint32_t larger_sp2_ring_atoms_(molecule_descriptor_t const &descriptor, std::uint32_t first,
                                            std::uint32_t center, std::uint32_t second) {
    auto is_larger_sp2_ring_atom = [&](std::uint32_t atom) {
        return descriptor.element[atom] > 13 && descriptor.hybridization[atom] == hybridization_sp2_k &&
               descriptor.atom_ring_bitmap[atom] != 0;
    };
    return static_cast<std::uint32_t>(is_larger_sp2_ring_atom(first)) +
           static_cast<std::uint32_t>(is_larger_sp2_ring_atom(center)) +
           static_cast<std::uint32_t>(is_larger_sp2_ring_atom(second));
}

/** Whether any SSSR ring named in @p ring_mask is a macrocycle. */
inline bool ring_mask_holds_macrocycle_(molecule_descriptor_t const &descriptor, std::uint32_t ring_mask) {
    for (std::uint32_t ring = 0; ring < descriptor.num_rings; ++ring) {
        if ((ring_mask & (1u << ring)) == 0) continue;
        if (descriptor.ring_offset[ring + 1] - descriptor.ring_offset[ring] >= macrocycle_ring_size_k) return true;
    }
    return false;
}

/** Ideal length in Å of @p bond, read from the element pair and the bond type. */
inline float bond_length_angstrom_(molecule_descriptor_t const &descriptor, std::uint32_t bond) {
    covalent_bond_t const covalent = (descriptor.bond_flags[bond] & bond_flag_aromatic_k) != 0
                                         ? covalent_bond_t::aromatic_k
                                     : descriptor.bond_order[bond] == 2 ? covalent_bond_t::double_k
                                     : descriptor.bond_order[bond] == 3 ? covalent_bond_t::triple_k
                                                                        : covalent_bond_t::single_k;
    return ideal_bond_length_angstrom(descriptor.element[descriptor.bond_atom_i[bond]],
                                      descriptor.element[descriptor.bond_atom_j[bond]], covalent);
}

/**
 *  @brief The 1-3 angle at @p center between its neighbours @p first and @p second.
 *
 *  A ring the three atoms all belong to fixes the angle whatever the centre's hybridization prefers,
 *  so the smallest such ring decides and hybridization answers only for an open chain.
 */
inline float interior_angle_radians_(molecule_descriptor_t const &descriptor, std::uint32_t first, std::uint32_t center,
                                     std::uint32_t second) {
    std::uint32_t const shared = descriptor.atom_ring_bitmap[first] & descriptor.atom_ring_bitmap[center] &
                                 descriptor.atom_ring_bitmap[second];
    std::uint32_t const shared_ring_size = smallest_ring_size_(descriptor, shared);
    if (shared_ring_size != 0) return ring_interior_angle_radians(descriptor.hybridization[center], shared_ring_size);

    // The pair is not in a ring, but the center may still be: an exocyclic substituent on a ring
    // atom. The ring closure has already spent part of the center's angle budget, and what is left
    // belongs to the pairs that reach outside — so the free-atom answer is wrong by however much the
    // ring took. At an sp2 center in a three-ring it is wrong by thirty degrees.
    std::uint32_t const center_ring_size = smallest_ring_size_(descriptor, descriptor.atom_ring_bitmap[center]);
    if (center_ring_size == 0) return ideal_angle_radians(descriptor.hybridization[center]);
    return exocyclic_angle_radians_(descriptor, center, center_ring_size);
}

/** How the 1-4 classification reads @p bond's order. */
inline bond_kind_t bond_kind_(molecule_descriptor_t const &descriptor, std::uint32_t bond) {
    if ((descriptor.bond_flags[bond] & bond_flag_order_aromatic_k) != 0) return bond_kind_t::other_k;
    if (descriptor.bond_order[bond] == 2) return bond_kind_t::double_k;
    if (descriptor.bond_order[bond] == 1) return bond_kind_t::single_k;
    return bond_kind_t::other_k;
}

/** The same path walked from the other end, so a directional predicate is asked in both directions. */
inline path_14_t reversed_(path_14_t const &path) {
    return {.first = path.fourth,
            .second = path.third,
            .third = path.second,
            .fourth = path.first,
            .bond_first_second = path.bond_third_fourth,
            .bond_second_third = path.bond_second_third,
            .bond_third_fourth = path.bond_first_second};
}

/**
 *  @brief End-to-end distance of a 1-4 chain whose central torsion has cosine @p cos_torsion.
 *
 *  RDKit's `RDGeom::compute14Dist3D`, written out as one law-of-cosines expression rather than as a
 *  point rotation. `+1` is cis and `-1` is trans, which is how a stereo-defined double bond reaches
 *  the bounds matrix at all.
 */
inline float one_four_distance_(float length_12, float length_23, float length_34, float angle_123, float angle_234,
                                float cos_torsion) {
    float const cos_123 = std::cos(angle_123), sin_123 = std::sin(angle_123);
    float const cos_234 = std::cos(angle_234), sin_234 = std::sin(angle_234);
    float const squared = length_12 * length_12 + length_23 * length_23 + length_34 * length_34 -
                          2.0f * length_12 * length_23 * cos_123 - 2.0f * length_23 * length_34 * cos_234 +
                          2.0f * length_12 * length_34 * (cos_123 * cos_234 - sin_123 * sin_234 * cos_torsion);
    return std::sqrt(std::max(0.01f, squared));
}

#pragma endregion Graph Queries

#pragma region Chemical Predicates

/*
 *  The nitrogen these predicates accept carries exactly one hydrogen, which is a secondary amide.
 *  That is not a statement about resonance -- a tertiary amide is every bit as flat -- but about what
 *  a bounds matrix can hold. Each 1-4 pair owns one interval, and `merge_intervals_` reconciles only
 *  the claims made on the same pair, so a preference is expressible here only when the two ends can be
 *  told apart. A secondary amide's nitrogen offers a hydrogen and a carbon, `ends_on_secondary_amide_hydrogen_`
 *  sends one anti and the other syn, and the two pairs agree. A tertiary nitrogen offers two carbons
 *  that no rule below separates, so both would be pinned syn to the same carbonyl oxygen -- two cells
 *  no arrangement satisfies, and nothing in the builder positioned to notice.
 *
 *  What holds a tertiary amide flat is the CSD torsion table, whose `NX3H0` rows carry a doubled
 *  periodicity and so name both planar minima instead of choosing one.
 */

/**
 *  @brief The E/Z relationship @p path's central bond fixes between its two end atoms.
 *
 *  The flag on the bond states the relationship of the two @b recorded reference substituents, one per
 *  double-bond end. Any other substituent on an end sits opposite that end's reference, so each
 *  mismatch flips the sense. Which reference sits on which end is read from the adjacency rather than
 *  from the order the two were stored, so a path that crosses the bond from the far end resolves the
 *  same way.
 */
inline path_stereo_t path_stereo_(molecule_descriptor_t const &descriptor, path_14_t const &path) {
    std::uint32_t const bond = path.bond_second_third;
    std::uint8_t const flags = descriptor.bond_flags[bond];
    if ((flags & (bond_flag_stereo_e_k | bond_flag_stereo_z_k)) == 0) return path_stereo_t::unspecified_k;
    std::uint32_t const reference_i = descriptor.bond_stereo_atom_i[bond];
    std::uint32_t const reference_j = descriptor.bond_stereo_atom_j[bond];
    if (reference_i == no_stereo_atom_k || reference_j == no_stereo_atom_k) return path_stereo_t::unspecified_k;

    bool const reference_i_near = bond_between_(descriptor, reference_i, path.second).has_value();
    std::uint32_t const near_reference = reference_i_near ? reference_i : reference_j;
    std::uint32_t const far_reference = reference_i_near ? reference_j : reference_i;
    bool cis = (flags & bond_flag_stereo_z_k) != 0;
    if (path.first != near_reference) cis = !cis;
    if (path.fourth != far_reference) cis = !cis;
    return cis ? path_stereo_t::cis_k : path_stereo_t::trans_k;
}

/** Whether both ends of the central bond are trigonal, so the path lies in one plane. */
inline bool central_pair_is_flat_(molecule_descriptor_t const &descriptor, path_14_t const &path) {
    return descriptor.hybridization[path.second] == hybridization_sp2_k &&
           descriptor.hybridization[path.third] == hybridization_sp2_k;
}

/** Whether `second` and `third` are the two divalent sulfurs of a disulfide. */
inline bool central_pair_is_disulfide_(molecule_descriptor_t const &descriptor, path_14_t const &path) {
    return descriptor.element[path.second] == 16 && descriptor.element[path.third] == 16 &&
           descriptor.degree[path.second] == 2 && descriptor.degree[path.third] == 2;
}

/** Whether @p atom is a carbon carrying a doubly bonded oxygen or nitrogen. */
inline bool is_carbonyl_(molecule_descriptor_t const &descriptor, std::uint32_t atom) {
    if (descriptor.element[atom] != 6 || descriptor.degree[atom] <= 2) return false;
    std::uint32_t const first_entry = descriptor.neighbor_offset[atom];
    std::uint32_t const last_entry = descriptor.neighbor_offset[atom + 1];
    for (std::uint32_t entry = first_entry; entry < last_entry; ++entry) {
        std::uint8_t const element = descriptor.element[descriptor.neighbor_atom[entry]];
        if (element != 8 && element != 7) continue;
        if (bond_kind_(descriptor, descriptor.neighbor_bond[entry]) == bond_kind_t::double_k) return true;
    }
    return false;
}

/**
 *  @brief Whether the path reads `first — second — third = fourth` with `third` a carbonyl carbon.
 *
 *  @verbatim
 *          fourth      <- the carbonyl oxygen
 *          ||
 *   first  third
 *       \  /   \
 *      second   *      <- second is the ester oxygen or the amide nitrogen
 *  @endverbatim
 */
inline bool is_amide_or_ester_14_(molecule_descriptor_t const &descriptor, path_14_t const &path) {
    std::uint8_t const element_second = descriptor.element[path.second];
    std::uint8_t const element_fourth = descriptor.element[path.fourth];
    return descriptor.element[path.third] == 6 &&
           bond_kind_(descriptor, path.bond_third_fourth) == bond_kind_t::double_k &&
           (element_fourth == 8 || element_fourth == 7) &&
           bond_kind_(descriptor, path.bond_first_second) == bond_kind_t::single_k &&
           (element_second == 8 || (element_second == 7 && descriptor.num_hydrogens[path.second] == 1));
}

/** The same arrangement one atom further out, where the path's ends are the amide's 1 and 5. */
inline bool is_amide_or_ester_15_(molecule_descriptor_t const &descriptor, path_14_t const &path) {
    std::uint8_t const element_second = descriptor.element[path.second];
    if (element_second != 8 && !(element_second == 7 && descriptor.num_hydrogens[path.second] == 1)) return false;
    if (bond_kind_(descriptor, path.bond_first_second) != bond_kind_t::single_k) return false;
    if (descriptor.element[path.third] != 6) return false;
    if (bond_kind_(descriptor, path.bond_third_fourth) != bond_kind_t::single_k) return false;
    return is_carbonyl_(descriptor, path.third);
}

/**
 *  @brief `is_amide_or_ester_14_` widened for a macrocycle, where the nitrogen may carry no hydrogen.
 *
 *  A methylated amide inside a large ring is still an amide, and the hydrogen count that the open-chain
 *  predicate leans on is the one thing a ring closure can take away.
 */
inline bool is_macrocyclic_amide_or_ester_14_(molecule_descriptor_t const &descriptor, path_14_t const &path) {
    std::uint8_t const element_second = descriptor.element[path.second];
    std::uint8_t const element_fourth = descriptor.element[path.fourth];
    return descriptor.element[path.first] != 1 && descriptor.element[path.third] == 6 &&
           bond_kind_(descriptor, path.bond_third_fourth) == bond_kind_t::double_k &&
           (element_fourth == 8 || element_fourth == 7) &&
           bond_kind_(descriptor, path.bond_first_second) == bond_kind_t::single_k &&
           (element_second == 8 || element_second == 7);
}

/** The only neighbour of @p atom that is neither @p excluded_first nor @p excluded_second. */
inline std::optional<std::uint32_t> other_neighbor_entry_(molecule_descriptor_t const &descriptor, std::uint32_t atom,
                                                          std::uint32_t excluded_first, std::uint32_t excluded_second) {
    std::uint32_t const first_entry = descriptor.neighbor_offset[atom];
    std::uint32_t const last_entry = descriptor.neighbor_offset[atom + 1];
    for (std::uint32_t entry = first_entry; entry < last_entry; ++entry) {
        std::uint32_t const neighbor = descriptor.neighbor_atom[entry];
        if (neighbor == excluded_first || neighbor == excluded_second) continue;
        return entry;
    }
    return std::nullopt;
}

/**
 *  @brief Whether an amide or ester spans the path with every one of its four substituents in the ring.
 *
 *  Where `is_macrocyclic_amide_or_ester_14_` reads the path's own four atoms, this one also inspects
 *  what hangs off the two central atoms: a carbon or a hydrogen on the nitrogen, and the carbonyl
 *  oxygen on the carbon. Both central atoms are three-coordinate, so each has exactly one such
 *  neighbour.
 */
inline bool is_macrocyclic_ring_amide_or_ester_14_(molecule_descriptor_t const &descriptor, path_14_t const &path) {
    std::uint8_t const element_second = descriptor.element[path.second];
    if (descriptor.element[path.third] != 6) return false;
    if (element_second != 7 && element_second != 8) return false;
    if (descriptor.degree[path.second] != 3 || descriptor.degree[path.third] != 3) return false;

    std::optional<std::uint32_t> const substituent = other_neighbor_entry_(descriptor, path.second, path.first,
                                                                           path.third);
    if (substituent) {
        std::uint8_t const element = descriptor.element[descriptor.neighbor_atom[*substituent]];
        if (element != 6 && element != 1) return false;
        if (bond_kind_(descriptor, descriptor.neighbor_bond[*substituent]) != bond_kind_t::single_k) return false;
    }
    std::optional<std::uint32_t> const carbonyl = other_neighbor_entry_(descriptor, path.third, path.second,
                                                                        path.fourth);
    if (carbonyl) {
        if (descriptor.element[descriptor.neighbor_atom[*carbonyl]] != 8) return false;
        if (bond_kind_(descriptor, descriptor.neighbor_bond[*carbonyl]) != bond_kind_t::double_k) return false;
    }
    return true;
}

/** Whether an end of the path is the N-H hydrogen of a secondary amide. */
inline bool ends_on_secondary_amide_hydrogen_(molecule_descriptor_t const &descriptor, path_14_t const &path) {
    auto const hydrogen_on_nitrogen = [&](std::uint32_t hydrogen, std::uint32_t nitrogen) {
        return descriptor.element[hydrogen] == 1 && descriptor.element[nitrogen] == 7 &&
               descriptor.degree[nitrogen] == 3 && descriptor.num_hydrogens[nitrogen] == 1;
    };
    return hydrogen_on_nitrogen(path.first, path.second) || hydrogen_on_nitrogen(path.fourth, path.third);
}

/** A rule written for one direction along a 1-4 path. */
using path_14_predicate_t = bool (*)(molecule_descriptor_t const &, path_14_t const &);

/**
 *  @brief Whether @p predicate holds on @p path walked from either end.
 *
 *  Every rule above reads the path's positions in order, so the arrangement it names is found only when
 *  the path happens to be walked the way the rule is written. Which direction matched is not reported,
 *  and the caller's own @p path keeps its orientation for whatever it reads next.
 */
inline bool holds_in_either_direction_(molecule_descriptor_t const &descriptor, path_14_t const &path,
                                       path_14_predicate_t predicate) {
    return predicate(descriptor, path) || predicate(descriptor, reversed_(path));
}

#pragma endregion Chemical Predicates

#pragma region Torsion Targets

/** The stereo-defined answer for a path across a double bond, or `flexible_k` where none is stated. */
inline torsion_preference_t stereo_preference_(molecule_descriptor_t const &descriptor, path_14_t const &path) {
    switch (path_stereo_(descriptor, path)) {
    case path_stereo_t::cis_k: return torsion_preference_t::cis_k;
    case path_stereo_t::trans_k: return torsion_preference_t::trans_k;
    case path_stereo_t::unspecified_k: return torsion_preference_t::flexible_k;
    }
    return torsion_preference_t::flexible_k;
}

/** The open-chain case: a cumulated double bond, a disulfide, an amide or ester, or free rotation. */
inline torsion_target_t chain_preference_(molecule_descriptor_t const &descriptor, path_14_t const &path) {
    switch (bond_kind_(descriptor, path.bond_second_third)) {
    case bond_kind_t::double_k:
        // An allene's two double bonds share a plane and eclipse the ends.
        if (bond_kind_(descriptor, path.bond_first_second) == bond_kind_t::double_k ||
            bond_kind_(descriptor, path.bond_third_fourth) == bond_kind_t::double_k)
            return {.preference = torsion_preference_t::cis_k};
        return {.preference = stereo_preference_(descriptor, path)};
    case bond_kind_t::single_k:
        // Two lone pairs per sulfur hold a disulfide near a right angle.
        if (central_pair_is_disulfide_(descriptor, path)) return {.preference = torsion_preference_t::right_angle_k};
        if (holds_in_either_direction_(descriptor, path, is_amide_or_ester_14_))
            return {.preference = ends_on_secondary_amide_hydrogen_(descriptor, path) ? torsion_preference_t::trans_k
                                                                                      : torsion_preference_t::cis_k};
        // One atom further out the sense inverts: the amide's own N-H is now syn to the far end.
        if (holds_in_either_direction_(descriptor, path, is_amide_or_ester_15_))
            return {.preference = ends_on_secondary_amide_hydrogen_(descriptor, path) ? torsion_preference_t::cis_k
                                                                                      : torsion_preference_t::trans_k};
        return {.preference = torsion_preference_t::flexible_k};
    case bond_kind_t::other_k: return {.preference = torsion_preference_t::flexible_k};
    }
    return {.preference = torsion_preference_t::flexible_k};
}

/**
 *  @brief The case where the ring closure itself decides, and the one both cross-ring cases delegate to.
 *
 *  @p ring_size is 0 when the three bonds share no ring, which is how `two_in_different_rings_k` and
 *  `shares_ring_bond_k` reach the same answer as a small ring: a flat centre pair eclipses its ends.
 */
inline torsion_target_t in_ring_preference_(molecule_descriptor_t const &descriptor, path_14_t const &path,
                                            std::uint32_t ring_size) {
    path_stereo_t const stereo = path_stereo_(descriptor, path);
    // A ring past eight atoms has room for a trans arrangement, so its flat centre pins nothing.
    bool const closure_pins_cis = ring_size <= largest_pinned_ring_size_k && central_pair_is_flat_(descriptor, path) &&
                                  stereo != path_stereo_t::trans_k;
    if (closure_pins_cis || stereo == path_stereo_t::cis_k) return {.preference = torsion_preference_t::cis_k};
    if (stereo == path_stereo_t::trans_k) return {.preference = torsion_preference_t::trans_k};
    return {.preference = torsion_preference_t::flexible_k};
}

/** The case where the path leaves the ring at one end, so a flat ring pushes that end anti. */
inline torsion_target_t two_in_same_ring_preference_(molecule_descriptor_t const &descriptor, path_14_t const &path,
                                                     exocyclic_preference_t exocyclic) {
    path_stereo_t const stereo = path_stereo_(descriptor, path);
    bool const ring_pushes_trans = exocyclic == exocyclic_preference_t::prefers_trans_k &&
                                   central_pair_is_flat_(descriptor, path) && stereo != path_stereo_t::cis_k;
    if (ring_pushes_trans || stereo == path_stereo_t::trans_k) return {.preference = torsion_preference_t::trans_k};
    if (stereo == path_stereo_t::cis_k) return {.preference = torsion_preference_t::cis_k};
    return {.preference = torsion_preference_t::flexible_k};
}

/** `two_in_same_ring_preference_` for a macrocycle, where an amide across the ring reads cis. */
inline torsion_target_t macrocycle_two_in_same_ring_preference_(molecule_descriptor_t const &descriptor,
                                                                path_14_t const &path) {
    if (holds_in_either_direction_(descriptor, path, is_macrocyclic_amide_or_ester_14_))
        return {.preference = torsion_preference_t::cis_k};
    return {.preference = stereo_preference_(descriptor, path)};
}

/** `chain_preference_` for a path whose three bonds all lie in one macrocycle. */
inline torsion_target_t macrocycle_all_in_ring_preference_(molecule_descriptor_t const &descriptor,
                                                           path_14_t const &path) {
    switch (bond_kind_(descriptor, path.bond_second_third)) {
    case bond_kind_t::double_k:
        if (bond_kind_(descriptor, path.bond_first_second) == bond_kind_t::double_k ||
            bond_kind_(descriptor, path.bond_third_fourth) == bond_kind_t::double_k)
            return {.preference = torsion_preference_t::cis_k};
        return {.preference = stereo_preference_(descriptor, path)};
    case bond_kind_t::single_k:
        if (central_pair_is_disulfide_(descriptor, path)) return {.preference = torsion_preference_t::right_angle_k};
        if (holds_in_either_direction_(descriptor, path, is_macrocyclic_ring_amide_or_ester_14_))
            return {.preference = torsion_preference_t::trans_k, .extra_reach = macrocycle_amide_reach_k};
        if (holds_in_either_direction_(descriptor, path, is_amide_or_ester_15_)) {
            // The secondary amide's own hydrogen is placed by the shorter path across the same amide,
            // so bounding it from here as well would fight that answer.
            bool const own_amide_hydrogen = descriptor.element[path.second] == 7 &&
                                            descriptor.degree[path.second] == 3 &&
                                            descriptor.element[path.first] == 1 &&
                                            descriptor.num_hydrogens[path.second] == 1;
            if (own_amide_hydrogen) return {.preference = torsion_preference_t::none_k};
            return {.preference = torsion_preference_t::trans_k};
        }
        return {.preference = torsion_preference_t::flexible_k};
    case bond_kind_t::other_k: return {.preference = torsion_preference_t::flexible_k};
    }
    return {.preference = torsion_preference_t::flexible_k};
}

/**
 *  @brief Which of the seven cases @p path is in, read from the ring membership of its three bonds.
 *
 *  @param[in] descriptor Molecule graph.
 *  @param[in] bond_ring_bitmap One word per bond, bit `ring` set when the bond belongs to that SSSR ring.
 *  @param[in] path The 1-4 path to classify.
 *  @return The case, plus the ring size and the exocyclic bias the case reads.
 */
inline path_14_case_t classify_path_14_(molecule_descriptor_t const &descriptor,
                                        custd::span<std::uint32_t const> bond_ring_bitmap, path_14_t const &path) {
    std::uint32_t const rings_first = bond_ring_bitmap[path.bond_first_second];
    std::uint32_t const rings_central = bond_ring_bitmap[path.bond_second_third];
    std::uint32_t const rings_last = bond_ring_bitmap[path.bond_third_fourth];

    std::uint32_t const ring_size_all = smallest_ring_size_(descriptor, rings_first & rings_central & rings_last);
    if (ring_size_all != 0)
        return {.kind = ring_size_all >= macrocycle_ring_size_k ? path_14_kind_t::macrocycle_all_in_ring_k
                                                                : path_14_kind_t::all_in_ring_k,
                .ring_size = ring_size_all};

    std::uint32_t const ring_size_front = smallest_ring_size_(descriptor, rings_first & rings_central);
    std::uint32_t const ring_size_back = smallest_ring_size_(descriptor, rings_central & rings_last);
    if (ring_size_front != 0 || ring_size_back != 0) {
        // The middle bond belonging to a macrocycle is what makes the path a macrocycle case, whichever
        // of its two neighbours is the one sharing a ring with it.
        if (ring_mask_holds_macrocycle_(descriptor, rings_central))
            return {.kind = path_14_kind_t::macrocycle_two_in_same_ring_k};
        bool const flat_ring = (ring_size_front != 0 && ring_size_front <= largest_pinned_ring_size_k) ||
                               (ring_size_back != 0 && ring_size_back <= largest_pinned_ring_size_k);
        return {.kind = path_14_kind_t::two_in_same_ring_k,
                .exocyclic = flat_ring ? exocyclic_preference_t::prefers_trans_k : exocyclic_preference_t::unbiased_k};
    }
    // Two ring bonds that share no ring: the third bond belongs to neither, so nothing is coplanar.
    if ((rings_first != 0 && rings_central != 0) || (rings_central != 0 && rings_last != 0))
        return {.kind = path_14_kind_t::two_in_different_rings_k};
    if (rings_central != 0) return {.kind = path_14_kind_t::shares_ring_bond_k};
    return {.kind = path_14_kind_t::in_chain_k};
}

/** The torsion @p path's case asks for. */
inline torsion_target_t path_14_target_(molecule_descriptor_t const &descriptor, path_14_t const &path,
                                        path_14_case_t const &path_case) {
    switch (path_case.kind) {
    case path_14_kind_t::in_chain_k: return chain_preference_(descriptor, path);
    case path_14_kind_t::all_in_ring_k: return in_ring_preference_(descriptor, path, path_case.ring_size);
    case path_14_kind_t::two_in_same_ring_k: return two_in_same_ring_preference_(descriptor, path, path_case.exocyclic);
    case path_14_kind_t::two_in_different_rings_k: return in_ring_preference_(descriptor, path, 0);
    case path_14_kind_t::shares_ring_bond_k: return in_ring_preference_(descriptor, path, 0);
    case path_14_kind_t::macrocycle_two_in_same_ring_k:
        return macrocycle_two_in_same_ring_preference_(descriptor, path);
    case path_14_kind_t::macrocycle_all_in_ring_k: return macrocycle_all_in_ring_preference_(descriptor, path);
    }
    return {.preference = torsion_preference_t::flexible_k};
}

#pragma endregion Torsion Targets

#pragma region Interval Merge

/** The distance interval @p target claims for @p path, in Å. */
inline bound_interval_t path_14_interval_(molecule_descriptor_t const &descriptor, path_14_t const &path,
                                          torsion_target_t const &target) {
    float const length_12 = bond_length_angstrom_(descriptor, path.bond_first_second);
    float const length_23 = bond_length_angstrom_(descriptor, path.bond_second_third);
    float const length_34 = bond_length_angstrom_(descriptor, path.bond_third_fourth);
    float const angle_123 = interior_angle_radians_(descriptor, path.first, path.second, path.third);
    float const angle_234 = interior_angle_radians_(descriptor, path.second, path.third, path.fourth);
    auto const distance_at = [&](float cos_torsion) {
        return one_four_distance_(length_12, length_23, length_34, angle_123, angle_234, cos_torsion);
    };

    if (target.preference == torsion_preference_t::flexible_k) {
        float lower = distance_at(1.0f);
        float upper = distance_at(-1.0f);
        // A strained centre can invert the two, and a degenerate interval is no constraint at all.
        if (upper < lower) std::swap(lower, upper);
        if (upper - lower < degenerate_interval_k) {
            lower -= one_four_tolerance_k;
            upper += one_four_tolerance_k;
        }
        return {.lower = lower, .upper = upper};
    }

    float const cos_torsion = target.preference == torsion_preference_t::cis_k     ? 1.0f
                              : target.preference == torsion_preference_t::trans_k ? -1.0f
                                                                                   : 0.0f;
    float const distance = distance_at(cos_torsion) + target.extra_reach;
    return {.lower = distance - one_four_tolerance_k, .upper = distance + one_four_tolerance_k};
}

/**
 *  @brief Collapse the intervals several 1-4 paths claim for one atom pair into a single interval.
 *
 *  A pair reachable by more than one three-bond path — a stereo double bond on each of two routes, a
 *  fused ring system, an amide inside a macrocycle — is claimed once per path, and a bounds matrix
 *  holds one interval per pair. Taking the union of the maximal intersections keeps every claim that
 *  can be satisfied together and widens to cover the rest, so a pair whose paths disagree outright
 *  ends up loose rather than contradictory. Ported from RDKit's `merge`,
 *  `Code/GraphMol/DistGeomHelpers/BoundsMatrixBuilderDetails.h`.
 *
 *  @param[inout] intervals Caller-owned claims, at least one, sorted in place by lower bound.
 *  @return The merged interval.
 */
inline bound_interval_t merge_intervals_(custd::span<bound_interval_t> intervals) {
    std::sort(intervals.begin(), intervals.end(),
              [](bound_interval_t lhs, bound_interval_t rhs) { return lhs.lower < rhs.lower; });

    bound_interval_t current = intervals[0];
    float component_upper = current.upper;
    std::optional<float> first_maximal_lower;
    for (std::uint32_t index = 1; index < intervals.size(); ++index) {
        bound_interval_t const claim = intervals[index];
        if (claim.lower <= current.upper) {
            // The claim overlaps what is open, so it narrows it; the sort makes its lower the larger one.
            current.lower = claim.lower;
            current.upper = std::min(current.upper, claim.upper);
        }
        else {
            // Nothing more can join the open intersection, so it is maximal and a new one starts.
            if (!first_maximal_lower) first_maximal_lower = current.lower;
            current.lower = claim.lower;
            current.upper = claim.lower <= component_upper ? std::min(component_upper, claim.upper) : claim.upper;
        }
        component_upper = std::max(component_upper, claim.upper);
    }
    return {.lower = first_maximal_lower.value_or(current.lower), .upper = current.upper};
}

#pragma endregion Interval Merge

#pragma region Matrix Assembly

/**
 *  @brief Fill @p bounds with every pair's distance interval, in Å.
 *
 *  Four sweeps in increasing path length, each owning the pairs at its own hop distance so no later
 *  sweep can overwrite an earlier, tighter answer. Which triangle carries which end of a pair is
 *  `types/views.hpp`'s convention, and `upper_distance_bound_at` and `lower_distance_bound_at` are how
 *  every write below states it.
 *
 *  @param[in] descriptor Molecule graph with explicit hydrogens already added.
 *  @param[out] bounds Square view over the molecule's atoms, zeroed by the caller — `reset_square_matrix`
 *      does both — since the diagonal is the one part no sweep writes.
 */
inline void build_bounds_matrix(molecule_descriptor_t const &descriptor, packed_matrix<float> bounds) {
    std::uint32_t const num_atoms = descriptor.num_atoms;
    sf_assert_(bounds.extent(0) == num_atoms && bounds.extent(1) == num_atoms);

    // The 1-4 and long-range sweeps below are the only host consumers of the hop matrix, so it
    // lives on this frame at the molecule's own row stride rather than inside the descriptor.
    std::uint8_t bond_hop_storage[max_atoms_per_molecule_k * max_atoms_per_molecule_k];
    hop_matrix_t const bond_hops {bond_hop_storage, num_atoms, num_atoms};
    fill_bond_distances(descriptor, bond_hops);

    // Initial upper bound for every pair: large (effectively unconstrained).
    constexpr float upper_initial_k = 1000.0f;
    for (std::uint32_t row_atom = 0; row_atom < num_atoms; ++row_atom)
        for (std::uint32_t column_atom = row_atom + 1; column_atom < num_atoms; ++column_atom) {
            upper_distance_bound_at(bounds, row_atom, column_atom) = upper_initial_k;
            lower_distance_bound_at(bounds, row_atom, column_atom) = 0.0f;
        }

    // set12 — exact ideal bond lengths inside an absolute band, `DIST12_DELTA` upstream.
    //
    // A fraction of the bond is the wrong shape for this: it makes a long bond's window wide and a
    // short one's narrow, when the uncertainty being expressed is how far a real bond sits from its
    // rest length and has nothing to do with the length. RDKit only reaches a percentage on the
    // branch where UFF cannot type an atom, so ± 2 % was a tighter copy of its give-up path — three
    // times the band for C-C, three and a half for C-S.
    for (std::uint32_t bond = 0; bond < descriptor.num_bonds; ++bond) {
        std::uint32_t const atom_i = descriptor.bond_atom_i[bond];
        std::uint32_t const atom_j = descriptor.bond_atom_j[bond];
        float const ideal = bond_length_angstrom_(descriptor, bond);
        upper_distance_bound_at(bounds, atom_i, atom_j) = ideal + bond_tolerance_angstrom_k;
        lower_distance_bound_at(bounds, atom_i, atom_j) = ideal - bond_tolerance_angstrom_k;
    }

    // set13 — one 1-3 bound per pair of neighbors around every center atom.
    // Loop is O(N × d²) where d is the typical degree (small).
    //
    // A 1-3 write must never land on a pair an earlier, lower-order sweep already answered. Two neighbours
    // of a center are themselves bonded in every three-membered ring, and the law of cosines below would
    // replace an exact bond length with a 109.47°-derived distance — cyclopropane's C-C going from 1.51 Å
    // to 2.51 Å, with triangle smoothing finding that perfectly consistent. RDKit carried the same defect
    // for years; see `c5de9f884357` (#9184) and `bb18fd56f294` (#9412), which introduced the visited-bound
    // discipline upstream. The hop matrix already distinguishes the orders, so it is the guard here.
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint32_t const nbegin = descriptor.neighbor_offset[center];
        std::uint32_t const nend = descriptor.neighbor_offset[center + 1];
        for (std::uint32_t first_entry = nbegin; first_entry < nend; ++first_entry) {
            std::uint32_t const first_neighbor = descriptor.neighbor_atom[first_entry];
            float const d_ci = bond_length_angstrom_(descriptor, descriptor.neighbor_bond[first_entry]);
            for (std::uint32_t second_entry = first_entry + 1; second_entry < nend; ++second_entry) {
                std::uint32_t const second_neighbor = descriptor.neighbor_atom[second_entry];
                float const d_cj = bond_length_angstrom_(descriptor, descriptor.neighbor_bond[second_entry]);
                if (bond_hops(first_neighbor, second_neighbor) != 2) continue;

                float const theta = interior_angle_radians_(descriptor, first_neighbor, center, second_neighbor);
                float const cos_theta = std::cos(theta);
                float const d_ij = std::sqrt(
                    std::max(0.0f, d_ci * d_ci + d_cj * d_cj - 2.0f * d_ci * d_cj * cos_theta));
                // An absolute band, `DIST13_TOL` upstream, doubled once per heavy sp2 ring atom in the
                // triple. That widening is not slack for its own sake: a thiophene or furan 1-3 spans
                // atoms whose ring geometry the polygon angle above describes badly, and upstream
                // loosens exactly there rather than manufacturing an infeasible window.
                float const tolerance = angle_tolerance_angstrom_k *
                                        static_cast<float>(1u << larger_sp2_ring_atoms_(descriptor, first_neighbor,
                                                                                        center, second_neighbor));
                float const lower = d_ij - tolerance, upper = d_ij + tolerance;
                float &current_lower = lower_distance_bound_at(bounds, first_neighbor, second_neighbor);
                float &current_upper = upper_distance_bound_at(bounds, first_neighbor, second_neighbor);
                // A pair sharing two centers is claimed once per center, and the two claims differ when
                // the centers differ in hybridization or in bond length — a beta-lactam ring reaches it.
                // Widening rather than replacing keeps both, so the pair does not depend on which center
                // the walk found first; that is what RDKit's `_checkAndSetBounds` does at
                // `setIfBetter=false`, and it is not what `merge_intervals_` does to the 1-4 claims,
                // which is to intersect them. The unset test reads the upper because no 1-3 distance
                // reaches a thousand angstroms, where a degenerate one could reach the lower's zero.
                bool const unbounded = current_upper == upper_initial_k;
                current_lower = unbounded ? lower : std::min(current_lower, lower);
                current_upper = unbounded ? upper : std::max(current_upper, upper);
            }
        }
    }

    // The ring membership of the three bonds is the whole 1-4 classification, and the descriptor
    // labels rings on atoms rather than on bonds, so transpose it once here. A ring's atoms arrive in
    // ring order, so each consecutive pair — the last wrapping to the first — names one ring bond.
    std::uint32_t bond_ring_bitmap[max_bonds_k] = {};
    for (std::uint32_t ring = 0; ring < descriptor.num_rings; ++ring) {
        std::uint32_t const ring_first = descriptor.ring_offset[ring];
        std::uint32_t const ring_last = descriptor.ring_offset[ring + 1];
        if (ring_last - ring_first < 3) continue;
        for (std::uint32_t member = ring_first; member < ring_last; ++member) {
            std::uint32_t const next_member = member + 1 == ring_last ? ring_first : member + 1;
            std::optional<std::uint32_t> const bond = bond_between_(descriptor, descriptor.ring_atoms[member],
                                                                    descriptor.ring_atoms[next_member]);
            if (bond) bond_ring_bitmap[*bond] |= 1u << ring;
        }
    }
    custd::span<std::uint32_t const> const ring_bitmap_span {bond_ring_bitmap, descriptor.num_bonds};

    // set14 — one interval per three-bond path between the pair, merged into the bound the pair keeps.
    // The pair's hop distance being exactly three is what makes every such path a genuine 1-4 contact:
    // were `first` and `third` bonded, or `second` and `fourth`, the pair would sit two hops apart and
    // the 1-3 sweep would already own it, which is the fused-ring case RDKit tests for by hand.
    bound_interval_t claims[max_paths_per_pair_k];
    for (std::uint32_t row_atom = 0; row_atom < num_atoms; ++row_atom) {
        for (std::uint32_t column_atom = row_atom + 1; column_atom < num_atoms; ++column_atom) {
            if (bond_hops(row_atom, column_atom) != 3) continue;

            std::uint32_t num_claims = 0;
            std::uint32_t const second_first = descriptor.neighbor_offset[row_atom];
            std::uint32_t const second_last = descriptor.neighbor_offset[row_atom + 1];
            for (std::uint32_t second_entry = second_first; second_entry < second_last; ++second_entry) {
                std::uint32_t const second = descriptor.neighbor_atom[second_entry];
                std::uint32_t const third_first = descriptor.neighbor_offset[second];
                std::uint32_t const third_last = descriptor.neighbor_offset[second + 1];
                for (std::uint32_t third_entry = third_first; third_entry < third_last; ++third_entry) {
                    std::uint32_t const third = descriptor.neighbor_atom[third_entry];
                    if (third == row_atom) continue; // walking straight back down the first bond
                    std::optional<std::uint32_t> const closing = bond_between_(descriptor, third, column_atom);
                    if (!closing) continue;
                    if (num_claims == max_paths_per_pair_k) break;

                    path_14_t const path {.first = row_atom,
                                          .second = second,
                                          .third = third,
                                          .fourth = column_atom,
                                          .bond_first_second = descriptor.neighbor_bond[second_entry],
                                          .bond_second_third = descriptor.neighbor_bond[third_entry],
                                          .bond_third_fourth = *closing};
                    path_14_case_t const path_case = classify_path_14_(descriptor, ring_bitmap_span, path);
                    torsion_target_t const target = path_14_target_(descriptor, path, path_case);
                    if (target.preference == torsion_preference_t::none_k) continue;
                    claims[num_claims++] = path_14_interval_(descriptor, path, target);
                }
            }

            // Every path declining to bound the pair leaves it as loose as a non-bonded contact.
            if (num_claims == 0) {
                lower_distance_bound_at(bounds, row_atom, column_atom) = vdw_lower_bound_angstrom(
                    descriptor.element[row_atom], descriptor.element[column_atom]);
                continue;
            }
            bound_interval_t const merged = merge_intervals_({claims, num_claims});
            upper_distance_bound_at(bounds, row_atom, column_atom) = merged.upper;
            lower_distance_bound_at(bounds, row_atom, column_atom) = merged.lower;
        }
    }

    // Long-range pairs (more than three hops apart): VdW lower bound, keep upper unconstrained.
    // A hydrogen bond brings its donor hydrogen and its acceptor far inside van der Waals contact, so a
    // pair that can close one is given the hydrogen-bond length instead; without it the bounds matrix
    // forbids the intramolecular hydrogen bonds that fix a conformer's shape. Ported from RDKit's
    // `f7a3f044ffd0` (#9188), with thanks — theirs never labels atom zero, since its sweep starts at one.
    for (std::uint32_t row_atom = 0; row_atom < num_atoms; ++row_atom) {
        for (std::uint32_t column_atom = row_atom + 1; column_atom < num_atoms; ++column_atom) {
            if (bond_hops(row_atom, column_atom) <= 3) continue;
            std::uint8_t const row_flags = descriptor.atom_flags[row_atom];
            std::uint8_t const column_flags = descriptor.atom_flags[column_atom];
            bool const hydrogen_bonded =
                ((row_flags & atom_flag_hbond_donor_h_k) != 0 && (column_flags & atom_flag_hbond_acceptor_k) != 0) ||
                ((column_flags & atom_flag_hbond_donor_h_k) != 0 && (row_flags & atom_flag_hbond_acceptor_k) != 0);
            lower_distance_bound_at(bounds, row_atom, column_atom) =
                hydrogen_bonded
                    ? hydrogen_bond_length_k
                    : vdw_lower_bound_angstrom(descriptor.element[row_atom], descriptor.element[column_atom]);
        }
    }
}

#pragma endregion Matrix Assembly

} // namespace smallfield::preprocess
