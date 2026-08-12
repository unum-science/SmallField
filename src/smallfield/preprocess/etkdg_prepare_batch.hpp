/**
 *  @file src/smallfield/preprocess/etkdg_prepare_batch.hpp
 *  @brief CPU-side ETKDG preprocessing using RDKit C++ API.
 *
 *  Converts molecular graphs into GPU-ready data structures:
 *   - Distance bounds matrix — N × N, unsmoothed; triangle smoothing runs on the device
 *   - Chiral center descriptors — index-ordered neighbors, target volume sign
 *   - ETK torsion descriptors — CSD Fourier coefficients
 *   - 1-2 / 1-3 distance constraints
 *
 *  All functions operate on RDKit ROMol objects and produce flat arrays suitable for upload to GPU global memory.
 */

#pragma once

#include <cassert> // `assert`
#include <cmath>   // `std::cos`, `std::sin`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`
#include <cstring> // `std::memcpy`

#include <algorithm>   // `std::max`, `std::min`, `std::sort`
#include <array>       // `std::array`
#include <limits>      // `std::numeric_limits`
#include <memory>      // `std::unique_ptr`
#include <optional>    // `std::optional`
#include <sstream>     // RDKit's `AtomTyper.h` needs it but does not include it
#include <string_view> // `std::string_view`
#include <utility>     // `std::move`
#include <vector>      // `std::vector`

#ifdef _OPENMP
#include <omp.h>
#endif

#include <boost/shared_array.hpp>

#include <DistGeom/BoundsMatrix.h>
#include <ForceField/MMFF/Nonbonded.h>
#include <ForceField/MMFF/Params.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/Chirality.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Descriptors/MolDescriptors.h>

#include "smallfield/api/wave_slice_pool.hpp"
#include "smallfield/molecule_parse.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/storage.hpp"
#include "smallfield/types/views.hpp"
#include "smallfield/preprocess/bounds_matrix_tables.hpp"
#include "smallfield/preprocess/csd_torsion_dispatcher.hpp"
#include "smallfield/preprocess/device_capacities.hpp"
#include "smallfield/preprocess/molecule_complexity.hpp"
#include "smallfield/mmff94/mmff94_typer.hpp"
#include "smallfield/types/mmff94_bounds.hpp"

namespace smallfield::preprocess {
using smallfield::api::conformer_status_t;
using smallfield::api::place_region_;
using smallfield::api::reserve_outcome_t;
using smallfield::api::wave_slice_layout_t;
using smallfield::api::wave_slice_pool_t;
using smallfield::types::atom_flag_aromatic_k;
using smallfield::types::atom_flag_hbond_acceptor_k;
using smallfield::types::atom_flag_hbond_donor_h_k;
using smallfield::types::atom_has_flag;
using smallfield::types::atom_role_first_k;
using smallfield::types::atom_role_fourth_k;
using smallfield::types::atom_role_second_k;
using smallfield::types::atom_role_third_k;
using smallfield::types::bond_flag_aromatic_k;
using smallfield::types::bond_flag_order_aromatic_k;
using smallfield::types::bond_flag_stereo_e_k;
using smallfield::types::bond_flag_stereo_z_k;
using smallfield::types::build_descriptor;
using smallfield::types::chiral_center_descriptor_t;
using smallfield::types::distance_constraint_t;
using smallfield::types::etk_atom_interaction_t;
using smallfield::types::etk_atom_offsets_t;
using smallfield::types::etk_distance_constraint_k;
using smallfield::types::etk_torsion_descriptor_t;
using smallfield::types::etk_torsion_k;
using smallfield::types::etkdg_molecule_metadata_t;
using smallfield::types::fill_bond_distances;
using smallfield::types::hop_matrix_t;
using smallfield::types::hybridization_sp2_k;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::max_bonds_k;
using smallfield::types::max_four_coordinate_volume_k;
using smallfield::types::max_mmff_interactions_per_mol_k;
using smallfield::types::max_three_coordinate_volume_k;
using smallfield::types::min_four_coordinate_volume_k;
using smallfield::types::min_three_coordinate_volume_k;
using smallfield::types::mmff94_angle_center_k;
using smallfield::types::mmff94_angle_end_k;
using smallfield::types::mmff94_angle_t;
using smallfield::types::mmff94_atom_interaction_t;
using smallfield::types::mmff94_atom_offsets_t;
using smallfield::types::mmff94_atom_vdw_params_t;
using smallfield::types::mmff94_bond_k;
using smallfield::types::mmff94_bond_t;
using smallfield::types::mmff94_oop_t;
using smallfield::types::mmff94_out_of_plane_k;
using smallfield::types::mmff94_packed_offsets_t;
using smallfield::types::mmff94_stretch_bend_k;
using smallfield::types::mmff94_stretchbend_t;
using smallfield::types::mmff94_torsion_k;
using smallfield::types::mmff94_torsion_t;
using smallfield::types::mmff94_typing_patch_t;
using smallfield::types::molecule_descriptor_t;
using smallfield::types::no_stereo_atom_k;

/**
 *  @brief Complete preprocessed molecule ready for GPU pipeline upload.
 */
struct etkdg_prepared_molecule_t {
    std::uint32_t num_atoms;
    /** N × N flat, unsmoothed. */
    std::vector<float> bounds_matrix;
    std::vector<chiral_center_descriptor_t> chiral_centers;
    std::vector<etk_torsion_descriptor_t> etk_torsions;
    /** 1-2 bond, 1-3 angle. */
    std::vector<distance_constraint_t> distance_constraints;
    // MMFF94 per-atom data — the kernel enumerates nonbonded pairs on the fly from bond_distance.
    /** One per atom. */
    std::vector<mmff94_atom_vdw_params_t> mmff_atom_vdw;
    /** One per atom. */
    std::vector<float> mmff_atom_charge;
    std::vector<mmff94_bond_t> mmff_bonds;
    std::vector<mmff94_angle_t> mmff_angles;
    std::vector<mmff94_torsion_t> mmff_torsions;
    std::vector<mmff94_oop_t> mmff_oops;
    std::vector<mmff94_stretchbend_t> mmff_stretchbends;
    // Per-atom MMFF94 interaction lists, with no nonbonded entries.
    std::vector<mmff94_atom_interaction_t> mmff_interactions;
    std::vector<mmff94_atom_offsets_t> mmff_atom_offsets;
    // Per-atom ETK interaction lists: torsions and distance constraints.
    std::vector<etk_atom_interaction_t> etk_interactions;
    std::vector<etk_atom_offsets_t> etk_atom_offsets;
    etkdg_molecule_metadata_t metadata;
    /** Flat topology the device preprocessing kernels read; filled by every prep path. */
    molecule_descriptor_t descriptor;
    /** Elements this molecule claims in each device-filled region. */
    device_capacities_t capacities;
    /** Atom types RDKit assigned, present only when the native typer rejected the molecule. */
    std::vector<std::uint8_t> patch_atom_types;
    /** Partial charges RDKit assigned, present only alongside `patch_atom_types`. */
    std::vector<float> patch_atom_charges;
};

#pragma region One Four Taxonomy

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

/** How a 1-4 path across an amide or an ester is bounded. */
enum class amide_geometry_t : std::uint8_t {
    /** A secondary amide's N-H is placed anti to the carbonyl oxygen and every other such path syn. */
    trans_locked_k,
    /** The path roams from cis to trans and the force field planarizes it afterwards. */
    freely_rotating_k,
};

/**
 *  @brief RDKit's `forceTransAmides`, at the setting every RDKit comparison in this tree runs.
 *
 *  ETKDGv3 leaves it on, and `test/`, `bench/` and `cli/` all embed with `RDKit::DGeomHelpers::ETKDGv3`,
 *  so the amide branches below are reached rather than configured away.
 */
inline constexpr amide_geometry_t amide_geometry_k = amide_geometry_t::trans_locked_k;

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

/**
 *  @brief The most 1-4 paths one atom pair collects, past which the remaining paths are dropped.
 *
 *  A pair's path count is the product of the two end atoms' degrees, and a hypervalent centre in the
 *  elements MMFF94 parametrises carries at most six bonds, so thirty-six is the chemical ceiling and
 *  the buffer clears it. The cap exists because the claims sit on this frame rather than on the heap.
 */
inline constexpr std::uint32_t max_paths_per_pair_k = 64;

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

/** Size of the largest SSSR ring named in @p ring_mask; 0 when the mask names none. */
inline std::uint32_t largest_ring_size_(molecule_descriptor_t const &descriptor, std::uint32_t ring_mask) {
    std::uint32_t largest = 0;
    for (std::uint32_t ring = 0; ring < descriptor.num_rings; ++ring) {
        if ((ring_mask & (1u << ring)) == 0) continue;
        std::uint32_t const ring_size = descriptor.ring_offset[ring + 1] - descriptor.ring_offset[ring];
        if (ring_size > largest) largest = ring_size;
    }
    return largest;
}

/** Ideal length in Å of @p bond, read from the element pair and the bond order. */
inline float bond_length_angstrom_(molecule_descriptor_t const &descriptor, std::uint32_t bond) {
    bool const aromatic = (descriptor.bond_flags[bond] & bond_flag_aromatic_k) != 0;
    return ideal_bond_length_angstrom(descriptor.element[descriptor.bond_atom_i[bond]],
                                      descriptor.element[descriptor.bond_atom_j[bond]], descriptor.bond_order[bond],
                                      aromatic);
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
    return ring_interior_angle_radians(descriptor.hybridization[center], smallest_ring_size_(descriptor, shared));
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
        if (amide_geometry_k == amide_geometry_t::freely_rotating_k)
            return {.preference = torsion_preference_t::flexible_k};
        if (is_amide_or_ester_14_(descriptor, path) || is_amide_or_ester_14_(descriptor, reversed_(path)))
            return {.preference = ends_on_secondary_amide_hydrogen_(descriptor, path) ? torsion_preference_t::trans_k
                                                                                      : torsion_preference_t::cis_k};
        // One atom further out the sense inverts: the amide's own N-H is now syn to the far end.
        if (is_amide_or_ester_15_(descriptor, path) || is_amide_or_ester_15_(descriptor, reversed_(path)))
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
    if (is_macrocyclic_amide_or_ester_14_(descriptor, path) ||
        is_macrocyclic_amide_or_ester_14_(descriptor, reversed_(path)))
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
        if (is_macrocyclic_ring_amide_or_ester_14_(descriptor, path) ||
            is_macrocyclic_ring_amide_or_ester_14_(descriptor, reversed_(path)))
            return {.preference = torsion_preference_t::trans_k, .extra_reach = macrocycle_amide_reach_k};
        if (is_amide_or_ester_15_(descriptor, path) || is_amide_or_ester_15_(descriptor, reversed_(path))) {
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
        if (largest_ring_size_(descriptor, rings_central) >= macrocycle_ring_size_k)
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

#pragma endregion One Four Taxonomy

/**
 *  @brief Native zero-heap builder for the distance bounds matrix.
 *
 *  Replaces RDKit's `setTopolBounds`, which allocates a fresh `DistGeom::BoundsMatrix` plus two
 *  `boost::shared_array<>` per molecule and walks RDKit's atom typer, roughly 140 KB of throwaway
 *  heap per call. This version reads only from the flat `molecule_descriptor_t` and the small
 *  `bounds_matrix_tables.hpp`, and writes directly into `bounds`.
 *
 *  The values are close but not bit-identical to `setTopolBounds`'s output. Downstream triangle
 *  smoothing and FIRE refinement, both on the GPU, absorb the small residual; `bench_conformer_backends`
 *  reports the paired energy distribution against the RDKit baseline.
 *
 *  Output layout, N × N row-major:
 *  @verbatim
 *  bounds[row_atom * N + column_atom], row_atom < column_atom   upper bound distance in angstroms
 *  bounds[row_atom * N + column_atom], row_atom > column_atom   lower bound distance in angstroms
 *  @endverbatim
 *
 *  @param[in] descriptor Molecule graph with explicit hydrogens already added.
 *  @param[out] bounds Flat N × N float array filled in row-major order.
 */
inline void build_bounds_matrix(molecule_descriptor_t const &descriptor, std::vector<float> &bounds) {
    std::uint32_t const num_atoms = descriptor.num_atoms;
    bounds.assign(static_cast<std::size_t>(num_atoms) * num_atoms, 0.0f);

    // The 1-4 and long-range sweeps below are the only host consumers of the hop matrix, so it
    // lives on this frame at the molecule's own row stride rather than inside the descriptor.
    std::uint8_t bond_hops[max_atoms_per_molecule_k * max_atoms_per_molecule_k];
    fill_bond_distances(descriptor, hop_matrix_t {bond_hops, num_atoms, num_atoms});

    auto set_pair = [&](std::uint32_t first_atom, std::uint32_t second_atom, float lower, float upper) {
        std::uint32_t const row = std::min(first_atom, second_atom);
        std::uint32_t const column = std::max(first_atom, second_atom);
        bounds[row * num_atoms + column] = upper;
        bounds[column * num_atoms + row] = lower;
    };

    // Initial upper bound for every pair: large (effectively unconstrained).
    constexpr float upper_initial_k = 1000.0f;
    for (std::uint32_t row_atom = 0; row_atom < num_atoms; ++row_atom)
        for (std::uint32_t column_atom = row_atom + 1; column_atom < num_atoms; ++column_atom)
            set_pair(row_atom, column_atom, 0.0f, upper_initial_k);

    // set12 — exact ideal bond lengths ± 2% tolerance per bond from the descriptor.
    constexpr float bond_tolerance_k = 0.02f;
    for (std::uint32_t bond = 0; bond < descriptor.num_bonds; ++bond) {
        float const ideal = bond_length_angstrom_(descriptor, bond);
        set_pair(descriptor.bond_atom_i[bond], descriptor.bond_atom_j[bond], ideal * (1.0f - bond_tolerance_k),
                 ideal * (1.0f + bond_tolerance_k));
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
                if (bond_hops[first_neighbor * num_atoms + second_neighbor] != 2) continue;

                float const theta = interior_angle_radians_(descriptor, first_neighbor, center, second_neighbor);
                float const cos_theta = std::cos(theta);
                float const d_ij = std::sqrt(
                    std::max(0.0f, d_ci * d_ci + d_cj * d_cj - 2.0f * d_ci * d_cj * cos_theta));
                // ± 3% tolerance — 1-3 distances tend to have slightly more slack.
                set_pair(first_neighbor, second_neighbor, d_ij * 0.97f, d_ij * 1.03f);
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
            if (bond_hops[row_atom * num_atoms + column_atom] != 3) continue;

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
                bounds[column_atom * num_atoms + row_atom] = vdw_lower_bound_angstrom(descriptor.element[row_atom],
                                                                                      descriptor.element[column_atom]);
                continue;
            }
            bound_interval_t const merged = merge_intervals_({claims, num_claims});
            set_pair(row_atom, column_atom, merged.lower, merged.upper);
        }
    }

    // Long-range pairs (more than three hops apart): VdW lower bound, keep upper unconstrained.
    // A hydrogen bond brings its donor hydrogen and its acceptor far inside van der Waals contact, so a
    // pair that can close one is given the hydrogen-bond length instead; without it the bounds matrix
    // forbids the intramolecular hydrogen bonds that fix a conformer's shape. Ported from RDKit's
    // `f7a3f044ffd0` (#9188), with thanks — theirs never labels atom zero, since its sweep starts at one.
    for (std::uint32_t row_atom = 0; row_atom < num_atoms; ++row_atom) {
        for (std::uint32_t column_atom = row_atom + 1; column_atom < num_atoms; ++column_atom) {
            if (bond_hops[row_atom * num_atoms + column_atom] <= 3) continue;
            std::uint8_t const row_flags = descriptor.atom_flags[row_atom];
            std::uint8_t const column_flags = descriptor.atom_flags[column_atom];
            bool const hydrogen_bonded =
                ((row_flags & atom_flag_hbond_donor_h_k) != 0 && (column_flags & atom_flag_hbond_acceptor_k) != 0) ||
                ((column_flags & atom_flag_hbond_donor_h_k) != 0 && (row_flags & atom_flag_hbond_acceptor_k) != 0);
            float const lower = hydrogen_bonded ? hydrogen_bond_length_k
                                                : vdw_lower_bound_angstrom(descriptor.element[row_atom],
                                                                           descriptor.element[column_atom]);
            bounds[column_atom * num_atoms + row_atom] = lower; // lower triangle
        }
    }
}

/**
 *  @brief Extract graph-only chiral center descriptors.
 *
 *  The downstream constraint is the sign of the triple product over the center and
 *  the first three stored neighbors, and that sign is fixed by exactly two things:
 *  the atom's tetrahedral tag, which RDKit defines against the order the bonds appear
 *  on the atom, and the parity of the stored neighbor order against that same bond
 *  order. The descriptor's CSR adjacency is filled in bond order, so sorting the
 *  neighbors by ascending atom index and counting the swaps yields that parity, and
 *  an odd count flips the handedness the tag encodes.
 *
 *  The rule is uniform across coordination numbers. A three-coordinate center such as
 *  a sulfoxide sulfur or a phosphine phosphorus carries a lone pair as its fourth
 *  substituent, which sits behind the three bonded neighbors in the bond order and in
 *  the stored order alike, so it contributes no parity of its own and drops out.
 *
 *  @param[in] molecule RDKit molecule; stereochemistry is assigned in place.
 *  @param[in] descriptor Flat descriptor supplying the chiral tags and the CSR adjacency.
 *  @param[out] chiral_centers Filled with one descriptor per tetrahedral center.
 */
inline void extract_chiral_centers(RDKit::ROMol const &molecule, molecule_descriptor_t const &descriptor,
                                   std::vector<chiral_center_descriptor_t> &chiral_centers) {
    chiral_centers.clear();
    auto &stereo_molecule = const_cast<RDKit::ROMol &>(molecule);
    RDKit::MolOps::assignStereochemistry(stereo_molecule, true, true, true);

    std::uint32_t const num_atoms = descriptor.num_atoms;
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        if (descriptor.chiral_tag[atom] == 0) continue;
        if (descriptor.degree[atom] < 3) continue;

        // `assignStereochemistry` strips the tag off every atom that parsed with one but is
        // not a stereocenter, so a surviving tag is the membership test for this center.
        auto const assigned_tag = molecule.getAtomWithIdx(atom)->getChiralTag();
        bool const is_stereocenter = assigned_tag == RDKit::Atom::CHI_TETRAHEDRAL_CW ||
                                     assigned_tag == RDKit::Atom::CHI_TETRAHEDRAL_CCW;
        if (!is_stereocenter) continue;

        chiral_center_descriptor_t chiral_center {};
        chiral_center.center_atom = static_cast<std::uint8_t>(atom);

        // Collect the neighbors in CSR order, which is the bond order the tag is defined against.
        std::uint8_t ordered[4] = {0, 0, 0, 0};
        std::uint32_t count = 0;
        std::uint16_t const neighbor_first = descriptor.neighbor_offset[atom];
        std::uint16_t const neighbor_last = descriptor.neighbor_offset[atom + 1];
        for (std::uint16_t entry = neighbor_first; entry < neighbor_last && count < 4; ++entry)
            ordered[count++] = descriptor.neighbor_atom[entry];

        // Insertion sort by ascending atom index over at most four entries, counting the swaps.
        std::uint32_t swaps = 0;
        for (std::uint32_t upper = 1; upper < count; ++upper) {
            std::uint8_t const moving = ordered[upper];
            std::uint32_t slot = upper;
            while (slot > 0 && ordered[slot - 1] > moving) {
                ordered[slot] = ordered[slot - 1];
                --slot;
                ++swaps;
            }
            ordered[slot] = moving;
        }

        for (std::uint32_t slot = 0; slot < 4; ++slot) chiral_center.neighbor_atoms[slot] = ordered[slot];

        int const tag_sign = (descriptor.chiral_tag[atom] == 1) ? 1 : -1;
        int const order_parity = (swaps % 2 == 0) ? 1 : -1;
        chiral_center.target_sign = static_cast<std::int8_t>(-tag_sign * order_parity);

        // A center with a lone pair in place of a fourth bond spans a smaller tetrahedron over the
        // three neighbors it does have, so it carries the wider window.
        bool const four_coordinate = count >= 4;
        chiral_center.minimum_volume = four_coordinate ? min_four_coordinate_volume_k : min_three_coordinate_volume_k;
        chiral_center.maximum_volume = four_coordinate ? max_four_coordinate_volume_k : max_three_coordinate_volume_k;

        chiral_centers.push_back(chiral_center);
    }
}

/**
 *  @brief Extract 1-2 bond and 1-3 angle-derived distance constraints for the ETK refinement stage.
 *
 *  @param[in] descriptor Flat descriptor supplying the bond list and the CSR adjacency.
 *  @param[in] bounds_matrix Flat N × N distance bounds, row-major, from `build_bounds_matrix`.
 *  @param[out] constraints Filled with 1-2 bond and 1-3 angle distance constraints.
 */
inline void extract_distance_constraints(molecule_descriptor_t const &descriptor,
                                         std::vector<float> const &bounds_matrix,
                                         std::vector<distance_constraint_t> &constraints) {
    constraints.clear();
    std::uint32_t const num_atoms = descriptor.num_atoms;

    // Reserve exact capacity: 1-2 count = num_bonds, 1-3 count = sum choose(degree, 2).
    std::uint32_t num_13 = 0;
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        std::uint32_t degree = descriptor.degree[atom];
        num_13 += degree * (degree - 1) / 2;
    }
    constraints.reserve(descriptor.num_bonds + num_13);

    // 1-2 bond constraints: one emit per bond, exact bounds from the matrix.
    for (std::uint32_t bond = 0; bond < descriptor.num_bonds; ++bond) {
        std::uint32_t const atom_i = descriptor.bond_atom_i[bond];
        std::uint32_t const atom_j = descriptor.bond_atom_j[bond];
        std::uint32_t const row_min = std::min(atom_i, atom_j);
        std::uint32_t const row_max = std::max(atom_i, atom_j);

        distance_constraint_t constraint {};
        constraint.atom_index_i = static_cast<std::uint8_t>(atom_i);
        constraint.atom_index_j = static_cast<std::uint8_t>(atom_j);
        constraint.upper_bound_angstrom = bounds_matrix[row_min * num_atoms + row_max];
        constraint.lower_bound_angstrom = bounds_matrix[row_max * num_atoms + row_min];
        constraint.penalty_weight = 10.0f;
        constraints.push_back(constraint);
    }

    // 1-3 constraints: iterate each atom as center, enumerate neighbor pairs via CSR.
    //
    // Two neighbours of a center are themselves bonded in every three-membered ring, and the pair then
    // already has a 1-2 constraint above; emitting it again as a 1-3 would restate the bond at half the
    // weight. That is the same guard the 1-3 bounds sweep carries, spelled through the adjacency rather
    // than through a hop matrix, because bonded is the only way two neighbours of one center are closer
    // than two hops apart.
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint16_t const start = descriptor.neighbor_offset[center];
        std::uint16_t const end = descriptor.neighbor_offset[center + 1];
        for (std::uint16_t first_entry = start; first_entry < end; ++first_entry) {
            for (std::uint16_t second_entry = first_entry + 1; second_entry < end; ++second_entry) {
                std::uint32_t const first_neighbor = descriptor.neighbor_atom[first_entry];
                std::uint32_t const second_neighbor = descriptor.neighbor_atom[second_entry];
                if (bond_between_(descriptor, first_neighbor, second_neighbor)) continue;
                std::uint32_t const row_min = std::min(first_neighbor, second_neighbor);
                std::uint32_t const row_max = std::max(first_neighbor, second_neighbor);

                distance_constraint_t constraint {};
                constraint.atom_index_i = static_cast<std::uint8_t>(first_neighbor);
                constraint.atom_index_j = static_cast<std::uint8_t>(second_neighbor);
                constraint.upper_bound_angstrom = bounds_matrix[row_min * num_atoms + row_max];
                constraint.lower_bound_angstrom = bounds_matrix[row_max * num_atoms + row_min];
                constraint.penalty_weight = 5.0f;
                constraints.push_back(constraint);
            }
        }
    }
}

/**
 *  @brief Build per-atom interaction lists for ETK torsions and distance constraints.
 *
 *  @param[inout] result Prepared molecule; ETK interaction lists are filled in place.
 */
inline void build_etk_interaction_lists(etkdg_prepared_molecule_t &result) {
    std::uint32_t const num_atoms = result.num_atoms;
    auto const &torsions = result.etk_torsions;
    auto const &constraints = result.distance_constraints;

    // pass 1: count interactions per atom. `num_atoms` <= `max_atoms_per_molecule_k`, so the
    // fixed-size `std::array` lives on the stack and avoids a per-molecule heap allocation on the
    // hot prep path.
    std::array<std::uint32_t, max_atoms_per_molecule_k> counts {};
    for (auto const &torsion : torsions) {
        counts[torsion.atom_indices[0]]++;
        counts[torsion.atom_indices[1]]++;
        counts[torsion.atom_indices[2]]++;
        counts[torsion.atom_indices[3]]++;
    }
    for (auto const &constraint : constraints) {
        counts[constraint.atom_index_i]++;
        counts[constraint.atom_index_j]++;
    }

    // compute offsets
    result.etk_atom_offsets.resize(num_atoms);
    std::uint32_t running = 0;
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        result.etk_atom_offsets[atom] = {running, counts[atom]};
        running += counts[atom];
    }

    // pass 2: fill entries
    result.etk_interactions.resize(running);
    std::array<std::uint32_t, max_atoms_per_molecule_k> cursors {};
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
        cursors[atom] = result.etk_atom_offsets[atom].interaction_offset;

    for (std::uint32_t term = 0; term < torsions.size(); ++term) {
        result.etk_interactions[cursors[torsions[term].atom_indices[0]]++] = {term, etk_torsion_k, atom_role_first_k};
        result.etk_interactions[cursors[torsions[term].atom_indices[1]]++] = {term, etk_torsion_k, atom_role_second_k};
        result.etk_interactions[cursors[torsions[term].atom_indices[2]]++] = {term, etk_torsion_k, atom_role_third_k};
        result.etk_interactions[cursors[torsions[term].atom_indices[3]]++] = {term, etk_torsion_k, atom_role_fourth_k};
    }
    for (std::uint32_t term = 0; term < constraints.size(); ++term) {
        result.etk_interactions[cursors[constraints[term].atom_index_i]++] = {term, etk_distance_constraint_k,
                                                                              atom_role_first_k};
        result.etk_interactions[cursors[constraints[term].atom_index_j]++] = {term, etk_distance_constraint_k,
                                                                              atom_role_second_k};
    }
}

/**
 *  @brief Extract MMFF94 force-field parameters and build per-atom interaction lists.
 *
 *  @param[in] molecule RDKit molecule with explicit hydrogens.
 *  @param[inout] result Prepared molecule; MMFF94 arrays are filled in place.
 *  @return `conformer_status_t::success_k` on success;
 *          `conformer_status_t::unsupported_element_k` if any atom's
 *          MMFF94 atom type is 0 (element outside the MMFF94 parameter
 *          set — transition metals, lanthanides, etc.) or if the
 *          `MMFFMolProperties` constructor otherwise reports the
 *          molecule as not parametrizable.
 */
inline smallfield::api::conformer_status_t extract_mmff94_params_via_rdkit(RDKit::ROMol const &molecule,
                                                                           molecule_descriptor_t const &descriptor,
                                                                           etkdg_prepared_molecule_t &result) {
    result.mmff_atom_vdw.clear();
    result.mmff_atom_charge.clear();
    result.mmff_bonds.clear();
    result.mmff_angles.clear();
    result.mmff_torsions.clear();

    auto &mutable_molecule = const_cast<RDKit::ROMol &>(molecule);
    // Skip redundant Kekulize inside `MMFFMolProperties` — `SmilesToMol` already sanitized.
    if (!mutable_molecule.hasProp(RDKit::common_properties::_MMFFSanitized)) {
        mutable_molecule.setProp(RDKit::common_properties::_MMFFSanitized, 1, true);
    }
    RDKit::MMFF::MMFFMolProperties mmff_props(mutable_molecule);

    std::uint32_t const num_atoms = descriptor.num_atoms;

    // `getMMFFAtomType` returns 0 for atoms outside the MMFF94 parameter set — yttrium,
    // lanthanides, transition metals and the like. Detect that here on the host so the molecule
    // never reaches the GPU pipeline: without this gate, `etkdg_pipeline_warp_kernel` does an OOB
    // write through element-keyed parameter LUTs in `bounds_matrix_tables.hpp` and poisons every
    // subsequent CUDA call on every device.
    std::uint8_t atom_type[max_atoms_per_molecule_k];
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        atom_type[atom] = static_cast<std::uint8_t>(mmff_props.getMMFFAtomType(atom));
        if (atom_type[atom] == 0) return smallfield::api::conformer_status_t::unsupported_element_k;
    }

    // Defence in depth: catch the rare case of all-typed atoms but unparametrized interactions,
    // such as missing torsion params. Same routing as the per-atom case — the molecule can't be
    // MMFF-optimised on-device and must fall back to CPU ETKDGv3.
    if (!mmff_props.isValid()) return smallfield::api::conformer_status_t::unsupported_element_k;
    auto find_bond_index = [&](std::uint32_t first_atom, std::uint32_t second_atom) -> std::int32_t {
        for (std::uint16_t entry = descriptor.neighbor_offset[first_atom];
             entry < descriptor.neighbor_offset[first_atom + 1]; ++entry)
            if (descriptor.neighbor_atom[entry] == second_atom) return descriptor.neighbor_bond[entry];
        return -1;
    };

    // Per-atom data: VdW params + partial charge. The kernel's nonbonded loop enumerates
    // pairs on the fly from the bond_distance matrix and applies Halgren's R* combining
    // rule plus Slater-Kirkwood epsilon combining inline — no per-pair host emission.
    result.mmff_atom_vdw.resize(num_atoms);
    result.mmff_atom_charge.resize(num_atoms);
    auto const *mmff_vdw_coll = RDKit::MMFF::DefaultParameters::getMMFFVdW();
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        ForceFields::MMFF::MMFFVdW const *vdw_params = (*mmff_vdw_coll)(atom_type[atom]);
        mmff94_atom_vdw_params_t params {};
        if (vdw_params) {
            params.alpha = static_cast<float>(vdw_params->alpha_i);
            params.n_effective = static_cast<float>(vdw_params->N_i);
            params.a_scale = static_cast<float>(vdw_params->A_i);
            params.g_factor = static_cast<float>(vdw_params->G_i);
            params.r_star = static_cast<float>(vdw_params->R_star);
            // DA flag: RDKit stores the character 'D'/'A'/'-' in vdw_params->DA.
            params.donor_acceptor = (vdw_params->DA == 'D') ? 2 : (vdw_params->DA == 'A') ? 1 : 0;
        }
        result.mmff_atom_vdw[atom] = params;
        result.mmff_atom_charge[atom] = static_cast<float>(mmff_props.getMMFFPartialCharge(atom));
    }

    // Build bond terms — iterate descriptor's bond list, with no RDKit traversal.
    for (std::uint32_t bond = 0; bond < descriptor.num_bonds; ++bond) {
        std::uint32_t const atom_i = descriptor.bond_atom_i[bond];
        std::uint32_t const atom_j = descriptor.bond_atom_j[bond];
        unsigned int bond_type;
        ForceFields::MMFF::MMFFBond bond_params;
        // Dropping the term would understate the molecule's energy by a whole bond stretch, and the
        // device hop-count BFS derives adjacency from this list, so a missing entry also shifts the
        // 1-2 / 1-3 / 1-4 nonbonded exclusions. Reject the molecule instead and let the fallback
        // role handle it.
        if (!mmff_props.getMMFFBondStretchParams(mutable_molecule, atom_i, atom_j, bond_type, bond_params))
            return smallfield::api::conformer_status_t::unsupported_element_k;
        result.mmff_bonds.push_back({.atom_i = static_cast<std::uint8_t>(atom_i),
                                     .atom_j = static_cast<std::uint8_t>(atom_j),
                                     .padding = {0, 0},
                                     .force_constant = static_cast<float>(bond_params.kb),
                                     .equilibrium_distance = static_cast<float>(bond_params.r0),
                                     .padding_float = 0.0f});
    }

    // Build angle terms — iterate atoms as angle centers, enumerate neighbor pairs via CSR.
    for (std::uint32_t atom_j = 0; atom_j < num_atoms; ++atom_j) {
        std::uint16_t const nstart = descriptor.neighbor_offset[atom_j];
        std::uint16_t const nend = descriptor.neighbor_offset[atom_j + 1];
        for (std::uint16_t first_entry = nstart; first_entry < nend; ++first_entry) {
            for (std::uint16_t second_entry = first_entry + 1; second_entry < nend; ++second_entry) {
                std::uint32_t const atom_i = descriptor.neighbor_atom[first_entry];
                std::uint32_t const atom_k = descriptor.neighbor_atom[second_entry];
                std::int32_t const bond_index_ij = find_bond_index(atom_i, atom_j);
                std::int32_t const bond_index_jk = find_bond_index(atom_j, atom_k);
                if (bond_index_ij < 0 || bond_index_jk < 0) continue;
                unsigned int angle_type;
                ForceFields::MMFF::MMFFAngle angle_params;
                if (!mmff_props.getMMFFAngleBendParams(mutable_molecule, atom_i, atom_j, atom_k, angle_type,
                                                       angle_params))
                    continue;
                // `linh` depends only on the central atom's MMFF type.
                auto const *prop_center = (*RDKit::MMFF::DefaultParameters::getMMFFProp())(atom_type[atom_j]);
                std::uint8_t const is_linear = (prop_center && prop_center->linh) ? 1u : 0u;
                result.mmff_angles.push_back(
                    {.atom_i = static_cast<std::uint8_t>(atom_i),
                     .atom_j = static_cast<std::uint8_t>(atom_j),
                     .atom_k = static_cast<std::uint8_t>(atom_k),
                     .is_linear = is_linear,
                     .force_constant = static_cast<float>(angle_params.ka),
                     .equilibrium_angle = static_cast<float>(angle_params.theta0 * M_PI / 180.0),
                     .padding_float = 0.0f});
            }
        }
    }

    // Build torsion terms — iterate descriptor's bonds as central, enumerate end-atom neighbors.
    for (std::uint32_t bond = 0; bond < descriptor.num_bonds; ++bond) {
        std::uint32_t const atom_j = descriptor.bond_atom_i[bond];
        std::uint32_t const atom_k = descriptor.bond_atom_j[bond];
        std::uint16_t const jstart = descriptor.neighbor_offset[atom_j];
        std::uint16_t const jend = descriptor.neighbor_offset[atom_j + 1];
        std::uint16_t const kstart = descriptor.neighbor_offset[atom_k];
        std::uint16_t const kend = descriptor.neighbor_offset[atom_k + 1];
        for (std::uint16_t atom_j_entry = jstart; atom_j_entry < jend; ++atom_j_entry) {
            std::uint32_t const atom_i = descriptor.neighbor_atom[atom_j_entry];
            if (atom_i == atom_k) continue;
            for (std::uint16_t atom_k_entry = kstart; atom_k_entry < kend; ++atom_k_entry) {
                std::uint32_t const atom_l = descriptor.neighbor_atom[atom_k_entry];
                if (atom_l == atom_j) continue;
                unsigned int torsion_type;
                ForceFields::MMFF::MMFFTor torsion_params;
                if (!mmff_props.getMMFFTorsionParams(mutable_molecule, atom_i, atom_j, atom_k, atom_l, torsion_type,
                                                     torsion_params))
                    continue;
                result.mmff_torsions.push_back({.atom_i = static_cast<std::uint8_t>(atom_i),
                                                .atom_j = static_cast<std::uint8_t>(atom_j),
                                                .atom_k = static_cast<std::uint8_t>(atom_k),
                                                .atom_l = static_cast<std::uint8_t>(atom_l),
                                                .torsion_v1 = static_cast<float>(torsion_params.V1),
                                                .torsion_v2 = static_cast<float>(torsion_params.V2),
                                                .torsion_v3 = static_cast<float>(torsion_params.V3)});
            }
        }
    }

    // Build out-of-plane bending terms for sp2 centers with ≥3 neighbors.
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint32_t const degree = descriptor.degree[center];
        if (degree < 3) continue;
        std::uint16_t const nstart = descriptor.neighbor_offset[center];
        std::uint8_t neighbor_cache[8];
        std::uint32_t neighbor_count = 0;
        for (std::uint16_t entry = nstart; entry < descriptor.neighbor_offset[center + 1] && neighbor_count < 8;
             ++entry)
            neighbor_cache[neighbor_count++] = descriptor.neighbor_atom[entry];

        for (std::uint32_t apex_slot = 0; apex_slot < neighbor_count; ++apex_slot) {
            std::uint8_t plane[2];
            std::uint32_t plane_count = 0;
            for (std::uint32_t plane_slot = 0; plane_slot < neighbor_count && plane_count < 2; ++plane_slot)
                if (plane_slot != apex_slot) plane[plane_count++] = neighbor_cache[plane_slot];
            if (plane_count < 2) continue;
            ForceFields::MMFF::MMFFOop oop_params;
            if (!mmff_props.getMMFFOopBendParams(mutable_molecule, neighbor_cache[apex_slot], center, plane[0],
                                                 plane[1], oop_params))
                continue;
            result.mmff_oops.push_back({.atom_i = neighbor_cache[apex_slot],
                                        .atom_j = static_cast<std::uint8_t>(center),
                                        .atom_k = plane[0],
                                        .atom_l = plane[1],
                                        .force_constant = static_cast<float>(oop_params.koop),
                                        .padding = {0.0f, 0.0f}});
        }
    }

    // Build stretch-bend terms — same topology as angles.
    for (std::uint32_t atom_j = 0; atom_j < num_atoms; ++atom_j) {
        std::uint16_t const nstart = descriptor.neighbor_offset[atom_j];
        std::uint16_t const nend = descriptor.neighbor_offset[atom_j + 1];
        for (std::uint16_t first_entry = nstart; first_entry < nend; ++first_entry) {
            for (std::uint16_t second_entry = first_entry + 1; second_entry < nend; ++second_entry) {
                std::uint32_t const atom_i = descriptor.neighbor_atom[first_entry];
                std::uint32_t const atom_k = descriptor.neighbor_atom[second_entry];
                std::int32_t const bond_index_ij = find_bond_index(atom_i, atom_j);
                std::int32_t const bond_index_jk = find_bond_index(atom_j, atom_k);
                if (bond_index_ij < 0 || bond_index_jk < 0) continue;
                unsigned int stretchbend_type;
                ForceFields::MMFF::MMFFStbn stretchbend_params;
                ForceFields::MMFF::MMFFBond bond_params[2];
                ForceFields::MMFF::MMFFAngle angle_params;
                if (!mmff_props.getMMFFStretchBendParams(mutable_molecule, atom_i, atom_j, atom_k, stretchbend_type,
                                                         stretchbend_params, bond_params, angle_params))
                    continue;
                result.mmff_stretchbends.push_back(
                    {.atom_i = static_cast<std::uint8_t>(atom_i),
                     .atom_j = static_cast<std::uint8_t>(atom_j),
                     .atom_k = static_cast<std::uint8_t>(atom_k),
                     .padding = 0,
                     .kba_ijk = static_cast<float>(stretchbend_params.kbaIJK),
                     .kba_kji = static_cast<float>(stretchbend_params.kbaKJI),
                     .equilibrium_angle = static_cast<float>(angle_params.theta0 * M_PI / 180.0),
                     .equilibrium_distance_ij = static_cast<float>(bond_params[0].r0),
                     .equilibrium_distance_kj = static_cast<float>(bond_params[1].r0),
                     .padding_float = 0.0f});
            }
        }
    }

    // Build per-atom interaction lists for bonded terms only. Nonbonded pairs are
    // enumerated on the GPU from bond_distance — they do not go into the CSR.
    auto const &bonds = result.mmff_bonds;
    auto const &angles = result.mmff_angles;
    auto const &torsions = result.mmff_torsions;
    auto const &oops = result.mmff_oops;
    auto const &stretchbends = result.mmff_stretchbends;

    // pass 1: count per-type interactions per atom. Fixed-size stack array
    // sized to the hard atom-count cap `max_atoms_per_molecule_k` — no per-molecule
    // heap allocation on the prep path.
    // One counter per MMFF term kind: term tallies in pass 1, write cursors in pass 2.
    struct per_atom_term_counters_t {
        // `uint32_t` not `uint8_t`: see `mmff94_atom_offsets_t` comment.
        std::uint32_t bond, angle, torsion, oop, stretchbend;
    };
    std::array<per_atom_term_counters_t, max_atoms_per_molecule_k> type_counts {};
    for (auto const &bond : bonds) {
        type_counts[bond.atom_i].bond++;
        type_counts[bond.atom_j].bond++;
    }
    for (auto const &angle : angles) {
        type_counts[angle.atom_i].angle++;
        type_counts[angle.atom_j].angle++;
        type_counts[angle.atom_k].angle++;
    }
    for (auto const &torsion : torsions) {
        type_counts[torsion.atom_i].torsion++;
        type_counts[torsion.atom_j].torsion++;
        type_counts[torsion.atom_k].torsion++;
        type_counts[torsion.atom_l].torsion++;
    }
    for (auto const &oop : oops) {
        type_counts[oop.atom_i].oop++;
        type_counts[oop.atom_j].oop++;
        type_counts[oop.atom_k].oop++;
        type_counts[oop.atom_l].oop++;
    }
    for (auto const &stretchbend : stretchbends) {
        type_counts[stretchbend.atom_i].stretchbend++;
        type_counts[stretchbend.atom_j].stretchbend++;
        type_counts[stretchbend.atom_k].stretchbend++;
    }

    // compute per-type offsets: bonds first, then angles, etc. No nonbonded range.
    // Sanity-assert that no per-atom interaction count exceeds the documented bound
    // `max_mmff_interactions_per_mol_k`. A `uint8_t` counter silently wraps here for dense
    // drug-like atoms with more than 255 torsions, so the counts are `uint32_t`, which removes the
    // ceiling; the assert catches any upstream bug that would still stage corrupt data to the GPU.
    result.mmff_atom_offsets.resize(num_atoms);
    std::uint32_t running = 0;
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        auto const &counts = type_counts[atom];
        auto &offsets = result.mmff_atom_offsets[atom];
        assert(counts.bond <= max_mmff_interactions_per_mol_k && "per-atom bond_count exceeds mol cap");
        assert(counts.angle <= max_mmff_interactions_per_mol_k && "per-atom angle_count exceeds mol cap");
        assert(counts.torsion <= max_mmff_interactions_per_mol_k && "per-atom torsion_count exceeds mol cap");
        assert(counts.oop <= max_mmff_interactions_per_mol_k && "per-atom oop_count exceeds mol cap");
        assert(counts.stretchbend <= max_mmff_interactions_per_mol_k && "per-atom stretchbend_count exceeds mol cap");
        offsets.bond_offset = running;
        offsets.bond_count = counts.bond;
        running += counts.bond;
        offsets.angle_offset = running;
        offsets.angle_count = counts.angle;
        running += counts.angle;
        offsets.torsion_offset = running;
        offsets.torsion_count = counts.torsion;
        running += counts.torsion;
        offsets.oop_offset = running;
        offsets.oop_count = counts.oop;
        running += counts.oop;
        offsets.stretchbend_offset = running;
        offsets.stretchbend_count = counts.stretchbend;
        running += counts.stretchbend;
    }

    // pass 2: fill interaction entries grouped by type per atom
    std::uint32_t const total_interactions = running;
    result.mmff_interactions.resize(total_interactions);

    // per-type cursors per atom — stack-sized, same rationale as type_counts.
    std::array<per_atom_term_counters_t, max_atoms_per_molecule_k> cursors {};
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        auto const &offsets = result.mmff_atom_offsets[atom];
        cursors[atom] = {offsets.bond_offset, offsets.angle_offset, offsets.torsion_offset, offsets.oop_offset,
                         offsets.stretchbend_offset};
    }

    for (std::uint32_t term = 0; term < bonds.size(); ++term) {
        result.mmff_interactions[cursors[bonds[term].atom_i].bond++] = {term, mmff94_bond_k, atom_role_first_k};
        result.mmff_interactions[cursors[bonds[term].atom_j].bond++] = {term, mmff94_bond_k, atom_role_second_k};
    }
    for (std::uint32_t term = 0; term < angles.size(); ++term) {
        result.mmff_interactions[cursors[angles[term].atom_i].angle++] = {term, mmff94_angle_end_k, atom_role_first_k};
        result.mmff_interactions[cursors[angles[term].atom_j].angle++] = {term, mmff94_angle_center_k,
                                                                          atom_role_second_k};
        result.mmff_interactions[cursors[angles[term].atom_k].angle++] = {term, mmff94_angle_end_k, atom_role_third_k};
    }
    for (std::uint32_t term = 0; term < torsions.size(); ++term) {
        result.mmff_interactions[cursors[torsions[term].atom_i].torsion++] = {term, mmff94_torsion_k,
                                                                              atom_role_first_k};
        result.mmff_interactions[cursors[torsions[term].atom_j].torsion++] = {term, mmff94_torsion_k,
                                                                              atom_role_second_k};
        result.mmff_interactions[cursors[torsions[term].atom_k].torsion++] = {term, mmff94_torsion_k,
                                                                              atom_role_third_k};
        result.mmff_interactions[cursors[torsions[term].atom_l].torsion++] = {term, mmff94_torsion_k,
                                                                              atom_role_fourth_k};
    }
    for (std::uint32_t term = 0; term < oops.size(); ++term) {
        result.mmff_interactions[cursors[oops[term].atom_i].oop++] = {term, mmff94_out_of_plane_k, atom_role_first_k};
        result.mmff_interactions[cursors[oops[term].atom_j].oop++] = {term, mmff94_out_of_plane_k, atom_role_second_k};
        result.mmff_interactions[cursors[oops[term].atom_k].oop++] = {term, mmff94_out_of_plane_k, atom_role_third_k};
        result.mmff_interactions[cursors[oops[term].atom_l].oop++] = {term, mmff94_out_of_plane_k, atom_role_fourth_k};
    }
    for (std::uint32_t term = 0; term < stretchbends.size(); ++term) {
        result.mmff_interactions[cursors[stretchbends[term].atom_i].stretchbend++] = {term, mmff94_stretch_bend_k,
                                                                                      atom_role_first_k};
        result.mmff_interactions[cursors[stretchbends[term].atom_j].stretchbend++] = {term, mmff94_stretch_bend_k,
                                                                                      atom_role_second_k};
        result.mmff_interactions[cursors[stretchbends[term].atom_k].stretchbend++] = {term, mmff94_stretch_bend_k,
                                                                                      atom_role_third_k};
    }
    return smallfield::api::conformer_status_t::success_k;
}

/**
 *  @brief Fill a caller-owned @p result with the full preprocessing output for one SMILES.
 *
 *  Reached through `prepare_molecule`; the batched path runs
 *  `prepare_molecule_topology_into` instead. The caller-owned `result` is meant
 *  to be reused across molecules — each molecule's vectors keep their capacity
 *  from the previous call, so `push_back` stays on the hot path without heap
 *  allocation after the first few molecules. Sets
 *  `result.num_atoms = 0` on failure and returns the rejection reason; on
 *  success returns `conformer_status_t::success_k`. Callers that don't need
 *  the per-molecule reason code can still test `result.num_atoms == 0` as
 *  the in-band drop signal.
 */
inline smallfield::api::conformer_status_t prepare_molecule_into(std::string_view smiles,
                                                                 etkdg_prepared_molecule_t &result) {
    result.num_atoms = 0;
    result.bounds_matrix.clear();
    result.chiral_centers.clear();
    result.etk_torsions.clear();
    result.distance_constraints.clear();
    result.mmff_atom_vdw.clear();
    result.mmff_atom_charge.clear();
    result.mmff_bonds.clear();
    result.mmff_angles.clear();
    result.mmff_torsions.clear();
    result.mmff_oops.clear();
    result.mmff_stretchbends.clear();
    result.mmff_interactions.clear();
    result.mmff_atom_offsets.clear();
    result.etk_interactions.clear();
    result.etk_atom_offsets.clear();

    smallfield::parse_options_t parse_options;
    parse_options.max_atoms = max_atoms_per_molecule_k; // single-warp pipeline cap
    parse_options.metadata_policy = smallfield::compute_metadata_k;
    smallfield::parse_result_t parsed = smallfield::parse_smiles(smiles, parse_options);
    if (parsed.status != conformer_status_t::success_k) return parsed.status;
    std::unique_ptr<RDKit::RWMol> molecule = std::move(parsed.molecule);
    std::uint32_t const num_atoms = parsed.num_atoms;
    result.num_atoms = num_atoms;

    // Optional host-side readback fields, measured on the same molecule the geometry describes so
    // callers can populate `n_heavy_atoms`, `n_bonds` and `molecular_weight` without a second parse.
    result.metadata.num_heavy_atoms = parsed.num_heavy_atoms;
    result.metadata.num_bonds = parsed.num_bonds;
    result.metadata.molecular_weight = parsed.molecular_weight;

    conformer_status_t mmff_status = conformer_status_t::success_k;
    try {
        molecule_descriptor_t descriptor;
        conformer_status_t const descriptor_status = build_descriptor(*molecule, descriptor);
        if (descriptor_status != conformer_status_t::success_k) {
            result.num_atoms = 0;
            return descriptor_status;
        }
        build_bounds_matrix(descriptor, result.bounds_matrix);
        extract_chiral_centers(*molecule, descriptor, result.chiral_centers);
        extract_etk_torsions_native(descriptor, result.etk_torsions);
        extract_distance_constraints(descriptor, result.bounds_matrix, result.distance_constraints);
        build_etk_interaction_lists(result);
        mmff_status = extract_mmff94_params_via_rdkit(*molecule, descriptor, result);
    }
    catch (...) {
        result.num_atoms = 0;
        return conformer_status_t::invalid_smiles_k;
    }
    if (mmff_status != conformer_status_t::success_k) {
        // MMFF could not parametrise this molecule — say, an atomic number outside the MMFF94 set.
        // The mmff_* result vectors are empty after the early return inside
        // `extract_mmff94_params_via_rdkit`, but ETK fields above are populated; clear num_atoms so
        // downstream packing skips this mol.
        result.num_atoms = 0;
        return mmff_status;
    }

    result.metadata.num_atoms = result.num_atoms;
    result.metadata.num_chiral_centers = static_cast<std::uint32_t>(result.chiral_centers.size());
    result.metadata.num_etk_torsions = static_cast<std::uint32_t>(result.etk_torsions.size());
    result.metadata.num_distance_constraints = static_cast<std::uint32_t>(result.distance_constraints.size());
    result.metadata.num_mmff_bonds = static_cast<std::uint32_t>(result.mmff_bonds.size());
    result.metadata.num_mmff_angles = static_cast<std::uint32_t>(result.mmff_angles.size());
    result.metadata.num_mmff_torsions = static_cast<std::uint32_t>(result.mmff_torsions.size());
    result.metadata.num_mmff_oops = static_cast<std::uint32_t>(result.mmff_oops.size());
    result.metadata.num_mmff_stretchbends = static_cast<std::uint32_t>(result.mmff_stretchbends.size());
    result.metadata.bounds_offset = 0;
    result.metadata.chiral_offset = 0;
    result.metadata.torsion_offset = 0;
    result.metadata.constraint_offset = 0;
    result.metadata.mmff_atom_vdw_offset = 0;
    result.metadata.mmff_atom_charge_offset = 0;
    result.metadata.bond_distance_offset = 0;
    result.metadata.mmff_bond_offset = 0;
    result.metadata.mmff_angle_offset = 0;
    result.metadata.mmff_torsion_offset = 0;
    result.metadata.mmff_oop_offset = 0;
    result.metadata.mmff_stretchbend_offset = 0;
    result.metadata.mmff_atom_csr_offset = 0;
    result.metadata.etk_interaction_offset = 0;
    result.metadata.etk_atom_csr_offset = 0;
    return conformer_status_t::success_k;
}

/**
 *  @brief Convert a SMILES string to a fresh `etkdg_prepared_molecule_t`.
 *
 *  Thin wrapper around `prepare_molecule_into` used by tests and by the non-pipelined path. Allocates
 *  a fresh struct per call — @b not the hot path. The batched path inside `prepare_molecule_batch`
 *  reuses each worker's `batch_prep_thread_contribution_t::scratch` instead, so vector capacities
 *  survive from one molecule to the next.
 */
inline etkdg_prepared_molecule_t prepare_molecule(std::string_view smiles) {
    etkdg_prepared_molecule_t result {};
    prepare_molecule_into(smiles, result);
    return result;
}

/**
 *  @brief Read RDKit's MMFF94 atom types and partial charges for one molecule.
 *
 *  Reached only when `native_compute_mmff_atom_types` leaves an atom untyped, which happens for
 *  4.63 % of a 5000-SMILES stratified PubChem sample and 0.02 % of a 5000-SMILES Enamine REAL
 *  sample. The device extractor is a pure function of the types and charges it is handed, so
 *  where RDKit can type such a molecule, supplying RDKit's answer keeps it on the same path as
 *  every other one; where RDKit cannot, the caller rejects it. On both samples above RDKit
 *  rejects every one of them, so this returns `unsupported_element_k` throughout.
 *
 *  @param[in] molecule RDKit molecule with explicit hydrogens, already sanitized.
 *  @param[in] num_atoms Atoms in @p molecule.
 *  @param[out] atom_types Filled with one MMFF94 numeric atom type per atom.
 *  @param[out] atom_charges Filled with one MMFF94 partial charge per atom.
 *  @return `conformer_status_t::success_k`, or `unsupported_element_k` when RDKit cannot type the
 *          molecule either — an atomic number outside the MMFF94 parameter set.
 */
inline smallfield::api::conformer_status_t extract_mmff94_typing_via_rdkit_(RDKit::ROMol const &molecule,
                                                                            std::uint32_t num_atoms,
                                                                            std::vector<std::uint8_t> &atom_types,
                                                                            std::vector<float> &atom_charges) {
    auto &mutable_molecule = const_cast<RDKit::ROMol &>(molecule);
    // `SmilesToMol` already sanitized, so tell `MMFFMolProperties` to skip its own Kekulize pass.
    if (!mutable_molecule.hasProp(RDKit::common_properties::_MMFFSanitized))
        mutable_molecule.setProp(RDKit::common_properties::_MMFFSanitized, 1, true);
    RDKit::MMFF::MMFFMolProperties properties(mutable_molecule);
    if (!properties.isValid()) return smallfield::api::conformer_status_t::unsupported_element_k;

    atom_types.resize(num_atoms);
    atom_charges.resize(num_atoms);
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        std::uint32_t const type = properties.getMMFFAtomType(atom);
        if (type == 0) return smallfield::api::conformer_status_t::unsupported_element_k;
        atom_types[atom] = static_cast<std::uint8_t>(type);
        atom_charges[atom] = static_cast<float>(properties.getMMFFPartialCharge(atom));
    }
    return smallfield::api::conformer_status_t::success_k;
}

/**
 *  @brief Fill @p result with everything the device pipeline needs the host to produce.
 *
 *  The host keeps what depends on RDKit's molecular graph — the parse, `AddHs`, the flat
 *  descriptor, the bounds matrix, the chiral centres, the 1-2 / 1-3 constraints — plus the
 *  per-region capacities the launch path turns into per-molecule bases. The CSD torsion match,
 *  the MMFF atom typing and charges, the MMFF term extraction and both per-atom CSRs run on the
 *  device from the descriptor this function fills.
 *
 *  Sets `result.num_atoms` to zero on rejection and returns the reason; on success returns
 *  `conformer_status_t::success_k`.
 *
 *  @param[in] smiles Input string; not required to be null-terminated.
 *  @param[in] salt_policy Whether a multi-fragment input keeps its largest fragment.
 *  @param[in] output_atom_capacity Atoms the caller's per-molecule coordinate slot holds. Zero
 *             means the caller imposes no limit of its own and only the pipeline cap applies.
 *  @param[out] result Caller-owned buffer, cleared on entry and refilled.
 */
inline smallfield::api::conformer_status_t prepare_molecule_topology_into(std::string_view smiles,
                                                                          smallfield::salt_policy_t salt_policy,
                                                                          std::uint32_t output_atom_capacity,
                                                                          etkdg_prepared_molecule_t &result) {
    using smallfield::mmff94::native_compute_mmff_atom_types;

    result.num_atoms = 0;
    result.bounds_matrix.clear();
    result.chiral_centers.clear();
    result.distance_constraints.clear();
    result.patch_atom_types.clear();
    result.patch_atom_charges.clear();
    result.capacities = device_capacities_t {};

    // Two independent ceilings, and a molecule has to clear both. `max_atoms_per_molecule_k` is what
    // the single-warp pipeline and the flat descriptor hold; `output_atom_capacity` is how wide the
    // caller's coordinate slot is. Gating on the smaller here turns an oversized molecule into a
    // `too_many_atoms_k` rejection at the parse, which is the only place that can refuse it — past
    // this point the wave's slot stride rises to cover it and every molecule in the wave is then
    // written at offsets the caller's buffer does not have.
    smallfield::parse_options_t parse_options;
    parse_options.max_atoms = output_atom_capacity == 0 ? max_atoms_per_molecule_k
                                                        : std::min(output_atom_capacity, max_atoms_per_molecule_k);
    parse_options.metadata_policy = smallfield::compute_metadata_k;
    parse_options.salt_policy = salt_policy;
    smallfield::parse_result_t parsed = smallfield::parse_smiles(smiles, parse_options);
    if (parsed.status != conformer_status_t::success_k) return parsed.status;
    std::unique_ptr<RDKit::RWMol> molecule = std::move(parsed.molecule);
    std::uint32_t const num_atoms = parsed.num_atoms;

    result.metadata = etkdg_molecule_metadata_t {};
    result.metadata.num_heavy_atoms = parsed.num_heavy_atoms;
    result.metadata.num_bonds = parsed.num_bonds;
    result.metadata.molecular_weight = parsed.molecular_weight;

    conformer_status_t typing_status = conformer_status_t::success_k;
    try {
        conformer_status_t const descriptor_status = build_descriptor(*molecule, result.descriptor);
        if (descriptor_status != conformer_status_t::success_k) return descriptor_status;
        build_bounds_matrix(result.descriptor, result.bounds_matrix);
        extract_chiral_centers(*molecule, result.descriptor, result.chiral_centers);
        extract_distance_constraints(result.descriptor, result.bounds_matrix, result.distance_constraints);

        // The device typer runs the same code on the same descriptor, so deciding here which
        // molecules it will reject needs no readback: the answer is already computable.
        std::array<std::uint8_t, max_atoms_per_molecule_k> atom_types_scratch {};
        custd::span<std::uint8_t> atom_types_span {atom_types_scratch.data(), num_atoms};
        if (!native_compute_mmff_atom_types(result.descriptor, atom_types_span))
            typing_status = extract_mmff94_typing_via_rdkit_(*molecule, num_atoms, result.patch_atom_types,
                                                             result.patch_atom_charges);
    }
    catch (...) {
        return conformer_status_t::invalid_smiles_k;
    }
    if (typing_status != conformer_status_t::success_k) return typing_status;

    result.num_atoms = num_atoms;
    result.capacities = compute_device_capacities(result.descriptor,
                                                  static_cast<std::uint32_t>(result.distance_constraints.size()));

    result.metadata.num_atoms = num_atoms;
    result.metadata.num_chiral_centers = static_cast<std::uint32_t>(result.chiral_centers.size());
    result.metadata.num_distance_constraints = static_cast<std::uint32_t>(result.distance_constraints.size());
    // The device stamps the real term counts once it has extracted them. Seeding the fields with
    // the capacities keeps `molecule_complexity_score_` a usable balance estimate in the meantime,
    // since every capacity is monotonic in the term count it bounds.
    result.metadata.num_etk_torsions = result.capacities.etk_torsions;
    result.metadata.num_mmff_bonds = result.capacities.mmff_bonds;
    result.metadata.num_mmff_angles = result.capacities.mmff_angles;
    result.metadata.num_mmff_torsions = result.capacities.mmff_torsions;
    result.metadata.num_mmff_oops = result.capacities.mmff_oops;
    result.metadata.num_mmff_stretchbends = result.capacities.mmff_stretchbends;
    return conformer_status_t::success_k;
}

/**
 *  @brief Batch of molecules packed into a single pinned host buffer.
 *
 *  Type alias onto `wave_slice_pool_t`, which owns one contiguous pinned host allocation plus a
 *  byte-level `wave_slice_layout_t` describing every sub-region. The launch path issues @b one
 *  `cudaMemcpyAsync` of the uploaded prefix into a per-GPU device mirror; the regions past that
 *  prefix exist only on the device and the preprocessing kernels fill them there.
 */
using etkdg_molecule_batch_t = wave_slice_pool_t;

/**
 *  @brief Molecules one batch may carry at @p atom_capacity atoms each, set by the width of the
 *  per-molecule region offsets rather than by memory.
 *
 *  The widest region a molecule claims is its N × N bounds matrix, so the bound is quadratic in the
 *  capacity the caller declares: 4 194 303 molecules at 32 atoms against 116 508 at 192. A wrapped
 *  offset aliases another molecule's rows rather than failing, which is why this is a refusal.
 *
 *  @param[in] atom_capacity The caller's own cap. Zero means none, so the pipeline's cap applies.
 */
inline constexpr std::uint32_t max_molecules_for_atom_capacity(std::uint32_t atom_capacity) noexcept {
    std::uint32_t const effective = atom_capacity == 0 || atom_capacity > max_atoms_per_molecule_k
                                        ? max_atoms_per_molecule_k
                                        : atom_capacity;
    return static_cast<std::uint32_t>(std::numeric_limits<std::uint32_t>::max() /
                                      (static_cast<std::uint64_t>(effective) * effective));
}

/** The bound at the pipeline's own cap, which is the smallest it ever takes. */
inline constexpr std::uint32_t max_molecules_per_batch_k = max_molecules_for_atom_capacity(max_atoms_per_molecule_k);
static_assert(static_cast<std::uint64_t>(max_molecules_per_batch_k) * max_mmff_interactions_per_mol_k <=
                  std::numeric_limits<std::uint32_t>::max(),
              "the bounds matrix must stay the widest per-molecule region the cap is derived from");

/**
 *  @brief Narrow a batch-wide element count to the `std::uint32_t` a per-molecule region offset
 *  carries.
 *
 *  `max_molecules_per_batch_k` is what makes the narrowing lossless, so the assertion names the
 *  invariant the cap exists to hold rather than restating the cast.
 *
 *  @param[in] element_count Elements between the start of the region and this molecule's share.
 */
inline std::uint32_t as_region_offset_(std::uint64_t element_count) noexcept {
    assert(element_count <= std::numeric_limits<std::uint32_t>::max() &&
           "region offset outgrew its 32-bit field; max_molecules_per_batch_k was not honoured");
    return static_cast<std::uint32_t>(element_count);
}

/**
 *  @brief Per-molecule element counts of every region a batch carries.
 *
 *  Packed into one cache-line-aligned struct so the prefix scan over per-molecule sizes is one
 *  pass over a single struct rather than a dozen parallel accumulator chains. The device-filled
 *  regions contribute their capacity rather than their eventual content, since that is what the
 *  host reserves for them. The `valid` field doubles as the dense-output compaction counter.
 *
 *  The two N × N regions lead because they are the only counts quadratic in the atom count, and
 *  a `std::uint32_t` running total of them wraps inside one wave: 104 atoms is 10816 elements per
 *  molecule, so the sum reaches 2^32 at 397094 molecules, under the 400000-molecule wave clamp.
 *  Every remaining region is linear and bounded by `max_mmff_interactions_per_mol_k` per molecule,
 *  which `max_molecules_per_batch_k` covers with room to spare.
 */
struct alignas(64) per_mol_sizes_t {
    /** N × N floats, one row-major bounds matrix per molecule. */
    std::uint64_t bounds;
    /** N × N bytes, one hop-count matrix per molecule. */
    std::uint64_t bond_distance;
    std::uint32_t chirals;
    std::uint32_t distance_constraints;
    /** Per-atom regions: types, charges, VdW params, and both CSR offset arrays. */
    std::uint32_t atoms;
    std::uint32_t etk_interactions;
    std::uint32_t mmff_bonds;
    std::uint32_t mmff_angles;
    std::uint32_t mmff_torsions;
    std::uint32_t mmff_oops;
    std::uint32_t mmff_stretchbends;
    std::uint32_t mmff_interactions;
    std::uint32_t patch_atoms;
    std::uint32_t patch_records;
    /** 0 if the molecule failed preprocessing, 1 otherwise. */
    std::uint32_t valid;
};
static_assert(sizeof(per_mol_sizes_t) == 128, "per_mol_sizes_t must be a whole number of cache lines");

/** Accumulate one molecule's capacities into a running total. */
inline void accumulate_sizes(per_mol_sizes_t &totals, etkdg_molecule_metadata_t const &metadata,
                             device_capacities_t const &capacities) {
    std::uint32_t const num_atoms = metadata.num_atoms;
    // The bounds matrix and the hop-count matrix are the same N × N shape at different element
    // widths, so one square serves both and the two can never disagree.
    std::uint64_t const square_elements = static_cast<std::uint64_t>(num_atoms) * num_atoms;
    totals.bounds += square_elements;
    totals.chirals += metadata.num_chiral_centers;
    totals.distance_constraints += metadata.num_distance_constraints;
    totals.atoms += num_atoms;
    totals.bond_distance += square_elements;
    totals.etk_interactions += capacities.etk_interactions;
    totals.mmff_bonds += capacities.mmff_bonds;
    totals.mmff_angles += capacities.mmff_angles;
    totals.mmff_torsions += capacities.mmff_torsions;
    totals.mmff_oops += capacities.mmff_oops;
    totals.mmff_stretchbends += capacities.mmff_stretchbends;
    totals.mmff_interactions += capacities.mmff_interactions;
    totals.valid += 1;
}

/**
 *  @brief Derive the `wave_slice_layout_t` from total per-region counts. Every region's byte offset is
 *  aligned to `wave_slice_layout_t::region_alignment_k` so the GPU pool slicing in `launch_gpu_pipeline`
 *  inherits correct alignment for free.
 *
 *  The uploaded prefix carries only what the host produces. Everything the preprocessing kernels
 *  write — both CSRs, the ETK torsions, every MMFF term array, the per-atom types and charges, the
 *  typer status and the hop-count matrix — is placed past `bytes_uploaded` and never crosses PCIe.
 *
 *  @param[in] totals Batch-wide element counts of every region.
 *  @param[in] n_valid Molecules that survived preprocessing.
 *  @param[in] torsion_stride Slots the CSD matcher reserves per molecule.
 */
inline wave_slice_layout_t compute_batch_layout(per_mol_sizes_t const &totals, std::uint32_t n_valid,
                                                std::uint32_t torsion_stride) {
    wave_slice_layout_t layout {};
    std::size_t running = 0;
    layout.torsion_stride = torsion_stride;

    layout.bounds_byte_offset = place_region_<float>(running, totals.bounds);
    layout.num_bounds = totals.bounds;
    layout.chirals_byte_offset = place_region_<chiral_center_descriptor_t>(running, totals.chirals);
    layout.num_chirals = totals.chirals;
    layout.distance_constraints_byte_offset = place_region_<distance_constraint_t>(running,
                                                                                   totals.distance_constraints);
    layout.num_distance_constraints = totals.distance_constraints;
    layout.descriptors_byte_offset = place_region_<molecule_descriptor_t>(running, n_valid);
    layout.num_descriptors = n_valid;
    layout.atom_bases_byte_offset = place_region_<std::uint32_t>(running, n_valid);
    layout.num_atom_bases = n_valid;
    layout.mmff_bases_byte_offset = place_region_<mmff94_packed_offsets_t>(running, n_valid);
    layout.num_mmff_bases = n_valid;
    layout.typing_patches_byte_offset = place_region_<mmff94_typing_patch_t>(running, totals.patch_records);
    layout.num_typing_patches = totals.patch_records;
    layout.patch_atom_types_byte_offset = place_region_<std::uint8_t>(running, totals.patch_atoms);
    layout.num_patch_atom_types = totals.patch_atoms;
    layout.patch_atom_charges_byte_offset = place_region_<float>(running, totals.patch_atoms);
    layout.num_patch_atom_charges = totals.patch_atoms;
    layout.metadata_byte_offset = place_region_<etkdg_molecule_metadata_t>(running, n_valid);
    layout.num_metadata = n_valid;
    layout.input_indices_byte_offset = place_region_<std::uint32_t>(running, n_valid);
    layout.num_input_indices = n_valid;

    layout.bytes_uploaded = running;

    std::size_t const torsion_slots = static_cast<std::size_t>(n_valid) * torsion_stride;
    layout.torsions_byte_offset = place_region_<etk_torsion_descriptor_t>(running, torsion_slots);
    layout.num_torsions = torsion_slots;
    layout.torsion_counts_byte_offset = place_region_<std::uint32_t>(running, n_valid);
    layout.num_torsion_counts = n_valid;
    layout.etk_interactions_byte_offset = place_region_<etk_atom_interaction_t>(running, totals.etk_interactions);
    layout.num_etk_interactions = totals.etk_interactions;
    layout.etk_atom_offsets_byte_offset = place_region_<etk_atom_offsets_t>(running, totals.atoms);
    layout.num_etk_atom_offsets = totals.atoms;
    layout.atom_types_byte_offset = place_region_<std::uint8_t>(running, totals.atoms);
    layout.num_atom_types = totals.atoms;
    layout.typing_status_byte_offset = place_region_<std::uint8_t>(running, n_valid);
    layout.num_typing_status = n_valid;
    layout.mmff_atom_vdw_byte_offset = place_region_<mmff94_atom_vdw_params_t>(running, totals.atoms);
    layout.num_mmff_atom_vdw = totals.atoms;
    layout.mmff_atom_charge_byte_offset = place_region_<float>(running, totals.atoms);
    layout.num_mmff_atom_charge = totals.atoms;
    layout.mmff_bonds_byte_offset = place_region_<mmff94_bond_t>(running, totals.mmff_bonds);
    layout.num_mmff_bonds = totals.mmff_bonds;
    layout.mmff_angles_byte_offset = place_region_<mmff94_angle_t>(running, totals.mmff_angles);
    layout.num_mmff_angles = totals.mmff_angles;
    layout.mmff_torsions_byte_offset = place_region_<mmff94_torsion_t>(running, totals.mmff_torsions);
    layout.num_mmff_torsions = totals.mmff_torsions;
    layout.mmff_oops_byte_offset = place_region_<mmff94_oop_t>(running, totals.mmff_oops);
    layout.num_mmff_oops = totals.mmff_oops;
    layout.mmff_stretchbends_byte_offset = place_region_<mmff94_stretchbend_t>(running, totals.mmff_stretchbends);
    layout.num_mmff_stretchbends = totals.mmff_stretchbends;
    layout.mmff_interactions_byte_offset = place_region_<mmff94_atom_interaction_t>(running, totals.mmff_interactions);
    layout.num_mmff_interactions = totals.mmff_interactions;
    layout.mmff_atom_offsets_byte_offset = place_region_<mmff94_atom_offsets_t>(running, totals.atoms);
    layout.num_mmff_atom_offsets = totals.atoms;
    layout.bond_distance_byte_offset = place_region_<std::uint8_t>(running, totals.bond_distance);
    layout.num_bond_distance = totals.bond_distance;
    layout.bytes_total = running;
    return layout;
}

/**
 *  @brief One per-OpenMP-thread record accumulated during `prepare_per_thread_contribs_`.
 *  Every offset field on the metadata holds a @b thread-local index; the pack pass rebases them by
 *  adding the thread's pool-wide base.
 */
struct batch_prep_per_mol_record_t {
    std::uint32_t original_index;
    /** Thread-local base of this molecule inside the MMFF interaction array. The metadata
     *  has no field for it because only the per-atom CSR entries reference it, and those are
     *  device-built; the launch path passes it through `mmff_bases` instead. */
    std::uint32_t mmff_interaction_base;
    etkdg_molecule_metadata_t metadata;
};

/**
 *  @brief Per-OpenMP-thread contribution buffer for the three-stage `preprocess-step` of
 *  `prepare_molecule_batch_into`.
 *
 *  One instance per thread. The per-region `*_bytes` accumulators hold the concatenation of the
 *  host-produced regions for every molecule that thread processed, in append order; the `*_count`
 *  accumulators hold the reservations for the device-filled regions, which carry no host bytes.
 *  Persisted across calls via a file-scope `static` vector so the `std::vector` capacities survive.
 */
struct alignas(64) batch_prep_thread_contribution_t {
    etkdg_prepared_molecule_t scratch;
    std::vector<std::byte> bounds_bytes;
    std::vector<std::byte> chirals_bytes;
    std::vector<std::byte> distance_constraints_bytes;
    std::vector<std::byte> descriptors_bytes;
    std::vector<std::byte> patch_atom_types_bytes;
    std::vector<std::byte> patch_atom_charges_bytes;
    std::vector<mmff94_typing_patch_t> typing_patches;
    /** Running element counts of this thread's share of every device-filled region. The
     *  bytes are never staged: the kernels write them in place after the upload, so the thread
     *  accumulates only the room its molecules claim. */
    std::size_t atom_count = 0;
    std::size_t bond_distance_count = 0;
    std::size_t etk_interaction_count = 0;
    std::size_t mmff_bond_count = 0;
    std::size_t mmff_angle_count = 0;
    std::size_t mmff_torsion_count = 0;
    std::size_t mmff_oop_count = 0;
    std::size_t mmff_stretchbend_count = 0;
    std::size_t mmff_interaction_count = 0;
    std::vector<batch_prep_per_mol_record_t> mols;
    /** Slice-local input indices for every molecule this thread's slice rejected, each
     *  tagged with the reason. Aggregated upstream into `wave_slice_pool_t::rejected_input_indices`. */
    std::vector<smallfield::api::rejected_mol_t> rejected_input_indices;
    std::uint32_t max_atoms_seen = 0;
    std::uint32_t max_bonds_seen = 0;
};

/** Append `bytes` bytes from `source` to the back of `destination`; return the pre-append offset. */
static inline std::size_t append_bytes_(std::vector<std::byte> &destination, void const *source, std::size_t bytes) {
    std::size_t const offset = destination.size();
    if (bytes > 0) {
        destination.resize(offset + bytes);
        std::memcpy(destination.data() + offset, source, bytes);
    }
    return offset;
}

/**
 *  @brief Force every RDKit lazy-init singleton to initialise on the main thread before an OMP
 *  parallel-for races on first access.
 *
 *  Touches the SMILES parser, addHs, `assignStereochemistry` (which pulls in the CIP-rank global
 *  state), the MMFF94 parameter tables, and an ETKDG-style substructure match path. These all
 *  have process-wide lazy-loaded caches that in some RDKit versions are not safe for concurrent
 *  first-time access.
 */
inline void warm_up_rdkit_globals_once_() {
    [[maybe_unused]] static bool const warmed = [] {
        try {
            auto warm = RDKit::SmilesToMol("C[C@H](O)[C@@H](Cl)c1ccccc1");
            if (warm) {
                RDKit::MolOps::addHs(*warm);
                RDKit::MolOps::assignStereochemistry(*warm, /*cleanIt=*/true, /*force=*/true,
                                                     /*flagPossibleStereoCenters=*/true);
                RDKit::MMFF::MMFFMolProperties properties(*warm);
                [[maybe_unused]] bool const parametrizable = properties.isValid();
            }
        }
        catch (...) {
            // Warm-up is best-effort. If it throws, the subsequent parallel-for call hits the
            // same cache-miss path; it costs the race protection, not correctness.
        }
        return true;
    }();
}

/**
 *  @brief First stage of the preprocess-step: parallel per-molecule RDKit parse and byte-append into
 *  each OpenMP thread's contribution buffer.
 *
 *  After return, each thread's contribution carries its share of the input molecules with
 *  thread-local metadata offsets.
 *
 *  @param[in] smiles_list Slice of input SMILES to prepare.
 *  @param[in] salt_policy Whether a multi-fragment input keeps its largest fragment.
 *  @param[in] output_atom_capacity Atoms the caller's per-molecule coordinate slot holds; a wider
 *             molecule is rejected rather than admitted at a stride the caller cannot address.
 *  @param[inout] contribs One entry per OpenMP thread, grown and cleared here.
 */
static inline void prepare_per_thread_contribs_(custd::span<std::string_view const> smiles_list,
                                                smallfield::salt_policy_t salt_policy,
                                                std::uint32_t output_atom_capacity,
                                                std::vector<batch_prep_thread_contribution_t> &contribs) {
    warm_up_rdkit_globals_once_();
    int const n_threads_max = omp_get_max_threads();
    if (static_cast<int>(contribs.size()) < n_threads_max) contribs.resize(n_threads_max);
    for (auto &contribution : contribs) {
        contribution.bounds_bytes.clear();
        contribution.chirals_bytes.clear();
        contribution.distance_constraints_bytes.clear();
        contribution.descriptors_bytes.clear();
        contribution.patch_atom_types_bytes.clear();
        contribution.patch_atom_charges_bytes.clear();
        contribution.typing_patches.clear();
        contribution.mols.clear();
        contribution.rejected_input_indices.clear();
        contribution.max_atoms_seen = 0;
        contribution.max_bonds_seen = 0;
        contribution.atom_count = 0;
        contribution.bond_distance_count = 0;
        contribution.etk_interaction_count = 0;
        contribution.mmff_bond_count = 0;
        contribution.mmff_angle_count = 0;
        contribution.mmff_torsion_count = 0;
        contribution.mmff_oop_count = 0;
        contribution.mmff_stretchbend_count = 0;
        contribution.mmff_interaction_count = 0;
    }

    std::size_t const molecule_count = smiles_list.size();
    // This file is included by `smallfield.cu`, which is compiled with nvcc. For OpenMP to actually
    // parallelize here, the CMake target must pass `-Xcompiler=-fopenmp` through nvcc to the host
    // compiler. Otherwise nvcc silently ignores the pragma and this loop runs single-threaded — a
    // 20x slowdown.
#pragma omp parallel
    {
        int const thread_rank = omp_get_thread_num();
        auto &contribution = contribs[thread_rank];
#pragma omp for schedule(dynamic, 64)
        for (std::size_t molecule = 0; molecule < molecule_count; ++molecule) {
            auto const status = prepare_molecule_topology_into(smiles_list[molecule], salt_policy, output_atom_capacity,
                                                               contribution.scratch);
            auto const &prepared = contribution.scratch;
            if (prepared.num_atoms == 0) {
                contribution.rejected_input_indices.push_back({static_cast<std::uint32_t>(molecule), status});
                continue;
            }
            std::uint32_t const num_atoms = prepared.num_atoms;
            std::uint32_t const num_bonds = prepared.descriptor.num_bonds;
            if (num_atoms > contribution.max_atoms_seen) contribution.max_atoms_seen = num_atoms;
            if (num_bonds > contribution.max_bonds_seen) contribution.max_bonds_seen = num_bonds;

            batch_prep_per_mol_record_t record;
            record.original_index = static_cast<std::uint32_t>(molecule);
            record.metadata = prepared.metadata;
            record.metadata.bounds_offset = static_cast<std::uint32_t>(
                append_bytes_(contribution.bounds_bytes, prepared.bounds_matrix.data(),
                              prepared.bounds_matrix.size() * sizeof(float)) /
                sizeof(float));
            record.metadata.chiral_offset = static_cast<std::uint32_t>(
                append_bytes_(contribution.chirals_bytes, prepared.chiral_centers.data(),
                              prepared.chiral_centers.size() * sizeof(chiral_center_descriptor_t)) /
                sizeof(chiral_center_descriptor_t));
            record.metadata.constraint_offset = static_cast<std::uint32_t>(
                append_bytes_(contribution.distance_constraints_bytes, prepared.distance_constraints.data(),
                              prepared.distance_constraints.size() * sizeof(distance_constraint_t)) /
                sizeof(distance_constraint_t));
            append_bytes_(contribution.descriptors_bytes, &prepared.descriptor, sizeof(molecule_descriptor_t));

            // Every per-atom region shares one base: the molecule's first atom in the batch.
            std::uint32_t const atom_base = static_cast<std::uint32_t>(contribution.atom_count);
            record.metadata.mmff_atom_vdw_offset = atom_base;
            record.metadata.mmff_atom_charge_offset = atom_base;
            record.metadata.mmff_atom_csr_offset = atom_base;
            record.metadata.etk_atom_csr_offset = atom_base;
            contribution.atom_count += num_atoms;

            record.metadata.bond_distance_offset = static_cast<std::uint32_t>(contribution.bond_distance_count);
            contribution.bond_distance_count += static_cast<std::size_t>(num_atoms) * num_atoms;
            record.metadata.etk_interaction_offset = static_cast<std::uint32_t>(contribution.etk_interaction_count);
            contribution.etk_interaction_count += prepared.capacities.etk_interactions;
            record.metadata.mmff_bond_offset = static_cast<std::uint32_t>(contribution.mmff_bond_count);
            contribution.mmff_bond_count += prepared.capacities.mmff_bonds;
            record.metadata.mmff_angle_offset = static_cast<std::uint32_t>(contribution.mmff_angle_count);
            contribution.mmff_angle_count += prepared.capacities.mmff_angles;
            record.metadata.mmff_torsion_offset = static_cast<std::uint32_t>(contribution.mmff_torsion_count);
            contribution.mmff_torsion_count += prepared.capacities.mmff_torsions;
            record.metadata.mmff_oop_offset = static_cast<std::uint32_t>(contribution.mmff_oop_count);
            contribution.mmff_oop_count += prepared.capacities.mmff_oops;
            record.metadata.mmff_stretchbend_offset = static_cast<std::uint32_t>(contribution.mmff_stretchbend_count);
            contribution.mmff_stretchbend_count += prepared.capacities.mmff_stretchbends;
            record.mmff_interaction_base = static_cast<std::uint32_t>(contribution.mmff_interaction_count);
            contribution.mmff_interaction_count += prepared.capacities.mmff_interactions;

            if (!prepared.patch_atom_types.empty()) {
                mmff94_typing_patch_t patch {};
                patch.molecule_index = static_cast<std::uint32_t>(contribution.mols.size());
                patch.atom_base = atom_base;
                patch.patch_base = static_cast<std::uint32_t>(
                    append_bytes_(contribution.patch_atom_types_bytes, prepared.patch_atom_types.data(), num_atoms));
                append_bytes_(contribution.patch_atom_charges_bytes, prepared.patch_atom_charges.data(),
                              num_atoms * sizeof(float));
                patch.atom_count = num_atoms;
                contribution.typing_patches.push_back(patch);
            }

            contribution.mols.push_back(record);
        }
    }
}

/**
 *  @brief Merge every worker's refusals, plus the tail the batch could not address, into @p out.
 *
 *  Each entry carries the slice-local input index and the `conformer_status_t` reason. The tail
 *  past @p admitted_count is refused for the same reason a single oversized molecule is: the batch
 *  cannot address that many atoms, so the caller's `fallback-role` sees it on the path it has.
 */
inline void collect_rejections_into_(std::vector<batch_prep_thread_contribution_t> const &contribs,
                                     std::size_t admitted_count, std::size_t input_count,
                                     std::vector<smallfield::api::rejected_mol_t> &out) {
    out.clear();
    for (auto const &contribution : contribs)
        out.insert(out.end(), contribution.rejected_input_indices.begin(), contribution.rejected_input_indices.end());
    for (std::size_t molecule = admitted_count; molecule < input_count; ++molecule)
        out.push_back({static_cast<std::uint32_t>(molecule), conformer_status_t::too_many_atoms_k});
}

/**
 *  @brief Grow-only scratch for `pack_contribs_into_device_pools_`.
 *
 *  Holds the sort permutation, the CSR-packed per-device assignment list, the per-device CSR
 *  offset array, the shared-batch molecule-to-patch lookup, and the per-device per-molecule
 *  capacity staging the pack loop consumes. One instance per caller; reused across every
 *  `embedding-wave` so the distribute hot path never allocates.
 */
struct distribute_scratch_t {
    /** Where one molecule's host-produced bytes live: which worker produced it, and its position
     *  within that worker's append-ordered blobs. Every region offset on the molecule's metadata is
     *  relative to its own worker, so the pair is enough to address all of it. */
    struct molecule_source_t {
        std::uint32_t thread_rank;
        std::uint32_t index_in_thread;
    };
    /** Length: total_mol_count, in worker-then-append order. */
    std::vector<molecule_source_t> molecule_sources;
    /** Length: total_mol_count. */
    std::vector<std::uint32_t> sort_permutation;
    /** Length: total_mol_count, CSR values. */
    std::vector<std::uint32_t> per_device_flat;
    /** Index into the producing worker's own typing-patch array for each molecule, or
     *  `no_typing_patch_k` when the molecule was typed natively. */
    std::vector<std::uint32_t> patch_of_molecule;
    static constexpr std::uint32_t no_typing_patch_k = 0xFFFFFFFFu;
    // Room for twice the `smallfield::api::max_devices_k` the API admits; `smallfield.cu` asserts the
    // two against each other where it clamps the device list.
    static constexpr std::uint32_t max_devices_k = 32u;
    /** Length: device_count + 1. */
    std::array<std::uint32_t, max_devices_k + 1u> per_device_offsets {};
    std::array<std::vector<device_capacities_t>, max_devices_k> capacities_per_mol {};
};

/**
 *  @brief Pack the workers' contributions straight into `device_count` per-device pinned pools,
 *  partitioned complexity-descending so per-device work distributions are near-identical.
 *
 *  Each molecule's host-produced bytes are copied once, from the worker that produced them into the
 *  assigned device's pool, and its metadata, atom base, MMFF bases and typing patch are rebased to
 *  that device's layout. There is no shared staging pool: a molecule's device is decided from
 *  metadata the workers already carry, before anything is packed, so the wave costs one pass rather
 *  than a pack followed by a copy.
 *
 *  Caller ensures `per_device_out[d].host` can be grown to hold this device's slice.
 *
 *  @param[in] contribs Per-worker contributions filled by `prepare_per_thread_contribs_`.
 *  @param[in] device_count Devices to partition across; 1 packs a single pool.
 *  @param[out] per_device_out One pinned pool per device, filled in place.
 *  @param[inout] scratch Grow-only caller-owned scratch; see `distribute_scratch_t`.
 */
inline void pack_contribs_into_device_pools_(std::vector<batch_prep_thread_contribution_t> const &contribs,
                                             std::uint32_t device_count,
                                             custd::span<etkdg_molecule_batch_t> per_device_out,
                                             distribute_scratch_t &scratch) {

    for (std::uint32_t device_index = 0; device_index < device_count; ++device_index) {
        per_device_out[device_index].num_molecules = 0;
        per_device_out[device_index].observed_max_atoms = 0;
        per_device_out[device_index].layout = wave_slice_layout_t {};
    }

    std::uint32_t total_mol_count = 0;
    for (auto const &contribution : contribs) total_mol_count += static_cast<std::uint32_t>(contribution.mols.size());
    if (total_mol_count == 0) return;

    // Flatten every worker's molecules into one addressable list, and invert each worker's patch
    // list against it so a molecule finds its own patch in constant time.
    if (scratch.molecule_sources.size() < total_mol_count) scratch.molecule_sources.resize(total_mol_count);
    scratch.patch_of_molecule.assign(total_mol_count, distribute_scratch_t::no_typing_patch_k);
    for (std::uint32_t thread_rank = 0, flat = 0; thread_rank < contribs.size(); ++thread_rank) {
        auto const &contribution = contribs[thread_rank];
        std::uint32_t const thread_base = flat;
        for (std::uint32_t index = 0; index < contribution.mols.size(); ++index, ++flat)
            scratch.molecule_sources[flat] = {thread_rank, index};
        for (std::uint32_t patch = 0; patch < contribution.typing_patches.size(); ++patch)
            scratch.patch_of_molecule[thread_base + contribution.typing_patches[patch].molecule_index] = patch;
    }

    auto const metadata_at = [&contribs, &scratch](std::uint32_t flat_index) -> etkdg_molecule_metadata_t const & {
        auto const &source = scratch.molecule_sources[flat_index];
        return contribs[source.thread_rank].mols[source.index_in_thread].metadata;
    };

    // Sort permutation in place on grow-only scratch.
    if (scratch.sort_permutation.size() < total_mol_count) scratch.sort_permutation.resize(total_mol_count);
    if (scratch.per_device_flat.size() < total_mol_count) scratch.per_device_flat.resize(total_mol_count);
    std::iota(scratch.sort_permutation.begin(), scratch.sort_permutation.begin() + total_mol_count, 0u);
    std::sort(scratch.sort_permutation.begin(), scratch.sort_permutation.begin() + total_mol_count,
              [&metadata_at](std::uint32_t first_molecule, std::uint32_t second_molecule) {
                  return molecule_complexity_score_(metadata_at(first_molecule)) >
                         molecule_complexity_score_(metadata_at(second_molecule));
              });

    // CSR offsets for the per-device assignment: device `device_index` gets
    // `ceil((total_mol_count - device_index) / device_count)` mols from the stripe.
    scratch.per_device_offsets[0] = 0u;
    for (std::uint32_t device_index = 0; device_index < device_count; ++device_index) {
        std::uint32_t const count_for_device = (total_mol_count + (device_count - 1u) - device_index) / device_count;
        scratch.per_device_offsets[device_index + 1u] = scratch.per_device_offsets[device_index] + count_for_device;
    }

    // Single scatter pass: a molecule's sorted `rank` maps to device `rank % device_count`, at position
    // `rank / device_count` within that device's CSR range.
    for (std::uint32_t rank = 0; rank < total_mol_count; ++rank) {
        std::uint32_t const device_index = rank % device_count;
        std::uint32_t const position = rank / device_count;
        scratch.per_device_flat[scratch.per_device_offsets[device_index] + position] = scratch.sort_permutation[rank];
    }

    // In-place reverse each per-device range so the kernel bin-dispatch sees complexity-ascending
    // mols first — the smallest `atom-limited-bin`.
    for (std::uint32_t device_index = 0; device_index < device_count; ++device_index) {
        auto *begin = scratch.per_device_flat.data() + scratch.per_device_offsets[device_index];
        auto *end = scratch.per_device_flat.data() + scratch.per_device_offsets[device_index + 1u];
        std::reverse(begin, end);
    }

#pragma omp parallel for num_threads(static_cast<int>(device_count)) schedule(static, 1)
    for (std::uint32_t device_index = 0; device_index < device_count; ++device_index) {
        custd::span<std::uint32_t const> const mol_list {
            scratch.per_device_flat.data() + scratch.per_device_offsets[device_index],
            scratch.per_device_offsets[device_index + 1u] - scratch.per_device_offsets[device_index],
        };
        etkdg_molecule_batch_t &out = per_device_out[device_index];
        std::uint32_t const device_mol_count = static_cast<std::uint32_t>(mol_list.size());

        auto &capacities_per_mol = scratch.capacities_per_mol[device_index];
        if (capacities_per_mol.size() < device_mol_count) capacities_per_mol.resize(device_mol_count);

        per_mol_sizes_t totals {};
        std::uint32_t max_atoms_seen = 0;
        std::uint32_t max_bonds_seen = 0;
        for (std::uint32_t position = 0; position < device_mol_count; ++position) {
            auto const &source = scratch.molecule_sources[mol_list[position]];
            auto const &contribution = contribs[source.thread_rank];
            auto const &meta = contribution.mols[source.index_in_thread].metadata;
            auto const &descriptor = reinterpret_cast<molecule_descriptor_t const *>(
                contribution.descriptors_bytes.data())[source.index_in_thread];
            if (meta.num_atoms > max_atoms_seen) max_atoms_seen = meta.num_atoms;
            if (descriptor.num_bonds > max_bonds_seen) max_bonds_seen = descriptor.num_bonds;
            capacities_per_mol[position] = compute_device_capacities(descriptor, meta.num_distance_constraints);
            accumulate_sizes(totals, meta, capacities_per_mol[position]);
            if (scratch.patch_of_molecule[mol_list[position]] != distribute_scratch_t::no_typing_patch_k) {
                totals.patch_records += 1;
                totals.patch_atoms += meta.num_atoms;
            }
        }

        wave_slice_layout_t device_layout = compute_batch_layout(totals, device_mol_count,
                                                                 std::max<std::uint32_t>(max_bonds_seen, 1u));
        out.num_molecules = device_mol_count;
        out.observed_max_atoms = max_atoms_seen;
        out.layout = device_layout;
        // This runs inside the per-device fan-out, where the pre-allocation discipline wants
        // `unchanged_k`. The per-device pinned pools are the one exception the pipeline documents:
        // they carry no constructor reservation and size themselves from the first wave, which is
        // the largest, so `grown_k` fires once per pool and never again.
        if (device_layout.bytes_uploaded > 0 &&
            out.host.try_reserve(device_layout.bytes_uploaded) == reserve_outcome_t::failed_k) {
            out.num_molecules = 0;
            out.layout = wave_slice_layout_t {};
            continue;
        }
        if (device_mol_count == 0) continue;

        auto const out_view = out.host_view();
        auto bounds_destination = out_view.bounds_matrices();
        auto chirals_destination = out_view.chiral_centers();
        auto constraints_destination = out_view.distance_constraints();
        auto descriptors_destination = out_view.descriptors();
        auto atom_bases_destination = out_view.atom_bases();
        auto mmff_bases_destination = out_view.mmff_bases();
        auto patches_destination = out_view.typing_patches();
        auto patch_types_destination = out_view.patch_atom_types();
        auto patch_charges_destination = out_view.patch_atom_charges();
        auto metadata_destination = out_view.molecule_metadata();
        auto input_indices_destination = out_view.input_indices();

        per_mol_sizes_t running {};
        for (std::uint32_t position = 0; position < device_mol_count; ++position) {
            std::uint32_t const flat_index = mol_list[position];
            auto const &source = scratch.molecule_sources[flat_index];
            auto const &contribution = contribs[source.thread_rank];
            auto const &source_meta = contribution.mols[source.index_in_thread].metadata;
            auto const &capacities = capacities_per_mol[position];
            std::uint32_t const atom_count = source_meta.num_atoms;

            std::memcpy(bounds_destination.data() + running.bounds,
                        reinterpret_cast<float const *>(contribution.bounds_bytes.data()) + source_meta.bounds_offset,
                        static_cast<std::size_t>(atom_count) * atom_count * sizeof(float));
            std::memcpy(chirals_destination.data() + running.chirals,
                        reinterpret_cast<chiral_center_descriptor_t const *>(contribution.chirals_bytes.data()) +
                            source_meta.chiral_offset,
                        source_meta.num_chiral_centers * sizeof(chiral_center_descriptor_t));
            std::memcpy(
                constraints_destination.data() + running.distance_constraints,
                reinterpret_cast<distance_constraint_t const *>(contribution.distance_constraints_bytes.data()) +
                    source_meta.constraint_offset,
                source_meta.num_distance_constraints * sizeof(distance_constraint_t));
            descriptors_destination[position] = reinterpret_cast<molecule_descriptor_t const *>(
                contribution.descriptors_bytes.data())[source.index_in_thread];

            std::uint32_t const patch_index = scratch.patch_of_molecule[flat_index];
            if (patch_index != distribute_scratch_t::no_typing_patch_k) {
                auto const &source_patch = contribution.typing_patches[patch_index];
                std::memcpy(patch_types_destination.data() + running.patch_atoms,
                            contribution.patch_atom_types_bytes.data() + source_patch.patch_base, atom_count);
                std::memcpy(patch_charges_destination.data() + running.patch_atoms,
                            reinterpret_cast<float const *>(contribution.patch_atom_charges_bytes.data()) +
                                source_patch.patch_base,
                            atom_count * sizeof(float));
                mmff94_typing_patch_t patch {};
                patch.molecule_index = position;
                patch.atom_base = running.atoms;
                patch.patch_base = running.patch_atoms;
                patch.atom_count = atom_count;
                patches_destination[running.patch_records] = patch;
                running.patch_records += 1;
                running.patch_atoms += atom_count;
            }

            etkdg_molecule_metadata_t destination_meta = source_meta;
            destination_meta.bounds_offset = as_region_offset_(running.bounds);
            destination_meta.chiral_offset = running.chirals;
            destination_meta.constraint_offset = running.distance_constraints;
            destination_meta.torsion_offset = position * device_layout.torsion_stride;
            destination_meta.mmff_atom_vdw_offset = running.atoms;
            destination_meta.mmff_atom_charge_offset = running.atoms;
            destination_meta.mmff_atom_csr_offset = running.atoms;
            destination_meta.etk_atom_csr_offset = running.atoms;
            destination_meta.bond_distance_offset = as_region_offset_(running.bond_distance);
            destination_meta.etk_interaction_offset = running.etk_interactions;
            destination_meta.mmff_bond_offset = running.mmff_bonds;
            destination_meta.mmff_angle_offset = running.mmff_angles;
            destination_meta.mmff_torsion_offset = running.mmff_torsions;
            destination_meta.mmff_oop_offset = running.mmff_oops;
            destination_meta.mmff_stretchbend_offset = running.mmff_stretchbends;
            metadata_destination[position] = destination_meta;
            input_indices_destination[position] = contribution.mols[source.index_in_thread].original_index;
            atom_bases_destination[position] = running.atoms;
            mmff_bases_destination[position] = {running.mmff_bonds,        running.mmff_angles,
                                                running.mmff_torsions,     running.mmff_oops,
                                                running.mmff_stretchbends, running.mmff_interactions};

            accumulate_sizes(running, source_meta, capacities);
        }
    }
}

/**
 *  @brief Preprocess + distribute entry: produce `device_count` per-device pinned batch pools with
 *  a complexity-descending stripe partition. Each device's pool holds only its assigned mols in
 *  complexity-ascending order — atom-count-ascending on average — for kernel bin dispatch.
 *
 *  The workers' contributions are packed straight into the per-device pools, so a molecule's bytes
 *  are written once. Two file-scope `static` scratch instances hold the per-worker contributions
 *  and the partition buffers so neither allocation fires on the hot path; that is also what makes
 *  this entry non-reentrant, and the reason a caller must not drive it from two threads.
 *
 *  @param[in] smiles_list Slice of input SMILES to prepare, at most `max_molecules_per_batch_k` of
 *             which are admitted; the tail past that is refused with `too_many_atoms_k`.
 *  @param[in] device_count Devices to partition across; 1 fills a single pool.
 *  @param[in] output_atom_capacity Atoms the caller's per-molecule coordinate slot holds. Every
 *             admitted molecule fits it, so a slice's `observed_max_atoms` — the stride the launch
 *             path derives its coordinate buffers from — can never outgrow the caller's own.
 *  @param[out] per_device_out One pinned pool per device.
 *  @param[out] rejected_input_indices_out Optional sink for the slice-local rejection records.
 *  @param[in] salt_policy Whether a multi-fragment input keeps its largest fragment.
 */
inline void prepare_per_device_batches_into(
    custd::span<std::string_view const> smiles_list, std::uint32_t device_count, std::uint32_t output_atom_capacity,
    custd::span<etkdg_molecule_batch_t> per_device_out,
    std::vector<smallfield::api::rejected_mol_t> *rejected_input_indices_out = nullptr,
    smallfield::salt_policy_t salt_policy = smallfield::keep_largest_fragment_k) {
    static std::vector<batch_prep_thread_contribution_t> contribs;
    static distribute_scratch_t scratch;

    // Past this many, a molecule's own region offsets stop fitting the 32-bit fields that carry
    // them, so the tail is refused here rather than packed at a wrapped offset that aliases an
    // earlier molecule's rows. Whoever sizes the wave applies the same bound, so reaching this is a
    // caller that did not.
    std::size_t const admitted_count = std::min<std::size_t>(smiles_list.size(),
                                                             max_molecules_for_atom_capacity(output_atom_capacity));
    prepare_per_thread_contribs_(smiles_list.first(admitted_count), salt_policy, output_atom_capacity, contribs);
    pack_contribs_into_device_pools_(contribs, device_count, per_device_out, scratch);
    if (rejected_input_indices_out != nullptr)
        collect_rejections_into_(contribs, admitted_count, smiles_list.size(), *rejected_input_indices_out);
}

/**
 *  @brief In-place `preprocess-step` driver for a single pool: the one-device partition.
 *
 *  Recycling the pinned buffer across `embedding-waves` is the lever that kills pinned-memory
 *  allocator contention at 8-GPU scale — `cudaHostAlloc`/`cudaFreeHost` serialize on a driver lock.
 *
 *  @param[in] smiles_list Slice of input SMILES to prepare.
 *  @param[in] output_atom_capacity Atoms the caller's per-molecule coordinate slot holds; a wider
 *             molecule is refused rather than raising the whole wave's slot stride past it.
 *  @param[out] batch Pinned pool, refilled in place.
 *  @param[in] salt_policy Whether a multi-fragment input keeps its largest fragment.
 */
inline void prepare_molecule_batch_into(custd::span<std::string_view const> smiles_list,
                                        std::uint32_t output_atom_capacity, etkdg_molecule_batch_t &batch,
                                        smallfield::salt_policy_t salt_policy = smallfield::keep_largest_fragment_k) {
    prepare_per_device_batches_into(smiles_list, 1u, output_atom_capacity, {&batch, 1}, &batch.rejected_input_indices,
                                    salt_policy);
}

/**
 *  @brief Convenience wrapper returning a fresh batch, allocating a new pinned buffer every call.
 *
 *  Serves call sites outside the pipelined dispatcher, such as tests and one-shot invocations. The
 *  hot path should prefer `prepare_molecule_batch_into`. The default @p output_atom_capacity is the
 *  pipeline's own cap, which is what a call site with no coordinate buffer of its own wants.
 */
inline etkdg_molecule_batch_t prepare_molecule_batch(
    custd::span<std::string_view const> smiles_list,
    smallfield::salt_policy_t salt_policy = smallfield::keep_largest_fragment_k,
    std::uint32_t output_atom_capacity = max_atoms_per_molecule_k) {
    etkdg_molecule_batch_t batch {};
    prepare_molecule_batch_into(smiles_list, output_atom_capacity, batch, salt_policy);
    return batch;
}

} // namespace smallfield::preprocess
