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

#include <algorithm>   // `std::copy_n`, `std::max`, `std::min`, `std::reverse`, `std::sort`
#include <array>       // `std::array`
#include <limits>      // `std::numeric_limits`
#include <memory>      // `std::unique_ptr`
#include <numeric>     // `std::iota`
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
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/api/wave_slice_pool.hpp"
#include "smallfield/molecule_parse.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/storage.hpp"
#include "smallfield/types/views.hpp"
#include "smallfield/preprocess/bounds_matrix_builder.hpp"
#include "smallfield/preprocess/csd_torsion_dispatcher.hpp"
#include "smallfield/preprocess/device_capacities.hpp"
#include "smallfield/preprocess/molecule_complexity.hpp"
#include "smallfield/mmff94/mmff94_typer.hpp"
#include "smallfield/types/mmff94_bounds.hpp"

namespace smallfield::preprocess {
using smallfield::api::conformer_status_t;
using smallfield::api::place_pool_region_;
using smallfield::api::reserve_outcome_t;
using smallfield::api::wave_slice_layout_t;
using smallfield::api::wave_slice_pool_t;
using smallfield::types::atom_flag_aromatic_k;
using smallfield::types::atom_flag_hbond_acceptor_k;
using smallfield::types::atom_flag_hbond_donor_h_k;
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
using smallfield::types::lower_distance_bound_at;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::max_bonds_k;
using smallfield::types::max_mmff_interactions_per_mol_k;
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
using smallfield::types::packed_matrix;
using smallfield::types::reset_square_matrix;
using smallfield::types::upper_distance_bound_at;

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

#pragma region Chiral Centers

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

        // Which floor applies is decided by whether RDKit's own bound transfers into our volume
        // convention, and it does exactly when the center has no fourth bonded neighbor.
        chiral_center.minimum_volume = count >= 4 ? min_four_coordinate_volume_k : min_three_coordinate_volume_k;

        chiral_centers.push_back(chiral_center);
    }
}

#pragma endregion Chiral Centers

#pragma region Distance Constraints

/**
 *  @brief Extract 1-2 bond and 1-3 angle-derived distance constraints for the ETK refinement stage.
 *
 *  @param[in] descriptor Flat descriptor supplying the bond list and the CSR adjacency.
 *  @param[in] bounds Square N × N distance bounds over the molecule's atoms, from `build_bounds_matrix`.
 *  @param[out] constraints Filled with 1-2 bond and 1-3 angle distance constraints.
 */
inline void extract_distance_constraints(molecule_descriptor_t const &descriptor, packed_matrix<float const> bounds,
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

        distance_constraint_t constraint {};
        constraint.atom_index_i = static_cast<std::uint8_t>(atom_i);
        constraint.atom_index_j = static_cast<std::uint8_t>(atom_j);
        constraint.upper_bound_angstrom = upper_distance_bound_at(bounds, atom_i, atom_j);
        constraint.lower_bound_angstrom = lower_distance_bound_at(bounds, atom_i, atom_j);
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

                distance_constraint_t constraint {};
                constraint.atom_index_i = static_cast<std::uint8_t>(first_neighbor);
                constraint.atom_index_j = static_cast<std::uint8_t>(second_neighbor);
                constraint.upper_bound_angstrom = upper_distance_bound_at(bounds, first_neighbor, second_neighbor);
                constraint.lower_bound_angstrom = lower_distance_bound_at(bounds, first_neighbor, second_neighbor);
                constraint.penalty_weight = 5.0f;
                constraints.push_back(constraint);
            }
        }
    }
}

#pragma endregion Distance Constraints

#pragma region ETK Interaction Lists

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

#pragma endregion ETK Interaction Lists

#pragma region MMFF94 Extraction

/**
 *  @brief Build the per-atom MMFF94 CSR from the term arrays already in @p result.
 *
 *  Bonded terms only: the kernel enumerates nonbonded pairs on the fly from `bond_distance`, so they
 *  never enter the CSR. Reads the five term vectors and writes `mmff_atom_offsets` and
 *  `mmff_interactions`, which is everything here that RDKit is not needed for.
 *
 *  @param[inout] result Prepared molecule; the term arrays are read and the two CSR arrays filled.
 *  @param[in] num_atoms Atoms the CSR spans.
 */
static inline void build_mmff_atom_csr_(etkdg_prepared_molecule_t &result, std::uint32_t num_atoms) {
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
}

/**
 *  @brief Extract MMFF94 force-field parameters and build per-atom interaction lists.
 *
 *  Every MMFF94 array on @p result is emptied on entry, so the destination needs no preparation of
 *  its own and a refusal below carries nothing of the previous molecule forward.
 *
 *  @param[in] molecule RDKit molecule with explicit hydrogens.
 *  @param[in] descriptor Flat descriptor supplying the bond list and the CSR adjacency.
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
    // `result` is caller-owned scratch reused across molecules, so every array this function writes is
    // emptied here rather than at whichever call site happened to remember. That covers the two it
    // resizes as well, because the element-gate and validity refusals below return ahead of the resize.
    result.mmff_atom_vdw.clear();
    result.mmff_atom_charge.clear();
    result.mmff_bonds.clear();
    result.mmff_angles.clear();
    result.mmff_torsions.clear();
    result.mmff_oops.clear();
    result.mmff_stretchbends.clear();
    result.mmff_interactions.clear();
    result.mmff_atom_offsets.clear();

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

    // Build angle and stretch-bend terms — iterate atoms as angle centers, enumerate neighbor pairs
    // via CSR. Both term classes are defined over the same i-j-k triple, so one walk emits both.
    // Neither emission is a precondition of the other: RDKit answers each triple independently and
    // each `push_back` lands in its own vector, so each vector's sequence is what the two separate
    // walks produced.
    for (std::uint32_t atom_j = 0; atom_j < num_atoms; ++atom_j) {
        std::uint16_t const nstart = descriptor.neighbor_offset[atom_j];
        std::uint16_t const nend = descriptor.neighbor_offset[atom_j + 1];
        for (std::uint16_t first_entry = nstart; first_entry < nend; ++first_entry) {
            for (std::uint16_t second_entry = first_entry + 1; second_entry < nend; ++second_entry) {
                std::uint32_t const atom_i = descriptor.neighbor_atom[first_entry];
                std::uint32_t const atom_k = descriptor.neighbor_atom[second_entry];

                unsigned int angle_type;
                ForceFields::MMFF::MMFFAngle angle_params;
                if (mmff_props.getMMFFAngleBendParams(mutable_molecule, atom_i, atom_j, atom_k, angle_type,
                                                      angle_params)) {
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

                unsigned int stretchbend_type;
                ForceFields::MMFF::MMFFStbn stretchbend_params;
                ForceFields::MMFF::MMFFBond stretchbend_bond_params[2];
                // Its own `MMFFAngle`, not the one the angle lookup filled: RDKit answers the two
                // queries separately and the stretch-bend term carries the answer to its own.
                ForceFields::MMFF::MMFFAngle stretchbend_angle_params;
                if (!mmff_props.getMMFFStretchBendParams(mutable_molecule, atom_i, atom_j, atom_k, stretchbend_type,
                                                         stretchbend_params, stretchbend_bond_params,
                                                         stretchbend_angle_params))
                    continue;
                result.mmff_stretchbends.push_back(
                    {.atom_i = static_cast<std::uint8_t>(atom_i),
                     .atom_j = static_cast<std::uint8_t>(atom_j),
                     .atom_k = static_cast<std::uint8_t>(atom_k),
                     .padding = 0,
                     .kba_ijk = static_cast<float>(stretchbend_params.kbaIJK),
                     .kba_kji = static_cast<float>(stretchbend_params.kbaKJI),
                     .equilibrium_angle = static_cast<float>(stretchbend_angle_params.theta0 * M_PI / 180.0),
                     .equilibrium_distance_ij = static_cast<float>(stretchbend_bond_params[0].r0),
                     .equilibrium_distance_kj = static_cast<float>(stretchbend_bond_params[1].r0),
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

    build_mmff_atom_csr_(result, num_atoms);
    return smallfield::api::conformer_status_t::success_k;
}

#pragma endregion MMFF94 Extraction

#pragma region Single Molecule

/**
 *  @brief Fill a caller-owned @p result with the full preprocessing output for one SMILES.
 *
 *  The single-molecule entry the tests drive; the batched path runs `prepare_molecule_topology_into`
 *  instead. The caller-owned `result` is meant to be reused across molecules — each molecule's vectors
 *  keep their capacity from the previous call, so `push_back` stays on the hot path without heap
 *  allocation after the first few molecules. Sets `result.num_atoms = 0` on failure and returns the
 *  rejection reason; on success returns `conformer_status_t::success_k`. Callers that don't need the
 *  per-molecule reason code can still test `result.num_atoms == 0` as the in-band drop signal.
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
        auto const bounds = reset_square_matrix(result.bounds_matrix, descriptor.num_atoms, 0.0f);
        build_bounds_matrix(descriptor, bounds);
        extract_chiral_centers(*molecule, descriptor, result.chiral_centers);
        extract_etk_torsions_native(descriptor, result.etk_torsions);
        extract_distance_constraints(descriptor, bounds, result.distance_constraints);
        build_etk_interaction_lists(result);
        mmff_status = extract_mmff94_params_via_rdkit(*molecule, descriptor, result);
    }
    catch (...) {
        result.num_atoms = 0;
        return conformer_status_t::invalid_smiles_k;
    }
    if (mmff_status != conformer_status_t::success_k) {
        // MMFF could not parametrise this molecule — say, an atomic number outside the MMFF94 set.
        // The mmff_* vectors hold whatever the extractor reached before it refused and the ETK fields
        // above are fully populated, so zeroing `num_atoms` is what makes downstream packing skip it.
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
        auto const bounds = reset_square_matrix(result.bounds_matrix, result.descriptor.num_atoms, 0.0f);
        build_bounds_matrix(result.descriptor, bounds);
        extract_chiral_centers(*molecule, result.descriptor, result.chiral_centers);
        extract_distance_constraints(result.descriptor, bounds, result.distance_constraints);

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
#pragma endregion Single Molecule

using etkdg_molecule_batch_t = wave_slice_pool_t;

#pragma region Batch Layout

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
 *  aligned to `pool_region_alignment_k` so the GPU pool slicing in `launch_gpu_pipeline` inherits
 *  correct alignment for free.
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

    layout.bounds = place_pool_region_<float>(running, totals.bounds);
    layout.chirals = place_pool_region_<chiral_center_descriptor_t>(running, totals.chirals);
    layout.distance_constraints = place_pool_region_<distance_constraint_t>(running, totals.distance_constraints);
    layout.descriptors = place_pool_region_<molecule_descriptor_t>(running, n_valid);
    layout.atom_bases = place_pool_region_<std::uint32_t>(running, n_valid);
    layout.mmff_bases = place_pool_region_<mmff94_packed_offsets_t>(running, n_valid);
    layout.typing_patches = place_pool_region_<mmff94_typing_patch_t>(running, totals.patch_records);
    layout.patch_atom_types = place_pool_region_<std::uint8_t>(running, totals.patch_atoms);
    layout.patch_atom_charges = place_pool_region_<float>(running, totals.patch_atoms);
    layout.metadata = place_pool_region_<etkdg_molecule_metadata_t>(running, n_valid);
    layout.input_indices = place_pool_region_<std::uint32_t>(running, n_valid);

    layout.bytes_uploaded = running;

    std::size_t const torsion_slots = static_cast<std::size_t>(n_valid) * torsion_stride;
    layout.torsions = place_pool_region_<etk_torsion_descriptor_t>(running, torsion_slots);
    layout.torsion_counts = place_pool_region_<std::uint32_t>(running, n_valid);
    layout.etk_interactions = place_pool_region_<etk_atom_interaction_t>(running, totals.etk_interactions);
    layout.etk_atom_offsets = place_pool_region_<etk_atom_offsets_t>(running, totals.atoms);
    layout.atom_types = place_pool_region_<std::uint8_t>(running, totals.atoms);
    layout.typing_status = place_pool_region_<std::uint8_t>(running, n_valid);
    layout.mmff_atom_vdw = place_pool_region_<mmff94_atom_vdw_params_t>(running, totals.atoms);
    layout.mmff_atom_charge = place_pool_region_<float>(running, totals.atoms);
    layout.mmff_bonds = place_pool_region_<mmff94_bond_t>(running, totals.mmff_bonds);
    layout.mmff_angles = place_pool_region_<mmff94_angle_t>(running, totals.mmff_angles);
    layout.mmff_torsions = place_pool_region_<mmff94_torsion_t>(running, totals.mmff_torsions);
    layout.mmff_oops = place_pool_region_<mmff94_oop_t>(running, totals.mmff_oops);
    layout.mmff_stretchbends = place_pool_region_<mmff94_stretchbend_t>(running, totals.mmff_stretchbends);
    layout.mmff_interactions = place_pool_region_<mmff94_atom_interaction_t>(running, totals.mmff_interactions);
    layout.mmff_atom_offsets = place_pool_region_<mmff94_atom_offsets_t>(running, totals.atoms);
    layout.bond_distance = place_pool_region_<std::uint8_t>(running, totals.bond_distance);
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
 *  One instance per thread. Each region accumulator holds the concatenation of the host-produced
 *  region for every molecule that thread processed, in append order, at the element type that region
 *  carries; the `*_count` accumulators hold the reservations for the device-filled regions, which
 *  carry no host elements at all.
 *  Persisted across calls via a file-scope `static` vector so the `std::vector` capacities survive.
 */
struct alignas(64) batch_prep_thread_contribution_t {
    etkdg_prepared_molecule_t scratch;
    std::vector<float> bounds;
    std::vector<chiral_center_descriptor_t> chirals;
    std::vector<distance_constraint_t> distance_constraints;
    std::vector<molecule_descriptor_t> descriptors;
    std::vector<std::uint8_t> patch_atom_types;
    std::vector<float> patch_atom_charges;
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
        contribution.bounds.clear();
        contribution.chirals.clear();
        contribution.distance_constraints.clear();
        contribution.descriptors.clear();
        contribution.patch_atom_types.clear();
        contribution.patch_atom_charges.clear();
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
            record.metadata.bounds_offset = static_cast<std::uint32_t>(contribution.bounds.size());
            contribution.bounds.insert(contribution.bounds.end(), prepared.bounds_matrix.begin(),
                                       prepared.bounds_matrix.end());
            record.metadata.chiral_offset = static_cast<std::uint32_t>(contribution.chirals.size());
            contribution.chirals.insert(contribution.chirals.end(), prepared.chiral_centers.begin(),
                                        prepared.chiral_centers.end());
            record.metadata.constraint_offset = static_cast<std::uint32_t>(contribution.distance_constraints.size());
            contribution.distance_constraints.insert(contribution.distance_constraints.end(),
                                                     prepared.distance_constraints.begin(),
                                                     prepared.distance_constraints.end());
            contribution.descriptors.push_back(prepared.descriptor);

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
                // One base addresses both arrays, so both take the same count from the same molecule.
                patch.patch_base = static_cast<std::uint32_t>(contribution.patch_atom_types.size());
                contribution.patch_atom_types.insert(contribution.patch_atom_types.end(),
                                                     prepared.patch_atom_types.data(),
                                                     prepared.patch_atom_types.data() + num_atoms);
                contribution.patch_atom_charges.insert(contribution.patch_atom_charges.end(),
                                                       prepared.patch_atom_charges.data(),
                                                       prepared.patch_atom_charges.data() + num_atoms);
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
 *  @brief Decide which device every admitted molecule lands on, touching none of its bytes.
 *
 *  Flattens the workers' molecules into one addressable list, inverts each worker's typing-patch list
 *  against it, ranks that list complexity-descending, and stripes the ranking across the devices so
 *  each device draws evenly from every complexity band. Each device's CSR range is then reversed, so
 *  the kernel bin-dispatch meets the smallest `atom-limited-bin` first.
 *
 *  Pure index computation over the metadata the workers already carry: it reads no molecule payload,
 *  writes no pool, and needs neither RDKit nor a device to run.
 *
 *  @param[in] contribs Per-worker contributions filled by `prepare_per_thread_contribs_`.
 *  @param[in] device_count Devices to partition across; 1 assigns every molecule to one range.
 *  @param[inout] scratch Grow-only caller-owned scratch; `molecule_sources`, `patch_of_molecule`,
 *                `sort_permutation`, `per_device_flat` and `per_device_offsets` are filled here.
 *  @return Molecules assigned, which is the sum of the `device_count` CSR range lengths.
 */
static inline std::uint32_t assign_molecules_to_devices_(std::vector<batch_prep_thread_contribution_t> const &contribs,
                                                         std::uint32_t device_count, distribute_scratch_t &scratch) {

    std::uint32_t total_mol_count = 0;
    for (auto const &contribution : contribs) total_mol_count += static_cast<std::uint32_t>(contribution.mols.size());

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

    return total_mol_count;
}

/**
 *  @brief Pack the workers' contributions straight into `device_count` per-device pinned pools,
 *  partitioned complexity-descending so per-device work distributions are near-identical.
 *
 *  Each molecule's host-produced bytes are copied once, from the worker that produced them into the
 *  pool of the device `assign_molecules_to_devices_` gave it, and its metadata, atom base, MMFF bases
 *  and typing patch are rebased to that device's layout. There is no shared staging pool: a
 *  molecule's device is decided from
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

    std::uint32_t const assigned_mol_count = assign_molecules_to_devices_(contribs, device_count, scratch);
    if (assigned_mol_count == 0) return;

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
            auto const &descriptor = contribution.descriptors[source.index_in_thread];
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

            std::copy_n(contribution.bounds.data() + source_meta.bounds_offset,
                        static_cast<std::size_t>(atom_count) * atom_count, bounds_destination.data() + running.bounds);
            std::copy_n(contribution.chirals.data() + source_meta.chiral_offset, source_meta.num_chiral_centers,
                        chirals_destination.data() + running.chirals);
            std::copy_n(contribution.distance_constraints.data() + source_meta.constraint_offset,
                        source_meta.num_distance_constraints,
                        constraints_destination.data() + running.distance_constraints);
            descriptors_destination[position] = contribution.descriptors[source.index_in_thread];

            std::uint32_t const patch_index = scratch.patch_of_molecule[flat_index];
            if (patch_index != distribute_scratch_t::no_typing_patch_k) {
                auto const &source_patch = contribution.typing_patches[patch_index];
                std::copy_n(contribution.patch_atom_types.data() + source_patch.patch_base, atom_count,
                            patch_types_destination.data() + running.patch_atoms);
                std::copy_n(contribution.patch_atom_charges.data() + source_patch.patch_base, atom_count,
                            patch_charges_destination.data() + running.patch_atoms);
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

#pragma endregion Batch Layout

} // namespace smallfield::preprocess
