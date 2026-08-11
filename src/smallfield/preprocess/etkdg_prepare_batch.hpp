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

#include <algorithm>   // `std::max`, `std::min`
#include <array>       // `std::array`
#include <memory>      // `std::unique_ptr`
#include <mutex>       // `std::call_once`, `std::once_flag`
#include <optional>    // `std::optional`, `std::nullopt`
#include <span>        // `std::span`
#include <sstream>     // RDKit's `AtomTyper.h` needs it but does not include it
#include <string_view> // `std::string_view`
#include <utility>     // `std::move`
#include <vector>      // `std::vector`

#include "smallfield/module_fwd.hpp"

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
#include "smallfield/parameters/forcefield_tables.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/preprocess/csd_torsion_dispatcher.hpp"
#include "smallfield/preprocess/device_capacities.hpp"
#include "smallfield/preprocess/molecule_complexity.hpp"
#include "smallfield/mmff94/mmff94_typer.hpp"
#include "smallfield/mmff94/mmff94_param_cache.hpp"
#include "smallfield/types/mmff94_bounds.hpp"

namespace smallfield::preprocess {
using smallfield::api::conformer_status_t;
using smallfield::api::place_region_;
using smallfield::api::wave_slice_layout_t;
using smallfield::api::wave_slice_pool_t;
using smallfield::mmff94::make_angle_key_;
using smallfield::mmff94::make_bond_stretch_key_;
using smallfield::mmff94::make_oop_key_;
using smallfield::mmff94::make_stretchbend_key_;
using smallfield::mmff94::make_torsion_key_;
using smallfield::mmff94::mmff_angle_cache;
using smallfield::mmff94::mmff_angle_cached_t;
using smallfield::mmff94::mmff_bond_cache;
using smallfield::mmff94::mmff_bond_cached_t;
using smallfield::mmff94::mmff_cache;
using smallfield::mmff94::mmff_oop_cache;
using smallfield::mmff94::mmff_oop_cached_t;
using smallfield::mmff94::mmff_stretchbend_cache;
using smallfield::mmff94::mmff_stretchbend_cached_t;
using smallfield::mmff94::mmff_torsion_cache;
using smallfield::mmff94::mmff_torsion_cached_t;
using smallfield::mmff94::pack_bond_signature_;
using smallfield::parameters::ideal_angle_radians;
using smallfield::parameters::ideal_bond_length_angstrom;
using smallfield::parameters::vdw_lower_bound_angstrom;
using smallfield::types::atom_flag_aromatic_k;
using smallfield::types::atom_has_flag;
using smallfield::types::atom_role_first_k;
using smallfield::types::atom_role_fourth_k;
using smallfield::types::atom_role_second_k;
using smallfield::types::atom_role_third_k;
using smallfield::types::bond_flag_aromatic_k;
using smallfield::types::bond_flag_in_ring_k;
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
using smallfield::types::hybridization_sp2_k;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::max_bonds_k;
using smallfield::types::max_mmff_interactions_per_mol_k;
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

/** Which side of the double bond the two reference substituents sit on. */
enum double_bond_geometry_t : std::uint8_t {
    double_bond_trans_k = 0,
    double_bond_cis_k = 1,
};

/**
 *  @brief End-to-end distance of a 1-4 chain `first — second = third — fourth` whose
 *  central bond is planar, at a torsion of 0 for cis or pi for trans.
 *
 *  Unlike a freely rotatable 1-4 pair, a stereo-defined double bond fixes this distance,
 *  which is how ETKDG enforces E/Z through the bounds matrix.
 */
inline float stereo_one_four_distance(molecule_descriptor_t const &descriptor, std::uint32_t first,
                                      std::uint32_t second, std::uint32_t third, std::uint32_t fourth,
                                      std::uint32_t bond_first_second, std::uint32_t bond_second_third,
                                      std::uint32_t bond_third_fourth, double_bond_geometry_t geometry) {
    auto bond_length = [&](std::uint32_t bond_index, std::uint32_t first_atom, std::uint32_t second_atom) {
        bool const aromatic = (descriptor.bond_flags[bond_index] & bond_flag_aromatic_k) != 0;
        return ideal_bond_length_angstrom(descriptor.element[first_atom], descriptor.element[second_atom],
                                          descriptor.bond_order[bond_index], aromatic);
    };
    float const length_12 = bond_length(bond_first_second, first, second);
    float const length_23 = bond_length(bond_second_third, second, third);
    float const length_34 = bond_length(bond_third_fourth, third, fourth);

    float const angle_123 = ideal_angle_radians(descriptor.hybridization[second]);
    float const angle_234 = ideal_angle_radians(descriptor.hybridization[third]);
    float const cos_123 = std::cos(angle_123), sin_123 = std::sin(angle_123);
    float const cos_234 = std::cos(angle_234), sin_234 = std::sin(angle_234);
    float const cos_torsion = geometry == double_bond_cis_k ? 1.0f : -1.0f;

    float const squared = length_12 * length_12 + length_23 * length_23 + length_34 * length_34 -
                          2.0f * length_12 * length_23 * cos_123 - 2.0f * length_23 * length_34 * cos_234 +
                          2.0f * length_12 * length_34 * (cos_123 * cos_234 - sin_123 * sin_234 * cos_torsion);
    return std::sqrt(std::max(0.01f, squared));
}

/**
 *  @brief Native zero-heap builder for the distance bounds matrix.
 *
 *  Replaces RDKit's `setTopolBounds`, which allocates a fresh `DistGeom::BoundsMatrix` plus two
 *  `boost::shared_array<>` per molecule and walks RDKit's atom typer, roughly 140 KB of throwaway
 *  heap per call. This version reads only from the flat `molecule_descriptor_t` and the small
 *  `forcefield_tables.hpp`, and writes directly into `bounds`.
 *
 *  The values are close but not bit-identical to `setTopolBounds`'s output. Downstream triangle
 *  smoothing and FIRE refinement, both on the GPU, absorb the small residual; `bench_conformers`
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
    fill_bond_distances(descriptor, bond_hops, num_atoms);

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
        std::uint32_t const first_atom = descriptor.bond_atom_i[bond];
        std::uint32_t const second_atom = descriptor.bond_atom_j[bond];
        bool const aromatic = (descriptor.bond_flags[bond] & bond_flag_aromatic_k) != 0;
        float const ideal = ideal_bond_length_angstrom(descriptor.element[first_atom], descriptor.element[second_atom],
                                                       descriptor.bond_order[bond], aromatic);
        set_pair(first_atom, second_atom, ideal * (1.0f - bond_tolerance_k), ideal * (1.0f + bond_tolerance_k));
    }

    // set13 — one 1-3 bound per pair of neighbors around every center atom.
    // Loop is O(N × d²) where d is the typical degree (small).
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint32_t const nbegin = descriptor.neighbor_offset[center];
        std::uint32_t const nend = descriptor.neighbor_offset[center + 1];
        float const theta = ideal_angle_radians(descriptor.hybridization[center]);
        float const cos_theta = std::cos(theta);
        for (std::uint32_t first_entry = nbegin; first_entry < nend; ++first_entry) {
            std::uint32_t const first_neighbor = descriptor.neighbor_atom[first_entry];
            std::uint32_t const bond_ci = descriptor.neighbor_bond[first_entry];
            bool const ar_ci = (descriptor.bond_flags[bond_ci] & bond_flag_aromatic_k) != 0;
            float const d_ci = ideal_bond_length_angstrom(
                descriptor.element[center], descriptor.element[first_neighbor], descriptor.bond_order[bond_ci], ar_ci);
            for (std::uint32_t second_entry = first_entry + 1; second_entry < nend; ++second_entry) {
                std::uint32_t const second_neighbor = descriptor.neighbor_atom[second_entry];
                std::uint32_t const bond_cj = descriptor.neighbor_bond[second_entry];
                bool const ar_cj = (descriptor.bond_flags[bond_cj] & bond_flag_aromatic_k) != 0;
                float const d_cj = ideal_bond_length_angstrom(descriptor.element[center],
                                                              descriptor.element[second_neighbor],
                                                              descriptor.bond_order[bond_cj], ar_cj);
                float const d_ij = std::sqrt(
                    std::max(0.0f, d_ci * d_ci + d_cj * d_cj - 2.0f * d_ci * d_cj * cos_theta));
                // ± 3% tolerance — 1-3 distances tend to have slightly more slack.
                set_pair(first_neighbor, second_neighbor, d_ij * 0.97f, d_ij * 1.03f);
            }
        }
    }

    // Collect the E/Z-defined double bonds up front. Drug-like molecules have none or a
    // handful, so the 1-4 sweep below scans a tiny list rather than all bonds, and skips
    // the check entirely in the common zero case.
    std::uint32_t stereo_bonds[max_bonds_k];
    std::uint32_t num_stereo_bonds = 0;
    for (std::uint32_t bond = 0; bond < descriptor.num_bonds; ++bond)
        if ((descriptor.bond_flags[bond] & (bond_flag_stereo_e_k | bond_flag_stereo_z_k)) != 0 &&
            descriptor.bond_stereo_atom_i[bond] != no_stereo_atom_k)
            stereo_bonds[num_stereo_bonds++] = bond;

    auto find_bond_between = [&](std::uint32_t first_atom, std::uint32_t second_atom) -> std::int32_t {
        std::uint32_t const nbegin = descriptor.neighbor_offset[first_atom];
        std::uint32_t const nend = descriptor.neighbor_offset[first_atom + 1];
        for (std::uint32_t entry = nbegin; entry < nend; ++entry)
            if (descriptor.neighbor_atom[entry] == second_atom)
                return static_cast<std::int32_t>(descriptor.neighbor_bond[entry]);
        return -1;
    };

    // set14 — conservative envelope for 1-4 pairs. Takes the sum of two bond lengths
    // plus a torsion-dependent spread as the upper bound; VdW minimum sum as the lower.
    for (std::uint32_t row_atom = 0; row_atom < num_atoms; ++row_atom) {
        for (std::uint32_t column_atom = row_atom + 1; column_atom < num_atoms; ++column_atom) {
            std::uint8_t const hops = bond_hops[row_atom * num_atoms + column_atom];
            if (hops != 3) continue;

            // A 1-4 pair spanning a stereo-defined double bond has a determined
            // end-to-end distance rather than a free envelope. Leaving the generic
            // 4.5 Å upper in place lets smoothing and FIRE settle on either isomer.
            bool stereo_constrained = false;
            for (std::uint32_t stereo_bond_slot = 0; stereo_bond_slot < num_stereo_bonds && !stereo_constrained;
                 ++stereo_bond_slot) {
                std::uint32_t const bond = stereo_bonds[stereo_bond_slot];
                std::uint32_t end_near_i = descriptor.bond_atom_i[bond];
                std::uint32_t end_near_j = descriptor.bond_atom_j[bond];
                std::int32_t bond_i = find_bond_between(row_atom, end_near_i);
                std::int32_t bond_j = find_bond_between(column_atom, end_near_j);
                if (bond_i < 0 || bond_j < 0) {
                    std::swap(end_near_i, end_near_j);
                    bond_i = find_bond_between(row_atom, end_near_i);
                    bond_j = find_bond_between(column_atom, end_near_j);
                    if (bond_i < 0 || bond_j < 0) continue;
                }
                // The flag fixes the relationship of the two recorded reference atoms.
                // Any other substituent on an end sits opposite that end's reference, so
                // each mismatch flips the sense.
                std::uint32_t const reference_i = descriptor.bond_stereo_atom_i[bond];
                std::uint32_t const reference_j = descriptor.bond_stereo_atom_j[bond];
                bool reference_on_i_side = find_bond_between(reference_i, end_near_i) >= 0;
                std::uint32_t const own_reference = reference_on_i_side ? reference_i : reference_j;
                std::uint32_t const other_reference = reference_on_i_side ? reference_j : reference_i;
                bool cis = (descriptor.bond_flags[bond] & bond_flag_stereo_z_k) != 0;
                if (row_atom != own_reference) cis = !cis;
                if (column_atom != other_reference) cis = !cis;
                double_bond_geometry_t const geometry = cis ? double_bond_cis_k : double_bond_trans_k;

                float const target = stereo_one_four_distance(descriptor, row_atom, end_near_i, end_near_j, column_atom,
                                                              static_cast<std::uint32_t>(bond_i), bond,
                                                              static_cast<std::uint32_t>(bond_j), geometry);
                set_pair(row_atom, column_atom, target * 0.95f, target * 1.05f);
                stereo_constrained = true;
            }
            if (stereo_constrained) continue;

            // Approximate: upper ≈ 3 × average C-C bond ≈ 4.5 Å; lower ≈ VdW contact.
            float const lower = vdw_lower_bound_angstrom(descriptor.element[row_atom], descriptor.element[column_atom]);
            float const upper = 4.5f;
            // Only widen — never replace a tighter (row_atom, column_atom) bound from set13.
            // The loop bounds give `row_atom < column_atom`, so the pair already indexes the
            // upper triangle directly and needs no min/max normalisation.
            float const current_upper = bounds[row_atom * num_atoms + column_atom];
            float const current_lower = bounds[column_atom * num_atoms + row_atom];
            if (current_upper >= upper_initial_k) bounds[row_atom * num_atoms + column_atom] = upper;
            if (current_lower == 0.0f) bounds[column_atom * num_atoms + row_atom] = lower;
        }
    }

    // Long-range pairs (more than three hops apart): VdW lower bound, keep upper unconstrained.
    for (std::uint32_t row_atom = 0; row_atom < num_atoms; ++row_atom) {
        for (std::uint32_t column_atom = row_atom + 1; column_atom < num_atoms; ++column_atom) {
            std::uint8_t const hops = bond_hops[row_atom * num_atoms + column_atom];
            if (hops <= 3) continue;
            float const lower = vdw_lower_bound_angstrom(descriptor.element[row_atom], descriptor.element[column_atom]);
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
        constraint.constraint_type = 0; // 1-2 bond
        constraint.upper_bound_angstrom = bounds_matrix[row_min * num_atoms + row_max];
        constraint.lower_bound_angstrom = bounds_matrix[row_max * num_atoms + row_min];
        constraint.penalty_weight = 10.0f;
        constraints.push_back(constraint);
    }

    // 1-3 constraints: iterate each atom as center, enumerate neighbor pairs via CSR.
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint16_t const start = descriptor.neighbor_offset[center];
        std::uint16_t const end = descriptor.neighbor_offset[center + 1];
        for (std::uint16_t first_entry = start; first_entry < end; ++first_entry) {
            for (std::uint16_t second_entry = first_entry + 1; second_entry < end; ++second_entry) {
                std::uint32_t const first_neighbor = descriptor.neighbor_atom[first_entry];
                std::uint32_t const second_neighbor = descriptor.neighbor_atom[second_entry];
                std::uint32_t const row_min = std::min(first_neighbor, second_neighbor);
                std::uint32_t const row_max = std::max(first_neighbor, second_neighbor);

                distance_constraint_t constraint {};
                constraint.atom_index_i = static_cast<std::uint8_t>(first_neighbor);
                constraint.atom_index_j = static_cast<std::uint8_t>(second_neighbor);
                constraint.constraint_type = 1; // 1-3 angle-derived
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
 *  @brief Look @p key up in @p cache, falling back to @p compute on a miss and memoising the result.
 *
 *  Every MMFF term kind shares the same shape — key the parameters on atom types plus bond
 *  signatures, hit the thread-local cache, and only then pay for RDKit's `std::map` walk. What the
 *  terms do @b not share is what an unparametrized term means: a missing bond stretch rejects the
 *  whole molecule, while a missing angle, torsion, out-of-plane or stretch-bend skips just that
 *  term. Returning `std::optional` keeps that decision at the call site.
 *
 *  @param[inout] cache Thread-local cache for this term kind.
 *  @param[in] key Packed atom types and bond signatures identifying the parameter tuple.
 *  @param[in] compute Callable returning `std::optional<cached_type_>`; `std::nullopt` means RDKit
 *                     has no parameters for this tuple, and nothing is memoised.
 *  @return The cached or freshly computed parameters, or `std::nullopt` when @p compute failed.
 */
template <typename cached_type_, typename compute_type_>
inline std::optional<cached_type_> cache_or_compute_(mmff_cache<cached_type_> &cache, std::uint64_t key,
                                                     compute_type_ &&compute) {
    if (auto const *hit = cache.find(key)) return *hit;
    std::optional<cached_type_> const computed = compute();
    if (computed) cache.insert(key, *computed);
    return computed;
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
inline smallfield::api::conformer_status_t extract_mmff94_params(RDKit::ROMol const &molecule,
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

    // Precompute per-atom MMFF types and per-bond signatures for the global parameter cache in
    // `src/smallfield/mmff94/mmff94_param_cache.hpp`. These feed the cache-key packers,
    // short-circuiting RDKit's per-term `std::map` lookups.
    //
    // `getMMFFAtomType` returns 0 for atoms outside the MMFF94 parameter set — yttrium,
    // lanthanides, transition metals and the like. Detect that here on the host so the molecule
    // never reaches the GPU pipeline: without this gate, `etkdg_pipeline_warp_kernel` does an OOB
    // write through element-keyed parameter LUTs in `forcefield_tables.hpp` and poisons every
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
    std::uint8_t bond_signature[max_bonds_k];
    for (std::uint32_t bond = 0; bond < descriptor.num_bonds; ++bond) {
        bool const arom = (descriptor.bond_flags[bond] & bond_flag_aromatic_k) != 0;
        bool const in_ring = (descriptor.bond_flags[bond] & bond_flag_in_ring_k) != 0;
        bond_signature[bond] = pack_bond_signature_(descriptor.bond_order[bond], arom, in_ring);
    }
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

    // Build bond terms — iterate descriptor's bond list, with no RDKit traversal. Cache-first: key
    // by (atom_type_pair, bond_signature). Cache hit skips RDKit's `getMMFFBondStretchParams` and
    // its internal `std::map` lookup.
    for (std::uint32_t bond = 0; bond < descriptor.num_bonds; ++bond) {
        std::uint32_t const atom_i = descriptor.bond_atom_i[bond];
        std::uint32_t const atom_j = descriptor.bond_atom_j[bond];
        std::uint64_t const key = make_bond_stretch_key_(atom_type[atom_i], atom_type[atom_j], bond_signature[bond]);
        auto const cached_vals =
            cache_or_compute_(mmff_bond_cache(), key, [&]() -> std::optional<mmff_bond_cached_t> {
                unsigned int bond_type;
                ForceFields::MMFF::MMFFBond bond_params;
                if (!mmff_props.getMMFFBondStretchParams(mutable_molecule, atom_i, atom_j, bond_type, bond_params))
                    return std::nullopt;
                return mmff_bond_cached_t {.force_constant = static_cast<float>(bond_params.kb),
                                           .equilibrium_distance = static_cast<float>(bond_params.r0)};
            });
        // Dropping the term would understate the molecule's energy by a whole bond stretch, and the
        // device hop-count BFS derives adjacency from this list, so a missing entry also shifts the
        // 1-2 / 1-3 / 1-4 nonbonded exclusions. Reject the molecule instead and let the fallback
        // role handle it.
        if (!cached_vals) return smallfield::api::conformer_status_t::unsupported_element_k;
        result.mmff_bonds.push_back({.atom_i = static_cast<std::uint8_t>(atom_i),
                                     .atom_j = static_cast<std::uint8_t>(atom_j),
                                     .padding = {0, 0},
                                     .force_constant = cached_vals->force_constant,
                                     .equilibrium_distance = cached_vals->equilibrium_distance,
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
                std::uint64_t const key = make_angle_key_(atom_type[atom_i], atom_type[atom_j], atom_type[atom_k],
                                                          bond_signature[bond_index_ij], bond_signature[bond_index_jk]);
                auto const cached_vals =
                    cache_or_compute_(mmff_angle_cache(), key, [&]() -> std::optional<mmff_angle_cached_t> {
                        unsigned int angle_type;
                        ForceFields::MMFF::MMFFAngle angle_params;
                        if (!mmff_props.getMMFFAngleBendParams(mutable_molecule, atom_i, atom_j, atom_k, angle_type,
                                                               angle_params))
                            return std::nullopt;
                        return mmff_angle_cached_t {
                            .force_constant = static_cast<float>(angle_params.ka),
                            .equilibrium_angle_radians = static_cast<float>(angle_params.theta0 * M_PI / 180.0)};
                    });
                if (!cached_vals) continue;
                // `linh` depends only on the central atom's MMFF type, so it is read
                // outside the angle-parameter cache rather than widening that key.
                auto const *prop_center = (*RDKit::MMFF::DefaultParameters::getMMFFProp())(atom_type[atom_j]);
                std::uint8_t const is_linear = (prop_center && prop_center->linh) ? 1u : 0u;
                result.mmff_angles.push_back({.atom_i = static_cast<std::uint8_t>(atom_i),
                                              .atom_j = static_cast<std::uint8_t>(atom_j),
                                              .atom_k = static_cast<std::uint8_t>(atom_k),
                                              .is_linear = is_linear,
                                              .force_constant = cached_vals->force_constant,
                                              .equilibrium_angle = cached_vals->equilibrium_angle_radians,
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
        std::uint8_t const bsig_jk = bond_signature[bond];
        for (std::uint16_t atom_j_entry = jstart; atom_j_entry < jend; ++atom_j_entry) {
            std::uint32_t const atom_i = descriptor.neighbor_atom[atom_j_entry];
            if (atom_i == atom_k) continue;
            std::uint16_t const bond_index_ij = descriptor.neighbor_bond[atom_j_entry];
            std::uint8_t const bsig_ij = bond_signature[bond_index_ij];
            for (std::uint16_t atom_k_entry = kstart; atom_k_entry < kend; ++atom_k_entry) {
                std::uint32_t const atom_l = descriptor.neighbor_atom[atom_k_entry];
                if (atom_l == atom_j) continue;
                std::uint16_t const bond_index_kl = descriptor.neighbor_bond[atom_k_entry];
                std::uint8_t const bsig_kl = bond_signature[bond_index_kl];
                std::uint64_t const key = make_torsion_key_(atom_type[atom_i], atom_type[atom_j], atom_type[atom_k],
                                                            atom_type[atom_l], bsig_ij, bsig_jk, bsig_kl);
                auto const cached_vals =
                    cache_or_compute_(mmff_torsion_cache(), key, [&]() -> std::optional<mmff_torsion_cached_t> {
                        unsigned int torsion_type;
                        ForceFields::MMFF::MMFFTor torsion_params;
                        if (!mmff_props.getMMFFTorsionParams(mutable_molecule, atom_i, atom_j, atom_k, atom_l,
                                                             torsion_type, torsion_params))
                            return std::nullopt;
                        return mmff_torsion_cached_t {.torsion_v1 = static_cast<float>(torsion_params.V1),
                                                      .torsion_v2 = static_cast<float>(torsion_params.V2),
                                                      .torsion_v3 = static_cast<float>(torsion_params.V3)};
                    });
                if (!cached_vals) continue;
                result.mmff_torsions.push_back({.atom_i = static_cast<std::uint8_t>(atom_i),
                                                .atom_j = static_cast<std::uint8_t>(atom_j),
                                                .atom_k = static_cast<std::uint8_t>(atom_k),
                                                .atom_l = static_cast<std::uint8_t>(atom_l),
                                                .torsion_v1 = cached_vals->torsion_v1,
                                                .torsion_v2 = cached_vals->torsion_v2,
                                                .torsion_v3 = cached_vals->torsion_v3});
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
            std::uint64_t const key = make_oop_key_(atom_type[center], atom_type[neighbor_cache[apex_slot]],
                                                    atom_type[plane[0]], atom_type[plane[1]]);
            auto const cached_vals =
                cache_or_compute_(mmff_oop_cache(), key, [&]() -> std::optional<mmff_oop_cached_t> {
                    ForceFields::MMFF::MMFFOop oop_params;
                    if (!mmff_props.getMMFFOopBendParams(mutable_molecule, neighbor_cache[apex_slot], center, plane[0],
                                                         plane[1], oop_params))
                        return std::nullopt;
                    return mmff_oop_cached_t {.force_constant = static_cast<float>(oop_params.koop)};
                });
            if (!cached_vals) continue;
            result.mmff_oops.push_back({.atom_i = neighbor_cache[apex_slot],
                                        .atom_j = static_cast<std::uint8_t>(center),
                                        .atom_k = plane[0],
                                        .atom_l = plane[1],
                                        .force_constant = cached_vals->force_constant,
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
                std::uint64_t const key = make_stretchbend_key_(atom_type[atom_i], atom_type[atom_j], atom_type[atom_k],
                                                                bond_signature[bond_index_ij],
                                                                bond_signature[bond_index_jk]);
                auto const cached_vals =
                    cache_or_compute_(mmff_stretchbend_cache(), key, [&]() -> std::optional<mmff_stretchbend_cached_t> {
                        unsigned int stretchbend_type;
                        ForceFields::MMFF::MMFFStbn stretchbend_params;
                        ForceFields::MMFF::MMFFBond bond_params[2];
                        ForceFields::MMFF::MMFFAngle angle_params;
                        if (!mmff_props.getMMFFStretchBendParams(mutable_molecule, atom_i, atom_j, atom_k,
                                                                 stretchbend_type, stretchbend_params, bond_params,
                                                                 angle_params))
                            return std::nullopt;
                        return mmff_stretchbend_cached_t {
                            .kba_ijk = static_cast<float>(stretchbend_params.kbaIJK),
                            .kba_kji = static_cast<float>(stretchbend_params.kbaKJI),
                            .equilibrium_angle_radians = static_cast<float>(angle_params.theta0 * M_PI / 180.0),
                            .r0_ij = static_cast<float>(bond_params[0].r0),
                            .r0_kj = static_cast<float>(bond_params[1].r0)};
                    });
                if (!cached_vals) continue;
                result.mmff_stretchbends.push_back({.atom_i = static_cast<std::uint8_t>(atom_i),
                                                    .atom_j = static_cast<std::uint8_t>(atom_j),
                                                    .atom_k = static_cast<std::uint8_t>(atom_k),
                                                    .padding = 0,
                                                    .kba_ijk = cached_vals->kba_ijk,
                                                    .kba_kji = cached_vals->kba_kji,
                                                    .equilibrium_angle = cached_vals->equilibrium_angle_radians,
                                                    .equilibrium_distance_ij = cached_vals->r0_ij,
                                                    .equilibrium_distance_kj = cached_vals->r0_kj,
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
 *  `prepare_molecule_for_device_into` instead. The caller-owned `result` is meant
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
        if (!build_descriptor(*molecule, descriptor)) {
            result.num_atoms = 0;
            return conformer_status_t::invalid_smiles_k;
        }
        build_bounds_matrix(descriptor, result.bounds_matrix);
        extract_chiral_centers(*molecule, descriptor, result.chiral_centers);
        extract_etk_torsions_native(descriptor, result.etk_torsions);
        extract_distance_constraints(descriptor, result.bounds_matrix, result.distance_constraints);
        build_etk_interaction_lists(result);
        mmff_status = extract_mmff94_params(*molecule, descriptor, result);
    }
    catch (...) {
        result.num_atoms = 0;
        return conformer_status_t::invalid_smiles_k;
    }
    if (mmff_status != conformer_status_t::success_k) {
        // MMFF could not parametrise this molecule — say, an atomic number outside the MMFF94 set.
        // The mmff_* result vectors are empty after the early return inside
        // `extract_mmff94_params`, but ETK fields above are populated; clear num_atoms so
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
    result.metadata.bounds_min = 0.0f;
    result.metadata.bounds_inv_scale = 1.0f;
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
inline smallfield::api::conformer_status_t read_rdkit_mmff_typing(RDKit::ROMol const &molecule, std::uint32_t num_atoms,
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
 */
inline smallfield::api::conformer_status_t prepare_molecule_for_device_into(std::string_view smiles,
                                                                            smallfield::salt_policy_t salt_policy,
                                                                            etkdg_prepared_molecule_t &result) {
    using smallfield::mmff94::native_compute_mmff_atom_types;

    result.num_atoms = 0;
    result.bounds_matrix.clear();
    result.chiral_centers.clear();
    result.distance_constraints.clear();
    result.patch_atom_types.clear();
    result.patch_atom_charges.clear();
    result.capacities = device_capacities_t {};

    smallfield::parse_options_t parse_options;
    parse_options.max_atoms = max_atoms_per_molecule_k; // single-warp pipeline cap
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
        if (!build_descriptor(*molecule, result.descriptor)) return conformer_status_t::invalid_smiles_k;
        build_bounds_matrix(result.descriptor, result.bounds_matrix);
        extract_chiral_centers(*molecule, result.descriptor, result.chiral_centers);
        extract_distance_constraints(result.descriptor, result.bounds_matrix, result.distance_constraints);

        // The device typer runs the same code on the same descriptor, so deciding here which
        // molecules it will reject needs no readback: the answer is already computable.
        std::array<std::uint8_t, max_atoms_per_molecule_k> atom_types_scratch {};
        std::span<std::uint8_t> atom_types_span {atom_types_scratch.data(), num_atoms};
        if (!native_compute_mmff_atom_types(result.descriptor, atom_types_span))
            typing_status = read_rdkit_mmff_typing(*molecule, num_atoms, result.patch_atom_types,
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
    result.metadata.bounds_inv_scale = 1.0f;
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
 *  @brief Per-molecule element counts of every region a batch carries.
 *
 *  Packed into one cache-line-aligned struct so the prefix scan over per-molecule sizes is one
 *  pass over a single struct rather than a dozen parallel accumulator chains. The device-filled
 *  regions contribute their capacity rather than their eventual content, since that is what the
 *  host reserves for them. The `valid` field doubles as the dense-output compaction counter.
 */
struct alignas(64) per_mol_sizes_t {
    std::uint32_t bounds;
    std::uint32_t chirals;
    std::uint32_t distance_constraints;
    /** Per-atom regions: types, charges, VdW params, and both CSR offset arrays. */
    std::uint32_t atoms;
    std::uint32_t bond_distance;
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
    std::uint32_t padding;
};
static_assert(sizeof(per_mol_sizes_t) == 64, "per_mol_sizes_t must be one cache line");

/** Accumulate one molecule's capacities into a running total. */
inline void accumulate_sizes(per_mol_sizes_t &totals, etkdg_molecule_metadata_t const &metadata,
                             device_capacities_t const &capacities) {
    std::uint32_t const num_atoms = metadata.num_atoms;
    totals.bounds += num_atoms * num_atoms;
    totals.chirals += metadata.num_chiral_centers;
    totals.distance_constraints += metadata.num_distance_constraints;
    totals.atoms += num_atoms;
    totals.bond_distance += num_atoms * num_atoms;
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

/** Per-thread pool-wide base offsets computed by `scan_per_region_offsets_`. */
struct batch_prep_thread_bases_t {
    std::size_t bounds, chirals, distance_constraints;
    std::size_t atoms, bond_distance, etk_interactions;
    std::size_t mmff_bonds, mmff_angles, mmff_torsions, mmff_oops, mmff_stretchbends, mmff_interactions;
    std::size_t patch_atoms, patch_records;
    std::size_t dense_slot;
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
 *  first-time access. `std::call_once` guarantees the warm-up runs exactly once per process
 *  regardless of how many call sites reach here.
 */
inline void warm_up_rdkit_globals_once_() {
    static std::once_flag flag;
    std::call_once(flag, [] {
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
    });
}

/**
 *  @brief First stage of the preprocess-step: parallel per-molecule RDKit parse and byte-append into
 *  each OpenMP thread's contribution buffer.
 *
 *  After return, each thread's contribution carries its share of the input molecules with
 *  thread-local metadata offsets.
 */
static inline void prepare_per_thread_contribs_(std::span<std::string_view const> smiles_list,
                                                smallfield::salt_policy_t salt_policy,
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
            auto const status = prepare_molecule_for_device_into(smiles_list[molecule], salt_policy,
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
 *  @brief Second stage of the preprocess-step: serial prefix scan over per-thread contribution sizes to
 *  compute each thread's pool-wide base offset per region, plus the totals that drive the layout.
 *
 *  `bases` is resized to match `contribs.size()`.
 */
static inline void scan_per_region_offsets_(std::vector<batch_prep_thread_contribution_t> const &contribs,
                                            std::vector<batch_prep_thread_bases_t> &bases, per_mol_sizes_t &totals,
                                            std::uint32_t &n_valid, std::uint32_t &max_atoms_reduction,
                                            std::uint32_t &max_bonds_reduction) {
    int const n_threads_max = static_cast<int>(contribs.size());
    bases.assign(n_threads_max, batch_prep_thread_bases_t {});
    totals = per_mol_sizes_t {};
    n_valid = 0;
    max_atoms_reduction = 0;
    max_bonds_reduction = 0;
    for (int thread_rank = 0; thread_rank < n_threads_max; ++thread_rank) {
        auto const &contribution = contribs[thread_rank];
        auto &base = bases[thread_rank];
        base.bounds = totals.bounds;
        totals.bounds += contribution.bounds_bytes.size() / sizeof(float);
        base.chirals = totals.chirals;
        totals.chirals += contribution.chirals_bytes.size() / sizeof(chiral_center_descriptor_t);
        base.distance_constraints = totals.distance_constraints;
        totals.distance_constraints += contribution.distance_constraints_bytes.size() / sizeof(distance_constraint_t);
        base.atoms = totals.atoms;
        totals.atoms += contribution.atom_count;
        base.bond_distance = totals.bond_distance;
        totals.bond_distance += contribution.bond_distance_count;
        base.etk_interactions = totals.etk_interactions;
        totals.etk_interactions += contribution.etk_interaction_count;
        base.mmff_bonds = totals.mmff_bonds;
        totals.mmff_bonds += contribution.mmff_bond_count;
        base.mmff_angles = totals.mmff_angles;
        totals.mmff_angles += contribution.mmff_angle_count;
        base.mmff_torsions = totals.mmff_torsions;
        totals.mmff_torsions += contribution.mmff_torsion_count;
        base.mmff_oops = totals.mmff_oops;
        totals.mmff_oops += contribution.mmff_oop_count;
        base.mmff_stretchbends = totals.mmff_stretchbends;
        totals.mmff_stretchbends += contribution.mmff_stretchbend_count;
        base.mmff_interactions = totals.mmff_interactions;
        totals.mmff_interactions += contribution.mmff_interaction_count;
        base.patch_atoms = totals.patch_atoms;
        totals.patch_atoms += contribution.patch_atom_types_bytes.size();
        base.patch_records = totals.patch_records;
        totals.patch_records += static_cast<std::uint32_t>(contribution.typing_patches.size());
        base.dense_slot = n_valid;
        n_valid += static_cast<std::uint32_t>(contribution.mols.size());
        if (contribution.max_atoms_seen > max_atoms_reduction) max_atoms_reduction = contribution.max_atoms_seen;
        if (contribution.max_bonds_seen > max_bonds_reduction) max_bonds_reduction = contribution.max_bonds_seen;
    }
    totals.valid = n_valid;
}

/**
 *  @brief Third stage of the preprocess-step: parallel per-thread memcpy of every byte accumulator
 *  into the batch's pinned pool, plus metadata rebase from thread-local to pool-wide offsets.
 *
 *  @pre `batch.host.ensure_capacity(batch.layout.bytes_uploaded)` has succeeded;
 *       `batch.num_molecules`, `batch.observed_max_atoms`, and `batch.layout` are populated.
 */
static inline void pack_into_batch_pool_(std::vector<batch_prep_thread_contribution_t> const &contribs,
                                         std::vector<batch_prep_thread_bases_t> const &bases,
                                         etkdg_molecule_batch_t &batch) {
    int const n_threads_max = static_cast<int>(contribs.size());
    auto bounds_span = batch.bounds_matrices();
    auto chirals_span = batch.chiral_centers();
    auto distance_constraints_span = batch.distance_constraints();
    auto descriptors_span = batch.descriptors();
    auto atom_bases_span = batch.atom_bases();
    auto mmff_bases_span = batch.mmff_bases();
    auto typing_patches_span = batch.typing_patches();
    auto patch_types_span = batch.patch_atom_types();
    auto patch_charges_span = batch.patch_atom_charges();
    auto metadata_span = batch.molecule_metadata();
    auto input_indices_span = batch.input_indices();
    std::uint32_t const torsion_stride = batch.layout.torsion_stride;

#pragma omp parallel for schedule(static)
    for (int thread_rank = 0; thread_rank < n_threads_max; ++thread_rank) {
        auto const &contribution = contribs[thread_rank];
        auto const &base = bases[thread_rank];
        if (!contribution.bounds_bytes.empty())
            std::memcpy(bounds_span.data() + base.bounds, contribution.bounds_bytes.data(),
                        contribution.bounds_bytes.size());
        if (!contribution.chirals_bytes.empty())
            std::memcpy(chirals_span.data() + base.chirals, contribution.chirals_bytes.data(),
                        contribution.chirals_bytes.size());
        if (!contribution.distance_constraints_bytes.empty())
            std::memcpy(distance_constraints_span.data() + base.distance_constraints,
                        contribution.distance_constraints_bytes.data(), contribution.distance_constraints_bytes.size());
        if (!contribution.descriptors_bytes.empty())
            std::memcpy(descriptors_span.data() + base.dense_slot, contribution.descriptors_bytes.data(),
                        contribution.descriptors_bytes.size());
        if (!contribution.patch_atom_types_bytes.empty())
            std::memcpy(patch_types_span.data() + base.patch_atoms, contribution.patch_atom_types_bytes.data(),
                        contribution.patch_atom_types_bytes.size());
        if (!contribution.patch_atom_charges_bytes.empty())
            std::memcpy(patch_charges_span.data() + base.patch_atoms, contribution.patch_atom_charges_bytes.data(),
                        contribution.patch_atom_charges_bytes.size());

        for (std::size_t patch_index = 0; patch_index < contribution.typing_patches.size(); ++patch_index) {
            mmff94_typing_patch_t patch = contribution.typing_patches[patch_index];
            patch.molecule_index += static_cast<std::uint32_t>(base.dense_slot);
            patch.atom_base += static_cast<std::uint32_t>(base.atoms);
            patch.patch_base += static_cast<std::uint32_t>(base.patch_atoms);
            typing_patches_span[base.patch_records + patch_index] = patch;
        }

        // Per-molecule metadata: rebase every region offset by the thread's pool base.
        for (std::size_t molecule = 0; molecule < contribution.mols.size(); ++molecule) {
            auto const &record = contribution.mols[molecule];
            etkdg_molecule_metadata_t meta = record.metadata;
            std::size_t const slot = base.dense_slot + molecule;
            meta.bounds_offset += static_cast<std::uint32_t>(base.bounds);
            meta.chiral_offset += static_cast<std::uint32_t>(base.chirals);
            meta.constraint_offset += static_cast<std::uint32_t>(base.distance_constraints);
            meta.mmff_atom_vdw_offset += static_cast<std::uint32_t>(base.atoms);
            meta.mmff_atom_charge_offset += static_cast<std::uint32_t>(base.atoms);
            meta.mmff_atom_csr_offset += static_cast<std::uint32_t>(base.atoms);
            meta.etk_atom_csr_offset += static_cast<std::uint32_t>(base.atoms);
            meta.bond_distance_offset += static_cast<std::uint32_t>(base.bond_distance);
            meta.etk_interaction_offset += static_cast<std::uint32_t>(base.etk_interactions);
            meta.mmff_bond_offset += static_cast<std::uint32_t>(base.mmff_bonds);
            meta.mmff_angle_offset += static_cast<std::uint32_t>(base.mmff_angles);
            meta.mmff_torsion_offset += static_cast<std::uint32_t>(base.mmff_torsions);
            meta.mmff_oop_offset += static_cast<std::uint32_t>(base.mmff_oops);
            meta.mmff_stretchbend_offset += static_cast<std::uint32_t>(base.mmff_stretchbends);
            // The matcher writes at a fixed stride, so a molecule's torsions start at its slot.
            meta.torsion_offset = static_cast<std::uint32_t>(slot) * torsion_stride;
            metadata_span[slot] = meta;
            input_indices_span[slot] = record.original_index;
            atom_bases_span[slot] = meta.mmff_atom_vdw_offset;
            mmff_bases_span[slot] = {meta.mmff_bond_offset,
                                     meta.mmff_angle_offset,
                                     meta.mmff_torsion_offset,
                                     meta.mmff_oop_offset,
                                     meta.mmff_stretchbend_offset,
                                     record.mmff_interaction_base + static_cast<std::uint32_t>(base.mmff_interactions)};
        }
    }
}

/**
 *  @brief In-place `preprocess-step` driver: parses every SMILES, extracts the host half of the
 *  ETKDG + MMFF input, and packs it into @p batch's pinned host pool.
 *
 *  Recycling the pinned buffer across `embedding-waves` is the lever that kills pinned-memory
 *  allocator contention at 8-GPU scale — `cudaHostAlloc`/`cudaFreeHost` serialize on a driver
 *  lock. The `static` contribs vector persists capacities across calls so the allocator's fast
 *  path handles every call after the first.
 */
inline void prepare_molecule_batch_into(std::span<std::string_view const> smiles_list, etkdg_molecule_batch_t &batch,
                                        smallfield::salt_policy_t salt_policy = smallfield::keep_largest_fragment_k) {
    batch.num_molecules = 0;
    batch.observed_max_atoms = 0;
    batch.layout = wave_slice_layout_t {};

    // Persistent per-call scratch. Safe as long as this function is not called concurrently from
    // multiple outer threads. The CLI's 4-role pipeline guarantees this — `preprocess-step` runs
    // only on the gpu-runner-role thread.
    static std::vector<batch_prep_thread_contribution_t> contribs;
    prepare_per_thread_contribs_(smiles_list, salt_policy, contribs);

    std::vector<batch_prep_thread_bases_t> bases;
    per_mol_sizes_t totals {};
    std::uint32_t n_valid = 0;
    std::uint32_t max_atoms_reduction = 0;
    std::uint32_t max_bonds_reduction = 0;
    scan_per_region_offsets_(contribs, bases, totals, n_valid, max_atoms_reduction, max_bonds_reduction);

    batch.num_molecules = n_valid;
    batch.observed_max_atoms = max_atoms_reduction;
    batch.layout = compute_batch_layout(totals, n_valid, std::max<std::uint32_t>(max_bonds_reduction, 1u));
    if (batch.layout.bytes_uploaded > 0 && !batch.host.ensure_capacity(batch.layout.bytes_uploaded)) {
        batch.num_molecules = 0;
        batch.layout = wave_slice_layout_t {};
        return;
    }

    // Merge every thread's rejected-input indices into the batch's flat list. Each entry carries
    // the slice-local index plus the `conformer_status_t` reason code.
    batch.rejected_input_indices.clear();
    for (auto const &contribution : contribs)
        batch.rejected_input_indices.insert(batch.rejected_input_indices.end(),
                                            contribution.rejected_input_indices.begin(),
                                            contribution.rejected_input_indices.end());

    pack_into_batch_pool_(contribs, bases, batch);
}

/**
 *  @brief Convenience wrapper returning a fresh batch, allocating a new pinned buffer every call.
 *
 *  Serves call sites outside the pipelined dispatcher, such as tests and one-shot invocations. The
 *  hot path should prefer `prepare_molecule_batch_into`.
 */
inline etkdg_molecule_batch_t prepare_molecule_batch(
    std::span<std::string_view const> smiles_list,
    smallfield::salt_policy_t salt_policy = smallfield::keep_largest_fragment_k) {
    etkdg_molecule_batch_t batch {};
    prepare_molecule_batch_into(smiles_list, batch, salt_policy);
    return batch;
}

/**
 *  @brief Grow-only scratch for `distribute_batch_to_per_device_pools_`.
 *
 *  Holds the sort permutation, the CSR-packed per-device assignment list, the per-device CSR
 *  offset array, the shared-batch molecule-to-patch lookup, and the per-device per-molecule
 *  capacity staging the pack loop consumes. One instance per caller; reused across every
 *  `embedding-wave` so the distribute hot path never allocates.
 */
struct distribute_scratch_t {
    /** Length: total_mol_count. */
    std::vector<std::uint32_t> sort_permutation;
    /** Length: total_mol_count, CSR values. */
    std::vector<std::uint32_t> per_device_flat;
    /** Index into the shared batch's typing-patch array for each molecule, or
     *  `no_typing_patch_k` when the molecule was typed natively. */
    std::vector<std::uint32_t> patch_of_molecule;
    static constexpr std::uint32_t no_typing_patch_k = 0xFFFFFFFFu;
    // Room for twice the `smallfield::api::max_devices_k = 16` the API admits; `smallfield.cu`
    // compile-time-asserts the two against each other.
    static constexpr std::uint32_t max_devices_k = 32u;
    /** Length: device_count + 1. */
    std::array<std::uint32_t, max_devices_k + 1u> per_device_offsets {};
    std::array<std::vector<device_capacities_t>, max_devices_k> capacities_per_mol {};
};

/**
 *  @brief Distribute an already-packed shared batch into `device_count` per-device pinned pools
 *  with a complexity-descending stripe partition.
 *
 *  Each device ends up with a disjoint subset of the shared batch's molecules; the subset is
 *  chosen so that per-device work distributions are near-identical.
 *
 *  Each mol's host-produced bytes are copied from the shared pool into the assigned device's
 *  pinned pool at per-device-relative offsets, and its metadata, atom base, MMFF bases and typing
 *  patch are rebased to the per-device layout.
 *
 *  Caller ensures `per_device_out[d].host` is pre-sized to hold this device's slice.
 *
 *  @param[in] shared Packed shared batch produced by `prepare_molecule_batch_into`.
 *  @param[in] device_count Devices to partition across.
 *  @param[out] per_device_out One pinned pool per device, filled in place.
 *  @param[inout] scratch Grow-only caller-owned scratch; see `distribute_scratch_t`.
 */
inline void distribute_batch_to_per_device_pools_(etkdg_molecule_batch_t const &shared, std::uint32_t device_count,
                                                  std::span<etkdg_molecule_batch_t> per_device_out,
                                                  distribute_scratch_t &scratch) {

    std::uint32_t const total_mol_count = shared.num_molecules;
    auto const shared_view = shared.host_view();
    auto const shared_metadata = shared.molecule_metadata();
    auto const shared_input_indices = shared.input_indices();

    for (std::uint32_t device_index = 0; device_index < device_count; ++device_index) {
        per_device_out[device_index].num_molecules = 0;
        per_device_out[device_index].observed_max_atoms = 0;
        per_device_out[device_index].layout = wave_slice_layout_t {};
    }
    if (total_mol_count == 0) return;

    // Sort permutation in place on grow-only scratch.
    if (scratch.sort_permutation.size() < total_mol_count) scratch.sort_permutation.resize(total_mol_count);
    if (scratch.per_device_flat.size() < total_mol_count) scratch.per_device_flat.resize(total_mol_count);
    std::iota(scratch.sort_permutation.begin(), scratch.sort_permutation.begin() + total_mol_count, 0u);
    std::sort(scratch.sort_permutation.begin(), scratch.sort_permutation.begin() + total_mol_count,
              [&](std::uint32_t first_molecule, std::uint32_t second_molecule) {
                  return molecule_complexity_score_(shared_metadata[first_molecule]) >
                         molecule_complexity_score_(shared_metadata[second_molecule]);
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

    // Invert the shared batch's patch list so a molecule can find its own patch in constant time.
    auto const shared_patches = shared_view.typing_patches();
    scratch.patch_of_molecule.assign(total_mol_count, distribute_scratch_t::no_typing_patch_k);
    for (std::uint32_t patch_index = 0; patch_index < shared.layout.num_typing_patches; ++patch_index)
        scratch.patch_of_molecule[shared_patches[patch_index].molecule_index] = patch_index;

#pragma omp parallel for num_threads(static_cast<int>(device_count)) schedule(static, 1)
    for (std::uint32_t device_index = 0; device_index < device_count; ++device_index) {
        std::span<std::uint32_t const> const mol_list {
            scratch.per_device_flat.data() + scratch.per_device_offsets[device_index],
            scratch.per_device_offsets[device_index + 1u] - scratch.per_device_offsets[device_index],
        };
        etkdg_molecule_batch_t &out = per_device_out[device_index];
        std::uint32_t const device_mol_count = static_cast<std::uint32_t>(mol_list.size());

        auto &capacities_per_mol = scratch.capacities_per_mol[device_index];
        if (capacities_per_mol.size() < device_mol_count) capacities_per_mol.resize(device_mol_count);

        auto const descriptors_source = shared_view.descriptors();
        per_mol_sizes_t totals {};
        std::uint32_t max_atoms_seen = 0;
        std::uint32_t max_bonds_seen = 0;
        for (std::uint32_t position = 0; position < device_mol_count; ++position) {
            std::uint32_t const shared_mol_index = mol_list[position];
            auto const &meta = shared_metadata[shared_mol_index];
            auto const &descriptor = descriptors_source[shared_mol_index];
            if (meta.num_atoms > max_atoms_seen) max_atoms_seen = meta.num_atoms;
            if (descriptor.num_bonds > max_bonds_seen) max_bonds_seen = descriptor.num_bonds;
            capacities_per_mol[position] = compute_device_capacities(descriptor, meta.num_distance_constraints);
            accumulate_sizes(totals, meta, capacities_per_mol[position]);
            if (scratch.patch_of_molecule[shared_mol_index] != distribute_scratch_t::no_typing_patch_k) {
                totals.patch_records += 1;
                totals.patch_atoms += meta.num_atoms;
            }
        }

        wave_slice_layout_t device_layout = compute_batch_layout(totals, device_mol_count,
                                                                 std::max<std::uint32_t>(max_bonds_seen, 1u));
        out.num_molecules = device_mol_count;
        out.observed_max_atoms = max_atoms_seen;
        out.layout = device_layout;
        if (device_layout.bytes_uploaded > 0 && !out.host.ensure_capacity(device_layout.bytes_uploaded)) {
            out.num_molecules = 0;
            out.layout = wave_slice_layout_t {};
            continue;
        }
        if (device_mol_count == 0) continue;

        auto bounds_destination = out.bounds_matrices();
        auto chirals_destination = out.chiral_centers();
        auto constraints_destination = out.distance_constraints();
        auto descriptors_destination = out.descriptors();
        auto atom_bases_destination = out.atom_bases();
        auto mmff_bases_destination = out.mmff_bases();
        auto patches_destination = out.typing_patches();
        auto patch_types_destination = out.patch_atom_types();
        auto patch_charges_destination = out.patch_atom_charges();
        auto metadata_destination = out.molecule_metadata();
        auto input_indices_destination = out.input_indices();

        auto const bounds_source = shared_view.bounds_matrices();
        auto const chirals_source = shared_view.chiral_centers();
        auto const constraints_source = shared_view.distance_constraints();
        auto const patch_types_source = shared_view.patch_atom_types();
        auto const patch_charges_source = shared_view.patch_atom_charges();

        per_mol_sizes_t running {};
        for (std::uint32_t position = 0; position < device_mol_count; ++position) {
            std::uint32_t const shared_mol_index = mol_list[position];
            auto const &source_meta = shared_metadata[shared_mol_index];
            auto const &capacities = capacities_per_mol[position];
            std::uint32_t const atom_count = source_meta.num_atoms;

            std::memcpy(bounds_destination.data() + running.bounds, bounds_source.data() + source_meta.bounds_offset,
                        static_cast<std::size_t>(atom_count) * atom_count * sizeof(float));
            std::memcpy(chirals_destination.data() + running.chirals, chirals_source.data() + source_meta.chiral_offset,
                        source_meta.num_chiral_centers * sizeof(chiral_center_descriptor_t));
            std::memcpy(constraints_destination.data() + running.distance_constraints,
                        constraints_source.data() + source_meta.constraint_offset,
                        source_meta.num_distance_constraints * sizeof(distance_constraint_t));
            descriptors_destination[position] = descriptors_source[shared_mol_index];

            std::uint32_t const patch_index = scratch.patch_of_molecule[shared_mol_index];
            if (patch_index != distribute_scratch_t::no_typing_patch_k) {
                auto const &source_patch = shared_patches[patch_index];
                std::memcpy(patch_types_destination.data() + running.patch_atoms,
                            patch_types_source.data() + source_patch.patch_base, atom_count);
                std::memcpy(patch_charges_destination.data() + running.patch_atoms,
                            patch_charges_source.data() + source_patch.patch_base, atom_count * sizeof(float));
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
            destination_meta.bounds_offset = running.bounds;
            destination_meta.chiral_offset = running.chirals;
            destination_meta.constraint_offset = running.distance_constraints;
            destination_meta.torsion_offset = position * device_layout.torsion_stride;
            destination_meta.mmff_atom_vdw_offset = running.atoms;
            destination_meta.mmff_atom_charge_offset = running.atoms;
            destination_meta.mmff_atom_csr_offset = running.atoms;
            destination_meta.etk_atom_csr_offset = running.atoms;
            destination_meta.bond_distance_offset = running.bond_distance;
            destination_meta.etk_interaction_offset = running.etk_interactions;
            destination_meta.mmff_bond_offset = running.mmff_bonds;
            destination_meta.mmff_angle_offset = running.mmff_angles;
            destination_meta.mmff_torsion_offset = running.mmff_torsions;
            destination_meta.mmff_oop_offset = running.mmff_oops;
            destination_meta.mmff_stretchbend_offset = running.mmff_stretchbends;
            metadata_destination[position] = destination_meta;
            input_indices_destination[position] = shared_input_indices[shared_mol_index];
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
 *  Internally reuses the three-stage preprocess against a reusable shared batch, then distributes
 *  to the per-device pools. Two file-scope `static` scratch instances hold the pinned shared
 *  batch and the distribution CSR buffers so neither allocation fires on the hot path.
 *
 *  @param[in] smiles_list Slice of input SMILES to prepare.
 *  @param[in] device_count Devices to partition across.
 *  @param[out] per_device_out One pinned pool per device.
 *  @param[out] rejected_input_indices_out Optional sink for the slice-local rejection records.
 */
inline void prepare_per_device_batches_into(
    std::span<std::string_view const> smiles_list, std::uint32_t device_count,
    std::span<etkdg_molecule_batch_t> per_device_out,
    std::vector<smallfield::api::rejected_mol_t> *rejected_input_indices_out = nullptr) {
    static etkdg_molecule_batch_t shared_scratch;
    static distribute_scratch_t distribute_scratch;
    prepare_molecule_batch_into(smiles_list, shared_scratch);
    distribute_batch_to_per_device_pools_(shared_scratch, device_count, per_device_out, distribute_scratch);
    if (rejected_input_indices_out != nullptr) {
        rejected_input_indices_out->assign(shared_scratch.rejected_input_indices.begin(),
                                           shared_scratch.rejected_input_indices.end());
    }
}

} // namespace smallfield::preprocess
