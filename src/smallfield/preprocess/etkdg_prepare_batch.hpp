/**
 *  @file etkdg_prepare_batch.hpp
 *  @brief CPU-side ETKDG preprocessing using RDKit C++ API (stages 0-1).
 *
 *  Converts molecular graphs into GPU-ready data structures:
 *   - Distance bounds matrix (N x N, smoothed via triangle inequality)
 *   - Chiral center descriptors (index-ordered neighbors, target volume sign)
 *   - ETK torsion descriptors (CSD Fourier coefficients)
 *   - 1-2 / 1-3 distance constraints
 *
 *  All functions operate on RDKit ROMol objects and produce flat arrays
 *  suitable for upload to GPU global memory.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

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
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Descriptors/MolDescriptors.h>

#include "smallfield/api/batch_pool.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/parameters/forcefield_tables.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/preprocess/csd_torsion_dispatcher.hpp"
#include "smallfield/preprocess/molecule_complexity.hpp"
#include "smallfield/mmff94/mmff94_param_cache.hpp"
#include "smallfield/types/mmff94_bounds.hpp"

using smallfield::types::max_mmff_interactions_per_mol_k;

namespace smallfield::preprocess {
using smallfield::api::batch_layout_t;
using smallfield::api::batch_pool_t;
using smallfield::api::place_region_;
using smallfield::mmff94::make_angle_key_;
using smallfield::mmff94::make_bond_stretch_key_;
using smallfield::mmff94::make_oop_key_;
using smallfield::mmff94::make_stretchbend_key_;
using smallfield::mmff94::make_torsion_key_;
using smallfield::mmff94::mmff_angle_cache;
using smallfield::mmff94::mmff_angle_cached_t;
using smallfield::mmff94::mmff_bond_cache;
using smallfield::mmff94::mmff_bond_cached_t;
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
using smallfield::types::max_atoms_k;
using smallfield::types::max_bonds_k;
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
using smallfield::types::mmff94_stretch_bend_k;
using smallfield::types::mmff94_stretchbend_t;
using smallfield::types::mmff94_torsion_k;
using smallfield::types::mmff94_torsion_t;
using smallfield::types::molecule_descriptor_t;
using smallfield::types::no_stereo_atom_k;

/**
 *  @brief Complete preprocessed molecule ready for GPU pipeline upload.
 */
struct etkdg_prepared_molecule_t {
    std::uint32_t num_atoms;
    std::vector<float> bounds_matrix; ///< N x N flat, unsmoothed
    std::vector<chiral_center_descriptor_t> chiral_centers;
    std::vector<etk_torsion_descriptor_t> etk_torsions;
    std::vector<distance_constraint_t> distance_constraints; ///< 1-2 bond, 1-3 angle
    // MMFF94 per-atom data (kernel enumerates nonbonded pairs on-the-fly from bond_distance).
    std::vector<mmff94_atom_vdw_params_t> mmff_atom_vdw; ///< one per atom
    std::vector<float> mmff_atom_charge;                 ///< one per atom
    std::vector<mmff94_bond_t> mmff_bonds;
    std::vector<mmff94_angle_t> mmff_angles;
    std::vector<mmff94_torsion_t> mmff_torsions;
    std::vector<mmff94_oop_t> mmff_oops;
    std::vector<mmff94_stretchbend_t> mmff_stretchbends;
    // Per-atom MMFF94 interaction lists (no nonbonded entries).
    std::vector<mmff94_atom_interaction_t> mmff_interactions;
    std::vector<mmff94_atom_offsets_t> mmff_atom_offsets;
    // Per-atom ETK interaction lists (torsions + distance constraints).
    std::vector<etk_atom_interaction_t> etk_interactions;
    std::vector<etk_atom_offsets_t> etk_atom_offsets;
    etkdg_molecule_metadata_t metadata;
};

/**
 *  @brief Build the distance bounds matrix from a molecular graph.
 *
 *  Uses RDKit's BoundsMatrixBuilder to compute initial bounds from
 *  bond lengths, VdW radii, and angle-derived 1-3 distances.
 *
 *  Output layout (N x N row-major):
 *  @code{.cpp}
 *  bounds[i*N + j]  // i < j → upper bound (angstroms)
 *  bounds[i*N + j]  // i > j → lower bound (angstroms)
 *  @endcode
 *
 *  @param[in] molecule RDKit molecule with explicit hydrogens.
 *  @param[out] bounds Flat N x N float array filled in row-major order.
 */
/**
 *  @brief Native zero-heap bounds-matrix builder.
 *
 *  Replaces RDKit's `setTopolBounds`, which allocates a fresh
 *  `DistGeom::BoundsMatrix` plus two `boost::shared_array<>` per molecule and
 *  walks RDKit's atom typer — roughly 140 KB of throwaway heap per call.
 *  This version reads only from the flat `molecule_descriptor_t` and the
 *  small `forcefield_tables.hpp` and writes directly into `bounds`.
 *
 *  The values are close but not bit-identical to `setTopolBounds`'s output.
 *  Downstream triangle smoothing (GPU) + FIRE refinement (GPU) absorb the
 *  small residual; we validate P50 energy within 1 kcal/mol of the RDKit
 *  baseline via the quality gate in `bench_chembl`.
 */
/**
 *  @brief End-to-end distance of a 1-4 chain `first — second = third — fourth` whose
 *  central bond is planar, at a torsion of 0 for cis or pi for trans.
 *
 *  Unlike a freely rotatable 1-4 pair, a stereo-defined double bond fixes this distance,
 *  which is how ETKDG enforces E/Z through the bounds matrix.
 */
inline float stereo_one_four_distance(molecule_descriptor_t const &desc, std::uint32_t first, std::uint32_t second,
                                      std::uint32_t third, std::uint32_t fourth, std::uint32_t bond_first_second,
                                      std::uint32_t bond_second_third, std::uint32_t bond_third_fourth, bool is_cis) {
    auto bond_length = [&](std::uint32_t bond_index, std::uint32_t a, std::uint32_t b) {
        bool const aromatic = (desc.bond_flags[bond_index] & bond_flag_aromatic_k) != 0;
        return ideal_bond_length_angstrom(desc.element[a], desc.element[b], desc.bond_order[bond_index], aromatic);
    };
    float const length_12 = bond_length(bond_first_second, first, second);
    float const length_23 = bond_length(bond_second_third, second, third);
    float const length_34 = bond_length(bond_third_fourth, third, fourth);

    float const angle_123 = ideal_angle_radians(desc.hybridization[second]);
    float const angle_234 = ideal_angle_radians(desc.hybridization[third]);
    float const cos_123 = std::cos(angle_123), sin_123 = std::sin(angle_123);
    float const cos_234 = std::cos(angle_234), sin_234 = std::sin(angle_234);
    float const cos_torsion = is_cis ? 1.0f : -1.0f;

    float const squared = length_12 * length_12 + length_23 * length_23 + length_34 * length_34 -
                          2.0f * length_12 * length_23 * cos_123 - 2.0f * length_23 * length_34 * cos_234 +
                          2.0f * length_12 * length_34 * (cos_123 * cos_234 - sin_123 * sin_234 * cos_torsion);
    return std::sqrt(std::max(0.01f, squared));
}

inline void build_bounds_matrix(molecule_descriptor_t const &desc, std::vector<float> &bounds) {
    std::uint32_t const num_atoms = desc.num_atoms;
    bounds.assign(static_cast<std::size_t>(num_atoms) * num_atoms, 0.0f);

    // The 1-4 and long-range sweeps below are the only host consumers of the hop matrix, so it
    // lives on this frame at the molecule's own row stride rather than inside the descriptor.
    std::uint8_t bond_hops[max_atoms_k * max_atoms_k];
    fill_bond_distances(desc, bond_hops, num_atoms);

    auto set_pair = [&](std::uint32_t a, std::uint32_t b, float lower, float upper) {
        std::uint32_t const row = std::min(a, b);
        std::uint32_t const col = std::max(a, b);
        bounds[row * num_atoms + col] = upper;
        bounds[col * num_atoms + row] = lower;
    };

    // Initial upper bound for every pair: large (effectively unconstrained).
    constexpr float upper_initial_k = 1000.0f;
    for (std::uint32_t i = 0; i < num_atoms; ++i)
        for (std::uint32_t j = i + 1; j < num_atoms; ++j) set_pair(i, j, 0.0f, upper_initial_k);

    // set12 — exact ideal bond lengths ± 2% tolerance per bond from the descriptor.
    constexpr float bond_tolerance_k = 0.02f;
    for (std::uint32_t b = 0; b < desc.num_bonds; ++b) {
        std::uint32_t const i = desc.bond_atom_i[b];
        std::uint32_t const j = desc.bond_atom_j[b];
        bool const aromatic = (desc.bond_flags[b] & bond_flag_aromatic_k) != 0;
        float const ideal = ideal_bond_length_angstrom(desc.element[i], desc.element[j], desc.bond_order[b], aromatic);
        set_pair(i, j, ideal * (1.0f - bond_tolerance_k), ideal * (1.0f + bond_tolerance_k));
    }

    // set13 — one 1-3 bound per pair of neighbors around every center atom.
    // Loop is O(N × d²) where d is the typical degree (small).
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint32_t const nbegin = desc.neighbor_offset[center];
        std::uint32_t const nend = desc.neighbor_offset[center + 1];
        float const theta = ideal_angle_radians(desc.hybridization[center]);
        float const cos_theta = std::cos(theta);
        for (std::uint32_t p = nbegin; p < nend; ++p) {
            std::uint32_t const i = desc.neighbor_atom[p];
            std::uint32_t const bond_ci = desc.neighbor_bond[p];
            bool const ar_ci = (desc.bond_flags[bond_ci] & bond_flag_aromatic_k) != 0;
            float const d_ci = ideal_bond_length_angstrom(desc.element[center], desc.element[i],
                                                          desc.bond_order[bond_ci], ar_ci);
            for (std::uint32_t q = p + 1; q < nend; ++q) {
                std::uint32_t const j = desc.neighbor_atom[q];
                std::uint32_t const bond_cj = desc.neighbor_bond[q];
                bool const ar_cj = (desc.bond_flags[bond_cj] & bond_flag_aromatic_k) != 0;
                float const d_cj = ideal_bond_length_angstrom(desc.element[center], desc.element[j],
                                                              desc.bond_order[bond_cj], ar_cj);
                float const d_ij = std::sqrt(
                    std::max(0.0f, d_ci * d_ci + d_cj * d_cj - 2.0f * d_ci * d_cj * cos_theta));
                // ± 3% tolerance — 1-3 distances tend to have slightly more slack.
                set_pair(i, j, d_ij * 0.97f, d_ij * 1.03f);
            }
        }
    }

    // Collect the E/Z-defined double bonds up front.  Drug-like molecules have none or a
    // handful, so the 1-4 sweep below scans a tiny list rather than all bonds, and skips
    // the check entirely in the common zero case.
    std::uint32_t stereo_bonds[max_bonds_k];
    std::uint32_t num_stereo_bonds = 0;
    for (std::uint32_t b = 0; b < desc.num_bonds; ++b)
        if ((desc.bond_flags[b] & (bond_flag_stereo_e_k | bond_flag_stereo_z_k)) != 0 &&
            desc.bond_stereo_atom_i[b] != no_stereo_atom_k)
            stereo_bonds[num_stereo_bonds++] = b;

    auto find_bond_between = [&](std::uint32_t a, std::uint32_t b) -> std::int32_t {
        std::uint32_t const nbegin = desc.neighbor_offset[a];
        std::uint32_t const nend = desc.neighbor_offset[a + 1];
        for (std::uint32_t p = nbegin; p < nend; ++p)
            if (desc.neighbor_atom[p] == b) return static_cast<std::int32_t>(desc.neighbor_bond[p]);
        return -1;
    };

    // set14 — conservative envelope for 1-4 pairs. Takes the sum of two bond lengths
    // plus a torsion-dependent spread as the upper bound; VdW minimum sum as the lower.
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        for (std::uint32_t j = i + 1; j < num_atoms; ++j) {
            std::uint8_t const hops = bond_hops[i * num_atoms + j];
            if (hops != 3) continue;

            // A 1-4 pair spanning a stereo-defined double bond has a determined
            // end-to-end distance rather than a free envelope.  Leaving the generic
            // 4.5 Å upper in place lets smoothing and FIRE settle on either isomer.
            bool stereo_constrained = false;
            for (std::uint32_t s = 0; s < num_stereo_bonds && !stereo_constrained; ++s) {
                std::uint32_t const bond = stereo_bonds[s];
                std::uint32_t end_near_i = desc.bond_atom_i[bond];
                std::uint32_t end_near_j = desc.bond_atom_j[bond];
                std::int32_t bond_i = find_bond_between(i, end_near_i);
                std::int32_t bond_j = find_bond_between(j, end_near_j);
                if (bond_i < 0 || bond_j < 0) {
                    std::swap(end_near_i, end_near_j);
                    bond_i = find_bond_between(i, end_near_i);
                    bond_j = find_bond_between(j, end_near_j);
                    if (bond_i < 0 || bond_j < 0) continue;
                }
                // The flag fixes the relationship of the two recorded reference atoms.
                // Any other substituent on an end sits opposite that end's reference, so
                // each mismatch flips the sense.
                std::uint32_t const reference_i = desc.bond_stereo_atom_i[bond];
                std::uint32_t const reference_j = desc.bond_stereo_atom_j[bond];
                bool reference_on_i_side = find_bond_between(reference_i, end_near_i) >= 0;
                std::uint32_t const own_reference = reference_on_i_side ? reference_i : reference_j;
                std::uint32_t const other_reference = reference_on_i_side ? reference_j : reference_i;
                bool is_cis = (desc.bond_flags[bond] & bond_flag_stereo_z_k) != 0;
                if (i != own_reference) is_cis = !is_cis;
                if (j != other_reference) is_cis = !is_cis;

                float const target = stereo_one_four_distance(desc, i, end_near_i, end_near_j, j,
                                                              static_cast<std::uint32_t>(bond_i), bond,
                                                              static_cast<std::uint32_t>(bond_j), is_cis);
                set_pair(i, j, target * 0.95f, target * 1.05f);
                stereo_constrained = true;
            }
            if (stereo_constrained) continue;

            // Approximate: upper ≈ 3 × average C-C bond ≈ 4.5 Å; lower ≈ VdW contact.
            float const lower = vdw_lower_bound_angstrom(desc.element[i], desc.element[j]);
            float const upper = 4.5f;
            // Only widen — never replace a tighter (i,j) bound from set13.
            std::uint32_t const row = std::min(i, j);
            std::uint32_t const col = std::max(i, j);
            float const current_upper = bounds[row * num_atoms + col];
            float const current_lower = bounds[col * num_atoms + row];
            if (current_upper >= upper_initial_k) bounds[row * num_atoms + col] = upper;
            if (current_lower == 0.0f) bounds[col * num_atoms + row] = lower;
        }
    }

    // Long-range pairs (more than three hops apart): VdW lower bound, keep upper unconstrained.
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        for (std::uint32_t j = i + 1; j < num_atoms; ++j) {
            std::uint8_t const hops = bond_hops[i * num_atoms + j];
            if (hops <= 3) continue;
            float const lower = vdw_lower_bound_angstrom(desc.element[i], desc.element[j]);
            bounds[j * num_atoms + i] = lower; // lower triangle
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
 *  @param[in] desc Flat descriptor supplying the chiral tags and the CSR adjacency.
 *  @param[out] chiral_centers Filled with one descriptor per tetrahedral center.
 */
inline void extract_chiral_centers(RDKit::ROMol const &molecule, molecule_descriptor_t const &desc,
                                   std::vector<chiral_center_descriptor_t> &chiral_centers) {
    chiral_centers.clear();
    auto &stereo_molecule = const_cast<RDKit::ROMol &>(molecule);
    RDKit::MolOps::assignStereochemistry(stereo_molecule, true, true, true);

    std::uint32_t const num_atoms = desc.num_atoms;
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        if (desc.chiral_tag[a] == 0) continue;
        if (desc.degree[a] < 3) continue;

        // `assignStereochemistry` strips the tag off every atom that parsed with one but is
        // not a stereocenter, so a surviving tag is the membership test for this center.
        auto const assigned_tag = molecule.getAtomWithIdx(a)->getChiralTag();
        bool const is_stereocenter = assigned_tag == RDKit::Atom::CHI_TETRAHEDRAL_CW ||
                                     assigned_tag == RDKit::Atom::CHI_TETRAHEDRAL_CCW;
        if (!is_stereocenter) continue;

        chiral_center_descriptor_t descriptor {};
        descriptor.center_atom = static_cast<std::uint8_t>(a);

        // Collect the neighbors in CSR order, which is the bond order the tag is defined against.
        std::uint8_t ordered[4] = {0, 0, 0, 0};
        std::uint32_t count = 0;
        std::uint16_t const neighbor_first = desc.neighbor_offset[a];
        std::uint16_t const neighbor_last = desc.neighbor_offset[a + 1];
        for (std::uint16_t n = neighbor_first; n < neighbor_last && count < 4; ++n)
            ordered[count++] = desc.neighbor_atom[n];

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

        for (std::uint32_t slot = 0; slot < 4; ++slot) descriptor.neighbor_atoms[slot] = ordered[slot];

        int const tag_sign = (desc.chiral_tag[a] == 1) ? 1 : -1;
        int const order_parity = (swaps % 2 == 0) ? 1 : -1;
        descriptor.target_sign = static_cast<std::int8_t>(-tag_sign * order_parity);

        chiral_centers.push_back(descriptor);
    }
}

/**
 *  @brief Extract 1-2 bond and 1-3 angle-derived distance constraints for the ETK refinement stage.
 *
 *  Reads actual distance bounds from @p bounds_matrix instead of using
 *  hardcoded values. The bounds matrix already contains chemically-correct
 *  values from RDKit's setTopolBounds().
 *
 *  @param[in] molecule RDKit molecule whose topology defines the constraints.
 *  @param[in] bounds_matrix Flat N x N distance bounds (row-major, from build_bounds_matrix()).
 *  @param[out] constraints Filled with 1-2 bond and 1-3 angle distance constraints.
 */
inline void extract_distance_constraints(molecule_descriptor_t const &desc, std::vector<float> const &bounds_matrix,
                                         std::vector<distance_constraint_t> &constraints) {
    constraints.clear();
    std::uint32_t const num_atoms = desc.num_atoms;

    // Reserve exact capacity: 1-2 count = num_bonds, 1-3 count = sum choose(degree, 2).
    std::uint32_t num_13 = 0;
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        std::uint32_t d = desc.degree[a];
        num_13 += d * (d - 1) / 2;
    }
    constraints.reserve(desc.num_bonds + num_13);

    // 1-2 bond constraints: one emit per bond, exact bounds from the matrix.
    for (std::uint32_t b = 0; b < desc.num_bonds; ++b) {
        std::uint32_t const atom_i = desc.bond_atom_i[b];
        std::uint32_t const atom_j = desc.bond_atom_j[b];
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
        std::uint16_t const start = desc.neighbor_offset[center];
        std::uint16_t const end = desc.neighbor_offset[center + 1];
        for (std::uint16_t a = start; a < end; ++a) {
            for (std::uint16_t b = a + 1; b < end; ++b) {
                std::uint32_t const ni = desc.neighbor_atom[a];
                std::uint32_t const nj = desc.neighbor_atom[b];
                std::uint32_t const row_min = std::min(ni, nj);
                std::uint32_t const row_max = std::max(ni, nj);

                distance_constraint_t constraint {};
                constraint.atom_index_i = static_cast<std::uint8_t>(ni);
                constraint.atom_index_j = static_cast<std::uint8_t>(nj);
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
 *  @brief Extract ETK torsion terms with CSD Fourier coefficients.
 *
 *  Uses RDKit's experimental torsion angle preferences (Riniker & Landrum,
 *  JCIM 2015) via ForceFields::CrystalFF::getExperimentalTorsions() to obtain
 *  atom-type-specific Fourier coefficients from the Cambridge Structural
 *  Database torsion library. Falls back to generic 3-fold barriers when no
 *  CSD torsions are found.
 *
 *  @param[in] molecule RDKit molecule with explicit hydrogens.
 *  @param[out] torsions Filled with one descriptor per torsion term.
 */
inline void extract_etk_torsions(RDKit::ROMol const &molecule, std::vector<etk_torsion_descriptor_t> &torsions) {
    torsions.clear();

    // Get CSD experimental torsion preferences from RDKit.
    ForceFields::CrystalFF::CrystalFFDetails crystal_details;
    ForceFields::CrystalFF::getExperimentalTorsions(molecule, crystal_details,
                                                    /*useExpTorsions=*/true,
                                                    /*useSmallRingTorsions=*/true,
                                                    /*useMacrocycleTorsions=*/false,
                                                    /*useBasicKnowledge=*/true,
                                                    /*version=*/2);

    // Convert RDKit's torsion data to the SmallField descriptor format.
    for (std::size_t t = 0; t < crystal_details.expTorsionAtoms.size(); ++t) {
        auto const &atom_indices = crystal_details.expTorsionAtoms[t];
        if (atom_indices.size() < 4) continue;

        auto const &angle_data = crystal_details.expTorsionAngles[t];
        auto const &signs = angle_data.first;
        auto const &force_constants = angle_data.second;

        etk_torsion_descriptor_t torsion {};
        torsion.atom_indices[0] = static_cast<std::uint8_t>(atom_indices[0]);
        torsion.atom_indices[1] = static_cast<std::uint8_t>(atom_indices[1]);
        torsion.atom_indices[2] = static_cast<std::uint8_t>(atom_indices[2]);
        torsion.atom_indices[3] = static_cast<std::uint8_t>(atom_indices[3]);

        // Combine signs and force constants: Vn = signs[n] * forceConstants[n].
        for (std::size_t n = 0; n < 6 && n < force_constants.size(); ++n) {
            float sign = (n < signs.size()) ? static_cast<float>(signs[n]) : 1.0f;
            torsion.fourier_coefficients[n] = sign * static_cast<float>(force_constants[n]);
        }

        torsions.push_back(torsion);
    }

    // If no CSD torsions are found, fall back to generic 3-fold barriers for rotatable bonds.
    if (torsions.empty()) {
        for (auto const &bond : molecule.bonds()) {
            if (bond->getIsAromatic() || bond->getBondTypeAsDouble() > 1.5) continue;

            unsigned int atom_j = bond->getBeginAtomIdx();
            unsigned int atom_k = bond->getEndAtomIdx();
            unsigned int atom_i = static_cast<unsigned int>(-1);
            unsigned int atom_l = static_cast<unsigned int>(-1);

            for (auto const &neighbor : molecule.atomNeighbors(molecule.getAtomWithIdx(atom_j))) {
                if (neighbor->getIdx() != atom_k) {
                    atom_i = neighbor->getIdx();
                    break;
                }
            }
            for (auto const &neighbor : molecule.atomNeighbors(molecule.getAtomWithIdx(atom_k))) {
                if (neighbor->getIdx() != atom_j) {
                    atom_l = neighbor->getIdx();
                    break;
                }
            }
            if (atom_i == static_cast<unsigned int>(-1) || atom_l == static_cast<unsigned int>(-1)) continue;

            etk_torsion_descriptor_t torsion {};
            torsion.atom_indices[0] = static_cast<std::uint8_t>(atom_i);
            torsion.atom_indices[1] = static_cast<std::uint8_t>(atom_j);
            torsion.atom_indices[2] = static_cast<std::uint8_t>(atom_k);
            torsion.atom_indices[3] = static_cast<std::uint8_t>(atom_l);
            torsion.fourier_coefficients[2] = 1.0f; // Generic V₃ = 1.0.
            torsions.push_back(torsion);
        }
    }
}

/**
 *  @brief Build per-atom interaction lists for ETK torsions and distance constraints.
 *
 *  Reads torsion and constraint arrays from @p result and populates the
 *  per-atom ETK interaction list and offset arrays.
 *
 *  @param[inout] result Prepared molecule; ETK interaction lists are filled in place.
 */
inline void build_etk_interaction_lists(etkdg_prepared_molecule_t &result) {
    std::uint32_t const num_atoms = result.num_atoms;
    auto const &torsions = result.etk_torsions;
    auto const &constraints = result.distance_constraints;

    // pass 1: count interactions per atom. num_atoms <= max_atoms_k so the
    // fixed-size std::array lives on the stack and avoids a per-molecule heap
    // allocation on the hot prep path.
    std::array<std::uint32_t, max_atoms_k> counts {};
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
    std::array<std::uint32_t, max_atoms_k> cursors {};
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
        cursors[atom] = result.etk_atom_offsets[atom].interaction_start;

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
 *  Uses flat boolean matrices for O(1) exclusion/1-4 lookup (no std::set).
 *  All output is written directly to @p result arrays without temporary
 *  heap allocations.
 *
 *  @param[in] molecule RDKit molecule with explicit hydrogens.
 *  @param[inout] result Prepared molecule; MMFF94 arrays are filled in place.
 *  @return `conformer_status_t::success` on success;
 *          `conformer_status_t::unsupported_element` if any atom's
 *          MMFF94 atom type is 0 (element outside the MMFF94 parameter
 *          set — transition metals, lanthanides, etc.) or if the
 *          `MMFFMolProperties` constructor otherwise reports the
 *          molecule as not parametrizable.
 */
inline smallfield::api::conformer_status_t extract_mmff94_params(RDKit::ROMol const &molecule,
                                                                 molecule_descriptor_t const &desc,
                                                                 etkdg_prepared_molecule_t &result) {
    result.mmff_atom_vdw.clear();
    result.mmff_atom_charge.clear();
    result.mmff_bonds.clear();
    result.mmff_angles.clear();
    result.mmff_torsions.clear();

    auto &mol = const_cast<RDKit::ROMol &>(molecule);
    // Skip redundant Kekulize inside MMFFMolProperties — SmilesToMol already sanitized.
    if (!mol.hasProp(RDKit::common_properties::_MMFFSanitized)) {
        mol.setProp(RDKit::common_properties::_MMFFSanitized, 1, true);
    }
    RDKit::MMFF::MMFFMolProperties mmff_props(mol);

    std::uint32_t const num_atoms = desc.num_atoms;

    // Precompute per-atom MMFF types and per-bond signatures for the global parameter cache in
    // src/mmff94/mmff94_param_cache.hpp. Atom types are cheap array reads from mmff_props and
    // bond signatures come straight from `desc`. These feed the cache-key packers so we can short-circuit
    // RDKit's per-term std::map lookups.
    //
    // `getMMFFAtomType` returns 0 for atoms outside the MMFF94 parameter set
    // (Y, lanthanides, transition metals, etc.).  Detect that here on the
    // host so the molecule never reaches the GPU pipeline — without this
    // gate, `etkdg_pipeline_warp_kernel` does an OOB write through
    // element-keyed parameter LUTs in `forcefield_tables.hpp` and poisons
    // every subsequent CUDA call on every device.
    std::uint8_t atom_type[max_atoms_k];
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        atom_type[a] = static_cast<std::uint8_t>(mmff_props.getMMFFAtomType(a));
        if (atom_type[a] == 0) return smallfield::api::conformer_status_t::unsupported_element;
    }

    // Defence in depth: catch the rare case of all-typed atoms but
    // unparametrized interactions (e.g. missing torsion params).  Same
    // routing as the per-atom case — the molecule can't be MMFF-optimised
    // on-device and must fall back to CPU ETKDGv3.
    if (!mmff_props.isValid()) return smallfield::api::conformer_status_t::unsupported_element;
    std::uint8_t bond_signature[max_bonds_k];
    for (std::uint32_t b = 0; b < desc.num_bonds; ++b) {
        bool const arom = (desc.bond_flags[b] & bond_flag_aromatic_k) != 0;
        bool const in_ring = (desc.bond_flags[b] & bond_flag_in_ring_k) != 0;
        bond_signature[b] = pack_bond_signature_(desc.bond_order[b], arom, in_ring);
    }
    auto find_bond_idx = [&](std::uint32_t i, std::uint32_t j) -> std::int32_t {
        for (std::uint16_t n = desc.neighbor_offset[i]; n < desc.neighbor_offset[i + 1]; ++n)
            if (desc.neighbor_atom[n] == j) return desc.neighbor_bond[n];
        return -1;
    };

    // Per-atom data: VdW params + partial charge. The kernel's nonbonded loop enumerates
    // pairs on the fly from the bond_distance matrix and applies Halgren's R* combining
    // rule plus Slater-Kirkwood epsilon combining inline — no per-pair host emission.
    result.mmff_atom_vdw.resize(num_atoms);
    result.mmff_atom_charge.resize(num_atoms);
    auto const *mmff_vdw_coll = RDKit::MMFF::DefaultParameters::getMMFFVdW();
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        std::uint32_t const atom_type = mmff_props.getMMFFAtomType(a);
        ForceFields::MMFF::MMFFVdW const *vdw = (*mmff_vdw_coll)(atom_type);
        mmff94_atom_vdw_params_t params {};
        if (vdw) {
            params.alpha = static_cast<float>(vdw->alpha_i);
            params.n_effective = static_cast<float>(vdw->N_i);
            params.a_scale = static_cast<float>(vdw->A_i);
            params.g_factor = static_cast<float>(vdw->G_i);
            params.r_star = static_cast<float>(vdw->R_star);
            // DA flag: RDKit stores the character 'D'/'A'/'-' in vdw->DA.
            params.donor_acceptor = (vdw->DA == 'D') ? 2 : (vdw->DA == 'A') ? 1 : 0;
        }
        result.mmff_atom_vdw[a] = params;
        result.mmff_atom_charge[a] = static_cast<float>(mmff_props.getMMFFPartialCharge(a));
    }

    // Build bond terms — iterate descriptor's bond list (no RDKit traversal).
    // Cache-first: key by (atom_type_pair, bond_signature). Cache hit skips
    // RDKit's getMMFFBondStretchParams + its internal std::map lookup.
    for (std::uint32_t b = 0; b < desc.num_bonds; ++b) {
        std::uint32_t const i = desc.bond_atom_i[b];
        std::uint32_t const j = desc.bond_atom_j[b];
        std::uint64_t const key = make_bond_stretch_key_(atom_type[i], atom_type[j], bond_signature[b]);
        mmff_bond_cached_t cached_vals {};
        if (auto const *hit = mmff_bond_cache().find(key)) { cached_vals = *hit; }
        else {
            unsigned int bond_type;
            ForceFields::MMFF::MMFFBond bp;
            // Dropping the term would understate the molecule's energy by a whole bond stretch, and
            // the device hop-count BFS derives adjacency from this list, so a missing entry also
            // shifts the 1-2 / 1-3 / 1-4 nonbonded exclusions.  Reject the molecule instead and let
            // the fallback role handle it.
            if (!mmff_props.getMMFFBondStretchParams(mol, i, j, bond_type, bp))
                return smallfield::api::conformer_status_t::unsupported_element;
            cached_vals.kb = static_cast<float>(bp.kb);
            cached_vals.r0 = static_cast<float>(bp.r0);
            mmff_bond_cache().insert(key, cached_vals);
        }
        result.mmff_bonds.push_back(
            {static_cast<std::uint8_t>(i), static_cast<std::uint8_t>(j), {0, 0}, cached_vals.kb, cached_vals.r0, 0.0f});
    }

    // Build angle terms — iterate atoms as angle centers, enumerate neighbor pairs via CSR.
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint16_t const nstart = desc.neighbor_offset[center];
        std::uint16_t const nend = desc.neighbor_offset[center + 1];
        for (std::uint16_t a = nstart; a < nend; ++a) {
            for (std::uint16_t b = a + 1; b < nend; ++b) {
                std::uint32_t const ai = desc.neighbor_atom[a];
                std::uint32_t const aj = desc.neighbor_atom[b];
                std::int32_t const bond_index_ic = find_bond_idx(ai, center);
                std::int32_t const bond_index_cj = find_bond_idx(center, aj);
                if (bond_index_ic < 0 || bond_index_cj < 0) continue;
                std::uint64_t const key = make_angle_key_(atom_type[ai], atom_type[center], atom_type[aj],
                                                          bond_signature[bond_index_ic], bond_signature[bond_index_cj]);
                mmff_angle_cached_t cached_vals {};
                if (auto const *hit = mmff_angle_cache().find(key)) { cached_vals = *hit; }
                else {
                    unsigned int angle_type;
                    ForceFields::MMFF::MMFFAngle ap;
                    if (!mmff_props.getMMFFAngleBendParams(mol, ai, center, aj, angle_type, ap)) continue;
                    cached_vals.ka = static_cast<float>(ap.ka);
                    cached_vals.theta0_rad = static_cast<float>(ap.theta0 * M_PI / 180.0);
                    mmff_angle_cache().insert(key, cached_vals);
                }
                // `linh` depends only on the central atom's MMFF type, so it is read
                // outside the angle-parameter cache rather than widening that key.
                auto const *prop_center = (*RDKit::MMFF::DefaultParameters::getMMFFProp())(atom_type[center]);
                std::uint8_t const is_linear = (prop_center && prop_center->linh) ? 1u : 0u;
                result.mmff_angles.push_back({static_cast<std::uint8_t>(ai), static_cast<std::uint8_t>(center),
                                              static_cast<std::uint8_t>(aj), is_linear, cached_vals.ka,
                                              cached_vals.theta0_rad, 0.0f});
            }
        }
    }

    // Build torsion terms — iterate descriptor's bonds as central, enumerate end-atom neighbors.
    for (std::uint32_t b = 0; b < desc.num_bonds; ++b) {
        std::uint32_t const j = desc.bond_atom_i[b];
        std::uint32_t const k = desc.bond_atom_j[b];
        std::uint16_t const jstart = desc.neighbor_offset[j];
        std::uint16_t const jend = desc.neighbor_offset[j + 1];
        std::uint16_t const kstart = desc.neighbor_offset[k];
        std::uint16_t const kend = desc.neighbor_offset[k + 1];
        std::uint8_t const bsig_jk = bond_signature[b];
        for (std::uint16_t nj = jstart; nj < jend; ++nj) {
            std::uint32_t const i = desc.neighbor_atom[nj];
            if (i == k) continue;
            std::uint16_t const bond_index_ij = desc.neighbor_bond[nj];
            std::uint8_t const bsig_ij = bond_signature[bond_index_ij];
            for (std::uint16_t nk = kstart; nk < kend; ++nk) {
                std::uint32_t const l = desc.neighbor_atom[nk];
                if (l == j) continue;
                std::uint16_t const bond_index_kl = desc.neighbor_bond[nk];
                std::uint8_t const bsig_kl = bond_signature[bond_index_kl];
                std::uint64_t const key = make_torsion_key_(atom_type[i], atom_type[j], atom_type[k], atom_type[l],
                                                            bsig_ij, bsig_jk, bsig_kl);
                mmff_torsion_cached_t cached_vals {};
                if (auto const *hit = mmff_torsion_cache().find(key)) { cached_vals = *hit; }
                else {
                    unsigned int torsion_type;
                    ForceFields::MMFF::MMFFTor tp;
                    if (!mmff_props.getMMFFTorsionParams(mol, i, j, k, l, torsion_type, tp)) continue;
                    cached_vals.V1 = static_cast<float>(tp.V1);
                    cached_vals.V2 = static_cast<float>(tp.V2);
                    cached_vals.V3 = static_cast<float>(tp.V3);
                    mmff_torsion_cache().insert(key, cached_vals);
                }
                result.mmff_torsions.push_back({static_cast<std::uint8_t>(i), static_cast<std::uint8_t>(j),
                                                static_cast<std::uint8_t>(k), static_cast<std::uint8_t>(l),
                                                cached_vals.V1, cached_vals.V2, cached_vals.V3});
            }
        }
    }

    // Build out-of-plane bending terms for sp2 centers with ≥3 neighbors.
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
            for (std::uint32_t b = 0; b < neighbor_count && plane_count < 2; ++b)
                if (b != a) plane[plane_count++] = neighbor_cache[b];
            if (plane_count < 2) continue;
            std::uint64_t const key = make_oop_key_(atom_type[center], atom_type[neighbor_cache[a]],
                                                    atom_type[plane[0]], atom_type[plane[1]]);
            mmff_oop_cached_t cached_vals {};
            if (auto const *hit = mmff_oop_cache().find(key)) { cached_vals = *hit; }
            else {
                ForceFields::MMFF::MMFFOop oop_params;
                if (!mmff_props.getMMFFOopBendParams(mol, neighbor_cache[a], center, plane[0], plane[1], oop_params))
                    continue;
                cached_vals.koop = static_cast<float>(oop_params.koop);
                mmff_oop_cache().insert(key, cached_vals);
            }
            result.mmff_oops.push_back({neighbor_cache[a],
                                        static_cast<std::uint8_t>(center),
                                        plane[0],
                                        plane[1],
                                        cached_vals.koop,
                                        {0.0f, 0.0f}});
        }
    }

    // Build stretch-bend terms — same topology as angles.
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint16_t const nstart = desc.neighbor_offset[center];
        std::uint16_t const nend = desc.neighbor_offset[center + 1];
        for (std::uint16_t a = nstart; a < nend; ++a) {
            for (std::uint16_t b = a + 1; b < nend; ++b) {
                std::uint32_t const ai = desc.neighbor_atom[a];
                std::uint32_t const aj = desc.neighbor_atom[b];
                std::int32_t const bond_index_ic = find_bond_idx(ai, center);
                std::int32_t const bond_index_cj = find_bond_idx(center, aj);
                if (bond_index_ic < 0 || bond_index_cj < 0) continue;
                std::uint64_t const key = make_stretchbend_key_(atom_type[ai], atom_type[center], atom_type[aj],
                                                                bond_signature[bond_index_ic],
                                                                bond_signature[bond_index_cj]);
                mmff_stretchbend_cached_t cached_vals {};
                if (auto const *hit = mmff_stretchbend_cache().find(key)) { cached_vals = *hit; }
                else {
                    unsigned int stretchbend_type;
                    ForceFields::MMFF::MMFFStbn stretchbend_params;
                    ForceFields::MMFF::MMFFBond bond_params[2];
                    ForceFields::MMFF::MMFFAngle angle_params;
                    if (!mmff_props.getMMFFStretchBendParams(mol, ai, center, aj, stretchbend_type, stretchbend_params,
                                                             bond_params, angle_params))
                        continue;
                    cached_vals.kbaIJK = static_cast<float>(stretchbend_params.kbaIJK);
                    cached_vals.kbaKJI = static_cast<float>(stretchbend_params.kbaKJI);
                    cached_vals.theta0_rad = static_cast<float>(angle_params.theta0 * M_PI / 180.0);
                    cached_vals.r0_ij = static_cast<float>(bond_params[0].r0);
                    cached_vals.r0_jk = static_cast<float>(bond_params[1].r0);
                    mmff_stretchbend_cache().insert(key, cached_vals);
                }
                result.mmff_stretchbends.push_back({static_cast<std::uint8_t>(ai), static_cast<std::uint8_t>(center),
                                                    static_cast<std::uint8_t>(aj), 0, cached_vals.kbaIJK,
                                                    cached_vals.kbaKJI, cached_vals.theta0_rad, cached_vals.r0_ij,
                                                    cached_vals.r0_jk, 0.0f});
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
    // sized to the hard atom-count cap (max_atoms_k=128) — no per-molecule
    // heap allocation on the prep path.
    struct per_atom_counts_t {
        // `uint32_t` not `uint8_t`: see `mmff94_atom_offsets_t` comment.
        std::uint32_t bond, angle, torsion, oop, stretchbend;
    };
    std::array<per_atom_counts_t, max_atoms_k> type_counts {};
    for (auto const &b : bonds) {
        type_counts[b.atom_i].bond++;
        type_counts[b.atom_j].bond++;
    }
    for (auto const &a : angles) {
        type_counts[a.atom_i].angle++;
        type_counts[a.atom_j].angle++;
        type_counts[a.atom_k].angle++;
    }
    for (auto const &t : torsions) {
        type_counts[t.atom_i].torsion++;
        type_counts[t.atom_j].torsion++;
        type_counts[t.atom_k].torsion++;
        type_counts[t.atom_l].torsion++;
    }
    for (auto const &o : oops) {
        type_counts[o.atom_i].oop++;
        type_counts[o.atom_j].oop++;
        type_counts[o.atom_k].oop++;
        type_counts[o.atom_l].oop++;
    }
    for (auto const &s : stretchbends) {
        type_counts[s.atom_i].stretchbend++;
        type_counts[s.atom_j].stretchbend++;
        type_counts[s.atom_k].stretchbend++;
    }

    // compute per-type offsets: bonds first, then angles, etc. No nonbonded range.
    // Sanity-assert that no per-atom interaction count exceeds the documented
    // bound `max_mmff_interactions_per_mol_k`.  The original `uint8_t` storage
    // silently wrapped here for dense drug-like atoms with >255 torsions; the
    // new `uint32_t` storage removes the ceiling, but the assert catches any
    // upstream bug that would still stage corrupt data to the GPU.
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
        offsets.bond_start = running;
        offsets.bond_count = counts.bond;
        running += counts.bond;
        offsets.angle_start = running;
        offsets.angle_count = counts.angle;
        running += counts.angle;
        offsets.torsion_start = running;
        offsets.torsion_count = counts.torsion;
        running += counts.torsion;
        offsets.oop_start = running;
        offsets.oop_count = counts.oop;
        running += counts.oop;
        offsets.stretchbend_start = running;
        offsets.stretchbend_count = counts.stretchbend;
        running += counts.stretchbend;
    }

    // pass 2: fill interaction entries grouped by type per atom
    std::uint32_t const total_interactions = running;
    result.mmff_interactions.resize(total_interactions);

    // per-type cursors per atom — stack-sized, same rationale as type_counts.
    struct per_atom_cursors_t {
        std::uint32_t bond, angle, torsion, oop, stretchbend;
    };
    std::array<per_atom_cursors_t, max_atoms_k> cursors {};
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        auto const &off = result.mmff_atom_offsets[atom];
        cursors[atom] = {off.bond_start, off.angle_start, off.torsion_start, off.oop_start, off.stretchbend_start};
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
    return smallfield::api::conformer_status_t::success;
}

/**
 *  @brief Fill a caller-owned @p result with stages 0-1 output for one SMILES.
 *
 *  Called from @c prepare_molecule_batch with a thread-local @c result that is
 *  reused across every molecule the thread processes — each molecule's vectors
 *  keep their capacity from the previous call, so push_back stays on the hot
 *  path without heap allocation after the first few molecules.  Sets
 *  @c result.num_atoms = 0 on failure and returns the rejection reason; on
 *  success returns @c conformer_status_t::success.  Callers that don't need
 *  the per-molecule reason code can still test @c result.num_atoms == 0 as
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

    using smallfield::api::conformer_status_t;

    std::unique_ptr<RDKit::RWMol> mol;
    try {
        // RDKit::SmilesToMol takes std::string const& — one unavoidable std::string
        // construction per molecule. Kept scoped to this single call.
        // removeHs=false: we immediately call addHs() below, so letting the parser
        // strip explicit Hs first would waste ~5 % of CPU prep time.
        std::string const smiles_str {smiles};
        RDKit::SmilesParserParams params;
        params.sanitize = true;
        params.removeHs = false;
        params.strictCXSMILES = false;
        mol.reset(RDKit::SmilesToMol(smiles_str, params));
    }
    catch (...) {
        return conformer_status_t::invalid_smiles;
    }
    if (!mol) return conformer_status_t::invalid_smiles;

    try {
        RDKit::MolOps::addHs(*mol);
    }
    catch (...) {
        return conformer_status_t::invalid_smiles;
    }
    std::uint32_t const num_atoms = mol->getNumAtoms();
    if (num_atoms > 128) return conformer_status_t::too_many_atoms; // single-warp pipeline cap
    result.num_atoms = num_atoms;

    // Optional host-side readback fields — read once from the post-AddHs RDKit
    // mol so callers can populate `n_heavy_atoms`, `n_bonds`, `molecular_weight`
    // columns without a duplicate parse on the Python side.
    result.metadata.num_heavy_atoms = static_cast<std::uint32_t>(mol->getNumHeavyAtoms());
    result.metadata.num_bonds = static_cast<std::uint32_t>(mol->getNumBonds());
    try {
        result.metadata.molecular_weight = static_cast<float>(RDKit::Descriptors::calcExactMW(*mol));
    }
    catch (...) {
        result.metadata.molecular_weight = 0.0f;
    }

    conformer_status_t mmff_status = conformer_status_t::success;
    try {
        molecule_descriptor_t desc;
        if (!build_descriptor(*mol, desc)) {
            result.num_atoms = 0;
            return conformer_status_t::invalid_smiles;
        }
        build_bounds_matrix(desc, result.bounds_matrix);
        extract_chiral_centers(*mol, desc, result.chiral_centers);
        extract_etk_torsions_native(*mol, desc, result.etk_torsions);
        extract_distance_constraints(desc, result.bounds_matrix, result.distance_constraints);
        build_etk_interaction_lists(result);
        mmff_status = extract_mmff94_params(*mol, desc, result);
    }
    catch (...) {
        result.num_atoms = 0;
        return conformer_status_t::invalid_smiles;
    }
    if (mmff_status != conformer_status_t::success) {
        // MMFF could not parametrise this molecule (e.g. atomic number outside
        // the MMFF94 set).  The mmff_* result vectors are empty after the
        // early return inside `extract_mmff94_params`, but ETK fields above
        // are populated; clear num_atoms so downstream packing skips this mol.
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
    result.metadata.mmff_atom_offsets_start = 0;
    result.metadata.etk_interaction_offset = 0;
    result.metadata.etk_atom_offsets_start = 0;
    result.metadata.bounds_min = 0.0f;
    result.metadata.bounds_inv_scale = 1.0f;
    return conformer_status_t::success;
}

/**
 *  @brief Convert a SMILES string to a fresh `etkdg_prepared_molecule_t`.
 *
 *  Thin wrapper around `prepare_molecule_into` used by tests and by the
 *  non-pipelined path.  Allocates a fresh struct per call — NOT the hot path;
 *  the batched path inside `prepare_molecule_batch` uses thread_local scratch
 *  to keep vector capacities.
 */
inline etkdg_prepared_molecule_t prepare_molecule(std::string_view smiles) {
    etkdg_prepared_molecule_t result {};
    prepare_molecule_into(smiles, result);
    return result;
}

/**
 *  @brief Batch of molecules packed into a single pinned host buffer.
 *
 *  Type alias onto `batch_pool_t`, which owns one contiguous pinned host
 *  allocation plus a byte-level `batch_layout_t` describing 16 sub-regions
 *  (14 per-molecule arrays + metadata + input_indices).  Every sub-array is
 *  exposed as a `std::span<T>` view via member accessors on `batch_pool_t`.
 *  The launch path issues ONE `cudaMemcpyAsync` of the host buffer into a
 *  per-GPU device mirror, replacing 14 pageable-source transfers that
 *  previously serialized the dispatcher.
 */
using etkdg_molecule_batch_t = batch_pool_t;

/**
 *  @brief Per-molecule sizes of every output array in etkdg_molecule_batch_t.
 *
 *  Packed into one 64-byte-aligned struct so the prefix scan over per-molecule sizes
 *  is one pass over a single @c std::vector<per_mol_sizes_t>, not fourteen parallel
 *  push_back chains.  The @c valid field doubles as the dense-output compaction
 *  counter: after the scan, @c offsets[m].valid equals the count of valid molecules
 *  with index < m, i.e. the slot where metadata/input_indices for molecule m land.
 */
struct alignas(64) per_mol_sizes_t {
    std::uint32_t bounds;
    std::uint32_t chirals;
    std::uint32_t torsions;
    std::uint32_t dcons;
    std::uint32_t mmff_atom_vdw;    ///< = num_atoms for valid mols
    std::uint32_t mmff_atom_charge; ///< = num_atoms for valid mols
    std::uint32_t bond_distance;    ///< = num_atoms² for valid mols
    std::uint32_t mmff_bonds;
    std::uint32_t mmff_angles;
    std::uint32_t mmff_torsions;
    std::uint32_t mmff_oops;
    std::uint32_t mmff_stretchbends;
    std::uint32_t mmff_interactions;
    std::uint32_t mmff_atom_offsets;
    std::uint32_t etk_interactions;
    std::uint32_t etk_atom_offsets;
    std::uint32_t valid; ///< 0 if the molecule failed preprocessing, 1 otherwise
    std::uint32_t padding[3];
};
static_assert(sizeof(per_mol_sizes_t) == 128, "per_mol_sizes_t must be two cache lines");

/** Field-wise sum used both by the prefix scan and as the identity-additive operation. */
constexpr per_mol_sizes_t operator+(per_mol_sizes_t a, per_mol_sizes_t const &b) {
    a.bounds += b.bounds;
    a.chirals += b.chirals;
    a.torsions += b.torsions;
    a.dcons += b.dcons;
    a.mmff_atom_vdw += b.mmff_atom_vdw;
    a.mmff_atom_charge += b.mmff_atom_charge;
    a.bond_distance += b.bond_distance;
    a.mmff_bonds += b.mmff_bonds;
    a.mmff_angles += b.mmff_angles;
    a.mmff_torsions += b.mmff_torsions;
    a.mmff_oops += b.mmff_oops;
    a.mmff_stretchbends += b.mmff_stretchbends;
    a.mmff_interactions += b.mmff_interactions;
    a.mmff_atom_offsets += b.mmff_atom_offsets;
    a.etk_interactions += b.etk_interactions;
    a.etk_atom_offsets += b.etk_atom_offsets;
    a.valid += b.valid;
    return a;
}

/** Extract per-molecule output sizes; returns {} (all zeros, valid=0) for failed mols. */
inline per_mol_sizes_t extract_sizes(etkdg_prepared_molecule_t const &p) {
    if (p.num_atoms == 0) return per_mol_sizes_t {};
    per_mol_sizes_t s {};
    s.bounds = static_cast<std::uint32_t>(p.bounds_matrix.size());
    s.chirals = static_cast<std::uint32_t>(p.chiral_centers.size());
    s.torsions = static_cast<std::uint32_t>(p.etk_torsions.size());
    s.dcons = static_cast<std::uint32_t>(p.distance_constraints.size());
    s.mmff_atom_vdw = static_cast<std::uint32_t>(p.mmff_atom_vdw.size());
    s.mmff_atom_charge = static_cast<std::uint32_t>(p.mmff_atom_charge.size());
    s.bond_distance = p.num_atoms * p.num_atoms;
    s.mmff_bonds = static_cast<std::uint32_t>(p.mmff_bonds.size());
    s.mmff_angles = static_cast<std::uint32_t>(p.mmff_angles.size());
    s.mmff_torsions = static_cast<std::uint32_t>(p.mmff_torsions.size());
    s.mmff_oops = static_cast<std::uint32_t>(p.mmff_oops.size());
    s.mmff_stretchbends = static_cast<std::uint32_t>(p.mmff_stretchbends.size());
    s.mmff_interactions = static_cast<std::uint32_t>(p.mmff_interactions.size());
    s.mmff_atom_offsets = static_cast<std::uint32_t>(p.mmff_atom_offsets.size());
    s.etk_interactions = static_cast<std::uint32_t>(p.etk_interactions.size());
    s.etk_atom_offsets = static_cast<std::uint32_t>(p.etk_atom_offsets.size());
    s.valid = 1;
    return s;
}

/**
 *  Copy @p src into the slice of @p dst that starts at @p offset.  Uses std::memcpy
 *  because every output type is trivially copyable POD; the static_assert keeps us
 *  honest if that ever changes.
 */
template <typename element_>
inline void append_slice(std::span<element_> dst, std::uint32_t offset, std::vector<element_> const &src) {
    static_assert(std::is_trivially_copyable_v<element_>, "append_slice requires trivially-copyable elements");
    if (!src.empty()) std::memcpy(dst.data() + offset, src.data(), src.size() * sizeof(element_));
}

/** Build the per-molecule metadata entry with batch-wide offsets baked in. */
inline etkdg_molecule_metadata_t make_meta(etkdg_molecule_metadata_t base, per_mol_sizes_t const &o) {
    base.bounds_offset = o.bounds;
    base.chiral_offset = o.chirals;
    base.torsion_offset = o.torsions;
    base.constraint_offset = o.dcons;
    base.mmff_atom_vdw_offset = o.mmff_atom_vdw;
    base.mmff_atom_charge_offset = o.mmff_atom_charge;
    base.bond_distance_offset = o.bond_distance;
    base.mmff_bond_offset = o.mmff_bonds;
    base.mmff_angle_offset = o.mmff_angles;
    base.mmff_torsion_offset = o.mmff_torsions;
    base.mmff_oop_offset = o.mmff_oops;
    base.mmff_stretchbend_offset = o.mmff_stretchbends;
    base.mmff_atom_offsets_start = o.mmff_atom_offsets;
    base.etk_interaction_offset = o.etk_interactions;
    base.etk_atom_offsets_start = o.etk_atom_offsets;
    return base;
}

/**
 *  Derive the `batch_layout_t` from total per-region counts.  Every region's
 *  byte offset is aligned to `batch_layout_t::region_alignment` so the GPU
 *  pool slicing in `launch_gpu_pipeline` inherits correct alignment for free.
 *
 *  Regions are placed in a fixed order; the order itself is an implementation
 *  detail — consumers only read `offset_*` / `count_*` pairs — with one exception.  The
 *  hop-count region is placed last, past `bytes_uploaded`, because a warp BFS fills it on
 *  the device from the bond list; it exists in the device pool only and never crosses PCIe.
 */
inline batch_layout_t compute_batch_layout(per_mol_sizes_t const &totals, std::uint32_t n_valid) {
    batch_layout_t layout {};
    std::size_t running = 0;

    layout.offset_bounds = place_region_<float>(running, totals.bounds);
    layout.count_bounds = totals.bounds;
    layout.offset_chirals = place_region_<chiral_center_descriptor_t>(running, totals.chirals);
    layout.count_chirals = totals.chirals;
    layout.offset_torsions = place_region_<etk_torsion_descriptor_t>(running, totals.torsions);
    layout.count_torsions = totals.torsions;
    layout.offset_distance_cons = place_region_<distance_constraint_t>(running, totals.dcons);
    layout.count_distance_cons = totals.dcons;
    layout.offset_mmff_atom_vdw = place_region_<mmff94_atom_vdw_params_t>(running, totals.mmff_atom_vdw);
    layout.count_mmff_atom_vdw = totals.mmff_atom_vdw;
    layout.offset_mmff_atom_charge = place_region_<float>(running, totals.mmff_atom_charge);
    layout.count_mmff_atom_charge = totals.mmff_atom_charge;
    layout.offset_mmff_bonds = place_region_<mmff94_bond_t>(running, totals.mmff_bonds);
    layout.count_mmff_bonds = totals.mmff_bonds;
    layout.offset_mmff_angles = place_region_<mmff94_angle_t>(running, totals.mmff_angles);
    layout.count_mmff_angles = totals.mmff_angles;
    layout.offset_mmff_torsions = place_region_<mmff94_torsion_t>(running, totals.mmff_torsions);
    layout.count_mmff_torsions = totals.mmff_torsions;
    layout.offset_mmff_oops = place_region_<mmff94_oop_t>(running, totals.mmff_oops);
    layout.count_mmff_oops = totals.mmff_oops;
    layout.offset_mmff_stretchbends = place_region_<mmff94_stretchbend_t>(running, totals.mmff_stretchbends);
    layout.count_mmff_stretchbends = totals.mmff_stretchbends;
    layout.offset_mmff_interactions = place_region_<mmff94_atom_interaction_t>(running, totals.mmff_interactions);
    layout.count_mmff_interactions = totals.mmff_interactions;
    layout.offset_mmff_atom_offsets = place_region_<mmff94_atom_offsets_t>(running, totals.mmff_atom_offsets);
    layout.count_mmff_atom_offsets = totals.mmff_atom_offsets;
    layout.offset_etk_interactions = place_region_<etk_atom_interaction_t>(running, totals.etk_interactions);
    layout.count_etk_interactions = totals.etk_interactions;
    layout.offset_etk_atom_offsets = place_region_<etk_atom_offsets_t>(running, totals.etk_atom_offsets);
    layout.count_etk_atom_offsets = totals.etk_atom_offsets;
    layout.offset_metadata = place_region_<etkdg_molecule_metadata_t>(running, n_valid);
    layout.count_metadata = n_valid;
    layout.offset_input_indices = place_region_<std::uint32_t>(running, n_valid);
    layout.count_input_indices = n_valid;

    layout.bytes_uploaded = running;
    layout.offset_bond_distance = place_region_<std::uint8_t>(running, totals.bond_distance);
    layout.count_bond_distance = totals.bond_distance;
    layout.bytes_total = running;
    return layout;
}

/**
 *  @brief Prepare a batch of molecules from SMILES strings.
 *
 *  Pipeline: an OpenMP-parallel per-molecule prep, followed by a three-pass sequence — extract sizes
 *  in parallel, prefix-scan the struct-of-sizes serially for cache locality, then append slices in
 *  parallel. No push_back or insert calls anywhere on the hot path.
 *
 *  @param[in] smiles_list Vector of SMILES strings to process.
 *  @return Batched molecule data with contiguous arrays for GPU upload.
 */
/**
 *  In-place variant: fills @p batch in place, reusing its pinned host buffer
 *  across calls.  This is the lever that kills pinned-memory allocator
 *  contention at 8-GPU scale — `cudaHostAlloc`/`cudaFreeHost` serialize on a
 *  driver lock, so recycling the buffer across chunks matters a lot more than
 *  you'd expect.
 */
/**
 *  @brief One per-OpenMP-thread record accumulated during `prepare_per_thread_contribs_`.
 *  `*_offset` fields hold THREAD-LOCAL byte indices; the pack-pass rebases
 *  them by adding the thread's pool-wide base.
 */
struct batch_prep_per_mol_record_t {
    std::uint32_t original_idx;
    etkdg_molecule_metadata_t metadata;
};

/**
 *  @brief Per-OpenMP-thread contribution buffer for the three-phase
 *  `preprocess-step` of `prepare_molecule_batch_into`.
 *
 *  One instance per thread.  The per-region `*_bytes` accumulators hold the
 *  concatenation of every molecule that thread processed, in append order.
 *  The thread's `mols` records carry per-molecule metadata with thread-local
 *  offsets.  Persisted across calls via a file-scope `static` vector so the
 *  `std::vector` capacities survive and the allocator's fast path handles
 *  subsequent calls after the first embedding-wave warmup.
 */
struct alignas(64) batch_prep_thread_contribution_t {
    etkdg_prepared_molecule_t scratch;
    std::vector<std::byte> bounds_bytes;
    std::vector<std::byte> chirals_bytes;
    std::vector<std::byte> torsions_bytes;
    std::vector<std::byte> dcons_bytes;
    std::vector<std::byte> atom_vdw_bytes;
    std::vector<std::byte> atom_charge_bytes;
    std::vector<std::byte> mmff_bonds_bytes;
    std::vector<std::byte> mmff_angles_bytes;
    std::vector<std::byte> mmff_torsions_bytes;
    std::vector<std::byte> mmff_oops_bytes;
    std::vector<std::byte> mmff_stretchbends_bytes;
    std::vector<std::byte> mmff_interactions_bytes;
    std::vector<std::byte> mmff_atom_offsets_bytes;
    std::vector<std::byte> etk_interactions_bytes;
    std::vector<std::byte> etk_atom_offsets_bytes;
    /** @brief Running element count of this thread's share of the hop-count region.  The bytes
     *  themselves are never staged: the region is device-only and a warp BFS fills it after the
     *  upload, so the thread accumulates only the size its molecules claim. */
    std::size_t bond_distance_count = 0;
    std::vector<batch_prep_per_mol_record_t> mols;
    /** @brief Slice-local input indices for every molecule rejected by this
     *  thread's slice of `prepare_per_thread_contribs_`, each tagged with
     *  the rejection reason (`invalid_smiles`, `too_many_atoms`,
     *  `unsupported_element`, `preprocess_timeout`).  Aggregated upstream
     *  into `batch_pool_t::rejected_input_indices`. */
    std::vector<smallfield::api::rejected_mol_t> rejected_input_indices;
    std::uint32_t max_atoms_seen = 0;
};

/** @brief Per-thread pool-wide base offsets computed by `scan_per_region_offsets_`. */
struct batch_prep_thread_bases_t {
    std::size_t bounds, chirals, torsions, dcons;
    std::size_t atom_vdw, atom_charge, bond_distance;
    std::size_t mmff_bonds, mmff_angles, mmff_torsions, mmff_oops, mmff_stretchbends;
    std::size_t mmff_interactions, mmff_atom_offsets;
    std::size_t etk_interactions, etk_atom_offsets;
    std::size_t dense_slot;
};

/** @brief Append `bytes` bytes from `src` to the back of `dst`; return the pre-append offset. */
static inline std::size_t append_bytes_(std::vector<std::byte> &dst, void const *src, std::size_t bytes) {
    std::size_t const offset = dst.size();
    if (bytes > 0) {
        dst.resize(offset + bytes);
        std::memcpy(dst.data() + offset, src, bytes);
    }
    return offset;
}

/**
 *  @brief Phase 1 of the preprocess-step: parallel per-molecule RDKit parse,
 *  MMFF typing, and byte-append into each OpenMP thread's contribution buffer.
 *
 *  Resizes `contribs` to `omp_get_max_threads()` and clears prior contents
 *  while preserving the inner `std::vector` capacities.  After return, each
 *  thread's contribution carries its share of the input molecules with
 *  thread-local metadata offsets.
 */
/**
 *  @brief Force every RDKit lazy-init singleton to initialise on the main
 *  thread before an OMP parallel-for races on first access.
 *
 *  Touches the SMILES parser, addHs, `assignStereochemistry` (which pulls in
 *  the CIP-rank global state), the MMFF94 parameter tables, and an
 *  ETKDG-style substructure match path.  These all have process-wide
 *  lazy-loaded caches that in some RDKit versions are not safe for
 *  concurrent first-time access.  `std::call_once` guarantees the warm-up
 *  runs exactly once per process regardless of how many call sites reach
 *  here; subsequent calls on any thread are a relaxed-load no-op.
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
                RDKit::MMFF::MMFFMolProperties props(*warm);
                (void)props.isValid();
            }
        }
        catch (...) {
            // Warm-up is best-effort.  If it throws, the subsequent parallel-for
            // call will hit the same cache miss path; it only costs us the
            // race-protection benefit, not correctness.
        }
    });
}

static inline void prepare_per_thread_contribs_(std::span<std::string_view const> smiles_list,
                                                std::vector<batch_prep_thread_contribution_t> &contribs) {
    warm_up_rdkit_globals_once_();
    int const n_threads_max = omp_get_max_threads();
    if (static_cast<int>(contribs.size()) < n_threads_max) contribs.resize(n_threads_max);
    for (auto &type_counts : contribs) {
        type_counts.bounds_bytes.clear();
        type_counts.chirals_bytes.clear();
        type_counts.torsions_bytes.clear();
        type_counts.dcons_bytes.clear();
        type_counts.atom_vdw_bytes.clear();
        type_counts.atom_charge_bytes.clear();
        type_counts.mmff_bonds_bytes.clear();
        type_counts.mmff_angles_bytes.clear();
        type_counts.mmff_torsions_bytes.clear();
        type_counts.mmff_oops_bytes.clear();
        type_counts.mmff_stretchbends_bytes.clear();
        type_counts.mmff_interactions_bytes.clear();
        type_counts.mmff_atom_offsets_bytes.clear();
        type_counts.etk_interactions_bytes.clear();
        type_counts.etk_atom_offsets_bytes.clear();
        type_counts.mols.clear();
        type_counts.rejected_input_indices.clear();
        type_counts.max_atoms_seen = 0;
        type_counts.bond_distance_count = 0;
    }

    std::size_t const n = smiles_list.size();
    // CRITICAL: this file is included by `smallfield.cu` which is compiled with nvcc.
    // For OpenMP to actually parallelize here, the CMake target must pass
    // `-Xcompiler=-fopenmp` through nvcc to the host compiler.  Otherwise nvcc
    // silently ignores the pragma and this loop runs single-threaded — 20× slowdown.
#pragma omp parallel
    {
        int const tid = omp_get_thread_num();
        auto &type_counts = contribs[tid];
#pragma omp for schedule(dynamic, 64)
        for (std::size_t m = 0; m < n; ++m) {
            auto const status = prepare_molecule_into(smiles_list[m], type_counts.scratch);
            auto const &p = type_counts.scratch;
            if (p.num_atoms == 0) {
                // `prepare_molecule_into` reports a precise rejection reason
                // (invalid_smiles, too_many_atoms, unsupported_element, ...)
                // — propagate it so `generate_conformers` stamps the right
                // status instead of the default-init `invalid_smiles`.
                type_counts.rejected_input_indices.push_back({static_cast<std::uint32_t>(m), status});
                continue;
            }
            if (p.num_atoms > type_counts.max_atoms_seen) type_counts.max_atoms_seen = p.num_atoms;

            batch_prep_per_mol_record_t rec;
            rec.original_idx = static_cast<std::uint32_t>(m);
            rec.metadata = p.metadata;
            rec.metadata.bounds_offset = static_cast<std::uint32_t>(
                append_bytes_(type_counts.bounds_bytes, p.bounds_matrix.data(),
                              p.bounds_matrix.size() * sizeof(float)) /
                sizeof(float));
            rec.metadata.chiral_offset = static_cast<std::uint32_t>(
                append_bytes_(type_counts.chirals_bytes, p.chiral_centers.data(),
                              p.chiral_centers.size() * sizeof(chiral_center_descriptor_t)) /
                sizeof(chiral_center_descriptor_t));
            rec.metadata.torsion_offset = static_cast<std::uint32_t>(
                append_bytes_(type_counts.torsions_bytes, p.etk_torsions.data(),
                              p.etk_torsions.size() * sizeof(etk_torsion_descriptor_t)) /
                sizeof(etk_torsion_descriptor_t));
            rec.metadata.constraint_offset = static_cast<std::uint32_t>(
                append_bytes_(type_counts.dcons_bytes, p.distance_constraints.data(),
                              p.distance_constraints.size() * sizeof(distance_constraint_t)) /
                sizeof(distance_constraint_t));
            rec.metadata.mmff_atom_vdw_offset = static_cast<std::uint32_t>(
                append_bytes_(type_counts.atom_vdw_bytes, p.mmff_atom_vdw.data(),
                              p.mmff_atom_vdw.size() * sizeof(mmff94_atom_vdw_params_t)) /
                sizeof(mmff94_atom_vdw_params_t));
            rec.metadata.mmff_atom_charge_offset = static_cast<std::uint32_t>(
                append_bytes_(type_counts.atom_charge_bytes, p.mmff_atom_charge.data(),
                              p.mmff_atom_charge.size() * sizeof(float)) /
                sizeof(float));
            rec.metadata.bond_distance_offset = static_cast<std::uint32_t>(type_counts.bond_distance_count);
            type_counts.bond_distance_count += static_cast<std::size_t>(p.num_atoms) * p.num_atoms;
            rec.metadata.mmff_bond_offset = static_cast<std::uint32_t>(
                append_bytes_(type_counts.mmff_bonds_bytes, p.mmff_bonds.data(),
                              p.mmff_bonds.size() * sizeof(mmff94_bond_t)) /
                sizeof(mmff94_bond_t));
            rec.metadata.mmff_angle_offset = static_cast<std::uint32_t>(
                append_bytes_(type_counts.mmff_angles_bytes, p.mmff_angles.data(),
                              p.mmff_angles.size() * sizeof(mmff94_angle_t)) /
                sizeof(mmff94_angle_t));
            rec.metadata.mmff_torsion_offset = static_cast<std::uint32_t>(
                append_bytes_(type_counts.mmff_torsions_bytes, p.mmff_torsions.data(),
                              p.mmff_torsions.size() * sizeof(mmff94_torsion_t)) /
                sizeof(mmff94_torsion_t));
            rec.metadata.mmff_oop_offset = static_cast<std::uint32_t>(
                append_bytes_(type_counts.mmff_oops_bytes, p.mmff_oops.data(),
                              p.mmff_oops.size() * sizeof(mmff94_oop_t)) /
                sizeof(mmff94_oop_t));
            rec.metadata.mmff_stretchbend_offset = static_cast<std::uint32_t>(
                append_bytes_(type_counts.mmff_stretchbends_bytes, p.mmff_stretchbends.data(),
                              p.mmff_stretchbends.size() * sizeof(mmff94_stretchbend_t)) /
                sizeof(mmff94_stretchbend_t));
            // Thread-local start of this mol's block inside mmff_interactions / etk_interactions.
            std::uint32_t const tl_mmff_inter_base = static_cast<std::uint32_t>(
                append_bytes_(type_counts.mmff_interactions_bytes, p.mmff_interactions.data(),
                              p.mmff_interactions.size() * sizeof(mmff94_atom_interaction_t)) /
                sizeof(mmff94_atom_interaction_t));
            rec.metadata.mmff_atom_offsets_start = static_cast<std::uint32_t>(
                append_bytes_(type_counts.mmff_atom_offsets_bytes, p.mmff_atom_offsets.data(),
                              p.mmff_atom_offsets.size() * sizeof(mmff94_atom_offsets_t)) /
                sizeof(mmff94_atom_offsets_t));
            std::uint32_t const tl_etk_inter_base = static_cast<std::uint32_t>(
                append_bytes_(type_counts.etk_interactions_bytes, p.etk_interactions.data(),
                              p.etk_interactions.size() * sizeof(etk_atom_interaction_t)) /
                sizeof(etk_atom_interaction_t));
            rec.metadata.etk_atom_offsets_start = static_cast<std::uint32_t>(
                append_bytes_(type_counts.etk_atom_offsets_bytes, p.etk_atom_offsets.data(),
                              p.etk_atom_offsets.size() * sizeof(etk_atom_offsets_t)) /
                sizeof(etk_atom_offsets_t));
            // Rebase the per-atom offsets from per-molecule-local to thread-local by adding this
            // molecule's thread-local base in the interactions array. The downstream pool-wide pass
            // then adds the pool-wide base.
            std::size_t const aoff_byte = rec.metadata.mmff_atom_offsets_start * sizeof(mmff94_atom_offsets_t);
            auto *aoff_ptr = reinterpret_cast<mmff94_atom_offsets_t *>(type_counts.mmff_atom_offsets_bytes.data() +
                                                                       aoff_byte);
            for (std::size_t k = 0; k < p.mmff_atom_offsets.size(); ++k) {
                aoff_ptr[k].bond_start += tl_mmff_inter_base;
                aoff_ptr[k].angle_start += tl_mmff_inter_base;
                aoff_ptr[k].torsion_start += tl_mmff_inter_base;
                aoff_ptr[k].oop_start += tl_mmff_inter_base;
                aoff_ptr[k].stretchbend_start += tl_mmff_inter_base;
            }
            // rec.metadata.etk_interaction_offset isn't a real field on the metadata;
            // etk_atom_offsets entries carry the interaction_start which we rebase now.
            std::size_t const etk_aoff_byte = rec.metadata.etk_atom_offsets_start * sizeof(etk_atom_offsets_t);
            auto *etk_aoff_ptr = reinterpret_cast<etk_atom_offsets_t *>(type_counts.etk_atom_offsets_bytes.data() +
                                                                        etk_aoff_byte);
            for (std::size_t k = 0; k < p.etk_atom_offsets.size(); ++k)
                etk_aoff_ptr[k].interaction_start += tl_etk_inter_base;
            rec.metadata.etk_interaction_offset = tl_etk_inter_base;

            type_counts.mols.push_back(rec);
        }
    }
}

/**
 *  @brief Phase 2 of the preprocess-step: serial prefix-scan over per-thread
 *  contribution sizes to compute each thread's pool-wide base offset per
 *  region, plus the totals that drive the batch-pool layout.
 *
 *  `bases` is resized to match `contribs.size()`.
 */
static inline void scan_per_region_offsets_(std::vector<batch_prep_thread_contribution_t> const &contribs,
                                            std::vector<batch_prep_thread_bases_t> &bases, per_mol_sizes_t &totals,
                                            std::uint32_t &n_valid, std::uint32_t &max_atoms_reduction) {
    int const n_threads_max = static_cast<int>(contribs.size());
    bases.assign(n_threads_max, batch_prep_thread_bases_t {});
    totals = per_mol_sizes_t {};
    n_valid = 0;
    max_atoms_reduction = 0;
    for (int t = 0; t < n_threads_max; ++t) {
        auto const &type_counts = contribs[t];
        bases[t].bounds = totals.bounds;
        totals.bounds += type_counts.bounds_bytes.size() / sizeof(float);
        bases[t].chirals = totals.chirals;
        totals.chirals += type_counts.chirals_bytes.size() / sizeof(chiral_center_descriptor_t);
        bases[t].torsions = totals.torsions;
        totals.torsions += type_counts.torsions_bytes.size() / sizeof(etk_torsion_descriptor_t);
        bases[t].dcons = totals.dcons;
        totals.dcons += type_counts.dcons_bytes.size() / sizeof(distance_constraint_t);
        bases[t].atom_vdw = totals.mmff_atom_vdw;
        totals.mmff_atom_vdw += type_counts.atom_vdw_bytes.size() / sizeof(mmff94_atom_vdw_params_t);
        bases[t].atom_charge = totals.mmff_atom_charge;
        totals.mmff_atom_charge += type_counts.atom_charge_bytes.size() / sizeof(float);
        bases[t].bond_distance = totals.bond_distance;
        totals.bond_distance += type_counts.bond_distance_count;
        bases[t].mmff_bonds = totals.mmff_bonds;
        totals.mmff_bonds += type_counts.mmff_bonds_bytes.size() / sizeof(mmff94_bond_t);
        bases[t].mmff_angles = totals.mmff_angles;
        totals.mmff_angles += type_counts.mmff_angles_bytes.size() / sizeof(mmff94_angle_t);
        bases[t].mmff_torsions = totals.mmff_torsions;
        totals.mmff_torsions += type_counts.mmff_torsions_bytes.size() / sizeof(mmff94_torsion_t);
        bases[t].mmff_oops = totals.mmff_oops;
        totals.mmff_oops += type_counts.mmff_oops_bytes.size() / sizeof(mmff94_oop_t);
        bases[t].mmff_stretchbends = totals.mmff_stretchbends;
        totals.mmff_stretchbends += type_counts.mmff_stretchbends_bytes.size() / sizeof(mmff94_stretchbend_t);
        bases[t].mmff_interactions = totals.mmff_interactions;
        totals.mmff_interactions += type_counts.mmff_interactions_bytes.size() / sizeof(mmff94_atom_interaction_t);
        bases[t].mmff_atom_offsets = totals.mmff_atom_offsets;
        totals.mmff_atom_offsets += type_counts.mmff_atom_offsets_bytes.size() / sizeof(mmff94_atom_offsets_t);
        bases[t].etk_interactions = totals.etk_interactions;
        totals.etk_interactions += type_counts.etk_interactions_bytes.size() / sizeof(etk_atom_interaction_t);
        bases[t].etk_atom_offsets = totals.etk_atom_offsets;
        totals.etk_atom_offsets += type_counts.etk_atom_offsets_bytes.size() / sizeof(etk_atom_offsets_t);
        bases[t].dense_slot = n_valid;
        n_valid += static_cast<std::uint32_t>(type_counts.mols.size());
        if (type_counts.max_atoms_seen > max_atoms_reduction) max_atoms_reduction = type_counts.max_atoms_seen;
    }
    totals.valid = n_valid;
}

/**
 *  @brief Phase 3 of the preprocess-step: parallel per-thread memcpy of every
 *  byte-accumulator region into the batch's pinned pool, plus metadata rebase
 *  from thread-local offsets to pool-wide offsets.
 *
 *  @pre `batch.host.ensure_capacity(batch.layout.bytes_uploaded)` has succeeded;
 *       `batch.num_molecules`, `batch.max_atoms_per_molecule`, and
 *       `batch.layout` are already populated by the caller.
 */
static inline void pack_into_batch_pool_(std::vector<batch_prep_thread_contribution_t> const &contribs,
                                         std::vector<batch_prep_thread_bases_t> const &bases,
                                         etkdg_molecule_batch_t &batch) {
    int const n_threads_max = static_cast<int>(contribs.size());
    auto bounds_span = batch.bounds_matrices();
    auto chirals_span = batch.chiral_centers();
    auto torsions_span = batch.etk_torsions();
    auto dcons_span = batch.distance_constraints();
    auto mmff_vdw_span = batch.mmff_atom_vdw();
    auto mmff_charge_span = batch.mmff_atom_charge();
    auto mmff_bonds_span = batch.mmff_bonds();
    auto mmff_angles_span = batch.mmff_angles();
    auto mmff_tors_span = batch.mmff_torsions();
    auto mmff_oops_span = batch.mmff_oops();
    auto mmff_sb_span = batch.mmff_stretchbends();
    auto mmff_inter_span = batch.mmff_interactions();
    auto mmff_aoff_span = batch.mmff_atom_offsets();
    auto etk_inter_span = batch.etk_interactions();
    auto etk_aoff_span = batch.etk_atom_offsets();
    auto metadata_span = batch.molecule_metadata();
    auto input_idx_span = batch.input_indices();

    // Batched memcpy pass: parallel per-thread memcpy + metadata rebase.
#pragma omp parallel for schedule(static)
    for (int t = 0; t < n_threads_max; ++t) {
        auto const &type_counts = contribs[t];
        auto const &b = bases[t];
        if (!type_counts.bounds_bytes.empty())
            std::memcpy(bounds_span.data() + b.bounds, type_counts.bounds_bytes.data(),
                        type_counts.bounds_bytes.size());
        if (!type_counts.chirals_bytes.empty())
            std::memcpy(chirals_span.data() + b.chirals, type_counts.chirals_bytes.data(),
                        type_counts.chirals_bytes.size());
        if (!type_counts.torsions_bytes.empty())
            std::memcpy(torsions_span.data() + b.torsions, type_counts.torsions_bytes.data(),
                        type_counts.torsions_bytes.size());
        if (!type_counts.dcons_bytes.empty())
            std::memcpy(dcons_span.data() + b.dcons, type_counts.dcons_bytes.data(), type_counts.dcons_bytes.size());
        if (!type_counts.atom_vdw_bytes.empty())
            std::memcpy(mmff_vdw_span.data() + b.atom_vdw, type_counts.atom_vdw_bytes.data(),
                        type_counts.atom_vdw_bytes.size());
        if (!type_counts.atom_charge_bytes.empty())
            std::memcpy(mmff_charge_span.data() + b.atom_charge, type_counts.atom_charge_bytes.data(),
                        type_counts.atom_charge_bytes.size());
        if (!type_counts.mmff_bonds_bytes.empty())
            std::memcpy(mmff_bonds_span.data() + b.mmff_bonds, type_counts.mmff_bonds_bytes.data(),
                        type_counts.mmff_bonds_bytes.size());
        if (!type_counts.mmff_angles_bytes.empty())
            std::memcpy(mmff_angles_span.data() + b.mmff_angles, type_counts.mmff_angles_bytes.data(),
                        type_counts.mmff_angles_bytes.size());
        if (!type_counts.mmff_torsions_bytes.empty())
            std::memcpy(mmff_tors_span.data() + b.mmff_torsions, type_counts.mmff_torsions_bytes.data(),
                        type_counts.mmff_torsions_bytes.size());
        if (!type_counts.mmff_oops_bytes.empty())
            std::memcpy(mmff_oops_span.data() + b.mmff_oops, type_counts.mmff_oops_bytes.data(),
                        type_counts.mmff_oops_bytes.size());
        if (!type_counts.mmff_stretchbends_bytes.empty())
            std::memcpy(mmff_sb_span.data() + b.mmff_stretchbends, type_counts.mmff_stretchbends_bytes.data(),
                        type_counts.mmff_stretchbends_bytes.size());
        if (!type_counts.mmff_interactions_bytes.empty())
            std::memcpy(mmff_inter_span.data() + b.mmff_interactions, type_counts.mmff_interactions_bytes.data(),
                        type_counts.mmff_interactions_bytes.size());
        // mmff_atom_offsets: rebase thread-local interaction offsets by the thread's
        // pool-wide mmff_interactions base, then write to pool.
        std::size_t const n_aoff = type_counts.mmff_atom_offsets_bytes.size() / sizeof(mmff94_atom_offsets_t);
        auto const *src_aoff = reinterpret_cast<mmff94_atom_offsets_t const *>(
            type_counts.mmff_atom_offsets_bytes.data());
        for (std::size_t k = 0; k < n_aoff; ++k) {
            mmff94_atom_offsets_t entry = src_aoff[k];
            entry.bond_start += b.mmff_interactions;
            entry.angle_start += b.mmff_interactions;
            entry.torsion_start += b.mmff_interactions;
            entry.oop_start += b.mmff_interactions;
            entry.stretchbend_start += b.mmff_interactions;
            mmff_aoff_span[b.mmff_atom_offsets + k] = entry;
        }
        if (!type_counts.etk_interactions_bytes.empty())
            std::memcpy(etk_inter_span.data() + b.etk_interactions, type_counts.etk_interactions_bytes.data(),
                        type_counts.etk_interactions_bytes.size());
        std::size_t const n_etk_aoff = type_counts.etk_atom_offsets_bytes.size() / sizeof(etk_atom_offsets_t);
        auto const *src_etk_aoff = reinterpret_cast<etk_atom_offsets_t const *>(
            type_counts.etk_atom_offsets_bytes.data());
        for (std::size_t k = 0; k < n_etk_aoff; ++k) {
            etk_atom_offsets_t entry = src_etk_aoff[k];
            entry.interaction_start += b.etk_interactions;
            etk_aoff_span[b.etk_atom_offsets + k] = entry;
        }

        // Per-molecule metadata: rebase every region offset by the thread's pool base.
        for (std::size_t k = 0; k < type_counts.mols.size(); ++k) {
            auto const &rec = type_counts.mols[k];
            etkdg_molecule_metadata_t meta = rec.metadata;
            meta.bounds_offset += static_cast<std::uint32_t>(b.bounds);
            meta.chiral_offset += static_cast<std::uint32_t>(b.chirals);
            meta.torsion_offset += static_cast<std::uint32_t>(b.torsions);
            meta.constraint_offset += static_cast<std::uint32_t>(b.dcons);
            meta.mmff_atom_vdw_offset += static_cast<std::uint32_t>(b.atom_vdw);
            meta.mmff_atom_charge_offset += static_cast<std::uint32_t>(b.atom_charge);
            meta.bond_distance_offset += static_cast<std::uint32_t>(b.bond_distance);
            meta.mmff_bond_offset += static_cast<std::uint32_t>(b.mmff_bonds);
            meta.mmff_angle_offset += static_cast<std::uint32_t>(b.mmff_angles);
            meta.mmff_torsion_offset += static_cast<std::uint32_t>(b.mmff_torsions);
            meta.mmff_oop_offset += static_cast<std::uint32_t>(b.mmff_oops);
            meta.mmff_stretchbend_offset += static_cast<std::uint32_t>(b.mmff_stretchbends);
            meta.mmff_atom_offsets_start += static_cast<std::uint32_t>(b.mmff_atom_offsets);
            meta.etk_interaction_offset += static_cast<std::uint32_t>(b.etk_interactions);
            meta.etk_atom_offsets_start += static_cast<std::uint32_t>(b.etk_atom_offsets);
            std::size_t const slot = b.dense_slot + k;
            metadata_span[slot] = meta;
            input_idx_span[slot] = rec.original_idx;
        }
    }
}

/**
 *  @brief In-place `preprocess-step` driver: parses every SMILES, extracts
 *  ETKDG + MMFF preprocess data, and packs into @p batch's pinned host pool.
 *
 *  Three phases chained via helpers (see `GLOSSARY.md` for terminology):
 *    1. `prepare_per_thread_contribs_` — parallel per-molecule extraction.
 *    2. `scan_per_region_offsets_` — serial prefix scan over per-thread sizes.
 *    3. `pack_into_batch_pool_` — parallel memcpy + metadata rebase.
 *
 *  This is the lever that kills pinned-memory allocator contention at
 *  8-GPU scale — `cudaHostAlloc`/`cudaFreeHost` serialize on a driver lock,
 *  so recycling the buffer across `embedding-waves` matters a lot.  The
 *  `static` contribs vector persists capacities across calls so the
 *  allocator's fast path handles every call after the first.
 */
inline void prepare_molecule_batch_into(std::span<std::string_view const> smiles_list, etkdg_molecule_batch_t &batch) {
    batch.num_molecules = 0;
    batch.max_atoms_per_molecule = 0;
    batch.layout = batch_layout_t {};

    // Persistent per-call scratch.  Safe as long as this function is not
    // called concurrently from multiple outer threads.  The CLI's 4-role
    // pipeline guarantees this (preprocess-step runs only on the
    // gpu-runner-role thread).  A future streaming submit API that drives
    // prep concurrently would need a caller-owned scratch.
    static std::vector<batch_prep_thread_contribution_t> contribs;
    prepare_per_thread_contribs_(smiles_list, contribs);

    std::vector<batch_prep_thread_bases_t> bases;
    per_mol_sizes_t totals {};
    std::uint32_t n_valid = 0;
    std::uint32_t max_atoms_reduction = 0;
    scan_per_region_offsets_(contribs, bases, totals, n_valid, max_atoms_reduction);

    batch.num_molecules = n_valid;
    batch.max_atoms_per_molecule = max_atoms_reduction;
    batch.layout = compute_batch_layout(totals, n_valid);
    if (batch.layout.bytes_uploaded > 0 && !batch.host.ensure_capacity(batch.layout.bytes_uploaded)) {
        batch.num_molecules = 0;
        batch.layout = batch_layout_t {};
        return;
    }

    // Merge every thread's rejected-input indices into the batch's flat list.
    // Each entry carries the slice-local index plus the `conformer_status_t`
    // reason code; the caller stamps `entry.reason` directly into
    // `status_out`.  Small in the common case (<100 per 1M molecules) but
    // grow-only across calls.
    batch.rejected_input_indices.clear();
    for (auto const &type_counts : contribs)
        batch.rejected_input_indices.insert(batch.rejected_input_indices.end(),
                                            type_counts.rejected_input_indices.begin(),
                                            type_counts.rejected_input_indices.end());

    pack_into_batch_pool_(contribs, bases, batch);
}

/**
 *  Convenience wrapper that returns a fresh batch (allocates a new pinned
 *  buffer every call).  Kept for call sites that don't participate in the
 *  pipelined dispatcher — tests, one-shot invocations.  The hot path should
 *  prefer @ref prepare_molecule_batch_into.
 */
inline etkdg_molecule_batch_t prepare_molecule_batch(std::span<std::string_view const> smiles_list) {
    etkdg_molecule_batch_t batch {};
    prepare_molecule_batch_into(smiles_list, batch);
    return batch;
}

/**
 *  @brief Grow-only scratch for `distribute_batch_to_per_device_pools_`.
 *
 *  Holds the sort permutation, the CSR-packed per-device assignment list,
 *  the per-device CSR offset array, and the per-device per-molecule
 *  interaction-count staging vectors the downstream per-device pack loop
 *  consumes.  One instance per caller; reused across every
 *  `embedding-wave` so the distribute hot path never allocates.  Sibling
 *  to the pre-existing `static etkdg_molecule_batch_t shared_scratch` the
 *  same caller holds.
 */
struct distribute_scratch_t {
    std::vector<std::uint32_t> sort_permutation; ///< length: total_mol_count
    std::vector<std::uint32_t> per_device_flat;  ///< length: total_mol_count, CSR values
    // `32` mirrors the old hardcoded per_device_mol_list array; compile-time-
    // asserted against `smallfield::api::max_devices_k = 16` in smallfield.cu.
    static constexpr std::uint32_t max_devices_k = 32u;
    std::array<std::uint32_t, max_devices_k + 1u> per_device_offsets {}; ///< length: device_count + 1
    std::array<std::vector<std::uint32_t>, max_devices_k> mmff_interactions_per_mol {};
    std::array<std::vector<std::uint32_t>, max_devices_k> etk_interactions_per_mol {};
};

/**
 *  @brief Distribute an already-packed shared batch into `device_count`
 *  per-device pinned pools with a complexity-descending stripe partition.
 *
 *  Each device ends up with a disjoint subset of the shared batch's
 *  molecules; the subset is chosen so that per-device work distributions
 *  are near-identical — sort all mols by `molecule_complexity_score_`
 *  descending, stripe mol at sorted rank `k` to device `k % device_count`,
 *  then reverse each device's range so the per-device order ends up
 *  complexity-ascending (kernel `atom-limited-bin` dispatch wants smallest
 *  first, and complexity is a monotonic proxy of atom count on average).
 *
 *  Each mol's bytes are copied from the shared pool into the assigned
 *  device's pinned pool at per-device-relative offsets, and the mol's
 *  metadata is rebased to the per-device layout.
 *
 *  `scratch` carries the sort permutation and CSR-packed per-device
 *  assignment — grow-only, caller-owned; see `distribute_scratch_t`.
 *
 *  Caller ensures `per_device_out[d].host` is pre-sized (via
 *  `ensure_capacity`) to hold this device's slice of the shared batch;
 *  the ctor pre-sizing in `context_t` covers typical waves.
 */
inline void distribute_batch_to_per_device_pools_(etkdg_molecule_batch_t const &shared, std::uint32_t device_count,
                                                  std::span<etkdg_molecule_batch_t> per_device_out,
                                                  distribute_scratch_t &scratch) {

    std::uint32_t const total_mol_count = shared.num_molecules;
    auto const shared_metadata = shared.molecule_metadata();
    auto const shared_input_indices = shared.input_indices();

    for (std::uint32_t device_index = 0; device_index < device_count; ++device_index) {
        per_device_out[device_index].num_molecules = 0;
        per_device_out[device_index].max_atoms_per_molecule = 0;
        per_device_out[device_index].layout = batch_layout_t {};
    }
    if (total_mol_count == 0) return;

    // Sort permutation in place on grow-only scratch.
    if (scratch.sort_permutation.size() < total_mol_count) scratch.sort_permutation.resize(total_mol_count);
    if (scratch.per_device_flat.size() < total_mol_count) scratch.per_device_flat.resize(total_mol_count);
    std::iota(scratch.sort_permutation.begin(), scratch.sort_permutation.begin() + total_mol_count, 0u);
    std::sort(scratch.sort_permutation.begin(), scratch.sort_permutation.begin() + total_mol_count,
              [&](std::uint32_t a, std::uint32_t b) {
                  return molecule_complexity_score_(shared_metadata[a]) >
                         molecule_complexity_score_(shared_metadata[b]);
              });

    // CSR offsets for the per-device assignment: device d gets
    // `ceil((total_mol_count - d) / device_count)` mols from the stripe
    // (device 0 may get one more than the last device).
    scratch.per_device_offsets[0] = 0u;
    for (std::uint32_t device_index = 0; device_index < device_count; ++device_index) {
        std::uint32_t const count_d = (total_mol_count + (device_count - 1u) - device_index) / device_count;
        scratch.per_device_offsets[device_index + 1u] = scratch.per_device_offsets[device_index] + count_d;
    }

    // Single scatter pass: sorted rank `k` maps to device `k % device_count`
    // at position `k / device_count` within that device's CSR range.
    for (std::uint32_t k = 0; k < total_mol_count; ++k) {
        std::uint32_t const device_index = k % device_count;
        std::uint32_t const position = k / device_count;
        scratch.per_device_flat[scratch.per_device_offsets[device_index] + position] = scratch.sort_permutation[k];
    }

    // In-place reverse each per-device range so the kernel bin-dispatch
    // sees complexity-ascending mols first (smallest `atom-limited-bin`).
    for (std::uint32_t device_index = 0; device_index < device_count; ++device_index) {
        auto *begin = scratch.per_device_flat.data() + scratch.per_device_offsets[device_index];
        auto *end = scratch.per_device_flat.data() + scratch.per_device_offsets[device_index + 1u];
        std::reverse(begin, end);
    }

    auto const shared_view_for_counts = shared.host_view();
    auto const shared_mmff_atom_offsets = shared_view_for_counts.mmff_atom_offsets();
    auto const shared_etk_atom_offsets = shared_view_for_counts.etk_atom_offsets();
    auto compute_mol_interactions_count = [&](etkdg_molecule_metadata_t const &meta) -> std::uint32_t {
        std::uint32_t total = 0;
        for (std::uint32_t atom = 0; atom < meta.num_atoms; ++atom) {
            auto const &entry = shared_mmff_atom_offsets[meta.mmff_atom_offsets_start + atom];
            total += entry.bond_count + entry.angle_count + entry.torsion_count + entry.oop_count +
                     entry.stretchbend_count;
        }
        return total;
    };
    auto compute_mol_etk_interactions_count = [&](etkdg_molecule_metadata_t const &meta) -> std::uint32_t {
        std::uint32_t total = 0;
        for (std::uint32_t atom = 0; atom < meta.num_atoms; ++atom) {
            auto const &entry = shared_etk_atom_offsets[meta.etk_atom_offsets_start + atom];
            total += entry.interaction_count;
        }
        return total;
    };

#pragma omp parallel for num_threads(static_cast<int>(device_count)) schedule(static, 1)
    for (std::uint32_t device_index = 0; device_index < device_count; ++device_index) {
        std::span<std::uint32_t const> const mol_list {
            scratch.per_device_flat.data() + scratch.per_device_offsets[device_index],
            scratch.per_device_offsets[device_index + 1u] - scratch.per_device_offsets[device_index],
        };
        etkdg_molecule_batch_t &out = per_device_out[device_index];
        std::uint32_t const device_mol_count = static_cast<std::uint32_t>(mol_list.size());

        auto &mmff_interactions_per_mol = scratch.mmff_interactions_per_mol[device_index];
        auto &etk_interactions_per_mol = scratch.etk_interactions_per_mol[device_index];
        if (mmff_interactions_per_mol.size() < device_mol_count) mmff_interactions_per_mol.resize(device_mol_count);
        if (etk_interactions_per_mol.size() < device_mol_count) etk_interactions_per_mol.resize(device_mol_count);

        per_mol_sizes_t totals {};
        totals.valid = device_mol_count;
        std::uint32_t max_atoms_seen = 0;
        for (std::uint32_t pos = 0; pos < device_mol_count; ++pos) {
            auto const &meta = shared_metadata[mol_list[pos]];
            std::uint32_t const atom_count = meta.num_atoms;
            if (atom_count > max_atoms_seen) max_atoms_seen = atom_count;
            totals.bounds += atom_count * atom_count;
            totals.chirals += meta.num_chiral_centers;
            totals.torsions += meta.num_etk_torsions;
            totals.dcons += meta.num_distance_constraints;
            totals.mmff_atom_vdw += atom_count;
            totals.mmff_atom_charge += atom_count;
            totals.bond_distance += atom_count * atom_count;
            totals.mmff_bonds += meta.num_mmff_bonds;
            totals.mmff_angles += meta.num_mmff_angles;
            totals.mmff_torsions += meta.num_mmff_torsions;
            totals.mmff_oops += meta.num_mmff_oops;
            totals.mmff_stretchbends += meta.num_mmff_stretchbends;
            totals.mmff_atom_offsets += atom_count;
            totals.etk_atom_offsets += atom_count;
            mmff_interactions_per_mol[pos] = compute_mol_interactions_count(meta);
            etk_interactions_per_mol[pos] = compute_mol_etk_interactions_count(meta);
            totals.mmff_interactions += mmff_interactions_per_mol[pos];
            totals.etk_interactions += etk_interactions_per_mol[pos];
        }

        batch_layout_t device_layout = compute_batch_layout(totals, device_mol_count);
        out.num_molecules = device_mol_count;
        out.max_atoms_per_molecule = max_atoms_seen;
        out.layout = device_layout;
        if (device_layout.bytes_uploaded > 0 && !out.host.ensure_capacity(device_layout.bytes_uploaded)) {
            out.num_molecules = 0;
            out.layout = batch_layout_t {};
            continue;
        }
        if (device_mol_count == 0) continue;

        auto bounds_dst = out.bounds_matrices();
        auto chirals_dst = out.chiral_centers();
        auto torsions_dst = out.etk_torsions();
        auto dcons_dst = out.distance_constraints();
        auto vdw_dst = out.mmff_atom_vdw();
        auto charge_dst = out.mmff_atom_charge();
        auto bonds_dst = out.mmff_bonds();
        auto angles_dst = out.mmff_angles();
        auto torsions_mmff_dst = out.mmff_torsions();
        auto oops_dst = out.mmff_oops();
        auto stretchbends_dst = out.mmff_stretchbends();
        auto interactions_dst = out.mmff_interactions();
        auto aoff_dst = out.mmff_atom_offsets();
        auto etk_inter_dst = out.etk_interactions();
        auto etk_aoff_dst = out.etk_atom_offsets();
        auto metadata_dst = out.molecule_metadata();
        auto input_indices_dst = out.input_indices();

        auto const shared_view = shared.host_view();
        auto const bounds_src = shared_view.bounds_matrices();
        auto const chirals_src = shared_view.chiral_centers();
        auto const torsions_src = shared_view.etk_torsions();
        auto const dcons_src = shared_view.distance_constraints();
        auto const vdw_src = shared_view.mmff_atom_vdw();
        auto const charge_src = shared_view.mmff_atom_charge();
        auto const bonds_src = shared_view.mmff_bonds();
        auto const angles_src = shared_view.mmff_angles();
        auto const torsions_mmff_src = shared_view.mmff_torsions();
        auto const oops_src = shared_view.mmff_oops();
        auto const stretchbends_src = shared_view.mmff_stretchbends();
        auto const interactions_src = shared_view.mmff_interactions();
        auto const aoff_src = shared_view.mmff_atom_offsets();
        auto const etk_inter_src = shared_view.etk_interactions();
        auto const etk_aoff_src = shared_view.etk_atom_offsets();

        per_mol_sizes_t running {};
        for (std::uint32_t pos = 0; pos < device_mol_count; ++pos) {
            std::uint32_t const shared_mol_index = mol_list[pos];
            auto const &src_meta = shared_metadata[shared_mol_index];
            std::uint32_t const atom_count = src_meta.num_atoms;

            std::memcpy(bounds_dst.data() + running.bounds, bounds_src.data() + src_meta.bounds_offset,
                        static_cast<std::size_t>(atom_count) * atom_count * sizeof(float));
            std::memcpy(chirals_dst.data() + running.chirals, chirals_src.data() + src_meta.chiral_offset,
                        src_meta.num_chiral_centers * sizeof(chiral_center_descriptor_t));
            std::memcpy(torsions_dst.data() + running.torsions, torsions_src.data() + src_meta.torsion_offset,
                        src_meta.num_etk_torsions * sizeof(etk_torsion_descriptor_t));
            std::memcpy(dcons_dst.data() + running.dcons, dcons_src.data() + src_meta.constraint_offset,
                        src_meta.num_distance_constraints * sizeof(distance_constraint_t));
            std::memcpy(vdw_dst.data() + running.mmff_atom_vdw, vdw_src.data() + src_meta.mmff_atom_vdw_offset,
                        atom_count * sizeof(mmff94_atom_vdw_params_t));
            std::memcpy(charge_dst.data() + running.mmff_atom_charge,
                        charge_src.data() + src_meta.mmff_atom_charge_offset, atom_count * sizeof(float));
            std::memcpy(bonds_dst.data() + running.mmff_bonds, bonds_src.data() + src_meta.mmff_bond_offset,
                        src_meta.num_mmff_bonds * sizeof(mmff94_bond_t));
            std::memcpy(angles_dst.data() + running.mmff_angles, angles_src.data() + src_meta.mmff_angle_offset,
                        src_meta.num_mmff_angles * sizeof(mmff94_angle_t));
            std::memcpy(torsions_mmff_dst.data() + running.mmff_torsions,
                        torsions_mmff_src.data() + src_meta.mmff_torsion_offset,
                        src_meta.num_mmff_torsions * sizeof(mmff94_torsion_t));
            std::memcpy(oops_dst.data() + running.mmff_oops, oops_src.data() + src_meta.mmff_oop_offset,
                        src_meta.num_mmff_oops * sizeof(mmff94_oop_t));
            std::memcpy(stretchbends_dst.data() + running.mmff_stretchbends,
                        stretchbends_src.data() + src_meta.mmff_stretchbend_offset,
                        src_meta.num_mmff_stretchbends * sizeof(mmff94_stretchbend_t));
            // atom_offsets contain per-atom interaction-start offsets that
            // point into `mmff_interactions`; rebase each entry by the delta
            // between the source and destination interaction bases.
            std::uint32_t const source_atom_offsets_base = src_meta.mmff_atom_offsets_start;
            std::uint32_t const source_interactions_base = aoff_src[source_atom_offsets_base].bond_start;
            std::int32_t const mmff_interactions_rebase_delta = static_cast<std::int32_t>(running.mmff_interactions) -
                                                                static_cast<std::int32_t>(source_interactions_base);
            for (std::uint32_t atom = 0; atom < atom_count; ++atom) {
                mmff94_atom_offsets_t entry = aoff_src[source_atom_offsets_base + atom];
                entry.bond_start = static_cast<std::uint32_t>(static_cast<std::int32_t>(entry.bond_start) +
                                                              mmff_interactions_rebase_delta);
                entry.angle_start = static_cast<std::uint32_t>(static_cast<std::int32_t>(entry.angle_start) +
                                                               mmff_interactions_rebase_delta);
                entry.torsion_start = static_cast<std::uint32_t>(static_cast<std::int32_t>(entry.torsion_start) +
                                                                 mmff_interactions_rebase_delta);
                entry.oop_start = static_cast<std::uint32_t>(static_cast<std::int32_t>(entry.oop_start) +
                                                             mmff_interactions_rebase_delta);
                entry.stretchbend_start = static_cast<std::uint32_t>(
                    static_cast<std::int32_t>(entry.stretchbend_start) + mmff_interactions_rebase_delta);
                aoff_dst[running.mmff_atom_offsets + atom] = entry;
            }
            std::memcpy(interactions_dst.data() + running.mmff_interactions,
                        interactions_src.data() + source_interactions_base,
                        mmff_interactions_per_mol[pos] * sizeof(mmff94_atom_interaction_t));

            // Same rebase trick for the ETK interaction list.
            std::uint32_t const source_etk_atom_offsets_base = src_meta.etk_atom_offsets_start;
            std::uint32_t const source_etk_interactions_base =
                etk_aoff_src[source_etk_atom_offsets_base].interaction_start;
            std::int32_t const etk_interactions_rebase_delta = static_cast<std::int32_t>(running.etk_interactions) -
                                                               static_cast<std::int32_t>(source_etk_interactions_base);
            for (std::uint32_t atom = 0; atom < atom_count; ++atom) {
                etk_atom_offsets_t entry = etk_aoff_src[source_etk_atom_offsets_base + atom];
                entry.interaction_start = static_cast<std::uint32_t>(
                    static_cast<std::int32_t>(entry.interaction_start) + etk_interactions_rebase_delta);
                etk_aoff_dst[running.etk_atom_offsets + atom] = entry;
            }
            std::memcpy(etk_inter_dst.data() + running.etk_interactions,
                        etk_inter_src.data() + source_etk_interactions_base,
                        etk_interactions_per_mol[pos] * sizeof(etk_atom_interaction_t));

            etkdg_molecule_metadata_t dst_meta = src_meta;
            dst_meta.bounds_offset = running.bounds;
            dst_meta.chiral_offset = running.chirals;
            dst_meta.torsion_offset = running.torsions;
            dst_meta.constraint_offset = running.dcons;
            dst_meta.mmff_atom_vdw_offset = running.mmff_atom_vdw;
            dst_meta.mmff_atom_charge_offset = running.mmff_atom_charge;
            dst_meta.bond_distance_offset = running.bond_distance;
            dst_meta.mmff_bond_offset = running.mmff_bonds;
            dst_meta.mmff_angle_offset = running.mmff_angles;
            dst_meta.mmff_torsion_offset = running.mmff_torsions;
            dst_meta.mmff_oop_offset = running.mmff_oops;
            dst_meta.mmff_stretchbend_offset = running.mmff_stretchbends;
            dst_meta.mmff_atom_offsets_start = running.mmff_atom_offsets;
            dst_meta.etk_interaction_offset = running.etk_interactions;
            dst_meta.etk_atom_offsets_start = running.etk_atom_offsets;
            metadata_dst[pos] = dst_meta;
            input_indices_dst[pos] = shared_input_indices[shared_mol_index];

            running.bounds += atom_count * atom_count;
            running.chirals += src_meta.num_chiral_centers;
            running.torsions += src_meta.num_etk_torsions;
            running.dcons += src_meta.num_distance_constraints;
            running.mmff_atom_vdw += atom_count;
            running.mmff_atom_charge += atom_count;
            running.bond_distance += atom_count * atom_count;
            running.mmff_bonds += src_meta.num_mmff_bonds;
            running.mmff_angles += src_meta.num_mmff_angles;
            running.mmff_torsions += src_meta.num_mmff_torsions;
            running.mmff_oops += src_meta.num_mmff_oops;
            running.mmff_stretchbends += src_meta.num_mmff_stretchbends;
            running.mmff_atom_offsets += atom_count;
            running.etk_atom_offsets += atom_count;
            running.mmff_interactions += mmff_interactions_per_mol[pos];
            running.etk_interactions += etk_interactions_per_mol[pos];
        }
    }
}

/**
 *  @brief Preprocess + distribute entry: produce `device_count` per-device
 *  pinned batch pools with a complexity-descending stripe partition.  Each
 *  device's pool holds only its assigned mols in complexity-ascending
 *  (→ atom-count-ascending on average) order for kernel bin dispatch.
 *
 *  Internally reuses the three-phase preprocess (per-thread extract, serial
 *  prefix scan, parallel pack) against a reusable shared batch, then
 *  distributes to the per-device pools.  Two file-scope `static` scratch
 *  instances hold the pinned shared batch and the distribution CSR buffers
 *  so neither allocation fires on the hot path.
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
