/**
 *  @file etkdg_preprocess.hpp
 *  @brief CPU-side ETKDG preprocessing using RDKit C++ API (stages 0-1).
 *
 *  Converts molecular graphs into GPU-ready data structures:
 *   - Distance bounds matrix (N x N, smoothed via triangle inequality)
 *   - Chiral center descriptors (CIP-ordered neighbors, target sign)
 *   - ETK torsion descriptors (CSD Fourier coefficients)
 *   - 1-2 / 1-3 distance constraints
 *
 *  All functions operate on RDKit ROMol objects and produce flat arrays
 *  suitable for upload to GPU global memory.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
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

// clang-format off
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
// clang-format on

#include "batch_pool.hpp"
#include "etkdg_types.cuh"
#include "forcefield_tables.hpp"
#include "molecule_descriptor.hpp"
#include "csd_torsion_extract.hpp"

namespace smallfield {

/**
 *  @brief Complete preprocessed molecule ready for GPU pipeline upload.
 */
struct etkdg_prepared_molecule_t {
    std::uint32_t num_atoms;
    std::vector<float> bounds_matrix; ///< N x N flat, unsmoothed
    std::vector<chiral_center_descriptor_t> chiral_centers;
    std::vector<etk_torsion_descriptor_t> etk_torsions;
    std::vector<distance_constraint_t> distance_constraints; ///< 1-2 bond, 1-3 angle
    // MMFF94 force field parameters (stage 9).
    std::vector<mmff94_nonbonded_pair_t> mmff_nonbonded;
    std::vector<mmff94_bond_t> mmff_bonds;
    std::vector<mmff94_angle_t> mmff_angles;
    std::vector<mmff94_torsion_t> mmff_torsions;
    std::vector<mmff94_oop_t> mmff_oops;
    std::vector<mmff94_stretchbend_t> mmff_stretchbends;
    // Per-atom MMFF94 interaction lists built from the term arrays above.
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
inline void build_bounds_matrix(molecule_descriptor_t const &desc, std::vector<float> &bounds) {
    std::uint32_t const num_atoms = desc.num_atoms;
    bounds.assign(static_cast<std::size_t>(num_atoms) * num_atoms, 0.0f);

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

    // set14 — conservative envelope for 1-4 pairs.  Takes the sum of two bond lengths
    // plus a torsion-dependent spread as the upper bound; VdW minimum sum as the lower.
    // Uses descriptor BFS distances.
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        for (std::uint32_t j = i + 1; j < num_atoms; ++j) {
            std::uint8_t const bd = desc.bond_distance[i * max_atoms_k + j];
            if (bd != 3) continue;
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

    // Long-range pairs (bond_distance > 3): VdW lower bound, keep upper unconstrained.
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        for (std::uint32_t j = i + 1; j < num_atoms; ++j) {
            std::uint8_t const bd = desc.bond_distance[i * max_atoms_k + j];
            if (bd <= 3) continue;
            float const lower = vdw_lower_bound_angstrom(desc.element[i], desc.element[j]);
            bounds[j * num_atoms + i] = lower; // lower triangle
        }
    }
}

/**
 *  @brief Extract graph-only chiral center descriptors.
 *
 *  Neighbors are sorted by decreasing CIP priority. With that ordering, the GPU
 *  triple product over the top-three ligands has a stable sign convention:
 *  R → negative volume, S → positive volume.
 *
 *  @param[in] molecule RDKit molecule with stereochemistry assigned.
 *  @param[out] chiral_centers Filled with one descriptor per tetrahedral center.
 */
inline void extract_chiral_centers(RDKit::ROMol const &molecule, molecule_descriptor_t const &desc,
                                   std::vector<chiral_center_descriptor_t> &chiral_centers) {
    chiral_centers.clear();
    auto &stereo_molecule = const_cast<RDKit::ROMol &>(molecule);
    RDKit::MolOps::assignStereochemistry(stereo_molecule, true, true, true);

    RDKit::UINT_VECT cip_ranks(molecule.getNumAtoms(), 0);
    RDKit::Chirality::assignAtomCIPRanks(molecule, cip_ranks);

    std::uint32_t const num_atoms = desc.num_atoms;
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        if (desc.chiral_tag[a] == 0) continue;
        if (desc.degree[a] < 3) continue;

        auto const *atom = molecule.getAtomWithIdx(a);
        if (!atom->hasProp("_CIPCode")) continue;

        chiral_center_descriptor_t descriptor {};
        descriptor.center_atom = static_cast<std::uint8_t>(a);

        // Collect neighbors via descriptor's CSR adjacency, sort by CIP rank.
        std::uint8_t ordered[4] = {0, 0, 0, 0};
        std::uint32_t count = 0;
        std::uint16_t const nstart = desc.neighbor_offset[a];
        std::uint16_t const nend = desc.neighbor_offset[a + 1];
        for (std::uint16_t n = nstart; n < nend && count < 4; ++n) ordered[count++] = desc.neighbor_atom[n];
        // Insertion sort by (cip_rank desc, idx asc) — at most 4 elements.
        for (std::uint32_t i = 1; i < count; ++i) {
            std::uint8_t const v = ordered[i];
            std::uint32_t j = i;
            while (j > 0) {
                std::uint8_t const u = ordered[j - 1];
                bool const before = (cip_ranks[u] != cip_ranks[v]) ? (cip_ranks[u] < cip_ranks[v]) : (u > v);
                if (!before) break;
                ordered[j] = u;
                --j;
            }
            ordered[j] = v;
        }

        for (std::uint32_t i = 0; i < 4; ++i) descriptor.neighbor_atoms[i] = ordered[i];

        std::string cip_code;
        atom->getProp("_CIPCode", cip_code);
        descriptor.target_sign = (cip_code == "S") ? 1 : -1;

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
 *  @param[in,out] result Prepared molecule; ETK interaction lists are filled in place.
 */
inline void build_etk_interaction_lists(etkdg_prepared_molecule_t &result) {
    std::uint32_t const num_atoms = result.num_atoms;
    auto const &torsions = result.etk_torsions;
    auto const &constraints = result.distance_constraints;

    // pass 1: count interactions per atom
    std::vector<std::uint32_t> counts(num_atoms, 0);
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
    std::vector<std::uint32_t> cursors(num_atoms);
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
 *  @param[in,out] result Prepared molecule; MMFF94 arrays are filled in place.
 */
inline void extract_mmff94_params(RDKit::ROMol const &molecule, molecule_descriptor_t const &desc,
                                  etkdg_prepared_molecule_t &result) {
    result.mmff_nonbonded.clear();
    result.mmff_bonds.clear();
    result.mmff_angles.clear();
    result.mmff_torsions.clear();

    auto &mol = const_cast<RDKit::ROMol &>(molecule);
    // Skip redundant Kekulize inside MMFFMolProperties — SmilesToMol already sanitized.
    if (!mol.hasProp(RDKit::common_properties::_MMFFSanitized)) {
        mol.setProp(RDKit::common_properties::_MMFFSanitized, 1, true);
    }
    RDKit::MMFF::MMFFMolProperties mmff_props(mol);
    if (!mmff_props.isValid()) return;

    std::uint32_t const num_atoms = desc.num_atoms;

    // Cache partial charges in a single pass.
    float charges[128];
    for (std::uint32_t i = 0; i < num_atoms; ++i) charges[i] = static_cast<float>(mmff_props.getMMFFPartialCharge(i));

    // Build nonbonded pairs using descriptor's bond_distance for O(1) exclusion detection.
    // Excluded: bond_distance == 1 (1-2) or == 2 (1-3). 1-4: bond_distance == 3.
    //
    // VdW optimization: RDKit's MMFFMolProperties::getMMFFVdWParams(i, j) does a pair of
    // singleton-map lookups on atom type → MMFFVdW params for EACH of N²/2 pairs.
    // We hoist those lookups to a per-atom pass (N lookups), cache the pointers, and call
    // the Halgren combining-rule helpers directly. Eliminates ~N²/2 binary searches plus
    // the per-call MMFFVdWRijstarEps stack object and the isValid() branch.
    {
        auto const *mmff_vdw_coll = RDKit::MMFF::DefaultParameters::getMMFFVdW();
        ForceFields::MMFF::MMFFVdW const *per_atom_vdw[128] = {nullptr};
        for (std::uint32_t a = 0; a < num_atoms; ++a) {
            std::uint32_t const atom_type = mmff_props.getMMFFAtomType(a);
            per_atom_vdw[a] = (*mmff_vdw_coll)(atom_type);
        }
        for (std::uint32_t i = 0; i < num_atoms; ++i) {
            auto const *vdw_i = per_atom_vdw[i];
            if (vdw_i == nullptr) continue;
            for (std::uint32_t j = i + 1; j < num_atoms; ++j) {
                std::uint8_t const d = bonds_between(desc, i, j);
                if (d <= 2) continue; // 1-2 or 1-3 excluded
                auto const *vdw_j = per_atom_vdw[j];
                if (vdw_j == nullptr) continue;

                double r_star = ForceFields::MMFF::Utils::calcUnscaledVdWMinimum(mmff_vdw_coll, vdw_i, vdw_j);
                double epsilon = ForceFields::MMFF::Utils::calcUnscaledVdWWellDepth(r_star, vdw_i, vdw_j);
                ForceFields::MMFF::Utils::scaleVdWParams(r_star, epsilon, mmff_vdw_coll, vdw_i, vdw_j);

                std::uint8_t const is_1_4 = (d == 3) ? 1 : 0;
                result.mmff_nonbonded.push_back({static_cast<std::uint8_t>(i), static_cast<std::uint8_t>(j), is_1_4, 0,
                                                 static_cast<float>(epsilon), static_cast<float>(r_star),
                                                 charges[i] * charges[j]});
            }
        }
    }

    // Build bond terms — iterate descriptor's bond list (no RDKit traversal).
    for (std::uint32_t b = 0; b < desc.num_bonds; ++b) {
        std::uint32_t const i = desc.bond_atom_i[b];
        std::uint32_t const j = desc.bond_atom_j[b];
        unsigned int bond_type;
        ForceFields::MMFF::MMFFBond bp;
        if (!mmff_props.getMMFFBondStretchParams(mol, i, j, bond_type, bp)) continue;
        result.mmff_bonds.push_back({static_cast<std::uint8_t>(i),
                                     static_cast<std::uint8_t>(j),
                                     {0, 0},
                                     static_cast<float>(bp.kb),
                                     static_cast<float>(bp.r0),
                                     0.0f});
    }

    // Build angle terms — iterate atoms as angle centers, enumerate neighbor pairs via CSR.
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint16_t const nstart = desc.neighbor_offset[center];
        std::uint16_t const nend = desc.neighbor_offset[center + 1];
        for (std::uint16_t a = nstart; a < nend; ++a) {
            for (std::uint16_t b = a + 1; b < nend; ++b) {
                std::uint32_t const ai = desc.neighbor_atom[a];
                std::uint32_t const aj = desc.neighbor_atom[b];
                unsigned int angle_type;
                ForceFields::MMFF::MMFFAngle ap;
                if (!mmff_props.getMMFFAngleBendParams(mol, ai, center, aj, angle_type, ap)) continue;
                result.mmff_angles.push_back({static_cast<std::uint8_t>(ai), static_cast<std::uint8_t>(center),
                                              static_cast<std::uint8_t>(aj), 0, static_cast<float>(ap.ka),
                                              static_cast<float>(ap.theta0 * M_PI / 180.0), 0.0f});
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
        for (std::uint16_t nj = jstart; nj < jend; ++nj) {
            std::uint32_t const i = desc.neighbor_atom[nj];
            if (i == k) continue;
            for (std::uint16_t nk = kstart; nk < kend; ++nk) {
                std::uint32_t const l = desc.neighbor_atom[nk];
                if (l == j) continue;
                unsigned int torsion_type;
                ForceFields::MMFF::MMFFTor tp;
                if (!mmff_props.getMMFFTorsionParams(mol, i, j, k, l, torsion_type, tp)) continue;
                result.mmff_torsions.push_back({static_cast<std::uint8_t>(i), static_cast<std::uint8_t>(j),
                                                static_cast<std::uint8_t>(k), static_cast<std::uint8_t>(l),
                                                static_cast<float>(tp.V1), static_cast<float>(tp.V2),
                                                static_cast<float>(tp.V3)});
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
            ForceFields::MMFF::MMFFOop oop_params;
            if (!mmff_props.getMMFFOopBendParams(mol, neighbor_cache[a], center, plane[0], plane[1], oop_params))
                continue;
            result.mmff_oops.push_back({neighbor_cache[a],
                                        static_cast<std::uint8_t>(center),
                                        plane[0],
                                        plane[1],
                                        static_cast<float>(oop_params.koop),
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
                unsigned int stretchbend_type;
                ForceFields::MMFF::MMFFStbn stretchbend_params;
                ForceFields::MMFF::MMFFBond bond_params[2];
                ForceFields::MMFF::MMFFAngle angle_params;
                if (!mmff_props.getMMFFStretchBendParams(mol, ai, center, aj, stretchbend_type, stretchbend_params,
                                                         bond_params, angle_params))
                    continue;
                result.mmff_stretchbends.push_back(
                    {static_cast<std::uint8_t>(ai), static_cast<std::uint8_t>(center), static_cast<std::uint8_t>(aj), 0,
                     static_cast<float>(stretchbend_params.kbaIJK), static_cast<float>(stretchbend_params.kbaKJI),
                     static_cast<float>(angle_params.theta0 * M_PI / 180.0), static_cast<float>(bond_params[0].r0),
                     static_cast<float>(bond_params[1].r0), 0.0f});
            }
        }
    }

    // Build per-atom interaction lists in two passes without vector-of-vectors.
    auto const &nonbonded = result.mmff_nonbonded;
    auto const &bonds = result.mmff_bonds;
    auto const &angles = result.mmff_angles;
    auto const &torsions = result.mmff_torsions;
    auto const &oops = result.mmff_oops;
    auto const &stretchbends = result.mmff_stretchbends;

    // pass 1: count per-type interactions per atom
    struct per_atom_counts_t {
        std::uint8_t nonbonded, bond, angle, torsion, oop, stretchbend;
    };
    std::vector<per_atom_counts_t> type_counts(num_atoms, {0, 0, 0, 0, 0, 0});
    for (auto const &p : nonbonded) {
        type_counts[p.atom_i].nonbonded++;
        type_counts[p.atom_j].nonbonded++;
    }
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

    // compute per-type offsets: all nonbonded first, then bonds, then angles, etc.
    result.mmff_atom_offsets.resize(num_atoms);
    std::uint32_t running = 0;
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        auto const &counts = type_counts[atom];
        auto &offsets = result.mmff_atom_offsets[atom];
        offsets.nonbonded_start = static_cast<std::uint32_t>(running);
        offsets.nonbonded_count = counts.nonbonded;
        running += counts.nonbonded;
        offsets.bond_start = static_cast<std::uint32_t>(running);
        offsets.bond_count = counts.bond;
        running += counts.bond;
        offsets.angle_start = static_cast<std::uint32_t>(running);
        offsets.angle_count = counts.angle;
        running += counts.angle;
        offsets.torsion_start = static_cast<std::uint32_t>(running);
        offsets.torsion_count = counts.torsion;
        running += counts.torsion;
        offsets.oop_start = static_cast<std::uint32_t>(running);
        offsets.oop_count = counts.oop;
        running += counts.oop;
        offsets.stretchbend_start = static_cast<std::uint32_t>(running);
        offsets.stretchbend_count = counts.stretchbend;
        running += counts.stretchbend;
    }

    // pass 2: fill interaction entries grouped by type per atom
    std::uint32_t const total_interactions = running;
    result.mmff_interactions.resize(total_interactions);

    // per-type cursors per atom
    struct per_atom_cursors_t {
        std::uint32_t nonbonded, bond, angle, torsion, oop, stretchbend;
    };
    std::vector<per_atom_cursors_t> cursors(num_atoms);
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        auto const &off = result.mmff_atom_offsets[atom];
        cursors[atom] = {off.nonbonded_start, off.bond_start, off.angle_start,
                         off.torsion_start,   off.oop_start,  off.stretchbend_start};
    }

    for (std::uint32_t term = 0; term < nonbonded.size(); ++term) {
        result.mmff_interactions[cursors[nonbonded[term].atom_i].nonbonded++] = {term, mmff94_nonbonded_k,
                                                                                 atom_role_first_k};
        result.mmff_interactions[cursors[nonbonded[term].atom_j].nonbonded++] = {term, mmff94_nonbonded_k,
                                                                                 atom_role_second_k};
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
 *  @brief Convert a SMILES string to a fully prepared molecule (stages 0-1 complete).
 *
 *  Parses the SMILES, adds hydrogens, builds the bounds matrix, extracts chiral
 *  centers, torsions, distance constraints, and MMFF94 parameters. Returns a
 *  molecule with num_atoms == 0 on failure.
 *
 *  @param[in] smiles SMILES string for the input molecule.
 *  @return Fully populated etkdg_prepared_molecule_t, or an empty one on failure.
 */
inline etkdg_prepared_molecule_t prepare_molecule(std::string_view smiles) {
    etkdg_prepared_molecule_t result {};

    std::unique_ptr<RDKit::RWMol> mol;
    try {
        // RDKit::SmilesToMol takes std::string const& — one unavoidable copy per
        // molecule on this path.  Kept scoped to this single call so callers stay
        // allocation-free.
        std::string const smiles_str {smiles};
        mol.reset(RDKit::SmilesToMol(smiles_str));
    }
    catch (...) {
        return result;
    }
    if (!mol) return result;

    try {
        RDKit::MolOps::addHs(*mol);
    }
    catch (...) {
        return result;
    }
    result.num_atoms = mol->getNumAtoms();
    if (result.num_atoms > 128) return result; // Too large for the single-warp pipeline.

    try {
        // Build the flat descriptor once. All downstream stages read from this.
        molecule_descriptor_t desc;
        if (!build_descriptor(*mol, desc)) {
            result.num_atoms = 0;
            return result;
        }
        build_bounds_matrix(desc, result.bounds_matrix);
        extract_chiral_centers(*mol, desc, result.chiral_centers);
        extract_etk_torsions_native(*mol, desc, result.etk_torsions);
        extract_distance_constraints(desc, result.bounds_matrix, result.distance_constraints);
        build_etk_interaction_lists(result);
        extract_mmff94_params(*mol, desc, result);
    }
    catch (...) {
        result.num_atoms = 0; // Mark as failed.
        return result;
    }

    result.metadata.num_atoms = result.num_atoms;
    result.metadata.num_chiral_centers = static_cast<std::uint32_t>(result.chiral_centers.size());
    result.metadata.num_etk_torsions = static_cast<std::uint32_t>(result.etk_torsions.size());
    result.metadata.num_distance_constraints = static_cast<std::uint32_t>(result.distance_constraints.size());
    result.metadata.num_mmff_nonbonded = static_cast<std::uint32_t>(result.mmff_nonbonded.size());
    result.metadata.num_mmff_bonds = static_cast<std::uint32_t>(result.mmff_bonds.size());
    result.metadata.num_mmff_angles = static_cast<std::uint32_t>(result.mmff_angles.size());
    result.metadata.num_mmff_torsions = static_cast<std::uint32_t>(result.mmff_torsions.size());
    result.metadata.num_mmff_oops = static_cast<std::uint32_t>(result.mmff_oops.size());
    result.metadata.num_mmff_stretchbends = static_cast<std::uint32_t>(result.mmff_stretchbends.size());
    result.metadata.bounds_offset = 0;
    result.metadata.chiral_offset = 0;
    result.metadata.torsion_offset = 0;
    result.metadata.constraint_offset = 0;
    result.metadata.mmff_nonbonded_offset = 0;
    result.metadata.mmff_bond_offset = 0;
    result.metadata.mmff_angle_offset = 0;
    result.metadata.mmff_torsion_offset = 0;
    result.metadata.mmff_oop_offset = 0;
    result.metadata.mmff_stretchbend_offset = 0;
    result.metadata.mmff_atom_offsets_start = 0;
    result.metadata.etk_interaction_offset = 0;
    result.metadata.etk_atom_offsets_start = 0;

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
    std::uint32_t mmff_nonbonded;
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
    std::uint32_t padding;
};
static_assert(sizeof(per_mol_sizes_t) == 64, "per_mol_sizes_t must be exactly one cache line");

/** Field-wise sum used both by the prefix scan and as the identity-additive operation. */
constexpr per_mol_sizes_t operator+(per_mol_sizes_t a, per_mol_sizes_t const &b) {
    a.bounds += b.bounds;
    a.chirals += b.chirals;
    a.torsions += b.torsions;
    a.dcons += b.dcons;
    a.mmff_nonbonded += b.mmff_nonbonded;
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
    s.mmff_nonbonded = static_cast<std::uint32_t>(p.mmff_nonbonded.size());
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
template <typename T>
inline void append_slice(std::span<T> dst, std::uint32_t offset, std::vector<T> const &src) {
    static_assert(std::is_trivially_copyable_v<T>, "append_slice requires trivially-copyable elements");
    if (!src.empty()) std::memcpy(dst.data() + offset, src.data(), src.size() * sizeof(T));
}

/** Build the per-molecule metadata entry with batch-wide offsets baked in. */
inline etkdg_molecule_metadata_t make_meta(etkdg_molecule_metadata_t base, per_mol_sizes_t const &o) {
    base.bounds_offset = o.bounds;
    base.chiral_offset = o.chirals;
    base.torsion_offset = o.torsions;
    base.constraint_offset = o.dcons;
    base.mmff_nonbonded_offset = o.mmff_nonbonded;
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
 *  detail — consumers only read `offset_*` / `count_*` pairs.
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
    layout.offset_mmff_nonbonded = place_region_<mmff94_nonbonded_pair_t>(running, totals.mmff_nonbonded);
    layout.count_mmff_nonbonded = totals.mmff_nonbonded;
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
    layout.bytes_total = running;
    return layout;
}

/**
 *  @brief Prepare a batch of molecules from SMILES strings.
 *
 *  Phase 1 (OpenMP parallel): prepare each molecule independently.
 *  Phase 2 (three-pass): extract sizes (parallel), prefix-scan the struct-of-sizes
 *  (serial, cache-local), append slices (parallel).  No push_back / insert anywhere.
 *
 *  @param[in] smiles_list Vector of SMILES strings to process.
 *  @return Batched molecule data with contiguous arrays for GPU upload.
 */
inline etkdg_molecule_batch_t prepare_molecule_batch(std::span<std::string_view const> smiles_list) {

    etkdg_molecule_batch_t batch {};
    // Phase 1: Prepare molecules in parallel. RDKit calls are thread-safe for independent molecules.
    //
    // CRITICAL: this file is included by `smallfield.cu` which is compiled with nvcc.
    // For OpenMP to actually parallelize here, the CMake target must pass
    // `-Xcompiler=-fopenmp` through nvcc to the host compiler. Otherwise nvcc silently
    // ignores the pragma and this loop runs single-threaded — which costs 20x.
    std::size_t const n = smiles_list.size();
    std::vector<etkdg_prepared_molecule_t> prepared(n);
#pragma omp parallel for schedule(dynamic, 4)
    for (std::size_t i = 0; i < n; ++i) { prepared[i] = prepare_molecule(smiles_list[i]); }

    // Phase 2 — produce one contiguous pinned host buffer containing all 14
    // per-molecule arrays plus molecule_metadata and input_indices.  Three passes:
    //
    //   Pass 1 (parallel): extract per-molecule sizes into one `std::vector<per_mol_sizes_t>`.
    //   Pass 2 (serial):   prefix-scan into `offsets` (n + 1 entries).  Then derive the
    //                      `batch_layout_t` from the totals and pin-allocate one pool
    //                      buffer of size `layout.bytes_total`.
    //   Pass 3 (parallel): one loop over m memcpys each per-molecule sub-array into its
    //                      slice of the pinned pool.
    //
    // The dense metadata slot is `offsets[m].valid` — the prefix sum of the `valid`
    // flag doubles as the compaction index for `molecule_metadata` / `input_indices`.

    // Pass 1: parallel size extraction.
    std::vector<per_mol_sizes_t> sizes(n);
#pragma omp parallel for schedule(static)
    for (std::size_t m = 0; m < n; ++m) sizes[m] = extract_sizes(prepared[m]);

    // Pass 2: serial prefix scan over the struct of sizes.  offsets[m] is the
    // (exclusive) starting offset for molecule m into each batch array; offsets[n]
    // holds the total sizes.
    std::vector<per_mol_sizes_t> offsets(n + 1);
    offsets[0] = per_mol_sizes_t {};
    for (std::size_t m = 0; m < n; ++m) offsets[m + 1] = offsets[m] + sizes[m];
    std::uint32_t const n_valid = offsets[n].valid;
    batch.num_molecules = n_valid;

    // Single reduction pass over num_atoms to get the batch-wide max.
    std::uint32_t max_atoms_reduction = 0;
#pragma omp parallel for schedule(static) reduction(max : max_atoms_reduction)
    for (std::size_t m = 0; m < n; ++m)
        if (prepared[m].num_atoms > max_atoms_reduction) max_atoms_reduction = prepared[m].num_atoms;
    batch.max_atoms_per_molecule = max_atoms_reduction;

    // Compute the layout and allocate one pinned host buffer.  The launch path
    // mirrors this into a per-GPU device buffer via a single cudaMemcpyAsync.
    batch.layout = compute_batch_layout(offsets[n], n_valid);
    if (batch.layout.bytes_total > 0) {
        if (!batch.host.ensure_capacity(batch.layout.bytes_total)) {
            batch.num_molecules = 0;
            batch.layout = batch_layout_t {};
            return batch;
        }
    }

    // Bind typed spans over the pinned pool once so Pass 3 doesn't re-derive
    // them per iteration.  Each span is a non-owning view into `batch.host`.
    auto bounds_span = batch.bounds_matrices();
    auto chirals_span = batch.chiral_centers();
    auto torsions_span = batch.etk_torsions();
    auto dcons_span = batch.distance_constraints();
    auto mmff_nb_span = batch.mmff_nonbonded();
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

    // Pass 3: parallel copy.  Each iteration writes into a unique slice of every
    // output array — no data race.  Failed molecules (sizes[m].valid == 0) are
    // skipped cheaply.
#pragma omp parallel for schedule(static)
    for (std::size_t m = 0; m < n; ++m) {
        if (sizes[m].valid == 0) continue;
        auto const &p = prepared[m];
        auto const &o = offsets[m];

        append_slice(bounds_span, o.bounds, p.bounds_matrix);
        append_slice(chirals_span, o.chirals, p.chiral_centers);
        append_slice(torsions_span, o.torsions, p.etk_torsions);
        append_slice(dcons_span, o.dcons, p.distance_constraints);
        append_slice(mmff_nb_span, o.mmff_nonbonded, p.mmff_nonbonded);
        append_slice(mmff_bonds_span, o.mmff_bonds, p.mmff_bonds);
        append_slice(mmff_angles_span, o.mmff_angles, p.mmff_angles);
        append_slice(mmff_tors_span, o.mmff_torsions, p.mmff_torsions);
        append_slice(mmff_oops_span, o.mmff_oops, p.mmff_oops);
        append_slice(mmff_sb_span, o.mmff_stretchbends, p.mmff_stretchbends);
        append_slice(mmff_inter_span, o.mmff_interactions, p.mmff_interactions);
        append_slice(etk_inter_span, o.etk_interactions, p.etk_interactions);

        // MMFF atom offsets store offsets INTO the batched interaction array and
        // must be rebased by the batch-wide offset for this molecule.  Too many
        // fields to rebase in one memcpy, so do it inline.
        for (std::size_t k = 0; k < p.mmff_atom_offsets.size(); ++k) {
            auto entry = p.mmff_atom_offsets[k];
            entry.nonbonded_start += o.mmff_interactions;
            entry.bond_start += o.mmff_interactions;
            entry.angle_start += o.mmff_interactions;
            entry.torsion_start += o.mmff_interactions;
            entry.oop_start += o.mmff_interactions;
            entry.stretchbend_start += o.mmff_interactions;
            mmff_aoff_span[o.mmff_atom_offsets + k] = entry;
        }
        for (std::size_t k = 0; k < p.etk_atom_offsets.size(); ++k) {
            auto entry = p.etk_atom_offsets[k];
            entry.interaction_start += o.etk_interactions;
            etk_aoff_span[o.etk_atom_offsets + k] = entry;
        }

        std::uint32_t const vi = o.valid; // dense index — offsets[m].valid counts prior valid mols
        metadata_span[vi] = make_meta(p.metadata, o);
        input_idx_span[vi] = static_cast<std::uint32_t>(m);
    }

    return batch;
}

} // namespace smallfield
