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
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include <DistGeom/BoundsMatrix.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/Chirality.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
// clang-format on

#include "etkdg_types.cuh"

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
inline void build_bounds_matrix(RDKit::ROMol const &molecule, std::vector<float> &bounds) {
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    bounds.resize(num_atoms * num_atoms, 0.0f);

    // Use RDKit's bounds matrix builder.
    auto *bounds_raw = new DistGeom::BoundsMatrix(num_atoms);
    DistGeom::BoundsMatPtr bounds_mat(bounds_raw);
    RDKit::DGeomHelpers::initBoundsMat(bounds_raw);
    RDKit::DGeomHelpers::setTopolBounds(molecule, bounds_mat);
    // Do not smooth on the CPU. The GPU kernel handles Floyd-Warshall in shared memory.

    // Flatten to the row-major convention used by the GPU path.
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        for (std::uint32_t j = i + 1; j < num_atoms; ++j) {
            bounds[i * num_atoms + j] = static_cast<float>(bounds_mat->getUpperBound(i, j));
            bounds[j * num_atoms + i] = static_cast<float>(bounds_mat->getLowerBound(i, j));
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
inline void extract_chiral_centers(RDKit::ROMol const &molecule,
                                   std::vector<chiral_center_descriptor_t> &chiral_centers) {
    chiral_centers.clear();
    auto &stereo_molecule = const_cast<RDKit::ROMol &>(molecule);
    RDKit::MolOps::assignStereochemistry(stereo_molecule, true, true, true);

    RDKit::UINT_VECT cip_ranks(molecule.getNumAtoms(), 0);
    RDKit::Chirality::assignAtomCIPRanks(molecule, cip_ranks);

    for (auto const &atom : molecule.atoms()) {
        auto chirality = atom->getChiralTag();
        if (chirality != RDKit::Atom::CHI_TETRAHEDRAL_CW && chirality != RDKit::Atom::CHI_TETRAHEDRAL_CCW) continue;
        if (!atom->hasProp("_CIPCode")) continue;

        chiral_center_descriptor_t descriptor {};
        descriptor.center_atom = static_cast<std::uint8_t>(atom->getIdx());

        if (atom->getDegree() < 3) continue;

        std::vector<std::uint32_t> ordered_neighbors;
        ordered_neighbors.reserve(4);
        for (auto const &neighbor : molecule.atomNeighbors(atom)) { ordered_neighbors.push_back(neighbor->getIdx()); }
        std::sort(ordered_neighbors.begin(), ordered_neighbors.end(), [&](std::uint32_t lhs, std::uint32_t rhs) {
            if (cip_ranks[lhs] != cip_ranks[rhs]) return cip_ranks[lhs] > cip_ranks[rhs];
            return lhs < rhs;
        });
        std::uint32_t const neighbor_count = static_cast<std::uint32_t>(ordered_neighbors.size());
        if (neighbor_count < 3) continue;

        for (std::uint32_t i = 0; i < 4 && i < neighbor_count; ++i)
            descriptor.neighbor_atoms[i] = static_cast<std::uint8_t>(ordered_neighbors[i]);

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
inline void extract_distance_constraints(RDKit::ROMol const &molecule, std::vector<float> const &bounds_matrix,
                                         std::vector<distance_constraint_t> &constraints) {
    constraints.clear();
    std::uint32_t const num_atoms = molecule.getNumAtoms();

    // 1-2 bond constraints: read actual bounds from the matrix
    for (auto const &bond : molecule.bonds()) {
        std::uint32_t const atom_i = bond->getBeginAtomIdx();
        std::uint32_t const atom_j = bond->getEndAtomIdx();
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

    // 1-3 constraints from angles: read from bounds matrix
    for (auto const &atom : molecule.atoms()) {
        std::vector<unsigned int> neighbors;
        for (auto const &neighbor : molecule.atomNeighbors(atom)) { neighbors.push_back(neighbor->getIdx()); }

        for (std::size_t a = 0; a < neighbors.size(); ++a) {
            for (std::size_t b = a + 1; b < neighbors.size(); ++b) {
                std::uint32_t const ni = neighbors[a];
                std::uint32_t const nj = neighbors[b];
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
inline void extract_mmff94_params(RDKit::ROMol const &molecule, etkdg_prepared_molecule_t &result) {
    result.mmff_nonbonded.clear();
    result.mmff_bonds.clear();
    result.mmff_angles.clear();
    result.mmff_torsions.clear();

    auto &mol = const_cast<RDKit::ROMol &>(molecule);
    RDKit::MMFF::MMFFMolProperties mmff_props(mol);
    if (!mmff_props.isValid()) return;

    std::uint32_t const num_atoms = molecule.getNumAtoms();

    // Use stack-allocated bitset arrays for exclusion and 1-4 pair detection.
    constexpr std::uint32_t max_bits = 128 * 128;
    constexpr std::uint32_t bitset_words = (max_bits + 63) / 64;
    std::uint64_t excluded[bitset_words] = {};
    std::uint64_t is_one_four[bitset_words] = {};

    auto set_bit = [](std::uint64_t *bits, std::uint32_t index) {
        bits[index / 64] |= (std::uint64_t(1) << (index % 64));
    };
    auto get_bit = [](std::uint64_t const *bits, std::uint32_t index) -> bool {
        return (bits[index / 64] >> (index % 64)) & 1;
    };

    // Mark 1-2 bonds as excluded.
    for (auto const &bond : molecule.bonds()) {
        unsigned int a = bond->getBeginAtomIdx(), b = bond->getEndAtomIdx();
        set_bit(excluded, a * num_atoms + b);
        set_bit(excluded, b * num_atoms + a);
    }

    // Mark 1-3 pairs as excluded and detect 1-4 pairs from torsion terminals.
    for (auto const &atom : molecule.atoms()) {
        unsigned int center = atom->getIdx();
        // Collect neighbors by inline iteration without a temporary vector.
        unsigned int neighbor_indices[8]; // Max 8 neighbors (typical max valence).
        unsigned int neighbor_count = 0;
        for (auto const &nb : molecule.atomNeighbors(atom)) {
            if (neighbor_count < 8) neighbor_indices[neighbor_count++] = nb->getIdx();
        }
        // 1-3 exclusions: all pairs of neighbors
        for (unsigned int a = 0; a < neighbor_count; ++a) {
            for (unsigned int b = a + 1; b < neighbor_count; ++b) {
                set_bit(excluded, neighbor_indices[a] * num_atoms + neighbor_indices[b]);
                set_bit(excluded, neighbor_indices[b] * num_atoms + neighbor_indices[a]);
            }
        }
    }

    // 1-4 pairs from torsion terminals: for each bond j-k, pair (i,l) where i∈N(j)\k, l∈N(k)\j
    for (auto const &bond : molecule.bonds()) {
        unsigned int j = bond->getBeginAtomIdx(), k = bond->getEndAtomIdx();
        for (auto const &neighbor_j : molecule.atomNeighbors(molecule.getAtomWithIdx(j))) {
            if (neighbor_j->getIdx() == k) continue;
            for (auto const &neighbor_k : molecule.atomNeighbors(molecule.getAtomWithIdx(k))) {
                if (neighbor_k->getIdx() == j) continue;
                unsigned int i = neighbor_j->getIdx(), l = neighbor_k->getIdx();
                set_bit(is_one_four, i * num_atoms + l);
                set_bit(is_one_four, l * num_atoms + i);
            }
        }
    }

    // Cache partial charges in a single pass.
    float charges[256]; // Stack-allocated, max 256 atoms.
    for (unsigned int i = 0; i < num_atoms && i < 256; ++i)
        charges[i] = static_cast<float>(mmff_props.getMMFFPartialCharge(i));

    // Build nonbonded pairs with O(N²) iteration and O(1) exclusion lookup.
    for (unsigned int i = 0; i < num_atoms; ++i) {
        for (unsigned int j = i + 1; j < num_atoms; ++j) {
            if (get_bit(excluded, i * num_atoms + j)) continue;

            ForceFields::MMFF::MMFFVdWRijstarEps vdw_params;
            if (!mmff_props.getMMFFVdWParams(i, j, vdw_params)) continue;

            result.mmff_nonbonded.push_back({static_cast<std::uint8_t>(i), static_cast<std::uint8_t>(j),
                                             static_cast<std::uint8_t>(get_bit(is_one_four, i * num_atoms + j) ? 1 : 0),
                                             0, static_cast<float>(vdw_params.epsilon),
                                             static_cast<float>(vdw_params.R_ij_star), charges[i] * charges[j]});
        }
    }

    // Build bond terms.
    for (auto const &bond : molecule.bonds()) {
        unsigned int i = bond->getBeginAtomIdx(), j = bond->getEndAtomIdx();
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

    // Build angle terms by iterating atoms as centers and pairing their neighbors.
    for (auto const &atom : molecule.atoms()) {
        unsigned int j = atom->getIdx();
        unsigned int nonbonded[8];
        unsigned int neighbor_count = 0;
        for (auto const &n : molecule.atomNeighbors(atom))
            if (neighbor_count < 8) nonbonded[neighbor_count++] = n->getIdx();

        for (unsigned int a = 0; a < neighbor_count; ++a) {
            for (unsigned int b = a + 1; b < neighbor_count; ++b) {
                unsigned int angle_type;
                ForceFields::MMFF::MMFFAngle ap;
                if (!mmff_props.getMMFFAngleBendParams(mol, nonbonded[a], j, nonbonded[b], angle_type, ap)) continue;
                result.mmff_angles.push_back({static_cast<std::uint8_t>(nonbonded[a]), static_cast<std::uint8_t>(j),
                                              static_cast<std::uint8_t>(nonbonded[b]), 0, static_cast<float>(ap.ka),
                                              static_cast<float>(ap.theta0 * M_PI / 180.0), 0.0f});
            }
        }
    }

    // Build torsion terms.
    for (auto const &bond : molecule.bonds()) {
        unsigned int j = bond->getBeginAtomIdx(), k = bond->getEndAtomIdx();
        for (auto const &nj : molecule.atomNeighbors(molecule.getAtomWithIdx(j))) {
            if (nj->getIdx() == k) continue;
            for (auto const &nk : molecule.atomNeighbors(molecule.getAtomWithIdx(k))) {
                if (nk->getIdx() == j) continue;
                unsigned int torsion_type;
                ForceFields::MMFF::MMFFTor tp;
                if (!mmff_props.getMMFFTorsionParams(mol, nj->getIdx(), j, k, nk->getIdx(), torsion_type, tp)) continue;
                result.mmff_torsions.push_back({static_cast<std::uint8_t>(nj->getIdx()), static_cast<std::uint8_t>(j),
                                                static_cast<std::uint8_t>(k), static_cast<std::uint8_t>(nk->getIdx()),
                                                static_cast<float>(tp.V1), static_cast<float>(tp.V2),
                                                static_cast<float>(tp.V3)});
            }
        }
    }

    // Build out-of-plane bending terms for each sp2 center with 3 or more neighbors.
    for (auto const &atom : molecule.atoms()) {
        unsigned int j = atom->getIdx();
        unsigned int neighbor_indices[8];
        unsigned int neighbor_count = 0;
        for (auto const &n : molecule.atomNeighbors(atom))
            if (neighbor_count < 8) neighbor_indices[neighbor_count++] = n->getIdx();
        if (neighbor_count < 3) continue;

        for (unsigned int a = 0; a < neighbor_count; ++a) {
            // Treat neighbor_indices[a] as the out-of-plane atom and the other two as the plane atoms.
            unsigned int plane[2];
            unsigned int plane_count = 0;
            for (unsigned int b = 0; b < neighbor_count && plane_count < 2; ++b)
                if (b != a) plane[plane_count++] = neighbor_indices[b];
            if (plane_count < 2) continue;

            ForceFields::MMFF::MMFFOop oop_params;
            if (!mmff_props.getMMFFOopBendParams(mol, neighbor_indices[a], j, plane[0], plane[1], oop_params)) continue;
            result.mmff_oops.push_back({static_cast<std::uint8_t>(neighbor_indices[a]),
                                        static_cast<std::uint8_t>(j),
                                        static_cast<std::uint8_t>(plane[0]),
                                        static_cast<std::uint8_t>(plane[1]),
                                        static_cast<float>(oop_params.koop),
                                        {0.0f, 0.0f}});
        }
    }

    // Build stretch-bend terms by reusing angle topology.
    for (auto const &atom : molecule.atoms()) {
        unsigned int j = atom->getIdx();
        unsigned int neighbor_indices[8];
        unsigned int neighbor_count = 0;
        for (auto const &n : molecule.atomNeighbors(atom))
            if (neighbor_count < 8) neighbor_indices[neighbor_count++] = n->getIdx();

        for (unsigned int a = 0; a < neighbor_count; ++a) {
            for (unsigned int b = a + 1; b < neighbor_count; ++b) {
                unsigned int stretchbend_type;
                ForceFields::MMFF::MMFFStbn stretchbend_params;
                ForceFields::MMFF::MMFFBond bond_params[2];
                ForceFields::MMFF::MMFFAngle angle_params;
                if (!mmff_props.getMMFFStretchBendParams(mol, neighbor_indices[a], j, neighbor_indices[b],
                                                         stretchbend_type, stretchbend_params, bond_params,
                                                         angle_params))
                    continue;
                result.mmff_stretchbends.push_back(
                    {static_cast<std::uint8_t>(neighbor_indices[a]), static_cast<std::uint8_t>(j),
                     static_cast<std::uint8_t>(neighbor_indices[b]), 0, static_cast<float>(stretchbend_params.kbaIJK),
                     static_cast<float>(stretchbend_params.kbaKJI),
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
inline etkdg_prepared_molecule_t prepare_molecule(std::string const &smiles) {
    etkdg_prepared_molecule_t result {};

    std::unique_ptr<RDKit::RWMol> mol;
    try {
        mol.reset(RDKit::SmilesToMol(smiles));
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
        build_bounds_matrix(*mol, result.bounds_matrix);
        extract_chiral_centers(*mol, result.chiral_centers);
        extract_etk_torsions(*mol, result.etk_torsions);
        extract_distance_constraints(*mol, result.bounds_matrix, result.distance_constraints);
        build_etk_interaction_lists(result);
        extract_mmff94_params(*mol, result);
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
 *  @brief Batch of molecules packed into contiguous GPU-ready arrays.
 */
struct etkdg_molecule_batch_t {
    std::vector<float> bounds_matrices;                      ///< concatenated Nᵢ x Nᵢ
    std::vector<chiral_center_descriptor_t> chiral_centers;  ///< concatenated
    std::vector<etk_torsion_descriptor_t> etk_torsions;      ///< concatenated
    std::vector<distance_constraint_t> distance_constraints; ///< concatenated
    // MMFF94 parameters (stage 9)
    std::vector<mmff94_nonbonded_pair_t> mmff_nonbonded;      ///< concatenated
    std::vector<mmff94_bond_t> mmff_bonds;                    ///< concatenated
    std::vector<mmff94_angle_t> mmff_angles;                  ///< concatenated
    std::vector<mmff94_torsion_t> mmff_torsions;              ///< concatenated
    std::vector<mmff94_oop_t> mmff_oops;                      ///< concatenated
    std::vector<mmff94_stretchbend_t> mmff_stretchbends;      ///< concatenated
    std::vector<mmff94_atom_interaction_t> mmff_interactions; ///< per-atom MMFF94 interaction lists
    std::vector<mmff94_atom_offsets_t> mmff_atom_offsets;     ///< per-atom MMFF94 offsets
    std::vector<etk_atom_interaction_t> etk_interactions;     ///< per-atom ETK interaction lists
    std::vector<etk_atom_offsets_t> etk_atom_offsets;         ///< per-atom ETK offsets
    std::vector<etkdg_molecule_metadata_t> molecule_metadata; ///< one per molecule
    std::vector<std::uint32_t> input_indices;                 ///< packed molecule index -> original input index
    std::uint32_t num_molecules = 0;
    std::uint32_t max_atoms_per_molecule = 0;
};

/**
 *  @brief Prepare a batch of molecules from SMILES strings.
 *
 *  Phase 1 (OpenMP parallel): prepare each molecule independently.
 *  Phase 2 (sequential): pack into contiguous GPU-ready arrays.
 *
 *  @param[in] smiles_list Vector of SMILES strings to process.
 *  @return Batched molecule data with contiguous arrays for GPU upload.
 */
inline etkdg_molecule_batch_t prepare_molecule_batch(std::vector<std::string> const &smiles_list) {

    etkdg_molecule_batch_t batch {};
    // Phase 1: Prepare molecules in parallel. RDKit calls are thread-safe for independent molecules.
    std::vector<etkdg_prepared_molecule_t> prepared(smiles_list.size());
#pragma omp parallel for schedule(dynamic, 4)
    for (std::size_t i = 0; i < smiles_list.size(); ++i) { prepared[i] = prepare_molecule(smiles_list[i]); }

    // Pre-allocate batch vectors to avoid reallocation during the insert loop.
    {
        std::size_t total_bounds = 0, total_chirals = 0, total_etk = 0, total_dcons = 0;
        std::size_t total_nb = 0, total_bonds = 0, total_angles = 0, total_torsions = 0;
        std::size_t total_oops = 0, total_stretchbend = 0;
        std::size_t total_mmff_inter = 0, total_mmff_off = 0;
        std::size_t total_etk_inter = 0, total_etk_off = 0;
        for (auto const &p : prepared) {
            total_bounds += p.bounds_matrix.size();
            total_chirals += p.chiral_centers.size();
            total_etk += p.etk_torsions.size();
            total_dcons += p.distance_constraints.size();
            total_nb += p.mmff_nonbonded.size();
            total_bonds += p.mmff_bonds.size();
            total_angles += p.mmff_angles.size();
            total_torsions += p.mmff_torsions.size();
            total_oops += p.mmff_oops.size();
            total_stretchbend += p.mmff_stretchbends.size();
            total_mmff_inter += p.mmff_interactions.size();
            total_mmff_off += p.mmff_atom_offsets.size();
            total_etk_inter += p.etk_interactions.size();
            total_etk_off += p.etk_atom_offsets.size();
        }
        batch.bounds_matrices.reserve(total_bounds);
        batch.chiral_centers.reserve(total_chirals);
        batch.etk_torsions.reserve(total_etk);
        batch.distance_constraints.reserve(total_dcons);
        batch.mmff_nonbonded.reserve(total_nb);
        batch.mmff_bonds.reserve(total_bonds);
        batch.mmff_angles.reserve(total_angles);
        batch.mmff_torsions.reserve(total_torsions);
        batch.mmff_oops.reserve(total_oops);
        batch.mmff_stretchbends.reserve(total_stretchbend);
        batch.mmff_interactions.reserve(total_mmff_inter);
        batch.mmff_atom_offsets.reserve(total_mmff_off);
        batch.etk_interactions.reserve(total_etk_inter);
        batch.etk_atom_offsets.reserve(total_etk_off);
        batch.molecule_metadata.reserve(prepared.size());
        batch.input_indices.reserve(prepared.size());
    }

    // Phase 2: Pack into contiguous arrays sequentially without reallocations.
    for (std::uint32_t m = 0; m < static_cast<std::uint32_t>(prepared.size()); ++m) {
        if (prepared[m].num_atoms == 0) continue;

        batch.max_atoms_per_molecule = std::max(batch.max_atoms_per_molecule, prepared[m].num_atoms);

        // Set offsets before appending to the batched arrays.
        etkdg_molecule_metadata_t meta = prepared[m].metadata;
        meta.bounds_offset = static_cast<std::uint32_t>(batch.bounds_matrices.size());
        meta.chiral_offset = static_cast<std::uint32_t>(batch.chiral_centers.size());
        meta.torsion_offset = static_cast<std::uint32_t>(batch.etk_torsions.size());
        meta.constraint_offset = static_cast<std::uint32_t>(batch.distance_constraints.size());
        meta.mmff_nonbonded_offset = static_cast<std::uint32_t>(batch.mmff_nonbonded.size());
        meta.mmff_bond_offset = static_cast<std::uint32_t>(batch.mmff_bonds.size());
        meta.mmff_angle_offset = static_cast<std::uint32_t>(batch.mmff_angles.size());
        meta.mmff_torsion_offset = static_cast<std::uint32_t>(batch.mmff_torsions.size());
        meta.mmff_oop_offset = static_cast<std::uint32_t>(batch.mmff_oops.size());
        meta.mmff_stretchbend_offset = static_cast<std::uint32_t>(batch.mmff_stretchbends.size());
        meta.mmff_atom_offsets_start = static_cast<std::uint32_t>(batch.mmff_atom_offsets.size());
        meta.etk_interaction_offset = static_cast<std::uint32_t>(batch.etk_interactions.size());
        meta.etk_atom_offsets_start = static_cast<std::uint32_t>(batch.etk_atom_offsets.size());

        batch.molecule_metadata.push_back(meta);
        batch.input_indices.push_back(m);

        // Append the molecule data.
        batch.bounds_matrices.insert(batch.bounds_matrices.end(),
                                     std::make_move_iterator(prepared[m].bounds_matrix.begin()),
                                     std::make_move_iterator(prepared[m].bounds_matrix.end()));
        batch.chiral_centers.insert(batch.chiral_centers.end(),
                                    std::make_move_iterator(prepared[m].chiral_centers.begin()),
                                    std::make_move_iterator(prepared[m].chiral_centers.end()));
        batch.etk_torsions.insert(batch.etk_torsions.end(), std::make_move_iterator(prepared[m].etk_torsions.begin()),
                                  std::make_move_iterator(prepared[m].etk_torsions.end()));
        batch.distance_constraints.insert(batch.distance_constraints.end(),
                                          std::make_move_iterator(prepared[m].distance_constraints.begin()),
                                          std::make_move_iterator(prepared[m].distance_constraints.end()));
        // MMFF94 parameters
        batch.mmff_nonbonded.insert(batch.mmff_nonbonded.end(),
                                    std::make_move_iterator(prepared[m].mmff_nonbonded.begin()),
                                    std::make_move_iterator(prepared[m].mmff_nonbonded.end()));
        batch.mmff_bonds.insert(batch.mmff_bonds.end(), std::make_move_iterator(prepared[m].mmff_bonds.begin()),
                                std::make_move_iterator(prepared[m].mmff_bonds.end()));
        batch.mmff_angles.insert(batch.mmff_angles.end(), std::make_move_iterator(prepared[m].mmff_angles.begin()),
                                 std::make_move_iterator(prepared[m].mmff_angles.end()));
        batch.mmff_torsions.insert(batch.mmff_torsions.end(),
                                   std::make_move_iterator(prepared[m].mmff_torsions.begin()),
                                   std::make_move_iterator(prepared[m].mmff_torsions.end()));
        batch.mmff_oops.insert(batch.mmff_oops.end(), std::make_move_iterator(prepared[m].mmff_oops.begin()),
                               std::make_move_iterator(prepared[m].mmff_oops.end()));
        batch.mmff_stretchbends.insert(batch.mmff_stretchbends.end(),
                                       std::make_move_iterator(prepared[m].mmff_stretchbends.begin()),
                                       std::make_move_iterator(prepared[m].mmff_stretchbends.end()));
        // adjust per-atom per-type interaction offsets for the batched array position
        std::uint32_t const mmff_interactions_base = static_cast<std::uint32_t>(batch.mmff_interactions.size());
        batch.mmff_interactions.insert(batch.mmff_interactions.end(),
                                       std::make_move_iterator(prepared[m].mmff_interactions.begin()),
                                       std::make_move_iterator(prepared[m].mmff_interactions.end()));
        auto adjusted_offsets = prepared[m].mmff_atom_offsets;
        for (auto &off : adjusted_offsets) {
            off.nonbonded_start += mmff_interactions_base;
            off.bond_start += mmff_interactions_base;
            off.angle_start += mmff_interactions_base;
            off.torsion_start += mmff_interactions_base;
            off.oop_start += mmff_interactions_base;
            off.stretchbend_start += mmff_interactions_base;
        }
        batch.mmff_atom_offsets.insert(batch.mmff_atom_offsets.end(), adjusted_offsets.begin(), adjusted_offsets.end());

        // ETK interaction list: adjust offsets for batched array position
        std::uint32_t const etk_interactions_base = static_cast<std::uint32_t>(batch.etk_interactions.size());
        batch.etk_interactions.insert(batch.etk_interactions.end(),
                                      std::make_move_iterator(prepared[m].etk_interactions.begin()),
                                      std::make_move_iterator(prepared[m].etk_interactions.end()));
        auto etk_adjusted_offsets = prepared[m].etk_atom_offsets;
        for (auto &off : etk_adjusted_offsets) off.interaction_start += etk_interactions_base;
        batch.etk_atom_offsets.insert(batch.etk_atom_offsets.end(), etk_adjusted_offsets.begin(),
                                      etk_adjusted_offsets.end());
    }

    batch.num_molecules = static_cast<std::uint32_t>(batch.molecule_metadata.size());

    return batch;
}

} // namespace smallfield
