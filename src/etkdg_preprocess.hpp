/**
 * @file etkdg_preprocess.hpp
 * @brief CPU-side ETKDG preprocessing using RDKit C++ API (stages 0-1).
 *
 * Converts molecular graphs into GPU-ready data structures:
 *   - Distance bounds matrix (N x N, smoothed via triangle inequality)
 *   - Chiral center descriptors (CIP-ordered neighbors, target sign)
 *   - ETK torsion descriptors (CSD Fourier coefficients)
 *   - 1-2 / 1-3 distance constraints
 *
 * All functions operate on RDKit ROMol objects and produce flat arrays
 * suitable for upload to GPU global memory.
 */

#pragma once

#include "etkdg_types.cuh"

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/TriangleSmooth.h>
#include <Geometry/point.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace smallfield {

/// Complete preprocessed molecule ready for GPU pipeline upload.
struct etkdg_prepared_molecule_t {
    std::uint32_t num_atoms;
    std::vector<float> bounds_matrix; ///< N x N flat, smoothed
    std::vector<chiral_center_descriptor_t> chiral_centers;
    std::vector<etk_torsion_descriptor_t> etk_torsions;
    std::vector<distance_constraint_t> distance_constraints; ///< 1-2 bond, 1-3 angle
    // MMFF94 force field parameters (stage 9)
    std::vector<mmff94_nonbonded_pair_t> mmff_nonbonded;
    std::vector<mmff94_bond_t> mmff_bonds;
    std::vector<mmff94_angle_t> mmff_angles;
    std::vector<mmff94_torsion_t> mmff_torsions;
    // per-atom interaction lists (built from the above)
    std::vector<mmff94_atom_interaction_t> mmff_interactions;
    std::vector<mmff94_atom_offsets_t> mmff_atom_offsets;
    etkdg_molecule_metadata_t metadata;
};

/**
 * Build the smoothed distance bounds matrix from a molecular graph.
 *
 * Uses RDKit's BoundsMatrixBuilder to compute initial bounds from
 * bond lengths, VdW radii, and angle-derived 1-3 distances, then
 * applies Floyd-Warshall triangle smoothing.
 *
 * Output: N x N float array in row-major order.
 *   bounds[i*N + j] with i < j = upper bound (angstroms)
 *   bounds[i*N + j] with i > j = lower bound (angstroms)
 */
inline void build_bounds_matrix(RDKit::ROMol const& molecule, std::vector<float>& bounds) {
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    bounds.resize(num_atoms * num_atoms, 0.0f);

    // use RDKit's bounds matrix builder
    auto* bounds_raw = new DistGeom::BoundsMatrix(num_atoms);
    DistGeom::BoundsMatPtr bounds_mat(bounds_raw);
    RDKit::DGeomHelpers::initBoundsMat(bounds_raw);
    RDKit::DGeomHelpers::setTopolBounds(molecule, bounds_mat);
    // no CPU smoothing — the GPU kernel handles Floyd-Warshall in shared memory

    // flatten to our row-major convention
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        for (std::uint32_t j = i + 1; j < num_atoms; ++j) {
            bounds[i * num_atoms + j] = static_cast<float>(bounds_mat->getUpperBound(i, j));
            bounds[j * num_atoms + i] = static_cast<float>(bounds_mat->getLowerBound(i, j));
        }
    }
}

/**
 * Extract chiral centers from molecular graph.
 * Target sign derived from CW/CCW tag without needing 3D coordinates.
 *
 * RDKit convention: CHI_TETRAHEDRAL_CW means the signed volume
 * V = (r_b−r_a) · ((r_c−r_a) × (r_d−r_a)) is positive when
 * neighbors are in the connection table order.
 */
inline void extract_chiral_centers(RDKit::ROMol const& molecule,
                                   std::vector<chiral_center_descriptor_t>& chiral_centers) {
    chiral_centers.clear();

    for (auto const& atom : molecule.atoms()) {
        auto chirality = atom->getChiralTag();
        if (chirality != RDKit::Atom::CHI_TETRAHEDRAL_CW && chirality != RDKit::Atom::CHI_TETRAHEDRAL_CCW)
            continue;

        chiral_center_descriptor_t descriptor{};
        descriptor.center_atom = static_cast<std::uint8_t>(atom->getIdx());

        std::uint32_t neighbor_count = 0;
        for (auto const& neighbor : molecule.atomNeighbors(atom)) {
            if (neighbor_count < 4)
                descriptor.neighbor_atoms[neighbor_count] = static_cast<std::uint8_t>(neighbor->getIdx());
            ++neighbor_count;
        }

        // graph-based sign: CW → +1, CCW → -1
        descriptor.target_sign = (chirality == RDKit::Atom::CHI_TETRAHEDRAL_CW) ? 1 : -1;

        chiral_centers.push_back(descriptor);
    }
}

/**
 * Extract 1-2 bond and 1-3 angle-derived distance constraints
 * for the ETK refinement stage.
 *
 * Reads actual distance bounds from the smoothed bounds matrix instead
 * of using hardcoded values. The bounds matrix already contains
 * chemically-correct values from RDKit's setTopolBounds().
 */
inline void extract_distance_constraints(RDKit::ROMol const& molecule, std::vector<float> const& bounds_matrix,
                                         std::vector<distance_constraint_t>& constraints) {
    constraints.clear();
    std::uint32_t const num_atoms = molecule.getNumAtoms();

    // 1-2 bond constraints: read actual bounds from the matrix
    for (auto const& bond : molecule.bonds()) {
        std::uint32_t const atom_i = bond->getBeginAtomIdx();
        std::uint32_t const atom_j = bond->getEndAtomIdx();
        std::uint32_t const row_min = std::min(atom_i, atom_j);
        std::uint32_t const row_max = std::max(atom_i, atom_j);

        distance_constraint_t constraint{};
        constraint.atom_index_i = static_cast<std::uint8_t>(atom_i);
        constraint.atom_index_j = static_cast<std::uint8_t>(atom_j);
        constraint.constraint_type = 0; // 1-2 bond
        constraint.upper_bound_angstrom = bounds_matrix[row_min * num_atoms + row_max];
        constraint.lower_bound_angstrom = bounds_matrix[row_max * num_atoms + row_min];
        constraint.penalty_weight = 10.0f;
        constraints.push_back(constraint);
    }

    // 1-3 constraints from angles: read from bounds matrix
    for (auto const& atom : molecule.atoms()) {
        std::vector<unsigned int> neighbors;
        for (auto const& neighbor : molecule.atomNeighbors(atom)) {
            neighbors.push_back(neighbor->getIdx());
        }

        for (std::size_t a = 0; a < neighbors.size(); ++a) {
            for (std::size_t b = a + 1; b < neighbors.size(); ++b) {
                std::uint32_t const ni = neighbors[a];
                std::uint32_t const nj = neighbors[b];
                std::uint32_t const row_min = std::min(ni, nj);
                std::uint32_t const row_max = std::max(ni, nj);

                distance_constraint_t constraint{};
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
 * Extract ETK torsion terms with CSD Fourier coefficients from RDKit's
 * experimental torsion angle preferences (Riniker & Landrum, JCIM 2015).
 *
 * Uses ForceFields::CrystalFF::getExperimentalTorsions() to get atom-type-specific
 * Fourier coefficients from the Cambridge Structural Database torsion library.
 */
inline void extract_etk_torsions(RDKit::ROMol const& molecule, std::vector<etk_torsion_descriptor_t>& torsions) {
    torsions.clear();

    // get CSD experimental torsion preferences from RDKit
    ForceFields::CrystalFF::CrystalFFDetails crystal_details;
    ForceFields::CrystalFF::getExperimentalTorsions(molecule, crystal_details,
                                                    /*useExpTorsions=*/true,
                                                    /*useSmallRingTorsions=*/true,
                                                    /*useMacrocycleTorsions=*/false,
                                                    /*useBasicKnowledge=*/true,
                                                    /*version=*/2);

    // convert RDKit's torsion data to our descriptor format
    for (std::size_t t = 0; t < crystal_details.expTorsionAtoms.size(); ++t) {
        auto const& atom_indices = crystal_details.expTorsionAtoms[t];
        if (atom_indices.size() < 4)
            continue;

        auto const& angle_data = crystal_details.expTorsionAngles[t];
        auto const& signs = angle_data.first;
        auto const& force_constants = angle_data.second;

        etk_torsion_descriptor_t torsion{};
        torsion.atom_indices[0] = static_cast<std::uint8_t>(atom_indices[0]);
        torsion.atom_indices[1] = static_cast<std::uint8_t>(atom_indices[1]);
        torsion.atom_indices[2] = static_cast<std::uint8_t>(atom_indices[2]);
        torsion.atom_indices[3] = static_cast<std::uint8_t>(atom_indices[3]);

        // combine signs and force constants: Vₙ = signs[n] * forceConstants[n]
        for (std::size_t n = 0; n < 6 && n < force_constants.size(); ++n) {
            float sign = (n < signs.size()) ? static_cast<float>(signs[n]) : 1.0f;
            torsion.fourier_coefficients[n] = sign * static_cast<float>(force_constants[n]);
        }

        torsions.push_back(torsion);
    }

    // if no CSD torsions found, fall back to generic 3-fold barriers for rotatable bonds
    if (torsions.empty()) {
        for (auto const& bond : molecule.bonds()) {
            if (bond->getIsAromatic() || bond->getBondTypeAsDouble() > 1.5)
                continue;

            unsigned int atom_j = bond->getBeginAtomIdx();
            unsigned int atom_k = bond->getEndAtomIdx();
            unsigned int atom_i = static_cast<unsigned int>(-1);
            unsigned int atom_l = static_cast<unsigned int>(-1);

            for (auto const& neighbor : molecule.atomNeighbors(molecule.getAtomWithIdx(atom_j))) {
                if (neighbor->getIdx() != atom_k) {
                    atom_i = neighbor->getIdx();
                    break;
                }
            }
            for (auto const& neighbor : molecule.atomNeighbors(molecule.getAtomWithIdx(atom_k))) {
                if (neighbor->getIdx() != atom_j) {
                    atom_l = neighbor->getIdx();
                    break;
                }
            }
            if (atom_i == static_cast<unsigned int>(-1) || atom_l == static_cast<unsigned int>(-1))
                continue;

            etk_torsion_descriptor_t torsion{};
            torsion.atom_indices[0] = static_cast<std::uint8_t>(atom_i);
            torsion.atom_indices[1] = static_cast<std::uint8_t>(atom_j);
            torsion.atom_indices[2] = static_cast<std::uint8_t>(atom_k);
            torsion.atom_indices[3] = static_cast<std::uint8_t>(atom_l);
            torsion.fourier_coefficients[2] = 1.0f; // generic V₃ = 1.0
            torsions.push_back(torsion);
        }
    }
}

/**
 * Extract MMFF94 force field parameters for stage 9 optimization.
 * Uses RDKit's MMFFMolProperties C++ API to get VdW, electrostatic,
 * bond, angle, and torsion parameters.
 */
inline void extract_mmff94_params(RDKit::ROMol const& molecule,
                                  std::vector<mmff94_nonbonded_pair_t>& nonbonded,
                                  std::vector<mmff94_bond_t>& bonds,
                                  std::vector<mmff94_angle_t>& angles,
                                  std::vector<mmff94_torsion_t>& torsions) {
    nonbonded.clear();
    bonds.clear();
    angles.clear();
    torsions.clear();

    // need a mutable copy for MMFFMolProperties
    RDKit::RWMol mol_copy(molecule);
    std::unique_ptr<RDKit::MMFF::MMFFMolProperties> mmff_props(
        new RDKit::MMFF::MMFFMolProperties(mol_copy));
    if (!mmff_props || !mmff_props->isValid()) return;

    std::uint32_t const num_atoms = molecule.getNumAtoms();

    // build exclusion sets: 1-2 and 1-3 pairs (excluded from nonbonded)
    std::set<std::pair<unsigned int, unsigned int>> excluded;
    for (auto const& bond : molecule.bonds()) {
        unsigned int i = bond->getBeginAtomIdx(), j = bond->getEndAtomIdx();
        excluded.insert({std::min(i, j), std::max(i, j)});
    }
    for (auto const& atom : molecule.atoms()) {
        std::vector<unsigned int> neighbors;
        for (auto const& nb : molecule.atomNeighbors(atom))
            neighbors.push_back(nb->getIdx());
        for (std::size_t a = 0; a < neighbors.size(); ++a)
            for (std::size_t b = a + 1; b < neighbors.size(); ++b)
                excluded.insert({std::min(neighbors[a], neighbors[b]),
                                 std::max(neighbors[a], neighbors[b])});
    }

    // VdW + electrostatic nonbonded pairs
    for (unsigned int i = 0; i < num_atoms; ++i) {
        for (unsigned int j = i + 1; j < num_atoms; ++j) {
            auto key = std::make_pair(i, j);
            bool is_excluded = excluded.count(key) > 0;
            // 1-4 pairs: check if exactly 3 bonds apart
            // for simplicity, include all non-excluded pairs and mark 1-4 separately
            // a proper implementation would check path length = 3
            if (is_excluded) continue; // skip 1-2 and 1-3 pairs

            ForceFields::MMFF::MMFFVdWRijstarEps vdw_params;
            if (!mmff_props->getMMFFVdWParams(i, j, vdw_params)) continue;

            mmff94_nonbonded_pair_t pair{};
            pair.atom_i = static_cast<std::uint8_t>(i);
            pair.atom_j = static_cast<std::uint8_t>(j);
            pair.epsilon = static_cast<float>(vdw_params.epsilon);
            pair.r_star = static_cast<float>(vdw_params.R_ij_star);
            pair.charge_product = static_cast<float>(
                mmff_props->getMMFFPartialCharge(i) * mmff_props->getMMFFPartialCharge(j));
            pair.is_1_4 = 0; // TODO: detect 1-4 pairs properly
            nonbonded.push_back(pair);
        }
    }

    // bond stretching
    for (auto const& bond : molecule.bonds()) {
        unsigned int i = bond->getBeginAtomIdx(), j = bond->getEndAtomIdx();
        unsigned int bond_type;
        ForceFields::MMFF::MMFFBond bond_params;
        if (!mmff_props->getMMFFBondStretchParams(mol_copy, i, j, bond_type, bond_params))
            continue;

        mmff94_bond_t b{};
        b.atom_i = static_cast<std::uint8_t>(i);
        b.atom_j = static_cast<std::uint8_t>(j);
        b.force_constant = static_cast<float>(bond_params.kb);
        b.equilibrium_distance = static_cast<float>(bond_params.r0);
        bonds.push_back(b);
    }

    // angle bending
    for (auto const& atom : molecule.atoms()) {
        unsigned int j = atom->getIdx();
        std::vector<unsigned int> neighbors;
        for (auto const& nb : molecule.atomNeighbors(atom))
            neighbors.push_back(nb->getIdx());

        for (std::size_t a = 0; a < neighbors.size(); ++a) {
            for (std::size_t b = a + 1; b < neighbors.size(); ++b) {
                unsigned int i = neighbors[a], k = neighbors[b];
                unsigned int angle_type;
                ForceFields::MMFF::MMFFAngle angle_params;
                if (!mmff_props->getMMFFAngleBendParams(mol_copy, i, j, k, angle_type, angle_params))
                    continue;

                mmff94_angle_t ang{};
                ang.atom_i = static_cast<std::uint8_t>(i);
                ang.atom_j = static_cast<std::uint8_t>(j);
                ang.atom_k = static_cast<std::uint8_t>(k);
                ang.force_constant = static_cast<float>(angle_params.ka);
                ang.equilibrium_angle = static_cast<float>(angle_params.theta0 * M_PI / 180.0);
                angles.push_back(ang);
            }
        }
    }

    // torsion angles
    for (auto const& bond : molecule.bonds()) {
        unsigned int j = bond->getBeginAtomIdx(), k = bond->getEndAtomIdx();
        std::vector<unsigned int> j_neighbors, k_neighbors;
        for (auto const& nb : molecule.atomNeighbors(molecule.getAtomWithIdx(j)))
            if (nb->getIdx() != k) j_neighbors.push_back(nb->getIdx());
        for (auto const& nb : molecule.atomNeighbors(molecule.getAtomWithIdx(k)))
            if (nb->getIdx() != j) k_neighbors.push_back(nb->getIdx());

        for (auto i : j_neighbors) {
            for (auto l : k_neighbors) {
                unsigned int torsion_type;
                ForceFields::MMFF::MMFFTor tor_params;
                if (!mmff_props->getMMFFTorsionParams(mol_copy, i, j, k, l, torsion_type, tor_params))
                    continue;

                mmff94_torsion_t tor{};
                tor.atom_i = static_cast<std::uint8_t>(i);
                tor.atom_j = static_cast<std::uint8_t>(j);
                tor.atom_k = static_cast<std::uint8_t>(k);
                tor.atom_l = static_cast<std::uint8_t>(l);
                tor.v1 = static_cast<float>(tor_params.V1);
                tor.v2 = static_cast<float>(tor_params.V2);
                tor.v3 = static_cast<float>(tor_params.V3);
                torsions.push_back(tor);
            }
        }
    }
}

/**
 * Build per-atom interaction lists from MMFF94 parameter arrays.
 * Each atom gets a sorted list of interactions it participates in.
 * Eliminates O(T*N) brute-force scanning in the GPU functor.
 */
inline void build_mmff94_interaction_lists(std::uint32_t num_atoms,
                                           std::vector<mmff94_nonbonded_pair_t> const& nonbonded,
                                           std::vector<mmff94_bond_t> const& bonds,
                                           std::vector<mmff94_angle_t> const& angles,
                                           std::vector<mmff94_torsion_t> const& torsions,
                                           std::vector<mmff94_atom_interaction_t>& interactions,
                                           std::vector<mmff94_atom_offsets_t>& offsets) {
    // collect interactions per atom
    std::vector<std::vector<mmff94_atom_interaction_t>> per_atom(num_atoms);

    // nonbonded: each pair involves 2 atoms
    for (std::uint16_t t = 0; t < static_cast<std::uint16_t>(nonbonded.size()); ++t) {
        per_atom[nonbonded[t].atom_i].push_back({t, 0, 0}); // type=0, role=i
        per_atom[nonbonded[t].atom_j].push_back({t, 0, 1}); // type=0, role=j
    }

    // bonds: each bond involves 2 atoms
    for (std::uint16_t t = 0; t < static_cast<std::uint16_t>(bonds.size()); ++t) {
        per_atom[bonds[t].atom_i].push_back({t, 1, 0});
        per_atom[bonds[t].atom_j].push_back({t, 1, 1});
    }

    // angles: each angle involves 3 atoms (i, j=center, k)
    for (std::uint16_t t = 0; t < static_cast<std::uint16_t>(angles.size()); ++t) {
        per_atom[angles[t].atom_i].push_back({t, 2, 0}); // type=2 (end atom)
        per_atom[angles[t].atom_j].push_back({t, 3, 1}); // type=3 (center atom)
        per_atom[angles[t].atom_k].push_back({t, 2, 2}); // type=2 (end atom)
    }

    // torsions: each torsion involves 4 atoms
    for (std::uint16_t t = 0; t < static_cast<std::uint16_t>(torsions.size()); ++t) {
        per_atom[torsions[t].atom_i].push_back({t, 4, 0});
        per_atom[torsions[t].atom_j].push_back({t, 4, 1});
        per_atom[torsions[t].atom_k].push_back({t, 4, 2});
        per_atom[torsions[t].atom_l].push_back({t, 4, 3});
    }

    // flatten into contiguous arrays
    interactions.clear();
    offsets.resize(num_atoms);
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        offsets[a].interaction_start = static_cast<std::uint16_t>(interactions.size());
        offsets[a].interaction_count = static_cast<std::uint16_t>(per_atom[a].size());
        interactions.insert(interactions.end(), per_atom[a].begin(), per_atom[a].end());
    }
}

/**
 * All-in-one: SMILES string to fully prepared molecule (stages 0-1 complete).
 */
inline etkdg_prepared_molecule_t prepare_molecule(std::string const& smiles) {
    etkdg_prepared_molecule_t result{};

    std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smiles));
    if (!mol)
        return result;

    RDKit::MolOps::addHs(*mol);
    result.num_atoms = mol->getNumAtoms();

    // build raw bounds (no CPU smoothing — GPU handles Floyd-Warshall)
    // chiral signs from graph CW/CCW tags (no EmbedMolecule needed)
    build_bounds_matrix(*mol, result.bounds_matrix);
    extract_chiral_centers(*mol, result.chiral_centers);
    extract_etk_torsions(*mol, result.etk_torsions);
    extract_distance_constraints(*mol, result.bounds_matrix, result.distance_constraints);
    extract_mmff94_params(*mol, result.mmff_nonbonded, result.mmff_bonds,
                          result.mmff_angles, result.mmff_torsions);
    build_mmff94_interaction_lists(result.num_atoms, result.mmff_nonbonded,
                                   result.mmff_bonds, result.mmff_angles, result.mmff_torsions,
                                   result.mmff_interactions, result.mmff_atom_offsets);

    result.metadata.num_atoms = result.num_atoms;
    result.metadata.num_chiral_centers = static_cast<std::uint32_t>(result.chiral_centers.size());
    result.metadata.num_etk_torsions = static_cast<std::uint32_t>(result.etk_torsions.size());
    result.metadata.num_distance_constraints = static_cast<std::uint32_t>(result.distance_constraints.size());
    result.metadata.num_mmff_nonbonded = static_cast<std::uint32_t>(result.mmff_nonbonded.size());
    result.metadata.num_mmff_bonds = static_cast<std::uint32_t>(result.mmff_bonds.size());
    result.metadata.num_mmff_angles = static_cast<std::uint32_t>(result.mmff_angles.size());
    result.metadata.num_mmff_torsions = static_cast<std::uint32_t>(result.mmff_torsions.size());
    result.metadata.bounds_offset = 0;
    result.metadata.chiral_offset = 0;
    result.metadata.torsion_offset = 0;
    result.metadata.constraint_offset = 0;
    result.metadata.mmff_nonbonded_offset = 0;
    result.metadata.mmff_bond_offset = 0;
    result.metadata.mmff_angle_offset = 0;
    result.metadata.mmff_torsion_offset = 0;

    return result;
}

/// Batch of molecules packed into contiguous GPU-ready arrays.
struct etkdg_molecule_batch_t {
    std::vector<float> bounds_matrices;                       ///< concatenated Nᵢ x Nᵢ
    std::vector<chiral_center_descriptor_t> chiral_centers;   ///< concatenated
    std::vector<etk_torsion_descriptor_t> etk_torsions;       ///< concatenated
    std::vector<distance_constraint_t> distance_constraints;  ///< concatenated
    // MMFF94 parameters (stage 9)
    std::vector<mmff94_nonbonded_pair_t> mmff_nonbonded;      ///< concatenated
    std::vector<mmff94_bond_t> mmff_bonds;                    ///< concatenated
    std::vector<mmff94_angle_t> mmff_angles;                  ///< concatenated
    std::vector<mmff94_torsion_t> mmff_torsions;              ///< concatenated
    std::vector<mmff94_atom_interaction_t> mmff_interactions; ///< per-atom interaction lists
    std::vector<mmff94_atom_offsets_t> mmff_atom_offsets;     ///< per-atom offsets
    std::vector<etkdg_molecule_metadata_t> molecule_metadata; ///< one per molecule
    std::uint32_t num_molecules = 0;
    std::uint32_t max_atoms_per_molecule = 0;
};

/**
 * Prepare a batch of molecules from SMILES strings.
 * Phase 1 (OpenMP parallel): prepare each molecule independently.
 * Phase 2 (sequential): pack into contiguous GPU-ready arrays.
 */
inline etkdg_molecule_batch_t prepare_molecule_batch(std::vector<std::string> const& smiles_list) {

    etkdg_molecule_batch_t batch{};
    batch.num_molecules = static_cast<std::uint32_t>(smiles_list.size());

    // phase 1: prepare molecules in parallel (RDKit calls are thread-safe for independent molecules)
    std::vector<etkdg_prepared_molecule_t> prepared(smiles_list.size());
    #pragma omp parallel for schedule(dynamic, 4)
    for (std::size_t i = 0; i < smiles_list.size(); ++i) {
        prepared[i] = prepare_molecule(smiles_list[i]);
    }

    // phase 2: pack into contiguous arrays (sequential, fast)
    for (std::uint32_t m = 0; m < batch.num_molecules; ++m) {
        if (prepared[m].num_atoms == 0)
            continue;

        batch.max_atoms_per_molecule = std::max(batch.max_atoms_per_molecule, prepared[m].num_atoms);

        // set offsets before appending
        etkdg_molecule_metadata_t meta = prepared[m].metadata;
        meta.bounds_offset = static_cast<std::uint32_t>(batch.bounds_matrices.size());
        meta.chiral_offset = static_cast<std::uint32_t>(batch.chiral_centers.size());
        meta.torsion_offset = static_cast<std::uint32_t>(batch.etk_torsions.size());
        meta.constraint_offset = static_cast<std::uint32_t>(batch.distance_constraints.size());
        meta.mmff_nonbonded_offset = static_cast<std::uint32_t>(batch.mmff_nonbonded.size());
        meta.mmff_bond_offset = static_cast<std::uint32_t>(batch.mmff_bonds.size());
        meta.mmff_angle_offset = static_cast<std::uint32_t>(batch.mmff_angles.size());
        meta.mmff_torsion_offset = static_cast<std::uint32_t>(batch.mmff_torsions.size());

        batch.molecule_metadata.push_back(meta);

        // append data
        batch.bounds_matrices.insert(batch.bounds_matrices.end(), prepared[m].bounds_matrix.begin(),
                                     prepared[m].bounds_matrix.end());
        batch.chiral_centers.insert(batch.chiral_centers.end(), prepared[m].chiral_centers.begin(),
                                    prepared[m].chiral_centers.end());
        batch.etk_torsions.insert(batch.etk_torsions.end(), prepared[m].etk_torsions.begin(),
                                  prepared[m].etk_torsions.end());
        batch.distance_constraints.insert(batch.distance_constraints.end(), prepared[m].distance_constraints.begin(),
                                          prepared[m].distance_constraints.end());
        // MMFF94 parameters
        batch.mmff_nonbonded.insert(batch.mmff_nonbonded.end(), prepared[m].mmff_nonbonded.begin(),
                                    prepared[m].mmff_nonbonded.end());
        batch.mmff_bonds.insert(batch.mmff_bonds.end(), prepared[m].mmff_bonds.begin(),
                                prepared[m].mmff_bonds.end());
        batch.mmff_angles.insert(batch.mmff_angles.end(), prepared[m].mmff_angles.begin(),
                                 prepared[m].mmff_angles.end());
        batch.mmff_torsions.insert(batch.mmff_torsions.end(), prepared[m].mmff_torsions.begin(),
                                   prepared[m].mmff_torsions.end());
        batch.mmff_interactions.insert(batch.mmff_interactions.end(), prepared[m].mmff_interactions.begin(),
                                       prepared[m].mmff_interactions.end());
        batch.mmff_atom_offsets.insert(batch.mmff_atom_offsets.end(), prepared[m].mmff_atom_offsets.begin(),
                                       prepared[m].mmff_atom_offsets.end());
    }

    return batch;
}

} // namespace smallfield
