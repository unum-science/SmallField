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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <set>
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

/** Complete preprocessed molecule ready for GPU pipeline upload. */
struct etkdg_prepared_molecule_t {
    std::uint32_t num_atoms;
    std::vector<float> bounds_matrix;       ///< N x N flat, smoothed
    std::vector<float> initial_coordinates; ///< 4 x N SoA from SmallField metric embedding
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
    // Per-atom interaction lists built from the term arrays above.
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

    // Use RDKit's bounds matrix builder.
    auto* bounds_raw = new DistGeom::BoundsMatrix(num_atoms);
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

/** Read an upper-triangular bound from the flattened matrix. */
inline float read_upper_bound(std::vector<float> const& bounds, std::uint32_t num_atoms, std::uint32_t atom_a,
                              std::uint32_t atom_b) {
    std::uint32_t const row = std::min(atom_a, atom_b);
    std::uint32_t const col = std::max(atom_a, atom_b);
    return bounds[row * num_atoms + col];
}

/** Read a lower-triangular bound from the flattened matrix. */
inline float read_lower_bound(std::vector<float> const& bounds, std::uint32_t num_atoms, std::uint32_t atom_a,
                              std::uint32_t atom_b) {
    std::uint32_t const row = std::max(atom_a, atom_b);
    std::uint32_t const col = std::min(atom_a, atom_b);
    return bounds[row * num_atoms + col];
}

/** CPU reference triangle smoothing matching the GPU bound convention. */
inline void triangle_smooth_bounds_in_place(std::vector<float>& bounds, std::uint32_t num_atoms) {
    std::uint32_t const total_pairs = num_atoms * num_atoms;
    for (std::uint32_t k = 0; k < num_atoms; ++k) {
        for (std::uint32_t linear_index = 0; linear_index < total_pairs; ++linear_index) {
            std::uint32_t const i = linear_index / num_atoms;
            std::uint32_t const j = linear_index % num_atoms;
            if (i == j || i == k || j == k)
                continue;

            float const upper_ik = read_upper_bound(bounds, num_atoms, i, k);
            float const upper_kj = read_upper_bound(bounds, num_atoms, k, j);

            if (i < j) {
                float const candidate = upper_ik + upper_kj;
                float& current = bounds[i * num_atoms + j];
                if (candidate < current)
                    current = candidate;
            } else {
                float const lower_ik = read_lower_bound(bounds, num_atoms, i, k);
                float const lower_kj = read_lower_bound(bounds, num_atoms, k, j);
                float const c1 = lower_ik - upper_kj;
                float const c2 = lower_kj - upper_ik;
                float& current = bounds[i * num_atoms + j];
                current = std::max(current, std::max(c1, c2));
            }
        }
    }
}

/** Symmetric Jacobi eigendecomposition. Eigenvectors are returned column-major in `eigenvectors`. */
inline bool jacobi_symmetric_eigendecomposition(std::vector<double>& matrix, std::uint32_t num_atoms,
                                                std::vector<double>& eigenvectors, std::vector<double>& eigenvalues) {
    if (num_atoms == 0)
        return false;

    eigenvectors.assign(num_atoms * num_atoms, 0.0);
    for (std::uint32_t i = 0; i < num_atoms; ++i)
        eigenvectors[i * num_atoms + i] = 1.0;

    constexpr std::uint32_t max_sweeps = 64;
    constexpr double tolerance = 1.0e-10;

    for (std::uint32_t sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_off_diagonal = 0.0;
        std::uint32_t pivot_p = 0;
        std::uint32_t pivot_q = 0;

        for (std::uint32_t i = 0; i < num_atoms; ++i) {
            for (std::uint32_t j = i + 1; j < num_atoms; ++j) {
                double const value = std::fabs(matrix[i * num_atoms + j]);
                if (value > max_off_diagonal) {
                    max_off_diagonal = value;
                    pivot_p = i;
                    pivot_q = j;
                }
            }
        }

        if (max_off_diagonal < tolerance)
            break;

        double const app = matrix[pivot_p * num_atoms + pivot_p];
        double const aqq = matrix[pivot_q * num_atoms + pivot_q];
        double const apq = matrix[pivot_p * num_atoms + pivot_q];
        if (std::fabs(apq) < tolerance)
            continue;

        double const tau = (aqq - app) / (2.0 * apq);
        double const t =
            (tau >= 0.0) ? 1.0 / (tau + std::sqrt(1.0 + tau * tau)) : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
        double const c = 1.0 / std::sqrt(1.0 + t * t);
        double const s = t * c;

        for (std::uint32_t k = 0; k < num_atoms; ++k) {
            if (k == pivot_p || k == pivot_q)
                continue;
            double const aik = matrix[pivot_p * num_atoms + k];
            double const aqk = matrix[pivot_q * num_atoms + k];
            double const new_aik = c * aik - s * aqk;
            double const new_aqk = s * aik + c * aqk;
            matrix[pivot_p * num_atoms + k] = new_aik;
            matrix[k * num_atoms + pivot_p] = new_aik;
            matrix[pivot_q * num_atoms + k] = new_aqk;
            matrix[k * num_atoms + pivot_q] = new_aqk;
        }

        matrix[pivot_p * num_atoms + pivot_p] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
        matrix[pivot_q * num_atoms + pivot_q] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
        matrix[pivot_p * num_atoms + pivot_q] = 0.0;
        matrix[pivot_q * num_atoms + pivot_p] = 0.0;

        for (std::uint32_t k = 0; k < num_atoms; ++k) {
            double const vip = eigenvectors[k * num_atoms + pivot_p];
            double const viq = eigenvectors[k * num_atoms + pivot_q];
            eigenvectors[k * num_atoms + pivot_p] = c * vip - s * viq;
            eigenvectors[k * num_atoms + pivot_q] = s * vip + c * viq;
        }
    }

    eigenvalues.resize(num_atoms, 0.0);
    for (std::uint32_t i = 0; i < num_atoms; ++i)
        eigenvalues[i] = matrix[i * num_atoms + i];
    return true;
}

/** Build a SmallField-owned 4D metric embedding from the smoothed bounds matrix. */
inline bool build_metric_initial_coordinates(std::vector<float> const& bounds_matrix, std::uint32_t num_atoms,
                                             std::vector<float>& initial_coordinates) {
    if (num_atoms == 0)
        return false;

    std::vector<float> smoothed_bounds = bounds_matrix;
    triangle_smooth_bounds_in_place(smoothed_bounds, num_atoms);

    std::vector<double> squared_distances(num_atoms * num_atoms, 0.0);
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        for (std::uint32_t j = i + 1; j < num_atoms; ++j) {
            double const upper = static_cast<double>(smoothed_bounds[i * num_atoms + j]);
            double const lower = static_cast<double>(smoothed_bounds[j * num_atoms + i]);
            double const distance = 0.5 * (std::max(lower, 0.0) + std::max(upper, lower));
            double const squared = distance * distance;
            squared_distances[i * num_atoms + j] = squared;
            squared_distances[j * num_atoms + i] = squared;
        }
    }

    std::vector<double> row_means(num_atoms, 0.0);
    double grand_mean = 0.0;
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        double row_sum = 0.0;
        for (std::uint32_t j = 0; j < num_atoms; ++j)
            row_sum += squared_distances[i * num_atoms + j];
        row_means[i] = row_sum / static_cast<double>(num_atoms);
        grand_mean += row_means[i];
    }
    grand_mean /= static_cast<double>(num_atoms);

    std::vector<double> gram_matrix(num_atoms * num_atoms, 0.0);
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        for (std::uint32_t j = 0; j < num_atoms; ++j) {
            gram_matrix[i * num_atoms + j] =
                -0.5 * (squared_distances[i * num_atoms + j] - row_means[i] - row_means[j] + grand_mean);
        }
    }

    std::vector<double> eigenvectors;
    std::vector<double> eigenvalues;
    if (!jacobi_symmetric_eigendecomposition(gram_matrix, num_atoms, eigenvectors, eigenvalues))
        return false;

    std::vector<std::uint32_t> order(num_atoms);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(),
              [&](std::uint32_t lhs, std::uint32_t rhs) { return eigenvalues[lhs] > eigenvalues[rhs]; });

    initial_coordinates.assign(4 * num_atoms, 0.0f);
    constexpr double minimum_eigenvalue = 1.0e-8;
    for (std::uint32_t dim = 0; dim < 4 && dim < order.size(); ++dim) {
        std::uint32_t const eigen_index = order[dim];
        double const eigenvalue = std::max(eigenvalues[eigen_index], 0.0);
        if (eigenvalue < minimum_eigenvalue)
            continue;
        double const scale = std::sqrt(eigenvalue);
        for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
            initial_coordinates[dim * num_atoms + atom] =
                static_cast<float>(scale * eigenvectors[atom * num_atoms + eigen_index]);
        }
    }

    return true;
}

/**
 * Extract graph-only chiral center descriptors.
 *
 * Neighbors are sorted by decreasing CIP priority. With that ordering, the GPU
 * triple product over the top-three ligands has a stable sign convention:
 *   R -> negative volume, S -> positive volume.
 */
inline void extract_chiral_centers(RDKit::ROMol const& molecule,
                                   std::vector<chiral_center_descriptor_t>& chiral_centers) {
    chiral_centers.clear();
    auto& stereo_molecule = const_cast<RDKit::ROMol&>(molecule);
    RDKit::MolOps::assignStereochemistry(stereo_molecule, true, true, true);

    RDKit::UINT_VECT cip_ranks(molecule.getNumAtoms(), 0);
    RDKit::Chirality::assignAtomCIPRanks(molecule, cip_ranks);

    for (auto const& atom : molecule.atoms()) {
        auto chirality = atom->getChiralTag();
        if (chirality != RDKit::Atom::CHI_TETRAHEDRAL_CW && chirality != RDKit::Atom::CHI_TETRAHEDRAL_CCW)
            continue;
        if (!atom->hasProp("_CIPCode"))
            continue;

        chiral_center_descriptor_t descriptor{};
        descriptor.center_atom = static_cast<std::uint8_t>(atom->getIdx());

        if (atom->getDegree() < 3)
            continue;

        std::vector<std::uint32_t> ordered_neighbors;
        ordered_neighbors.reserve(4);
        for (auto const& neighbor : molecule.atomNeighbors(atom)) {
            ordered_neighbors.push_back(neighbor->getIdx());
        }
        std::sort(ordered_neighbors.begin(), ordered_neighbors.end(), [&](std::uint32_t lhs, std::uint32_t rhs) {
            if (cip_ranks[lhs] != cip_ranks[rhs])
                return cip_ranks[lhs] > cip_ranks[rhs];
            return lhs < rhs;
        });
        std::uint32_t const neighbor_count = static_cast<std::uint32_t>(ordered_neighbors.size());
        if (neighbor_count < 3)
            continue;

        for (std::uint32_t i = 0; i < 4 && i < neighbor_count; ++i)
            descriptor.neighbor_atoms[i] = static_cast<std::uint8_t>(ordered_neighbors[i]);

        std::string cip_code;
        atom->getProp("_CIPCode", cip_code);
        descriptor.target_sign = (cip_code == "S") ? 1 : -1;

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

        // Combine signs and force constants: Vn = signs[n] * forceConstants[n].
        for (std::size_t n = 0; n < 6 && n < force_constants.size(); ++n) {
            float sign = (n < signs.size()) ? static_cast<float>(signs[n]) : 1.0f;
            torsion.fourier_coefficients[n] = sign * static_cast<float>(force_constants[n]);
        }

        torsions.push_back(torsion);
    }

    // If no CSD torsions are found, fall back to generic 3-fold barriers for rotatable bonds.
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
 * Extract MMFF94 parameters and build per-atom interaction lists.
 * Uses flat boolean matrices for O(1) exclusion/1-4 lookup (no std::set).
 * No temporary heap allocations — all output written directly to result arrays.
 */
inline void extract_mmff94_params(RDKit::ROMol const& molecule, etkdg_prepared_molecule_t& result) {
    result.mmff_nonbonded.clear();
    result.mmff_bonds.clear();
    result.mmff_angles.clear();
    result.mmff_torsions.clear();

    auto& mol = const_cast<RDKit::ROMol&>(molecule);
    RDKit::MMFF::MMFFMolProperties mmff_props(mol);
    if (!mmff_props.isValid())
        return;

    std::uint32_t const num_atoms = molecule.getNumAtoms();

    // Use a flat exclusion matrix for 1-2 and 1-3 pairs.
    // Use a flat 1-4 matrix for atoms exactly 3 bonds apart.
    std::vector<bool> excluded(num_atoms * num_atoms, false);
    std::vector<bool> is_1_4(num_atoms * num_atoms, false);

    // Mark 1-2 bonds as excluded.
    for (auto const& bond : molecule.bonds()) {
        unsigned int a = bond->getBeginAtomIdx(), b = bond->getEndAtomIdx();
        excluded[a * num_atoms + b] = true;
        excluded[b * num_atoms + a] = true;
    }

    // Mark 1-3 pairs as excluded and detect 1-4 pairs from torsion terminals.
    for (auto const& atom : molecule.atoms()) {
        unsigned int center = atom->getIdx();
        // Collect neighbors by inline iteration without a temporary vector.
        unsigned int nb_indices[8]; // max 8 neighbors (typical max valence)
        unsigned int nb_count = 0;
        for (auto const& nb : molecule.atomNeighbors(atom)) {
            if (nb_count < 8)
                nb_indices[nb_count++] = nb->getIdx();
        }
        // 1-3 exclusions: all pairs of neighbors
        for (unsigned int a = 0; a < nb_count; ++a) {
            for (unsigned int b = a + 1; b < nb_count; ++b) {
                excluded[nb_indices[a] * num_atoms + nb_indices[b]] = true;
                excluded[nb_indices[b] * num_atoms + nb_indices[a]] = true;
            }
        }
    }

    // 1-4 pairs from torsion terminals: for each bond j-k, pair (i,l) where i∈N(j)\k, l∈N(k)\j
    for (auto const& bond : molecule.bonds()) {
        unsigned int j = bond->getBeginAtomIdx(), k = bond->getEndAtomIdx();
        for (auto const& nb_j : molecule.atomNeighbors(molecule.getAtomWithIdx(j))) {
            if (nb_j->getIdx() == k)
                continue;
            for (auto const& nb_k : molecule.atomNeighbors(molecule.getAtomWithIdx(k))) {
                if (nb_k->getIdx() == j)
                    continue;
                unsigned int i = nb_j->getIdx(), l = nb_k->getIdx();
                is_1_4[i * num_atoms + l] = true;
                is_1_4[l * num_atoms + i] = true;
            }
        }
    }

    // Cache partial charges in a single pass.
    float charges[256]; // stack-allocated, max 256 atoms
    for (unsigned int i = 0; i < num_atoms && i < 256; ++i)
        charges[i] = static_cast<float>(mmff_props.getMMFFPartialCharge(i));

    // Build nonbonded pairs with O(N^2) iteration and O(1) exclusion lookup.
    for (unsigned int i = 0; i < num_atoms; ++i) {
        for (unsigned int j = i + 1; j < num_atoms; ++j) {
            if (excluded[i * num_atoms + j])
                continue;

            ForceFields::MMFF::MMFFVdWRijstarEps vdw_params;
            if (!mmff_props.getMMFFVdWParams(i, j, vdw_params))
                continue;

            result.mmff_nonbonded.push_back({static_cast<std::uint8_t>(i), static_cast<std::uint8_t>(j),
                                             static_cast<std::uint8_t>(is_1_4[i * num_atoms + j] ? 1 : 0), 0,
                                             static_cast<float>(vdw_params.epsilon),
                                             static_cast<float>(vdw_params.R_ij_star), charges[i] * charges[j]});
        }
    }

    // Build bond terms.
    for (auto const& bond : molecule.bonds()) {
        unsigned int i = bond->getBeginAtomIdx(), j = bond->getEndAtomIdx();
        unsigned int bond_type;
        ForceFields::MMFF::MMFFBond bp;
        if (!mmff_props.getMMFFBondStretchParams(mol, i, j, bond_type, bp))
            continue;
        result.mmff_bonds.push_back({static_cast<std::uint8_t>(i),
                                     static_cast<std::uint8_t>(j),
                                     {0, 0},
                                     static_cast<float>(bp.kb),
                                     static_cast<float>(bp.r0),
                                     0.0f});
    }

    // Build angle terms by iterating atoms as centers and pairing their neighbors.
    for (auto const& atom : molecule.atoms()) {
        unsigned int j = atom->getIdx();
        unsigned int nonbonded[8];
        unsigned int neighbor_count = 0;
        for (auto const& n : molecule.atomNeighbors(atom))
            if (neighbor_count < 8)
                nonbonded[neighbor_count++] = n->getIdx();

        for (unsigned int a = 0; a < neighbor_count; ++a) {
            for (unsigned int b = a + 1; b < neighbor_count; ++b) {
                unsigned int angle_type;
                ForceFields::MMFF::MMFFAngle ap;
                if (!mmff_props.getMMFFAngleBendParams(mol, nonbonded[a], j, nonbonded[b], angle_type, ap))
                    continue;
                result.mmff_angles.push_back({static_cast<std::uint8_t>(nonbonded[a]), static_cast<std::uint8_t>(j),
                                              static_cast<std::uint8_t>(nonbonded[b]), 0, static_cast<float>(ap.ka),
                                              static_cast<float>(ap.theta0 * M_PI / 180.0), 0.0f});
            }
        }
    }

    // Build torsion terms.
    for (auto const& bond : molecule.bonds()) {
        unsigned int j = bond->getBeginAtomIdx(), k = bond->getEndAtomIdx();
        for (auto const& nj : molecule.atomNeighbors(molecule.getAtomWithIdx(j))) {
            if (nj->getIdx() == k)
                continue;
            for (auto const& nk : molecule.atomNeighbors(molecule.getAtomWithIdx(k))) {
                if (nk->getIdx() == j)
                    continue;
                unsigned int torsion_type;
                ForceFields::MMFF::MMFFTor tp;
                if (!mmff_props.getMMFFTorsionParams(mol, nj->getIdx(), j, k, nk->getIdx(), torsion_type, tp))
                    continue;
                result.mmff_torsions.push_back({static_cast<std::uint8_t>(nj->getIdx()), static_cast<std::uint8_t>(j),
                                                static_cast<std::uint8_t>(k), static_cast<std::uint8_t>(nk->getIdx()),
                                                static_cast<float>(tp.V1), static_cast<float>(tp.V2),
                                                static_cast<float>(tp.V3)});
            }
        }
    }

    // Build out-of-plane bending terms for each sp2 center with 3 or more neighbors.
    for (auto const& atom : molecule.atoms()) {
        unsigned int j = atom->getIdx();
        unsigned int neighbor_indices[8];
        unsigned int neighbor_count = 0;
        for (auto const& n : molecule.atomNeighbors(atom))
            if (neighbor_count < 8)
                neighbor_indices[neighbor_count++] = n->getIdx();
        if (neighbor_count < 3)
            continue;

        for (unsigned int a = 0; a < neighbor_count; ++a) {
            // Treat neighbor_indices[a] as the out-of-plane atom and the other two as the plane atoms.
            unsigned int plane[2];
            unsigned int plane_count = 0;
            for (unsigned int b = 0; b < neighbor_count && plane_count < 2; ++b)
                if (b != a)
                    plane[plane_count++] = neighbor_indices[b];
            if (plane_count < 2)
                continue;

            ForceFields::MMFF::MMFFOop oop_params;
            if (!mmff_props.getMMFFOopBendParams(mol, neighbor_indices[a], j, plane[0], plane[1], oop_params))
                continue;
            result.mmff_oops.push_back({static_cast<std::uint8_t>(neighbor_indices[a]),
                                        static_cast<std::uint8_t>(j),
                                        static_cast<std::uint8_t>(plane[0]),
                                        static_cast<std::uint8_t>(plane[1]),
                                        static_cast<float>(oop_params.koop),
                                        {0.0f, 0.0f}});
        }
    }

    // Build stretch-bend terms by reusing angle topology.
    for (auto const& atom : molecule.atoms()) {
        unsigned int j = atom->getIdx();
        unsigned int neighbor_indices[8];
        unsigned int neighbor_count = 0;
        for (auto const& n : molecule.atomNeighbors(atom))
            if (neighbor_count < 8)
                neighbor_indices[neighbor_count++] = n->getIdx();

        for (unsigned int a = 0; a < neighbor_count; ++a) {
            for (unsigned int b = a + 1; b < neighbor_count; ++b) {
                unsigned int stbn_type;
                ForceFields::MMFF::MMFFStbn stbn_params;
                ForceFields::MMFF::MMFFBond bond_params[2];
                ForceFields::MMFF::MMFFAngle angle_params;
                if (!mmff_props.getMMFFStretchBendParams(mol, neighbor_indices[a], j, neighbor_indices[b], stbn_type,
                                                         stbn_params, bond_params, angle_params))
                    continue;
                result.mmff_stretchbends.push_back(
                    {static_cast<std::uint8_t>(neighbor_indices[a]), static_cast<std::uint8_t>(j),
                     static_cast<std::uint8_t>(neighbor_indices[b]), 0, static_cast<float>(stbn_params.kbaIJK),
                     static_cast<float>(stbn_params.kbaKJI), static_cast<float>(angle_params.theta0 * M_PI / 180.0),
                     static_cast<float>(bond_params[0].r0), static_cast<float>(bond_params[1].r0), 0.0f});
            }
        }
    }

    // Build per-atom interaction lists in two passes without vector-of-vectors.
    auto const& nonbonded = result.mmff_nonbonded;
    auto const& bonds = result.mmff_bonds;
    auto const& angles = result.mmff_angles;
    auto const& torsions = result.mmff_torsions;
    auto const& oops = result.mmff_oops;
    auto const& stretchbends = result.mmff_stretchbends;

    // Pass 1: Count interactions per atom.
    std::vector<std::uint32_t> counts(num_atoms, 0);
    for (auto const& p : nonbonded) {
        counts[p.atom_i]++;
        counts[p.atom_j]++;
    }
    for (auto const& b : bonds) {
        counts[b.atom_i]++;
        counts[b.atom_j]++;
    }
    for (auto const& a : angles) {
        counts[a.atom_i]++;
        counts[a.atom_j]++;
        counts[a.atom_k]++;
    }
    for (auto const& t : torsions) {
        counts[t.atom_i]++;
        counts[t.atom_j]++;
        counts[t.atom_k]++;
        counts[t.atom_l]++;
    }
    for (auto const& o : oops) {
        counts[o.atom_i]++;
        counts[o.atom_j]++;
        counts[o.atom_k]++;
        counts[o.atom_l]++;
    }
    for (auto const& s : stretchbends) {
        counts[s.atom_i]++;
        counts[s.atom_j]++;
        counts[s.atom_k]++;
    }

    // Compute offsets from the per-atom counts.
    result.mmff_atom_offsets.resize(num_atoms);
    std::uint32_t running = 0;
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        result.mmff_atom_offsets[a] = {running, counts[a]};
        running += counts[a];
    }

    // Pass 2: Fill interaction entries by using per-atom write cursors.
    std::uint32_t const total_interactions = running;
    result.mmff_interactions.resize(total_interactions);
    std::vector<std::uint32_t> cursors(num_atoms);
    for (std::uint32_t a = 0; a < num_atoms; ++a)
        cursors[a] = result.mmff_atom_offsets[a].interaction_start;

    for (std::uint32_t t = 0; t < nonbonded.size(); ++t) {
        result.mmff_interactions[cursors[nonbonded[t].atom_i]++] = {t, 0, 0};
        result.mmff_interactions[cursors[nonbonded[t].atom_j]++] = {t, 0, 1};
    }
    for (std::uint32_t t = 0; t < bonds.size(); ++t) {
        result.mmff_interactions[cursors[bonds[t].atom_i]++] = {t, 1, 0};
        result.mmff_interactions[cursors[bonds[t].atom_j]++] = {t, 1, 1};
    }
    for (std::uint32_t t = 0; t < angles.size(); ++t) {
        result.mmff_interactions[cursors[angles[t].atom_i]++] = {t, 2, 0};
        result.mmff_interactions[cursors[angles[t].atom_j]++] = {t, 3, 1};
        result.mmff_interactions[cursors[angles[t].atom_k]++] = {t, 2, 2};
    }
    for (std::uint32_t t = 0; t < torsions.size(); ++t) {
        result.mmff_interactions[cursors[torsions[t].atom_i]++] = {t, 4, 0};
        result.mmff_interactions[cursors[torsions[t].atom_j]++] = {t, 4, 1};
        result.mmff_interactions[cursors[torsions[t].atom_k]++] = {t, 4, 2};
        result.mmff_interactions[cursors[torsions[t].atom_l]++] = {t, 4, 3};
    }
    for (std::uint32_t t = 0; t < oops.size(); ++t) {
        result.mmff_interactions[cursors[oops[t].atom_i]++] = {t, 5, 0};
        result.mmff_interactions[cursors[oops[t].atom_j]++] = {t, 5, 1};
        result.mmff_interactions[cursors[oops[t].atom_k]++] = {t, 5, 2};
        result.mmff_interactions[cursors[oops[t].atom_l]++] = {t, 5, 3};
    }
    for (std::uint32_t t = 0; t < stretchbends.size(); ++t) {
        result.mmff_interactions[cursors[stretchbends[t].atom_i]++] = {t, 6, 0};
        result.mmff_interactions[cursors[stretchbends[t].atom_j]++] = {t, 6, 1};
        result.mmff_interactions[cursors[stretchbends[t].atom_k]++] = {t, 6, 2};
    }
}

/**
 * All-in-one: SMILES string to fully prepared molecule (stages 0-1 complete).
 */
inline etkdg_prepared_molecule_t prepare_molecule(std::string const& smiles) {
    etkdg_prepared_molecule_t result{};

    std::unique_ptr<RDKit::RWMol> mol;
    try {
        mol.reset(RDKit::SmilesToMol(smiles));
    } catch (...) {
        return result;
    }
    if (!mol)
        return result;

    try {
        RDKit::MolOps::addHs(*mol);
    } catch (...) {
        return result;
    }
    result.num_atoms = mol->getNumAtoms();
    if (result.num_atoms > 128)
        return result; // Too large for the single-warp pipeline.

    try {
        build_bounds_matrix(*mol, result.bounds_matrix);
        build_metric_initial_coordinates(result.bounds_matrix, result.num_atoms, result.initial_coordinates);
        extract_chiral_centers(*mol, result.chiral_centers);
        extract_etk_torsions(*mol, result.etk_torsions);
        extract_distance_constraints(*mol, result.bounds_matrix, result.distance_constraints);
        extract_mmff94_params(*mol, result);
    } catch (...) {
        result.num_atoms = 0; // mark as failed
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
    result.metadata.initial_coordinates_offset = 0;
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

    return result;
}

/** Batch of molecules packed into contiguous GPU-ready arrays. */
struct etkdg_molecule_batch_t {
    std::vector<float> bounds_matrices;                      ///< concatenated Nᵢ x Nᵢ
    std::vector<float> initial_coordinates;                  ///< concatenated 3 x Nᵢ SoA embeddings
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
    std::vector<mmff94_atom_interaction_t> mmff_interactions; ///< per-atom interaction lists
    std::vector<mmff94_atom_offsets_t> mmff_atom_offsets;     ///< per-atom offsets
    std::vector<etkdg_molecule_metadata_t> molecule_metadata; ///< one per molecule
    std::vector<std::uint32_t> input_indices;                 ///< packed molecule index -> original input index
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
    // Phase 1: Prepare molecules in parallel. RDKit calls are thread-safe for independent molecules.
    std::vector<etkdg_prepared_molecule_t> prepared(smiles_list.size());
#pragma omp parallel for schedule(dynamic, 4)
    for (std::size_t i = 0; i < smiles_list.size(); ++i) {
        prepared[i] = prepare_molecule(smiles_list[i]);
    }

    // Pre-allocate batch vectors to avoid reallocation during the insert loop.
    {
        std::size_t total_bounds = 0, total_initial = 0, total_chirals = 0, total_etk = 0, total_dcons = 0;
        std::size_t total_nb = 0, total_bonds = 0, total_angles = 0, total_torsions = 0;
        std::size_t total_oops = 0, total_stbn = 0;
        std::size_t total_inter = 0, total_off = 0;
        for (auto const& p : prepared) {
            total_bounds += p.bounds_matrix.size();
            total_initial += p.initial_coordinates.size();
            total_chirals += p.chiral_centers.size();
            total_etk += p.etk_torsions.size();
            total_dcons += p.distance_constraints.size();
            total_nb += p.mmff_nonbonded.size();
            total_bonds += p.mmff_bonds.size();
            total_angles += p.mmff_angles.size();
            total_torsions += p.mmff_torsions.size();
            total_oops += p.mmff_oops.size();
            total_stbn += p.mmff_stretchbends.size();
            total_inter += p.mmff_interactions.size();
            total_off += p.mmff_atom_offsets.size();
        }
        batch.bounds_matrices.reserve(total_bounds);
        batch.initial_coordinates.reserve(total_initial);
        batch.chiral_centers.reserve(total_chirals);
        batch.etk_torsions.reserve(total_etk);
        batch.distance_constraints.reserve(total_dcons);
        batch.mmff_nonbonded.reserve(total_nb);
        batch.mmff_bonds.reserve(total_bonds);
        batch.mmff_angles.reserve(total_angles);
        batch.mmff_torsions.reserve(total_torsions);
        batch.mmff_oops.reserve(total_oops);
        batch.mmff_stretchbends.reserve(total_stbn);
        batch.mmff_interactions.reserve(total_inter);
        batch.mmff_atom_offsets.reserve(total_off);
        batch.molecule_metadata.reserve(prepared.size());
        batch.input_indices.reserve(prepared.size());
    }

    // Phase 2: Pack into contiguous arrays sequentially without reallocations.
    for (std::uint32_t m = 0; m < static_cast<std::uint32_t>(prepared.size()); ++m) {
        if (prepared[m].num_atoms == 0)
            continue;

        batch.max_atoms_per_molecule = std::max(batch.max_atoms_per_molecule, prepared[m].num_atoms);

        // Set offsets before appending to the batched arrays.
        etkdg_molecule_metadata_t meta = prepared[m].metadata;
        meta.bounds_offset = static_cast<std::uint32_t>(batch.bounds_matrices.size());
        meta.initial_coordinates_offset = static_cast<std::uint32_t>(batch.initial_coordinates.size());
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

        batch.molecule_metadata.push_back(meta);
        batch.input_indices.push_back(m);

        // Append the molecule data.
        batch.bounds_matrices.insert(batch.bounds_matrices.end(), prepared[m].bounds_matrix.begin(),
                                     prepared[m].bounds_matrix.end());
        batch.initial_coordinates.insert(batch.initial_coordinates.end(), prepared[m].initial_coordinates.begin(),
                                         prepared[m].initial_coordinates.end());
        batch.chiral_centers.insert(batch.chiral_centers.end(), prepared[m].chiral_centers.begin(),
                                    prepared[m].chiral_centers.end());
        batch.etk_torsions.insert(batch.etk_torsions.end(), prepared[m].etk_torsions.begin(),
                                  prepared[m].etk_torsions.end());
        batch.distance_constraints.insert(batch.distance_constraints.end(), prepared[m].distance_constraints.begin(),
                                          prepared[m].distance_constraints.end());
        // MMFF94 parameters
        batch.mmff_nonbonded.insert(batch.mmff_nonbonded.end(), prepared[m].mmff_nonbonded.begin(),
                                    prepared[m].mmff_nonbonded.end());
        batch.mmff_bonds.insert(batch.mmff_bonds.end(), prepared[m].mmff_bonds.begin(), prepared[m].mmff_bonds.end());
        batch.mmff_angles.insert(batch.mmff_angles.end(), prepared[m].mmff_angles.begin(),
                                 prepared[m].mmff_angles.end());
        batch.mmff_torsions.insert(batch.mmff_torsions.end(), prepared[m].mmff_torsions.begin(),
                                   prepared[m].mmff_torsions.end());
        batch.mmff_oops.insert(batch.mmff_oops.end(), prepared[m].mmff_oops.begin(), prepared[m].mmff_oops.end());
        batch.mmff_stretchbends.insert(batch.mmff_stretchbends.end(), prepared[m].mmff_stretchbends.begin(),
                                       prepared[m].mmff_stretchbends.end());
        // Adjust per-atom interaction offsets for the batched array position.
        std::uint32_t const interactions_base = static_cast<std::uint32_t>(batch.mmff_interactions.size());
        batch.mmff_interactions.insert(batch.mmff_interactions.end(), prepared[m].mmff_interactions.begin(),
                                       prepared[m].mmff_interactions.end());
        auto adjusted_offsets = prepared[m].mmff_atom_offsets;
        for (auto& off : adjusted_offsets)
            off.interaction_start += interactions_base;
        batch.mmff_atom_offsets.insert(batch.mmff_atom_offsets.end(), adjusted_offsets.begin(), adjusted_offsets.end());
    }

    batch.num_molecules = static_cast<std::uint32_t>(batch.molecule_metadata.size());

    return batch;
}

} // namespace smallfield
