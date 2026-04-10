/**
 * @file etkdg_types.cuh
 * @brief Data structures for the GPU ETKDG conformer generation pipeline.
 *
 * Defines the types used to pass preprocessed molecular data from CPU to GPU:
 * - Molecule metadata (atom counts, offsets into batched arrays)
 * - Chiral center descriptors (CIP-ordered neighbors, target sign)
 * - ETK torsion descriptors (CSD Fourier coefficients)
 * - Distance constraints (1-2 bond, 1-3 angle-derived bounds)
 *
 * All structs use `alignas` (C++17) for portable alignment.
 */

#pragma once

#include <cstdint>

namespace smallfield {

/// Per-molecule metadata for the ETKDG pipeline.
/// One per molecule in the batch; offsets point into concatenated arrays.
struct alignas(16) etkdg_molecule_metadata_t {
    std::uint32_t num_atoms;                ///< total atoms including explicit H
    std::uint32_t num_chiral_centers;       ///< chiral centers requiring validation
    std::uint32_t num_etk_torsions;         ///< CSD-derived torsion Fourier terms
    std::uint32_t num_distance_constraints; ///< 1-2 bond and 1-3 angle constraints
    std::uint32_t bounds_offset;            ///< byte offset into batched bounds matrices
    std::uint32_t chiral_offset;            ///< element offset into batched chirals
    std::uint32_t torsion_offset;           ///< element offset into batched torsions
    std::uint32_t constraint_offset;        ///< element offset into batched constraints
};

/// Chiral center descriptor.
/// Stores the central atom, its four CIP-priority-ordered neighbors,
/// and the expected sign of the signed tetrahedral volume.
///
/// Validation: V = (r_b - r_a) . ((r_c - r_a) x (r_d - r_a))
/// where a = center, b,c,d = neighbor_atoms[0..2]; sign(V) must equal target_sign.
struct alignas(8) chiral_center_descriptor_t {
    std::uint8_t center_atom;       ///< central atom index within molecule
    std::uint8_t neighbor_atoms[4]; ///< four CIP-ordered neighbor atom indices
    std::int8_t target_sign;        ///< +1 (R) or -1 (S) configuration
    std::uint8_t padding[2];
};

/// ETK torsion term from CSD (Cambridge Structural Database) preferences.
///
/// E_torsion = Σₙ₌₁⁶ Vₙ·cos(n·φ)
/// where φ is the dihedral angle defined by atom_indices[0]-[1]-[2]-[3].
struct alignas(32) etk_torsion_descriptor_t {
    std::uint8_t atom_indices[4]; ///< dihedral atoms i-j-k-l within molecule
    std::uint8_t padding[4];
    float fourier_coefficients[6]; ///< V₁ through V₆ (kcal/mol)
};

/// Distance constraint for distance geometry and ETK refinement stages.
///
/// E = penalty_weight · max(0, dᵢⱼ - upper)² + penalty_weight · max(0, lower - dᵢⱼ)²
/// where dᵢⱼ = ‖rᵢ - rⱼ‖
struct alignas(16) distance_constraint_t {
    std::uint8_t atom_index_i;    ///< first atom index within molecule
    std::uint8_t atom_index_j;    ///< second atom index within molecule
    std::uint8_t constraint_type; ///< 0 = 1-2 bond, 1 = 1-3 angle-derived, 2 = custom
    std::uint8_t padding;
    float lower_bound_angstrom; ///< minimum allowed interatomic distance
    float upper_bound_angstrom; ///< maximum allowed interatomic distance
    float penalty_weight;       ///< force constant for bound violation penalty
};

/// Bounds matrix layout convention:
///
/// Full N x N float matrix, row-major.
///   bounds[i*N + j] with i < j  stores the upper bound distance (angstroms)
///   bounds[i*N + j] with i > j  stores the lower bound distance (angstroms)
///   bounds[i*N + i]             = 0.0
///
/// This matches RDKit's DistGeom::BoundsMatrix convention.

} // namespace smallfield
