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

/**
 * Per-molecule metadata for the ETKDG pipeline.
 * One per molecule in the batch; offsets point into concatenated arrays.
 */
struct etkdg_molecule_metadata_t {
    std::uint32_t num_atoms;
    std::uint32_t num_chiral_centers;
    std::uint32_t num_etk_torsions;
    std::uint32_t num_distance_constraints;
    std::uint32_t bounds_offset;
    std::uint32_t chiral_offset;
    std::uint32_t torsion_offset;
    std::uint32_t constraint_offset;
    std::uint32_t num_mmff_nonbonded;
    std::uint32_t num_mmff_bonds;
    std::uint32_t num_mmff_angles;
    std::uint32_t num_mmff_torsions;
    std::uint32_t num_mmff_oops;
    std::uint32_t num_mmff_stretchbends;
    std::uint32_t mmff_nonbonded_offset;
    std::uint32_t mmff_bond_offset;
    std::uint32_t mmff_angle_offset;
    std::uint32_t mmff_torsion_offset;
    std::uint32_t mmff_oop_offset;
    std::uint32_t mmff_stretchbend_offset;
    std::uint32_t mmff_atom_offsets_start;
    std::uint32_t etk_interaction_offset;  ///< offset into batched ETK interaction list
    std::uint32_t etk_atom_offsets_start;  ///< offset into batched ETK atom offsets
    std::uint32_t padding[1];
};

/**
 * Chiral center descriptor.
 * Stores the central atom, its four CIP-priority-ordered neighbors,
 * and the expected sign of the signed tetrahedral volume.
 *
 * Validation: V = (r_b - r_a) . ((r_c - r_a) x (r_d - r_a))
 * where a = center, b,c,d = neighbor_atoms[0..2]; sign(V) must equal target_sign.
 */
struct alignas(8) chiral_center_descriptor_t {
    std::uint8_t center_atom;       ///< central atom index within molecule
    std::uint8_t neighbor_atoms[4]; ///< four CIP-ordered neighbor atom indices
    std::int8_t target_sign;        ///< +1 (R) or -1 (S) configuration
    std::uint8_t padding[2];
};

/**
 * ETK torsion term from CSD (Cambridge Structural Database) preferences.
 *
 * E_torsion = Σₙ₌₁⁶ Vₙ·cos(n·φ)
 * where φ is the dihedral angle defined by atom_indices[0]-[1]-[2]-[3].
 */
struct alignas(32) etk_torsion_descriptor_t {
    std::uint8_t atom_indices[4]; ///< dihedral atoms i-j-k-l within molecule
    std::uint8_t padding[4];
    float fourier_coefficients[6]; ///< V₁ through V₆ (kcal/mol)
};

/**
 * Distance constraint for distance geometry and ETK refinement stages.
 *
 * E = penalty_weight · max(0, dᵢⱼ - upper)² + penalty_weight · max(0, lower - dᵢⱼ)²
 * where dᵢⱼ = ‖rᵢ - rⱼ‖
 */
struct alignas(16) distance_constraint_t {
    std::uint8_t atom_index_i;    ///< first atom index within molecule
    std::uint8_t atom_index_j;    ///< second atom index within molecule
    std::uint8_t constraint_type; ///< 0 = 1-2 bond, 1 = 1-3 angle-derived, 2 = custom
    std::uint8_t padding;
    float lower_bound_angstrom; ///< minimum allowed interatomic distance
    float upper_bound_angstrom; ///< maximum allowed interatomic distance
    float penalty_weight;       ///< force constant for bound violation penalty
};

/**
 * MMFF94 nonbonded pair (VdW + electrostatic). 16 bytes.
 * Each pair stored once (not symmetric like neighbor_t).
 */
struct alignas(16) mmff94_nonbonded_pair_t {
    std::uint8_t atom_i; ///< first atom index within molecule
    std::uint8_t atom_j; ///< second atom index within molecule
    std::uint8_t is_1_4; ///< 1 if 1-4 pair (scaled electrostatics), 0 otherwise
    std::uint8_t padding;
    float epsilon;        ///< VdW well depth (kcal/mol)
    float r_star;         ///< VdW equilibrium distance (angstroms)
    float charge_product; ///< qi * qj (precomputed)
};

/** MMFF94 bond stretching term. E = 0.5·143.9325·kb·Δr²·(1 + cs·Δr + 7/12·cs²·Δr²) */
struct alignas(16) mmff94_bond_t {
    std::uint8_t atom_i;
    std::uint8_t atom_j;
    std::uint8_t padding[2];
    float force_constant;       ///< kb (mdyn/angstrom)
    float equilibrium_distance; ///< r0 (angstroms)
    float padding_float;
};

/** MMFF94 angle bending term. E = 0.5·0.043844·ka·Δθ²·(1 + cb·Δθ) */
struct alignas(16) mmff94_angle_t {
    std::uint8_t atom_i;
    std::uint8_t atom_j; ///< central atom
    std::uint8_t atom_k;
    std::uint8_t padding;
    float force_constant;    ///< ka (mdyn·angstrom/rad²)
    float equilibrium_angle; ///< θ₀ (radians)
    float padding_float;
};

/** MMFF94 torsion term. E = 0.5·(V₁·(1+cos(φ)) + V₂·(1−cos(2φ)) + V₃·(1+cos(3φ))) */
struct alignas(16) mmff94_torsion_t {
    std::uint8_t atom_i;
    std::uint8_t atom_j;
    std::uint8_t atom_k;
    std::uint8_t atom_l;
    float v1; ///< first barrier height (kcal/mol)
    float v2; ///< second barrier height (kcal/mol)
    float v3; ///< third barrier height (kcal/mol)
};

/**
 * MMFF94 out-of-plane bending. E = 0.5·0.043844·koop·chi² (chi in degrees).
 * Wilson angle: chi = angle between bond j→i and the plane j-k-l.
 */
struct alignas(16) mmff94_oop_t {
    std::uint8_t atom_i;  ///< out-of-plane atom
    std::uint8_t atom_j;  ///< central sp2 atom
    std::uint8_t atom_k;  ///< in-plane neighbor 1
    std::uint8_t atom_l;  ///< in-plane neighbor 2
    float force_constant; ///< koop (mdyn·angstrom/rad²)
    float padding[2];
};

/** MMFF94 stretch-bend coupling. E = 2.5121·(kba_ijk·Δr_ij + kba_kji·Δr_kj)·Δθ */
struct alignas(32) mmff94_stretchbend_t {
    std::uint8_t atom_i; ///< end atom 1
    std::uint8_t atom_j; ///< central atom
    std::uint8_t atom_k; ///< end atom 2
    std::uint8_t padding;
    float kba_ijk;                 ///< stretch-bend force constant (bond i-j)
    float kba_kji;                 ///< stretch-bend force constant (bond k-j)
    float equilibrium_angle;       ///< θ₀ (radians)
    float equilibrium_distance_ij; ///< r₀_ij (angstroms)
    float equilibrium_distance_kj; ///< r₀_kj (angstroms)
    float padding_float;
};

/**
 * Per-atom MMFF94 interaction descriptor for the per-atom adjacency list.
 * Each entry says: "atom X participates in term Y with role Z".
 * interaction_type: 0=nonbonded, 1=bond, 2=angle(end), 3=angle(center), 4=torsion, 5=oop, 6=stretch-bend
 */
struct mmff94_atom_interaction_t {
    std::uint32_t term_index;      ///< index into the type-specific parameter array
    std::uint8_t interaction_type; ///< 0=nb, 1=bond, 2=angle(end), 3=angle(ctr), 4=tor, 5=oop, 6=stbn
    std::uint8_t role;             ///< this atom's role (0=i, 1=j, 2=k, 3=l)
};

/** Per-atom offset into the MMFF94 interaction list. */
struct mmff94_atom_offsets_t {
    std::uint32_t interaction_start;
    std::uint32_t interaction_count;
};

/** Per-atom ETK interaction descriptor. interaction_type: 0=torsion, 1=distance_constraint. */
struct etk_atom_interaction_t {
    std::uint32_t term_index;
    std::uint8_t interaction_type; ///< 0=torsion, 1=distance_constraint
    std::uint8_t role;             ///< torsion: 0-3 (i,j,k,l), constraint: 0=atom_i, 1=atom_j
};

/** Per-atom offset into the ETK interaction list. */
struct etk_atom_offsets_t {
    std::uint32_t interaction_start;
    std::uint32_t interaction_count;
};

/**
 * Bounds matrix layout convention:
 *
 * Full N x N float matrix, row-major.
 * bounds[i*N + j] with i < j  stores the upper bound distance (angstroms)
 * bounds[i*N + j] with i > j  stores the lower bound distance (angstroms)
 * bounds[i*N + i]             = 0.0
 *
 * This matches RDKit's DistGeom::BoundsMatrix convention.
 */

} // namespace smallfield
