/**
 *  @file src/smallfield/types/etkdg_types.hpp
 *  @brief Preprocessed molecular data as the GPU ETKDG pipeline reads it, from CPU to GPU.
 *
 *  The bounds matrix is a full N × N row-major float matrix in angstroms, RDKit's `DistGeom::BoundsMatrix`
 *  convention.
 *
 *  @verbatim
 *  bounds[i * N + j], i < j   upper bound distance
 *  bounds[i * N + j], i > j   lower bound distance
 *  bounds[i * N + i]          0.0
 *  @endverbatim
 */

#pragma once

#include <cstdint> // `std::uint8_t`, `std::uint32_t`

namespace smallfield::types {

/** Positional index of an atom within a multi-atom interaction term: bonds first/second, angles
 *  first=end, second=center, third=end, torsions and OOP first through fourth in atom-list order. */
enum atom_role_t : std::uint8_t {
    atom_role_first_k = 0,
    atom_role_second_k = 1,
    atom_role_third_k = 2,
    atom_role_fourth_k = 3,
};

enum mmff94_interaction_type_t : std::uint8_t {
    mmff94_nonbonded_k = 0,
    mmff94_bond_k = 1,
    mmff94_angle_end_k = 2,
    mmff94_angle_center_k = 3,
    mmff94_torsion_k = 4,
    mmff94_out_of_plane_k = 5,
    mmff94_stretch_bend_k = 6,
};

enum etk_interaction_type_t : std::uint8_t {
    etk_torsion_k = 0,
    etk_distance_constraint_k = 1,
};

/**
 *  @brief Per-molecule metadata for the ETKDG pipeline, one per molecule in the batch.
 *
 *  Every offset counts @b elements, never bytes, and indexes the region rather than the pool:
 *  `smallfield::preprocess::carve_wave_slice` hands out each region as a typed subspan, so the byte
 *  arithmetic lives there and nothing here is ever added to a pool base. A kernel reads
 *  `region[metadata.bounds_offset]`, four times too far if the two are mixed up. `bond_distance_offset`
 *  indexes the batched per-molecule N × N `bond_distance` and `etk_interaction_offset` the batched ETK
 *  interaction list; `atom_base_offset` is this molecule's first atom in every batched per-atom array, the
 *  MMFF VdW params, the MMFF partial charges and the MMFF and ETK interaction CSRs sharing one
 *  concatenation order. `num_heavy_atoms`, `num_bonds` and `molecular_weight` are optional readbacks,
 *  bubbled to the caller through `context_t::generate_conformers`'s output spans and read by no GPU kernel
 *  — host-side passthroughs computed once per molecule from the post-`AddHs` RDKit `RWMol`, where heavy
 *  excludes hydrogen, the bond count includes it, and the weight is the exact one, in Daltons.
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
    std::uint32_t num_mmff_bonds;
    std::uint32_t num_mmff_angles;
    std::uint32_t num_mmff_torsions;
    std::uint32_t num_mmff_oops;
    std::uint32_t num_mmff_stretchbends;
    std::uint32_t atom_base_offset;
    std::uint32_t bond_distance_offset;
    std::uint32_t mmff_bond_offset;
    std::uint32_t mmff_angle_offset;
    std::uint32_t mmff_torsion_offset;
    std::uint32_t mmff_oop_offset;
    std::uint32_t mmff_stretchbend_offset;
    std::uint32_t etk_interaction_offset;
    std::uint32_t num_heavy_atoms;
    std::uint32_t num_bonds;
    float molecular_weight;
};
static_assert(sizeof(etkdg_molecule_metadata_t) == 96);

/*
 *  The floor a chiral center's volume must reach, in cubic angstrom, as a magnitude — handedness is
 *  `chiral_center_descriptor_t::target_sign` and lives in exactly one place. Both are RDKit's
 *  `DistGeom::ChiralSet` bounds, and one floor serves both the restoring force in
 *  `evaluate_distance_geometry_energy` and the `chiral_volume_slack_k` gate in `count_invalid_chiral_centers_`.
 *  @b Not `MIN_TETRAHEDRAL_CHIRAL_VOL`, the 0.5 over unit vectors in `Embedder.cpp`, which RDKit applies
 *  only to untagged degree-4 carbon and nitrogen, whose `ChiralSet` carries bounds of 0 and 0 and which
 *  never enter a force field — a planarity check on atoms with no stereochemistry to preserve. RDKit's
 *  `volUpperBound` has no counterpart here: at 100 cubic angstrom a center would need bonds over three
 *  angstroms long to reach it, so the upper branch of its penalty never runs and costs a register.
 */

/** Floor on |V| for a center with four bonded neighbors; RDKit's `volLowerBound`. */
inline constexpr float min_four_coordinate_volume_k = 5.0f;
/** Floor on |V| for a center whose fourth substituent is a lone pair; RDKit's github #5883 case. */
inline constexpr float min_three_coordinate_volume_k = 2.0f;

/**
 *  @brief One chiral center: the central atom `center_atom`, its `neighbor_atoms` in ascending atom-index
 *  order, the `target_sign` the signed tetrahedral volume takes over that order, and the `minimum_volume`
 *  magnitude it may not fall below, in cubic angstrom.
 *
 *  The ordering is index-based, not CIP-based: `target_sign` comes from the tetrahedral tag and the parity
 *  of this permutation against the bond order the tag is defined on, so any deterministic order works as
 *  long as derivation and validation use the same one. A three-coordinate center stores its three bonded
 *  neighbors and the center itself as the fourth, which is what RDKit substitutes for a missing bond. The
 *  sign and the floor are two halves of one statement, neither derivable from the other: a sign alone
 *  admits an arbitrarily small volume, and at zero it admits @b either answer, so a coplanar center
 *  satisfies whichever target it was given and the penalty meant to restore it reads no violation.
 *
 *  @verbatim
 *  Validation: V = dot(r_0 - r_3, cross(r_1 - r_3, r_2 - r_3))
 *  where r_i = neighbor_atoms[i], so the fourth neighbor is the pivot and the center is not read.
 *  sign(V) must equal target_sign, and |V| must reach minimum_volume.
 *  @endverbatim
 */
struct alignas(8) chiral_center_descriptor_t {
    std::uint8_t center_atom;
    std::uint8_t neighbor_atoms[4];
    std::int8_t target_sign;
    std::uint8_t padding[2];
    float minimum_volume;
    float padding_float;
};
static_assert(sizeof(chiral_center_descriptor_t) == 16);

/**
 *  @brief ETK torsion term from CSD preferences — the Cambridge Structural Database. `atom_indices` are
 *  the dihedral atoms i-j-k-l within the molecule, `fourier_coefficients` V₁ through V₆ in kcal/mol.
 *
 *  @verbatim
 *  E_torsion = Σ_{n=1..6} V_n * cos(n * φ)
 *  where φ is the dihedral angle defined by atom_indices[0]-[1]-[2]-[3].
 *  @endverbatim
 */
struct alignas(32) etk_torsion_descriptor_t {
    std::uint8_t atom_indices[4];
    std::uint8_t padding[4];
    float fourier_coefficients[6];
};
static_assert(sizeof(etk_torsion_descriptor_t) == 32);

/**
 *  @brief Distance constraint for distance geometry and ETK refinement, over atoms `atom_index_i` and
 *  `atom_index_j` within the molecule, bounds in angstrom and `penalty_weight` the violation's constant.
 *
 *  @verbatim
 *  E = penalty_weight * max(0, d_ij - upper)^2 + penalty_weight * max(0, lower - d_ij)^2
 *  where d_ij = |r_i - r_j|
 *  @endverbatim
 */
struct alignas(16) distance_constraint_t {
    std::uint8_t atom_index_i;
    std::uint8_t atom_index_j;
    std::uint8_t padding[2];
    float lower_bound_angstrom;
    float upper_bound_angstrom;
    float penalty_weight;
};
static_assert(sizeof(distance_constraint_t) == 16);

/**
 *  @brief Per-atom MMFF94 VdW parameters: RDKit's `MMFFVdW` entry plus the donor-acceptor flag
 *  `scaleVdWParams` reads.
 *
 *  Stored once per atom and combined pairwise inside the kernel — Halgren's rule for R*_ij,
 *  Slater-Kirkwood for epsilon_ij — over nonbonded pairs enumerated on the fly from the descriptor's
 *  `bond_distance` matrix rather than from a materialized pair array. `alpha` is the polarizability α,
 *  `n_effective` the effective valence-electron count N, `g_factor` the Slater-Kirkwood G_i, `r_star` the
 *  atom's VdW radius R*, `a_scale` the A_i kept for parity with other MMFF variants, and `donor_acceptor`
 *  0 for '-', 1 for 'A' acceptor, 2 for 'D' donor.
 */
struct alignas(8) mmff94_atom_vdw_params_t {
    float alpha;
    float n_effective;
    float a_scale;
    float g_factor;
    float r_star;
    std::uint8_t donor_acceptor;
    std::uint8_t padding[3];
};
static_assert(sizeof(mmff94_atom_vdw_params_t) == 24);

/**
 *  @brief Per-atom MMFF94 VdW parameters reduced to what a nonbonded pair evaluation reads.
 *
 *  Every one-atom factor of the Halgren and Slater-Kirkwood combining rules is folded once, as the polish
 *  kernel stages the table into shared memory, so the FIRE loop stops re-deriving it per pair per iteration;
 *  `a_scale` is absent because nothing reads it, which is where the other eight bytes go. `r_star` is the
 *  atom's VdW radius R*, `polarizability_numerator` its `G_i * alpha_i` share of the well-depth numerator,
 *  `polarizability_root` its `sqrt(alpha_i / N_i)` addend in that denominator, and `donor_acceptor` 0 for
 *  '-', 1 for 'A' acceptor, 2 for 'D' donor.
 */
struct alignas(8) mmff94_atom_vdw_folded_t {
    float r_star;
    float polarizability_numerator;
    float polarizability_root;
    std::uint8_t donor_acceptor;
    std::uint8_t padding[3];
};
static_assert(sizeof(mmff94_atom_vdw_folded_t) == 16);

/**
 *  @brief MMFF94 bond stretch over `atom_i` and `atom_j`; `kb` in mdyn per angstrom, `r0` in angstroms.
 *
 *  @verbatim
 *  E = 0.5 * 143.9325 * kb * dr^2 * (1 + cs * dr + 7/12 * cs^2 * dr^2)
 *  @endverbatim
 */
struct alignas(16) mmff94_bond_t {
    std::uint8_t atom_i;
    std::uint8_t atom_j;
    std::uint8_t padding[2];
    float force_constant;
    float equilibrium_distance;
    float padding_float;
};

static_assert(sizeof(mmff94_bond_t) == 16);

/**
 *  @brief MMFF94 angle bend with `atom_j` central; `ka` in mdyn·angstrom per rad², θ₀ in radians, and
 *  `is_linear` 1 when the central atom is MMFF94 `linh`, selecting the linear-bend form.
 *
 *  @verbatim
 *  E = 0.5 * 0.043844 * ka * Δθ^2 * (1 + cb * Δθ)
 *  @endverbatim
 */
struct alignas(16) mmff94_angle_t {
    std::uint8_t atom_i;
    std::uint8_t atom_j;
    std::uint8_t atom_k;
    std::uint8_t is_linear;
    float force_constant;
    float equilibrium_angle;
    float padding_float;
};
static_assert(sizeof(mmff94_angle_t) == 16);

/**
 *  @brief MMFF94 torsion over `atom_i`-`atom_j`-`atom_k`-`atom_l`; `V1`, `V2` and `V3` are the first,
 *  second and third barrier heights in kcal/mol.
 *
 *  @verbatim
 *  E = 0.5 * (V1*(1+cos(φ)) + V2*(1-cos(2*φ)) + V3*(1+cos(3*φ)))
 *  @endverbatim
 */
struct alignas(16) mmff94_torsion_t {
    std::uint8_t atom_i;
    std::uint8_t atom_j;
    std::uint8_t atom_k;
    std::uint8_t atom_l;
    float torsion_v1;
    float torsion_v2;
    float torsion_v3;
};

static_assert(sizeof(mmff94_torsion_t) == 16);

/**
 *  @brief MMFF94 out-of-plane bending: `atom_i` out of plane, `atom_j` the central sp2 atom, `atom_k` and
 *  `atom_l` its two in-plane neighbours, `koop` in mdyn·angstrom per rad².
 *
 *  @verbatim
 *  E = 0.5*0.043844*koop*chi^2 (chi in degrees)
 *  Wilson angle: chi = angle between bond j->i and the plane j-k-l.
 *  @endverbatim
 */
struct alignas(16) mmff94_oop_t {
    std::uint8_t atom_i;
    std::uint8_t atom_j;
    std::uint8_t atom_k;
    std::uint8_t atom_l;
    float force_constant;
    float padding[2];
};

static_assert(sizeof(mmff94_oop_t) == 16);

/**
 *  @brief MMFF94 stretch-bend coupling with `atom_j` central between end atoms `atom_i` and `atom_k`;
 *  `kba_ijk` and `kba_kji` are the force constants of bonds i-j and k-j, θ₀ is in radians, and r₀_ij
 *  and r₀_kj in angstroms.
 *
 *  @verbatim
 *  E = 2.5121 * (kba_ijk * Δr_ij + kba_kji * Δr_kj) * Δθ
 *  @endverbatim
 */
struct alignas(32) mmff94_stretchbend_t {
    std::uint8_t atom_i;
    std::uint8_t atom_j;
    std::uint8_t atom_k;
    std::uint8_t padding;
    float kba_ijk;
    float kba_kji;
    float equilibrium_angle;
    float equilibrium_distance_ij;
    float equilibrium_distance_kj;
    float padding_float;
};

static_assert(sizeof(mmff94_stretchbend_t) == 32);

/** Per-atom MMFF94 interaction descriptor for the per-atom adjacency list: this atom participates with
 *  `role` in `term_index` of `interaction_type` — nonbonded, bond, angle end or center, torsion, oop,
 *  stretch-bend — where `term_index` indexes that type's own parameter array. */
struct mmff94_atom_interaction_t {
    std::uint32_t term_index;
    mmff94_interaction_type_t interaction_type;
    atom_role_t role;
};
static_assert(sizeof(mmff94_atom_interaction_t) == 8);

/**
 *  @brief Per-atom per-type offsets into the MMFF94 interaction list.
 *
 *  No `nonbonded_*` fields: that kernel loop enumerates pairs on the fly from the `bond_distance` matrix.
 *  Counts are `uint32_t` because a central atom in a large drug-like molecule can participate in more than
 *  255 torsions, and a narrower counter wraps: the torsion writes overflow into the next atom's OOP region
 *  and the kernel then reads torsion term indices out of `oop_params`, an illegal memory access. `uint16_t`
 *  would raise the ceiling to 65535, and `uint32_t` removes it entirely for 40 bytes per atom-offset entry.
 */
struct mmff94_atom_offsets_t {
    std::uint32_t bond_offset;
    std::uint32_t angle_offset;
    std::uint32_t torsion_offset;
    std::uint32_t oop_offset;
    std::uint32_t stretchbend_offset;
    std::uint32_t bond_count;
    std::uint32_t angle_count;
    std::uint32_t torsion_count;
    std::uint32_t oop_count;
    std::uint32_t stretchbend_count;
};
static_assert(sizeof(mmff94_atom_offsets_t) == 40);

/** Per-atom ETK interaction descriptor; `interaction_type` is `etk_torsion_k` or
 *  `etk_distance_constraint_k`. */
struct etk_atom_interaction_t {
    std::uint32_t term_index;
    etk_interaction_type_t interaction_type;
    atom_role_t role;
};
static_assert(sizeof(etk_atom_interaction_t) == 8);

/** Per-atom offset into the ETK interaction list. */
struct etk_atom_offsets_t {
    std::uint32_t interaction_offset;
    std::uint32_t interaction_count;
};
static_assert(sizeof(etk_atom_offsets_t) == 8);

/** One molecule the device MMFF typer cannot type, with the host's answer for it. The native typer leaves
 *  an atom at type 0 for a minority of molecules — 4.63 % of a 5000-SMILES stratified PubChem sample,
 *  0.02 % of a 5000-SMILES Enamine REAL sample. The host runs the same typer over the same descriptor
 *  before the wave launches, so it knows which those are without a readback. Where RDKit
 *  can parametrise one, its atom types and partial charges are staged here and a device pass writes them
 *  over the typer's output; where RDKit cannot either, the molecule is rejected with
 *  `unsupported_element_k`, which usually covers every untyped molecule and leaves this array empty, making
 *  the patch a guard rather than a hot path. `molecule_index` is the molecule's position in the batch,
 *  `atom_base` its first atom in the batch-global per-atom arrays, `patch_base` its first atom in the
 *  staged patch arrays. */
struct mmff94_typing_patch_t {
    std::uint32_t molecule_index;
    std::uint32_t atom_base;
    std::uint32_t patch_base;
    std::uint32_t atom_count;
};
static_assert(sizeof(mmff94_typing_patch_t) == 16);

} // namespace smallfield::types
