/**
 *  @file etkdg_types.hpp
 *  @brief Data structures for the GPU ETKDG conformer generation pipeline.
 *
 *  Defines the types used to pass preprocessed molecular data from CPU to GPU:
 *  - Molecule metadata (atom counts, offsets into batched arrays)
 *  - Chiral center descriptors (CIP-ordered neighbors, target sign)
 *  - ETK torsion descriptors (CSD Fourier coefficients)
 *  - Distance constraints (1-2 bond, 1-3 angle-derived bounds)
 *
 *  All structs use `alignas` (C++17) for portable alignment.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <cstdint>

namespace smallfield::types {

/**
 *  @brief Positional index of an atom within a multi-atom interaction term.
 *
 *  For bonds: first/second. For angles: first=end, second=center, third=end.
 *  For torsions/OOP: first through fourth in atom-list order.
 */
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
 *  Per-molecule metadata for the ETKDG pipeline.
 *  One per molecule in the batch; offsets point into concatenated arrays.
 *
 *  Every offset here counts @b elements, never bytes.  The batch layout in
 *  @ref smallfield::api::wave_slice_layout_t carves the same regions out of the pool and its offsets
 *  count bytes, so the two are named apart — `bounds_offset` here against `bounds_byte_offset`
 *  there.  They used to differ only by word order, which reads the same and compiles the same
 *  while addressing memory four times too far.
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
    std::uint32_t mmff_atom_vdw_offset;    ///< offset into batched per-atom VdW params (= atom offset)
    std::uint32_t mmff_atom_charge_offset; ///< offset into batched per-atom charge (= atom offset)
    std::uint32_t bond_distance_offset;    ///< offset into batched per-molecule N×N bond_distance
    std::uint32_t mmff_bond_offset;
    std::uint32_t mmff_angle_offset;
    std::uint32_t mmff_torsion_offset;
    std::uint32_t mmff_oop_offset;
    std::uint32_t mmff_stretchbend_offset;
    std::uint32_t mmff_atom_csr_offset;
    std::uint32_t etk_interaction_offset; ///< offset into batched ETK interaction list
    std::uint32_t etk_atom_csr_offset;    ///< offset into batched ETK atom offsets
    /**
     *  @brief Reserved for a future host-side bounds quantisation step.
     *
     *  The current pipeline computes per-molecule (mol_min, inv_scale) inside
     *  etkdg_pipeline_warp_kernel because the metric prepass smooths bounds in
     *  global memory and the quantise step must scan the *post-smoothed*
     *  values.  If smoothing ever moves host-side these slots can carry the
     *  pre-quantised range — until then they are unused (set to {0, 1} by
     *  prepare_molecule_into) and the kernel ignores them.
     */
    float bounds_min;
    float bounds_inv_scale;
    /**
     *  @brief Optional readback fields, populated by preprocess and bubbled to the
     *  caller via the corresponding output spans on @ref context_t::generate_conformers.
     *  GPU kernels do NOT read these — they are pure host-side passthroughs computed
     *  once per molecule from the post-`AddHs` RDKit `RWMol`.
     */
    std::uint32_t num_heavy_atoms;        ///< heavy (non-H) atom count
    std::uint32_t num_bonds;              ///< total bond count, including H bonds
    float molecular_weight;               ///< exact molecular weight in Daltons
};

/**
 *  Chiral center descriptor.
 *  Stores the central atom, its neighbors in ascending atom-index order, and the expected sign of
 *  the signed tetrahedral volume over that order.
 *
 *  The ordering is index-based, not CIP-based: `target_sign` is derived from the tetrahedral tag
 *  and the parity of this permutation relative to the bond order the tag is defined against, so
 *  any deterministic order works as long as the same one is used at derivation and validation.
 *  A three-coordinate center stores its three bonded neighbors and leaves the fourth slot unused.
 *
 *  Validation: V = (r_b - r_a) . ((r_c - r_a) x (r_d - r_a))
 *  where a = center, b,c,d = neighbor_atoms[0..2]; sign(V) must equal target_sign.
 */
struct alignas(8) chiral_center_descriptor_t {
    std::uint8_t center_atom;       ///< central atom index within molecule
    std::uint8_t neighbor_atoms[4]; ///< neighbor atom indices, ascending
    std::int8_t target_sign;        ///< expected sign of the signed tetrahedral volume
    std::uint8_t padding[2];
};

/**
 *  ETK torsion term from CSD (Cambridge Structural Database) preferences.
 *
 *  E_torsion = Σₙ₌₁⁶ Vₙ·cos(n·φ)
 *  where φ is the dihedral angle defined by atom_indices[0]-[1]-[2]-[3].
 */
struct alignas(32) etk_torsion_descriptor_t {
    std::uint8_t atom_indices[4]; ///< dihedral atoms i-j-k-l within molecule
    std::uint8_t padding[4];
    float fourier_coefficients[6]; ///< V₁ through V₆ (kcal/mol)
};

/**
 *  Distance constraint for distance geometry and ETK refinement stages.
 *
 *  E = penalty_weight · max(0, dᵢⱼ - upper)² + penalty_weight · max(0, lower - dᵢⱼ)²
 *  where dᵢⱼ = ‖rᵢ - rⱼ‖
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
 *  @brief Per-atom MMFF94 VdW parameters.
 *
 *  Stored once per atom; combined pairwise inside the kernel via Halgren's
 *  combining rule (R*_ij) and Slater-Kirkwood (epsilon_ij).  Replaces the
 *  materialized nonbonded pair array — the kernel enumerates pairs on the
 *  fly from the descriptor's `bond_distance` matrix.
 *
 *  Matches the atom-level data surfaced by RDKit's MMFFVdW entry plus
 *  the donor-acceptor flag used by `scaleVdWParams`.
 */
struct alignas(8) mmff94_atom_vdw_params_t {
    float alpha;                 ///< polarizability (α)
    float n_effective;           ///< effective number of valence electrons (N)
    float a_scale;               ///< A_i — preserved for parity with other MMFF variants
    float g_factor;              ///< G_i (Slater-Kirkwood factor)
    float r_star;                ///< this atom's VdW radius (R*)
    std::uint8_t donor_acceptor; ///< 0 = '-', 1 = 'A' acceptor, 2 = 'D' donor
    std::uint8_t padding[3];
};
static_assert(sizeof(mmff94_atom_vdw_params_t) == 24);

/**
 *  @brief MMFF94 bond stretching term.
 *
 *  E = 0.5 * 143.9325 * kb * dr² * (1 + cs * dr + 7/12 * cs² * dr²)
 */
struct alignas(16) mmff94_bond_t {
    std::uint8_t atom_i;
    std::uint8_t atom_j;
    std::uint8_t padding[2];
    float force_constant;       ///< kb (mdyn/angstrom)
    float equilibrium_distance; ///< r0 (angstroms)
    float padding_float;
};

/**
 *  @brief MMFF94 angle bending term.
 *
 *  E = 0.5 * 0.043844 * ka * Δθ² * (1 + cb * Δθ)
 */
struct alignas(16) mmff94_angle_t {
    std::uint8_t atom_i;
    std::uint8_t atom_j;    ///< central atom
    std::uint8_t atom_k;
    std::uint8_t is_linear; ///< 1 when the central atom is MMFF94 `linh` — use the linear-bend form
    float force_constant;    ///< ka (mdyn·angstrom/rad²)
    float equilibrium_angle; ///< θ₀ (radians)
    float padding_float;
};
static_assert(sizeof(mmff94_angle_t) == 16);

/**
 *  @brief MMFF94 torsion term.
 *
 *  E = 0.5 * (V1*(1+cos(phi)) + V2*(1-cos(2*phi)) + V3*(1+cos(3*phi)))
 */
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
 *  MMFF94 out-of-plane bending. E = 0.5·0.043844·koop·chi² (chi in degrees).
 *  Wilson angle: chi = angle between bond j→i and the plane j-k-l.
 */
struct alignas(16) mmff94_oop_t {
    std::uint8_t atom_i;  ///< out-of-plane atom
    std::uint8_t atom_j;  ///< central sp2 atom
    std::uint8_t atom_k;  ///< in-plane neighbor 1
    std::uint8_t atom_l;  ///< in-plane neighbor 2
    float force_constant; ///< koop (mdyn·angstrom/rad²)
    float padding[2];
};

/**
 *  @brief MMFF94 stretch-bend coupling term.
 *
 *  E = 2.5121 * (kba_ijk * Δrᵢⱼ + kba_kji * Δrₖⱼ) * Δθ
 */
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
 *  Per-atom MMFF94 interaction descriptor for the per-atom adjacency list.
 *  Each entry says: "atom X participates in term Y with role Z".
 *  interaction_type: mmff94_nonbonded_k through mmff94_stretch_bend_k
 */
struct mmff94_atom_interaction_t {
    std::uint32_t term_index;                   ///< index into the type-specific parameter array
    mmff94_interaction_type_t interaction_type; ///< nonbonded, bond, angle(end/center), torsion, oop, stretch-bend
    atom_role_t role;
};

/**
 *  @brief Per-atom per-type offsets into the MMFF94 interaction list.
 *
 *  No `nonbonded_*` fields — the nonbonded kernel loop enumerates pairs
 *  on the fly from the `bond_distance` matrix.
 */
struct mmff94_atom_offsets_t {
    std::uint32_t bond_offset;
    std::uint32_t angle_offset;
    std::uint32_t torsion_offset;
    std::uint32_t oop_offset;
    std::uint32_t stretchbend_offset;
    // These are `uint32_t` because a central atom in a large drug-like molecule can participate in
    // more than 255 torsions.  A narrower counter wraps, the torsion writes overflow into the next
    // atom's OOP region, and the kernel then reads torsion term indices out of `oop_params` — an
    // illegal memory access.  `uint16_t` would raise the ceiling to 65535; `uint32_t` removes it
    // entirely for 40 bytes per atom-offset entry.
    std::uint32_t bond_count;
    std::uint32_t angle_count;
    std::uint32_t torsion_count;
    std::uint32_t oop_count;
    std::uint32_t stretchbend_count;
};

/**
 *  @brief Per-atom ETK interaction descriptor.
 *
 *  interaction_type: etk_torsion_k or etk_distance_constraint_k.
 */
struct etk_atom_interaction_t {
    std::uint32_t term_index;
    etk_interaction_type_t interaction_type; ///< etk_torsion_k or etk_distance_constraint_k
    atom_role_t role;
};

/**
 *  @brief Per-atom offset into the ETK interaction list.
 */
struct etk_atom_offsets_t {
    std::uint32_t interaction_offset;
    std::uint32_t interaction_count;
};

/**
 *  @brief One molecule the device MMFF typer cannot type, with the host's answer for it.
 *
 *  The native typer leaves an atom at type 0 for a minority of molecules — 4.63 % of a 5000-SMILES
 *  stratified PubChem sample, 0.02 % of a 5000-SMILES Enamine REAL sample.  The host runs the same
 *  typer over the same descriptor before the wave launches, so it knows which molecules those are
 *  without a readback; where RDKit can parametrise one, its atom types and partial charges are
 *  staged here, a device pass writes them over the typer's output and clears the molecule's typing
 *  status, and the extractor treats the molecule as typed.  Where RDKit cannot parametrise it
 *  either, the molecule is rejected with `unsupported_element` as it always was.  On both samples
 *  above that second case covers every untyped molecule and this array stays empty, so the patch is
 *  a guard against a corpus where the two typers disagree rather than a hot path.
 */
struct mmff94_typing_patch_t {
    std::uint32_t molecule_index; ///< position of the molecule inside the batch
    std::uint32_t atom_base;      ///< first atom of the molecule in the batch-global per-atom arrays
    std::uint32_t patch_base;     ///< first atom of the molecule inside the staged patch arrays
    std::uint32_t atom_count;
};

/**
 *  Bounds matrix layout convention:
 *
 *  Full N x N float matrix, row-major.
 *  bounds[i*N + j] with i < j  stores the upper bound distance (angstroms)
 *  bounds[i*N + j] with i > j  stores the lower bound distance (angstroms)
 *  bounds[i*N + i] = 0.0
 *
 *  This matches RDKit's DistGeom::BoundsMatrix convention.
 */

} // namespace smallfield::types
