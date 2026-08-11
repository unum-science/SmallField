/**
 *  @file src/smallfield/types/etkdg_types.hpp
 *  @brief Data structures for the GPU ETKDG conformer generation pipeline.
 *
 *  Defines the types used to pass preprocessed molecular data from CPU to GPU:
 *  - Molecule metadata — atom counts, offsets into batched arrays
 *  - Chiral center descriptors — index-ordered neighbors, target volume sign
 *  - ETK torsion descriptors — CSD Fourier coefficients
 *  - Distance constraints — 1-2 bond and 1-3 angle-derived bounds
 *
 *  The parameter and constraint structs carry an explicit `alignas`; the metadata, per-atom CSR, and
 *  typing-patch structs take their natural alignment.
 *
 *  The bounds matrix is a full N × N float matrix in row-major order, matching RDKit's `DistGeom::BoundsMatrix`
 *  convention. Distances are in angstroms.
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
 *  Every offset here counts @b elements, never bytes. The batch layout in
 *  `smallfield::api::wave_slice_layout_t` carves the same regions out of the pool and its offsets
 *  count bytes, so the two are named apart — `bounds_offset` here against `bounds_byte_offset`
 *  there — rather than differing only by word order, which would read and compile the same
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
    /** Offset into batched per-atom VdW params (= atom offset). */
    std::uint32_t mmff_atom_vdw_offset;
    /** Offset into batched per-atom charge (= atom offset). */
    std::uint32_t mmff_atom_charge_offset;
    /** Offset into batched per-molecule N×N bond_distance. */
    std::uint32_t bond_distance_offset;
    std::uint32_t mmff_bond_offset;
    std::uint32_t mmff_angle_offset;
    std::uint32_t mmff_torsion_offset;
    std::uint32_t mmff_oop_offset;
    std::uint32_t mmff_stretchbend_offset;
    std::uint32_t mmff_atom_csr_offset;
    /** Offset into batched ETK interaction list. */
    std::uint32_t etk_interaction_offset;
    /** Offset into batched ETK atom offsets. */
    std::uint32_t etk_atom_csr_offset;
    /**
     *  @brief Reserved for a future host-side bounds quantisation step.
     *
     *  The pipeline computes per-molecule (mol_min, inv_scale) inside `etkdg_pipeline_warp_kernel`
     *  because the metric prepass smooths bounds in global memory and the quantise step must scan
     *  the @b post-smoothed values. If smoothing ever moves host-side these slots can carry the
     *  pre-quantised range — until then they are unused, `prepare_molecule_into` sets them to
     *  {0, 1}, and the kernel ignores them.
     */
    float bounds_min;
    float bounds_inv_scale;
    /**
     *  @brief Heavy, non-hydrogen, atom count.
     *
     *  This and the two fields below are optional readbacks, populated by preprocess and bubbled
     *  to the caller via the corresponding output spans on `context_t::generate_conformers`. GPU
     *  kernels do @b not read these — they are pure host-side passthroughs computed once per
     *  molecule from the post-`AddHs` RDKit `RWMol`.
     */
    std::uint32_t num_heavy_atoms;
    /** Total bond count, including H bonds. */
    std::uint32_t num_bonds;
    /** Exact molecular weight in Daltons. */
    float molecular_weight;
};

/**
 *  @brief Chiral center descriptor.
 *  Stores the central atom, its neighbors in ascending atom-index order, and the expected sign of
 *  the signed tetrahedral volume over that order.
 *
 *  The ordering is index-based, not CIP-based: `target_sign` is derived from the tetrahedral tag
 *  and the parity of this permutation relative to the bond order the tag is defined against, so
 *  any deterministic order works as long as the same one is used at derivation and validation.
 *  A three-coordinate center stores its three bonded neighbors and leaves the fourth slot unused.
 *
 *  @verbatim
 *  Validation: V = dot(r_b - r_a, cross(r_c - r_a, r_d - r_a))
 *  where a = center and b, c, d = neighbor_atoms[0..2]; sign(V) must equal target_sign.
 *  @endverbatim
 */
struct alignas(8) chiral_center_descriptor_t {
    /** Central atom index within molecule. */
    std::uint8_t center_atom;
    /** Neighbor atom indices, ascending. */
    std::uint8_t neighbor_atoms[4];
    /** Expected sign of the signed tetrahedral volume. */
    std::int8_t target_sign;
    std::uint8_t padding[2];
};

/**
 *  ETK torsion term from CSD preferences — the Cambridge Structural Database.
 *
 *  @verbatim
 *  E_torsion = Σ_{n=1..6} V_n * cos(n * φ)
 *  where φ is the dihedral angle defined by atom_indices[0]-[1]-[2]-[3].
 *  @endverbatim
 */
struct alignas(32) etk_torsion_descriptor_t {
    /** Dihedral atoms i-j-k-l within molecule. */
    std::uint8_t atom_indices[4];
    std::uint8_t padding[4];
    /** V₁ through V₆ in kcal/mol. */
    float fourier_coefficients[6];
};

/**
 *  Distance constraint for distance geometry and ETK refinement stages.
 *
 *  @verbatim
 *  E = penalty_weight * max(0, d_ij - upper)^2 + penalty_weight * max(0, lower - d_ij)^2
 *  where d_ij = |r_i - r_j|
 *  @endverbatim
 */
struct alignas(16) distance_constraint_t {
    /** First atom index within molecule. */
    std::uint8_t atom_index_i;
    /** Second atom index within molecule. */
    std::uint8_t atom_index_j;
    /** 0 = 1-2 bond, 1 = 1-3 angle-derived, 2 = custom. */
    std::uint8_t constraint_type;
    std::uint8_t padding;
    /** Minimum allowed interatomic distance. */
    float lower_bound_angstrom;
    /** Maximum allowed interatomic distance. */
    float upper_bound_angstrom;
    /** Force constant for bound violation penalty. */
    float penalty_weight;
};

/**
 *  @brief Per-atom MMFF94 VdW parameters.
 *
 *  Stored once per atom; combined pairwise inside the kernel via Halgren's combining rule for
 *  R*_ij and Slater-Kirkwood for epsilon_ij. The kernel enumerates nonbonded pairs on the fly
 *  from the descriptor's `bond_distance` matrix rather than reading a materialized pair array.
 *
 *  Matches the atom-level data surfaced by RDKit's `MMFFVdW` entry plus the donor-acceptor flag
 *  used by `scaleVdWParams`.
 */
struct alignas(8) mmff94_atom_vdw_params_t {
    /** Polarizability α. */
    float alpha;
    /** Effective number of valence electrons N. */
    float n_effective;
    /** A_i — preserved for parity with other MMFF variants. */
    float a_scale;
    /** G_i, the Slater-Kirkwood factor. */
    float g_factor;
    /** This atom's VdW radius R*. */
    float r_star;
    /** 0 = '-', 1 = 'A' acceptor, 2 = 'D' donor. */
    std::uint8_t donor_acceptor;
    std::uint8_t padding[3];
};
static_assert(sizeof(mmff94_atom_vdw_params_t) == 24);

/**
 *  @brief MMFF94 bond stretching term.
 *
 *  @verbatim
 *  E = 0.5 * 143.9325 * kb * dr^2 * (1 + cs * dr + 7/12 * cs^2 * dr^2)
 *  @endverbatim
 */
struct alignas(16) mmff94_bond_t {
    std::uint8_t atom_i;
    std::uint8_t atom_j;
    std::uint8_t padding[2];
    /** `kb` in mdyn per angstrom. */
    float force_constant;
    /** `r0` in angstroms. */
    float equilibrium_distance;
    float padding_float;
};

static_assert(sizeof(mmff94_bond_t) == 16);

/**
 *  @brief MMFF94 angle bending term.
 *
 *  @verbatim
 *  E = 0.5 * 0.043844 * ka * Δθ^2 * (1 + cb * Δθ)
 *  @endverbatim
 */
struct alignas(16) mmff94_angle_t {
    std::uint8_t atom_i;
    /** Central atom. */
    std::uint8_t atom_j;
    std::uint8_t atom_k;
    /** 1 when the central atom is MMFF94 `linh` — use the linear-bend form. */
    std::uint8_t is_linear;
    /** `ka` in mdyn·angstrom per rad². */
    float force_constant;
    /** θ₀ in radians. */
    float equilibrium_angle;
    float padding_float;
};
static_assert(sizeof(mmff94_angle_t) == 16);

/**
 *  @brief MMFF94 torsion term.
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
    /** MMFF94 `V1`, the first barrier height in kcal/mol. */
    float torsion_v1;
    /** MMFF94 `V2`, the second barrier height in kcal/mol. */
    float torsion_v2;
    /** MMFF94 `V3`, the third barrier height in kcal/mol. */
    float torsion_v3;
};

static_assert(sizeof(mmff94_torsion_t) == 16);

/**
 *  MMFF94 out-of-plane bending.
 *
 *  @verbatim
 *  E = 0.5*0.043844*koop*chi^2 (chi in degrees)
 *  Wilson angle: chi = angle between bond j->i and the plane j-k-l.
 *  @endverbatim
 */
struct alignas(16) mmff94_oop_t {
    /** Out-of-plane atom. */
    std::uint8_t atom_i;
    /** Central sp2 atom. */
    std::uint8_t atom_j;
    /** In-plane neighbor 1. */
    std::uint8_t atom_k;
    /** In-plane neighbor 2. */
    std::uint8_t atom_l;
    /** `koop` in mdyn·angstrom per rad². */
    float force_constant;
    float padding[2];
};

static_assert(sizeof(mmff94_oop_t) == 16);

/**
 *  @brief MMFF94 stretch-bend coupling term.
 *
 *  @verbatim
 *  E = 2.5121 * (kba_ijk * Δr_ij + kba_kji * Δr_kj) * Δθ
 *  @endverbatim
 */
struct alignas(32) mmff94_stretchbend_t {
    /** End atom 1. */
    std::uint8_t atom_i;
    /** Central atom. */
    std::uint8_t atom_j;
    /** End atom 2. */
    std::uint8_t atom_k;
    std::uint8_t padding;
    /** Stretch-bend force constant for bond i-j. */
    float kba_ijk;
    /** Stretch-bend force constant for bond k-j. */
    float kba_kji;
    /** θ₀ in radians. */
    float equilibrium_angle;
    /** r₀_ij in angstroms. */
    float equilibrium_distance_ij;
    /** r₀_kj in angstroms. */
    float equilibrium_distance_kj;
    float padding_float;
};

static_assert(sizeof(mmff94_stretchbend_t) == 32);

/**
 *  @brief Per-atom MMFF94 interaction descriptor for the per-atom adjacency list.
 *  Each entry says: "atom X participates in term Y with role Z".
 */
struct mmff94_atom_interaction_t {
    /** Index into the type-specific parameter array. */
    std::uint32_t term_index;
    /** Nonbonded, bond, angle(end/center), torsion, oop, stretch-bend. */
    mmff94_interaction_type_t interaction_type;
    atom_role_t role;
};

/**
 *  @brief Per-atom per-type offsets into the MMFF94 interaction list.
 *
 *  No `nonbonded_*` fields — the nonbonded kernel loop enumerates pairs on the fly from the
 *  `bond_distance` matrix.
 *
 *  The counts are `uint32_t` because a central atom in a large drug-like molecule can participate
 *  in more than 255 torsions. A narrower counter wraps, the torsion writes overflow into the next
 *  atom's OOP region, and the kernel then reads torsion term indices out of `oop_params` — an
 *  illegal memory access. `uint16_t` would raise the ceiling to 65535, and `uint32_t` removes it
 *  entirely for 40 bytes per atom-offset entry.
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

/**
 *  @brief Per-atom ETK interaction descriptor.
 */
struct etk_atom_interaction_t {
    std::uint32_t term_index;
    /** etk_torsion_k or etk_distance_constraint_k. */
    etk_interaction_type_t interaction_type;
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
 *  stratified PubChem sample, 0.02 % of a 5000-SMILES Enamine REAL sample. The host runs the same
 *  typer over the same descriptor before the wave launches, so it knows which molecules those are
 *  without a readback; where RDKit can parametrise one, its atom types and partial charges are
 *  staged here, a device pass writes them over the typer's output and clears the molecule's typing
 *  status, and the extractor treats the molecule as typed. Where RDKit cannot parametrise it
 *  either, the molecule is rejected with `unsupported_element_k`. On both samples above that second
 *  case covers every untyped molecule and this array stays empty, so the patch is a guard against a
 *  corpus where the two typers disagree rather than a hot path.
 */
struct mmff94_typing_patch_t {
    /** Position of the molecule inside the batch. */
    std::uint32_t molecule_index;
    /** First atom of the molecule in the batch-global per-atom arrays. */
    std::uint32_t atom_base;
    /** First atom of the molecule inside the staged patch arrays. */
    std::uint32_t patch_base;
    std::uint32_t atom_count;
};

} // namespace smallfield::types
