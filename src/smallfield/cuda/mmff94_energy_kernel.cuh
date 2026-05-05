/**
 *  @file mmff94_energy_kernel.cuh
 *  @brief MMFF94 energy and gradient functor for ETKDG pipeline stage 9.
 *
 *  Uses per-atom interaction lists for zero-waste iteration.
 *  Full analytical gradients for VdW, electrostatic, bond, angle, torsion, OOP, and stretch-bend terms.
 *  Compatible with fire_minimize_ (same functor interface).
 *
 *  Positions in SoA float: [x₀..xₙ, y₀..yₙ, z₀..zₙ].
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/cuda/warp_primitives.cuh"

namespace smallfield::cuda {
using smallfield::types::atom_role_first_k;
using smallfield::types::atom_role_fourth_k;
using smallfield::types::atom_role_second_k;
using smallfield::types::atom_role_t;
using smallfield::types::atom_role_third_k;
using smallfield::types::mmff94_angle_t;
using smallfield::types::mmff94_atom_interaction_t;
using smallfield::types::mmff94_atom_offsets_t;
using smallfield::types::mmff94_atom_vdw_params_t;
using smallfield::types::mmff94_bond_t;
using smallfield::types::mmff94_oop_t;
using smallfield::types::mmff94_stretchbend_t;
using smallfield::types::mmff94_torsion_t;

constexpr float mmff_coulomb_k = 332.0716f;
constexpr float mmff_electrostatic_buffer_k = 0.05f;
constexpr float mmff_1_4_scale_k = 0.75f;
constexpr float mmff_vdw_delta_k = 0.07f; // delta
constexpr float mmff_vdw_gamma_k = 0.12f; // gamma
constexpr float mmff_bond_prefactor_k = 143.9325f;
constexpr float mmff_bond_cubic_stretch_k = -2.0f;
constexpr float mmff_angle_prefactor_k = 0.043844f;
constexpr float mmff_angle_cubic_bend_k = -0.007f;
constexpr float mmff_torsion_prefactor_k = 0.5f;
constexpr float mmff_stretch_bend_prefactor_k = 2.51210f;
constexpr float mmff_degrees_to_radians_k = 3.14159265f / 180.0f;
constexpr float mmff_radians_to_degrees_k = 180.0f / 3.14159265f;

/**
 *  @brief Compute nonbonded (VdW + electrostatic) energy and gradient contribution for one atom.
 *
 *  Inlines Halgren's R* combining rule (with the donor-suppression clause) and
 *  Slater-Kirkwood's epsilon combining rule, then applies donor-acceptor
 *  scaling, reproducing RDKit's `calcUnscaledVdWMinimum` + `calcUnscaledVdWWellDepth`
 *  + `scaleVdWParams` in registers.  Replaces the pre-computed pair-array path.
 *
 *  @param[in] positions SoA coordinate array [x0..xN, y0..yN, z0..zN].
 *  @param[in] num_atoms Number of atoms in the molecule.
 *  @param[in] atom Index of the atom this thread is accumulating for.
 *  @param[in] other Other atom participating in the pair.
 *  @param[in] p_atom Per-atom VdW params for @p atom.
 *  @param[in] p_other Per-atom VdW params for @p other.
 *  @param[in] charge_product Pre-multiplied q_atom * q_other.
 *  @param[in] is_1_4 Whether bond_distance == 3 (electrostatic scaling applies).
 *  @param[in] first_endpoint True if this call is from the "lower-index" endpoint —
 *      only first_endpoint calls accumulate energy to avoid double-counting.
 *  @param[inout] energy Accumulated energy (only incremented when @p first_endpoint).
 *  @param[inout] gradient_x Accumulated x-gradient for @p atom.
 *  @param[inout] gradient_y Accumulated y-gradient for @p atom.
 *  @param[inout] gradient_z Accumulated z-gradient for @p atom.
 */
__device__ __forceinline__ void mmff_nonbonded_contribution_(
    float const *positions, std::uint32_t num_atoms, std::uint32_t atom, std::uint32_t other,
    mmff94_atom_vdw_params_t const &p_atom, mmff94_atom_vdw_params_t const &p_other, float charge_product, bool is_1_4,
    bool first_endpoint, float *energy, float *gradient_x, float *gradient_y, float *gradient_z) {

    // Halgren R* combining rule with donor-suppression clause, plus Slater-Kirkwood epsilon.
    float const vdw_radius_i = p_atom.r_star;
    float const vdw_radius_j = p_other.r_star;
    float const gamma = (vdw_radius_i - vdw_radius_j) * __frcp_rn(vdw_radius_i + vdw_radius_j);
    bool const has_donor = (p_atom.donor_acceptor == 2) || (p_other.donor_acceptor == 2);
    float const factor = has_donor ? 0.0f : 0.2f * (1.0f - expf(-12.0f * gamma * gamma));
    float R_star_ij = 0.5f * (vdw_radius_i + vdw_radius_j) * (1.0f + factor);

    float const R2 = R_star_ij * R_star_ij;
    float const R6 = R2 * R2 * R2;
    float eps_ij = 181.16f * p_atom.g_factor * p_other.g_factor * p_atom.alpha * p_other.alpha *
                   __frcp_rn((sqrtf(p_atom.alpha * __frcp_rn(p_atom.n_effective)) +
                              sqrtf(p_other.alpha * __frcp_rn(p_other.n_effective))) *
                             R6);

    // Donor-acceptor scaling (RDKit's scaleVdWParams when DA pairing is detected).
    bool const is_DA = (p_atom.donor_acceptor == 2 && p_other.donor_acceptor == 1) ||
                       (p_atom.donor_acceptor == 1 && p_other.donor_acceptor == 2);
    if (is_DA) {
        R_star_ij *= 0.8f;
        eps_ij *= 0.5f;
    }

    // Pair geometry.
    float3 const delta = sub_float3_(load_position_(positions, num_atoms, other),
                                     load_position_(positions, num_atoms, atom));
    float const distance_squared = fmaxf(dot_float3_(delta, delta), 1e-8f);
    float const inverse_distance = rsqrtf(distance_squared);
    float const distance = distance_squared * inverse_distance;

    // VdW buffered 14-7 energy and gradient.
    float const density_ratio = distance * __frcp_rn(R_star_ij); // ρ = r / R*
    float const density_ratio_pow7 = density_ratio * density_ratio * density_ratio * density_ratio * density_ratio *
                                     density_ratio * density_ratio;
    float const repulsion_denominator = density_ratio + mmff_vdw_delta_k; // ρ + δ
    float const repulsion_base = (1.0f + mmff_vdw_delta_k) * __frcp_rn(repulsion_denominator);
    float const repulsion_pow7 = repulsion_base * repulsion_base * repulsion_base * repulsion_base * repulsion_base *
                                 repulsion_base * repulsion_base;
    float const attractive_term = (1.0f + mmff_vdw_gamma_k) * __frcp_rn(density_ratio_pow7 + mmff_vdw_gamma_k) - 2.0f;
    float const energy_vdw = eps_ij * repulsion_pow7 * attractive_term;
    float const d_repulsion = -7.0f * repulsion_pow7 * __frcp_rn(repulsion_denominator);
    float const d_attractive = -7.0f * (1.0f + mmff_vdw_gamma_k) * density_ratio_pow7 *
                               __frcp_rn(density_ratio * (density_ratio_pow7 + mmff_vdw_gamma_k) *
                                         (density_ratio_pow7 + mmff_vdw_gamma_k));
    float const energy_gradient_vdw = eps_ij * (d_repulsion * attractive_term + repulsion_pow7 * d_attractive) *
                                      __frcp_rn(R_star_ij);

    // Electrostatic energy and gradient.
    float const scale = is_1_4 ? mmff_1_4_scale_k : 1.0f;
    float const buffered_distance = distance + mmff_electrostatic_buffer_k;
    float const energy_electrostatic = mmff_coulomb_k * charge_product * __frcp_rn(buffered_distance) * scale;
    float const energy_gradient_electrostatic = -mmff_coulomb_k * charge_product *
                                                __frcp_rn(buffered_distance * buffered_distance) * scale;

    // Only the lower-index endpoint adds to total energy.
    if (first_endpoint) *energy += energy_vdw + energy_electrostatic;

    float const energy_gradient_distance = energy_gradient_vdw + energy_gradient_electrostatic;
    *gradient_x -= energy_gradient_distance * delta.x * inverse_distance;
    *gradient_y -= energy_gradient_distance * delta.y * inverse_distance;
    *gradient_z -= energy_gradient_distance * delta.z * inverse_distance;
}

/**
 *  @brief Compute bond stretching energy and gradient for one atom.
 *
 *  @param[in] positions SoA coordinate array [x0..xN, y0..yN, z0..zN].
 *  @param[in] num_atoms Number of atoms in the molecule.
 *  @param[in] atom Index of the atom this thread is accumulating for.
 *  @param[in] bond Bond stretching descriptor with force constant and equilibrium distance.
 *  @param[in] role This atom's role in the bond (atom_role_first_k, atom_role_second_k).
 *  @param[inout] energy Accumulated energy (incremented when @p role is 0).
 *  @param[inout] gradient_x Accumulated x-gradient for @p atom.
 *  @param[inout] gradient_y Accumulated y-gradient for @p atom.
 *  @param[inout] gradient_z Accumulated z-gradient for @p atom.
 */
__device__ __forceinline__ void mmff_bond_contribution_(float const *positions, std::uint32_t num_atoms,
                                                        std::uint32_t atom, mmff94_bond_t const &bond, atom_role_t role,
                                                        float *energy, float *gradient_x, float *gradient_y,
                                                        float *gradient_z) {

    std::uint32_t const other = (role == atom_role_first_k) ? bond.atom_j : bond.atom_i;
    float3 const delta = sub_float3_(load_position_(positions, num_atoms, other),
                                     load_position_(positions, num_atoms, atom));
    float const distance_squared = fmaxf(dot_float3_(delta, delta), 1e-8f);
    float const inverse_distance = rsqrtf(distance_squared);
    float const distance = distance_squared * inverse_distance;
    float const stretch = distance - bond.equilibrium_distance;
    float const cubic = mmff_bond_cubic_stretch_k;

    if (role == atom_role_first_k)
        *energy += 0.5f * mmff_bond_prefactor_k * bond.force_constant * stretch * stretch *
                   (1.0f + cubic * stretch + (7.0f / 12.0f) * cubic * cubic * stretch * stretch);

    float const energy_gradient_distance = mmff_bond_prefactor_k * bond.force_constant * stretch *
                                           (1.0f + 1.5f * cubic * stretch +
                                            (7.0f / 6.0f) * cubic * cubic * stretch * stretch);
    // bond gradient: same sign convention
    *gradient_x -= energy_gradient_distance * delta.x * inverse_distance;
    *gradient_y -= energy_gradient_distance * delta.y * inverse_distance;
    *gradient_z -= energy_gradient_distance * delta.z * inverse_distance;
}

/**
 *  @brief Compute angle bending energy and gradient for one atom.
 *
 *  @param[in] positions SoA coordinate array [x0..xN, y0..yN, z0..zN].
 *  @param[in] num_atoms Number of atoms in the molecule.
 *  @param[in] atom Index of the atom this thread is accumulating for.
 *  @param[in] angle Angle bending descriptor with force constant and equilibrium angle.
 *  @param[in] role This atom's role in the angle (first atom, second atom (center), third atom).
 *  @param[inout] energy Accumulated energy (incremented when @p role is 0).
 *  @param[inout] gradient_x Accumulated x-gradient for @p atom.
 *  @param[inout] gradient_y Accumulated y-gradient for @p atom.
 *  @param[inout] gradient_z Accumulated z-gradient for @p atom.
 */
__device__ __forceinline__ void mmff_angle_contribution_(float const *positions, std::uint32_t num_atoms,
                                                         std::uint32_t atom, mmff94_angle_t const &angle,
                                                         atom_role_t role, float *energy, float *gradient_x,
                                                         float *gradient_y, float *gradient_z) {

    float3 const position_i = load_position_(positions, num_atoms, angle.atom_i);
    float3 const position_j = load_position_(positions, num_atoms, angle.atom_j);
    float3 const position_k = load_position_(positions, num_atoms, angle.atom_k);

    // vectors from central atom j
    float3 const bond_ij = sub_float3_(position_i, position_j);
    float3 const bond_kj = sub_float3_(position_k, position_j);
    float const inverse_distance_ij = rsqrtf(fmaxf(dot_float3_(bond_ij, bond_ij), 1e-8f));
    float const inverse_distance_kj = rsqrtf(fmaxf(dot_float3_(bond_kj, bond_kj), 1e-8f));

    // unit vectors
    float const p12_x = bond_ij.x * inverse_distance_ij, p12_y = bond_ij.y * inverse_distance_ij,
                p12_z = bond_ij.z * inverse_distance_ij;
    float const p32_x = bond_kj.x * inverse_distance_kj, p32_y = bond_kj.y * inverse_distance_kj,
                p32_z = bond_kj.z * inverse_distance_kj;

    float const cos_theta = fmaxf(-1.0f, fminf(1.0f, p12_x * p32_x + p12_y * p32_y + p12_z * p32_z));
    float const sin_theta_sq = 1.0f - cos_theta * cos_theta;
    float const sin_theta = (sin_theta_sq > 1e-12f) ? sqrtf(sin_theta_sq) : 1e-6f;
    float const theta = acosf(cos_theta);
    float const dtheta_deg = (theta - angle.equilibrium_angle) * mmff_radians_to_degrees_k;

    // energy (counted once, by role 0 = atom_i)
    if (role == atom_role_first_k)
        *energy += 0.5f * mmff_angle_prefactor_k * angle.force_constant * dtheta_deg * dtheta_deg *
                   (1.0f + mmff_angle_cubic_bend_k * dtheta_deg);

    // ∂E/∂θ
    // the extra rad2deg converts the derivative to work with the -1/sinθ chain rule
    float const energy_gradient_angle = mmff_radians_to_degrees_k * mmff_angle_prefactor_k * angle.force_constant *
                                        dtheta_deg * (1.0f + 1.5f * mmff_angle_cubic_bend_k * dtheta_deg);
    float const inverse_sine = __frcp_rn(sin_theta);

    // ∂E/∂rᵢ = ∂E/∂θ · ∂cosθ/∂rᵢ · (−1/sinθ)
    // ∂cosθ/∂rᵢ = (p̂₃₂ − cosθ·p̂₁₂) / |r₁₂|
    float const factor = energy_gradient_angle * (-inverse_sine);

    if (role == atom_role_first_k) { // first atom: ∂cosθ/∂rᵢ = (p̂₃₂ − cosθ·p̂₁₂) / |r₁₂|
        *gradient_x += factor * (p32_x - cos_theta * p12_x) * inverse_distance_ij;
        *gradient_y += factor * (p32_y - cos_theta * p12_y) * inverse_distance_ij;
        *gradient_z += factor * (p32_z - cos_theta * p12_z) * inverse_distance_ij;
    }
    else if (role == atom_role_third_k) { // third atom: ∂cosθ/∂rₖ = (p̂₁₂ − cosθ·p̂₃₂) / |r₃₂|
        *gradient_x += factor * (p12_x - cos_theta * p32_x) * inverse_distance_kj;
        *gradient_y += factor * (p12_y - cos_theta * p32_y) * inverse_distance_kj;
        *gradient_z += factor * (p12_z - cos_theta * p32_z) * inverse_distance_kj;
    }
    else { // role == atom_role_second_k, center atom (center): -(grad_i + grad_k)
        float const gi_x = factor * (p32_x - cos_theta * p12_x) * inverse_distance_ij;
        float const gi_y = factor * (p32_y - cos_theta * p12_y) * inverse_distance_ij;
        float const gi_z = factor * (p32_z - cos_theta * p12_z) * inverse_distance_ij;
        float const gk_x = factor * (p12_x - cos_theta * p32_x) * inverse_distance_kj;
        float const gk_y = factor * (p12_y - cos_theta * p32_y) * inverse_distance_kj;
        float const gk_z = factor * (p12_z - cos_theta * p32_z) * inverse_distance_kj;
        *gradient_x -= gi_x + gk_x;
        *gradient_y -= gi_y + gk_y;
        *gradient_z -= gi_z + gk_z;
    }
}

/**
 *  @brief Compute torsion energy and gradient for one atom (Wilson B-matrix).
 *
 *  @param[in] positions SoA coordinate array [x0..xN, y0..yN, z0..zN].
 *  @param[in] num_atoms Number of atoms in the molecule.
 *  @param[in] atom Index of the atom this thread is accumulating for.
 *  @param[in] tor Torsion descriptor with barrier heights V1, V2, V3.
 *  @param[in] role This atom's role in the torsion (first through fourth in atom-list order).
 *  @param[inout] energy Accumulated energy (incremented when @p role is 0).
 *  @param[inout] gradient_x Accumulated x-gradient for @p atom.
 *  @param[inout] gradient_y Accumulated y-gradient for @p atom.
 *  @param[inout] gradient_z Accumulated z-gradient for @p atom.
 */
__device__ __forceinline__ void mmff_torsion_contribution_(float const *positions, std::uint32_t num_atoms,
                                                           std::uint32_t atom, mmff94_torsion_t const &tor,
                                                           atom_role_t role, float *energy, float *gradient_x,
                                                           float *gradient_y, float *gradient_z) {

    // bond vectors; bond_vec_2 = -bond_vec_1 by construction
    float3 const bond_vec_0 = sub_float3_(load_position_(positions, num_atoms, tor.atom_i),
                                          load_position_(positions, num_atoms, tor.atom_j));
    float3 const bond_vec_1 = sub_float3_(load_position_(positions, num_atoms, tor.atom_k),
                                          load_position_(positions, num_atoms, tor.atom_j));
    float3 const bond_vec_2 = scale_float3_(bond_vec_1, -1.0f);
    float3 const bond_vec_3 = sub_float3_(load_position_(positions, num_atoms, tor.atom_l),
                                          load_position_(positions, num_atoms, tor.atom_k));

    float3 normal_0 = cross_float3_(bond_vec_0, bond_vec_1);
    float const inverse_normal_length_0 = rsqrtf(fmaxf(dot_float3_(normal_0, normal_0), 1e-10f));
    normal_0 = scale_float3_(normal_0, inverse_normal_length_0);

    float3 normal_1 = cross_float3_(bond_vec_2, bond_vec_3);
    float const inverse_normal_length_1 = rsqrtf(fmaxf(dot_float3_(normal_1, normal_1), 1e-10f));
    normal_1 = scale_float3_(normal_1, inverse_normal_length_1);

    float const cos_phi = fmaxf(-1.0f, fminf(1.0f, dot_float3_(normal_0, normal_1)));

    // signed sin(φ) from the triple product (normal_0 × normal_1) · bond_vec_1_hat
    float const inv_bond_vec_1_len = rsqrtf(fmaxf(dot_float3_(bond_vec_1, bond_vec_1), 1e-10f));
    float const sin_phi = dot_float3_(cross_float3_(normal_0, normal_1), bond_vec_1) * inv_bond_vec_1_len;

    // energy (uses cos_phi only, sign-independent)
    float const cos2 = 2.0f * cos_phi * cos_phi - 1.0f;
    float const cos3 = cos_phi * (4.0f * cos_phi * cos_phi - 3.0f);
    float const tor_energy = mmff_torsion_prefactor_k *
                             (tor.v1 * (1.0f + cos_phi) + tor.v2 * (1.0f - cos2) + tor.v3 * (1.0f + cos3));

    if (role == atom_role_first_k) *energy += tor_energy;

    // ∂E/∂φ = 0.5 * (-V1·sin(φ) + 2V2·sin(2φ) - 3V3·sin(3φ)) using SIGNED sin
    float const sin2 = 2.0f * sin_phi * cos_phi;
    float const sin3 = sin_phi * (3.0f - 4.0f * sin_phi * sin_phi);
    float const energy_gradient_dihedral = mmff_torsion_prefactor_k *
                                           (-tor.v1 * sin_phi + 2.0f * tor.v2 * sin2 - 3.0f * tor.v3 * sin3);

    float const abs_sin = fabsf(sin_phi);
    if (fabsf(cos_phi) < 1e-8f && abs_sin < 1e-8f) return;
    float const sin_term = (abs_sin < 1e-8f) ? (-energy_gradient_dihedral / cos_phi)
                                             : (-energy_gradient_dihedral / sin_phi);

    float3 const cos_grad_0 = scale_float3_(sub_float3_(normal_1, scale_float3_(normal_0, cos_phi)),
                                            inverse_normal_length_0);
    float3 const cos_grad_1 = scale_float3_(sub_float3_(normal_0, scale_float3_(normal_1, cos_phi)),
                                            inverse_normal_length_1);

    // Wilson B-matrix chain rule for the atom at @p role; identical algebraic
    // structure to compute_torsion_contribution_ in etk_kernel.cuh.
    float3 force {0.0f, 0.0f, 0.0f};
    switch (role) {
    case atom_role_first_k:
        force = cross_float3_(bond_vec_1, cos_grad_0);
        break;
    case atom_role_second_k:
        force = sub_float3_(cross_float3_(cos_grad_0, sub_float3_(bond_vec_1, bond_vec_0)),
                            cross_float3_(cos_grad_1, bond_vec_3));
        break;
    case atom_role_third_k:
        force = add_float3_(cross_float3_(cos_grad_0, bond_vec_0),
                            cross_float3_(cos_grad_1, sub_float3_(bond_vec_3, bond_vec_2)));
        break;
    case atom_role_fourth_k:
        force = cross_float3_(cos_grad_1, bond_vec_2);
        break;
    }
    force = scale_float3_(force, sin_term);
    *gradient_x += force.x;
    *gradient_y += force.y;
    *gradient_z += force.z;
}

/**
 *  @brief OOP energy helper for explicit coordinates.
 *
 *  @param[in] position_i Out-of-plane atom i.
 *  @param[in] position_j Central sp2 atom j.
 *  @param[in] position_k In-plane neighbor k.
 *  @param[in] position_l In-plane neighbor l.
 *  @param[in] oop OOP bending descriptor with force constant.
 *  @return OOP bending energy (kcal/mol).
 */
__device__ __forceinline__ float mmff_oop_energy_(float3 const &position_i, float3 const &position_j,
                                                  float3 const &position_k, float3 const &position_l,
                                                  mmff94_oop_t const &oop) {
    float3 const bond_ji = sub_float3_(position_i, position_j);
    float3 const bond_jk = sub_float3_(position_k, position_j);
    float3 const bond_jl = sub_float3_(position_l, position_j);
    float3 const plane_normal = cross_float3_(bond_jk, bond_jl);
    float const inv_plane_len = rsqrtf(fmaxf(dot_float3_(plane_normal, plane_normal), 1e-12f));
    float const inv_bond_ji_len = rsqrtf(fmaxf(dot_float3_(bond_ji, bond_ji), 1e-12f));
    float const sin_chi = fmaxf(-1.0f, fminf(1.0f, dot_float3_(bond_ji, plane_normal) * inv_bond_ji_len * inv_plane_len));
    float const chi_deg = asinf(sin_chi) * mmff_radians_to_degrees_k;
    return 0.5f * mmff_angle_prefactor_k * oop.force_constant * chi_deg * chi_deg;
}

/**
 *  @brief OOP bending energy and gradient for one atom.
 *
 *  @param[in] positions SoA coordinate array [x0..xN, y0..yN, z0..zN].
 *  @param[in] num_atoms Number of atoms in the molecule.
 *  @param[in] atom Index of the atom this thread is accumulating for (unused in body).
 *  @param[in] oop OOP bending descriptor with atom indices and force constant.
 *  @param[in] role This atom's role (first=out-of-plane, second=center, third/fourth=in-plane).
 *  @param[inout] energy Accumulated energy (incremented when @p role is 0).
 *  @param[inout] gradient_x Accumulated x-gradient for @p atom.
 *  @param[inout] gradient_y Accumulated y-gradient for @p atom.
 *  @param[inout] gradient_z Accumulated z-gradient for @p atom.
 */
__device__ __forceinline__ void mmff_oop_contribution_(float const *positions, std::uint32_t num_atoms,
                                                       std::uint32_t /*atom*/, mmff94_oop_t const &oop,
                                                       atom_role_t role, float *energy, float *gradient_x,
                                                       float *gradient_y, float *gradient_z) {
    float3 const atom_i_position = load_position_(positions, num_atoms, oop.atom_i);
    float3 const atom_j_position = load_position_(positions, num_atoms, oop.atom_j);
    float3 const atom_k_position = load_position_(positions, num_atoms, oop.atom_k);
    float3 const atom_l_position = load_position_(positions, num_atoms, oop.atom_l);

    float const base_energy = mmff_oop_energy_(atom_i_position, atom_j_position, atom_k_position, atom_l_position, oop);
    if (role == atom_role_first_k) *energy += base_energy;

    float3 const bond_ji = sub_float3_(atom_i_position, atom_j_position);
    float3 const bond_jk = sub_float3_(atom_k_position, atom_j_position);
    float3 const bond_jl = sub_float3_(atom_l_position, atom_j_position);
    float3 const plane_normal = cross_float3_(bond_jk, bond_jl);

    float const inverse_bond_ji_length = rsqrtf(fmaxf(dot_float3_(bond_ji, bond_ji), 1e-12f));
    float const inverse_plane_normal_length = rsqrtf(fmaxf(dot_float3_(plane_normal, plane_normal), 1e-12f));

    float3 const bond_ji_unit = scale_float3_(bond_ji, inverse_bond_ji_length);
    float3 const plane_normal_unit = scale_float3_(plane_normal, inverse_plane_normal_length);

    float const sin_chi = fmaxf(-1.0f, fminf(1.0f, dot_float3_(bond_ji_unit, plane_normal_unit)));
    float const chi_deg = asinf(sin_chi) * mmff_radians_to_degrees_k;
    float const inverse_cos_chi = rsqrtf(fmaxf(1.0f - sin_chi * sin_chi, 1e-10f));
    float const energy_gradient_sin_chi = mmff_angle_prefactor_k * oop.force_constant * chi_deg *
                                          mmff_radians_to_degrees_k * inverse_cos_chi;

    float3 const sin_chi_gradient_ji = scale_float3_(
        sub_float3_(plane_normal_unit, scale_float3_(bond_ji_unit, sin_chi)), inverse_bond_ji_length);
    float3 const sin_chi_gradient_normal = scale_float3_(
        sub_float3_(bond_ji_unit, scale_float3_(plane_normal_unit, sin_chi)), inverse_plane_normal_length);
    float3 const sin_chi_gradient_jk = cross_float3_(bond_jl, sin_chi_gradient_normal);
    float3 const sin_chi_gradient_jl = cross_float3_(sin_chi_gradient_normal, bond_jk);

    float3 const gradient_i = scale_float3_(sin_chi_gradient_ji, energy_gradient_sin_chi);
    float3 const gradient_k = scale_float3_(sin_chi_gradient_jk, energy_gradient_sin_chi);
    float3 const gradient_l = scale_float3_(sin_chi_gradient_jl, energy_gradient_sin_chi);
    float3 const gradient_j = scale_float3_(
        add_float3_(add_float3_(sin_chi_gradient_ji, sin_chi_gradient_jk), sin_chi_gradient_jl),
        -energy_gradient_sin_chi);

    switch (role) {
    case atom_role_first_k:
        *gradient_x += gradient_i.x;
        *gradient_y += gradient_i.y;
        *gradient_z += gradient_i.z;
        break;
    case atom_role_second_k:
        *gradient_x += gradient_j.x;
        *gradient_y += gradient_j.y;
        *gradient_z += gradient_j.z;
        break;
    case atom_role_third_k:
        *gradient_x += gradient_k.x;
        *gradient_y += gradient_k.y;
        *gradient_z += gradient_k.z;
        break;
    case atom_role_fourth_k:
        *gradient_x += gradient_l.x;
        *gradient_y += gradient_l.y;
        *gradient_z += gradient_l.z;
        break;
    default: break;
    }
}

/**
 *  @brief Stretch-bend coupling energy and gradient for one atom.
 *
 *  @param[in] positions SoA coordinate array [x0..xN, y0..yN, z0..zN].
 *  @param[in] num_atoms Number of atoms in the molecule.
 *  @param[in] atom Index of the atom this thread is accumulating for.
 *  @param[in] sb Stretch-bend descriptor with coupling constants and equilibrium geometry.
 *  @param[in] role This atom's role (first atom end, second atom (center), third atom end).
 *  @param[inout] energy Accumulated energy (incremented when @p role is 0).
 *  @param[inout] gradient_x Accumulated x-gradient for @p atom.
 *  @param[inout] gradient_y Accumulated y-gradient for @p atom.
 *  @param[inout] gradient_z Accumulated z-gradient for @p atom.
 */
__device__ __forceinline__ void mmff_stretchbend_contribution_(float const *positions, std::uint32_t num_atoms,
                                                               std::uint32_t atom, mmff94_stretchbend_t const &sb,
                                                               atom_role_t role, float *energy, float *gradient_x,
                                                               float *gradient_y, float *gradient_z) {
    // vectors from central atom j
    float3 const position_j = load_position_(positions, num_atoms, sb.atom_j);
    float3 const bond_ij = sub_float3_(load_position_(positions, num_atoms, sb.atom_i), position_j);
    float3 const bond_kj = sub_float3_(load_position_(positions, num_atoms, sb.atom_k), position_j);

    float const bond_ij_len_sq = fmaxf(dot_float3_(bond_ij, bond_ij), 1e-8f);
    float const bond_kj_len_sq = fmaxf(dot_float3_(bond_kj, bond_kj), 1e-8f);
    float const inverse_distance_ij = rsqrtf(bond_ij_len_sq);
    float const inverse_distance_kj = rsqrtf(bond_kj_len_sq);
    float const rij_len = bond_ij_len_sq * inverse_distance_ij;
    float const rkj_len = bond_kj_len_sq * inverse_distance_kj;

    float const cos_theta = fmaxf(-1.0f, fminf(1.0f, dot_float3_(bond_ij, bond_kj) * inverse_distance_ij *
                                                         inverse_distance_kj));
    float const theta = acosf(cos_theta);
    float const sin_theta = sqrtf(fmaxf(1.0f - cos_theta * cos_theta, 1e-10f));

    float const delta_r_ij = rij_len - sb.equilibrium_distance_ij;
    float const delta_r_kj = rkj_len - sb.equilibrium_distance_kj;
    float const delta_theta = (theta - sb.equilibrium_angle) * mmff_radians_to_degrees_k; // in degrees

    // E = 2.5121 * (kba_ijk * Δr_ij + kba_kji * Δr_kj) * Δθ
    float const stretch_term = sb.kba_ijk * delta_r_ij + sb.kba_kji * delta_r_kj;
    if (role == atom_role_first_k) *energy += mmff_stretch_bend_prefactor_k * stretch_term * delta_theta;

    // gradient has two parts: ∂E/∂r (stretch) and ∂E/∂θ (bend)
    // ∂E/∂r_ij = 2.5121 * kba_ijk * Δθ * (r_ij_hat)
    // ∂E/∂θ = 2.5121 * stretch_term * rad2deg * (-1/sin(theta)) * ∂cos/∂r
    float const inverse_sine = __frcp_rn(fmaxf(sin_theta, 1e-6f));
    float const p12_x = bond_ij.x * inverse_distance_ij, p12_y = bond_ij.y * inverse_distance_ij,
                p12_z = bond_ij.z * inverse_distance_ij;
    float const p32_x = bond_kj.x * inverse_distance_kj, p32_y = bond_kj.y * inverse_distance_kj,
                p32_z = bond_kj.z * inverse_distance_kj;

    // stretch gradient component
    float const energy_gradient_distance_ij = mmff_stretch_bend_prefactor_k * sb.kba_ijk * delta_theta;
    float const energy_gradient_distance_kj = mmff_stretch_bend_prefactor_k * sb.kba_kji * delta_theta;

    // bend gradient component: ∂E/∂θ * dtheta/dr
    float const energy_gradient_angle = mmff_stretch_bend_prefactor_k * stretch_term * mmff_radians_to_degrees_k;
    float const bend_factor = energy_gradient_angle * (-inverse_sine);

    if (role == atom_role_first_k) { // first atom:
        *gradient_x += energy_gradient_distance_ij * p12_x +
                       bend_factor * (p32_x - cos_theta * p12_x) * inverse_distance_ij;
        *gradient_y += energy_gradient_distance_ij * p12_y +
                       bend_factor * (p32_y - cos_theta * p12_y) * inverse_distance_ij;
        *gradient_z += energy_gradient_distance_ij * p12_z +
                       bend_factor * (p32_z - cos_theta * p12_z) * inverse_distance_ij;
    }
    else if (role == atom_role_third_k) { // third atom:
        *gradient_x += energy_gradient_distance_kj * p32_x +
                       bend_factor * (p12_x - cos_theta * p32_x) * inverse_distance_kj;
        *gradient_y += energy_gradient_distance_kj * p32_y +
                       bend_factor * (p12_y - cos_theta * p32_y) * inverse_distance_kj;
        *gradient_z += energy_gradient_distance_kj * p32_z +
                       bend_factor * (p12_z - cos_theta * p32_z) * inverse_distance_kj;
    }
    else { // second atom (center): -(grad_i + grad_k)
        float const gi_x = energy_gradient_distance_ij * p12_x +
                           bend_factor * (p32_x - cos_theta * p12_x) * inverse_distance_ij;
        float const gi_y = energy_gradient_distance_ij * p12_y +
                           bend_factor * (p32_y - cos_theta * p12_y) * inverse_distance_ij;
        float const gi_z = energy_gradient_distance_ij * p12_z +
                           bend_factor * (p32_z - cos_theta * p12_z) * inverse_distance_ij;
        float const gk_x = energy_gradient_distance_kj * p32_x +
                           bend_factor * (p12_x - cos_theta * p32_x) * inverse_distance_kj;
        float const gk_y = energy_gradient_distance_kj * p32_y +
                           bend_factor * (p12_y - cos_theta * p32_y) * inverse_distance_kj;
        float const gk_z = energy_gradient_distance_kj * p32_z +
                           bend_factor * (p12_z - cos_theta * p32_z) * inverse_distance_kj;
        *gradient_x -= gi_x + gk_x;
        *gradient_y -= gi_y + gk_y;
        *gradient_z -= gi_z + gk_z;
    }
}

/**
 *  @brief MMFF94 energy and gradient evaluation.
 *
 *  Nonbonded pairs are enumerated on the fly from the `bond_distance` matrix and
 *  their parameters are combined in registers; bonded terms still iterate the
 *  pre-built per-atom CSR (`interactions[]` / `atom_offsets[]`) — those terms
 *  are small and not worth packing out.
 *
 *  @tparam write_gradient When true, computes and writes gradient; when false, energy only.
 *  @see fire_minimize_
 */
template <bool write_gradient>
__device__ __forceinline__ float evaluate_mmff94_energy(
    mmff94_atom_vdw_params_t const *__restrict__ atom_vdw_params, float const *__restrict__ atom_charge,
    std::uint8_t const *__restrict__ bond_distance, mmff94_bond_t const *__restrict__ bond_params,
    mmff94_angle_t const *__restrict__ angle_params, mmff94_torsion_t const *__restrict__ torsion_params,
    mmff94_oop_t const *__restrict__ oop_params, mmff94_stretchbend_t const *__restrict__ stretchbend_params,
    mmff94_atom_interaction_t const *__restrict__ interactions, mmff94_atom_offsets_t const *__restrict__ atom_offsets,
    std::uint32_t num_atoms, float const *positions, float *gradient, std::uint32_t num_elements, std::uint32_t lane) {

    if constexpr (write_gradient) {
        for (std::uint32_t i = lane; i < num_elements; i += warp_size_k) gradient[i] = 0.0f;
        __syncwarp(full_mask_k);
    }

    float total_energy = 0.0f;

    for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k) {
        float gradient_x = 0.0f, gradient_y = 0.0f, gradient_z = 0.0f;
        auto const &offsets = atom_offsets[atom];
        mmff94_atom_vdw_params_t const p_atom = atom_vdw_params[atom];
        float const q_atom = atom_charge[atom];

        // Nonbonded pairs, enumerated from bond_distance. Two passes: "j > atom" owns
        // the pair's energy + adds this atom's gradient contribution as the first
        // endpoint; "j < atom" adds this atom's gradient contribution as the second
        // endpoint (no energy — already counted by the other thread when atom < j).
        for (std::uint32_t j = atom + 1; j < num_atoms; ++j) {
            std::uint8_t const bd = bond_distance[atom * num_atoms + j];
            if (bd < 3 || bd == 255) continue;
            mmff_nonbonded_contribution_(positions, num_atoms, atom, j, p_atom, atom_vdw_params[j],
                                         q_atom * atom_charge[j], bd == 3, /*first_endpoint=*/true, &total_energy,
                                         &gradient_x, &gradient_y, &gradient_z);
        }
        for (std::uint32_t j = 0; j < atom; ++j) {
            std::uint8_t const bd = bond_distance[j * num_atoms + atom];
            if (bd < 3 || bd == 255) continue;
            mmff_nonbonded_contribution_(positions, num_atoms, atom, j, p_atom, atom_vdw_params[j],
                                         q_atom * atom_charge[j], bd == 3, /*first_endpoint=*/false, &total_energy,
                                         &gradient_x, &gradient_y, &gradient_z);
        }

        for (std::uint32_t index = offsets.bond_start; index < offsets.bond_start + offsets.bond_count; ++index)
            mmff_bond_contribution_(positions, num_atoms, atom, bond_params[interactions[index].term_index],
                                    interactions[index].role, &total_energy, &gradient_x, &gradient_y, &gradient_z);

        for (std::uint32_t index = offsets.angle_start; index < offsets.angle_start + offsets.angle_count; ++index)
            mmff_angle_contribution_(positions, num_atoms, atom, angle_params[interactions[index].term_index],
                                     interactions[index].role, &total_energy, &gradient_x, &gradient_y, &gradient_z);

        for (std::uint32_t index = offsets.torsion_start; index < offsets.torsion_start + offsets.torsion_count;
             ++index)
            mmff_torsion_contribution_(positions, num_atoms, atom, torsion_params[interactions[index].term_index],
                                       interactions[index].role, &total_energy, &gradient_x, &gradient_y, &gradient_z);

        for (std::uint32_t index = offsets.oop_start; index < offsets.oop_start + offsets.oop_count; ++index)
            mmff_oop_contribution_(positions, num_atoms, atom, oop_params[interactions[index].term_index],
                                   interactions[index].role, &total_energy, &gradient_x, &gradient_y, &gradient_z);

        for (std::uint32_t index = offsets.stretchbend_start;
             index < offsets.stretchbend_start + offsets.stretchbend_count; ++index)
            mmff_stretchbend_contribution_(positions, num_atoms, atom,
                                           stretchbend_params[interactions[index].term_index], interactions[index].role,
                                           &total_energy, &gradient_x, &gradient_y, &gradient_z);

        if constexpr (write_gradient) {
            gradient[atom] = gradient_x;
            gradient[num_atoms + atom] = gradient_y;
            gradient[2 * num_atoms + atom] = gradient_z;
        }
    }

    return warp_reduce_sum_f32_(total_energy);
}

/**
 *  @brief MMFF94 energy and gradient functor using per-atom interaction lists.
 *
 *  Thin wrapper around evaluate_mmff94_energy that provides the
 *  operator() / energy_only() interface expected by fire_minimize_.
 */
struct mmff94_energy_functor_t {
    mmff94_atom_vdw_params_t const *__restrict__ atom_vdw_params;
    float const *__restrict__ atom_charge;
    std::uint8_t const *__restrict__ bond_distance;
    mmff94_bond_t const *__restrict__ bond_params;
    mmff94_angle_t const *__restrict__ angle_params;
    mmff94_torsion_t const *__restrict__ torsion_params;
    mmff94_oop_t const *__restrict__ oop_params;
    mmff94_stretchbend_t const *__restrict__ stretchbend_params;
    mmff94_atom_interaction_t const *__restrict__ interactions;
    mmff94_atom_offsets_t const *__restrict__ atom_offsets;
    std::uint32_t num_atoms;

    __device__ __forceinline__ float operator()(float const *positions, float *gradient, std::uint32_t num_elements,
                                                std::uint32_t lane) const {
        return evaluate_mmff94_energy<true>(atom_vdw_params, atom_charge, bond_distance, bond_params, angle_params,
                                            torsion_params, oop_params, stretchbend_params, interactions, atom_offsets,
                                            num_atoms, positions, gradient, num_elements, lane);
    }

    __device__ __forceinline__ float energy_only(float const *positions, std::uint32_t num_elements,
                                                 std::uint32_t lane) const {
        return evaluate_mmff94_energy<false>(atom_vdw_params, atom_charge, bond_distance, bond_params, angle_params,
                                             torsion_params, oop_params, stretchbend_params, interactions, atom_offsets,
                                             num_atoms, positions, nullptr, num_elements, lane);
    }
};

} // namespace smallfield::cuda

#endif // __CUDACC__
