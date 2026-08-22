/**
 *  @file src/smallfield/cuda/mmff94_energy_kernel.cuh
 *  @brief MMFF94 energy and gradient functor for ETKDG pipeline stage 9.
 *
 *  Per-atom interaction lists for zero-waste iteration, with full analytical gradients for VdW, electrostatic,
 *  bond, angle, torsion, OOP and stretch-bend terms. Positions in SoA float [x₀..xₙ, y₀..yₙ, z₀..zₙ], and the
 *  functor interface is `fire_minimize_`'s.
 *
 *  Every `mmff_*_contribution_` takes the same shape: SoA @p positions with @p num_atoms as the row stride,
 *  the @p atom this thread accumulates for, that atom's @p role in the term, and the running @p energy —
 *  credited once per term, by role 0 — and @p gradient.
 */

#pragma once

#if defined(__CUDACC__)

#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/cuda/warp_primitives.cuh"

namespace smallfield::cuda {
using smallfield::parameters::mmff94_vdw_asymmetry_amplitude_k;
using smallfield::parameters::mmff94_vdw_asymmetry_decay_k;
using smallfield::parameters::mmff94_vdw_donor_acceptor_radius_scale_k;
using smallfield::parameters::mmff94_vdw_donor_acceptor_well_depth_scale_k;
using smallfield::types::atom_role_first_k;
using smallfield::types::atom_role_fourth_k;
using smallfield::types::atom_role_second_k;
using smallfield::types::atom_role_t;
using smallfield::types::atom_role_third_k;
using smallfield::types::chiral_center_descriptor_t;
using smallfield::types::mmff94_angle_t;
using smallfield::types::mmff94_atom_interaction_t;
using smallfield::types::mmff94_atom_offsets_t;
using smallfield::types::mmff94_atom_vdw_folded_t;
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
constexpr float mmff_angle_cubic_bend_k = -0.006981317f;
constexpr float mmff_linear_bend_prefactor_k = 143.9325f;
constexpr float mmff_torsion_prefactor_k = 0.5f;
constexpr float mmff_stretch_bend_prefactor_k = 2.51210f;
constexpr float mmff_radians_to_degrees_k = 180.0f / 3.14159265f;

/**
 *  Stiffness of the chiral restraint, in kcal/mol per cubic angstrom squared.
 *
 *  It sets the height of the barrier between a stereoisomer and its mirror, since the path between them passes
 *  through a flat center where the penalty is `w * v_min^2` — 2,500 kcal/mol at a four-coordinate center's 5
 *  cubic angstrom floor, far above the deepest descent the polish reaches. The bottom is flat, so it costs
 *  nothing wherever the restraint holds, which is what lets the number be this large without distorting.
 */
constexpr float mmff_chiral_volume_weight_k = 100.0f;

/**
 *  Fraction of a center's `minimum_volume` the center-based volume is held to.
 *
 *  The two volumes a center carries are not the same size, so `minimum_volume` cannot serve both: the center
 *  sits inside the tetrahedron its neighbors span rather than at a corner, so its volume runs several times
 *  smaller — a median 2.57 cubic angstrom against the pivot-based 8.38. The unscaled floor sits above the
 *  37th percentile of that distribution and would bend a third of the corpus to reach it. The fraction measures
 *  that distribution rather than converting between the two — `BENCH.md`, _A Centre Carries Two Signed
 *  Volumes_, carries the percentiles it was read off. A three-coordinate center is its own fourth neighbor, so
 *  its two volumes are one expression and this floor sits below the one already holding it, contributing none.
 */
constexpr float mmff_center_volume_fraction_k = 0.3f;

/** Whether a nonbonded pair is exactly three bonds apart, which scales electrostatics. */
enum class pair_separation_t : bool { beyond_one_four_k = false, one_four_k = true };

/** Reduce @p params to the pair-invariant factors the nonbonded sweep reads. */
__device__ __forceinline__ mmff94_atom_vdw_folded_t fold_atom_vdw_params_(mmff94_atom_vdw_params_t const &params) {
    return {.r_star = params.r_star,
            .polarizability_numerator = params.g_factor * params.alpha,
            .polarizability_root = sqrtf(params.alpha * (1.0f / (params.n_effective))),
            .donor_acceptor = params.donor_acceptor,
            .padding = {0, 0, 0}};
}

/**
 *  @brief Compute one nonbonded pair's (VdW + electrostatic) energy and the force it exerts.
 *
 *  Inlines Halgren's R* combining rule with its donor-suppression clause and Slater-Kirkwood's epsilon rule,
 *  then donor-acceptor scaling: RDKit's `calcUnscaledVdWMinimum`, `calcUnscaledVdWWellDepth` and
 *  `scaleVdWParams` in registers.
 *
 *  @param[in] positions SoA coordinate array [x0..xN, y0..yN, z0..zN]; @p num_atoms is its row stride.
 *  @param[in] atom Index of the atom this thread is accumulating for.
 *  @param[in] other Other atom participating in the pair.
 *  @param[in] p_atom Folded per-atom VdW factors for @p atom.
 *  @param[in] p_other Folded per-atom VdW factors for @p other.
 *  @param[in] charge_product Pre-multiplied q_atom * q_other.
 *  @param[in] separation Whether the pair is 1-4, which scales electrostatics.
 *  @param[inout] energy Accumulated energy; the pair is visited once, so it always adds.
 *  @param[out] force What this pair exerts on @p atom. Its exact negation is the force on @p other,
 *    so one evaluation serves both ends of the pair.
 */
__device__ __forceinline__ void mmff_nonbonded_contribution_(float const *positions, std::uint32_t num_atoms,
                                                             std::uint32_t atom, std::uint32_t other,
                                                             mmff94_atom_vdw_folded_t const &p_atom,
                                                             mmff94_atom_vdw_folded_t const &p_other,
                                                             float charge_product, pair_separation_t separation,
                                                             float *energy, float3 *force) {

    // Halgren R* combining rule with donor-suppression clause, plus Slater-Kirkwood epsilon.
    float const vdw_radius_i = p_atom.r_star;
    float const vdw_radius_j = p_other.r_star;
    float const gamma = (vdw_radius_i - vdw_radius_j) * (1.0f / (vdw_radius_i + vdw_radius_j));
    bool const has_donor = (p_atom.donor_acceptor == 2) || (p_other.donor_acceptor == 2);
    float const factor = has_donor ? 0.0f
                                   : mmff94_vdw_asymmetry_amplitude_k *
                                         (1.0f - expf(-mmff94_vdw_asymmetry_decay_k * gamma * gamma));
    float vdw_radius_combined = 0.5f * (vdw_radius_i + vdw_radius_j) * (1.0f + factor);

    float const vdw_radius_squared = vdw_radius_combined * vdw_radius_combined;
    float const vdw_radius_pow6 = vdw_radius_squared * vdw_radius_squared * vdw_radius_squared;
    float vdw_epsilon = 181.16f * p_atom.polarizability_numerator * p_other.polarizability_numerator *
                        (1.0f / ((p_atom.polarizability_root + p_other.polarizability_root) * vdw_radius_pow6));

    // Donor-acceptor scaling (RDKit's scaleVdWParams when DA pairing is detected).
    bool const is_donor_acceptor_pair = (p_atom.donor_acceptor == 2 && p_other.donor_acceptor == 1) ||
                                        (p_atom.donor_acceptor == 1 && p_other.donor_acceptor == 2);
    if (is_donor_acceptor_pair) {
        vdw_radius_combined *= mmff94_vdw_donor_acceptor_radius_scale_k;
        vdw_epsilon *= mmff94_vdw_donor_acceptor_well_depth_scale_k;
    }
    float const inverse_vdw_radius = 1.0f / vdw_radius_combined;

    // Pair geometry.
    float3 const delta = sub_float3_(load_position_(positions, num_atoms, other),
                                     load_position_(positions, num_atoms, atom));
    float const distance_squared = fmaxf(dot_float3_(delta, delta), 1e-8f);
    float const inverse_distance = rsqrtf(distance_squared);
    float const distance = distance_squared * inverse_distance;

    // VdW buffered 14-7 energy and gradient.
    float const density_ratio = distance * inverse_vdw_radius; // ρ = r / R*
    // `ρ⁶` is the same left-associated product `ρ⁷` already walks through, and the gradient's `ρ⁷ / ρ`
    // is exactly it — so the quotient never becomes a division.
    float const density_ratio_pow6 = density_ratio * density_ratio * density_ratio * density_ratio * density_ratio *
                                     density_ratio;
    float const density_ratio_pow7 = density_ratio_pow6 * density_ratio;
    float const repulsion_denominator = density_ratio + mmff_vdw_delta_k; // ρ + δ
    float const inverse_repulsion_denominator = 1.0f / repulsion_denominator;
    float const repulsion_base = (1.0f + mmff_vdw_delta_k) * inverse_repulsion_denominator;
    float const repulsion_pow7 = repulsion_base * repulsion_base * repulsion_base * repulsion_base * repulsion_base *
                                 repulsion_base * repulsion_base;
    // `1 / (ρ⁷ + γ)` appears once in the energy and squared in the gradient, so one reciprocal serves both.
    float const inverse_attractive_denominator = 1.0f / (density_ratio_pow7 + mmff_vdw_gamma_k);
    float const attractive_term = (1.0f + mmff_vdw_gamma_k) * inverse_attractive_denominator - 2.0f;
    float const energy_vdw = vdw_epsilon * repulsion_pow7 * attractive_term;
    float const d_repulsion = -7.0f * repulsion_pow7 * inverse_repulsion_denominator;
    float const d_attractive = -7.0f * (1.0f + mmff_vdw_gamma_k) * density_ratio_pow6 * inverse_attractive_denominator *
                               inverse_attractive_denominator;
    float const energy_gradient_vdw = vdw_epsilon * (d_repulsion * attractive_term + repulsion_pow7 * d_attractive) *
                                      inverse_vdw_radius;

    // Electrostatic energy and gradient; the buffered reciprocal serves the gradient squared.
    float const scale = separation == pair_separation_t::one_four_k ? mmff_1_4_scale_k : 1.0f;
    float const inverse_buffered_distance = 1.0f / (distance + mmff_electrostatic_buffer_k);
    float const scaled_charge_product = mmff_coulomb_k * charge_product * scale;
    float const energy_electrostatic = scaled_charge_product * inverse_buffered_distance;
    float const energy_gradient_electrostatic = -scaled_charge_product * inverse_buffered_distance *
                                                inverse_buffered_distance;

    *energy += energy_vdw + energy_electrostatic;

    float const energy_gradient_distance = energy_gradient_vdw + energy_gradient_electrostatic;
    *force = scale_float3_(delta, -energy_gradient_distance * inverse_distance);
}

/**
 *  @brief Compute bond stretching energy and gradient for one atom.
 *
 *  @p bond carries the force constant and equilibrium distance; @p role is first or second.
 */
__device__ __forceinline__ void mmff_bond_contribution_(float const *positions, std::uint32_t num_atoms,
                                                        std::uint32_t atom, mmff94_bond_t const &bond, atom_role_t role,
                                                        float *energy, float3 *gradient) {

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
    // bond gradient: same sign convention as nonbonded
    accumulate_gradient_(gradient, scale_float3_(delta, -energy_gradient_distance * inverse_distance));
}

/**
 *  @brief Compute angle bending energy and gradient for one atom.
 *
 *  @p angle carries the force constant and equilibrium angle; @p role is first, second for the center, third.
 */
__device__ __forceinline__ void mmff_angle_contribution_(float const *positions, std::uint32_t num_atoms,
                                                         std::uint32_t atom, mmff94_angle_t const &angle,
                                                         atom_role_t role, float *energy, float3 *gradient) {

    float3 const position_i = load_position_(positions, num_atoms, angle.atom_i);
    float3 const position_j = load_position_(positions, num_atoms, angle.atom_j);
    float3 const position_k = load_position_(positions, num_atoms, angle.atom_k);

    // vectors from central atom j
    float3 const bond_ij = sub_float3_(position_i, position_j);
    float3 const bond_kj = sub_float3_(position_k, position_j);
    float const inverse_distance_ij = rsqrtf(fmaxf(dot_float3_(bond_ij, bond_ij), 1e-8f));
    float const inverse_distance_kj = rsqrtf(fmaxf(dot_float3_(bond_kj, bond_kj), 1e-8f));
    float3 const unit_ij = scale_float3_(bond_ij, inverse_distance_ij);
    float3 const unit_kj = scale_float3_(bond_kj, inverse_distance_kj);

    float const cos_theta = fmaxf(-1.0f, fminf(1.0f, dot_float3_(unit_ij, unit_kj)));
    float const sin_theta_sq = 1.0f - cos_theta * cos_theta;
    float const sin_theta = (sin_theta_sq > 1e-12f) ? sqrtf(sin_theta_sq) : 1e-6f;
    float const theta = acosf(cos_theta);

    float factor;
    if (angle.is_linear) {
        // Linear centers — alkyne carbon, nitrile nitrogen, allene, CO₂ — use E = 143.9325 * ka * (1 + cos θ),
        // because the quadratic form over-stiffens badly here: at θ₀ = 180° and a 60° deviation it returns
        // roughly 1.5× the correct energy. Then ∂E/∂θ = -143.9325 * ka * sin θ, and the −1/sinθ chain rule
        // below cancels the sine exactly, so the factor is constant in θ.
        if (role == atom_role_first_k)
            *energy += mmff_linear_bend_prefactor_k * angle.force_constant * (1.0f + cos_theta);
        factor = mmff_linear_bend_prefactor_k * angle.force_constant;
    }
    else {
        float const dtheta_deg = (theta - angle.equilibrium_angle) * mmff_radians_to_degrees_k;

        // energy (counted once, by role 0 = atom_i)
        if (role == atom_role_first_k)
            *energy += 0.5f * mmff_angle_prefactor_k * angle.force_constant * dtheta_deg * dtheta_deg *
                       (1.0f + mmff_angle_cubic_bend_k * dtheta_deg);

        // ∂E/∂r_i = ∂E/∂θ * ∂cosθ/∂r_i * (-1/sinθ), with rad2deg putting the derivative in the chain rule's units.
        float const energy_gradient_angle = mmff_radians_to_degrees_k * mmff_angle_prefactor_k * angle.force_constant *
                                            dtheta_deg * (1.0f + 1.5f * mmff_angle_cubic_bend_k * dtheta_deg);
        factor = energy_gradient_angle * (-(1.0f / (sin_theta)));
    }

    float3 grad {0.0f, 0.0f, 0.0f};
    if (role == atom_role_first_k) {
        grad = scale_float3_(angle_cosine_gradient_(unit_ij, unit_kj, cos_theta, inverse_distance_ij), factor);
    }
    else if (role == atom_role_third_k) {
        grad = scale_float3_(angle_cosine_gradient_(unit_kj, unit_ij, cos_theta, inverse_distance_kj), factor);
    }
    else { // atom_role_second_k, center: -(grad_first + grad_third)
        float3 const grad_first = scale_float3_(
            angle_cosine_gradient_(unit_ij, unit_kj, cos_theta, inverse_distance_ij), factor);
        float3 const grad_third = scale_float3_(
            angle_cosine_gradient_(unit_kj, unit_ij, cos_theta, inverse_distance_kj), factor);
        grad = scale_float3_(add_float3_(grad_first, grad_third), -1.0f);
    }
    accumulate_gradient_(gradient, grad);
}

/**
 *  @brief Compute torsion energy and gradient for one atom, by the Wilson B-matrix.
 *
 *  @p tor carries the barrier heights V1, V2 and V3; @p role is first through fourth in atom-list order.
 */
__device__ __forceinline__ void mmff_torsion_contribution_(float const *positions, std::uint32_t num_atoms,
                                                           std::uint32_t atom, mmff94_torsion_t const &tor,
                                                           atom_role_t role, float *energy, float3 *gradient) {

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
    float const cos_2phi = 2.0f * cos_phi * cos_phi - 1.0f;
    float const cos_3phi = cos_phi * (4.0f * cos_phi * cos_phi - 3.0f);
    float const tor_energy = mmff_torsion_prefactor_k *
                             (tor.torsion_v1 * (1.0f + cos_phi) + tor.torsion_v2 * (1.0f - cos_2phi) +
                              tor.torsion_v3 * (1.0f + cos_3phi));

    // Both normals collapsing leaves the torsion undefined rather than planar, and an undefined torsion
    // owes neither energy nor force — hence the return above the energy accumulation rather than below.
    if (fabsf(cos_phi) < 1e-8f && fabsf(sin_phi) < 1e-8f) return;

    if (role == atom_role_first_k) *energy += tor_energy;

    // ∂E/∂φ = 0.5 · (-V1·sin(φ) + 2V2·sin(2φ) - 3V3·sin(3φ)) using SIGNED sin
    float const sin_2phi = 2.0f * sin_phi * cos_phi;
    float const sin_3phi = sin_phi * (3.0f - 4.0f * sin_phi * sin_phi);
    float const energy_gradient_dihedral = mmff_torsion_prefactor_k *
                                           (-tor.torsion_v1 * sin_phi + 2.0f * tor.torsion_v2 * sin_2phi -
                                            3.0f * tor.torsion_v3 * sin_3phi);

    // The chain rule wants -(∂E/∂φ)/sin(φ), and at φ = 0 or pi both go to zero because every sin(nφ) does.
    // L'Hopital gives the finite limit -(∂²E/∂φ²)/cos(φ), with
    // ∂²E/∂φ² = 0.5 · (-V1·cos(φ) + 4V2·cos(2φ) - 9V3·cos(3φ)). A `sin φ` this small puts `cos φ` at ±1, so
    // `cos 2φ` is 1, `cos 3φ` shares `cos φ`'s sign, and the quotient collapses to the closed form below.
    // Writing it out keeps the second derivative from ever existing: beside the first it costs eight registers
    // on sm_120, where the polish kernel has none to give, and four even confined to this branch.
    float const planar_sign = copysignf(1.0f, cos_phi);
    float const sin_term = (fabsf(sin_phi) > torsion_planarity_threshold_k)
                               ? (-energy_gradient_dihedral / sin_phi)
                               : (mmff_torsion_prefactor_k *
                                  (tor.torsion_v1 + 9.0f * tor.torsion_v3 - 4.0f * tor.torsion_v2 * planar_sign));

    float3 const cos_grad_0 = scale_float3_(sub_float3_(normal_1, scale_float3_(normal_0, cos_phi)),
                                            inverse_normal_length_0);
    float3 const cos_grad_1 = scale_float3_(sub_float3_(normal_0, scale_float3_(normal_1, cos_phi)),
                                            inverse_normal_length_1);

    // Wilson B-matrix chain rule for @p role; same algebraic structure as `compute_torsion_contribution_`.
    float3 force {0.0f, 0.0f, 0.0f};
    switch (role) {
    case atom_role_first_k: force = cross_float3_(bond_vec_1, cos_grad_0); break;
    case atom_role_second_k:
        force = sub_float3_(cross_float3_(cos_grad_0, sub_float3_(bond_vec_1, bond_vec_0)),
                            cross_float3_(cos_grad_1, bond_vec_3));
        break;
    case atom_role_third_k:
        force = add_float3_(cross_float3_(cos_grad_0, bond_vec_0),
                            cross_float3_(cos_grad_1, sub_float3_(bond_vec_3, bond_vec_2)));
        break;
    case atom_role_fourth_k: force = cross_float3_(cos_grad_1, bond_vec_2); break;
    }
    force = scale_float3_(force, sin_term);
    accumulate_gradient_(gradient, force);
}

/**
 *  @brief OOP bending energy and gradient for one atom.
 *
 *  Energy and gradient share the sin_chi and inverse-length intermediates, so both run together with the
 *  energy increment gated on @p role. @p oop carries the atom indices and force constant, and @p role is
 *  first for out-of-plane, second for the center, third and fourth in-plane.
 */
__device__ __forceinline__ void mmff_oop_contribution_(float const *positions, std::uint32_t num_atoms,
                                                       std::uint32_t /*atom*/, mmff94_oop_t const &oop,
                                                       atom_role_t role, float *energy, float3 *gradient) {
    float3 const atom_i_position = load_position_(positions, num_atoms, oop.atom_i);
    float3 const atom_j_position = load_position_(positions, num_atoms, oop.atom_j);
    float3 const atom_k_position = load_position_(positions, num_atoms, oop.atom_k);
    float3 const atom_l_position = load_position_(positions, num_atoms, oop.atom_l);

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

    if (role == atom_role_first_k) *energy += 0.5f * mmff_angle_prefactor_k * oop.force_constant * chi_deg * chi_deg;

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
    case atom_role_first_k: accumulate_gradient_(gradient, gradient_i); break;
    case atom_role_second_k: accumulate_gradient_(gradient, gradient_j); break;
    case atom_role_third_k: accumulate_gradient_(gradient, gradient_k); break;
    case atom_role_fourth_k: accumulate_gradient_(gradient, gradient_l); break;
    default: break;
    }
}

/**
 *  @brief Stretch-bend coupling energy and gradient for one atom.
 *
 *  @p sb carries the coupling constants and equilibrium geometry; @p role is first end, second for the
 *  center, third end.
 */
__device__ __forceinline__ void mmff_stretchbend_contribution_(float const *positions, std::uint32_t num_atoms,
                                                               std::uint32_t atom, mmff94_stretchbend_t const &sb,
                                                               atom_role_t role, float *energy, float3 *gradient) {
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

    float const cos_theta = fmaxf(
        -1.0f, fminf(1.0f, dot_float3_(bond_ij, bond_kj) * inverse_distance_ij * inverse_distance_kj));
    float const theta = acosf(cos_theta);
    float const sin_theta = sqrtf(fmaxf(1.0f - cos_theta * cos_theta, 1e-10f));

    float const delta_r_ij = rij_len - sb.equilibrium_distance_ij;
    float const delta_r_kj = rkj_len - sb.equilibrium_distance_kj;
    float const delta_theta = (theta - sb.equilibrium_angle) * mmff_radians_to_degrees_k; // in degrees

    // E = 2.5121 * (kba_ijk * Δr_ij + kba_kji * Δr_kj) * Δθ
    float const stretch_term = sb.kba_ijk * delta_r_ij + sb.kba_kji * delta_r_kj;
    if (role == atom_role_first_k) *energy += mmff_stretch_bend_prefactor_k * stretch_term * delta_theta;

    // gradient has two parts: ∂E/∂r (stretch) and ∂E/∂θ (bend)
    float3 const unit_ij = scale_float3_(bond_ij, inverse_distance_ij);
    float3 const unit_kj = scale_float3_(bond_kj, inverse_distance_kj);

    // stretch gradient component (along the unit vector pointing away from j)
    float const energy_gradient_distance_ij = mmff_stretch_bend_prefactor_k * sb.kba_ijk * delta_theta;
    float const energy_gradient_distance_kj = mmff_stretch_bend_prefactor_k * sb.kba_kji * delta_theta;

    // bend gradient component: ∂E/∂θ * dθ/dr
    float const energy_gradient_angle = mmff_stretch_bend_prefactor_k * stretch_term * mmff_radians_to_degrees_k;
    float const bend_factor = energy_gradient_angle * (-(1.0f / (fmaxf(sin_theta, 1e-6f))));

    auto stretch_plus_bend_first_ = [&]() {
        return add_float3_(
            scale_float3_(unit_ij, energy_gradient_distance_ij),
            scale_float3_(angle_cosine_gradient_(unit_ij, unit_kj, cos_theta, inverse_distance_ij), bend_factor));
    };
    auto stretch_plus_bend_third_ = [&]() {
        return add_float3_(
            scale_float3_(unit_kj, energy_gradient_distance_kj),
            scale_float3_(angle_cosine_gradient_(unit_kj, unit_ij, cos_theta, inverse_distance_kj), bend_factor));
    };

    float3 grad {0.0f, 0.0f, 0.0f};
    if (role == atom_role_first_k) { grad = stretch_plus_bend_first_(); }
    else if (role == atom_role_third_k) { grad = stretch_plus_bend_third_(); }
    else { // second atom (center): -(grad_first + grad_third)
        grad = scale_float3_(add_float3_(stretch_plus_bend_first_(), stretch_plus_bend_third_()), -1.0f);
    }
    accumulate_gradient_(gradient, grad);
}

/**
 *  @brief One declared chiral center's restraint contribution to @p atom, added to the MMFF94 terms.
 *
 *  Published MMFF94 has no chiral term, so its energy is blind to which of two mirror images it is
 *  scoring. The restraint is what makes the declared handedness cost something.
 *
 *  @verbatim
 *  N         = (r_1 - r_0) x (r_2 - r_0),  r_i = neighbor_atoms[i], c = center_atom
 *  V_pivot   = (r_0 - r_3) . N
 *  V_center  = (r_0 - c)   . N
 *  E_chiral  = w * (V_pivot - s * v_min)^2 + w * (V_center - s * f * v_min)^2
 *  @endverbatim
 *
 *  Both volumes are restrained because two stages read different ones: the embedding and
 *  `count_invalid_chiral_centers_` hold the pivot-based `DistGeom::ChiralSet` expression, RDKit's
 *  `assignChiralTypesFrom3D` the center-based one off the shipped coordinates. The pivot term puts no force on
 *  the center atom, so under it alone a minimizer satisfies every constraint given and still walks the center
 *  through its neighbors' plane, inverting the perceived handedness.
 *
 *  Both floors are signed, putting an inverted center and a correctly handed but nearly flat one on one axis
 *  whose distance is already the derivative, and both are flat-bottomed, so a correctly handed center far
 *  enough from planar leaves the geometry MMFF94 would have reached untouched. The pivot floor is
 *  `minimum_volume` unslacked where `count_invalid_chiral_centers_` calls a center flat below
 *  `chiral_volume_slack_k` of it, so the restraint pushes past the verdict rather than to it.
 *
 *  @param[in] positions SoA coordinate array; @p num_atoms is the stride between its dimensions.
 *  @param[in] atom The atom whose energy share and gradient this call accumulates.
 *  @param[in] center The declared center.
 *  @param[inout] energy Running energy, credited once per center by its central atom.
 *  @param[inout] gradient Running gradient of @p atom.
 */
__device__ __forceinline__ void mmff_chiral_contribution_(float const *positions, std::uint32_t num_atoms,
                                                          std::uint32_t atom, chiral_center_descriptor_t const &center,
                                                          float *energy, float3 *gradient) {
    std::uint8_t const neighbor_first = center.neighbor_atoms[0];
    std::uint8_t const neighbor_second = center.neighbor_atoms[1];
    std::uint8_t const neighbor_third = center.neighbor_atoms[2];
    std::uint8_t const pivot = center.neighbor_atoms[3];

    // Every lane walks every center, so the five atoms that carry a force are found by comparison;
    // the rest of the molecule leaves before loading a position.
    bool const scores = atom == center.center_atom;
    bool const on_pivot = atom == pivot;
    if (!scores && !on_pivot && atom != neighbor_first && atom != neighbor_second && atom != neighbor_third) return;

    float3 const position_first = load_position_(positions, num_atoms, neighbor_first);
    float3 const position_second = load_position_(positions, num_atoms, neighbor_second);
    float3 const position_third = load_position_(positions, num_atoms, neighbor_third);
    float3 const pivot_position = load_position_(positions, num_atoms, pivot);
    float3 const center_position = load_position_(positions, num_atoms, center.center_atom);
    float3 const normal = cross_float3_(sub_float3_(position_second, position_first),
                                        sub_float3_(position_third, position_first));

    // A shortfall is zero wherever the flat bottom reaches, and is the term's derivative elsewhere.
    float const pivot_floor = center.target_sign < 0 ? -center.minimum_volume : center.minimum_volume;
    float const center_floor = mmff_center_volume_fraction_k * pivot_floor;
    float const pivot_gap = dot_float3_(sub_float3_(position_first, pivot_position), normal) - pivot_floor;
    float const center_gap = dot_float3_(sub_float3_(position_first, center_position), normal) - center_floor;
    float const pivot_shortfall = signbit(pivot_floor) == signbit(pivot_gap) ? 0.0f : pivot_gap;
    float const center_shortfall = signbit(center_floor) == signbit(center_gap) ? 0.0f : center_gap;
    if (pivot_shortfall == 0.0f && center_shortfall == 0.0f) return;

    if (scores)
        *energy += mmff_chiral_volume_weight_k *
                   (pivot_shortfall * pivot_shortfall + center_shortfall * center_shortfall);

    float const pivot_scale = 2.0f * mmff_chiral_volume_weight_k * pivot_shortfall;
    float const center_scale = 2.0f * mmff_chiral_volume_weight_k * center_shortfall;

    // An apex takes the normal negated. A three-coordinate center is stored as its own pivot, so it is the
    // apex of both terms and collects both; its two volumes are one expression and the smaller floor never binds.
    float3 delta {0.0f, 0.0f, 0.0f};
    if (on_pivot) delta = scale_float3_(normal, -pivot_scale);
    if (scores) delta = add_float3_(delta, scale_float3_(normal, -center_scale));
    if (!scores && !on_pivot) {
        // A neighbor takes the cross product of the other two edges as measured from the apex, once
        // per term, and the two apexes give different edges.
        auto neighbor_delta_ = [&](float3 apex, float scale) -> float3 {
            float3 const edge_first = sub_float3_(position_first, apex);
            float3 const edge_second = sub_float3_(position_second, apex);
            float3 const edge_third = sub_float3_(position_third, apex);
            float3 cross_pair = cross_float3_(edge_second, edge_third);
            if (atom == neighbor_second) { cross_pair = cross_float3_(edge_third, edge_first); }
            else if (atom == neighbor_third) { cross_pair = cross_float3_(edge_first, edge_second); }
            return scale_float3_(cross_pair, scale);
        };
        delta = add_float3_(neighbor_delta_(pivot_position, pivot_scale),
                            neighbor_delta_(center_position, center_scale));
    }
    *gradient = add_float3_(*gradient, delta);
}

/**
 *  @brief MMFF94 energy and gradient over per-atom interaction lists, in the shape `fire_minimize_` calls.
 *
 *  Bonded terms iterate the pre-built per-atom CSR (`interactions[]` / `atom_offsets[]`) first, then nonbonded
 *  pairs are enumerated on the fly from the `bond_distance` matrix, once per pair, with their parameters
 *  combined in registers; that order is what keeps the bonded write a plain store. `chiral_centers` may be null
 *  exactly when `num_chiral_centers` is zero.
 *
 *  @see fire_minimize_
 */
struct mmff94_energy_functor_t {
    mmff94_atom_vdw_folded_t const *__restrict__ atom_vdw_params;
    float const *__restrict__ atom_charge;
    std::uint8_t const *__restrict__ bond_distance;
    mmff94_bond_t const *__restrict__ bond_params;
    mmff94_angle_t const *__restrict__ angle_params;
    mmff94_torsion_t const *__restrict__ torsion_params;
    mmff94_oop_t const *__restrict__ oop_params;
    mmff94_stretchbend_t const *__restrict__ stretchbend_params;
    mmff94_atom_interaction_t const *__restrict__ interactions;
    mmff94_atom_offsets_t const *__restrict__ atom_offsets;
    chiral_center_descriptor_t const *__restrict__ chiral_centers;
    std::uint32_t num_atoms;
    std::uint32_t num_chiral_centers;

    __device__ __forceinline__ float operator()(float const *positions, float *gradient, std::uint32_t num_elements,
                                                std::uint32_t lane) const {

        for (std::uint32_t element = lane; element < num_elements; element += warp_size_k) gradient[element] = 0.0f;
        __syncwarp(full_mask_k);

        float total_energy = 0.0f;

        // Bonded terms walk the per-atom CSR, one lane per atom, writing only the three words its own atom
        // owns; the pair sweep below is the only pass that reaches a partner, so this write stays a plain store.
        for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k) {
            float3 gradient_acc {0.0f, 0.0f, 0.0f};
            auto const &offsets = atom_offsets[atom];

            for (std::uint32_t index = offsets.bond_offset; index < offsets.bond_offset + offsets.bond_count; ++index)
                mmff_bond_contribution_(positions, num_atoms, atom, bond_params[interactions[index].term_index],
                                        interactions[index].role, &total_energy, &gradient_acc);

            for (std::uint32_t index = offsets.angle_offset; index < offsets.angle_offset + offsets.angle_count;
                 ++index)
                mmff_angle_contribution_(positions, num_atoms, atom, angle_params[interactions[index].term_index],
                                         interactions[index].role, &total_energy, &gradient_acc);

            for (std::uint32_t index = offsets.torsion_offset; index < offsets.torsion_offset + offsets.torsion_count;
                 ++index)
                mmff_torsion_contribution_(positions, num_atoms, atom, torsion_params[interactions[index].term_index],
                                           interactions[index].role, &total_energy, &gradient_acc);

            for (std::uint32_t index = offsets.oop_offset; index < offsets.oop_offset + offsets.oop_count; ++index)
                mmff_oop_contribution_(positions, num_atoms, atom, oop_params[interactions[index].term_index],
                                       interactions[index].role, &total_energy, &gradient_acc);

            for (std::uint32_t index = offsets.stretchbend_offset;
                 index < offsets.stretchbend_offset + offsets.stretchbend_count; ++index)
                mmff_stretchbend_contribution_(positions, num_atoms, atom,
                                               stretchbend_params[interactions[index].term_index],
                                               interactions[index].role, &total_energy, &gradient_acc);

            // The restraint rides this sweep rather than a pass of its own: the gradient is already
            // accumulating in a register and the store below is still plain, into the slot this lane owns, so
            // the four atoms a center moves need no barrier and no atomic. Its own sweep costs `sm_100` one more.
            for (std::uint32_t center = 0; center < num_chiral_centers; ++center)
                mmff_chiral_contribution_(positions, num_atoms, atom, chiral_centers[center], &total_energy,
                                          &gradient_acc);

            store_gradient_(gradient, num_atoms, atom, gradient_acc);
        }
        __syncwarp(full_mask_k);

        // Nonbonded pairs are enumerated from `bond_distance`, which a warp BFS over the undirected bond graph
        // fills (cuda/bond_distance_kernel.cuh), so it is symmetric and one endpoint's read answers for both.
        // Walking `other = (atom + offset) mod num_atoms` over the first half of the ring visits every pair
        // exactly once with all 32 lanes productive, where an `other > atom` test would idle half the warp's
        // issue slots on a warp-uniform predicate. The partner's share of the force is the exact negation of
        // this atom's, so one evaluation delivers both, and at a fixed offset the 32 lanes address 32 distinct
        // atoms whatever the atom count, so writing the partner's share needs no atomic.
        std::uint32_t const half_shell = num_atoms / 2;
        // An even atom count makes the last offset its own mirror, since atom and atom + N/2 reach each
        // other from both ends, so only the lower half of the ring speaks for that offset.
        bool const middle_offset_is_mirrored = (num_atoms % 2u) == 0u;

        // The partner write below lands in a slot another lane owns, so the "32 distinct atoms" argument holds
        // only while every lane is on the same `offset`. Both trip counts are warp-uniform and every test
        // inside is a predicate, not a `continue`: lanes at different offsets can target one atom — lane 2 at
        // offset 5 and lane 0 at offset 7 both address atom 7 — and independent thread scheduling promises no
        // reconvergence. As written all 32 lanes reach the barrier between one offset's writes and the next's.
        for (std::uint32_t base = 0; base < num_atoms; base += warp_size_k) {
            std::uint32_t const atom = base + lane;
            bool const owns_atom = atom < num_atoms;
            mmff94_atom_vdw_folded_t const p_atom = owns_atom ? atom_vdw_params[atom] : mmff94_atom_vdw_folded_t {};
            float const q_atom = owns_atom ? atom_charge[atom] : 0.0f;
            float3 own_force {0.0f, 0.0f, 0.0f};

            for (std::uint32_t offset = 1; offset <= half_shell; ++offset) {
                bool const mirrored_out = middle_offset_is_mirrored && offset == half_shell && atom >= half_shell;
                std::uint32_t const wrapped = atom + offset;
                std::uint32_t const other = wrapped < num_atoms ? wrapped : wrapped - num_atoms;
                bool const addressable = owns_atom && !mirrored_out;
                std::uint8_t const bond_hops = addressable ? bond_distance[atom * num_atoms + other] : 0u;
                // Bonded, geminal and unreachable pairs contribute nothing, and neither does a lane
                // holding no atom — all three arrive here as a zero force rather than as a skipped lane.
                bool const contributes = addressable && bond_hops >= 3 && bond_hops != 255;
                float3 pair_force {0.0f, 0.0f, 0.0f};
                if (contributes)
                    mmff_nonbonded_contribution_(
                        positions, num_atoms, atom, other, p_atom, atom_vdw_params[other], q_atom * atom_charge[other],
                        bond_hops == 3 ? pair_separation_t::one_four_k : pair_separation_t::beyond_one_four_k,
                        &total_energy, &pair_force);
                own_force = add_float3_(own_force, pair_force);
                if (contributes)
                    accumulate_gradient_into_(gradient, num_atoms, other, scale_float3_(pair_force, -1.0f));
                __syncwarp(full_mask_k);
            }
            if (owns_atom) accumulate_gradient_into_(gradient, num_atoms, atom, own_force);
            // This write is to the lane's own atom, but the next outer iteration's partner writes are not and
            // one can target exactly this slot; the inner barrier does not cover the gap between the loops.
            __syncwarp(full_mask_k);
        }

        return warp_reduce_sum_f32_(total_energy);
    }
};

} // namespace smallfield::cuda

#endif // __CUDACC__
