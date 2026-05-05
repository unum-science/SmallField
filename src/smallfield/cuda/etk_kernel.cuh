/**
 *  @file etk_kernel.cuh
 *  @brief Atom-centric ETK torsion energy and gradient for warp-level L-BFGS.
 *
 *  Implements CSD-derived torsion Fourier potentials for ETKDG stage 7:
 *   E_torsion    = Σₖ Σₙ₌₁⁶ Vₙ·cos(n·φₖ)
 *   E_constraint = Σ_c w_c·max(0, d_c − u_c)² + w_c·max(0, l_c − d_c)²
 *
 *  Architecture: atom-centric with __sincosf + Chebyshev recurrence.
 *  Each thread owns ceil(N/32) atoms, iterates all torsions/constraints,
 *  accumulates gradient in registers. No atomicAdd.
 *
 *  Positions layout (SoA): [x₀..xₙ, y₀..yₙ, z₀..zₙ] (3D only).
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
using smallfield::types::atom_role_third_k;
using smallfield::types::distance_constraint_t;
using smallfield::types::etk_atom_interaction_t;
using smallfield::types::etk_atom_offsets_t;
using smallfield::types::etk_torsion_descriptor_t;
using smallfield::types::etk_torsion_k;

/** Compute dihedral angle φ for atoms i-j-k-l. Returns radians in [-π, π]. */
__device__ __forceinline__ float compute_dihedral_angle_(float const *positions, std::uint32_t num_atoms,
                                                         std::uint8_t atom_i, std::uint8_t atom_j, std::uint8_t atom_k,
                                                         std::uint8_t atom_l) {
    float3 const position_i = load_position_(positions, num_atoms, atom_i);
    float3 const position_j = load_position_(positions, num_atoms, atom_j);
    float3 const position_k = load_position_(positions, num_atoms, atom_k);
    float3 const position_l = load_position_(positions, num_atoms, atom_l);

    float3 const bond_ij = sub_float3_(position_j, position_i);
    float3 const bond_jk = sub_float3_(position_k, position_j);
    float3 const bond_kl = sub_float3_(position_l, position_k);

    float3 const normal_ijk = cross_float3_(bond_ij, bond_jk);
    float3 const normal_jkl = cross_float3_(bond_jk, bond_kl);

    float const dot_normals = dot_float3_(normal_ijk, normal_jkl);
    float const triple_product = dot_float3_(cross_float3_(normal_ijk, normal_jkl), bond_jk);

    float const inverse_bond_jk_length = rsqrtf(fmaxf(dot_float3_(bond_jk, bond_jk), 1e-10f));

    return atan2f(triple_product * inverse_bond_jk_length, dot_normals);
}

/**
 *  @brief Compute torsion energy and per-atom gradient using the Wilson B-matrix approach.
 *
 *  Uses __sincosf + Chebyshev recurrence for the 6-term Fourier series.
 *  Accumulates gradient for the specified @p atom_role (first through fourth)
 *  into the provided @p grad_x / @p grad_y / @p grad_z accumulators.
 *
 *  @tparam write_gradient When true, computes and writes gradient; when false, energy only.
 *  @param[in] positions SoA coordinate array [x0..xN, y0..yN, z0..zN].
 *  @param[in] num_atoms Number of atoms in the molecule.
 *  @param[in] torsion Torsion descriptor with atom indices and Fourier coefficients.
 *  @param[in] atom_role Which atom this thread is accumulating for (first through fourth).
 *  @param[out] grad_x Accumulated x-gradient for the atom.
 *  @param[out] grad_y Accumulated y-gradient for the atom.
 *  @param[out] grad_z Accumulated z-gradient for the atom.
 *  @return Torsion energy contribution (kcal/mol).
 */
template <bool write_gradient>
__device__ __forceinline__ float compute_torsion_contribution_(float const *positions, std::uint32_t num_atoms,
                                                               etk_torsion_descriptor_t const &torsion,
                                                               std::uint32_t atom_role, float *grad_x, float *grad_y,
                                                               float *grad_z) {
    std::uint8_t const atom_i = torsion.atom_indices[0];
    std::uint8_t const atom_j = torsion.atom_indices[1];
    std::uint8_t const atom_k = torsion.atom_indices[2];
    std::uint8_t const atom_l = torsion.atom_indices[3];

    // Bond vectors used by both the dihedral and the Wilson B-matrix chain rule.
    // bond_vec_2 = -bond_vec_1 by construction; kept as a separate name so the
    // switch arms below read the way the Wilson derivation is written.
    float3 const bond_vec_0 = sub_float3_(load_position_(positions, num_atoms, atom_i),
                                          load_position_(positions, num_atoms, atom_j));
    float3 const bond_vec_1 = sub_float3_(load_position_(positions, num_atoms, atom_k),
                                          load_position_(positions, num_atoms, atom_j));
    float3 const bond_vec_2 = scale_float3_(bond_vec_1, -1.0f);
    float3 const bond_vec_3 = sub_float3_(load_position_(positions, num_atoms, atom_l),
                                          load_position_(positions, num_atoms, atom_k));

    float3 normal_0 = cross_float3_(bond_vec_0, bond_vec_1);
    float const inverse_normal_length_0 = rsqrtf(fmaxf(dot_float3_(normal_0, normal_0), 1e-10f));
    normal_0 = scale_float3_(normal_0, inverse_normal_length_0);

    float3 normal_1 = cross_float3_(bond_vec_2, bond_vec_3);
    float const inverse_normal_length_1 = rsqrtf(fmaxf(dot_float3_(normal_1, normal_1), 1e-10f));
    normal_1 = scale_float3_(normal_1, inverse_normal_length_1);

    float const cos_phi = fmaxf(-1.0f, fminf(1.0f, dot_float3_(normal_0, normal_1)));
    float const sin_phi_sq = 1.0f - cos_phi * cos_phi;
    float const sin_phi = (sin_phi_sq > 0.0f) ? sqrtf(sin_phi_sq) : 0.0f;

    // Chebyshev recurrence for 6-term Fourier series
    float energy = 0.0f;
    float energy_gradient_dihedral = 0.0f;
    float energy_second_gradient_dihedral = 0.0f;
    float sin_prev = 0.0f, cos_prev = 1.0f;
    float sin_curr = sin_phi, cos_curr = cos_phi;

    for (std::uint32_t n = 1; n <= 6; ++n) {
        float const coeff = torsion.fourier_coefficients[n - 1];
        if (coeff != 0.0f) {
            float const nf = static_cast<float>(n);
            energy += coeff * cos_curr;
            energy_gradient_dihedral -= nf * coeff * sin_curr;
            energy_second_gradient_dihedral -= nf * nf * coeff * cos_curr;
        }
        // Chebyshev: sin((n+1)phi) = 2cos(φ)sin(nφ) - sin((n-1)phi)
        float const sin_next = 2.0f * cos_phi * sin_curr - sin_prev;
        float const cos_next = 2.0f * cos_phi * cos_curr - cos_prev;
        sin_prev = sin_curr;
        cos_prev = cos_curr;
        sin_curr = sin_next;
        cos_curr = cos_next;
    }

    if constexpr (!write_gradient) return energy;

    // sin_term = -energy_gradient_dihedral / sin(φ), needed for the chain rule ∂E/∂r = sin_term * ∂cos(φ)/∂r
    // at phi = 0 or pi, sin(φ) → 0 but energy_gradient_dihedral → 0 too (sin(nφ) = 0 for integer n).
    // L'Hopital: lim[sin->0] energy_gradient_dihedral/sin(φ) = ∂²E/∂φ² / cos(φ)
    float const sin_term = (fabsf(sin_phi) > 1e-6f)
                               ? (-energy_gradient_dihedral / sin_phi)
                               : (-energy_second_gradient_dihedral / fmaxf(fabsf(cos_phi), 1e-10f));

    float3 const cos_grad_0 = scale_float3_(sub_float3_(normal_1, scale_float3_(normal_0, cos_phi)),
                                            inverse_normal_length_0);
    float3 const cos_grad_1 = scale_float3_(sub_float3_(normal_0, scale_float3_(normal_1, cos_phi)),
                                            inverse_normal_length_1);

    // Wilson B-matrix chain rule for the atom at @p atom_role.  Each role's
    // gradient is a sum of one or two cross products of the cosine gradients
    // and bond vectors (or differences of bond vectors at the central atoms
    // j and k).  The four force expressions sum to zero by Newton's 3rd law.
    float3 force {0.0f, 0.0f, 0.0f};
    switch (atom_role) {
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
    *grad_x += force.x;
    *grad_y += force.y;
    *grad_z += force.z;
    return energy;
}

/**
 *  @brief Atom-centric ETK torsion energy and gradient evaluation.
 *
 *  Each thread owns atoms, iterates all torsions checking participation.
 *  No atomicAdd.
 *
 *  @tparam write_gradient When true, computes and writes gradient; when false, energy only.
 *  @param[in] torsion_terms Array of torsion descriptors.
 *  @param[in] distance_constraints Array of distance constraint descriptors.
 *  @param[in] interactions Per-atom interaction list entries.
 *  @param[in] atom_offsets Per-atom offsets into the interaction list.
 *  @param[in] num_atoms Number of atoms in the molecule.
 *  @param[in] positions SoA coordinate array [x0..xN, y0..yN, z0..zN].
 *  @param[out] gradient Output gradient array (same layout as positions), unused when !write_gradient.
 *  @param[in] num_elements Total number of coordinate elements.
 *  @param[in] lane Warp lane index (0-31).
 *  @return Total energy (same value in all lanes after warp reduction).
 *  @see fire_minimize_
 */
template <bool write_gradient>
__device__ __forceinline__ float evaluate_etk_torsion_energy(
    etk_torsion_descriptor_t const *__restrict__ torsion_terms,
    distance_constraint_t const *__restrict__ distance_constraints,
    etk_atom_interaction_t const *__restrict__ interactions, etk_atom_offsets_t const *__restrict__ atom_offsets,
    std::uint32_t num_atoms, float const *positions, float *gradient, std::uint32_t num_elements, std::uint32_t lane) {

    if constexpr (write_gradient) {
        for (std::uint32_t i = lane; i < num_elements; i += warp_size_k) gradient[i] = 0.0f;
        __syncwarp(full_mask_k);
    }

    float total_energy = 0.0f;

    for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k) {
        float gradient_x = 0.0f, gradient_y = 0.0f, gradient_z = 0.0f;

        std::uint32_t const interaction_start = atom_offsets[atom].interaction_start;
        std::uint32_t const interaction_count = atom_offsets[atom].interaction_count;

        for (std::uint32_t index = interaction_start; index < interaction_start + interaction_count; ++index) {
            auto const &entry = interactions[index];

            if (entry.interaction_type == etk_torsion_k) {
                // torsion term
                float torsion_energy = compute_torsion_contribution_<write_gradient>(
                    positions, num_atoms, torsion_terms[entry.term_index], entry.role, &gradient_x, &gradient_y,
                    &gradient_z);
                if (entry.role == atom_role_first_k) total_energy += torsion_energy;
            }
            else {
                // distance constraint
                auto const &constraint = distance_constraints[entry.term_index];
                std::uint32_t const atom_i = constraint.atom_index_i;
                std::uint32_t const atom_j = constraint.atom_index_j;

                float3 const delta = sub_float3_(load_position_(positions, num_atoms, atom_j),
                                                 load_position_(positions, num_atoms, atom_i));
                float const distance_squared = dot_float3_(delta, delta);
                float const inverse_distance = rsqrtf(fmaxf(distance_squared, 1e-10f));
                float const distance = distance_squared * inverse_distance;

                float const upper_bound = constraint.upper_bound_angstrom;
                float const lower_bound = constraint.lower_bound_angstrom;
                float const weight = constraint.penalty_weight;
                float const direction = (entry.role == atom_role_first_k) ? -1.0f : 1.0f;

                if (distance > upper_bound) {
                    if (entry.role == atom_role_first_k)
                        total_energy += weight * (distance - upper_bound) * (distance - upper_bound);
                    if constexpr (write_gradient) {
                        float const scale = 2.0f * weight * (distance - upper_bound) * inverse_distance;
                        gradient_x += direction * scale * delta.x;
                        gradient_y += direction * scale * delta.y;
                        gradient_z += direction * scale * delta.z;
                    }
                }
                else if (distance < lower_bound) {
                    if (entry.role == atom_role_first_k)
                        total_energy += weight * (lower_bound - distance) * (lower_bound - distance);
                    if constexpr (write_gradient) {
                        float const scale = -2.0f * weight * (lower_bound - distance) * inverse_distance;
                        gradient_x += direction * scale * delta.x;
                        gradient_y += direction * scale * delta.y;
                        gradient_z += direction * scale * delta.z;
                    }
                }
            }
        }

        if constexpr (write_gradient) {
            gradient[atom] = gradient_x;
            gradient[num_atoms + atom] = gradient_y;
            gradient[2 * num_atoms + atom] = gradient_z;
        }
    }

    return warp_reduce_sum_f32_(total_energy);
}

/**
 *  @brief Atom-centric ETK torsion energy and gradient functor.
 *
 *  Thin wrapper around evaluate_etk_torsion_energy that provides the
 *  operator() / energy_only() interface expected by fire_minimize_.
 */
struct etk_torsion_energy_functor_t {
    etk_torsion_descriptor_t const *__restrict__ torsion_terms;
    distance_constraint_t const *__restrict__ distance_constraints;
    etk_atom_interaction_t const *__restrict__ interactions;
    etk_atom_offsets_t const *__restrict__ atom_offsets;
    std::uint32_t num_atoms;

    __device__ __forceinline__ float operator()(float const *positions, float *gradient, std::uint32_t num_elements,
                                                std::uint32_t lane) const {
        return evaluate_etk_torsion_energy<true>(torsion_terms, distance_constraints, interactions, atom_offsets,
                                                 num_atoms, positions, gradient, num_elements, lane);
    }

    __device__ __forceinline__ float energy_only(float const *positions, std::uint32_t num_elements,
                                                 std::uint32_t lane) const {
        return evaluate_etk_torsion_energy<false>(torsion_terms, distance_constraints, interactions, atom_offsets,
                                                  num_atoms, positions, nullptr, num_elements, lane);
    }
};

} // namespace smallfield::cuda

#endif // __CUDACC__
