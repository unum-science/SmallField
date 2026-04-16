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
    float const bond_ij_x = positions[atom_j] - positions[atom_i];
    float const bond_ij_y = positions[num_atoms + atom_j] - positions[num_atoms + atom_i];
    float const bond_ij_z = positions[2 * num_atoms + atom_j] - positions[2 * num_atoms + atom_i];

    float const bond_jk_x = positions[atom_k] - positions[atom_j];
    float const bond_jk_y = positions[num_atoms + atom_k] - positions[num_atoms + atom_j];
    float const bond_jk_z = positions[2 * num_atoms + atom_k] - positions[2 * num_atoms + atom_j];

    float const bond_kl_x = positions[atom_l] - positions[atom_k];
    float const bond_kl_y = positions[num_atoms + atom_l] - positions[num_atoms + atom_k];
    float const bond_kl_z = positions[2 * num_atoms + atom_l] - positions[2 * num_atoms + atom_k];

    float const n1_x = bond_ij_y * bond_jk_z - bond_ij_z * bond_jk_y;
    float const n1_y = bond_ij_z * bond_jk_x - bond_ij_x * bond_jk_z;
    float const n1_z = bond_ij_x * bond_jk_y - bond_ij_y * bond_jk_x;

    float const n2_x = bond_jk_y * bond_kl_z - bond_jk_z * bond_kl_y;
    float const n2_y = bond_jk_z * bond_kl_x - bond_jk_x * bond_kl_z;
    float const n2_z = bond_jk_x * bond_kl_y - bond_jk_y * bond_kl_x;

    float const dot_n = n1_x * n2_x + n1_y * n2_y + n1_z * n2_z;
    float const cross_dot = (n1_y * n2_z - n1_z * n2_y) * bond_jk_x + (n1_z * n2_x - n1_x * n2_z) * bond_jk_y +
                            (n1_x * n2_y - n1_y * n2_x) * bond_jk_z;

    float const inverse_bond_jk_length = rsqrtf(
        fmaxf(fmaf(bond_jk_x, bond_jk_x, fmaf(bond_jk_y, bond_jk_y, bond_jk_z * bond_jk_z)), 1e-10f));

    return atan2f(cross_dot * inverse_bond_jk_length, dot_n);
}

/**
 *  @brief Compute torsion energy and per-atom gradient using the Wilson B-matrix approach.
 *
 *  Uses __sincosf + Chebyshev recurrence for the 6-term Fourier series.
 *  Accumulates gradient for the specified @p atom_role (first through fourth)
 *  into the provided @p grad_x / @p grad_y / @p grad_z accumulators.
 *
 *  @tparam write_gradient  When true, computes and writes gradient; when false, energy only.
 *  @param[in] positions SoA coordinate array [x0..xN, y0..yN, z0..zN].
 *  @param[in] num_atoms Number of atoms in the molecule.
 *  @param[in] torsion Torsion descriptor with atom indices and Fourier coefficients.
 *  @param[in] atom_role Which atom this thread is accumulating for (first through fourth).
 *  @param[out] grad_x Accumulated x-gradient for the atom.
 *  @param[out] grad_y Accumulated y-gradient for the atom.
 *  @param[out] grad_z Accumulated z-gradient for the atom.
 *  @return  Torsion energy contribution (kcal/mol).
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

    // bond vectors
    float const bond_vec_0_x = positions[atom_i] - positions[atom_j];
    float const bond_vec_0_y = positions[num_atoms + atom_i] - positions[num_atoms + atom_j];
    float const bond_vec_0_z = positions[2 * num_atoms + atom_i] - positions[2 * num_atoms + atom_j];

    float const bond_vec_1_x = positions[atom_k] - positions[atom_j];
    float const bond_vec_1_y = positions[num_atoms + atom_k] - positions[num_atoms + atom_j];
    float const bond_vec_1_z = positions[2 * num_atoms + atom_k] - positions[2 * num_atoms + atom_j];

    float const bond_vec_2_x = -bond_vec_1_x, bond_vec_2_y = -bond_vec_1_y, bond_vec_2_z = -bond_vec_1_z;

    float const bond_vec_3_x = positions[atom_l] - positions[atom_k];
    float const bond_vec_3_y = positions[num_atoms + atom_l] - positions[num_atoms + atom_k];
    float const bond_vec_3_z = positions[2 * num_atoms + atom_l] - positions[2 * num_atoms + atom_k];

    // normals
    float normal_0_x = bond_vec_0_y * bond_vec_1_z - bond_vec_0_z * bond_vec_1_y;
    float normal_0_y = bond_vec_0_z * bond_vec_1_x - bond_vec_0_x * bond_vec_1_z;
    float normal_0_z = bond_vec_0_x * bond_vec_1_y - bond_vec_0_y * bond_vec_1_x;
    float const inverse_normal_length_0 = rsqrtf(
        fmaxf(normal_0_x * normal_0_x + normal_0_y * normal_0_y + normal_0_z * normal_0_z, 1e-10f));
    normal_0_x *= inverse_normal_length_0;
    normal_0_y *= inverse_normal_length_0;
    normal_0_z *= inverse_normal_length_0;

    float normal_1_x = bond_vec_2_y * bond_vec_3_z - bond_vec_2_z * bond_vec_3_y;
    float normal_1_y = bond_vec_2_z * bond_vec_3_x - bond_vec_2_x * bond_vec_3_z;
    float normal_1_z = bond_vec_2_x * bond_vec_3_y - bond_vec_2_y * bond_vec_3_x;
    float const inverse_normal_length_1 = rsqrtf(
        fmaxf(normal_1_x * normal_1_x + normal_1_y * normal_1_y + normal_1_z * normal_1_z, 1e-10f));
    normal_1_x *= inverse_normal_length_1;
    normal_1_y *= inverse_normal_length_1;
    normal_1_z *= inverse_normal_length_1;

    float const cos_phi = fmaxf(
        -1.0f, fminf(1.0f, normal_0_x * normal_1_x + normal_0_y * normal_1_y + normal_0_z * normal_1_z));
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

    float const cosine_gradient[6] = {
        inverse_normal_length_0 * (normal_1_x - cos_phi * normal_0_x),
        inverse_normal_length_0 * (normal_1_y - cos_phi * normal_0_y),
        inverse_normal_length_0 * (normal_1_z - cos_phi * normal_0_z),
        inverse_normal_length_1 * (normal_0_x - cos_phi * normal_1_x),
        inverse_normal_length_1 * (normal_0_y - cos_phi * normal_1_y),
        inverse_normal_length_1 * (normal_0_z - cos_phi * normal_1_z),
    };

    // Wilson B-matrix chain rule for the atom at atom_role
    float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;
    switch (atom_role) {
    case atom_role_first_k:
        force_x = sin_term * (cosine_gradient[2] * bond_vec_1_y - cosine_gradient[1] * bond_vec_1_z);
        force_y = sin_term * (cosine_gradient[0] * bond_vec_1_z - cosine_gradient[2] * bond_vec_1_x);
        force_z = sin_term * (cosine_gradient[1] * bond_vec_1_x - cosine_gradient[0] * bond_vec_1_y);
        break;
    case atom_role_second_k:
        force_x = sin_term * (cosine_gradient[1] * (bond_vec_1_z - bond_vec_0_z) +
                              cosine_gradient[2] * (bond_vec_0_y - bond_vec_1_y) +
                              cosine_gradient[4] * (-bond_vec_3_z) + cosine_gradient[5] * bond_vec_3_y);
        force_y = sin_term * (cosine_gradient[0] * (bond_vec_0_z - bond_vec_1_z) +
                              cosine_gradient[2] * (bond_vec_1_x - bond_vec_0_x) + cosine_gradient[3] * bond_vec_3_z +
                              cosine_gradient[5] * (-bond_vec_3_x));
        force_z = sin_term * (cosine_gradient[0] * (bond_vec_1_y - bond_vec_0_y) +
                              cosine_gradient[1] * (bond_vec_0_x - bond_vec_1_x) +
                              cosine_gradient[3] * (-bond_vec_3_y) + cosine_gradient[4] * bond_vec_3_x);
        break;
    case atom_role_third_k:
        force_x = sin_term * (cosine_gradient[1] * bond_vec_0_z + cosine_gradient[2] * (-bond_vec_0_y) +
                              cosine_gradient[4] * (bond_vec_3_z - bond_vec_2_z) +
                              cosine_gradient[5] * (bond_vec_2_y - bond_vec_3_y));
        force_y = sin_term * (cosine_gradient[0] * (-bond_vec_0_z) + cosine_gradient[2] * bond_vec_0_x +
                              cosine_gradient[3] * (bond_vec_2_z - bond_vec_3_z) +
                              cosine_gradient[5] * (bond_vec_3_x - bond_vec_2_x));
        force_z = sin_term * (cosine_gradient[0] * bond_vec_0_y + cosine_gradient[1] * (-bond_vec_0_x) +
                              cosine_gradient[3] * (bond_vec_3_y - bond_vec_2_y) +
                              cosine_gradient[4] * (bond_vec_2_x - bond_vec_3_x));
        break;
    case atom_role_fourth_k:
        force_x = sin_term * (cosine_gradient[4] * bond_vec_2_z - cosine_gradient[5] * bond_vec_2_y);
        force_y = sin_term * (cosine_gradient[5] * bond_vec_2_x - cosine_gradient[3] * bond_vec_2_z);
        force_z = sin_term * (cosine_gradient[3] * bond_vec_2_y - cosine_gradient[4] * bond_vec_2_x);
        break;
    }
    *grad_x += force_x;
    *grad_y += force_y;
    *grad_z += force_z;
    return energy;
}

/**
 *  @brief Atom-centric ETK torsion energy and gradient evaluation.
 *
 *  Each thread owns atoms, iterates all torsions checking participation.
 *  No atomicAdd.
 *
 *  @tparam write_gradient  When true, computes and writes gradient; when false, energy only.
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

                float const delta_x = positions[atom_j] - positions[atom_i];
                float const delta_y = positions[num_atoms + atom_j] - positions[num_atoms + atom_i];
                float const delta_z = positions[2 * num_atoms + atom_j] - positions[2 * num_atoms + atom_i];
                float const distance_squared = fmaf(delta_x, delta_x, fmaf(delta_y, delta_y, delta_z * delta_z));
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
                        gradient_x += direction * scale * delta_x;
                        gradient_y += direction * scale * delta_y;
                        gradient_z += direction * scale * delta_z;
                    }
                }
                else if (distance < lower_bound) {
                    if (entry.role == atom_role_first_k)
                        total_energy += weight * (lower_bound - distance) * (lower_bound - distance);
                    if constexpr (write_gradient) {
                        float const scale = -2.0f * weight * (lower_bound - distance) * inverse_distance;
                        gradient_x += direction * scale * delta_x;
                        gradient_y += direction * scale * delta_y;
                        gradient_z += direction * scale * delta_z;
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
