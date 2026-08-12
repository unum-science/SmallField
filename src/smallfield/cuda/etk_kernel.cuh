/**
 *  @file src/smallfield/cuda/etk_kernel.cuh
 *  @brief Atom-centric ETK torsion energy and gradient for warp-level L-BFGS.
 *
 *  Implements CSD-derived torsion Fourier potentials for ETKDG stage 7.
 *
 *  @verbatim
 *  E_torsion    = Σ_k Σ_n V_n * cos(n * φ_k), for n = 1..6
 *  E_constraint = Σ_c w_c * max(0, d_c - u_c)^2 + w_c * max(0, l_c - d_c)^2
 *  @endverbatim
 *
 *  Architecture: atom-centric with `__sincosf` + Chebyshev recurrence. Each thread owns ceil(N/32)
 *  atoms, iterates all torsions/constraints, accumulates gradient in registers. No atomicAdd.
 *
 *  Positions layout (SoA): [x₀..xₙ, y₀..yₙ, z₀..zₙ] (3D only).
 */

#pragma once

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

/**
 *  @brief Compute torsion energy and per-atom gradient using the Wilson B-matrix approach.
 *
 *  Uses `__sincosf` + Chebyshev recurrence for the 6-term Fourier series. Accumulates gradient for the
 *  specified @p atom_role (first through fourth) into the provided @p grad_x / @p grad_y / @p grad_z
 *  accumulators.
 *
 *  @tparam write_gradient_ When true, computes and writes gradient; when false, energy only.
 *  @param[in] positions SoA coordinate array [x0..xN, y0..yN, z0..zN].
 *  @param[in] num_atoms Number of atoms in the molecule.
 *  @param[in] torsion Torsion descriptor with atom indices and Fourier coefficients.
 *  @param[in] atom_role Which atom this thread is accumulating for (first through fourth).
 *  @param[inout] gradient Accumulated gradient for the atom.
 *  @return Torsion energy contribution (kcal/mol).
 */
template <bool write_gradient_>
__device__ __forceinline__ float compute_torsion_contribution_(float const *positions, std::uint32_t num_atoms,
                                                               etk_torsion_descriptor_t const &torsion,
                                                               std::uint32_t atom_role, float3 *gradient) {
    std::uint8_t const atom_i = torsion.atom_indices[0];
    std::uint8_t const atom_j = torsion.atom_indices[1];
    std::uint8_t const atom_k = torsion.atom_indices[2];
    std::uint8_t const atom_l = torsion.atom_indices[3];

    // Bond vectors of the i-j-k-l chain. The second half-plane is spanned by (j-k) and (l-k),
    // and (j-k) is the negation of `bond_vec_1`, so these three cover both planes.
    float3 const bond_vec_0 = sub_float3_(load_position_(positions, num_atoms, atom_i),
                                          load_position_(positions, num_atoms, atom_j));
    float3 const bond_vec_1 = sub_float3_(load_position_(positions, num_atoms, atom_k),
                                          load_position_(positions, num_atoms, atom_j));
    float3 const bond_vec_3 = sub_float3_(load_position_(positions, num_atoms, atom_l),
                                          load_position_(positions, num_atoms, atom_k));

    float3 normal_0 = cross_float3_(bond_vec_0, bond_vec_1);
    float const inverse_normal_length_0 = rsqrtf(fmaxf(dot_float3_(normal_0, normal_0), 1e-10f));
    normal_0 = scale_float3_(normal_0, inverse_normal_length_0);

    // cross(-bond_vec_1, bond_vec_3) written with the negation absorbed by swapping operands.
    float3 normal_1 = cross_float3_(bond_vec_3, bond_vec_1);
    float const inverse_normal_length_1 = rsqrtf(fmaxf(dot_float3_(normal_1, normal_1), 1e-10f));
    normal_1 = scale_float3_(normal_1, inverse_normal_length_1);

    float const cos_phi = fmaxf(-1.0f, fminf(1.0f, dot_float3_(normal_0, normal_1)));
    float const sin_phi_sq = 1.0f - cos_phi * cos_phi;
    float const sin_phi = (sin_phi_sq > 0.0f) ? sqrtf(sin_phi_sq) : 0.0f;

    // Direction of this atom's torsion force, which the Fourier coefficients only rescale.
    // The four roles' forces still sum to zero by Newton's 3rd law.
    float3 force_direction {0.0f, 0.0f, 0.0f};
    if constexpr (write_gradient_) {
        float3 lever_0 {0.0f, 0.0f, 0.0f};
        float3 lever_1 {0.0f, 0.0f, 0.0f};
        switch (atom_role) {
        case atom_role_first_k: lever_0 = scale_float3_(bond_vec_1, -inverse_normal_length_0); break;
        case atom_role_second_k:
            lever_0 = scale_float3_(sub_float3_(bond_vec_1, bond_vec_0), inverse_normal_length_0);
            lever_1 = scale_float3_(bond_vec_3, -inverse_normal_length_1);
            break;
        case atom_role_third_k:
            lever_0 = scale_float3_(bond_vec_0, inverse_normal_length_0);
            lever_1 = scale_float3_(add_float3_(bond_vec_3, bond_vec_1), inverse_normal_length_1);
            break;
        case atom_role_fourth_k: lever_1 = scale_float3_(bond_vec_1, -inverse_normal_length_1); break;
        }
        force_direction = add_float3_(cross_float3_(normal_0, sub_float3_(lever_1, scale_float3_(lever_0, cos_phi))),
                                      cross_float3_(normal_1, sub_float3_(lever_0, scale_float3_(lever_1, cos_phi))));
    }

    // Chebyshev recurrence for 6-term Fourier series
    float energy = 0.0f;
    float energy_gradient_dihedral = 0.0f;
    float energy_second_gradient_dihedral = 0.0f;
    float sin_prev = 0.0f, cos_prev = 1.0f;
    float sin_curr = sin_phi, cos_curr = cos_phi;

    for (std::uint32_t harmonic = 1; harmonic <= 6; ++harmonic) {
        float const coeff = torsion.fourier_coefficients[harmonic - 1];
        if (coeff != 0.0f) {
            float const harmonic_as_float = static_cast<float>(harmonic);
            energy += coeff * cos_curr;
            energy_gradient_dihedral -= harmonic_as_float * coeff * sin_curr;
            energy_second_gradient_dihedral -= harmonic_as_float * harmonic_as_float * coeff * cos_curr;
        }
        // Chebyshev step from harmonic n to n+1: sin((n+1)φ) = 2cos(φ)sin(nφ) - sin((n-1)φ)
        float const sin_next = 2.0f * cos_phi * sin_curr - sin_prev;
        float const cos_next = 2.0f * cos_phi * cos_curr - cos_prev;
        sin_prev = sin_curr;
        cos_prev = cos_curr;
        sin_curr = sin_next;
        cos_curr = cos_next;
    }

    if constexpr (!write_gradient_) return energy;

    // sin_term = -energy_gradient_dihedral / sin(φ), needed for the chain rule ∂E/∂r = sin_term * ∂cos(φ)/∂r
    // at φ = 0 or pi, sin(φ) -> 0 but energy_gradient_dihedral -> 0 too (sin(nφ) = 0 for integer n).
    // L'Hopital: lim[sin->0] energy_gradient_dihedral/sin(φ) = ∂^2E/∂φ^2 / cos(φ)
    float const sin_term = (fabsf(sin_phi) > torsion_planarity_threshold_k)
                               ? (-energy_gradient_dihedral / sin_phi)
                               : (-energy_second_gradient_dihedral / copysignf(fmaxf(fabsf(cos_phi), 1e-10f), cos_phi));

    accumulate_gradient_(gradient, scale_float3_(force_direction, sin_term));
    return energy;
}

/**
 *  @brief Atom-centric ETK torsion energy and gradient evaluation.
 *
 *  Each thread owns atoms, iterates all torsions checking participation.
 *  No atomicAdd.
 *
 *  @tparam write_gradient_ When true, computes and writes gradient; when false, energy only.
 *  @param[in] torsion_terms Array of torsion descriptors.
 *  @param[in] distance_constraints Array of distance constraint descriptors.
 *  @param[in] interactions Per-atom interaction list entries.
 *  @param[in] atom_offsets Per-atom offsets into the interaction list.
 *  @param[in] num_atoms Number of atoms in the molecule.
 *  @param[in] positions SoA coordinate array [x0..xN, y0..yN, z0..zN].
 *  @param[out] gradient Output gradient array (same layout as positions), unused when !write_gradient_.
 *  @param[in] num_elements Total number of coordinate elements.
 *  @param[in] lane Warp lane index (0-31).
 *  @return Total energy (same value in all lanes after warp reduction).
 *  @see fire_minimize_
 */
template <bool write_gradient_>
__device__ __forceinline__ float evaluate_etk_torsion_energy(
    etk_torsion_descriptor_t const *__restrict__ torsion_terms,
    distance_constraint_t const *__restrict__ distance_constraints,
    etk_atom_interaction_t const *__restrict__ interactions, etk_atom_offsets_t const *__restrict__ atom_offsets,
    std::uint32_t num_atoms, float const *positions, float *gradient, std::uint32_t num_elements, std::uint32_t lane) {

    if constexpr (write_gradient_) {
        for (std::uint32_t element = lane; element < num_elements; element += warp_size_k) gradient[element] = 0.0f;
        __syncwarp(full_mask_k);
    }

    float total_energy = 0.0f;

    for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k) {
        float3 gradient_acc {0.0f, 0.0f, 0.0f};

        std::uint32_t const interaction_offset = atom_offsets[atom].interaction_offset;
        std::uint32_t const interaction_count = atom_offsets[atom].interaction_count;

        for (std::uint32_t index = interaction_offset; index < interaction_offset + interaction_count; ++index) {
            auto const &entry = interactions[index];

            if (entry.interaction_type == etk_torsion_k) {
                // torsion term
                float torsion_energy = compute_torsion_contribution_<write_gradient_>(
                    positions, num_atoms, torsion_terms[entry.term_index], entry.role, &gradient_acc);
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
                    if constexpr (write_gradient_) {
                        float const scale = 2.0f * weight * (distance - upper_bound) * inverse_distance;
                        accumulate_gradient_(&gradient_acc, scale_float3_(delta, direction * scale));
                    }
                }
                else if (distance < lower_bound) {
                    if (entry.role == atom_role_first_k)
                        total_energy += weight * (lower_bound - distance) * (lower_bound - distance);
                    if constexpr (write_gradient_) {
                        float const scale = -2.0f * weight * (lower_bound - distance) * inverse_distance;
                        accumulate_gradient_(&gradient_acc, scale_float3_(delta, direction * scale));
                    }
                }
            }
        }

        if constexpr (write_gradient_) store_gradient_(gradient, num_atoms, atom, gradient_acc);
    }

    return warp_reduce_sum_f32_(total_energy);
}

/**
 *  @brief Atom-centric ETK torsion energy and gradient functor.
 *
 *  Thin wrapper around `evaluate_etk_torsion_energy` that provides the `operator()` interface
 *  expected by `fire_minimize_`.
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
};

} // namespace smallfield::cuda

#endif // __CUDACC__
