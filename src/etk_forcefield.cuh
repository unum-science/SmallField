/**
 * @file etk_forcefield.cuh
 * @brief Atom-centric ETK torsion energy and gradient for warp-level L-BFGS.
 *
 * Implements CSD-derived torsion Fourier potentials for ETKDG stage 7:
 *   E_torsion    = Σₖ Σₙ₌₁⁶ Vₙ·cos(n·φₖ)
 *   E_constraint = Σ_c w_c·max(0, d_c − u_c)² + w_c·max(0, l_c − d_c)²
 *
 * Architecture: atom-centric with __sincosf + Chebyshev recurrence.
 * Each thread owns ceil(N/32) atoms, iterates all torsions/constraints,
 * accumulates gradient in registers. No atomicAdd.
 *
 * Positions layout (SoA): [x₀..xₙ, y₀..yₙ, z₀..zₙ] (3D only).
 */

#pragma once

#if defined(__CUDACC__)

#include "etkdg_types.cuh"
#include "warp_primitives.cuh"

namespace smallfield {

/** Compute dihedral angle φ for atoms i-j-k-l. Returns radians in [-π, π]. */
__device__ __forceinline__ float compute_dihedral_angle_(float const* positions, std::uint32_t num_atoms,
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

    float const inv_jk_len =
        rsqrtf(fmaxf(fmaf(bond_jk_x, bond_jk_x, fmaf(bond_jk_y, bond_jk_y, bond_jk_z * bond_jk_z)), 1e-10f));

    return atan2f(cross_dot * inv_jk_len, dot_n);
}

/**
 * Compute torsion energy and per-atom gradient using the Wilson B-matrix approach.
 * Uses __sincosf + Chebyshev recurrence for the 6-term Fourier series.
 *
 * Accumulates gradient for the specified atom_role (0=i, 1=j, 2=k, 3=l)
 * into the provided grad_x/y/z accumulators.
 */
__device__ __forceinline__ float compute_torsion_contribution_(float const* positions, std::uint32_t num_atoms,
                                                               etk_torsion_descriptor_t const& torsion,
                                                               std::uint32_t atom_role, float* grad_x, float* grad_y,
                                                               float* grad_z, bool write_gradient) {
    std::uint8_t const ai = torsion.atom_indices[0];
    std::uint8_t const aj = torsion.atom_indices[1];
    std::uint8_t const ak = torsion.atom_indices[2];
    std::uint8_t const al = torsion.atom_indices[3];

    // bond vectors
    float const r0x = positions[ai] - positions[aj];
    float const r0y = positions[num_atoms + ai] - positions[num_atoms + aj];
    float const r0z = positions[2 * num_atoms + ai] - positions[2 * num_atoms + aj];

    float const r1x = positions[ak] - positions[aj];
    float const r1y = positions[num_atoms + ak] - positions[num_atoms + aj];
    float const r1z = positions[2 * num_atoms + ak] - positions[2 * num_atoms + aj];

    float const r2x = -r1x, r2y = -r1y, r2z = -r1z;

    float const r3x = positions[al] - positions[ak];
    float const r3y = positions[num_atoms + al] - positions[num_atoms + ak];
    float const r3z = positions[2 * num_atoms + al] - positions[2 * num_atoms + ak];

    // normals
    float t0x = r0y * r1z - r0z * r1y;
    float t0y = r0z * r1x - r0x * r1z;
    float t0z = r0x * r1y - r0y * r1x;
    float const inv_d0 = rsqrtf(fmaxf(t0x * t0x + t0y * t0y + t0z * t0z, 1e-10f));
    t0x *= inv_d0;
    t0y *= inv_d0;
    t0z *= inv_d0;

    float t1x = r2y * r3z - r2z * r3y;
    float t1y = r2z * r3x - r2x * r3z;
    float t1z = r2x * r3y - r2y * r3x;
    float const inv_d1 = rsqrtf(fmaxf(t1x * t1x + t1y * t1y + t1z * t1z, 1e-10f));
    t1x *= inv_d1;
    t1y *= inv_d1;
    t1z *= inv_d1;

    float const cos_phi = fmaxf(-1.0f, fminf(1.0f, t0x * t1x + t0y * t1y + t0z * t1z));
    float const sin_phi_sq = 1.0f - cos_phi * cos_phi;
    float const sin_phi = (sin_phi_sq > 0.0f) ? sqrtf(sin_phi_sq) : 0.0f;

    // Chebyshev recurrence for 6-term Fourier series
    float energy = 0.0f;
    float dE_dphi = 0.0f;
    float d2E_dphi2 = 0.0f;
    float sin_prev = 0.0f, cos_prev = 1.0f;
    float sin_curr = sin_phi, cos_curr = cos_phi;

    for (std::uint32_t n = 1; n <= 6; ++n) {
        float const coeff = torsion.fourier_coefficients[n - 1];
        if (coeff != 0.0f) {
            float const nf = static_cast<float>(n);
            energy += coeff * cos_curr;
            dE_dphi -= nf * coeff * sin_curr;
            d2E_dphi2 -= nf * nf * coeff * cos_curr;
        }
        // Chebyshev: sin((n+1)φ) = 2cos(φ)sin(nφ) − sin((n−1)φ)
        float const sin_next = 2.0f * cos_phi * sin_curr - sin_prev;
        float const cos_next = 2.0f * cos_phi * cos_curr - cos_prev;
        sin_prev = sin_curr;
        cos_prev = cos_curr;
        sin_curr = sin_next;
        cos_curr = cos_next;
    }

    if (!write_gradient)
        return energy;

    // sin_term = -dE_dphi / sin(φ), needed for the chain rule ∂E/∂r = sin_term · ∂cos(φ)/∂r
    // at φ = 0 or π, sin(φ) → 0 but dE_dphi → 0 too (sin(nφ) = 0 for integer n).
    // L'Hopital: lim[sin→0] dE_dphi/sin(φ) = d²E/dφ² / cos(φ)
    float const sin_term =
        (fabsf(sin_phi) > 1e-6f) ? (-dE_dphi / sin_phi) : (-d2E_dphi2 / fmaxf(fabsf(cos_phi), 1e-10f));

    float const dCos_dT[6] = {
        inv_d0 * (t1x - cos_phi * t0x), inv_d0 * (t1y - cos_phi * t0y), inv_d0 * (t1z - cos_phi * t0z),
        inv_d1 * (t0x - cos_phi * t1x), inv_d1 * (t0y - cos_phi * t1y), inv_d1 * (t0z - cos_phi * t1z),
    };

    // Wilson B-matrix chain rule for the atom at atom_role
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    switch (atom_role) {
    case 0: // atom i
        fx = sin_term * (dCos_dT[2] * r1y - dCos_dT[1] * r1z);
        fy = sin_term * (dCos_dT[0] * r1z - dCos_dT[2] * r1x);
        fz = sin_term * (dCos_dT[1] * r1x - dCos_dT[0] * r1y);
        break;
    case 1: // atom j
        fx = sin_term * (dCos_dT[1] * (r1z - r0z) + dCos_dT[2] * (r0y - r1y) + dCos_dT[4] * (-r3z) + dCos_dT[5] * r3y);
        fy = sin_term * (dCos_dT[0] * (r0z - r1z) + dCos_dT[2] * (r1x - r0x) + dCos_dT[3] * r3z + dCos_dT[5] * (-r3x));
        fz = sin_term * (dCos_dT[0] * (r1y - r0y) + dCos_dT[1] * (r0x - r1x) + dCos_dT[3] * (-r3y) + dCos_dT[4] * r3x);
        break;
    case 2: // atom k
        fx = sin_term * (dCos_dT[1] * r0z + dCos_dT[2] * (-r0y) + dCos_dT[4] * (r3z - r2z) + dCos_dT[5] * (r2y - r3y));
        fy = sin_term * (dCos_dT[0] * (-r0z) + dCos_dT[2] * r0x + dCos_dT[3] * (r2z - r3z) + dCos_dT[5] * (r3x - r2x));
        fz = sin_term * (dCos_dT[0] * r0y + dCos_dT[1] * (-r0x) + dCos_dT[3] * (r3y - r2y) + dCos_dT[4] * (r2x - r3x));
        break;
    case 3: // atom l
        fx = sin_term * (dCos_dT[4] * r2z - dCos_dT[5] * r2y);
        fy = sin_term * (dCos_dT[5] * r2x - dCos_dT[3] * r2z);
        fz = sin_term * (dCos_dT[3] * r2y - dCos_dT[4] * r2x);
        break;
    }
    *grad_x += fx;
    *grad_y += fy;
    *grad_z += fz;
    return energy;
}

/**
 * Atom-centric ETK torsion energy and gradient functor.
 * Each thread owns atoms, iterates all torsions checking participation.
 * No atomicAdd. Compatible with fire_minimize_.
 */
struct etk_torsion_energy_functor_t {
    etk_torsion_descriptor_t const* __restrict__ torsion_terms;
    distance_constraint_t const* __restrict__ distance_constraints;
    etk_atom_interaction_t const* __restrict__ interactions;
    etk_atom_offsets_t const* __restrict__ atom_offsets;
    std::uint32_t num_atoms;

    __device__ __forceinline__ float operator()(float const* positions, float* gradient, std::uint32_t num_elements,
                                                std::uint32_t lane) const {
        return evaluate_(positions, gradient, num_elements, lane, true);
    }

    __device__ __forceinline__ float energy_only(float const* positions, std::uint32_t num_elements,
                                                 std::uint32_t lane) const {
        return evaluate_(positions, nullptr, num_elements, lane, false);
    }

  private:
    __device__ __forceinline__ float evaluate_(float const* positions, float* gradient, std::uint32_t num_elements,
                                               std::uint32_t lane, bool write_gradient) const {

        if (write_gradient) {
            for (std::uint32_t i = lane; i < num_elements; i += warp_size)
                gradient[i] = 0.0f;
            __syncwarp(full_mask);
        }

        float total_energy = 0.0f;

        for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size) {
            float gradient_x = 0.0f, gradient_y = 0.0f, gradient_z = 0.0f;

            std::uint32_t const interaction_start = atom_offsets[atom].interaction_start;
            std::uint32_t const interaction_count = atom_offsets[atom].interaction_count;

            for (std::uint32_t index = interaction_start; index < interaction_start + interaction_count; ++index) {
                auto const& entry = interactions[index];

                if (entry.interaction_type == 0) {
                    // torsion term
                    float torsion_energy =
                        compute_torsion_contribution_(positions, num_atoms, torsion_terms[entry.term_index], entry.role,
                                                      &gradient_x, &gradient_y, &gradient_z, write_gradient);
                    if (entry.role == 0)
                        total_energy += torsion_energy;
                } else {
                    // distance constraint
                    auto const& constraint = distance_constraints[entry.term_index];
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
                    float const direction = (entry.role == 0) ? -1.0f : 1.0f;

                    if (distance > upper_bound) {
                        if (entry.role == 0)
                            total_energy += weight * (distance - upper_bound) * (distance - upper_bound);
                        if (write_gradient) {
                            float const scale = 2.0f * weight * (distance - upper_bound) * inverse_distance;
                            gradient_x += direction * scale * delta_x;
                            gradient_y += direction * scale * delta_y;
                            gradient_z += direction * scale * delta_z;
                        }
                    } else if (distance < lower_bound) {
                        if (entry.role == 0)
                            total_energy += weight * (lower_bound - distance) * (lower_bound - distance);
                        if (write_gradient) {
                            float const scale = -2.0f * weight * (lower_bound - distance) * inverse_distance;
                            gradient_x += direction * scale * delta_x;
                            gradient_y += direction * scale * delta_y;
                            gradient_z += direction * scale * delta_z;
                        }
                    }
                }
            }

            if (write_gradient) {
                gradient[atom] = gradient_x;
                gradient[num_atoms + atom] = gradient_y;
                gradient[2 * num_atoms + atom] = gradient_z;
            }
        }

        return warp_reduce_sum_f32_(total_energy);
    }
};

} // namespace smallfield

#endif // __CUDACC__
