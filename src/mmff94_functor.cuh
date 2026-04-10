/**
 * @file mmff94_functor.cuh
 * @brief MMFF94 energy and gradient functor for ETKDG pipeline stage 9.
 *
 * Uses per-atom interaction lists for zero-waste iteration.
 * Full analytical gradients for VdW, electrostatic, bond, angle, and torsion terms.
 * Compatible with fire_minimize_ (same functor interface).
 *
 * Positions in SoA float: [x₀..xₙ, y₀..yₙ, z₀..zₙ].
 */

#pragma once

#if defined(__CUDACC__)

#include "etkdg_types.cuh"
#include "fire_warp.cuh"

namespace smallfield {

// MMFF94 constants
constexpr float mmff_coulomb = 332.0716f;
constexpr float mmff_elec_buf = 0.05f;
constexpr float mmff_1_4_scale = 0.75f;
constexpr float mmff_vdw_d = 0.07f; // delta
constexpr float mmff_vdw_g = 0.12f; // gamma
constexpr float mmff_bond_pf = 143.9325f;
constexpr float mmff_bond_cs = -2.0f;
constexpr float mmff_angle_pf = 0.043844f;
constexpr float mmff_angle_cb = -0.007f;
constexpr float mmff_tor_pf = 0.5f;
constexpr float mmff_deg2rad = 3.14159265f / 180.0f;
constexpr float mmff_rad2deg = 180.0f / 3.14159265f;

/** Compute nonbonded (VdW + electrostatic) energy and gradient contribution for one atom. */
__device__ __forceinline__ void mmff_nonbonded_contribution_(float const* positions, std::uint32_t num_atoms,
                                                             std::uint32_t atom, mmff94_nonbonded_pair_t const& pair,
                                                             std::uint8_t role, float* energy, float* gx, float* gy,
                                                             float* gz) {

    std::uint32_t const other = (role == 0) ? pair.atom_j : pair.atom_i;
    float const ax = positions[atom], ay = positions[num_atoms + atom], az = positions[2 * num_atoms + atom];
    float const bx = positions[other], by = positions[num_atoms + other], bz = positions[2 * num_atoms + other];
    float const dx = bx - ax, dy = by - ay, dz = bz - az;
    float const r = sqrtf(fmaxf(fmaf(dx, dx, fmaf(dy, dy, dz * dz)), 1e-8f));
    float const inv_r = __frcp_rn(r);

    // VdW: buffered 14-7
    float const rho = r * __frcp_rn(pair.r_star);
    float const rho7 = rho * rho * rho * rho * rho * rho * rho;
    float const rep_d = rho + mmff_vdw_d;
    float const rep_ratio = (1.0f + mmff_vdw_d) * __frcp_rn(rep_d);
    float const rep7 = rep_ratio * rep_ratio * rep_ratio * rep_ratio * rep_ratio * rep_ratio * rep_ratio;
    float const att = (1.0f + mmff_vdw_g) * __frcp_rn(rho7 + mmff_vdw_g) - 2.0f;
    float const e_vdw = pair.epsilon * rep7 * att;
    float const d_rep = -7.0f * rep7 * __frcp_rn(rep_d);
    float const d_att = -7.0f * (1.0f + mmff_vdw_g) * rho7 * __frcp_rn(rho * (rho7 + mmff_vdw_g) * (rho7 + mmff_vdw_g));
    float const dE_dr_vdw = pair.epsilon * (d_rep * att + rep7 * d_att) * __frcp_rn(pair.r_star);

    // electrostatic
    float const scale = (pair.is_1_4 != 0) ? mmff_1_4_scale : 1.0f;
    float const r_buf = r + mmff_elec_buf;
    float const e_elec = mmff_coulomb * pair.charge_product * __frcp_rn(r_buf) * scale;
    float const dE_dr_elec = -mmff_coulomb * pair.charge_product * __frcp_rn(r_buf * r_buf) * scale;

    // count energy only once per pair (role=0 is atom_i)
    if (role == 0)
        *energy += e_vdw + e_elec;

    float const dE_dr = dE_dr_vdw + dE_dr_elec;
    // gradient: ∂E/∂rᵢ = −dE/dr · dx/r (dx = other − atom)
    *gx -= dE_dr * dx * inv_r;
    *gy -= dE_dr * dy * inv_r;
    *gz -= dE_dr * dz * inv_r;
}

/** Compute bond stretching energy and gradient for one atom. */
__device__ __forceinline__ void mmff_bond_contribution_(float const* positions, std::uint32_t num_atoms,
                                                        std::uint32_t atom, mmff94_bond_t const& bond,
                                                        std::uint8_t role, float* energy, float* gx, float* gy,
                                                        float* gz) {

    std::uint32_t const other = (role == 0) ? bond.atom_j : bond.atom_i;
    float const ax = positions[atom], ay = positions[num_atoms + atom], az = positions[2 * num_atoms + atom];
    float const bx = positions[other], by = positions[num_atoms + other], bz = positions[2 * num_atoms + other];
    float const dx = bx - ax, dy = by - ay, dz = bz - az;
    float const r = sqrtf(fmaxf(fmaf(dx, dx, fmaf(dy, dy, dz * dz)), 1e-8f));
    float const inv_r = __frcp_rn(r);
    float const dr = r - bond.equilibrium_distance;
    float const cs = mmff_bond_cs;

    if (role == 0)
        *energy +=
            0.5f * mmff_bond_pf * bond.force_constant * dr * dr * (1.0f + cs * dr + (7.0f / 12.0f) * cs * cs * dr * dr);

    float const dE_dr =
        mmff_bond_pf * bond.force_constant * dr * (1.0f + 1.5f * cs * dr + (7.0f / 6.0f) * cs * cs * dr * dr);
    // bond gradient: same sign convention
    *gx -= dE_dr * dx * inv_r;
    *gy -= dE_dr * dy * inv_r;
    *gz -= dE_dr * dz * inv_r;
}

/** Compute angle bending energy and gradient for one atom. */
__device__ __forceinline__ void mmff_angle_contribution_(float const* positions, std::uint32_t num_atoms,
                                                         std::uint32_t atom, mmff94_angle_t const& angle,
                                                         std::uint8_t role, float* energy, float* gx, float* gy,
                                                         float* gz) {

    float const ix = positions[angle.atom_i], iy = positions[num_atoms + angle.atom_i],
                iz = positions[2 * num_atoms + angle.atom_i];
    float const jx = positions[angle.atom_j], jy = positions[num_atoms + angle.atom_j],
                jz = positions[2 * num_atoms + angle.atom_j];
    float const kx = positions[angle.atom_k], ky = positions[num_atoms + angle.atom_k],
                kz = positions[2 * num_atoms + angle.atom_k];

    // vectors from central atom j
    float const rij_x = ix - jx, rij_y = iy - jy, rij_z = iz - jz;
    float const rkj_x = kx - jx, rkj_y = ky - jy, rkj_z = kz - jz;
    float const rij_len = sqrtf(fmaxf(fmaf(rij_x, rij_x, fmaf(rij_y, rij_y, rij_z * rij_z)), 1e-8f));
    float const rkj_len = sqrtf(fmaxf(fmaf(rkj_x, rkj_x, fmaf(rkj_y, rkj_y, rkj_z * rkj_z)), 1e-8f));
    float const inv_rij = __frcp_rn(rij_len);
    float const inv_rkj = __frcp_rn(rkj_len);

    // unit vectors
    float const p12_x = rij_x * inv_rij, p12_y = rij_y * inv_rij, p12_z = rij_z * inv_rij;
    float const p32_x = rkj_x * inv_rkj, p32_y = rkj_y * inv_rkj, p32_z = rkj_z * inv_rkj;

    float const cos_theta = fmaxf(-1.0f, fminf(1.0f, p12_x * p32_x + p12_y * p32_y + p12_z * p32_z));
    float const sin_theta_sq = 1.0f - cos_theta * cos_theta;
    float const sin_theta = (sin_theta_sq > 1e-12f) ? sqrtf(sin_theta_sq) : 1e-6f;
    float const theta = acosf(cos_theta);
    float const dtheta_deg = (theta - angle.equilibrium_angle) * mmff_rad2deg;

    // energy (counted once, by role 0 = atom_i)
    if (role == 0)
        *energy +=
            0.5f * mmff_angle_pf * angle.force_constant * dtheta_deg * dtheta_deg * (1.0f + mmff_angle_cb * dtheta_deg);

    // dE/dθ matching CPU mmff94.hpp:755: dE_dtheta = rad2deg * c2 * ka * angle_term * (...)
    // the extra rad2deg converts the derivative to work with the -1/sinθ chain rule
    float const dE_dtheta =
        mmff_rad2deg * mmff_angle_pf * angle.force_constant * dtheta_deg * (1.0f + 1.5f * mmff_angle_cb * dtheta_deg);
    float const inv_sin = __frcp_rn(sin_theta);

    // ∂E/∂rᵢ = dE_dθ · ∂cosθ/∂rᵢ · (−1/sinθ)
    // ∂cosθ/∂rᵢ = (p̂₃₂ − cosθ·p̂₁₂) / |r₁₂|   (from CPU mmff94.hpp:759)
    float const factor = dE_dtheta * (-inv_sin);

    if (role == 0) { // atom_i: ∂cosθ/∂rᵢ = (p̂₃₂ − cosθ·p̂₁₂) / |r₁₂|
        *gx += factor * (p32_x - cos_theta * p12_x) * inv_rij;
        *gy += factor * (p32_y - cos_theta * p12_y) * inv_rij;
        *gz += factor * (p32_z - cos_theta * p12_z) * inv_rij;
    } else if (role == 2) { // atom_k: ∂cosθ/∂rₖ = (p̂₁₂ − cosθ·p̂₃₂) / |r₃₂|
        *gx += factor * (p12_x - cos_theta * p32_x) * inv_rkj;
        *gy += factor * (p12_y - cos_theta * p32_y) * inv_rkj;
        *gz += factor * (p12_z - cos_theta * p32_z) * inv_rkj;
    } else { // role == 1, atom_j (center): -(grad_i + grad_k)
        float const gi_x = factor * (p32_x - cos_theta * p12_x) * inv_rij;
        float const gi_y = factor * (p32_y - cos_theta * p12_y) * inv_rij;
        float const gi_z = factor * (p32_z - cos_theta * p12_z) * inv_rij;
        float const gk_x = factor * (p12_x - cos_theta * p32_x) * inv_rkj;
        float const gk_y = factor * (p12_y - cos_theta * p32_y) * inv_rkj;
        float const gk_z = factor * (p12_z - cos_theta * p32_z) * inv_rkj;
        *gx -= gi_x + gk_x;
        *gy -= gi_y + gk_y;
        *gz -= gi_z + gk_z;
    }
}

/** Compute torsion energy and gradient for one atom (Wilson B-matrix). */
__device__ __forceinline__ void mmff_torsion_contribution_(float const* positions, std::uint32_t num_atoms,
                                                           std::uint32_t atom, mmff94_torsion_t const& tor,
                                                           std::uint8_t role, float* energy, float* gx, float* gy,
                                                           float* gz) {

    float const* p = positions;
    std::uint32_t const n = num_atoms;

    // bond vectors
    float const r0x = p[tor.atom_i] - p[tor.atom_j];
    float const r0y = p[n + tor.atom_i] - p[n + tor.atom_j];
    float const r0z = p[2 * n + tor.atom_i] - p[2 * n + tor.atom_j];
    float const r1x = p[tor.atom_k] - p[tor.atom_j];
    float const r1y = p[n + tor.atom_k] - p[n + tor.atom_j];
    float const r1z = p[2 * n + tor.atom_k] - p[2 * n + tor.atom_j];
    float const r2x = -r1x, r2y = -r1y, r2z = -r1z;
    float const r3x = p[tor.atom_l] - p[tor.atom_k];
    float const r3y = p[n + tor.atom_l] - p[n + tor.atom_k];
    float const r3z = p[2 * n + tor.atom_l] - p[2 * n + tor.atom_k];

    // normals
    float t0x = r0y * r1z - r0z * r1y, t0y = r0z * r1x - r0x * r1z, t0z = r0x * r1y - r0y * r1x;
    float const d0 = sqrtf(fmaxf(t0x * t0x + t0y * t0y + t0z * t0z, 1e-10f));
    float const id0 = __frcp_rn(d0);
    t0x *= id0;
    t0y *= id0;
    t0z *= id0;

    float t1x = r2y * r3z - r2z * r3y, t1y = r2z * r3x - r2x * r3z, t1z = r2x * r3y - r2y * r3x;
    float const d1 = sqrtf(fmaxf(t1x * t1x + t1y * t1y + t1z * t1z, 1e-10f));
    float const id1 = __frcp_rn(d1);
    t1x *= id1;
    t1y *= id1;
    t1z *= id1;

    float const cos_phi = fmaxf(-1.0f, fminf(1.0f, t0x * t1x + t0y * t1y + t0z * t1z));

    // signed sin(φ) from the triple product (t0 × t1) · r1_hat
    // r1 = bond j→k (the central bond), same direction as bond_jk
    float const r1_len = sqrtf(fmaxf(r1x * r1x + r1y * r1y + r1z * r1z, 1e-10f));
    float const cross_n_x = t0y * t1z - t0z * t1y;
    float const cross_n_y = t0z * t1x - t0x * t1z;
    float const cross_n_z = t0x * t1y - t0y * t1x;
    float const sin_phi = (cross_n_x * r1x + cross_n_y * r1y + cross_n_z * r1z) * __frcp_rn(r1_len);

    // energy (uses cos_phi only, sign-independent)
    float const cos2 = 2.0f * cos_phi * cos_phi - 1.0f;
    float const cos3 = cos_phi * (4.0f * cos_phi * cos_phi - 3.0f);
    float const tor_energy =
        mmff_tor_pf * (tor.v1 * (1.0f + cos_phi) + tor.v2 * (1.0f - cos2) + tor.v3 * (1.0f + cos3));

    if (role == 0)
        *energy += tor_energy;

    // dE/dφ = 0.5 * (-V1·sin(φ) + 2V2·sin(2φ) - 3V3·sin(3φ)) using SIGNED sin
    float const sin2 = 2.0f * sin_phi * cos_phi;
    float const sin3 = sin_phi * (3.0f - 4.0f * sin_phi * sin_phi);
    float const dE_dphi = mmff_tor_pf * (-tor.v1 * sin_phi + 2.0f * tor.v2 * sin2 - 3.0f * tor.v3 * sin3);

    float const abs_sin = fabsf(sin_phi);
    if (fabsf(cos_phi) < 1e-8f && abs_sin < 1e-8f)
        return;
    float const sin_term = (abs_sin < 1e-8f) ? (-dE_dphi / cos_phi) : (-dE_dphi / sin_phi);

    float const dc[6] = {
        id0 * (t1x - cos_phi * t0x), id0 * (t1y - cos_phi * t0y), id0 * (t1z - cos_phi * t0z),
        id1 * (t0x - cos_phi * t1x), id1 * (t0y - cos_phi * t1y), id1 * (t0z - cos_phi * t1z),
    };

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    switch (role) {
    case 0:
        fx = sin_term * (dc[2] * r1y - dc[1] * r1z);
        fy = sin_term * (dc[0] * r1z - dc[2] * r1x);
        fz = sin_term * (dc[1] * r1x - dc[0] * r1y);
        break;
    case 1:
        fx = sin_term * (dc[1] * (r1z - r0z) + dc[2] * (r0y - r1y) + dc[4] * (-r3z) + dc[5] * r3y);
        fy = sin_term * (dc[0] * (r0z - r1z) + dc[2] * (r1x - r0x) + dc[3] * r3z + dc[5] * (-r3x));
        fz = sin_term * (dc[0] * (r1y - r0y) + dc[1] * (r0x - r1x) + dc[3] * (-r3y) + dc[4] * r3x);
        break;
    case 2:
        fx = sin_term * (dc[1] * r0z + dc[2] * (-r0y) + dc[4] * (r3z - r2z) + dc[5] * (r2y - r3y));
        fy = sin_term * (dc[0] * (-r0z) + dc[2] * r0x + dc[3] * (r2z - r3z) + dc[5] * (r3x - r2x));
        fz = sin_term * (dc[0] * r0y + dc[1] * (-r0x) + dc[3] * (r3y - r2y) + dc[4] * (r2x - r3x));
        break;
    case 3:
        fx = sin_term * (dc[4] * r2z - dc[5] * r2y);
        fy = sin_term * (dc[5] * r2x - dc[3] * r2z);
        fz = sin_term * (dc[3] * r2y - dc[4] * r2x);
        break;
    }
    *gx += fx;
    *gy += fy;
    *gz += fz;
}

/**
 * MMFF94 energy and gradient functor using per-atom interaction lists.
 * Zero-waste iteration: each atom only processes its own interactions.
 */
struct mmff94_energy_functor_t {
    mmff94_nonbonded_pair_t const* nonbonded_params;
    mmff94_bond_t const* bond_params;
    mmff94_angle_t const* angle_params;
    mmff94_torsion_t const* torsion_params;
    mmff94_atom_interaction_t const* interactions;
    mmff94_atom_offsets_t const* atom_offsets;
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
            for (std::uint32_t i = lane; i < num_elements; i += fire_warp_size)
                gradient[i] = 0.0f;
            __syncwarp(fire_full_mask);
        }

        float total_energy = 0.0f;

        for (std::uint32_t atom = lane; atom < num_atoms; atom += fire_warp_size) {
            float gx = 0.0f, gy = 0.0f, gz = 0.0f;

            std::uint32_t const start = atom_offsets[atom].interaction_start;
            std::uint32_t const count = atom_offsets[atom].interaction_count;

            for (std::uint32_t idx = start; idx < start + count; ++idx) {
                auto const& inter = interactions[idx];
                switch (inter.interaction_type) {
                case 0: // nonbonded
                    mmff_nonbonded_contribution_(positions, num_atoms, atom, nonbonded_params[inter.term_index],
                                                 inter.role, &total_energy, &gx, &gy, &gz);
                    break;
                case 1: // bond
                    mmff_bond_contribution_(positions, num_atoms, atom, bond_params[inter.term_index], inter.role,
                                            &total_energy, &gx, &gy, &gz);
                    break;
                case 2: // angle (end atom)
                case 3: // angle (center atom)
                    mmff_angle_contribution_(positions, num_atoms, atom, angle_params[inter.term_index], inter.role,
                                             &total_energy, &gx, &gy, &gz);
                    break;
                case 4: // torsion
                    mmff_torsion_contribution_(positions, num_atoms, atom, torsion_params[inter.term_index], inter.role,
                                               &total_energy, &gx, &gy, &gz);
                    break;
                }
            }

            if (write_gradient) {
                gradient[atom] = gx;
                gradient[num_atoms + atom] = gy;
                gradient[2 * num_atoms + atom] = gz;
            }
        }

        // remove center-of-mass force (zero mean gradient per dimension)
        // prevents rigid-body translation during FIRE minimization
        if (write_gradient && num_atoms > 1) {
            __syncwarp(fire_full_mask);
            float const inv_n = __frcp_rn(static_cast<float>(num_atoms));
            for (std::uint32_t d = 0; d < 3; ++d) {
                float mean_g = 0.0f;
                for (std::uint32_t i = lane; i < num_atoms; i += fire_warp_size)
                    mean_g += gradient[d * num_atoms + i];
                mean_g = fire_warp_reduce_sum_(mean_g) * inv_n;
                for (std::uint32_t i = lane; i < num_atoms; i += fire_warp_size)
                    gradient[d * num_atoms + i] -= mean_g;
            }
            __syncwarp(fire_full_mask);
        }

        return fire_warp_reduce_sum_(total_energy);
    }
};

} // namespace smallfield

#endif // __CUDACC__
