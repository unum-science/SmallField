/**
 * @file dg_forcefield.cuh
 * @brief Atom-centric distance geometry energy and gradient for warp-level L-BFGS.
 *
 * Implements the DG violation energy used in ETKDG stages 4 and 6:
 *   E_total = E_distance + E_chiral + E_4th_dim
 *   E_distance = Σᵢ<ⱼ [ w_upper·max(0, dᵢⱼ − uᵢⱼ)² + w_lower·max(0, lᵢⱼ − dᵢⱼ)² ]
 *   E_chiral   = Σ_c  w_chiral·V_c²   (when sign(V_c) ≠ target)
 *   E_4th_dim  = w₄·Σᵢ xᵢ₄²
 *
 * Architecture: atom-centric. Each thread owns ceil(N/32) atoms and accumulates
 * their gradients in registers. Coordinates are broadcast via __shfl_sync in
 * 32-atom tiles. No atomicAdd. No pair index scan.
 *
 * Bounds matrix is stored in half precision (__half) to halve shared memory usage,
 * enabling 2 blocks/SM for N=128 molecules.
 */

#pragma once

#if defined(__CUDACC__)

#include "etkdg_types.cuh"
#include "lbfgs_warp.cuh"
#include <cuda_fp16.h>

namespace smallfield {

/**
 * Compute the signed tetrahedral volume for one chiral center.
 * V = (r_b − r_a) · ((r_c − r_a) × (r_d − r_a))
 * Uses only xyz (ignores the 4th dimension).
 * Positions in SoA layout: positions[d*num_atoms + i] for dimension d.
 */
__device__ __forceinline__ float compute_signed_tetrahedral_volume_(float const* positions, std::uint32_t num_atoms,
                                                                    std::uint8_t atom_a, std::uint8_t atom_b,
                                                                    std::uint8_t atom_c, std::uint8_t atom_d) {
    float const ax = positions[atom_a];
    float const ay = positions[num_atoms + atom_a];
    float const az = positions[2 * num_atoms + atom_a];

    float const bx = positions[atom_b] - ax;
    float const by = positions[num_atoms + atom_b] - ay;
    float const bz = positions[2 * num_atoms + atom_b] - az;

    float const cx = positions[atom_c] - ax;
    float const cy = positions[num_atoms + atom_c] - ay;
    float const cz = positions[2 * num_atoms + atom_c] - az;

    float const dx = positions[atom_d] - ax;
    float const dy = positions[num_atoms + atom_d] - ay;
    float const dz = positions[2 * num_atoms + atom_d] - az;

    float const cross_x = cy * dz - cz * dy;
    float const cross_y = cz * dx - cx * dz;
    float const cross_z = cx * dy - cy * dx;

    return bx * cross_x + by * cross_y + bz * cross_z;
}

/**
 * Validate all chiral centers. Returns true iff every signed volume matches its target sign.
 * Uses __ballot_sync for warp-wide consensus.
 */
__device__ __forceinline__ bool validate_chirality_(float const* positions, std::uint32_t num_atoms,
                                                    chiral_center_descriptor_t const* chiral_centers,
                                                    std::uint32_t num_chiral_centers, std::uint32_t lane) {
    bool lane_ok = true;
    for (std::uint32_t c = lane; c < num_chiral_centers; c += warp_size) {
        auto const& center = chiral_centers[c];
        float const volume =
            compute_signed_tetrahedral_volume_(positions, num_atoms, center.center_atom, center.neighbor_atoms[0],
                                               center.neighbor_atoms[1], center.neighbor_atoms[2]);
        std::int8_t const actual_sign = (volume > 0.0f) ? 1 : -1;
        if (actual_sign != center.target_sign)
            lane_ok = false;
    }
    return __ballot_sync(full_mask, lane_ok) == full_mask;
}

/**
 * Atom-centric distance geometry energy and gradient functor.
 *
 * Each thread owns atoms {lane, lane+32, lane+64, ...} and accumulates their
 * gradients in registers. All-pairs distance violations are computed by iterating
 * other atoms in 32-atom tiles via __shfl_sync.
 *
 * Bounds matrix stored as __half in shared memory.
 * Positions and gradients in SoA float: [x₀..xₙ, y₀..yₙ, z₀..zₙ, (w₀..wₙ)]
 */
struct distance_geometry_energy_functor_t {
    __half const* bounds_matrix;
    chiral_center_descriptor_t const* chiral_centers;
    std::uint32_t num_atoms;
    std::uint32_t num_chiral_centers;
    std::uint32_t spatial_dimension; // 3 or 4
    float upper_violation_weight;
    float lower_violation_weight;
    float chiral_volume_weight;
    float fourth_dimension_penalty_weight;

    /**
     * Compute energy and gradient. Writes ∂E/∂xᵢ into gradient array.
     * Returns total energy (same value in all lanes after warp reduction).
     */
    __device__ __forceinline__ float operator()(float const* positions, float* gradient, std::uint32_t num_elements,
                                                std::uint32_t lane) const {
        return evaluate_(positions, gradient, num_elements, lane, true);
    }

    /**
     * Energy-only evaluation (skips gradient writes). Used by line search.
     */
    __device__ __forceinline__ float energy_only(float const* positions, std::uint32_t num_elements,
                                                 std::uint32_t lane) const {
        return evaluate_(positions, nullptr, num_elements, lane, false);
    }

  private:
    __device__ __forceinline__ float evaluate_(float const* positions, float* gradient, std::uint32_t num_elements,
                                               std::uint32_t lane, bool write_gradient) const {

        std::uint32_t const dim = spatial_dimension;
        std::uint32_t const num_tiles = (num_atoms + warp_size - 1) / warp_size;

        if (write_gradient) {
            for (std::uint32_t i = lane; i < num_elements; i += warp_size)
                gradient[i] = 0.0f;
            __syncwarp(full_mask);
        }

        float total_energy = 0.0f;

        // process atoms in 32-atom tiles
        // for each tile of "owned" atoms, iterate all other atoms via shuffle tiles
        for (std::uint32_t tile_a = 0; tile_a < num_tiles; ++tile_a) {
            std::uint32_t const atom = tile_a * warp_size + lane;
            bool const atom_valid = atom < num_atoms;

            // load this tile's coordinates into registers
            float coord_a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (atom_valid) {
                for (std::uint32_t d = 0; d < dim; ++d)
                    coord_a[d] = positions[d * num_atoms + atom];
            }

            // gradient accumulators for this atom (registers, no shared memory)
            float grad_a[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            // iterate all other atoms in tiles
            for (std::uint32_t tile_b = 0; tile_b < num_tiles; ++tile_b) {
                std::uint32_t const other_base = tile_b * warp_size;

                // load tile_b's coordinates into registers for shuffling
                float coord_b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                std::uint32_t const other_load = other_base + lane;
                if (other_load < num_atoms) {
                    for (std::uint32_t d = 0; d < dim; ++d)
                        coord_b[d] = positions[d * num_atoms + other_load];
                }

                // each lane broadcasts one atom from tile_b
                // all lanes MUST participate in __shfl_sync (full_mask),
                // so shuffles are outside the validity check
                for (std::uint32_t slot = 0; slot < warp_size; ++slot) {
                    std::uint32_t const other = other_base + slot;

                    // all lanes participate in shuffles
                    float const bx = __shfl_sync(full_mask, coord_b[0], slot);
                    float const by = __shfl_sync(full_mask, coord_b[1], slot);
                    float const bz = __shfl_sync(full_mask, coord_b[2], slot);
                    float const bw = (dim == 4) ? __shfl_sync(full_mask, coord_b[3], slot) : 0.0f;

                    // skip invalid pairs after shuffles
                    if (!atom_valid || other >= num_atoms || other == atom)
                        continue;

                    float const dx = bx - coord_a[0];
                    float const dy = by - coord_a[1];
                    float const dz = bz - coord_a[2];
                    float distance_squared = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
                    float dw = 0.0f;
                    if (dim == 4) {
                        dw = bw - coord_a[3];
                        distance_squared = fmaf(dw, dw, distance_squared);
                    }

                    float const distance = sqrtf(fmaxf(distance_squared, 1e-12f));
                    float const inverse_distance = __frcp_rn(distance);

                    // read bounds (half → float)
                    std::uint32_t const min_idx = (atom < other) ? atom : other;
                    std::uint32_t const max_idx = (atom < other) ? other : atom;
                    float const upper_bound = __half2float(bounds_matrix[min_idx * num_atoms + max_idx]);
                    float const lower_bound = __half2float(bounds_matrix[max_idx * num_atoms + min_idx]);

                    // upper bound violation
                    if (distance > upper_bound) {
                        float const violation = distance - upper_bound;
                        if (atom < other)
                            total_energy += upper_violation_weight * violation * violation;

                        if (write_gradient) {
                            float const scale = 2.0f * upper_violation_weight * violation * inverse_distance;
                            grad_a[0] -= scale * dx;
                            grad_a[1] -= scale * dy;
                            grad_a[2] -= scale * dz;
                            if (dim == 4)
                                grad_a[3] -= scale * dw;
                        }
                    }

                    // lower bound violation
                    if (distance < lower_bound) {
                        float const violation = lower_bound - distance;
                        if (atom < other)
                            total_energy += lower_violation_weight * violation * violation;

                        if (write_gradient) {
                            float const scale = -2.0f * lower_violation_weight * violation * inverse_distance;
                            grad_a[0] -= scale * dx;
                            grad_a[1] -= scale * dy;
                            grad_a[2] -= scale * dz;
                            if (dim == 4)
                                grad_a[3] -= scale * dw;
                        }
                    }
                }
            }

            // chiral volume contributions for this atom
            if (atom_valid) {
                for (std::uint32_t c = 0; c < num_chiral_centers; ++c) {
                    auto const& center = chiral_centers[c];
                    float const volume = compute_signed_tetrahedral_volume_(
                        positions, num_atoms, center.center_atom, center.neighbor_atoms[0], center.neighbor_atoms[1],
                        center.neighbor_atoms[2]);

                    float const target_sign = static_cast<float>(center.target_sign);
                    if (volume * target_sign >= 0.0f)
                        continue;

                    // only one thread per chiral center counts the energy
                    if (atom == center.center_atom)
                        total_energy += chiral_volume_weight * volume * volume;

                    if (write_gradient) {
                        // ∂V/∂r for this atom's role
                        std::uint8_t const aa = center.center_atom;
                        std::uint8_t const ab = center.neighbor_atoms[0];
                        std::uint8_t const ac = center.neighbor_atoms[1];
                        std::uint8_t const ad = center.neighbor_atoms[2];
                        float const scale = 2.0f * chiral_volume_weight * volume;

                        float const edge_bx = positions[ab] - positions[aa];
                        float const edge_by = positions[num_atoms + ab] - positions[num_atoms + aa];
                        float const edge_bz = positions[2 * num_atoms + ab] - positions[2 * num_atoms + aa];
                        float const edge_cx = positions[ac] - positions[aa];
                        float const edge_cy = positions[num_atoms + ac] - positions[num_atoms + aa];
                        float const edge_cz = positions[2 * num_atoms + ac] - positions[2 * num_atoms + aa];
                        float const edge_dx = positions[ad] - positions[aa];
                        float const edge_dy = positions[num_atoms + ad] - positions[num_atoms + aa];
                        float const edge_dz = positions[2 * num_atoms + ad] - positions[2 * num_atoms + aa];

                        // ∂V/∂r_b = c × d,  ∂V/∂r_c = d × b,  ∂V/∂r_d = b × c
                        if (atom == ab) {
                            grad_a[0] += scale * (edge_cy * edge_dz - edge_cz * edge_dy);
                            grad_a[1] += scale * (edge_cz * edge_dx - edge_cx * edge_dz);
                            grad_a[2] += scale * (edge_cx * edge_dy - edge_cy * edge_dx);
                        } else if (atom == ac) {
                            grad_a[0] += scale * (edge_dy * edge_bz - edge_dz * edge_by);
                            grad_a[1] += scale * (edge_dz * edge_bx - edge_dx * edge_bz);
                            grad_a[2] += scale * (edge_dx * edge_by - edge_dy * edge_bx);
                        } else if (atom == ad) {
                            grad_a[0] += scale * (edge_by * edge_cz - edge_bz * edge_cy);
                            grad_a[1] += scale * (edge_bz * edge_cx - edge_bx * edge_cz);
                            grad_a[2] += scale * (edge_bx * edge_cy - edge_by * edge_cx);
                        } else if (atom == aa) {
                            float const dvdb_x = edge_cy * edge_dz - edge_cz * edge_dy;
                            float const dvdb_y = edge_cz * edge_dx - edge_cx * edge_dz;
                            float const dvdb_z = edge_cx * edge_dy - edge_cy * edge_dx;
                            float const dvdc_x = edge_dy * edge_bz - edge_dz * edge_by;
                            float const dvdc_y = edge_dz * edge_bx - edge_dx * edge_bz;
                            float const dvdc_z = edge_dx * edge_by - edge_dy * edge_bx;
                            float const dvdd_x = edge_by * edge_cz - edge_bz * edge_cy;
                            float const dvdd_y = edge_bz * edge_cx - edge_bx * edge_cz;
                            float const dvdd_z = edge_bx * edge_cy - edge_by * edge_cx;
                            grad_a[0] -= scale * (dvdb_x + dvdc_x + dvdd_x);
                            grad_a[1] -= scale * (dvdb_y + dvdc_y + dvdd_y);
                            grad_a[2] -= scale * (dvdb_z + dvdc_z + dvdd_z);
                        }
                    }
                }

                // fourth-dimension crush penalty: w₄·xᵢ₄²
                if (fourth_dimension_penalty_weight > 0.0f && dim == 4) {
                    float const w_coord = coord_a[3];
                    total_energy += fourth_dimension_penalty_weight * w_coord * w_coord;
                    if (write_gradient)
                        grad_a[3] += 2.0f * fourth_dimension_penalty_weight * w_coord;
                }

                // write gradient for this atom (direct write, no atomicAdd)
                if (write_gradient) {
                    for (std::uint32_t d = 0; d < dim; ++d)
                        gradient[d * num_atoms + atom] = grad_a[d];
                }
            }
        }

        return warp_reduce_sum_f32_(total_energy);
    }
};

} // namespace smallfield

#endif // __CUDACC__
