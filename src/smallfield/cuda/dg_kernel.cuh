/**
 *  @file dg_kernel.cuh
 *  @brief Atom-centric distance geometry energy and gradient for warp-level L-BFGS.
 *
 *  Implements the DG violation energy used in ETKDG stages 4 and 6:
 *   E_total = E_distance + E_chiral + E_4th_dim
 *   E_distance = Σᵢ<ⱼ [ w_upper·max(0, dᵢⱼ − uᵢⱼ)² + w_lower·max(0, lᵢⱼ − dᵢⱼ)² ]
 *   E_chiral   = Σ_c  w_chiral·V_c²   (when sign(V_c) ≠ target)
 *   E_4th_dim  = w₄·Σᵢ xᵢ₄²
 *
 *  Architecture: atom-centric. Each thread owns ceil(N/32) atoms and accumulates
 *  their gradients in registers. Coordinates are broadcast via __shfl_sync in
 *  32-atom tiles. No atomicAdd. No pair index scan.
 *
 *  Bounds matrix is int8-quantized with a per-molecule (min, inv_scale) pair
 *  (floats broadcast in registers).  Storage per pair is 1 byte — half the
 *  __half layout — and at N=128 the shared footprint drops from 32 KB to 16 KB.
 *  Dequantize at read site with a single FMA: `bound = q · inv_scale + mol_min`.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>

#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/cuda/warp_primitives.cuh"

namespace smallfield::cuda {
using smallfield::types::chiral_center_descriptor_t;

/**
 *  @brief Compute the signed tetrahedral volume for one chiral center.
 *
 *  V = (r_b - r_a) . ((r_c - r_a) x (r_d - r_a)).
 *  Uses only xyz (ignores the 4th dimension).
 *  Positions in SoA layout: @p positions[d * num_atoms + i] for dimension d.
 *
 *  @param[in] positions SoA coordinate array.
 *  @param[in] num_atoms Number of atoms (stride between dimensions).
 *  @param[in] atom_a Center atom index.
 *  @param[in] atom_b First neighbor atom index.
 *  @param[in] atom_c Second neighbor atom index.
 *  @param[in] atom_d Third neighbor atom index.
 *  @return Signed tetrahedral volume.
 */
__device__ __forceinline__ float compute_signed_tetrahedral_volume_(float const *positions, std::uint32_t num_atoms,
                                                                    std::uint8_t atom_a, std::uint8_t atom_b,
                                                                    std::uint8_t atom_c, std::uint8_t atom_d) {
    float3 const position_a = load_position_(positions, num_atoms, atom_a);
    float3 const edge_b = sub_float3_(load_position_(positions, num_atoms, atom_b), position_a);
    float3 const edge_c = sub_float3_(load_position_(positions, num_atoms, atom_c), position_a);
    float3 const edge_d = sub_float3_(load_position_(positions, num_atoms, atom_d), position_a);
    return dot_float3_(edge_b, cross_float3_(edge_c, edge_d));
}

/**
 *  @brief Validate all chiral centers.
 *
 *  Returns true iff every signed volume matches its target sign.
 *  Uses __ballot_sync for warp-wide consensus.
 *
 *  @param[in] positions SoA coordinate array.
 *  @param[in] num_atoms Number of atoms.
 *  @param[in] chiral_centers Array of chiral center descriptors.
 *  @param[in] num_chiral_centers Number of chiral centers.
 *  @param[in] lane Warp lane index (0-31).
 *  @return True if all chiral centers have correct sign.
 */
__device__ __forceinline__ bool validate_chirality_(float const *positions, std::uint32_t num_atoms,
                                                    chiral_center_descriptor_t const *chiral_centers,
                                                    std::uint32_t num_chiral_centers, std::uint32_t lane) {
    bool lane_ok = true;
    for (std::uint32_t c = lane; c < num_chiral_centers; c += warp_size_k) {
        auto const &center = chiral_centers[c];
        float const volume = compute_signed_tetrahedral_volume_(positions, num_atoms, center.center_atom,
                                                                center.neighbor_atoms[0], center.neighbor_atoms[1],
                                                                center.neighbor_atoms[2]);
        std::int8_t const actual_sign = (volume > 0.0f) ? 1 : -1;
        if (actual_sign != center.target_sign) lane_ok = false;
    }
    return __ballot_sync(full_mask_k, lane_ok) == full_mask_k;
}

/** @brief Row stride in bytes for the shared int8 bounds matrix.
 *
 *  The DG inner loop reads `bounds[atom * stride + other]` with `atom` lane-varying and
 *  `other` warp-uniform, so lanes are separated by `stride` bytes.  Shared memory banks
 *  are 4-byte words, making the conflict degree `32 / gcd(stride / 4, 32)` — 32-way at an
 *  unpadded N=128.  Rounding to a multiple of 4 with an odd quotient forces that gcd to
 *  1 for every N, so all 32 lanes land on distinct banks.
 */
__host__ __device__ __forceinline__ std::uint32_t compute_bounds_row_stride(std::uint32_t num_atoms) {
    std::uint32_t stride = (num_atoms + 3u) & ~3u;
    if (((stride / 4u) & 1u) == 0u) stride += 4u;
    return stride;
}

/**
 *  @brief Atom-centric distance geometry energy and gradient evaluation.
 *
 *  Each thread owns atoms {lane, lane+32, lane+64, ...} and accumulates their
 *  gradients in registers. All-pairs distance violations are computed by iterating
 *  other atoms in 32-atom tiles via __shfl_sync.
 *
 *  @tparam write_gradient When true, computes and writes gradient; when false, energy only.
 *  @param[in] bounds_matrix Int8-quantized bounds matrix in shared memory. Upper bound at
 *      [min_idx × bounds_row_stride + max_idx], lower at [max_idx × bounds_row_stride + min_idx].
 *      Dequantize with `q · inv_scale + mol_min`.
 *  @param[in] mol_min Per-molecule floor — the dequantization baseline.
 *  @param[in] inv_scale Per-molecule range / 255 — the per-step size.
 *  @param[in] bounds_row_stride Row stride of @p bounds_matrix in bytes; padded away from
 *      @p num_atoms so the lane-varying row term does not collide on shared-memory banks.
 *  @param[in] chiral_centers Array of chiral center descriptors.
 *  @param[in] num_atoms Number of atoms in the molecule.
 *  @param[in] num_chiral_centers Number of chiral centers.
 *  @param[in] spatial_dimension 3 or 4.
 *  @param[in] upper_violation_weight Weight for upper bound violations.
 *  @param[in] lower_violation_weight Weight for lower bound violations.
 *  @param[in] chiral_volume_weight Weight for chiral volume violations.
 *  @param[in] fourth_dimension_penalty_weight Weight for 4th dimension crush penalty.
 *  @param[in] positions SoA coordinate array.
 *  @param[out] gradient Output gradient array (same layout as positions), unused when !write_gradient.
 *  @param[in] num_elements Total number of coordinate elements.
 *  @param[in] lane Warp lane index (0-31).
 *  @return Total energy (same value in all lanes after warp reduction).
 *  @see fire_minimize_
 */
template <bool write_gradient>
__device__ __forceinline__ float evaluate_distance_geometry_energy(
    std::uint8_t const *__restrict__ bounds_matrix, float mol_min, float inv_scale,
    std::uint32_t bounds_row_stride, chiral_center_descriptor_t const *__restrict__ chiral_centers,
    std::uint32_t num_atoms, std::uint32_t num_chiral_centers, std::uint32_t spatial_dimension,
    float upper_violation_weight, float lower_violation_weight, float chiral_volume_weight,
    float fourth_dimension_penalty_weight, float const *positions, float *gradient, std::uint32_t num_elements,
    std::uint32_t lane) {

    std::uint32_t const dim = spatial_dimension;
    std::uint32_t const num_tiles = (num_atoms + warp_size_k - 1) / warp_size_k;

    if constexpr (write_gradient) {
        for (std::uint32_t i = lane; i < num_elements; i += warp_size_k) gradient[i] = 0.0f;
        __syncwarp(full_mask_k);
    }

    float total_energy = 0.0f;

    // process atoms in 32-atom tiles
    // for each tile of "owned" atoms, iterate all other atoms via shuffle tiles
    for (std::uint32_t tile_a = 0; tile_a < num_tiles; ++tile_a) {
        std::uint32_t const atom = tile_a * warp_size_k + lane;
        bool const atom_valid = atom < num_atoms;

        // load this tile's coordinates into registers
        float coord_a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (atom_valid) {
            for (std::uint32_t d = 0; d < dim; ++d) coord_a[d] = positions[d * num_atoms + atom];
        }

        // gradient accumulators for this atom (registers, no shared memory)
        float grad_a[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // iterate all other atoms in tiles
        for (std::uint32_t tile_b = 0; tile_b < num_tiles; ++tile_b) {
            std::uint32_t const other_base = tile_b * warp_size_k;

            // load tile_b's coordinates into registers for shuffling
            float coord_b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            std::uint32_t const other_load = other_base + lane;
            if (other_load < num_atoms) {
                for (std::uint32_t d = 0; d < dim; ++d) coord_b[d] = positions[d * num_atoms + other_load];
            }

            // each lane broadcasts one atom from tile_b
            // all lanes MUST participate in __shfl_sync (full_mask_k),
            // so shuffles are outside the validity check
            for (std::uint32_t slot = 0; slot < warp_size_k; ++slot) {
                std::uint32_t const other = other_base + slot;

                // all lanes participate in shuffles
                float const bx = __shfl_sync(full_mask_k, coord_b[0], slot);
                float const by = __shfl_sync(full_mask_k, coord_b[1], slot);
                float const bz = __shfl_sync(full_mask_k, coord_b[2], slot);
                float const bw = (dim == 4) ? __shfl_sync(full_mask_k, coord_b[3], slot) : 0.0f;

                // skip invalid pairs after shuffles
                if (!atom_valid || other >= num_atoms || other == atom) continue;

                float const delta_x = bx - coord_a[0];
                float const delta_y = by - coord_a[1];
                float const delta_z = bz - coord_a[2];
                float distance_squared = fmaf(delta_x, delta_x, fmaf(delta_y, delta_y, delta_z * delta_z));
                float delta_w = 0.0f;
                if (dim == 4) {
                    delta_w = bw - coord_a[3];
                    distance_squared = fmaf(delta_w, delta_w, distance_squared);
                }

                float const inverse_distance = rsqrtf(fmaxf(distance_squared, 1e-12f));
                float const distance = distance_squared * inverse_distance;

                // read bounds (int8 → dequantized float via a single FMA per value)
                std::uint32_t const min_idx = (atom < other) ? atom : other;
                std::uint32_t const max_idx = (atom < other) ? other : atom;
                std::uint8_t const upper_q = bounds_matrix[min_idx * bounds_row_stride + max_idx];
                std::uint8_t const lower_q = bounds_matrix[max_idx * bounds_row_stride + min_idx];
                float const upper_bound = __fmaf_rn(__uint2float_rn(upper_q), inv_scale, mol_min);
                float const lower_bound = __fmaf_rn(__uint2float_rn(lower_q), inv_scale, mol_min);

                // upper bound violation
                if (distance > upper_bound) {
                    float const violation = distance - upper_bound;
                    if (atom < other) total_energy += upper_violation_weight * violation * violation;

                    if constexpr (write_gradient) {
                        float const scale = 2.0f * upper_violation_weight * violation * inverse_distance;
                        grad_a[0] -= scale * delta_x;
                        grad_a[1] -= scale * delta_y;
                        grad_a[2] -= scale * delta_z;
                        if (dim == 4) grad_a[3] -= scale * delta_w;
                    }
                }

                // lower bound violation
                if (distance < lower_bound) {
                    float const violation = lower_bound - distance;
                    if (atom < other) total_energy += lower_violation_weight * violation * violation;

                    if constexpr (write_gradient) {
                        float const scale = -2.0f * lower_violation_weight * violation * inverse_distance;
                        grad_a[0] -= scale * delta_x;
                        grad_a[1] -= scale * delta_y;
                        grad_a[2] -= scale * delta_z;
                        if (dim == 4) grad_a[3] -= scale * delta_w;
                    }
                }
            }
        }

        // chiral volume contributions for this atom; volume and gradient share
        // the same edge_b/c/d and the cross(edge_c, edge_d) intermediate.
        if (atom_valid) {
            for (std::uint32_t c = 0; c < num_chiral_centers; ++c) {
                auto const &center = chiral_centers[c];
                std::uint8_t const chiral_center_atom = center.center_atom;
                std::uint8_t const chiral_neighbor_a = center.neighbor_atoms[0];
                std::uint8_t const chiral_neighbor_b = center.neighbor_atoms[1];
                std::uint8_t const chiral_neighbor_c = center.neighbor_atoms[2];

                float3 const center_position = load_position_(positions, num_atoms, chiral_center_atom);
                float3 const edge_b = sub_float3_(load_position_(positions, num_atoms, chiral_neighbor_a),
                                                  center_position);
                float3 const edge_c = sub_float3_(load_position_(positions, num_atoms, chiral_neighbor_b),
                                                  center_position);
                float3 const edge_d = sub_float3_(load_position_(positions, num_atoms, chiral_neighbor_c),
                                                  center_position);
                float3 const cross_cd = cross_float3_(edge_c, edge_d);
                float const volume = dot_float3_(edge_b, cross_cd);

                float const target_sign = static_cast<float>(center.target_sign);
                if (volume * target_sign >= 0.0f) continue;

                // only one thread per chiral center counts the energy
                if (atom == chiral_center_atom) total_energy += chiral_volume_weight * volume * volume;

                if constexpr (write_gradient) {
                    // V = edge_b · (edge_c × edge_d).  Chain rule:
                    //   dV/dr_b = edge_c × edge_d  (= cross_cd, already computed)
                    //   dV/dr_c = edge_d × edge_b
                    //   dV/dr_d = edge_b × edge_c
                    //   dV/dr_center = -(dV/dr_b + dV/dr_c + dV/dr_d)
                    float const scale = 2.0f * chiral_volume_weight * volume;
                    auto add_scaled_to_grad_a_ = [&](float3 const &delta) {
                        grad_a[0] += scale * delta.x;
                        grad_a[1] += scale * delta.y;
                        grad_a[2] += scale * delta.z;
                    };

                    if (atom == chiral_neighbor_a) { add_scaled_to_grad_a_(cross_cd); }
                    else if (atom == chiral_neighbor_b) { add_scaled_to_grad_a_(cross_float3_(edge_d, edge_b)); }
                    else if (atom == chiral_neighbor_c) { add_scaled_to_grad_a_(cross_float3_(edge_b, edge_c)); }
                    else if (atom == chiral_center_atom) {
                        float3 const dvdc = cross_float3_(edge_d, edge_b);
                        float3 const dvdd = cross_float3_(edge_b, edge_c);
                        add_scaled_to_grad_a_(scale_float3_(add_float3_(add_float3_(cross_cd, dvdc), dvdd), -1.0f));
                    }
                }
            }

            // fourth-dimension crush penalty: w4*xi4²
            if (fourth_dimension_penalty_weight > 0.0f && dim == 4) {
                float const w_coord = coord_a[3];
                total_energy += fourth_dimension_penalty_weight * w_coord * w_coord;
                if constexpr (write_gradient) grad_a[3] += 2.0f * fourth_dimension_penalty_weight * w_coord;
            }

            // write gradient for this atom (direct write, no atomicAdd)
            if constexpr (write_gradient) {
                for (std::uint32_t d = 0; d < dim; ++d) gradient[d * num_atoms + atom] = grad_a[d];
            }
        }
    }

    return warp_reduce_sum_f32_(total_energy);
}

/**
 *  @brief Atom-centric distance geometry energy and gradient functor.
 *
 *  Thin wrapper around evaluate_distance_geometry_energy that provides the
 *  operator() / energy_only() interface expected by fire_minimize_.
 *
 *  Bounds matrix stored as uint8_t in shared memory, with the per-molecule
 *  dequantization pair (mol_min, inv_scale) held in registers.
 *  Positions and gradients in SoA float: [x0..xN, y0..yN, z0..zN, (w0..wN)].
 */
struct distance_geometry_energy_functor_t {
    std::uint8_t const *__restrict__ bounds_matrix;
    chiral_center_descriptor_t const *__restrict__ chiral_centers;
    float mol_min;
    float inv_scale;
    std::uint32_t bounds_row_stride; ///< row stride of `bounds_matrix`, padded off `num_atoms`
    std::uint32_t num_atoms;
    std::uint32_t num_chiral_centers;
    std::uint32_t spatial_dimension; // 3 or 4
    float upper_violation_weight;
    float lower_violation_weight;
    float chiral_volume_weight;
    float fourth_dimension_penalty_weight;

    /**
     *  @brief Compute energy and gradient.
     *
     *  Writes ∂E/∂x_i into @p gradient array.
     *
     *  @param[in] positions SoA coordinate array.
     *  @param[out] gradient Output gradient array (same layout as @p positions).
     *  @param[in] num_elements Total number of coordinate elements.
     *  @param[in] lane Warp lane index (0-31).
     *  @return Total energy (same value in all lanes after warp reduction).
     */
    __device__ __forceinline__ float operator()(float const *positions, float *gradient, std::uint32_t num_elements,
                                                std::uint32_t lane) const {
        return evaluate_distance_geometry_energy<true>(
            bounds_matrix, mol_min, inv_scale, bounds_row_stride, chiral_centers, num_atoms, num_chiral_centers,
            spatial_dimension, upper_violation_weight, lower_violation_weight, chiral_volume_weight,
            fourth_dimension_penalty_weight,
            positions, gradient, num_elements, lane);
    }

    /**
     *  @brief Energy-only evaluation (skips gradient writes).
     *
     *  Used by line search.
     *
     *  @param[in] positions SoA coordinate array.
     *  @param[in] num_elements Total number of coordinate elements.
     *  @param[in] lane Warp lane index (0-31).
     *  @return Total energy (same value in all lanes after warp reduction).
     */
    __device__ __forceinline__ float energy_only(float const *positions, std::uint32_t num_elements,
                                                 std::uint32_t lane) const {
        return evaluate_distance_geometry_energy<false>(
            bounds_matrix, mol_min, inv_scale, bounds_row_stride, chiral_centers, num_atoms, num_chiral_centers,
            spatial_dimension, upper_violation_weight, lower_violation_weight, chiral_volume_weight,
            fourth_dimension_penalty_weight,
            positions, nullptr, num_elements, lane);
    }
};

} // namespace smallfield::cuda

#endif // __CUDACC__
