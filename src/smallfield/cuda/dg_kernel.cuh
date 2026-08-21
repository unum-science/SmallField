/**
 *  @file src/smallfield/cuda/dg_kernel.cuh
 *  @brief Atom-centric distance geometry energy and gradient for warp-level L-BFGS, ETKDG stages 4 and 6.
 *
 *  @verbatim
 *  E_total    = E_distance + E_chiral + E_4th_dim
 *  E_distance = Σ_{i<j} [ w_upper * max(0, d_ij - u_ij)^2 + w_lower * max(0, l_ij - d_ij)^2 ]
 *  E_chiral   = Σ_c  w_chiral * (V_c - s_c*v_min_c)^2   (when V_c falls past the signed floor)
 *  E_4th_dim  = w_4 * Σ_i x_i4^2
 *  @endverbatim
 *
 *  Atom-centric: each thread owns ceil(N/32) atoms and accumulates their gradients in registers, and the inner
 *  loop's partner atom is warp-uniform, so its coordinates come from a broadcast read of the shared-memory
 *  position array. No atomicAdd, no pair index scan. The bounds matrix is int8-quantized with a per-molecule
 *  min and inv_scale broadcast in registers: one byte per pair, half what `__half` costs, dequantized at the
 *  read site by a single FMA, `bound = q * inv_scale + mol_min`.
 */

#pragma once

#if defined(__CUDACC__)

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/cuda/warp_primitives.cuh"

namespace smallfield::cuda {
using smallfield::types::chiral_center_descriptor_t;

/**
 *  @brief Compute the signed tetrahedral volume for one chiral center.
 *
 *  V = (r_b - r_a) · ((r_c - r_a) × (r_d - r_a)). Uses only xyz and ignores the 4th dimension. Positions
 *  are in SoA layout: @p positions[dimension * num_atoms + atom].
 *
 *  @param[in] num_atoms Stride between coordinate dimensions, which need not be the molecule's own count.
 *  @param[in] atom_a Pivot atom index — the point the three edges are measured from.
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
 *  Fraction of a center's `minimum_volume` it may fall to before validation calls it flat.
 *  RDKit's `checkChiralCenters` in `Embedder.cpp` compares `vol / lb < .8` against the same window.
 */
constexpr float chiral_volume_slack_k = 0.8f;

/**
 *  @brief Count the chiral centers whose geometry contradicts their descriptor.
 *
 *  A center fails on either of two grounds: the signed volume is handed the wrong way, or its magnitude has
 *  collapsed to a fraction of the window's floor, since a center flat enough to be read either way has proved
 *  nothing. Handedness compares `signbit` against the stored sign rather than a product against it — RDKit
 *  #7560, where a product says nothing at negative zero and underflows for the magnitudes this test catches.
 *
 *  Centers are spread across the warp, so the count is warp-reduced and every lane returns it.
 *
 *  @param[in] num_atoms Stride between coordinate dimensions of @p positions.
 *  @param[in] lane Warp lane index, 0-31.
 *  @return How many centers failed; zero means the stereochemistry is intact.
 */
__device__ __forceinline__ std::uint32_t count_invalid_chiral_centers_(float const *positions, std::uint32_t num_atoms,
                                                                       chiral_center_descriptor_t const *chiral_centers,
                                                                       std::uint32_t num_chiral_centers,
                                                                       std::uint32_t lane) {
    std::uint32_t lane_invalid = 0;
    for (std::uint32_t center_index = lane; center_index < num_chiral_centers; center_index += warp_size_k) {
        auto const &center = chiral_centers[center_index];
        float const volume = compute_signed_tetrahedral_volume_(positions, num_atoms, center.neighbor_atoms[3],
                                                                center.neighbor_atoms[0], center.neighbor_atoms[1],
                                                                center.neighbor_atoms[2]);
        bool const wrong_handed = (signbit(volume) != 0) != (center.target_sign < 0);
        bool const too_flat = fabsf(volume) < chiral_volume_slack_k * center.minimum_volume;
        lane_invalid += (wrong_handed || too_flat) ? 1u : 0u;
    }
    for (std::uint32_t offset = warp_size_k / 2; offset > 0; offset >>= 1)
        lane_invalid += __shfl_xor_sync(full_mask_k, lane_invalid, offset);
    return lane_invalid;
}

/** Row stride in bytes for the shared int8 bounds matrix.
 *
 *  The DG inner loop reads `bounds[atom * stride + other]` with `atom` lane-varying and `other` warp-uniform,
 *  so lanes sit `stride` bytes apart; against 4-byte banks the conflict degree is `32 / gcd(stride / 4, 32)`,
 *  32-way at an unpadded N=128. A multiple of 4 with an odd quotient forces that gcd to 1 for every N.
 */
constexpr std::uint32_t compute_bounds_row_stride(std::uint32_t num_atoms) {
    std::uint32_t stride = (num_atoms + 3u) & ~3u;
    if (((stride / 4u) & 1u) == 0u) stride += 4u;
    return stride;
}

/**
 *  @brief Atom-centric distance geometry energy and gradient, in the shape `fire_minimize_` calls.
 *
 *  Each lane owns atoms {lane, lane+32, lane+64, ...} and accumulates their gradients in registers; all-pairs
 *  distance violations walk every other atom in a warp-uniform inner loop whose coordinates broadcast straight
 *  out of the positions. Positions and gradients are SoA float [x0..xN, y0..yN, z0..zN, (w0..wN)], with
 *  `spatial_dimension` 3 or 4.
 *
 *  `bounds_matrix` is int8 in shared memory, upper at [min × stride + max] and lower at [max × stride + min],
 *  dequantized as `q * inv_scale + mol_min` from the per-molecule floor and range/255 held in registers.
 *  `bounds_row_stride` pads that row off `num_atoms` so the lane-varying row term misses the bank conflict.
 *
 *  @see fire_minimize_
 */
struct distance_geometry_energy_functor_t {
    std::uint8_t const *__restrict__ bounds_matrix;
    chiral_center_descriptor_t const *__restrict__ chiral_centers;
    float mol_min;
    float inv_scale;
    std::uint32_t bounds_row_stride;
    std::uint32_t num_atoms;
    std::uint32_t num_chiral_centers;
    std::uint32_t spatial_dimension;
    float upper_violation_weight;
    float lower_violation_weight;
    float chiral_volume_weight;
    float fourth_dimension_penalty_weight;

    /**
     *  @param[out] gradient Output gradient array, same layout as @p positions.
     *  @param[in] lane Warp lane index, 0-31.
     *  @return Total energy, the same value in all lanes after the warp reduction.
     */
    __device__ __forceinline__ float operator()(float const *positions, float *gradient, std::uint32_t num_elements,
                                                std::uint32_t lane) const {

        std::uint32_t const dim = spatial_dimension;

        for (std::uint32_t element = lane; element < num_elements; element += warp_size_k) gradient[element] = 0.0f;
        __syncwarp(full_mask_k);

        float total_energy = 0.0f;

        // Each lane owns the atoms {lane, lane+32, ...} and walks every other atom for each of them.
        for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k) {

            // Coordinates and gradient accumulator of the owned atom, held in registers.
            float coord_a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (std::uint32_t dimension = 0; dimension < dim; ++dimension)
                coord_a[dimension] = positions[dimension * num_atoms + atom];
            float grad_a[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            // The row index is fixed for this atom, so the row base is hoisted; the transposed read keeps
            // `atom` in the column, conflict-free too because the column varies by one per lane.
            std::uint8_t const *__restrict__ own_bounds_row = bounds_matrix + atom * bounds_row_stride;

            // `other` is warp-uniform, so each partner read broadcasts one shared word: no registers, no shuffles.
            for (std::uint32_t other = 0; other < num_atoms; ++other) {
                if (other == atom) continue;

                float const delta_x = positions[other] - coord_a[0];
                float const delta_y = positions[num_atoms + other] - coord_a[1];
                float const delta_z = positions[2 * num_atoms + other] - coord_a[2];
                float distance_squared = fmaf(delta_x, delta_x, fmaf(delta_y, delta_y, delta_z * delta_z));
                float delta_w = 0.0f;
                if (dim == 4) {
                    delta_w = positions[3 * num_atoms + other] - coord_a[3];
                    distance_squared = fmaf(delta_w, delta_w, distance_squared);
                }

                float const inverse_distance = rsqrtf(fmaxf(distance_squared, 1e-12f));
                float const distance = distance_squared * inverse_distance;

                // Int8 bounds, one FMA each to dequantize; the upper lives above the diagonal and the lower
                // below, so the atom ordering decides which read is which.
                bool const atom_first = atom < other;
                std::uint8_t const own_row_q = own_bounds_row[other];
                std::uint8_t const other_row_q = bounds_matrix[other * bounds_row_stride + atom];
                float const upper_bound = __fmaf_rn(__uint2float_rn(atom_first ? own_row_q : other_row_q), inv_scale,
                                                    mol_min);
                float const lower_bound = __fmaf_rn(__uint2float_rn(atom_first ? other_row_q : own_row_q), inv_scale,
                                                    mol_min);

                // upper bound violation
                if (distance > upper_bound) {
                    float const violation = distance - upper_bound;
                    if (atom_first) total_energy += upper_violation_weight * violation * violation;

                    float const scale = 2.0f * upper_violation_weight * violation * inverse_distance;
                    grad_a[0] -= scale * delta_x;
                    grad_a[1] -= scale * delta_y;
                    grad_a[2] -= scale * delta_z;
                    if (dim == 4) grad_a[3] -= scale * delta_w;
                }

                // lower bound violation
                if (distance < lower_bound) {
                    float const violation = lower_bound - distance;
                    if (atom_first) total_energy += lower_violation_weight * violation * violation;

                    float const scale = -2.0f * lower_violation_weight * violation * inverse_distance;
                    grad_a[0] -= scale * delta_x;
                    grad_a[1] -= scale * delta_y;
                    grad_a[2] -= scale * delta_z;
                    if (dim == 4) grad_a[3] -= scale * delta_w;
                }
            }

            // Chiral volumes for this atom; volume and gradient share the three edges and cross(second, third).
            for (std::uint32_t center_index = 0; center_index < num_chiral_centers; ++center_index) {
                auto const &center = chiral_centers[center_index];
                std::uint8_t const chiral_center_atom = center.center_atom;
                std::uint8_t const chiral_neighbor_first = center.neighbor_atoms[0];
                std::uint8_t const chiral_neighbor_second = center.neighbor_atoms[1];
                std::uint8_t const chiral_neighbor_third = center.neighbor_atoms[2];
                std::uint8_t const chiral_pivot_atom = center.neighbor_atoms[3];

                float3 const pivot_position = load_position_(positions, num_atoms, chiral_pivot_atom);
                float3 const edge_first = sub_float3_(load_position_(positions, num_atoms, chiral_neighbor_first),
                                                      pivot_position);
                float3 const edge_second = sub_float3_(load_position_(positions, num_atoms, chiral_neighbor_second),
                                                       pivot_position);
                float3 const edge_third = sub_float3_(load_position_(positions, num_atoms, chiral_neighbor_third),
                                                      pivot_position);
                float3 const cross_second_third = cross_float3_(edge_second, edge_third);
                float const volume = dot_float3_(edge_first, cross_second_third);

                // Signing the floor puts both failure modes on one axis: a wrongly handed volume and a
                // correctly handed but nearly flat one both land past it, and the distance to it is already
                // the derivative the gradient wants. A sign test alone leaves the flat case no restoring force.
                float const signed_minimum = center.target_sign < 0 ? -center.minimum_volume : center.minimum_volume;
                float const shortfall = volume - signed_minimum;
                float const excess_volume = (signbit(signed_minimum) != signbit(shortfall)) ? shortfall : 0.0f;
                if (excess_volume == 0.0f) continue;

                // only one thread per chiral center counts the energy
                if (atom == chiral_center_atom) total_energy += chiral_volume_weight * excess_volume * excess_volume;

                // V = edge_first · (edge_second × edge_third), so the chain rule makes each gradient the cross
                // of the other two edges and the pivot the negated sum; cross(second, third) is in hand.
                float const scale = 2.0f * chiral_volume_weight * excess_volume;
                auto add_scaled_to_grad_a_ = [&](float3 const &delta) {
                    grad_a[0] += scale * delta.x;
                    grad_a[1] += scale * delta.y;
                    grad_a[2] += scale * delta.z;
                };

                if (atom == chiral_neighbor_first) { add_scaled_to_grad_a_(cross_second_third); }
                else if (atom == chiral_neighbor_second) {
                    add_scaled_to_grad_a_(cross_float3_(edge_third, edge_first));
                }
                else if (atom == chiral_neighbor_third) {
                    add_scaled_to_grad_a_(cross_float3_(edge_first, edge_second));
                }
                else if (atom == chiral_pivot_atom) {
                    float3 const gradient_second = cross_float3_(edge_third, edge_first);
                    float3 const gradient_third = cross_float3_(edge_first, edge_second);
                    add_scaled_to_grad_a_(scale_float3_(
                        add_float3_(add_float3_(cross_second_third, gradient_second), gradient_third), -1.0f));
                }
            }

            // fourth-dimension crush penalty: w_4 * x_i4^2
            if (fourth_dimension_penalty_weight > 0.0f && dim == 4) {
                float const w_coord = coord_a[3];
                total_energy += fourth_dimension_penalty_weight * w_coord * w_coord;
                grad_a[3] += 2.0f * fourth_dimension_penalty_weight * w_coord;
            }

            // write gradient for this atom (direct write, no atomicAdd)
            for (std::uint32_t dimension = 0; dimension < dim; ++dimension)
                gradient[dimension * num_atoms + atom] = grad_a[dimension];
        }

        return warp_reduce_sum_f32_(total_energy);
    }
};

} // namespace smallfield::cuda

#endif // __CUDACC__
