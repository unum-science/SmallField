/**
 *  @file src/smallfield/postprocess/fragment_separation.hpp
 *  @brief Pulls the connected components of one conformer apart so their bounding spheres clear each other.
 *
 *  A salt or co-crystal embeds as several disconnected fragments whose coordinates the embedder is free to
 *  interleave in space. Downstream shape descriptors and viewers both read better when each component sits in
 *  its own region, so this lays them along the +x axis by rigid translation, leaving every intra-component
 *  distance untouched.
 *
 *  Reads the component labels off `molecule_shape_context_t`, whose only producer is `build_shape_context` in
 *  `postprocess/shape_descriptors.hpp`.
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <vector> // `std::vector`

#include "smallfield/postprocess/shape_descriptors.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::postprocess {

/**
 *  @brief Rigidly translates each connected component so bounding spheres do not overlap.
 *
 *  Lays every component's centroid on the +x axis in label order, spacing consecutive centroids by the sum of
 *  their bounding-sphere radii plus a fixed margin. Each component's atoms move by the same delta, a pure
 *  translation, so intra-component distances are preserved. Single-component molecules are left untouched.
 */
inline void separate_fragments(custd::span<float> coords_interleaved, molecule_shape_context_t const &context) {
    if (context.num_components <= 1) return;

    constexpr double margin_k = 2.0;
    std::uint32_t const num_components = context.num_components;

    // Per-component centroid and count.
    std::vector<point3_t> centroid(num_components);
    std::vector<std::uint32_t> count(num_components, 0);
    for (std::uint32_t atom_index = 0; atom_index < context.num_atoms; ++atom_index) {
        std::uint8_t const label = context.component_label[atom_index];
        std::size_t const base = static_cast<std::size_t>(atom_index) * 3u;
        centroid[label].x += coords_interleaved[base + 0u];
        centroid[label].y += coords_interleaved[base + 1u];
        centroid[label].z += coords_interleaved[base + 2u];
        ++count[label];
    }
    for (std::uint32_t component = 0; component < num_components; ++component) {
        if (count[component] > 0) {
            centroid[component].x /= count[component];
            centroid[component].y /= count[component];
            centroid[component].z /= count[component];
        }
    }

    // Bounding-sphere radius of each component about its centroid.
    std::vector<double> radius(num_components, 0.0);
    for (std::uint32_t atom_index = 0; atom_index < context.num_atoms; ++atom_index) {
        std::uint8_t const label = context.component_label[atom_index];
        double const distance = point_distance_(read_point_(coords_interleaved, atom_index), centroid[label]);
        if (distance > radius[label]) radius[label] = distance;
    }

    // Target centroids strung along +x so consecutive spheres clear each other.
    double cursor = 0.0;
    for (std::uint32_t component = 0; component < num_components; ++component) {
        double const target_x = cursor + radius[component];
        double const delta_x = target_x - centroid[component].x;
        double const delta_y = -centroid[component].y;
        double const delta_z = -centroid[component].z;
        for (std::uint32_t atom_index = 0; atom_index < context.num_atoms; ++atom_index) {
            if (context.component_label[atom_index] != component) continue;
            std::size_t const base = static_cast<std::size_t>(atom_index) * 3u;
            coords_interleaved[base + 0u] += static_cast<float>(delta_x);
            coords_interleaved[base + 1u] += static_cast<float>(delta_y);
            coords_interleaved[base + 2u] += static_cast<float>(delta_z);
        }
        cursor = target_x + radius[component] + margin_k;
    }
}

} // namespace smallfield::postprocess
