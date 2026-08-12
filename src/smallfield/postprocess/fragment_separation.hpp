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
 *  @brief One connected component's bounding sphere and the rigid move that places it.
 *
 *  The three quantities are per-component and walked by the same label, so they are one record rather than
 *  three arrays a label can index inconsistently.
 */
struct component_placement_t {
    point3_t centroid;
    double radius = 0.0;
    std::uint32_t num_atoms = 0;
    point3_t translation;
};

/**
 *  @brief Rigidly translates each connected component so bounding spheres do not overlap.
 *
 *  Lays every component's centroid on the +x axis in label order, spacing consecutive centroids by the sum of
 *  their bounding-sphere radii plus a fixed margin. Each component's atoms move by the same delta, a pure
 *  translation, so intra-component distances are preserved. Single-component molecules are left untouched.
 *
 *  Every pass is over the atoms once: the placement is solved per component first, then applied in one sweep,
 *  rather than rescanning every atom once per component.
 *
 *  @param[inout] coords_interleaved Tight (num_atoms, 3) row-major xyz for one conformer.
 *  @param[in] context Context from `build_shape_context` supplying the per-atom component labels.
 */
inline void separate_fragments(custd::span<float> coords_interleaved, molecule_shape_context_t const &context) {
    if (context.num_components <= 1) return;
    sf_assert_(context.component_label.size() >= context.num_atoms);
    sf_assert_(coords_interleaved.size() >= static_cast<std::size_t>(context.num_atoms) * 3u);

    constexpr double margin_k = 2.0;
    std::uint32_t const num_atoms = context.num_atoms;
    std::vector<component_placement_t> placements(context.num_components);

    // Centroid of each component.
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        sf_assert_(context.component_label[atom_index] < context.num_components);
        component_placement_t &placement = placements[context.component_label[atom_index]];
        point3_t const point = read_point_(coords_interleaved, atom_index);
        placement.centroid.x += point.x;
        placement.centroid.y += point.y;
        placement.centroid.z += point.z;
        ++placement.num_atoms;
    }
    for (component_placement_t &placement : placements) {
        if (placement.num_atoms == 0) continue; // a label no atom carries has no centroid to normalize
        placement.centroid.x /= placement.num_atoms;
        placement.centroid.y /= placement.num_atoms;
        placement.centroid.z /= placement.num_atoms;
    }

    // Bounding-sphere radius of each component about its centroid.
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        component_placement_t &placement = placements[context.component_label[atom_index]];
        double const distance = point_distance_(read_point_(coords_interleaved, atom_index), placement.centroid);
        if (distance > placement.radius) placement.radius = distance;
    }

    // Target centroids strung along +x so consecutive spheres clear each other.
    double cursor = 0.0;
    for (component_placement_t &placement : placements) {
        double const target_x = cursor + placement.radius;
        placement.translation = {target_x - placement.centroid.x, -placement.centroid.y, -placement.centroid.z};
        cursor = target_x + placement.radius + margin_k;
    }

    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        point3_t const &translation = placements[context.component_label[atom_index]].translation;
        std::size_t const base = static_cast<std::size_t>(atom_index) * 3u;
        coords_interleaved[base + 0u] += static_cast<float>(translation.x);
        coords_interleaved[base + 1u] += static_cast<float>(translation.y);
        coords_interleaved[base + 2u] += static_cast<float>(translation.z);
    }
}

} // namespace smallfield::postprocess
