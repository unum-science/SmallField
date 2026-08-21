/**
 *  @file test/preprocess/test_chirality_kernels.hpp
 *  @brief Device harness for the chirality-validation kernel, for use by its unit tests.
 */

#pragma once

#include <cstdint> // `std::uint8_t`

#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::tests {

/** What the device made of one molecule's chiral targets: `satisfied_k` every center realizes the signed tetrahedral
 *  volume its target demands, `violated_k` at least one contradicts it, `unavailable_k` the device refused the
 *  staging so nothing was checked. Three states rather than a `bool`, because "never asked" is not a verdict: folded
 *  into `false` it reads exactly like an inverted stereocenter, and a refused allocation reports a parity mismatch. */
enum class chirality_verdict_t : std::uint8_t {
    satisfied_k,
    violated_k,
    unavailable_k,
};

/** One molecule's axis-major conformer, read and never written by the device check. */
using conformer_soa_reader_t =
    smallfield::types::coordinates_soa<smallfield::types::coordinate_role_t::position_k, float const,
                                       smallfield::types::conformer_dimensions_k>;

/**
 *  @brief Validate every chiral center of one molecule on the device from its 3D coordinates.
 *  @param[in] host_coords Axis-major; the atom count is the view's first extent, and an atom-major block would
 *      validate a different molecule.
 *  @return Whether every center matches its expected parity, or that the device could not be asked.
 */
chirality_verdict_t gpu_validate_chirality(
    conformer_soa_reader_t host_coords, custd::span<smallfield::types::chiral_center_descriptor_t const> host_chirals);

} // namespace smallfield::tests
