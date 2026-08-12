/**
 *  @file test/preprocess/test_chirality_kernels.hpp
 *  @brief Device harness for the chirality-validation kernel, for use by its unit tests.
 */

#pragma once

#include <cstdint> // `std::uint8_t`

#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::tests {

/**
 *  @brief What the device made of one molecule's chiral targets.
 *
 *  Three states rather than a `bool`, because "the device was never asked" is not a verdict: folded into
 *  `false` it reads exactly like an inverted stereocenter, and the caller reports a parity mismatch for
 *  what was really a refused allocation.
 */
enum class chirality_verdict_t : std::uint8_t {
    /** Every center realizes the signed tetrahedral volume its target demands. */
    satisfied_k,
    /** At least one center contradicts its target. */
    violated_k,
    /** The device refused the staging, so nothing was checked. */
    unavailable_k,
};

/** One molecule's conformer, read and never written by the device check. */
using conformer_reader_t = smallfield::types::coordinates_soa<smallfield::types::coordinate_role_t::position_k,
                                                              float const, smallfield::types::conformer_dimensions_k>;

/**
 *  @brief Validate every chiral center of one molecule on the device from its 3D coordinates.
 *
 *  @param[in] host_coords Axis-major coordinates; the atom count is the view's own first extent, and an
 *      atom-major block passed here would validate a different molecule.
 *  @param[in] host_chirals Chiral centers to check.
 *  @return Whether every center matches its expected parity, or that the device could not be asked.
 */
chirality_verdict_t gpu_validate_chirality(
    conformer_reader_t host_coords, custd::span<smallfield::types::chiral_center_descriptor_t const> host_chirals);

} // namespace smallfield::tests
