/**
 *  @file test/preprocess/test_chirality_kernels.hpp
 *  @brief Device harness for the chirality-validation kernel, for use by its unit tests.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

#include "smallfield/types/etkdg_types.hpp"

namespace smallfield::tests {

/**
 *  @brief Validate every chiral center of one molecule on the device from its 3D coordinates.
 *
 *  @param[in] host_coords Flat interleaved xyz coordinates, `3 * num_atoms` entries.
 *  @param[in] host_chirals Chiral centers to check.
 *  @param[in] num_atoms Atom count.
 *  @param[in] num_chirals Number of entries in @p host_chirals.
 *  @return True when every chiral center matches its expected parity.
 */
bool gpu_validate_chirality(float const *host_coords, smallfield::types::chiral_center_descriptor_t const *host_chirals,
                            std::uint32_t num_atoms, std::uint32_t num_chirals);

} // namespace smallfield::tests
