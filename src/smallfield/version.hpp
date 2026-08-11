/**
 *  @file src/smallfield/version.hpp
 *  @brief Library version, as preprocessor macros and as constants.
 *
 *  The `VERSION` file at the repository root is the single source of truth. The release workflow
 *  rewrites the three macros below from it, so nothing here needs to be edited by hand and nothing
 *  needs to assert that the copies agree.
 */

#pragma once

#define SMALLFIELD_VERSION_MAJOR 0
#define SMALLFIELD_VERSION_MINOR 3
#define SMALLFIELD_VERSION_PATCH 0

namespace smallfield {

inline constexpr int version_major_k = SMALLFIELD_VERSION_MAJOR;
inline constexpr int version_minor_k = SMALLFIELD_VERSION_MINOR;
inline constexpr int version_patch_k = SMALLFIELD_VERSION_PATCH;

} // namespace smallfield
