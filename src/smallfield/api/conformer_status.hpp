/**
 *  @file conformer_status.hpp
 *  @brief Public per-conformer status enum.
 *
 *  Lives in its own header so `batch_pool.hpp` (which the public
 *  `smallfield.hpp` includes) can reference it without creating a cyclic
 *  include with `smallfield.hpp`.
 */

#pragma once

#include <cstdint>

namespace smallfield::api {

/**
 *  @brief Per-conformer status code.
 */
enum class conformer_status_t : std::uint8_t {
    success = 0,
    stereo_failure = 1,
    mmff_unconverged = 2,
    invalid_smiles = 3,
    too_many_atoms = 4,
    /** @brief `preprocess-step` exceeded the per-`molecule` deadline (RDKit VF2 worst-case).
     *  Downstream `fallback-role` may still successfully produce ETKDGv3 conformers. */
    preprocess_timeout = 5,
    /** @brief Molecule contains an atom whose `MMFF94` atom type cannot be assigned
     *  by `RDKit::MMFF::MMFFMolProperties::getMMFFAtomType` (returns 0).  Covers
     *  transition metals, lanthanides, and other elements outside the MMFF94
     *  parameter set.  Routed to the `fallback-role` for ETKDGv3 (no MMFF). */
    unsupported_element = 6,
};

} // namespace smallfield::api
