/**
 *  @file src/smallfield/api/conformer_status.hpp
 *  @brief Public per-conformer status enum.
 *
 *  Lives in its own header so `wave_slice_pool.hpp`, which the public `smallfield.hpp` includes, can
 *  reference it without creating a cyclic include with `smallfield.hpp`.
 */

#pragma once

#include <cstdint> // `std::uint8_t`

namespace smallfield::api {

/**
 *  @brief Per-conformer status code.
 */
enum class conformer_status_t : std::uint8_t {
    success_k = 0,
    stereo_failure_k = 1,
    mmff_unconverged_k = 2,
    invalid_smiles_k = 3,
    too_many_atoms_k = 4,
    /** Reserved. Nothing produces this: it marked a `preprocess-step` that exceeded its
     *  per-`molecule` deadline, and the deadline guarded RDKit VF2's pathological worst case. The
     *  native matcher covers the whole pattern corpus, so no substructure search runs.
     *
     *  The value stays because status codes are written as integers into every shipped
     *  `*.3D.parquet`: removing it renumbers everything after it and silently changes what
     *  already-written shards mean. */
    preprocess_timeout_k = 5,
    /** Molecule contains an atom whose `MMFF94` atom type cannot be assigned
     *  by `RDKit::MMFF::MMFFMolProperties::getMMFFAtomType` (returns 0). Covers
     *  transition metals, lanthanides, boron, and other elements outside the MMFF94
     *  parameter set. The `fallback-role` sees these too, and rejects them on the same typing
     *  check before spending an embedding on a molecule it could never minimise. */
    unsupported_element_k = 6,
};

} // namespace smallfield::api
