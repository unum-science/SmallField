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
 *  @brief Per-conformer status code, and the vocabulary every rejection in the pipeline speaks.
 *
 *  @b Append only. Every value is written as an integer into every shipped `*.3D.parquet`, so
 *  renumbering or removing one silently changes what already-written shards mean. A value that
 *  stops being produced keeps its slot and says so; a new refusal takes the next free number.
 *
 *  `success_k` is the only value a conformer survives; everything else names why it did not.
 */
enum class conformer_status_t : std::uint8_t {
    success_k = 0,
    stereo_failure_k = 1,
    mmff_unconverged_k = 2,
    invalid_smiles_k = 3,
    /** More atoms than the single-warp pipeline's `max_atoms_per_molecule_k`. */
    too_many_atoms_k = 4,
    /** Reserved. Nothing produces this: it marked a `preprocess-step` that exceeded its
     *  per-`molecule` deadline, and the deadline guarded RDKit VF2's pathological worst case. The
     *  native matcher covers the whole pattern corpus, so no substructure search runs. */
    preprocess_timeout_k = 5,
    /** Molecule contains an atom whose `MMFF94` atom type cannot be assigned
     *  by `RDKit::MMFF::MMFFMolProperties::getMMFFAtomType` (returns 0). Covers
     *  transition metals, lanthanides, boron, and other elements outside the MMFF94
     *  parameter set. The `fallback-role` sees these too, and rejects them on the same typing
     *  check before spending an embedding on a molecule it could never minimise. */
    unsupported_element_k = 6,
    /** More bonds than `max_bonds_k`, which only a molecule under the atom cap carrying an
     *  unusual number of ring closures reaches. Distinct from `invalid_smiles_k`: the SMILES
     *  parsed, and the molecule is simply wider than the flat descriptor's bond arrays. */
    too_many_bonds_k = 7,
    /** More SSSR rings than `atom_ring_bitmap` has bits, or more ring-membership entries than
     *  `ring_atoms` holds. Either one leaves a genuinely in-ring atom carrying no ring label, and
     *  MMFF typing and the CSD torsion match then read the molecule as acyclic where it is not,
     *  so the molecule is refused rather than parametrised from partial ring perception. */
    too_many_rings_k = 8,
};

} // namespace smallfield::api
