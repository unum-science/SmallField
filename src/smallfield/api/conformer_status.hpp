/**
 *  @file src/smallfield/api/conformer_status.hpp
 *  @brief Public per-conformer status enum.
 *
 *  Lives in its own header so `wave_slice_pool.hpp`, which the public `smallfield.hpp` includes, can
 *  reference it without creating a cyclic include with `smallfield.hpp`.
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`

#include <string_view> // `std::string_view`

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
    /** Triangle smoothing drove some atom pair's lower bound above its upper, so no set of
     *  coordinates satisfies the molecule's bounds matrix. `smoothing_t::infeasible_k` is the
     *  device-side verdict that names it, and the smoothed matrix the prepass persists carries the
     *  crossed pair. Distinct from `stereo_failure_k`: an impossible matrix embedded anyway fails
     *  downstream under a name that describes the symptom rather than the cause. */
    bounds_infeasible_k = 9,
};

/** One past the highest `conformer_status_t`, so a per-status array is indexed by the enumerator
 *  itself. The enum is append-only, so its newest value is always its widest. */
inline constexpr std::size_t conformer_status_count_k =
    static_cast<std::size_t>(conformer_status_t::bounds_infeasible_k) + 1u;

/** What a status byte no enumerator claims prints as, which a shard from a newer producer can carry. */
inline constexpr std::string_view unnamed_status_k = "unnamed_status";

/**
 *  @brief The name a per-status report prints, which is the enumerator without its suffix.
 *
 *  Exhaustive and carrying no `default:`, so appending to the enum is a `-Wswitch` warning here rather
 *  than a row printed under a neighbour's name.
 */
inline constexpr std::string_view conformer_status_name(conformer_status_t status) noexcept {
    switch (status) {
    case conformer_status_t::success_k: return "success";
    case conformer_status_t::stereo_failure_k: return "stereo_failure";
    case conformer_status_t::mmff_unconverged_k: return "mmff_unconverged";
    case conformer_status_t::invalid_smiles_k: return "invalid_smiles";
    case conformer_status_t::too_many_atoms_k: return "too_many_atoms";
    case conformer_status_t::preprocess_timeout_k: return "preprocess_timeout";
    case conformer_status_t::unsupported_element_k: return "unsupported_element";
    case conformer_status_t::too_many_bonds_k: return "too_many_bonds";
    case conformer_status_t::too_many_rings_k: return "too_many_rings";
    case conformer_status_t::bounds_infeasible_k: return "bounds_infeasible";
    }
    return unnamed_status_k;
}

static_assert(conformer_status_name(static_cast<conformer_status_t>(conformer_status_count_k - 1)) !=
                      unnamed_status_k &&
                  conformer_status_name(static_cast<conformer_status_t>(conformer_status_count_k)) == unnamed_status_k,
              "every enumerator carries a name, and nothing past the last one does");

} // namespace smallfield::api
