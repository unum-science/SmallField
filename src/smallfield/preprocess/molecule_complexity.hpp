/**
 *  @file src/smallfield/preprocess/molecule_complexity.hpp
 *  @brief Per-molecule complexity predicates, shared by the host stripe partition and the device kernels.
 *
 *  Atom count alone is a weak proxy for kernel wall time. Within one `embedding-wave` the dominant GPU costs are:
 *
 *    1. Bounds-matrix smoothing and the nonbonded pair scan: O(num_atoms²).
 *    2. FIRE minimisation over MMFF and ETK torsions: linear in torsion count per iteration, and the iteration
 *       count itself scales with torsion count via `select_refinement_policy_`.
 *    3. A heavy-rescue branch in `mmff_polish_warp_kernel` that runs a second FIRE stage, doubling the iterations,
 *       when `is_heavy_refinement_case` holds.
 *
 *  Nothing here touches CUDA, so `src/smallfield/cuda/etkdg_pipeline.cuh` includes it and the balance estimate
 *  reads the same definition the kernels dispatch on rather than a restatement that has to be kept in step.
 *  Pure functions, no allocation, no state.
 */

#pragma once

#include <cstdint> // `std::uint64_t`

#include "smallfield/types/etkdg_types.hpp"

namespace smallfield::preprocess {

/** Approximate MMFF94 term count for one molecule. The nonbonded terms are enumerated on the fly rather than
 *  stored, so they enter as the atom-pair count instead of a length. */
inline constexpr std::uint32_t total_mmff_terms(smallfield::types::etkdg_molecule_metadata_t const &metadata) noexcept {
    std::uint32_t const nonbonded_pairs = metadata.num_atoms * (metadata.num_atoms - 1u) / 2u;
    return nonbonded_pairs + metadata.num_mmff_bonds + metadata.num_mmff_angles + metadata.num_mmff_torsions +
           metadata.num_mmff_oops + metadata.num_mmff_stretchbends;
}

/** True iff the molecule takes the widened FIRE budget and the `mmff_polish_warp_kernel` heavy-rescue branch. */
inline constexpr bool is_heavy_refinement_case(smallfield::types::etkdg_molecule_metadata_t const &metadata) noexcept {
    std::uint32_t const torsion_complexity = metadata.num_etk_torsions + metadata.num_mmff_torsions;
    return metadata.num_atoms >= 96u || total_mmff_terms(metadata) >= 900u || torsion_complexity >= 64u;
}

/** Returns a unit-less work estimate for one molecule's `embedding-step` on
 *  the GPU. Monotonic with `num_atoms` in expectation, so an ascending
 *  sort by this score still keeps the per-device `atom-limited-bin`
 *  dispatch order sane. Larger values mean more work, so the stripe
 *  partition sorts by this descending and assigns sorted rank `k` to
 *  device `k % device_count`. */
inline constexpr std::uint64_t molecule_complexity_score_(
    smallfield::types::etkdg_molecule_metadata_t const &metadata) noexcept {
    std::uint64_t const num_atoms = metadata.num_atoms;
    std::uint64_t const bounds_cost = num_atoms * num_atoms;
    std::uint64_t const torsion_cost = static_cast<std::uint64_t>(metadata.num_mmff_torsions) +
                                       static_cast<std::uint64_t>(metadata.num_etk_torsions);
    std::uint64_t const term_cost = static_cast<std::uint64_t>(metadata.num_mmff_bonds) +
                                    static_cast<std::uint64_t>(metadata.num_mmff_angles) +
                                    static_cast<std::uint64_t>(metadata.num_mmff_oops) +
                                    static_cast<std::uint64_t>(metadata.num_mmff_stretchbends);
    std::uint64_t const heavy_multiplier = is_heavy_refinement_case(metadata) ? 2u : 1u;
    return heavy_multiplier * (bounds_cost + 4u * torsion_cost + term_cost);
}

} // namespace smallfield::preprocess
