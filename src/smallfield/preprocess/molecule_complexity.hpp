/**
 *  @file molecule_complexity.hpp
 *  @brief Per-molecule work estimate used by the complexity-descending
 *  stripe partition in `distribute_batch_to_per_device_pools_`.
 *
 *  Atom count alone is a weak proxy for kernel wall time.  Within one
 *  `embedding-wave` the dominant GPU costs are:
 *
 *    1. Bounds-matrix smoothing + nonbonded pair scan: O(num_atoms²).
 *    2. FIRE minimisation over MMFF + ETK torsions: linear in torsion count
 *       per iteration, and the iteration count itself scales with torsion
 *       count via `select_refinement_policy_`.
 *    3. A heavy-rescue branch in `mmff_polish_warp_kernel` that runs a
 *       second FIRE stage (doubling iterations) when the molecule crosses
 *       the `is_heavy_refinement_case_` threshold — see
 *       `source/smallfield/cuda/etkdg_pipeline.cuh:86-97`.
 *
 *  The weights here mirror the kernel-side heuristics so the balance
 *  estimate agrees with the bin-dispatch and rescue decisions the kernels
 *  themselves make at runtime.  Pure function, no allocation, no state.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <cstdint>

#include "smallfield/types/etkdg_types.hpp"

namespace smallfield::preprocess {

/** Returns a unit-less work estimate for one molecule's `embedding-step` on
 *  the GPU.  Monotonic with `num_atoms` in expectation, so an ascending
 *  sort by this score still keeps the per-device `atom-limited-bin`
 *  dispatch order sane.  Larger values mean more work, so the stripe
 *  partition sorts by this descending and assigns sorted rank `k` to
 *  device `k % device_count`. */
inline constexpr std::uint64_t molecule_complexity_score_(
    smallfield::types::etkdg_molecule_metadata_t const &meta) noexcept {
    std::uint64_t const num_atoms = meta.num_atoms;
    std::uint64_t const bounds_cost = num_atoms * num_atoms;
    std::uint64_t const torsion_cost = static_cast<std::uint64_t>(meta.num_mmff_torsions) +
                                       static_cast<std::uint64_t>(meta.num_etk_torsions);
    std::uint64_t const term_cost = static_cast<std::uint64_t>(meta.num_mmff_bonds) +
                                    static_cast<std::uint64_t>(meta.num_mmff_angles) +
                                    static_cast<std::uint64_t>(meta.num_mmff_oops) +
                                    static_cast<std::uint64_t>(meta.num_mmff_stretchbends);
    // Matches the kernel-side `total_mmff_terms_` formula (approximate nonbonded as N²/2).
    std::uint64_t const total_mmff_terms = bounds_cost / 2u + term_cost + torsion_cost;
    bool const heavy = num_atoms >= 96u || total_mmff_terms >= 900u || torsion_cost >= 64u;
    std::uint64_t const heavy_multiplier = heavy ? 2u : 1u;
    return heavy_multiplier * (bounds_cost + 4u * torsion_cost + term_cost);
}

} // namespace smallfield::preprocess
