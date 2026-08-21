/**
 *  @file src/smallfield/types/conformer_status.hpp
 *  @brief Per-conformer status enum, and the vocabulary every rejection in the pipeline speaks.
 *
 *  A leaf header with no project dependency: `build_descriptor` answers with one from a bare flat
 *  descriptor, the warp kernels write one per slot, `api/smallfield.hpp` re-exports it to callers.
 */

#pragma once

#include <cmath>   // `std::isfinite`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`

#include <string_view> // `std::string_view`

namespace smallfield::types {

/**
 *  @brief Per-conformer status code. `success_k` is the only value a conformer survives.
 *
 *  @b Append only. Every value is written as an integer into every shipped `*.3D.parquet`, so renumbering
 *  or removing one silently changes what already-written shards mean; a value that stops being produced
 *  keeps its slot and says so, and a new refusal takes the next free number.
 */
enum class conformer_status_t : std::uint8_t {
    /** Embedded, polished, and carrying an energy `energy_is_physical` accepts. */
    success_k = 0,
    /** The embedding could not satisfy a declared chiral center or E/Z bond within the attempt budget. */
    stereo_failure_k = 1,
    /** The MMFF94 minimiser ran and did not reach a stationary point. Also the batch-level verdict
     *  `generate_conformers` returns when no conformer in the call succeeded. */
    mmff_unconverged_k = 2,
    /** RDKit refused the SMILES. Also the pre-fill every output slot carries, so a molecule no wave
     *  reached reads as unparsed rather than as whatever the caller's buffer held. */
    invalid_smiles_k = 3,
    /** More atoms than the single-warp pipeline's `max_atoms_per_molecule_k`. */
    too_many_atoms_k = 4,
    /** Reserved, produced by nothing: it marked a `preprocess-step` past its per-molecule deadline, and
     *  that deadline guarded RDKit VF2's pathological worst case, which the native matcher removed by
     *  covering the whole pattern corpus with no substructure search. Keeps its slot under the append-only
     *  rule above, because the values after it are already written into shipped shards. */
    preprocess_timeout_k = 5,
    /** An atom `RDKit::MMFF::MMFFMolProperties::getMMFFAtomType` cannot assign — a transition metal,
     *  lanthanide, boron, anything outside MMFF94. `fallback-role` refuses it there, before embedding. */
    unsupported_element_k = 6,
    /** More bonds than `max_bonds_k`. Distinct from `invalid_smiles_k`: the SMILES parsed, and the
     *  molecule is simply wider than the flat descriptor's bond arrays. */
    too_many_bonds_k = 7,
    /** More SSSR rings than `atom_ring_bitmap` has bits, or more membership entries than `ring_atoms`
     *  holds. Either leaves an in-ring atom unlabelled, so the molecule is refused, not half-perceived. */
    too_many_rings_k = 8,
    /** Triangle smoothing drove some atom pair's lower bound above its upper, so no coordinates satisfy
     *  the matrix. Named apart from `stereo_failure_k`, the symptom such a matrix shows downstream. */
    bounds_infeasible_k = 9,
    /** The slot's MMFF94 energy is not finite, so neither is the geometry behind it: a `NaN` breaks the
     *  strict-weak ordering `std::stable_sort` requires and an infinity would be published as the
     *  molecule's best conformer. The slot keeps the value it was refused for, the only evidence left. */
    nonphysical_energy_k = 10,
    /** The molecule fits every extent it was measured against; the batch it landed in did not fit its
     *  producer. Three sites reach it: a slice outgrowing what a 32-bit per-molecule region offset
     *  addresses, a device whose wave pool or pinned staging could not be reserved, and a metric prepass
     *  bin its scratch cannot hold. Not `too_many_atoms_k`: the same input at a smaller wave succeeds,
     *  so an atom-count name would report a sizing bug as a property of the corpus. */
    batch_overflow_k = 11,
    /** No kernel wrote this slot: it carries what the wave's priming walk left there. Distinct from
     *  `mmff_unconverged_k`, where a geometry exists and is simply not at a stationary point. */
    slot_unwritten_k = 12,
    /** The MMFF polish moved a declared tetrahedral center onto the wrong handedness, caught by the
     *  post-polish validation. Split from `stereo_failure_k` so a corpus run sees which stage inverts. */
    polish_inverted_center_k = 13,
    /** The same, for a declared E/Z double bond. Separate from the center above because a corpus can be
     *  wrong in both at once at rates a single count cannot tell apart. */
    polish_inverted_bond_k = 14,
};

/** One past the highest `conformer_status_t`, so a per-status array is indexed by the enumerator
 *  itself. The enum is append-only, so its newest value is always its widest. */
inline constexpr std::size_t conformer_status_count_k =
    static_cast<std::size_t>(conformer_status_t::polish_inverted_bond_k) + 1u;

/** What a status byte no enumerator claims prints as, which a shard from a newer producer can carry. */
inline constexpr std::string_view unnamed_status_k = "unnamed_status";

/** The name a per-status report prints, the enumerator without its suffix. Exhaustive and carrying no
 *  `default:`, so appending to the enum is a `-Wswitch` warning rather than a row under a wrong name. */
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
    case conformer_status_t::nonphysical_energy_k: return "nonphysical_energy";
    case conformer_status_t::batch_overflow_k: return "batch_overflow";
    case conformer_status_t::slot_unwritten_k: return "slot_unwritten";
    case conformer_status_t::polish_inverted_center_k: return "polish_inverted_center";
    case conformer_status_t::polish_inverted_bond_k: return "polish_inverted_bond";
    }
    return unnamed_status_k;
}

static_assert(conformer_status_name(static_cast<conformer_status_t>(conformer_status_count_k - 1)) !=
                      unnamed_status_k &&
                  conformer_status_name(static_cast<conformer_status_t>(conformer_status_count_k)) == unnamed_status_k,
              "every enumerator carries a name, and nothing past the last one does");

/** Whether an MMFF94 energy is one a conformer can be ranked and published by. Shared by both producers
 *  of a `success_k` slot, so the GPU pipeline and the `fallback-role` cannot disagree. Finiteness only:
 *  telling a diverged minimization from a genuinely low energy needs a paired delta against RDKit that
 *  production does not have, so a magnitude bound would refuse on a number nothing here can judge. */
inline constexpr auto energy_is_physical = [](float energy) noexcept { return std::isfinite(energy); };

} // namespace smallfield::types
