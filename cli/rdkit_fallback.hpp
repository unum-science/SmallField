/**
 *  @file cli/rdkit_fallback.hpp
 *  @brief Declares `rdkit_fallback`, RDKit conformer generation for molecules the GPU pipeline failed on.
 *
 *  Each recovered conformer overwrites its coordinate and energy slot and flips it to `success_k`; an unfilled
 *  slot keeps the status the GPU left, which the writer emits as a null row.
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <array>       // `std::array`
#include <bit>         // `std::bit_width`
#include <string_view> // `std::string_view`

#include <cuda_fp16.h>

#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/types/conformer_status.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::cli {

#pragma region Dispatch Policy

/** Whether the `fallback-role` can do anything with a slot the `gpu-runner-role` refused. */
enum class fallback_disposition_t : std::uint8_t {
    /** RDKit embeds by a different route and may fill the slot. */
    retry_k,
    /** Nothing worth an embedding: either `admit_molecule_` re-derives the refusal from the same caps and
     *  parser, buying one `SmilesToMol` and the verdict the slot holds, or the refusal is the producer's own. */
    terminal_k,
};

/** Whether dispatching a slot at @p status to the `fallback-role` could ever fill it. Exhaustive over
 *  `conformer_status_t` with no `default:`, so a status appended to that append-only enum is a `-Wswitch`
 *  warning here rather than a policy inherited in silence. Each verdict follows from what `admit_molecule_`
 *  runs — `parse_smiles` under the GPU path's `max_atoms` and salt policy, then
 *  `MMFF::MMFFMolProperties::isValid` — which builds no `molecule_descriptor_t`, so its extents are not
 *  re-derivable. */
[[nodiscard]] constexpr fallback_disposition_t fallback_disposition(types::conformer_status_t status) noexcept {
    switch (status) {

    // Terminal: the fallback reaches the same answer by the same route.
    // The slot already holds a conformer, and the fallback leaves a filled slot alone.
    case types::conformer_status_t::success_k: return fallback_disposition_t::terminal_k;
    // `parse_smiles` gates the atom count against the cap the `gpu-runner-role` passed.
    case types::conformer_status_t::too_many_atoms_k: return fallback_disposition_t::terminal_k;
    // `MMFFMolProperties::isValid` is the same typing check on the same molecule.
    case types::conformer_status_t::unsupported_element_k: return fallback_disposition_t::terminal_k;
    // Terminal though RDKit could embed these: the population is unbounded where every other retried status is
    // per-molecule and rare, and one mis-sized wave refuses a whole device's batch — at 20,088 µs of CPU per
    // molecule against 250 µs on GPU that is hours of fallback reading as a hang, not the loud, cheap,
    // re-runnable error it is. Two producers reach it: a device-side pool failure, which `cuda_error_count`
    // reports, and a host-side pinned reservation, which raises none — so the exit code gates on this status's
    // own terminal tally. Re-run the shard at a wave that fits, not laundered one molecule at a time.
    case types::conformer_status_t::batch_overflow_k: return fallback_disposition_t::terminal_k;

    // Retry: nothing in `admit_molecule_` reproduces these.
    // ETKDG perceives and enforces chirality by its own route, so a chiral volume the driver missed differs.
    case types::conformer_status_t::stereo_failure_k: return fallback_disposition_t::retry_k;
    // A different MMFF minimizer from the device FIRE driver may still converge where it did not.
    case types::conformer_status_t::mmff_unconverged_k: return fallback_disposition_t::retry_k;
    // No kernel reached the slot, so there is no device verdict to re-derive and the molecule was never refused.
    case types::conformer_status_t::slot_unwritten_k: return fallback_disposition_t::retry_k;
    // Both are the post-polish stereo refusal `stereo_failure_k` already routes here for: a centre or bond our
    // force field walked out of its basin is not a verdict RDKit's own chirality enforcement shares.
    case types::conformer_status_t::polish_inverted_center_k: return fallback_disposition_t::retry_k;
    case types::conformer_status_t::polish_inverted_bond_k: return fallback_disposition_t::retry_k;
    // Two producers, and only the parse is re-derived: `prepare_molecule_topology_into` can also throw
    // out of descriptor, bounds or chirality construction, which the fallback never runs.
    case types::conformer_status_t::invalid_smiles_k: return fallback_disposition_t::retry_k;
    // Nothing produces it, and the fallback carries no deadline that could re-derive it.
    case types::conformer_status_t::preprocess_timeout_k: return fallback_disposition_t::retry_k;
    // Both are `molecule_descriptor_t` extents the fallback never builds; RDKit embeds wider molecules fine.
    case types::conformer_status_t::too_many_bonds_k: return fallback_disposition_t::retry_k;
    case types::conformer_status_t::too_many_rings_k: return fallback_disposition_t::retry_k;
    // `EmbedMultipleConfs` smooths its own bounds by a different route and may not agree the matrix
    // is infeasible. Whether it ever does is what the per-status recovery tally measures.
    case types::conformer_status_t::bounds_infeasible_k: return fallback_disposition_t::retry_k;
    // RDKit reaches geometry by its own embedding and minimizer, so a non-finite energy is not one it re-derives.
    case types::conformer_status_t::nonphysical_energy_k: return fallback_disposition_t::retry_k;
    }
    // A status byte outside the enum keeps its second chance rather than being dropped.
    return fallback_disposition_t::retry_k;
}

/** The status that sends one molecule's slots to the `fallback-role`, or `success_k` when none does. */
[[nodiscard]] inline types::conformer_status_t fallback_dispatch_cause(
    custd::span<types::conformer_status_t const> slots) noexcept {
    for (types::conformer_status_t const slot_status : slots)
        if (fallback_disposition(slot_status) == fallback_disposition_t::retry_k) return slot_status;
    return types::conformer_status_t::success_k;
}

#pragma endregion Dispatch Policy

#pragma region Cost Accounting

/** One stage of what the `fallback-role` spends on a molecule, in the order the stages run. */
enum class fallback_stage_t : std::uint8_t {
    /** `parse_smiles`: `SmilesToMol`, sanitization, counterion removal and `addHs`. */
    parse_k,
    /** `MMFFMolProperties`: aromaticity re-perception, ring build, atom typing and partial charges. */
    type_k,
    /** `EmbedMultipleConfs`: bounds smoothing plus one distance-geometry embedding per unfilled slot. */
    embed_k,
    /** `constructForceField`: MMFF term construction over one conformer. */
    build_k,
    /** `ForceField::minimize` and the energy read-back. */
    minimize_k,
    /** `write_conformer_into_slot_`: the fp16 scatter into the batch's coordinate arrays. */
    store_k,
};

constexpr std::size_t fallback_stage_count_k = 6;

/** Report names in enumerator order, so a printed line and the enum cannot drift. */
constexpr std::array<std::string_view, fallback_stage_count_k> fallback_stage_names_k {"parse", "type",     "embed",
                                                                                       "build", "minimize", "store"};

/** Host nanoseconds spent per stage, indexed by `fallback_stage_t`. */
using fallback_stage_nanoseconds_t = std::array<std::uint64_t, fallback_stage_count_k>;

[[nodiscard]] inline std::uint64_t &nanoseconds_of(fallback_stage_nanoseconds_t &tally,
                                                   fallback_stage_t stage) noexcept {
    return tally[static_cast<std::size_t>(stage)];
}

/** Power-of-two millisecond bins one molecule's whole attempt lands in; the last one is open-ended. */
constexpr std::size_t fallback_duration_bin_count_k = 16;

/** Bin of @p nanoseconds: bin `b` above zero holds `[2^(b-1), 2^b)` milliseconds, bin 0 everything under one. */
[[nodiscard]] inline std::size_t fallback_duration_bin(std::uint64_t nanoseconds) noexcept {
    std::uint64_t const milliseconds = nanoseconds / 1'000'000u;
    if (milliseconds == 0) return 0;
    std::size_t const bin = static_cast<std::size_t>(std::bit_width(milliseconds));
    return bin < fallback_duration_bin_count_k ? bin : fallback_duration_bin_count_k - 1;
}

/** Lower edge in milliseconds of the bin `fallback_duration_bin` returns, so a report labels its own bins. */
[[nodiscard]] inline std::uint64_t fallback_duration_bin_floor_milliseconds(std::size_t bin) noexcept {
    return bin == 0 ? 0 : std::uint64_t {1} << (bin - 1);
}

#pragma endregion Cost Accounting

#pragma region Reporting

/** Counter arrays here are indexed by the enumerator itself, so they take their extent from the enum. */
using types::conformer_status_count_k;
using types::unnamed_status_k;

/** Molecules counted per `conformer_status_t`, with one bucket for a byte no enumerator claims.
 *  `conformer_status_t` is append-only because every value is persisted into a shipped shard that outlives the
 *  binaries reading it, so a status this reader has no name for is expected. `of` routes such a byte into
 *  `unnamed`, where a non-zero count says the reader is older than the shard, rather than past `named`'s end. */
struct status_tally_t {
    /** Counts indexed by the enumerator itself. */
    std::array<std::uint64_t, conformer_status_count_k> named {};
    /** Counts for every status byte outside the enum this binary was built against. */
    std::uint64_t unnamed = 0;

    /** The counter @p status belongs in, which is `unnamed` for a byte no enumerator claims. */
    [[nodiscard]] std::uint64_t &of(types::conformer_status_t status) noexcept {
        std::size_t const index = static_cast<std::size_t>(status);
        return index < conformer_status_count_k ? named[index] : unnamed;
    }

    /** Molecules across every status, named or not. */
    [[nodiscard]] std::uint64_t total() const noexcept {
        std::uint64_t summed = unnamed;
        for (std::uint64_t const count : named) summed += count;
        return summed;
    }
};

/** Per-status account of what one `coprocessed-batch` of refusals cost and bought. Every tally is keyed by the
 *  `conformer_status_t` that decided the molecule's fate and counts @b molecules, not conformer slots.
 *  `skipped` is filled where the dispatch list is built, the rest by `rdkit_fallback`. */
struct fallback_report_t {
    /** Molecules a terminal status kept out of the fallback. */
    status_tally_t skipped {};
    /** Molecules dispatched, by the status that dispatched them. */
    status_tally_t attempted {};
    /** Of `attempted`, those RDKit filled at least one slot for. */
    status_tally_t recovered {};
    /** Host nanoseconds the dispatched molecules cost, by stage, summed over every worker thread. */
    fallback_stage_nanoseconds_t nanoseconds {};
    /** Dispatched molecules by the millisecond bin their attempt fell in, a shape no mean over a tail shows. */
    std::array<std::uint64_t, fallback_duration_bin_count_k> molecules_by_duration {};
    /** The single slowest attempt of the batch, and the molecule that ran it. */
    std::uint64_t slowest_nanoseconds = 0;
    std::uint32_t slowest_molecule_index = 0;
};

#pragma endregion Reporting

struct rdkit_fallback_inputs_t {
    /** Length N (may contain empty strings). */
    custd::span<std::string_view const> smiles;
    /** Subset of [0, N) to re-attempt. */
    custd::span<std::uint32_t const> retry_indices;
    /** Slot stride and conformer count, shared with the GPU producer and the Parquet writer. */
    api::conformer_slot_layout_t layout;
    int num_threads;
    std::uint64_t random_seed;
};

struct rdkit_fallback_outputs_t {
    /** Length N × K (updated in place). */
    custd::span<types::conformer_status_t> status;
    /** Length N (written when discovered). */
    custd::span<std::uint8_t> num_atoms_per_molecule;
    /** Length N × K × max_atoms × 3. */
    custd::span<__half> coordinates;
    /** Length N × K. */
    custd::span<float> energies;
};

/** Run RDKit ETKDG + MMFF on all `retry_indices`, filling @p report's `attempted` and `recovered`
 *  with the molecules each dispatching status brought in and the ones it produced a conformer for. */
void rdkit_fallback(rdkit_fallback_inputs_t const &inputs, rdkit_fallback_outputs_t &outputs,
                    fallback_report_t &report);

} // namespace smallfield::cli
