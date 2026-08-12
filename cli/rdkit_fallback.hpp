/**
 *  @file cli/rdkit_fallback.hpp
 *  @brief Declares `rdkit_fallback`, RDKit-based conformer generation for molecules the GPU pipeline
 *  failed on.
 *
 *  Each recovered conformer overwrites its coordinate and energy slot and flips that slot's status to
 *  `success_k`; a slot the fallback could not fill keeps the status the GPU left, which the writer emits as
 *  a null row.
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <array>       // `std::array`
#include <string_view> // `std::string_view`

#include <cuda_fp16.h>

#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/api/conformer_status.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::cli {

#pragma region Dispatch Policy

/** Whether the `fallback-role` can do anything with a slot the `gpu-runner-role` refused. */
enum class fallback_disposition_t : std::uint8_t {
    /** RDKit embeds by a different route and may fill the slot. */
    retry_k,
    /** `admit_molecule_` re-derives this refusal from the same caps and the same parser, so
     *  dispatching it buys one `SmilesToMol` and the verdict the slot already carries. */
    terminal_k,
};

/**
 *  @brief Whether dispatching a slot at @p status to the `fallback-role` could ever fill it.
 *
 *  Exhaustive over `conformer_status_t` and carrying no `default:`, so a status appended to that
 *  append-only enum is a `-Wswitch` warning here rather than a policy inherited in silence.
 *
 *  Each verdict follows from what `admit_molecule_` runs: `parse_smiles` under the `max_atoms` and
 *  the salt policy the GPU path used, then `MMFF::MMFFMolProperties::isValid`. It builds no
 *  `molecule_descriptor_t`, so no extent of that descriptor is re-derivable.
 */
[[nodiscard]] constexpr fallback_disposition_t fallback_disposition(api::conformer_status_t status) noexcept {
    switch (status) {

    /* Terminal: the fallback reaches the same answer by the same route. */
    // The slot already holds a conformer, and the fallback leaves a filled slot alone.
    case api::conformer_status_t::success_k: return fallback_disposition_t::terminal_k;
    // `parse_smiles` gates the atom count against the cap the `gpu-runner-role` passed.
    case api::conformer_status_t::too_many_atoms_k: return fallback_disposition_t::terminal_k;
    // `MMFFMolProperties::isValid` is the same typing check on the same molecule.
    case api::conformer_status_t::unsupported_element_k: return fallback_disposition_t::terminal_k;

    /* Retry: nothing in `admit_molecule_` reproduces these. */
    // ETKDG perceives and enforces chirality by its own route, so a chiral volume the device driver
    // could not satisfy is not a verdict RDKit shares.
    case api::conformer_status_t::stereo_failure_k: return fallback_disposition_t::retry_k;
    // A different MMFF minimizer from the device FIRE driver, and the status the kernels also leave
    // on a slot they never reached.
    case api::conformer_status_t::mmff_unconverged_k: return fallback_disposition_t::retry_k;
    // Two producers and only one is re-derived: `parse_smiles` refusing the string, which the
    // fallback repeats, and `prepare_molecule_topology_into` catching an exception out of descriptor,
    // bounds, chirality or MMFF typing construction — none of which the fallback performs, and whose
    // molecules parse and type cleanly.
    case api::conformer_status_t::invalid_smiles_k: return fallback_disposition_t::retry_k;
    // Nothing produces it, and the fallback carries no deadline that could re-derive it.
    case api::conformer_status_t::preprocess_timeout_k: return fallback_disposition_t::retry_k;
    // Both are extents of `molecule_descriptor_t`, which the fallback never builds; RDKit embeds a
    // molecule wider than the flat descriptor without difficulty.
    case api::conformer_status_t::too_many_bonds_k: return fallback_disposition_t::retry_k;
    case api::conformer_status_t::too_many_rings_k: return fallback_disposition_t::retry_k;
    // `EmbedMultipleConfs` smooths its own bounds by a different route and may not agree the matrix
    // is infeasible. Whether it ever does is what the per-status recovery tally measures.
    case api::conformer_status_t::bounds_infeasible_k: return fallback_disposition_t::retry_k;
    }
    // A status byte outside the enum keeps its second chance rather than being dropped.
    return fallback_disposition_t::retry_k;
}

/** The status that sends one molecule's slots to the `fallback-role`, or `success_k` when none does. */
[[nodiscard]] inline api::conformer_status_t fallback_dispatch_cause(
    custd::span<api::conformer_status_t const> slots) noexcept {
    for (api::conformer_status_t const slot_status : slots)
        if (fallback_disposition(slot_status) == fallback_disposition_t::retry_k) return slot_status;
    return api::conformer_status_t::success_k;
}

#pragma endregion Dispatch Policy

#pragma region Reporting

/** One past the highest `conformer_status_t`, so a counter array is indexed by the enumerator itself.
 *  The enum is append-only, so its newest value is always its widest. */
constexpr std::size_t conformer_status_count_k =
    static_cast<std::size_t>(api::conformer_status_t::bounds_infeasible_k) + 1u;

/**
 *  @brief Per-status account of what one `coprocessed-batch` of refusals cost and bought.
 *
 *  Every array is indexed by the `conformer_status_t` that decided the molecule's fate, and every
 *  entry counts @b molecules rather than conformer slots. `skipped` is filled where the dispatch list
 *  is built, `attempted` and `recovered` by `rdkit_fallback` itself.
 */
struct fallback_report_t {
    /** Molecules a terminal status kept out of the fallback. */
    std::array<std::uint64_t, conformer_status_count_k> skipped {};
    /** Molecules dispatched, by the status that dispatched them. */
    std::array<std::uint64_t, conformer_status_count_k> attempted {};
    /** Of `attempted`, those RDKit filled at least one slot for. */
    std::array<std::uint64_t, conformer_status_count_k> recovered {};
};

/** Molecules across every status of one of a report's arrays. */
[[nodiscard]] inline std::uint64_t total_of(
    std::array<std::uint64_t, conformer_status_count_k> const &counts) noexcept {
    std::uint64_t total = 0;
    for (std::uint64_t const count : counts) total += count;
    return total;
}

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
    custd::span<api::conformer_status_t> status;
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
