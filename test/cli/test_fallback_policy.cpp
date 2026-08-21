/**
 *  @file test/cli/test_fallback_policy.cpp
 *  @brief Pin every verdict of `fallback_disposition`, and the ordering rule `fallback_dispatch_cause` runs.
 *
 *  Thirteen statuses each carry an argued verdict on whether a CPU embedding recovers a refused slot, and a
 *  wrong one is invisible in production: a `terminal_k` that should retry drops recoverable molecules, and a
 *  `retry_k` that should be terminal spends ~20 ms of CPU per molecule re-deriving a verdict the slot already
 *  carries. `-Wswitch` catches an appended status; a changed verdict is caught only here.
 *
 *  `fallback_disposition` is `constexpr`, so verdicts pin at compile time and the section passes by compiling
 *  -- no device, no `SMALLFIELD_BUILD_CLI`. Only the rule for which slot decides a molecule's fate runs.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <array>       // `std::array`
#include <string_view> // `std::string_view`

#include <fmt/format.h> // `fmt::println`

#include "cli/rdkit_fallback.hpp"
#include "smallfield/types/conformer_status.hpp"

#include "test/test_helpers.hpp"

namespace sfcli = smallfield::cli;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_fallback_policy";

#pragma region Dispositions

/*  One `static_assert` per enumerator in the enum's own order, so the roster and `conformer_status_t` read
 *  line by line. The count assertion lands an appended status here as well as at the header's `-Wswitch`,
 *  so it cannot inherit a verdict this file never states. */

static_assert(sft::conformer_status_count_k == 15,
              "every `conformer_status_t` carries one pinned verdict below, so a new one is pinned here too");

/* A filled slot is not offered to the fallback, which leaves filled slots alone. */
static_assert(sfcli::fallback_disposition(sft::conformer_status_t::success_k) ==
                  sfcli::fallback_disposition_t::terminal_k,
              "a slot that already holds a conformer buys nothing from an embedding");

/* `admit_molecule_` re-derives it: RDKit ETKDG perceives chirality by its own route. */
static_assert(sfcli::fallback_disposition(sft::conformer_status_t::stereo_failure_k) ==
                  sfcli::fallback_disposition_t::retry_k,
              "a chiral volume the device driver could not satisfy is not a verdict RDKit shares");

static_assert(sfcli::fallback_disposition(sft::conformer_status_t::mmff_unconverged_k) ==
                  sfcli::fallback_disposition_t::retry_k,
              "a different MMFF minimizer from the device FIRE driver may still converge");

static_assert(sfcli::fallback_disposition(sft::conformer_status_t::invalid_smiles_k) ==
                  sfcli::fallback_disposition_t::retry_k,
              "only one of this status's producers is re-derived, so the other is still worth an attempt");

static_assert(sfcli::fallback_disposition(sft::conformer_status_t::too_many_atoms_k) ==
                  sfcli::fallback_disposition_t::terminal_k,
              "`parse_smiles` gates the atom count against the same cap the GPU path passed");

static_assert(sfcli::fallback_disposition(sft::conformer_status_t::preprocess_timeout_k) ==
                  sfcli::fallback_disposition_t::retry_k,
              "nothing produces it, and the fallback carries no deadline that could re-derive it");

static_assert(sfcli::fallback_disposition(sft::conformer_status_t::unsupported_element_k) ==
                  sfcli::fallback_disposition_t::terminal_k,
              "`MMFFMolProperties::isValid` is the same typing check on the same molecule");

static_assert(sfcli::fallback_disposition(sft::conformer_status_t::too_many_bonds_k) ==
                  sfcli::fallback_disposition_t::retry_k,
              "an extent of `molecule_descriptor_t`, which the fallback never builds");

static_assert(sfcli::fallback_disposition(sft::conformer_status_t::too_many_rings_k) ==
                  sfcli::fallback_disposition_t::retry_k,
              "an extent of `molecule_descriptor_t`, which the fallback never builds");

static_assert(sfcli::fallback_disposition(sft::conformer_status_t::bounds_infeasible_k) ==
                  sfcli::fallback_disposition_t::retry_k,
              "`EmbedMultipleConfs` smooths its own bounds and may not agree the matrix is infeasible");

static_assert(sfcli::fallback_disposition(sft::conformer_status_t::nonphysical_energy_k) ==
                  sfcli::fallback_disposition_t::retry_k,
              "RDKit reaches the geometry by its own embedding and its own minimizer");

/*  The one terminal verdict not re-derived: RDKit would embed these, but a mis-sized wave refuses a whole
 *  device's batch, and that population at 20 ms of CPU each reads as a hang, not a re-runnable sizing error. */
static_assert(sfcli::fallback_disposition(sft::conformer_status_t::batch_overflow_k) ==
                  sfcli::fallback_disposition_t::terminal_k,
              "a mis-sized wave parks the shard, and is not laundered one molecule at a time");

static_assert(sfcli::fallback_disposition(sft::conformer_status_t::slot_unwritten_k) ==
                  sfcli::fallback_disposition_t::retry_k,
              "no kernel reached the slot, so there is no device verdict for the fallback to re-derive");

static_assert(sfcli::fallback_disposition(sft::conformer_status_t::polish_inverted_center_k) ==
                  sfcli::fallback_disposition_t::retry_k,
              "RDKit enforces chirality by its own route, so a centre our polish inverted is not its verdict");

static_assert(sfcli::fallback_disposition(sft::conformer_status_t::polish_inverted_bond_k) ==
                  sfcli::fallback_disposition_t::retry_k,
              "the same for a double bond, which RDKit perceives from its own embedding");

static_assert(sfcli::fallback_disposition(static_cast<sft::conformer_status_t>(sft::conformer_status_count_k)) ==
                  sfcli::fallback_disposition_t::retry_k,
              "a status byte from a newer producer keeps its second chance rather than being dropped");

#pragma endregion Dispositions

namespace smallfield::tests {

#pragma region Dispatch Cause

/** Slots the longest probe below carries; a molecule's real slot count is its conformer count. */
constexpr std::size_t max_probe_slots_k = 4;

/** One molecule's slots, and the status the ordering rule must answer with. */
struct dispatch_probe_t {
    std::string_view name;
    std::array<types::conformer_status_t, max_probe_slots_k> slots;
    std::size_t num_slots;
    types::conformer_status_t cause;
};

/** The byte no enumerator claims, which a shard written by a newer producer can carry. */
constexpr types::conformer_status_t unnamed_status_byte_k = static_cast<types::conformer_status_t>(
    types::conformer_status_count_k);

constexpr std::array<dispatch_probe_t, 5> dispatch_probes_k {{
    // The rule: the earliest `retry_k` slot decides, and terminal slots before it are stepped over.
    {"first_retry_wins",
     {types::conformer_status_t::success_k, types::conformer_status_t::too_many_atoms_k,
      types::conformer_status_t::stereo_failure_k, types::conformer_status_t::invalid_smiles_k},
     4,
     types::conformer_status_t::stereo_failure_k},
    // A retry in the last slot is still found, so the scan reaches the end rather than sampling a prefix.
    {"trailing_retry_found",
     {types::conformer_status_t::batch_overflow_k, types::conformer_status_t::too_many_atoms_k,
      types::conformer_status_t::nonphysical_energy_k, types::conformer_status_t::success_k},
     3,
     types::conformer_status_t::nonphysical_energy_k},
    // Every slot terminal answers `success_k`, which is how a molecule stays out of the dispatch list.
    {"all_terminal_is_success",
     {types::conformer_status_t::success_k, types::conformer_status_t::too_many_atoms_k,
      types::conformer_status_t::unsupported_element_k, types::conformer_status_t::batch_overflow_k},
     4,
     types::conformer_status_t::success_k},
    // A molecule with no slots at all takes the same answer rather than reading the span past its end.
    {"empty_span_is_success", {}, 0, types::conformer_status_t::success_k},
    // A byte outside the enum is a retry, and it dispatches under its own byte rather than a neighbour's.
    {"unnamed_byte_dispatches",
     {types::conformer_status_t::success_k, unnamed_status_byte_k},
     2,
     unnamed_status_byte_k},
}};

/** Drive every probe through `fallback_dispatch_cause` and report the byte that came back. */
static void check_dispatch_causes_(diagnostic_reporter_t &reporter) {
    for (dispatch_probe_t const &probe : dispatch_probes_k) {
        sf_verify_(probe.num_slots <= max_probe_slots_k,
                   fmt::format("probe {} claims more slots than it stores", probe.name));
        custd::span<types::conformer_status_t const> const slots {probe.slots.data(), probe.num_slots};
        types::conformer_status_t const observed = cli::fallback_dispatch_cause(slots);
        reporter
            .mismatch("fallback_dispatch_cause", "probe={} slots={} cause={}", probe.name, probe.num_slots,
                      types::conformer_status_name(probe.cause))
            .field("cause_byte", static_cast<std::uint32_t>(probe.cause), static_cast<std::uint32_t>(observed),
                   exactly_equal);
    }
}

#pragma endregion Dispatch Cause

} // namespace smallfield::tests

int main() {
    fmt::println("{}: the thirteen fallback dispositions, and the slot that decides a molecule's dispatch",
                 test_name_k);
    fmt::println("dispositions are `constexpr`, so this binary having linked at all is that section's verdict");

    sftest::diagnostic_reporter_t reporter("policy", "observed");
    sftest::check_dispatch_causes_(reporter);
    reporter.summarize("fallback_dispatch_cause");

    if (!reporter.passed())
        return sftest::report_fail(test_name_k, "{} mismatches over {} comparisons", reporter.mismatch_count(),
                                   reporter.comparison_count());
    return sftest::report_pass(test_name_k);
}
