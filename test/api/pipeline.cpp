/**
 *  @file test/api/pipeline.cpp
 *  @brief End-to-end ETKDG over the SmallField library API: SMILES → `context_t::generate_conformers` →
 *  RDKit cross-validation, plus wave-sizing arithmetic, per-wave timing, and two-context reentrancy.
 */

#include <cmath>   // `std::isfinite`, `std::fabs`, `std::sqrt`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::tmpfile`, `std::sscanf` -- the wave-table capture below reads a `FILE *`

#include <algorithm>   // `std::min`
#include <array>       // `std::array`
#include <limits>      // `std::numeric_limits`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <thread>      // `std::thread`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`

#include <cuda_fp16.h>

#include "smallfield/api/smallfield.hpp"
#include "smallfield/types/views.hpp"

#include "test/helpers.hpp"
#include "test/rdkit_reference.hpp"

namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "smallfield_test_pipeline";

/** Molecules taken from a named corpus when no cap is given; the built-in molecules are always included. */
static constexpr std::size_t default_max_molecules_k = 100;

/** Printed mismatch lines one corpus may spend; the rest are counted and reported as `suppressed`. */
static constexpr std::size_t max_reported_mismatches_k = 20;

namespace smallfield::api {

/** Wave-sizing arithmetic from `src/smallfield/api/smallfield.cu`, not API and so out of the public header:
 *  lifted from a `cudaMemGetInfo` probe so its edge cases assert without a GPU whose free VRAM a test dictates. */
std::size_t wave_budget_bytes(std::size_t free_probe_count, std::size_t smallest_free_per_slot,
                              std::size_t smallest_total_per_slot) noexcept;
std::uint32_t divide_rounding_up(std::uint32_t numerator, std::uint32_t denominator) noexcept;
std::uint32_t reserve_molecules_per_device(std::uint32_t wave_size, std::uint32_t device_count) noexcept;

} // namespace smallfield::api

namespace smallfield::tests {

using api::conformer_status_t;

#pragma region Wave Sizing Arithmetic

/**
 *  @brief The tightest-slot budget must survive a total probe failure.
 *  With no free-VRAM reading the sizer must still budget from the total-VRAM census — recorded outside
 *  the branch a successful free reading guards — not fall to zero and its hardcoded 50'000 default.
 */
static void check_wave_budget_arithmetic(diagnostic_reporter_t &reporter) {
    constexpr std::size_t gigabyte_k = 1024ull * 1024ull * 1024ull;

    /** One `wave_budget_bytes` census and the budget the sizer owes it. */
    struct budget_case_t {
        std::size_t free_probe_count;
        std::size_t free_per_slot_bytes;
        std::size_t total_per_slot_bytes;
        std::size_t expected_budget_bytes;
    };
    constexpr std::array<budget_case_t, 5> cases {{
        // A device answered: 75 % of the tightest free slot leaves peak-allocation headroom, at any count.
        {.free_probe_count = 1,
         .free_per_slot_bytes = 8 * gigabyte_k,
         .total_per_slot_bytes = 80 * gigabyte_k,
         .expected_budget_bytes = 6 * gigabyte_k},
        {.free_probe_count = 4,
         .free_per_slot_bytes = 8 * gigabyte_k,
         .total_per_slot_bytes = 80 * gigabyte_k,
         .expected_budget_bytes = 6 * gigabyte_k},
        // No device answered: half the tightest total slot — not zero, and never the meaningless free one.
        {.free_probe_count = 0,
         .free_per_slot_bytes = 0,
         .total_per_slot_bytes = 80 * gigabyte_k,
         .expected_budget_bytes = 40 * gigabyte_k},
        {.free_probe_count = 0,
         .free_per_slot_bytes = 8 * gigabyte_k,
         .total_per_slot_bytes = 80 * gigabyte_k,
         .expected_budget_bytes = 40 * gigabyte_k},
        // Nothing known about the hardware at all: zero, which is what drops the sizer onto its default.
        {.free_probe_count = 0, .free_per_slot_bytes = 0, .total_per_slot_bytes = 0, .expected_budget_bytes = 0},
    }};

    for (auto const &budget_case : cases)
        reporter
            .mismatch("wave_budget", "free_probes={} free_per_slot={} total_per_slot={}", budget_case.free_probe_count,
                      budget_case.free_per_slot_bytes, budget_case.total_per_slot_bytes)
            .field("budget_bytes", budget_case.expected_budget_bytes,
                   api::wave_budget_bytes(budget_case.free_probe_count, budget_case.free_per_slot_bytes,
                                          budget_case.total_per_slot_bytes),
                   exactly_equal);
    reporter.summarize("wave_budget");
}

/**
 *  @brief The per-device reservation must reach its answer by arithmetic, not by two wraps.
 *  The quotient is checked too: a zero-device case underflowing to `0xFFFFFFFF`, dividing by one and
 *  wrapping back to zero lands on the right final number for the wrong reason, and only that exposes it.
 */
static void check_per_device_reservation(diagnostic_reporter_t &reporter) {
    /** One wave spread over one device count, with both numbers the split owes it. */
    struct division_case_t {
        std::uint32_t wave_size;
        std::uint32_t device_count;
        std::uint32_t expected_quotient;
        std::uint32_t expected_reservation;
    };
    constexpr std::array<division_case_t, 5> cases {{
        // No devices and no wave: nothing to divide, and the floor is one slot.
        {.wave_size = 0u, .device_count = 0u, .expected_quotient = 0u, .expected_reservation = 1u},
        // One device, empty wave: the slack molecule alone.
        {.wave_size = 0u, .device_count = 1u, .expected_quotient = 0u, .expected_reservation = 1u},
        // No devices to spread a molecule over.
        {.wave_size = 1u, .device_count = 0u, .expected_quotient = 0u, .expected_reservation = 1u},
        // The ordinary case: an exact split plus slack.
        {.wave_size = 50'000u, .device_count = 8u, .expected_quotient = 6'250u, .expected_reservation = 6'251u},
        // Rounding up is what the reservation rests on: an uneven split may not leave a device short.
        {.wave_size = 50'001u, .device_count = 8u, .expected_quotient = 6'251u, .expected_reservation = 6'252u},
    }};

    for (auto const &division_case : cases)
        reporter
            .mismatch("wave_sizing", "wave_size={} device_count={}", division_case.wave_size,
                      division_case.device_count)
            .field("quotient", division_case.expected_quotient,
                   api::divide_rounding_up(division_case.wave_size, division_case.device_count), exactly_equal)
            .field("reservation", division_case.expected_reservation,
                   api::reserve_molecules_per_device(division_case.wave_size, division_case.device_count),
                   exactly_equal);
    reporter.summarize("wave_sizing");
}

#pragma endregion Wave Sizing Arithmetic

#pragma region Wave Timeline Capture

/** One row of the verbose per-wave timing table `print_wave_timeline_` emits. */
struct wave_row_t {
    std::uint32_t wave_index;
    std::uint32_t molecule_count;
    float gpu_average_ms;
    float gpu_max_ms;
};

/**
 *  @brief Parse the verbose wave table out of captured @p log_text, ignoring every other line; one entry
 *  per parsed row, in the order the table printed them.
 *
 *  A contract with `print_wave_timeline_` in `src/smallfield/api/smallfield.cu`: a logger line opens with a
 *  `[wall | +elapsed] ` prefix the payload starts after, and a row is two counts, four millisecond columns,
 *  then a literal ` / ` before the per-wave maximum. Change either and rows stop parsing silently — an
 *  empty timeline, not a failure.
 */
static std::vector<wave_row_t> parse_wave_rows(std::string const &log_text) {
    std::vector<wave_row_t> rows;
    for (std::size_t line_start = 0; line_start < log_text.size();) {
        std::size_t line_end = log_text.find('\n', line_start);
        if (line_end == std::string::npos) line_end = log_text.size();
        std::string const line = log_text.substr(line_start, line_end - line_start);
        line_start = line_end + 1;

        // Every logger line carries a `[wall | +elapsed] ` prefix before the payload.
        std::size_t const payload_start = line.find("] ");
        if (payload_start == std::string::npos) continue;
        std::string const payload = line.substr(payload_start + 2);

        wave_row_t row {};
        float prep_ms = 0.f, launch_ms = 0.f, finalize_ms = 0.f;
        int const fields = std::sscanf(payload.c_str(), "%u %u %f %f %f %f / %f", &row.wave_index, &row.molecule_count,
                                       &prep_ms, &launch_ms, &finalize_ms, &row.gpu_average_ms, &row.gpu_max_ms);
        if (fields == 7) rows.push_back(row);
    }
    return rows;
}

/** Every per-conformer array `generate_conformers` fills, owned together so no loop below allocates. */
struct conformer_outputs_t {
    std::vector<__half> coordinates;
    std::vector<float> energies;
    std::vector<conformer_status_t> status;
    std::vector<std::uint8_t> num_atoms;

    /** Size every array for @p num_molecules molecules addressed through @p layout, the slot
     *  addressing the pool and its readers agree on. */
    void reset_for(std::uint32_t num_molecules, api::conformer_slot_layout_t const &layout) {
        std::size_t const slots = static_cast<std::size_t>(num_molecules) * layout.num_conformers;
        coordinates.assign(slots * layout.slot_stride(), __half {});
        energies.assign(slots, 0.f);
        // Poisoned rather than zeroed, so a slot the pipeline never wrote cannot read as `success_k`.
        status.assign(slots, conformer_status_t::invalid_smiles_k);
        num_atoms.assign(num_molecules, std::uint8_t {0});
    }
};

/**
 *  @brief Generate one batch of @p smiles under @p config, capped at @p max_atoms, into the already-sized
 *  @p outputs, with the verbose timeline captured into @p log_text and the batch-level status returned.
 *
 *  @p context has its logger redirected for the duration of the call. The logger writes to a `FILE *`, so
 *  capture is a `std::tmpfile`; stream and verbosity are restored before returning.
 */
static conformer_status_t generate_capturing_timeline_(api::context_t &context,
                                                       custd::span<std::string_view const> smiles,
                                                       api::config_t const &config, std::uint32_t max_atoms,
                                                       conformer_outputs_t &outputs, std::string &log_text) {
    std::FILE *const capture = std::tmpfile();
    sf_verify_(capture != nullptr, "could not open a capture stream for the wave timeline");
    context.logger.verbose_level = 1;
    context.logger.color_enabled = false;
    context.logger.info_stream = capture;
    conformer_status_t const batch_status = context.generate_conformers(
        smiles, config, max_atoms, outputs.coordinates, outputs.energies, outputs.status, outputs.num_atoms);
    context.logger.info_stream = stdout;
    context.logger.verbose_level = 0;

    log_text.clear();
    std::rewind(capture);
    for (char buffer[4096];;) {
        std::size_t const read_bytes = std::fread(buffer, 1, sizeof(buffer), capture);
        if (read_bytes == 0) break;
        log_text.append(buffer, read_bytes);
    }
    std::fclose(capture);
    return batch_status;
}

/**
 *  @brief A wave whose molecules preprocess rejected wholesale must report no GPU time.
 *
 *  Three waves of two with the element-gate rejects last, since input sorts by descending SMILES length.
 *  The pipelined path rings the same two timing vectors over 2 slots, so wave 2 reads the slot wave 0
 *  wrote, and a launch returning before it resets that slot republishes wave 0's numbers.
 */
static void check_empty_wave_timings(diagnostic_reporter_t &reporter) {
    std::vector<std::string_view> const smiles = {
        "CC(=O)Oc1ccccc1C(=O)O", // 21 chars
        "c1ccccc1",              // 8
        "CCOCCO",                // 6
        "CCCCC",                 // 5
        "[Fe]",                  // 4 — no MMFF94 atom type, rejected by the element gate
        "[La]",                  // 4 — likewise
    };
    // Bounds of the two rejects above, a property of this fixed corpus rather than a cap on the sweep.
    constexpr std::uint32_t first_rejected_molecule_k = 4, past_last_rejected_molecule_k = 6;

    constexpr std::uint32_t num_conformers_k = 2;
    api::conformer_slot_layout_t const layout {128u, num_conformers_k};

    api::config_t config;
    config.num_conformers = num_conformers_k;
    config.wave_size = 2; // three waves over six molecules, and the pipelined path with it
    config.random_seed = 7;

    conformer_outputs_t outputs;
    outputs.reset_for(static_cast<std::uint32_t>(smiles.size()), layout);

    int const device_id = 0;
    api::context_t context(custd::span<int const>(&device_id, 1));
    std::string log_text;
    [[maybe_unused]] conformer_status_t const batch_status = generate_capturing_timeline_(
        context, smiles, config, layout.max_atoms, outputs, log_text);

    std::vector<wave_row_t> const rows = parse_wave_rows(log_text);
    sf_verify_(rows.size() == 3, "expected three wave rows in the verbose timeline");
    // Real GPU time in waves 0 and 1 keeps the empty-wave comparison below from passing vacuously.
    sf_verify_(rows[0].gpu_max_ms > 0.f, "wave 0 must record real GPU time, or the check below is vacuous");
    sf_verify_(rows[1].gpu_max_ms > 0.f, "wave 1 must record real GPU time");

    // The rejected molecules never reach a GPU pool, so the last wave is empty by construction.
    reporter.mismatch("empty_wave", "wave={}", rows[2].wave_index)
        .field("molecule_count", 2u, rows[2].molecule_count, exactly_equal)
        .field("gpu_max_ms", 0.f, rows[2].gpu_max_ms, exactly_equal)
        .field("gpu_average_ms", 0.f, rows[2].gpu_average_ms, exactly_equal);
    reporter.summarize("empty_wave");

    for (std::uint32_t molecule = first_rejected_molecule_k; molecule < past_last_rejected_molecule_k; ++molecule)
        for (std::uint32_t conformer = 0; conformer < num_conformers_k; ++conformer)
            reporter
                .mismatch("element_gate", "molecule={} conformer={} smiles={}", molecule, conformer, smiles[molecule])
                .field("status", static_cast<unsigned>(conformer_status_t::unsupported_element_k),
                       static_cast<unsigned>(outputs.status[molecule * num_conformers_k + conformer]), exactly_equal);
    reporter.summarize("element_gate");

    fmt::println("  empty-wave timings: {} rows parsed, wave0 gpu_max={:.1f} ms, wave2 gpu_max={:.1f} ms (must be 0)",
                 rows.size(), rows[0].gpu_max_ms, rows[2].gpu_max_ms);
}

#pragma endregion Wave Timeline Capture

#pragma region Concurrent Contexts

/** Bytes differing between runs @p first and @p second of one slice, summed over the four arrays
 *  `generate_conformers` fills and sized for the same slice and layout. Raw storage, not numbers: a
 *  corrupted coordinate is still finite, so an approximate compare passes it. */
static std::size_t differing_bytes_(conformer_outputs_t const &first, conformer_outputs_t const &second) noexcept {
    auto const count_over = [](void const *left, void const *right, std::size_t bytes) {
        auto const *left_bytes = static_cast<std::uint8_t const *>(left);
        auto const *right_bytes = static_cast<std::uint8_t const *>(right);
        std::size_t differing = 0;
        for (std::size_t offset = 0; offset < bytes; ++offset) differing += left_bytes[offset] != right_bytes[offset];
        return differing;
    };
    return count_over(first.coordinates.data(), second.coordinates.data(), first.coordinates.size() * sizeof(__half)) +
           count_over(first.energies.data(), second.energies.data(), first.energies.size() * sizeof(float)) +
           count_over(first.status.data(), second.status.data(), first.status.size() * sizeof(conformer_status_t)) +
           count_over(first.num_atoms.data(), second.num_atoms.data(), first.num_atoms.size());
}

/**
 *  @brief Two contexts generating on two threads must answer what the same two slices answer alone.
 *
 *  `prepare_per_device_batches_into` keeps its per-worker contributions and partition buffers caller-owned:
 *  in file-scope statics the two threads share one set, each clearing what the other is mid-way through
 *  filling and packing some of the other's molecules into its own pinned pools, and nothing warns because
 *  the corruption reads as ordinary geometry. Byte equality against the serial baseline is the whole
 *  measurable effect. Both baselines run on their own fresh context, so the only difference between the
 *  two phases is that the second pair overlaps in time.
 */
static void check_concurrent_contexts(diagnostic_reporter_t &reporter) {
    constexpr std::uint32_t num_conformers_k = 2;
    api::conformer_slot_layout_t const layout {128u, num_conformers_k};

    api::config_t config;
    config.num_conformers = num_conformers_k;
    config.random_seed = 11;

    std::vector<std::string> const storage = standard_test_smiles();
    std::vector<std::string_view> const corpus(storage.begin(), storage.end());
    std::size_t const split = corpus.size() / 2;
    sf_verify_(split > 0, "the built-in corpus must split into two non-empty slices");
    custd::span<std::string_view const> const first_slice {corpus.data(), split};
    custd::span<std::string_view const> const second_slice {corpus.data() + split, corpus.size() - split};

    int const device_id = 0;
    auto const generate = [&layout, &config](custd::span<std::string_view const> slice, api::context_t &context,
                                             conformer_outputs_t &outputs) {
        outputs.reset_for(static_cast<std::uint32_t>(slice.size()), layout);
        [[maybe_unused]] conformer_status_t const batch_status = context.generate_conformers(
            slice, config, layout.max_atoms, outputs.coordinates, outputs.energies, outputs.status, outputs.num_atoms);
    };

    conformer_outputs_t first_alone, second_alone, first_together, second_together;
    {
        api::context_t context(custd::span<int const>(&device_id, 1));
        generate(first_slice, context, first_alone);
    }
    {
        api::context_t context(custd::span<int const>(&device_id, 1));
        generate(second_slice, context, second_alone);
    }

    // Both contexts are built here and driven from the threads below: same VRAM census, no construction race.
    {
        api::context_t first_context(custd::span<int const>(&device_id, 1));
        api::context_t second_context(custd::span<int const>(&device_id, 1));
        std::thread first_thread {[&] { generate(first_slice, first_context, first_together); }};
        std::thread second_thread {[&] { generate(second_slice, second_context, second_together); }};
        first_thread.join();
        second_thread.join();
    }

    // A baseline of nothing but refusals would match a corrupted run byte for byte.
    std::size_t succeeded = 0;
    for (conformer_status_t const status : first_alone.status) succeeded += status == conformer_status_t::success_k;
    for (conformer_status_t const status : second_alone.status) succeeded += status == conformer_status_t::success_k;
    sf_verify_(succeeded > 0, "the serial baseline produced no conformer, so byte equality is vacuous");

    reporter.mismatch("reentrancy", "slice=first molecules={}", first_slice.size())
        .field("differing_bytes", std::size_t {0}, differing_bytes_(first_alone, first_together), exactly_equal);
    reporter.mismatch("reentrancy", "slice=second molecules={}", second_slice.size())
        .field("differing_bytes", std::size_t {0}, differing_bytes_(second_alone, second_together), exactly_equal);
    reporter.summarize("reentrancy");

    fmt::println("  two contexts on two threads: {} + {} molecules, {} successful slots in the serial baseline",
                 first_slice.size(), second_slice.size(), succeeded);
}

#pragma endregion Concurrent Contexts

#pragma region RDKit Cross Validation

/** Beyond this the two MMFF94 implementations are counted as diverging; nothing gates on the count. */
static constexpr double energy_agreement_kcal_k = 0.012;

/**
 *  @brief Below this an atom pair is closer than any real bond, so the geometry carrying it is not one.
 *
 *  MMFF94's shortest parameterised bond is around 0.95 Å, so a conformer with a pair under half that
 *  is not the geometry its own energy came from. Counted, not gated: those divergent energies come off it.
 */
static constexpr double clash_distance_angstrom_k = 0.5;

/**
 *  @brief Smallest interatomic distance in angstroms over an atom-major interleaved `(atom, 3)` block.
 *  @param[in] num_atoms Atoms the block carries; fewer than two has no pair and answers infinity.
 */
static double closest_atom_pair_(float const *interleaved, std::uint32_t num_atoms) noexcept {
    double closest = std::numeric_limits<double>::infinity();
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
        for (std::uint32_t neighbor = atom + 1; neighbor < num_atoms; ++neighbor) {
            double squared = 0.0;
            for (std::uint32_t axis = 0; axis < 3u; ++axis) {
                double const difference = interleaved[atom * 3u + axis] - interleaved[neighbor * 3u + axis];
                squared += difference * difference;
            }
            closest = std::min(closest, std::sqrt(squared));
        }
    return closest;
}

/** What one cross-validation pass counted, threaded by reference so the pass owns no module state.
 *  `energies_nonphysical` is gated at zero — `finalize_gpu_pipeline` demotes any successful slot whose
 *  energy no consumer can rank by. The `reference_energies_*` pair is RDKit's energy at our coordinates,
 *  which is the reference rather than our own output. */
struct cross_validation_tally_t {
    std::uint32_t molecules_compared = 0;
    std::uint32_t molecules_unparsed = 0;
    std::uint32_t molecules_multi_fragment = 0;
    std::uint32_t molecules_rejected = 0;
    std::uint32_t conformers_attempted = 0;
    std::uint32_t conformers_succeeded = 0;
    std::uint32_t energies_nonphysical = 0;
    std::uint32_t reference_energies_finite = 0;
    std::uint32_t reference_energies_reasonable = 0;
    std::uint32_t energies_diverged = 0;
    std::uint32_t conformers_clashing = 0;
    double worst_energy_delta = 0.0;
    std::uint32_t worst_molecule = 0;
    std::uint32_t worst_conformer = 0;
    double closest_atom_pair = std::numeric_limits<double>::infinity();
    std::uint32_t closest_molecule = 0;
};

/**
 *  @brief Generate conformers for @p corpus — built-ins plus whatever corpus was named — and read every
 *  successful one back through RDKit, counting into @p tally. @p checks owns the atom-count comparison
 *  section and @p energies narrates the reference energy beside ours.
 *
 *  Two gates: the post-preprocess atom count must equal RDKit's own with hydrogens added, which is what
 *  makes the gather address this molecule's atoms rather than a neighbouring slot's padding; and no slot
 *  called a success may carry a non-finite energy, which is what the status byte promises. Two narrated
 *  and counted rather than gated: RDKit's MMFF94 energy at our own coordinates, the two agreeing to about
 *  0.01 kcal/mol, and atom pairs closer than any bond, which is what a divergent energy usually is.
 */
static void check_generation_against_rdkit(custd::span<std::string_view const> corpus, diagnostic_reporter_t &checks,
                                           diagnostic_reporter_t &energies, cross_validation_tally_t &tally) {
    constexpr std::uint32_t num_conformers_k = 5;
    api::conformer_slot_layout_t const layout {128u, num_conformers_k};

    api::config_t config;
    config.num_conformers = num_conformers_k;
    config.max_stereo_attempts = 20;
    config.random_seed = 42;

    auto const num_molecules = static_cast<std::uint32_t>(corpus.size());
    conformer_outputs_t outputs;
    outputs.reset_for(num_molecules, layout);

    // One GPU: the arithmetic that splits a wave across several is checked directly above.
    int const device_id = 0;
    api::context_t context(custd::span<int const>(&device_id, 1));
    conformer_status_t const batch_status = context.generate_conformers(
        corpus, config, layout.max_atoms, outputs.coordinates, outputs.energies, outputs.status, outputs.num_atoms);
    fmt::println("  batch status: {}", static_cast<unsigned>(batch_status));

    // Scratch the molecule loop reuses: both buffers grow to the widest molecule once and stay there.
    std::vector<float> interleaved;
    std::vector<int> fragment_mapping;

    for (std::uint32_t molecule = 0; molecule < num_molecules; ++molecule) {
        auto rdkit_molecule = mol_from_smiles(std::string(corpus[molecule]));
        if (!rdkit_molecule) { // RDKit refused the SMILES, so there is no reference to compare against
            ++tally.molecules_unparsed;
            continue;
        }
        if (RDKit::MolOps::getMolFrags(*rdkit_molecule, fragment_mapping) != 1) {
            // Preprocess keeps the largest fragment only: a different molecule, so nothing reads back.
            ++tally.molecules_multi_fragment;
            continue;
        }

        if (outputs.num_atoms[molecule] == 0) {
            // Preprocess refused it — too many atoms, or an untypeable element — in every conformer status.
            fmt::println("  no geometry molecule={} status={} smiles={}", molecule,
                         static_cast<unsigned>(outputs.status[static_cast<std::size_t>(molecule) * num_conformers_k]),
                         corpus[molecule]);
            ++tally.molecules_rejected;
            continue;
        }

        std::uint32_t const num_atoms = rdkit_molecule->getNumAtoms();
        checks.mismatch("atom_count", "molecule={} smiles={}", molecule, corpus[molecule])
            .field("num_atoms", num_atoms, static_cast<std::uint32_t>(outputs.num_atoms[molecule]), exactly_equal);
        if (outputs.num_atoms[molecule] != num_atoms) continue; // gathering past our own count reads padding
        ++tally.molecules_compared;

        attach_empty_conformer(*rdkit_molecule); // the slot `set_coords_interleaved` overwrites
        interleaved.resize(static_cast<std::size_t>(num_atoms) * 3u);

        for (std::uint32_t conformer = 0; conformer < num_conformers_k; ++conformer) {
            ++tally.conformers_attempted;
            std::size_t const slot = static_cast<std::size_t>(molecule) * num_conformers_k + conformer;
            if (outputs.status[slot] != conformer_status_t::success_k) {
                fmt::println("  no conformer molecule={} conformer={} status={}", molecule, conformer,
                             static_cast<unsigned>(outputs.status[slot]));
                continue;
            }
            ++tally.conformers_succeeded;

            api::gather_slot_into_interleaved(outputs.coordinates.data() + layout.slot_offset(molecule, conformer),
                                              layout, num_atoms, interleaved.data());
            double const closest_pair = closest_atom_pair_(interleaved.data(), num_atoms);
            if (closest_pair < clash_distance_angstrom_k) ++tally.conformers_clashing;
            if (closest_pair < tally.closest_atom_pair) {
                tally.closest_atom_pair = closest_pair;
                tally.closest_molecule = molecule;
            }

            // Our own energy, through the predicate the producer set this status with. A non-finite one
            // is invisible below — `fabs(NaN) > tolerance` is false — so it needs this tally of its own.
            if (!types::energy_is_physical(outputs.energies[slot])) ++tally.energies_nonphysical;

            set_coords_interleaved(*rdkit_molecule, interleaved.data());
            double const rdkit_energy = rdkit_mmff94_energy(*rdkit_molecule);
            if (std::isfinite(rdkit_energy)) {
                ++tally.reference_energies_finite;
                if (rdkit_energy < 500.0) ++tally.reference_energies_reasonable;
            }

            double const delta = static_cast<double>(outputs.energies[slot]) - rdkit_energy;
            if (std::fabs(delta) > energy_agreement_kcal_k) ++tally.energies_diverged;
            if (std::fabs(delta) > std::fabs(tally.worst_energy_delta)) {
                tally.worst_energy_delta = delta;
                tally.worst_molecule = molecule;
                tally.worst_conformer = conformer;
            }
            if (conformer == 0)
                energies
                    .note("conformer_energy", "molecule={} conformer={} smiles={}", molecule, conformer,
                          corpus[molecule])
                    .field("mmff94_energy_kcal_per_mol", rdkit_energy, outputs.energies[slot]);
        }
    }
    checks.summarize("atom_count");
}

/** Print what the cross-validation counted, divergence included, since nothing gates on that one. */
static void print_cross_validation_summary(cross_validation_tally_t const &tally, double success_rate) {
    fmt::println("\n  summary:");
    fmt::println("    molecules compared:    {} ({} unparsed, {} multi-fragment, {} rejected by preprocess)",
                 fmt::group_digits(tally.molecules_compared), fmt::group_digits(tally.molecules_unparsed),
                 fmt::group_digits(tally.molecules_multi_fragment), fmt::group_digits(tally.molecules_rejected));
    fmt::println("    conformers attempted:  {}", fmt::group_digits(tally.conformers_attempted));
    fmt::println("    conformers succeeded:  {} ({:.3f}%)", fmt::group_digits(tally.conformers_succeeded),
                 success_rate);
    fmt::println("    nonphysical energy:    {} of our own successful slots", //
                 fmt::group_digits(tally.energies_nonphysical));
    fmt::println(
        "    finite RDKit energy:   {} ({:.3f}% of successes)", fmt::group_digits(tally.reference_energies_finite),
        tally.conformers_succeeded > 0 ? 100.0 * tally.reference_energies_finite / tally.conformers_succeeded : 0.0);
    fmt::println("    reasonable RDKit:      {} (< 500 kcal/mol)",
                 fmt::group_digits(tally.reference_energies_reasonable));
    fmt::println("    energy divergence:     {} conformers beyond {:.3g} kcal/mol, " //
                 "worst {:+.3g} at molecule={} conformer={}",
                 fmt::group_digits(tally.energies_diverged), energy_agreement_kcal_k, tally.worst_energy_delta,
                 tally.worst_molecule, tally.worst_conformer);
    fmt::println("    clashing geometry:     {} conformers under {:.3g} A, closest pair {:.3g} A at molecule={}",
                 fmt::group_digits(tally.conformers_clashing), clash_distance_angstrom_k, tally.closest_atom_pair,
                 tally.closest_molecule);
}

#pragma endregion RDKit Cross Validation

} // namespace smallfield::tests

int main(int argc, char **argv) {
    fmt::println("{}: wave sizing, empty-wave timings, two-context reentrancy, and end-to-end ETKDG generation",
                 test_name_k);

    auto const selection = sftest::select_corpus(argc, argv, default_max_molecules_k);
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        std::fflush(stdout);
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", test_name_k);
        return 2;
    }

    sftest::diagnostic_reporter_t checks {"expected", "observed"};
    checks.limit_printed_lines(max_reported_mismatches_k);
    sftest::check_wave_budget_arithmetic(checks);
    sftest::check_per_device_reservation(checks);
    sftest::check_empty_wave_timings(checks);
    sftest::check_concurrent_contexts(checks);

    std::vector<std::string> corpus_storage = sftest::standard_test_smiles();
    if (selection.knob == sftest::corpus_knob_t::named_k) {
        std::vector<std::string> streamed;
        auto const outcome = sftest::load_smiles_into(selection.path, selection.max_molecules, streamed);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), selection.path);
        fmt::println("  corpus: {} ({} molecules) plus the built-ins", selection.path,
                     fmt::group_digits(streamed.size()));
        for (auto &smiles : streamed) corpus_storage.push_back(std::move(smiles));
    }
    else fmt::println("  corpus: built-ins only; name one on the command line or in SMALLFIELD_TEST_CORPUS");
    std::vector<std::string_view> const corpus(corpus_storage.begin(), corpus_storage.end());

    sftest::diagnostic_reporter_t energies {"rdkit", "smallfield"};
    sftest::cross_validation_tally_t tally;
    sftest::check_generation_against_rdkit(corpus, checks, energies, tally);

    double const success_rate = tally.conformers_attempted > 0
                                    ? 100.0 * tally.conformers_succeeded / tally.conformers_attempted
                                    : 0.0;
    sftest::print_cross_validation_summary(tally, success_rate);

    if (!checks.passed())
        return sftest::report_fail(test_name_k, "{} mismatches over {} comparisons", checks.mismatch_count(),
                                   checks.comparison_count());
    if (tally.conformers_succeeded == 0)
        return sftest::report_fail(test_name_k, "no conformer succeeded over {} attempts", tally.conformers_attempted);
    if (tally.energies_nonphysical > 0)
        return sftest::report_fail(test_name_k, "{} successful slots carry a nonphysical energy",
                                   tally.energies_nonphysical);
    if (success_rate < 50.0)
        return sftest::report_fail(test_name_k, "success rate {:.3f}% is below the 50% floor", success_rate);
    return sftest::report_pass(test_name_k);
}
