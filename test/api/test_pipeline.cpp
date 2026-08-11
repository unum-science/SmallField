/**
 *  @file test/api/test_pipeline.cpp
 *  @brief Test the end-to-end ETKDG pipeline using the SmallField library API.
 *
 *  Tests: SMILES → context_t::generate_conformers → RDKit cross-validation, plus the wave-sizing
 *  arithmetic and the per-wave timing bookkeeping that surround it.
 */

#include <cmath>   // `std::isfinite`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `printf`, `std::tmpfile`

#include <array>       // `std::array`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <cuda_fp16.h>

#include "smallfield/api/smallfield.hpp"

#include "test/test_rdkit_reference_helpers.hpp"

namespace sfapi = smallfield::api;
namespace sftest = smallfield::tests;

namespace smallfield::api {

/*  Wave-sizing arithmetic defined in `src/smallfield/api/smallfield.cu`. Declared here rather than
 *  in the public header because none of it is API: the three functions are lifted out of a
 *  `cudaMemGetInfo` probe precisely so their edge cases can be asserted without a GPU whose free
 *  VRAM a test could dictate. */
std::size_t wave_budget_bytes(std::size_t free_probe_count, std::size_t smallest_free_per_slot,
                              std::size_t smallest_total_per_slot) noexcept;
std::uint32_t divide_rounding_up(std::uint32_t numerator, std::uint32_t denominator) noexcept;
std::uint32_t reserve_molecules_per_device(std::uint32_t wave_size, std::uint32_t device_count) noexcept;

} // namespace smallfield::api

namespace smallfield::tests {

/** One row of the verbose per-wave timing table `print_wave_timeline_` emits. */
struct wave_row_t {
    std::uint32_t wave_index;
    std::uint32_t molecule_count;
    float gpu_average_ms;
    float gpu_max_ms;
};

/**
 *  @brief The tightest-slot budget must survive a total probe failure.
 *
 *  With no free-VRAM reading at all the sizer used to fall through to zero and then to its
 *  hardcoded 50'000 default, because the only assignment to the total-VRAM census sat inside the
 *  branch that a successful free reading guards.
 */
static void check_wave_budget_arithmetic() {
    constexpr std::size_t gigabyte = 1024ull * 1024ull * 1024ull;

    // At least one device answered: 75 % of the tightest free slot, headroom for peak allocation.
    sf_verify_(sfapi::wave_budget_bytes(1, 8 * gigabyte, 80 * gigabyte) == 6 * gigabyte,
               "one free probe must yield 75% of the tightest free slot");
    sf_verify_(sfapi::wave_budget_bytes(4, 8 * gigabyte, 80 * gigabyte) == 6 * gigabyte,
               "the tightest free slot binds regardless of how many devices answered");

    // No device answered: half of the tightest total slot, which is the degraded estimate — not
    // zero, and not the smallest_free argument, which is meaningless in this arm.
    sf_verify_(sfapi::wave_budget_bytes(0, 0, 80 * gigabyte) == 40 * gigabyte,
               "zero free probes must fall back to half of total VRAM");
    sf_verify_(sfapi::wave_budget_bytes(0, 8 * gigabyte, 80 * gigabyte) == 40 * gigabyte,
               "the free reading must not leak into the total-only arm");

    // Nothing known about the hardware at all: zero, which is what drops the sizer onto its default.
    sf_verify_(sfapi::wave_budget_bytes(0, 0, 0) == 0, "an empty census must yield a zero budget");
}

/**
 *  @brief The per-device reservation must reach its answer by arithmetic, not by two wraps.
 *
 *  Asserts the quotient as well as the reservation: the zero-device case used to underflow to
 *  `0xFFFFFFFF`, divide by one, and wrap back to zero, landing on the right final number for
 *  entirely the wrong reason.
 */
static void check_per_device_reservation() {
    struct division_case_t {
        std::uint32_t wave_size;
        std::uint32_t device_count;
        std::uint32_t expected_quotient;
        std::uint32_t expected_reservation;
    };
    constexpr std::array<division_case_t, 4> cases {{
        {0u, 0u, 0u, 1u},              // no devices, no wave: nothing to divide, floor of one slot
        {0u, 1u, 0u, 1u},              // one device, empty wave: the slack molecule alone
        {1u, 0u, 0u, 1u},              // no devices to spread a molecule over
        {50'000u, 8u, 6'250u, 6'251u}, // the ordinary case: an exact split plus slack
    }};
    for (auto const &division_case : cases) {
        std::uint32_t const quotient = sfapi::divide_rounding_up(division_case.wave_size, division_case.device_count);
        std::uint32_t const reservation = sfapi::reserve_molecules_per_device(division_case.wave_size,
                                                                              division_case.device_count);
        if (quotient != division_case.expected_quotient || reservation != division_case.expected_reservation)
            std::fprintf(stderr, "FAIL: wave=%u devices=%u quotient=%u (want %u) reservation=%u (want %u)\n",
                         division_case.wave_size, division_case.device_count, quotient, division_case.expected_quotient,
                         reservation, division_case.expected_reservation);
        sf_verify_(quotient == division_case.expected_quotient, "per-device quotient wrapped or rounded wrongly");
        sf_verify_(reservation == division_case.expected_reservation, "per-device reservation is wrong");
    }
    // Rounding up is what the reservation rests on: an uneven split may not leave a device short.
    sf_verify_(sfapi::divide_rounding_up(50'001u, 8u) == 6'251u, "an uneven split must round up");
}

/** Parse the verbose wave table out of captured logger output; ignores every other line. */
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

/**
 *  @brief A wave whose molecules preprocess rejected wholesale must report no GPU time.
 *
 *  Three waves of two, with the two element-gate rejects last — the input is sorted by descending
 *  SMILES length, so the shortest strings land in the final wave. The pipelined path hands the
 *  same two timing vectors round a 2-slot ring, so wave 2 reads the slot wave 0 wrote; a launch
 *  that returns before resetting that slot republishes wave 0's numbers as wave 2's.
 */
static void check_empty_wave_timings() {
    std::vector<std::string_view> const smiles = {
        "CC(=O)Oc1ccccc1C(=O)O", // 21 chars
        "c1ccccc1",              // 8
        "CCOCCO",                // 6
        "CCCCC",                 // 5
        "[Fe]",                  // 4 — no MMFF94 atom type, rejected by the element gate
        "[La]",                  // 4 — likewise
    };

    constexpr std::uint32_t num_conformers = 2;
    constexpr std::uint32_t max_atoms = 128;
    sfapi::config_t config;
    config.num_conformers = num_conformers;
    config.wave_size = 2; // three waves over six molecules, and the pipelined path with it
    config.random_seed = 7;

    std::uint32_t const total_conformers = static_cast<std::uint32_t>(smiles.size()) * num_conformers;
    std::vector<__half> coordinates(static_cast<std::size_t>(total_conformers) * max_atoms * 3);
    std::vector<float> energies(total_conformers);
    std::vector<sfapi::conformer_status_t> status(total_conformers, sfapi::conformer_status_t::invalid_smiles_k);

    int const device_id = 0;
    sfapi::context_t context(std::span<int const>(&device_id, 1));

    std::FILE *const capture = std::tmpfile();
    sf_verify_(capture != nullptr, "could not open a capture stream for the wave timeline");
    context.logger.verbose_level = 1;
    context.logger.color_enabled = false;
    context.logger.info_stream = capture;
    [[maybe_unused]] sfapi::conformer_status_t const batch_status = context.generate_conformers(
        smiles, config, max_atoms, coordinates, energies, status);
    context.logger.info_stream = stdout;
    context.logger.verbose_level = 0;

    std::string log_text;
    std::rewind(capture);
    for (char buffer[4096];;) {
        std::size_t const read_bytes = std::fread(buffer, 1, sizeof(buffer), capture);
        if (read_bytes == 0) break;
        log_text.append(buffer, read_bytes);
    }
    std::fclose(capture);

    std::vector<wave_row_t> const rows = parse_wave_rows(log_text);
    sf_verify_(rows.size() == 3, "expected three wave rows in the verbose timeline");

    // The rejected molecules never reach a GPU pool, so the last wave is empty by construction.
    sf_verify_(rows[2].molecule_count == 2, "the last wave must still carry its two rejected molecules");
    sf_verify_(rows[0].gpu_max_ms > 0.f, "wave 0 must record real GPU time, or the check below is vacuous");
    sf_verify_(rows[1].gpu_max_ms > 0.f, "wave 1 must record real GPU time");
    sf_verify_(rows[2].gpu_max_ms == 0.f, "an empty wave must not inherit wave 0's GPU timings");
    sf_verify_(rows[2].gpu_average_ms == 0.f, "an empty wave must not inherit wave 0's GPU timings");

    for (std::uint32_t molecule = 4; molecule < 6; ++molecule)
        for (std::uint32_t conformer = 0; conformer < num_conformers; ++conformer)
            sf_verify_(
                status[molecule * num_conformers + conformer] == sfapi::conformer_status_t::unsupported_element_k,
                "the last wave's molecules must be rejected by the element gate");

    std::printf("  empty-wave timings: wave0 gpu_max=%.3f ms, wave2 gpu_max=%.3f ms (must be 0)\n", rows[0].gpu_max_ms,
                rows[2].gpu_max_ms);
}

} // namespace smallfield::tests

int main() {
    sftest::check_wave_budget_arithmetic();
    sftest::check_per_device_reservation();
    sftest::check_empty_wave_timings();
    std::printf("test_pipeline: wave sizing + empty-wave timing checks passed\n");

    std::printf("test_pipeline: end-to-end ETKDG conformer generation via library API\n");

    auto test_smiles_str = sftest::standard_test_smiles();
    std::vector<std::string_view> test_smiles(test_smiles_str.begin(), test_smiles_str.end());

    constexpr std::uint32_t num_conformers = 5;
    constexpr std::uint32_t max_atoms = 128;

    sfapi::config_t config;
    config.num_conformers = num_conformers;
    config.max_stereo_attempts = 20;
    config.random_seed = 42;

    std::uint32_t const num_molecules = static_cast<std::uint32_t>(test_smiles.size());
    std::uint32_t const total_conformers = num_molecules * num_conformers;
    std::uint32_t const total_coords = total_conformers * max_atoms * 3;

    std::vector<__half> coordinates(total_coords);
    std::vector<float> energies(total_conformers);
    std::vector<sfapi::conformer_status_t> status(total_conformers, sfapi::conformer_status_t::invalid_smiles_k);

    // use a single GPU for the test
    int const device_id = 0;
    sfapi::context_t context(std::span<int const>(&device_id, 1));

    sfapi::conformer_status_t const batch_status = context.generate_conformers(test_smiles, config, max_atoms,
                                                                               coordinates, energies, status);

    std::printf("  batch status: %u\n", static_cast<unsigned>(batch_status));

    // cross-validate against RDKit
    std::uint32_t total_attempted = 0;
    std::uint32_t total_successful = 0;
    std::uint32_t total_finite_energy = 0;
    std::uint32_t total_reasonable_energy = 0;

    for (std::uint32_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        std::printf("  molecule %u (%.*s):\n", molecule_index, static_cast<int>(test_smiles[molecule_index].size()),
                    test_smiles[molecule_index].data());

        auto rdkit_mol = sftest::mol_from_smiles(std::string(test_smiles[molecule_index]));
        if (!rdkit_mol) {
            std::printf("    skipped (RDKit parse failure)\n");
            continue;
        }
        bool const embedded = sftest::embed_molecule(*rdkit_mol);
        if (!embedded) {
            std::printf("    skipped (RDKit embed failure)\n");
            continue;
        }

        std::uint32_t const num_atoms = rdkit_mol->getNumAtoms();

        for (std::uint32_t conformer = 0; conformer < num_conformers; ++conformer) {
            total_attempted++;
            std::uint32_t const slot = molecule_index * num_conformers + conformer;

            if (status[slot] != sfapi::conformer_status_t::success_k) {
                std::printf("    conformer %u: status=%u\n", conformer, static_cast<unsigned>(status[slot]));
                continue;
            }
            total_successful++;

            // convert half-precision SoA coordinates to float interleaved for RDKit
            std::uint32_t const coord_offset = slot * max_atoms * 3;
            std::vector<float> interleaved(num_atoms * 3);
            for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
                interleaved[atom * 3 + 0] = __half2float(coordinates[coord_offset + 0 * max_atoms + atom]);
                interleaved[atom * 3 + 1] = __half2float(coordinates[coord_offset + 1 * max_atoms + atom]);
                interleaved[atom * 3 + 2] = __half2float(coordinates[coord_offset + 2 * max_atoms + atom]);
            }

            sftest::set_coords_interleaved(*rdkit_mol, interleaved.data());
            double const rdkit_energy = sftest::rdkit_mmff94_energy(*rdkit_mol);

            if (std::isfinite(rdkit_energy)) {
                total_finite_energy++;
                if (rdkit_energy < 500.0) total_reasonable_energy++;
            }

            if (conformer == 0) {
                std::printf("    conformer 0: rdkit=%.2f sf=%.4f kcal/mol\n", rdkit_energy, energies[slot]);
            }
        }
    }

    double const success_rate = total_attempted > 0 ? 100.0 * total_successful / total_attempted : 0.0;
    std::printf("\n  summary:\n");
    std::printf("    conformers attempted:  %u\n", total_attempted);
    std::printf("    conformers succeeded:  %u (%.1f%%)\n", total_successful, success_rate);
    std::printf("    finite MMFF94 energy:  %u (%.1f%% of successes)\n", total_finite_energy,
                total_successful > 0 ? 100.0 * total_finite_energy / total_successful : 0.0);
    std::printf("    reasonable energy:     %u (< 500 kcal/mol)\n", total_reasonable_energy);

    sf_verify_(total_successful > 0, "at least one conformer must succeed");
    sf_verify_(success_rate >= 50.0, "success rate must be at least 50%%");

    std::printf("\ntest_pipeline: PASSED\n");
    return 0;
}
