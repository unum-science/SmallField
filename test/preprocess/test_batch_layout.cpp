/**
 *  @file test/preprocess/test_batch_layout.cpp
 *  @brief Validate the batch-layout arithmetic against 32-bit wrap, with no GPU and no RDKit corpus.
 *
 *  Two regions a batch carries are quadratic in the atom count — the N × N distance bounds matrix and the
 *  N × N hop-count matrix — and every other region is linear. A `std::uint32_t` running total of a
 *  quadratic region wraps inside a single `embedding-wave` at drug-like sizes, and a wrapped total
 *  under-sizes the pinned allocation rather than failing, so the test drives `accumulate_sizes` and
 *  `compute_batch_layout` at molecule counts either side of the wrap and checks the totals against
 *  arithmetic done in 64 bits.
 *
 *  Nothing here allocates the batch it describes: `compute_batch_layout` is pure arithmetic over element
 *  counts, so a 17 GB layout costs the test a few microseconds.
 */

#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <limits>      // `std::numeric_limits`
#include <string_view> // `std::string_view`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"

#include "test/test_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_batch_layout";

/** Atoms per molecule in the headline sweep — the pipeline's own per-molecule cap. */
static constexpr std::uint32_t sweep_atom_count_k = smallfield::types::max_atoms_per_molecule_k;

/** Molecules in the headline sweep, chosen so the 32-bit total of the two N × N regions wraps. */
static constexpr std::uint32_t sweep_molecule_count_k = 120'000;

/** Atoms per molecule at which the wrap first bites inside the shipped 400000-molecule wave clamp. */
static constexpr std::uint32_t drug_like_atom_count_k = 104;

/** The clamp `compute_wave_size_` applies before this cap was derived. */
static constexpr std::uint32_t wave_size_clamp_k = 400'000;

namespace smallfield::tests {

/** Every element count the layout has to reproduce for a batch of same-sized molecules, in 64 bits. */
struct uniform_batch_expectation_t {
    std::uint64_t molecules;
    std::uint64_t atoms_per_molecule;
    std::uint64_t square_elements;
    std::uint64_t bounds;
    std::uint64_t bond_distance;
    std::uint64_t atoms;
};

/** Compute what a batch of @p molecules molecules of @p atoms_per_molecule atoms must total. */
static uniform_batch_expectation_t expect_uniform_batch_(std::uint64_t molecules,
                                                         std::uint64_t atoms_per_molecule) noexcept {
    uniform_batch_expectation_t expectation {};
    expectation.molecules = molecules;
    expectation.atoms_per_molecule = atoms_per_molecule;
    expectation.square_elements = atoms_per_molecule * atoms_per_molecule;
    expectation.bounds = molecules * expectation.square_elements;
    expectation.bond_distance = expectation.bounds;
    expectation.atoms = molecules * atoms_per_molecule;
    return expectation;
}

/** Accumulate @p expectation's molecules one at a time, exactly as the preprocess prefix scan does. */
static sfp::per_mol_sizes_t accumulate_uniform_batch_(uniform_batch_expectation_t const &expectation) {
    smallfield::types::etkdg_molecule_metadata_t metadata {};
    metadata.num_atoms = static_cast<std::uint32_t>(expectation.atoms_per_molecule);
    sfp::device_capacities_t const capacities {};

    sfp::per_mol_sizes_t totals {};
    for (std::uint64_t molecule = 0; molecule < expectation.molecules; ++molecule)
        sfp::accumulate_sizes(totals, metadata, capacities);
    return totals;
}

/**
 *  @brief Check that the accumulators reproduce the 64-bit arithmetic for one uniform batch.
 *
 *  @param[in] expectation Counts the batch must total.
 *  @param[inout] reporter Sink for the per-field diagnostics and the section's comparison count.
 */
static void check_accumulators_(uniform_batch_expectation_t const &expectation, diagnostic_reporter_t &reporter) {
    sfp::per_mol_sizes_t const totals = accumulate_uniform_batch_(expectation);
    reporter.mismatch("accumulate", "molecules={} atoms={}", expectation.molecules, expectation.atoms_per_molecule)
        .field("bounds", expectation.bounds, static_cast<std::uint64_t>(totals.bounds), exactly_equal)
        .field("bond_distance", expectation.bond_distance, static_cast<std::uint64_t>(totals.bond_distance),
               exactly_equal)
        .field("atoms", expectation.atoms, static_cast<std::uint64_t>(totals.atoms), exactly_equal)
        .field("valid", expectation.molecules, static_cast<std::uint64_t>(totals.valid), exactly_equal);
}

/**
 *  @brief Check that the layout places every region the accumulators sized, past the 32-bit ceiling.
 *
 *  The bounds region is the first one placed and the hop-count matrix the last, so a wrapped total
 *  shows up both as an element count and as a total byte size too small to cover the batch.
 *
 *  @param[in] expectation Counts the batch must total.
 *  @param[inout] reporter Sink for the per-field diagnostics and the section's comparison count.
 */
static void check_layout_(uniform_batch_expectation_t const &expectation, diagnostic_reporter_t &reporter) {
    sfp::per_mol_sizes_t const totals = accumulate_uniform_batch_(expectation);
    std::uint32_t const molecule_count = static_cast<std::uint32_t>(expectation.molecules);
    // One torsion slot per molecule keeps the device-only torsion region out of the comparison.
    smallfield::api::wave_slice_layout_t const layout = sfp::compute_batch_layout(totals, molecule_count, 1u);

    std::uint64_t const bounds_bytes = expectation.bounds * sizeof(float);
    std::uint64_t const bond_distance_bytes = expectation.bond_distance * sizeof(std::uint8_t);
    auto const at_least = [](std::uint64_t reference, std::uint64_t observed) { return observed >= reference; };
    reporter.mismatch("layout", "molecules={} atoms={}", expectation.molecules, expectation.atoms_per_molecule)
        .field("num_bounds", expectation.bounds, static_cast<std::uint64_t>(layout.num_bounds), exactly_equal)
        .field("num_bond_distance", expectation.bond_distance, static_cast<std::uint64_t>(layout.num_bond_distance),
               exactly_equal)
        .field("bytes_uploaded", bounds_bytes, static_cast<std::uint64_t>(layout.bytes_uploaded), at_least)
        .field("bytes_total", bounds_bytes + bond_distance_bytes, static_cast<std::uint64_t>(layout.bytes_total),
               at_least)
        .field("bond_distance_byte_offset", static_cast<std::uint64_t>(layout.bytes_uploaded),
               static_cast<std::uint64_t>(layout.bond_distance_byte_offset), at_least);
}

/**
 *  @brief Check that `max_molecules_per_batch_k` is exactly what the 32-bit offset fields can address.
 *
 *  The cap is the last molecule count whose final bounds offset still fits a `std::uint32_t`, so both
 *  sides of it are asserted: the cap itself fits and one more does not.
 *
 *  @param[inout] reporter Sink for the per-field diagnostics and the section's comparison count.
 */
static void check_offset_cap_(diagnostic_reporter_t &reporter) {
    constexpr std::uint64_t offset_ceiling_k = std::numeric_limits<std::uint32_t>::max();
    constexpr std::uint64_t square_elements_k = static_cast<std::uint64_t>(sweep_atom_count_k) * sweep_atom_count_k;
    constexpr std::uint64_t cap_k = sfp::max_molecules_per_batch_k;

    auto const fits = [](std::uint64_t ceiling, std::uint64_t observed) { return observed <= ceiling; };
    auto const exceeds = [](std::uint64_t ceiling, std::uint64_t observed) { return observed > ceiling; };
    auto const below = [](std::uint64_t ceiling, std::uint64_t observed) { return observed < ceiling; };
    reporter.mismatch("cap", "cap={} atoms={}", cap_k, sweep_atom_count_k)
        .field("offset_at_cap", offset_ceiling_k, cap_k * square_elements_k, fits)
        .field("offset_past_cap", offset_ceiling_k, (cap_k + 1u) * square_elements_k, exceeds)
        // Every linear region is bounded per molecule by the MMFF interaction reservation, which is
        // what makes the bounds matrix the region the cap has to be derived from.
        .field("mmff_interactions_at_cap", offset_ceiling_k, cap_k * smallfield::types::max_mmff_interactions_per_mol_k,
               fits)
        // The cap has to bind before the wave clamp does, or it never gets a chance to.
        .field("cap_under_wave_clamp", wave_size_clamp_k, cap_k, below);
}

/**
 *  @brief Check the molecule count at which a 32-bit total first wraps for drug-like molecules.
 *
 *  The figure is what makes the defect concrete rather than theoretical: it sits below the wave clamp,
 *  so a shipped configuration reaches it.
 *
 *  @param[inout] reporter Sink for the per-field diagnostics and the section's comparison count.
 */
static void check_drug_like_wrap_(diagnostic_reporter_t &reporter) {
    constexpr std::uint64_t wrap_at_k = std::uint64_t {1} << 32;
    constexpr std::uint64_t square_elements_k = static_cast<std::uint64_t>(drug_like_atom_count_k) *
                                                drug_like_atom_count_k;
    constexpr std::uint64_t first_wrapping_count_k = 397'094;

    auto const fits = [](std::uint64_t ceiling, std::uint64_t observed) { return observed < ceiling; };
    auto const exceeds = [](std::uint64_t ceiling, std::uint64_t observed) { return observed >= ceiling; };
    auto const below = [](std::uint64_t ceiling, std::uint64_t observed) { return observed < ceiling; };
    reporter.mismatch("drug_like", "atoms={} molecules={}", drug_like_atom_count_k, first_wrapping_count_k)
        .field("total_below_wrap", wrap_at_k, (first_wrapping_count_k - 1u) * square_elements_k, fits)
        .field("total_at_wrap", wrap_at_k, first_wrapping_count_k * square_elements_k, exceeds)
        .field("wrap_under_wave_clamp", wave_size_clamp_k, first_wrapping_count_k, below);
}

} // namespace smallfield::tests

int main() {
    fmt::println("{}: batch region totals and offset widths past the 32-bit ceiling", test_name_k);

    sftest::diagnostic_reporter_t reporter {"expected", "computed"};

    sftest::check_offset_cap_(reporter);
    reporter.summarize("cap");

    sftest::check_drug_like_wrap_(reporter);
    reporter.summarize("drug_like");

    // The headline sweep: 120000 molecules at the per-molecule atom cap, which totals 4423680000
    // elements in each N × N region and wraps a 32-bit accumulator to 128712704.
    auto const sweep = sftest::expect_uniform_batch_(sweep_molecule_count_k, sweep_atom_count_k);
    fmt::println("Sweep: {} molecules x {} atoms = {} elements per N x N region", fmt::group_digits(sweep.molecules),
                 sweep.atoms_per_molecule, fmt::group_digits(sweep.bounds));
    sftest::check_accumulators_(sweep, reporter);
    reporter.summarize("accumulate");

    sftest::check_layout_(sweep, reporter);
    reporter.summarize("layout");

    if (!reporter.passed())
        return sftest::report_fail(test_name_k, "{} of {} comparisons disagreed", reporter.mismatch_count(),
                                   reporter.comparison_count());
    return sftest::report_pass(test_name_k);
}
