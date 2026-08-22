/**
 *  @file test/preprocess/batch_layout.cpp
 *  @brief Validate the batch-layout arithmetic against 32-bit wrap, with no GPU and no RDKit corpus.
 *
 *  Two regions a batch carries are quadratic in the atom count — the N × N distance bounds matrix and the
 *  N × N hop-count matrix — and every other is linear. A `std::uint32_t` running total of a quadratic
 *  region wraps inside a single `embedding-wave` at drug-like sizes and under-sizes the pinned allocation
 *  rather than failing, so `accumulate_sizes` and the slice walk are driven at molecule counts either side
 *  of the wrap and checked against arithmetic done in 64 bits.
 *
 *  Nothing here allocates the batch it describes: an unallocated cursor hands out empty spans and only
 *  totals bytes, so a 17 GB slice costs a few microseconds, and one walk over a one-molecule allocation
 *  then ties those byte totals to the spans they pay for. A slice sized past what a host can page lock is
 *  the same arithmetic's other half, so the pack pass is driven at one and checked to refuse its device's
 *  molecules by name rather than silently.
 */

#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <limits>      // `std::numeric_limits`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"

#include "test/helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "smallfield_test_batch_layout";

/** Atoms per molecule in the headline sweep — the pipeline's own per-molecule cap. */
static constexpr std::uint32_t sweep_atom_count_k = smallfield::types::max_atoms_per_molecule_k;

/** Molecules in the headline sweep, chosen so the 32-bit total of the two N × N regions wraps. */
static constexpr std::uint32_t sweep_molecule_count_k = 120'000;

/** Atoms per molecule at which the wrap first bites inside the shipped 400000-molecule wave clamp. */
static constexpr std::uint32_t drug_like_atom_count_k = 104;

/** The molecule-count clamp `compute_wave_size_` applies, which the cap and the wrap point sit under. */
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

/** Check that the accumulators reproduce the 64-bit arithmetic for one uniform batch, @p expectation naming the
 *  counts it must total and @p reporter taking the per-field diagnostics. */
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
 *  @brief Check that the walk sizes every region the accumulators counted, past the 32-bit ceiling.
 *
 *  Driven over an unallocated cursor, which is how the pool is sized: every carve hands back an empty span and only
 *  the byte total advances. The bounds matrix is placed first and the hop-count matrix last, so a wrapped total shows
 *  up as an uploaded prefix too small for the bounds matrix or a device-only remainder too small for the hop counts.
 *  A carve that also fits the pool is `check_carve_`'s job.
 */
static void check_layout_(uniform_batch_expectation_t const &expectation, diagnostic_reporter_t &reporter) {
    // One torsion slot per molecule keeps the device-only torsion region out of the comparison.
    sfp::wave_slice_shape_t const shape {.totals = accumulate_uniform_batch_(expectation), .torsion_stride = 1u};
    std::uint64_t const uploaded_bytes = sfp::uploaded_wave_slice_bytes(shape);
    std::uint64_t const slice_bytes = sfp::wave_slice_bytes(shape);

    std::uint64_t const bounds_bytes = expectation.bounds * sizeof(float);
    std::uint64_t const bond_distance_bytes = expectation.bond_distance * sizeof(std::uint8_t);
    auto const at_least = [](std::uint64_t reference, std::uint64_t observed) { return observed >= reference; };
    reporter.mismatch("layout", "molecules={} atoms={}", expectation.molecules, expectation.atoms_per_molecule)
        .field("molecules", expectation.molecules, static_cast<std::uint64_t>(shape.totals.valid), exactly_equal)
        .field("uploaded_bytes", bounds_bytes, uploaded_bytes, at_least)
        .field("slice_bytes", bounds_bytes + bond_distance_bytes, slice_bytes, at_least)
        .field("device_only_bytes", bond_distance_bytes, slice_bytes - uploaded_bytes, at_least);
}

/** Check that a walk over an allocation hands back the same regions the sizing walk paid for. Run at one molecule,
 *  since the point is the relation between the two walks, not the arithmetic width: every span lands inside the pool
 *  the sizing walk asked for, the uploaded prefix stops before the first device-only region, and both extents are the
 *  element counts the shape stated. */
static void check_carve_(diagnostic_reporter_t &reporter) {
    auto const single = expect_uniform_batch_(1u, drug_like_atom_count_k);
    sfp::wave_slice_shape_t const shape {.totals = accumulate_uniform_batch_(single), .torsion_stride = 1u};
    std::size_t const slice_bytes = sfp::wave_slice_bytes(shape);

    std::vector<std::byte> pool(slice_bytes);
    sfp::pool_cursor<std::byte> cursor {custd::span<std::byte> {pool}};
    sfp::wave_slice_view_t const view = sfp::carve_wave_slice(shape, cursor);

    auto const *const base = pool.data();
    std::uint64_t const bounds_end = static_cast<std::uint64_t>(
        reinterpret_cast<std::byte const *>(view.bounds_matrices.data() + view.bounds_matrices.size()) - base);
    std::uint64_t const hops_end = static_cast<std::uint64_t>(
        reinterpret_cast<std::byte const *>(view.bond_distance.data() + view.bond_distance.size()) - base);
    std::uint64_t const hops_start = static_cast<std::uint64_t>(
        reinterpret_cast<std::byte const *>(view.bond_distance.data()) - base);

    auto const at_most = [](std::uint64_t ceiling, std::uint64_t observed) { return observed <= ceiling; };
    auto const at_least = [](std::uint64_t reference, std::uint64_t observed) { return observed >= reference; };
    reporter.mismatch("carve", "molecules=1 atoms={}", drug_like_atom_count_k)
        .field("num_bounds", single.bounds, static_cast<std::uint64_t>(view.bounds_matrices.size()), exactly_equal)
        .field("num_bond_distance", single.bond_distance, static_cast<std::uint64_t>(view.bond_distance.size()),
               exactly_equal)
        .field("bounds_inside_pool", slice_bytes, bounds_end, at_most)
        .field("hops_inside_pool", slice_bytes, hops_end, at_most)
        // The bounds matrix crosses PCIe and the hop counts do not, so the transfer's prefix covers the
        // first and stops before the second.
        .field("uploaded_covers_bounds", bounds_end, static_cast<std::uint64_t>(view.uploaded.size()), at_least)
        .field("uploaded_stops_before_hops", static_cast<std::uint64_t>(view.uploaded.size()), hops_start, at_least);
}

/** Check that `max_molecules_per_batch_k` is exactly what the 32-bit offset fields can address: the cap is the last
 *  molecule count whose final bounds offset still fits a `std::uint32_t`, so both sides are asserted - the cap fits,
 *  one more does not. */
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
        // The MMFF interaction reservation bounds every linear region per molecule, which is what makes
        // the bounds matrix the region the cap derives from.
        .field("mmff_interactions_at_cap", offset_ceiling_k, cap_k * smallfield::types::max_mmff_interactions_per_mol_k,
               fits)
        // The cap has to bind before the wave clamp does, or it never gets a chance to.
        .field("cap_under_wave_clamp", wave_size_clamp_k, cap_k, below);
}

/**
 *  @brief Check the bound across atom capacities, which is where a wave and a slice have to agree.
 *
 *  Two sites read this at the caller's own atom capacity — the wave sizing that decides how many molecules a wave
 *  carries, and the preprocess refusal that drops whatever a slice hands it past the bound. They agree only while it
 *  answers identically for every spelling of "no capacity of my own" and never rises as the capacity widens, so both
 *  are asserted here rather than left to one call site each.
 */
static void check_capacity_bound_(diagnostic_reporter_t &reporter) {
    constexpr std::uint64_t cap_k = sfp::max_molecules_per_batch_k;

    auto const at_least = [](std::uint64_t reference, std::uint64_t observed) { return observed >= reference; };
    // Zero says the caller imposes no capacity of its own, and a capacity wider than the pipeline's cap
    // cannot buy room the pipeline lacks; both fold onto the cap.
    reporter.mismatch("capacity", "atoms={} cap={}", sweep_atom_count_k, cap_k)
        .field("unbounded_equals_cap", cap_k, static_cast<std::uint64_t>(sfp::max_molecules_for_atom_capacity(0u)),
               exactly_equal)
        .field("above_cap_equals_cap", cap_k,
               static_cast<std::uint64_t>(sfp::max_molecules_for_atom_capacity(sweep_atom_count_k + 1u)), exactly_equal)
        // The bound is quadratic in the capacity, so halving it admits at least four times the molecules,
        // which is what makes the cap at the pipeline's own atom cap the tightest.
        .field("half_capacity_admits_four_times", 4u * cap_k,
               static_cast<std::uint64_t>(sfp::max_molecules_for_atom_capacity(sweep_atom_count_k / 2u)), at_least);
}

/** Check the molecule count at which a 32-bit total first wraps for drug-like molecules. The figure makes the defect
 *  concrete: it sits below the wave clamp, so a shipped configuration reaches it. */
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

/**
 *  @brief Check that a device whose pinned staging cannot be reserved refuses its batch out loud.
 *
 *  The pack pass sizes each device's pinned pool from the molecules assigned to it, and the one thing it cannot do is
 *  hand the launch a pool it never allocated: the pool then carries no molecules, the launch skips the device, and
 *  every molecule keeps whatever the caller pre-filled. So the refusal is filed per molecule under `batch_overflow_k`
 *  — this is that status's producer, and it is terminal, so the shard parks and is re-run at a wave that fits. It is
 *  driven by fabricating molecules whose bounds matrices alone outspan what any host can page lock, which is what a
 *  mis-sized wave looks like from inside the pack pass; the reservation is refused before a byte is packed, so the
 *  check needs neither a GPU nor a corpus.
 */
static void check_pinned_refusal_(diagnostic_reporter_t &reporter) {
    // Four molecules of 400000 atoms ask 2.5 TB of bounds matrix, and `per_mol_sizes_t::bounds` is
    // 64-bit, so the total states that size instead of wrapping down to a reservable one.
    constexpr std::uint32_t refused_molecule_count_k = 4;
    constexpr std::uint32_t unpinnable_atom_count_k = 400'000;

    std::vector<sfp::batch_prep_thread_contribution_t> contribs(1);
    auto &contribution = contribs.front();
    for (std::uint32_t molecule = 0; molecule < refused_molecule_count_k; ++molecule) {
        sfp::batch_prep_per_mol_record_t record {};
        record.original_index = molecule;
        record.metadata.num_atoms = unpinnable_atom_count_k;
        contribution.mols.push_back(record);
        // A zeroed descriptor claims no bonds and no neighbours, so every device-filled region is empty
        // and the bounds matrix is the whole reservation.
        contribution.descriptors.push_back(smallfield::types::molecule_descriptor_t {});
    }

    sfp::distribute_scratch_t distribution;
    sfp::wave_slice_pool_t pool;
    // A stride survives a reset that forgets it, so it is set here rather than left at its default.
    pool.observed_max_atoms = unpinnable_atom_count_k;
    sfp::pack_contribs_into_device_pools_(contribs, 1u, {&pool, 1}, distribution);

    std::vector<sfp::rejected_mol_t> rejections;
    sfp::collect_rejections_into_(contribs, distribution, refused_molecule_count_k, refused_molecule_count_k,
                                  rejections);

    // Exactly once each, so neither a dropped molecule nor a doubled one reads as a filed refusal.
    std::vector<std::uint32_t> filings(refused_molecule_count_k, 0u);
    std::uint64_t wrong_reason = 0;
    std::uint64_t out_of_range = 0;
    for (sfp::rejected_mol_t const &entry : rejections) {
        if (entry.reason != smallfield::types::conformer_status_t::batch_overflow_k) ++wrong_reason;
        if (entry.slice_local_index >= refused_molecule_count_k) {
            ++out_of_range;
            continue;
        }
        filings[entry.slice_local_index] += 1u;
    }
    std::uint64_t filed_once = 0;
    for (std::uint32_t const count : filings) filed_once += count == 1u ? 1u : 0u;

    reporter.mismatch("pinned_refusal", "molecules={} atoms={}", refused_molecule_count_k, unpinnable_atom_count_k)
        .field("rejections", refused_molecule_count_k, static_cast<std::uint64_t>(rejections.size()), exactly_equal)
        .field("filed_exactly_once", refused_molecule_count_k, filed_once, exactly_equal)
        .field("out_of_range_indices", std::uint64_t {0}, out_of_range, exactly_equal)
        .field("wrong_reason", std::uint64_t {0}, wrong_reason, exactly_equal)
        .field("observed_max_atoms", std::uint64_t {0}, static_cast<std::uint64_t>(pool.observed_max_atoms),
               exactly_equal)
        .field("num_molecules", std::uint64_t {0}, static_cast<std::uint64_t>(pool.num_molecules()), exactly_equal);
}

} // namespace smallfield::tests

int main() {
    fmt::println("{}: batch totals past the 32-bit ceiling, and the refusal an unpinnable slice files", test_name_k);

    sftest::diagnostic_reporter_t reporter {"expected", "computed"};

    sftest::check_offset_cap_(reporter);
    reporter.summarize("cap");

    sftest::check_capacity_bound_(reporter);
    reporter.summarize("capacity");

    sftest::check_drug_like_wrap_(reporter);
    reporter.summarize("drug_like");

    // The headline sweep: 120000 molecules at the per-molecule atom cap totals 4423680000 elements in
    // each N × N region and wraps a 32-bit accumulator to 128712704.
    auto const sweep = sftest::expect_uniform_batch_(sweep_molecule_count_k, sweep_atom_count_k);
    fmt::println("Sweep: {} molecules x {} atoms = {} elements per N x N region", fmt::group_digits(sweep.molecules),
                 sweep.atoms_per_molecule, fmt::group_digits(sweep.bounds));
    sftest::check_accumulators_(sweep, reporter);
    reporter.summarize("accumulate");

    sftest::check_layout_(sweep, reporter);
    reporter.summarize("layout");

    sftest::check_carve_(reporter);
    reporter.summarize("carve");

    sftest::check_pinned_refusal_(reporter);
    reporter.summarize("pinned_refusal");

    if (!reporter.passed())
        return sftest::report_fail(test_name_k, "{} of {} comparisons disagreed", reporter.mismatch_count(),
                                   reporter.comparison_count());
    return sftest::report_pass(test_name_k);
}
