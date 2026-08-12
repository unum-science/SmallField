/**
 *  @file test/cuda/test_bounds_infeasible.cpp
 *  @brief Test that a bounds matrix no coordinates satisfy comes back as `bounds_infeasible_k`.
 *
 *  The metric prepass is the only stage that can tell an impossible bounds matrix from a hard one, and its
 *  verdict travels two launches before it becomes the status a shard carries. This walks that whole route:
 *  hand-built matrices whose feasibility is known by construction go through the production prepass driver,
 *  the production pipeline kernel and the stamp behind it, and each one's conformer slot has to name what
 *  its matrix was.
 *
 *  Each fixture is checked twice over — the status it lands on, and whether the smoothed matrix the prepass
 *  persisted actually holds a crossed pair — because a status that agrees while the matrix does not is a
 *  verdict reached for the wrong reason.
 *
 *  The fixtures are those of `test_triangle_smooth`, which checks the same matrices against
 *  `DistGeom::triangleSmoothBounds` — so that this suite states what the verdict is worth and that one states
 *  that the verdict is right.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`
#include <cstdlib> // `std::exit`

#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`

#include "smallfield/cuda/triangle_smooth.cuh"
#include "smallfield/types/conformer_status.hpp"
#include "smallfield/types/views.hpp"

#include "test/cuda/test_bounds_infeasible_kernels.hpp"
#include "test/test_helpers.hpp"

namespace sftest = smallfield::tests;

using sftest::exactly_equal;
using sftest::gpu_pipeline_status_of_bounds;
using smallfield::types::conformer_status_t;
using smallfield::types::packed_matrix;

static constexpr std::string_view test_name_k = "test_bounds_infeasible";

#pragma region Fixtures

/** One atom pair's window, in angstrom, as a fixture states it. */
struct pair_bound_t {
    std::uint32_t first_atom;
    std::uint32_t second_atom;
    float lower;
    float upper;
};

/** A hand-built bounds matrix and the status the wave owes the conformer it produces. */
struct bounds_fixture_t {
    std::string_view name;
    std::uint32_t num_atoms;
    conformer_status_t expected_status;
    custd::span<pair_bound_t const> pairs;
};

/* Two rigid 1 A bonds from atom 0 hold atoms 1 and 2 within 2 A of one another, and their own window asks
 * for at least 4. Nothing can satisfy both. */
static constexpr pair_bound_t three_atom_contradiction_k[] = {
    {0, 1, 1.0f, 1.0f}, {0, 2, 1.0f, 1.0f}, {1, 2, 4.0f, 5.0f}};

/* The same two bonds with a window the triangle can meet: the upper falls from 5 to 2 and the lower holds. */
static constexpr pair_bound_t three_atom_tightened_k[] = {{0, 1, 1.0f, 1.0f}, {0, 2, 1.0f, 1.0f}, {1, 2, 0.5f, 5.0f}};

/* A rigid four-bond chain of 1.5 A steps, whose non-bonded windows arrive wide open and close onto the chain
 * length that reaches them. */
static constexpr pair_bound_t five_atom_chain_k[] = {
    {0, 1, 1.5f, 1.5f},   {1, 2, 1.5f, 1.5f},   {2, 3, 1.5f, 1.5f},   {3, 4, 1.5f, 1.5f},   {0, 2, 1.0f, 100.0f},
    {0, 3, 1.0f, 100.0f}, {0, 4, 1.0f, 100.0f}, {1, 3, 1.0f, 100.0f}, {1, 4, 1.0f, 100.0f}, {2, 4, 1.0f, 100.0f}};

/* The same chain with its two ends held at least 9 A apart, which four 1.5 A steps cannot reach. */
static constexpr pair_bound_t five_atom_chain_overstretched_k[] = {
    {0, 1, 1.5f, 1.5f},   {1, 2, 1.5f, 1.5f},  {2, 3, 1.5f, 1.5f},   {3, 4, 1.5f, 1.5f},   {0, 2, 1.0f, 100.0f},
    {0, 3, 1.0f, 100.0f}, {0, 4, 9.0f, 12.0f}, {1, 3, 1.0f, 100.0f}, {1, 4, 1.0f, 100.0f}, {2, 4, 1.0f, 100.0f}};

/* Both feasible fixtures are here to keep the refusal narrow: a guard that refused everything would satisfy
 * the two infeasible rows on its own. */
static constexpr bounds_fixture_t fixtures_k[] = {
    {"three_atom_contradiction", 3, conformer_status_t::bounds_infeasible_k, three_atom_contradiction_k},
    {"three_atom_tightened", 3, conformer_status_t::success_k, three_atom_tightened_k},
    {"five_atom_chain", 5, conformer_status_t::success_k, five_atom_chain_k},
    {"five_atom_chain_overstretched", 5, conformer_status_t::bounds_infeasible_k, five_atom_chain_overstretched_k},
};

#pragma endregion Fixtures

#pragma region Sections

/**
 *  @brief Lay a fixture's pair windows into the packed square the kernels read.
 *
 *  @param[in] fixture Pair windows; every unordered pair of its atoms appears exactly once.
 *  @param[inout] storage Caller-owned buffer, resized to N × N and fully overwritten.
 *  @return View over @p storage.
 */
static packed_matrix<float> flatten_fixture_into_(bounds_fixture_t const &fixture, std::vector<float> &storage) {
    auto const matrix = smallfield::types::reset_square_matrix(storage, fixture.num_atoms, 0.0f);
    for (auto const &pair : fixture.pairs) {
        matrix(pair.first_atom, pair.second_atom) = pair.upper;
        matrix(pair.second_atom, pair.first_atom) = pair.lower;
    }
    return matrix;
}

/**
 *  @brief Widest amount by which a pair's lower bound overshoots its upper, zero when none does.
 *
 *  @param[in] matrix Smoothed bounds matrix.
 *  @return `max(lower - upper)` over every unordered pair, floored at zero.
 */
static float widest_crossing_(packed_matrix<float const> matrix) {
    float widest = 0.0f;
    for (std::uint32_t row = 0; row < matrix.extent(0); ++row)
        for (std::uint32_t column = row + 1; column < matrix.extent(0); ++column) {
            float const crossing = smallfield::types::lower_distance_bound_at(matrix, row, column) -
                                   smallfield::types::upper_distance_bound_at(matrix, row, column);
            widest = crossing > widest ? crossing : widest;
        }
    return widest;
}

/**
 *  @brief Run every fixture through the three launches and compare the slot it lands in.
 *
 *  @param[inout] reporter Where a disagreeing fixture reports itself.
 *  @param[inout] bounds_storage Caller-owned bounds buffer, reused across fixtures.
 */
static void compare_fixtures_(sftest::diagnostic_reporter_t &reporter, std::vector<float> &bounds_storage) {
    for (auto const &fixture : fixtures_k) {
        std::size_t const expected_pairs = std::size_t {fixture.num_atoms} * (fixture.num_atoms - 1u) / 2u;
        sf_verify_(fixture.pairs.size() == expected_pairs,
                   fmt::format("fixture {} leaves an atom pair without a window", fixture.name));

        auto const matrix = flatten_fixture_into_(fixture, bounds_storage);
        conformer_status_t const observed = gpu_pipeline_status_of_bounds(matrix);

        // The smoothed matrix the prepass persists is what the verdict was read off, so a status that
        // agrees while the matrix does not is a verdict reached for the wrong reason.
        bool const expects_refusal = fixture.expected_status == conformer_status_t::bounds_infeasible_k;
        bool const crossed = widest_crossing_(packed_matrix<float const> {matrix.data_handle(), fixture.num_atoms,
                                                                          fixture.num_atoms}) >
                             smallfield::cuda::bounds_crossing_tolerance_k;
        reporter.mismatch("fixture", "name={} atoms={}", fixture.name, fixture.num_atoms)
            .field("status", static_cast<unsigned>(fixture.expected_status), static_cast<unsigned>(observed),
                   exactly_equal)
            .field("bounds_crossed", expects_refusal, crossed, exactly_equal);
    }
}

#pragma endregion Sections

int main() {
    fmt::println("{}: metric prepass and pipeline kernel over bounds matrices of known feasibility", test_name_k);

    // Declared above the loop so the buffer grows once for the whole run.
    std::vector<float> bounds_storage;
    sftest::diagnostic_reporter_t reporter {"expected", "device"};

    compare_fixtures_(reporter, bounds_storage);
    reporter.summarize("fixture");

    if (!reporter.passed())
        return sftest::report_fail(test_name_k, "the wave's verdict disagrees over {} fields",
                                   reporter.mismatch_count());
    return sftest::report_pass(test_name_k);
}
