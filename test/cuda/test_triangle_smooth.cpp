/**
 *  @file test/cuda/test_triangle_smooth.cpp
 *  @brief Test GPU triangle smoothing, and the verdict it returns, against the RDKit reference.
 *
 *  Compiled with g++, not nvcc, to avoid RDKit C++20 header conflicts; the CUDA kernel calls are in
 *  test_triangle_smooth_kernels.cu. Two sections: `fixture` smooths hand-built matrices whose verdict is
 *  known by construction and confirms each against `DistGeom::triangleSmoothBounds`, the project's only
 *  independent answer to "is this matrix impossible"; `corpus` sweeps real molecules, compares every cell
 *  against RDKit's so a verdict agreeing for the wrong reason still fails, and re-smooths each to pin what
 *  a second sweep may move.
 *
 *  Corpus: the built-in standards by default, so a bare `ctest` run stays self-contained; name a `.smi` in
 *  `argv[1]` or `SMALLFIELD_TEST_CORPUS` for a wider sweep. Molecules the warp kernel cannot hold are
 *  counted and skipped rather than failing the run, and a sweep that smoothed none fails.
 */

#include <cmath>   // `std::fabs`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`
#include <cstdlib> // `std::exit`

#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/TriangleSmooth.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>

#include "smallfield/types/views.hpp"

#include "test/cuda/test_triangle_smooth_kernels.hpp"
#include "test/test_helpers.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sftest = smallfield::tests;

using sftest::exactly_equal;
using sftest::gpu_triangle_smooth;
using smallfield::cuda::smoothing_name;
using smallfield::cuda::smoothing_t;
using smallfield::types::packed_matrix;

static constexpr std::string_view test_name_k = "test_triangle_smooth";

/** Molecules taken from a named corpus when no cap is given; the built-in standards are always walked. */
static constexpr std::size_t default_max_molecules_k = 50;

/** Float32 against RDKit's float64, over a sweep whose every intermediate sum is in single precision. */
static constexpr float bound_tolerance_k = 1e-2f;

/** Both operands land in float before the comparison, because single precision is what is under test. */
static constexpr auto within_bound_tolerance = [](auto reference, auto observed) {
    return !(std::fabs(static_cast<float>(observed) - static_cast<float>(reference)) > bound_tolerance_k);
};

#pragma region Fixtures

/** One atom pair's window, in angstrom, as a fixture states it. */
struct pair_bound_t {
    std::uint32_t first_atom;
    std::uint32_t second_atom;
    double lower;
    double upper;
};

/** A hand-built bounds matrix and the verdict smoothing it has to reach. */
struct bounds_fixture_t {
    std::string_view name;
    std::uint32_t num_atoms;
    smoothing_t expected_verdict;
    custd::span<pair_bound_t const> pairs;
};

/** Two rigid 1 A bonds hold atoms 1 and 2 within 2 A; their own window asks 4. Nothing satisfies both. */
static constexpr pair_bound_t three_atom_contradiction_k[] = {{0, 1, 1.0, 1.0}, {0, 2, 1.0, 1.0}, {1, 2, 4.0, 5.0}};

/** The same two bonds with a window the triangle can meet: the upper falls from 5 to 2, the lower holds. */
static constexpr pair_bound_t three_atom_tightened_k[] = {{0, 1, 1.0, 1.0}, {0, 2, 1.0, 1.0}, {1, 2, 0.5, 5.0}};

/** The same geometry stated at its closure, so every inequality holds on arrival and nothing moves. */
static constexpr pair_bound_t three_atom_already_tight_k[] = {{0, 1, 1.0, 1.0}, {0, 2, 1.0, 1.0}, {1, 2, 0.0, 2.0}};

/** A rigid four-bond chain of 1.5 A steps; every non-bonded window arrives open and closes onto the chain. */
static constexpr pair_bound_t five_atom_chain_k[] = {
    {0, 1, 1.5, 1.5},   {1, 2, 1.5, 1.5},   {2, 3, 1.5, 1.5},   {3, 4, 1.5, 1.5},   {0, 2, 1.0, 100.0},
    {0, 3, 1.0, 100.0}, {0, 4, 1.0, 100.0}, {1, 3, 1.0, 100.0}, {1, 4, 1.0, 100.0}, {2, 4, 1.0, 100.0}};

/** The same chain with its two ends held at least 9 A apart, which four 1.5 A steps cannot reach. */
static constexpr pair_bound_t five_atom_chain_overstretched_k[] = {
    {0, 1, 1.5, 1.5},   {1, 2, 1.5, 1.5},  {2, 3, 1.5, 1.5},   {3, 4, 1.5, 1.5},   {0, 2, 1.0, 100.0},
    {0, 3, 1.0, 100.0}, {0, 4, 9.0, 12.0}, {1, 3, 1.0, 100.0}, {1, 4, 1.0, 100.0}, {2, 4, 1.0, 100.0}};

static constexpr bounds_fixture_t fixtures_k[] = {
    {"three_atom_contradiction", 3, smoothing_t::infeasible_k, three_atom_contradiction_k},
    {"three_atom_tightened", 3, smoothing_t::tightened_k, three_atom_tightened_k},
    {"three_atom_already_tight", 3, smoothing_t::consistent_k, three_atom_already_tight_k},
    {"five_atom_chain", 5, smoothing_t::tightened_k, five_atom_chain_k},
    {"five_atom_chain_overstretched", 5, smoothing_t::infeasible_k, five_atom_chain_overstretched_k},
};

#pragma endregion Fixtures

#pragma region Scratch

/** Buffers each pass writes into, so a sweep over the corpus allocates once rather than per molecule. */
struct bounds_scratch_t {
    /** Host-smoothed reference bounds over a corpus molecule, N × N row-major. */
    std::vector<double> reference;
    /** The same for a fixture, whose windows are authored in float precision anyway. */
    std::vector<float> fixture_reference;
    /** Unsmoothed bounds on the way in, device-smoothed in place on the way out. */
    std::vector<float> device;
    /** The device matrix as it stood before the second sweep `resmooth_` runs over it. */
    std::vector<float> previous;
};

/**
 *  @brief What the sweep did with each molecule, so a run that smoothed nothing cannot read like a pass.
 *  `widest_crossing` is the widest `lower - upper` any satisfiable molecule ended on, which `main` gates
 *  against `bounds_crossing_tolerance_k`.
 */
struct run_counts_t {
    std::size_t smoothed_molecules = 0;
    std::size_t oversized_molecules = 0;
    std::size_t unparsed_molecules = 0;
    std::size_t infeasible_molecules = 0;
    float widest_crossing = 0.0f;
};

#pragma endregion Scratch

#pragma region Reference

/**
 *  @brief Copy an RDKit bounds matrix into the packed square the kernel reads, upper bounds in the upper
 *  triangle and lower in the lower. Reference and device inputs both come through here, so host smoothing
 *  is the only way they differ.
 *  @param[in] bounds RDKit's own matrix, smoothed or not.
 *  @param[in] num_atoms Row and column count of @p bounds.
 *  @param[inout] storage Caller-owned buffer, resized to N × N and fully overwritten.
 */
template <typename scalar_type_>
static packed_matrix<scalar_type_> flatten_bounds_into_(DistGeom::BoundsMatrix const &bounds, std::uint32_t num_atoms,
                                                        std::vector<scalar_type_> &storage) {
    auto const matrix = smallfield::types::reset_square_matrix(storage, num_atoms, static_cast<scalar_type_>(0));
    for (std::uint32_t row = 0; row < num_atoms; ++row)
        for (std::uint32_t column = row + 1; column < num_atoms; ++column) {
            matrix(row, column) = static_cast<scalar_type_>(bounds.getUpperBound(row, column));
            matrix(column, row) = static_cast<scalar_type_>(bounds.getLowerBound(row, column));
        }
    return matrix;
}

/**
 *  @brief Lay a fixture's pair windows into an RDKit bounds matrix.
 *  @param[in] fixture Pair windows; every unordered pair of its atoms appears exactly once.
 *  @param[out] bounds Matrix of the fixture's atom count, fully written.
 */
static void apply_fixture_(bounds_fixture_t const &fixture, DistGeom::BoundsMatrix &bounds) {
    for (auto const &pair : fixture.pairs) {
        bounds.setLowerBound(pair.first_atom, pair.second_atom, pair.lower);
        bounds.setUpperBound(pair.first_atom, pair.second_atom, pair.upper);
    }
}

/**
 *  @brief RDKit's topological bounds for @p molecule, before any smoothing. Returns RDKit's own owning
 *  handle because `setTopolBounds` takes nothing else.
 *  @return A freshly built bounds matrix over the molecule's atoms.
 */
static DistGeom::BoundsMatPtr topological_bounds_(RDKit::RWMol &molecule) {
    auto *bounds_raw = new DistGeom::BoundsMatrix(molecule.getNumAtoms());
    DistGeom::BoundsMatPtr bounds(bounds_raw);
    RDKit::DGeomHelpers::initBoundsMat(bounds_raw);
    RDKit::DGeomHelpers::setTopolBounds(molecule, bounds);
    return bounds;
}

#pragma endregion Reference

#pragma region Sections

/**
 *  @brief Widest amount by which a pair's lower bound overshoots its upper, zero when none does.
 *  @return `max(lower - upper)` over every unordered pair, floored at zero.
 */
static float widest_crossing_(packed_matrix<float const> matrix) {
    float widest = 0.0f;
    for (std::uint32_t row = 0; row < matrix.extent(0); ++row)
        for (std::uint32_t column = row + 1; column < matrix.extent(0); ++column) {
            float const crossing = matrix(column, row) - matrix(row, column);
            if (crossing > widest) widest = crossing;
        }
    return widest;
}

/**
 *  @brief Assert every smoothed upper bound satisfies the triangle inequalities it was smoothed against.
 *  A post-condition, not a comparison, so it aborts on the first violation: one violated triangle means
 *  the sweep never reached closure and the rest of the matrix says nothing.
 */
static void verify_triangle_closure_(packed_matrix<float const> matrix) {
    using smallfield::types::upper_distance_bound_at;
    for (std::uint32_t row = 0; row < matrix.extent(0); ++row)
        for (std::uint32_t column = row + 1; column < matrix.extent(0); ++column)
            for (std::uint32_t via_atom = 0; via_atom < matrix.extent(0); ++via_atom) {
                if (via_atom == row || via_atom == column) continue;
                float const routed = upper_distance_bound_at(matrix, row, via_atom) +
                                     upper_distance_bound_at(matrix, via_atom, column);
                sf_verify_(matrix(row, column) <= routed + bound_tolerance_k,
                           "a smoothed upper bound still exceeds the two legs it routes through");
            }
}

/**
 *  @brief Smooth an already-smoothed matrix again and check which way each bound is allowed to move.
 *
 *  Monotonicity is the property the crossing scan rests on: an upper bound is only ever assigned a smaller
 *  value and a lower bound a larger one, so a pair crossed at any point in a sweep is still crossed at the
 *  end and no later sweep un-crosses it.
 *  A second sweep is not a no-op, and neither is RDKit's: the lower-bound update reads upper bounds the
 *  same sweep is still lowering, so an early pivot leaves a lower bound a further sweep raises again, and
 *  in float the upper bounds shed a last ULP when `u_ik + u_kj` recomputed from rounded legs lands under
 *  the sum that set `u_ij`. Monotonicity survives both, so it is what is asserted.
 *  @param[inout] matrix Smoothed matrix, smoothed once more in place.
 *  @param[inout] previous Caller-owned buffer holding the matrix as it stood before this sweep.
 *  @param[inout] reporter Where a bound that moved the wrong way prints.
 *  @param[in] smiles Molecule the matrix belongs to, for the diagnostic's key column.
 */
static void resmooth_(packed_matrix<float> matrix, std::vector<float> &previous,
                      sftest::diagnostic_reporter_t &reporter, std::string const &smiles) {
    static constexpr auto never_rises = [](float earlier, float later) { return later <= earlier; };
    static constexpr auto never_falls = [](float earlier, float later) { return later >= earlier; };

    std::uint32_t const num_atoms = matrix.extent(0);
    auto const before = smallfield::types::reset_square_matrix(previous, num_atoms, 0.0f);
    for (std::uint32_t row = 0; row < num_atoms; ++row)
        for (std::uint32_t column = 0; column < num_atoms; ++column) before(row, column) = matrix(row, column);

    smoothing_t const verdict = gpu_triangle_smooth(matrix);
    for (std::uint32_t row = 0; row < num_atoms; ++row)
        for (std::uint32_t column = 0; column < num_atoms; ++column) {
            if (row == column) continue;
            auto group = reporter.mismatch("resmooth", "smiles={} row={} column={} verdict={}", smiles, row, column,
                                           smoothing_name(verdict));
            if (row < column) group.field("upper_bound", before(row, column), matrix(row, column), never_rises);
            else group.field("lower_bound", before(row, column), matrix(row, column), never_falls);
        }
}

/**
 *  @brief Smooth every hand-built fixture and check its verdict against construction and against RDKit.
 *  @param[inout] reporter Where a disagreeing verdict or cell prints.
 *  @param[inout] scratch Bounds buffers reused across the fixtures.
 */
static void compare_fixtures_(sftest::diagnostic_reporter_t &reporter, bounds_scratch_t &scratch) {
    for (auto const &fixture : fixtures_k) {
        DistGeom::BoundsMatrix host_bounds(fixture.num_atoms), device_bounds(fixture.num_atoms);
        apply_fixture_(fixture, host_bounds);
        apply_fixture_(fixture, device_bounds);
        bool const rdkit_satisfiable = DistGeom::triangleSmoothBounds(&host_bounds);

        auto const reference = flatten_bounds_into_(host_bounds, fixture.num_atoms, scratch.fixture_reference);
        auto const device = flatten_bounds_into_(device_bounds, fixture.num_atoms, scratch.device);
        smoothing_t const verdict = gpu_triangle_smooth(device);

        fmt::println("  fixture {} atoms={} verdict={}", fixture.name, fixture.num_atoms, smoothing_name(verdict));
        auto group = reporter.mismatch("fixture", "name={} verdict={}", fixture.name, smoothing_name(verdict));
        group.field("verdict", static_cast<std::uint8_t>(fixture.expected_verdict), static_cast<std::uint8_t>(verdict),
                    exactly_equal);
        group.field("satisfiable", rdkit_satisfiable, verdict != smoothing_t::infeasible_k, exactly_equal);

        // RDKit returns the moment it finds a crossed pair, so an unsatisfiable fixture's reference is only
        // partly smoothed and has nothing to compare against; a satisfiable one reached closure.
        if (!rdkit_satisfiable) continue;
        for (std::uint32_t row = 0; row < fixture.num_atoms; ++row)
            for (std::uint32_t column = 0; column < fixture.num_atoms; ++column) {
                if (row == column) continue;
                reporter.mismatch("fixture_cell", "name={} row={} column={}", fixture.name, row, column)
                    .field(row < column ? "upper_bound" : "lower_bound", reference(row, column), device(row, column),
                           within_bound_tolerance);
            }
    }
}

/**
 *  @brief Smooth one molecule on the device and compare every cell against the host reference. One the
 *  kernel cannot hold, or that does not parse, is counted in @p counts and compared against nothing; the
 *  tally in `main` is what keeps a silent sweep from passing.
 *  @param[inout] scratch Bounds buffers reused across the corpus.
 *  @param[inout] counts Running tally of smoothed and skipped molecules.
 *  @param[inout] reporter Where a disagreeing verdict or cell prints.
 *  @param[inout] resweep Where a second sweep that moved a bound the wrong way prints.
 */
static void compare_molecule_(std::string const &smiles, bounds_scratch_t &scratch, run_counts_t &counts,
                              sftest::diagnostic_reporter_t &reporter, sftest::diagnostic_reporter_t &resweep) {
    auto molecule = sftest::mol_from_smiles(smiles);
    if (molecule == nullptr) {
        ++counts.unparsed_molecules;
        return;
    }

    std::uint32_t const num_atoms = molecule->getNumAtoms();
    if (num_atoms > sftest::max_smoothed_atoms_k) {
        ++counts.oversized_molecules;
        return;
    }

    auto const device_bounds = topological_bounds_(*molecule);
    auto const host_bounds = topological_bounds_(*molecule);
    bool const rdkit_satisfiable = DistGeom::triangleSmoothBounds(host_bounds);

    auto const device = flatten_bounds_into_(*device_bounds, num_atoms, scratch.device);
    auto const reference = flatten_bounds_into_(*host_bounds, num_atoms, scratch.reference);
    smoothing_t const verdict = gpu_triangle_smooth(device);
    ++counts.smoothed_molecules;

    reporter.mismatch("corpus_verdict", "smiles={} verdict={}", smiles, smoothing_name(verdict))
        .field("satisfiable", rdkit_satisfiable, verdict != smoothing_t::infeasible_k, exactly_equal);
    if (verdict == smoothing_t::infeasible_k) {
        ++counts.infeasible_molecules;
        return;
    }

    float const crossing = widest_crossing_(device);
    if (crossing > counts.widest_crossing) counts.widest_crossing = crossing;
    verify_triangle_closure_(device);

    if (rdkit_satisfiable) // a partly smoothed reference has no cell worth comparing
        for (std::uint32_t row = 0; row < num_atoms; ++row)
            for (std::uint32_t column = 0; column < num_atoms; ++column) {
                if (row == column) continue;
                // The field name says which bound disagreed rather than leaving it inferred from row < column.
                reporter.mismatch("corpus_cell", "smiles={} row={} column={}", smiles, row, column)
                    .field(row < column ? "upper_bound" : "lower_bound", reference(row, column), device(row, column),
                           within_bound_tolerance);
            }

    resmooth_(device, scratch.previous, resweep, smiles);
}

#pragma endregion Sections

int main(int argc, char **argv) {
    fmt::println("{}: GPU warp smoothing vs the RDKit reference bounds matrix", test_name_k);

    auto const selection = sftest::select_corpus(argc, argv, default_max_molecules_k);
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", argv[0]);
        return 2;
    }

    std::vector<std::string> molecules = sftest::standard_test_smiles();
    if (selection.knob == sftest::corpus_knob_t::named_k) {
        std::vector<std::string> loaded;
        auto const outcome = sftest::load_smiles_into(selection.path, selection.max_molecules, loaded);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), selection.path);
        fmt::println("Corpus: {} ({} molecules) plus built-ins", selection.path, fmt::group_digits(loaded.size()));
        for (auto &smiles : loaded) molecules.push_back(std::move(smiles));
    }
    else fmt::println("Corpus: built-ins only; name one on the command line or in SMALLFIELD_TEST_CORPUS");

    // Declared above the loops so the bounds matrices grow once for the whole run.
    bounds_scratch_t scratch;
    run_counts_t counts;
    sftest::diagnostic_reporter_t reporter {"rdkit", "gpu"};
    sftest::diagnostic_reporter_t resweep {"first_sweep", "second_sweep"};
    // A 128-atom molecule would otherwise emit sixteen thousand lines from one section.
    reporter.limit_printed_lines(20);
    resweep.limit_printed_lines(20);

    compare_fixtures_(reporter, scratch);
    reporter.summarize("fixture");

    // Every molecule runs before any verdict: one mismatch should not hide the rest.
    for (auto const &smiles : molecules) compare_molecule_(smiles, scratch, counts, reporter, resweep);
    reporter.summarize("corpus");
    resweep.summarize("resmooth");

    fmt::println("  smoothed={} infeasible={} skipped_oversized={} skipped_unparsed={} widest_crossing={}",
                 fmt::group_digits(counts.smoothed_molecules), fmt::group_digits(counts.infeasible_molecules),
                 fmt::group_digits(counts.oversized_molecules), fmt::group_digits(counts.unparsed_molecules),
                 counts.widest_crossing);
    sf_verify_(counts.smoothed_molecules > 0, "no molecule was small enough for warp-level smoothing");
    sf_verify_(counts.widest_crossing <= smallfield::cuda::bounds_crossing_tolerance_k,
               "a molecule RDKit calls satisfiable ends wider than the crossing tolerance allows");

    if (!reporter.passed() || !resweep.passed())
        return sftest::report_fail(test_name_k, "GPU smoothing disagrees over {} fields, and re-smoothing over {}",
                                   reporter.mismatch_count(), resweep.mismatch_count());
    return sftest::report_pass(test_name_k);
}
