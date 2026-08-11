/**
 *  @file test/cuda/test_triangle_smooth.cpp
 *  @brief Test GPU triangle smoothing against the RDKit reference.
 *
 *  Compiled with g++ (not nvcc) to avoid RDKit C++20 header conflicts. CUDA kernel calls are in
 *  test_triangle_smooth_kernels.cu.
 *
 *  Corpus: the built-in standards by default, so a bare `ctest` run stays self-contained; name a `.smi`
 *  file in `argv[1]` or in `SMALLFIELD_TEST_CORPUS` for a wider sweep. Molecules the warp kernel cannot
 *  hold are counted and skipped rather than failing the run, and a sweep that smoothed none of them fails.
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

#include "test/cuda/test_triangle_smooth_kernels.hpp"
#include "test/test_helpers.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sftest = smallfield::tests;

using sftest::gpu_triangle_smooth;

static constexpr std::string_view test_name_k = "test_triangle_smooth";

/** Molecules taken from a named corpus when no cap is given; the built-in standards are always walked. */
static constexpr std::size_t default_max_molecules_k = 50;

/** Atom ceiling of the warp-level kernel: the whole matrix has to fit one shared-memory tile. */
static constexpr std::uint32_t max_smoothed_atoms_k = 128;

/** Which smoothing a flattened bounds matrix has already had when it reaches the caller. */
enum class bounds_smoothing_t {
    /** Topological bounds only — the matrix the device kernel is handed. */
    unsmoothed_k,
    /** RDKit's own `triangleSmoothBounds` applied on the host — the reference. */
    host_smoothed_k,
};

/** Buffers the per-molecule pass writes into, so the sweep over the corpus allocates once rather than per molecule. */
struct bounds_scratch_t {
    /** Host-smoothed reference bounds, N × N row-major. */
    std::vector<double> reference;
    /** Unsmoothed bounds on the way in, device-smoothed in place on the way out. */
    std::vector<float> device;
};

/** What the sweep did with each molecule, so a run that smoothed nothing cannot read like a pass. */
struct run_counts_t {
    std::size_t smoothed_molecules = 0;
    std::size_t oversized_molecules = 0;
    std::size_t unparsed_molecules = 0;
};

/**
 *  @brief Flatten RDKit's bounds for @p molecule into @p flat as a row-major N × N matrix.
 *
 *  Upper triangle holds upper bounds and lower triangle lower bounds, matching what the kernel expects.
 *  Reference and device inputs come from this one builder, so @p smoothing is the only way they differ.
 *
 *  @param[inout] molecule Molecule to derive topological bounds for.
 *  @param[in] smoothing Whether the host smooths the matrix before it is flattened.
 *  @param[out] flat Caller-owned storage, resized to N × N and fully overwritten.
 */
template <typename scalar_type_>
static void fill_bounds_into_(RDKit::RWMol &molecule, bounds_smoothing_t smoothing, std::vector<scalar_type_> &flat) {
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    auto *bounds_raw = new DistGeom::BoundsMatrix(num_atoms);
    DistGeom::BoundsMatPtr bounds_matrix(bounds_raw);
    RDKit::DGeomHelpers::initBoundsMat(bounds_raw);
    RDKit::DGeomHelpers::setTopolBounds(molecule, bounds_matrix);
    if (smoothing == bounds_smoothing_t::host_smoothed_k) DistGeom::triangleSmoothBounds(bounds_raw);

    flat.assign(static_cast<std::size_t>(num_atoms) * num_atoms, static_cast<scalar_type_>(0));
    for (std::uint32_t row = 0; row < num_atoms; ++row) {
        for (std::uint32_t column = row + 1; column < num_atoms; ++column) {
            flat[row * num_atoms + column] = static_cast<scalar_type_>(bounds_matrix->getUpperBound(row, column));
            flat[column * num_atoms + row] = static_cast<scalar_type_>(bounds_matrix->getLowerBound(row, column));
        }
    }
}

/**
 *  @brief Smooth one molecule on the device and compare it against the host reference.
 *
 *  A molecule the kernel cannot hold, or one that does not parse, is counted in @p counts and answers
 *  "nothing disagreed" — it is the tally in `main`, not this answer, that keeps a silent sweep from passing.
 *
 *  @param[in] smiles Molecule under test.
 *  @param[inout] scratch Bounds buffers reused across the corpus.
 *  @param[inout] counts Running tally of smoothed and skipped molecules.
 *  @return Whether this molecule's GPU bounds matched the reference; the caller decides the verdict.
 */
static bool test_triangle_smoothing_single_molecule(std::string const &smiles, bounds_scratch_t &scratch,
                                                    run_counts_t &counts) {
    auto molecule = sftest::mol_from_smiles(smiles);
    if (molecule == nullptr) {
        ++counts.unparsed_molecules;
        return true;
    }

    std::uint32_t const num_atoms = molecule->getNumAtoms();
    if (num_atoms > max_smoothed_atoms_k) {
        ++counts.oversized_molecules;
        return true;
    }

    fmt::println("  testing: {}", smiles);
    fill_bounds_into_(*molecule, bounds_smoothing_t::host_smoothed_k, scratch.reference);
    fill_bounds_into_(*molecule, bounds_smoothing_t::unsmoothed_k, scratch.device);
    gpu_triangle_smooth(scratch.device.data(), num_atoms);
    ++counts.smoothed_molecules;

    float max_error = 0.0f;
    float const tolerance = 1e-2f; // float32 vs float64 tolerance

    sftest::diagnostic_reporter_t reporter {"rdkit", "gpu"};
    reporter.limit_printed_lines(5); // a 128-atom molecule would otherwise emit sixteen thousand lines
    auto const within_tolerance = [tolerance](double rdkit_value, float gpu_value) {
        return !(std::fabs(gpu_value - static_cast<float>(rdkit_value)) > tolerance);
    };

    for (std::uint32_t row = 0; row < num_atoms; ++row) {
        for (std::uint32_t column = 0; column < num_atoms; ++column) {
            if (row == column) continue;

            float const gpu_value = scratch.device[row * num_atoms + column];
            double const rdkit_value = scratch.reference[row * num_atoms + column];
            float const error = std::fabs(gpu_value - static_cast<float>(rdkit_value));

            if (error > max_error) max_error = error;
            // The upper triangle holds upper bounds and the lower triangle lower bounds, so the field
            // name says which of the two disagreed rather than leaving it to be inferred from row < column.
            reporter.mismatch("triangle_smooth", "row={} column={}", row, column)
                .field(row < column ? "upper_bound" : "lower_bound", rdkit_value, gpu_value, within_tolerance);
        }
    }

    fmt::println("    atoms={} max_error={:.6f}", num_atoms, max_error);
    reporter.summarize("triangle_smooth");

    // verify triangle inequality holds after smoothing
    for (std::uint32_t row = 0; row < num_atoms; ++row) {
        for (std::uint32_t column = row + 1; column < num_atoms; ++column) {
            float const upper_row_column = scratch.device[row * num_atoms + column];
            for (std::uint32_t via_atom = 0; via_atom < num_atoms; ++via_atom) {
                if (via_atom == row || via_atom == column) continue;

                float const upper_row_via = (row < via_atom) ? scratch.device[row * num_atoms + via_atom]
                                                             : scratch.device[via_atom * num_atoms + row];
                float const upper_via_column = (via_atom < column) ? scratch.device[via_atom * num_atoms + column]
                                                                   : scratch.device[column * num_atoms + via_atom];

                sf_verify_(upper_row_column <= upper_row_via + upper_via_column + 0.01f,
                           "triangle inequality violation for upper bounds");
            }
        }
    }

    return reporter.passed();
}

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

    // Declared above the loop so the two bounds matrices grow once for the whole corpus.
    bounds_scratch_t scratch;
    run_counts_t counts;

    // Every molecule runs before the verdict: one mismatch should not hide the rest.  The call sits
    // left of `&&` so it is never short-circuited away.
    bool all_matched = true;
    for (auto const &smiles : molecules)
        all_matched = test_triangle_smoothing_single_molecule(smiles, scratch, counts) && all_matched;

    fmt::println("  smoothed={} skipped_oversized={} skipped_unparsed={}", fmt::group_digits(counts.smoothed_molecules),
                 fmt::group_digits(counts.oversized_molecules), fmt::group_digits(counts.unparsed_molecules));
    sf_verify_(counts.smoothed_molecules > 0, "no molecule was small enough for warp-level smoothing");

    if (!all_matched) return sftest::report_fail(test_name_k, "GPU smoothed bounds disagree with the RDKit reference");
    return sftest::report_pass(test_name_k);
}
