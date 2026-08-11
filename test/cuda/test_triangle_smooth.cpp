/**
 *  @file test/cuda/test_triangle_smooth.cpp
 *  @brief Test GPU triangle smoothing against the RDKit reference.
 *
 *  Compiled with g++ (not nvcc) to avoid RDKit C++20 header conflicts. CUDA kernel calls are in
 *  test_triangle_smooth_kernels.cu.
 */

#include <cmath> // `std::fabs`

#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/TriangleSmooth.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>

#include "test/cuda/test_triangle_smooth_kernels.hpp"
#include "test/test_helpers.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sftest = smallfield::tests;

using sftest::gpu_triangle_smooth;

static constexpr std::string_view test_name_k = "test_triangle_smooth";

static void test_triangle_smoothing_single_molecule(std::string const &smiles) {
    fmt::println("  testing: {}", smiles);

    auto molecule = sftest::mol_from_smiles(smiles);
    sf_verify_(molecule != nullptr, "failed to parse SMILES");

    std::uint32_t const num_atoms = molecule->getNumAtoms();
    sf_verify_(num_atoms <= 128, "molecule too large for warp-level smoothing");

    // get RDKit reference bounds matrix (already smoothed)
    auto rdkit_bounds = sftest::rdkit_bounds_matrix(*molecule);

    // build our unsmoothed bounds matrix, then smooth on GPU
    std::vector<float> gpu_bounds(num_atoms * num_atoms, 0.0f);
    {
        auto *bounds_raw = new DistGeom::BoundsMatrix(num_atoms);
        DistGeom::BoundsMatPtr bounds_mat(bounds_raw);
        RDKit::DGeomHelpers::initBoundsMat(bounds_raw);
        RDKit::DGeomHelpers::setTopolBounds(*molecule, bounds_mat);
        // do @b not smooth on CPU — the GPU kernel does that

        for (std::uint32_t row = 0; row < num_atoms; ++row) {
            for (std::uint32_t column = row + 1; column < num_atoms; ++column) {
                gpu_bounds[row * num_atoms + column] = static_cast<float>(bounds_mat->getUpperBound(row, column));
                gpu_bounds[column * num_atoms + row] = static_cast<float>(bounds_mat->getLowerBound(row, column));
            }
        }
    }

    // run GPU triangle smoothing (modifies gpu_bounds in-place)
    gpu_triangle_smooth(gpu_bounds.data(), num_atoms);

    // compare against RDKit reference
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

            float const gpu_value = gpu_bounds[row * num_atoms + column];
            double const rdkit_value = rdkit_bounds[row * num_atoms + column];
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
            float const upper_row_column = gpu_bounds[row * num_atoms + column];
            for (std::uint32_t via_atom = 0; via_atom < num_atoms; ++via_atom) {
                if (via_atom == row || via_atom == column) continue;

                float const upper_row_via = (row < via_atom) ? gpu_bounds[row * num_atoms + via_atom]
                                                             : gpu_bounds[via_atom * num_atoms + row];
                float const upper_via_column = (via_atom < column) ? gpu_bounds[via_atom * num_atoms + column]
                                                                   : gpu_bounds[column * num_atoms + via_atom];

                sf_verify_(upper_row_column <= upper_row_via + upper_via_column + 0.01f,
                           "triangle inequality violation for upper bounds");
            }
        }
    }
}

int main() {
    fmt::println("test_triangle_smooth: GPU warp smoothing vs the RDKit reference bounds matrix");

    auto molecules = sftest::standard_test_smiles();
    for (auto const &smiles : molecules) { test_triangle_smoothing_single_molecule(smiles); }

    return sftest::report_pass(test_name_k);
}
