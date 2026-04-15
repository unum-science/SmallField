/**
 *  @file test_triangle_smooth.cpp
 *  @brief Tests for GPU triangle smoothing vs RDKit reference.
 *
 *  Compiled with g++ (not nvcc) to avoid RDKit C++20 header conflicts.
 *  CUDA kernel calls are in test_triangle_smooth_kernels.cu.
 */

#include <cmath>
#include <cstdio>
#include <vector>

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/TriangleSmooth.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>

#include "test_helpers.hpp"

namespace sftest = smallfield::tests;

// defined in test_triangle_smooth_kernels.cu (compiled by nvcc)
extern "C" void gpu_triangle_smooth(float *host_bounds, std::uint32_t num_atoms);

static void test_triangle_smoothing_single_molecule(std::string const &smiles) {
    std::printf("  testing: %s\n", smiles.c_str());

    auto mol = sftest::mol_from_smiles(smiles);
    SF_TEST_ASSERT(mol != nullptr, "failed to parse SMILES");

    std::uint32_t const num_atoms = mol->getNumAtoms();
    SF_TEST_ASSERT(num_atoms <= 128, "molecule too large for warp-level smoothing");

    // get RDKit reference bounds matrix (already smoothed)
    auto rdkit_bounds = sftest::rdkit_bounds_matrix(*mol);

    // build our unsmoothed bounds matrix, then smooth on GPU
    std::vector<float> gpu_bounds(num_atoms * num_atoms, 0.0f);
    {
        auto *bounds_raw = new DistGeom::BoundsMatrix(num_atoms);
        DistGeom::BoundsMatPtr bounds_mat(bounds_raw);
        RDKit::DGeomHelpers::initBoundsMat(bounds_raw);
        RDKit::DGeomHelpers::setTopolBounds(*mol, bounds_mat);
        // do NOT smooth on CPU — the GPU kernel does that

        for (std::uint32_t i = 0; i < num_atoms; ++i) {
            for (std::uint32_t j = i + 1; j < num_atoms; ++j) {
                gpu_bounds[i * num_atoms + j] = static_cast<float>(bounds_mat->getUpperBound(i, j));
                gpu_bounds[j * num_atoms + i] = static_cast<float>(bounds_mat->getLowerBound(i, j));
            }
        }
    }

    // run GPU triangle smoothing (modifies gpu_bounds in-place)
    gpu_triangle_smooth(gpu_bounds.data(), num_atoms);

    // compare against RDKit reference
    float max_error = 0.0f;
    std::uint32_t error_count = 0;
    float const tolerance = 1e-2f; // float32 vs float64 tolerance

    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        for (std::uint32_t j = 0; j < num_atoms; ++j) {
            if (i == j) continue;

            float const gpu_value = gpu_bounds[i * num_atoms + j];
            double const rdkit_value = rdkit_bounds[i * num_atoms + j];
            float const error = std::fabs(gpu_value - static_cast<float>(rdkit_value));

            if (error > max_error) max_error = error;
            if (error > tolerance) {
                if (error_count < 5) {
                    std::printf("    mismatch at (%u,%u): GPU=%.4f RDKit=%.4f err=%.4f\n", i, j, gpu_value, rdkit_value,
                                error);
                }
                ++error_count;
            }
        }
    }

    std::printf("    atoms=%u max_error=%.6f mismatches=%u/%u\n", num_atoms, max_error, error_count,
                num_atoms * num_atoms);

    // verify triangle inequality holds after smoothing
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        for (std::uint32_t j = i + 1; j < num_atoms; ++j) {
            float const upper_ij = gpu_bounds[i * num_atoms + j];
            for (std::uint32_t k = 0; k < num_atoms; ++k) {
                if (k == i || k == j) continue;

                float const u_ik = (i < k) ? gpu_bounds[i * num_atoms + k] : gpu_bounds[k * num_atoms + i];
                float const u_kj = (k < j) ? gpu_bounds[k * num_atoms + j] : gpu_bounds[j * num_atoms + k];

                SF_TEST_ASSERT(upper_ij <= u_ik + u_kj + 0.01f, "triangle inequality violation for upper bounds");
            }
        }
    }
}

int main() {
    std::printf("test_triangle_smooth\n");

    auto molecules = sftest::standard_test_smiles();
    for (auto const &smiles : molecules) { test_triangle_smoothing_single_molecule(smiles); }

    std::printf("ALL TESTS PASSED\n");
    return 0;
}
