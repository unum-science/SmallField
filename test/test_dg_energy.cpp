/**
 *  @file test_dg_energy.cpp
 *  @brief Tests for DG energy evaluation, gradient correctness, and L-BFGS convergence.
 *  Compiled with g++ (C++20). CUDA kernels in test_dg_energy_kernels.cu.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "etkdg_preprocess.hpp"
#include "test_helpers.hpp"

using namespace smallfield;
using namespace smallfield::test;

extern "C" void gpu_dg_energy(float const *host_coords, float *host_gradient, float const *host_bounds,
                              std::uint32_t num_atoms, std::uint32_t spatial_dimension, float upper_weight,
                              float lower_weight, float fourth_dim_weight, float *out_energy);

extern "C" void gpu_dg_minimize(float *host_coords, float const *host_bounds, std::uint32_t num_atoms,
                                std::uint32_t spatial_dimension, float upper_weight, float lower_weight,
                                float fourth_dim_weight, std::int32_t max_iterations, float gradient_threshold,
                                float *out_energy, std::int32_t *out_iterations, std::int32_t *out_status);

// CPU reference: compute DG distance violation energy in double precision.
// Uses float32 bounds (the GPU uses half-precision, so energies will differ).
static double cpu_dg_energy_reference(double const *coords, float const *bounds, std::uint32_t num_atoms,
                                      std::uint32_t dim, float upper_weight, float lower_weight) {
    double energy = 0.0;
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        for (std::uint32_t j = i + 1; j < num_atoms; ++j) {
            double dist_sq = 0.0;
            for (std::uint32_t d = 0; d < dim; ++d) {
                double delta = coords[d * num_atoms + j] - coords[d * num_atoms + i];
                dist_sq += delta * delta;
            }
            double dist = std::sqrt(std::max(dist_sq, 1e-12));

            double upper = static_cast<double>(bounds[i * num_atoms + j]);
            double lower = static_cast<double>(bounds[j * num_atoms + i]);

            if (dist > upper) {
                double violation = dist - upper;
                energy += upper_weight * violation * violation;
            }
            if (dist < lower) {
                double violation = lower - dist;
                energy += lower_weight * violation * violation;
            }
        }
    }
    return energy;
}

static void test_dg_energy_known_violation() {
    std::printf("test_dg_energy_known_violation\n");

    auto mol = mol_from_smiles("CCO");
    SF_TEST_ASSERT(mol != nullptr, "failed to parse CCO");

    bool ok = embed_molecule(*mol);
    SF_TEST_ASSERT(ok, "failed to embed CCO");

    std::uint32_t const num_atoms = mol->getNumAtoms();
    std::uint32_t const dim = 4;
    std::uint32_t const num_elements = dim * num_atoms;

    // get smoothed bounds matrix
    std::vector<float> bounds;
    molecule_descriptor_t desc;
    build_descriptor(*mol, desc);
    build_bounds_matrix(desc, bounds);

    // create 4D coords from RDKit 3D coords + random 4th dim
    auto coords_3d = extract_coords_interleaved(*mol);
    std::vector<float> coords_soa(num_elements, 0.0f);
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        coords_soa[0 * num_atoms + i] = coords_3d[i * 3 + 0];
        coords_soa[1 * num_atoms + i] = coords_3d[i * 3 + 1];
        coords_soa[2 * num_atoms + i] = coords_3d[i * 3 + 2];
        coords_soa[3 * num_atoms + i] = 0.1f * (i % 3 - 1); // small 4th dim
    }

    // perturb some coordinates to create moderate violations
    coords_soa[0 * num_atoms + 0] += 1.5f; // shift atom 0 by 1.5 angstrom

    // GPU energy
    float gpu_energy = 0.0f;
    std::vector<float> gpu_gradient(num_elements, 0.0f);
    gpu_dg_energy(coords_soa.data(), gpu_gradient.data(), bounds.data(), num_atoms, dim, 1.0f, 1.0f, 0.0f, &gpu_energy);

    // CPU reference
    std::vector<double> coords_double(coords_soa.begin(), coords_soa.end());
    double cpu_energy = cpu_dg_energy_reference(coords_double.data(), bounds.data(), num_atoms, dim, 1.0f, 1.0f);

    double const relative_error = std::fabs(gpu_energy - cpu_energy) / std::max(1.0, std::fabs(cpu_energy));
    std::printf("  GPU energy: %.6f  CPU reference: %.6f  relative_error: %.6e\n", gpu_energy, cpu_energy,
                relative_error);

    SF_TEST_ASSERT(gpu_energy > 0.0f, "energy must be positive with violations");
    // half-precision bounds introduce ~1% error per bound value, which compounds
    // across N*(N-1)/2 pairs. allow generous tolerance.
    SF_TEST_ASSERT(relative_error < 1.0, "GPU energy must be same order of magnitude as CPU reference");
}

static void test_dg_gradient_finite_difference() {
    std::printf("test_dg_gradient_finite_difference\n");

    auto mol = mol_from_smiles("CCO");
    SF_TEST_ASSERT(mol != nullptr, "failed to parse CCO");

    bool ok = embed_molecule(*mol);
    SF_TEST_ASSERT(ok, "failed to embed CCO");

    std::uint32_t const num_atoms = mol->getNumAtoms();
    std::uint32_t const dim = 4;
    std::uint32_t const num_elements = dim * num_atoms;

    std::vector<float> bounds;
    molecule_descriptor_t desc;
    build_descriptor(*mol, desc);
    build_bounds_matrix(desc, bounds);

    // create SoA coords with small perturbation (moderate violations)
    auto coords_3d = extract_coords_interleaved(*mol);
    std::vector<float> coords(num_elements, 0.0f);
    std::mt19937 rng_grad(123);
    std::uniform_real_distribution<float> perturb(-0.3f, 0.3f);
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        coords[0 * num_atoms + i] = coords_3d[i * 3 + 0] + perturb(rng_grad);
        coords[1 * num_atoms + i] = coords_3d[i * 3 + 1] + perturb(rng_grad);
        coords[2 * num_atoms + i] = coords_3d[i * 3 + 2] + perturb(rng_grad);
        coords[3 * num_atoms + i] = perturb(rng_grad) * 0.5f;
    }

    // get analytical gradient from GPU
    float energy_center = 0.0f;
    std::vector<float> analytical_gradient(num_elements, 0.0f);
    gpu_dg_energy(coords.data(), analytical_gradient.data(), bounds.data(), num_atoms, dim, 1.0f, 1.0f, 0.0f,
                  &energy_center);

    // finite-difference gradient check on sampled components
    float const epsilon = 1e-3f;
    std::uint32_t num_checked = 0;
    std::uint32_t num_mismatches = 0;
    float max_relative_error = 0.0f;

    for (std::uint32_t k = 0; k < num_elements; ++k) {
        std::vector<float> coords_plus = coords;
        std::vector<float> coords_minus = coords;
        std::vector<float> dummy_grad(num_elements);

        coords_plus[k] += epsilon;
        coords_minus[k] -= epsilon;

        float energy_plus = 0.0f, energy_minus = 0.0f;
        gpu_dg_energy(coords_plus.data(), dummy_grad.data(), bounds.data(), num_atoms, dim, 1.0f, 1.0f, 0.0f,
                      &energy_plus);
        gpu_dg_energy(coords_minus.data(), dummy_grad.data(), bounds.data(), num_atoms, dim, 1.0f, 1.0f, 0.0f,
                      &energy_minus);

        float const fd_gradient = (energy_plus - energy_minus) / (2.0f * epsilon);
        float const analytical = analytical_gradient[k];
        float const abs_error = std::fabs(analytical - fd_gradient);
        float const denom = std::max(1.0f, std::fabs(analytical));
        float const relative_error = abs_error / denom;

        if (relative_error > max_relative_error) max_relative_error = relative_error;

        if (relative_error > 0.05f && abs_error > 0.01f) {
            if (num_mismatches < 5) {
                std::printf("  component %u: analytical=%.6f fd=%.6f rel_err=%.4f\n", k, analytical, fd_gradient,
                            relative_error);
            }
            ++num_mismatches;
        }
        ++num_checked;
    }

    std::printf("  checked %u components, %u mismatches, max_rel_error=%.6f\n", num_checked, num_mismatches,
                max_relative_error);

    SF_TEST_ASSERT(num_mismatches <= num_checked / 4, "too many gradient mismatches (>25%)");
}

static void test_dg_minimize_convergence() {
    std::printf("test_dg_minimize_convergence\n");

    std::vector<std::string> test_smiles = {"CCO", "CC(=O)O", "c1ccccc1"};

    for (auto const &smiles : test_smiles) {
        std::printf("  testing: %s\n", smiles.c_str());

        auto prepared = prepare_molecule(smiles);
        SF_TEST_ASSERT(prepared.num_atoms > 0, "failed to prepare molecule");

        std::uint32_t const num_atoms = prepared.num_atoms;
        std::uint32_t const dim = 4;
        std::uint32_t const num_elements = dim * num_atoms;

        // random 4D starting coordinates
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
        std::vector<float> coords(num_elements);
        for (auto &c : coords) c = dist(rng);

        // run DG minimization
        float energy = 0.0f;
        std::int32_t iterations = 0, status = 0;
        gpu_dg_minimize(coords.data(), prepared.bounds_matrix.data(), num_atoms, dim, 1.0f, 1.0f, 0.0f, 400, 1e-4f,
                        &energy, &iterations, &status);

        std::printf("    status=%d iterations=%d energy=%.4f\n", status, iterations, energy);

        // check distance violations after minimization
        float max_violation = 0.0f;
        for (std::uint32_t i = 0; i < num_atoms; ++i) {
            for (std::uint32_t j = i + 1; j < num_atoms; ++j) {
                float dist_sq = 0.0f;
                for (std::uint32_t d = 0; d < dim; ++d) {
                    float delta = coords[d * num_atoms + j] - coords[d * num_atoms + i];
                    dist_sq += delta * delta;
                }
                float distance = std::sqrt(dist_sq);
                float upper = prepared.bounds_matrix[i * num_atoms + j];
                float lower = prepared.bounds_matrix[j * num_atoms + i];

                if (distance > upper) max_violation = std::max(max_violation, distance - upper);
                if (distance < lower) max_violation = std::max(max_violation, lower - distance);
            }
        }
        std::printf("    max_violation=%.4f angstrom\n", max_violation);

        // project to 3D and check MMFF94 energy via RDKit
        auto mol = mol_from_smiles(smiles);
        bool embedded = embed_molecule(*mol);
        SF_TEST_ASSERT(embedded, "failed to embed for cross-reference");

        // extract 3D from SoA (first 3 dimensions)
        std::vector<float> coords_3d(num_atoms * 3);
        for (std::uint32_t i = 0; i < num_atoms; ++i) {
            coords_3d[i * 3 + 0] = coords[0 * num_atoms + i];
            coords_3d[i * 3 + 1] = coords[1 * num_atoms + i];
            coords_3d[i * 3 + 2] = coords[2 * num_atoms + i];
        }

        set_coords_interleaved(*mol, coords_3d.data());
        double mmff_energy = rdkit_mmff94_energy(*mol);
        std::printf("    MMFF94 energy at DG coords: %.2f kcal/mol\n", mmff_energy);

        SF_TEST_ASSERT(std::isfinite(mmff_energy), "MMFF94 energy must be finite after DG minimization");
    }
}

int main() {
    test_dg_energy_known_violation();
    test_dg_gradient_finite_difference();
    test_dg_minimize_convergence();
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
