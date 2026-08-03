/**
 *  @file test_embedding_determinism.cpp
 *  @brief Asserts that the 4D metric embedding is bitwise reproducible.
 *
 *  The metric prepass seeds every later ETKDG stage, so an order-dependent block
 *  reduction inside it — a shared-memory float @c atomicAdd, say, whose commit order
 *  the scheduler picks and whose addition does not associate — makes the coordinates
 *  of a shipped shard unregenerable.  A dataset that cannot be reproduced byte for
 *  byte cannot be audited, so this test pins the property twice over: the same bounds
 *  matrix embedded in two separate launches must agree, and thousands of identical
 *  copies embedded inside one launch must agree with each other, since they differ
 *  only in which streaming multiprocessor and which neighbours they draw.
 */

#include <cmath>
#include <cstdio>
#include <cstring>

#include <random>
#include <vector>

#include "test_helpers.hpp"

// Defined in test_embedding_determinism_kernels.cu, compiled by nvcc.
extern "C" int gpu_embed_replicated(float const *host_bounds, std::uint32_t num_atoms, std::uint32_t num_molecules,
                                    float *host_coordinates);

/** Builds a satisfiable bounds matrix by bracketing the distances of a random point cloud. */
static std::vector<float> synthetic_bounds(std::uint32_t num_atoms) {
    std::mt19937 generator(7);
    std::uniform_real_distribution<float> spread(-5.0f, 5.0f);
    std::vector<float> positions(static_cast<std::size_t>(num_atoms) * 3);
    for (float &coordinate : positions) coordinate = spread(generator);

    std::vector<float> bounds(static_cast<std::size_t>(num_atoms) * num_atoms, 0.0f);
    for (std::uint32_t first = 0; first < num_atoms; ++first)
        for (std::uint32_t second = 0; second < num_atoms; ++second) {
            if (first == second) continue;
            float const delta_x = positions[first * 3 + 0] - positions[second * 3 + 0];
            float const delta_y = positions[first * 3 + 1] - positions[second * 3 + 1];
            float const delta_z = positions[first * 3 + 2] - positions[second * 3 + 2];
            float const distance = std::sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
            // Upper bounds live above the diagonal, lower bounds below it.
            bounds[first * num_atoms + second] = first < second ? distance * 1.1f : distance * 0.9f;
        }
    return bounds;
}

int main() {
    std::printf("test_embedding_determinism: bitwise reproducibility of the 4D metric embedding\n");

    constexpr std::uint32_t num_atoms_k = 128;
    constexpr std::uint32_t num_molecules_k = 4096;

    std::vector<float> const bounds = synthetic_bounds(num_atoms_k);
    std::size_t const coordinates_per_molecule = static_cast<std::size_t>(num_atoms_k) * 4;
    std::vector<float> first_run(coordinates_per_molecule * num_molecules_k);
    std::vector<float> second_run(first_run.size());

    int const first_status = gpu_embed_replicated(bounds.data(), num_atoms_k, num_molecules_k, first_run.data());
    SF_TEST_ASSERT(first_status == 0, "first embedding launch must succeed");
    int const second_status = gpu_embed_replicated(bounds.data(), num_atoms_k, num_molecules_k, second_run.data());
    SF_TEST_ASSERT(second_status == 0, "second embedding launch must succeed");

    std::size_t const molecule_bytes = coordinates_per_molecule * sizeof(float);
    bool any_nonzero = false;
    for (std::size_t component = 0; component < coordinates_per_molecule; ++component)
        if (first_run[component] != 0.0f) any_nonzero = true;
    SF_TEST_ASSERT(any_nonzero, "the embedding must not be all zeros, or the comparison proves nothing");

    // Reports up to four differing components so a failure names the magnitude, not just a count.
    auto report_divergence = [&](char const *what, std::uint32_t molecule, float const *left, float const *right) {
        std::printf("  first %s divergence at copy %u:\n", what, molecule);
        for (std::size_t component = 0, shown = 0; component < coordinates_per_molecule && shown < 4; ++component)
            if (left[component] != right[component]) {
                std::printf("    component %zu: %.9g vs %.9g\n", component, left[component], right[component]);
                ++shown;
            }
    };

    std::uint32_t across_launches = 0;
    std::uint32_t within_launch = 0;
    std::uint32_t first_across_launches = num_molecules_k;
    std::uint32_t first_within_launch = num_molecules_k;
    for (std::uint32_t molecule = 0; molecule < num_molecules_k; ++molecule) {
        float const *first_copy = first_run.data() + molecule * coordinates_per_molecule;
        float const *second_copy = second_run.data() + molecule * coordinates_per_molecule;
        if (std::memcmp(first_copy, second_copy, molecule_bytes) != 0) {
            ++across_launches;
            if (first_across_launches == num_molecules_k) first_across_launches = molecule;
        }
        if (std::memcmp(first_run.data(), first_copy, molecule_bytes) != 0) {
            ++within_launch;
            if (first_within_launch == num_molecules_k) first_within_launch = molecule;
        }
    }

    std::printf("  atoms: %u, identical copies per launch: %u\n", num_atoms_k, num_molecules_k);
    std::printf("  copies differing between the two launches: %u\n", across_launches);
    std::printf("  copies differing from copy 0 inside one launch: %u\n", within_launch);
    if (first_across_launches != num_molecules_k)
        report_divergence("across-launch", first_across_launches,
                          first_run.data() + first_across_launches * coordinates_per_molecule,
                          second_run.data() + first_across_launches * coordinates_per_molecule);
    if (first_within_launch != num_molecules_k)
        report_divergence("within-launch", first_within_launch, first_run.data(),
                          first_run.data() + first_within_launch * coordinates_per_molecule);

    SF_TEST_ASSERT(across_launches == 0, "two launches over the same bounds must produce identical coordinates");
    SF_TEST_ASSERT(within_launch == 0, "identical molecules in one launch must produce identical coordinates");

    std::printf("\ntest_embedding_determinism: PASSED\n");
    return 0;
}
