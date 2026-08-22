/**
 *  @file test/cuda/test_embedding_determinism.cpp
 *  @brief Assert that the 4D metric embedding is bitwise reproducible.
 *
 *  The metric prepass seeds every later ETKDG stage, so an order-dependent block reduction inside it -- a
 *  shared-memory float `atomicAdd` whose commit order the scheduler picks and whose addition does not
 *  associate -- makes a shipped shard's coordinates unregenerable, and an unreproducible dataset cannot be
 *  audited. The property is pinned twice: one bounds matrix embedded in two separate launches must agree,
 *  and thousands of identical copies inside one launch must agree with each other, differing only in which
 *  streaming multiprocessor and which neighbours they draw.
 */

#include <cmath>   // `std::sqrt`
#include <cstring> // `std::memcmp`

#include <algorithm>   // `std::any_of`
#include <random>      // `std::mt19937`, `std::uniform_real_distribution`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include "smallfield/types/views.hpp"

#include "test/cuda/test_embedding_determinism_kernels.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

using sftest::gpu_embed_replicated;

static constexpr std::string_view test_name_k = "test_embedding_determinism";

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

/**
 *  @brief How many replicated copies diverged, and which one diverged first. `count` is the "nothing found"
 *  answer, so `first_copy` never doubles as one: an index standing in for its own absence is read correctly
 *  once and misread thereafter.
 */
struct divergence_t {
    std::uint32_t count = 0;
    std::uint32_t first_copy = 0;
};

int main() {
    fmt::println("{}: bitwise reproducibility of the 4D metric embedding", test_name_k);

    constexpr std::uint32_t num_atoms_k = 128;
    constexpr std::uint32_t num_molecules_k = 4096;

    std::vector<float> const bounds = synthetic_bounds(num_atoms_k);
    sft::packed_matrix<float const> const bounds_matrix {bounds.data(), num_atoms_k, num_atoms_k};
    std::size_t const coordinates_per_molecule = static_cast<std::size_t>(num_atoms_k) * sft::embedding_dimensions_k;
    std::vector<float> first_run(coordinates_per_molecule * num_molecules_k);
    std::vector<float> second_run(first_run.size());

    int const first_status = gpu_embed_replicated(bounds_matrix, num_molecules_k, first_run);
    sf_verify_(first_status == 0, "first embedding launch must succeed");
    int const second_status = gpu_embed_replicated(bounds_matrix, num_molecules_k, second_run);
    sf_verify_(second_status == 0, "second embedding launch must succeed");

    std::size_t const molecule_bytes = coordinates_per_molecule * sizeof(float);
    bool const any_nonzero = std::any_of(first_run.begin(), first_run.begin() + coordinates_per_molecule,
                                         [](float component) { return component != 0.0f; });
    sf_verify_(any_nonzero, "the embedding must not be all zeros, or the comparison proves nothing");

    // Reports up to four differing components so a failure names the magnitude, not just a count.
    auto report_divergence = [&](std::string_view scope, std::uint32_t molecule, float const *left, float const *right) {
        std::fflush(stdout);
        for (std::size_t component = 0, shown = 0; component < coordinates_per_molecule && shown < 4; ++component)
            if (left[component] != right[component]) {
                fmt::println(stderr, "MISMATCH embedding scope={} copy={} component={} field=coordinate left={} right={}",
                             scope, molecule, component, left[component], right[component]);
                ++shown;
            }
    };

    divergence_t across_launches;
    divergence_t within_launch;
    for (std::uint32_t molecule = 0; molecule < num_molecules_k; ++molecule) {
        float const *first_copy = first_run.data() + molecule * coordinates_per_molecule;
        float const *second_copy = second_run.data() + molecule * coordinates_per_molecule;
        if (std::memcmp(first_copy, second_copy, molecule_bytes) != 0) {
            if (across_launches.count == 0) across_launches.first_copy = molecule;
            ++across_launches.count;
        }
        if (std::memcmp(first_run.data(), first_copy, molecule_bytes) != 0) {
            if (within_launch.count == 0) within_launch.first_copy = molecule;
            ++within_launch.count;
        }
    }

    fmt::println("  atoms: {}, identical copies per launch: {}", num_atoms_k, fmt::group_digits(num_molecules_k));
    fmt::println("  copies differing between the two launches: {}", fmt::group_digits(across_launches.count));
    fmt::println("  copies differing from copy 0 inside one launch: {}", fmt::group_digits(within_launch.count));
    if (across_launches.count != 0)
        report_divergence("across_launch", across_launches.first_copy,
                          first_run.data() + across_launches.first_copy * coordinates_per_molecule,
                          second_run.data() + across_launches.first_copy * coordinates_per_molecule);
    if (within_launch.count != 0)
        report_divergence("within_launch", within_launch.first_copy, first_run.data(),
                          first_run.data() + within_launch.first_copy * coordinates_per_molecule);

    sf_verify_(across_launches.count == 0, "two launches over the same bounds must produce identical coordinates");
    sf_verify_(within_launch.count == 0, "identical molecules in one launch must produce identical coordinates");

    return sftest::report_pass(test_name_k);
}
