/**
 *  @file test/api/test_seed_determinism.cpp
 *  @brief Assert that a fixed `random_seed` reproduces conformers bit for bit.
 *
 *  Two `generate_conformers` calls in one process, same input, same seed, default OpenMP thread count. The
 *  threading matters: preprocess hands molecules to host threads with a dynamic schedule and then
 *  atom-count-sorts them per device, so a molecule's offset inside the device batch is a function of thread
 *  timing. Seeding the pipeline off that offset made the pipeline irreproducible in a way that pinning to one
 *  thread would hide, which is why this test deliberately leaves OpenMP at its default width and uses enough
 *  molecules for the dynamic chunks to interleave differently between the two calls.
 */

#include <cstdint> // `std::uint32_t`
#include <cstring> // `std::memcmp`

#include <span>        // `std::span`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <cuda_fp16.h>

#include "smallfield/api/smallfield.hpp"

#include "test/test_rdkit_reference_helpers.hpp"

namespace sfapi = smallfield::api;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_seed_determinism";

/**
 *  @brief Build a corpus of distinct, valid SMILES large enough to expose scheduling jitter.
 *
 *  Cycles a handful of scaffolds and grows an alkyl tail so no two entries are the same
 *  molecule and the atom counts spread across several of the pipeline's atom-count bins.
 *
 *  @param[in] num_molecules How many SMILES to produce.
 *  @param[out] corpus_out Filled with the generated strings; cleared first.
 */
static void fill_corpus_into(std::uint32_t num_molecules, std::vector<std::string> &corpus_out) {
    static char const *const scaffolds[] = {"c1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCNCC1", "c1ccsc1", "C1COCCN1"};
    static char const *const heads[] = {"CC", "CCO", "CCN", "CC(=O)", "CCC(N)", "COC"};
    constexpr std::uint32_t num_scaffolds = sizeof(scaffolds) / sizeof(scaffolds[0]);
    constexpr std::uint32_t num_heads = sizeof(heads) / sizeof(heads[0]);

    corpus_out.clear();
    corpus_out.reserve(num_molecules);
    for (std::uint32_t molecule = 0; molecule < num_molecules; ++molecule) {
        std::string smiles = heads[molecule % num_heads];
        std::uint32_t const tail_length = 1u + (molecule / (num_heads * num_scaffolds)) % 6u;
        for (std::uint32_t carbon = 0; carbon < tail_length; ++carbon) smiles += 'C';
        smiles += scaffolds[(molecule / num_heads) % num_scaffolds];
        corpus_out.push_back(smiles);
    }
}

int main() {
    fmt::println("test_seed_determinism: fixed random_seed reproduces conformers bit for bit");

    constexpr std::uint32_t num_molecules = 768;
    constexpr std::uint32_t num_conformers = 3;
    constexpr std::uint32_t max_atoms = 128;

    std::vector<std::string> corpus;
    fill_corpus_into(num_molecules, corpus);
    std::vector<std::string_view> smiles_views(corpus.begin(), corpus.end());

    sfapi::config_t config;
    config.num_conformers = num_conformers;
    config.max_stereo_attempts = 20;
    config.random_seed = 4242;

    std::size_t const total_conformers = static_cast<std::size_t>(num_molecules) * num_conformers;
    std::size_t const total_coordinates = total_conformers * max_atoms * 3;

    std::vector<__half> coordinates_first(total_coordinates), coordinates_second(total_coordinates);
    std::vector<float> energies_first(total_conformers), energies_second(total_conformers);
    std::vector<sfapi::conformer_status_t> status_first(total_conformers), status_second(total_conformers);

    int const device_id = 0;
    sfapi::context_t context(std::span<int const>(&device_id, 1));

    [[maybe_unused]] auto const first_status = context.generate_conformers(
        smiles_views, config, max_atoms, coordinates_first, energies_first, status_first);
    [[maybe_unused]] auto const second_status = context.generate_conformers(
        smiles_views, config, max_atoms, coordinates_second, energies_second, status_second);

    std::uint32_t num_successful = 0;
    for (auto code : status_first)
        if (code == sfapi::conformer_status_t::success_k) ++num_successful;
    fmt::println("  {} / {} conformers succeeded on the first run", fmt::group_digits(num_successful),
                 fmt::group_digits(total_conformers));
    if (num_successful == 0)
        return sftest::report_fail(test_name_k,
                                   "the corpus produced no conformers, so the comparison proves nothing");

    std::uint32_t status_mismatches = 0, energy_mismatches = 0, coordinate_mismatches = 0;
    for (std::size_t slot = 0; slot < total_conformers; ++slot) {
        if (status_first[slot] != status_second[slot]) ++status_mismatches;
        if (std::memcmp(&energies_first[slot], &energies_second[slot], sizeof(float)) != 0) ++energy_mismatches;
    }
    for (std::size_t element = 0; element < total_coordinates; ++element)
        if (std::memcmp(&coordinates_first[element], &coordinates_second[element], sizeof(__half)) != 0)
            ++coordinate_mismatches;

    fmt::println("  status mismatches: {}, energy mismatches: {}, coordinate mismatches: {}",
                 fmt::group_digits(status_mismatches), fmt::group_digits(energy_mismatches),
                 fmt::group_digits(coordinate_mismatches));

    if (status_mismatches != 0 || energy_mismatches != 0 || coordinate_mismatches != 0)
        return sftest::report_fail(
            test_name_k, "two runs at seed {} diverged: status={} energy={} coordinate={} mismatches",
            config.random_seed, status_mismatches, energy_mismatches, coordinate_mismatches);

    return sftest::report_pass(test_name_k);
}
