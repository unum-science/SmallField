/**
 *  @file bench/bench_profile.cpp
 *  @brief Minimal CPU-prep profile driver: runs SmallField generate_conformers once.
 *
 *  No RDKit CPU baseline, no cross-validation, no warmup — lets perf record focus on prep CPU only.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstdio>  // `stderr`

#include <array>       // `std::array`
#include <chrono>      // `std::chrono::steady_clock`, `std::chrono::duration`
#include <limits>      // `std::numeric_limits`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <cuda_fp16.h>
#include <fmt/format.h>

#include "smallfield/api/smallfield.hpp"

#include "bench/bench_arguments.hpp"
#include "bench/bench_reporting.hpp"

namespace sfapi = smallfield::api;
namespace sfbench = smallfield::bench;

/** Name printed on every throughput line so a concatenated CI log stays attributable. */
static constexpr char const *executable_name_k = "bench_profile";

/**
 *  @brief Ceiling on `--conformers`.
 *
 *  The coordinate buffer is `molecules × conformers × max_atoms × 3` halves, so an unbounded count turns a typo
 *  into a multi-terabyte `resize` and a `std::bad_alloc` with no hint of which token caused it.
 */
static constexpr std::uint32_t max_conformers_per_molecule_k = 1024;

/** Ceiling on the host coordinate buffer, refused before allocation rather than during it. */
static constexpr std::size_t max_coordinate_bytes_k = 64ull << 30;

#pragma region Command Line

/** Everything this driver takes from the command line. */
struct profile_arguments_t {
    /** Corpus to profile over, from `--smi` or from `SMALLFIELD_TEST_CORPUS`. There is no default. */
    std::string_view smiles_path;
    /** Conformers requested per molecule. */
    std::uint32_t conformers_per_molecule = 1;
};

using sfbench::arguments_parse_t;

/** Print the one usage line, naming every flag this driver understands. */
static void print_usage_(char const *invocation) {
    fmt::println(stderr, "usage: {} --smi PATH [--conformers N]", invocation);
}

/**
 *  @brief Fill @p arguments from `argv`, or explain on `stderr` why the command line was refused.
 *
 *  An unrecognised flag and a missing value are both refusals: a profile run under a mistyped flag that
 *  silently used the defaults would report a corpus nobody asked for.
 */
static arguments_parse_t parse_arguments_(int argc, char **argv, profile_arguments_t &arguments) {
    for (sfbench::command_line_t line {executable_name_k, argc, argv}; line.advance();) {
        arguments_parse_t status = arguments_parse_t::accepted_k;
        if (line.flag() == "--smi") status = line.take_text(arguments.smiles_path);
        else if (line.flag() == "--conformers")
            status = line.take_number(max_conformers_per_molecule_k, arguments.conformers_per_molecule);
        else {
            print_usage_(argv[0]);
            status = line.refuse_unknown();
        }
        if (status != arguments_parse_t::accepted_k) return arguments_parse_t::refused_k;
    }
    if (arguments.smiles_path.empty()) arguments.smiles_path = sfbench::corpus_path_from_environment();
    return arguments_parse_t::accepted_k;
}

#pragma endregion Command Line

int main(int argc, char **argv) {
    profile_arguments_t arguments;
    if (parse_arguments_(argc, argv, arguments) != arguments_parse_t::accepted_k) return 2;

    fmt::println("{}: SmallField conformer generation, one pass, no baseline", executable_name_k);

    // A corpus that never opened must not read as an empty corpus: `Loaded 0 SMILES` followed by a clean exit
    // is indistinguishable from a successful run over nothing.
    std::vector<std::string> storage;
    auto const load_status = sfbench::load_corpus_into(arguments.smiles_path, std::numeric_limits<std::size_t>::max(),
                                                       storage);
    if (load_status != sfbench::corpus_load_t::loaded_k) {
        fmt::println(stderr, "{}: corpus '{}' {}", executable_name_k, arguments.smiles_path,
                     sfbench::describe_corpus_load(load_status));
        print_usage_(argv[0]);
        return 2;
    }
    fmt::println("Loaded {} SMILES from {}", fmt::group_digits(storage.size()), arguments.smiles_path);

    std::vector<std::string_view> smiles_views;
    smiles_views.reserve(storage.size());
    for (auto const &smiles : storage) smiles_views.emplace_back(smiles);

    std::uint32_t const max_atoms = smallfield::types::max_atoms_per_molecule_k;
    fmt::println("Conformers/molecule: {}", arguments.conformers_per_molecule);

    std::size_t const total_conformers = storage.size() * arguments.conformers_per_molecule;
    std::size_t const coordinate_bytes = total_conformers * max_atoms * 3 * sizeof(__half);
    if (coordinate_bytes > max_coordinate_bytes_k) {
        fmt::println(
            stderr,
            "{}: coordinate buffer would take {:.1f} GB for {} molecules × {} conformers, " "over the {:.1f} GB cap",
            executable_name_k, static_cast<double>(coordinate_bytes) / 1.0e9, fmt::group_digits(storage.size()),
            arguments.conformers_per_molecule, static_cast<double>(max_coordinate_bytes_k) / 1.0e9);
        return 2;
    }

    std::vector<__half> coords(total_conformers * max_atoms * 3);
    std::vector<float> energies(total_conformers);
    std::vector<sfapi::conformer_status_t> status(total_conformers);

    sfapi::context_t gpu_context;
    sfapi::config_t config;
    config.num_conformers = arguments.conformers_per_molecule;

    auto generation_start = std::chrono::steady_clock::now();
    [[maybe_unused]] auto const batch_status = gpu_context.generate_conformers(smiles_views, config, max_atoms, coords,
                                                                               energies, status);
    auto generation_end = std::chrono::steady_clock::now();
    double milliseconds = std::chrono::duration<double, std::milli>(generation_end - generation_start).count();

    // Attempted throughput counts molecules that never reached the GPU, so a corpus that rejects
    // more looks faster than it is. Report produced conformers alongside it, and break the
    // failures down by reason so a rejection class cannot grow unnoticed.
    auto const histogram = sfbench::status_histogram_of(status);
    std::uint64_t const succeeded = sfbench::successful_conformers_of(histogram);
    fmt::println("{}: wall {:.1f} ms, mol/s {:.1f}, conf/s {:.1f}, succeeded {} / {}", executable_name_k, milliseconds,
                 static_cast<double>(storage.size()) * 1000.0 / milliseconds,
                 static_cast<double>(succeeded) * 1000.0 / milliseconds, fmt::group_digits(succeeded),
                 fmt::group_digits(status.size()));
    sfbench::print_status_histogram(histogram, status.size());

    return 0;
}
