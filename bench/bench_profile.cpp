/**
 *  @file bench/bench_profile.cpp
 *  @brief Minimal CPU-prep profile driver: runs SmallField generate_conformers once.
 *
 *  No RDKit CPU baseline, no cross-validation, no warmup — lets perf record focus on prep CPU only.
 */

#include <array>       // `std::array`
#include <chrono>      // `std::chrono::steady_clock`, `std::chrono::duration`
#include <fstream>     // `std::ifstream`
#include <string>      // `std::string`, `std::getline`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <cuda_fp16.h>
#include <fmt/format.h>

#include "smallfield/api/smallfield.hpp"

#include "bench/bench_arguments.hpp"

namespace sfapi = smallfield::api;
namespace sfbench = smallfield::bench;

/** Name printed on every throughput line so a concatenated CI log stays attributable. */
static constexpr char const *executable_name_k = "bench_profile";

/**
 *  @brief Ceiling on `[conformers_per_molecule]`.
 *
 *  The coordinate buffer is `molecules × conformers × max_atoms × 3` halves, so an unbounded count turns a typo
 *  into a multi-terabyte `resize` and a `std::bad_alloc` with no hint of which token caused it.
 */
static constexpr std::uint32_t max_conformers_per_molecule_k = 1024;

/** @brief Ceiling on the host coordinate buffer, refused before allocation rather than during it. */
static constexpr std::size_t max_coordinate_bytes_k = 64ull << 30;

int main(int argc, char **argv) {
    char const *path = argc > 1 ? argv[1] : "/tmp/chembl_240k.smi";
    std::uint32_t conformers_per_molecule = 1;
    if (argc > 2) {
        auto const status = sfbench::parse_number(std::string_view {argv[2]}, 1u, max_conformers_per_molecule_k,
                                                  conformers_per_molecule);
        if (status != sfbench::number_parse_t::parsed_k) {
            fmt::println(stderr, "{}: conformers_per_molecule '{}' {} (accepted range 1..{})", executable_name_k,
                         argv[2], sfbench::describe_number_parse(status), max_conformers_per_molecule_k);
            fmt::println(stderr, "usage: {} <corpus.smi> [conformers_per_molecule]", argv[0]);
            return 2;
        }
    }

    // A corpus that never opened must not read as an empty corpus: `Loaded 0 SMILES` followed by a clean exit
    // is indistinguishable from a successful run over nothing.
    std::ifstream input_stream(path);
    if (!input_stream) {
        fmt::println(stderr, "{}: cannot open corpus '{}'", executable_name_k, path);
        return 2;
    }

    std::vector<std::string> storage;
    std::string line;
    while (std::getline(input_stream, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto separator_index = line.find_first_of(" \t");
        if (separator_index != std::string::npos) line.resize(separator_index);
        if (!line.empty()) storage.push_back(std::move(line));
    }
    fmt::println("Loaded {} SMILES from {}", fmt::group_digits(storage.size()), path);
    if (storage.empty()) {
        fmt::println(stderr, "{}: corpus '{}' holds no SMILES", executable_name_k, path);
        return 2;
    }

    std::vector<std::string_view> smiles_views;
    smiles_views.reserve(storage.size());
    for (auto const &smiles : storage) smiles_views.emplace_back(smiles);

    std::uint32_t const max_atoms = smallfield::types::max_atoms_per_molecule_k;
    fmt::println("Conformers/molecule: {}", conformers_per_molecule);

    std::size_t const total_conformers = storage.size() * conformers_per_molecule;
    std::size_t const coordinate_bytes = total_conformers * max_atoms * 3 * sizeof(__half);
    if (coordinate_bytes > max_coordinate_bytes_k) {
        fmt::println(stderr,
                     "{}: coordinate buffer would take {:.1f} GB for {} molecules × {} conformers, "
                     "over the {:.1f} GB cap",
                     executable_name_k, static_cast<double>(coordinate_bytes) / 1.0e9,
                     fmt::group_digits(storage.size()), conformers_per_molecule,
                     static_cast<double>(max_coordinate_bytes_k) / 1.0e9);
        return 2;
    }

    std::vector<__half> coords(total_conformers * max_atoms * 3);
    std::vector<float> energies(total_conformers);
    std::vector<sfapi::conformer_status_t> status(total_conformers);

    sfapi::context_t gpu_context;
    sfapi::config_t config;
    config.num_conformers = conformers_per_molecule;

    auto generation_start = std::chrono::steady_clock::now();
    [[maybe_unused]] auto const batch_status = gpu_context.generate_conformers(smiles_views, config, max_atoms, coords,
                                                                               energies, status);
    auto generation_end = std::chrono::steady_clock::now();
    double milliseconds = std::chrono::duration<double, std::milli>(generation_end - generation_start).count();
    // Attempted throughput counts molecules that never reached the GPU, so a corpus that rejects
    // more looks faster than it is. Report produced conformers alongside it, and break the
    // failures down by reason so a rejection class cannot grow unnoticed.
    std::uint64_t ok = 0;
    std::array<std::uint64_t, 8> by_status {};
    for (auto status_code : status) {
        if (status_code == sfapi::conformer_status_t::success_k) ++ok;
        auto const code = static_cast<std::size_t>(status_code);
        if (code < by_status.size()) ++by_status[code];
    }
    fmt::println("{}: wall {:.1f} ms, mol/s {:.1f}, conf/s {:.1f}, succeeded {} / {}", executable_name_k, milliseconds,
                 static_cast<double>(storage.size()) * 1000.0 / milliseconds,
                 static_cast<double>(ok) * 1000.0 / milliseconds, fmt::group_digits(ok),
                 fmt::group_digits(status.size()));

    constexpr std::array<char const *, 8> status_names {
        "success",        "stereo_failure",     "mmff_unconverged",    "invalid_smiles",
        "too_many_atoms", "preprocess_timeout", "unsupported_element", "other"};
    for (std::size_t code = 1; code < by_status.size(); ++code)
        if (by_status[code] > 0)
            fmt::println("  {:<20} {:>12}  {:.3f} %", status_names[code], fmt::group_digits(by_status[code]),
                         100.0 * static_cast<double>(by_status[code]) / static_cast<double>(status.size()));

    return 0;
}
