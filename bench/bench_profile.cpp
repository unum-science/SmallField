/**
 *  @file bench/bench_profile.cpp
 *  @brief Minimal CPU-prep profile driver: runs SmallField generate_conformers once.
 *
 *  No RDKit CPU baseline, no cross-validation, no warmup — lets perf record focus on prep CPU only.
 */

#include <cstdio> // `printf`

#include <array>   // `std::array`
#include <chrono>  // `std::chrono::steady_clock`, `std::chrono::duration`
#include <fstream> // `std::ifstream`
#include <string>  // `std::string`, `std::getline`
#include <vector>  // `std::vector`

#include <cuda_fp16.h>

#include "smallfield/api/smallfield.hpp"

namespace sfapi = smallfield::api;

int main(int argc, char **argv) {
    char const *path = argc > 1 ? argv[1] : "/tmp/chembl_240k.smi";
    std::uint32_t const n_conf_arg = argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 1;
    std::vector<std::string> storage;
    std::ifstream input_stream(path);
    std::string line;
    while (std::getline(input_stream, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto separator_index = line.find_first_of(" \t");
        if (separator_index != std::string::npos) line.resize(separator_index);
        if (!line.empty()) storage.push_back(std::move(line));
    }
    std::printf("Loaded %zu SMILES\n", storage.size());
    std::vector<std::string_view> smiles_views;
    smiles_views.reserve(storage.size());
    for (auto const &smiles : storage) smiles_views.emplace_back(smiles);

    std::uint32_t const max_atoms = smallfield::types::max_atoms_per_molecule_k;
    std::uint32_t const n_conf = n_conf_arg;
    std::printf("Conformers/molecule: %u\n", n_conf);
    std::vector<__half> coords(storage.size() * n_conf * max_atoms * 3);
    std::vector<float> energies(storage.size() * n_conf);
    std::vector<sfapi::conformer_status_t> status(storage.size() * n_conf);

    sfapi::context_t gpu_context;
    sfapi::config_t config;
    config.num_conformers = n_conf;

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
    std::printf("Wall %.1f ms, mol/s %.1f, conf/s %.1f, succeeded %lu / %zu\n", milliseconds,
                storage.size() * 1000.0 / milliseconds, ok * 1000.0 / milliseconds, (unsigned long)ok, status.size());

    constexpr std::array<char const *, 8> status_names {
        "success",        "stereo_failure",     "mmff_unconverged",    "invalid_smiles",
        "too_many_atoms", "preprocess_timeout", "unsupported_element", "other"};
    for (std::size_t code = 1; code < by_status.size(); ++code)
        if (by_status[code] > 0)
            std::printf("  %-20s %10lu  %5.2f %%\n", status_names[code], (unsigned long)by_status[code],
                        100.0 * static_cast<double>(by_status[code]) / static_cast<double>(status.size()));

    return 0;
}
