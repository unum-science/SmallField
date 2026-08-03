// Minimal CPU-prep profile driver: just runs SmallField generate_conformers once on 241K.
// No RDKit CPU baseline, no cross-val, no warmup. Lets perf record focus on prep CPU only.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "smallfield/api/smallfield.hpp"

namespace sfapi = smallfield::api;

int main(int argc, char **argv) {
    char const *path = argc > 1 ? argv[1] : "/tmp/chembl_240k.smi";
    std::uint32_t const n_conf_arg = argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 1;
    std::vector<std::string> storage;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto sp = line.find_first_of(" \t");
        if (sp != std::string::npos) line.resize(sp);
        if (!line.empty()) storage.push_back(std::move(line));
    }
    std::printf("Loaded %zu SMILES\n", storage.size());
    std::vector<std::string_view> smiles_views;
    smiles_views.reserve(storage.size());
    for (auto const &s : storage) smiles_views.emplace_back(s);

    std::uint32_t const max_atoms = 128;
    std::uint32_t const n_conf = n_conf_arg;
    std::printf("Conformers/molecule: %u\n", n_conf);
    std::vector<__half> coords(storage.size() * n_conf * max_atoms * 3);
    std::vector<float> energies(storage.size() * n_conf);
    std::vector<sfapi::conformer_status_t> status(storage.size() * n_conf);

    sfapi::context_t ctx;
    sfapi::config_t config;
    config.num_conformers = n_conf;

    auto t0 = std::chrono::steady_clock::now();
    ctx.generate_conformers(smiles_views, config, max_atoms, coords, energies, status);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::uint64_t ok = 0;
    for (auto status_code : status)
        if (status_code == sfapi::conformer_status_t::success) ++ok;
    std::printf("Wall %.1f ms, mol/s %.1f, succeeded %lu / %zu\n", ms, storage.size() * 1000.0 / ms, (unsigned long)ok,
                status.size());

    return 0;
}
