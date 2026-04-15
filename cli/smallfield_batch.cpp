/**
 *  @file smallfield_batch.cpp
 *  @brief Restartable CLI for bulk SmallField conformer generation over
 *         a folder of Parquet input shards.
 *
 *  Pipeline (per shard):
 *    1. Read SMILES column into a std::vector<std::string>.
 *    2. Call context_t::generate_conformers to run the GPU pipeline.
 *    3. Partition outputs by status; re-run failures through RDKit ETKDG + MMFF.
 *    4. Remaining failures → NULL coordinates/energies, solver=2.
 *    5. Write output Parquet shard.
 *
 *  Restart: deterministic input→output filename mapping; any output whose
 *  row count equals `input_rows * num_conformers` is skipped.
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_fp16.h>

#include "smallfield/api/smallfield.hpp"

#include "parquet_reader.hpp"
#include "parquet_writer.hpp"
#include "rdkit_fallback.hpp"
#include "shard_inventory.hpp"

namespace fs = std::filesystem;
namespace sf = smallfield;
namespace sfapi = smallfield::api;
namespace sfcli = smallfield::cli;

namespace {

struct cli_config_t {
    fs::path input_dir;
    fs::path output_dir;
    std::string smiles_column = "smiles";
    std::uint32_t num_conformers = 3;
    std::uint32_t max_atoms = 128;
    std::vector<int> gpu_ids;       // empty = all visible
    int rdkit_fallback_threads = 0; // 0 = omp_get_max_threads() at call time
    bool resume = true;
    std::uint64_t random_seed = 42;
};

void print_usage() {
    std::fprintf(stderr, R"(Usage: smallfield_batch --input DIR --output DIR [options]
  --input DIR                 folder of *.parquet input shards
  --output DIR                folder to write *_3D.parquet outputs
  --smiles-column NAME        input column name (default: smiles)
  --num-conformers K          conformers per molecule (default: 3)
  --max-atoms N               per-molecule atom cap (default: 128)
  --gpus a,b,c                GPU device IDs (default: all visible)
  --rdkit-fallback-threads N  OpenMP threads for RDKit fallback (default: max)
  --restart                   overwrite already-complete shards
  --seed N                    RNG seed for both GPU and RDKit fallback
)");
}

bool parse_int_list(char const *spec, std::vector<int> &out) {
    out.clear();
    while (*spec) {
        char *end = nullptr;
        long value = std::strtol(spec, &end, 10);
        if (end == spec) return false;
        out.push_back(static_cast<int>(value));
        spec = end;
        if (*spec == ',') ++spec;
    }
    return true;
}

bool parse_cli(int argc, char **argv, cli_config_t &config) {
    for (int argument_index = 1; argument_index < argc; ++argument_index) {
        std::string_view flag = argv[argument_index];
        auto require_value = [&](char const *flag_name) -> char const * {
            if (argument_index + 1 >= argc) {
                std::fprintf(stderr, "--%s requires a value\n", flag_name);
                return nullptr;
            }
            return argv[++argument_index];
        };
        if (flag == "--input") {
            auto value = require_value("input");
            if (!value) return false;
            config.input_dir = value;
        }
        else if (flag == "--output") {
            auto value = require_value("output");
            if (!value) return false;
            config.output_dir = value;
        }
        else if (flag == "--smiles-column") {
            auto value = require_value("smiles-column");
            if (!value) return false;
            config.smiles_column = value;
        }
        else if (flag == "--num-conformers") {
            auto value = require_value("num-conformers");
            if (!value) return false;
            config.num_conformers = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
        }
        else if (flag == "--max-atoms") {
            auto value = require_value("max-atoms");
            if (!value) return false;
            config.max_atoms = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
        }
        else if (flag == "--gpus") {
            auto value = require_value("gpus");
            if (!value) return false;
            if (!parse_int_list(value, config.gpu_ids)) return false;
        }
        else if (flag == "--rdkit-fallback-threads") {
            auto value = require_value("rdkit-fallback-threads");
            if (!value) return false;
            config.rdkit_fallback_threads = static_cast<int>(std::strtol(value, nullptr, 10));
        }
        else if (flag == "--restart") { config.resume = false; }
        else if (flag == "--resume") { config.resume = true; }
        else if (flag == "--seed") {
            auto value = require_value("seed");
            if (!value) return false;
            config.random_seed = std::strtoull(value, nullptr, 10);
        }
        else if (flag == "--help" || flag == "-h") {
            print_usage();
            return false;
        }
        else {
            std::fprintf(stderr, "unknown flag: %.*s\n", static_cast<int>(flag.size()), flag.data());
            return false;
        }
    }
    if (config.input_dir.empty() || config.output_dir.empty()) {
        print_usage();
        return false;
    }
    return true;
}

void process_shard(sfapi::context_t &gpu_context, cli_config_t const &cli_config,
                   sfcli::shard_pair_t const &shard_pair) {
    std::printf("[shard] %s → %s (%zu rows)\n", shard_pair.input_path.filename().string().c_str(),
                shard_pair.output_path.filename().string().c_str(),
                static_cast<std::size_t>(shard_pair.input_num_rows));
    std::fflush(stdout);

    auto const start_time = std::chrono::high_resolution_clock::now();

    sfcli::shard_data_t shard_data;
    if (!sfcli::read_shard(shard_pair.input_path, cli_config.smiles_column, shard_data)) {
        std::fprintf(stderr, "[shard] SKIP (read failed): %s\n", shard_pair.input_path.string().c_str());
        return;
    }

    std::uint64_t const num_input_rows = shard_data.smiles.size();
    std::uint64_t const total_conformers = num_input_rows * static_cast<std::uint64_t>(cli_config.num_conformers);
    std::size_t const coordinate_stride = static_cast<std::size_t>(cli_config.max_atoms) * 3u;
    std::size_t const coordinate_total = total_conformers * coordinate_stride;

    std::vector<__half> coordinates(coordinate_total);
    std::vector<float> energies(total_conformers, 0.0f);
    std::vector<sfapi::conformer_status_t> status(total_conformers, sfapi::conformer_status_t::mmff_unconverged);

    std::vector<std::string_view> smiles_views;
    smiles_views.reserve(num_input_rows);
    for (auto const &smiles : shard_data.smiles) smiles_views.emplace_back(smiles);

    sfapi::config_t generation_config;
    generation_config.num_conformers = cli_config.num_conformers;
    generation_config.random_seed = cli_config.random_seed;

    gpu_context.generate_conformers(smiles_views, generation_config, cli_config.max_atoms,
                                    std::span<__half>(coordinates), std::span<float>(energies),
                                    std::span<sfapi::conformer_status_t>(status));

    // Partition failures for RDKit fallback.
    std::vector<std::uint8_t> solver_codes(total_conformers, 0);
    std::vector<std::uint8_t> status_codes(total_conformers, 0);
    std::vector<std::uint32_t> retry_indices;
    retry_indices.reserve(num_input_rows / 32);
    for (std::uint64_t molecule_index = 0; molecule_index < num_input_rows; ++molecule_index) {
        bool molecule_has_failure = false;
        for (std::uint32_t conformer_index = 0; conformer_index < cli_config.num_conformers; ++conformer_index) {
            std::uint64_t const flat_index = molecule_index * cli_config.num_conformers + conformer_index;
            status_codes[flat_index] = static_cast<std::uint8_t>(status[flat_index]);
            if (status[flat_index] == sfapi::conformer_status_t::success) { solver_codes[flat_index] = 0; }
            else {
                solver_codes[flat_index] = 2;
                molecule_has_failure = true;
            }
        }
        if (molecule_has_failure) retry_indices.push_back(static_cast<std::uint32_t>(molecule_index));
    }

    std::vector<std::uint8_t> num_atoms_per_molecule(num_input_rows, 0);

    if (!retry_indices.empty()) {
        sfcli::rdkit_fallback_inputs_t fallback_inputs {
            std::span<std::string const>(shard_data.smiles),
            std::span<std::uint32_t const>(retry_indices),
            cli_config.num_conformers,
            cli_config.max_atoms,
            cli_config.rdkit_fallback_threads,
            cli_config.random_seed,
        };
        sfcli::rdkit_fallback_outputs_t fallback_outputs {
            std::span<std::uint8_t>(status_codes),
            std::span<std::uint8_t>(solver_codes),
            std::span<std::uint8_t>(num_atoms_per_molecule),
            std::span<__half>(coordinates),
            std::span<float>(energies),
        };
        auto const num_recovered = sfcli::rdkit_fallback(fallback_inputs, fallback_outputs);
        std::printf("[shard] rdkit fallback: %zu → %zu recovered\n", retry_indices.size(),
                    static_cast<std::size_t>(num_recovered));
    }

    sfcli::writer_inputs_t writer_inputs {
        shard_pair.input_path.stem().string(),
        std::span<std::uint64_t const>(shard_data.input_row_indices),
        cli_config.num_conformers,
        cli_config.max_atoms,
        std::span<std::uint8_t const>(num_atoms_per_molecule),
        std::span<std::uint8_t const>(status_codes),
        std::span<std::uint8_t const>(solver_codes),
        std::span<__half const>(coordinates),
        std::span<float const>(energies),
    };

    if (!sfcli::write_shard(shard_pair.output_path, writer_inputs)) {
        std::fprintf(stderr, "[shard] WRITE FAILED: %s\n", shard_pair.output_path.string().c_str());
        return;
    }

    auto const end_time = std::chrono::high_resolution_clock::now();
    double const elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::printf("[shard] done %.1f ms (%.1f mol/s)\n", elapsed_ms, num_input_rows * 1000.0 / elapsed_ms);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char **argv) {
    cli_config_t config;
    if (!parse_cli(argc, argv, config)) return 2;

    auto shard_pairs = sfcli::inventory_shards(config.input_dir, config.output_dir, config.num_conformers,
                                               config.resume);
    if (shard_pairs.empty()) {
        std::fprintf(stderr, "no input shards found in %s\n", config.input_dir.string().c_str());
        return 1;
    }

    std::uint64_t num_pending_shards = 0;
    for (auto const &shard_pair : shard_pairs)
        if (!shard_pair.already_complete) ++num_pending_shards;

    std::printf("smallfield_batch: %zu shards, %zu pending (resume=%d)\n", shard_pairs.size(),
                static_cast<std::size_t>(num_pending_shards), config.resume ? 1 : 0);

    sfapi::context_t gpu_context(std::span<int const>(config.gpu_ids));

    for (auto const &shard_pair : shard_pairs) {
        if (shard_pair.already_complete) {
            std::printf("[shard] SKIP (complete): %s\n", shard_pair.input_path.filename().string().c_str());
            continue;
        }
        process_shard(gpu_context, config, shard_pair);
    }

    std::puts("smallfield_batch: done");
    return 0;
}
