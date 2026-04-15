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

#include "api/smallfield.hpp"

#include "parquet_reader.hpp"
#include "parquet_writer.hpp"
#include "rdkit_fallback.hpp"
#include "shard_inventory.hpp"

namespace fs = std::filesystem;
using namespace smallfield;
using namespace smallfield::batch;

namespace {

struct cli_config_t {
    fs::path input_dir;
    fs::path output_dir;
    std::string smiles_column = "smiles";
    std::uint32_t num_conformers = 3;
    std::uint32_t max_atoms = 128;
    std::vector<int> gpu_ids;              // empty = all visible
    int rdkit_fallback_threads = 0;        // 0 = omp_get_max_threads() at call time
    bool resume = true;
    std::uint64_t random_seed = 42;
};

void print_usage() {
    std::fprintf(stderr,
        "Usage: smallfield_batch --input DIR --output DIR [options]\n"
        "  --input DIR                 folder of *.parquet input shards\n"
        "  --output DIR                folder to write *_3D.parquet outputs\n"
        "  --smiles-column NAME        input column name (default: smiles)\n"
        "  --num-conformers K          conformers per molecule (default: 3)\n"
        "  --max-atoms N               per-molecule atom cap (default: 128)\n"
        "  --gpus a,b,c                GPU device IDs (default: all visible)\n"
        "  --rdkit-fallback-threads N  OpenMP threads for RDKit fallback (default: max)\n"
        "  --restart                   overwrite already-complete shards\n"
        "  --seed N                    RNG seed for both GPU and RDKit fallback\n");
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
            auto value = require_value("input"); if (!value) return false;
            config.input_dir = value;
        }
        else if (flag == "--output") {
            auto value = require_value("output"); if (!value) return false;
            config.output_dir = value;
        }
        else if (flag == "--smiles-column") {
            auto value = require_value("smiles-column"); if (!value) return false;
            config.smiles_column = value;
        }
        else if (flag == "--num-conformers") {
            auto value = require_value("num-conformers"); if (!value) return false;
            config.num_conformers = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
        }
        else if (flag == "--max-atoms") {
            auto value = require_value("max-atoms"); if (!value) return false;
            config.max_atoms = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
        }
        else if (flag == "--gpus") {
            auto value = require_value("gpus"); if (!value) return false;
            if (!parse_int_list(value, config.gpu_ids)) return false;
        }
        else if (flag == "--rdkit-fallback-threads") {
            auto value = require_value("rdkit-fallback-threads"); if (!value) return false;
            config.rdkit_fallback_threads = static_cast<int>(std::strtol(value, nullptr, 10));
        }
        else if (flag == "--restart") {
            config.resume = false;
        }
        else if (flag == "--resume") {
            config.resume = true;
        }
        else if (flag == "--seed") {
            auto value = require_value("seed"); if (!value) return false;
            config.random_seed = std::strtoull(value, nullptr, 10);
        }
        else if (flag == "--help" || flag == "-h") {
            print_usage(); return false;
        }
        else {
            std::fprintf(stderr, "unknown flag: %.*s\n",
                         static_cast<int>(flag.size()), flag.data());
            return false;
        }
    }
    if (config.input_dir.empty() || config.output_dir.empty()) {
        print_usage(); return false;
    }
    return true;
}

void process_shard(context_t &gpu_context, cli_config_t const &config,
                   shard_pair_t const &shard) {
    std::printf("[shard] %s → %s (%zu rows)\n",
                shard.input_path.filename().string().c_str(),
                shard.output_path.filename().string().c_str(),
                static_cast<std::size_t>(shard.input_num_rows));
    std::fflush(stdout);

    auto begin_time = std::chrono::high_resolution_clock::now();

    shard_data_t data;
    if (!read_shard(shard.input_path, config.smiles_column, data)) {
        std::fprintf(stderr, "[shard] SKIP (read failed): %s\n",
                     shard.input_path.string().c_str());
        return;
    }

    std::uint64_t const num_input_rows = data.smiles.size();
    std::uint64_t const total_conformers = num_input_rows
                                           * static_cast<std::uint64_t>(config.num_conformers);
    std::size_t const coordinate_stride = static_cast<std::size_t>(config.max_atoms) * 3u;
    std::size_t const coordinate_total = total_conformers * coordinate_stride;

    std::vector<__half> coordinates(coordinate_total);
    std::vector<float> energies(total_conformers, 0.0f);
    std::vector<conformer_status_t> status(total_conformers, conformer_status_t::mmff_unconverged);

    std::vector<std::string_view> smiles_views;
    smiles_views.reserve(num_input_rows);
    for (auto const &smiles : data.smiles) smiles_views.emplace_back(smiles);

    config_t gpu_config;
    gpu_config.num_conformers = config.num_conformers;
    gpu_config.random_seed = config.random_seed;

    gpu_context.generate_conformers(smiles_views, gpu_config, config.max_atoms,
                                     std::span<__half>(coordinates),
                                     std::span<float>(energies),
                                     std::span<conformer_status_t>(status));

    // Partition failures for RDKit fallback.
    std::vector<std::uint8_t> solver(total_conformers, 0);
    std::vector<std::uint8_t> status_u8(total_conformers, 0);
    std::vector<std::uint32_t> retry_indices;
    retry_indices.reserve(num_input_rows / 32);
    for (std::uint64_t molecule_index = 0; molecule_index < num_input_rows; ++molecule_index) {
        bool molecule_has_failure = false;
        for (std::uint32_t conformer_index = 0; conformer_index < config.num_conformers; ++conformer_index) {
            std::uint64_t const flat = molecule_index * config.num_conformers + conformer_index;
            status_u8[flat] = static_cast<std::uint8_t>(status[flat]);
            if (status[flat] == conformer_status_t::success) {
                solver[flat] = 0;
            }
            else {
                solver[flat] = 2;
                molecule_has_failure = true;
            }
        }
        if (molecule_has_failure) retry_indices.push_back(static_cast<std::uint32_t>(molecule_index));
    }

    std::vector<std::uint8_t> num_atoms_per_molecule(num_input_rows, 0);

    if (!retry_indices.empty()) {
        rdkit_fallback_inputs_t fallback_inputs {
            std::span<std::string const>(data.smiles),
            std::span<std::uint32_t const>(retry_indices),
            config.num_conformers,
            config.max_atoms,
            config.rdkit_fallback_threads,
            config.random_seed,
        };
        rdkit_fallback_outputs_t fallback_outputs {
            std::span<std::uint8_t>(status_u8),
            std::span<std::uint8_t>(solver),
            std::span<std::uint8_t>(num_atoms_per_molecule),
            std::span<__half>(coordinates),
            std::span<float>(energies),
        };
        auto recovered = rdkit_fallback(fallback_inputs, fallback_outputs);
        std::printf("[shard] rdkit fallback: %zu → %zu recovered\n",
                    retry_indices.size(), static_cast<std::size_t>(recovered));
    }

    writer_inputs_t writer_inputs {
        shard.input_path.stem().string(),
        std::span<std::uint64_t const>(data.input_row_indices),
        config.num_conformers,
        config.max_atoms,
        std::span<std::uint8_t const>(num_atoms_per_molecule),
        std::span<std::uint8_t const>(status_u8),
        std::span<std::uint8_t const>(solver),
        std::span<__half const>(coordinates),
        std::span<float const>(energies),
    };

    if (!write_shard(shard.output_path, writer_inputs)) {
        std::fprintf(stderr, "[shard] WRITE FAILED: %s\n", shard.output_path.string().c_str());
        return;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double const elapsed_ms = std::chrono::duration<double, std::milli>(end_time - begin_time).count();
    std::printf("[shard] done %.1f ms (%.1f mol/s)\n",
                elapsed_ms, num_input_rows * 1000.0 / elapsed_ms);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char **argv) {
    cli_config_t config;
    if (!parse_cli(argc, argv, config)) return 2;

    auto shards = inventory_shards(config.input_dir, config.output_dir,
                                   config.num_conformers, config.resume);
    if (shards.empty()) {
        std::fprintf(stderr, "no input shards found in %s\n", config.input_dir.string().c_str());
        return 1;
    }

    std::uint64_t num_todo = 0;
    for (auto const &shard : shards) if (!shard.already_complete) ++num_todo;

    std::printf("smallfield_batch: %zu shards, %zu pending (resume=%d)\n",
                shards.size(), static_cast<std::size_t>(num_todo), config.resume ? 1 : 0);

    context_t gpu_context(std::span<int const>(config.gpu_ids));

    for (auto const &shard : shards) {
        if (shard.already_complete) {
            std::printf("[shard] SKIP (complete): %s\n",
                        shard.input_path.filename().string().c_str());
            continue;
        }
        process_shard(gpu_context, config, shard);
    }

    std::puts("smallfield_batch: done");
    return 0;
}
