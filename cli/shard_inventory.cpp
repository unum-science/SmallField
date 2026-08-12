/**
 *  @file cli/shard_inventory.cpp
 *  @brief Walks the input directory, matches each shard to its deterministic output path, and checks
 *  `.encoded` sentinels to find already-complete shards.
 */

#include <algorithm>   // `std::sort`
#include <string_view> // `std::string_view`
#include <system_error>

#include <fmt/std.h> // formats `std::filesystem::path`

#include "cli/shard_inventory.hpp"

namespace fs = std::filesystem;

namespace smallfield::cli {

/** A `*.parquet` whose name ends with `.3D.parquet` is one of this tool's own outputs. */
static bool is_output_shard(fs::path const &path) { return path.filename().string().ends_with(".3D.parquet"); }

std::vector<shard_pair_t> inventory_shards(fs::path const &input_dir, fs::path const &output_dir,
                                           restart_policy_t restart_policy, logger_t const *log) {
    std::vector<shard_pair_t> shards;
    logger_t const fallback;
    logger_t const &logger = log != nullptr ? *log : fallback;

    std::error_code error_code;
    if (!fs::is_directory(input_dir, error_code)) {
        logger.error("input dir does not exist: {}", input_dir);
        return shards;
    }
    fs::create_directories(output_dir, error_code);

    for (auto const &entry : fs::directory_iterator(input_dir, error_code)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".parquet") continue;
        if (is_output_shard(entry.path())) continue; // skip its own outputs when --output == --input
        shard_pair_t pair;
        pair.input_path = entry.path();
        // Deterministic output name: "<name>.parquet" → "<name>.3D.parquet".
        pair.output_path = output_dir / (entry.path().stem().string() + ".3D.parquet");
        if (restart_policy == restart_policy_t::resume_k)
            pair.already_complete = fs::exists(fs::path {pair.output_path.string() + ".encoded"}, error_code);
        shards.push_back(std::move(pair));
    }

    std::sort(shards.begin(), shards.end(),
              [](shard_pair_t const &left, shard_pair_t const &right) { return left.input_path < right.input_path; });
    return shards;
}

} // namespace smallfield::cli
