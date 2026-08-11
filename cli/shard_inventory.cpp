/**
 *  @file cli/shard_inventory.cpp
 *  @brief Walks the input directory, matches each shard to its deterministic output path, and checks
 *  `.encoded` sentinels to find already-complete shards.
 */

#include <algorithm>   // `std::sort`
#include <string_view> // `std::string_view`
#include <system_error>

#include <arrow/io/file.h>
#include <fmt/std.h> // formats `std::filesystem::path`
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

#include "cli/shard_inventory.hpp"

namespace fs = std::filesystem;

namespace smallfield::cli {

static std::uint64_t parquet_row_count(fs::path const &path) {
    try {
        auto reader = parquet::ParquetFileReader::OpenFile(path.string());
        if (!reader) return 0;
        auto metadata = reader->metadata();
        return static_cast<std::uint64_t>(metadata->num_rows());
    }
    catch (std::exception const &) {
        return 0;
    }
}

/** A `*.parquet` whose name ends with `.3D.parquet` is one of this tool's own outputs. */
static bool is_output_shard(fs::path const &path) {
    std::string const filename = path.filename().string();
    constexpr std::string_view suffix_k = ".3D.parquet";
    return filename.size() >= suffix_k.size() &&
           filename.compare(filename.size() - suffix_k.size(), suffix_k.size(), suffix_k) == 0;
}

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
        shard_pair_t pair {};
        pair.input_path = entry.path();
        // Deterministic output name: "<name>.parquet" → "<name>.3D.parquet".
        auto out_stem = entry.path().stem().string() + ".3D";
        pair.output_path = output_dir / (out_stem + ".parquet");
        pair.input_num_rows = parquet_row_count(pair.input_path);
        pair.already_complete = false;
        if (restart_policy == restart_policy_t::resume_k) {
            fs::path const sentinel = pair.output_path;
            std::string sentinel_string = sentinel.string() + ".encoded";
            pair.already_complete = fs::exists(fs::path(sentinel_string), error_code);
        }
        shards.push_back(std::move(pair));
    }

    std::sort(shards.begin(), shards.end(),
              [](shard_pair_t const &left, shard_pair_t const &right) { return left.input_path < right.input_path; });
    return shards;
}

std::vector<fs::path> list_input_shards(fs::path const &input_dir, logger_t const *log) {
    std::vector<fs::path> paths;
    logger_t const fallback;
    logger_t const &logger = log != nullptr ? *log : fallback;

    std::error_code error_code;
    if (!fs::is_directory(input_dir, error_code)) {
        logger.error("input dir does not exist: {}", input_dir);
        return paths;
    }
    for (auto const &entry : fs::directory_iterator(input_dir, error_code)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".parquet") continue;
        if (is_output_shard(entry.path())) continue;
        paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

} // namespace smallfield::cli
