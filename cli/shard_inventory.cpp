#include "shard_inventory.hpp"

#include <algorithm>
#include <cstdio>
#include <string_view>
#include <system_error>

#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

namespace smallfield::cli {

namespace fs = std::filesystem;

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

/// @brief A *.parquet whose name ends with ".3D.parquet" is one of our own outputs.
static bool is_output_shard(fs::path const &path) {
    std::string const filename = path.filename().string();
    constexpr std::string_view suffix = ".3D.parquet";
    return filename.size() >= suffix.size() &&
           filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<shard_pair_t> inventory_shards(fs::path const &input_dir, fs::path const &output_dir, bool resume) {
    std::vector<shard_pair_t> shards;

    std::error_code ec;
    if (!fs::is_directory(input_dir, ec)) {
        std::fprintf(stderr, "[smallfield_batch] input dir does not exist: %s\n", input_dir.string().c_str());
        return shards;
    }
    fs::create_directories(output_dir, ec);

    for (auto const &entry : fs::directory_iterator(input_dir, ec)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".parquet") continue;
        if (is_output_shard(entry.path())) continue; // skip our own outputs when --output == --input
        shard_pair_t pair {};
        pair.input_path = entry.path();
        // Deterministic output name: "<name>.parquet" → "<name>.3D.parquet".
        auto out_stem = entry.path().stem().string() + ".3D";
        pair.output_path = output_dir / (out_stem + ".parquet");
        pair.input_num_rows = parquet_row_count(pair.input_path);
        pair.already_complete = false;
        if (resume) {
            fs::path const sentinel = pair.output_path;
            std::string sentinel_str = sentinel.string() + ".encoded";
            pair.already_complete = fs::exists(fs::path(sentinel_str), ec);
        }
        shards.push_back(std::move(pair));
    }

    std::sort(shards.begin(), shards.end(),
              [](shard_pair_t const &a, shard_pair_t const &b) { return a.input_path < b.input_path; });
    return shards;
}

std::vector<fs::path> list_input_shards(fs::path const &input_dir) {
    std::vector<fs::path> paths;
    std::error_code ec;
    if (!fs::is_directory(input_dir, ec)) {
        std::fprintf(stderr, "[shard_inventory] input dir does not exist: %s\n", input_dir.string().c_str());
        return paths;
    }
    for (auto const &entry : fs::directory_iterator(input_dir, ec)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".parquet") continue;
        if (is_output_shard(entry.path())) continue;
        paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

} // namespace smallfield::cli
