#include "shard_inventory.hpp"

#include <algorithm>
#include <cstdio>
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

std::vector<shard_pair_t> inventory_shards(fs::path const &input_dir, fs::path const &output_dir,
                                           std::uint32_t num_conformers, bool resume) {
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
        shard_pair_t pair {};
        pair.input_path = entry.path();
        // Deterministic output name: replace ".parquet" with "_3D.parquet".
        auto out_stem = entry.path().stem().string() + "_3D";
        pair.output_path = output_dir / (out_stem + ".parquet");
        pair.input_num_rows = parquet_row_count(pair.input_path);
        pair.already_complete = false;
        if (resume && fs::exists(pair.output_path)) {
            std::uint64_t const expected = pair.input_num_rows * static_cast<std::uint64_t>(num_conformers);
            std::uint64_t const actual = parquet_row_count(pair.output_path);
            pair.already_complete = (actual == expected && expected > 0);
        }
        shards.push_back(std::move(pair));
    }

    std::sort(shards.begin(), shards.end(),
              [](shard_pair_t const &a, shard_pair_t const &b) { return a.input_path < b.input_path; });
    return shards;
}

} // namespace smallfield::cli
