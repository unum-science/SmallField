/**
 *  @file shard_inventory.hpp
 *  @brief Glob input Parquet shards and decide which outputs already exist
 *         so a `--resume` run skips finished shards.
 *
 *  Restart policy: each input shard `<name>.parquet` deterministically maps
 *  to output `<name>.3D.parquet` (note the dot — matches the canonical
 *  `*.3D.parquet` naming shared with the Python reference reimplementation).
 *  An output is considered complete iff a sentinel file `<output>.encoded`
 *  exists. The sentinel is written by the writer ONLY after the parquet has
 *  been atomically closed, so a kill mid-write never falsely flags
 *  completion.
 *
 *  Files whose name already ends with `.3D.parquet` are treated as outputs
 *  and excluded from the input set — important when `--output` defaults to
 *  the same directory as `--input`.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace smallfield::cli {

/** @brief One input/output Parquet shard with its row count and completion flag. */
struct shard_pair_t {
    std::filesystem::path input_path;
    std::filesystem::path output_path;
    std::uint64_t input_num_rows;
    bool already_complete;
};

/**
 *  @brief Inventory every *.parquet file in @p input_dir (excluding any
 *  *.3D.parquet siblings), compute the matching output path in @p output_dir,
 *  and mark shards whose `<output>.encoded` sentinel exists as
 *  `already_complete`.
 *
 *  @param resume when true, already-complete shards are flagged; when false,
 *         all outputs are considered pending (caller will overwrite).
 */
std::vector<shard_pair_t> inventory_shards(std::filesystem::path const &input_dir,
                                           std::filesystem::path const &output_dir, bool resume);

/**
 *  @brief List every `*.parquet` file in @p input_dir (excluding *.3D.parquet),
 *  sorted lexicographically.
 */
std::vector<std::filesystem::path> list_input_shards(std::filesystem::path const &input_dir);

} // namespace smallfield::cli
