/**
 *  @file shard_inventory.hpp
 *  @brief Glob input Parquet shards and decide which outputs already exist
 *         so a `--resume` run skips finished shards.
 *
 *  Restart policy (no metadata file): each input shard
 *  `<name>.parquet` deterministically maps to output `<name>_3D.parquet`.
 *  An output is considered complete if and only if it exists AND its row
 *  count equals `input_row_count * num_conformers`.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace smallfield::batch {

struct shard_pair_t {
    std::filesystem::path input_path;
    std::filesystem::path output_path;
    std::uint64_t input_num_rows;
    bool already_complete;
};

/**
 *  @brief Inventory every *.parquet file in @p input_dir, compute the matching
 *         output path in @p output_dir, and mark shards whose output already
 *         has the expected row count as `already_complete`.
 *
 *  @param num_conformers used to compute expected_output_rows = input_rows × K.
 *  @param resume when true, already-complete shards are flagged; when false,
 *         all outputs are considered pending (caller will overwrite).
 */
std::vector<shard_pair_t> inventory_shards(std::filesystem::path const &input_dir,
                                           std::filesystem::path const &output_dir,
                                           std::uint32_t num_conformers,
                                           bool resume);

} // namespace smallfield::batch
