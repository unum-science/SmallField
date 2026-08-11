/**
 *  @file cli/shard_inventory.hpp
 *  @brief Glob input Parquet shards and decide which outputs already exist so a `--resume` run skips finished shards.
 *
 *  Restart policy: each input shard `<name>.parquet` deterministically maps to output `<name>.3D.parquet`, note the
 *  dot, matching the canonical `*.3D.parquet` naming shared with the Python reference reimplementation. An output
 *  is considered complete iff a sentinel file `<output>.encoded` exists. The sentinel is written by the writer @b
 *  only after the parquet has been atomically closed, so a kill mid-write never falsely flags completion.
 *
 *  Files whose name already ends with `.3D.parquet` are treated as outputs and excluded from the input set —
 *  important when `--output` defaults to the same directory as `--input`.
 */

#pragma once

#include <cstdint> // `std::uint64_t`

#include <filesystem> // `std::filesystem::path`
#include <vector>     // `std::vector`

#include "smallfield/log.hpp"

namespace smallfield::cli {

/** One input/output Parquet shard with its row count and completion flag. */
struct shard_pair_t {
    std::filesystem::path input_path;
    std::filesystem::path output_path;
    std::uint64_t input_num_rows;
    bool already_complete;
};

/** Whether an output shard that already carries its `.encoded` sentinel is kept or produced again. */
enum class restart_policy_t {
    /** A shard whose sentinel exists is complete, so the run skips it. */
    resume_k,
    /** Every shard is pending, so the run overwrites the outputs it finds. */
    restart_k,
};

/**
 *  @brief Inventory every `*.parquet` file in @p input_dir, excluding any `*.3D.parquet` siblings, compute
 *  the matching output path in @p output_dir, and mark shards whose `<output>.encoded` sentinel exists as
 *  `already_complete`.
 *
 *  @param[in] restart_policy Whether an existing sentinel marks a shard complete, or is ignored so the
 *         caller reproduces every output.
 *  @param[in] log Logger diagnostics are reported through; null falls back to a default one.
 */
std::vector<shard_pair_t> inventory_shards(std::filesystem::path const &input_dir,
                                           std::filesystem::path const &output_dir, restart_policy_t restart_policy,
                                           logger_t const *log = nullptr);

/**
 *  @brief List every `*.parquet` file in @p input_dir, excluding `*.3D.parquet`, sorted lexicographically.
 *
 *  @param[in] log Logger diagnostics are reported through; null falls back to a default one.
 */
std::vector<std::filesystem::path> list_input_shards(std::filesystem::path const &input_dir,
                                                     logger_t const *log = nullptr);

} // namespace smallfield::cli
