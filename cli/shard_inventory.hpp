/**
 *  @file cli/shard_inventory.hpp
 *  @brief Glob input Parquet shards and decide which outputs already exist so a `--resume` run skips finished shards.
 *
 *  Each input shard `<name>.parquet` maps to the output `<name>.3D.parquet` — note the dot, which is the
 *  canonical naming the Python reference reimplementation also keys on. An output counts as complete only
 *  once its `<output>.encoded` sentinel exists, and the writer emits that sentinel after the parquet closes,
 *  so a kill mid-write never flags completion. A file already ending in `.3D.parquet` is an output and never
 *  an input, which is what lets `--output` default to `--input`.
 */

#pragma once

#include <filesystem> // `std::filesystem::path`
#include <vector>     // `std::vector`

#include "smallfield/log.hpp"

namespace smallfield::cli {

/** One input Parquet shard, the output it maps to, and whether that output is already sealed. */
struct shard_pair_t {
    std::filesystem::path input_path;
    std::filesystem::path output_path;
    bool already_complete = false;
};

/** Whether an output shard that already carries its `.encoded` sentinel is kept or produced again. */
enum class restart_policy_t {
    /** A shard whose sentinel exists is complete, so the run skips it. */
    resume_k,
    /** Every shard is pending, so the run overwrites the outputs it finds. */
    restart_k,
};

/**
 *  @brief Pair every input shard in @p input_dir with its output in @p output_dir, sorted by input path.
 *
 *  @param[in] log Logger diagnostics are reported through; null falls back to a default one.
 */
std::vector<shard_pair_t> inventory_shards(std::filesystem::path const &input_dir,
                                           std::filesystem::path const &output_dir, restart_policy_t restart_policy,
                                           logger_t const *log = nullptr);

} // namespace smallfield::cli
