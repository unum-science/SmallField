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

#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::fopen`

#include <filesystem>   // `std::filesystem::path`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::error_code`
#include <vector>       // `std::vector`

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

/** Suffix marking an output shard complete; every producer appends it and `inventory_shards` looks for it. */
inline constexpr std::string_view encoded_sentinel_suffix_k = ".encoded";

/**
 *  @brief Seal @p output_path by writing the sentinel a `--resume` run keys on, at `<output_path>.encoded`.
 *
 *  Removed again if the write was short, because a truncated sentinel would make the next run skip an
 *  unfinished shard as though it were done. A null @p log falls back to a default one.
 */
inline void write_encoded_sentinel(std::filesystem::path const &output_path, logger_t const *log = nullptr) {
    constexpr std::string_view body_k = "v1\n";
    logger_t const fallback;
    logger_t const &logger = log != nullptr ? *log : fallback;
    std::filesystem::path sentinel = output_path;
    sentinel += encoded_sentinel_suffix_k;

    FILE *sentinel_file = std::fopen(sentinel.string().c_str(), "wb");
    if (sentinel_file == nullptr) {
        logger.error("sentinel write failed: {}", sentinel.string());
        return;
    }
    std::size_t const written = std::fwrite(body_k.data(), 1, body_k.size(), sentinel_file);
    std::fclose(sentinel_file);
    if (written == body_k.size()) return;
    std::error_code remove_error;
    std::filesystem::remove(sentinel, remove_error);
    logger.error("sentinel write truncated, removed: {}", sentinel.string());
}

/**
 *  @brief Pair every input shard in @p input_dir with its output in @p output_dir, sorted by input path.
 *
 *  @param[in] log Logger diagnostics are reported through; null falls back to a default one.
 */
std::vector<shard_pair_t> inventory_shards(std::filesystem::path const &input_dir,
                                           std::filesystem::path const &output_dir, restart_policy_t restart_policy,
                                           logger_t const *log = nullptr);

} // namespace smallfield::cli
