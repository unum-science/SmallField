/**
 *  @file cli/shard_inventory.hpp
 *  @brief Glob input Parquet shards and decide which outputs already exist so a `--resume` run skips finished shards.
 *
 *  Each input `<name>.parquet` maps to the output `<name>.3D.parquet` — note the dot, the canonical naming the
 *  Python reference also keys on. An output is complete only once its `<output>.encoded` sentinel exists, and
 *  the writer emits it after the parquet closes, so a kill mid-write never flags completion. A file already
 *  ending in `.3D.parquet` is an output and never an input, which lets `--output` default to `--input`. Sealing
 *  and parking live here as the only two ways an input shard leaves the set a later run is offered.
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

/** Whether an output shard reached the only state downstream reads as complete. Anything but `sealed_k` leaves
 *  the input unsealed, so every later run offers it again — the caller's failure counter, not a droppable one. */
enum class sentinel_write_t {
    sealed_k,
    open_failed_k,
    truncated_k,
};

/** Seal @p output_path by writing the sentinel a `--resume` run keys on, at `<output_path>.encoded`. Removed
 *  again if the write was short, because a truncated sentinel would make the next run skip an unfinished shard
 *  as though it were done. A null @p log falls back to a default one. */
[[nodiscard]] inline sentinel_write_t write_encoded_sentinel(std::filesystem::path const &output_path,
                                                             logger_t const *log = nullptr) {
    constexpr std::string_view body_k = "v1\n";
    logger_t const fallback;
    logger_t const &logger = log != nullptr ? *log : fallback;
    std::filesystem::path sentinel = output_path;
    sentinel += encoded_sentinel_suffix_k;

    FILE *sentinel_file = std::fopen(sentinel.string().c_str(), "wb");
    if (sentinel_file == nullptr) {
        logger.error("sentinel write failed: {}", sentinel.string());
        return sentinel_write_t::open_failed_k;
    }
    std::size_t const written = std::fwrite(body_k.data(), 1, body_k.size(), sentinel_file);
    std::fclose(sentinel_file);
    if (written == body_k.size()) return sentinel_write_t::sealed_k;
    std::error_code remove_error;
    std::filesystem::remove(sentinel, remove_error);
    logger.error("sentinel write truncated, removed: {}", sentinel.string());
    return sentinel_write_t::truncated_k;
}

/** Whether a shard that poisoned the driver context left the set a later run is offered. */
enum class park_outcome_t {
    parked_k,
    rename_failed_k,
};

/**
 *  @brief Move the shard @p input_path names into a sibling `_skip` folder, so no later run is offered it.
 *
 *  `--input` is often a staging directory of symlinks into a corpus the runner globs separately, and renaming
 *  a symlink parks nothing — the shard stays where the runner finds it and faults again. So the resolved file
 *  moves into a `_skip` sibling of @b its own directory, the dangling link is dropped. A null @p log defaults.
 */
[[nodiscard]] inline park_outcome_t park_input_shard(std::filesystem::path const &input_path,
                                                     logger_t const *log = nullptr) {
    namespace fs = std::filesystem;
    logger_t const fallback;
    logger_t const &logger = log != nullptr ? *log : fallback;

    std::error_code error_code;
    fs::path const resolved = fs::weakly_canonical(input_path, error_code);
    fs::path const shard = error_code ? input_path : resolved;
    fs::path const shard_dir = shard.parent_path();
    fs::path const skip_dir = shard_dir.parent_path() / (shard_dir.filename().string() + "_skip");
    fs::create_directories(skip_dir, error_code);
    fs::path const destination = skip_dir / shard.filename();

    fs::rename(shard, destination, error_code);
    if (error_code) {
        logger.error("parking {} -> {} failed: {}", shard.string(), destination.string(), error_code.message());
        return park_outcome_t::rename_failed_k;
    }
    logger.error("parked shard {} -> {}", shard.string(), destination.string());
    // The staging link now names nothing; it exists only because this run was pointed at a view of the corpus.
    if (fs::is_symlink(input_path, error_code)) fs::remove(input_path, error_code);
    return park_outcome_t::parked_k;
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
