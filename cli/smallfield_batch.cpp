/**
 *  @file smallfield_batch.cpp
 *  @brief Restartable batch processor for bulk conformer generation.
 *
 *  Four-stage ring-buffered pipeline:
 *    reader        → reads input-files, combines into gpus-batches
 *    gpu_runner    → runs generate_conformers on each gpus-batch
 *    fallback      → runs RDKit on conformer slots the GPU could not fill
 *    writer        → emits one output-file per input-file from resolved gpus-batches
 *
 *  Each stage owns a depth-3 ring-buffer handoff to the next.
 *  The reader combines multiple input-files until the gpus-batch target size is
 *  reached, ensuring all GPUs are saturated per generate_conformers call.  The
 *  fallback and writer stages run concurrently with the GPU on the next batch
 *  thanks to the deeper queues.  The fallback is conformer-granular: only the
 *  per-slot failures are re-embedded; partial GPU successes are preserved.
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <omp.h>

#include <cuda_fp16.h>

#include "smallfield/api/smallfield.hpp"
#include "smallfield/log.hpp"

#include "parquet_reader.hpp"
#include "parquet_writer.hpp"
#include "rdkit_fallback.hpp"
#include "shard_inventory.hpp"

namespace fs = std::filesystem;
namespace sfapi = smallfield::api;
namespace sfcli = smallfield::cli;
using smallfield::logger_t;
using smallfield::thousands;

/** All CLI flags.  Immutable after parse. */
struct cli_config_t {
    fs::path input_dir;
    fs::path output_dir;
    std::string smiles_column = "smiles";
    std::uint32_t num_conformers = 3;
    std::uint32_t max_atoms = 128;
    std::uint32_t gpus_batch_size = 0; ///< 0 = auto (100K per GPU)
    std::vector<int> gpu_ids;
    int rdkit_fallback_threads = 0;
    bool resume = true;
    std::uint64_t random_seed = 42;
    int verbose_level = 0; ///< 0 silent; 1 chunk table; 2 per-device detail
};

void print_usage() {
    fmt::print(stderr, R"(Usage: smallfield_batch --input DIR --output DIR [options]
  --input DIR                 folder of *.parquet input-files
  --output DIR                folder to write *_3D.parquet output-files
  --smiles-column NAME        input column name (default: smiles)
  --num-conformers K          conformers per molecule (default: 3)
  --max-atoms N               per-molecule atom cap (default: 128)
  --gpus-batch-size N         molecules per gpus-batch (default: auto = 100K per GPU)
  --chunk-size N              alias for --gpus-batch-size
  --gpus a,b,c                GPU device IDs (default: all visible)
  --rdkit-fallback-threads N  OpenMP threads for RDKit fallback (default: max)
  --restart                   overwrite already-complete output-files
  --seed N                    RNG seed for both GPU and RDKit fallback
  --verbose[=LEVEL]           library diagnostics: 1 = chunk table, 2 = +per-device
)");
}

bool parse_int_list(char const *spec, std::vector<int> &out) {
    out.clear();
    while (*spec) {
        char *end = nullptr;
        long value = std::strtol(spec, &end, 10);
        if (end == spec) return false;
        out.push_back(static_cast<int>(value));
        spec = end;
        if (*spec == ',') ++spec;
    }
    return true;
}

bool parse_cli(int argc, char **argv, cli_config_t &config) {
    for (int argument_index = 1; argument_index < argc; ++argument_index) {
        std::string_view flag = argv[argument_index];
        auto require_value = [&](char const *flag_name) -> char const * {
            if (argument_index + 1 >= argc) {
                fmt::print(stderr, "--{} requires a value\n", flag_name);
                return nullptr;
            }
            return argv[++argument_index];
        };
        if (flag == "--input") {
            auto value = require_value("input");
            if (!value) return false;
            config.input_dir = value;
        }
        else if (flag == "--output") {
            auto value = require_value("output");
            if (!value) return false;
            config.output_dir = value;
        }
        else if (flag == "--smiles-column") {
            auto value = require_value("smiles-column");
            if (!value) return false;
            config.smiles_column = value;
        }
        else if (flag == "--num-conformers") {
            auto value = require_value("num-conformers");
            if (!value) return false;
            config.num_conformers = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
        }
        else if (flag == "--max-atoms") {
            auto value = require_value("max-atoms");
            if (!value) return false;
            config.max_atoms = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
        }
        else if (flag == "--gpus-batch-size") {
            auto value = require_value("gpus-batch-size");
            if (!value) return false;
            config.gpus_batch_size = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
        }
        else if (flag == "--gpus") {
            auto value = require_value("gpus");
            if (!value) return false;
            if (!parse_int_list(value, config.gpu_ids)) return false;
        }
        else if (flag == "--rdkit-fallback-threads") {
            auto value = require_value("rdkit-fallback-threads");
            if (!value) return false;
            config.rdkit_fallback_threads = static_cast<int>(std::strtol(value, nullptr, 10));
        }
        else if (flag == "--chunk-size") {
            auto value = require_value("chunk-size");
            if (!value) return false;
            config.gpus_batch_size = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
        }
        else if (flag == "--restart") { config.resume = false; }
        else if (flag == "--resume") { config.resume = true; }
        else if (flag == "--seed") {
            auto value = require_value("seed");
            if (!value) return false;
            config.random_seed = std::strtoull(value, nullptr, 10);
        }
        else if (flag == "--verbose") { config.verbose_level = 1; }
        else if (flag.starts_with("--verbose=")) {
            config.verbose_level = static_cast<int>(std::strtol(flag.data() + 10, nullptr, 10));
        }
        else if (flag == "--help" || flag == "-h") {
            print_usage();
            return false;
        }
        else {
            fmt::print(stderr, "unknown flag: {}\n", flag);
            return false;
        }
    }
    if (config.input_dir.empty() || config.output_dir.empty()) {
        print_usage();
        return false;
    }
    return true;
}

/** Pre-allocated coordinate/energy/status arrays, owned by the buffer pool.
 *  Never freed on the hot path — only grows via resize_for(). */
struct output_buffers_t {
    std::vector<__half> coordinates;
    std::vector<float> energies;
    std::vector<sfapi::conformer_status_t> status;
    std::vector<std::uint8_t> status_codes;
    std::vector<std::uint8_t> solver_codes;
    std::vector<std::uint8_t> num_atoms_per_molecule;
    std::vector<std::uint32_t> retry_indices;

    void resize_for(std::uint64_t num_molecules, std::uint32_t num_conformers, std::uint32_t max_atoms) {
        std::uint64_t const total_conformers = num_molecules * num_conformers;
        std::size_t const coordinate_total = total_conformers * max_atoms * 3u;
        if (coordinates.size() < coordinate_total) coordinates.resize(coordinate_total);
        if (energies.size() < total_conformers) energies.resize(total_conformers);
        if (status.size() < total_conformers) status.resize(total_conformers);
        if (status_codes.size() < total_conformers) status_codes.resize(total_conformers);
        if (solver_codes.size() < total_conformers) solver_codes.resize(total_conformers);
        if (num_atoms_per_molecule.size() < num_molecules) num_atoms_per_molecule.resize(num_molecules);
    }
};

/** Combined input-files for one gpus-batch.
 *  Owns the Arrow tables that back the SMILES string_views.
 *  The reader fills this until total_molecules >= target, then hands it off. */
struct gpus_batch_accumulator_t {
    std::vector<sfcli::shard_data_t> input_file_data;
    std::vector<sfcli::shard_pair_t const *> input_file_pairs;
    std::vector<std::uint64_t> input_file_boundaries;
    std::vector<std::string_view> combined_smiles;
    std::uint64_t total_molecules = 0;
    bool done = false;

    void add_input_file(sfcli::shard_data_t data, sfcli::shard_pair_t const *pair) {
        input_file_boundaries.push_back(total_molecules);
        std::uint64_t const count = data.num_rows();
        for (auto sv : data.smiles) combined_smiles.push_back(sv);
        total_molecules += count;
        input_file_data.push_back(std::move(data));
        input_file_pairs.push_back(pair);
    }

    void clear() {
        input_file_data.clear();
        input_file_pairs.clear();
        input_file_boundaries.clear();
        combined_smiles.clear();
        total_molecules = 0;
        done = false;
    }
};

/** Gpus-batch after GPU processing, handed to the fallback runner.
 *  The fallback patches coordinates/energies in-place via retry_indices. */
struct gpus_batch_result_t {
    gpus_batch_accumulator_t accumulator;
    output_buffers_t *buffers = nullptr;
    bool done = false;
    /// When the gpu_runner picked up this batch — used by the writer to derive
    /// per-shard wall time (GPU + fallback + write) for the single progress line.
    std::chrono::steady_clock::time_point batch_start {};
    std::uint64_t fallback_attempted = 0; ///< number of molecules sent into RDKit fallback
    std::uint64_t fallback_recovered = 0; ///< number for which RDKit filled at least one slot
};

/** Bounded ring-buffer producer/consumer channel.
 *  Sender blocks when full, receiver blocks when empty.
 *  Depth picks how many gpus-batches the upstream can stage ahead before
 *  back-pressuring; the larger the depth the more memory the pipeline
 *  consumes but the better it tolerates downstream stalls. */
template <typename payload_>
struct handoff_t {
    std::mutex mutex;
    std::condition_variable has_item;
    std::condition_variable has_space;
    std::vector<payload_> ring;
    std::size_t head = 0;
    std::size_t tail = 0;
    std::size_t count = 0;

    explicit handoff_t(std::size_t depth) : ring(depth) {}

    void send(payload_ item) {
        std::unique_lock lock(mutex);
        has_space.wait(lock, [&] { return count < ring.size(); });
        ring[tail] = std::move(item);
        tail = (tail + 1) % ring.size();
        ++count;
        has_item.notify_one();
    }

    payload_ receive() {
        std::unique_lock lock(mutex);
        has_item.wait(lock, [&] { return count > 0; });
        payload_ result = std::move(ring[head]);
        head = (head + 1) % ring.size();
        --count;
        has_space.notify_one();
        return result;
    }
};

/** Pool of N output buffer sets cycled between gpu-runner, fallback, and writer.
 *  N matches handoff depth + slack so no stage stalls waiting for a free slot.
 *  Each output_buffers_t owns its grow-only vectors; pool.acquire / pool.release
 *  is just slot accounting, no allocation. */
struct buffer_pool_t {
    std::mutex mutex;
    std::condition_variable available;
    std::vector<output_buffers_t> buffers;
    std::vector<bool> in_use;

    explicit buffer_pool_t(std::size_t n) : buffers(n), in_use(n, false) {}

    output_buffers_t *acquire() {
        std::unique_lock lock(mutex);
        available.wait(lock, [&] {
            for (auto u : in_use)
                if (!u) return true;
            return false;
        });
        for (std::size_t s = 0; s < in_use.size(); ++s) {
            if (!in_use[s]) {
                in_use[s] = true;
                return &buffers[s];
            }
        }
        return nullptr;
    }

    void release(output_buffers_t *buf) {
        std::unique_lock lock(mutex);
        std::size_t const slot = static_cast<std::size_t>(buf - buffers.data());
        in_use[slot] = false;
        available.notify_one();
    }
};

/**
 *  @brief Pipeline CPU thread allocation.
 *
 *  Partitions the machine's cores across the four sections so their sum stays
 *  under num_procs with a small OS reserve.  Proportions (1.5 / 3 / 33 / rest %)
 *  assume fallback is the pipeline pacer on drug-like workloads: every core
 *  added to fallback directly reduces per-batch wall, while the other stages
 *  plateau past a fixed thread count.  The user can override fallback via
 *  @c --rdkit-fallback-threads when the workload's failure rate is atypical.
 *
 *  All bounds are clamped to [1, num_procs-1] so tiny (8-core dev boxes) and
 *  huge (192-core production) machines both land on sensible splits.
 */
struct thread_budget_t {
    int reader;
    int writer;
    int gpu_runner;
    int fallback;
};

thread_budget_t compute_thread_budget(int user_fallback_override) {
    int const num_procs = std::max(1, omp_get_num_procs());
    // Reader is I/O-bound on Arrow deserialization, keep it tiny.
    int const reader = std::clamp(num_procs / 128, 1, 2);
    // Writer is Arrow-serialize-bound, also mostly single-threaded inside Arrow.
    int const writer = std::clamp(num_procs / 64, 2, 4);
    // GPU runner's CPU share is chunk prep (RDKit parsing + MMFF typer over
    // molecules in OpenMP parallel-for).  Most of a chunk's wall is GPU time;
    // match the old proven v8 budget (procs - 4) as the default since lower
    // thread counts showed correlated CUDA instability we could not root-cause.
    int const gpu_runner = std::max(1, num_procs - 4);
    // OS reserve: keep a handful of cores free so kernel work (cuda driver,
    // memory pressure daemons, tmux, log flush) doesn't preempt hot threads.
    int const os_reserve = std::clamp(num_procs / 32, 2, 8);
    // Fallback gets the same broad thread budget as gpu_runner; sections run
    // at different times (prep vs RDKit), so per-stage oversubscription is
    // fine and matches the v8 configuration that ran stably.
    int const fallback = user_fallback_override > 0 ? user_fallback_override : std::max(1, num_procs - 4);
    (void)os_reserve;
    return {reader, writer, gpu_runner, fallback};
}

/** Cumulative counters shared across pipeline stages. */
struct progress_counters_t {
    std::atomic<std::uint64_t> molecules_completed {0};
    std::atomic<std::uint64_t> input_files_completed {0};
    std::uint64_t total_input_files = 0;
    std::chrono::steady_clock::time_point start_time;
};

void reader_loop(std::vector<sfcli::shard_pair_t const *> const &pending, cli_config_t const &config,
                 std::uint32_t gpus_batch_target, thread_budget_t const &budget, logger_t const &logger,
                 handoff_t<gpus_batch_accumulator_t> &out) {
    omp_set_num_threads(budget.reader);
    gpus_batch_accumulator_t accumulator;

    for (auto const *pair : pending) {
        sfcli::shard_data_t data;
        if (!sfcli::read_shard(pair->input_path, config.smiles_column, data)) {
            logger.error("read failed, skipping: {}", pair->input_path.string());
            continue;
        }
        accumulator.add_input_file(std::move(data), pair);

        if (accumulator.total_molecules >= gpus_batch_target) {
            out.send(std::move(accumulator));
            accumulator.clear();
        }
    }

    if (accumulator.total_molecules > 0) { out.send(std::move(accumulator)); }

    gpus_batch_accumulator_t sentinel;
    sentinel.done = true;
    out.send(std::move(sentinel));
}

void gpu_runner_loop(sfapi::context_t &gpu_context, cli_config_t const &config, thread_budget_t const &budget,
                     handoff_t<gpus_batch_accumulator_t> &in, handoff_t<gpus_batch_result_t> &out, buffer_pool_t &pool,
                     progress_counters_t &counters) {
    omp_set_num_threads(budget.gpu_runner);
    (void)counters;

    sfapi::config_t generation_config;
    generation_config.num_conformers = config.num_conformers;
    generation_config.random_seed = config.random_seed;

    for (;;) {
        auto accumulator = in.receive();
        if (accumulator.done) break;

        std::uint64_t const num_molecules = accumulator.total_molecules;
        std::uint64_t const total_conformers = num_molecules * config.num_conformers;
        auto const batch_start = std::chrono::steady_clock::now();

        output_buffers_t *buf = pool.acquire();
        buf->resize_for(num_molecules, config.num_conformers, config.max_atoms);
        std::fill_n(buf->energies.data(), total_conformers, 0.0f);
        std::fill_n(buf->status.data(), total_conformers, sfapi::conformer_status_t::mmff_unconverged);

        gpu_context.generate_conformers(
            std::span<std::string_view const>(accumulator.combined_smiles), generation_config, config.max_atoms,
            std::span<__half>(buf->coordinates.data(), total_conformers * config.max_atoms * 3u),
            std::span<float>(buf->energies.data(), total_conformers),
            std::span<sfapi::conformer_status_t>(buf->status.data(), total_conformers));

        std::fill_n(buf->solver_codes.data(), total_conformers, static_cast<std::uint8_t>(0));
        std::fill_n(buf->status_codes.data(), total_conformers, static_cast<std::uint8_t>(0));
        std::fill_n(buf->num_atoms_per_molecule.data(), num_molecules, static_cast<std::uint8_t>(0));

        buf->retry_indices.clear();
        for (std::uint64_t mol = 0; mol < num_molecules; ++mol) {
            bool has_failure = false;
            for (std::uint32_t conf = 0; conf < config.num_conformers; ++conf) {
                std::uint64_t const flat = mol * config.num_conformers + conf;
                buf->status_codes[flat] = static_cast<std::uint8_t>(buf->status[flat]);
                if (buf->status[flat] == sfapi::conformer_status_t::success) { buf->solver_codes[flat] = 0; }
                else {
                    buf->solver_codes[flat] = 2;
                    has_failure = true;
                }
            }
            if (has_failure) buf->retry_indices.push_back(static_cast<std::uint32_t>(mol));
        }

        gpus_batch_result_t result;
        result.accumulator = std::move(accumulator);
        result.buffers = buf;
        result.batch_start = batch_start;
        out.send(std::move(result));
    }

    gpus_batch_result_t sentinel;
    sentinel.done = true;
    out.send(std::move(sentinel));
}

void fallback_loop(cli_config_t const &config, thread_budget_t const &budget, handoff_t<gpus_batch_result_t> &in,
                   handoff_t<gpus_batch_result_t> &out) {
    omp_set_num_threads(budget.fallback);

    for (;;) {
        auto result = in.receive();
        if (result.done) {
            out.send(std::move(result));
            break;
        }

        output_buffers_t *buf = result.buffers;
        auto &acc = result.accumulator;
        std::uint64_t const num_molecules = acc.total_molecules;
        std::uint64_t const total_conformers = num_molecules * config.num_conformers;

        if (!buf->retry_indices.empty()) {
            sfcli::rdkit_fallback_inputs_t fallback_inputs {
                std::span<std::string_view const>(acc.combined_smiles),
                std::span<std::uint32_t const>(buf->retry_indices),
                config.num_conformers,
                config.max_atoms,
                budget.fallback,
                config.random_seed,
            };
            sfcli::rdkit_fallback_outputs_t fallback_outputs {
                std::span<std::uint8_t>(buf->status_codes.data(), total_conformers),
                std::span<std::uint8_t>(buf->solver_codes.data(), total_conformers),
                std::span<std::uint8_t>(buf->num_atoms_per_molecule.data(), num_molecules),
                std::span<__half>(buf->coordinates.data(), total_conformers * config.max_atoms * 3u),
                std::span<float>(buf->energies.data(), total_conformers),
            };
            auto const num_recovered = sfcli::rdkit_fallback(fallback_inputs, fallback_outputs);
            result.fallback_attempted = buf->retry_indices.size();
            result.fallback_recovered = num_recovered;
        }

        out.send(std::move(result));
    }
}

void writer_loop(cli_config_t const &config, thread_budget_t const &budget, logger_t const &logger,
                 handoff_t<gpus_batch_result_t> &in, buffer_pool_t &pool, progress_counters_t &counters) {
    omp_set_num_threads(budget.writer);

    for (;;) {
        auto result = in.receive();
        if (result.done) break;

        output_buffers_t *buf = result.buffers;
        auto &acc = result.accumulator;
        std::uint64_t const num_molecules = acc.total_molecules;

        acc.input_file_boundaries.push_back(num_molecules);
        for (std::size_t file_idx = 0; file_idx < acc.input_file_pairs.size(); ++file_idx) {
            std::uint64_t const mol_begin = acc.input_file_boundaries[file_idx];
            std::uint64_t const mol_end = acc.input_file_boundaries[file_idx + 1];
            std::uint64_t const file_molecules = mol_end - mol_begin;
            std::uint64_t const conf_begin = mol_begin * config.num_conformers;
            std::uint64_t const file_conformers = file_molecules * config.num_conformers;
            std::size_t const coord_stride = static_cast<std::size_t>(config.max_atoms) * 3u;

            sfcli::writer_inputs_t writer_inputs {
                acc.input_file_pairs[file_idx]->input_path.stem().string(),
                acc.input_file_data[file_idx].base_row_index,
                file_molecules,
                config.num_conformers,
                config.max_atoms,
                std::span<std::string_view const>(acc.combined_smiles.data() + mol_begin, file_molecules),
                std::span<std::uint8_t const>(buf->num_atoms_per_molecule.data() + mol_begin, file_molecules),
                std::span<std::uint8_t const>(buf->status_codes.data() + conf_begin, file_conformers),
                std::span<std::uint8_t const>(buf->solver_codes.data() + conf_begin, file_conformers),
                std::span<__half const>(buf->coordinates.data() + conf_begin * coord_stride,
                                        file_conformers * coord_stride),
                std::span<float const>(buf->energies.data() + conf_begin, file_conformers),
            };

            if (!sfcli::write_shard(acc.input_file_pairs[file_idx]->output_path, writer_inputs)) {
                logger.error("write failed: {}", acc.input_file_pairs[file_idx]->output_path.string());
            }

            counters.molecules_completed.fetch_add(file_molecules);
            counters.input_files_completed.fetch_add(1);
        }

        pool.release(buf);

        auto const now = std::chrono::steady_clock::now();
        double const batch_seconds = std::chrono::duration<double>(now - result.batch_start).count();
        double const wall = std::chrono::duration<double>(now - counters.start_time).count();
        std::uint64_t const cumul_mol = counters.molecules_completed.load();
        double const cumul_rate = static_cast<double>(cumul_mol) / std::max(wall, 0.001);
        std::uint64_t const files_done = counters.input_files_completed.load();
        std::uint64_t const files_remaining = counters.total_input_files - files_done;
        double const remaining_seconds = files_remaining > 0 && files_done > 0
                                             ? static_cast<double>(files_remaining) * wall / files_done
                                             : 0.0;
        auto const eta_wall = std::chrono::system_clock::now() +
                              std::chrono::milliseconds(static_cast<std::int64_t>(remaining_seconds * 1000.0));

        // Collapse gpus-batch start + fallback + done into one line.  Fallback
        // fragment is omitted when no molecule needed RDKit recovery.
        std::string fallback_fragment;
        if (result.fallback_attempted > 0)
            fallback_fragment =
                fmt::format("  fallback {}→{} rec", thousands(result.fallback_attempted),
                            thousands(result.fallback_recovered));

        logger.info("{:>4}/{:<4} done {:.0f}s  {} mol/s{}  cumulative {}  avg {} mol/s  ETA {}",
                    thousands(files_done), thousands(counters.total_input_files), batch_seconds,
                    thousands(static_cast<std::uint64_t>(num_molecules / std::max(batch_seconds, 0.001))),
                    fallback_fragment, thousands(cumul_mol),
                    thousands(static_cast<std::uint64_t>(cumul_rate)),
                    smallfield::format_wall_time(eta_wall));
    }
}

int main(int argc, char **argv) {
    cli_config_t config;
    if (!parse_cli(argc, argv, config)) return 2;

    logger_t logger;
    logger.verbose_level = config.verbose_level;

    auto shard_pairs = sfcli::inventory_shards(config.input_dir, config.output_dir, config.num_conformers,
                                               config.resume);
    if (shard_pairs.empty()) {
        logger.error("no input-files found in {}", config.input_dir.string());
        return 1;
    }

    std::vector<sfcli::shard_pair_t const *> pending;
    std::uint64_t skipped = 0;
    for (auto const &pair : shard_pairs) {
        if (pair.already_complete) { ++skipped; }
        else { pending.push_back(&pair); }
    }

    logger.info("smallfield_batch: {} input-files, {} pending, {} complete", thousands(shard_pairs.size()),
                thousands(pending.size()), thousands(skipped));

    if (pending.empty()) {
        logger.info("nothing to do");
        return 0;
    }

    sfapi::context_t gpu_context(std::span<int const>(config.gpu_ids));
    // Share the CLI's verbose level with the library so `--verbose=1|2` enables
    // the per-chunk table and per-device breakdown emitted from smallfield.cu.
    gpu_context.logger.verbose_level = config.verbose_level;
    gpu_context.logger.start_time = logger.start_time;
    std::uint32_t const num_gpus = config.gpu_ids.empty() ? static_cast<std::uint32_t>(gpu_context.device_ids.size())
                                                          : static_cast<std::uint32_t>(config.gpu_ids.size());
    std::uint32_t const gpus_batch_target = config.gpus_batch_size > 0 ? config.gpus_batch_size : num_gpus * 100'000u;

    thread_budget_t const budget = compute_thread_budget(config.rdkit_fallback_threads);
    logger.info("  {} GPUs, gpus-batch target {} mols, threads reader={} gpu_runner={} fallback={} writer={}", num_gpus,
                thousands(gpus_batch_target), budget.reader, budget.gpu_runner, budget.fallback, budget.writer);

    omp_set_max_active_levels(2);

    progress_counters_t counters;
    counters.total_input_files = pending.size();
    counters.start_time = logger.start_time;

    // Pipeline depth picks how many gpus-batches the upstream may stage before
    // back-pressuring.  Depth 3 lets reader stay 2 ahead of GPU and GPU stay 1
    // ahead of fallback while writer drains; pool sized to depth+1 so no stage
    // ever blocks waiting for a free output buffer.
    constexpr std::size_t kPipelineDepth = 3;
    buffer_pool_t pool {kPipelineDepth + 1};
    handoff_t<gpus_batch_accumulator_t> reader_to_gpu {kPipelineDepth};
    handoff_t<gpus_batch_result_t> gpu_to_fallback {kPipelineDepth};
    handoff_t<gpus_batch_result_t> fallback_to_writer {kPipelineDepth};

#pragma omp parallel sections num_threads(4)
    {
#pragma omp section
        { reader_loop(pending, config, gpus_batch_target, budget, logger, reader_to_gpu); }
#pragma omp section
        { gpu_runner_loop(gpu_context, config, budget, reader_to_gpu, gpu_to_fallback, pool, counters); }
#pragma omp section
        { fallback_loop(config, budget, gpu_to_fallback, fallback_to_writer); }
#pragma omp section
        { writer_loop(config, budget, logger, fallback_to_writer, pool, counters); }
    }

    double const total = std::chrono::duration<double>(std::chrono::steady_clock::now() - counters.start_time).count();
    std::uint64_t const done = counters.molecules_completed.load();
    logger.info("complete  {} mols  {} mol/s",
                thousands(done), thousands(static_cast<std::uint64_t>(done / std::max(total, 0.001))));
    std::uint64_t const cuda_errors = gpu_context.cuda_error_count.load();
    if (cuda_errors > 0) {
        logger.error("{} CUDA errors reported by the library — output is unreliable", thousands(cuda_errors));
        return 2;
    }
    return 0;
}
