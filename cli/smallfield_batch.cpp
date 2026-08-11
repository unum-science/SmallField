/**
 *  @file cli/smallfield_batch.cpp
 *  @brief Restartable batch processor for bulk conformer generation.
 *
 *  Four `pipeline-roles` — see GLOSSARY.md — with depth-3 ring-buffer handoffs:
 *  @verbatim
 *    reader-role     → reads parquet-shards, combines into coprocessed-batches
 *    gpu-runner-role → runs `generate_conformers` on each coprocessed-batch
 *    fallback-role   → runs RDKit on conformer slots the GPU could not fill
 *    writer-role     → emits one output parquet-shard per consumed input shard
 *  @endverbatim
 *
 *  The reader-role accumulates parquet-shards until the coprocessed-batch size target is reached, ensuring
 *  all GPUs are saturated per `generate_conformers` call. Fallback and writer roles run concurrently with the
 *  GPU on later batches thanks to the deeper queues. The fallback is conformer-granular: only the per-slot
 *  failures are re-embedded; partial GPU successes are kept.
 */

#include <csignal>
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstdio>  // `stderr`, `fprintf`
#include <cstdlib> // `std::strtol`

#include <algorithm>          // `std::unique`, `std::max`
#include <array>              // `std::array`
#include <atomic>             // `std::atomic`
#include <chrono>             // `std::chrono::steady_clock`, `std::chrono::duration`
#include <condition_variable> // `std::condition_variable`
#include <filesystem>         // `std::filesystem::path`, `std::filesystem::create_directories`
#include <mutex>              // `std::mutex`, `std::unique_lock`
#include <span>               // `std::span`
#include <string>             // `std::string`
#include <string_view>        // `std::string_view`
#include <vector>             // `std::vector`

#include <omp.h>
#include <cuda_fp16.h>

#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/api/smallfield.hpp"
#include "smallfield/log.hpp"
#include "smallfield/nvtx.hpp"
#include "smallfield/postprocess.hpp"

#include "parquet_reader.hpp"
#include "parquet_writer.hpp"
#include "rdkit_fallback.hpp"
#include "shard_inventory.hpp"

namespace fs = std::filesystem;
namespace sfapi = smallfield::api;
namespace sfcli = smallfield::cli;
namespace sfpost = smallfield::postprocess;
using smallfield::logger_t;
using smallfield::thousands;

namespace smallfield::cli {

/** Set by SIGINT/SIGTERM. The reader-role stops feeding new shards at the next shard boundary so
 *  in-flight batches drain and their shards are written before the pipeline exits cleanly (exit 0).
 *  This is the graceful pre-emption path: no partial-shard loss, and resume skips the completed
 *  shards. Async-signal-safe: a lock-free atomic flag, nothing else touched in the handler.
 *
 *  `handle_stop_signal_` keeps C language linkage for `std::signal`, which an ordinary `static`
 *  function cannot combine with, so the unnamed namespace supplies the internal linkage here. */
namespace {
std::atomic<bool> g_stop_requested_ {false};
extern "C" void handle_stop_signal_(int) { g_stop_requested_.store(true, std::memory_order_relaxed); }
} // namespace

/** All CLI flags. Immutable after parse. */
struct cli_config_t {
    fs::path input_dir;
    fs::path output_dir;
    std::string smiles_column = "smiles";
    /** Provenance dataset name (e.g. "pubchem", "gdb13"); "" = omit from metadata. */
    std::string dataset;
    std::uint32_t num_conformers = 3;
    std::uint32_t max_atoms = 128;
    /** Target `coprocessed-batch` size in molecules; 0 = auto (100 K per visible GPU). */
    std::uint32_t batch_size = 0;
    std::vector<int> gpu_ids;
    int rdkit_fallback_threads = 0;
    /** 0 = auto split vs fallback-role, non-zero overrides. */
    int gpu_runner_threads = 0;
    bool resume = true;
    std::uint64_t random_seed = 42;
    /** Library verbose tier: 0 silent, 1 per-`embedding-wave` summary, 2 + per-device breakdown. */
    int verbose_level = 0;
};

static void print_usage_() {
    fmt::print(stderr, R"(Usage: smallfield_batch --input DIR [--output DIR] [options]
  --input DIR                 folder of input parquet-shards (*.parquet)
  --output DIR                folder for output parquet-shards (*.3D.parquet)
                              (default: same as --input — sister files alongside)
  --smiles-column NAME        input column name (default: smiles)
  --dataset NAME              provenance dataset name embedded in output metadata (e.g. gdb13)
  --num-conformers K          conformers per molecule (default: 3)
  --max-atoms N               per-molecule atom cap (default: 128)
  --batch-size N              molecules per coprocessed-batch (default: auto = 100K per GPU)
  --coprocessed-batch-size N         alias for --batch-size
  --chunk-size N              alias for --batch-size
  --gpus a,b,c                GPU device IDs (default: all visible)
  --rdkit-fallback-threads N  OpenMP threads for fallback-role (default: num_procs - 4)
  --gpu-runner-threads N      OpenMP threads for gpu-runner-role preprocess (default: num_procs - 4)
  --restart                   overwrite already-complete output shards
  --seed N                    RNG seed for both GPU and RDKit fallback
  --verbose[=LEVEL]           library diagnostics: 1 = wave table, 2 = +per-device
)");
}

static bool parse_int_list_(char const *spec, std::vector<int> &out) {
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

static bool parse_cli_(int argc, char **argv, cli_config_t &config) {
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
        else if (flag == "--dataset") {
            auto value = require_value("dataset");
            if (!value) return false;
            config.dataset = value;
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
        else if (flag == "--batch-size" || flag == "--coprocessed-batch-size") {
            auto value = require_value(flag == "--batch-size" ? "batch-size" : "coprocessed-batch-size");
            if (!value) return false;
            config.batch_size = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
        }
        else if (flag == "--gpus") {
            auto value = require_value("gpus");
            if (!value) return false;
            if (!parse_int_list_(value, config.gpu_ids)) return false;
        }
        else if (flag == "--rdkit-fallback-threads") {
            auto value = require_value("rdkit-fallback-threads");
            if (!value) return false;
            config.rdkit_fallback_threads = static_cast<int>(std::strtol(value, nullptr, 10));
        }
        else if (flag == "--gpu-runner-threads") {
            auto value = require_value("gpu-runner-threads");
            if (!value) return false;
            config.gpu_runner_threads = static_cast<int>(std::strtol(value, nullptr, 10));
        }
        else if (flag == "--chunk-size") {
            auto value = require_value("chunk-size");
            if (!value) return false;
            config.batch_size = static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
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
            print_usage_();
            return false;
        }
        else {
            fmt::print(stderr, "unknown flag: {}\n", flag);
            return false;
        }
    }
    if (config.input_dir.empty()) {
        print_usage_();
        return false;
    }
    // --output defaults to --input: the *.3D.parquet sister files live next to the
    // sources unless the caller explicitly redirects them elsewhere.
    if (config.output_dir.empty()) config.output_dir = config.input_dir;
    return true;
}

/** Pre-allocated coordinate/energy/status arrays, owned by the buffer pool. Never freed on the hot path —
 *  only grows via resize_for(). */
struct output_buffers_t {
    std::vector<__half> coordinates;
    std::vector<float> energies;
    std::vector<sfapi::conformer_status_t> status;
    std::vector<std::uint8_t> status_codes;
    /** 0 = GPU success, 1 = RDKit fallback success, 2 = failed. */
    std::vector<std::uint8_t> solver_codes;
    std::vector<std::uint8_t> num_atoms_per_molecule;
    std::vector<std::uint16_t> num_heavy_atoms_per_molecule;
    std::vector<std::uint16_t> num_bonds_per_molecule;
    std::vector<float> molecular_weight_per_molecule;
    /** K conformers per molecule, zero if the molecule failed. */
    std::vector<std::uint8_t> n_conformers_per_molecule;
    /** 60 per conformer slot (USR = first 12). */
    std::vector<double> usrcat;
    std::vector<std::uint32_t> retry_indices;

    void resize_for(std::uint64_t num_molecules, std::uint32_t num_conformers, std::uint32_t max_atoms) {
        std::uint64_t const total_conformers = num_molecules * num_conformers;
        std::size_t const coordinate_total = total_conformers * max_atoms * 3u;
        if (coordinates.size() < coordinate_total) coordinates.resize(coordinate_total);
        if (energies.size() < total_conformers) energies.resize(total_conformers);
        if (usrcat.size() < total_conformers * 60u) usrcat.resize(total_conformers * 60u);
        if (status.size() < total_conformers) status.resize(total_conformers);
        if (status_codes.size() < total_conformers) status_codes.resize(total_conformers);
        if (solver_codes.size() < total_conformers) solver_codes.resize(total_conformers);
        if (num_atoms_per_molecule.size() < num_molecules) num_atoms_per_molecule.resize(num_molecules);
        if (num_heavy_atoms_per_molecule.size() < num_molecules) num_heavy_atoms_per_molecule.resize(num_molecules);
        if (num_bonds_per_molecule.size() < num_molecules) num_bonds_per_molecule.resize(num_molecules);
        if (molecular_weight_per_molecule.size() < num_molecules) molecular_weight_per_molecule.resize(num_molecules);
        if (n_conformers_per_molecule.size() < num_molecules) n_conformers_per_molecule.resize(num_molecules);
    }
};

/** Combined input parquet-shards for one `coprocessed-batch`. Owns the Arrow tables that back the SMILES
 *  string_views. The reader-role fills this until total_molecules >= target, then hands off. */
struct coprocessed_batch_accumulator_t {
    std::vector<sfcli::shard_data_t> input_file_data;
    std::vector<sfcli::shard_pair_t const *> input_file_pairs;
    std::vector<std::uint64_t> input_file_boundaries;
    std::vector<std::string_view> combined_smiles;
    std::uint64_t total_molecules = 0;
    bool done = false;

    void add_input_file(sfcli::shard_data_t data, sfcli::shard_pair_t const *pair) {
        input_file_boundaries.push_back(total_molecules);
        std::uint64_t const count = data.num_rows();
        for (auto smiles_view : data.smiles) combined_smiles.push_back(smiles_view);
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

/** `coprocessed-batch` after the `gpu-runner-role`, handed to the `fallback-role`.
 *  The fallback-role patches coordinates/energies in-place via retry_indices. */
struct coprocessed_batch_result_t {
    coprocessed_batch_accumulator_t accumulator;
    output_buffers_t *buffers = nullptr;
    bool done = false;
    /**
     *  @brief When the `gpu-runner-role` picked up this batch, used by the `writer-role` to derive
     *  per-shard wall time, GPU plus fallback plus write, for the single progress line.
     */
    std::chrono::steady_clock::time_point batch_start {};
    /** Number of molecules sent into RDKit fallback. */
    std::uint64_t fallback_attempted = 0;
    /** Number for which RDKit filled at least one slot. */
    std::uint64_t fallback_recovered = 0;
};

/** Bounded ring-buffer producer/consumer channel. Depth picks how many coprocessed-batches the upstream
 *  can stage ahead before back-pressuring; the larger the depth the more memory the pipeline consumes but
 *  the better it tolerates downstream stalls. */
template <typename payload_>
struct handoff {
    std::mutex mutex;
    std::condition_variable has_item;
    std::condition_variable has_space;
    std::vector<payload_> ring;
    std::size_t head = 0;
    std::size_t tail = 0;
    std::size_t count = 0;

    explicit handoff(std::size_t depth) : ring(depth) {}

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

/** Pool of N output buffer sets cycled between gpu-runner, fallback, and writer. N matches handoff depth
 *  + slack so no stage stalls waiting for a free slot. Each output_buffers_t owns its grow-only vectors;
 *  pool.acquire / pool.release is just slot accounting, no allocation. */
struct buffer_pool_t {
    std::mutex mutex;
    std::condition_variable available;
    std::vector<output_buffers_t> buffers;
    std::vector<bool> in_use;

    explicit buffer_pool_t(std::size_t num_slots) : buffers(num_slots), in_use(num_slots, false) {}

    output_buffers_t *acquire() {
        std::unique_lock lock(mutex);
        available.wait(lock, [&] {
            for (auto slot_in_use : in_use)
                if (!slot_in_use) return true;
            return false;
        });
        for (std::size_t slot = 0; slot < in_use.size(); ++slot) {
            if (!in_use[slot]) {
                in_use[slot] = true;
                return &buffers[slot];
            }
        }
        return nullptr;
    }

    void release(output_buffers_t *buffer) {
        std::unique_lock lock(mutex);
        std::size_t const slot = static_cast<std::size_t>(buffer - buffers.data());
        in_use[slot] = false;
        available.notify_one();
    }
};

/**
 *  @brief Pipeline CPU thread allocation.
 *
 *  Partitions the machine's cores across the four sections so their sum stays under num_procs with a small OS
 *  reserve. Proportions (1.5 / 3 / 33 / rest %) assume fallback is the pipeline pacer on drug-like workloads:
 *  every core added to fallback directly reduces per-batch wall, while the other stages plateau past a fixed
 *  thread count. The user can override fallback via `--rdkit-fallback-threads` when the workload's failure
 *  rate is atypical.
 *
 *  All bounds are clamped to [1, num_procs-1] so tiny (8-core dev boxes) and huge (192-core production)
 *  machines both land on sensible splits.
 */
struct thread_budget_t {
    int reader;
    int writer;
    int gpu_runner;
    int fallback;
};

static thread_budget_t compute_thread_budget_(int user_fallback_override, int user_gpu_runner_override) {
    int const num_procs = std::max(1, omp_get_num_procs());
    // Reader is I/O-bound on Arrow deserialization, keep it tiny.
    int const reader = std::clamp(num_procs / 128, 1, 2);
    // Writer is Arrow-serialize-bound, also mostly single-threaded inside Arrow.
    int const writer = std::clamp(num_procs / 64, 2, 4);
    // `gpu-runner-role` and `fallback-role` run concurrently in separate
    // OpenMP sections. On a healthy pipeline (multiple `parquet-shards` per
    // `smallfield_batch` invocation so the depth-3 ring genuinely holds
    // multiple `coprocessed-batches` in flight) at most one of the two
    // roles is mid-parallel-for at any instant — fallback bursts on RDKit
    // while gpu-runner is stream-syncing, and vice versa — so each gets
    // the full `num_procs - 4` budget without mutual starvation.
    int const gpu_runner = user_gpu_runner_override > 0 ? user_gpu_runner_override : std::max(1, num_procs - 4);
    int const fallback = user_fallback_override > 0 ? user_fallback_override : std::max(1, num_procs - 4);
    return {reader, writer, gpu_runner, fallback};
}

/** Cumulative counters shared across pipeline stages. */
struct progress_counters_t {
    std::atomic<std::uint64_t> molecules_completed {0};
    std::atomic<std::uint64_t> input_files_completed {0};
    std::uint64_t total_input_files = 0;
    std::chrono::steady_clock::time_point start_time;
};

static void reader_loop_(std::vector<sfcli::shard_pair_t const *> const &pending, cli_config_t const &config,
                         std::uint32_t batch_target, thread_budget_t const &budget, logger_t const &logger,
                         handoff<coprocessed_batch_accumulator_t> &output_queue) {
    omp_set_num_threads(budget.reader);
    coprocessed_batch_accumulator_t accumulator;

    for (auto const *pair : pending) {
        // Graceful pre-emption: stop enqueuing new shards at this boundary; already-accumulated and
        // in-flight batches still drain and get written below, then the pipeline exits cleanly.
        if (g_stop_requested_.load(std::memory_order_relaxed)) {
            logger.info("stop requested — draining in-flight work and exiting");
            break;
        }
        sfcli::shard_data_t data;
        if (!sfcli::read_shard(pair->input_path, config.smiles_column, data)) {
            logger.error("read failed, skipping: {}", pair->input_path.string());
            continue;
        }
        accumulator.add_input_file(std::move(data), pair);

        if (accumulator.total_molecules >= batch_target) {
            output_queue.send(std::move(accumulator));
            accumulator.clear();
        }
    }

    if (accumulator.total_molecules > 0) { output_queue.send(std::move(accumulator)); }

    coprocessed_batch_accumulator_t sentinel;
    sentinel.done = true;
    output_queue.send(std::move(sentinel));
}

static void gpu_runner_loop_(sfapi::context_t &gpu_context, cli_config_t const &config, thread_budget_t const &budget,
                             handoff<coprocessed_batch_accumulator_t> &input_queue,
                             handoff<coprocessed_batch_result_t> &output_queue, buffer_pool_t &pool,
                             [[maybe_unused]] progress_counters_t &counters) {
    omp_set_num_threads(budget.gpu_runner);

    sfapi::config_t generation_config;
    generation_config.num_conformers = config.num_conformers;
    generation_config.random_seed = config.random_seed;

    for (;;) {
        auto accumulator = input_queue.receive();
        if (accumulator.done) break;
        SF_NVTX_RANGE_COLORED("gpu_runner: preprocess+launch", 0x2E86C1);

        std::uint64_t const num_molecules = accumulator.total_molecules;
        std::uint64_t const total_conformers = num_molecules * config.num_conformers;
        auto const batch_start = std::chrono::steady_clock::now();

        output_buffers_t *buffer = pool.acquire();
        buffer->resize_for(num_molecules, config.num_conformers, config.max_atoms);
        std::fill_n(buffer->energies.data(), total_conformers, 0.0f);
        std::fill_n(buffer->status.data(), total_conformers, sfapi::conformer_status_t::mmff_unconverged_k);

        // Aggregate status is advisory only: per-conformer status sits in `buffer->status` and that is
        // what the fallback + writer act on. Even a batch-level `mmff_unconverged_k` return may still
        // have the majority of conformers succeed, so the aggregate is discarded and the per-conformer
        // codes drive subsequent stages.
        [[maybe_unused]] sfapi::conformer_status_t const batch_status = gpu_context.generate_conformers(
            std::span<std::string_view const>(accumulator.combined_smiles), generation_config, config.max_atoms,
            std::span<__half>(buffer->coordinates.data(), total_conformers * config.max_atoms * 3u),
            std::span<float>(buffer->energies.data(), total_conformers),
            std::span<sfapi::conformer_status_t>(buffer->status.data(), total_conformers),
            std::span<std::uint8_t>(buffer->num_atoms_per_molecule.data(), num_molecules),
            std::span<std::uint16_t>(buffer->num_heavy_atoms_per_molecule.data(), num_molecules),
            std::span<std::uint16_t>(buffer->num_bonds_per_molecule.data(), num_molecules),
            std::span<float>(buffer->molecular_weight_per_molecule.data(), num_molecules));

        // Kernel-OOB guardrail: any CUDA error during this batch poisons the driver context on every device,
        // and every subsequent shard would also silently produce garbage. Park the shards that contributed
        // to this batch into `{input_dir}/../{input_dir_name}_skip/` so the wrapper's next invocation skips
        // them, and exit non-zero so the wrapper relaunches with a fresh CUDA context. One bad shard out of
        // N in the batch gets collateral-parked; in practice each batch has one shard because
        // Enamine/PubChem shards exceed the 800 K target.
        if (gpu_context.cuda_error_count.load() > 0) {
            std::filesystem::path const input_dir = accumulator.input_file_pairs.front()->input_path.parent_path();
            std::filesystem::path const skip_dir = input_dir.parent_path() / (input_dir.filename().string() + "_skip");
            std::error_code error_code;
            std::filesystem::create_directories(skip_dir, error_code);
            for (auto const *pair : accumulator.input_file_pairs) {
                std::filesystem::path const destination = skip_dir / pair->input_path.filename();
                std::filesystem::rename(pair->input_path, destination, error_code);
                std::fprintf(stderr, "[smallfield_batch] CUDA error — parked shard %s -> %s (%s)\n",
                             pair->input_path.string().c_str(), destination.string().c_str(),
                             error_code ? error_code.message().c_str() : "ok");
            }
            std::fprintf(stderr,
                         "[smallfield_batch] aborting invocation after %llu CUDA errors; wrapper will relaunch\n",
                         static_cast<unsigned long long>(gpu_context.cuda_error_count.load()));
            std::_Exit(2);
        }

        std::fill_n(buffer->solver_codes.data(), total_conformers, static_cast<std::uint8_t>(0));
        std::fill_n(buffer->status_codes.data(), total_conformers, static_cast<std::uint8_t>(0));
        // `num_atoms_per_molecule` is populated by `generate_conformers` above for every
        // molecule that survived preprocess; no zero-fill here.

        buffer->retry_indices.clear();
        for (std::uint64_t molecule = 0; molecule < num_molecules; ++molecule) {
            bool has_failure = false;
            for (std::uint32_t conformer = 0; conformer < config.num_conformers; ++conformer) {
                std::uint64_t const slot = molecule * config.num_conformers + conformer;
                buffer->status_codes[slot] = static_cast<std::uint8_t>(buffer->status[slot]);
                if (buffer->status[slot] == sfapi::conformer_status_t::success_k) { buffer->solver_codes[slot] = 0; }
                else {
                    buffer->solver_codes[slot] = 2;
                    has_failure = true;
                }
            }
            if (has_failure) buffer->retry_indices.push_back(static_cast<std::uint32_t>(molecule));
        }

        coprocessed_batch_result_t result;
        result.accumulator = std::move(accumulator);
        result.buffers = buffer;
        result.batch_start = batch_start;
        output_queue.send(std::move(result));
    }

    coprocessed_batch_result_t sentinel;
    sentinel.done = true;
    output_queue.send(std::move(sentinel));
}

/** Compute USR/USRCAT shape descriptors and separate disconnected fragments on the finished
 *  coordinates of a batch, after the RDKit fallback so every slot is final. Pharmacophores are
 *  typed once per molecule (`build_shape_context`); then, per successful conformer, the
 *  structure-of-arrays slot is read into an interleaved `(atom, 3)` block, multi-fragment geometry
 *  is separated in place and written back so the writer emits the fixed coordinates, and the 12-dim
 *  USR + 60-dim USRCAT vectors are computed. Descriptors for non-success or preprocess-failed slots
 *  stay zero (the writer nulls those rows). Reuses the fallback-role thread budget. */
static void compute_shape_descriptors_(cli_config_t const &config, thread_budget_t const &budget,
                                       coprocessed_batch_accumulator_t const &accumulator, output_buffers_t *buffer) {
    std::uint64_t const num_molecules = accumulator.total_molecules;
    std::uint32_t const num_conformers = config.num_conformers;
    std::uint32_t const max_atoms = config.max_atoms;
    sfapi::conformer_slot_layout_t const slot_layout {max_atoms, num_conformers};
    std::uint64_t const total_conformers = num_molecules * num_conformers;
    std::fill_n(buffer->usrcat.data(), total_conformers * 60u, 0.0);
    int const threads = budget.fallback > 0 ? budget.fallback : omp_get_max_threads();

#pragma omp parallel for schedule(dynamic, 256) num_threads(threads) if (threads > 1)
    for (std::uint64_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        std::uint8_t const num_atoms_u8 = buffer->num_atoms_per_molecule[molecule_index];
        if (num_atoms_u8 == 0) continue; // preprocess failed: no coordinates to describe
        sfpost::molecule_shape_context_t const context = sfpost::build_shape_context(
            accumulator.combined_smiles[molecule_index]);
        // Guard: the descriptor parse must reproduce the exact atom order the coordinates were
        // written in. Same parse params make this deterministic; a mismatch means skip, not corrupt.
        if (!context.ok || context.num_atoms != num_atoms_u8) continue;
        std::uint32_t const num_atoms = num_atoms_u8;

        std::array<float, sfapi::max_atoms_per_molecule_k * 3u> interleaved {};
        for (std::uint32_t conformer = 0; conformer < num_conformers; ++conformer) {
            std::uint64_t const flat = molecule_index * num_conformers + conformer;
            if (buffer->status_codes[flat] != static_cast<std::uint8_t>(sfapi::conformer_status_t::success_k)) continue;
            std::size_t const slot_base = slot_layout.slot_offset(molecule_index, conformer);
            for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
                for (std::uint32_t axis = 0; axis < 3u; ++axis)
                    interleaved[atom * 3u + axis] = __half2float(
                        buffer->coordinates[slot_base + slot_layout.component_offset(axis, atom)]);

            std::span<float> coords_view {interleaved.data(), static_cast<std::size_t>(num_atoms) * 3u};
            if (context.num_components > 1) {
                sfpost::separate_fragments(coords_view, context);
                for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
                    for (std::uint32_t axis = 0; axis < 3u; ++axis)
                        buffer->coordinates[slot_base + slot_layout.component_offset(axis, atom)] = __float2half(
                            interleaved[atom * 3u + axis]);
            }

            std::array<double, 60> usrcat {};
            sfpost::usrcat_descriptor(coords_view, context, usrcat);
            std::copy(usrcat.begin(), usrcat.end(), buffer->usrcat.data() + flat * 60u);
        }
    }
}

static void fallback_loop_(cli_config_t const &config, thread_budget_t const &budget,
                           handoff<coprocessed_batch_result_t> &input_queue,
                           handoff<coprocessed_batch_result_t> &output_queue) {
    omp_set_num_threads(budget.fallback);

    for (;;) {
        auto result = input_queue.receive();
        if (result.done) {
            output_queue.send(std::move(result));
            break;
        }

        output_buffers_t *buffer = result.buffers;
        auto &accumulator = result.accumulator;
        std::uint64_t const num_molecules = accumulator.total_molecules;
        std::uint64_t const total_conformers = num_molecules * config.num_conformers;
        SF_NVTX_RANGE_COLORED("fallback: rdkit recovery", 0xE67E22);

        if (!buffer->retry_indices.empty()) {
            sfcli::rdkit_fallback_inputs_t fallback_inputs {
                std::span<std::string_view const>(accumulator.combined_smiles),
                std::span<std::uint32_t const>(buffer->retry_indices),
                config.num_conformers,
                config.max_atoms,
                budget.fallback,
                config.random_seed,
            };
            sfcli::rdkit_fallback_outputs_t fallback_outputs {
                std::span<std::uint8_t>(buffer->status_codes.data(), total_conformers),
                std::span<std::uint8_t>(buffer->solver_codes.data(), total_conformers),
                std::span<std::uint8_t>(buffer->num_atoms_per_molecule.data(), num_molecules),
                std::span<__half>(buffer->coordinates.data(), total_conformers * config.max_atoms * 3u),
                std::span<float>(buffer->energies.data(), total_conformers),
            };
            auto const num_recovered = sfcli::rdkit_fallback(fallback_inputs, fallback_outputs);
            result.fallback_attempted = buffer->retry_indices.size();
            result.fallback_recovered = num_recovered;
        }

        {
            SF_NVTX_RANGE_COLORED("postprocess: descriptors+fragments", 0x8E44AD);
            compute_shape_descriptors_(config, budget, accumulator, buffer);
        }

        output_queue.send(std::move(result));
    }
}

static void writer_loop_(cli_config_t const &config, thread_budget_t const &budget, logger_t const &logger,
                         handoff<coprocessed_batch_result_t> &input_queue, buffer_pool_t &pool,
                         progress_counters_t &counters) {
    omp_set_num_threads(budget.writer);

    for (;;) {
        auto result = input_queue.receive();
        if (result.done) break;

        output_buffers_t *buffer = result.buffers;
        auto &accumulator = result.accumulator;
        std::uint64_t const num_molecules = accumulator.total_molecules;
        SF_NVTX_RANGE_COLORED("writer: parquet", 0x27AE60);

        // Deduplication is intentionally not applied (the K seeds are already diverse — see the
        // corpus regeneration note): every surviving molecule emits all K candidate slots; failed
        // molecules emit zero, which the writer translates into a single all-null row so every input
        // row keeps at least one output row.
        for (std::uint64_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
            buffer->n_conformers_per_molecule[molecule_index] = buffer->num_atoms_per_molecule[molecule_index] != 0
                                                                    ? static_cast<std::uint8_t>(config.num_conformers)
                                                                    : 0u;
        }

        accumulator.input_file_boundaries.push_back(num_molecules);
        for (std::size_t file_index = 0; file_index < accumulator.input_file_pairs.size(); ++file_index) {
            std::uint64_t const mol_begin = accumulator.input_file_boundaries[file_index];
            std::uint64_t const mol_end = accumulator.input_file_boundaries[file_index + 1];
            std::uint64_t const file_molecules = mol_end - mol_begin;
            std::uint64_t const conf_begin = mol_begin * config.num_conformers;
            std::uint64_t const file_conformers = file_molecules * config.num_conformers;
            std::size_t const coord_stride = static_cast<std::size_t>(config.max_atoms) * 3u;

            sfcli::writer_inputs_t writer_inputs {
                accumulator.input_file_pairs[file_index]->input_path.stem().string(),
                config.dataset,
                accumulator.input_file_data[file_index].base_row_index,
                file_molecules,
                config.num_conformers,
                config.max_atoms,
                std::span<std::string_view const>(accumulator.combined_smiles.data() + mol_begin, file_molecules),
                std::span<std::uint8_t const>(buffer->num_atoms_per_molecule.data() + mol_begin, file_molecules),
                std::span<std::uint16_t const>(buffer->num_heavy_atoms_per_molecule.data() + mol_begin, file_molecules),
                std::span<std::uint16_t const>(buffer->num_bonds_per_molecule.data() + mol_begin, file_molecules),
                std::span<float const>(buffer->molecular_weight_per_molecule.data() + mol_begin, file_molecules),
                std::span<std::uint8_t const>(buffer->n_conformers_per_molecule.data() + mol_begin, file_molecules),
                std::span<std::uint8_t const>(buffer->status_codes.data() + conf_begin, file_conformers),
                std::span<__half const>(buffer->coordinates.data() + conf_begin * coord_stride,
                                        file_conformers * coord_stride),
                std::span<float const>(buffer->energies.data() + conf_begin, file_conformers),
                std::span<double const>(buffer->usrcat.data() + conf_begin * 60u, file_conformers * 60u),
            };

            auto const &output_path = accumulator.input_file_pairs[file_index]->output_path;
            if (!sfcli::write_shard(output_path, writer_inputs)) {
                logger.error("write failed: {}", output_path.string());
            }
            else {
                // Touch a 3-byte ".encoded" sentinel so resume runs can skip without
                // re-opening the parquet to count rows.
                std::error_code sentinel_error_code;
                std::filesystem::path sentinel = output_path;
                sentinel += ".encoded";
                FILE *sentinel_file = std::fopen(sentinel.string().c_str(), "wb");
                if (sentinel_file) {
                    // `--resume` skips any shard whose sentinel exists, so a short write would
                    // leave a file that makes the next run skip an unencoded shard.
                    std::size_t const written = std::fwrite("v1\n", 1, 3, sentinel_file);
                    std::fclose(sentinel_file);
                    if (written != 3) {
                        std::error_code remove_error;
                        std::filesystem::remove(sentinel, remove_error);
                        logger.error("sentinel write truncated, removed: {}", sentinel.string());
                    }
                }
                else { logger.error("sentinel write failed: {}", sentinel.string()); }
            }

            counters.molecules_completed.fetch_add(file_molecules);
            counters.input_files_completed.fetch_add(1);
        }

        pool.release(buffer);

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

        // Collapse coprocessed-batch start + fallback + done into one line. Fallback
        // fragment is omitted when no molecule needed RDKit recovery.
        std::string fallback_fragment;
        if (result.fallback_attempted > 0)
            fallback_fragment = fmt::format("  fallback {}→{} rec", thousands(result.fallback_attempted),
                                            thousands(result.fallback_recovered));

        logger.info("{:>4}/{:<4} done {:.0f}s  {} mol/s{}  cumulative {}  avg {} mol/s  ETA {}", thousands(files_done),
                    thousands(counters.total_input_files), batch_seconds,
                    thousands(static_cast<std::uint64_t>(num_molecules / std::max(batch_seconds, 0.001))),
                    fallback_fragment, thousands(cumul_mol), thousands(static_cast<std::uint64_t>(cumul_rate)),
                    smallfield::format_wall_time(eta_wall));
    }
}

} // namespace smallfield::cli

int main(int argc, char **argv) {
    sfcli::cli_config_t config;
    if (!sfcli::parse_cli_(argc, argv, config)) return 2;

    // Graceful pre-emption: SIGINT/SIGTERM ask the reader to stop at the next shard boundary so
    // in-flight shards finish and write. A runner wrapper pairs this with a pause sentinel to hold
    // off relaunch, freeing the GPU on demand.
    std::signal(SIGINT, sfcli::handle_stop_signal_);
    std::signal(SIGTERM, sfcli::handle_stop_signal_);

    logger_t logger;
    logger.verbose_level = config.verbose_level;

    auto shard_pairs = sfcli::inventory_shards(config.input_dir, config.output_dir, config.resume);
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
    // the per-wave table and per-device breakdown emitted from smallfield.cu.
    gpu_context.logger.verbose_level = config.verbose_level;
    gpu_context.logger.start_time = logger.start_time;
    std::uint32_t const num_gpus = config.gpu_ids.empty() ? static_cast<std::uint32_t>(gpu_context.device_ids.size())
                                                          : static_cast<std::uint32_t>(config.gpu_ids.size());
    std::uint32_t const batch_target = config.batch_size > 0 ? config.batch_size : num_gpus * 100'000u;

    sfcli::thread_budget_t const budget = sfcli::compute_thread_budget_(config.rdkit_fallback_threads,
                                                                        config.gpu_runner_threads);
    logger.info("  {} GPUs, coprocessed-batch target {} mols, threads reader={} gpu_runner={} fallback={} writer={}",
                num_gpus, thousands(batch_target), budget.reader, budget.gpu_runner, budget.fallback, budget.writer);

    omp_set_max_active_levels(2);

    sfcli::progress_counters_t counters;
    counters.total_input_files = pending.size();
    counters.start_time = logger.start_time;

    // Pipeline depth picks how many coprocessed-batches the upstream may stage before back-pressuring.
    // Depth 5 lets reader stay 4 ahead of GPU (smooths over slow RDKit-prep batches) and GPU stay 3 ahead
    // of fallback while writer drains; pool sized to depth+1 so no stage ever blocks waiting for a free
    // output buffer. This hides H2D/D2H tails on multi-GPU runs; costs ~70 GB pinned host per extra slot at
    // batch_target = 700K mols. See GPU-utilisation roadmap for sizing rationale.
    constexpr std::size_t pipeline_depth_k = 5;
    sfcli::buffer_pool_t pool {pipeline_depth_k + 1};
    sfcli::handoff<sfcli::coprocessed_batch_accumulator_t> reader_to_gpu {pipeline_depth_k};
    sfcli::handoff<sfcli::coprocessed_batch_result_t> gpu_to_fallback {pipeline_depth_k};
    sfcli::handoff<sfcli::coprocessed_batch_result_t> fallback_to_writer {pipeline_depth_k};

#pragma omp parallel sections num_threads(4)
    {
#pragma omp section
        { sfcli::reader_loop_(pending, config, batch_target, budget, logger, reader_to_gpu); }
#pragma omp section
        { sfcli::gpu_runner_loop_(gpu_context, config, budget, reader_to_gpu, gpu_to_fallback, pool, counters); }
#pragma omp section
        { sfcli::fallback_loop_(config, budget, gpu_to_fallback, fallback_to_writer); }
#pragma omp section
        { sfcli::writer_loop_(config, budget, logger, fallback_to_writer, pool, counters); }
    }

    double const total = std::chrono::duration<double>(std::chrono::steady_clock::now() - counters.start_time).count();
    std::uint64_t const done = counters.molecules_completed.load();
    logger.info("complete  {} mols  {} mol/s", thousands(done),
                thousands(static_cast<std::uint64_t>(done / std::max(total, 0.001))));
    std::uint64_t const cuda_errors = gpu_context.cuda_error_count.load();
    if (cuda_errors > 0) {
        logger.error("{} CUDA errors reported by the library — output is unreliable", thousands(cuda_errors));
        return 2;
    }
    return 0;
}
