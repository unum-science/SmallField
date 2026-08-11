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
#include <cstdio>  // `stderr`, `std::fopen`, `std::fwrite`
#include <cstdlib> // `std::strtol`

#include <algorithm>          // `std::max`
#include <array>              // `std::array`
#include <atomic>             // `std::atomic`
#include <charconv>           // `std::from_chars`
#include <chrono>             // `std::chrono::steady_clock`, `std::chrono::duration`
#include <condition_variable> // `std::condition_variable`
#include <filesystem>         // `std::filesystem::path`, `std::filesystem::create_directories`
#include <limits>             // `std::numeric_limits`
#include <mutex>              // `std::mutex`, `std::unique_lock`
#include <span>               // `std::span`
#include <stdexcept>          // `std::runtime_error`
#include <string>             // `std::string`
#include <string_view>        // `std::string_view`
#include <system_error>       // `std::errc`
#include <vector>             // `std::vector`

#include <argparse/argparse.hpp>
#include <omp.h>
#include <cuda_fp16.h>

#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/api/smallfield.hpp"
#include "smallfield/log.hpp"
#include "smallfield/nvtx.hpp"
#include "smallfield/postprocess/fragment_separation.hpp"
#include "smallfield/postprocess/shape_descriptors.hpp"

#include "cli/parquet_reader.hpp"
#include "cli/parquet_writer.hpp"
#include "cli/rdkit_fallback.hpp"
#include "cli/shard_inventory.hpp"

namespace fs = std::filesystem;
namespace sfapi = smallfield::api;
namespace sfcli = smallfield::cli;
namespace sfpost = smallfield::postprocess;
using smallfield::logger_t;

namespace smallfield::cli {

/** Set by SIGINT/SIGTERM. The reader-role stops feeding new shards at the next shard boundary so
 *  in-flight batches drain and their shards are written before the pipeline exits cleanly (exit 0).
 *  This is the graceful pre-emption path: no partial-shard loss, and resume skips the completed
 *  shards. Async-signal-safe: a lock-free atomic flag, nothing else touched in the handler.
 *
 *  `handle_stop_signal_` keeps C language linkage for `std::signal`, which an ordinary `static`
 *  function cannot combine with, so the unnamed namespace supplies the internal linkage here. */
namespace {
std::atomic<bool> stop_requested_ {false};
extern "C" void handle_stop_signal_(int) { stop_requested_.store(true, std::memory_order_relaxed); }
} // namespace

/** Ceiling on `--rdkit-fallback-threads` and `--gpu-runner-threads`: an OpenMP team wider than this is a
 *  mistyped token, not a machine. */
constexpr int max_threads_k = 4096;

/** Ceiling on `--batch-size` in molecules. Six batches are in flight at once at roughly 100 KB of pinned
 *  host memory per molecule per slot — 70 GB per slot at 700 K — so a larger target names a batch no
 *  machine this runs on can hold. */
constexpr std::uint32_t max_batch_size_k = 4'000'000;

/** Highest tier `--verbose` names: 1 is the per-`embedding-wave` summary, 2 adds the per-device breakdown. */
constexpr int max_verbose_level_k = 2;

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
    restart_policy_t restart_policy = restart_policy_t::resume_k;
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
  --num-conformers K          conformers per molecule, 1 to 16 (default: 3)
  --max-atoms N               per-molecule atom cap, 1 to 192 (default: 128)
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

/** Outcome of a command line: a usable configuration, a help request, or a rejected command line. */
enum class cli_parse_t {
    parsed_k,
    help_requested_k,
    invalid_k,
};

/** Why a command-line token was accepted or refused as a number. */
enum class number_parse_t {
    /** The whole token is a number inside the accepted range. */
    parsed_k,
    /** The token is not a number, or carries trailing characters. */
    malformed_k,
    /** The token is a number the destination type cannot hold. */
    out_of_range_k,
    /** The token parses but falls under the caller's floor. */
    below_minimum_k,
    /** The token parses but exceeds the caller's ceiling. */
    above_maximum_k,
};

/**
 *  @brief Parse @p text whole as a number in `[minimum, maximum]`, leaving @p number untouched on refusal.
 *
 *  Partial parses are refusals: `12x` is malformed, not twelve. The destination keeps its prior value unless
 *  the token is accepted, so a caller may seed it with the default and read it back either way.
 *
 *  @param[in] text Token as it appeared on the command line.
 *  @param[in] minimum Smallest value the flag accepts.
 *  @param[in] maximum Largest value the flag accepts.
 *  @param[out] number Value the token denotes; untouched unless the whole token parses inside the bounds.
 */
template <typename number_type_>
static number_parse_t parse_number_(std::string_view text, number_type_ minimum, number_type_ maximum,
                                    number_type_ &number) {
    number_type_ parsed {};
    auto const [unparsed, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error == std::errc::result_out_of_range) return number_parse_t::out_of_range_k;
    if (error != std::errc {} || unparsed != text.data() + text.size()) return number_parse_t::malformed_k;
    if (parsed < minimum) return number_parse_t::below_minimum_k;
    if (parsed > maximum) return number_parse_t::above_maximum_k;
    number = parsed;
    return number_parse_t::parsed_k;
}

/** Sentence fragment naming the refusal, meant to follow the flag name in a diagnostic. */
static char const *describe_number_parse_(number_parse_t status) {
    switch (status) {
    case number_parse_t::parsed_k: return "parsed";
    case number_parse_t::malformed_k: return "is not a whole number";
    case number_parse_t::out_of_range_k: return "does not fit the destination type";
    case number_parse_t::below_minimum_k: return "is below the accepted minimum";
    case number_parse_t::above_maximum_k: return "is above the accepted maximum";
    }
    return "was refused";
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

/**
 *  @brief Bind a string flag straight to its destination, so the parsed value never sits anywhere but the config.
 *
 *  `append` marks the flag repeatable, which is what keeps the last occurrence winning: without it `argparse` rejects
 *  a second occurrence outright, and a wrapper that appends an override to a fixed prefix stops working.
 */
template <typename string_type_>
static void add_string_(argparse::ArgumentParser &parser, char const *flag, string_type_ &destination) {
    parser.add_argument(flag).append().action([&destination](std::string const &text) { destination = text; });
}

/**
 *  @brief Bind a numeric flag bounded by `[minimum, maximum]`, naming both the flag and the reason on refusal.
 *
 *  The conversion stays ours rather than `argparse`'s `scan`, whose diagnostic is `pattern 'abc' does not match to
 *  the end` — it never names the flag the operator actually typed, and it has no bounds to enforce.
 */
template <typename number_type_>
static void add_number_(argparse::ArgumentParser &parser, char const *flag, number_type_ minimum, number_type_ maximum,
                        number_type_ &number) {
    parser.add_argument(flag).append().action([flag, minimum, maximum, &number](std::string const &text) {
        number_parse_t const status = parse_number_(text, minimum, maximum, number);
        if (status != number_parse_t::parsed_k)
            throw std::runtime_error(fmt::format("{} {}, got: {}", flag, describe_number_parse_(status), text));
    });
}

static cli_parse_t parse_cli_(int argc, char **argv, cli_config_t &config) {
    // `default_arguments::none` suppresses the `-h` and `--version` that `argparse` would otherwise add: its own
    // generated help cannot express `--verbose[=LEVEL]` or the three names sharing one field, so `print_usage_`
    // stays the help text and `-h` is bound by hand below.
    argparse::ArgumentParser parser {"smallfield_batch", "", argparse::default_arguments::none};
    bool help_requested = false;

    add_string_(parser, "--input", config.input_dir);
    add_string_(parser, "--output", config.output_dir);
    add_string_(parser, "--smiles-column", config.smiles_column);
    add_string_(parser, "--dataset", config.dataset);
    // Bounds are enforced in the parse, so a refused token never reaches a `resize` or a kernel launch. Zero
    // conformers asks the library for a wave of nothing, which it reports as a CUDA fault — and the fault path
    // parks the operator's input shards. Zero atoms makes the conformer slot stride zero, and an atom cap above
    // the library's own cap lets a molecule overrun the interleaved scratch the descriptor pass gathers into.
    add_number_(parser, "--num-conformers", 1u, sfapi::max_conformer_count_k, config.num_conformers);
    add_number_(parser, "--max-atoms", 1u, sfapi::max_atoms_per_molecule_k, config.max_atoms);
    // Three names for one field, registered separately rather than as aliases of one option, so a rejected value is
    // reported under the name the operator typed. Zero is the documented "auto", so it is the floor rather than a
    // refusal.
    add_number_(parser, "--batch-size", 0u, max_batch_size_k, config.batch_size);
    add_number_(parser, "--coprocessed-batch-size", 0u, max_batch_size_k, config.batch_size);
    add_number_(parser, "--chunk-size", 0u, max_batch_size_k, config.batch_size);
    // Zero threads is the documented "auto split", and a negative team is a typo rather than a smaller one.
    add_number_(parser, "--rdkit-fallback-threads", 0, max_threads_k, config.rdkit_fallback_threads);
    add_number_(parser, "--gpu-runner-threads", 0, max_threads_k, config.gpu_runner_threads);
    // Every 64-bit value is a seed, so the bounds only reject a token that is not one.
    add_number_(parser, "--seed", std::uint64_t {0}, std::numeric_limits<std::uint64_t>::max(), config.random_seed);
    // The comma-separated device list fills the caller's vector; nothing is returned by value.
    parser.add_argument("--gpus").append().action([&config](std::string const &text) {
        if (!parse_int_list_(text.c_str(), config.gpu_ids))
            throw std::runtime_error(fmt::format("--gpus requires comma-separated device IDs, got: {}", text));
    });
    // `--restart` and `--resume` write the same field, so each is an action rather than a queried flag: that is what
    // makes the last one on the command line win.
    parser.add_argument("--restart").flag().append().action([&config](std::string const &) {
        config.restart_policy = restart_policy_t::restart_k;
    });
    parser.add_argument("--resume").flag().append().action([&config](std::string const &) {
        config.restart_policy = restart_policy_t::resume_k;
    });
    // `--verbose` alone is level 1 and `--verbose=LEVEL` names it. `nargs(0, 1)` consumes nothing in the first case,
    // so the action never fires, which is how the two are told apart.
    bool verbose_level_given = false;
    parser.add_argument("--verbose").nargs(0, 1).append().action([&](std::string const &text) {
        number_parse_t const status = parse_number_(text, 0, max_verbose_level_k, config.verbose_level);
        if (status != number_parse_t::parsed_k)
            throw std::runtime_error(fmt::format("--verbose {}, got: {}", describe_number_parse_(status), text));
        verbose_level_given = true;
    });
    parser.add_argument("-h", "--help").flag().append().action([&](std::string const &) {
        print_usage_();
        help_requested = true;
    });

    try {
        parser.parse_args(argc, argv);
        if (help_requested) return cli_parse_t::help_requested_k;
        if (parser.is_used("--verbose") && !verbose_level_given) config.verbose_level = 1;
    }
    catch (std::exception const &error) {
        // `argparse` reports every rejection by throwing, so this is the only place a bad command line surfaces.
        // `--help` alongside a rejected flag is still a help request, since the usage is what the operator asked for.
        if (help_requested) return cli_parse_t::help_requested_k;
        fmt::print(stderr, "{}\n", error.what());
        return cli_parse_t::invalid_k;
    }

    if (config.input_dir.empty()) {
        print_usage_();
        return cli_parse_t::invalid_k;
    }
    // --output defaults to --input: the *.3D.parquet sister files live next to the
    // sources unless the caller explicitly redirects them elsewhere.
    if (config.output_dir.empty()) config.output_dir = config.input_dir;
    return cli_parse_t::parsed_k;
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
        if (stop_requested_.load(std::memory_order_relaxed)) {
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

/** How the `gpu-runner-role` left its loop: normally, or with the driver context poisoned. */
enum class gpu_runner_outcome_t {
    completed_k,
    cuda_fault_k,
};

static gpu_runner_outcome_t gpu_runner_loop_(sfapi::context_t &gpu_context, cli_config_t const &config,
                                             thread_budget_t const &budget, logger_t const &logger,
                                             handoff<coprocessed_batch_accumulator_t> &input_queue,
                                             handoff<coprocessed_batch_result_t> &output_queue, buffer_pool_t &pool,
                                             [[maybe_unused]] progress_counters_t &counters) {
    omp_set_num_threads(budget.gpu_runner);

    sfapi::config_t generation_config;
    generation_config.num_conformers = config.num_conformers;
    generation_config.random_seed = config.random_seed;
    gpu_runner_outcome_t outcome = gpu_runner_outcome_t::completed_k;

    for (;;) {
        auto accumulator = input_queue.receive();
        if (accumulator.done) break;
        // Once the driver context is poisoned nothing further is launched, but the loop keeps draining so
        // the reader-role never blocks on a full ring and the whole pipeline can shut down in order.
        if (outcome == gpu_runner_outcome_t::cuda_fault_k) continue;
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
        // them, then stop feeding the GPU and let the pipeline drain so `main` can return non-zero and the
        // wrapper relaunches with a fresh CUDA context. One bad shard out of N in the batch gets
        // collateral-parked; in practice each batch has one shard because Enamine/PubChem shards exceed the
        // 800 K target.
        if (gpu_context.cuda_error_count.load() > 0) {
            std::filesystem::path const input_dir = accumulator.input_file_pairs.front()->input_path.parent_path();
            std::filesystem::path const skip_dir = input_dir.parent_path() / (input_dir.filename().string() + "_skip");
            std::error_code error_code;
            std::filesystem::create_directories(skip_dir, error_code);
            for (auto const *pair : accumulator.input_file_pairs) {
                std::filesystem::path const destination = skip_dir / pair->input_path.filename();
                std::filesystem::rename(pair->input_path, destination, error_code);
                logger.error("CUDA error — parked shard {} -> {} ({})", pair->input_path.string(), destination.string(),
                             error_code ? error_code.message() : "ok");
            }
            logger.error("aborting invocation after {} CUDA errors; wrapper will relaunch",
                         fmt::group_digits(gpu_context.cuda_error_count.load()));
            // Ask the reader-role to stop at its next shard boundary; this batch's output is discarded.
            stop_requested_.store(true, std::memory_order_relaxed);
            pool.release(buffer);
            outcome = gpu_runner_outcome_t::cuda_fault_k;
            continue;
        }

        std::fill_n(buffer->solver_codes.data(), total_conformers, static_cast<std::uint8_t>(0));
        std::fill_n(buffer->status_codes.data(), total_conformers, static_cast<std::uint8_t>(0));
        // `num_atoms_per_molecule` is populated by `generate_conformers` above for every
        // molecule that survived preprocess; no zero-fill here.

        buffer->retry_indices.clear();
        for (std::uint64_t molecule = 0; molecule < num_molecules; ++molecule) {
            bool has_failure = false;
            // Deliberately non-short-circuiting: every conformer slot of this molecule is written here, and
            // both arrays were just zero-filled, where zero encodes `success_k` in `status_codes` and a GPU
            // success in `solver_codes`. Breaking on the first failure would therefore leave the later slots
            // reading as successful conformers, and the writer would emit them to parquet with the all-zero
            // coordinates they still hold. `has_failure` is the summary; it is not a reason to stop.
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
    return outcome;
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
            sfcli::gather_slot_into_interleaved(buffer->coordinates.data() + slot_base, slot_layout, num_atoms,
                                                interleaved.data());

            std::span<float> coords_view {interleaved.data(), static_cast<std::size_t>(num_atoms) * 3u};
            if (context.num_components > 1) {
                sfpost::separate_fragments(coords_view, context);
                sfcli::scatter_interleaved_into_slot(interleaved.data(), slot_layout, num_atoms,
                                                     buffer->coordinates.data() + slot_base);
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
            // `write_shard` names the column or the file that stopped it in the log, so a second generic
            // "write failed" line here would bury the reason it already reported. The sentinel a `--resume`
            // run keys on is only written on the success path, so a refused write never leaves one behind.
            sfcli::shard_write_report_t const report = sfcli::write_shard(output_path, writer_inputs, &logger);
            if (report.status == sfcli::shard_write_t::written_k) {
                // Touch a 3-byte ".encoded" sentinel so resume runs can skip without
                // re-opening the parquet to count rows.
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
            fallback_fragment = fmt::format("  fallback {}→{} rec", fmt::group_digits(result.fallback_attempted),
                                            fmt::group_digits(result.fallback_recovered));

        logger.info("{:>4}/{:<4} done {:.0f}s  {} mol/s{}  cumulative {}  avg {} mol/s  ETA {}",
                    fmt::group_digits(files_done), fmt::group_digits(counters.total_input_files), batch_seconds,
                    fmt::group_digits(static_cast<std::uint64_t>(num_molecules / std::max(batch_seconds, 0.001))),
                    fallback_fragment, fmt::group_digits(cumul_mol),
                    fmt::group_digits(static_cast<std::uint64_t>(cumul_rate)),
                    smallfield::format_wall_time(eta_wall));
    }
}

} // namespace smallfield::cli

int main(int argc, char **argv) {
    sfcli::cli_config_t config;
    switch (sfcli::parse_cli_(argc, argv, config)) {
    case sfcli::cli_parse_t::parsed_k: break;
    case sfcli::cli_parse_t::help_requested_k: return 0;
    case sfcli::cli_parse_t::invalid_k: return 2;
    }

    // Graceful pre-emption: SIGINT/SIGTERM ask the reader to stop at the next shard boundary so
    // in-flight shards finish and write. A runner wrapper pairs this with a pause sentinel to hold
    // off relaunch, freeing the GPU on demand.
    std::signal(SIGINT, sfcli::handle_stop_signal_);
    std::signal(SIGTERM, sfcli::handle_stop_signal_);

    logger_t logger;
    logger.verbose_level = config.verbose_level;

    auto shard_pairs = sfcli::inventory_shards(config.input_dir, config.output_dir, config.restart_policy);
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

    logger.info("smallfield_batch: {} input-files, {} pending, {} complete", fmt::group_digits(shard_pairs.size()),
                fmt::group_digits(pending.size()), fmt::group_digits(skipped));

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
                num_gpus, fmt::group_digits(batch_target), budget.reader, budget.gpu_runner, budget.fallback,
                budget.writer);

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

    auto gpu_runner_outcome = sfcli::gpu_runner_outcome_t::completed_k;

#pragma omp parallel sections num_threads(4)
    {
#pragma omp section
        { sfcli::reader_loop_(pending, config, batch_target, budget, logger, reader_to_gpu); }
#pragma omp section
        {
            gpu_runner_outcome = sfcli::gpu_runner_loop_(gpu_context, config, budget, logger, reader_to_gpu,
                                                         gpu_to_fallback, pool, counters);
        }
#pragma omp section
        { sfcli::fallback_loop_(config, budget, gpu_to_fallback, fallback_to_writer); }
#pragma omp section
        { sfcli::writer_loop_(config, budget, logger, fallback_to_writer, pool, counters); }
    }

    double const total = std::chrono::duration<double>(std::chrono::steady_clock::now() - counters.start_time).count();
    std::uint64_t const done = counters.molecules_completed.load();
    logger.info("complete  {} mols  {} mol/s", fmt::group_digits(done),
                fmt::group_digits(static_cast<std::uint64_t>(done / std::max(total, 0.001))));
    // Exit codes mean one thing each: 1 is a run that failed, 2 is reserved for a refused command line.
    if (gpu_runner_outcome == sfcli::gpu_runner_outcome_t::cuda_fault_k) return 1;
    std::uint64_t const cuda_errors = gpu_context.cuda_error_count.load();
    if (cuda_errors > 0) {
        logger.error("{} CUDA errors reported by the library — output is unreliable", fmt::group_digits(cuda_errors));
        return 1;
    }
    return 0;
}
