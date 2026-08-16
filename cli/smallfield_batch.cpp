/**
 *  @file cli/smallfield_batch.cpp
 *  @brief Restartable batch processor for bulk conformer generation.
 *
 *  Four `pipeline-roles` — see GLOSSARY.md — with depth-5 ring-buffer handoffs:
 *  @verbatim
 *    reader-role     → reads parquet-shards, combines into coprocessed-batches
 *    gpu-runner-role → runs `generate_conformers` on each coprocessed-batch
 *    fallback-role   → runs RDKit on conformer slots the GPU could not fill
 *    writer-role     → emits one output parquet-shard per consumed input shard
 *  @endverbatim
 *
 *  The reader-role accumulates parquet-shards until the coprocessed-batch size target is reached, which is
 *  what saturates every GPU per `generate_conformers` call. The fallback is conformer-granular: only the
 *  per-slot failures are re-embedded, and partial GPU successes are kept.
 */

#include <csignal>
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstdio>  // `stderr`, `std::fopen`, `std::fwrite`
#include <cstdlib> // `std::strtol`

#include <algorithm>          // `std::max`
#include <array>              // `std::array`
#include <atomic>             // `std::atomic`
#include <chrono>             // `std::chrono::steady_clock`, `std::chrono::duration`
#include <condition_variable> // `std::condition_variable`
#include <filesystem>         // `std::filesystem::path`, `std::filesystem::create_directories`
#include <limits>             // `std::numeric_limits`
#include <mutex>              // `std::mutex`, `std::unique_lock`
#include <stdexcept>          // `std::runtime_error`
#include <string>             // `std::string`
#include <string_view>        // `std::string_view`
#include <system_error>       // `std::error_code`
#include <vector>             // `std::vector`

#include <argparse/argparse.hpp>
#include <omp.h>
#include <cuda_fp16.h>

#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/api/smallfield.hpp"
#include "smallfield/log.hpp"
#include "smallfield/molecule_parse.hpp"
#include "smallfield/nvtx.hpp"
#include "smallfield/postprocess/fragment_separation.hpp"
#include "smallfield/postprocess/shape_descriptors.hpp"
#include "smallfield/types/views.hpp"

#include "cli/cli_arguments.hpp"
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

/* Set by SIGINT/SIGTERM. The reader-role stops feeding new shards at the next shard boundary, so
 * in-flight batches drain and write and the run still exits zero; a runner wrapper pairs this with
 * a pause sentinel to hold off relaunch and free the GPU on demand.
 *
 * `handle_stop_signal_` keeps C language linkage for `std::signal`, which an ordinary `static`
 * function cannot combine with, so the unnamed namespace supplies the internal linkage here. */
namespace {
std::atomic<bool> stop_requested_ {false};
extern "C" void handle_stop_signal_(int) { stop_requested_.store(true, std::memory_order_relaxed); }
} // namespace

/** The one name this binary prints, so the announcement and the usage cannot drift from the executable. */
constexpr char const *executable_name_k = "smallfield_batch";

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
    /** Empty omits the key from the shard's metadata. */
    std::string dataset;
    std::uint32_t num_conformers = 3;
    std::uint32_t max_atoms = sfapi::max_atoms_per_molecule_k;
    /** Molecules per `coprocessed-batch`; zero is auto, 100 K per visible GPU. */
    std::uint32_t batch_size = 0;
    std::vector<int> gpu_ids;
    /* Zero threads is auto on both, resolved in `compute_thread_budget_`. */
    int rdkit_fallback_threads = 0;
    int gpu_runner_threads = 0;
    restart_policy_t restart_policy = restart_policy_t::resume_k;
    std::uint64_t random_seed = 42;
    /** Costs a device region, a device-to-host copy and a host pass, all per conformer slot. */
    sfapi::polish_telemetry_t polish_telemetry = sfapi::polish_telemetry_t::disabled_k;
    int verbose_level = 0;
};

static void print_usage_() {
    fmt::print(stderr, "Usage: {} --input DIR [--output DIR] [options]\n", executable_name_k);
    fmt::print(stderr, R"(  --input DIR                 folder of input parquet-shards (*.parquet)
  --output DIR                folder for output parquet-shards (*.3D.parquet)
                              (default: same as --input — sister files alongside)
  --smiles-column NAME        input column name (default: smiles)
  --dataset NAME              provenance dataset name embedded in output metadata (e.g. gdb13)
  --num-conformers K          conformers per molecule, 1 to 16 (default: 3)
  --max-atoms N               per-molecule atom cap, 1 to 192 (default: 192)
  --batch-size N              molecules per coprocessed-batch (default: auto = 100K per GPU)
  --coprocessed-batch-size N         alias for --batch-size
  --chunk-size N              alias for --batch-size
  --gpus a,b,c                GPU device IDs (default: all visible)
  --rdkit-fallback-threads N  OpenMP threads for fallback-role (default: num_procs - 4)
  --gpu-runner-threads N      OpenMP threads for gpu-runner-role preprocess (default: num_procs - 4)
  --restart                   overwrite already-complete output shards
  --polish-telemetry          one POLISH line per wave, paid for with a per-conformer
                              device region and its readback; disabled by default
  --seed N                    RNG seed for both GPU and RDKit fallback
  --verbose[=LEVEL]           library diagnostics: 1 = wave table, 2 = +per-device
)");
}

/**
 *  @brief Parse a comma-separated device list into @p out, which is cleared first.
 *
 *  @return Whether every token was a number; a refused list leaves @p out holding whatever parsed before it.
 */
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

static cli_parse_t parse_cli_(int argc, char **argv, cli_config_t &config) {
    // `default_arguments::none` suppresses `argparse`'s own `-h` and `--version`, whose generated help cannot
    // express `--verbose[=LEVEL]` or three names sharing one field. `-h` is bound by hand below.
    argparse::ArgumentParser parser {executable_name_k, "", argparse::default_arguments::none};
    bool help_requested = false;

    add_string(parser, "--input", config.input_dir);
    add_string(parser, "--output", config.output_dir);
    add_string(parser, "--smiles-column", config.smiles_column);
    add_string(parser, "--dataset", config.dataset);
    // Zero conformers asks the library for a wave of nothing, which it reports as a CUDA fault; zero atoms makes
    // the conformer slot stride zero; an atom cap above the library's own lets a molecule overrun the interleaved
    // scratch the descriptor pass gathers into. All three are refused here rather than at the launch.
    add_number(parser, "--num-conformers", 1u, sfapi::max_conformer_count_k, config.num_conformers);
    add_number(parser, "--max-atoms", 1u, sfapi::max_atoms_per_molecule_k, config.max_atoms);
    // Three names for one field, registered separately rather than as aliases, so a rejected value is reported
    // under the name the operator typed.
    add_number(parser, "--batch-size", 0u, max_batch_size_k, config.batch_size);
    add_number(parser, "--coprocessed-batch-size", 0u, max_batch_size_k, config.batch_size);
    add_number(parser, "--chunk-size", 0u, max_batch_size_k, config.batch_size);
    add_number(parser, "--rdkit-fallback-threads", 0, max_threads_k, config.rdkit_fallback_threads);
    add_number(parser, "--gpu-runner-threads", 0, max_threads_k, config.gpu_runner_threads);
    // Every 64-bit value is a seed, so the bounds only reject a token that is not one.
    add_number(parser, "--seed", std::uint64_t {0}, std::numeric_limits<std::uint64_t>::max(), config.random_seed);
    parser.add_argument("--gpus").append().action([&config](std::string const &text) {
        if (!parse_int_list_(text.c_str(), config.gpu_ids))
            throw std::runtime_error(fmt::format("--gpus requires comma-separated device IDs, got: {}", text));
    });
    // Each is an action rather than a queried flag, which is what makes the last one on the line win.
    parser.add_argument("--restart").flag().append().action([&config](std::string const &) {
        config.restart_policy = restart_policy_t::restart_k;
    });
    parser.add_argument("--resume").flag().append().action([&config](std::string const &) {
        config.restart_policy = restart_policy_t::resume_k;
    });
    // The same last-one-wins shape, over the library's own two-state telemetry switch.
    parser.add_argument("--polish-telemetry").flag().append().action([&config](std::string const &) {
        config.polish_telemetry = sfapi::polish_telemetry_t::summarized_k;
    });
    parser.add_argument("--no-polish-telemetry").flag().append().action([&config](std::string const &) {
        config.polish_telemetry = sfapi::polish_telemetry_t::disabled_k;
    });
    // Under `nargs(0, 1)` a bare `--verbose` consumes nothing and the action never fires, which is how it is
    // told apart from `--verbose=LEVEL` below.
    bool verbose_level_given = false;
    parser.add_argument("--verbose").nargs(0, 1).append().action([&](std::string const &text) {
        number_parse_t const status = parse_number(text, 0, max_verbose_level_k, config.verbose_level);
        if (status != number_parse_t::parsed_k)
            throw std::runtime_error(fmt::format("--verbose {}, got: {}", describe_number_parse(status), text));
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
        // `--help` alongside a rejected flag is still a help request.
        if (help_requested) return cli_parse_t::help_requested_k;
        fmt::print(stderr, "{}\n", error.what());
        return cli_parse_t::invalid_k;
    }

    if (config.input_dir.empty()) {
        print_usage_();
        return cli_parse_t::invalid_k;
    }
    if (config.output_dir.empty()) config.output_dir = config.input_dir;
    return cli_parse_t::parsed_k;
}

/**
 *  @brief Pre-allocated coordinate, energy and status arrays, owned by the buffer pool.
 *
 *  `resize_for` only ever grows, so a run allocates its widest batch once and nothing on the hot path
 *  frees. The spans handed downstream carry the extent of the batch in flight, not of the allocation.
 */
struct output_buffers_t {
    std::vector<__half> coordinates;
    std::vector<float> energies;
    std::vector<sfapi::conformer_status_t> status;
    std::vector<std::uint8_t> num_atoms_per_molecule;
    std::vector<std::uint16_t> num_heavy_atoms_per_molecule;
    std::vector<std::uint16_t> num_bonds_per_molecule;
    std::vector<float> molecular_weight_per_molecule;
    /** Zero if the molecule failed preprocessing. */
    std::vector<std::uint8_t> num_conformers_per_molecule;
    /** `usrcat_components_k` per conformer slot. */
    std::vector<double> usrcat;
    /** Whether a molecule's shape context was rebuilt, so the writer can tell an absent descriptor from one. */
    std::vector<sfcli::shape_described_t> shape_described;
    std::vector<std::uint32_t> retry_indices;

    void resize_for(std::uint64_t num_molecules, sfapi::conformer_slot_layout_t const &layout) {
        std::size_t const total_conformers = static_cast<std::size_t>(num_molecules) * layout.num_conformers;
        auto grow_ = [](auto &buffer, std::size_t count) {
            if (buffer.size() < count) buffer.resize(count);
        };
        grow_(coordinates, total_conformers * layout.slot_stride());
        grow_(energies, total_conformers);
        grow_(usrcat, total_conformers * sfcli::usrcat_components_k);
        grow_(status, total_conformers);
        grow_(num_atoms_per_molecule, num_molecules);
        grow_(num_heavy_atoms_per_molecule, num_molecules);
        grow_(num_bonds_per_molecule, num_molecules);
        grow_(molecular_weight_per_molecule, num_molecules);
        grow_(num_conformers_per_molecule, num_molecules);
        grow_(shape_described, num_molecules);
    }
};

/** Whether a handoff payload carries work, or is the marker that closes the stream behind it. */
enum class pipeline_item_t : bool {
    batch_k,
    end_of_stream_k,
};

/** Combined input parquet-shards for one `coprocessed-batch`, owning the Arrow tables its SMILES view. */
struct coprocessed_batch_accumulator_t {
    std::vector<sfcli::shard_data_t> input_file_data;
    std::vector<sfcli::shard_pair_t const *> input_file_pairs;
    /** First molecule of each input file; the last file runs to the end. */
    std::vector<std::uint64_t> input_file_boundaries;
    /** Every input file's SMILES end to end; its extent @b is the batch's molecule count. */
    std::vector<std::string_view> combined_smiles;
    pipeline_item_t item = pipeline_item_t::batch_k;

    [[nodiscard]] std::uint64_t num_molecules() const noexcept { return combined_smiles.size(); }

    [[nodiscard]] std::uint64_t first_molecule_of(std::size_t file_index) const noexcept {
        return input_file_boundaries[file_index];
    }
    [[nodiscard]] std::uint64_t molecules_of(std::size_t file_index) const noexcept {
        std::uint64_t const end = file_index + 1 < input_file_boundaries.size() ? input_file_boundaries[file_index + 1]
                                                                                : num_molecules();
        return end - input_file_boundaries[file_index];
    }

    void add_input_file(sfcli::shard_data_t data, sfcli::shard_pair_t const *pair) {
        input_file_boundaries.push_back(num_molecules());
        for (auto smiles_view : data.smiles) combined_smiles.push_back(smiles_view);
        input_file_data.push_back(std::move(data));
        input_file_pairs.push_back(pair);
    }

    void clear() {
        input_file_data.clear();
        input_file_pairs.clear();
        input_file_boundaries.clear();
        combined_smiles.clear();
        item = pipeline_item_t::batch_k;
    }
};

/** The slot an `end_of_stream_k` payload claims, which is none: it carries no batch and no buffers. */
constexpr std::size_t unclaimed_buffer_slot_k = static_cast<std::size_t>(-1);

/** A `coprocessed-batch` after the `gpu-runner-role`, which the `fallback-role` patches in place. */
struct coprocessed_batch_result_t {
    coprocessed_batch_accumulator_t accumulator;
    /** Buffer-pool slot this batch holds, released by the writer-role; unclaimed on a sentinel. */
    std::size_t buffer_slot = unclaimed_buffer_slot_k;
    pipeline_item_t item = pipeline_item_t::batch_k;
    /** When the `gpu-runner-role` picked the batch up, so the progress line spans GPU, fallback and write. */
    std::chrono::steady_clock::time_point batch_start {};
    /** What the batch's refusals cost and bought, by the status that decided each molecule. */
    sfcli::fallback_report_t fallback_report {};
    /** Molecules with coordinates whose shape context could not be rebuilt, so they ship no USRCAT. */
    std::uint64_t molecules_undescribed = 0;
};

/** Bounded ring-buffer producer/consumer channel; see `run_pipeline_` for what its depth buys. */
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

/**
 *  @brief Pool of output buffer sets cycled between gpu-runner, fallback, and writer.
 *
 *  Each `output_buffers_t` owns its grow-only vectors, so acquire and release are slot accounting and never
 *  allocate. A batch carries the slot index rather than a pointer, because a slot travels through two
 *  queues before it is released.
 */
struct buffer_pool_t {
    std::mutex mutex;
    std::condition_variable available;
    std::vector<output_buffers_t> buffers;
    std::vector<bool> in_use;

    explicit buffer_pool_t(std::size_t num_slots) : buffers(num_slots), in_use(num_slots, false) {}

    [[nodiscard]] output_buffers_t &at(std::size_t slot) noexcept { return buffers[slot]; }

    /** Claim a free slot, blocking until one is released. */
    [[nodiscard]] std::size_t acquire() {
        std::unique_lock lock(mutex);
        auto first_free = [&] { return std::find(in_use.begin(), in_use.end(), false); };
        available.wait(lock, [&] { return first_free() != in_use.end(); });
        std::size_t const slot = static_cast<std::size_t>(first_free() - in_use.begin());
        in_use[slot] = true;
        return slot;
    }

    void release(std::size_t slot) {
        std::unique_lock lock(mutex);
        in_use[slot] = false;
        available.notify_one();
    }
};

/**
 *  @brief Threads each `pipeline-role` runs its OpenMP regions with.
 *
 *  Reader and writer are Arrow-bound and mostly single-threaded inside Arrow, so both stay tiny. The other
 *  two each take the machine less a small OS reserve, and their sum deliberately exceeds the core count: on
 *  a pipeline deep enough to hold several batches in flight at most one of them is mid-parallel-for at any
 *  instant, because the fallback bursts on RDKit while the gpu-runner is stream-syncing. Where a workload's
 *  failure rate makes that untrue, `--rdkit-fallback-threads` overrides the split.
 */
struct thread_budget_t {
    int reader;
    int writer;
    int gpu_runner;
    int fallback;
};

static thread_budget_t compute_thread_budget_(int user_fallback_override, int user_gpu_runner_override) {
    int const num_procs = std::max(1, omp_get_num_procs());
    int const reader = std::clamp(num_procs / 128, 1, 2);
    int const writer = std::clamp(num_procs / 64, 2, 4);
    int const gpu_runner = user_gpu_runner_override > 0 ? user_gpu_runner_override : std::max(1, num_procs - 4);
    int const fallback = user_fallback_override > 0 ? user_fallback_override : std::max(1, num_procs - 4);
    return {reader, writer, gpu_runner, fallback};
}

/** Host time one timed span of the pipeline cost, and the molecules that span walked. */
struct role_timing_t {
    std::atomic<std::uint64_t> nanoseconds {0};
    std::atomic<std::uint64_t> molecules {0};
};

/**
 *  @brief Where a run's host time went, one entry per timed span of the outer pipeline.
 *
 *  `fallback` and `descriptors` both sit inside the `fallback-role` and are timed apart because they walk
 *  different populations: `rdkit_fallback` runs on the molecules the GPU refused a slot to, while
 *  `compute_shape_descriptors_` reparses every molecule of the `coprocessed-batch`. Summed, they are what
 *  that role costs; apart, they say which of the two is paying for it.
 */
struct pipeline_timings_t {
    role_timing_t reader;
    role_timing_t gpu_runner;
    role_timing_t fallback;
    role_timing_t descriptors;
    role_timing_t writer;
};

/**
 *  @brief Folds one span of work into @p timing when it leaves scope.
 *
 *  @p molecules is read at scope exit rather than at entry, so a body that only learns its count on the way
 *  out — the reader-role, which has to open the shard first — reports the same way as one that knows it up
 *  front, and must therefore outlive the span.
 */
class scoped_role_span_t {
    role_timing_t &timing_;
    std::uint64_t const &molecules_;
    std::chrono::steady_clock::time_point const start_;

  public:
    scoped_role_span_t(role_timing_t &timing, std::uint64_t const &molecules) noexcept
        : timing_(timing), molecules_(molecules), start_(std::chrono::steady_clock::now()) {}
    ~scoped_role_span_t() noexcept {
        auto const elapsed = std::chrono::steady_clock::now() - start_;
        timing_.nanoseconds.fetch_add(
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
            std::memory_order_relaxed);
        timing_.molecules.fetch_add(molecules_, std::memory_order_relaxed);
    }

    scoped_role_span_t(scoped_role_span_t const &) = delete;
    scoped_role_span_t &operator=(scoped_role_span_t const &) = delete;
};

/** Cumulative counters shared across pipeline stages. */
struct progress_counters_t {
    std::atomic<std::uint64_t> molecules_completed {0};
    std::atomic<std::uint64_t> input_files_completed {0};
    /** Batches the writer-role has finished, which is what says whether any two roles ever overlapped. */
    std::atomic<std::uint64_t> batches_completed {0};
    /** Pending input-files that reached no output shard, on either end of the pipeline; drives the exit code. */
    std::atomic<std::uint64_t> input_files_failed {0};
    /** Molecules that shipped coordinates but no USRCAT, because their shape context could not be rebuilt. */
    std::atomic<std::uint64_t> molecules_undescribed {0};
    /** Molecules a wave refused for its own sizing rather than for anything about them, summed across
     *  batches. One producer is device-side and already raises `cuda_error_count`; the other is a
     *  host-side pinned reservation, which raises nothing at all — so without this the run would report
     *  a whole device's batch as refused and still exit 0. */
    std::atomic<std::uint64_t> molecules_batch_overflowed {0};
    std::uint64_t total_input_files = 0;
    std::chrono::steady_clock::time_point start_time;
};

static void reader_loop_(custd::span<shard_pair_t const *const> pending, cli_config_t const &config,
                         std::uint32_t batch_target, thread_budget_t const &budget, logger_t const &logger,
                         handoff<coprocessed_batch_accumulator_t> &output_queue, progress_counters_t &counters,
                         pipeline_timings_t &timings) {
    omp_set_num_threads(budget.reader);
    coprocessed_batch_accumulator_t accumulator;

    for (auto const *pair : pending) {
        if (stop_requested_.load(std::memory_order_relaxed)) {
            logger.info("stop requested — draining in-flight work and exiting");
            break;
        }
        sfcli::shard_data_t data;
        std::uint64_t molecules_read = 0;
        {
            SF_NVTX_RANGE_COLORED("reader: parquet", 0xC0392B);
            scoped_role_span_t const timed {timings.reader, molecules_read};
            // `read_shard` already named the refusal in the log, so this only says the shard was left behind.
            // It leaves no sentinel either, so `--resume` picks it up again on the next invocation.
            if (sfcli::read_shard(pair->input_path, config.smiles_column, data, &logger) !=
                sfcli::input_read_t::read_k) {
                logger.error("skipping {}", pair->input_path.string());
                counters.input_files_failed.fetch_add(1);
                continue;
            }
            molecules_read = data.smiles.size();
        }
        accumulator.add_input_file(std::move(data), pair);

        if (accumulator.num_molecules() >= batch_target) {
            output_queue.send(std::move(accumulator));
            accumulator.clear();
        }
    }

    if (accumulator.num_molecules() > 0) { output_queue.send(std::move(accumulator)); }

    coprocessed_batch_accumulator_t sentinel;
    sentinel.item = pipeline_item_t::end_of_stream_k;
    output_queue.send(std::move(sentinel));
}

/** How the `gpu-runner-role` left its loop: normally, or with the driver context poisoned. */
enum class gpu_runner_outcome_t {
    completed_k,
    cuda_fault_k,
};

/**
 *  @brief Move every input shard that fed this batch into a sibling `_skip` folder.
 *
 *  A CUDA error poisons the driver context on every device, so parking is what makes the wrapper's next
 *  invocation skip these shards rather than fault on them again. Every shard of the batch is parked, not
 *  just the offending one; on Enamine and PubChem a batch holds one shard anyway.
 */
static void park_batch_shards_(coprocessed_batch_accumulator_t const &accumulator, logger_t const &logger) {
    fs::path const input_dir = accumulator.input_file_pairs.front()->input_path.parent_path();
    fs::path const skip_dir = input_dir.parent_path() / (input_dir.filename().string() + "_skip");
    std::error_code error_code;
    fs::create_directories(skip_dir, error_code);
    for (auto const *pair : accumulator.input_file_pairs) {
        fs::path const destination = skip_dir / pair->input_path.filename();
        fs::rename(pair->input_path, destination, error_code);
        logger.error("CUDA error — parked shard {} -> {} ({})", pair->input_path.string(), destination.string(),
                     error_code ? error_code.message() : "ok");
    }
}

/**
 *  @brief List the molecules the `fallback-role` could still fill a slot for into @p retry_indices,
 *  which is cleared first, and tally the ones it is spared into @p report.
 *
 *  A molecule whose every unfilled slot carries a status `admit_molecule_` re-derives is not
 *  dispatched: it would pay a `SmilesToMol`, an `addHs` and an `MMFFMolProperties` to arrive at the
 *  verdict its slots already hold. Those molecules ship at exactly the status they already carry.
 */
static void fill_retry_indices_into_(custd::span<sfapi::conformer_status_t const> status, std::uint32_t num_conformers,
                                     std::vector<std::uint32_t> &retry_indices, sfcli::fallback_report_t &report) {
    retry_indices.clear();
    report.skipped = {};
    std::uint64_t const num_molecules = status.size() / num_conformers;
    for (std::uint64_t molecule = 0; molecule < num_molecules; ++molecule) {
        auto const slots = status.subspan(molecule * num_conformers, num_conformers);
        if (sfcli::fallback_dispatch_cause(slots) != sfapi::conformer_status_t::success_k) {
            retry_indices.push_back(static_cast<std::uint32_t>(molecule));
            continue;
        }
        // Nothing here the fallback can fill; name the refusal that ships, where there is one.
        for (sfapi::conformer_status_t const slot_status : slots)
            if (slot_status != sfapi::conformer_status_t::success_k) {
                ++report.skipped.of(slot_status);
                break;
            }
    }
}

/** Run one accumulated batch through `generate_conformers` into @p buffer. */
static void generate_batch_(sfapi::context_t &gpu_context, cli_config_t const &config,
                            coprocessed_batch_accumulator_t const &accumulator, output_buffers_t &buffer) {
    sfapi::conformer_slot_layout_t const layout {config.max_atoms, config.num_conformers};
    std::uint64_t const num_molecules = accumulator.num_molecules();
    std::size_t const total_conformers = static_cast<std::size_t>(num_molecules) * config.num_conformers;

    buffer.resize_for(num_molecules, layout);
    std::fill_n(buffer.energies.data(), total_conformers, 0.0f);
    std::fill_n(buffer.status.data(), total_conformers, sfapi::conformer_status_t::mmff_unconverged_k);

    sfapi::config_t generation_config;
    generation_config.num_conformers = config.num_conformers;
    generation_config.random_seed = config.random_seed;
    generation_config.polish_telemetry = config.polish_telemetry;

    // The aggregate status is discarded because a batch-level `mmff_unconverged_k` may still carry a
    // majority of successful conformers; the per-conformer codes in `buffer.status` drive every stage below.
    [[maybe_unused]] sfapi::conformer_status_t const batch_status = gpu_context.generate_conformers(
        accumulator.combined_smiles, generation_config, config.max_atoms,
        custd::span<__half>(buffer.coordinates.data(), total_conformers * layout.slot_stride()),
        custd::span<float>(buffer.energies.data(), total_conformers),
        custd::span<sfapi::conformer_status_t>(buffer.status.data(), total_conformers),
        custd::span<std::uint8_t>(buffer.num_atoms_per_molecule.data(), num_molecules),
        custd::span<std::uint16_t>(buffer.num_heavy_atoms_per_molecule.data(), num_molecules),
        custd::span<std::uint16_t>(buffer.num_bonds_per_molecule.data(), num_molecules),
        custd::span<float>(buffer.molecular_weight_per_molecule.data(), num_molecules));
}

static gpu_runner_outcome_t gpu_runner_loop_(sfapi::context_t &gpu_context, cli_config_t const &config,
                                             thread_budget_t const &budget, logger_t const &logger,
                                             handoff<coprocessed_batch_accumulator_t> &input_queue,
                                             handoff<coprocessed_batch_result_t> &output_queue, buffer_pool_t &pool,
                                             pipeline_timings_t &timings) {
    omp_set_num_threads(budget.gpu_runner);
    gpu_runner_outcome_t outcome = gpu_runner_outcome_t::completed_k;

    for (;;) {
        auto accumulator = input_queue.receive();
        if (accumulator.item == pipeline_item_t::end_of_stream_k) break;
        // Once the driver context is poisoned nothing further is launched, but the loop keeps draining so
        // the reader-role never blocks on a full ring and the whole pipeline can shut down in order.
        if (outcome == gpu_runner_outcome_t::cuda_fault_k) continue;
        SF_NVTX_RANGE_COLORED("gpu_runner: preprocess+launch", 0x2E86C1);
        std::uint64_t const batch_molecules = accumulator.num_molecules();
        scoped_role_span_t const timed {timings.gpu_runner, batch_molecules};

        auto const batch_start = std::chrono::steady_clock::now();
        std::size_t const buffer_slot = pool.acquire();
        output_buffers_t &buffer = pool.at(buffer_slot);
        generate_batch_(gpu_context, config, accumulator, buffer);

        // Park this batch's shards and stop feeding the GPU, so `main` returns non-zero and the wrapper
        // relaunches with a fresh CUDA context.
        if (gpu_context.cuda_error_count.load() > 0) {
            park_batch_shards_(accumulator, logger);
            logger.error("aborting invocation after {} CUDA errors; wrapper will relaunch",
                         fmt::group_digits(gpu_context.cuda_error_count.load()));
            stop_requested_.store(true, std::memory_order_relaxed);
            pool.release(buffer_slot);
            outcome = gpu_runner_outcome_t::cuda_fault_k;
            continue;
        }

        std::size_t const total_conformers = static_cast<std::size_t>(batch_molecules) * config.num_conformers;
        sfcli::fallback_report_t report;
        fill_retry_indices_into_({buffer.status.data(), total_conformers}, config.num_conformers, buffer.retry_indices,
                                 report);

        output_queue.send({.accumulator = std::move(accumulator),
                           .buffer_slot = buffer_slot,
                           .item = pipeline_item_t::batch_k,
                           .batch_start = batch_start,
                           .fallback_report = report});
    }

    coprocessed_batch_result_t sentinel;
    sentinel.item = pipeline_item_t::end_of_stream_k;
    output_queue.send(std::move(sentinel));
    return outcome;
}

/**
 *  @brief Compute USRCAT descriptors and separate disconnected fragments over a batch's coordinates.
 *
 *  Runs after the RDKit fallback, so every slot is final. A molecule whose shape context cannot be rebuilt is
 *  marked `undescribed_k`, which is what makes the writer null its `usrcat` rows instead of publishing the
 *  zero-filled block they hold.
 *
 *  @return Molecules that had coordinates to describe and were left undescribed anyway.
 */
static std::uint64_t compute_shape_descriptors_(cli_config_t const &config, thread_budget_t const &budget,
                                                coprocessed_batch_accumulator_t const &accumulator,
                                                output_buffers_t &buffer) {
    std::uint64_t const num_molecules = accumulator.num_molecules();
    std::uint32_t const num_conformers = config.num_conformers;
    sfapi::conformer_slot_layout_t const slot_layout {config.max_atoms, num_conformers};
    std::size_t const total_conformers = static_cast<std::size_t>(num_molecules) * num_conformers;
    std::fill_n(buffer.usrcat.data(), total_conformers * sfcli::usrcat_components_k, 0.0);
    std::fill_n(buffer.shape_described.data(), num_molecules, sfcli::shape_described_t::undescribed_k);
    int const threads = budget.fallback > 0 ? budget.fallback : omp_get_max_threads();
    std::uint64_t undescribed = 0;

    // `private` hands each thread its own context, which keeps the capacity of its five vectors across the
    // molecules that thread walks; declaring it inside the body would allocate and free once per molecule.
    sfpost::molecule_shape_context_t context;

#pragma omp parallel for schedule(dynamic, 256) num_threads(threads) private(context) \
    reduction(+ : undescribed) if (threads > 1)
    for (std::uint64_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        std::uint8_t const stored_atoms = buffer.num_atoms_per_molecule[molecule_index];
        if (stored_atoms == 0) continue; // preprocess failed: no coordinates to describe
        // Every refusal means the reparse could not reproduce the atom order the coordinates were written
        // in, so the molecule stays undescribed and ships no shape fingerprint at all.
        if (sfpost::build_shape_context_into(accumulator.combined_smiles[molecule_index], stored_atoms,
                                             smallfield::keep_largest_fragment_k,
                                             context) != sfpost::shape_typing_t::typed_k) {
            ++undescribed;
            continue;
        }
        buffer.shape_described[molecule_index] = sfcli::shape_described_t::described_k;
        std::uint32_t const num_atoms = stored_atoms;

        // Left uninitialized: `gather_slot_into_interleaved` writes all `3 * num_atoms` components below, and
        // nothing past them is ever read.
        std::array<float, sfapi::max_atoms_per_molecule_k * 3u> interleaved;
        for (std::uint32_t conformer = 0; conformer < num_conformers; ++conformer) {
            std::size_t const flat = slot_layout.flat_slot(molecule_index, conformer);
            if (buffer.status[flat] != sfapi::conformer_status_t::success_k) continue;
            std::size_t const slot_base = slot_layout.slot_offset(molecule_index, conformer);
            sfapi::gather_slot_into_interleaved(buffer.coordinates.data() + slot_base, slot_layout, num_atoms,
                                                interleaved.data());

            custd::span<float> coords_view {interleaved.data(), static_cast<std::size_t>(num_atoms) * 3u};
            if (context.num_components > 1) {
                sfpost::separate_fragments(coords_view, context);
                sfapi::scatter_interleaved_into_slot(interleaved.data(), slot_layout, num_atoms,
                                                     buffer.coordinates.data() + slot_base);
            }

            sfpost::usrcat_descriptor(
                coords_view, context,
                {buffer.usrcat.data() + flat * sfcli::usrcat_components_k, sfcli::usrcat_components_k});
        }
    }
    return undescribed;
}

static void fallback_loop_(cli_config_t const &config, thread_budget_t const &budget, buffer_pool_t &pool,
                           handoff<coprocessed_batch_result_t> &input_queue,
                           handoff<coprocessed_batch_result_t> &output_queue, pipeline_timings_t &timings) {
    omp_set_num_threads(budget.fallback);
    sfapi::conformer_slot_layout_t const layout {config.max_atoms, config.num_conformers};

    for (;;) {
        auto result = input_queue.receive();
        if (result.item == pipeline_item_t::end_of_stream_k) {
            output_queue.send(std::move(result));
            break;
        }

        output_buffers_t &buffer = pool.at(result.buffer_slot);
        auto const &accumulator = result.accumulator;
        std::uint64_t const num_molecules = accumulator.num_molecules();
        std::size_t const total_conformers = static_cast<std::size_t>(num_molecules) * config.num_conformers;

        if (!buffer.retry_indices.empty()) {
            // Both the range and the span name the recovery alone. The descriptor pass below is the other
            // half of this role and walks a population two orders of magnitude wider.
            SF_NVTX_RANGE_COLORED("fallback: rdkit recovery", 0xE67E22);
            std::uint64_t const retry_molecules = buffer.retry_indices.size();
            scoped_role_span_t const timed {timings.fallback, retry_molecules};
            sfcli::rdkit_fallback_inputs_t const fallback_inputs {
                .smiles = accumulator.combined_smiles,
                .retry_indices = buffer.retry_indices,
                .layout = layout,
                .num_threads = budget.fallback,
                .random_seed = config.random_seed,
            };
            sfcli::rdkit_fallback_outputs_t fallback_outputs {
                .status = {buffer.status.data(), total_conformers},
                .num_atoms_per_molecule = {buffer.num_atoms_per_molecule.data(), num_molecules},
                .coordinates = {buffer.coordinates.data(), total_conformers * layout.slot_stride()},
                .energies = {buffer.energies.data(), total_conformers},
            };
            sfcli::rdkit_fallback(fallback_inputs, fallback_outputs, result.fallback_report);
        }

        {
            SF_NVTX_RANGE_COLORED("postprocess: descriptors+fragments", 0x8E44AD);
            scoped_role_span_t const timed {timings.descriptors, num_molecules};
            result.molecules_undescribed = compute_shape_descriptors_(config, budget, accumulator, buffer);
        }

        output_queue.send(std::move(result));
    }
}

/** The slice of a batch's buffers one input file owns, ready for `write_shard`. */
static sfcli::writer_inputs_t shard_writer_inputs_(cli_config_t const &config,
                                                   coprocessed_batch_accumulator_t const &accumulator,
                                                   output_buffers_t const &buffer, std::size_t file_index) {
    sfapi::conformer_slot_layout_t const layout {config.max_atoms, config.num_conformers};
    std::uint64_t const first_molecule = accumulator.first_molecule_of(file_index);
    std::uint64_t const num_molecules = accumulator.molecules_of(file_index);
    std::size_t const first_slot = static_cast<std::size_t>(first_molecule) * config.num_conformers;
    std::size_t const num_slots = static_cast<std::size_t>(num_molecules) * config.num_conformers;

    return {
        .provenance = {.input_shard_stem = accumulator.input_file_pairs[file_index]->input_path.stem().string(),
                       .dataset = config.dataset,
                       // Each input file is read whole, so its molecules are its own rows counted from zero.
                       .base_row_index = 0},
        .molecules = {.smiles = {accumulator.combined_smiles.data() + first_molecule, num_molecules},
                      .num_atoms = {buffer.num_atoms_per_molecule.data() + first_molecule, num_molecules},
                      .num_conformers = {buffer.num_conformers_per_molecule.data() + first_molecule, num_molecules},
                      .num_heavy_atoms = {buffer.num_heavy_atoms_per_molecule.data() + first_molecule, num_molecules},
                      .num_bonds = {buffer.num_bonds_per_molecule.data() + first_molecule, num_molecules},
                      .molecular_weight = {buffer.molecular_weight_per_molecule.data() + first_molecule, num_molecules},
                      .shape_described = {buffer.shape_described.data() + first_molecule, num_molecules}},
        .conformers = {.layout = layout,
                       .status = {buffer.status.data() + first_slot, num_slots},
                       .energies = {buffer.energies.data() + first_slot, num_slots},
                       .coordinates = {buffer.coordinates.data() + first_slot * layout.slot_stride(),
                                       num_slots * layout.slot_stride()},
                       .usrcat = {buffer.usrcat.data() + first_slot * sfcli::usrcat_components_k,
                                  num_slots * sfcli::usrcat_components_k}},
    };
}

/**
 *  @brief Emit one output shard per input file the batch consumed, sealing each with its `.encoded` sentinel.
 *
 *  `parquet::arrow::WriteTable` carries no threading of its own, so the role's budget is spent across the
 *  batch's files rather than inside one: each file owns a disjoint slice of the batch's buffers, reads them
 *  and writes its own path, and the counters it raises are atomic.
 */
static void write_batch_shards_(cli_config_t const &config, thread_budget_t const &budget,
                                coprocessed_batch_accumulator_t const &accumulator, output_buffers_t &buffer,
                                logger_t const &logger, progress_counters_t &counters) {
    // No deduplication: every surviving molecule emits all K slots, and a failed one emits zero, which the
    // writer turns into a single all-null row so every input row keeps at least one output row.
    for (std::uint64_t molecule_index = 0; molecule_index < accumulator.num_molecules(); ++molecule_index)
        buffer.num_conformers_per_molecule[molecule_index] = buffer.num_atoms_per_molecule[molecule_index] != 0
                                                                 ? static_cast<std::uint8_t>(config.num_conformers)
                                                                 : 0u;

    std::size_t const num_files = accumulator.input_file_pairs.size();
    int const workers = static_cast<int>(
        std::min<std::size_t>(num_files, static_cast<std::size_t>(std::max(budget.writer, 1))));

#pragma omp parallel for schedule(dynamic, 1) num_threads(workers) if (workers > 1)
    for (std::size_t file_index = 0; file_index < num_files; ++file_index) {
        auto const &output_path = accumulator.input_file_pairs[file_index]->output_path;
        // `write_shard` already logged the column or the file that stopped it, so nothing is said here.
        sfcli::shard_write_report_t const report = sfcli::write_shard(
            output_path, shard_writer_inputs_(config, accumulator, buffer, file_index), &logger);
        // A refused shard publishes nothing, so it counts as neither molecules nor a completed input-file, and
        // its sentinel is left unwritten so `--resume` retries it on the next invocation.
        if (report.status != sfcli::shard_write_t::written_k) {
            counters.input_files_failed.fetch_add(1);
            continue;
        }
        // `write_shard` closed the shard before it returned, so the sentinel this iteration writes can only
        // seal a file that is already durable — the one ordering the corpus's completeness signal depends on.
        sfcli::write_encoded_sentinel(output_path, &logger);

        counters.molecules_completed.fetch_add(accumulator.molecules_of(file_index));
        counters.input_files_completed.fetch_add(1);
    }
}

/** Share of @p whole that @p part is, as a percentage; zero when nothing was counted. */
static double percent_of_(std::uint64_t part, std::uint64_t whole) noexcept {
    return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

using sfapi::conformer_status_name;

/**
 *  @brief One line per status that sent molecules to the `fallback-role`, or held them back from it.
 *
 *  Recovery read against the dispatching status is what decides whether that status should stay
 *  `fallback_disposition_t::retry_k`: one whose molecules RDKit never fills is one the dispatch pays
 *  a full parse for and gets nothing from. Statuses nothing counted are left unprinted.
 */
static void log_fallback_breakdown_(coprocessed_batch_result_t const &result, logger_t const &logger) {
    for (std::size_t index = 0; index < sfcli::conformer_status_count_k; ++index) {
        auto const status = static_cast<sfapi::conformer_status_t>(index);
        std::uint64_t const attempted = result.fallback_report.attempted.named[index];
        std::uint64_t const skipped = result.fallback_report.skipped.named[index];
        if (attempted > 0)
            logger.info("  fallback cause={} attempted={} recovered={} {:.3f}%", conformer_status_name(status),
                        fmt::group_digits(attempted), fmt::group_digits(result.fallback_report.recovered.named[index]),
                        percent_of_(result.fallback_report.recovered.named[index], attempted));
        if (skipped > 0)
            logger.info("  fallback cause={} terminal={}", conformer_status_name(status), fmt::group_digits(skipped));
    }
    // A status byte this binary has no enumerator for, which is what a shard from a newer producer carries.
    std::uint64_t const unnamed_attempted = result.fallback_report.attempted.unnamed;
    std::uint64_t const unnamed_skipped = result.fallback_report.skipped.unnamed;
    if (unnamed_attempted > 0 || unnamed_skipped > 0)
        logger.info("  fallback cause={} attempted={} recovered={} {:.3f}% terminal={}", sfcli::unnamed_status_k,
                    fmt::group_digits(unnamed_attempted), fmt::group_digits(result.fallback_report.recovered.unnamed),
                    percent_of_(result.fallback_report.recovered.unnamed, unnamed_attempted),
                    fmt::group_digits(unnamed_skipped));
}

/**
 *  @brief What the dispatched molecules cost, by stage and by how long the whole attempt took.
 *
 *  Stage time is summed over the whole OpenMP team, so it exceeds the role's wall time by roughly the width
 *  of that team. The duration bins are what a per-molecule mean cannot show: a single molecule in the top
 *  bin outweighs a thousand in the bottom one, and which of the two a batch is made of decides whether the
 *  fallback is worth making faster or worth making smaller.
 */
static void log_fallback_cost_(coprocessed_batch_result_t const &result, logger_t const &logger) {
    auto const &report = result.fallback_report;
    std::uint64_t summed_nanoseconds = 0;
    for (std::uint64_t const nanoseconds : report.nanoseconds) summed_nanoseconds += nanoseconds;
    if (summed_nanoseconds == 0) return;

    for (std::size_t stage = 0; stage < sfcli::fallback_stage_count_k; ++stage)
        logger.info("  fallback stage={} seconds={:.1f} {:.3f}%", sfcli::fallback_stage_names_k[stage],
                    static_cast<double>(report.nanoseconds[stage]) * 1e-9,
                    percent_of_(report.nanoseconds[stage], summed_nanoseconds));

    for (std::size_t bin = 0; bin < sfcli::fallback_duration_bin_count_k; ++bin)
        if (report.molecules_by_duration[bin] > 0)
            logger.info("  fallback duration_ms>={} molecules={}",
                        fmt::group_digits(sfcli::fallback_duration_bin_floor_milliseconds(bin)),
                        fmt::group_digits(report.molecules_by_duration[bin]));

    logger.info("  fallback slowest seconds={:.1f} smiles={}", static_cast<double>(report.slowest_nanoseconds) * 1e-9,
                result.accumulator.combined_smiles[report.slowest_molecule_index]);
}

/** One progress line per batch: its own throughput, the cumulative rate, and the projected finish. */
static void log_batch_progress_(coprocessed_batch_result_t const &result, logger_t const &logger,
                                progress_counters_t const &counters) {
    auto const now = std::chrono::steady_clock::now();
    double const batch_seconds = std::chrono::duration<double>(now - result.batch_start).count();
    double const wall = std::chrono::duration<double>(now - counters.start_time).count();
    std::uint64_t const molecules_done = counters.molecules_completed.load();
    std::uint64_t const files_done = counters.input_files_completed.load();
    std::uint64_t const files_remaining = counters.total_input_files - files_done;
    double const remaining_seconds = files_remaining > 0 && files_done > 0
                                         ? static_cast<double>(files_remaining) * wall / static_cast<double>(files_done)
                                         : 0.0;
    auto const finish_wall = std::chrono::system_clock::now() +
                             std::chrono::milliseconds(static_cast<std::int64_t>(remaining_seconds * 1000.0));

    std::uint64_t const attempted = result.fallback_report.attempted.total();
    std::uint64_t const recovered = result.fallback_report.recovered.total();
    std::uint64_t const skipped = result.fallback_report.skipped.total();
    std::string fallback_fragment;
    if (attempted > 0 || skipped > 0)
        fallback_fragment = fmt::format("  fallback attempted={} recovered={} {:.3f}% terminal={}",
                                        fmt::group_digits(attempted), fmt::group_digits(recovered),
                                        percent_of_(recovered, attempted), fmt::group_digits(skipped));
    // Molecules whose coordinates ship without a shape fingerprint, which is a hole in a similarity index
    // rather than a missing row, so it is named on the same line as the batch that produced it.
    std::string undescribed_fragment;
    if (result.molecules_undescribed > 0)
        undescribed_fragment = fmt::format(
            "  undescribed={} {:.3f}%", fmt::group_digits(result.molecules_undescribed),
            percent_of_(result.molecules_undescribed, result.accumulator.num_molecules()));

    double const batch_rate = static_cast<double>(result.accumulator.num_molecules()) / std::max(batch_seconds, 0.001);
    double const cumulative_rate = static_cast<double>(molecules_done) / std::max(wall, 0.001);
    logger.info("{:>4}/{:<4} done {:.0f}s  {} mol/s{}{}  cumulative {}  avg {} mol/s  ETA {}",
                fmt::group_digits(files_done), fmt::group_digits(counters.total_input_files), batch_seconds,
                fmt::group_digits(static_cast<std::uint64_t>(batch_rate)), fallback_fragment, undescribed_fragment,
                fmt::group_digits(molecules_done), fmt::group_digits(static_cast<std::uint64_t>(cumulative_rate)),
                smallfield::format_wall_time(finish_wall));
}

static void writer_loop_(cli_config_t const &config, thread_budget_t const &budget, logger_t const &logger,
                         handoff<coprocessed_batch_result_t> &input_queue, buffer_pool_t &pool,
                         progress_counters_t &counters, pipeline_timings_t &timings) {
    omp_set_num_threads(budget.writer);

    for (;;) {
        auto result = input_queue.receive();
        if (result.item == pipeline_item_t::end_of_stream_k) break;
        SF_NVTX_RANGE_COLORED("writer: parquet", 0x27AE60);
        std::uint64_t const batch_molecules = result.accumulator.num_molecules();
        scoped_role_span_t const timed {timings.writer, batch_molecules};

        write_batch_shards_(config, budget, result.accumulator, pool.at(result.buffer_slot), logger, counters);
        pool.release(result.buffer_slot);
        counters.batches_completed.fetch_add(1);
        counters.molecules_undescribed.fetch_add(result.molecules_undescribed);
        counters.molecules_batch_overflowed.fetch_add(
            result.fallback_report.skipped.of(sfapi::conformer_status_t::batch_overflow_k));
        log_batch_progress_(result, logger, counters);
        log_fallback_breakdown_(result, logger);
        log_fallback_cost_(result, logger);
    }
}

/**
 *  @brief Pick out the shards that still need encoding and announce the split.
 *
 *  @param[out] pending Points into @p shard_pairs, which must outlive it.
 */
static void fill_pending_shards_into_(std::vector<shard_pair_t> const &shard_pairs, logger_t const &logger,
                                      std::vector<shard_pair_t const *> &pending) {
    pending.clear();
    for (auto const &pair : shard_pairs)
        if (!pair.already_complete) pending.push_back(&pair);
    logger.info("{}: {} input-files, {} pending, {} complete", executable_name_k, fmt::group_digits(shard_pairs.size()),
                fmt::group_digits(pending.size()), fmt::group_digits(shard_pairs.size() - pending.size()));
}

/**
 *  @brief Run the four `pipeline-roles` concurrently until every pending shard has been written.
 *
 *  The depth is how many coprocessed-batches the upstream may stage before back-pressuring, and the pool is
 *  one deeper so no stage blocks on a free output buffer. It hides the H2D/D2H tails on multi-GPU runs and
 *  costs about 70 GB of pinned host memory per extra slot at a 700 K-molecule batch target.
 */
static gpu_runner_outcome_t run_pipeline_(sfapi::context_t &gpu_context, cli_config_t const &config,
                                          custd::span<shard_pair_t const *const> pending, std::uint32_t batch_target,
                                          thread_budget_t const &budget, logger_t const &logger,
                                          progress_counters_t &counters, pipeline_timings_t &timings) {
    constexpr std::size_t pipeline_depth_k = 5;
    buffer_pool_t pool {pipeline_depth_k + 1};
    handoff<coprocessed_batch_accumulator_t> reader_to_gpu {pipeline_depth_k};
    handoff<coprocessed_batch_result_t> gpu_to_fallback {pipeline_depth_k};
    handoff<coprocessed_batch_result_t> fallback_to_writer {pipeline_depth_k};
    auto outcome = gpu_runner_outcome_t::completed_k;

    omp_set_max_active_levels(2);
#pragma omp parallel sections num_threads(4)
    {
#pragma omp section
        { reader_loop_(pending, config, batch_target, budget, logger, reader_to_gpu, counters, timings); }
#pragma omp section
        {
            outcome = gpu_runner_loop_(gpu_context, config, budget, logger, reader_to_gpu, gpu_to_fallback, pool,
                                       timings);
        }
#pragma omp section
        { fallback_loop_(config, budget, pool, gpu_to_fallback, fallback_to_writer, timings); }
#pragma omp section
        { writer_loop_(config, budget, logger, fallback_to_writer, pool, counters, timings); }
    }
    return outcome;
}

/** Host seconds a span accumulated over the run. */
static double seconds_of_(role_timing_t const &timing) noexcept {
    return static_cast<double>(timing.nanoseconds.load()) * 1e-9;
}

/** One timed span of the pipeline, paired with the name the report prints it under. */
struct named_span_t {
    std::string_view name;
    role_timing_t const &timing;
};

/**
 *  @brief Where the run's host time went, one line per timed span, printed once at the end.
 *
 *  A span opens when its role already holds work, so blocking on an empty queue belongs to no span and the
 *  summed span time is what the run spent doing something. Shares are of that sum rather than of the wall
 *  clock, because the spans overlap — their ratio is the overlap factor on the last line.
 *
 *  One `coprocessed-batch` has nothing to overlap with, so a single-batch run decomposes four roles that ran
 *  strictly serially. That is a cold decomposition and not a steady-state share, which is why the batch count
 *  leads the report.
 */
static void log_pipeline_timings_(pipeline_timings_t const &timings, progress_counters_t const &counters,
                                  double wall_seconds, logger_t const &logger) {
    std::array<named_span_t, 5> const spans {{
        {"reader-role", timings.reader},
        {"gpu-runner-role", timings.gpu_runner},
        {"rdkit-fallback", timings.fallback},
        {"shape-descriptors", timings.descriptors},
        {"writer-role", timings.writer},
    }};

    double summed_seconds = 0.0;
    for (named_span_t const &span : spans) summed_seconds += seconds_of_(span.timing);

    std::uint64_t const batches = counters.batches_completed.load();
    logger.info("role decomposition  coprocessed-batches={} input-files={}  {}", fmt::group_digits(batches),
                fmt::group_digits(counters.input_files_completed.load()),
                batches < 2 ? "cold: the roles ran strictly serially" : "steady state: the roles overlap");
    for (named_span_t const &span : spans) {
        double const seconds = seconds_of_(span.timing);
        std::uint64_t const molecules = span.timing.molecules.load();
        double const share = summed_seconds > 0.0 ? 100.0 * seconds / summed_seconds : 0.0;
        double const per_molecule_us = molecules > 0 ? 1e6 * seconds / static_cast<double>(molecules) : 0.0;
        logger.info("  span={:<18} wall={:>8.1f}s share={:>7.3f}% per_molecule={:>7.1f}us molecules={}", span.name,
                    seconds, share, per_molecule_us, fmt::group_digits(molecules));
    }
    // Naming the two middle spans as one role on the closing line, so their sum is read as the fallback-role's
    // cost and neither is read as all of it.
    logger.info("  spans summed {:.1f}s over {:.1f}s wall, overlap {:.2f}x; " //
                "rdkit-fallback and shape-descriptors are both the fallback-role",
                summed_seconds, wall_seconds, wall_seconds > 0.0 ? summed_seconds / wall_seconds : 0.0);
}

} // namespace smallfield::cli

int main(int argc, char **argv) {
    sfcli::cli_config_t config;
    switch (sfcli::parse_cli_(argc, argv, config)) {
    case sfcli::cli_parse_t::parsed_k: break;
    case sfcli::cli_parse_t::help_requested_k: return 0;
    case sfcli::cli_parse_t::invalid_k: return 2;
    }

    std::signal(SIGINT, sfcli::handle_stop_signal_);
    std::signal(SIGTERM, sfcli::handle_stop_signal_);

    logger_t logger;
    logger.verbose_level = config.verbose_level;

    auto const shard_pairs = sfcli::inventory_shards(config.input_dir, config.output_dir, config.restart_policy);
    if (shard_pairs.empty()) {
        logger.error("no input-files found in {}", config.input_dir.string());
        return 1;
    }
    std::vector<sfcli::shard_pair_t const *> pending;
    sfcli::fill_pending_shards_into_(shard_pairs, logger, pending);
    if (pending.empty()) {
        logger.info("nothing to do");
        return 0;
    }

    sfapi::context_t gpu_context(config.gpu_ids);
    gpu_context.logger.verbose_level = config.verbose_level;
    gpu_context.logger.start_time = logger.start_time;
    std::uint32_t const num_gpus = static_cast<std::uint32_t>(gpu_context.device_ids.size());
    std::uint32_t const batch_target = config.batch_size > 0 ? config.batch_size : num_gpus * 100'000u;

    sfcli::thread_budget_t const budget = sfcli::compute_thread_budget_(config.rdkit_fallback_threads,
                                                                        config.gpu_runner_threads);
    logger.info("  {} GPUs, coprocessed-batch target {} mols, threads reader={} gpu_runner={} fallback={} writer={}",
                num_gpus, fmt::group_digits(batch_target), budget.reader, budget.gpu_runner, budget.fallback,
                budget.writer);

    sfcli::progress_counters_t counters;
    counters.total_input_files = pending.size();
    counters.start_time = logger.start_time;
    sfcli::pipeline_timings_t timings;
    sfcli::gpu_runner_outcome_t const outcome = sfcli::run_pipeline_(gpu_context, config, pending, batch_target, budget,
                                                                     logger, counters, timings);

    double const total = std::chrono::duration<double>(std::chrono::steady_clock::now() - counters.start_time).count();
    std::uint64_t const done = counters.molecules_completed.load();
    std::uint64_t const undescribed = counters.molecules_undescribed.load();
    logger.info("complete  {} mols  {} mol/s  undescribed {} {:.3f}%", fmt::group_digits(done),
                fmt::group_digits(static_cast<std::uint64_t>(static_cast<double>(done) / std::max(total, 0.001))),
                fmt::group_digits(undescribed), sfcli::percent_of_(undescribed, done));
    sfcli::log_pipeline_timings_(timings, counters, total, logger);
    std::uint64_t const files_failed = counters.input_files_failed.load();
    if (files_failed > 0)
        logger.error("{} of {} input-files produced no output shard — none of them was sealed, so a rerun retries them",
                     fmt::group_digits(files_failed), fmt::group_digits(counters.total_input_files));

    if (outcome == sfcli::gpu_runner_outcome_t::cuda_fault_k) return 1;
    std::uint64_t const cuda_errors = gpu_context.cuda_error_count.load();
    if (cuda_errors > 0) {
        logger.error("{} CUDA errors reported by the library — output is unreliable", fmt::group_digits(cuda_errors));
        return 1;
    }
    // A wave that could not size itself refuses molecules the corpus never had a problem with, and the
    // host-side producer of that refusal raises no CUDA error — so this is the only thing between a
    // silently short shard and a rerun at a wave that fits.
    std::uint64_t const batch_overflowed = counters.molecules_batch_overflowed.load();
    if (batch_overflowed > 0) {
        logger.error("{} molecules refused for wave sizing, not chemistry; rerun at a smaller --batch-size",
                     fmt::group_digits(batch_overflowed));
        return 1;
    }
    return files_failed > 0 ? 1 : 0;
}
