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

/** Cumulative counters shared across pipeline stages. */
struct progress_counters_t {
    std::atomic<std::uint64_t> molecules_completed {0};
    std::atomic<std::uint64_t> input_files_completed {0};
    std::uint64_t total_input_files = 0;
    std::chrono::steady_clock::time_point start_time;
};

static void reader_loop_(custd::span<shard_pair_t const *const> pending, cli_config_t const &config,
                         std::uint32_t batch_target, thread_budget_t const &budget, logger_t const &logger,
                         handoff<coprocessed_batch_accumulator_t> &output_queue) {
    omp_set_num_threads(budget.reader);
    coprocessed_batch_accumulator_t accumulator;

    for (auto const *pair : pending) {
        if (stop_requested_.load(std::memory_order_relaxed)) {
            logger.info("stop requested — draining in-flight work and exiting");
            break;
        }
        sfcli::shard_data_t data;
        // `read_shard` already named the refusal in the log, so this only says the shard was left behind.
        if (sfcli::read_shard(pair->input_path, config.smiles_column, data, &logger) != sfcli::input_read_t::read_k) {
            logger.error("skipping {}", pair->input_path.string());
            continue;
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
                ++report.skipped[static_cast<std::size_t>(slot_status)];
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
                                             handoff<coprocessed_batch_result_t> &output_queue, buffer_pool_t &pool) {
    omp_set_num_threads(budget.gpu_runner);
    gpu_runner_outcome_t outcome = gpu_runner_outcome_t::completed_k;

    for (;;) {
        auto accumulator = input_queue.receive();
        if (accumulator.item == pipeline_item_t::end_of_stream_k) break;
        // Once the driver context is poisoned nothing further is launched, but the loop keeps draining so
        // the reader-role never blocks on a full ring and the whole pipeline can shut down in order.
        if (outcome == gpu_runner_outcome_t::cuda_fault_k) continue;
        SF_NVTX_RANGE_COLORED("gpu_runner: preprocess+launch", 0x2E86C1);

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

        std::size_t const total_conformers = static_cast<std::size_t>(accumulator.num_molecules()) *
                                             config.num_conformers;
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
 *  Runs after the RDKit fallback, so every slot is final. A non-success or preprocess-failed slot keeps
 *  the zero-filled descriptor, which the writer emits as a null.
 */
static void compute_shape_descriptors_(cli_config_t const &config, thread_budget_t const &budget,
                                       coprocessed_batch_accumulator_t const &accumulator, output_buffers_t &buffer) {
    std::uint64_t const num_molecules = accumulator.num_molecules();
    std::uint32_t const num_conformers = config.num_conformers;
    sfapi::conformer_slot_layout_t const slot_layout {config.max_atoms, num_conformers};
    std::size_t const total_conformers = static_cast<std::size_t>(num_molecules) * num_conformers;
    std::fill_n(buffer.usrcat.data(), total_conformers * sfcli::usrcat_components_k, 0.0);
    int const threads = budget.fallback > 0 ? budget.fallback : omp_get_max_threads();

#pragma omp parallel for schedule(dynamic, 256) num_threads(threads) if (threads > 1)
    for (std::uint64_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        std::uint8_t const stored_atoms = buffer.num_atoms_per_molecule[molecule_index];
        if (stored_atoms == 0) continue; // preprocess failed: no coordinates to describe
        sfpost::molecule_shape_context_t const context = sfpost::build_shape_context(
            accumulator.combined_smiles[molecule_index]);
        // The reparse must reproduce the atom order the coordinates were written in, and a mismatched
        // count is exactly where it did not.
        if (!context.ok || context.num_atoms != stored_atoms) continue;
        std::uint32_t const num_atoms = stored_atoms;

        std::array<float, sfapi::max_atoms_per_molecule_k * 3u> interleaved {};
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

            std::array<double, sfcli::usrcat_components_k> usrcat {};
            sfpost::usrcat_descriptor(coords_view, context, usrcat);
            std::copy(usrcat.begin(), usrcat.end(), buffer.usrcat.data() + flat * sfcli::usrcat_components_k);
        }
    }
}

static void fallback_loop_(cli_config_t const &config, thread_budget_t const &budget, buffer_pool_t &pool,
                           handoff<coprocessed_batch_result_t> &input_queue,
                           handoff<coprocessed_batch_result_t> &output_queue) {
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
        SF_NVTX_RANGE_COLORED("fallback: rdkit recovery", 0xE67E22);

        if (!buffer.retry_indices.empty()) {
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
            compute_shape_descriptors_(config, budget, accumulator, buffer);
        }

        output_queue.send(std::move(result));
    }
}

/**
 *  @brief Touch the 3-byte `<output>.encoded` sentinel a `--resume` run keys on.
 *
 *  Removed again if the write was short, because a truncated sentinel would make the next run skip an
 *  unencoded shard as though it were finished.
 */
static void write_encoded_sentinel_(fs::path const &output_path, logger_t const &logger) {
    fs::path sentinel = output_path;
    sentinel += ".encoded";
    FILE *sentinel_file = std::fopen(sentinel.string().c_str(), "wb");
    if (!sentinel_file) {
        logger.error("sentinel write failed: {}", sentinel.string());
        return;
    }
    std::size_t const written = std::fwrite("v1\n", 1, 3, sentinel_file);
    std::fclose(sentinel_file);
    if (written == 3) return;
    std::error_code remove_error;
    fs::remove(sentinel, remove_error);
    logger.error("sentinel write truncated, removed: {}", sentinel.string());
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
                      .molecular_weight = {buffer.molecular_weight_per_molecule.data() + first_molecule,
                                           num_molecules}},
        .conformers = {.layout = layout,
                       .status = {buffer.status.data() + first_slot, num_slots},
                       .energies = {buffer.energies.data() + first_slot, num_slots},
                       .coordinates = {buffer.coordinates.data() + first_slot * layout.slot_stride(),
                                       num_slots * layout.slot_stride()},
                       .usrcat = {buffer.usrcat.data() + first_slot * sfcli::usrcat_components_k,
                                  num_slots * sfcli::usrcat_components_k}},
    };
}

/** Emit one output shard per input file the batch consumed, sealing each with its `.encoded` sentinel. */
static void write_batch_shards_(cli_config_t const &config, coprocessed_batch_accumulator_t const &accumulator,
                                output_buffers_t &buffer, logger_t const &logger, progress_counters_t &counters) {
    // No deduplication: every surviving molecule emits all K slots, and a failed one emits zero, which the
    // writer turns into a single all-null row so every input row keeps at least one output row.
    for (std::uint64_t molecule_index = 0; molecule_index < accumulator.num_molecules(); ++molecule_index)
        buffer.num_conformers_per_molecule[molecule_index] = buffer.num_atoms_per_molecule[molecule_index] != 0
                                                                 ? static_cast<std::uint8_t>(config.num_conformers)
                                                                 : 0u;

    for (std::size_t file_index = 0; file_index < accumulator.input_file_pairs.size(); ++file_index) {
        auto const &output_path = accumulator.input_file_pairs[file_index]->output_path;
        // `write_shard` already logged the column or the file that stopped it, so nothing is said here.
        sfcli::shard_write_report_t const report = sfcli::write_shard(
            output_path, shard_writer_inputs_(config, accumulator, buffer, file_index), &logger);
        if (report.status == sfcli::shard_write_t::written_k) write_encoded_sentinel_(output_path, logger);

        counters.molecules_completed.fetch_add(accumulator.molecules_of(file_index));
        counters.input_files_completed.fetch_add(1);
    }
}

/** Share of @p whole that @p part is, as a percentage; zero when nothing was counted. */
static double percent_of_(std::uint64_t part, std::uint64_t whole) noexcept {
    return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

/** The name a per-status report prints, which is the enumerator without its suffix. */
static std::string_view conformer_status_name_(sfapi::conformer_status_t status) noexcept {
    switch (status) {
    case sfapi::conformer_status_t::success_k: return "success";
    case sfapi::conformer_status_t::stereo_failure_k: return "stereo_failure";
    case sfapi::conformer_status_t::mmff_unconverged_k: return "mmff_unconverged";
    case sfapi::conformer_status_t::invalid_smiles_k: return "invalid_smiles";
    case sfapi::conformer_status_t::too_many_atoms_k: return "too_many_atoms";
    case sfapi::conformer_status_t::preprocess_timeout_k: return "preprocess_timeout";
    case sfapi::conformer_status_t::unsupported_element_k: return "unsupported_element";
    case sfapi::conformer_status_t::too_many_bonds_k: return "too_many_bonds";
    case sfapi::conformer_status_t::too_many_rings_k: return "too_many_rings";
    case sfapi::conformer_status_t::bounds_infeasible_k: return "bounds_infeasible";
    }
    return "unknown";
}

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
        std::uint64_t const attempted = result.fallback_report.attempted[index];
        std::uint64_t const skipped = result.fallback_report.skipped[index];
        if (attempted > 0)
            logger.info("  fallback cause={} attempted={} recovered={} {:.3f}%", conformer_status_name_(status),
                        fmt::group_digits(attempted), fmt::group_digits(result.fallback_report.recovered[index]),
                        percent_of_(result.fallback_report.recovered[index], attempted));
        if (skipped > 0)
            logger.info("  fallback cause={} terminal={}", conformer_status_name_(status), fmt::group_digits(skipped));
    }
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

    std::uint64_t const attempted = sfcli::total_of(result.fallback_report.attempted);
    std::uint64_t const recovered = sfcli::total_of(result.fallback_report.recovered);
    std::uint64_t const skipped = sfcli::total_of(result.fallback_report.skipped);
    std::string fallback_fragment;
    if (attempted > 0 || skipped > 0)
        fallback_fragment = fmt::format("  fallback attempted={} recovered={} {:.3f}% terminal={}",
                                        fmt::group_digits(attempted), fmt::group_digits(recovered),
                                        percent_of_(recovered, attempted), fmt::group_digits(skipped));

    double const batch_rate = static_cast<double>(result.accumulator.num_molecules()) / std::max(batch_seconds, 0.001);
    double const cumulative_rate = static_cast<double>(molecules_done) / std::max(wall, 0.001);
    logger.info("{:>4}/{:<4} done {:.0f}s  {} mol/s{}  cumulative {}  avg {} mol/s  ETA {}",
                fmt::group_digits(files_done), fmt::group_digits(counters.total_input_files), batch_seconds,
                fmt::group_digits(static_cast<std::uint64_t>(batch_rate)), fallback_fragment,
                fmt::group_digits(molecules_done), fmt::group_digits(static_cast<std::uint64_t>(cumulative_rate)),
                smallfield::format_wall_time(finish_wall));
}

static void writer_loop_(cli_config_t const &config, thread_budget_t const &budget, logger_t const &logger,
                         handoff<coprocessed_batch_result_t> &input_queue, buffer_pool_t &pool,
                         progress_counters_t &counters) {
    omp_set_num_threads(budget.writer);

    for (;;) {
        auto result = input_queue.receive();
        if (result.item == pipeline_item_t::end_of_stream_k) break;
        SF_NVTX_RANGE_COLORED("writer: parquet", 0x27AE60);

        write_batch_shards_(config, result.accumulator, pool.at(result.buffer_slot), logger, counters);
        pool.release(result.buffer_slot);
        log_batch_progress_(result, logger, counters);
        log_fallback_breakdown_(result, logger);
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
                                          progress_counters_t &counters) {
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
        { reader_loop_(pending, config, batch_target, budget, logger, reader_to_gpu); }
#pragma omp section
        { outcome = gpu_runner_loop_(gpu_context, config, budget, logger, reader_to_gpu, gpu_to_fallback, pool); }
#pragma omp section
        { fallback_loop_(config, budget, pool, gpu_to_fallback, fallback_to_writer); }
#pragma omp section
        { writer_loop_(config, budget, logger, fallback_to_writer, pool, counters); }
    }
    return outcome;
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
    sfcli::gpu_runner_outcome_t const outcome = sfcli::run_pipeline_(gpu_context, config, pending, batch_target, budget,
                                                                     logger, counters);

    double const total = std::chrono::duration<double>(std::chrono::steady_clock::now() - counters.start_time).count();
    std::uint64_t const done = counters.molecules_completed.load();
    logger.info("complete  {} mols  {} mol/s", fmt::group_digits(done),
                fmt::group_digits(static_cast<std::uint64_t>(static_cast<double>(done) / std::max(total, 0.001))));
    if (outcome == sfcli::gpu_runner_outcome_t::cuda_fault_k) return 1;
    std::uint64_t const cuda_errors = gpu_context.cuda_error_count.load();
    if (cuda_errors > 0) {
        logger.error("{} CUDA errors reported by the library — output is unreliable", fmt::group_digits(cuda_errors));
        return 1;
    }
    return 0;
}
