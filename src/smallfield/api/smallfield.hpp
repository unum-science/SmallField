/**
 *  @file src/smallfield/api/smallfield.hpp
 *  @brief Public C++ API for SmallField conformer generation.
 *
 *  Single header for C++ users. Provides batch GPU conformer generation with automatic multi-GPU dispatch,
 *  wave-based scheduling, and half-precision coordinate output.
 *
 *  @code{.cpp}
 *  smallfield::context_t context;  // all GPUs
 *  smallfield::config_t config;
 *  config.num_conformers = 10;
 *
 *  std::vector<std::string_view> smiles = {"CCO", "c1ccccc1", ...};
 *  std::vector<__half> coordinates(smiles.size() * config.num_conformers * 128 * 3);
 *  std::vector<float> energies(smiles.size() * config.num_conformers);
 *  std::vector<smallfield::conformer_status_t> status(smiles.size() * config.num_conformers);
 *
 *  context.generate_conformers(smiles, config, 128, coordinates, energies, status);
 *  @endcode
 */

#pragma once

#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <array>       // `std::array`
#include <atomic>      // `std::atomic`
#include <memory>      // `std::unique_ptr`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <cuda_fp16.h>    // `__half`
#include <cuda_runtime.h> // `cudaStream_t`

#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/log.hpp"
#include "smallfield/preprocess/wave_slice_pool.hpp"
#include "smallfield/types/atom_limited_bins.hpp"
#include "smallfield/types/conformer_status.hpp"
#include "smallfield/types/storage.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::api {

/** The vocabulary every rejection speaks. Defined in `types/` because `build_descriptor` answers with
 *  one and the warp kernels write one; re-exported here because it is also the status a caller of
 *  `generate_conformers` reads, and the name that caller reaches for is `smallfield::api::`. */
using types::conformer_status_count_k;
using types::conformer_status_name;
using types::conformer_status_t;

using preprocess::wave_slice_pool_t;
using types::device_indices_t;

/**
 *  @brief Whether the MMFF94 polish reports what it did with each conformer slot.
 *
 *  The kernel writes the same six floats per slot either way; what this selects is whether a region is
 *  carved for them, staged back, and reduced into the per-wave tally the pipeline logs. Off by default
 *  because the download and the reduce are per-slot work no production output reads.
 */
enum class polish_telemetry_t : std::uint8_t {
    /** No region is carved, so the kernel is handed no array and writes none. */
    disabled_k,
    /** One row per conformer slot is staged and folded into one log line per wave. */
    summarized_k,
};

/**
 *  @brief Pipeline configuration.
 */
struct config_t {
    std::uint32_t num_conformers = 3;
    std::uint32_t max_stereo_attempts = 10;
    /** Target `embedding-wave` size for pipelined dispatch; 0 = VRAM-aware auto. */
    std::uint32_t wave_size = 0;
    std::uint64_t random_seed = 42;
    polish_telemetry_t polish_telemetry = polish_telemetry_t::disabled_k;
};

/** Upper bound on devices a single `context_t` can manage. */
inline constexpr std::size_t max_devices_k = 16;

/** Atom-count bins for `atom-limited-bin` kernel dispatch; one definition, shared with the
 *  metric prepass so the two cannot partition the same wave differently. */
using types::atom_limited_bin_caps_k;
using types::atom_limited_bin_count_k;

/*  `max_atoms_per_molecule_k` lives in `conformer_layout.hpp`, included above. It matches
 *  `molecule_descriptor_t::max_atoms_per_molecule_k`, and larger molecules are rejected by
 *  preprocess with `conformer_status_t::too_many_atoms_k`. */

/** Worst-case conformer count used for pre-allocation at construction. */
inline constexpr std::uint32_t max_conformer_count_k = 16;

/**
 *  @brief Per-device launch-configuration cache populated once at `context_t`
 *  construction, so the per-wave hot path never calls
 *  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` from inside the OMP-parallel
 *  launch region.
 */
struct device_launch_config_t {
    int streaming_multiprocessor_count = 0;
    std::array<int, atom_limited_bin_count_k> blocks_per_multiprocessor_etkdg {};
    std::array<int, atom_limited_bin_count_k> blocks_per_multiprocessor_mmff {};
};

/**
 *  @brief Per-GPU persistent scratch buffers for the metric-embedding prepass.
 *
 *  Allocated once per `context_t` so the prepass can run fully stream-async
 *  without per-wave `cudaMalloc` / `cudaFree` pairs that otherwise serialize
 *  the launch phase. Sized for the worst case: all molecules in a wave having
 *  the maximum supported atom count.
 */
struct metric_scratch_t {
    /** Owning grow-only device list of which molecule goes into which bin, allocated and released along
     *  this device's stream. Its `capacity()` is the largest group a wave may bin, so the extent cannot
     *  drift from the allocation it describes. */
    device_indices_t group_indices;
    std::uint32_t max_atoms = 0;
    /** Host-side per-bin molecule lists, refilled each wave; grows once. */
    std::array<std::vector<std::uint32_t>, atom_limited_bin_count_k> bin_groups {};
    /*  `embed_4d_power_iter_kernel` holds the Gram matrix, means, eigenvectors and reduction scratch
     *  in shared memory for its whole lifetime, so the only persistent device scratch is
     *  `group_indices`. */
};

/**
 *  @brief Forward-declares persistent per-device launch scratch.
 *
 *  Defined in `smallfield.cu` because it holds this device's copy of the MMFF94 parameter tables and
 *  the compiled CSD torsion patterns, whose headers would otherwise reach every consumer of the
 *  public API. CUDA RAII wrappers are not the reason — `metric_scratch_t::group_indices` is one, right
 *  above.
 */
struct launch_scratch_t;

/**
 *  @brief Waves in flight at once, and with it the slot count of every per-wave ring.
 *
 *  Two, and that is a property of the code rather than a tuning knob. Four buffers are one per device
 *  and re-carved from offset zero on every launch — `launch_scratch_t::device_pool`, `::host_staging`
 *  and `::mmff_staging`, plus this context's `metric_scratch` — so a second resident wave would write
 *  its regions over the first wave's. Raising the depth means one set per slot, and the wave is sized
 *  to 75 % of free VRAM, so a second set halves the wave rather than deepening the pipeline.
 *
 *  Depth 2 is also already all the dispatch loop can use: it finalizes wave N−1 before launching wave
 *  N, so a third slot would never be live. Preparation is not the constraint — that state is
 *  caller-owned on `preprocess::batch_prep_scratch_t` and the entry is reentrant.
 *
 *  Named for the pinned prep ring rather than for "the pipeline", because `cli/smallfield_batch.cpp`
 *  carries a depth of its own for the four-role handoff and the two are unrelated numbers.
 */
inline constexpr std::uint32_t pinned_prep_depth_k = 2;

/** The ring slot wave @p wave_index owns. The one place the ring modulus is written. */
[[nodiscard]] constexpr std::uint32_t pinned_prep_slot_of(std::uint32_t wave_index) noexcept {
    return wave_index % pinned_prep_depth_k;
}

/**
 *  @brief GPU context owning the per-device CUDA streams, scratch and pinned pools a wave runs on.
 *
 *  Reuse across multiple `generate_conformers` calls to avoid re-initialization.
 */
struct context_t {
    std::vector<int> device_ids;
    /** One CUDA stream per device. All per-wave work on a given
     *  device (H2D, kernels, D2H) runs stream-ordered on its own stream. */
    std::vector<cudaStream_t> streams;
    /** One per device. */
    std::vector<metric_scratch_t> metric_scratch;
    /** One per device. */
    std::vector<std::unique_ptr<launch_scratch_t>> launch_scratch;
    /** One per device. */
    std::vector<device_launch_config_t> launch_configs;
    /**
     *  @brief Persistent ring of per-device pinned batch pools, one slot per wave in flight.
     *
     *  Slot-major because that is the shape `prepare_per_device_batches_into` fills: it takes one
     *  contiguous run of pools and writes every molecule's bytes into the pool of its target device.
     *  Each GPU's host-to-device copy then reads only from its own pool, so no two devices pull from
     *  one pinned source. Reach a slot through `wave_slot`, never a hand-written modulus.
     */
    std::array<std::array<wave_slice_pool_t, max_devices_k>, pinned_prep_depth_k> batch_slots_per_device {};
    /** Per-device molecule count every pool above was pre-sized for. The wave
     *  sizer clamps to this so a runtime wave can never exceed the construction-time
     *  reservation and trigger a grow from inside the per-device launch fan-out. */
    std::uint32_t per_device_reserved_molecules = 0;

    /**
     *  @brief Owned per-context logger. Tune verbosity via `context.logger.verbose_level = 1;`
     *  before calling `generate_conformers`. Library diagnostics — wave timing table, per-device breakdown —
     *  route through `logger.trace` / `logger.trace2`; CUDA errors through `logger.error`.
     */
    smallfield::logger_t logger;

    /**
     *  @brief Per-context CUDA error accumulator, incremented by `cuda_error_sink_t::check` every time a
     *  CUDA API call returns a non-success status. Production callers read this after each
     *  `generate_conformers` call and exit non-zero if it is positive, catching silent failures that would
     *  otherwise hide behind per-wave zero-filled outputs. Reset via `context.cuda_error_count.store(0)`.
     */
    std::atomic<std::uint64_t> cuda_error_count {0};

    /**
     *  @brief Construct for the given GPUs.
     *
     *  @param[in] devices GPU device IDs. Empty span = all available GPUs.
     */
    explicit context_t(custd::span<int const> devices = {});
    ~context_t();

    context_t(context_t const &) = delete;
    context_t &operator=(context_t const &) = delete;

    /**
     *  @brief The per-device pinned pools wave @p wave_index prepares into, one per device in order.
     *
     *  Extent is the live device count rather than `max_devices_k`, so preprocess cannot write past
     *  the devices this context actually manages.
     */
    [[nodiscard]] custd::span<wave_slice_pool_t> wave_slot(std::uint32_t wave_index) noexcept {
        return {batch_slots_per_device[pinned_prep_slot_of(wave_index)].data(), device_ids.size()};
    }

    /**
     *  @brief Generate 3D conformers for a batch of molecules.
     *
     *  Only a molecule's largest fragment is embedded — a counterion has no meaningful conformer,
     *  and MMFF94 cannot type an isolated one — so every readback below describes that parent
     *  fragment rather than the input string.
     *
     *  @param[in] smiles Input SMILES strings (one per molecule); salt forms are accepted.
     *  @param[in] config Pipeline configuration.
     *  @param[in] max_atoms_per_molecule Maximum atom count (with H) across the batch.
     *      Molecules exceeding this are skipped.
     *  @param[out] coordinates_out Output half-precision coordinates. Size: molecules * conformers * max_atoms * 3.
     *      Structure-of-arrays per slot -- address it through `conformer_slot_layout_t`,
     *      never by a contiguous `3 * num_atoms` read from the slot base.
     *  @param[out] energies_out Output MMFF94 energies (kcal/mol). Size: molecules * conformers.
     *  @param[out] status_out Output per-conformer status. Size: molecules * conformers.
     *  @param[out] num_atoms_out Optional. If non-empty, filled with each `molecule`'s
     *      atom count (heavy + explicit H) post-preprocess. Size: molecules (one per
     *      input SMILES). 0 for molecules that failed preprocess (invalid_smiles /
     *      too_many_atoms / preprocess_timeout). Lets downstream writers emit tight
     *      per-row coordinate lists instead of padded fixed-size-list blocks.
     *  @param[out] num_heavy_atoms_out Optional. If non-empty, filled with each
     *      `molecule`'s heavy (non-H) atom count post-preprocess. Size: molecules.
     *      0 for molecules that failed preprocess.
     *  @param[out] num_bonds_out Optional. If non-empty, filled with each `molecule`'s
     *      bond count (including H bonds, AddHs-applied) post-preprocess. Size: molecules.
     *      0 for molecules that failed preprocess.
     *  @param[out] molecular_weight_out Optional. If non-empty, filled with each
     *      `molecule`'s exact molecular weight in Daltons. Size: molecules.
     *      0.0 for molecules that failed preprocess.
     *  @return Batch-level status (success if at least one conformer succeeded).
     */
    [[nodiscard]] conformer_status_t generate_conformers(
        custd::span<std::string_view const> smiles, config_t const &config, std::uint32_t max_atoms_per_molecule,
        custd::span<__half> coordinates_out, custd::span<float> energies_out,
        custd::span<conformer_status_t> status_out, custd::span<std::uint8_t> num_atoms_out = {},
        custd::span<std::uint16_t> num_heavy_atoms_out = {}, custd::span<std::uint16_t> num_bonds_out = {},
        custd::span<float> molecular_weight_out = {});
};

} // namespace smallfield::api
