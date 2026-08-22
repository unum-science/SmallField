/**
 *  @file src/smallfield/api/smallfield.hpp
 *  @brief Public C++ API for SmallField conformer generation: a single header giving batch GPU conformer
 *  generation with automatic multi-GPU dispatch, wave-based scheduling and half-precision coordinates.
 *
 *  @code{.cpp}
 *  smallfield::context_t context;  // all GPUs
 *  smallfield::config_t config;
 *  config.num_conformers = 10;
 *  std::vector<std::string_view> smiles = {"CCO", "c1ccccc1", ...};
 *  std::vector<__half> coordinates(smiles.size() * config.num_conformers * 128 * 3);
 *  std::vector<float> energies(smiles.size() * config.num_conformers);
 *  std::vector<smallfield::conformer_status_t> status(smiles.size() * config.num_conformers);
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

/** Re-exported so a caller of `generate_conformers` spells it `smallfield::api::conformer_status_t`. */
using types::conformer_status_count_k;
using types::conformer_status_name;
using types::conformer_status_t;

using preprocess::wave_slice_pool_t;
using types::device_indices_t;
using types::device_stream_t;

/** Whether the MMFF94 polish reports what it did with each conformer slot. The kernel writes the same six
 *  floats per slot either way; this selects whether a region is carved for them, staged back and reduced
 *  into the per-wave tally the pipeline logs. Off by default, because that download and reduce are
 *  per-slot work no production output reads. */
enum class polish_telemetry_t : std::uint8_t {
    /** No region is carved, so the kernel is handed no array and writes none. */
    disabled_k,
    /** One row per conformer slot is staged and folded into one log line per wave. */
    summarized_k,
};

/** Pipeline configuration. `wave_size` is the target `embedding-wave` size for pipelined dispatch, and
 *  zero asks for a VRAM-aware automatic one. */
struct config_t {
    std::uint32_t num_conformers = 3;
    std::uint32_t max_stereo_attempts = 10;
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

/** Worst-case conformer count used for pre-allocation at construction. */
inline constexpr std::uint32_t max_conformer_count_k = 16;

/** Per-device launch-configuration cache filled once at `context_t` construction, so the per-wave hot path
 *  never calls `cudaOccupancyMaxActiveBlocksPerMultiprocessor` inside the OMP-parallel launch region. */
struct device_launch_config_t {
    int streaming_multiprocessor_count = 0;
    std::array<int, atom_limited_bin_count_k> blocks_per_multiprocessor_etkdg {};
    std::array<int, atom_limited_bin_count_k> blocks_per_multiprocessor_mmff {};
};

/**
 *  @brief Per-GPU persistent scratch for the metric-embedding prepass, allocated once per `context_t`.
 *
 *  The prepass then runs fully stream-async, with no per-wave `cudaMalloc` / `cudaFree` pair to serialize
 *  the launch phase, and is sized for the worst case of every molecule in a wave at the maximum supported
 *  atom count. `embed_4d_power_iter_kernel` keeps its Gram matrix, means, eigenvectors and reduction
 *  scratch in shared memory for its whole lifetime, so `group_indices` is the only device buffer here.
 */
struct metric_scratch_t {
    /** Grow-only device list of which molecule goes into which bin, allocated and released on this
     *  device's stream; its `capacity()` is the largest group a wave may bin, so the extent cannot
     *  drift from the allocation it describes. */
    device_indices_t group_indices;
    /** Largest bin this scratch was prepared for. No allocation depends on it; it is the ceiling
     *  `build_metric_initial_coordinates_on_gpu` refuses a wider bin against. */
    std::uint32_t max_atoms = 0;
    /** Host-side per-bin molecule lists, refilled each wave; grows once. */
    std::array<std::vector<std::uint32_t>, atom_limited_bin_count_k> bin_groups {};
};

/** Persistent per-device launch scratch, defined in `smallfield.cu` because it holds this device's copy of
 *  the MMFF94 parameter tables and the compiled CSD torsion patterns, whose headers would otherwise reach
 *  every consumer of the public API. */
struct launch_scratch_t;

/**
 *  @brief Waves in flight at once, and with it the slot count of every per-wave ring.
 *
 *  Not a tuning knob. `launch_scratch_t::device_pool`, `::host_staging`, `::mmff_staging` and this
 *  context's `metric_scratch` are one set per device re-carved from offset zero on every launch, so a
 *  second resident wave would write over the first's regions; one set per slot, at 75 % of free VRAM per
 *  wave, halves the wave rather than deepening the pipeline. The dispatch loop could not use a third slot
 *  anyway, finalizing wave N−1 before launching wave N. Named for the pinned prep ring rather than the
 *  pipeline, because `cli/generate.cpp` carries an unrelated depth for the four-role handoff.
 */
inline constexpr std::uint32_t pinned_prep_depth_k = 2;

/** The ring slot wave @p wave_index owns. The one place the ring modulus is written. */
[[nodiscard]] constexpr std::uint32_t pinned_prep_slot_of(std::uint32_t wave_index) noexcept {
    return wave_index % pinned_prep_depth_k;
}

/** GPU context owning the per-device CUDA streams, scratch and pinned pools a wave runs on; reuse it
 *  across `generate_conformers` calls to avoid re-initialization. */
struct context_t {
    /** Owned per-context logger; set `context.logger.verbose_level = 1;` before calling
     *  `generate_conformers`. Diagnostics route through `logger.trace` / `logger.trace2`, CUDA errors
     *  through `logger.error`. Declared first, with the error counter, so reverse-order destruction
     *  leaves both alive for every member below that might report a failed release through them. */
    smallfield::logger_t logger;

    /** Per-context CUDA error accumulator, incremented by `cuda_error_sink_t::check` on every non-success
     *  CUDA status and reset through `context.cuda_error_count.store(0)`. Production callers read it after
     *  each `generate_conformers` and exit non-zero if it is positive, catching failures that otherwise
     *  hide behind zero-filled outputs. */
    std::atomic<std::uint64_t> cuda_error_count {0};

    /** The devices that came up, in the order every per-device vector below is indexed by. One physical
     *  GPU may appear more than once, which is how K entries run K streams on it as K virtual devices. */
    std::vector<int> device_ids;
    /** One owned CUDA stream per device; all per-wave work on a device — H2D, kernels, D2H — runs
     *  stream-ordered on its own. Declared ahead of every member allocated against these streams, so
     *  destruction in reverse declaration order releases those buffers first and no `cudaFreeAsync`
     *  ever reaches a stream that is already gone. */
    std::vector<device_stream_t> streams;
    /** One per device. */
    std::vector<metric_scratch_t> metric_scratch;
    /** One per device. */
    std::vector<std::unique_ptr<launch_scratch_t>> launch_scratch;
    /** One per device. */
    std::vector<device_launch_config_t> launch_configs;
    /** Persistent ring of per-device pinned batch pools, one slot per wave in flight. Slot-major because
     *  that is the shape `prepare_per_device_batches_into` fills: one contiguous run of pools, every
     *  molecule's bytes into its target device's pool, so no two devices pull from one pinned source.
     *  Reach a slot through `wave_slot`, never a hand-written modulus. */
    std::array<std::array<wave_slice_pool_t, max_devices_k>, pinned_prep_depth_k> batch_slots_per_device {};
    /** Per-device molecule count every pool above was pre-sized for; the wave sizer clamps to it so no
     *  runtime wave outgrows the construction-time reservation and fires a grow inside the launch
     *  fan-out. */
    std::uint32_t per_device_reserved_molecules = 0;

    /** Construct for the given GPUs; @p devices holds device IDs, and an empty span means all of them. */
    explicit context_t(custd::span<int const> devices = {});
    /** Defaulted, and defined where `launch_scratch_t` is complete rather than here. Teardown is the
     *  members' own, in reverse declaration order, so no statement can put a free after its stream. */
    ~context_t();

    context_t(context_t const &) = delete;
    context_t &operator=(context_t const &) = delete;

    /** The per-device pinned pools wave @p wave_index prepares into, one per device in order. The extent
     *  is the live device count rather than `max_devices_k`, so preprocess cannot write past the devices
     *  this context actually manages. */
    [[nodiscard]] custd::span<wave_slice_pool_t> wave_slot(std::uint32_t wave_index) noexcept {
        return {batch_slots_per_device[pinned_prep_slot_of(wave_index)].data(), device_ids.size()};
    }

    /**
     *  @brief Generate 3D conformers for a batch of molecules.
     *
     *  Only a molecule's largest fragment is embedded — a counterion has no meaningful conformer, and MMFF94
     *  cannot type an isolated one — so every readback below describes that parent fragment, not the input
     *  string. Sizes are fixed by the batch: `energies_out` and `status_out` hold `molecules * conformers`
     *  entries, `coordinates_out` a further `max_atoms_per_molecule * 3` halves per slot, and each of the
     *  four optional readbacks one entry per input SMILES — left empty to skip it, zero where preprocess
     *  refused the molecule.
     *  @param[in] smiles One per molecule; salt forms are accepted.
     *  @param[in] max_atoms_per_molecule Atom count including hydrogens above which a molecule is skipped,
     *      and the stride `coordinates_out` is addressed at.
     *  @param[out] coordinates_out Half-precision, structure-of-arrays per slot: address it through
     *      `conformer_slot_layout_t`, never by a contiguous `3 * num_atoms` read from the slot base.
     *  @param[out] energies_out MMFF94 energies in kcal/mol.
     *  @param[out] num_atoms_out Atom count including explicit hydrogens, which lets a writer emit tight
     *      per-row coordinate lists instead of padded fixed-size-list blocks.
     *  @param[out] num_heavy_atoms_out Non-hydrogen atom count.
     *  @param[out] num_bonds_out Bond count, hydrogen bonds included.
     *  @param[out] molecular_weight_out Exact molecular weight in Daltons.
     *  @return Batch-level status; `success_k` when at least one conformer succeeded.
     */
    [[nodiscard]] conformer_status_t generate_conformers(
        custd::span<std::string_view const> smiles, config_t const &config, std::uint32_t max_atoms_per_molecule,
        custd::span<__half> coordinates_out, custd::span<float> energies_out,
        custd::span<conformer_status_t> status_out, custd::span<std::uint8_t> num_atoms_out = {},
        custd::span<std::uint16_t> num_heavy_atoms_out = {}, custd::span<std::uint16_t> num_bonds_out = {},
        custd::span<float> molecular_weight_out = {});
};

} // namespace smallfield::api
