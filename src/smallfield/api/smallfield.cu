/**
 *  @file src/smallfield/api/smallfield.cu
 *  @brief Library implementation of the SmallField public API.
 *
 *  The one translation unit the header-only library funnels through. Holds `context_t`'s per-device
 *  CUDA resources, the VRAM-aware wave sizer, and the three steps every `embedding-wave` takes:
 *  prepare on the CPU, launch across every device, finalize back into the caller's arrays. The
 *  Python module is a separate translation unit, `python/smallfield/_binding.cpp`.
 */

#include <cstring> // `std::memcpy`

#include <algorithm>     // `std::max`, `std::min`, `std::sort`
#include <array>         // `std::array`
#include <chrono>        // `std::chrono::high_resolution_clock`, `std::chrono::duration`
#include <numeric>       // `std::iota`
#include <string_view>   // `std::string_view`
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

#include "smallfield/api/smallfield.hpp"
#include "smallfield/api/wave_capture.hpp"
#include "smallfield/cuda/bond_distance_kernel.cuh"
#include "smallfield/cuda/csd_torsion_kernel.cuh"
#include "smallfield/cuda/etkdg_pipeline.cuh"
#include "smallfield/cuda/metric_embedding_prepass.cuh"
#include "smallfield/cuda/preprocess_kernel.cuh"
#include "smallfield/cuda/stereo_validation.cuh"
#include "smallfield/cuda_error.hpp"
#include "smallfield/mmff94/mmff94_extract_device.cuh"
#include "smallfield/mmff94/mmff94_typer_device.cuh"
#include "smallfield/nvtx.hpp"
#include "smallfield/parameters/mmff94_tables_device.hpp"
#include "smallfield/parameters/mmff94_tables_rdkit.hpp"
#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/utilities.hpp"

namespace smallfield::api {
using smallfield::cuda::build_csd_torsion_table;
using smallfield::cuda::build_metric_initial_coordinates_on_gpu;
using smallfield::cuda::compute_mmff_polish_shared_memory_bytes;
using smallfield::cuda::compute_pipeline_shared_memory_bytes;
using smallfield::cuda::convert_coords_fp32_to_fp16_kernel;
using smallfield::cuda::csd_torsion_match_kernel;
using smallfield::cuda::csd_torsion_table_device_t;
using smallfield::cuda::csd_torsion_table_host_t;
using smallfield::cuda::etk_build_csr_kernel;
using smallfield::cuda::etkdg_pipeline_warp_kernel;
using smallfield::cuda::launch_bond_distance_bfs;
using smallfield::cuda::mmff94_apply_typing_patch_kernel;
using smallfield::cuda::mmff_polish_warp_kernel;
using smallfield::cuda::stamp_refused_molecules_kernel;
using smallfield::cuda::upload_csd_torsion_table_to_device;
using smallfield::cuda::validate_polished_stereo_kernel;
using smallfield::cuda::warp_size_k;
using smallfield::mmff94::mmff94_build_csr_kernel;
using smallfield::mmff94::mmff94_compact_terms_kernel;
using smallfield::mmff94::mmff94_extract_terms_kernel;
using smallfield::mmff94::mmff94_per_mol_counts_t;
using smallfield::mmff94::mmff94_type_and_charge_kernel;
using smallfield::parameters::extract_mmff94_tables_from_rdkit;
using smallfield::parameters::mmff94_tables_device_t;
using smallfield::parameters::mmff94_tables_host_t;
using smallfield::parameters::upload_mmff94_tables_to_device;
using smallfield::preprocess::batch_prep_scratch_t;
using smallfield::preprocess::carve_wave_slice;
using smallfield::preprocess::prepare_per_device_batches_into;
using smallfield::preprocess::rejected_mol_t;
using smallfield::preprocess::wave_slice_bytes;
using smallfield::preprocess::wave_slice_const_view_t;
using smallfield::preprocess::wave_slice_view_t;
using smallfield::types::atom_limited_bin_caps_k;
using smallfield::types::atom_limited_bin_count_k;
using smallfield::types::carve;
using smallfield::types::device_allocator_t;
using smallfield::types::device_bytes_t;
using smallfield::types::download_from;
using smallfield::types::energy_is_physical;
using smallfield::types::etk_torsion_descriptor_t;
using smallfield::types::etkdg_molecule_metadata_t;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::max_mmff_angles_per_mol_k;
using smallfield::types::max_mmff_bonds_per_mol_k;
using smallfield::types::max_mmff_interactions_per_mol_k;
using smallfield::types::max_mmff_oops_per_mol_k;
using smallfield::types::max_mmff_stretchbends_per_mol_k;
using smallfield::types::max_mmff_torsions_per_mol_k;
using smallfield::types::mmff94_angle_t;
using smallfield::types::mmff94_atom_interaction_t;
using smallfield::types::mmff94_bond_t;
using smallfield::types::mmff94_oop_t;
using smallfield::types::mmff94_stretchbend_t;
using smallfield::types::mmff94_torsion_t;
using smallfield::types::molecule_descriptor_t;
using smallfield::types::pinned_bytes_t;
using smallfield::types::pool_alignment_k;
using smallfield::types::pool_cursor;
using smallfield::types::reserve_outcome_t;
using smallfield::types::upload_into;

using smallfield::cuda_error_sink_t;

#pragma region Wave Sizing

/**
 *  @brief Rough upper bound on the per-molecule batch-pool footprint in a wave.
 *
 *  The wave sizer divides a VRAM budget by this, so it covers both sides of the pool: the prefix the
 *  host uploads, and the device-only regions the preprocessing kernels fill. At the
 *  `max_atoms_per_molecule_k` cap the bounds matrix dominates; the per-atom slack covers the
 *  constraint list, whose length scales with the bond and neighbour-pair counts rather than with the
 *  atom count alone.
 *
 *  @param[in] max_atoms Atom-count cap the wave is sized against.
 *  @return Bytes to reserve per molecule across the device pool and the uploaded prefix together.
 */
static inline std::size_t estimate_per_molecule_batch_bytes(std::uint32_t max_atoms) noexcept {
    std::size_t const atom_count = max_atoms;
    std::size_t const bounds_bytes = atom_count * atom_count * sizeof(float);
    std::size_t const descriptor_bytes = sizeof(molecule_descriptor_t);
    std::size_t const per_atom_slack = atom_count * 128u;
    std::size_t const per_mol_fixed = 1024u;
    std::size_t const bond_distance_bytes = atom_count * atom_count * sizeof(std::uint8_t);
    // A molecule with explicit hydrogens carries roughly (atom count + 16) bonds; the term
    // reservations scale with that bond count and with the neighbour-pair count around each atom.
    std::size_t const bonds = atom_count + 16u;
    std::size_t const torsion_bytes = bonds * sizeof(etk_torsion_descriptor_t);
    std::size_t const mmff_term_bytes = 4u * bonds *
                                        (sizeof(mmff94_bond_t) + sizeof(mmff94_angle_t) + sizeof(mmff94_torsion_t) +
                                         sizeof(mmff94_oop_t) + sizeof(mmff94_stretchbend_t));
    std::size_t const csr_bytes = 32u * bonds * sizeof(mmff94_atom_interaction_t);
    return bounds_bytes + descriptor_bytes + per_atom_slack + per_mol_fixed + bond_distance_bytes + torsion_bytes +
           mmff_term_bytes + csr_bytes;
}

/**
 *  @brief Bytes one virtual slot may spend on a wave, given the tightest per-slot VRAM census.
 *
 *  Leaves 25 % of the free pool untouched for peak-allocation headroom. When @p free_probe_count
 *  is zero — every `cudaMemGetInfo` on every device failed — degrades to half of the smallest
 *  per-slot *total* VRAM instead of to nothing, which would drop the sizer onto its hardcoded
 *  default. Both arms yield zero only when nothing at all is known about the hardware.
 *
 *  Deliberately not `static`: the probe that feeds it cannot be steered from a test, so
 *  `test/api/test_pipeline.cpp` drives this arithmetic directly.
 */
std::size_t wave_budget_bytes(std::size_t free_probe_count, std::size_t smallest_free_per_slot,
                              std::size_t smallest_total_per_slot) noexcept {
    if (free_probe_count > 0) return smallest_free_per_slot * 3u / 4u;
    return smallest_total_per_slot / 2u;
}

/**
 *  @brief Ceiling of @p numerator divided by @p denominator, zero when there is nothing to divide by.
 *
 *  Split out of `reserve_molecules_per_device` so the zero-denominator edge is one named,
 *  test-visible answer rather than an unsigned wrap that a second wrap happens to undo.
 */
std::uint32_t divide_rounding_up(std::uint32_t numerator, std::uint32_t denominator) noexcept {
    if (denominator == 0) return 0;
    return (numerator + denominator - 1u) / denominator;
}

/**
 *  @brief Molecules to pre-reserve on each device for a @p wave_size wave spread over
 *  @p device_count devices.
 *
 *  Adds one slack molecule so a wave whose remainder lands unevenly never leaves a device one
 *  slot short. With no devices there is nothing to spread and the reservation is the floor of one.
 *
 *  Deliberately not `static`: see `wave_budget_bytes`.
 */
std::uint32_t reserve_molecules_per_device(std::uint32_t wave_size, std::uint32_t device_count) noexcept {
    if (device_count == 0) return 1u;
    return divide_rounding_up(wave_size, device_count) + 1u;
}

/** Fold @p per_slot to its smallest entry, or zero when nothing was probed. */
static std::size_t smallest_slot_bytes_(std::unordered_map<int, std::size_t> const &per_slot) noexcept {
    std::size_t smallest = 0;
    for (auto const &[device_id, slot_bytes] : per_slot)
        if (smallest == 0 || slot_bytes < smallest) smallest = slot_bytes;
    return smallest;
}

/**
 *  @brief Probe every physical device behind @p device_ids once and answer the binding per-slot budget.
 *
 *  A physical device is probed once however many virtual slots run on it — the `virtual-device` pattern
 *  runs K streams on one GPU as K virtual devices — and the probe divides by that replica count on the
 *  spot, so the smallest entry already describes one slot's share.
 */
static std::size_t probe_wave_budget_bytes_(custd::span<int const> device_ids, cuda_error_sink_t const &sink) {
    std::unordered_map<int, std::uint32_t> replicas;
    std::unordered_map<int, std::size_t> free_bytes_per_slot;
    std::unordered_map<int, std::size_t> total_bytes_per_slot;
    for (int const device_id : device_ids) replicas[device_id]++;
    for (auto const &[device_id, replica_count] : replicas) {
        std::size_t free_bytes = 0, total_bytes = 0;
        if (sink.check(cudaSetDevice(device_id), "cudaSetDevice(memgetinfo)") &&
            sink.check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo")) {
            free_bytes_per_slot[device_id] = free_bytes / replica_count;
            total_bytes_per_slot[device_id] = total_bytes / replica_count;
            continue;
        }
        // `cudaGetDeviceProperties` reads the device's static description and needs no current
        // context, so it still answers for a device whose `cudaMemGetInfo` did not. That is what
        // keeps the half-of-total arm of `wave_budget_bytes` reachable rather than dead.
        cudaDeviceProp properties {};
        if (sink.check(cudaGetDeviceProperties(&properties, device_id), "cudaGetDeviceProperties"))
            total_bytes_per_slot[device_id] = properties.totalGlobalMem / replica_count;
    }
    return wave_budget_bytes(free_bytes_per_slot.size(), smallest_slot_bytes_(free_bytes_per_slot),
                             smallest_slot_bytes_(total_bytes_per_slot));
}

/**
 *  @brief Size an `embedding-wave` in molecules against the tightest per-virtual-slot VRAM.
 *
 *  Applies the `wave_budget_bytes` VRAM fraction to the census, divides by the per-molecule footprint at
 *  @p max_atoms_per_molecule and @p conformer_count, and clamps to `[10K, 400K]` for kernel-launch
 *  amortisation and pinned-host sanity respectively.
 *
 *  The constructor calls this at `max_conformer_count_k` to size its pre-allocation and
 *  `choose_wave_size_` at the caller's K; what differs between them lives there rather than here.
 */
static std::uint32_t compute_wave_size_(custd::span<int const> device_ids, std::uint32_t max_atoms_per_molecule,
                                        std::uint32_t conformer_count, cuda_error_sink_t const &sink) {
    std::size_t const budget_bytes = probe_wave_budget_bytes_(device_ids, sink);
    std::size_t const bytes_per_molecule = estimate_per_molecule_batch_bytes(max_atoms_per_molecule) +
                                           static_cast<std::size_t>(max_atoms_per_molecule) * conformer_count * 3u *
                                               sizeof(float) * 6u;
    std::uint64_t auto_wave_size = 50'000;
    if (bytes_per_molecule > 0 && budget_bytes > 0) auto_wave_size = budget_bytes / bytes_per_molecule;
    if (auto_wave_size < 10'000u) auto_wave_size = 10'000u;
    if (auto_wave_size > 400'000u) auto_wave_size = 400'000u;
    // A wave is also bounded by what a 32-bit per-molecule region offset can address, which is
    // quadratic in the caller's atom cap: 400'000 molecules is addressable up to 103 atoms each,
    // and no further. Bounding here is what keeps preprocess's own refusal unreachable.
    std::uint32_t const addressable = smallfield::preprocess::max_molecules_for_atom_capacity(max_atoms_per_molecule);
    if (auto_wave_size > addressable) auto_wave_size = addressable;
    return static_cast<std::uint32_t>(auto_wave_size);
}

#pragma endregion Wave Sizing

#pragma region MMFF Staging

/**
 *  @brief Molecules the MMFF extractor processes per pass through its fixed-stride staging.
 *
 *  The extractor and the CSR builder write into per-molecule slots of fixed stride, which costs
 *  118 KB per molecule; a whole wave's worth of that would be gigabytes of VRAM standing idle
 *  between kernels. Running the extract-build-compact triple over a fixed window instead caps the
 *  staging at this many molecules — 243 MB, allocated once per device — and costs three kernel
 *  launches per window against the milliseconds each window spends in the extractor.
 */
inline constexpr std::uint32_t mmff_extract_window_k = 2048;

/** The extractor's fixed-stride staging arrays for one window, viewed over a caller-owned allocation. */
struct mmff_staging_t {
    custd::span<mmff94_bond_t> bonds {};
    custd::span<mmff94_angle_t> angles {};
    custd::span<mmff94_torsion_t> torsions {};
    custd::span<mmff94_oop_t> oops {};
    custd::span<mmff94_stretchbend_t> stretchbends {};
    custd::span<mmff94_atom_interaction_t> interactions {};
    custd::span<mmff94_per_mol_counts_t> counts {};
};

/**
 *  @brief Place the window's staging arrays back to back out of @p cursor.
 *
 *  Every extent is a compile-time cap, so the same walk that carves a wave's staging also answers what
 *  the reservation at construction has to be — run once here over nothing, at compile time.
 *
 *  @param[inout] cursor Walk position over the staging allocation, or over nothing while sizing one.
 */
constexpr mmff_staging_t carve_mmff_staging_(pool_cursor<std::byte> &cursor) noexcept {
    std::size_t const window = mmff_extract_window_k;
    mmff_staging_t staging {};
    staging.bonds = carve<mmff94_bond_t>(cursor, window * max_mmff_bonds_per_mol_k);
    staging.angles = carve<mmff94_angle_t>(cursor, window * max_mmff_angles_per_mol_k);
    staging.torsions = carve<mmff94_torsion_t>(cursor, window * max_mmff_torsions_per_mol_k);
    staging.oops = carve<mmff94_oop_t>(cursor, window * max_mmff_oops_per_mol_k);
    staging.stretchbends = carve<mmff94_stretchbend_t>(cursor, window * max_mmff_stretchbends_per_mol_k);
    staging.interactions = carve<mmff94_atom_interaction_t>(cursor, window * max_mmff_interactions_per_mol_k);
    staging.counts = carve<mmff94_per_mol_counts_t>(cursor, window);
    return staging;
}

/** Bytes one window's staging spans, which is what a device reserves for it once. */
constexpr std::size_t mmff_staging_bytes_() noexcept {
    pool_cursor<std::byte> cursor {};
    [[maybe_unused]] mmff_staging_t const unallocated = carve_mmff_staging_(cursor);
    return cursor.bytes;
}

inline constexpr std::size_t mmff_staging_bytes_k = mmff_staging_bytes_();
static_assert(mmff_staging_bytes_k > 0,
              "a zero-byte staging total would reserve nothing and hand the extractor a null base");

#pragma endregion MMFF Staging

#pragma region Wave Timing

/** The phases one wave spends on a device, in the order its stream runs them. */
enum class wave_phase_t : std::uint8_t {
    /** Host-to-device copy of the batch pool's uploaded prefix. */
    upload_k,
    /** Device preprocessing plus `build_metric_initial_coordinates_on_gpu`. */
    metric_k,
    /** The wave-binned pipeline and polish launches, and the half-precision conversion. */
    kernels_k,
    /** Device-to-host copies of coordinates, energies and status. */
    download_k,
};

/** Phases a wave is timed in. One more boundary event than this is needed to bracket them all. */
inline constexpr std::size_t wave_phase_count_k = 4;

/**
 *  @brief One wave's per-phase GPU and DMA wall on one device, in milliseconds.
 *
 *  Measured on events recorded into the device's own stream, so this is device time rather than
 *  host-side scheduling.
 */
struct gpu_phase_ms_t {
    /** Indexed by `wave_phase_t`. */
    std::array<float, wave_phase_count_k> phase_ms {};
    /** First boundary to last, so it also covers whatever sits between two phases. */
    float total_ms = 0.f;
};

/**
 *  @brief Stream-local timestamps bracketing every phase of one wave on one device.
 *
 *  One boundary more than there are phases, so phase `p` spans `[boundaries[p], boundaries[p + 1])`.
 *  Creation, recording, readback and destruction all walk the same array, which is what stops a phase
 *  from being timed against its neighbour's event.
 *
 *  Created once per device and reused: a `cudaEventDestroy` on one stream racing a `cudaEventCreate`
 *  from the next launch's fan-out thread is what per-wave events cost. `cudaEventBlockingSync` keeps
 *  the host out of a busy-wait inside `cudaStreamSynchronize` when many streams share CPU cores.
 */
struct phase_events_t {
    std::array<cudaEvent_t, wave_phase_count_k + 1> boundaries {};

    /** Whether the boundaries exist, and therefore whether a readback would mean anything. */
    [[nodiscard]] bool created() const noexcept { return boundaries.back() != nullptr; }

    /** Create every boundary event on the current device. */
    void create(cuda_error_sink_t const &sink) noexcept {
        for (cudaEvent_t &boundary : boundaries) {
            [[maybe_unused]] bool const made = sink.check(cudaEventCreateWithFlags(&boundary, cudaEventBlockingSync),
                                                          "cudaEventCreateWithFlags");
        }
    }

    /** Destroy every boundary event; safe to call on a set that was never created. */
    void destroy() noexcept {
        for (cudaEvent_t &boundary : boundaries) {
            if (boundary == nullptr) continue;
            [[maybe_unused]] cudaError_t const status = cudaEventDestroy(boundary);
            boundary = nullptr;
        }
    }

    /** Stamp the boundary that opens @p phase into @p stream, before that phase's first call. */
    void open(wave_phase_t phase, cudaStream_t stream) const noexcept {
        record_(static_cast<std::size_t>(phase), stream);
    }

    /** Stamp the closing boundary, after the last phase's work is enqueued. */
    void close(cudaStream_t stream) const noexcept { record_(wave_phase_count_k, stream); }

    /** Read every phase's duration out of the recorded boundaries; valid only once the stream drains. */
    void elapsed_into(gpu_phase_ms_t &timings) const noexcept {
        if (!created()) return;
        for (std::size_t phase = 0; phase < wave_phase_count_k; ++phase) {
            [[maybe_unused]] cudaError_t const status = cudaEventElapsedTime(&timings.phase_ms[phase],
                                                                             boundaries[phase], boundaries[phase + 1]);
        }
        [[maybe_unused]] cudaError_t const status = cudaEventElapsedTime(&timings.total_ms, boundaries.front(),
                                                                         boundaries.back());
    }

  private:
    void record_(std::size_t boundary, cudaStream_t stream) const noexcept {
        if (boundaries[boundary] == nullptr) return;
        [[maybe_unused]] cudaError_t const status = cudaEventRecord(boundaries[boundary], stream);
    }
};

/** Milliseconds since construction, which is what every host-side wave boundary is stated against. */
struct wall_clock_t {
    std::chrono::high_resolution_clock::time_point origin = std::chrono::high_resolution_clock::now();

    [[nodiscard]] double elapsed_ms() const noexcept {
        return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - origin).count();
    }
};

/** Host wall-clock `[start, end)` of one step, in milliseconds relative to the call. */
struct phase_span_ms_t {
    double start_ms = 0;
    double end_ms = 0;

    [[nodiscard]] double duration_ms() const noexcept { return end_ms - start_ms; }
};

/**
 *  @brief Stamps a host wall-clock span on entry and on exit, so a step's two boundaries cannot drift.
 */
class scoped_span_ms_t {
    wall_clock_t const &clock_;
    phase_span_ms_t &span_;

  public:
    scoped_span_ms_t(wall_clock_t const &clock, phase_span_ms_t &span) noexcept : clock_(clock), span_(span) {
        span_.start_ms = clock_.elapsed_ms();
    }
    ~scoped_span_ms_t() noexcept { span_.end_ms = clock_.elapsed_ms(); }

    scoped_span_ms_t(scoped_span_ms_t const &) = delete;
    scoped_span_ms_t &operator=(scoped_span_ms_t const &) = delete;
};

/**
 *  @brief Per-`embedding-wave` timing record, pretty-printed by `print_wave_timeline_` at the end of
 *  `generate_conformers` when `logger.verbose_level >= 1`.
 */
struct wave_record_t {
    std::uint32_t wave_index = 0;
    std::uint32_t num_molecules = 0;
    phase_span_ms_t prepare {};
    phase_span_ms_t launch {};
    phase_span_ms_t finalize {};
    /** One entry per device, filled when the wave is finalized. */
    std::vector<gpu_phase_ms_t> per_device_gpu {};
};

/** One wave's per-device GPU timings folded to the two figures the timeline prints. */
struct wave_gpu_summary_t {
    float average_ms = 0.f;
    float max_ms = 0.f;
    /** Indexed by `wave_phase_t`, averaged over the devices that did work. */
    std::array<float, wave_phase_count_k> average_phase_ms {};
};

/** Fold the devices that did work; a device whose stream reported nothing is left out of the average. */
static wave_gpu_summary_t summarize_wave_gpu_(std::vector<gpu_phase_ms_t> const &per_device) noexcept {
    wave_gpu_summary_t summary {};
    float total_sum = 0.f;
    std::uint32_t active_devices = 0;
    for (gpu_phase_ms_t const &device_timing : per_device) {
        if (device_timing.total_ms <= 0.f) continue;
        summary.max_ms = std::max(summary.max_ms, device_timing.total_ms);
        total_sum += device_timing.total_ms;
        for (std::size_t phase = 0; phase < wave_phase_count_k; ++phase)
            summary.average_phase_ms[phase] += device_timing.phase_ms[phase];
        ++active_devices;
    }
    if (active_devices == 0) return summary;
    float const inverse_active = 1.f / static_cast<float>(active_devices);
    summary.average_ms = total_sum * inverse_active;
    for (float &phase_average : summary.average_phase_ms) phase_average *= inverse_active;
    return summary;
}

/**
 *  @brief Verbose-tier-1 per-wave summary plus a verbose-tier-2 per-device breakdown.
 *
 *  No-op when `logger.verbose_level < 1`.
 */
static void print_wave_timeline_(logger_t const &logger, std::vector<wave_record_t> const &wave_records,
                                 double total_wall_ms) {
    if (logger.verbose_level < 1) return;

    // One width per column kind, shared by the header and every row through `{:>{}}` dynamic width, so the
    // two cannot drift apart. `time_width_k` is 8 because the widest header word — `finalize`, `download` —
    // is 8 characters, and a width narrower than its own label is what silently misaligned this table before.
    constexpr int wave_width_k = 4;
    constexpr int mols_width_k = 7;
    constexpr int time_width_k = 8;

    logger.trace("{:>{}} {:>{}} {:>{}} {:>{}} {:>{}} {:>{}} / {:>{}} {:>{}} / {:>{}} / {:>{}} / {:>{}}", //
                 "wave", wave_width_k, "mols", mols_width_k, "prep", time_width_k, "launch", time_width_k, "finalize",
                 time_width_k, "gpu avg", time_width_k, "max", time_width_k, "upload", time_width_k, "metric",
                 time_width_k, "kernels", time_width_k, "download", time_width_k);
    for (wave_record_t const &wave_record : wave_records) {
        wave_gpu_summary_t const summary = summarize_wave_gpu_(wave_record.per_device_gpu);
        logger.trace("{:>{}} {:>{}} {:>{}.1f} {:>{}.1f} {:>{}.1f} {:>{}.1f} / {:>{}.1f} {:>{}.1f} / {:>{}.1f} / " //
                     "{:>{}.1f} / {:>{}.1f}",
                     wave_record.wave_index, wave_width_k, wave_record.num_molecules, mols_width_k,
                     wave_record.prepare.duration_ms(), time_width_k, wave_record.launch.duration_ms(), time_width_k,
                     wave_record.finalize.duration_ms(), time_width_k, summary.average_ms, time_width_k, summary.max_ms,
                     time_width_k, summary.average_phase_ms[static_cast<std::size_t>(wave_phase_t::upload_k)],
                     time_width_k, summary.average_phase_ms[static_cast<std::size_t>(wave_phase_t::metric_k)],
                     time_width_k, summary.average_phase_ms[static_cast<std::size_t>(wave_phase_t::kernels_k)],
                     time_width_k, summary.average_phase_ms[static_cast<std::size_t>(wave_phase_t::download_k)],
                     time_width_k);
    }
    logger.trace("wall {:.1f} ms", total_wall_ms);
    for (wave_record_t const &wave_record : wave_records) {
        for (std::uint32_t device_index = 0; device_index < wave_record.per_device_gpu.size(); ++device_index) {
            gpu_phase_ms_t const &device_timing = wave_record.per_device_gpu[device_index];
            if (device_timing.total_ms <= 0.f) continue;
            logger.trace2(
                "  wave={} dev={}  upload={:.1f}  metric={:.1f}  kernels={:.1f}  download={:.1f}  total={:.1f}",
                wave_record.wave_index, device_index,
                device_timing.phase_ms[static_cast<std::size_t>(wave_phase_t::upload_k)],
                device_timing.phase_ms[static_cast<std::size_t>(wave_phase_t::metric_k)],
                device_timing.phase_ms[static_cast<std::size_t>(wave_phase_t::kernels_k)],
                device_timing.phase_ms[static_cast<std::size_t>(wave_phase_t::download_k)], device_timing.total_ms);
        }
    }
}

#pragma endregion Wave Timing

#pragma region Launch Scratch

/**
 *  @brief Persistent per-device scratch for the kernel launch path.
 *
 *  `device_pool` holds the batch-input mirror, the per-launch outputs and the metric scratch;
 *  `host_staging` holds the pinned readback regions the three device-to-host copies land in. Both are
 *  grow-only and reused across waves, so the single-threaded dispatcher never allocates per launch.
 */
struct launch_scratch_t {
    /**
     *  @brief Bind this device's stream, which every device allocation here is ordered against.
     *
     *  @param[in] stream Outlives this scratch: `~context_t` resets the scratch before destroying the
     *                    stream, which is what `cudaFreeAsync` requires.
     */
    explicit launch_scratch_t(cudaStream_t stream) noexcept
        : device_pool(device_allocator_t {stream}), mmff_staging(device_allocator_t {stream}) {}

    device_bytes_t device_pool;
    pinned_bytes_t host_staging;
    /** Fixed-stride staging the MMFF extractor and CSR builder write into, one window's
     *  worth; the compaction pass copies out of it into the batch pool's packed arrays. */
    device_bytes_t mmff_staging;
    /** This device's copy of the MMFF94 parameter tables, uploaded once at construction. */
    mmff94_tables_device_t mmff_tables;
    /** This device's copy of the compiled CSD torsion patterns, uploaded once. */
    csd_torsion_table_device_t csd_torsion_table;
    /** Persistent phase-timing boundaries, created once and re-recorded by every wave. */
    phase_events_t events;
};

/** Device-side regions one wave carves out of `launch_scratch_t::device_pool`. */
struct device_wave_regions_t {
    /** Mirror of the batch pool: the uploaded prefix plus the regions the preprocess kernels fill. */
    custd::span<std::byte> batch {};
    /** Kernel output, in single precision. */
    custd::span<float> coordinates_fp32 {};
    /** Post-conversion source of the coordinate download; halves the transfer. */
    custd::span<__half> coordinates_fp16 {};
    custd::span<float> energies {};
    /** One `conformer_status_t` byte per slot, written by the pipeline kernel and then by the stamp. */
    custd::span<std::uint8_t> status {};
    /** Four-dimensional metric embedding the prepass writes and the pipeline kernel starts from. */
    custd::span<float> metric_embedding {};
    /** What the prepass decided about each molecule's bounds, which only the stamp reads. */
    custd::span<conformer_status_t> molecule_verdicts {};
};

/** Pinned host regions the wave's three device-to-host copies land in. */
struct host_wave_regions_t {
    custd::span<__half> coordinates {};
    custd::span<float> energies {};
    custd::span<std::uint8_t> status {};
};

/**
 *  @brief Element counts one wave's regions are carved at, derived once from a dispatch's geometry.
 *
 *  The device and host walks both read these, so a coordinate count the two sides disagreed on is not
 *  spellable — and the constructor's worst-case reservation goes through the same two walks the launch
 *  path carves with.
 */
struct wave_pool_shape_t {
    /** Bytes the batch mirror spans, uploaded prefix and device-only regions together. */
    std::size_t batch_bytes = 0;
    /** Conformer slots the wave holds, one per molecule per requested conformer. */
    std::size_t conformer_slots = 0;
    /** Floats one wave's coordinates span, at three per atom of the wave's own stride. */
    std::size_t coordinate_count = 0;
    /** Floats the four-dimensional metric embedding spans, one molecule at a time. */
    std::size_t embedding_count = 0;
    /** Molecules this device carries. */
    std::uint32_t molecule_count = 0;
};

/**
 *  @brief Derive a wave's region extents from the shape of the dispatch.
 *
 *  @param[in] batch_bytes Bytes the batch pool spans, uploaded prefix and device-only regions together.
 *  @param[in] molecule_count Molecules this device carries.
 *  @param[in] conformer_count Conformers requested per molecule.
 *  @param[in] max_atoms Coordinate stride, which is this wave's own largest atom count.
 */
static wave_pool_shape_t compute_wave_pool_shape_(std::size_t batch_bytes, std::uint32_t molecule_count,
                                                  std::uint32_t conformer_count, std::uint32_t max_atoms) noexcept {
    std::size_t const conformer_slots = static_cast<std::size_t>(molecule_count) * conformer_count;
    return {.batch_bytes = batch_bytes,
            .conformer_slots = conformer_slots,
            .coordinate_count = conformer_slots * max_atoms * types::conformer_dimensions_k,
            .embedding_count = static_cast<std::size_t>(molecule_count) * max_atoms * types::embedding_dimensions_k,
            .molecule_count = molecule_count};
}

/** Carve the device side of one wave out of @p cursor, or size it when the cursor holds no pool. */
static device_wave_regions_t carve_device_wave_(wave_pool_shape_t const &shape,
                                                pool_cursor<std::byte> &cursor) noexcept {
    device_wave_regions_t regions {};
    regions.batch = carve<std::byte>(cursor, shape.batch_bytes);
    regions.coordinates_fp32 = carve<float>(cursor, shape.coordinate_count);
    regions.coordinates_fp16 = carve<__half>(cursor, shape.coordinate_count);
    regions.energies = carve<float>(cursor, shape.conformer_slots);
    regions.status = carve<std::uint8_t>(cursor, shape.conformer_slots);
    regions.metric_embedding = carve<float>(cursor, shape.embedding_count);
    regions.molecule_verdicts = carve<conformer_status_t>(cursor, shape.molecule_count);
    return regions;
}

/** Carve the pinned host side of one wave out of @p cursor, or size it when the cursor holds no pool. */
static host_wave_regions_t carve_host_wave_(wave_pool_shape_t const &shape, pool_cursor<std::byte> &cursor) noexcept {
    host_wave_regions_t staged {};
    staged.coordinates = carve<__half>(cursor, shape.coordinate_count);
    staged.energies = carve<float>(cursor, shape.conformer_slots);
    staged.status = carve<std::uint8_t>(cursor, shape.conformer_slots);
    return staged;
}

/** Bytes the device side of a wave of this shape spans, from the walk that carves it. */
static std::size_t device_wave_bytes_(wave_pool_shape_t const &shape) noexcept {
    pool_cursor<std::byte> cursor {};
    [[maybe_unused]] device_wave_regions_t const unallocated = carve_device_wave_(shape, cursor);
    return align_up_(cursor.bytes, pool_alignment_k);
}

/** Bytes the pinned host side of a wave of this shape spans, from the walk that carves it. */
static std::size_t host_wave_bytes_(wave_pool_shape_t const &shape) noexcept {
    pool_cursor<std::byte> cursor {};
    [[maybe_unused]] host_wave_regions_t const unallocated = carve_host_wave_(shape, cursor);
    return align_up_(cursor.bytes, pool_alignment_k);
}

/**
 *  @brief Grow the per-device metric scratch for the given group size and atom count; no-op on shrink.
 *
 *  The first wave of a workload pays the allocation and every later one reuses it. Declared here
 *  because `context_t`'s constructor pre-warms it, so the first user call does not allocate. A refused
 *  grow leaves the previous allocation and its capacity in place, and the prepass re-tests that
 *  capacity against the group it is about to bin, so the wave degrades rather than overruns.
 */
void ensure_metric_scratch(context_t &context, std::size_t device_index, std::size_t max_group_size,
                           std::uint32_t max_atoms) {
    auto &scratch = context.metric_scratch[device_index];
    if (max_group_size <= scratch.group_indices.capacity() && max_atoms <= scratch.max_atoms &&
        scratch.group_indices.data() != nullptr)
        return;

    std::size_t const new_group = std::max(scratch.group_indices.capacity(), max_group_size);
    std::uint32_t const new_atoms = std::max<std::uint32_t>(scratch.max_atoms, max_atoms);

    cudaSetDevice(context.device_ids[device_index]);

    // `embed_4d_power_iter_kernel` runs entirely out of shared memory — Gram matrix, means,
    // eigenvectors and reduction scratch all live there for the kernel's lifetime — so the only
    // persistent device scratch is the `group_indices` uint32 list of which molecule goes into
    // which bin, which is KB-scale.
    if (scratch.group_indices.try_reserve(new_group) == reserve_outcome_t::failed_k) {
        context.logger.error("ensure_metric_scratch: device allocation ({} bytes) failed",
                             new_group * sizeof(std::uint32_t));
        return;
    }
    scratch.max_atoms = new_atoms;
}

#pragma endregion Launch Scratch

#pragma region Wave Launch

/**
 *  @brief Where a wave's results land in the caller's arrays, and the permutation that takes them there.
 *
 *  `sort_perm` maps a position in the SMILES-length-sorted input to the caller's own row, and
 *  `wave_base` is the first sorted position this wave covers. The three spans are the caller's whole
 *  buffers rather than this wave's slice, because the permutation scatters across all of them.
 */
struct wave_outputs_t {
    custd::span<__half> coordinates {};
    custd::span<float> energies {};
    custd::span<conformer_status_t> status {};
    custd::span<std::uint32_t const> sort_perm {};
    std::uint32_t wave_base = 0;

    /** Caller row the molecule at @p input_index within this wave belongs to. */
    [[nodiscard]] std::uint32_t caller_index_of(std::uint32_t input_index) const noexcept {
        return sort_perm[wave_base + input_index];
    }
};

/**
 *  @brief The four optional per-molecule readbacks, each empty when the caller did not ask for it.
 *
 *  Every one describes the parent fragment preprocess kept, and is indexed by the caller's own row.
 */
struct molecule_readback_t {
    custd::span<std::uint8_t> num_atoms {};
    custd::span<std::uint16_t> num_heavy_atoms {};
    custd::span<std::uint16_t> num_bonds {};
    custd::span<float> molecular_weight {};

    /** Whether any of the four was asked for, so a wave can skip the readback walk entirely. */
    [[nodiscard]] bool wanted() const noexcept {
        return !num_atoms.empty() || !num_heavy_atoms.empty() || !num_bonds.empty() || !molecular_weight.empty();
    }

    /** Reset every requested array to what a molecule that never reached the GPU keeps. */
    void clear() const noexcept {
        std::fill_n(num_atoms.begin(), num_atoms.size(), std::uint8_t {0});
        std::fill_n(num_heavy_atoms.begin(), num_heavy_atoms.size(), std::uint16_t {0});
        std::fill_n(num_bonds.begin(), num_bonds.size(), std::uint16_t {0});
        std::fill_n(molecular_weight.begin(), molecular_weight.size(), 0.0f);
    }

    /** Record one molecule's post-preprocess description at the caller's row. */
    void write(std::uint32_t caller_index, etkdg_molecule_metadata_t const &metadata) const noexcept {
        if (!num_atoms.empty()) num_atoms[caller_index] = static_cast<std::uint8_t>(metadata.num_atoms);
        if (!num_heavy_atoms.empty())
            num_heavy_atoms[caller_index] = static_cast<std::uint16_t>(metadata.num_heavy_atoms);
        if (!num_bonds.empty()) num_bonds[caller_index] = static_cast<std::uint16_t>(metadata.num_bonds);
        if (!molecular_weight.empty()) molecular_weight[caller_index] = metadata.molecular_weight;
    }
};

/**
 *  @brief Widen the byte a kernel wrote into the public per-conformer status, and check the energy
 *  behind a success.
 *
 *  Four values reach this buffer: `write_conformer_output_` writes `success_k` or `stereo_failure_k`, and
 *  `stamp_refused_molecules_kernel` overwrites both with the molecule's verdict — `bounds_infeasible_k`
 *  where the metric prepass found bounds no coordinates satisfy, `batch_overflow_k` where that prepass
 *  refused the whole batch over scratch it could not fit. Anything else is a slot no kernel reached, and
 *  it reports as unconverged rather than as a success the coordinates would not back.
 *
 *  A success is only a success while its energy is a number a consumer can rank by, so the two facts are
 *  decided together here rather than left for six consumers to re-derive from the status byte alone.
 */
static conformer_status_t conformer_status_of_(std::uint8_t raw, float energy) noexcept {
    auto const status = static_cast<conformer_status_t>(raw);
    switch (status) {
    case conformer_status_t::success_k:
        return energy_is_physical(energy) ? status : conformer_status_t::nonphysical_energy_k;
    case conformer_status_t::stereo_failure_k:
    case conformer_status_t::bounds_infeasible_k:
    case conformer_status_t::batch_overflow_k: return status;
    default: return conformer_status_t::mmff_unconverged_k;
    }
}

/**
 *  @brief Report the launch error the preceding `<<<>>>` left on the device, if any.
 *
 *  `cudaGetLastError` rather than `cudaPeekAtLastError`, because peeking leaves the sticky slot set:
 *  this runs eleven times per wave, so one refused kernel was re-reported by every later call under
 *  the name of a kernel that had not failed, and `cuda_error_count` counted one failure eleven times.
 *  It also picked up an error left by an earlier `cudaMemcpyAsync` on the same host thread and blamed
 *  it on whichever launch came next.
 */
static void check_launch_(cuda_error_sink_t const &sink, char const *label) noexcept {
    [[maybe_unused]] bool const launched = sink.check(cudaGetLastError(), label);
}

/**
 *  @brief Enqueue everything that turns an uploaded batch of descriptors into the force field the
 *  conformer kernels read.
 *
 *  Six passes, all asynchronous on @p stream:
 *    1. the hop-count BFS over each descriptor's bond list;
 *    2. the MMFF atom typer and its partial-charge pass;
 *    3. the typing patch, which hands the molecules the native typer rejects RDKit's answer;
 *    4. the MMFF term extractor, CSR builder and compaction, one window of molecules at a time;
 *    5. the CSD torsion matcher;
 *    6. the ETK per-atom CSR builder.
 *
 *  No pass reads anything back. The per-molecule bases each writer needs came up with the batch
 *  in `mmff_bases`, prefix-summed on the host from capacity bounds, so the packed-array sizing
 *  never has to wait on `mmff94_scan_counts_kernel`'s totals.
 *
 *  @param[in] device_view Device mirror of the batch pool.
 *  @param[in] torsion_stride Slots the CSD matcher writes per molecule, as the slice was carved.
 *  @param[inout] launch_buffers This device's persistent scratch, holding the parameter tables and
 *                the extractor's fixed-stride staging.
 *  @param[in] molecule_count Molecules in the batch.
 *  @param[in] stream Stream to enqueue on.
 *  @param[in] sink Error sink routing failures through the caller's logger.
 */
static void enqueue_device_preprocess_(wave_slice_view_t const &device_view, std::uint32_t torsion_stride,
                                       launch_scratch_t &launch_buffers, std::uint32_t molecule_count,
                                       cudaStream_t stream, cuda_error_sink_t const &sink) {
    if (molecule_count == 0) return;

    auto *const descriptors = device_view.descriptors.data();
    auto *const metadata = device_view.molecule_metadata.data();
    auto *const atom_bases = device_view.atom_bases.data();
    auto *const mmff_bases = device_view.mmff_bases.data();
    auto *const atom_types = device_view.atom_types.data();
    auto *const atom_charges = device_view.mmff_atom_charge.data();
    auto *const typing_status = device_view.typing_status.data();
    auto *const atom_offsets = device_view.mmff_atom_offsets.data();

    launch_bond_distance_bfs(descriptors, metadata, device_view.bond_distance.data(), molecule_count, stream);
    check_launch_(sink, "bond_distance_bfs_kernel launch");

    mmff94_type_and_charge_kernel<<<molecule_count, warp_size_k, 0, stream>>>(descriptors, atom_bases, molecule_count,
                                                                              launch_buffers.mmff_tables.view,
                                                                              atom_types, atom_charges, typing_status);
    check_launch_(sink, "mmff94_type_and_charge_kernel launch");

    std::uint32_t const patch_count = static_cast<std::uint32_t>(device_view.typing_patches.size());
    if (sink.log != nullptr)
        sink.log->trace2("device preprocess: {} of {} molecules take the host typing path", patch_count,
                         molecule_count);
    // A batch every molecule typed natively leaves no patch records, and a zero-block launch is a
    // configuration error rather than a no-op.
    if (patch_count != 0) {
        mmff94_apply_typing_patch_kernel<<<patch_count, warp_size_k, 0, stream>>>(
            device_view.typing_patches.data(), patch_count, device_view.patch_atom_types.data(),
            device_view.patch_atom_charges.data(), atom_types, atom_charges, typing_status);
        check_launch_(sink, "mmff94_apply_typing_patch_kernel launch");
    }

    pool_cursor<std::byte> staging_cursor {launch_buffers.mmff_staging.view()};
    mmff_staging_t const staging = carve_mmff_staging_(staging_cursor);

    for (std::uint32_t window_start = 0; window_start < molecule_count; window_start += mmff_extract_window_k) {
        std::uint32_t const window_size = std::min(mmff_extract_window_k, molecule_count - window_start);
        mmff94_extract_terms_kernel<<<window_size, warp_size_k, 0, stream>>>(
            descriptors + window_start, atom_types, atom_bases + window_start, window_size,
            launch_buffers.mmff_tables.view, device_view.mmff_atom_vdw.data(), staging.bonds.data(),
            staging.angles.data(), staging.torsions.data(), staging.oops.data(), staging.stretchbends.data(),
            staging.counts.data(), typing_status + window_start);
        check_launch_(sink, "mmff94_extract_terms_kernel launch");

        mmff94_build_csr_kernel<<<window_size, warp_size_k, 0, stream>>>(
            descriptors + window_start, atom_bases + window_start, window_size, staging.counts.data(),
            staging.bonds.data(), staging.angles.data(), staging.torsions.data(), staging.oops.data(),
            staging.stretchbends.data(), staging.interactions.data(), atom_offsets, typing_status + window_start);
        check_launch_(sink, "mmff94_build_csr_kernel launch");

        mmff94_compact_terms_kernel<<<window_size, warp_size_k, 0, stream>>>(
            descriptors + window_start, atom_bases + window_start, window_size, staging.counts.data(),
            mmff_bases + window_start, staging.bonds.data(), staging.angles.data(), staging.torsions.data(),
            staging.oops.data(), staging.stretchbends.data(), staging.interactions.data(),
            device_view.mmff_bonds.data(), device_view.mmff_angles.data(), device_view.mmff_torsions.data(),
            device_view.mmff_oops.data(), device_view.mmff_stretchbends.data(), device_view.mmff_interactions.data(),
            atom_offsets, metadata + window_start, typing_status + window_start);
        check_launch_(sink, "mmff94_compact_terms_kernel launch");
    }

    csd_torsion_match_kernel<<<molecule_count, warp_size_k, 0, stream>>>(
        descriptors, molecule_count, launch_buffers.csd_torsion_table.view, device_view.etk_torsions.data(),
        torsion_stride, device_view.torsion_counts.data());
    check_launch_(sink, "csd_torsion_match_kernel launch");

    etk_build_csr_kernel<<<molecule_count, warp_size_k, 0, stream>>>(
        device_view.etk_torsions.data(), torsion_stride, device_view.torsion_counts.data(),
        device_view.distance_constraints.data(), molecule_count, device_view.etk_interactions.data(),
        device_view.etk_atom_offsets.data(), metadata);
    check_launch_(sink, "etk_build_csr_kernel launch");
}

/**
 *  @brief Per-wave, per-GPU state carried from `launch_gpu_pipeline` to `finalize_gpu_pipeline`.
 *
 *  Holds only what finalize still needs: the stream to drain, the staging the download lands in, the
 *  two permutations that take a staged slot back to a caller row, and the two coordinate strides.
 *  The device-side regions do not appear, because nothing reads them after the launch returns.
 */
struct gpu_pipeline_handle_t {
    int device_id = -1;
    cudaStream_t stream = nullptr;
    /**
     *  @brief Sorted-index to input-index map, viewing this device's pinned pool.
     *
     *  A view rather than a pointer to the pool, because the pinned allocation does not move while a
     *  wave is in flight but the container holding the pools may be reshaped. Its extent is this
     *  device's molecule count, so the two never have to be carried separately.
     */
    custd::span<std::uint32_t const> input_indices {};
    std::uint32_t conformer_count = 0;
    /**
     *  @brief The largest atom count this wave actually carries, driving the GPU coordinate stride
     *  and the device-to-host staging stride.
     *
     *  An observation rather than a cap, so a wave of small molecules pays no full-stride tax in
     *  output VRAM or download bandwidth. It never exceeds `host_max_atoms`, and that is a gate rather
     *  than a coincidence: `prepare_per_device_batches_into` refuses any molecule above the caller's
     *  `output_atom_capacity` with `too_many_atoms_k`.
     */
    std::uint32_t gpu_max_atoms = 0;
    /**
     *  @brief The caller's own output stride, which is what `outputs.coordinates` was allocated at.
     *
     *  Finalize copies `gpu_max_atoms` halves per axis into a slot `host_max_atoms` wide, leaving the
     *  trailing halves at their pre-zeroed value. The two strides are independent numbers and
     *  conflating them is what once wrote one wave's y plane over another's x.
     */
    std::uint32_t host_max_atoms = 0;
    /** Pinned staging the download wrote into; finalize drains the stream, then permutes out of it. */
    host_wave_regions_t staged {};
    /** Caller-owned destination, and the permutation from a staged slot to a caller row. */
    wave_outputs_t outputs {};
    /** Phase boundaries this wave recorded, borrowed from the device's persistent scratch. */
    phase_events_t events {};
    /** Filled by `finalize_gpu_pipeline` once the stream drains. */
    gpu_phase_ms_t timings {};
    /** Routes errors through the caller's logger and counter. */
    cuda_error_sink_t sink {};

    /** Whether this device carried any molecules, and therefore whether finalize has work. */
    [[nodiscard]] bool has_work() const noexcept { return !input_indices.empty(); }
};

/**
 *  @brief Enqueue the per-bin pipeline and polish launches for one device's whole batch, and the stamp
 *  that closes the status buffer behind them.
 *
 *  Molecules arrive atom-count ascending, so each atom-limited bin is a contiguous run and the bin's
 *  shared-memory footprint and occupancy come straight out of the construction-time launch cache.
 *  Both kernels assign one warp per `(molecule, conformer)` pair, so the grid covers the pair count
 *  rather than the molecule count; several bins hold fewer molecules than the device has resident-warp
 *  slots, and scaling by the conformer count is what lets those launches fill the device.
 *
 *  @param[in] device_view Device mirror of the batch pool.
 *  @param[in] host_metadata This device's molecules in the same order, read to find the bin edges.
 *  @param[in] regions Device output regions this wave writes into.
 *  @param[in] launch_config Per-bin occupancy, probed once at construction.
 *  @param[in] config Conformer count, stereo retry limit and seed.
 *  @param[in] gpu_max_atoms Coordinate stride these launches write at.
 *  @param[in] stream Stream to enqueue on.
 *  @param[in] sink Error sink routing failures through the caller's logger.
 */
static void enqueue_binned_kernels_(wave_slice_view_t const &device_view,
                                    custd::span<etkdg_molecule_metadata_t const> host_metadata,
                                    device_wave_regions_t const &regions, device_launch_config_t const &launch_config,
                                    config_t const &config, std::uint32_t gpu_max_atoms, cudaStream_t stream,
                                    cuda_error_sink_t const &sink) {
    std::uint32_t const molecule_count = static_cast<std::uint32_t>(host_metadata.size());
    std::uint32_t const conformer_count = config.num_conformers;
    auto *const device_metadata = device_view.molecule_metadata.data();
    int const multiprocessor_count = launch_config.streaming_multiprocessor_count;

    std::uint32_t bin_start = 0;
    for (std::size_t bin_index = 0; bin_index < atom_limited_bin_count_k && bin_start < molecule_count; ++bin_index) {
        std::uint32_t bin_end = bin_start;
        while (bin_end < molecule_count && host_metadata[bin_end].num_atoms <= atom_limited_bin_caps_k[bin_index])
            ++bin_end;
        if (bin_end == bin_start) continue;

        std::uint32_t const bin_size = bin_end - bin_start;
        std::uint32_t const bin_max_atoms = atom_limited_bin_caps_k[bin_index];
        std::uint32_t const shared_bytes_etkdg = compute_pipeline_shared_memory_bytes(bin_max_atoms);
        std::uint32_t const shared_bytes_mmff = compute_mmff_polish_shared_memory_bytes(bin_max_atoms);
        std::uint32_t const bin_pair_count = bin_size * conformer_count;
        std::uint32_t const grid_size_etkdg = std::min(
            bin_pair_count, static_cast<std::uint32_t>(launch_config.blocks_per_multiprocessor_etkdg[bin_index] *
                                                       multiprocessor_count));
        std::uint32_t const grid_size_mmff = std::min(
            bin_pair_count,
            static_cast<std::uint32_t>(launch_config.blocks_per_multiprocessor_mmff[bin_index] * multiprocessor_count));
        std::size_t const coordinate_base = static_cast<std::size_t>(bin_start) * conformer_count * gpu_max_atoms *
                                            types::conformer_dimensions_k;
        std::size_t const slot_base = static_cast<std::size_t>(bin_start) * conformer_count;

        capture_outcome_t const capture = capture_bin_launch(device_view, regions.batch, regions.metric_embedding,
                                                             {.bin_start = bin_start,
                                                              .bin_size = bin_size,
                                                              .conformer_count = conformer_count,
                                                              .max_stereo_attempts = config.max_stereo_attempts,
                                                              .bin_max_atoms = bin_max_atoms,
                                                              .gpu_max_atoms = gpu_max_atoms,
                                                              .grid_etkdg = grid_size_etkdg,
                                                              .grid_mmff = grid_size_mmff,
                                                              .shared_bytes_etkdg = shared_bytes_etkdg,
                                                              .shared_bytes_mmff = shared_bytes_mmff,
                                                              .random_seed = config.random_seed},
                                                             stream);

        etkdg_pipeline_warp_kernel<<<grid_size_etkdg, warp_size_k, shared_bytes_etkdg, stream>>>(
            device_view.bounds_matrices.data(),
            regions.metric_embedding.data() +
                static_cast<std::size_t>(bin_start) * gpu_max_atoms * types::embedding_dimensions_k,
            device_view.chiral_centers.data(), device_view.etk_torsions.data(), device_view.distance_constraints.data(),
            device_view.etk_interactions.data(), device_view.etk_atom_offsets.data(), device_metadata + bin_start,
            device_view.input_indices.data() + bin_start, regions.coordinates_fp32.data() + coordinate_base,
            regions.energies.data() + slot_base, regions.status.data() + slot_base, bin_size, conformer_count,
            config.max_stereo_attempts, bin_max_atoms, gpu_max_atoms, gpu_max_atoms, config.random_seed);
        check_launch_(sink, "etkdg_pipeline_warp_kernel launch");

        mmff_polish_warp_kernel<<<grid_size_mmff, warp_size_k, shared_bytes_mmff, stream>>>(
            device_view.mmff_atom_vdw.data(), device_view.mmff_atom_charge.data(), device_view.bond_distance.data(),
            device_view.mmff_bonds.data(), device_view.mmff_angles.data(), device_view.mmff_torsions.data(),
            device_view.mmff_oops.data(), device_view.mmff_stretchbends.data(), device_view.mmff_interactions.data(),
            device_view.mmff_atom_offsets.data(), device_metadata + bin_start,
            regions.coordinates_fp32.data() + coordinate_base, regions.energies.data() + slot_base,
            regions.status.data() + slot_base, bin_size, conformer_count, gpu_max_atoms, bin_max_atoms);
        check_launch_(sink, "mmff_polish_warp_kernel launch");
        if (capture == capture_outcome_t::captured_k) report_captured_bin_timing(stream);

        bin_start = bin_end;
    }

    // Last on the status buffer, and after every bin: the two warp kernels above write a verdict for each
    // slot they touch, and this replaces it wherever the metric prepass refused the molecule's bounds
    // outright. A molecule refused there was still minimised from a zeroed embedding above, so what it holds
    // is a conformer nothing asked for — the status is what stops it being read as one.
    if (molecule_count != 0) {
        // Before the stamp and before the half-precision narrowing: the polish kernel above rewrote
        // every coordinate after the pipeline kernel's own chirality gate, so this is the only place a
        // center it inverted can still be seen. One warp per slot, over the whole batch at once — every
        // bin writes coordinates at the same stride, so no per-bin offset walk is needed here.
        std::uint32_t const validate_grid = std::min<std::uint32_t>(
            molecule_count * conformer_count, static_cast<std::uint32_t>(8 * multiprocessor_count));
        validate_polished_stereo_kernel<<<validate_grid, warp_size_k, 0, stream>>>(
            regions.coordinates_fp32.data(), device_view.chiral_centers.data(), device_view.descriptors.data(),
            device_metadata, regions.status.data(), molecule_count, conformer_count, gpu_max_atoms);
        check_launch_(sink, "validate_polished_stereo_kernel launch");

        constexpr std::uint32_t stamp_block_k = 256;
        stamp_refused_molecules_kernel<<<divide_rounding_up(molecule_count, stamp_block_k), stamp_block_k, 0, stream>>>(
            regions.molecule_verdicts.data(), regions.status.data(), molecule_count, conformer_count);
        check_launch_(sink, "stamp_refused_molecules_kernel launch");
    }
}

/** Enqueue the half-precision conversion and the three device-to-host copies this wave reads back. */
static void enqueue_download_(device_wave_regions_t const &regions, host_wave_regions_t const &staged,
                              cudaStream_t stream, cuda_error_sink_t const &sink) {
    if (!regions.coordinates_fp32.empty()) {
        constexpr std::uint32_t convert_block_k = 256;
        std::uint32_t const element_count = static_cast<std::uint32_t>(regions.coordinates_fp32.size());
        std::uint32_t const convert_grid = divide_rounding_up(element_count, convert_block_k);
        convert_coords_fp32_to_fp16_kernel<<<convert_grid, convert_block_k, 0, stream>>>(
            regions.coordinates_fp32.data(), regions.coordinates_fp16.data(), element_count);
        check_launch_(sink, "convert_coords_fp32_to_fp16_kernel launch");
    }
    [[maybe_unused]] bool const copied_coordinates = download_from(
        staged.coordinates, custd::span<__half const> {regions.coordinates_fp16}, stream, sink);
    [[maybe_unused]] bool const copied_energies = download_from(
        staged.energies, custd::span<float const> {regions.energies}, stream, sink);
    [[maybe_unused]] bool const copied_status = download_from(
        staged.status, custd::span<std::uint8_t const> {regions.status}, stream, sink);
}

/**
 *  @brief Enqueue upload, device preprocessing, the metric prepass, the kernels and the download on
 *  one device's stream.
 *
 *  Returns as soon as the work is enqueued; the caller eventually hands the returned handle to
 *  `finalize_gpu_pipeline`. A handle with no work — an allocation that failed, or a device this wave
 *  gave no molecules — finalizes to nothing.
 *
 *  @param[inout] context Owns this device's stream, scratch and launch cache.
 *  @param[in] device_index Position of this device in `context.device_ids`.
 *  @param[in] batch This device's prepared pinned batch pool.
 *  @param[in] config Conformer count, stereo retry limit and seed.
 *  @param[in] host_max_atoms The caller's output stride.
 *  @param[in] outputs Caller destination and the permutation into it.
 *  @param[in] sink Error sink routing failures through the caller's logger.
 */
static gpu_pipeline_handle_t launch_gpu_pipeline(context_t &context, std::size_t device_index,
                                                 wave_slice_pool_t const &batch, config_t const &config,
                                                 std::uint32_t host_max_atoms, wave_outputs_t const &outputs,
                                                 cuda_error_sink_t sink) {
    int const device_id = context.device_ids[device_index];
    cudaStream_t const stream = context.streams[device_index];
    launch_scratch_t &launch_buffers = *context.launch_scratch[device_index];
    [[maybe_unused]] bool const bound = sink.check(cudaSetDevice(device_id), "cudaSetDevice(launch_gpu_pipeline)");

    // Preprocess already partitioned and atom-count sorted this device's molecules, so the kernels read
    // the uploaded metadata in place and no per-device sort happens here. The metadata region's extent
    // is this device's molecule count, which is the count every step below walks.
    wave_slice_const_view_t const host_view = batch.host_view();
    custd::span<etkdg_molecule_metadata_t const> const host_metadata = host_view.molecule_metadata;
    sf_assert_(host_metadata.size() == batch.num_molecules());

    // The GPU coordinate buffer and the pinned staging are sized to this wave's actual largest atom
    // count rather than the caller's cap; finalize zero-pads to the caller's stride per slot.
    std::uint32_t const gpu_max_atoms = std::max<std::uint32_t>(batch.observed_max_atoms, 1u);
    wave_pool_shape_t const shape = compute_wave_pool_shape_(wave_slice_bytes(batch.shape),
                                                             static_cast<std::uint32_t>(host_metadata.size()),
                                                             config.num_conformers, gpu_max_atoms);
    std::size_t const device_bytes = device_wave_bytes_(shape);
    std::size_t const host_bytes = host_wave_bytes_(shape);

    // Pools were pre-sized to the worst case at construction, so these are no-ops barring an input that
    // exceeded the worst-case probe. A grow here means an allocation fired on the path the reservation
    // exists to keep clear, which is a sizing bug rather than a memory shortage.
    reserve_outcome_t const device_outcome = launch_buffers.device_pool.try_reserve(device_bytes);
    reserve_outcome_t const host_outcome = launch_buffers.host_staging.try_reserve(host_bytes);
    sf_assert_(device_outcome != reserve_outcome_t::grown_k);
    sf_assert_(host_outcome != reserve_outcome_t::grown_k);
    if (device_outcome == reserve_outcome_t::failed_k || host_outcome == reserve_outcome_t::failed_k) {
        // Through the sink rather than the logger, because `cuda_error_count` is what the batch CLI
        // reads to park a shard and to set its exit code. Logging alone left this device's molecules
        // carrying the `invalid_smiles_k` pre-fill, so an out-of-memory device shipped as "these
        // SMILES failed to parse" and the process still returned zero.
        if (sink.log != nullptr)
            sink.log->error("launch_gpu_pipeline: pool allocation failed on device {} ({} device, {} host bytes)",
                            device_id, device_bytes, host_bytes);
        [[maybe_unused]] bool const reserved = sink.check(cudaErrorMemoryAllocation,
                                                          "launch_gpu_pipeline pool reserve");
        // The handle finalizes to nothing, so this is the only place these rows are ever written. The
        // pre-fill would otherwise name the molecules, and nothing about them is what refused: the same
        // input at a smaller wave fits. `prepare` has already translated the pool's indices into caller
        // rows, so the walk lands directly.
        for (std::uint32_t const caller_index : host_view.input_indices)
            for (std::uint32_t conformer = 0; conformer < config.num_conformers; ++conformer)
                outputs.status[static_cast<std::size_t>(caller_index) * config.num_conformers + conformer] =
                    conformer_status_t::batch_overflow_k;
        return {};
    }

    pool_cursor<std::byte> device_cursor {launch_buffers.device_pool.view()};
    device_wave_regions_t const regions = carve_device_wave_(shape, device_cursor);
    pool_cursor<std::byte> host_cursor {launch_buffers.host_staging.view()};
    host_wave_regions_t const staged = carve_host_wave_(shape, host_cursor);
    pool_cursor<std::byte> batch_cursor {regions.batch};
    wave_slice_view_t const device_view = carve_wave_slice(batch.shape, batch_cursor);
    phase_events_t const &events = launch_buffers.events;

    events.open(wave_phase_t::upload_k, stream);
    // One host-to-device copy of the uploaded prefix, out of this device's private pinned pool. Both
    // ends are the same walk's prefix, so neither side can name a length the other does not.
    [[maybe_unused]] bool const uploaded = upload_into(device_view.uploaded, host_view.uploaded, stream, sink);

    // `write_conformer_output_` fills only the first `num_atoms` slots of each axis, so the padding up
    // to the stride would reach `convert_coords_fp32_to_fp16_kernel` uninitialised — compute-sanitizer
    // counts about 16K such reads on the 50-conformer corpus, and reading them risks propagating a NaN.
    // The memset is bandwidth-trivial against the FIRE relaxation that follows.
    if (!regions.coordinates_fp32.empty()) {
        [[maybe_unused]] bool const zeroed = sink.check(
            cudaMemsetAsync(regions.coordinates_fp32.data(), 0, regions.coordinates_fp32.size_bytes(), stream),
            "memset device_coords_fp32 zero");
    }

    // The prepass writes a verdict only for the molecules it refuses, so the rest have to arrive saying
    // nothing. The pool is reused wave after wave, and without this a previous wave's refusal would be
    // stamped onto whichever molecule now occupies that slot.
    if (!regions.molecule_verdicts.empty()) {
        [[maybe_unused]] bool const primed = sink.check(
            cudaMemsetAsync(regions.molecule_verdicts.data(), static_cast<int>(conformer_status_t::success_k),
                            regions.molecule_verdicts.size_bytes(), stream),
            "memset device molecule verdicts");
    }

    // Everything from here to the metric prepass reads the uploaded descriptors and writes the regions
    // past the uploaded prefix, so the host ships the molecular graph and the GPU derives the force
    // field from it. One stream, in order, so each pass sees the previous one's output with no event.
    enqueue_device_preprocess_(device_view, batch.shape.torsion_stride, launch_buffers, batch.num_molecules(), stream,
                               sink);

    events.open(wave_phase_t::metric_k, stream);
    bool const embedded = build_metric_initial_coordinates_on_gpu(
        device_view.bounds_matrices.data(), device_view.molecule_metadata.data(), host_metadata, gpu_max_atoms,
        regions.metric_embedding.data(), regions.molecule_verdicts.data(), context.metric_scratch[device_index], sink,
        stream);

    // The prepass refuses a bin its scratch cannot hold, and it refuses before writing that bin's
    // embedding. The kernels below run either way, so without this they would relax the zeroed embedding
    // into a geometry that met no bounds and report it as a conformer. One verdict per molecule rather
    // than a bail-out, because the download and the timings behind it are already enqueued and the stamp
    // is the channel every other refusal already travels.
    if (!embedded && !regions.molecule_verdicts.empty()) {
        [[maybe_unused]] bool const refused = sink.check(
            cudaMemsetAsync(regions.molecule_verdicts.data(), static_cast<int>(conformer_status_t::batch_overflow_k),
                            regions.molecule_verdicts.size_bytes(), stream),
            "memset device molecule verdicts (batch overflow)");
    }

    events.open(wave_phase_t::kernels_k, stream);
    enqueue_binned_kernels_(device_view, host_metadata, regions, context.launch_configs[device_index], config,
                            gpu_max_atoms, stream, sink);

    events.open(wave_phase_t::download_k, stream);
    enqueue_download_(regions, staged, stream, sink);
    events.close(stream);

    return {.device_id = device_id,
            .stream = stream,
            .input_indices = host_view.input_indices,
            .conformer_count = config.num_conformers,
            .gpu_max_atoms = gpu_max_atoms,
            .host_max_atoms = host_max_atoms,
            .staged = staged,
            .outputs = outputs,
            .events = events,
            .sink = sink};
}

/**
 *  @brief Wait for an already-launched wave to finish and permute its results into the caller's arrays.
 *
 *  All three device-to-host transfers were enqueued by `launch_gpu_pipeline` into pinned staging, so
 *  this drains the stream, reads the phase timings back, and scatters staging into caller rows.
 *
 *  @param[inout] handle The wave's state; its `timings` are filled here.
 */
static void finalize_gpu_pipeline(gpu_pipeline_handle_t &handle) {
    [[maybe_unused]] bool const bound = handle.sink.check(cudaSetDevice(handle.device_id),
                                                          "cudaSetDevice(finalize_gpu_pipeline)");
    // Draining the stream, rather than syncing on the closing event, is correct whether or not that
    // event was ever recorded, and costs about 5 ms per wave per device. See ROADMAP.md under
    // "Optimizations tried and rejected" for why the event form is not used.
    [[maybe_unused]] bool const drained = handle.sink.check(cudaStreamSynchronize(handle.stream),
                                                            "cudaStreamSynchronize(finalize)");
    handle.events.elapsed_into(handle.timings);

    // Preprocess wrote this device's batch in atom-count ascending order, and `prepare` has already
    // translated `input_indices` into the caller's own rows, so a staged slot maps to its destination
    // with no further indirection.
    std::uint32_t const molecule_count = static_cast<std::uint32_t>(handle.input_indices.size());
    std::uint32_t const conformer_count = handle.conformer_count;
    conformer_slot_layout_t const destination_layout {handle.host_max_atoms, conformer_count};
    conformer_slot_layout_t const source_layout {handle.gpu_max_atoms, conformer_count};
    // The caller's coordinate buffer was zero-filled at the top of `generate_conformers`, so the halves
    // past `gpu_max_atoms` in each axis stay zero without an explicit memset here.
    //
    // This loop runs between the stream drain above and the next wave's launch, so every GPU is idle
    // for its duration — at 400K molecules and K=3 it is millions of scattered 256-byte copies.
    // `input_indices` is a permutation and the destination slot is a pure function of the source slot,
    // so no two iterations touch the same output bytes and the loop parallelises as written. Devices
    // are still finalized one at a time, so the fan-out here gives each device's permutation every core.
#pragma omp parallel for schedule(static)
    for (std::uint32_t sorted_index = 0; sorted_index < molecule_count; ++sorted_index) {
        std::uint32_t const caller_index = handle.input_indices[sorted_index];
        for (std::uint32_t conformer = 0; conformer < conformer_count; ++conformer) {
            std::size_t const source_slot = static_cast<std::size_t>(sorted_index) * conformer_count + conformer;
            std::size_t const destination_slot = static_cast<std::size_t>(caller_index) * conformer_count + conformer;
            // The demoted slot keeps the energy it was demoted for: the status says not to read it, and
            // the value is the only evidence of what the minimizer did.
            float const energy = handle.staged.energies[source_slot];
            handle.outputs.energies[destination_slot] = energy;
            handle.outputs.status[destination_slot] = conformer_status_of_(handle.staged.status[source_slot], energy);
            // Both sides are structure-of-arrays but at different strides, so each axis is copied into
            // its own section. `conformer_slot_layout_t` is the contract shared with the RDKit fallback
            // producer and the Parquet writer.
            __half *const destination = handle.outputs.coordinates.data() +
                                        destination_slot * destination_layout.slot_stride();
            __half const *const source = handle.staged.coordinates.data() + source_slot * source_layout.slot_stride();
            for (std::uint32_t axis = 0; axis < types::conformer_dimensions_k; ++axis)
                std::memcpy(destination + destination_layout.component_offset(axis, 0u),
                            source + source_layout.component_offset(axis, 0u), handle.gpu_max_atoms * sizeof(__half));
        }
    }
}

#pragma endregion Wave Launch

#pragma region Context

/**
 *  @brief Populate the per-device launch-configuration cache.
 *
 *  One-shot at construction: probes `cudaDeviceGetAttribute` and
 *  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for the two warp kernels across every atom-limited
 *  bin, so the per-wave hot path reads `device_launch_config_t` and never calls the driver again.
 */
static void populate_device_launch_config_(int device_id, device_launch_config_t &config,
                                           cuda_error_sink_t const &sink) {
    [[maybe_unused]] bool const probed = sink.check(
        cudaDeviceGetAttribute(&config.streaming_multiprocessor_count, cudaDevAttrMultiProcessorCount, device_id),
        "cudaDeviceGetAttribute(multiprocessor count)");
    for (std::size_t bin_index = 0; bin_index < atom_limited_bin_count_k; ++bin_index) {
        std::uint32_t const bin_max_atoms = atom_limited_bin_caps_k[bin_index];
        [[maybe_unused]] bool const etkdg_occupancy = sink.check(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&config.blocks_per_multiprocessor_etkdg[bin_index],
                                                          etkdg_pipeline_warp_kernel, warp_size_k,
                                                          compute_pipeline_shared_memory_bytes(bin_max_atoms)),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(etkdg)");
        [[maybe_unused]] bool const mmff_occupancy = sink.check(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&config.blocks_per_multiprocessor_mmff[bin_index],
                                                          mmff_polish_warp_kernel, warp_size_k,
                                                          compute_mmff_polish_shared_memory_bytes(bin_max_atoms)),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(mmff)");
    }
}

context_t::context_t(custd::span<int const> devices) {
    if (devices.empty()) {
        int count = 0;
        cudaGetDeviceCount(&count);
        for (int device_index = 0; device_index < count; ++device_index) device_ids.push_back(device_index);
    }
    else { device_ids.assign(devices.begin(), devices.end()); }

    // `batch_slots_per_device` holds `max_devices_k` slots per ring position and reaches
    // `prepare_per_device_batches_into` as a dynamic-extent span, so the callee indexes it by the device
    // count alone. An unclamped list writes past the array, and the default constructor enumerates every
    // visible GPU, so nothing unusual on the caller's side is needed to reach it.
    static_assert(max_devices_k <= smallfield::preprocess::distribute_scratch_t::max_devices_k,
                  "preprocess indexes its per-device scratch by the count this API admits");
    if (device_ids.size() > max_devices_k) {
        this->logger.error("context_t: {} devices visible, using the first {}; raise max_devices_k to use the rest",
                           device_ids.size(), max_devices_k);
        device_ids.resize(max_devices_k);
    }

    // Every container below is filled only on the success path of the bring-up loop, so a device that
    // refuses is absent rather than present-and-null. `device_ids` is rewritten from the survivors at the
    // end, which is what stops preprocess partitioning molecules across a device the launch cannot reach.
    // Reserved rather than resized: the loop appends, and reserving up front means no regrow moves a
    // half-built entry.
    std::vector<int> const requested_device_ids = device_ids;
    std::size_t const requested_count = requested_device_ids.size();
    device_ids.clear();
    device_ids.reserve(requested_count);
    streams.reserve(requested_count);
    metric_scratch.reserve(requested_count);
    launch_scratch.reserve(requested_count);
    launch_configs.reserve(requested_count);

    // The MMFF94 parameter tables and the compiled CSD torsion patterns are process-wide constants
    // extracted once from RDKit; each device gets its own arena so no kernel reaches across a
    // device boundary for a parameter lookup.
    mmff94_tables_host_t mmff_tables_host;
    extract_mmff94_tables_from_rdkit(mmff_tables_host);
    csd_torsion_table_host_t csd_torsion_host;
    build_csd_torsion_table(csd_torsion_host);
    cuda_error_sink_t const sink {&this->logger, &this->cuda_error_count};

    // `cudaFuncAttributeMaxDynamicSharedMemorySize` on the embed kernel stays in effect for the lifetime
    // of each device's primary context. At the `max_atoms_per_molecule_k` cap that kernel needs about
    // 148.5 KB of shared memory, above the 48 KB default.
    std::size_t const max_embed_shared_bytes = smallfield::cuda::embed_shared_bytes(max_atoms_per_molecule_k);

    for (int const device_id : requested_device_ids) {
        if (!sink.check(cudaSetDevice(device_id), "cudaSetDevice(context_t)")) continue;
        cudaStream_t stream = nullptr;
        if (!sink.check(cudaStreamCreate(&stream), "cudaStreamCreate")) continue;

        auto scratch_owner = std::make_unique<launch_scratch_t>(stream);
        launch_scratch_t &scratch = *scratch_owner;
        upload_mmff94_tables_to_device(mmff_tables_host, scratch.mmff_tables, stream, sink);
        upload_csd_torsion_table_to_device(csd_torsion_host, scratch.csd_torsion_table, stream, sink);
        scratch.events.create(sink);
        [[maybe_unused]] bool const shared_raised = sink.check(
            cudaFuncSetAttribute(smallfield::cuda::embed_4d_power_iter_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(max_embed_shared_bytes)),
            "cudaFuncSetAttribute(embed_4d_power_iter_kernel)");
        if (scratch.mmff_staging.try_reserve(mmff_staging_bytes_k) == reserve_outcome_t::failed_k)
            this->logger.error("context_t: MMFF staging allocation ({} bytes) failed on device {}",
                               mmff_staging_bytes_k, device_id);

        device_launch_config_t launch_config {};
        populate_device_launch_config_(device_id, launch_config, sink);
        // The metric scratch is reserved lazily by `ensure_metric_scratch`, so it only needs its stream now.
        metric_scratch_t scratch_for_metric;
        scratch_for_metric.group_indices = device_indices_t {device_allocator_t {stream}};

        device_ids.push_back(device_id);
        streams.push_back(stream);
        launch_scratch.push_back(std::move(scratch_owner));
        launch_configs.push_back(launch_config);
        metric_scratch.push_back(std::move(scratch_for_metric));
    }

    if (device_ids.size() != requested_count)
        this->logger.error("context_t: {} of {} devices came up; the rest are not used", device_ids.size(),
                           requested_count);
    std::size_t const device_count = device_ids.size();

    // Pre-allocate every per-device wave pool so the hot path never grows: a grow discards the
    // buffer's contents whatever it costs, so `launch_gpu_pipeline` asserts it finds `unchanged_k`.
    // Sizing runs after the loop above, so the VRAM census sees the tables and the staging already
    // resident and does not promise a wave the free pool cannot hold.
    std::uint32_t const worst_case_wave_size = compute_wave_size_(device_ids, max_atoms_per_molecule_k,
                                                                  max_conformer_count_k, sink);
    per_device_reserved_molecules = reserve_molecules_per_device(worst_case_wave_size,
                                                                 static_cast<std::uint32_t>(device_count));
    wave_pool_shape_t const worst_case_shape = compute_wave_pool_shape_(
        estimate_per_molecule_batch_bytes(max_atoms_per_molecule_k) * per_device_reserved_molecules,
        per_device_reserved_molecules, max_conformer_count_k, max_atoms_per_molecule_k);
    std::size_t const worst_case_device_bytes = device_wave_bytes_(worst_case_shape);
    std::size_t const worst_case_host_bytes = host_wave_bytes_(worst_case_shape);

    for (std::size_t device_index = 0; device_index < device_count; ++device_index) {
        int const device_id = device_ids[device_index];
        if (!sink.check(cudaSetDevice(device_id), "cudaSetDevice(context_t reserve)")) continue;
        // Probes at `per_device_reserved_molecules` molecules of `max_atoms_per_molecule_k` atoms each.
        ensure_metric_scratch(*this, device_index, per_device_reserved_molecules, max_atoms_per_molecule_k);
        // This is the reservation the launch path's own calls are expected to find already sufficient,
        // so a failure here is worth naming now rather than as a null pointer several waves later.
        launch_scratch_t &scratch = *launch_scratch[device_index];
        if (scratch.device_pool.try_reserve(worst_case_device_bytes) == reserve_outcome_t::failed_k ||
            scratch.host_staging.try_reserve(worst_case_host_bytes) == reserve_outcome_t::failed_k)
            this->logger.error("context_t: pool reservation failed on device {} ({} device, {} host bytes)", device_id,
                               worst_case_device_bytes, worst_case_host_bytes);
    }
    // The pinned batch slots are deliberately left unreserved. `prepare_per_device_batches_into`
    // sizes each one from the wave's own uploaded prefix, and preprocess partitions atom-count
    // descending, so the first wave carries the largest molecules and every slot reaches its peak
    // capacity on that wave; later waves find the buffer already large enough. Reserving here
    // instead would have to assume the `max_atoms_per_molecule_k` cap, which costs far more per
    // molecule than a corpus like GDB13 actually stages.
}

context_t::~context_t() {
    // Every entry is whole: the constructor appends a device only once its stream and scratch exist, so
    // there is no half-built slot to test for here. The scratch is released before its stream, because
    // `cudaFreeAsync` against a destroyed stream is undefined.
    for (std::size_t device_index = 0; device_index < device_ids.size(); ++device_index) {
        cudaSetDevice(device_ids[device_index]);
        launch_scratch[device_index]->events.destroy();
        launch_scratch[device_index].reset();
        cudaStreamDestroy(streams[device_index]);
    }
}

#pragma endregion Context

#pragma region Wave Dispatch

/**
 *  @brief Half-open `[begin, end)` slice of the sorted input that one `embedding-wave` covers.
 *
 *  Indices are positions in the SMILES-length-sorted view, not in the caller's array; `begin`
 *  doubles as the wave base that `wave_outputs_t` adds before consulting the sort permutation.
 */
struct wave_range_t {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;

    [[nodiscard]] std::uint32_t size() const noexcept { return end - begin; }
};

/**
 *  @brief Build a descending-by-SMILES-length permutation of the input and write the reordered
 *  SMILES views into @p sorted_smiles_storage.
 *
 *  Putting the smallest molecules in the last `embedding-wave` shortens the tail: the last wave's GPU
 *  work is the only fully unhidden one, so making it the cheapest shrinks overall wall.
 *  `finalize_gpu_pipeline` folds this permutation into the same reorder that undoes the per-GPU
 *  atom-count sort.
 */
static void sort_input_by_smiles_length_(custd::span<std::string_view const> smiles,
                                         std::vector<std::uint32_t> &sort_perm,
                                         std::vector<std::string_view> &sorted_smiles_storage) {
    std::uint32_t const num_input = static_cast<std::uint32_t>(smiles.size());
    sort_perm.resize(num_input);
    std::iota(sort_perm.begin(), sort_perm.end(), 0u);
    std::sort(sort_perm.begin(), sort_perm.end(),
              [smiles](std::uint32_t left, std::uint32_t right) { return smiles[left].size() > smiles[right].size(); });
    sorted_smiles_storage.resize(num_input);
    for (std::uint32_t position = 0; position < num_input; ++position)
        sorted_smiles_storage[position] = smiles[sort_perm[position]];
}

/**
 *  @brief Molecules per wave, and with it how many waves the run takes.
 *
 *  A caller's explicit `wave_size` is bounded by @p wave_size_ceiling exactly as the auto-sized one is,
 *  which is what keeps preprocess's `batch_overflow_k` tail empty: past that bound a molecule's rows
 *  would be packed at a wrapped 32-bit region offset, so the tail is refused rather than aliased.
 *
 *  An auto-sized wave may additionally never exceed what the constructor reserved, because going past it
 *  fires a grow from inside the per-device fan-out that discards a pool the wave is mid-way through
 *  filling. The two sizings disagree without that ceiling: the constructor assumes `max_conformer_count_k`
 *  while this uses the caller's K, which at K=3 halves the per-molecule estimate.
 */
static std::uint32_t choose_wave_size_(context_t const &context, config_t const &config,
                                       std::uint32_t max_atoms_per_molecule, std::uint32_t wave_size_ceiling,
                                       cuda_error_sink_t const &sink) {
    if (config.wave_size != 0) return std::min(config.wave_size, wave_size_ceiling);
    std::uint64_t const auto_wave = compute_wave_size_(context.device_ids, max_atoms_per_molecule,
                                                       config.num_conformers, sink);
    std::uint64_t const reserved_ceiling = static_cast<std::uint64_t>(context.per_device_reserved_molecules) *
                                           std::max<std::size_t>(1u, context.device_ids.size());
    if (reserved_ceiling > 0 && auto_wave > reserved_ceiling) return static_cast<std::uint32_t>(reserved_ceiling);
    return static_cast<std::uint32_t>(auto_wave);
}

/**
 *  @brief Cut @p num_input positions into the waves the dispatch loop walks.
 *
 *  An input that fits within half again the target rides in a single wave, which is what the
 *  un-pipelined dispatch amounted to: with one wave there is nothing to overlap with, so the ring
 *  never turns and the loop below degenerates to prepare, launch, finalize.
 *
 *  @p wave_size_ceiling bounds that shortcut as well as the target. `choose_wave_size_` already holds
 *  its answer under what a 32-bit per-molecule region offset addresses, and half again a bounded target
 *  is not bounded — the shortcut is the one path that can hand preprocess a slice it has to refuse.
 */
static std::vector<wave_range_t> split_into_waves_(std::uint32_t num_input, std::uint32_t wave_size_target,
                                                   std::uint32_t wave_size_ceiling) {
    std::vector<wave_range_t> waves;
    if (num_input == 0) return waves;
    std::uint32_t const single_wave_limit = std::min(wave_size_target * 3u / 2u, wave_size_ceiling);
    std::uint32_t const stride = num_input <= single_wave_limit ? num_input
                                                                : std::max<std::uint32_t>(wave_size_target, 1u);
    waves.reserve(divide_rounding_up(num_input, stride));
    for (std::uint32_t begin = 0; begin < num_input; begin += stride)
        waves.push_back({.begin = begin, .end = std::min(begin + stride, num_input)});
    return waves;
}

/**
 *  @brief The three steps one `embedding-wave` takes, and the state all three share.
 *
 *  A receiver rather than a cluster of capture-everything lambdas: each method names the wave it acts
 *  on, and the loop in `generate_conformers` reads as prepare, finalize the previous, launch.
 */
struct wave_dispatcher_t {
    context_t &context;
    config_t const &config;
    /** Input in descending-SMILES-length order; each wave takes a contiguous slice. */
    custd::span<std::string_view const> sorted_smiles;
    /** The caller's output stride, and the atom cap preprocess refuses molecules against. */
    std::uint32_t host_max_atoms = 0;
    /** Caller destination and sort permutation; `wave_base` is filled per wave. */
    wave_outputs_t outputs {};
    molecule_readback_t readback {};
    wall_clock_t const &clock;
    cuda_error_sink_t sink {};
    /** Grow-only across waves; `prepare` refills it from each slice. */
    std::vector<rejected_mol_t> rejected {};
    /** Preprocess's per-worker and partition buffers. One dispatcher owns one, which is what lets two
     *  `context_t` objects prepare waves on two threads; grow-only across this run's waves. */
    batch_prep_scratch_t prep_scratch {};

    [[nodiscard]] std::uint32_t device_count() const noexcept {
        return static_cast<std::uint32_t>(context.device_ids.size());
    }

    /** Where this wave's results belong in the caller's arrays. */
    [[nodiscard]] wave_outputs_t outputs_for(wave_range_t range) const noexcept {
        wave_outputs_t wave_outputs = outputs;
        wave_outputs.wave_base = range.begin;
        return wave_outputs;
    }

    /**
     *  @brief Parse and pack one slice into this wave's ring slot, and stamp what preprocess refused.
     *
     *  Preprocess partitions by atom count descending and writes each device's pool atom-count
     *  ascending, so the kernels can read the batch metadata directly with no per-device sort. The
     *  molecules it rejects — unsupported element, too many atoms, invalid SMILES — never reach the
     *  GPU pools; their status is stamped here so the `fallback-role` picks them up caller-side.
     */
    void prepare(wave_range_t range, std::uint32_t wave_index, wave_record_t &record) {
        SF_NVTX_RANGE_COLORED("prep: rdkit preprocess", 0x8E44AD);
        scoped_span_ms_t const timed {clock, record.prepare};
        custd::span<wave_slice_pool_t> const slot = context.wave_slot(wave_index);
        prepare_per_device_batches_into(sorted_smiles.subspan(range.begin, range.size()), device_count(),
                                        host_max_atoms, slot, prep_scratch, &rejected);

        wave_outputs_t const wave_outputs = outputs_for(range);

        // Preprocess numbers its molecules by their position in the slice it was handed, and that slice is a
        // wave of the length-sorted order. Translate to the caller's own rows here, once, so everything
        // downstream reads one fact rather than two: the finalize scatter indexes the caller's buffers
        // directly, and the kernel seeds Philox from a row that does not move when the batch changes.
        for (wave_slice_pool_t &batch : slot) {
            custd::span<std::uint32_t> const input_indices = batch.host_view().input_indices;
            for (std::uint32_t &index : input_indices) index = wave_outputs.caller_index_of(index);
        }
        for (rejected_mol_t const &entry : rejected) {
            std::uint32_t const caller_index = wave_outputs.caller_index_of(entry.slice_local_index);
            for (std::uint32_t conformer = 0; conformer < config.num_conformers; ++conformer)
                wave_outputs.status[static_cast<std::size_t>(caller_index) * config.num_conformers + conformer] =
                    entry.reason;
        }
        if (!readback.wanted()) return;
        // Molecules preprocess refused keep the zero written before the first wave.
        for (wave_slice_pool_t const &batch : slot) {
            wave_slice_const_view_t const view = batch.host_view();
            custd::span<etkdg_molecule_metadata_t const> const metadata = view.molecule_metadata;
            custd::span<std::uint32_t const> const input_indices = view.input_indices;
            for (std::uint32_t molecule = 0; molecule < batch.num_molecules(); ++molecule)
                readback.write(input_indices[molecule], metadata[molecule]);
        }
    }

    /**
     *  @brief Fan out across the devices, each enqueueing its own upload, compute and download chain.
     *
     *  Every call inside is asynchronous against that device's own stream and reads only from that
     *  device's pinned pool, so concurrent launches share neither a pinned source nor a driver lock.
     */
    void launch(wave_range_t range, std::uint32_t wave_index, wave_record_t &record,
                std::vector<gpu_pipeline_handle_t> &handles) {
        SF_NVTX_RANGE_COLORED("launch: gpu dispatch", 0x2E86C1);
        scoped_span_ms_t const timed {clock, record.launch};
        custd::span<wave_slice_pool_t> const slot = context.wave_slot(wave_index);
        handles.assign(device_count(), gpu_pipeline_handle_t {});

        // Small-molecule corpora pack most of a wave into a single atom-count bin, which can exceed the
        // metric scratch sized at construction from a wave-size estimate; the prepass refuses an
        // oversized bin. Growing here is a no-op after the first such wave and costs a uint32 per
        // molecule, and doing it serially keeps the one-time allocation out of the launch fan-out.
        bool any_molecules = false;
        for (std::uint32_t device_index = 0; device_index < device_count(); ++device_index) {
            if (slot[device_index].num_molecules() == 0) continue;
            any_molecules = true;
            ensure_metric_scratch(context, device_index, slot[device_index].num_molecules(), max_atoms_per_molecule_k);
        }
        if (!any_molecules) return;

        wave_outputs_t const wave_outputs = outputs_for(range);
#pragma omp parallel for num_threads(device_count()) schedule(static, 1)
        for (std::uint32_t device_index = 0; device_index < device_count(); ++device_index) {
            if (slot[device_index].num_molecules() == 0) continue;
            handles[device_index] = launch_gpu_pipeline(context, device_index, slot[device_index], config,
                                                        host_max_atoms, wave_outputs, sink);
        }
    }

    /**
     *  @brief Drain every device's stream in turn and permute the wave's results out of staging.
     *
     *  The first drain blocks until that stream finishes; the others started within microseconds of it
     *  and carry equal work, so they return almost immediately.
     */
    void finalize(std::vector<gpu_pipeline_handle_t> &handles, wave_record_t &record) {
        scoped_span_ms_t const timed {clock, record.finalize};
        record.per_device_gpu.assign(device_count(), gpu_phase_ms_t {});
        for (std::uint32_t device_index = 0; device_index < handles.size(); ++device_index) {
            if (!handles[device_index].has_work()) continue;
            finalize_gpu_pipeline(handles[device_index]);
            record.per_device_gpu[device_index] = handles[device_index].timings;
        }
    }
};

/** Whether any conformer in the already-permuted status buffer came back successful. */
static bool any_conformer_succeeded_(custd::span<conformer_status_t const> status) noexcept {
    for (conformer_status_t const entry : status)
        if (entry == conformer_status_t::success_k) return true;
    return false;
}

conformer_status_t context_t::generate_conformers(
    custd::span<std::string_view const> smiles, config_t const &config, std::uint32_t max_atoms_per_molecule,
    custd::span<__half> coordinates_out, custd::span<float> energies_out, custd::span<conformer_status_t> status_out,
    custd::span<std::uint8_t> num_atoms_out, custd::span<std::uint16_t> num_heavy_atoms_out,
    custd::span<std::uint16_t> num_bonds_out, custd::span<float> molecular_weight_out) {

    cuda_error_sink_t const sink {&this->logger, &this->cuda_error_count};
    molecule_readback_t const readback {.num_atoms = num_atoms_out,
                                        .num_heavy_atoms = num_heavy_atoms_out,
                                        .num_bonds = num_bonds_out,
                                        .molecular_weight = molecular_weight_out};

    // Every slot a wave does not reach keeps these defaults, so a molecule preprocess refused reads as
    // an invalid SMILES with zeroed coordinates rather than as whatever the caller's buffer held.
    std::fill_n(status_out.begin(), status_out.size(), conformer_status_t::invalid_smiles_k);
    std::fill_n(energies_out.begin(), energies_out.size(), 0.0f);
    std::fill_n(coordinates_out.begin(), coordinates_out.size(), __float2half_rn(0.0f));
    readback.clear();

    std::vector<std::uint32_t> sort_perm;
    std::vector<std::string_view> sorted_smiles;
    sort_input_by_smiles_length_(smiles, sort_perm, sorted_smiles);

    std::uint32_t const num_input = static_cast<std::uint32_t>(smiles.size());
    // Molecules one batch can address at this atom cap, which bounds the wave both times it is decided:
    // once when the size is chosen and once when the input is cut, because the single-wave shortcut takes
    // half again the chosen size.
    std::uint32_t const addressable_molecules = smallfield::preprocess::max_molecules_for_atom_capacity(
        max_atoms_per_molecule);
    std::uint32_t const wave_size_target = choose_wave_size_(*this, config, max_atoms_per_molecule,
                                                             addressable_molecules, sink);
    std::vector<wave_range_t> const waves = split_into_waves_(num_input, wave_size_target, addressable_molecules);
    this->logger.trace("generate_conformers: n={} K={} wave={} waves={}", num_input, config.num_conformers,
                       wave_size_target, waves.size());
    if (waves.empty()) return conformer_status_t::mmff_unconverged_k;

    // The constructor pre-grew the device pool, the pinned staging and the metric scratch to the
    // worst-case wave, so no mid-run grow fires on the hot path and every per-launch `try_reserve`
    // returns `unchanged_k` at steady state. The pinned batch slots are the exception: they size
    // themselves from the first wave, which is the largest because preprocess sorts atom-count
    // descending.
    wall_clock_t const clock;
    std::vector<wave_record_t> records(waves.size());
    for (std::uint32_t wave = 0; wave < waves.size(); ++wave)
        records[wave] = {.wave_index = wave, .num_molecules = waves[wave].size()};

    wave_dispatcher_t dispatcher {.context = *this,
                                  .config = config,
                                  .sorted_smiles = sorted_smiles,
                                  .host_max_atoms = max_atoms_per_molecule,
                                  .outputs = {.coordinates = coordinates_out,
                                              .energies = energies_out,
                                              .status = status_out,
                                              .sort_perm = sort_perm},
                                  .readback = readback,
                                  .clock = clock,
                                  .sink = sink};

    // One wave is GPU-resident at a time: N-1 is finalized before N is launched, so a single handle
    // vector spans exactly the window it describes. The pinned ring is deeper than this for a different
    // reason — wave N-1's host-to-device copy is still reading its slot while `prepare` fills wave N's —
    // and sizing the handles by that depth implied a second resident wave this loop has never issued.
    std::vector<gpu_pipeline_handle_t> in_flight;
    dispatcher.prepare(waves[0], 0, records[0]);
    dispatcher.launch(waves[0], 0, records[0], in_flight);
    for (std::uint32_t wave = 1; wave < waves.size(); ++wave) {
        dispatcher.prepare(waves[wave], wave, records[wave]);
        dispatcher.finalize(in_flight, records[wave - 1]);
        dispatcher.launch(waves[wave], wave, records[wave], in_flight);
    }
    dispatcher.finalize(in_flight, records.back());

    print_wave_timeline_(this->logger, records, clock.elapsed_ms());

    // Every permutation already happened inside `finalize_gpu_pipeline`, which folds each wave's
    // per-GPU sort and the outer SMILES-length sort into one reorder straight into caller rows. What
    // remains is a scan of the permuted status buffer for the aggregate return code.
    return any_conformer_succeeded_(status_out) ? conformer_status_t::success_k
                                                : conformer_status_t::mmff_unconverged_k;
}

#pragma endregion Wave Dispatch

} // namespace smallfield::api
