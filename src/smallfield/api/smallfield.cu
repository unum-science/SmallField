/**
 *  @file src/smallfield/api/smallfield.cu
 *  @brief Library implementation of the SmallField public API and pybind11 bindings.
 *
 *  Implements `context_t`, multi-GPU CUDA stream/cuSOLVER management, and `generate_conformers()`, CPU
 *  preprocessing, wave-based GPU dispatch, half-precision output. Exposes a minimal Python module at the
 *  bottom.
 */

#include <cstring> // `std::memcpy`

#include <algorithm>     // `std::max`, `std::min`
#include <array>         // `std::array`
#include <chrono>        // `std::chrono::high_resolution_clock`, `std::chrono::duration`
#include <string>        // `std::string`
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

#if defined(SMALLFIELD_BUILD_PYTHON)
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

#include "smallfield/api/wave_slice_pool.hpp"
#include "smallfield/cuda/etkdg_pipeline.cuh"
#include "smallfield/cuda_error.hpp"
#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/cuda/bond_distance_kernel.cuh"
#include "smallfield/cuda/csd_torsion_kernel.cuh"
#include "smallfield/cuda/preprocess_kernel.cuh"
#include "smallfield/cuda/metric_embedding_prepass.cuh"
#include "smallfield/mmff94/mmff94_extract_device.cuh"
#include "smallfield/mmff94/mmff94_typer_device.cuh"
#include "smallfield/parameters/mmff94_tables_device.cuh"
#include "smallfield/parameters/mmff94_tables_rdkit.hpp"
#include "smallfield/api/smallfield.hpp"
#include "smallfield/nvtx.hpp"
#include "smallfield/utilities.hpp"

namespace smallfield::api {
using smallfield::cuda::build_csd_torsion_table;
using smallfield::cuda::build_metric_initial_coordinates_on_gpu;
using smallfield::cuda::compute_mmff_polish_shared_memory_bytes;
using smallfield::cuda::compute_pipeline_shared_memory_bytes;
using smallfield::cuda::convert_coords_fp32_to_fp16_kernel;
using smallfield::cuda::csd_torsion_table_device_t;
using smallfield::cuda::csd_torsion_table_host_t;
using smallfield::cuda::etkdg_pipeline_warp_kernel;
using smallfield::cuda::launch_bond_distance_bfs;
using smallfield::cuda::launch_csd_torsion_match;
using smallfield::cuda::launch_etk_build_csr;
using smallfield::cuda::launch_mmff94_apply_typing_patch;
using smallfield::cuda::mmff_polish_warp_kernel;
using smallfield::cuda::upload_csd_torsion_table_to_device;
using smallfield::mmff94::launch_mmff94_build_csr;
using smallfield::mmff94::launch_mmff94_compact_terms;
using smallfield::mmff94::launch_mmff94_extract_terms;
using smallfield::mmff94::launch_mmff94_type_and_charge;
using smallfield::mmff94::mmff94_per_mol_counts_t;
using smallfield::parameters::extract_mmff94_tables_from_rdkit;
using smallfield::parameters::mmff94_tables_device_t;
using smallfield::parameters::mmff94_tables_host_t;
using smallfield::parameters::upload_mmff94_tables_to_device;
using smallfield::preprocess::prepare_per_device_batches_into;
using smallfield::types::atom_limited_bin_caps_k;
using smallfield::types::atom_limited_bin_count_k;
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

using smallfield::cuda_error_sink_t;

/**
 *  @brief Rough upper bound on the per-molecule bytes a wave stages in pinned host memory.
 *
 *  Covers the uploaded prefix only — the bounds matrix, the descriptor, the chiral centres, the
 *  1-2 / 1-3 constraints and the per-molecule bases and metadata — because the host buffer stops at
 *  `wave_slice_layout_t::bytes_uploaded` and never addresses the device-only regions. At the
 *  `max_atoms_per_molecule_k` cap the bounds matrix dominates; the slack term covers the constraint
 *  list, whose length scales with the bond and neighbour-pair counts rather than with the atom count
 *  alone.
 *
 *  @param[in] max_atoms Atom-count cap the wave is sized against.
 *  @return Bytes to reserve per molecule in a pinned host slot.
 */
static inline std::size_t estimate_per_molecule_uploaded_bytes(std::uint32_t max_atoms) noexcept {
    std::size_t const atom_count = max_atoms;
    std::size_t const bounds_bytes = atom_count * atom_count * sizeof(float);
    std::size_t const descriptor_bytes = sizeof(molecule_descriptor_t);
    std::size_t const per_atom_slack = atom_count * 128u;
    std::size_t const per_mol_fixed = 1024u;
    return bounds_bytes + descriptor_bytes + per_atom_slack + per_mol_fixed;
}

/**
 *  @brief Rough upper bound on the per-molecule batch-pool footprint in a wave.
 *
 *  Used by the wave sizer to divide a VRAM budget into molecules. It has to cover both sides of
 *  the pool: the uploaded prefix `estimate_per_molecule_uploaded_bytes` accounts for, and the
 *  device-only regions the preprocessing kernels fill — hop-count matrix, ETK torsions and their
 *  CSR, every MMFF term array and its CSR. The term reservations are the capacity bounds
 *  `compute_device_capacities` hands out.
 *
 *  Sizing a pinned host slot with this figure over-reserves, since the device-only regions never
 *  reach the host; use the uploaded bound for that.
 *
 *  @param[in] max_atoms Atom-count cap the wave is sized against.
 *  @return Bytes to reserve per molecule across the device pool and the uploaded prefix together.
 */
static inline std::size_t estimate_per_molecule_batch_bytes(std::uint32_t max_atoms) noexcept {
    std::size_t const atom_count = max_atoms;
    std::size_t const bond_distance_bytes = atom_count * atom_count * sizeof(std::uint8_t);
    // A molecule with explicit hydrogens carries roughly (atom count + 16) bonds; the term
    // reservations scale with that bond count and with the neighbour-pair count around each atom.
    std::size_t const bonds = atom_count + 16u;
    std::size_t const torsion_bytes = bonds * sizeof(etk_torsion_descriptor_t);
    std::size_t const mmff_term_bytes = 4u * bonds *
                                        (sizeof(mmff94_bond_t) + sizeof(mmff94_angle_t) + sizeof(mmff94_torsion_t) +
                                         sizeof(mmff94_oop_t) + sizeof(mmff94_stretchbend_t));
    std::size_t const csr_bytes = 32u * bonds * sizeof(mmff94_atom_interaction_t);
    return estimate_per_molecule_uploaded_bytes(max_atoms) + bond_distance_bytes + torsion_bytes + mmff_term_bytes +
           csr_bytes;
}

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

/** Byte offsets of the MMFF extractor's fixed-stride staging arrays within one buffer. */
struct mmff_staging_layout_t {
    std::size_t offset_bonds;
    std::size_t offset_angles;
    std::size_t torsions_byte_offset;
    std::size_t offset_oops;
    std::size_t offset_stretchbends;
    std::size_t offset_interactions;
    std::size_t offset_counts;
    std::size_t bytes_total;
};

/** Place the staging arrays for `mmff_extract_window_k` molecules back to back. */
static mmff_staging_layout_t compute_mmff_staging_layout_() noexcept {
    std::size_t const window = mmff_extract_window_k;
    mmff_staging_layout_t layout {};
    std::size_t running = 0;
    auto place = [&](std::size_t bytes) {
        std::size_t const here = align_up_(running, 256);
        running = here + bytes;
        return here;
    };
    layout.offset_bonds = place(window * max_mmff_bonds_per_mol_k * sizeof(mmff94_bond_t));
    layout.offset_angles = place(window * max_mmff_angles_per_mol_k * sizeof(mmff94_angle_t));
    layout.torsions_byte_offset = place(window * max_mmff_torsions_per_mol_k * sizeof(mmff94_torsion_t));
    layout.offset_oops = place(window * max_mmff_oops_per_mol_k * sizeof(mmff94_oop_t));
    layout.offset_stretchbends = place(window * max_mmff_stretchbends_per_mol_k * sizeof(mmff94_stretchbend_t));
    layout.offset_interactions = place(window * max_mmff_interactions_per_mol_k * sizeof(mmff94_atom_interaction_t));
    layout.offset_counts = place(window * sizeof(mmff94_per_mol_counts_t));
    layout.bytes_total = running;
    return layout;
}

/**
 *  @brief Persistent per-device scratch for the kernel launch path.
 *
 *  `device_pool` holds the batch-input mirror + per-launch outputs + scratch;
 *  `host_staging` holds the sorted-metadata subset (CPU → GPU) plus pinned
 *  readback regions for coords / energies / status (GPU → CPU). Both are
 *  grow-only and reused across waves so the single-threaded dispatcher
 *  avoids per-launch allocation churn.
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
    // Persistent CUDA events for per-phase timing. Created once per device on
    // first launch and reused across every wave, avoiding a race between
    // cudaEventDestroy on one stream and cudaEventCreate from the next launch's
    // omp parallel-for thread. cudaEventBlockingSync avoids spin-waiting inside
    // cudaStreamSynchronize, which matters when many streams share host CPUs.
    cudaEvent_t event_upload_start = nullptr;
    cudaEvent_t event_metric_start = nullptr;
    cudaEvent_t event_kernels_start = nullptr;
    cudaEvent_t event_download_start = nullptr;
    cudaEvent_t event_end = nullptr;
};

/**
 *  @brief Grow the per-device metric scratch pool to the given dimensions; no-op on shrink.
 *
 *  Definition lives below. `context_t` "context_t's" ctor pre-warms the
 *  per-device metric scratch at construction time so the first user call
 *  doesn't pay a `cudaMalloc` on the hot path.
 */
void ensure_metric_scratch(context_t &context, std::size_t device_index, std::size_t max_group_size,
                           std::uint32_t max_atoms);

/**
 *  @brief Offsets + sizes for every region carved from `launch_scratch_t`'s
 *  device pool and pinned host staging, for one `embedding-step`.
 *
 *  Device pool regions, in order:
 *    - batch-input mirror — target of the single H2D
 *    - sorted metadata subset — tiny H2D from the persistent host scratch
 *    - coords_fp32 — kernel output, fp32
 *    - coords_fp16 — post-convert, D2H source; halves DMA bandwidth
 *    - energies_out — fp32
 *    - status_out — u8
 *    - 4D metric-embedding scratch — GPU-only, consumed by the prepass
 *
 *  Host staging regions, in order:
 *    - coords_fp16 / energies / status — pinned D2H destinations
 */
struct pipeline_pool_layout_t {
    std::size_t device_off_batch;
    std::size_t device_off_sorted_meta;
    std::size_t device_off_coords_fp32;
    std::size_t device_off_coords_fp16;
    std::size_t device_off_energies;
    std::size_t device_off_status;
    std::size_t device_off_initial;
    std::size_t device_pool_bytes;

    std::size_t host_off_coords;
    std::size_t host_off_energies;
    std::size_t host_off_status;
    std::size_t host_pool_bytes;
};

/** Compute every offset and size for the launch pool given this dispatch's shape. */
static pipeline_pool_layout_t compute_pipeline_pool_layout_(std::size_t batch_bytes_total,
                                                            std::uint32_t per_device_molecule_count,
                                                            std::uint32_t conformer_count,
                                                            std::uint32_t max_atoms_per_molecule) {
    auto const sizeof_meta = sizeof(etkdg_molecule_metadata_t);
    std::size_t const total_conformer_slots = static_cast<std::size_t>(per_device_molecule_count) * conformer_count;
    std::size_t const total_coords = total_conformer_slots * max_atoms_per_molecule * 3;

    pipeline_pool_layout_t layout {};
    std::size_t device_offset = 0;
    layout.device_off_batch = align_up_(device_offset, 256);
    device_offset = layout.device_off_batch + align_up_(batch_bytes_total, 256);
    layout.device_off_sorted_meta = device_offset;
    device_offset += align_up_(per_device_molecule_count * sizeof_meta, 256);
    layout.device_off_coords_fp32 = device_offset;
    device_offset += align_up_(total_coords * sizeof(float), 256);
    layout.device_off_coords_fp16 = device_offset;
    device_offset += align_up_(total_coords * sizeof(__half), 256);
    layout.device_off_energies = device_offset;
    device_offset += align_up_(total_conformer_slots * sizeof(float), 256);
    layout.device_off_status = device_offset;
    device_offset += align_up_(total_conformer_slots * sizeof(std::uint8_t), 256);
    layout.device_off_initial = device_offset;
    device_offset += align_up_(
        static_cast<std::size_t>(per_device_molecule_count) * max_atoms_per_molecule * 4 * sizeof(float), 256);
    layout.device_pool_bytes = device_offset;

    std::size_t host_offset = 0;
    layout.host_off_coords = align_up_(host_offset, 256);
    host_offset = layout.host_off_coords + align_up_(total_coords * sizeof(__half), 256);
    layout.host_off_energies = host_offset;
    host_offset += align_up_(total_conformer_slots * sizeof(float), 256);
    layout.host_off_status = host_offset;
    host_offset += align_up_(total_conformer_slots * sizeof(std::uint8_t), 256);
    layout.host_pool_bytes = host_offset;
    return layout;
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

/**
 *  @brief Size an `embedding-wave` in molecules against the tightest per-virtual-slot VRAM.
 *
 *  Probes each *physical* device once, deduping repeated device-ids in @p device_ids — the
 *  `virtual-device` pattern runs K streams on one physical GPU as K virtual devices — and divides
 *  by that device's replica count on the spot, so each virtual slot's budget is
 *  `physical_free / replicas_for_this_physical`. Applies the `wave_budget_bytes` VRAM fraction,
 *  divides by the per-molecule pinned-host + device coords footprint at @p max_atoms_per_molecule
 *  and @p conformer_count, and clamps to `[10K, 400K]` for kernel-launch amortisation and
 *  pinned-host sanity respectively. `embed_4d_power_iter_kernel`'s surviving scratch
 *  (`group_indices`) is KB-scale, so no per-GPU cap is needed to bound it.
 *
 *  Both callers share this. The constructor sizes its worst-case pre-allocation at
 *  `max_conformer_count_k`; `generate_conformers` sizes the actual wave at the caller's K. The
 *  concerns that differ between them — honouring `config.wave_size` and clamping to what the
 *  constructor reserved — live in `generate_conformers`, not here.
 */
static std::uint32_t compute_wave_size_(custd::span<int const> device_ids, std::uint32_t max_atoms_per_molecule,
                                        std::uint32_t conformer_count, cuda_error_sink_t const &sink) {
    // Count replicas per physical device-id and probe each unique device once. The probe divides
    // by the replica count on the spot, so the slot budget below reads the maps directly.
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

    // The smallest per-virtual-slot budget binds: each slot owns a 1/replicas share of its
    // physical device's VRAM. Every key came from `device_ids`, so folding the map folds the slots.
    auto smallest_of = [](std::unordered_map<int, std::size_t> const &per_slot) noexcept -> std::size_t {
        if (per_slot.empty()) return 0;
        std::size_t smallest = per_slot.begin()->second;
        for (auto const &[device_id, slot_bytes] : per_slot) smallest = std::min(smallest, slot_bytes);
        return smallest;
    };
    std::size_t const budget_bytes = wave_budget_bytes(free_bytes_per_slot.size(), smallest_of(free_bytes_per_slot),
                                                       smallest_of(total_bytes_per_slot));

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

/**
 *  @brief Populate the per-device launch-configuration cache. One-shot at ctor:
 *  probes `cudaDeviceGetAttribute` and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
 *  for the two warp kernels across all atom-limited bins. The per-wave hot
 *  path reads from `device_launch_config_t`, never calls the driver again.
 */
static void populate_device_launch_configs_(custd::span<int const> device_ids,
                                            std::vector<device_launch_config_t> &launch_configs_out) {
    launch_configs_out.assign(device_ids.size(), device_launch_config_t {});
    for (std::size_t device_index = 0; device_index < device_ids.size(); ++device_index) {
        int const device_id = device_ids[device_index];
        cudaSetDevice(device_id);
        auto &config = launch_configs_out[device_index];
        cudaDeviceGetAttribute(&config.streaming_multiprocessor_count, cudaDevAttrMultiProcessorCount, device_id);
        for (std::size_t bin_index = 0; bin_index < atom_limited_bin_count_k; ++bin_index) {
            std::uint32_t const bin_max_atoms = atom_limited_bin_caps_k[bin_index];
            std::uint32_t const shared_bytes_etkdg = compute_pipeline_shared_memory_bytes(bin_max_atoms);
            std::uint32_t const shared_bytes_mmff = compute_mmff_polish_shared_memory_bytes(bin_max_atoms);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&config.blocks_per_multiprocessor_etkdg[bin_index],
                                                          etkdg_pipeline_warp_kernel, 32, shared_bytes_etkdg);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&config.blocks_per_multiprocessor_mmff[bin_index],
                                                          mmff_polish_warp_kernel, 32, shared_bytes_mmff);
        }
    }
}

context_t::context_t(custd::span<int const> devices) {
    if (devices.empty()) {
        int count = 0;
        cudaGetDeviceCount(&count);
        for (int device_index = 0; device_index < count; ++device_index) device_ids.push_back(device_index);
    }
    else { device_ids.assign(devices.begin(), devices.end()); }

    // `batch_slots_per_device` holds `max_devices_k` slots per parity and reaches
    // `prepare_per_device_batches_into` as a dynamic-extent span, so the fixed extent is erased at the
    // call and the callee indexes it by this count alone.  An unclamped list writes past the array, and
    // the default constructor enumerates every visible GPU, so nothing unusual on the caller's side is
    // needed to reach it.
    static_assert(max_devices_k <= smallfield::preprocess::distribute_scratch_t::max_devices_k,
                  "preprocess indexes its per-device scratch by the count this API admits");
    if (device_ids.size() > max_devices_k) {
        this->logger.error("context_t: {} devices visible, using the first {}; raise max_devices_k to use the rest",
                           device_ids.size(), max_devices_k);
        device_ids.resize(max_devices_k);
    }

    std::size_t const device_count = device_ids.size();
    streams.resize(device_count);
    metric_scratch.resize(device_count);
    launch_scratch.resize(device_count);

    // The MMFF94 parameter tables and the compiled CSD torsion patterns are process-wide constants
    // extracted once from RDKit; each device gets its own arena so no kernel reaches across a
    // device boundary for a parameter lookup.
    mmff94_tables_host_t mmff_tables_host;
    extract_mmff94_tables_from_rdkit(mmff_tables_host);
    csd_torsion_table_host_t csd_torsion_host;
    build_csd_torsion_table(csd_torsion_host);
    mmff_staging_layout_t const staging_layout = compute_mmff_staging_layout_();
    cuda_error_sink_t const sink {&this->logger, &this->cuda_error_count};

    for (std::size_t device_index = 0; device_index < device_count; ++device_index) {
        cudaSetDevice(device_ids[device_index]);
        cudaStreamCreate(&streams[device_index]);
        cudaStream_t const stream = streams[device_index];
        launch_scratch[device_index] = std::make_unique<launch_scratch_t>(stream);
        auto &scratch = *launch_scratch[device_index];
        // The metric pool is reserved lazily by `ensure_metric_scratch`, so it only needs its stream now.
        metric_scratch[device_index].pool = device_bytes_t {device_allocator_t {stream}};
        upload_mmff94_tables_to_device(mmff_tables_host, scratch.mmff_tables, stream, sink);
        upload_csd_torsion_table_to_device(csd_torsion_host, scratch.csd_torsion_table, stream, sink);
        if (scratch.mmff_staging.try_reserve(staging_layout.bytes_total) == reserve_outcome_t::failed_k)
            this->logger.error("context_t: MMFF staging allocation ({} bytes) failed on device {}",
                               staging_layout.bytes_total, device_ids[device_index]);
        // Persistent timing events; `cudaEventBlockingSync` keeps host-side
        // synchronize off the CPU busy-wait path when many streams share cores.
        cudaEventCreateWithFlags(&scratch.event_upload_start, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&scratch.event_metric_start, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&scratch.event_kernels_start, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&scratch.event_download_start, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&scratch.event_end, cudaEventBlockingSync);
    }

    // One-shot launch-config cache so the per-wave dispatch never probes again.
    populate_device_launch_configs_(device_ids, launch_configs);

    // `cudaFuncAttributeMaxDynamicSharedMemorySize` on the embed kernel stays
    // in effect for the lifetime of each device's primary context. At the
    // `max_atoms_per_molecule_k` cap the kernel needs ~148.5 KB shared, above the 48 KB default.
    std::size_t const max_embed_shared_bytes = smallfield::cuda::embed_shared_bytes(max_atoms_per_molecule_k);
    for (int const device_id : device_ids) {
        cudaSetDevice(device_id);
        cudaFuncSetAttribute(smallfield::cuda::embed_4d_power_iter_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(max_embed_shared_bytes));
    }

    // Pre-allocate every per-device pool to a worst-case wave so the hot path never grows. A grow
    // discards the buffer's contents whatever it costs to allocate, so `launch_gpu_pipeline` asserts
    // it finds `unchanged_k`.
    std::uint32_t const worst_case_wave_size = compute_wave_size_(device_ids, max_atoms_per_molecule_k,
                                                                  max_conformer_count_k, sink);
    std::uint32_t const per_device_worst_case_count = reserve_molecules_per_device(
        worst_case_wave_size, static_cast<std::uint32_t>(device_count));
    std::size_t const per_device_batch_bytes = estimate_per_molecule_batch_bytes(max_atoms_per_molecule_k) *
                                               per_device_worst_case_count;
    pipeline_pool_layout_t const worst_case_layout = compute_pipeline_pool_layout_(
        per_device_batch_bytes, per_device_worst_case_count, max_conformer_count_k, max_atoms_per_molecule_k);
    // Published so `generate_conformers` cannot run a wave larger than what was just reserved:
    // the same sizer answers again there at the caller's K, which at K=3 halves the per-molecule
    // estimate and doubles the wave, so the ceiling is what keeps the two answers compatible.
    per_device_reserved_molecules = per_device_worst_case_count;

    for (std::size_t device_index = 0; device_index < device_count; ++device_index) {
        cudaSetDevice(device_ids[device_index]);
        // Probes at `per_device_worst_case_count` molecules of `max_atoms_per_molecule_k` atoms each.
        ensure_metric_scratch(*this, device_index, per_device_worst_case_count, max_atoms_per_molecule_k);
        auto &scratch = *launch_scratch[device_index];
        // This is the reservation the launch path's own calls are expected to find already sufficient,
        // so a failure here is worth naming now rather than as a null pointer several waves later.
        if (scratch.device_pool.try_reserve(worst_case_layout.device_pool_bytes) == reserve_outcome_t::failed_k ||
            scratch.host_staging.try_reserve(worst_case_layout.host_pool_bytes) == reserve_outcome_t::failed_k)
            this->logger.error("context_t: pool reservation failed on device {} ({} device, {} host bytes)",
                               device_ids[device_index], worst_case_layout.device_pool_bytes,
                               worst_case_layout.host_pool_bytes);
    }
    // The pinned batch slots are deliberately left unreserved. `prepare_molecule_batch_into`
    // sizes each one from the wave's own `bytes_uploaded`, and preprocess partitions atom-count
    // descending, so the first wave carries the largest molecules and every slot reaches its peak
    // capacity on that wave; later waves find the buffer already large enough. Reserving here
    // instead would have to assume the `max_atoms_per_molecule_k` cap, which costs far more per
    // molecule than a corpus like GDB13 actually stages.
}

context_t::~context_t() {
    for (std::size_t device_index = 0; device_index < device_ids.size(); ++device_index) {
        cudaSetDevice(device_ids[device_index]);
        if (launch_scratch[device_index]) {
            auto &scratch = *launch_scratch[device_index];
            if (scratch.event_upload_start) cudaEventDestroy(scratch.event_upload_start);
            if (scratch.event_metric_start) cudaEventDestroy(scratch.event_metric_start);
            if (scratch.event_kernels_start) cudaEventDestroy(scratch.event_kernels_start);
            if (scratch.event_download_start) cudaEventDestroy(scratch.event_download_start);
            if (scratch.event_end) cudaEventDestroy(scratch.event_end);
        }
        launch_scratch[device_index].reset();
        cudaStreamDestroy(streams[device_index]);
    }
}

/**
 *  @brief Ensure the per-device metric scratch pool is sized for the given
 *  (max_group_size, max_atoms) combination. Grows only — never shrinks — so
 *  the first wave of a workload pays the allocation cost and subsequent
 *  waves reuse it.
 */
void ensure_metric_scratch(context_t &context, std::size_t device_index, std::size_t max_group_size,
                           std::uint32_t max_atoms) {
    auto &scratch = context.metric_scratch[device_index];
    if (max_group_size <= scratch.max_group_size && max_atoms <= scratch.max_atoms && scratch.pool.data() != nullptr)
        return;

    // Grow to at least the new request, but round up a bit to avoid thrashing
    // across similarly-sized waves.
    std::size_t const new_group = std::max(scratch.max_group_size, max_group_size);
    std::uint32_t const new_atoms = std::max<std::uint32_t>(scratch.max_atoms, max_atoms);

    cudaSetDevice(context.device_ids[device_index]);

    // `embed_4d_power_iter_kernel` runs entirely out of shared memory — Gram matrix, means,
    // eigenvectors and reduction scratch all live there for the kernel's lifetime — so the only
    // persistent device scratch is the `group_indices` uint32 list of which molecule goes into
    // which bin, which is KB-scale.
    std::size_t const group_bytes = align_up_(new_group * sizeof(std::uint32_t), 256);
    if (scratch.pool.try_reserve(group_bytes) == reserve_outcome_t::failed_k) {
        context.logger.error("ensure_metric_scratch: device allocation ({} bytes) failed", group_bytes);
        return;
    }
    auto *base = reinterpret_cast<char *>(scratch.pool.data());
    scratch.group_indices = {reinterpret_cast<std::uint32_t *>(base), new_group};
    scratch.max_group_size = new_group;
    scratch.max_atoms = new_atoms;
}

/**
 *  @brief Per-wave, per-GPU stream-local phase timings, populated in
 *  `finalize_gpu_pipeline` after the stream drains. All values are in
 *  milliseconds and are measured by `cudaEventElapsedTime` on events that
 *  were recorded into the device's stream — they reflect actual GPU + DMA
 *  wall, not host-side scheduling. Aggregator in `generate_conformers`
 *  combines these across devices to derive per-wave GPU wall (max) and
 *  total GPU work (sum).
 */
struct gpu_phase_ms_t {
    /** H2D batch + metadata copies. */
    float upload_ms = 0.f;
    /** `build_metric_initial_coordinates_on_gpu` (embed kernel). */
    float metric_ms = 0.f;
    /** Wave-binned `etkdg_pipeline_warp_kernel` launches. */
    float kernels_ms = 0.f;
    /** D2H coords/energies/status copies. */
    float download_ms = 0.f;
    /** Upload+metric+kernels+download (all on the same stream). */
    float total_ms = 0.f;
};

/**
 *  @brief Per-wave, per-GPU state carried from `launch_gpu_pipeline` to `finalize_gpu_pipeline`.
 *
 *  Holds the device memory pool that stays alive until the stream drains, the host
 *  staging buffers that the async download writes into, and the metadata needed to
 *  unsort results into the caller's output arrays. Move-only by design.
 */
struct gpu_pipeline_handle_t {
    int device_id = -1;
    cudaStream_t stream = nullptr;
    /**
     *  @brief Sorted-index to input-index map, viewing this device's pinned pool.
     *
     *  A view rather than a pointer to the pool, because the pinned allocation does not move while a
     *  wave is in flight but the container holding the pools may be reshaped.
     */
    custd::span<std::uint32_t const> input_indices {};
    /** Non-owning — persistent in context_t. */
    launch_scratch_t *scratch = nullptr;
    std::uint32_t per_device_molecule_count = 0;
    std::uint32_t conformer_count = 0;
    /**
     *  @brief The largest atom count this wave actually carries, driving the GPU coordinate stride
     *  and the device-to-host staging stride.
     *
     *  An observation rather than a cap, so a wave of small molecules pays no full-stride tax in
     *  output VRAM or download bandwidth. It never exceeds `host_max_atoms`, and the reason is a
     *  gate rather than a coincidence: `prepare_per_device_batches_into` takes the caller's
     *  `output_atom_capacity` and refuses any molecule above it with `too_many_atoms_k`, so a
     *  molecule that would widen the stride past the caller's buffer never enters the wave.
     */
    std::uint32_t gpu_max_atoms = 0;
    /**
     *  @brief The caller's own output stride, which is what `host_coords_out` was allocated at.
     *
     *  Finalize copies `3 * gpu_max_atoms` halves out of staging into a slot `3 * host_max_atoms`
     *  wide, leaving the trailing halves at their pre-zeroed value. The two strides are independent
     *  numbers and conflating them is what once wrote one wave's y plane over another's x.
     */
    std::uint32_t host_max_atoms = 0;
    /** Offset into caller's output (sorted space). */
    std::uint32_t wave_base = 0;
    /*  Device-side output spans, pointing into `scratch->device_pool`. */
    /** Kernel output in fp32. */
    custd::span<float> device_coords_fp32 {};
    /** Post-convert coords for D2H. */
    custd::span<__half> device_coords_fp16 {};
    custd::span<float> device_energies_out {};
    custd::span<std::uint8_t> device_status_out {};
    /*  Pinned host-side readback spans, pointing into `scratch->host_staging`. Finalize drains the
     *  stream, then permutes from these into the caller's output slice. */
    custd::span<__half> host_coords_sorted {};
    custd::span<float> host_energies_sorted {};
    custd::span<std::uint8_t> host_status_sorted {};
    /*  Caller-owned output slices, non-owning. */
    custd::span<__half> host_coords_out {};
    custd::span<float> host_energies_out {};
    custd::span<conformer_status_t> host_status_out {};
    /** Inverse SMILES-length sort; empty means no extra unsort on the non-pipelined path. */
    custd::span<std::uint32_t const> sort_perm {};
    /*  Stream-local phase timing events, recorded in launch and consumed in finalize. */
    cudaEvent_t event_upload_start = nullptr;
    cudaEvent_t event_metric_start = nullptr;
    cudaEvent_t event_kernels_start = nullptr;
    cudaEvent_t event_download_start = nullptr;
    cudaEvent_t event_end = nullptr;
    /** Caller-owned slot; finalize writes per-phase ms. */
    gpu_phase_ms_t *timings_out = nullptr;
    /** Routes errors through caller's logger + counter. */
    cuda_error_sink_t sink {};
};

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
 *  @param[in] batch_layout Layout the mirror was carved with; supplies the torsion stride.
 *  @param[inout] launch_buffers This device's persistent scratch, holding the parameter tables and
 *                the extractor's fixed-stride staging.
 *  @param[in] molecule_count Molecules in the batch.
 *  @param[in] stream Stream to enqueue on.
 *  @param[in] sink Error sink routing failures through the caller's logger.
 */
static void enqueue_device_preprocess_(batch_pool_dev_view_t const &device_view,
                                       wave_slice_layout_t const &batch_layout, launch_scratch_t &launch_buffers,
                                       std::uint32_t molecule_count, cudaStream_t stream,
                                       cuda_error_sink_t const &sink) {
    if (molecule_count == 0) return;

    auto *const descriptors = device_view.descriptors().data();
    auto *const metadata = device_view.molecule_metadata().data();
    auto *const atom_bases = device_view.atom_bases().data();
    auto *const mmff_bases = device_view.mmff_bases().data();
    auto *const atom_types = device_view.atom_types().data();
    auto *const atom_charges = device_view.mmff_atom_charge().data();
    auto *const typing_status = device_view.typing_status().data();
    auto *const atom_offsets = device_view.mmff_atom_offsets().data();

    launch_bond_distance_bfs(descriptors, metadata, device_view.bond_distance().data(), molecule_count, stream);
    sink.check(cudaPeekAtLastError(), "bond_distance_bfs_kernel launch");

    launch_mmff94_type_and_charge(descriptors, atom_bases, molecule_count, launch_buffers.mmff_tables.view, atom_types,
                                  atom_charges, typing_status, stream);
    sink.check(cudaPeekAtLastError(), "mmff94_type_and_charge_kernel launch");

    std::uint32_t const patch_count = static_cast<std::uint32_t>(batch_layout.num_typing_patches);
    if (sink.log != nullptr)
        sink.log->trace2("device preprocess: {} of {} molecules take the host typing path", patch_count,
                         molecule_count);
    launch_mmff94_apply_typing_patch(device_view.typing_patches().data(), patch_count,
                                     device_view.patch_atom_types().data(), device_view.patch_atom_charges().data(),
                                     atom_types, atom_charges, typing_status, stream);
    sink.check(cudaPeekAtLastError(), "mmff94_apply_typing_patch_kernel launch");

    mmff_staging_layout_t const staging = compute_mmff_staging_layout_();
    std::byte *const staging_base = launch_buffers.mmff_staging.data();
    auto *const staged_bonds = reinterpret_cast<mmff94_bond_t *>(staging_base + staging.offset_bonds);
    auto *const staged_angles = reinterpret_cast<mmff94_angle_t *>(staging_base + staging.offset_angles);
    auto *const staged_torsions = reinterpret_cast<mmff94_torsion_t *>(staging_base + staging.torsions_byte_offset);
    auto *const staged_oops = reinterpret_cast<mmff94_oop_t *>(staging_base + staging.offset_oops);
    auto *const staged_stretchbends = reinterpret_cast<mmff94_stretchbend_t *>(staging_base +
                                                                               staging.offset_stretchbends);
    auto *const staged_interactions = reinterpret_cast<mmff94_atom_interaction_t *>(staging_base +
                                                                                    staging.offset_interactions);
    auto *const window_counts = reinterpret_cast<mmff94_per_mol_counts_t *>(staging_base + staging.offset_counts);

    for (std::uint32_t window_start = 0; window_start < molecule_count; window_start += mmff_extract_window_k) {
        std::uint32_t const window_size = std::min(mmff_extract_window_k, molecule_count - window_start);
        launch_mmff94_extract_terms(descriptors + window_start, atom_types, atom_charges, atom_bases + window_start,
                                    window_size, launch_buffers.mmff_tables.view, device_view.mmff_atom_vdw().data(),
                                    atom_charges, staged_bonds, staged_angles, staged_torsions, staged_oops,
                                    staged_stretchbends, window_counts, typing_status + window_start, stream);
        sink.check(cudaPeekAtLastError(), "mmff94_extract_terms_kernel launch");

        launch_mmff94_build_csr(descriptors + window_start, atom_bases + window_start, window_size, window_counts,
                                staged_bonds, staged_angles, staged_torsions, staged_oops, staged_stretchbends,
                                staged_interactions, atom_offsets, typing_status + window_start, stream);
        sink.check(cudaPeekAtLastError(), "mmff94_build_csr_kernel launch");

        launch_mmff94_compact_terms(descriptors + window_start, atom_bases + window_start, window_size, window_counts,
                                    mmff_bases + window_start, staged_bonds, staged_angles, staged_torsions,
                                    staged_oops, staged_stretchbends, staged_interactions,
                                    device_view.mmff_bonds().data(), device_view.mmff_angles().data(),
                                    device_view.mmff_torsions().data(), device_view.mmff_oops().data(),
                                    device_view.mmff_stretchbends().data(), device_view.mmff_interactions().data(),
                                    atom_offsets, metadata + window_start, typing_status + window_start, stream);
        sink.check(cudaPeekAtLastError(), "mmff94_compact_terms_kernel launch");
    }

    std::uint32_t const torsion_stride = batch_layout.torsion_stride;
    launch_csd_torsion_match(descriptors, molecule_count, launch_buffers.csd_torsion_table.view,
                             device_view.etk_torsions().data(), torsion_stride, device_view.torsion_counts().data(),
                             stream);
    sink.check(cudaPeekAtLastError(), "csd_torsion_match_kernel launch");

    launch_etk_build_csr(device_view.etk_torsions().data(), torsion_stride, device_view.torsion_counts().data(),
                         device_view.distance_constraints().data(), molecule_count,
                         device_view.etk_interactions().data(), device_view.etk_atom_offsets().data(), metadata,
                         stream);
    sink.check(cudaPeekAtLastError(), "etk_build_csr_kernel launch");
}

/**
 *  @brief Enqueue upload, metric prepass, kernels, and download on this
 *  device's three-stream pipeline. Returns immediately; caller eventually
 *  calls `finalize_gpu_pipeline` with the returned handle.
 */
gpu_pipeline_handle_t launch_gpu_pipeline(int device_id, cudaStream_t stream, metric_scratch_t &scratch,
                                          launch_scratch_t &launch_buffers, wave_slice_pool_t const &per_device_batch,
                                          device_launch_config_t const &launch_config, std::uint32_t conformer_count,
                                          std::uint32_t stereo_retry_limit, std::uint32_t host_max_atoms_per_molecule,
                                          std::uint64_t seed, std::uint32_t wave_base,
                                          custd::span<std::uint32_t const> sort_perm,
                                          custd::span<__half> host_coords_out, custd::span<float> host_energies_out,
                                          custd::span<conformer_status_t> host_status_out, cuda_error_sink_t sink) {

    sink.check(cudaSetDevice(device_id), "cudaSetDevice(launch_gpu_pipeline)");

    std::uint32_t const per_device_molecule_count = per_device_batch.num_molecules;
    // GPU-side coord buffer + pinned D2H staging are sized to this wave's *actual*
    // max atom count (from preprocess), not the caller's worst-case `max_atoms_per_molecule_k`.
    // Caller's `host_coords_out` keeps that fixed-stride contract; finalize zero-pads on the
    // per-slot copy.
    std::uint32_t const gpu_max_atoms_per_molecule = std::max<std::uint32_t>(per_device_batch.observed_max_atoms, 1u);
    std::size_t const total_conformer_slots = static_cast<std::size_t>(per_device_molecule_count) * conformer_count;
    std::size_t const total_coords = total_conformer_slots * gpu_max_atoms_per_molecule * 3;
    // The per-device batch has already been partitioned + atom-count sorted by
    // preprocess; kernels read `molecule_metadata` directly in place of the
    // old device-side sorted-meta region. No per-device sort here.
    auto host_metadata = per_device_batch.molecule_metadata();

    pipeline_pool_layout_t const layout = compute_pipeline_pool_layout_(
        per_device_batch.layout.bytes_total, per_device_molecule_count, conformer_count, gpu_max_atoms_per_molecule);

    gpu_pipeline_handle_t handle;
    handle.device_id = device_id;
    handle.stream = stream;
    handle.input_indices = per_device_batch.input_indices();
    handle.scratch = &launch_buffers;
    handle.per_device_molecule_count = per_device_molecule_count;
    handle.conformer_count = conformer_count;
    handle.gpu_max_atoms = gpu_max_atoms_per_molecule;
    handle.host_max_atoms = host_max_atoms_per_molecule;
    handle.wave_base = wave_base;
    handle.host_coords_out = host_coords_out;
    handle.host_energies_out = host_energies_out;
    handle.host_status_out = host_status_out;
    handle.sort_perm = sort_perm;
    handle.sink = sink;
    handle.event_upload_start = launch_buffers.event_upload_start;
    handle.event_metric_start = launch_buffers.event_metric_start;
    handle.event_kernels_start = launch_buffers.event_kernels_start;
    handle.event_download_start = launch_buffers.event_download_start;
    handle.event_end = launch_buffers.event_end;

    // Pools were pre-sized to worst case at ctor; calls here are no-ops barring a
    // pathologically large input that exceeded the worst-case probe. A failure leaves
    // the buffer pointer null, so bail with `device_id` at its -1 default and let
    // `finalize_wave` skip this handle rather than memcpy through nothing.
    reserve_outcome_t const device_pool_outcome = launch_buffers.device_pool.try_reserve(layout.device_pool_bytes);
    reserve_outcome_t const host_staging_outcome = launch_buffers.host_staging.try_reserve(layout.host_pool_bytes);
    // The constructor reserved for the worst case, so reaching `grown_k` here means an allocation fired on
    // the wave path the pre-reservation exists to keep clear. That is a sizing bug, not a memory shortage.
    sf_assert_(device_pool_outcome != reserve_outcome_t::grown_k);
    sf_assert_(host_staging_outcome != reserve_outcome_t::grown_k);
    if (device_pool_outcome == reserve_outcome_t::failed_k || host_staging_outcome == reserve_outcome_t::failed_k) {
        if (sink.log != nullptr)
            sink.log->error("launch_gpu_pipeline: pool allocation failed on device {} ({} device, {} host bytes)",
                            device_id, layout.device_pool_bytes, layout.host_pool_bytes);
        return gpu_pipeline_handle_t {};
    }

    std::byte *const device_base = launch_buffers.device_pool.data();
    std::byte *const host_base = launch_buffers.host_staging.data();

    batch_pool_dev_view_t device_view {device_base + layout.device_off_batch, &per_device_batch.layout};
    handle.device_coords_fp32 = {reinterpret_cast<float *>(device_base + layout.device_off_coords_fp32), total_coords};
    handle.device_coords_fp16 = {reinterpret_cast<__half *>(device_base + layout.device_off_coords_fp16), total_coords};
    handle.device_energies_out = {reinterpret_cast<float *>(device_base + layout.device_off_energies),
                                  total_conformer_slots};
    handle.device_status_out = {reinterpret_cast<std::uint8_t *>(device_base + layout.device_off_status),
                                total_conformer_slots};
    auto *device_initial_coordinates = reinterpret_cast<float *>(device_base + layout.device_off_initial);

    handle.host_coords_sorted = {reinterpret_cast<__half *>(host_base + layout.host_off_coords), total_coords};
    handle.host_energies_sorted = {reinterpret_cast<float *>(host_base + layout.host_off_energies),
                                   total_conformer_slots};
    handle.host_status_sorted = {reinterpret_cast<std::uint8_t *>(host_base + layout.host_off_status),
                                 total_conformer_slots};

    // Upload: single H2D of the per-device batch pool (reads from this
    // device's private pinned buffer — no cross-device sharing).
    cudaEventRecord(handle.event_upload_start, stream);
    if (per_device_batch.layout.bytes_uploaded > 0)
        sink.check(cudaMemcpyAsync(device_base + layout.device_off_batch, per_device_batch.host.data(),
                                   per_device_batch.layout.bytes_uploaded, cudaMemcpyHostToDevice, stream),
                   "H2D per-device batch pool");

    // Zero the fp32 coordinate buffer before kernel launches. write_conformer_output_
    // only writes the first num_atoms slots per dim, leaving [num_atoms, output_stride)
    // uninitialized when the molecule is shorter than max_atoms_per_molecule. The
    // downstream convert_coords_fp32_to_fp16_kernel processes the full stride
    // unconditionally — without zero-init here, those slots are read uninitialized
    // (compute-sanitizer initcheck flags ~16K reads on the 50-conformer test corpus).
    // Reading uninit memory is UB and risks NaN/Inf propagation; the per-launch
    // memset is bandwidth-trivial vs the FIRE compute that follows.
    if (total_coords > 0)
        sink.check(cudaMemsetAsync(handle.device_coords_fp32.data(), 0, total_coords * sizeof(float), stream),
                   "memset device_coords_fp32 zero");

    // The device preprocessing chain. Everything from here to the metric prepass reads the
    // uploaded descriptors and writes the regions past `bytes_uploaded`, so the host ships the
    // molecular graph and the GPU derives the force field from it. One stream, in order, so each
    // pass sees the previous one's output without an event or a synchronize anywhere.
    enqueue_device_preprocess_(device_view, per_device_batch.layout, launch_buffers, per_device_molecule_count, stream,
                               sink);

    cudaEventRecord(handle.event_metric_start, stream);

    // Kernels read metadata directly from the H2D'd batch mirror; preprocess wrote mols in atom-count
    // ascending order per device, so the per-bin slicing below can index the batch's metadata region
    // with `bin_start`.
    auto *device_metadata_sorted = device_view.molecule_metadata().data();

    build_metric_initial_coordinates_on_gpu(device_view.bounds_matrices().data(), device_metadata_sorted, host_metadata,
                                            gpu_max_atoms_per_molecule, device_initial_coordinates, scratch, sink,
                                            stream);
    cudaEventRecord(handle.event_kernels_start, stream);

    int const streaming_multiprocessor_count = launch_config.streaming_multiprocessor_count;

    std::uint32_t bin_start = 0;
    for (std::size_t bin_index = 0; bin_index < atom_limited_bin_count_k && bin_start < per_device_molecule_count;
         ++bin_index) {
        std::uint32_t bin_end = bin_start;
        while (bin_end < per_device_molecule_count &&
               host_metadata[bin_end].num_atoms <= atom_limited_bin_caps_k[bin_index])
            ++bin_end;
        if (bin_end == bin_start) continue;

        std::uint32_t const bin_size = bin_end - bin_start;
        std::uint32_t const bin_max_atoms = atom_limited_bin_caps_k[bin_index];
        std::uint32_t const shared_bytes_etkdg = compute_pipeline_shared_memory_bytes(bin_max_atoms);
        std::uint32_t const shared_bytes_mmff = compute_mmff_polish_shared_memory_bytes(bin_max_atoms);
        int const blocks_per_multiprocessor_etkdg = launch_config.blocks_per_multiprocessor_etkdg[bin_index];
        int const blocks_per_multiprocessor_mmff = launch_config.blocks_per_multiprocessor_mmff[bin_index];
        // Both kernels assign one warp per (molecule, conformer) pair, so the index space the
        // grid has to cover is the pair count, not the molecule count. Several atom-count bins
        // hold fewer molecules than the device has resident-warp slots; scaling by the conformer
        // count is what lets those launches fill the device.
        std::uint32_t const bin_pair_count = bin_size * conformer_count;
        std::uint32_t const grid_size_etkdg = std::min(
            bin_pair_count,
            static_cast<std::uint32_t>(blocks_per_multiprocessor_etkdg * streaming_multiprocessor_count));
        std::uint32_t const grid_size_mmff = std::min(
            bin_pair_count,
            static_cast<std::uint32_t>(blocks_per_multiprocessor_mmff * streaming_multiprocessor_count));

        etkdg_pipeline_warp_kernel<<<grid_size_etkdg, 32, shared_bytes_etkdg, stream>>>(
            device_view.bounds_matrices().data(),
            device_initial_coordinates + bin_start * gpu_max_atoms_per_molecule * 4,
            device_view.chiral_centers().data(), device_view.etk_torsions().data(),
            device_view.distance_constraints().data(), device_view.etk_interactions().data(),
            device_view.etk_atom_offsets().data(), device_metadata_sorted + bin_start,
            device_view.input_indices().data() + bin_start,
            handle.device_coords_fp32.data() + bin_start * conformer_count * gpu_max_atoms_per_molecule * 3,
            handle.device_energies_out.data() + bin_start * conformer_count,
            handle.device_status_out.data() + bin_start * conformer_count, bin_size, conformer_count,
            stereo_retry_limit, bin_max_atoms, gpu_max_atoms_per_molecule, gpu_max_atoms_per_molecule, wave_base, seed);
        sink.check(cudaPeekAtLastError(), "etkdg_pipeline_warp_kernel launch");

        mmff_polish_warp_kernel<<<grid_size_mmff, 32, shared_bytes_mmff, stream>>>(
            device_view.mmff_atom_vdw().data(), device_view.mmff_atom_charge().data(),
            device_view.bond_distance().data(), device_view.mmff_bonds().data(), device_view.mmff_angles().data(),
            device_view.mmff_torsions().data(), device_view.mmff_oops().data(), device_view.mmff_stretchbends().data(),
            device_view.mmff_interactions().data(), device_view.mmff_atom_offsets().data(),
            device_metadata_sorted + bin_start,
            handle.device_coords_fp32.data() + bin_start * conformer_count * gpu_max_atoms_per_molecule * 3,
            handle.device_energies_out.data() + bin_start * conformer_count,
            handle.device_status_out.data() + bin_start * conformer_count, bin_size, conformer_count,
            gpu_max_atoms_per_molecule, bin_max_atoms);
        sink.check(cudaPeekAtLastError(), "mmff_polish_warp_kernel launch");

        bin_start = bin_end;
    }

    if (total_coords > 0) {
        constexpr std::uint32_t convert_block_k = 256;
        std::uint32_t const num_elements = static_cast<std::uint32_t>(total_coords);
        std::uint32_t const convert_grid = (num_elements + convert_block_k - 1) / convert_block_k;
        convert_coords_fp32_to_fp16_kernel<<<convert_grid, convert_block_k, 0, stream>>>(
            handle.device_coords_fp32.data(), handle.device_coords_fp16.data(), num_elements);
        sink.check(cudaPeekAtLastError(), "convert_coords_fp32_to_fp16_kernel launch");
    }

    cudaEventRecord(handle.event_download_start, stream);
    if (total_coords > 0)
        sink.check(cudaMemcpyAsync(handle.host_coords_sorted.data(), handle.device_coords_fp16.data(),
                                   total_coords * sizeof(__half), cudaMemcpyDeviceToHost, stream),
                   "D2H coords_fp16");
    if (total_conformer_slots > 0) {
        sink.check(cudaMemcpyAsync(handle.host_energies_sorted.data(), handle.device_energies_out.data(),
                                   total_conformer_slots * sizeof(float), cudaMemcpyDeviceToHost, stream),
                   "D2H energies");
        sink.check(cudaMemcpyAsync(handle.host_status_sorted.data(), handle.device_status_out.data(),
                                   total_conformer_slots * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream),
                   "D2H status");
    }
    cudaEventRecord(handle.event_end, stream);
    return handle;
}

/**
 *  @brief Wait for an already-launched wave to finish and permute results into
 *  the caller's output arrays. All three D2H transfers were already enqueued by
 *  `launch_gpu_pipeline` into pinned host staging; this just drains the stream
 *  and does a permutation memcpy.
 */
void finalize_gpu_pipeline(gpu_pipeline_handle_t handle) {
    handle.sink.check(cudaSetDevice(handle.device_id), "cudaSetDevice(finalize_gpu_pipeline)");
    // Draining the stream, rather than syncing on `handle.event_end`, is correct whether or not
    // the end event was ever recorded, and costs about 5 ms per wave per device. See ROADMAP.md
    // under "Optimizations tried and rejected" for why the event form is not used.
    handle.sink.check(cudaStreamSynchronize(handle.stream), "cudaStreamSynchronize(finalize)");

    if (handle.timings_out != nullptr && handle.event_end != nullptr) {
        cudaEventElapsedTime(&handle.timings_out->upload_ms, handle.event_upload_start, handle.event_metric_start);
        cudaEventElapsedTime(&handle.timings_out->metric_ms, handle.event_metric_start, handle.event_kernels_start);
        cudaEventElapsedTime(&handle.timings_out->kernels_ms, handle.event_kernels_start, handle.event_download_start);
        cudaEventElapsedTime(&handle.timings_out->download_ms, handle.event_download_start, handle.event_end);
        cudaEventElapsedTime(&handle.timings_out->total_ms, handle.event_upload_start, handle.event_end);
    }

    // Preprocess wrote the per-device batch in atom-count ascending order, so
    // `input_indices[sorted_index]` directly maps back to the input position.
    // The outer SMILES-length sort (`sort_perm`, if present) then converts
    // input position to the caller's destination index.
    auto input_indices = handle.input_indices;
    // GPU-side coord stride is the wave's actual max atoms (smaller for waves whose mols are
    // below the `max_atoms_per_molecule_k` cap); caller's `host_coords_out` keeps a fixed
    // full stride. The caller's coords buffer was zero-init'd at the top of generate_conformers,
    // so the trailing halves — three times the difference between `host_max_atoms` and
    // `gpu_max_atoms` — stay zero without an explicit memset here.
    std::uint32_t const gpu_coord_stride = handle.gpu_max_atoms * 3;
    std::uint32_t const host_coord_stride = handle.host_max_atoms * 3;
    std::uint32_t const conformer_count = handle.conformer_count;
    bool const has_perm = !handle.sort_perm.empty();
    // This loop runs between the stream drain above and the next wave's launch, so
    // every GPU is idle for its duration — at 400K molecules and K=3 it is millions of
    // scattered 256-byte copies. `input_indices` is a permutation and `destination_slot`
    // is a pure function of `sorted_index` and `conformer`, so no two iterations touch the
    // same output bytes and the loop parallelises as written. Devices are still
    // finalized one at a time: they were launched together and carry equal work, so the
    // later stream drains return immediately, and keeping the fan-out on this loop lets
    // each device's permutation use every core rather than one core per device.
#pragma omp parallel for schedule(static)
    for (std::uint32_t sorted_index = 0; sorted_index < handle.per_device_molecule_count; ++sorted_index) {
        std::uint32_t const input_index = input_indices[sorted_index];
        std::uint32_t const caller_index = has_perm ? handle.sort_perm[handle.wave_base + input_index] : input_index;
        for (std::uint32_t conformer = 0; conformer < conformer_count; ++conformer) {
            std::uint32_t const source_slot = sorted_index * conformer_count + conformer;
            std::uint32_t const destination_slot = caller_index * conformer_count + conformer;
            handle.host_energies_out[destination_slot] = handle.host_energies_sorted[source_slot];
            std::uint8_t const raw = handle.host_status_sorted[source_slot];
            handle.host_status_out[destination_slot] = raw == 0   ? conformer_status_t::success_k
                                                       : raw == 1 ? conformer_status_t::stereo_failure_k
                                                                  : conformer_status_t::mmff_unconverged_k;
            // Per-slot layout is SoA: `[x_0..x_{stride-1}, y_0..y_{stride-1}, z_0..z_{stride-1}]`
            // (see write_conformer_output_ in etkdg_pipeline.cuh). Source uses
            // `gpu_max_atoms` stride; destination uses `host_max_atoms` stride.
            // Copy each dim into its dim-section of the destination — the
            // trailing `(host_max_atoms - gpu_max_atoms)` halves per dim stay
            // at the pre-zeroed value.
            __half *dest_slot = handle.host_coords_out.data() +
                                static_cast<std::size_t>(destination_slot) * host_coord_stride;
            __half const *source_slot_ptr = handle.host_coords_sorted.data() +
                                            static_cast<std::size_t>(source_slot) * gpu_coord_stride;
            // Both sides are structure-of-arrays but with different strides, so copy
            // component-by-component. See `conformer_slot_layout_t` for the contract
            // shared with the RDKit fallback producer and the Parquet writer.
            conformer_slot_layout_t const destination_layout {handle.host_max_atoms, 1u};
            conformer_slot_layout_t const source_layout {handle.gpu_max_atoms, 1u};
            for (std::uint32_t axis = 0; axis < 3u; ++axis)
                std::memcpy(dest_slot + destination_layout.component_offset(axis, 0u),
                            source_slot_ptr + source_layout.component_offset(axis, 0u),
                            handle.gpu_max_atoms * sizeof(__half));
        }
    }
}

/**
 *  @brief Per-`embedding-wave` timing accumulator populated by `launch_wave`
 *  and `finalize_wave`. Pretty-printed by `print_wave_timeline_` at the
 *  end of `generate_conformers` when `logger.verbose_level >= 1`.
 */
struct wave_record_t {
    std::uint32_t wave_index;
    std::uint32_t n_mol;
    /** Host wall clock in ms relative to call start, for every boundary below. */
    double prep_start_ms;
    double prep_end_ms;
    /** Entry to and return from `launch_wave`. */
    double launch_start_ms;
    double launch_end_ms;
    /** Entry to and return from `finalize_wave` for this wave. */
    double finalize_start_ms;
    double finalize_end_ms;
    std::vector<gpu_phase_ms_t> per_device_gpu;
};

/**
 *  @brief Build a descending-by-SMILES-length permutation of the input and
 *  write the reordered SMILES views into @p sorted_smiles_storage.
 *
 *  Putting the smallest molecules in the last `embedding-wave` shortens the
 *  tail: wave N's GPU work is the only fully unhidden wave-step (no wave N+1
 *  to overlap with), so making wave N the cheapest shrinks overall wall.
 *  `finalize_gpu_pipeline` folds this permutation into the same reorder that
 *  undoes the per-GPU atom-count sort.
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
 *  @brief Verbose-tier-1 per-wave summary + verbose-tier-2 per-device
 *  breakdown. No-op when `logger.verbose_level < 1`.
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
    for (auto const &wave_record : wave_records) {
        float gpu_max = 0.f, gpu_sum = 0.f, upload = 0.f, metric = 0.f, kernels = 0.f, download = 0.f;
        std::uint32_t n_active = 0;
        for (auto const &device_timing : wave_record.per_device_gpu) {
            if (device_timing.total_ms <= 0.f) continue;
            gpu_max = std::max(gpu_max, device_timing.total_ms);
            gpu_sum += device_timing.total_ms;
            upload += device_timing.upload_ms;
            metric += device_timing.metric_ms;
            kernels += device_timing.kernels_ms;
            download += device_timing.download_ms;
            ++n_active;
        }
        float const inverse_active_count = n_active > 0 ? 1.f / static_cast<float>(n_active) : 0.f;
        logger.trace("{:>{}} {:>{}} {:>{}.1f} {:>{}.1f} {:>{}.1f} {:>{}.1f} / {:>{}.1f} {:>{}.1f} / {:>{}.1f} / " //
                     "{:>{}.1f} / {:>{}.1f}",
                     wave_record.wave_index, wave_width_k, wave_record.n_mol, mols_width_k,
                     wave_record.prep_end_ms - wave_record.prep_start_ms, time_width_k,
                     wave_record.launch_end_ms - wave_record.launch_start_ms, time_width_k,
                     wave_record.finalize_end_ms - wave_record.finalize_start_ms, time_width_k,
                     gpu_sum * inverse_active_count, time_width_k, gpu_max, time_width_k, upload * inverse_active_count,
                     time_width_k, metric * inverse_active_count, time_width_k, kernels * inverse_active_count,
                     time_width_k, download * inverse_active_count, time_width_k);
    }
    logger.trace("wall {:.1f} ms", total_wall_ms);
    for (auto const &wave_record : wave_records) {
        for (std::uint32_t device_index = 0; device_index < wave_record.per_device_gpu.size(); ++device_index) {
            auto const &device_timing = wave_record.per_device_gpu[device_index];
            if (device_timing.total_ms <= 0.f) continue;
            logger.trace2(
                "  wave={} dev={}  upload={:.1f}  metric={:.1f}  kernels={:.1f}  download={:.1f}  total={:.1f}",
                wave_record.wave_index, device_index, device_timing.upload_ms, device_timing.metric_ms,
                device_timing.kernels_ms, device_timing.download_ms, device_timing.total_ms);
        }
    }
}

/**
 *  @brief Half-open `[begin, end)` slice of the sorted input that one `embedding-wave` covers.
 *
 *  Indices are positions in the SMILES-length-sorted view, not in the caller's array; `begin`
 *  doubles as the wave base `finalize_gpu_pipeline` adds before consulting the sort permutation.
 */
struct wave_range_t {
    std::uint32_t begin;
    std::uint32_t end;
};

conformer_status_t context_t::generate_conformers(
    custd::span<std::string_view const> smiles, config_t const &config, std::uint32_t max_atoms_per_molecule,
    custd::span<__half> coordinates_out, custd::span<float> energies_out, custd::span<conformer_status_t> status_out,
    custd::span<std::uint8_t> num_atoms_out, custd::span<std::uint16_t> num_heavy_atoms_out,
    custd::span<std::uint16_t> num_bonds_out, custd::span<float> molecular_weight_out) {

    cuda_error_sink_t const sink {&this->logger, &this->cuda_error_count};

    std::uint32_t const num_input = static_cast<std::uint32_t>(smiles.size());
    std::uint32_t const num_conformers = config.num_conformers;

    // Pre-initialize caller output buffers. `finalize_gpu_pipeline` writes only
    // into the slots whose molecules made it through preprocessing; the rest
    // stay at these defaults (invalid_smiles / zero coords / zero energy).
    std::fill_n(status_out.begin(), status_out.size(), conformer_status_t::invalid_smiles_k);
    std::fill_n(energies_out.begin(), energies_out.size(), 0.0f);
    {
        __half const zero = __float2half_rn(0.0f);
        std::fill_n(coordinates_out.begin(), coordinates_out.size(), zero);
    }
    if (!num_atoms_out.empty()) std::fill_n(num_atoms_out.begin(), num_atoms_out.size(), std::uint8_t {0});
    if (!num_heavy_atoms_out.empty())
        std::fill_n(num_heavy_atoms_out.begin(), num_heavy_atoms_out.size(), std::uint16_t {0});
    if (!num_bonds_out.empty()) std::fill_n(num_bonds_out.begin(), num_bonds_out.size(), std::uint16_t {0});
    if (!molecular_weight_out.empty()) std::fill_n(molecular_weight_out.begin(), molecular_weight_out.size(), 0.0f);

    // Sort SMILES by descending length so the last `embedding-wave` carries
    // the smallest molecules — shortens the unhidden tail wave.
    std::vector<std::uint32_t> sort_perm;
    std::vector<std::string_view> sorted_smiles_storage;
    sort_input_by_smiles_length_(smiles, sort_perm, sorted_smiles_storage);
    custd::span<std::string_view const> smiles_view = sorted_smiles_storage;
    custd::span<std::uint32_t const> const sort_perm_view = sort_perm;

    std::uint32_t const num_devices = static_cast<std::uint32_t>(device_ids.size());
    // A non-zero `config.wave_size` is taken as given; an auto-sized wave additionally may never
    // exceed what the constructor reserved. Pools are pre-grown per device, so that ceiling is
    // `per_device_reserved_molecules` times the device count; going past it would fire a grow from
    // inside the per-device OMP fan-out, discarding a pool the wave is mid-way through filling.
    // The two sizings disagree otherwise: the
    // constructor assumes the worst-case `max_conformer_count_k`, while this one uses the caller's
    // actual K, which at K=3 halves the per-molecule estimate and doubles the wave it would return.
    std::uint32_t const wave_size_target = [&]() -> std::uint32_t {
        if (config.wave_size != 0) return config.wave_size;
        std::uint64_t const auto_wave = compute_wave_size_(device_ids, max_atoms_per_molecule, num_conformers, sink);
        std::uint64_t const reserved_ceiling = static_cast<std::uint64_t>(per_device_reserved_molecules) *
                                               std::max<std::size_t>(1u, device_ids.size());
        if (reserved_ceiling > 0 && auto_wave > reserved_ceiling) return static_cast<std::uint32_t>(reserved_ceiling);
        return static_cast<std::uint32_t>(auto_wave);
    }();
    this->logger.trace("generate_conformers: n={} K={} wave={} pipeline={}", num_input, num_conformers,
                       wave_size_target, num_input > wave_size_target * 3 / 2 ? "yes" : "no");

    bool const pipeline = num_input > wave_size_target * 3 / 2;

    // `context_t::context_t` pre-grew the device pool, the pinned staging and the metric scratch
    // to the worst-case wave footprint using `cudaMemGetInfo` + the [10K, 400K] wave clamps, so no
    // mid-run grow can fire on the hot path. The per-launch `try_reserve` calls in
    // `launch_gpu_pipeline` return `unchanged_k` at steady state. The pinned batch slots are the
    // exception: they size themselves from the first wave's `bytes_uploaded`, which is the largest
    // any wave will need because preprocess partitions atom-count descending.

    std::vector<wave_record_t> wave_records;
    auto const t_call_start = std::chrono::high_resolution_clock::now();
    auto host_ms = [&](std::chrono::high_resolution_clock::time_point timestamp) {
        return std::chrono::duration<double, std::milli>(timestamp - t_call_start).count();
    };

    // Launch fans out across GPUs with an OMP parallel-for; each worker enqueues
    // its device's H2D / compute / D2H stream chain. Every CUDA call is
    // stream-async against its per-device stream and reads only from its
    // per-device pinned batch pool, so concurrent launches across devices no
    // longer share a pinned source or a driver-lock contention window.
    auto launch_wave = [&](custd::span<wave_slice_pool_t const> per_device_batches, std::uint32_t wave_base,
                           std::vector<gpu_pipeline_handle_t> &handles_out, std::vector<gpu_phase_ms_t> &timings_out) {
        SF_NVTX_RANGE_COLORED("launch: gpu dispatch", 0x2E86C1);
        // Zero the caller's timing slot before anything else. The pipelined path hands the same
        // two vectors round a 2-slot ring, so a wave that preprocess emptied would otherwise be
        // reported with wave N-2's numbers still sitting in its slot.
        timings_out.assign(num_devices, gpu_phase_ms_t {});

        bool any_molecules = false;
        for (std::uint32_t device_index = 0; device_index < num_devices; ++device_index)
            if (per_device_batches[device_index].num_molecules > 0) {
                any_molecules = true;
                break;
            }
        if (!any_molecules) return;

        // Small-molecule datasets (e.g. GDB13) pack most of a wave into a single atom-count bin,
        // which can exceed the metric-embedding scratch sized at construction from a wave-size
        // estimate. The prepass bails (returns false) on an oversized bin, so grow the scratch to
        // this wave's per-device molecule count first. It is grow-only (a no-op after the first
        // oversized wave) and the buffer is a uint32 per molecule (~KB-MB scale), so this is cheap;
        // done serially here to keep the one-time cudaMalloc out of the per-device launch fan-out.
        for (std::uint32_t device_index = 0; device_index < num_devices; ++device_index)
            if (per_device_batches[device_index].num_molecules > 0)
                ensure_metric_scratch(*this, device_index, per_device_batches[device_index].num_molecules,
                                      max_atoms_per_molecule_k);

        handles_out.resize(num_devices);
#pragma omp parallel for num_threads(num_devices) schedule(static, 1)
        for (std::uint32_t device_index = 0; device_index < num_devices; ++device_index) {
            if (per_device_batches[device_index].num_molecules == 0) continue;
            handles_out[device_index] = launch_gpu_pipeline(
                device_ids[device_index], streams[device_index], metric_scratch[device_index],
                *launch_scratch[device_index], per_device_batches[device_index], launch_configs[device_index],
                num_conformers, config.max_stereo_attempts, max_atoms_per_molecule, config.random_seed, wave_base,
                sort_perm_view, coordinates_out, energies_out, status_out, sink);
            handles_out[device_index].timings_out = &timings_out[device_index];
        }
    };

    // Finalize walks the 8 per-GPU handles sequentially. The first finalize's
    // cudaStreamSynchronize blocks until that stream drains; by the time it returns,
    // the other 7 GPUs have almost always finished their wave too (they started
    // within microseconds of each other and have the same work), so the remaining
    // finalize calls are near-instant.
    auto finalize_wave = [&](std::vector<gpu_pipeline_handle_t> &handles) {
        for (auto &handle : handles)
            if (handle.device_id >= 0) finalize_gpu_pipeline(std::move(handle));
        handles.clear();
    };

    auto finalize_wave_timed = [&](std::vector<gpu_pipeline_handle_t> &handles, std::uint32_t wave_index) {
        auto finalize_start = std::chrono::high_resolution_clock::now();
        finalize_wave(handles);
        auto finalize_end = std::chrono::high_resolution_clock::now();
        if (wave_index < wave_records.size()) {
            wave_records[wave_index].finalize_start_ms = host_ms(finalize_start);
            wave_records[wave_index].finalize_end_ms = host_ms(finalize_end);
            // Pull the per-device timings out of the handles' slots — finalize_wave already invoked
            // finalize_gpu_pipeline which wrote into each timings_out. The slots live in launch_wave's
            // `timings_out` argument which the caller owns; the read-back happens through a separate
            // vector below.
        }
    };

    // Prep a wave's input SMILES into `N` per-device pinned batch pools held
    // on `context_t::batch_slots_per_device[slot]`. Preprocess partitions by
    // atom-count descending and writes each device's pool in atom-count
    // ascending order, so kernels can use the batch metadata directly (no
    // per-device sort in launch).
    //
    // Molecules rejected by the `preprocess-step` (deadline tripped, MMFF94
    // unsupported element, too many atoms, invalid SMILES, ...) never reach
    // the GPU pools — they're recorded in `rejected_buf` as
    // `{slice_local_index, conformer_status_t}` pairs and their status is
    // stamped here, so the `fallback-role` picks them up on the caller side
    // and retries via ETKDGv3 where appropriate.
    std::vector<rejected_mol_t> rejected_buf;
    auto prep_slice_into_slot = [&](std::size_t slot, std::uint32_t begin, std::uint32_t end) {
        SF_NVTX_RANGE_COLORED("prep: rdkit preprocess", 0x8E44AD);
        custd::span<wave_slice_pool_t, max_devices_k> slot_span {batch_slots_per_device[slot].data(), max_devices_k};
        prepare_per_device_batches_into(smiles_view.subspan(begin, end - begin), num_devices, max_atoms_per_molecule,
                                        slot_span, &rejected_buf);
        for (rejected_mol_t const &entry : rejected_buf) {
            std::uint32_t const sorted_index = begin + entry.slice_local_index;
            std::uint32_t const caller_index = sort_perm_view[sorted_index];
            for (std::uint32_t conformer = 0; conformer < num_conformers; ++conformer)
                status_out[static_cast<std::size_t>(caller_index) * num_conformers + conformer] = entry.reason;
        }
        // Fill caller's num_atoms_out for every molecule that survived preprocess.
        // Sorted-slice → caller-unsorted via sort_perm. Invalid / too_many / timeout
        // mols keep the zero-init written at the top of generate_conformers.
        bool const want_extras = !num_atoms_out.empty() || !num_heavy_atoms_out.empty() || !num_bonds_out.empty() ||
                                 !molecular_weight_out.empty();
        if (want_extras) {
            for (std::uint32_t device_index = 0; device_index < num_devices; ++device_index) {
                auto const &batch = batch_slots_per_device[slot][device_index];
                auto const metadata = batch.molecule_metadata();
                auto const input_indices = batch.input_indices();
                for (std::uint32_t molecule = 0; molecule < batch.num_molecules; ++molecule) {
                    std::uint32_t const sorted_index = begin + input_indices[molecule];
                    std::uint32_t const caller_index = sort_perm_view[sorted_index];
                    if (!num_atoms_out.empty())
                        num_atoms_out[caller_index] = static_cast<std::uint8_t>(metadata[molecule].num_atoms);
                    if (!num_heavy_atoms_out.empty())
                        num_heavy_atoms_out[caller_index] = static_cast<std::uint16_t>(
                            metadata[molecule].num_heavy_atoms);
                    if (!num_bonds_out.empty())
                        num_bonds_out[caller_index] = static_cast<std::uint16_t>(metadata[molecule].num_bonds);
                    if (!molecular_weight_out.empty())
                        molecular_weight_out[caller_index] = metadata[molecule].molecular_weight;
                }
            }
        }
    };

    // View helper: first `num_devices` per-device batch pools in a slot.
    auto slot_view = [&](std::size_t slot) -> custd::span<wave_slice_pool_t const> {
        return custd::span<wave_slice_pool_t const>(batch_slots_per_device[slot].data(), num_devices);
    };

    if (!pipeline) {
        wave_records.push_back({.wave_index = 0,
                                .n_mol = num_input,
                                .prep_start_ms = 0,
                                .prep_end_ms = 0,
                                .launch_start_ms = 0,
                                .launch_end_ms = 0,
                                .finalize_start_ms = 0,
                                .finalize_end_ms = 0,
                                .per_device_gpu = {}});
        wave_records.back().prep_start_ms = host_ms(std::chrono::high_resolution_clock::now());
        prep_slice_into_slot(0, 0u, num_input);
        wave_records.back().prep_end_ms = host_ms(std::chrono::high_resolution_clock::now());
        std::vector<gpu_pipeline_handle_t> handles;
        std::vector<gpu_phase_ms_t> timings;
        wave_records.back().launch_start_ms = host_ms(std::chrono::high_resolution_clock::now());
        launch_wave(slot_view(0), /*wave_base=*/0, handles, timings);
        wave_records.back().launch_end_ms = host_ms(std::chrono::high_resolution_clock::now());
        finalize_wave_timed(handles, 0);
        wave_records.back().per_device_gpu = timings;
    }
    else {
        // Pipelined: split input into waves, use the persistent context-level
        // 2-slot ring of per-device pinned pools. Slot `wave % 2` alternates;
        // prep(wave) writes to the opposite slot from the wave still in flight.
        std::vector<wave_range_t> waves;
        for (std::uint32_t wave_start = 0; wave_start < num_input; wave_start += wave_size_target)
            waves.push_back({wave_start, std::min(wave_start + wave_size_target, num_input)});
        wave_records.resize(waves.size());
        for (std::uint32_t wave = 0; wave < waves.size(); ++wave) {
            wave_records[wave].wave_index = wave;
            wave_records[wave].n_mol = waves[wave].end - waves[wave].begin;
        }

        std::array<std::vector<gpu_pipeline_handle_t>, 2> in_flight {};
        std::array<std::vector<gpu_phase_ms_t>, 2> timing_slots {};
        in_flight[0].resize(num_devices);
        in_flight[1].resize(num_devices);

        wave_records[0].prep_start_ms = host_ms(std::chrono::high_resolution_clock::now());
        prep_slice_into_slot(0, waves[0].begin, waves[0].end);
        wave_records[0].prep_end_ms = host_ms(std::chrono::high_resolution_clock::now());
        wave_records[0].launch_start_ms = host_ms(std::chrono::high_resolution_clock::now());
        launch_wave(slot_view(0), waves[0].begin, in_flight[0], timing_slots[0]);
        wave_records[0].launch_end_ms = host_ms(std::chrono::high_resolution_clock::now());

        for (std::size_t wave = 1; wave < waves.size(); ++wave) {
            wave_records[wave].prep_start_ms = host_ms(std::chrono::high_resolution_clock::now());
            prep_slice_into_slot(wave % 2, waves[wave].begin, waves[wave].end);
            wave_records[wave].prep_end_ms = host_ms(std::chrono::high_resolution_clock::now());
            finalize_wave_timed(in_flight[(wave - 1) % 2], static_cast<std::uint32_t>(wave - 1));
            wave_records[wave - 1].per_device_gpu = timing_slots[(wave - 1) % 2];
            wave_records[wave].launch_start_ms = host_ms(std::chrono::high_resolution_clock::now());
            launch_wave(slot_view(wave % 2), waves[wave].begin, in_flight[wave % 2], timing_slots[wave % 2]);
            wave_records[wave].launch_end_ms = host_ms(std::chrono::high_resolution_clock::now());
        }
        std::uint32_t const last_index = static_cast<std::uint32_t>(waves.size() - 1);
        finalize_wave_timed(in_flight[last_index % 2], last_index);
        wave_records[last_index].per_device_gpu = timing_slots[last_index % 2];
    }

    print_wave_timeline_(this->logger, wave_records, host_ms(std::chrono::high_resolution_clock::now()));

    // All output permutation already happened inside `finalize_gpu_pipeline`: each wave's per-GPU sort
    // + the outer SMILES-length sort are folded into one reorder that writes directly into caller_index
    // space. What remains here is a scan of the already-permuted status buffer to derive the aggregate
    // return code.
    bool any_success = false;
    for (auto status : status_out)
        if (status == conformer_status_t::success_k) {
            any_success = true;
            break;
        }

    return any_success ? conformer_status_t::success_k : conformer_status_t::mmff_unconverged_k;
}

} // namespace smallfield::api

#if defined(SMALLFIELD_BUILD_PYTHON)

PYBIND11_MODULE(_smallfield, smallfield_module) {
    smallfield_module.doc() = "SmallField: GPU ETKDG + MMFF94 conformer generation";

    smallfield_module.def("supports_cuda", []() { return true; });

    py::enum_<smallfield::api::conformer_status_t>(smallfield_module, "ConformerStatus")
        .value("success", smallfield::api::conformer_status_t::success_k)
        .value("stereo_failure", smallfield::api::conformer_status_t::stereo_failure_k)
        .value("mmff_unconverged", smallfield::api::conformer_status_t::mmff_unconverged_k)
        .value("invalid_smiles", smallfield::api::conformer_status_t::invalid_smiles_k)
        .value("too_many_atoms", smallfield::api::conformer_status_t::too_many_atoms_k)
        .value("preprocess_timeout", smallfield::api::conformer_status_t::preprocess_timeout_k)
        .value("unsupported_element", smallfield::api::conformer_status_t::unsupported_element_k);

    py::class_<smallfield::api::context_t>(smallfield_module, "Context",
                                           "GPU context managing CUDA streams and cuSOLVER handles.")
        .def(py::init([](std::vector<int> device_ids) {
                 custd::span<int const> span(device_ids.data(), device_ids.size());
                 return std::make_unique<smallfield::api::context_t>(span);
             }),
             py::arg("device_ids") = std::vector<int> {})
        .def(
            "generate_conformers",
            [](smallfield::api::context_t &context, std::vector<std::string> const &smiles_list,
               std::uint32_t num_conformers, std::uint32_t max_atoms_per_molecule, std::uint32_t max_stereo_attempts,
               std::uint64_t random_seed) {
                // Build config.
                smallfield::api::config_t config;
                config.num_conformers = num_conformers;
                config.max_stereo_attempts = max_stereo_attempts;
                config.random_seed = random_seed;

                std::uint32_t const num_mols = static_cast<std::uint32_t>(smiles_list.size());
                std::uint32_t const total_conf = num_mols * num_conformers;
                std::uint32_t const coord_stride = max_atoms_per_molecule * 3;
                std::size_t const total_coords = static_cast<std::size_t>(total_conf) * coord_stride;

                // Allocate output numpy arrays.
                auto coords_np = py::array_t<float>(static_cast<py::ssize_t>(total_coords));
                auto energies_np = py::array_t<float>(static_cast<py::ssize_t>(total_conf));
                auto status_np = py::array_t<std::uint8_t>(static_cast<py::ssize_t>(total_conf));

                // Build string_view span from the input strings.
                std::vector<std::string_view> smiles_views(smiles_list.begin(), smiles_list.end());
                custd::span<std::string_view const> smiles_span(smiles_views);

                // Temporary half-precision buffer.
                std::vector<__half> half_coords(total_coords);
                custd::span<__half> coords_span(half_coords);
                custd::span<float> energies_span(energies_np.mutable_data(), total_conf);

                // Status buffer: use conformer_status_t then copy to uint8_t.
                std::vector<smallfield::api::conformer_status_t> status_buf(total_conf);
                custd::span<smallfield::api::conformer_status_t> status_span(status_buf);

                {
                    // Release the GIL for the duration. This call runs minutes on a production
                    // batch and touches no Python object — every buffer above is already owned by
                    // C++ — so holding the interpreter lock would freeze every other thread in the
                    // process for the whole run.
                    py::gil_scoped_release const unlocked;
                    context.generate_conformers(smiles_span, config, max_atoms_per_molecule, coords_span, energies_span,
                                                status_span);
                }

                // Convert __half coordinates back to float for Python.
                float *coords_ptr = coords_np.mutable_data();
                for (std::size_t element = 0; element < total_coords; ++element)
                    coords_ptr[element] = __half2float(half_coords[element]);

                // Copy status enum values to uint8_t numpy array.
                std::uint8_t *status_ptr = status_np.mutable_data();
                for (std::uint32_t slot = 0; slot < total_conf; ++slot)
                    status_ptr[slot] = static_cast<std::uint8_t>(status_buf[slot]);

                return py::make_tuple(coords_np, energies_np, status_np);
            },
            "Generate conformers for a batch of SMILES strings.\n" //
            "Returns (coordinates, energies, status) as numpy arrays.",
            py::arg("smiles"), py::arg("num_conformers") = 3, py::arg("max_atoms_per_molecule") = 128,
            py::arg("max_stereo_attempts") = 10, py::arg("random_seed") = 42);
}

#endif // SMALLFIELD_BUILD_PYTHON
