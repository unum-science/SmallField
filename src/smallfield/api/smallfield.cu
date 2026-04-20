/**
 *  @file smallfield.cu
 *  @brief Library implementation of the SmallField public API and pybind11 bindings.
 *
 *  Implements context_t (multi-GPU CUDA stream/cuSOLVER management) and
 *  generate_conformers() (CPU preprocessing, wave-based GPU dispatch,
 *  half-precision output). Exposes a minimal Python module at the bottom.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "smallfield/preprocess/scaffold_cache.hpp"

#if defined(SMALLFIELD_BUILD_PYTHON)
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

#include "smallfield/api/batch_pool.hpp"
#include "smallfield/cuda/etkdg_pipeline.cuh"
#include "smallfield/cuda_error.hpp"
#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/cuda/metric_embedding_prepass.cuh"
#include "smallfield/api/smallfield.hpp"

namespace smallfield::api {
using namespace smallfield::cuda;
using namespace smallfield::mmff94;
using namespace smallfield::parameters;
using namespace smallfield::preprocess;
using namespace smallfield::types;

using smallfield::cuda_error_sink_t;

static inline std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

/**
 *  @brief Rough upper bound on the per-molecule pinned-host footprint in a wave.
 *
 *  Used by @ref generate_conformers to pre-size the per-call 2-slot batch-prep
 *  ring so the pinned host buffer never reallocates while a wave's H2D is in
 *  flight.  Overestimates: at @c max_atoms=128 ~91 KB/molecule covers the
 *  bounds matrix, bond-distance byte matrix, MMFF per-atom vdw/charge, and the
 *  variable-length interaction lists plus slack.
 */
static inline std::size_t estimate_per_molecule_batch_bytes(std::uint32_t max_atoms) noexcept {
    std::size_t const n = max_atoms;
    std::size_t const bounds_bytes = n * n * sizeof(float);
    std::size_t const bond_distance_bytes = n * n * sizeof(std::uint8_t);
    std::size_t const per_atom_slack = n * 64u;
    std::size_t const per_mol_fixed = 1024u;
    return bounds_bytes + bond_distance_bytes + per_atom_slack + per_mol_fixed;
}

/**
 *  @brief Persistent per-device scratch for the kernel launch path.
 *
 *  @c device_pool holds the batch-input mirror + per-launch outputs + scratch;
 *  @c host_staging holds the sorted-metadata subset (CPU → GPU) plus pinned
 *  readback regions for coords / energies / status (GPU → CPU).  Both are
 *  grow-only and reused across waves so the single-threaded dispatcher
 *  avoids per-launch allocation churn.
 */
struct launch_scratch_t {
    device_byte_buffer_t device_pool;
    pinned_byte_buffer_t host_staging;
    // Persistent CUDA events for per-phase timing. Created once per device on
    // first launch and reused across every wave. Allocating events inside the
    // hot loop and destroying them after each finalize triggered an intermittent
    // multi-GPU regression (waves 3-4 silently dropped at 240K) — likely a
    // race between cudaEventDestroy on one stream and cudaEventCreate from the
    // next launch's omp parallel-for thread. Persistent reuse sidesteps the
    // problem entirely. cudaEventBlockingSync avoids spin-waiting inside
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
 *  Definition lives below.  @ref context_t "context_t's" ctor pre-warms the
 *  per-device metric scratch at construction time so the first user call
 *  doesn't pay the cusolver workspace probe + `cudaMalloc` cost.
 */
void ensure_metric_scratch(context_t &ctx, std::size_t device_idx, std::size_t max_group_size, std::uint32_t max_atoms);

context_t::context_t(std::span<int const> devices) {
    if (devices.empty()) {
        int count = 0;
        cudaGetDeviceCount(&count);
        for (int i = 0; i < count; ++i) device_ids.push_back(i);
    }
    else { device_ids.assign(devices.begin(), devices.end()); }

    streams.resize(device_ids.size());
    solvers.resize(device_ids.size());
    metric_scratch.resize(device_ids.size());
    launch_scratch.resize(device_ids.size());

    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        cudaSetDevice(device_ids[i]);
        cudaStreamCreate(&streams[i]);
        cusolverDnCreate(&solvers[i]);
        cusolverDnSetStream(solvers[i], streams[i]);
        // Reuse one cusolverDnParams_t per device across all calls. The
        // generic 64-bit batched eigen API (cusolverDnXsyevBatched) takes this
        // instead of the legacy syevjInfo_t — lower per-call CPU overhead.
        cusolverDnCreateParams(&metric_scratch[i].xsyev_params);
        launch_scratch[i] = std::make_unique<launch_scratch_t>();
        // Persistent timing events live with the per-device launch scratch.
        // BlockingSync = host doesn't busy-wait inside cudaStreamSynchronize.
        cudaEventCreateWithFlags(&launch_scratch[i]->event_upload_start, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&launch_scratch[i]->event_metric_start, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&launch_scratch[i]->event_kernels_start, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&launch_scratch[i]->event_download_start, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&launch_scratch[i]->event_end, cudaEventBlockingSync);
    }
    // Pre-warm the dispatcher so the first user call does not pay cold-start costs.
    // Cold wave 0 has "mallocpool" ~100-130 ms
    // per device for the first cudaMalloc + cudaHostAlloc (910 ms total across
    // 8 GPUs per prior nsys capture), plus a ~60 ms cusolver workspace probe
    // inside ensure_metric_scratch. Pre-sizing both at construction moves
    // those costs off the user's hot path. The sizes are conservative
    // first-call defaults: a 5 K-molecule wave with max_atoms=128 covers any
    // small-batch workload; bigger batches grow the pool on demand via
    // ensure_capacity (which is already the existing growth path).
    constexpr std::size_t kPrewarmGroup = 5'000;
    constexpr std::uint32_t kPrewarmAtoms = 128;
    constexpr std::size_t kPrewarmDevicePoolBytes = 256ULL << 20;  // 256 MB
    constexpr std::size_t kPrewarmHostStagingBytes = 256ULL << 20; // 256 MB
    // Set the max-dynamic-shared-memory attribute on `embed_4d_power_iter_kernel`
    // ONCE per device at context construction time.  At bin_n=128 the kernel
    // needs ~67 KB shared memory, above the 48 KB hardware default.  Calling
    // this per-wave per-device (the old path) interleaved a host-API call with
    // in-flight async kernel launches on sibling devices, which correlated
    // with a Blackwell SM 12.0 multi-GPU illegal-memory-access.
    constexpr std::uint32_t max_bin_n = 128;
    constexpr std::size_t max_shared_bytes =
        static_cast<std::size_t>(max_bin_n * max_bin_n + max_bin_n + 1 + 4 * max_bin_n + max_bin_n + 4 + 1) *
        sizeof(float);
    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        cudaSetDevice(device_ids[i]);
        // Probes cusolver + allocates metric_scratch.pool. Sets max_group_size,
        // max_atoms, max_device_workspace_bytes so subsequent calls with ≤ these
        // dimensions skip the slow probe.
        ensure_metric_scratch(*this, i, kPrewarmGroup, kPrewarmAtoms);
        // Pre-grow the per-launch device + host staging pools so the first
        // launch's cudaMalloc / cudaHostAlloc doesn't stall the hot path.
        launch_scratch[i]->device_pool.ensure_capacity(kPrewarmDevicePoolBytes);
        launch_scratch[i]->host_staging.ensure_capacity(kPrewarmHostStagingBytes);
        // Set the shared-memory limit on this device's copy of
        // `embed_4d_power_iter_kernel` — stays in effect for the context's
        // lifetime so nothing in the hot path mutates it.
        cudaFuncSetAttribute(smallfield::cuda::embed_4d_power_iter_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(max_shared_bytes));
    }
}

context_t::~context_t() {
    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        cudaSetDevice(device_ids[i]);
        if (metric_scratch[i].xsyev_params) cusolverDnDestroyParams(metric_scratch[i].xsyev_params);
        if (launch_scratch[i]) {
            if (launch_scratch[i]->event_upload_start) cudaEventDestroy(launch_scratch[i]->event_upload_start);
            if (launch_scratch[i]->event_metric_start) cudaEventDestroy(launch_scratch[i]->event_metric_start);
            if (launch_scratch[i]->event_kernels_start) cudaEventDestroy(launch_scratch[i]->event_kernels_start);
            if (launch_scratch[i]->event_download_start) cudaEventDestroy(launch_scratch[i]->event_download_start);
            if (launch_scratch[i]->event_end) cudaEventDestroy(launch_scratch[i]->event_end);
        }
        launch_scratch[i].reset(); // device_byte_buffer_t + pinned_byte_buffer_t free their holdings
        cusolverDnDestroy(solvers[i]);
        cudaStreamDestroy(streams[i]);
    }
    scaffold_cache.reset(); // no-op if null
}

void context_t::enable_scaffold_cache() {
    if (!scaffold_cache) scaffold_cache = std::make_unique<scaffold_cache_t>();
}

scaffold_cache_stats_t context_t::get_scaffold_cache_stats() const {
    if (!scaffold_cache) return {};
    return scaffold_cache->stats();
}

/**
 *  @brief Ensure the per-device metric scratch pool is sized for the given
 *  (max_group_size, max_atoms) combination.  Grows only — never shrinks — so
 *  the first wave of a workload pays the allocation cost and subsequent
 *  waves reuse it.
 */
void ensure_metric_scratch(context_t &ctx, std::size_t device_idx, std::size_t max_group_size,
                           std::uint32_t max_atoms) {
    cuda_error_sink_t const sink {&ctx.logger, &ctx.cuda_error_count};
    auto &s = ctx.metric_scratch[device_idx];
    if (max_group_size <= s.max_group_size && max_atoms <= s.max_atoms && s.pool.data() != nullptr) return;

    // Grow to at least the new request, but round up a bit to avoid thrashing
    // across similarly-sized waves.
    std::size_t const new_group = std::max(s.max_group_size, max_group_size);
    std::uint32_t const new_atoms = std::max<std::uint32_t>(s.max_atoms, max_atoms);

    cudaSetDevice(ctx.device_ids[device_idx]);

    // Probe cusolver for the worst-case workspace size at the new dimensions.
    // Probe cusolverDnXsyevBatched for the worst-case device + host workspace sizes.
    // Once per growth, not per wave.
    std::size_t new_dev_ws_bytes = 0;
    std::size_t new_host_ws_bytes = 0;
    {
        float *probe_gram = nullptr;
        float *probe_eig = nullptr;
        sink.check(cudaMalloc(&probe_gram, sizeof(float)), "cudaMalloc(probe_gram)");
        sink.check(cudaMalloc(&probe_eig, sizeof(float)), "cudaMalloc(probe_eig)");
        cusolverDnXsyevBatched_bufferSize(
            ctx.solvers[device_idx], s.xsyev_params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
            static_cast<int64_t>(new_atoms), CUDA_R_32F, probe_gram, static_cast<int64_t>(new_atoms), CUDA_R_32F,
            probe_eig, CUDA_R_32F, &new_dev_ws_bytes, &new_host_ws_bytes, static_cast<int64_t>(new_group));
        sink.check(cudaFree(probe_gram), "cudaFree(probe_gram)");
        sink.check(cudaFree(probe_eig), "cudaFree(probe_eig)");
    }

    // Compute pool layout with 256-byte alignment.
    std::size_t const group_bytes = align_up(new_group * sizeof(std::uint32_t), 256);
    std::size_t const gram_bytes = align_up(new_group * static_cast<std::size_t>(new_atoms) * new_atoms * sizeof(float),
                                            256);
    std::size_t const eig_bytes = align_up(new_group * new_atoms * sizeof(float), 256);
    std::size_t const info_bytes = align_up(new_group * sizeof(int), 256);
    std::size_t const ws_bytes = align_up(new_dev_ws_bytes, 256);
    std::size_t const total = group_bytes + gram_bytes + eig_bytes + info_bytes + ws_bytes;

    if (!s.pool.ensure_capacity(total)) {
        ctx.logger.error("ensure_metric_scratch: device allocation ({} bytes) failed", total);
        return;
    }
    std::size_t off = 0;
    auto *base = reinterpret_cast<char *>(s.pool.data());
    s.group_indices = {reinterpret_cast<std::uint32_t *>(base + off), new_group};
    off += group_bytes;
    s.gram_matrices = {reinterpret_cast<float *>(base + off),
                       new_group * static_cast<std::size_t>(new_atoms) * new_atoms};
    off += gram_bytes;
    s.eigenvalues = {reinterpret_cast<float *>(base + off), new_group * new_atoms};
    off += eig_bytes;
    s.info = {reinterpret_cast<int *>(base + off), new_group};
    off += info_bytes;
    // Device workspace as a std::span<float> over raw bytes — cusolver just needs a
    // pointer + bytes, so we store the float pointer but the true capacity is
    // @c max_device_workspace_bytes.
    s.workspace = {reinterpret_cast<float *>(base + off), new_dev_ws_bytes / sizeof(float)};
    s.host_workspace.resize(new_host_ws_bytes);
    s.max_group_size = new_group;
    s.max_atoms = new_atoms;
    s.max_device_workspace_bytes = new_dev_ws_bytes;
    s.max_host_workspace_bytes = new_host_ws_bytes;
}

/**
 *  @brief Per-wave, per-GPU stream-local phase timings, populated in
 *  @ref finalize_gpu_pipeline after the stream drains.  All values are in
 *  milliseconds and are measured by @c cudaEventElapsedTime on events that
 *  were recorded into the device's stream — they reflect actual GPU + DMA
 *  wall, not host-side scheduling.  Aggregator in @ref generate_conformers
 *  combines these across devices to derive per-wave GPU wall (max) and
 *  total GPU work (sum).
 */
struct gpu_phase_ms_t {
    float upload_ms = 0.f;   ///< H2D batch + metadata copies
    float metric_ms = 0.f;   ///< @c build_metric_initial_coordinates_on_gpu (embed kernel)
    float kernels_ms = 0.f;  ///< wave-binned @c etkdg_pipeline_warp_kernel launches
    float download_ms = 0.f; ///< D2H coords/energies/status copies
    float total_ms = 0.f;    ///< upload+metric+kernels+download (all on the same stream)
};

/**
 *  @brief Per-wave, per-GPU state carried from `launch_gpu_pipeline` to `finalize_gpu_pipeline`.
 *
 *  Holds the device memory pool that stays alive until the stream drains, the host
 *  staging buffers that the async download writes into, and the metadata needed to
 *  unsort results into the caller's output arrays.  Move-only by design.
 */
struct gpu_pipeline_handle_t {
    int device_id = -1;
    cudaStream_t stream = nullptr;
    batch_pool_t const *batch = nullptr; ///< non-owning — lives in the caller's ring buffer
    launch_scratch_t *scratch = nullptr; ///< non-owning — persistent in context_t
    std::uint32_t local_count = 0;
    std::uint32_t num_conformers = 0;
    std::uint32_t max_atoms_per_molecule = 0;
    std::uint32_t wave_base = 0; ///< offset into caller's output (sorted space)
    std::vector<std::uint32_t> order;
    std::vector<std::uint32_t> local_indices_copy;
    // Device-side output spans into scratch->device_pool.
    std::span<float> device_coords_fp32 {};  ///< kernel output (fp32)
    std::span<__half> device_coords_fp16 {}; ///< post-convert coords for D2H
    std::span<float> device_energies_out {};
    std::span<std::uint8_t> device_status_out {};
    // Pinned host-side readback spans into scratch->host_staging; finalize drains
    // the stream then permutes from these into the caller's output slice.
    std::span<__half> host_coords_sorted {};
    std::span<float> host_energies_sorted {};
    std::span<std::uint8_t> host_status_sorted {};
    // Caller-owned output slices (non-owning).
    std::span<__half> host_coords_out {};
    std::span<float> host_energies_out {};
    std::span<conformer_status_t> host_status_out {};
    // Inverse SMILES-length sort. Empty = no extra unsort (non-pipelined).
    std::span<std::uint32_t const> sort_perm {};
    // Stream-local phase timing events — recorded in launch, consumed in finalize.
    cudaEvent_t event_upload_start = nullptr;
    cudaEvent_t event_metric_start = nullptr;
    cudaEvent_t event_kernels_start = nullptr;
    cudaEvent_t event_download_start = nullptr;
    cudaEvent_t event_end = nullptr;
    gpu_phase_ms_t *timings_out = nullptr; ///< caller-owned slot; finalize writes per-phase ms
    cuda_error_sink_t sink {};             ///< routes errors through caller's logger + counter
};

/**
 *  @brief Offsets + sizes for every region carved from `launch_scratch_t`'s
 *  device pool and pinned host staging, for one `embedding-step`.
 *
 *  Device pool regions, in order:
 *    - batch-input mirror (target of the single H2D)
 *    - sorted metadata subset (tiny H2D from pinned host staging)
 *    - coords_fp32 (kernel output, fp32)
 *    - coords_fp16 (post-convert, D2H source; halves DMA bandwidth)
 *    - energies_out (fp32)
 *    - status_out (u8)
 *    - 4D metric-embedding scratch (GPU-only, consumed by the prepass)
 *  Host staging regions, in order:
 *    - sorted metadata (CPU-written, source of the H2D)
 *    - coords_fp16 / energies / status (pinned D2H destinations)
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

    std::size_t host_off_sorted_meta;
    std::size_t host_off_coords;
    std::size_t host_off_energies;
    std::size_t host_off_status;
    std::size_t host_pool_bytes;
};

/** @brief Compute every offset and size for the launch pool given this dispatch's shape. */
static pipeline_pool_layout_t compute_pipeline_pool_layout_(std::size_t batch_bytes_total, std::uint32_t local_count,
                                                            std::uint32_t num_conformers,
                                                            std::uint32_t max_atoms_per_molecule) {
    auto const sizeof_meta = sizeof(etkdg_molecule_metadata_t);
    std::size_t const total_conf = static_cast<std::size_t>(local_count) * num_conformers;
    std::size_t const total_coords = total_conf * max_atoms_per_molecule * 3;

    pipeline_pool_layout_t layout {};
    std::size_t device_offset = 0;
    layout.device_off_batch = align_up(device_offset, 256);
    device_offset = layout.device_off_batch + align_up(batch_bytes_total, 256);
    layout.device_off_sorted_meta = device_offset;
    device_offset += align_up(local_count * sizeof_meta, 256);
    layout.device_off_coords_fp32 = device_offset;
    device_offset += align_up(total_coords * sizeof(float), 256);
    layout.device_off_coords_fp16 = device_offset;
    device_offset += align_up(total_coords * sizeof(__half), 256);
    layout.device_off_energies = device_offset;
    device_offset += align_up(total_conf * sizeof(float), 256);
    layout.device_off_status = device_offset;
    device_offset += align_up(total_conf * sizeof(std::uint8_t), 256);
    layout.device_off_initial = device_offset;
    device_offset += align_up(static_cast<std::size_t>(local_count) * max_atoms_per_molecule * 4 * sizeof(float), 256);
    layout.device_pool_bytes = device_offset;

    std::size_t host_offset = 0;
    layout.host_off_sorted_meta = align_up(host_offset, 256);
    host_offset = layout.host_off_sorted_meta + align_up(local_count * sizeof_meta, 256);
    layout.host_off_coords = host_offset;
    host_offset += align_up(total_coords * sizeof(__half), 256);
    layout.host_off_energies = host_offset;
    host_offset += align_up(total_conf * sizeof(float), 256);
    layout.host_off_status = host_offset;
    host_offset += align_up(total_conf * sizeof(std::uint8_t), 256);
    layout.host_pool_bytes = host_offset;
    return layout;
}

/**
 *  @brief Enqueue all upload, metric prepass, kernel, and download operations on the
 *  GPU stream for one `embedding-wave`'s GPU slice; return immediately.
 *  Caller eventually calls `finalize_gpu_pipeline` with the returned handle.
 */
gpu_pipeline_handle_t launch_gpu_pipeline(int device_id, cudaStream_t stream, cusolverDnHandle_t solver_handle,
                                          metric_scratch_t &scratch, launch_scratch_t &launch_buffers,
                                          batch_pool_t const &batch, std::span<std::uint32_t const> local_indices,
                                          std::uint32_t num_conformers, std::uint32_t max_stereo_attempts,
                                          std::uint32_t max_atoms_per_molecule, std::uint64_t seed,
                                          std::uint32_t wave_base, std::span<std::uint32_t const> sort_perm,
                                          std::span<__half> host_coords_out, std::span<float> host_energies_out,
                                          std::span<conformer_status_t> host_status_out, cuda_error_sink_t sink) {

    sink.check(cudaSetDevice(device_id), "cudaSetDevice(launch_gpu_pipeline)");

    std::uint32_t const local_count = static_cast<std::uint32_t>(local_indices.size());

    // Sort this GPU's molecules by atom count to feed `atom-limited-bin` dispatch.
    auto meta_full = batch.molecule_metadata();
    std::vector<std::uint32_t> order(local_count);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        return meta_full[local_indices[a]].num_atoms < meta_full[local_indices[b]].num_atoms;
    });

    std::size_t const total_conf = static_cast<std::size_t>(local_count) * num_conformers;
    std::size_t const total_coords = total_conf * max_atoms_per_molecule * 3;
    pipeline_pool_layout_t const layout =
        compute_pipeline_pool_layout_(batch.layout.bytes_total, local_count, num_conformers, max_atoms_per_molecule);
    std::size_t const device_off_batch = layout.device_off_batch;
    std::size_t const device_off_sorted_meta = layout.device_off_sorted_meta;
    std::size_t const device_off_coords_fp32 = layout.device_off_coords_fp32;
    std::size_t const device_off_coords_fp16 = layout.device_off_coords_fp16;
    std::size_t const device_off_energies = layout.device_off_energies;
    std::size_t const device_off_status = layout.device_off_status;
    std::size_t const device_off_initial = layout.device_off_initial;
    std::size_t const device_pool_bytes = layout.device_pool_bytes;
    std::size_t const host_off_sorted_meta = layout.host_off_sorted_meta;
    std::size_t const host_off_coords = layout.host_off_coords;
    std::size_t const host_off_energies = layout.host_off_energies;
    std::size_t const host_off_status = layout.host_off_status;
    std::size_t const host_pool_bytes = layout.host_pool_bytes;
    auto const sizeof_meta = sizeof(etkdg_molecule_metadata_t);

    gpu_pipeline_handle_t handle;
    handle.device_id = device_id;
    handle.stream = stream;
    handle.batch = &batch;
    handle.scratch = &launch_buffers;
    handle.local_count = local_count;
    handle.num_conformers = num_conformers;
    handle.max_atoms_per_molecule = max_atoms_per_molecule;
    handle.wave_base = wave_base;
    handle.order = std::move(order);
    handle.local_indices_copy.assign(local_indices.begin(), local_indices.end());
    handle.host_coords_out = host_coords_out;
    handle.host_energies_out = host_energies_out;
    handle.host_status_out = host_status_out;
    handle.sort_perm = sort_perm;
    handle.sink = sink;

    if (!launch_buffers.device_pool.ensure_capacity(device_pool_bytes)) {
        if (sink.log != nullptr)
            sink.log->error("launch_gpu_pipeline: cudaMalloc({} bytes) failed on gpu {}", device_pool_bytes, device_id);
        return handle;
    }
    if (!launch_buffers.host_staging.ensure_capacity(host_pool_bytes)) {
        if (sink.log != nullptr)
            sink.log->error("launch_gpu_pipeline: cudaHostAlloc({} bytes) failed on gpu {}", host_pool_bytes,
                            device_id);
        return handle;
    }
    // Reuse the persistent per-device events from launch_scratch_t. These were
    // allocated once at context construction and live until context destruction;
    // re-recording them here is safe because the prior wave's
    // cudaStreamSynchronize in finalize_gpu_pipeline drained their previous use.
    handle.event_upload_start = launch_buffers.event_upload_start;
    handle.event_metric_start = launch_buffers.event_metric_start;
    handle.event_kernels_start = launch_buffers.event_kernels_start;
    handle.event_download_start = launch_buffers.event_download_start;
    handle.event_end = launch_buffers.event_end;

    std::byte *const device_base = launch_buffers.device_pool.data();
    std::byte *const host_base = launch_buffers.host_staging.data();

    // Device view over the batch-input mirror — same layout as batch.host_view().
    batch_pool_dev_view_t device_view {device_base + device_off_batch, &batch.layout};

    auto *device_metadata_sorted = reinterpret_cast<etkdg_molecule_metadata_t *>(device_base + device_off_sorted_meta);
    handle.device_coords_fp32 = {reinterpret_cast<float *>(device_base + device_off_coords_fp32), total_coords};
    handle.device_coords_fp16 = {reinterpret_cast<__half *>(device_base + device_off_coords_fp16), total_coords};
    handle.device_energies_out = {reinterpret_cast<float *>(device_base + device_off_energies), total_conf};
    handle.device_status_out = {reinterpret_cast<std::uint8_t *>(device_base + device_off_status), total_conf};
    auto *device_initial_coordinates = reinterpret_cast<float *>(device_base + device_off_initial);

    auto *host_sorted_meta = reinterpret_cast<etkdg_molecule_metadata_t *>(host_base + host_off_sorted_meta);
    handle.host_coords_sorted = {reinterpret_cast<__half *>(host_base + host_off_coords), total_coords};
    handle.host_energies_sorted = {reinterpret_cast<float *>(host_base + host_off_energies), total_conf};
    handle.host_status_sorted = {reinterpret_cast<std::uint8_t *>(host_base + host_off_status), total_conf};

    // Gather the sorted metadata directly into the pinned staging region so the
    // subsequent H2D does not degrade to synchronous.
    for (std::uint32_t i = 0; i < local_count; ++i) host_sorted_meta[i] = meta_full[local_indices[handle.order[i]]];

    // Replaces 14 pageable-source cudaMemcpyAsync calls with ONE pinned-source
    // transfer that actually returns immediately and runs on the stream.
    cudaEventRecord(handle.event_upload_start, stream);
    if (batch.layout.bytes_total > 0)
        sink.check(cudaMemcpyAsync(device_base + device_off_batch, batch.host.data(), batch.layout.bytes_total,
                                   cudaMemcpyHostToDevice, stream),
                   "H2D batch pool");
    if (local_count > 0)
        sink.check(cudaMemcpyAsync(device_metadata_sorted, host_sorted_meta, local_count * sizeof_meta,
                                   cudaMemcpyHostToDevice, stream),
                   "H2D sorted metadata");

    cudaEventRecord(handle.event_metric_start, stream);
    build_metric_initial_coordinates_on_gpu(device_view.bounds_matrices().data(), device_metadata_sorted,
                                            std::span<etkdg_molecule_metadata_t const> {host_sorted_meta, local_count},
                                            max_atoms_per_molecule, device_initial_coordinates, solver_handle, scratch,
                                            sink, stream);
    cudaEventRecord(handle.event_kernels_start, stream);

    // Atom-limited-bin kernel dispatch (8 caps: 16, 32, 48, 64, 80, 96, 112, 128).
    constexpr std::uint32_t atom_limited_bin_caps[] = {16, 32, 48, 64, 80, 96, 112, 128};
    constexpr std::uint32_t num_bins = 8;

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);

    std::uint32_t bin_start = 0;
    for (std::uint32_t bin = 0; bin < num_bins && bin_start < local_count; ++bin) {
        std::uint32_t bin_end = bin_start;
        while (bin_end < local_count && host_sorted_meta[bin_end].num_atoms <= atom_limited_bin_caps[bin]) ++bin_end;

        if (bin_end == bin_start) continue;

        std::uint32_t const bin_size = bin_end - bin_start;
        std::uint32_t const bin_max_atoms = atom_limited_bin_caps[bin];

        // Kernel 1 — DG + stereo retry + fourth-dim crush + ETK torsion refinement.
        // The per-bin dynamic shared-memory requirement is always ≤ 22.5 KB at
        // bin_max_atoms=128 (16 KB int8 bounds + 6 KB FIRE scratch), well under
        // the 48 KB default.  No `cudaFuncSetAttribute` call needed; skipping it
        // eliminates a per-bin host-API round-trip that, when called mid-stream
        // during the multi-wave async pipeline, correlated with an illegal
        // memory access on Blackwell SM 12.0.
        std::uint32_t const shared_bytes_dg = compute_pipeline_shared_memory_bytes(bin_max_atoms, 0);

        int blocks_per_sm_dg = 0;
        sink.check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm_dg, etkdg_pipeline_warp_kernel, 32,
                                                                 shared_bytes_dg),
                   "cudaOccupancyMaxActiveBlocksPerMultiprocessor(etkdg)");
        std::uint32_t const grid_size_dg = std::min(bin_size, static_cast<std::uint32_t>(blocks_per_sm_dg * num_sms));

        etkdg_pipeline_warp_kernel<<<grid_size_dg, 32, shared_bytes_dg, stream>>>(
            device_view.bounds_matrices().data(), device_initial_coordinates + bin_start * max_atoms_per_molecule * 4,
            device_view.chiral_centers().data(), device_view.etk_torsions().data(),
            device_view.distance_constraints().data(), device_view.etk_interactions().data(),
            device_view.etk_atom_offsets().data(), device_metadata_sorted + bin_start,
            handle.device_coords_fp32.data() + bin_start * num_conformers * max_atoms_per_molecule * 3,
            handle.device_energies_out.data() + bin_start * num_conformers,
            handle.device_status_out.data() + bin_start * num_conformers, bin_size, num_conformers,
            max_stereo_attempts, bin_max_atoms, max_atoms_per_molecule, max_atoms_per_molecule, seed);
        sink.check(cudaPeekAtLastError(), "etkdg_pipeline_warp_kernel launch");

        // Kernel 2 — MMFF94 polish in place on the per-conformer coordinates
        // produced by Kernel 1.  Skips slots whose status is non-zero.
        // Shared-memory footprint is 3 × 3 × bin_max_atoms × sizeof(float) ≤
        // 4.6 KB at bin_max_atoms=128 — under the 48 KB default, so no
        // `cudaFuncSetAttribute` call is needed (same rationale as the etkdg
        // kernel above).
        std::uint32_t const shared_bytes_mmff = compute_mmff_polish_shared_memory_bytes(bin_max_atoms);

        int blocks_per_sm_mmff = 0;
        sink.check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm_mmff, mmff_polish_warp_kernel, 32,
                                                                 shared_bytes_mmff),
                   "cudaOccupancyMaxActiveBlocksPerMultiprocessor(mmff)");
        std::uint32_t const grid_size_mmff = std::min(bin_size,
                                                      static_cast<std::uint32_t>(blocks_per_sm_mmff * num_sms));

        mmff_polish_warp_kernel<<<grid_size_mmff, 32, shared_bytes_mmff, stream>>>(
            device_view.mmff_atom_vdw().data(), device_view.mmff_atom_charge().data(),
            device_view.bond_distance().data(), device_view.mmff_bonds().data(), device_view.mmff_angles().data(),
            device_view.mmff_torsions().data(), device_view.mmff_oops().data(), device_view.mmff_stretchbends().data(),
            device_view.mmff_interactions().data(), device_view.mmff_atom_offsets().data(),
            device_metadata_sorted + bin_start,
            handle.device_coords_fp32.data() + bin_start * num_conformers * max_atoms_per_molecule * 3,
            handle.device_energies_out.data() + bin_start * num_conformers,
            handle.device_status_out.data() + bin_start * num_conformers, bin_size, num_conformers,
            max_atoms_per_molecule);
        sink.check(cudaPeekAtLastError(), "mmff_polish_warp_kernel launch");

        bin_start = bin_end;
    }

    // Post-kernel fp32→fp16 conversion of the coord buffer, on-device, before D2H.
    // Halves the D2H bandwidth for what is by far the largest transfer.
    if (total_coords > 0) {
        constexpr std::uint32_t convert_block = 256;
        std::uint32_t const num_elements = static_cast<std::uint32_t>(total_coords);
        std::uint32_t const convert_grid = (num_elements + convert_block - 1) / convert_block;
        convert_coords_fp32_to_fp16_kernel<<<convert_grid, convert_block, 0, stream>>>(
            handle.device_coords_fp32.data(), handle.device_coords_fp16.data(), num_elements);
        sink.check(cudaPeekAtLastError(), "convert_coords_fp32_to_fp16_kernel launch");
    }

    // Enqueue pinned D2H for all three output arrays. Pinned destination means
    // these return immediately and the transfers run on the stream after the
    // kernels. `finalize_gpu_pipeline` just waits on the stream and permutes.
    cudaEventRecord(handle.event_download_start, stream);
    if (total_coords > 0)
        sink.check(cudaMemcpyAsync(handle.host_coords_sorted.data(), handle.device_coords_fp16.data(),
                                   total_coords * sizeof(__half), cudaMemcpyDeviceToHost, stream),
                   "D2H coords_fp16");
    if (total_conf > 0) {
        sink.check(cudaMemcpyAsync(handle.host_energies_sorted.data(), handle.device_energies_out.data(),
                                   total_conf * sizeof(float), cudaMemcpyDeviceToHost, stream),
                   "D2H energies");
        sink.check(cudaMemcpyAsync(handle.host_status_sorted.data(), handle.device_status_out.data(),
                                   total_conf * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream),
                   "D2H status");
    }
    cudaEventRecord(handle.event_end, stream);
    return handle;
}

/**
 *  @brief Wait for a previously-launched wave to finish and permute results into
 *  the caller's output arrays.  All three D2H transfers were already enqueued by
 *  `launch_gpu_pipeline` into pinned host staging; this just drains the stream
 *  and does a permutation memcpy.
 */
void finalize_gpu_pipeline(gpu_pipeline_handle_t handle) {
    handle.sink.check(cudaSetDevice(handle.device_id), "cudaSetDevice(finalize_gpu_pipeline)");
    handle.sink.check(cudaStreamSynchronize(handle.stream), "cudaStreamSynchronize(finalize)");

    // Compute stream-local phase elapsed times — events are persistent in the
    // launch_scratch_t so no destroy here. All ignored failures (events not
    // recorded, etc.) leave the timing slot at its default zero.
    if (handle.timings_out != nullptr && handle.event_end != nullptr) {
        cudaEventElapsedTime(&handle.timings_out->upload_ms, handle.event_upload_start, handle.event_metric_start);
        cudaEventElapsedTime(&handle.timings_out->metric_ms, handle.event_metric_start, handle.event_kernels_start);
        cudaEventElapsedTime(&handle.timings_out->kernels_ms, handle.event_kernels_start, handle.event_download_start);
        cudaEventElapsedTime(&handle.timings_out->download_ms, handle.event_download_start, handle.event_end);
        cudaEventElapsedTime(&handle.timings_out->total_ms, handle.event_upload_start, handle.event_end);
    }

    // Single reorder: fold the per-GPU atom-count sort AND the outer
    // descending-SMILES-length sort into one permutation memcpy.
    auto input_indices = handle.batch->input_indices();
    std::uint32_t const coord_stride = handle.max_atoms_per_molecule * 3;
    std::uint32_t const num_conformers = handle.num_conformers;
    bool const has_perm = !handle.sort_perm.empty();
    for (std::uint32_t sorted_idx = 0; sorted_idx < handle.local_count; ++sorted_idx) {
        std::uint32_t const local_idx = handle.order[sorted_idx];
        std::uint32_t const orig_idx = handle.local_indices_copy[local_idx];
        std::uint32_t const input_idx = input_indices[orig_idx];
        std::uint32_t const caller_idx = has_perm ? handle.sort_perm[handle.wave_base + input_idx] : input_idx;
        for (std::uint32_t c = 0; c < num_conformers; ++c) {
            std::uint32_t const src_slot = sorted_idx * num_conformers + c;
            std::uint32_t const dst_slot = caller_idx * num_conformers + c;
            handle.host_energies_out[dst_slot] = handle.host_energies_sorted[src_slot];
            std::uint8_t const raw = handle.host_status_sorted[src_slot];
            handle.host_status_out[dst_slot] = raw == 0   ? conformer_status_t::success
                                               : raw == 1 ? conformer_status_t::stereo_failure
                                                          : conformer_status_t::mmff_unconverged;
            std::memcpy(handle.host_coords_out.data() + static_cast<std::size_t>(dst_slot) * coord_stride,
                        handle.host_coords_sorted.data() + static_cast<std::size_t>(src_slot) * coord_stride,
                        coord_stride * sizeof(__half));
        }
    }
}

/**
 *  @brief Per-`embedding-wave` timing accumulator populated by `launch_wave`
 *  and `finalize_wave`.  Pretty-printed by @ref print_wave_timeline_ at the
 *  end of @ref generate_conformers when `logger.verbose_level >= 1`.
 */
struct wave_record_t {
    std::uint32_t wave_idx;
    std::uint32_t n_mol;
    double prep_start_ms;     ///< host wall, ms relative to call start
    double prep_end_ms;       ///< "
    double launch_start_ms;   ///< "  (when launch_wave entered)
    double launch_end_ms;     ///< "  (when launch_wave returned)
    double finalize_start_ms; ///< "  (when finalize_wave entered for this wave)
    double finalize_end_ms;   ///< "
    std::vector<gpu_phase_ms_t> per_device_gpu;
};

/**
 *  @brief Compute the target `embedding-wave` size in molecules.
 *
 *  Honours `config.wave_size` when non-zero; otherwise auto-sizes against the
 *  smallest visible GPU's free VRAM (75 % budget, 6× headroom on top of the
 *  per-molecule mirror + outputs estimate).  Clamps to [10 K, 400 K] and caps
 *  per-GPU group size at 10 K so the cusolver batched-eigen workspace stays
 *  reasonable at N=128 (~1 MB/molecule at that bin).
 */
static std::uint32_t compute_wave_size_(context_t const &ctx, config_t const &config,
                                        std::uint32_t max_atoms_per_molecule, cuda_error_sink_t const &sink) {
    if (config.wave_size != 0) return config.wave_size;

    std::uint32_t const num_devices = static_cast<std::uint32_t>(ctx.device_ids.size());
    std::uint32_t const num_conformers = config.num_conformers;

    std::size_t smallest_free = 0;
    std::size_t smallest_total = 0;
    bool have_any = false;
    for (std::size_t d = 0; d < ctx.device_ids.size(); ++d) {
        if (!sink.check(cudaSetDevice(ctx.device_ids[d]), "cudaSetDevice(memgetinfo)")) continue;
        std::size_t free_b = 0, total_b = 0;
        if (!sink.check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo")) continue;
        if (!have_any || free_b < smallest_free) smallest_free = free_b;
        if (!have_any || total_b < smallest_total) smallest_total = total_b;
        have_any = true;
    }
    // Leave 25 % of the free pool untouched for cusolver + peak spikes.
    std::size_t const budget_bytes = have_any ? smallest_free * 3u / 4u
                                              : (smallest_total > 0 ? smallest_total / 2u : 0u);

    std::size_t const bytes_per_molecule =
        estimate_per_molecule_batch_bytes(max_atoms_per_molecule) +
        static_cast<std::size_t>(max_atoms_per_molecule) * num_conformers * 3u * sizeof(float) * 6u;

    std::uint64_t auto_wave = 50'000;
    if (bytes_per_molecule > 0 && budget_bytes > 0) auto_wave = budget_bytes / bytes_per_molecule;
    if (auto_wave < 10'000u) auto_wave = 10'000u;
    if (auto_wave > 400'000u) auto_wave = 400'000u;

    // Cusolver batched eigen workspace in ensure_metric_scratch is the dominant
    // per-GPU allocation (~1 MB/molecule at N=128).  Cap per-GPU group size at
    // 10 K so even a 1-GPU run has headroom.  Matches the v8 working density.
    constexpr std::uint64_t per_gpu_wave_cap = 10'000;
    std::uint64_t const total_cap_from_cusolver = per_gpu_wave_cap * std::max<std::uint64_t>(num_devices, 1u);
    if (auto_wave > total_cap_from_cusolver) auto_wave = total_cap_from_cusolver;
    return static_cast<std::uint32_t>(auto_wave);
}

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
static void sort_input_by_smiles_length_(std::span<std::string_view const> smiles,
                                         std::vector<std::uint32_t> &sort_perm,
                                         std::vector<std::string_view> &sorted_smiles_storage) {
    std::uint32_t const num_input = static_cast<std::uint32_t>(smiles.size());
    sort_perm.resize(num_input);
    std::iota(sort_perm.begin(), sort_perm.end(), 0u);
    std::sort(sort_perm.begin(), sort_perm.end(),
              [smiles](std::uint32_t a, std::uint32_t b) { return smiles[a].size() > smiles[b].size(); });
    sorted_smiles_storage.resize(num_input);
    for (std::uint32_t k = 0; k < num_input; ++k) sorted_smiles_storage[k] = smiles[sort_perm[k]];
}

/**
 *  @brief Verbose-tier-1 per-wave summary + verbose-tier-2 per-device
 *  breakdown.  No-op when `logger.verbose_level < 1`.
 */
static void print_wave_timeline_(logger_t const &logger, std::vector<wave_record_t> const &wave_records,
                                 double total_wall_ms) {
    if (logger.verbose_level < 1) return;
    logger.trace("wave   mols     prep   launch  finalize    gpu avg /    max   upload / metric / kernels / "
                 "download");
    for (auto const &cr : wave_records) {
        float gpu_max = 0.f, gpu_sum = 0.f, upload = 0.f, metric = 0.f, kernels = 0.f, download = 0.f;
        std::uint32_t n_active = 0;
        for (auto const &pd : cr.per_device_gpu) {
            if (pd.total_ms <= 0.f) continue;
            gpu_max = std::max(gpu_max, pd.total_ms);
            gpu_sum += pd.total_ms;
            upload += pd.upload_ms;
            metric += pd.metric_ms;
            kernels += pd.kernels_ms;
            download += pd.download_ms;
            ++n_active;
        }
        float const inv = n_active > 0 ? 1.f / static_cast<float>(n_active) : 0.f;
        logger.trace("{:>5} {:>6}  {:>6.1f}  {:>6.1f}   {:>6.1f}  {:>6.1f} / {:>6.1f}  {:>5.1f} / {:>5.1f} / "
                     "{:>6.1f} / {:>6.1f}",
                     cr.wave_idx, cr.n_mol, cr.prep_end_ms - cr.prep_start_ms,
                     cr.launch_end_ms - cr.launch_start_ms, cr.finalize_end_ms - cr.finalize_start_ms,
                     gpu_sum * inv, gpu_max, upload * inv, metric * inv, kernels * inv, download * inv);
    }
    logger.trace("wall {:.1f} ms", total_wall_ms);
    for (auto const &cr : wave_records) {
        for (std::uint32_t d = 0; d < cr.per_device_gpu.size(); ++d) {
            auto const &pd = cr.per_device_gpu[d];
            if (pd.total_ms <= 0.f) continue;
            logger.trace2("  ck={} dev={}  upload={:.1f}  metric={:.1f}  kernels={:.1f}  download={:.1f}  total={:.1f}",
                          cr.wave_idx, d, pd.upload_ms, pd.metric_ms, pd.kernels_ms, pd.download_ms, pd.total_ms);
        }
    }
}

conformer_status_t context_t::generate_conformers(std::span<std::string_view const> smiles, config_t const &config,
                                                  std::uint32_t max_atoms_per_molecule,
                                                  std::span<__half> coordinates_out, std::span<float> energies_out,
                                                  std::span<conformer_status_t> status_out) {

    cuda_error_sink_t const sink {&this->logger, &this->cuda_error_count};

    std::uint32_t const num_input = static_cast<std::uint32_t>(smiles.size());
    std::uint32_t const num_conformers = config.num_conformers;

    // Pre-initialize caller output buffers.  `finalize_gpu_pipeline` writes only
    // into the slots whose molecules made it through preprocessing; the rest
    // stay at these defaults (invalid_smiles / zero coords / zero energy).
    std::fill_n(status_out.begin(), status_out.size(), conformer_status_t::invalid_smiles);
    std::fill_n(energies_out.begin(), energies_out.size(), 0.0f);
    {
        __half const zero = __float2half_rn(0.0f);
        std::fill_n(coordinates_out.begin(), coordinates_out.size(), zero);
    }

    // Sort SMILES by descending length so the last `embedding-wave` carries
    // the smallest molecules — shortens the unhidden tail wave.
    std::vector<std::uint32_t> sort_perm;
    std::vector<std::string_view> sorted_smiles_storage;
    sort_input_by_smiles_length_(smiles, sort_perm, sorted_smiles_storage);
    std::span<std::string_view const> smiles_view = sorted_smiles_storage;
    std::span<std::uint32_t const> const sort_perm_view = sort_perm;

    std::uint32_t const num_devices = static_cast<std::uint32_t>(device_ids.size());
    std::uint32_t const wave_size_target = compute_wave_size_(*this, config, max_atoms_per_molecule, sink);
    this->logger.trace("generate_conformers: n={} K={} wave={} pipeline={}", num_input, num_conformers,
                       wave_size_target, num_input > wave_size_target * 3 / 2 ? "yes" : "no");

    bool const pipeline = num_input > wave_size_target * 3 / 2;

    // CRITICAL: keep the kernel stride fixed at the caller's max_atoms_per_molecule
    // across ALL waves.  Shrinking to the first wave's observed max caused
    // "illegal memory access" in multi-wave runs where the caller's buffer is
    // sized for max_atoms_per_molecule but the kernel wrote with a smaller stride.
    std::uint32_t const effective_max_atoms = max_atoms_per_molecule;

    // Pre-grow every persistent per-device pool (launch_scratch.device_pool,
    // launch_scratch.host_staging, metric_scratch.pool) to the worst-case wave
    // footprint for this call BEFORE any wave launches.  Rationale: the
    // grow-only `byte_buffer_t::ensure_capacity` frees the old allocation and
    // cudaMallocs a new one on any size bump; in the pipelined multi-wave
    // path, a later wave's prep can compute a slightly larger
    // `batch.layout.bytes_total` (driven by the molecules in that wave) and
    // trigger a mid-pipeline `cudaFree(old_pool) + cudaMalloc(new_pool)` on
    // the SAME launch_scratch that the prior wave's stream still has async
    // reads against.  Pre-sizing once per call collapses every later
    // `ensure_capacity` to a no-op and closes this stream-use-after-free.
    {
        // Worst-case per-GPU slice = ceil(wave_size_target / num_devices) with slack
        // so an uneven round-robin leaves no shortfall.
        std::uint32_t const per_gpu_slack = (num_devices > 0) ? (num_devices - 1u) : 0u;
        std::uint32_t const worst_case_local_count =
            (wave_size_target + per_gpu_slack) / std::max<std::uint32_t>(num_devices, 1u) + 1u;
        std::size_t const bytes_per_mol = estimate_per_molecule_batch_bytes(effective_max_atoms);
        std::size_t const worst_case_batch_bytes = bytes_per_mol * worst_case_local_count;
        pipeline_pool_layout_t const worst_case_layout = compute_pipeline_pool_layout_(
            worst_case_batch_bytes, worst_case_local_count, num_conformers, effective_max_atoms);
        // Round to the next metric-prepass bin boundary (must match launch_wave's bin_ceiling).
        auto bin_ceiling = [](std::uint32_t n) -> std::uint32_t {
            static constexpr std::uint32_t bin_ns[] = {16, 32, 48, 64, 80, 96, 112, 128};
            for (auto b : bin_ns)
                if (n <= b) return b;
            return bin_ns[7];
        };
        std::uint32_t const metric_bin_atoms = bin_ceiling(effective_max_atoms);
        for (std::size_t d = 0; d < device_ids.size(); ++d) {
            sink.check(cudaSetDevice(device_ids[d]), "cudaSetDevice(pre-grow)");
            (void)launch_scratch[d]->device_pool.ensure_capacity(worst_case_layout.device_pool_bytes);
            (void)launch_scratch[d]->host_staging.ensure_capacity(worst_case_layout.host_pool_bytes);
            // Pre-grow the cuSOLVER metric-prepass pool similarly — reuses the
            // grow-only semantics, so subsequent wave-level calls are no-ops.
            ensure_metric_scratch(*this, d, worst_case_local_count, metric_bin_atoms);
        }
    }

    std::vector<wave_record_t> wave_records;
    auto const t_call_start = std::chrono::high_resolution_clock::now();
    auto host_ms = [&](std::chrono::high_resolution_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(t - t_call_start).count();
    };

    // Launch fans out across GPUs with a pure single-threaded for-loop. Every CUDA
    // call below is stream-async after the cusolver scratch is pre-allocated, so the
    // loop just enqueues work and returns. GPU d's stream runs concurrently with
    // GPU d+1's enqueue and with the main thread's next prep_slice().
    auto launch_wave = [&](etkdg_molecule_batch_t const &batch, std::uint32_t wave_base,
                            std::vector<gpu_pipeline_handle_t> &handles_out, std::vector<gpu_phase_ms_t> &timings_out) {
        if (batch.num_molecules == 0) return;

        // Distribute molecules across devices via round-robin (i % num_devices).
        // One flat arena + per-device [start, end) offsets, rather than
        // num_devices nested std::vectors. Two passes: first counts, then fills.
        std::vector<std::uint32_t> per_gpu_indices_flat(batch.num_molecules);
        std::vector<std::uint32_t> per_gpu_offsets(num_devices + 1, 0);
        for (std::uint32_t i = 0; i < batch.num_molecules; ++i) ++per_gpu_offsets[(i % num_devices) + 1];
        for (std::uint32_t d = 1; d <= num_devices; ++d) per_gpu_offsets[d] += per_gpu_offsets[d - 1];
        {
            std::vector<std::uint32_t> cursors = per_gpu_offsets;
            for (std::uint32_t i = 0; i < batch.num_molecules; ++i) {
                std::uint32_t const d = i % num_devices;
                per_gpu_indices_flat[cursors[d]++] = i;
            }
        }
        auto per_gpu_indices_span = [&](std::uint32_t d) -> std::span<std::uint32_t const> {
            return std::span<std::uint32_t const>(per_gpu_indices_flat.data() + per_gpu_offsets[d],
                                                  per_gpu_offsets[d + 1] - per_gpu_offsets[d]);
        };

        // Round to the next metric-prepass bin boundary; cusolver runs at binned N,
        // not raw atoms, so the workspace + gram pool must fit that. Must stay in
        // sync with bin_ns[] in build_metric_initial_coordinates_on_gpu.
        auto bin_ceiling = [](std::uint32_t n) -> std::uint32_t {
            static constexpr std::uint32_t bin_ns[] = {16, 32, 48, 64, 80, 96, 112, 128};
            for (auto b : bin_ns)
                if (n <= b) return b;
            return bin_ns[7];
        };
        std::uint32_t const metric_bin_atoms = bin_ceiling(effective_max_atoms);
        for (std::uint32_t d = 0; d < num_devices; ++d) {
            auto gpu_span = per_gpu_indices_span(d);
            if (gpu_span.empty()) continue;
            ensure_metric_scratch(*this, d, gpu_span.size(), metric_bin_atoms);
        }

        handles_out.resize(num_devices);
        timings_out.assign(num_devices, gpu_phase_ms_t {});
        // Dispatcher runs the 8 launches in parallel on 8 OpenMP threads. Each
        // launch call takes ~500 ms of CPU-side work (mostly cusolver state update
        // for the metric prepass), and the eight per-device state structs
        // (streams[d], solvers[d], metric_scratch[d], launch_scratch[d]) are
        // isolated, so the parallelization is safe and effectively free.
        // schedule(static, 1) pins one thread per device so cudaSetDevice stickiness
        // is deterministic.
#pragma omp parallel for num_threads(num_devices) schedule(static, 1)
        for (std::uint32_t d = 0; d < num_devices; ++d) {
            auto gpu_span = per_gpu_indices_span(d);
            if (gpu_span.empty()) continue;
            auto _lt0 = std::chrono::high_resolution_clock::now();
            handles_out[d] = launch_gpu_pipeline(
                device_ids[d], streams[d], solvers[d], metric_scratch[d], *launch_scratch[d], batch, gpu_span,
                num_conformers, config.max_stereo_attempts, effective_max_atoms, config.random_seed, wave_base,
                sort_perm_view, coordinates_out, energies_out, status_out, sink);
            handles_out[d].timings_out = &timings_out[d];
            (void)_lt0;
        }
    };

    // Finalize walks the 8 per-GPU handles sequentially. The first finalize's
    // cudaStreamSynchronize blocks until that stream drains; by the time it returns,
    // the other 7 GPUs have almost always finished their wave too (they started
    // within microseconds of each other and have the same work), so the remaining
    // finalize calls are near-instant.
    auto finalize_wave = [&](std::vector<gpu_pipeline_handle_t> &handles) {
        for (auto &h : handles)
            if (h.device_id >= 0) finalize_gpu_pipeline(std::move(h));
        handles.clear();
    };

    auto finalize_wave_timed = [&](std::vector<gpu_pipeline_handle_t> &handles, std::uint32_t wave_idx) {
        auto t0 = std::chrono::high_resolution_clock::now();
        finalize_wave(handles);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (wave_idx < wave_records.size()) {
            wave_records[wave_idx].finalize_start_ms = host_ms(t0);
            wave_records[wave_idx].finalize_end_ms = host_ms(t1);
            // Pull the per-device timings out of the handles' slots — finalize_wave
            // already invoked finalize_gpu_pipeline which wrote into each timings_out.
            // The slots live in launch_wave's `timings_out` argument which the caller
            // owns; we read them back from there via a separate vector below.
        }
    };

    if (!pipeline) {
        // Non-pipelined: single prep, single launch, single finalize. Identical
        // behaviour to the pre-pipelining code path.
        wave_records.push_back({0, num_input, 0, 0, 0, 0, 0, 0, {}});
        wave_records.back().prep_start_ms = host_ms(std::chrono::high_resolution_clock::now());
        etkdg_molecule_batch_t batch = prepare_molecule_batch(smiles_view, scaffold_cache.get());
        wave_records.back().prep_end_ms = host_ms(std::chrono::high_resolution_clock::now());
        // effective_max_atoms is const (kept at caller's max_atoms_per_molecule) — no update here.
        std::vector<gpu_pipeline_handle_t> handles;
        std::vector<gpu_phase_ms_t> timings;
        wave_records.back().launch_start_ms = host_ms(std::chrono::high_resolution_clock::now());
        launch_wave(batch, /*wave_base=*/0, handles, timings);
        wave_records.back().launch_end_ms = host_ms(std::chrono::high_resolution_clock::now());
        finalize_wave_timed(handles, 0);
        wave_records.back().per_device_gpu = timings;
    }
    else {
        // Pipelined: split input into ~50K-molecule waves, double-buffer the batches,
        // interleave CPU prep of wave N+1 with stream-async GPU work of wave N.
        std::vector<std::pair<std::uint32_t, std::uint32_t>> waves; // [begin, end)
        for (std::uint32_t i = 0; i < num_input; i += wave_size_target)
            waves.emplace_back(i, std::min(i + wave_size_target, num_input));
        wave_records.resize(waves.size());
        for (std::uint32_t i = 0; i < waves.size(); ++i) {
            wave_records[i].wave_idx = i;
            wave_records[i].n_mol = waves[i].second - waves[i].first;
        }

        // Reuse the same batches slots across all waves so their pinned host
        // buffers (allocated once at first wave, grown-only) don't round-trip
        // through cudaHostAlloc/cudaFreeHost — those serialize on a driver
        // lock and dominate CUDA API time at 8-GPU scale (measured via nsys).
        auto prep_slice_into = [&](etkdg_molecule_batch_t &batch, std::uint32_t begin, std::uint32_t end) {
            prepare_molecule_batch_into(smiles_view.subspan(begin, end - begin), batch, scaffold_cache.get());
        };

        // Two-slot ring buffer for both batches and handles. Fixed-size
        // `std::array` keeps everything in contiguous storage — no vector growth,
        // no indirection per access, and `gpu_pipeline_handle_t::batch` can hold
        // a stable pointer into `batches[slot]` for the lifetime of one wave.
        // At most one wave is in flight (finalize_wave(i-1) runs before
        // launch_wave(i) overwrites the opposite slot), so two slots suffice.
        // Persistent cross-call batch buffers previously correlated with a
        // multi-wave illegal-memory-access on the MMFF polish kernel; using
        // local per-call pinned allocations (matching the v8 working
        // configuration) eliminates that fault at the cost of one
        // cudaHostAlloc/cudaFreeHost pair per generate_conformers call.
        std::array<batch_pool_t, 2> batches {};
        std::array<std::vector<gpu_pipeline_handle_t>, 2> in_flight {};
        std::array<std::vector<gpu_phase_ms_t>, 2> timing_slots {};
        in_flight[0].resize(num_devices);
        in_flight[1].resize(num_devices);

        // Bootstrap: prep + launch wave 0. wave 0's GPU work runs in the background
        // on the CUDA streams while the main-thread loop below preps wave 1.
        wave_records[0].prep_start_ms = host_ms(std::chrono::high_resolution_clock::now());
        prep_slice_into(batches[0], waves[0].first, waves[0].second);
        wave_records[0].prep_end_ms = host_ms(std::chrono::high_resolution_clock::now());
        // effective_max_atoms stays at caller's max_atoms_per_molecule — see declaration.
        wave_records[0].launch_start_ms = host_ms(std::chrono::high_resolution_clock::now());
        launch_wave(batches[0], waves[0].first, in_flight[0], timing_slots[0]);
        wave_records[0].launch_end_ms = host_ms(std::chrono::high_resolution_clock::now());

        // Main pipeline loop. At the top of iteration i:
        //   - GPU work for wave i-1 is in flight in slot (i-1) % 2.
        //   - Batch slot (i % 2) held wave i-2, which was finalized at iteration i-1.
        //     So it's safe to overwrite with wave i's batch.
        for (std::size_t i = 1; i < waves.size(); ++i) {
            wave_records[i].prep_start_ms = host_ms(std::chrono::high_resolution_clock::now());
            prep_slice_into(batches[i % 2], waves[i].first, waves[i].second); // CPU work (overlaps GPU wave i-1)
            wave_records[i].prep_end_ms = host_ms(std::chrono::high_resolution_clock::now());
            // effective_max_atoms stays constant across waves — see declaration.
            finalize_wave_timed(in_flight[(i - 1) % 2], static_cast<std::uint32_t>(i - 1));
            wave_records[i - 1].per_device_gpu = timing_slots[(i - 1) % 2];
            wave_records[i].launch_start_ms = host_ms(std::chrono::high_resolution_clock::now());
            launch_wave(batches[i % 2], waves[i].first, in_flight[i % 2], timing_slots[i % 2]); // async dispatch
            wave_records[i].launch_end_ms = host_ms(std::chrono::high_resolution_clock::now());
        }
        std::uint32_t const last_idx = static_cast<std::uint32_t>(waves.size() - 1);
        finalize_wave_timed(in_flight[last_idx % 2], last_idx);
        wave_records[last_idx].per_device_gpu = timing_slots[last_idx % 2];
    }

    print_wave_timeline_(this->logger, wave_records, host_ms(std::chrono::high_resolution_clock::now()));

    // All output permutation already happened inside `finalize_gpu_pipeline`:
    // each wave's per-GPU sort + the outer SMILES-length sort are folded into
    // one reorder that writes directly into caller_idx space. Here we just scan
    // the already-permuted status buffer to derive the aggregate return code.
    bool any_success = false;
    for (auto s : status_out)
        if (s == conformer_status_t::success) {
            any_success = true;
            break;
        }

    return any_success ? conformer_status_t::success : conformer_status_t::mmff_unconverged;
}

} // namespace smallfield::api

#if defined(SMALLFIELD_BUILD_PYTHON)

PYBIND11_MODULE(_smallfield, m) {
    m.doc() = "SmallField: GPU ETKDG + MMFF94 conformer generation";

    m.def("supports_cuda", []() { return true; });

    py::enum_<smallfield::api::conformer_status_t>(m, "ConformerStatus")
        .value("success", smallfield::api::conformer_status_t::success)
        .value("stereo_failure", smallfield::api::conformer_status_t::stereo_failure)
        .value("mmff_unconverged", smallfield::api::conformer_status_t::mmff_unconverged)
        .value("invalid_smiles", smallfield::api::conformer_status_t::invalid_smiles)
        .value("too_many_atoms", smallfield::api::conformer_status_t::too_many_atoms);

    py::class_<smallfield::api::context_t>(m, "Context", "GPU context managing CUDA streams and cuSOLVER handles.")
        .def(py::init([](std::vector<int> device_ids) {
                 std::span<int const> span(device_ids.data(), device_ids.size());
                 return std::make_unique<smallfield::api::context_t>(span);
             }),
             py::arg("device_ids") = std::vector<int> {})
        .def(
            "generate_conformers",
            [](smallfield::api::context_t &ctx, std::vector<std::string> const &smiles_list,
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
                std::span<std::string_view const> smiles_span(smiles_views);

                // Temporary half-precision buffer.
                std::vector<__half> half_coords(total_coords);
                std::span<__half> coords_span(half_coords);
                std::span<float> energies_span(energies_np.mutable_data(), total_conf);

                // Status buffer: use conformer_status_t then copy to uint8_t.
                std::vector<smallfield::api::conformer_status_t> status_buf(total_conf);
                std::span<smallfield::api::conformer_status_t> status_span(status_buf);

                ctx.generate_conformers(smiles_span, config, max_atoms_per_molecule, coords_span, energies_span,
                                        status_span);

                // Convert __half coordinates back to float for Python.
                float *coords_ptr = coords_np.mutable_data();
                for (std::size_t i = 0; i < total_coords; ++i) coords_ptr[i] = __half2float(half_coords[i]);

                // Copy status enum values to uint8_t numpy array.
                std::uint8_t *status_ptr = status_np.mutable_data();
                for (std::uint32_t i = 0; i < total_conf; ++i) status_ptr[i] = static_cast<std::uint8_t>(status_buf[i]);

                return py::make_tuple(coords_np, energies_np, status_np);
            },
            "Generate conformers for a batch of SMILES strings.\n" "Returns (coordinates, energies, status) as " "numpy" " " "a" "r" "r" "a" "y" "s" ".",
            py::arg("smiles"), py::arg("num_conformers") = 3, py::arg("max_atoms_per_molecule") = 128,
            py::arg("max_stereo_attempts") = 10, py::arg("random_seed") = 42);
}

#endif // SMALLFIELD_BUILD_PYTHON
