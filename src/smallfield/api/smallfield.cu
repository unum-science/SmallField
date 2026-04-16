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
#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/cuda/metric_embedding_prepass.cuh"
#include "smallfield/api/smallfield.hpp"

namespace smallfield::api {
using namespace smallfield::cuda;
using namespace smallfield::mmff94;
using namespace smallfield::parameters;
using namespace smallfield::preprocess;
using namespace smallfield::types;

static inline std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

/**
 *  @brief Persistent per-device scratch for the kernel launch path.
 *
 *  @c device_pool holds the batch-input mirror + per-launch outputs + scratch;
 *  @c host_staging holds the sorted-metadata subset (CPU → GPU) plus pinned
 *  readback regions for coords / energies / status (GPU → CPU).  Both are
 *  grow-only and reused across chunks so the single-threaded dispatcher
 *  avoids per-launch allocation churn.
 */
struct launch_scratch_t {
    device_byte_buffer_t device_pool;
    pinned_byte_buffer_t host_staging;
    // Persistent CUDA events for per-phase timing. Created once per device on
    // first launch and reused across every chunk. Allocating events inside the
    // hot loop and destroying them after each finalize triggered an intermittent
    // multi-GPU regression (chunks 3-4 silently dropped at 240K) — likely a
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
 *  @brief Shared pinned buffers for the caller's float32 coords / energies / status
 *  output region.  Grow-only, allocated once per @c context_t, reused across every
 *  @c generate_conformers call.  Each chunk's GPU writes into a slice of these via
 *  the launch handle; @c finalize_gpu_pipeline then permutes sorted → input-order.
 */
struct output_staging_t {
    pinned_byte_buffer_t pool;
    std::size_t coords_offset = 0;
    std::size_t coords_count = 0;
    std::size_t energies_offset = 0;
    std::size_t energies_count = 0;
    std::size_t status_offset = 0;
    std::size_t status_count = 0;

    std::span<float> coords() noexcept {
        return {reinterpret_cast<float *>(pool.data() + coords_offset), coords_count};
    }
    std::span<float> energies() noexcept {
        return {reinterpret_cast<float *>(pool.data() + energies_offset), energies_count};
    }
    std::span<std::uint8_t> status() noexcept {
        return {reinterpret_cast<std::uint8_t *>(pool.data() + status_offset), status_count};
    }
};

// Forward declaration — definition lives below. context_t's ctor pre-warms
// the per-device metric scratch at construction time so the first user call
// doesn't pay the cusolver workspace probe + cudaMalloc cost.
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
    launch_scratch.resize(device_ids.size(), nullptr);
    output_staging = new output_staging_t;

    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        cudaSetDevice(device_ids[i]);
        cudaStreamCreate(&streams[i]);
        cusolverDnCreate(&solvers[i]);
        cusolverDnSetStream(solvers[i], streams[i]);
        // Reuse one cusolverDnParams_t per device across all calls. The
        // generic 64-bit batched eigen API (cusolverDnXsyevBatched) takes this
        // instead of the legacy syevjInfo_t — lower per-call CPU overhead.
        cusolverDnCreateParams(&metric_scratch[i].xsyev_params);
        launch_scratch[i] = new launch_scratch_t;
        // Persistent timing events live with the per-device launch scratch.
        // BlockingSync = host doesn't busy-wait inside cudaStreamSynchronize.
        cudaEventCreateWithFlags(&launch_scratch[i]->event_upload_start, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&launch_scratch[i]->event_metric_start, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&launch_scratch[i]->event_kernels_start, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&launch_scratch[i]->event_download_start, cudaEventBlockingSync);
        cudaEventCreateWithFlags(&launch_scratch[i]->event_end, cudaEventBlockingSync);
    }
    // Pre-warm the dispatcher so the first user call does not pay cold-start costs.
    // Cold chunk 0 has "mallocpool" ~100-130 ms
    // per device for the first cudaMalloc + cudaHostAlloc (910 ms total across
    // 8 GPUs per prior nsys capture), plus a ~60 ms cusolver workspace probe
    // inside ensure_metric_scratch. Pre-sizing both at construction moves
    // those costs off the user's hot path. The sizes are conservative
    // first-call defaults: a 5 K-molecule chunk with max_atoms=128 covers any
    // small-batch workload; bigger batches grow the pool on demand via
    // ensure_capacity (which is already the existing growth path).
    constexpr std::size_t kPrewarmGroup = 5'000;
    constexpr std::uint32_t kPrewarmAtoms = 128;
    constexpr std::size_t kPrewarmDevicePoolBytes = 256ULL << 20;  // 256 MB
    constexpr std::size_t kPrewarmHostStagingBytes = 256ULL << 20; // 256 MB
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
    }
    // Pre-warm the shared pinned output buffer to a size that covers a
    // 241 K × K=1 workload (≈ 370 MB). Larger K values or larger batches
    // grow on demand via `output_staging->pool.ensure_capacity` inside
    // `ensure_output_staging`, which remains grow-only so subsequent calls
    // reuse the allocation. Without this pre-warm, the first
    // `generate_conformers` call pays a ≈ 370 MB `cudaHostAlloc` serialized
    // on the driver lock — nsys showed it hit 2.44 s of wall time.
    cudaSetDevice(device_ids[0]);
    constexpr std::size_t kPrewarmOutputBytes = 512ULL << 20; // 512 MB
    output_staging->pool.ensure_capacity(kPrewarmOutputBytes);
}

context_t::~context_t() {
    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        cudaSetDevice(device_ids[i]);
        if (metric_scratch[i].xsyev_params) cusolverDnDestroyParams(metric_scratch[i].xsyev_params);
        if (metric_scratch[i].pool) cudaFree(metric_scratch[i].pool);
        if (launch_scratch[i] != nullptr) {
            if (launch_scratch[i]->event_upload_start) cudaEventDestroy(launch_scratch[i]->event_upload_start);
            if (launch_scratch[i]->event_metric_start) cudaEventDestroy(launch_scratch[i]->event_metric_start);
            if (launch_scratch[i]->event_kernels_start) cudaEventDestroy(launch_scratch[i]->event_kernels_start);
            if (launch_scratch[i]->event_download_start) cudaEventDestroy(launch_scratch[i]->event_download_start);
            if (launch_scratch[i]->event_end) cudaEventDestroy(launch_scratch[i]->event_end);
        }
        delete launch_scratch[i]; // device_byte_buffer_t + pinned_byte_buffer_t free their holdings
        cusolverDnDestroy(solvers[i]);
        cudaStreamDestroy(streams[i]);
    }
    delete output_staging;
    delete scaffold_cache; // no-op if null
}

void context_t::enable_scaffold_cache() {
    if (scaffold_cache == nullptr) scaffold_cache = new scaffold_cache_t;
}

scaffold_cache_stats_t context_t::get_scaffold_cache_stats() const {
    if (scaffold_cache == nullptr) return {};
    return scaffold_cache->stats();
}

/**
 *  @brief Size the shared pinned output region for a given batch shape.
 *  Grow-only; successive calls reuse the allocation if it already fits.
 */
void ensure_output_staging(output_staging_t &s, std::size_t total_conformers, std::size_t coord_stride) {
    std::size_t const coords_bytes = align_up(total_conformers * coord_stride * sizeof(float), 256);
    std::size_t const energies_bytes = align_up(total_conformers * sizeof(float), 256);
    std::size_t const status_bytes = align_up(total_conformers * sizeof(std::uint8_t), 256);
    std::size_t const total = coords_bytes + energies_bytes + status_bytes;
    s.pool.ensure_capacity(total);
    s.coords_offset = 0;
    s.coords_count = total_conformers * coord_stride;
    s.energies_offset = coords_bytes;
    s.energies_count = total_conformers;
    s.status_offset = coords_bytes + energies_bytes;
    s.status_count = total_conformers;
}

/**
 *  @brief Ensure the per-device metric scratch pool is sized for the given
 *  (max_group_size, max_atoms) combination.  Grows only — never shrinks — so
 *  the first chunk of a workload pays the allocation cost and subsequent
 *  chunks reuse it.
 */
void ensure_metric_scratch(context_t &ctx, std::size_t device_idx, std::size_t max_group_size,
                           std::uint32_t max_atoms) {
    auto &s = ctx.metric_scratch[device_idx];
    if (max_group_size <= s.max_group_size && max_atoms <= s.max_atoms && s.pool != nullptr) return;

    // Grow to at least the new request, but round up a bit to avoid thrashing
    // across similarly-sized chunks.
    std::size_t const new_group = std::max(s.max_group_size, max_group_size);
    std::uint32_t const new_atoms = std::max<std::uint32_t>(s.max_atoms, max_atoms);

    cudaSetDevice(ctx.device_ids[device_idx]);

    // Probe cusolver for the worst-case workspace size at the new dimensions.
    // Probe cusolverDnXsyevBatched for the worst-case device + host workspace sizes.
    // Once per growth, not per chunk.
    std::size_t new_dev_ws_bytes = 0;
    std::size_t new_host_ws_bytes = 0;
    {
        float *probe_gram = nullptr;
        float *probe_eig = nullptr;
        cudaMalloc(&probe_gram, sizeof(float));
        cudaMalloc(&probe_eig, sizeof(float));
        cusolverDnXsyevBatched_bufferSize(
            ctx.solvers[device_idx], s.xsyev_params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
            static_cast<int64_t>(new_atoms), CUDA_R_32F, probe_gram, static_cast<int64_t>(new_atoms), CUDA_R_32F,
            probe_eig, CUDA_R_32F, &new_dev_ws_bytes, &new_host_ws_bytes, static_cast<int64_t>(new_group));
        cudaFree(probe_gram);
        cudaFree(probe_eig);
    }

    // Compute pool layout with 256-byte alignment.
    std::size_t const group_bytes = align_up(new_group * sizeof(std::uint32_t), 256);
    std::size_t const gram_bytes = align_up(new_group * static_cast<std::size_t>(new_atoms) * new_atoms * sizeof(float),
                                            256);
    std::size_t const eig_bytes = align_up(new_group * new_atoms * sizeof(float), 256);
    std::size_t const info_bytes = align_up(new_group * sizeof(int), 256);
    std::size_t const ws_bytes = align_up(new_dev_ws_bytes, 256);
    std::size_t const total = group_bytes + gram_bytes + eig_bytes + info_bytes + ws_bytes;

    if (s.pool) cudaFree(s.pool);
    cudaMalloc(&s.pool, total);
    s.pool_bytes = total;
    std::size_t off = 0;
    auto *base = static_cast<char *>(s.pool);
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
 *  @brief Per-chunk, per-GPU stream-local phase timings, populated in
 *  @ref finalize_gpu_pipeline after the stream drains.  All values are in
 *  milliseconds and are measured by @c cudaEventElapsedTime on events that
 *  were recorded into the device's stream — they reflect actual GPU + DMA
 *  wall, not host-side scheduling.  Aggregator in @ref generate_conformers
 *  combines these across devices to derive per-chunk GPU wall (max) and
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
 *  @brief Per-chunk, per-GPU state carried from `launch_gpu_pipeline` to `finalize_gpu_pipeline`.
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
    std::vector<std::uint32_t> order;
    std::vector<std::uint32_t> local_indices_copy;
    // Device-side output spans into scratch->device_pool.
    std::span<float> device_coords_out {};
    std::span<float> device_energies_out {};
    std::span<std::uint8_t> device_status_out {};
    // Pinned host-side readback spans into scratch->host_staging; finalize drains
    // the stream then permutes from these into the caller's output slice.
    std::span<float> host_coords_sorted {};
    std::span<float> host_energies_sorted {};
    std::span<std::uint8_t> host_status_sorted {};
    // Caller-owned output slices (non-owning).
    std::span<float> host_coords_out {};
    std::span<float> host_energies_out {};
    std::span<std::uint8_t> host_status_out {};
    // Stream-local phase timing events — recorded in launch, consumed in finalize.
    cudaEvent_t event_upload_start = nullptr;
    cudaEvent_t event_metric_start = nullptr;
    cudaEvent_t event_kernels_start = nullptr;
    cudaEvent_t event_download_start = nullptr;
    cudaEvent_t event_end = nullptr;
    gpu_phase_ms_t *timings_out = nullptr; ///< caller-owned slot; finalize writes per-phase ms
};

/**
 *  @brief Enqueue all upload, metric prepass, kernel, and download operations on the
 *  GPU stream; return immediately (no CPU-side sync).  Caller must eventually call
 *  `finalize_gpu_pipeline` with the returned handle.
 */
gpu_pipeline_handle_t launch_gpu_pipeline(int device_id, cudaStream_t stream, cusolverDnHandle_t solver_handle,
                                          metric_scratch_t &scratch, launch_scratch_t &launch_buffers,
                                          batch_pool_t const &batch, std::span<std::uint32_t const> local_indices,
                                          std::uint32_t num_conformers, std::uint32_t max_stereo_attempts,
                                          std::uint32_t max_atoms_per_molecule, std::uint64_t seed,
                                          std::span<float> host_coords_out, std::span<float> host_energies_out,
                                          std::span<std::uint8_t> host_status_out) {

    auto _P0 = std::chrono::high_resolution_clock::now();
    cudaSetDevice(device_id);

    std::uint32_t const local_count = static_cast<std::uint32_t>(local_indices.size());

    // Sort this GPU's molecules by atom count for wave-based dispatch.
    auto meta_full = batch.molecule_metadata();
    std::vector<std::uint32_t> order(local_count);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        return meta_full[local_indices[a]].num_atoms < meta_full[local_indices[b]].num_atoms;
    });

    std::uint32_t const total_conf = local_count * num_conformers;
    std::uint32_t const total_coords = total_conf * max_atoms_per_molecule * 3;

    // Region 1: batch-input mirror (target of the single H2D).
    // Region 2: sorted metadata subset (tiny H2D from pinned host_staging).
    // Region 3: kernel outputs (coords / energies / status). Downloaded D2H
    //           to host_staging at the end of this function.
    // Region 4: GPU-only 4D scratch produced by the metric prepass.
    auto const sizeof_meta = sizeof(etkdg_molecule_metadata_t);
    std::size_t device_offset = 0;
    std::size_t const device_off_batch = align_up(device_offset, 256);
    device_offset = device_off_batch + align_up(batch.layout.bytes_total, 256);
    std::size_t const device_off_sorted_meta = device_offset;
    device_offset += align_up(local_count * sizeof_meta, 256);
    std::size_t const device_off_coords = device_offset;
    device_offset += align_up(total_coords * sizeof(float), 256);
    std::size_t const device_off_energies = device_offset;
    device_offset += align_up(total_conf * sizeof(float), 256);
    std::size_t const device_off_status = device_offset;
    device_offset += align_up(total_conf * sizeof(std::uint8_t), 256);
    std::size_t const device_off_initial = device_offset;
    device_offset += align_up(static_cast<std::size_t>(local_count) * max_atoms_per_molecule * 4 * sizeof(float), 256);
    std::size_t const device_pool_bytes = device_offset;

    // Inputs: sorted metadata (written by CPU, async-uploaded).
    // Outputs: kernel results (async-downloaded into here so finalize is free).
    std::size_t host_offset = 0;
    std::size_t const host_off_sorted_meta = align_up(host_offset, 256);
    host_offset = host_off_sorted_meta + align_up(local_count * sizeof_meta, 256);
    std::size_t const host_off_coords = host_offset;
    host_offset += align_up(total_coords * sizeof(float), 256);
    std::size_t const host_off_energies = host_offset;
    host_offset += align_up(total_conf * sizeof(float), 256);
    std::size_t const host_off_status = host_offset;
    host_offset += align_up(total_conf * sizeof(std::uint8_t), 256);
    std::size_t const host_pool_bytes = host_offset;

    auto _P_pre_malloc = std::chrono::high_resolution_clock::now();

    gpu_pipeline_handle_t handle;
    handle.device_id = device_id;
    handle.stream = stream;
    handle.batch = &batch;
    handle.scratch = &launch_buffers;
    handle.local_count = local_count;
    handle.num_conformers = num_conformers;
    handle.max_atoms_per_molecule = max_atoms_per_molecule;
    handle.order = std::move(order);
    handle.local_indices_copy.assign(local_indices.begin(), local_indices.end());
    handle.host_coords_out = host_coords_out;
    handle.host_energies_out = host_energies_out;
    handle.host_status_out = host_status_out;

    if (!launch_buffers.device_pool.ensure_capacity(device_pool_bytes)) {
        std::fprintf(stderr, "launch_gpu_pipeline: cudaMalloc(%zu bytes) failed on gpu %d\n", device_pool_bytes,
                     device_id);
        return handle;
    }
    if (!launch_buffers.host_staging.ensure_capacity(host_pool_bytes)) {
        std::fprintf(stderr, "launch_gpu_pipeline: cudaHostAlloc(%zu bytes) failed on gpu %d\n", host_pool_bytes,
                     device_id);
        return handle;
    }
    auto _P_post_malloc = std::chrono::high_resolution_clock::now();

    // Reuse the persistent per-device events from launch_scratch_t. These were
    // allocated once at context construction and live until context destruction;
    // re-recording them here is safe because the prior chunk's
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
    handle.device_coords_out = {reinterpret_cast<float *>(device_base + device_off_coords), total_coords};
    handle.device_energies_out = {reinterpret_cast<float *>(device_base + device_off_energies), total_conf};
    handle.device_status_out = {reinterpret_cast<std::uint8_t *>(device_base + device_off_status), total_conf};
    auto *device_initial_coordinates = reinterpret_cast<float *>(device_base + device_off_initial);

    auto *host_sorted_meta = reinterpret_cast<etkdg_molecule_metadata_t *>(host_base + host_off_sorted_meta);
    handle.host_coords_sorted = {reinterpret_cast<float *>(host_base + host_off_coords), total_coords};
    handle.host_energies_sorted = {reinterpret_cast<float *>(host_base + host_off_energies), total_conf};
    handle.host_status_sorted = {reinterpret_cast<std::uint8_t *>(host_base + host_off_status), total_conf};

    // Gather the sorted metadata directly into the pinned staging region so the
    // subsequent H2D does not degrade to synchronous.
    for (std::uint32_t i = 0; i < local_count; ++i) host_sorted_meta[i] = meta_full[local_indices[handle.order[i]]];

    // Replaces 14 pageable-source cudaMemcpyAsync calls with ONE pinned-source
    // transfer that actually returns immediately and runs on the stream.
    cudaEventRecord(handle.event_upload_start, stream);
    if (batch.layout.bytes_total > 0)
        cudaMemcpyAsync(device_base + device_off_batch, batch.host.data(), batch.layout.bytes_total,
                        cudaMemcpyHostToDevice, stream);
    if (local_count > 0)
        cudaMemcpyAsync(device_metadata_sorted, host_sorted_meta, local_count * sizeof_meta, cudaMemcpyHostToDevice,
                        stream);

    auto _P_post_upload = std::chrono::high_resolution_clock::now();
    cudaEventRecord(handle.event_metric_start, stream);
    build_metric_initial_coordinates_on_gpu(device_view.bounds_matrices().data(), device_metadata_sorted,
                                            std::span<etkdg_molecule_metadata_t const> {host_sorted_meta, local_count},
                                            max_atoms_per_molecule, device_initial_coordinates, solver_handle, scratch,
                                            stream);
    auto _P_post_metric = std::chrono::high_resolution_clock::now();
    cudaEventRecord(handle.event_kernels_start, stream);

    // Wave-based kernel dispatch (8 bins: 16,32,48,64,80,96,112,128).
    constexpr std::uint32_t wave_bins[] = {16, 32, 48, 64, 80, 96, 112, 128};
    constexpr std::uint32_t num_bins = 8;

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);

    std::uint32_t wave_start = 0;
    for (std::uint32_t bin = 0; bin < num_bins && wave_start < local_count; ++bin) {
        std::uint32_t wave_end = wave_start;
        while (wave_end < local_count && host_sorted_meta[wave_end].num_atoms <= wave_bins[bin]) ++wave_end;

        if (wave_end == wave_start) continue;

        std::uint32_t const wave_size = wave_end - wave_start;
        std::uint32_t const wave_max_atoms = wave_bins[bin];

        // Kernel 1 — DG + stereo retry + fourth-dim crush + ETK torsion refinement.
        std::uint32_t const shared_bytes_dg = compute_pipeline_shared_memory_bytes(wave_max_atoms, 0);
        cudaFuncSetAttribute(etkdg_pipeline_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes_dg);

        int blocks_per_sm_dg = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm_dg, etkdg_pipeline_warp_kernel, 32, shared_bytes_dg);
        std::uint32_t const grid_size_dg = std::min(wave_size, static_cast<std::uint32_t>(blocks_per_sm_dg * num_sms));

        etkdg_pipeline_warp_kernel<<<grid_size_dg, 32, shared_bytes_dg, stream>>>(
            device_view.bounds_matrices().data(), device_initial_coordinates + wave_start * max_atoms_per_molecule * 4,
            device_view.chiral_centers().data(), device_view.etk_torsions().data(),
            device_view.distance_constraints().data(), device_view.etk_interactions().data(),
            device_view.etk_atom_offsets().data(), device_metadata_sorted + wave_start,
            handle.device_coords_out.data() + wave_start * num_conformers * max_atoms_per_molecule * 3,
            handle.device_energies_out.data() + wave_start * num_conformers,
            handle.device_status_out.data() + wave_start * num_conformers, wave_size, num_conformers,
            max_stereo_attempts, wave_max_atoms, max_atoms_per_molecule, max_atoms_per_molecule, seed);

        // Kernel 2 — MMFF94 polish in place on the per-conformer coordinates
        // produced by Kernel 1.  Skips slots whose status is non-zero.
        std::uint32_t const shared_bytes_mmff = compute_mmff_polish_shared_memory_bytes(wave_max_atoms);
        cudaFuncSetAttribute(mmff_polish_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes_mmff);

        int blocks_per_sm_mmff = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm_mmff, mmff_polish_warp_kernel, 32, shared_bytes_mmff);
        std::uint32_t const grid_size_mmff =
            std::min(wave_size, static_cast<std::uint32_t>(blocks_per_sm_mmff * num_sms));

        mmff_polish_warp_kernel<<<grid_size_mmff, 32, shared_bytes_mmff, stream>>>(
            device_view.mmff_atom_vdw().data(), device_view.mmff_atom_charge().data(),
            device_view.bond_distance().data(), device_view.mmff_bonds().data(), device_view.mmff_angles().data(),
            device_view.mmff_torsions().data(), device_view.mmff_oops().data(), device_view.mmff_stretchbends().data(),
            device_view.mmff_interactions().data(), device_view.mmff_atom_offsets().data(),
            device_metadata_sorted + wave_start,
            handle.device_coords_out.data() + wave_start * num_conformers * max_atoms_per_molecule * 3,
            handle.device_energies_out.data() + wave_start * num_conformers,
            handle.device_status_out.data() + wave_start * num_conformers, wave_size, num_conformers,
            max_atoms_per_molecule);

        wave_start = wave_end;
    }

    // Enqueue pinned D2H for all three output arrays. Pinned destination means
    // these return immediately and the transfers run on the stream after the
    // kernels. `finalize_gpu_pipeline` just waits on the stream and permutes.
    cudaEventRecord(handle.event_download_start, stream);
    if (total_coords > 0)
        cudaMemcpyAsync(handle.host_coords_sorted.data(), handle.device_coords_out.data(), total_coords * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    if (total_conf > 0) {
        cudaMemcpyAsync(handle.host_energies_sorted.data(), handle.device_energies_out.data(),
                        total_conf * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(handle.host_status_sorted.data(), handle.device_status_out.data(),
                        total_conf * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream);
    }
    cudaEventRecord(handle.event_end, stream);

    auto _P_end = std::chrono::high_resolution_clock::now();
    auto elapsed_milliseconds = [](auto begin_time, auto end_time) {
        return std::chrono::duration<double, std::milli>(end_time - begin_time).count();
    };
    std::printf(
        "  [gpu%d] setup=%.1f mallocpool=%.1f upload=%.1f metric=%.1f kernels_post=%.1f total=%.1f\n", device_id,
        elapsed_milliseconds(_P0, _P_pre_malloc), elapsed_milliseconds(_P_pre_malloc, _P_post_malloc),
        elapsed_milliseconds(_P_post_malloc, _P_post_upload), elapsed_milliseconds(_P_post_upload, _P_post_metric),
        elapsed_milliseconds(_P_post_metric, _P_end), elapsed_milliseconds(_P0, _P_end));
    std::fflush(stdout);
    return handle;
}

/**
 *  @brief Wait for a previously-launched chunk to finish and permute results into
 *  the caller's output arrays.  All three D2H transfers were already enqueued by
 *  `launch_gpu_pipeline` into pinned host staging; this just drains the stream
 *  and does a permutation memcpy.
 */
void finalize_gpu_pipeline(gpu_pipeline_handle_t handle) {
    cudaSetDevice(handle.device_id);
    cudaStreamSynchronize(handle.stream);

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

    auto input_indices = handle.batch->input_indices();
    std::uint32_t const coord_stride = handle.max_atoms_per_molecule * 3;
    for (std::uint32_t sorted_idx = 0; sorted_idx < handle.local_count; ++sorted_idx) {
        std::uint32_t const local_idx = handle.order[sorted_idx];
        std::uint32_t const orig_idx = handle.local_indices_copy[local_idx];
        std::uint32_t const input_idx = input_indices[orig_idx];
        for (std::uint32_t c = 0; c < handle.num_conformers; ++c) {
            std::uint32_t const src_slot = sorted_idx * handle.num_conformers + c;
            std::uint32_t const dst_slot = input_idx * handle.num_conformers + c;
            handle.host_energies_out[dst_slot] = handle.host_energies_sorted[src_slot];
            handle.host_status_out[dst_slot] = handle.host_status_sorted[src_slot];
            std::memcpy(handle.host_coords_out.data() + static_cast<std::size_t>(dst_slot) * coord_stride,
                        handle.host_coords_sorted.data() + static_cast<std::size_t>(src_slot) * coord_stride,
                        coord_stride * sizeof(float));
        }
    }
}

/** Backward-compatible synchronous wrapper that takes the context's solver + scratch. */
void run_gpu_pipeline(int device_id, cudaStream_t stream, cusolverDnHandle_t solver_handle, metric_scratch_t &scratch,
                      launch_scratch_t &launch_buffers, etkdg_molecule_batch_t const &batch,
                      std::span<std::uint32_t const> local_indices, std::uint32_t num_conformers,
                      std::uint32_t max_stereo_attempts, std::uint32_t max_atoms_per_molecule, std::uint64_t seed,
                      std::span<float> host_coords_out, std::span<float> host_energies_out,
                      std::span<std::uint8_t> host_status_out) {
    auto handle = launch_gpu_pipeline(device_id, stream, solver_handle, scratch, launch_buffers, batch, local_indices,
                                      num_conformers, max_stereo_attempts, max_atoms_per_molecule, seed,
                                      host_coords_out, host_energies_out, host_status_out);
    finalize_gpu_pipeline(std::move(handle));
}

conformer_status_t context_t::generate_conformers(std::span<std::string_view const> smiles, config_t const &config,
                                                  std::uint32_t max_atoms_per_molecule,
                                                  std::span<__half> coordinates_out, std::span<float> energies_out,
                                                  std::span<conformer_status_t> status_out) {

    std::uint32_t const num_input = static_cast<std::uint32_t>(smiles.size());
    std::uint32_t const num_conformers = config.num_conformers;
    std::uint32_t const total_conf = num_input * num_conformers;
    std::uint32_t const coord_stride = max_atoms_per_molecule * 3;

    // Keep SMILES as string_view throughout — the single unavoidable std::string copy
    // lives inside prepare_molecule at the RDKit::SmilesToMol call site.

    // Sort molecules by descending SMILES length so the last chunk
    // contains the smallest molecules. At 8 GPUs the per-chunk wall is dominated
    // by max(CPU prep, GPU kernels). Chunk N's GPU work is the only fully unhidden
    // phase (no chunk N+1 to overlap with), so making chunk N the cheapest shrinks
    // the tail. SMILES length is a fast proxy for atom count — strongly correlated
    // and avoids a parse pass on the host. Permutation is undone in the output
    // write loop below so callers see results in their original input order.
    std::vector<std::uint32_t> sort_perm(num_input);
    std::iota(sort_perm.begin(), sort_perm.end(), 0u);
    std::sort(sort_perm.begin(), sort_perm.end(),
              [smiles](std::uint32_t a, std::uint32_t b) { return smiles[a].size() > smiles[b].size(); });
    std::vector<std::string_view> sorted_smiles_storage(num_input);
    for (std::uint32_t k = 0; k < num_input; ++k) sorted_smiles_storage[k] = smiles[sort_perm[k]];
    std::span<std::string_view const> smiles_view = sorted_smiles_storage;

    // Grow the shared pinned output region in context_t to fit this call. Allocated
    // once per context and reused — avoids a ~1 GB pageable malloc per call at 241K.
    ensure_output_staging(*output_staging, total_conf, coord_stride);
    std::span<float> host_coords = output_staging->coords();
    std::span<float> host_energies = output_staging->energies();
    std::span<std::uint8_t> host_status = output_staging->status();
    std::fill(host_coords.begin(), host_coords.end(), 0.0f);
    std::fill(host_energies.begin(), host_energies.end(), 0.0f);
    std::fill(host_status.begin(), host_status.end(), std::uint8_t {255});
    std::vector<bool> preprocessed_ok(num_input, false);

    std::uint32_t const num_devices = static_cast<std::uint32_t>(device_ids.size());

    // Chunk policy: small batches take the non-pipelined path. For large batches we
    // split into ~50 K-molecule chunks so CPU prep of chunk N+1 overlaps GPU work of
    // chunk N. All GPU operations are stream-async, so the pipeline is driven by the
    // calling thread — no std::thread, no queue.
    std::uint32_t const chunk_size_target = 50'000;
    bool const pipeline = num_input > chunk_size_target * 3 / 2;

    std::uint32_t effective_max_atoms = max_atoms_per_molecule;

    // Per-chunk timing accumulator: one row per chunk holding host-wall markers and
    // per-device GPU phase elapsed times. Filled in launch_chunk + finalize_chunk;
    // pretty-printed at the end of generate_conformers as an ASCII timeline.
    struct chunk_record_t {
        std::uint32_t chunk_idx;
        std::uint32_t n_mol;
        double prep_start_ms;     ///< host wall, ms relative to call start
        double prep_end_ms;       ///<     "
        double launch_start_ms;   ///<     "  (when launch_chunk entered)
        double launch_end_ms;     ///<     "  (when launch_chunk returned)
        double finalize_start_ms; ///<     "  (when finalize_chunk entered for this chunk)
        double finalize_end_ms;   ///<     "
        std::vector<gpu_phase_ms_t> per_device_gpu;
    };
    std::vector<chunk_record_t> chunk_records;
    auto const t_call_start = std::chrono::high_resolution_clock::now();
    auto host_ms = [&](std::chrono::high_resolution_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(t - t_call_start).count();
    };

    // Launch fans out across GPUs with a pure single-threaded for-loop. Every CUDA
    // call below is stream-async after the cusolver scratch is pre-allocated, so the
    // loop just enqueues work and returns. GPU d's stream runs concurrently with
    // GPU d+1's enqueue and with the main thread's next prep_slice().
    auto launch_chunk = [&](etkdg_molecule_batch_t const &batch, std::uint32_t chunk_base,
                            std::vector<gpu_pipeline_handle_t> &handles_out, std::vector<gpu_phase_ms_t> &timings_out) {
        auto batch_input_indices = batch.input_indices();
        for (std::uint32_t i = 0; i < batch.num_molecules; ++i)
            preprocessed_ok[chunk_base + batch_input_indices[i]] = true;
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

        std::size_t const conf_offset = static_cast<std::size_t>(chunk_base) * num_conformers;
        std::size_t const chunk_conformers = static_cast<std::size_t>(batch.num_molecules) * num_conformers;
        std::span<float> chunk_coords {host_coords.data() + conf_offset * coord_stride,
                                       chunk_conformers * coord_stride};
        std::span<float> chunk_energies {host_energies.data() + conf_offset, chunk_conformers};
        std::span<std::uint8_t> chunk_status {host_status.data() + conf_offset, chunk_conformers};

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
            handles_out[d] = launch_gpu_pipeline(device_ids[d], streams[d], solvers[d], metric_scratch[d],
                                                 *launch_scratch[d], batch, gpu_span, num_conformers,
                                                 config.max_stereo_attempts, effective_max_atoms, config.random_seed,
                                                 chunk_coords, chunk_energies, chunk_status);
            handles_out[d].timings_out = &timings_out[d];
            auto _lt1 = std::chrono::high_resolution_clock::now();
            std::printf("[launch gpu%d] %.1fms\n", d, std::chrono::duration<double, std::milli>(_lt1 - _lt0).count());
            std::fflush(stdout);
        }
    };

    // Finalize walks the 8 per-GPU handles sequentially. The first finalize's
    // cudaStreamSynchronize blocks until that stream drains; by the time it returns,
    // the other 7 GPUs have almost always finished their chunk too (they started
    // within microseconds of each other and have the same work), so the remaining
    // finalize calls are near-instant.
    auto finalize_chunk = [&](std::vector<gpu_pipeline_handle_t> &handles) {
        for (auto &h : handles)
            if (h.device_id >= 0) finalize_gpu_pipeline(std::move(h));
        handles.clear();
    };

    auto finalize_chunk_timed = [&](std::vector<gpu_pipeline_handle_t> &handles, std::uint32_t chunk_idx) {
        auto t0 = std::chrono::high_resolution_clock::now();
        finalize_chunk(handles);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (chunk_idx < chunk_records.size()) {
            chunk_records[chunk_idx].finalize_start_ms = host_ms(t0);
            chunk_records[chunk_idx].finalize_end_ms = host_ms(t1);
            // Pull the per-device timings out of the handles' slots — finalize_chunk
            // already invoked finalize_gpu_pipeline which wrote into each timings_out.
            // The slots live in launch_chunk's `timings_out` argument which the caller
            // owns; we read them back from there via a separate vector below.
        }
    };

    if (!pipeline) {
        // Non-pipelined: single prep, single launch, single finalize. Identical
        // behaviour to the pre-pipelining code path.
        chunk_records.push_back({0, num_input, 0, 0, 0, 0, 0, 0, {}});
        chunk_records.back().prep_start_ms = host_ms(std::chrono::high_resolution_clock::now());
        etkdg_molecule_batch_t batch = prepare_molecule_batch(smiles_view, scaffold_cache);
        chunk_records.back().prep_end_ms = host_ms(std::chrono::high_resolution_clock::now());
        effective_max_atoms = std::max(max_atoms_per_molecule, batch.max_atoms_per_molecule);
        std::vector<gpu_pipeline_handle_t> handles;
        std::vector<gpu_phase_ms_t> timings;
        chunk_records.back().launch_start_ms = host_ms(std::chrono::high_resolution_clock::now());
        launch_chunk(batch, /*chunk_base=*/0, handles, timings);
        chunk_records.back().launch_end_ms = host_ms(std::chrono::high_resolution_clock::now());
        finalize_chunk_timed(handles, 0);
        chunk_records.back().per_device_gpu = timings;
    }
    else {
        // Pipelined: split input into ~50K-molecule chunks, double-buffer the batches,
        // interleave CPU prep of chunk N+1 with stream-async GPU work of chunk N.
        std::vector<std::pair<std::uint32_t, std::uint32_t>> chunks; // [begin, end)
        for (std::uint32_t i = 0; i < num_input; i += chunk_size_target)
            chunks.emplace_back(i, std::min(i + chunk_size_target, num_input));
        chunk_records.resize(chunks.size());
        for (std::uint32_t i = 0; i < chunks.size(); ++i) {
            chunk_records[i].chunk_idx = i;
            chunk_records[i].n_mol = chunks[i].second - chunks[i].first;
        }

        // Reuse the same batches slots across all chunks so their pinned host
        // buffers (allocated once at first chunk, grown-only) don't round-trip
        // through cudaHostAlloc/cudaFreeHost — those serialize on a driver
        // lock and dominate CUDA API time at 8-GPU scale (measured via nsys).
        auto prep_slice_into = [&](etkdg_molecule_batch_t &batch, std::uint32_t begin, std::uint32_t end) {
            prepare_molecule_batch_into(smiles_view.subspan(begin, end - begin), batch, scaffold_cache);
        };

        // Two-slot ring buffer for both batches and handles. Fixed-size
        // `std::array` keeps everything in contiguous storage — no vector growth,
        // no indirection per access, and `gpu_pipeline_handle_t::batch` can hold
        // a stable pointer into `batches[slot]` for the lifetime of one chunk.
        // At most one chunk is in flight (finalize_chunk(i-1) runs before
        // launch_chunk(i) overwrites the opposite slot), so two slots suffice.
        std::array<etkdg_molecule_batch_t, 2> batches {};
        std::array<std::vector<gpu_pipeline_handle_t>, 2> in_flight {};
        std::array<std::vector<gpu_phase_ms_t>, 2> timing_slots {};
        in_flight[0].resize(num_devices);
        in_flight[1].resize(num_devices);

        // Bootstrap: prep + launch chunk 0. chunk 0's GPU work runs in the background
        // on the CUDA streams while the main-thread loop below preps chunk 1.
        chunk_records[0].prep_start_ms = host_ms(std::chrono::high_resolution_clock::now());
        prep_slice_into(batches[0], chunks[0].first, chunks[0].second);
        chunk_records[0].prep_end_ms = host_ms(std::chrono::high_resolution_clock::now());
        effective_max_atoms = std::max(effective_max_atoms, batches[0].max_atoms_per_molecule);
        chunk_records[0].launch_start_ms = host_ms(std::chrono::high_resolution_clock::now());
        launch_chunk(batches[0], chunks[0].first, in_flight[0], timing_slots[0]);
        chunk_records[0].launch_end_ms = host_ms(std::chrono::high_resolution_clock::now());

        // Main pipeline loop. At the top of iteration i:
        //   - GPU work for chunk i-1 is in flight in slot (i-1) % 2.
        //   - Batch slot (i % 2) held chunk i-2, which was finalized at iteration i-1.
        //     So it's safe to overwrite with chunk i's batch.
        for (std::size_t i = 1; i < chunks.size(); ++i) {
            chunk_records[i].prep_start_ms = host_ms(std::chrono::high_resolution_clock::now());
            prep_slice_into(batches[i % 2], chunks[i].first, chunks[i].second); // CPU work
            chunk_records[i].prep_end_ms = host_ms(std::chrono::high_resolution_clock::now());
            effective_max_atoms = std::max(effective_max_atoms, batches[i % 2].max_atoms_per_molecule);
            finalize_chunk_timed(in_flight[(i - 1) % 2], static_cast<std::uint32_t>(i - 1));
            chunk_records[i - 1].per_device_gpu = timing_slots[(i - 1) % 2];
            chunk_records[i].launch_start_ms = host_ms(std::chrono::high_resolution_clock::now());
            launch_chunk(batches[i % 2], chunks[i].first, in_flight[i % 2], timing_slots[i % 2]); // async dispatch
            chunk_records[i].launch_end_ms = host_ms(std::chrono::high_resolution_clock::now());
        }
        std::uint32_t const last_idx = static_cast<std::uint32_t>(chunks.size() - 1);
        finalize_chunk_timed(in_flight[last_idx % 2], last_idx);
        chunk_records[last_idx].per_device_gpu = timing_slots[last_idx % 2];
    }

    // Pretty-print per-chunk timeline. See generate_conformers epilogue below.
    {
        double const total_wall_ms = host_ms(std::chrono::high_resolution_clock::now());
        std::printf("\n=== chunk timeline (ms, host wall + per-device GPU phase) ===\n");
        std::printf("%4s %7s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n", "ck", "n_mol", "prep_lo", "prep_hi", "lnch_lo",
                    "lnch_hi", "fin_lo", "fin_hi", "gpu_max", "gpu_sum", "gpu_avg");
        for (auto const &cr : chunk_records) {
            float gpu_max = 0.f, gpu_sum = 0.f;
            std::uint32_t n_active = 0;
            for (auto const &pd : cr.per_device_gpu) {
                if (pd.total_ms <= 0.f) continue;
                gpu_max = std::max(gpu_max, pd.total_ms);
                gpu_sum += pd.total_ms;
                ++n_active;
            }
            float gpu_avg = n_active > 0 ? gpu_sum / n_active : 0.f;
            std::printf("%4u %7u %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f\n", cr.chunk_idx, cr.n_mol,
                        cr.prep_start_ms, cr.prep_end_ms, cr.launch_start_ms, cr.launch_end_ms, cr.finalize_start_ms,
                        cr.finalize_end_ms, gpu_max, gpu_sum, gpu_avg);
        }
        // Per-device per-chunk phase breakdown (compact).
        std::printf("\n=== per-device phase breakdown (ms): U=upload  M=metric  K=kernels  D=download  T=total ===\n");
        for (auto const &cr : chunk_records) {
            for (std::uint32_t d = 0; d < cr.per_device_gpu.size(); ++d) {
                auto const &pd = cr.per_device_gpu[d];
                if (pd.total_ms <= 0.f) continue;
                std::printf("ck=%u dev=%u  U=%5.1f  M=%5.1f  K=%6.1f  D=%5.1f  T=%6.1f\n", cr.chunk_idx, d,
                            pd.upload_ms, pd.metric_ms, pd.kernels_ms, pd.download_ms, pd.total_ms);
            }
        }
        std::printf("\nWall time (call total): %.1f ms\n", total_wall_ms);
    }

    // Invert the sort_perm permutation here so callers see
    // results in their original input order. host_* arrays are indexed by
    // sorted_idx (the position in our descending-size processing order); we
    // copy each element to position sort_perm[sorted_idx] in the caller's
    // output. The conformer-stride and coord-stride permutations are blocky
    // enough that this is cache-friendly.
    bool any_success = false;
    for (std::uint32_t sorted_idx = 0; sorted_idx < num_input; ++sorted_idx) {
        std::uint32_t const orig_idx = sort_perm[sorted_idx];
        for (std::uint32_t c = 0; c < num_conformers; ++c) {
            std::uint32_t const sorted_slot = sorted_idx * num_conformers + c;
            std::uint32_t const orig_slot = orig_idx * num_conformers + c;
            // Coords (coord_stride floats per slot).
            std::size_t const sorted_base = static_cast<std::size_t>(sorted_slot) * coord_stride;
            std::size_t const orig_base = static_cast<std::size_t>(orig_slot) * coord_stride;
            for (std::size_t k = 0; k < coord_stride; ++k)
                coordinates_out[orig_base + k] = __float2half(host_coords[sorted_base + k]);
            // Energy.
            energies_out[orig_slot] = host_energies[sorted_slot];
            // Status: preprocessed_ok was indexed by sorted_idx (set in launch_chunk
            // via chunk_base + batch_input_indices[i] which is in sorted space).
            if (!preprocessed_ok[sorted_idx]) { status_out[orig_slot] = conformer_status_t::invalid_smiles; }
            else {
                std::uint8_t const raw = host_status[sorted_slot];
                if (raw == 0) {
                    status_out[orig_slot] = conformer_status_t::success;
                    any_success = true;
                }
                else if (raw == 1) { status_out[orig_slot] = conformer_status_t::stereo_failure; }
                else { status_out[orig_slot] = conformer_status_t::mmff_unconverged; }
            }
        }
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
               bool use_csd_torsions, std::uint64_t random_seed) {
                // Build config.
                smallfield::api::config_t config;
                config.num_conformers = num_conformers;
                config.max_stereo_attempts = max_stereo_attempts;
                config.use_csd_torsions = use_csd_torsions;
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
            py::arg("max_stereo_attempts") = 10, py::arg("use_csd_torsions") = true, py::arg("random_seed") = 42);
}

#endif // SMALLFIELD_BUILD_PYTHON
