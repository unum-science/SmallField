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

#if defined(SMALLFIELD_BUILD_PYTHON)
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

#include "batch_pool.hpp"
#include "etkdg_pipeline.cuh"
#include "etkdg_preprocess.hpp"
#include "metric_embedding_prepass.cuh"
#include "smallfield.hpp"

namespace smallfield {

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
        // Reuse one cusolverDnParams_t per device across all calls.  The
        // generic 64-bit batched eigen API (cusolverDnXsyevBatched) takes this
        // instead of the legacy syevjInfo_t — lower per-call CPU overhead.
        cusolverDnCreateParams(&metric_scratch[i].xsyev_params);
        launch_scratch[i] = new launch_scratch_t;
    }
}

context_t::~context_t() {
    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        cudaSetDevice(device_ids[i]);
        if (metric_scratch[i].xsyev_params) cusolverDnDestroyParams(metric_scratch[i].xsyev_params);
        if (metric_scratch[i].pool) cudaFree(metric_scratch[i].pool);
        delete launch_scratch[i]; // device_byte_buffer_t + pinned_byte_buffer_t free their holdings
        cusolverDnDestroy(solvers[i]);
        cudaStreamDestroy(streams[i]);
    }
    delete output_staging;
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
    // Region 3: kernel outputs (coords / energies / status).  Downloaded D2H
    //           to host_staging at the end of this function.
    // Region 4: GPU-only 4D scratch produced by the metric prepass.
    auto const sizeof_meta = sizeof(etkdg_molecule_metadata_t);
    std::size_t dev_offset = 0;
    std::size_t const dev_off_batch = align_up(dev_offset, 256);
    dev_offset = dev_off_batch + align_up(batch.layout.bytes_total, 256);
    std::size_t const dev_off_sorted_meta = dev_offset;
    dev_offset += align_up(local_count * sizeof_meta, 256);
    std::size_t const dev_off_coords = dev_offset;
    dev_offset += align_up(total_coords * sizeof(float), 256);
    std::size_t const dev_off_energies = dev_offset;
    dev_offset += align_up(total_conf * sizeof(float), 256);
    std::size_t const dev_off_status = dev_offset;
    dev_offset += align_up(total_conf * sizeof(std::uint8_t), 256);
    std::size_t const dev_off_initial = dev_offset;
    dev_offset += align_up(static_cast<std::size_t>(local_count) * max_atoms_per_molecule * 4 * sizeof(float), 256);
    std::size_t const dev_pool_bytes = dev_offset;

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

    if (!launch_buffers.device_pool.ensure_capacity(dev_pool_bytes)) {
        std::fprintf(stderr, "launch_gpu_pipeline: cudaMalloc(%zu bytes) failed on gpu %d\n", dev_pool_bytes,
                     device_id);
        return handle;
    }
    if (!launch_buffers.host_staging.ensure_capacity(host_pool_bytes)) {
        std::fprintf(stderr, "launch_gpu_pipeline: cudaHostAlloc(%zu bytes) failed on gpu %d\n", host_pool_bytes,
                     device_id);
        return handle;
    }
    auto _P_post_malloc = std::chrono::high_resolution_clock::now();

    std::byte *const dev_base = launch_buffers.device_pool.data();
    std::byte *const host_base = launch_buffers.host_staging.data();

    // Device view over the batch-input mirror — same layout as batch.host_view().
    batch_pool_dev_view_t dev_view {dev_base + dev_off_batch, &batch.layout};

    auto *device_metadata_sorted = reinterpret_cast<etkdg_molecule_metadata_t *>(dev_base + dev_off_sorted_meta);
    handle.device_coords_out = {reinterpret_cast<float *>(dev_base + dev_off_coords), total_coords};
    handle.device_energies_out = {reinterpret_cast<float *>(dev_base + dev_off_energies), total_conf};
    handle.device_status_out = {reinterpret_cast<std::uint8_t *>(dev_base + dev_off_status), total_conf};
    auto *device_initial_coordinates = reinterpret_cast<float *>(dev_base + dev_off_initial);

    auto *host_sorted_meta = reinterpret_cast<etkdg_molecule_metadata_t *>(host_base + host_off_sorted_meta);
    handle.host_coords_sorted = {reinterpret_cast<float *>(host_base + host_off_coords), total_coords};
    handle.host_energies_sorted = {reinterpret_cast<float *>(host_base + host_off_energies), total_conf};
    handle.host_status_sorted = {reinterpret_cast<std::uint8_t *>(host_base + host_off_status), total_conf};

    // Gather the sorted metadata directly into the pinned staging region so the
    // subsequent H2D does not degrade to synchronous.
    for (std::uint32_t i = 0; i < local_count; ++i) host_sorted_meta[i] = meta_full[local_indices[handle.order[i]]];

    // Replaces 14 pageable-source cudaMemcpyAsync calls with ONE pinned-source
    // transfer that actually returns immediately and runs on the stream.
    if (batch.layout.bytes_total > 0)
        cudaMemcpyAsync(dev_base + dev_off_batch, batch.host.data(), batch.layout.bytes_total, cudaMemcpyHostToDevice,
                        stream);
    if (local_count > 0)
        cudaMemcpyAsync(device_metadata_sorted, host_sorted_meta, local_count * sizeof_meta, cudaMemcpyHostToDevice,
                        stream);

    auto _P_post_upload = std::chrono::high_resolution_clock::now();
    build_metric_initial_coordinates_on_gpu(dev_view.bounds_matrices().data(), device_metadata_sorted,
                                            std::span<etkdg_molecule_metadata_t const> {host_sorted_meta, local_count},
                                            max_atoms_per_molecule, device_initial_coordinates, solver_handle, scratch,
                                            stream);
    auto _P_post_metric = std::chrono::high_resolution_clock::now();

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

        std::uint32_t const shared_bytes = compute_pipeline_shared_memory_bytes(wave_max_atoms, 0);
        cudaFuncSetAttribute(etkdg_pipeline_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);

        int blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, etkdg_pipeline_warp_kernel, 32, shared_bytes);
        std::uint32_t const grid_size = std::min(wave_size, static_cast<std::uint32_t>(blocks_per_sm * num_sms));

        etkdg_pipeline_warp_kernel<<<grid_size, 32, shared_bytes, stream>>>(
            dev_view.bounds_matrices().data(), device_initial_coordinates + wave_start * max_atoms_per_molecule * 4,
            dev_view.chiral_centers().data(), dev_view.etk_torsions().data(), dev_view.distance_constraints().data(),
            dev_view.etk_interactions().data(), dev_view.etk_atom_offsets().data(), dev_view.mmff_atom_vdw().data(),
            dev_view.mmff_atom_charge().data(), dev_view.bond_distance().data(), dev_view.mmff_bonds().data(),
            dev_view.mmff_angles().data(), dev_view.mmff_torsions().data(), dev_view.mmff_oops().data(),
            dev_view.mmff_stretchbends().data(), dev_view.mmff_interactions().data(),
            dev_view.mmff_atom_offsets().data(), device_metadata_sorted + wave_start,
            handle.device_coords_out.data() + wave_start * num_conformers * max_atoms_per_molecule * 3,
            handle.device_energies_out.data() + wave_start * num_conformers,
            handle.device_status_out.data() + wave_start * num_conformers, wave_size, num_conformers,
            max_stereo_attempts, wave_max_atoms, max_atoms_per_molecule, max_atoms_per_molecule, seed);

        wave_start = wave_end;
    }

    // Enqueue pinned D2H for all three output arrays.  Pinned destination means
    // these return immediately and the transfers run on the stream after the
    // kernels.  `finalize_gpu_pipeline` just waits on the stream and permutes.
    if (total_coords > 0)
        cudaMemcpyAsync(handle.host_coords_sorted.data(), handle.device_coords_out.data(), total_coords * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    if (total_conf > 0) {
        cudaMemcpyAsync(handle.host_energies_sorted.data(), handle.device_energies_out.data(),
                        total_conf * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(handle.host_status_sorted.data(), handle.device_status_out.data(),
                        total_conf * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream);
    }

    auto _P_end = std::chrono::high_resolution_clock::now();
    auto _ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::printf("  [gpu%d] setup=%.1f mallocpool=%.1f upload=%.1f metric=%.1f kernels_post=%.1f total=%.1f\n",
                device_id, _ms(_P0, _P_pre_malloc), _ms(_P_pre_malloc, _P_post_malloc),
                _ms(_P_post_malloc, _P_post_upload), _ms(_P_post_upload, _P_post_metric), _ms(_P_post_metric, _P_end),
                _ms(_P0, _P_end));
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

    // Grow the shared pinned output region in context_t to fit this call.  Allocated
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

    // Chunk policy: small batches take the non-pipelined path.  For large batches we
    // split into ~50 K-molecule chunks so CPU prep of chunk N+1 overlaps GPU work of
    // chunk N.  All GPU operations are stream-async, so the pipeline is driven by the
    // calling thread — no std::thread, no queue.
    std::uint32_t const chunk_size_target = 50'000;
    bool const pipeline = num_input > chunk_size_target * 3 / 2;

    std::uint32_t effective_max_atoms = max_atoms_per_molecule;

    // Launch fans out across GPUs with a pure single-threaded for-loop.  Every CUDA
    // call below is stream-async after the cusolver scratch is pre-allocated, so the
    // loop just enqueues work and returns.  GPU d's stream runs concurrently with
    // GPU d+1's enqueue and with the main thread's next prep_slice().
    auto launch_chunk = [&](etkdg_molecule_batch_t const &batch, std::uint32_t chunk_base,
                            std::vector<gpu_pipeline_handle_t> &handles_out) {
        auto batch_input_indices = batch.input_indices();
        for (std::uint32_t i = 0; i < batch.num_molecules; ++i)
            preprocessed_ok[chunk_base + batch_input_indices[i]] = true;
        if (batch.num_molecules == 0) return;

        std::vector<std::vector<std::uint32_t>> per_gpu_indices(num_devices);
        for (std::uint32_t i = 0; i < batch.num_molecules; ++i) per_gpu_indices[i % num_devices].push_back(i);

        // Round to the next metric-prepass bin boundary; cusolver runs at binned N,
        // not raw atoms, so the workspace + gram pool must fit that.  Must stay in
        // sync with bin_ns[] in build_metric_initial_coordinates_on_gpu.
        auto bin_ceiling = [](std::uint32_t n) -> std::uint32_t {
            static constexpr std::uint32_t bin_ns[] = {16, 32, 48, 64, 80, 96, 112, 128};
            for (auto b : bin_ns)
                if (n <= b) return b;
            return bin_ns[7];
        };
        std::uint32_t const metric_bin_atoms = bin_ceiling(effective_max_atoms);
        for (std::uint32_t d = 0; d < num_devices; ++d) {
            if (per_gpu_indices[d].empty()) continue;
            ensure_metric_scratch(*this, d, per_gpu_indices[d].size(), metric_bin_atoms);
        }

        std::size_t const conf_offset = static_cast<std::size_t>(chunk_base) * num_conformers;
        std::size_t const chunk_conformers = static_cast<std::size_t>(batch.num_molecules) * num_conformers;
        std::span<float> chunk_coords {host_coords.data() + conf_offset * coord_stride,
                                       chunk_conformers * coord_stride};
        std::span<float> chunk_energies {host_energies.data() + conf_offset, chunk_conformers};
        std::span<std::uint8_t> chunk_status {host_status.data() + conf_offset, chunk_conformers};

        handles_out.resize(num_devices);
        // Dispatcher runs the 8 launches in parallel on 8 OpenMP threads.  Each
        // launch call takes ~500 ms of CPU-side work (mostly cusolver state update
        // for the metric prepass), and the eight per-device state structs
        // (streams[d], solvers[d], metric_scratch[d], launch_scratch[d]) are
        // isolated, so the parallelization is safe and effectively free.
        // schedule(static, 1) pins one thread per device so cudaSetDevice stickiness
        // is deterministic.
#pragma omp parallel for num_threads(num_devices) schedule(static, 1)
        for (std::uint32_t d = 0; d < num_devices; ++d) {
            if (per_gpu_indices[d].empty()) continue;
            auto _lt0 = std::chrono::high_resolution_clock::now();
            handles_out[d] = launch_gpu_pipeline(device_ids[d], streams[d], solvers[d], metric_scratch[d],
                                                 *launch_scratch[d], batch, per_gpu_indices[d], num_conformers,
                                                 config.max_stereo_attempts, effective_max_atoms, config.random_seed,
                                                 chunk_coords, chunk_energies, chunk_status);
            auto _lt1 = std::chrono::high_resolution_clock::now();
            std::printf("[launch gpu%d] %.1fms\n", d, std::chrono::duration<double, std::milli>(_lt1 - _lt0).count());
            std::fflush(stdout);
        }
    };

    // Finalize walks the 8 per-GPU handles sequentially.  The first finalize's
    // cudaStreamSynchronize blocks until that stream drains; by the time it returns,
    // the other 7 GPUs have almost always finished their chunk too (they started
    // within microseconds of each other and have the same work), so the remaining
    // finalize calls are near-instant.
    auto finalize_chunk = [&](std::vector<gpu_pipeline_handle_t> &handles) {
        for (auto &h : handles)
            if (h.device_id >= 0) finalize_gpu_pipeline(std::move(h));
        handles.clear();
    };

    if (!pipeline) {
        // Non-pipelined: single prep, single launch, single finalize.  Identical
        // behaviour to the pre-pipelining code path.
        etkdg_molecule_batch_t batch = prepare_molecule_batch(smiles);
        effective_max_atoms = std::max(max_atoms_per_molecule, batch.max_atoms_per_molecule);
        std::vector<gpu_pipeline_handle_t> handles;
        launch_chunk(batch, /*chunk_base=*/0, handles);
        finalize_chunk(handles);
    }
    else {
        // Pipelined: split input into ~50K-molecule chunks, double-buffer the batches,
        // interleave CPU prep of chunk N+1 with stream-async GPU work of chunk N.
        std::vector<std::pair<std::uint32_t, std::uint32_t>> chunks; // [begin, end)
        for (std::uint32_t i = 0; i < num_input; i += chunk_size_target)
            chunks.emplace_back(i, std::min(i + chunk_size_target, num_input));

        auto prep_slice = [&](std::uint32_t begin, std::uint32_t end) {
            return prepare_molecule_batch(smiles.subspan(begin, end - begin));
        };

        // Two-slot ring buffer for both batches and handles.  Fixed-size
        // `std::array` keeps everything in contiguous storage — no vector growth,
        // no indirection per access, and `gpu_pipeline_handle_t::batch` can hold
        // a stable pointer into `batches[slot]` for the lifetime of one chunk.
        // At most one chunk is in flight (finalize_chunk(i-1) runs before
        // launch_chunk(i) overwrites the opposite slot), so two slots suffice.
        std::array<etkdg_molecule_batch_t, 2> batches {};
        std::array<std::vector<gpu_pipeline_handle_t>, 2> in_flight {};
        in_flight[0].resize(num_devices);
        in_flight[1].resize(num_devices);

        // Bootstrap: prep + launch chunk 0.  chunk 0's GPU work runs in the background
        // on the CUDA streams while the main-thread loop below preps chunk 1.
        batches[0] = prep_slice(chunks[0].first, chunks[0].second);
        effective_max_atoms = std::max(effective_max_atoms, batches[0].max_atoms_per_molecule);
        launch_chunk(batches[0], chunks[0].first, in_flight[0]);

        // Main pipeline loop.  At the top of iteration i:
        //   - GPU work for chunk i-1 is in flight in slot (i-1) % 2.
        //   - Batch slot (i % 2) held chunk i-2, which was finalized at iteration i-1.
        //     So it's safe to overwrite with chunk i's batch.
        for (std::size_t i = 1; i < chunks.size(); ++i) {
            batches[i % 2] = prep_slice(chunks[i].first, chunks[i].second); // CPU work
            effective_max_atoms = std::max(effective_max_atoms, batches[i % 2].max_atoms_per_molecule);
            finalize_chunk(in_flight[(i - 1) % 2]);                          // sync + download
            launch_chunk(batches[i % 2], chunks[i].first, in_flight[i % 2]); // async dispatch
        }
        finalize_chunk(in_flight[(chunks.size() - 1) % 2]);
    }

    // Convert float32 coordinates to __half and write to output span.
    for (std::size_t i = 0; i < static_cast<std::size_t>(total_conf) * coord_stride; ++i)
        coordinates_out[i] = __float2half(host_coords[i]);

    // Map uint8_t kernel status to conformer_status_t and write to output span.
    bool any_success = false;
    for (std::uint32_t mol = 0; mol < num_input; ++mol) {
        for (std::uint32_t c = 0; c < num_conformers; ++c) {
            std::uint32_t const slot = mol * num_conformers + c;
            energies_out[slot] = host_energies[slot];

            if (!preprocessed_ok[mol]) { status_out[slot] = conformer_status_t::invalid_smiles; }
            else {
                std::uint8_t const raw = host_status[slot];
                if (raw == 0) {
                    status_out[slot] = conformer_status_t::success;
                    any_success = true;
                }
                else if (raw == 1) { status_out[slot] = conformer_status_t::stereo_failure; }
                else { status_out[slot] = conformer_status_t::mmff_unconverged; }
            }
        }
    }

    return any_success ? conformer_status_t::success : conformer_status_t::mmff_unconverged;
}

} // namespace smallfield

#if defined(SMALLFIELD_BUILD_PYTHON)

PYBIND11_MODULE(_smallfield, m) {
    m.doc() = "SmallField: GPU ETKDG + MMFF94 conformer generation";

    m.def("supports_cuda", []() { return true; });

    py::enum_<smallfield::conformer_status_t>(m, "ConformerStatus")
        .value("success", smallfield::conformer_status_t::success)
        .value("stereo_failure", smallfield::conformer_status_t::stereo_failure)
        .value("mmff_unconverged", smallfield::conformer_status_t::mmff_unconverged)
        .value("invalid_smiles", smallfield::conformer_status_t::invalid_smiles)
        .value("too_many_atoms", smallfield::conformer_status_t::too_many_atoms);

    py::class_<smallfield::context_t>(m, "Context", "GPU context managing CUDA streams and cuSOLVER handles.")
        .def(py::init([](std::vector<int> device_ids) {
                 std::span<int const> span(device_ids.data(), device_ids.size());
                 return std::make_unique<smallfield::context_t>(span);
             }),
             py::arg("device_ids") = std::vector<int> {})
        .def(
            "generate_conformers",
            [](smallfield::context_t &ctx, std::vector<std::string> const &smiles_list, std::uint32_t num_conformers,
               std::uint32_t max_atoms_per_molecule, std::uint32_t max_stereo_attempts, bool use_csd_torsions,
               std::uint64_t random_seed) {
                // Build config.
                smallfield::config_t config;
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
                std::vector<smallfield::conformer_status_t> status_buf(total_conf);
                std::span<smallfield::conformer_status_t> status_span(status_buf);

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
