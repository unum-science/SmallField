/**
 *  @file src/smallfield/api/wave_capture.hpp
 *  @brief One-shot snapshot of everything the two warp kernels read for a single atom-limited bin.
 *
 *  Every NVIDIA profiler crashes a binary that links the conda stack, so a kernel measurement has to
 *  happen in a process that links nothing but the CUDA runtime. This writes the launch of one bin to
 *  disk — the batch pool, the metric embedding, the byte offset of each pointer argument inside them,
 *  and the launch geometry — and `bench/bench_kernel_replay.cu` reads it back and launches the same two
 *  kernels with the same arguments.
 *
 *  The capture is off unless `SMALLFIELD_CAPTURE_DIR` names a writable directory, and it fires for the
 *  first bin whose atom cap equals `SMALLFIELD_CAPTURE_BIN`, once per process.
 */
#pragma once

#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstdio>  // `std::fopen`, `std::fprintf`
#include <cstdlib> // `std::getenv`, `std::strtoul`
#include <cstring> // `std::memcpy`

#include <atomic> // `std::atomic`
#include <string>  // `std::string`
#include <utility> // `std::pair`
#include <vector>  // `std::vector`

#include <cuda_runtime.h>

#include "smallfield/api/wave_slice_pool.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::api {

/** The launch geometry and the scalar arguments one captured bin replays with. */
struct wave_capture_scalars_t {
    std::uint32_t bin_start = 0;
    std::uint32_t bin_size = 0;
    std::uint32_t conformer_count = 0;
    std::uint32_t max_stereo_attempts = 0;
    std::uint32_t bin_max_atoms = 0;
    std::uint32_t gpu_max_atoms = 0;
    std::uint32_t grid_etkdg = 0;
    std::uint32_t grid_mmff = 0;
    std::uint32_t shared_bytes_etkdg = 0;
    std::uint32_t shared_bytes_mmff = 0;
    std::uint64_t random_seed = 0;
};

/** The atom cap of the bin the capture waits for, or zero when `SMALLFIELD_CAPTURE_DIR` is unset. */
inline std::uint32_t capture_bin_atom_cap() noexcept {
    static std::uint32_t const atom_cap = []() -> std::uint32_t {
        char const *const directory = std::getenv("SMALLFIELD_CAPTURE_DIR");
        if (directory == nullptr || directory[0] == '\0') return 0;
        char const *const bin = std::getenv("SMALLFIELD_CAPTURE_BIN");
        return bin == nullptr ? 48u : static_cast<std::uint32_t>(std::strtoul(bin, nullptr, 10));
    }();
    return atom_cap;
}

/** Copy @p bytes device bytes at @p source into @p path, reporting on `stderr` what it could not do. */
inline bool write_device_region_(std::string const &path, void const *source, std::size_t bytes) {
    std::vector<std::byte> staged(bytes);
    if (cudaError_t const status = cudaMemcpy(staged.data(), source, bytes, cudaMemcpyDeviceToHost);
        status != cudaSuccess) {
        std::fprintf(stderr, "wave_capture: %s could not be read back: %s\n", path.c_str(), cudaGetErrorString(status));
        return false;
    }
    std::FILE *const file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        std::fprintf(stderr, "wave_capture: %s could not be opened\n", path.c_str());
        return false;
    }
    std::size_t const written = std::fwrite(staged.data(), 1, bytes, file);
    std::fclose(file);
    if (written != bytes) {
        std::fprintf(stderr, "wave_capture: %s took %zu of %zu bytes\n", path.c_str(), written, bytes);
        return false;
    }
    return true;
}

/** Whether a bin was the one the capture was waiting for, which is also what decides if it is timed. */
enum class capture_outcome_t { skipped_k, captured_k };

/** The pair of events bracketing the captured bin's two launches, so the replay has a number to match. */
inline std::pair<cudaEvent_t, cudaEvent_t> capture_timing_events_() {
    static std::pair<cudaEvent_t, cudaEvent_t> const events = [] {
        cudaEvent_t before = nullptr, after = nullptr;
        cudaEventCreate(&before);
        cudaEventCreate(&after);
        return std::pair<cudaEvent_t, cudaEvent_t> {before, after};
    }();
    return events;
}

/** Close the bracket `capture_bin_launch` opened and print what the pipeline's own launch cost. */
inline void report_captured_bin_timing(cudaStream_t stream) {
    auto const events = capture_timing_events_();
    cudaEventRecord(events.second, stream);
    cudaEventSynchronize(events.second);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, events.first, events.second);
    std::fprintf(stderr, "wave_capture: pipeline pair_ms=%.1f\n", static_cast<double>(elapsed_ms));
}

/**
 *  @brief Write one bin's launch to `SMALLFIELD_CAPTURE_DIR`, once per process.
 *
 *  Called from the enqueue loop immediately before the two launches, so the pointer arithmetic here and
 *  the pointer arithmetic in the launch read the same locals. The stream is synchronized first, because
 *  the preprocess kernels that fill the batch pool are still in flight at that point.
 *
 *  @param[in] device_view Device mirror of the batch pool, whose `base` anchors every recorded offset.
 *  @param[in] batch_bytes Bytes the batch mirror spans.
 *  @param[in] metric_embedding The whole wave's four-dimensional embedding region.
 *  @param[in] scalars Launch geometry and scalar kernel arguments for this bin.
 *  @param[in] stream Stream the preprocessing rode, drained before the read-back.
 */
inline capture_outcome_t capture_bin_launch(batch_pool_view_t device_view, std::size_t batch_bytes,
                                            custd::span<float const> metric_embedding,
                                            wave_capture_scalars_t const &scalars, cudaStream_t stream) {
    if (scalars.bin_max_atoms != capture_bin_atom_cap()) return capture_outcome_t::skipped_k;
    // The enqueue loop runs one thread per device, so this latch is read and written concurrently. An
    // exchange is what makes "once per process" true rather than aspirational: two devices passing it
    // would both write `manifest.txt`, and the timing events below - created lazily by whoever arrives
    // first - would be recorded onto a stream belonging to the other one.
    static std::atomic<bool> captured {false};
    if (captured.exchange(true, std::memory_order_relaxed)) return capture_outcome_t::skipped_k;

    std::string const directory = std::getenv("SMALLFIELD_CAPTURE_DIR");
    cudaStreamSynchronize(stream);

    auto const *const batch_base = reinterpret_cast<char const *>(device_view.base);
    auto offset_of_ = [batch_base](void const *pointer) -> std::size_t {
        return static_cast<std::size_t>(reinterpret_cast<char const *>(pointer) - batch_base);
    };

    // The nine bin-relative pointers the two launches build, spelled here exactly as they are spelled there.
    auto const *const metadata = device_view.molecule_metadata().data() + scalars.bin_start;
    auto const *const input_indices = device_view.input_indices().data() + scalars.bin_start;
    std::size_t const embedding_offset = static_cast<std::size_t>(scalars.bin_start) * scalars.gpu_max_atoms *
                                         types::embedding_dimensions_k;

    std::string const manifest_path = directory + "/manifest.txt";
    std::FILE *const manifest = std::fopen(manifest_path.c_str(), "w");
    if (manifest == nullptr) {
        std::fprintf(stderr, "wave_capture: %s could not be opened\n", manifest_path.c_str());
        return capture_outcome_t::skipped_k;
    }
    auto line_ = [manifest](char const *key, std::size_t value) { std::fprintf(manifest, "%s=%zu\n", key, value); };
    line_("bin_size", scalars.bin_size);
    line_("conformer_count", scalars.conformer_count);
    line_("max_stereo_attempts", scalars.max_stereo_attempts);
    line_("bin_max_atoms", scalars.bin_max_atoms);
    line_("gpu_max_atoms", scalars.gpu_max_atoms);
    line_("grid_etkdg", scalars.grid_etkdg);
    line_("grid_mmff", scalars.grid_mmff);
    line_("shared_bytes_etkdg", scalars.shared_bytes_etkdg);
    line_("shared_bytes_mmff", scalars.shared_bytes_mmff);
    line_("random_seed", static_cast<std::size_t>(scalars.random_seed));
    line_("batch_bytes", batch_bytes);
    line_("embedding_floats", metric_embedding.size());
    line_("embedding_offset_floats", embedding_offset);
    line_("offset_bounds_matrices", offset_of_(device_view.bounds_matrices().data()));
    line_("offset_chiral_centers", offset_of_(device_view.chiral_centers().data()));
    line_("offset_etk_torsions", offset_of_(device_view.etk_torsions().data()));
    line_("offset_distance_constraints", offset_of_(device_view.distance_constraints().data()));
    line_("offset_etk_interactions", offset_of_(device_view.etk_interactions().data()));
    line_("offset_etk_atom_offsets", offset_of_(device_view.etk_atom_offsets().data()));
    line_("offset_molecule_metadata", offset_of_(metadata));
    line_("offset_input_indices", offset_of_(input_indices));
    line_("offset_mmff_atom_vdw", offset_of_(device_view.mmff_atom_vdw().data()));
    line_("offset_mmff_atom_charge", offset_of_(device_view.mmff_atom_charge().data()));
    line_("offset_bond_distance", offset_of_(device_view.bond_distance().data()));
    line_("offset_mmff_bonds", offset_of_(device_view.mmff_bonds().data()));
    line_("offset_mmff_angles", offset_of_(device_view.mmff_angles().data()));
    line_("offset_mmff_torsions", offset_of_(device_view.mmff_torsions().data()));
    line_("offset_mmff_oops", offset_of_(device_view.mmff_oops().data()));
    line_("offset_mmff_stretchbends", offset_of_(device_view.mmff_stretchbends().data()));
    line_("offset_mmff_interactions", offset_of_(device_view.mmff_interactions().data()));
    line_("offset_mmff_atom_offsets", offset_of_(device_view.mmff_atom_offsets().data()));
    std::fclose(manifest);

    bool const wrote_batch = write_device_region_(directory + "/batch.bin", device_view.base, batch_bytes);
    bool const wrote_embedding = write_device_region_(directory + "/embedding.bin", metric_embedding.data(),
                                                      metric_embedding.size() * sizeof(float));
    std::fprintf(stderr, "wave_capture: bin_max_atoms=%u bin_size=%u batch_bytes=%zu embedding_floats=%zu ok=%d\n",
                 scalars.bin_max_atoms, scalars.bin_size, batch_bytes, metric_embedding.size(),
                 static_cast<int>(wrote_batch && wrote_embedding));

    cudaEventRecord(capture_timing_events_().first, stream);
    return capture_outcome_t::captured_k;
}

} // namespace smallfield::api
