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

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__) || defined(__CUDA_RUNTIME_API_H__)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#else
/** Forward-declares for non-CUDA translation units. */
struct __half;
using cudaStream_t = void *;
#endif

#include <array>       // `std::array`
#include <atomic>      // `std::atomic`
#include <cstdint>     // `std::uint32_t`, `std::uint64_t`
#include <memory>      // `std::unique_ptr`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include "smallfield/api/wave_slice_pool.hpp"
#include "smallfield/types/atom_limited_bins.hpp"
#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/api/conformer_status.hpp"
#include "smallfield/log.hpp"

namespace smallfield::api {

/*  `conformer_status_t` is defined in `smallfield/api/conformer_status.hpp`, transitively included
 *  above, so `wave_slice_pool.hpp` can name it without a cyclic dependency on this header. */

/**
 *  @brief Pipeline configuration.
 */
struct config_t {
    std::uint32_t num_conformers = 3;
    std::uint32_t max_stereo_attempts = 10;
    /** Target `embedding-wave` size for pipelined dispatch; 0 = VRAM-aware auto. */
    std::uint32_t wave_size = 0;
    std::uint64_t random_seed = 42;
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
    /** Owning grow-only device buffer. */
    device_byte_buffer_t pool;
    /** Non-owning span into `pool`, sized `[max_group_size]` — rebound in
     *  `ensure_metric_scratch` on growth. */
    std::span<std::uint32_t> group_indices {};
    std::size_t max_group_size = 0;
    std::uint32_t max_atoms = 0;
    /*  `embed_4d_power_iter_kernel` holds the Gram matrix, means, eigenvectors and reduction scratch
     *  in shared memory for its whole lifetime, so the only persistent device scratch is
     *  `group_indices` — a per-bin uint32 list of which molecule goes into which bin. */
};

/**
 *  @brief Forward-declares persistent per-device launch scratch.
 *
 *  Defined in `smallfield.cu` so the public header does not need CUDA RAII wrappers.
 */
struct launch_scratch_t;

/**
 *  @brief GPU context managing CUDA streams and cuSOLVER handles.
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
     *  @brief Persistent per-device 2-slot pinned batch pool ring.
     *
     *  Each GPU owns its own pinned source buffer per slot; preprocess writes
     *  every molecule's bytes into the pool assigned to that molecule's target
     *  device, and the per-device `cudaMemcpyAsync H2D` reads only from its
     *  own pool. Eliminates the concurrent-pinned-source-DMA pattern that
     *  arose when all devices read the same shared pinned buffer.
     */
    std::array<std::array<wave_slice_pool_t, max_devices_k>, 2> batch_slots_per_device {};
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
     *  @brief Per-context CUDA error accumulator, incremented by `check_cuda_status_` every time a CUDA /
     *  cuSOLVER API call returns a non-success status. Production callers read this after each
     *  `generate_conformers` call and exit non-zero if it is positive, catching silent failures that would
     *  otherwise hide behind per-wave zero-filled outputs. Reset via `context.cuda_error_count.store(0)`.
     */
    std::atomic<std::uint64_t> cuda_error_count {0};

    /**
     *  @brief Construct for the given GPUs.
     *
     *  @param[in] devices GPU device IDs. Empty span = all available GPUs.
     */
    explicit context_t(std::span<int const> devices = {});
    ~context_t();

    context_t(context_t const &) = delete;
    context_t &operator=(context_t const &) = delete;

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
        std::span<std::string_view const> smiles, config_t const &config, std::uint32_t max_atoms_per_molecule,
        std::span<__half> coordinates_out, std::span<float> energies_out, std::span<conformer_status_t> status_out,
        std::span<std::uint8_t> num_atoms_out = {}, std::span<std::uint16_t> num_heavy_atoms_out = {},
        std::span<std::uint16_t> num_bonds_out = {}, std::span<float> molecular_weight_out = {});
};

} // namespace smallfield::api
