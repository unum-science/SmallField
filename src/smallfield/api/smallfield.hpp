/**
 *  @file smallfield.hpp
 *  @brief Public C++ API for SmallField conformer generation.
 *
 *  Single header for C++ users. Provides batch GPU conformer generation
 *  with automatic multi-GPU dispatch, wave-based scheduling, and
 *  half-precision coordinate output.
 *
 *  @code{.cpp}
 *  smallfield::context_t ctx;  // all GPUs
 *  smallfield::config_t config;
 *  config.num_conformers = 10;
 *
 *  std::vector<std::string_view> smiles = {"CCO", "c1ccccc1", ...};
 *  std::vector<__half> coords(smiles.size() * config.num_conformers * 128 * 3);
 *  std::vector<float> energies(smiles.size() * config.num_conformers);
 *  std::vector<smallfield::conformer_status_t> status(smiles.size() * config.num_conformers);
 *
 *  ctx.generate_conformers(smiles, config, 128, coords, energies, status);
 *  @endcode
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__) || defined(__CUDA_RUNTIME_API_H__)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#else
// forward-declare for non-CUDA translation units
struct __half;
typedef void *cudaStream_t;
typedef void *cusolverDnHandle_t;
typedef void *syevjInfo_t;
typedef void *cusolverDnParams_t;
#endif

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "smallfield/api/batch_pool.hpp"

namespace smallfield::preprocess {
class scaffold_cache_t;
struct scaffold_cache_config_t;
struct scaffold_cache_stats_t;
} // namespace smallfield::preprocess

namespace smallfield::api {

/**
 *  @brief Per-conformer status code.
 */
enum class conformer_status_t : std::uint8_t {
    success = 0,
    stereo_failure = 1,
    mmff_unconverged = 2,
    invalid_smiles = 3,
    too_many_atoms = 4,
};

/**
 *  @brief Pipeline configuration.
 */
struct config_t {
    std::uint32_t num_conformers = 3;
    std::uint32_t max_stereo_attempts = 10;
    /** @brief Target chunk size for pipelined dispatch; 0 = VRAM-aware auto. */
    std::uint32_t chunk_size = 0;
    /** @brief Emit per-chunk and per-device breadcrumb lines when true. */
    bool verbose = false;
    std::uint64_t random_seed = 42;
};

/**
 *  @brief GPU context managing CUDA streams and cuSOLVER handles.
 *
 *  Reuse across multiple generate_conformers() calls to avoid re-initialization.
 */
/**
 *  @brief Per-GPU persistent scratch buffers for the metric-embedding prepass.
 *
 *  Allocated once per @c context_t so the prepass can run fully stream-async
 *  without per-chunk @c cudaMalloc / @c cudaFree pairs that otherwise serialize
 *  the launch phase.  Sized for the worst case: all molecules in a chunk having
 *  the maximum supported atom count.
 */
struct metric_scratch_t {
    device_byte_buffer_t pool; ///< owning grow-only device buffer, carved below
    // Non-owning spans into @c pool. Rebind in @c ensure_metric_scratch on growth.
    std::span<std::uint32_t> group_indices {}; ///< [max_group_size]
    std::span<float> gram_matrices {};         ///< [max_group_size × max_atoms²]
    std::span<float> eigenvalues {};           ///< [max_group_size × max_atoms]
    std::span<int> info {};                    ///< [max_group_size]
    std::span<float> workspace {};             ///< device workspace for cusolverDnXsyevBatched
    std::vector<std::byte> host_workspace;     ///< host workspace for cusolverDnXsyevBatched
    std::size_t max_group_size = 0;
    std::uint32_t max_atoms = 0;
    std::size_t max_device_workspace_bytes = 0;
    std::size_t max_host_workspace_bytes = 0;
    cusolverDnParams_t xsyev_params = nullptr; ///< shared across calls — reused
};

// Forward-declare persistent per-device launch scratch. Defined in smallfield.cu
// so the public header does not need CUDA RAII wrappers.
struct launch_scratch_t;
using smallfield::preprocess::scaffold_cache_config_t;
using smallfield::preprocess::scaffold_cache_stats_t;
using smallfield::preprocess::scaffold_cache_t;

struct context_t {
    std::vector<int> device_ids;
    std::vector<cudaStream_t> streams;
    std::vector<cusolverDnHandle_t> solvers;
    std::vector<metric_scratch_t> metric_scratch;                ///< one per device
    std::vector<std::unique_ptr<launch_scratch_t>> launch_scratch; ///< one per device
    /**
     *  @brief Persistent 2-slot ring of batch prep buffers.
     *
     *  Moving the chunk-prep buffers out of @c generate_conformers and onto the
     *  context fixes a use-after-free: a later chunk's @c prepare_molecule_batch_into
     *  can trigger @c host.ensure_capacity, reallocating the pinned pointer while an
     *  earlier chunk still has H2D commands queued against the old address.  With the
     *  ring persistent and pre-sized at call entry, no reallocation happens mid-call.
     */
    std::array<batch_pool_t, 2> batch_ring {};
    /**
     *  @brief Optional scaffold cache, disabled by default.  When enabled,
     *  SubstructMatch calls for the CSD torsion patterns that remain RDKit
     *  fallback (patterns not yet handled natively) are amortised across
     *  molecules sharing the same Bemis–Murcko scaffold.  Enable via
     *  @ref enable_scaffold_cache; stats via @ref scaffold_cache_stats.
     */
    std::unique_ptr<scaffold_cache_t> scaffold_cache;

    /**
     *  @brief Construct for the given GPUs.
     *
     *  @param[in] devices GPU device IDs. Empty span = all available GPUs.
     */
    explicit context_t(std::span<int const> devices = {});
    ~context_t();

    /**
     *  @brief Opt in to scaffold-based SubstructMatch caching.  Must be called
     *  before the first @ref generate_conformers call.  No-op if already
     *  enabled.  Cache lives for the lifetime of the context_t.
     */
    void enable_scaffold_cache();
    /** @brief Snapshot of hit / miss / entry-count counters. */
    scaffold_cache_stats_t get_scaffold_cache_stats() const;

    context_t(context_t const &) = delete;
    context_t &operator=(context_t const &) = delete;

    /**
     *  @brief Generate 3D conformers for a batch of molecules.
     *
     *  @param[in] smiles Input SMILES strings (one per molecule).
     *  @param[in] config Pipeline configuration.
     *  @param[in] max_atoms_per_molecule Maximum atom count (with H) across the batch.
     *      Molecules exceeding this are skipped.
     *  @param[out] coordinates_out Output half-precision coordinates. Size: molecules * conformers * max_atoms * 3.
     *  @param[out] energies_out Output MMFF94 energies (kcal/mol). Size: molecules * conformers.
     *  @param[out] status_out Output per-conformer status. Size: molecules * conformers.
     *  @return Batch-level status (success if at least one conformer succeeded).
     */
    conformer_status_t generate_conformers(std::span<std::string_view const> smiles, config_t const &config,
                                           std::uint32_t max_atoms_per_molecule, std::span<__half> coordinates_out,
                                           std::span<float> energies_out, std::span<conformer_status_t> status_out);
};

/**
 *  @brief Process-wide CUDA error accumulator, incremented by the library's internal
 *  `check_cuda_status_` helper every time a CUDA API call returns non-success.
 *
 *  Inline so a single instance exists across every translation unit that includes this header.
 *  Callers use the @ref cuda_error_count / @ref reset_cuda_error_count accessors.
 */
inline std::atomic<std::uint64_t> cuda_error_counter {0};

/**
 *  @brief Read the cumulative CUDA error count since process start or the last reset.
 *
 *  Exposed so production callers and benchmarks can detect silent failures that would otherwise
 *  hide behind per-chunk zero-filled outputs.  A robust driver calls @ref reset_cuda_error_count
 *  before each batch, invokes @ref generate_conformers, and exits non-zero if this returns
 *  a positive value.
 */
inline std::uint64_t cuda_error_count() noexcept { return cuda_error_counter.load(std::memory_order_relaxed); }

/** @brief Reset @ref cuda_error_count to zero. */
inline void reset_cuda_error_count() noexcept { cuda_error_counter.store(0, std::memory_order_relaxed); }

} // namespace smallfield::api
