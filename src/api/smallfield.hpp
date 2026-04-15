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

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace smallfield {

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
    bool use_csd_torsions = true;
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
    void *pool = nullptr; ///< one contiguous device allocation, carved below
    std::size_t pool_bytes = 0;
    // Non-owning spans into @c pool.  Rebind in @c ensure_metric_scratch on growth.
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

// Forward-declare persistent per-device launch scratch + per-context output staging.
// Defined in smallfield.cu so the public header does not need CUDA RAII wrappers.
struct launch_scratch_t;
struct output_staging_t;
class scaffold_cache_t;
struct scaffold_cache_config_t;
struct scaffold_cache_stats_t;

struct context_t {
    std::vector<int> device_ids;
    std::vector<cudaStream_t> streams;
    std::vector<cusolverDnHandle_t> solvers;
    std::vector<metric_scratch_t> metric_scratch;   ///< one per device
    std::vector<launch_scratch_t *> launch_scratch; ///< one per device, heap-allocated
    output_staging_t *output_staging = nullptr;     ///< shared across devices, pinned
    /**
     *  @brief Optional scaffold cache, disabled by default.  When enabled,
     *  SubstructMatch calls for the CSD torsion patterns that remain RDKit
     *  fallback (patterns not yet handled natively) are amortised across
     *  molecules sharing the same Bemis–Murcko scaffold.  Enable via
     *  @ref enable_scaffold_cache; stats via @ref scaffold_cache_stats.
     */
    scaffold_cache_t *scaffold_cache = nullptr;

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

} // namespace smallfield
