/**
 *  @file smallfield_mace.hpp
 *  @brief Public C++ API for SmallField MACE-OFF inference.
 *
 *  Composes naturally with `smallfield::api::context_t::generate_conformers`:
 *
 *  @code{.cpp}
 *  // Stage 1 — ETKDG + MMFF94 (existing API)
 *  smallfield::api::context_t etkdg_ctx;
 *  std::vector<__half> coordinates(...);
 *  etkdg_ctx.generate_conformers(smiles, ..., coordinates, energies, status);
 *
 *  // Stage 2 — MACE-OFF refinement
 *  smallfield::mace::context_t mace_ctx;
 *  auto W = smallfield::mace::weights_t::load("MACE-OFF23-small.weights.bin");
 *  mace_ctx.refine_batch(W, batch_view, {.num_steps = 20});
 *  @endcode
 *
 *  The MACE-OFF inference module is functional but historically required
 *  callers to use the raw `cuda::mace_inference_*` driver functions.  This
 *  header gives a stable, RAII-style C++ surface that hides the scratch
 *  arena, captured-graph cache, and weight-loading details.
 *
 *  Phase H of the optimisation roadmap (`MACE_OPTIMIZATION_ROADMAP.md`).
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace smallfield::types {
struct mace_arch_t;
struct mace_batch_t;
} // namespace smallfield::types

namespace smallfield::mace {

/**
 *  @brief Per-call status.
 */
enum class status_t : std::uint8_t {
    success = 0,
    invalid_input = 1,
    weights_load_failed = 2,
    cuda_error = 3,
    not_supported = 4,
};

/**
 *  @brief Configuration for batched LBFGS refinement.
 */
struct relax_config_t {
    /** @brief Maximum LBFGS steps per conformer. */
    std::uint32_t num_steps = 20;
    /** @brief Convergence threshold on per-atom force magnitude (eV / Å). */
    float fmax_eV_per_A = 0.05f;
    /** @brief Initial gradient-descent step (LBFGS scales it via the line search). */
    float step_size = 0.01f;
};

/**
 *  @brief RAII wrapper around the device-resident MACE weight set.
 *
 *  Loads weights from the `.bin` produced by
 *  `python/smallfield/mace_export.py`.  Construction copies the weights to
 *  the GPU once; reused by every `context_t::infer` / `context_t::refine`
 *  call.
 */
class weights_t {
  public:
    weights_t();
    ~weights_t();
    weights_t(weights_t &&) noexcept;
    weights_t &operator=(weights_t &&) noexcept;
    weights_t(weights_t const &) = delete;
    weights_t &operator=(weights_t const &) = delete;

    /**
     *  @brief Load weights from the SmallField .bin format.
     *  @throws std::runtime_error on I/O or format error.
     */
    static weights_t load(std::string const &path);

    /** @brief Architecture metadata (n_channels, num_interactions, etc.). */
    smallfield::types::mace_arch_t const &arch() const noexcept;

    /** @return non-null on success; nullptr if no weights are loaded. */
    bool is_loaded() const noexcept;

    /** @brief Internal device-side weights handle for the kernel driver. */
    void const *device_weights() const noexcept;

  private:
    struct impl_t;
    std::unique_ptr<impl_t> p_;
};

/**
 *  @brief A view of a packed batch of conformers, ready for inference.
 *
 *  The MACE inference driver consumes already-packed batches: a flat list
 *  of atoms (positions, elements, graph membership) plus a flat edge list
 *  with explicit source / target indices.  This struct makes the layout
 *  explicit at the API boundary.
 *
 *  All pointers are device-resident.  Lifetime of the underlying buffers
 *  is the caller's responsibility.
 */
struct batch_view_t {
    std::uint32_t n_atoms;  ///< Total atoms across all conformers.
    std::uint32_t n_edges;  ///< Total edges across all conformers.
    std::uint32_t n_graphs; ///< Number of conformers in this batch.

    float const *positions_xyz;           ///< [n_atoms × 3], fp32
    std::int32_t const *atom_element;     ///< [n_atoms], element index in [0, n_elements)
    std::int32_t const *edge_source_atom; ///< [n_edges]
    std::int32_t const *edge_target_atom; ///< [n_edges]
    std::int32_t const *graph_of_atom;    ///< [n_atoms], conformer index in [0, n_graphs)

    /** @brief Optional pre-sorted segment ranges enabling the segmented
     *  conv_tp+scatter fast path.  When non-null, `receiver_edge_segment_start[a]`
     *  and `receiver_edge_segment_end[a]` bracket the range of `edge_target_atom`
     *  values pointing at atom `a`, which lets the TK-fused conv_tp+scatter
     *  kernel emit messages directly into per-receiver accumulators without a
     *  per-edge atomicAdd or DRAM round-trip.  Setting both to nullptr
     *  selects the (slower) atomic fallback path. */
    std::int32_t const *receiver_edge_segment_start = nullptr;
    std::int32_t const *receiver_edge_segment_end = nullptr;
};

/**
 *  @brief Forward-declared CUDA stream type so the public header does not
 *  need `<cuda_runtime.h>`.
 */
#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__) || defined(__CUDA_RUNTIME_API_H__)
#else
using cudaStream_t = void *;
#endif

/**
 *  @brief Persistent inference context.
 *
 *  Owns the per-device CUDA stream, the `mace_scratch_t` workspace, and (in
 *  later phases) the captured-graph cache.  Construct once; reuse across
 *  many inference calls.
 */
class context_t {
  public:
    /**
     *  @param[in] device_id GPU to run on (default 0).
     *  @param[in] max_atoms Upper bound on `batch_view_t::n_atoms` for any call.
     *  @param[in] max_edges Upper bound on `batch_view_t::n_edges`.
     *  @param[in] max_graphs Upper bound on `batch_view_t::n_graphs`.
     */
    context_t(int device_id, smallfield::types::mace_arch_t const &arch, std::uint32_t max_atoms,
              std::uint32_t max_edges, std::uint32_t max_graphs);
    ~context_t();
    context_t(context_t &&) noexcept;
    context_t &operator=(context_t &&) noexcept;
    context_t(context_t const &) = delete;
    context_t &operator=(context_t const &) = delete;

    /**
     *  @brief Single forward pass — energies only.
     *
     *  @param[out] energies_out Device-resident output buffer, [n_graphs] fp32.
     */
    [[nodiscard]] status_t infer_energy(weights_t const &weights, batch_view_t const &batch, float *energies_out);

    /**
     *  @brief Forward + analytic-adjoint backward — energies and per-atom forces.
     *
     *  @param[out] energies_out [n_graphs] fp32, device-resident.
     *  @param[out] forces_out [n_atoms × 3] fp32, device-resident.  May be
     *                        nullptr to skip the backward pass.
     */
    [[nodiscard]] status_t infer_energy_and_forces(weights_t const &weights, batch_view_t const &batch,
                                                   float *energies_out, float *forces_out);

    /**
     *  @brief Batched LBFGS relaxation: N steps in place on the position
     *  buffer.  Today the LBFGS step itself runs host-side (one GD step per
     *  iteration plus a re-run of the forward and adjoint); Phase K of the
     *  optimisation roadmap replaces with a fully GPU-resident captured
     *  graph that stays inside the device for the entire relaxation.
     *
     *  @param[in,out] positions_xyz_inout    [n_atoms × 3] fp32 device-resident.
     *  @param[out]    energies_out           [n_graphs]    fp32 device-resident.
     *  @param[out]    step_count_per_mol_out [n_graphs] uint32 device-resident,
     *                                        OPTIONAL (may be nullptr).  If
     *                                        provided, after the relaxation each
     *                                        slot holds the 1-based step index at
     *                                        which that conformer first converged
     *                                        (`fmax ≤ config.fmax_eV_per_A`), or 0
     *                                        if the conformer didn't converge in
     *                                        `config.num_steps`.  Lets callers
     *                                        replace "p95 batch" projection models
     *                                        with real per-mol step distributions.
     *  @param[out]    final_fmax_per_mol_out [n_graphs] fp32 device-resident,
     *                                        OPTIONAL (may be nullptr).  If
     *                                        provided, holds the per-conformer
     *                                        max-abs force at the last step that
     *                                        ran for that conformer (in eV/Å).
     */
    [[nodiscard]] status_t refine_batch(weights_t const &weights, batch_view_t batch, relax_config_t const &config,
                                        float *positions_xyz_inout, float *energies_out,
                                        std::uint32_t *step_count_per_mol_out = nullptr,
                                        float *final_fmax_per_mol_out = nullptr);

    /** @brief CUDA stream the context schedules its work on. */
    cudaStream_t stream() const noexcept;

  private:
    struct impl_t;
    std::unique_ptr<impl_t> p_;
};

} // namespace smallfield::mace
