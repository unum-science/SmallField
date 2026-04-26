/**
 *  @file mace_graph_capture.cuh
 *  @brief Track E1 — capture an entire MACE forward pass into a CUDA Graph.
 *
 *  Every kernel in `mace_inference_forward` is launched on a single stream
 *  with shape parameters that are constant for one atom-limited-bin (the
 *  bin size, channels, irreps, atom count, edge count).  That is exactly
 *  the regime where `cudaStreamCaptureModeGlobal` + `cudaGraphInstantiate`
 *  shines: capture once at first call, replay on every subsequent step
 *  with new positions written into the same input buffer.
 *
 *  The cueq segmented_polynomial path (which we are replacing) currently
 *  breaks graph capture — we measured this in the benchmark report.  Our
 *  hand-rolled kernels do not have that problem because they only call
 *  standard CUDA primitives.
 *
 *  Hot-loop replay cost on Hopper: ~5-10 µs per replay, vs ~7-10 ms of
 *  per-kernel-launch overhead per eager forward.  At 20 LBFGS steps × 28 B
 *  configs / 8 cards, the saved overhead alone is days.
 *
 *  ----------------------------------------------------------------------
 *  Status: forward-only capture is live.  Force-adjoint capture is a
 *  follow-up once Track D adjoints land — the API will gain a second
 *  function `mace_capture_forward_with_force_adjoint` that wraps both
 *  forward and adjoint in one graph.
 *  ----------------------------------------------------------------------
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cuda_runtime.h>

#include "smallfield/types/mace_types.hpp"
#include "smallfield/cuda/mace_inference.cuh"

namespace smallfield::cuda {

/**
 *  @brief Captured MACE forward graph for one atom-limited-bin.
 *
 *  Holds device pointers fixed at capture time; positions are written into
 *  the input pointer by the caller and the graph is replayed.  Output
 *  energy appears in `scratch.energy` when the stream syncs.
 *
 *  The captured graph references device pointer ADDRESSES, not pointer
 *  values — so once a graph is captured, the caller MUST keep writing
 *  position updates into the same `batch.positions` allocation.  Re-pointing
 *  `batch.positions` to a different buffer breaks subsequent replays.
 *
 *  The captured node_energy reset (the `cudaMemsetAsync` from
 *  `mace_inference_reset_per_atom_energy_accumulator`) is part of the
 *  captured graph, so callers do NOT need to call it separately before
 *  each replay.
 */
struct mace_captured_graph_t {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    types::mace_arch_t arch{};
    std::uint32_t n_atoms = 0;
    std::uint32_t n_edges = 0;
    std::uint32_t n_graphs = 0;

    /** Replay the captured graph on `stream`.  Positions must already be in
     *  the input buffer; energy will be in `scratch.energy` when the stream
     *  syncs. */
    void replay(cudaStream_t stream) const {
        if (graph_exec) cudaGraphLaunch(graph_exec, stream);
    }

    ~mace_captured_graph_t() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
    }

    mace_captured_graph_t() = default;
    mace_captured_graph_t(mace_captured_graph_t const &) = delete;
    mace_captured_graph_t &operator=(mace_captured_graph_t const &) = delete;
    mace_captured_graph_t(mace_captured_graph_t &&other) noexcept
        : graph(other.graph), graph_exec(other.graph_exec), arch(other.arch),
          n_atoms(other.n_atoms), n_edges(other.n_edges), n_graphs(other.n_graphs) {
        other.graph = nullptr;
        other.graph_exec = nullptr;
    }
    mace_captured_graph_t &operator=(mace_captured_graph_t &&other) noexcept {
        if (this != &other) {
            if (graph_exec) cudaGraphExecDestroy(graph_exec);
            if (graph) cudaGraphDestroy(graph);
            graph = other.graph;
            graph_exec = other.graph_exec;
            arch = other.arch;
            n_atoms = other.n_atoms;
            n_edges = other.n_edges;
            n_graphs = other.n_graphs;
            other.graph = nullptr;
            other.graph_exec = nullptr;
        }
        return *this;
    }
};

/**
 *  @brief Capture a full forward pass into a `cudaGraph_t`.
 *
 *  Wraps the per-atom-energy reset + `mace_inference_forward` in a
 *  `cudaStreamBeginCapture` / `cudaStreamEndCapture` block and instantiates
 *  the resulting graph.  The captured graph is keyed by the atom-bin shape
 *  (n_atoms, n_edges, n_graphs) — the caller is responsible for caching
 *  one graph per shape and reusing it across LBFGS steps.
 *
 *  The driver is run once eagerly before capture begins to trigger any
 *  first-call lazy allocation (notably the TK W3_eff precompute under
 *  `SMALLFIELD_USE_THUNDERKITTENS`, which would otherwise call `cudaMalloc`
 *  inside the captured region and abort capture).
 */
inline void mace_capture_forward(
    types::mace_arch_t const &arch,
    types::mace_weights_device_t const &W,
    types::mace_batch_t const &batch,
    mace_scratch_t &scratch,
    cudaStream_t capture_stream,
    mace_captured_graph_t &out
) {
    // 1. Warm up the driver path once so any first-call lazy allocation
    //    (e.g. `tk_precomputed_w3_effective_layer` under
    //    SMALLFIELD_USE_THUNDERKITTENS) happens BEFORE we start capture.
    //    `cudaMalloc` is not legal inside a stream-capture region.
    mace_inference_reset_per_atom_energy_accumulator(scratch, batch.n_atoms, capture_stream);
    mace_inference_forward(arch, W, batch, scratch, capture_stream);
    cudaStreamSynchronize(capture_stream);

    // 2. Begin capture and re-issue the same kernel sequence.
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
    mace_inference_reset_per_atom_energy_accumulator(scratch, batch.n_atoms, capture_stream);
    mace_inference_forward(arch, W, batch, scratch, capture_stream);
    cudaGraph_t graph = nullptr;
    cudaStreamEndCapture(capture_stream, &graph);

    // 3. Instantiate the executable graph.
    cudaGraphExec_t graph_exec = nullptr;
    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

    // 4. Populate the out struct (move-friendly, releases any prior graph).
    if (out.graph_exec) cudaGraphExecDestroy(out.graph_exec);
    if (out.graph) cudaGraphDestroy(out.graph);
    out.graph = graph;
    out.graph_exec = graph_exec;
    out.arch = arch;
    out.n_atoms = batch.n_atoms;
    out.n_edges = batch.n_edges;
    out.n_graphs = batch.n_graphs;
}

} // namespace smallfield::cuda

#endif // __CUDACC__
