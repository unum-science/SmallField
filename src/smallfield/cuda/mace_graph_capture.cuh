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

#include <algorithm>
#include <cstdio>
#include <vector>

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
    cudaError_t const warmup_err = cudaGetLastError();
    if (warmup_err != cudaSuccess) {
        std::fprintf(stderr, "mace_capture_forward: warmup failed (%s)\n",
                     cudaGetErrorString(warmup_err));
        return;
    }

    // 2. Begin capture and re-issue the same kernel sequence.
    cudaError_t const begin_err = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
    if (begin_err != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamBeginCapture failed: %s\n",
                     cudaGetErrorString(begin_err));
        return;
    }
    mace_inference_reset_per_atom_energy_accumulator(scratch, batch.n_atoms, capture_stream);
    mace_inference_forward(arch, W, batch, scratch, capture_stream);
    cudaGraph_t graph = nullptr;
    cudaError_t const end_err = cudaStreamEndCapture(capture_stream, &graph);
    if (end_err != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamEndCapture failed: %s\n",
                     cudaGetErrorString(end_err));
        return;
    }

    // 3. Instantiate the executable graph.  Use the with-flags form so we get
    //    the failing-node identifier and a log buffer back when something
    //    inside the captured region cannot be replayed.
    cudaGraphExec_t graph_exec = nullptr;
    char log_buffer[1024] = {0};
    cudaGraphNode_t error_node = nullptr;
    cudaError_t const inst_err = cudaGraphInstantiate(
        &graph_exec, graph, &error_node, log_buffer, sizeof(log_buffer));
    if (inst_err != cudaSuccess) {
        std::fprintf(stderr,
                     "mace_capture_forward: cudaGraphInstantiate failed (%s) "
                     "— error_node=%p log=\"%s\"\n",
                     cudaGetErrorString(inst_err),
                     static_cast<void *>(error_node), log_buffer);
        return;
    }

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

/**
 *  @brief Capture forward + force-adjoint into a single `cudaGraph_t`.
 *
 *  Identical capture protocol to `mace_capture_forward` but extends the
 *  captured region with a `mace_inference_run_force_adjoint` call.  Replay
 *  produces both energy (in `scratch.energy`) and forces (in `scratch.forces`)
 *  in one `cudaGraphLaunch`.
 *
 *  The capture warmup runs both forward AND adjoint once eagerly so any
 *  first-call lazy allocation in either pass (e.g. the TK W3_eff precompute)
 *  is amortised before capture begins.
 *
 *  The earlier separate `apply_cutoff` backward step (with its transient
 *  per-layer `grad_cutoff_envelope` scratch) was removed once the forward
 *  was changed to fold the cutoff envelope directly into `radial_basis`
 *  (see `mace_inference.cuh` step 3j).  No `cudaMallocAsync` /
 *  `cudaFreeAsync` calls remain on the captured chain.
 */
inline void mace_capture_forward_with_force_adjoint(
    types::mace_arch_t const &arch,
    types::mace_weights_device_t const &W,
    types::mace_batch_t const &batch,
    mace_scratch_t &scratch,
    cudaStream_t capture_stream,
    mace_captured_graph_t &out
) {
    // 1. Warmup: run forward + adjoint once eagerly so all first-call lazy
    //    allocations land before capture begins.
    mace_inference_reset_per_atom_energy_accumulator(scratch, batch.n_atoms, capture_stream);
    mace_inference_forward(arch, W, batch, scratch, capture_stream);
    mace_inference_run_force_adjoint(arch, W, batch, scratch, capture_stream);
    cudaStreamSynchronize(capture_stream);
    cudaError_t const warmup_err = cudaGetLastError();
    if (warmup_err != cudaSuccess) {
        std::fprintf(stderr, "mace_capture_forward_with_force_adjoint: warmup failed (%s)\n",
                     cudaGetErrorString(warmup_err));
        return;
    }

    // 2. Begin capture and re-issue forward + adjoint.
    cudaError_t const begin_err = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
    if (begin_err != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamBeginCapture failed: %s\n",
                     cudaGetErrorString(begin_err));
        return;
    }
    mace_inference_reset_per_atom_energy_accumulator(scratch, batch.n_atoms, capture_stream);
    mace_inference_forward(arch, W, batch, scratch, capture_stream);
    mace_inference_run_force_adjoint(arch, W, batch, scratch, capture_stream);
    cudaGraph_t graph = nullptr;
    cudaError_t const end_err = cudaStreamEndCapture(capture_stream, &graph);
    if (end_err != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamEndCapture failed: %s\n",
                     cudaGetErrorString(end_err));
        return;
    }

    // 3. Instantiate.
    cudaGraphExec_t graph_exec = nullptr;
    char log_buffer[1024] = {0};
    cudaGraphNode_t error_node = nullptr;
    cudaError_t const inst_err = cudaGraphInstantiate(
        &graph_exec, graph, &error_node, log_buffer, sizeof(log_buffer));
    if (inst_err != cudaSuccess) {
        std::fprintf(stderr,
                     "mace_capture_forward_with_force_adjoint: cudaGraphInstantiate failed (%s) "
                     "— error_node=%p log=\"%s\"\n",
                     cudaGetErrorString(inst_err),
                     static_cast<void *>(error_node), log_buffer);
        return;
    }

    // 4. Populate the out struct.
    if (out.graph_exec) cudaGraphExecDestroy(out.graph_exec);
    if (out.graph) cudaGraphDestroy(out.graph);
    out.graph = graph;
    out.graph_exec = graph_exec;
    out.arch = arch;
    out.n_atoms = batch.n_atoms;
    out.n_edges = batch.n_edges;
    out.n_graphs = batch.n_graphs;
}

/**
 *  @brief Per-atom-bin graph cache.
 *
 *  An LBFGS pre-pass classifies each conformation into a fixed-shape bin
 *  by `(n_atoms, n_edges)` — a 30-atom drug-like structure with 480 edges
 *  shares a captured graph with every other 30/480 conformation in the bin.
 *
 *  The cache lives in host memory; lookup is O(log n_bins) via a sorted
 *  vector of `(key, captured_graph)` pairs.  Eviction is opt-in (LRU is
 *  overkill for the LBFGS workload — each bin is reused for the entire
 *  relaxation step's worth of replays).
 */
struct mace_captured_graph_cache_t {
    struct bin_key_t {
        std::uint32_t n_atoms = 0;
        std::uint32_t n_edges = 0;
        std::uint32_t n_graphs = 0;
        bool operator<(bin_key_t const &other) const noexcept {
            if (n_atoms != other.n_atoms) return n_atoms < other.n_atoms;
            if (n_edges != other.n_edges) return n_edges < other.n_edges;
            return n_graphs < other.n_graphs;
        }
        bool operator==(bin_key_t const &other) const noexcept {
            return n_atoms == other.n_atoms && n_edges == other.n_edges && n_graphs == other.n_graphs;
        }
    };

    /**
     *  @brief Look up an existing captured graph by bin key, or capture a
     *         new one if missing.
     *
     *  When `with_force_adjoint` is true the captured graph runs both
     *  forward and the force-adjoint pass.  The caller must already have
     *  invoked `S.allocate(...)` with at least the bin's max shapes.
     *
     *  Pointer ownership: returned graph is owned by the cache.  Caller
     *  invokes `replay()` on it; the cache destroys it on destruction.
     */
    mace_captured_graph_t const *get_or_capture(
        types::mace_arch_t const &arch,
        types::mace_weights_device_t const &W,
        types::mace_batch_t const &batch,
        mace_scratch_t &S,
        cudaStream_t capture_stream,
        bool with_force_adjoint
    ) {
        bin_key_t const key{batch.n_atoms, batch.n_edges, batch.n_graphs};
        auto it = std::lower_bound(entries_.begin(), entries_.end(), key,
                                    [](entry_t const &e, bin_key_t const &k) { return e.key < k; });
        if (it != entries_.end() && it->key == key) {
            return &it->graph;
        }
        // Miss: capture a new graph and insert.
        entry_t fresh{};
        fresh.key = key;
        if (with_force_adjoint) {
            mace_capture_forward_with_force_adjoint(arch, W, batch, S, capture_stream, fresh.graph);
        } else {
            mace_capture_forward(arch, W, batch, S, capture_stream, fresh.graph);
        }
        if (fresh.graph.graph_exec == nullptr) {
            // Capture failed — don't poison the cache.
            return nullptr;
        }
        auto inserted = entries_.insert(it, std::move(fresh));
        return &inserted->graph;
    }

    /** Number of cached bins currently held. */
    std::size_t size() const noexcept { return entries_.size(); }

    /** Clear all entries — destroys their underlying graphs. */
    void clear() { entries_.clear(); }

private:
    struct entry_t {
        bin_key_t key{};
        mace_captured_graph_t graph;
    };
    std::vector<entry_t> entries_;
};

} // namespace smallfield::cuda

#endif // __CUDACC__
