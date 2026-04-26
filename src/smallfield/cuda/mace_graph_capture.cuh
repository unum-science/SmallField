/**
 *  @file mace_graph_capture.cuh
 *  @brief Phase 4 — capture an entire MACE forward + force-adjoint pass into a CUDA Graph.
 *
 *  Once Phases 1-3 land, every kernel in the inference path is launched on
 *  a single stream with shape parameters that are constant for one
 *  atom-limited-bin (the bin size, channels, irreps, atom count, edge
 *  count).  That is exactly the regime where `cudaStreamCaptureModeGlobal`
 *  + `cudaGraphInstantiate` shines: capture once at first call, replay
 *  on every subsequent LBFGS step with new positions.
 *
 *  The tricky part is that the cueq segmented_polynomial path (which we
 *  are replacing) currently breaks graph capture — we measured this in
 *  the benchmark report.  Our hand-rolled kernels do not have that
 *  problem because they only call standard CUDA primitives.
 *
 *  Hot-loop replay cost on Hopper: ~5-10 µs per replay, vs ~50-100 µs of
 *  Python + ASE + cueq launch overhead per forward today.  At 20 LBFGS
 *  steps × 28 B configs / 8 cards, the saved overhead alone is days.
 *
 *  ----------------------------------------------------------------------
 *  Phase-4 status: signature-only.  The capture flow is a 50-line wrapper
 *  around `mace_inference_forward` + `mace_inference_run_force_adjoint`
 *  — but it is gated on those two being bit-exact.  Wired up after
 *  Phase 3 lands.
 *  ----------------------------------------------------------------------
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cuda_runtime.h>

#include "smallfield/types/mace_types.hpp"

namespace smallfield::cuda {

/**
 *  @brief Captured MACE forward + force-adjoint graph for one atom-limited-bin.
 *
 *  Holds device pointers fixed at capture time; positions are written into
 *  the input pointer by the caller and the graph is replayed.  Output
 *  energy and forces appear in the output pointers.
 */
struct mace_captured_graph_t {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    types::mace_arch_t arch{};
    std::uint32_t n_atoms = 0;
    std::uint32_t n_edges = 0;
    std::uint32_t n_graphs = 0;

    /** Replay the captured graph on `stream`.  Positions must already be in the
     *  input buffer; energy and forces will be in the output buffers when the
     *  stream syncs. */
    void replay(cudaStream_t stream) const {
        if (graph_exec) cudaGraphLaunch(graph_exec, stream);
    }

    ~mace_captured_graph_t() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
    }
};

/**
 *  @brief Capture a full forward + force-adjoint pass into a `cudaGraph_t`.
 *
 *  Calls `mace_inference_forward` + `mace_inference_run_force_adjoint`
 *  inside a `cudaStreamBeginCapture` / `cudaStreamEndCapture` block.
 *  Returns a ready-to-replay graph.
 *
 *  @todo Phase-4 stub.
 */
inline void mace_capture_forward_with_force_adjoint(
    /* args TBD */
    cudaStream_t /*stream*/,
    mace_captured_graph_t &/*out*/
) {
    // TODO(phase4): cudaStreamBeginCapture, run forward + force-adjoint, cudaStreamEndCapture,
    // cudaGraphInstantiate.  ~30 lines.
}

} // namespace smallfield::cuda

#endif // __CUDACC__
