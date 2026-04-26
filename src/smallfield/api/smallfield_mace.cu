/**
 *  @file smallfield_mace.cu
 *  @brief Implementation of the public smallfield::mace API surface.
 *
 *  Wraps the existing kernel-driver entry points from
 *  `src/smallfield/cuda/mace_inference.cuh` behind a stable C++ API.
 *  The public header (`smallfield/api/smallfield_mace.hpp`) does NOT pull
 *  in any CUDA headers — that's why every kernel-side call lives here.
 */

#include "smallfield/api/smallfield_mace.hpp"

#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include "smallfield/parameters/mace_weights.hpp"
#include "smallfield/types/mace_types.hpp"
#include "smallfield/cuda/mace_inference.cuh"

namespace smallfield::mace {

// ---------------------------------------------------------------------------
//  weights_t — RAII wrapper around parameters::mace_weights_t
// ---------------------------------------------------------------------------

struct weights_t::impl_t {
    smallfield::parameters::mace_weights_t inner;
    bool loaded = false;
};

weights_t::weights_t() : p_(std::make_unique<impl_t>()) {}
weights_t::~weights_t() = default;
weights_t::weights_t(weights_t &&) noexcept = default;
weights_t &weights_t::operator=(weights_t &&) noexcept = default;

weights_t weights_t::load(std::string const &path) {
    weights_t out;
    out.p_->inner.load(path);
    out.p_->loaded = true;
    return out;
}

smallfield::types::mace_arch_t const &weights_t::arch() const noexcept {
    return p_->inner.arch();
}

bool weights_t::is_loaded() const noexcept {
    return p_->loaded;
}

void const *weights_t::device_weights() const noexcept {
    if (!p_->loaded) return nullptr;
    return static_cast<void const *>(&p_->inner.device());
}

// Resolve a public weights_t handle to the device-resident weights pointer
// the kernel driver expects.  The arch is fetched via the public arch()
// accessor; the device_weights() accessor returns a pointer to the
// mace_weights_device_t the kernel driver consumes.
namespace detail_ {
inline smallfield::types::mace_weights_device_t const &
device_weights_ref(weights_t const &W) {
    return *static_cast<smallfield::types::mace_weights_device_t const *>(W.device_weights());
}
}  // namespace detail_

// ---------------------------------------------------------------------------
//  context_t — owns scratch + stream
// ---------------------------------------------------------------------------

struct context_t::impl_t {
    int device_id = 0;
    cudaStream_t stream = nullptr;
    smallfield::cuda::mace_scratch_t scratch;
    std::uint32_t max_atoms = 0;
    std::uint32_t max_edges = 0;
    std::uint32_t max_graphs = 0;
    smallfield::types::mace_arch_t arch{};

    impl_t(int dev,
           smallfield::types::mace_arch_t const &a,
           std::uint32_t na, std::uint32_t ne, std::uint32_t ng)
        : device_id(dev), max_atoms(na), max_edges(ne), max_graphs(ng), arch(a) {
        cudaSetDevice(device_id);
        cudaStreamCreate(&stream);
        scratch.allocate(arch, max_atoms, max_edges, max_graphs);
    }

    ~impl_t() {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        scratch.free();
    }
};

context_t::context_t(int device_id, smallfield::types::mace_arch_t const &arch,
                     std::uint32_t max_atoms, std::uint32_t max_edges, std::uint32_t max_graphs)
    : p_(std::make_unique<impl_t>(device_id, arch, max_atoms, max_edges, max_graphs)) {}
context_t::~context_t() = default;
context_t::context_t(context_t &&) noexcept = default;
context_t &context_t::operator=(context_t &&) noexcept = default;

cudaStream_t context_t::stream() const noexcept { return p_->stream; }

// ---------------------------------------------------------------------------
//  Helpers (TU-local)
// ---------------------------------------------------------------------------

namespace {

inline smallfield::types::mace_batch_t to_internal_(batch_view_t const &B) {
    smallfield::types::mace_batch_t out{};
    out.n_atoms = B.n_atoms;
    out.n_edges = B.n_edges;
    out.n_graphs = B.n_graphs;
    out.positions = const_cast<float *>(B.positions_xyz);
    out.atom_element = const_cast<std::int32_t *>(B.atom_element);
    out.edge_source_atom = const_cast<std::int32_t *>(B.edge_source_atom);
    out.edge_target_atom = const_cast<std::int32_t *>(B.edge_target_atom);
    out.graph_of_atom = const_cast<std::int32_t *>(B.graph_of_atom);
    out.receiver_edge_segment_start = const_cast<std::int32_t *>(B.receiver_edge_segment_start);
    out.receiver_edge_segment_end = const_cast<std::int32_t *>(B.receiver_edge_segment_end);
    return out;
}

inline bool batch_fits_(std::uint32_t max_atoms, std::uint32_t max_edges, std::uint32_t max_graphs,
                         batch_view_t const &B) {
    return B.n_atoms <= max_atoms && B.n_edges <= max_edges && B.n_graphs <= max_graphs;
}

}  // namespace

// ---------------------------------------------------------------------------
//  Inference entry points
// ---------------------------------------------------------------------------

status_t context_t::infer_energy(weights_t const &W, batch_view_t const &B, float *energies_out) {
    if (!W.is_loaded()) return status_t::weights_load_failed;
    if (!batch_fits_(p_->max_atoms, p_->max_edges, p_->max_graphs, B)) return status_t::invalid_input;
    if (energies_out == nullptr) return status_t::invalid_input;

    cudaSetDevice(p_->device_id);
    // Reach into the weights_t::impl_t (opaque from the header but defined
    // in this TU) to get both the arch and the device weights handle.
    auto const &W_arch = W.arch();
    auto const &W_device = detail_::device_weights_ref(W);
    auto batch_internal = to_internal_(B);

    smallfield::cuda::mace_inference_reset_per_atom_energy_accumulator(
        p_->scratch, batch_internal.n_atoms, p_->stream);
    smallfield::cuda::mace_inference_forward(
        W_arch, W_device, batch_internal, p_->scratch, p_->stream);

    cudaMemcpyAsync(energies_out, p_->scratch.energy,
                    batch_internal.n_graphs * sizeof(float),
                    cudaMemcpyDeviceToDevice, p_->stream);

    cudaError_t const err = cudaGetLastError();
    if (err != cudaSuccess) return status_t::cuda_error;
    return status_t::success;
}

status_t context_t::infer_energy_and_forces(weights_t const &W,
                                              batch_view_t const &B,
                                              float *energies_out,
                                              float *forces_out) {
    if (!W.is_loaded()) return status_t::weights_load_failed;
    if (!batch_fits_(p_->max_atoms, p_->max_edges, p_->max_graphs, B)) return status_t::invalid_input;
    if (energies_out == nullptr) return status_t::invalid_input;

    cudaSetDevice(p_->device_id);
    auto const &W_arch = W.arch();
    auto const &W_device = detail_::device_weights_ref(W);
    auto batch_internal = to_internal_(B);

    smallfield::cuda::mace_inference_reset_per_atom_energy_accumulator(
        p_->scratch, batch_internal.n_atoms, p_->stream);
    smallfield::cuda::mace_inference_forward(
        W_arch, W_device, batch_internal, p_->scratch, p_->stream);

    cudaMemcpyAsync(energies_out, p_->scratch.energy,
                    batch_internal.n_graphs * sizeof(float),
                    cudaMemcpyDeviceToDevice, p_->stream);

    if (forces_out != nullptr) {
        smallfield::cuda::mace_inference_run_force_adjoint(
            W_arch, W_device, batch_internal, p_->scratch, p_->stream);
        cudaMemcpyAsync(forces_out, p_->scratch.forces,
                        batch_internal.n_atoms * 3u * sizeof(float),
                        cudaMemcpyDeviceToDevice, p_->stream);
    }

    cudaError_t const err = cudaGetLastError();
    if (err != cudaSuccess) return status_t::cuda_error;
    return status_t::success;
}

// ---------------------------------------------------------------------------
//  Refinement (placeholder GD step — Phase K replaces with on-device LBFGS).
// ---------------------------------------------------------------------------

namespace {

/** In-place GD step: positions[i] -= step_size * (-forces[i]) === += step * forces.
 *  Forces in MACE-OFF inference are -∂E/∂r, so positions move in the direction
 *  of decreasing energy when we add `step_size * forces`. */
__global__ void mace_api_gd_step_kernel_(std::uint32_t n_components,
                                          float step_size,
                                          float const *forces,
                                          float *positions) {
    std::uint32_t element_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_index >= n_components) return;
    positions[element_index] += step_size * forces[element_index];
}

inline void launch_gd_step_(std::uint32_t n_atoms, float step_size,
                             float const *forces, float *positions, cudaStream_t stream) {
    if (n_atoms == 0u) return;
    std::uint32_t const n_components = n_atoms * 3u;
    constexpr std::uint32_t block = 256u;
    std::uint32_t const grid = (n_components + block - 1u) / block;
    mace_api_gd_step_kernel_<<<grid, block, 0, stream>>>(n_components, step_size, forces, positions);
}

}  // namespace

status_t context_t::refine_batch(weights_t const &W,
                                  batch_view_t batch,
                                  relax_config_t const &config,
                                  float *positions_xyz_inout,
                                  float *energies_out) {
    if (!W.is_loaded()) return status_t::weights_load_failed;
    if (!batch_fits_(p_->max_atoms, p_->max_edges, p_->max_graphs, batch)) return status_t::invalid_input;
    if (positions_xyz_inout == nullptr) return status_t::invalid_input;
    if (energies_out == nullptr) return status_t::invalid_input;
    if (config.num_steps == 0u) return status_t::success;

    cudaSetDevice(p_->device_id);
    auto const &W_arch = W.arch();
    auto const &W_device = detail_::device_weights_ref(W);

    // Rewire the view onto the user's in/out buffer for the duration of the loop.
    batch.positions_xyz = positions_xyz_inout;

    for (std::uint32_t step_index = 0u; step_index < config.num_steps; ++step_index) {
        auto batch_internal = to_internal_(batch);

        smallfield::cuda::mace_inference_reset_per_atom_energy_accumulator(
            p_->scratch, batch_internal.n_atoms, p_->stream);
        smallfield::cuda::mace_inference_forward(
            W_arch, W_device, batch_internal, p_->scratch, p_->stream);
        smallfield::cuda::mace_inference_run_force_adjoint(
            W_arch, W_device, batch_internal, p_->scratch, p_->stream);

        launch_gd_step_(batch_internal.n_atoms, config.step_size,
                        p_->scratch.forces, positions_xyz_inout, p_->stream);
    }

    // Final forward to populate energies_out at the converged geometry.
    auto batch_internal_final = to_internal_(batch);
    smallfield::cuda::mace_inference_reset_per_atom_energy_accumulator(
        p_->scratch, batch_internal_final.n_atoms, p_->stream);
    smallfield::cuda::mace_inference_forward(
        W_arch, W_device, batch_internal_final, p_->scratch, p_->stream);
    cudaMemcpyAsync(energies_out, p_->scratch.energy,
                    batch_internal_final.n_graphs * sizeof(float),
                    cudaMemcpyDeviceToDevice, p_->stream);

    (void)config.fmax_eV_per_A;  // Phase K will use this for convergence detection.

    cudaError_t const err = cudaGetLastError();
    if (err != cudaSuccess) return status_t::cuda_error;
    return status_t::success;
}

}  // namespace smallfield::mace
