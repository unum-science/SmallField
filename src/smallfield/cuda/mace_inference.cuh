/**
 *  @file mace_inference.cuh
 *  @brief Top-level MACE-OFF inference driver — Phase 1 fp32 reference forward.
 *
 *  Orchestrates the per-edge preamble (geometry, Bessel basis, spherical
 *  harmonics, radial MLP), the two interaction-block + product-block + readout
 *  cycles, and the final per-graph energy reduction.
 *
 *  Internal layout for irrep tensors throughout: `[n_atoms, n_channels, 16]`
 *  with the inner 16 dims = [L=0 (1), L=1 (3), L=2 (5), L=3 (7)] concatenated.
 *  This is the post-`reshape_irreps` layout used by the symmetric contraction
 *  and the per-irrep linears.
 *
 *  Force computation (analytic adjoint) lives in `mace_force_adjoints.cuh`.
 *  Phase 1 is energy-only.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#include "smallfield/types/mace_types.hpp"
#include "smallfield/cuda/mace_radial_kernel.cuh"
#include "smallfield/cuda/mace_sph_harm_kernel.cuh"
#include "smallfield/cuda/mace_node_embed_kernel.cuh"
#include "smallfield/cuda/mace_linear_kernel.cuh"
#include "smallfield/cuda/mace_scatter_kernel.cuh"
#include "smallfield/cuda/mace_tensor_product_kernel.cuh"
#include "smallfield/cuda/mace_skip_tp_kernel.cuh"
#include "smallfield/cuda/mace_symmetric_contraction_kernel.cuh"
#include "smallfield/cuda/mace_force_adjoints.cuh"

// ThunderKittens forward path — every TK kernel is gated on
// `SMALLFIELD_USE_THUNDERKITTENS`.  When the macro is undefined the headers
// below are still included (they self-gate) but no TK code is emitted; the
// driver below uses the fp32 reference launchers exclusively.  When the
// macro is defined, the driver branches to the TK launchers at each call
// site.  Keeping the fp32 path in tree as the regression target is
// deliberate.
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
#  include "smallfield/cuda/mace_bf16.cuh"
#  include "smallfield/cuda/mace_tk_radial_mlp_kernel.cuh"
#  include "smallfield/cuda/mace_tk_per_irrep_linear_kernel.cuh"
#  include "smallfield/cuda/mace_tk_conv_tp_scatter_fused_kernel.cuh"
#  include "smallfield/cuda/mace_tk_element_conditioned_residual_kernel.cuh"
#  include "smallfield/cuda/mace_tk_symmetric_contraction_kernel.cuh"
#  include "smallfield/cuda/mace_tk_readout_kernel.cuh"
#endif

namespace smallfield::cuda {

namespace mace_detail_ {

/** Project the L=0 component of a [n_atoms, c, 16] tensor to a [n_atoms, c] tensor. */
__global__ void mace_extract_l0_irrep_block_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t irrep_dim,
    float const *src,        // [n_atoms, c, irrep_dim]
    float *dst                // [n_atoms, c]
) {
    std::uint32_t atom_index = blockIdx.x;
    std::uint32_t channel_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (atom_index >= n_atoms || channel_index >= n_channels) return;
    dst[static_cast<std::uint64_t>(atom_index) * n_channels + channel_index] =
        src[(static_cast<std::uint64_t>(atom_index) * n_channels + channel_index) * irrep_dim + 0];
}

/** Place a [n_atoms, c] tensor into the L=0 slot of a [n_atoms, c, 16] tensor (zero rest). */
__global__ void mace_inject_l0_irrep_block_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t irrep_dim,
    float const *src,        // [n_atoms, c]
    float *dst                // [n_atoms, c, irrep_dim]
) {
    std::uint32_t atom_index = blockIdx.x;
    std::uint32_t channel_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (atom_index >= n_atoms || channel_index >= n_channels) return;
    std::uint64_t const output_offset = (static_cast<std::uint64_t>(atom_index) * n_channels + channel_index) * irrep_dim;
    dst[output_offset + 0] = src[static_cast<std::uint64_t>(atom_index) * n_channels + channel_index];
    for (std::uint32_t irrep_component_index = 1; irrep_component_index < irrep_dim; ++irrep_component_index) dst[output_offset + irrep_component_index] = 0.0f;
}

/** Per-graph reduce: sum atomic baseline + scale_shift(node_energy) into energy[g]. */
__global__ void mace_per_graph_energy_reduce_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    float scale,
    float shift,
    std::int32_t const *atom_element,
    std::int32_t const *graph_of_atom,
    float const *atomic_energies,        // [n_elements]
    float const *node_energy_scalar,     // [n_atoms]  — sum of all readout outputs
    float *energy                         // [n_graphs]
) {
    std::uint32_t atom_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_index >= n_atoms) return;
    int const element_index = atom_element[atom_index];
    int const graph_index = graph_of_atom[atom_index];
    float const e_atom = atomic_energies[element_index] + scale * node_energy_scalar[atom_index] + shift;
    atomicAdd(energy + graph_index, e_atom);
}

/** Linear readout: e[a] = (1/sqrt(c)) * Σ_c W[c] * inv[a, c]. One block per atom; warp-reduces.
 *  The 1/sqrt(c) is the e3nn `o3.Linear` path-weight normalisation
 *  (irreps_in = `c x 0e`, irreps_out = `1 x 0e`, alpha = c). */
__global__ void mace_readout_linear_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    float scale,                            // 1/sqrt(n_channels)
    float const *node_scalar_invariants,    // [n_atoms, n_channels]
    float const *readout_w,                 // [n_channels]
    float *node_energy_out                  // [n_atoms]  — accumulates (atomicAdd)
) {
    std::uint32_t atom_index = blockIdx.x;
    if (atom_index >= n_atoms) return;
    float accumulator = 0.0f;
    for (std::uint32_t channel_index = threadIdx.x; channel_index < n_channels; channel_index += blockDim.x) {
        accumulator += readout_w[channel_index] * node_scalar_invariants[static_cast<std::uint64_t>(atom_index) * n_channels + channel_index];
    }
    for (int shfl_offset = 16; shfl_offset > 0; shfl_offset /= 2) accumulator += __shfl_xor_sync(0xFFFFFFFF, accumulator, shfl_offset);
    if (threadIdx.x == 0) atomicAdd(node_energy_out + atom_index, scale * accumulator);
}

/**
 *  @brief Nonlinear readout MLP (used on the final interaction layer).
 *
 *  Two-layer MLP with SiLU between, applied to per-atom invariants.
 *      h[a, k]  =  silu(Σ_c W1[c, k] * inv[a, c])         ; k = 0..hidden_dim-1
 *      e[a]     =        Σ_k W2[k]    * h[a, k]
 *
 *  Hidden dim is 16 for MACE-OFF (the second-to-last `_Layer.weight` shape
 *  is `[n_channels, hidden_dim=16]`).
 */
__global__ void mace_readout_nonlinear_mlp_kernel(
    std::uint32_t n_atoms,
    std::uint32_t n_channels,
    std::uint32_t hidden_dim,
    float scale1,                           // 1/sqrt(n_channels) — first linear path normalisation
    float scale2,                           // 1/sqrt(hidden_dim) — second linear path normalisation
    float const *node_scalar_invariants,
    float const *w1,                        // [n_channels, hidden_dim]
    float const *w2,                        // [hidden_dim]
    float *node_energy_out                  // [n_atoms]
) {
    std::uint32_t atom_index = blockIdx.x;
    if (atom_index >= n_atoms) return;
    extern __shared__ float h_shared[];
    if (threadIdx.x < hidden_dim) h_shared[threadIdx.x] = 0.0f;
    __syncthreads();
    if (threadIdx.x < hidden_dim) {
        std::uint32_t hidden_index = threadIdx.x;
        float accumulator = 0.0f;
        for (std::uint32_t channel_index = 0; channel_index < n_channels; ++channel_index) {
            accumulator += node_scalar_invariants[static_cast<std::uint64_t>(atom_index) * n_channels + channel_index]
                 * w1[static_cast<std::uint64_t>(channel_index) * hidden_dim + hidden_index];
        }
        float const pre_act = scale1 * accumulator;
        float const sigmoid_value = 1.0f / (1.0f + __expf(-pre_act));
        // e3nn `nn.Activation` wraps its activation in `normalize2mom`, so the SiLU
        // here is `silu(x) * normalize2mom_constant` — same scaling the radial MLP
        // applies via mace_silu_normalize_to_unit_second_moment_kernel.
        h_shared[hidden_index] = pre_act * sigmoid_value * mace_silu_normalize_to_unit_second_moment_constant_k;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float energy_accumulator = 0.0f;
        for (std::uint32_t hidden_index = 0; hidden_index < hidden_dim; ++hidden_index) energy_accumulator += w2[hidden_index] * h_shared[hidden_index];
        atomicAdd(node_energy_out + atom_index, scale2 * energy_accumulator);
    }
}

} // namespace mace_detail_

/** Single-allocation scratch arena holding all device tensors for one forward pass. */
struct mace_scratch_t {
    // edge-scoped
    float *r_norm = nullptr;
    float *r_unit = nullptr;
    float *radial_basis = nullptr;
    float *cutoff_envelope = nullptr;
    float *sph_harm = nullptr;
    float *radial_mlp_hidden_0 = nullptr;          // [n_edges, hidden_dim]
    float *radial_mlp_hidden_1 = nullptr;
    float *radial_mlp_hidden_2 = nullptr;
    float *tensor_product_weights_per_edge = nullptr;  // [n_edges, n_paths * n_channels]
    float *edge_messages = nullptr;                // [n_edges, n_channels * 16]

    // node-scoped
    float *node_features = nullptr;                // [n_atoms, n_channels, 16] — current
    float *node_features_scratch = nullptr;        // [n_atoms, n_channels, 16] — scratch
    float *node_features_l0_only = nullptr;        // [n_atoms, n_channels]
    float *node_scalar_invariants = nullptr;       // [n_atoms, n_channels]
    float *layer_n_residual_l0 = nullptr;          // [n_atoms, n_channels] — layer-N skip_tp residual,
                                                   // computed at the start of the iteration and added
                                                   // back to node_features_l0_only after product_linear
    float *node_energy = nullptr;                  // [n_atoms]

    // graph-scoped
    float *energy = nullptr;                       // [n_graphs]

    // ---------------------------------------------------------------------
    // Force-adjoint scratch.  Allocated unconditionally so that
    // `mace_inference_run_force_adjoint` always has somewhere to write.
    // The forward path also saves a few per-layer intermediates that the
    // adjoint pass reads back (the bare `node_features` / `node_features_scratch`
    // buffers above are overwritten during the second iteration of the
    // forward loop, so layer-0 contributions need their own home).
    // ---------------------------------------------------------------------
    float *node_features_input_layer[2] = {nullptr, nullptr};        ///< pre-linear_up node features, per layer
    float *node_features_post_residual_layer[2] = {nullptr, nullptr};///< post-skip_tp / pre-symcontraction node features, per layer
    float *node_features_l0_only_layer[2] = {nullptr, nullptr};      ///< post-linear_up scalar features, per layer
    float *node_scalar_invariants_layer[2] = {nullptr, nullptr};     ///< post-symcontraction scalar features, per layer
    float *node_features_l0_only_post_product_layer[2] = {nullptr, nullptr}; ///< post-product-linear (+ residual for layer-N) L=0 features — readout MLP input
    float *tensor_product_weights_pre_cutoff_layer[2] = {nullptr, nullptr};  ///< pre-apply_cutoff tp_weights — needed by apply_cutoff backward + radial MLP backward

    // Backward-pass gradient buffers.
    float *forces = nullptr;                       ///< [n_atoms, 3] final forces output (populated by run_force_adjoint)
    float *grad_positions = nullptr;               ///< [n_atoms, 3] dpositions accumulator before negation
    float *grad_per_graph_energy = nullptr;        ///< [n_graphs] input gradient (set to 1.0 by caller for force computation)
    float *grad_node_energy = nullptr;             ///< [n_atoms] dnode_energy
    float *grad_node_scalar_invariants = nullptr;  ///< [n_atoms, n_channels]
    float *grad_node_features = nullptr;           ///< [n_atoms, n_channels, 16]
    float *grad_node_features_scratch = nullptr;   ///< [n_atoms, n_channels, 16] aux for chain-rule sum
    float *grad_node_features_l0_only = nullptr;   ///< [n_atoms, n_channels]
    float *grad_layer_residual_l0_scratch = nullptr; ///< [n_atoms, n_channels] — staging for layer-N residual contribution to the layer-input gradient
    float *grad_edge_messages = nullptr;           ///< [n_edges, n_channels, 16]
    float *grad_tp_weights = nullptr;              ///< [n_edges, 4 * n_channels]
    float *grad_sph_harm = nullptr;                ///< [n_edges, 16]
    float *grad_r_unit = nullptr;                  ///< [n_edges, 3]
    float *grad_radial_basis = nullptr;            ///< [n_edges, n_radial]

#if defined(SMALLFIELD_USE_THUNDERKITTENS)
    // ThunderKittens scratch — all `_tk_` buffers exist only on the TK forward path.
    // Each is the bf16 view of the corresponding fp32 buffer above plus the
    // `cast back to fp32` round-trip needed to bridge from a TK kernel's bf16
    // output to the next stage's fp32 input.  We could collapse these by
    // having the TK kernels read/write bf16 directly, but for this first
    // integration we keep activations canonically fp32 between stages — the
    // bf16 buffers are local to each TK kernel + cast pair.
    __nv_bfloat16 *tk_tensor_product_weights_per_edge_bf16 = nullptr;  // [n_edges, 4 * n_channels]
    __nv_bfloat16 *tk_node_messages_full_irrep_bf16 = nullptr;         // [n_atoms, n_channels, 16]
    __nv_bfloat16 *tk_node_features_post_linear_bf16 = nullptr;        // [n_atoms, n_channels, 16]
    __nv_bfloat16 *tk_node_features_post_residual_bf16 = nullptr;      // [n_atoms, n_channels, 16]
    __nv_bfloat16 *tk_node_features_l0_only_bf16 = nullptr;            // [n_atoms, n_channels]

    // Order-3 ACE precomputed weights, one device buffer per interaction layer:
    //   `W3_eff[Z, irrep_index_i, irrep_index_j, irrep_index_k, channel_index]
    //         = sum over path of U_3[i,j,k,path] * weights_max[Z, path, c]`
    // Allocated lazily on first forward and freed in `mace_scratch_t::free`.
    float *tk_precomputed_w3_effective_layer[2] = {nullptr, nullptr};
    bool tk_precompute_done = false;
#endif

    std::size_t total_bytes = 0;

    void allocate(types::mace_arch_t const &arch,
                  std::uint32_t max_atoms, std::uint32_t max_edges, std::uint32_t max_graphs,
                  std::uint32_t radial_mlp_hidden_dim = 64u) {
        free();
        auto alloc = [&](float **p, std::size_t n) {
            cudaMalloc(reinterpret_cast<void **>(p), n * sizeof(float));
            total_bytes += n * sizeof(float);
        };
        constexpr std::uint32_t irrep_dim = 16u;
        alloc(&r_norm, max_edges);
        alloc(&r_unit, max_edges * 3u);
        alloc(&radial_basis, max_edges * arch.n_radial);
        alloc(&cutoff_envelope, max_edges);
        alloc(&sph_harm, max_edges * irrep_dim);
        alloc(&radial_mlp_hidden_0, max_edges * radial_mlp_hidden_dim);
        alloc(&radial_mlp_hidden_1, max_edges * radial_mlp_hidden_dim);
        alloc(&radial_mlp_hidden_2, max_edges * radial_mlp_hidden_dim);
        alloc(&tensor_product_weights_per_edge, max_edges * mace_tensor_product_path_count_k * arch.n_channels);
        alloc(&edge_messages, max_edges * arch.n_channels * irrep_dim);
        alloc(&node_features,         max_atoms * arch.n_channels * irrep_dim);
        alloc(&node_features_scratch, max_atoms * arch.n_channels * irrep_dim);
        alloc(&node_features_l0_only, max_atoms * arch.n_channels);
        alloc(&node_scalar_invariants, max_atoms * arch.n_channels);
        alloc(&layer_n_residual_l0, max_atoms * arch.n_channels);
        alloc(&node_energy, max_atoms);
        alloc(&energy, max_graphs);
        // Per-layer forward saves for the backward pass.
        for (std::uint32_t layer_index = 0; layer_index < 2u; ++layer_index) {
            alloc(&node_features_input_layer[layer_index],         max_atoms * arch.n_channels * irrep_dim);
            alloc(&node_features_post_residual_layer[layer_index], max_atoms * arch.n_channels * irrep_dim);
            alloc(&node_features_l0_only_layer[layer_index],       max_atoms * arch.n_channels);
            alloc(&node_scalar_invariants_layer[layer_index],      max_atoms * arch.n_channels);
            alloc(&node_features_l0_only_post_product_layer[layer_index], max_atoms * arch.n_channels);
            alloc(&tensor_product_weights_pre_cutoff_layer[layer_index],
                  max_edges * mace_tensor_product_path_count_k * arch.n_channels);
        }
        // Backward gradient buffers.
        alloc(&forces,                       max_atoms * 3u);
        alloc(&grad_positions,               max_atoms * 3u);
        alloc(&grad_per_graph_energy,        max_graphs);
        alloc(&grad_node_energy,             max_atoms);
        alloc(&grad_node_scalar_invariants,  max_atoms * arch.n_channels);
        alloc(&grad_node_features,           max_atoms * arch.n_channels * irrep_dim);
        alloc(&grad_node_features_scratch,   max_atoms * arch.n_channels * irrep_dim);
        alloc(&grad_node_features_l0_only,   max_atoms * arch.n_channels);
        alloc(&grad_layer_residual_l0_scratch, max_atoms * arch.n_channels);
        alloc(&grad_edge_messages,           max_edges * arch.n_channels * irrep_dim);
        alloc(&grad_tp_weights,              max_edges * mace_tensor_product_path_count_k * arch.n_channels);
        alloc(&grad_sph_harm,                max_edges * irrep_dim);
        alloc(&grad_r_unit,                  max_edges * 3u);
        alloc(&grad_radial_basis,            max_edges * arch.n_radial);
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
        // bf16 scratch for the TK forward path (sized in fp32 elements; the cast
        // is `n_elements * 2 bytes`).  Allocated only when TK is compiled in.
        auto alloc_bf16 = [&](__nv_bfloat16 **p, std::size_t n) {
            cudaMalloc(reinterpret_cast<void **>(p), n * sizeof(__nv_bfloat16));
            total_bytes += n * sizeof(__nv_bfloat16);
        };
        alloc_bf16(&tk_tensor_product_weights_per_edge_bf16,
                   max_edges * mace_tensor_product_path_count_k * arch.n_channels);
        alloc_bf16(&tk_node_messages_full_irrep_bf16,        max_atoms * arch.n_channels * irrep_dim);
        alloc_bf16(&tk_node_features_post_linear_bf16,        max_atoms * arch.n_channels * irrep_dim);
        alloc_bf16(&tk_node_features_post_residual_bf16,      max_atoms * arch.n_channels * irrep_dim);
        alloc_bf16(&tk_node_features_l0_only_bf16,            max_atoms * arch.n_channels);
        // W3_eff buffers are precomputed lazily on first forward (we need the
        // weight pointers to do the reduction); leave them nullptr here.
        tk_precompute_done = false;
#endif
    }

    void free() noexcept {
        for (float **p : {&r_norm, &r_unit, &radial_basis, &cutoff_envelope, &sph_harm,
                          &radial_mlp_hidden_0, &radial_mlp_hidden_1, &radial_mlp_hidden_2,
                          &tensor_product_weights_per_edge, &edge_messages,
                          &node_features, &node_features_scratch,
                          &node_features_l0_only, &node_scalar_invariants,
                          &layer_n_residual_l0,
                          &node_energy, &energy,
                          &forces, &grad_positions, &grad_per_graph_energy,
                          &grad_node_energy, &grad_node_scalar_invariants,
                          &grad_node_features, &grad_node_features_scratch,
                          &grad_node_features_l0_only,
                          &grad_layer_residual_l0_scratch,
                          &grad_edge_messages, &grad_tp_weights,
                          &grad_sph_harm, &grad_r_unit, &grad_radial_basis}) {
            if (*p) cudaFree(*p);
            *p = nullptr;
        }
        for (std::uint32_t layer_index = 0; layer_index < 2u; ++layer_index) {
            for (float **p : {&node_features_input_layer[layer_index],
                              &node_features_post_residual_layer[layer_index],
                              &node_features_l0_only_layer[layer_index],
                              &node_scalar_invariants_layer[layer_index],
                              &node_features_l0_only_post_product_layer[layer_index],
                              &tensor_product_weights_pre_cutoff_layer[layer_index]}) {
                if (*p) cudaFree(*p);
                *p = nullptr;
            }
        }
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
        for (__nv_bfloat16 **p : {&tk_tensor_product_weights_per_edge_bf16,
                                   &tk_node_messages_full_irrep_bf16,
                                   &tk_node_features_post_linear_bf16,
                                   &tk_node_features_post_residual_bf16,
                                   &tk_node_features_l0_only_bf16}) {
            if (*p) cudaFree(*p);
            *p = nullptr;
        }
        for (float *&p : tk_precomputed_w3_effective_layer) {
            if (p) cudaFree(p);
            p = nullptr;
        }
        tk_precompute_done = false;
#endif
        total_bytes = 0;
    }

    ~mace_scratch_t() { free(); }
};

/**
 *  @brief Run one fp32 reference forward pass.
 *
 *  All inputs are device-resident.  Energy is written into
 *  @ref mace_scratch_t::energy on the same stream; the caller is responsible
 *  for `cudaStreamSynchronize` and any D2H copy.
 *
 *  Numerical correctness vs the Python reference is gated on a few
 *  normalisation constants that match e3nn's defaults:
 *    - SiLU `normalize2mom` constant for the radial MLP (see `mace_linear_kernel.cuh`)
 *    - The `1/sqrt(h_in)` factor inside each `_Layer` of FullyConnectedNet
 *    - The `1/sqrt(num_paths_per_irrep)` element-path normalisation in the conv_tp
 *      (1.0 here because each output L has only 1 incoming path)
 *    - `avg_num_neighbors` divisor on the post-scatter message — currently
 *      hardcoded to 1.0 below; the export tool should write the live value
 *      into the arch header in a follow-up.
 */
inline void mace_inference_forward(
    types::mace_arch_t const &arch,
    types::mace_weights_device_t const &W,
    types::mace_batch_t const &batch,
    mace_scratch_t &S,
    cudaStream_t stream
) {
    constexpr std::uint32_t irrep_dim = 16u;
    constexpr std::uint32_t radial_mlp_hidden_dim = 64u;
    constexpr std::uint32_t mlp_out_dim = 4u * /*n_channels*/96u;  // verified at run via arch.n_channels assertion below
    (void)mlp_out_dim;
    float const avg_num_neighbors = arch.avg_num_neighbors;

#if defined(SMALLFIELD_USE_THUNDERKITTENS)
    // Lazy one-shot precompute of `W3_eff[Z, i, j, k, c]` per interaction layer.
    // The reduction (over the path index against `weights_max[Z, path, c]`)
    // depends only on weights and U_3 — both fixed once `mace_weights_t` is
    // constructed.  We do it on first forward rather than at scratch alloc
    // so the driver can stay decoupled from the weight constructor; subsequent
    // forwards just reuse the cached buffers.
    constexpr std::uint32_t symmetric_contraction_path_count_max = 23u;
    if (!S.tk_precompute_done) {
        for (std::uint32_t layer_index = 0; layer_index < arch.num_interactions && layer_index < 2u; ++layer_index) {
            auto const &L = W.layers[layer_index];
            S.tk_precomputed_w3_effective_layer[layer_index] = mace_tk_precompute_w3_effective(
                arch.n_elements, arch.n_channels, symmetric_contraction_path_count_max,
                L.u_matrix_3, L.weights_max, stream);
        }
        S.tk_precompute_done = true;
    }
#endif

    // 1. Edge preamble: geometry → Bessel × cutoff → spherical harmonics.
    launch_mace_edge_geometry(batch.n_edges, batch.positions, batch.edge_source_atom, batch.edge_target_atom,
                               S.r_norm, S.r_unit, stream);
    launch_mace_radial_basis(arch, batch.n_edges, W.bessel_weights,
                              S.r_norm, S.radial_basis, S.cutoff_envelope, stream);
    launch_mace_sph_harm(batch.n_edges, S.r_unit, S.sph_harm, stream);

    // 2. Node embedding: scalar features per atom from the one-hot element.
    launch_mace_node_embed(batch.n_atoms, arch.n_channels, arch.n_elements, irrep_dim,
                            batch.atom_element, W.node_embedding, S.node_features, stream);

    // 3. Two interaction layers.
    for (std::uint32_t layer_index = 0; layer_index < arch.num_interactions && layer_index < 2u; ++layer_index) {
        auto const &L = W.layers[layer_index];

        // Save the layer's input node features for the force-adjoint pass.
        // The forward overwrites `S.node_features` mid-iteration (inject_l0
        // step 3j), so without this snapshot the layer-0 backward has no
        // way to recover the original input.  Cheap: ~5-50 KB per save on
        // drug-like molecules, ordered on the same stream so it lands
        // before any downstream kernel reads it.
        if (S.node_features_input_layer[layer_index] != nullptr) {
            cudaMemcpyAsync(S.node_features_input_layer[layer_index],
                            S.node_features,
                            batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // 3a. linear_up: scalar-only channel mix.  Output goes into S.node_features_l0_only via the
        //     L=0 extract → linear → reinject path.
        // For the first iteration node_features has only L=0 populated by the embedding,
        // so extract just the L=0 slab.
        {
            constexpr std::uint32_t block_dim_x = 64;
            dim3 const block(block_dim_x, 1, 1);
            dim3 const grid(batch.n_atoms, (arch.n_channels + block_dim_x - 1u) / block_dim_x, 1);
            mace_detail_::mace_extract_l0_irrep_block_kernel<<<grid, block, 0, stream>>>(
                batch.n_atoms, arch.n_channels, irrep_dim, S.node_features, S.node_scalar_invariants);
        }
        // The linear_up weight is stored as [c_out, c_in] flat (n_channels=96 → 9216 floats).
        // Apply as a single "per-irrep" linear with active_irrep_block_count=1.  Reinterpret the
        // scalar input/output as `[n_atoms, c, 1]`.
        launch_mace_linear_per_irrep(batch.n_atoms, arch.n_channels, arch.n_channels,
                                      /*irrep_dim*/1u, /*active_irrep_block_count*/1u,
                                      L.linear_up_w, S.node_scalar_invariants, S.node_features_l0_only, stream);

        // Save the post-linear_up scalar features for the force-adjoint
        // (input to conv_tp; the conv_tp adjoint reads it back as `node_l0`).
        if (S.node_features_l0_only_layer[layer_index] != nullptr) {
            cudaMemcpyAsync(S.node_features_l0_only_layer[layer_index],
                            S.node_features_l0_only,
                            batch.n_atoms * arch.n_channels * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // 3a-bis. Layer-N (N >= 1) residual: in MACE the
        //   `RealAgnosticResidualInteractionBlock` computes
        //     sc = skip_tp(input_node_feats, atom_one_hot)        [shape n_atoms x n_channels]
        //   BEFORE linear_up overwrites things, then the
        //   `EquivariantProductBasisBlock` adds it AFTER product_linear:
        //     post_product_l0 += sc.
        //   Layer-0 (`RealAgnosticInteractionBlock`) doesn't have this — its skip_tp
        //   is applied to the post-linear message instead (handled in step 3g below).
        //
        //   The input to layer-N's skip_tp is the layer's input node features
        //   `S.node_features`, which after step 3j of the previous iteration has
        //   only the L=0 component populated (post-product features).  We've already
        //   extracted that L=0 slab into `S.node_scalar_invariants` above, so we
        //   reuse it here as the skip_tp input with `irrep_dim=1`.
        //
        //   Weight `L.skip_tp_w` for layer-N is sized for one irrep-block
        //   (96x0e × 10x0e → 96x0e ≡ 92160 floats).  active_irrep_block_count=1.
        if (layer_index >= 1u) {
            launch_mace_skip_tp(batch.n_atoms, arch.n_channels, arch.n_elements, arch.n_channels,
                                 /*irrep_dim*/1u, /*active_irrep_block_count*/1u,
                                 batch.atom_element, L.skip_tp_w,
                                 S.node_scalar_invariants, S.layer_n_residual_l0, stream);
        }

        // 3b. Radial MLP: 4-layer FullyConnectedNet with SiLU between.
        //     Input: radial_basis [n_edges, n_radial=8].
        //     Output: tensor_product_weights_per_edge  [n_edges, 4*c=384].
        //     Each hidden layer scales by (1/sqrt(h_in)) and applies normalize2mom(silu).
        //     Final layer scales by (1/sqrt(h_in)) only.
        // Layer 0: 8 → 64
        launch_mace_matmul_rowmajor(batch.n_edges, radial_mlp_hidden_dim, arch.n_radial,
                                    S.radial_basis, L.radial_mlp_w0,
                                    1.0f / sqrtf(static_cast<float>(arch.n_radial)),
                                    S.radial_mlp_hidden_0, stream);
        launch_mace_silu_norm2mom(batch.n_edges * radial_mlp_hidden_dim, S.radial_mlp_hidden_0, S.radial_mlp_hidden_0, stream);
        // Layer 1: 64 → 64
        launch_mace_matmul_rowmajor(batch.n_edges, radial_mlp_hidden_dim, radial_mlp_hidden_dim,
                                    S.radial_mlp_hidden_0, L.radial_mlp_w1,
                                    1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
                                    S.radial_mlp_hidden_1, stream);
        launch_mace_silu_norm2mom(batch.n_edges * radial_mlp_hidden_dim, S.radial_mlp_hidden_1, S.radial_mlp_hidden_1, stream);
        // Layer 2: 64 → 64
        launch_mace_matmul_rowmajor(batch.n_edges, radial_mlp_hidden_dim, radial_mlp_hidden_dim,
                                    S.radial_mlp_hidden_1, L.radial_mlp_w2,
                                    1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
                                    S.radial_mlp_hidden_2, stream);
        launch_mace_silu_norm2mom(batch.n_edges * radial_mlp_hidden_dim, S.radial_mlp_hidden_2, S.radial_mlp_hidden_2, stream);
        // Layer 3 (final, no act): 64 → 4*c
        std::uint32_t const tp_dim = mace_tensor_product_path_count_k * arch.n_channels;
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
        // TK-fused radial MLP — one launch instead of 7 (4 matmul + 3 SiLU
        // norm2mom).  Reads radial_basis fp32 directly; the kernel handles
        // the per-layer (1/sqrt(h_in)) scale + normalize2mom(silu) epilogue
        // internally and emits bf16 tp_weights.  We then cast bf16→fp32 and
        // apply the cutoff multiply in fp32 on the existing buffer so the
        // downstream conv_tp consumer (C2 fused TK or fp32 reference)
        // sees the same fp32 tp_weights buffer regardless of which radial
        // MLP path produced it.
        launch_mace_tk_radial_mlp(batch.n_edges, arch.n_radial,
                                  S.radial_basis,
                                  L.radial_mlp_w0, L.radial_mlp_w1,
                                  L.radial_mlp_w2, L.radial_mlp_w3,
                                  S.tk_tensor_product_weights_per_edge_bf16, stream);
        launch_mace_bf16_to_fp32(batch.n_edges * tp_dim,
                                 S.tk_tensor_product_weights_per_edge_bf16,
                                 S.tensor_product_weights_per_edge, stream);
#else
        launch_mace_matmul_rowmajor(batch.n_edges, tp_dim, radial_mlp_hidden_dim,
                                    S.radial_mlp_hidden_2, L.radial_mlp_w3,
                                    1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
                                    S.tensor_product_weights_per_edge, stream);
#endif
        // Save pre-cutoff tp_weights for the apply_cutoff backward (the apply_cutoff
        // multiply contributes a Σ_k grad_tp[e,k] * tp_pre[e,k] term to dr/d_positions
        // through dEnv(r); without the saved pre-cutoff buffer that path is lost).
        if (S.tensor_product_weights_pre_cutoff_layer[layer_index] != nullptr) {
            cudaMemcpyAsync(S.tensor_product_weights_pre_cutoff_layer[layer_index],
                            S.tensor_product_weights_per_edge,
                            batch.n_edges * tp_dim * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // Multiply per-edge by the cutoff envelope (mirrors `tp_weights *= cutoff` in the live forward).
        launch_mace_apply_cutoff(batch.n_edges, tp_dim, S.cutoff_envelope, S.tensor_product_weights_per_edge, stream);

#if defined(SMALLFIELD_USE_THUNDERKITTENS)
        // 3c+3d.  Two routes depending on whether the host has supplied
        // pre-sorted edge segments:
        //
        //   - Segments provided (`batch.receiver_edge_segment_start/end != nullptr`):
        //     dispatch the TK fused conv_tp+scatter, which emits messages
        //     directly into the per-receiver bf16 accumulator and saves the
        //     `[n_edges, c, 16]` DRAM round-trip (~28× bandwidth measured
        //     in the C2 smoke test).  Then cast bf16 → fp32 for downstream
        //     fp32 consumers.
        //
        //   - Segments NOT provided: avoid `launch_mace_tk_conv_tp_scatter_fused`
        //     entirely.  Its atomic fallback path does `cudaMalloc` /
        //     `cudaStreamSynchronize` / `cudaFree` per call, which are not
        //     legal inside `cudaStreamBeginCapture`/`cudaStreamEndCapture`
        //     and would block the Track-E graph-capture replay.  Instead,
        //     reuse the existing fp32 reference kernels (TP + atomic scatter)
        //     into pre-allocated scratch buffers — no per-call allocations,
        //     graph-capture-safe.  Same arithmetic; no bf16 round-off in
        //     this path because we never cast.
        if (batch.receiver_edge_segment_start != nullptr &&
            batch.receiver_edge_segment_end != nullptr) {
            launch_mace_tk_conv_tp_scatter_fused(batch.n_atoms, batch.n_edges,
                                                  arch.n_channels,
                                                  batch.edge_source_atom, batch.edge_target_atom,
                                                  batch.receiver_edge_segment_start,
                                                  batch.receiver_edge_segment_end,
                                                  S.node_features_l0_only, S.sph_harm,
                                                  S.tensor_product_weights_per_edge,
                                                  S.tk_node_messages_full_irrep_bf16, stream);
            launch_mace_bf16_to_fp32(batch.n_atoms * arch.n_channels * irrep_dim,
                                     S.tk_node_messages_full_irrep_bf16,
                                     S.node_features_scratch, stream);
        } else {
            launch_mace_tensor_product(arch, batch.n_edges, batch.edge_source_atom,
                                        S.node_features_l0_only, S.sph_harm,
                                        S.tensor_product_weights_per_edge,
                                        S.edge_messages, stream);
            launch_mace_scatter_atomic(batch.n_atoms, batch.n_edges,
                                        arch.n_channels * irrep_dim,
                                        batch.edge_target_atom, S.edge_messages,
                                        S.node_features_scratch, stream);
        }
#else
        // 3c. Tensor product: per-edge messages.
        launch_mace_tensor_product(arch, batch.n_edges, batch.edge_source_atom,
                                    S.node_features_l0_only, S.sph_harm, S.tensor_product_weights_per_edge, S.edge_messages, stream);

        // 3d. Scatter sum into receivers, full feature dim = n_channels * 16.
        launch_mace_scatter_atomic(batch.n_atoms, batch.n_edges,
                                    arch.n_channels * irrep_dim,
                                    batch.edge_target_atom, S.edge_messages, S.node_features_scratch, stream);
#endif

        // 3e. linear_post_tp: per-irrep channel mix on all 4 L blocks.
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
        // TK 4-block fused linear: one CTA per 4 atoms × 96 c_out, single
        // shared input tile, four WGMMA passes (one per L block).  Output
        // bf16 → cast back to fp32 in the existing node_features buffer.
        launch_mace_tk_per_irrep_linear(batch.n_atoms, arch.n_channels, arch.n_channels,
                                         /*active_irrep_block_count*/4u,
                                         L.linear_post_tp_w,
                                         S.node_features_scratch,
                                         S.tk_node_features_post_linear_bf16, stream);
        launch_mace_bf16_to_fp32(batch.n_atoms * arch.n_channels * irrep_dim,
                                 S.tk_node_features_post_linear_bf16,
                                 S.node_features, stream);
#else
        launch_mace_linear_per_irrep(batch.n_atoms, arch.n_channels, arch.n_channels,
                                      irrep_dim, /*active_irrep_block_count*/4u,
                                      L.linear_post_tp_w, S.node_features_scratch, S.node_features, stream);
#endif

        // 3f. Divide by avg_num_neighbors (in-place scale of node_features).
        if (avg_num_neighbors != 1.0f) {
            launch_mace_scale(batch.n_atoms * arch.n_channels * irrep_dim,
                              S.node_features, 1.0f / avg_num_neighbors, S.node_features, stream);
        }

        // 3g. skip_tp: layer-0 form is applied to the message AFTER linear_post_tp;
        //     layer-1 form is applied to the ORIGINAL node features (residual style).
        //     For Phase 1 we wire only the layer-0 form (in-place message conditioning).
        //     The layer-1 residual is a one-line additional kernel — a follow-up.
        if (layer_index == 0u) {
            std::uint32_t const active_irrep_block_count = 4u;
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
            // TK element-conditioned residual: same algebra as
            // `mace_element_conditioned_residual_kernel` but using the
            // Z-indirect WGMMA path (96×96 weight slab per atom, two passes
            // because c_out=96 doesn't divide WGMMA M=64 cleanly).
            launch_mace_tk_element_conditioned_residual(
                batch.n_atoms, arch.n_channels, arch.n_elements, arch.n_channels,
                /*active_irrep_block_count*/4u,
                batch.atom_element, L.skip_tp_w,
                S.node_features,
                S.tk_node_features_post_residual_bf16, stream);
            launch_mace_bf16_to_fp32(batch.n_atoms * arch.n_channels * irrep_dim,
                                     S.tk_node_features_post_residual_bf16,
                                     S.node_features, stream);
#else
            launch_mace_skip_tp(batch.n_atoms, arch.n_channels, arch.n_elements, arch.n_channels,
                                 irrep_dim, active_irrep_block_count, batch.atom_element,
                                 L.skip_tp_w, S.node_features, S.node_features_scratch, stream);
            // Copy back to node_features (or just use scratch as the "current" buffer next).
            cudaMemcpyAsync(S.node_features, S.node_features_scratch,
                            batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
#endif
        }

        // Save the pre-symmetric-contraction node_features (post-skip_tp /
        // post-residual / post-/avg_num_neighbors) for the force-adjoint:
        // the symmetric-contraction adjoint takes this exact tensor as its
        // `X` input.  Layer-0 form rolls in skip_tp; layer-N residual is
        // applied AFTER product_linear, so the post_residual save here is
        // pre-residual for layer >= 1.
        if (S.node_features_post_residual_layer[layer_index] != nullptr) {
            cudaMemcpyAsync(S.node_features_post_residual_layer[layer_index],
                            S.node_features,
                            batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // 3h. Symmetric contraction: produce per-channel scalar invariants.
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
        // TK path uses precomputed `W3_eff[Z, i, j, k, c]` so the order-3
        // contraction becomes a sum-of-cubic-products with `c` as the
        // pointwise axis.  Output is fp32 (the order-3 accumulator's
        // precision is load-bearing); no bf16 cast bridge here.
        launch_mace_tk_symmetric_contraction(
            batch.n_atoms, arch.n_channels, arch.n_elements,
            batch.atom_element,
            S.node_features,
            L.u_matrix_1, L.u_matrix_2,
            L.weights_lo_0, L.weights_lo_1,
            S.tk_precomputed_w3_effective_layer[layer_index],
            S.node_scalar_invariants, stream);
#else
        launch_mace_symmetric_contraction(arch, L, batch.n_atoms, batch.atom_element,
                                           S.node_features, S.node_scalar_invariants, stream);
#endif

        // Save the post-symmetric-contraction invariants for the
        // force-adjoint (input to product_linear; the product_linear
        // adjoint reads it back).
        if (S.node_scalar_invariants_layer[layer_index] != nullptr) {
            cudaMemcpyAsync(S.node_scalar_invariants_layer[layer_index],
                            S.node_scalar_invariants,
                            batch.n_atoms * arch.n_channels * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // 3i. product_linear: channel mix on the scalar invariants (96 → 96).
        launch_mace_linear_per_irrep(batch.n_atoms, arch.n_channels, arch.n_channels,
                                      /*irrep_dim*/1u, /*active_irrep_block_count*/1u,
                                      L.product_linear_w, S.node_scalar_invariants, S.node_features_l0_only, stream);

        // 3i-bis. Add the layer-N skip_tp residual back into the post-product L=0 features.
        //   Mirrors `EquivariantProductBasisBlock.forward`:
        //     return self.linear(symmetric_contractions(...)) + sc_residual.
        //   Layer-0 has no residual (sc is None there).
        if (layer_index >= 1u) {
            launch_mace_add_inplace(batch.n_atoms * arch.n_channels,
                                     S.layer_n_residual_l0, S.node_features_l0_only, stream);
        }

        // Save post-product-linear (post-residual for layer-N) L=0 features.
        // This is the readout MLP's forward input, needed by the readout MLP backward.
        if (S.node_features_l0_only_post_product_layer[layer_index] != nullptr) {
            cudaMemcpyAsync(S.node_features_l0_only_post_product_layer[layer_index],
                            S.node_features_l0_only,
                            batch.n_atoms * arch.n_channels * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // 3j. Inject the post-product L=0 features back into node_features for the next iteration's input.
        {
            constexpr std::uint32_t block_dim_x = 64;
            dim3 const block(block_dim_x, 1, 1);
            dim3 const grid(batch.n_atoms, (arch.n_channels + block_dim_x - 1u) / block_dim_x, 1);
            mace_detail_::mace_inject_l0_irrep_block_kernel<<<grid, block, 0, stream>>>(
                batch.n_atoms, arch.n_channels, irrep_dim, S.node_features_l0_only, S.node_features);
        }

        // 3k. Apply the readout for this layer to S.node_features_l0_only (the per-channel L=0 features).
        //     TK build dispatches to the WGMMA readout kernels (commit 0b998e6); the
        //     fp32 reference build keeps the warp-reduce kernels in `mace_detail_`.
        if (layer_index == 0u) {
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
            launch_mace_tk_readout_linear(batch.n_atoms, arch.n_channels,
                                          S.node_features_l0_only, W.readout_linear_w,
                                          S.node_energy, stream);
#else
            float const ro_scale = 1.0f / sqrtf(static_cast<float>(arch.n_channels));
            mace_detail_::mace_readout_linear_kernel<<<batch.n_atoms, 32u, 0, stream>>>(
                batch.n_atoms, arch.n_channels, ro_scale, S.node_features_l0_only, W.readout_linear_w, S.node_energy);
#endif
        } else {
            constexpr std::uint32_t hidden_dim = 16u;
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
            launch_mace_tk_readout_nonlinear_mlp(batch.n_atoms, arch.n_channels, hidden_dim,
                                                 S.node_features_l0_only,
                                                 W.readout_mlp_w0, W.readout_mlp_w1,
                                                 S.node_energy, stream);
#else
            float const sc1 = 1.0f / sqrtf(static_cast<float>(arch.n_channels));
            float const sc2 = 1.0f / sqrtf(static_cast<float>(hidden_dim));
            mace_detail_::mace_readout_nonlinear_mlp_kernel<<<batch.n_atoms, 32u,
                                                          hidden_dim * sizeof(float), stream>>>(
                batch.n_atoms, arch.n_channels, hidden_dim, sc1, sc2, S.node_features_l0_only,
                W.readout_mlp_w0, W.readout_mlp_w1, S.node_energy);
#endif
        }
    }

    // 4. Reduce per-atom contributions to per-graph energy with scale_shift + atomic baseline.
    cudaMemsetAsync(S.energy, 0, batch.n_graphs * sizeof(float), stream);
    {
        constexpr std::uint32_t block = 128;
        std::uint32_t const grid = (batch.n_atoms + block - 1u) / block;
        mace_detail_::mace_per_graph_energy_reduce_kernel<<<grid, block, 0, stream>>>(
            batch.n_atoms, arch.n_channels, arch.scale, arch.shift,
            batch.atom_element, batch.graph_of_atom,
            W.atomic_energies, S.node_energy, S.energy);
    }
}

/** Convenience: zero `node_energy` before each forward (since we accumulate). */
inline void mace_inference_reset_per_atom_energy_accumulator(mace_scratch_t &S, std::uint32_t n_atoms, cudaStream_t stream) {
    cudaMemsetAsync(S.node_energy, 0, n_atoms * sizeof(float), stream);
}

/**
 *  @brief Run the force-adjoint pass to produce per-atom forces from a
 *         post-forward `mace_scratch_t`.
 *
 *  Forces are `-∂E/∂r`.  The implementation walks the forward chain in
 *  reverse, calling each adjoint kernel in `mace_force_adjoints.cuh` against
 *  the corresponding forward intermediate stashed in `mace_scratch_t`.  Output
 *  lands in `S.forces[n_atoms, 3]` (fp32).
 *
 *  Status: this is a SCAFFOLD wired up to the energy-reduce + readout
 *  adjoints.  The full chain (per-layer scatter / TP / linear / skip_tp /
 *  symmetric-contraction / radial / sph-harm adjoints) is staged via the
 *  per-layer `node_features_*_layer[2]` saves added to `mace_scratch_t`,
 *  but the forward currently does not yet populate those layer-saves.
 *  The next iteration:
 *
 *    1. Modify `mace_inference_forward` to copy `S.node_features` and
 *       `S.node_scalar_invariants` (and `tensor_product_weights_per_edge`)
 *       into the per-layer save buffers at each layer's boundary.
 *    2. Implement the full reverse loop here, calling each adjoint launcher
 *       declared in `mace_force_adjoints.cuh`.
 *    3. Validate per-atom force RMS against the `grad_positions` golden
 *       tensor exported by `python/smallfield/mace_export.py` (= `-forces`
 *       from `compute_force=True` in MACE Python).
 *
 *  Caller responsibility: must run `mace_inference_forward` on the same
 *  `(arch, W, batch, S)` immediately before calling this function.
 *  `S.grad_per_graph_energy` is initialised here to all 1.0 (one unit of
 *  output energy gradient per graph) before the chain begins.
 */
inline void mace_inference_run_force_adjoint(
    types::mace_arch_t const &arch,
    types::mace_weights_device_t const &W,
    types::mace_batch_t const &batch,
    mace_scratch_t &S,
    cudaStream_t stream
) {
    constexpr std::uint32_t irrep_dim = 16u;
    constexpr std::uint32_t readout_mlp_hidden_dim = 16u;
    constexpr std::uint32_t radial_mlp_hidden_dim = 64u;
    std::uint32_t const tp_dim = mace_tensor_product_path_count_k * arch.n_channels;
    float const avg_num_neighbors = arch.avg_num_neighbors;

    // 1. Initialise gradient buffers.
    //    - grad_per_graph_energy = 1.0 (one unit of output gradient per graph
    //      energy; equivalent to differentiating sum-of-energies wrt positions).
    //    - All other grad_* buffers are zeroed; adjoints accumulate via += or atomicAdd.
    //    Use a device-side fill (graph-capture safe) rather than a host->device
    //    memcpy — host data wouldn't survive into the captured graph.
    launch_mace_fill_constant(S.grad_per_graph_energy, batch.n_graphs, 1.0f, stream);
    cudaMemsetAsync(S.grad_node_energy,             0, batch.n_atoms * sizeof(float), stream);
    cudaMemsetAsync(S.grad_node_scalar_invariants,  0, batch.n_atoms * arch.n_channels * sizeof(float), stream);
    cudaMemsetAsync(S.grad_node_features,           0, batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float), stream);
    cudaMemsetAsync(S.grad_node_features_scratch,   0, batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float), stream);
    cudaMemsetAsync(S.grad_node_features_l0_only,   0, batch.n_atoms * arch.n_channels * sizeof(float), stream);
    cudaMemsetAsync(S.grad_edge_messages,           0, batch.n_edges * arch.n_channels * irrep_dim * sizeof(float), stream);
    cudaMemsetAsync(S.grad_tp_weights,              0, batch.n_edges * tp_dim * sizeof(float), stream);
    cudaMemsetAsync(S.grad_sph_harm,                0, batch.n_edges * irrep_dim * sizeof(float), stream);
    cudaMemsetAsync(S.grad_r_unit,                  0, batch.n_edges * 3u * sizeof(float), stream);
    cudaMemsetAsync(S.grad_radial_basis,            0, batch.n_edges * arch.n_radial * sizeof(float), stream);
    cudaMemsetAsync(S.grad_positions,               0, batch.n_atoms * 3u * sizeof(float), stream);

    // 2. Per-graph energy reduce adjoint:
    //    grad_node_energy[a] = arch.scale * grad_per_graph_energy[graph_of_atom[a]].
    //    The atomic_energies baseline and the constant `shift` have zero
    //    position-gradient.
    launch_mace_per_graph_energy_force_adjoint(
        batch.n_atoms, arch.scale, batch.graph_of_atom,
        S.grad_per_graph_energy, S.grad_node_energy, stream);

    // 3. Per-layer reverse loop.  Layers run in reverse: N-1 first.  Each
    //    iteration produces grad_node_features_input_layer (the gradient of
    //    the loss wrt this layer's input), which becomes the "post-product
    //    l0 gradient" for the previous layer once we extract the L=0 slot.
    std::uint32_t const num_layers = (arch.num_interactions < 2u) ? arch.num_interactions : 2u;
    for (std::int32_t layer_signed = static_cast<std::int32_t>(num_layers) - 1; layer_signed >= 0; --layer_signed) {
        std::uint32_t const layer_index = static_cast<std::uint32_t>(layer_signed);
        auto const &L = W.layers[layer_index];

        // 3a. Readout backward → grad_node_features_l0_only (post-product l0 grad).
        //     Layer 0 uses the linear readout (alpha = n_channels);
        //     layer N uses the nonlinear MLP (hidden_dim = 16, alpha = hidden_dim
        //     for the final stage, alpha = n_channels for the first).
        if (layer_index == 0u) {
            float const ro_scale = 1.0f / sqrtf(static_cast<float>(arch.n_channels));
            launch_mace_readout_linear_force_adjoint(
                batch.n_atoms, arch.n_channels, ro_scale, W.readout_linear_w,
                S.grad_node_energy, S.grad_node_features_l0_only, stream);
        } else {
            float const sc1 = 1.0f / sqrtf(static_cast<float>(arch.n_channels));
            float const sc2 = 1.0f / sqrtf(static_cast<float>(readout_mlp_hidden_dim));
            launch_mace_readout_nonlinear_mlp_force_adjoint(
                batch.n_atoms, arch.n_channels, readout_mlp_hidden_dim, sc1, sc2,
                S.node_features_l0_only_post_product_layer[layer_index],
                W.readout_mlp_w0, W.readout_mlp_w1,
                S.grad_node_energy, S.grad_node_features_l0_only, stream);
        }

        // 3b. (layer-N >= 1) Residual addition `post_product_l0 += layer_n_residual_l0`
        //     has identity gradient — both incoming and residual paths see the
        //     same upstream.  The residual path back-propagates via:
        //         skip_tp_layerN_backward → grad_layer_residual_l0_scratch (scalar)
        //     storing the layer-input scalar (L=0) residual contribution that
        //     we'll merge with the message-path layer-input gradient at the
        //     end of this layer's reverse pass (step 3o).  Using a dedicated
        //     scratch keeps the residual contribution out of the
        //     `grad_node_features` buffer that the symmetric-contraction and
        //     downstream adjoints reuse.
        if (layer_index >= 1u) {
            cudaMemsetAsync(S.grad_layer_residual_l0_scratch, 0,
                            batch.n_atoms * arch.n_channels * sizeof(float), stream);
            launch_mace_element_conditioned_residual_force_adjoint(
                batch.n_atoms, arch.n_channels, arch.n_elements, arch.n_channels,
                /*irrep_dim*/1u, /*active_irrep_block_count*/1u,
                batch.atom_element, L.skip_tp_w,
                S.grad_node_features_l0_only,         // upstream (post-product l0 gradient)
                S.grad_layer_residual_l0_scratch,     // grad of skip_tp INPUT (scalar)
                stream);
        }

        // 3c. product_linear adjoint: post-product l0 → grad_node_scalar_invariants.
        //     Forward: post_product[a, c_out] = (1/sqrt(c_in)) * Σ_{c_in} W[c_in, c_out] * sym[a, c_in].
        //     Backward: dsym[a, c_in] += (1/sqrt(c_in)) * Σ_{c_out} W[c_in, c_out] * dpost[a, c_out].
        //     The kernel uses `+=`; zero the destination so we don't accumulate
        //     onto stale data from the previous layer's reverse pass (the
        //     linear_up backward writes here at step 3l).
        cudaMemsetAsync(S.grad_node_scalar_invariants, 0,
                        batch.n_atoms * arch.n_channels * sizeof(float), stream);
        launch_mace_linear_per_irrep_force_adjoint(
            batch.n_atoms, arch.n_channels, arch.n_channels,
            /*irrep_dim*/1u, /*active_irrep_block_count*/1u,
            L.product_linear_w,
            S.grad_node_features_l0_only,
            S.grad_node_scalar_invariants, stream);

        // 3d. Symmetric contraction adjoint:
        //     grad_node_features (full irreps, post-skip_tp) gets the gradient
        //     w.r.t. the symcontraction's X input.  X = node_features_post_residual_layer[L].
        //     Launcher signature is (weights_max, weights_2, weights_1).  Pass
        //     (L.weights_max, L.weights_lo_0, L.weights_lo_1) since `weights_lo_0`
        //     is the order-2 buffer and `weights_lo_1` the order-1 buffer (matches
        //     the forward launcher in `mace_symmetric_contraction_kernel.cuh`).
        launch_mace_symmetric_contraction_force_adjoint(
            batch.n_atoms, arch.n_channels, irrep_dim, arch.n_elements,
            batch.atom_element,
            S.node_features_post_residual_layer[layer_index],
            L.u_matrix_1, L.u_matrix_2, L.u_matrix_3,
            L.weights_max, L.weights_lo_0, L.weights_lo_1,
            S.grad_node_scalar_invariants,
            S.grad_node_features, stream);

        // 3e. (layer 0) Element-conditioned residual on the message path:
        //     `node_features_post_skip_tp = skip_tp(node_features_post_avg, atom_element)`.
        //     Reverse: grad_node_features_scratch = skip_tp_backward(grad_node_features).
        //     For layer-N (>= 1) skip_tp on the message path is identity, so
        //     just copy the gradient through.
        if (layer_index == 0u) {
            launch_mace_element_conditioned_residual_force_adjoint(
                batch.n_atoms, arch.n_channels, arch.n_elements, arch.n_channels,
                irrep_dim, /*active_irrep_block_count*/4u,
                batch.atom_element, L.skip_tp_w,
                S.grad_node_features,                 // upstream (post-skip_tp grad)
                S.grad_node_features_scratch,         // pre-skip_tp grad
                stream);
        } else {
            cudaMemcpyAsync(S.grad_node_features_scratch, S.grad_node_features,
                            batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // 3f. /avg_num_neighbors backward: scale the gradient by 1/avg_num_neighbors.
        if (avg_num_neighbors != 1.0f) {
            launch_mace_scale_force_adjoint(
                batch.n_atoms * arch.n_channels * irrep_dim,
                1.0f / avg_num_neighbors, S.grad_node_features_scratch, stream);
        }

        // 3g. linear_post_tp adjoint: grad_node_features_scratch (post-linear)
        //     → grad_node_features (pre-linear, = post-scatter gradient).
        //     We must zero grad_node_features first since the launcher uses +=
        //     and the buffer holds the symmetric-contraction gradient already.
        cudaMemsetAsync(S.grad_node_features, 0,
                        batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float), stream);
        launch_mace_linear_per_irrep_force_adjoint(
            batch.n_atoms, arch.n_channels, arch.n_channels,
            irrep_dim, /*active_irrep_block_count*/4u,
            L.linear_post_tp_w,
            S.grad_node_features_scratch,
            S.grad_node_features, stream);

        // 3h. Fused tensor-product + scatter adjoint: takes the post-scatter
        //     receiver-grad `grad_node_features [n_atoms, n_channels, 16]`
        //     directly (the kernel performs the gather internally), and emits
        //     gradients for the three forward inputs:
        //       grad_node_l0 (atomicAdd via source atom),
        //       grad_tp_weights (per-edge non-atomic),
        //       grad_sph_harm (atomicAdd accumulating).
        //     Use grad_node_features_l0_only as the grad_node_l0 destination
        //     (it was just consumed by product_linear backward, free to reuse).
        cudaMemsetAsync(S.grad_node_features_l0_only, 0,
                        batch.n_atoms * arch.n_channels * sizeof(float), stream);
        launch_mace_tensor_product_and_scatter_force_adjoint(
            batch.n_atoms, batch.n_edges, arch.n_channels,
            batch.edge_source_atom, batch.edge_target_atom,
            S.node_features_l0_only_layer[layer_index],
            S.sph_harm,
            S.tensor_product_weights_per_edge,        // post-cutoff (matches forward state)
            S.grad_node_features,                     // upstream: per-atom (target-indexed)
            S.grad_node_features_l0_only,             // atomicAdd accumulator
            S.grad_tp_weights,                        // overwritten
            S.grad_sph_harm                           // atomicAdd accumulator
            , stream);

        // 3j. apply_cutoff adjoint: rescales grad_tp by cutoff_envelope, and
        //     reduces a grad_env contribution per edge that we route through
        //     dpositions via the env-derivative chain.
        //     We reuse `S.grad_radial_basis` slot 0 as a scratch grad_env
        //     accumulator?  No — grad_radial_basis is [n_edges, n_radial]
        //     and grad_env is [n_edges].  Use `S.r_norm`'s shape — but
        //     that's used in forward.  Allocate a temporary on the stream:
        //     the scratch arena doesn't have a [n_edges] grad-env buffer, so
        //     we route grad_env via grad_radial_basis's first column slot
        //     temporarily and then atomic-add the dpositions update from a
        //     short kernel.
        //
        //     Simpler: use a separate scratch.  We'll cudaMallocAsync for
        //     `n_edges` floats only when this layer's reverse pass runs;
        //     this allocation is small and rare (cudaMallocAsync is graph-
        //     capture safe on Hopper).  But the work plan's graph-capture
        //     constraint forbids per-call cudaMalloc — so instead reuse
        //     `S.cutoff_envelope` AS the destination since by this point in
        //     the forward+backward the forward has already consumed it
        //     (both the radial-basis kernel and the apply_cutoff multiply
        //     have already read it; the forward radial-geometry adjoint
        //     doesn't read it again).  Caveat: the radial-basis adjoint
        //     below DOES read env via recompute (not via cutoff_envelope
        //     directly), so the cutoff_envelope buffer is free to repurpose
        //     here.
        //
        //     IMPORTANT: this destroys S.cutoff_envelope, so callers must
        //     not rely on it after run_force_adjoint.  Documented in the
        //     scratch comment above.
        //
        //     But wait — in the layer loop we may need cutoff_envelope at
        //     the start of layer 0's reverse pass too (we just consumed
        //     layer 1's grad_tp; layer 0 will multiply its grad_tp by env
        //     too).  So we cannot destroy cutoff_envelope in layer 1.
        //
        //     Fix: use a small per-layer scratch.  Add `grad_cutoff_envelope`
        //     to scratch in a follow-up if profile shows this matters.  For
        //     now, route grad_env into a transient device alloc — graph
        //     capture isn't yet wired with the backward pass, so the
        //     cudaMallocAsync is allowed.  When G1 lands we'll promote
        //     this to a permanent scratch buffer.
        float *grad_cutoff_envelope_scratch = nullptr;
        cudaMallocAsync(reinterpret_cast<void **>(&grad_cutoff_envelope_scratch),
                        batch.n_edges * sizeof(float), stream);
        launch_mace_apply_cutoff_force_adjoint(
            batch.n_edges, tp_dim,
            S.cutoff_envelope,
            S.tensor_product_weights_pre_cutoff_layer[layer_index],
            S.grad_tp_weights,                        // in-place rescale
            grad_cutoff_envelope_scratch,
            stream);
        // Route grad_env through dEnv/dr → atomic-add into grad_positions.
        launch_mace_cutoff_grad_env_to_positions(
            arch, batch.n_edges, S.r_norm, S.r_unit,
            batch.edge_source_atom, batch.edge_target_atom,
            grad_cutoff_envelope_scratch, S.grad_positions, stream);
        cudaFreeAsync(grad_cutoff_envelope_scratch, stream);

        // 3k. radial_mlp adjoint: grad_tp_weights → grad_radial_basis.
        //     Backward through the 4-layer MLP with normalize2mom(silu)
        //     epilogues.  We need the post-activation buffers for layers
        //     0/1/2 to recover pre-activation via Newton inversion.
        //
        //     Note: in the TK build the forward radial MLP is fused — the
        //     post-activation buffers `S.radial_mlp_hidden_*` are NOT
        //     populated by the TK kernel.  In that case we re-run the
        //     fp32 forward radial MLP as a precondition for the backward
        //     (cheap: 4 small matmuls × 8 atoms-bin worth of edges).
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
        // Re-run the fp32 radial MLP forward to populate hidden buffers so
        // the backward kernel can recover pre-activations.
        launch_mace_matmul_rowmajor(batch.n_edges, radial_mlp_hidden_dim, arch.n_radial,
                                    S.radial_basis, L.radial_mlp_w0,
                                    1.0f / sqrtf(static_cast<float>(arch.n_radial)),
                                    S.radial_mlp_hidden_0, stream);
        launch_mace_silu_norm2mom(batch.n_edges * radial_mlp_hidden_dim,
                                   S.radial_mlp_hidden_0, S.radial_mlp_hidden_0, stream);
        launch_mace_matmul_rowmajor(batch.n_edges, radial_mlp_hidden_dim, radial_mlp_hidden_dim,
                                    S.radial_mlp_hidden_0, L.radial_mlp_w1,
                                    1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
                                    S.radial_mlp_hidden_1, stream);
        launch_mace_silu_norm2mom(batch.n_edges * radial_mlp_hidden_dim,
                                   S.radial_mlp_hidden_1, S.radial_mlp_hidden_1, stream);
        launch_mace_matmul_rowmajor(batch.n_edges, radial_mlp_hidden_dim, radial_mlp_hidden_dim,
                                    S.radial_mlp_hidden_1, L.radial_mlp_w2,
                                    1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
                                    S.radial_mlp_hidden_2, stream);
        launch_mace_silu_norm2mom(batch.n_edges * radial_mlp_hidden_dim,
                                   S.radial_mlp_hidden_2, S.radial_mlp_hidden_2, stream);
#endif
        // Layer 3 backward (no activation): grad_h2 = (1/sqrt(h)) * Σ_out W3[h, out] * grad_out.
        // Reuse `S.edge_messages` as scratch for grad_h2 (its forward content is no
        // longer needed here in the reverse pass).
        launch_mace_radial_mlp_layer_force_adjoint(
            batch.n_edges, radial_mlp_hidden_dim, tp_dim,
            1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
            /*apply_silu_n2m_inverse*/false,
            /*accumulate*/false,
            /*post_activation*/nullptr,    // unused when no epilogue
            L.radial_mlp_w3, S.grad_tp_weights,
            S.edge_messages, stream);
        // Layer 2 backward (with epilogue):
        launch_mace_radial_mlp_layer_force_adjoint(
            batch.n_edges, radial_mlp_hidden_dim, radial_mlp_hidden_dim,
            1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
            /*apply_silu_n2m_inverse*/true,
            /*accumulate*/false,
            S.radial_mlp_hidden_2, L.radial_mlp_w2, S.edge_messages,
            S.grad_node_features, stream);  // reuse S.grad_node_features as scratch
        // Layer 1 backward (with epilogue):
        launch_mace_radial_mlp_layer_force_adjoint(
            batch.n_edges, radial_mlp_hidden_dim, radial_mlp_hidden_dim,
            1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
            /*apply_silu_n2m_inverse*/true,
            /*accumulate*/false,
            S.radial_mlp_hidden_1, L.radial_mlp_w1, S.grad_node_features,
            S.edge_messages, stream);
        // Layer 0 backward (with epilogue): grad_radial_basis += contribution.
        // Both interaction layers' radial MLPs share the same radial_basis input,
        // so we ACCUMULATE to preserve the previous layer's contribution.
        launch_mace_radial_mlp_layer_force_adjoint(
            batch.n_edges, arch.n_radial, radial_mlp_hidden_dim,
            1.0f / sqrtf(static_cast<float>(arch.n_radial)),
            /*apply_silu_n2m_inverse*/true,
            /*accumulate*/true,
            S.radial_mlp_hidden_0, L.radial_mlp_w0, S.edge_messages,
            S.grad_radial_basis, stream);

        // 3l. linear_up adjoint: grad_node_l0 (post-linear scalar, in
        //     `grad_node_features_l0_only`) → scalar gradient on
        //     pre-linear_up scalar input.  Stage into grad_node_scalar_invariants
        //     (overwrite — re-zero first).
        cudaMemsetAsync(S.grad_node_scalar_invariants, 0,
                        batch.n_atoms * arch.n_channels * sizeof(float), stream);
        launch_mace_linear_per_irrep_force_adjoint(
            batch.n_atoms, arch.n_channels, arch.n_channels,
            /*irrep_dim*/1u, /*active_irrep_block_count*/1u,
            L.linear_up_w,
            S.grad_node_features_l0_only,
            S.grad_node_scalar_invariants, stream);

        // 3m. extract_l0 adjoint: scalar gradient → L=0 slot of
        //     grad_node_features (which is the layer-input gradient
        //     accumulator).  Re-zero grad_node_features first since
        //     the radial MLP backward used it as scratch.
        cudaMemsetAsync(S.grad_node_features, 0,
                        batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float), stream);
        launch_mace_extract_l0_irrep_block_force_adjoint(
            batch.n_atoms, arch.n_channels, irrep_dim,
            S.grad_node_scalar_invariants, S.grad_node_features, stream);

        // 3o. (layer-N >= 1) Merge in the residual contribution we staged at
        //     step 3b: add the scalar `grad_layer_residual_l0_scratch` onto
        //     the L=0 slot of `grad_node_features` (the layer-input gradient).
        if (layer_index >= 1u) {
            launch_mace_extract_l0_irrep_block_force_adjoint(
                batch.n_atoms, arch.n_channels, irrep_dim,
                S.grad_layer_residual_l0_scratch, S.grad_node_features, stream);
        }

        // 3n. Hand off to the previous layer:
        //     The output of the previous layer (or the embedding for layer 0)
        //     is `inject_l0(post_product_l0[layer-1])` — only the L=0 slot
        //     has non-zero forward sensitivity.  Extract into the new
        //     grad_node_features_l0_only.
        if (layer_index >= 1u) {
            cudaMemsetAsync(S.grad_node_features_l0_only, 0,
                            batch.n_atoms * arch.n_channels * sizeof(float), stream);
            launch_mace_inject_l0_irrep_block_force_adjoint(
                batch.n_atoms, arch.n_channels, irrep_dim,
                S.grad_node_features, S.grad_node_features_l0_only, stream);
        }
    }

    // 4. sph_harm adjoint: grad_sph_harm → grad_r_unit.
    launch_mace_sph_harm_force_adjoint(batch.n_edges, S.r_unit, S.grad_sph_harm,
                                        S.grad_r_unit, stream);

    // 5. radial_basis × cutoff envelope + edge geometry adjoint:
    //    grad_radial_basis + grad_r_unit → grad_positions (atomic accumulate).
    launch_mace_radial_basis_and_edge_geometry_force_adjoint(
        arch, batch.n_edges, W.bessel_weights,
        S.r_norm, S.r_unit, batch.edge_source_atom, batch.edge_target_atom,
        S.grad_radial_basis, S.grad_r_unit, S.grad_positions, stream);

    // 6. forces[a, k] = -grad_positions[a, k].
    launch_mace_force_negate(batch.n_atoms, S.grad_positions, S.forces, stream);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
