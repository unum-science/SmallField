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
#include <cstdlib>
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
#  include "smallfield/cuda/mace_tk_per_irrep_linear_force_adjoint_kernel.cuh"
#  include "smallfield/cuda/mace_tk_conv_tp_scatter_fused_force_adjoint_kernel.cuh"
#  include "smallfield/cuda/mace_tk_element_conditioned_residual_force_adjoint_kernel.cuh"
#  include "smallfield/cuda/mace_tk_symmetric_contraction_force_adjoint_kernel.cuh"
#  include "smallfield/cuda/mace_tk_radial_mlp_force_adjoint_kernel.cuh"
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
    float *radial_mlp_hidden_0 = nullptr;          // [n_edges, hidden_dim] — POST-silu_n2m
    float *radial_mlp_hidden_1 = nullptr;
    float *radial_mlp_hidden_2 = nullptr;
    // Pre-activation tape, populated by the backward chain's per-layer rerun.
    // silu_n2m is non-monotonic (min ≈ -0.467 * N at x ≈ -1.278), so inverting
    // from post-activation via Newton picks the wrong branch when the forward
    // sat below x = -1.278 — kills sign of the chain rule for those elements.
    // Saving the pre-activation directly avoids the inversion.
    float *radial_mlp_hidden_0_pre = nullptr;
    float *radial_mlp_hidden_1_pre = nullptr;
    float *radial_mlp_hidden_2_pre = nullptr;
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
    // Per-layer pre-activation outputs from the TK fused radial MLP (Phase B6) —
    // saves the fp32 forward rerun in the backward pass.  Shape [n_edges, 64]
    // (64 = radial-MLP hidden dim).  Indexed by interaction layer because each
    // layer has its own radial MLP weights; without per-layer saves the
    // backward iteration would see the LAST layer's pre-acts in every iter.
    __nv_bfloat16 *tk_radial_mlp_hidden_0_pre_bf16_layer[2] = {nullptr, nullptr};
    __nv_bfloat16 *tk_radial_mlp_hidden_1_pre_bf16_layer[2] = {nullptr, nullptr};
    __nv_bfloat16 *tk_radial_mlp_hidden_2_pre_bf16_layer[2] = {nullptr, nullptr};

    // Order-3 ACE precomputed weights, one device buffer per interaction layer:
    //   `W3_eff[Z, irrep_index_i, irrep_index_j, irrep_index_k, channel_index]
    //         = sum over path of U_3[i,j,k,path] * weights_max[Z, path, c]`
    // Allocated lazily on first forward and freed in `mace_scratch_t::free`.
    float *tk_precomputed_w3_effective_layer[2] = {nullptr, nullptr};
    // Transform #1 — symmetrise W3_eff over (i, j, k) permutations into the
    // canonical `(i ≤ j ≤ k)` cell.  Forward kernel iterates 816 canonical
    // triples instead of 4096 ordered triples.  Same shape as W3_eff.
    float *tk_precomputed_w3_sym_total_layer[2] = {nullptr, nullptr};
    // Transform #1 (backward) — precomputed W3_partsym_sym[Z, m, a, b, c] for
    // a ≤ b.  The backward kernel evaluates ∂B/∂X[m] = Σ_{a≤b} W3_partsym_sym
    // [m,a,b,c] · X[a]·X[b] using a 136-entry X·X cache.  ~5.6× FMA-count
    // reduction vs the original 16³ × 3 cubic-write structure.
    float *tk_precomputed_w3_partsym_layer[2] = {nullptr, nullptr};
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
        alloc(&radial_mlp_hidden_0_pre, max_edges * radial_mlp_hidden_dim);
        alloc(&radial_mlp_hidden_1_pre, max_edges * radial_mlp_hidden_dim);
        alloc(&radial_mlp_hidden_2_pre, max_edges * radial_mlp_hidden_dim);
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
        // Phase B6: per-layer pre-activation buffers from the TK fused radial MLP.
        for (std::uint32_t layer_index = 0; layer_index < 2u; ++layer_index) {
            alloc_bf16(&tk_radial_mlp_hidden_0_pre_bf16_layer[layer_index], max_edges * radial_mlp_hidden_dim);
            alloc_bf16(&tk_radial_mlp_hidden_1_pre_bf16_layer[layer_index], max_edges * radial_mlp_hidden_dim);
            alloc_bf16(&tk_radial_mlp_hidden_2_pre_bf16_layer[layer_index], max_edges * radial_mlp_hidden_dim);
        }
        // W3_eff buffers are precomputed lazily on first forward (we need the
        // weight pointers to do the reduction); leave them nullptr here.
        tk_precompute_done = false;
#endif
    }

    void free() noexcept {
        for (float **p : {&r_norm, &r_unit, &radial_basis, &cutoff_envelope, &sph_harm,
                          &radial_mlp_hidden_0, &radial_mlp_hidden_1, &radial_mlp_hidden_2,
                          &radial_mlp_hidden_0_pre, &radial_mlp_hidden_1_pre, &radial_mlp_hidden_2_pre,
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
        for (std::uint32_t layer_index = 0; layer_index < 2u; ++layer_index) {
            for (__nv_bfloat16 **p : {&tk_radial_mlp_hidden_0_pre_bf16_layer[layer_index],
                                       &tk_radial_mlp_hidden_1_pre_bf16_layer[layer_index],
                                       &tk_radial_mlp_hidden_2_pre_bf16_layer[layer_index]}) {
                if (*p) cudaFree(*p);
                *p = nullptr;
            }
        }
        for (float *&p : tk_precomputed_w3_effective_layer) {
            if (p) cudaFree(p);
            p = nullptr;
        }
        for (float *&p : tk_precomputed_w3_sym_total_layer) {
            if (p) cudaFree(p);
            p = nullptr;
        }
        for (float *&p : tk_precomputed_w3_partsym_layer) {
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
            float *w3_eff = mace_tk_precompute_w3_effective(
                arch.n_elements, arch.n_channels, symmetric_contraction_path_count_max,
                L.u_matrix_3, L.weights_max, stream);
            S.tk_precomputed_w3_effective_layer[layer_index] = w3_eff;
            // Transform #1 — fold all 6 (i, j, k) permutations of W3_eff into
            // the canonical `(i ≤ j ≤ k)` representative cell.  The forward
            // kernel reads from this buffer and iterates 816 canonical triples
            // instead of 4096 ordered triples.
            S.tk_precomputed_w3_sym_total_layer[layer_index] = mace_tk_precompute_w3_sym_total(
                arch.n_elements, arch.n_channels, w3_eff, stream);
            // Transform #1 backward — precompute W3_partsym_sym[Z, m, a, b, c]
            // (canonical a ≤ b) so the adjoint kernel runs as 16 length-136
            // quadratic forms instead of a 16³ × 3 cubic-write structure.
            S.tk_precomputed_w3_partsym_layer[layer_index] = mace_tk_precompute_w3_partsym(
                arch.n_elements, arch.n_channels, w3_eff, stream);
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
        std::uint32_t const tp_dim = mace_tensor_product_path_count_k * arch.n_channels;
#if !defined(SMALLFIELD_USE_THUNDERKITTENS)
        // Fp32 reference path: 4 matmul + 3 SiLU launches.  Under the TK
        // build all four layers are folded into one persistent kernel below;
        // the fp32 layers 0/1/2 here would be writing dead data (the TK
        // kernel reads radial_basis directly), so we skip them.
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
#endif
        // Layer 3 (final, no act): 64 → 4*c
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
        // TK-fused radial MLP — one launch instead of 7 (4 matmul + 3 SiLU
        // norm2mom).  Reads radial_basis fp32 directly; the kernel handles
        // the per-layer (1/sqrt(h_in)) scale + normalize2mom(silu) epilogue
        // internally and emits bf16 tp_weights.  We then cast bf16→fp32 and
        // apply the cutoff multiply in fp32 on the existing buffer so the
        // downstream conv_tp consumer (C2 fused TK or fp32 reference)
        // sees the same fp32 tp_weights buffer regardless of which radial
        // MLP path produced it.
        // Phase B6: pass per-layer pre-activation buffers so the backward can
        // skip the fp32 forward rerun.  Kernel writes bf16 [n_edges, 64] for
        // each non-null layer slot; backward casts to fp32 once and consumes
        // via the existing radial-MLP layer adjoint.
        launch_mace_tk_radial_mlp(batch.n_edges, arch.n_radial,
                                  S.radial_basis,
                                  L.radial_mlp_w0, L.radial_mlp_w1,
                                  L.radial_mlp_w2, L.radial_mlp_w3,
                                  S.tk_tensor_product_weights_per_edge_bf16, stream,
                                  S.tk_radial_mlp_hidden_0_pre_bf16_layer[layer_index],
                                  S.tk_radial_mlp_hidden_1_pre_bf16_layer[layer_index],
                                  S.tk_radial_mlp_hidden_2_pre_bf16_layer[layer_index]);
        launch_mace_bf16_to_fp32(batch.n_edges * tp_dim,
                                 S.tk_tensor_product_weights_per_edge_bf16,
                                 S.tensor_product_weights_per_edge, stream);
#else
        launch_mace_matmul_rowmajor(batch.n_edges, tp_dim, radial_mlp_hidden_dim,
                                    S.radial_mlp_hidden_2, L.radial_mlp_w3,
                                    1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
                                    S.tensor_product_weights_per_edge, stream);
#endif
        // NOTE: MACE-OFF's RadialEmbeddingBlock returns `radial * cutoff, None` (no
        // separate cutoff returned).  The cutoff envelope is therefore already
        // baked into our `S.radial_basis` (kernel multiplies by `env` per edge).
        // The interaction block then does `if cutoff is not None: tp_weights *=
        // cutoff` — and cutoff IS None here, so the multiply is a no-op.  Earlier
        // wiring incorrectly applied the cutoff again here, which double-counted
        // it (forward off by `env` per edge, backward correspondingly off).
        //
        // Per-layer save: the conv_tp+scatter backward needs THIS layer's tp_weights
        // to compute grad_sph_harm and grad_node_l0 (formula uses tp_weights × node_l0
        // × upstream); the shared `S.tensor_product_weights_per_edge` buffer gets
        // overwritten by the next forward layer, so layer-0's reverse pass must read
        // back its own snapshot.  We reuse the legacy `pre_cutoff_layer` slot — with
        // cutoff folded into `radial_basis` upstream, the post-MLP value IS the
        // tp_weights value used by conv_tp.
        if (S.tensor_product_weights_pre_cutoff_layer[layer_index] != nullptr) {
            cudaMemcpyAsync(S.tensor_product_weights_pre_cutoff_layer[layer_index],
                            S.tensor_product_weights_per_edge,
                            batch.n_edges * tp_dim * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

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
        // Returns false if the channel count doesn't match the templated
        // kernel — in that case fall back to the fp32 reference path
        // (medium 128 ch and large 224 ch fall here until Transform #4 lands).
        bool const tk_post_tp_ok = launch_mace_tk_per_irrep_linear(
            batch.n_atoms, arch.n_channels, arch.n_channels,
            /*active_irrep_block_count*/4u,
            L.linear_post_tp_w,
            S.node_features_scratch,
            S.tk_node_features_post_linear_bf16, stream);
        if (tk_post_tp_ok) {
            launch_mace_bf16_to_fp32(batch.n_atoms * arch.n_channels * irrep_dim,
                                     S.tk_node_features_post_linear_bf16,
                                     S.node_features, stream);
        } else {
            launch_mace_linear_per_irrep(batch.n_atoms, arch.n_channels, arch.n_channels,
                                          irrep_dim, /*active_irrep_block_count*/4u,
                                          L.linear_post_tp_w, S.node_features_scratch,
                                          S.node_features, stream);
        }
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
        // Forward kernel reads the original ordered W3_eff — Transform #1
        // (canonical triples) was found NOT to help this kernel because the
        // single-accumulator dependency chain was already decoupled into 4
        // chains by the i-unroll, and the kernel is not FMA-count-bound.
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
        //
        //     The kernel accumulates with `+=`; pre-zero so stale grad_node_features
        //     from the previous layer's reverse pass (its layer-input gradient,
        //     written at steps 3m/3o then handed to inject_l0 backward at 3n) does
        //     NOT bleed into this layer's `grad inter_k_out`.  inter_k_out flows
        //     only into symmetric_contractions; there is no other adjoint path
        //     into this buffer at this point in the chain.
        cudaMemsetAsync(S.grad_node_features, 0,
                        batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float), stream);
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
        // TK D4 — Transform #1: consumes `W3_partsym_sym[Z, m, α, β, c]` (canonical
        // α ≤ β cells) so the kernel runs as 16 length-136 quadratic forms with
        // a shared 136-entry X·X cache.  The legacy W3_eff buffer is still passed
        // for ABI parity but the kernel ignores it under Transform #1.
        launch_mace_tk_symmetric_contraction_force_adjoint(
            batch.n_atoms, arch.n_channels, arch.n_elements,
            batch.atom_element,
            S.node_features_post_residual_layer[layer_index],
            L.u_matrix_1, L.u_matrix_2,
            L.weights_lo_0, L.weights_lo_1,
            S.tk_precomputed_w3_effective_layer[layer_index],
            S.tk_precomputed_w3_partsym_layer[layer_index],
            S.grad_node_scalar_invariants,
            S.grad_node_features, stream);
#else
        launch_mace_symmetric_contraction_force_adjoint(
            batch.n_atoms, arch.n_channels, irrep_dim, arch.n_elements,
            batch.atom_element,
            S.node_features_post_residual_layer[layer_index],
            L.u_matrix_1, L.u_matrix_2, L.u_matrix_3,
            L.weights_max, L.weights_lo_0, L.weights_lo_1,
            S.grad_node_scalar_invariants,
            S.grad_node_features, stream);
#endif

        // 3e. (layer 0) Element-conditioned residual on the message path:
        //     `node_features_post_skip_tp = skip_tp(node_features_post_avg, atom_element)`.
        //     Reverse: grad_node_features_scratch = skip_tp_backward(grad_node_features).
        //     For layer-N (>= 1) skip_tp on the message path is identity, so
        //     just copy the gradient through.
        //     The kernel uses `+=`; pre-zero so layer-1 iteration's leftovers
        //     in grad_node_features_scratch (linear_post_tp_bwd output, written
        //     by the preceding L1 step 3g) do NOT bleed into this layer's
        //     skip_tp pre-image gradient.
        if (layer_index == 0u) {
            cudaMemsetAsync(S.grad_node_features_scratch, 0,
                            batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float), stream);
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
            // TK D3: element-conditioned residual backward, bf16 WGMMA.
            launch_mace_tk_element_conditioned_residual_force_adjoint(
                batch.n_atoms, arch.n_channels, arch.n_elements, arch.n_channels,
                /*active_irrep_block_count*/4u,
                batch.atom_element, L.skip_tp_w,
                S.grad_node_features,                 // upstream (post-skip_tp grad)
                S.grad_node_features_scratch,         // pre-skip_tp grad (fp32 +=)
                stream);
#else
            launch_mace_element_conditioned_residual_force_adjoint(
                batch.n_atoms, arch.n_channels, arch.n_elements, arch.n_channels,
                irrep_dim, /*active_irrep_block_count*/4u,
                batch.atom_element, L.skip_tp_w,
                S.grad_node_features,                 // upstream (post-skip_tp grad)
                S.grad_node_features_scratch,         // pre-skip_tp grad
                stream);
#endif
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
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
        // TK D1 path: kernel emits bf16 with overwrite semantics, so the
        // explicit memset is unnecessary.  Reuse `tk_node_features_post_linear_bf16`
        // as the bf16 staging slot (same shape, unused during the backward
        // pass after its forward write was already cast to fp32).  The cast
        // back to fp32 also writes with overwrite semantics, so step 3h's
        // consumer sees a clean grad_node_features.
        launch_mace_tk_per_irrep_linear_force_adjoint(
            batch.n_atoms, arch.n_channels, arch.n_channels,
            /*active_irrep_block_count*/4u,
            L.linear_post_tp_w,
            S.grad_node_features_scratch,
            S.tk_node_features_post_linear_bf16,
            stream);
        launch_mace_bf16_to_fp32(
            batch.n_atoms * arch.n_channels * irrep_dim,
            S.tk_node_features_post_linear_bf16,
            S.grad_node_features, stream);
#else
        // fp32 reference path uses += semantics; zero the destination first.
        cudaMemsetAsync(S.grad_node_features, 0,
                        batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float), stream);
        launch_mace_linear_per_irrep_force_adjoint(
            batch.n_atoms, arch.n_channels, arch.n_channels,
            irrep_dim, /*active_irrep_block_count*/4u,
            L.linear_post_tp_w,
            S.grad_node_features_scratch,
            S.grad_node_features, stream);
#endif

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
        // Use the per-layer tp_weights snapshot — `S.tensor_product_weights_per_edge`
        // is overwritten by layer 1's forward, so layer 0's reverse pass needs its own
        // saved tp_weights from the forward pass.
        float const *tp_weights_for_this_layer =
            S.tensor_product_weights_pre_cutoff_layer[layer_index] != nullptr
            ? S.tensor_product_weights_pre_cutoff_layer[layer_index]
            : S.tensor_product_weights_per_edge;
#if defined(SMALLFIELD_USE_THUNDERKITTENS)
        // TK D2: same three-output backward, bf16 inputs through the kernel,
        // fp32 atomicAdd accumulators on the output side (per the f56ca64
        // smoke-test report — bf16 atomics are non-commutative on Hopper).
        launch_mace_tk_tensor_product_and_scatter_force_adjoint(
            batch.n_atoms, batch.n_edges, arch.n_channels,
            batch.edge_source_atom, batch.edge_target_atom,
            batch.receiver_edge_segment_start,
            batch.receiver_edge_segment_end,
            S.node_features_l0_only_layer[layer_index],
            S.sph_harm,
            tp_weights_for_this_layer,
            S.grad_node_features,
            S.grad_node_features_l0_only,
            S.grad_tp_weights,
            S.grad_sph_harm,
            stream);
#else
        launch_mace_tensor_product_and_scatter_force_adjoint(
            batch.n_atoms, batch.n_edges, arch.n_channels,
            batch.edge_source_atom, batch.edge_target_atom,
            S.node_features_l0_only_layer[layer_index],
            S.sph_harm,
            tp_weights_for_this_layer,                // per-layer tp_weights (post-MLP, no double-cutoff)
            S.grad_node_features,                     // upstream: per-atom (target-indexed)
            S.grad_node_features_l0_only,             // atomicAdd accumulator
            S.grad_tp_weights,                        // overwritten
            S.grad_sph_harm                           // atomicAdd accumulator
            , stream);
#endif

        // 3j. NO apply_cutoff backward — the forward folds the cutoff envelope
        //   directly into `radial_basis = b_n(r) * env(r)` (matching MACE-OFF's
        //   RadialEmbeddingBlock that returns `radial * cutoff, None`).  The
        //   downstream radial-basis backward (step 5 below) computes
        //   `d rb[n] / d r = dbn_dr * env + b_n * denv_dr` and routes both terms
        //   into dpositions, so the cutoff-envelope derivative chain is already
        //   complete.  Earlier wiring multiplied tp_weights by env again here
        //   and added a separate apply_cutoff backward + grad_env_to_positions
        //   step — both stale.

        // 3k. radial_mlp adjoint: grad_tp_weights → grad_radial_basis.
        //     Backward through the 4-layer MLP with normalize2mom(silu)
        //     epilogues.  Each interaction layer has its OWN radial MLP weights.
        //     Phase B6: instead of re-running the fp32 forward MLP to populate
        //     `_pre` activations.  When the TK backward path is enabled we feed
        //     the saved bf16 pre-acts directly into the TK kernel, avoiding the
        //     three bf16→fp32 cast launches that the fp32 reference path
        //     required (Phase E1).  Layer 0 backward stays on the fp32 path
        //     because n_in=8 doesn't tile well to WGMMA N=64.
        bool tk_radial_backward_ok = false;
#       if defined(SMALLFIELD_USE_THUNDERKITTENS)
        {
            float const scale_h = 1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim));
            // Layer 3 backward (no activation): grad_h2[N, 64] = scale * grad_out[N, 384] @ W3[64, 384]^T.
            bool ok3 = launch_mace_tk_radial_mlp_layer_force_adjoint(
                batch.n_edges, radial_mlp_hidden_dim, tp_dim, scale_h,
                /*apply_silu*/false,
                S.grad_tp_weights, /*pre_act_bf16*/nullptr, L.radial_mlp_w3,
                S.edge_messages, stream);
            // Layer 2 backward (with silu): folds silu·n2m'(pre_act_2) on the way in.
            bool ok2 = launch_mace_tk_radial_mlp_layer_force_adjoint(
                batch.n_edges, radial_mlp_hidden_dim, radial_mlp_hidden_dim, scale_h,
                /*apply_silu*/true,
                S.edge_messages, S.tk_radial_mlp_hidden_2_pre_bf16_layer[layer_index],
                L.radial_mlp_w2, S.grad_node_features, stream);
            // Layer 1 backward (with silu).
            bool ok1 = launch_mace_tk_radial_mlp_layer_force_adjoint(
                batch.n_edges, radial_mlp_hidden_dim, radial_mlp_hidden_dim, scale_h,
                /*apply_silu*/true,
                S.grad_node_features, S.tk_radial_mlp_hidden_1_pre_bf16_layer[layer_index],
                L.radial_mlp_w1, S.edge_messages, stream);
            tk_radial_backward_ok = ok1 && ok2 && ok3;
        }
#       endif
        if (!tk_radial_backward_ok) {
            // Fallback: fp32 reference path needs the pre-acts as fp32, so cast
            // first.  This branch only fires under SMALLFIELD_USE_THUNDERKITTENS=OFF
            // or for shapes the TK kernel doesn't cover.
            launch_mace_bf16_to_fp32(batch.n_edges * radial_mlp_hidden_dim,
                                      S.tk_radial_mlp_hidden_0_pre_bf16_layer[layer_index],
                                      S.radial_mlp_hidden_0_pre, stream);
            launch_mace_bf16_to_fp32(batch.n_edges * radial_mlp_hidden_dim,
                                      S.tk_radial_mlp_hidden_1_pre_bf16_layer[layer_index],
                                      S.radial_mlp_hidden_1_pre, stream);
            launch_mace_bf16_to_fp32(batch.n_edges * radial_mlp_hidden_dim,
                                      S.tk_radial_mlp_hidden_2_pre_bf16_layer[layer_index],
                                      S.radial_mlp_hidden_2_pre, stream);
            launch_mace_radial_mlp_layer_force_adjoint(
                batch.n_edges, radial_mlp_hidden_dim, tp_dim,
                1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
                /*apply_silu_n2m_inverse*/false,
                /*accumulate*/false,
                /*activation*/nullptr,
                L.radial_mlp_w3, S.grad_tp_weights,
                S.edge_messages, stream);
            launch_mace_radial_mlp_layer_force_adjoint(
                batch.n_edges, radial_mlp_hidden_dim, radial_mlp_hidden_dim,
                1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
                /*apply_silu_n2m_inverse*/true,
                /*accumulate*/false,
                S.radial_mlp_hidden_2_pre, L.radial_mlp_w2, S.edge_messages,
                S.grad_node_features, stream,
                /*activation_is_pre_act*/true);
            launch_mace_radial_mlp_layer_force_adjoint(
                batch.n_edges, radial_mlp_hidden_dim, radial_mlp_hidden_dim,
                1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
                /*apply_silu_n2m_inverse*/true,
                /*accumulate*/false,
                S.radial_mlp_hidden_1_pre, L.radial_mlp_w1, S.grad_node_features,
                S.edge_messages, stream,
                /*activation_is_pre_act*/true);
        }
        // Layer 0 backward (with epilogue): grad_radial_basis += contribution.
        // Both interaction layers' radial MLPs share the same radial_basis input,
        // so we ACCUMULATE to preserve the previous layer's contribution.  The
        // n_in=8 shape doesn't fit the TK WGMMA N=64 tile cleanly, so this stays
        // on the fp32 reference path — but we only need the layer-0 pre-act as
        // fp32, not all three.
        launch_mace_bf16_to_fp32(batch.n_edges * radial_mlp_hidden_dim,
                                  S.tk_radial_mlp_hidden_0_pre_bf16_layer[layer_index],
                                  S.radial_mlp_hidden_0_pre, stream);
        launch_mace_radial_mlp_layer_force_adjoint(
            batch.n_edges, arch.n_radial, radial_mlp_hidden_dim,
            1.0f / sqrtf(static_cast<float>(arch.n_radial)),
            /*apply_silu_n2m_inverse*/true,
            /*accumulate*/true,
            S.radial_mlp_hidden_0_pre, L.radial_mlp_w0, S.edge_messages,
            S.grad_radial_basis, stream,
            /*activation_is_pre_act*/true);

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

/**
 *  @brief Diagnostic snapshot points captured during a debug run of
 *         `mace_inference_run_force_adjoint`.  Each pointer, if non-null,
 *         receives a per-step copy of the relevant grad buffer immediately
 *         AFTER the corresponding step in the chain.  Two layer slots; if
 *         a buffer for layer 1 is provided but the forward only ran one
 *         interaction layer, the buffer is left untouched.
 *
 *  Naming follows the GOLDEN tensor names emitted by `mace_export.py`:
 *    - `grad_node_scalar_invariants_layer_k`  (after step 3a, readout backward)
 *      = grad of `prod_k_out` (post-product L=0 features).  Lives in
 *      `S.grad_node_features_l0_only` at that point in the chain.
 *    - `grad_node_features_layer_k`           (after step 3d, symcontraction backward)
 *      = grad of `inter_k_out` (post-interaction full irreps).  Lives in
 *      `S.grad_node_features` at that point.
 *  Plus auxiliary checkpoints for downstream localisation.
 */
struct mace_force_adjoint_debug_snapshots_t {
    // Layer-N first (reverse chain order):
    float *grad_node_scalar_invariants_after_readout_layer1 = nullptr;  // [n_atoms, n_channels]
    float *grad_node_scalar_invariants_after_product_layer1 = nullptr;  // [n_atoms, n_channels] — post 3c
    float *grad_node_features_after_symcontraction_layer1   = nullptr;  // [n_atoms, n_channels, 16] — post 3d
    float *grad_node_features_after_skip_tp_layer1          = nullptr;  // [n_atoms, n_channels, 16] — post 3e
    float *grad_node_features_after_linear_post_tp_layer1   = nullptr;  // [n_atoms, n_channels, 16] — post 3g
    float *grad_edge_messages_after_tp_scatter_layer1       = nullptr;  // [n_edges, n_channels, 16] — receiver-gathered
    float *grad_node_features_layer_input_layer1            = nullptr;  // [n_atoms, n_channels, 16] — end-of-iteration

    // Layer-0 (last to execute in reverse chain):
    float *grad_node_scalar_invariants_after_readout_layer0 = nullptr;
    float *grad_node_scalar_invariants_after_product_layer0 = nullptr;
    float *grad_node_features_after_symcontraction_layer0   = nullptr;
    float *grad_node_features_after_skip_tp_layer0          = nullptr;
    float *grad_node_features_after_linear_post_tp_layer0   = nullptr;
    float *grad_edge_messages_after_tp_scatter_layer0       = nullptr;
    float *grad_node_features_layer_input_layer0            = nullptr;

    // Final downstream gradients (already accessible via S.* but mirrored for parity):
    float *grad_radial_basis = nullptr;  // [n_edges, n_radial]
    float *grad_sph_harm     = nullptr;  // [n_edges, 16]
};

/**
 *  @brief Debug variant of `mace_inference_run_force_adjoint` that copies
 *         intermediate grad-buffer state into the caller-provided snapshot
 *         pointers.  Chain logic is byte-identical to the production
 *         function — this variant only adds optional `cudaMemcpyAsync`
 *         calls at the documented checkpoints.  Use to localise which
 *         step in the reverse chain first deviates from a numerical /
 *         autograd reference.
 */
inline void mace_inference_run_force_adjoint_debug(
    types::mace_arch_t const &arch,
    types::mace_weights_device_t const &W,
    types::mace_batch_t const &batch,
    mace_scratch_t &S,
    mace_force_adjoint_debug_snapshots_t const &snap,
    cudaStream_t stream
) {
    constexpr std::uint32_t irrep_dim = 16u;
    constexpr std::uint32_t readout_mlp_hidden_dim = 16u;
    constexpr std::uint32_t radial_mlp_hidden_dim = 64u;
    std::uint32_t const tp_dim = mace_tensor_product_path_count_k * arch.n_channels;
    float const avg_num_neighbors = arch.avg_num_neighbors;

    auto snap_copy = [&](float *dst, float const *src, std::size_t n_floats) {
        if (dst == nullptr) return;
        cudaMemcpyAsync(dst, src, n_floats * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    };

    // 1. Initialise gradient buffers.
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

    launch_mace_per_graph_energy_force_adjoint(
        batch.n_atoms, arch.scale, batch.graph_of_atom,
        S.grad_per_graph_energy, S.grad_node_energy, stream);

    std::uint32_t const num_layers = (arch.num_interactions < 2u) ? arch.num_interactions : 2u;
    for (std::int32_t layer_signed = static_cast<std::int32_t>(num_layers) - 1; layer_signed >= 0; --layer_signed) {
        std::uint32_t const layer_index = static_cast<std::uint32_t>(layer_signed);
        auto const &L = W.layers[layer_index];

        // 3a. Readout backward.
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
        // SNAPSHOT: post-3a — grad_node_scalar_invariants_layer_k golden mate.
        snap_copy(layer_index == 1u ? snap.grad_node_scalar_invariants_after_readout_layer1
                                    : snap.grad_node_scalar_invariants_after_readout_layer0,
                  S.grad_node_features_l0_only,
                  static_cast<std::size_t>(batch.n_atoms) * arch.n_channels);

        // 3b. layer-N residual backward.
        if (layer_index >= 1u) {
            cudaMemsetAsync(S.grad_layer_residual_l0_scratch, 0,
                            batch.n_atoms * arch.n_channels * sizeof(float), stream);
            launch_mace_element_conditioned_residual_force_adjoint(
                batch.n_atoms, arch.n_channels, arch.n_elements, arch.n_channels,
                /*irrep_dim*/1u, /*active_irrep_block_count*/1u,
                batch.atom_element, L.skip_tp_w,
                S.grad_node_features_l0_only,
                S.grad_layer_residual_l0_scratch,
                stream);
        }

        // 3c. product_linear adjoint.
        cudaMemsetAsync(S.grad_node_scalar_invariants, 0,
                        batch.n_atoms * arch.n_channels * sizeof(float), stream);
        launch_mace_linear_per_irrep_force_adjoint(
            batch.n_atoms, arch.n_channels, arch.n_channels,
            /*irrep_dim*/1u, /*active_irrep_block_count*/1u,
            L.product_linear_w,
            S.grad_node_features_l0_only,
            S.grad_node_scalar_invariants, stream);
        // SNAPSHOT: post-3c (pre-product-linear scalar grad).
        snap_copy(layer_index == 1u ? snap.grad_node_scalar_invariants_after_product_layer1
                                    : snap.grad_node_scalar_invariants_after_product_layer0,
                  S.grad_node_scalar_invariants,
                  static_cast<std::size_t>(batch.n_atoms) * arch.n_channels);

        // 3d. Symmetric contraction adjoint.
        cudaMemsetAsync(S.grad_node_features, 0,
                        batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float), stream);
        launch_mace_symmetric_contraction_force_adjoint(
            batch.n_atoms, arch.n_channels, irrep_dim, arch.n_elements,
            batch.atom_element,
            S.node_features_post_residual_layer[layer_index],
            L.u_matrix_1, L.u_matrix_2, L.u_matrix_3,
            L.weights_max, L.weights_lo_0, L.weights_lo_1,
            S.grad_node_scalar_invariants,
            S.grad_node_features, stream);
        // SNAPSHOT: post-3d — grad_node_features_layer_k golden mate.
        snap_copy(layer_index == 1u ? snap.grad_node_features_after_symcontraction_layer1
                                    : snap.grad_node_features_after_symcontraction_layer0,
                  S.grad_node_features,
                  static_cast<std::size_t>(batch.n_atoms) * arch.n_channels * irrep_dim);

        // 3e. message-path skip_tp backward.
        if (layer_index == 0u) {
            launch_mace_element_conditioned_residual_force_adjoint(
                batch.n_atoms, arch.n_channels, arch.n_elements, arch.n_channels,
                irrep_dim, /*active_irrep_block_count*/4u,
                batch.atom_element, L.skip_tp_w,
                S.grad_node_features,
                S.grad_node_features_scratch,
                stream);
        } else {
            cudaMemcpyAsync(S.grad_node_features_scratch, S.grad_node_features,
                            batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }
        // SNAPSHOT: post-3e — pre-skip_tp grad (this is the message-path gradient).
        snap_copy(layer_index == 1u ? snap.grad_node_features_after_skip_tp_layer1
                                    : snap.grad_node_features_after_skip_tp_layer0,
                  S.grad_node_features_scratch,
                  static_cast<std::size_t>(batch.n_atoms) * arch.n_channels * irrep_dim);

        // 3f. /avg_num_neighbors backward.
        if (avg_num_neighbors != 1.0f) {
            launch_mace_scale_force_adjoint(
                batch.n_atoms * arch.n_channels * irrep_dim,
                1.0f / avg_num_neighbors, S.grad_node_features_scratch, stream);
        }

        // 3g. linear_post_tp adjoint.
        cudaMemsetAsync(S.grad_node_features, 0,
                        batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float), stream);
        launch_mace_linear_per_irrep_force_adjoint(
            batch.n_atoms, arch.n_channels, arch.n_channels,
            irrep_dim, /*active_irrep_block_count*/4u,
            L.linear_post_tp_w,
            S.grad_node_features_scratch,
            S.grad_node_features, stream);
        // SNAPSHOT: post-3g — pre-linear_post_tp grad (= grad of post-scatter messages on the receiver atom).
        snap_copy(layer_index == 1u ? snap.grad_node_features_after_linear_post_tp_layer1
                                    : snap.grad_node_features_after_linear_post_tp_layer0,
                  S.grad_node_features,
                  static_cast<std::size_t>(batch.n_atoms) * arch.n_channels * irrep_dim);

        // 3h. Fused tensor-product + scatter adjoint.
        cudaMemsetAsync(S.grad_node_features_l0_only, 0,
                        batch.n_atoms * arch.n_channels * sizeof(float), stream);
        // SNAPSHOT: BEFORE the conv_tp+scatter backward, capture the per-atom
        //   `grad_node_features` and gather it onto edges to compare against the
        //   `grad_edge_messages_layer_k` golden (which is exactly that gather).
        if (auto *dst = (layer_index == 1u ? snap.grad_edge_messages_after_tp_scatter_layer1
                                           : snap.grad_edge_messages_after_tp_scatter_layer0)) {
            launch_mace_scatter_force_adjoint(
                batch.n_edges, arch.n_channels * irrep_dim,
                batch.edge_target_atom, S.grad_node_features, dst, stream);
        }
        // Use the per-layer tp_weights snapshot.
        float const *tp_weights_for_this_layer_dbg =
            S.tensor_product_weights_pre_cutoff_layer[layer_index] != nullptr
            ? S.tensor_product_weights_pre_cutoff_layer[layer_index]
            : S.tensor_product_weights_per_edge;
        launch_mace_tensor_product_and_scatter_force_adjoint(
            batch.n_atoms, batch.n_edges, arch.n_channels,
            batch.edge_source_atom, batch.edge_target_atom,
            S.node_features_l0_only_layer[layer_index],
            S.sph_harm,
            tp_weights_for_this_layer_dbg,
            S.grad_node_features,
            S.grad_node_features_l0_only,
            S.grad_tp_weights,
            S.grad_sph_harm
            , stream);

        // 3j. NO apply_cutoff backward — see production-variant comment above.

        // 3k. radial_mlp adjoint — see production-variant comment.
        launch_mace_matmul_rowmajor(batch.n_edges, radial_mlp_hidden_dim, arch.n_radial,
                                    S.radial_basis, L.radial_mlp_w0,
                                    1.0f / sqrtf(static_cast<float>(arch.n_radial)),
                                    S.radial_mlp_hidden_0_pre, stream);
        launch_mace_silu_norm2mom(batch.n_edges * radial_mlp_hidden_dim,
                                   S.radial_mlp_hidden_0_pre, S.radial_mlp_hidden_0, stream);
        launch_mace_matmul_rowmajor(batch.n_edges, radial_mlp_hidden_dim, radial_mlp_hidden_dim,
                                    S.radial_mlp_hidden_0, L.radial_mlp_w1,
                                    1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
                                    S.radial_mlp_hidden_1_pre, stream);
        launch_mace_silu_norm2mom(batch.n_edges * radial_mlp_hidden_dim,
                                   S.radial_mlp_hidden_1_pre, S.radial_mlp_hidden_1, stream);
        launch_mace_matmul_rowmajor(batch.n_edges, radial_mlp_hidden_dim, radial_mlp_hidden_dim,
                                    S.radial_mlp_hidden_1, L.radial_mlp_w2,
                                    1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
                                    S.radial_mlp_hidden_2_pre, stream);
        launch_mace_silu_norm2mom(batch.n_edges * radial_mlp_hidden_dim,
                                   S.radial_mlp_hidden_2_pre, S.radial_mlp_hidden_2, stream);
        launch_mace_radial_mlp_layer_force_adjoint(
            batch.n_edges, radial_mlp_hidden_dim, tp_dim,
            1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
            /*apply_silu_n2m_inverse*/false,
            /*accumulate*/false,
            /*activation*/nullptr,
            L.radial_mlp_w3, S.grad_tp_weights,
            S.edge_messages, stream);
        launch_mace_radial_mlp_layer_force_adjoint(
            batch.n_edges, radial_mlp_hidden_dim, radial_mlp_hidden_dim,
            1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
            /*apply_silu_n2m_inverse*/true,
            /*accumulate*/false,
            S.radial_mlp_hidden_2_pre, L.radial_mlp_w2, S.edge_messages,
            S.grad_node_features, stream,
            /*activation_is_pre_act*/true);
        launch_mace_radial_mlp_layer_force_adjoint(
            batch.n_edges, radial_mlp_hidden_dim, radial_mlp_hidden_dim,
            1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
            /*apply_silu_n2m_inverse*/true,
            /*accumulate*/false,
            S.radial_mlp_hidden_1_pre, L.radial_mlp_w1, S.grad_node_features,
            S.edge_messages, stream,
            /*activation_is_pre_act*/true);
        launch_mace_radial_mlp_layer_force_adjoint(
            batch.n_edges, arch.n_radial, radial_mlp_hidden_dim,
            1.0f / sqrtf(static_cast<float>(arch.n_radial)),
            /*apply_silu_n2m_inverse*/true,
            /*accumulate*/true,
            S.radial_mlp_hidden_0_pre, L.radial_mlp_w0, S.edge_messages,
            S.grad_radial_basis, stream,
            /*activation_is_pre_act*/true);

        // 3l. linear_up adjoint.
        cudaMemsetAsync(S.grad_node_scalar_invariants, 0,
                        batch.n_atoms * arch.n_channels * sizeof(float), stream);
        launch_mace_linear_per_irrep_force_adjoint(
            batch.n_atoms, arch.n_channels, arch.n_channels,
            /*irrep_dim*/1u, /*active_irrep_block_count*/1u,
            L.linear_up_w,
            S.grad_node_features_l0_only,
            S.grad_node_scalar_invariants, stream);

        // 3m. extract_l0 adjoint.
        cudaMemsetAsync(S.grad_node_features, 0,
                        batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float), stream);
        launch_mace_extract_l0_irrep_block_force_adjoint(
            batch.n_atoms, arch.n_channels, irrep_dim,
            S.grad_node_scalar_invariants, S.grad_node_features, stream);

        // 3o. layer-N residual merge.
        if (layer_index >= 1u) {
            launch_mace_extract_l0_irrep_block_force_adjoint(
                batch.n_atoms, arch.n_channels, irrep_dim,
                S.grad_layer_residual_l0_scratch, S.grad_node_features, stream);
        }
        // SNAPSHOT: end-of-iteration — `grad_node_features` is now the gradient
        //   of the loss wrt the LAYER-INPUT `node_features_input_layer[k]`.
        snap_copy(layer_index == 1u ? snap.grad_node_features_layer_input_layer1
                                    : snap.grad_node_features_layer_input_layer0,
                  S.grad_node_features,
                  static_cast<std::size_t>(batch.n_atoms) * arch.n_channels * irrep_dim);

        // 3n. inject_l0 adjoint hand-off to previous layer.
        if (layer_index >= 1u) {
            cudaMemsetAsync(S.grad_node_features_l0_only, 0,
                            batch.n_atoms * arch.n_channels * sizeof(float), stream);
            launch_mace_inject_l0_irrep_block_force_adjoint(
                batch.n_atoms, arch.n_channels, irrep_dim,
                S.grad_node_features, S.grad_node_features_l0_only, stream);
        }
    }

    // 4. sph_harm adjoint.
    launch_mace_sph_harm_force_adjoint(batch.n_edges, S.r_unit, S.grad_sph_harm,
                                        S.grad_r_unit, stream);

    // SNAPSHOT: grad_sph_harm just before consumed.  Captured AFTER step 4 so
    // that the buffer reflects the value the production chain actually feeds
    // forward — but step 4 only READS grad_sph_harm and writes grad_r_unit, so
    // a copy here is identical to a copy before step 4.
    snap_copy(snap.grad_sph_harm, S.grad_sph_harm,
              static_cast<std::size_t>(batch.n_edges) * irrep_dim);

    // 5. radial_basis × cutoff envelope + edge geometry adjoint.
    launch_mace_radial_basis_and_edge_geometry_force_adjoint(
        arch, batch.n_edges, W.bessel_weights,
        S.r_norm, S.r_unit, batch.edge_source_atom, batch.edge_target_atom,
        S.grad_radial_basis, S.grad_r_unit, S.grad_positions, stream);
    snap_copy(snap.grad_radial_basis, S.grad_radial_basis,
              static_cast<std::size_t>(batch.n_edges) * arch.n_radial);

    // 6. forces[a, k] = -grad_positions[a, k].
    launch_mace_force_negate(batch.n_atoms, S.grad_positions, S.forces, stream);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
