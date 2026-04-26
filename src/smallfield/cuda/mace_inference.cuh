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
        float const s = 1.0f / (1.0f + __expf(-pre_act));
        h_shared[hidden_index] = pre_act * s;
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
    float *node_energy = nullptr;                  // [n_atoms]

    // graph-scoped
    float *energy = nullptr;                       // [n_graphs]

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
        alloc(&node_energy, max_atoms);
        alloc(&energy, max_graphs);
    }

    void free() noexcept {
        for (float **p : {&r_norm, &r_unit, &radial_basis, &cutoff_envelope, &sph_harm,
                          &radial_mlp_hidden_0, &radial_mlp_hidden_1, &radial_mlp_hidden_2,
                          &tensor_product_weights_per_edge, &edge_messages,
                          &node_features, &node_features_scratch,
                          &node_features_l0_only, &node_scalar_invariants, &node_energy, &energy}) {
            if (*p) cudaFree(*p);
            *p = nullptr;
        }
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
        launch_mace_matmul_rowmajor(batch.n_edges, tp_dim, radial_mlp_hidden_dim,
                                    S.radial_mlp_hidden_2, L.radial_mlp_w3,
                                    1.0f / sqrtf(static_cast<float>(radial_mlp_hidden_dim)),
                                    S.tensor_product_weights_per_edge, stream);
        // Multiply per-edge by the cutoff envelope (mirrors `tp_weights *= cutoff` in the live forward).
        launch_mace_apply_cutoff(batch.n_edges, tp_dim, S.cutoff_envelope, S.tensor_product_weights_per_edge, stream);

        // 3c. Tensor product: per-edge messages.
        launch_mace_tensor_product(arch, batch.n_edges, batch.edge_source_atom,
                                    S.node_features_l0_only, S.sph_harm, S.tensor_product_weights_per_edge, S.edge_messages, stream);

        // 3d. Scatter sum into receivers, full feature dim = n_channels * 16.
        launch_mace_scatter_atomic(batch.n_atoms, batch.n_edges,
                                    arch.n_channels * irrep_dim,
                                    batch.edge_target_atom, S.edge_messages, S.node_features_scratch, stream);

        // 3e. linear_post_tp: per-irrep channel mix on all 4 L blocks.
        launch_mace_linear_per_irrep(batch.n_atoms, arch.n_channels, arch.n_channels,
                                      irrep_dim, /*active_irrep_block_count*/4u,
                                      L.linear_post_tp_w, S.node_features_scratch, S.node_features, stream);

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
            launch_mace_skip_tp(batch.n_atoms, arch.n_channels, arch.n_elements, arch.n_channels,
                                 irrep_dim, active_irrep_block_count, batch.atom_element,
                                 L.skip_tp_w, S.node_features, S.node_features_scratch, stream);
            // Copy back to node_features (or just use scratch as the "current" buffer next).
            cudaMemcpyAsync(S.node_features, S.node_features_scratch,
                            batch.n_atoms * arch.n_channels * irrep_dim * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }

        // 3h. Symmetric contraction: produce per-channel scalar invariants.
        launch_mace_symmetric_contraction(arch, L, batch.n_atoms, batch.atom_element,
                                           S.node_features, S.node_scalar_invariants, stream);

        // 3i. product_linear: channel mix on the scalar invariants (96 → 96).
        launch_mace_linear_per_irrep(batch.n_atoms, arch.n_channels, arch.n_channels,
                                      /*irrep_dim*/1u, /*active_irrep_block_count*/1u,
                                      L.product_linear_w, S.node_scalar_invariants, S.node_features_l0_only, stream);

        // 3j. Inject the post-product L=0 features back into node_features for the next iteration's input.
        {
            constexpr std::uint32_t block_dim_x = 64;
            dim3 const block(block_dim_x, 1, 1);
            dim3 const grid(batch.n_atoms, (arch.n_channels + block_dim_x - 1u) / block_dim_x, 1);
            mace_detail_::mace_inject_l0_irrep_block_kernel<<<grid, block, 0, stream>>>(
                batch.n_atoms, arch.n_channels, irrep_dim, S.node_features_l0_only, S.node_features);
        }

        // 3k. Apply the readout for this layer to S.node_features_l0_only (the per-channel L=0 features).
        if (layer_index == 0u) {
            // Linear readout for the first layer (alpha = n_channels).
            float const ro_scale = 1.0f / sqrtf(static_cast<float>(arch.n_channels));
            mace_detail_::mace_readout_linear_kernel<<<batch.n_atoms, 32u, 0, stream>>>(
                batch.n_atoms, arch.n_channels, ro_scale, S.node_features_l0_only, W.readout_linear_w, S.node_energy);
        } else {
            // Nonlinear MLP readout for the last layer.
            constexpr std::uint32_t hidden_dim = 16u;
            float const sc1 = 1.0f / sqrtf(static_cast<float>(arch.n_channels));
            float const sc2 = 1.0f / sqrtf(static_cast<float>(hidden_dim));
            mace_detail_::mace_readout_nonlinear_mlp_kernel<<<batch.n_atoms, 32u,
                                                          hidden_dim * sizeof(float), stream>>>(
                batch.n_atoms, arch.n_channels, hidden_dim, sc1, sc2, S.node_features_l0_only,
                W.readout_mlp_w0, W.readout_mlp_w1, S.node_energy);
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

} // namespace smallfield::cuda

#endif // __CUDACC__
