/**
 *  @file mace_types.hpp
 *  @brief Type definitions for the MACE-OFF inference module.
 *
 *  MACE-OFF is a graph neural network potential.  Inference walks two
 *  message-passing rounds over the atom graph, with each round computing
 *  per-edge equivariant features (radial × spherical harmonics, then a
 *  tensor product against per-node features) and a per-node residual update.
 *  See `MACE_INFERENCE.md` at the repo root for the algorithmic background.
 *
 *  The shapes below are sized for the MACE-OFF23 family (small/medium/large)
 *  with `L_max = 3` and `correlation_order = 3`.  All sizes that vary between
 *  the three model sizes (channels, radial-MLP hidden, number of interaction
 *  layers) are runtime parameters carried in @ref mace_arch_t.
 *
 *  Naming conventions match the rest of `smallfield::types`:
 *    - struct names end in `_t`
 *    - constants end in `_k`
 *    - all sizes are `std::uint32_t` unless they index something atomic-scale
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <cstdint>

#include <cuda_fp16.h>

namespace smallfield::types {

/** Maximum spherical-harmonic degree exposed by MACE-OFF23. */
constexpr std::uint32_t mace_l_max_k = 3u;
/** Number of irrep blocks for `L_max = 3` (one per L = 0,1,2,3). */
constexpr std::uint32_t mace_n_irreps_k = mace_l_max_k + 1u;
/** Number of (m_l) components summed across all irrep blocks: 1 + 3 + 5 + 7. */
constexpr std::uint32_t mace_irrep_dim_k = 16u;
/** Correlation order of the Atomic Cluster Expansion symmetric contraction. */
constexpr std::uint32_t mace_correlation_order_k = 3u;

/**
 *  @brief Architectural parameters read from the weight file's JSON manifest.
 *
 *  These values are constants once a model size is fixed but vary across
 *  small/medium/large.  Loaded by @ref smallfield::parameters::load_mace_weights.
 */
struct mace_arch_t {
    std::uint32_t n_elements;        ///< number of atomic species supported (10 for MACE-OFF23: H,C,N,O,F,P,S,Cl,Br,I)
    std::uint32_t n_radial;          ///< number of Bessel radial basis functions (8 for MACE-OFF23)
    std::uint32_t n_channels;        ///< feature channels per node (96 / 128 / 192 for small / medium / large)
    std::uint32_t n_irreps_msg;      ///< sum of (2L+1) over message irrep blocks; 16 for L_max=3
    std::uint32_t l_max;             ///< maximum spherical-harmonic degree (3)
    std::uint32_t correlation_order; ///< symmetric-contraction correlation order (3)
    std::uint32_t num_interactions;  ///< number of interaction blocks (2)
    float scale;                     ///< ScaleShift output scale (~1.07 for MACE-OFF23)
    float shift;                     ///< ScaleShift output shift (~0.0)
    float cutoff_p;                  ///< polynomial-cutoff order p (5.0 for MACE-OFF23)
    float r_max;                     ///< edge cutoff radius in Angstrom (4.5 for MACE-OFF23-small)
    float bessel_prefactor;          ///< sqrt(2 / r_max) prefactor on the Bessel basis (~ 0.667)
    float avg_num_neighbors;         ///< per-layer divisor on scattered messages (~15.65 for MACE-OFF23-small)
};

/**
 *  @brief Pointer-only bundle of device-resident weight tensors for one MACE forward.
 *
 *  All tensors live in a single contiguous device allocation; this struct
 *  carries pointers into that allocation.  Layout matches the export tool in
 *  @c python/smallfield/mace_export.py and the loader in
 *  @c smallfield::parameters::mace_weights_t.
 *
 *  Pointers may be `nullptr` if the corresponding tensor is unused for a
 *  given model size (e.g. `readout_mlp_*` is only populated for the last
 *  interaction's nonlinear readout).
 */
struct mace_weights_device_t {
    std::int64_t const *atomic_numbers; ///< [n_elements] supported Z values
    float const *atomic_energies;       ///< [n_elements] per-element baseline energy in eV
    float const *bessel_weights;        ///< [n_radial] Bessel basis function frequencies
    float const *node_embedding;        ///< [n_elements * n_channels] one-hot → channel embedding

    /** One per interaction block. */
    struct layer_t {
        float const *linear_up_w;   ///< [n_channels, n_channels] first channel mixer
        float const *radial_mlp_w0; ///< [n_radial, hidden] radial-MLP layer 0
        float const *radial_mlp_w1; ///< [hidden, hidden] radial-MLP layer 1
        float const *radial_mlp_w2; ///< [hidden, hidden] radial-MLP layer 2
        float const *radial_mlp_w3; ///< [hidden, n_paths*n_channels] radial-MLP layer 3 (output projects to TP weights)
        float const *linear_post_tp_w; ///< post-tensor-product linear (per-irrep channel mix)
        float const *skip_tp_w;        ///< skip-connection tensor product weight
        float const *u_matrix_1;       ///< symmetric-contraction order-1 path tensor [16, n_paths_1]
        float const *u_matrix_2;       ///< symmetric-contraction order-2 path tensor [16, 16, n_paths_2]
        float const *u_matrix_3;       ///< symmetric-contraction order-3 path tensor [16, 16, 16, n_paths_max]
        float const *weights_max;      ///< [n_elements, n_paths_max, n_channels] order-`correlation_order` weights
        float const *weights_lo_0;     ///< [n_elements, n_paths_2, n_channels] order-2 weights
        float const *weights_lo_1;     ///< [n_elements, n_paths_1, n_channels] order-1 weights
        float const *product_linear_w; ///< [n_channels, n_channels] post-contraction linear
    };
    layer_t layers[2]; ///< interaction layers — sized for MACE-OFF23 num_interactions = 2

    float const *readout_linear_w; ///< [n_channels] linear readout for layer 0
    float const *readout_mlp_w0;   ///< [n_channels, hidden] nonlinear readout layer 1, layer N-1 only
    float const *readout_mlp_w1;   ///< [hidden] nonlinear readout layer 2, scalar output
};

/**
 *  @brief Per-batch atom and edge layout for one MACE inference call.
 *
 *  Edges are directed: an undirected bond at runtime becomes two entries here
 *  (one for each direction), matching the convention used by `mace.data.AtomicData`.
 *  `edge_source_atom[e]` is the sender node, `edge_target_atom[e]` is the receiver — messages
 *  scatter into `edge_target_atom`.
 */
struct mace_batch_t {
    std::uint32_t n_atoms;                ///< number of atoms across the whole batch
    std::uint32_t n_edges;                ///< number of directed edges (2× undirected)
    std::uint32_t n_graphs;               ///< number of independent molecules in the batch
    float const *positions;               ///< [n_atoms, 3] atom positions in Angstrom
    std::int32_t const *atom_element;     ///< [n_atoms] index into the atomic_numbers / atomic_energies tables
    std::int32_t const *edge_source_atom; ///< [n_edges] sender index
    std::int32_t const *edge_target_atom; ///< [n_edges] receiver index
    std::int32_t const *graph_of_atom;    ///< [n_atoms] which molecule each atom belongs to (for energy reduction)

    // Optional fp16 view of the positions, populated when the batch comes from
    // the upstream coprocessed-batch flow that stores geometries in fp16.
    // nullptr means the fp32 `positions` field above is the canonical input.
    __half const *positions_fp16 = nullptr;

    // CSR-style per-graph layout — `graph_atom_offset[g]` = first atom index of
    // graph g, `graph_atom_offset[n_graphs]` = total_atom_count = n_atoms.
    // `graph_edge_offset` is the analogous for edges.  When these are nullptr
    // the batch is a single contiguous run (graph 0 has all atoms / edges).
    std::int32_t const *graph_atom_offset = nullptr;
    std::int32_t const *graph_edge_offset = nullptr;

    // Pre-sorted edge list (sorted ascending by edge_target_atom) plus the
    // per-receiver segment ranges for the segmented-scatter kernel that
    // already exists in `mace_scatter_kernel.cuh`.  When nullptr the driver
    // falls back to the atomic-scatter path.
    std::int32_t const *receiver_edge_segment_start = nullptr;
    std::int32_t const *receiver_edge_segment_end   = nullptr;
};

/**
 *  @brief Per-edge scratch tensors produced by the radial + spherical-harmonic preamble.
 *
 *  Lives for one forward pass; reused across both interaction layers because the
 *  edge geometry does not change between layers.  All tensors are device-resident
 *  and sized off `mace_batch_t::n_edges`.
 */
struct mace_edge_features_t {
    float *r_norm;          ///< [n_edges] euclidean distance ‖r_dst − r_src‖
    float *r_unit;          ///< [n_edges, 3] normalized direction (r_dst − r_src) / r_norm
    float *radial_basis;    ///< [n_edges, n_radial] Bessel basis × polynomial cutoff
    float *cutoff_envelope; ///< [n_edges] polynomial cutoff envelope (1 inside r_max, smooth → 0)
    float *sph_harm;        ///< [n_edges, mace_irrep_dim_k] spherical harmonics y_lm(r_unit)
};

/**
 *  @brief Output bundle for one batched MACE inference call.
 *
 *  Energies live in fp32 (load-bearing for downstream ranking and free-energy
 *  calculations); per-atom forces are quantised to fp16 to match SmallField's
 *  geometry-storage convention — fp16 dynamic range is ample for forces
 *  whose magnitudes are typically below 10 eV/Å for stable structures.
 *  The fp32 energy buffer is required even when forces are not requested,
 *  so callers do not have to special-case the energy-only path.
 */
struct mace_results_t {
    float *graph_energy_fp32 = nullptr;       ///< [n_graphs] per-graph total energy in eV
    __half *atom_force_fp16 = nullptr;        ///< [n_atoms, 3] per-atom forces in eV/Å, fp16
};

} // namespace smallfield::types
