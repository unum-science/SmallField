/**
 *  @file csd_torsion_device.cuh
 *  @brief Device-resident CSD torsion table and the warp-per-molecule matcher that reads it.
 *
 *  The CSD torsion preferences compile to 470 stack-only patterns whose matcher, in
 *  `preprocess/csd_torsion_native.hpp`, is a pure function of @ref molecule_descriptor_t.  This
 *  header carries that matcher to the GPU: it flattens the compiled patterns into POD mirrors,
 *  uploads them into one device arena the way `parameters/mmff94_tables_device.cuh` uploads the
 *  MMFF tables, and runs one warp per molecule over the result.
 *
 *  Two members of the host pattern cannot cross.  A @ref recursive_pool_t is a fixed 40-node pool
 *  even though the widest body in the corpus spends seven nodes, and a @ref simple_ring_closure_t
 *  is half a kilobyte that only the ring-closing patterns use.  Both move into shared arenas the
 *  pattern indexes by offset, which is what turns a 2.6 MB table into a 660 KB one, and the
 *  matcher reads a pattern through @ref csd_pattern_view_t so the same body serves both layouts.
 *
 *  Work assignment: a lane owns one bond and walks the pattern list in file order until a pattern
 *  binds, which is exactly the "first pattern claims the bond" rule the host loop implements from
 *  the other side.  The pattern is warp-uniform, so every lane reads the same address and the
 *  table streams through L2 once per warp rather than once per lane.  The ring torsions that
 *  RDKit's `useBasicKnowledge` adds are order-dependent between rings and run on lane 0.
 *
 *  Correctness: identical torsion set per molecule to @ref extract_etk_torsions_native, checked
 *  molecule by molecule in `test/kernels/test_csd_torsion_device.cpp`.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <cstdint>
#include <cstdio> // `std::fprintf`

#include <vector>

#include <cuda_runtime.h>

#include "smallfield/api/batch_pool.hpp"
#include "smallfield/preprocess/csd_torsion_dispatcher.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::cuda {

using smallfield::api::device_byte_buffer_t;
using smallfield::preprocess::bond_excluded_by_ring_fusion;
using smallfield::preprocess::bond_ring_count;
using smallfield::preprocess::build_molecule_gate;
using smallfield::preprocess::csd_compiled_pattern_t;
using smallfield::preprocess::csd_molecule_gate_t;
using smallfield::preprocess::csd_pattern_view_t;
using smallfield::preprocess::find_bond_between;
using smallfield::preprocess::match_pattern_at_bond;
using smallfield::preprocess::no_bond_index_k;
using smallfield::preprocess::recursive_node_t;
using smallfield::preprocess::ring_atom_count;
using smallfield::preprocess::simple_atom_constraint_t;
using smallfield::preprocess::simple_bond_constraint_t;
using smallfield::preprocess::simple_ring_closure_t;
using smallfield::types::etk_torsion_descriptor_t;
using smallfield::types::hybridization_sp2_k;
using smallfield::types::max_bonds_k;
using smallfield::types::molecule_descriptor_t;

/**
 *  Per-thread device stack the matcher needs.  Sub-query evaluation still recurses, bounded by the
 *  compiled corpus rather than by the molecule: a body nests at most @ref max_recursive_depth_k
 *  levels of `$(...)` and walks at most @ref max_recursive_hops_k hops from its root.  Driving the
 *  limit down until the kernel faults puts the peak between 512 and 768 bytes across 39168 PubChem
 *  and Enamine REAL molecules, so the CUDA default of 1024 already suffices and @ref
 *  ensure_csd_torsion_device_stack only has work to do when a caller lowered it.
 */
inline constexpr std::size_t csd_torsion_stack_bytes_k = 1024;

/** Value of @ref csd_device_pattern_t::closure_offset for a pattern that closes no ring. */
inline constexpr std::uint32_t no_closure_offset_k = 0xFFFFFFFFu;

/** Bonds one warp claims per pass; one lane owns one bond. */
inline constexpr std::uint32_t warp_size_k = 32;

/**
 *  POD mirror of one compiled torsion pattern.  Everything the matcher reads is either inline here
 *  or reachable through an offset into the table's node and closure arenas, so the whole struct is
 *  memcpy-able to the device.  The six Fourier coefficients are the host's `signs[i] * V[i]`
 *  products, folded at upload time because the kernel never needs the two factors apart.
 */
struct alignas(8) csd_device_pattern_t {
    simple_atom_constraint_t atom[4];     ///< the labelled atoms `:1` through `:4`
    simple_bond_constraint_t bond[3];     ///< chain bonds; the central `:2`-`:3` bond is index 1
    std::uint64_t required_elements;      ///< bit N set when a match needs element N present
    std::uint32_t pool_offset;            ///< first sub-query node of this pattern in the node arena
    std::uint32_t closure_offset;         ///< ring closure in the closure arena; @ref no_closure_offset_k when absent
    float fourier_coefficients[6];        ///< V₁ through V₆ with the pattern's signs already applied
    std::uint8_t required_hybridizations; ///< bit H set when a match needs hybridization H present
    std::uint8_t requires_aromatic;       ///< 1 when a match needs an aromatic atom
    std::uint8_t padding[6];
};

/** Device-side handle on the uploaded table: the patterns plus the arenas they index. */
struct csd_torsion_table_view_t {
    csd_device_pattern_t const *patterns = nullptr;
    recursive_node_t const *nodes = nullptr;
    simple_ring_closure_t const *closures = nullptr;
    std::uint32_t num_patterns = 0;
};

/** Host staging buffers for the table, laid out exactly as the device arena receives them. */
struct csd_torsion_table_host_t {
    std::vector<csd_device_pattern_t> patterns;
    std::vector<recursive_node_t> nodes;
    std::vector<simple_ring_closure_t> closures;

    /** @brief Total bytes the three arrays occupy on the device, before alignment padding. */
    std::size_t byte_size() const {
        return patterns.size() * sizeof(csd_device_pattern_t) + nodes.size() * sizeof(recursive_node_t) +
               closures.size() * sizeof(simple_ring_closure_t);
    }
};

/**
 *  @brief Flatten the process-wide compiled pattern corpus into the POD staging buffers.
 *
 *  Node indices stay pattern-local — a disjunct's `recursive_root` keeps indexing its own pattern's
 *  nodes — because the kernel hands the matcher a node pointer already biased by the pattern's
 *  @ref csd_device_pattern_t::pool_offset.
 *
 *  @param[out] out Staging buffers; any prior contents are dropped.
 */
inline void build_csd_torsion_table(csd_torsion_table_host_t &out) {
    auto const &compiled = smallfield::preprocess::csd_patterns();
    out.patterns.clear();
    out.nodes.clear();
    out.closures.clear();
    out.patterns.reserve(compiled.size());
    out.nodes.reserve(compiled.size() * 2);
    out.closures.reserve(32);

    for (auto const &source : compiled) {
        auto const &native = source.native_form;
        csd_device_pattern_t pattern {};
        for (std::uint32_t slot = 0; slot < 4; ++slot) pattern.atom[slot] = native.atom[slot];
        for (std::uint32_t slot = 0; slot < 3; ++slot) pattern.bond[slot] = native.bond[slot];
        pattern.required_elements = source.required_elements;
        pattern.required_hybridizations = source.required_hybridizations;
        pattern.requires_aromatic = source.requires_aromatic ? 1u : 0u;
        for (std::uint32_t term = 0; term < 6; ++term)
            pattern.fourier_coefficients[term] = static_cast<float>(native.signs[term]) * native.V[term];

        pattern.pool_offset = static_cast<std::uint32_t>(out.nodes.size());
        for (std::uint32_t node_index = 0; node_index < native.pool.num_nodes; ++node_index)
            out.nodes.push_back(native.pool.nodes[node_index]);

        if (native.closure.active) {
            pattern.closure_offset = static_cast<std::uint32_t>(out.closures.size());
            out.closures.push_back(native.closure);
        }
        else { pattern.closure_offset = no_closure_offset_k; }

        out.patterns.push_back(pattern);
    }
}

/** Per-GPU device-resident table.  Owns one grow-only arena holding all three arrays. */
struct csd_torsion_table_device_t {
    device_byte_buffer_t arena;    ///< single device allocation, RAII-managed
    csd_torsion_table_view_t view; ///< pointers into @ref arena
    std::size_t arena_bytes = 0;   ///< bytes actually placed, padding included

    csd_torsion_table_device_t() = default;
    csd_torsion_table_device_t(csd_torsion_table_device_t const &) = delete;
    csd_torsion_table_device_t &operator=(csd_torsion_table_device_t const &) = delete;
};

namespace detail_csd_torsion {

inline std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

/** @brief Copy one host array into the arena at the next 256-byte boundary and return its address. */
template <typename element_>
element_ *place_array(std::byte *arena_base, std::size_t &offset, std::vector<element_> const &source) {
    if (source.empty()) return nullptr;
    offset = align_up(offset, 256);
    element_ *destination = reinterpret_cast<element_ *>(arena_base + offset);
    cudaMemcpy(destination, source.data(), source.size() * sizeof(element_), cudaMemcpyHostToDevice);
    offset += source.size() * sizeof(element_);
    return destination;
}

} // namespace detail_csd_torsion

/** @brief Raise the per-thread device stack to @ref csd_torsion_stack_bytes_k when it sits lower. */
inline void ensure_csd_torsion_device_stack() {
    std::size_t current = 0;
    if (cudaDeviceGetLimit(&current, cudaLimitStackSize) != cudaSuccess) return;
    if (current < csd_torsion_stack_bytes_k) cudaDeviceSetLimit(cudaLimitStackSize, csd_torsion_stack_bytes_k);
}

/**
 *  @brief Upload the staged table into one device arena and fill in the view that indexes it.
 *
 *  @param[in] host Staging buffers filled by @ref build_csd_torsion_table.
 *  @param[out] table Device arena and view; the caller keeps it alive for every kernel that reads it.
 */
inline void upload_csd_torsion_table_to_device(csd_torsion_table_host_t const &host,
                                               csd_torsion_table_device_t &table) {
    ensure_csd_torsion_device_stack();
    std::size_t const total = host.byte_size() + 3 * 256;
    if (!table.arena.ensure_capacity(total)) {
        std::fprintf(stderr, "upload_csd_torsion_table_to_device: device arena allocation (%zu bytes) failed\n", total);
        return;
    }
    std::byte *const arena_base = table.arena.data();
    std::size_t offset = 0;
    table.view.patterns = detail_csd_torsion::place_array(arena_base, offset, host.patterns);
    table.view.nodes = detail_csd_torsion::place_array(arena_base, offset, host.nodes);
    table.view.closures = detail_csd_torsion::place_array(arena_base, offset, host.closures);
    table.view.num_patterns = static_cast<std::uint32_t>(host.patterns.size());
    table.arena_bytes = offset;
    cudaDeviceSynchronize();
}

/** @brief Assemble the matcher's read handle on one device pattern. */
__device__ inline csd_pattern_view_t device_pattern_view(csd_torsion_table_view_t const &table,
                                                         csd_device_pattern_t const &pattern) {
    simple_ring_closure_t const *closure = pattern.closure_offset == no_closure_offset_k
                                               ? nullptr
                                               : table.closures + pattern.closure_offset;
    return csd_pattern_view_t(pattern.atom, pattern.bond, closure, table.nodes + pattern.pool_offset);
}

/**
 *  @brief Extract the CSD torsions of one molecule per warp.
 *
 *  @param[in] descriptors Device-resident descriptors, one per molecule.
 *  @param[in] num_molecules Descriptors to process; one warp each.
 *  @param[in] table Uploaded pattern table.
 *  @param[out] torsions Output slots, @p torsion_stride per molecule.
 *  @param[in] torsion_stride Slots reserved per molecule; a molecule emitting more is truncated.
 *  @param[out] torsion_counts Torsions written per molecule, before truncation.
 */
__global__ __launch_bounds__(32, 8) void csd_torsion_match_kernel(molecule_descriptor_t const *__restrict__ descriptors,
                                                                  std::uint32_t num_molecules,
                                                                  csd_torsion_table_view_t table,
                                                                  etk_torsion_descriptor_t *__restrict__ torsions,
                                                                  std::uint32_t torsion_stride,
                                                                  std::uint32_t *__restrict__ torsion_counts) {

    std::uint32_t const molecule_index = blockIdx.x;
    if (molecule_index >= num_molecules) return;
    std::uint32_t const lane = threadIdx.x;

    molecule_descriptor_t const &desc = descriptors[molecule_index];
    etk_torsion_descriptor_t *const output = torsions + static_cast<std::size_t>(molecule_index) * torsion_stride;

    // Bonds an earlier pattern already claimed, one bit per bond, shared so the ring pass on lane 0
    // sees what the whole warp claimed.
    __shared__ std::uint32_t claimed_words[max_bonds_k / warp_size_k];
    if (lane < max_bonds_k / warp_size_k) claimed_words[lane] = 0;
    __syncwarp();

    csd_molecule_gate_t const gate = build_molecule_gate(desc);
    std::uint32_t const num_bonds = desc.num_bonds;
    std::uint32_t written = 0;

    for (std::uint32_t bond_base = 0; bond_base < num_bonds; bond_base += warp_size_k) {
        std::uint32_t const bond_index = bond_base + lane;
        bool const owns_bond = bond_index < num_bonds;
        bool claimed = false;
        std::uint8_t matched_atoms[4] = {0, 0, 0, 0};
        float matched_coefficients[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (std::uint32_t pattern_index = 0; pattern_index < table.num_patterns; ++pattern_index) {
            if (__all_sync(0xFFFFFFFFu, !owns_bond || claimed)) break;
            csd_device_pattern_t const &pattern = table.patterns[pattern_index];
            if ((pattern.required_elements & ~gate.elements) != 0) continue;
            if (pattern.requires_aromatic && !gate.has_aromatic) continue;
            if ((pattern.required_hybridizations & ~gate.hybridizations) != 0) continue;
            if (!owns_bond || claimed) continue;

            csd_pattern_view_t const view = device_pattern_view(table, pattern);
            bool const bound = match_pattern_at_bond(
                desc, view, bond_index,
                [&](std::uint8_t a1, std::uint8_t a2, std::uint8_t a3, std::uint8_t a4, std::uint32_t) -> bool {
                    matched_atoms[0] = a1;
                    matched_atoms[1] = a2;
                    matched_atoms[2] = a3;
                    matched_atoms[3] = a4;
                    return true;
                });
            if (!bound) continue;
            claimed = true;
            for (std::uint32_t term = 0; term < 6; ++term)
                matched_coefficients[term] = pattern.fourier_coefficients[term];
        }

        // A claimed bond on a fused small ring, or on more than three rings, carries no torsion —
        // it stays claimed so the ring pass below leaves it alone too.
        bool emits = claimed;
        if (claimed && (bond_excluded_by_ring_fusion(desc, bond_index) || bond_ring_count(desc, bond_index) > 3))
            emits = false;

        std::uint32_t const claim_mask = __ballot_sync(0xFFFFFFFFu, claimed);
        if (lane == 0) claimed_words[bond_base / warp_size_k] = claim_mask;

        std::uint32_t const emit_mask = __ballot_sync(0xFFFFFFFFu, emits);
        std::uint32_t const slot = written + __popc(emit_mask & ((1u << lane) - 1u));
        if (emits && slot < torsion_stride) {
            etk_torsion_descriptor_t torsion {};
            for (std::uint32_t position = 0; position < 4; ++position)
                torsion.atom_indices[position] = matched_atoms[position];
            for (std::uint32_t term = 0; term < 6; ++term)
                torsion.fourier_coefficients[term] = matched_coefficients[term];
            output[slot] = torsion;
        }
        written += __popc(emit_mask);
    }

    __syncwarp();

    // Ring torsions mirror RDKit's `useBasicKnowledge=true`: a flat four-atom run around a small
    // ring claims its central bond, and which run wins depends on ring order, so this stays serial.
    if (lane == 0) {
        for (std::uint32_t ring_index = 0; ring_index < desc.num_rings; ++ring_index) {
            std::uint32_t const size = ring_atom_count(desc, ring_index);
            if (size < 4 || size > 6) continue;
            std::uint16_t const start = desc.ring_offset[ring_index];
            for (std::uint32_t position = 0; position < size; ++position) {
                std::uint8_t const a1 = desc.ring_atoms[start + position];
                std::uint8_t const a2 = desc.ring_atoms[start + (position + 1) % size];
                std::uint8_t const a3 = desc.ring_atoms[start + (position + 2) % size];
                std::uint8_t const a4 = desc.ring_atoms[start + (position + 3) % size];
                std::uint32_t const bond_index = find_bond_between(desc, a2, a3);
                if (bond_index == no_bond_index_k) continue;
                std::uint32_t const word = bond_index / warp_size_k;
                std::uint32_t const bit = 1u << (bond_index % warp_size_k);
                if ((claimed_words[word] & bit) != 0) continue;
                constexpr std::uint8_t planar_k = hybridization_sp2_k;
                if (desc.hybridization[a1] != planar_k || desc.hybridization[a2] != planar_k ||
                    desc.hybridization[a3] != planar_k || desc.hybridization[a4] != planar_k)
                    continue;

                claimed_words[word] |= bit;
                if (written < torsion_stride) {
                    etk_torsion_descriptor_t torsion {};
                    torsion.atom_indices[0] = a1;
                    torsion.atom_indices[1] = a2;
                    torsion.atom_indices[2] = a3;
                    torsion.atom_indices[3] = a4;
                    // Flat-ring signature: signs[1]=-1, V[1]=100.0 (from RDKit's hardcoded values).
                    for (std::uint32_t term = 0; term < 6; ++term) torsion.fourier_coefficients[term] = 0.0f;
                    torsion.fourier_coefficients[1] = -100.0f;
                    output[written] = torsion;
                }
                ++written;
            }
        }
        torsion_counts[molecule_index] = written;
    }
}

/**
 *  @brief Launch @ref csd_torsion_match_kernel, one warp per molecule.
 *
 *  @param[in] descriptors Device-resident descriptors.
 *  @param[in] num_molecules Molecules to process.
 *  @param[in] table Uploaded pattern table.
 *  @param[out] torsions Device output slots, @p torsion_stride per molecule.
 *  @param[in] torsion_stride Slots reserved per molecule.
 *  @param[out] torsion_counts Device array of per-molecule torsion counts.
 *  @param[in] stream Stream to launch on.
 */
inline void launch_csd_torsion_match(molecule_descriptor_t const *descriptors, std::uint32_t num_molecules,
                                     csd_torsion_table_view_t table, etk_torsion_descriptor_t *torsions,
                                     std::uint32_t torsion_stride, std::uint32_t *torsion_counts,
                                     cudaStream_t stream = nullptr) {
    if (num_molecules == 0) return;
    csd_torsion_match_kernel<<<num_molecules, warp_size_k, 0, stream>>>(descriptors, num_molecules, table, torsions,
                                                                        torsion_stride, torsion_counts);
}

} // namespace smallfield::cuda
