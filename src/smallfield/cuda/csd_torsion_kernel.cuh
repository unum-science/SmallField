/**
 *  @file src/smallfield/cuda/csd_torsion_kernel.cuh
 *  @brief Device-resident CSD torsion table and the warp-per-molecule matcher that reads it.
 *
 *  The CSD torsion preferences compile to 470 stack-only patterns whose matcher, in
 *  `preprocess/csd_torsion_native.hpp`, is a pure function of `molecule_descriptor_t`. This
 *  header carries that matcher to the GPU: it flattens the compiled patterns into POD mirrors,
 *  uploads each of the three arrays into its own stream-ordered device buffer, and runs one warp
 *  per molecule over the result.
 *
 *  Two members of the host pattern cannot cross. A `recursive_pool_t` is a fixed 40-node pool
 *  even though the widest body in the corpus spends seven nodes, and a `simple_ring_closure_t`
 *  is half a kilobyte that only the ring-closing patterns use. Both move into shared arenas the
 *  pattern indexes by offset, which is what turns a 2.6 MB table into a 660 KB one, and the
 *  matcher reads a pattern through `csd_pattern_view_t` so the same body serves both layouts.
 *
 *  Work assignment: a lane owns one bond and walks the pattern list in file order until a pattern
 *  binds, which is exactly the "first pattern claims the bond" rule the host loop implements from
 *  the other side. The pattern is warp-uniform, so every lane reads the same address and the
 *  table streams through L2 once per warp rather than once per lane. The ring torsions that
 *  RDKit's `useBasicKnowledge` adds are order-dependent between rings and run on lane 0.
 *
 *  Correctness: identical torsion set per molecule to `extract_etk_torsions_native`, checked
 *  molecule by molecule in `test/cuda/test_csd_torsion_device.cpp`.
 */

#pragma once

#if defined(__CUDACC__)

#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <vector> // `std::vector`

#include <cuda_runtime.h>

#include "smallfield/cuda/warp_primitives.cuh"
#include "smallfield/cuda_error.hpp"
#include "smallfield/preprocess/csd_torsion_dispatcher.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/storage.hpp"

namespace smallfield::cuda {

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
using smallfield::types::buffer;
using smallfield::types::device_allocator_t;
using smallfield::types::etk_torsion_descriptor_t;
using smallfield::types::hybridization_sp2_k;
using smallfield::types::max_bonds_k;
using smallfield::types::molecule_descriptor_t;
using smallfield::types::reserve_outcome_t;
using smallfield::types::table_residency_t;
using smallfield::types::upload_into;

/**
 *  @brief Per-thread device stack the matcher needs. Sub-query evaluation still recurses, bounded by the
 *  compiled corpus rather than by the molecule: a body nests at most `max_recursive_depth_k`
 *  levels of `$(...)` and walks at most `max_recursive_hops_k` hops from its root. Driving the
 *  limit down until the kernel faults puts the peak between 512 and 768 bytes across 39168 PubChem
 *  and Enamine REAL molecules, so the CUDA default of 1024 already suffices and @ref
 *  ensure_csd_torsion_device_stack only has work to do when a caller lowered it.
 */
inline constexpr std::size_t csd_torsion_stack_bytes_k = 1024;

/** Value of `csd_device_pattern_t::closure_offset` for a pattern that closes no ring. */
inline constexpr std::uint32_t no_closure_offset_k = 0xFFFFFFFFu;

/**
 *  @brief POD mirror of one compiled torsion pattern. Everything the matcher reads is either inline here
 *  or reachable through an offset into the table's node and closure arenas, so the whole struct is
 *  memcpy-able to the device.
 */
struct alignas(8) csd_device_pattern_t {
    /** The labelled atoms `:1` through `:4`. */
    simple_atom_constraint_t atom[4];
    /** Chain bonds; the central `:2`-`:3` bond is index 1. */
    simple_bond_constraint_t bond[3];
    /** Bit N set when a match needs element N present. */
    std::uint64_t required_elements;
    /** First sub-query node of this pattern in the node arena. */
    std::uint32_t pool_offset;
    /** Ring closure in the closure arena; `no_closure_offset_k` when absent. */
    std::uint32_t closure_offset;
    /** V₁ through V₆ with the pattern's signs already applied. */
    float fourier_coefficients[6];
    /** Bit H set when a match needs hybridization H present. */
    std::uint8_t required_hybridizations;
    /** 1 when a match needs an aromatic atom. */
    std::uint8_t requires_aromatic;
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

    /** Total bytes the three arrays occupy on the device, before alignment padding. */
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
 *  `csd_device_pattern_t::pool_offset`.
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
            pattern.fourier_coefficients[term] = source.fourier_coefficients[term];

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

/**
 *  @brief Per-GPU device-resident table: one stream-ordered buffer per array, plus the view over them.
 *
 *  There is no shared arena. Sub-allocation would be worth its bookkeeping only if it collapsed the upload
 *  into a single `cudaMemcpyAsync`, and the three element types differ, so it is three copies either way.
 *  Each buffer carries the stream that allocated it, which is also the stream its free is ordered against,
 *  so the moves are defaulted and no destructor is written by hand.
 */
struct csd_torsion_table_device_t {
    /** The POD patterns the matcher walks in file order. */
    buffer<csd_device_pattern_t, device_allocator_t> patterns;
    /** Sub-query nodes, indexed through `csd_device_pattern_t::pool_offset`. */
    buffer<recursive_node_t, device_allocator_t> nodes;
    /** Ring closures, indexed through `csd_device_pattern_t::closure_offset`. */
    buffer<simple_ring_closure_t, device_allocator_t> closures;
    /** Raw handles on the three buffers, passed to the kernel by value. */
    csd_torsion_table_view_t view;

    csd_torsion_table_device_t() = default;
    csd_torsion_table_device_t(csd_torsion_table_device_t const &) = delete;
    csd_torsion_table_device_t &operator=(csd_torsion_table_device_t const &) = delete;
    csd_torsion_table_device_t(csd_torsion_table_device_t &&) noexcept = default;
    csd_torsion_table_device_t &operator=(csd_torsion_table_device_t &&) noexcept = default;
};

/**
 *  @brief Rebind one device array to @p stream, grow it to hold @p source, and enqueue the copy.
 *
 *  The buffer is rebound rather than merely reserved, because the stream carries the device both the
 *  allocation and its eventual free belong to, and the table is staged once per device at start-up.
 *
 *  @param[out] destination Device buffer to fill; any prior contents and their allocation are dropped.
 *  @param[in] source Host array to copy.
 *  @param[in] stream Orders the allocation and the copy, and names the device they land on.
 *  @param[in] sink Routes a refused allocation or copy to the logger and the error counter.
 *  @return Whether the array is resident. An empty source is resident. On failure the stream carries the reason.
 */
template <typename element_>
[[nodiscard]] table_residency_t upload_table_array_(buffer<element_, device_allocator_t> &destination,
                                                    std::vector<element_> const &source, cudaStream_t stream,
                                                    cuda_error_sink_t const &sink) {
    destination = buffer<element_, device_allocator_t>(device_allocator_t {.stream = stream});
    if (source.empty()) return table_residency_t::resident_k;
    if (destination.try_reserve(source.size()) == reserve_outcome_t::failed_k) {
        [[maybe_unused]] bool const reported = sink.check(cudaErrorMemoryAllocation,
                                                          "upload_csd_torsion_table_to_device");
        return table_residency_t::refused_k;
    }
    if (!upload_into<element_>(destination.view(), {source.data(), source.size()}, stream, sink))
        return table_residency_t::refused_k;
    return table_residency_t::resident_k;
}

/** Raise the per-thread device stack to `csd_torsion_stack_bytes_k` when it sits lower. */
inline void ensure_csd_torsion_device_stack() {
    std::size_t current = 0;
    if (cudaDeviceGetLimit(&current, cudaLimitStackSize) != cudaSuccess) return;
    if (current < csd_torsion_stack_bytes_k) cudaDeviceSetLimit(cudaLimitStackSize, csd_torsion_stack_bytes_k);
}

/**
 *  @brief Upload the staged table into the device buffers and fill in the view that indexes them.
 *
 *  The view is cleared first and only rebuilt once all three arrays are resident, so a caller that
 *  finds `csd_torsion_table_view_t::patterns` null knows no kernel may read the table. Nothing waits
 *  here: the copies ride @p stream, which is the same stream the matcher launches on.
 *
 *  @param[in] host Staging buffers filled by `build_csd_torsion_table`.
 *  @param[out] device Device buffers and view; the caller keeps them alive for every kernel that reads them.
 *  @param[in] stream Orders the allocations and the copies, and outlives the buffers they produce.
 *  @param[in] sink Routes a refused allocation or copy to the logger and the error counter.
 *  @return Whether the whole table is resident. A refused device must not be given molecules.
 */
[[nodiscard]] inline table_residency_t upload_csd_torsion_table_to_device(csd_torsion_table_host_t const &host,
                                                                          csd_torsion_table_device_t &device,
                                                                          cudaStream_t stream,
                                                                          cuda_error_sink_t const &sink = {}) {
    ensure_csd_torsion_device_stack();
    device.view = {};
    if (upload_table_array_(device.patterns, host.patterns, stream, sink) == table_residency_t::refused_k)
        return table_residency_t::refused_k;
    if (upload_table_array_(device.nodes, host.nodes, stream, sink) == table_residency_t::refused_k)
        return table_residency_t::refused_k;
    if (upload_table_array_(device.closures, host.closures, stream, sink) == table_residency_t::refused_k)
        return table_residency_t::refused_k;

    custd::span<csd_device_pattern_t const> const patterns = device.patterns.view();
    device.view.patterns = patterns.data();
    device.view.nodes = device.nodes.data();
    device.view.closures = device.closures.data();
    device.view.num_patterns = static_cast<std::uint32_t>(patterns.size());
    return table_residency_t::resident_k;
}

/** Assemble the matcher's read handle on one device pattern. */
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
static __global__ __launch_bounds__(32, 8) void csd_torsion_match_kernel(
    molecule_descriptor_t const *__restrict__ descriptors, std::uint32_t num_molecules, csd_torsion_table_view_t table,
    etk_torsion_descriptor_t *__restrict__ torsions, std::uint32_t torsion_stride,
    std::uint32_t *__restrict__ torsion_counts) {

    std::uint32_t const molecule_index = blockIdx.x;
    if (molecule_index >= num_molecules) return;
    std::uint32_t const lane = threadIdx.x;

    molecule_descriptor_t const &descriptor = descriptors[molecule_index];
    etk_torsion_descriptor_t *const output = torsions + static_cast<std::size_t>(molecule_index) * torsion_stride;

    // Bonds an earlier pattern already claimed, one bit per bond, shared so the ring pass on lane 0
    // sees what the whole warp claimed.
    __shared__ std::uint32_t claimed_words[max_bonds_k / warp_size_k];
    if (lane < max_bonds_k / warp_size_k) claimed_words[lane] = 0;
    __syncwarp();

    csd_molecule_gate_t const gate = build_molecule_gate(descriptor);
    std::uint32_t const num_bonds = descriptor.num_bonds;
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
                descriptor, view, bond_index,
                [&](std::uint8_t atom_first, std::uint8_t atom_second, std::uint8_t atom_third,
                    std::uint8_t atom_fourth, std::uint32_t) -> bool {
                    matched_atoms[0] = atom_first;
                    matched_atoms[1] = atom_second;
                    matched_atoms[2] = atom_third;
                    matched_atoms[3] = atom_fourth;
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
        if (claimed &&
            (bond_excluded_by_ring_fusion(descriptor, bond_index) || bond_ring_count(descriptor, bond_index) > 3))
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
        for (std::uint32_t ring_index = 0; ring_index < descriptor.num_rings; ++ring_index) {
            std::uint32_t const size = ring_atom_count(descriptor, ring_index);
            if (size < 4 || size > 6) continue;
            std::uint16_t const start = descriptor.ring_offset[ring_index];
            for (std::uint32_t position = 0; position < size; ++position) {
                std::uint8_t const atom_first = descriptor.ring_atoms[start + position];
                std::uint8_t const atom_second = descriptor.ring_atoms[start + (position + 1) % size];
                std::uint8_t const atom_third = descriptor.ring_atoms[start + (position + 2) % size];
                std::uint8_t const atom_fourth = descriptor.ring_atoms[start + (position + 3) % size];
                std::uint32_t const bond_index = find_bond_between(descriptor, atom_second, atom_third);
                if (bond_index == no_bond_index_k) continue;
                std::uint32_t const word = bond_index / warp_size_k;
                std::uint32_t const bit_mask = 1u << (bond_index % warp_size_k);
                if ((claimed_words[word] & bit_mask) != 0) continue;
                constexpr std::uint8_t planar_k = hybridization_sp2_k;
                if (descriptor.hybridization[atom_first] != planar_k ||
                    descriptor.hybridization[atom_second] != planar_k ||
                    descriptor.hybridization[atom_third] != planar_k ||
                    descriptor.hybridization[atom_fourth] != planar_k)
                    continue;

                claimed_words[word] |= bit_mask;
                if (written < torsion_stride) {
                    etk_torsion_descriptor_t torsion {};
                    torsion.atom_indices[0] = atom_first;
                    torsion.atom_indices[1] = atom_second;
                    torsion.atom_indices[2] = atom_third;
                    torsion.atom_indices[3] = atom_fourth;
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

} // namespace smallfield::cuda

#endif // __CUDACC__
