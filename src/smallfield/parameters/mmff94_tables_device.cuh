/**
 *  @file src/smallfield/parameters/mmff94_tables_device.cuh
 *  @brief Upload MMFF94 tables from host to device.
 *
 *  Mirrors `mmff94_tables_host_t` on the device. The host side extracts RDKit's tables into flat POD
 *  vectors, see `mmff94_tables.hpp`; this header owns one grow-only device buffer per table and fills an
 *  `mmff94_tables_view_t` whose spans point at them. Every `lookup_*` in `mmff94_tables.hpp` then works
 *  identically host-side and device-side with no logic duplication.
 *
 *  @section per_table One buffer per table, not one arena
 *
 *  A slab carved by hand into typed regions exists mostly to amortize `cudaMalloc`, which synchronizes the
 *  whole device. `device_allocator_t` draws from the per-device pool in stream order instead, so the slab
 *  buys nothing here. It never made the transfer single either — every table is its own copy in both
 *  shapes, because the entries differ in size and each span must start aligned for its own type.
 *
 *  The tables are read from every MMFF kernel through `const __restrict__` pointers; L1 and L2 warm-up is
 *  essentially free given the warp-per-molecule access pattern.
 */

#pragma once

#include <vector> // `std::vector`

#include <cuda_runtime.h>

#include "smallfield/cuda_error.hpp"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/types/storage.hpp"

namespace smallfield::parameters {

using smallfield::types::buffer;
using smallfield::types::device_allocator_t;
using smallfield::types::reserve_outcome_t;
using smallfield::types::upload_into;

#pragma region Device Storage

/**
 *  @brief Device-resident MMFF94 tables, one stream-ordered buffer per table.
 *
 *  `view` is the POD every kernel takes by value; the buffers beside it are the storage those spans point
 *  at, so this struct must outlive any launch that reads the view. Each buffer carries the stream its
 *  allocation was ordered against, and therefore the device its free reaches, which is what lets the
 *  destructor and both move operations be the compiler's own.
 *
 *  Not copyable, because two owners would free one allocation twice.
 */
struct mmff94_tables_device_t {
    /** Sorted by atom type. */
    buffer<mmff94_vdw_entry_t, device_allocator_t> vdw;
    /** Sorted by (i, j, bond_type). */
    buffer<mmff94_bond_entry_t, device_allocator_t> bond;
    /** Sorted by (j, i, k, angle_type). */
    buffer<mmff94_angle_entry_t, device_allocator_t> angle;
    /** Sorted by (j, k, i, l, tor_type). */
    buffer<mmff94_torsion_entry_t, device_allocator_t> torsion;
    /** Sorted by (j, i, k, l). */
    buffer<mmff94_oop_entry_t, device_allocator_t> oop;
    /** Sorted by (j, i, k, sb_type). */
    buffer<mmff94_stbn_entry_t, device_allocator_t> stbn;
    /** Sorted by (row_i, row_j, row_k). */
    buffer<mmff94_dfsb_entry_t, device_allocator_t> dfsb;
    /** Dense: indexed directly by atom type. */
    buffer<mmff94_def_entry_t, device_allocator_t> def;
    /** Dense: indexed directly by atom type. */
    buffer<mmff94_prop_entry_t, device_allocator_t> prop;
    /** Sorted by atom type. */
    buffer<mmff94_pbci_entry_t, device_allocator_t> pbci;
    /** Sorted by (i, j, bond_type). */
    buffer<mmff94_chg_entry_t, device_allocator_t> chg;
    /** Sorted by (atomic_num_i, atomic_num_j). */
    buffer<mmff94_bndk_entry_t, device_allocator_t> bndk;
    /** Sorted by (row_i, row_j). */
    buffer<mmff94_hl_entry_t, device_allocator_t> hl;
    /** Sorted by atomic number. */
    buffer<mmff94_covrad_entry_t, device_allocator_t> covrad;

    /** Spans over the buffers above plus the scalar VdW constants, passed by value into every kernel. */
    mmff94_tables_view_t view;

    mmff94_tables_device_t() = default;
    mmff94_tables_device_t(mmff94_tables_device_t const &) = delete;
    mmff94_tables_device_t &operator=(mmff94_tables_device_t const &) = delete;
    mmff94_tables_device_t(mmff94_tables_device_t &&) noexcept = default;
    mmff94_tables_device_t &operator=(mmff94_tables_device_t &&) noexcept = default;
};

#pragma endregion Device Storage

#pragma region Upload

/**
 *  @brief Reserve device storage for one table, enqueue its upload, and hand back the device-side span.
 *
 *  Rebinding the buffer to @p stream is what ties the allocation and its eventual free to one device, so
 *  it happens before the reservation rather than at declaration. An empty host table allocates nothing and
 *  yields an empty span, which every `lookup_*` cascade already reads as "no entry".
 *
 *  @param[in] source Sorted host table, in the key order the lookup cascades expect.
 *  @param[out] destination Grow-only device storage for this one table.
 *  @param[in] stream Orders the allocation and the copy against the rest of the caller's work.
 *  @param[in] sink Routes a refused reservation or a rejected copy to the logger and the error counter.
 *  @return The span to store into `mmff94_tables_view_t`, empty when nothing was uploaded.
 */
template <typename element_>
custd::span<element_ const> upload_table_(std::vector<element_> const &source,
                                          buffer<element_, device_allocator_t> &destination, cudaStream_t stream,
                                          cuda_error_sink_t const &sink) noexcept {
    if (source.empty()) return {};

    destination = buffer<element_, device_allocator_t> {device_allocator_t {.stream = stream}};
    if (destination.try_reserve(source.size()) == reserve_outcome_t::failed_k) {
        sink.check(cudaErrorMemoryAllocation, "upload_table_");
        return {};
    }

    custd::span<element_ const> const host_view {source.data(), source.size()};
    if (!upload_into(destination.view(), host_view, stream, sink)) return {};
    return {destination.data(), source.size()};
}

/**
 *  @brief Upload every MMFF94 parameter table onto the device along @p stream.
 *
 *  Fills `device.view` with spans into `device`'s own buffers, so the caller must keep @p device alive for
 *  the lifetime of any launch that reads that view, and @p stream alive for the lifetime of @p device.
 *
 *  The copies are only enqueued. `host` is pageable, so the driver stages it and the vectors must survive
 *  until the caller's next synchronization on @p stream, which is also when a device-side lookup first
 *  becomes valid.
 *
 *  @param[in] host Tables extracted from RDKit once at context init.
 *  @param[out] device Per-device storage and the view over it.
 *  @param[in] stream Orders every allocation and copy this issues.
 *  @param[in] sink Routes a refused allocation or a rejected copy to the logger and the error counter.
 */
inline void upload_mmff94_tables_to_device(mmff94_tables_host_t const &host, mmff94_tables_device_t &device,
                                           cudaStream_t stream, cuda_error_sink_t const &sink = {}) {
    device.view.vdw = upload_table_(host.vdw, device.vdw, stream, sink);
    device.view.bond = upload_table_(host.bond, device.bond, stream, sink);
    device.view.angle = upload_table_(host.angle, device.angle, stream, sink);
    device.view.torsion = upload_table_(host.torsion, device.torsion, stream, sink);
    device.view.oop = upload_table_(host.oop, device.oop, stream, sink);
    device.view.stbn = upload_table_(host.stbn, device.stbn, stream, sink);
    device.view.dfsb = upload_table_(host.dfsb, device.dfsb, stream, sink);
    device.view.def = upload_table_(host.def, device.def, stream, sink);
    device.view.prop = upload_table_(host.prop, device.prop, stream, sink);
    device.view.pbci = upload_table_(host.pbci, device.pbci, stream, sink);
    device.view.chg = upload_table_(host.chg, device.chg, stream, sink);
    device.view.bndk = upload_table_(host.bndk, device.bndk, stream, sink);
    device.view.hl = upload_table_(host.hl, device.hl, stream, sink);
    device.view.covrad = upload_table_(host.covrad, device.covrad, stream, sink);

    device.view.vdw_power = host.vdw_power;
    device.view.vdw_b = host.vdw_b;
    device.view.vdw_beta = host.vdw_beta;
    device.view.vdw_darad = host.vdw_darad;
    device.view.vdw_daeps = host.vdw_daeps;
    device.view.max_atom_type = host.max_atom_type;
}

#pragma endregion Upload

} // namespace smallfield::parameters
