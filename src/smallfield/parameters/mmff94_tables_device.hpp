/**
 *  @file src/smallfield/parameters/mmff94_tables_device.hpp
 *  @brief Upload MMFF94 tables from host to device.
 *
 *  Mirrors `mmff94_tables_host_t` on the device: one grow-only device buffer per table, and an
 *  `mmff94_tables_view_t` whose spans point at them, so every `lookup_*` in `mmff94_tables.hpp` runs
 *  unchanged on either side.
 *
 *  One buffer per table rather than one carved arena. Every table is its own `cudaMemcpyAsync` in both
 *  shapes anyway, because the entries differ in size and each span must start aligned for its own type,
 *  and stream-ordered allocation already removes the `cudaMalloc` cost an arena existed to amortize.
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
using smallfield::types::table_residency_t;
using smallfield::types::upload_into;

#pragma region Device Storage

/**
 *  @brief Device-resident MMFF94 tables, one stream-ordered buffer per table.
 *
 *  One buffer per table of `mmff94_tables_host_t`, in that container's key order. `view` spans them, so
 *  this struct must outlive any launch that reads the view. Each buffer carries the stream its allocation
 *  was ordered against, and therefore the device its free reaches, which is what lets the destructor and
 *  both move operations be the compiler's own.
 */
struct mmff94_tables_device_t {
    buffer<mmff94_vdw_entry_t, device_allocator_t> vdw;
    buffer<mmff94_bond_entry_t, device_allocator_t> bond;
    buffer<mmff94_angle_entry_t, device_allocator_t> angle;
    buffer<mmff94_torsion_entry_t, device_allocator_t> torsion;
    buffer<mmff94_oop_entry_t, device_allocator_t> oop;
    buffer<mmff94_stbn_entry_t, device_allocator_t> stbn;
    buffer<mmff94_dfsb_entry_t, device_allocator_t> dfsb;
    buffer<mmff94_def_entry_t, device_allocator_t> def;
    buffer<mmff94_prop_entry_t, device_allocator_t> prop;
    buffer<mmff94_pbci_entry_t, device_allocator_t> pbci;
    buffer<mmff94_chg_entry_t, device_allocator_t> chg;
    buffer<mmff94_bndk_entry_t, device_allocator_t> bndk;
    buffer<mmff94_hl_entry_t, device_allocator_t> hl;
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
 *  @brief Reserve device storage for one table, enqueue its upload, and point @p staged at the result.
 *
 *  Rebinding the buffer to @p stream is what ties the allocation and its eventual free to one device, so
 *  it happens before the reservation rather than at declaration. An empty host table allocates nothing and
 *  leaves @p staged empty, which every `lookup_*` cascade already reads as "no entry" — and is `resident_k`,
 *  because a table that has no rows is a different fact from a table whose rows never arrived.
 *
 *  @param[in] source Host table to copy; may legitimately be empty.
 *  @param[out] destination Device buffer to fill; any prior contents and their allocation are dropped.
 *  @param[out] staged Span over @p destination, empty unless rows landed.
 *  @param[in] stream Orders the allocation and the copy, and names the device they land on.
 *  @param[in] sink Routes a refused allocation or copy to the logger and the error counter.
 */
template <typename element_>
[[nodiscard]] table_residency_t upload_table_(std::vector<element_> const &source,
                                              buffer<element_, device_allocator_t> &destination,
                                              custd::span<element_ const> &staged, cudaStream_t stream,
                                              cuda_error_sink_t const &sink) noexcept {
    destination = buffer<element_, device_allocator_t> {device_allocator_t {.stream = stream}};
    staged = {};
    if (source.empty()) return table_residency_t::resident_k;

    if (destination.try_reserve(source.size()) == reserve_outcome_t::failed_k) {
        [[maybe_unused]] bool const reported = sink.check(cudaErrorMemoryAllocation, "upload_table_");
        return table_residency_t::refused_k;
    }

    custd::span<element_ const> const host_view {source.data(), source.size()};
    if (!upload_into(destination.view(), host_view, stream, sink)) return table_residency_t::refused_k;
    staged = {destination.data(), source.size()};
    return table_residency_t::resident_k;
}

/**
 *  @brief Upload every MMFF94 parameter table onto the device along @p stream.
 *
 *  The spans are built into a staging view and published to `device.view` only once every table has landed,
 *  so a caller that finds the view empty knows no kernel may read it — a partially uploaded set would
 *  otherwise present as a complete force field whose missing lookups fall through to the empirical rules.
 *  @p device must outlive any launch that reads the view, and @p stream must outlive @p device.
 *
 *  The copies are only enqueued, and @p host is pageable, so its vectors must survive until the caller's
 *  next synchronization on @p stream — which is also when a device-side lookup first becomes valid.
 *
 *  @param[in] host Tables extracted from RDKit.
 *  @param[out] device Device buffers and the view over them; empty on refusal.
 *  @param[in] stream Orders the allocations and the copies, and outlives the buffers they produce.
 *  @param[in] sink Routes a refused allocation or copy to the logger and the error counter.
 *  @return Whether the whole set is resident. A refused device must not be given molecules.
 */
[[nodiscard]] inline table_residency_t upload_mmff94_tables_to_device(mmff94_tables_host_t const &host,
                                                                      mmff94_tables_device_t &device,
                                                                      cudaStream_t stream,
                                                                      cuda_error_sink_t const &sink = {}) {
    device.view = {};
    mmff94_tables_view_t staging {};

    // One refusal is the whole force field's refusal, so the walk stops at the first one and the staging
    // view never reaches `device.view`.
    table_residency_t residency = table_residency_t::resident_k;
    auto stage = [&](auto const &source, auto &destination, auto &staged) {
        if (residency == table_residency_t::resident_k)
            residency = upload_table_(source, destination, staged, stream, sink);
    };
    stage(host.vdw, device.vdw, staging.vdw);
    stage(host.bond, device.bond, staging.bond);
    stage(host.angle, device.angle, staging.angle);
    stage(host.torsion, device.torsion, staging.torsion);
    stage(host.oop, device.oop, staging.oop);
    stage(host.stbn, device.stbn, staging.stbn);
    stage(host.dfsb, device.dfsb, staging.dfsb);
    stage(host.def, device.def, staging.def);
    stage(host.prop, device.prop, staging.prop);
    stage(host.pbci, device.pbci, staging.pbci);
    stage(host.chg, device.chg, staging.chg);
    stage(host.bndk, device.bndk, staging.bndk);
    stage(host.hl, device.hl, staging.hl);
    stage(host.covrad, device.covrad, staging.covrad);
    if (residency == table_residency_t::refused_k) return residency;

    staging.max_atom_type = host.max_atom_type;
    device.view = staging;
    return table_residency_t::resident_k;
}

#pragma endregion Upload

} // namespace smallfield::parameters
