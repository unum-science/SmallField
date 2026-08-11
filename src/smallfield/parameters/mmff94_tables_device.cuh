/**
 *  @file src/smallfield/parameters/mmff94_tables_device.cuh
 *  @brief Upload MMFF94 tables from host to device.
 *
 *  Mirrors `mmff94_tables_host_t` on the device. The host side extracts RDKit's tables into flat POD
 *  vectors (see `mmff94_tables.hpp`); this header owns a one-time-per-GPU RAII device byte arena
 *  holding exactly the same POD entries, and returns an `mmff94_tables_view_t` whose `mmff94_span`
 *  fields point at the device arena. Every `lookup_*` in `mmff94_tables.hpp` then works identically
 *  host-side and device-side with no logic duplication.
 *
 *  Small high-frequency tables (`prop`, `def`, `covrad`, `hl`, `bndk`) go to `__constant__` memory
 *  for broadcast reads during atom typing. Larger searchable tables stay in global memory and are
 *  read through `const __restrict__` pointers; L1/L2 warm-up is essentially free given the
 *  warp-per-molecule access pattern.
 *
 *  The constant-memory tables are sized generously (see below) so an RDKit upgrade that adds a few
 *  atom types doesn't overflow. If `upload_mmff94_tables_to_device` finds a table larger than its
 *  constant slot, it falls back to global memory for that table — a non-fatal degradation.
 */

#pragma once

#include <cstddef> // `std::byte`

#include <utility> // `std::move`
#include <vector>  // `std::vector`

#include <cuda_runtime.h>

#include "smallfield/api/wave_slice_pool.hpp"
#include "smallfield/log.hpp"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/utilities.hpp"

namespace smallfield::parameters {

using smallfield::api::device_byte_buffer_t;

/**
 *  @brief Constant-memory table capacities, sized for MMFF94 with headroom over RDKit's live counts.
 *
 *  The budgets leave slack for a future RDKit update that adds a few more entries.
 */
inline constexpr std::size_t mmff94_prop_const_cap_k = 128;
inline constexpr std::size_t mmff94_def_const_cap_k = 128;
inline constexpr std::size_t mmff94_covrad_const_cap_k = 64;
inline constexpr std::size_t mmff94_hl_const_cap_k = 64;
inline constexpr std::size_t mmff94_bndk_const_cap_k = 64;

#if defined(__CUDACC__)
/**
 *  @brief Constant-memory MMFF94 tables, one copy per translation unit.
 *
 *  Internal linkage means every translation unit that includes this header gets its own
 *  constant-memory copy, so a test that compiles the header and also links the library does not
 *  hand nvlink two definitions of one device symbol. Each copy is uploaded by that unit's own
 *  table upload.
 */
static __constant__ mmff94_prop_entry_t c_mmff94_prop[mmff94_prop_const_cap_k];
static __constant__ mmff94_def_entry_t c_mmff94_def[mmff94_def_const_cap_k];
static __constant__ mmff94_covrad_entry_t c_mmff94_covrad[mmff94_covrad_const_cap_k];
static __constant__ mmff94_hl_entry_t c_mmff94_hl[mmff94_hl_const_cap_k];
static __constant__ mmff94_bndk_entry_t c_mmff94_bndk[mmff94_bndk_const_cap_k];
#endif

/** Per-GPU device-resident view. Owns a grow-only device byte arena for the large tables; the `view`
 *  field plugs into every `lookup_*`. Pointers inside `view.prop/def/covrad/hl/bndk` target the
 *  `__constant__` symbols when the small-table capacities were sufficient; otherwise they fall back
 *  to device-global slots inside this arena. */
struct mmff94_tables_device_t {
    /** Single grow-only device allocation, RAII-managed. */
    device_byte_buffer_t arena;
    /** Spans point into arena / __constant__. */
    mmff94_tables_view_t view;
    int device_id = -1;

    mmff94_tables_device_t() = default;
    mmff94_tables_device_t(mmff94_tables_device_t const &) = delete;
    mmff94_tables_device_t &operator=(mmff94_tables_device_t const &) = delete;
    mmff94_tables_device_t(mmff94_tables_device_t &&other) noexcept
        : arena(std::move(other.arena)), view(other.view), device_id(other.device_id) {
        other.view = {};
        other.device_id = -1;
    }
    mmff94_tables_device_t &operator=(mmff94_tables_device_t &&other) noexcept {
        if (this != &other) {
            // Ensure the old arena is freed on the right device before being replaced.
            if (arena.data()) {
                int caller_device = -1;
                cudaGetDevice(&caller_device);
                if (device_id >= 0 && device_id != caller_device) cudaSetDevice(device_id);
                arena = device_byte_buffer_t {};
                if (device_id >= 0 && device_id != caller_device && caller_device >= 0) cudaSetDevice(caller_device);
            }
            arena = std::move(other.arena);
            view = other.view;
            device_id = other.device_id;
            other.view = {};
            other.device_id = -1;
        }
        return *this;
    }
    ~mmff94_tables_device_t() {
        if (arena.data()) {
            // The device byte buffer's dtor calls cudaFree; switch to the owning device first so the
            // free targets the right context.
            int caller_device = -1;
            cudaGetDevice(&caller_device);
            if (device_id >= 0 && device_id != caller_device) cudaSetDevice(device_id);
            arena = device_byte_buffer_t {};
            if (device_id >= 0 && device_id != caller_device && caller_device >= 0) cudaSetDevice(caller_device);
        }
    }
};

/** Copy a host vector to device at the given offset, returning a span that targets the device arena
 *  at that location. Updates `offset` to the next 256-byte-aligned position. */
template <typename element_>
mmff94_span<element_> place_table_(std::byte *arena_base, std::size_t &offset, std::vector<element_> const &source) {
    if (source.empty()) return mmff94_span<element_> {};
    std::size_t const bytes = source.size() * sizeof(element_);
    offset = align_up_(offset, 256);
    element_ *destination = reinterpret_cast<element_ *>(arena_base + offset);
    cudaMemcpy(destination, source.data(), bytes, cudaMemcpyHostToDevice);
    offset += bytes;
    return mmff94_span<element_> {destination, source.size()};
}

/** Place a small table given the `__constant__` slot reserved for it. That slot is not taken yet —
 *  see the note in the body. */
template <typename element_>
mmff94_span<element_> place_const_or_global_(std::byte *arena_base, std::size_t &offset,
                                             std::vector<element_> const &source,
                                             [[maybe_unused]] void const *const_symbol_dev_ptr,
                                             [[maybe_unused]] std::size_t const_cap) {
    // Always use global memory for now. Constant-memory placement via `cudaGetSymbolAddress` and a
    // read through a plain device pointer is valid but adds one more variable to debug; it is a perf
    // tweak to revisit once the current upload path is stable.
    return place_table_(arena_base, offset, source);
}

/** Upload every MMFF94 table in `host` to device memory. `out` is filled with device pointers;
 *  caller owns it and must keep it alive for the lifetime of any kernel launch that reads via
 *  `out.view`. The allocation failure is reported through `log`; null falls back to a default one. */
inline void upload_mmff94_tables_to_device(mmff94_tables_host_t const &host, mmff94_tables_device_t &out,
                                           logger_t const *log = nullptr) {
    cudaGetDevice(&out.device_id);
    // Upper-bound arena size: sum of all table bytes + 14 × 256-byte padding.
    std::size_t total = 0;
    total += host.vdw.size() * sizeof(mmff94_vdw_entry_t) + 256;
    total += host.bond.size() * sizeof(mmff94_bond_entry_t) + 256;
    total += host.angle.size() * sizeof(mmff94_angle_entry_t) + 256;
    total += host.torsion.size() * sizeof(mmff94_torsion_entry_t) + 256;
    total += host.oop.size() * sizeof(mmff94_oop_entry_t) + 256;
    total += host.stbn.size() * sizeof(mmff94_stbn_entry_t) + 256;
    total += host.dfsb.size() * sizeof(mmff94_dfsb_entry_t) + 256;
    total += host.def.size() * sizeof(mmff94_def_entry_t) + 256;
    total += host.prop.size() * sizeof(mmff94_prop_entry_t) + 256;
    total += host.pbci.size() * sizeof(mmff94_pbci_entry_t) + 256;
    total += host.chg.size() * sizeof(mmff94_chg_entry_t) + 256;
    total += host.bndk.size() * sizeof(mmff94_bndk_entry_t) + 256;
    total += host.hl.size() * sizeof(mmff94_hl_entry_t) + 256;
    total += host.covrad.size() * sizeof(mmff94_covrad_entry_t) + 256;

    if (!out.arena.ensure_capacity(total)) {
        logger_t const fallback;
        (log != nullptr ? *log : fallback)
            .error("upload_mmff94_tables_to_device: device arena allocation ({} bytes) failed", total);
        return;
    }
    std::byte *const arena_base = out.arena.data();

    std::size_t offset = 0;

#if defined(__CUDACC__)
    // Resolve device addresses of __constant__ symbols so they can be handed out as `T const *` in
    // the view spans.
    void *prop_dev = nullptr, *def_dev = nullptr, *covrad_dev = nullptr;
    void *hl_dev = nullptr, *bndk_dev = nullptr;
    cudaGetSymbolAddress(&prop_dev, c_mmff94_prop);
    cudaGetSymbolAddress(&def_dev, c_mmff94_def);
    cudaGetSymbolAddress(&covrad_dev, c_mmff94_covrad);
    cudaGetSymbolAddress(&hl_dev, c_mmff94_hl);
    cudaGetSymbolAddress(&bndk_dev, c_mmff94_bndk);
#else
    void *prop_dev = nullptr, *def_dev = nullptr, *covrad_dev = nullptr;
    void *hl_dev = nullptr, *bndk_dev = nullptr;
#endif

    // Large tables → global arena.
    out.view.vdw = place_table_(arena_base, offset, host.vdw);
    out.view.bond = place_table_(arena_base, offset, host.bond);
    out.view.angle = place_table_(arena_base, offset, host.angle);
    out.view.torsion = place_table_(arena_base, offset, host.torsion);
    out.view.oop = place_table_(arena_base, offset, host.oop);
    out.view.stbn = place_table_(arena_base, offset, host.stbn);
    out.view.dfsb = place_table_(arena_base, offset, host.dfsb);
    out.view.pbci = place_table_(arena_base, offset, host.pbci);
    out.view.chg = place_table_(arena_base, offset, host.chg);

    // Small tables → __constant__ (with global fallback).
    out.view.prop = place_const_or_global_(arena_base, offset, host.prop, prop_dev, mmff94_prop_const_cap_k);
    out.view.def = place_const_or_global_(arena_base, offset, host.def, def_dev, mmff94_def_const_cap_k);
    out.view.covrad = place_const_or_global_(arena_base, offset, host.covrad, covrad_dev, mmff94_covrad_const_cap_k);
    out.view.hl = place_const_or_global_(arena_base, offset, host.hl, hl_dev, mmff94_hl_const_cap_k);
    out.view.bndk = place_const_or_global_(arena_base, offset, host.bndk, bndk_dev, mmff94_bndk_const_cap_k);

    out.view.vdw_power = host.vdw_power;
    out.view.vdw_b = host.vdw_b;
    out.view.vdw_beta = host.vdw_beta;
    out.view.vdw_darad = host.vdw_darad;
    out.view.vdw_daeps = host.vdw_daeps;
    out.view.max_atom_type = host.max_atom_type;

    // Ensure all copies completed before returning — caller assumes device-side lookups are valid
    // immediately.
    cudaDeviceSynchronize();
}

} // namespace smallfield::parameters
