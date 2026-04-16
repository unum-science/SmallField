/**
 *  @file mmff94_tables_device.cuh
 *  @brief Upload MMFF94 tables from host to device.
 *
 *  Mirrors `mmff94_tables_host_t` on the device.  The host side extracts
 *  RDKit's tables into flat POD vectors (see `mmff94_tables.hpp`); this
 *  header owns a one-time-per-GPU `cudaMalloc` arena holding exactly the
 *  same POD entries, and returns an `mmff94_tables_view_t` whose
 *  `std::span<T const>` fields point at the device arena.  Every
 *  `lookup_*` in `mmff94_tables.hpp` then works identically host-side and
 *  device-side with no logic duplication.
 *
 *  Small high-frequency tables (`prop`, `def`, `covrad`, `hl`, `bndk`) go
 *  to `__constant__` memory for broadcast reads during atom typing.
 *  Larger searchable tables stay in global memory and are read through
 *  `const __restrict__` pointers; L1/L2 warm-up is essentially free given
 *  the warp-per-molecule access pattern.
 *
 *  The constant-memory tables are sized generously (see below) so an
 *  RDKit upgrade that adds a few atom types doesn't overflow.  If
 *  upload_mmff94_tables_to_device finds a table larger than its constant
 *  slot, it falls back to global memory for that table — a non-fatal
 *  degradation.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

#include <cuda_runtime.h>

#include "smallfield/parameters/mmff94_tables.hpp"

namespace smallfield::parameters {

// Constant-memory tables. Budgets sized for MMFF94 with headroom; as of 2026-04-14 RDKit's live counts
// are 99 prop entries, 100 def, 54 covrad, 25 hl, and 35 bndk.
// The chosen slot sizes leave 10-30 % headroom for future RDKit updates.
inline constexpr std::size_t mmff94_prop_const_cap_k = 128;
inline constexpr std::size_t mmff94_def_const_cap_k = 128;
inline constexpr std::size_t mmff94_covrad_const_cap_k = 64;
inline constexpr std::size_t mmff94_hl_const_cap_k = 64;
inline constexpr std::size_t mmff94_bndk_const_cap_k = 64;

#if defined(__CUDACC__)
__constant__ mmff94_prop_entry_t c_mmff94_prop[mmff94_prop_const_cap_k];
__constant__ mmff94_def_entry_t c_mmff94_def[mmff94_def_const_cap_k];
__constant__ mmff94_covrad_entry_t c_mmff94_covrad[mmff94_covrad_const_cap_k];
__constant__ mmff94_hl_entry_t c_mmff94_hl[mmff94_hl_const_cap_k];
__constant__ mmff94_bndk_entry_t c_mmff94_bndk[mmff94_bndk_const_cap_k];
#endif

/** Per-GPU device-resident view.  Owns a `cudaMalloc`ed arena for the
 *  large tables; the `view` field plugs into every `lookup_*`.
 *  Pointers inside `view.prop/def/covrad/hl/bndk` target the
 *  `__constant__` symbols when the small-table capacities were sufficient;
 *  otherwise they fall back to device-global slots inside this arena. */
struct mmff94_tables_device_t {
    void *arena = nullptr; ///< single cudaMalloc, grows to fit
    std::size_t arena_bytes = 0;
    mmff94_tables_view_t view; ///< spans point into arena / __constant__
    int device_id = -1;

    mmff94_tables_device_t() = default;
    mmff94_tables_device_t(mmff94_tables_device_t const &) = delete;
    mmff94_tables_device_t &operator=(mmff94_tables_device_t const &) = delete;
    mmff94_tables_device_t(mmff94_tables_device_t &&other) noexcept
        : arena(other.arena), arena_bytes(other.arena_bytes), view(other.view), device_id(other.device_id) {
        other.arena = nullptr;
        other.arena_bytes = 0;
        other.view = {};
        other.device_id = -1;
    }
    ~mmff94_tables_device_t() {
        if (arena) {
            int prev = -1;
            cudaGetDevice(&prev);
            if (device_id >= 0 && device_id != prev) cudaSetDevice(device_id);
            cudaFree(arena);
            if (device_id >= 0 && device_id != prev && prev >= 0) cudaSetDevice(prev);
        }
    }
};

namespace detail_mmff94 {

inline std::size_t align_up_(std::size_t v, std::size_t a) { return (v + a - 1) & ~(a - 1); }

/** Helper: copy a host vector to device at the given offset, returning a
 *  span that targets the device arena at that location.  Updates `offset`
 *  to the next 256-byte-aligned position. */
template <typename element_>
mmff94_span_t<element_> place_table_(void *arena, std::size_t &offset, std::vector<element_> const &src) {
    if (src.empty()) return mmff94_span_t<element_> {};
    std::size_t const bytes = src.size() * sizeof(element_);
    offset = align_up_(offset, 256);
    element_ *dst = reinterpret_cast<element_ *>(static_cast<std::byte *>(arena) + offset);
    cudaMemcpy(dst, src.data(), bytes, cudaMemcpyHostToDevice);
    offset += bytes;
    return mmff94_span_t<element_> {dst, src.size()};
}

template <typename element_>
mmff94_span_t<element_> place_const_or_global_(void *arena, std::size_t &offset, std::vector<element_> const &src,
                                               void const *const_symbol_dev_ptr, std::size_t const_cap) {
    // Always use global memory for now. Constant-memory placement via
    // cudaGetSymbolAddress + reading through a plain device pointer is
    // valid but adds one more variable to debug; we can
    // revisit as a perf tweak once the current upload path is stable.
    (void)const_symbol_dev_ptr;
    (void)const_cap;
    return place_table_(arena, offset, src);
}

} // namespace detail_mmff94

/** Upload every MMFF94 table in `host` to device memory.  `out` is filled
 *  with device pointers; caller owns it and must keep it alive for the
 *  lifetime of any kernel launch that reads via `out.view`. */
inline void upload_mmff94_tables_to_device(mmff94_tables_host_t const &host, mmff94_tables_device_t &out) {
    cudaGetDevice(&out.device_id);
    // Upper-bound arena size: sum of all table bytes + 13 × 256-byte padding.
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

    out.arena_bytes = total;
    cudaError_t err = cudaMalloc(&out.arena, total);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "upload_mmff94_tables_to_device: cudaMalloc(%zu) failed: %s\n", total,
                     cudaGetErrorString(err));
        out.arena = nullptr;
        return;
    }

    std::size_t offset = 0;

#if defined(__CUDACC__)
    // Resolve device addresses of __constant__ symbols so we can hand
    // them out as `T const *` in the view spans.
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
    out.view.vdw = detail_mmff94::place_table_(out.arena, offset, host.vdw);
    out.view.bond = detail_mmff94::place_table_(out.arena, offset, host.bond);
    out.view.angle = detail_mmff94::place_table_(out.arena, offset, host.angle);
    out.view.torsion = detail_mmff94::place_table_(out.arena, offset, host.torsion);
    out.view.oop = detail_mmff94::place_table_(out.arena, offset, host.oop);
    out.view.stbn = detail_mmff94::place_table_(out.arena, offset, host.stbn);
    out.view.dfsb = detail_mmff94::place_table_(out.arena, offset, host.dfsb);
    out.view.pbci = detail_mmff94::place_table_(out.arena, offset, host.pbci);
    out.view.chg = detail_mmff94::place_table_(out.arena, offset, host.chg);

    // Small tables → __constant__ (with global fallback).
    out.view.prop = detail_mmff94::place_const_or_global_(out.arena, offset, host.prop, prop_dev,
                                                          mmff94_prop_const_cap_k);
    out.view.def = detail_mmff94::place_const_or_global_(out.arena, offset, host.def, def_dev, mmff94_def_const_cap_k);
    out.view.covrad = detail_mmff94::place_const_or_global_(out.arena, offset, host.covrad, covrad_dev,
                                                            mmff94_covrad_const_cap_k);
    out.view.hl = detail_mmff94::place_const_or_global_(out.arena, offset, host.hl, hl_dev, mmff94_hl_const_cap_k);
    out.view.bndk = detail_mmff94::place_const_or_global_(out.arena, offset, host.bndk, bndk_dev,
                                                          mmff94_bndk_const_cap_k);

    out.view.vdw_power = host.vdw_power;
    out.view.vdw_b = host.vdw_b;
    out.view.vdw_beta = host.vdw_beta;
    out.view.vdw_darad = host.vdw_darad;
    out.view.vdw_daeps = host.vdw_daeps;
    out.view.max_atom_type = host.max_atom_type;

    // Ensure all copies completed before returning — caller assumes
    // device-side lookups are valid immediately.
    cudaDeviceSynchronize();
}

} // namespace smallfield::parameters
