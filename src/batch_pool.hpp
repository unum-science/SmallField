/**
 *  @file batch_pool.hpp
 *  @brief Unified per-chunk batch pool: one pinned host allocation + a
 *  shared `batch_layout_t` describing every per-molecule array.
 *
 *  Replaces the pre-existing design where `etkdg_molecule_batch_t` held 14
 *  separate `std::vector<T>` fields and the launch path did 14 pageable
 *  `cudaMemcpyAsync` calls (which silently degrade to synchronous).
 *
 *  Flow:
 *    1. Phase 2 of `prepare_molecule_batch` computes `batch_layout_t`
 *       from the per-molecule prefix-scanned sizes, resizes one pinned
 *       host buffer, and memcpys per-molecule arrays at their byte offsets.
 *    2. `launch_gpu_pipeline` issues ONE `cudaMemcpyAsync` of the host
 *       buffer into a per-GPU persistent device mirror (owned by
 *       `context_t`), then carves typed `std::span<T>` views over the
 *       device pool via the same layout.
 *
 *  We considered `cudaMallocManaged` + `cudaMemPrefetchAsync` for a
 *  truly unified pointer, but on consumer-tier PCIe Blackwell the HMM
 *  overhead and prefetch bookkeeping were 2× slower than pinned+explicit
 *  for batch sizes this project ships at.  The layout and span surface
 *  are identical — swap the backing store if Grace-Hopper-class hardware
 *  makes managed competitive.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <utility>

#include <cuda_runtime.h>

#include "etkdg_types.cuh"

namespace smallfield {

/** Grow-only RAII byte buffer, parameterized by the allocator / deallocator pair. */
template <typename Alloc, typename Dealloc>
class byte_buffer_t {
    std::byte *ptr_ = nullptr;
    std::size_t bytes_ = 0;
    [[no_unique_address]] Alloc alloc_;
    [[no_unique_address]] Dealloc dealloc_;

  public:
    byte_buffer_t() noexcept = default;
    ~byte_buffer_t() noexcept {
        if (ptr_) dealloc_(ptr_);
    }
    byte_buffer_t(byte_buffer_t const &) = delete;
    byte_buffer_t &operator=(byte_buffer_t const &) = delete;
    byte_buffer_t(byte_buffer_t &&other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}
    byte_buffer_t &operator=(byte_buffer_t &&other) noexcept {
        if (this != &other) {
            if (ptr_) dealloc_(ptr_);
            ptr_ = std::exchange(other.ptr_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }

    /** Grow to at least @p bytes. No-op if already large enough. Contents may be invalidated on growth. */
    bool ensure_capacity(std::size_t bytes) noexcept {
        if (bytes <= bytes_) return true;
        if (ptr_) dealloc_(ptr_);
        void *raw = alloc_(bytes);
        if (!raw) {
            ptr_ = nullptr;
            bytes_ = 0;
            return false;
        }
        ptr_ = static_cast<std::byte *>(raw);
        bytes_ = bytes;
        return true;
    }

    std::byte *data() noexcept { return ptr_; }
    std::byte const *data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return bytes_; }
};

struct pinned_alloc_t {
    void *operator()(std::size_t bytes) const noexcept {
        void *raw = nullptr;
        return cudaHostAlloc(&raw, bytes, cudaHostAllocDefault) == cudaSuccess ? raw : nullptr;
    }
};
struct pinned_dealloc_t {
    void operator()(void *ptr) const noexcept { cudaFreeHost(ptr); }
};

struct device_alloc_t {
    void *operator()(std::size_t bytes) const noexcept {
        void *raw = nullptr;
        return cudaMalloc(&raw, bytes) == cudaSuccess ? raw : nullptr;
    }
};
struct device_dealloc_t {
    void operator()(void *ptr) const noexcept { cudaFree(ptr); }
};

/** RAII wrapper for a grow-only pinned host allocation. */
using pinned_byte_buffer_t = byte_buffer_t<pinned_alloc_t, pinned_dealloc_t>;

/** RAII wrapper for a grow-only device allocation. */
using device_byte_buffer_t = byte_buffer_t<device_alloc_t, device_dealloc_t>;

/**
 *  @brief Byte-level layout of the 16 per-molecule sub-regions inside a
 *  `batch_pool_t`.
 *
 *  Every region is `(byte_offset, element_count)`; the byte size is derived
 *  from `count * sizeof(T)` at the call site.  All offsets are aligned to
 *  `region_alignment` (256 bytes by default — matches the CUDA alignment
 *  convention the old `launch_gpu_pipeline` used when carving its pool).
 *
 *  `bytes_uploaded` vs `bytes_total`:
 *  - Uploaded regions (offsets in `[0, bytes_uploaded)`) live in both the
 *    host buffer and the device buffer; one single `cudaMemcpyAsync` copies
 *    the uploaded prefix H→D.
 *  - Device-only regions (offsets in `[bytes_uploaded, bytes_total)`) are
 *    written by GPU kernels; the host buffer does not allocate space for
 *    them.
 *
 *  Workstream A: all regions are uploaded, so `bytes_uploaded == bytes_total`.
 */
struct batch_layout_t {
    static constexpr std::size_t region_alignment = 256;

    std::size_t bytes_uploaded = 0;
    std::size_t bytes_total = 0;

    // One entry per sub-region; `count` is in T-sized elements, NOT bytes.
    std::size_t offset_bounds = 0;
    std::size_t count_bounds = 0; // float
    std::size_t offset_chirals = 0;
    std::size_t count_chirals = 0; // chiral_center_descriptor_t
    std::size_t offset_torsions = 0;
    std::size_t count_torsions = 0; // etk_torsion_descriptor_t
    std::size_t offset_distance_cons = 0;
    std::size_t count_distance_cons = 0; // distance_constraint_t
    std::size_t offset_mmff_atom_vdw = 0;
    std::size_t count_mmff_atom_vdw = 0; // mmff94_atom_vdw_params_t (one per atom)
    std::size_t offset_mmff_atom_charge = 0;
    std::size_t count_mmff_atom_charge = 0; // float (one per atom)
    std::size_t offset_bond_distance = 0;
    std::size_t count_bond_distance = 0; // std::uint8_t (N² per molecule)
    std::size_t offset_mmff_bonds = 0;
    std::size_t count_mmff_bonds = 0; // mmff94_bond_t
    std::size_t offset_mmff_angles = 0;
    std::size_t count_mmff_angles = 0; // mmff94_angle_t
    std::size_t offset_mmff_torsions = 0;
    std::size_t count_mmff_torsions = 0; // mmff94_torsion_t
    std::size_t offset_mmff_oops = 0;
    std::size_t count_mmff_oops = 0; // mmff94_oop_t
    std::size_t offset_mmff_stretchbends = 0;
    std::size_t count_mmff_stretchbends = 0; // mmff94_stretchbend_t
    std::size_t offset_mmff_interactions = 0;
    std::size_t count_mmff_interactions = 0; // mmff94_atom_interaction_t
    std::size_t offset_mmff_atom_offsets = 0;
    std::size_t count_mmff_atom_offsets = 0; // mmff94_atom_offsets_t
    std::size_t offset_etk_interactions = 0;
    std::size_t count_etk_interactions = 0; // etk_atom_interaction_t
    std::size_t offset_etk_atom_offsets = 0;
    std::size_t count_etk_atom_offsets = 0; // etk_atom_offsets_t
    std::size_t offset_metadata = 0;
    std::size_t count_metadata = 0; // etkdg_molecule_metadata_t
    std::size_t offset_input_indices = 0;
    std::size_t count_input_indices = 0; // std::uint32_t
};

constexpr std::size_t align_up_bytes_(std::size_t value, std::size_t alignment) noexcept {
    return (value + alignment - 1) & ~(alignment - 1);
}

/**
 *  Place one region starting at @p running_offset (aligned), advance the
 *  cursor, return the aligned offset.  The caller assigns this into the
 *  layout's `offset_*` field.
 */
template <typename T>
constexpr std::size_t place_region_(std::size_t &running_offset, std::size_t count) noexcept {
    running_offset = align_up_bytes_(running_offset, batch_layout_t::region_alignment);
    std::size_t const here = running_offset;
    running_offset += count * sizeof(T);
    return here;
}

/** View a sub-region of a byte buffer as a typed span. */
template <typename T, typename Buffer>
std::span<T> span_at_(Buffer &buffer, std::size_t byte_offset, std::size_t count) noexcept {
    return {reinterpret_cast<T *>(buffer.data() + byte_offset), count};
}

template <typename T, typename Buffer>
std::span<T const> span_at_(Buffer const &buffer, std::size_t byte_offset, std::size_t count) noexcept {
    return {reinterpret_cast<T const *>(buffer.data() + byte_offset), count};
}

/**
 *  @brief Byte-level view onto a batch pool.
 *
 *  Parameterized by a raw `std::byte *` base (pinned host memory, device
 *  memory, or any mirror) plus a layout pointer.  Every sub-array is
 *  exposed as a `std::span<T>` through this view.  Same class for host
 *  and device sides because the layout is identical.
 */
struct batch_pool_view_t {
    std::byte *base = nullptr;
    batch_layout_t const *layout = nullptr;

    template <typename T>
    std::span<T> region(std::size_t byte_offset, std::size_t count) const noexcept {
        if (!base) return {};
        return {reinterpret_cast<T *>(base + byte_offset), count};
    }

    std::span<float> bounds_matrices() const noexcept {
        return region<float>(layout->offset_bounds, layout->count_bounds);
    }
    std::span<chiral_center_descriptor_t> chiral_centers() const noexcept {
        return region<chiral_center_descriptor_t>(layout->offset_chirals, layout->count_chirals);
    }
    std::span<etk_torsion_descriptor_t> etk_torsions() const noexcept {
        return region<etk_torsion_descriptor_t>(layout->offset_torsions, layout->count_torsions);
    }
    std::span<distance_constraint_t> distance_constraints() const noexcept {
        return region<distance_constraint_t>(layout->offset_distance_cons, layout->count_distance_cons);
    }
    std::span<mmff94_atom_vdw_params_t> mmff_atom_vdw() const noexcept {
        return region<mmff94_atom_vdw_params_t>(layout->offset_mmff_atom_vdw, layout->count_mmff_atom_vdw);
    }
    std::span<float> mmff_atom_charge() const noexcept {
        return region<float>(layout->offset_mmff_atom_charge, layout->count_mmff_atom_charge);
    }
    std::span<std::uint8_t> bond_distance() const noexcept {
        return region<std::uint8_t>(layout->offset_bond_distance, layout->count_bond_distance);
    }
    std::span<mmff94_bond_t> mmff_bonds() const noexcept {
        return region<mmff94_bond_t>(layout->offset_mmff_bonds, layout->count_mmff_bonds);
    }
    std::span<mmff94_angle_t> mmff_angles() const noexcept {
        return region<mmff94_angle_t>(layout->offset_mmff_angles, layout->count_mmff_angles);
    }
    std::span<mmff94_torsion_t> mmff_torsions() const noexcept {
        return region<mmff94_torsion_t>(layout->offset_mmff_torsions, layout->count_mmff_torsions);
    }
    std::span<mmff94_oop_t> mmff_oops() const noexcept {
        return region<mmff94_oop_t>(layout->offset_mmff_oops, layout->count_mmff_oops);
    }
    std::span<mmff94_stretchbend_t> mmff_stretchbends() const noexcept {
        return region<mmff94_stretchbend_t>(layout->offset_mmff_stretchbends, layout->count_mmff_stretchbends);
    }
    std::span<mmff94_atom_interaction_t> mmff_interactions() const noexcept {
        return region<mmff94_atom_interaction_t>(layout->offset_mmff_interactions, layout->count_mmff_interactions);
    }
    std::span<mmff94_atom_offsets_t> mmff_atom_offsets() const noexcept {
        return region<mmff94_atom_offsets_t>(layout->offset_mmff_atom_offsets, layout->count_mmff_atom_offsets);
    }
    std::span<etk_atom_interaction_t> etk_interactions() const noexcept {
        return region<etk_atom_interaction_t>(layout->offset_etk_interactions, layout->count_etk_interactions);
    }
    std::span<etk_atom_offsets_t> etk_atom_offsets() const noexcept {
        return region<etk_atom_offsets_t>(layout->offset_etk_atom_offsets, layout->count_etk_atom_offsets);
    }
    std::span<etkdg_molecule_metadata_t> molecule_metadata() const noexcept {
        return region<etkdg_molecule_metadata_t>(layout->offset_metadata, layout->count_metadata);
    }
    std::span<std::uint32_t> input_indices() const noexcept {
        return region<std::uint32_t>(layout->offset_input_indices, layout->count_input_indices);
    }
};

/**
 *  @brief Per-chunk batch pool: one pinned host allocation + layout.
 *
 *  `prepare_molecule_batch` fills the pool on the CPU.  The launch path
 *  uses a device-side mirror (inside `launch_scratch_t`) carved by the
 *  same `batch_layout_t`, and issues one `cudaMemcpyAsync` of the whole
 *  host buffer to the device mirror.
 */
struct batch_pool_t {
    pinned_byte_buffer_t host;
    batch_layout_t layout;
    std::uint32_t num_molecules = 0;
    std::uint32_t max_atoms_per_molecule = 0;

    batch_pool_view_t host_view() noexcept { return {host.data(), &layout}; }
    batch_pool_view_t host_view() const noexcept { return {const_cast<std::byte *>(host.data()), &layout}; }

    // Thin forwarders so call sites can keep using `pool.foo()` as with the
    // prior std::vector<T>-backed design.  Equivalent to `host_view().foo()`.
    std::span<float> bounds_matrices() noexcept { return host_view().bounds_matrices(); }
    std::span<chiral_center_descriptor_t> chiral_centers() noexcept { return host_view().chiral_centers(); }
    std::span<etk_torsion_descriptor_t> etk_torsions() noexcept { return host_view().etk_torsions(); }
    std::span<distance_constraint_t> distance_constraints() noexcept { return host_view().distance_constraints(); }
    std::span<mmff94_atom_vdw_params_t> mmff_atom_vdw() noexcept { return host_view().mmff_atom_vdw(); }
    std::span<float> mmff_atom_charge() noexcept { return host_view().mmff_atom_charge(); }
    std::span<std::uint8_t> bond_distance() noexcept { return host_view().bond_distance(); }
    std::span<mmff94_bond_t> mmff_bonds() noexcept { return host_view().mmff_bonds(); }
    std::span<mmff94_angle_t> mmff_angles() noexcept { return host_view().mmff_angles(); }
    std::span<mmff94_torsion_t> mmff_torsions() noexcept { return host_view().mmff_torsions(); }
    std::span<mmff94_oop_t> mmff_oops() noexcept { return host_view().mmff_oops(); }
    std::span<mmff94_stretchbend_t> mmff_stretchbends() noexcept { return host_view().mmff_stretchbends(); }
    std::span<mmff94_atom_interaction_t> mmff_interactions() noexcept { return host_view().mmff_interactions(); }
    std::span<mmff94_atom_offsets_t> mmff_atom_offsets() noexcept { return host_view().mmff_atom_offsets(); }
    std::span<etk_atom_interaction_t> etk_interactions() noexcept { return host_view().etk_interactions(); }
    std::span<etk_atom_offsets_t> etk_atom_offsets() noexcept { return host_view().etk_atom_offsets(); }
    std::span<etkdg_molecule_metadata_t> molecule_metadata() noexcept { return host_view().molecule_metadata(); }
    std::span<std::uint32_t> input_indices() noexcept { return host_view().input_indices(); }

    std::span<etkdg_molecule_metadata_t const> molecule_metadata() const noexcept {
        return {reinterpret_cast<etkdg_molecule_metadata_t const *>(host.data() + layout.offset_metadata),
                layout.count_metadata};
    }
    std::span<std::uint32_t const> input_indices() const noexcept {
        return {reinterpret_cast<std::uint32_t const *>(host.data() + layout.offset_input_indices),
                layout.count_input_indices};
    }
};

using batch_pool_dev_view_t = batch_pool_view_t;

} // namespace smallfield
