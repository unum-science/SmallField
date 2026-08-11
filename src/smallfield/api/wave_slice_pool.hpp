/**
 *  @file src/smallfield/api/wave_slice_pool.hpp
 *  @brief Unified per-wave-slice pool: one pinned host allocation + a shared `wave_slice_layout_t` describing every
 *  per-molecule array.
 *
 *  One pinned host allocation carries every per-molecule array, so the launch path issues a single
 *  `cudaMemcpyAsync` instead of one call per field — a multi-vector layout would copy each field through
 *  pageable memory, which silently degrades to a synchronous transfer.
 *
 *  Flow:
 *    1. `prepare_molecule_batch` computes `wave_slice_layout_t` from the per-molecule prefix-scanned sizes,
 *       resizes one pinned host buffer, and memcpys per-molecule arrays at their byte offsets.
 *    2. `launch_gpu_pipeline` issues a single `cudaMemcpyAsync` of the host buffer into a per-GPU persistent
 *       device mirror owned by `context_t`, then carves typed `std::span<T>` views over the device pool via
 *       the same layout.
 */

#pragma once

#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <span>    // `std::span`
#include <utility> // `std::exchange`
#include <vector>  // `std::vector`

#include <cuda_runtime.h>

#include "smallfield/api/conformer_status.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/mmff94_bounds.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::api {

/**
 *  @brief Tagged rejection record: which slice-local input was dropped, and why.
 *
 *  Every preprocess-step rejection — `invalid_smiles_k` (parse fail, addHs fail, descriptor build fail),
 *  `too_many_atoms_k`, `unsupported_element_k` (no MMFF94 atom type), `preprocess_timeout_k` (deadline
 *  tripped) — flows through one `std::vector<rejected_mol_t>` and the caller stamps `entry.reason`
 *  directly into `status_out`. Packs to 8 bytes: a 4-byte index, a 1-byte reason, 3 bytes of padding.
 */
struct rejected_mol_t {
    std::uint32_t slice_local_index;
    conformer_status_t reason;
};
using smallfield::types::chiral_center_descriptor_t;
using smallfield::types::distance_constraint_t;
using smallfield::types::etk_atom_interaction_t;
using smallfield::types::etk_atom_offsets_t;
using smallfield::types::etk_torsion_descriptor_t;
using smallfield::types::etkdg_molecule_metadata_t;
using smallfield::types::mmff94_angle_t;
using smallfield::types::mmff94_atom_interaction_t;
using smallfield::types::mmff94_atom_offsets_t;
using smallfield::types::mmff94_atom_vdw_params_t;
using smallfield::types::mmff94_bond_t;
using smallfield::types::mmff94_oop_t;
using smallfield::types::mmff94_packed_offsets_t;
using smallfield::types::mmff94_stretchbend_t;
using smallfield::types::mmff94_torsion_t;
using smallfield::types::mmff94_typing_patch_t;
using smallfield::types::molecule_descriptor_t;

/** Grow-only RAII byte buffer — used for both pinned-host and device memory pools. */
template <typename allocator_, typename deallocator_>
class byte_buffer {
    std::byte *pointer_ = nullptr;
    std::size_t bytes_ = 0;
    [[no_unique_address]] allocator_ alloc_;
    [[no_unique_address]] deallocator_ dealloc_;

  public:
    byte_buffer() noexcept = default;
    ~byte_buffer() noexcept {
        if (pointer_) dealloc_(pointer_);
    }
    byte_buffer(byte_buffer const &) = delete;
    byte_buffer &operator=(byte_buffer const &) = delete;
    byte_buffer(byte_buffer &&other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}
    byte_buffer &operator=(byte_buffer &&other) noexcept {
        if (this != &other) {
            if (pointer_) dealloc_(pointer_);
            pointer_ = std::exchange(other.pointer_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }

    /** Grow to at least @p bytes. No-op if already large enough. Contents may be invalidated on growth. */
    bool ensure_capacity(std::size_t bytes) noexcept {
        if (bytes <= bytes_) return true;
        if (pointer_) dealloc_(pointer_);
        void *raw = alloc_(bytes);
        if (!raw) {
            pointer_ = nullptr;
            bytes_ = 0;
            return false;
        }
        pointer_ = static_cast<std::byte *>(raw);
        bytes_ = bytes;
        return true;
    }

    std::byte *data() noexcept { return pointer_; }
    std::byte const *data() const noexcept { return pointer_; }
    std::size_t size() const noexcept { return bytes_; }
};

/** CUDA pinned-host memory allocator functor for `byte_buffer`. */
struct pinned_alloc_t {
    void *operator()(std::size_t bytes) const noexcept {
        void *raw = nullptr;
        return cudaHostAlloc(&raw, bytes, cudaHostAllocDefault) == cudaSuccess ? raw : nullptr;
    }
};
/** CUDA pinned-host memory deallocator functor for `byte_buffer`. */
struct pinned_dealloc_t {
    void operator()(void *pointer) const noexcept { cudaFreeHost(pointer); }
};

/** CUDA device memory allocator functor for `byte_buffer`. */
struct device_alloc_t {
    void *operator()(std::size_t bytes) const noexcept {
        void *raw = nullptr;
        return cudaMalloc(&raw, bytes) == cudaSuccess ? raw : nullptr;
    }
};
/** CUDA device memory deallocator functor for `byte_buffer`. */
struct device_dealloc_t {
    void operator()(void *pointer) const noexcept { cudaFree(pointer); }
};

/** RAII wrapper for a grow-only pinned host allocation. */
using pinned_byte_buffer_t = byte_buffer<pinned_alloc_t, pinned_dealloc_t>;

/** RAII wrapper for a grow-only device allocation. */
using device_byte_buffer_t = byte_buffer<device_alloc_t, device_dealloc_t>;

/**
 *  @brief Byte-level layout of the 16 per-molecule sub-regions inside a `wave_slice_pool_t`.
 *
 *  Every region is `(byte_offset, element_count)` — offsets here count @b bytes, while the per-molecule
 *  offsets in `smallfield::types::etkdg_molecule_metadata_t` count @b elements; the byte size is derived
 *  from `count * sizeof(T)` at the call site. All offsets are aligned to `region_alignment_k`, 256 bytes
 *  by default, matching the CUDA alignment convention `launch_gpu_pipeline` uses when carving its pool.
 *
 *  `bytes_uploaded` vs `bytes_total`:
 *  - Uploaded regions, at offsets in `[0, bytes_uploaded)`, live in both the host buffer and the device
 *    buffer; one single `cudaMemcpyAsync` copies the uploaded prefix H→D.
 *  - Device-only regions, at offsets in `[bytes_uploaded, bytes_total)`, are written by GPU kernels; the
 *    host buffer does not allocate space for them.
 *
 *  Everything the device preprocessing kernels produce is device-only: the ETK torsions and their
 *  per-atom CSR, every MMFF term array and its CSR, the per-atom types and charges, and the
 *  hop-count matrix. The host contributes the descriptors those kernels read, the bounds matrix,
 *  the chiral centres, the distance constraints, the per-molecule bases each kernel writes at, and
 *  the typing patch that carries the host's answer for the molecules the device typer rejects.
 *  Reading a device-only region off a host view is out of bounds; only the device view addresses
 *  those regions, and the host buffer stops at `bytes_uploaded`.
 */
struct wave_slice_layout_t {
    static constexpr std::size_t region_alignment_k = 256;

    std::size_t bytes_uploaded = 0;
    std::size_t bytes_total = 0;

    /** Slots the CSD torsion matcher reserves per molecule; the batch's largest bond count. */
    std::uint32_t torsion_stride = 0;

    // One entry per sub-region; `count` is in T-sized elements, @b not bytes.
    std::size_t bounds_byte_offset = 0;
    std::size_t num_bounds = 0; // float
    std::size_t chirals_byte_offset = 0;
    std::size_t num_chirals = 0; // chiral_center_descriptor_t
    std::size_t torsions_byte_offset = 0;
    std::size_t num_torsions = 0; // etk_torsion_descriptor_t
    std::size_t distance_constraints_byte_offset = 0;
    std::size_t num_distance_constraints = 0; // distance_constraint_t
    std::size_t mmff_atom_vdw_byte_offset = 0;
    std::size_t num_mmff_atom_vdw = 0; // mmff94_atom_vdw_params_t (one per atom)
    std::size_t mmff_atom_charge_byte_offset = 0;
    std::size_t num_mmff_atom_charge = 0; // float (one per atom)
    std::size_t bond_distance_byte_offset = 0;
    std::size_t num_bond_distance = 0; // std::uint8_t (N² per molecule)
    std::size_t mmff_bonds_byte_offset = 0;
    std::size_t num_mmff_bonds = 0; // mmff94_bond_t
    std::size_t mmff_angles_byte_offset = 0;
    std::size_t num_mmff_angles = 0; // mmff94_angle_t
    std::size_t mmff_torsions_byte_offset = 0;
    std::size_t num_mmff_torsions = 0; // mmff94_torsion_t
    std::size_t mmff_oops_byte_offset = 0;
    std::size_t num_mmff_oops = 0; // mmff94_oop_t
    std::size_t mmff_stretchbends_byte_offset = 0;
    std::size_t num_mmff_stretchbends = 0; // mmff94_stretchbend_t
    std::size_t mmff_interactions_byte_offset = 0;
    std::size_t num_mmff_interactions = 0; // mmff94_atom_interaction_t
    std::size_t mmff_atom_offsets_byte_offset = 0;
    std::size_t num_mmff_atom_offsets = 0; // mmff94_atom_offsets_t
    std::size_t etk_interactions_byte_offset = 0;
    std::size_t num_etk_interactions = 0; // etk_atom_interaction_t
    std::size_t etk_atom_offsets_byte_offset = 0;
    std::size_t num_etk_atom_offsets = 0; // etk_atom_offsets_t
    std::size_t metadata_byte_offset = 0;
    std::size_t num_metadata = 0; // etkdg_molecule_metadata_t
    std::size_t input_indices_byte_offset = 0;
    std::size_t num_input_indices = 0; // std::uint32_t
    std::size_t descriptors_byte_offset = 0;
    std::size_t num_descriptors = 0; // molecule_descriptor_t (one per molecule)
    std::size_t atom_bases_byte_offset = 0;
    std::size_t num_atom_bases = 0; // std::uint32_t (first atom of each molecule)
    std::size_t mmff_bases_byte_offset = 0;
    std::size_t num_mmff_bases = 0; // mmff94_packed_offsets_t (one per molecule)
    std::size_t typing_patches_byte_offset = 0;
    std::size_t num_typing_patches = 0; // mmff94_typing_patch_t
    std::size_t patch_atom_types_byte_offset = 0;
    std::size_t num_patch_atom_types = 0; // std::uint8_t (one per patched atom)
    std::size_t patch_atom_charges_byte_offset = 0;
    std::size_t num_patch_atom_charges = 0; // float (one per patched atom)
    std::size_t atom_types_byte_offset = 0;
    std::size_t num_atom_types = 0; // std::uint8_t (one per atom)
    std::size_t typing_status_byte_offset = 0;
    std::size_t num_typing_status = 0; // std::uint8_t (one per molecule)
    std::size_t torsion_counts_byte_offset = 0;
    std::size_t num_torsion_counts = 0; // std::uint32_t (one per molecule)
};

constexpr std::size_t align_up_bytes_(std::size_t value, std::size_t alignment) noexcept {
    return (value + alignment - 1) & ~(alignment - 1);
}

/**
 *  @brief Place one region starting at the aligned @p running_offset, advance the cursor, return the aligned
 *  offset. The caller assigns this into the layout's `offset_*` field.
 */
template <typename element_>
constexpr std::size_t place_region_(std::size_t &running_offset, std::size_t count) noexcept {
    running_offset = align_up_bytes_(running_offset, wave_slice_layout_t::region_alignment_k);
    std::size_t const here = running_offset;
    running_offset += count * sizeof(element_);
    return here;
}

/** View a sub-region of a byte buffer as a typed span. */
template <typename element_, typename buffer_>
std::span<element_> span_at_(buffer_ &buffer, std::size_t byte_offset, std::size_t count) noexcept {
    return {reinterpret_cast<element_ *>(buffer.data() + byte_offset), count};
}

template <typename element_, typename buffer_>
std::span<element_ const> span_at_(buffer_ const &buffer, std::size_t byte_offset, std::size_t count) noexcept {
    return {reinterpret_cast<element_ const *>(buffer.data() + byte_offset), count};
}

/**
 *  @brief Byte-level view onto a batch pool.
 */
struct batch_pool_view_t {
    std::byte *base = nullptr;
    wave_slice_layout_t const *layout = nullptr;

    template <typename element_>
    std::span<element_> region(std::size_t byte_offset, std::size_t count) const noexcept {
        if (!base) return {};
        return {reinterpret_cast<element_ *>(base + byte_offset), count};
    }

    std::span<float> bounds_matrices() const noexcept {
        return region<float>(layout->bounds_byte_offset, layout->num_bounds);
    }
    std::span<chiral_center_descriptor_t> chiral_centers() const noexcept {
        return region<chiral_center_descriptor_t>(layout->chirals_byte_offset, layout->num_chirals);
    }
    std::span<etk_torsion_descriptor_t> etk_torsions() const noexcept {
        return region<etk_torsion_descriptor_t>(layout->torsions_byte_offset, layout->num_torsions);
    }
    std::span<distance_constraint_t> distance_constraints() const noexcept {
        return region<distance_constraint_t>(layout->distance_constraints_byte_offset,
                                             layout->num_distance_constraints);
    }
    std::span<mmff94_atom_vdw_params_t> mmff_atom_vdw() const noexcept {
        return region<mmff94_atom_vdw_params_t>(layout->mmff_atom_vdw_byte_offset, layout->num_mmff_atom_vdw);
    }
    std::span<float> mmff_atom_charge() const noexcept {
        return region<float>(layout->mmff_atom_charge_byte_offset, layout->num_mmff_atom_charge);
    }
    std::span<std::uint8_t> bond_distance() const noexcept {
        return region<std::uint8_t>(layout->bond_distance_byte_offset, layout->num_bond_distance);
    }
    std::span<mmff94_bond_t> mmff_bonds() const noexcept {
        return region<mmff94_bond_t>(layout->mmff_bonds_byte_offset, layout->num_mmff_bonds);
    }
    std::span<mmff94_angle_t> mmff_angles() const noexcept {
        return region<mmff94_angle_t>(layout->mmff_angles_byte_offset, layout->num_mmff_angles);
    }
    std::span<mmff94_torsion_t> mmff_torsions() const noexcept {
        return region<mmff94_torsion_t>(layout->mmff_torsions_byte_offset, layout->num_mmff_torsions);
    }
    std::span<mmff94_oop_t> mmff_oops() const noexcept {
        return region<mmff94_oop_t>(layout->mmff_oops_byte_offset, layout->num_mmff_oops);
    }
    std::span<mmff94_stretchbend_t> mmff_stretchbends() const noexcept {
        return region<mmff94_stretchbend_t>(layout->mmff_stretchbends_byte_offset, layout->num_mmff_stretchbends);
    }
    std::span<mmff94_atom_interaction_t> mmff_interactions() const noexcept {
        return region<mmff94_atom_interaction_t>(layout->mmff_interactions_byte_offset, layout->num_mmff_interactions);
    }
    std::span<mmff94_atom_offsets_t> mmff_atom_offsets() const noexcept {
        return region<mmff94_atom_offsets_t>(layout->mmff_atom_offsets_byte_offset, layout->num_mmff_atom_offsets);
    }
    std::span<etk_atom_interaction_t> etk_interactions() const noexcept {
        return region<etk_atom_interaction_t>(layout->etk_interactions_byte_offset, layout->num_etk_interactions);
    }
    std::span<etk_atom_offsets_t> etk_atom_offsets() const noexcept {
        return region<etk_atom_offsets_t>(layout->etk_atom_offsets_byte_offset, layout->num_etk_atom_offsets);
    }
    std::span<etkdg_molecule_metadata_t> molecule_metadata() const noexcept {
        return region<etkdg_molecule_metadata_t>(layout->metadata_byte_offset, layout->num_metadata);
    }
    std::span<std::uint32_t> input_indices() const noexcept {
        return region<std::uint32_t>(layout->input_indices_byte_offset, layout->num_input_indices);
    }
    std::span<molecule_descriptor_t> descriptors() const noexcept {
        return region<molecule_descriptor_t>(layout->descriptors_byte_offset, layout->num_descriptors);
    }
    std::span<std::uint32_t> atom_bases() const noexcept {
        return region<std::uint32_t>(layout->atom_bases_byte_offset, layout->num_atom_bases);
    }
    std::span<mmff94_packed_offsets_t> mmff_bases() const noexcept {
        return region<mmff94_packed_offsets_t>(layout->mmff_bases_byte_offset, layout->num_mmff_bases);
    }
    std::span<mmff94_typing_patch_t> typing_patches() const noexcept {
        return region<mmff94_typing_patch_t>(layout->typing_patches_byte_offset, layout->num_typing_patches);
    }
    std::span<std::uint8_t> patch_atom_types() const noexcept {
        return region<std::uint8_t>(layout->patch_atom_types_byte_offset, layout->num_patch_atom_types);
    }
    std::span<float> patch_atom_charges() const noexcept {
        return region<float>(layout->patch_atom_charges_byte_offset, layout->num_patch_atom_charges);
    }
    std::span<std::uint8_t> atom_types() const noexcept {
        return region<std::uint8_t>(layout->atom_types_byte_offset, layout->num_atom_types);
    }
    std::span<std::uint8_t> typing_status() const noexcept {
        return region<std::uint8_t>(layout->typing_status_byte_offset, layout->num_typing_status);
    }
    std::span<std::uint32_t> torsion_counts() const noexcept {
        return region<std::uint32_t>(layout->torsion_counts_byte_offset, layout->num_torsion_counts);
    }
};

/**
 *  @brief Per-wave-slice pool: one pinned host allocation + layout.
 *
 *  `prepare_molecule_batch` fills the pool on the CPU. The launch path uses a device-side mirror inside
 *  `launch_scratch_t`, carved by the same `wave_slice_layout_t`, and issues one `cudaMemcpyAsync` of the
 *  whole host buffer to the device mirror.
 */
struct wave_slice_pool_t {
    pinned_byte_buffer_t host;
    wave_slice_layout_t layout;
    std::uint32_t num_molecules = 0;
    /** The largest atom count actually present in this slice — an observation, not the cap.
     *  `smallfield::types::max_atoms_per_molecule_k` is the cap; conflating the two sizes a kernel stride
     *  from one wave and a caller's buffer from the other. */
    std::uint32_t observed_max_atoms = 0;
    /** Slice-local indices for every molecule rejected by the `preprocess-step`, tagged with the rejection
     *  reason. Populated by `prepare_molecule_batch_into`; consumed by the caller to stamp
     *  `conformer_status_t` per molecule and route rejected mols to the `fallback-role`. Grow-only across
     *  calls. */
    std::vector<rejected_mol_t> rejected_input_indices;

    batch_pool_view_t host_view() noexcept { return {host.data(), &layout}; }
    batch_pool_view_t host_view() const noexcept { return {const_cast<std::byte *>(host.data()), &layout}; }

    // Thin forwarders so call sites can use `pool.foo()` directly. Equivalent to `host_view().foo()`.
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
    std::span<molecule_descriptor_t> descriptors() noexcept { return host_view().descriptors(); }
    std::span<std::uint32_t> atom_bases() noexcept { return host_view().atom_bases(); }
    std::span<mmff94_packed_offsets_t> mmff_bases() noexcept { return host_view().mmff_bases(); }
    std::span<mmff94_typing_patch_t> typing_patches() noexcept { return host_view().typing_patches(); }
    std::span<std::uint8_t> patch_atom_types() noexcept { return host_view().patch_atom_types(); }
    std::span<float> patch_atom_charges() noexcept { return host_view().patch_atom_charges(); }

    std::span<etkdg_molecule_metadata_t const> molecule_metadata() const noexcept {
        return {reinterpret_cast<etkdg_molecule_metadata_t const *>(host.data() + layout.metadata_byte_offset),
                layout.num_metadata};
    }
    std::span<std::uint32_t const> input_indices() const noexcept {
        return {reinterpret_cast<std::uint32_t const *>(host.data() + layout.input_indices_byte_offset),
                layout.num_input_indices};
    }
};

using batch_pool_dev_view_t = batch_pool_view_t;

} // namespace smallfield::api
