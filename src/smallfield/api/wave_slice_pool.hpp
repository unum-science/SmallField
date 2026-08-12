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
 *    1. `prepare_molecule_batch_into` computes `wave_slice_layout_t` from the per-molecule prefix-scanned sizes,
 *       resizes one pinned host buffer, and memcpys per-molecule arrays at their byte offsets.
 *    2. `launch_gpu_pipeline` issues a single `cudaMemcpyAsync` of the host buffer into a per-GPU persistent
 *       device mirror owned by `context_t`, then carves typed `custd::span<T>` views over the device pool via
 *       the same layout.
 */

#pragma once

#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <type_traits> // `std::conditional_t`, `std::is_const_v`
#include <vector>      // `std::vector`

#include <cuda_runtime.h>

#include "smallfield/api/conformer_status.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/mmff94_bounds.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/storage.hpp"
#include "smallfield/types/views.hpp"
#include "smallfield/utilities.hpp"

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

using smallfield::types::device_allocator_t;
using smallfield::types::device_bytes_t;
using smallfield::types::device_indices_t;
using smallfield::types::pinned_bytes_t;
using smallfield::types::reserve_outcome_t;

#pragma region Pool Regions

/** Alignment every carved region starts at, matching the CUDA convention `launch_gpu_pipeline` carves with. */
inline constexpr std::size_t pool_region_alignment_k = 256;

/**
 *  @brief Where one region of a carved pool starts, and how many elements it holds.
 *
 *  The offset and the count are one fact rather than two fields kept in step by hand, so an offset
 *  paired with a neighbouring region's count stops being spellable. `over` is the only place a region
 *  turns back into a pointer, and the count it carries becomes the extent of the span it hands out.
 */
struct pool_region_t {
    std::size_t byte_offset = 0;
    std::size_t count = 0;

    /** View this region over @p base, which must be the pool the region was placed in. */
    template <typename element_>
    [[nodiscard]] custd::span<element_> over(std::byte *base) const noexcept {
        return {reinterpret_cast<element_ *>(base + byte_offset), count};
    }

    /** The same view over a pool nobody may write, so constness travels with the bytes. */
    template <typename element_>
    [[nodiscard]] custd::span<element_ const> over(std::byte const *base) const noexcept {
        return {reinterpret_cast<element_ const *>(base + byte_offset), count};
    }
};

/** Place @p count elements of @p element_ at the next aligned offset and advance @p cursor past them. */
template <typename element_>
constexpr pool_region_t place_pool_region_(std::size_t &cursor, std::size_t count) noexcept {
    cursor = align_up_(cursor, pool_region_alignment_k);
    pool_region_t const region {.byte_offset = cursor, .count = count};
    cursor += count * sizeof(element_);
    return region;
}

#pragma endregion Pool Regions

/**
 *  @brief Byte-level layout of every per-molecule sub-region inside a `wave_slice_pool_t`.
 *
 *  Every region is one `pool_region_t`, whose offset counts @b bytes while the per-molecule offsets in
 *  `smallfield::types::etkdg_molecule_metadata_t` count @b elements. The regions are declared in the
 *  order `compute_batch_layout` places them, so a region missing from either side shows up by reading
 *  the two next to each other.
 *
 *  `bytes_uploaded` against `bytes_total`:
 *  - Uploaded regions, at offsets in `[0, bytes_uploaded)`, live in both the host buffer and the device
 *    buffer; one single `cudaMemcpyAsync` copies the uploaded prefix H→D.
 *  - Device-only regions, at offsets in `[bytes_uploaded, bytes_total)`, are written by GPU kernels; the
 *    host buffer does not allocate space for them. Reading one off a host view is out of bounds.
 */
struct wave_slice_layout_t {
    std::size_t bytes_uploaded = 0;
    std::size_t bytes_total = 0;

    /** Slots the CSD torsion matcher reserves per molecule; the batch's largest bond count. */
    std::uint32_t torsion_stride = 0;

    /*  What the host produces and ships across PCIe: the descriptors the device kernels read, the
     *  bounds matrix, the chiral centres, the distance constraints, the per-molecule bases each kernel
     *  writes at, and the typing patch carrying the host's answer for the molecules the device typer
     *  rejects. Their offsets are what `bytes_uploaded` bounds.
     */
    pool_region_t bounds {};
    pool_region_t chirals {};
    pool_region_t distance_constraints {};
    pool_region_t descriptors {};
    pool_region_t atom_bases {};
    pool_region_t mmff_bases {};
    pool_region_t typing_patches {};
    pool_region_t patch_atom_types {};
    pool_region_t patch_atom_charges {};
    pool_region_t metadata {};
    pool_region_t input_indices {};

    /*  What the device preprocessing kernels produce, and nothing else: the ETK torsions and their
     *  per-atom CSR, every MMFF term array and its CSR, the per-atom types and charges, and the
     *  hop-count matrix.
     */
    pool_region_t torsions {};
    pool_region_t torsion_counts {};
    pool_region_t etk_interactions {};
    pool_region_t etk_atom_offsets {};
    pool_region_t atom_types {};
    pool_region_t typing_status {};
    pool_region_t mmff_atom_vdw {};
    pool_region_t mmff_atom_charge {};
    pool_region_t mmff_bonds {};
    pool_region_t mmff_angles {};
    pool_region_t mmff_torsions {};
    pool_region_t mmff_oops {};
    pool_region_t mmff_stretchbends {};
    pool_region_t mmff_interactions {};
    pool_region_t mmff_atom_offsets {};
    pool_region_t bond_distance {};
};

/**
 *  @brief Typed spans over one batch pool's regions, const-propagating from the bytes it views.
 *
 *  The template parameter is the byte type rather than a residence tag, and the two specializations
 *  below are what a reader sees: `batch_pool_view_t` hands out mutable spans, `batch_pool_const_view_t`
 *  hands out spans of `element const`. Constness travels through `element_of`, so the accessors are
 *  written once for both.
 *
 *  Residence is @b not carried here, and cannot be until the one caller that builds this over a device
 *  allocation moves with it: `launch_gpu_pipeline` and `test/cuda/test_bond_distance_kernels.cu` both
 *  construct the mutable specialization over device bytes, while `wave_slice_pool_t::host_view` builds
 *  it over pinned host bytes. Nothing but the call site says which.
 */
template <typename byte_type_>
struct batch_pool_view {
    byte_type_ *base = nullptr;
    wave_slice_layout_t const *layout = nullptr;

    /** The element type this view hands out: `element_` itself, or `element_ const` over const bytes. */
    template <typename element_>
    using element_of = std::conditional_t<std::is_const_v<byte_type_>, element_ const, element_>;

    template <typename element_>
    custd::span<element_of<element_>> region(pool_region_t placement) const noexcept {
        if (!base) return {};
        return placement.over<element_>(base);
    }

    custd::span<element_of<float>> bounds_matrices() const noexcept { return region<float>(layout->bounds); }
    custd::span<element_of<chiral_center_descriptor_t>> chiral_centers() const noexcept {
        return region<chiral_center_descriptor_t>(layout->chirals);
    }
    custd::span<element_of<distance_constraint_t>> distance_constraints() const noexcept {
        return region<distance_constraint_t>(layout->distance_constraints);
    }
    custd::span<element_of<molecule_descriptor_t>> descriptors() const noexcept {
        return region<molecule_descriptor_t>(layout->descriptors);
    }
    custd::span<element_of<std::uint32_t>> atom_bases() const noexcept {
        return region<std::uint32_t>(layout->atom_bases);
    }
    custd::span<element_of<mmff94_packed_offsets_t>> mmff_bases() const noexcept {
        return region<mmff94_packed_offsets_t>(layout->mmff_bases);
    }
    custd::span<element_of<mmff94_typing_patch_t>> typing_patches() const noexcept {
        return region<mmff94_typing_patch_t>(layout->typing_patches);
    }
    custd::span<element_of<std::uint8_t>> patch_atom_types() const noexcept {
        return region<std::uint8_t>(layout->patch_atom_types);
    }
    custd::span<element_of<float>> patch_atom_charges() const noexcept {
        return region<float>(layout->patch_atom_charges);
    }
    custd::span<element_of<etkdg_molecule_metadata_t>> molecule_metadata() const noexcept {
        return region<etkdg_molecule_metadata_t>(layout->metadata);
    }
    custd::span<element_of<std::uint32_t>> input_indices() const noexcept {
        return region<std::uint32_t>(layout->input_indices);
    }

    custd::span<element_of<etk_torsion_descriptor_t>> etk_torsions() const noexcept {
        return region<etk_torsion_descriptor_t>(layout->torsions);
    }
    custd::span<element_of<std::uint32_t>> torsion_counts() const noexcept {
        return region<std::uint32_t>(layout->torsion_counts);
    }
    custd::span<element_of<etk_atom_interaction_t>> etk_interactions() const noexcept {
        return region<etk_atom_interaction_t>(layout->etk_interactions);
    }
    custd::span<element_of<etk_atom_offsets_t>> etk_atom_offsets() const noexcept {
        return region<etk_atom_offsets_t>(layout->etk_atom_offsets);
    }
    custd::span<element_of<std::uint8_t>> atom_types() const noexcept {
        return region<std::uint8_t>(layout->atom_types);
    }
    custd::span<element_of<std::uint8_t>> typing_status() const noexcept {
        return region<std::uint8_t>(layout->typing_status);
    }
    custd::span<element_of<mmff94_atom_vdw_params_t>> mmff_atom_vdw() const noexcept {
        return region<mmff94_atom_vdw_params_t>(layout->mmff_atom_vdw);
    }
    custd::span<element_of<float>> mmff_atom_charge() const noexcept { return region<float>(layout->mmff_atom_charge); }
    custd::span<element_of<mmff94_bond_t>> mmff_bonds() const noexcept {
        return region<mmff94_bond_t>(layout->mmff_bonds);
    }
    custd::span<element_of<mmff94_angle_t>> mmff_angles() const noexcept {
        return region<mmff94_angle_t>(layout->mmff_angles);
    }
    custd::span<element_of<mmff94_torsion_t>> mmff_torsions() const noexcept {
        return region<mmff94_torsion_t>(layout->mmff_torsions);
    }
    custd::span<element_of<mmff94_oop_t>> mmff_oops() const noexcept { return region<mmff94_oop_t>(layout->mmff_oops); }
    custd::span<element_of<mmff94_stretchbend_t>> mmff_stretchbends() const noexcept {
        return region<mmff94_stretchbend_t>(layout->mmff_stretchbends);
    }
    custd::span<element_of<mmff94_atom_interaction_t>> mmff_interactions() const noexcept {
        return region<mmff94_atom_interaction_t>(layout->mmff_interactions);
    }
    custd::span<element_of<mmff94_atom_offsets_t>> mmff_atom_offsets() const noexcept {
        return region<mmff94_atom_offsets_t>(layout->mmff_atom_offsets);
    }
    custd::span<element_of<std::uint8_t>> bond_distance() const noexcept {
        return region<std::uint8_t>(layout->bond_distance);
    }
};

/** Writable view, held by whoever fills the pool or launches kernels against its device mirror. */
using batch_pool_view_t = batch_pool_view<std::byte>;

/** Read-only view, which is all a `wave_slice_pool_t const` can hand out. */
using batch_pool_const_view_t = batch_pool_view<std::byte const>;

/**
 *  @brief Per-wave-slice pool: one pinned host allocation + layout.
 *
 *  `prepare_molecule_batch_into` fills the pool on the CPU. The launch path uses a device-side mirror inside
 *  `launch_scratch_t`, carved by the same `wave_slice_layout_t`, and issues one `cudaMemcpyAsync` of the
 *  whole host buffer to the device mirror.
 */
struct wave_slice_pool_t {
    pinned_bytes_t host;
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
    batch_pool_const_view_t host_view() const noexcept { return {host.data(), &layout}; }
};

} // namespace smallfield::api
