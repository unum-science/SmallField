/**
 *  @file src/smallfield/preprocess/wave_slice_pool.hpp
 *  @brief Unified per-wave-slice pool: one pinned host allocation, and one walk that carves every
 *  per-molecule array out of it as a subspan.
 *
 *  One pinned host allocation carries every per-molecule array, so the launch path issues a single
 *  `cudaMemcpyAsync` instead of one call per field — a multi-vector layout would copy each field through
 *  pageable memory, which silently degrades to a synchronous transfer.
 *
 *  Flow:
 *    1. `prepare_molecule_batch_into` accumulates a `wave_slice_shape_t` from the per-molecule sizes,
 *       walks it once over an empty cursor to learn what the slice spans, resizes one pinned host
 *       buffer, and walks it again to memcpy per-molecule arrays into the subspans it hands back.
 *    2. `launch_gpu_pipeline` issues a single `cudaMemcpyAsync` of `wave_slice_view::uploaded` into a
 *       per-GPU persistent device mirror owned by `context_t`, then runs the same walk over the device
 *       pool to reach every region the kernels read and write.
 */

#pragma once

#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <vector> // `std::vector`

#include <cuda_runtime.h>

#include "smallfield/api/conformer_status.hpp"
#include "smallfield/preprocess/device_capacities.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/mmff94_bounds.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/storage.hpp"
#include "smallfield/types/views.hpp"
#include "smallfield/utilities.hpp"

namespace smallfield::preprocess {

using smallfield::api::conformer_status_t;

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

using smallfield::types::carve;
using smallfield::types::pool_cursor;
using smallfield::types::pool_element_t;

#pragma region Shape

/**
 *  @brief Per-molecule element counts of every region a slice carries.
 *
 *  Packed into one cache-line-aligned struct so the prefix scan over per-molecule sizes is one
 *  pass over a single struct rather than a dozen parallel accumulator chains. The device-filled
 *  regions contribute their capacity rather than their eventual content, since that is what the
 *  host reserves for them. The `valid` field doubles as the dense-output compaction counter, and is
 *  the molecule count every walk below reads.
 *
 *  The two N × N regions lead because they are the only counts quadratic in the atom count, and
 *  a `std::uint32_t` running total of them wraps inside one wave: 104 atoms is 10816 elements per
 *  molecule, so the sum reaches 2^32 at 397094 molecules, under the 400000-molecule wave clamp.
 *  Every remaining region is linear and bounded by `max_mmff_interactions_per_mol_k` per molecule,
 *  which `max_molecules_per_batch_k` covers with room to spare.
 */
struct alignas(64) per_mol_sizes_t {
    /** N × N floats, one row-major bounds matrix per molecule. */
    std::uint64_t bounds;
    /** N × N bytes, one hop-count matrix per molecule. */
    std::uint64_t bond_distance;
    std::uint32_t chirals;
    std::uint32_t distance_constraints;
    /** Per-atom regions: types, charges, VdW params, and both CSR offset arrays. */
    std::uint32_t atoms;
    std::uint32_t etk_interactions;
    std::uint32_t mmff_bonds;
    std::uint32_t mmff_angles;
    std::uint32_t mmff_torsions;
    std::uint32_t mmff_oops;
    std::uint32_t mmff_stretchbends;
    std::uint32_t mmff_interactions;
    std::uint32_t patch_atoms;
    std::uint32_t patch_records;
    /** 0 if the molecule failed preprocessing, 1 otherwise. */
    std::uint32_t valid;
};
static_assert(sizeof(per_mol_sizes_t) == 128, "per_mol_sizes_t must be a whole number of cache lines");

/** Accumulate one molecule's capacities into a running total. */
inline void accumulate_sizes(per_mol_sizes_t &totals, etkdg_molecule_metadata_t const &metadata,
                             device_capacities_t const &capacities) {
    std::uint32_t const num_atoms = metadata.num_atoms;
    // The bounds matrix and the hop-count matrix are the same N × N shape at different element
    // widths, so one square serves both and the two can never disagree.
    std::uint64_t const square_elements = static_cast<std::uint64_t>(num_atoms) * num_atoms;
    totals.bounds += square_elements;
    totals.chirals += metadata.num_chiral_centers;
    totals.distance_constraints += metadata.num_distance_constraints;
    totals.atoms += num_atoms;
    totals.bond_distance += square_elements;
    totals.etk_interactions += capacities.etk_interactions;
    totals.mmff_bonds += capacities.mmff_bonds;
    totals.mmff_angles += capacities.mmff_angles;
    totals.mmff_torsions += capacities.mmff_torsions;
    totals.mmff_oops += capacities.mmff_oops;
    totals.mmff_stretchbends += capacities.mmff_stretchbends;
    totals.mmff_interactions += capacities.mmff_interactions;
    totals.valid += 1;
}

/**
 *  @brief Everything the walk needs to carve a slice: the element count of every region, and the
 *  per-molecule torsion stride the CSD matcher writes at.
 *
 *  Stored by the pool instead of a table of byte offsets, because it is the smaller fact: offsets and
 *  extents both fall out of walking it, and two pools of the same shape carve identically whether the
 *  bytes behind them are pinned host memory, device memory, or not there yet.
 */
struct wave_slice_shape_t {
    per_mol_sizes_t totals {};
    /** Slots the CSD torsion matcher reserves per molecule; the slice's largest bond count. */
    std::uint32_t torsion_stride = 0;
};

#pragma endregion Shape

#pragma region Carving

/**
 *  @brief Typed subspans of one slice pool, const-propagating from the bytes behind them.
 *
 *  The template parameter is the byte type rather than a residence tag, and the two aliases below are
 *  what a reader sees: `wave_slice_view_t` over writable bytes, `wave_slice_const_view_t` over bytes
 *  nobody may write. Every member is a subspan of the same pool, so an extent never travels beside the
 *  data it measures.
 *
 *  Residence is @b not carried here, and cannot be until the one caller that walks a device allocation
 *  moves with it: `launch_gpu_pipeline` and `test/cuda/test_bond_distance_kernels.cu` both walk device
 *  bytes, while `wave_slice_pool_t::host_view` walks pinned host bytes. Nothing but the call site says
 *  which.
 */
template <typename byte_type_>
struct wave_slice_view {
    /** The element type this view hands out: `element_` itself, or `element_ const` over const bytes. */
    template <typename element_>
    using element_of = pool_element_t<byte_type_, element_>;

    /*  What the host produces and ships across PCIe: the descriptors the device kernels read, the
     *  bounds matrix, the chiral centres, the distance constraints, the per-molecule bases each kernel
     *  writes at, and the typing patch carrying the host's answer for the molecules the device typer
     *  rejects. `carve_uploaded_` is where all of them are placed.
     */
    custd::span<element_of<float>> bounds_matrices {};
    custd::span<element_of<chiral_center_descriptor_t>> chiral_centers {};
    custd::span<element_of<distance_constraint_t>> distance_constraints {};
    custd::span<element_of<molecule_descriptor_t>> descriptors {};
    custd::span<element_of<std::uint32_t>> atom_bases {};
    custd::span<element_of<mmff94_packed_offsets_t>> mmff_bases {};
    custd::span<element_of<mmff94_typing_patch_t>> typing_patches {};
    custd::span<element_of<std::uint8_t>> patch_atom_types {};
    custd::span<element_of<float>> patch_atom_charges {};
    custd::span<element_of<etkdg_molecule_metadata_t>> molecule_metadata {};
    custd::span<element_of<std::uint32_t>> input_indices {};

    /*  What the device preprocessing kernels produce, and nothing else: the ETK torsions and their
     *  per-atom CSR, every MMFF term array and its CSR, the per-atom types and charges, and the
     *  hop-count matrix. `carve_device_only_` is where all of them are placed.
     */
    custd::span<element_of<etk_torsion_descriptor_t>> etk_torsions {};
    custd::span<element_of<std::uint32_t>> torsion_counts {};
    custd::span<element_of<etk_atom_interaction_t>> etk_interactions {};
    custd::span<element_of<etk_atom_offsets_t>> etk_atom_offsets {};
    custd::span<element_of<std::uint8_t>> atom_types {};
    custd::span<element_of<std::uint8_t>> typing_status {};
    custd::span<element_of<mmff94_atom_vdw_params_t>> mmff_atom_vdw {};
    custd::span<element_of<float>> mmff_atom_charge {};
    custd::span<element_of<mmff94_bond_t>> mmff_bonds {};
    custd::span<element_of<mmff94_angle_t>> mmff_angles {};
    custd::span<element_of<mmff94_torsion_t>> mmff_torsions {};
    custd::span<element_of<mmff94_oop_t>> mmff_oops {};
    custd::span<element_of<mmff94_stretchbend_t>> mmff_stretchbends {};
    custd::span<element_of<mmff94_atom_interaction_t>> mmff_interactions {};
    custd::span<element_of<mmff94_atom_offsets_t>> mmff_atom_offsets {};
    custd::span<element_of<std::uint8_t>> bond_distance {};

    /** The prefix one `cudaMemcpyAsync` carries H→D: every uploaded region, and nothing after them. */
    custd::span<byte_type_> uploaded {};
};

/** Writable view, held by whoever fills the pool or launches kernels against its device mirror. */
using wave_slice_view_t = wave_slice_view<std::byte>;

/** Read-only view, which is all a `wave_slice_pool_t const` can hand out. */
using wave_slice_const_view_t = wave_slice_view<std::byte const>;

/**
 *  @brief Carve the regions the host produces and PCIe carries, and nothing else.
 *
 *  Which of this and `carve_device_only_` a region is written in @b is the uploaded-versus-device-only
 *  split: there is no position between them for a region to land in by accident, and moving one across
 *  changes what crosses the bus. `cursor.bytes` on return is the prefix the upload copies.
 *
 *  @param[in] shape Element counts every carve draws from.
 *  @param[inout] cursor Walk position, advanced past every region placed here.
 *  @param[out] view Receives the subspans, or empty ones when the cursor holds no pool.
 */
template <typename byte_type_>
void carve_uploaded_(wave_slice_shape_t const &shape, pool_cursor<byte_type_> &cursor,
                     wave_slice_view<byte_type_> &view) noexcept {
    per_mol_sizes_t const &totals = shape.totals;
    std::size_t const molecules = totals.valid;
    view.bounds_matrices = carve<float>(cursor, totals.bounds);
    view.chiral_centers = carve<chiral_center_descriptor_t>(cursor, totals.chirals);
    view.distance_constraints = carve<distance_constraint_t>(cursor, totals.distance_constraints);
    view.descriptors = carve<molecule_descriptor_t>(cursor, molecules);
    view.atom_bases = carve<std::uint32_t>(cursor, molecules);
    view.mmff_bases = carve<mmff94_packed_offsets_t>(cursor, molecules);
    view.typing_patches = carve<mmff94_typing_patch_t>(cursor, totals.patch_records);
    view.patch_atom_types = carve<std::uint8_t>(cursor, totals.patch_atoms);
    view.patch_atom_charges = carve<float>(cursor, totals.patch_atoms);
    view.molecule_metadata = carve<etkdg_molecule_metadata_t>(cursor, molecules);
    view.input_indices = carve<std::uint32_t>(cursor, molecules);
}

/**
 *  @brief Carve the regions the preprocessing kernels fill in place, which never cross PCIe.
 *
 *  Every region here sits past the uploaded prefix, so the pinned host buffer reserves no room for it
 *  and only the device mirror is walked far enough to reach one.
 *
 *  @param[in] shape Element counts every carve draws from.
 *  @param[inout] cursor Walk position, advanced past every region placed here.
 *  @param[out] view Receives the subspans, or empty ones when the cursor holds no pool.
 */
template <typename byte_type_>
void carve_device_only_(wave_slice_shape_t const &shape, pool_cursor<byte_type_> &cursor,
                        wave_slice_view<byte_type_> &view) noexcept {
    per_mol_sizes_t const &totals = shape.totals;
    std::size_t const molecules = totals.valid;
    view.etk_torsions = carve<etk_torsion_descriptor_t>(cursor, molecules * shape.torsion_stride);
    view.torsion_counts = carve<std::uint32_t>(cursor, molecules);
    view.etk_interactions = carve<etk_atom_interaction_t>(cursor, totals.etk_interactions);
    view.etk_atom_offsets = carve<etk_atom_offsets_t>(cursor, totals.atoms);
    view.atom_types = carve<std::uint8_t>(cursor, totals.atoms);
    view.typing_status = carve<std::uint8_t>(cursor, molecules);
    view.mmff_atom_vdw = carve<mmff94_atom_vdw_params_t>(cursor, totals.atoms);
    view.mmff_atom_charge = carve<float>(cursor, totals.atoms);
    view.mmff_bonds = carve<mmff94_bond_t>(cursor, totals.mmff_bonds);
    view.mmff_angles = carve<mmff94_angle_t>(cursor, totals.mmff_angles);
    view.mmff_torsions = carve<mmff94_torsion_t>(cursor, totals.mmff_torsions);
    view.mmff_oops = carve<mmff94_oop_t>(cursor, totals.mmff_oops);
    view.mmff_stretchbends = carve<mmff94_stretchbend_t>(cursor, totals.mmff_stretchbends);
    view.mmff_interactions = carve<mmff94_atom_interaction_t>(cursor, totals.mmff_interactions);
    view.mmff_atom_offsets = carve<mmff94_atom_offsets_t>(cursor, totals.atoms);
    view.bond_distance = carve<std::uint8_t>(cursor, totals.bond_distance);
}

/**
 *  @brief Carve only the regions that cross PCIe, which is all a pinned host allocation holds.
 *
 *  The device-only members come back empty, because the memory behind them does not exist on this side
 *  of the bus. Walking past the prefix over host bytes would form subspans of an allocation that ends
 *  before them.
 *
 *  @param[in] shape Element counts and torsion stride of the slice.
 *  @param[inout] cursor Walk position over the pool, or over nothing while sizing one.
 *  @return The uploaded regions and the prefix the transfer rides.
 */
template <typename byte_type_>
[[nodiscard]] wave_slice_view<byte_type_> carve_uploaded_slice(wave_slice_shape_t const &shape,
                                                               pool_cursor<byte_type_> &cursor) noexcept {
    wave_slice_view<byte_type_> view {};
    carve_uploaded_(shape, cursor, view);
    if (!cursor.pool.empty()) view.uploaded = cursor.pool.first(cursor.bytes);
    return view;
}

/**
 *  @brief Walk one slice's regions in placement order, handing out a subspan of @p cursor for each.
 *
 *  The one walk serves both jobs the pool has. Over a cursor holding nothing it hands back empty spans
 *  and leaves `cursor.bytes` at what the allocation must span; over a cursor holding that allocation it
 *  hands back the regions themselves. Every carve is aligned to `pool_alignment_k`, which is what lets
 *  a device kernel reach any region as the pool base plus an offset.
 *
 *  Only a device mirror reserves this much. The pinned host buffer stops at the uploaded prefix, so it
 *  is walked by `carve_uploaded_slice` instead.
 *
 *  @param[in] shape Element counts and torsion stride of the slice.
 *  @param[inout] cursor Walk position over the pool, or over nothing while sizing one.
 *  @return Every region of the slice, plus the uploaded prefix the transfer rides.
 */
template <typename byte_type_>
[[nodiscard]] wave_slice_view<byte_type_> carve_wave_slice(wave_slice_shape_t const &shape,
                                                           pool_cursor<byte_type_> &cursor) noexcept {
    wave_slice_view<byte_type_> view = carve_uploaded_slice(shape, cursor);
    carve_device_only_(shape, cursor, view);
    return view;
}

/**
 *  @brief Bytes a slice of this shape spans, from the same walk that carves it over nothing.
 *
 *  This is what the device mirror has to reserve before anything can be carved out of it.
 *
 *  @param[in] shape Element counts and torsion stride of the slice.
 */
inline std::size_t wave_slice_bytes(wave_slice_shape_t const &shape) noexcept {
    pool_cursor<std::byte> cursor {};
    [[maybe_unused]] wave_slice_view_t const unallocated = carve_wave_slice(shape, cursor);
    return cursor.bytes;
}

/**
 *  @brief Bytes the uploaded prefix spans, which is all the pinned host buffer reserves.
 *
 *  Half of the same walk, stopped where the regions stop crossing PCIe.
 *
 *  @param[in] shape Element counts and torsion stride of the slice.
 */
inline std::size_t uploaded_wave_slice_bytes(wave_slice_shape_t const &shape) noexcept {
    pool_cursor<std::byte> cursor {};
    [[maybe_unused]] wave_slice_view_t const unallocated = carve_uploaded_slice(shape, cursor);
    return cursor.bytes;
}

#pragma endregion Carving

/**
 *  @brief Per-wave-slice pool: one pinned host allocation, and the shape that carves it.
 *
 *  `prepare_molecule_batch_into` fills the pool on the CPU. The launch path uses a device-side mirror
 *  inside `launch_scratch_t`, walked from the same `wave_slice_shape_t`, and issues one
 *  `cudaMemcpyAsync` of the host buffer's uploaded prefix to that mirror.
 */
struct wave_slice_pool_t {
    pinned_bytes_t host;
    wave_slice_shape_t shape;
    /** The largest atom count actually present in this slice — an observation, not the cap.
     *  `smallfield::types::max_atoms_per_molecule_k` is the cap; conflating the two sizes a kernel stride
     *  from one wave and a caller's buffer from the other. */
    std::uint32_t observed_max_atoms = 0;
    /** Slice-local indices for every molecule rejected by the `preprocess-step`, tagged with the rejection
     *  reason. Populated by `prepare_molecule_batch_into`; consumed by the caller to stamp
     *  `conformer_status_t` per molecule and route rejected mols to the `fallback-role`. Grow-only across
     *  calls. */
    std::vector<rejected_mol_t> rejected_input_indices;

    /** Molecules the slice carries, which is the extent of every per-molecule region it holds. */
    [[nodiscard]] std::uint32_t num_molecules() const noexcept { return shape.totals.valid; }

    /** The uploaded regions this pool holds. Device-only members are empty; only the mirror has them. */
    [[nodiscard]] wave_slice_view_t host_view() noexcept {
        pool_cursor<std::byte> cursor {host.view()};
        return carve_uploaded_slice(shape, cursor);
    }
    /** The uploaded regions this pool holds. Device-only members are empty; only the mirror has them. */
    [[nodiscard]] wave_slice_const_view_t host_view() const noexcept {
        pool_cursor<std::byte const> cursor {host.view()};
        return carve_uploaded_slice(shape, cursor);
    }
};

} // namespace smallfield::preprocess
