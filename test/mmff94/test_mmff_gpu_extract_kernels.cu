/**
 *  @file test/mmff94/test_mmff_gpu_extract_kernels.cu
 *  @brief Validate the GPU MMFF term-extractor kernel.
 *
 *  Runs it and hands results back to the host for byte-level diff vs the CPU native extractor.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <type_traits> // `std::remove_cvref_t`

#include <fmt/format.h> // `fmt::println` -- host-side only; nothing below a `__global__` uses it

#include <cuda_runtime.h>

#include "smallfield/mmff94/mmff94_extract_device.cuh"
#include "smallfield/mmff94/mmff94_typer_device.cuh"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_device.cuh"
#include "smallfield/types/storage.hpp"

#include "test/mmff94/test_mmff_gpu_extract_kernels.hpp"

namespace sfmmff = smallfield::mmff94;
namespace sfparam = smallfield::parameters;
namespace sft = smallfield::types;

namespace smallfield::tests {

/** Grow-only device storage over the legacy default stream, which is the one every kernel below rides. */
template <typename value_type_>
using device_array = sft::buffer<value_type_, sft::device_allocator_t>;

/** Every device array the extraction pipeline stages, released in stream order however the launch leaves. */
struct extract_arena_t {
    device_array<sft::molecule_descriptor_t> descriptors;
    device_array<std::uint32_t> atom_offsets;
    device_array<std::uint8_t> atom_types;
    device_array<float> atom_charges;
    device_array<std::uint8_t> typing_status;
    device_array<sft::mmff94_atom_vdw_params_t> vdw;
    device_array<float> charges_out;
    device_array<sft::mmff94_bond_t> bonds;
    device_array<sft::mmff94_angle_t> angles;
    device_array<sft::mmff94_torsion_t> torsions;
    device_array<sft::mmff94_oop_t> oops;
    device_array<sft::mmff94_stretchbend_t> stretch_bends;
    device_array<sfmmff::mmff94_per_mol_counts_t> counts_per_molecule;
    device_array<sft::mmff94_atom_interaction_t> interactions;
    device_array<sft::mmff94_atom_offsets_t> atom_offsets_lookup;
    device_array<sfmmff::mmff94_packed_offsets_t> packed_offsets;
    device_array<sfmmff::mmff94_packed_offsets_t> packed_totals;
    device_array<sft::mmff94_bond_t> packed_bonds;
    device_array<sft::mmff94_angle_t> packed_angles;
    device_array<sft::mmff94_torsion_t> packed_torsions;
    device_array<sft::mmff94_oop_t> packed_oops;
    device_array<sft::mmff94_stretchbend_t> packed_stretch_bends;
    device_array<sft::mmff94_atom_interaction_t> packed_interactions;
    device_array<sft::etkdg_molecule_metadata_t> metadata;

    /**
     *  @brief Reserve every fixed-stride array for @p num_molecules molecules over @p total_atoms atoms.
     *
     *  Each term array is sized at the per-molecule maximum stride, which is what the staging outputs
     *  mirror on the host side.
     */
    [[nodiscard]] bool resize_for(std::uint32_t num_molecules, std::size_t total_atoms) noexcept {
        bool reserved = true;
        auto reserve_ = [&](auto &array, std::size_t count) {
            reserved &= array.try_reserve(count) != sft::reserve_outcome_t::failed_k;
        };
        reserve_(descriptors, num_molecules);
        reserve_(atom_offsets, num_molecules);
        reserve_(atom_types, total_atoms);
        reserve_(atom_charges, total_atoms);
        reserve_(typing_status, num_molecules);
        reserve_(vdw, total_atoms);
        reserve_(charges_out, total_atoms);
        reserve_(bonds, std::size_t {num_molecules} * sft::max_mmff_bonds_per_mol_k);
        reserve_(angles, std::size_t {num_molecules} * sft::max_mmff_angles_per_mol_k);
        reserve_(torsions, std::size_t {num_molecules} * sft::max_mmff_torsions_per_mol_k);
        reserve_(oops, std::size_t {num_molecules} * sft::max_mmff_oops_per_mol_k);
        reserve_(stretch_bends, std::size_t {num_molecules} * sft::max_mmff_stretchbends_per_mol_k);
        reserve_(counts_per_molecule, num_molecules);
        reserve_(interactions, std::size_t {num_molecules} * sft::max_mmff_interactions_per_mol_k);
        reserve_(atom_offsets_lookup, total_atoms);
        reserve_(packed_offsets, num_molecules);
        reserve_(packed_totals, 1);
        reserve_(metadata, num_molecules);
        return reserved;
    }

    /**
     *  @brief Reserve the compacted arrays once the scan has reported what the batch actually holds.
     *
     *  Separate from `resize_for` because these extents are a device-computed result rather than a
     *  property of the batch: nothing on the host knows them until the scan kernel's totals come back.
     *  One slot past each total keeps a batch whose every count is zero from asking for an empty array.
     */
    [[nodiscard]] bool resize_packed_for(sfmmff::mmff94_packed_offsets_t const &totals) noexcept {
        bool reserved = true;
        auto reserve_ = [&](auto &array, std::size_t count) {
            reserved &= array.try_reserve(count) != sft::reserve_outcome_t::failed_k;
        };
        reserve_(packed_bonds, std::size_t {totals.bonds} + 1);
        reserve_(packed_angles, std::size_t {totals.angles} + 1);
        reserve_(packed_torsions, std::size_t {totals.torsions} + 1);
        reserve_(packed_oops, std::size_t {totals.oops} + 1);
        reserve_(packed_stretch_bends, std::size_t {totals.stretchbends} + 1);
        reserve_(packed_interactions, std::size_t {totals.interactions} + 1);
        return reserved;
    }
};

void launch_mmff94_gpu_extract_pipeline(
    sfparam::mmff94_tables_host_t const &tables_host, custd::span<sft::molecule_descriptor_t const> descriptors,
    custd::span<std::uint32_t const> atom_offsets, custd::span<sfmmff::mmff94_per_mol_counts_t> out_counts,
    custd::span<std::uint8_t> out_status, custd::span<sft::mmff94_bond_t> out_bonds,
    custd::span<sft::mmff94_angle_t> out_angles, custd::span<sft::mmff94_torsion_t> out_torsions,
    custd::span<sft::mmff94_oop_t> out_oops, custd::span<sft::mmff94_stretchbend_t> out_stbns,
    custd::span<sft::mmff94_atom_interaction_t> out_interactions,
    custd::span<sft::mmff94_atom_offsets_t> out_atom_offsets, mmff94_packed_host_result_t &packed_out) {
    sfparam::mmff94_tables_device_t device_tables;
    sfparam::upload_mmff94_tables_to_device(tables_host, device_tables, nullptr);
    if (device_tables.view.vdw.empty()) {
        fmt::println(stderr, "NOTE mmff_extract_kernels field=table_upload succeeded=0");
        return;
    }

    // The CSR offset array carries one record per atom, so it is what states the batch's atom total.
    std::uint32_t const num_mols = static_cast<std::uint32_t>(descriptors.size());
    std::size_t const total_atoms = out_atom_offsets.size();

    extract_arena_t arena;
    if (!arena.resize_for(num_mols, total_atoms)) {
        fmt::println(stderr, "NOTE mmff_extract_kernels field=device_arena molecules={} succeeded=0", num_mols);
        return;
    }
    if (!sft::upload_into(arena.descriptors.view(), descriptors, nullptr) ||
        !sft::upload_into(arena.atom_offsets.view(), atom_offsets, nullptr)) {
        fmt::println(stderr, "NOTE mmff_extract_kernels field=upload molecules={} succeeded=0", num_mols);
        return;
    }

    sfmmff::mmff94_type_and_charge_kernel<<<num_mols, sfmmff::warp_size_k, 0, nullptr>>>(
        arena.descriptors.data(), arena.atom_offsets.data(), num_mols, device_tables.view, arena.atom_types.data(),
        arena.atom_charges.data(), arena.typing_status.data());
    cudaDeviceSynchronize();

    sfmmff::mmff94_extract_terms_kernel<<<num_mols, sfmmff::warp_size_k, 0, nullptr>>>(
        arena.descriptors.data(), arena.atom_types.data(), arena.atom_charges.data(), arena.atom_offsets.data(),
        num_mols, device_tables.view, arena.vdw.data(), arena.charges_out.data(), arena.bonds.data(),
        arena.angles.data(), arena.torsions.data(), arena.oops.data(), arena.stretch_bends.data(),
        arena.counts_per_molecule.data(), arena.typing_status.data());
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        fmt::println(stderr, "NOTE mmff_extract_kernels field=extract_launch cuda_error={}", cudaGetErrorString(error));

    sfmmff::mmff94_build_csr_kernel<<<num_mols, sfmmff::warp_size_k, 0, nullptr>>>(
        arena.descriptors.data(), arena.atom_offsets.data(), num_mols, arena.counts_per_molecule.data(),
        arena.bonds.data(), arena.angles.data(), arena.torsions.data(), arena.oops.data(), arena.stretch_bends.data(),
        arena.interactions.data(), arena.atom_offsets_lookup.data(), arena.typing_status.data());
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
        fmt::println(stderr, "NOTE mmff_extract_kernels field=csr_build_launch cuda_error={}",
                     cudaGetErrorString(error));

    custd::span<sft::mmff94_atom_interaction_t const> const device_interactions = arena.interactions.view();
    custd::span<sft::mmff94_atom_offsets_t const> const device_atom_offsets = arena.atom_offsets_lookup.view();
    bool copied = sft::download_from(out_interactions, device_interactions, nullptr);
    copied &= sft::download_from(out_atom_offsets, device_atom_offsets, nullptr);

    // Compaction: scan the per-molecule counts, read the totals back to size the
    // packed arrays, then compact the fixed-stride staging into them.
    // A single block scans the whole batch, so the grid is one whatever the molecule count is.
    sfmmff::mmff94_scan_counts_kernel<<<1, sfmmff::mmff94_scan_block_k, 0, nullptr>>>(
        arena.counts_per_molecule.data(), num_mols, arena.packed_offsets.data(), arena.packed_totals.data());
    cudaDeviceSynchronize();

    sfmmff::mmff94_packed_offsets_t packed_totals {};
    custd::span<sfmmff::mmff94_packed_offsets_t const> const device_totals = arena.packed_totals.view();
    copied &= sft::download_from(custd::span<sfmmff::mmff94_packed_offsets_t> {&packed_totals, 1}, device_totals,
                                 nullptr);
    // The totals size every allocation below, so the host waits for them rather than reading a copy in flight.
    cudaStreamSynchronize(nullptr);

    packed_out.bonds.resize(packed_totals.bonds);
    packed_out.angles.resize(packed_totals.angles);
    packed_out.torsions.resize(packed_totals.torsions);
    packed_out.oops.resize(packed_totals.oops);
    packed_out.stretchbends.resize(packed_totals.stretchbends);
    packed_out.interactions.resize(packed_totals.interactions);
    packed_out.atom_offsets.resize(total_atoms);
    packed_out.metadata.resize(num_mols);

    if (!arena.resize_packed_for(packed_totals)) {
        fmt::println(stderr, "NOTE mmff_extract_kernels field=packed_arena molecules={} succeeded=0", num_mols);
        return;
    }
    cudaMemset(arena.metadata.data(), 0, arena.metadata.view().size_bytes());

    sfmmff::mmff94_compact_terms_kernel<<<num_mols, sfmmff::warp_size_k, 0, nullptr>>>(
        arena.descriptors.data(), arena.atom_offsets.data(), num_mols, arena.counts_per_molecule.data(),
        arena.packed_offsets.data(), arena.bonds.data(), arena.angles.data(), arena.torsions.data(), arena.oops.data(),
        arena.stretch_bends.data(), arena.interactions.data(), arena.packed_bonds.data(), arena.packed_angles.data(),
        arena.packed_torsions.data(), arena.packed_oops.data(), arena.packed_stretch_bends.data(),
        arena.packed_interactions.data(), arena.atom_offsets_lookup.data(), arena.metadata.data(),
        arena.typing_status.data());
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
        fmt::println(stderr, "NOTE mmff_extract_kernels field=compaction_launch cuda_error={}",
                     cudaGetErrorString(error));

    // Each packed array holds one slot past its total, so the host extent is what decides the copy.
    auto download_packed_ = [&](auto &host_array, auto const &device_source) {
        using element_t = typename std::remove_cvref_t<decltype(device_source)>::value_type;
        custd::span<element_t const> const source {device_source.data(), host_array.size()};
        copied &= sft::download_from(custd::span<element_t> {host_array}, source, nullptr);
    };
    download_packed_(packed_out.bonds, arena.packed_bonds);
    download_packed_(packed_out.angles, arena.packed_angles);
    download_packed_(packed_out.torsions, arena.packed_torsions);
    download_packed_(packed_out.oops, arena.packed_oops);
    download_packed_(packed_out.stretchbends, arena.packed_stretch_bends);
    download_packed_(packed_out.interactions, arena.packed_interactions);

    custd::span<sft::etkdg_molecule_metadata_t const> const device_metadata = arena.metadata.view();
    copied &= sft::download_from(custd::span<sft::mmff94_atom_offsets_t> {packed_out.atom_offsets}, device_atom_offsets,
                                 nullptr);
    copied &= sft::download_from(custd::span<sft::etkdg_molecule_metadata_t> {packed_out.metadata}, device_metadata,
                                 nullptr);

    custd::span<std::uint8_t const> const device_status = arena.typing_status.view();
    custd::span<sfmmff::mmff94_per_mol_counts_t const> const device_counts = arena.counts_per_molecule.view();
    copied &= sft::download_from(out_status, device_status, nullptr);
    copied &= sft::download_from(out_counts, device_counts, nullptr);
    // Full per-molecule strides, so the caller can inspect any slot.
    copied &= sft::download_from(out_bonds, custd::span<sft::mmff94_bond_t const> {arena.bonds.view()}, nullptr);
    copied &= sft::download_from(out_angles, custd::span<sft::mmff94_angle_t const> {arena.angles.view()}, nullptr);
    copied &= sft::download_from(out_torsions, custd::span<sft::mmff94_torsion_t const> {arena.torsions.view()},
                                 nullptr);
    copied &= sft::download_from(out_oops, custd::span<sft::mmff94_oop_t const> {arena.oops.view()}, nullptr);
    copied &= sft::download_from(out_stbns, custd::span<sft::mmff94_stretchbend_t const> {arena.stretch_bends.view()},
                                 nullptr);

    if (!copied) fmt::println(stderr, "NOTE mmff_extract_kernels field=download molecules={} succeeded=0", num_mols);
    cudaStreamSynchronize(nullptr);
}

} // namespace smallfield::tests
