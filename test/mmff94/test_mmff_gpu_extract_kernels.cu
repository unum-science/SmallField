/**
 *  @file test/mmff94/test_mmff_gpu_extract_kernels.cu
 *  @brief Validate the GPU MMFF term-extractor kernel.
 *
 *  Runs it and hands results back to the host for byte-level diff vs the CPU native extractor.
 */

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <fmt/format.h> // `fmt::println` -- host-side only; nothing below a `__global__` uses it

#include <cuda_runtime.h>

#include "smallfield/mmff94/mmff94_extract_device.cuh"
#include "smallfield/mmff94/mmff94_typer_device.cuh"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_device.cuh"

#include "test/mmff94/test_mmff_gpu_extract_kernels.hpp"

namespace sfmmff = smallfield::mmff94;
namespace sfparam = smallfield::parameters;
namespace sft = smallfield::types;

namespace smallfield::tests {

void launch_mmff94_gpu_extract_pipeline(
    sfparam::mmff94_tables_host_t const &tables_host, sft::molecule_descriptor_t const *descriptors,
    std::uint32_t num_mols, std::uint32_t const *atom_offsets, std::uint32_t total_atoms,
    sfmmff::mmff94_per_mol_counts_t *out_counts, std::uint8_t *out_status, sft::mmff94_bond_t *out_bonds,
    std::uint32_t bond_stride, sft::mmff94_angle_t *out_angles, std::uint32_t angle_stride,
    sft::mmff94_torsion_t *out_torsions, std::uint32_t torsion_stride, sft::mmff94_oop_t *out_oops,
    std::uint32_t oop_stride, sft::mmff94_stretchbend_t *out_stbns, std::uint32_t stbn_stride,
    sft::mmff94_atom_interaction_t *out_interactions, sft::mmff94_atom_offsets_t *out_atom_offsets,
    mmff94_packed_host_result_t &packed_out) {
    sfparam::mmff94_tables_device_t device_tables;
    sfparam::upload_mmff94_tables_to_device(tables_host, device_tables);
    if (device_tables.arena.data() == nullptr) {
        fmt::println(stderr, "NOTE mmff_extract_kernels field=table_upload succeeded=0");
        return;
    }

    sft::molecule_descriptor_t *descriptor_device = nullptr;
    std::uint32_t *atom_offset_device = nullptr;
    std::uint8_t *atom_types_device = nullptr;
    float *atom_charges_device = nullptr;
    std::uint8_t *typing_status_device = nullptr;
    sft::mmff94_atom_vdw_params_t *vdw_device = nullptr;
    float *charge_out_device = nullptr;
    sft::mmff94_bond_t *bond_device = nullptr;
    sft::mmff94_angle_t *angle_device = nullptr;
    sft::mmff94_torsion_t *torsion_device = nullptr;
    sft::mmff94_oop_t *oop_device = nullptr;
    sft::mmff94_stretchbend_t *stretch_bend_device = nullptr;
    sfmmff::mmff94_per_mol_counts_t *per_molecule_counts_device = nullptr;

    cudaMalloc(&descriptor_device, num_mols * sizeof(sft::molecule_descriptor_t));
    cudaMalloc(&atom_offset_device, num_mols * sizeof(std::uint32_t));
    cudaMalloc(&atom_types_device, total_atoms * sizeof(std::uint8_t));
    cudaMalloc(&atom_charges_device, total_atoms * sizeof(float));
    cudaMalloc(&typing_status_device, num_mols * sizeof(std::uint8_t));
    cudaMalloc(&vdw_device, total_atoms * sizeof(sft::mmff94_atom_vdw_params_t));
    cudaMalloc(&charge_out_device, total_atoms * sizeof(float));
    cudaMalloc(&bond_device,
               static_cast<std::size_t>(num_mols) * sft::max_mmff_bonds_per_mol_k * sizeof(sft::mmff94_bond_t));
    cudaMalloc(&angle_device,
               static_cast<std::size_t>(num_mols) * sft::max_mmff_angles_per_mol_k * sizeof(sft::mmff94_angle_t));
    cudaMalloc(&torsion_device,
               static_cast<std::size_t>(num_mols) * sft::max_mmff_torsions_per_mol_k * sizeof(sft::mmff94_torsion_t));
    cudaMalloc(&oop_device,
               static_cast<std::size_t>(num_mols) * sft::max_mmff_oops_per_mol_k * sizeof(sft::mmff94_oop_t));
    cudaMalloc(&stretch_bend_device, static_cast<std::size_t>(num_mols) * sft::max_mmff_stretchbends_per_mol_k *
                                         sizeof(sft::mmff94_stretchbend_t));
    cudaMalloc(&per_molecule_counts_device, num_mols * sizeof(sfmmff::mmff94_per_mol_counts_t));

    cudaMemcpy(descriptor_device, descriptors, num_mols * sizeof(sft::molecule_descriptor_t), cudaMemcpyHostToDevice);
    cudaMemcpy(atom_offset_device, atom_offsets, num_mols * sizeof(std::uint32_t), cudaMemcpyHostToDevice);

    sfmmff::launch_mmff94_type_and_charge(descriptor_device, atom_offset_device, num_mols, device_tables.view,
                                          atom_types_device, atom_charges_device, typing_status_device);
    cudaDeviceSynchronize();

    sfmmff::launch_mmff94_extract_terms(descriptor_device, atom_types_device, atom_charges_device, atom_offset_device,
                                        num_mols, device_tables.view, vdw_device, charge_out_device, bond_device,
                                        angle_device, torsion_device, oop_device, stretch_bend_device,
                                        per_molecule_counts_device, typing_status_device);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        fmt::println(stderr, "NOTE mmff_extract_kernels field=extract_launch cuda_error={}",
                     cudaGetErrorString(error));

    // CSR build kernel.
    sft::mmff94_atom_interaction_t *interaction_device = nullptr;
    sft::mmff94_atom_offsets_t *atom_offsets_device = nullptr;
    cudaMalloc(&interaction_device, static_cast<std::size_t>(num_mols) * sft::max_mmff_interactions_per_mol_k *
                                        sizeof(sft::mmff94_atom_interaction_t));
    cudaMalloc(&atom_offsets_device, static_cast<std::size_t>(total_atoms) * sizeof(sft::mmff94_atom_offsets_t));
    sfmmff::launch_mmff94_build_csr(descriptor_device, atom_offset_device, num_mols, per_molecule_counts_device,
                                    bond_device, angle_device, torsion_device, oop_device, stretch_bend_device,
                                    interaction_device, atom_offsets_device, typing_status_device);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
        fmt::println(stderr, "NOTE mmff_extract_kernels field=csr_build_launch cuda_error={}",
                     cudaGetErrorString(error));
    cudaMemcpy(out_interactions, interaction_device,
               static_cast<std::size_t>(num_mols) * sft::max_mmff_interactions_per_mol_k *
                   sizeof(sft::mmff94_atom_interaction_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out_atom_offsets, atom_offsets_device,
               static_cast<std::size_t>(total_atoms) * sizeof(sft::mmff94_atom_offsets_t), cudaMemcpyDeviceToHost);

    // Compaction: scan the per-molecule counts, read the totals back to size the
    // packed arrays, then compact the fixed-stride staging into them.
    sfmmff::mmff94_packed_offsets_t *packed_offsets_device = nullptr;
    sfmmff::mmff94_packed_offsets_t *packed_totals_device = nullptr;
    cudaMalloc(&packed_offsets_device, num_mols * sizeof(sfmmff::mmff94_packed_offsets_t));
    cudaMalloc(&packed_totals_device, sizeof(sfmmff::mmff94_packed_offsets_t));
    sfmmff::launch_mmff94_scan_counts(per_molecule_counts_device, num_mols, packed_offsets_device,
                                      packed_totals_device);
    cudaDeviceSynchronize();
    sfmmff::mmff94_packed_offsets_t packed_totals {};
    cudaMemcpy(&packed_totals, packed_totals_device, sizeof(packed_totals), cudaMemcpyDeviceToHost);

    packed_out.bonds.resize(packed_totals.bonds);
    packed_out.angles.resize(packed_totals.angles);
    packed_out.torsions.resize(packed_totals.torsions);
    packed_out.oops.resize(packed_totals.oops);
    packed_out.stretchbends.resize(packed_totals.stretchbends);
    packed_out.interactions.resize(packed_totals.interactions);
    packed_out.atom_offsets.resize(total_atoms);
    packed_out.metadata.resize(num_mols);

    sft::mmff94_bond_t *packed_bond_device = nullptr;
    sft::mmff94_angle_t *packed_angle_device = nullptr;
    sft::mmff94_torsion_t *packed_torsion_device = nullptr;
    sft::mmff94_oop_t *packed_oop_device = nullptr;
    sft::mmff94_stretchbend_t *packed_stretch_bend_device = nullptr;
    sft::mmff94_atom_interaction_t *packed_interaction_device = nullptr;
    sft::etkdg_molecule_metadata_t *metadata_device = nullptr;
    cudaMalloc(&packed_bond_device, (packed_totals.bonds + 1) * sizeof(sft::mmff94_bond_t));
    cudaMalloc(&packed_angle_device, (packed_totals.angles + 1) * sizeof(sft::mmff94_angle_t));
    cudaMalloc(&packed_torsion_device, (packed_totals.torsions + 1) * sizeof(sft::mmff94_torsion_t));
    cudaMalloc(&packed_oop_device, (packed_totals.oops + 1) * sizeof(sft::mmff94_oop_t));
    cudaMalloc(&packed_stretch_bend_device, (packed_totals.stretchbends + 1) * sizeof(sft::mmff94_stretchbend_t));
    cudaMalloc(&packed_interaction_device, (packed_totals.interactions + 1) * sizeof(sft::mmff94_atom_interaction_t));
    cudaMalloc(&metadata_device, num_mols * sizeof(sft::etkdg_molecule_metadata_t));
    cudaMemset(metadata_device, 0, num_mols * sizeof(sft::etkdg_molecule_metadata_t));

    sfmmff::launch_mmff94_compact_terms(
        descriptor_device, atom_offset_device, num_mols, per_molecule_counts_device, packed_offsets_device, bond_device,
        angle_device, torsion_device, oop_device, stretch_bend_device, interaction_device, packed_bond_device,
        packed_angle_device, packed_torsion_device, packed_oop_device, packed_stretch_bend_device,
        packed_interaction_device, atom_offsets_device, metadata_device, typing_status_device);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
        fmt::println(stderr, "NOTE mmff_extract_kernels field=compaction_launch cuda_error={}",
                     cudaGetErrorString(error));

    cudaMemcpy(packed_out.bonds.data(), packed_bond_device, packed_totals.bonds * sizeof(sft::mmff94_bond_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(packed_out.angles.data(), packed_angle_device, packed_totals.angles * sizeof(sft::mmff94_angle_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(packed_out.torsions.data(), packed_torsion_device,
               packed_totals.torsions * sizeof(sft::mmff94_torsion_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(packed_out.oops.data(), packed_oop_device, packed_totals.oops * sizeof(sft::mmff94_oop_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(packed_out.stretchbends.data(), packed_stretch_bend_device,
               packed_totals.stretchbends * sizeof(sft::mmff94_stretchbend_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(packed_out.interactions.data(), packed_interaction_device,
               packed_totals.interactions * sizeof(sft::mmff94_atom_interaction_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(packed_out.atom_offsets.data(), atom_offsets_device,
               static_cast<std::size_t>(total_atoms) * sizeof(sft::mmff94_atom_offsets_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(packed_out.metadata.data(), metadata_device, num_mols * sizeof(sft::etkdg_molecule_metadata_t),
               cudaMemcpyDeviceToHost);

    cudaFree(packed_offsets_device);
    cudaFree(packed_totals_device);
    cudaFree(packed_bond_device);
    cudaFree(packed_angle_device);
    cudaFree(packed_torsion_device);
    cudaFree(packed_oop_device);
    cudaFree(packed_stretch_bend_device);
    cudaFree(packed_interaction_device);
    cudaFree(metadata_device);
    cudaFree(interaction_device);
    cudaFree(atom_offsets_device);

    cudaMemcpy(out_status, typing_status_device, num_mols * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_counts, per_molecule_counts_device, num_mols * sizeof(sfmmff::mmff94_per_mol_counts_t),
               cudaMemcpyDeviceToHost);
    // Copy full per-molecule strides so the caller can inspect any slot.
    cudaMemcpy(out_bonds, bond_device, static_cast<std::size_t>(num_mols) * bond_stride * sizeof(sft::mmff94_bond_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out_angles, angle_device,
               static_cast<std::size_t>(num_mols) * angle_stride * sizeof(sft::mmff94_angle_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_torsions, torsion_device,
               static_cast<std::size_t>(num_mols) * torsion_stride * sizeof(sft::mmff94_torsion_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out_oops, oop_device, static_cast<std::size_t>(num_mols) * oop_stride * sizeof(sft::mmff94_oop_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out_stbns, stretch_bend_device,
               static_cast<std::size_t>(num_mols) * stbn_stride * sizeof(sft::mmff94_stretchbend_t),
               cudaMemcpyDeviceToHost);

    cudaFree(descriptor_device);
    cudaFree(atom_offset_device);
    cudaFree(atom_types_device);
    cudaFree(atom_charges_device);
    cudaFree(typing_status_device);
    cudaFree(vdw_device);
    cudaFree(charge_out_device);
    cudaFree(bond_device);
    cudaFree(angle_device);
    cudaFree(torsion_device);
    cudaFree(oop_device);
    cudaFree(stretch_bend_device);
    cudaFree(per_molecule_counts_device);
}

} // namespace smallfield::tests
