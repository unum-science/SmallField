/**
 *  @file test_mmff_gpu_extract_kernels.cu
 *  @brief MG-E validation: run the GPU MMFF term extractor and hand
 *  results back to the host for byte-level diff vs the CPU native
 *  extractor.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define SMALLFIELD_MMFF_TABLES_NO_RDKIT 1
#include "mmff94/mmff94_extract_device.cuh"
#include "parameters/mmff94_tables.hpp"
#include "parameters/mmff94_tables_device.cuh"
#include "mmff94/mmff94_typer_device.cuh"

namespace smallfield {

/** Opaque device-state wrapper for the end-to-end GPU MMFF extractor
 *  (type+charge kernel followed by term-extract kernel). */
struct gpu_extract_state_t {
    mmff94_tables_device_t tables;
    molecule_descriptor_t *desc_dev = nullptr;
    std::uint32_t *offsets_dev = nullptr;
    std::uint8_t *atom_types_dev = nullptr;
    float *atom_charges_dev = nullptr;
    std::uint8_t *typing_status_dev = nullptr;
    mmff94_atom_vdw_params_t *vdw_dev = nullptr;
    float *charge_out_dev = nullptr;
    mmff94_bond_t *bonds_dev = nullptr;
    mmff94_angle_t *angles_dev = nullptr;
    mmff94_torsion_t *torsions_dev = nullptr;
    mmff94_oop_t *oops_dev = nullptr;
    mmff94_stretchbend_t *stbns_dev = nullptr;
    mmff94_per_mol_counts_t *counts_dev = nullptr;
};

void run_mmff94_gpu_extract(mmff94_tables_host_t const &tables_host,
                             molecule_descriptor_t const *descriptors,
                             std::uint32_t num_mols,
                             std::uint32_t const *atom_offsets,
                             std::uint32_t total_atoms,
                             mmff94_per_mol_counts_t *out_counts,
                             std::uint8_t *out_status,
                             mmff94_bond_t *out_bonds, std::uint32_t bond_stride,
                             mmff94_angle_t *out_angles, std::uint32_t angle_stride,
                             mmff94_torsion_t *out_torsions, std::uint32_t torsion_stride,
                             mmff94_oop_t *out_oops, std::uint32_t oop_stride,
                             mmff94_stretchbend_t *out_stbns, std::uint32_t stbn_stride,
                             mmff94_atom_interaction_t *out_interactions,
                             mmff94_atom_offsets_t *out_atom_offsets) {
    mmff94_tables_device_t tables_dev;
    upload_mmff94_tables_to_device(tables_host, tables_dev);
    if (tables_dev.arena == nullptr) {
        std::fprintf(stderr, "table upload failed\n");
        return;
    }

    molecule_descriptor_t *d_desc = nullptr;
    std::uint32_t *d_off = nullptr;
    std::uint8_t *d_types = nullptr;
    float *d_charges = nullptr;
    std::uint8_t *d_status = nullptr;
    mmff94_atom_vdw_params_t *d_vdw = nullptr;
    float *d_charge_out = nullptr;
    mmff94_bond_t *d_bonds = nullptr;
    mmff94_angle_t *d_angles = nullptr;
    mmff94_torsion_t *d_tors = nullptr;
    mmff94_oop_t *d_oops = nullptr;
    mmff94_stretchbend_t *d_stbn = nullptr;
    mmff94_per_mol_counts_t *d_counts = nullptr;

    cudaMalloc(&d_desc, num_mols * sizeof(molecule_descriptor_t));
    cudaMalloc(&d_off, num_mols * sizeof(std::uint32_t));
    cudaMalloc(&d_types, total_atoms * sizeof(std::uint8_t));
    cudaMalloc(&d_charges, total_atoms * sizeof(float));
    cudaMalloc(&d_status, num_mols * sizeof(std::uint8_t));
    cudaMalloc(&d_vdw, total_atoms * sizeof(mmff94_atom_vdw_params_t));
    cudaMalloc(&d_charge_out, total_atoms * sizeof(float));
    cudaMalloc(&d_bonds, (std::size_t)num_mols * max_mmff_bonds_per_mol_k * sizeof(mmff94_bond_t));
    cudaMalloc(&d_angles, (std::size_t)num_mols * max_mmff_angles_per_mol_k * sizeof(mmff94_angle_t));
    cudaMalloc(&d_tors, (std::size_t)num_mols * max_mmff_torsions_per_mol_k * sizeof(mmff94_torsion_t));
    cudaMalloc(&d_oops, (std::size_t)num_mols * max_mmff_oops_per_mol_k * sizeof(mmff94_oop_t));
    cudaMalloc(&d_stbn, (std::size_t)num_mols * max_mmff_stretchbends_per_mol_k * sizeof(mmff94_stretchbend_t));
    cudaMalloc(&d_counts, num_mols * sizeof(mmff94_per_mol_counts_t));

    cudaMemcpy(d_desc, descriptors, num_mols * sizeof(molecule_descriptor_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, atom_offsets, num_mols * sizeof(std::uint32_t), cudaMemcpyHostToDevice);

    launch_mmff94_type_and_charge(d_desc, d_off, num_mols, tables_dev.view,
                                    d_types, d_charges, d_status);
    cudaDeviceSynchronize();

    launch_mmff94_extract_terms(d_desc, d_types, d_charges, d_off, num_mols, tables_dev.view,
                                 d_vdw, d_charge_out, d_bonds, d_angles, d_tors,
                                 d_oops, d_stbn, d_counts, d_status);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "extract kernel error: %s\n", cudaGetErrorString(err));
    }

    // CSR build kernel.
    mmff94_atom_interaction_t *d_inter = nullptr;
    mmff94_atom_offsets_t *d_aoff = nullptr;
    cudaMalloc(&d_inter, (std::size_t)num_mols * max_mmff_interactions_per_mol_k * sizeof(mmff94_atom_interaction_t));
    cudaMalloc(&d_aoff, (std::size_t)total_atoms * sizeof(mmff94_atom_offsets_t));
    launch_mmff94_build_csr(d_desc, d_off, num_mols, d_counts,
                             d_bonds, d_angles, d_tors, d_oops, d_stbn,
                             d_inter, d_aoff, d_status);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CSR build kernel error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(out_interactions, d_inter,
                (std::size_t)num_mols * max_mmff_interactions_per_mol_k * sizeof(mmff94_atom_interaction_t),
                cudaMemcpyDeviceToHost);
    cudaMemcpy(out_atom_offsets, d_aoff,
                (std::size_t)total_atoms * sizeof(mmff94_atom_offsets_t),
                cudaMemcpyDeviceToHost);
    cudaFree(d_inter); cudaFree(d_aoff);

    cudaMemcpy(out_status, d_status, num_mols * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_counts, d_counts, num_mols * sizeof(mmff94_per_mol_counts_t), cudaMemcpyDeviceToHost);
    // Copy full per-molecule strides so the caller can inspect any slot.
    cudaMemcpy(out_bonds, d_bonds,
                (std::size_t)num_mols * bond_stride * sizeof(mmff94_bond_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_angles, d_angles,
                (std::size_t)num_mols * angle_stride * sizeof(mmff94_angle_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_torsions, d_tors,
                (std::size_t)num_mols * torsion_stride * sizeof(mmff94_torsion_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_oops, d_oops,
                (std::size_t)num_mols * oop_stride * sizeof(mmff94_oop_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_stbns, d_stbn,
                (std::size_t)num_mols * stbn_stride * sizeof(mmff94_stretchbend_t), cudaMemcpyDeviceToHost);

    cudaFree(d_desc); cudaFree(d_off); cudaFree(d_types); cudaFree(d_charges);
    cudaFree(d_status); cudaFree(d_vdw); cudaFree(d_charge_out);
    cudaFree(d_bonds); cudaFree(d_angles); cudaFree(d_tors);
    cudaFree(d_oops); cudaFree(d_stbn); cudaFree(d_counts);
}

} // namespace smallfield
