/**
 *  @file bench_mmff_gpu_kernels.cu
 *  @brief Device-side launcher for the MMFF GPU kernel trio.  Times
 *  each kernel with `cudaEvent`s and reports the totals back to the
 *  host benchmark driver.  Drops the output buffers back to the host
 *  only for the first molecule (smoke-check) to avoid ≥10 GB D2H on
 *  large batches.
 */

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define SMALLFIELD_MMFF_TABLES_NO_RDKIT 1
#include "mmff94_extract_device.cuh"
#include "mmff94_tables.hpp"
#include "mmff94_tables_device.cuh"
#include "mmff94_typer_device.cuh"

namespace smallfield {

void bench_run_gpu_mmff(mmff94_tables_host_t const &tables_host,
                         molecule_descriptor_t const *descriptors,
                         std::uint32_t num_mols,
                         std::uint32_t const *atom_offsets,
                         std::uint32_t total_atoms,
                         float *type_ms, float *extract_ms, float *csr_ms,
                         float *total_ms) {
    // Upload tables once.
    mmff94_tables_device_t tables_dev;
    upload_mmff94_tables_to_device(tables_host, tables_dev);
    if (tables_dev.arena == nullptr) {
        std::fprintf(stderr, "table upload failed\n");
        *type_ms = *extract_ms = *csr_ms = *total_ms = -1.0f;
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
    mmff94_atom_interaction_t *d_inter = nullptr;
    mmff94_atom_offsets_t *d_aoff = nullptr;

    std::size_t const per_bond = (std::size_t)num_mols * max_mmff_bonds_per_mol_k * sizeof(mmff94_bond_t);
    std::size_t const per_angle = (std::size_t)num_mols * max_mmff_angles_per_mol_k * sizeof(mmff94_angle_t);
    std::size_t const per_tor = (std::size_t)num_mols * max_mmff_torsions_per_mol_k * sizeof(mmff94_torsion_t);
    std::size_t const per_oop = (std::size_t)num_mols * max_mmff_oops_per_mol_k * sizeof(mmff94_oop_t);
    std::size_t const per_stbn = (std::size_t)num_mols * max_mmff_stretchbends_per_mol_k * sizeof(mmff94_stretchbend_t);
    std::size_t const per_inter = (std::size_t)num_mols * max_mmff_interactions_per_mol_k * sizeof(mmff94_atom_interaction_t);

    auto const dump_mem = [&](char const *tag) {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::fprintf(stderr, "%s: GPU free=%.1f GB / total=%.1f GB\n",
                     tag, free_mem / 1.0e9, total_mem / 1.0e9);
    };
    dump_mem("before alloc");
    std::fprintf(stderr,
                 "alloc plan: desc=%.2f GB bonds=%.2f GB angles=%.2f GB "
                 "tors=%.2f GB oops=%.2f GB stbn=%.2f GB inter=%.2f GB\n",
                 num_mols * sizeof(molecule_descriptor_t) / 1.0e9,
                 per_bond / 1.0e9, per_angle / 1.0e9, per_tor / 1.0e9,
                 per_oop / 1.0e9, per_stbn / 1.0e9, per_inter / 1.0e9);

    cudaMalloc(&d_desc, num_mols * sizeof(molecule_descriptor_t));
    cudaMalloc(&d_off, num_mols * sizeof(std::uint32_t));
    cudaMalloc(&d_types, total_atoms * sizeof(std::uint8_t));
    cudaMalloc(&d_charges, total_atoms * sizeof(float));
    cudaMalloc(&d_status, num_mols * sizeof(std::uint8_t));
    cudaMalloc(&d_vdw, total_atoms * sizeof(mmff94_atom_vdw_params_t));
    cudaMalloc(&d_charge_out, total_atoms * sizeof(float));
    cudaMalloc(&d_bonds, per_bond);
    cudaMalloc(&d_angles, per_angle);
    cudaMalloc(&d_tors, per_tor);
    cudaMalloc(&d_oops, per_oop);
    cudaMalloc(&d_stbn, per_stbn);
    cudaMalloc(&d_counts, num_mols * sizeof(mmff94_per_mol_counts_t));
    cudaMalloc(&d_inter, per_inter);
    cudaMalloc(&d_aoff, total_atoms * sizeof(mmff94_atom_offsets_t));
    cudaError_t const alloc_err = cudaGetLastError();
    if (alloc_err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(alloc_err));
        *type_ms = *extract_ms = *csr_ms = *total_ms = -1.0f;
        return;
    }

    // H2D: descriptors + offsets.
    cudaMemcpy(d_desc, descriptors, num_mols * sizeof(molecule_descriptor_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, atom_offsets, num_mols * sizeof(std::uint32_t), cudaMemcpyHostToDevice);

    cudaEvent_t e_start, e_after_type, e_after_extract, e_after_csr;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_after_type);
    cudaEventCreate(&e_after_extract);
    cudaEventCreate(&e_after_csr);

    // Warm-up the driver (fair comparison — first kernel launch has
    // some one-shot setup that isn't part of the steady-state cost).
    launch_mmff94_type_and_charge(d_desc, d_off, num_mols, tables_dev.view,
                                   d_types, d_charges, d_status);
    cudaDeviceSynchronize();

    cudaEventRecord(e_start);
    launch_mmff94_type_and_charge(d_desc, d_off, num_mols, tables_dev.view,
                                   d_types, d_charges, d_status);
    cudaEventRecord(e_after_type);
    launch_mmff94_extract_terms(d_desc, d_types, d_charges, d_off, num_mols, tables_dev.view,
                                 d_vdw, d_charge_out, d_bonds, d_angles, d_tors,
                                 d_oops, d_stbn, d_counts, d_status);
    cudaEventRecord(e_after_extract);
    launch_mmff94_build_csr(d_desc, d_off, num_mols, d_counts,
                             d_bonds, d_angles, d_tors, d_oops, d_stbn,
                             d_inter, d_aoff, d_status);
    cudaEventRecord(e_after_csr);
    cudaEventSynchronize(e_after_csr);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kernel error: %s\n", cudaGetErrorString(err));
        *type_ms = *extract_ms = *csr_ms = *total_ms = -1.0f;
        goto cleanup;
    }

    cudaEventElapsedTime(type_ms, e_start, e_after_type);
    cudaEventElapsedTime(extract_ms, e_after_type, e_after_extract);
    cudaEventElapsedTime(csr_ms, e_after_extract, e_after_csr);
    cudaEventElapsedTime(total_ms, e_start, e_after_csr);

cleanup:
    cudaEventDestroy(e_start);
    cudaEventDestroy(e_after_type);
    cudaEventDestroy(e_after_extract);
    cudaEventDestroy(e_after_csr);

    cudaFree(d_desc); cudaFree(d_off); cudaFree(d_types); cudaFree(d_charges);
    cudaFree(d_status); cudaFree(d_vdw); cudaFree(d_charge_out);
    cudaFree(d_bonds); cudaFree(d_angles); cudaFree(d_tors);
    cudaFree(d_oops); cudaFree(d_stbn); cudaFree(d_counts);
    cudaFree(d_inter); cudaFree(d_aoff);
}

} // namespace smallfield
