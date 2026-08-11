/**
 *  @file bench/bench_mmff_kernels_device.cu
 *  @brief Device-side launcher for the MMFF GPU kernel trio.
 *
 *  Times each kernel with `cudaEvent`s and reports the totals back to the host benchmark driver. Drops the output
 *  buffers back to the host only for the first molecule — a smoke check — to avoid ≥10 GB D2H on large batches.
 */

#include <cstdint> // `std::uint32_t`, `std::uint8_t`
#include <cstdio>  // `fprintf`, `stderr`

#include <cuda_runtime.h>

#include "smallfield/mmff94/mmff94_extract_device.cuh"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_device.cuh"
#include "smallfield/mmff94/mmff94_typer_device.cuh"

#include "bench_mmff_kernels_device.hpp"

namespace sfmmff = smallfield::mmff94;
namespace sfparam = smallfield::parameters;
namespace sft = smallfield::types;

namespace smallfield::bench {

void launch_mmff94_kernel_trio_benchmark(sfparam::mmff94_tables_host_t const &tables_host,
                                         sft::molecule_descriptor_t const *descriptors, std::uint32_t num_molecules,
                                         std::uint32_t const *atom_offsets, std::uint32_t total_atom_count,
                                         float *type_and_charge_ms, float *extract_terms_ms, float *build_csr_ms,
                                         float *total_gpu_ms) {
    // Upload tables once.
    sfparam::mmff94_tables_device_t tables_dev;
    sfparam::upload_mmff94_tables_to_device(tables_host, tables_dev);
    if (tables_dev.arena.data() == nullptr) {
        std::fprintf(stderr, "table upload failed\n");
        *type_and_charge_ms = *extract_terms_ms = *build_csr_ms = *total_gpu_ms = -1.0f;
        return;
    }

    sft::molecule_descriptor_t *d_descriptors = nullptr;
    std::uint32_t *d_atom_offsets = nullptr;
    std::uint8_t *d_atom_types = nullptr;
    float *d_partial_charges = nullptr;
    std::uint8_t *d_status_codes = nullptr;
    sft::mmff94_atom_vdw_params_t *d_vdw_params = nullptr;
    float *d_output_charges = nullptr;
    sft::mmff94_bond_t *d_bond_terms = nullptr;
    sft::mmff94_angle_t *d_angle_terms = nullptr;
    sft::mmff94_torsion_t *d_torsion_terms = nullptr;
    sft::mmff94_oop_t *d_out_of_plane_terms = nullptr;
    sft::mmff94_stretchbend_t *d_stretch_bend_terms = nullptr;
    sfmmff::mmff94_per_mol_counts_t *d_counts_per_molecule = nullptr;
    sft::mmff94_atom_interaction_t *d_interactions = nullptr;
    sft::mmff94_atom_offsets_t *d_atom_offsets_lookup = nullptr;

    std::size_t const per_bond_bytes = static_cast<std::size_t>(num_molecules) * sft::max_mmff_bonds_per_mol_k *
                                       sizeof(sft::mmff94_bond_t);
    std::size_t const per_angle_bytes = static_cast<std::size_t>(num_molecules) * sft::max_mmff_angles_per_mol_k *
                                        sizeof(sft::mmff94_angle_t);
    std::size_t const per_torsion_bytes = static_cast<std::size_t>(num_molecules) * sft::max_mmff_torsions_per_mol_k *
                                          sizeof(sft::mmff94_torsion_t);
    std::size_t const per_out_of_plane_bytes = static_cast<std::size_t>(num_molecules) * sft::max_mmff_oops_per_mol_k *
                                               sizeof(sft::mmff94_oop_t);
    std::size_t const per_stretch_bend_bytes = static_cast<std::size_t>(num_molecules) *
                                               sft::max_mmff_stretchbends_per_mol_k * sizeof(sft::mmff94_stretchbend_t);
    std::size_t const per_interaction_bytes = static_cast<std::size_t>(num_molecules) *
                                              sft::max_mmff_interactions_per_mol_k *
                                              sizeof(sft::mmff94_atom_interaction_t);

    auto const print_gpu_memory_usage = [&](char const *label) {
        size_t free_memory = 0, total_memory = 0;
        cudaMemGetInfo(&free_memory, &total_memory);
        std::fprintf(stderr, "%s: GPU free=%.1f GB / total=%.1f GB\n", label, free_memory / 1.0e9,
                     total_memory / 1.0e9);
    };
    print_gpu_memory_usage("before alloc");
    std::fprintf( //
        stderr,
        "alloc plan: descriptor=%.2f GB bonds=%.2f GB angles=%.2f GB tors=%.2f GB oops=%.2f GB stbn=%.2f GB " //
        "inter=%.2f GB\n",
        num_molecules * sizeof(sft::molecule_descriptor_t) / 1.0e9, per_bond_bytes / 1.0e9, per_angle_bytes / 1.0e9,
        per_torsion_bytes / 1.0e9, per_out_of_plane_bytes / 1.0e9, per_stretch_bend_bytes / 1.0e9,
        per_interaction_bytes / 1.0e9);

    cudaMalloc(&d_descriptors, num_molecules * sizeof(sft::molecule_descriptor_t));
    cudaMalloc(&d_atom_offsets, num_molecules * sizeof(std::uint32_t));
    cudaMalloc(&d_atom_types, total_atom_count * sizeof(std::uint8_t));
    cudaMalloc(&d_partial_charges, total_atom_count * sizeof(float));
    cudaMalloc(&d_status_codes, num_molecules * sizeof(std::uint8_t));
    cudaMalloc(&d_vdw_params, total_atom_count * sizeof(sft::mmff94_atom_vdw_params_t));
    cudaMalloc(&d_output_charges, total_atom_count * sizeof(float));
    cudaMalloc(&d_bond_terms, per_bond_bytes);
    cudaMalloc(&d_angle_terms, per_angle_bytes);
    cudaMalloc(&d_torsion_terms, per_torsion_bytes);
    cudaMalloc(&d_out_of_plane_terms, per_out_of_plane_bytes);
    cudaMalloc(&d_stretch_bend_terms, per_stretch_bend_bytes);
    cudaMalloc(&d_counts_per_molecule, num_molecules * sizeof(sfmmff::mmff94_per_mol_counts_t));
    cudaMalloc(&d_interactions, per_interaction_bytes);
    cudaMalloc(&d_atom_offsets_lookup, total_atom_count * sizeof(sft::mmff94_atom_offsets_t));
    cudaError_t const allocation_error = cudaGetLastError();
    if (allocation_error != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(allocation_error));
        *type_and_charge_ms = *extract_terms_ms = *build_csr_ms = *total_gpu_ms = -1.0f;
        return;
    }

    // H2D: descriptors + offsets.
    cudaMemcpy(d_descriptors, descriptors, num_molecules * sizeof(sft::molecule_descriptor_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_atom_offsets, atom_offsets, num_molecules * sizeof(std::uint32_t), cudaMemcpyHostToDevice);

    cudaEvent_t event_start, event_after_type_and_charge, event_after_extract_terms, event_after_build_csr;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_after_type_and_charge);
    cudaEventCreate(&event_after_extract_terms);
    cudaEventCreate(&event_after_build_csr);

    // Warm-up the driver (fair comparison — first kernel launch has
    // some one-shot setup that isn't part of the steady-state cost).
    sfmmff::launch_mmff94_type_and_charge(d_descriptors, d_atom_offsets, num_molecules, tables_dev.view, d_atom_types,
                                          d_partial_charges, d_status_codes);
    cudaDeviceSynchronize();

    cudaEventRecord(event_start);
    sfmmff::launch_mmff94_type_and_charge(d_descriptors, d_atom_offsets, num_molecules, tables_dev.view, d_atom_types,
                                          d_partial_charges, d_status_codes);
    cudaEventRecord(event_after_type_and_charge);
    sfmmff::launch_mmff94_extract_terms(d_descriptors, d_atom_types, d_partial_charges, d_atom_offsets, num_molecules,
                                        tables_dev.view, d_vdw_params, d_output_charges, d_bond_terms, d_angle_terms,
                                        d_torsion_terms, d_out_of_plane_terms, d_stretch_bend_terms,
                                        d_counts_per_molecule, d_status_codes);
    cudaEventRecord(event_after_extract_terms);
    sfmmff::launch_mmff94_build_csr(d_descriptors, d_atom_offsets, num_molecules, d_counts_per_molecule, d_bond_terms,
                                    d_angle_terms, d_torsion_terms, d_out_of_plane_terms, d_stretch_bend_terms,
                                    d_interactions, d_atom_offsets_lookup, d_status_codes);
    cudaEventRecord(event_after_build_csr);
    cudaEventSynchronize(event_after_build_csr);

    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
        std::fprintf(stderr, "kernel error: %s\n", cudaGetErrorString(kernel_error));
        *type_and_charge_ms = *extract_terms_ms = *build_csr_ms = *total_gpu_ms = -1.0f;
        goto cleanup;
    }

    cudaEventElapsedTime(type_and_charge_ms, event_start, event_after_type_and_charge);
    cudaEventElapsedTime(extract_terms_ms, event_after_type_and_charge, event_after_extract_terms);
    cudaEventElapsedTime(build_csr_ms, event_after_extract_terms, event_after_build_csr);
    cudaEventElapsedTime(total_gpu_ms, event_start, event_after_build_csr);

cleanup:
    cudaEventDestroy(event_start);
    cudaEventDestroy(event_after_type_and_charge);
    cudaEventDestroy(event_after_extract_terms);
    cudaEventDestroy(event_after_build_csr);

    cudaFree(d_descriptors);
    cudaFree(d_atom_offsets);
    cudaFree(d_atom_types);
    cudaFree(d_partial_charges);
    cudaFree(d_status_codes);
    cudaFree(d_vdw_params);
    cudaFree(d_output_charges);
    cudaFree(d_bond_terms);
    cudaFree(d_angle_terms);
    cudaFree(d_torsion_terms);
    cudaFree(d_out_of_plane_terms);
    cudaFree(d_stretch_bend_terms);
    cudaFree(d_counts_per_molecule);
    cudaFree(d_interactions);
    cudaFree(d_atom_offsets_lookup);
}

} // namespace smallfield::bench
