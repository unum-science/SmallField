/**
 * @file bench_chembl_kernels.cu
 * @brief CUDA kernel wrapper with wave-based dispatch sorted by molecule size.
 *
 * Sorts molecules by atom count and launches the pipeline kernel in waves.
 * Small molecules get more blocks/SM (less shared memory), large molecules fewer.
 * Reports per-wave timing and overall upload/kernel/download breakdown.
 */

#include "bench_chembl_kernels.h"
#include "etkdg_pipeline.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <numeric>
#include <vector>

using namespace smallfield;

extern "C" {

gpu_pipeline_result_t gpu_etkdg_mmff_pipeline(
    float const* host_bounds, std::uint32_t bounds_total_floats, chiral_center_descriptor_t const* host_chirals,
    std::uint32_t chirals_total, etk_torsion_descriptor_t const* host_torsions, std::uint32_t torsions_total,
    distance_constraint_t const* host_constraints, std::uint32_t constraints_total,
    mmff94_nonbonded_pair_t const* host_mmff_nb, std::uint32_t mmff_nb_total, mmff94_bond_t const* host_mmff_bonds,
    std::uint32_t mmff_bonds_total, mmff94_angle_t const* host_mmff_angles, std::uint32_t mmff_angles_total,
    mmff94_torsion_t const* host_mmff_torsions, std::uint32_t mmff_torsions_total,
    mmff94_atom_interaction_t const* host_mmff_interactions, std::uint32_t mmff_interactions_total,
    mmff94_atom_offsets_t const* host_mmff_offsets, std::uint32_t mmff_offsets_total,
    etkdg_molecule_metadata_t const* host_metadata, std::uint32_t num_molecules, float* host_coords_out,
    float* host_energies_out, std::uint8_t* host_status_out, std::uint32_t num_conformers,
    std::uint32_t max_stereo_attempts, std::uint32_t max_atoms_per_molecule, std::uint64_t seed) {

    // sort molecules by atom count for wave-based dispatch
    std::vector<std::uint32_t> order(num_molecules);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        return host_metadata[a].num_atoms < host_metadata[b].num_atoms;
    });

    // reorder metadata according to sort
    std::vector<etkdg_molecule_metadata_t> sorted_meta(num_molecules);
    for (std::uint32_t i = 0; i < num_molecules; ++i)
        sorted_meta[i] = host_metadata[order[i]];

    // allocate device memory for all input arrays
    float* dev_bounds;
    chiral_center_descriptor_t* dev_chirals;
    etk_torsion_descriptor_t* dev_torsions;
    distance_constraint_t* dev_constraints;
    mmff94_nonbonded_pair_t* dev_mmff_nb;
    mmff94_bond_t* dev_mmff_bonds;
    mmff94_angle_t* dev_mmff_angles;
    mmff94_torsion_t* dev_mmff_torsions;
    mmff94_atom_interaction_t* dev_mmff_inter;
    mmff94_atom_offsets_t* dev_mmff_offsets;
    etkdg_molecule_metadata_t* dev_metadata;

    cudaMalloc(&dev_bounds, bounds_total_floats * sizeof(float));
    cudaMalloc(&dev_chirals, std::max(chirals_total, 1u) * sizeof(chiral_center_descriptor_t));
    cudaMalloc(&dev_torsions, std::max(torsions_total, 1u) * sizeof(etk_torsion_descriptor_t));
    cudaMalloc(&dev_constraints, std::max(constraints_total, 1u) * sizeof(distance_constraint_t));
    cudaMalloc(&dev_mmff_nb, std::max(mmff_nb_total, 1u) * sizeof(mmff94_nonbonded_pair_t));
    cudaMalloc(&dev_mmff_bonds, std::max(mmff_bonds_total, 1u) * sizeof(mmff94_bond_t));
    cudaMalloc(&dev_mmff_angles, std::max(mmff_angles_total, 1u) * sizeof(mmff94_angle_t));
    cudaMalloc(&dev_mmff_torsions, std::max(mmff_torsions_total, 1u) * sizeof(mmff94_torsion_t));
    cudaMalloc(&dev_mmff_inter, std::max(mmff_interactions_total, 1u) * sizeof(mmff94_atom_interaction_t));
    cudaMalloc(&dev_mmff_offsets, std::max(mmff_offsets_total, 1u) * sizeof(mmff94_atom_offsets_t));
    cudaMalloc(&dev_metadata, num_molecules * sizeof(etkdg_molecule_metadata_t));

    // allocate output buffers (using global max_atoms for output stride)
    std::uint32_t const total_conf = num_molecules * num_conformers;
    std::uint32_t const total_coords = total_conf * max_atoms_per_molecule * 3;
    float* dev_coords_out;
    float* dev_energies_out;
    std::uint8_t* dev_status_out;
    cudaMalloc(&dev_coords_out, total_coords * sizeof(float));
    cudaMalloc(&dev_energies_out, total_conf * sizeof(float));
    cudaMalloc(&dev_status_out, total_conf * sizeof(std::uint8_t));

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

    // CUDA events for timing
    cudaEvent_t ev_upload_start, ev_upload_end, ev_kernel_start, ev_kernel_end, ev_download_end;
    cudaEventCreate(&ev_upload_start);
    cudaEventCreate(&ev_upload_end);
    cudaEventCreate(&ev_kernel_start);
    cudaEventCreate(&ev_kernel_end);
    cudaEventCreate(&ev_download_end);

    // phase 1: upload (all parameter arrays are shared across waves)
    cudaEventRecord(ev_upload_start);
    cudaMemcpy(dev_bounds, host_bounds, bounds_total_floats * sizeof(float), cudaMemcpyHostToDevice);
    if (chirals_total > 0)
        cudaMemcpy(dev_chirals, host_chirals, chirals_total * sizeof(chiral_center_descriptor_t),
                   cudaMemcpyHostToDevice);
    if (torsions_total > 0)
        cudaMemcpy(dev_torsions, host_torsions, torsions_total * sizeof(etk_torsion_descriptor_t),
                   cudaMemcpyHostToDevice);
    if (constraints_total > 0)
        cudaMemcpy(dev_constraints, host_constraints, constraints_total * sizeof(distance_constraint_t),
                   cudaMemcpyHostToDevice);
    if (mmff_nb_total > 0)
        cudaMemcpy(dev_mmff_nb, host_mmff_nb, mmff_nb_total * sizeof(mmff94_nonbonded_pair_t), cudaMemcpyHostToDevice);
    if (mmff_bonds_total > 0)
        cudaMemcpy(dev_mmff_bonds, host_mmff_bonds, mmff_bonds_total * sizeof(mmff94_bond_t), cudaMemcpyHostToDevice);
    if (mmff_angles_total > 0)
        cudaMemcpy(dev_mmff_angles, host_mmff_angles, mmff_angles_total * sizeof(mmff94_angle_t),
                   cudaMemcpyHostToDevice);
    if (mmff_torsions_total > 0)
        cudaMemcpy(dev_mmff_torsions, host_mmff_torsions, mmff_torsions_total * sizeof(mmff94_torsion_t),
                   cudaMemcpyHostToDevice);
    if (mmff_interactions_total > 0)
        cudaMemcpy(dev_mmff_inter, host_mmff_interactions, mmff_interactions_total * sizeof(mmff94_atom_interaction_t),
                   cudaMemcpyHostToDevice);
    if (mmff_offsets_total > 0)
        cudaMemcpy(dev_mmff_offsets, host_mmff_offsets, mmff_offsets_total * sizeof(mmff94_atom_offsets_t),
                   cudaMemcpyHostToDevice);
    cudaMemcpy(dev_metadata, sorted_meta.data(), num_molecules * sizeof(etkdg_molecule_metadata_t),
               cudaMemcpyHostToDevice);
    cudaEventRecord(ev_upload_end);

    // phase 2: wave-based kernel dispatch
    constexpr std::uint32_t wave_bins[] = {32, 64, 96, 128};
    constexpr std::uint32_t num_bins = 4;

    cudaEventRecord(ev_kernel_start);

    std::uint32_t wave_start = 0;
    std::uint32_t total_grid = 0;
    std::uint32_t max_bpsm = 0;

    for (std::uint32_t bin = 0; bin < num_bins && wave_start < num_molecules; ++bin) {
        // find end of this wave: all molecules with num_atoms <= wave_bins[bin]
        std::uint32_t wave_end = wave_start;
        while (wave_end < num_molecules && sorted_meta[wave_end].num_atoms <= wave_bins[bin])
            ++wave_end;

        if (wave_end == wave_start)
            continue;

        std::uint32_t const wave_size = wave_end - wave_start;
        std::uint32_t const wave_max_atoms = wave_bins[bin];

        std::uint32_t const shared_bytes = compute_pipeline_shared_memory_bytes(wave_max_atoms, 0);
        cudaFuncSetAttribute(etkdg_pipeline_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);

        int blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, etkdg_pipeline_warp_kernel, 32, shared_bytes);
        std::uint32_t const grid_size = std::min(wave_size, static_cast<std::uint32_t>(blocks_per_sm * num_sms));

        std::printf("    wave %u: atoms<=%-3u  mols=%-5u  shared=%-5u  blocks/SM=%-2d  grid=%-5u\n", bin,
                    wave_max_atoms, wave_size, shared_bytes, blocks_per_sm, grid_size);

        // launch kernel for this wave's molecules
        // output written to sorted slots [wave_start*num_conf .. wave_end*num_conf)
        etkdg_pipeline_warp_kernel<<<grid_size, 32, shared_bytes>>>(
            dev_bounds, dev_chirals, dev_torsions, dev_constraints, dev_mmff_nb, dev_mmff_bonds, dev_mmff_angles,
            dev_mmff_torsions, dev_mmff_inter, dev_mmff_offsets, dev_metadata + wave_start,
            dev_coords_out + wave_start * num_conformers * max_atoms_per_molecule * 3,
            dev_energies_out + wave_start * num_conformers, dev_status_out + wave_start * num_conformers, wave_size,
            num_conformers, max_stereo_attempts, wave_max_atoms, max_atoms_per_molecule, seed);

        total_grid += grid_size;
        max_bpsm = std::max(max_bpsm, static_cast<std::uint32_t>(blocks_per_sm));
        wave_start = wave_end;
    }

    cudaEventRecord(ev_kernel_end);

    // phase 3: download and unsort results
    std::vector<float> sorted_coords(total_coords);
    std::vector<float> sorted_energies(total_conf);
    std::vector<std::uint8_t> sorted_status(total_conf);

    cudaMemcpy(sorted_coords.data(), dev_coords_out, total_coords * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sorted_energies.data(), dev_energies_out, total_conf * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sorted_status.data(), dev_status_out, total_conf * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(ev_download_end);
    cudaEventSynchronize(ev_download_end);

    // unsort: map sorted index back to original molecule index
    std::uint32_t const coord_stride = max_atoms_per_molecule * 3;
    for (std::uint32_t sorted_idx = 0; sorted_idx < num_molecules; ++sorted_idx) {
        std::uint32_t const orig_idx = order[sorted_idx];
        for (std::uint32_t c = 0; c < num_conformers; ++c) {
            std::uint32_t const src_slot = sorted_idx * num_conformers + c;
            std::uint32_t const dst_slot = orig_idx * num_conformers + c;
            host_energies_out[dst_slot] = sorted_energies[src_slot];
            host_status_out[dst_slot] = sorted_status[src_slot];
            std::memcpy(host_coords_out + dst_slot * coord_stride, sorted_coords.data() + src_slot * coord_stride,
                        coord_stride * sizeof(float));
        }
    }

    // count successes
    std::uint32_t num_succeeded = 0;
    for (std::uint32_t i = 0; i < total_conf; ++i)
        if (host_status_out[i] == 0)
            num_succeeded++;

    float upload_ms = 0.0f, kernel_ms = 0.0f, download_ms = 0.0f;
    cudaEventElapsedTime(&upload_ms, ev_upload_start, ev_upload_end);
    cudaEventElapsedTime(&kernel_ms, ev_kernel_start, ev_kernel_end);
    cudaEventElapsedTime(&download_ms, ev_kernel_end, ev_download_end);

    // cleanup
    cudaEventDestroy(ev_upload_start);
    cudaEventDestroy(ev_upload_end);
    cudaEventDestroy(ev_kernel_start);
    cudaEventDestroy(ev_kernel_end);
    cudaEventDestroy(ev_download_end);
    cudaFree(dev_bounds);
    cudaFree(dev_chirals);
    cudaFree(dev_torsions);
    cudaFree(dev_constraints);
    cudaFree(dev_mmff_nb);
    cudaFree(dev_mmff_bonds);
    cudaFree(dev_mmff_angles);
    cudaFree(dev_mmff_torsions);
    cudaFree(dev_mmff_inter);
    cudaFree(dev_mmff_offsets);
    cudaFree(dev_metadata);
    cudaFree(dev_coords_out);
    cudaFree(dev_energies_out);
    cudaFree(dev_status_out);

    return {upload_ms, kernel_ms, download_ms, num_succeeded, total_grid, max_bpsm};
}

} // extern "C"
