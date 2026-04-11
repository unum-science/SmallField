/**
 * @file bench_chembl_kernels.cu
 * @brief CUDA kernel wrapper with wave-based dispatch sorted by molecule size.
 *
 * Sorts molecules by atom count and launches the pipeline kernel in waves.
 * Small molecules get more blocks/SM (less shared memory), large molecules fewer.
 * Reports per-wave timing and overall upload/kernel/download breakdown.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "bench_chembl_kernels.h"
#include "etkdg_pipeline.cuh"
#include "metric_embedding_prepass.cuh"

using namespace smallfield;

extern "C" {

int gpu_get_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

gpu_pipeline_result_t gpu_etkdg_mmff_pipeline(
    float const* host_bounds, std::uint32_t bounds_total_floats, chiral_center_descriptor_t const* host_chirals,
    std::uint32_t chirals_total, etk_torsion_descriptor_t const* host_torsions, std::uint32_t torsions_total,
    distance_constraint_t const* host_constraints, std::uint32_t constraints_total,
    etk_atom_interaction_t const* host_etk_interactions, std::uint32_t etk_interactions_total,
    etk_atom_offsets_t const* host_etk_atom_offsets, std::uint32_t etk_atom_offsets_total,
    mmff94_nonbonded_pair_t const* host_mmff_nb, std::uint32_t mmff_nb_total, mmff94_bond_t const* host_mmff_bonds,
    std::uint32_t mmff_bonds_total, mmff94_angle_t const* host_mmff_angles, std::uint32_t mmff_angles_total,
    mmff94_torsion_t const* host_mmff_torsions, std::uint32_t mmff_torsions_total, mmff94_oop_t const* host_mmff_oops,
    std::uint32_t mmff_oops_total, mmff94_stretchbend_t const* host_mmff_stbn, std::uint32_t mmff_stbn_total,
    mmff94_atom_interaction_t const* host_mmff_interactions, std::uint32_t mmff_interactions_total,
    mmff94_atom_offsets_t const* host_mmff_offsets, std::uint32_t mmff_offsets_total,
    etkdg_molecule_metadata_t const* host_metadata, std::uint32_t num_molecules, float* host_coords_out,
    float* host_energies_out, std::uint8_t* host_status_out, std::uint32_t num_conformers,
    std::uint32_t max_stereo_attempts, std::uint32_t max_atoms_per_molecule, std::uint64_t seed, int device_id) {

    cudaSetDevice(device_id);

    // Sort molecules by atom count for wave-based dispatch.
    std::vector<std::uint32_t> order(num_molecules);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        return host_metadata[a].num_atoms < host_metadata[b].num_atoms;
    });

    // Reorder metadata according to the sort order.
    std::vector<etkdg_molecule_metadata_t> sorted_meta(num_molecules);
    for (std::uint32_t i = 0; i < num_molecules; ++i)
        sorted_meta[i] = host_metadata[order[i]];

    // Allocate all device memory from a single pool with 256-byte alignment.
    std::uint32_t const total_conf = num_molecules * num_conformers;
    std::uint32_t const total_coords = total_conf * max_atoms_per_molecule * 3;

    auto align_up = [](std::size_t value, std::size_t alignment) -> std::size_t {
        return (value + alignment - 1) & ~(alignment - 1);
    };

    std::size_t pool_size = 0;
    pool_size += align_up(bounds_total_floats * sizeof(float), 256);
    pool_size += align_up(std::max(chirals_total, 1u) * sizeof(chiral_center_descriptor_t), 256);
    pool_size += align_up(std::max(torsions_total, 1u) * sizeof(etk_torsion_descriptor_t), 256);
    pool_size += align_up(std::max(constraints_total, 1u) * sizeof(distance_constraint_t), 256);
    pool_size += align_up(std::max(etk_interactions_total, 1u) * sizeof(etk_atom_interaction_t), 256);
    pool_size += align_up(std::max(etk_atom_offsets_total, 1u) * sizeof(etk_atom_offsets_t), 256);
    pool_size += align_up(std::max(mmff_nb_total, 1u) * sizeof(mmff94_nonbonded_pair_t), 256);
    pool_size += align_up(std::max(mmff_bonds_total, 1u) * sizeof(mmff94_bond_t), 256);
    pool_size += align_up(std::max(mmff_angles_total, 1u) * sizeof(mmff94_angle_t), 256);
    pool_size += align_up(std::max(mmff_torsions_total, 1u) * sizeof(mmff94_torsion_t), 256);
    pool_size += align_up(std::max(mmff_oops_total, 1u) * sizeof(mmff94_oop_t), 256);
    pool_size += align_up(std::max(mmff_stbn_total, 1u) * sizeof(mmff94_stretchbend_t), 256);
    pool_size += align_up(std::max(mmff_interactions_total, 1u) * sizeof(mmff94_atom_interaction_t), 256);
    pool_size += align_up(std::max(mmff_offsets_total, 1u) * sizeof(mmff94_atom_offsets_t), 256);
    pool_size += align_up(num_molecules * sizeof(etkdg_molecule_metadata_t), 256);
    pool_size += align_up(total_coords * sizeof(float), 256);
    pool_size += align_up(total_conf * sizeof(float), 256);
    pool_size += align_up(total_conf * sizeof(std::uint8_t), 256);
    pool_size += align_up(static_cast<std::size_t>(num_molecules) * max_atoms_per_molecule * 4 * sizeof(float), 256);

    char* device_pool = nullptr;
    cudaMalloc(&device_pool, pool_size);

    std::size_t offset = 0;

    float* dev_bounds = reinterpret_cast<float*>(device_pool + offset);
    offset += align_up(bounds_total_floats * sizeof(float), 256);

    auto* dev_chirals = reinterpret_cast<chiral_center_descriptor_t*>(device_pool + offset);
    offset += align_up(std::max(chirals_total, 1u) * sizeof(chiral_center_descriptor_t), 256);

    auto* dev_torsions = reinterpret_cast<etk_torsion_descriptor_t*>(device_pool + offset);
    offset += align_up(std::max(torsions_total, 1u) * sizeof(etk_torsion_descriptor_t), 256);

    auto* dev_constraints = reinterpret_cast<distance_constraint_t*>(device_pool + offset);
    offset += align_up(std::max(constraints_total, 1u) * sizeof(distance_constraint_t), 256);

    auto* dev_etk_interactions = reinterpret_cast<etk_atom_interaction_t*>(device_pool + offset);
    offset += align_up(std::max(etk_interactions_total, 1u) * sizeof(etk_atom_interaction_t), 256);

    auto* dev_etk_atom_offsets = reinterpret_cast<etk_atom_offsets_t*>(device_pool + offset);
    offset += align_up(std::max(etk_atom_offsets_total, 1u) * sizeof(etk_atom_offsets_t), 256);

    auto* dev_mmff_nb = reinterpret_cast<mmff94_nonbonded_pair_t*>(device_pool + offset);
    offset += align_up(std::max(mmff_nb_total, 1u) * sizeof(mmff94_nonbonded_pair_t), 256);

    auto* dev_mmff_bonds = reinterpret_cast<mmff94_bond_t*>(device_pool + offset);
    offset += align_up(std::max(mmff_bonds_total, 1u) * sizeof(mmff94_bond_t), 256);

    auto* dev_mmff_angles = reinterpret_cast<mmff94_angle_t*>(device_pool + offset);
    offset += align_up(std::max(mmff_angles_total, 1u) * sizeof(mmff94_angle_t), 256);

    auto* dev_mmff_torsions = reinterpret_cast<mmff94_torsion_t*>(device_pool + offset);
    offset += align_up(std::max(mmff_torsions_total, 1u) * sizeof(mmff94_torsion_t), 256);

    auto* dev_mmff_oops = reinterpret_cast<mmff94_oop_t*>(device_pool + offset);
    offset += align_up(std::max(mmff_oops_total, 1u) * sizeof(mmff94_oop_t), 256);

    auto* dev_mmff_stbn = reinterpret_cast<mmff94_stretchbend_t*>(device_pool + offset);
    offset += align_up(std::max(mmff_stbn_total, 1u) * sizeof(mmff94_stretchbend_t), 256);

    auto* dev_mmff_inter = reinterpret_cast<mmff94_atom_interaction_t*>(device_pool + offset);
    offset += align_up(std::max(mmff_interactions_total, 1u) * sizeof(mmff94_atom_interaction_t), 256);

    auto* dev_mmff_offsets = reinterpret_cast<mmff94_atom_offsets_t*>(device_pool + offset);
    offset += align_up(std::max(mmff_offsets_total, 1u) * sizeof(mmff94_atom_offsets_t), 256);

    auto* dev_metadata = reinterpret_cast<etkdg_molecule_metadata_t*>(device_pool + offset);
    offset += align_up(num_molecules * sizeof(etkdg_molecule_metadata_t), 256);

    float* dev_coords_out = reinterpret_cast<float*>(device_pool + offset);
    offset += align_up(total_coords * sizeof(float), 256);

    float* dev_energies_out = reinterpret_cast<float*>(device_pool + offset);
    offset += align_up(total_conf * sizeof(float), 256);

    auto* dev_status_out = reinterpret_cast<std::uint8_t*>(device_pool + offset);
    offset += align_up(total_conf * sizeof(std::uint8_t), 256);

    float* dev_initial_coordinates = reinterpret_cast<float*>(device_pool + offset);
    offset += align_up(static_cast<std::size_t>(num_molecules) * max_atoms_per_molecule * 4 * sizeof(float), 256);

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);

    // CUDA events for timing.
    cudaEvent_t ev_upload_start, ev_upload_end, ev_kernel_start, ev_kernel_end, ev_download_end;
    cudaEventCreate(&ev_upload_start);
    cudaEventCreate(&ev_upload_end);
    cudaEventCreate(&ev_kernel_start);
    cudaEventCreate(&ev_kernel_end);
    cudaEventCreate(&ev_download_end);

    // Create a stream for asynchronous uploads.
    cudaStream_t upload_stream;
    cudaStreamCreate(&upload_stream);

    // Run phase 1 upload because all parameter arrays are shared across waves.
    cudaEventRecord(ev_upload_start, upload_stream);
    cudaMemcpyAsync(dev_bounds, host_bounds, bounds_total_floats * sizeof(float), cudaMemcpyHostToDevice,
                    upload_stream);
    if (chirals_total > 0)
        cudaMemcpyAsync(dev_chirals, host_chirals, chirals_total * sizeof(chiral_center_descriptor_t),
                        cudaMemcpyHostToDevice, upload_stream);
    if (torsions_total > 0)
        cudaMemcpyAsync(dev_torsions, host_torsions, torsions_total * sizeof(etk_torsion_descriptor_t),
                        cudaMemcpyHostToDevice, upload_stream);
    if (constraints_total > 0)
        cudaMemcpyAsync(dev_constraints, host_constraints, constraints_total * sizeof(distance_constraint_t),
                        cudaMemcpyHostToDevice, upload_stream);
    if (etk_interactions_total > 0)
        cudaMemcpyAsync(dev_etk_interactions, host_etk_interactions,
                        etk_interactions_total * sizeof(etk_atom_interaction_t), cudaMemcpyHostToDevice, upload_stream);
    if (etk_atom_offsets_total > 0)
        cudaMemcpyAsync(dev_etk_atom_offsets, host_etk_atom_offsets,
                        etk_atom_offsets_total * sizeof(etk_atom_offsets_t), cudaMemcpyHostToDevice, upload_stream);
    if (mmff_nb_total > 0)
        cudaMemcpyAsync(dev_mmff_nb, host_mmff_nb, mmff_nb_total * sizeof(mmff94_nonbonded_pair_t),
                        cudaMemcpyHostToDevice, upload_stream);
    if (mmff_bonds_total > 0)
        cudaMemcpyAsync(dev_mmff_bonds, host_mmff_bonds, mmff_bonds_total * sizeof(mmff94_bond_t),
                        cudaMemcpyHostToDevice, upload_stream);
    if (mmff_angles_total > 0)
        cudaMemcpyAsync(dev_mmff_angles, host_mmff_angles, mmff_angles_total * sizeof(mmff94_angle_t),
                        cudaMemcpyHostToDevice, upload_stream);
    if (mmff_torsions_total > 0)
        cudaMemcpyAsync(dev_mmff_torsions, host_mmff_torsions, mmff_torsions_total * sizeof(mmff94_torsion_t),
                        cudaMemcpyHostToDevice, upload_stream);
    if (mmff_oops_total > 0)
        cudaMemcpyAsync(dev_mmff_oops, host_mmff_oops, mmff_oops_total * sizeof(mmff94_oop_t), cudaMemcpyHostToDevice,
                        upload_stream);
    if (mmff_stbn_total > 0)
        cudaMemcpyAsync(dev_mmff_stbn, host_mmff_stbn, mmff_stbn_total * sizeof(mmff94_stretchbend_t),
                        cudaMemcpyHostToDevice, upload_stream);
    if (mmff_interactions_total > 0)
        cudaMemcpyAsync(dev_mmff_inter, host_mmff_interactions,
                        mmff_interactions_total * sizeof(mmff94_atom_interaction_t), cudaMemcpyHostToDevice,
                        upload_stream);
    if (mmff_offsets_total > 0)
        cudaMemcpyAsync(dev_mmff_offsets, host_mmff_offsets, mmff_offsets_total * sizeof(mmff94_atom_offsets_t),
                        cudaMemcpyHostToDevice, upload_stream);
    cudaMemcpyAsync(dev_metadata, sorted_meta.data(), num_molecules * sizeof(etkdg_molecule_metadata_t),
                    cudaMemcpyHostToDevice, upload_stream);

    // Synchronize before the metric embedding prepass which reads the uploaded data.
    cudaStreamSynchronize(upload_stream);
    bool const prepass_ok = build_metric_initial_coordinates_on_gpu(
        dev_bounds, dev_metadata, sorted_meta.data(), num_molecules, max_atoms_per_molecule, dev_initial_coordinates);
    if (!prepass_ok)
        std::fprintf(stderr, "Metric embedding prepass failed.\n");
    cudaEventRecord(ev_upload_end);

    // Run phase 2 wave-based kernel dispatch.
    constexpr std::uint32_t wave_bins[] = {16, 32, 48, 64, 80, 96, 112, 128};
    constexpr std::uint32_t num_bins = 8;

    cudaEventRecord(ev_kernel_start);

    std::uint32_t wave_start = 0;
    std::uint32_t total_grid = 0;
    std::uint32_t max_bpsm = 0;

    for (std::uint32_t bin = 0; bin < num_bins && wave_start < num_molecules; ++bin) {
        // Find the end of the current wave.
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

        // Launch the kernel for the molecules in the current wave.
        // Write output to sorted slots [wave_start * num_conf, wave_end * num_conf).
        etkdg_pipeline_warp_kernel<<<grid_size, 32, shared_bytes>>>(
            dev_bounds, dev_initial_coordinates + wave_start * max_atoms_per_molecule * 4, dev_chirals, dev_torsions,
            dev_constraints, dev_etk_interactions, dev_etk_atom_offsets, dev_mmff_nb, dev_mmff_bonds, dev_mmff_angles,
            dev_mmff_torsions, dev_mmff_oops, dev_mmff_stbn, dev_mmff_inter, dev_mmff_offsets,
            dev_metadata + wave_start, dev_coords_out + wave_start * num_conformers * max_atoms_per_molecule * 3,
            dev_energies_out + wave_start * num_conformers, dev_status_out + wave_start * num_conformers, wave_size,
            num_conformers, max_stereo_attempts, wave_max_atoms, max_atoms_per_molecule, max_atoms_per_molecule, seed);

        total_grid += grid_size;
        max_bpsm = std::max(max_bpsm, static_cast<std::uint32_t>(blocks_per_sm));
        wave_start = wave_end;
    }

    cudaEventRecord(ev_kernel_end);

    // check for kernel errors before download
    {
        cudaError_t const err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            std::fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Run phase 3 download and unsort the results.
    std::vector<float> sorted_coords(total_coords);
    std::vector<float> sorted_energies(total_conf);
    std::vector<std::uint8_t> sorted_status(total_conf);

    cudaMemcpyAsync(sorted_coords.data(), dev_coords_out, total_coords * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(sorted_energies.data(), dev_energies_out, total_conf * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(sorted_status.data(), dev_status_out, total_conf * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(ev_download_end);
    cudaEventSynchronize(ev_download_end);

    // Map each sorted index back to the original molecule index.
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

    // Count successes.
    std::uint32_t num_succeeded = 0;
    for (std::uint32_t i = 0; i < total_conf; ++i)
        if (host_status_out[i] == 0)
            num_succeeded++;

    float upload_ms = 0.0f, kernel_ms = 0.0f, download_ms = 0.0f;
    cudaEventElapsedTime(&upload_ms, ev_upload_start, ev_upload_end);
    cudaEventElapsedTime(&kernel_ms, ev_kernel_start, ev_kernel_end);
    cudaEventElapsedTime(&download_ms, ev_kernel_end, ev_download_end);

    // Clean up.
    cudaEventDestroy(ev_upload_start);
    cudaEventDestroy(ev_upload_end);
    cudaEventDestroy(ev_kernel_start);
    cudaEventDestroy(ev_kernel_end);
    cudaEventDestroy(ev_download_end);
    cudaStreamDestroy(upload_stream);
    cudaFree(device_pool);

    return {upload_ms, kernel_ms, download_ms, num_succeeded, total_grid, max_bpsm};
}

} // extern "C"
