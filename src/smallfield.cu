/**
 *  @file smallfield.cu
 *  @brief Library implementation of the SmallField public API and pybind11 bindings.
 *
 *  Implements context_t (multi-GPU CUDA stream/cuSOLVER management) and
 *  generate_conformers() (CPU preprocessing, wave-based GPU dispatch,
 *  half-precision output). Exposes a minimal Python module at the bottom.
 */

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#if defined(SMALLFIELD_BUILD_PYTHON)
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

#include "etkdg_pipeline.cuh"
#include "etkdg_preprocess.hpp"
#include "metric_embedding_prepass.cuh"
#include "smallfield.hpp"

namespace smallfield {

context_t::context_t(std::span<int const> devices) {
    if (devices.empty()) {
        int count = 0;
        cudaGetDeviceCount(&count);
        for (int i = 0; i < count; ++i) device_ids.push_back(i);
    }
    else { device_ids.assign(devices.begin(), devices.end()); }

    streams.resize(device_ids.size());
    solvers.resize(device_ids.size());

    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        cudaSetDevice(device_ids[i]);
        cudaStreamCreate(&streams[i]);
        cusolverDnCreate(&solvers[i]);
        cusolverDnSetStream(solvers[i], streams[i]);
    }
}

context_t::~context_t() {
    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        cudaSetDevice(device_ids[i]);
        cusolverDnDestroy(solvers[i]);
        cudaStreamDestroy(streams[i]);
    }
}

std::size_t align_up(std::size_t value, std::size_t alignment) { return (value + alignment - 1) & ~(alignment - 1); }

void run_gpu_pipeline(int device_id, cudaStream_t stream, etkdg_molecule_batch_t const &batch,
                      std::uint32_t const *local_indices, std::uint32_t local_count, std::uint32_t num_conformers,
                      std::uint32_t max_stereo_attempts, std::uint32_t max_atoms_per_molecule, std::uint64_t seed,
                      float *host_coords_out, float *host_energies_out, std::uint8_t *host_status_out) {

    cudaSetDevice(device_id);

    // Build local metadata from the batch for the molecules assigned to this GPU.
    std::vector<etkdg_molecule_metadata_t> local_meta(local_count);
    for (std::uint32_t i = 0; i < local_count; ++i) local_meta[i] = batch.molecule_metadata[local_indices[i]];

    // Sort by atom count for wave-based dispatch.
    std::vector<std::uint32_t> order(local_count);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(),
              [&](std::uint32_t a, std::uint32_t b) { return local_meta[a].num_atoms < local_meta[b].num_atoms; });

    std::vector<etkdg_molecule_metadata_t> sorted_meta(local_count);
    for (std::uint32_t i = 0; i < local_count; ++i) sorted_meta[i] = local_meta[order[i]];

    // Compute pool size with 256-byte alignment.
    auto const bounds_total_floats = static_cast<std::uint32_t>(batch.bounds_matrices.size());
    auto const chirals_total = static_cast<std::uint32_t>(batch.chiral_centers.size());
    auto const torsions_total = static_cast<std::uint32_t>(batch.etk_torsions.size());
    auto const constraints_total = static_cast<std::uint32_t>(batch.distance_constraints.size());
    auto const etk_interactions_total = static_cast<std::uint32_t>(batch.etk_interactions.size());
    auto const etk_atom_offsets_total = static_cast<std::uint32_t>(batch.etk_atom_offsets.size());
    auto const mmff_nonbonded_total = static_cast<std::uint32_t>(batch.mmff_nonbonded.size());
    auto const mmff_bonds_total = static_cast<std::uint32_t>(batch.mmff_bonds.size());
    auto const mmff_angles_total = static_cast<std::uint32_t>(batch.mmff_angles.size());
    auto const mmff_torsions_total = static_cast<std::uint32_t>(batch.mmff_torsions.size());
    auto const mmff_oops_total = static_cast<std::uint32_t>(batch.mmff_oops.size());
    auto const mmff_stretchbend_total = static_cast<std::uint32_t>(batch.mmff_stretchbends.size());
    auto const mmff_interactions_total = static_cast<std::uint32_t>(batch.mmff_interactions.size());
    auto const mmff_offsets_total = static_cast<std::uint32_t>(batch.mmff_atom_offsets.size());

    std::uint32_t const total_conf = local_count * num_conformers;
    std::uint32_t const total_coords = total_conf * max_atoms_per_molecule * 3;

    std::size_t pool_size = 0;
    pool_size += align_up(bounds_total_floats * sizeof(float), 256);
    pool_size += align_up(std::max(chirals_total, 1u) * sizeof(chiral_center_descriptor_t), 256);
    pool_size += align_up(std::max(torsions_total, 1u) * sizeof(etk_torsion_descriptor_t), 256);
    pool_size += align_up(std::max(constraints_total, 1u) * sizeof(distance_constraint_t), 256);
    pool_size += align_up(std::max(etk_interactions_total, 1u) * sizeof(etk_atom_interaction_t), 256);
    pool_size += align_up(std::max(etk_atom_offsets_total, 1u) * sizeof(etk_atom_offsets_t), 256);
    pool_size += align_up(std::max(mmff_nonbonded_total, 1u) * sizeof(mmff94_nonbonded_pair_t), 256);
    pool_size += align_up(std::max(mmff_bonds_total, 1u) * sizeof(mmff94_bond_t), 256);
    pool_size += align_up(std::max(mmff_angles_total, 1u) * sizeof(mmff94_angle_t), 256);
    pool_size += align_up(std::max(mmff_torsions_total, 1u) * sizeof(mmff94_torsion_t), 256);
    pool_size += align_up(std::max(mmff_oops_total, 1u) * sizeof(mmff94_oop_t), 256);
    pool_size += align_up(std::max(mmff_stretchbend_total, 1u) * sizeof(mmff94_stretchbend_t), 256);
    pool_size += align_up(std::max(mmff_interactions_total, 1u) * sizeof(mmff94_atom_interaction_t), 256);
    pool_size += align_up(std::max(mmff_offsets_total, 1u) * sizeof(mmff94_atom_offsets_t), 256);
    pool_size += align_up(local_count * sizeof(etkdg_molecule_metadata_t), 256);
    pool_size += align_up(total_coords * sizeof(float), 256);
    pool_size += align_up(total_conf * sizeof(float), 256);
    pool_size += align_up(total_conf * sizeof(std::uint8_t), 256);
    pool_size += align_up(static_cast<std::size_t>(local_count) * max_atoms_per_molecule * 4 * sizeof(float), 256);

    char *device_pool = nullptr;
    cudaMalloc(&device_pool, pool_size);

    // Carve out sub-allocations from the pool.
    std::size_t offset = 0;

    auto *device_bounds = reinterpret_cast<float *>(device_pool + offset);
    offset += align_up(bounds_total_floats * sizeof(float), 256);

    auto *device_chirals = reinterpret_cast<chiral_center_descriptor_t *>(device_pool + offset);
    offset += align_up(std::max(chirals_total, 1u) * sizeof(chiral_center_descriptor_t), 256);

    auto *device_torsions = reinterpret_cast<etk_torsion_descriptor_t *>(device_pool + offset);
    offset += align_up(std::max(torsions_total, 1u) * sizeof(etk_torsion_descriptor_t), 256);

    auto *device_constraints = reinterpret_cast<distance_constraint_t *>(device_pool + offset);
    offset += align_up(std::max(constraints_total, 1u) * sizeof(distance_constraint_t), 256);

    auto *device_etk_interactions = reinterpret_cast<etk_atom_interaction_t *>(device_pool + offset);
    offset += align_up(std::max(etk_interactions_total, 1u) * sizeof(etk_atom_interaction_t), 256);

    auto *device_etk_atom_offsets = reinterpret_cast<etk_atom_offsets_t *>(device_pool + offset);
    offset += align_up(std::max(etk_atom_offsets_total, 1u) * sizeof(etk_atom_offsets_t), 256);

    auto *device_mmff_nonbonded = reinterpret_cast<mmff94_nonbonded_pair_t *>(device_pool + offset);
    offset += align_up(std::max(mmff_nonbonded_total, 1u) * sizeof(mmff94_nonbonded_pair_t), 256);

    auto *device_mmff_bonds = reinterpret_cast<mmff94_bond_t *>(device_pool + offset);
    offset += align_up(std::max(mmff_bonds_total, 1u) * sizeof(mmff94_bond_t), 256);

    auto *device_mmff_angles = reinterpret_cast<mmff94_angle_t *>(device_pool + offset);
    offset += align_up(std::max(mmff_angles_total, 1u) * sizeof(mmff94_angle_t), 256);

    auto *device_mmff_torsions = reinterpret_cast<mmff94_torsion_t *>(device_pool + offset);
    offset += align_up(std::max(mmff_torsions_total, 1u) * sizeof(mmff94_torsion_t), 256);

    auto *device_mmff_oops = reinterpret_cast<mmff94_oop_t *>(device_pool + offset);
    offset += align_up(std::max(mmff_oops_total, 1u) * sizeof(mmff94_oop_t), 256);

    auto *device_mmff_stretchbend = reinterpret_cast<mmff94_stretchbend_t *>(device_pool + offset);
    offset += align_up(std::max(mmff_stretchbend_total, 1u) * sizeof(mmff94_stretchbend_t), 256);

    auto *device_mmff_interactions = reinterpret_cast<mmff94_atom_interaction_t *>(device_pool + offset);
    offset += align_up(std::max(mmff_interactions_total, 1u) * sizeof(mmff94_atom_interaction_t), 256);

    auto *device_mmff_offsets = reinterpret_cast<mmff94_atom_offsets_t *>(device_pool + offset);
    offset += align_up(std::max(mmff_offsets_total, 1u) * sizeof(mmff94_atom_offsets_t), 256);

    auto *device_metadata = reinterpret_cast<etkdg_molecule_metadata_t *>(device_pool + offset);
    offset += align_up(local_count * sizeof(etkdg_molecule_metadata_t), 256);

    auto *device_coords_out = reinterpret_cast<float *>(device_pool + offset);
    offset += align_up(total_coords * sizeof(float), 256);

    auto *device_energies_out = reinterpret_cast<float *>(device_pool + offset);
    offset += align_up(total_conf * sizeof(float), 256);

    auto *device_status_out = reinterpret_cast<std::uint8_t *>(device_pool + offset);
    offset += align_up(total_conf * sizeof(std::uint8_t), 256);

    auto *device_initial_coordinates = reinterpret_cast<float *>(device_pool + offset);

    // Upload shared parameter arrays.
    cudaMemcpyAsync(device_bounds, batch.bounds_matrices.data(), bounds_total_floats * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    if (chirals_total > 0)
        cudaMemcpyAsync(device_chirals, batch.chiral_centers.data(), chirals_total * sizeof(chiral_center_descriptor_t),
                        cudaMemcpyHostToDevice, stream);
    if (torsions_total > 0)
        cudaMemcpyAsync(device_torsions, batch.etk_torsions.data(), torsions_total * sizeof(etk_torsion_descriptor_t),
                        cudaMemcpyHostToDevice, stream);
    if (constraints_total > 0)
        cudaMemcpyAsync(device_constraints, batch.distance_constraints.data(),
                        constraints_total * sizeof(distance_constraint_t), cudaMemcpyHostToDevice, stream);
    if (etk_interactions_total > 0)
        cudaMemcpyAsync(device_etk_interactions, batch.etk_interactions.data(),
                        etk_interactions_total * sizeof(etk_atom_interaction_t), cudaMemcpyHostToDevice, stream);
    if (etk_atom_offsets_total > 0)
        cudaMemcpyAsync(device_etk_atom_offsets, batch.etk_atom_offsets.data(),
                        etk_atom_offsets_total * sizeof(etk_atom_offsets_t), cudaMemcpyHostToDevice, stream);
    if (mmff_nonbonded_total > 0)
        cudaMemcpyAsync(device_mmff_nonbonded, batch.mmff_nonbonded.data(),
                        mmff_nonbonded_total * sizeof(mmff94_nonbonded_pair_t), cudaMemcpyHostToDevice, stream);
    if (mmff_bonds_total > 0)
        cudaMemcpyAsync(device_mmff_bonds, batch.mmff_bonds.data(), mmff_bonds_total * sizeof(mmff94_bond_t),
                        cudaMemcpyHostToDevice, stream);
    if (mmff_angles_total > 0)
        cudaMemcpyAsync(device_mmff_angles, batch.mmff_angles.data(), mmff_angles_total * sizeof(mmff94_angle_t),
                        cudaMemcpyHostToDevice, stream);
    if (mmff_torsions_total > 0)
        cudaMemcpyAsync(device_mmff_torsions, batch.mmff_torsions.data(),
                        mmff_torsions_total * sizeof(mmff94_torsion_t), cudaMemcpyHostToDevice, stream);
    if (mmff_oops_total > 0)
        cudaMemcpyAsync(device_mmff_oops, batch.mmff_oops.data(), mmff_oops_total * sizeof(mmff94_oop_t),
                        cudaMemcpyHostToDevice, stream);
    if (mmff_stretchbend_total > 0)
        cudaMemcpyAsync(device_mmff_stretchbend, batch.mmff_stretchbends.data(),
                        mmff_stretchbend_total * sizeof(mmff94_stretchbend_t), cudaMemcpyHostToDevice, stream);
    if (mmff_interactions_total > 0)
        cudaMemcpyAsync(device_mmff_interactions, batch.mmff_interactions.data(),
                        mmff_interactions_total * sizeof(mmff94_atom_interaction_t), cudaMemcpyHostToDevice, stream);
    if (mmff_offsets_total > 0)
        cudaMemcpyAsync(device_mmff_offsets, batch.mmff_atom_offsets.data(),
                        mmff_offsets_total * sizeof(mmff94_atom_offsets_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_metadata, sorted_meta.data(), local_count * sizeof(etkdg_molecule_metadata_t),
                    cudaMemcpyHostToDevice, stream);

    // Synchronize before the metric embedding prepass which reads uploaded data.
    cudaStreamSynchronize(stream);
    build_metric_initial_coordinates_on_gpu(device_bounds, device_metadata, sorted_meta.data(), local_count,
                                            max_atoms_per_molecule, device_initial_coordinates, stream);

    // Wave-based kernel dispatch (8 bins: 16,32,48,64,80,96,112,128).
    constexpr std::uint32_t wave_bins[] = {16, 32, 48, 64, 80, 96, 112, 128};
    constexpr std::uint32_t num_bins = 8;

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);

    std::uint32_t wave_start = 0;
    for (std::uint32_t bin = 0; bin < num_bins && wave_start < local_count; ++bin) {
        std::uint32_t wave_end = wave_start;
        while (wave_end < local_count && sorted_meta[wave_end].num_atoms <= wave_bins[bin]) ++wave_end;

        if (wave_end == wave_start) continue;

        std::uint32_t const wave_size = wave_end - wave_start;
        std::uint32_t const wave_max_atoms = wave_bins[bin];

        std::uint32_t const shared_bytes = compute_pipeline_shared_memory_bytes(wave_max_atoms, 0);
        cudaFuncSetAttribute(etkdg_pipeline_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);

        int blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, etkdg_pipeline_warp_kernel, 32, shared_bytes);
        std::uint32_t const grid_size = std::min(wave_size, static_cast<std::uint32_t>(blocks_per_sm * num_sms));

        etkdg_pipeline_warp_kernel<<<grid_size, 32, shared_bytes, stream>>>(
            device_bounds, device_initial_coordinates + wave_start * max_atoms_per_molecule * 4, device_chirals,
            device_torsions, device_constraints, device_etk_interactions, device_etk_atom_offsets,
            device_mmff_nonbonded, device_mmff_bonds, device_mmff_angles, device_mmff_torsions, device_mmff_oops,
            device_mmff_stretchbend, device_mmff_interactions, device_mmff_offsets, device_metadata + wave_start,
            device_coords_out + wave_start * num_conformers * max_atoms_per_molecule * 3,
            device_energies_out + wave_start * num_conformers, device_status_out + wave_start * num_conformers,
            wave_size, num_conformers, max_stereo_attempts, wave_max_atoms, max_atoms_per_molecule,
            max_atoms_per_molecule, seed);

        wave_start = wave_end;
    }

    cudaStreamSynchronize(stream);

    // Download results from sorted layout.
    std::vector<float> sorted_coords(total_coords);
    std::vector<float> sorted_energies(total_conf);
    std::vector<std::uint8_t> sorted_status(total_conf);

    cudaMemcpyAsync(sorted_coords.data(), device_coords_out, total_coords * sizeof(float), cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(sorted_energies.data(), device_energies_out, total_conf * sizeof(float), cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(sorted_status.data(), device_status_out, total_conf * sizeof(std::uint8_t), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);

    // Unsort: map sorted index → original local index → output slot.
    std::uint32_t const coord_stride = max_atoms_per_molecule * 3;
    for (std::uint32_t sorted_idx = 0; sorted_idx < local_count; ++sorted_idx) {
        std::uint32_t const local_idx = order[sorted_idx];
        std::uint32_t const orig_idx = local_indices[local_idx];
        // orig_idx is the index into batch.molecule_metadata; we need
        // the original input index to write into the caller's output arrays.
        std::uint32_t const input_idx = batch.input_indices[orig_idx];
        for (std::uint32_t c = 0; c < num_conformers; ++c) {
            std::uint32_t const src_slot = sorted_idx * num_conformers + c;
            std::uint32_t const dst_slot = input_idx * num_conformers + c;
            host_energies_out[dst_slot] = sorted_energies[src_slot];
            host_status_out[dst_slot] = sorted_status[src_slot];
            std::memcpy(host_coords_out + static_cast<std::size_t>(dst_slot) * coord_stride,
                        sorted_coords.data() + static_cast<std::size_t>(src_slot) * coord_stride,
                        coord_stride * sizeof(float));
        }
    }

    cudaFree(device_pool);
}

conformer_status_t context_t::generate_conformers(std::span<std::string_view const> smiles, config_t const &config,
                                                  std::uint32_t max_atoms_per_molecule,
                                                  std::span<__half> coordinates_out, std::span<float> energies_out,
                                                  std::span<conformer_status_t> status_out) {

    std::uint32_t const num_input = static_cast<std::uint32_t>(smiles.size());
    std::uint32_t const num_conformers = config.num_conformers;
    std::uint32_t const total_conf = num_input * num_conformers;
    std::uint32_t const coord_stride = max_atoms_per_molecule * 3;

    // Convert string_view inputs to std::string for RDKit.
    std::vector<std::string> smiles_strings(smiles.begin(), smiles.end());

    // CPU preprocessing.
    etkdg_molecule_batch_t batch = prepare_molecule_batch(smiles_strings);

    // Clamp max_atoms to at least the batch maximum.
    std::uint32_t const effective_max_atoms = std::max(max_atoms_per_molecule, batch.max_atoms_per_molecule);

    // Allocate intermediate float32 buffers for GPU output.
    std::vector<float> host_coords(static_cast<std::size_t>(total_conf) * coord_stride, 0.0f);
    std::vector<float> host_energies(total_conf, 0.0f);
    std::vector<std::uint8_t> host_status(total_conf, 255);

    // Mark molecules that failed preprocessing as invalid_smiles.
    // build a set of successfully preprocessed input indices
    std::vector<bool> preprocessed_ok(num_input, false);
    for (std::uint32_t i = 0; i < batch.num_molecules; ++i) preprocessed_ok[batch.input_indices[i]] = true;

    std::uint32_t const num_devices = static_cast<std::uint32_t>(device_ids.size());

    if (batch.num_molecules > 0) {
        if (num_devices == 1) {
            // Single GPU: all molecules go to device 0.
            std::vector<std::uint32_t> all_indices(batch.num_molecules);
            std::iota(all_indices.begin(), all_indices.end(), 0u);
            run_gpu_pipeline(device_ids[0], streams[0], batch, all_indices.data(), batch.num_molecules, num_conformers,
                             config.max_stereo_attempts, effective_max_atoms, config.random_seed, host_coords.data(),
                             host_energies.data(), host_status.data());
        }
        else {
            // Multi-GPU: round-robin split across devices.
            std::vector<std::vector<std::uint32_t>> per_gpu_indices(num_devices);
            for (std::uint32_t i = 0; i < batch.num_molecules; ++i) per_gpu_indices[i % num_devices].push_back(i);

            std::vector<std::thread> threads;
            threads.reserve(num_devices);
            for (std::uint32_t d = 0; d < num_devices; ++d) {
                if (per_gpu_indices[d].empty()) continue;
                threads.emplace_back([&, d]() {
                    run_gpu_pipeline(device_ids[d], streams[d], batch, per_gpu_indices[d].data(),
                                     static_cast<std::uint32_t>(per_gpu_indices[d].size()), num_conformers,
                                     config.max_stereo_attempts, effective_max_atoms, config.random_seed,
                                     host_coords.data(), host_energies.data(), host_status.data());
                });
            }
            for (auto &t : threads) t.join();
        }
    }

    // Convert float32 coordinates to __half and write to output span.
    for (std::size_t i = 0; i < static_cast<std::size_t>(total_conf) * coord_stride; ++i)
        coordinates_out[i] = __float2half(host_coords[i]);

    // Map uint8_t kernel status to conformer_status_t and write to output span.
    bool any_success = false;
    for (std::uint32_t mol = 0; mol < num_input; ++mol) {
        for (std::uint32_t c = 0; c < num_conformers; ++c) {
            std::uint32_t const slot = mol * num_conformers + c;
            energies_out[slot] = host_energies[slot];

            if (!preprocessed_ok[mol]) { status_out[slot] = conformer_status_t::invalid_smiles; }
            else {
                std::uint8_t const raw = host_status[slot];
                if (raw == 0) {
                    status_out[slot] = conformer_status_t::success;
                    any_success = true;
                }
                else if (raw == 1) { status_out[slot] = conformer_status_t::stereo_failure; }
                else { status_out[slot] = conformer_status_t::mmff_unconverged; }
            }
        }
    }

    return any_success ? conformer_status_t::success : conformer_status_t::mmff_unconverged;
}

} // namespace smallfield

#if defined(SMALLFIELD_BUILD_PYTHON)

PYBIND11_MODULE(_smallfield, m) {
    m.doc() = "SmallField: GPU ETKDG + MMFF94 conformer generation";

    m.def("supports_cuda", []() { return true; });

    py::enum_<smallfield::conformer_status_t>(m, "ConformerStatus")
        .value("success", smallfield::conformer_status_t::success)
        .value("stereo_failure", smallfield::conformer_status_t::stereo_failure)
        .value("mmff_unconverged", smallfield::conformer_status_t::mmff_unconverged)
        .value("invalid_smiles", smallfield::conformer_status_t::invalid_smiles)
        .value("too_many_atoms", smallfield::conformer_status_t::too_many_atoms);

    py::class_<smallfield::context_t>(m, "Context", "GPU context managing CUDA streams and cuSOLVER handles.")
        .def(py::init([](std::vector<int> device_ids) {
                 std::span<int const> span(device_ids.data(), device_ids.size());
                 return std::make_unique<smallfield::context_t>(span);
             }),
             py::arg("device_ids") = std::vector<int> {})
        .def(
            "generate_conformers",
            [](smallfield::context_t &ctx, std::vector<std::string> const &smiles_list, std::uint32_t num_conformers,
               std::uint32_t max_atoms_per_molecule, std::uint32_t max_stereo_attempts, bool use_csd_torsions,
               std::uint64_t random_seed) {
                // Build config.
                smallfield::config_t config;
                config.num_conformers = num_conformers;
                config.max_stereo_attempts = max_stereo_attempts;
                config.use_csd_torsions = use_csd_torsions;
                config.random_seed = random_seed;

                std::uint32_t const num_mols = static_cast<std::uint32_t>(smiles_list.size());
                std::uint32_t const total_conf = num_mols * num_conformers;
                std::uint32_t const coord_stride = max_atoms_per_molecule * 3;
                std::size_t const total_coords = static_cast<std::size_t>(total_conf) * coord_stride;

                // Allocate output numpy arrays.
                auto coords_np = py::array_t<float>(static_cast<py::ssize_t>(total_coords));
                auto energies_np = py::array_t<float>(static_cast<py::ssize_t>(total_conf));
                auto status_np = py::array_t<std::uint8_t>(static_cast<py::ssize_t>(total_conf));

                // Build string_view span from the input strings.
                std::vector<std::string_view> smiles_views(smiles_list.begin(), smiles_list.end());
                std::span<std::string_view const> smiles_span(smiles_views);

                // Temporary half-precision buffer.
                std::vector<__half> half_coords(total_coords);
                std::span<__half> coords_span(half_coords);
                std::span<float> energies_span(energies_np.mutable_data(), total_conf);

                // Status buffer: use conformer_status_t then copy to uint8_t.
                std::vector<smallfield::conformer_status_t> status_buf(total_conf);
                std::span<smallfield::conformer_status_t> status_span(status_buf);

                ctx.generate_conformers(smiles_span, config, max_atoms_per_molecule, coords_span, energies_span,
                                        status_span);

                // Convert __half coordinates back to float for Python.
                float *coords_ptr = coords_np.mutable_data();
                for (std::size_t i = 0; i < total_coords; ++i) coords_ptr[i] = __half2float(half_coords[i]);

                // Copy status enum values to uint8_t numpy array.
                std::uint8_t *status_ptr = status_np.mutable_data();
                for (std::uint32_t i = 0; i < total_conf; ++i) status_ptr[i] = static_cast<std::uint8_t>(status_buf[i]);

                return py::make_tuple(coords_np, energies_np, status_np);
            },
            "Generate conformers for a batch of SMILES strings.\n" "Returns (coordinates, energies, status) as " "numpy" " " "a" "r" "r" "a" "y" "s" ".",
            py::arg("smiles"), py::arg("num_conformers") = 3, py::arg("max_atoms_per_molecule") = 128,
            py::arg("max_stereo_attempts") = 10, py::arg("use_csd_torsions") = true, py::arg("random_seed") = 42);
}

#endif // SMALLFIELD_BUILD_PYTHON
