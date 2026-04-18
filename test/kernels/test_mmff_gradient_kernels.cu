/**
 *  @file test_mmff_gradient_kernels.cu
 *  @brief CUDA kernel wrapper for MMFF94 energy and gradient tests.
 */

#include <algorithm>
#include <cstdint>

#include <cuda_runtime.h>

#include "smallfield/cuda/mmff94_energy_kernel.cuh"

namespace sfc = smallfield::cuda;
namespace sft = smallfield::types;

namespace {

__global__ void mmff_energy_eval_kernel(
    float const *positions, float *gradient, sft::mmff94_atom_vdw_params_t const *atom_vdw, float const *atom_charge,
    std::uint8_t const *bond_distance, sft::mmff94_bond_t const *bond_params, sft::mmff94_angle_t const *angle_params,
    sft::mmff94_torsion_t const *torsion_params, sft::mmff94_oop_t const *oop_params,
    sft::mmff94_stretchbend_t const *stretchbend_params, sft::mmff94_atom_interaction_t const *interactions,
    sft::mmff94_atom_offsets_t const *atom_offsets, std::uint32_t num_atoms, float *out_energy) {

    std::uint32_t const lane = threadIdx.x;
    sfc::mmff94_energy_functor_t functor {atom_vdw,     atom_charge,    bond_distance, bond_params,
                                          angle_params, torsion_params, oop_params,    stretchbend_params,
                                          interactions, atom_offsets,   num_atoms};

    float const energy = functor(positions, gradient, 3 * num_atoms, lane);
    if (lane == 0) *out_energy = energy;
}

} // namespace

extern "C" {

void gpu_mmff_energy(float const *host_positions, std::uint32_t num_atoms,
                     sft::mmff94_atom_vdw_params_t const *host_atom_vdw, float const *host_atom_charge,
                     std::uint8_t const *host_bond_distance, sft::mmff94_bond_t const *host_bond_params,
                     std::uint32_t num_bond_params, sft::mmff94_angle_t const *host_angle_params,
                     std::uint32_t num_angle_params, sft::mmff94_torsion_t const *host_torsion_params,
                     std::uint32_t num_torsion_params, sft::mmff94_oop_t const *host_oop_params,
                     std::uint32_t num_oop_params, sft::mmff94_stretchbend_t const *host_stretchbend_params,
                     std::uint32_t num_stretchbend_params, sft::mmff94_atom_interaction_t const *host_interactions,
                     std::uint32_t num_interactions, sft::mmff94_atom_offsets_t const *host_atom_offsets,
                     std::uint32_t num_atom_offsets, float *host_gradient, float *out_energy) {

    std::uint32_t const num_elements = 3 * num_atoms;

    float *device_positions = nullptr;
    float *device_gradient = nullptr;
    float *device_energy = nullptr;
    sft::mmff94_atom_vdw_params_t *device_atom_vdw = nullptr;
    float *device_atom_charge = nullptr;
    std::uint8_t *device_bond_distance = nullptr;
    sft::mmff94_bond_t *device_bond_params = nullptr;
    sft::mmff94_angle_t *device_angle_params = nullptr;
    sft::mmff94_torsion_t *device_torsion_params = nullptr;
    sft::mmff94_oop_t *device_oop_params = nullptr;
    sft::mmff94_stretchbend_t *device_stretchbend_params = nullptr;
    sft::mmff94_atom_interaction_t *device_interactions = nullptr;
    sft::mmff94_atom_offsets_t *device_atom_offsets = nullptr;

    cudaMalloc(&device_positions, num_elements * sizeof(float));
    cudaMalloc(&device_gradient, num_elements * sizeof(float));
    cudaMalloc(&device_energy, sizeof(float));
    cudaMalloc(&device_atom_vdw, std::max(num_atoms, 1u) * sizeof(sft::mmff94_atom_vdw_params_t));
    cudaMalloc(&device_atom_charge, std::max(num_atoms, 1u) * sizeof(float));
    cudaMalloc(&device_bond_distance, std::max(num_atoms * num_atoms, 1u) * sizeof(std::uint8_t));
    cudaMalloc(&device_bond_params, std::max(num_bond_params, 1u) * sizeof(sft::mmff94_bond_t));
    cudaMalloc(&device_angle_params, std::max(num_angle_params, 1u) * sizeof(sft::mmff94_angle_t));
    cudaMalloc(&device_torsion_params, std::max(num_torsion_params, 1u) * sizeof(sft::mmff94_torsion_t));
    cudaMalloc(&device_oop_params, std::max(num_oop_params, 1u) * sizeof(sft::mmff94_oop_t));
    cudaMalloc(&device_stretchbend_params, std::max(num_stretchbend_params, 1u) * sizeof(sft::mmff94_stretchbend_t));
    cudaMalloc(&device_interactions, std::max(num_interactions, 1u) * sizeof(sft::mmff94_atom_interaction_t));
    cudaMalloc(&device_atom_offsets, std::max(num_atom_offsets, 1u) * sizeof(sft::mmff94_atom_offsets_t));

    cudaMemcpy(device_positions, host_positions, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    if (num_atoms > 0) {
        cudaMemcpy(device_atom_vdw, host_atom_vdw, num_atoms * sizeof(sft::mmff94_atom_vdw_params_t),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(device_atom_charge, host_atom_charge, num_atoms * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_bond_distance, host_bond_distance, num_atoms * num_atoms * sizeof(std::uint8_t),
                   cudaMemcpyHostToDevice);
    }
    if (num_bond_params > 0)
        cudaMemcpy(device_bond_params, host_bond_params, num_bond_params * sizeof(sft::mmff94_bond_t),
                   cudaMemcpyHostToDevice);
    if (num_angle_params > 0)
        cudaMemcpy(device_angle_params, host_angle_params, num_angle_params * sizeof(sft::mmff94_angle_t),
                   cudaMemcpyHostToDevice);
    if (num_torsion_params > 0)
        cudaMemcpy(device_torsion_params, host_torsion_params, num_torsion_params * sizeof(sft::mmff94_torsion_t),
                   cudaMemcpyHostToDevice);
    if (num_oop_params > 0)
        cudaMemcpy(device_oop_params, host_oop_params, num_oop_params * sizeof(sft::mmff94_oop_t),
                   cudaMemcpyHostToDevice);
    if (num_stretchbend_params > 0)
        cudaMemcpy(device_stretchbend_params, host_stretchbend_params,
                   num_stretchbend_params * sizeof(sft::mmff94_stretchbend_t), cudaMemcpyHostToDevice);
    if (num_interactions > 0)
        cudaMemcpy(device_interactions, host_interactions, num_interactions * sizeof(sft::mmff94_atom_interaction_t),
                   cudaMemcpyHostToDevice);
    if (num_atom_offsets > 0)
        cudaMemcpy(device_atom_offsets, host_atom_offsets, num_atom_offsets * sizeof(sft::mmff94_atom_offsets_t),
                   cudaMemcpyHostToDevice);

    mmff_energy_eval_kernel<<<1, 32>>>(device_positions, device_gradient, device_atom_vdw, device_atom_charge,
                                       device_bond_distance, device_bond_params, device_angle_params,
                                       device_torsion_params, device_oop_params, device_stretchbend_params,
                                       device_interactions, device_atom_offsets, num_atoms, device_energy);
    cudaDeviceSynchronize();

    cudaMemcpy(host_gradient, device_gradient, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_energy, device_energy, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_positions);
    cudaFree(device_gradient);
    cudaFree(device_energy);
    cudaFree(device_atom_vdw);
    cudaFree(device_atom_charge);
    cudaFree(device_bond_distance);
    cudaFree(device_bond_params);
    cudaFree(device_angle_params);
    cudaFree(device_torsion_params);
    cudaFree(device_oop_params);
    cudaFree(device_stretchbend_params);
    cudaFree(device_interactions);
    cudaFree(device_atom_offsets);
}

} // extern "C"
