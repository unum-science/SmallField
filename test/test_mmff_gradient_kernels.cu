/**
 * @file test_mmff_gradient_kernels.cu
 * @brief CUDA kernel wrapper for MMFF94 energy and gradient tests.
 */

#include <algorithm>
#include <cstdint>

#include <cuda_runtime.h>

#include "mmff94_functor.cuh"

using namespace smallfield;

namespace {

__global__ void
mmff_energy_eval_kernel(float const* positions, float* gradient, mmff94_nonbonded_pair_t const* nonbonded_params,
                        mmff94_bond_t const* bond_params, mmff94_angle_t const* angle_params,
                        mmff94_torsion_t const* torsion_params, mmff94_oop_t const* oop_params,
                        mmff94_stretchbend_t const* stretchbend_params, mmff94_atom_interaction_t const* interactions,
                        mmff94_atom_offsets_t const* atom_offsets, std::uint32_t num_atoms, float* out_energy) {

    std::uint32_t const lane = threadIdx.x;
    mmff94_energy_functor_t functor{nonbonded_params,   bond_params,  angle_params, torsion_params, oop_params,
                                    stretchbend_params, interactions, atom_offsets, num_atoms};

    float const energy = functor(positions, gradient, 3 * num_atoms, lane);
    if (lane == 0)
        *out_energy = energy;
}

} // namespace

extern "C" {

void gpu_mmff_energy(float const* host_positions, std::uint32_t num_atoms,
                     mmff94_nonbonded_pair_t const* host_nonbonded_params, std::uint32_t num_nonbonded_params,
                     mmff94_bond_t const* host_bond_params, std::uint32_t num_bond_params,
                     mmff94_angle_t const* host_angle_params, std::uint32_t num_angle_params,
                     mmff94_torsion_t const* host_torsion_params, std::uint32_t num_torsion_params,
                     mmff94_oop_t const* host_oop_params, std::uint32_t num_oop_params,
                     mmff94_stretchbend_t const* host_stretchbend_params, std::uint32_t num_stretchbend_params,
                     mmff94_atom_interaction_t const* host_interactions, std::uint32_t num_interactions,
                     mmff94_atom_offsets_t const* host_atom_offsets, std::uint32_t num_atom_offsets,
                     float* host_gradient, float* out_energy) {

    std::uint32_t const num_elements = 3 * num_atoms;

    float* dev_positions = nullptr;
    float* dev_gradient = nullptr;
    float* dev_energy = nullptr;
    mmff94_nonbonded_pair_t* dev_nonbonded_params = nullptr;
    mmff94_bond_t* dev_bond_params = nullptr;
    mmff94_angle_t* dev_angle_params = nullptr;
    mmff94_torsion_t* dev_torsion_params = nullptr;
    mmff94_oop_t* dev_oop_params = nullptr;
    mmff94_stretchbend_t* dev_stretchbend_params = nullptr;
    mmff94_atom_interaction_t* dev_interactions = nullptr;
    mmff94_atom_offsets_t* dev_atom_offsets = nullptr;

    cudaMalloc(&dev_positions, num_elements * sizeof(float));
    cudaMalloc(&dev_gradient, num_elements * sizeof(float));
    cudaMalloc(&dev_energy, sizeof(float));
    cudaMalloc(&dev_nonbonded_params, std::max(num_nonbonded_params, 1u) * sizeof(mmff94_nonbonded_pair_t));
    cudaMalloc(&dev_bond_params, std::max(num_bond_params, 1u) * sizeof(mmff94_bond_t));
    cudaMalloc(&dev_angle_params, std::max(num_angle_params, 1u) * sizeof(mmff94_angle_t));
    cudaMalloc(&dev_torsion_params, std::max(num_torsion_params, 1u) * sizeof(mmff94_torsion_t));
    cudaMalloc(&dev_oop_params, std::max(num_oop_params, 1u) * sizeof(mmff94_oop_t));
    cudaMalloc(&dev_stretchbend_params, std::max(num_stretchbend_params, 1u) * sizeof(mmff94_stretchbend_t));
    cudaMalloc(&dev_interactions, std::max(num_interactions, 1u) * sizeof(mmff94_atom_interaction_t));
    cudaMalloc(&dev_atom_offsets, std::max(num_atom_offsets, 1u) * sizeof(mmff94_atom_offsets_t));

    cudaMemcpy(dev_positions, host_positions, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    if (num_nonbonded_params > 0)
        cudaMemcpy(dev_nonbonded_params, host_nonbonded_params, num_nonbonded_params * sizeof(mmff94_nonbonded_pair_t),
                   cudaMemcpyHostToDevice);
    if (num_bond_params > 0)
        cudaMemcpy(dev_bond_params, host_bond_params, num_bond_params * sizeof(mmff94_bond_t), cudaMemcpyHostToDevice);
    if (num_angle_params > 0)
        cudaMemcpy(dev_angle_params, host_angle_params, num_angle_params * sizeof(mmff94_angle_t),
                   cudaMemcpyHostToDevice);
    if (num_torsion_params > 0)
        cudaMemcpy(dev_torsion_params, host_torsion_params, num_torsion_params * sizeof(mmff94_torsion_t),
                   cudaMemcpyHostToDevice);
    if (num_oop_params > 0)
        cudaMemcpy(dev_oop_params, host_oop_params, num_oop_params * sizeof(mmff94_oop_t), cudaMemcpyHostToDevice);
    if (num_stretchbend_params > 0)
        cudaMemcpy(dev_stretchbend_params, host_stretchbend_params,
                   num_stretchbend_params * sizeof(mmff94_stretchbend_t), cudaMemcpyHostToDevice);
    if (num_interactions > 0)
        cudaMemcpy(dev_interactions, host_interactions, num_interactions * sizeof(mmff94_atom_interaction_t),
                   cudaMemcpyHostToDevice);
    if (num_atom_offsets > 0)
        cudaMemcpy(dev_atom_offsets, host_atom_offsets, num_atom_offsets * sizeof(mmff94_atom_offsets_t),
                   cudaMemcpyHostToDevice);

    mmff_energy_eval_kernel<<<1, 32>>>(dev_positions, dev_gradient, dev_nonbonded_params, dev_bond_params,
                                       dev_angle_params, dev_torsion_params, dev_oop_params, dev_stretchbend_params,
                                       dev_interactions, dev_atom_offsets, num_atoms, dev_energy);
    cudaDeviceSynchronize();

    cudaMemcpy(host_gradient, dev_gradient, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_energy, dev_energy, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_positions);
    cudaFree(dev_gradient);
    cudaFree(dev_energy);
    cudaFree(dev_nonbonded_params);
    cudaFree(dev_bond_params);
    cudaFree(dev_angle_params);
    cudaFree(dev_torsion_params);
    cudaFree(dev_oop_params);
    cudaFree(dev_stretchbend_params);
    cudaFree(dev_interactions);
    cudaFree(dev_atom_offsets);
}

} // extern "C"
