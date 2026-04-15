/**
 *  @file test_mmff_gpu_typer_kernels.cu
 *  @brief Validate the GPU MMFF typer kernel: run it on a batch of
 *  descriptors and hand results back to the host .cpp for byte-level
 *  comparison vs the CPU native typer.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define SMALLFIELD_MMFF_TABLES_NO_RDKIT 1
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_device.cuh"
#include "smallfield/mmff94/mmff94_typer_device.cuh"
#include "test_mmff_gpu_typer_kernels.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace sfmmff = smallfield::mmff94;
namespace sfparam = smallfield::parameters;
namespace sft = smallfield::types;

/** Takes an array of N host-side descriptors, uploads them, runs the
 *  kernel, returns atom types + charges + statuses via out-parameters.
 *
 *  Host-only path: this function is called from test_mmff_gpu_typer.cpp
 *  (compiled by g++); the kernel launch lives in this .cu. */
void launch_mmff94_gpu_typer_batch(sfparam::mmff94_tables_view_t tables_view,
                                   sft::molecule_descriptor_t const *descriptors, std::uint32_t num_molecules,
                                   std::uint32_t const *atom_offsets, std::uint32_t total_atoms,
                                   std::uint8_t *atom_types_out, float *atom_charges_out,
                                   std::uint8_t *typing_status_out) {
    sft::molecule_descriptor_t *descriptor_device = nullptr;
    std::uint32_t *atom_offset_device = nullptr;
    std::uint8_t *atom_types_device = nullptr;
    float *atom_charges_device = nullptr;
    std::uint8_t *typing_status_device = nullptr;
    cudaMalloc(&descriptor_device, num_molecules * sizeof(sft::molecule_descriptor_t));
    cudaMalloc(&atom_offset_device, num_molecules * sizeof(std::uint32_t));
    cudaMalloc(&atom_types_device, total_atoms * sizeof(std::uint8_t));
    cudaMalloc(&atom_charges_device, total_atoms * sizeof(float));
    cudaMalloc(&typing_status_device, num_molecules * sizeof(std::uint8_t));

    cudaMemcpy(descriptor_device, descriptors, num_molecules * sizeof(sft::molecule_descriptor_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(atom_offset_device, atom_offsets, num_molecules * sizeof(std::uint32_t), cudaMemcpyHostToDevice);

    sfmmff::launch_mmff94_type_and_charge(descriptor_device, atom_offset_device, num_molecules, tables_view,
                                          atom_types_device, atom_charges_device, typing_status_device);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { std::fprintf(stderr, "GPU typer kernel error: %s\n", cudaGetErrorString(error)); }

    cudaMemcpy(atom_types_out, atom_types_device, total_atoms * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(atom_charges_out, atom_charges_device, total_atoms * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(typing_status_out, typing_status_device, num_molecules * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(descriptor_device);
    cudaFree(atom_offset_device);
    cudaFree(atom_types_device);
    cudaFree(atom_charges_device);
    cudaFree(typing_status_device);
}

void *create_mmff94_device_tables_view(sfparam::mmff94_tables_host_t const &host,
                                       sfparam::mmff94_tables_view_t &out_view) {
    auto *device_tables = new sfparam::mmff94_tables_device_t;
    sfparam::upload_mmff94_tables_to_device(host, *device_tables);
    if (device_tables->arena == nullptr) {
        delete device_tables;
        return nullptr;
    }
    out_view = device_tables->view;
    return device_tables;
}

void destroy_mmff94_device_tables_view(void *handle) { delete static_cast<sfparam::mmff94_tables_device_t *>(handle); }
