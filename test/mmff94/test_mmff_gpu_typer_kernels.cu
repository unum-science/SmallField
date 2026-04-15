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
#include "parameters/mmff94_tables.hpp"
#include "parameters/mmff94_tables_device.cuh"
#include "mmff94/mmff94_typer_device.cuh"
#include "types/molecule_descriptor.hpp"

namespace smallfield {

/** Takes an array of N host-side descriptors, uploads them, runs the
 *  kernel, returns atom types + charges + statuses via out-parameters.
 *
 *  Host-only path: this function is called from test_mmff_gpu_typer.cpp
 *  (compiled by g++); the kernel launch lives in this .cu. */
void run_mmff94_gpu_typer(mmff94_tables_view_t tables_view,
                          molecule_descriptor_t const *descriptors,
                          std::uint32_t num_molecules,
                          std::uint32_t const *atom_offsets,
                          std::uint32_t total_atoms,
                          std::uint8_t *atom_types_out,
                          float *atom_charges_out,
                          std::uint8_t *typing_status_out) {
    molecule_descriptor_t *desc_dev = nullptr;
    std::uint32_t *offsets_dev = nullptr;
    std::uint8_t *types_dev = nullptr;
    float *charges_dev = nullptr;
    std::uint8_t *status_dev = nullptr;
    cudaMalloc(&desc_dev, num_molecules * sizeof(molecule_descriptor_t));
    cudaMalloc(&offsets_dev, num_molecules * sizeof(std::uint32_t));
    cudaMalloc(&types_dev, total_atoms * sizeof(std::uint8_t));
    cudaMalloc(&charges_dev, total_atoms * sizeof(float));
    cudaMalloc(&status_dev, num_molecules * sizeof(std::uint8_t));

    cudaMemcpy(desc_dev, descriptors,
                num_molecules * sizeof(molecule_descriptor_t), cudaMemcpyHostToDevice);
    cudaMemcpy(offsets_dev, atom_offsets,
                num_molecules * sizeof(std::uint32_t), cudaMemcpyHostToDevice);

    launch_mmff94_type_and_charge(desc_dev, offsets_dev, num_molecules, tables_view,
                                   types_dev, charges_dev, status_dev);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "GPU typer kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(atom_types_out, types_dev, total_atoms * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(atom_charges_out, charges_dev, total_atoms * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(typing_status_out, status_dev, num_molecules * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(desc_dev);
    cudaFree(offsets_dev);
    cudaFree(types_dev);
    cudaFree(charges_dev);
    cudaFree(status_dev);
}

void *device_tables_create(mmff94_tables_host_t const &host,
                           mmff94_tables_view_t &out_view) {
    auto *dev = new mmff94_tables_device_t;
    upload_mmff94_tables_to_device(host, *dev);
    if (dev->arena == nullptr) {
        delete dev;
        return nullptr;
    }
    out_view = dev->view;
    return dev;
}

void device_tables_destroy(void *handle) {
    delete static_cast<mmff94_tables_device_t *>(handle);
}

} // namespace smallfield
