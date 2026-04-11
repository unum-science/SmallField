/**
 *  @file test_dg_energy_kernels.cu
 *  @brief CUDA kernel wrappers for DG energy test. Compiled by nvcc.
 */

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "dg_forcefield.cuh"
#include "fire_warp.cuh"

using namespace smallfield;

/** Evaluate DG energy with half-precision bounds in shared memory. */
__global__ void dg_energy_eval_kernel(float const *positions, float *gradient, float const *bounds_float,
                                      std::uint32_t num_atoms, std::uint32_t spatial_dimension, float upper_weight,
                                      float lower_weight, float fourth_dim_weight, float *out_energy) {

    extern __shared__ __half shared_bounds[];
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const bounds_size = num_atoms * num_atoms;

    for (std::uint32_t i = lane; i < bounds_size; i += warp_size) shared_bounds[i] = __float2half(bounds_float[i]);
    __syncwarp(full_mask);

    distance_geometry_energy_functor_t functor {
        shared_bounds, nullptr, num_atoms, 0, spatial_dimension, upper_weight, lower_weight, 0.0f, fourth_dim_weight};

    std::uint32_t const num_elements = spatial_dimension * num_atoms;
    float const energy = functor(positions, gradient, num_elements, lane);

    if (lane == 0) *out_energy = energy;
}

/** Minimize DG energy with FIRE optimizer and half-precision bounds. */
__global__ void dg_minimize_kernel(float *positions, float const *bounds_float, std::uint32_t num_atoms,
                                   std::uint32_t spatial_dimension, float upper_weight, float lower_weight,
                                   float fourth_dim_weight, std::int32_t max_iterations, float gradient_threshold,
                                   float *out_energy, std::int32_t *out_iterations, std::int32_t *out_status) {

    extern __shared__ char shared_raw[];
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const num_elements = spatial_dimension * num_atoms;
    std::uint32_t const bounds_size = num_atoms * num_atoms;

    // shared memory: bounds_half + positions + velocities + gradient (FIRE needs 3 arrays)
    __half *shared_bounds = reinterpret_cast<__half *>(shared_raw);
    for (std::uint32_t i = lane; i < bounds_size; i += warp_size) shared_bounds[i] = __float2half(bounds_float[i]);
    __syncwarp(full_mask);

    std::uint32_t const bounds_half_bytes = bounds_size * sizeof(__half);
    std::uint32_t const float_offset = (bounds_half_bytes + sizeof(float) - 1) / sizeof(float);
    float *shared_float = reinterpret_cast<float *>(shared_raw) + float_offset;

    float *velocities = shared_float;
    float *gradient = velocities + num_elements;

    distance_geometry_energy_functor_t functor {
        shared_bounds, nullptr, num_atoms, 0, spatial_dimension, upper_weight, lower_weight, 0.0f, fourth_dim_weight};

    fire_minimize_(positions, velocities, gradient, num_elements, max_iterations, gradient_threshold, lane, functor,
                   out_energy, out_iterations, out_status);
}

extern "C" {

void gpu_dg_energy(float const *host_coords, float *host_gradient, float const *host_bounds, std::uint32_t num_atoms,
                   std::uint32_t spatial_dimension, float upper_weight, float lower_weight, float fourth_dim_weight,
                   float *out_energy) {

    std::uint32_t const num_elements = spatial_dimension * num_atoms;
    std::uint32_t const bounds_size = num_atoms * num_atoms;

    float *dev_coords, *dev_gradient, *dev_bounds, *dev_energy;
    cudaMalloc(&dev_coords, num_elements * sizeof(float));
    cudaMalloc(&dev_gradient, num_elements * sizeof(float));
    cudaMalloc(&dev_bounds, bounds_size * sizeof(float));
    cudaMalloc(&dev_energy, sizeof(float));

    cudaMemcpy(dev_coords, host_coords, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bounds, host_bounds, bounds_size * sizeof(float), cudaMemcpyHostToDevice);

    std::uint32_t const shared_bytes = bounds_size * sizeof(__half) + num_elements * sizeof(float);
    dg_energy_eval_kernel<<<1, 32, shared_bytes>>>(dev_coords, dev_gradient, dev_bounds, num_atoms, spatial_dimension,
                                                   upper_weight, lower_weight, fourth_dim_weight, dev_energy);
    cudaDeviceSynchronize();

    cudaMemcpy(out_energy, dev_energy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_gradient, dev_gradient, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_coords);
    cudaFree(dev_gradient);
    cudaFree(dev_bounds);
    cudaFree(dev_energy);
}

void gpu_dg_minimize(float *host_coords, float const *host_bounds, std::uint32_t num_atoms,
                     std::uint32_t spatial_dimension, float upper_weight, float lower_weight, float fourth_dim_weight,
                     std::int32_t max_iterations, float gradient_threshold, float *out_energy,
                     std::int32_t *out_iterations, std::int32_t *out_status) {

    std::uint32_t const num_elements = spatial_dimension * num_atoms;
    std::uint32_t const bounds_size = num_atoms * num_atoms;

    float *dev_coords, *dev_bounds, *dev_energy;
    std::int32_t *dev_iterations, *dev_status;
    cudaMalloc(&dev_coords, num_elements * sizeof(float));
    cudaMalloc(&dev_bounds, bounds_size * sizeof(float));
    cudaMalloc(&dev_energy, sizeof(float));
    cudaMalloc(&dev_iterations, sizeof(std::int32_t));
    cudaMalloc(&dev_status, sizeof(std::int32_t));

    cudaMemcpy(dev_coords, host_coords, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bounds, host_bounds, bounds_size * sizeof(float), cudaMemcpyHostToDevice);

    // shared memory: bounds_half + velocities + gradient (FIRE: 2 arrays beyond positions)
    std::uint32_t const bounds_half_bytes = bounds_size * sizeof(__half);
    std::uint32_t const shared_bytes = bounds_half_bytes + 2 * num_elements * sizeof(float);

    dg_minimize_kernel<<<1, 32, shared_bytes>>>(dev_coords, dev_bounds, num_atoms, spatial_dimension, upper_weight,
                                                lower_weight, fourth_dim_weight, max_iterations, gradient_threshold,
                                                dev_energy, dev_iterations, dev_status);
    cudaDeviceSynchronize();

    cudaMemcpy(host_coords, dev_coords, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_energy, dev_energy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_iterations, dev_iterations, sizeof(std::int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_status, dev_status, sizeof(std::int32_t), cudaMemcpyDeviceToHost);

    cudaFree(dev_coords);
    cudaFree(dev_bounds);
    cudaFree(dev_energy);
    cudaFree(dev_iterations);
    cudaFree(dev_status);
}

} // extern "C"
