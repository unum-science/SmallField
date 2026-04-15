/**
 *  @file test_dg_energy_kernels.cu
 *  @brief CUDA kernel wrappers for DG energy test. Compiled by nvcc.
 */

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "smallfield/cuda/dg_kernel.cuh"
#include "smallfield/cuda/fire_warp.cuh"

namespace sfc = smallfield::cuda;

/** Evaluate DG energy with half-precision bounds in shared memory. */
__global__ void dg_energy_eval_kernel(float const *positions, float *gradient, float const *bounds_float,
                                      std::uint32_t num_atoms, std::uint32_t spatial_dimension, float upper_weight,
                                      float lower_weight, float fourth_dim_weight, float *out_energy) {

    extern __shared__ __half shared_bounds[];
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const bounds_size = num_atoms * num_atoms;

    for (std::uint32_t i = lane; i < bounds_size; i += sfc::warp_size) shared_bounds[i] = __float2half(bounds_float[i]);
    __syncwarp(sfc::full_mask);

    sfc::distance_geometry_energy_functor_t functor {
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
    for (std::uint32_t i = lane; i < bounds_size; i += sfc::warp_size) shared_bounds[i] = __float2half(bounds_float[i]);
    __syncwarp(sfc::full_mask);

    std::uint32_t const bounds_half_bytes = bounds_size * sizeof(__half);
    std::uint32_t const float_offset = (bounds_half_bytes + sizeof(float) - 1) / sizeof(float);
    float *shared_float = reinterpret_cast<float *>(shared_raw) + float_offset;

    float *velocities = shared_float;
    float *gradient = velocities + num_elements;

    sfc::distance_geometry_energy_functor_t functor {
        shared_bounds, nullptr, num_atoms, 0, spatial_dimension, upper_weight, lower_weight, 0.0f, fourth_dim_weight};

    sfc::fire_minimize_(positions, velocities, gradient, num_elements, max_iterations, gradient_threshold, lane,
                        functor, out_energy, out_iterations, out_status);
}

extern "C" {

void gpu_dg_energy(float const *host_coords, float *host_gradient, float const *host_bounds, std::uint32_t num_atoms,
                   std::uint32_t spatial_dimension, float upper_weight, float lower_weight, float fourth_dim_weight,
                   float *out_energy) {

    std::uint32_t const num_elements = spatial_dimension * num_atoms;
    std::uint32_t const bounds_size = num_atoms * num_atoms;

    float *device_coords, *device_gradient, *device_bounds, *device_energy;
    cudaMalloc(&device_coords, num_elements * sizeof(float));
    cudaMalloc(&device_gradient, num_elements * sizeof(float));
    cudaMalloc(&device_bounds, bounds_size * sizeof(float));
    cudaMalloc(&device_energy, sizeof(float));

    cudaMemcpy(device_coords, host_coords, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_bounds, host_bounds, bounds_size * sizeof(float), cudaMemcpyHostToDevice);

    std::uint32_t const shared_bytes = bounds_size * sizeof(__half) + num_elements * sizeof(float);
    dg_energy_eval_kernel<<<1, 32, shared_bytes>>>(device_coords, device_gradient, device_bounds, num_atoms,
                                                   spatial_dimension, upper_weight, lower_weight, fourth_dim_weight,
                                                   device_energy);
    cudaDeviceSynchronize();

    cudaMemcpy(out_energy, device_energy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_gradient, device_gradient, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_coords);
    cudaFree(device_gradient);
    cudaFree(device_bounds);
    cudaFree(device_energy);
}

void gpu_dg_minimize(float *host_coords, float const *host_bounds, std::uint32_t num_atoms,
                     std::uint32_t spatial_dimension, float upper_weight, float lower_weight, float fourth_dim_weight,
                     std::int32_t max_iterations, float gradient_threshold, float *out_energy,
                     std::int32_t *out_iterations, std::int32_t *out_status) {

    std::uint32_t const num_elements = spatial_dimension * num_atoms;
    std::uint32_t const bounds_size = num_atoms * num_atoms;

    float *device_coords, *device_bounds, *device_energy;
    std::int32_t *device_iterations, *device_status;
    cudaMalloc(&device_coords, num_elements * sizeof(float));
    cudaMalloc(&device_bounds, bounds_size * sizeof(float));
    cudaMalloc(&device_energy, sizeof(float));
    cudaMalloc(&device_iterations, sizeof(std::int32_t));
    cudaMalloc(&device_status, sizeof(std::int32_t));

    cudaMemcpy(device_coords, host_coords, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_bounds, host_bounds, bounds_size * sizeof(float), cudaMemcpyHostToDevice);

    // shared memory: bounds_half + velocities + gradient (FIRE: 2 arrays beyond positions)
    std::uint32_t const bounds_half_bytes = bounds_size * sizeof(__half);
    std::uint32_t const shared_bytes = bounds_half_bytes + 2 * num_elements * sizeof(float);

    dg_minimize_kernel<<<1, 32, shared_bytes>>>(device_coords, device_bounds, num_atoms, spatial_dimension,
                                                upper_weight, lower_weight, fourth_dim_weight, max_iterations,
                                                gradient_threshold, device_energy, device_iterations, device_status);
    cudaDeviceSynchronize();

    cudaMemcpy(host_coords, device_coords, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_energy, device_energy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_iterations, device_iterations, sizeof(std::int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_status, device_status, sizeof(std::int32_t), cudaMemcpyDeviceToHost);

    cudaFree(device_coords);
    cudaFree(device_bounds);
    cudaFree(device_energy);
    cudaFree(device_iterations);
    cudaFree(device_status);
}

} // extern "C"
