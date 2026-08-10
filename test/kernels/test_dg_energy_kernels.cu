/**
 *  @file test_dg_energy_kernels.cu
 *  @brief CUDA kernel wrappers for DG energy test. Compiled by nvcc.
 */

#include <cstdint> // `std::uint8_t`, `std::uint32_t`

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "smallfield/cuda/dg_kernel.cuh"
#include "smallfield/cuda/fire_warp.cuh"

namespace sfc = smallfield::cuda;

/**
 *  @brief Quantize the incoming float bounds into the int8 shared-memory layout that
 *  the production `distance_geometry_energy_functor_t` consumes. The source is
 *  packed at `num_atoms`; the destination uses the bank-conflict-free padded
 *  stride, so the flat index is decomposed on the way in — mirroring the
 *  production quantization loop in `etkdg_pipeline_warp_kernel`.
 *  Returns the per-warp dequantization pair via the out-params.
 */
__device__ __forceinline__ void quantize_bounds_into_shared_(float const *bounds_float, std::uint8_t *shared_bounds,
                                                             std::uint32_t num_atoms, std::uint32_t lane,
                                                             float *out_mol_min, float *out_inv_scale) {
    std::uint32_t const bounds_size = num_atoms * num_atoms;
    std::uint32_t const bounds_row_stride = sfc::compute_bounds_row_stride(num_atoms);
    float lane_min = __FLT_MAX__;
    float lane_max = -__FLT_MAX__;
    for (std::uint32_t i = lane; i < bounds_size; i += sfc::warp_size_k) {
        float const v = bounds_float[i];
        lane_min = fminf(lane_min, v);
        lane_max = fmaxf(lane_max, v);
    }
    float const mol_min = sfc::warp_reduce_min_f32_(lane_min);
    float const mol_max = sfc::warp_reduce_max_f32_(lane_max);
    float const range = fmaxf(mol_max - mol_min, 1e-6f);
    float const scale_quantize = 255.0f / range;
    for (std::uint32_t i = lane; i < bounds_size; i += sfc::warp_size_k) {
        int q = __float2int_rn((bounds_float[i] - mol_min) * scale_quantize);
        q = max(0, min(255, q));
        std::uint32_t const row = i / num_atoms;
        std::uint32_t const column = i - row * num_atoms;
        shared_bounds[row * bounds_row_stride + column] = static_cast<std::uint8_t>(q);
    }
    __syncwarp(sfc::full_mask_k);
    *out_mol_min = mol_min;
    *out_inv_scale = range * (1.0f / 255.0f);
}

/** Evaluate DG energy with int8-quantized bounds in shared memory. */
__global__ void dg_energy_eval_kernel(float const *positions, float *gradient, float const *bounds_float,
                                      std::uint32_t num_atoms, std::uint32_t spatial_dimension, float upper_weight,
                                      float lower_weight, float fourth_dim_weight, float *out_energy) {

    extern __shared__ std::uint8_t shared_bounds[];
    std::uint32_t const lane = threadIdx.x;

    float mol_min = 0.0f, inv_scale = 0.0f;
    quantize_bounds_into_shared_(bounds_float, shared_bounds, num_atoms, lane, &mol_min, &inv_scale);

    sfc::distance_geometry_energy_functor_t functor {shared_bounds,
                                                     nullptr,
                                                     mol_min,
                                                     inv_scale,
                                                     sfc::compute_bounds_row_stride(num_atoms),
                                                     num_atoms,
                                                     0,
                                                     spatial_dimension,
                                                     upper_weight,
                                                     lower_weight,
                                                     0.0f,
                                                     fourth_dim_weight};

    std::uint32_t const num_elements = spatial_dimension * num_atoms;
    float const energy = functor(positions, gradient, num_elements, lane);

    if (lane == 0) *out_energy = energy;
}

/** Minimize DG energy with FIRE optimizer and int8-quantized bounds. */
__global__ void dg_minimize_kernel(float *positions, float const *bounds_float, std::uint32_t num_atoms,
                                   std::uint32_t spatial_dimension, float upper_weight, float lower_weight,
                                   float fourth_dim_weight, std::int32_t max_iterations, float gradient_threshold,
                                   float *out_energy, std::int32_t *out_iterations, std::int32_t *out_status) {

    extern __shared__ char shared_raw[];
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const num_elements = spatial_dimension * num_atoms;
    std::uint32_t const bounds_row_stride = sfc::compute_bounds_row_stride(num_atoms);

    // Shared layout: [bounds_uint8: N × padded row stride] [velocities + gradient: 2 × num_elements floats].
    std::uint8_t *shared_bounds = reinterpret_cast<std::uint8_t *>(shared_raw);
    float mol_min = 0.0f, inv_scale = 0.0f;
    quantize_bounds_into_shared_(bounds_float, shared_bounds, num_atoms, lane, &mol_min, &inv_scale);

    std::uint32_t const bounds_bytes_aligned =
        (num_atoms * bounds_row_stride * sizeof(std::uint8_t) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float *shared_float = reinterpret_cast<float *>(shared_raw + bounds_bytes_aligned);

    float *velocities = shared_float;
    float *gradient = velocities + num_elements;

    sfc::distance_geometry_energy_functor_t functor {shared_bounds,     nullptr,      mol_min, inv_scale,
                                                     bounds_row_stride, num_atoms,    0,       spatial_dimension,
                                                     upper_weight,      lower_weight, 0.0f,    fourth_dim_weight};

    sfc::fire_result_t const result = sfc::fire_minimize_(positions, velocities, gradient, num_elements, max_iterations,
                                                          gradient_threshold, lane, functor);
    if (lane == 0) {
        *out_energy = result.energy;
        *out_iterations = result.iterations;
        *out_status = result.status;
    }
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

    std::uint32_t const shared_bytes = num_atoms * sfc::compute_bounds_row_stride(num_atoms) * sizeof(std::uint8_t);
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

    // Shared memory: bounds_uint8 + velocities + gradient (FIRE: 2 arrays beyond positions).
    std::uint32_t const bounds_bytes_aligned =
        (num_atoms * sfc::compute_bounds_row_stride(num_atoms) * sizeof(std::uint8_t) + sizeof(float) - 1) &
        ~(sizeof(float) - 1);
    std::uint32_t const shared_bytes = bounds_bytes_aligned + 2 * num_elements * sizeof(float);

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
