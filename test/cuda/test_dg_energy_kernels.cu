/**
 *  @file test/cuda/test_dg_energy_kernels.cu
 *  @brief CUDA kernel wrappers for the distance-geometry energy, chirality, and FIRE tests.
 */

#include <cstdint> // `std::uint8_t`, `std::uint32_t`

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "smallfield/cuda/dg_kernel.cuh"
#include "smallfield/cuda/fire_warp.cuh"

#include "test/cuda/test_dg_energy_kernels.hpp"

namespace sfc = smallfield::cuda;
namespace sft = smallfield::types;

namespace smallfield::tests {

/** The device-side mirror of `dg_problem_t`, with the two host pointers replaced by device ones. */
struct device_problem_t {
    std::uint8_t const *bounds = nullptr;
    sft::chiral_center_descriptor_t const *chiral_centers = nullptr;
    float mol_min = 0.0f;
    float inverse_scale = 0.0f;
    std::uint32_t num_atoms = 0;
    std::uint32_t num_chiral_centers = 0;
    std::uint32_t spatial_dimension = 4;
    float upper_weight = 1.0f;
    float lower_weight = 1.0f;
    float chiral_weight = 0.0f;
    float fourth_dimension_weight = 0.0f;
};

} // namespace smallfield::tests

namespace sftest = smallfield::tests;

/**
 *  @brief Quantize the incoming float bounds into the int8 shared-memory layout the production
 *  `distance_geometry_energy_functor_t` consumes.
 *
 *  The source is packed at `num_atoms`; the destination uses the bank-conflict-free padded stride,
 *  so the flat index is decomposed on the way in — mirroring the production quantization loop in
 *  `etkdg_pipeline_warp_kernel`. Fills in the dequantization pair the functor reads back through.
 */
__device__ __forceinline__ void quantize_bounds_into_shared_(float const *bounds_float, std::uint8_t *shared_bounds,
                                                             std::uint32_t lane, sftest::device_problem_t *problem) {
    std::uint32_t const num_atoms = problem->num_atoms;
    std::uint32_t const bounds_size = num_atoms * num_atoms;
    std::uint32_t const bounds_row_stride = sfc::compute_bounds_row_stride(num_atoms);
    float lane_min = __FLT_MAX__;
    float lane_max = -__FLT_MAX__;
    for (std::uint32_t entry = lane; entry < bounds_size; entry += sfc::warp_size_k) {
        float const value = bounds_float[entry];
        lane_min = fminf(lane_min, value);
        lane_max = fmaxf(lane_max, value);
    }
    float const mol_min = sfc::warp_reduce_min_f32_(lane_min);
    float const mol_max = sfc::warp_reduce_max_f32_(lane_max);
    float const range = fmaxf(mol_max - mol_min, 1e-6f);
    float const scale_quantize = 255.0f / range;
    for (std::uint32_t entry = lane; entry < bounds_size; entry += sfc::warp_size_k) {
        int quantized = __float2int_rn((bounds_float[entry] - mol_min) * scale_quantize);
        quantized = max(0, min(255, quantized));
        std::uint32_t const row = entry / num_atoms;
        std::uint32_t const column = entry - row * num_atoms;
        shared_bounds[row * bounds_row_stride + column] = static_cast<std::uint8_t>(quantized);
    }
    __syncwarp(sfc::full_mask_k);
    problem->bounds = shared_bounds;
    problem->mol_min = mol_min;
    problem->inverse_scale = range * (1.0f / 255.0f);
}

/** Build the production functor from a staged problem, so both kernels below spell it once. */
__device__ __forceinline__ sfc::distance_geometry_energy_functor_t make_functor_(
    sftest::device_problem_t const &problem) {
    return sfc::distance_geometry_energy_functor_t {
        .bounds_matrix = problem.bounds,
        .chiral_centers = problem.chiral_centers,
        .mol_min = problem.mol_min,
        .inv_scale = problem.inverse_scale,
        .bounds_row_stride = sfc::compute_bounds_row_stride(problem.num_atoms),
        .num_atoms = problem.num_atoms,
        .num_chiral_centers = problem.num_chiral_centers,
        .spatial_dimension = problem.spatial_dimension,
        .upper_violation_weight = problem.upper_weight,
        .lower_violation_weight = problem.lower_weight,
        .chiral_volume_weight = problem.chiral_weight,
        .fourth_dimension_penalty_weight = problem.fourth_dimension_weight};
}

/** Evaluate DG energy with int8-quantized bounds in shared memory. */
__global__ void dg_energy_eval_kernel(sftest::device_problem_t problem, float const *bounds_float,
                                      float const *positions, float *gradient, float *out_energy) {
    extern __shared__ std::uint8_t shared_bounds[];
    std::uint32_t const lane = threadIdx.x;
    quantize_bounds_into_shared_(bounds_float, shared_bounds, lane, &problem);

    std::uint32_t const num_elements = problem.spatial_dimension * problem.num_atoms;
    float const energy = make_functor_(problem)(positions, gradient, num_elements, lane);
    if (lane == 0) *out_energy = energy;
}

/** Count the chiral centers whose geometry contradicts their descriptor. */
__global__ void dg_chirality_kernel(sftest::device_problem_t problem, float const *positions,
                                    std::uint32_t *out_invalid) {
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const invalid = sfc::count_invalid_chiral_centers_(
        positions, problem.num_atoms, problem.chiral_centers, problem.num_chiral_centers, lane);
    if (lane == 0) *out_invalid = invalid;
}

/** Minimize DG energy with the FIRE optimizer and int8-quantized bounds. */
__global__ void dg_minimize_kernel(sftest::device_problem_t problem, float const *bounds_float, float *positions,
                                   std::int32_t max_iterations, float gradient_threshold,
                                   sftest::dg_minimize_report_t *out_report) {
    extern __shared__ char shared_raw[];
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const num_elements = problem.spatial_dimension * problem.num_atoms;
    std::uint32_t const bounds_row_stride = sfc::compute_bounds_row_stride(problem.num_atoms);

    // Shared layout: [bounds_uint8: N × padded row stride] [velocities + gradient: 2 × num_elements floats].
    quantize_bounds_into_shared_(bounds_float, reinterpret_cast<std::uint8_t *>(shared_raw), lane, &problem);

    std::uint32_t const bounds_bytes_aligned =
        (problem.num_atoms * bounds_row_stride * sizeof(std::uint8_t) + sizeof(float) - 1) & ~(sizeof(float) - 1);
    float *shared_float = reinterpret_cast<float *>(shared_raw + bounds_bytes_aligned);
    float *velocities = shared_float;
    float *gradient = velocities + num_elements;

    auto const functor = make_functor_(problem);
    sfc::fire_result_t const result = sfc::fire_minimize_(positions, velocities, gradient, num_elements, max_iterations,
                                                          gradient_threshold, lane, functor);
    if (lane == 0) {
        out_report->energy = result.energy;
        out_report->rms_gradient = result.rms_gradient;
        out_report->iterations = result.iterations;
        out_report->status = result.status;
    }
}

namespace smallfield::tests {

/**
 *  @brief Device-side copies of one molecule's inputs, released when the call returns.
 *
 *  Every entry point below stages the same four arrays, so the allocation and the release live
 *  here once rather than being spelled out and paired by hand at each of them. Allocation is
 *  stream-ordered: `cudaMalloc` and `cudaFree` synchronize the whole device, which the pipeline
 *  never does and a harness sharing the box with it should not either.
 */
struct staged_problem_t {
    device_problem_t problem;
    cudaStream_t stream = nullptr;
    float *bounds = nullptr;
    float *coordinates = nullptr;
    float *gradient = nullptr;
    sft::chiral_center_descriptor_t *chiral_centers = nullptr;
    std::uint32_t num_elements = 0;

    staged_problem_t(dg_problem_t const &host_problem, float const *host_coords) {
        problem.num_atoms = host_problem.num_atoms;
        problem.num_chiral_centers = host_problem.num_chiral_centers;
        problem.spatial_dimension = host_problem.spatial_dimension;
        problem.upper_weight = host_problem.upper_weight;
        problem.lower_weight = host_problem.lower_weight;
        problem.chiral_weight = host_problem.chiral_weight;
        problem.fourth_dimension_weight = host_problem.fourth_dimension_weight;
        num_elements = host_problem.spatial_dimension * host_problem.num_atoms;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

        std::size_t const bounds_bytes = std::size_t {host_problem.num_atoms} * host_problem.num_atoms * sizeof(float);
        std::size_t const coordinate_bytes = std::size_t {num_elements} * sizeof(float);
        cudaMallocAsync(&bounds, bounds_bytes, stream);
        cudaMallocAsync(&coordinates, coordinate_bytes, stream);
        cudaMallocAsync(&gradient, coordinate_bytes, stream);
        cudaMemcpyAsync(bounds, host_problem.bounds, bounds_bytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(coordinates, host_coords, coordinate_bytes, cudaMemcpyHostToDevice, stream);

        if (host_problem.num_chiral_centers != 0) {
            std::size_t const chiral_bytes = std::size_t {host_problem.num_chiral_centers} *
                                             sizeof(sft::chiral_center_descriptor_t);
            cudaMallocAsync(&chiral_centers, chiral_bytes, stream);
            cudaMemcpyAsync(chiral_centers, host_problem.chiral_centers, chiral_bytes, cudaMemcpyHostToDevice, stream);
        }
        problem.chiral_centers = chiral_centers;
    }

    ~staged_problem_t() {
        cudaFreeAsync(bounds, stream);
        cudaFreeAsync(coordinates, stream);
        cudaFreeAsync(gradient, stream);
        cudaFreeAsync(chiral_centers, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    staged_problem_t(staged_problem_t const &) = delete;
    staged_problem_t &operator=(staged_problem_t const &) = delete;

    /** Bytes the bounds matrix alone occupies in shared memory, at the padded row stride. */
    std::uint32_t bounds_shared_bytes() const noexcept {
        return problem.num_atoms * sfc::compute_bounds_row_stride(problem.num_atoms) * sizeof(std::uint8_t);
    }
};

void gpu_dg_energy(dg_problem_t const &host_problem, float const *host_coords, float *host_gradient,
                   float *out_energy) {
    staged_problem_t staged(host_problem, host_coords);
    float *device_energy = nullptr;
    cudaMallocAsync(&device_energy, sizeof(float), staged.stream);

    dg_energy_eval_kernel<<<1, 32, staged.bounds_shared_bytes(), staged.stream>>>(
        staged.problem, staged.bounds, staged.coordinates, staged.gradient, device_energy);

    cudaMemcpyAsync(out_energy, device_energy, sizeof(float), cudaMemcpyDeviceToHost, staged.stream);
    cudaMemcpyAsync(host_gradient, staged.gradient, std::size_t {staged.num_elements} * sizeof(float),
                    cudaMemcpyDeviceToHost, staged.stream);
    cudaFreeAsync(device_energy, staged.stream);
    cudaStreamSynchronize(staged.stream);
}

std::uint32_t gpu_count_invalid_chiral_centers(dg_problem_t const &host_problem, float const *host_coords) {
    staged_problem_t staged(host_problem, host_coords);
    std::uint32_t *device_invalid = nullptr;
    cudaMallocAsync(&device_invalid, sizeof(std::uint32_t), staged.stream);

    dg_chirality_kernel<<<1, 32, 0, staged.stream>>>(staged.problem, staged.coordinates, device_invalid);

    std::uint32_t invalid = 0;
    cudaMemcpyAsync(&invalid, device_invalid, sizeof(std::uint32_t), cudaMemcpyDeviceToHost, staged.stream);
    cudaFreeAsync(device_invalid, staged.stream);
    cudaStreamSynchronize(staged.stream);
    return invalid;
}

void gpu_dg_minimize(dg_problem_t const &host_problem, float *host_coords, std::int32_t max_iterations,
                     float gradient_threshold, dg_minimize_report_t *out_report) {
    staged_problem_t staged(host_problem, host_coords);
    dg_minimize_report_t *device_report = nullptr;
    cudaMallocAsync(&device_report, sizeof(dg_minimize_report_t), staged.stream);

    // FIRE needs velocities and a gradient beside the bounds; positions stay in global memory.
    std::uint32_t const bounds_bytes_aligned = (staged.bounds_shared_bytes() + sizeof(float) - 1) &
                                               ~(sizeof(float) - 1);
    std::uint32_t const shared_bytes = bounds_bytes_aligned + 2 * staged.num_elements * sizeof(float);

    dg_minimize_kernel<<<1, 32, shared_bytes, staged.stream>>>(staged.problem, staged.bounds, staged.coordinates,
                                                               max_iterations, gradient_threshold, device_report);

    cudaMemcpyAsync(host_coords, staged.coordinates, std::size_t {staged.num_elements} * sizeof(float),
                    cudaMemcpyDeviceToHost, staged.stream);
    cudaMemcpyAsync(out_report, device_report, sizeof(dg_minimize_report_t), cudaMemcpyDeviceToHost, staged.stream);
    cudaFreeAsync(device_report, staged.stream);
    cudaStreamSynchronize(staged.stream);
}

} // namespace smallfield::tests
