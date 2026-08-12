/**
 *  @file test/cuda/test_dg_energy_kernels.cu
 *  @brief CUDA kernel wrappers for the distance-geometry energy, chirality, and FIRE tests.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::uint32_t`

#include <utility> // `std::exchange`, `std::swap`

#include <fmt/format.h> // `fmt::println` -- host-side only; nothing below a `__global__` uses it

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "smallfield/cuda/dg_kernel.cuh"
#include "smallfield/cuda/fire_warp.cuh"
#include "smallfield/types/storage.hpp"

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
    sfc::fire_scratch<sft::embedding_dimensions_k> const scratch {
        .positions = sfc::fire_positions<sft::embedding_dimensions_k> {positions, problem.num_atoms},
        .velocities = sfc::fire_velocities<sft::embedding_dimensions_k> {shared_float, problem.num_atoms},
        .gradient = sfc::fire_gradient<sft::embedding_dimensions_k> {shared_float + num_elements, problem.num_atoms}};

    auto const functor = make_functor_(problem);
    sfc::fire_result_t const result = sfc::fire_minimize_(scratch, max_iterations, gradient_threshold, lane, functor);
    if (lane == 0) {
        out_report->energy = result.energy;
        out_report->rms_gradient = result.rms_gradient;
        out_report->iterations = result.iterations;
        out_report->status = result.status;
    }
}

namespace smallfield::tests {

/** Grow-only device storage over the stream a staged problem creates for itself. */
template <typename value_type_>
using device_array = sft::buffer<value_type_, sft::device_allocator_t>;

/**
 *  @brief A non-blocking stream that outlives every buffer allocated against it.
 *
 *  Declared before those buffers so it is destroyed after them: `cudaFreeAsync` against a destroyed stream
 *  is undefined, which is the lifetime rule every stream-ordered resource in the pipeline already follows.
 */
struct owned_stream_t {
    cudaStream_t handle = nullptr;

    owned_stream_t() noexcept { cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking); }
    ~owned_stream_t() noexcept {
        if (handle == nullptr) return;
        cudaStreamSynchronize(handle);
        cudaStreamDestroy(handle);
    }

    owned_stream_t(owned_stream_t &&other) noexcept : handle(std::exchange(other.handle, nullptr)) {}
    owned_stream_t &operator=(owned_stream_t &&other) noexcept {
        std::swap(handle, other.handle);
        return *this;
    }
};

/**
 *  @brief Device-side copies of one molecule's inputs, released when the call returns.
 *
 *  Every entry point below stages the same four arrays, so the reservation and the release live here once
 *  rather than being spelled out and paired by hand at each of them. Allocation is stream-ordered:
 *  `cudaMalloc` and `cudaFree` synchronize the whole device, which the pipeline never does and a harness
 *  sharing the box with it should not either.
 */
struct staged_problem_t {
    device_problem_t problem;
    owned_stream_t stream;
    device_array<float> bounds;
    device_array<float> coordinates;
    device_array<float> gradient;
    device_array<sft::chiral_center_descriptor_t> chiral_centers;
    std::uint32_t num_elements = 0;
    /** Whether every array reached the device; a launch on a half-staged problem would score garbage. */
    bool ready = false;

    staged_problem_t(dg_problem_t const &host_problem, float const *host_coords)
        : bounds(sft::device_allocator_t {.stream = stream.handle}),
          coordinates(sft::device_allocator_t {.stream = stream.handle}),
          gradient(sft::device_allocator_t {.stream = stream.handle}),
          chiral_centers(sft::device_allocator_t {.stream = stream.handle}) {
        problem.num_atoms = host_problem.num_atoms;
        problem.num_chiral_centers = host_problem.num_chiral_centers;
        problem.spatial_dimension = host_problem.spatial_dimension;
        problem.upper_weight = host_problem.upper_weight;
        problem.lower_weight = host_problem.lower_weight;
        problem.chiral_weight = host_problem.chiral_weight;
        problem.fourth_dimension_weight = host_problem.fourth_dimension_weight;
        num_elements = host_problem.spatial_dimension * host_problem.num_atoms;

        std::size_t const bounds_entries = std::size_t {host_problem.num_atoms} * host_problem.num_atoms;
        bool reserved = true;
        auto reserve_ = [&](auto &array, std::size_t count) {
            reserved &= array.try_reserve(count) != sft::reserve_outcome_t::failed_k;
        };
        reserve_(bounds, bounds_entries);
        reserve_(coordinates, num_elements);
        reserve_(gradient, num_elements);
        reserve_(chiral_centers, host_problem.num_chiral_centers);
        problem.chiral_centers = chiral_centers.data();
        if (!reserved) return;

        ready = sft::upload_into(bounds.view(), custd::span<float const> {host_problem.bounds, bounds_entries},
                                 stream.handle) &&
                sft::upload_into(coordinates.view(), custd::span<float const> {host_coords, num_elements},
                                 stream.handle) &&
                sft::upload_into(chiral_centers.view(),
                                 custd::span<sft::chiral_center_descriptor_t const> {host_problem.chiral_centers,
                                                                                     host_problem.num_chiral_centers},
                                 stream.handle);
    }

    /** Bytes the bounds matrix alone occupies in shared memory, at the padded row stride. */
    std::uint32_t bounds_shared_bytes() const noexcept {
        return problem.num_atoms * sfc::compute_bounds_row_stride(problem.num_atoms) * sizeof(std::uint8_t);
    }
};

void gpu_dg_energy(dg_problem_t const &host_problem, float const *host_coords, float *host_gradient,
                   float *out_energy) {
    staged_problem_t staged(host_problem, host_coords);
    device_array<float> device_energy {sft::device_allocator_t {.stream = staged.stream.handle}};
    if (!staged.ready || device_energy.try_reserve(1) == sft::reserve_outcome_t::failed_k) {
        fmt::println(stderr, "NOTE dg_energy_kernels field=staging atoms={} succeeded=0", host_problem.num_atoms);
        return;
    }

    dg_energy_eval_kernel<<<1, sfc::warp_size_k, staged.bounds_shared_bytes(), staged.stream.handle>>>(
        staged.problem, staged.bounds.data(), staged.coordinates.data(), staged.gradient.data(), device_energy.data());

    bool copied = sft::download_from(custd::span<float> {out_energy, 1},
                                     custd::span<float const> {device_energy.view()}, staged.stream.handle);
    copied &= sft::download_from(custd::span<float> {host_gradient, staged.num_elements},
                                 custd::span<float const> {staged.gradient.view()}, staged.stream.handle);
    if (!copied)
        fmt::println(stderr, "NOTE dg_energy_kernels field=download atoms={} succeeded=0", host_problem.num_atoms);
    cudaStreamSynchronize(staged.stream.handle);
}

std::uint32_t gpu_count_invalid_chiral_centers(dg_problem_t const &host_problem, float const *host_coords) {
    staged_problem_t staged(host_problem, host_coords);
    device_array<std::uint32_t> device_invalid {sft::device_allocator_t {.stream = staged.stream.handle}};
    if (!staged.ready || device_invalid.try_reserve(1) == sft::reserve_outcome_t::failed_k) {
        fmt::println(stderr, "NOTE dg_energy_kernels field=staging atoms={} succeeded=0", host_problem.num_atoms);
        return 0;
    }

    dg_chirality_kernel<<<1, sfc::warp_size_k, 0, staged.stream.handle>>>(staged.problem, staged.coordinates.data(),
                                                                          device_invalid.data());

    std::uint32_t invalid = 0;
    if (!sft::download_from(custd::span<std::uint32_t> {&invalid, 1},
                            custd::span<std::uint32_t const> {device_invalid.view()}, staged.stream.handle))
        fmt::println(stderr, "NOTE dg_energy_kernels field=download atoms={} succeeded=0", host_problem.num_atoms);
    cudaStreamSynchronize(staged.stream.handle);
    return invalid;
}

void gpu_dg_minimize(dg_problem_t const &host_problem, float *host_coords, std::int32_t max_iterations,
                     float gradient_threshold, dg_minimize_report_t *out_report) {
    staged_problem_t staged(host_problem, host_coords);
    device_array<dg_minimize_report_t> device_report {sft::device_allocator_t {.stream = staged.stream.handle}};
    if (!staged.ready || device_report.try_reserve(1) == sft::reserve_outcome_t::failed_k) {
        fmt::println(stderr, "NOTE dg_energy_kernels field=staging atoms={} succeeded=0", host_problem.num_atoms);
        return;
    }

    // FIRE needs velocities and a gradient beside the bounds; positions stay in global memory.
    std::uint32_t const bounds_bytes_aligned = align_up_(staged.bounds_shared_bytes(), alignof(float));
    std::uint32_t const shared_bytes = bounds_bytes_aligned +
                                       2 * staged.num_elements * static_cast<std::uint32_t>(sizeof(float));

    dg_minimize_kernel<<<1, sfc::warp_size_k, shared_bytes, staged.stream.handle>>>(
        staged.problem, staged.bounds.data(), staged.coordinates.data(), max_iterations, gradient_threshold,
        device_report.data());

    bool copied = sft::download_from(custd::span<float> {host_coords, staged.num_elements},
                                     custd::span<float const> {staged.coordinates.view()}, staged.stream.handle);
    copied &= sft::download_from(custd::span<dg_minimize_report_t> {out_report, 1},
                                 custd::span<dg_minimize_report_t const> {device_report.view()}, staged.stream.handle);
    if (!copied)
        fmt::println(stderr, "NOTE dg_energy_kernels field=download atoms={} succeeded=0", host_problem.num_atoms);
    cudaStreamSynchronize(staged.stream.handle);
}

} // namespace smallfield::tests
