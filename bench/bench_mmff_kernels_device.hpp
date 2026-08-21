/**
 *  @file bench/bench_mmff_kernels_device.hpp
 *  @brief Device-side launcher for the GPU MMFF kernel-trio scaling benchmark.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::bench {

/** Milliseconds each kernel of the trio took, and the span of the three together. */
struct kernel_trio_timings_t {
    float type_and_charge_ms = 0.0f;
    float extract_terms_ms = 0.0f;
    float build_csr_ms = 0.0f;
    float total_ms = 0.0f;
};

/** Why a launch produced timings, or which stage stopped it; `describe_trio_launch` renders each. */
enum class trio_launch_t {
    timed_k,
    table_upload_failed_k,
    allocation_failed_k,
    kernel_failed_k,
};

/** Sentence fragment naming the refusal, meant to follow the executable's name in a diagnostic. */
char const *describe_trio_launch(trio_launch_t status);

/**
 *  @brief Time the MMFF94 kernel trio over one batch of descriptors.
 *
 *  @param[in] atom_offsets Where each molecule's atoms start in the shared arena, walked by @p descriptors' index.
 *  @param[out] timings Left untouched on every refusal.
 */
trio_launch_t launch_mmff94_kernel_trio_benchmark(parameters::mmff94_tables_host_t const &tables_host,
                                                  custd::span<types::molecule_descriptor_t const> descriptors,
                                                  custd::span<std::uint32_t const> atom_offsets,
                                                  std::uint32_t total_atom_count, kernel_trio_timings_t &timings);

} // namespace smallfield::bench
