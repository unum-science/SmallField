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

/**
 *  @brief Why a launch produced timings, or which stage stopped it.
 *
 *  A negative millisecond count used to carry all three refusals, so a driver dividing a molecule count by
 *  it reported a plausible-looking negative throughput instead of naming what failed.
 */
enum class trio_launch_t {
    /** Every kernel ran and the events were read. */
    timed_k,
    /** The MMFF94 parameter tables could not be uploaded. */
    table_upload_failed_k,
    /** A device array for the batch could not be allocated. */
    allocation_failed_k,
    /** A kernel reported an error, so the timings describe nothing. */
    kernel_failed_k,
};

/** Sentence fragment naming the refusal, meant to follow the executable's name in a diagnostic. */
char const *describe_trio_launch(trio_launch_t status);

/**
 *  @brief Time the MMFF94 kernel trio over one batch of descriptors.
 *
 *  @param[in] descriptors One per molecule, indexed alongside @p atom_offsets.
 *  @param[in] atom_offsets Where each molecule's atoms start in the shared atom arena.
 *  @param[in] total_atom_count Atoms the arena spans, which the per-atom outputs are sized against.
 *  @param[out] timings Per-kernel milliseconds, left untouched on every refusal.
 *  @return `timed_k`, or the stage that stopped the launch.
 */
trio_launch_t launch_mmff94_kernel_trio_benchmark(parameters::mmff94_tables_host_t const &tables_host,
                                                  custd::span<types::molecule_descriptor_t const> descriptors,
                                                  custd::span<std::uint32_t const> atom_offsets,
                                                  std::uint32_t total_atom_count, kernel_trio_timings_t &timings);

} // namespace smallfield::bench
