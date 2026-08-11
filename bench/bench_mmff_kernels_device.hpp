/**
 *  @file bench/bench_mmff_kernels_device.hpp
 *  @brief Device-side launcher for the GPU MMFF kernel-trio scaling benchmark.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::bench {

void launch_mmff94_kernel_trio_benchmark(smallfield::parameters::mmff94_tables_host_t const &tables_host,
                                         smallfield::types::molecule_descriptor_t const *descriptors,
                                         std::uint32_t num_molecules, std::uint32_t const *atom_offsets,
                                         std::uint32_t total_atom_count, float *type_and_charge_ms,
                                         float *extract_terms_ms, float *build_csr_ms, float *total_gpu_ms);

} // namespace smallfield::bench
