/**
 *  @file test/parameters/mmff94_tables_kernels.hpp
 *  @brief Device launchers for the MMFF94 parameter table lookups, for use by their unit tests.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

#include <cuda_runtime.h>

#include "smallfield/parameters/mmff94_tables.hpp"

#include "test/parameters/mmff94_lookup_probes.hpp"

namespace smallfield::tests {

/*  Each launcher takes the stream its buffers were allocated and staged on. A probe stream is non-blocking, so it
 *  is exempt from the legacy default stream's implicit synchronization, and a launch left on that default would be
 *  free to read the queries before their upload landed. */

void launch_device_vdw_lookup(smallfield::parameters::mmff94_tables_view_t view, vdw_query_t const *queries,
                              vdw_result_t *results, std::uint32_t query_count, cudaStream_t stream);
void launch_device_bond_lookup(smallfield::parameters::mmff94_tables_view_t view, bond_query_t const *queries,
                               bond_result_t *results, std::uint32_t query_count, cudaStream_t stream);
void launch_device_angle_lookup(smallfield::parameters::mmff94_tables_view_t view, angle_query_t const *queries,
                                angle_result_t *results, std::uint32_t query_count, cudaStream_t stream);
void launch_device_torsion_lookup(smallfield::parameters::mmff94_tables_view_t view, torsion_query_t const *queries,
                                  torsion_result_t *results, std::uint32_t query_count, cudaStream_t stream);

} // namespace smallfield::tests
