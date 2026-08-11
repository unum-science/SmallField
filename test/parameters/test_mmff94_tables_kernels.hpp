/**
 *  @file test/parameters/test_mmff94_tables_kernels.hpp
 *  @brief Device launchers for the MMFF94 parameter table lookups, for use by their unit tests.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_device.cuh"

#include "mmff94_lookup_probes.hpp"

namespace smallfield::tests {

void launch_device_vdw_lookup(smallfield::parameters::mmff94_tables_view_t view, vdw_query_t const *queries,
                              vdw_result_t *results, std::uint32_t query_count);
void launch_device_bond_lookup(smallfield::parameters::mmff94_tables_view_t view, bond_query_t const *queries,
                               bond_result_t *results, std::uint32_t query_count);
void launch_device_angle_lookup(smallfield::parameters::mmff94_tables_view_t view, angle_query_t const *queries,
                                angle_result_t *results, std::uint32_t query_count);
void launch_device_torsion_lookup(smallfield::parameters::mmff94_tables_view_t view, torsion_query_t const *queries,
                                  torsion_result_t *results, std::uint32_t query_count);
void launch_device_oop_lookup(smallfield::parameters::mmff94_tables_view_t view, oop_query_t const *queries,
                              oop_result_t *results, std::uint32_t query_count);
void launch_device_stbn_lookup(smallfield::parameters::mmff94_tables_view_t view, stbn_query_t const *queries,
                               stbn_result_t *results, std::uint32_t query_count);

void *device_tables_create(smallfield::parameters::mmff94_tables_host_t const &host_tables,
                           smallfield::parameters::mmff94_tables_view_t &out_view);
void device_tables_destroy(void *handle);

} // namespace smallfield::tests
