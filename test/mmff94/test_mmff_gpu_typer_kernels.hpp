/**
 *  @file test/mmff94/test_mmff_gpu_typer_kernels.hpp
 *  @brief Device harness for the MMFF94 GPU atom typer, for use by its unit tests.
 */

#pragma once

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::tests {

void *create_mmff94_device_tables_view(smallfield::parameters::mmff94_tables_host_t const &host,
                                       smallfield::parameters::mmff94_tables_view_t &out_view);

void destroy_mmff94_device_tables_view(void *handle);

/**
 *  @brief Upload a batch of descriptors, run `mmff94_type_and_charge_kernel`, and copy the results back.
 *
 *  The batch's atom total is the extent of @p atom_types_out, so the device arrays are sized by what the
 *  caller can receive rather than by a count restated beside it.
 *
 *  @param[in] tables_view Device-resident MMFF94 parameter tables.
 *  @param[in] descriptors One descriptor per molecule.
 *  @param[in] atom_offsets Each molecule's first atom in the batch-global atom array.
 *  @param[out] atom_types_out Receives one MMFF atom type per atom of the batch.
 *  @param[out] atom_charges_out Receives one partial charge per atom, same extent as @p atom_types_out.
 *  @param[out] typing_status_out Receives one status per molecule, zero where the typer accepted it.
 */
void launch_mmff94_gpu_typer_batch(smallfield::parameters::mmff94_tables_view_t tables_view,
                                   custd::span<smallfield::types::molecule_descriptor_t const> descriptors,
                                   custd::span<std::uint32_t const> atom_offsets,
                                   custd::span<std::uint8_t> atom_types_out, custd::span<float> atom_charges_out,
                                   custd::span<std::uint8_t> typing_status_out);

} // namespace smallfield::tests
