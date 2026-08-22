/**
 *  @file test/cuda/csd_torsion_device_kernels.hpp
 *  @brief Device harness for the CSD torsion matcher, for use by its unit tests.
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::tests {

/** Compile the CSD pattern corpus, upload it, and return an opaque owner handle.
 *  @param[out] table_bytes Bytes the uploaded table occupies on the device.
 *  @return Handle to pass to the launcher, or null when the upload failed. */
void *create_csd_torsion_device_table(std::size_t &table_bytes);

/** Release a handle from `create_csd_torsion_device_table`.
 *  @param[in] handle Handle to release; null is ignored. */
void destroy_csd_torsion_device_table(void *handle);

/** Run the device matcher over a batch of descriptors and copy the results back.
 *  @param[in] descriptors Host-side descriptors, one per molecule.
 *  @param[in] torsion_stride Output slots reserved per molecule.
 *  @param[out] torsions_out Host buffer of at least one stride per descriptor.
 *  @param[out] counts_out Host buffer of at least one torsion count per descriptor.
 *  @return True when the batch was staged and the kernel completed without a CUDA error. */
bool run_csd_torsion_device_batch(void *handle, custd::span<smallfield::types::molecule_descriptor_t const> descriptors,
                                  std::uint32_t torsion_stride,
                                  custd::span<smallfield::types::etk_torsion_descriptor_t> torsions_out,
                                  custd::span<std::uint32_t> counts_out);

} // namespace smallfield::tests
