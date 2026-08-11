/**
 *  @file test/cuda/test_csd_torsion_device_kernels.hpp
 *  @brief Device harness for the CSD torsion matcher, for use by its unit tests.
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::tests {

/** Compile the CSD pattern corpus, upload it, and return an opaque owner handle.
 *  @param[out] table_bytes Bytes the uploaded table occupies on the device.
 *  @return Handle to pass to the launcher, or null when the upload failed. */
void *create_csd_torsion_device_table(std::size_t &table_bytes);

/** Release a handle from `create_csd_torsion_device_table`.
 *  @param[in] handle Handle to release; null is ignored. */
void destroy_csd_torsion_device_table(void *handle);

/** Run the device matcher over a batch of descriptors and copy the results back.
 *  @param[in] handle Table handle.
 *  @param[in] descriptors Host-side descriptors, one per molecule.
 *  @param[in] num_molecules Molecules in the batch.
 *  @param[in] torsion_stride Output slots reserved per molecule.
 *  @param[out] torsions_out Host buffer of @p num_molecules times @p torsion_stride entries.
 *  @param[out] counts_out Host buffer of @p num_molecules torsion counts.
 *  @return True when the kernel completed without a CUDA error. */
bool run_csd_torsion_device_batch(void *handle, smallfield::types::molecule_descriptor_t const *descriptors,
                                  std::uint32_t num_molecules, std::uint32_t torsion_stride,
                                  smallfield::types::etk_torsion_descriptor_t *torsions_out, std::uint32_t *counts_out);

} // namespace smallfield::tests
