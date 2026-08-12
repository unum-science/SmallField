/**
 *  @file cli/rdkit_fallback.hpp
 *  @brief Declares `rdkit_fallback`, RDKit-based conformer generation for molecules the GPU pipeline
 *  failed on.
 *
 *  OpenMP-parallel EmbedMultipleConfs + MMFFOptimizeConfs. Called by `smallfield_batch` after the GPU run;
 *  only rows where the GPU reported non-success are passed in. Each successful RDKit conformer overwrites
 *  the coordinate / energy slot and flips its status to `success_k`; remaining failures keep the status the
 *  GPU left, which the writer emits as a null row.
 */

#pragma once

#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <string_view> // `std::string_view`

#include <cuda_fp16.h>

#include "smallfield/api/conformer_layout.hpp"
#include "smallfield/api/conformer_status.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::cli {

struct rdkit_fallback_inputs_t {
    /** Length N (may contain empty strings). */
    custd::span<std::string_view const> smiles;
    /** Subset of [0, N) to re-attempt. */
    custd::span<std::uint32_t const> retry_indices;
    /** Slot stride and conformer count, shared with the GPU producer and the Parquet writer. */
    api::conformer_slot_layout_t layout;
    int num_threads;
    std::uint64_t random_seed;
};

struct rdkit_fallback_outputs_t {
    /** Length N × K (updated in place). */
    custd::span<api::conformer_status_t> status;
    /** Length N (written when discovered). */
    custd::span<std::uint8_t> num_atoms_per_molecule;
    /** Length N × K × max_atoms × 3. */
    custd::span<__half> coordinates;
    /** Length N × K. */
    custd::span<float> energies;
};

/** Run RDKit ETKDG + MMFF on all `retry_indices`. Returns how many
 *  molecules produced at least one successful conformer. */
std::uint64_t rdkit_fallback(rdkit_fallback_inputs_t const &inputs, rdkit_fallback_outputs_t &outputs);

} // namespace smallfield::cli
