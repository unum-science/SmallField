/**
 *  @file rdkit_fallback.hpp
 *  @brief RDKit-based conformer generation for molecules the GPU pipeline
 *         failed on.  OpenMP-parallel EmbedMultipleConfs + MMFFOptimizeConfs.
 *
 *  Called by `smallfield_batch` after the GPU run; only rows where the GPU
 *  reported non-success are passed in.  Each successful RDKit conformer
 *  overwrites the coordinate / energy slot and flips `solver` from 2 → 1.
 *  Remaining failures keep `solver = 2` (will be written as NULL).
 */

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_fp16.h>

namespace smallfield::cli {

struct rdkit_fallback_inputs_t {
    std::span<std::string_view const> smiles;     ///< length N (may contain empty strings)
    std::span<std::uint32_t const> retry_indices; ///< subset of [0, N) to re-attempt
    std::uint32_t num_conformers;
    std::uint32_t max_atoms;
    int num_threads;
    std::uint64_t random_seed;
};

struct rdkit_fallback_outputs_t {
    std::span<std::uint8_t> status;                 ///< length N × K (updated in place)
    std::span<std::uint8_t> solver;                 ///< length N × K (2 → 1 on success)
    std::span<std::uint8_t> num_atoms_per_molecule; ///< length N (written when discovered)
    std::span<__half> coordinates;                  ///< length N × K × max_atoms × 3
    std::span<float> energies;                      ///< length N × K
};

/** Run RDKit ETKDG + MMFF on all `retry_indices`.  Returns how many
 *  molecules produced at least one successful conformer. */
std::uint64_t rdkit_fallback(rdkit_fallback_inputs_t const &inputs, rdkit_fallback_outputs_t &outputs);

} // namespace smallfield::cli
