/**
 *  @file src/smallfield/types/atom_limited_bins.hpp
 *  @brief The atom-count ladder every size-dependent dispatch shares.
 *
 *  Molecules are sorted by atom count and grouped into bins whose members all fit under one cap. The ladder
 *  is even to 128 and then jumps to 192, because past 128 atoms molecules are rare — about 4 % of PubChem
 *  and none of GDB13 or Enamine REAL — and each bin costs shared memory quadratic in its cap, so one wide
 *  bin serves them where a finer ladder would buy nothing and cost occupancy on the bins carrying the work.
 *  Two consumers key off the partition — the pipeline dispatch sizing shared memory and occupancy per bin,
 *  the metric prepass batching its eigendecomposition per bin — and one definition keeps them from
 *  partitioning the same wave differently.
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <array> // `std::array`

#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::types {

/** Atom-count caps for `atom-limited-bin` dispatch, ascending. */
inline constexpr auto atom_limited_bin_caps_k = std::to_array<std::uint32_t>({16, 32, 48, 64, 80, 96, 112, 128, 192});

/** Bins the ladder defines, counted from the ladder rather than restated beside it. */
inline constexpr std::size_t atom_limited_bin_count_k = atom_limited_bin_caps_k.size();

static_assert(atom_limited_bin_caps_k.back() == max_atoms_per_molecule_k,
              "the last bin has to be the atom cap, or molecules the pipeline accepts fall in no bin");

/** Index of the bin a molecule of @p num_atoms — hydrogens included — belongs to, or
 *  `atom_limited_bin_count_k` when the molecule exceeds every cap. */
inline constexpr std::size_t atom_limited_bin_for(std::uint32_t num_atoms) noexcept {
    for (std::size_t bin_index = 0; bin_index < atom_limited_bin_count_k; ++bin_index)
        if (num_atoms <= atom_limited_bin_caps_k[bin_index]) return bin_index;
    return atom_limited_bin_count_k;
}

} // namespace smallfield::types
