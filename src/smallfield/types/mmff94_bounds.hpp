/**
 *  @file src/smallfield/types/mmff94_bounds.hpp
 *  @brief Per-molecule MMFF94 term-count upper bounds, shared between the CPU extractor in
 *  `src/smallfield/mmff94/mmff94_extract.hpp` and the GPU one in `mmff94_extract_device.cuh`.
 *
 *  They give every stack-sized and shared-memory scratch array a compile-time constant bound, and
 *  `compute_device_capacities` clamps each per-molecule capacity to the matching one here; drug-like
 *  molecules at the 192-atom pipeline cap stay well under all six.
 *
 *  A molecule that did exceed one has its extra terms dropped rather than being refused: both extractors
 *  stop emitting past the cap and both say so — the device through `mmff94_per_mol_counts_t::overflow`, the
 *  host through `mmff94_extract_outcome_t` — but nothing downstream reads either, so a saturated molecule
 *  is otherwise indistinguishable.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

namespace smallfield::types {

inline constexpr std::uint32_t max_mmff_bonds_per_mol_k = 256;
inline constexpr std::uint32_t max_mmff_angles_per_mol_k = 512;
inline constexpr std::uint32_t max_mmff_torsions_per_mol_k = 1024;
inline constexpr std::uint32_t max_mmff_oops_per_mol_k = 256;
inline constexpr std::uint32_t max_mmff_stretchbends_per_mol_k = 512;
inline constexpr std::uint32_t max_mmff_interactions_per_mol_k =
    max_mmff_bonds_per_mol_k * 2 + max_mmff_angles_per_mol_k * 3 + max_mmff_torsions_per_mol_k * 4 +
    max_mmff_oops_per_mol_k * 4 + max_mmff_stretchbends_per_mol_k * 3;

/** Batch-global base element of one molecule inside each packed MMFF term array. The device compaction
 *  pass reads one entry per molecule and writes that molecule's terms from the named element; the pipeline
 *  fills the array on the host by prefix-summing the per-molecule capacities, which keeps the packed-array
 *  sizing off the device and out of a readback. */
struct mmff94_packed_offsets_t {
    std::uint32_t bonds;
    std::uint32_t angles;
    std::uint32_t torsions;
    std::uint32_t oops;
    std::uint32_t stretchbends;
    std::uint32_t interactions;
};
static_assert(sizeof(mmff94_packed_offsets_t) == 24);

} // namespace smallfield::types
