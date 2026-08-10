/**
 *  @file mmff94_bounds.hpp
 *  @brief Per-molecule MMFF94 term-count upper bounds shared between the
 *         CPU extractor (`source/mmff94/mmff94_cpu.hpp`) and the GPU extractor
 *         (`source/mmff94/mmff94_extract_device.cuh`).
 *
 *  Drug-like molecules with 128 heavy+H atoms never approach these limits;
 *  they exist so every stack-sized scratch array has a compile-time
 *  constant bound. A molecule that exceeds any of them is flagged in
 *  per-molecule status and routed to the CPU RDKit fallback path.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

#include "smallfield/module_fwd.hpp"

namespace smallfield::types {

inline constexpr std::uint32_t max_mmff_bonds_per_mol_k = 256;
inline constexpr std::uint32_t max_mmff_angles_per_mol_k = 512;
inline constexpr std::uint32_t max_mmff_torsions_per_mol_k = 1024;
inline constexpr std::uint32_t max_mmff_oops_per_mol_k = 256;
inline constexpr std::uint32_t max_mmff_stretchbends_per_mol_k = 512;
inline constexpr std::uint32_t max_mmff_interactions_per_mol_k =
    max_mmff_bonds_per_mol_k * 2 + max_mmff_angles_per_mol_k * 3 + max_mmff_torsions_per_mol_k * 4 +
    max_mmff_oops_per_mol_k * 4 + max_mmff_stretchbends_per_mol_k * 3;

/**
 *  @brief Batch-global base element of one molecule inside each packed MMFF term array.
 *
 *  The device compaction pass reads one entry per molecule and writes that molecule's terms
 *  starting at the named element. The pipeline fills the array on the host by prefix-summing
 *  the per-molecule capacities, which is what keeps the packed-array sizing off the device and
 *  out of a readback.
 */
struct mmff94_packed_offsets_t {
    std::uint32_t bonds;
    std::uint32_t angles;
    std::uint32_t torsions;
    std::uint32_t oops;
    std::uint32_t stretchbends;
    std::uint32_t interactions;
};

} // namespace smallfield::types
