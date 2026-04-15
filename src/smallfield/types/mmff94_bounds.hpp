/**
 *  @file mmff94_bounds.hpp
 *  @brief Per-molecule MMFF94 term-count upper bounds shared between the
 *         CPU extractor (`src/mmff94/mmff94_cpu.hpp`) and the GPU extractor
 *         (`src/mmff94/mmff94_extract_device.cuh`).
 *
 *  Drug-like molecules with 128 heavy+H atoms never approach these limits;
 *  they exist so every stack-sized scratch array has a compile-time
 *  constant bound.  A molecule that exceeds any of them is flagged in
 *  per-molecule status and routed to the CPU RDKit fallback path.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <cstdint>

namespace smallfield::types {

inline constexpr std::uint32_t max_mmff_bonds_per_mol_k = 256;
inline constexpr std::uint32_t max_mmff_angles_per_mol_k = 512;
inline constexpr std::uint32_t max_mmff_torsions_per_mol_k = 1024;
inline constexpr std::uint32_t max_mmff_oops_per_mol_k = 256;
inline constexpr std::uint32_t max_mmff_stretchbends_per_mol_k = 512;
inline constexpr std::uint32_t max_mmff_interactions_per_mol_k =
    max_mmff_bonds_per_mol_k * 2 + max_mmff_angles_per_mol_k * 3 + max_mmff_torsions_per_mol_k * 4 +
    max_mmff_oops_per_mol_k * 4 + max_mmff_stretchbends_per_mol_k * 3;

} // namespace smallfield::types
