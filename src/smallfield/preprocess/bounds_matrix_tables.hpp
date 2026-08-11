/**
 *  @file src/smallfield/preprocess/bounds_matrix_tables.hpp
 *  @brief Element-keyed geometry tables the native distance-bounds builder reads.
 *
 *  These are covalent radii, van der Waals radii, and ideal angles — the inputs to
 *  `build_bounds_matrix`, not MMFF94 parameters. The MMFF94 tables are a separate
 *  thing entirely and live in `parameters/mmff94_tables.hpp`.
 *
 *  Covers the drug-like subset of the periodic table — H, C, N, O, F, P, S, Cl, Br, I and a few
 *  others. Elements outside the table fall back to generic carbon-like values; in practice
 *  `preprocess/etkdg_prepare_batch.hpp` rejects any molecule with an element outside MMFF94's
 *  coverage before it reaches these tables, so the fallback values are not exercised in production.
 *
 *  All values in Å and radians. Rendered as static constexpr arrays indexed by atomic number;
 *  lookup is O(1) with a bounds check.
 */

#pragma once

#include <cstdint> // `std::uint8_t`

#include <array> // `std::array`

#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::preprocess {
using smallfield::types::hybridization_sp2_k;
using smallfield::types::hybridization_sp3_k;
using smallfield::types::hybridization_sp3d2_k;
using smallfield::types::hybridization_sp3d_k;
using smallfield::types::hybridization_sp_k;

/** Covalent radius in Å, indexed by atomic number (0-54 covers drug-like space). */
inline constexpr std::array<float, 55> covalent_radius_angstrom_k = []() {
    std::array<float, 55> radii {};
    for (auto &radius : radii) radius = 0.76f; // carbon fallback
    radii[1] = 0.31f;                          // H
    radii[5] = 0.85f;                          // B
    radii[6] = 0.76f;                          // C
    radii[7] = 0.71f;                          // N
    radii[8] = 0.66f;                          // O
    radii[9] = 0.57f;                          // F
    radii[14] = 1.11f;                         // Si
    radii[15] = 1.07f;                         // P
    radii[16] = 1.05f;                         // S
    radii[17] = 1.02f;                         // Cl
    radii[33] = 1.21f;                         // As
    radii[34] = 1.20f;                         // Se
    radii[35] = 1.20f;                         // Br
    radii[53] = 1.39f;                         // I
    return radii;
}();

/** VdW radius in Å, indexed by atomic number. */
inline constexpr std::array<float, 55> vdw_radius_angstrom_k = []() {
    std::array<float, 55> radii {};
    for (auto &radius : radii) radius = 1.70f; // carbon fallback
    radii[1] = 1.20f;
    radii[5] = 1.92f;
    radii[6] = 1.70f;
    radii[7] = 1.55f;
    radii[8] = 1.52f;
    radii[9] = 1.47f;
    radii[14] = 2.10f;
    radii[15] = 1.80f;
    radii[16] = 1.80f;
    radii[17] = 1.75f;
    radii[33] = 1.85f;
    radii[34] = 1.90f;
    radii[35] = 1.85f;
    radii[53] = 1.98f;
    return radii;
}();

/** Bond-order shortening factor applied to the sum of covalent radii. */
inline constexpr float bond_order_factor(std::uint8_t bond_order, bool aromatic) noexcept {
    if (aromatic) return 0.92f;
    switch (bond_order) {
    case 1: return 1.00f;
    case 2: return 0.87f;
    case 3: return 0.78f;
    default: return 1.00f;
    }
}

/**
 *  @brief Ideal bond length in Å for the bond between atoms i and j, whose atomic numbers are Z_i and Z_j.
 *
 *  Close enough to RDKit's `setTopolBounds` output for drug-like bonds that downstream triangle
 *  smoothing + FIRE refinement absorbs the residual.
 */
inline float ideal_bond_length_angstrom(std::uint8_t atomic_number_i, std::uint8_t atomic_number_j,
                                        std::uint8_t bond_order, bool aromatic) noexcept {
    float const covalent_radius_i = atomic_number_i < covalent_radius_angstrom_k.size()
                                        ? covalent_radius_angstrom_k[atomic_number_i]
                                        : 0.76f;
    float const covalent_radius_j = atomic_number_j < covalent_radius_angstrom_k.size()
                                        ? covalent_radius_angstrom_k[atomic_number_j]
                                        : 0.76f;
    return (covalent_radius_i + covalent_radius_j) * bond_order_factor(bond_order, aromatic);
}

/** VdW contact lower bound for atoms i and j, keyed by atomic number. Used for pairs with bond_distance > 3. */
inline float vdw_lower_bound_angstrom(std::uint8_t atomic_number_i, std::uint8_t atomic_number_j) noexcept {
    float const vdw_radius_i = atomic_number_i < vdw_radius_angstrom_k.size() ? vdw_radius_angstrom_k[atomic_number_i]
                                                                              : 1.70f;
    float const vdw_radius_j = atomic_number_j < vdw_radius_angstrom_k.size() ? vdw_radius_angstrom_k[atomic_number_j]
                                                                              : 1.70f;
    return 0.8f * (vdw_radius_i + vdw_radius_j);
}

/** Ideal 1-3 angle in radians by the center atom's hybridization. */
inline float ideal_angle_radians(std::uint8_t hybridization) noexcept {
    switch (hybridization) {
    case hybridization_sp_k: return 3.14159265f;    // 180°
    case hybridization_sp2_k: return 2.09439510f;   // 120°
    case hybridization_sp3_k: return 1.91063324f;   // 109.47°
    case hybridization_sp3d_k: return 1.57079633f;  // 90° (axial)
    case hybridization_sp3d2_k: return 1.57079633f; // 90°
    default: return 1.91063324f;                    // default tetrahedral
    }
}

} // namespace smallfield::preprocess
