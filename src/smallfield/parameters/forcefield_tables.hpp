/**
 *  @file forcefield_tables.hpp
 *  @brief Minimal force-field lookup tables for the native bounds-matrix builder.
 *
 *  Covers the drug-like subset of the periodic table (H, C, N, O, F, P, S,
 *  Cl, Br, I and a few others).  For elements outside the table we fall back
 *  to generic carbon-like values — never fails, at worst looses a few percent
 *  of quality on unusual elements that are already outside MMFF94's native
 *  coverage.
 *
 *  All values in Å and radians.  Rendered as static constexpr arrays indexed
 *  by atomic number; lookup is O(1) with a bounds check.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <array>
#include <cmath>
#include <cstdint>

#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::parameters {
using namespace smallfield::types;

/** Covalent radius in Å, indexed by atomic number (0-54 covers drug-like space). */
inline constexpr std::array<float, 55> covalent_radius_angstrom_k = []() {
    std::array<float, 55> r {};
    for (auto &x : r) x = 0.76f; // carbon fallback
    r[1] = 0.31f;                // H
    r[5] = 0.85f;                // B
    r[6] = 0.76f;                // C
    r[7] = 0.71f;                // N
    r[8] = 0.66f;                // O
    r[9] = 0.57f;                // F
    r[14] = 1.11f;               // Si
    r[15] = 1.07f;               // P
    r[16] = 1.05f;               // S
    r[17] = 1.02f;               // Cl
    r[33] = 1.21f;               // As
    r[34] = 1.20f;               // Se
    r[35] = 1.20f;               // Br
    r[53] = 1.39f;               // I
    return r;
}();

/** VdW radius in Å, indexed by atomic number. */
inline constexpr std::array<float, 55> vdw_radius_angstrom_k = []() {
    std::array<float, 55> r {};
    for (auto &x : r) x = 1.70f; // carbon fallback
    r[1] = 1.20f;
    r[5] = 1.92f;
    r[6] = 1.70f;
    r[7] = 1.55f;
    r[8] = 1.52f;
    r[9] = 1.47f;
    r[14] = 2.10f;
    r[15] = 1.80f;
    r[16] = 1.80f;
    r[17] = 1.75f;
    r[33] = 1.85f;
    r[34] = 1.90f;
    r[35] = 1.85f;
    r[53] = 1.98f;
    return r;
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
 *  @brief Ideal bond length in Å for atoms (Z_i, Z_j) with given bond order.
 *
 *  Close enough to RDKit's `setTopolBounds` output for drug-like bonds that
 *  downstream triangle smoothing + FIRE refinement absorbs the residual.
 */
inline float ideal_bond_length_angstrom(std::uint8_t z_i, std::uint8_t z_j, std::uint8_t bond_order,
                                        bool aromatic) noexcept {
    float const r_i = z_i < covalent_radius_angstrom_k.size() ? covalent_radius_angstrom_k[z_i] : 0.76f;
    float const r_j = z_j < covalent_radius_angstrom_k.size() ? covalent_radius_angstrom_k[z_j] : 0.76f;
    return (r_i + r_j) * bond_order_factor(bond_order, aromatic);
}

/** VdW contact lower bound for atoms (Z_i, Z_j). Used for pairs with bond_distance > 3. */
inline float vdw_lower_bound_angstrom(std::uint8_t z_i, std::uint8_t z_j) noexcept {
    float const r_i = z_i < vdw_radius_angstrom_k.size() ? vdw_radius_angstrom_k[z_i] : 1.70f;
    float const r_j = z_j < vdw_radius_angstrom_k.size() ? vdw_radius_angstrom_k[z_j] : 1.70f;
    return 0.8f * (r_i + r_j);
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

} // namespace smallfield::parameters
