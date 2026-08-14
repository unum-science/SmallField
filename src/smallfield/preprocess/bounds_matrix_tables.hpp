/**
 *  @file src/smallfield/preprocess/bounds_matrix_tables.hpp
 *  @brief Element-keyed geometry tables the native distance-bounds builder reads.
 *
 *  These are covalent radii, van der Waals radii, and ideal angles — the inputs to
 *  `build_bounds_matrix`, not MMFF94 parameters. The MMFF94 tables are a separate
 *  thing entirely and live in `parameters/mmff94_tables.hpp`.
 *
 *  Covers every element MMFF94 parametrises, plus B, As and Se. Anything outside the table falls back
 *  to carbon-like values, and that fallback is reached: `preprocess/etkdg_prepare_batch.hpp` builds the
 *  bounds matrix before it types the molecule, so an exotic element is measured here and only then
 *  refused. The bounds it produced are discarded with the rest of the molecule, so a carbon-like
 *  covalent radius costs a few multiplications rather than a wrong geometry.
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
    radii[3] = 1.28f;                          // Li
    radii[5] = 0.85f;                          // B
    radii[6] = 0.76f;                          // C
    radii[7] = 0.71f;                          // N
    radii[8] = 0.66f;                          // O
    radii[9] = 0.57f;                          // F
    radii[11] = 1.66f;                         // Na
    radii[12] = 1.41f;                         // Mg
    radii[14] = 1.11f;                         // Si
    radii[15] = 1.07f;                         // P
    radii[16] = 1.05f;                         // S
    radii[17] = 1.02f;                         // Cl
    radii[19] = 2.03f;                         // K
    radii[20] = 1.76f;                         // Ca
    radii[26] = 1.32f;                         // Fe, low spin
    radii[29] = 1.32f;                         // Cu
    radii[30] = 1.22f;                         // Zn
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
    radii[1] = 1.20f;                          // H
    radii[3] = 2.20f;                          // Li
    radii[5] = 1.92f;                          // B
    radii[6] = 1.70f;                          // C
    radii[7] = 1.55f;                          // N
    radii[8] = 1.52f;                          // O
    radii[9] = 1.47f;                          // F
    radii[11] = 2.40f;                         // Na
    radii[12] = 2.20f;                         // Mg
    radii[14] = 2.10f;                         // Si
    radii[15] = 1.80f;                         // P
    radii[16] = 1.80f;                         // S
    radii[17] = 1.75f;                         // Cl
    radii[19] = 2.80f;                         // K
    radii[20] = 2.40f;                         // Ca
    radii[26] = 2.05f;                         // Fe
    radii[29] = 2.00f;                         // Cu
    radii[30] = 2.10f;                         // Zn
    radii[33] = 1.85f;                         // As
    radii[34] = 1.90f;                         // Se
    radii[35] = 1.85f;                         // Br
    radii[53] = 1.98f;                         // I
    return radii;
}();

/** The bond types the covalent-radius shortening is keyed by; an aromatic bond is its own case, not an order. */
enum class covalent_bond_t : std::uint8_t { single_k, double_k, triple_k, aromatic_k };

/**
 *  @brief Ideal bond length in Å for the bond between atoms i and j, whose atomic numbers are Z_i and Z_j.
 *
 *  The sum of the two covalent radii, shortened by the bond type. Close enough to RDKit's `setTopolBounds`
 *  output for drug-like bonds that downstream triangle smoothing + FIRE refinement absorbs the residual.
 */
inline float ideal_bond_length_angstrom(std::uint8_t atomic_number_i, std::uint8_t atomic_number_j,
                                        covalent_bond_t bond) noexcept {
    float const covalent_radius_i = atomic_number_i < covalent_radius_angstrom_k.size()
                                        ? covalent_radius_angstrom_k[atomic_number_i]
                                        : 0.76f;
    float const covalent_radius_j = atomic_number_j < covalent_radius_angstrom_k.size()
                                        ? covalent_radius_angstrom_k[atomic_number_j]
                                        : 0.76f;
    float shortening = 1.00f;
    switch (bond) {
    case covalent_bond_t::single_k: break; // not shortened
    case covalent_bond_t::double_k: shortening = 0.87f; break;
    case covalent_bond_t::triple_k: shortening = 0.78f; break;
    case covalent_bond_t::aromatic_k: shortening = 0.92f; break;
    }
    return (covalent_radius_i + covalent_radius_j) * shortening;
}

/** VdW contact lower bound for atoms i and j, keyed by atomic number. Used for pairs with bond_distance > 3. */
inline float vdw_lower_bound_angstrom(std::uint8_t atomic_number_i, std::uint8_t atomic_number_j) noexcept {
    float const vdw_radius_i = atomic_number_i < vdw_radius_angstrom_k.size() ? vdw_radius_angstrom_k[atomic_number_i]
                                                                              : 1.70f;
    float const vdw_radius_j = atomic_number_j < vdw_radius_angstrom_k.size() ? vdw_radius_angstrom_k[atomic_number_j]
                                                                              : 1.70f;
    return 0.8f * (vdw_radius_i + vdw_radius_j);
}

/** Straight angle, 180°. The sp ideal, and the scale of a regular polygon's interior angle. */
inline constexpr float straight_angle_radians_k = 3.14159265f;

/** Trigonal angle, 120°. The sp2 ideal, and the fallback for a ring centre no other case answers for. */
inline constexpr float trigonal_angle_radians_k = 2.09439510f;

/** Tetrahedral angle, 109.47°. The sp3 ideal, and the fallback for an unrecognised hybridization. */
inline constexpr float tetrahedral_angle_radians_k = 1.91063324f;

/** Right angle, 90°. The sp3d axial and sp3d2 ideal. */
inline constexpr float right_angle_radians_k = 1.57079633f;

/** Full turn, 360°. The angle budget a trigonal centre divides among its neighbour pairs. */
inline constexpr float full_turn_radians_k = 6.28318531f;

/**
 *  Half-widths of the 1-2 and 1-3 distance windows, in angstrom. `DIST12_DELTA` and `DIST13_TOL`
 *  upstream. Absolute rather than a fraction of the distance: what they express is how far a real
 *  bond or angle sits from its rest value, which does not scale with the bond it belongs to.
 */
inline constexpr float bond_tolerance_angstrom_k = 0.01f;
inline constexpr float angle_tolerance_angstrom_k = 0.04f;

/**
 *  116° and 112°, what an sp3 centre gives a substituent reaching out of a three- or four-membered
 *  ring. Upstream states these as literals rather than deriving them, `BoundsMatrixBuilder.cpp:638`:
 *  the ring compresses its own internal angle far below tetrahedral and the exocyclic pair opens to
 *  compensate, but not by the full amount a budget would predict.
 */
inline constexpr float sp3_in_three_ring_angle_radians_k = 2.02458193f;
inline constexpr float sp3_in_four_ring_angle_radians_k = 1.95476876f;

/** 104°, the angle an sp3 centre takes inside a five-membered ring. */
inline constexpr float cyclopentane_angle_radians_k = 1.81514242f;

/** 105°, the angle an sp3d centre takes inside a ring. */
inline constexpr float ring_sp3d_angle_radians_k = 1.83259571f;

/** Ideal 1-3 angle in radians by the center atom's hybridization. */
inline float ideal_angle_radians(std::uint8_t hybridization) noexcept {
    switch (hybridization) {
    case hybridization_sp_k: return straight_angle_radians_k;
    case hybridization_sp2_k: return trigonal_angle_radians_k;
    case hybridization_sp3_k: return tetrahedral_angle_radians_k;
    case hybridization_sp3d_k: return right_angle_radians_k;
    case hybridization_sp3d2_k: return right_angle_radians_k;
    default: return tetrahedral_angle_radians_k;
    }
}

/**
 *  @brief Ideal 1-3 angle in radians for a center whose two neighbours share a ring of @p ring_size with it.
 *
 *  Hybridization alone answers for an open chain and is wrong inside a small ring, where the ring closure
 *  fixes the angle whatever the center prefers: cyclobutane's cross-ring 1-3 comes out 2.45 Å from the
 *  tetrahedral 109.47° against 2.12 Å from the polygon's 90°. Ported from RDKit's `_getRingAngle`,
 *  `Code/GraphMol/DistGeomHelpers/BoundsMatrixBuilder.cpp:487`.
 *
 *  @param[in] hybridization Center atom's hybridization.
 *  @param[in] ring_size Size of the ring all three atoms share; 0 selects the open-chain answer.
 *  @return Angle in radians.
 */
inline float ring_interior_angle_radians(std::uint8_t hybridization, std::uint32_t ring_size) noexcept {
    if (ring_size == 0) return ideal_angle_radians(hybridization);
    // A three- or four-membered ring leaves no freedom, and an sp2 center in a ring up to eight is
    // likewise pinned by the closure rather than by its preferred 120°.
    bool const closure_decides = ring_size == 3 || ring_size == 4 ||
                                 (hybridization == hybridization_sp2_k && ring_size <= 8);
    if (closure_decides) return straight_angle_radians_k * (1.0f - 2.0f / static_cast<float>(ring_size));
    if (hybridization == hybridization_sp3_k)
        return ring_size == 5 ? cyclopentane_angle_radians_k : tetrahedral_angle_radians_k;
    if (hybridization == hybridization_sp3d_k) return ring_sp3d_angle_radians_k;
    if (hybridization == hybridization_sp3d2_k) return right_angle_radians_k;
    return trigonal_angle_radians_k;
}

} // namespace smallfield::preprocess
