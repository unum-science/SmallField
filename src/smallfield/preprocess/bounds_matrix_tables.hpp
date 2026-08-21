/**
 *  @file src/smallfield/preprocess/bounds_matrix_tables.hpp
 *  @brief Element-keyed geometry tables the native distance-bounds builder reads.
 *
 *  These are UFF bond parameters, van der Waals radii, and ideal angles — the inputs to
 *  `build_bounds_matrix`, not MMFF94 parameters. The MMFF94 tables are a separate
 *  thing entirely and live in `parameters/mmff94_tables.hpp`.
 *
 *  Every lookup here is safe for an arbitrary atomic number, and that matters rather than being defensive
 *  habit: `preprocess/etkdg_prepare_batch.hpp` builds the bounds matrix @b before it types the molecule,
 *  so an exotic element is measured here and only then refused. Both tables stand carbon in for what they
 *  do not carry, and the bounds are discarded with the rest of the molecule a stage later.
 *
 *  All values in Å and radians, in constexpr arrays indexed by atomic number or by `uff_type_t`; every
 *  lookup is O(1) behind a bounds check or a total switch.
 */

#pragma once

#include <cmath>   // `std::log`, `std::sqrt`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`

#include <array> // `std::array`

#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::preprocess {
using smallfield::types::hybridization_sp2_k;
using smallfield::types::hybridization_sp3_k;
using smallfield::types::hybridization_sp3d2_k;
using smallfield::types::hybridization_sp3d_k;
using smallfield::types::hybridization_sp_k;

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

/** VdW radius of @p atomic_number, falling back to carbon's past the table. */
inline float vdw_radius_angstrom(std::uint8_t atomic_number) noexcept {
    return atomic_number < vdw_radius_angstrom_k.size() ? vdw_radius_angstrom_k[atomic_number] : 1.70f;
}

/** VdW contact lower bound for atoms i and j, keyed by atomic number. Used for pairs with bond_distance > 3. */
inline float vdw_lower_bound_angstrom(std::uint8_t atomic_number_i, std::uint8_t atomic_number_j) noexcept {
    return 0.8f * (vdw_radius_angstrom(atomic_number_i) + vdw_radius_angstrom(atomic_number_j));
}

#pragma region UFF Rest Lengths

/**
 *  @brief The UFF parameter row an atom reads, one enumerator per row the element gate can reach.
 *
 *  RDKit spells these as strings — `C_3`, `O_R`, `S_3+2` — and looks them up in a 127-row table of eleven
 *  columns, of which `calcBondRestLength` reads exactly two. The gate in `mmff94/mmff94_typer.hpp` admits
 *  nineteen elements, which reach thirty-four of those rows, so the table below is thirty-four pairs
 *  rather than a transcription of the whole thing.
 *
 *  `untyped_k` is not a failure to be fixed, and it is narrower than "we have no row". It means @b upstream
 *  has no row either — a hypervalent sulfur, a calcium or iron outside the one coordination its row is
 *  written for — and it selects upstream's give-up branch rather than a substitute row. An element this
 *  table simply does not carry is a different case and takes carbon's row; see `uff_type_of`.
 */
enum class uff_type_t : std::uint8_t {
    /** No row: upstream falls back to half the van der Waals sum for any bond touching such an atom. */
    untyped_k,
    hydrogen_k,
    lithium_k,
    carbon_sp_k,
    carbon_sp2_k,
    carbon_resonant_k,
    carbon_sp3_k,
    nitrogen_sp_k,
    nitrogen_sp2_k,
    nitrogen_resonant_k,
    nitrogen_sp3_k,
    oxygen_sp_k,
    oxygen_sp2_k,
    oxygen_resonant_k,
    oxygen_sp3_k,
    fluorine_k,
    sodium_k,
    magnesium_k,
    silicon_k,
    phosphorus_trivalent_k,
    phosphorus_pentavalent_k,
    sulfur_sp2_k,
    sulfur_resonant_k,
    sulfur_divalent_k,
    sulfur_tetravalent_k,
    sulfur_hexavalent_k,
    chlorine_k,
    potassium_k,
    calcium_k,
    iron_tetrahedral_k,
    iron_octahedral_k,
    copper_k,
    zinc_k,
    bromine_k,
    iodine_k,
};

/** The two columns of a UFF row the rest length is built from. */
struct uff_atom_params_t {
    /** `r1` upstream, the valence bond radius in Å. */
    float valence_radius = 0.0f;
    /** `GMP_Xi` upstream, the generalized Mulliken-Pauling electronegativity. */
    float electronegativity = 0.0f;
};

/**
 *  @brief The thirty-four reachable UFF rows, indexed by `uff_type_t`.
 *
 *  Assigned by enumerator rather than written as a positional list, so reordering the enumeration cannot
 *  silently give an element another element's radius.
 */
inline constexpr std::array<uff_atom_params_t, 35> uff_atom_params_k = []() {
    std::array<uff_atom_params_t, 35> rows {};
    auto assign = [&rows](uff_type_t type, float radius, float electronegativity) {
        rows[static_cast<std::size_t>(type)] = {radius, electronegativity};
    };
    assign(uff_type_t::hydrogen_k, 0.354f, 4.528f);               // H_
    assign(uff_type_t::lithium_k, 1.336f, 3.006f);                // Li
    assign(uff_type_t::carbon_sp_k, 0.706f, 5.343f);              // C_1
    assign(uff_type_t::carbon_sp2_k, 0.732f, 5.343f);             // C_2
    assign(uff_type_t::carbon_resonant_k, 0.729f, 5.343f);        // C_R
    assign(uff_type_t::carbon_sp3_k, 0.757f, 5.343f);             // C_3
    assign(uff_type_t::nitrogen_sp_k, 0.656f, 6.899f);            // N_1
    assign(uff_type_t::nitrogen_sp2_k, 0.685f, 6.899f);           // N_2
    assign(uff_type_t::nitrogen_resonant_k, 0.699f, 6.899f);      // N_R
    assign(uff_type_t::nitrogen_sp3_k, 0.700f, 6.899f);           // N_3
    assign(uff_type_t::oxygen_sp_k, 0.639f, 8.741f);              // O_1
    assign(uff_type_t::oxygen_sp2_k, 0.634f, 8.741f);             // O_2
    assign(uff_type_t::oxygen_resonant_k, 0.680f, 8.741f);        // O_R
    assign(uff_type_t::oxygen_sp3_k, 0.658f, 8.741f);             // O_3
    assign(uff_type_t::fluorine_k, 0.668f, 10.874f);              // F_
    assign(uff_type_t::sodium_k, 1.539f, 2.843f);                 // Na
    assign(uff_type_t::magnesium_k, 1.421f, 3.951f);              // Mg3+2
    assign(uff_type_t::silicon_k, 1.117f, 4.168f);                // Si3
    assign(uff_type_t::phosphorus_trivalent_k, 1.101f, 5.463f);   // P_3+3
    assign(uff_type_t::phosphorus_pentavalent_k, 1.056f, 5.463f); // P_3+5
    assign(uff_type_t::sulfur_sp2_k, 0.854f, 6.928f);             // S_2
    assign(uff_type_t::sulfur_resonant_k, 1.077f, 6.928f);        // S_R
    assign(uff_type_t::sulfur_divalent_k, 1.064f, 6.928f);        // S_3+2
    assign(uff_type_t::sulfur_tetravalent_k, 1.049f, 6.928f);     // S_3+4
    assign(uff_type_t::sulfur_hexavalent_k, 1.027f, 6.928f);      // S_3+6
    assign(uff_type_t::chlorine_k, 1.044f, 8.564f);               // Cl
    assign(uff_type_t::potassium_k, 1.953f, 2.421f);              // K_
    assign(uff_type_t::calcium_k, 1.761f, 3.231f);                // Ca6+2
    assign(uff_type_t::iron_tetrahedral_k, 1.270f, 3.760f);       // Fe3+2
    assign(uff_type_t::iron_octahedral_k, 1.335f, 3.760f);        // Fe6+2
    assign(uff_type_t::copper_k, 1.302f, 4.200f);                 // Cu3+1
    assign(uff_type_t::zinc_k, 1.193f, 5.106f);                   // Zn3+2
    assign(uff_type_t::bromine_k, 1.192f, 7.790f);                // Br
    assign(uff_type_t::iodine_k, 1.382f, 6.822f);                 // I_
    return rows;
}();

/** Whether an sp2 centre sits on its element's resonant row, which is aromaticity or a conjugated bond. */
enum class resonance_t : std::uint8_t { isolated_k, resonant_k };

/** `Params::lambda` upstream, the scaling on the Pauling bond-order correction. */
inline constexpr float uff_bond_order_lambda_k = 0.1332f;

/**
 *  @brief UFF rest length in Å for a bond of order @p bond_order between two typed atoms.
 *
 *  `calcBondRestLength` upstream: the two valence radii, plus Pauling's logarithmic bond-order term, minus
 *  O'Keefe and Breese's electronegativity term. Aromatic bonds arrive at order 1.5 and need no case of
 *  their own — the bond-order term alone puts aromatic C-C at upstream's 1.3793 Å.
 *
 *  @param[in] first UFF row of one end; must not be `untyped_k`.
 *  @param[in] second UFF row of the other end; must not be `untyped_k`.
 *  @param[in] bond_order Bond order as a real number, 1.5 for aromatic.
 *  @return Rest length in Å.
 */
inline float uff_bond_rest_length_angstrom(uff_type_t first, uff_type_t second, float bond_order) noexcept {
    uff_atom_params_t const params_first = uff_atom_params_k[static_cast<std::size_t>(first)];
    uff_atom_params_t const params_second = uff_atom_params_k[static_cast<std::size_t>(second)];
    float const radius_first = params_first.valence_radius, radius_second = params_second.valence_radius;
    float const radius_sum = radius_first + radius_second;

    float const bond_order_term = -uff_bond_order_lambda_k * radius_sum * std::log(bond_order);
    // The electronegativities enter as square roots whose squares are the stored values again, so the
    // denominator is written with the stored values directly rather than squaring two roots back.
    float const root_first = std::sqrt(params_first.electronegativity);
    float const root_second = std::sqrt(params_second.electronegativity);
    float const root_difference = root_first - root_second;
    float const electronegativity_term = radius_first * radius_second * root_difference * root_difference /
                                         (params_first.electronegativity * radius_first +
                                          params_second.electronegativity * radius_second);
    return radius_sum + bond_order_term - electronegativity_term;
}

/**
 *  @brief Length upstream gives a bond one of whose ends UFF cannot type: half the van der Waals sum.
 *
 *  Neither the valence radii nor the bond order reach this branch — a manganese-oxygen double bond and a
 *  nickel-carbon single bond are both centred on half their van der Waals sum. Recovered by fitting
 *  `setTopolBounds` output rather than from the published formula, exact on all five probes reachable
 *  with a bonded untypable atom.
 */
inline float untypable_bond_rest_length_angstrom(std::uint8_t atomic_number_i, std::uint8_t atomic_number_j) noexcept {
    return 0.5f * (vdw_radius_angstrom(atomic_number_i) + vdw_radius_angstrom(atomic_number_j));
}

/** Multiplicative half-width that same bond takes, where a typed bond takes `bond_tolerance_angstrom_k`. */
inline constexpr float untypable_bond_window_k = 0.5f;

/** Carbon's row for @p hybridization, which is also the stand-in for an element this table omits. */
inline uff_type_t carbon_uff_type_(std::uint8_t hybridization, resonance_t resonance) noexcept {
    if (hybridization == hybridization_sp_k) return uff_type_t::carbon_sp_k;
    if (hybridization == hybridization_sp2_k)
        return resonance == resonance_t::resonant_k ? uff_type_t::carbon_resonant_k : uff_type_t::carbon_sp2_k;
    return uff_type_t::carbon_sp3_k;
}

/**
 *  @brief Which UFF row an atom reads, from its element, hybridization, resonance and degree.
 *
 *  Derived by walking RDKit's own `getAtomLabel` over 5,000 molecules and recording every distinct answer,
 *  rather than transliterated from its switch. Two things that reading alone would have missed: an sp2 C,
 *  N, O or S goes to the resonant row on a conjugated bond and not only on aromaticity, and a metal outside
 *  the one coordination its row is written for gets a label that is @b absent from upstream's own parameter
 *  table — `Ca+2`, `Fe2+2`, `Cu5+1` — so it answers `untyped_k` here for exactly upstream's reason.
 *
 *  An element this table does not carry at all is the opposite case and must not answer `untyped_k`:
 *  upstream types boron, germanium and arsenic perfectly well and pins their bonds, so answering the
 *  give-up branch there would assert a half-angstrom window where upstream asserts a hundredth. Such an
 *  atom takes carbon's row for its hybridization. It is a wrong length rather than a wrong kind of claim,
 *  and it is never shipped — the element gate in `mmff94/mmff94_typer.hpp` refuses the molecule a stage
 *  later, which is why the bounds matrix is allowed to be built before typing at all.
 *
 *  @param[in] atomic_number Element.
 *  @param[in] hybridization `hybridization_t` value.
 *  @param[in] resonance Whether the atom is aromatic or carries a conjugated bond.
 *  @param[in] degree Explicit bond count, which is what splits the sulfur and phosphorus valence rows.
 *  @param[in] formal_charge Signed formal charge, which promotes a three-coordinate phosphorus.
 *  @return The row, `untyped_k` where upstream also has none, or carbon's row for an omitted element.
 */
inline uff_type_t uff_type_of(std::uint8_t atomic_number, std::uint8_t hybridization, resonance_t resonance,
                              std::uint8_t degree, std::int8_t formal_charge) noexcept {
    bool const is_resonant = resonance == resonance_t::resonant_k;
    switch (atomic_number) {
    case 1: return uff_type_t::hydrogen_k;
    case 3: return uff_type_t::lithium_k;
    case 6: return carbon_uff_type_(hybridization, resonance);
    case 7:
        if (hybridization == hybridization_sp_k) return uff_type_t::nitrogen_sp_k;
        if (hybridization == hybridization_sp2_k)
            return is_resonant ? uff_type_t::nitrogen_resonant_k : uff_type_t::nitrogen_sp2_k;
        return uff_type_t::nitrogen_sp3_k;
    case 8:
        if (hybridization == hybridization_sp_k) return uff_type_t::oxygen_sp_k;
        if (hybridization == hybridization_sp2_k)
            return is_resonant ? uff_type_t::oxygen_resonant_k : uff_type_t::oxygen_sp2_k;
        return uff_type_t::oxygen_sp3_k;
    case 9: return uff_type_t::fluorine_k;
    case 11: return uff_type_t::sodium_k;
    case 12: return hybridization == hybridization_sp3_k ? uff_type_t::magnesium_k : uff_type_t::untyped_k;
    case 14: return uff_type_t::silicon_k;
    case 15:
        return degree >= 4 || formal_charge > 0 ? uff_type_t::phosphorus_pentavalent_k
                                                : uff_type_t::phosphorus_trivalent_k;
    case 16:
        if (hybridization == hybridization_sp2_k)
            return is_resonant ? uff_type_t::sulfur_resonant_k : uff_type_t::sulfur_sp2_k;
        if (hybridization != hybridization_sp3_k) return uff_type_t::untyped_k;
        if (degree <= 2) return uff_type_t::sulfur_divalent_k;
        return degree == 3 ? uff_type_t::sulfur_tetravalent_k : uff_type_t::sulfur_hexavalent_k;
    case 17: return uff_type_t::chlorine_k;
    case 19: return uff_type_t::potassium_k;
    case 20: return hybridization == hybridization_sp3d2_k ? uff_type_t::calcium_k : uff_type_t::untyped_k;
    case 26:
        if (hybridization == hybridization_sp3_k) return uff_type_t::iron_tetrahedral_k;
        return hybridization == hybridization_sp3d2_k ? uff_type_t::iron_octahedral_k : uff_type_t::untyped_k;
    case 29: return hybridization == hybridization_sp3_k ? uff_type_t::copper_k : uff_type_t::untyped_k;
    case 30: return hybridization == hybridization_sp3_k ? uff_type_t::zinc_k : uff_type_t::untyped_k;
    case 35: return uff_type_t::bromine_k;
    case 53: return uff_type_t::iodine_k;
    }
    return carbon_uff_type_(hybridization, resonance);
}

#pragma endregion UFF Rest Lengths

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
