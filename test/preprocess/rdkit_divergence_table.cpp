/**
 *  @file   test/preprocess/rdkit_divergence_table.cpp
 *  @brief  Every recorded place where SmallField and RDKit disagree and SmallField is right.
 *
 *  Rows are grouped by the chemistry that reaches the defect rather than by the upstream file, because
 *  the chemistry is what a reader has in hand when a parity comparison surprises them. Each family is
 *  introduced by a block comment rather than by a `#pragma region`, which is the one place in this tree
 *  the marker cannot go: a pragma inside a braced initializer is ill-formed, and splitting the array
 *  per family so the markers fit would replace one span with a hand-maintained concatenation.
 *
 *  Two candidates were measured and deliberately @b not recorded, and they are listed here so nobody
 *  spends the afternoon rediscovering them:
 *
 *  - `DistGeom/TriangleSmooth.cpp`, in the lower-bound update, chains two candidates with `else if` and
 *    so appears to discard the larger of them where a maximum belongs. It cannot: the two candidates
 *    are `L_ik - U_jk` and `L_jk - U_ik`, whose sum is `(L_ik - U_ik) + (L_jk - U_jk)`, and every term
 *    of that is non-positive on any matrix with `L <= U`. At most one candidate can exceed a
 *    non-negative `L_ij`, so the branch and a maximum agree on every consistent input. A sweep over
 *    400 corpus molecules, and an exhaustive search over every single-pair tightening of three small
 *    molecules, found zero inputs where they differ. Ours takes the maximum because that is what the
 *    inequality says; it is not a divergence.
 *  - The 1-3 bound around a three-membered ring, where a law-of-cosines write can overwrite an exact
 *    bond length. Upstream corrected it in `c5de9f884357` and `bb18fd56f294`, and our builder carries
 *    the same visited-bound discipline, so there is nothing left to record.
 */

#include <cstddef> // `std::size_t`

#include "test/preprocess/rdkit_divergence_table.hpp"

namespace smallfield::tests {

#pragma region Divergence rows

static constexpr rdkit_divergence_row_t divergence_rows_k[] = {

    /*  Ring double bonds.
     *
     *  A double bond inside a ring gets its 1-4 bounds from `_getInRing14Type`, which asks
     *  `_getAtomStereo` whether the pair under test sits cis or trans. That helper exclusive-ors two
     *  comparisons against the bond's recorded stereo atoms, which are pinned to the bond's begin and
     *  end atoms, while the pair reaching it comes from a ring walk whose direction `RingInfo` fixes
     *  arbitrarily. Walk the ring the other way and the exclusive-or flips, so a pair that is trans is
     *  handed the cis distance. Both isomers of this cyclo-undecene invert: the Z form, whose stereo
     *  atoms are 3 and 6, puts atom 3 cis to the chlorine and therefore trans to atom 7, yet receives
     *  the short cis bound; the E form receives the long trans bound for a pair that is cis. */
    {.smiles = "C1CCC/C=C(Cl)/CCCC1",
     .kind = divergence_kind_t::bounds_1_4_k,
     .atoms = {3, 7, 0, 0},
     .rdkit_values = {2.757833f, 2.877833f},
     .our_values = {3.814331f, 3.934331f},
     .rdkit_tolerance = 0.01f,
     .our_tolerance = 0.01f,
     .divergence_threshold = 0.5f,
     .upstream = "BoundsMatrixBuilder.cpp:_getAtomStereo",
     .note = "Ring traversal direction decides cis against trans, so both isomers "     //
             "receive the other one's 1-4 distance; ours reads the stereo atoms "       //
             "relative to the bond end each substituent hangs off, which no traversal " //
             "can reorder.",
     .since = rdkit_release_t::release_2025_09_k,
     .fixed_in = rdkit_release_t::not_yet_fixed_k},

    /*  Saturated small rings.
     *
     *  The CSD torsion energy is a six-term cosine series and its analytic gradient expands each term
     *  by hand. The sixth expansion reads `forceConstants[4]` and `signs[4]` -- the fifth term's
     *  coefficients -- and its `sin(6 phi)` identity ends in `6 sin(phi)` where `6 sin(phi) cos(phi)`
     *  belongs. Of the 105 small-ring patterns, 37 carry a non-zero sixth coefficient and all 37 of
     *  those carry a zero fifth one, so the entire six-fold force silently vanishes. Cyclopentane is
     *  the smallest molecule that reaches it: one pattern, one coefficient, `V6 = 10` kcal/mol. Our
     *  kernel cannot express either mistake, because one indexed coefficient drives both the energy and
     *  the derivative and a Chebyshev recurrence supplies `sin(n phi)`. */
    {.smiles = "C1CCCC1",
     .kind = divergence_kind_t::torsion_preference_k,
     .atoms = {0, 1, 2, 3},
     .rdkit_values = {-18.660254f, 0.0f},
     .our_values = {-18.660254f, -30.0f},
     .rdkit_tolerance = 0.05f,
     .our_tolerance = 0.05f,
     .divergence_threshold = 1.0f,
     .upstream = "TorsionAngleContribs.cpp:TorsionAngleContribs::getGrad",
     .note = "The six-fold gradient term reads the fifth term's force constant and " //
             "drops a cos(phi) factor, so RDKit reports zero force where its own "   //
             "energy has a slope; the energy slot is the control and agrees to the " //
             "last digit.",
     .since = rdkit_release_t::release_2025_09_k,
     .fixed_in = rdkit_release_t::not_yet_fixed_k},
};

#pragma endregion Divergence rows

divergence_kind_traits_t divergence_kind_traits(divergence_kind_t kind) noexcept {
    switch (kind) {
    case divergence_kind_t::bounds_1_2_k: return {"bounds_1_2", {"lower_bound", "upper_bound"}, 2};
    case divergence_kind_t::bounds_1_3_k: return {"bounds_1_3", {"lower_bound", "upper_bound"}, 2};
    case divergence_kind_t::bounds_1_4_k: return {"bounds_1_4", {"lower_bound", "upper_bound"}, 2};
    case divergence_kind_t::chiral_volume_k: return {"chiral_volume", {"target_volume", "enforced_volume"}, 4};
    case divergence_kind_t::torsion_preference_k:
        return {"torsion_preference", {"energy_delta", "torsion_gradient"}, 4};
    case divergence_kind_t::smoothing_consistency_k:
        return {"smoothing_consistency", {"lower_bound", "upper_bound"}, 2};
    }
    // A kind byte outside the enum names itself, so a diagnostic can never borrow a real kind's labels.
    return {"unknown_kind", {"first_value", "second_value"}, 2};
}

std::uint32_t rdkit_release_code_from_version(std::string_view version_text) noexcept {
    // `2026.03.5`: four digits, a dot, two digits. Anything else is a shape this table cannot place.
    if (version_text.size() < 7 || version_text[4] != '.') return 0;
    std::uint32_t code = 0;
    for (std::size_t position = 0; position < 7; ++position) {
        if (position == 4) continue;
        char const digit = version_text[position];
        if (digit < '0' || digit > '9') return 0;
        code = code * 10 + static_cast<std::uint32_t>(digit - '0');
    }
    return code;
}

custd::span<rdkit_divergence_row_t const> rdkit_divergence_rows() { return divergence_rows_k; }

divergence_expectation_t divergence_expectation(rdkit_divergence_row_t const &row,
                                                std::uint32_t installed_release_code) noexcept {
    if (installed_release_code < static_cast<std::uint32_t>(row.since))
        return divergence_expectation_t::predates_installed_release_k;
    if (row.fixed_in == rdkit_release_t::not_yet_fixed_k) return divergence_expectation_t::still_diverges_k;
    return installed_release_code >= static_cast<std::uint32_t>(row.fixed_in)
               ? divergence_expectation_t::fixed_upstream_k
               : divergence_expectation_t::still_diverges_k;
}

} // namespace smallfield::tests
