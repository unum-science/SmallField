/**
 *  @file test/preprocess/rdkit_divergence_table.hpp
 *  @brief The registry of places where disagreeing with RDKit is the correct answer.
 *
 *  Widening a tolerance until a comparison stops failing is right for arithmetic noise and wrong where the
 *  reference is the mistaken side, since it also hides every future regression of that size. A row instead
 *  names one upstream defect, records what each implementation says, and asserts they still say it, so a change
 *  on either side breaks the build. The rows live in a host-only `.cpp`, for the same reason
 *  `csd_torsion_table.cpp` does: they are string literals no kernel reads.
 *
 *  @see test/preprocess/test_rdkit_divergences.cpp, which is the whole of this table's contract.
 */

#pragma once

#include <cstdint> // `std::uint8_t`, `std::uint32_t`

#include <string_view> // `std::string_view`

#include "smallfield/types/views.hpp"

namespace smallfield::tests {

#pragma region Vocabulary

/**
 *  @brief What a row's atoms index and what its two recorded values mean.
 *
 *  The kind is the row's whole configuration: the arity of `atoms`, the meaning and unit of both value slots, and
 *  the measurement producing them. No second knob sits beside it -- any combination one could spell does not.
 */
enum class divergence_kind_t : std::uint8_t {
    /** Two bonded atoms; the values are the lower and upper bound in angstroms. */
    bounds_1_2_k,
    /** Two atoms two bonds apart; the values are the lower and upper bound in angstroms. */
    bounds_1_3_k,
    /** Two atoms three bonds apart; the values are the lower and upper bound in angstroms. */
    bounds_1_4_k,
    /** Four atoms of a tetrahedral center; values are the targeted and enforced signed volumes, cubic angstroms. */
    chiral_volume_k,
    /**
     *  Four atoms of one CSD torsion; values are `E(phi) - E(0)` in kcal/mol and `dE/dphi` in kcal/mol per radian,
     *  both at `torsion_probe_dihedral_radians_k` for the Fourier coefficients the molecule's matched pattern
     *  supplies. The energy slot is the control: a gradient defect moves the second value and leaves the first.
     */
    torsion_preference_k,
    /** Two atoms; bounds in angstroms after triangle smoothing of one identical unsmoothed matrix per side. */
    smoothing_consistency_k,
};

/** Everything a kind fixes about the row carrying it, which is the whole of that row's shape. */
struct divergence_kind_traits_t {
    /** The literal the kind prints as, which is also the subject its diagnostic lines open with. */
    std::string_view name;
    /** What the kind calls its two value slots, in the order a row stores them; a line's `field=`. */
    char const *value_names[2];
    /** How many entries of a row's `atoms` array the kind gives meaning to. */
    std::uint8_t atom_arity;
};

/**
 *  @brief The name, the two slot names and the atom arity belonging to @p kind.
 *
 *  Exhaustive over `divergence_kind_t` and carrying no `default:`, so a kind appended to that enum raises a
 *  `-Wswitch` warning here rather than inheriting three wrong labels in silence. All three travel together
 *  because a kind fixes all three at once, and a row mislabelled in one is mislabelled in all.
 */
divergence_kind_traits_t divergence_kind_traits(divergence_kind_t kind) noexcept;

/**
 *  @brief An RDKit release, as the `YYYYMM` its version string opens with.
 *
 *  Comparable as an integer, so "at or below the installed release" is one `<=`. `not_yet_fixed_k` is a named
 *  state rather than a large number, and every consumer tests for it by name before comparing.
 */
enum class rdkit_release_t : std::uint32_t {
    /** No release has corrected this yet, which is what makes the row an assertion rather than a note. */
    not_yet_fixed_k = 0,
    release_2025_09_k = 202509,
    release_2026_03_k = 202603,
};

/** The `YYYYMM` in an RDKit version string such as `2026.03.5`, or zero when it cannot be read. */
std::uint32_t rdkit_release_code_from_version(std::string_view version_text) noexcept;

/** Where a row stands against the RDKit the suite is linked to. */
enum class divergence_expectation_t : std::uint8_t {
    /** The installed release predates the one the row was first reproduced on, so it is not asserted. */
    predates_installed_release_k,
    /** Upstream has not corrected it, so the two implementations must still disagree. */
    still_diverges_k,
    /** Upstream has corrected it, so the assertion flips and the row becomes a regression test. */
    fixed_upstream_k,
};

#pragma endregion Vocabulary

#pragma region Rows

/**
 *  @brief One place where SmallField and RDKit disagree and SmallField is right.
 *
 *  Everything a row carries is either measurable or a citation, and the test re-measures all of it every run.
 *  `rdkit_values` and `our_values` are the two sides as last recorded, `rdkit_tolerance` and `our_tolerance` how
 *  far each may drift before the row is stale, and `divergence_threshold` how far apart they must stay.
 *  A row whose threshold sits inside the sum of its tolerances is refused at load: it is a coin toss.
 */
struct rdkit_divergence_row_t {
    /** Molecule the row is measured on. Points into static storage, so it outlives every caller. */
    char const *smiles;
    /** Fixes the arity of `atoms` and the meaning of both value slots. */
    divergence_kind_t kind;
    /** Atom indices into @p smiles after `addHs`; only the kind's own `atom_arity` of them count. */
    std::uint8_t atoms[4];
    /** What RDKit produced when the row was last measured. */
    float rdkit_values[2];
    /** What SmallField produced when the row was last measured. */
    float our_values[2];
    /** How far RDKit may drift from `rdkit_values` and still be the side this row describes. */
    float rdkit_tolerance;
    /** How far SmallField may drift from `our_values` and still be the side this row describes. */
    float our_tolerance;
    /** How far apart the two must remain for the disagreement to still be the one recorded. */
    float divergence_threshold;
    /** RDKit source file and symbol the defect lives in, as `<file>:<symbol>`; never a line number. */
    char const *upstream;
    /** What upstream gets wrong, in one sentence, and what we do instead. */
    char const *note;
    /** Release the divergence was first reproduced on. */
    rdkit_release_t since;
    /** Release that corrected it, or `not_yet_fixed_k`. */
    rdkit_release_t fixed_in;
};

/** Dihedral the torsion rows are probed at, in radians: 25 degrees, where no harmonic's sine vanishes. */
inline constexpr float torsion_probe_dihedral_radians_k = 0.4363323129985824f;

/** Every recorded divergence, grouped by the chemistry that reaches it. */
custd::span<rdkit_divergence_row_t const> rdkit_divergence_rows();

/**
 *  @brief Whether @p row is still asserted as a disagreement against the installed RDKit.
 *  @param[in] installed_release_code `YYYYMM` of the linked RDKit, from `rdkit_release_code_from_version`.
 *  @return Which of the three assertions the test should make about this row.
 */
divergence_expectation_t divergence_expectation(rdkit_divergence_row_t const &row,
                                                std::uint32_t installed_release_code) noexcept;

#pragma endregion Rows

} // namespace smallfield::tests
