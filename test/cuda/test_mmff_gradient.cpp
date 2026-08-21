/**
 *  @file test/cuda/test_mmff_gradient.cpp
 *  @brief Gate the device MMFF94 energy and its analytical gradient against host references.
 *
 *  Neither reference alone is enough: RDKit's `calcEnergy` gates the total at identical coordinates, a
 *  central difference of that total gates the gradient beside it. An energy matching RDKit under a
 *  gradient that does not differentiate it is what a minimizer turns into a wrong conformer, and no
 *  total-energy comparison sees it. The out-of-plane bend is probed alone first, being the one MMFF94
 *  form whose derivative shares no algebra with the stage, so a sign error in it hides inside a total.
 *
 *  `ctest` gates on the built-ins, every one inside `energies_agree_`'s band, so the sweep exempts none.
 *  A named corpus turns it into a survey: over a 40-molecule PubChem sample the gradient sections stay
 *  clean while the energy section reports 18 molecules beyond its band, four past 20 kcal per mol --
 *  the state of the two implementations rather than a property of this harness.
 */

#include <cmath>   // `std::fabs`, `std::isfinite`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`
#include <cstdio>  // `std::fflush`, `stdout`

#include <algorithm>   // `std::max`
#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/types/views.hpp"

#include "test/cuda/test_mmff_gradient_kernels.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

using sftest::gpu_mmff_energy;

static constexpr std::string_view test_name_k = "test_mmff_gradient";

namespace smallfield::tests {

#pragma region Coordinate Layout

/** A gradient carries one component per coordinate, so it is read through the coordinate layout itself. */
using gradient_soa_t =
    sft::coordinates_soa<sft::coordinate_role_t::gradient_k, float const, sft::conformer_dimensions_k>;

/** The field name one gradient component prints under, so a mismatch line names it rather than numbering it. */
static constexpr std::string_view gradient_field_name_(std::uint32_t axis) noexcept {
    switch (axis) {
    case 0: return "gradient_x";
    case 1: return "gradient_y";
    default: return "gradient_z";
    }
}

#pragma endregion Coordinate Layout

#pragma region Device Evaluation

/**
 *  @brief How far one coordinate moves either way for a central difference, in angstroms.
 *
 *  Cancellation bounds the step from below, not truncation: `float` rounds two energies of magnitude E to
 *  `E * 2^-23` before subtracting and dividing by `2 * step`, so at glucose's 138 kcal per mol a 1e-4 step
 *  left the quotient a 0.17 resolution against analytical components of 1.3 and read as a mismatch. Ten
 *  times the step divides that by ten; truncation grows as the square of it, four orders below the band.
 */
static constexpr float finite_difference_step_k = 1.0e-3f;

/** Components of the full gradient each molecule probes; the stride spreads them over all three axes. */
static constexpr std::size_t sampled_components_per_molecule_k = 12;

/** Mismatching fields each section prints before the rest are only counted. */
static constexpr std::size_t max_printed_lines_k = 32;

/**
 *  @brief Buffers every molecule and probe reuses, so the sweep allocates once rather than per call.
 *
 *  Each component costs two device evaluations, some fifty per sixty-atom molecule, so anything sized per
 *  call would be an allocation per probe. `prepared` is refilled per SMILES, `single_oop` is the same
 *  molecule stripped to its first out-of-plane bend, `atom_major` holds RDKit's conformer and `axis_major`
 *  the transposed copy a probe perturbs and restores; `probe_gradient` is written and never read, and
 *  `bond_hops` backs the hop matrix every energy call rebuilds.
 */
struct gradient_scratch_t {
    sfp::etkdg_prepared_molecule_t prepared {};
    sfp::etkdg_prepared_molecule_t single_oop {};
    std::vector<float> atom_major;
    std::vector<float> axis_major;
    std::vector<float> stage_gradient;
    std::vector<float> term_gradient;
    std::vector<float> probe_gradient;
    std::vector<std::uint8_t> bond_hops;
};

/**
 *  @brief Evaluate one molecule's MMFF94 energy and gradient on the device. The pool ships no hop matrix,
 *  so this harness rebuilds it from the bond list as the device does per wave, leaving the nonbonded
 *  sweep production's exclusions.
 *  @param[in] positions Axis-major coordinates, `3 * prepared.num_atoms` components.
 *  @param[inout] scratch Supplies the hop-matrix storage this call refills.
 *  @param[out] gradient Grown to match @p positions, then filled axis-major.
 *  @param[out] energy Receives the total in kcal per mol; zero until the device answers.
 *  @return Whether the device round trip completed, and which step refused it.
 */
[[nodiscard]] static mmff_energy_probe_t mmff_energy_and_gradient_(sfp::etkdg_prepared_molecule_t const &prepared,
                                                                   custd::span<float const> positions,
                                                                   gradient_scratch_t &scratch,
                                                                   std::vector<float> &gradient, float &energy) {
    gradient.assign(positions.size(), 0.0f);
    auto const bond_hops = sft::reset_square_matrix(scratch.bond_hops, prepared.num_atoms, sft::unreachable_hops_k);
    fill_bond_hops_into(prepared.mmff_bonds, bond_hops);
    mmff_energy_terms_t const terms {
        .atom_vdw = prepared.mmff_atom_vdw,
        .atom_charge = prepared.mmff_atom_charge,
        .bond_distance = bond_hops,
        .bonds = prepared.mmff_bonds,
        .angles = prepared.mmff_angles,
        .torsions = prepared.mmff_torsions,
        .oops = prepared.mmff_oops,
        .stretchbends = prepared.mmff_stretchbends,
        .interactions = prepared.mmff_interactions,
        .atom_offsets = prepared.mmff_atom_offsets,
    };
    energy = 0.0f;
    return gpu_mmff_energy(positions, prepared.num_atoms, terms, gradient, &energy);
}

/**
 *  @brief Central difference of the device energy with respect to one coordinate of `scratch.axis_major`.
 *  The perturbed component is restored exactly, so a caller may walk a whole molecule through one buffer.
 *  @param[in] axis Spatial dimension of the perturbed component, `0` for x through `2` for z.
 *  @param[inout] scratch Holds the coordinates, and the buffers the two energy calls fill.
 *  @param[out] derivative Receives the derivative in kcal per mol per angstrom; untouched on a refusal.
 *  @return Whether both device round trips completed, and which step refused one.
 */
[[nodiscard]] static mmff_energy_probe_t finite_difference_component_(sfp::etkdg_prepared_molecule_t const &prepared,
                                                                      std::uint32_t axis, std::uint32_t atom,
                                                                      gradient_scratch_t &scratch, float &derivative) {
    sft::conformer_soa_t const coordinates {scratch.axis_major.data(), prepared.num_atoms};
    float const original = coordinates(atom, axis);
    float energy_plus = 0.0f, energy_minus = 0.0f;

    coordinates(atom, axis) = original + finite_difference_step_k;
    mmff_energy_probe_t probe = mmff_energy_and_gradient_(prepared, scratch.axis_major, scratch, scratch.probe_gradient,
                                                          energy_plus);
    coordinates(atom, axis) = original - finite_difference_step_k;
    if (probe == mmff_energy_probe_t::completed_k)
        probe = mmff_energy_and_gradient_(prepared, scratch.axis_major, scratch, scratch.probe_gradient, energy_minus);

    // Restored before the refusal is answered, so a probe leaves borrowed coordinates as it found them.
    coordinates(atom, axis) = original;
    if (probe != mmff_energy_probe_t::completed_k) return probe;
    derivative = (energy_plus - energy_minus) / (2.0f * finite_difference_step_k);
    return mmff_energy_probe_t::completed_k;
}

#pragma endregion Device Evaluation

#pragma region Tolerances

/**
 *  @brief The band an analytical derivative is trusted to against a central difference of a `float` energy.
 *  `absolute` is accepted outright and decides a component whose derivative is near zero; `relative` is
 *  against the difference itself and takes over above unit magnitude.
 */
struct gradient_tolerance_t {
    float absolute = 0.0f;
    float relative = 0.0f;

    bool agree(float finite_difference, float analytical) const noexcept {
        float const error = std::fabs(analytical - finite_difference);
        return error < absolute || error / std::max(1.0f, std::fabs(finite_difference)) < relative;
    }
};

/**
 *  @brief The band one isolated out-of-plane term is held to. Its energy is a few kcal per mol whatever
 *  surrounds it, so cancellation does not grow with atom count: the worst error over the built-ins is
 *  3e-5, which the run prints beside this band.
 */
static constexpr gradient_tolerance_t single_term_tolerance_k {.absolute = 2.0e-3f, .relative = 1.0e-2f};

/**
 *  @brief The band the full stage is held to, decided by the size of the total it differentiates: a
 *  hundred-atom molecule pushes a few hundred kcal per mol through the same `float` subtraction, so the
 *  largest built-in sets it rather than the smallest. Worst error over them is 0.016.
 */
static constexpr gradient_tolerance_t full_stage_tolerance_k {.absolute = 2.5e-2f, .relative = 7.5e-2f};

/**
 *  @brief Whether the device total and RDKit's are the same number, at the resolution `float` allows.
 *
 *  The band is what the two implementations measurably achieve, not what they can be made to pass at: all
 *  ten built-ins agree to 3e-4 kcal per mol or better, so a whole kcal would hide three orders of regression.
 */
static bool energies_agree_(double rdkit_energy, double device_energy) noexcept {
    double const band = std::max(1.0e-2, 1.0e-4 * std::fabs(rdkit_energy));
    return std::fabs(device_energy - rdkit_energy) < band;
}

/**
 *  @brief Whether one axis sum of a gradient cancelled to zero, at the resolution its magnitudes allow.
 *  A translation-invariant field sums to zero exactly in real arithmetic; in `float` it sums to the
 *  rounding of what cancelled, so the band scales with those magnitudes and floors at 1e-2.
 */
static bool axis_sum_vanishes_(float magnitude_sum, float axis_sum) noexcept {
    return std::fabs(axis_sum) < std::max(1.0e-2f, 1.0e-5f * magnitude_sum);
}

#pragma endregion Tolerances

#pragma region Chiral Restraint

/**
 *  @brief How far one coordinate moves either way for the restraint's own central difference.
 *
 *  Five times the stage's step: the restraint's energy is not a chemical quantity, and at a weight of 100
 *  an inverted center carries some 2e4 kcal per mol, which `float` resolves to 2e-3 -- the stage's 1e-3
 *  step would leave the quotient a resolution of one. Truncation grows as the square of the step and
 *  stays well under the band either way.
 */
static constexpr float chiral_difference_step_k = 5.0e-3f;

/**
 *  @brief The band the restraint's derivative is trusted to against a central difference of its energy.
 *
 *  The absolute floor comes from the energy scale, not the derivative's: a true derivative of zero still
 *  reads back as the cancellation noise of two 2e4 totals, which no lower floor can accept. The relative
 *  band is ten times tighter than the stage's, because a wrong restraint gradient is wrong by a whole
 *  term -- a missing force on the center atom is a hundred percent error, not a percent.
 */
static constexpr gradient_tolerance_t chiral_restraint_tolerance_k {.absolute = 2.0f, .relative = 1.0e-3f};

/** Bond length every tetrahedral fixture places its neighbors at, in angstroms. */
static constexpr float fixture_bond_length_k = 1.5f;

/** What the restraint must make of one fixture; the three cases are the ones the two floors admit. */
enum class restraint_expectation_t : std::uint8_t {
    /** Both volumes clear their floors, so the restraint owes no energy and no force to anyone. */
    flat_bottom_k,
    /** Both volumes sit past their floors on the wrong side, so all five atoms are pushed. */
    both_terms_k,
    /** The pivot volume is satisfied and the center-based one is not, so only the center term pushes. */
    center_term_only_k,
};

/**
 *  @brief One geometry the restraint is scored on, with the descriptor the polish kernel would be handed.
 *  Positions are axis-major, `3 * num_atoms` components, and atom zero is the center.
 */
struct chiral_fixture_t {
    std::string_view name;
    std::vector<float> positions;
    std::uint32_t num_atoms = 0;
    sft::chiral_center_descriptor_t center {};
    restraint_expectation_t expectation = restraint_expectation_t::flat_bottom_k;
};

/**
 *  @brief The four fixtures, built around one ideal tetrahedron.
 *
 *  At a 1.5 angstrom bond the pivot-based volume is 10.4 cubic angstrom against its floor of 5 and the
 *  center-based one 2.60 against its floor of 1.5, which is why the second floor is not the first.
 *  `center_through_plane` is the geometry this restraint exists for: neighbors untouched, so the
 *  pivot-based volume is still 10.4 and its term flat, while the center alone has crossed their plane --
 *  what a restraint applying no force to the center cannot see, and what `assignChiralTypesFrom3D` reads
 *  as the opposite handedness.
 */
static std::vector<chiral_fixture_t> chiral_fixtures_() {
    float const offset = fixture_bond_length_k / 1.7320508f;
    // Four ideal tetrahedral directions, as (atom, axis) rows behind atom zero, the center.
    float const directions[4][3] = {{offset, offset, offset},
                                    {offset, -offset, -offset},
                                    {-offset, offset, -offset},
                                    {-offset, -offset, offset}};

    auto build_ = [&](std::string_view name, std::uint32_t num_atoms, std::int8_t target_sign, float minimum_volume,
                      std::uint8_t pivot, float center_offset,
                      restraint_expectation_t expectation) -> chiral_fixture_t {
        chiral_fixture_t fixture {.name = name,
                                  .positions = std::vector<float>(std::size_t {3} * num_atoms, 0.0f),
                                  .num_atoms = num_atoms,
                                  .center = {.center_atom = 0,
                                             .neighbor_atoms = {1, 2, 3, pivot},
                                             .target_sign = target_sign,
                                             .padding = {0, 0},
                                             .minimum_volume = minimum_volume,
                                             .padding_float = 0.0f},
                                  .expectation = expectation};
        sft::conformer_soa_t const coordinates {fixture.positions.data(), num_atoms};
        for (std::uint32_t neighbor = 1; neighbor < num_atoms; ++neighbor)
            for (std::uint32_t axis = 0; axis < 3; ++axis) coordinates(neighbor, axis) = directions[neighbor - 1][axis];
        // The plane normal of the first three neighbors is (1, 1, -1) normalized; walking the center
        // along it by more than 0.5 angstrom carries the center-based volume through zero.
        float const normal[3] = {0.5773503f, 0.5773503f, -0.5773503f};
        for (std::uint32_t axis = 0; axis < 3; ++axis) coordinates(0, axis) = center_offset * normal[axis];
        return fixture;
    };

    std::vector<chiral_fixture_t> fixtures;
    fixtures.push_back(build_("four_coordinate_satisfied", 5, 1, sft::min_four_coordinate_volume_k, 4, 0.0f,
                              restraint_expectation_t::flat_bottom_k));
    fixtures.push_back(build_("four_coordinate_inverted", 5, -1, sft::min_four_coordinate_volume_k, 4, 0.0f,
                              restraint_expectation_t::both_terms_k));
    fixtures.push_back(build_("center_through_plane", 5, 1, sft::min_four_coordinate_volume_k, 4, 0.8f,
                              restraint_expectation_t::center_term_only_k));
    // A three-coordinate center is stored as its own fourth neighbor, so both expressions read the same
    // volume and the second floor, at a fraction of the first, can never bind on its own.
    fixtures.push_back(build_("three_coordinate_inverted", 4, -1, sft::min_three_coordinate_volume_k, 0, 0.0f,
                              restraint_expectation_t::both_terms_k));
    return fixtures;
}

/** Central difference of the restraint's energy with respect to one coordinate of @p positions. */
[[nodiscard]] static mmff_energy_probe_t chiral_difference_component_(chiral_fixture_t const &fixture,
                                                                      std::uint32_t atom, std::uint32_t axis,
                                                                      std::vector<float> &positions,
                                                                      std::vector<float> &probe_gradient,
                                                                      float &derivative) {

    custd::span<sft::chiral_center_descriptor_t const> const centers {&fixture.center, 1};
    sft::conformer_soa_t const coordinates {positions.data(), fixture.num_atoms};
    float const original = coordinates(atom, axis);
    float energy_plus = 0.0f, energy_minus = 0.0f;

    coordinates(atom, axis) = original + chiral_difference_step_k;
    mmff_energy_probe_t probe = sftest::gpu_chiral_restraint(positions, fixture.num_atoms, centers, probe_gradient,
                                                             &energy_plus);
    coordinates(atom, axis) = original - chiral_difference_step_k;
    if (probe == mmff_energy_probe_t::completed_k)
        probe = sftest::gpu_chiral_restraint(positions, fixture.num_atoms, centers, probe_gradient, &energy_minus);

    coordinates(atom, axis) = original;
    if (probe != mmff_energy_probe_t::completed_k) return probe;
    derivative = (energy_plus - energy_minus) / (2.0f * chiral_difference_step_k);
    return mmff_energy_probe_t::completed_k;
}

/** Sum of squares of one atom's three gradient components, which names whether it was pushed at all. */
static float gradient_magnitude_(gradient_soa_t gradient, std::uint32_t atom) noexcept {
    float total = 0.0f;
    for (std::uint32_t axis = 0; axis < 3; ++axis) total += gradient(atom, axis) * gradient(atom, axis);
    return total;
}

/**
 *  @brief Gate the chiral restraint's own energy and derivative, which no MMFF94 reference can see.
 *
 *  Published MMFF94 carries no chiral term, so the sweep below runs with the restraint off: its only
 *  reference is a central difference of its own energy, its only claim which atoms its two terms push.
 *  @param[inout] terms Receives one comparison per structural claim of every fixture.
 *  @param[inout] derivatives Receives one comparison per gradient component of every fixture.
 *  @param[out] worst_error Receives the largest error any component showed, so the band stays auditable.
 *  @return Whether every fixture's device round trip completed.
 */
[[nodiscard]] static mmff_energy_probe_t check_chiral_restraint_(diagnostic_reporter_t &terms,
                                                                 diagnostic_reporter_t &derivatives,
                                                                 float &worst_error) {
    std::vector<float> analytical, probe_gradient;
    for (chiral_fixture_t &fixture : chiral_fixtures_()) {
        std::size_t const num_components = std::size_t {3} * fixture.num_atoms;
        analytical.assign(num_components, 0.0f);
        probe_gradient.assign(num_components, 0.0f);
        custd::span<sft::chiral_center_descriptor_t const> const centers {&fixture.center, 1};

        float energy = 0.0f;
        mmff_energy_probe_t probe = sftest::gpu_chiral_restraint(fixture.positions, fixture.num_atoms, centers,
                                                                 analytical, &energy);
        if (probe != mmff_energy_probe_t::completed_k) return probe;

        gradient_soa_t const gradient {analytical.data(), fixture.num_atoms};
        std::uint32_t const pivot = fixture.center.neighbor_atoms[3];
        float const center_push = gradient_magnitude_(gradient, 0);
        float const pivot_push = gradient_magnitude_(gradient, pivot);

        // One line per structural claim, so a fixture that violates one names which.
        auto claim = terms.mismatch("restraint", "fixture={}", fixture.name);
        switch (fixture.expectation) {
        case restraint_expectation_t::flat_bottom_k:
            claim.field("energy", 0.0f, energy, exactly_equal);
            claim.field("center_push", 0.0f, center_push, exactly_equal);
            claim.field("pivot_push", 0.0f, pivot_push, exactly_equal);
            break;
        case restraint_expectation_t::both_terms_k:
            claim.field("energy_positive", true, energy > 0.0f, exactly_equal);
            claim.field("center_pushed", true, center_push > 0.0f, exactly_equal);
            claim.field("pivot_pushed", true, pivot_push > 0.0f, exactly_equal);
            break;
        case restraint_expectation_t::center_term_only_k:
            claim.field("energy_positive", true, energy > 0.0f, exactly_equal);
            claim.field("center_pushed", true, center_push > 0.0f, exactly_equal);
            claim.field("pivot_push", 0.0f, pivot_push, exactly_equal);
            break;
        }

        for (std::uint32_t atom = 0; atom < fixture.num_atoms; ++atom)
            for (std::uint32_t axis = 0; axis < 3; ++axis) {
                float difference = 0.0f;
                probe = chiral_difference_component_(fixture, atom, axis, fixture.positions, probe_gradient,
                                                     difference);
                if (probe != mmff_energy_probe_t::completed_k) return probe;
                worst_error = std::max(worst_error, std::fabs(gradient(atom, axis) - difference));
                derivatives.mismatch("restraint_gradient", "fixture={} atom={}", fixture.name, atom)
                    .field(gradient_field_name_(axis), difference, gradient(atom, axis),
                           [](float reference, float observed) {
                               return chiral_restraint_tolerance_k.agree(reference, observed);
                           });
            }
    }
    return mmff_energy_probe_t::completed_k;
}

#pragma endregion Chiral Restraint

#pragma region Sections

/** How one molecule's turn through the sweep ended. */
enum class molecule_outcome_t : std::uint8_t {
    /** Both references were available and every section received its comparisons. */
    compared_k,
    /** Preprocessing refused it: an unsupported element, too many atoms, or a SMILES that will not parse. */
    unprepared_k,
    /** ETKDG produced no conformer, so there are no coordinates to compare at. */
    unembedded_k,
    /** RDKit's atom count differs from preprocessing's, so the two atom orders cannot be aligned. */
    unaligned_k,
    /** RDKit cannot build MMFF94 properties for it, so there is no reference energy. */
    unreferenced_k,
    /** The device refused a launch, so the gradient it would have been compared against was never written. */
    refused_k,
};

/**
 *  @brief What the sweep counted, threaded by reference so the sweep owns no module state.
 *
 *  One counter per `molecule_outcome_t`, plus the molecules with no out-of-plane term for that section to
 *  probe. A refusal is a failure of the run rather than of a molecule, so `first_refusal` keeps the step
 *  the count cannot carry. The two worst errors are the margin the gradient bands were chosen against,
 *  printed every run: ten times the worst error is slack, just above it fails on the next molecule.
 */
struct gradient_tally_t {
    std::uint32_t compared = 0;
    std::uint32_t unprepared = 0;
    std::uint32_t unembedded = 0;
    std::uint32_t unaligned = 0;
    std::uint32_t unreferenced = 0;
    std::uint32_t refused = 0;
    mmff_energy_probe_t first_refusal = mmff_energy_probe_t::completed_k;
    std::uint32_t without_oop = 0;
    float worst_oop_error = 0.0f;
    float worst_stage_error = 0.0f;
};

/**
 *  @brief One reporter per compared section, so each closes with its own `compared=` count. Separate
 *  reporters rather than several `summarize` calls on one: a molecule contributes to all of them before
 *  the next starts, and a shared reporter would credit every count to whichever summarized first.
 */
struct gradient_reporters_t {
    /** Analytical gradient of one isolated out-of-plane bend against a central difference of its energy. */
    diagnostic_reporter_t oop_gradient {"finite_difference", "analytical"};
    /** Device total energy against RDKit's, at identical coordinates. */
    diagnostic_reporter_t stage_energy {"rdkit", "device"};
    /** The three axis sums of the full gradient, which a translation-invariant field leaves at zero. */
    diagnostic_reporter_t translation_invariance {"invariant", "device"};
    /** Full-stage analytical gradient against a central difference, over sampled components. */
    diagnostic_reporter_t stage_gradient {"finite_difference", "analytical"};
    /** Which atoms each of the chiral restraint's two terms pushes, and what it charges for them. */
    diagnostic_reporter_t chiral_terms {"expected", "observed"};
    /** The chiral restraint's analytical derivative against a central difference of its own energy. */
    diagnostic_reporter_t chiral_restraint {"finite_difference", "analytical"};

    /** Cap what each section prints, so one pathological molecule cannot bury the log. */
    void limit_printed_lines(std::size_t maximum) noexcept {
        oop_gradient.limit_printed_lines(maximum);
        stage_energy.limit_printed_lines(maximum);
        translation_invariance.limit_printed_lines(maximum);
        stage_gradient.limit_printed_lines(maximum);
        chiral_terms.limit_printed_lines(maximum);
        chiral_restraint.limit_printed_lines(maximum);
    }

    std::size_t comparison_count() const noexcept {
        return oop_gradient.comparison_count() + stage_energy.comparison_count() +
               translation_invariance.comparison_count() + stage_gradient.comparison_count() +
               chiral_terms.comparison_count() + chiral_restraint.comparison_count();
    }

    std::size_t mismatch_count() const noexcept {
        return oop_gradient.mismatch_count() + stage_energy.mismatch_count() + translation_invariance.mismatch_count() +
               stage_gradient.mismatch_count() + chiral_terms.mismatch_count() + chiral_restraint.mismatch_count();
    }

    /** Close all six sections; true when each compared something and every field agreed. */
    bool summarize_sections() {
        oop_gradient.summarize("oop_gradient");
        stage_energy.summarize("stage_energy");
        translation_invariance.summarize("translation_invariance");
        stage_gradient.summarize("stage_gradient");
        chiral_terms.summarize("chiral_terms");
        chiral_restraint.summarize("chiral_restraint");
        return oop_gradient.passed() && stage_energy.passed() && translation_invariance.passed() &&
               stage_gradient.passed() && chiral_terms.passed() && chiral_restraint.passed();
    }
};

/**
 *  @brief Fill @p single with @p prepared's first out-of-plane bend and nothing else.
 *
 *  Every other category is emptied and the term's four positions become the whole interaction list; with
 *  no van der Waals parameters or charges the nonbonded sweep adds nothing, so i, j, k and l see only it.
 *  @param[out] single Reused across molecules, so every category it owns is refilled or cleared here.
 */
static void fill_single_oop_term_into_(sfp::etkdg_prepared_molecule_t const &prepared,
                                       sfp::etkdg_prepared_molecule_t &single) {
    single.num_atoms = prepared.num_atoms;
    single.mmff_atom_vdw.clear();
    single.mmff_atom_charge.clear();
    single.mmff_bonds.clear();
    single.mmff_angles.clear();
    single.mmff_torsions.clear();
    single.mmff_stretchbends.clear();
    single.mmff_interactions.clear();
    single.mmff_oops.assign(1, prepared.mmff_oops.front());
    single.mmff_atom_offsets.assign(prepared.num_atoms, {});

    sft::mmff94_oop_t const &oop = single.mmff_oops.front();
    std::uint32_t const atoms[] = {oop.atom_i, oop.atom_j, oop.atom_k, oop.atom_l};
    for (std::uint32_t role = 0; role < 4; ++role) { // the term's i, j, k and l positions
        std::uint32_t const position = static_cast<std::uint32_t>(single.mmff_interactions.size());
        single.mmff_atom_offsets[atoms[role]].oop_offset = position;
        single.mmff_atom_offsets[atoms[role]].oop_count = 1;
        single.mmff_interactions.push_back({0, sft::mmff94_out_of_plane_k, static_cast<sft::atom_role_t>(role)});
    }
}

/**
 *  @brief Compare the isolated out-of-plane gradient against a central difference of its own energy.
 *  @param[in] smiles Molecule under test, carried into the diagnostic keys.
 *  @param[inout] scratch Holds the coordinates the probes perturb and the buffers they fill.
 *  @param[inout] tally Keeps the worst error this section saw.
 *  @return Whether every device round trip completed, and which step refused one.
 */
[[nodiscard]] static mmff_energy_probe_t compare_oop_gradient_(std::string const &smiles, gradient_scratch_t &scratch,
                                                               diagnostic_reporter_t &reporter,
                                                               gradient_tally_t &tally) {
    fill_single_oop_term_into_(scratch.prepared, scratch.single_oop);
    [[maybe_unused]] float term_energy = 0.0f;
    if (mmff_energy_probe_t const staged = mmff_energy_and_gradient_(scratch.single_oop, scratch.axis_major, scratch,
                                                                     scratch.term_gradient, term_energy);
        staged != mmff_energy_probe_t::completed_k)
        return staged;
    gradient_soa_t const term_gradient {scratch.term_gradient.data(), scratch.single_oop.num_atoms};

    sft::mmff94_oop_t const &oop = scratch.single_oop.mmff_oops.front();
    std::uint32_t const oop_atoms[] = {oop.atom_i, oop.atom_j, oop.atom_k, oop.atom_l};
    auto const agree = [](float finite_difference, float analytical) noexcept {
        return single_term_tolerance_k.agree(finite_difference, analytical);
    };

    for (std::uint32_t role = 0; role < 4; ++role) {     // the term's i, j, k and l positions
        for (std::uint32_t axis = 0; axis < 3; ++axis) { // the three spatial dimensions
            std::uint32_t const atom = oop_atoms[role];
            float finite_difference = 0.0f;
            if (mmff_energy_probe_t const probed = finite_difference_component_(scratch.single_oop, axis, atom, scratch,
                                                                                finite_difference);
                probed != mmff_energy_probe_t::completed_k)
                return probed;
            float const analytical = term_gradient(atom, axis);
            tally.worst_oop_error = std::max(tally.worst_oop_error, std::fabs(analytical - finite_difference));
            reporter.mismatch("oop_gradient", "smiles={} atom={} role={}", smiles, atom, role)
                .field(gradient_field_name_(axis), finite_difference, analytical, agree);
        }
    }
    return mmff_energy_probe_t::completed_k;
}

/**
 *  @brief Compare the three axis sums of the full gradient against the zero translation invariance requires.
 *  @param[in] smiles Molecule under test, carried into the diagnostic keys.
 *  @param[in] gradient Full-stage analytical gradient, axis-major.
 */
static void compare_translation_invariance_(std::string const &smiles, gradient_soa_t gradient,
                                            diagnostic_reporter_t &reporter) {
    std::uint32_t const num_atoms = gradient.extent(0);
    float axis_sums[3] = {0.0f, 0.0f, 0.0f};
    float magnitude_sum = 0.0f;
    for (std::uint32_t axis = 0; axis < 3; ++axis) // the three spatial dimensions
        for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
            float const component = gradient(atom, axis);
            axis_sums[axis] += component;
            magnitude_sum += std::fabs(component);
        }

    auto const vanishes = [magnitude_sum](float invariant, float axis_sum) noexcept {
        return axis_sum_vanishes_(magnitude_sum, axis_sum - invariant);
    };
    reporter.mismatch("translation_invariance", "smiles={} atoms={}", smiles, num_atoms)
        .field("gradient_sum_x", 0.0f, axis_sums[0], vanishes)
        .field("gradient_sum_y", 0.0f, axis_sums[1], vanishes)
        .field("gradient_sum_z", 0.0f, axis_sums[2], vanishes);
}

/**
 *  @brief Compare sampled components of the full-stage gradient against central differences.
 *
 *  The stride follows the molecule's own component count, so samples span all three axis blocks whatever
 *  the atom count; a fixed stride lands entirely inside x past a few dozen atoms.
 *  @param[in] smiles Molecule under test, carried into the diagnostic keys.
 *  @param[inout] scratch Holds the coordinates the probes perturb and the gradient they are compared to.
 *  @param[inout] tally Keeps the worst error this section saw.
 *  @return Whether every device round trip completed, and which step refused one.
 */
[[nodiscard]] static mmff_energy_probe_t compare_stage_gradient_(std::string const &smiles, gradient_scratch_t &scratch,
                                                                 diagnostic_reporter_t &reporter,
                                                                 gradient_tally_t &tally) {
    std::uint32_t const num_atoms = scratch.prepared.num_atoms;
    gradient_soa_t const stage_gradient {scratch.stage_gradient.data(), num_atoms};
    std::size_t const total_components = stage_gradient.size();
    std::size_t const stride = std::max<std::size_t>(1, total_components / sampled_components_per_molecule_k);
    auto const agree = [](float finite_difference, float analytical) noexcept {
        return full_stage_tolerance_k.agree(finite_difference, analytical);
    };

    // An axis-major view packs one whole axis at a time, so the axis is the quotient and the atom the remainder.
    for (std::size_t component = 0; component < total_components; component += stride) {
        std::uint32_t const axis = static_cast<std::uint32_t>(component / num_atoms);
        std::uint32_t const atom = static_cast<std::uint32_t>(component % num_atoms);
        float finite_difference = 0.0f;
        if (mmff_energy_probe_t const probed = finite_difference_component_(scratch.prepared, axis, atom, scratch,
                                                                            finite_difference);
            probed != mmff_energy_probe_t::completed_k)
            return probed;
        float const analytical = stage_gradient(atom, axis);
        tally.worst_stage_error = std::max(tally.worst_stage_error, std::fabs(analytical - finite_difference));
        reporter.mismatch("stage_gradient", "smiles={} atom={}", smiles, atom)
            .field(gradient_field_name_(axis), finite_difference, analytical, agree);
    }
    return mmff_energy_probe_t::completed_k;
}

#pragma endregion Sections

#pragma region Sweep

/**
 *  @brief Run one molecule through every section.
 *  @param[inout] scratch Every buffer the molecule needs, carried in from above the loop.
 *  @param[inout] tally Receives what the sections cannot: the skips, the roster hits and the worst errors.
 *  @return Whether the molecule was comparable, and why not when it was not.
 */
static molecule_outcome_t compare_molecule_(std::string const &smiles, gradient_reporters_t &reporters,
                                            gradient_scratch_t &scratch, gradient_tally_t &tally) {
    sfp::prepare_molecule_into(smiles, scratch.prepared);
    if (scratch.prepared.num_atoms == 0) return molecule_outcome_t::unprepared_k;

    std::unique_ptr<RDKit::RWMol> molecule = mol_from_smiles(smiles);
    if (!molecule) return molecule_outcome_t::unprepared_k;
    if (!embed_molecule(*molecule)) return molecule_outcome_t::unembedded_k;

    // A salt reaches here: preprocessing keeps a different atom set than `addHs`, so no order aligns.
    std::uint32_t const rdkit_atoms = molecule->getNumAtoms();
    if (rdkit_atoms != scratch.prepared.num_atoms) {
        reporters.stage_energy.note("atom_count", "smiles={}", smiles)
            .field("num_atoms", rdkit_atoms, scratch.prepared.num_atoms);
        return molecule_outcome_t::unaligned_k;
    }

    auto const atom_major = sft::reset_coordinates_aos<sft::coordinate_role_t::position_k>(scratch.atom_major,
                                                                                           rdkit_atoms);
    read_conformer_into(*molecule, atom_major);
    sft::conformer_soa_t const axis_major =
        sft::reset_coordinates_soa<sft::coordinate_role_t::position_k, sft::conformer_dimensions_k>(scratch.axis_major,
                                                                                                    rdkit_atoms);
    sft::transpose_into(atom_major, axis_major);

    double const rdkit_energy = rdkit_mmff94_energy(*molecule);
    if (!std::isfinite(rdkit_energy)) return molecule_outcome_t::unreferenced_k;

    // Keep the first refusal the molecule met, so the run's failure can name the step that refused.
    auto const refuse_ = [&tally](mmff_energy_probe_t probe) {
        if (tally.first_refusal == mmff_energy_probe_t::completed_k) tally.first_refusal = probe;
        return molecule_outcome_t::refused_k;
    };

    float device_energy = 0.0f;
    if (mmff_energy_probe_t const probe = mmff_energy_and_gradient_(scratch.prepared, scratch.axis_major, scratch,
                                                                    scratch.stage_gradient, device_energy);
        probe != mmff_energy_probe_t::completed_k)
        return refuse_(probe);
    reporters.stage_energy.mismatch("stage_energy", "smiles={} atoms={}", smiles, scratch.prepared.num_atoms)
        .field("total_energy", rdkit_energy, static_cast<double>(device_energy), energies_agree_);

    compare_translation_invariance_(smiles, gradient_soa_t {scratch.stage_gradient.data(), rdkit_atoms},
                                    reporters.translation_invariance);
    if (mmff_energy_probe_t const probe = compare_stage_gradient_(smiles, scratch, reporters.stage_gradient, tally);
        probe != mmff_energy_probe_t::completed_k)
        return refuse_(probe);

    if (scratch.prepared.mmff_oops.empty()) ++tally.without_oop;
    else if (mmff_energy_probe_t const probe = compare_oop_gradient_(smiles, scratch, reporters.oop_gradient, tally);
             probe != mmff_energy_probe_t::completed_k)
        return refuse_(probe);
    return molecule_outcome_t::compared_k;
}

/**
 *  @brief Walk the corpus, comparing every molecule the two references both cover. Scratch declared once
 *  above the loop grows to the largest molecule and no further, and the sweep runs to completion whatever
 *  it finds, because the counts are the output.
 *  @param[in] corpus SMILES to compare, built-ins first.
 *  @param[inout] tally Filled with the per-molecule outcomes.
 */
static void sweep_corpus_(std::vector<std::string> const &corpus, gradient_reporters_t &reporters,
                          gradient_tally_t &tally) {
    gradient_scratch_t scratch;
    for (std::string const &smiles : corpus) {
        switch (compare_molecule_(smiles, reporters, scratch, tally)) {
        case molecule_outcome_t::compared_k: ++tally.compared; break;
        case molecule_outcome_t::unprepared_k: ++tally.unprepared; break;
        case molecule_outcome_t::unembedded_k: ++tally.unembedded; break;
        case molecule_outcome_t::unaligned_k: ++tally.unaligned; break;
        case molecule_outcome_t::unreferenced_k: ++tally.unreferenced; break;
        case molecule_outcome_t::refused_k: ++tally.refused; break;
        }
    }
}

#pragma endregion Sweep

} // namespace smallfield::tests

int main(int argc, char **argv) {
    fmt::println("{}: device MMFF94 energy and gradient vs RDKit and central differences", test_name_k);

    auto const corpus_selection = sftest::select_corpus(argc, argv, 50);
    if (corpus_selection.knob == sftest::corpus_knob_t::refused_k) {
        std::fflush(stdout);
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", test_name_k);
        return 2;
    }

    std::vector<std::string> corpus = sftest::standard_test_smiles();
    if (corpus_selection.knob == sftest::corpus_knob_t::named_k) {
        std::vector<std::string> streamed;
        auto const outcome = sftest::load_smiles_into(corpus_selection.path, corpus_selection.max_molecules, streamed);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome),
                                       corpus_selection.path);
        fmt::println("corpus={} molecules={} plus built-ins", corpus_selection.path,
                     fmt::group_digits(streamed.size()));
        for (auto &smiles : streamed) corpus.push_back(std::move(smiles));
    }
    else fmt::println("corpus=built-ins only; name one on the command line or in SMALLFIELD_TEST_CORPUS");

    sftest::gradient_reporters_t reporters;
    reporters.limit_printed_lines(sftest::max_printed_lines_k);

    // Scored before the corpus: RDKit's MMFF94 has no chiral term, so the sweep runs with it off.
    float worst_restraint_error = 0.0f;
    sftest::mmff_energy_probe_t const restraint_probe = sftest::check_chiral_restraint_(
        reporters.chiral_terms, reporters.chiral_restraint, worst_restraint_error);
    if (restraint_probe != sftest::mmff_energy_probe_t::completed_k)
        return sftest::report_fail(test_name_k, "the device refused a chiral restraint fixture as {}",
                                   sftest::mmff_energy_probe_reason(restraint_probe));
    fmt::println("worst chiral restraint gradient error {} against an absolute floor of {:.3g}", worst_restraint_error,
                 sftest::chiral_restraint_tolerance_k.absolute);

    sftest::gradient_tally_t tally;
    sftest::sweep_corpus_(corpus, reporters, tally);

    fmt::println("molecules compared={} unprepared={} unembedded={} unaligned={} " // what the sweep saw
                 "unreferenced={} refused={} without_oop={}",                      // and what it stepped past
                 tally.compared, tally.unprepared, tally.unembedded, tally.unaligned, tally.unreferenced, tally.refused,
                 tally.without_oop);
    fmt::println("worst gradient error oop={} stage={} against absolute floors {:.3g} and {:.3g}, " // the margin
                 "above which the relative band decides",                                           // and the band
                 tally.worst_oop_error, tally.worst_stage_error, sftest::single_term_tolerance_k.absolute,
                 sftest::full_stage_tolerance_k.absolute);
    bool const sections_passed = reporters.summarize_sections();
    if (tally.refused != 0)
        return sftest::report_fail(test_name_k, "the device refused {} molecules, first as {}", tally.refused,
                                   sftest::mmff_energy_probe_reason(tally.first_refusal));
    if (tally.compared == 0) return sftest::report_fail(test_name_k, "corpus yielded no comparable molecule");
    if (!sections_passed)
        return sftest::report_fail(test_name_k, "{} mismatches over {} comparisons", reporters.mismatch_count(),
                                   reporters.comparison_count());
    return sftest::report_pass(test_name_k);
}
