/**
 *  @file test/preprocess/test_rdkit_divergences.cpp
 *  @brief Assert every recorded disagreement with RDKit that we believe RDKit is wrong about.
 *
 *  `rdkit_divergence_table.cpp` names each defect once and records both sides; this binary re-measures it
 *  all and asserts, per run:
 *
 *   - @b Reachable: the SMILES parses and the named atoms exist. Unmeasurable fails rather than skips,
 *     being indistinguishable from satisfied.
 *   - @b Reference-still-says-it: RDKit reproduces `rdkit_values` within `rdkit_tolerance` -- the upgrade
 *     tripwire, firing whether upstream fixed the defect or merely moved it.
 *   - @b We-still-say-it: SmallField reproduces `our_values` within `our_tolerance`.
 *   - @b Still-apart: the two stay more than `divergence_threshold` apart; convergence retires a row rather
 *     than relaxing it.
 *   - @b Reaching-alike: over every corpus pair both sides agree whether it is constrained at all, which
 *     comes first -- two numbers compare only where both produced one.
 *   - @b Complete: a corpus rescan finds no disagreement of a recorded kind that no row explains.
 *
 *  `fixed_in` at or below the installed release inverts @b Still-apart to @b agree, so a retired divergence
 *  stays as a permanent regression test. The rescan covers the pairwise bounds kinds a molecule enumerates;
 *  a torsion preference is instead a property of a Fourier coefficient set that 37 of the 105 small-ring
 *  patterns share, so its row carries the defect once. Corpus: the built-in standards, the registry's own
 *  molecules -- always rescanned against itself -- and `adversarial.smi` beside this file, which makes the
 *  rescan a gate rather than a formality; `argv[1]` or `SMALLFIELD_TEST_CORPUS` replaces that last part.
 */

#include <cmath>   // `std::fabs`, `std::cos`, `std::sin`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::uint32_t`

#include <iterator>    // `std::size`
#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <DistGeom/BoundsMatrix.h>
#include <ForceField/ForceField.h>
#include <Geometry/point.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionAngleContribs.h>
#include <GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.h>
#include <RDGeneral/versions.h>

#include "smallfield/preprocess/csd_torsion_table.hpp"
#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

#include "test/preprocess/rdkit_divergence_table.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_rdkit_divergences";

/**
 *  @brief Molecules taken from a corpus when no cap is given; the registry's own are always walked.
 *
 *  Above the built-in corpus so the default run walks all of it, and high enough that naming a larger corpus
 *  without `SMALLFIELD_TEST_MAX_MOLS` gets that corpus rather than a truncation reading like a sweep.
 */
static constexpr std::size_t default_max_molecules_k = 10000;

namespace smallfield::tests {

#pragma region Measurement

/** Why a row could not be measured, or that it was. */
enum class divergence_reach_t : std::uint8_t {
    /** Both sides produced a value pair. */
    measured_k,
    /** `SmilesToMol` refused the row's molecule. */
    unparsable_smiles_k,
    /** An atom index the row names is past the molecule's atom count. */
    atom_out_of_range_k,
    /** The named atoms are not the bond separation the row's kind claims. */
    topology_does_not_match_kind_k,
    /** No experimental torsion of the molecule spans the four atoms the row names. */
    no_such_torsion_k,
    /** `build_descriptor` refused the molecule, so our own side has nothing to say. */
    descriptor_refused_k,
    /** The row's kind has no measurement written for it yet. */
    no_measurement_for_kind_k,
};

/** The sentence a reach outcome reads as, for the diagnostic that reports the row unreachable. */
static std::string_view divergence_reach_reason(divergence_reach_t reach) noexcept {
    switch (reach) {
    case divergence_reach_t::measured_k: return "measured";
    case divergence_reach_t::unparsable_smiles_k: return "smiles did not parse";
    case divergence_reach_t::atom_out_of_range_k: return "atom index past the molecule";
    case divergence_reach_t::topology_does_not_match_kind_k: return "bond separation is not the kind's";
    case divergence_reach_t::no_such_torsion_k: return "no experimental torsion spans those atoms";
    case divergence_reach_t::descriptor_refused_k: return "build_descriptor refused the molecule";
    case divergence_reach_t::no_measurement_for_kind_k: return "no measurement for that kind";
    }
    return "unrecorded reach outcome";
}

/** Both sides of one row, as measured now. */
struct divergence_measurement_t {
    float rdkit_values[2] = {0.0f, 0.0f};
    float our_values[2] = {0.0f, 0.0f};
};

/**
 *  Buffers the sweep reuses, declared above every loop so a corpus pass grows them at first use only. Both
 *  bounds matrices are N × N row-major -- upper above the diagonal, lower below -- so one index expression
 *  reads either, and `hops` carries each pair's bond separation, which fixes its 1-2, 1-3 or 1-4 kind.
 */
struct divergence_scratch_t {
    std::vector<float> our_bounds;
    std::vector<double> rdkit_bounds;
    std::vector<std::uint8_t> hops;
    sft::molecule_descriptor_t descriptor;
};

/**
 *  @brief Flatten RDKit's topological bounds for @p molecule into @p flat, in our own N × N row-major layout,
 *         so the pair lookups below read both matrices with one index expression rather than two conventions.
 *  @param[in] molecule Hydrogens must already be explicit.
 *  @param[out] flat Resized to N × N and fully overwritten.
 */
static void fill_rdkit_bounds_into_(RDKit::ROMol const &molecule, std::vector<double> &flat) {
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    auto *bounds_raw = new DistGeom::BoundsMatrix(num_atoms);
    DistGeom::BoundsMatPtr bounds_matrix(bounds_raw);
    RDKit::DGeomHelpers::initBoundsMat(bounds_raw);
    RDKit::DGeomHelpers::setTopolBounds(molecule, bounds_matrix);

    flat.assign(static_cast<std::size_t>(num_atoms) * num_atoms, 0.0);
    for (std::uint32_t row = 0; row < num_atoms; ++row) {
        for (std::uint32_t column = row + 1; column < num_atoms; ++column) {
            flat[row * num_atoms + column] = bounds_matrix->getUpperBound(row, column);
            flat[column * num_atoms + row] = bounds_matrix->getLowerBound(row, column);
        }
    }
}

/** Read one pair out of a bounds matrix in the shared layout, as `{lower, upper}`. */
template <typename scalar_type_>
static void read_bounds_pair_(std::vector<scalar_type_> const &flat, std::uint32_t num_atoms, std::uint32_t first_atom,
                              std::uint32_t second_atom, float (&out)[2]) {
    std::uint32_t const row = first_atom < second_atom ? first_atom : second_atom;
    std::uint32_t const column = first_atom < second_atom ? second_atom : first_atom;
    out[0] = static_cast<float>(flat[column * num_atoms + row]);
    out[1] = static_cast<float>(flat[row * num_atoms + column]);
}

/**
 *  @brief The pairwise bounds kind each bond separation sorts a pair into, indexed by `hops - 1`.
 *
 *  One table read one way is the whole relation -- the rescan labels a found pair with it, a registry row is
 *  checked against it. An inverse table would copy the same three facts with nothing to catch a disagreement.
 */
static constexpr divergence_kind_t bounds_kinds_by_separation_k[] = {
    divergence_kind_t::bounds_1_2_k,
    divergence_kind_t::bounds_1_3_k,
    divergence_kind_t::bounds_1_4_k,
};

/** Widest separation the table above labels, taken from it so a guard and the table cannot disagree. */
static constexpr std::uint8_t widest_labelled_separation_k = static_cast<std::uint8_t>(
    std::size(bounds_kinds_by_separation_k));

/**
 *  @brief Fill both bounds matrices and the hop matrix for @p molecule, which has explicit hydrogens.
 *  @param[inout] scratch Buffers to fill; grown at most once per run.
 *  @return Whether our own descriptor could be built, which is the only way this fails.
 */
static divergence_reach_t fill_bounds_scratch_(RDKit::ROMol const &molecule, divergence_scratch_t &scratch) {
    if (sft::build_descriptor(molecule, scratch.descriptor) != types::conformer_status_t::success_k)
        return divergence_reach_t::descriptor_refused_k;

    std::uint32_t const num_atoms = molecule.getNumAtoms();
    fill_rdkit_bounds_into_(molecule, scratch.rdkit_bounds);
    sfp::build_bounds_matrix(scratch.descriptor, sft::reset_square_matrix(scratch.our_bounds, num_atoms, 0.0f));
    scratch.hops.assign(static_cast<std::size_t>(num_atoms) * num_atoms, 0);
    sft::fill_bond_distances(scratch.descriptor, sft::hop_matrix_t {scratch.hops.data(), num_atoms, num_atoms});
    return divergence_reach_t::measured_k;
}

/** Measure one bounds row: RDKit's `setTopolBounds` against our `build_bounds_matrix`. */
static divergence_reach_t measure_bounds_row_(rdkit_divergence_row_t const &row, RDKit::ROMol const &molecule,
                                              divergence_scratch_t &scratch, divergence_measurement_t &measurement) {
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    std::uint32_t const first_atom = row.atoms[0], second_atom = row.atoms[1];
    if (first_atom >= num_atoms || second_atom >= num_atoms || first_atom == second_atom)
        return divergence_reach_t::atom_out_of_range_k;

    divergence_reach_t const filled = fill_bounds_scratch_(molecule, scratch);
    if (filled != divergence_reach_t::measured_k) return filled;
    std::uint8_t const hops = scratch.hops[first_atom * num_atoms + second_atom];
    if (hops == 0 || hops > widest_labelled_separation_k || bounds_kinds_by_separation_k[hops - 1] != row.kind)
        return divergence_reach_t::topology_does_not_match_kind_k;

    read_bounds_pair_(scratch.rdkit_bounds, num_atoms, first_atom, second_atom, measurement.rdkit_values);
    read_bounds_pair_(scratch.our_bounds, num_atoms, first_atom, second_atom, measurement.our_values);
    return divergence_reach_t::measured_k;
}

/**
 *  @brief The four probe positions of a torsion held at @p dihedral_radians, in the frame the rows use.
 *
 *  The frame makes one coordinate the dihedral: central bond along z, first atom's off-axis component along
 *  x, fourth atom on a unit circle at phi. Its velocity is then exactly `(-sin phi, cos phi, 0)`, turning a
 *  Cartesian gradient into `dE/dphi` with a dot product and no Jacobian to get wrong.
 *  @param[out] positions Twelve doubles, four points of three, as RDKit's force field reads them.
 */
static void fill_torsion_probe_positions_(float dihedral_radians, double (&positions)[12]) {
    double const cos_phi = std::cos(static_cast<double>(dihedral_radians));
    double const sin_phi = std::sin(static_cast<double>(dihedral_radians));
    double const frame[12] = {
        1.0,     0.0,     -0.5, // first atom, off the axis along x
        0.0,     0.0,     0.0,  // second atom, at the origin
        0.0,     0.0,     1.5,  // third atom, so the central bond is the z axis
        cos_phi, sin_phi, 2.0,  // fourth atom, on the unit circle at phi
    };
    for (std::size_t element = 0; element < 12; ++element) positions[element] = frame[element];
}

/** RDKit's own `dE/dphi`, recovered from its Cartesian gradient by moving only the fourth atom. */
static double rdkit_torsion_gradient_(std::vector<double> const &force_constants, std::vector<int> const &signs,
                                      float dihedral_radians) {
    double positions[12] = {};
    fill_torsion_probe_positions_(dihedral_radians, positions);

    RDGeom::Point3D probe_points[4];
    ForceFields::ForceField owner(3);
    for (std::size_t point = 0; point < 4; ++point) {
        probe_points[point] = RDGeom::Point3D(positions[point * 3 + 0], positions[point * 3 + 1],
                                              positions[point * 3 + 2]);
        owner.positions().push_back(&probe_points[point]);
    }

    ForceFields::CrystalFF::TorsionAngleContribs contribs(&owner);
    contribs.addContrib(0, 1, 2, 3, force_constants, signs);
    double gradient[12] = {};
    contribs.getGrad(positions, gradient);

    double const cos_phi = std::cos(static_cast<double>(dihedral_radians));
    double const sin_phi = std::sin(static_cast<double>(dihedral_radians));
    return gradient[9] * -sin_phi + gradient[10] * cos_phi;
}

/** Our own `E(phi) - E(0)` for a six-term cosine series, which is the closed form the kernel sums. */
static double our_torsion_energy_delta_(std::vector<double> const &force_constants, std::vector<int> const &signs,
                                        float dihedral_radians) {
    double accumulated = 0.0;
    for (std::uint32_t harmonic = 1; harmonic <= 6; ++harmonic) {
        double const coefficient = force_constants[harmonic - 1] * static_cast<double>(signs[harmonic - 1]);
        accumulated += coefficient * (std::cos(harmonic * static_cast<double>(dihedral_radians)) - 1.0);
    }
    return accumulated;
}

/** Our own `dE/dphi`, which is the same series differentiated term by term. */
static double our_torsion_gradient_(std::vector<double> const &force_constants, std::vector<int> const &signs,
                                    float dihedral_radians) {
    double accumulated = 0.0;
    for (std::uint32_t harmonic = 1; harmonic <= 6; ++harmonic) {
        double const coefficient = force_constants[harmonic - 1] * static_cast<double>(signs[harmonic - 1]);
        accumulated -= harmonic * coefficient * std::sin(harmonic * static_cast<double>(dihedral_radians));
    }
    return accumulated;
}

/** Whether @p atoms and the four indices @p torsion names are the same chain, in either direction. */
static bool torsion_matches_(std::vector<int> const &torsion, std::uint8_t const (&atoms)[4]) noexcept {
    if (torsion.size() != 4) return false;
    bool forward = true, backward = true;
    for (std::size_t position = 0; position < 4; ++position) {
        forward = forward && torsion[position] == static_cast<int>(atoms[position]);
        backward = backward && torsion[3 - position] == static_cast<int>(atoms[position]);
    }
    return forward || backward;
}

/** Measure one torsion row: RDKit's analytic gradient against ours, with the energy beside it as a control. */
static divergence_reach_t measure_torsion_row_(rdkit_divergence_row_t const &row, RDKit::ROMol const &molecule,
                                               divergence_measurement_t &measurement) {
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    for (std::uint8_t const atom : row.atoms)
        if (atom >= num_atoms) return divergence_reach_t::atom_out_of_range_k;

    ForceFields::CrystalFF::CrystalFFDetails details;
    ForceFields::CrystalFF::getExperimentalTorsions(molecule, details, /*useExpTorsions=*/true,
                                                    /*useSmallRingTorsions=*/true, /*useMacrocycleTorsions=*/true,
                                                    /*useBasicKnowledge=*/true, /*version=*/2, /*verbose=*/false);
    for (std::size_t torsion = 0; torsion < details.expTorsionAtoms.size(); ++torsion) {
        if (!torsion_matches_(details.expTorsionAtoms[torsion], row.atoms)) continue;
        auto const &signs = details.expTorsionAngles[torsion].first;
        auto const &force_constants = details.expTorsionAngles[torsion].second;
        float const dihedral = torsion_probe_dihedral_radians_k;

        // RDKit's energy is the same six-term series ours is, so the energy slot is the control that
        // says the divergence below is in the derivative rather than in the potential.
        double const rdkit_energy = ForceFields::CrystalFF::calcTorsionEnergy(force_constants, signs,
                                                                              std::cos(static_cast<double>(dihedral)));
        double const rdkit_energy_at_zero = ForceFields::CrystalFF::calcTorsionEnergy(force_constants, signs, 1.0);
        measurement.rdkit_values[0] = static_cast<float>(rdkit_energy - rdkit_energy_at_zero);
        measurement.rdkit_values[1] = static_cast<float>(rdkit_torsion_gradient_(force_constants, signs, dihedral));
        measurement.our_values[0] = static_cast<float>(our_torsion_energy_delta_(force_constants, signs, dihedral));
        measurement.our_values[1] = static_cast<float>(our_torsion_gradient_(force_constants, signs, dihedral));
        return divergence_reach_t::measured_k;
    }
    return divergence_reach_t::no_such_torsion_k;
}

/** Measure one row, whatever its kind, on a freshly parsed and hydrogenated molecule. */
static divergence_reach_t measure_row_(rdkit_divergence_row_t const &row, divergence_scratch_t &scratch,
                                       divergence_measurement_t &measurement) {
    std::unique_ptr<RDKit::RWMol> molecule = mol_from_smiles(row.smiles);
    if (!molecule) return divergence_reach_t::unparsable_smiles_k;
    switch (row.kind) {
    case divergence_kind_t::bounds_1_2_k:
    case divergence_kind_t::bounds_1_3_k:
    case divergence_kind_t::bounds_1_4_k: return measure_bounds_row_(row, *molecule, scratch, measurement);
    case divergence_kind_t::torsion_preference_k: return measure_torsion_row_(row, *molecule, measurement);
    // Neither kind has a measurement above, so a row carrying one reads unreachable, not satisfied.
    case divergence_kind_t::chiral_volume_k:
    case divergence_kind_t::smoothing_consistency_k: return divergence_reach_t::no_measurement_for_kind_k;
    }
    return divergence_reach_t::no_measurement_for_kind_k;
}

#pragma endregion Measurement

#pragma region Registry assertions

/**
 *  @brief One reporter per pair of operand roles, because a delta only reads when its sign is fixed.
 *
 *  Each carries the two labels its lines print, so the questions cannot share one: `shape` weighs a row's
 *  declared threshold against a floor before anything is measured, `drift` a recorded value against a measured
 *  one, `disagreement` RDKit against us -- the assertion the row exists to make -- `reach` whether the sides
 *  constrain the same pairs at all before any of their numbers meet, and `completeness` whether a corpus
 *  disagreement is one some row already explains.
 */
struct divergence_reporters_t {
    diagnostic_reporter_t shape {"required", "declared"};
    diagnostic_reporter_t drift {"recorded", "measured"};
    diagnostic_reporter_t disagreement {"rdkit", "ours"};
    diagnostic_reporter_t reach {"expected", "observed"};
    diagnostic_reporter_t completeness {"expected", "observed"};

    /** True when every section held; a caller gates its verdict on this rather than on one reporter. */
    bool passed() const noexcept {
        return shape.passed() && drift.passed() && disagreement.passed() && reach.passed() && completeness.passed();
    }

    /** Divergences asserted across every section, for the verdict line. */
    std::size_t divergence_count() const noexcept {
        return shape.divergence_count() + drift.divergence_count() + disagreement.divergence_count() +
               reach.divergence_count() + completeness.divergence_count();
    }

    /** Mismatches across every section, for the verdict line. */
    std::size_t mismatch_count() const noexcept {
        return shape.mismatch_count() + drift.mismatch_count() + disagreement.mismatch_count() +
               reach.mismatch_count() + completeness.mismatch_count();
    }

    /** Comparisons across every section, so the verdict states what the mismatch count is out of. */
    std::size_t comparison_count() const noexcept {
        return shape.comparison_count() + drift.comparison_count() + disagreement.comparison_count() +
               reach.comparison_count() + completeness.comparison_count();
    }
};

/** The two atom-index columns a row's diagnostic lines carry, as one `key=value` fragment. */
static std::string row_atom_keys_(rdkit_divergence_row_t const &row) {
    if (divergence_kind_traits(row.kind).atom_arity == 4)
        return fmt::format("atoms={},{},{},{}", row.atoms[0], row.atoms[1], row.atoms[2], row.atoms[3]);
    return fmt::format("pair={},{}", row.atoms[0], row.atoms[1]);
}

/** What a registry pass did, so a run that asserted nothing cannot read like a run that asserted everything. */
struct registry_tally_t {
    std::size_t rows = 0;
    std::size_t measured = 0;
    std::size_t predating = 0;
    std::size_t unreachable = 0;
};

/**
 *  @brief Refuse a row whose threshold sits inside the noise its own tolerances admit.
 *
 *  Such a row passes or fails on which way the last bit rounded, and a flapping assertion teaches the next
 *  reader to delete it rather than read it. A row for a pair only one side constrains needs no exemption: it
 *  records the same two bounds as any other, one of them the `unconstrained_upper_k` sentinel, which both
 *  sides spell identically and neither rounds. All three numbers stay distances in angstroms.
 *  @return How many rows were refused.
 */
static std::size_t refuse_flapping_rows_(custd::span<rdkit_divergence_row_t const> rows,
                                         diagnostic_reporter_t &reporter) {
    auto const strictly_greater = [](float noise_band, float threshold) { return threshold > noise_band; };
    auto const strictly_positive = [](float floor, float tolerance) { return tolerance > floor; };
    std::size_t refused = 0;
    for (auto const &row : rows) {
        float const noise_band = row.rdkit_tolerance + row.our_tolerance;
        std::size_t const before = reporter.mismatch_count();
        reporter.mismatch(divergence_kind_traits(row.kind).name, "smiles={} {}", row.smiles, row_atom_keys_(row))
            .field("rdkit_tolerance", 0.0f, row.rdkit_tolerance, strictly_positive)
            .field("our_tolerance", 0.0f, row.our_tolerance, strictly_positive)
            .field("divergence_threshold", noise_band, row.divergence_threshold, strictly_greater);
        if (reporter.mismatch_count() != before) ++refused;
    }
    return refused;
}

/**
 *  @brief Re-measure one row and make the four per-row assertions on it.
 *  @param[in] installed_release_code `YYYYMM` of the linked RDKit.
 *  @param[inout] scratch Bounds and descriptor buffers reused across rows.
 *  @param[inout] tally Counters advanced by every row.
 */
static void check_row_(rdkit_divergence_row_t const &row, std::uint32_t installed_release_code,
                       divergence_scratch_t &scratch, divergence_reporters_t &reporters, registry_tally_t &tally) {
    ++tally.rows;
    divergence_kind_traits_t const traits = divergence_kind_traits(row.kind);
    std::string const keys = row_atom_keys_(row);
    std::string_view const subject = traits.name;

    divergence_expectation_t const expectation = divergence_expectation(row, installed_release_code);
    if (expectation == divergence_expectation_t::predates_installed_release_k) {
        ++tally.predating;
        reporters.drift.note(subject, "smiles={} {} upstream={}", row.smiles, keys, row.upstream)
            .field("release", static_cast<std::uint32_t>(row.since), installed_release_code);
        return;
    }

    divergence_measurement_t measurement;
    divergence_reach_t const reach = measure_row_(row, scratch, measurement);
    if (reach != divergence_reach_t::measured_k) {
        ++tally.unreachable;
        // An unreachable row fails rather than skips: unmeasured reads exactly like held.
        reporters.drift.mismatch(subject, "smiles={} {} reason={}", row.smiles, keys, divergence_reach_reason(reach))
            .field("reachable", 1, 0, exactly_equal);
        return;
    }
    ++tally.measured;

    auto const within = [](float tolerance) {
        return [tolerance](float recorded, float measured) { return !(std::fabs(measured - recorded) > tolerance); };
    };
    auto rdkit_drift = reporters.drift.mismatch(subject, "smiles={} {} side=rdkit upstream={}", row.smiles, keys,
                                                row.upstream);
    auto our_drift = reporters.drift.mismatch(subject, "smiles={} {} side=ours", row.smiles, keys);
    for (std::size_t slot = 0; slot < 2; ++slot) {
        rdkit_drift.field(traits.value_names[slot], row.rdkit_values[slot], measurement.rdkit_values[slot],
                          within(row.rdkit_tolerance));
        our_drift.field(traits.value_names[slot], row.our_values[slot], measurement.our_values[slot],
                        within(row.our_tolerance));
    }

    if (expectation == divergence_expectation_t::fixed_upstream_k) {
        // A retired row: upstream agrees here, so the assertion demands absence of a difference, not presence.
        auto agreement = reporters.disagreement.mismatch(subject, "smiles={} {} fixed_in={}", row.smiles, keys,
                                                         static_cast<std::uint32_t>(row.fixed_in));
        for (std::size_t slot = 0; slot < 2; ++slot)
            agreement.field(traits.value_names[slot], measurement.rdkit_values[slot], measurement.our_values[slot],
                            within(row.rdkit_tolerance + row.our_tolerance));
        return;
    }

    // Every slot prints: those apart are the assertion and carry the citation, those agreeing are the control.
    auto divergent = reporters.disagreement.divergence(subject, "smiles={} {} upstream={}", row.smiles, keys,
                                                       row.upstream);
    auto agreeing = reporters.disagreement.note(subject, "smiles={} {} agrees=1", row.smiles, keys);
    std::size_t apart = 0;
    for (std::size_t slot = 0; slot < 2; ++slot) {
        float const deviation = std::fabs(measurement.our_values[slot] - measurement.rdkit_values[slot]);
        if (deviation > row.divergence_threshold) {
            ++apart;
            divergent.field(traits.value_names[slot], measurement.rdkit_values[slot], measurement.our_values[slot]);
        }
        else agreeing.field(traits.value_names[slot], measurement.rdkit_values[slot], measurement.our_values[slot]);
    }
    auto const at_least = [](std::size_t minimum, std::size_t actual) { return actual >= minimum; };
    reporters.drift.mismatch(subject, "smiles={} {} threshold={:.3g}", row.smiles, keys, row.divergence_threshold)
        .field("divergent_slots", std::size_t {1}, apart, at_least);
}

#pragma endregion Registry assertions

#pragma region Completeness

/**
 *  @brief Widest a determined bound may be before the pair counts as bracketed rather than pinned.
 *
 *  Our builder is approximate, so a pair both sides merely bracket -- a freely rotating 1-4, whose upper bound
 *  we flatten to 4.5 angstroms -- differs by modelling slack, not disagreement. Only pairs both sides pin are
 *  comparable, and a pinned bound is a percentage tolerance on one computed distance, so its spread stays small.
 */
static constexpr float determined_spread_k = 0.6f;

/**
 *  @brief The upper bound a side writes when it declines to constrain a pair at all.
 *
 *  Both builders start every pair at the same literal -- ours `bounds_matrix_builder.hpp`'s `upper_initial_k`,
 *  RDKit's `MAX_UPPER` -- so "never written" is one comparison meaning the same on both sides, and no real
 *  1-2, 1-3 or 1-4 distance approaches it.
 */
static constexpr float unconstrained_upper_k = 1000.0f;

/**
 *  @brief How far apart two sides' interval widths may sit before one is constraining what the other is not.
 *
 *  The reaches below name this readably but cannot be the assertion: they are a cliff at
 *  `determined_spread_k`, and a PubChem sweep puts a real population of 1-4 windows within 0.03 angstrom of
 *  it, so comparing labels would report a divergence on nothing but which side each landed. The width has no
 *  cliff and its distribution leaves an empty band: over 19,555 PubChem molecules and 5,180,376 pairs the
 *  differences run continuously to 0.70, then nothing until 1.24 and a long tail past it. One angstrom sits
 *  mid-band and is the scale the band is made of -- a side allowing a full cis-to-trans inversion more freedom
 *  than the other describes different chemistry, not the same chemistry more loosely. A smaller corpus shows a
 *  wider band and would justify a tighter constant, hence the larger sweep. The widest difference actually
 *  called agreement prints at the end of every run.
 */
static constexpr float spread_agreement_k = 1.0f;

/** Whether two sides constrain a pair to comparably wide intervals, whatever the two intervals sit at. */
inline constexpr auto spreads_agree = [](float rdkit_spread, float our_spread) {
    return !(std::fabs(our_spread - rdkit_spread) > spread_agreement_k);
};

/**
 *  @brief How much of a pair one side claims, which decides whether its number compares to the other's.
 *
 *  Reported rather than asserted -- see `spread_agreement_k`. A pinned 1-4 window is about 0.12 angstrom where
 *  a freely rotating one spans the full cis-to-trans inversion at 1.0 and up, and `determined_spread_k`
 *  separates them well enough to decide comparability, which is all it decides.
 */
enum class constraint_reach_t : std::uint8_t {
    /** The upper bound is still the sentinel, so this side put no constraint on the pair whatsoever. */
    declined_k,
    /** Constrained, but to a range rather than a distance -- modelling slack, not a claim about geometry. */
    bracketed_k,
    /** Constrained inside `determined_spread_k`, which is the only state in which two bounds compare. */
    pinned_k,
};

/** The literal a reach prints as, which is what a diagnostic line carries instead of a raw byte. */
static std::string_view constraint_reach_name(constraint_reach_t reach) noexcept {
    switch (reach) {
    case constraint_reach_t::declined_k: return "declined";
    case constraint_reach_t::bracketed_k: return "bracketed";
    case constraint_reach_t::pinned_k: return "pinned";
    }
    return "unrecorded";
}

/** Sort one side's `{lower, upper}` into the three reaches; the upper alone separates the first two. */
static constraint_reach_t constraint_reach_of_(float const (&values)[2]) noexcept {
    if (values[1] >= unconstrained_upper_k) return constraint_reach_t::declined_k;
    return values[1] - values[0] > determined_spread_k ? constraint_reach_t::bracketed_k
                                                       : constraint_reach_t::pinned_k;
}

/**
 *  @brief How far two pinned bounds must differ before the rescan calls it a candidate divergence.
 *
 *  The rescan's sensitivity floor, set to clear the residual the two builders' ideal-geometry tables leave on
 *  the built-in corpus while sitting well under the roughly one-angstrom cis-against-trans inversion a
 *  registry row records. The widest deviation it lets through prints at the end of every run rather than being
 *  asserted here, so the headroom is a number a reader can watch shrink. Lowering it finds the next divergence
 *  of this class; raising it hides one, which is the move this suite exists to make unnecessary.
 */
static constexpr float scan_threshold_k = 0.5f;

/** Whether some registry row already explains this molecule's pair at this kind. */
static bool covered_by_registry_(custd::span<rdkit_divergence_row_t const> rows, std::string const &smiles,
                                 divergence_kind_t kind, std::uint32_t first_atom, std::uint32_t second_atom) {
    for (auto const &row : rows) {
        if (row.kind != kind || smiles != row.smiles) continue;
        bool const forward = row.atoms[0] == first_atom && row.atoms[1] == second_atom;
        bool const backward = row.atoms[0] == second_atom && row.atoms[1] == first_atom;
        if (forward || backward) return true;
    }
    return false;
}

/** Print the row a maintainer would paste into the table, with the two fields only a human can fill. */
static void print_candidate_row_(std::string const &smiles, divergence_kind_t kind, std::uint32_t first_atom,
                                 std::uint32_t second_atom, divergence_measurement_t const &measurement,
                                 std::uint32_t installed_release_code) {
    fmt::println("    {{.smiles = \"{}\",", smiles);
    fmt::println("     .kind = divergence_kind_t::{}_k,", divergence_kind_traits(kind).name);
    fmt::println("     .atoms = {{{}, {}, 0, 0}},", first_atom, second_atom);
    fmt::println("     .rdkit_values = {{{}f, {}f}},", measurement.rdkit_values[0], measurement.rdkit_values[1]);
    fmt::println("     .our_values = {{{}f, {}f}},", measurement.our_values[0], measurement.our_values[1]);
    fmt::println("     .rdkit_tolerance = 0.01f,");
    fmt::println("     .our_tolerance = 0.01f,");
    fmt::println("     .divergence_threshold = {:.3g}f,", scan_threshold_k);
    fmt::println("     .upstream = \"TODO\",");
    fmt::println("     .note = \"TODO\",");
    fmt::println("     .since = rdkit_release_t::release_{}_{:02}_k,", installed_release_code / 100,
                 installed_release_code % 100);
    fmt::println("     .fixed_in = rdkit_release_t::not_yet_fixed_k}},");
}

/**
 *  What one rescan of the corpus looked at, so a scan that compared nothing cannot read like a clean one.
 *  `pairs` counts every 1-2, 1-3 and 1-4 pair whatever either side did with it, `pinned_pairs` only those both
 *  sides pinned -- the sole ones whose numbers compare -- and `one_sided` those the two reach differently, one
 *  asserting a geometry the other declines to; `unlisted_one_sided` splits the unlisted the same way. The two
 *  `widest_*` fields are the headroom the rescan floor and the reach floor still have.
 */
struct completeness_tally_t {
    std::size_t molecules = 0;
    std::size_t skipped = 0;
    std::size_t pairs = 0;
    std::size_t pinned_pairs = 0;
    std::size_t one_sided = 0;
    std::size_t unlisted = 0;
    std::size_t unlisted_one_sided = 0;
    float widest_within_floor = 0.0f;
    float widest_spread_within_floor = 0.0f;
};

/**
 *  @brief Rescan one molecule for pairwise bounds disagreements no registry row explains.
 *
 *  Every pair is looked at twice and both looks count. The first asks whether the two sides reach the pair
 *  alike, which comparing their numbers presupposes; the second compares the numbers and runs only where both
 *  pinned. A pair excluded from the second is still carried by the first, which stops a whole class of
 *  disagreement -- every pair where one side asserts a geometry the other declines -- being counted nowhere.
 *  @param[in] installed_release_code `YYYYMM` of the linked RDKit, for the printed candidate row.
 *  @param[inout] scratch Bounds and descriptor buffers reused across the corpus, grown at first use.
 *  @param[inout] tally Counters advanced by every molecule and every pair.
 */
static void rescan_molecule_(std::string const &smiles, custd::span<rdkit_divergence_row_t const> rows,
                             std::uint32_t installed_release_code, divergence_scratch_t &scratch,
                             divergence_reporters_t &reporters, completeness_tally_t &tally) {
    std::unique_ptr<RDKit::RWMol> molecule = mol_from_smiles(smiles);
    if (!molecule || molecule->getNumAtoms() > sft::max_atoms_per_molecule_k) {
        ++tally.skipped;
        return;
    }
    if (fill_bounds_scratch_(*molecule, scratch) != divergence_reach_t::measured_k) {
        ++tally.skipped;
        return;
    }
    ++tally.molecules;

    std::uint32_t const num_atoms = molecule->getNumAtoms();
    for (std::uint32_t first_atom = 0; first_atom < num_atoms; ++first_atom) {
        for (std::uint32_t second_atom = first_atom + 1; second_atom < num_atoms; ++second_atom) {
            std::uint8_t const hops = scratch.hops[first_atom * num_atoms + second_atom];
            // Beyond 1-4 both sides only bracket the pair, so there is nothing to compare.
            if (hops == 0 || hops > widest_labelled_separation_k) continue;

            divergence_measurement_t measurement;
            read_bounds_pair_(scratch.rdkit_bounds, num_atoms, first_atom, second_atom, measurement.rdkit_values);
            read_bounds_pair_(scratch.our_bounds, num_atoms, first_atom, second_atom, measurement.our_values);
            ++tally.pairs;

            divergence_kind_t const kind = bounds_kinds_by_separation_k[hops - 1];
            std::string_view const subject = divergence_kind_traits(kind).name;
            bool const listed = covered_by_registry_(rows, smiles, kind, first_atom, second_atom);

            // How far each side reached comes before what it said, since two numbers compare only where both
            // produced one. The widths are the operands, so the predicate compares what the line prints and no
            // cliff decides it; the reach names ride the key columns so a line reads without recomputing them.
            constraint_reach_t const rdkit_reach = constraint_reach_of_(measurement.rdkit_values);
            constraint_reach_t const our_reach = constraint_reach_of_(measurement.our_values);
            float const rdkit_spread = measurement.rdkit_values[1] - measurement.rdkit_values[0];
            float const our_spread = measurement.our_values[1] - measurement.our_values[0];
            bool const reaches_alike = spreads_agree(rdkit_spread, our_spread);
            if (!reaches_alike) ++tally.one_sided;
            reporters.reach
                .mismatch(subject, "smiles={} pair={},{} rdkit_reach={} our_reach={}", smiles, first_atom, second_atom,
                          constraint_reach_name(rdkit_reach), constraint_reach_name(our_reach))
                .field("interval_spread", rdkit_spread, our_spread, [listed](float reference, float observed) {
                    return spreads_agree(reference, observed) || listed;
                });
            float const spread_gap = std::fabs(our_spread - rdkit_spread);
            if (reaches_alike && spread_gap > tally.widest_spread_within_floor)
                tally.widest_spread_within_floor = spread_gap;
            if (!reaches_alike && !listed) {
                ++tally.unlisted;
                ++tally.unlisted_one_sided;
                fmt::println("Unlisted {} one-sided pair on {} between atoms {} and {}: rdkit {}, ours {}; "
                             "the row it would take:",
                             subject, smiles, first_atom, second_atom, constraint_reach_name(rdkit_reach),
                             constraint_reach_name(our_reach));
                print_candidate_row_(smiles, kind, first_atom, second_atom, measurement, installed_release_code);
            }
            // A pair either side left as a range carries slack, not a claim, so the section below skips it.
            if (rdkit_reach != constraint_reach_t::pinned_k || our_reach != constraint_reach_t::pinned_k) continue;

            ++tally.pinned_pairs;
            float widest = 0.0f;
            for (std::size_t slot = 0; slot < 2; ++slot) {
                float const deviation = std::fabs(measurement.our_values[slot] - measurement.rdkit_values[slot]);
                if (deviation > widest) widest = deviation;
            }

            // Every pinned pair counts as a comparison: looking at nothing must not read like finding nothing.
            bool const explained = widest <= scan_threshold_k || listed;
            reporters.completeness
                .mismatch(subject, "smiles={} pair={},{} widest={:.3g}", smiles, first_atom, second_atom, widest)
                .field("explained", true, explained, exactly_equal);
            if (explained) {
                if (widest <= scan_threshold_k && widest > tally.widest_within_floor)
                    tally.widest_within_floor = widest;
                continue;
            }

            ++tally.unlisted;
            fmt::println("Unlisted {} disagreement on {} between atoms {} and {}; the row it would take:", subject,
                         smiles, first_atom, second_atom);
            print_candidate_row_(smiles, kind, first_atom, second_atom, measurement, installed_release_code);
        }
    }
}

#pragma endregion Completeness

#pragma region Narration

/**
 *  @brief Report how much of the small-ring corpus the six-fold gradient defect silently disarms.
 *
 *  Narration rather than assertion, but measured every run rather than written into a comment, because a
 *  number a comment states is a number nothing re-checks.
 */
static void narrate_sixfold_reach_() {
    std::size_t with_sixfold = 0, of_which_fifth_is_zero = 0;
    for (auto const &pattern : sfp::csd_torsion_rows_small_rings()) {
        if (pattern.magnitudes[5] == 0.0f) continue;
        ++with_sixfold;
        if (pattern.magnitudes[4] == 0.0f) ++of_which_fifth_is_zero;
    }
    fmt::println("Small-ring patterns: {} in the corpus, {} carry a six-fold term, {} of those carry no fifth one",
                 sfp::csd_torsion_rows_small_rings().size(), with_sixfold, of_which_fifth_is_zero);
}

#pragma endregion Narration

} // namespace smallfield::tests

int main(int argc, char **argv) {
    fmt::println("{}: recorded RDKit divergences, re-measured and asserted", test_name_k);

    auto const selection = sftest::select_corpus(argc, argv, default_max_molecules_k,
                                                 sftest::adversarial_corpus_path());
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", argv[0]);
        return 2;
    }

    std::uint32_t const installed_release_code = sftest::rdkit_release_code_from_version(RDKit::rdkitVersion);
    fmt::println("RDKit: {} (release code {})", RDKit::rdkitVersion, installed_release_code);
    if (installed_release_code == 0)
        return sftest::report_fail(test_name_k, "cannot read a YYYYMM release out of {}", RDKit::rdkitVersion);

    auto const rows = sftest::rdkit_divergence_rows();
    sftest::divergence_reporters_t reporters;
    reporters.completeness.limit_printed_lines(60); // one pathological molecule must not bury the rest
    reporters.reach.limit_printed_lines(60);

    std::size_t const refused = sftest::refuse_flapping_rows_(rows, reporters.shape);
    reporters.shape.summarize("row_shape");
    sf_verify_(refused == 0, "a row whose divergence threshold sits inside its own tolerances would flap");

    sftest::narrate_sixfold_reach_();

    // Declared above every loop below so the bounds matrices grow once for the whole run.
    sftest::divergence_scratch_t scratch;
    sftest::registry_tally_t registry;
    for (auto const &row : rows) sftest::check_row_(row, installed_release_code, scratch, reporters, registry);
    reporters.drift.summarize("recorded_values");
    reporters.disagreement.summarize("divergence");
    fmt::println("Registry: {} rows, {} measured, {} unreachable, {} predating the installed release", registry.rows,
                 registry.measured, registry.unreachable, registry.predating);

    // The registry's own molecules always rescan, so no row describes what the completeness pass would miss.
    std::vector<std::string> corpus = sftest::standard_test_smiles();
    for (auto const &row : rows) corpus.emplace_back(row.smiles);
    std::string const &corpus_path = selection.path;
    std::vector<std::string> loaded;
    auto const outcome = sftest::load_smiles_into(corpus_path, selection.max_molecules, loaded);
    if (outcome != sftest::smiles_load_t::loaded_k)
        return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), corpus_path);
    fmt::println("Corpus: {} ({} molecules, capped at {}) plus the built-ins and the registry's own", corpus_path,
                 fmt::group_digits(loaded.size()), fmt::group_digits(selection.max_molecules));
    for (auto &smiles : loaded) corpus.push_back(std::move(smiles));

    sftest::completeness_tally_t completeness;
    for (auto const &smiles : corpus)
        sftest::rescan_molecule_(smiles, rows, installed_release_code, scratch, reporters, completeness);
    reporters.reach.summarize("constraint_reach");
    reporters.completeness.summarize("completeness");
    fmt::println("Completeness: {} molecules ({} skipped), {} pairs of which {} pinned on both sides, " //
                 "{} one-sided, {} unlisted ({} of them one-sided), widest below the rescan floor {:.3g} " //
                 "and below the reach floor {:.3g}",
                 fmt::group_digits(completeness.molecules), completeness.skipped, fmt::group_digits(completeness.pairs),
                 fmt::group_digits(completeness.pinned_pairs), fmt::group_digits(completeness.one_sided),
                 completeness.unlisted, completeness.unlisted_one_sided, completeness.widest_within_floor,
                 completeness.widest_spread_within_floor);

    if (!reporters.passed())
        return sftest::report_fail(test_name_k, "{} mismatches over {} comparisons, {} divergences asserted",
                                   reporters.mismatch_count(), reporters.comparison_count(),
                                   reporters.divergence_count());
    return sftest::report_pass(test_name_k);
}
