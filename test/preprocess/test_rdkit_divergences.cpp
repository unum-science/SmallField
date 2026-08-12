/**
 *  @file test/preprocess/test_rdkit_divergences.cpp
 *  @brief Assert every recorded disagreement with RDKit that we believe RDKit is wrong about.
 *
 *  A parity test that meets a difference it cannot explain has one move: widen the tolerance. Where
 *  the difference is arithmetic noise that is right, and where the reference is the mistaken side it is
 *  a silent surrender -- the widened tolerance now also admits every future regression of that size, so
 *  the number stops meaning anything and nobody can tell which of the two it was written for.
 *
 *  This suite is the other move. `rdkit_divergence_table.cpp` names each upstream defect once and
 *  records what both implementations say about it; this binary re-measures all of it and makes five
 *  assertions per run:
 *
 *   - @b Reachable: the SMILES parses and the named atoms exist. A row nobody can measure is a failure,
 *     never a skip, because an unmeasurable row is indistinguishable from a satisfied one.
 *   - @b Reference-still-says-it: RDKit reproduces `rdkit_values` within `rdkit_tolerance`. This is the
 *     upgrade tripwire, and it fires whether upstream fixed the defect or merely moved it.
 *   - @b We-still-say-it: SmallField reproduces `our_values` within `our_tolerance`.
 *   - @b Still-apart: the two remain more than `divergence_threshold` apart. Convergence retires a row
 *     rather than relaxing it.
 *   - @b Complete: a rescan of the corpus finds no disagreement of a recorded kind that no row explains.
 *
 *  When `fixed_in` names a release at or below the installed one, the fourth assertion inverts: the two
 *  must now @b agree, so a retired divergence stays in the table as a permanent regression test instead
 *  of being deleted along with the reason it existed.
 *
 *  The rescan covers the pairwise bounds kinds, which are the ones a molecule enumerates. A torsion
 *  preference is not one of them: its divergence is a property of a Fourier coefficient set that 37 of
 *  the 105 small-ring patterns share, so a per-molecule listing would be the same fact copied until the
 *  table stopped being readable. Its row carries the defect once, and the rescan says so rather than
 *  pretending to have covered it.
 *
 *  Corpus: the built-in standards plus every molecule the registry names, so the registry is always
 *  rescanned against itself; `argv[1]` or `SMALLFIELD_TEST_CORPUS` widens it.
 */

#include <cmath>   // `std::fabs`, `std::cos`, `std::sin`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::uint32_t`

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

/** Molecules taken from a named corpus when no cap is given; the registry's own are always walked. */
static constexpr std::size_t default_max_molecules_k = 50;

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
    default: return "no measurement for that kind";
    }
}

/** Both sides of one row, as measured now. */
struct divergence_measurement_t {
    float rdkit_values[2] = {0.0f, 0.0f};
    float our_values[2] = {0.0f, 0.0f};
};

/** Buffers the sweep reuses, declared once above every loop so a corpus pass grows them only at first use. */
struct divergence_scratch_t {
    /** Our own topological bounds, N × N row-major: upper above the diagonal, lower below. */
    std::vector<float> our_bounds;
    /** RDKit's topological bounds in the same layout, so one index expression reads both. */
    std::vector<double> rdkit_bounds;
    /** Bond separation of every pair, which is what sorts a pair into a 1-2, 1-3 or 1-4 kind. */
    std::vector<std::uint8_t> hops;
    sft::molecule_descriptor_t descriptor;
};

/**
 *  @brief Flatten RDKit's topological bounds for @p molecule into @p flat, N × N row-major.
 *
 *  The layout is ours -- upper bound above the diagonal, lower below -- so the pair lookups below read
 *  both matrices with one index expression rather than two conventions a reader has to keep apart.
 *
 *  @param[in] molecule Molecule to derive bounds for; hydrogens must already be explicit.
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

/** How many bonds apart a pair must be for each bounds kind; anything else is a mislabelled row. */
static std::uint8_t bounds_kind_separation_(divergence_kind_t kind) noexcept {
    switch (kind) {
    case divergence_kind_t::bounds_1_2_k: return 1;
    case divergence_kind_t::bounds_1_3_k: return 2;
    default: return 3;
    }
}

/**
 *  @brief Fill both bounds matrices and the hop matrix for @p molecule.
 *
 *  @param[in] molecule Molecule with explicit hydrogens.
 *  @param[inout] scratch Buffers to fill; grown at most once per run.
 *  @return Whether our own descriptor could be built, which is the only way this fails.
 */
static divergence_reach_t fill_bounds_scratch_(RDKit::ROMol const &molecule, divergence_scratch_t &scratch) {
    if (sft::build_descriptor(molecule, scratch.descriptor) != api::conformer_status_t::success_k)
        return divergence_reach_t::descriptor_refused_k;

    std::uint32_t const num_atoms = molecule.getNumAtoms();
    fill_rdkit_bounds_into_(molecule, scratch.rdkit_bounds);
    sfp::build_bounds_matrix(scratch.descriptor, scratch.our_bounds);
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
    if (scratch.hops[first_atom * num_atoms + second_atom] != bounds_kind_separation_(row.kind))
        return divergence_reach_t::topology_does_not_match_kind_k;

    read_bounds_pair_(scratch.rdkit_bounds, num_atoms, first_atom, second_atom, measurement.rdkit_values);
    read_bounds_pair_(scratch.our_bounds, num_atoms, first_atom, second_atom, measurement.our_values);
    return divergence_reach_t::measured_k;
}

/**
 *  @brief The four probe positions of a torsion held at @p dihedral_radians, in the frame the rows use.
 *
 *  The frame is chosen so that one coordinate is the dihedral: the central bond lies along z, the first
 *  atom's off-axis component points along x, and the fourth atom rides a unit circle at angle phi. The
 *  fourth atom's velocity is then exactly `(-sin phi, cos phi, 0)`, which is what turns a Cartesian
 *  gradient into `dE/dphi` with a dot product and no Jacobian to get wrong.
 *
 *  @param[in] dihedral_radians Dihedral to hold the chain at.
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
    default: return divergence_reach_t::no_measurement_for_kind_k;
    }
}

#pragma endregion Measurement

#pragma region Registry assertions

/**
 *  @brief One reporter per pair of operand roles, because a delta only reads when its sign is fixed.
 *
 *  A reporter is constructed with the two labels its lines carry, so the four questions this suite asks
 *  cannot share one: "does the row declare a usable threshold" compares a floor against a declaration,
 *  "has either side drifted" compares what was recorded against what was just measured, "are they still
 *  apart" compares RDKit against us, and the rescan compares an expectation against an observation.
 */
struct divergence_reporters_t {
    /** Row shape, checked before anything is measured. */
    diagnostic_reporter_t shape {"required", "declared"};
    /** Each side against the value the row recorded for it. */
    diagnostic_reporter_t drift {"recorded", "measured"};
    /** The two sides against each other, which is the assertion the row exists to make. */
    diagnostic_reporter_t disagreement {"rdkit", "ours"};
    /** Whether a corpus disagreement is one some row already explains. */
    diagnostic_reporter_t completeness {"expected", "observed"};

    /** True when every section held; a caller gates its verdict on this rather than on one reporter. */
    bool passed() const noexcept {
        return shape.passed() && drift.passed() && disagreement.passed() && completeness.passed();
    }

    /** Divergences asserted across every section, for the verdict line. */
    std::size_t divergence_count() const noexcept { return disagreement.divergence_count(); }

    /** Mismatches across every section, for the verdict line. */
    std::size_t mismatch_count() const noexcept {
        return shape.mismatch_count() + drift.mismatch_count() + disagreement.mismatch_count() +
               completeness.mismatch_count();
    }

    /** Comparisons across every section, so the verdict states what the mismatch count is out of. */
    std::size_t comparison_count() const noexcept {
        return shape.comparison_count() + drift.comparison_count() + disagreement.comparison_count() +
               completeness.comparison_count();
    }
};

/** The two atom-index columns a row's diagnostic lines carry, as one `key=value` fragment. */
static std::string row_atom_keys_(rdkit_divergence_row_t const &row) {
    if (divergence_atom_arity(row.kind) == 4)
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
 *  Such a row passes or fails on which way the last bit rounded, and a flapping assertion teaches the
 *  next reader to delete it rather than to read it.
 *
 *  @param[in] rows Whole registry.
 *  @param[inout] reporter Shape reporter the refusals are printed through.
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
        reporter.mismatch(divergence_kind_name(row.kind), "smiles={} {}", row.smiles, row_atom_keys_(row))
            .field("rdkit_tolerance", 0.0f, row.rdkit_tolerance, strictly_positive)
            .field("our_tolerance", 0.0f, row.our_tolerance, strictly_positive)
            .field("divergence_threshold", noise_band, row.divergence_threshold, strictly_greater);
        if (reporter.mismatch_count() != before) ++refused;
    }
    return refused;
}

/**
 *  @brief Re-measure one row and make the four per-row assertions on it.
 *
 *  @param[in] row Registry row under test.
 *  @param[in] installed_release_code `YYYYMM` of the linked RDKit.
 *  @param[inout] scratch Bounds and descriptor buffers reused across rows.
 *  @param[inout] reporter Reporter the mismatches and divergences are printed through.
 *  @param[inout] tally Counters advanced by every row.
 */
static void check_row_(rdkit_divergence_row_t const &row, std::uint32_t installed_release_code,
                       divergence_scratch_t &scratch, divergence_reporters_t &reporters, registry_tally_t &tally) {
    ++tally.rows;
    std::string const keys = row_atom_keys_(row);
    std::string_view const subject = divergence_kind_name(row.kind);

    divergence_expectation_t const expectation = divergence_expectation(row, installed_release_code);
    if (expectation == divergence_expectation_t::predates_installed_release_k) {
        ++tally.predating;
        reporters.drift.note(subject, "smiles={} {} upstream={}", row.smiles, keys, row.upstream)
            .field("release", rdkit_release_code(row.since), installed_release_code);
        return;
    }

    divergence_measurement_t measurement;
    divergence_reach_t const reach = measure_row_(row, scratch, measurement);
    if (reach != divergence_reach_t::measured_k) {
        ++tally.unreachable;
        // An unreachable row is a failed comparison and not a skipped one: a row nobody can measure
        // reads exactly like a row that holds, and the whole point of the registry is that it cannot.
        reporters.drift.mismatch(subject, "smiles={} {} reason={}", row.smiles, keys, divergence_reach_reason(reach))
            .field("reachable", 1, 0, exactly_equal);
        return;
    }
    ++tally.measured;

    divergence_value_names_t const names = divergence_value_names(row.kind);
    char const *const slot_names[2] = {names.first, names.second};

    auto const within = [](float tolerance) {
        return [tolerance](float recorded, float measured) { return !(std::fabs(measured - recorded) > tolerance); };
    };
    auto rdkit_drift = reporters.drift.mismatch(subject, "smiles={} {} side=rdkit upstream={}", row.smiles, keys,
                                                row.upstream);
    auto our_drift = reporters.drift.mismatch(subject, "smiles={} {} side=ours", row.smiles, keys);
    for (std::size_t slot = 0; slot < 2; ++slot) {
        rdkit_drift.field(slot_names[slot], row.rdkit_values[slot], measurement.rdkit_values[slot],
                          within(row.rdkit_tolerance));
        our_drift.field(slot_names[slot], row.our_values[slot], measurement.our_values[slot],
                        within(row.our_tolerance));
    }

    if (expectation == divergence_expectation_t::fixed_upstream_k) {
        // The row is retired: upstream now agrees, and the assertion that used to demand a difference
        // demands its absence instead, so the fix cannot silently regress.
        auto agreement = reporters.disagreement.mismatch(subject, "smiles={} {} fixed_in={}", row.smiles, keys,
                                                         rdkit_release_code(row.fixed_in));
        for (std::size_t slot = 0; slot < 2; ++slot)
            agreement.field(slot_names[slot], measurement.rdkit_values[slot], measurement.our_values[slot],
                            within(row.rdkit_tolerance + row.our_tolerance));
        return;
    }

    // Every slot is printed. The ones still apart are the assertion and carry the citation; the ones
    // that agree are the control that says which half of the reference moved and which did not.
    auto divergent = reporters.disagreement.divergence(subject, "smiles={} {} upstream={}", row.smiles, keys,
                                                       row.upstream);
    auto agreeing = reporters.disagreement.note(subject, "smiles={} {} agrees=1", row.smiles, keys);
    std::size_t apart = 0;
    for (std::size_t slot = 0; slot < 2; ++slot) {
        float const deviation = std::fabs(measurement.our_values[slot] - measurement.rdkit_values[slot]);
        if (deviation > row.divergence_threshold) {
            ++apart;
            divergent.field(slot_names[slot], measurement.rdkit_values[slot], measurement.our_values[slot]);
        }
        else agreeing.field(slot_names[slot], measurement.rdkit_values[slot], measurement.our_values[slot]);
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
 *  Our builder is documented as approximate, so a pair both sides merely bracket -- a freely rotating
 *  1-4, whose upper bound we flatten to 4.5 angstroms -- carries a difference that is modelling slack
 *  rather than disagreement. Only pairs both sides pin are comparable at all, and a pinned bound comes
 *  out of a percentage tolerance on one computed distance, so its spread stays small.
 */
static constexpr float determined_spread_k = 0.6f;

/**
 *  @brief How far two pinned bounds must differ before the rescan calls it a candidate divergence.
 *
 *  This is the rescan's sensitivity floor, chosen to clear the residual the two builders' ideal-geometry
 *  tables leave on the built-in corpus while sitting well under the roughly one-angstrom
 *  cis-against-trans inversion a registry row records. The widest deviation it let through is printed
 *  at the end of every run rather than asserted in this sentence, so the headroom is a number a reader
 *  can watch shrink. Lowering it is how the next divergence of this class gets found; raising it is how
 *  one gets hidden, which is the move this whole suite exists to make unnecessary.
 */
static constexpr float scan_threshold_k = 0.5f;

/** The kind a bond separation sorts a pair into. */
static divergence_kind_t bounds_kind_for_separation_(std::uint8_t hops) noexcept {
    switch (hops) {
    case 1: return divergence_kind_t::bounds_1_2_k;
    case 2: return divergence_kind_t::bounds_1_3_k;
    default: return divergence_kind_t::bounds_1_4_k;
    }
}

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
    fmt::println("     .kind = divergence_kind_t::{}_k,", divergence_kind_name(kind));
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

/** What one rescan of the corpus looked at, so a scan that compared nothing cannot read like a clean one. */
struct completeness_tally_t {
    std::size_t molecules = 0;
    std::size_t skipped = 0;
    std::size_t pairs = 0;
    std::size_t unlisted = 0;
    /** Widest deviation the rescan let through, which is what says whether its floor is still right. */
    float widest_within_floor = 0.0f;
};

/**
 *  @brief Rescan one molecule for pairwise bounds disagreements no registry row explains.
 *
 *  @param[in] smiles Molecule to rescan.
 *  @param[in] rows Registry to check candidates against.
 *  @param[in] installed_release_code `YYYYMM` of the linked RDKit, for the printed candidate row.
 *  @param[inout] scratch Bounds and descriptor buffers reused across the corpus.
 *  @param[inout] reporter Reporter the unlisted candidates are printed through.
 *  @param[inout] tally Counters advanced by every molecule and every pair.
 */
static void rescan_molecule_(std::string const &smiles, custd::span<rdkit_divergence_row_t const> rows,
                             std::uint32_t installed_release_code, divergence_scratch_t &scratch,
                             diagnostic_reporter_t &reporter, completeness_tally_t &tally) {
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
            if (hops == 0 || hops > 3) continue; // beyond 1-4 both sides only bracket, so nothing to compare

            divergence_measurement_t measurement;
            read_bounds_pair_(scratch.rdkit_bounds, num_atoms, first_atom, second_atom, measurement.rdkit_values);
            read_bounds_pair_(scratch.our_bounds, num_atoms, first_atom, second_atom, measurement.our_values);
            if (measurement.rdkit_values[1] - measurement.rdkit_values[0] > determined_spread_k) continue;
            if (measurement.our_values[1] - measurement.our_values[0] > determined_spread_k) continue;

            ++tally.pairs;
            float widest = 0.0f;
            for (std::size_t slot = 0; slot < 2; ++slot) {
                float const deviation = std::fabs(measurement.our_values[slot] - measurement.rdkit_values[slot]);
                if (deviation > widest) widest = deviation;
            }
            divergence_kind_t const kind = bounds_kind_for_separation_(hops);

            // Every determined pair is a comparison, whether or not it disagrees: a rescan that
            // silently looked at nothing and a rescan that found nothing must not read alike.
            bool const explained = widest <= scan_threshold_k ||
                                   covered_by_registry_(rows, smiles, kind, first_atom, second_atom);
            reporter
                .mismatch(divergence_kind_name(kind), "smiles={} pair={},{} widest={:.3g}", smiles, first_atom,
                          second_atom, widest)
                .field("explained", true, explained, exactly_equal);
            if (explained) {
                if (widest <= scan_threshold_k && widest > tally.widest_within_floor)
                    tally.widest_within_floor = widest;
                continue;
            }

            ++tally.unlisted;
            fmt::println("Unlisted {} disagreement on {} between atoms {} and {}; the row it would take:",
                         divergence_kind_name(kind), smiles, first_atom, second_atom);
            print_candidate_row_(smiles, kind, first_atom, second_atom, measurement, installed_release_code);
        }
    }
}

#pragma endregion Completeness

#pragma region Narration

/**
 *  @brief Report how much of the small-ring corpus the six-fold gradient defect silently disarms.
 *
 *  The count is narration rather than an assertion, but it is measured on every run rather than written
 *  into a comment, because a number a comment states is a number nothing re-checks.
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

    auto const selection = sftest::select_corpus(argc, argv, default_max_molecules_k);
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

    // The registry's own molecules are always rescanned, so a row can never describe a disagreement the
    // completeness pass would not have found on its own.
    std::vector<std::string> corpus = sftest::standard_test_smiles();
    for (auto const &row : rows) corpus.emplace_back(row.smiles);
    if (selection.knob == sftest::corpus_knob_t::named_k) {
        std::vector<std::string> loaded;
        auto const outcome = sftest::load_smiles_into(selection.path, selection.max_molecules, loaded);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), selection.path);
        fmt::println("Corpus: {} ({} molecules) plus the built-ins and the registry's own", selection.path,
                     fmt::group_digits(loaded.size()));
        for (auto &smiles : loaded) corpus.push_back(std::move(smiles));
    }
    else fmt::println("Corpus: built-ins and the registry's own; name one in SMALLFIELD_TEST_CORPUS to widen it");

    sftest::completeness_tally_t completeness;
    for (auto const &smiles : corpus)
        sftest::rescan_molecule_(smiles, rows, installed_release_code, scratch, reporters.completeness, completeness);
    reporters.completeness.summarize("completeness");
    fmt::println("Completeness: {} molecules ({} skipped), {} determined pairs, " //
                 "{} unlisted, widest below the rescan floor {:.3g}",
                 fmt::group_digits(completeness.molecules), completeness.skipped, fmt::group_digits(completeness.pairs),
                 completeness.unlisted, completeness.widest_within_floor);

    if (!reporters.passed())
        return sftest::report_fail(test_name_k, "{} mismatches over {} comparisons, {} divergences asserted",
                                   reporters.mismatch_count(), reporters.comparison_count(),
                                   reporters.divergence_count());
    return sftest::report_pass(test_name_k);
}
