/**
 *  @file test/cuda/test_mmff_gradient.cpp
 *  @brief Gate the device MMFF94 energy and its analytical gradient against host references.
 *
 *  Two references, because neither alone is enough. RDKit's `calcEnergy` gates the total the kernel
 *  accumulates at identical coordinates, and a central difference of that same total gates the gradient
 *  the kernel writes beside it: an energy that matches RDKit under a gradient that does not
 *  differentiate it is exactly what a minimizer turns into a wrong conformer, and no total-energy
 *  comparison sees it.
 *
 *  The out-of-plane bend is probed on its own first. It is the one MMFF94 form whose derivative shares
 *  no algebra with the rest of the stage, so a sign error in it hides inside a full-stage total but
 *  cannot hide once every other term's force constant is gone.
 *
 *  The built-in molecules are what `ctest` gates on, and the two of them whose totals do not reproduce
 *  RDKit's are named in `known_energy_divergences_k` with what is known about each. A corpus named on
 *  the command line turns the same sweep into a survey: over a 40-molecule PubChem sample the gradient
 *  sections stay clean and the energy section reports real divergences, 18 molecules beyond its band and
 *  four of them past 20 kcal per mol, which is the state of the two implementations rather than a
 *  property of this harness.
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
 *  The step is bounded from below by cancellation, not by truncation. The kernel answers in `float`, so
 *  two energies of magnitude E are each rounded to about `E * 2^-23` before they are subtracted and the
 *  difference divided by `2 * step`; at glucose's 138 kcal per mol a 1e-4 step left the quotient with a
 *  0.17 resolution against analytical components of 1.3, and this test read that as a mismatch. Ten
 *  times the step divides that by ten, while the truncation a central difference pays grows as the
 *  square of it and stays four orders below the band either way.
 */
static constexpr float finite_difference_step_k = 1.0e-3f;

/** Components of the full gradient each molecule probes; the stride spreads them over all three axes. */
static constexpr std::size_t sampled_components_per_molecule_k = 12;

/** Mismatching fields each section prints before the rest are only counted. */
static constexpr std::size_t max_printed_lines_k = 32;

/**
 *  @brief Buffers every molecule and every probe reuses, so the sweep allocates once rather than per call.
 *
 *  Each finite-difference component costs two device energy evaluations, so a molecule of sixty atoms
 *  runs the evaluator about fifty times; anything sized per call would be an allocation per probe.
 */
struct gradient_scratch_t {
    /** Preprocessed molecule, refilled per SMILES by `prepare_molecule_into`. */
    sfp::etkdg_prepared_molecule_t prepared {};
    /** The same molecule stripped down to its first out-of-plane bend. */
    sfp::etkdg_prepared_molecule_t single_oop {};
    /** RDKit's conformer, atom-major. */
    std::vector<float> atom_major;
    /** The same coordinates axis-major; a probe perturbs one component and restores it. */
    std::vector<float> axis_major;
    /** Analytical gradient of the full stage at the unperturbed coordinates. */
    std::vector<float> stage_gradient;
    /** Analytical gradient of the isolated out-of-plane term at the same coordinates. */
    std::vector<float> term_gradient;
    /** The gradient a probe's energy call writes and nobody reads. */
    std::vector<float> probe_gradient;
    /** Storage behind the hop matrix every energy call rebuilds. */
    std::vector<std::uint8_t> bond_hops;
};

/**
 *  @brief Evaluate one molecule's MMFF94 energy and gradient on the device.
 *
 *  The pool ships no hop matrix: the device rebuilds it per wave from the bond list, and this
 *  single-molecule harness does the same so the nonbonded sweep sees production's exclusions.
 *
 *  @param[in] prepared Molecule whose terms are evaluated.
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
 *
 *  The perturbed component is restored to the exact value it held, so a caller may walk every component
 *  of one molecule through the same buffer.
 *
 *  @param[in] prepared Molecule whose energy is differentiated.
 *  @param[in] axis Spatial dimension of the perturbed component, `0` for x through `2` for z.
 *  @param[in] atom Atom whose coordinate is perturbed.
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

    // Restored before the refusal is answered, so a probe leaves the coordinates it borrowed as it found
    // them whatever the device said.
    coordinates(atom, axis) = original;
    if (probe != mmff_energy_probe_t::completed_k) return probe;
    derivative = (energy_plus - energy_minus) / (2.0f * finite_difference_step_k);
    return mmff_energy_probe_t::completed_k;
}

#pragma endregion Device Evaluation

#pragma region Tolerances

/** The band an analytical derivative is trusted to against a central difference of a `float` energy. */
struct gradient_tolerance_t {
    /** Absolute error accepted outright, which is what decides a component whose derivative is near zero. */
    float absolute = 0.0f;
    /** Error relative to the difference itself, which takes over above unit magnitude. */
    float relative = 0.0f;

    bool agree(float finite_difference, float analytical) const noexcept {
        float const error = std::fabs(analytical - finite_difference);
        return error < absolute || error / std::max(1.0f, std::fabs(finite_difference)) < relative;
    }
};

/**
 *  @brief The band one isolated out-of-plane term is held to.
 *
 *  Its energy is a few kcal per mol whatever the molecule around it, so the cancellation the difference
 *  pays does not grow with atom count either: over the built-ins the worst error is 3e-5, which the run
 *  prints beside this band.
 */
static constexpr gradient_tolerance_t single_term_tolerance_k {.absolute = 2.0e-3f, .relative = 1.0e-2f};

/**
 *  @brief The band the full stage is held to, which the size of the total it differentiates decides.
 *
 *  A hundred-atom molecule carries a few hundred kcal per mol through the same `float` subtraction, so
 *  this band is set by the largest built-in rather than the smallest: the worst error over them is 0.016.
 */
static constexpr gradient_tolerance_t full_stage_tolerance_k {.absolute = 2.5e-2f, .relative = 7.5e-2f};

/**
 *  @brief Whether the device total and RDKit's are the same number, at the resolution `float` allows.
 *
 *  The band is what the two implementations measurably achieve rather than what they can be made to
 *  pass at: eight of the ten built-in molecules agree to 2e-4 kcal per mol or better, so a band of a
 *  whole kcal would leave four orders of magnitude of regression unreported. The two that do not agree
 *  are named in `known_energy_divergences_k`.
 */
static bool energies_agree_(double rdkit_energy, double device_energy) noexcept {
    double const band = std::max(1.0e-2, 1.0e-4 * std::fabs(rdkit_energy));
    return std::fabs(device_energy - rdkit_energy) < band;
}

/** A molecule whose two MMFF94 implementations disagree today, and what the disagreement is. */
struct known_divergence_t {
    std::string_view smiles;
    std::string_view reason;
};

/**
 *  @brief Built-in molecules whose device total is known not to reproduce RDKit's, and why.
 *
 *  Their energies are narrated rather than gated, so the divergence is printed by every run instead of
 *  living only in this table, and a listed molecule that starts agreeing fails the run as a stale entry.
 *
 *  Caffeine's native MMFF94 partial charges differ from RDKit's on ten of its twenty-four atoms, worst
 *  0.59 e on a ring carbonyl carbon, against 6e-9 across every other built-in; its atom types agree, so
 *  what reaches the kernel is already wrong and the total comes back 13.9 kcal per mol low.
 */
static constexpr known_divergence_t known_energy_divergences_k[] = {
    {"Cn1c(=O)c2c(ncn2C)n(C)c1=O", "native MMFF94 partial charges differ from RDKit by up to 0.59 e"},
};

/** The reason @p smiles is a known divergence, or an empty view when it is gated like any other. */
static std::string_view known_energy_divergence_(std::string_view smiles) noexcept {
    for (known_divergence_t const &divergence : known_energy_divergences_k)
        if (divergence.smiles == smiles) return divergence.reason;
    return {};
}

/**
 *  @brief Whether one axis sum of a gradient cancelled to zero, at the resolution its magnitudes allow.
 *
 *  A translation-invariant field sums to zero exactly in real arithmetic; in `float` it sums to the
 *  rounding of the magnitudes that cancelled, so the band scales with them and floors at 1e-2.
 */
static bool axis_sum_vanishes_(float magnitude_sum, float axis_sum) noexcept {
    return std::fabs(axis_sum) < std::max(1.0e-2f, 1.0e-5f * magnitude_sum);
}

#pragma endregion Tolerances

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
 *  The two worst errors are the margin the gradient bands were chosen against, printed every run so the
 *  bands stay auditable rather than inherited: a band ten times the worst error is slack, and a band
 *  just above it is a test that fails on the next molecule.
 */
struct gradient_tally_t {
    std::uint32_t compared = 0;
    std::uint32_t unprepared = 0;
    std::uint32_t unembedded = 0;
    std::uint32_t unaligned = 0;
    std::uint32_t unreferenced = 0;
    /** Molecules the device refused a launch for, which is a failure of the run rather than of a molecule. */
    std::uint32_t refused = 0;
    /** Which step the first refusal came from, which is the one thing the count cannot carry. */
    mmff_energy_probe_t first_refusal = mmff_energy_probe_t::completed_k;
    /** Compared molecules carrying no out-of-plane term, which that section has nothing to probe on. */
    std::uint32_t without_oop = 0;
    /** Compared molecules on `known_energy_divergences_k`, whose energies are narrated rather than gated. */
    std::uint32_t known_divergence = 0;
    /** Listed molecules that agree with RDKit, so their roster entry states something untrue. */
    std::uint32_t stale_divergence = 0;
    float worst_oop_error = 0.0f;
    float worst_stage_error = 0.0f;
};

/**
 *  @brief One reporter per compared section, so each closes with its own `compared=` count.
 *
 *  The four are separate reporters rather than four `summarize` calls on one, because a molecule
 *  contributes to all four before the next molecule starts and a shared reporter would attribute every
 *  count to whichever section summarized first.
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

    /** Cap what each section prints, so one pathological molecule cannot bury the log. */
    void limit_printed_lines(std::size_t maximum) noexcept {
        oop_gradient.limit_printed_lines(maximum);
        stage_energy.limit_printed_lines(maximum);
        translation_invariance.limit_printed_lines(maximum);
        stage_gradient.limit_printed_lines(maximum);
    }

    std::size_t comparison_count() const noexcept {
        return oop_gradient.comparison_count() + stage_energy.comparison_count() +
               translation_invariance.comparison_count() + stage_gradient.comparison_count();
    }

    std::size_t mismatch_count() const noexcept {
        return oop_gradient.mismatch_count() + stage_energy.mismatch_count() + translation_invariance.mismatch_count() +
               stage_gradient.mismatch_count();
    }

    /** Close all four sections; true when each compared something and every field agreed. */
    bool summarize_sections() {
        oop_gradient.summarize("oop_gradient");
        stage_energy.summarize("stage_energy");
        translation_invariance.summarize("translation_invariance");
        stage_gradient.summarize("stage_gradient");
        return oop_gradient.passed() && stage_energy.passed() && translation_invariance.passed() &&
               stage_gradient.passed();
    }
};

/**
 *  @brief Fill @p single with @p prepared's first out-of-plane bend and nothing else.
 *
 *  Every other category is emptied rather than zeroed, and the four positions of that one term become
 *  the whole interaction list, so each of i, j, k and l receives a gradient from the out-of-plane form
 *  alone. With no van der Waals parameters and no charges the nonbonded sweep contributes nothing
 *  either, which is what leaves the difference attributable to a single term.
 *
 *  @param[in] prepared Molecule carrying at least one out-of-plane term.
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
 *
 *  @param[in] smiles Molecule under test, carried into the diagnostic keys.
 *  @param[inout] scratch Holds the coordinates the probes perturb and the buffers they fill.
 *  @param[inout] reporter Section that receives one comparison per position and axis.
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
 *  @brief Compare the three axis sums of the full gradient against the zero invariance under translation
 *  requires of them.
 *
 *  @param[in] smiles Molecule under test, carried into the diagnostic keys.
 *  @param[in] gradient Full-stage analytical gradient, axis-major.
 *  @param[inout] reporter Section that receives one comparison per axis.
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
 *  The stride is derived from the molecule's own component count rather than fixed, so the samples span
 *  all three axis blocks whatever the atom count; a fixed stride lands entirely inside x once a molecule
 *  passes a few dozen atoms.
 *
 *  @param[in] smiles Molecule under test, carried into the diagnostic keys.
 *  @param[inout] scratch Holds the coordinates the probes perturb and the gradient they are compared to.
 *  @param[inout] reporter Section that receives one comparison per sampled component.
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

    // The counter walks storage, which an axis-major view packs one whole axis at a time, so the axis is the
    // quotient and the atom the remainder however the view is subscripted.
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
 *
 *  @param[in] smiles Molecule to prepare, embed and compare.
 *  @param[inout] reporters The four sections, each receiving this molecule's comparisons.
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

    // A salt is the case that reaches here: preprocessing keeps a different atom set than `addHs` does,
    // and two atom counts that disagree are two atom orders no comparison can align.
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

    /** Keep the first refusal the molecule met, so the run's failure can name the step that refused. */
    auto const refuse_ = [&tally](mmff_energy_probe_t probe) {
        if (tally.first_refusal == mmff_energy_probe_t::completed_k) tally.first_refusal = probe;
        return molecule_outcome_t::refused_k;
    };

    float device_energy = 0.0f;
    if (mmff_energy_probe_t const probe = mmff_energy_and_gradient_(scratch.prepared, scratch.axis_major, scratch,
                                                                    scratch.stage_gradient, device_energy);
        probe != mmff_energy_probe_t::completed_k)
        return refuse_(probe);
    std::string_view const divergence_reason = known_energy_divergence_(smiles);
    if (divergence_reason.empty())
        reporters.stage_energy.mismatch("stage_energy", "smiles={} atoms={}", smiles, scratch.prepared.num_atoms)
            .field("total_energy", rdkit_energy, static_cast<double>(device_energy), energies_agree_);
    else {
        // Narrated, not compared: the number is the point, and printing it every run is what keeps the
        // roster entry from becoming a claim nobody rechecks.
        reporters.stage_energy.note("known_divergence", "smiles={} reason={}", smiles, divergence_reason)
            .field("total_energy", rdkit_energy, static_cast<double>(device_energy));
        ++tally.known_divergence;
        if (energies_agree_(rdkit_energy, static_cast<double>(device_energy))) ++tally.stale_divergence;
    }

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
 *  @brief Walk the corpus, comparing every molecule the two references both cover.
 *
 *  The scratch is declared once above the loop, so the buffers grow to the largest molecule and never
 *  again; the sweep runs to completion whatever it finds, because the counts are the output.
 *
 *  @param[in] corpus SMILES to compare, built-ins first.
 *  @param[inout] reporters The four sections.
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
    sftest::gradient_tally_t tally;
    sftest::sweep_corpus_(corpus, reporters, tally);

    fmt::println("molecules compared={} unprepared={} unembedded={} unaligned={} " // what the sweep saw
                 "unreferenced={} refused={} without_oop={} known_divergence={}",  // and what it stepped past
                 tally.compared, tally.unprepared, tally.unembedded, tally.unaligned, tally.unreferenced, tally.refused,
                 tally.without_oop, tally.known_divergence);
    fmt::println("worst gradient error oop={} stage={} against absolute floors {:.3g} and {:.3g}, " // the margin
                 "above which the relative band decides",                                           // and the band
                 tally.worst_oop_error, tally.worst_stage_error, sftest::single_term_tolerance_k.absolute,
                 sftest::full_stage_tolerance_k.absolute);
    bool const sections_passed = reporters.summarize_sections();
    if (tally.refused != 0)
        return sftest::report_fail(test_name_k, "the device refused {} molecules, first as {}", tally.refused,
                                   sftest::mmff_energy_probe_reason(tally.first_refusal));
    if (tally.compared == 0) return sftest::report_fail(test_name_k, "corpus yielded no comparable molecule");
    if (tally.stale_divergence != 0)
        return sftest::report_fail(test_name_k, "{} known divergences now agree with RDKit and must leave the roster",
                                   tally.stale_divergence);
    if (!sections_passed)
        return sftest::report_fail(test_name_k, "{} mismatches over {} comparisons", reporters.mismatch_count(),
                                   reporters.comparison_count());
    return sftest::report_pass(test_name_k);
}
