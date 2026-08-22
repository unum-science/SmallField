/**
 *  @file test/cuda/distance_geometry.cpp
 *  @brief Gate the device distance-geometry energy, its gradient, its chirality window, and FIRE minimization.
 *
 *  Distance geometry embeds in @b four dimensions. The fourth axis is a real coordinate, the one the crush
 *  penalty drives toward zero before the structure is read back as 3D, so the axis count lives in the
 *  view's type `sft::embedding_soa_t`; only the transposition to RDKit's interleaved `(atom, 3)` layout is
 *  three-dimensional. The device quantizes the bounds matrix to int8 over the molecule's whole bound range,
 *  so one level is the range over 255 and the reference below mirrors that rather than float bounds the
 *  kernel never saw. The range is a property of the input, so this file smooths first as the metric prepass
 *  does, replacing the builder's 1000 angstrom placeholders with real diameters, which costs around 0.05 % of
 *  the energy rather than twenty times it; what it does cost is narrated beside the comparison, which is gated
 *  at `energy_relative_tolerance_k`. The chirality section presses
 *  each center onto its neighbors' plane and asserts the volume window rejects it, since a signed volume
 *  alone reads a coplanar center as whichever handedness it was asked for.
 *
 *  Corpus: a `.smi` path in `argv[1]` or `SMALLFIELD_TEST_CORPUS`, capped by `argv[2]` or
 *  `SMALLFIELD_TEST_MAX_MOLS`; absent both the built-in list runs, which is what `ctest` invokes. The
 *  finite-difference section costs two launches per component, so it walks only the first few molecules.
 *  Compiled with g++, C++20. CUDA kernels in distance_geometry_kernels.cu.
 */

#include <cmath>   // `std::sqrt`, `std::fabs`, `std::isfinite`, `std::lrintf`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::uint32_t`, `std::int32_t`

#include <algorithm>   // `std::max`, `std::min`
#include <limits>      // `std::numeric_limits`
#include <memory>      // `std::unique_ptr`
#include <random>      // `std::mt19937`, `std::uniform_real_distribution`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/views.hpp"

#include "test/cuda/distance_geometry_kernels.hpp"
#include "test/rdkit_reference.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

using sft::coordinate_role_t;
using sftest::gpu_dg_energy;
using sftest::gpu_dg_minimize;

static constexpr std::string_view test_name_k = "smallfield_test_distance_geometry";

#pragma region Coordinate Views

/**
 *  @brief A gradient over an embedding's four axes; nothing but the role separates it from the embedding.
 *  Both are addressed `(atom, axis)`, axes zero through two being x, y, z and axis three the w the crush
 *  penalty flattens before projection back to 3D. Every buffer this file hands a kernel carries all four.
 */
using gradient_t = sft::coordinates_soa<coordinate_role_t::gradient_k, float, sft::embedding_dimensions_k>;

/** Grow @p storage to a four-axis embedding over @p num_atoms atoms, zero it, and view it. */
static sft::embedding_soa_t reset_embedding_(std::vector<float> &storage, std::uint32_t num_atoms) {
    return sft::reset_coordinates_soa<coordinate_role_t::embedding_k, sft::embedding_dimensions_k>(storage, num_atoms);
}

/** Grow @p storage to a gradient over a four-axis embedding, zero it, and view it. */
static gradient_t reset_gradient_(std::vector<float> &storage, std::uint32_t num_atoms) {
    return sft::reset_coordinates_soa<coordinate_role_t::gradient_k, sft::embedding_dimensions_k>(storage, num_atoms);
}

/** Grow @p storage to @p num_atoms atom-major triples, zero it, and view it. */
static sft::conformer_aos_t reset_interleaved_(std::vector<float> &storage, std::uint32_t num_atoms) {
    return sft::reset_coordinates_aos<coordinate_role_t::position_k>(storage, num_atoms);
}

/**
 *  @brief Triangle-smooth a bounds matrix in place, the way the metric prepass does before the kernel.
 *
 *  The builder leaves every unconstrained pair's upper bound at its 1000 angstrom placeholder, which no
 *  pipeline kernel ever sees: the metric prepass smooths and writes back before quantization. Handing the
 *  placeholder over instead spans the int8 range across a thousand angstroms, so one of the 256 levels
 *  covers four, every lower bound under two angstroms quantizes to zero, and no repulsion is left -- two
 *  coincident atoms then being the correct answer to the question actually asked.
 *  Upper bounds contract along the triangle inequality; lower bounds expand against both candidates it
 *  offers, rather than the one an `else if` would keep.
 *  @param[inout] bounds Square matrix, upper bounds above the diagonal and lower bounds below.
 */
static void smooth_bounds_(sft::packed_matrix<float> bounds) {
    std::uint32_t const num_atoms = bounds.extent(0);
    for (std::uint32_t pivot = 0; pivot < num_atoms; ++pivot)
        for (std::uint32_t row = 0; row < num_atoms; ++row) {
            if (row == pivot) continue;
            for (std::uint32_t column = row + 1; column < num_atoms; ++column) {
                if (column == pivot) continue;
                float const upper_row_pivot = sft::upper_distance_bound_at(bounds, row, pivot);
                float const upper_pivot_column = sft::upper_distance_bound_at(bounds, pivot, column);
                float const lower_row_pivot = sft::lower_distance_bound_at(bounds, row, pivot);
                float const lower_pivot_column = sft::lower_distance_bound_at(bounds, pivot, column);

                float &upper = bounds(row, column);
                float &lower = bounds(column, row);
                upper = std::min(upper, upper_row_pivot + upper_pivot_column);
                lower = std::max(lower,
                                 std::max(lower_row_pivot - upper_pivot_column, lower_pivot_column - upper_row_pivot));
                lower = std::min(lower, upper);
            }
        }
}

/**
 *  @brief Build the bounds matrix into caller storage and hand back a view whose extent cannot drift.
 *  @param[inout] storage Caller-owned buffer, reused across molecules so growth happens once.
 *  @return Row-major N × N view: upper bounds above the diagonal, lower bounds below.
 */
static sft::packed_matrix<float const> reset_bounds_matrix_(sft::molecule_descriptor_t const &descriptor,
                                                            std::vector<float> &storage) {
    std::uint32_t const num_atoms = descriptor.num_atoms;
    sft::packed_matrix<float> const bounds = sft::reset_square_matrix(storage, num_atoms, 0.0f);
    sfp::build_bounds_matrix(descriptor, bounds);
    smooth_bounds_(bounds);
    return bounds;
}

#pragma endregion Coordinate Views

#pragma region Host Reference

/** The four penalty weights the kernels take, kept together so a call site cannot transpose them. */
struct dg_weights_t {
    float upper = 1.0f;
    float lower = 1.0f;
    float chiral = 0.0f;
    float fourth_dimension = 0.0f;
};

/** Bind one molecule's bounds and chiral centers to a section's weights, for one device call. */
static sftest::dg_problem_t problem_for_(sft::packed_matrix<float const> bounds,
                                         custd::span<sft::chiral_center_descriptor_t const> centers,
                                         dg_weights_t weights) {
    return sftest::dg_problem_t {.bounds = bounds.data_handle(),
                                 .chiral_centers = centers.empty() ? nullptr : centers.data(),
                                 .num_atoms = bounds.extent(0),
                                 .num_chiral_centers = static_cast<std::uint32_t>(centers.size()),
                                 .spatial_dimension = sft::embedding_dimensions_k,
                                 .upper_weight = weights.upper,
                                 .lower_weight = weights.lower,
                                 .chiral_weight = weights.chiral,
                                 .fourth_dimension_weight = weights.fourth_dimension};
}

/**
 *  @brief The int8 bounds the device actually reads, reproduced on the host.
 *
 *  `quantize_bounds_into_shared_` rescales the whole matrix into 256 levels and the energy functor reads
 *  each bound back through one fused multiply-add; a reference skipping this compares against numbers the
 *  kernel never saw.
 */
struct quantized_bounds_t {
    sft::packed_matrix<std::uint8_t> levels;
    float minimum = 0.0f;
    float inverse_scale = 0.0f;

    /** The bound the kernel reads at @p row, @p column -- upper above the diagonal, lower below. */
    float bound(std::uint32_t row, std::uint32_t column) const noexcept {
        return minimum + static_cast<float>(levels(row, column)) * inverse_scale;
    }
};

/** Quantize @p bounds the way the device does, into caller-owned @p storage. */
static quantized_bounds_t quantize_bounds_(sft::packed_matrix<float const> bounds, std::vector<std::uint8_t> &storage) {
    std::uint32_t const num_atoms = bounds.extent(0);
    float minimum = std::numeric_limits<float>::max();
    float maximum = -std::numeric_limits<float>::max();
    for (std::uint32_t row = 0; row < num_atoms; ++row)
        for (std::uint32_t column = 0; column < num_atoms; ++column) {
            minimum = std::min(minimum, bounds(row, column));
            maximum = std::max(maximum, bounds(row, column));
        }

    float const range = std::max(maximum - minimum, 1e-6f);
    float const scale = 255.0f / range;

    quantized_bounds_t quantized;
    quantized.levels = sft::reset_square_matrix<std::uint8_t>(storage, num_atoms, std::uint8_t {0});
    for (std::uint32_t row = 0; row < num_atoms; ++row)
        for (std::uint32_t column = 0; column < num_atoms; ++column) {
            long const level = std::lrintf((bounds(row, column) - minimum) * scale);
            quantized.levels(row, column) = static_cast<std::uint8_t>(std::max(0L, std::min(255L, level)));
        }
    quantized.minimum = minimum;
    quantized.inverse_scale = range * (1.0f / 255.0f);
    return quantized;
}

/**
 *  @brief Host mirror of `evaluate_distance_geometry_energy` for one molecule, in double precision.
 *  @param[in] coordinates Four-axis SoA embedding; every axis participates in the pair distances.
 *  @param[in] read_bound Called as `read_bound(row, column)`, upper above the diagonal and lower below.
 *  @param[in] weights Penalty weights, matching what the kernel was handed.
 *  @return Total energy.
 */
template <typename bound_reader_>
static double reference_dg_energy_(sft::embedding_soa_t coordinates, bound_reader_ read_bound, dg_weights_t weights) {
    std::uint32_t const num_atoms = coordinates.extent(0);
    double energy = 0.0;
    for (std::uint32_t first_atom = 0; first_atom < num_atoms; ++first_atom) {
        for (std::uint32_t second_atom = first_atom + 1; second_atom < num_atoms; ++second_atom) {
            double distance_squared = 0.0;
            for (std::uint32_t dimension = 0; dimension < coordinates.extent(1); ++dimension) {
                double const delta = coordinates(second_atom, dimension) - coordinates(first_atom, dimension);
                distance_squared += delta * delta;
            }
            double const distance = std::sqrt(std::max(distance_squared, 1e-12));

            double const upper = read_bound(first_atom, second_atom);
            double const lower = read_bound(second_atom, first_atom);
            if (distance > upper) energy += weights.upper * (distance - upper) * (distance - upper);
            if (distance < lower) energy += weights.lower * (lower - distance) * (lower - distance);
        }
    }

    // The crush penalty is per atom rather than per pair, and only the fourth axis feels it.
    if (weights.fourth_dimension > 0.0f)
        for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
            double const fourth = coordinates(atom, 3);
            energy += weights.fourth_dimension * fourth * fourth;
        }
    return energy;
}

/** The largest amount, in angstroms, by which any pair sits outside its bounds. */
static float max_bound_violation_(sft::embedding_soa_t coordinates, sft::packed_matrix<float const> bounds) {
    std::uint32_t const num_atoms = coordinates.extent(0);
    float worst = 0.0f;
    for (std::uint32_t first_atom = 0; first_atom < num_atoms; ++first_atom)
        for (std::uint32_t second_atom = first_atom + 1; second_atom < num_atoms; ++second_atom) {
            float distance_squared = 0.0f;
            for (std::uint32_t dimension = 0; dimension < coordinates.extent(1); ++dimension) {
                float const delta = coordinates(second_atom, dimension) - coordinates(first_atom, dimension);
                distance_squared += delta * delta;
            }
            float const distance = std::sqrt(distance_squared);
            worst = std::max(worst, distance - bounds(first_atom, second_atom));
            worst = std::max(worst, bounds(second_atom, first_atom) - distance);
        }
    return std::max(worst, 0.0f);
}

/** The largest distance any pair is allowed, which is what an atom has to clear to violate all of them. */
static float largest_bound_(sft::packed_matrix<float const> bounds) {
    float largest = 0.0f;
    for (std::uint32_t row = 0; row < bounds.extent(0); ++row)
        for (std::uint32_t column = 0; column < bounds.extent(1); ++column)
            largest = std::max(largest, bounds(row, column));
    return largest;
}

/** Widest extent of the structure along any one axis. */
static float coordinate_span_(sft::embedding_soa_t coordinates) {
    float span = 0.0f;
    for (std::uint32_t axis = 0; axis < coordinates.extent(1); ++axis) {
        float lowest = coordinates(0, axis), highest = coordinates(0, axis);
        for (std::uint32_t atom = 0; atom < coordinates.extent(0); ++atom) {
            lowest = std::min(lowest, coordinates(atom, axis));
            highest = std::max(highest, coordinates(atom, axis));
        }
        span = std::max(span, highest - lowest);
    }
    return span;
}

/** Mean magnitude of the embedding's fourth axis — the quantity the crush penalty exists to shrink. */
static float mean_fourth_dimension_magnitude_(sft::embedding_soa_t coordinates) {
    std::uint32_t const num_atoms = coordinates.extent(0);
    double total = 0.0;
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) total += std::fabs(coordinates(atom, 3));
    return static_cast<float>(total / std::max(1u, num_atoms));
}

/**
 *  @brief Fill the embedding's fourth axis with small deterministic offsets. The arithmetic is signed on
 *  purpose: `atom % 3 - 1` on an unsigned index wraps to 4294967295 for every third atom, putting the
 *  kernel's coordinates near 4e8 with reference and kernel agreeing on the nonsense.
 */
static void fill_fourth_dimension_(sft::embedding_soa_t coordinates) {
    for (std::uint32_t atom = 0; atom < coordinates.extent(0); ++atom)
        coordinates(atom, 3) = 0.1f * (static_cast<float>(atom % 3u) - 1.0f);
}

#pragma endregion Host Reference

#pragma region Predicates and Constants

/** Relative agreement demanded of the kernel against the host mirror of the same arithmetic. */
static constexpr double energy_relative_tolerance_k = 1e-3;
/** A finite-difference component disagrees only when it is off both relatively and absolutely. */
static constexpr float gradient_relative_tolerance_k = 0.05f;
static constexpr float gradient_absolute_tolerance_k = 0.01f;
/** Central-difference step, in angstroms. */
static constexpr float finite_difference_step_k = 1e-3f;
/** Clearance added past the largest bound when one atom is pushed out of every bound it has. */
static constexpr float bound_clearance_angstrom_k = 1.0f;
/** Half-width of the uniform box the minimization starts from, in angstroms. */
static constexpr float random_start_span_k = 5.0f;
static constexpr std::int32_t max_iterations_k = 400;
static constexpr float gradient_threshold_k = 1e-4f;
/** Weight on the fourth-dimension crush penalty wherever a section exercises it, matching the pipeline's. */
static constexpr float fourth_dimension_weight_k = 10.0f;
/** The finite-difference sweep costs two launches per component, so it walks only the first few molecules. */
static constexpr std::size_t max_gradient_molecules_k = 3;

/**
 *  @brief How far each coordinate of a well-posed start is displaced, in angstrom.
 *
 *  Large enough that the minimizer cannot pass by standing still, small enough that it relaxes back into
 *  the basin RDKit's own embedding sits in. From the random `random_start_span_k` cloud only descent is
 *  demanded, since requiring convergence out of noise would ratchet the suite on noise.
 */
static constexpr float perturbation_span_angstrom_k = 0.2f;

/** Molecules the convergence section walks, which pays a full minimization each. */
static constexpr std::size_t max_convergence_molecules_k = 3;
/** The chirality sweep costs the same per center, so it too walks only the first few chiral molecules. */
static constexpr std::size_t max_chiral_molecules_k = 4;
/** One pathological molecule must not bury the log; suppressed lines are still counted. */
static constexpr std::size_t max_printed_lines_k = 10;

static bool energies_agree_(double reference, double observed) {
    return std::fabs(observed - reference) <= energy_relative_tolerance_k * std::max(1.0, std::fabs(reference));
}

static bool gradients_agree_(double finite_difference, double analytical) {
    double const absolute_error = std::fabs(analytical - finite_difference);
    double const relative_error = absolute_error / std::max(1.0, std::fabs(analytical));
    return relative_error <= gradient_relative_tolerance_k || absolute_error <= gradient_absolute_tolerance_k;
}

/** Weight on the chiral volume penalty wherever a section exercises it, matching the pipeline's. */
static constexpr float chiral_weight_k = 100.0f;
/** Fraction of its out-of-plane height a flattened center keeps, small enough to be flat, not zero. */
static constexpr float near_flat_retention_k = 0.02f;
/**
 *  Fraction kept for the finite-difference sweep. The penalty is active while `|V| < minimum_volume`, so
 *  a realized volume times this must still land under that floor; otherwise `excess_volume` is zero and
 *  the sweep compares zero against zero. `chirality_penalty_is_active` checks that it still holds.
 */
static constexpr float penalized_retention_k = 0.3f;

/** Relative agreement demanded between the minimiser's own report and a fresh evaluation of it. */
static constexpr double reported_state_relative_tolerance_k = 1e-3;

static bool rms_gradients_agree_(double recomputed, double reported) {
    return std::fabs(reported - recomputed) <=
           reported_state_relative_tolerance_k * std::max(1e-6, std::fabs(recomputed));
}

/** The word a `fire_status_t` prints as, so the diagnostic key is readable and greppable. */
static std::string_view fire_status_name_(smallfield::cuda::fire_status_t status) noexcept {
    return status == smallfield::cuda::fire_status_t::converged_k ? "converged" : "exhausted";
}

static bool increased_(double before, double after) { return after > before; }
static bool decreased_(double before, double after) { return after < before; }
static bool at_most_(double ceiling, double observed) { return observed <= ceiling; }
static bool both_finite_(double reference, double observed) {
    return std::isfinite(reference) && std::isfinite(observed);
}

#pragma endregion Predicates and Constants

#pragma region Molecule Checks

/** Buffers the sections reuse across the corpus, so a growth phase happens once per run. */
struct molecule_scratch_t {
    std::vector<float> bounds;
    std::vector<std::uint8_t> bound_levels;
    std::vector<float> embedding;
    std::vector<float> gradient;
    std::vector<float> spare_gradient;
    std::vector<float> interleaved;
    std::vector<sft::chiral_center_descriptor_t> chiral_centers;
};

/**
 *  @brief One molecule as every section sees it. The bounds carry their own extent and the centers their
 *  own count, so nothing restates either and the three travel together rather than as five arguments.
 */
struct molecule_case_t {
    std::string_view smiles;
    sft::packed_matrix<float const> bounds;
    custd::span<sft::chiral_center_descriptor_t const> chiral_centers;
};

/**
 *  @brief The reporters this test closes: one per gated section, plus one that only narrates.
 *
 *  `quantization` is never summarized because it compares nothing -- both operands are host references,
 *  and their divergence is the expected cost of the device's int8 bounds rather than a defect. A section
 *  comparing nothing is a failure by construction, which is the rule keeping it out.
 */
struct dg_reporters_t {
    sftest::diagnostic_reporter_t energy {"cpu", "gpu"};
    sftest::diagnostic_reporter_t gradient {"finite_difference", "analytical"};
    sftest::diagnostic_reporter_t chirality {"expected", "observed"};
    sftest::diagnostic_reporter_t minimize {"before", "after"};
    sftest::diagnostic_reporter_t convergence {"threshold", "achieved"};
    sftest::diagnostic_reporter_t quantization {"exact_bounds", "int8_bounds"};

    void limit_printed_lines(std::size_t maximum) noexcept {
        energy.limit_printed_lines(maximum);
        gradient.limit_printed_lines(maximum);
        chirality.limit_printed_lines(maximum);
        minimize.limit_printed_lines(maximum);
        convergence.limit_printed_lines(maximum);
        quantization.limit_printed_lines(maximum);
    }

    bool passed() const noexcept {
        return energy.passed() && gradient.passed() && chirality.passed() && minimize.passed() && convergence.passed();
    }
};

/** How far a molecule got before a section could use it. */
enum class molecule_setup_t {
    /** Parsed, embedded, and described. */
    ready_k,
    /** SMILES the parser refused. */
    unparsed_k,
    /** ETKDG produced no conformer. */
    unembeddable_k,
    /** Too many atoms or bonds for the flat descriptor. */
    undescribable_k,
};

/** The sentence a `molecule_setup_t` reads as, for the run's skip tally. */
static std::string_view molecule_setup_reason_(molecule_setup_t outcome) noexcept {
    switch (outcome) {
    case molecule_setup_t::unparsed_k: return "unparsed";
    case molecule_setup_t::unembeddable_k: return "unembeddable";
    case molecule_setup_t::undescribable_k: return "undescribable";
    default: return "ready";
    }
}

/** Parse, embed, and describe one SMILES; every section needs all three. */
static molecule_setup_t setup_molecule_(std::string const &smiles, std::unique_ptr<RDKit::RWMol> &molecule,
                                        sft::molecule_descriptor_t &descriptor) {
    molecule = sftest::mol_from_smiles(smiles);
    if (molecule == nullptr) return molecule_setup_t::unparsed_k;
    if (!sftest::embed_molecule(*molecule)) return molecule_setup_t::unembeddable_k;
    if (sft::build_descriptor(*molecule, descriptor) != sft::conformer_status_t::success_k)
        return molecule_setup_t::undescribable_k;
    return molecule_setup_t::ready_k;
}

/**
 *  @brief Gate the device energy against the host mirror, and gate that the penalty responds to a violation.
 *
 *  Two comparisons per call: the kernel's arithmetic against a reference reading the same quantized bounds,
 *  and the energy across a known 1.5 angstrom shift. The first would pass on a kernel that always returned
 *  zero; the second rules that out.
 *  @return Whether every launch ran, so a refusal reaches the gate instead of being compared.
 */
static sftest::dg_probe_t check_energy_(sftest::diagnostic_reporter_t &reporter,
                                        sftest::diagnostic_reporter_t &quantization, molecule_case_t const &subject,
                                        RDKit::ROMol const &molecule, dg_weights_t weights,
                                        molecule_scratch_t &scratch) {
    auto const bounds = subject.bounds;
    auto const &smiles = subject.smiles;
    std::uint32_t const num_atoms = bounds.extent(0);
    auto const coordinates = reset_embedding_(scratch.embedding, num_atoms);
    auto const gradient = reset_gradient_(scratch.gradient, num_atoms);
    auto const interleaved = reset_interleaved_(scratch.interleaved, num_atoms);

    sftest::read_conformer_into(molecule, interleaved);
    sft::transpose_into(interleaved, coordinates);
    fill_fourth_dimension_(coordinates);

    auto const problem = problem_for_(bounds, {}, weights);

    float relaxed_energy = 0.0f;
    sftest::dg_probe_t probe = gpu_dg_energy(problem, coordinates.data_handle(), gradient.data_handle(),
                                             &relaxed_energy);
    if (probe != sftest::dg_probe_t::completed_k) return probe;

    // Push atom zero along x past every bound it has: each of its pairs lands strictly on the upper-bound
    // branch and further out than before while no other pair moves, making the energy's rise a theorem
    // rather than an expectation. A small fixed nudge is not, and fails on ring diols.
    coordinates(0, 0) += largest_bound_(bounds) + coordinate_span_(coordinates) + bound_clearance_angstrom_k;

    float gpu_energy = 0.0f;
    probe = gpu_dg_energy(problem, coordinates.data_handle(), gradient.data_handle(), &gpu_energy);
    if (probe != sftest::dg_probe_t::completed_k) return probe;

    auto const quantized = quantize_bounds_(bounds, scratch.bound_levels);
    double const quantized_reference = reference_dg_energy_(
        coordinates, [&](std::uint32_t row, std::uint32_t column) { return quantized.bound(row, column); }, weights);
    double const exact_reference = reference_dg_energy_(
        coordinates, [&](std::uint32_t row, std::uint32_t column) { return bounds(row, column); }, weights);

    reporter
        .mismatch("dg_energy", "smiles={} atoms={} fourth_dimension_weight={}", smiles, num_atoms,
                  weights.fourth_dimension)
        .field("energy", quantized_reference, gpu_energy, energies_agree_)
        .field("energy_under_shift", relaxed_energy, gpu_energy, increased_);

    // Narrated rather than compared: with unconstrained pairs at 1000 angstroms, one of the 256 levels spans
    // nearly four angstroms.
    quantization
        .note("dg_bounds", "smiles={} atoms={} fourth_dimension_weight={}", smiles, num_atoms, weights.fourth_dimension)
        .field("energy", exact_reference, quantized_reference);
    return sftest::dg_probe_t::completed_k;
}

/**
 *  @brief Gate the analytic gradient against a central difference of the same kernel's energy. All four
 *  axes are swept under a non-zero fourth-dimension weight, so the crush term's contribution to w is
 *  covered rather than multiplied away.
 *  @return Whether every launch ran, so a refusal reaches the gate instead of being compared.
 */
static sftest::dg_probe_t check_gradient_(sftest::diagnostic_reporter_t &reporter, molecule_case_t const &subject,
                                          RDKit::ROMol const &molecule, dg_weights_t weights,
                                          molecule_scratch_t &scratch) {
    auto const bounds = subject.bounds;
    auto const &smiles = subject.smiles;
    std::uint32_t const num_atoms = bounds.extent(0);
    auto const coordinates = reset_embedding_(scratch.embedding, num_atoms);
    auto const analytical = reset_gradient_(scratch.gradient, num_atoms);
    auto const spare = reset_gradient_(scratch.spare_gradient, num_atoms);
    auto const interleaved = reset_interleaved_(scratch.interleaved, num_atoms);

    sftest::read_conformer_into(molecule, interleaved);
    sft::transpose_into(interleaved, coordinates);

    // A conformer sits close to its bounds and a gradient of exact zeros compares equal to anything, so
    // displacing every component by up to 0.3 angstroms puts the penalties on their active branch.
    std::mt19937 generator(123);
    std::uniform_real_distribution<float> displacement(-0.3f, 0.3f);
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
        for (std::uint32_t dimension = 0; dimension < coordinates.extent(1); ++dimension)
            coordinates(atom, dimension) += displacement(generator);

    auto const problem = problem_for_(bounds, subject.chiral_centers, weights);

    float energy_at_center = 0.0f;
    sftest::dg_probe_t probe = gpu_dg_energy(problem, coordinates.data_handle(), analytical.data_handle(),
                                             &energy_at_center);
    if (probe != sftest::dg_probe_t::completed_k) return probe;

    for (std::uint32_t dimension = 0; dimension < coordinates.extent(1); ++dimension) {
        for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
            float const original = coordinates(atom, dimension);

            coordinates(atom, dimension) = original + finite_difference_step_k;
            float energy_plus = 0.0f;
            probe = gpu_dg_energy(problem, coordinates.data_handle(), spare.data_handle(), &energy_plus);

            coordinates(atom, dimension) = original - finite_difference_step_k;
            float energy_minus = 0.0f;
            if (probe == sftest::dg_probe_t::completed_k)
                probe = gpu_dg_energy(problem, coordinates.data_handle(), spare.data_handle(), &energy_minus);

            coordinates(atom, dimension) = original;
            // Restored first, so a refusal still leaves the borrowed coordinates as the sweep found them.
            if (probe != sftest::dg_probe_t::completed_k) return probe;

            float const finite_difference = (energy_plus - energy_minus) / (2.0f * finite_difference_step_k);
            reporter.mismatch("dg_gradient", "smiles={} atom={} dimension={}", smiles, atom, dimension)
                .field("gradient", finite_difference, analytical(atom, dimension), gradients_agree_);
        }
    }
    return sftest::dg_probe_t::completed_k;
}

/**
 *  @brief Slide every chiral center's pivot toward the plane of the other three stored neighbors.
 *
 *  At @p retained zero the signed volume is exactly zero, which a sign test cannot classify at all; at a
 *  small positive fraction the sign is still the descriptor's and only the magnitude has collapsed, the
 *  case a sign test passes and a volume window rejects. The atom moved is `neighbor_atoms[3]`, the point
 *  the volume is measured from, which for a three-coordinate center is the center itself.
 *  @param[inout] coordinates Four-axis embedding, modified in place.
 *  @param[in] retained Fraction of the out-of-plane height each center keeps.
 */
static void flatten_chiral_centers_(sft::embedding_soa_t coordinates,
                                    custd::span<sft::chiral_center_descriptor_t const> centers, float retained) {
    for (auto const &center : centers) {
        std::uint32_t const origin = center.neighbor_atoms[0];
        std::uint32_t const pivot = center.neighbor_atoms[3];
        float first_edge[3] = {}, second_edge[3] = {}, normal[3] = {};
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            first_edge[axis] = coordinates(center.neighbor_atoms[1], axis) - coordinates(origin, axis);
            second_edge[axis] = coordinates(center.neighbor_atoms[2], axis) - coordinates(origin, axis);
        }
        normal[0] = first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1];
        normal[1] = first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2];
        normal[2] = first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0];
        float const length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        if (length < 1e-6f) continue; // three collinear neighbors span no plane to project onto

        float height = 0.0f;
        for (std::uint32_t axis = 0; axis < 3; ++axis)
            height += (coordinates(pivot, axis) - coordinates(origin, axis)) * normal[axis] / length;
        for (std::uint32_t axis = 0; axis < 3; ++axis)
            coordinates(pivot, axis) -= (1.0f - retained) * height * normal[axis] / length;
    }
}

/** Euclidean norm of a gradient over every axis, which is zero exactly when nothing pushed back. */
static double gradient_norm_(gradient_t gradient) {
    double total = 0.0;
    for (std::uint32_t atom = 0; atom < gradient.extent(0); ++atom)
        for (std::uint32_t axis = 0; axis < gradient.extent(1); ++axis)
            total += static_cast<double>(gradient(atom, axis)) * gradient(atom, axis);
    return std::sqrt(total);
}

/**
 *  @brief Gate that a chiral center has to prove it is not flat.
 *
 *  Three geometries per molecule, distance penalties off so every number belongs to the chiral term. The
 *  embedded conformer must validate and carry no penalty; pressed exactly onto its neighbors' plane it
 *  must fail validation, a signed volume of zero satisfying neither target, and carry both an energy and
 *  a restoring force; the near-flat case, whose sign is still correct, must do the same, since a sign-only
 *  test reads it as healthy and leaves no force at all. The chiral gradient is then swept against a
 *  central difference well inside the penalty's active branch, so it never crosses the window's kink.
 *  @return Whether every launch ran, so a refusal reaches the gate instead of being compared.
 */
static sftest::dg_probe_t check_chirality_(sftest::diagnostic_reporter_t &reporter, molecule_case_t const &subject,
                                           RDKit::ROMol const &molecule, molecule_scratch_t &scratch) {
    auto const centers = subject.chiral_centers;
    if (centers.empty()) return sftest::dg_probe_t::completed_k;

    std::uint32_t const num_atoms = subject.bounds.extent(0);
    auto const coordinates = reset_embedding_(scratch.embedding, num_atoms);
    auto const gradient = reset_gradient_(scratch.gradient, num_atoms);
    auto const spare = reset_gradient_(scratch.spare_gradient, num_atoms);
    auto const interleaved = reset_interleaved_(scratch.interleaved, num_atoms);
    auto const expected = static_cast<std::uint32_t>(centers.size());

    dg_weights_t const chiral_only {.upper = 0.0f, .lower = 0.0f, .chiral = chiral_weight_k, .fourth_dimension = 0.0f};
    auto const problem = problem_for_(subject.bounds, centers, chiral_only);

    // Reload the embedded conformer and press its centers toward flatness by the given fraction.
    auto const reload_ = [&](float retained) {
        sftest::read_conformer_into(molecule, interleaved);
        sft::transpose_into(interleaved, coordinates);
        flatten_chiral_centers_(coordinates, centers, retained);
    };

    sftest::read_conformer_into(molecule, interleaved);
    sft::transpose_into(interleaved, coordinates);

    float upright_energy = 0.0f;
    sftest::dg_probe_t probe = gpu_dg_energy(problem, coordinates.data_handle(), gradient.data_handle(),
                                             &upright_energy);
    if (probe != sftest::dg_probe_t::completed_k) return probe;
    std::uint32_t upright_invalid = 0;
    probe = gpu_count_invalid_chiral_centers(problem, coordinates.data_handle(), upright_invalid);
    if (probe != sftest::dg_probe_t::completed_k) return probe;
    reporter.mismatch("dg_chirality", "smiles={} centers={} geometry=upright", subject.smiles, expected)
        .field("invalid_centers", 0u, upright_invalid, sftest::exactly_equal)
        .field("energy", 0.0f, upright_energy, sftest::exactly_equal);

    for (float const retained : {0.0f, near_flat_retention_k}) {
        reload_(retained);
        float energy = 0.0f;
        probe = gpu_dg_energy(problem, coordinates.data_handle(), gradient.data_handle(), &energy);
        if (probe != sftest::dg_probe_t::completed_k) return probe;
        std::uint32_t invalid = 0;
        probe = gpu_count_invalid_chiral_centers(problem, coordinates.data_handle(), invalid);
        if (probe != sftest::dg_probe_t::completed_k) return probe;
        reporter.mismatch("dg_chirality", "smiles={} centers={} retained={}", subject.smiles, expected, retained)
            .field("invalid_centers", expected, invalid, sftest::exactly_equal)
            .field("energy", 0.0f, energy, increased_)
            .field("restoring_force", 0.0, gradient_norm_(gradient), increased_);
    }

    // Finite differences of the chiral term, over the atoms it actually touches.
    reload_(penalized_retention_k);
    float energy_at_center = 0.0f;
    probe = gpu_dg_energy(problem, coordinates.data_handle(), gradient.data_handle(), &energy_at_center);
    if (probe != sftest::dg_probe_t::completed_k) return probe;

    // The sweep means something only inside the penalty's active branch; outside it `excess_volume` is
    // zero and every comparison passes having exercised nothing. Under chiral-only weights a positive
    // energy is exactly the statement that the flattened centers are still being penalised.
    reporter.mismatch("dg_chirality", "smiles={} retained={}", subject.smiles, penalized_retention_k)
        .field("chirality_penalty_is_active", true, energy_at_center > 0.0f, sftest::exactly_equal);
    for (auto const &center : centers)
        for (std::uint8_t const atom :
             {center.neighbor_atoms[0], center.neighbor_atoms[1], center.neighbor_atoms[2], center.neighbor_atoms[3]})
            for (std::uint32_t axis = 0; axis < 3; ++axis) {
                float const original = coordinates(atom, axis);

                coordinates(atom, axis) = original + finite_difference_step_k;
                float energy_plus = 0.0f;
                probe = gpu_dg_energy(problem, coordinates.data_handle(), spare.data_handle(), &energy_plus);

                coordinates(atom, axis) = original - finite_difference_step_k;
                float energy_minus = 0.0f;
                if (probe == sftest::dg_probe_t::completed_k)
                    probe = gpu_dg_energy(problem, coordinates.data_handle(), spare.data_handle(), &energy_minus);

                coordinates(atom, axis) = original;
                // Restored first, so a refusal still leaves the borrowed coordinates as the sweep found them.
                if (probe != sftest::dg_probe_t::completed_k) return probe;

                float const finite_difference = (energy_plus - energy_minus) / (2.0f * finite_difference_step_k);
                reporter.mismatch("dg_chirality", "smiles={} atom={} axis={}", subject.smiles, atom, axis)
                    .field("chiral_gradient", finite_difference, gradient(atom, axis), gradients_agree_);
            }
    return sftest::dg_probe_t::completed_k;
}

/**
 *  @brief Gate that FIRE minimization improves what it is asked to improve.
 *
 *  A random 4D start is minimized and each of the three quantities the energy is built from must end below
 *  where it began: the energy, the worst bound violation, and the mean fourth-dimension magnitude the
 *  crush penalty flattens. The MMFF94 energy of the projected structure is then compared for finiteness
 *  against RDKit's own embedding, making it a reference rather than an `isfinite` any garbage satisfies.
 *  @return Whether every launch ran, so a refusal reaches the gate instead of being compared.
 */
static sftest::dg_probe_t check_minimization_(sftest::diagnostic_reporter_t &reporter, molecule_case_t const &subject,
                                              RDKit::RWMol &molecule, dg_weights_t weights,
                                              molecule_scratch_t &scratch) {
    auto const bounds = subject.bounds;
    auto const &smiles = subject.smiles;
    std::uint32_t const num_atoms = bounds.extent(0);
    auto const coordinates = reset_embedding_(scratch.embedding, num_atoms);
    auto const gradient = reset_gradient_(scratch.gradient, num_atoms);
    auto const interleaved = reset_interleaved_(scratch.interleaved, num_atoms);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> position(-random_start_span_k, random_start_span_k);
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
        for (std::uint32_t dimension = 0; dimension < coordinates.extent(1); ++dimension)
            coordinates(atom, dimension) = position(generator);

    double const mmff_at_embedding = sftest::rdkit_mmff94_energy(molecule);
    auto const problem = problem_for_(bounds, subject.chiral_centers, weights);

    float start_energy = 0.0f;
    sftest::dg_probe_t probe = gpu_dg_energy(problem, coordinates.data_handle(), gradient.data_handle(), &start_energy);
    if (probe != sftest::dg_probe_t::completed_k) return probe;
    float const start_violation = max_bound_violation_(coordinates, bounds);
    float const start_fourth = mean_fourth_dimension_magnitude_(coordinates);

    sftest::dg_minimize_report_t report;
    probe = gpu_dg_minimize(problem, coordinates.data_handle(), max_iterations_k, gradient_threshold_k, &report);
    if (probe != sftest::dg_probe_t::completed_k) return probe;
    float const final_energy = report.energy;

    float const final_violation = max_bound_violation_(coordinates, bounds);
    float const final_fourth = mean_fourth_dimension_magnitude_(coordinates);

    // The report must describe the positions handed back, not a state one step earlier, so both of its
    // numbers are checked against a fresh evaluation exactly where it stopped.
    float recomputed_energy = 0.0f;
    probe = gpu_dg_energy(problem, coordinates.data_handle(), gradient.data_handle(), &recomputed_energy);
    if (probe != sftest::dg_probe_t::completed_k) return probe;
    double const element_count = static_cast<double>(coordinates.extent(0)) * coordinates.extent(1);
    double const recomputed_rms = gradient_norm_(gradient) / std::sqrt(element_count);

    sft::transpose_into(coordinates, interleaved);
    sftest::set_coords_interleaved(molecule, interleaved.data_handle());
    double const mmff_at_dg_coords = sftest::rdkit_mmff94_energy(molecule);

    reporter
        .mismatch("dg_minimize", "smiles={} atoms={} status={} iterations={}", smiles, num_atoms,
                  fire_status_name_(report.status), report.iterations)
        .field("energy", start_energy, final_energy, decreased_)
        .field("max_violation", start_violation, final_violation, decreased_)
        .field("fourth_dimension_magnitude", start_fourth, final_fourth, decreased_)
        .field("reported_energy", recomputed_energy, final_energy, energies_agree_)
        .field("reported_rms_gradient", recomputed_rms, report.rms_gradient, rms_gradients_agree_)
        .field("mmff94_energy", mmff_at_embedding, mmff_at_dg_coords, both_finite_);
    return sftest::dg_probe_t::completed_k;
}

/**
 *  @brief Relax from a well-posed start and require the minimizer to report that it converged.
 *
 *  `check_minimization_` demands only that energy, violation and fourth-dimension magnitude fall, which
 *  one downhill step satisfies, so it cannot tell a minimizer that finishes from one that stops early.
 *  Here the start is RDKit's own embedding displaced by `perturbation_span_angstrom_k` per coordinate, a
 *  basin the distance bounds were built from, so `exhausted_k` inside `max_iterations_k` means the budget
 *  is short rather than the problem hard. Status and gradient are both gated: `fire_minimize_` returns
 *  `converged_k` exactly when the RMS gradient fell under its threshold, and asserting the pair keeps
 *  that equivalence checked rather than assumed.
 *  @return Whether every launch ran, so a refusal reaches the gate instead of being compared.
 */
static sftest::dg_probe_t check_convergence_(sftest::diagnostic_reporter_t &reporter, molecule_case_t const &subject,
                                             RDKit::RWMol &molecule, dg_weights_t weights,
                                             molecule_scratch_t &scratch) {
    auto const bounds = subject.bounds;
    auto const &smiles = subject.smiles;
    std::uint32_t const num_atoms = bounds.extent(0);
    auto const coordinates = reset_embedding_(scratch.embedding, num_atoms);
    auto const interleaved = reset_interleaved_(scratch.interleaved, num_atoms);

    sftest::read_conformer_into(molecule, interleaved);
    sft::transpose_into(interleaved, coordinates);
    fill_fourth_dimension_(coordinates);

    // A fixed generator, so a failure here is reproducible rather than a run the next one cannot repeat.
    std::mt19937 generator(4242);
    std::uniform_real_distribution<float> displacement(-perturbation_span_angstrom_k, perturbation_span_angstrom_k);
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
        for (std::uint32_t dimension = 0; dimension < 3; ++dimension)
            coordinates(atom, dimension) += displacement(generator);

    auto const problem = problem_for_(bounds, subject.chiral_centers, weights);
    sftest::dg_minimize_report_t report;
    sftest::dg_probe_t const probe = gpu_dg_minimize(problem, coordinates.data_handle(), max_iterations_k,
                                                     gradient_threshold_k, &report);
    if (probe != sftest::dg_probe_t::completed_k) return probe;

    // The status rides in the key where it is greppable: `append_delta` needs subtractable operands, so a
    // name cannot be the compared value.
    reporter
        .mismatch("dg_convergence", "smiles={} atoms={} status={} iterations={}", smiles, num_atoms,
                  fire_status_name_(report.status), report.iterations)
        .field("converged", true, report.status == smallfield::cuda::fire_status_t::converged_k, sftest::exactly_equal)
        .field("rms_gradient", static_cast<double>(gradient_threshold_k), static_cast<double>(report.rms_gradient),
               at_most_);
    return sftest::dg_probe_t::completed_k;
}

#pragma endregion Molecule Checks

/**
 *  @brief The molecules the bare `ctest` invocation walks.
 *
 *  Small and always embeddable: an aromatic ring, and four stereocenters covering the shapes the chiral
 *  window distinguishes -- an open-chain carbon, one in a ring, two adjacent, and a three-coordinate
 *  sulfoxide sulfur whose fourth substituent is a lone pair.
 */
static std::vector<std::string> builtin_smiles_() {
    return {"CCO",
            "CC(=O)O",
            "c1ccccc1",
            "C[C@H](N)C(=O)O",
            "O[C@H]1CC[C@@H](O)CC1",
            "C[C@@H](C(=O)O)[C@@H](C)C(=O)O",
            "C[S@](=O)c1ccccc1"};
}

int main(int argc, char **argv) {
    fmt::println("{}: four-dimensional DG energy, gradient, chirality, and FIRE minimization against a host mirror",
                 test_name_k);

    auto const selection = sftest::select_corpus(argc, argv, 25);
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", argv[0]);
        return 2;
    }

    std::vector<std::string> smiles_list = builtin_smiles_();
    if (selection.knob == sftest::corpus_knob_t::named_k) {
        std::vector<std::string> streamed;
        auto const outcome = sftest::load_smiles_into(selection.path, selection.max_molecules, streamed);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), selection.path);
        fmt::println("Corpus: {} ({} molecules) plus built-ins", selection.path, fmt::group_digits(streamed.size()));
        for (auto &smiles : streamed) smiles_list.push_back(std::move(smiles));
    }
    else fmt::println("Corpus: built-ins only; name one on the command line or in SMALLFIELD_TEST_CORPUS");

    dg_reporters_t reporters;
    reporters.limit_printed_lines(max_printed_lines_k);

    molecule_scratch_t scratch;
    sft::molecule_descriptor_t descriptor;
    std::unique_ptr<RDKit::RWMol> molecule;
    std::size_t ready_molecules = 0, skipped_molecules = 0, gradient_molecules = 0, chiral_molecules = 0;
    std::size_t convergence_molecules = 0;
    std::size_t refused_sections = 0;

    // A declined launch leaves the kernel's outputs as the allocation found them, so a section that met
    // one names the refusal rather than closing as though it had compared something.
    auto const record_ = [&refused_sections](std::string_view section, std::string const &smiles,
                                             sftest::dg_probe_t probe) {
        if (probe == sftest::dg_probe_t::completed_k) return;
        fmt::println("  NOTE {} device_probe={} smiles={}", section, sftest::dg_probe_reason(probe), smiles);
        ++refused_sections;
    };

    // Without the crush penalty the fourth axis only enters through the pair distances; with it, the term
    // that collapses the embedding is live. Both are gated, because the kernel branches on the weight.
    dg_weights_t const flat_weights {.upper = 1.0f, .lower = 1.0f, .chiral = 0.0f, .fourth_dimension = 0.0f};
    dg_weights_t const crushed_weights {.upper = 1.0f,
                                        .lower = 1.0f,
                                        .chiral = 0.0f,
                                        .fourth_dimension = fourth_dimension_weight_k};

    // Minimization runs with the chiral term the pipeline carries; without it a random 4D start can relax
    // into coincident atoms, a degenerate minimum of the distance penalties alone.
    dg_weights_t const minimize_weights {.upper = 1.0f,
                                         .lower = 1.0f,
                                         .chiral = chiral_weight_k,
                                         .fourth_dimension = fourth_dimension_weight_k};

    for (auto const &smiles : smiles_list) {
        molecule_setup_t const outcome = setup_molecule_(smiles, molecule, descriptor);
        if (outcome != molecule_setup_t::ready_k) {
            fmt::println("  NOTE dg_setup smiles={} outcome={}", smiles, molecule_setup_reason_(outcome));
            ++skipped_molecules;
            continue;
        }
        ++ready_molecules;

        auto const bounds = reset_bounds_matrix_(descriptor, scratch.bounds);
        sf_verify_(bounds.extent(0) == molecule->getNumAtoms(),
                   "the descriptor and the RDKit molecule disagree on the atom count");
        sfp::extract_chiral_centers(*molecule, descriptor, scratch.chiral_centers);
        molecule_case_t const subject {.smiles = smiles, .bounds = bounds, .chiral_centers = {scratch.chiral_centers}};

        record_("dg_energy", smiles,
                check_energy_(reporters.energy, reporters.quantization, subject, *molecule, flat_weights, scratch));
        record_("dg_energy", smiles,
                check_energy_(reporters.energy, reporters.quantization, subject, *molecule, crushed_weights, scratch));

        if (gradient_molecules < max_gradient_molecules_k) {
            record_("dg_gradient", smiles,
                    check_gradient_(reporters.gradient, subject, *molecule, crushed_weights, scratch));
            ++gradient_molecules;
        }

        if (!subject.chiral_centers.empty() && chiral_molecules < max_chiral_molecules_k) {
            record_("dg_chirality", smiles, check_chirality_(reporters.chirality, subject, *molecule, scratch));
            ++chiral_molecules;
        }

        // Before the random-start section, which overwrites the conformer this one reads.
        if (convergence_molecules < max_convergence_molecules_k) {
            record_("dg_convergence", smiles,
                    check_convergence_(reporters.convergence, subject, *molecule, minimize_weights, scratch));
            ++convergence_molecules;
        }

        // Last per molecule: it overwrites the conformer the other sections read.
        record_("dg_minimize", smiles,
                check_minimization_(reporters.minimize, subject, *molecule, minimize_weights, scratch));
    }

    fmt::println("  {} molecules walked, {} skipped, {} swept for finite differences, {} flattened at their centers",
                 fmt::group_digits(ready_molecules), fmt::group_digits(skipped_molecules),
                 fmt::group_digits(gradient_molecules), fmt::group_digits(chiral_molecules));
    sf_verify_(ready_molecules > 0, "no molecule survived setup, so nothing was compared");
    sf_verify_(chiral_molecules > 0, "no molecule carried a chiral center, so the volume window was never exercised");

    reporters.energy.summarize("dg_energy");
    reporters.gradient.summarize("dg_gradient");
    reporters.chirality.summarize("dg_chirality");
    reporters.minimize.summarize("dg_minimize");
    reporters.convergence.summarize("dg_convergence");

    if (refused_sections != 0)
        return sftest::report_fail(test_name_k, "the device refused {} sections before either operand existed",
                                   refused_sections);
    if (!reporters.passed())
        return sftest::report_fail(
            test_name_k, "{} energy, {} gradient, {} chirality, {} minimization, and {} convergence mismatches",
            reporters.energy.mismatch_count(), reporters.gradient.mismatch_count(),
            reporters.chirality.mismatch_count(), reporters.minimize.mismatch_count(),
            reporters.convergence.mismatch_count());
    return sftest::report_pass(test_name_k);
}
