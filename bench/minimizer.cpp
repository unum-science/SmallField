/**
 *  @file bench/minimizer.cpp
 *  @brief Separate an algorithmic stall from an unreachable threshold from too small a budget, in the MMFF94 polish.
 *
 *  Production telemetry reports the Barzilai-Borwein polish exhausting its whole budget on essentially every slot at
 *  an RMS gradient three decades above its threshold. Three explanations fit that number and one measurement set
 *  separates them: sweeping the budget says whether the residual is a plateau or a decline, and re-evaluating our
 *  final geometry through RDKit's double-precision MMFF94 says whether our `float` gradient measures the geometry or
 *  its own rounding. Every molecule starts from RDKit's own ETKDGv3 embedding, rounded once to `float` and written
 *  back into the conformer, so our minimizer and RDKit's start from bit-identical coordinates. The two convergence
 *  tests are @b not the same quantity and are never presented as one: ours is an absolute root mean square over all
 *  `3N` gradient components, RDKit's `ForceField::minimize` tests a scaled residual against `forceTol`, so its
 *  default of `1e-4` is not comparable to ours of `5e-5` and only gradient norms at the same geometry are.
 */

#include <cmath>   // `std::sqrt`, `std::fabs`, `std::isfinite`, `std::log10`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::int32_t`
#include <cstdio>  // `stderr`

#include <algorithm>   // `std::sort`, `std::max`
#include <array>       // `std::array`
#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <ForceField/ForceField.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>

#include "smallfield/cuda/fire_warp.cuh"
#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/types/views.hpp"

#include "bench/arguments.hpp"
#include "test/cuda/mmff_gradient_kernels.hpp"
#include "test/rdkit_reference.hpp"

namespace sfc = smallfield::cuda;
namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

namespace smallfield::bench {

static constexpr std::string_view executable_name_k = "smallfield_bench_minimizer";

#pragma region Sweep Points

/** Iteration budgets the sweep runs each molecule at, from the production cap outward in both directions. */
static constexpr std::array<std::int32_t, 5> iteration_caps_k = {200, 800, 3200, 12800, 51200};

/** Index into `iteration_caps_k` of the production cap, which the precision probe and the energy gap use. */
static constexpr std::size_t production_cap_index_k = 1;

/** Threshold handed to the sweep, low enough that no cap converges out from under the measurement. The question is
 *  where the residual lands, so a run stopping early would answer with the threshold rather than the minimizer;
 *  production's `5e-5` is reported separately, as the bar each measured residual is compared against. */
static constexpr float sweep_gradient_threshold_k = 1.0e-12f;

/** Production's own bar, quoted beside every residual so the comparison is visible rather than remembered. */
static constexpr float production_gradient_threshold_k = 5.0e-5f;

/** Iterations RDKit is allowed per `minimize` call, which is also its own documented default. */
static constexpr unsigned int rdkit_iterations_per_round_k = 200;

/** Rounds of `rdkit_iterations_per_round_k` RDKit may spend, matching the sweep's largest budget. */
static constexpr unsigned int rdkit_max_rounds_k = 64;

/** RDKit's default scaled force tolerance, which is a different test from our absolute RMS and stays named. */
static constexpr double rdkit_force_tolerance_k = 1.0e-4;

#pragma endregion Sweep Points

#pragma region Records

/** Everything one molecule contributed, so the four tables are assembled from one pass rather than four. Arrays are
 *  indexed by `iteration_caps_k`, and every gradient is an absolute RMS over all `3N` components in kcal per mol per
 *  angstrom — ours in `float`, RDKit's in double. The names say which geometry each was measured at:
 *  `_start` is the untouched ETKDGv3 embedding, where the two definitions are gated against each other before
 *  anything moves; `_at_ours` is our relaxed geometry, which is the precision probe; `_at_rdkit_minimum` is the
 *  geometry RDKit's own minimizer stopped on, the floor any threshold of ours has to clear. `final_step` is the last
 *  Barzilai-Borwein step, saying whether a clamp is binding, `iterations` equals its cap unless the run converged,
 *  and `rdkit_rounds` counts `rdkit_iterations_per_round_k` blocks spent before RDKit stopped asking for more.
 */
struct molecule_record_t {
    std::uint32_t num_atoms = 0;
    std::array<float, iteration_caps_k.size()> rms_gradient {};
    std::array<float, iteration_caps_k.size()> energy {};
    std::array<float, iteration_caps_k.size()> final_step {};
    std::array<std::int32_t, iteration_caps_k.size()> iterations {};
    std::array<sfc::fire_status_t, iteration_caps_k.size()> status {};
    float rms_gradient_start = 0.0f;
    double rdkit_rms_gradient_start = 0.0;
    double rdkit_rms_gradient_at_ours = 0.0;
    double rdkit_energy_at_ours = 0.0;
    double rdkit_final_energy = 0.0;
    double rdkit_final_rms_gradient = 0.0;
    float our_rms_gradient_at_rdkit_minimum = 0.0f;
    unsigned int rdkit_rounds = 0;
    bool rdkit_converged = false;
};

/** What the sweep stepped past, threaded by reference so the run owns no module state. A molecule counts as
 *  `unprepared` when preprocessing refuses the SMILES, `unembedded` when ETKDGv3 yields no conformer, `unaligned`
 *  when RDKit's atom count differs from preprocessing's so the orders cannot be matched, `unreferenced` when RDKit
 *  cannot build MMFF94 properties, and `refused` when the device refuses a launch; `first_refusal` names the step,
 *  which the counts alone cannot. */
struct sweep_tally_t {
    std::uint32_t measured = 0;
    std::uint32_t unprepared = 0;
    std::uint32_t unembedded = 0;
    std::uint32_t unaligned = 0;
    std::uint32_t unreferenced = 0;
    std::uint32_t refused = 0;
    sftest::mmff_energy_probe_t first_refusal = sftest::mmff_energy_probe_t::completed_k;
};

#pragma endregion Records

#pragma region Statistics

/** Quantile of an already-sorted sample, by nearest rank; an empty sample answers zero. */
static double sorted_quantile_(std::vector<double> const &sorted, double fraction) {
    if (sorted.empty()) return 0.0;
    std::size_t const rank = static_cast<std::size_t>(fraction * static_cast<double>(sorted.size() - 1) + 0.5);
    return sorted[rank];
}

/** Pearson correlation of two equal-length samples; fewer than two points, or a constant one, answers zero. */
static double pearson_correlation_(std::vector<double> const &first, std::vector<double> const &second) {
    std::size_t const count = std::min(first.size(), second.size());
    if (count < 2) return 0.0;
    double first_mean = 0.0, second_mean = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        first_mean += first[index];
        second_mean += second[index];
    }
    first_mean /= static_cast<double>(count);
    second_mean /= static_cast<double>(count);
    double covariance = 0.0, first_variance = 0.0, second_variance = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        double const first_deviation = first[index] - first_mean;
        double const second_deviation = second[index] - second_mean;
        covariance += first_deviation * second_deviation;
        first_variance += first_deviation * first_deviation;
        second_variance += second_deviation * second_deviation;
    }
    if (first_variance <= 0.0 || second_variance <= 0.0) return 0.0;
    return covariance / std::sqrt(first_variance * second_variance);
}

/** Print the median and the 90th percentile of @p sample under @p label, sorting it in place. */
static void print_quantiles_(std::string_view label, std::vector<double> &sample) {
    std::sort(sample.begin(), sample.end());
    fmt::println("  {:<28} n={:<5} min={:<12.4g} median={:<12.4g} p90={:<12.4g} max={:.4g}", label, sample.size(),
                 sorted_quantile_(sample, 0.0), sorted_quantile_(sample, 0.5), sorted_quantile_(sample, 0.9),
                 sorted_quantile_(sample, 1.0));
}

#pragma endregion Statistics

#pragma region RDKit Reference

/** Root mean square over all `3N` components of an RDKit double-precision MMFF94 gradient. */
static double rdkit_rms_gradient_(ForceFields::ForceField &force_field, std::vector<double> &scratch) {
    std::size_t const num_components = 3 * force_field.numPoints();
    scratch.assign(num_components, 0.0);
    force_field.calcGrad(scratch.data());
    double sum_of_squares = 0.0;
    for (double const component : scratch) sum_of_squares += component * component;
    return std::sqrt(sum_of_squares / static_cast<double>(num_components));
}

/** Write axis-major `float` coordinates onto a molecule's conformer, widening once. Both engines then read the same
 *  numbers — we hold these `float` values exactly, RDKit their exact double widening — so nothing in the comparison
 *  is a rounding of the other side's start. */
static void write_axis_major_into_conformer_(RDKit::RWMol &molecule, custd::span<float const> axis_major) {
    auto &conformer = molecule.getConformer();
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
        conformer.setAtomPos(atom, RDGeom::Point3D(static_cast<double>(axis_major[atom]),
                                                   static_cast<double>(axis_major[num_atoms + atom]),
                                                   static_cast<double>(axis_major[2 * num_atoms + atom])));
}

#pragma endregion RDKit Reference

#pragma region Sweep

/** Buffers every molecule reuses, declared above the loop so the growth phase happens once; `relaxed_axis_major`
 *  holds our geometry at the production cap, kept for the precision probe. */
struct sweep_scratch_t {
    sfp::etkdg_prepared_molecule_t prepared {};
    std::vector<float> atom_major;
    std::vector<float> start_axis_major;
    std::vector<float> working_axis_major;
    std::vector<float> relaxed_axis_major;
    std::vector<std::uint8_t> bond_hops;
    std::vector<double> gradient_components;
};

/** How one molecule's turn ended, so the tally names the reason rather than counting a bare skip. */
enum class molecule_outcome_t : std::uint8_t {
    measured_k,
    unprepared_k,
    unembedded_k,
    unaligned_k,
    unreferenced_k,
    refused_k,
};

/**
 *  @brief Run one molecule through the budget sweep, RDKit's own minimizer, and the precision probe.
 *
 *  @param[inout] scratch Every buffer the molecule needs, carried in from above the loop.
 *  @param[out] record Filled only when the outcome is `measured_k`.
 *  @param[inout] tally Receives the step a device refusal came from, which the outcome alone cannot carry.
 */
static molecule_outcome_t measure_molecule_(std::string const &smiles, sweep_scratch_t &scratch,
                                            molecule_record_t &record, sweep_tally_t &tally) {
    // Keep the first refusal the run met, so the report can name the step that refused.
    auto const refuse_ = [&tally](sftest::mmff_energy_probe_t probe) {
        if (tally.first_refusal == sftest::mmff_energy_probe_t::completed_k) tally.first_refusal = probe;
        return molecule_outcome_t::refused_k;
    };
    sfp::prepare_molecule_into(smiles, scratch.prepared);
    if (scratch.prepared.num_atoms == 0) return molecule_outcome_t::unprepared_k;

    std::unique_ptr<RDKit::RWMol> molecule = sftest::mol_from_smiles(smiles);
    if (!molecule) return molecule_outcome_t::unprepared_k;
    if (!sftest::embed_molecule(*molecule)) return molecule_outcome_t::unembedded_k;

    std::uint32_t const num_atoms = molecule->getNumAtoms();
    if (num_atoms != scratch.prepared.num_atoms) return molecule_outcome_t::unaligned_k;
    record.num_atoms = num_atoms;

    // Read out of RDKit, rounded once to `float`, written straight back: both engines relax the same numbers.
    auto const atom_major = sft::reset_coordinates_aos<sft::coordinate_role_t::position_k>(scratch.atom_major,
                                                                                           num_atoms);
    sftest::read_conformer_into(*molecule, atom_major);
    sft::conformer_soa_t const start_axis_major =
        sft::reset_coordinates_soa<sft::coordinate_role_t::position_k, sft::conformer_dimensions_k>(
            scratch.start_axis_major, num_atoms);
    sft::transpose_into(atom_major, start_axis_major);
    write_axis_major_into_conformer_(*molecule, scratch.start_axis_major);

    std::unique_ptr<RDKit::MMFF::MMFFMolProperties> properties(new RDKit::MMFF::MMFFMolProperties(*molecule));
    if (!properties || !properties->isValid()) return molecule_outcome_t::unreferenced_k;

    auto const bond_hops = sft::reset_square_matrix(scratch.bond_hops, num_atoms, sft::unreachable_hops_k);
    sftest::fill_bond_hops_into(scratch.prepared.mmff_bonds, bond_hops);
    sftest::mmff_energy_terms_t const terms {
        .atom_vdw = scratch.prepared.mmff_atom_vdw,
        .atom_charge = scratch.prepared.mmff_atom_charge,
        .bond_distance = bond_hops,
        .bonds = scratch.prepared.mmff_bonds,
        .angles = scratch.prepared.mmff_angles,
        .torsions = scratch.prepared.mmff_torsions,
        .oops = scratch.prepared.mmff_oops,
        .stretchbends = scratch.prepared.mmff_stretchbends,
        .interactions = scratch.prepared.mmff_interactions,
        .atom_offsets = scratch.prepared.mmff_atom_offsets,
    };

    // A budget of one evaluates and measures without stepping, which is the residual at the start itself.
    sftest::mmff_minimize_result_t opening {};
    scratch.working_axis_major = scratch.start_axis_major;
    if (sftest::mmff_energy_probe_t const probe = sftest::gpu_mmff_minimize(
            scratch.working_axis_major, num_atoms, terms, 1, sweep_gradient_threshold_k, opening);
        probe != sftest::mmff_energy_probe_t::completed_k)
        return refuse_(probe);
    record.rms_gradient_start = opening.rms_gradient;

    for (std::size_t point = 0; point < iteration_caps_k.size(); ++point) {
        scratch.working_axis_major = scratch.start_axis_major;
        sftest::mmff_minimize_result_t result {};
        if (sftest::mmff_energy_probe_t const probe = sftest::gpu_mmff_minimize(scratch.working_axis_major, num_atoms,
                                                                                terms, iteration_caps_k[point],
                                                                                sweep_gradient_threshold_k, result);
            probe != sftest::mmff_energy_probe_t::completed_k)
            return refuse_(probe);
        record.rms_gradient[point] = result.rms_gradient;
        record.energy[point] = result.energy;
        record.final_step[point] = result.final_step;
        record.iterations[point] = result.iterations;
        record.status[point] = result.status;
        if (point == production_cap_index_k) scratch.relaxed_axis_major = scratch.working_axis_major;
    }

    // The control: differing here would make every later comparison one of two force fields, not two minimizers.
    {
        std::unique_ptr<ForceFields::ForceField> field(RDKit::MMFF::constructForceField(*molecule, properties.get()));
        if (!field) return molecule_outcome_t::unreferenced_k;
        field->initialize();
        record.rdkit_rms_gradient_start = rdkit_rms_gradient_(*field, scratch.gradient_components);
    }

    // The precision probe: RDKit's double MMFF94 at the geometry our `float` minimizer stopped on, so the residual
    // is measured twice at one geometry rather than once at each of two.
    write_axis_major_into_conformer_(*molecule, scratch.relaxed_axis_major);
    {
        std::unique_ptr<ForceFields::ForceField> field(RDKit::MMFF::constructForceField(*molecule, properties.get()));
        if (!field) return molecule_outcome_t::unreferenced_k;
        field->initialize();
        record.rdkit_rms_gradient_at_ours = rdkit_rms_gradient_(*field, scratch.gradient_components);
        record.rdkit_energy_at_ours = field->calcEnergy();
    }

    // RDKit relaxing the same start, which bounds what this problem admits. `minimize` restarts its
    // conjugate-gradient state per call, so the rounds repeat its default budget and the iteration figure is a ceiling.
    write_axis_major_into_conformer_(*molecule, scratch.start_axis_major);
    {
        std::unique_ptr<ForceFields::ForceField> field(RDKit::MMFF::constructForceField(*molecule, properties.get()));
        if (!field) return molecule_outcome_t::unreferenced_k;
        field->initialize();
        while (record.rdkit_rounds < rdkit_max_rounds_k) {
            ++record.rdkit_rounds;
            if (field->minimize(rdkit_iterations_per_round_k, rdkit_force_tolerance_k) == 0) {
                record.rdkit_converged = true;
                break;
            }
        }
        record.rdkit_final_energy = field->calcEnergy();
        record.rdkit_final_rms_gradient = rdkit_rms_gradient_(*field, scratch.gradient_components);
    }

    // Our `float` evaluator at RDKit's converged geometry, budget one so it measures without stepping: whatever we
    // report on a structure RDKit calls relaxed is the floor our absolute test can distinguish from zero.
    {
        sftest::read_conformer_into(*molecule, atom_major);
        sft::transpose_into(atom_major, sft::conformer_soa_t {scratch.working_axis_major.data(), num_atoms});
        sftest::mmff_minimize_result_t at_minimum {};
        if (sftest::mmff_energy_probe_t const probe = sftest::gpu_mmff_minimize(
                scratch.working_axis_major, num_atoms, terms, 1, sweep_gradient_threshold_k, at_minimum);
            probe != sftest::mmff_energy_probe_t::completed_k)
            return refuse_(probe);
        record.our_rms_gradient_at_rdkit_minimum = at_minimum.rms_gradient;
    }

    if (!std::isfinite(record.rdkit_final_energy) || !std::isfinite(record.rdkit_rms_gradient_at_ours))
        return molecule_outcome_t::unreferenced_k;
    return molecule_outcome_t::measured_k;
}

/** Walk the corpus, keeping one record per measured molecule; the run completes whatever it finds. */
static void sweep_corpus_(std::vector<std::string> const &corpus, std::vector<molecule_record_t> &records,
                          sweep_tally_t &tally) {
    sweep_scratch_t scratch;
    molecule_record_t record;
    for (std::string const &smiles : corpus) {
        record = molecule_record_t {};
        switch (measure_molecule_(smiles, scratch, record, tally)) {
        case molecule_outcome_t::measured_k:
            ++tally.measured;
            records.push_back(record);
            break;
        case molecule_outcome_t::unprepared_k: ++tally.unprepared; break;
        case molecule_outcome_t::unembedded_k: ++tally.unembedded; break;
        case molecule_outcome_t::unaligned_k: ++tally.unaligned; break;
        case molecule_outcome_t::unreferenced_k: ++tally.unreferenced; break;
        case molecule_outcome_t::refused_k: ++tally.refused; break;
        }
    }
}

#pragma endregion Sweep

#pragma region Report

/** Table one: where the residual gradient lands as the budget grows, and how often the budget ran out. */
static void report_iteration_sweep_(std::vector<molecule_record_t> const &records) {
    // Absolute rms gradient over 3N components; the sweep threshold is deliberately unreachable, the production bar
    // printed beside it for scale.
    fmt::println("iteration_sweep: sweep_threshold={:.3g} production_threshold={:.3g}", sweep_gradient_threshold_k,
                 production_gradient_threshold_k);
    std::vector<double> sample;
    for (std::size_t point = 0; point < iteration_caps_k.size(); ++point) {
        sample.clear();
        std::uint32_t converged = 0;
        for (molecule_record_t const &record : records) {
            sample.push_back(record.rms_gradient[point]);
            converged += record.status[point] == sfc::fire_status_t::converged_k;
        }
        std::sort(sample.begin(), sample.end());
        fmt::println("  cap={:<6} median_rms={:<12.4g} p90_rms={:<12.4g} max_rms={:<12.4g} converged={}/{}",
                     iteration_caps_k[point], sorted_quantile_(sample, 0.5), sorted_quantile_(sample, 0.9),
                     sorted_quantile_(sample, 1.0), converged, records.size());
    }

    sample.clear();
    for (molecule_record_t const &record : records) sample.push_back(record.rms_gradient_start);
    print_quantiles_("start_rms_ours", sample);

    // The secant step is clamped into `[1e-6, 1]` and again by the trust region; a last step on either bound is one
    // the curvature estimate did not choose.
    for (std::size_t point = 0; point < iteration_caps_k.size(); ++point) {
        sample.clear();
        for (molecule_record_t const &record : records) sample.push_back(record.final_step[point]);
        print_quantiles_(fmt::format("final_step_at_{}", iteration_caps_k[point]), sample);
    }

    // The ratio between consecutive caps separates a plateau from a decline: a factor near one has stopped moving.
    for (std::size_t point = 1; point < iteration_caps_k.size(); ++point) {
        sample.clear();
        for (molecule_record_t const &record : records)
            if (record.rms_gradient[point] > 0.0f)
                sample.push_back(record.rms_gradient[point - 1] / record.rms_gradient[point]);
        print_quantiles_(fmt::format("decline_{}_over_{}", iteration_caps_k[point - 1], iteration_caps_k[point]),
                         sample);
    }

    for (std::size_t point = 1; point < iteration_caps_k.size(); ++point) {
        sample.clear();
        for (molecule_record_t const &record : records)
            sample.push_back(static_cast<double>(record.energy[point - 1]) - record.energy[point]);
        print_quantiles_(fmt::format("energy_drop_{}_to_{}", iteration_caps_k[point - 1], iteration_caps_k[point]),
                         sample);
    }
}

/** Table two: what RDKit's own minimizer reaches from the identical start. */
static void report_rdkit_reference_(std::vector<molecule_record_t> const &records) {
    fmt::println(
        "rdkit_reference: ForceField::minimize, scaled forceTol {:.3g}, {} iterations per round, " "up to {} rounds",
        rdkit_force_tolerance_k, rdkit_iterations_per_round_k, rdkit_max_rounds_k);
    std::uint32_t converged = 0;
    std::vector<double> rounds, final_gradient, final_energy, start_gradient;
    for (molecule_record_t const &record : records) {
        converged += record.rdkit_converged;
        rounds.push_back(record.rdkit_rounds);
        final_gradient.push_back(record.rdkit_final_rms_gradient);
        final_energy.push_back(record.rdkit_final_energy);
        start_gradient.push_back(record.rdkit_rms_gradient_start);
    }
    fmt::println("  converged={}/{} under its own scaled test", converged, records.size());
    print_quantiles_("rdkit_rounds_of_200", rounds);
    print_quantiles_("rdkit_final_rms_gradient", final_gradient);
    print_quantiles_("rdkit_final_energy", final_energy);
    print_quantiles_("rdkit_start_rms_gradient", start_gradient);
}

/** Table three: our `float` residual and RDKit's double residual, measured at one and the same geometry. */
static void report_precision_probe_(std::vector<molecule_record_t> const &records) {
    fmt::println("precision_probe: rms gradient at OUR cap={} geometry, ours in float against RDKit in double",
                 iteration_caps_k[production_cap_index_k]);
    std::vector<double> ours, theirs, ratio, log_ours, log_theirs, start_ratio;
    for (molecule_record_t const &record : records) {
        double const our_value = record.rms_gradient[production_cap_index_k];
        ours.push_back(our_value);
        theirs.push_back(record.rdkit_rms_gradient_at_ours);
        if (our_value > 0.0) {
            ratio.push_back(record.rdkit_rms_gradient_at_ours / our_value);
            log_ours.push_back(std::log10(our_value));
            log_theirs.push_back(std::log10(std::max(record.rdkit_rms_gradient_at_ours, 1e-30)));
        }
        if (record.rms_gradient_start > 0.0f)
            start_ratio.push_back(record.rdkit_rms_gradient_start / record.rms_gradient_start);
    }
    fmt::println("  pearson_r_on_log10={:.4f} over n={}", pearson_correlation_(log_ours, log_theirs), log_ours.size());
    print_quantiles_("ours_float", ours);
    print_quantiles_("rdkit_double_same_geometry", theirs);
    print_quantiles_("ratio_rdkit_over_ours", ratio);
    print_quantiles_("start_ratio_rdkit_over_ours", start_ratio);

    // What our `float` evaluator reports on a structure RDKit calls relaxed: the floor our thresholds sit above.
    std::vector<double> ours_at_minimum, theirs_at_minimum, disagreement;
    for (molecule_record_t const &record : records) {
        ours_at_minimum.push_back(record.our_rms_gradient_at_rdkit_minimum);
        theirs_at_minimum.push_back(record.rdkit_final_rms_gradient);
        disagreement.push_back(std::fabs(record.our_rms_gradient_at_rdkit_minimum - record.rdkit_final_rms_gradient));
    }
    print_quantiles_("ours_float_at_rdkit_minimum", ours_at_minimum);
    print_quantiles_("rdkit_double_at_rdkit_minimum", theirs_at_minimum);
    print_quantiles_("float_noise_floor_at_minimum", disagreement);
}

/** Table four: how far above RDKit's own basin our relaxed geometry sits. */
static void report_energy_gap_(std::vector<molecule_record_t> const &records) {
    fmt::println("energy_gap: kcal per mol above the energy RDKit reached from the same start");
    std::vector<double> in_double, in_float;
    for (molecule_record_t const &record : records) {
        in_double.push_back(record.rdkit_energy_at_ours - record.rdkit_final_energy);
        in_float.push_back(static_cast<double>(record.energy[production_cap_index_k]) - record.rdkit_final_energy);
    }
    std::uint32_t within_one = 0, within_ten = 0;
    for (double const gap : in_double) {
        within_one += gap < 1.0;
        within_ten += gap < 10.0;
    }
    print_quantiles_("rdkit_double_at_ours_minus_rdkit", in_double);
    print_quantiles_("our_float_minus_rdkit", in_float);
    fmt::println("  within_1_kcal={}/{} within_10_kcal={}/{}", within_one, records.size(), within_ten, records.size());
}

#pragma endregion Report

} // namespace smallfield::bench

namespace sfb = smallfield::bench;

int main(int argc, char **argv) {
    fmt::println("{}: Barzilai-Borwein MMFF94 polish against its own budget and against RDKit's minimizer",
                 sfb::executable_name_k);

    std::string_view corpus_path = sfb::corpus_path_from_environment();
    std::size_t max_molecules = 200;
    sfb::command_line_t command_line {sfb::executable_name_k.data(), argc, argv};
    while (command_line.advance()) {
        std::string_view const flag = command_line.flag();
        sfb::arguments_parse_t parsed = sfb::arguments_parse_t::accepted_k;
        if (flag == "--smi") parsed = command_line.take_text(corpus_path);
        else if (flag == "--molecules") parsed = command_line.take_number<std::size_t>(1'000'000, max_molecules);
        else parsed = command_line.refuse_unknown();
        if (parsed != sfb::arguments_parse_t::accepted_k) {
            fmt::println(stderr, "usage: {} [--smi PATH] [--molecules N]", sfb::executable_name_k);
            return 2;
        }
    }

    std::vector<std::string> corpus;
    sfb::corpus_load_t const loaded = sfb::load_corpus_into(corpus_path, max_molecules, corpus);
    if (loaded != sfb::corpus_load_t::loaded_k) {
        fmt::println(stderr, "{}: corpus '{}' {}", sfb::executable_name_k, corpus_path,
                     sfb::describe_corpus_load(loaded));
        return 2;
    }
    fmt::println("corpus={} molecules={}", corpus_path, fmt::group_digits(corpus.size()));

    std::vector<sfb::molecule_record_t> records;
    sfb::sweep_tally_t tally;
    sfb::sweep_corpus_(corpus, records, tally);
    fmt::println("molecules measured={} unprepared={} unembedded={} unaligned={} unreferenced={} refused={}",
                 tally.measured, tally.unprepared, tally.unembedded, tally.unaligned, tally.unreferenced,
                 tally.refused);
    if (tally.refused != 0)
        fmt::println(stderr, "{}: the device refused {} molecules, first as {}", sfb::executable_name_k, tally.refused,
                     sftest::mmff_energy_probe_reason(tally.first_refusal));
    if (records.empty()) {
        fmt::println(stderr, "{}: corpus yielded no measurable molecule", sfb::executable_name_k);
        return 1;
    }

    sfb::report_iteration_sweep_(records);
    sfb::report_rdkit_reference_(records);
    sfb::report_precision_probe_(records);
    sfb::report_energy_gap_(records);
    return 0;
}
