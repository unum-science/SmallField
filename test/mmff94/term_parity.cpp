/**
 *  @file test/mmff94/term_parity.cpp
 *  @brief Gate per-term MMFF94 energy and gradient parity between the GPU kernel and RDKit.
 *
 *  A total-energy check cannot localize a divergence, and two terms wrong in opposite directions mask each other, so
 *  RDKit's per-term toggles isolate each term on identical coordinates and each closes its own `compared=` section.
 *  Both halves are compared: `calcEnergy` answers the kernel's scalar and `calcGrad` its gradient component by
 *  component, since an energy cannot see a sign error the scalar integrates away and the gradient is what the
 *  minimizer consumes. Isolation on the native side keeps only one category's offsets and interactions; the
 *  nonbonded sweep, enumerated on the fly not from a list, is suppressed by marking every pair BFS-unreachable.
 *
 *  Coverage is a curated set chosen for the terms most likely to diverge, each scored embedded and again distorted,
 *  plus an optional SMILES corpus named by `argv[1]` or `SMALLFIELD_TEST_CORPUS` and capped by `argv[2]` or
 *  `SMALLFIELD_TEST_MAX_MOLS`. A named corpus that does not open is a failure, never an empty run.
 */

#include <cmath>   // `std::fabs`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`, `std::uint8_t`
#include <cstdio>  // `std::fflush`, `stderr`

#include <algorithm>   // `std::max`, `std::min`, `std::sort`
#include <array>       // `std::array`
#include <iostream>    // RDKit's `AtomTyper.h` needs it but does not include it
#include <memory>      // `std::unique_ptr`
#include <sstream>     // RDKit's `AtomTyper.h` needs it but does not include it
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::format`

#include <boost/shared_ptr.hpp>
#include <GraphMol/GraphMol.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/types/views.hpp"

#include "test/cuda/mmff_gradient_kernels.hpp"
#include "test/rdkit_reference.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "smallfield_test_term_parity";

/** A corpus disagrees at thousands of components at a time, and a few dozen lines already name the pattern. */
static constexpr std::size_t max_printed_lines_k = 64;

/** Molecules taken from a named corpus when no cap is given; each is scored twice, embedded and distorted. */
static constexpr std::size_t default_max_molecules_k = 200;

/** Angstroms of distortion applied to the second scoring of every corpus molecule. */
static constexpr float corpus_distortion_k = 0.20f;

/**
 *  @brief Ceiling on one corpus term's disagreement, in parts per million of that term's comparisons. One budget
 *  covers every term, so none hides a regression behind another's allowance.
 *
 *  The curated set is held to exact agreement; a corpus is not, because the terms come from this project's parse of
 *  the SMILES while the coordinates come from RDKit's, and indexing is fingerprinted before any comparison, so what
 *  remains is chemistry the two parses perceive differently. Measured 2026-08-14 over the first 300 SMILES of
 *  `data/usearch-molecules/example`, 526 of 600 scorings reaching a comparison and 73,282 comparisons per term:
 *  bond and nonbonded exact, out-of-plane and torsion 123 ppm, stretch-bend 396 ppm, angle 423 ppm — gradient
 *  components only, no energy — so this budget is five times the worst and an order of magnitude below a regression.
 *
 *  Torsion's own 80,000 ppm exception once admitted a 47,644 ppm divergence, 149 of them energies rather than
 *  gradient components; it now measures 123 ppm on the same corpus. Triage a rise against `cpu_extract` and
 *  the SMILES parser rather than reintroducing a per-term exception.
 */
static constexpr double max_corpus_term_mismatch_ppm_k = 2000.0;

namespace smallfield::tests {

#pragma region Coordinate Layouts

/* `smallfield/types/views.hpp` owns the extent-inside-the-view vocabulary for matrices and coordinates alike: a
   layout is a type, so handing an atom-major buffer to an axis-major reader stops compiling. */

using sft::conformer_aos_t;
using sft::conformer_soa_t;
using sft::coordinate_role_t;
using sft::coordinates_aos;
using sft::coordinates_soa;
using sft::reset_coordinates_aos;
using sft::reset_coordinates_soa;
using sft::transpose_into;

/** The kernel's gradient, axis-major over the three spatial axes. */
using gradient_soa_t = coordinates_soa<coordinate_role_t::gradient_k, float, sft::conformer_dimensions_k>;

/** RDKit's gradient, atom-major and double-precision, which is what `calcGrad` accumulates into. */
using gradient_aos_t = coordinates_aos<coordinate_role_t::gradient_k, double>;

#pragma endregion Coordinate Layouts

#pragma region Term Isolation

/** Which single MMFF94 term a comparison isolates. */
enum class term_kind_t { bond_k, angle_k, stretchbend_k, oop_k, torsion_k, nonbonded_k };

/** Every term the comparison isolates, in the order the sections print. */
static constexpr term_kind_t term_kinds_k[] = {term_kind_t::bond_k, term_kind_t::angle_k,   term_kind_t::stretchbend_k,
                                               term_kind_t::oop_k,  term_kind_t::torsion_k, term_kind_t::nonbonded_k};

static constexpr std::size_t term_kind_count_k = sizeof(term_kinds_k) / sizeof(term_kinds_k[0]);

static char const *term_name_(term_kind_t kind) {
    switch (kind) {
    case term_kind_t::bond_k: return "bond";
    case term_kind_t::angle_k: return "angle";
    case term_kind_t::stretchbend_k: return "stretch-bend";
    case term_kind_t::oop_k: return "out-of-plane";
    case term_kind_t::torsion_k: return "torsion";
    default: return "nonbonded";
    }
}

/** Fill @p single with @p full, every term but @p kind neutralized. The per-atom interaction CSR is left as
 *  production builds it — rebuilding it by hand would test the reconstruction rather than the kernel. Excluded
 *  bonded terms are zeroed at their force constants, contributing neither energy nor gradient; the nonbonded sweep
 *  reads the hop matrix rather than a list, so the caller isolates it with a matrix of the unreachable sentinel.
 *  @param[in] full Prepared molecule carrying every term.
 *  @param[in] kind The one term left intact.
 *  @param[out] single Caller-owned copy, so the assignment reuses its buffers across terms. */
static void isolate_term_into_(sfp::etkdg_prepared_molecule_t const &full, term_kind_t kind,
                               sfp::etkdg_prepared_molecule_t &single) {
    single = full;

    if (kind != term_kind_t::bond_k)
        for (auto &term : single.mmff_bonds) term.force_constant = 0.0f;
    if (kind != term_kind_t::angle_k)
        for (auto &term : single.mmff_angles) term.force_constant = 0.0f;
    if (kind != term_kind_t::torsion_k)
        for (auto &term : single.mmff_torsions) term.torsion_v1 = term.torsion_v2 = term.torsion_v3 = 0.0f;
    if (kind != term_kind_t::oop_k)
        for (auto &term : single.mmff_oops) term.force_constant = 0.0f;
    if (kind != term_kind_t::stretchbend_k)
        for (auto &term : single.mmff_stretchbends) term.kba_ijk = term.kba_kji = 0.0f;
}

/** Whether RDKit produced a reference for one isolated term, or which stage declined it: `MMFFMolProperties`
 *  came back invalid, or `constructForceField` returned nothing for the enabled term. */
enum class reference_state_t {
    evaluated_k,
    untypable_k,
    unconstructable_k,
};

/** RDKit's MMFF94 energy and gradient with exactly @p kind enabled, at the molecule's coordinates. The six setters
 *  take a `bool` because that is RDKit's API; the answer is derived from `kind` at the call itself, so the enum is
 *  what this file reasons about. `calcGrad` accumulates, so @p gradient arrives zeroed from `reset_coordinates_aos`.
 *  @param[inout] molecule Molecule whose conformer is scored; RDKit caches properties on it.
 *  @param[in] kind The one term left enabled.
 *  @param[out] gradient Receives dE/dx per atom, atom-major, as RDKit lays it out.
 *  @param[out] energy Receives the term's energy in kcal/mol; untouched unless `evaluated_k` is returned.
 *  @return Whether a reference was produced. */
static reference_state_t rdkit_single_term_(RDKit::RWMol &molecule, term_kind_t kind, gradient_aos_t gradient,
                                            double &energy) {
    RDKit::MMFF::MMFFMolProperties props(molecule);
    if (!props.isValid()) return reference_state_t::untypable_k;
    props.setMMFFBondTerm(kind == term_kind_t::bond_k);
    props.setMMFFAngleTerm(kind == term_kind_t::angle_k);
    props.setMMFFStretchBendTerm(kind == term_kind_t::stretchbend_k);
    props.setMMFFOopTerm(kind == term_kind_t::oop_k);
    props.setMMFFTorsionTerm(kind == term_kind_t::torsion_k);
    props.setMMFFVdWTerm(kind == term_kind_t::nonbonded_k);
    props.setMMFFEleTerm(kind == term_kind_t::nonbonded_k);
    std::unique_ptr<ForceFields::ForceField> field {
        RDKit::MMFF::constructForceField(molecule, &props, 100.0, -1, false)};
    if (!field) return reference_state_t::unconstructable_k;
    field->initialize();
    energy = field->calcEnergy();
    field->calcGrad(gradient.data_handle());
    return reference_state_t::evaluated_k;
}

/** Deterministically displace every atom by roughly @p amplitude angstroms. An RDKit-embedded geometry sits near
 *  equilibrium, where several MMFF forms coincide — at a linear center, theta = 180 gives the same energy under the
 *  quadratic and the linear bend — so distorting it moves the terms onto the parts of their curves that differ.
 *  @param[inout] positions Coordinates displaced in place; the walk order fixes the sequence.
 *  @param[in] amplitude Half-width of the per-component displacement, in angstroms. */
static void perturb_(conformer_soa_t positions, float amplitude) {
    std::uint32_t state = 0x9E3779B9u;
    // Axis-outer so the sequence it draws stays what it was; the view is addressed atom-first, and only the
    // subscript follows the layout.
    for (std::uint32_t axis = 0; axis < sft::conformer_dimensions_k; ++axis)
        for (std::uint32_t atom = 0; atom < positions.extent(0); ++atom) {
            state = state * 1664525u + 1013904223u;
            float const unit = static_cast<float>(state >> 8) * (1.0f / 16777216.0f) - 0.5f;
            positions(atom, axis) += 2.0f * amplitude * unit;
        }
}

#pragma endregion Term Isolation

#pragma region Comparison

/** Whether a case narrates every term it scores, or only the components that disagree. */
enum class narration_t { silent_k, narrated_k };

/** One comparison: a molecule, and the angstroms of deterministic per-component displacement applied to its
 *  embedded geometry before scoring, zero scoring that geometry as it is. */
struct term_parity_case_t {
    std::string smiles;
    float perturbation = 0.0f;
};

/** Census of the cases that never reached a comparison; the reporters own everything that did, so nothing here
 *  duplicates a `compared=` or a `mismatches=`. `preprocess_rejected` and `unparsable` are the two parsers refusing
 *  the SMILES and `embed_failures` is ETKDG leaving no coordinates; `fragment_differs` is preprocessing keeping one
 *  fragment where RDKit kept the salt; `parses_differ` is the same molecule under two atom indexings; `untypable`
 *  is RDKit's MMFF94 typing failing, so it can neither fingerprint the indexing nor reference a term.
 *  `reference_unavailable` and `device_refused` count case-and-term pairs, the latter having written neither energy
 *  nor gradient; `first_refusal` carries the step of the first, which a count cannot; and `cases_differing` lets a
 *  rate read per molecule as well as per component. */
struct intake_tally_t {
    std::uint64_t cases = 0;
    std::uint64_t compared = 0;
    std::uint64_t preprocess_rejected = 0;
    std::uint64_t unparsable = 0;
    std::uint64_t embed_failures = 0;
    std::uint64_t fragment_differs = 0;
    std::uint64_t parses_differ = 0;
    std::uint64_t untypable = 0;
    std::uint64_t reference_unavailable = 0;
    std::uint64_t device_refused = 0;
    mmff_energy_probe_t first_refusal = mmff_energy_probe_t::completed_k;
    std::uint64_t cases_differing = 0;
};

/** One diagnostic sink per term, so each closes its own `compared=` section: a shared sink could not say which
 *  term a comparison came from, and a term whose energy is identically zero corpus-wide would hide in another's
 *  count. */
struct term_reporters_t {
    std::array<diagnostic_reporter_t, term_kind_count_k> per_term {
        diagnostic_reporter_t {"rdkit", "gpu"}, diagnostic_reporter_t {"rdkit", "gpu"},
        diagnostic_reporter_t {"rdkit", "gpu"}, diagnostic_reporter_t {"rdkit", "gpu"},
        diagnostic_reporter_t {"rdkit", "gpu"}, diagnostic_reporter_t {"rdkit", "gpu"}};

    diagnostic_reporter_t &operator[](term_kind_t kind) { return per_term[static_cast<std::size_t>(kind)]; }

    void limit_printed_lines(std::size_t maximum) noexcept {
        for (auto &reporter : per_term) reporter.limit_printed_lines(maximum);
    }

    /** Close one section per term, which is also what fails a term that compared nothing.
     *  @param[in] sweep Prefixes every subject, so the curated and corpus sections stay distinct. */
    void summarize_terms(std::string_view sweep) {
        for (term_kind_t kind : term_kinds_k) (*this)[kind].summarize(fmt::format("{}_{}", sweep, term_name_(kind)));
    }

    bool passed() const noexcept {
        for (auto const &reporter : per_term)
            if (!reporter.passed()) return false;
        return true;
    }

    std::size_t comparison_count() const noexcept {
        std::size_t total = 0;
        for (auto const &reporter : per_term) total += reporter.comparison_count();
        return total;
    }

    std::size_t mismatch_count() const noexcept {
        std::size_t total = 0;
        for (auto const &reporter : per_term) total += reporter.mismatch_count();
        return total;
    }
};

/** Every buffer one case needs, owned by the sweep so the growth phase happens once. Each is refilled through a
 *  `reset_` builder that hands back a view, so no extent here is carried beside its storage. `isolated` is
 *  `prepared` with one term left intact, rewritten per term; the two `*_bond_keys` are each parser's sorted bond
 *  keys, compared once per molecule. */
struct molecule_scratch_t {
    std::unique_ptr<RDKit::RWMol> molecule;
    sfp::etkdg_prepared_molecule_t prepared;
    sfp::etkdg_prepared_molecule_t isolated;
    std::vector<float> interleaved;
    std::vector<float> positions;
    std::vector<float> gpu_gradient;
    std::vector<double> rdkit_gradient;
    std::vector<std::uint8_t> bond_hops;
    std::vector<std::uint8_t> unreachable_hops;
    std::vector<std::uint32_t> native_bond_keys;
    std::vector<std::uint32_t> rdkit_bond_keys;
};

/** Pack an undirected bond into one sortable key, so two bond lists compare as sets. */
static std::uint32_t packed_bond_key_(std::uint32_t atom_i, std::uint32_t atom_j) {
    return (std::min(atom_i, atom_j) << 8) | std::max(atom_i, atom_j);
}

/** Whether the native parse and RDKit's can be compared atom for atom: RDKit's MMFF94 typing failed, so there is
 *  nothing to fingerprint against, or the same molecule arrived under two atom indexings. */
enum class parse_agreement_t {
    agrees_k,
    untypable_k,
    differs_k,
};

/** Whether the native parse and RDKit's describe the same molecule under the same atom indices. The terms come from
 *  this project's parser and the coordinates from RDKit's, so a molecule the two index differently would have its
 *  bond, angle and torsion parameters scored against another atom's position. Two fingerprints settle it — the bond
 *  set pins the topology and the per-atom partial charge pins each index to an environment — so what survives both
 *  is a relabeling preserving adjacency and charge, under which the two term lists are the same list.
 *  @param[in] prepared Terms the native parse produced.
 *  @param[inout] molecule RDKit's parse of the same SMILES; typing caches properties on it.
 *  @param[out] native_bond_keys Scratch, refilled with the native bond keys.
 *  @param[out] rdkit_bond_keys Scratch, refilled with RDKit's bond keys.
 *  @return Whether the two indexings agree, or why the question could not be answered. */
static parse_agreement_t parses_agree_(sfp::etkdg_prepared_molecule_t const &prepared, RDKit::RWMol &molecule,
                                       std::vector<std::uint32_t> &native_bond_keys,
                                       std::vector<std::uint32_t> &rdkit_bond_keys) {
    if (prepared.mmff_bonds.size() != molecule.getNumBonds()) return parse_agreement_t::differs_k;
    native_bond_keys.clear();
    rdkit_bond_keys.clear();
    for (auto const &bond : prepared.mmff_bonds) native_bond_keys.push_back(packed_bond_key_(bond.atom_i, bond.atom_j));
    for (auto const *bond : molecule.bonds())
        rdkit_bond_keys.push_back(packed_bond_key_(bond->getBeginAtomIdx(), bond->getEndAtomIdx()));
    std::sort(native_bond_keys.begin(), native_bond_keys.end());
    std::sort(rdkit_bond_keys.begin(), rdkit_bond_keys.end());
    if (native_bond_keys != rdkit_bond_keys) return parse_agreement_t::differs_k;

    RDKit::MMFF::MMFFMolProperties props(molecule);
    if (!props.isValid()) return parse_agreement_t::untypable_k;
    for (std::uint32_t atom = 0; atom < prepared.num_atoms; ++atom)
        if (std::fabs(static_cast<double>(prepared.mmff_atom_charge[atom]) - props.getMMFFPartialCharge(atom)) > 1e-3)
            return parse_agreement_t::differs_k;
    return parse_agreement_t::agrees_k;
}

/** Energies agree within 0.05 kcal/mol plus one percent, the budget a float kernel owes a double reference. */
static bool energies_agree_(double reference, double observed) {
    return std::fabs(observed - reference) <= 0.05 + 0.01 * std::fabs(reference);
}

/** The same budget one derivative down, where the absolute floor carries the near-zero components. */
static bool gradient_components_agree_(double reference, double observed) {
    return std::fabs(observed - reference) <= 0.02 + 0.01 * std::fabs(reference);
}

/** One field name per axis, so a differing component names itself instead of carrying an index. */
static constexpr std::string_view gradient_field_names_k[3] = {"gradient_x", "gradient_y", "gradient_z"};

/** Score one molecule term by term, on one geometry, against RDKit.
 *  @param[in] narration Whether every term's energy prints, or only the components that disagree.
 *  @param[inout] scratch Buffers reused across cases.
 *  @param[inout] reporters Sinks that count and print, one per term.
 *  @param[inout] tally Counters for the cases that never reached a comparison. */
static void compare_case_(term_parity_case_t const &probe, narration_t narration, molecule_scratch_t &scratch,
                          term_reporters_t &reporters, intake_tally_t &tally) {
    ++tally.cases;
    if (sfp::prepare_molecule_into(probe.smiles, scratch.prepared) != sft::conformer_status_t::success_k) {
        ++tally.preprocess_rejected;
        return;
    }
    scratch.molecule = mol_from_smiles(probe.smiles);
    if (!scratch.molecule) {
        ++tally.unparsable;
        return;
    }
    RDKit::RWMol &molecule = *scratch.molecule;
    if (!embed_molecule(molecule)) {
        ++tally.embed_failures;
        return;
    }
    // Preprocessing keeps only the largest fragment, so a salt leaves RDKit holding atoms the prepared terms never
    // describe — two different molecules.
    std::uint32_t const num_atoms = scratch.prepared.num_atoms;
    if (molecule.getNumAtoms() != num_atoms) {
        ++tally.fragment_differs;
        return;
    }
    // Same molecule is not enough; a relabeling between the native parser's terms and RDKit's coordinates would
    // score one atom's parameters at another's position.
    parse_agreement_t const agreement = parses_agree_(scratch.prepared, molecule, scratch.native_bond_keys,
                                                      scratch.rdkit_bond_keys);
    if (agreement == parse_agreement_t::untypable_k) {
        ++tally.untypable;
        return;
    }
    if (agreement == parse_agreement_t::differs_k) {
        ++tally.parses_differ;
        return;
    }
    ++tally.compared;
    std::size_t const mismatches_before = reporters.mismatch_count();

    auto const interleaved = reset_coordinates_aos<coordinate_role_t::position_k>(scratch.interleaved, num_atoms);
    auto const positions = reset_coordinates_soa<coordinate_role_t::position_k, sft::conformer_dimensions_k>(
        scratch.positions, num_atoms);
    read_conformer_into(molecule, interleaved);
    transpose_into(interleaved, positions);
    if (probe.perturbation > 0.0f) {
        perturb_(positions, probe.perturbation);
        // RDKit must score the same distorted geometry, not the embedded one.
        transpose_into(positions, interleaved);
        set_coords_interleaved(molecule, interleaved.data_handle());
    }

    // Production rebuilds the hop matrix on the device; this single-molecule harness derives it from the bond list,
    // and hands the sweep an all-unreachable matrix whenever the comparison isolates some other term.
    auto const bond_hops = sft::reset_square_matrix(scratch.bond_hops, num_atoms, sft::unreachable_hops_k);
    fill_bond_hops_into(scratch.prepared.mmff_bonds, bond_hops);
    auto const unreachable_hops = sft::reset_square_matrix(scratch.unreachable_hops, num_atoms,
                                                           sft::unreachable_hops_k);

    for (term_kind_t kind : term_kinds_k) {
        isolate_term_into_(scratch.prepared, kind, scratch.isolated);
        auto const &single = scratch.isolated;
        auto const gpu_gradient = reset_coordinates_soa<coordinate_role_t::gradient_k, sft::conformer_dimensions_k>(
            scratch.gpu_gradient, num_atoms);
        float gpu_energy = 0.0f;
        mmff_energy_terms_t const terms {
            .atom_vdw = single.mmff_atom_vdw,
            .atom_charge = single.mmff_atom_charge,
            .bond_distance = kind == term_kind_t::nonbonded_k ? bond_hops : unreachable_hops,
            .bonds = single.mmff_bonds,
            .angles = single.mmff_angles,
            .torsions = single.mmff_torsions,
            .oops = single.mmff_oops,
            .stretchbends = single.mmff_stretchbends,
            .interactions = single.mmff_interactions,
            .atom_offsets = single.mmff_atom_offsets,
        };
        // A refused launch never writes the gradient and the buffer is not zeroed first, so a declined term would
        // otherwise be scored against RDKit on whatever it held.
        if (mmff_energy_probe_t const probe = gpu_mmff_energy(scratch.positions, single.num_atoms, terms,
                                                              scratch.gpu_gradient, &gpu_energy);
            probe != mmff_energy_probe_t::completed_k) {
            if (tally.first_refusal == mmff_energy_probe_t::completed_k) tally.first_refusal = probe;
            ++tally.device_refused;
            continue;
        }

        auto const rdkit_gradient = reset_coordinates_aos<coordinate_role_t::gradient_k>(scratch.rdkit_gradient,
                                                                                         num_atoms);
        double rdkit_energy = 0.0;
        if (rdkit_single_term_(molecule, kind, rdkit_gradient, rdkit_energy) != reference_state_t::evaluated_k) {
            ++tally.reference_unavailable;
            continue;
        }

        // A curated term prints whether it agrees or not, since the number is what the reader came for; a corpus
        // sweep spends its line budget on the components that disagree.
        if (narration == narration_t::narrated_k)
            reporters[kind]
                .note("mmff_term", "smiles={} distortion={:.2f} term={}", probe.smiles, probe.perturbation,
                      term_name_(kind))
                .field("energy", rdkit_energy, static_cast<double>(gpu_energy));
        reporters[kind]
            .mismatch("mmff_term", "smiles={} distortion={:.2f} term={}", probe.smiles, probe.perturbation,
                      term_name_(kind))
            .field("energy", rdkit_energy, static_cast<double>(gpu_energy), energies_agree_);

        // Each gradient is read through its own layout: RDKit fills atom-major triples and the kernel axis-major
        // planes, yet both views are addressed `(atom, axis)`, so the two reads cannot be crossed.
        for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
            auto group = reporters[kind].mismatch("mmff_term", "smiles={} distortion={:.2f} term={} atom={}",
                                                  probe.smiles, probe.perturbation, term_name_(kind), atom);
            for (std::uint32_t axis = 0; axis < sft::conformer_dimensions_k; ++axis)
                group.field(gradient_field_names_k[axis], rdkit_gradient(atom, axis),
                            static_cast<double>(gpu_gradient(atom, axis)), gradient_components_agree_);
        }
    }
    if (reporters.mismatch_count() != mismatches_before) ++tally.cases_differing;
}

/** Score every case in @p probes, non-short-circuiting: the counters are the output and feed the gate, so a
 *  differing molecule is reported and then walked past.
 *  @param[in] probes Cases to score, in order.
 *  @param[in] narration Whether every term's energy prints, or only the components that disagree.
 *  @param[inout] scratch Buffers reused across cases.
 *  @param[inout] reporters Sinks that count and print, one per term.
 *  @param[inout] tally Counters for the cases that never reached a comparison. */
static void sweep_(custd::span<term_parity_case_t const> probes, narration_t narration, molecule_scratch_t &scratch,
                   term_reporters_t &reporters, intake_tally_t &tally) {
    for (auto const &probe : probes) compare_case_(probe, narration, scratch, reporters, tally);
}

/** Print what one sweep took in; the comparison counts belong to the terms' own sections.
 *  @param[in] heading Which sweep the counters describe. */
static void report_intake_(std::string_view heading, intake_tally_t const &tally) {
    fmt::println("NOTE intake sweep={} cases={} compared={}"                     //
                 " preprocess_rejected={} unparsable={} embed_failures={}"       //
                 " fragment_differs={} parses_differ={} untypable={}"            //
                 " reference_unavailable={} device_refused={} first_refusal={}", //
                 heading, tally.cases, tally.compared, tally.preprocess_rejected, tally.unparsable,
                 tally.embed_failures, tally.fragment_differs, tally.parses_differ, tally.untypable,
                 tally.reference_unavailable, tally.device_refused, mmff_energy_probe_reason(tally.first_refusal));
}

#pragma endregion Comparison

#pragma region Corpora

/** The molecules every run scores, whatever else the command line names. Aspirin is the general case; the rest
 *  target linear centers and the double-bond geometry the torsion term reads. The distorted repeats are not
 *  redundant: at a linear center the quadratic and the linear bend coincide at equilibrium, so only a bent geometry
 *  tells them apart.
 *  @return One case per scoring, built once at start-up. */
static std::vector<term_parity_case_t> curated_cases_() {
    return {
        {.smiles = "CC(=O)Oc1ccccc1C(=O)O", .perturbation = 0.0f},  // aspirin
        {.smiles = "CC#N", .perturbation = 0.0f},                   // acetonitrile
        {.smiles = "CC#CC", .perturbation = 0.0f},                  // 2-butyne
        {.smiles = "C/C=C/C", .perturbation = 0.0f},                // trans-2-butene
        {.smiles = "C/C=C\\C", .perturbation = 0.0f},               // cis-2-butene
        {.smiles = "c1ccccc1", .perturbation = 0.0f},               // benzene
        {.smiles = "CC#N", .perturbation = 0.30f},                  // acetonitrile, bent
        {.smiles = "CC#CC", .perturbation = 0.30f},                 // 2-butyne, bent
        {.smiles = "CC(=O)Oc1ccccc1C(=O)O", .perturbation = 0.20f}, // aspirin, distorted
    };
}

/** Score every corpus molecule twice, embedded and distorted.
 *  @param[in] corpus SMILES as loaded, in order.
 *  @param[out] probes Cleared, then filled with two cases per SMILES. */
static void fill_corpus_cases_into_(custd::span<std::string const> corpus, std::vector<term_parity_case_t> &probes) {
    probes.clear();
    probes.reserve(corpus.size() * 2);
    for (auto const &smiles : corpus) {
        probes.push_back({.smiles = smiles, .perturbation = 0.0f});
        probes.push_back({.smiles = smiles, .perturbation = corpus_distortion_k});
    }
}

#pragma endregion Corpora

#pragma region Verdicts

/** Close the curated sections and answer with the exit code they decided. The curated set is chosen rather than
 *  sampled, so it is held to exact agreement and to full intake: a molecule that preprocessing, RDKit or ETKDG
 *  drops here is a regression in one of them.
 *  @param[inout] reporters Sinks the curated sweep reported through.
 *  @param[in] tally What the curated sweep took in.
 *  @return `0` when every curated term agrees, `1` otherwise. */
static int report_curated_(term_reporters_t &reporters, intake_tally_t const &tally) {
    report_intake_("curated", tally);
    reporters.summarize_terms("curated");
    if (tally.compared != tally.cases)
        return report_fail(test_name_k, "{} of {} curated scorings never reached a comparison",
                           tally.cases - tally.compared, tally.cases);
    if (tally.reference_unavailable != 0)
        return report_fail(test_name_k, "RDKit declined to reference {} curated terms", tally.reference_unavailable);
    if (tally.device_refused != 0)
        return report_fail(test_name_k, "the device refused {} curated terms, first as {}", tally.device_refused,
                           mmff_energy_probe_reason(tally.first_refusal));
    if (!reporters.passed())
        return report_fail(test_name_k, "{} of {} curated comparisons disagree with RDKit", reporters.mismatch_count(),
                           reporters.comparison_count());
    return 0;
}

/** Close the corpus sections and answer with the exit code the budget decided.
 *  @param[inout] reporters Sinks the corpus sweep reported through.
 *  @param[in] tally What the corpus sweep took in.
 *  @param[in] path The corpus, named in the failure so the run is reproducible.
 *  @return `0` when the disagreement stays inside the budget, `1` otherwise. */
static int report_corpus_(term_reporters_t &reporters, intake_tally_t const &tally, std::string_view path) {
    report_intake_("corpus", tally);
    if (tally.compared == 0)
        return report_fail(test_name_k, "corpus {} produced no comparison over {} scorings", path, tally.cases);
    if (tally.device_refused != 0)
        return report_fail(test_name_k, "the device refused {} corpus terms, first as {}", tally.device_refused,
                           mmff_energy_probe_reason(tally.first_refusal));

    // Not through `summarize`: on a corpus a disagreement inside budget is not a failure, and the verb at column
    // zero has to agree with the exit code. Each term is judged against its own budget so none hides behind
    // another's allowance, and the loop runs to completion because every line of it is the report.
    std::size_t terms_over_budget = 0;
    for (term_kind_t kind : term_kinds_k) {
        auto const &reporter = reporters[kind];
        std::size_t const compared = reporter.comparison_count();
        // A term that compared nothing reports as wholly differing: a vacuous section must not read clean.
        double const differing_ppm = compared == 0 ? 1.0e6
                                                   : 1.0e6 * static_cast<double>(reporter.mismatch_count()) / compared;
        double const budget = max_corpus_term_mismatch_ppm_k;
        diagnostic_verb_t const verb = differing_ppm > budget ? diagnostic_verb_t::fail_k : diagnostic_verb_t::note_k;
        std::FILE *const stream = diagnostic_stream(verb);
        if (stream == stderr) std::fflush(stdout);
        fmt::println(stream, "{} corpus_{} compared={} mismatches={} suppressed={} differing_ppm={:.3f} budget={:.3f}",
                     diagnostic_verb_name(verb), term_name_(kind), compared, reporter.mismatch_count(),
                     reporter.suppressed_line_count(), differing_ppm, budget);
        if (verb == diagnostic_verb_t::fail_k) ++terms_over_budget;
    }
    fmt::println("NOTE corpus scorings_differing={} of {}", tally.cases_differing, tally.compared);
    if (terms_over_budget != 0)
        return report_fail(test_name_k, "{} of {} terms disagree beyond budget over corpus {}", terms_over_budget,
                           term_kind_count_k, path);
    return 0;
}

#pragma endregion Verdicts

} // namespace smallfield::tests

int main(int argc, char **argv) {
    fmt::println("{}: MMFF94 per-term energy and gradient against RDKit, one term at a time", test_name_k);

    auto const selection = sftest::select_corpus(argc, argv, default_max_molecules_k,
                                                 sftest::adversarial_corpus_path());
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        std::fflush(stdout);
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", test_name_k);
        return 2;
    }

    sftest::molecule_scratch_t scratch;
    sftest::term_reporters_t curated_reporters;
    curated_reporters.limit_printed_lines(max_printed_lines_k);
    sftest::intake_tally_t curated;
    auto const curated_probes = sftest::curated_cases_();
    sftest::sweep_(curated_probes, sftest::narration_t::narrated_k, scratch, curated_reporters, curated);
    if (int const verdict = sftest::report_curated_(curated_reporters, curated); verdict != 0) return verdict;

    if (selection.knob == sftest::corpus_knob_t::named_k) {
        // Naming a corpus is a request to validate against it, so a path that does not open is a failure, never an
        // empty run that passes having compared nothing.
        std::vector<std::string> corpus;
        auto const outcome = sftest::load_smiles_into(selection.path, selection.max_molecules, corpus);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), selection.path);
        fmt::println("NOTE corpus path={} smiles={}", selection.path, corpus.size());

        std::vector<sftest::term_parity_case_t> corpus_probes;
        sftest::fill_corpus_cases_into_(corpus, corpus_probes);
        sftest::term_reporters_t corpus_reporters;
        corpus_reporters.limit_printed_lines(max_printed_lines_k);
        sftest::intake_tally_t sampled;
        sftest::sweep_(corpus_probes, sftest::narration_t::silent_k, scratch, corpus_reporters, sampled);
        if (int const verdict = sftest::report_corpus_(corpus_reporters, sampled, selection.path); verdict != 0)
            return verdict;
    }

    return sftest::report_pass(test_name_k);
}
