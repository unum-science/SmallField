/**
 *  @file test/mmff94/test_mmff_gpu_extract.cpp
 *  @brief Compare the GPU MMFF term extractor against the CPU native extractor.
 *
 *  Runs a batch of drug-like SMILES through both paths. Byte identity is required for every bonded
 *  term of every molecule where the GPU typer flagged success, meaning `typing_status == 0`.
 */

#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::fflush`
#include <cstring> // `std::memcmp`

#include <iterator>    // `std::begin`, `std::end`
#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <boost/shared_ptr.hpp>
#include <cuda_runtime.h>
#include <RDGeneral/types.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/mmff94/mmff94_extract.hpp"
#include "smallfield/mmff94/mmff94_extract_device.cuh"
#include "smallfield/mmff94/mmff94_term_equations.hpp"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_rdkit.hpp"
#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

#include "test/mmff94/test_mmff_gpu_extract_kernels.hpp"
#include "test/test_helpers.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfmmff = smallfield::mmff94;
namespace sfparam = smallfield::parameters;
namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_mmff_gpu_extract";

namespace smallfield::tests {

#pragma region Batch

/*  `comparison_group_t::field` invokes its predicate as `equals(reference, observed)`. Term indices
 *  and counts must agree exactly; the floats carry the file's magnitude-scaled tolerance, which is
 *  the same predicate the hand-rolled `if` chains applied. */
static constexpr auto exactly_equal_ = [](auto reference, auto observed) { return reference == observed; };
static constexpr auto within_tolerance_ = [](float cpu_value, float gpu_value) {
    return approximately_equal(cpu_value, gpu_value);
};

/** How many mismatching fields any one section prints before the rest are only counted. */
static constexpr std::size_t max_printed_lines_k = 32;

/**
 *  @brief The drug-like smoke set every run compares, whether or not a corpus is named.
 *
 *  Small enough to keep a bare `ctest` under a second, and wide enough to reach every term class:
 *  an ester, three fused or substituted aromatics, a formal charge, an aromatic cation, an ether
 *  ring, and a strained ring whose angles no other member exercises.
 */
static constexpr std::string_view builtin_smiles_k[] = {
    "CC(=O)OC1=CC=CC=C1C(=O)O",      // aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  // caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", // ibuprofen
    "CC(=O)NC1=CC=C(O)C=C1",         // paracetamol
    "CC(=O)[O-]",                    // acetate
    "c1cc[nH+]c(c1)",                // pyridinium
    "C1CCOC1",                       // tetrahydrofuran
    "C1CC1",                         // cyclopropane
    "c1ccc2[nH]ccc2c1",              // indole
};

/**
 *  @brief What one molecule's fixed-stride staging slot costs, whatever the molecule holds.
 *
 *  The numerator of the compaction ratio the run reports: staging is sized for the worst molecule
 *  the bounds admit, while the packed arrays cost what the batch actually produced.
 */
static constexpr std::size_t staged_bytes_per_molecule_k =
    sft::max_mmff_bonds_per_mol_k * sizeof(sft::mmff94_bond_t) +
    sft::max_mmff_angles_per_mol_k * sizeof(sft::mmff94_angle_t) +
    sft::max_mmff_torsions_per_mol_k * sizeof(sft::mmff94_torsion_t) +
    sft::max_mmff_oops_per_mol_k * sizeof(sft::mmff94_oop_t) +
    sft::max_mmff_stretchbends_per_mol_k * sizeof(sft::mmff94_stretchbend_t) +
    sft::max_mmff_interactions_per_mol_k * sizeof(sft::mmff94_atom_interaction_t);

/**
 *  @brief One batch of molecules beside every array the GPU pipeline wrote for it.
 *
 *  Each per-molecule array is strided by its own `max_..._per_mol_k` ceiling, so a molecule's slot
 *  is reached through a named accessor rather than through the same index expression written out at
 *  each use. The arrays are sized once, by `resize_staging`, and reused for the whole run.
 */
struct extract_batch_t {
    /** One entry per molecule that parsed and yielded a descriptor. */
    std::vector<sft::molecule_descriptor_t> descriptors;
    /** Index of each molecule's first atom in the batch-global atom array. */
    std::vector<std::uint32_t> first_atoms;
    /** Atoms across the whole batch, which is what the CSR arrays are sized by. */
    std::uint32_t total_atoms = 0;

    /** Per-molecule term counts the extractor reports. */
    std::vector<sfmmff::mmff94_per_mol_counts_t> counts;
    /** Zero exactly where the GPU typer accepted the molecule. */
    std::vector<std::uint8_t> typing_status;
    std::vector<sft::mmff94_bond_t> bonds;
    std::vector<sft::mmff94_angle_t> angles;
    std::vector<sft::mmff94_torsion_t> torsions;
    std::vector<sft::mmff94_oop_t> oops;
    std::vector<sft::mmff94_stretchbend_t> stretch_bends;
    std::vector<sft::mmff94_atom_interaction_t> interactions;
    /** Per-atom CSR offsets, each numbered from zero inside its own molecule's slot. */
    std::vector<sft::mmff94_atom_offsets_t> staged_atom_offsets;
    /** What the compaction pass emitted for the same batch, with batch-global offsets. */
    mmff94_packed_host_result_t packed;

    std::uint32_t num_molecules() const noexcept { return static_cast<std::uint32_t>(descriptors.size()); }

    /** Size every staging array for the descriptors collected so far. */
    void resize_staging() {
        std::size_t const molecules = descriptors.size();
        counts.resize(molecules);
        typing_status.resize(molecules);
        bonds.resize(molecules * sft::max_mmff_bonds_per_mol_k);
        angles.resize(molecules * sft::max_mmff_angles_per_mol_k);
        torsions.resize(molecules * sft::max_mmff_torsions_per_mol_k);
        oops.resize(molecules * sft::max_mmff_oops_per_mol_k);
        stretch_bends.resize(molecules * sft::max_mmff_stretchbends_per_mol_k);
        interactions.resize(molecules * sft::max_mmff_interactions_per_mol_k);
        staged_atom_offsets.resize(total_atoms);
    }

    sft::mmff94_bond_t const *bonds_of(std::uint32_t molecule) const noexcept {
        return bonds.data() + static_cast<std::size_t>(molecule) * sft::max_mmff_bonds_per_mol_k;
    }
    sft::mmff94_angle_t const *angles_of(std::uint32_t molecule) const noexcept {
        return angles.data() + static_cast<std::size_t>(molecule) * sft::max_mmff_angles_per_mol_k;
    }
    sft::mmff94_torsion_t const *torsions_of(std::uint32_t molecule) const noexcept {
        return torsions.data() + static_cast<std::size_t>(molecule) * sft::max_mmff_torsions_per_mol_k;
    }
    sft::mmff94_oop_t const *oops_of(std::uint32_t molecule) const noexcept {
        return oops.data() + static_cast<std::size_t>(molecule) * sft::max_mmff_oops_per_mol_k;
    }
    sft::mmff94_stretchbend_t const *stretch_bends_of(std::uint32_t molecule) const noexcept {
        return stretch_bends.data() + static_cast<std::size_t>(molecule) * sft::max_mmff_stretchbends_per_mol_k;
    }
    sft::mmff94_atom_interaction_t const *interactions_of(std::uint32_t molecule) const noexcept {
        return interactions.data() + static_cast<std::size_t>(molecule) * sft::max_mmff_interactions_per_mol_k;
    }
    /** The molecule's own run of per-atom CSR offsets, which is strided by atom count rather than by a ceiling. */
    sft::mmff94_atom_offsets_t const *staged_offsets_of(std::uint32_t molecule) const noexcept {
        return staged_atom_offsets.data() + first_atoms[molecule];
    }
};

/**
 *  @brief Parse each SMILES, add hydrogens, and append every molecule that yields a descriptor.
 *
 *  A molecule RDKit refuses, or one past the batch's atom ceiling, is dropped without a diagnostic:
 *  a convenience corpus is a sample, and this test gates the extractor rather than the parser.
 *
 *  @param[in] smiles_list SMILES to parse, in order.
 *  @param[out] batch Descriptors, first-atom indices, and the batch atom total; all three cleared first.
 */
static void fill_descriptors_into_(std::vector<std::string> const &smiles_list, extract_batch_t &batch) {
    batch.descriptors.clear();
    batch.first_atoms.clear();
    batch.total_atoms = 0;
    batch.descriptors.reserve(smiles_list.size());
    batch.first_atoms.reserve(smiles_list.size());

    RDKit::SmilesParserParams parser_params;
    parser_params.sanitize = true;
    parser_params.removeHs = false;
    parser_params.strictCXSMILES = false;

    std::unique_ptr<RDKit::RWMol> molecule;
    for (auto const &smiles : smiles_list) {
        try {
            molecule.reset(RDKit::SmilesToMol(smiles, parser_params));
            if (!molecule) continue; // RDKit refused the SMILES
            RDKit::MolOps::addHs(*molecule);
        }
        catch (...) {
            continue; // parsing, sanitization, or hydrogen addition threw
        }
        // A sample ceiling shared with the sibling MMFF tests, well under the descriptor's own
        // capacity: this gates the extractor on drug-like input rather than on the atom limit.
        if (molecule->getNumAtoms() > 128) continue;
        auto &descriptor = batch.descriptors.emplace_back();
        if (!sft::build_descriptor(*molecule, descriptor)) {
            batch.descriptors.pop_back(); // above the descriptor's atom or bond capacity
            continue;
        }
        batch.first_atoms.push_back(batch.total_atoms);
        batch.total_atoms += descriptor.num_atoms;
    }
}

#pragma endregion Batch

#pragma region Comparisons

/**
 *  @brief One reporter per compared section, so every section closes with its own `compared=` count.
 *
 *  A single reporter cannot serve them all: `summarize` closes the span since the previous call, and
 *  these sections are interleaved -- every molecule contributes to each of them in turn.
 */
struct extract_reporters_t {
    diagnostic_reporter_t term_count {"cpu", "gpu"};
    diagnostic_reporter_t bond {"cpu", "gpu"};
    diagnostic_reporter_t angle {"cpu", "gpu"};
    diagnostic_reporter_t torsion {"cpu", "gpu"};
    diagnostic_reporter_t oop {"cpu", "gpu"};
    diagnostic_reporter_t stretch_bend {"cpu", "gpu"};
    diagnostic_reporter_t csr {"cpu", "gpu"};
    diagnostic_reporter_t packed {"staged", "packed"};

    /** Cap what each section prints, so one pathological molecule cannot bury the log. */
    void limit_printed_lines(std::size_t maximum) noexcept {
        term_count.limit_printed_lines(maximum);
        bond.limit_printed_lines(maximum);
        angle.limit_printed_lines(maximum);
        torsion.limit_printed_lines(maximum);
        oop.limit_printed_lines(maximum);
        stretch_bend.limit_printed_lines(maximum);
        csr.limit_printed_lines(maximum);
        packed.limit_printed_lines(maximum);
    }

    /** Close every section in order; true when each compared something and every field agreed. */
    bool summarize_sections() {
        term_count.summarize("term_count");
        bond.summarize("bond");
        angle.summarize("angle");
        torsion.summarize("torsion");
        oop.summarize("oop");
        stretch_bend.summarize("stretchbend");
        csr.summarize("csr");
        packed.summarize("packed");
        return term_count.passed() && bond.passed() && angle.passed() && torsion.passed() && oop.passed() &&
               stretch_bend.passed() && csr.passed() && packed.passed();
    }
};

/** What became of one molecule; the four states partition the batch. */
enum class molecule_outcome_t {
    /** Every compared field agreed. */
    matched_k,
    /** At least one field disagreed, or the term counts did and the arrays could not be walked. */
    mismatched_k,
    /** The GPU typer refused it, so it never extracted terms to compare. */
    typer_refused_k,
    /** The CPU extractor refused it, so there is no reference to hold the GPU to. */
    extractor_refused_k,
};

/** Per-molecule outcomes of the sweep; the four buckets partition the batch. */
struct extract_tally_t {
    std::uint32_t matched = 0;
    std::uint32_t mismatched = 0;
    std::uint32_t typer_refused = 0;
    std::uint32_t extractor_refused = 0;

    /** Bucket one molecule's outcome. */
    void record(molecule_outcome_t outcome) noexcept {
        switch (outcome) {
        case molecule_outcome_t::matched_k: ++matched; break;
        case molecule_outcome_t::mismatched_k: ++mismatched; break;
        case molecule_outcome_t::typer_refused_k: ++typer_refused; break;
        case molecule_outcome_t::extractor_refused_k: ++extractor_refused; break;
        }
    }

    /** Molecules no comparison could be made for, whichever side refused them. */
    std::uint32_t skipped() const noexcept { return typer_refused + extractor_refused; }
};

/**
 *  @brief Compare the five per-molecule term counts, returning how many of them disagree.
 *
 *  Term arrays of different lengths cannot be walked in step, so a caller that sees a non-zero
 *  answer must stop rather than index into them.
 */
static std::size_t compare_term_counts_(diagnostic_reporter_t &reporter, std::uint32_t molecule_index,
                                        sfp::etkdg_prepared_molecule_t const &cpu_reference,
                                        sfmmff::mmff94_per_mol_counts_t const &gpu_counts) {
    auto molecule = reporter.mismatch("mmff_extract", "molecule={}", molecule_index);
    molecule.field("bonds", cpu_reference.mmff_bonds.size(), std::size_t {gpu_counts.bonds}, exactly_equal_)
        .field("angles", cpu_reference.mmff_angles.size(), std::size_t {gpu_counts.angles}, exactly_equal_)
        .field("torsions", cpu_reference.mmff_torsions.size(), std::size_t {gpu_counts.torsions}, exactly_equal_)
        .field("oops", cpu_reference.mmff_oops.size(), std::size_t {gpu_counts.oops}, exactly_equal_)
        .field("stretchbends", cpu_reference.mmff_stretchbends.size(), std::size_t {gpu_counts.stretchbends},
               exactly_equal_);
    return molecule.reported_field_count();
}

/*  Each comparison below walks one term array of one molecule and answers with the number of terms
 *  that differ in any field. The loops are deliberately non-short-circuiting: their counts are what
 *  the molecule verdict and the section's `compared=` line are built from, and stopping at the first
 *  bad term would understate both. */

static std::size_t compare_bonds_(diagnostic_reporter_t &reporter, std::uint32_t molecule_index,
                                  std::vector<sft::mmff94_bond_t> const &cpu_terms,
                                  sft::mmff94_bond_t const *gpu_terms) {
    std::size_t mismatching_terms = 0;
    for (std::size_t term_index = 0; term_index < cpu_terms.size(); ++term_index) {
        auto const &cpu_term = cpu_terms[term_index];
        auto const &gpu_term = gpu_terms[term_index];
        auto term = reporter.mismatch("bond", "molecule={} term={}", molecule_index, term_index);
        term.field("atom_i", cpu_term.atom_i, gpu_term.atom_i, exactly_equal_)
            .field("atom_j", cpu_term.atom_j, gpu_term.atom_j, exactly_equal_)
            .field("force_constant", cpu_term.force_constant, gpu_term.force_constant, within_tolerance_)
            .field("equilibrium_distance", cpu_term.equilibrium_distance, gpu_term.equilibrium_distance,
                   within_tolerance_);
        mismatching_terms += term.reported_field_count() != 0;
    }
    return mismatching_terms;
}

static std::size_t compare_angles_(diagnostic_reporter_t &reporter, std::uint32_t molecule_index,
                                   std::vector<sft::mmff94_angle_t> const &cpu_terms,
                                   sft::mmff94_angle_t const *gpu_terms) {
    std::size_t mismatching_terms = 0;
    for (std::size_t term_index = 0; term_index < cpu_terms.size(); ++term_index) {
        auto const &cpu_term = cpu_terms[term_index];
        auto const &gpu_term = gpu_terms[term_index];
        auto term = reporter.mismatch("angle", "molecule={} term={}", molecule_index, term_index);
        term.field("atom_i", cpu_term.atom_i, gpu_term.atom_i, exactly_equal_)
            .field("atom_j", cpu_term.atom_j, gpu_term.atom_j, exactly_equal_)
            .field("atom_k", cpu_term.atom_k, gpu_term.atom_k, exactly_equal_)
            .field("force_constant", cpu_term.force_constant, gpu_term.force_constant, within_tolerance_)
            .field("equilibrium_angle", cpu_term.equilibrium_angle, gpu_term.equilibrium_angle, within_tolerance_);
        mismatching_terms += term.reported_field_count() != 0;
    }
    return mismatching_terms;
}

static std::size_t compare_torsions_(diagnostic_reporter_t &reporter, std::uint32_t molecule_index,
                                     std::vector<sft::mmff94_torsion_t> const &cpu_terms,
                                     sft::mmff94_torsion_t const *gpu_terms) {
    std::size_t mismatching_terms = 0;
    for (std::size_t term_index = 0; term_index < cpu_terms.size(); ++term_index) {
        auto const &cpu_term = cpu_terms[term_index];
        auto const &gpu_term = gpu_terms[term_index];
        auto term = reporter.mismatch("torsion", "molecule={} term={}", molecule_index, term_index);
        term.field("atom_i", cpu_term.atom_i, gpu_term.atom_i, exactly_equal_)
            .field("atom_j", cpu_term.atom_j, gpu_term.atom_j, exactly_equal_)
            .field("atom_k", cpu_term.atom_k, gpu_term.atom_k, exactly_equal_)
            .field("atom_l", cpu_term.atom_l, gpu_term.atom_l, exactly_equal_)
            .field("torsion_v1", cpu_term.torsion_v1, gpu_term.torsion_v1, within_tolerance_)
            .field("torsion_v2", cpu_term.torsion_v2, gpu_term.torsion_v2, within_tolerance_)
            .field("torsion_v3", cpu_term.torsion_v3, gpu_term.torsion_v3, within_tolerance_);
        mismatching_terms += term.reported_field_count() != 0;
    }
    return mismatching_terms;
}

static std::size_t compare_oops_(diagnostic_reporter_t &reporter, std::uint32_t molecule_index,
                                 std::vector<sft::mmff94_oop_t> const &cpu_terms, sft::mmff94_oop_t const *gpu_terms) {
    std::size_t mismatching_terms = 0;
    for (std::size_t term_index = 0; term_index < cpu_terms.size(); ++term_index) {
        auto const &cpu_term = cpu_terms[term_index];
        auto const &gpu_term = gpu_terms[term_index];
        auto term = reporter.mismatch("oop", "molecule={} term={}", molecule_index, term_index);
        term.field("atom_i", cpu_term.atom_i, gpu_term.atom_i, exactly_equal_)
            .field("atom_j", cpu_term.atom_j, gpu_term.atom_j, exactly_equal_)
            .field("atom_k", cpu_term.atom_k, gpu_term.atom_k, exactly_equal_)
            .field("atom_l", cpu_term.atom_l, gpu_term.atom_l, exactly_equal_)
            .field("force_constant", cpu_term.force_constant, gpu_term.force_constant, within_tolerance_);
        mismatching_terms += term.reported_field_count() != 0;
    }
    return mismatching_terms;
}

static std::size_t compare_stretch_bends_(diagnostic_reporter_t &reporter, std::uint32_t molecule_index,
                                          std::vector<sft::mmff94_stretchbend_t> const &cpu_terms,
                                          sft::mmff94_stretchbend_t const *gpu_terms) {
    std::size_t mismatching_terms = 0;
    for (std::size_t term_index = 0; term_index < cpu_terms.size(); ++term_index) {
        auto const &cpu_term = cpu_terms[term_index];
        auto const &gpu_term = gpu_terms[term_index];
        auto term = reporter.mismatch("stretchbend", "molecule={} term={}", molecule_index, term_index);
        term.field("atom_i", cpu_term.atom_i, gpu_term.atom_i, exactly_equal_)
            .field("atom_j", cpu_term.atom_j, gpu_term.atom_j, exactly_equal_)
            .field("atom_k", cpu_term.atom_k, gpu_term.atom_k, exactly_equal_)
            .field("kba_ijk", cpu_term.kba_ijk, gpu_term.kba_ijk, within_tolerance_)
            .field("kba_kji", cpu_term.kba_kji, gpu_term.kba_kji, within_tolerance_)
            .field("equilibrium_angle", cpu_term.equilibrium_angle, gpu_term.equilibrium_angle, within_tolerance_)
            .field("equilibrium_distance_ij", cpu_term.equilibrium_distance_ij, gpu_term.equilibrium_distance_ij,
                   within_tolerance_)
            .field("equilibrium_distance_kj", cpu_term.equilibrium_distance_kj, gpu_term.equilibrium_distance_kj,
                   within_tolerance_);
        mismatching_terms += term.reported_field_count() != 0;
    }
    return mismatching_terms;
}

/**
 *  @brief Compare the per-atom CSR term counts, returning how many atoms disagree.
 *
 *  Both sides number their offsets from zero within the molecule, so the counts rather than the
 *  offsets are what carries across. Per-atom ordering is deterministic on both sides: atoms are
 *  walked in index order and term types in bond, angle, torsion, oop, stretch-bend order.
 */
static std::size_t compare_csr_counts_(diagnostic_reporter_t &reporter, std::uint32_t molecule_index,
                                       std::uint32_t num_atoms,
                                       std::vector<sft::mmff94_atom_offsets_t> const &cpu_offsets,
                                       sft::mmff94_atom_offsets_t const *gpu_offsets) {
    std::size_t mismatching_atoms = 0;
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        auto const &cpu_offset = cpu_offsets[atom_index];
        auto const &gpu_offset = gpu_offsets[atom_index];
        auto atom = reporter.mismatch("csr", "molecule={} atom={}", molecule_index, atom_index);
        atom.field("bond_count", cpu_offset.bond_count, gpu_offset.bond_count, exactly_equal_)
            .field("angle_count", cpu_offset.angle_count, gpu_offset.angle_count, exactly_equal_)
            .field("torsion_count", cpu_offset.torsion_count, gpu_offset.torsion_count, exactly_equal_)
            .field("oop_count", cpu_offset.oop_count, gpu_offset.oop_count, exactly_equal_)
            .field("stretchbend_count", cpu_offset.stretchbend_count, gpu_offset.stretchbend_count, exactly_equal_);
        mismatching_atoms += atom.reported_field_count() != 0;
    }
    return mismatching_atoms;
}

/** Byte-compare a staged slice against its image inside the packed array. */
static bool slices_equal_(void const *staged, void const *packed_slice, std::size_t bytes) {
    return std::memcmp(staged, packed_slice, bytes) == 0;
}

/**
 *  @brief Check the compaction pass, returning how many of its fields disagree.
 *
 *  The staged slices are already known good against the CPU by the time this runs, so compaction is
 *  correct exactly when the packed arrays reproduce them byte for byte at the metadata-declared
 *  offsets, with every CSR start shifted into the batch-global interaction array.
 */
static std::size_t compare_packed_layout_(diagnostic_reporter_t &reporter, extract_batch_t const &batch,
                                          std::uint32_t molecule_index) {
    auto const &gpu_counts = batch.counts[molecule_index];
    auto const &meta = batch.packed.metadata[molecule_index];
    std::uint32_t const num_atoms = batch.descriptors[molecule_index].num_atoms;
    std::uint32_t const atom_offset_base = batch.first_atoms[molecule_index];

    auto layout = reporter.mismatch("packed", "molecule={}", molecule_index);
    layout.field("num_mmff_bonds", std::uint32_t {gpu_counts.bonds}, meta.num_mmff_bonds, exactly_equal_)
        .field("num_mmff_angles", std::uint32_t {gpu_counts.angles}, meta.num_mmff_angles, exactly_equal_)
        .field("num_mmff_torsions", std::uint32_t {gpu_counts.torsions}, meta.num_mmff_torsions, exactly_equal_)
        .field("num_mmff_oops", std::uint32_t {gpu_counts.oops}, meta.num_mmff_oops, exactly_equal_)
        .field("num_mmff_stretchbends", std::uint32_t {gpu_counts.stretchbends}, meta.num_mmff_stretchbends,
               exactly_equal_)
        .field("mmff_atom_csr_offset", atom_offset_base, meta.mmff_atom_csr_offset, exactly_equal_);
    // The offsets below index the packed arrays, so reading them is only safe once they agree.
    if (layout.reported_field_count() != 0) return layout.reported_field_count();

    layout
        .field("bond_slice", true,
               slices_equal_(batch.bonds_of(molecule_index), batch.packed.bonds.data() + meta.mmff_bond_offset,
                             gpu_counts.bonds * sizeof(sft::mmff94_bond_t)),
               exactly_equal_)
        .field("angle_slice", true,
               slices_equal_(batch.angles_of(molecule_index), batch.packed.angles.data() + meta.mmff_angle_offset,
                             gpu_counts.angles * sizeof(sft::mmff94_angle_t)),
               exactly_equal_)
        .field("torsion_slice", true,
               slices_equal_(batch.torsions_of(molecule_index), batch.packed.torsions.data() + meta.mmff_torsion_offset,
                             gpu_counts.torsions * sizeof(sft::mmff94_torsion_t)),
               exactly_equal_)
        .field("oop_slice", true,
               slices_equal_(batch.oops_of(molecule_index), batch.packed.oops.data() + meta.mmff_oop_offset,
                             gpu_counts.oops * sizeof(sft::mmff94_oop_t)),
               exactly_equal_)
        .field("stretchbend_slice", true,
               slices_equal_(batch.stretch_bends_of(molecule_index),
                             batch.packed.stretchbends.data() + meta.mmff_stretchbend_offset,
                             gpu_counts.stretchbends * sizeof(sft::mmff94_stretchbend_t)),
               exactly_equal_);
    // An atomless molecule has no CSR run, so the interaction base below would not exist.
    if (num_atoms == 0) return layout.reported_field_count();

    auto const staged_offsets = batch.staged_offsets_of(molecule_index);
    std::uint32_t const interaction_base = batch.packed.atom_offsets[atom_offset_base].bond_offset;
    layout.field(
        "interaction_slice", true,
        slices_equal_(batch.interactions_of(molecule_index), batch.packed.interactions.data() + interaction_base,
                      gpu_counts.interactions * sizeof(sft::mmff94_atom_interaction_t)),
        exactly_equal_);
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        auto const &staged_offset = staged_offsets[atom_index];
        auto const &packed_offset = batch.packed.atom_offsets[atom_offset_base + atom_index];
        layout
            .field("bond_offset", staged_offset.bond_offset + interaction_base, packed_offset.bond_offset,
                   exactly_equal_)
            .field("angle_offset", staged_offset.angle_offset + interaction_base, packed_offset.angle_offset,
                   exactly_equal_)
            .field("torsion_offset", staged_offset.torsion_offset + interaction_base, packed_offset.torsion_offset,
                   exactly_equal_)
            .field("oop_offset", staged_offset.oop_offset + interaction_base, packed_offset.oop_offset, exactly_equal_)
            .field("stretchbend_offset", staged_offset.stretchbend_offset + interaction_base,
                   packed_offset.stretchbend_offset, exactly_equal_);
    }
    return layout.reported_field_count();
}

/**
 *  @brief Cross-validate one molecule's staged and packed GPU terms against the CPU extractor.
 *
 *  @param[inout] reporters One section per term class; each accumulates this molecule's fields.
 *  @param[in] tables MMFF94 parameter tables the CPU reference reads.
 *  @param[in] batch The staged batch, holding both the descriptors and everything the GPU wrote.
 *  @param[in] molecule_index Which molecule of the batch to compare.
 *  @param[inout] cpu_reference Scratch the CPU extractor refills; hoisted so it grows once per run.
 *  @return Which of the four buckets this molecule falls in.
 */
static molecule_outcome_t compare_molecule_(extract_reporters_t &reporters, sfparam::mmff94_tables_view_t const &tables,
                                            extract_batch_t const &batch, std::uint32_t molecule_index,
                                            sfp::etkdg_prepared_molecule_t &cpu_reference) {
    if (batch.typing_status[molecule_index] != 0) return molecule_outcome_t::typer_refused_k;

    auto const &descriptor = batch.descriptors[molecule_index];
    cpu_reference.num_atoms = descriptor.num_atoms;
    if (!sfmmff::extract_mmff94_params_fully_native(descriptor, tables, cpu_reference))
        return molecule_outcome_t::extractor_refused_k;

    // Term arrays of different lengths cannot be walked in step, so the molecule stops here.
    if (compare_term_counts_(reporters.term_count, molecule_index, cpu_reference, batch.counts[molecule_index]) != 0)
        return molecule_outcome_t::mismatched_k;

    std::size_t const mismatches =
        compare_bonds_(reporters.bond, molecule_index, cpu_reference.mmff_bonds, batch.bonds_of(molecule_index)) +
        compare_angles_(reporters.angle, molecule_index, cpu_reference.mmff_angles, batch.angles_of(molecule_index)) +
        compare_torsions_(reporters.torsion, molecule_index, cpu_reference.mmff_torsions,
                          batch.torsions_of(molecule_index)) +
        compare_oops_(reporters.oop, molecule_index, cpu_reference.mmff_oops, batch.oops_of(molecule_index)) +
        compare_stretch_bends_(reporters.stretch_bend, molecule_index, cpu_reference.mmff_stretchbends,
                               batch.stretch_bends_of(molecule_index)) +
        compare_csr_counts_(reporters.csr, molecule_index, descriptor.num_atoms, cpu_reference.mmff_atom_offsets,
                            batch.staged_offsets_of(molecule_index)) +
        compare_packed_layout_(reporters.packed, batch, molecule_index);
    return mismatches == 0 ? molecule_outcome_t::matched_k : molecule_outcome_t::mismatched_k;
}

#pragma endregion Comparisons

#pragma region Reporting

/**
 *  @brief Print what the batch cost staged against packed, and how much of it took the fallback.
 *
 *  The staging arrays are fixed-stride, so their cost is the same for every molecule; the packed
 *  arrays cost what the batch actually holds. Both numbers are per molecule, so the ratio between
 *  them is the compaction win.
 */
static void report_footprint_(extract_batch_t const &batch, extract_tally_t const &tally) {
    std::size_t const packed_bytes = batch.packed.bonds.size() * sizeof(sft::mmff94_bond_t) +
                                     batch.packed.angles.size() * sizeof(sft::mmff94_angle_t) +
                                     batch.packed.torsions.size() * sizeof(sft::mmff94_torsion_t) +
                                     batch.packed.oops.size() * sizeof(sft::mmff94_oop_t) +
                                     batch.packed.stretchbends.size() * sizeof(sft::mmff94_stretchbend_t) +
                                     batch.packed.interactions.size() * sizeof(sft::mmff94_atom_interaction_t);
    double const molecules = static_cast<double>(batch.num_molecules());
    fmt::println("bytes/mol: staged {}, packed {:.1f} ({:.1f}x)", fmt::group_digits(staged_bytes_per_molecule_k),
                 static_cast<double>(packed_bytes) / molecules,
                 static_cast<double>(staged_bytes_per_molecule_k) * molecules / static_cast<double>(packed_bytes));
    fmt::println("fallback: typer {:.3f} % ({}), extractor {:.3f} % ({}), combined {:.3f} % of {}",
                 100.0 * tally.typer_refused / molecules, fmt::group_digits(tally.typer_refused),
                 100.0 * tally.extractor_refused / molecules, fmt::group_digits(tally.extractor_refused),
                 100.0 * tally.skipped() / molecules, fmt::group_digits(batch.num_molecules()));
}

#pragma endregion Reporting

} // namespace smallfield::tests

int main(int argc, char **argv) {
    fmt::println("{}: GPU vs CPU MMFF extractor byte-equality gate", test_name_k);

    auto const corpus = sftest::select_corpus(argc, argv, 200);
    if (corpus.knob == sftest::corpus_knob_t::refused_k) {
        std::fflush(stdout);
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", test_name_k);
        return 2;
    }

    std::vector<std::string> smiles_batch(std::begin(sftest::builtin_smiles_k), std::end(sftest::builtin_smiles_k));
    if (corpus.knob == sftest::corpus_knob_t::named_k) {
        std::vector<std::string> corpus_smiles;
        auto const outcome = sftest::load_smiles_into(corpus.path, corpus.max_molecules, corpus_smiles);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), corpus.path);
        for (auto &smiles : corpus_smiles) smiles_batch.push_back(std::move(smiles));
    }

    sfparam::mmff94_tables_host_t tables_host {};
    sfparam::extract_mmff94_tables_from_rdkit(tables_host);
    sfparam::mmff94_tables_view_t const host_view = tables_host.view();

    sftest::extract_batch_t batch;
    sftest::fill_descriptors_into_(smiles_batch, batch);
    if (batch.num_molecules() == 0)
        return sftest::report_fail(test_name_k, "none of the {} SMILES yielded a descriptor", smiles_batch.size());
    batch.resize_staging();
    fmt::println("batch: {} molecules, {} total atoms", fmt::group_digits(batch.num_molecules()),
                 fmt::group_digits(batch.total_atoms));

    sftest::launch_mmff94_gpu_extract_pipeline(
        tables_host, batch.descriptors.data(), batch.num_molecules(), batch.first_atoms.data(), batch.total_atoms,
        batch.counts.data(), batch.typing_status.data(), batch.bonds.data(), sft::max_mmff_bonds_per_mol_k,
        batch.angles.data(), sft::max_mmff_angles_per_mol_k, batch.torsions.data(), sft::max_mmff_torsions_per_mol_k,
        batch.oops.data(), sft::max_mmff_oops_per_mol_k, batch.stretch_bends.data(),
        sft::max_mmff_stretchbends_per_mol_k, batch.interactions.data(), batch.staged_atom_offsets.data(),
        batch.packed);

    // The CPU reference comes from `extract_mmff94_params_fully_native`, one molecule at a time. The
    // sweep is deliberately non-short-circuiting: every molecule feeds the section counts below.
    sftest::extract_reporters_t reporters;
    reporters.limit_printed_lines(sftest::max_printed_lines_k);
    sftest::extract_tally_t tally;
    sfp::etkdg_prepared_molecule_t cpu_reference;
    for (std::uint32_t molecule = 0; molecule < batch.num_molecules(); ++molecule)
        tally.record(sftest::compare_molecule_(reporters, host_view, batch, molecule, cpu_reference));

    fmt::println("molecules: {} match / {} bad / {} skipped-to-fallback", fmt::group_digits(tally.matched),
                 fmt::group_digits(tally.mismatched), fmt::group_digits(tally.skipped()));
    bool const sections_passed = reporters.summarize_sections();
    sftest::report_footprint_(batch, tally);

    if (tally.mismatched > 0)
        return sftest::report_fail(test_name_k, "{} of {} molecules diverge from the CPU extractor", tally.mismatched,
                                   batch.num_molecules());
    if (!sections_passed)
        return sftest::report_fail(test_name_k, "a comparison section mismatched or compared nothing at all");
    return sftest::report_pass(test_name_k);
}
