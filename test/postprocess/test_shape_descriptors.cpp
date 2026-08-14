/**
 *  @file test/postprocess/test_shape_descriptors.cpp
 *  @brief Confirm native USR / USRCAT match RDKit within tolerance, and that fragment separation is rigid.
 *
 *  Each molecule is parsed with the library's parse contract, embedded as one ETKDGv3 conformer with a fixed
 *  seed and MMFF-optimized; that conformer's float coordinates then feed both the native descriptors and
 *  RDKit's reference `Descriptors::USR` / `USRCAT`. To compare like with like, the RDKit conformer is
 *  overwritten with the same float-rounded coordinates the native path sees, so any residual difference is
 *  accumulation order only. Every one of the 12 USR and 60 USRCAT elements is compared at 1e-4, rather than
 *  their worst deviation alone, so a single divergent moment names itself.
 *
 *  `separate_fragments` documents a rigid translation that strings the component centroids along +x with their
 *  bounding spheres clear, so the checks are exactly those three claims: intra-component distances unchanged,
 *  each centroid on the axis, and consecutive centroids at least the sum of their two radii apart. A gap that
 *  merely grew is not the contract — two fragments the embedder left far apart legitimately move closer.
 *
 *  The builtin molecules run unconditionally; naming a `.smi` file on the command line, or in
 *  `SMALLFIELD_TEST_CORPUS`, widens both halves, and every multi-component corpus molecule also reaches the
 *  separation checks on the conformer it was already embedded for.
 */

#include <cmath>   // `std::fabs`, `std::sqrt`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <algorithm>   // `std::max`
#include <array>       // `std::array`
#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <GraphMol/Descriptors/USRDescriptor.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/postprocess/fragment_separation.hpp"
#include "smallfield/postprocess/shape_descriptors.hpp"

#include "test/test_rdkit_reference_helpers.hpp"

namespace sfpost = smallfield::postprocess;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_shape_descriptors";

/** One pathological molecule must not bury the log; the reporters keep counting past this. */
static constexpr std::size_t max_printed_lines_k = 40;

/** Corpus lines walked when neither `argv[2]` nor `SMALLFIELD_TEST_MAX_MOLS` caps the sweep. */
static constexpr std::size_t default_max_molecules_k = 100;

/** Fixed so a rerun embeds the same conformer and a diff of two runs stays empty. */
static constexpr int embedding_seed_k = 0xC0FFEE;

/** Absolute agreement demanded of every USR and USRCAT element. */
static constexpr double descriptor_tolerance_k = 1e-4;

/**
 *  Relative slack on the intra-component distance sum, which a rigid translation leaves alone.
 *
 *  The sum is over every pair in the component, so on a 400-atom fragment it reaches five figures and carries
 *  the float coordinates' rounding proportionally: an absolute bound would reject a translation that moved it
 *  by 3e-8 of itself. Any transform that is not rigid moves the sum by orders of magnitude more than this.
 */
static constexpr double rigid_relative_tolerance_k = 1e-5;

/** Absolute slack on a centroid the separation put on the x axis, where the coordinate is near zero. */
static constexpr double on_axis_tolerance_k = 1e-3;

namespace smallfield::tests {

#pragma region Preparation

/** The USR slots, in the reference-point-major order `usr_descriptor` writes them. */
static constexpr std::array<std::string_view, sfpost::usr_components_k> usr_slot_names_k {
    "ctd_mean", "ctd_standard_deviation", "ctd_skewness", //
    "cst_mean", "cst_standard_deviation", "cst_skewness", //
    "fct_mean", "fct_standard_deviation", "fct_skewness", //
    "ftf_mean", "ftf_standard_deviation", "ftf_skewness",
};

/** USRCAT's blocks, the plain USR one and then the CREDO subsets, in the order `usrcat_descriptor` writes them. */
static constexpr std::array<std::string_view, sfpost::credo_subset_count_k + 1> usrcat_block_names_k {
    "all", "hydrophobic", "aromatic", "acceptor", "donor",
};

/** How far a molecule got before the descriptors could be compared. */
enum class prepare_outcome_t {
    prepared_k,
    /** `SmilesToMol` refused the line, or threw on it. */
    unparsed_k,
    /** ETKDGv3 found no conformer at this seed. */
    unembedded_k,
    /** The parser's aromaticity perception could not be restored after MMFF typing rewrote it. */
    unsanitized_k,
};

/**
 *  @brief Parse with the same contract `build_shape_context_into` uses, add Hs, embed one ETKDGv3 conformer at a
 *  fixed seed, and MMFF-optimize it.
 *
 *  @param[in] smiles SMILES to prepare.
 *  @param[out] molecule Owns the prepared molecule on `prepared_k`, and is left null otherwise.
 *  @return Which of the two ways preparation can end short, or `prepared_k`.
 */
static prepare_outcome_t prepare_molecule_(std::string_view smiles, std::unique_ptr<RDKit::RWMol> &molecule) {
    molecule.reset();
    try {
        RDKit::SmilesParserParams params;
        params.sanitize = true;
        params.removeHs = false;
        params.strictCXSMILES = false;
        molecule.reset(RDKit::SmilesToMol(std::string {smiles}, params));
        if (!molecule) return prepare_outcome_t::unparsed_k;
        RDKit::MolOps::addHs(*molecule);
    }
    catch (...) {
        molecule.reset();
        return prepare_outcome_t::unparsed_k;
    }

    RDKit::DGeomHelpers::EmbedParameters embed_params = RDKit::DGeomHelpers::ETKDGv3;
    embed_params.randomSeed = embedding_seed_k;
    int const conformer_id = RDKit::DGeomHelpers::EmbedMolecule(*molecule, embed_params);
    if (conformer_id < 0) {
        molecule.reset();
        return prepare_outcome_t::unembedded_k;
    }
    RDKit::MMFF::MMFFOptimizeMolecule(*molecule);

    // MMFF typing kekulizes the molecule and re-perceives aromaticity under the MMFF94 model, which marks
    // strictly fewer atoms than the model the parser used — 6 rather than 20 on a fused heteroaromatic.
    // RDKit's reference USRCAT matches its `[a]` subset against this same object, so leaving MMFF's
    // perception in place types the reference on a different graph from the one `build_shape_context_into` saw.
    try {
        RDKit::MolOps::sanitizeMol(*molecule);
    }
    catch (...) {
        molecule.reset();
        return prepare_outcome_t::unsanitized_k;
    }
    return prepare_outcome_t::prepared_k;
}

#pragma endregion Preparation

#pragma region Reporting

/** One reporter per section, so a section's `compared=` is never diluted by its neighbour's. */
struct shape_reporters_t {
    diagnostic_reporter_t usr {"rdkit", "ours"};
    diagnostic_reporter_t usrcat {"rdkit", "ours"};
    diagnostic_reporter_t separation {"before", "after"};

    void limit_printed_lines(std::size_t maximum) {
        usr.limit_printed_lines(maximum);
        usrcat.limit_printed_lines(maximum);
        separation.limit_printed_lines(maximum);
    }

    bool passed() const { return usr.passed() && usrcat.passed() && separation.passed(); }

    std::size_t mismatch_count() const {
        return usr.mismatch_count() + usrcat.mismatch_count() + separation.mismatch_count();
    }

    std::size_t comparison_count() const {
        return usr.comparison_count() + usrcat.comparison_count() + separation.comparison_count();
    }
};

/** What one component measured before `separate_fragments` moved it, against which the after side is checked. */
struct component_before_t {
    double intra_distance_sum = 0.0;
    std::array<double, 3> centroid {0.0, 0.0, 0.0};
};

/** Buffers that outlive the molecule loop, so a corpus sweep grows each of them once. */
struct shape_scratch_t {
    /** Typed once per molecule into the same object the production loops hoist, capacity included. */
    sfpost::molecule_shape_context_t context;
    std::vector<float> coordinates;
    std::array<double, sfpost::usr_components_k> native_usr {};
    std::array<double, sfpost::usrcat_components_k> native_usrcat {};
    std::vector<double> rdkit_usr;
    std::vector<double> rdkit_usrcat;
    std::vector<std::vector<unsigned int>> rdkit_usrcat_atoms;
    std::vector<std::uint32_t> component_atoms;
    std::vector<component_before_t> components_before;
};

/** Where every corpus line went, so a section that compared little says why. */
struct shape_tally_t {
    std::uint64_t compared = 0;
    std::uint64_t unparsed = 0;
    std::uint64_t unembedded = 0;
    std::uint64_t unsanitized = 0;
    std::uint64_t untyped = 0;
    std::uint64_t too_many_fragments = 0;
    std::uint64_t atom_count_divergence = 0;
    std::uint64_t separated = 0;
    std::uint64_t reference_refused = 0;
};

#pragma endregion Reporting

#pragma region Descriptors

/**
 *  @brief Compare all 72 native descriptor elements against RDKit's, on identical float coordinates.
 *
 *  @param[inout] molecule Prepared molecule; its conformer is overwritten with the float-rounded coordinates.
 *  @param[in] smiles The molecule's SMILES, carried into every diagnostic key.
 *  @param[inout] reporters Reporters counting and printing the comparisons.
 *  @param[inout] scratch Buffers reused across molecules, carrying this molecule's typing context.
 */
static void compare_descriptors_(RDKit::RWMol &molecule, std::string_view smiles, shape_reporters_t &reporters,
                                 shape_scratch_t &scratch) {
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    read_conformer_into(
        molecule, types::reset_coordinates_aos<types::coordinate_role_t::position_k>(scratch.coordinates, num_atoms));
    set_coords_interleaved(molecule, scratch.coordinates.data());

    sfpost::usr_descriptor(scratch.coordinates, num_atoms, scratch.native_usr);
    scratch.rdkit_usr.assign(scratch.native_usr.size(), 0.0);
    RDKit::Descriptors::USR(molecule, scratch.rdkit_usr);

    sfpost::usrcat_descriptor(scratch.coordinates, scratch.context, scratch.native_usrcat);
    scratch.rdkit_usrcat.assign(scratch.native_usrcat.size(), 0.0);
    // `USRCAT` reads the atom-id argument as an input whenever it arrives non-empty, so a reused buffer would
    // type this molecule with the previous one's CREDO subsets and compare two unrelated descriptors.
    scratch.rdkit_usrcat_atoms.clear();
    RDKit::Descriptors::USRCAT(molecule, scratch.rdkit_usrcat, scratch.rdkit_usrcat_atoms);

    auto const within_tolerance = [](double reference, double observed) {
        return std::fabs(reference - observed) < descriptor_tolerance_k;
    };

    auto usr_group = reporters.usr.mismatch("usr", "smiles={}", smiles);
    for (std::size_t slot = 0; slot < usr_slot_names_k.size(); ++slot)
        usr_group.field(usr_slot_names_k[slot], scratch.rdkit_usr[slot], scratch.native_usr[slot], within_tolerance);

    for (std::size_t block = 0; block < usrcat_block_names_k.size(); ++block) {
        auto usrcat_group = reporters.usrcat.mismatch("usrcat", "smiles={} subset={}", smiles,
                                                      usrcat_block_names_k[block]);
        // Block 0 is the plain USR over every atom; the four CREDO subsets follow it, and RDKit hands back the
        // atom ids it typed, so a typing divergence names itself here instead of surfacing as twelve moments.
        if (block > 0)
            usrcat_group.field("subset_size", scratch.rdkit_usrcat_atoms[block - 1].size(),
                               scratch.context.credo_subsets[block - 1].size(), exactly_equal);
        std::size_t const base = block * usr_slot_names_k.size();
        for (std::size_t slot = 0; slot < usr_slot_names_k.size(); ++slot)
            usrcat_group.field(usr_slot_names_k[slot], scratch.rdkit_usrcat[base + slot],
                               scratch.native_usrcat[base + slot], within_tolerance);
    }
}

#pragma endregion Descriptors

#pragma region Fragment Separation

/*  The helpers below re-derive the centroid, the radius and the pairwise distances rather than calling
 *  `centroid_of_` and `point_distance_` from `shape_descriptors.hpp`, which the code under test uses. Sharing
 *  those would let a defect in either appear on both sides of the comparison and cancel itself out.
 */

/** Sum of pairwise distances within one component — invariant under a rigid translation. */
static double intra_component_distance_sum_(std::vector<float> const &coords, std::vector<int> const &label,
                                            int component, std::vector<std::uint32_t> &component_atoms) {
    component_atoms.clear();
    for (std::uint32_t atom_index = 0; atom_index < label.size(); ++atom_index)
        if (label[atom_index] == component) component_atoms.push_back(atom_index);
    double total = 0.0;
    for (std::size_t first = 0; first < component_atoms.size(); ++first)
        for (std::size_t second = first + 1; second < component_atoms.size(); ++second) {
            std::size_t const first_base = static_cast<std::size_t>(component_atoms[first]) * 3u;
            std::size_t const second_base = static_cast<std::size_t>(component_atoms[second]) * 3u;
            double const delta_x = static_cast<double>(coords[first_base + 0]) - coords[second_base + 0];
            double const delta_y = static_cast<double>(coords[first_base + 1]) - coords[second_base + 1];
            double const delta_z = static_cast<double>(coords[first_base + 2]) - coords[second_base + 2];
            total += std::sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
        }
    return total;
}

static std::array<double, 3> component_centroid_(std::vector<float> const &coords, std::vector<int> const &label,
                                                 int component) {
    std::array<double, 3> centroid {0.0, 0.0, 0.0};
    std::uint32_t count = 0;
    for (std::uint32_t atom_index = 0; atom_index < label.size(); ++atom_index) {
        if (label[atom_index] != component) continue;
        std::size_t const base = static_cast<std::size_t>(atom_index) * 3u;
        centroid[0] += coords[base + 0];
        centroid[1] += coords[base + 1];
        centroid[2] += coords[base + 2];
        ++count;
    }
    if (count > 0) {
        centroid[0] /= count;
        centroid[1] /= count;
        centroid[2] /= count;
    }
    return centroid;
}

/** Bounding-sphere radius of one component about its own centroid. */
static double component_radius_(std::vector<float> const &coords, std::vector<int> const &label, int component,
                                std::array<double, 3> const &centroid) {
    double radius = 0.0;
    for (std::uint32_t atom_index = 0; atom_index < label.size(); ++atom_index) {
        if (label[atom_index] != component) continue;
        std::size_t const base = static_cast<std::size_t>(atom_index) * 3u;
        double const delta_x = static_cast<double>(coords[base + 0]) - centroid[0];
        double const delta_y = static_cast<double>(coords[base + 1]) - centroid[1];
        double const delta_z = static_cast<double>(coords[base + 2]) - centroid[2];
        double const distance = std::sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
        if (distance > radius) radius = distance;
    }
    return radius;
}

static double centroid_distance_(std::array<double, 3> const &first, std::array<double, 3> const &second) {
    double const delta_x = first[0] - second[0];
    double const delta_y = first[1] - second[1];
    double const delta_z = first[2] - second[2];
    return std::sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
}

/** How far a centroid sits off the +x axis `separate_fragments` lays the components along. */
static double centroid_offaxis_(std::array<double, 3> const &centroid) {
    return std::sqrt(centroid[1] * centroid[1] + centroid[2] * centroid[2]);
}

/**
 *  @brief Separate the components of one conformer and check the three claims the function documents.
 *
 *  Only consecutive centroid pairs are checked for clearance, because the components are laid along one axis in
 *  label order, so consecutive clearance implies every pair's.
 *
 *  @param[in] smiles The molecule's SMILES, carried into every diagnostic key.
 *  @param[inout] reporters Reporters counting and printing the comparisons.
 *  @param[inout] scratch Carries this molecule's multi-component context and the coordinates in, and is left
 *      holding the separated ones.
 */
static void check_separation_(std::string_view smiles, shape_reporters_t &reporters, shape_scratch_t &scratch) {
    sfpost::molecule_shape_context_t const &context = scratch.context;
    std::uint32_t const num_components = context.num_components;
    scratch.components_before.assign(num_components, component_before_t {});
    for (std::uint32_t component = 0; component < num_components; ++component) {
        int const label = static_cast<int>(component);
        component_before_t &before = scratch.components_before[component];
        before.intra_distance_sum = intra_component_distance_sum_(scratch.coordinates, context.component_label, label,
                                                                  scratch.component_atoms);
        before.centroid = component_centroid_(scratch.coordinates, context.component_label, label);
    }

    sfpost::separate_fragments(scratch.coordinates, context);

    auto const unchanged = [](double reference, double observed) {
        return std::fabs(reference - observed) <= rigid_relative_tolerance_k * std::max(1.0, std::fabs(reference));
    };
    // The claim is where the centroid ended up, not how far it moved, so the before value is reported as context
    // and deliberately not consulted.
    auto const on_axis = [](double, double observed) { return observed < on_axis_tolerance_k; };

    std::array<double, 3> previous_centroid {0.0, 0.0, 0.0};
    double previous_radius = 0.0;
    for (std::uint32_t component = 0; component < num_components; ++component) {
        int const label = static_cast<int>(component);
        double const intra_sum_after = intra_component_distance_sum_(scratch.coordinates, context.component_label,
                                                                     label, scratch.component_atoms);
        std::array<double, 3> const centroid_after = component_centroid_(scratch.coordinates, context.component_label,
                                                                         label);
        double const radius_after = component_radius_(scratch.coordinates, context.component_label, label,
                                                      centroid_after);

        component_before_t const &before = scratch.components_before[component];
        reporters.separation.mismatch("fragment_separation", "smiles={} component={}", smiles, component)
            .field("intra_distance_sum", before.intra_distance_sum, intra_sum_after, unchanged)
            .field("centroid_offaxis", centroid_offaxis_(before.centroid), centroid_offaxis_(centroid_after), on_axis);

        if (component > 0) {
            // The two bounding spheres must clear each other, which a gap that merely grew does not promise.
            double const clearance = previous_radius + radius_after;
            auto const clears = [clearance](double, double observed) { return observed >= clearance; };
            reporters.separation
                .mismatch("fragment_separation", "smiles={} components={}-{} clearance={:.3g}", smiles, component - 1,
                          component, clearance)
                .field("centroid_gap",
                       centroid_distance_(scratch.components_before[component - 1].centroid, before.centroid),
                       centroid_distance_(previous_centroid, centroid_after), clears);
        }
        previous_centroid = centroid_after;
        previous_radius = radius_after;
    }
}

#pragma endregion Fragment Separation

#pragma region Corpora

/** Molecules whose CREDO subsets between them reach every USRCAT block. */
static constexpr std::array<std::string_view, 6> builtin_smiles_k {
    "CCO",                    //
    "c1ccccc1",               //
    "CC(=O)Oc1ccccc1C(=O)O",  // aspirin
    "Nc1ccc(cc1)S(=O)(=O)N",  // sulfanilamide: donors + acceptors
    "OC(=O)CCc1cc(O)c(O)cc1", // multiple donors/acceptors/aromatic
    "CC(=O)Nc1ccc(O)cc1",     // paracetamol
};

/** The salt whose two components exercise `separate_fragments` when no corpus supplies one. */
static constexpr std::string_view builtin_salt_k = "CC(=O)[O-].[Na+]";

/** The builtin molecules are curated, so anything that stops one short of the comparison is a defect. */
static void check_builtin_(std::string_view smiles, shape_reporters_t &reporters, shape_scratch_t &scratch) {
    std::unique_ptr<RDKit::RWMol> molecule;
    sf_verify_(prepare_molecule_(smiles, molecule) == prepare_outcome_t::prepared_k,
               fmt::format("a builtin SMILES must parse and embed, but {} did not", smiles));

    // One verdict covers both claims a builtin has to meet: it types, and it types the molecule that was
    // just embedded, which is what makes the coordinates below address the atoms the context describes.
    sfpost::shape_typing_t const typing = sfpost::build_shape_context_into(
        smiles, molecule->getNumAtoms(), smallfield::keep_largest_fragment_k, scratch.context);
    sf_verify_(typing == sfpost::shape_typing_t::typed_k,
               fmt::format("a builtin SMILES must type against its own embedded molecule, but {} {}", smiles,
                           sfpost::describe_shape_typing(typing)));
    compare_descriptors_(*molecule, smiles, reporters, scratch);
}

/**
 *  @brief Run the separation checks over the builtin salt.
 *
 *  `separate_fragments` is a geometric utility over a multi-component context; the production path strips
 *  counterions before it can ever see one, so the context is built with every fragment kept.
 */
static void check_builtin_salt_(shape_reporters_t &reporters, shape_scratch_t &scratch) {
    std::unique_ptr<RDKit::RWMol> molecule;
    sf_verify_(prepare_molecule_(builtin_salt_k, molecule) == prepare_outcome_t::prepared_k, "builtin salt must embed");

    sfpost::shape_typing_t const typing = sfpost::build_shape_context_into(
        builtin_salt_k, molecule->getNumAtoms(), smallfield::keep_all_fragments_k, scratch.context);
    sf_verify_(
        typing == sfpost::shape_typing_t::typed_k,
        fmt::format("the builtin salt must type, but {} {}", builtin_salt_k, sfpost::describe_shape_typing(typing)));
    sf_verify_(scratch.context.num_components == 2, "the builtin salt must have exactly two components");

    read_conformer_into(*molecule, types::reset_coordinates_aos<types::coordinate_role_t::position_k>(
                                       scratch.coordinates, molecule->getNumAtoms()));
    check_separation_(builtin_salt_k, reporters, scratch);
}

/**
 *  @brief Walk one corpus line, counting every way it can miss a comparison instead of failing on it.
 *
 *  The context keeps every fragment here, because the reference descriptors come from a molecule parsed
 *  straight out of the same SMILES: a context that stripped a counterion would be typing a different molecule
 *  from the one RDKit measures.
 */
static void check_corpus_line_(std::string_view smiles, shape_reporters_t &reporters, shape_scratch_t &scratch,
                               shape_tally_t &tally) {
    std::unique_ptr<RDKit::RWMol> molecule;
    switch (prepare_molecule_(smiles, molecule)) {
    case prepare_outcome_t::unparsed_k: ++tally.unparsed; return;
    case prepare_outcome_t::unembedded_k: ++tally.unembedded; return;
    case prepare_outcome_t::unsanitized_k: ++tally.unsanitized; return;
    case prepare_outcome_t::prepared_k: break;
    }

    // Each refusal is a corpus line the comparison never reaches rather than a defect, so each is counted
    // under its own name and the section's `compared=` stays honest about what it saw.
    switch (sfpost::build_shape_context_into(smiles, molecule->getNumAtoms(), smallfield::keep_all_fragments_k,
                                             scratch.context)) {
    case sfpost::shape_typing_t::unparsed_k: ++tally.untyped; return;
    case sfpost::shape_typing_t::too_many_fragments_k: ++tally.too_many_fragments; return;
    case sfpost::shape_typing_t::atom_count_divergence_k: ++tally.atom_count_divergence; return;
    case sfpost::shape_typing_t::typed_k: break;
    }

    try {
        compare_descriptors_(*molecule, smiles, reporters, scratch);
    }
    catch (...) {
        ++tally.reference_refused;
        return;
    }
    ++tally.compared;

    if (scratch.context.num_components < 2) return; // nothing to pull apart
    ++tally.separated;
    check_separation_(smiles, reporters, scratch);
}

#pragma endregion Corpora

} // namespace smallfield::tests

int main(int argc, char **argv) {
    fmt::println("{}: native USR / USRCAT against RDKit, then rigid fragment separation", test_name_k);

    auto const selection = sftest::select_corpus(argc, argv, default_max_molecules_k);
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", argv[0]);
        return 2;
    }

    sftest::shape_reporters_t reporters;
    reporters.limit_printed_lines(max_printed_lines_k);
    sftest::shape_scratch_t scratch;

    for (std::string_view const smiles : sftest::builtin_smiles_k) sftest::check_builtin_(smiles, reporters, scratch);
    reporters.usr.summarize("usr_builtin");
    reporters.usrcat.summarize("usrcat_builtin");

    if (selection.knob == sftest::corpus_knob_t::named_k) {
        // Naming a corpus is a request to validate against it, so a path that does not open is a failure
        // rather than an empty run: the sections would otherwise close with nothing compared.
        std::vector<std::string> corpus;
        auto const outcome = sftest::load_smiles_into(selection.path, selection.max_molecules, corpus);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), selection.path);
        fmt::println("{}: corpus {}, {} SMILES", test_name_k, selection.path, fmt::group_digits(corpus.size()));

        sftest::shape_tally_t tally;
        for (auto const &smiles : corpus) sftest::check_corpus_line_(smiles, reporters, scratch, tally);
        fmt::println("  NOTE shape_corpus molecules={} separated={} unparsed={} unembedded={} "  //
                     "unsanitized={} untyped={} too_many_fragments={} atom_count_divergence={} " //
                     "reference_refused={}",                                                     //
                     tally.compared, tally.separated, tally.unparsed, tally.unembedded, tally.unsanitized,
                     tally.untyped, tally.too_many_fragments, tally.atom_count_divergence, tally.reference_refused);
        reporters.usr.summarize("usr_corpus");
        reporters.usrcat.summarize("usrcat_corpus");
    }

    // The builtin salt closes the separation section last, so that section covers the corpus fragments too and
    // still compares something when every corpus molecule turned out to be a single fragment.
    sftest::check_builtin_salt_(reporters, scratch);
    reporters.separation.summarize("fragment_separation");

    if (!reporters.passed())
        return sftest::report_fail(test_name_k, "{} mismatches over {} comparisons", reporters.mismatch_count(),
                                   reporters.comparison_count());
    return sftest::report_pass(test_name_k);
}
