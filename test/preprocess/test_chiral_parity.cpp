/**
 *  @file test/preprocess/test_chiral_parity.cpp
 *  @brief Police the tag-and-parity derivation of the chiral volume sign with two exact oracles.
 *
 *  `extract_chiral_centers` derives the target signed-volume sign from the atom's tetrahedral tag and the parity
 *  of the stored neighbor order. Two independent oracles police that derivation, and both demand exact agreement:
 *
 *  The CIP oracle rebuilds the answer straight from RDKit - `assignAtomCIPRanks` for the neighbor ordering, the
 *  `_CIPCode` property for the R/S label - and transports that sign onto the neighbor order the production code
 *  emits. It speaks for four-coordinate centers whose neighbors carry four distinct CIP ranks. A tie leaves the
 *  CIP ordering, and therefore the sign it implies, undetermined - the C3 of a tropane is exactly such a
 *  pseudo-asymmetric case. A three-coordinate center hides its lone pair from the ranking entirely, and RDKit's
 *  R/S label there does not compose with a triple product over the three bonded neighbors, as the sulfoxides in
 *  the curated set demonstrate against their own embedded geometry.
 *
 *  The geometric oracle embeds the molecule with ETKDG, which builds its chiral volume constraints from the same
 *  tetrahedral tags, and compares the sign of the realized tetrahedron against the target. It speaks at every
 *  center, tie or not.
 *
 *  Coverage comes from a curated set of stereo-rich molecules, an exhaustive enumeration of tetrahedral tag
 *  assignments over those molecules, and an optional SMILES corpus named by `argv[1]` or by
 *  `SMALLFIELD_TEST_CORPUS`, capped by `argv[2]` or `SMALLFIELD_TEST_MAX_MOLS`. A named corpus that
 *  does not open is a failure, never an empty run.
 */

#include <cmath>   // `std::fabs`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`, `std::uint8_t`

#include <algorithm>   // `std::find`, `std::min`, `std::sort`
#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <GraphMol/Chirality.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/types/views.hpp"

#include "test/test_helpers.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_chiral_parity";

/** A stereo-rich corpus can disagree at thousands of centers, and ten lines already name the pattern. */
static constexpr std::size_t max_printed_lines_k = 10;

namespace smallfield::tests {

/**
 *  @brief Census of the centers each oracle saw and of the ones it had to decline.
 *
 *  Only what the reporters below do not already hold: they own every comparison and every
 *  mismatch, and `summarize` prints those, so counting them a second time here would be two
 *  bookkeeping schemes for one fact.
 */
struct oracle_tally_t {
    std::uint64_t molecules = 0;
    std::uint64_t centers = 0;
    std::uint64_t three_coordinate = 0;
    std::uint64_t rank_ties = 0;
    std::uint64_t unlabeled = 0;
    std::uint64_t geometry_three_coordinate = 0;
    std::uint64_t embed_failures = 0;
};

/**
 *  @brief One diagnostic sink per oracle, so each closes its own `compared=` section.
 *
 *  The two speak for different populations - the CIP oracle declines ties and three-coordinate
 *  centers, the geometric one declines near-planar arrangements - so a single shared count could
 *  never say which oracle a comparison came from.
 */
struct oracle_reporters_t {
    diagnostic_reporter_t cip {"cip", "derived"};
    diagnostic_reporter_t geometry {"embedded", "derived"};

    void limit_printed_lines(std::size_t maximum) noexcept {
        cip.limit_printed_lines(maximum);
        geometry.limit_printed_lines(maximum);
    }
};

/**
 *  @brief Parity of the permutation carrying @p from onto @p to.
 *
 *  Both sequences hold the same atom indices in a different order. Counting the
 *  inversions of the position sequence gives the sign a signed volume picks up when
 *  its arguments are reordered.
 *
 *  @param[in] from Source ordering of atom indices.
 *  @param[in] to Target ordering, a permutation of @p from.
 *  @return +1 for an even permutation, -1 for an odd one, 0 when the two disagree.
 */
static int permutation_parity_(custd::span<std::uint8_t const> from, custd::span<std::uint8_t const> to) {
    std::uint32_t positions[4] = {0, 0, 0, 0};
    for (std::size_t target = 0; target < to.size(); ++target) {
        auto const source = std::find(from.begin(), from.end(), to[target]);
        if (source == from.end()) return 0;
        positions[target] = static_cast<std::uint32_t>(source - from.begin());
    }
    std::uint32_t inversions = 0;
    for (std::size_t left = 0; left < to.size(); ++left)
        for (std::size_t right = left + 1; right < to.size(); ++right)
            if (positions[left] > positions[right]) ++inversions;
    return (inversions % 2 == 0) ? 1 : -1;
}

/**
 *  @brief Whether any two entries of @p order share a CIP rank.
 *
 *  @param[in] cip_ranks CIP rank per atom index, as `assignAtomCIPRanks` filled it.
 *  @param[in] order Neighbor atom indices to inspect.
 *  @return True as soon as one repeated rank is found.
 */
static bool any_rank_tie_(RDKit::UINT_VECT const &cip_ranks, custd::span<std::uint8_t const> order) {
    for (std::size_t left = 0; left < order.size(); ++left)
        for (std::size_t right = left + 1; right < order.size(); ++right)
            if (cip_ranks[order[left]] == cip_ranks[order[right]]) return true;
    return false;
}

/**
 *  @brief Compare the extracted signs against RDKit's CIP labels.
 *
 *  @param[in] molecule Molecule already carrying assigned stereochemistry.
 *  @param[in] descriptor Flat descriptor the centers were extracted from.
 *  @param[in] centers Extracted chiral center descriptors.
 *  @param[in] label Printed alongside any mismatch so the offender is identifiable.
 *  @param[inout] reporters Sink that counts and prints the CIP oracle's comparisons.
 *  @param[inout] tally Counters advanced by this molecule.
 */
static void check_against_cip_(RDKit::ROMol const &molecule, sft::molecule_descriptor_t const &descriptor,
                               std::vector<sft::chiral_center_descriptor_t> const &centers, std::string const &label,
                               oracle_reporters_t &reporters, oracle_tally_t &tally) {
    RDKit::UINT_VECT cip_ranks(molecule.getNumAtoms(), 0);
    RDKit::Chirality::assignAtomCIPRanks(molecule, cip_ranks);

    for (auto const &center : centers) {
        std::uint32_t const atom = center.center_atom;
        auto const *rdkit_atom = molecule.getAtomWithIdx(atom);

        ++tally.centers;
        if (!rdkit_atom->hasProp("_CIPCode")) {
            ++tally.unlabeled;
            continue;
        }

        // Rebuild the neighbor order the CIP path produced: highest priority first,
        // ties broken by ascending atom index.
        std::uint8_t reference_order[4] = {0, 0, 0, 0};
        std::uint32_t count = 0;
        std::uint16_t const neighbor_first = descriptor.neighbor_offset[atom];
        std::uint16_t const neighbor_last = descriptor.neighbor_offset[atom + 1];
        for (std::uint16_t entry = neighbor_first; entry < neighbor_last && count < 4; ++entry)
            reference_order[count++] = descriptor.neighbor_atom[entry];
        // A lone pair never enters the CIP ranking, so a three-coordinate center has no
        // CIP ordering that the triple product can be transported onto. The geometric
        // oracle covers these centers instead.
        if (count < 4) {
            ++tally.three_coordinate;
            continue;
        }

        // A repeated rank leaves the CIP ordering, and with it the sign that ordering
        // implies, undetermined. The geometric oracle covers these centers instead.
        custd::span<std::uint8_t> const cip_ordered {reference_order, count};
        if (any_rank_tie_(cip_ranks, cip_ordered)) {
            ++tally.rank_ties;
            continue;
        }

        std::sort(cip_ordered.begin(), cip_ordered.end(), [&](std::uint8_t left, std::uint8_t right) {
            if (cip_ranks[left] != cip_ranks[right]) return cip_ranks[left] > cip_ranks[right];
            return left < right;
        });

        std::string cip_code;
        rdkit_atom->getProp("_CIPCode", cip_code);
        int const reference_sign = (cip_code == "S") ? 1 : -1;

        int const parity = permutation_parity_(cip_ordered, {center.neighbor_atoms, count});
        sf_verify_(parity != 0, "emitted neighbors must be a permutation of the CIP-ordered neighbors");
        int const expected_sign = reference_sign * parity;

        reporters.cip.mismatch("chiral_parity", "oracle=cip molecule={} atom={} cip_code={}", label, atom, cip_code)
            .field("target_sign", expected_sign, static_cast<int>(center.target_sign), exactly_equal);
    }
}

/**
 *  @brief Compare the extracted signs against an ETKDG embedding of the same molecule.
 *
 *  @param[in] molecule Molecule with explicit hydrogens; a copy of it is embedded.
 *  @param[in] descriptor Flat descriptor the centers were extracted from.
 *  @param[in] centers Extracted chiral center descriptors.
 *  @param[in] label Printed alongside any mismatch so the offender is identifiable.
 *  @param[inout] reporters Sink that counts and prints the geometric oracle's comparisons.
 *  @param[inout] tally Counters advanced by this molecule.
 */
static void check_against_geometry_(RDKit::ROMol const &molecule, sft::molecule_descriptor_t const &descriptor,
                                    std::vector<sft::chiral_center_descriptor_t> const &centers,
                                    std::string const &label, oracle_reporters_t &reporters, oracle_tally_t &tally) {
    RDKit::RWMol embedded(molecule);
    if (!sftest::embed_molecule(embedded)) {
        ++tally.embed_failures;
        return;
    }
    auto const &conformer = embedded.getConformer();

    for (auto const &center : centers) {
        auto const &origin = conformer.getAtomPos(center.center_atom);
        double edges[3][3] = {};
        for (std::uint32_t slot = 0; slot < 3; ++slot) {
            auto const &position = conformer.getAtomPos(center.neighbor_atoms[slot]);
            edges[slot][0] = position.x - origin.x;
            edges[slot][1] = position.y - origin.y;
            edges[slot][2] = position.z - origin.z;
        }
        double const cross[3] = {edges[1][1] * edges[2][2] - edges[1][2] * edges[2][1],
                                 edges[1][2] * edges[2][0] - edges[1][0] * edges[2][2],
                                 edges[1][0] * edges[2][1] - edges[1][1] * edges[2][0]};
        double const volume = edges[0][0] * cross[0] + edges[0][1] * cross[1] + edges[0][2] * cross[2];

        // A near-planar arrangement carries no handedness to compare against.
        if (std::fabs(volume) < 0.5) continue;

        if (descriptor.degree[center.center_atom] < 4) ++tally.geometry_three_coordinate;
        int const realized_sign = (volume > 0.0) ? 1 : -1;
        reporters.geometry
            .mismatch("chiral_parity", "oracle=geometry molecule={} atom={} signed_volume={}", label,
                      static_cast<unsigned>(center.center_atom), volume)
            .field("target_sign", realized_sign, static_cast<int>(center.target_sign), exactly_equal);
    }
}

/** Which oracles a molecule's extracted centers are put to. */
enum class chiral_oracles_t {
    /** The CIP oracle alone, for a corpus too large to embed. */
    cip_only_k,
    /** Both oracles, which is what the curated set is small enough to afford. */
    cip_and_geometry_k,
};

/**
 *  @brief Extract one molecule's stereocenters and hand them to the requested oracles.
 *
 *  @param[inout] molecule Molecule with explicit hydrogens; stereochemistry is assigned in place.
 *  @param[in] label Printed alongside any mismatch.
 *  @param[in] oracles Which oracles to run; embedding is what makes the geometric one expensive.
 *  @param[inout] reporters Sinks the requested oracles report through.
 *  @param[inout] tally Counters advanced by this molecule.
 */
static void check_molecule_(RDKit::RWMol &molecule, std::string const &label, chiral_oracles_t oracles,
                            oracle_reporters_t &reporters, oracle_tally_t &tally) {
    if (molecule.getNumAtoms() > sft::max_atoms_per_molecule_k || molecule.getNumBonds() > sft::max_bonds_k) return;

    sft::molecule_descriptor_t descriptor;
    if (sft::build_descriptor(molecule, descriptor) != sft::conformer_status_t::success_k) return;

    std::vector<sft::chiral_center_descriptor_t> centers;
    sfp::extract_chiral_centers(molecule, descriptor, centers);
    ++tally.molecules;
    if (centers.empty()) return;

    // The extraction has already run `assignStereochemistry`, so the CIP labels and ranks
    // below describe exactly the molecule the descriptors were derived from.
    check_against_cip_(molecule, descriptor, centers, label, reporters, tally);
    if (oracles == chiral_oracles_t::cip_and_geometry_k)
        check_against_geometry_(molecule, descriptor, centers, label, reporters, tally);
}

/**
 *  @brief Stereo-rich molecules whose centers must agree with both oracles.
 *
 *  @return SMILES strings spanning single centers, sugars, steroids, alkaloids and
 *          drugs, plus the three-coordinate sulfoxide and phosphine cases where a
 *          lone pair takes the fourth substituent slot.
 */
static std::vector<std::string> stereo_test_smiles_() {
    return {
        "C[C@@H](N)C(=O)O",                                                               // L-alanine
        "C[C@H](N)C(=O)O",                                                                // D-alanine
        "C[C@@H](O)[C@@H](N)C(=O)O",                                                      // L-threonine
        "N[C@@H](Cc1ccccc1)C(=O)O",                                                       // L-phenylalanine
        "N[C@@H](CC(=O)O)C(=O)O",                                                         // L-aspartic acid
        "C[C@H](O)CC",                                                                    // (S)-2-butanol
        "C[C@@H](O)CC",                                                                   // (R)-2-butanol
        "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O",                                                 // (R)-ibuprofen
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",                                                // (S)-ibuprofen
        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",                                         // glucose
        "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",                                    // alpha-D-glucopyranose
        "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@H]1O",                                      // alpha-D-galactopyranose
        "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C", // cholesterol
        "C[C@]12CC[C@H]3[C@@H](CC[C@@H]4CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O",               // dihydrotestosterone
        "CN1[C@H]2CC[C@@H]1C[C@@H](C2)OC(=O)C(CO)c1ccccc1",                    // atropine, pseudo-asymmetric C3
        "CC(=O)O[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3CC[C@]2(C)C1", // steroid acetate
        "C[C@@H]1CC[C@H](C(C)C)CC1",                                           // menthane diastereomer
        "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)O)C(=O)O",                      // glutathione
        "C[S@](=O)c1ccccc1",                                                   // three-coordinate sulfoxide
        "C[S@@](=O)CC",                                                        // three-coordinate sulfoxide
        "C[P@](CC)c1ccccc1",                                                   // three-coordinate phosphine
        "COc1ccc2[nH]c([S@](=O)Cc3ncc(C)c(OC)c3C)nc2c1",                       // esomeprazole
    };
}

/**
 *  @brief Walk one molecule and every reachable assignment of its tetrahedral tags.
 *
 *  Flipping the tags multiplies the coverage of a single skeleton by two per center
 *  without changing the topology, and RDKit relabels the CIP codes for each variant,
 *  so the oracle stays honest across the whole stereoisomer family.
 *
 *  @param[in] smiles Molecule to expand.
 *  @param[inout] reporters Sinks both oracles report through.
 *  @param[inout] tally Counters advanced by every variant.
 */
static void check_stereoisomers_(std::string const &smiles, oracle_reporters_t &reporters, oracle_tally_t &tally) {
    auto base = sftest::mol_from_smiles(smiles);
    sf_verify_(base != nullptr, "curated SMILES must parse");

    std::vector<std::uint32_t> tagged_atoms;
    for (auto const *atom : base->atoms()) {
        auto const tag = atom->getChiralTag();
        if (tag == RDKit::Atom::CHI_TETRAHEDRAL_CW || tag == RDKit::Atom::CHI_TETRAHEDRAL_CCW)
            tagged_atoms.push_back(atom->getIdx());
    }
    if (tagged_atoms.empty()) fmt::println("  NOTE chiral_parity smiles={} field=tetrahedral_tags surviving=0", smiles);
    sf_verify_(!tagged_atoms.empty(), "curated SMILES must carry at least one tetrahedral tag");

    std::uint32_t const enumerated_atoms = std::min<std::uint32_t>(tagged_atoms.size(), 4);
    std::uint32_t const variants = 1u << enumerated_atoms;
    for (std::uint32_t assignment = 0; assignment < variants; ++assignment) {
        RDKit::RWMol variant(*base);
        for (std::uint32_t position = 0; position < enumerated_atoms; ++position) {
            bool const clockwise = (assignment >> position) & 1u;
            variant.getAtomWithIdx(tagged_atoms[position])
                ->setChiralTag(clockwise ? RDKit::Atom::CHI_TETRAHEDRAL_CW : RDKit::Atom::CHI_TETRAHEDRAL_CCW);
        }
        check_molecule_(variant, smiles, chiral_oracles_t::cip_and_geometry_k, reporters, tally);
    }
}

/**
 *  @brief Print one census block; the comparison counts belong to the reporters' own sections.
 *
 *  @param[in] heading Line printed above the counters.
 *  @param[in] tally Counters to report.
 */
static void report_(std::string_view heading, oracle_tally_t const &tally) {
    fmt::println("  {}", heading);
    fmt::println("    molecules {}, centers {} ({} three-coordinate)", fmt::group_digits(tally.molecules),
                 fmt::group_digits(tally.centers), fmt::group_digits(tally.three_coordinate));
    fmt::println("    CIP oracle: {} skipped on rank ties, {} unlabeled", fmt::group_digits(tally.rank_ties),
                 fmt::group_digits(tally.unlabeled));
    fmt::println("    geometry oracle: {} three-coordinate centers reached, {} embeddings failed",
                 fmt::group_digits(tally.geometry_three_coordinate), fmt::group_digits(tally.embed_failures));
}

} // namespace smallfield::tests

int main(int argc, char **argv) {
    sftest::oracle_tally_t curated;
    sftest::oracle_reporters_t reporters;
    reporters.limit_printed_lines(max_printed_lines_k);

    fmt::println("{}: curated stereo set with exhaustive tag enumeration", test_name_k);
    for (auto const &smiles : sftest::stereo_test_smiles_()) sftest::check_stereoisomers_(smiles, reporters, curated);
    sftest::report_("curated", curated);
    reporters.cip.summarize("curated_cip");
    reporters.geometry.summarize("curated_geometry");
    sf_verify_(reporters.cip.comparison_count() >= 400, "curated set must reach several hundred CIP comparisons");
    sf_verify_(curated.three_coordinate > 0, "curated set must exercise three-coordinate centers");
    sf_verify_(curated.rank_ties > 0, "curated set must exercise a pseudo-asymmetric center");
    sf_verify_(reporters.geometry.comparison_count() >= 50, "curated set must reach the geometric oracle");
    sf_verify_(curated.geometry_three_coordinate > 0, "geometric oracle must reach a three-coordinate center");
    sf_verify_(reporters.cip.passed(), "every curated stereocenter must match the CIP oracle");
    sf_verify_(reporters.geometry.passed(), "every curated stereocenter must match its embedded geometry");

    auto const selection = sftest::select_corpus(argc, argv, 20000);
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", argv[0]);
        return 2;
    }
    if (selection.knob == sftest::corpus_knob_t::named_k) {
        // Naming a corpus is a request to validate against it, so an empty result is a failure and not an
        // empty corpus: it would otherwise run zero comparisons and satisfy `cip_mismatches == 0`
        // vacuously, reporting a pass for work that never happened. The outcome distinguishes the two
        // ways to get there without a second open.
        std::vector<std::string> corpus;
        auto const outcome = sftest::load_smiles_into(selection.path, selection.max_molecules, corpus);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), selection.path);
        fmt::println("{}: corpus {}, {} SMILES", test_name_k, selection.path, fmt::group_digits(corpus.size()));

        sftest::oracle_tally_t corpus_tally;
        for (auto const &smiles : corpus) {
            // `SmilesToMol` throws on some malformed input rather than returning null, so it belongs
            // inside the guard alongside `addHs`.
            try {
                std::unique_ptr<RDKit::RWMol> molecule(RDKit::SmilesToMol(smiles));
                if (!molecule) continue;
                RDKit::MolOps::addHs(*molecule);
                sftest::check_molecule_(*molecule, smiles, sftest::chiral_oracles_t::cip_only_k, reporters,
                                        corpus_tally);
            }
            catch (...) {
                continue;
            }
        }
        sftest::report_("corpus", corpus_tally);
        // The same reasoning one level down: every molecule may have been skipped, and a section of zero
        // comparisons is indistinguishable from a section of zero mismatches.
        if (reporters.cip.summarize("corpus_cip").compared == 0)
            return sftest::report_fail(test_name_k, "corpus {} produced no CIP comparisons over {} SMILES",
                                       selection.path, corpus.size());
        sf_verify_(reporters.cip.passed(), "every corpus stereocenter must match the CIP oracle");
    }

    return sftest::report_pass(test_name_k);
}
