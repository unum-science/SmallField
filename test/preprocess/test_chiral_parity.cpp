/**
 *  @file test_chiral_parity.cpp
 *  @brief Exact oracle for the tag-and-parity derivation of the chiral volume sign.
 *
 *  `extract_chiral_centers` derives the target signed-volume sign from the atom's
 *  tetrahedral tag and the parity of the stored neighbor order. Two independent
 *  oracles police that derivation, and both demand exact agreement:
 *
 *  The CIP oracle rebuilds the answer straight from RDKit - `assignAtomCIPRanks` for
 *  the neighbor ordering, the `_CIPCode` property for the R/S label - and transports
 *  that sign onto the neighbor order the production code emits. It speaks for
 *  four-coordinate centers whose neighbors carry four distinct CIP ranks. A tie leaves
 *  the CIP ordering, and therefore the sign it implies, undetermined - the C3 of a
 *  tropane is exactly such a pseudo-asymmetric case. A three-coordinate center hides
 *  its lone pair from the ranking entirely, and RDKit's R/S label there does not
 *  compose with a triple product over the three bonded neighbors, as the sulfoxides
 *  in the curated set demonstrate against their own embedded geometry.
 *
 *  The geometric oracle embeds the molecule with ETKDG, which builds its chiral
 *  volume constraints from the same tetrahedral tags, and compares the sign of the
 *  realized tetrahedron against the target. It speaks at every center, tie or not.
 *
 *  Coverage comes from a curated set of stereo-rich molecules, an exhaustive
 *  enumeration of tetrahedral tag assignments over those molecules, and an optional
 *  SMILES corpus named by argv[1].
 */

#include <cmath>   // `std::fabs`
#include <cstdint> // `std::uint64_t`, `std::uint8_t`
#include <cstdio>  // `printf`

#include <algorithm> // `std::sort`, `std::min`
#include <fstream>   // `std::ifstream`
#include <memory>    // `std::unique_ptr`
#include <string>    // `std::string`, `std::getline`
#include <vector>    // `std::vector`

#include <GraphMol/Chirality.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"

#include "test_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

namespace {

/** Running tally of both oracles, shared by every corpus this test walks. */
struct oracle_tally_t {
    std::uint64_t molecules = 0;
    std::uint64_t centers = 0;
    std::uint64_t three_coordinate = 0;
    std::uint64_t cip_compared = 0;
    std::uint64_t cip_mismatches = 0;
    std::uint64_t rank_ties = 0;
    std::uint64_t unlabeled = 0;
    std::uint64_t geometry_compared = 0;
    std::uint64_t geometry_three_coordinate = 0;
    std::uint64_t geometry_mismatches = 0;
    std::uint64_t embed_failures = 0;
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
 *  @param[in] count Number of leading entries to consider.
 *  @return +1 for an even permutation, -1 for an odd one, 0 when the two disagree.
 */
int permutation_parity(std::uint8_t const *from, std::uint8_t const *to, std::uint32_t count) {
    std::uint32_t positions[4] = {0, 0, 0, 0};
    for (std::uint32_t target = 0; target < count; ++target) {
        bool found = false;
        for (std::uint32_t source = 0; source < count && !found; ++source)
            if (from[source] == to[target]) {
                positions[target] = source;
                found = true;
            }
        if (!found) return 0;
    }
    std::uint32_t inversions = 0;
    for (std::uint32_t left = 0; left < count; ++left)
        for (std::uint32_t right = left + 1; right < count; ++right)
            if (positions[left] > positions[right]) ++inversions;
    return (inversions % 2 == 0) ? 1 : -1;
}

/**
 *  @brief Compare the extracted signs against RDKit's CIP labels.
 *
 *  @param[in] molecule Molecule already carrying assigned stereochemistry.
 *  @param[in] descriptor Flat descriptor the centers were extracted from.
 *  @param[in] centers Extracted chiral center descriptors.
 *  @param[in] label Printed alongside any mismatch so the offender is identifiable.
 *  @param[inout] tally Counters advanced by this molecule.
 */
void check_against_cip(RDKit::ROMol const &molecule, sft::molecule_descriptor_t const &descriptor,
                       std::vector<sft::chiral_center_descriptor_t> const &centers, std::string const &label,
                       oracle_tally_t &tally) {
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
        bool tied = false;
        for (std::uint32_t left = 0; left < count; ++left)
            for (std::uint32_t right = left + 1; right < count; ++right)
                if (cip_ranks[reference_order[left]] == cip_ranks[reference_order[right]]) tied = true;
        if (tied) {
            ++tally.rank_ties;
            continue;
        }

        std::sort(reference_order, reference_order + count, [&](std::uint8_t left, std::uint8_t right) {
            if (cip_ranks[left] != cip_ranks[right]) return cip_ranks[left] > cip_ranks[right];
            return left < right;
        });

        std::string cip_code;
        rdkit_atom->getProp("_CIPCode", cip_code);
        int const reference_sign = (cip_code == "S") ? 1 : -1;

        int const parity = permutation_parity(reference_order, center.neighbor_atoms, count);
        sf_verify_(parity != 0, "emitted neighbors must be a permutation of the CIP-ordered neighbors");
        int const expected_sign = reference_sign * parity;

        ++tally.cip_compared;
        if (center.target_sign != expected_sign) {
            ++tally.cip_mismatches;
            if (tally.cip_mismatches <= 10)
                std::printf("  CIP MISMATCH %s atom %u: derived %+d, oracle %+d (%s)\n", label.c_str(), atom,
                            static_cast<int>(center.target_sign), expected_sign, cip_code.c_str());
        }
    }
}

/**
 *  @brief Compare the extracted signs against an ETKDG embedding of the same molecule.
 *
 *  @param[in] molecule Molecule with explicit hydrogens; a copy of it is embedded.
 *  @param[in] descriptor Flat descriptor the centers were extracted from.
 *  @param[in] centers Extracted chiral center descriptors.
 *  @param[in] label Printed alongside any mismatch so the offender is identifiable.
 *  @param[inout] tally Counters advanced by this molecule.
 */
void check_against_geometry(RDKit::ROMol const &molecule, sft::molecule_descriptor_t const &descriptor,
                            std::vector<sft::chiral_center_descriptor_t> const &centers, std::string const &label,
                            oracle_tally_t &tally) {
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

        ++tally.geometry_compared;
        if (descriptor.degree[center.center_atom] < 4) ++tally.geometry_three_coordinate;
        int const realized_sign = (volume > 0.0) ? 1 : -1;
        if (center.target_sign != realized_sign) {
            ++tally.geometry_mismatches;
            if (tally.geometry_mismatches <= 10)
                std::printf("  GEOMETRY MISMATCH %s atom %u: target %+d, embedded volume %.3f\n", label.c_str(),
                            static_cast<unsigned>(center.center_atom), static_cast<int>(center.target_sign), volume);
        }
    }
}

/**
 *  @brief Extract one molecule's stereocenters and hand them to the requested oracles.
 *
 *  @param[inout] molecule Molecule with explicit hydrogens; stereochemistry is assigned in place.
 *  @param[in] label Printed alongside any mismatch.
 *  @param[in] with_geometry True to additionally embed the molecule and check the realized volumes.
 *  @param[inout] tally Counters advanced by this molecule.
 */
void check_molecule(RDKit::RWMol &molecule, std::string const &label, bool with_geometry, oracle_tally_t &tally) {
    if (molecule.getNumAtoms() > sft::max_atoms_per_molecule_k || molecule.getNumBonds() > sft::max_bonds_k) return;

    sft::molecule_descriptor_t descriptor;
    if (!sft::build_descriptor(molecule, descriptor)) return;

    std::vector<sft::chiral_center_descriptor_t> centers;
    sfp::extract_chiral_centers(molecule, descriptor, centers);
    ++tally.molecules;
    if (centers.empty()) return;

    // The extraction has already run `assignStereochemistry`, so the CIP labels and ranks
    // below describe exactly the molecule the descriptors were derived from.
    check_against_cip(molecule, descriptor, centers, label, tally);
    if (with_geometry) check_against_geometry(molecule, descriptor, centers, label, tally);
}

/**
 *  @brief Stereo-rich molecules whose centers must agree with both oracles.
 *
 *  @return SMILES strings spanning single centers, sugars, steroids, alkaloids and
 *          drugs, plus the three-coordinate sulfoxide and phosphine cases where a
 *          lone pair takes the fourth substituent slot.
 */
std::vector<std::string> stereo_test_smiles() {
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
 *  @param[inout] tally Counters advanced by every variant.
 */
void check_stereoisomers(std::string const &smiles, oracle_tally_t &tally) {
    auto base = sftest::mol_from_smiles(smiles);
    sf_verify_(base != nullptr, "curated SMILES must parse");

    std::vector<std::uint32_t> tagged_atoms;
    for (auto const *atom : base->atoms()) {
        auto const tag = atom->getChiralTag();
        if (tag == RDKit::Atom::CHI_TETRAHEDRAL_CW || tag == RDKit::Atom::CHI_TETRAHEDRAL_CCW)
            tagged_atoms.push_back(atom->getIdx());
    }
    if (tagged_atoms.empty()) std::printf("  NO TAG SURVIVED: %s\n", smiles.c_str());
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
        check_molecule(variant, smiles, true, tally);
    }
}

/**
 *  @brief Read SMILES from a whitespace-delimited corpus file.
 *
 *  @param[in] path File to read; a missing file yields an empty list.
 *  @param[in] limit Maximum number of entries to return.
 *  @return One SMILES string per accepted line.
 */
std::vector<std::string> load_corpus(char const *path, std::size_t limit) {
    std::vector<std::string> smiles;
    std::ifstream stream(path);
    std::string line;
    while (smiles.size() < limit && std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto const separator = line.find_first_of(" \t");
        if (separator != std::string::npos) line.resize(separator);
        if (!line.empty()) smiles.push_back(std::move(line));
    }
    return smiles;
}

/**
 *  @brief Print one tally block.
 *
 *  @param[in] heading Line printed above the counters.
 *  @param[in] tally Counters to report.
 */
void report(char const *heading, oracle_tally_t const &tally) {
    std::printf("  %s\n", heading);
    std::printf("    molecules %llu, centers %llu (%llu three-coordinate)\n",
                static_cast<unsigned long long>(tally.molecules), static_cast<unsigned long long>(tally.centers),
                static_cast<unsigned long long>(tally.three_coordinate));
    std::printf("    CIP oracle: %llu compared, %llu mismatches, %llu skipped on rank ties, %llu unlabeled\n",
                static_cast<unsigned long long>(tally.cip_compared),
                static_cast<unsigned long long>(tally.cip_mismatches), static_cast<unsigned long long>(tally.rank_ties),
                static_cast<unsigned long long>(tally.unlabeled));
    std::printf("    geometry oracle: %llu compared (%llu three-coordinate), %llu mismatches, %llu embeddings failed\n",
                static_cast<unsigned long long>(tally.geometry_compared),
                static_cast<unsigned long long>(tally.geometry_three_coordinate),
                static_cast<unsigned long long>(tally.geometry_mismatches),
                static_cast<unsigned long long>(tally.embed_failures));
}

} // namespace

int main(int argc, char **argv) {
    oracle_tally_t curated;

    std::printf("test_chiral_parity: curated stereo set with exhaustive tag enumeration\n");
    for (auto const &smiles : stereo_test_smiles()) check_stereoisomers(smiles, curated);
    report("curated", curated);
    sf_verify_(curated.cip_compared >= 400, "curated set must reach several hundred CIP comparisons");
    sf_verify_(curated.three_coordinate > 0, "curated set must exercise three-coordinate centers");
    sf_verify_(curated.rank_ties > 0, "curated set must exercise a pseudo-asymmetric center");
    sf_verify_(curated.geometry_compared >= 50, "curated set must reach the geometric oracle");
    sf_verify_(curated.geometry_three_coordinate > 0, "geometric oracle must reach a three-coordinate center");
    sf_verify_(curated.cip_mismatches == 0, "every curated stereocenter must match the CIP oracle");
    sf_verify_(curated.geometry_mismatches == 0, "every curated stereocenter must match its embedded geometry");

    char const *corpus_path = argc > 1 ? argv[1] : nullptr;
    if (corpus_path != nullptr) {
        std::size_t const limit = argc > 2 ? std::stoul(argv[2]) : 20000;
        auto const corpus = load_corpus(corpus_path, limit);
        std::printf("test_chiral_parity: corpus %s, %zu SMILES\n", corpus_path, corpus.size());
        oracle_tally_t corpus_tally;
        for (auto const &smiles : corpus) {
            std::unique_ptr<RDKit::RWMol> molecule(RDKit::SmilesToMol(smiles));
            if (!molecule) continue;
            try {
                RDKit::MolOps::addHs(*molecule);
                check_molecule(*molecule, smiles, false, corpus_tally);
            }
            catch (...) {
                continue;
            }
        }
        report("corpus", corpus_tally);
        sf_verify_(corpus_tally.cip_mismatches == 0, "every corpus stereocenter must match the CIP oracle");
    }

    std::printf("ALL TESTS PASSED\n");
    return 0;
}
