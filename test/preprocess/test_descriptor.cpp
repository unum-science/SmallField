/**
 *  @file test/preprocess/test_descriptor.cpp
 *  @brief Validate `molecule_descriptor_t` against the RDKit accessors it is derived from.
 *
 *  Every field the descriptor carries for an atom, a bond, a neighbour list and a ring is compared against RDKit's
 *  own answer, and the derived bond-hop matrix against `MolOps::getDistanceMat`, RDKit's Floyd-Warshall. The builtin
 *  drug-like set runs unconditionally; a `.smi` file named on the command line or in `SMALLFIELD_TEST_CORPUS` adds a
 *  second section over that corpus.
 *
 *  A third pair of sections covers `fill_mmff_perception`, the one writer putting two aromatic models in one
 *  descriptor: that it moves nothing outside `mmff_bond_type` and `atom_flag_mmff_aromatic_k`, which keeps ETKDG
 *  reading the perception it is defined on, and that those two land where RDKit's own MMFF force field reads them.
 *
 *  `build_descriptor` answers a `conformer_status_t`, forwarded rather than re-derived: a corpus line it refuses is
 *  counted under that status, and a builtin it refuses is a defect, the builtin set being curated to fit. The return
 *  is never discarded, because a refusal writes nothing and the descriptor still holds the previous molecule.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`
#include <cstring> // `std::memcmp`, `std::memset`

#include <algorithm>   // `std::sort`
#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <GraphMol/GraphMol.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>

#include "smallfield/types/conformer_status.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

#include "test/test_rdkit_reference_helpers.hpp"

namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_descriptor";

/** One pathological molecule must not bury the log; the reporter keeps counting past this. */
static constexpr std::size_t max_printed_lines_k = 40;

/** Corpus lines walked when neither `argv[2]` nor `SMALLFIELD_TEST_MAX_MOLS` caps the sweep. */
static constexpr std::size_t default_max_molecules_k = 500;

/** Above this, RDKit's distance matrix is carrying its `LOCAL_INF` rather than a hop count. */
static constexpr double rdkit_unreachable_distance_k = 1e7;

namespace smallfield::tests {

#pragma region Comparison

/** Buffers that outlive the molecule loop, so a corpus sweep grows each of them once. */
struct descriptor_scratch_t {
    sft::molecule_descriptor_t descriptor {};
    std::vector<std::uint32_t> rdkit_neighbors;
    std::vector<std::uint32_t> descriptor_neighbors;
    std::vector<std::uint8_t> bond_hops;
};

/** Where every corpus line went, so a section that compared little says why. `perceived` counts compared lines whose
 *  MMFF perception also reached RDKit's own answer, the rest being ones RDKit will not type; `refused` is indexed by
 *  the status that kept a line out. */
struct descriptor_tally_t {
    std::size_t compared = 0;
    std::size_t perceived = 0;
    std::size_t refused[sft::conformer_status_count_k] = {};
};

static void compare_atoms_(RDKit::ROMol const &molecule, sft::molecule_descriptor_t const &descriptor,
                           std::string const &smiles, diagnostic_reporter_t &reporter) {
    auto const *ring_info = molecule.getRingInfo();
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        auto const *atom = molecule.getAtomWithIdx(atom_index);
        reporter.mismatch("atom", "smiles={} atom={}", smiles, atom_index)
            .field("element", atom->getAtomicNum(), descriptor.element[atom_index], exactly_equal)
            .field("hybridization", static_cast<std::uint8_t>(atom->getHybridization()),
                   descriptor.hybridization[atom_index], exactly_equal)
            .field("degree", atom->getDegree(), descriptor.degree[atom_index], exactly_equal)
            .field("num_hydrogens", atom->getTotalNumHs(true), descriptor.num_hydrogens[atom_index], exactly_equal)
            .field("formal_charge", atom->getFormalCharge(), descriptor.formal_charge[atom_index], exactly_equal)
            .field("aromatic", atom->getIsAromatic(),
                   (descriptor.atom_flags[atom_index] & sft::atom_flag_aromatic_k) != 0, exactly_equal)
            .field("in_ring", ring_info->numAtomRings(atom_index) > 0,
                   (descriptor.atom_flags[atom_index] & sft::atom_flag_in_ring_k) != 0, exactly_equal);
    }
}

static void compare_bonds_(RDKit::ROMol const &molecule, sft::molecule_descriptor_t const &descriptor,
                           std::string const &smiles, diagnostic_reporter_t &reporter) {
    auto const *ring_info = molecule.getRingInfo();
    std::uint32_t const num_bonds = molecule.getNumBonds();
    for (std::uint32_t bond_index = 0; bond_index < num_bonds; ++bond_index) {
        auto const *bond = molecule.getBondWithIdx(bond_index);
        reporter.mismatch("bond", "smiles={} bond={}", smiles, bond_index)
            .field("atom_i", bond->getBeginAtomIdx(), descriptor.bond_atom_i[bond_index], exactly_equal)
            .field("atom_j", bond->getEndAtomIdx(), descriptor.bond_atom_j[bond_index], exactly_equal)
            .field("aromatic", bond->getIsAromatic(),
                   (descriptor.bond_flags[bond_index] & sft::bond_flag_aromatic_k) != 0, exactly_equal)
            .field("in_ring", ring_info->numBondRings(bond_index) > 0,
                   (descriptor.bond_flags[bond_index] & sft::bond_flag_in_ring_k) != 0, exactly_equal);
    }
}

static void compare_neighbors_(RDKit::ROMol const &molecule, descriptor_scratch_t &scratch, std::string const &smiles,
                               diagnostic_reporter_t &reporter) {
    sft::molecule_descriptor_t const &descriptor = scratch.descriptor;
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        auto const *atom = molecule.getAtomWithIdx(atom_index);
        scratch.rdkit_neighbors.clear();
        for (auto const *neighbor : molecule.atomNeighbors(atom)) scratch.rdkit_neighbors.push_back(neighbor->getIdx());
        std::sort(scratch.rdkit_neighbors.begin(), scratch.rdkit_neighbors.end());

        scratch.descriptor_neighbors.clear();
        std::uint16_t const start = descriptor.neighbor_offset[atom_index];
        std::uint16_t const end = descriptor.neighbor_offset[atom_index + 1];
        for (std::uint16_t entry = start; entry < end; ++entry)
            scratch.descriptor_neighbors.push_back(descriptor.neighbor_atom[entry]);
        std::sort(scratch.descriptor_neighbors.begin(), scratch.descriptor_neighbors.end());

        reporter.mismatch("neighbors", "smiles={} atom={}", smiles, atom_index)
            .field("count", scratch.rdkit_neighbors.size(), scratch.descriptor_neighbors.size(), exactly_equal);
        if (scratch.rdkit_neighbors.size() != scratch.descriptor_neighbors.size())
            continue; // lists of different lengths cannot be paired slot by slot
        for (std::size_t slot = 0; slot < scratch.rdkit_neighbors.size(); ++slot)
            reporter.mismatch("neighbors", "smiles={} atom={} slot={}", smiles, atom_index, slot)
                .field("neighbor_atom", scratch.rdkit_neighbors[slot], scratch.descriptor_neighbors[slot],
                       exactly_equal);
    }
}

static void compare_bond_hops_(RDKit::ROMol const &molecule, descriptor_scratch_t &scratch, std::string const &smiles,
                               diagnostic_reporter_t &reporter) {
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    std::size_t const cell_count = static_cast<std::size_t>(num_atoms) * num_atoms;
    if (scratch.bond_hops.size() < cell_count) scratch.bond_hops.resize(cell_count);
    sft::fill_bond_distances(scratch.descriptor, sft::hop_matrix_t {scratch.bond_hops.data(), num_atoms, num_atoms});

    double const *rdkit_distances = RDKit::MolOps::getDistanceMat(molecule);
    for (std::uint32_t row = 0; row < num_atoms; ++row) {
        for (std::uint32_t column = 0; column < num_atoms; ++column) {
            double const reference = rdkit_distances[row * num_atoms + column];
            std::uint8_t const hops = scratch.bond_hops[row * num_atoms + column];
            bool const reference_reachable = reference < rdkit_unreachable_distance_k;
            auto group = reporter.mismatch("bond_hops", "smiles={} row={} column={}", smiles, row, column);
            group.field("reachable", reference_reachable, hops != sft::unreachable_hops_k, exactly_equal);
            if (!reference_reachable) continue; // the two encodings of "no bonded path" are not comparable
            group.field("hops", reference, hops, exactly_equal);
        }
    }
}

/** Build the descriptor for one molecule, then compare every field it carries. A refusal writes nothing, so it
 *  returns before the comparisons rather than reading the previous molecule's descriptor against this one's.
 *  @return What `build_descriptor` answered, so the caller decides whether a refusal is a defect. */
static sft::conformer_status_t check_molecule_(RDKit::ROMol const &molecule, std::string const &smiles,
                                               diagnostic_reporter_t &reporter, descriptor_scratch_t &scratch) {
    sft::conformer_status_t const built = sft::build_descriptor(molecule, scratch.descriptor);
    if (built != sft::conformer_status_t::success_k) return built;

    auto const *ring_info = molecule.getRingInfo();
    reporter.mismatch("counts", "smiles={}", smiles)
        .field("num_atoms", molecule.getNumAtoms(), scratch.descriptor.num_atoms, exactly_equal)
        .field("num_bonds", molecule.getNumBonds(), scratch.descriptor.num_bonds, exactly_equal)
        .field("num_rings", ring_info->numRings(), scratch.descriptor.num_rings, exactly_equal);

    compare_atoms_(molecule, scratch.descriptor, smiles, reporter);
    compare_bonds_(molecule, scratch.descriptor, smiles, reporter);
    compare_neighbors_(molecule, scratch, smiles, reporter);
    compare_bond_hops_(molecule, scratch, smiles, reporter);
    return built;
}

#pragma endregion Comparison

#pragma region MMFF Perception

/** The two descriptors one molecule produces, and the third RDKit's own MMFF pass leaves behind. */
struct perception_scratch_t {
    sft::molecule_descriptor_t before {};
    sft::molecule_descriptor_t after {};
    sft::molecule_descriptor_t rdkit {};
};

/** The fields `fill_mmff_perception` owns, blanked, so what remains is what ETKDG reads. */
static void blank_mmff_fields_(sft::molecule_descriptor_t &descriptor) {
    std::memset(descriptor.mmff_bond_type, 0, sizeof(descriptor.mmff_bond_type));
    for (std::uint32_t atom_index = 0; atom_index < sft::max_atoms_per_molecule_k; ++atom_index)
        descriptor.atom_flags[atom_index] &= static_cast<std::uint8_t>(~sft::atom_flag_mmff_aromatic_k);
}

/**
 *  @brief Perceive MMFF's aromaticity onto a copy of one molecule's descriptor and check what moved.
 *
 *  Two claims. First, what keeps the geometry out of it: outside the two MMFF fields every byte is the byte
 *  `build_descriptor` wrote, so the bounds matrix, the CSD torsion match and the chirality pass read the perception
 *  ETKDG is defined on. Second, those two fields carry what RDKit's own force field reads, the molecule
 *  `MMFFMolProperties` leaves behind — taken from a second parse, because that constructor perceives once and
 *  `MolOps::setMMFFAromaticity` is not idempotent on every molecule.
 *
 *  @param[inout] molecule Already compared against RDKit; its aromaticity is rewritten here, and @p smiles is
 *      reparsed for the oracle. @p etkdg_reporter takes the fields that must not move, @p mmff_reporter the two.
 *  @return Whether the molecule reached the comparison against RDKit at all.
 */
static bool check_mmff_perception_(RDKit::RWMol &molecule, std::string const &smiles,
                                   diagnostic_reporter_t &etkdg_reporter, diagnostic_reporter_t &mmff_reporter,
                                   perception_scratch_t &scratch) {
    if (sft::build_descriptor(molecule, scratch.before) != sft::conformer_status_t::success_k) return false;
    scratch.after = scratch.before;
    sft::fill_mmff_perception(molecule, scratch.after);

    sft::molecule_descriptor_t masked_before = scratch.before, masked_after = scratch.after;
    blank_mmff_fields_(masked_before);
    blank_mmff_fields_(masked_after);
    etkdg_reporter.mismatch("mmff_perception", "smiles={}", smiles)
        .field("etkdg_field_bytes", std::memcmp(&masked_before, &masked_after, sizeof(masked_before)), 0,
               exactly_equal);

    // A second parse rather than a copy, and guarded, because a corpus line reaches this too.
    std::unique_ptr<RDKit::RWMol> reference;
    try {
        reference.reset(RDKit::SmilesToMol(smiles));
        if (!reference) return false;
        RDKit::MolOps::addHs(*reference);
        RDKit::MMFF::MMFFMolProperties properties(*reference);
        if (!properties.isValid()) return false; // RDKit will not type it, so it perceived for nothing
    }
    catch (...) {
        return false;
    }
    if (sft::build_descriptor(*reference, scratch.rdkit) != sft::conformer_status_t::success_k) return false;

    std::uint32_t const num_atoms = scratch.after.num_atoms;
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index)
        mmff_reporter.mismatch("mmff_perception", "smiles={} atom={}", smiles, atom_index)
            .field("mmff_aromatic", (scratch.rdkit.atom_flags[atom_index] & sft::atom_flag_mmff_aromatic_k) != 0,
                   (scratch.after.atom_flags[atom_index] & sft::atom_flag_mmff_aromatic_k) != 0, exactly_equal);
    std::uint32_t const num_bonds = scratch.after.num_bonds;
    for (std::uint32_t bond_index = 0; bond_index < num_bonds; ++bond_index)
        mmff_reporter.mismatch("mmff_perception", "smiles={} bond={}", smiles, bond_index)
            .field("mmff_bond_type", scratch.rdkit.mmff_bond_type[bond_index], scratch.after.mmff_bond_type[bond_index],
                   exactly_equal);
    return true;
}

#pragma endregion MMFF Perception

#pragma region Corpora

/** Molecules whose graphs stress a feature the drug-like standards do not reach. */
static std::vector<std::string> stress_test_smiles_() {
    return {
        "N#Cc1ccc(cc1)/C=C/c2ccc(C#N)cc2", // stereo + nitriles
        "[H+].[H+].[Cl-].[Cl-]",           // ions, so most atom pairs have no bonded path
        "OC(=O)CCCCCCCCCCCCCCC(=O)O",      // long chain (hexadecanedioic acid)
        "C1CC2CC3CC1CC(C2)C3",             // bridged ring (adamantane)
        "c1ccc2c(c1)[nH]c3ccccc23",        // fused aromatic (carbazole)
    };
}

/** The builtin molecules are curated, so anything that stops one short of the comparison is a defect. */
static void check_builtin_(std::string const &smiles, diagnostic_reporter_t &reporter, descriptor_scratch_t &scratch,
                           diagnostic_reporter_t &perception_reporter, perception_scratch_t &perception) {
    std::unique_ptr<RDKit::RWMol> molecule = mol_from_smiles(smiles);
    sf_verify_(molecule != nullptr, ("builtin SMILES must parse: " + smiles).c_str());
    sft::conformer_status_t const built = check_molecule_(*molecule, smiles, reporter, scratch);
    sf_verify_(
        built == sft::conformer_status_t::success_k,
        fmt::format("build_descriptor refused the builtin SMILES {} as {}", smiles, sft::conformer_status_name(built)));
    // Last, because it rewrites the molecule the comparisons above read.
    check_mmff_perception_(*molecule, smiles, perception_reporter, reporter, perception);
}

/** A corpus line is untrusted input, so each way it can miss the comparison is counted and named. */
static void check_corpus_line_(std::string const &smiles, diagnostic_reporter_t &reporter,
                               descriptor_scratch_t &scratch, descriptor_tally_t &tally,
                               diagnostic_reporter_t &perception_reporter, perception_scratch_t &perception) {
    // `SmilesToMol` throws on some malformed input instead of returning null, so it belongs inside the
    // guard beside `addHs`.
    std::unique_ptr<RDKit::RWMol> molecule;
    try {
        molecule.reset(RDKit::SmilesToMol(smiles));
        if (molecule) RDKit::MolOps::addHs(*molecule);
    }
    catch (...) {
        molecule.reset();
    }
    sft::conformer_status_t const built = molecule ? check_molecule_(*molecule, smiles, reporter, scratch)
                                                   : sft::conformer_status_t::invalid_smiles_k;
    if (built != sft::conformer_status_t::success_k) {
        ++tally.refused[static_cast<std::size_t>(built)];
        return;
    }
    ++tally.compared;
    // Last, because it rewrites the molecule the comparisons above read.
    tally.perceived += check_mmff_perception_(*molecule, smiles, perception_reporter, reporter, perception);
}

/** Narrate where the corpus lines went, naming each refusal with the status that refused it. */
static void announce_corpus_tally_(descriptor_tally_t const &tally) {
    fmt::print("NOTE descriptor_corpus molecules={} mmff_perceived={}", tally.compared, tally.perceived);
    for (std::size_t status = 0; status < sft::conformer_status_count_k; ++status) {
        if (tally.refused[status] == 0) continue; // a refusal nothing in this corpus reached
        fmt::print(" {}={}", sft::conformer_status_name(static_cast<sft::conformer_status_t>(status)),
                   tally.refused[status]);
    }
    fmt::println("");
}

#pragma endregion Corpora

} // namespace smallfield::tests

int main(int argc, char **argv) {
    fmt::println("{}: molecule_descriptor_t vs the RDKit accessors", test_name_k);

    auto const selection = sftest::select_corpus(argc, argv, default_max_molecules_k);
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", argv[0]);
        return 2;
    }

    sftest::diagnostic_reporter_t reporter {"rdkit", "ours"};
    reporter.limit_printed_lines(max_printed_lines_k);
    // Its own reporter: the operands are one descriptor before and after a pass of ours, so calling either
    // "rdkit" would misname what is asserted.
    sftest::diagnostic_reporter_t perception_reporter {"before", "after"};
    perception_reporter.limit_printed_lines(max_printed_lines_k);
    sftest::descriptor_scratch_t scratch;
    sftest::perception_scratch_t perception;

    for (auto const &smiles : sftest::standard_test_smiles())
        sftest::check_builtin_(smiles, reporter, scratch, perception_reporter, perception);
    for (auto const &smiles : sftest::stress_test_smiles_())
        sftest::check_builtin_(smiles, reporter, scratch, perception_reporter, perception);
    reporter.summarize("descriptor_builtin");
    perception_reporter.summarize("mmff_perception_builtin");

    if (selection.knob == sftest::corpus_knob_t::named_k) {
        // Naming a corpus is a request to validate against it, so a path that does not open fails rather
        // than closing every section with nothing compared.
        std::vector<std::string> corpus;
        auto const outcome = sftest::load_smiles_into(selection.path, selection.max_molecules, corpus);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), selection.path);
        fmt::println("{}: corpus {}, {} SMILES", test_name_k, selection.path, fmt::group_digits(corpus.size()));

        sftest::descriptor_tally_t tally;
        for (auto const &smiles : corpus)
            sftest::check_corpus_line_(smiles, reporter, scratch, tally, perception_reporter, perception);
        sftest::announce_corpus_tally_(tally);
        reporter.summarize("descriptor_corpus");
        perception_reporter.summarize("mmff_perception_corpus");
    }

    if (!reporter.passed())
        return sftest::report_fail(test_name_k, "{} mismatches over {} comparisons", reporter.mismatch_count(),
                                   reporter.comparison_count());
    if (!perception_reporter.passed())
        return sftest::report_fail(test_name_k, "{} of {} MMFF perception comparisons moved a field ETKDG reads",
                                   perception_reporter.mismatch_count(), perception_reporter.comparison_count());
    return sftest::report_pass(test_name_k);
}
