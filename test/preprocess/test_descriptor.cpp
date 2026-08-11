/**
 *  @file test/preprocess/test_descriptor.cpp
 *  @brief Validate `molecule_descriptor_t` against the RDKit accessors it is derived from.
 *
 *  Every field the descriptor carries for an atom, a bond, a neighbour list and a ring is compared against
 *  RDKit's own answer, and the derived bond-hop matrix against `MolOps::getDistanceMat`, RDKit's Floyd-Warshall.
 *  The builtin drug-like set runs unconditionally; naming a `.smi` file on the command line, or in
 *  `SMALLFIELD_TEST_CORPUS`, adds a second section over that corpus.
 *
 *  `build_descriptor` answers one `bool` for two different refusals, so a molecule's extents are classified
 *  before the call: an oversized corpus line is counted and skipped, and a refusal with both extents in range is
 *  a defect. The return is never discarded, because a discarded one leaves the previous molecule's descriptor in
 *  place and compares it against this molecule's RDKit answers.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <algorithm>   // `std::min`, `std::sort`
#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::common_type_t`, `std::decay_t`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <GraphMol/GraphMol.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

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

/** Exact equality, promoted so a signed operand and an unsigned one still compare as numbers. */
static constexpr auto exactly_equal_ = [](auto const &reference, auto const &observed) {
    using promoted_type_ = std::common_type_t<std::decay_t<decltype(reference)>, std::decay_t<decltype(observed)>>;
    return static_cast<promoted_type_>(reference) == static_cast<promoted_type_>(observed);
};

/** Buffers that outlive the molecule loop, so a corpus sweep grows each of them once. */
struct descriptor_scratch_t {
    sft::molecule_descriptor_t descriptor {};
    std::vector<std::uint32_t> rdkit_neighbors;
    std::vector<std::uint32_t> descriptor_neighbors;
    std::vector<std::uint8_t> bond_hops;
};

/** Where every corpus line went, so a section that compared little says why. */
struct descriptor_tally_t {
    std::uint64_t compared = 0;
    std::uint64_t unparsed = 0;
    std::uint64_t too_many_atoms = 0;
    std::uint64_t too_many_bonds = 0;
};

/** Which of the two extents `build_descriptor` folds into its single `false` a molecule would trip. */
enum class descriptor_capacity_t {
    fits_k,
    too_many_atoms_k,
    too_many_bonds_k,
};

/** Recover the refusal reason the bare `bool` return discards, before making the call. */
static descriptor_capacity_t classify_capacity_(RDKit::ROMol const &molecule) {
    if (molecule.getNumAtoms() > sft::max_atoms_per_molecule_k) return descriptor_capacity_t::too_many_atoms_k;
    if (molecule.getNumBonds() > sft::max_bonds_k) return descriptor_capacity_t::too_many_bonds_k;
    return descriptor_capacity_t::fits_k;
}

/**
 *  @brief Whether every ring label the descriptor carries per atom is complete for this molecule.
 *
 *  `build_descriptor` keeps the first `max_rings_k` SSSR rings and stops writing ring atoms at the capacity of
 *  `ring_atoms`, so beyond either bound a genuinely in-ring atom carries no `atom_flag_in_ring_k` and the
 *  comparison would report a clamp as a defect. Bond ring flags come from `numBondRings` directly and are never
 *  clamped, so they stay compared either way.
 */
static bool ring_labels_are_complete_(RDKit::RingInfo const &ring_info) {
    if (ring_info.numRings() > sft::max_rings_k) return false;
    std::size_t ring_atom_count = 0;
    for (auto const &ring : ring_info.atomRings()) ring_atom_count += ring.size();
    return ring_atom_count <= sft::max_atoms_per_molecule_k * 2u;
}

static void compare_atoms_(RDKit::ROMol const &molecule, sft::molecule_descriptor_t const &descriptor,
                           std::string const &smiles, diagnostic_reporter_t &reporter) {
    auto const *ring_info = molecule.getRingInfo();
    bool const ring_labels_complete = ring_labels_are_complete_(*ring_info);
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        auto const *atom = molecule.getAtomWithIdx(atom_index);
        auto group = reporter.mismatch("atom", "smiles={} atom={}", smiles, atom_index);
        group.field("element", atom->getAtomicNum(), descriptor.element[atom_index], exactly_equal_)
            .field("hybridization", static_cast<std::uint8_t>(atom->getHybridization()),
                   descriptor.hybridization[atom_index], exactly_equal_)
            .field("degree", atom->getDegree(), descriptor.degree[atom_index], exactly_equal_)
            .field("num_hydrogens", atom->getTotalNumHs(true), descriptor.num_hydrogens[atom_index], exactly_equal_)
            .field("formal_charge", atom->getFormalCharge(), descriptor.formal_charge[atom_index], exactly_equal_)
            .field("aromatic", atom->getIsAromatic(),
                   sft::atom_has_flag(descriptor, atom_index, sft::atom_flag_aromatic_k), exactly_equal_);
        if (!ring_labels_complete) continue; // the ring cap dropped labels this molecule legitimately lacks
        group.field("in_ring", ring_info->numAtomRings(atom_index) > 0,
                    sft::atom_has_flag(descriptor, atom_index, sft::atom_flag_in_ring_k), exactly_equal_);
    }
}

static void compare_bonds_(RDKit::ROMol const &molecule, sft::molecule_descriptor_t const &descriptor,
                           std::string const &smiles, diagnostic_reporter_t &reporter) {
    auto const *ring_info = molecule.getRingInfo();
    std::uint32_t const num_bonds = molecule.getNumBonds();
    for (std::uint32_t bond_index = 0; bond_index < num_bonds; ++bond_index) {
        auto const *bond = molecule.getBondWithIdx(bond_index);
        reporter.mismatch("bond", "smiles={} bond={}", smiles, bond_index)
            .field("atom_i", bond->getBeginAtomIdx(), descriptor.bond_atom_i[bond_index], exactly_equal_)
            .field("atom_j", bond->getEndAtomIdx(), descriptor.bond_atom_j[bond_index], exactly_equal_)
            .field("aromatic", bond->getIsAromatic(),
                   sft::bond_has_flag(descriptor, bond_index, sft::bond_flag_aromatic_k), exactly_equal_)
            .field("in_ring", ring_info->numBondRings(bond_index) > 0,
                   sft::bond_has_flag(descriptor, bond_index, sft::bond_flag_in_ring_k), exactly_equal_);
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
            .field("count", scratch.rdkit_neighbors.size(), scratch.descriptor_neighbors.size(), exactly_equal_);
        if (scratch.rdkit_neighbors.size() != scratch.descriptor_neighbors.size())
            continue; // lists of different lengths cannot be paired slot by slot
        for (std::size_t slot = 0; slot < scratch.rdkit_neighbors.size(); ++slot)
            reporter.mismatch("neighbors", "smiles={} atom={} slot={}", smiles, atom_index, slot)
                .field("neighbor_atom", scratch.rdkit_neighbors[slot], scratch.descriptor_neighbors[slot],
                       exactly_equal_);
    }
}

static void compare_bond_hops_(RDKit::ROMol const &molecule, descriptor_scratch_t &scratch, std::string const &smiles,
                               diagnostic_reporter_t &reporter) {
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    std::size_t const cell_count = static_cast<std::size_t>(num_atoms) * num_atoms;
    if (scratch.bond_hops.size() < cell_count) scratch.bond_hops.resize(cell_count);
    sft::fill_bond_distances(scratch.descriptor, scratch.bond_hops.data(), num_atoms);

    double const *rdkit_distances = RDKit::MolOps::getDistanceMat(molecule);
    for (std::uint32_t row = 0; row < num_atoms; ++row) {
        for (std::uint32_t column = 0; column < num_atoms; ++column) {
            double const reference = rdkit_distances[row * num_atoms + column];
            std::uint8_t const hops = scratch.bond_hops[row * num_atoms + column];
            bool const reference_reachable = reference < rdkit_unreachable_distance_k;
            auto group = reporter.mismatch("bond_hops", "smiles={} row={} column={}", smiles, row, column);
            group.field("reachable", reference_reachable, hops != sft::unreachable_hops_k, exactly_equal_);
            if (!reference_reachable) continue; // the two encodings of "no bonded path" are not comparable
            group.field("hops", reference, hops, exactly_equal_);
        }
    }
}

/**
 *  @brief Build the descriptor for one molecule already known to fit, then compare every field it carries.
 *
 *  @param[in] molecule Sanitized molecule with hydrogens added.
 *  @param[in] smiles The molecule's SMILES, carried into every diagnostic key.
 *  @param[inout] reporter Reporter counting and printing the comparisons.
 *  @param[inout] scratch Buffers reused across molecules.
 */
static void check_molecule_(RDKit::ROMol const &molecule, std::string const &smiles, diagnostic_reporter_t &reporter,
                            descriptor_scratch_t &scratch) {
    bool const built = sft::build_descriptor(molecule, scratch.descriptor);
    sf_verify_(built, ("build_descriptor refused a molecule inside both extents: " + smiles).c_str());

    auto const *ring_info = molecule.getRingInfo();
    reporter.mismatch("counts", "smiles={}", smiles)
        .field("num_atoms", molecule.getNumAtoms(), scratch.descriptor.num_atoms, exactly_equal_)
        .field("num_bonds", molecule.getNumBonds(), scratch.descriptor.num_bonds, exactly_equal_)
        .field("num_rings", std::min<std::uint32_t>(ring_info->numRings(), sft::max_rings_k),
               scratch.descriptor.num_rings, exactly_equal_);

    compare_atoms_(molecule, scratch.descriptor, smiles, reporter);
    compare_bonds_(molecule, scratch.descriptor, smiles, reporter);
    compare_neighbors_(molecule, scratch, smiles, reporter);
    compare_bond_hops_(molecule, scratch, smiles, reporter);
}

#pragma endregion Comparison

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
static void check_builtin_(std::string const &smiles, diagnostic_reporter_t &reporter, descriptor_scratch_t &scratch) {
    std::unique_ptr<RDKit::RWMol> molecule = mol_from_smiles(smiles);
    sf_verify_(molecule != nullptr, ("builtin SMILES must parse: " + smiles).c_str());
    sf_verify_(classify_capacity_(*molecule) == descriptor_capacity_t::fits_k,
               ("builtin SMILES must fit the descriptor extents: " + smiles).c_str());
    check_molecule_(*molecule, smiles, reporter, scratch);
}

/** A corpus line is untrusted input, so each way it can miss the comparison is counted and named. */
static void check_corpus_line_(std::string const &smiles, diagnostic_reporter_t &reporter,
                               descriptor_scratch_t &scratch, descriptor_tally_t &tally) {
    // `SmilesToMol` throws on some malformed input rather than returning null, so it belongs inside the
    // guard alongside `addHs`.
    std::unique_ptr<RDKit::RWMol> molecule;
    try {
        molecule.reset(RDKit::SmilesToMol(smiles));
        if (molecule) RDKit::MolOps::addHs(*molecule);
    }
    catch (...) {
        molecule.reset();
    }
    if (!molecule) {
        ++tally.unparsed;
        return;
    }
    switch (classify_capacity_(*molecule)) {
    case descriptor_capacity_t::too_many_atoms_k: ++tally.too_many_atoms; return;
    case descriptor_capacity_t::too_many_bonds_k: ++tally.too_many_bonds; return;
    case descriptor_capacity_t::fits_k: break;
    }
    ++tally.compared;
    check_molecule_(*molecule, smiles, reporter, scratch);
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
    sftest::descriptor_scratch_t scratch;

    for (auto const &smiles : sftest::standard_test_smiles()) sftest::check_builtin_(smiles, reporter, scratch);
    for (auto const &smiles : sftest::stress_test_smiles_()) sftest::check_builtin_(smiles, reporter, scratch);
    reporter.summarize("descriptor_builtin");

    if (selection.knob == sftest::corpus_knob_t::named_k) {
        // Naming a corpus is a request to validate against it, so a path that does not open is a failure
        // rather than an empty run: the sections would otherwise close with nothing compared.
        std::vector<std::string> corpus;
        auto const outcome = sftest::load_smiles_into(selection.path, selection.max_molecules, corpus);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), selection.path);
        fmt::println("{}: corpus {}, {} SMILES", test_name_k, selection.path, fmt::group_digits(corpus.size()));

        sftest::descriptor_tally_t tally;
        for (auto const &smiles : corpus) sftest::check_corpus_line_(smiles, reporter, scratch, tally);
        fmt::println("  NOTE descriptor_corpus molecules={} unparsed={} too_many_atoms={} too_many_bonds={}",
                     tally.compared, tally.unparsed, tally.too_many_atoms, tally.too_many_bonds);
        reporter.summarize("descriptor_corpus");
    }

    if (!reporter.passed())
        return sftest::report_fail(test_name_k, "{} mismatches over {} comparisons", reporter.mismatch_count(),
                                   reporter.comparison_count());
    return sftest::report_pass(test_name_k);
}
