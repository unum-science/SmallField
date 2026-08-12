/**
 *  @file test/parameters/test_mmff94_tables.cpp
 *  @brief Cross-validate the extracted MMFF94 tables against RDKit on the host, then against the device.
 *
 *  Two comparisons run over one shared set of probe keys. The host suite puts every key to RDKit's live
 *  `operator()` and to our binary-search cascade; the device suite puts the same keys to the lookup kernels
 *  and to the host cascade the first comparison has just vouched for.
 *
 *  Keys come from three sources. A sweep of the whole atom-type range, including two types past the last
 *  tabulated one, covers the vdW table exhaustively. One key per tabulated entry reaches every parameter the
 *  tables carry. A uniform draw over the type range reaches the equivalence cascades and the miss paths that
 *  neither of the first two exercise. A corpus named on the command line adds the keys real molecules ask
 *  for, taken from RDKit's own bond, angle and torsion typing of each structure.
 *
 *  Every key is a comparison whether or not it finds an entry, because two lookups agreeing that a table has
 *  nothing at a key is exactly the agreement a spurious device-side hit would break. How many keys reached a
 *  real entry is reported separately: a suite that agreed ten thousand times about empty space has compared
 *  no parameters at all, and its `compared=` count alone cannot say so.
 */

#include <cmath>   // `std::abs`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`
#include <cstdio>  // `std::fflush`
#include <cstring> // `std::memcpy`

#include <initializer_list> // `std::initializer_list`
#include <iostream>         // RDKit's `AtomTyper.h` defaults a parameter to `std::cout` without including it
#include <memory>           // `std::unique_ptr`
#include <random>           // `std::mt19937`, `std::uniform_int_distribution`
#include <sstream>          // RDKit's `AtomTyper.h` uses `std::stringstream` without including it
#include <string>           // `std::string`
#include <string_view>      // `std::string_view`
#include <utility>          // `std::pair`
#include <vector>           // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <boost/shared_ptr.hpp>
#include <cuda_runtime.h>
#include <RDGeneral/types.h>
#include <ForceField/MMFF/Params.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>

#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_rdkit.hpp"

#include "test/parameters/test_mmff94_tables_kernels.hpp"
#include "test/test_helpers.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sf = smallfield;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_mmff94_tables";

namespace smallfield::tests {

#pragma region Predicates and Budgets

static bool approximately_equal_(float lhs, double rhs, float tolerance = 1e-5f) {
    float const rhs_as_float = static_cast<float>(rhs);
    return std::abs(lhs - rhs_as_float) < tolerance;
}

static bool bitwise_equal_(float lhs, float rhs) {
    std::uint32_t lhs_bits;
    std::uint32_t rhs_bits;
    std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
    std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
    return lhs_bits == rhs_bits;
}

/*  `comparison_group_t::field` invokes its predicate as `equals(reference, observed)`, so each suite's
 *  comparison is wrapped rather than reimplemented. RDKit tabulates in `double` and we narrow to `float`, so
 *  the host suite carries a tolerance; the device reads the very bytes the host uploaded, so any difference
 *  there is a lookup bug and the test is bitwise. */
static constexpr auto rdkit_within_tolerance_ = [](double rdkit_value, float our_value) {
    return approximately_equal_(our_value, rdkit_value);
};
static constexpr auto bits_identical_ = [](float host_value, float device_value) {
    return bitwise_equal_(host_value, device_value);
};

/** How many mismatching fields a reporter prints before the rest are only counted. A systematic divergence
 *  would otherwise print one line per probe and bury every other section in the log. */
static constexpr std::size_t max_reported_mismatches_k = 64;

/** Uniform draws per table, over the whole atom-type range. These are the keys that reach the equivalence
 *  cascades and the miss paths, since a key taken from a tabulated entry hits by construction. */
static constexpr std::uint32_t random_probes_per_table_k = 2000;

/** Molecules read from a named corpus when neither the flag nor `SMALLFIELD_TEST_MAX_MOLS` caps it. */
static constexpr std::size_t default_max_molecules_k = 256;

/** Corpus molecules above this atom count are skipped: the angle and torsion walks below are quadratic in
 *  the neighborhood, and one 900-atom aggregate contributes more keys than a hundred drug-like structures. */
static constexpr std::uint32_t max_corpus_atoms_k = 128;

#pragma endregion Predicates and Budgets

#pragma region Coverage

/**
 *  @brief Probe-granularity coverage for one lookup suite, owned by `main` and threaded by reference.
 *
 *  The reporter beside it owns the verdict and counts field comparisons, so one vdW probe that hits
 *  contributes six comparisons there and one probe here. What the reporter cannot know is how many probes
 *  reached a real entry, and that is the number separating a suite that compared parameters from one that
 *  agreed several thousand times that a table is empty at a random key. Neither count is derivable from the
 *  other, and only the reporter's decides the verdict.
 */
struct lookup_coverage_t {
    std::uint64_t probes = 0;
    std::uint64_t table_hits = 0;
};

/** Coverage of every table one side compares, so `main` declares one object per side rather than eight. */
struct suite_coverage_t {
    lookup_coverage_t van_der_waals;
    lookup_coverage_t bond;
    lookup_coverage_t angle;
    lookup_coverage_t torsion;
};

/** Close a section's coverage, above the `compared=` line that closes its comparisons. */
static void report_coverage_(std::string_view subject, lookup_coverage_t const &coverage) {
    fmt::println("NOTE {} probes={} table_hits={}", subject, coverage.probes, coverage.table_hits);
}

#pragma endregion Coverage

#pragma region Probe Keys

/**
 *  @brief Every key both comparisons put to the tables, filled once and read twice.
 *
 *  One key set for the host and the device suite is what makes their two verdicts comparable: a key that our
 *  cascade resolved against RDKit on the host is the same key the kernel resolves against our cascade.
 */
struct probe_keys_t {
    std::vector<vdw_query_t> vdw_keys;
    std::vector<bond_query_t> bond_keys;
    std::vector<angle_query_t> angle_keys;
    std::vector<torsion_query_t> torsion_keys;
};

/**
 *  @brief The highest atom type RDKit's own equivalence table defines.
 *
 *  95 against the 99 the vdW table reaches, and the gap is not cosmetic: the cascades below index `MMFFDef`
 *  by atom type and dereference the row without a null check, so a type past its last row is an RDKit-side
 *  segfault rather than a miss.
 */
static std::uint8_t highest_defined_atom_type_() {
    return static_cast<std::uint8_t>(RDKit::MMFF::DefaultParameters::getMMFFDef()->d_params.size());
}

/**
 *  @brief Whether every atom type in a key is one RDKit's angle and torsion lookups can be asked about.
 *
 *  Those two walk the equivalence table and dereference what it answers, and it answers null both for the
 *  wildcard 0 and for anything past its last row, so either kind of key faults inside RDKit instead of
 *  missing. Neither is lost coverage: the wildcard rows are what the last level of the cascade rewrites a
 *  real type into, and an undefined type reaches no parameters on either side.
 *
 *  @param[in] highest_defined Last atom type the equivalence table carries.
 *  @param[in] atom_types The key's atom types, in any order.
 *  @return True when every one of them lies in the closed range from 1 to @p highest_defined .
 */
static bool every_type_is_defined_(std::uint8_t highest_defined, std::initializer_list<std::uint8_t> atom_types) {
    for (std::uint8_t const atom_type : atom_types)
        if (atom_type == 0 || atom_type > highest_defined) return false;
    return true;
}

/**
 *  @brief Append the exhaustive sweep, one key per tabulated entry, and a uniform draw over the type range.
 *
 *  @param[in] host_tables Extracted tables, read for their entry keys and their type ceiling.
 *  @param[inout] keys Key set to extend.
 */
static void append_synthetic_keys_into_(sf::parameters::mmff94_tables_host_t const &host_tables, probe_keys_t &keys) {
    // Two types past the last tabulated one, so the out-of-range miss is covered on both sides.
    for (std::uint32_t atom_type = 0; atom_type <= host_tables.max_atom_type + 2u; ++atom_type)
        keys.vdw_keys.push_back({.atom_type = static_cast<std::uint8_t>(atom_type)});

    for (auto const &entry : host_tables.bond)
        keys.bond_keys.push_back(
            {.bond_type = entry.bond_type, .atom_type_i = entry.i_type, .atom_type_j = entry.j_type});
    std::uint8_t const highest_defined = highest_defined_atom_type_();
    for (auto const &entry : host_tables.angle) {
        if (!every_type_is_defined_(highest_defined, {entry.i_type, entry.j_type, entry.k_type})) continue;
        keys.angle_keys.push_back({.angle_type = entry.angle_type,
                                   .atom_type_i = entry.i_type,
                                   .atom_type_j_center = entry.j_type,
                                   .atom_type_k = entry.k_type});
    }
    for (auto const &entry : host_tables.torsion) {
        if (!every_type_is_defined_(highest_defined, {entry.i_type, entry.j_type, entry.k_type, entry.l_type}))
            continue;
        keys.torsion_keys.push_back({.primary_torsion_type = entry.tor_type,
                                     .secondary_torsion_type = entry.tor_type,
                                     .atom_type_i = entry.i_type,
                                     .atom_type_j = entry.j_type,
                                     .atom_type_k = entry.k_type,
                                     .atom_type_l = entry.l_type});
    }

    std::mt19937 generator(2026);
    std::uniform_int_distribution<std::uint32_t> atom_type_distribution(1, highest_defined);
    std::uniform_int_distribution<std::uint32_t> bond_type_distribution(0, 1);
    std::uniform_int_distribution<std::uint32_t> angle_type_distribution(0, 7);
    std::uniform_int_distribution<std::uint32_t> torsion_type_distribution(0, 5);
    auto random_atom_type = [&] { return static_cast<std::uint8_t>(atom_type_distribution(generator)); };
    for (std::uint32_t trial = 0; trial < random_probes_per_table_k; ++trial) {
        keys.bond_keys.push_back({.bond_type = static_cast<std::uint8_t>(bond_type_distribution(generator)),
                                  .atom_type_i = random_atom_type(),
                                  .atom_type_j = random_atom_type()});
        keys.angle_keys.push_back({.angle_type = static_cast<std::uint8_t>(angle_type_distribution(generator)),
                                   .atom_type_i = random_atom_type(),
                                   .atom_type_j_center = random_atom_type(),
                                   .atom_type_k = random_atom_type()});
        keys.torsion_keys.push_back(
            {.primary_torsion_type = static_cast<std::uint8_t>(torsion_type_distribution(generator)),
             .secondary_torsion_type = static_cast<std::uint8_t>(torsion_type_distribution(generator)),
             .atom_type_i = random_atom_type(),
             .atom_type_j = random_atom_type(),
             .atom_type_k = random_atom_type(),
             .atom_type_l = random_atom_type()});
    }
}

/** Neighbor indices reused across molecules, so the walk below allocates once instead of per molecule. */
struct neighbor_scratch_t {
    std::vector<std::uint32_t> around_center;
    std::vector<std::uint32_t> around_first;
    std::vector<std::uint32_t> around_second;
};

/**
 *  @brief Collect the atom indices bonded to @p atom.
 *
 *  @param[in] molecule Molecule to walk.
 *  @param[in] atom Index of the atom whose neighbors are wanted.
 *  @param[out] out Cleared, then filled with the neighbor indices.
 */
static void fill_neighbors_into_(RDKit::ROMol const &molecule, std::uint32_t atom, std::vector<std::uint32_t> &out) {
    out.clear();
    for (auto const *neighbor : molecule.atomNeighbors(molecule.getAtomWithIdx(atom)))
        out.push_back(neighbor->getIdx());
}

/** Whether RDKit could type a molecule well enough for its terms to become probe keys. */
enum class molecule_typing_t {
    typed_k,
    refused_k,
};

/**
 *  @brief Append the keys one molecule's MMFF terms actually use.
 *
 *  RDKit answers with the bond, angle and torsion types the force field itself would build, so these keys
 *  carry the distribution a production run sees where the uniform draws carry the one that reaches the miss
 *  paths. The vdW table is left alone: the sweep above already covers every atom type exhaustively, and a
 *  key per atom would only add duplicates.
 *
 *  @param[inout] molecule Molecule with explicit hydrogens; RDKit marks and types it in place.
 *  @param[inout] scratch Neighbor lists, reused across molecules.
 *  @param[inout] keys Key set to extend.
 *  @return Whether RDKit typed the molecule.
 */
static molecule_typing_t append_molecule_keys_into_(RDKit::RWMol &molecule, neighbor_scratch_t &scratch,
                                                    probe_keys_t &keys) {
    // `MMFFMolProperties` guards only its Kekulize behind `_MMFFSanitized`, and `SmilesToMol` has already
    // sanitized, so marking the molecule suppresses the redundant pass and nothing else.
    if (!molecule.hasProp(RDKit::common_properties::_MMFFSanitized))
        molecule.setProp(RDKit::common_properties::_MMFFSanitized, 1, true);
    RDKit::MMFF::MMFFMolProperties properties(molecule);
    if (!properties.isValid()) return molecule_typing_t::refused_k;

    std::uint8_t const highest_defined = highest_defined_atom_type_();
    for (auto const *bond : molecule.bonds()) {
        std::uint32_t const atom_i = bond->getBeginAtomIdx();
        std::uint32_t const atom_j = bond->getEndAtomIdx();
        keys.bond_keys.push_back({.bond_type = static_cast<std::uint8_t>(properties.getMMFFBondType(bond)),
                                  .atom_type_i = properties.getMMFFAtomType(atom_i),
                                  .atom_type_j = properties.getMMFFAtomType(atom_j)});

        // A torsion is a bond plus one neighbor at each end, which is how RDKit enumerates them too.
        fill_neighbors_into_(molecule, atom_i, scratch.around_first);
        fill_neighbors_into_(molecule, atom_j, scratch.around_second);
        for (std::uint32_t const outer : scratch.around_first) {
            if (outer == atom_j) continue; // the bond itself is not a torsion arm
            for (std::uint32_t const inner : scratch.around_second) {
                if (inner == atom_i || inner == outer) continue; // a three-membered ring closes onto itself
                if (!every_type_is_defined_(highest_defined,
                                            {properties.getMMFFAtomType(outer), properties.getMMFFAtomType(atom_i),
                                             properties.getMMFFAtomType(atom_j), properties.getMMFFAtomType(inner)}))
                    continue; // RDKit's own torsion lookup faults on such a key
                auto const torsion_type = properties.getMMFFTorsionType(molecule, outer, atom_i, atom_j, inner);
                keys.torsion_keys.push_back({.primary_torsion_type = static_cast<std::uint8_t>(torsion_type.first),
                                             .secondary_torsion_type = static_cast<std::uint8_t>(torsion_type.second),
                                             .atom_type_i = properties.getMMFFAtomType(outer),
                                             .atom_type_j = properties.getMMFFAtomType(atom_i),
                                             .atom_type_k = properties.getMMFFAtomType(atom_j),
                                             .atom_type_l = properties.getMMFFAtomType(inner)});
            }
        }
    }

    std::uint32_t const num_atoms = molecule.getNumAtoms();
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        fill_neighbors_into_(molecule, center, scratch.around_center);
        for (std::size_t left = 0; left + 1 < scratch.around_center.size(); ++left)
            for (std::size_t right = left + 1; right < scratch.around_center.size(); ++right) {
                std::uint32_t const atom_i = scratch.around_center[left];
                std::uint32_t const atom_k = scratch.around_center[right];
                if (!every_type_is_defined_(highest_defined,
                                            {properties.getMMFFAtomType(atom_i), properties.getMMFFAtomType(center),
                                             properties.getMMFFAtomType(atom_k)}))
                    continue; // RDKit's own angle lookup faults on such a key
                auto const angle_type = properties.getMMFFAngleType(molecule, atom_i, center, atom_k);
                keys.angle_keys.push_back({.angle_type = static_cast<std::uint8_t>(angle_type),
                                           .atom_type_i = properties.getMMFFAtomType(atom_i),
                                           .atom_type_j_center = properties.getMMFFAtomType(center),
                                           .atom_type_k = properties.getMMFFAtomType(atom_k)});
            }
    }
    return molecule_typing_t::typed_k;
}

/**
 *  @brief Type every corpus molecule and append the keys its terms use.
 *
 *  @param[in] corpus SMILES already loaded from the named path.
 *  @param[inout] keys Key set to extend.
 *  @return How many molecules RDKit typed; one it refuses contributes no keys.
 */
static std::size_t append_corpus_keys_into_(std::vector<std::string> const &corpus, probe_keys_t &keys) {
    neighbor_scratch_t scratch;
    std::size_t typed_molecules = 0;
    for (auto const &smiles : corpus) {
        std::unique_ptr<RDKit::RWMol> molecule;
        // `SmilesToMol` throws on some malformed input rather than returning null, so it belongs inside the
        // guard alongside `addHs`.
        try {
            RDKit::SmilesParserParams parser_params;
            parser_params.sanitize = true;
            parser_params.removeHs = false;
            molecule.reset(RDKit::SmilesToMol(smiles, parser_params));
            if (!molecule) continue;
            RDKit::MolOps::addHs(*molecule);
        }
        catch (...) {
            continue;
        }
        if (molecule->getNumAtoms() > max_corpus_atoms_k) continue;
        try {
            if (append_molecule_keys_into_(*molecule, scratch, keys) == molecule_typing_t::typed_k) ++typed_molecules;
        }
        catch (...) {
            continue; // RDKit throws out of its own typing on a few structures rather than reporting invalid
        }
    }
    return typed_molecules;
}

#pragma endregion Probe Keys

#pragma region Host Against RDKit

static void compare_vdw_against_rdkit_(sf::parameters::mmff94_tables_view_t const &tables,
                                       std::vector<vdw_query_t> const &queries, diagnostic_reporter_t &reporter,
                                       lookup_coverage_t &coverage) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFVdW();
    for (auto const &query : queries) {
        auto const *ours = sf::parameters::lookup_vdw(tables, query.atom_type);
        auto const *reference = (*collection)(query.atom_type);
        ++coverage.probes;
        auto entry = reporter.mismatch("vdw", "atom_type={}", query.atom_type);
        entry.field("present", reference != nullptr, ours != nullptr, exactly_equal);
        if (reference == nullptr || ours == nullptr) continue; // one side carries no parameters to compare
        ++coverage.table_hits;
        // RDKit stores the donor/acceptor flag as the tabulated character; the extraction maps it onto a
        // small enumeration, and this restates that mapping rather than reading it back from the extractor.
        std::uint8_t const reference_donor_acceptor = static_cast<std::uint8_t>((reference->DA == 'D')   ? 2
                                                                                : (reference->DA == 'A') ? 1
                                                                                                         : 0);
        entry.field("alpha", reference->alpha_i, ours->alpha_i, rdkit_within_tolerance_)
            .field("n_effective", reference->N_i, ours->n_effective, rdkit_within_tolerance_)
            .field("a_scale", reference->A_i, ours->a_scale, rdkit_within_tolerance_)
            .field("g_factor", reference->G_i, ours->g_factor, rdkit_within_tolerance_)
            .field("r_star", reference->R_star, ours->r_star, rdkit_within_tolerance_)
            .field("donor_acceptor", reference_donor_acceptor, ours->donor_acceptor, exactly_equal);
    }
    report_coverage_("vdw", coverage);
    reporter.summarize("vdw");
}

static void compare_bond_against_rdkit_(sf::parameters::mmff94_tables_view_t const &tables,
                                        std::vector<bond_query_t> const &queries, diagnostic_reporter_t &reporter,
                                        lookup_coverage_t &coverage) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFBond();
    for (auto const &query : queries) {
        auto const *ours = sf::parameters::lookup_bond(tables, query.bond_type, query.atom_type_i, query.atom_type_j);
        auto const *reference = (*collection)(query.bond_type, query.atom_type_i, query.atom_type_j);
        ++coverage.probes;
        auto entry = reporter.mismatch("bond", "bond_type={} atom_i={} atom_j={}", query.bond_type, query.atom_type_i,
                                       query.atom_type_j);
        entry.field("present", reference != nullptr, ours != nullptr, exactly_equal);
        if (reference == nullptr || ours == nullptr) continue; // one side carries no parameters to compare
        ++coverage.table_hits;
        entry.field("force_constant", reference->kb, ours->force_constant, rdkit_within_tolerance_)
            .field("equilibrium_distance", reference->r0, ours->equilibrium_distance, rdkit_within_tolerance_);
    }
    report_coverage_("bond", coverage);
    reporter.summarize("bond");
}

static void compare_angle_against_rdkit_(sf::parameters::mmff94_tables_view_t const &tables,
                                         std::vector<angle_query_t> const &queries, diagnostic_reporter_t &reporter,
                                         lookup_coverage_t &coverage) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFAngle();
    auto const *definitions = RDKit::MMFF::DefaultParameters::getMMFFDef();
    for (auto const &query : queries) {
        auto const *ours = sf::parameters::lookup_angle(tables, query.angle_type, query.atom_type_i,
                                                        query.atom_type_j_center, query.atom_type_k);
        auto const *reference = (*collection)(definitions, query.angle_type, query.atom_type_i,
                                              query.atom_type_j_center, query.atom_type_k);
        ++coverage.probes;
        auto entry = reporter.mismatch("angle", "angle_type={} atom_i={} atom_j={} atom_k={}", query.angle_type,
                                       query.atom_type_i, query.atom_type_j_center, query.atom_type_k);
        entry.field("present", reference != nullptr, ours != nullptr, exactly_equal);
        if (reference == nullptr || ours == nullptr) continue; // one side carries no parameters to compare
        ++coverage.table_hits;
        entry.field("force_constant", reference->ka, ours->force_constant, rdkit_within_tolerance_)
            .field("theta0_deg", reference->theta0, ours->theta0_deg, rdkit_within_tolerance_);
    }
    report_coverage_("angle", coverage);
    reporter.summarize("angle");
}

static void compare_torsion_against_rdkit_(sf::parameters::mmff94_tables_view_t const &tables,
                                           std::vector<torsion_query_t> const &queries, diagnostic_reporter_t &reporter,
                                           lookup_coverage_t &coverage) {
    auto const *collection = RDKit::MMFF::DefaultParameters::getMMFFTor(false);
    auto const *definitions = RDKit::MMFF::DefaultParameters::getMMFFDef();
    for (auto const &query : queries) {
        auto const ours = sf::parameters::lookup_torsion(tables, query.primary_torsion_type,
                                                         query.secondary_torsion_type, query.atom_type_i,
                                                         query.atom_type_j, query.atom_type_k, query.atom_type_l);
        // RDKit takes the primary and secondary torsion types as one pair, which the braced list constructs.
        auto const reference = collection->getMMFFTorParams(
            definitions, {query.primary_torsion_type, query.secondary_torsion_type}, query.atom_type_i,
            query.atom_type_j, query.atom_type_k, query.atom_type_l);
        ++coverage.probes;
        auto entry = reporter.mismatch(
            "torsion", "primary_torsion_type={} secondary_torsion_type={} atom_i={} atom_j={} atom_k={} atom_l={}",
            query.primary_torsion_type, query.secondary_torsion_type, query.atom_type_i, query.atom_type_j,
            query.atom_type_k, query.atom_type_l);
        entry.field("present", reference.second != nullptr, ours.second != nullptr, exactly_equal);
        if (reference.second == nullptr || ours.second == nullptr) continue; // nothing to compare on one side
        ++coverage.table_hits;
        entry.field("canonical_torsion_type", reference.first, ours.first, exactly_equal)
            .field("torsion_v1", reference.second->V1, ours.second->torsion_v1, rdkit_within_tolerance_)
            .field("torsion_v2", reference.second->V2, ours.second->torsion_v2, rdkit_within_tolerance_)
            .field("torsion_v3", reference.second->V3, ours.second->torsion_v3, rdkit_within_tolerance_);
    }
    report_coverage_("torsion", coverage);
    reporter.summarize("torsion");
}

/**
 *  @brief Put every key to RDKit's live lookups and to ours.
 *
 *  @param[in] host_tables Extracted tables, viewed for the lookups under test.
 *  @param[in] keys Shared probe keys.
 *  @param[inout] reporter Owns the verdict for all four sections.
 *  @param[inout] coverage Probe counts per table.
 */
static void compare_against_rdkit_(sf::parameters::mmff94_tables_host_t const &host_tables, probe_keys_t const &keys,
                                   diagnostic_reporter_t &reporter, suite_coverage_t &coverage) {
    // Both suites close a section per table under the same subject names, so each says which one it is.
    fmt::println("NOTE suite=host reference=rdkit observed=ours");
    auto const tables = host_tables.view();
    compare_vdw_against_rdkit_(tables, keys.vdw_keys, reporter, coverage.van_der_waals);
    compare_bond_against_rdkit_(tables, keys.bond_keys, reporter, coverage.bond);
    compare_angle_against_rdkit_(tables, keys.angle_keys, reporter, coverage.angle);
    compare_torsion_against_rdkit_(tables, keys.torsion_keys, reporter, coverage.torsion);
}

#pragma endregion Host Against RDKit

#pragma region Device Against Host

/** Which step of a lookup kernel's round trip refused, so a skipped section can say why. */
enum class device_probe_t {
    completed_k,
    allocation_failed_k,
    transfer_failed_k,
    launch_failed_k,
};

/** The sentence a `device_probe_t` reads as, for the `NOTE` line above a skipped section. */
static std::string_view device_probe_reason_(device_probe_t outcome) noexcept {
    switch (outcome) {
    case device_probe_t::allocation_failed_k: return "device_allocation_failed";
    case device_probe_t::transfer_failed_k: return "device_transfer_failed";
    case device_probe_t::launch_failed_k: return "kernel_launch_failed";
    default: return "completed";
    }
}

/**
 *  @brief Ship @p queries to the device, run @p launch over them, and bring the results back.
 *
 *  Every section's round trip is this one function, so the allocate, copy, launch, copy, free sequence is
 *  written once and each section holds only its comparison.
 *
 *  @param[in] queries Probe keys for one table.
 *  @param[out] results Resized to match @p queries and overwritten with what the kernel wrote.
 *  @param[in] launch Kernel launcher, taking the device query and result pointers and the query count.
 *  @return Whether the round trip completed, and which step refused it.
 */
template <typename query_type_, typename result_type_, typename launch_type_>
static device_probe_t run_device_probes_(std::vector<query_type_> const &queries, std::vector<result_type_> &results,
                                         launch_type_ launch) {
    results.assign(queries.size(), result_type_ {});
    if (queries.empty()) return device_probe_t::completed_k;

    std::size_t const query_bytes = queries.size() * sizeof(query_type_);
    std::size_t const result_bytes = results.size() * sizeof(result_type_);
    query_type_ *query_device = nullptr;
    result_type_ *result_device = nullptr;
    device_probe_t outcome = device_probe_t::completed_k;
    if (cudaMalloc(&query_device, query_bytes) != cudaSuccess ||
        cudaMalloc(&result_device, result_bytes) != cudaSuccess)
        outcome = device_probe_t::allocation_failed_k;
    else if (cudaMemcpy(query_device, queries.data(), query_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        outcome = device_probe_t::transfer_failed_k;
    else {
        launch(query_device, result_device, static_cast<std::uint32_t>(queries.size()));
        if (cudaDeviceSynchronize() != cudaSuccess) outcome = device_probe_t::launch_failed_k;
        else if (cudaMemcpy(results.data(), result_device, result_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
            outcome = device_probe_t::transfer_failed_k;
    }
    cudaFree(query_device);
    cudaFree(result_device);
    return outcome;
}

/**
 *  @brief Close a section that never ran, so the skip reaches the gate.
 *
 *  The vacuous-section count only rises inside `summarize`, so a section that returns before closing leaves
 *  a passing reporter behind and takes the whole comparison with it silently.
 */
static void close_skipped_section_(std::string_view subject, std::string_view reason, diagnostic_reporter_t &reporter) {
    // Deliberately not the word `FAILED`: this line is narration, and a CI log filtered on the verdict
    // grammar must not answer a `FAIL` grep with a line that never decided anything.
    fmt::println("NOTE {} device_probe={}", subject, reason);
    reporter.summarize(subject);
}

static void compare_vdw_on_device_(sf::parameters::mmff94_tables_view_t const &host_tables,
                                   sf::parameters::mmff94_tables_view_t const &device_tables,
                                   std::vector<vdw_query_t> const &queries, diagnostic_reporter_t &reporter,
                                   lookup_coverage_t &coverage) {
    std::vector<vdw_result_t> results;
    auto const outcome = run_device_probes_(
        queries, results, [&](vdw_query_t const *device_queries, vdw_result_t *device_results, std::uint32_t count) {
            launch_device_vdw_lookup(device_tables, device_queries, device_results, count);
        });
    if (outcome != device_probe_t::completed_k)
        return close_skipped_section_("vdw", device_probe_reason_(outcome), reporter);

    for (std::size_t probe = 0; probe < queries.size(); ++probe) {
        auto const *ours = sf::parameters::lookup_vdw(host_tables, queries[probe].atom_type);
        bool const host_hit = ours != nullptr;
        ++coverage.probes;
        auto entry = reporter.mismatch("vdw", "atom_type={}", queries[probe].atom_type);
        entry.field("hit", host_hit, results[probe].hit != 0, exactly_equal);
        if (!host_hit || results[probe].hit == 0) continue; // one side resolved nothing to compare
        ++coverage.table_hits;
        entry.field("alpha", ours->alpha_i, results[probe].alpha, bits_identical_)
            .field("n_effective", ours->n_effective, results[probe].effective_electron_count, bits_identical_)
            .field("a_scale", ours->a_scale, results[probe].polarizability_scale, bits_identical_)
            .field("g_factor", ours->g_factor, results[probe].slater_kirkwood_factor, bits_identical_)
            .field("r_star", ours->r_star, results[probe].r_star, bits_identical_)
            .field("donor_acceptor", ours->donor_acceptor, results[probe].donor_acceptor, exactly_equal);
    }
    report_coverage_("vdw", coverage);
    reporter.summarize("vdw");
}

static void compare_bond_on_device_(sf::parameters::mmff94_tables_view_t const &host_tables,
                                    sf::parameters::mmff94_tables_view_t const &device_tables,
                                    std::vector<bond_query_t> const &queries, diagnostic_reporter_t &reporter,
                                    lookup_coverage_t &coverage) {
    std::vector<bond_result_t> results;
    auto const outcome = run_device_probes_(
        queries, results, [&](bond_query_t const *device_queries, bond_result_t *device_results, std::uint32_t count) {
            launch_device_bond_lookup(device_tables, device_queries, device_results, count);
        });
    if (outcome != device_probe_t::completed_k)
        return close_skipped_section_("bond", device_probe_reason_(outcome), reporter);

    for (std::size_t probe = 0; probe < queries.size(); ++probe) {
        auto const *ours = sf::parameters::lookup_bond(host_tables, queries[probe].bond_type,
                                                       queries[probe].atom_type_i, queries[probe].atom_type_j);
        bool const host_hit = ours != nullptr;
        ++coverage.probes;
        auto entry = reporter.mismatch("bond", "bond_type={} atom_i={} atom_j={}", queries[probe].bond_type,
                                       queries[probe].atom_type_i, queries[probe].atom_type_j);
        entry.field("hit", host_hit, results[probe].hit != 0, exactly_equal);
        if (!host_hit || results[probe].hit == 0) continue; // one side resolved nothing to compare
        ++coverage.table_hits;
        entry.field("force_constant", ours->force_constant, results[probe].force_constant, bits_identical_)
            .field("equilibrium_distance", ours->equilibrium_distance, results[probe].equilibrium_distance,
                   bits_identical_);
    }
    report_coverage_("bond", coverage);
    reporter.summarize("bond");
}

static void compare_angle_on_device_(sf::parameters::mmff94_tables_view_t const &host_tables,
                                     sf::parameters::mmff94_tables_view_t const &device_tables,
                                     std::vector<angle_query_t> const &queries, diagnostic_reporter_t &reporter,
                                     lookup_coverage_t &coverage) {
    std::vector<angle_result_t> results;
    auto const outcome = run_device_probes_(
        queries, results,
        [&](angle_query_t const *device_queries, angle_result_t *device_results, std::uint32_t count) {
            launch_device_angle_lookup(device_tables, device_queries, device_results, count);
        });
    if (outcome != device_probe_t::completed_k)
        return close_skipped_section_("angle", device_probe_reason_(outcome), reporter);

    for (std::size_t probe = 0; probe < queries.size(); ++probe) {
        auto const *ours = sf::parameters::lookup_angle(host_tables, queries[probe].angle_type,
                                                        queries[probe].atom_type_i, queries[probe].atom_type_j_center,
                                                        queries[probe].atom_type_k);
        bool const host_hit = ours != nullptr;
        ++coverage.probes;
        auto entry = reporter.mismatch("angle", "angle_type={} atom_i={} atom_j={} atom_k={}",
                                       queries[probe].angle_type, queries[probe].atom_type_i,
                                       queries[probe].atom_type_j_center, queries[probe].atom_type_k);
        entry.field("hit", host_hit, results[probe].hit != 0, exactly_equal);
        if (!host_hit || results[probe].hit == 0) continue; // one side resolved nothing to compare
        ++coverage.table_hits;
        entry.field("force_constant", ours->force_constant, results[probe].force_constant, bits_identical_)
            .field("theta0_deg", ours->theta0_deg, results[probe].equilibrium_angle_degrees, bits_identical_);
    }
    report_coverage_("angle", coverage);
    reporter.summarize("angle");
}

static void compare_torsion_on_device_(sf::parameters::mmff94_tables_view_t const &host_tables,
                                       sf::parameters::mmff94_tables_view_t const &device_tables,
                                       std::vector<torsion_query_t> const &queries, diagnostic_reporter_t &reporter,
                                       lookup_coverage_t &coverage) {
    std::vector<torsion_result_t> results;
    auto const outcome = run_device_probes_(
        queries, results,
        [&](torsion_query_t const *device_queries, torsion_result_t *device_results, std::uint32_t count) {
            launch_device_torsion_lookup(device_tables, device_queries, device_results, count);
        });
    if (outcome != device_probe_t::completed_k)
        return close_skipped_section_("torsion", device_probe_reason_(outcome), reporter);

    for (std::size_t probe = 0; probe < queries.size(); ++probe) {
        auto const ours = sf::parameters::lookup_torsion(host_tables, queries[probe].primary_torsion_type,
                                                         queries[probe].secondary_torsion_type,
                                                         queries[probe].atom_type_i, queries[probe].atom_type_j,
                                                         queries[probe].atom_type_k, queries[probe].atom_type_l);
        bool const host_hit = ours.second != nullptr;
        ++coverage.probes;
        auto entry = reporter.mismatch(
            "torsion", "primary_torsion_type={} secondary_torsion_type={} atom_i={} atom_j={} atom_k={} atom_l={}",
            queries[probe].primary_torsion_type, queries[probe].secondary_torsion_type, queries[probe].atom_type_i,
            queries[probe].atom_type_j, queries[probe].atom_type_k, queries[probe].atom_type_l);
        entry.field("hit", host_hit, results[probe].hit != 0, exactly_equal);
        if (!host_hit || results[probe].hit == 0) continue; // one side resolved nothing to compare
        ++coverage.table_hits;
        entry
            .field("canonical_torsion_type", static_cast<std::uint8_t>(ours.first),
                   results[probe].canonical_torsion_type, exactly_equal)
            .field("torsion_v1", ours.second->torsion_v1, results[probe].torsion_v1, bits_identical_)
            .field("torsion_v2", ours.second->torsion_v2, results[probe].torsion_v2, bits_identical_)
            .field("torsion_v3", ours.second->torsion_v3, results[probe].torsion_v3, bits_identical_);
    }
    report_coverage_("torsion", coverage);
    reporter.summarize("torsion");
}

/**
 *  @brief Upload the tables and put every key to the kernels, against the host lookups.
 *
 *  @param[in] host_tables Extracted tables, uploaded and also viewed as the reference.
 *  @param[in] keys Shared probe keys.
 *  @param[inout] reporter Owns the verdict for all four sections.
 *  @param[inout] coverage Probe counts per table.
 */
static void compare_against_device_(sf::parameters::mmff94_tables_host_t const &host_tables, probe_keys_t const &keys,
                                    diagnostic_reporter_t &reporter, suite_coverage_t &coverage) {
    fmt::println("NOTE suite=device reference=host observed=device");
    sf::parameters::mmff94_tables_view_t device_tables {};
    void *device_handle = device_tables_create(host_tables, device_tables);
    if (device_handle == nullptr) {
        // One skipped section rather than four: no upload means no device suite at all, and closing it
        // empty is what carries the skip to the gate in `main`.
        close_skipped_section_("device_upload", "device_upload_failed", reporter);
        return;
    }

    auto const tables = host_tables.view();
    compare_vdw_on_device_(tables, device_tables, keys.vdw_keys, reporter, coverage.van_der_waals);
    compare_bond_on_device_(tables, device_tables, keys.bond_keys, reporter, coverage.bond);
    compare_angle_on_device_(tables, device_tables, keys.angle_keys, reporter, coverage.angle);
    compare_torsion_on_device_(tables, device_tables, keys.torsion_keys, reporter, coverage.torsion);
    device_tables_destroy(device_handle);
}

#pragma endregion Device Against Host

#pragma region Narration

/** One line per extracted table, so a table that came back empty is visible before anything reads it. */
static void report_table_sizes_(sf::parameters::mmff94_tables_host_t const &host_tables) {
    fmt::println("NOTE mmff94_tables vdw={} bond={} angle={} torsion={} oop={} stbn={} dfsb={} " //
                 "definitions={} pbci={} chg={} herschbach_laurie={} max_atom_type={} vdw_power={}",
                 host_tables.vdw.size(), host_tables.bond.size(), host_tables.angle.size(), host_tables.torsion.size(),
                 host_tables.oop.size(), host_tables.stbn.size(), host_tables.dfsb.size(), host_tables.def.size(),
                 host_tables.pbci.size(), host_tables.chg.size(), host_tables.hl.size(), host_tables.max_atom_type,
                 host_tables.vdw_power);
}

/** One line naming how many keys each table will be probed with, before either suite runs. */
static void report_key_counts_(probe_keys_t const &keys) {
    fmt::println("NOTE probe_keys vdw={} bond={} angle={} torsion={}", keys.vdw_keys.size(), keys.bond_keys.size(),
                 keys.angle_keys.size(), keys.torsion_keys.size());
}

static void print_usage_() {
    fmt::println("usage: {} [corpus.smi] [max_molecules]", test_name_k);
    fmt::println("  corpus.smi     SMILES whose MMFF terms supply extra probe keys; falls back to");
    fmt::println("                 SMALLFIELD_TEST_CORPUS, and there is no built in default");
    fmt::println("  max_molecules  cap on molecules read from it, falling back to SMALLFIELD_TEST_MAX_MOLS,");
    fmt::println("                 then to {}", default_max_molecules_k);
}

/** How the command line resolved, before `select_corpus` reads the positional arguments behind it. */
enum class command_line_t {
    parsed_k,
    help_requested_k,
    refused_k,
};

/**
 *  @brief Answer the two conventional flags, so an operator reaching for either gets the usage.
 *
 *  The corpus itself arrives positionally, as it does in the seven other corpus-driven tests, so this walks
 *  the arguments only far enough to tell a flag from a path.
 *
 *  @param[in] argc Argument count as `main` received it.
 *  @param[in] argv Argument vector as `main` received it.
 *  @return Whether to carry on, print the usage, or refuse the command line.
 */
static command_line_t parse_flags_(int argc, char const *const *argv) {
    for (int argument = 1; argument < argc; ++argument) {
        std::string_view const text {argv[argument]};
        if (text == "-h" || text == "--help") return command_line_t::help_requested_k;
        if (text.size() > 1 && text[0] == '-') {
            std::fflush(stdout);
            fmt::println(stderr, "{}: unknown option {}", test_name_k, text);
            return command_line_t::refused_k;
        }
    }
    return command_line_t::parsed_k;
}

#pragma endregion Narration

} // namespace smallfield::tests

int main(int argc, char **argv) {
    switch (sftest::parse_flags_(argc, argv)) {
    case sftest::command_line_t::help_requested_k: sftest::print_usage_(); return 0;
    case sftest::command_line_t::refused_k: sftest::print_usage_(); return 2;
    case sftest::command_line_t::parsed_k: break;
    }
    auto const corpus_selection = sftest::select_corpus(argc, argv, sftest::default_max_molecules_k);
    if (corpus_selection.knob == sftest::corpus_knob_t::refused_k) {
        sftest::print_usage_();
        return 2;
    }

    fmt::println("{}: MMFF94 table lookups against RDKit on the host, then against the host on the device",
                 test_name_k);

    // Naming a corpus is a request to probe with it, so a path that does not open is a command line that
    // named the wrong thing rather than an empty run: exit 2, before any work, naming the path.
    std::vector<std::string> corpus;
    if (corpus_selection.knob == sftest::corpus_knob_t::named_k) {
        auto const outcome = sftest::load_smiles_into(corpus_selection.path, corpus_selection.max_molecules, corpus);
        if (outcome != sftest::smiles_load_t::loaded_k) {
            std::fflush(stdout);
            fmt::println(stderr, "{}: {} {}", test_name_k, sftest::smiles_load_reason(outcome), corpus_selection.path);
            return 2;
        }
    }

    sf::parameters::mmff94_tables_host_t host_tables;
    sf::parameters::extract_mmff94_tables_from_rdkit(host_tables);
    sftest::report_table_sizes_(host_tables);

    sftest::probe_keys_t keys;
    sftest::append_synthetic_keys_into_(host_tables, keys);
    if (!corpus.empty()) {
        std::size_t const typed = sftest::append_corpus_keys_into_(corpus, keys);
        fmt::println("NOTE corpus path={} smiles={} typed={}", corpus_selection.path, corpus.size(), typed);
    }
    sftest::report_key_counts_(keys);

    // RDKit's live lookups are the reference for the host suite; the host lookup is in turn the reference
    // for the kernels reading the uploaded tables.
    sftest::diagnostic_reporter_t host_reporter {"rdkit", "ours"};
    sftest::diagnostic_reporter_t device_reporter {"host", "device"};
    host_reporter.limit_printed_lines(sftest::max_reported_mismatches_k);
    device_reporter.limit_printed_lines(sftest::max_reported_mismatches_k);
    sftest::suite_coverage_t host_coverage;
    sftest::suite_coverage_t device_coverage;

    sftest::compare_against_rdkit_(host_tables, keys, host_reporter, host_coverage);
    sftest::compare_against_device_(host_tables, keys, device_reporter, device_coverage);

    std::size_t const compared = host_reporter.comparison_count() + device_reporter.comparison_count();
    std::size_t const mismatched = host_reporter.mismatch_count() + device_reporter.mismatch_count();
    fmt::println("NOTE mmff94_tables compared={} mismatches={} host_compared={} device_compared={}", compared,
                 mismatched, host_reporter.comparison_count(), device_reporter.comparison_count());
    if (mismatched != 0)
        return sftest::report_fail(test_name_k, "{} of {} field comparisons disagreed", mismatched, compared);
    // A section that compared nothing is indistinguishable from a passing one in the count above, so the
    // reporters gate on having actually compared something as well as on having agreed.
    if (!host_reporter.passed() || !device_reporter.passed())
        return sftest::report_fail(test_name_k, "a lookup section compared nothing at all");
    return sftest::report_pass(test_name_k);
}
