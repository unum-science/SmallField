/**
 *  @file test/preprocess/test_csd_torsion_predicates.cpp
 *  @brief Cross-validate the native CSD torsion matcher against RDKit.
 *
 *  Two layers of correctness gate:
 *   - @b Sub-query: every distinct `$(...)` body in the CSD pattern corpus is compiled by
 *     `compile_recursive_body` and cross-checked against `RDKit::SmartsToMol` and `SubstructMatch` at every
 *     atom of every test molecule.
 *   - @b Full-pattern: for every pattern the dispatcher marks `native==true`, the native matcher's
 *     per-torsion 4-tuples are compared byte-for-byte to `SubstructMatch`, reduced to
 *     `(atom_1,atom_2,atom_3,atom_4)` indices as an unordered multi-set modulo endpoint swap, the central
 *     bond being symmetric.
 *
 *  Corpus: the builtin standards by default, which already reach every predicate so a fresh clone stays
 *  green; `argv[1]` or `SMALLFIELD_TEST_CORPUS` streams SMILES from a `.smi` file instead, a named path
 *  that does not open being a failure and never an empty run; `argv[2]` or `SMALLFIELD_TEST_MAX_MOLS` caps
 *  molecules per corpus.
 *
 *  Divergence means the native semantics differ from RDKit — debug the native side, never weaken the
 *  assertion.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::uint32_t`
#include <cstdio>  // `stderr`
#include <cstring> // `std::memset`

#include <algorithm>     // `std::min`
#include <iterator>      // `std::size`
#include <memory>        // `std::unique_ptr`
#include <set>           // `std::set`
#include <string>        // `std::string`
#include <string_view>   // `std::string_view`
#include <tuple>         // `std::tuple`
#include <unordered_set> // `std::unordered_set`
#include <utility>       // `std::move`, `std::swap`
#include <vector>        // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include "smallfield/preprocess/csd_torsion_dispatcher.hpp"
#include "smallfield/preprocess/csd_torsion_native.hpp"
#include "smallfield/preprocess/csd_torsion_predicates.hpp"
#include "smallfield/types/conformer_status.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

#include "test/test_rdkit_reference_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_csd_torsion_predicates";

/** Every distinct `$(...)` body the CSD torsion corpus writes, as it appears between `$(` and `)`. Each compiles
 *  through the pattern compiler's own path, cross-checked atom by atom against RDKit. Single characters carry the
 *  chemistry: `H` hydrogens, `X` connections, `r` ring membership, lower-case aromaticity, `@` ring closure. */
static constexpr std::string_view recursive_bodies_k[] = {
    "C([F])([F])[F]",
    "C([NX3])N",
    "C([NX3])n",
    "C=O",
    "N(C)[!#1]",
    "N(C)~[!#1]",
    "N[CX4]",
    "S(=O)(=O)",
    "S(=O)=O",
    "[CX3](=[!O])",
    "[CX3]([C])([H])",
    "[CX3](a)=O",
    "[CX3]=O",
    "[CX4&r]([C;r])([C;r])",
    "[CX4H1]C(=O)O",
    "[C](=O)",
    "[C](=O)([$([NX3H1]),$([NX3H2])])[NX3H1]",
    "[NH0][CX4]",
    "[NH1,NH2,CH2]",
    "[NH1]",
    "[NX3H1,NX3H2]",
    "[NX3H1]",
    "[NX3H1]C",
    "[NX3H2]",
    "[NX3](c([nH1])n)",
    "[NX3](c([nX2H0])([nX2H0]))",
    "[NX3](cn)",
    "[NX3]c[nH0]",
    "[N][!#1]",
    "[N][CX4]",
    "[aH0](!@O)",
    "[a]([cH1])",
    "[a]([nH0,o])",
    "[c&r6](c[OH])",
    "[cH0]!@;-[*^2]",
    "[cH0](=O)",
    "[cH0](Cl)",
    "[cH0](F)",
    "[cH0]([$([NX3H2]),$([NX3H1])])",
    "[cH0]([CX3])",
    "[cH0]@;-[*^2]",
    "[cH0]Cl",
    "[cH0]F",
    "[cH0][NH1,NH2]",
    "[cH0][OH0]",
    "[cH0][OH1]",
    "[c]([NH1,NH2])",
    "[c]([cH,nX2H0])[cH,nX2H0]",
    "[n,N](-a)",
    "a([!nX2H0])([nX2H0])!@;-[NX3H1]",
    "a([!nX2H0])([nX2H0])@;-[NX3H1]",
    "aC(=O)(O)",
    "a[NH1,NH2]",
    "a[OH1]",
    "c!@c",
    "c([nX2H0])([nX2H0])",
    "c([nX2H0])(a(a)(a))!@;-c[nX2H0]",
    "c([nX2H0])(a(a)(a))@;-c[nX2H0]",
    "cO",
};

namespace smallfield::tests {

/** One torsion, as the four atom indices its pattern labelled `:1 :2 :3 :4`. */
using torsion_tuple_t = std::tuple<std::uint8_t, std::uint8_t, std::uint8_t, std::uint8_t>;

/** One recursive sub-query body, compiled by both matchers under comparison. Both sides compile once for the whole
 *  sweep, not once per molecule, so the corpus loop exercises matching rather than parsing. */
struct recursive_case_t {
    std::string_view body;
    sfp::recursive_pool_t pool;
    std::uint8_t root;
    std::unique_ptr<RDKit::ROMol const> query;
};

/** Comparison buffers the sweep owns, cleared at each use rather than rebuilt; `matches` and `rdkit_roots` keep their
 *  capacity across molecules, so the inner loop over the pattern corpus stops growing them after the first few. */
struct match_scratch_t {
    std::vector<RDKit::MatchVectType> matches;
    std::unordered_set<std::uint32_t> rdkit_roots;
    std::set<torsion_tuple_t> rdkit_tuples;
    std::set<torsion_tuple_t> native_tuples;
};

/** Collect into @p roots the root-atom indices `SubstructMatch(query)` hits in `molecule`. */
static void fill_rdkit_root_matches_into_(RDKit::ROMol const &molecule, RDKit::ROMol const &query,
                                          std::vector<RDKit::MatchVectType> &matches,
                                          std::unordered_set<std::uint32_t> &roots) {
    matches.clear();
    roots.clear();
    RDKit::SubstructMatch(molecule, query, matches, /*uniquify=*/false, /*recursionPossible=*/true);
    for (auto const &match : matches) {
        if (match.empty()) continue;
        roots.insert(static_cast<std::uint32_t>(match[0].second));
    }
}

/** Cross-validate one compiled sub-query against RDKit on one molecule, aborting on the first disagreement. */
static void validate_recursive_for_molecule_(recursive_case_t const &compiled, RDKit::ROMol const &molecule,
                                             sft::molecule_descriptor_t const &descriptor, std::string const &smiles,
                                             match_scratch_t &scratch, diagnostic_reporter_t &reporter) {
    fill_rdkit_root_matches_into_(molecule, *compiled.query, scratch.matches, scratch.rdkit_roots);

    std::uint32_t const num_atoms = descriptor.num_atoms;
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        auto group = reporter.mismatch("csd_subquery", "body=$({}) smiles={} atom={}", compiled.body, smiles, atom);
        group.field("root_match", scratch.rdkit_roots.count(atom) != 0,
                    sfp::recursive_query_match(descriptor, atom, compiled.pool, compiled.root), exactly_equal);
        sf_verify_(group.reported_field_count() == 0,
                   fmt::format("the native sub-query $({}) disagrees with RDKit on {}", compiled.body, smiles));
    }
}

/** Reduce RDKit `MatchVectType` hits to the torsion-atom 4-tuples the pattern's `:1 :2 :3 :4` labels identify,
 *  normalised so an endpoint swap — atom_1↔atom_4, atom_2↔atom_3 — gives the same canonical form. The two matchers
 *  may emit the orientations in different order, but torsion semantics is symmetric. */
static void fill_rdkit_torsion_tuples_into_(RDKit::ROMol const &molecule,
                                            sfp::csd_compiled_pattern_t const &compiled_pattern,
                                            std::vector<RDKit::MatchVectType> &matches,
                                            std::set<torsion_tuple_t> &out) {
    matches.clear();
    out.clear();
    RDKit::SubstructMatch(molecule, *compiled_pattern.pattern, matches, /*uniquify=*/false,
                          /*recursionPossible=*/true);
    for (auto const &match : matches) {
        std::uint32_t atom_1 = static_cast<std::uint32_t>(match[compiled_pattern.index[0]].second);
        std::uint32_t atom_2 = static_cast<std::uint32_t>(match[compiled_pattern.index[1]].second);
        std::uint32_t atom_3 = static_cast<std::uint32_t>(match[compiled_pattern.index[2]].second);
        std::uint32_t atom_4 = static_cast<std::uint32_t>(match[compiled_pattern.index[3]].second);
        if (atom_1 > atom_4 || (atom_1 == atom_4 && atom_2 > atom_3)) {
            std::swap(atom_1, atom_4);
            std::swap(atom_2, atom_3);
        }
        out.emplace(static_cast<std::uint8_t>(atom_1), static_cast<std::uint8_t>(atom_2),
                    static_cast<std::uint8_t>(atom_3), static_cast<std::uint8_t>(atom_4));
    }
}

/** Compare one torsion tuple's presence on each side, so a tuple only one matcher emitted names itself. */
static void compare_torsion_tuple_(diagnostic_reporter_t &reporter, std::string_view smarts, std::string const &smiles,
                                   torsion_tuple_t const &tuple, std::set<torsion_tuple_t> const &rdkit_tuples,
                                   std::set<torsion_tuple_t> const &native_tuples) {
    reporter
        .mismatch("csd_pattern", "smarts={} smiles={} torsion={},{},{},{}", smarts, smiles, std::get<0>(tuple),
                  std::get<1>(tuple), std::get<2>(tuple), std::get<3>(tuple))
        .field("emitted", rdkit_tuples.count(tuple) != 0, native_tuples.count(tuple) != 0, exactly_equal);
}

/** Run every native-parsed pattern against @p molecule and compare the emitted 4-tuples to `SubstructMatch` one at a
 *  time, aborting on the first pattern that disagrees. */
static void validate_patterns_for_molecule_(RDKit::ROMol const &molecule, sft::molecule_descriptor_t const &descriptor,
                                            std::string const &smiles,
                                            std::vector<sfp::csd_compiled_pattern_t> const &patterns,
                                            match_scratch_t &scratch, diagnostic_reporter_t &reporter) {
    std::uint8_t done_bonds_u8[sft::max_bonds_k] = {};
    for (auto const &compiled_pattern : patterns) {
        // A fresh stub that never marks a bond done: the dispatcher's `done_bonds` shortcut would give
        // only the first emission per bond, and every emission is wanted here.
        std::set<torsion_tuple_t> &native_tuples = scratch.native_tuples;
        native_tuples.clear();
        std::memset(done_bonds_u8, 0, sizeof(done_bonds_u8));
        sfp::match_linear_pattern(
            descriptor, compiled_pattern.native_form, done_bonds_u8,
            [&](std::uint8_t atom_1, std::uint8_t atom_2, std::uint8_t atom_3, std::uint8_t atom_4,
                std::uint32_t /*bond_index*/) {
                std::uint8_t canonical_1 = atom_1, canonical_2 = atom_2, canonical_3 = atom_3, canonical_4 = atom_4;
                if (canonical_1 > canonical_4 || (canonical_1 == canonical_4 && canonical_2 > canonical_3)) {
                    std::swap(canonical_1, canonical_4);
                    std::swap(canonical_2, canonical_3);
                }
                native_tuples.emplace(canonical_1, canonical_2, canonical_3, canonical_4);
                return false; // keep scanning — we want every emission
            });

        fill_rdkit_torsion_tuples_into_(molecule, compiled_pattern, scratch.matches, scratch.rdkit_tuples);
        std::set<torsion_tuple_t> const &rdkit_tuples = scratch.rdkit_tuples;
        for (auto const &tuple : rdkit_tuples)
            compare_torsion_tuple_(reporter, compiled_pattern.smarts, smiles, tuple, rdkit_tuples, native_tuples);
        for (auto const &tuple : native_tuples) {
            if (rdkit_tuples.count(tuple) != 0) continue; // already compared from RDKit's side of the union
            compare_torsion_tuple_(reporter, compiled_pattern.smarts, smiles, tuple, rdkit_tuples, native_tuples);
        }
        sf_verify_(native_tuples == rdkit_tuples, fmt::format("the native CSD pattern {} disagrees with RDKit on {}",
                                                              compiled_pattern.smarts, smiles));
    }
}

/** Compile every body in `recursive_bodies_k` on both sides. The native side takes the pattern compiler's own path;
 *  the RDKit side is built once here so no molecule in the sweep pays to reparse it. */
static void compile_recursive_cases_into_(std::vector<recursive_case_t> &cases) {
    cases.clear();
    cases.reserve(std::size(recursive_bodies_k));
    for (std::string_view const body : recursive_bodies_k) {
        recursive_case_t compiled {.body = body,
                                   .pool = {},
                                   .root = sfp::no_recursive_node_k,
                                   .query = std::unique_ptr<RDKit::ROMol const>(RDKit::SmartsToMol(std::string(body)))};
        compiled.pool.num_nodes = 0;
        sf_verify_(compiled.query != nullptr, fmt::format("RDKit could not compile the sub-query $({})", body));
        sf_verify_(sfp::compile_recursive_body(compiled.body, 0, compiled.pool, compiled.root),
                   fmt::format("the native compiler rejected the sub-query $({})", body));
        cases.push_back(std::move(compiled));
    }
}

/** Parse @p smiles, add hydrogens, and build the flat descriptor both matchers read, filling @p molecule and
 *  @p descriptor only when the molecule is admitted.
 *  @return `success_k` when the molecule joins the sweep, otherwise the status that refused it. */
static sft::conformer_status_t admit_molecule_(std::string const &smiles, std::unique_ptr<RDKit::RWMol> &molecule,
                                               sft::molecule_descriptor_t &descriptor) {
    // `SmilesToMol` throws on some malformed input instead of returning null, so it belongs inside the
    // guard beside `addHs`.
    try {
        molecule.reset(RDKit::SmilesToMol(smiles));
        if (molecule) RDKit::MolOps::addHs(*molecule);
    }
    catch (...) {
        molecule.reset();
    }
    if (!molecule) return sft::conformer_status_t::invalid_smiles_k;
    return sft::build_descriptor(*molecule, descriptor);
}

/** What one sweep over a SMILES corpus compared, and where the lines it did not compare went; `refused` is
 *  indexed by the status that kept each line out. */
struct predicate_tally_t {
    std::size_t validated = 0;
    std::size_t refused[sft::conformer_status_count_k] = {};
};

/** Narrate what the sweep covered, naming each refusal with the status that refused it. */
static void announce_corpus_tally_(predicate_tally_t const &tally) {
    fmt::print("NOTE csd_predicates molecules={}", tally.validated);
    for (std::size_t status = 0; status < sft::conformer_status_count_k; ++status) {
        if (tally.refused[status] == 0) continue; // a refusal nothing in this corpus reached
        fmt::print(" {}={}", sft::conformer_status_name(static_cast<sft::conformer_status_t>(status)),
                   tally.refused[status]);
    }
    fmt::println("");
}

/** Sweep a corpus: every sub-query at every atom, then every native pattern, molecule by molecule. Only the natively
 *  parsed @p patterns are compared, and @p max_molecules counts molecules validated rather than corpus lines read. */
static void validate_corpus_(std::vector<std::string> const &smiles_list, std::vector<recursive_case_t> const &cases,
                             std::vector<sfp::csd_compiled_pattern_t> const &patterns, std::size_t max_molecules,
                             predicate_tally_t &tally, diagnostic_reporter_t &reporter) {
    std::unique_ptr<RDKit::RWMol> molecule;
    sft::molecule_descriptor_t descriptor;
    match_scratch_t scratch;
    for (auto const &smiles : smiles_list) {
        sft::conformer_status_t const admitted = admit_molecule_(smiles, molecule, descriptor);
        if (admitted != sft::conformer_status_t::success_k) {
            ++tally.refused[static_cast<std::size_t>(admitted)];
            continue;
        }

        for (auto const &compiled : cases)
            validate_recursive_for_molecule_(compiled, *molecule, descriptor, smiles, scratch, reporter);

        // Full-pattern byte-equivalence pass across every native pattern.
        validate_patterns_for_molecule_(*molecule, descriptor, smiles, patterns, scratch, reporter);

        ++tally.validated;
        if (tally.validated >= max_molecules) break;
    }
}

} // namespace smallfield::tests

using sftest::recursive_case_t;

int main(int argc, char **argv) {
    fmt::println("{}: native CSD predicates and patterns vs RDKit SubstructMatch", test_name_k);

    auto const selection = sftest::select_corpus(argc, argv, 200, sftest::adversarial_corpus_path());
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", argv[0]);
        return 2;
    }

    std::vector<recursive_case_t> cases;
    sftest::compile_recursive_cases_into_(cases);

    std::vector<std::string> smiles_list;
    if (selection.knob == sftest::corpus_knob_t::named_k) {
        auto const outcome = sftest::load_smiles_into(selection.path, selection.max_molecules, smiles_list);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), selection.path);
        fmt::println("Corpus: {} ({} molecules)", selection.path, fmt::group_digits(smiles_list.size()));
    }
    else {
        // The builtin standards already exercise every predicate, so a fresh clone with no corpus to
        // hand stays green rather than failing in a way that would hide the real failures.
        smiles_list = sftest::standard_test_smiles();
        fmt::println("Corpus: builtin standards only; name a .smi file on the command line " //
                     "or in SMALLFIELD_TEST_CORPUS for a wider sweep");
    }

    // Every pattern `csd_patterns` hands out compiled natively, one that did not being dropped at load with
    // a diagnostic, so this count is the corpus size and a drop shows up as a shortfall against the rows.
    auto const &patterns = sfp::csd_patterns();
    fmt::println("{}: {} sub-queries and {} native CSD patterns per molecule", test_name_k,
                 fmt::group_digits(cases.size()), fmt::group_digits(patterns.size()));

    sftest::diagnostic_reporter_t reporter {"rdkit", "native"};
    sftest::predicate_tally_t tally;
    sftest::validate_corpus_(smiles_list, cases, patterns, selection.max_molecules, tally, reporter);
    sftest::announce_corpus_tally_(tally);
    reporter.summarize("csd_predicates");

    std::size_t const expected_min = std::min<std::size_t>(selection.max_molecules, 8);
    sf_verify_(tally.validated >= expected_min, "fewer molecules validated than the corpus floor — bad test data?");
    if (!reporter.passed())
        return sftest::report_fail(test_name_k, "{} mismatches over {} comparisons", reporter.mismatch_count(),
                                   reporter.comparison_count());
    return sftest::report_pass(test_name_k);
}
