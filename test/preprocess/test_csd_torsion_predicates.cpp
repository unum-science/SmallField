/**
 *  @file test/preprocess/test_csd_torsion_predicates.cpp
 *  @brief Cross-validate the native CSD torsion matcher against RDKit.
 *
 *  Two layers of correctness gate:
 *   - @b Sub-query-validation: every distinct `$(...)` body appearing in the CSD pattern corpus is compiled by
 *     `compile_recursive_body` and cross-checked against `RDKit::SmartsToMol` and `SubstructMatch` at every atom
 *     of every test molecule.
 *   - @b Full-pattern-validation: for every CSD torsion pattern the dispatcher marks `native==true`, the native
 *     matcher's per-torsion 4-tuple output is compared byte-for-byte to RDKit's `SubstructMatch` output, reduced to
 *     the set of `(atom_1,atom_2,atom_3,atom_4)` torsion indices and considered as an unordered multi-set modulo
 *     endpoint swap, since the central bond is symmetric.
 *
 *  Molecule corpus:
 *   - default: the builtin standards, which already reach every predicate, so a fresh clone stays green.
 *   - `argv[1]`, or `SMALLFIELD_TEST_CORPUS`: stream SMILES from a `.smi` file. A named path that does not
 *     open is a failure, never an empty run.
 *   - `argv[2]`, or `SMALLFIELD_TEST_MAX_MOLS`: cap molecules per corpus.
 *
 *  Divergence means the native implementation's semantics differ from RDKit — always debug the native side, never
 *  weaken the assertion.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::uint32_t`
#include <cstdio>  // `std::fflush`, `stdout`
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
#include <fmt/ranges.h> // `fmt::join`

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include "smallfield/preprocess/csd_torsion_dispatcher.hpp"
#include "smallfield/preprocess/csd_torsion_native.hpp"
#include "smallfield/preprocess/csd_torsion_predicates.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

#include "test/test_rdkit_reference_helpers.hpp"

namespace sfapi = smallfield::api;
namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_csd_torsion_predicates";

/**
 *  @brief Every distinct `$(...)` body the CSD torsion corpus writes, as it appears between `$(` and `)`.
 *
 *  Each is compiled through the same path the pattern compiler uses and cross-checked atom by atom
 *  against RDKit. The single characters carry the chemistry: `H` counts hydrogens, `X` the connection
 *  count, `r` ring membership, a lower-case element symbol aromaticity, and `@` ring closure.
 */
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

/**
 *  @brief One recursive sub-query body, compiled by both of the matchers under comparison.
 *
 *  Both sides compile once for the whole sweep rather than once per molecule, so what the corpus
 *  loop exercises is matching rather than parsing.
 */
struct recursive_case_t {
    std::string_view body;
    sfp::recursive_pool_t pool;
    std::uint8_t root;
    std::unique_ptr<RDKit::ROMol const> query;
};

/**
 *  @brief Comparison buffers the sweep owns, cleared at each use rather than rebuilt.
 *
 *  `matches` and `rdkit_roots` keep their capacity across molecules, so the inner loop over the
 *  pattern corpus stops growing them after the first few.
 */
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

/** Cross-validate one compiled sub-query against RDKit on one molecule. */
static void validate_recursive_for_molecule_(recursive_case_t const &compiled, RDKit::ROMol const &molecule,
                                             sft::molecule_descriptor_t const &descriptor, std::string const &smiles,
                                             match_scratch_t &scratch) {
    fill_rdkit_root_matches_into_(molecule, *compiled.query, scratch.matches, scratch.rdkit_roots);

    std::uint32_t const num_atoms = descriptor.num_atoms;
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        bool const native_hit = sfp::recursive_query_match(descriptor, atom, compiled.pool, compiled.root);
        bool const rdkit_hit = (scratch.rdkit_roots.count(atom) != 0);
        if (native_hit != rdkit_hit) {
            std::fflush(stdout);
            fmt::println(stderr,
                         "MISMATCH csd_subquery body=$({}) smiles={} atom={} field=root_match rdkit={} native={}",
                         compiled.body, smiles, atom, rdkit_hit ? 1 : 0, native_hit ? 1 : 0);
            fmt::println(
                stderr, "NOTE csd_subquery atom={} element={} aromatic={} degree={} num_hydrogens={} hybridization={}",
                atom, descriptor.element[atom], sft::atom_has_flag(descriptor, atom, sft::atom_flag_aromatic_k) ? 1 : 0,
                descriptor.degree[atom], descriptor.num_hydrogens[atom], descriptor.hybridization[atom]);
            sf_verify_(false, "sub-query mismatch vs RDKit");
        }
    }
}

/**
 *  @brief Reduce a set of RDKit `MatchVectType` hits to the set of torsion-atom 4-tuples identified by the
 *  pattern's `:1 :2 :3 :4` labels. Each tuple is normalised so swapping endpoints (atom_1↔atom_4,
 *  atom_2↔atom_3) yields the same canonical form — the native matcher and RDKit may emit the two
 *  orientations in different order, but the torsion semantics is symmetric.
 */
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

/**
 *  @brief Run every native-parsed pattern against @p molecule and compare the emitted 4-tuple set to
 *  RDKit's SubstructMatch. Returns the number of patterns fully agreed; aborts on any disagreement.
 */
static std::size_t validate_patterns_for_molecule_(RDKit::ROMol const &molecule,
                                                   sft::molecule_descriptor_t const &descriptor,
                                                   std::string const &smiles,
                                                   std::vector<sfp::csd_compiled_pattern_t> const &patterns,
                                                   match_scratch_t &scratch) {
    std::uint8_t done_bonds_u8[sft::max_bonds_k] = {};
    std::size_t ok = 0;
    for (auto const &compiled_pattern : patterns) {
        // Collect native matches. We cannot use the dispatcher's `done_bonds`
        // shortcut here because we want @b every emission, not just the first
        // per bond; use a fresh stub that never marks the bond done.
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

        std::set<torsion_tuple_t> const &rdkit_tuples = scratch.rdkit_tuples;
        fill_rdkit_torsion_tuples_into_(molecule, compiled_pattern, scratch.matches, scratch.rdkit_tuples);
        if (native_tuples != rdkit_tuples) {
            std::fflush(stdout);
            fmt::println(stderr,
                         "MISMATCH csd_pattern smarts={} smiles={} field=torsion_tuple_count rdkit={} native={}",
                         compiled_pattern.smarts, smiles, rdkit_tuples.size(), native_tuples.size());
            fmt::println(stderr, "NOTE csd_pattern smarts={} rdkit_tuples=[{}] native_tuples=[{}]",
                         compiled_pattern.smarts, fmt::join(rdkit_tuples, " "), fmt::join(native_tuples, " "));
            sf_verify_(false, "pattern mismatch vs RDKit");
        }
        ++ok;
    }
    return ok;
}

/**
 *  @brief Compile every body in `recursive_bodies_k` on both sides of the comparison.
 *
 *  The native side goes through the path the pattern compiler itself takes; the RDKit side is
 *  built once here so no molecule in the sweep pays to reparse it.
 */
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

/**
 *  @brief Announce the pattern corpus.
 *
 *  Every pattern `csd_patterns` hands out compiled natively — one that did not is dropped at load
 *  with a diagnostic — so this count is the corpus size, and a drop shows up here as a shortfall
 *  against the 470 rows the table carries.
 */
static std::size_t announce_pattern_corpus_(std::vector<sfp::csd_compiled_pattern_t> const &patterns) {
    fmt::println("CSD patterns: {} native", fmt::group_digits(patterns.size()));
    return patterns.size();
}

/** Why a corpus molecule was left out of the sweep, or that it was not. */
enum class molecule_admission_t {
    /** Parsed, hydrogenated, and inside every descriptor capacity. */
    admitted_k,
    /** `SmilesToMol` refused it, or `addHs` threw on it. */
    unparsable_k,
    /** Beyond `max_atoms_per_molecule_k` or `max_bonds_k`. */
    exceeds_atom_or_bond_capacity_k,
    /** More SSSR rings than `max_rings_k`. */
    exceeds_ring_capacity_k,
    /** `build_descriptor` refused it. */
    no_descriptor_k,
};

/**
 *  @brief Parse @p smiles, add hydrogens, and build the flat descriptor both matchers read.
 *
 *  @param[in] smiles Molecule to rebuild, as it appears in the corpus.
 *  @param[out] molecule Reset to the parsed, hydrogenated molecule when it is admitted.
 *  @param[out] descriptor Filled when the molecule is admitted, untouched otherwise.
 *  @return Whether the molecule joins the sweep, and why it does not when it does not.
 */
static molecule_admission_t admit_molecule_(std::string const &smiles, std::unique_ptr<RDKit::RWMol> &molecule,
                                            sft::molecule_descriptor_t &descriptor) {
    molecule.reset(RDKit::SmilesToMol(smiles));
    if (!molecule) return molecule_admission_t::unparsable_k;
    try {
        RDKit::MolOps::addHs(*molecule);
    }
    catch (...) {
        return molecule_admission_t::unparsable_k;
    }
    if (molecule->getNumAtoms() > sft::max_atoms_per_molecule_k || molecule->getNumBonds() > sft::max_bonds_k)
        return molecule_admission_t::exceeds_atom_or_bond_capacity_k;

    // Ring-based pattern constraints silently under-match past the descriptor's ring capacity,
    // because `atom_ring_bitmap` and `smallest_ring` are only populated for rings [0, max_rings_k).
    // Pathological polycyclic cages -- the fused-cyclopropane P+ scaffold in USearch, say -- are the
    // only inputs that reach it, and the production preprocess rejects them upstream via the
    // 128-atom cap for everything except spiro cages.
    auto const *ring_info = molecule->getRingInfo();
    if (ring_info != nullptr && ring_info->isInitialized() && ring_info->atomRings().size() > sft::max_rings_k)
        return molecule_admission_t::exceeds_ring_capacity_k;

    if (sft::build_descriptor(*molecule, descriptor) != sfapi::conformer_status_t::success_k)
        return molecule_admission_t::no_descriptor_k;
    return molecule_admission_t::admitted_k;
}

/** What one sweep over a SMILES corpus compared. */
struct predicate_tally_t {
    std::size_t validated = 0;
    std::size_t skipped = 0;
    std::size_t sub_query_checks = 0;
    std::size_t pattern_checks = 0;
};

/**
 *  @brief Sweep a corpus: every sub-query at every atom, then every native pattern, molecule by molecule.
 *
 *  @param[in] smiles_list Corpus to walk, in order.
 *  @param[in] cases Compiled sub-queries to cross-check.
 *  @param[in] patterns Compiled CSD torsion patterns; only the natively parsed ones are compared.
 *  @param[in] max_molecules Stop once this many molecules have been validated.
 *  @param[inout] tally Counters advanced by every molecule.
 */
static void validate_corpus_(std::vector<std::string> const &smiles_list, std::vector<recursive_case_t> const &cases,
                             std::vector<sfp::csd_compiled_pattern_t> const &patterns, std::size_t max_molecules,
                             predicate_tally_t &tally) {
    std::unique_ptr<RDKit::RWMol> molecule;
    sft::molecule_descriptor_t descriptor;
    match_scratch_t scratch;
    for (auto const &smiles : smiles_list) {
        if (admit_molecule_(smiles, molecule, descriptor) != molecule_admission_t::admitted_k) {
            ++tally.skipped;
            continue;
        }

        for (auto const &compiled : cases) {
            validate_recursive_for_molecule_(compiled, *molecule, descriptor, smiles, scratch);
            ++tally.sub_query_checks;
        }

        // Full-pattern byte-equivalence pass across every native pattern.
        tally.pattern_checks += validate_patterns_for_molecule_(*molecule, descriptor, smiles, patterns, scratch);

        ++tally.validated;
        if (tally.validated >= max_molecules) break;
    }
}

} // namespace smallfield::tests

using sftest::recursive_case_t;

int main(int argc, char **argv) {
    fmt::println("{}: native CSD predicates and patterns vs RDKit SubstructMatch", test_name_k);

    auto const selection = sftest::select_corpus(argc, argv, 200);
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
        // The builtin standards already exercise every predicate, so a fresh clone with no corpus
        // to hand stays green rather than reporting a failure that would hide the real ones.
        smiles_list = sftest::standard_test_smiles();
        fmt::println("Corpus: builtin standards only; name a .smi file on the command line " //
                     "or in SMALLFIELD_TEST_CORPUS for a wider sweep");
    }

    auto const &patterns = sfp::csd_patterns();
    std::size_t const native_patterns = sftest::announce_pattern_corpus_(patterns);

    sftest::predicate_tally_t tally;
    sftest::validate_corpus_(smiles_list, cases, patterns, selection.max_molecules, tally);

    fmt::println("Validated {} molecules (skipped {}), {} sub-query checks", fmt::group_digits(tally.validated),
                 fmt::group_digits(tally.skipped), fmt::group_digits(tally.sub_query_checks));
    fmt::println("Pattern checks: {} (= {} molecules x {} native patterns)", fmt::group_digits(tally.pattern_checks),
                 fmt::group_digits(tally.validated), fmt::group_digits(native_patterns));
    std::size_t const expected_min = std::min<std::size_t>(selection.max_molecules, 8);
    sf_verify_(tally.validated >= expected_min, "fewer molecules validated than the corpus floor — bad test data?");
    return sftest::report_pass(test_name_k);
}
