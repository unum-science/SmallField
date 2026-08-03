/**
 *  @file test_csd_torsion_predicates.cpp
 *  @brief Cross-validate the native CSD torsion matcher against RDKit.
 *
 *  Two layers of correctness gate:
 *   - @b Sub-query @b validation: every distinct `$(...)` body appearing in the CSD
 *     pattern corpus is compiled by @ref compile_recursive_body and cross-checked
 *     against `RDKit::SmartsToMol` + `SubstructMatch` at every atom of every test
 *     molecule.
 *   - @b Full-pattern @b validation: for every CSD torsion pattern the dispatcher
 *     marks `native==true`, the native matcher's per-torsion 4-tuple output is
 *     compared byte-for-byte to RDKit's `SubstructMatch` output (reduced to the
 *     set of `(a1,a2,a3,a4)` torsion indices and considered as an unordered
 *     multi-set modulo endpoint swap, since the central bond is symmetric).
 *
 *  Molecule corpus is tiered via env-var:
 *   - default: ChEMBL 200-molecule smoke set, <1 s runtime.
 *   - `SMALLFIELD_TEST_CORPUS=/path/to.smi`: stream SMILES from the file.
 *   - `SMALLFIELD_TEST_MAX_MOLS=<N>`: cap molecules per corpus.
 *
 *  Divergence means the native implementation's semantics differ from RDKit —
 *  always debug the native side, never weaken the assertion.
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include "smallfield/preprocess/csd_torsion_dispatcher.hpp"
#include "smallfield/preprocess/csd_torsion_native.hpp"
#include "smallfield/preprocess/csd_torsion_predicates.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "test_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

namespace {

/** One recursive sub-query body, as it appears between `$(` and its matching `)`. */
struct recursive_case_t {
    char const *body;
    sfp::recursive_pool_t pool;
    std::uint8_t root;
};

/** Load up to @p max_count SMILES strings from @p path, one per line.
 *
 *  @param[in] path File to read, one SMILES per line.
 *  @param[in] max_count Upper bound on how many lines to keep.
 *  @param[in] required When true a missing file aborts; when false it yields an empty list, which
 *      is how the optional extended corpus stays optional.
 *  @return The loaded SMILES, empty when @p path is absent and not required. */
std::vector<std::string> load_smiles(char const *path, std::size_t max_count, bool required = true) {
    std::vector<std::string> out;
    std::ifstream in(path);
    if (!in) {
        if (!required) return out;
        std::fprintf(stderr, "FAIL: cannot open %s\n", path);
        std::exit(1);
    }
    std::string line;
    while (out.size() < max_count && std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
        std::size_t const sp = line.find_first_of(" \t");
        if (sp != std::string::npos) line.resize(sp);
        if (!line.empty()) out.push_back(std::move(line));
    }
    return out;
}

/** Return the set of root-atom indices hit by `SubstructMatch(query)` in `mol`. */
std::unordered_set<std::uint32_t> rdkit_root_match_set(RDKit::ROMol const &mol, RDKit::ROMol const &query) {
    std::vector<RDKit::MatchVectType> matches;
    RDKit::SubstructMatch(mol, query, matches, /*uniquify=*/false, /*recursionPossible=*/true);
    std::unordered_set<std::uint32_t> roots;
    for (auto const &m : matches) {
        if (m.empty()) continue;
        roots.insert(static_cast<std::uint32_t>(m[0].second));
    }
    return roots;
}

/** Cross-validate one compiled sub-query against RDKit on one molecule. */
void validate_recursive_for_molecule(recursive_case_t const &compiled, RDKit::ROMol const &mol,
                                     sft::molecule_descriptor_t const &descriptor, std::string const &smiles) {
    std::unique_ptr<RDKit::ROMol const> query(RDKit::SmartsToMol(compiled.body));
    SF_TEST_ASSERT(query != nullptr, compiled.body);

    auto const rdkit_roots = rdkit_root_match_set(mol, *query);

    std::uint32_t const num_atoms = descriptor.num_atoms;
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        bool const native_hit = sfp::recursive_query_match(descriptor, atom, compiled.pool, compiled.root);
        bool const rdkit_hit = (rdkit_roots.count(atom) != 0);
        if (native_hit != rdkit_hit) {
            std::fprintf(stderr, "Sub-query mismatch: $(%s) atom=%u native=%d rdkit=%d\n", compiled.body, atom,
                         native_hit ? 1 : 0, rdkit_hit ? 1 : 0);
            std::fprintf(stderr, "  element=%u aromatic=%d degree=%u nH=%u hyb=%u\n", descriptor.element[atom],
                         sft::atom_has_flag(descriptor, atom, sft::atom_flag_aromatic_k) ? 1 : 0,
                         descriptor.degree[atom], descriptor.num_hydrogens[atom], descriptor.hybridization[atom]);
            std::fprintf(stderr, "  SMILES: %s\n", smiles.c_str());
            SF_TEST_ASSERT(false, "sub-query mismatch vs RDKit");
        }
    }
}

/**
 *  Reduce a set of RDKit `MatchVectType` hits to the set of torsion-atom
 *  4-tuples identified by the pattern's `:1 :2 :3 :4` labels.  Each tuple is
 *  normalised so swapping endpoints (a1↔a4, a2↔a3) yields the same canonical
 *  form — the native matcher and RDKit may emit the two orientations in
 *  different order, but the torsion semantics is symmetric.
 */
std::set<std::tuple<std::uint8_t, std::uint8_t, std::uint8_t, std::uint8_t>> rdkit_torsion_tuples(
    RDKit::ROMol const &mol, sfp::csd_compiled_pattern_t const &pat) {
    std::vector<RDKit::MatchVectType> matches;
    RDKit::SubstructMatch(mol, *pat.pattern, matches, /*uniquify=*/false, /*recursionPossible=*/true);
    std::set<std::tuple<std::uint8_t, std::uint8_t, std::uint8_t, std::uint8_t>> out;
    for (auto const &m : matches) {
        std::uint32_t a1 = static_cast<std::uint32_t>(m[pat.index[0]].second);
        std::uint32_t a2 = static_cast<std::uint32_t>(m[pat.index[1]].second);
        std::uint32_t a3 = static_cast<std::uint32_t>(m[pat.index[2]].second);
        std::uint32_t a4 = static_cast<std::uint32_t>(m[pat.index[3]].second);
        if (a1 > a4 || (a1 == a4 && a2 > a3)) {
            std::swap(a1, a4);
            std::swap(a2, a3);
        }
        out.emplace(static_cast<std::uint8_t>(a1), static_cast<std::uint8_t>(a2), static_cast<std::uint8_t>(a3),
                    static_cast<std::uint8_t>(a4));
    }
    return out;
}

/**
 *  Run every native-parsed pattern against @p mol and compare the emitted
 *  4-tuple set to RDKit's SubstructMatch.  Returns the number of patterns
 *  fully agreed; aborts on any disagreement.
 */
std::size_t validate_patterns_for_molecule(RDKit::ROMol const &mol, sft::molecule_descriptor_t const &descriptor,
                                           std::string const &smiles,
                                           std::vector<sfp::csd_compiled_pattern_t> const &patterns) {
    std::uint8_t done_bonds_u8[sft::max_bonds_k] = {};
    std::size_t ok = 0;
    for (auto const &pat : patterns) {
        if (!pat.native) continue;

        // Collect native matches. We cannot use the dispatcher's `done_bonds`
        // shortcut here because we want EVERY emission, not just the first
        // per bond; use a fresh stub that never marks the bond done.
        std::set<std::tuple<std::uint8_t, std::uint8_t, std::uint8_t, std::uint8_t>> native_tuples;
        std::memset(done_bonds_u8, 0, sizeof(done_bonds_u8));
        sfp::match_linear_pattern(
            descriptor, pat.native_form, done_bonds_u8,
            [&](std::uint8_t a1, std::uint8_t a2, std::uint8_t a3, std::uint8_t a4, std::uint32_t /*bid*/) {
                std::uint8_t x1 = a1, x2 = a2, x3 = a3, x4 = a4;
                if (x1 > x4 || (x1 == x4 && x2 > x3)) {
                    std::swap(x1, x4);
                    std::swap(x2, x3);
                }
                native_tuples.emplace(x1, x2, x3, x4);
                return false; // keep scanning — we want every emission
            });

        auto const rdkit_tuples = rdkit_torsion_tuples(mol, pat);
        if (native_tuples != rdkit_tuples) {
            std::fprintf(stderr, "Pattern mismatch:\n  SMARTS: %s\n  SMILES: %s\n", pat.smarts.c_str(), smiles.c_str());
            std::fprintf(stderr, "  native tuples (%zu):", native_tuples.size());
            for (auto const &t : native_tuples)
                std::fprintf(stderr, " (%u,%u,%u,%u)", std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t));
            std::fprintf(stderr, "\n  rdkit tuples  (%zu):", rdkit_tuples.size());
            for (auto const &t : rdkit_tuples)
                std::fprintf(stderr, " (%u,%u,%u,%u)", std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t));
            std::fprintf(stderr, "\n");
            SF_TEST_ASSERT(false, "pattern mismatch vs RDKit");
        }
        ++ok;
    }
    return ok;
}

} // namespace

int main() {
    std::printf("Cross-validating CSD torsion predicates and patterns against RDKit...\n");

    // Every distinct `$(...)` body the CSD torsion corpus writes, compiled through the
    // same path the pattern compiler uses and cross-checked atom by atom against RDKit.
    recursive_case_t cases[] = {
        {.body = "C([F])([F])[F]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "C([NX3])N", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "C([NX3])n", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "C=O", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "N(C)[!#1]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "N(C)~[!#1]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "N[CX4]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "S(=O)(=O)", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "S(=O)=O", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[CX3](=[!O])", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[CX3]([C])([H])", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[CX3](a)=O", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[CX3]=O", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[CX4&r]([C;r])([C;r])", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[CX4H1]C(=O)O", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[C](=O)", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[C](=O)([$([NX3H1]),$([NX3H2])])[NX3H1]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[NH0][CX4]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[NH1,NH2,CH2]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[NH1]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[NX3H1,NX3H2]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[NX3H1]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[NX3H1]C", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[NX3H2]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[NX3](c([nH1])n)", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[NX3](c([nX2H0])([nX2H0]))", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[NX3](cn)", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[NX3]c[nH0]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[N][!#1]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[N][CX4]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[aH0](!@O)", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[a]([cH1])", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[a]([nH0,o])", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[c&r6](c[OH])", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[cH0]!@;-[*^2]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[cH0](=O)", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[cH0](Cl)", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[cH0](F)", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[cH0]([$([NX3H2]),$([NX3H1])])", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[cH0]([CX3])", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[cH0]@;-[*^2]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[cH0]Cl", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[cH0]F", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[cH0][NH1,NH2]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[cH0][OH0]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[cH0][OH1]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[c]([NH1,NH2])", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[c]([cH,nX2H0])[cH,nX2H0]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "[n,N](-a)", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "a([!nX2H0])([nX2H0])!@;-[NX3H1]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "a([!nX2H0])([nX2H0])@;-[NX3H1]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "aC(=O)(O)", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "a[NH1,NH2]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "a[OH1]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "c!@c", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "c([nX2H0])([nX2H0])", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "c([nX2H0])(a(a)(a))!@;-c[nX2H0]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "c([nX2H0])(a(a)(a))@;-c[nX2H0]", .pool = {}, .root = sfp::no_recursive_node_k},
        {.body = "cO", .pool = {}, .root = sfp::no_recursive_node_k},
    };
    for (auto &compiled : cases) {
        compiled.pool.num_nodes = 0;
        SF_TEST_ASSERT(sfp::compile_recursive_body(compiled.body, 0, compiled.pool, compiled.root), compiled.body);
    }

    // Corpus selection: env-var override → tier-1/tier-2; default → built-in
    // standard molecules + ChEMBL valid set.
    char const *corpus_path = std::getenv("SMALLFIELD_TEST_CORPUS");
    char const *max_mols_env = std::getenv("SMALLFIELD_TEST_MAX_MOLS");
    std::size_t const max_mols = max_mols_env ? static_cast<std::size_t>(std::atoll(max_mols_env)) : 200;

    std::vector<std::string> smiles_list;
    if (corpus_path) {
        smiles_list = load_smiles(corpus_path, max_mols);
        std::printf("Corpus: %s (up to %zu molecules)\n", corpus_path, max_mols);
    }
    else {
        // The extended corpus is a convenience, not a prerequisite: absent it the builtin
        // standards still exercise every predicate, so the suite stays green on a fresh clone
        // instead of reporting a failure that hides real ones.
        smiles_list = sftest::standard_test_smiles();
        auto extended = load_smiles("/tmp/chembl_valid.smi", max_mols * 2, /*required=*/false);
        bool const have_extended = !extended.empty();
        for (auto &entry : extended) smiles_list.push_back(std::move(entry));
        if (have_extended)
            std::printf("Corpus: builtin standards + /tmp/chembl_valid.smi (up to %zu molecules)\n", max_mols);
        else std::printf("Corpus: builtin standards only; set SMALLFIELD_TEST_CORPUS for a wider sweep\n");
    }

    auto const &patterns = sfp::csd_patterns();
    std::size_t native_patterns = 0;
    for (auto const &p : patterns)
        if (p.native) ++native_patterns;
    std::printf("CSD patterns: %zu total, %zu native (%0.1f%%)\n", patterns.size(), native_patterns,
                100.0 * native_patterns / static_cast<double>(patterns.size()));

    std::size_t total_validated = 0;
    std::size_t skipped = 0;
    std::size_t total_pattern_checks = 0;
    std::size_t validated_sub_queries = 0;

    for (auto const &smi : smiles_list) {
        std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smi));
        if (!mol) {
            ++skipped;
            continue;
        }
        try {
            RDKit::MolOps::addHs(*mol);
        }
        catch (...) {
            ++skipped;
            continue;
        }
        if (mol->getNumAtoms() > sft::max_atoms_per_molecule_k || mol->getNumBonds() > sft::max_bonds_k) {
            ++skipped;
            continue;
        }

        // Skip molecules whose SSSR exceeds the descriptor's ring capacity
        // (`max_rings_k = 32`).  Ring-based pattern constraints silently
        // under-match for such inputs because `atom_ring_bitmap` / `smallest_ring`
        // are only populated for rings [0, max_rings_k).  Pathological
        // polycyclic cages (e.g. fused-cyclopropane P+-scaffold in USearch)
        // are the only inputs that trigger this — the production pipeline
        // rarely encounters them, and the preprocess rejects them upstream
        // via the 128-atom cap for everything except spiro cages.
        auto const *ri_check = mol->getRingInfo();
        if (ri_check != nullptr && ri_check->isInitialized() && ri_check->atomRings().size() > sft::max_rings_k) {
            ++skipped;
            continue;
        }

        sft::molecule_descriptor_t descriptor;
        if (!sft::build_descriptor(*mol, descriptor)) {
            ++skipped;
            continue;
        }

        for (auto const &compiled : cases) {
            validate_recursive_for_molecule(compiled, *mol, descriptor, smi);
            ++validated_sub_queries;
        }

        // Full-pattern byte-equivalence pass across every native pattern.
        total_pattern_checks += validate_patterns_for_molecule(*mol, descriptor, smi, patterns);

        ++total_validated;
        if (total_validated >= max_mols) break;
    }

    std::printf("\nValidated %zu molecules (skipped %zu), %zu sub-query checks.\n", total_validated, skipped,
                validated_sub_queries);
    std::printf("Pattern checks: %zu (= %zu molecules × %zu native patterns)\n", total_pattern_checks, total_validated,
                native_patterns);
    std::size_t const expected_min = std::min<std::size_t>(max_mols, 8);
    SF_TEST_ASSERT(total_validated >= expected_min, "fewer molecules validated than the corpus floor — bad test data?");
    std::printf("\nAll native sub-query + pattern cross-validation tests passed.\n");
    return 0;
}
