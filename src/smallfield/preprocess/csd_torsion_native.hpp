/**
 *  @file csd_torsion_native.hpp
 *  @brief Stack-only native matcher for simple CSD torsion SMARTS patterns.
 *
 *  Handles 4-atom torsion paths like "[C]-[C]-[N]-[*]" entirely on the stack, with zero heap allocations in the
 *  hot path. The four atoms are bound by their `:1`-`:4` map labels, so a labelled atom inside a branch —
 *  `[a:1][c:2]!@;-[C:3](=[N:4])(-N)` — binds like any other. Atom-level OR (`[cH,nX2H0]`), bond-level OR
 *  (`@;=,:`), ring-size ranges (`r{5-8}`), ring-bond connectivity (`x3`), a ring closure that walks the four
 *  labelled atoms back around a cycle, and recursive `$(...)` bodies compiled into the bounded sub-query pool
 *  of `csd_torsion_predicates`.hpp are all handled natively — the whole 470-pattern CSD library compiles.
 *
 *  Why this matters: 192-thread OpenMP preprocessing saturates at 32 threads today because every SubstructMatch
 *  call allocates and destroys several std::vector objects plus VF2 state arrays. Removing those allocations
 *  restores linear thread scaling.
 *
 *  Correctness: identical matches to RDKit for every pattern. Validated by a per-pattern per-atom
 *  byte-equivalence test against RDKit's `SubstructMatch` (see `test_csd_torsion_predicates.cpp`).
 */

#pragma once

#include <cctype>  // `std::isdigit`, `std::isalpha`
#include <cstdint> // `std::uint8_t`, `std::int8_t`

#include <limits>      // `std::numeric_limits`
#include <string_view> // `std::string_view`

#include "smallfield/module_fwd.hpp"
#include "smallfield/preprocess/csd_torsion_predicates.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::preprocess {
using smallfield::types::atom_flag_aromatic_k;
using smallfield::types::atom_flag_in_ring_k;
using smallfield::types::bond_flag_aromatic_k;
using smallfield::types::bond_flag_in_ring_k;
using smallfield::types::hybridization_sp2_k;
using smallfield::types::hybridization_sp3_k;
using smallfield::types::hybridization_sp3d2_k;
using smallfield::types::hybridization_sp3d_k;
using smallfield::types::hybridization_sp_k;
using smallfield::types::molecule_descriptor_t;

/**
 *  @brief Outcome of folding one SMARTS primitive run into an `atom_disjunct_t`. A
 *  contradiction between two primitives of the same conjunction — `r3` next to
 *  `r{5-6}`, two different `X<n>` — is not a parse failure: the disjunct simply
 *  matches nothing and is dropped, while the rest of the disjunction stands.
 *  Syntax the compiler cannot represent is a different answer entirely and sends
 *  the whole pattern to RDKit.
 */
enum term_parse_t : std::uint8_t {
    /** Primitives folded in, the disjunct is satisfiable. */
    term_parsed_k = 0,
    /** Two primitives contradict, drop this disjunct. */
    term_unsatisfiable_k = 1,
    /** Syntax outside the compiler's vocabulary. */
    term_unsupported_k = 2,
};

/**
 *  @brief Per-atom branch-neighbor constraint. Represents an inline neighbour specified
 *  in SMARTS via `]([…])` — e.g. `[c:2]([cH1])` = carbon with an aromatic-CH
 *  neighbour. The branch body compiles into the same sub-query pool the `$(...)`
 *  bodies use, so `(cO)` and `([cH0]Cl)` walk two hops with no extra machinery,
 *  and the bond operator that opens the branch stays here because it applies to
 *  the edge from the torsion atom rather than to anything inside the body.
 */
struct simple_neighbor_constraint_t {
    bond_disjunct_t bond_disjuncts[max_disjuncts_k];
    /** 1 for an explicit op, 2 for the implicit single-or-aromatic default. */
    std::uint8_t num_bond_disjuncts;
    /** Sub-query node the neighbour atom must satisfy. */
    std::uint8_t node;
    /** 1 if this slot carries a real constraint. */
    std::uint8_t active;
    std::uint8_t padding[5];
};

/**
 *  @brief Per-atom constraint extracted from SMARTS. All fields stack-allocated.
 *  OR-of-element/of-primitives (`[N,O,S]`, `[cH1,n]`) is expressed via a short
 *  `atom_disjunct_t` list, each disjunct carrying its own recursive guard so
 *  a spec like `[$([NX3H1]),$([NX3H2])]` distributes correctly.
 */
struct simple_atom_constraint_t {
    atom_disjunct_t disjuncts[max_disjuncts_k];
    /** 1 for plain spec, up to 4 for `[a,b,c,d]`. */
    std::uint8_t num_disjuncts;
    /** 0, 1, or 2 inline `(...)` neighbour branches. */
    std::uint8_t num_neighbor_branches;
    std::uint8_t padding[6];
    /** Active slots defined by num_neighbor_branches. */
    simple_neighbor_constraint_t neighbor_branch[2];
};

/** Per-bond constraint extracted from SMARTS. */
struct simple_bond_constraint_t {
    bond_disjunct_t disjuncts[max_disjuncts_k];
    /** 1 for plain op, up to 4 for `a,b,c,d`. */
    std::uint8_t num_disjuncts;
    std::uint8_t padding[3];
};

/**
 *  @brief Longest run of unlabelled atoms a ring-closing pattern may place between `:4`
 *  and `:1`. The widest ring in the CSD small-ring library is the eight-membered
 *  `[CX4;r{5-8}:1]1@[CX4;r{5-8}:2]@;-[CX4;x3;r{5-8}:3]@[!#1;r{5-8}:4]@[!#1]@[!#1]@[!#1]@[!#1]1`.
 */
inline constexpr std::uint8_t max_ring_closure_atoms_k = 4;

/**
 *  @brief One unlabelled atom on the arc that closes a ring pattern back onto `:1`,
 *  together with the edge reaching it from the preceding arc atom. Arc atoms
 *  carry no inline branches in the CSD library, so a bare disjunct list is the
 *  whole constraint.
 */
struct simple_ring_atom_t {
    atom_disjunct_t atom_disjuncts[max_disjuncts_k];
    /** Edge from the previous arc atom, or from `:4`. */
    bond_disjunct_t bond_disjuncts[max_disjuncts_k];
    std::uint8_t num_atom_disjuncts;
    std::uint8_t num_bond_disjuncts;
    std::uint8_t padding[6];
};

/**
 *  @brief The arc a ring-closing pattern draws from `:4` back to `:1`, walked in that
 *  direction. With `num_atoms` zero the two labelled atoms are bonded
 *  directly, which is how the four-membered `[…:1]1@[…:2]@;-[…:3]@[…:4]1` reads.
 */
struct simple_ring_closure_t {
    simple_ring_atom_t atom[max_ring_closure_atoms_k];
    /** Edge from the last arc atom onto `:1`. */
    simple_bond_constraint_t closing_bond;
    std::uint8_t num_atoms;
    /** 1 when the pattern closes a ring at all. */
    std::uint8_t active;
    std::uint8_t padding[6];
};

/**
 *  @brief Fully parsed linear 4-atom torsion pattern: [atom0]-bond01-[atom1]-bond12-[atom2]-bond23-[atom3].
 *  Labels :1,:2,:3,:4 in SMARTS correspond to atom[0],atom[1],atom[2],atom[3] here.
 *  Central bond is bond[1] (between atom[1] and atom[2]).
 */
struct simple_torsion_pattern_t {
    simple_atom_constraint_t atom[4];
    simple_bond_constraint_t bond[3];
    simple_ring_closure_t closure;
    /** Every `$(...)` body and inline branch body the pattern compiled to. */
    recursive_pool_t pool;
    std::int8_t signs[6];
    float V[6];
};

/**
 *  @brief Borrowed handle on the four atom constraints, three bond constraints, ring closure and
 *  sub-query nodes of one pattern. The matcher reads a pattern only through this handle, so a
 *  host `simple_torsion_pattern_t` that owns everything inline and a device pattern whose
 *  closure and nodes live in shared arenas both drive the same match. A null `closure` means
 *  the pattern closes no ring, which is how a device table drops the 500 bytes an inactive
 *  closure would otherwise cost in every one of the 470 entries.
 */
struct csd_pattern_view_t {
    /** Four labelled atoms, `:1` through `:4`. */
    simple_atom_constraint_t const *atom = nullptr;
    /** Three chain bonds, central bond at index 1. */
    simple_bond_constraint_t const *bond = nullptr;
    /** Arc back from `:4` onto `:1`; null when absent. */
    simple_ring_closure_t const *closure = nullptr;
    /** Sub-query nodes the constraints index into. */
    recursive_pool_view_t pool;

    csd_pattern_view_t() = default;
    SF_HOST_DEVICE_ csd_pattern_view_t(simple_atom_constraint_t const *atom_constraints,
                                       simple_bond_constraint_t const *bond_constraints,
                                       simple_ring_closure_t const *ring_closure, recursive_pool_view_t node_pool)
        : atom(atom_constraints), bond(bond_constraints), closure(ring_closure), pool(node_pool) {}
    SF_HOST_DEVICE_ csd_pattern_view_t(simple_torsion_pattern_t const &pattern)
        : atom(pattern.atom), bond(pattern.bond), closure(&pattern.closure), pool(pattern.pool) {}
};

/** Lookup atomic number from a 1-2 letter symbol. Returns 0 if unknown. */
inline std::uint8_t symbol_to_element(std::string_view sym) {
    // Minimal periodic table for the elements that appear in CSD torsion patterns.
    // Both aliphatic (uppercase-first) and aromatic (lowercase) variants map to the same number.
    struct entry_t {
        char const *sym;
        std::uint8_t num;
    };
    constexpr entry_t table[] = {
        {"H", 1},  {"He", 2}, {"Li", 3},  {"Be", 4},  {"B", 5},   {"C", 6},   {"N", 7},
        {"O", 8},  {"F", 9},  {"Ne", 10}, {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14},
        {"P", 15}, {"S", 16}, {"Cl", 17}, {"Ar", 18}, {"K", 19},  {"Ca", 20}, {"Br", 35},
        {"I", 53}, {"b", 5},  {"c", 6},   {"n", 7},   {"o", 8},   {"p", 15},  {"s", 16},
    };
    for (auto const &e : table)
        if (sym == e.sym) return e.num;
    return 0;
}

/**
 *  @brief Maximum number of atoms — chain atoms plus map-labelled inline branches —
 *  that the SMARTS scanner records before declaring a pattern too wide for the
 *  4-atom matcher.
 */
inline constexpr std::size_t max_pattern_atoms_k = 8;

/** Maximum inline `(...)` branches the scanner records for a single atom. */
inline constexpr std::size_t max_atom_branches_k = 3;

/**
 *  @brief Sentinel atom index: the parent of a pattern root, and an unbound torsion
 *  slot while the four labels are still being resolved.
 */
inline constexpr std::uint8_t no_atom_k = 0xFF;

/**
 *  @brief One atom of a scanned SMARTS pattern together with the edge to the atom it
 *  hangs off. `parse_linear_torsion` scans a pattern into this shape first
 *  and binds the torsion atoms afterwards by their `:n` map label, so a labelled
 *  atom may sit anywhere in the text as long as the labels form a path.
 */
struct scanned_atom_t {
    /** Atom text with its brackets, reused as branch text. */
    std::string_view text;
    /** Bracket interior, empty for a shorthand element. */
    std::string_view spec;
    /** Bond operator on the edge to `parent`. */
    std::string_view bond_op;
    /** Unlabelled inline branches, bond operator included. */
    std::string_view branch_text[max_atom_branches_k];
    std::uint8_t num_branches;
    /** SMARTS `:n` label; 0 when the atom carries none. */
    std::uint8_t map_number;
    /** Atom this one hangs off, `no_atom_k` at the root. */
    std::uint8_t parent;
    /** 1 for `[…]`, 0 for a shorthand element symbol. */
    std::uint8_t bracketed;
};

/**
 *  @brief Read the SMARTS atom-map label (`:n`) out of a bracket interior.
 *
 *  A colon inside a recursive `$(...)` body is that sub-query's aromatic-bond
 *  operator, so only depth-zero colons followed by digits are labels.
 *
 *  @param[in] spec Bracket interior, e.g. `CX3:2` or `$(C=O):1`.
 *  @return The label, or 0 when the atom carries none.
 */
inline std::uint8_t spec_map_number(std::string_view spec) {
    int depth = 0;
    for (std::size_t position = 0; position < spec.size(); ++position) {
        char const character = spec[position];
        if (character == '(') ++depth;
        else if (character == ')') --depth;
        else if (character == ':' && depth == 0) {
            std::uint32_t label = 0;
            std::size_t scan = position + 1;
            while (scan < spec.size() && std::isdigit(static_cast<unsigned char>(spec[scan]))) {
                label = label * 10 + (spec[scan] - '0');
                ++scan;
            }
            if (scan != position + 1 && label < 256) return static_cast<std::uint8_t>(label);
        }
    }
    return 0;
}

/**
 *  @brief Find the `]` closing the bracket opened at @p open_position.
 *
 *  Nested brackets and parentheses are tracked so a recursive `$(...)` body
 *  cannot end the atom early.
 *
 *  @param[in] text SMARTS text being scanned.
 *  @param[in] open_position Index of the opening `[`.
 *  @return Index of the matching `]`, or `std::string_view::npos` when unbalanced.
 */
inline std::size_t find_bracket_end(std::string_view text, std::size_t open_position) {
    std::size_t position = open_position + 1;
    int depth_bracket = 0;
    int depth_paren = 0;
    while (position < text.size() && (text[position] != ']' || depth_bracket > 0 || depth_paren > 0)) {
        char const character = text[position];
        if (character == '[') ++depth_bracket;
        else if (character == ']') --depth_bracket;
        else if (character == '(') ++depth_paren;
        else if (character == ')') --depth_paren;
        ++position;
    }
    if (position >= text.size()) return std::string_view::npos;
    return position;
}

/**
 *  @brief Compile a recursive `$(...)` body into @p pool and report the root node.
 *
 *  @param[in] body Text between `$(` and its matching `)`.
 *  @param[in] nesting How many `$(...)` levels enclose this one, the outermost being 1.
 *  @param[inout] pool Sub-query nodes; the body appends its own.
 *  @param[out] root Index of the node the body is rooted at.
 *  @return True when the body fits the compiler's vocabulary.
 */
inline bool compile_recursive_body(std::string_view body, std::uint8_t nesting, recursive_pool_t &pool,
                                   std::uint8_t &root);

/**
 *  @brief Parse one disjunct (the text between `,` separators inside a SMARTS
 *  bracket expression) into an `atom_disjunct_t`.
 *
 *  A "term" is a conjunction of primitives separated by `&`/`;` or implicit
 *  concatenation. Each atomic-OR branch of the parent expression lives in one
 *  disjunct and shares one element identity, one aromaticity, one H-count, etc.
 *
 *  Tokens supported:
 *  - `*` `A` `a` — wildcard / aliphatic / aromatic
 *  - `#<n>` — atomic number
 *  - `C`, `N`, `Cl`, `Br`, ... — element symbol (uppercase = aliphatic; lowercase = aromatic)
 *  - `X<n>` — connectivity
 *  - `D<n>` — heavy-atom degree
 *  - `H` / `H<n>` / `!H<n>` — H count
 *  - `R` / `R0` / `!R` — ring membership
 *  - `r<n>` / `r{N-M}` / `r{N-}` / `r{-M}` / `!r` — smallest ring size / range
 *  - `x<n>` — ring-bond connectivity
 *  - `^<n>` — hybridisation
 *  - `+` / `-` / `+<n>` / `-<n>` — formal charge
 *  - `:<n>` — atom map label (ignored; labels are tracked externally via index[])
 *  - `$(...)` / `!$(...)` — recursive sub-query, compiled into @p pool
 *
 *  @param[in] token The disjunct body (no leading/trailing separators).
 *  @param[in] nesting How many `$(...)` levels enclose this term, zero at a pattern's own atoms.
 *  @param[out] disjunct Accumulator populated with parsed primitives.
 *  @param[inout] pool Sub-query nodes a `$(...)` primitive appends to.
 *  @return `term_parsed_k`, `term_unsatisfiable_k`, or `term_unsupported_k`.
 */
inline term_parse_t parse_atom_term(std::string_view token, std::uint8_t nesting, atom_disjunct_t &disjunct,
                                    recursive_pool_t &pool) {
    bool negate_next = false;
    std::size_t p = 0;
    bool aromatic_positive = false;  // non-negated lowercase symbol, e.g. `c` or `n`
    bool aliphatic_positive = false; // non-negated uppercase symbol, e.g. `C` or `N`
    bool term_has_element = false;
    bool saw_bare_H = false; // `H` without trailing digit — may be element or H-count
    bool saw_wildcard = false;

    while (p < token.size()) {
        char c = token[p];
        if (c == '&' || c == ';') {
            ++p;
            continue;
        }
        if (c == '!') {
            negate_next = true;
            ++p;
            continue;
        }
        if (c == ':') {
            ++p;
            while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) ++p;
            continue;
        }
        if (c == '$') {
            if (p + 1 >= token.size() || token[p + 1] != '(') return term_unsupported_k;
            std::size_t scan = p + 2;
            int depth = 1;
            while (scan < token.size() && depth > 0) {
                if (token[scan] == '(') ++depth;
                else if (token[scan] == ')') --depth;
                if (depth > 0) ++scan;
            }
            if (scan >= token.size()) return term_unsupported_k; // unbalanced
            // Two sub-queries in one conjunction would need two guards on one disjunct;
            // the corpus never writes that, so it stays outside the vocabulary.
            if (disjunct.recursive_root != no_recursive_node_k) return term_unsupported_k;
            std::uint8_t root = no_recursive_node_k;
            if (!compile_recursive_body(token.substr(p + 2, scan - (p + 2)), static_cast<std::uint8_t>(nesting + 1),
                                        pool, root))
                return term_unsupported_k;
            disjunct.recursive_root = root;
            disjunct.recursive_negate = negate_next ? 1 : 0;
            negate_next = false;
            p = scan + 1;
            continue;
        }
        if (c == '*') {
            if (negate_next) return term_unsupported_k;
            saw_wildcard = true;
            ++p;
            continue;
        }
        if (c == 'A') {
            if (negate_next) return term_unsupported_k;
            aliphatic_positive = true;
            ++p;
            continue;
        }
        if (c == 'a') {
            if (negate_next) return term_unsupported_k;
            aromatic_positive = true;
            ++p;
            continue;
        }
        if (c == '#') {
            ++p;
            std::uint32_t num = 0;
            if (p >= token.size() || !std::isdigit(static_cast<unsigned char>(token[p]))) return term_unsupported_k;
            while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                num = num * 10 + (token[p] - '0');
                ++p;
            }
            if (num == 0 || num >= 64) return term_unsupported_k;
            std::uint64_t const bit = (1ULL << num);
            if (negate_next) {
                // `!#n` → element ≠ n. Encode as exclusive mask so elements
                // beyond bit-63 (heavy metals in the corpus) still match.
                if (!term_has_element) {
                    disjunct.element_mask = bit;
                    disjunct.element_mask_exclusive = 1;
                    term_has_element = true;
                }
                else if (disjunct.element_mask_exclusive) { disjunct.element_mask |= bit; }
                else {
                    // Inclusive mask in place; negating a different element
                    // narrows it — just drop the bit if present.
                    disjunct.element_mask &= ~bit;
                }
            }
            else {
                if (!term_has_element) {
                    disjunct.element_mask = bit;
                    disjunct.element_mask_exclusive = 0;
                    term_has_element = true;
                }
                else if (!disjunct.element_mask_exclusive) { disjunct.element_mask |= bit; }
                else {
                    // Exclusive mask already; a positive primitive narrows to
                    // just that element minus the existing exclusions. If the
                    // element was in the exclusion set, the spec is unsatisfiable.
                    if (disjunct.element_mask & bit) return term_unsatisfiable_k;
                    disjunct.element_mask = bit;
                    disjunct.element_mask_exclusive = 0;
                }
            }
            negate_next = false;
            continue;
        }
        if (c == 'X') {
            if (negate_next) return term_unsupported_k;
            ++p;
            std::uint32_t num = 0;
            if (p >= token.size() || !std::isdigit(static_cast<unsigned char>(token[p]))) return term_unsupported_k;
            while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                num = num * 10 + (token[p] - '0');
                ++p;
            }
            if (disjunct.min_degree != 0 && disjunct.min_degree != num) return term_unsatisfiable_k;
            disjunct.min_degree = static_cast<std::uint8_t>(num);
            continue;
        }
        if (c == 'D') {
            // SMARTS `D<n>` — explicit degree (includes explicit H). Running
            // on an `addHs`-ed molecule, this is the same as the atom's
            // `getDegree()` and matches the SMARTS `X<n>` semantics on the
            // same input. Both primitives therefore write to `min_degree`.
            if (negate_next) return term_unsupported_k;
            ++p;
            std::uint32_t num = 0;
            if (p >= token.size() || !std::isdigit(static_cast<unsigned char>(token[p]))) return term_unsupported_k;
            while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                num = num * 10 + (token[p] - '0');
                ++p;
            }
            if (disjunct.min_degree != 0 && disjunct.min_degree != num) return term_unsatisfiable_k;
            disjunct.min_degree = static_cast<std::uint8_t>(num);
            continue;
        }
        if (c == 'x') {
            // SMARTS `x<n>` — ring-bond connectivity, the number of the atom's
            // bonds that lie in a ring. `x0` is a real constraint, so the
            // "unconstrained" sentinel is 0xFF rather than zero.
            if (negate_next) return term_unsupported_k;
            ++p;
            std::uint32_t num = 0;
            if (p >= token.size() || !std::isdigit(static_cast<unsigned char>(token[p]))) return term_unsupported_k;
            while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                num = num * 10 + (token[p] - '0');
                ++p;
            }
            if (num >= 0xFF) return term_unsupported_k;
            if (disjunct.ring_bond_degree != 0xFF && disjunct.ring_bond_degree != num) return term_unsatisfiable_k;
            disjunct.ring_bond_degree = static_cast<std::uint8_t>(num);
            if (num != 0) disjunct.require_flags |= atom_flag_in_ring_k;
            continue;
        }
        if (c == 'H') {
            ++p;
            std::uint32_t count = 1;
            bool explicit_count = false;
            if (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                count = 0;
                while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                    count = count * 10 + (token[p] - '0');
                    ++p;
                }
                explicit_count = true;
            }
            if (count >= 0xFF) return term_unsupported_k;
            if (negate_next) {
                // RDKit reads `[!H]` as `!H1` rather than "not hydrogen": the bare-`H`-is-the-
                // element reading survives only in the positive `[H]`, which it writes `[#1]`.
                if (disjunct.forbid_h_count != 0xFF && disjunct.forbid_h_count != count) return term_unsupported_k;
                disjunct.forbid_h_count = static_cast<std::uint8_t>(count);
                negate_next = false;
                continue;
            }
            if (explicit_count) {
                if (disjunct.exact_h_count != 0xFF && disjunct.exact_h_count != count) return term_unsatisfiable_k;
                disjunct.exact_h_count = static_cast<std::uint8_t>(count);
            }
            else {
                // Bare `H` — decide at end-of-parse: if no element cue appears
                // in the rest of the term, this was the hydrogen ATOM ([H]);
                // otherwise it is the implicit H-count-of-1 modifier ([CH]).
                saw_bare_H = true;
            }
            continue;
        }
        if (c == 'R') {
            ++p;
            if (negate_next) {
                // `!R` — atom is in no ring. A negated ring COUNT (`!R2`) is a
                // different query and stays outside the vocabulary.
                if (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) return term_unsupported_k;
                disjunct.forbid_flags |= atom_flag_in_ring_k;
                negate_next = false;
                continue;
            }
            if (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                std::uint32_t num = 0;
                while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                    num = num * 10 + (token[p] - '0');
                    ++p;
                }
                if (num == 0) {
                    // R0 — atom is in zero rings.
                    disjunct.forbid_flags |= atom_flag_in_ring_k;
                }
                else {
                    // R<n> — atom is a member of exactly n SSSR rings. Matched
                    // via `__builtin_popcount(atom_ring_bitmap)` by
                    // `atom_disjunct_match`. Multiple `R<n>` primitives
                    // with different counts in the same conjunction is
                    // unsatisfiable; drop the disjunct.
                    if (disjunct.ring_count != 0 && disjunct.ring_count != num) return term_unsatisfiable_k;
                    if (num > 255) return term_unsupported_k;
                    disjunct.ring_count = static_cast<std::uint8_t>(num);
                    disjunct.require_flags |= atom_flag_in_ring_k;
                }
            }
            else {
                // Bare `R` — same as SMARTS: atom is in at least one ring.
                disjunct.require_flags |= atom_flag_in_ring_k;
            }
            continue;
        }
        if (c == 'r') {
            ++p;
            if (negate_next) {
                // `!r` — atom is in no ring. A negated ring SIZE (`!r5`) is a
                // different query and stays outside the vocabulary.
                if (p < token.size() && (std::isdigit(static_cast<unsigned char>(token[p])) || token[p] == '{'))
                    return term_unsupported_k;
                disjunct.forbid_flags |= atom_flag_in_ring_k;
                negate_next = false;
                continue;
            }
            std::uint32_t low = 0;
            std::uint32_t high = 0;
            if (p < token.size() && token[p] == '{') {
                // r{N-M}: smallest ring size in [N, M]. The open form r{-M} means "≤ M"
                // (N defaults to 3 since 3 is the smallest possible ring).
                ++p; // skip '{'
                while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                    low = low * 10 + (token[p] - '0');
                    ++p;
                }
                if (low == 0) low = 3; // `r{-N}` open-ended lower bound → 3 (smallest SSSR ring)
                if (p >= token.size() || token[p] != '-') return term_unsupported_k;
                ++p; // skip '-'
                while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                    high = high * 10 + (token[p] - '0');
                    ++p;
                }
                if (high == 0) high = 255; // `r{N-}` open-ended upper bound → saturate
                if (p >= token.size() || token[p] != '}') return term_unsupported_k;
                ++p; // skip '}'
                if (low > high || high > 255) return term_unsupported_k;
            }
            else if (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                    low = low * 10 + (token[p] - '0');
                    ++p;
                }
                if (low == 0 || low > 255) return term_unsupported_k;
                high = low;
            }
            else {
                // Bare `r` — same as bare `R`: atom is in at least one ring.
                disjunct.require_flags |= atom_flag_in_ring_k;
                continue;
            }
            // Several ring-size primitives in one conjunction intersect: `r5` beside
            // `r{5-8}` is `r5`, and `r3` beside `r{5-6}` is nothing at all.
            if (disjunct.ring_size != 0) {
                if (disjunct.ring_size > low) low = disjunct.ring_size;
                if (disjunct.ring_size_max < high) high = disjunct.ring_size_max;
                if (low > high) return term_unsatisfiable_k;
            }
            disjunct.ring_size = static_cast<std::uint8_t>(low);
            disjunct.ring_size_max = static_cast<std::uint8_t>(high);
            disjunct.require_flags |= atom_flag_in_ring_k;
            continue;
        }
        if (c == '+' || c == '-') {
            if (negate_next) return term_unsupported_k;
            std::int8_t sign = (c == '+') ? static_cast<std::int8_t>(1) : static_cast<std::int8_t>(-1);
            ++p;
            std::int8_t magnitude = 1;
            if (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                std::uint32_t num = 0;
                while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                    num = num * 10 + (token[p] - '0');
                    ++p;
                }
                if (num == 0 || num > 8) return term_unsupported_k;
                magnitude = static_cast<std::int8_t>(num);
            }
            std::int8_t const charge = static_cast<std::int8_t>(sign * magnitude);
            std::int8_t constexpr no_charge_k = std::numeric_limits<std::int8_t>::min();
            if (disjunct.formal_charge != no_charge_k && disjunct.formal_charge != charge) return term_unsatisfiable_k;
            disjunct.formal_charge = charge;
            continue;
        }
        if (c == '^') {
            if (negate_next) return term_unsupported_k;
            ++p;
            if (!(p < token.size() && std::isdigit(static_cast<unsigned char>(token[p])))) return term_unsupported_k;
            std::uint32_t num = 0;
            while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                num = num * 10 + (token[p] - '0');
                ++p;
            }
            std::uint8_t requested = 0;
            switch (num) {
            case 1: requested = hybridization_sp_k; break;
            case 2: requested = hybridization_sp2_k; break;
            case 3: requested = hybridization_sp3_k; break;
            case 4: requested = hybridization_sp3d_k; break;
            case 5: requested = hybridization_sp3d2_k; break;
            default: return term_unsupported_k;
            }
            if (disjunct.hybridization != 0 && disjunct.hybridization != requested) return term_unsatisfiable_k;
            disjunct.hybridization = requested;
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c))) {
            std::string_view symbol = token.substr(p, 1);
            std::uint8_t element = symbol_to_element(symbol);
            if (p + 1 < token.size() && std::isalpha(static_cast<unsigned char>(token[p + 1])) &&
                std::islower(static_cast<unsigned char>(token[p + 1]))) {
                std::uint8_t two_letter = symbol_to_element(token.substr(p, 2));
                if (two_letter != 0) {
                    element = two_letter;
                    symbol = token.substr(p, 2);
                }
            }
            if (element == 0) return term_unsupported_k;
            if (element >= 64) return term_unsupported_k;
            std::uint64_t const bit = (1ULL << element);
            bool const lowercase_first = std::islower(static_cast<unsigned char>(symbol[0]));
            if (negate_next) {
                if (!term_has_element) {
                    disjunct.element_mask = bit;
                    disjunct.element_mask_exclusive = 1;
                    term_has_element = true;
                }
                else if (disjunct.element_mask_exclusive) { disjunct.element_mask |= bit; }
                else { disjunct.element_mask &= ~bit; }
            }
            else {
                if (lowercase_first) aromatic_positive = true;
                else aliphatic_positive = true;
                if (!term_has_element) {
                    disjunct.element_mask = bit;
                    disjunct.element_mask_exclusive = 0;
                    term_has_element = true;
                }
                else if (!disjunct.element_mask_exclusive) { disjunct.element_mask |= bit; }
                else {
                    if (disjunct.element_mask & bit) return term_unsatisfiable_k;
                    disjunct.element_mask = bit;
                    disjunct.element_mask_exclusive = 0;
                }
            }
            negate_next = false;
            p += symbol.size();
            continue;
        }
        return term_unsupported_k;
    }

    // Resolve bare `H`: if no element / wildcard / aromatic flag was declared,
    // `[H]` refers to the hydrogen ATOM (element 1); otherwise it is the
    // implicit H-count-of-1 modifier (e.g. `[CH]`).
    if (saw_bare_H) {
        bool const any_element_cue = term_has_element || aromatic_positive || aliphatic_positive || saw_wildcard;
        if (!any_element_cue) {
            disjunct.element_mask = (1ULL << 1);
            term_has_element = true;
        }
        else {
            if (disjunct.exact_h_count != 0xFF && disjunct.exact_h_count != 1) return term_unsatisfiable_k;
            disjunct.exact_h_count = 1;
        }
    }
    if (disjunct.forbid_h_count != 0xFF && disjunct.exact_h_count == disjunct.forbid_h_count)
        return term_unsatisfiable_k;

    // Aromaticity from element-symbol case: only driven by NON-negated element
    // primitives. `[!O]` (negated uppercase O) leaves aromaticity unconstrained
    // so that an aromatic C or aromatic O can still match "not aliphatic-O".
    if (aromatic_positive && !aliphatic_positive) disjunct.require_flags |= atom_flag_aromatic_k;
    else if (aliphatic_positive && !aromatic_positive) disjunct.forbid_flags |= atom_flag_aromatic_k;

    return term_parsed_k;
}

/** Initialise @p disjunct to an "unconstrained" state where every field is the sentinel. */
inline void reset_atom_disjunct(atom_disjunct_t &disjunct) {
    disjunct = {};
    disjunct.element_mask_exclusive = 0;
    disjunct.exact_h_count = 0xFF;
    disjunct.forbid_h_count = 0xFF;
    disjunct.ring_bond_degree = 0xFF;
    disjunct.recursive_root = no_recursive_node_k;
    disjunct.formal_charge = std::numeric_limits<std::int8_t>::min();
}
/**
 *  @brief Parse the content of a SMARTS atom bracket (`[...]`) into a fully populated
 *  `simple_atom_constraint_t`.
 *
 *  SMARTS logical-operator precedence: `!` (highest) → `&` (implicit/explicit
 *  high-precedence AND) → `,` (OR) → `;` (low-precedence AND). Concretely:
 *
 *  - `A,a;r{5-8}`   = `(A,a);(r{5-8})`           = `(A OR a) AND r{5-8}`.
 *  - `!#1;R2,R3;r5` = `(!#1);(R2,R3);(r5)`       = `!#1 AND (R2 OR R3) AND r5`.
 *  - `cH1,n`        = single conjunct of 2 alts  = `cH1 OR n`.
 *
 *  Representation: each `atom_disjunct_t` stores one pure conjunction of
 *  primitives, and an atom matches the parent spec if any disjunct matches.
 *  Mixed AND/OR specs are converted to disjunctive normal form via the
 *  cross-product of the `,`-alternatives across every `;`-separated conjunct.
 *  A recursive `$(...)` rides along in whichever disjunct it was written in, so
 *  `[$([NX3H1]),$([NX3H2])]` yields two disjuncts each carrying its own guard.
 *
 *  @param[in] spec Bracket interior, without the surrounding `[` and `]`.
 *  @param[in] nesting How many `$(...)` levels enclose this spec, zero at a pattern's own atoms.
 *  @param[out] constraint Populated disjunct list.
 *  @param[inout] pool Sub-query nodes the spec's recursive guards append to.
 *  @return True when the spec fits the compiler's vocabulary.
 */
inline bool parse_atom_spec(std::string_view spec, std::uint8_t nesting, simple_atom_constraint_t &constraint,
                            recursive_pool_t &pool) {
    constraint = {};
    constraint.num_disjuncts = 0;
    for (auto &disjunct : constraint.disjuncts) reset_atom_disjunct(disjunct);
    if (spec.empty()) return false;
    if (spec.find('|') != std::string_view::npos) return false;
    std::string_view working_spec = spec;

    // Step 1: split working_spec on top-level `;` into conjuncts (respecting
    // `(` depth in case a recursive predicate leaves behind parens). Each
    // conjunct is later split on top-level `,` into OR-alternatives.
    constexpr std::size_t max_conjuncts_k = 6;
    std::string_view conjuncts[max_conjuncts_k];
    std::size_t num_conjuncts = 0;
    {
        std::size_t p = 0;
        while (p < working_spec.size()) {
            std::size_t start = p;
            int depth = 0;
            while (p < working_spec.size() && !(working_spec[p] == ';' && depth == 0)) {
                if (working_spec[p] == '(') ++depth;
                else if (working_spec[p] == ')') --depth;
                ++p;
            }
            if (num_conjuncts >= max_conjuncts_k) return false;
            std::string_view term = working_spec.substr(start, p - start);
            if (term.empty()) return false;
            conjuncts[num_conjuncts++] = term;
            if (p < working_spec.size()) ++p; // skip ';'
        }
    }
    if (num_conjuncts == 0) return false;

    // Step 2: for each conjunct, split on top-level `,` into OR-alternatives.
    constexpr std::size_t max_alts_per_conjunct_k = 4;
    std::string_view alts[max_conjuncts_k][max_alts_per_conjunct_k];
    std::size_t num_alts[max_conjuncts_k] = {};
    for (std::size_t c = 0; c < num_conjuncts; ++c) {
        std::string_view cstr = conjuncts[c];
        std::size_t p = 0;
        while (p < cstr.size()) {
            std::size_t start = p;
            int depth = 0;
            while (p < cstr.size() && !(cstr[p] == ',' && depth == 0)) {
                if (cstr[p] == '(') ++depth;
                else if (cstr[p] == ')') --depth;
                ++p;
            }
            if (num_alts[c] >= max_alts_per_conjunct_k) return false;
            std::string_view alt = cstr.substr(start, p - start);
            if (alt.empty()) return false;
            alts[c][num_alts[c]++] = alt;
            if (p < cstr.size()) ++p; // skip ','
        }
        if (num_alts[c] == 0) return false;
    }

    // Step 3: cross-product to DNF. Abort if the result exceeds the storage
    // capacity of `simple_atom_constraint_t`.
    std::size_t total = 1;
    for (std::size_t c = 0; c < num_conjuncts; ++c) total *= num_alts[c];
    if (total > max_disjuncts_k) return false;

    // Iterate combinations via mixed-radix counter over num_alts[].
    std::size_t chosen[max_conjuncts_k] = {};
    for (std::size_t combo = 0; combo < total; ++combo) {
        atom_disjunct_t &disjunct = constraint.disjuncts[constraint.num_disjuncts];
        reset_atom_disjunct(disjunct);
        // A combination whose primitives contradict each other matches nothing, so it
        // drops out of the disjunction; only syntax we cannot represent bails out.
        bool satisfiable = true;
        for (std::size_t c = 0; c < num_conjuncts && satisfiable; ++c) {
            switch (parse_atom_term(alts[c][chosen[c]], nesting, disjunct, pool)) {
            case term_parsed_k: break;
            case term_unsatisfiable_k: satisfiable = false; break;
            case term_unsupported_k: return false;
            }
        }
        if (satisfiable) ++constraint.num_disjuncts;
        // increment mixed-radix
        for (std::size_t c = 0; c < num_conjuncts; ++c) {
            if (++chosen[c] < num_alts[c]) break;
            chosen[c] = 0;
        }
    }
    return constraint.num_disjuncts != 0;
}

/**
 *  @brief Parse the text of a bond operator chunk (AND-conjunction of primitives) into a
 *  single `bond_disjunct_t`. Shared helper used by `parse_bond_op` and by
 *  bond-OR splitting.
 */
inline bool parse_bond_conjunct(std::string_view op, bond_disjunct_t &bc) {
    bc = {};
    if (op.empty()) {
        // No explicit operator — SMARTS default is "single or aromatic".
        // Encode as bond_order=0 (any order), no flag constraint.
        return true;
    }
    std::size_t p = 0;
    bool negate = false;
    while (p < op.size()) {
        char c = op[p];
        if (c == ';') {
            ++p;
            continue;
        }
        if (c == '!') {
            negate = true;
            ++p;
            continue;
        }
        if (c == '-') {
            if (negate) return false;
            if (bc.bond_order != 0 && bc.bond_order != 1) return false;
            bc.bond_order = 1;
            bc.forbid_flags |= bond_flag_aromatic_k;
            negate = false;
            ++p;
            continue;
        }
        if (c == '=') {
            if (negate) return false;
            if (bc.bond_order != 0 && bc.bond_order != 2) return false;
            bc.bond_order = 2;
            negate = false;
            ++p;
            continue;
        }
        if (c == '#') {
            if (negate) return false;
            if (bc.bond_order != 0 && bc.bond_order != 3) return false;
            bc.bond_order = 3;
            negate = false;
            ++p;
            continue;
        }
        if (c == ':') {
            if (negate) return false;
            // Aromatic bonds are stored as order-1 with the aromatic flag in
            // the descriptor; pinning bond_order prevents stray order-3
            // aromatised edges (pathological SMILES) from sneaking through.
            if (bc.bond_order != 0 && bc.bond_order != 1) return false;
            bc.bond_order = 1;
            bc.require_flags |= bond_flag_aromatic_k;
            negate = false;
            ++p;
            continue;
        }
        if (c == '~') {
            if (negate) return false;
            negate = false;
            ++p;
            continue;
        }
        if (c == '@') {
            if (negate) bc.forbid_flags |= bond_flag_in_ring_k;
            else bc.require_flags |= bond_flag_in_ring_k;
            negate = false;
            ++p;
            continue;
        }
        // `/` and `\\` would carry directional-bond stereo; not in any loaded
        // pattern, so reject to keep the matcher surface minimal.
        return false;
    }
    return true;
}

/**
 *  @brief Parse a bond operator into `simple_bond_constraint_t`, flattening to
 *  disjunctive normal form. Follows the same SMARTS precedence as @ref
 *  parse_atom_spec: `;` is lowest (AND), `,` is OR, `&`/implicit is higher AND.
 *
 *  Empty @p op is the SMARTS default "single or aromatic" (two disjuncts).
 *  A single comma as in `@;=,:` yields two disjuncts: `@ AND =` and `@ AND :`.
 */
inline bool parse_bond_op(std::string_view op, simple_bond_constraint_t &bc) {
    bc = {};
    if (op.empty()) {
        // Implicit bond = "single or aromatic" = order-1-and-non-aromatic
        // OR order-1-and-aromatic. Both disjuncts pin bond_order to 1 so a
        // triple bond with an incidental aromatic flag cannot slip through.
        bc.num_disjuncts = 2;
        bc.disjuncts[0] = {};
        bc.disjuncts[0].bond_order = 1;
        bc.disjuncts[0].forbid_flags = bond_flag_aromatic_k;
        bc.disjuncts[1] = {};
        bc.disjuncts[1].bond_order = 1;
        bc.disjuncts[1].require_flags = bond_flag_aromatic_k;
        return true;
    }

    constexpr std::size_t max_conjuncts_k = 4;
    constexpr std::size_t max_alts_per_conjunct_k = 4;
    std::string_view conjuncts[max_conjuncts_k];
    std::size_t num_conjuncts = 0;
    {
        std::size_t p = 0;
        while (p < op.size()) {
            std::size_t start = p;
            while (p < op.size() && op[p] != ';') ++p;
            if (num_conjuncts >= max_conjuncts_k) return false;
            std::string_view term = op.substr(start, p - start);
            if (term.empty()) return false;
            conjuncts[num_conjuncts++] = term;
            if (p < op.size()) ++p;
        }
    }
    if (num_conjuncts == 0) return false;

    std::string_view alts[max_conjuncts_k][max_alts_per_conjunct_k];
    std::size_t num_alts[max_conjuncts_k] = {};
    for (std::size_t c = 0; c < num_conjuncts; ++c) {
        std::string_view cstr = conjuncts[c];
        std::size_t p = 0;
        while (p < cstr.size()) {
            std::size_t start = p;
            while (p < cstr.size() && cstr[p] != ',') ++p;
            if (num_alts[c] >= max_alts_per_conjunct_k) return false;
            std::string_view alt = cstr.substr(start, p - start);
            if (alt.empty()) return false;
            alts[c][num_alts[c]++] = alt;
            if (p < cstr.size()) ++p;
        }
        if (num_alts[c] == 0) return false;
    }

    std::size_t total = 1;
    for (std::size_t c = 0; c < num_conjuncts; ++c) total *= num_alts[c];
    if (total > max_disjuncts_k) return false;

    std::size_t index[max_conjuncts_k] = {};
    for (std::size_t combo = 0; combo < total; ++combo) {
        bond_disjunct_t &disjunct = bc.disjuncts[bc.num_disjuncts];
        disjunct = {};
        for (std::size_t c = 0; c < num_conjuncts; ++c) {
            // Compose "alt[c][index[c]]" as the current conjunct's primitive
            // and fold into the disjunct. parse_bond_conjunct starts from a
            // fresh disjunct; here we want accumulation, so hand it an empty
            // then merge the three uint8 fields.
            bond_disjunct_t next {};
            if (!parse_bond_conjunct(alts[c][index[c]], next)) return false;
            if (next.bond_order != 0) {
                if (disjunct.bond_order != 0 && disjunct.bond_order != next.bond_order) return false;
                disjunct.bond_order = next.bond_order;
            }
            disjunct.require_flags |= next.require_flags;
            disjunct.forbid_flags |= next.forbid_flags;
        }
        ++bc.num_disjuncts;
        for (std::size_t c = 0; c < num_conjuncts; ++c) {
            if (++index[c] < num_alts[c]) break;
            index[c] = 0;
        }
    }
    return bc.num_disjuncts > 0;
}

/** Atoms one recursive `$(...)` body or one inline branch body may spell out. */
inline constexpr std::size_t max_recursive_body_atoms_k = 8;

/**
 *  @brief Scan a SMARTS chain into a flat parent-linked atom tree.
 *
 *  Unlike the pattern-level scan this one descends into `(...)` branches instead of
 *  keeping them as text, because a sub-query body is a tree of arbitrary shape rather
 *  than a path with leaves. Ring closures and anything else that would make the body
 *  cyclic are outside the vocabulary and bail out.
 *
 *  @param[in] text Chain to scan, e.g. `[C](=O)([NX3H1])[NX3H1]`.
 *  @param[in] parent Scanned index the first atom of the chain hangs off, `no_atom_k` at the root.
 *  @param[inout] atoms Flat tree filled in scan order.
 *  @param[inout] atom_count Entries already written to @p atoms.
 *  @return True when the whole chain scanned.
 */
inline bool scan_recursive_chain(std::string_view text, std::uint8_t parent, scanned_atom_t *atoms,
                                 std::size_t &atom_count) {
    std::size_t position = 0;
    std::uint8_t previous = parent;
    while (position < text.size()) {
        std::size_t const operator_start = position;
        while (position < text.size() && text[position] != '[' && text[position] != '*' &&
               !std::isalpha(static_cast<unsigned char>(text[position]))) {
            char const character = text[position];
            if (std::isdigit(static_cast<unsigned char>(character))) return false; // ring closure inside a body
            if (character == '(' || character == ')' || character == '$') return false;
            ++position;
        }
        if (position >= text.size()) return false; // bond operator with no atom after it

        if (atom_count >= max_recursive_body_atoms_k) return false;
        std::size_t const current = atom_count;
        scanned_atom_t &atom = atoms[current];
        atom = {};
        atom.parent = previous;
        atom.bond_op = text.substr(operator_start, position - operator_start);

        std::size_t const atom_start = position;
        if (text[position] == '[') {
            std::size_t const bracket_end = find_bracket_end(text, position);
            if (bracket_end == std::string_view::npos) return false;
            atom.spec = text.substr(position + 1, bracket_end - position - 1);
            atom.bracketed = 1;
            position = bracket_end + 1;
        }
        else if (text[position] == '*' || text[position] == 'A' || text[position] == 'a') { ++position; }
        else {
            std::size_t length = 1;
            if (position + 1 < text.size() && std::islower(static_cast<unsigned char>(text[position + 1])) &&
                symbol_to_element(text.substr(position, 2)) != 0)
                length = 2;
            if (symbol_to_element(text.substr(position, length)) == 0) return false;
            position += length;
        }
        atom.text = text.substr(atom_start, position - atom_start);
        ++atom_count;
        previous = static_cast<std::uint8_t>(current);

        while (position < text.size() && text[position] == '(') {
            std::size_t const branch_start = position + 1;
            std::size_t branch_end = branch_start;
            int depth_bracket = 0;
            int depth_paren = 0;
            while (branch_end < text.size() && (text[branch_end] != ')' || depth_bracket > 0 || depth_paren > 0)) {
                char const character = text[branch_end];
                if (character == '[') ++depth_bracket;
                else if (character == ']') --depth_bracket;
                else if (character == '(') ++depth_paren;
                else if (character == ')') --depth_paren;
                ++branch_end;
            }
            if (branch_end >= text.size()) return false;
            if (!scan_recursive_chain(text.substr(branch_start, branch_end - branch_start),
                                      static_cast<std::uint8_t>(current), atoms, atom_count))
                return false;
            position = branch_end + 1;
        }
    }
    return true;
}

/**
 *  @brief Turn one scanned atom and everything hanging off it into pool nodes.
 *
 *  @param[in] atoms Flat tree produced by `scan_recursive_chain`.
 *  @param[in] atom_count Entries in @p atoms.
 *  @param[in] scanned_index Atom to compile.
 *  @param[in] nesting How many `$(...)` levels enclose the body being compiled.
 *  @param[in] hops Distance in bonds from the body's root atom.
 *  @param[inout] pool Sub-query nodes; this subtree appends its own.
 *  @param[out] node Index of the node @p scanned_index compiled to.
 *  @return True when the whole subtree fits the compiler's vocabulary.
 */
inline bool build_recursive_nodes(scanned_atom_t const *atoms, std::size_t atom_count, std::uint8_t scanned_index,
                                  std::uint8_t nesting, std::uint8_t hops, recursive_pool_t &pool, std::uint8_t &node) {
    if (hops > max_recursive_hops_k) return false;
    if (pool.num_nodes >= max_recursive_nodes_k) return false;
    std::uint8_t const node_index = pool.num_nodes++;
    pool.nodes[node_index] = {};
    pool.nodes[node_index].first_child = no_recursive_node_k;
    pool.nodes[node_index].next_sibling = no_recursive_node_k;

    scanned_atom_t const &scanned = atoms[scanned_index];
    simple_atom_constraint_t constraint {};
    if (!parse_atom_spec(scanned.bracketed ? scanned.spec : scanned.text, nesting, constraint, pool)) return false;
    if (constraint.num_neighbor_branches != 0) return false;
    simple_bond_constraint_t bond {};
    if (!parse_bond_op(scanned.bond_op, bond)) return false;

    pool.nodes[node_index].num_atom_disjuncts = constraint.num_disjuncts;
    for (std::uint8_t slot = 0; slot < constraint.num_disjuncts; ++slot)
        pool.nodes[node_index].atom_disjuncts[slot] = constraint.disjuncts[slot];
    pool.nodes[node_index].num_bond_disjuncts = bond.num_disjuncts;
    for (std::uint8_t slot = 0; slot < bond.num_disjuncts; ++slot)
        pool.nodes[node_index].bond_disjuncts[slot] = bond.disjuncts[slot];

    std::uint8_t previous_child = no_recursive_node_k;
    for (std::size_t index = 0; index < atom_count; ++index) {
        if (atoms[index].parent != scanned_index) continue;
        std::uint8_t child = no_recursive_node_k;
        if (!build_recursive_nodes(atoms, atom_count, static_cast<std::uint8_t>(index), nesting,
                                   static_cast<std::uint8_t>(hops + 1), pool, child))
            return false;
        if (previous_child == no_recursive_node_k) pool.nodes[node_index].first_child = child;
        else pool.nodes[previous_child].next_sibling = child;
        previous_child = child;
    }
    node = node_index;
    return true;
}

inline bool compile_recursive_body(std::string_view body, std::uint8_t nesting, recursive_pool_t &pool,
                                   std::uint8_t &root) {
    if (nesting > max_recursive_depth_k) return false;
    if (body.empty()) return false;
    scanned_atom_t atoms[max_recursive_body_atoms_k];
    std::size_t atom_count = 0;
    if (!scan_recursive_chain(body, no_atom_k, atoms, atom_count)) return false;
    if (atom_count == 0) return false;
    return build_recursive_nodes(atoms, atom_count, 0, nesting, 0, pool, root);
}

/**
 *  @brief Parse the content of an inline `(...)` branch into a neighbour constraint.
 *
 *  The branch content is what sits between the parentheses — `cH1` from `([cH1])`,
 *  `=O` from `(=O)`, `[cH0]Cl` from `([cH0]Cl)`. A leading bond operator applies to
 *  the edge leaving the torsion atom and stays on the constraint; everything after it
 *  compiles into the sub-query pool, so a branch may be a chain, may fan out, and may
 *  carry `$(...)` guards of its own.
 *
 *  @param[in] branch Branch content, parentheses already stripped.
 *  @param[inout] pool Sub-query nodes the branch body appends to.
 *  @param[out] constraint Populated with active=1 on success.
 *  @return True when the branch fits the compiler's vocabulary.
 */
inline bool parse_branch_spec(std::string_view branch, recursive_pool_t &pool,
                              simple_neighbor_constraint_t &constraint) {
    constraint = {};
    constraint.node = no_recursive_node_k;
    if (branch.empty()) return false;

    std::size_t position = 0;
    while (position < branch.size() && branch[position] != '[' && branch[position] != '*' &&
           !std::isalpha(static_cast<unsigned char>(branch[position])))
        ++position;
    if (position >= branch.size()) return false; // bond operator with no atom
    simple_bond_constraint_t bond {};
    if (!parse_bond_op(branch.substr(0, position), bond)) return false;

    std::uint8_t root = no_recursive_node_k;
    if (!compile_recursive_body(branch.substr(position), 0, pool, root)) return false;

    constraint.node = root;
    constraint.num_bond_disjuncts = bond.num_disjuncts;
    for (std::uint8_t slot = 0; slot < bond.num_disjuncts; ++slot)
        constraint.bond_disjuncts[slot] = bond.disjuncts[slot];
    constraint.active = 1;
    return true;
}

/** One undirected edge of a scanned SMARTS pattern, chain edge or ring closure alike. */
struct scanned_edge_t {
    /** Empty for a ring closure, which carries the SMARTS default bond. */
    std::string_view bond_op;
    std::uint8_t first;
    std::uint8_t second;
    /** 1 for the three edges the labelled path already owns. */
    std::uint8_t forbidden;
    std::uint8_t padding;
};

/**
 *  @brief Depth-first walk for the arc that closes a ring pattern.
 *
 *  The scanned pattern is a tree plus at most one closure edge, so exactly one
 *  cycle exists; with the three labelled-path edges struck out, the route from
 *  `:4` back to `:1` is the rest of that cycle and nothing else reaches it.
 *
 *  @param[in] edges Every edge of the scanned pattern, path edges already struck out.
 *  @param[in] edge_count Number of entries in @p edges.
 *  @param[in] target Scanned index of `:1`, where the walk stops.
 *  @param[in] node_limit Largest route the closure representation can hold, endpoints included.
 *  @param[inout] route Scanned indices walked so far; the caller seeds it with `:4`.
 *  @param[inout] route_length Entries currently in @p route.
 *  @return True when @p route ends on @p target.
 */
inline bool walk_closure_arc(scanned_edge_t const *edges, std::size_t edge_count, std::uint8_t target,
                             std::uint8_t node_limit, std::uint8_t *route, std::uint8_t &route_length) {
    std::uint8_t const current = route[route_length - 1];
    if (current == target) return true;
    if (route_length >= node_limit) return false;
    for (std::size_t edge = 0; edge < edge_count; ++edge) {
        if (edges[edge].forbidden) continue;
        std::uint8_t next = no_atom_k;
        if (edges[edge].first == current) next = edges[edge].second;
        else if (edges[edge].second == current) next = edges[edge].first;
        else continue;
        bool seen = false;
        for (std::uint8_t step = 0; step < route_length; ++step) seen = seen || route[step] == next;
        if (seen) continue;
        route[route_length++] = next;
        if (walk_closure_arc(edges, edge_count, target, node_limit, route, route_length)) return true;
        --route_length;
    }
    return false;
}

/**
 *  @brief Attempt to compile a 4-atom torsion SMARTS into `simple_torsion_pattern_t`.
 *  Returns false on any feature the matcher cannot represent: a ring closure, an
 *  unknown recursive body, a spare atom that is itself branched, or more than two
 *  inline neighbour constraints on one torsion atom.
 *
 *  The four torsion atoms are bound by their `:1`-`:4` map labels and have to form
 *  a connected path 1-2-3-4; where they sit in the text does not matter, so
 *  `[a:1][c:2]!@;-[C:3](=[N:4])(-N)` binds even though `:4` lives inside a branch.
 *  Every other atom of the pattern must be a leaf hanging off one of the four, and
 *  becomes an inline neighbour constraint on it.
 */
inline bool parse_linear_torsion(std::string_view smarts, simple_torsion_pattern_t &pat) {
    pat = {};
    pat.pool.num_nodes = 0;

    scanned_atom_t atoms[max_pattern_atoms_k];
    std::size_t atom_count = 0;
    std::size_t position = 0;
    std::uint8_t previous_chain_atom = no_atom_k;

    // Ring-closure bookkeeping: one open slot per digit, and the single closure edge
    // the matcher can represent. A second closure would draw a fused system.
    std::uint8_t closure_open[10];
    for (auto &slot : closure_open) slot = no_atom_k;
    std::uint8_t closure_first = no_atom_k;
    std::uint8_t closure_second = no_atom_k;

    while (position < smarts.size()) {
        // Bond operator run preceding the next atom. A digit here would put a bond
        // operator on a ring closure and a parenthesis is a branch hanging off
        // nothing: neither reads back into the four-atom path.
        std::size_t const operator_start = position;
        while (position < smarts.size() && smarts[position] != '[' &&
               !std::isalpha(static_cast<unsigned char>(smarts[position]))) {
            char const character = smarts[position];
            if (std::isdigit(static_cast<unsigned char>(character))) return false;
            if (character == '(' || character == ')') return false;
            ++position;
        }
        if (position >= smarts.size()) return false; // bond operator with no atom after it

        if (atom_count >= max_pattern_atoms_k) return false;
        std::size_t const current = atom_count;
        scanned_atom_t &atom = atoms[current];
        atom = {};
        atom.parent = previous_chain_atom;
        atom.bond_op = smarts.substr(operator_start, position - operator_start);

        std::size_t const atom_start = position;
        if (smarts[position] == '[') {
            std::size_t const bracket_end = find_bracket_end(smarts, position);
            if (bracket_end == std::string_view::npos) return false;
            atom.spec = smarts.substr(position + 1, bracket_end - position - 1);
            atom.bracketed = 1;
            atom.map_number = spec_map_number(atom.spec);
            position = bracket_end + 1;
        }
        else {
            // Shorthand element outside brackets, such as the leading `N` of
            // `N[C:2](=[O:1])!@;-[CH2:3][N:4]`.
            std::size_t length = 1;
            if (position + 1 < smarts.size() && std::islower(static_cast<unsigned char>(smarts[position + 1])) &&
                symbol_to_element(smarts.substr(position, 2)) != 0)
                length = 2;
            if (symbol_to_element(smarts.substr(position, length)) == 0) return false;
            position += length;
        }
        atom.text = smarts.substr(atom_start, position - atom_start);
        ++atom_count;
        previous_chain_atom = static_cast<std::uint8_t>(current);

        // Ring-closure digits and inline `(...)` branches, in whichever order the text
        // writes them. A branch holding a map label becomes a scanned atom of its own
        // so that label can join the torsion path; every other branch stays verbatim
        // text for `parse_branch_spec`.
        while (position < smarts.size()) {
            if (std::isdigit(static_cast<unsigned char>(smarts[position]))) {
                std::uint8_t const digit = static_cast<std::uint8_t>(smarts[position] - '0');
                ++position;
                if (closure_open[digit] == no_atom_k) {
                    closure_open[digit] = static_cast<std::uint8_t>(current);
                    continue;
                }
                if (closure_first != no_atom_k) return false; // a second ring closure fuses two cycles
                closure_first = closure_open[digit];
                closure_second = static_cast<std::uint8_t>(current);
                closure_open[digit] = no_atom_k;
                continue;
            }
            if (smarts[position] != '(') break;
            std::size_t const branch_start = position + 1;
            std::size_t branch_end = branch_start;
            int depth_bracket = 0;
            int depth_paren = 0;
            while (branch_end < smarts.size() && (smarts[branch_end] != ')' || depth_bracket > 0 || depth_paren > 0)) {
                char const character = smarts[branch_end];
                if (character == '[') ++depth_bracket;
                else if (character == ']') --depth_bracket;
                else if (character == '(') ++depth_paren;
                else if (character == ')') --depth_paren;
                ++branch_end;
            }
            if (branch_end >= smarts.size()) return false;
            std::string_view const branch = smarts.substr(branch_start, branch_end - branch_start);
            position = branch_end + 1;

            // A labelled branch is one bond operator plus one bracketed atom and
            // nothing else, e.g. `=[N:4]`.
            std::size_t bond_length = 0;
            while (bond_length < branch.size() && branch[bond_length] != '[' &&
                   !std::isalpha(static_cast<unsigned char>(branch[bond_length])))
                ++bond_length;
            std::uint8_t label = 0;
            std::string_view labelled_spec;
            if (bond_length < branch.size() && branch[bond_length] == '[') {
                std::size_t const bracket_end = find_bracket_end(branch, bond_length);
                if (bracket_end + 1 == branch.size()) {
                    labelled_spec = branch.substr(bond_length + 1, bracket_end - bond_length - 1);
                    label = spec_map_number(labelled_spec);
                }
            }

            if (label != 0) {
                if (atom_count >= max_pattern_atoms_k) return false;
                scanned_atom_t &labelled = atoms[atom_count];
                labelled = {};
                labelled.parent = static_cast<std::uint8_t>(current);
                labelled.bond_op = branch.substr(0, bond_length);
                labelled.spec = labelled_spec;
                labelled.text = branch.substr(bond_length);
                labelled.bracketed = 1;
                labelled.map_number = label;
                ++atom_count;
                continue;
            }

            if (atom.num_branches >= max_atom_branches_k) return false;
            atom.branch_text[atom.num_branches++] = branch;
        }
    }

    // Bind the torsion atoms by label, then demand the 1-2, 2-3 and 3-4 edges.
    std::uint8_t path[4] = {no_atom_k, no_atom_k, no_atom_k, no_atom_k};
    for (std::size_t index = 0; index < atom_count; ++index) {
        std::uint8_t const label = atoms[index].map_number;
        if (label < 1 || label > 4) continue;
        if (path[label - 1] != no_atom_k) return false; // the same label twice
        path[label - 1] = static_cast<std::uint8_t>(index);
    }
    for (std::uint8_t const bound : path)
        if (bound == no_atom_k) return false;

    std::string_view bond_ops[3];
    for (std::size_t step = 0; step < 3; ++step) {
        std::uint8_t const first = path[step];
        std::uint8_t const second = path[step + 1];
        if (atoms[second].parent == first) bond_ops[step] = atoms[second].bond_op;
        else if (atoms[first].parent == second) bond_ops[step] = atoms[first].bond_op;
        else return false; // the labels do not form a connected path
    }

    // Ring closure: the labelled path plus the closure edge draws one cycle, and the
    // arc from `:4` back to `:1` is whatever of that cycle the labels do not cover.
    std::uint8_t arc[max_ring_closure_atoms_k + 2] = {};
    std::uint8_t arc_length = 0;
    for (std::uint8_t const slot : closure_open)
        if (slot != no_atom_k) return false; // a ring digit opened and never closed
    if (closure_first != no_atom_k) {
        scanned_edge_t edges[max_pattern_atoms_k];
        std::size_t edge_count = 0;
        for (std::size_t index = 0; index < atom_count; ++index) {
            if (atoms[index].parent == no_atom_k) continue;
            if (edge_count >= max_pattern_atoms_k) return false;
            edges[edge_count++] = {.bond_op = atoms[index].bond_op,
                                   .first = static_cast<std::uint8_t>(index),
                                   .second = atoms[index].parent,
                                   .forbidden = 0,
                                   .padding = 0};
        }
        if (edge_count >= max_pattern_atoms_k) return false;
        edges[edge_count++] = {.bond_op = {},
                               .first = closure_first,
                               .second = closure_second, //
                               .forbidden = 0,
                               .padding = 0};

        for (std::size_t step = 0; step < 3; ++step)
            for (std::size_t edge = 0; edge < edge_count; ++edge) {
                bool const forward = edges[edge].first == path[step] && edges[edge].second == path[step + 1];
                bool const backward = edges[edge].second == path[step] && edges[edge].first == path[step + 1];
                if (forward || backward) edges[edge].forbidden = 1;
            }

        arc[0] = path[3];
        arc_length = 1;
        if (!walk_closure_arc(edges, edge_count, path[0], static_cast<std::uint8_t>(max_ring_closure_atoms_k + 2), arc,
                              arc_length))
            return false;

        pat.closure.active = 1;
        pat.closure.num_atoms = static_cast<std::uint8_t>(arc_length - 2);
        for (std::uint8_t step = 0; step + 1 < arc_length; ++step) {
            std::string_view step_bond_op;
            bool found = false;
            for (std::size_t edge = 0; edge < edge_count && !found; ++edge) {
                bool const forward = edges[edge].first == arc[step] && edges[edge].second == arc[step + 1];
                bool const backward = edges[edge].second == arc[step] && edges[edge].first == arc[step + 1];
                if (!forward && !backward) continue;
                step_bond_op = edges[edge].bond_op;
                found = true;
            }
            if (!found) return false;
            simple_bond_constraint_t bond {};
            if (!parse_bond_op(step_bond_op, bond)) return false;
            if (step + 2 == arc_length) {
                pat.closure.closing_bond = bond;
                break;
            }
            scanned_atom_t const &arc_atom = atoms[arc[step + 1]];
            if (arc_atom.num_branches != 0) return false; // an arc atom carries no inline branch
            if (arc_atom.map_number != 0) return false;   // a fifth label has no slot in a torsion
            simple_atom_constraint_t constraint {};
            if (!parse_atom_spec(arc_atom.bracketed ? arc_atom.spec : arc_atom.text, 0, constraint, pat.pool))
                return false;
            if (constraint.num_neighbor_branches != 0) return false;
            simple_ring_atom_t &target = pat.closure.atom[step];
            target.num_atom_disjuncts = constraint.num_disjuncts;
            for (std::uint8_t slot = 0; slot < constraint.num_disjuncts; ++slot)
                target.atom_disjuncts[slot] = constraint.disjuncts[slot];
            target.num_bond_disjuncts = bond.num_disjuncts;
            for (std::uint8_t slot = 0; slot < bond.num_disjuncts; ++slot)
                target.bond_disjuncts[slot] = bond.disjuncts[slot];
        }
    }

    // Inline neighbour constraints of each torsion atom: its own verbatim
    // branches first, then every spare atom that leans on it.
    std::string_view branches[4][2];
    std::uint8_t num_branches[4] = {0, 0, 0, 0};
    for (std::size_t slot = 0; slot < 4; ++slot) {
        scanned_atom_t const &bound = atoms[path[slot]];
        if (!bound.bracketed) return false; // a labelled atom always carries brackets
        if (bound.num_branches > 2) return false;
        for (std::uint8_t branch_slot = 0; branch_slot < bound.num_branches; ++branch_slot)
            branches[slot][branch_slot] = bound.branch_text[branch_slot];
        num_branches[slot] = bound.num_branches;
    }

    for (std::size_t index = 0; index < atom_count; ++index) {
        bool on_path = false;
        for (std::size_t step = 0; step < 4; ++step) on_path = on_path || (path[step] == index);
        for (std::uint8_t step = 0; step < arc_length; ++step) on_path = on_path || (arc[step] == index);
        if (on_path) continue;
        if (atoms[index].num_branches != 0) return false; // a spare atom has to be a leaf

        // Exactly one edge, and it has to land on a torsion atom.
        std::uint8_t neighbor = atoms[index].parent;
        std::size_t degree = (neighbor == no_atom_k) ? 0 : 1;
        for (std::size_t other = 0; other < atom_count; ++other)
            if (atoms[other].parent == index) {
                neighbor = static_cast<std::uint8_t>(other);
                ++degree;
            }
        if (degree != 1) return false;
        std::size_t slot = 0;
        bool attached = false;
        for (std::size_t step = 0; step < 4 && !attached; ++step)
            if (path[step] == neighbor) {
                slot = step;
                attached = true;
            }
        if (!attached) return false;

        // A branch reads as its bond operator followed by its atom, and the two
        // are contiguous in the source when the spare atom is the child. When it
        // is instead the parent of a torsion atom, the operator sits on the far
        // side of the atom text, so only an implicit edge reads back.
        std::string_view branch_form;
        if (atoms[index].parent == neighbor)
            branch_form = std::string_view(atoms[index].bond_op.data(),
                                           atoms[index].bond_op.size() + atoms[index].text.size());
        else if (atoms[neighbor].bond_op.empty()) branch_form = atoms[index].text;
        else return false;

        if (num_branches[slot] >= 2) return false;
        branches[slot][num_branches[slot]++] = branch_form;
    }

    for (std::size_t slot = 0; slot < 4; ++slot)
        if (!parse_atom_spec(atoms[path[slot]].spec, 0, pat.atom[slot], pat.pool)) return false;
    for (std::size_t step = 0; step < 3; ++step)
        if (!parse_bond_op(bond_ops[step], pat.bond[step])) return false;
    for (std::size_t slot = 0; slot < 4; ++slot) {
        pat.atom[slot].num_neighbor_branches = num_branches[slot];
        for (std::uint8_t branch_slot = 0; branch_slot < num_branches[slot]; ++branch_slot)
            if (!parse_branch_spec(branches[slot][branch_slot], pat.pool, pat.atom[slot].neighbor_branch[branch_slot]))
                return false;
    }

    return true;
}

/**
 *  @brief Test whether a single atom satisfies a parsed atom constraint.
 *
 *  @param[in] descriptor Molecule descriptor with topology and property arrays.
 *  @param[in] atom_index Atom to test.
 *  @param[in] constraint Disjunct list parsed from one bracketed atom expression.
 *  @param[in] pool Sub-query nodes the constraint's recursive guards index into.
 *  @return True when any disjunct matches.
 */
SF_HOST_DEVICE_ inline bool atom_match(molecule_descriptor_t const &descriptor, std::uint32_t atom_index,
                                       simple_atom_constraint_t const &constraint, recursive_pool_view_t pool) {
    for (std::uint8_t slot = 0; slot < constraint.num_disjuncts; ++slot)
        if (atom_disjunct_match(descriptor, atom_index, constraint.disjuncts[slot], pool)) return true;
    return false;
}

/**
 *  @brief Test whether the neighbour of a torsion atom satisfies one inline branch.
 *
 *  The bond rule is checked first: it is the cheapest and the most selective for a
 *  branch like `(=O)`. The branch body then evaluates as a sub-query rooted at the
 *  neighbour, seeded with the atoms the torsion match already claimed so the body
 *  cannot walk back onto them — which is exactly the distinctness RDKit enforces.
 *
 *  @param[in] descriptor Molecule descriptor with topology and property arrays.
 *  @param[in] atom_index Index of the neighbour atom to test.
 *  @param[in] bond_index Index of the bond connecting the torsion atom to @p atom_index.
 *  @param[in] constraint Neighbour constraint parsed from a branch sub-pattern.
 *  @param[in] pool Sub-query nodes the branch body compiled into.
 *  @param[in] seed Atoms the surrounding match has already claimed.
 *  @return True when the neighbour and everything the branch hangs off it bind.
 */
SF_HOST_DEVICE_ inline bool neighbor_match(molecule_descriptor_t const &descriptor, std::uint32_t atom_index,
                                           std::uint32_t bond_index, simple_neighbor_constraint_t const &constraint,
                                           recursive_pool_view_t pool, recursive_visit_t const &seed) {
    bool any_bond_disjunct = false;
    for (std::uint8_t slot = 0; slot < constraint.num_bond_disjuncts && !any_bond_disjunct; ++slot)
        any_bond_disjunct = bond_disjunct_match(descriptor, bond_index, constraint.bond_disjuncts[slot]);
    if (!any_bond_disjunct) return false;
    recursive_visit_t visited = seed;
    return recursive_match_node(descriptor, pool, constraint.node, atom_index, visited);
}

/**
 *  @brief Verify that @p atom_index has enough neighbours satisfying @p constraint's branch
 *  list. Branch neighbours are drawn from @p atom_index's adjacency, excluding the two
 *  chain-adjacent atoms @p chain_a and @p chain_b (pass 0xFF if unused). When two
 *  branches are present we try both orderings so a pattern like `(~[N,n])(~[N,n])` can
 *  map either way.
 *
 *  @param[in] descriptor Molecule descriptor with topology and property arrays.
 *  @param[in] atom_index Torsion atom whose branches are being checked.
 *  @param[in] constraint Parsed constraint carrying the branch list.
 *  @param[in] pool Sub-query nodes the branch bodies compiled into.
 *  @param[in] chain_a First chain-adjacent atom to keep out of the branch assignment.
 *  @param[in] chain_b Second chain-adjacent atom to keep out of the branch assignment.
 *  @return True when every branch binds to a distinct neighbour.
 */
SF_HOST_DEVICE_ inline bool check_branch_neighbors(molecule_descriptor_t const &descriptor, std::uint32_t atom_index,
                                                   simple_atom_constraint_t const &constraint,
                                                   recursive_pool_view_t pool, std::uint32_t chain_a,
                                                   std::uint32_t chain_b) {
    std::uint8_t const num_branches = constraint.num_neighbor_branches;
    if (num_branches == 0) return true;
    std::uint16_t const start = descriptor.neighbor_offset[atom_index];
    std::uint16_t const end = descriptor.neighbor_offset[atom_index + 1];

    recursive_visit_t seed;
    seed.count = 0;
    seed.atoms[seed.count++] = static_cast<std::uint8_t>(atom_index);
    if (chain_a < max_atoms_per_molecule_k) seed.atoms[seed.count++] = static_cast<std::uint8_t>(chain_a);
    if (chain_b < max_atoms_per_molecule_k) seed.atoms[seed.count++] = static_cast<std::uint8_t>(chain_b);

    if (num_branches == 1) {
        for (std::uint16_t entry = start; entry < end; ++entry) {
            std::uint8_t const candidate = descriptor.neighbor_atom[entry];
            if (candidate == chain_a || candidate == chain_b) continue;
            if (neighbor_match(descriptor, candidate, descriptor.neighbor_bond[entry], constraint.neighbor_branch[0],
                               pool, seed))
                return true;
        }
        return false;
    }

    // Two branches: two distinct neighbours, one per branch slot.
    for (std::uint16_t first = start; first < end; ++first) {
        std::uint8_t const first_atom = descriptor.neighbor_atom[first];
        if (first_atom == chain_a || first_atom == chain_b) continue;
        if (!neighbor_match(descriptor, first_atom, descriptor.neighbor_bond[first], constraint.neighbor_branch[0],
                            pool, seed))
            continue;
        recursive_visit_t second_seed = seed;
        second_seed.atoms[second_seed.count++] = first_atom;
        for (std::uint16_t second = start; second < end; ++second) {
            if (second == first) continue;
            std::uint8_t const second_atom = descriptor.neighbor_atom[second];
            if (second_atom == chain_a || second_atom == chain_b) continue;
            if (neighbor_match(descriptor, second_atom, descriptor.neighbor_bond[second], constraint.neighbor_branch[1],
                               pool, second_seed))
                return true;
        }
    }
    return false;
}

/** Test whether a bond satisfies a parsed bond constraint. */
SF_HOST_DEVICE_ inline bool bond_match(molecule_descriptor_t const &descriptor, std::uint32_t bond_index,
                                       simple_bond_constraint_t const &constraint) {
    if (constraint.num_disjuncts == 0) return true; // empty constraint = "any bond" (defensive)
    for (std::uint8_t slot = 0; slot < constraint.num_disjuncts; ++slot)
        if (bond_disjunct_match(descriptor, bond_index, constraint.disjuncts[slot])) return true;
    return false;
}

/**
 *  @brief Walk the ring arc of a closing pattern from @p origin toward @p target.
 *
 *  Every arc atom is bound in order and the walk stops on the first arc that satisfies the
 *  pattern, because the emitted torsion is named by the four labelled atoms alone: which ring
 *  atoms carried it does not change the answer. The depth-first search carries its own explicit
 *  stack of at most `max_ring_closure_atoms_k` levels rather than recursing, so a GPU thread
 *  runs it inside a frame the compiler sizes once.
 *
 *  @param[in] descriptor Molecule descriptor with topology and property arrays.
 *  @param[in] closure Parsed arc, walked from `:4` toward `:1`.
 *  @param[in] pool Sub-query nodes the arc atoms' recursive guards index into.
 *  @param[in] origin Molecule atom bound to `:4`, where the arc starts.
 *  @param[in] target Molecule atom bound to `:1`, where the arc has to land.
 *  @param[inout] visited Atoms already claimed by this match; the caller seeds the four torsion atoms.
 *  @param[in] seed_count Entries the caller placed in @p visited.
 *  @return True when some assignment of the arc atoms closes onto @p target.
 */
SF_HOST_DEVICE_ inline bool ring_closure_walk(molecule_descriptor_t const &descriptor,
                                              simple_ring_closure_t const &closure, recursive_pool_view_t pool,
                                              std::uint32_t origin, std::uint32_t target, std::uint8_t *visited,
                                              std::uint8_t seed_count) {
    std::uint8_t const total = closure.num_atoms;
    std::uint16_t cursor[max_ring_closure_atoms_k + 1];
    std::uint8_t standing[max_ring_closure_atoms_k + 1];
    standing[0] = static_cast<std::uint8_t>(origin);
    cursor[0] = descriptor.neighbor_offset[origin];
    std::uint8_t depth = 0;
    std::uint8_t visited_count = seed_count;

    while (true) {
        std::uint8_t const current = standing[depth];
        if (depth == total) {
            std::uint16_t const start = descriptor.neighbor_offset[current];
            std::uint16_t const end = descriptor.neighbor_offset[current + 1];
            for (std::uint16_t entry = start; entry < end; ++entry)
                if (descriptor.neighbor_atom[entry] == target &&
                    bond_match(descriptor, descriptor.neighbor_bond[entry], closure.closing_bond))
                    return true;
            if (depth == 0) return false;
            --depth;
            --visited_count;
            continue;
        }

        simple_ring_atom_t const &wanted = closure.atom[depth];
        std::uint16_t const end = descriptor.neighbor_offset[current + 1];
        bool descended = false;
        while (cursor[depth] < end) {
            std::uint16_t const entry = cursor[depth]++;
            std::uint8_t const candidate = descriptor.neighbor_atom[entry];
            bool claimed = false;
            for (std::uint8_t slot = 0; slot < visited_count; ++slot) claimed = claimed || visited[slot] == candidate;
            if (claimed) continue;

            bool bond_ok = false;
            for (std::uint8_t slot = 0; slot < wanted.num_bond_disjuncts && !bond_ok; ++slot)
                bond_ok = bond_disjunct_match(descriptor, descriptor.neighbor_bond[entry], wanted.bond_disjuncts[slot]);
            if (!bond_ok) continue;

            bool atom_ok = false;
            for (std::uint8_t slot = 0; slot < wanted.num_atom_disjuncts && !atom_ok; ++slot)
                atom_ok = atom_disjunct_match(descriptor, candidate, wanted.atom_disjuncts[slot], pool);
            if (!atom_ok) continue;

            visited[visited_count++] = candidate;
            ++depth;
            standing[depth] = candidate;
            cursor[depth] = descriptor.neighbor_offset[candidate];
            descended = true;
            break;
        }
        if (descended) continue;
        if (depth == 0) return false;
        --depth;
        --visited_count;
    }
}

/** Verify the ring a closing pattern draws around an already-bound torsion. */
SF_HOST_DEVICE_ inline bool ring_closure_match(molecule_descriptor_t const &descriptor,
                                               simple_ring_closure_t const &closure, recursive_pool_view_t pool,
                                               std::uint32_t a1, std::uint32_t a2, std::uint32_t a3, std::uint32_t a4) {
    if (!closure.active) return true;
    std::uint8_t visited[4 + max_ring_closure_atoms_k];
    visited[0] = static_cast<std::uint8_t>(a1);
    visited[1] = static_cast<std::uint8_t>(a2);
    visited[2] = static_cast<std::uint8_t>(a3);
    visited[3] = static_cast<std::uint8_t>(a4);
    return ring_closure_walk(descriptor, closure, pool, a4, a1, visited, 4);
}

/** Shortest ring RDKit's torsion extraction still treats as a macrocycle rather than a fused ring. */
inline constexpr std::uint32_t min_macrocycle_size_k = 9;

/** Sentinel returned by `find_bond_between` when the two atoms are not bonded. */
inline constexpr std::uint32_t no_bond_index_k = 0xFFFFFFFFu;

/**
 *  @brief Bond index joining two atoms, or `no_bond_index_k` when they are not bonded.
 *
 *  @param[in] descriptor Molecule descriptor supplying the CSR adjacency.
 *  @param[in] first_atom One end of the sought bond.
 *  @param[in] second_atom Other end of the sought bond.
 *  @return Bond index, or the sentinel.
 */
SF_HOST_DEVICE_ inline std::uint32_t find_bond_between(molecule_descriptor_t const &descriptor,
                                                       std::uint32_t first_atom, std::uint32_t second_atom) {
    std::uint16_t const start = descriptor.neighbor_offset[first_atom];
    std::uint16_t const end = descriptor.neighbor_offset[first_atom + 1];
    for (std::uint16_t entry = start; entry < end; ++entry)
        if (descriptor.neighbor_atom[entry] == second_atom) return descriptor.neighbor_bond[entry];
    return no_bond_index_k;
}

/**
 *  @brief Molecule-wide prefilter keys: which elements and hybridizations a molecule offers and whether
 *  anything in it is aromatic, so a pattern demanding something absent is skipped before a single
 *  bond is touched.
 */
struct csd_molecule_gate_t {
    /** Bit N set when element N is present. */
    std::uint64_t elements;
    /** Bit H set when hybridization H is present. */
    std::uint8_t hybridizations;
    /** 1 when any atom is aromatic. */
    std::uint8_t has_aromatic;
    std::uint8_t padding[6];
};

/**
 *  @brief Fold a molecule's atoms into the prefilter keys the pattern gates are tested against.
 *
 *  @param[in] descriptor Molecule descriptor with the per-atom property arrays.
 *  @return The molecule's element, hybridization and aromaticity keys.
 */
SF_HOST_DEVICE_ inline csd_molecule_gate_t build_molecule_gate(molecule_descriptor_t const &descriptor) {
    csd_molecule_gate_t gate {};
    std::uint32_t const num_atoms = descriptor.num_atoms;
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        std::uint8_t const element = descriptor.element[atom_index];
        if (element > 0 && element < 64) gate.elements |= (1ULL << element);
        if ((descriptor.atom_flags[atom_index] & atom_flag_aromatic_k) != 0) gate.has_aromatic = 1;
        std::uint8_t const hybridization = descriptor.hybridization[atom_index];
        if (hybridization < 8) gate.hybridizations |= static_cast<std::uint8_t>(1u << hybridization);
    }
    return gate;
}

/** Atom count of ring @p ring_index. */
SF_HOST_DEVICE_ inline std::uint32_t ring_atom_count(molecule_descriptor_t const &descriptor,
                                                     std::uint32_t ring_index) {
    return static_cast<std::uint32_t>(descriptor.ring_offset[ring_index + 1] - descriptor.ring_offset[ring_index]);
}

/**
 *  @brief Whether @p first_atom and @p second_atom are consecutive on ring @p ring_index.
 *
 *  The descriptor stores each SSSR ring as its atoms in cycle order, so the ring's bond set is the
 *  set of consecutive pairs, wrapping from the last atom back to the first. Two ring atoms that
 *  are bonded without being consecutive belong to a bridge, and that bond is not part of the ring.
 *
 *  @param[in] descriptor Molecule descriptor carrying the ring table.
 *  @param[in] ring_index Ring to inspect.
 *  @param[in] first_atom One end of the bond.
 *  @param[in] second_atom Other end of the bond.
 *  @return True when the bond lies on the ring.
 */
SF_HOST_DEVICE_ inline bool ring_holds_bond(molecule_descriptor_t const &descriptor, std::uint32_t ring_index,
                                            std::uint8_t first_atom, std::uint8_t second_atom) {
    std::uint16_t const start = descriptor.ring_offset[ring_index];
    std::uint32_t const size = ring_atom_count(descriptor, ring_index);
    for (std::uint32_t position = 0; position < size; ++position) {
        std::uint8_t const current = descriptor.ring_atoms[start + position];
        std::uint8_t const following = descriptor.ring_atoms[start + (position + 1) % size];
        if (current == first_atom && following == second_atom) return true;
        if (current == second_atom && following == first_atom) return true;
    }
    return false;
}

/**
 *  @brief How many SSSR rings a bond lies on, the value RDKit reports as `numBondRings`.
 *
 *  @param[in] descriptor Molecule descriptor carrying the ring table.
 *  @param[in] bond_index Bond to count.
 *  @return Ring count, capped by the descriptor's ring capacity.
 */
SF_HOST_DEVICE_ inline std::uint32_t bond_ring_count(molecule_descriptor_t const &descriptor,
                                                     std::uint32_t bond_index) {
    std::uint8_t const first_atom = descriptor.bond_atom_i[bond_index];
    std::uint8_t const second_atom = descriptor.bond_atom_j[bond_index];
    std::uint32_t const shared = descriptor.atom_ring_bitmap[first_atom] & descriptor.atom_ring_bitmap[second_atom];
    std::uint32_t count = 0;
    for (std::uint32_t ring_index = 0; ring_index < descriptor.num_rings; ++ring_index) {
        if ((shared & (1u << ring_index)) == 0) continue;
        if (ring_holds_bond(descriptor, ring_index, first_atom, second_atom)) ++count;
    }
    return count;
}

/** Whether two rings share more than one bond, which is what marks them fused. */
SF_HOST_DEVICE_ inline bool rings_share_two_bonds(molecule_descriptor_t const &descriptor, std::uint32_t first_ring,
                                                  std::uint32_t second_ring) {
    std::uint16_t const start = descriptor.ring_offset[first_ring];
    std::uint32_t const size = ring_atom_count(descriptor, first_ring);
    std::uint32_t shared = 0;
    for (std::uint32_t position = 0; position < size; ++position) {
        std::uint8_t const current = descriptor.ring_atoms[start + position];
        std::uint8_t const following = descriptor.ring_atoms[start + (position + 1) % size];
        if (!ring_holds_bond(descriptor, second_ring, current, following)) continue;
        if (++shared > 1) return true;
    }
    return false;
}

/**
 *  @brief Whether a bond sits on a small ring fused to another ring, which bars it from a torsion.
 *
 *  RDKit's experimental-torsion extraction drops every bond of a ring that shares more than one
 *  bond with a second ring, unless both rings are macrocycles. A bond only ever falls out through
 *  a ring smaller than `min_macrocycle_size_k`, so the macrocycle exemption is already implied
 *  by the size test on the ring that carries the bond.
 *
 *  @param[in] descriptor Molecule descriptor carrying the ring table.
 *  @param[in] bond_index Bond to test.
 *  @return True when the bond may not carry a torsion.
 */
SF_HOST_DEVICE_ inline bool bond_excluded_by_ring_fusion(molecule_descriptor_t const &descriptor,
                                                         std::uint32_t bond_index) {
    std::uint8_t const first_atom = descriptor.bond_atom_i[bond_index];
    std::uint8_t const second_atom = descriptor.bond_atom_j[bond_index];
    std::uint32_t const shared = descriptor.atom_ring_bitmap[first_atom] & descriptor.atom_ring_bitmap[second_atom];
    std::uint32_t const num_rings = descriptor.num_rings;
    for (std::uint32_t carrier = 0; carrier < num_rings; ++carrier) {
        if ((shared & (1u << carrier)) == 0) continue;
        if (ring_atom_count(descriptor, carrier) >= min_macrocycle_size_k) continue;
        if (!ring_holds_bond(descriptor, carrier, first_atom, second_atom)) continue;
        for (std::uint32_t other = 0; other < num_rings; ++other) {
            if (other == carrier) continue;
            if (rings_share_two_bonds(descriptor, carrier, other)) return true;
        }
    }
    return false;
}

/**
 *  @brief Enumerate every way @p pattern binds with @p bond_index as its central `:2`-`:3` bond.
 *
 *  Both orientations of the central bond are tried, because SMARTS is directional while torsion
 *  semantics is symmetric, and the two labelled end atoms are drawn from the CSR adjacency in
 *  index order. Every binding is handed to @p emit; returning true from @p emit stops the scan,
 *  which is how the dispatcher takes the first match and leaves the rest unexplored.
 *
 *  @param[in] descriptor Molecule descriptor with topology and property arrays.
 *  @param[in] pattern Handle on the pattern's constraints and sub-query nodes.
 *  @param[in] bond_index Bond to bind as the central one.
 *  @param[in] emit Callable taking `(a1, a2, a3, a4, bond_index)` and returning true to stop.
 *  @return True when @p emit stopped the scan.
 */
template <typename emit_fn_>
SF_HOST_DEVICE_ inline bool match_pattern_at_bond(molecule_descriptor_t const &descriptor,
                                                  csd_pattern_view_t const &pattern, std::uint32_t bond_index,
                                                  emit_fn_ &&emit) {
    std::uint8_t const begin_atom = descriptor.bond_atom_i[bond_index];
    std::uint8_t const end_atom = descriptor.bond_atom_j[bond_index];

    for (std::uint32_t flip = 0; flip < 2; ++flip) {
        std::uint8_t const a2 = flip ? end_atom : begin_atom;
        std::uint8_t const a3 = flip ? begin_atom : end_atom;

        if (!atom_match(descriptor, a2, pattern.atom[1], pattern.pool)) continue;
        if (!atom_match(descriptor, a3, pattern.atom[2], pattern.pool)) continue;
        if (!bond_match(descriptor, bond_index, pattern.bond[1])) continue;

        // Iterate neighbors of a2 (excluding a3) for atom[0] (aka :1).
        std::uint16_t const second_start = descriptor.neighbor_offset[a2];
        std::uint16_t const second_end = descriptor.neighbor_offset[a2 + 1];
        constexpr std::uint32_t no_chain_atom_k = 0xFFu;
        for (std::uint16_t first_entry = second_start; first_entry < second_end; ++first_entry) {
            std::uint8_t const a1 = descriptor.neighbor_atom[first_entry];
            if (a1 == a3) continue;
            std::uint16_t const bond_01 = descriptor.neighbor_bond[first_entry];
            if (!atom_match(descriptor, a1, pattern.atom[0], pattern.pool)) continue;
            if (!bond_match(descriptor, bond_01, pattern.bond[0])) continue;

            // Branches on atom[0] depend only on (a1, a2); branches on
            // atom[1] depend on (a2, a1, a3). Both are independent of a4,
            // so check them before iterating a4's candidates.
            if (!check_branch_neighbors(descriptor, a1, pattern.atom[0], pattern.pool, a2, no_chain_atom_k)) continue;
            if (!check_branch_neighbors(descriptor, a2, pattern.atom[1], pattern.pool, a1, a3)) continue;

            // Iterate neighbors of a3 (excluding a2) for atom[3] (aka :4).
            std::uint16_t const third_start = descriptor.neighbor_offset[a3];
            std::uint16_t const third_end = descriptor.neighbor_offset[a3 + 1];
            for (std::uint16_t fourth_entry = third_start; fourth_entry < third_end; ++fourth_entry) {
                std::uint8_t const a4 = descriptor.neighbor_atom[fourth_entry];
                if (a4 == a2 || a4 == a1) continue; // all 4 torsion atoms must be distinct
                std::uint16_t const bond_23 = descriptor.neighbor_bond[fourth_entry];
                if (!atom_match(descriptor, a4, pattern.atom[3], pattern.pool)) continue;
                if (!bond_match(descriptor, bond_23, pattern.bond[2])) continue;

                // Branches on atom[2] depend on a4; atom[3] depends on a3, a4.
                if (!check_branch_neighbors(descriptor, a3, pattern.atom[2], pattern.pool, a2, a4)) continue;
                if (!check_branch_neighbors(descriptor, a4, pattern.atom[3], pattern.pool, a3, no_chain_atom_k))
                    continue;
                if (pattern.closure != nullptr &&
                    !ring_closure_match(descriptor, *pattern.closure, pattern.pool, a1, a2, a3, a4))
                    continue;

                if (emit(a1, a2, a3, a4, bond_index)) return true;
            }
        }
    }
    return false;
}

/**
 *  @brief Try @p pattern against every bond of @p descriptor that @p done_bonds has not claimed.
 *
 *  @param[in] descriptor Molecule descriptor with topology and property arrays.
 *  @param[in] pattern Handle on the pattern's constraints and sub-query nodes.
 *  @param[in] done_bonds One byte per bond, non-zero for a bond an earlier pattern already claimed.
 *  @param[in] emit Callable taking `(a1, a2, a3, a4, bond_index)`; true moves on to the next bond.
 */
template <typename emit_fn_>
inline void match_linear_pattern(molecule_descriptor_t const &descriptor, csd_pattern_view_t const &pattern,
                                 std::uint8_t const *done_bonds, emit_fn_ &&emit) {
    std::uint32_t const num_bonds = descriptor.num_bonds;
    for (std::uint32_t bond_index = 0; bond_index < num_bonds; ++bond_index) {
        if (done_bonds[bond_index]) continue;
        match_pattern_at_bond(descriptor, pattern, bond_index, emit);
    }
}

} // namespace smallfield::preprocess
