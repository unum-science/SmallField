/**
 *  @file csd_torsion_native.hpp
 *  @brief Stack-only native matcher for simple CSD torsion SMARTS patterns.
 *
 *  Handles linear 4-atom chains like "[C]-[C]-[N]-[*]" entirely on the stack, with
 *  zero heap allocations in the hot path. Atom-level OR (`[cH,nX2H0]`), bond-level
 *  OR (`@;=,:`), ring-size ranges (`r{5-8}`), and a set of hand-coded recursive
 *  predicates (`$(S(=O)=O)`, `$(C([F])([F])[F])`, …) are all handled natively. Only
 *  patterns with non-linear topology (ring-closure digits, >4 bracketed atoms, the
 *  4-torsion-atom embedded inside a branch) or unknown recursive bodies fall back.
 *
 *  Why this matters: 192-thread OpenMP preprocessing saturates at 32 threads today
 *  because every SubstructMatch call allocates and destroys several std::vector
 *  objects plus VF2 state arrays. Removing those allocations for the bulk of the
 *  patterns restores linear thread scaling.
 *
 *  Correctness: identical matches to RDKit for the patterns we handle natively.
 *  Validated by a per-pattern per-atom byte-equivalence test against RDKit's
 *  `SubstructMatch` across the full ChEMBL / USearch validation corpora (see
 *  `test_csd_torsion_predicates.cpp`).
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

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
 *  Maximum number of OR-disjuncts the parser will accept inside a single atom or
 *  bond bracket.  Covers every case observed in the CSD pattern corpus (the
 *  longest is `[NX3H0,CX4H0,c]` — three).
 */
inline constexpr std::uint8_t max_disjuncts_k = 4;

/**
 *  One branch of an atom-level SMARTS disjunction, e.g. `cH1` inside `[cH1,n]`.
 *
 *  A single `atom_disjunct_t` is equivalent to a chain of SMARTS primitives
 *  joined by `&`/`;` — an AND of element, degree, H-count, hybridisation,
 *  aromaticity and charge constraints.  Multiple disjuncts joined by `,`
 *  (logical OR) are stored alongside each other in an @ref atom_spec_t.
 *
 *  Matching semantics: an atom satisfies the parent spec if *any* disjunct
 *  matches; constraints that apply uniformly across the disjunction
 *  (recursive predicate, neighbour branches) are stored on the parent and
 *  checked once, regardless of which disjunct matched.
 */
struct atom_disjunct_t {
    std::uint64_t element_mask; ///< inclusive: bit N set = element N allowed (0 = any)
                                ///< exclusive (when @ref element_mask_exclusive): bit N set = element N FORBIDDEN
    std::uint8_t element_mask_exclusive; ///< 1 if @ref element_mask is an exclude-set; elements > 63 implicitly match
    std::uint8_t exact_h_count;          ///< exact H count; 0xFF = any
    std::uint8_t min_degree;             ///< exact total degree (SMARTS X<n>); 0 = any
    std::uint8_t heavy_degree;           ///< exact heavy-atom degree (SMARTS D<n>); 0 = any
    std::uint8_t hybridization;          ///< required hybridization (SMARTS ^<n>); 0 = any
    std::uint8_t require_flags;          ///< atom_flag_t bits that MUST be set
    std::uint8_t forbid_flags;           ///< atom_flag_t bits that MUST NOT be set
    std::uint8_t ring_size;              ///< smallest-ring size lower bound; 0 = any
    std::uint8_t ring_size_max;          ///< smallest-ring size upper bound; 0 = exact match on ring_size
    std::int8_t formal_charge;           ///< exact formal charge; INT8_MIN = no constraint
    std::uint8_t ring_count;             ///< SMARTS R<n> — exact number of SSSR rings atom belongs to; 0 = any
    std::uint8_t padding[4];
};

/**
 *  One branch of a bond-level SMARTS disjunction, e.g. `=` inside `=,:`.
 *  Bond-OR appears in patterns like `@;=,:@;-` (ring-bond AND (double OR
 *  aromatic)).  Defined up here so @ref simple_neighbor_constraint_t can
 *  embed an array of it directly.
 */
struct bond_disjunct_t {
    std::uint8_t bond_order;    ///< 1/2/3; 0 = any
    std::uint8_t require_flags; ///< bond_flag_t bits that MUST be set
    std::uint8_t forbid_flags;  ///< bond_flag_t bits that MUST NOT be set
    std::uint8_t padding;
};

/**
 *  Per-atom branch-neighbor constraint.  Represents an inline neighbour specified
 *  in SMARTS via `]([…])` — e.g. `[c:2]([cH1])` = carbon with an aromatic-CH
 *  neighbour.  Supports atom-OR inside the branch (`(~[N,n])`) via a short
 *  disjunct list and the same OR-bond encoding as top-level bonds (one slot is
 *  enough for every bond op seen in the CSD corpus, including implicit
 *  single-or-aromatic which expands to two disjuncts).
 *  Branches cannot themselves have nested branches or recursive predicates.
 */
struct simple_neighbor_constraint_t {
    atom_disjunct_t atom_disjuncts[max_disjuncts_k];
    bond_disjunct_t bond_disjuncts[max_disjuncts_k];
    std::uint8_t num_atom_disjuncts; ///< 1 for a plain atom; up to 4 for `[a,b,c,d]`
    std::uint8_t num_bond_disjuncts; ///< 1 for an explicit op; 2 for the implicit single-or-aromatic default
    std::uint8_t active;             ///< 1 if this slot carries a real constraint
    std::uint8_t predicate;          ///< csd_predicate_t; applied to the neighbour atom for chain-in-branch
                                     ///< bodies like `(cO)` which we match via the already-present 2-hop
                                     ///< predicates in @ref csd_torsion_predicates.hpp.
    std::uint8_t padding[4];
};

/**
 *  Per-atom constraint extracted from SMARTS.  All fields stack-allocated.
 *  OR-of-element/of-primitives (`[N,O,S]`, `[cH1,n]`) is expressed via a short
 *  @ref atom_disjunct_t list; non-disjunct-specific constraints (recursive
 *  predicate, inline neighbour branches) live at the parent level and apply
 *  regardless of which disjunct an atom matched.
 */
struct simple_atom_constraint_t {
    atom_disjunct_t disjuncts[max_disjuncts_k];
    std::uint8_t num_disjuncts;         ///< 1 for plain spec; up to 4 for `[a,b,c,d]`
    std::uint8_t predicate;             ///< csd_predicate_t; csd_predicate_none_k (0) = no recursive predicate check
    std::uint8_t num_neighbor_branches; ///< 0, 1, or 2 inline `(...)` neighbour branches
    std::uint8_t padding[5];
    simple_neighbor_constraint_t neighbor_branch[2]; ///< active slots defined by num_neighbor_branches
};

/** Per-bond constraint extracted from SMARTS. */
struct simple_bond_constraint_t {
    bond_disjunct_t disjuncts[max_disjuncts_k];
    std::uint8_t num_disjuncts; ///< 1 for plain op; up to 4 for `a,b,c,d`
    std::uint8_t padding[3];
};

/**
 *  Fully parsed linear 4-atom torsion pattern: [atom0]-bond01-[atom1]-bond12-[atom2]-bond23-[atom3].
 *  Labels :1,:2,:3,:4 in SMARTS correspond to atom[0],atom[1],atom[2],atom[3] here.
 *  Central bond is bond[1] (between atom[1] and atom[2]).
 */
struct simple_torsion_pattern_t {
    simple_atom_constraint_t atom[4];
    simple_bond_constraint_t bond[3];
    std::int8_t signs[6];
    float V[6];
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
 *  @brief Parse one disjunct (the text between `,` separators inside a SMARTS
 *  bracket expression) into an @ref atom_disjunct_t.
 *
 *  A "term" is a conjunction of primitives separated by `&`/`;` or implicit
 *  concatenation.  Each atomic-OR branch of the parent expression lives in one
 *  disjunct and shares one element identity, one aromaticity, one H-count, etc.
 *
 *  Tokens supported:
 *  - `*` `A` `a` — wildcard / aliphatic / aromatic
 *  - `#<n>` — atomic number
 *  - `C`, `N`, `Cl`, `Br`, ... — element symbol (uppercase = aliphatic; lowercase = aromatic)
 *  - `X<n>` — connectivity
 *  - `D<n>` — heavy-atom degree
 *  - `H` / `H<n>` — H count
 *  - `R` / `R0` — ring membership
 *  - `r<n>` / `r{N-M}` / `r{N-}` / `r{-M}` — smallest ring size / range
 *  - `^<n>` — hybridisation
 *  - `+` / `-` / `+<n>` / `-<n>` — formal charge
 *  - `:<n>` — atom map label (ignored; labels are tracked externally via idx[])
 *
 *  @param[in] token The disjunct body (no leading/trailing separators).
 *  @param[out] disjunct Accumulator populated with parsed primitives.
 *  @return True on success, false on unsupported syntax.
 */
inline bool parse_atom_term(std::string_view token, atom_disjunct_t &disjunct) {
    bool negate_next = false;
    std::size_t p = 0;
    bool aromatic_positive = false;  ///< non-negated lowercase symbol (e.g. `c`, `n`)
    bool aliphatic_positive = false; ///< non-negated uppercase symbol (e.g. `C`, `N`)
    bool term_has_element = false;
    bool saw_bare_H = false; ///< `H` without trailing digit — may be element or H-count
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
        if (c == '*') {
            if (negate_next) return false;
            saw_wildcard = true;
            ++p;
            continue;
        }
        if (c == 'A') {
            if (negate_next) return false;
            aliphatic_positive = true;
            ++p;
            continue;
        }
        if (c == 'a') {
            if (negate_next) return false;
            aromatic_positive = true;
            ++p;
            continue;
        }
        if (c == '#') {
            ++p;
            std::uint32_t num = 0;
            if (p >= token.size() || !std::isdigit(static_cast<unsigned char>(token[p]))) return false;
            while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                num = num * 10 + (token[p] - '0');
                ++p;
            }
            if (num == 0 || num >= 64) return false;
            std::uint64_t const bit = (1ULL << num);
            if (negate_next) {
                // `!#n` → element ≠ n.  Encode as exclusive mask so elements
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
                    // just that element minus the existing exclusions.  If the
                    // element was in the exclusion set, the spec is unsatisfiable.
                    if (disjunct.element_mask & bit) return false;
                    disjunct.element_mask = bit;
                    disjunct.element_mask_exclusive = 0;
                }
            }
            negate_next = false;
            continue;
        }
        if (c == 'X') {
            if (negate_next) return false;
            ++p;
            std::uint32_t num = 0;
            if (p >= token.size() || !std::isdigit(static_cast<unsigned char>(token[p]))) return false;
            while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                num = num * 10 + (token[p] - '0');
                ++p;
            }
            if (disjunct.min_degree != 0 && disjunct.min_degree != num) return false;
            disjunct.min_degree = static_cast<std::uint8_t>(num);
            continue;
        }
        if (c == 'D') {
            // SMARTS `D<n>` — explicit degree (includes explicit H).  Running
            // on an `addHs`-ed molecule, this is the same as the atom's
            // `getDegree()` and matches the SMARTS `X<n>` semantics on the
            // same input.  Both primitives therefore write to `min_degree`.
            if (negate_next) return false;
            ++p;
            std::uint32_t num = 0;
            if (p >= token.size() || !std::isdigit(static_cast<unsigned char>(token[p]))) return false;
            while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                num = num * 10 + (token[p] - '0');
                ++p;
            }
            if (disjunct.min_degree != 0 && disjunct.min_degree != num) return false;
            disjunct.min_degree = static_cast<std::uint8_t>(num);
            continue;
        }
        if (c == 'H') {
            if (negate_next) return false;
            ++p;
            if (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                std::uint32_t num = 0;
                while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                    num = num * 10 + (token[p] - '0');
                    ++p;
                }
                if (disjunct.exact_h_count != 0xFF && disjunct.exact_h_count != num) return false;
                disjunct.exact_h_count = static_cast<std::uint8_t>(num);
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
            if (negate_next) return false; // negated ring-count not observed in CSD library
            ++p;
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
                    // R<n> — atom is a member of exactly n SSSR rings.  Matched
                    // via `__builtin_popcount(atom_ring_bitmap)` by
                    // @ref atom_disjunct_match.  Multiple `R<n>` primitives
                    // with different counts in the same conjunction is
                    // unsatisfiable; reject.
                    if (disjunct.ring_count != 0 && disjunct.ring_count != num) return false;
                    if (num > 255) return false;
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
            if (negate_next) return false;
            ++p;
            if (p < token.size() && token[p] == '{') {
                // r{N-M}: smallest ring size in [N, M]. The open form r{-M} means "≤ M"
                // (N defaults to 3 since 3 is the smallest possible ring).
                ++p; // skip '{'
                std::uint32_t lo = 0;
                while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                    lo = lo * 10 + (token[p] - '0');
                    ++p;
                }
                if (lo == 0) lo = 3; // `r{-N}` open-ended lower bound → 3 (smallest SSSR ring)
                if (p >= token.size() || token[p] != '-') return false;
                ++p; // skip '-'
                std::uint32_t hi = 0;
                while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                    hi = hi * 10 + (token[p] - '0');
                    ++p;
                }
                if (hi == 0) hi = 255; // `r{N-}` open-ended upper bound → saturate
                if (p >= token.size() || token[p] != '}') return false;
                ++p; // skip '}'
                if (lo > hi || hi > 255) return false;
                if (disjunct.ring_size != 0 || disjunct.ring_size_max != 0) {
                    if (disjunct.ring_size != lo || disjunct.ring_size_max != hi) return false;
                }
                else {
                    disjunct.ring_size = static_cast<std::uint8_t>(lo);
                    disjunct.ring_size_max = static_cast<std::uint8_t>(hi);
                }
                disjunct.require_flags |= atom_flag_in_ring_k;
                continue;
            }
            if (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                std::uint32_t num = 0;
                while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                    num = num * 10 + (token[p] - '0');
                    ++p;
                }
                if (num > 0 && num < 256) {
                    disjunct.ring_size = static_cast<std::uint8_t>(num);
                    disjunct.ring_size_max = 0;
                }
                else return false;
            }
            else { disjunct.require_flags |= atom_flag_in_ring_k; }
            continue;
        }
        if (c == '+' || c == '-') {
            if (negate_next) return false;
            std::int8_t sign = (c == '+') ? static_cast<std::int8_t>(1) : static_cast<std::int8_t>(-1);
            ++p;
            std::int8_t magnitude = 1;
            if (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                std::uint32_t num = 0;
                while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                    num = num * 10 + (token[p] - '0');
                    ++p;
                }
                if (num == 0 || num > 8) return false;
                magnitude = static_cast<std::int8_t>(num);
            }
            std::int8_t const chg = static_cast<std::int8_t>(sign * magnitude);
            std::int8_t constexpr kNoCharge = std::numeric_limits<std::int8_t>::min();
            if (disjunct.formal_charge != kNoCharge && disjunct.formal_charge != chg) return false;
            disjunct.formal_charge = chg;
            continue;
        }
        if (c == '^') {
            if (negate_next) return false;
            ++p;
            if (!(p < token.size() && std::isdigit(static_cast<unsigned char>(token[p])))) return false;
            std::uint32_t num = 0;
            while (p < token.size() && std::isdigit(static_cast<unsigned char>(token[p]))) {
                num = num * 10 + (token[p] - '0');
                ++p;
            }
            std::uint8_t h = 0;
            switch (num) {
            case 1: h = hybridization_sp_k; break;
            case 2: h = hybridization_sp2_k; break;
            case 3: h = hybridization_sp3_k; break;
            case 4: h = hybridization_sp3d_k; break;
            case 5: h = hybridization_sp3d2_k; break;
            default: return false;
            }
            if (disjunct.hybridization != 0 && disjunct.hybridization != h) return false;
            disjunct.hybridization = h;
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c))) {
            std::string_view sym = token.substr(p, 1);
            std::uint8_t elem = symbol_to_element(sym);
            if (p + 1 < token.size() && std::isalpha(static_cast<unsigned char>(token[p + 1])) &&
                std::islower(static_cast<unsigned char>(token[p + 1]))) {
                std::uint8_t elem2 = symbol_to_element(token.substr(p, 2));
                if (elem2 != 0) {
                    elem = elem2;
                    sym = token.substr(p, 2);
                }
            }
            if (elem == 0) return false;
            if (elem >= 64) return false;
            std::uint64_t const bit = (1ULL << elem);
            bool const lowercase_first = std::islower(static_cast<unsigned char>(sym[0]));
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
                    if (disjunct.element_mask & bit) return false;
                    disjunct.element_mask = bit;
                    disjunct.element_mask_exclusive = 0;
                }
            }
            negate_next = false;
            p += sym.size();
            continue;
        }
        return false;
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
            if (disjunct.exact_h_count != 0xFF && disjunct.exact_h_count != 1) return false;
            disjunct.exact_h_count = 1;
        }
    }

    // Aromaticity from element-symbol case: only driven by NON-negated element
    // primitives.  `[!O]` (negated uppercase O) leaves aromaticity unconstrained
    // so that an aromatic C or aromatic O can still match "not aliphatic-O".
    if (aromatic_positive && !aliphatic_positive) disjunct.require_flags |= atom_flag_aromatic_k;
    else if (aliphatic_positive && !aromatic_positive) disjunct.forbid_flags |= atom_flag_aromatic_k;

    return true;
}

/** Initialise @p disjunct to an "unconstrained" state where every field is the sentinel. */
inline void reset_atom_disjunct(atom_disjunct_t &disjunct) {
    disjunct = {};
    disjunct.element_mask_exclusive = 0;
    disjunct.exact_h_count = 0xFF;
    disjunct.formal_charge = std::numeric_limits<std::int8_t>::min();
}

/**
 *  @brief Map a recursive-SMARTS body string to a compiled predicate ID.
 *
 *  The body is the text between `$(` and its matching `)`. Returns
 *  @ref csd_predicate_none_k when the body is not one of the hand-coded
 *  predicates (caller should then bail to the RDKit fallback).
 */
inline csd_predicate_t lookup_predicate(std::string_view body) {
    struct entry_t {
        char const *body;
        csd_predicate_t id;
    };
    constexpr entry_t table[] = {
        {"C=O", csd_predicate_c_eq_o_k},
        {"S(=O)", csd_predicate_s_db_o_k},
        {"[CX3]=O", csd_predicate_cx3_eq_o_k},
        {"[C](=O)", csd_predicate_c_eq_o_branch_k},
        {"[NX3H1]", csd_predicate_nx3h1_k},
        {"[NX3H2]", csd_predicate_nx3h2_k},
        {"C([NX3])", csd_predicate_c_nb_nx3_k},
        {"[CX3]([C])", csd_predicate_cx3_nb_c_k},
        {"[cH0](Cl)", csd_predicate_ch0_nb_cl_k},
        {"[cH0]Cl", csd_predicate_ch0_nb_cl_k}, // alias: no-paren form
        // Top-of-distribution SubstructMatch hot-spots.
        {"[a]([cH1])", csd_predicate_a_nb_cH1_k},
        {"[aH0](!@O)", csd_predicate_aH0_nb_nonring_O_k},
        {"[c]([cH,nX2H0])[cH,nX2H0]", csd_predicate_c_two_cH_or_nX2H0_k},
        {"[cH0][NH1,NH2]", csd_predicate_cH0_nb_NH12_k},
        {"[cH0]([NH1,NH2])", csd_predicate_cH0_nb_NH12_k}, // alias: paren form
        // Sulfonyl / carbonyl / fluorine families.
        {"S(=O)=O", csd_predicate_sulfonyl_k},
        {"S(=O)(=O)", csd_predicate_sulfonyl_k}, // alias: chain vs branch form
        {"[CX3](=[!O])", csd_predicate_cx3_eq_non_o_k},
        {"C([F])([F])[F]", csd_predicate_c_trifluoro_k},
        {"[CX3]([C])([H])", csd_predicate_cx3_nb_c_h_k},
        {"C([NX3])N", csd_predicate_c_nx3_and_aliph_n_k},
        {"C([NX3])n", csd_predicate_c_nx3_and_arom_n_k},
        {"[N][CX4]", csd_predicate_n_nb_cx4_k},
        {"N[CX4]", csd_predicate_n_nb_cx4_k},     // alias: unbracketed N
        {"[NH0][CX4]", csd_predicate_n_nb_cx4_k}, // alias: 0-H aliphatic N
        {"[cH0]F", csd_predicate_cH0_nb_F_k},
        {"[cH0](F)", csd_predicate_cH0_nb_F_k}, // alias: paren form
        {"[cH0][OH1]", csd_predicate_cH0_nb_OH1_k},
        {"[cH0][OH0]", csd_predicate_cH0_nb_OH0_k},
        {"[cH0]([CX3])", csd_predicate_cH0_nb_cx3_k},
        {"cO", csd_predicate_c_nb_aliph_O_k},
        {"a[OH1]", csd_predicate_a_nb_OH1_k},
        {"a[NH1,NH2]", csd_predicate_a_nb_NH12_k},
        {"c([NH1,NH2])", csd_predicate_c_nb_NH12_k},
        {"[c]([NH1,NH2])", csd_predicate_c_nb_NH12_k}, // alias: bracketed-c form
        // Round-2 single-hop additions.
        {"[cH0](=O)", csd_predicate_cH0_eq_o_k},
        {"c!@c", csd_predicate_c_not_ring_c_k},
        {"[n,N](-a)", csd_predicate_nN_nb_aromatic_k},
        {"[a]([nH0,o])", csd_predicate_a_nb_nH0_or_o_k},
    };
    for (auto const &e : table)
        if (body == e.body) return e.id;
    return csd_predicate_none_k;
}

/**
 *  @brief Strip top-level `$(...)` recursive sub-queries out of @p spec in place into
 *  @p stripped, and return the compiled predicate ID for the (at most one) sub-query
 *  encountered. Returns false if the spec contains an unsupported sub-query body or
 *  more than one recursive sub-query.
 *
 *  Only top-level `$(...)` — i.e. not nested inside another `(...)` — is extracted,
 *  which is how all CSD torsion preferences currently express recursive sub-queries
 *  at the atom-spec root.
 */
inline bool strip_recursive_subqueries(std::string_view spec, std::string &stripped, csd_predicate_t &predicate_out) {
    stripped.clear();
    stripped.reserve(spec.size());
    predicate_out = csd_predicate_none_k;
    std::size_t p = 0;
    while (p < spec.size()) {
        if (spec[p] == '$' && p + 1 < spec.size() && spec[p + 1] == '(') {
            // Find matching ')'
            std::size_t q = p + 2;
            int depth = 1;
            while (q < spec.size() && depth > 0) {
                if (spec[q] == '(') ++depth;
                else if (spec[q] == ')') --depth;
                if (depth > 0) ++q;
            }
            if (q >= spec.size()) return false; // unbalanced
            std::string_view body = spec.substr(p + 2, q - (p + 2));
            csd_predicate_t const id = lookup_predicate(body);
            if (id == csd_predicate_none_k) return false;            // unknown body → bail to RDKit
            if (predicate_out != csd_predicate_none_k) return false; // more than one → bail
            predicate_out = id;
            p = q + 1;
            // Swallow the separator (`;` or `&`) that sometimes follows a sub-query
            // in conjunction form, so the stripped spec does not start with one.
            if (stripped.empty() && p < spec.size() && (spec[p] == ';' || spec[p] == '&')) ++p;
            continue;
        }
        stripped.push_back(spec[p]);
        ++p;
    }
    // Trim trailing separators left over when the sub-query was the final conjunct.
    while (!stripped.empty() && (stripped.back() == ';' || stripped.back() == '&')) stripped.pop_back();
    return true;
}

/**
 *  Parse the content of a SMARTS atom bracket (`[...]`) into a fully populated
 *  @ref simple_atom_constraint_t.
 *
 *  SMARTS logical-operator precedence: `!` (highest) → `&` (implicit/explicit
 *  high-precedence AND) → `,` (OR) → `;` (low-precedence AND).  Concretely:
 *
 *  - `A,a;r{5-8}`   = `(A,a);(r{5-8})`           = `(A OR a) AND r{5-8}`.
 *  - `!#1;R2,R3;r5` = `(!#1);(R2,R3);(r5)`       = `!#1 AND (R2 OR R3) AND r5`.
 *  - `cH1,n`        = single conjunct of 2 alts  = `cH1 OR n`.
 *
 *  Representation: each @ref atom_disjunct_t stores one pure conjunction of
 *  primitives, and an atom matches the parent spec if any disjunct matches.
 *  Mixed AND/OR specs are converted to disjunctive normal form via the
 *  cross-product of the `,`-alternatives across every `;`-separated conjunct.
 *
 *  A single top-level recursive sub-query `$(...)` is lifted out and attached
 *  as @ref simple_atom_constraint_t::predicate; recursive bodies inside a
 *  top-level `,` are rejected — we cannot fold a disjunct-varying predicate
 *  into a single spec.
 */
inline bool parse_atom_spec(std::string_view spec, simple_atom_constraint_t &ac) {
    ac = {};
    ac.num_disjuncts = 0;
    for (auto &d : ac.disjuncts) reset_atom_disjunct(d);
    if (spec.empty()) return false;
    if (spec.find('|') != std::string_view::npos) return false;

    // Recursive sub-query extraction (top-level `$(...)`).
    csd_predicate_t predicate_for_spec = csd_predicate_none_k;
    std::string stripped_storage;
    std::string_view working_spec = spec;
    if (spec.find('$') != std::string_view::npos) {
        {
            int depth = 0;
            for (char c : spec) {
                if (c == '(') ++depth;
                else if (c == ')') --depth;
                else if (c == ',' && depth == 0) return false;
            }
        }
        if (!strip_recursive_subqueries(spec, stripped_storage, predicate_for_spec)) return false;
        working_spec = stripped_storage;
    }

    // Step 1: split working_spec on top-level `;` into conjuncts (respecting
    // `(` depth in case a recursive predicate leaves behind parens).  Each
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

    // Step 3: cross-product to DNF.  Abort if the result exceeds the storage
    // capacity of @ref simple_atom_constraint_t.
    std::size_t total = 1;
    for (std::size_t c = 0; c < num_conjuncts; ++c) total *= num_alts[c];
    if (total > max_disjuncts_k) return false;

    // Iterate combinations via mixed-radix counter over num_alts[].
    std::size_t idx[max_conjuncts_k] = {};
    for (std::size_t combo = 0; combo < total; ++combo) {
        atom_disjunct_t &disjunct = ac.disjuncts[ac.num_disjuncts];
        reset_atom_disjunct(disjunct);
        for (std::size_t c = 0; c < num_conjuncts; ++c) {
            if (!parse_atom_term(alts[c][idx[c]], disjunct)) return false;
        }
        ++ac.num_disjuncts;
        // increment mixed-radix
        for (std::size_t c = 0; c < num_conjuncts; ++c) {
            if (++idx[c] < num_alts[c]) break;
            idx[c] = 0;
        }
    }
    if (ac.num_disjuncts == 0) return false;

    ac.predicate = static_cast<std::uint8_t>(predicate_for_spec);
    return true;
}

/**
 *  Parse the text of a bond operator chunk (AND-conjunction of primitives) into a
 *  single @ref bond_disjunct_t.  Shared helper used by @ref parse_bond_op and by
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
 *  Parse a bond operator into @ref simple_bond_constraint_t, flattening to
 *  disjunctive normal form.  Follows the same SMARTS precedence as @ref
 *  parse_atom_spec: `;` is lowest (AND), `,` is OR, `&`/implicit is higher AND.
 *
 *  Empty @p op is the SMARTS default "single or aromatic" (two disjuncts).
 *  A single comma as in `@;=,:` yields two disjuncts: `@ AND =` and `@ AND :`.
 */
inline bool parse_bond_op(std::string_view op, simple_bond_constraint_t &bc) {
    bc = {};
    if (op.empty()) {
        // Implicit bond = "single or aromatic" = order-1-and-non-aromatic
        // OR order-1-and-aromatic.  Both disjuncts pin bond_order to 1 so a
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

    std::size_t idx[max_conjuncts_k] = {};
    for (std::size_t combo = 0; combo < total; ++combo) {
        bond_disjunct_t &disjunct = bc.disjuncts[bc.num_disjuncts];
        disjunct = {};
        for (std::size_t c = 0; c < num_conjuncts; ++c) {
            // Compose "alt[c][idx[c]]" as the current conjunct's primitive
            // and fold into the disjunct.  parse_bond_conjunct starts from a
            // fresh disjunct; here we want accumulation, so hand it an empty
            // then merge the three uint8 fields.
            bond_disjunct_t next {};
            if (!parse_bond_conjunct(alts[c][idx[c]], next)) return false;
            if (next.bond_order != 0) {
                if (disjunct.bond_order != 0 && disjunct.bond_order != next.bond_order) return false;
                disjunct.bond_order = next.bond_order;
            }
            disjunct.require_flags |= next.require_flags;
            disjunct.forbid_flags |= next.forbid_flags;
        }
        ++bc.num_disjuncts;
        for (std::size_t c = 0; c < num_conjuncts; ++c) {
            if (++idx[c] < num_alts[c]) break;
            idx[c] = 0;
        }
    }
    return bc.num_disjuncts > 0;
}

/**
 *  Parse a branch content string (what sits between `(` and `)` after an atom
 *  bracket in SMARTS — e.g. `"cH1"` from `([cH1])`, `"=O"` from `(=O)`, or
 *  `"~[N,n]"` from `(~[N,n])`).  Accepts an optional leading bond operator
 *  followed by either a bracketed atom spec `[…]` or a shorthand element
 *  symbol.  Bond-OR is folded into a single disjunct (the bond operator here
 *  is rarely compound) and atom-OR is preserved as a disjunct list.
 *
 *  Returns false on any feature we can't flatten into a @ref
 *  simple_neighbor_constraint_t — nested branches, recursive sub-queries,
 *  atom-map labels.  On success, @p nb is fully populated with active=1.
 */
/**
 *  @brief Chain-in-branch bodies: two-atom sequences that appear inside an
 *  inline branch such as `(cO)`, `(C=O)`, `([cH0]Cl)`.
 *
 *  These short chains are semantically equivalent to requiring that the
 *  branch neighbour satisfies an already-compiled recursive predicate.  The
 *  table lets us reuse the predicates from `csd_torsion_predicates.hpp`
 *  without growing a 2-hop matcher.
 */
struct chain_branch_entry_t {
    char const *body;          ///< exact branch content (after the bond-op prefix)
    char const *root_spec;     ///< shorthand element or `[…]` spec for the neighbour
    csd_predicate_t predicate; ///< predicate the neighbour atom must additionally satisfy
};

inline chain_branch_entry_t const *lookup_chain_branch(std::string_view body) {
    static chain_branch_entry_t const table[] = {
        {"cO", "c", csd_predicate_c_nb_aliph_O_k},       {"C=O", "C", csd_predicate_c_eq_o_k},
        {"[cH0]Cl", "[cH0]", csd_predicate_ch0_nb_cl_k}, {"[cH0]F", "[cH0]", csd_predicate_cH0_nb_F_k},
        {"c[OH]", "c", csd_predicate_a_nb_OH1_k},        {"a[OH]", "a", csd_predicate_a_nb_OH1_k},
    };
    for (auto const &e : table)
        if (body == e.body) return &e;
    return nullptr;
}

inline bool parse_branch_spec(std::string_view branch, simple_neighbor_constraint_t &nb) {
    nb = {};
    for (auto &d : nb.atom_disjuncts) reset_atom_disjunct(d);
    if (branch.empty()) return false;
    if (branch.find('(') != std::string_view::npos) return false;
    if (branch.find(':') != std::string_view::npos) return false;

    // Split into bond-op prefix + atom body. Bond ops use only a handful of
    // characters (-, =, #, :, ~, @, !, ;), none of which start an atom spec.
    std::size_t p = 0;
    while (p < branch.size() && branch[p] != '[' && !std::isalpha(static_cast<unsigned char>(branch[p]))) ++p;
    simple_bond_constraint_t bc {};
    if (!parse_bond_op(branch.substr(0, p), bc)) return false;

    // Chain-in-branch shortcut: two-atom bodies like `cO`, `C=O`, `[cH0]Cl`
    // lift to an existing recursive predicate on the neighbour, avoiding a
    // generic 2-hop matcher.  The leading bond op (already consumed above)
    // applies to the primary edge; the second-hop bond is single-or-aromatic
    // (the chain-table predicates all assume that).
    {
        std::string_view body = branch.substr(p);
        if (auto const *e = lookup_chain_branch(body)) {
            simple_atom_constraint_t ac {};
            std::string_view root(e->root_spec);
            if (!root.empty() && root[0] == '[') {
                root = root.substr(1, root.size() - 2);
                if (!parse_atom_spec(root, ac)) return false;
            }
            else {
                atom_disjunct_t &d = ac.disjuncts[0];
                reset_atom_disjunct(d);
                std::uint8_t const elem = symbol_to_element(root);
                if (elem == 0 || elem >= 64) return false;
                d.element_mask = (1ULL << elem);
                if (std::islower(static_cast<unsigned char>(root[0]))) d.require_flags |= atom_flag_aromatic_k;
                else d.forbid_flags |= atom_flag_aromatic_k;
                ac.num_disjuncts = 1;
            }
            nb.num_atom_disjuncts = ac.num_disjuncts;
            for (std::uint8_t i = 0; i < ac.num_disjuncts; ++i) nb.atom_disjuncts[i] = ac.disjuncts[i];
            nb.num_bond_disjuncts = bc.num_disjuncts;
            for (std::uint8_t i = 0; i < bc.num_disjuncts; ++i) nb.bond_disjuncts[i] = bc.disjuncts[i];
            nb.predicate = static_cast<std::uint8_t>(e->predicate);
            nb.active = 1;
            return true;
        }
    }

    simple_atom_constraint_t ac {};
    if (p < branch.size() && branch[p] == '[') {
        ++p;
        std::size_t atom_start = p;
        int depth_b = 0, depth_p = 0;
        while (p < branch.size() && (branch[p] != ']' || depth_b > 0 || depth_p > 0)) {
            if (branch[p] == '[') ++depth_b;
            else if (branch[p] == ']') --depth_b;
            else if (branch[p] == '(') ++depth_p;
            else if (branch[p] == ')') --depth_p;
            ++p;
        }
        if (p >= branch.size() || branch[p] != ']') return false;
        std::string_view spec = branch.substr(atom_start, p - atom_start);
        ++p;                                  // skip ']'
        if (p != branch.size()) return false; // stray chars after ']'
        if (!parse_atom_spec(spec, ac)) return false;
    }
    else if (p < branch.size()) {
        // Shorthand element: 1-2 letters (e.g. "C", "c", "Cl", "Br"). No modifiers.
        std::string_view rest = branch.substr(p);
        std::uint8_t elem = 0;
        std::size_t consumed = 0;
        if (rest.size() >= 2 && std::isalpha(static_cast<unsigned char>(rest[0])) &&
            std::islower(static_cast<unsigned char>(rest[1]))) {
            elem = symbol_to_element(rest.substr(0, 2));
            if (elem != 0) consumed = 2;
        }
        if (consumed == 0) {
            elem = symbol_to_element(rest.substr(0, 1));
            if (elem != 0) consumed = 1;
        }
        if (consumed == 0 || elem >= 64 || consumed != rest.size()) return false;
        atom_disjunct_t &d = ac.disjuncts[0];
        reset_atom_disjunct(d);
        d.element_mask = (1ULL << elem);
        if (std::islower(static_cast<unsigned char>(rest[0]))) d.require_flags |= atom_flag_aromatic_k;
        else d.forbid_flags |= atom_flag_aromatic_k;
        ac.num_disjuncts = 1;
    }
    else {
        return false; // bond op without atom
    }

    // Branches must not recurse or carry predicates — both are handled by
    // falling through to the RDKit fallback for the whole pattern.
    if (ac.num_neighbor_branches != 0) return false;
    if (ac.predicate != csd_predicate_none_k) return false;
    if (ac.num_disjuncts == 0) return false;

    // Copy atom + bond disjuncts over verbatim — the representation matches on both sides.
    nb.num_atom_disjuncts = ac.num_disjuncts;
    for (std::uint8_t i = 0; i < ac.num_disjuncts; ++i) nb.atom_disjuncts[i] = ac.disjuncts[i];
    nb.num_bond_disjuncts = bc.num_disjuncts;
    for (std::uint8_t i = 0; i < bc.num_disjuncts; ++i) nb.bond_disjuncts[i] = bc.disjuncts[i];
    nb.active = 1;
    return true;
}

/**
 *  Attempt to parse a linear 4-atom SMARTS pattern into simple_torsion_pattern_t.
 *  Returns false on any feature we don't handle (recursion, nested branches, …).
 *
 *  Accepted syntax: [atom0]<op01>[atom1](<br>)…<op12>[atom2](<br>)…<op23>[atom3](<br>)…
 *  where each `<opNN>` is empty or a bond operator, and each atom may carry up
 *  to 2 inline `(<br>)` branches specifying one neighbour constraint.
 */
inline bool parse_linear_torsion(std::string_view smarts, simple_torsion_pattern_t &pat) {
    pat = {};
    // We expect exactly 4 bracketed atom specs separated by bond operators.
    std::string_view atoms[4];
    std::string_view ops[3];
    std::string_view branch_text[4][2];
    std::uint8_t branch_count[4] = {0, 0, 0, 0};
    std::size_t atom_count = 0;
    std::size_t p = 0;

    while (p < smarts.size()) {
        // Parse bond op before this atom (atoms 1/2/3 have preceding op; atom 0 has none).
        // `(` characters belong to post-atom branches we already consumed, so they
        // should never appear here — if they do, the input is malformed for us.
        std::size_t op_start = p;
        while (p < smarts.size() && smarts[p] != '[') ++p;
        if (p >= smarts.size()) {
            // End of input reached.  Any non-empty skipped region after the
            // 4th atom (e.g. the `=O` trailing tail in `…[C:4]=O`) indicates
            // extra topology we can't represent as a 4-atom linear chain.
            if (atom_count >= 4 && op_start != p) return false;
            break;
        }
        if (atom_count > 0 && atom_count <= 3) ops[atom_count - 1] = smarts.substr(op_start, p - op_start);
        else if (op_start != p) return false;   // stray characters before first atom
        else if (atom_count >= 4) return false; // >4 bracketed atoms — non-linear

        // Parse atom spec. Track both '[' and '(' depths so a recursive sub-query
        // `$(...)` inside the atom bracket does not end the bracket prematurely.
        if (smarts[p] != '[') return false;
        ++p; // skip '['
        std::size_t atom_start = p;
        int depth_bracket = 0;
        int depth_paren = 0;
        while (p < smarts.size() && (smarts[p] != ']' || depth_bracket > 0 || depth_paren > 0)) {
            if (smarts[p] == '[') ++depth_bracket;
            else if (smarts[p] == ']') --depth_bracket;
            else if (smarts[p] == '(') ++depth_paren;
            else if (smarts[p] == ')') --depth_paren;
            ++p;
        }
        if (p >= smarts.size() || smarts[p] != ']') return false;
        if (atom_count >= 4) return false;
        std::size_t const current = atom_count;
        atoms[atom_count++] = smarts.substr(atom_start, p - atom_start);
        ++p; // skip ']'

        // Consume any inline `(...)` branches attached to this atom. Each branch
        // becomes one @ref simple_neighbor_constraint_t attached to the atom;
        // more than 2 branches per atom forces a fallback.
        while (p < smarts.size() && smarts[p] == '(') {
            ++p; // skip '('
            std::size_t branch_start = p;
            int bd = 0, bp = 0;
            while (p < smarts.size() && (smarts[p] != ')' || bd > 0 || bp > 0)) {
                if (smarts[p] == '[') ++bd;
                else if (smarts[p] == ']') --bd;
                else if (smarts[p] == '(') ++bp;
                else if (smarts[p] == ')') --bp;
                ++p;
            }
            if (p >= smarts.size() || smarts[p] != ')') return false;
            if (branch_count[current] >= 2) return false;
            branch_text[current][branch_count[current]++] = smarts.substr(branch_start, p - branch_start);
            ++p; // skip ')'
        }
    }

    if (atom_count != 4) return false;

    // Parse each atom and bond spec, then attach any branches.
    for (std::uint32_t i = 0; i < 4; ++i)
        if (!parse_atom_spec(atoms[i], pat.atom[i])) return false;
    for (std::uint32_t i = 0; i < 3; ++i)
        if (!parse_bond_op(ops[i], pat.bond[i])) return false;
    for (std::uint32_t i = 0; i < 4; ++i) {
        pat.atom[i].num_neighbor_branches = branch_count[i];
        for (std::uint32_t j = 0; j < branch_count[i]; ++j)
            if (!parse_branch_spec(branch_text[i][j], pat.atom[i].neighbor_branch[j])) return false;
    }

    return true;
}

/** @brief Test whether a bond satisfies one @ref bond_disjunct_t. */
[[gnu::hot, gnu::always_inline]] inline bool bond_disjunct_match(molecule_descriptor_t const &desc,
                                                                 std::uint32_t bond_idx, bond_disjunct_t const &bd) {
    if (bd.bond_order != 0 && desc.bond_order[bond_idx] != bd.bond_order) return false;
    std::uint8_t const flags = desc.bond_flags[bond_idx];
    if ((flags & bd.require_flags) != bd.require_flags) return false;
    if ((flags & bd.forbid_flags) != 0) return false;
    return true;
}

/**
 *  @brief Test whether a single atom satisfies one @ref atom_disjunct_t (one OR
 *  branch of a SMARTS atom spec).  Called by @ref atom_match for every disjunct
 *  until one matches.
 */
[[gnu::hot, gnu::always_inline]] inline bool atom_disjunct_match(molecule_descriptor_t const &desc,
                                                                 std::uint32_t atom_idx, atom_disjunct_t const &d) {
    std::uint8_t const element = desc.element[atom_idx];
    if (d.element_mask != 0) {
        if (d.element_mask_exclusive) {
            // Exclude-set: bit N set ⇒ element N is forbidden. Elements > 63
            // are implicitly allowed (they cannot collide with a 64-bit mask).
            if (element < 64 && (d.element_mask & (1ULL << element)) != 0) return false;
        }
        else {
            // Include-set: bit N set ⇒ element N is accepted.  Elements > 63
            // cannot match a 64-bit positive enumeration and are rejected.
            if (element >= 64) return false;
            if ((d.element_mask & (1ULL << element)) == 0) return false;
        }
    }
    if (d.min_degree != 0 && desc.degree[atom_idx] != d.min_degree) return false;
    if (d.heavy_degree != 0) {
        std::uint8_t const dg = desc.degree[atom_idx];
        std::uint8_t const num_hydrogens = desc.num_hydrogens[atom_idx];
        std::uint8_t const heavy = (dg >= num_hydrogens) ? static_cast<std::uint8_t>(dg - num_hydrogens) : 0;
        if (heavy != d.heavy_degree) return false;
    }
    if (d.exact_h_count != 0xFF && desc.num_hydrogens[atom_idx] != d.exact_h_count) return false;
    if (d.ring_size != 0) {
        std::uint8_t const rs = desc.smallest_ring[atom_idx];
        if (d.ring_size_max == 0) {
            if (rs != d.ring_size) return false;
        }
        else {
            if (rs < d.ring_size || rs > d.ring_size_max) return false;
        }
    }
    if (d.ring_count != 0) {
        // SMARTS R<n>: atom belongs to exactly n SSSR rings.  The descriptor
        // caps ring-tracking at `max_rings_k` (32); atoms belonging to more
        // rings than that are conservatively under-counted, but RDKit's own
        // R<n> matcher has the same SSSR dependency so divergence only
        // appears on molecules the descriptor already treats specially.
        std::uint8_t const rc = static_cast<std::uint8_t>(__builtin_popcount(desc.atom_ring_bitmap[atom_idx]));
        if (rc != d.ring_count) return false;
    }
    if (d.hybridization != 0 && desc.hybridization[atom_idx] != d.hybridization) return false;
    std::int8_t constexpr kNoCharge = std::numeric_limits<std::int8_t>::min();
    if (d.formal_charge != kNoCharge && desc.formal_charge[atom_idx] != d.formal_charge) return false;
    std::uint8_t const flags = desc.atom_flags[atom_idx];
    if ((flags & d.require_flags) != d.require_flags) return false;
    if ((flags & d.forbid_flags) != 0) return false;
    return true;
}

/** @brief Test whether a single atom satisfies a parsed atom constraint. */
inline bool atom_match(molecule_descriptor_t const &desc, std::uint32_t atom_idx, simple_atom_constraint_t const &ac) {
    // Atom-OR: any disjunct matches is enough.
    bool any_disjunct = false;
    for (std::uint8_t i = 0; i < ac.num_disjuncts; ++i) {
        if (atom_disjunct_match(desc, atom_idx, ac.disjuncts[i])) {
            any_disjunct = true;
            break;
        }
    }
    if (!any_disjunct) return false;
    // Recursive predicate applies atom-wide, after the disjunct match.
    if (ac.predicate != csd_predicate_none_k &&
        !csd_predicate_check(desc, atom_idx, static_cast<csd_predicate_t>(ac.predicate)))
        return false;
    return true;
}

/**
 *  @brief Test whether the neighbor of atom_idx across bond_idx satisfies a single-atom constraint from a parsed
 *  torsion-pattern term.
 *
 *  Inlines the atom and bond rules so the hot path has no indirection: the
 *  parent @ref atom_match already covered the parent itself.
 *
 *  @param[in] desc Molecule descriptor with topology and property arrays.
 *  @param[in] atom_idx Index of the neighbor atom to test.
 *  @param[in] bond_idx Index of the bond connecting the parent atom to @p atom_idx.
 *  @param[in] nc Neighbor constraint parsed from a branch sub-pattern.
 *  @return True if the neighbor satisfies every primitive in @p nc.
 */
inline bool neighbor_match(molecule_descriptor_t const &desc, std::uint32_t atom_idx, std::uint32_t bond_idx,
                           simple_neighbor_constraint_t const &nc) {
    // Bond rules first — cheapest + most selective for double-bond branches like `(=O)`.
    bool any_bond_disjunct = false;
    for (std::uint8_t i = 0; i < nc.num_bond_disjuncts; ++i) {
        if (bond_disjunct_match(desc, bond_idx, nc.bond_disjuncts[i])) {
            any_bond_disjunct = true;
            break;
        }
    }
    if (!any_bond_disjunct) return false;

    // Atom-OR: any disjunct matches is enough.
    bool any_atom_disjunct = false;
    for (std::uint8_t i = 0; i < nc.num_atom_disjuncts; ++i) {
        if (atom_disjunct_match(desc, atom_idx, nc.atom_disjuncts[i])) {
            any_atom_disjunct = true;
            break;
        }
    }
    if (!any_atom_disjunct) return false;

    // Chain-in-branch predicate: when the branch body is a two-atom chain
    // (`cO`, `C=O`, `[cH0]Cl`, …) the neighbour atom itself must satisfy an
    // already-compiled recursive predicate that encodes the second hop.
    if (nc.predicate != csd_predicate_none_k &&
        !csd_predicate_check(desc, atom_idx, static_cast<csd_predicate_t>(nc.predicate)))
        return false;

    return true;
}

/**
 *  Verify that @p atom_idx has enough neighbours satisfying @p ac's branch list.
 *  Branch neighbours are drawn from @p atom_idx's adjacency, excluding the two
 *  chain-adjacent atoms @p chain_a and @p chain_b (pass 0xFF if unused).  When
 *  two branches are present we try both orderings so a pattern like
 *  `(~[N,n])(~[N,n])` can map either way.
 */
inline bool check_branch_neighbors(molecule_descriptor_t const &desc, std::uint32_t atom_idx,
                                   simple_atom_constraint_t const &ac, std::uint32_t chain_a, std::uint32_t chain_b) {
    std::uint8_t const n_br = ac.num_neighbor_branches;
    if (n_br == 0) return true;
    std::uint16_t const s = desc.neighbor_offset[atom_idx];
    std::uint16_t const e = desc.neighbor_offset[atom_idx + 1];

    if (n_br == 1) {
        for (std::uint16_t n = s; n < e; ++n) {
            std::uint8_t const na = desc.neighbor_atom[n];
            if (na == chain_a || na == chain_b) continue;
            if (neighbor_match(desc, na, desc.neighbor_bond[n], ac.neighbor_branch[0])) return true;
        }
        return false;
    }

    // n_br == 2: need two distinct neighbours, one per branch slot.
    for (std::uint16_t n0 = s; n0 < e; ++n0) {
        std::uint8_t const a0 = desc.neighbor_atom[n0];
        if (a0 == chain_a || a0 == chain_b) continue;
        std::uint16_t const b0 = desc.neighbor_bond[n0];
        if (!neighbor_match(desc, a0, b0, ac.neighbor_branch[0])) continue;
        for (std::uint16_t n1 = s; n1 < e; ++n1) {
            if (n1 == n0) continue;
            std::uint8_t const a1 = desc.neighbor_atom[n1];
            if (a1 == chain_a || a1 == chain_b) continue;
            if (neighbor_match(desc, a1, desc.neighbor_bond[n1], ac.neighbor_branch[1])) return true;
        }
    }
    return false;
}

/** @brief Test whether a bond satisfies a parsed bond constraint. */
inline bool bond_match(molecule_descriptor_t const &desc, std::uint32_t bond_idx, simple_bond_constraint_t const &bc) {
    if (bc.num_disjuncts == 0) return true; // empty constraint = "any bond" (defensive)
    for (std::uint8_t i = 0; i < bc.num_disjuncts; ++i) {
        if (bond_disjunct_match(desc, bond_idx, bc.disjuncts[i])) return true;
    }
    return false;
}

/**
 *  Try to match @p pat against every rotatable bond in @p desc as the central :2-:3 bond.
 *  Tries both orientations (pat as-is and pat reversed) because SMARTS patterns
 *  are inherently directional but torsion semantics are symmetric.
 *
 *  On each successful match, calls @p emit with (a1, a2, a3, a4) atom indices.
 *
 *  Stops scanning once @p done[bond_idx] == true for the central bond.
 */
template <typename emit_fn_>
inline void match_linear_pattern(molecule_descriptor_t const &desc, simple_torsion_pattern_t const &pat,
                                 std::uint8_t const *done_bonds, emit_fn_ &&emit) {
    std::uint32_t const nb = desc.num_bonds;

    for (std::uint32_t b = 0; b < nb; ++b) {
        if (done_bonds[b]) continue;
        std::uint8_t const bi = desc.bond_atom_i[b];
        std::uint8_t const bj = desc.bond_atom_j[b];

        // Two orientations of the central bond: (a2=bi, a3=bj) and (a2=bj, a3=bi).
        // We try each; the emit callback handles marking done_bonds.
        for (std::uint32_t flip = 0; flip < 2; ++flip) {
            std::uint8_t const a2 = flip ? bj : bi;
            std::uint8_t const a3 = flip ? bi : bj;

            if (!atom_match(desc, a2, pat.atom[1])) continue;
            if (!atom_match(desc, a3, pat.atom[2])) continue;
            if (!bond_match(desc, b, pat.bond[1])) continue;

            // Iterate neighbors of a2 (excluding a3) for atom[0] (aka :1).
            std::uint16_t const s2 = desc.neighbor_offset[a2];
            std::uint16_t const e2 = desc.neighbor_offset[a2 + 1];
            constexpr std::uint32_t kNone = 0xFFu;
            for (std::uint16_t n1 = s2; n1 < e2; ++n1) {
                std::uint8_t const a1 = desc.neighbor_atom[n1];
                if (a1 == a3) continue;
                std::uint16_t const b01 = desc.neighbor_bond[n1];
                if (!atom_match(desc, a1, pat.atom[0])) continue;
                if (!bond_match(desc, b01, pat.bond[0])) continue;

                // Branches on atom[0] depend only on (a1, a2); branches on
                // atom[1] depend on (a2, a1, a3). Both are independent of a4,
                // so check them before iterating a4's candidates.
                if (!check_branch_neighbors(desc, a1, pat.atom[0], a2, kNone)) continue;
                if (!check_branch_neighbors(desc, a2, pat.atom[1], a1, a3)) continue;

                // Iterate neighbors of a3 (excluding a2) for atom[3] (aka :4).
                std::uint16_t const s3 = desc.neighbor_offset[a3];
                std::uint16_t const e3 = desc.neighbor_offset[a3 + 1];
                for (std::uint16_t n4 = s3; n4 < e3; ++n4) {
                    std::uint8_t const a4 = desc.neighbor_atom[n4];
                    if (a4 == a2 || a4 == a1) continue; // all 4 torsion atoms must be distinct
                    std::uint16_t const b23 = desc.neighbor_bond[n4];
                    if (!atom_match(desc, a4, pat.atom[3])) continue;
                    if (!bond_match(desc, b23, pat.bond[2])) continue;

                    // Branches on atom[2] depend on a4; atom[3] depends on a3, a4.
                    if (!check_branch_neighbors(desc, a3, pat.atom[2], a2, a4)) continue;
                    if (!check_branch_neighbors(desc, a4, pat.atom[3], a3, kNone)) continue;

                    // Match! Call emit; it marks done_bonds and returns.
                    if (emit(a1, a2, a3, a4, b)) goto next_bond;
                }
            }
        }
    next_bond:;
    }
}

} // namespace smallfield::preprocess
