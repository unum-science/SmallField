/**
 *  @file src/smallfield/preprocess/csd_torsion_native.hpp
 *  @brief Compiler from CSD torsion SMARTS text into the stack-only pattern the matcher evaluates.
 *
 *  Host-only and reached once per process from `csd_torsion_dispatcher.hpp`; the shapes it fills live in
 *  `csd_torsion_predicates.hpp`, which is the half the device kernel compiles.
 *
 *  The whole 470-pattern CSD library fits the vocabulary: four atoms bound by their `:1`-`:4` map labels
 *  wherever they sit in the text, atom-level OR (`[cH,nX2H0]`), bond-level OR (`@;=,:`), ring-size ranges
 *  (`r{5-8}`), ring-bond connectivity (`x3`), one ring closure walking the labelled atoms back around a
 *  cycle, and recursive `$(...)` bodies.
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::int8_t`

#include <iterator>    // `std::size`
#include <string_view> // `std::string_view`

#include "smallfield/preprocess/csd_torsion_predicates.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::preprocess {
using smallfield::types::bond_flag_aromatic_k;
using smallfield::types::bond_flag_in_ring_k;
using smallfield::types::hybridization_sp2_k;
using smallfield::types::hybridization_sp3_k;
using smallfield::types::hybridization_sp3d2_k;
using smallfield::types::hybridization_sp3d_k;
using smallfield::types::hybridization_sp_k;

#pragma region Lexer

/*  SMARTS is ASCII and these run inside a `constexpr` cursor, so the `<cctype>` predicates would only
 *  add a locale the grammar does not have and an `unsigned char` cast at every call. */
constexpr bool is_digit_(char character) noexcept { return character >= '0' && character <= '9'; }
constexpr bool is_lower_(char character) noexcept { return character >= 'a' && character <= 'z'; }
constexpr bool is_alpha_(char character) noexcept {
    return is_lower_(character) || (character >= 'A' && character <= 'Z');
}

/** Read cursor over one piece of SMARTS text. Past the end `peek` answers `\0`, which is not a SMARTS
 *  character, so no caller tests `done` before looking. */
struct smarts_cursor_t {
    std::string_view text;
    std::size_t position = 0;

    constexpr bool done() const noexcept { return position >= text.size(); }
    constexpr char peek() const noexcept { return done() ? '\0' : text[position]; }
    constexpr void advance() noexcept {
        if (!done()) ++position;
    }
    constexpr char take() noexcept {
        char const character = peek();
        advance();
        return character;
    }
    /** Text from @p start up to the cursor, which is how a run is captured once its end is known. */
    constexpr std::string_view since(std::size_t start) const noexcept { return text.substr(start, position - start); }
    constexpr void skip_digits() noexcept {
        while (is_digit_(peek())) advance();
    }
    /** Read the decimal run under the cursor into @p value, which is left untouched — so a caller may
     *  pre-load a default — unless the whole run is read and fits 16 bits. */
    constexpr bool take_unsigned(std::uint32_t &value) noexcept {
        if (!is_digit_(peek())) return false;
        std::uint32_t scanned = 0;
        while (is_digit_(peek())) {
            scanned = scanned * 10 + static_cast<std::uint32_t>(take() - '0');
            if (scanned > 0xFFFFu) return false;
        }
        value = scanned;
        return true;
    }
};

/** Atomic number of a one- or two-letter element symbol, aliphatic or aromatic spelling alike, or 0 when
 *  the symbol names no element the CSD corpus uses. */
inline std::uint8_t symbol_to_element(std::string_view symbol) {
    struct entry_t {
        char const *symbol;
        std::uint8_t atomic_number;
    };
    constexpr entry_t table[] = {
        {"H", 1},  {"He", 2}, {"Li", 3},  {"Be", 4},  {"B", 5},   {"C", 6},   {"N", 7},
        {"O", 8},  {"F", 9},  {"Ne", 10}, {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14},
        {"P", 15}, {"S", 16}, {"Cl", 17}, {"Ar", 18}, {"K", 19},  {"Ca", 20}, {"Br", 35},
        {"I", 53}, {"b", 5},  {"c", 6},   {"n", 7},   {"o", 8},   {"p", 15},  {"s", 16},
    };
    for (auto const &entry : table)
        if (symbol == entry.symbol) return entry.atomic_number;
    return 0;
}

/** Read the element symbol standing under the cursor, preferring the two-letter reading. The cursor steps
 *  past it only when one is read. */
inline std::uint8_t take_element_symbol(smarts_cursor_t &cursor) {
    if (is_lower_(cursor.text.substr(cursor.position).size() > 1 ? cursor.text[cursor.position + 1] : '\0')) {
        std::uint8_t const two_letter = symbol_to_element(cursor.text.substr(cursor.position, 2));
        if (two_letter != 0) {
            cursor.position += 2;
            return two_letter;
        }
    }
    std::uint8_t const one_letter = symbol_to_element(cursor.text.substr(cursor.position, 1));
    if (one_letter != 0) cursor.advance();
    return one_letter;
}

/**
 *  @brief Index of the delimiter closing the `[` or `(` at @p open_position, or `std::string_view::npos`
 *  when unbalanced.
 *
 *  Bracket and parenthesis depth are both tracked, so a recursive `$(...)` body cannot end its enclosing
 *  atom early and a branch cannot end inside one.
 */
inline std::size_t find_group_end(std::string_view text, std::size_t open_position) {
    char const closer = text[open_position] == '[' ? ']' : ')';
    std::size_t position = open_position + 1;
    int depth_bracket = 0;
    int depth_paren = 0;
    while (position < text.size() && (text[position] != closer || depth_bracket > 0 || depth_paren > 0)) {
        char const character = text[position];
        if (character == '[') ++depth_bracket;
        else if (character == ']') --depth_bracket;
        else if (character == '(') ++depth_paren;
        else if (character == ')') --depth_paren;
        ++position;
    }
    return position < text.size() ? position : std::string_view::npos;
}

/**
 *  @brief The SMARTS atom-map label out of a bracket interior such as `CX3:2`, or 0 when it carries none.
 *
 *  A colon inside a recursive `$(...)` body is that sub-query's aromatic-bond operator, so only
 *  depth-zero colons followed by digits are labels.
 */
inline std::uint8_t spec_map_number(std::string_view spec) {
    smarts_cursor_t cursor {spec};
    int depth = 0;
    while (!cursor.done()) {
        char const character = cursor.take();
        if (character == '(') ++depth;
        else if (character == ')') --depth;
        else if (character == ':' && depth == 0) {
            std::uint32_t label = 0;
            if (cursor.take_unsigned(label) && label < 256) return static_cast<std::uint8_t>(label);
        }
    }
    return 0;
}

#pragma endregion Lexer

#pragma region Atom Primitives

/**
 *  @brief Outcome of folding one SMARTS primitive run into an `atom_disjunct_t`.
 *
 *  A contradiction between two primitives of the same conjunction — `r3` next to `r{5-6}`, two different
 *  `X<n>` — is not a parse failure: the disjunct matches nothing and is dropped while the rest of the
 *  disjunction stands. Syntax the compiler cannot represent refuses the whole pattern instead.
 */
enum term_parse_t : std::uint8_t {
    /** Primitives folded in, the disjunct is satisfiable. */
    term_parsed_k = 0,
    /** Two primitives contradict, drop this disjunct. */
    term_unsatisfiable_k = 1,
    /** Syntax outside the compiler's vocabulary. */
    term_unsupported_k = 2,
};

/** Whether a `!` immediately preceded the primitive under the cursor. */
enum class primitive_sense_t : std::uint8_t { positive_k, negated_k };

/** What a term's own symbols said about aromaticity: `c` and `n` are aromatic, `C` and `N` aliphatic, and
 *  a term naming both constrains neither. */
enum class aromaticity_cue_t : std::uint8_t { unconstrained_k, aromatic_k, aliphatic_k, conflicting_k };

/** SMARTS `^<n>` names a hybridization by its shell count, so the level indexes this table directly. */
inline constexpr std::uint8_t hybridization_by_level_k[] = {
    0, hybridization_sp_k, hybridization_sp2_k, hybridization_sp3_k, hybridization_sp3d_k, hybridization_sp3d2_k,
};

inline bool compile_recursive_body(std::string_view body, std::uint8_t nesting, recursive_pool_t &pool,
                                   std::uint8_t &root);

/**
 *  @brief One conjunction of SMARTS atom primitives, in the middle of being folded into a disjunct.
 *
 *  Two readings cannot be settled until the term ends: a bare `H` is the hydrogen atom only if nothing
 *  else named an element, and an element symbol's case constrains aromaticity only if the term did not
 *  also name the other case. Those are what the cues carry, and `resolve` is where they land.
 *
 *  The disjunct is shared with the term's siblings under `;`, so a mask already in it was written by an
 *  earlier conjunct of the same combination and every fold intersects rather than overwrites.
 */
struct atom_term_t {
    atom_disjunct_t &disjunct;
    aromaticity_cue_t aromaticity = aromaticity_cue_t::unconstrained_k;
    /** `*` appeared, so a bare `H` reads as an H-count rather than as the element. */
    bool matched_any_element = false;
    /** `H` with no digit appeared, and waits on the element cues. */
    bool saw_bare_hydrogen = false;

    constexpr void note_aromaticity(aromaticity_cue_t observed) noexcept {
        if (aromaticity == aromaticity_cue_t::unconstrained_k) aromaticity = observed;
        else if (aromaticity != observed) aromaticity = aromaticity_cue_t::conflicting_k;
    }

    /**
     *  @brief Intersect the disjunct's element set with one element primitive.
     *
     *  The set is an include-list until a negated primitive turns it into an exclude-list, because `!#1`
     *  has to keep admitting the elements past bit 63 that no include-list can name. An include-list
     *  narrowed to nothing is unsatisfiable rather than unconstrained, which is the difference between
     *  `[C;!#6]` matching no atom and it matching every atom.
     */
    term_parse_t fold_element(std::uint8_t element, primitive_sense_t sense) noexcept {
        if (element == 0 || element >= 64) return term_unsupported_k;
        std::uint64_t const bit = 1ULL << element;
        bool const first = disjunct.element_mask == 0;
        if (sense == primitive_sense_t::negated_k) {
            if (first) {
                disjunct.element_mask = bit;
                disjunct.element_mask_exclusive = 1;
                return term_parsed_k;
            }
            if (disjunct.element_mask_exclusive) {
                disjunct.element_mask |= bit;
                return term_parsed_k;
            }
            disjunct.element_mask &= ~bit;
            return disjunct.element_mask == 0 ? term_unsatisfiable_k : term_parsed_k;
        }
        if (first) {
            disjunct.element_mask = bit;
            disjunct.element_mask_exclusive = 0;
            return term_parsed_k;
        }
        // Both remaining cases narrow to the single named element, and disagree only on whether the
        // set that stood before already ruled it out.
        bool const held = (disjunct.element_mask & bit) != 0;
        if (held == (disjunct.element_mask_exclusive != 0)) return term_unsatisfiable_k;
        disjunct.element_mask = bit;
        disjunct.element_mask_exclusive = 0;
        return term_parsed_k;
    }

    /** `#<n>` — element by atomic number. */
    term_parse_t fold_atomic_number(smarts_cursor_t &cursor, primitive_sense_t sense) noexcept {
        cursor.advance();
        std::uint32_t atomic_number = 0;
        if (!cursor.take_unsigned(atomic_number) || atomic_number >= 64) return term_unsupported_k;
        return fold_element(static_cast<std::uint8_t>(atomic_number), sense);
    }

    /** A bare element symbol, whose case is also what constrains aromaticity. */
    term_parse_t fold_symbol(smarts_cursor_t &cursor, primitive_sense_t sense) noexcept {
        bool const lowercase = is_lower_(cursor.peek());
        std::uint8_t const element = take_element_symbol(cursor);
        if (element == 0) return term_unsupported_k;
        // `[!O]` leaves aromaticity alone on purpose, so that an aromatic oxygen still reads as
        // "not aliphatic oxygen".
        if (sense == primitive_sense_t::positive_k)
            note_aromaticity(lowercase ? aromaticity_cue_t::aromatic_k : aromaticity_cue_t::aliphatic_k);
        return fold_element(element, sense);
    }

    /** `*`, `A` and `a` — the wildcard and the two aromaticity classes, none of them negatable. */
    term_parse_t fold_wildcard(smarts_cursor_t &cursor, primitive_sense_t sense) noexcept {
        char const character = cursor.take();
        if (sense == primitive_sense_t::negated_k) return term_unsupported_k;
        if (character == 'A') note_aromaticity(aromaticity_cue_t::aliphatic_k);
        else if (character == 'a') note_aromaticity(aromaticity_cue_t::aromatic_k);
        else matched_any_element = true;
        return term_parsed_k;
    }

    /** `X<n>` and `D<n>` — total degree. On an `addHs`-ed molecule RDKit's `getDegree()` answers both,
     *  so the two primitives pin the same field. */
    term_parse_t fold_degree(smarts_cursor_t &cursor, primitive_sense_t sense) noexcept {
        cursor.advance();
        if (sense == primitive_sense_t::negated_k) return term_unsupported_k;
        std::uint32_t degree = 0;
        if (!cursor.take_unsigned(degree) || degree >= 0xFF) return term_unsupported_k;
        if (disjunct.degree != 0 && disjunct.degree != degree) return term_unsatisfiable_k;
        disjunct.degree = static_cast<std::uint8_t>(degree);
        return term_parsed_k;
    }

    /** `x<n>` — the number of the atom's bonds that lie in a ring. `x0` is a real constraint, which is
     *  why the unconstrained sentinel is `no_count_k` rather than zero. */
    term_parse_t fold_ring_bond_degree(smarts_cursor_t &cursor, primitive_sense_t sense) noexcept {
        cursor.advance();
        if (sense == primitive_sense_t::negated_k) return term_unsupported_k;
        std::uint32_t ring_bond_degree = 0;
        if (!cursor.take_unsigned(ring_bond_degree) || ring_bond_degree >= no_count_k) return term_unsupported_k;
        if (disjunct.ring_bond_degree != no_count_k && disjunct.ring_bond_degree != ring_bond_degree)
            return term_unsatisfiable_k;
        disjunct.ring_bond_degree = static_cast<std::uint8_t>(ring_bond_degree);
        if (ring_bond_degree != 0) disjunct.require_flags |= atom_flag_in_ring_k;
        return term_parsed_k;
    }

    /** `H`, `H<n>` and `!H<n>` — the attached-hydrogen count. */
    term_parse_t fold_hydrogen_count(smarts_cursor_t &cursor, primitive_sense_t sense) noexcept {
        cursor.advance();
        std::uint32_t count = 1;
        bool const explicit_count = cursor.take_unsigned(count);
        if (count >= no_count_k) return term_unsupported_k;
        if (sense == primitive_sense_t::negated_k) {
            // RDKit reads `[!H]` as `!H1` rather than "not hydrogen": the bare-`H`-is-the-element
            // reading survives only in the positive `[H]`, which it writes `[#1]`.
            if (disjunct.forbid_h_count != no_count_k && disjunct.forbid_h_count != count) return term_unsupported_k;
            disjunct.forbid_h_count = static_cast<std::uint8_t>(count);
            return term_parsed_k;
        }
        if (!explicit_count) {
            saw_bare_hydrogen = true;
            return term_parsed_k;
        }
        if (disjunct.exact_h_count != no_count_k && disjunct.exact_h_count != count) return term_unsatisfiable_k;
        disjunct.exact_h_count = static_cast<std::uint8_t>(count);
        return term_parsed_k;
    }

    /** `R`, `R<n>` and `!R` — SSSR ring membership by count. A negated count is a different query. */
    term_parse_t fold_ring_count(smarts_cursor_t &cursor, primitive_sense_t sense) noexcept {
        cursor.advance();
        if (sense == primitive_sense_t::negated_k) {
            if (is_digit_(cursor.peek())) return term_unsupported_k;
            disjunct.forbid_flags |= atom_flag_in_ring_k;
            return term_parsed_k;
        }
        std::uint32_t ring_count = 0;
        if (!cursor.take_unsigned(ring_count)) {
            // Bare `R` — same as SMARTS: the atom is in at least one ring.
            disjunct.require_flags |= atom_flag_in_ring_k;
            return term_parsed_k;
        }
        if (ring_count == 0) {
            disjunct.forbid_flags |= atom_flag_in_ring_k;
            return term_parsed_k;
        }
        if (ring_count > 255) return term_unsupported_k;
        if (disjunct.ring_count != 0 && disjunct.ring_count != ring_count) return term_unsatisfiable_k;
        disjunct.ring_count = static_cast<std::uint8_t>(ring_count);
        disjunct.require_flags |= atom_flag_in_ring_k;
        return term_parsed_k;
    }

    /** `r`, `r<n>`, `r{N-M}` and `!r` — the size of the atom's smallest ring, as a range. Several in one
     *  conjunction intersect: `r5` beside `r{5-8}` is `r5`, and `r3` beside `r{5-6}` is nothing at all. */
    term_parse_t fold_ring_size(smarts_cursor_t &cursor, primitive_sense_t sense) noexcept {
        cursor.advance();
        if (sense == primitive_sense_t::negated_k) {
            // A negated ring size, `!r5`, is a different query and stays outside the vocabulary.
            if (is_digit_(cursor.peek()) || cursor.peek() == '{') return term_unsupported_k;
            disjunct.forbid_flags |= atom_flag_in_ring_k;
            return term_parsed_k;
        }

        std::uint32_t lower_ring_size = 3; // `r{-M}` leaves the lower bound open at the smallest ring
        std::uint32_t upper_ring_size = 255;
        if (cursor.peek() == '{') {
            cursor.advance();
            [[maybe_unused]] bool const read_lower = cursor.take_unsigned(lower_ring_size);
            if (cursor.take() != '-') return term_unsupported_k;
            [[maybe_unused]] bool const read_upper = cursor.take_unsigned(upper_ring_size);
            if (cursor.take() != '}') return term_unsupported_k;
            if (lower_ring_size == 0) lower_ring_size = 3;
            if (upper_ring_size == 0) upper_ring_size = 255;
            if (lower_ring_size > upper_ring_size || upper_ring_size > 255) return term_unsupported_k;
        }
        else if (is_digit_(cursor.peek())) {
            if (!cursor.take_unsigned(lower_ring_size)) return term_unsupported_k;
            if (lower_ring_size == 0 || lower_ring_size > 255) return term_unsupported_k;
            upper_ring_size = lower_ring_size;
        }
        else {
            // Bare `r` — same as bare `R`: the atom is in at least one ring.
            disjunct.require_flags |= atom_flag_in_ring_k;
            return term_parsed_k;
        }

        if (disjunct.ring_size != 0) {
            if (disjunct.ring_size > lower_ring_size) lower_ring_size = disjunct.ring_size;
            if (disjunct.ring_size_max < upper_ring_size) upper_ring_size = disjunct.ring_size_max;
            if (lower_ring_size > upper_ring_size) return term_unsatisfiable_k;
        }
        disjunct.ring_size = static_cast<std::uint8_t>(lower_ring_size);
        disjunct.ring_size_max = static_cast<std::uint8_t>(upper_ring_size);
        disjunct.require_flags |= atom_flag_in_ring_k;
        return term_parsed_k;
    }

    /** `+`, `-`, `+<n>` and `-<n>` — the formal charge. */
    term_parse_t fold_charge(smarts_cursor_t &cursor, primitive_sense_t sense) noexcept {
        std::int8_t const sign = cursor.take() == '+' ? static_cast<std::int8_t>(1) : static_cast<std::int8_t>(-1);
        if (sense == primitive_sense_t::negated_k) return term_unsupported_k;
        std::uint32_t magnitude = 1;
        if (cursor.take_unsigned(magnitude) && (magnitude == 0 || magnitude > 8)) return term_unsupported_k;
        std::int8_t const charge = static_cast<std::int8_t>(sign * static_cast<std::int8_t>(magnitude));
        if (disjunct.formal_charge != no_formal_charge_k && disjunct.formal_charge != charge)
            return term_unsatisfiable_k;
        disjunct.formal_charge = charge;
        return term_parsed_k;
    }

    /** `^<n>` — hybridization by shell count. */
    term_parse_t fold_hybridization(smarts_cursor_t &cursor, primitive_sense_t sense) noexcept {
        cursor.advance();
        if (sense == primitive_sense_t::negated_k) return term_unsupported_k;
        std::uint32_t level = 0;
        if (!cursor.take_unsigned(level) || level == 0 || level >= std::size(hybridization_by_level_k))
            return term_unsupported_k;
        std::uint8_t const requested = hybridization_by_level_k[level];
        if (disjunct.hybridization != 0 && disjunct.hybridization != requested) return term_unsatisfiable_k;
        disjunct.hybridization = requested;
        return term_parsed_k;
    }

    /** `$(...)` and `!$(...)` — a recursive sub-query compiled into @p pool. */
    term_parse_t fold_recursive(smarts_cursor_t &cursor, primitive_sense_t sense, std::uint8_t nesting,
                                recursive_pool_t &pool) {
        cursor.advance();
        if (cursor.peek() != '(') return term_unsupported_k;
        std::size_t const body_end = find_group_end(cursor.text, cursor.position);
        if (body_end == std::string_view::npos) return term_unsupported_k;
        // Two sub-queries in one conjunction would need two guards on one disjunct, which the corpus
        // never writes.
        if (disjunct.recursive_root != no_recursive_node_k) return term_unsupported_k;
        std::uint8_t root = no_recursive_node_k;
        std::string_view const body = cursor.text.substr(cursor.position + 1, body_end - cursor.position - 1);
        if (!compile_recursive_body(body, static_cast<std::uint8_t>(nesting + 1), pool, root))
            return term_unsupported_k;
        disjunct.recursive_root = root;
        disjunct.recursive_negate = sense == primitive_sense_t::negated_k ? 1 : 0;
        cursor.position = body_end + 1;
        return term_parsed_k;
    }

    /** Settle the two readings that needed the whole term, and fold the aromaticity cue. */
    term_parse_t resolve() noexcept {
        if (saw_bare_hydrogen) {
            // `[H]` is the hydrogen atom when nothing else in the term named an element, and the
            // implicit H-count-of-one modifier otherwise, which is how `[CH]` reads.
            bool const named_element = disjunct.element_mask != 0 || matched_any_element ||
                                       aromaticity != aromaticity_cue_t::unconstrained_k;
            if (!named_element) disjunct.element_mask = 1ULL << 1;
            else if (disjunct.exact_h_count != no_count_k && disjunct.exact_h_count != 1) return term_unsatisfiable_k;
            else disjunct.exact_h_count = 1;
        }
        if (disjunct.forbid_h_count != no_count_k && disjunct.exact_h_count == disjunct.forbid_h_count)
            return term_unsatisfiable_k;
        if (aromaticity == aromaticity_cue_t::aromatic_k) disjunct.require_flags |= atom_flag_aromatic_k;
        else if (aromaticity == aromaticity_cue_t::aliphatic_k) disjunct.forbid_flags |= atom_flag_aromatic_k;
        return term_parsed_k;
    }
};

/**
 *  @brief Parse one disjunct — the text between `,` separators inside a SMARTS bracket — into
 *  @p disjunct, which the term intersects into rather than overwrites.
 *
 *  A term is a conjunction of primitives separated by `&`/`;` or by nothing at all, and the whole of it
 *  shares one element identity, one aromaticity, one H-count. @p nesting counts the `$(...)` levels
 *  enclosing the term, zero at a pattern's own atoms.
 */
inline term_parse_t parse_atom_term(std::string_view token, std::uint8_t nesting, atom_disjunct_t &disjunct,
                                    recursive_pool_t &pool) {
    smarts_cursor_t cursor {token};
    atom_term_t term {.disjunct = disjunct};
    primitive_sense_t sense = primitive_sense_t::positive_k;

    while (!cursor.done()) {
        term_parse_t outcome = term_parsed_k;
        switch (cursor.peek()) {
        // The term is already one alternative of one conjunct, so `&` and `;` coincide here.
        case '&':
        case ';': cursor.advance(); continue;
        case '!':
            cursor.advance();
            sense = primitive_sense_t::negated_k;
            continue;
        // A depth-zero `:n` is the atom map label, which binds a torsion slot rather than the atom.
        case ':':
            cursor.advance();
            cursor.skip_digits();
            continue;
        case '$': outcome = term.fold_recursive(cursor, sense, nesting, pool); break;
        case '*':
        case 'A':
        case 'a': outcome = term.fold_wildcard(cursor, sense); break;
        case '#': outcome = term.fold_atomic_number(cursor, sense); break;
        case 'X':
        case 'D': outcome = term.fold_degree(cursor, sense); break;
        case 'x': outcome = term.fold_ring_bond_degree(cursor, sense); break;
        case 'H': outcome = term.fold_hydrogen_count(cursor, sense); break;
        case 'R': outcome = term.fold_ring_count(cursor, sense); break;
        case 'r': outcome = term.fold_ring_size(cursor, sense); break;
        case '+':
        case '-': outcome = term.fold_charge(cursor, sense); break;
        case '^': outcome = term.fold_hybridization(cursor, sense); break;
        default:
            if (!is_alpha_(cursor.peek())) return term_unsupported_k;
            outcome = term.fold_symbol(cursor, sense);
            break;
        }
        if (outcome != term_parsed_k) return outcome;
        sense = primitive_sense_t::positive_k;
    }
    return term.resolve();
}

#pragma endregion Atom Primitives

#pragma region Expressions

/**
 *  @brief Split @p text on every @p separator sitting at parenthesis depth zero, answering 0 when a piece
 *  comes out empty or @p max_fields is exceeded.
 *
 *  Separators inside a recursive `$(...)` body belong to that sub-query, so `[$(C,N);X3]` has one
 *  top-level `;` and no top-level `,`.
 */
inline std::size_t split_top_level(std::string_view text, char separator, std::string_view *fields,
                                   std::size_t max_fields) {
    std::size_t num_fields = 0;
    std::size_t position = 0;
    while (position < text.size()) {
        std::size_t const start = position;
        int depth = 0;
        while (position < text.size() && !(text[position] == separator && depth == 0)) {
            if (text[position] == '(') ++depth;
            else if (text[position] == ')') --depth;
            ++position;
        }
        if (num_fields >= max_fields) return 0;
        std::string_view const field = text.substr(start, position - start);
        if (field.empty()) return 0;
        fields[num_fields++] = field;
        if (position < text.size()) ++position; // skip the separator
    }
    return num_fields;
}

/**
 *  @brief The disjunctive-normal-form skeleton of a bracket interior or a bond operator, plus the cursor
 *  that walks its combinations.
 *
 *  SMARTS logical-operator precedence is `!` then `&` then `,` then `;`, so an expression is a
 *  conjunction of alternations and its DNF is the cross product of one alternative per conjunct. The atom
 *  and bond parsers differ in what a combination means, not in how the combinations are enumerated.
 */
struct dnf_skeleton_t {
    static constexpr std::size_t max_conjuncts_k = 6;
    static constexpr std::size_t max_alternatives_k = 4;

    std::string_view alternative[max_conjuncts_k][max_alternatives_k];
    std::size_t num_alternatives[max_conjuncts_k] = {};
    std::size_t num_conjuncts = 0;
    std::size_t num_combinations = 0;
    std::size_t chosen[max_conjuncts_k] = {};

    /** Split @p text into conjuncts and alternatives, refusing anything that would need more room than
     *  the skeleton or the caller's @p max_combinations has. */
    bool split(std::string_view text, std::size_t max_combinations) {
        num_conjuncts = split_top_level(text, ';', conjunct_, max_conjuncts_k);
        if (num_conjuncts == 0) return false;
        num_combinations = 1;
        for (std::size_t conjunct = 0; conjunct < num_conjuncts; ++conjunct) {
            num_alternatives[conjunct] = split_top_level(conjunct_[conjunct], ',', alternative[conjunct],
                                                         max_alternatives_k);
            if (num_alternatives[conjunct] == 0) return false;
            num_combinations *= num_alternatives[conjunct];
        }
        return num_combinations <= max_combinations;
    }

    /** The alternative @p conjunct contributes to the combination the cursor stands on. */
    std::string_view chosen_alternative(std::size_t conjunct) const { return alternative[conjunct][chosen[conjunct]]; }

    /** Step the mixed-radix cursor to the next combination. */
    void advance() {
        for (std::size_t conjunct = 0; conjunct < num_conjuncts; ++conjunct) {
            if (++chosen[conjunct] < num_alternatives[conjunct]) break;
            chosen[conjunct] = 0;
        }
    }

  private:
    std::string_view conjunct_[max_conjuncts_k];
};

/**
 *  @brief Parse the interior of a SMARTS atom bracket into a `simple_atom_constraint_t`.
 *
 *  A mixed AND/OR spec becomes the cross product of its alternatives, and a recursive `$(...)` rides
 *  along in whichever combination it was written in, so `[$([NX3H1]),$([NX3H2])]` yields two disjuncts
 *  each carrying its own guard. A combination whose primitives contradict drops out of the disjunction.
 */
inline bool parse_atom_spec(std::string_view spec, std::uint8_t nesting, simple_atom_constraint_t &constraint,
                            recursive_pool_t &pool) {
    constraint = {};
    if (spec.empty()) return false;
    if (spec.find('|') != std::string_view::npos) return false;

    dnf_skeleton_t skeleton;
    if (!skeleton.split(spec, max_disjuncts_k)) return false;

    for (std::size_t combination = 0; combination < skeleton.num_combinations; ++combination, skeleton.advance()) {
        atom_disjunct_t &disjunct = constraint.disjuncts[constraint.num_disjuncts];
        disjunct = {};
        bool satisfiable = true;
        for (std::size_t conjunct = 0; conjunct < skeleton.num_conjuncts && satisfiable; ++conjunct) {
            switch (parse_atom_term(skeleton.chosen_alternative(conjunct), nesting, disjunct, pool)) {
            case term_parsed_k: break;
            case term_unsatisfiable_k: satisfiable = false; break;
            case term_unsupported_k: return false;
            }
        }
        if (satisfiable) ++constraint.num_disjuncts;
    }
    return constraint.num_disjuncts != 0;
}

/** Parse one bond-operator conjunction — the text between `,` separators — into a single
 *  `bond_disjunct_t`. `/` and `\` carry directional-bond stereo, which no loaded pattern uses, so they
 *  stay outside the vocabulary. */
inline bool parse_bond_conjunct(std::string_view bond_operator, bond_disjunct_t &bond_disjunct) {
    bond_disjunct = {};
    smarts_cursor_t cursor {bond_operator};
    bool negate = false;
    while (!cursor.done()) {
        char const character = cursor.take();
        if (character == ';') continue;
        if (character == '!') {
            negate = true;
            continue;
        }
        // `@` is the only primitive a `!` may precede; a negated bond order is outside the vocabulary.
        if (character == '@') {
            if (negate) bond_disjunct.forbid_flags |= bond_flag_in_ring_k;
            else bond_disjunct.require_flags |= bond_flag_in_ring_k;
            negate = false;
            continue;
        }
        if (negate) return false;
        std::uint8_t order = 0;
        switch (character) {
        case '-':
            order = 1;
            bond_disjunct.forbid_flags |= bond_flag_aromatic_k;
            break;
        case '=': order = 2; break;
        case '#': order = 3; break;
        // Aromatic bonds are stored as order-1 with the aromatic flag in the descriptor; pinning the
        // order keeps a stray order-3 aromatised edge from pathological SMILES from sneaking through.
        case ':':
            order = 1;
            bond_disjunct.require_flags |= bond_flag_aromatic_k;
            break;
        case '~': continue; // any bond, so nothing to pin
        default: return false;
        }
        if (bond_disjunct.bond_order != 0 && bond_disjunct.bond_order != order) return false;
        bond_disjunct.bond_order = order;
    }
    return true;
}

/**
 *  @brief Parse a bond operator into `simple_bond_constraint_t`, flattened to disjunctive normal form the
 *  same way `parse_atom_spec` flattens an atom bracket.
 *
 *  Empty @p bond_operator is the SMARTS default "single or aromatic", written as two disjuncts that both
 *  pin the order to 1 so a triple bond carrying an incidental aromatic flag cannot slip through.
 */
inline bool parse_bond_op(std::string_view bond_operator, simple_bond_constraint_t &bond_constraint) {
    bond_constraint = {};
    if (bond_operator.empty()) {
        bond_constraint.num_disjuncts = 2;
        bond_constraint.disjuncts[0].bond_order = 1;
        bond_constraint.disjuncts[0].forbid_flags = bond_flag_aromatic_k;
        bond_constraint.disjuncts[1].bond_order = 1;
        bond_constraint.disjuncts[1].require_flags = bond_flag_aromatic_k;
        return true;
    }

    dnf_skeleton_t skeleton;
    if (!skeleton.split(bond_operator, max_disjuncts_k)) return false;

    for (std::size_t combination = 0; combination < skeleton.num_combinations; ++combination, skeleton.advance()) {
        bond_disjunct_t &disjunct = bond_constraint.disjuncts[bond_constraint.num_disjuncts];
        disjunct = {};
        for (std::size_t conjunct = 0; conjunct < skeleton.num_conjuncts; ++conjunct) {
            bond_disjunct_t next {};
            if (!parse_bond_conjunct(skeleton.chosen_alternative(conjunct), next)) return false;
            if (next.bond_order != 0) {
                if (disjunct.bond_order != 0 && disjunct.bond_order != next.bond_order) return false;
                disjunct.bond_order = next.bond_order;
            }
            disjunct.require_flags |= next.require_flags;
            disjunct.forbid_flags |= next.forbid_flags;
        }
        ++bond_constraint.num_disjuncts;
    }
    return bond_constraint.num_disjuncts > 0;
}

#pragma endregion Expressions

#pragma region Chain Scanning

/** Chain atoms plus map-labelled inline branches the scanner records before declaring a pattern too wide
 *  for the 4-atom matcher. */
inline constexpr std::size_t max_pattern_atoms_k = 8;

/** Maximum inline `(...)` branches the scanner records for a single atom. */
inline constexpr std::size_t max_atom_branches_k = 3;

/** Atoms one recursive `$(...)` body or one inline branch body may spell out. */
inline constexpr std::size_t max_recursive_body_atoms_k = 8;

/** Sentinel atom index: the parent of a pattern root, and an unbound torsion slot while the four labels
 *  are still being resolved. */
inline constexpr std::uint8_t no_atom_k = 0xFF;

/**
 *  @brief One atom of a scanned SMARTS pattern together with the edge to the atom it hangs off.
 *
 *  `parse_linear_torsion` scans a pattern into this shape first and binds the torsion atoms afterwards by
 *  their `:n` map label, so a labelled atom may sit anywhere in the text as long as the labels form a
 *  path.
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

    /** The text `parse_atom_spec` reads: a bracket's interior, or the bare symbol itself. */
    std::string_view constraint_text() const noexcept { return bracketed ? spec : text; }
};

/** Whether the cursor stands where an atom may begin rather than inside a bond operator. */
inline bool stands_on_atom_(smarts_cursor_t const &cursor) {
    char const character = cursor.peek();
    return character == '[' || character == '*' || is_alpha_(character);
}

/**
 *  @brief Read one chain atom — the bond-operator run before it and the atom itself — into @p atom,
 *  stepping @p cursor past the whole of it.
 *
 *  A digit in the operator run would put an operator on a ring closure, a parenthesis would open a branch
 *  hanging off nothing, and a `$` outside a bracket is a sub-query with no atom to guard; none of the
 *  three reads back into a chain.
 */
inline bool scan_chain_atom_(smarts_cursor_t &cursor, std::uint8_t parent, scanned_atom_t &atom) {
    atom = {};
    atom.parent = parent;

    std::size_t const operator_start = cursor.position;
    while (!cursor.done() && !stands_on_atom_(cursor)) {
        char const character = cursor.peek();
        if (is_digit_(character) || character == '(' || character == ')' || character == '$') return false;
        cursor.advance();
    }
    if (cursor.done()) return false; // a bond operator with no atom after it
    atom.bond_op = cursor.since(operator_start);

    std::size_t const atom_start = cursor.position;
    if (cursor.peek() == '[') {
        std::size_t const bracket_end = find_group_end(cursor.text, cursor.position);
        if (bracket_end == std::string_view::npos) return false;
        atom.spec = cursor.text.substr(cursor.position + 1, bracket_end - cursor.position - 1);
        atom.bracketed = 1;
        atom.map_number = spec_map_number(atom.spec);
        cursor.position = bracket_end + 1;
    }
    else if (cursor.peek() == '*' || cursor.peek() == 'A' || cursor.peek() == 'a') { cursor.advance(); }
    else if (take_element_symbol(cursor) == 0) return false;
    atom.text = cursor.since(atom_start);
    return true;
}

/**
 *  @brief Scan a SMARTS chain into a flat parent-linked atom tree, appended to @p atoms in scan order.
 *
 *  Unlike the pattern-level scan this one descends into `(...)` branches instead of keeping them as text,
 *  because a sub-query body is a tree of arbitrary shape rather than a path with leaves. Ring closures
 *  and anything else that would make the body cyclic are outside the vocabulary.
 */
inline bool scan_recursive_chain(std::string_view text, std::uint8_t parent, scanned_atom_t *atoms,
                                 std::size_t &atom_count) {
    smarts_cursor_t cursor {text};
    std::uint8_t previous = parent;
    while (!cursor.done()) {
        if (atom_count >= max_recursive_body_atoms_k) return false;
        std::size_t const current = atom_count;
        if (!scan_chain_atom_(cursor, previous, atoms[current])) return false;
        ++atom_count;
        previous = static_cast<std::uint8_t>(current);

        while (cursor.peek() == '(') {
            std::size_t const branch_end = find_group_end(cursor.text, cursor.position);
            if (branch_end == std::string_view::npos) return false;
            std::string_view const branch = cursor.text.substr(cursor.position + 1, branch_end - cursor.position - 1);
            if (!scan_recursive_chain(branch, static_cast<std::uint8_t>(current), atoms, atom_count)) return false;
            cursor.position = branch_end + 1;
        }
    }
    return true;
}

#pragma endregion Chain Scanning

#pragma region Sub Query Compilation

/** Turn one scanned atom and everything hanging off it into pool nodes, reporting the root of the
 *  subtree in @p node. @p hops is its distance in bonds from the body's own root. */
inline bool build_recursive_nodes(custd::span<scanned_atom_t const> atoms, std::uint8_t scanned_index,
                                  std::uint8_t nesting, std::uint8_t hops, recursive_pool_t &pool, std::uint8_t &node) {
    if (hops > max_recursive_hops_k) return false;
    if (pool.num_nodes >= max_recursive_nodes_k) return false;
    std::uint8_t const node_index = pool.num_nodes++;
    recursive_node_t &built = pool.nodes[node_index];
    built = {};
    built.first_child = no_recursive_node_k;
    built.next_sibling = no_recursive_node_k;

    simple_atom_constraint_t constraint {};
    if (!parse_atom_spec(atoms[scanned_index].constraint_text(), nesting, constraint, pool)) return false;
    if (constraint.num_neighbor_branches != 0) return false;
    simple_bond_constraint_t bond {};
    if (!parse_bond_op(atoms[scanned_index].bond_op, bond)) return false;

    // `pool.nodes` may have been reallocated in place by the recursive guards `parse_atom_spec` just
    // compiled, so the node is reached through the pool again rather than through `built`.
    pool.nodes[node_index].num_atom_disjuncts = constraint.num_disjuncts;
    for (std::uint8_t slot = 0; slot < constraint.num_disjuncts; ++slot)
        pool.nodes[node_index].atom_disjuncts[slot] = constraint.disjuncts[slot];
    pool.nodes[node_index].num_bond_disjuncts = bond.num_disjuncts;
    for (std::uint8_t slot = 0; slot < bond.num_disjuncts; ++slot)
        pool.nodes[node_index].bond_disjuncts[slot] = bond.disjuncts[slot];

    std::uint8_t previous_child = no_recursive_node_k;
    for (std::size_t index = 0; index < atoms.size(); ++index) {
        if (atoms[index].parent != scanned_index) continue;
        std::uint8_t child = no_recursive_node_k;
        if (!build_recursive_nodes(atoms, static_cast<std::uint8_t>(index), nesting,
                                   static_cast<std::uint8_t>(hops + 1), pool, child))
            return false;
        if (previous_child == no_recursive_node_k) pool.nodes[node_index].first_child = child;
        else pool.nodes[previous_child].next_sibling = child;
        previous_child = child;
    }
    node = node_index;
    return true;
}

/** Compile the text between a `$(` and its matching `)` into @p pool and report the root node.
 *  @p nesting counts the `$(...)` levels enclosing this one, the outermost being 1. */
inline bool compile_recursive_body(std::string_view body, std::uint8_t nesting, recursive_pool_t &pool,
                                   std::uint8_t &root) {
    if (nesting > max_recursive_depth_k) return false;
    if (body.empty()) return false;
    scanned_atom_t atoms[max_recursive_body_atoms_k];
    std::size_t atom_count = 0;
    if (!scan_recursive_chain(body, no_atom_k, atoms, atom_count)) return false;
    if (atom_count == 0) return false;
    return build_recursive_nodes({atoms, atom_count}, 0, nesting, 0, pool, root);
}

/**
 *  @brief Parse the content of an inline `(...)` branch, parentheses already stripped, into a neighbour
 *  constraint.
 *
 *  A leading bond operator applies to the edge leaving the torsion atom and stays on the constraint;
 *  everything after it compiles into the sub-query pool, so a branch may be a chain, may fan out, and may
 *  carry `$(...)` guards of its own.
 */
inline bool parse_branch_spec(std::string_view branch, recursive_pool_t &pool,
                              simple_neighbor_constraint_t &constraint) {
    constraint = {};
    constraint.node = no_recursive_node_k;
    if (branch.empty()) return false;

    smarts_cursor_t cursor {branch};
    while (!cursor.done() && !stands_on_atom_(cursor)) cursor.advance();
    if (cursor.done()) return false; // a bond operator with no atom
    simple_bond_constraint_t bond {};
    if (!parse_bond_op(branch.substr(0, cursor.position), bond)) return false;

    std::uint8_t root = no_recursive_node_k;
    if (!compile_recursive_body(branch.substr(cursor.position), 0, pool, root)) return false;

    constraint.node = root;
    constraint.num_bond_disjuncts = bond.num_disjuncts;
    for (std::uint8_t slot = 0; slot < bond.num_disjuncts; ++slot)
        constraint.bond_disjuncts[slot] = bond.disjuncts[slot];
    return true;
}

#pragma endregion Sub Query Compilation

#pragma region Torsion Compilation

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

/** Everything the scanner reads out of a pattern's text before any of it is compiled. */
struct scanned_pattern_t {
    scanned_atom_t atom[max_pattern_atoms_k];
    std::size_t num_atoms = 0;
    /** The one ring-closure edge the matcher can represent; `no_atom_k` at both ends when absent. */
    std::uint8_t closure_first = no_atom_k;
    std::uint8_t closure_second = no_atom_k;
};

/** The four labelled atoms of a scanned pattern, as scanned indices, and the edges joining them. */
struct labelled_path_t {
    std::uint8_t atom[4] = {no_atom_k, no_atom_k, no_atom_k, no_atom_k};
    std::string_view bond_op[3];
};

/** The route a ring-closing pattern takes from `:4` back to `:1`, as scanned indices with both labelled
 *  endpoints included. Length zero means the pattern closes no ring. */
struct scanned_arc_t {
    std::uint8_t atom[max_ring_closure_atoms_k + 2] = {};
    std::uint8_t length = 0;

    constexpr bool holds(std::size_t scanned_index) const noexcept {
        bool found = false;
        for (std::uint8_t step = 0; step < length; ++step) found = found || atom[step] == scanned_index;
        return found;
    }
};

/** The inline branches gathered for one labelled atom, as the text each of them compiles from. */
struct atom_branches_t {
    std::string_view text[2];
    std::uint8_t count = 0;

    /** @return False when the atom already carries the two branches a constraint can hold. */
    bool add(std::string_view branch) {
        if (count >= 2) return false;
        text[count++] = branch;
        return true;
    }
};

/**
 *  @brief Scan a torsion pattern's text into atoms, ring-closure ends and per-atom branch text.
 *
 *  A branch holding a map label becomes a scanned atom of its own so that label can join the torsion
 *  path; every other branch stays verbatim text for `parse_branch_spec`.
 */
inline bool scan_torsion_pattern_(std::string_view smarts, scanned_pattern_t &scanned) {
    smarts_cursor_t cursor {smarts};
    std::uint8_t previous_chain_atom = no_atom_k;
    // One open slot per ring-bond digit. A second closed digit would draw a fused system.
    std::uint8_t closure_open[10];
    for (auto &slot : closure_open) slot = no_atom_k;

    while (!cursor.done()) {
        if (scanned.num_atoms >= max_pattern_atoms_k) return false;
        std::size_t const current = scanned.num_atoms;
        if (!scan_chain_atom_(cursor, previous_chain_atom, scanned.atom[current])) return false;
        ++scanned.num_atoms;
        previous_chain_atom = static_cast<std::uint8_t>(current);

        // Ring-closure digits and inline `(...)` branches, in whichever order the text writes them.
        while (!cursor.done()) {
            if (is_digit_(cursor.peek())) {
                std::uint8_t const digit = static_cast<std::uint8_t>(cursor.take() - '0');
                if (closure_open[digit] == no_atom_k) {
                    closure_open[digit] = static_cast<std::uint8_t>(current);
                    continue;
                }
                if (scanned.closure_first != no_atom_k) return false; // a second closure fuses two cycles
                scanned.closure_first = closure_open[digit];
                scanned.closure_second = static_cast<std::uint8_t>(current);
                closure_open[digit] = no_atom_k;
                continue;
            }
            if (cursor.peek() != '(') break;
            std::size_t const branch_end = find_group_end(cursor.text, cursor.position);
            if (branch_end == std::string_view::npos) return false;
            std::string_view const branch = cursor.text.substr(cursor.position + 1, branch_end - cursor.position - 1);
            cursor.position = branch_end + 1;

            // A labelled branch is one bond operator plus one bracketed atom and nothing else, as
            // `=[N:4]` is; it joins the scan so its label can bind a torsion slot.
            smarts_cursor_t branch_cursor {branch};
            while (!branch_cursor.done() && !stands_on_atom_(branch_cursor)) branch_cursor.advance();
            std::uint8_t label = 0;
            std::string_view labelled_spec;
            if (branch_cursor.peek() == '[') {
                std::size_t const bracket_end = find_group_end(branch, branch_cursor.position);
                if (bracket_end + 1 == branch.size()) {
                    labelled_spec = branch.substr(branch_cursor.position + 1, bracket_end - branch_cursor.position - 1);
                    label = spec_map_number(labelled_spec);
                }
            }

            if (label != 0) {
                if (scanned.num_atoms >= max_pattern_atoms_k) return false;
                scanned_atom_t &labelled = scanned.atom[scanned.num_atoms++];
                labelled = {};
                labelled.parent = static_cast<std::uint8_t>(current);
                labelled.bond_op = branch.substr(0, branch_cursor.position);
                labelled.spec = labelled_spec;
                labelled.text = branch.substr(branch_cursor.position);
                labelled.bracketed = 1;
                labelled.map_number = label;
                continue;
            }

            scanned_atom_t &host = scanned.atom[current];
            if (host.num_branches >= max_atom_branches_k) return false;
            host.branch_text[host.num_branches++] = branch;
        }
    }

    for (std::uint8_t const slot : closure_open)
        if (slot != no_atom_k) return false; // a ring digit opened and never closed
    return true;
}

/**
 *  @brief Bind the four torsion atoms by their map labels and demand the 1-2, 2-3 and 3-4 edges.
 *
 *  Where a labelled atom sits in the text does not matter, so `[a:1][c:2]!@;-[C:3](=[N:4])(-N)` binds
 *  even though `:4` lives inside a branch — only the labels have to form a connected path.
 */
inline bool bind_labelled_path_(scanned_pattern_t const &scanned, labelled_path_t &path) {
    for (std::size_t index = 0; index < scanned.num_atoms; ++index) {
        std::uint8_t const label = scanned.atom[index].map_number;
        if (label < 1 || label > 4) continue;
        if (path.atom[label - 1] != no_atom_k) return false; // the same label twice
        path.atom[label - 1] = static_cast<std::uint8_t>(index);
    }
    for (std::uint8_t const bound : path.atom)
        if (bound == no_atom_k) return false;

    for (std::size_t step = 0; step < 3; ++step) {
        std::uint8_t const first = path.atom[step];
        std::uint8_t const second = path.atom[step + 1];
        if (scanned.atom[second].parent == first) path.bond_op[step] = scanned.atom[second].bond_op;
        else if (scanned.atom[first].parent == second) path.bond_op[step] = scanned.atom[first].bond_op;
        else return false; // the labels do not form a connected path
    }
    return true;
}

/** Index of the edge joining @p first and @p second, or `edges.size()` when they are not joined. */
inline std::size_t find_scanned_edge_(custd::span<scanned_edge_t const> edges, std::uint8_t first,
                                      std::uint8_t second) {
    for (std::size_t edge = 0; edge < edges.size(); ++edge) {
        bool const forward = edges[edge].first == first && edges[edge].second == second;
        bool const backward = edges[edge].second == first && edges[edge].first == second;
        if (forward || backward) return edge;
    }
    return edges.size();
}

/**
 *  @brief Depth-first walk for the arc that closes a ring pattern, from the `:4` the caller seeded
 *  @p arc with to the `:1` in @p target.
 *
 *  The scanned pattern is a tree plus at most one closure edge, so exactly one cycle exists; with the
 *  three labelled-path edges struck out of @p edges, the route from `:4` back to `:1` is the rest of that
 *  cycle and nothing else reaches it.
 */
inline bool walk_closure_arc(custd::span<scanned_edge_t const> edges, std::uint8_t target, scanned_arc_t &arc) {
    std::uint8_t const current = arc.atom[arc.length - 1];
    if (current == target) return true;
    if (arc.length >= std::size(arc.atom)) return false;
    for (auto const &edge : edges) {
        if (edge.forbidden) continue;
        std::uint8_t next = no_atom_k;
        if (edge.first == current) next = edge.second;
        else if (edge.second == current) next = edge.first;
        else continue;
        if (arc.holds(next)) continue;
        arc.atom[arc.length++] = next;
        if (walk_closure_arc(edges, target, arc)) return true;
        --arc.length;
    }
    return false;
}

/**
 *  @brief Compile the ring a closing pattern draws from `:4` back to `:1` into @p pattern, reporting the
 *  scanned indices it runs through in @p arc, `:4` first and `:1` last.
 *
 *  The labelled path plus the closure edge draws exactly one cycle, so the arc is whatever of that cycle
 *  the labels do not already cover. A pattern that closes no ring succeeds with an empty @p arc.
 */
inline bool compile_ring_closure_(scanned_pattern_t const &scanned, labelled_path_t const &path, scanned_arc_t &arc,
                                  simple_torsion_pattern_t &pattern) {
    if (scanned.closure_first == no_atom_k) return true;

    scanned_edge_t edges[max_pattern_atoms_k];
    std::size_t edge_count = 0;
    for (std::size_t index = 0; index < scanned.num_atoms; ++index) {
        if (scanned.atom[index].parent == no_atom_k) continue;
        if (edge_count >= max_pattern_atoms_k) return false;
        edges[edge_count++] = {.bond_op = scanned.atom[index].bond_op,
                               .first = static_cast<std::uint8_t>(index),
                               .second = scanned.atom[index].parent,
                               .forbidden = 0,
                               .padding = 0};
    }
    if (edge_count >= max_pattern_atoms_k) return false;
    edges[edge_count++] = {.bond_op = {},
                           .first = scanned.closure_first,
                           .second = scanned.closure_second, //
                           .forbidden = 0,
                           .padding = 0};

    custd::span<scanned_edge_t const> const all_edges {edges, edge_count};
    for (std::size_t step = 0; step < 3; ++step) {
        std::size_t const edge = find_scanned_edge_(all_edges, path.atom[step], path.atom[step + 1]);
        if (edge != edge_count) edges[edge].forbidden = 1;
    }

    arc.atom[0] = path.atom[3];
    arc.length = 1;
    if (!walk_closure_arc(all_edges, path.atom[0], arc)) return false;

    pattern.closure.active = 1;
    pattern.closure.num_atoms = static_cast<std::uint8_t>(arc.length - 2);
    for (std::uint8_t step = 0; step + 1 < arc.length; ++step) {
        std::size_t const edge = find_scanned_edge_(all_edges, arc.atom[step], arc.atom[step + 1]);
        if (edge == edge_count) return false;
        simple_bond_constraint_t bond {};
        if (!parse_bond_op(edges[edge].bond_op, bond)) return false;
        if (step + 2 == arc.length) {
            pattern.closure.closing_bond = bond;
            break;
        }
        scanned_atom_t const &arc_atom = scanned.atom[arc.atom[step + 1]];
        if (arc_atom.num_branches != 0) return false; // an arc atom carries no inline branch
        if (arc_atom.map_number != 0) return false;   // a fifth label has no slot in a torsion
        simple_atom_constraint_t constraint {};
        if (!parse_atom_spec(arc_atom.constraint_text(), 0, constraint, pattern.pool)) return false;
        if (constraint.num_neighbor_branches != 0) return false;
        simple_ring_atom_t &target = pattern.closure.atom[step];
        target.num_atom_disjuncts = constraint.num_disjuncts;
        for (std::uint8_t slot = 0; slot < constraint.num_disjuncts; ++slot)
            target.atom_disjuncts[slot] = constraint.disjuncts[slot];
        target.num_bond_disjuncts = bond.num_disjuncts;
        for (std::uint8_t slot = 0; slot < bond.num_disjuncts; ++slot)
            target.bond_disjuncts[slot] = bond.disjuncts[slot];
    }
    return true;
}

/**
 *  @brief Gather each labelled atom's inline neighbour branches: its own verbatim branches first, then
 *  every spare atom that leans on it.
 *
 *  Every atom of the pattern that is neither labelled nor on the ring @p arc has to be a leaf hanging off
 *  one of the four, and becomes a branch on it; anything else refuses the pattern.
 */
inline bool collect_branch_texts_(scanned_pattern_t const &scanned, labelled_path_t const &path,
                                  scanned_arc_t const &arc, atom_branches_t (&branches)[4]) {
    for (std::size_t slot = 0; slot < 4; ++slot) {
        scanned_atom_t const &bound = scanned.atom[path.atom[slot]];
        if (!bound.bracketed) return false; // a labelled atom always carries brackets
        for (std::uint8_t branch = 0; branch < bound.num_branches; ++branch)
            if (!branches[slot].add(bound.branch_text[branch])) return false;
    }

    for (std::size_t index = 0; index < scanned.num_atoms; ++index) {
        bool on_path = arc.holds(index);
        for (std::size_t step = 0; step < 4; ++step) on_path = on_path || (path.atom[step] == index);
        if (on_path) continue;
        if (scanned.atom[index].num_branches != 0) return false; // a spare atom has to be a leaf

        // Exactly one edge, and it has to land on a labelled atom.
        std::uint8_t neighbor = scanned.atom[index].parent;
        std::size_t degree = (neighbor == no_atom_k) ? 0 : 1;
        for (std::size_t other = 0; other < scanned.num_atoms; ++other)
            if (scanned.atom[other].parent == index) {
                neighbor = static_cast<std::uint8_t>(other);
                ++degree;
            }
        if (degree != 1) return false;
        std::size_t slot = 0;
        bool attached = false;
        for (std::size_t step = 0; step < 4 && !attached; ++step)
            if (path.atom[step] == neighbor) {
                slot = step;
                attached = true;
            }
        if (!attached) return false;

        // A branch reads as its bond operator followed by its atom, contiguous in the source when the
        // spare atom is the child. When it is instead the parent of a labelled atom the operator sits on
        // the far side of the atom text, so only an implicit edge reads back.
        std::string_view branch_form;
        if (scanned.atom[index].parent == neighbor)
            branch_form = std::string_view(scanned.atom[index].bond_op.data(),
                                           scanned.atom[index].bond_op.size() + scanned.atom[index].text.size());
        else if (scanned.atom[neighbor].bond_op.empty()) branch_form = scanned.atom[index].text;
        else return false;

        if (!branches[slot].add(branch_form)) return false;
    }
    return true;
}

/**
 *  @brief Attempt to compile a 4-atom torsion SMARTS into `simple_torsion_pattern_t`.
 *
 *  Refuses any feature the matcher cannot represent — a second ring closure, an unknown recursive body, a
 *  spare atom that is itself branched, more than two inline neighbour constraints on one torsion atom —
 *  without distinguishing them, because the corpus is checked-in data and the caller's load-time
 *  diagnostic already names the offending pattern.
 */
inline bool parse_linear_torsion(std::string_view smarts, simple_torsion_pattern_t &pattern) {
    pattern = {};

    scanned_pattern_t scanned;
    if (!scan_torsion_pattern_(smarts, scanned)) return false;

    labelled_path_t path;
    if (!bind_labelled_path_(scanned, path)) return false;

    scanned_arc_t arc;
    if (!compile_ring_closure_(scanned, path, arc, pattern)) return false;

    atom_branches_t branches[4];
    if (!collect_branch_texts_(scanned, path, arc, branches)) return false;

    for (std::size_t slot = 0; slot < 4; ++slot)
        if (!parse_atom_spec(scanned.atom[path.atom[slot]].spec, 0, pattern.atom[slot], pattern.pool)) return false;
    for (std::size_t step = 0; step < 3; ++step)
        if (!parse_bond_op(path.bond_op[step], pattern.bond[step])) return false;
    for (std::size_t slot = 0; slot < 4; ++slot) {
        pattern.atom[slot].num_neighbor_branches = branches[slot].count;
        for (std::uint8_t branch = 0; branch < branches[slot].count; ++branch)
            if (!parse_branch_spec(branches[slot].text[branch], pattern.pool,
                                   pattern.atom[slot].neighbor_branch[branch]))
                return false;
    }
    return true;
}

#pragma endregion Torsion Compilation

} // namespace smallfield::preprocess
