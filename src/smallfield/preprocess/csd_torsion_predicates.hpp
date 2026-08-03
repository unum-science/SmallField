/**
 *  @file csd_torsion_predicates.hpp
 *  @brief Bounded representation and evaluator for recursive SMARTS sub-queries.
 *
 *  Roughly a quarter of the CSD torsion preference patterns hang a `$(...)` recursive
 *  sub-query off one of their atoms — `$(C=O)`, `$(S(=O)=O)`, `$([NX3](c([nX2H0])([nX2H0])))`,
 *  `$([C](=O)([$([NX3H1]),$([NX3H2])])[NX3H1])`.  A sub-query is a small rooted tree of
 *  atom constraints joined by bond constraints, rooted at the atom under test, and this
 *  header is the flat form that tree compiles into plus the walk that evaluates it.
 *
 *  Everything is stack-resident and bounded: a pool holds at most @ref max_recursive_nodes_k
 *  nodes, a body nests at most @ref max_recursive_depth_k levels of `$(...)`, and the walk
 *  carries its own visited list so the sub-query's atoms stay distinct the way RDKit's
 *  `SubstructMatch` demands.  No heap allocation, no virtual dispatch, no RDKit call —
 *  which is what removes the allocator-driven scaling collapse that `SubstructMatch`
 *  triggered on every pattern carrying a sub-query.
 *
 *  The compiler that fills a pool from SMARTS text lives in @ref csd_torsion_native.hpp,
 *  next to the atom- and bond-primitive parsers it shares.  Semantics are intended to match
 *  RDKit's `SmartsToMol(body)` + `SubstructMatch` on the first atom of the sub-query; the
 *  companion test `test_csd_torsion_predicates.cpp` cross-validates every sub-query body in
 *  the pattern corpus against RDKit at every atom of every molecule it reads.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <cstdint>
#include <limits>

#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::preprocess {
using smallfield::types::atom_flag_aromatic_k;
using smallfield::types::atom_flag_in_ring_k;
using smallfield::types::max_atoms_k;
using smallfield::types::molecule_descriptor_t;

/**
 *  Maximum number of OR-disjuncts accepted inside a single atom or bond bracket.
 *  Covers every case observed in the CSD pattern corpus, the longest being the
 *  three-way `[C^2,c,x4;r{5-8}]`.
 */
inline constexpr std::uint8_t max_disjuncts_k = 4;

/** Sentinel for an absent node reference — no sub-query, no child, no sibling. */
inline constexpr std::uint8_t no_recursive_node_k = 0xFF;

/**
 *  Node capacity of one pattern's sub-query pool.  A torsion pattern carries at most
 *  four labelled atoms plus their inline branches, and the widest body in the corpus,
 *  `$(c([nX2H0])(a(a)(a))!@;-c[nX2H0])`, spends seven nodes.
 */
inline constexpr std::uint8_t max_recursive_nodes_k = 40;

/** Deepest `$(...)` nesting the compiler accepts, counting the outermost body as level 1. */
inline constexpr std::uint8_t max_recursive_depth_k = 3;

/** Longest chain of hops a sub-query may walk away from its root atom. */
inline constexpr std::uint8_t max_recursive_hops_k = 3;

/**
 *  Atoms one walk may claim: the body it evaluates spells out at most eight, and an
 *  inline branch seeds three more for the torsion atoms the surrounding match already
 *  holds.  Keeping the list this short is what makes seeding it a register-sized copy.
 */
inline constexpr std::uint8_t max_recursive_visited_k = 16;

/**
 *  One branch of an atom-level SMARTS disjunction, e.g. `cH1` inside `[cH1,n]`.
 *
 *  A single @ref atom_disjunct_t is equivalent to a chain of SMARTS primitives joined
 *  by `&`/`;` — an AND of element, degree, H-count, hybridisation, aromaticity, ring
 *  and charge constraints, optionally guarded by one recursive sub-query.  Multiple
 *  disjuncts joined by `,` (logical OR) are stored alongside each other, and an atom
 *  satisfies the parent expression when any one of them matches.
 */
struct atom_disjunct_t {
    /** @brief Element set, read as an include-list or an exclude-list depending on
     *  @ref element_mask_exclusive: inclusive, bit N set means element N is allowed and
     *  zero means any; exclusive, bit N set means element N is forbidden. */
    std::uint64_t element_mask;
    std::uint8_t element_mask_exclusive; ///< 1 if @ref element_mask is an exclude-set; elements > 63 implicitly match
    std::uint8_t exact_h_count;          ///< exact H count; 0xFF = any
    std::uint8_t forbid_h_count;         ///< H count the atom must NOT carry (SMARTS !H<n>); 0xFF = none
    std::uint8_t min_degree;             ///< exact total degree (SMARTS X<n>); 0 = any
    std::uint8_t heavy_degree;           ///< exact heavy-atom degree (SMARTS D<n>); 0 = any
    std::uint8_t hybridization;          ///< required hybridization (SMARTS ^<n>); 0 = any
    std::uint8_t require_flags;          ///< atom_flag_t bits that MUST be set
    std::uint8_t forbid_flags;           ///< atom_flag_t bits that MUST NOT be set
    std::uint8_t ring_size;              ///< smallest-ring size lower bound; 0 = any
    std::uint8_t ring_size_max;          ///< smallest-ring size upper bound; meaningful when @ref ring_size is set
    std::int8_t formal_charge;           ///< exact formal charge; INT8_MIN = no constraint
    std::uint8_t ring_count;             ///< SMARTS R<n> — exact number of SSSR rings atom belongs to; 0 = any
    std::uint8_t ring_bond_degree;       ///< SMARTS x<n> — exact ring-bond count; 0xFF = any
    std::uint8_t recursive_root;         ///< pool node the atom must satisfy; @ref no_recursive_node_k = none
    std::uint8_t recursive_negate;       ///< 1 when the sub-query is written `!$(...)`
};

/**
 *  One branch of a bond-level SMARTS disjunction, e.g. `=` inside `=,:`.
 *  Bond-OR appears in patterns like `@;=,:@;-` — ring-bond AND (double OR aromatic).
 */
struct bond_disjunct_t {
    std::uint8_t bond_order;    ///< 1/2/3; 0 = any
    std::uint8_t require_flags; ///< bond_flag_t bits that MUST be set
    std::uint8_t forbid_flags;  ///< bond_flag_t bits that MUST NOT be set
    std::uint8_t padding;
};

/**
 *  One atom of a compiled recursive sub-query, together with the edge reaching it from
 *  its parent.  Children of a node form a singly linked list through @ref next_sibling,
 *  so the pool can be filled in one pass without reordering.  The root node's bond
 *  disjuncts are unused: nothing precedes it.
 */
struct recursive_node_t {
    atom_disjunct_t atom_disjuncts[max_disjuncts_k];
    bond_disjunct_t bond_disjuncts[max_disjuncts_k];
    std::uint8_t num_atom_disjuncts;
    std::uint8_t num_bond_disjuncts;
    std::uint8_t first_child;  ///< @ref no_recursive_node_k when the node is a leaf
    std::uint8_t next_sibling; ///< @ref no_recursive_node_k at the end of a child list
    std::uint8_t padding[4];
};

/** Every sub-query node one torsion pattern needs, indexed by @ref atom_disjunct_t::recursive_root. */
struct recursive_pool_t {
    recursive_node_t nodes[max_recursive_nodes_k];
    std::uint8_t num_nodes;
    std::uint8_t padding[7];
};

/**
 *  Atoms a sub-query walk has already claimed.  RDKit maps every atom of a query to a
 *  distinct molecule atom, so a walk may not revisit one; the list is short enough that
 *  a linear scan beats any bitmap.
 */
struct recursive_visit_t {
    std::uint8_t atoms[max_recursive_visited_k];
    std::uint8_t count;
};

/** @brief Test whether a bond satisfies one @ref bond_disjunct_t. */
[[gnu::hot]] inline bool bond_disjunct_match(molecule_descriptor_t const &desc, std::uint32_t bond_index,
                                             bond_disjunct_t const &disjunct) {
    if (disjunct.bond_order != 0 && desc.bond_order[bond_index] != disjunct.bond_order) return false;
    std::uint8_t const flags = desc.bond_flags[bond_index];
    if ((flags & disjunct.require_flags) != disjunct.require_flags) return false;
    if ((flags & disjunct.forbid_flags) != 0) return false;
    return true;
}

inline bool recursive_match_node(molecule_descriptor_t const &desc, recursive_pool_t const &pool,
                                 std::uint8_t node_index, std::uint32_t atom_index, recursive_visit_t &visited);

/**
 *  @brief Test whether a single atom satisfies one @ref atom_disjunct_t — one OR branch
 *  of a SMARTS atom expression, including its recursive guard when it carries one.
 *
 *  @param[in] desc Molecule descriptor with topology and property arrays.
 *  @param[in] atom_index Atom to test.
 *  @param[in] disjunct One conjunction of SMARTS primitives.
 *  @param[in] pool Sub-query nodes the disjunct's recursive guard indexes into.
 *  @return True when every primitive holds.
 */
[[gnu::hot]] inline bool atom_disjunct_match(molecule_descriptor_t const &desc, std::uint32_t atom_index,
                                             atom_disjunct_t const &disjunct, recursive_pool_t const &pool) {
    std::uint8_t const element = desc.element[atom_index];
    if (disjunct.element_mask != 0) {
        if (disjunct.element_mask_exclusive) {
            // Exclude-set: bit N set ⇒ element N is forbidden. Elements > 63
            // are implicitly allowed (they cannot collide with a 64-bit mask).
            if (element < 64 && (disjunct.element_mask & (1ULL << element)) != 0) return false;
        }
        else {
            // Include-set: bit N set ⇒ element N is accepted.  Elements > 63
            // cannot match a 64-bit positive enumeration and are rejected.
            if (element >= 64) return false;
            if ((disjunct.element_mask & (1ULL << element)) == 0) return false;
        }
    }
    if (disjunct.min_degree != 0 && desc.degree[atom_index] != disjunct.min_degree) return false;
    if (disjunct.heavy_degree != 0) {
        std::uint8_t const degree = desc.degree[atom_index];
        std::uint8_t const num_hydrogens = desc.num_hydrogens[atom_index];
        std::uint8_t const heavy = (degree >= num_hydrogens) ? static_cast<std::uint8_t>(degree - num_hydrogens) : 0;
        if (heavy != disjunct.heavy_degree) return false;
    }
    if (disjunct.exact_h_count != 0xFF && desc.num_hydrogens[atom_index] != disjunct.exact_h_count) return false;
    if (disjunct.forbid_h_count != 0xFF && desc.num_hydrogens[atom_index] == disjunct.forbid_h_count) return false;
    if (disjunct.ring_bond_degree != 0xFF && desc.ring_bond_degree[atom_index] != disjunct.ring_bond_degree)
        return false;
    if (disjunct.ring_size != 0) {
        std::uint8_t const smallest = desc.smallest_ring[atom_index];
        if (smallest < disjunct.ring_size || smallest > disjunct.ring_size_max) return false;
    }
    if (disjunct.ring_count != 0) {
        // SMARTS R<n>: atom belongs to exactly n SSSR rings.  The descriptor
        // caps ring-tracking at `max_rings_k` (32); atoms belonging to more
        // rings than that are conservatively under-counted, but RDKit's own
        // R<n> matcher has the same SSSR dependency so divergence only
        // appears on molecules the descriptor already treats specially.
        std::uint8_t const count = static_cast<std::uint8_t>(__builtin_popcount(desc.atom_ring_bitmap[atom_index]));
        if (count != disjunct.ring_count) return false;
    }
    if (disjunct.hybridization != 0 && desc.hybridization[atom_index] != disjunct.hybridization) return false;
    std::int8_t constexpr no_charge_k = std::numeric_limits<std::int8_t>::min();
    if (disjunct.formal_charge != no_charge_k && desc.formal_charge[atom_index] != disjunct.formal_charge) return false;
    std::uint8_t const flags = desc.atom_flags[atom_index];
    if ((flags & disjunct.require_flags) != disjunct.require_flags) return false;
    if ((flags & disjunct.forbid_flags) != 0) return false;
    if (disjunct.recursive_root != no_recursive_node_k) {
        recursive_visit_t visited;
        visited.count = 0;
        bool const satisfied = recursive_match_node(desc, pool, disjunct.recursive_root, atom_index, visited);
        if (satisfied == (disjunct.recursive_negate != 0)) return false;
    }
    return true;
}

/**
 *  @brief Bind @p child_index and every sibling after it to distinct neighbours of @p parent_atom.
 *
 *  Assignment backtracks across the sibling list, which is what lets a body like
 *  `$([c]([cH,nX2H0])[cH,nX2H0])` demand two different neighbours rather than one
 *  neighbour counted twice.
 *
 *  @param[in] desc Molecule descriptor with topology and property arrays.
 *  @param[in] pool Sub-query nodes.
 *  @param[in] child_index First sibling still to bind; @ref no_recursive_node_k when the list is exhausted.
 *  @param[in] parent_atom Molecule atom the siblings hang off.
 *  @param[inout] visited Atoms already claimed by this walk; restored on every failed branch.
 *  @return True when the whole sibling list binds.
 */
inline bool recursive_match_children(molecule_descriptor_t const &desc, recursive_pool_t const &pool,
                                     std::uint8_t child_index, std::uint32_t parent_atom, recursive_visit_t &visited) {
    if (child_index == no_recursive_node_k) return true;
    recursive_node_t const &child = pool.nodes[child_index];
    std::uint16_t const start = desc.neighbor_offset[parent_atom];
    std::uint16_t const end = desc.neighbor_offset[parent_atom + 1];
    for (std::uint16_t entry = start; entry < end; ++entry) {
        std::uint8_t const candidate = desc.neighbor_atom[entry];
        bool claimed = false;
        for (std::uint8_t slot = 0; slot < visited.count; ++slot) claimed = claimed || visited.atoms[slot] == candidate;
        if (claimed) continue;

        bool bond_ok = false;
        for (std::uint8_t slot = 0; slot < child.num_bond_disjuncts && !bond_ok; ++slot)
            bond_ok = bond_disjunct_match(desc, desc.neighbor_bond[entry], child.bond_disjuncts[slot]);
        if (!bond_ok) continue;

        std::uint8_t const mark = visited.count;
        if (!recursive_match_node(desc, pool, child_index, candidate, visited)) {
            visited.count = mark;
            continue;
        }
        if (recursive_match_children(desc, pool, child.next_sibling, parent_atom, visited)) return true;
        visited.count = mark;
    }
    return false;
}

/**
 *  @brief Bind one sub-query node to @p atom_index and its whole subtree below that.
 *
 *  @param[in] desc Molecule descriptor with topology and property arrays.
 *  @param[in] pool Sub-query nodes.
 *  @param[in] node_index Node to bind.
 *  @param[in] atom_index Molecule atom to bind it to.
 *  @param[inout] visited Atoms already claimed by this walk; @p atom_index joins them on success.
 *  @return True when the node and its subtree bind.
 */
inline bool recursive_match_node(molecule_descriptor_t const &desc, recursive_pool_t const &pool,
                                 std::uint8_t node_index, std::uint32_t atom_index, recursive_visit_t &visited) {
    recursive_node_t const &node = pool.nodes[node_index];
    bool any_disjunct = false;
    for (std::uint8_t slot = 0; slot < node.num_atom_disjuncts && !any_disjunct; ++slot)
        any_disjunct = atom_disjunct_match(desc, atom_index, node.atom_disjuncts[slot], pool);
    if (!any_disjunct) return false;
    if (visited.count >= max_recursive_visited_k) return false;
    visited.atoms[visited.count++] = static_cast<std::uint8_t>(atom_index);
    return recursive_match_children(desc, pool, node.first_child, atom_index, visited);
}

/**
 *  @brief Evaluate a compiled recursive sub-query against one atom.
 *
 *  @param[in] desc Molecule descriptor filled by @ref build_descriptor.
 *  @param[in] atom_index Atom to test; must be below `desc.num_atoms`.
 *  @param[in] pool Sub-query nodes the root indexes into.
 *  @param[in] root Node the sub-query is rooted at, or @ref no_recursive_node_k for "no constraint".
 *  @return True when @p atom_index satisfies the sub-query.
 */
inline bool recursive_query_match(molecule_descriptor_t const &desc, std::uint32_t atom_index,
                                  recursive_pool_t const &pool, std::uint8_t root) {
    if (root == no_recursive_node_k) return true;
    recursive_visit_t visited;
    visited.count = 0;
    return recursive_match_node(desc, pool, root, atom_index, visited);
}

} // namespace smallfield::preprocess
