/**
 *  @file src/smallfield/preprocess/csd_torsion_predicates.hpp
 *  @brief Bounded representation of a compiled CSD torsion pattern, and the matcher that evaluates it.
 *
 *  A torsion pattern is four labelled atoms joined by three bonds, optionally closing a ring back onto
 *  the first, with inline neighbour branches and `$(...)` recursive sub-queries hanging off any of them.
 *  This header is the flat form all of that compiles into plus every predicate that reads it, from the
 *  single-primitive test up to the whole-molecule scan.
 *
 *  Everything is stack-resident and bounded: a pool holds at most `max_recursive_nodes_k` nodes, a body
 *  nests at most `max_recursive_depth_k` levels of `$(...)`, and each walk carries its own visited list
 *  so a query's atoms stay distinct the way RDKit's `SubstructMatch` demands. No heap allocation, no
 *  virtual dispatch, no RDKit call — which is what removes the allocator-driven scaling collapse that
 *  `SubstructMatch` triggered on every pattern carrying a sub-query.
 *
 *  Nothing here parses text. The compiler that fills these structures from SMARTS lives in
 *  `csd_torsion_native.hpp`, is host-only, and is the only thing between the two files that RDKit or the
 *  standard string headers reach; a pattern arrives on the device as bytes and is matched by the bodies
 *  below unchanged, which is what lets `cuda/csd_torsion_kernel.cuh` share this code rather than mirror
 *  it. Semantics are intended to match RDKit's `SmartsToMol` + `SubstructMatch`, cross-validated pattern
 *  by pattern and atom by atom in `test_csd_torsion_predicates.cpp`.
 */

#pragma once

#include <cstdint> // `std::uint8_t`, `std::uint64_t`

#include <limits> // `std::numeric_limits`

#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::preprocess {
using smallfield::types::atom_flag_aromatic_k;
using smallfield::types::atom_flag_in_ring_k;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::molecule_descriptor_t;

#pragma region Capacities

/**
 *  @brief Maximum number of OR-disjuncts accepted inside a single atom or bond bracket.
 *  Covers every case observed in the CSD pattern corpus, the longest being the
 *  three-way `[C^2,c,x4;r{5-8}]`.
 */
inline constexpr std::uint8_t max_disjuncts_k = 4;

/** Sentinel for an absent node reference — no sub-query, no child, no sibling. */
inline constexpr std::uint8_t no_recursive_node_k = 0xFF;

/** Sentinel for a count no primitive pinned — an H count, an H count to forbid, a ring-bond degree. */
inline constexpr std::uint8_t no_count_k = 0xFF;

/** Sentinel for a formal charge no primitive pinned, distinct from a neutral atom's zero. */
inline constexpr std::int8_t no_formal_charge_k = std::numeric_limits<std::int8_t>::min();

/**
 *  @brief Node capacity of one pattern's sub-query pool. A torsion pattern carries at most
 *  four labelled atoms plus their inline branches, and the widest body in the corpus,
 *  `$(c([nX2H0])(a(a)(a))!@;-c[nX2H0])`, spends seven nodes.
 */
inline constexpr std::uint8_t max_recursive_nodes_k = 40;

/** Deepest `$(...)` nesting the compiler accepts, counting the outermost body as level 1. */
inline constexpr std::uint8_t max_recursive_depth_k = 3;

/** Longest chain of hops a sub-query may walk away from its root atom. */
inline constexpr std::uint8_t max_recursive_hops_k = 3;

/**
 *  @brief Atoms one walk may claim: the body it evaluates spells out at most eight, and an
 *  inline branch seeds three more for the torsion atoms the surrounding match already
 *  holds. Keeping the list this short is what makes seeding it a register-sized copy.
 */
inline constexpr std::uint8_t max_recursive_visited_k = 16;

/**
 *  @brief Longest run of unlabelled atoms a ring-closing pattern may place between `:4`
 *  and `:1`. The widest ring in the CSD small-ring library is the eight-membered
 *  `[CX4;r{5-8}:1]1@[CX4;r{5-8}:2]@;-[CX4;x3;r{5-8}:3]@[!#1;r{5-8}:4]@[!#1]@[!#1]@[!#1]@[!#1]1`.
 */
inline constexpr std::uint8_t max_ring_closure_atoms_k = 4;

#pragma endregion Capacities

#pragma region Compiled Representation

/**
 *  @brief One branch of an atom-level SMARTS disjunction, e.g. `cH1` inside `[cH1,n]`.
 *
 *  A single `atom_disjunct_t` is equivalent to a chain of SMARTS primitives joined
 *  by `&`/`;` — an AND of element, degree, H-count, hybridisation, aromaticity, ring
 *  and charge constraints, optionally guarded by one recursive sub-query. Multiple
 *  disjuncts joined by `,` (logical OR) are stored alongside each other, and an atom
 *  satisfies the parent expression when any one of them matches.
 *
 *  A default-constructed disjunct constrains nothing, so the compiler folds each primitive it reads
 *  into a `{}` and never has to remember which fields sit at a sentinel rather than at zero.
 */
struct atom_disjunct_t {
    /** Element set, read as an include-list or an exclude-list depending on
     *  `element_mask_exclusive:` inclusive, bit N set means element N is allowed and
     *  zero means any; exclusive, bit N set means element N is forbidden. */
    std::uint64_t element_mask = 0;
    /** 1 if `element_mask` is an exclude-set; elements > 63 implicitly match. */
    std::uint8_t element_mask_exclusive = 0;
    /** Exact H count; `no_count_k` = any. */
    std::uint8_t exact_h_count = no_count_k;
    /** H count the atom must @b not carry (SMARTS !H<n>), `no_count_k` = none. */
    std::uint8_t forbid_h_count = no_count_k;
    /** Exact total degree (SMARTS X<n> and D<n> on a hydrogenated molecule); 0 = any. */
    std::uint8_t degree = 0;
    /** Required hybridization (SMARTS ^<n>); 0 = any. */
    std::uint8_t hybridization = 0;
    /** `atom_flag_t` bits that must be set. */
    std::uint8_t require_flags = 0;
    /** `atom_flag_t` bits that must @b not be set. */
    std::uint8_t forbid_flags = 0;
    /** Smallest-ring size lower bound; 0 = any. */
    std::uint8_t ring_size = 0;
    /** Smallest-ring size upper bound; meaningful when `ring_size` is set. */
    std::uint8_t ring_size_max = 0;
    /** Exact formal charge; `no_formal_charge_k` = no constraint. */
    std::int8_t formal_charge = no_formal_charge_k;
    /** SMARTS R<n> — exact number of SSSR rings atom belongs to; 0 = any. */
    std::uint8_t ring_count = 0;
    /** SMARTS x<n> — exact ring-bond count; `no_count_k` = any. */
    std::uint8_t ring_bond_degree = no_count_k;
    /** Pool node the atom must satisfy, `no_recursive_node_k` = none. */
    std::uint8_t recursive_root = no_recursive_node_k;
    /** 1 when the sub-query is written `!$(...)`. */
    std::uint8_t recursive_negate = 0;
};

/**
 *  @brief One branch of a bond-level SMARTS disjunction, e.g. `=` inside `=,:`.
 *  Bond-OR appears in patterns like `@;=,:@;-` — ring-bond AND (double OR aromatic).
 */
struct bond_disjunct_t {
    /** 1/2/3, 0 = any. */
    std::uint8_t bond_order;
    /** `bond_flag_t` bits that must be set. */
    std::uint8_t require_flags;
    /** `bond_flag_t` bits that must @b not be set. */
    std::uint8_t forbid_flags;
    std::uint8_t padding;
};

/**
 *  @brief One atom of a compiled recursive sub-query, together with the edge reaching it from
 *  its parent. Children of a node form a singly linked list through `next_sibling`,
 *  so the pool can be filled in one pass without reordering. The root node's bond
 *  disjuncts are unused: nothing precedes it.
 */
struct recursive_node_t {
    atom_disjunct_t atom_disjuncts[max_disjuncts_k];
    bond_disjunct_t bond_disjuncts[max_disjuncts_k];
    std::uint8_t num_atom_disjuncts;
    std::uint8_t num_bond_disjuncts;
    /** `no_recursive_node_k` when the node is a leaf. */
    std::uint8_t first_child;
    /** `no_recursive_node_k` at the end of a child list. */
    std::uint8_t next_sibling;
    std::uint8_t padding[4];

    /** The disjuncts the atom must satisfy one of. */
    constexpr custd::span<atom_disjunct_t const> active_atom_disjuncts() const noexcept {
        return {atom_disjuncts, num_atom_disjuncts};
    }
    /** The disjuncts the edge from the parent must satisfy one of. */
    constexpr custd::span<bond_disjunct_t const> active_bond_disjuncts() const noexcept {
        return {bond_disjuncts, num_bond_disjuncts};
    }
};

/** Every sub-query node one torsion pattern needs, indexed by `atom_disjunct_t::recursive_root`. */
struct recursive_pool_t {
    recursive_node_t nodes[max_recursive_nodes_k];
    std::uint8_t num_nodes;
    std::uint8_t padding[7];
};

/**
 *  @brief Borrowed handle on one pattern's sub-query nodes. A host pattern owns its nodes inline in a
 *  `recursive_pool_t`; a device pattern indexes a shared node arena that holds every pattern's
 *  nodes back to back. The evaluator takes this handle so both layouts run the same code, and the
 *  converting constructor keeps every host call site writing `pattern.pool` unchanged.
 */
struct recursive_pool_view_t {
    recursive_node_t const *nodes = nullptr;

    recursive_pool_view_t() = default;
    constexpr recursive_pool_view_t(recursive_node_t const *node_arena) : nodes(node_arena) {}
    constexpr recursive_pool_view_t(recursive_pool_t const &pool) : nodes(pool.nodes) {}
};

/**
 *  @brief Atoms a walk has already claimed. RDKit maps every atom of a query to a distinct molecule
 *  atom, so neither a sub-query walk nor a ring-closure walk may revisit one; the list is short
 *  enough that a linear scan beats any bitmap.
 */
struct recursive_visit_t {
    std::uint8_t atoms[max_recursive_visited_k];
    std::uint8_t count;

    /** Whether @p atom_index is already claimed. */
    constexpr bool claimed(std::uint8_t atom_index) const noexcept {
        bool held = false;
        for (std::uint8_t slot = 0; slot < count; ++slot) held = held || atoms[slot] == atom_index;
        return held;
    }
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
    std::uint8_t padding[6];

    /** The disjuncts the edge from the torsion atom must satisfy one of. */
    constexpr custd::span<bond_disjunct_t const> active_bond_disjuncts() const noexcept {
        return {bond_disjuncts, num_bond_disjuncts};
    }
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

    /** The disjuncts the atom must satisfy one of. */
    constexpr custd::span<atom_disjunct_t const> active_atom_disjuncts() const noexcept {
        return {disjuncts, num_disjuncts};
    }
};

/** Per-bond constraint extracted from SMARTS. */
struct simple_bond_constraint_t {
    bond_disjunct_t disjuncts[max_disjuncts_k];
    /** 1 for plain op, up to 4 for `a,b,c,d`. */
    std::uint8_t num_disjuncts;
    std::uint8_t padding[3];

    /** The disjuncts the bond must satisfy one of. */
    constexpr custd::span<bond_disjunct_t const> active_bond_disjuncts() const noexcept {
        return {disjuncts, num_disjuncts};
    }
};

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

    /** The disjuncts the arc atom must satisfy one of. */
    constexpr custd::span<atom_disjunct_t const> active_atom_disjuncts() const noexcept {
        return {atom_disjuncts, num_atom_disjuncts};
    }
    /** The disjuncts the edge reaching the arc atom must satisfy one of. */
    constexpr custd::span<bond_disjunct_t const> active_bond_disjuncts() const noexcept {
        return {bond_disjuncts, num_bond_disjuncts};
    }
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
    /** 1 when the pattern closes a ring at all; a view over an inactive closure is null instead. */
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
    constexpr csd_pattern_view_t(simple_atom_constraint_t const *atom_constraints,
                                 simple_bond_constraint_t const *bond_constraints,
                                 simple_ring_closure_t const *ring_closure, recursive_pool_view_t node_pool)
        : atom(atom_constraints), bond(bond_constraints), closure(ring_closure), pool(node_pool) {}
    constexpr csd_pattern_view_t(simple_torsion_pattern_t const &pattern)
        : atom(pattern.atom), bond(pattern.bond), closure(pattern.closure.active ? &pattern.closure : nullptr),
          pool(pattern.pool) {}
};

/*  The device table memcpies these, so a member added without its padding adjusted would change the
 *  layout on one side of the upload only. */
static_assert(sizeof(atom_disjunct_t) == 24, "atom_disjunct_t crosses to the device by memcpy");
static_assert(sizeof(bond_disjunct_t) == 4, "bond_disjunct_t crosses to the device by memcpy");
static_assert(sizeof(recursive_node_t) == 120, "recursive_node_t crosses to the device by memcpy");
static_assert(sizeof(simple_neighbor_constraint_t) == 24, "simple_neighbor_constraint_t crosses to the device");
static_assert(sizeof(simple_atom_constraint_t) == 152, "simple_atom_constraint_t crosses to the device by memcpy");
static_assert(sizeof(simple_bond_constraint_t) == 20, "simple_bond_constraint_t crosses to the device by memcpy");
static_assert(sizeof(simple_ring_atom_t) == 120, "simple_ring_atom_t crosses to the device by memcpy");
static_assert(sizeof(simple_ring_closure_t) == 512, "simple_ring_closure_t crosses to the device by memcpy");

#pragma endregion Compiled Representation

#pragma region Primitive Tests

/** Test whether a bond satisfies one `bond_disjunct_t`. */
[[gnu::hot]] inline constexpr bool bond_disjunct_match(molecule_descriptor_t const &descriptor,
                                                       std::uint32_t bond_index, bond_disjunct_t const &disjunct) {
    if (disjunct.bond_order != 0 && descriptor.bond_order[bond_index] != disjunct.bond_order) return false;
    std::uint8_t const flags = descriptor.bond_flags[bond_index];
    if ((flags & disjunct.require_flags) != disjunct.require_flags) return false;
    if ((flags & disjunct.forbid_flags) != 0) return false;
    return true;
}

inline constexpr bool recursive_match_node(molecule_descriptor_t const &descriptor, recursive_pool_view_t pool,
                                           std::uint8_t node_index, std::uint32_t atom_index,
                                           recursive_visit_t &visited);

/**
 *  @brief Test whether a single atom satisfies one `atom_disjunct_t` — one OR branch
 *  of a SMARTS atom expression, including its recursive guard when it carries one.
 *
 *  @param[in] descriptor Molecule descriptor with topology and property arrays.
 *  @param[in] atom_index Atom to test.
 *  @param[in] disjunct One conjunction of SMARTS primitives.
 *  @param[in] pool Sub-query nodes the disjunct's recursive guard indexes into.
 *  @return True when every primitive holds.
 */
[[gnu::hot]] inline constexpr bool atom_disjunct_match(molecule_descriptor_t const &descriptor,
                                                       std::uint32_t atom_index, atom_disjunct_t const &disjunct,
                                                       recursive_pool_view_t pool) {
    std::uint8_t const element = descriptor.element[atom_index];
    if (disjunct.element_mask != 0) {
        if (disjunct.element_mask_exclusive) {
            // Exclude-set: bit N set ⇒ element N is forbidden. Elements > 63
            // are implicitly allowed (they cannot collide with a 64-bit mask).
            if (element < 64 && (disjunct.element_mask & (1ULL << element)) != 0) return false;
        }
        else {
            // Include-set: bit N set ⇒ element N is accepted. Elements > 63
            // cannot match a 64-bit positive enumeration and are rejected.
            if (element >= 64) return false;
            if ((disjunct.element_mask & (1ULL << element)) == 0) return false;
        }
    }
    if (disjunct.degree != 0 && descriptor.degree[atom_index] != disjunct.degree) return false;
    if (disjunct.exact_h_count != no_count_k && descriptor.num_hydrogens[atom_index] != disjunct.exact_h_count)
        return false;
    if (disjunct.forbid_h_count != no_count_k && descriptor.num_hydrogens[atom_index] == disjunct.forbid_h_count)
        return false;
    if (disjunct.ring_bond_degree != no_count_k && descriptor.ring_bond_degree[atom_index] != disjunct.ring_bond_degree)
        return false;
    if (disjunct.ring_size != 0) {
        std::uint8_t const smallest = descriptor.smallest_ring[atom_index];
        if (smallest < disjunct.ring_size || smallest > disjunct.ring_size_max) return false;
    }
    if (disjunct.ring_count != 0) {
        // SMARTS R<n>: atom belongs to exactly n SSSR rings. The descriptor
        // caps ring-tracking at `max_rings_k` (32); atoms belonging to more
        // rings than that are conservatively under-counted, but RDKit's own
        // R<n> matcher has the same SSSR dependency so divergence only
        // appears on molecules the descriptor already treats specially.
        std::uint8_t const count = static_cast<std::uint8_t>(
            __builtin_popcount(descriptor.atom_ring_bitmap[atom_index]));
        if (count != disjunct.ring_count) return false;
    }
    if (disjunct.hybridization != 0 && descriptor.hybridization[atom_index] != disjunct.hybridization) return false;
    if (disjunct.formal_charge != no_formal_charge_k && descriptor.formal_charge[atom_index] != disjunct.formal_charge)
        return false;
    std::uint8_t const flags = descriptor.atom_flags[atom_index];
    if ((flags & disjunct.require_flags) != disjunct.require_flags) return false;
    if ((flags & disjunct.forbid_flags) != 0) return false;
    if (disjunct.recursive_root != no_recursive_node_k) {
        recursive_visit_t visited;
        visited.count = 0;
        bool const satisfied = recursive_match_node(descriptor, pool, disjunct.recursive_root, atom_index, visited);
        if (satisfied == (disjunct.recursive_negate != 0)) return false;
    }
    return true;
}

/** Whether @p atom_index satisfies any of @p disjuncts, which is what a SMARTS `,` means. */
inline constexpr bool atom_match(molecule_descriptor_t const &descriptor, std::uint32_t atom_index,
                                 custd::span<atom_disjunct_t const> disjuncts, recursive_pool_view_t pool) {
    for (auto const &disjunct : disjuncts)
        if (atom_disjunct_match(descriptor, atom_index, disjunct, pool)) return true;
    return false;
}

/** Whether @p bond_index satisfies any of @p disjuncts, which is what a SMARTS `,` means. */
inline constexpr bool bond_match(molecule_descriptor_t const &descriptor, std::uint32_t bond_index,
                                 custd::span<bond_disjunct_t const> disjuncts) {
    for (auto const &disjunct : disjuncts)
        if (bond_disjunct_match(descriptor, bond_index, disjunct)) return true;
    return false;
}

#pragma endregion Primitive Tests

#pragma region Sub Query Walk

/**
 *  @brief Bind @p child_index and every sibling after it to distinct neighbours of @p parent_atom.
 *
 *  Assignment backtracks across the sibling list, which is what lets a body like
 *  `$([c]([cH,nX2H0])[cH,nX2H0])` demand two different neighbours rather than one
 *  neighbour counted twice.
 *
 *  @param[in] descriptor Molecule descriptor with topology and property arrays.
 *  @param[in] pool Sub-query nodes.
 *  @param[in] child_index First sibling still to bind; `no_recursive_node_k` when the list is exhausted.
 *  @param[in] parent_atom Molecule atom the siblings hang off.
 *  @param[inout] visited Atoms already claimed by this walk; restored on every failed branch.
 *  @return True when the whole sibling list binds.
 */
inline constexpr bool recursive_match_children(molecule_descriptor_t const &descriptor, recursive_pool_view_t pool,
                                               std::uint8_t child_index, std::uint32_t parent_atom,
                                               recursive_visit_t &visited) {
    if (child_index == no_recursive_node_k) return true;
    recursive_node_t const &child = pool.nodes[child_index];
    std::uint16_t const start = descriptor.neighbor_offset[parent_atom];
    std::uint16_t const end = descriptor.neighbor_offset[parent_atom + 1];
    for (std::uint16_t entry = start; entry < end; ++entry) {
        std::uint8_t const candidate = descriptor.neighbor_atom[entry];
        if (visited.claimed(candidate)) continue;
        if (!bond_match(descriptor, descriptor.neighbor_bond[entry], child.active_bond_disjuncts())) continue;

        std::uint8_t const mark = visited.count;
        if (!recursive_match_node(descriptor, pool, child_index, candidate, visited)) {
            visited.count = mark;
            continue;
        }
        if (recursive_match_children(descriptor, pool, child.next_sibling, parent_atom, visited)) return true;
        visited.count = mark;
    }
    return false;
}

/**
 *  @brief Bind one sub-query node to @p atom_index and its whole subtree below that.
 *
 *  @param[in] descriptor Molecule descriptor with topology and property arrays.
 *  @param[in] pool Sub-query nodes.
 *  @param[in] node_index Node to bind.
 *  @param[in] atom_index Molecule atom to bind it to.
 *  @param[inout] visited Atoms already claimed by this walk; @p atom_index joins them on success.
 *  @return True when the node and its subtree bind.
 */
inline constexpr bool recursive_match_node(molecule_descriptor_t const &descriptor, recursive_pool_view_t pool,
                                           std::uint8_t node_index, std::uint32_t atom_index,
                                           recursive_visit_t &visited) {
    recursive_node_t const &node = pool.nodes[node_index];
    if (!atom_match(descriptor, atom_index, node.active_atom_disjuncts(), pool)) return false;
    if (visited.count >= max_recursive_visited_k) return false;
    visited.atoms[visited.count++] = static_cast<std::uint8_t>(atom_index);
    return recursive_match_children(descriptor, pool, node.first_child, atom_index, visited);
}

/**
 *  @brief Evaluate a compiled recursive sub-query against one atom.
 *
 *  @param[in] descriptor Molecule descriptor filled by `build_descriptor`.
 *  @param[in] atom_index Atom to test; must be below `descriptor.num_atoms`.
 *  @param[in] pool Sub-query nodes the root indexes into.
 *  @param[in] root Node the sub-query is rooted at, or `no_recursive_node_k` for "no constraint".
 *  @return True when @p atom_index satisfies the sub-query.
 */
inline constexpr bool recursive_query_match(molecule_descriptor_t const &descriptor, std::uint32_t atom_index,
                                            recursive_pool_view_t pool, std::uint8_t root) {
    if (root == no_recursive_node_k) return true;
    recursive_visit_t visited;
    visited.count = 0;
    return recursive_match_node(descriptor, pool, root, atom_index, visited);
}

#pragma endregion Sub Query Walk

#pragma region Ring Topology

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
inline constexpr std::uint32_t find_bond_between(molecule_descriptor_t const &descriptor, std::uint32_t first_atom,
                                                 std::uint32_t second_atom) {
    std::uint16_t const start = descriptor.neighbor_offset[first_atom];
    std::uint16_t const end = descriptor.neighbor_offset[first_atom + 1];
    for (std::uint16_t entry = start; entry < end; ++entry)
        if (descriptor.neighbor_atom[entry] == second_atom) return descriptor.neighbor_bond[entry];
    return no_bond_index_k;
}

/** Atom count of ring @p ring_index. */
inline constexpr std::uint32_t ring_atom_count(molecule_descriptor_t const &descriptor, std::uint32_t ring_index) {
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
inline constexpr bool ring_holds_bond(molecule_descriptor_t const &descriptor, std::uint32_t ring_index,
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
inline constexpr std::uint32_t bond_ring_count(molecule_descriptor_t const &descriptor, std::uint32_t bond_index) {
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
inline constexpr bool rings_share_two_bonds(molecule_descriptor_t const &descriptor, std::uint32_t first_ring,
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
inline constexpr bool bond_excluded_by_ring_fusion(molecule_descriptor_t const &descriptor, std::uint32_t bond_index) {
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

#pragma endregion Ring Topology

#pragma region Torsion Matching

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
inline constexpr bool neighbor_match(molecule_descriptor_t const &descriptor, std::uint32_t atom_index,
                                     std::uint32_t bond_index, simple_neighbor_constraint_t const &constraint,
                                     recursive_pool_view_t pool, recursive_visit_t const &seed) {
    if (!bond_match(descriptor, bond_index, constraint.active_bond_disjuncts())) return false;
    recursive_visit_t visited = seed;
    return recursive_match_node(descriptor, pool, constraint.node, atom_index, visited);
}

/**
 *  @brief Verify that @p atom_index has enough neighbours satisfying @p constraint's branch
 *  list. Branch neighbours are drawn from @p atom_index's adjacency, excluding the two
 *  chain-adjacent atoms @p chain_a and @p chain_b (pass 0xFF if unused). When two
 *  branches are present both orderings are tried, so a pattern like `(~[N,n])(~[N,n])` can
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
inline constexpr bool check_branch_neighbors(molecule_descriptor_t const &descriptor, std::uint32_t atom_index,
                                             simple_atom_constraint_t const &constraint, recursive_pool_view_t pool,
                                             std::uint32_t chain_a, std::uint32_t chain_b) {
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
 *  @return True when some assignment of the arc atoms closes onto @p target.
 */
inline constexpr bool ring_closure_walk(molecule_descriptor_t const &descriptor, simple_ring_closure_t const &closure,
                                        recursive_pool_view_t pool, std::uint32_t origin, std::uint32_t target,
                                        recursive_visit_t &visited) {
    std::uint8_t const total = closure.num_atoms;
    std::uint16_t cursor[max_ring_closure_atoms_k + 1];
    std::uint8_t standing[max_ring_closure_atoms_k + 1];
    standing[0] = static_cast<std::uint8_t>(origin);
    cursor[0] = descriptor.neighbor_offset[origin];
    std::uint8_t depth = 0;

    while (true) {
        std::uint8_t const current = standing[depth];
        if (depth == total) {
            std::uint16_t const start = descriptor.neighbor_offset[current];
            std::uint16_t const end = descriptor.neighbor_offset[current + 1];
            for (std::uint16_t entry = start; entry < end; ++entry)
                if (descriptor.neighbor_atom[entry] == target &&
                    bond_match(descriptor, descriptor.neighbor_bond[entry],
                               closure.closing_bond.active_bond_disjuncts()))
                    return true;
            if (depth == 0) return false;
            --depth;
            --visited.count;
            continue;
        }

        simple_ring_atom_t const &wanted = closure.atom[depth];
        std::uint16_t const end = descriptor.neighbor_offset[current + 1];
        bool descended = false;
        while (cursor[depth] < end) {
            std::uint16_t const entry = cursor[depth]++;
            std::uint8_t const candidate = descriptor.neighbor_atom[entry];
            if (visited.claimed(candidate)) continue;
            if (!bond_match(descriptor, descriptor.neighbor_bond[entry], wanted.active_bond_disjuncts())) continue;
            if (!atom_match(descriptor, candidate, wanted.active_atom_disjuncts(), pool)) continue;

            visited.atoms[visited.count++] = candidate;
            ++depth;
            standing[depth] = candidate;
            cursor[depth] = descriptor.neighbor_offset[candidate];
            descended = true;
            break;
        }
        if (descended) continue;
        if (depth == 0) return false;
        --depth;
        --visited.count;
    }
}

/** Verify the ring a closing pattern draws around an already-bound torsion. */
inline constexpr bool ring_closure_match(molecule_descriptor_t const &descriptor, simple_ring_closure_t const &closure,
                                         recursive_pool_view_t pool, std::uint32_t atom_1, std::uint32_t atom_2,
                                         std::uint32_t atom_3, std::uint32_t atom_4) {
    recursive_visit_t visited;
    visited.count = 0;
    visited.atoms[visited.count++] = static_cast<std::uint8_t>(atom_1);
    visited.atoms[visited.count++] = static_cast<std::uint8_t>(atom_2);
    visited.atoms[visited.count++] = static_cast<std::uint8_t>(atom_3);
    visited.atoms[visited.count++] = static_cast<std::uint8_t>(atom_4);
    return ring_closure_walk(descriptor, closure, pool, atom_4, atom_1, visited);
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
inline constexpr csd_molecule_gate_t build_molecule_gate(molecule_descriptor_t const &descriptor) {
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
 *  @param[in] emit Callable taking `(atom_1, atom_2, atom_3, atom_4, bond_index)` and returning true to stop.
 *  @return True when @p emit stopped the scan.
 */
template <typename emit_fn_>
inline constexpr bool match_pattern_at_bond(molecule_descriptor_t const &descriptor, csd_pattern_view_t const &pattern,
                                            std::uint32_t bond_index, emit_fn_ &&emit) {
    std::uint8_t const begin_atom = descriptor.bond_atom_i[bond_index];
    std::uint8_t const end_atom = descriptor.bond_atom_j[bond_index];

    for (std::uint32_t flip = 0; flip < 2; ++flip) {
        std::uint8_t const atom_2 = flip ? end_atom : begin_atom;
        std::uint8_t const atom_3 = flip ? begin_atom : end_atom;

        if (!atom_match(descriptor, atom_2, pattern.atom[1].active_atom_disjuncts(), pattern.pool)) continue;
        if (!atom_match(descriptor, atom_3, pattern.atom[2].active_atom_disjuncts(), pattern.pool)) continue;
        if (!bond_match(descriptor, bond_index, pattern.bond[1].active_bond_disjuncts())) continue;

        // Iterate neighbors of atom_2 (excluding atom_3) for atom[0] (aka :1).
        std::uint16_t const second_start = descriptor.neighbor_offset[atom_2];
        std::uint16_t const second_end = descriptor.neighbor_offset[atom_2 + 1];
        constexpr std::uint32_t no_chain_atom_k = 0xFFu;
        for (std::uint16_t first_entry = second_start; first_entry < second_end; ++first_entry) {
            std::uint8_t const atom_1 = descriptor.neighbor_atom[first_entry];
            if (atom_1 == atom_3) continue;
            std::uint16_t const bond_01 = descriptor.neighbor_bond[first_entry];
            if (!atom_match(descriptor, atom_1, pattern.atom[0].active_atom_disjuncts(), pattern.pool)) continue;
            if (!bond_match(descriptor, bond_01, pattern.bond[0].active_bond_disjuncts())) continue;

            // Branches on atom[0] depend only on (atom_1, atom_2); branches on
            // atom[1] depend on (atom_2, atom_1, atom_3). Both are independent
            // of atom_4, so check them before iterating atom_4's candidates.
            if (!check_branch_neighbors(descriptor, atom_1, pattern.atom[0], pattern.pool, atom_2, no_chain_atom_k))
                continue;
            if (!check_branch_neighbors(descriptor, atom_2, pattern.atom[1], pattern.pool, atom_1, atom_3)) continue;

            // Iterate neighbors of atom_3 (excluding atom_2) for atom[3] (aka :4).
            std::uint16_t const third_start = descriptor.neighbor_offset[atom_3];
            std::uint16_t const third_end = descriptor.neighbor_offset[atom_3 + 1];
            for (std::uint16_t fourth_entry = third_start; fourth_entry < third_end; ++fourth_entry) {
                std::uint8_t const atom_4 = descriptor.neighbor_atom[fourth_entry];
                if (atom_4 == atom_2 || atom_4 == atom_1) continue; // all 4 torsion atoms must be distinct
                std::uint16_t const bond_23 = descriptor.neighbor_bond[fourth_entry];
                if (!atom_match(descriptor, atom_4, pattern.atom[3].active_atom_disjuncts(), pattern.pool)) continue;
                if (!bond_match(descriptor, bond_23, pattern.bond[2].active_bond_disjuncts())) continue;

                // Branches on atom[2] depend on atom_4; atom[3] depends on atom_3, atom_4.
                if (!check_branch_neighbors(descriptor, atom_3, pattern.atom[2], pattern.pool, atom_2, atom_4))
                    continue;
                if (!check_branch_neighbors(descriptor, atom_4, pattern.atom[3], pattern.pool, atom_3, no_chain_atom_k))
                    continue;
                if (pattern.closure != nullptr &&
                    !ring_closure_match(descriptor, *pattern.closure, pattern.pool, atom_1, atom_2, atom_3, atom_4))
                    continue;

                if (emit(atom_1, atom_2, atom_3, atom_4, bond_index)) return true;
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
 *  @param[in] emit Callable taking `(atom_1, atom_2, atom_3, atom_4, bond_index)`; true moves on to the next bond.
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

#pragma endregion Torsion Matching

} // namespace smallfield::preprocess
