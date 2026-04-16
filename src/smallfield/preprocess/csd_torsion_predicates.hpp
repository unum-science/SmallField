/**
 *  @file csd_torsion_predicates.hpp
 *  @brief Hand-written predicate functions for recursive SMARTS sub-queries appearing
 *  in the CSD torsion preference library.
 *
 *  A handful of `$(...)` recursive sub-queries dominate the CSD torsion preference
 *  patterns. The top nine — `C=O`, `S(=O)`, `[CX3]=O`, `[C](=O)`, `[NX3H1]`,
 *  `[NX3H2]`, `C([NX3])`, `[CX3]([C])`, `[cH0](Cl)` — account for ≈90 of the 120
 *  recursive-sub-query uses (≈75%).
 *
 *  Each sub-query is re-implemented here as a short, stack-only predicate that
 *  reads only `molecule_descriptor_t` fields. No heap allocations, no RDKit calls,
 *  no virtual dispatch. The predicate IDs are compiled into the native atom
 *  constraint by the SMARTS parser (@ref csd_torsion_native.hpp) and evaluated
 *  in the hot path alongside element / degree / H-count checks — which removes
 *  the allocator-driven scaling collapse that `RDKit::SubstructMatch` triggered
 *  on patterns containing these sub-queries.
 *
 *  Semantics are intended to match RDKit's `SmartsToMol(body)` + `SubstructMatch`
 *  behaviour on the first atom of the sub-query; the companion test
 *  @ref test/test_torsion_predicates.cpp cross-validates every atom of every
 *  molecule in the ChEMBL validation set against RDKit.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <cstdint>

#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::preprocess {
using smallfield::types::atom_flag_aromatic_k;
using smallfield::types::bond_flag_aromatic_k;
using smallfield::types::bond_flag_in_ring_k;
using smallfield::types::molecule_descriptor_t;

/**
 *  @brief Identifier for a compiled recursive SMARTS sub-query.
 *
 *  The 0 value is reserved for "no predicate" so the identifier can be stored in
 *  a single `std::uint8_t` inside @ref simple_atom_constraint_t with zero-init
 *  semantics. `csd_predicate_count_k` must stay last for bounds-checking loops.
 */
enum csd_predicate_t : std::uint8_t {
    csd_predicate_none_k = 0,
    csd_predicate_c_eq_o_k,        ///< $(C=O)        — aliphatic C with a double bond to aliphatic O
    csd_predicate_s_db_o_k,        ///< $(S(=O))      — S with a double bond to aliphatic O
    csd_predicate_cx3_eq_o_k,      ///< $([CX3]=O)    — sp² C (degree 3) with a double bond to aliphatic O
    csd_predicate_c_eq_o_branch_k, ///< $([C](=O))    — aliphatic C with a double bond to aliphatic O (bracketed form)
    csd_predicate_nx3h1_k,         ///< $([NX3H1])    — aliphatic N, degree 3, exactly one H
    csd_predicate_nx3h2_k,         ///< $([NX3H2])    — aliphatic N, degree 3, exactly two H
    csd_predicate_c_nb_nx3_k,      ///< $(C([NX3]))   — aliphatic C with an aliphatic degree-3 N neighbour
    csd_predicate_cx3_nb_c_k,      ///< $([CX3]([C])) — sp² C (degree 3) with an aliphatic C neighbour
    csd_predicate_ch0_nb_cl_k,     ///< $([cH0](Cl))  — aromatic C with zero H and a Cl neighbour
    /** @name Top SubstructMatch hot-spots.
     *  Multi-dataset profiling identified two patterns dominating 75-85 % of the RDKit-fallback CPU on
     *  every tested dataset. These three predicates cover the recursive sub-queries that block the
     *  native matcher from accepting patterns p94, p95 and p114.
     *  @{ */
    csd_predicate_a_nb_cH1_k,          ///< $([a]([cH1]))           — aromatic atom with an aromatic-CH (1-H) neighbour
    csd_predicate_aH0_nb_nonring_O_k,  ///< $([aH0](!@O))           — aromatic, 0 H, with a non-ring aliphatic O
                                       ///< neighbour
    csd_predicate_c_two_cH_or_nX2H0_k, ///< $([c]([cH,nX2H0])[cH,nX2H0]) — aromatic C with two neighbours each (cH OR
                                       ///< nX2H0)
    csd_predicate_cH0_nb_NH12_k,       ///< $([cH0]([NH1,NH2]))     — aromatic C with 0 H and an aliphatic NH1 or NH2
                                       ///< neighbour
    /** @} */
    csd_predicate_count_k, ///< number of defined predicates (keep last)
};

/** @brief `$(C=O)` — aliphatic C with a double bond to aliphatic O. */
inline bool predicate_c_eq_o(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) != 0) return false;
    // Neighbour must be aliphatic O with a non-aromatic double bond.
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 8) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        std::uint16_t const bid = desc.neighbor_bond[n];
        if (desc.bond_order[bid] != 2) continue;
        if ((desc.bond_flags[bid] & bond_flag_aromatic_k) != 0) continue;
        return true;
    }
    return false;
}

/** @brief `$(S(=O))` — S with a double bond to aliphatic O. */
inline bool predicate_s_db_o(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 16) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 8) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        std::uint16_t const bid = desc.neighbor_bond[n];
        if (desc.bond_order[bid] != 2) continue;
        if ((desc.bond_flags[bid] & bond_flag_aromatic_k) != 0) continue;
        return true;
    }
    return false;
}

/** @brief `$([CX3]=O)` — aliphatic sp² C (degree 3) with a double bond to aliphatic O. */
inline bool predicate_cx3_eq_o(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) != 0) return false;
    if (desc.degree[atom_idx] != 3) return false;
    return predicate_c_eq_o(desc, atom_idx);
}

/** @brief `$([C](=O))` — aliphatic C with a double bond to aliphatic O (bracket form). */
inline bool predicate_c_eq_o_branch(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    // Same semantics as $(C=O); RDKit treats `C` and `[C]` identically for matching
    // purposes (both mean "aliphatic carbon, any H count"). The sub-query differs only
    // in branch grouping, which has no effect when the inner expression has a single
    // atom beyond the query root.
    return predicate_c_eq_o(desc, atom_idx);
}

/** @brief `$([NX3H1])` — aliphatic N, degree 3, exactly 1 hydrogen. */
inline bool predicate_nx3h1(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 7) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) != 0) return false;
    if (desc.degree[atom_idx] != 3) return false;
    if (desc.num_hydrogens[atom_idx] != 1) return false;
    return true;
}

/** @brief `$([NX3H2])` — aliphatic N, degree 3, exactly 2 hydrogens. */
inline bool predicate_nx3h2(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 7) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) != 0) return false;
    if (desc.degree[atom_idx] != 3) return false;
    if (desc.num_hydrogens[atom_idx] != 2) return false;
    return true;
}

/**
 *  @brief True iff the bond with index @p bond_idx satisfies the default SMARTS bond
 *  operator, i.e. "single or aromatic" (which is what an implicit bond means in
 *  SMARTS between two query atoms). Double and triple bonds are rejected.
 */
inline bool bond_is_single_or_aromatic(molecule_descriptor_t const &desc, std::uint16_t bond_idx) {
    std::uint8_t const bo = desc.bond_order[bond_idx];
    bool const is_aromatic = (desc.bond_flags[bond_idx] & bond_flag_aromatic_k) != 0;
    // Aromatic bonds are stored with order 1 plus the aromatic flag — both cases qualify.
    return bo == 1 || is_aromatic;
}

/** @brief `$(C([NX3]))` — aliphatic C with an aliphatic degree-3 N neighbour via a single/aromatic bond. */
inline bool predicate_c_nb_nx3(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) != 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 7) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        if (desc.degree[nb] != 3) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
    }
    return false;
}

/** @brief `$([CX3]([C]))` — aliphatic sp² C (degree 3) with an aliphatic C neighbour via a single/aromatic bond. */
inline bool predicate_cx3_nb_c(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) != 0) return false;
    if (desc.degree[atom_idx] != 3) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 6) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
    }
    return false;
}

/** @brief `$([cH0](Cl))` — aromatic C with zero hydrogens and a Cl neighbour via a single/aromatic bond. */
inline bool predicate_ch0_nb_cl(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    if (desc.num_hydrogens[atom_idx] != 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 17) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
    }
    return false;
}

/**
 *  @brief `$([a]([cH1]))` — aromatic atom of any element with at least one neighbour that is an
 *  aromatic C with exactly one H. Covers the dominant pattern p114 hot-spot in GDB-13 and Enamine
 *  combinatorial datasets.
 */
inline bool predicate_a_nb_cH1(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 6) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) == 0) continue;
        if (desc.num_hydrogens[nb] != 1) continue;
        return true;
    }
    return false;
}

/**
 *  @brief `$([aH0](!@O))` — aromatic atom with 0 hydrogens and at least one neighbour that is an
 *  aliphatic O reached via a non-ring bond. Covers the second half of pattern p114.
 */
inline bool predicate_aH0_nb_nonring_O(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    if (desc.num_hydrogens[atom_idx] != 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 8) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue; // uppercase O = aliphatic only
        std::uint16_t const bid = desc.neighbor_bond[n];
        if ((desc.bond_flags[bid] & bond_flag_in_ring_k) != 0) continue; // !@ = non-ring bond
        return true;
    }
    return false;
}

/**
 *  @brief `$([cH0]([NH1,NH2]))` — aromatic C with 0 H and at least one neighbour that is an
 *  aliphatic N with degree 3 and either 1 or 2 H. Covers pattern p249.
 */
inline bool predicate_cH0_nb_NH12(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    if (desc.num_hydrogens[atom_idx] != 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 7) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue; // uppercase N = aliphatic
        if (desc.degree[nb] != 3) continue;
        std::uint8_t const h = desc.num_hydrogens[nb];
        if (h != 1 && h != 2) continue;
        return true;
    }
    return false;
}

/**
 *  @brief `$([c]([cH,nX2H0])[cH,nX2H0])` — aromatic C with at least two distinct
 *  neighbours, each of which is either an aromatic C with exactly one H, or an aromatic N with
 *  degree 2 and zero H. Covers patterns p94 and p95.
 */
inline bool predicate_c_two_cH_or_nX2H0(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    auto neighbour_qualifies = [&](std::uint8_t nb) {
        bool const aromatic = (desc.atom_flags[nb] & atom_flag_aromatic_k) != 0;
        if (!aromatic) return false;
        std::uint8_t const element = desc.element[nb];
        std::uint8_t const h = desc.num_hydrogens[nb];
        std::uint8_t const dg = desc.degree[nb];
        // cH = aromatic C with exactly 1 H.
        if (element == 6 && h == 1) return true;
        // nX2H0 = aromatic N with degree 2 and 0 H.
        if (element == 7 && dg == 2 && h == 0) return true;
        return false;
    };
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    std::uint32_t hits = 0;
    for (std::uint16_t n = start; n < end; ++n) {
        if (neighbour_qualifies(desc.neighbor_atom[n])) ++hits;
        if (hits >= 2) return true;
    }
    return false;
}

/**
 *  @brief Evaluate a recursive-SMARTS predicate against an atom in a descriptor.
 *
 *  @param desc Molecule descriptor filled by @ref build_descriptor.
 *  @param atom_idx Atom to test (must be < desc.num_atoms).
 *  @param predicate Predicate ID (must be < csd_predicate_count_k).
 *  @return true if @p atom_idx satisfies the sub-query, false otherwise.
 *
 *  @note Passing `csd_predicate_none_k` returns true — treated as "no constraint".
 */
inline bool csd_predicate_check(molecule_descriptor_t const &desc, std::uint32_t atom_idx, csd_predicate_t predicate) {
    switch (predicate) {
    case csd_predicate_none_k: return true;
    case csd_predicate_c_eq_o_k: return predicate_c_eq_o(desc, atom_idx);
    case csd_predicate_s_db_o_k: return predicate_s_db_o(desc, atom_idx);
    case csd_predicate_cx3_eq_o_k: return predicate_cx3_eq_o(desc, atom_idx);
    case csd_predicate_c_eq_o_branch_k: return predicate_c_eq_o_branch(desc, atom_idx);
    case csd_predicate_nx3h1_k: return predicate_nx3h1(desc, atom_idx);
    case csd_predicate_nx3h2_k: return predicate_nx3h2(desc, atom_idx);
    case csd_predicate_c_nb_nx3_k: return predicate_c_nb_nx3(desc, atom_idx);
    case csd_predicate_cx3_nb_c_k: return predicate_cx3_nb_c(desc, atom_idx);
    case csd_predicate_ch0_nb_cl_k: return predicate_ch0_nb_cl(desc, atom_idx);
    case csd_predicate_a_nb_cH1_k: return predicate_a_nb_cH1(desc, atom_idx);
    case csd_predicate_aH0_nb_nonring_O_k: return predicate_aH0_nb_nonring_O(desc, atom_idx);
    case csd_predicate_c_two_cH_or_nX2H0_k: return predicate_c_two_cH_or_nX2H0(desc, atom_idx);
    case csd_predicate_cH0_nb_NH12_k: return predicate_cH0_nb_NH12(desc, atom_idx);
    case csd_predicate_count_k: return false;
    }
    return false;
}

} // namespace smallfield::preprocess
