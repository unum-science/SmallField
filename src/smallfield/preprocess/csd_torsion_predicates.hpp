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
    /** @name Sulfonyl / carbonyl / fluorine families.
     *  Cover 40+ fallback patterns dominated by sulfonamide (`S(=O)=O`) and
     *  haloaromatic predicates in the CSD torsion library.
     *  @{ */
    csd_predicate_sulfonyl_k,          ///< $(S(=O)=O) / $(S(=O)(=O)) — S with ≥2 double-bonded aliphatic O
    csd_predicate_cx3_eq_non_o_k,      ///< $([CX3](=[!O])) — aliphatic sp² C (degree 3) with a double bond to a non-O
    csd_predicate_c_trifluoro_k,       ///< $(C([F])([F])[F]) — aliphatic C with ≥3 F neighbours (CF3)
    csd_predicate_cx3_nb_c_h_k,        ///< $([CX3]([C])([H])) — aliphatic sp² C with both a C neighbour and ≥1 H
    csd_predicate_c_nx3_and_aliph_n_k, ///< $(C([NX3])N) — aliphatic C with NX3 and aliphatic N (two distinct) neighbours
    csd_predicate_c_nx3_and_arom_n_k,  ///< $(C([NX3])n) — aliphatic C with NX3 and aromatic n (two distinct) neighbours
    csd_predicate_n_nb_cx4_k,          ///< $([N][CX4]) / $(N[CX4]) / $([NH0][CX4]) — aliphatic N with a CX4 neighbour
    csd_predicate_cH0_nb_F_k,          ///< $([cH0]F) / $([cH0](F)) — aromatic C with 0 H and a F neighbour
    csd_predicate_cH0_nb_OH1_k,        ///< $([cH0][OH1]) — aromatic C with 0 H and a hydroxyl O neighbour
    csd_predicate_cH0_nb_OH0_k,        ///< $([cH0][OH0]) — aromatic C with 0 H and an O-with-no-H (ether/carbonyl) neighbour
    csd_predicate_cH0_nb_cx3_k,        ///< $([cH0]([CX3])) — aromatic C with 0 H and an aliphatic sp² C neighbour
    csd_predicate_c_nb_aliph_O_k,      ///< $(cO) — aromatic C with an aliphatic O neighbour
    csd_predicate_a_nb_OH1_k,          ///< $(a[OH1]) — aromatic atom with a hydroxyl O neighbour
    csd_predicate_a_nb_NH12_k,         ///< $(a[NH1,NH2]) — aromatic atom with an aliphatic NH1/NH2 neighbour
    csd_predicate_c_nb_NH12_k,         ///< $(c([NH1,NH2])) / $([c]([NH1,NH2])) — aromatic C with an aliphatic NH1/NH2 neighbour
    /** @} */
    /** @name Round-2 additions (2026-04-19).  Cover the last simple single-hop bodies in the residual fallback. @{ */
    csd_predicate_cH0_eq_o_k,          ///< $([cH0](=O)) — aromatic C with 0 H and a double bond to aliphatic O
    csd_predicate_c_not_ring_c_k,      ///< $(c!@c) — aromatic C with an aromatic-C neighbour via a non-ring bond
    csd_predicate_nN_nb_aromatic_k,    ///< $([n,N](-a)) — aromatic n OR aliphatic N with an aromatic-atom neighbour via a single bond
    csd_predicate_a_nb_nH0_or_o_k,     ///< $([a]([nH0,o])) — aromatic atom with an aromatic-N-0H or aromatic-O neighbour
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
 *  @brief True iff the bond with index @p bond_idx satisfies the default SMARTS
 *  bond operator, i.e. "single or aromatic" (the meaning of an implicit bond
 *  between two query atoms).  Aromatic bonds are normalised to order 1 in
 *  @ref build_descriptor, so the underlying check is simply "order == 1".
 *
 *  Pathological RDKit inputs (e.g. a triple bond inside an aromatised ring:
 *  `C#[N+]` in a 6-membered aromatic ring) produce an order-3 bond that
 *  nevertheless carries the aromatic flag.  RDKit's own SMARTS matcher
 *  rejects such a bond against the implicit `-,:` query because the bond
 *  type remains TRIPLE, so matching `order == 1` gives the correct behaviour.
 */
inline bool bond_is_single_or_aromatic(molecule_descriptor_t const &desc, std::uint16_t bond_idx) {
    return desc.bond_order[bond_idx] == 1;
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
 *  aromatic C with exactly one H, reached via a single/aromatic bond.
 *  Covers the dominant pattern p114 hot-spot in GDB-13 and Enamine combinatorial datasets.
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
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
    }
    return false;
}

/**
 *  @brief `$([aH0](!@O))` — aromatic atom with 0 hydrogens and at least one neighbour that is an
 *  aliphatic O reached via a non-ring bond.  The SMARTS operator `!@` negates ring-membership
 *  only; it does NOT constrain bond order, so double-bonded aliphatic O (e.g. the ring-exterior
 *  `c(=O)` carbonyl in a nitroaromatic side chain) also matches.
 */
inline bool predicate_aH0_nb_nonring_O(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    if (desc.num_hydrogens[atom_idx] != 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 8) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        std::uint16_t const bid = desc.neighbor_bond[n];
        if ((desc.bond_flags[bid] & bond_flag_in_ring_k) != 0) continue;
        return true;
    }
    return false;
}

/**
 *  @brief `$([cH0][NH1,NH2])` / `$([cH0]([NH1,NH2]))` — aromatic C with 0 H
 *  and a neighbouring aliphatic N carrying exactly 1 or 2 hydrogens.  SMARTS
 *  primitives `NH1` and `NH2` do not constrain total degree (a protonated
 *  amine like `[NH2+]` bonded to two heavy atoms still qualifies), so the
 *  check is element + H-count + aromaticity only.
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
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        std::uint8_t const h = desc.num_hydrogens[nb];
        if (h != 1 && h != 2) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
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
    auto neighbour_qualifies = [&](std::uint8_t nb, std::uint16_t bid) {
        if (!bond_is_single_or_aromatic(desc, bid)) return false;
        bool const aromatic = (desc.atom_flags[nb] & atom_flag_aromatic_k) != 0;
        if (!aromatic) return false;
        std::uint8_t const element = desc.element[nb];
        std::uint8_t const h = desc.num_hydrogens[nb];
        std::uint8_t const dg = desc.degree[nb];
        if (element == 6 && h == 1) return true;         // cH
        if (element == 7 && dg == 2 && h == 0) return true; // nX2H0
        return false;
    };
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    std::uint32_t hits = 0;
    for (std::uint16_t n = start; n < end; ++n) {
        if (neighbour_qualifies(desc.neighbor_atom[n], desc.neighbor_bond[n])) ++hits;
        if (hits >= 2) return true;
    }
    return false;
}

/** @brief `$(S(=O)=O)` / `$(S(=O)(=O))` — sulfur with ≥2 aliphatic-O double-bond neighbours (sulfonyl). */
inline bool predicate_sulfonyl(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 16) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    std::uint32_t double_o_count = 0;
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 8) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        std::uint16_t const bid = desc.neighbor_bond[n];
        if (desc.bond_order[bid] != 2) continue;
        if ((desc.bond_flags[bid] & bond_flag_aromatic_k) != 0) continue;
        if (++double_o_count >= 2) return true;
    }
    return false;
}

/** @brief `$([CX3](=[!O]))` — aliphatic sp² C with a double bond to a non-O atom. */
inline bool predicate_cx3_eq_non_o(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) != 0) return false;
    if (desc.degree[atom_idx] != 3) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint16_t const bid = desc.neighbor_bond[n];
        if (desc.bond_order[bid] != 2) continue;
        if ((desc.bond_flags[bid] & bond_flag_aromatic_k) != 0) continue;
        if (desc.element[desc.neighbor_atom[n]] == 8) continue;
        return true;
    }
    return false;
}

/** @brief `$(C([F])([F])[F])` — aliphatic C with ≥3 F neighbours (CF3). */
inline bool predicate_c_trifluoro(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) != 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    std::uint32_t f_count = 0;
    for (std::uint16_t n = start; n < end; ++n) {
        if (desc.element[desc.neighbor_atom[n]] == 9) ++f_count;
        if (f_count >= 3) return true;
    }
    return false;
}

/** @brief `$([CX3]([C])([H]))` — aliphatic sp² C (degree 3) with both a C neighbour and ≥1 attached H. */
inline bool predicate_cx3_nb_c_h(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) != 0) return false;
    if (desc.degree[atom_idx] != 3) return false;
    if (desc.num_hydrogens[atom_idx] == 0) return false;
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

/**
 *  @brief Shared helper for @ref predicate_c_nx3_and_aliph_n and @ref
 *  predicate_c_nx3_and_arom_n: aliphatic C with two DISTINCT N neighbours —
 *  one aliphatic NX3 (degree 3, not aromatic) via single/aromatic bond, and a
 *  second N whose aromaticity matches @p second_aromatic.
 */
inline bool predicate_c_nx3_and_n_impl(molecule_descriptor_t const &desc, std::uint32_t atom_idx,
                                       bool second_aromatic) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) != 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    std::uint8_t nx3_atom = 0xFF;
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 7) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        bool const nb_aromatic = (desc.atom_flags[nb] & atom_flag_aromatic_k) != 0;
        if (!nb_aromatic && desc.degree[nb] == 3) {
            nx3_atom = nb;
            break;
        }
    }
    if (nx3_atom == 0xFF) return false;
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (nb == nx3_atom) continue;
        if (desc.element[nb] != 7) continue;
        bool const nb_aromatic = (desc.atom_flags[nb] & atom_flag_aromatic_k) != 0;
        if (nb_aromatic != second_aromatic) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
    }
    return false;
}

/** @brief `$(C([NX3])N)` — aliphatic C with an NX3 and a second aliphatic N. */
inline bool predicate_c_nx3_and_aliph_n(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    return predicate_c_nx3_and_n_impl(desc, atom_idx, /*second_aromatic=*/false);
}

/** @brief `$(C([NX3])n)` — aliphatic C with an NX3 and a second aromatic n. */
inline bool predicate_c_nx3_and_arom_n(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    return predicate_c_nx3_and_n_impl(desc, atom_idx, /*second_aromatic=*/true);
}

/** @brief `$([N][CX4])` / `$(N[CX4])` / `$([NH0][CX4])` — aliphatic N with ≥1 aliphatic CX4 neighbour. */
inline bool predicate_n_nb_cx4(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 7) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) != 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 6) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        if (desc.degree[nb] != 4) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
    }
    return false;
}

/** @brief `$([cH0]F)` / `$([cH0](F))` — aromatic C with 0 H and a F neighbour (single/aromatic bond). */
inline bool predicate_cH0_nb_F(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    if (desc.num_hydrogens[atom_idx] != 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        if (desc.element[desc.neighbor_atom[n]] != 9) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
    }
    return false;
}

/** @brief `$([cH0][OH1])` — aromatic C with 0 H and an aliphatic hydroxyl O (OH1) neighbour (single/aromatic bond). */
inline bool predicate_cH0_nb_OH1(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    if (desc.num_hydrogens[atom_idx] != 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 8) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        if (desc.num_hydrogens[nb] != 1) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
    }
    return false;
}

/**
 *  @brief `$([cH0][OH0])` — aromatic C with 0 H and an aliphatic O-with-zero-H
 *  neighbour connected by a single/aromatic bond.  The double-bond carbonyl form
 *  `c=O` does NOT match here because SMARTS chain `[c][O]` means single/aromatic.
 */
inline bool predicate_cH0_nb_OH0(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    if (desc.num_hydrogens[atom_idx] != 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 8) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        if (desc.num_hydrogens[nb] != 0) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
    }
    return false;
}

/** @brief `$([cH0]([CX3]))` — aromatic C with 0 H and an aliphatic sp² C neighbour (degree 3, not aromatic). */
inline bool predicate_cH0_nb_cx3(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    if (desc.num_hydrogens[atom_idx] != 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 6) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        if (desc.degree[nb] != 3) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
    }
    return false;
}

/** @brief `$(cO)` — aromatic C with an aliphatic O neighbour (single/aromatic bond). */
inline bool predicate_c_nb_aliph_O(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 8) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
    }
    return false;
}

/** @brief `$(a[OH1])` — aromatic atom with a hydroxyl O (OH1) neighbour (single/aromatic bond). */
inline bool predicate_a_nb_OH1(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 8) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        if (desc.num_hydrogens[nb] != 1) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
    }
    return false;
}

/**
 *  @brief `$(a[NH1,NH2])` — aromatic atom (any element) with a neighbouring
 *  aliphatic N carrying 1 or 2 H.  Same degree-agnostic NH1/NH2 semantics as
 *  @ref predicate_cH0_nb_NH12.
 */
inline bool predicate_a_nb_NH12(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 7) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        std::uint8_t const h = desc.num_hydrogens[nb];
        if (h != 1 && h != 2) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
    }
    return false;
}

/** @brief `$(c([NH1,NH2]))` / `$([c]([NH1,NH2]))` — aromatic C with an aliphatic NH1/NH2 neighbour. */
inline bool predicate_c_nb_NH12(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 7) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) != 0) continue;
        std::uint8_t const h = desc.num_hydrogens[nb];
        if (h != 1 && h != 2) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
    }
    return false;
}

/** @brief `$([cH0](=O))` — aromatic C with 0 H and a double bond to aliphatic O. */
inline bool predicate_cH0_eq_o(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    if (desc.num_hydrogens[atom_idx] != 0) return false;
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

/** @brief `$(c!@c)` — aromatic C with an aromatic-C neighbour connected by a non-ring bond. */
inline bool predicate_c_not_ring_c(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 6) return false;
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] != 6) continue;
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) == 0) continue;
        std::uint16_t const bid = desc.neighbor_bond[n];
        if ((desc.bond_flags[bid] & bond_flag_in_ring_k) != 0) continue;
        return true;
    }
    return false;
}

/**
 *  @brief `$([n,N](-a))` — atom is either an aromatic n or an aliphatic N, with at
 *  least one aromatic-atom neighbour reached via an explicit single bond.
 */
inline bool predicate_nN_nb_aromatic(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if (desc.element[atom_idx] != 7) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) == 0) continue;
        std::uint16_t const bid = desc.neighbor_bond[n];
        if (desc.bond_order[bid] != 1) continue;
        if ((desc.bond_flags[bid] & bond_flag_aromatic_k) != 0) continue;
        return true;
    }
    return false;
}

/**
 *  @brief `$([a]([nH0,o]))` — aromatic atom with a neighbour that is either
 *  aromatic nitrogen with 0 H or aromatic oxygen, via single/aromatic bond.
 */
inline bool predicate_a_nb_nH0_or_o(molecule_descriptor_t const &desc, std::uint32_t atom_idx) {
    if ((desc.atom_flags[atom_idx] & atom_flag_aromatic_k) == 0) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if ((desc.atom_flags[nb] & atom_flag_aromatic_k) == 0) continue;
        std::uint8_t const el = desc.element[nb];
        if (!((el == 7 && desc.num_hydrogens[nb] == 0) || el == 8)) continue;
        if (!bond_is_single_or_aromatic(desc, desc.neighbor_bond[n])) continue;
        return true;
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
    case csd_predicate_sulfonyl_k: return predicate_sulfonyl(desc, atom_idx);
    case csd_predicate_cx3_eq_non_o_k: return predicate_cx3_eq_non_o(desc, atom_idx);
    case csd_predicate_c_trifluoro_k: return predicate_c_trifluoro(desc, atom_idx);
    case csd_predicate_cx3_nb_c_h_k: return predicate_cx3_nb_c_h(desc, atom_idx);
    case csd_predicate_c_nx3_and_aliph_n_k: return predicate_c_nx3_and_aliph_n(desc, atom_idx);
    case csd_predicate_c_nx3_and_arom_n_k: return predicate_c_nx3_and_arom_n(desc, atom_idx);
    case csd_predicate_n_nb_cx4_k: return predicate_n_nb_cx4(desc, atom_idx);
    case csd_predicate_cH0_nb_F_k: return predicate_cH0_nb_F(desc, atom_idx);
    case csd_predicate_cH0_nb_OH1_k: return predicate_cH0_nb_OH1(desc, atom_idx);
    case csd_predicate_cH0_nb_OH0_k: return predicate_cH0_nb_OH0(desc, atom_idx);
    case csd_predicate_cH0_nb_cx3_k: return predicate_cH0_nb_cx3(desc, atom_idx);
    case csd_predicate_c_nb_aliph_O_k: return predicate_c_nb_aliph_O(desc, atom_idx);
    case csd_predicate_a_nb_OH1_k: return predicate_a_nb_OH1(desc, atom_idx);
    case csd_predicate_a_nb_NH12_k: return predicate_a_nb_NH12(desc, atom_idx);
    case csd_predicate_c_nb_NH12_k: return predicate_c_nb_NH12(desc, atom_idx);
    case csd_predicate_cH0_eq_o_k: return predicate_cH0_eq_o(desc, atom_idx);
    case csd_predicate_c_not_ring_c_k: return predicate_c_not_ring_c(desc, atom_idx);
    case csd_predicate_nN_nb_aromatic_k: return predicate_nN_nb_aromatic(desc, atom_idx);
    case csd_predicate_a_nb_nH0_or_o_k: return predicate_a_nb_nH0_or_o(desc, atom_idx);
    case csd_predicate_count_k: return false;
    }
    return false;
}

} // namespace smallfield::preprocess
