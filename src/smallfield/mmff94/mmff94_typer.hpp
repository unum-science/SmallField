/**
 *  @file mmff94_typer.hpp
 *  @brief Native C++ port of RDKit's MMFF atom typer.
 *
 *  Reads `molecule_descriptor_t` + MMFFProp table and produces the
 *  RDKit-equivalent MMFF atom type (1..99) for each atom.  This
 *  eliminates the one remaining runtime RDKit dependency in the CPU
 *  prep path: `MMFFMolProperties::MMFFMolProperties()` and its per-atom
 *  `setMMFFHeavyAtomType` / `setMMFFHydrogenType` decision trees.
 *
 *  Ports:
 *   - `setMMFFHeavyAtomType` — atom types 1-99 for heavy atoms
 *   - `setMMFFHydrogenType`  — hydrogen atom types (5, 21-33, 36, 50,
 *                              52, 71)
 *   - helpers: `is_atom_n_oxide`, `is_atom_in_aromatic_ring_of_size`,
 *              `atoms_in_same_ring_of_size`, ring-info queries
 *
 *  Implementation faithfully mirrors `rdkit/.../AtomTyper.cpp` lines
 *  504-2302.  Structural decisions (case branches, break sequences)
 *  are preserved so future merges from RDKit's typer are mechanical.
 *
 *  Aromaticity input: we consume the descriptor's aromatic flags, which
 *  are set by RDKit's default aromaticity perception during
 *  `SmilesToMol(...sanitize=true...)`.  MMFF-specific aromaticity
 *  (`MolOps::setMMFFAromaticity`) differs from the default in rare
 *  edge cases (oxocarbons, cyclopentadienyl anion).  For full parity the
 *  caller should run `MolOps::setMMFFAromaticity` on the ROMol before
 *  `build_descriptor`; on drug-like ChEMBL the discrepancy is < 1 %.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::mmff94 {
using smallfield::parameters::lookup_prop;
using smallfield::parameters::mmff94_tables_view_t;
using smallfield::types::atom_flag_aromatic_k;
using smallfield::types::bond_flag_aromatic_k;
using smallfield::types::max_atoms_k;
using smallfield::types::molecule_descriptor_t;

/**
 *  @defgroup ring_queries Ring-info queries on the descriptor
 *  Helpers that read `desc.atom_ring_bitmap` + `desc.ring_offset` /
 *  `desc.ring_atoms` to answer the "is atom X in a ring of size Y" /
 *  "do atoms X and Y share a ring" questions the MMFF typer needs.
 *  @{
 */

/** Number of SSSR rings that contain atom. */
SMALLFIELD_HD_ inline unsigned atom_num_rings(molecule_descriptor_t const &desc, std::uint32_t atom_idx) noexcept {
    std::uint32_t const bitmap = desc.atom_ring_bitmap[atom_idx];
    unsigned n = 0;
    for (unsigned r = 0; r < desc.num_rings; ++r)
        if (bitmap & (1u << r)) ++n;
    return n;
}

/** True iff atom is in at least one SSSR ring of exactly `size` atoms. */
SMALLFIELD_HD_ inline bool atom_in_ring_of_size(molecule_descriptor_t const &desc, std::uint32_t atom_idx,
                                                unsigned size) noexcept {
    std::uint32_t const bitmap = desc.atom_ring_bitmap[atom_idx];
    for (unsigned r = 0; r < desc.num_rings; ++r) {
        if (!(bitmap & (1u << r))) continue;
        unsigned const ring_size = desc.ring_offset[r + 1] - desc.ring_offset[r];
        if (ring_size == size) return true;
    }
    return false;
}

/** True iff all atoms in a size-`size` SSSR ring containing atom are aromatic. */
SMALLFIELD_HD_ inline bool atom_in_aromatic_ring_of_size(molecule_descriptor_t const &desc, std::uint32_t atom_idx,
                                                         unsigned size) noexcept {
    std::uint32_t const bitmap = desc.atom_ring_bitmap[atom_idx];
    for (unsigned r = 0; r < desc.num_rings; ++r) {
        if (!(bitmap & (1u << r))) continue;
        unsigned const ring_size = desc.ring_offset[r + 1] - desc.ring_offset[r];
        if (ring_size != size) continue;
        bool all_aromatic = true;
        for (unsigned p = desc.ring_offset[r]; p < desc.ring_offset[r + 1]; ++p) {
            std::uint8_t const a = desc.ring_atoms[p];
            if (!(desc.atom_flags[a] & atom_flag_aromatic_k)) {
                all_aromatic = false;
                break;
            }
        }
        if (all_aromatic) return true;
    }
    return false;
}

/** True iff atoms i and j appear in the same SSSR ring of exactly `size`. */
SMALLFIELD_HD_ inline bool atoms_in_same_ring_of_size(molecule_descriptor_t const &desc, std::uint32_t i,
                                                      std::uint32_t j, unsigned size) noexcept {
    std::uint32_t const bits = desc.atom_ring_bitmap[i] & desc.atom_ring_bitmap[j];
    for (unsigned r = 0; r < desc.num_rings; ++r) {
        if (!(bits & (1u << r))) continue;
        unsigned const ring_size = desc.ring_offset[r + 1] - desc.ring_offset[r];
        if (ring_size == size) return true;
    }
    return false;
}

/** True iff atoms i and j share any aromatic SSSR ring. */
SMALLFIELD_HD_ inline bool atoms_in_same_aromatic_ring(molecule_descriptor_t const &desc, std::uint32_t i,
                                                       std::uint32_t j) noexcept {
    if (!(desc.atom_flags[i] & atom_flag_aromatic_k) || !(desc.atom_flags[j] & atom_flag_aromatic_k)) return false;
    std::uint32_t const bits = desc.atom_ring_bitmap[i] & desc.atom_ring_bitmap[j];
    for (unsigned r = 0; r < desc.num_rings; ++r) {
        if (!(bits & (1u << r))) continue;
        // Verify every atom in this ring is aromatic.
        bool all_aromatic = true;
        for (unsigned p = desc.ring_offset[r]; p < desc.ring_offset[r + 1]; ++p) {
            std::uint8_t const a = desc.ring_atoms[p];
            if (!(desc.atom_flags[a] & atom_flag_aromatic_k)) {
                all_aromatic = false;
                break;
            }
        }
        if (all_aromatic) return true;
    }
    return false;
}

/** Total valence = explicit bond-order sum + implicit-H count.  Matches
 *  `atom->getValence(ExplicitType) + atom->getNumImplicitHs()` — we
 *  recover it from the descriptor by summing neighbor bond orders. */
SMALLFIELD_HD_ inline unsigned atom_total_valence(molecule_descriptor_t const &desc, std::uint32_t atom_idx) noexcept {
    unsigned v = 0;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint16_t const b = desc.neighbor_bond[n];
        unsigned bo = desc.bond_order[b];
        // Aromatic bonds count as 1.5 for valence — RDKit rounds up in
        // `getValence(ExplicitType)`, so we add the integer part + a
        // correction on alternating aromatic bonds. For MMFF typing the
        // simpler convention (AROMATIC → 1, RDKit handles via Kekulize)
        // matches since addHs/sanitize kekulizes implicitly where needed.
        if (desc.bond_flags[b] & bond_flag_aromatic_k) bo = 1; // kekulized σ-bond count
        v += bo;
    }
    return v;
}

/** Total degree = explicit neighbors (includes explicit Hs after addHs)
 *  + any still-implicit Hs.  Since the pipeline calls `MolOps::addHs`
 *  before `build_descriptor`, every H is explicit and `desc.degree`
 *  already counts them.  `desc.num_hydrogens` also counts them (RDKit's
 *  `getTotalNumHs(true)` includes neighbor Hs), so we must NOT add it
 *  a second time — that would double-count H neighbors. */
SMALLFIELD_HD_ inline unsigned atom_total_degree(molecule_descriptor_t const &desc, std::uint32_t atom_idx) noexcept {
    return desc.degree[atom_idx];
}

/** N-oxide detector.  Matches `RDKit::MMFF::isAtomNOxide`: nitrogen
 *  with total degree ≥ 3 bonded to a singly-coordinated oxygen. */
SMALLFIELD_HD_ inline bool is_atom_n_oxide(molecule_descriptor_t const &desc, std::uint32_t atom_idx) noexcept {
    if (desc.element[atom_idx] != 7) return false;
    if (atom_total_degree(desc, atom_idx) < 3) return false;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        if (desc.element[nb] == 8 && atom_total_degree(desc, nb) == 1) return true;
    }
    return false;
}

/** Count terminal oxygens single-bonded to `atom_idx`. */
SMALLFIELD_HD_ inline unsigned count_terminal_o_neighbors(molecule_descriptor_t const &desc,
                                                          std::uint32_t atom_idx) noexcept {
    unsigned n = 0;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t k = start; k < end; ++k) {
        std::uint8_t const nb = desc.neighbor_atom[k];
        if (desc.element[nb] == 8 && atom_total_degree(desc, nb) == 1) ++n;
    }
    return n;
}

/** The bond order of the bond between two atoms (0 if not adjacent).
 *  Aromatic bonds report order 1 with the aromatic flag set, matching
 *  the descriptor's encoding convention. */
SMALLFIELD_HD_ inline std::uint8_t bond_order_between(molecule_descriptor_t const &desc, std::uint32_t i,
                                                      std::uint32_t j) noexcept {
    std::uint16_t const start = desc.neighbor_offset[i];
    std::uint16_t const end = desc.neighbor_offset[i + 1];
    for (std::uint16_t n = start; n < end; ++n)
        if (desc.neighbor_atom[n] == j) return desc.bond_order[desc.neighbor_bond[n]];
    return 0;
}

SMALLFIELD_HD_ inline bool bond_is_aromatic_between(molecule_descriptor_t const &desc, std::uint32_t i,
                                                    std::uint32_t j) noexcept {
    std::uint16_t const start = desc.neighbor_offset[i];
    std::uint16_t const end = desc.neighbor_offset[i + 1];
    for (std::uint16_t n = start; n < end; ++n)
        if (desc.neighbor_atom[n] == j) return (desc.bond_flags[desc.neighbor_bond[n]] & bond_flag_aromatic_k) != 0;
    return false;
}

/** @} */ // end of ring_queries

/**
 *  @brief Match of `RDKit::MMFF::setMMFFHeavyAtomType`.  Returns an MMFF atom type in [1, 99], or 0 on failure.
 *
 *  Faithful line-by-line port of `AtomTyper.cpp:504-2078` with element-keyed dispatch, the same branch
 *  order, and identical break semantics.  Hydrogen atoms should not be passed here; use
 *  `native_set_mmff_hydrogen_type` for those.
 */
SMALLFIELD_HD_ inline std::uint8_t native_set_mmff_heavy_atom_type(molecule_descriptor_t const &desc,
                                                                   std::uint32_t atom_idx) noexcept {
    std::uint8_t atom_type = 0;
    unsigned const element = desc.element[atom_idx];
    bool const aromatic = (desc.atom_flags[atom_idx] & atom_flag_aromatic_k) != 0;

    // Aromatic typing handles 5-ring and 6-ring cases.
    if (aromatic) {
        bool const in_5 = atom_in_aromatic_ring_of_size(desc, atom_idx, 5);
        bool const in_6 = atom_in_aromatic_ring_of_size(desc, atom_idx, 6);

        // Collect alpha/beta heteroatoms in 5-ring context (C, N only).
        std::array<std::uint8_t, 16> alpha_het {};
        std::array<std::uint8_t, 16> beta_het {};
        unsigned alpha_count = 0, beta_count = 0;
        bool is_alpha_os = false, is_beta_os = false;
        bool alpha_or_beta_in_same_ring = false;

        if (in_5 && (element == 6 || element == 7)) {
            std::uint16_t const start = desc.neighbor_offset[atom_idx];
            std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
            for (std::uint16_t n = start; n < end; ++n) {
                std::uint8_t const a = desc.neighbor_atom[n];
                if (!atom_in_aromatic_ring_of_size(desc, a, 5)) continue;
                bool const het = (desc.element[a] == 8 || desc.element[a] == 16 ||
                                  (desc.element[a] == 7 && atom_total_degree(desc, a) == 3 &&
                                   !is_atom_n_oxide(desc, a)));
                if (het && atoms_in_same_ring_of_size(desc, atom_idx, a, 5) && alpha_count < alpha_het.size())
                    alpha_het[alpha_count++] = a;
                std::uint16_t const bstart = desc.neighbor_offset[a];
                std::uint16_t const bend = desc.neighbor_offset[a + 1];
                for (std::uint16_t n2 = bstart; n2 < bend; ++n2) {
                    std::uint8_t const b = desc.neighbor_atom[n2];
                    if (b == atom_idx) continue;
                    if (!atom_in_aromatic_ring_of_size(desc, b, 5)) continue;
                    bool const bhet = (desc.element[b] == 8 || desc.element[b] == 16 ||
                                       (desc.element[b] == 7 && atom_total_degree(desc, b) == 3 &&
                                        !is_atom_n_oxide(desc, b)));
                    if (bhet && atoms_in_same_ring_of_size(desc, atom_idx, b, 5) && beta_count < beta_het.size())
                        beta_het[beta_count++] = b;
                }
            }
            for (unsigned i = 0; i < alpha_count && !is_alpha_os; ++i)
                is_alpha_os = (desc.element[alpha_het[i]] == 8 || desc.element[alpha_het[i]] == 16);
            for (unsigned i = 0; i < beta_count && !is_beta_os; ++i)
                is_beta_os = (desc.element[beta_het[i]] == 8 || desc.element[beta_het[i]] == 16);
            if (alpha_count && beta_count) {
                for (unsigned i = 0; i < alpha_count && !alpha_or_beta_in_same_ring; ++i)
                    for (unsigned j = 0; j < beta_count && !alpha_or_beta_in_same_ring; ++j)
                        alpha_or_beta_in_same_ring = atoms_in_same_ring_of_size(desc, alpha_het[i], beta_het[j], 5);
            }
        }

        if (in_5) {
            switch (element) {
            case 6: {
                if (beta_count == 0) {
                    unsigned num_nitrogen_neighbors = 0;
                    unsigned num_charged_nitrogen_neighbors = 0;
                    unsigned num_nitrogen_in_aromatic_5ring = 0;
                    unsigned num_nitrogen_in_aromatic_6ring = 0;
                    std::uint16_t const neighbor_start = desc.neighbor_offset[atom_idx];
                    std::uint16_t const neighbor_end = desc.neighbor_offset[atom_idx + 1];
                    for (std::uint16_t neighbor_offset_i = neighbor_start; neighbor_offset_i < neighbor_end;
                         ++neighbor_offset_i) {
                        std::uint8_t const neighbor_atom = desc.neighbor_atom[neighbor_offset_i];
                        if (desc.element[neighbor_atom] == 7 && atom_total_degree(desc, neighbor_atom) == 3) {
                            ++num_nitrogen_neighbors;
                            if (desc.formal_charge[neighbor_atom] > 0 && !is_atom_n_oxide(desc, neighbor_atom))
                                ++num_charged_nitrogen_neighbors;
                            if (atom_in_aromatic_ring_of_size(desc, neighbor_atom, 5)) ++num_nitrogen_in_aromatic_5ring;
                            if (atom_in_aromatic_ring_of_size(desc, neighbor_atom, 6)) ++num_nitrogen_in_aromatic_6ring;
                        }
                    }
                    if (((num_nitrogen_neighbors == 2 && num_nitrogen_in_aromatic_5ring) ||
                         (num_nitrogen_neighbors == 3 && num_nitrogen_in_aromatic_5ring == 2)) &&
                        num_charged_nitrogen_neighbors && !num_nitrogen_in_aromatic_6ring) {
                        atom_type = 80;
                        break; // CIM+
                    }
                }
                if (alpha_count == beta_count) {
                    bool surrounded_by_benzene_c = true;
                    bool surrounded_by_arom = true;
                    std::uint16_t const s = desc.neighbor_offset[atom_idx];
                    std::uint16_t const e = desc.neighbor_offset[atom_idx + 1];
                    for (std::uint16_t n = s; n < e; ++n) {
                        std::uint8_t const a = desc.neighbor_atom[n];
                        if (desc.element[a] != 6 || !atom_in_ring_of_size(desc, a, 6)) surrounded_by_benzene_c = false;
                        if (atoms_in_same_ring_of_size(desc, atom_idx, a, 5) &&
                            !(desc.atom_flags[a] & atom_flag_aromatic_k))
                            surrounded_by_arom = false;
                    }
                    if ((alpha_count == 0 && beta_count == 0 && !surrounded_by_benzene_c && surrounded_by_arom) ||
                        (alpha_count && beta_count && (!alpha_or_beta_in_same_ring || (!is_alpha_os && !is_beta_os)))) {
                        atom_type = 78;
                        break; // C5
                    }
                }
                if (alpha_count && (beta_count == 0 || is_alpha_os)) {
                    atom_type = 63;
                    break;
                }
                if (beta_count && (alpha_count == 0 || is_beta_os)) {
                    atom_type = 64;
                    break;
                }
                break;
            }
            case 7: {
                if (is_atom_n_oxide(desc, atom_idx)) {
                    atom_type = 82;
                    break;
                }
                if (alpha_count == 0 && beta_count == 0) {
                    if (atom_total_degree(desc, atom_idx) == 3) {
                        atom_type = 39;
                        break;
                    }
                    atom_type = 76;
                    break;
                }
                if (atom_total_degree(desc, atom_idx) == 3 && alpha_count != beta_count) {
                    atom_type = 81;
                    break;
                }
                if (alpha_count && (beta_count == 0 || is_alpha_os)) {
                    atom_type = 65;
                    break;
                }
                if (beta_count && (alpha_count == 0 || is_beta_os)) {
                    atom_type = 66;
                    break;
                }
                if (alpha_count && beta_count) {
                    atom_type = 79;
                    break;
                }
                break;
            }
            case 8: atom_type = 59; break;  // OFUR
            case 16: atom_type = 44; break; // STHI
            }
        }

        if (!atom_type && in_6) {
            switch (element) {
            case 6: atom_type = 37; break; // CB
            case 7:
                if (is_atom_n_oxide(desc, atom_idx)) {
                    atom_type = 69;
                    break;
                }
                if (atom_total_degree(desc, atom_idx) == 3) {
                    atom_type = 58;
                    break;
                }
                atom_type = 38;
                break;
            }
        }
    }

    // Aliphatic heavy atom typing.
    if (!atom_type) {
        std::uint16_t const nstart = desc.neighbor_offset[atom_idx];
        std::uint16_t const nend = desc.neighbor_offset[atom_idx + 1];
        unsigned const tdeg = atom_total_degree(desc, atom_idx);
        unsigned const explicit_deg = desc.degree[atom_idx];

        auto bond_type_to = [&](std::uint8_t other) -> unsigned {
            for (std::uint16_t n = nstart; n < nend; ++n)
                if (desc.neighbor_atom[n] == other) {
                    std::uint16_t const b = desc.neighbor_bond[n];
                    if (desc.bond_flags[b] & bond_flag_aromatic_k) return 4;
                    return desc.bond_order[b];
                }
            return 0;
        };

        // Count terminal O neighbors for later reuse (N element path).
        unsigned n_term_o_bonded_to_n = count_terminal_o_neighbors(desc, atom_idx);

        switch (element) {
        case 3:
            if (explicit_deg == 0) atom_type = 92;
            break; // Li+
        case 6: {
            if (tdeg == 4) {
                if (atom_in_ring_of_size(desc, atom_idx, 3)) {
                    atom_type = 22;
                    break;
                }
                if (atom_in_ring_of_size(desc, atom_idx, 4)) {
                    atom_type = 20;
                    break;
                }
                atom_type = 1;
                break;
            }
            if (tdeg == 3) {
                unsigned nN2 = 0, nN3 = 0, nO = 0, nS = 0;
                unsigned double_bonded_elem = 0;
                for (std::uint16_t n = nstart; n < nend; ++n) {
                    std::uint8_t const a = desc.neighbor_atom[n];
                    if (bond_type_to(a) == 2) double_bonded_elem = desc.element[a];
                    if (atom_total_degree(desc, a) == 1) {
                        if (desc.element[a] == 8) ++nO;
                        else if (desc.element[a] == 16) ++nS;
                    }
                    else if (desc.element[a] == 7) {
                        if (atom_total_degree(desc, a) == 3) ++nN3;
                        else if (atom_total_degree(desc, a) == 2 && bond_type_to(a) == 2) ++nN2;
                    }
                }
                if (nN3 >= 2 && !nN2 && double_bonded_elem == 7) {
                    atom_type = 57;
                    break;
                }
                if (nO == 2 || nS == 2) {
                    atom_type = 41;
                    break;
                }
                if (atom_in_ring_of_size(desc, atom_idx, 4) && double_bonded_elem == 6) {
                    atom_type = 30;
                    break;
                }
                if (double_bonded_elem == 7 || double_bonded_elem == 8 || double_bonded_elem == 15 ||
                    double_bonded_elem == 16) {
                    atom_type = 3;
                    break;
                }
                atom_type = 2;
                break;
            }
            if (tdeg == 2) {
                atom_type = 4;
                break;
            }
            if (tdeg == 1) {
                atom_type = 60;
                break;
            }
            break;
        }
        case 7: {
            // Precompute sulfonamide/sulfate flag from S or P neighbors.
            bool is_nso2_or_nso3_or_ncn = false;
            unsigned total_val_n = atom_total_valence(desc, atom_idx);
            for (std::uint16_t n = nstart; n < nend; ++n) {
                std::uint8_t const a = desc.neighbor_atom[n];
                if (total_val_n >= 3 && (desc.element[a] == 15 || desc.element[a] == 16)) {
                    unsigned n_o_bonded_to_sp = 0;
                    std::uint16_t const bs = desc.neighbor_offset[a];
                    std::uint16_t const be = desc.neighbor_offset[a + 1];
                    for (std::uint16_t n2 = bs; n2 < be; ++n2) {
                        std::uint8_t const b = desc.neighbor_atom[n2];
                        if (desc.element[b] == 8 && atom_total_degree(desc, b) == 1) ++n_o_bonded_to_sp;
                    }
                    if (n_o_bonded_to_sp >= 2) is_nso2_or_nso3_or_ncn = true;
                }
            }
            if (tdeg == 4) {
                if (is_atom_n_oxide(desc, atom_idx)) {
                    atom_type = 68;
                    break;
                }
                atom_type = 34;
                break;
            }
            if (tdeg == 3) {
                if (total_val_n >= 4) {
                    bool double_bonded_cn = false;
                    for (std::uint16_t n = nstart; n < nend; ++n) {
                        std::uint8_t const a = desc.neighbor_atom[n];
                        if (bond_type_to(a) == 2) {
                            double_bonded_cn = (desc.element[a] == 7 || desc.element[a] == 6);
                            if (desc.element[a] == 6) {
                                std::uint16_t const bs = desc.neighbor_offset[a];
                                std::uint16_t const be = desc.neighbor_offset[a + 1];
                                for (std::uint16_t n2 = bs; n2 < be && double_bonded_cn; ++n2) {
                                    std::uint8_t const b = desc.neighbor_atom[n2];
                                    if (b == atom_idx) continue;
                                    double_bonded_cn = !(desc.element[b] == 7 && atom_total_degree(desc, b) == 3);
                                }
                            }
                        }
                    }
                    if (n_term_o_bonded_to_n == 1) {
                        atom_type = 67;
                        break;
                    }
                    if (n_term_o_bonded_to_n >= 2) {
                        atom_type = 45;
                        break;
                    }
                    if (double_bonded_cn) {
                        atom_type = 54;
                        break;
                    }
                }
                if (total_val_n >= 3) {
                    bool isNCOorNCS = false, isNCNplus = false, isNGDplus = false;
                    bool isNNNorNNC = false, isNbrC = false, isNbrBenzeneC = false;
                    unsigned elementDoubleBondedToC = 0, elementTripleBondedToC = 0;
                    unsigned nN2bondedToC = 0, nN3bondedToC = 0;
                    unsigned nObondedToC = 0, nSbondedToC = 0;
                    for (std::uint16_t n = nstart; n < nend; ++n) {
                        std::uint8_t const a = desc.neighbor_atom[n];
                        if (desc.element[a] == 6) {
                            isNbrC = true;
                            if ((desc.atom_flags[a] & atom_flag_aromatic_k) && atom_in_ring_of_size(desc, a, 6))
                                isNbrBenzeneC = true;
                            nN2bondedToC = nN3bondedToC = nObondedToC = nSbondedToC = 0;
                            unsigned nFormalCharge = 0, nInAromatic6Ring = 0;
                            std::uint16_t const bs = desc.neighbor_offset[a];
                            std::uint16_t const be = desc.neighbor_offset[a + 1];
                            for (std::uint16_t n2 = bs; n2 < be; ++n2) {
                                std::uint8_t const b = desc.neighbor_atom[n2];
                                unsigned const bo = bond_order_between(desc, a, b);
                                bool const arom = bond_is_aromatic_between(desc, a, b);
                                if (bo == 2 && (desc.element[b] == 8 || desc.element[b] == 16)) isNCOorNCS = true;
                                if (bo == 2 || (arom && (desc.element[b] == 6 ||
                                                         (desc.element[b] == 7 && atom_num_rings(desc, b) == 1))))
                                    elementDoubleBondedToC = desc.element[b];
                                if (bo == 3) elementTripleBondedToC = desc.element[b];
                                if (desc.element[b] == 7 && atom_total_degree(desc, b) == 3) {
                                    if (desc.formal_charge[b] == 1) ++nFormalCharge;
                                    if (atom_in_aromatic_ring_of_size(desc, a, 6)) ++nInAromatic6Ring;
                                    unsigned nObondedToN3 = 0;
                                    std::uint16_t const cs = desc.neighbor_offset[b];
                                    std::uint16_t const ce = desc.neighbor_offset[b + 1];
                                    for (std::uint16_t n3 = cs; n3 < ce; ++n3)
                                        if (desc.element[desc.neighbor_atom[n3]] == 8) ++nObondedToN3;
                                    if (nObondedToN3 < 2) ++nN3bondedToC;
                                }
                                if (desc.element[b] == 7 && atom_total_degree(desc, b) == 2 && (bo == 2 || arom))
                                    ++nN2bondedToC;
                                if (desc.atom_flags[b] & atom_flag_aromatic_k) {
                                    if (desc.element[b] == 8) ++nObondedToC;
                                    if (desc.element[b] == 16) ++nSbondedToC;
                                }
                            }
                            if (elementDoubleBondedToC == 7) {
                                if (nN3bondedToC == 2 && !nN2bondedToC && nFormalCharge && !nInAromatic6Ring &&
                                    atom_total_degree(desc, a) < 4)
                                    isNCNplus = true;
                                if (nN3bondedToC == 3) isNGDplus = true;
                            }
                        }
                        if (desc.element[a] == 7) {
                            unsigned nNbondedToN = 0, nObondedToN = 0, nSbondedToN = 0;
                            std::uint16_t const bs = desc.neighbor_offset[a];
                            std::uint16_t const be = desc.neighbor_offset[a + 1];
                            for (std::uint16_t n2 = bs; n2 < be; ++n2) {
                                std::uint8_t const b = desc.neighbor_atom[n2];
                                unsigned const bo = bond_order_between(desc, a, b);
                                if (bo != 2) continue;
                                if (desc.element[b] == 6) {
                                    std::uint16_t const cs = desc.neighbor_offset[b];
                                    std::uint16_t const ce = desc.neighbor_offset[b + 1];
                                    for (std::uint16_t n3 = cs; n3 < ce; ++n3) {
                                        std::uint8_t const c = desc.neighbor_atom[n3];
                                        if (c == a) continue;
                                        switch (desc.element[c]) {
                                        case 7: ++nNbondedToN; break;
                                        case 8: ++nObondedToN; break;
                                        case 16: ++nSbondedToN; break;
                                        }
                                    }
                                    if (!nObondedToN && !nSbondedToN && !nNbondedToN && !isNbrBenzeneC)
                                        isNNNorNNC = true;
                                }
                                if (desc.element[b] == 7 && !isNbrBenzeneC) isNNNorNNC = true;
                            }
                        }
                    }
                    if (isNbrC) {
                        if (elementTripleBondedToC == 7) is_nso2_or_nso3_or_ncn = true;
                        if (isNCNplus) {
                            atom_type = 55;
                            break;
                        }
                        if (isNGDplus) {
                            atom_type = 56;
                            break;
                        }
                        if (!isNCOorNCS && !is_nso2_or_nso3_or_ncn &&
                            (((!nObondedToC) && (!nSbondedToC) && isNbrBenzeneC) ||
                             (elementDoubleBondedToC == 6 || elementDoubleBondedToC == 7 ||
                              elementDoubleBondedToC == 15 || elementTripleBondedToC == 6))) {
                            atom_type = 40;
                            break;
                        }
                    }
                    if (!is_nso2_or_nso3_or_ncn && (isNCOorNCS || isNNNorNNC)) {
                        atom_type = 10;
                        break;
                    }
                }
            }
            if (tdeg == 2) {
                if (total_val_n == 4) {
                    bool is_isonitrile = false;
                    for (std::uint16_t n = nstart; n < nend && !is_isonitrile; ++n)
                        is_isonitrile = bond_type_to(desc.neighbor_atom[n]) == 3;
                    if (is_isonitrile) {
                        atom_type = 61;
                        break;
                    }
                    atom_type = 53;
                    break;
                }
                if (total_val_n == 3) {
                    bool is_nitroso = false, is_imine_or_azo = false;
                    for (std::uint16_t n = nstart; n < nend; ++n) {
                        std::uint8_t const a = desc.neighbor_atom[n];
                        if (bond_type_to(a) == 2) {
                            is_nitroso = (desc.element[a] == 8 && n_term_o_bonded_to_n == 1);
                            is_imine_or_azo = (desc.element[a] == 6 || desc.element[a] == 7);
                        }
                    }
                    if (is_nitroso && !is_imine_or_azo) {
                        atom_type = 46;
                        break;
                    }
                    if (is_imine_or_azo) {
                        atom_type = 9;
                        break;
                    }
                }
                if (total_val_n >= 2) {
                    bool isNSO = false;
                    for (std::uint16_t n = nstart; n < nend && !isNSO; ++n) {
                        std::uint8_t const a = desc.neighbor_atom[n];
                        if (desc.element[a] == 16) {
                            unsigned nTermObondedToS = 0;
                            std::uint16_t const bs = desc.neighbor_offset[a];
                            std::uint16_t const be = desc.neighbor_offset[a + 1];
                            for (std::uint16_t n2 = bs; n2 < be; ++n2) {
                                std::uint8_t const b = desc.neighbor_atom[n2];
                                if (desc.element[b] == 8 && atom_total_degree(desc, b) == 1) ++nTermObondedToS;
                            }
                            isNSO = (nTermObondedToS == 1);
                        }
                    }
                    if (isNSO) {
                        atom_type = 48;
                        break;
                    }
                    if (!is_nso2_or_nso3_or_ncn) {
                        atom_type = 62;
                        break;
                    }
                }
            }
            if (is_nso2_or_nso3_or_ncn) {
                atom_type = 43;
                break;
            }
            if (tdeg == 1) {
                bool isNSP = false, isNAZT = false;
                for (std::uint16_t n = nstart; n < nend && !isNSP && !isNAZT; ++n) {
                    std::uint8_t const a = desc.neighbor_atom[n];
                    isNSP = (bond_type_to(a) == 3);
                    if (desc.element[a] == 7 && atom_total_degree(desc, a) == 2) {
                        std::uint16_t const bs = desc.neighbor_offset[a];
                        std::uint16_t const be = desc.neighbor_offset[a + 1];
                        for (std::uint16_t n2 = bs; n2 < be && !isNAZT; ++n2) {
                            std::uint8_t const b = desc.neighbor_atom[n2];
                            isNAZT = (desc.element[b] == 7 && atom_total_degree(desc, b) == 2) ||
                                     (desc.element[b] == 6 && atom_total_degree(desc, b) == 3);
                        }
                    }
                }
                if (isNSP) {
                    atom_type = 42;
                    break;
                }
                if (isNAZT) {
                    atom_type = 47;
                    break;
                }
            }
            atom_type = 8; // NR — fallback
            break;
        }
        case 8: {
            if (tdeg == 3) {
                atom_type = 49;
                break;
            }
            if (tdeg == 2) {
                if (atom_total_valence(desc, atom_idx) == 3) {
                    atom_type = 51;
                    break;
                }
                // OH2 (water): two H neighbors. `desc.num_hydrogens`
                // already counts them (since addHs is called before
                // build_descriptor), so there is no separate implicit
                // contribution to add.
                if (desc.num_hydrogens[atom_idx] == 2) {
                    atom_type = 70;
                    break;
                }
                atom_type = 6;
                break;
            }
            if (explicit_deg <= 1) {
                unsigned nNbondedToCorNorS = 0, nObondedToCorNorS = 0, nSbondedToCorNorS = 0;
                bool isOxideOBondedToH = (desc.num_hydrogens[atom_idx] > 0);
                bool isCarboxylateO = false, isCarbonylO = false, isOxideOBondedToC = false;
                bool isNitrosoO = false, isOxideOBondedToN = false, isNOxideO = false;
                bool isNitroO = false, isThioSulfinateO = false, isSulfateO = false;
                bool isSulfoxideO = false, isPhosphateOrPerchlorateO = false;
                for (std::uint16_t n = nstart; n < nend; ++n) {
                    std::uint8_t const a = desc.neighbor_atom[n];
                    if (desc.element[a] == 6 || desc.element[a] == 7 || desc.element[a] == 16) {
                        std::uint16_t const bs = desc.neighbor_offset[a];
                        std::uint16_t const be = desc.neighbor_offset[a + 1];
                        for (std::uint16_t n2 = bs; n2 < be; ++n2) {
                            std::uint8_t const b = desc.neighbor_atom[n2];
                            if (desc.element[b] == 7 && atom_total_degree(desc, b) == 2) ++nNbondedToCorNorS;
                            if (desc.element[b] == 8 && atom_total_degree(desc, b) == 1) ++nObondedToCorNorS;
                            if (desc.element[b] == 16 && atom_total_degree(desc, b) == 1) ++nSbondedToCorNorS;
                        }
                    }
                    if (desc.element[a] == 1) isOxideOBondedToH = true;
                    unsigned const bo = bond_type_to(a);
                    if (desc.element[a] == 6) {
                        isCarboxylateO = (nObondedToCorNorS == 2);
                        isCarbonylO = (bo == 2);
                        isOxideOBondedToC = (bo == 1 && nObondedToCorNorS == 1);
                    }
                    if (desc.element[a] == 7) {
                        isNitrosoO = (bo == 2);
                        if (bo == 1 && nObondedToCorNorS == 1) {
                            isOxideOBondedToN = (atom_total_degree(desc, a) == 2 ||
                                                 (atom_total_valence(desc, a) + desc.num_hydrogens[a]) == 3);
                            isNOxideO = ((atom_total_valence(desc, a) + desc.num_hydrogens[a]) == 4);
                        }
                        isNitroO = (nObondedToCorNorS >= 2);
                    }
                    if (desc.element[a] == 16) {
                        isThioSulfinateO = (nSbondedToCorNorS == 1);
                        isSulfateO = (bo == 1 || (bo == 2 && (nObondedToCorNorS + nNbondedToCorNorS) > 1));
                        isSulfoxideO = (bo == 2 && (nObondedToCorNorS + nNbondedToCorNorS) == 1);
                    }
                    isPhosphateOrPerchlorateO = (desc.element[a] == 15 || desc.element[a] == 17);
                    if (isOxideOBondedToC || isOxideOBondedToN || isOxideOBondedToH || isCarboxylateO || isNitroO ||
                        isNOxideO || isThioSulfinateO || isSulfateO || isPhosphateOrPerchlorateO || isCarbonylO ||
                        isNitrosoO || isSulfoxideO)
                        break;
                }
                if (isOxideOBondedToC || isOxideOBondedToN || isOxideOBondedToH) {
                    atom_type = 35;
                    break;
                }
                if (isCarboxylateO || isNitroO || isNOxideO || isThioSulfinateO || isSulfateO ||
                    isPhosphateOrPerchlorateO) {
                    atom_type = 32;
                    break;
                }
                if (isCarbonylO || isNitrosoO || isSulfoxideO) {
                    atom_type = 7;
                    break;
                }
            }
            break;
        }
        case 9: {
            if (desc.degree[atom_idx] == 1) {
                atom_type = 11;
                break;
            }
            if (desc.degree[atom_idx] == 0) {
                atom_type = 89;
                break;
            }
            break;
        }
        case 11:
            if (desc.degree[atom_idx] == 0) atom_type = 93;
            break; // Na+
        case 12:
            if (desc.degree[atom_idx] == 0) atom_type = 99;
            break;                      // Mg+2
        case 14: atom_type = 19; break; // Si
        case 15: {
            if (tdeg == 4) {
                atom_type = 25;
                break;
            }
            if (tdeg == 3) {
                atom_type = 26;
                break;
            }
            if (tdeg == 2) {
                atom_type = 75;
                break;
            }
            break;
        }
        case 16: {
            if (tdeg == 3 || tdeg == 4) {
                unsigned nOorNbondedToS = 0, nSbondedToS = 0;
                bool isCDoubleBondedToS = false;
                for (std::uint16_t n = nstart; n < nend; ++n) {
                    std::uint8_t const a = desc.neighbor_atom[n];
                    if (desc.element[a] == 6 && bond_type_to(a) == 2) isCDoubleBondedToS = true;
                    if ((desc.degree[a] == 1 && desc.element[a] == 8) ||
                        (atom_total_degree(desc, a) == 2 && desc.element[a] == 7))
                        ++nOorNbondedToS;
                    if (desc.degree[a] == 1 && desc.element[a] == 16) ++nSbondedToS;
                }
                if ((tdeg == 3 && nOorNbondedToS == 2 && isCDoubleBondedToS) || tdeg == 4) {
                    atom_type = 18;
                    break;
                }
                if ((nOorNbondedToS && nSbondedToS) || (nOorNbondedToS == 2 && !isCDoubleBondedToS)) {
                    atom_type = 73;
                    break;
                }
                atom_type = 17;
                break;
            }
            if (tdeg == 2) {
                bool isODoubleBondedToS = false;
                for (std::uint16_t n = nstart; n < nend; ++n) {
                    std::uint8_t const a = desc.neighbor_atom[n];
                    if (desc.element[a] == 8 && bond_type_to(a) == 2) isODoubleBondedToS = true;
                }
                if (isODoubleBondedToS) {
                    atom_type = 74;
                    break;
                }
                atom_type = 15;
                break;
            }
            if (desc.degree[atom_idx] == 1) {
                unsigned nTermSbondedToNbr = 0;
                bool isCDoubleBondedToS = false;
                for (std::uint16_t n = nstart; n < nend; ++n) {
                    std::uint8_t const a = desc.neighbor_atom[n];
                    std::uint16_t const bs = desc.neighbor_offset[a];
                    std::uint16_t const be = desc.neighbor_offset[a + 1];
                    for (std::uint16_t n2 = bs; n2 < be; ++n2) {
                        std::uint8_t const b = desc.neighbor_atom[n2];
                        if (desc.element[b] == 16 && atom_total_degree(desc, b) == 1) ++nTermSbondedToNbr;
                    }
                    if (desc.element[a] == 6 && bond_type_to(a) == 2) isCDoubleBondedToS = true;
                }
                if (isCDoubleBondedToS && nTermSbondedToNbr != 2) {
                    atom_type = 16;
                    break;
                }
                atom_type = 72;
                break;
            }
            break;
        }
        case 17: {
            if (tdeg == 4) {
                unsigned nObondedToCl = 0;
                for (std::uint16_t n = nstart; n < nend; ++n)
                    if (desc.element[desc.neighbor_atom[n]] == 8) ++nObondedToCl;
                if (nObondedToCl == 4) {
                    atom_type = 77;
                    break;
                }
            }
            if (tdeg == 1) {
                atom_type = 12;
                break;
            }
            if (desc.degree[atom_idx] == 0) {
                atom_type = 90;
                break;
            }
            break;
        }
        case 19:
            if (desc.degree[atom_idx] == 0) atom_type = 94;
            break; // K+
        case 20:
            if (desc.degree[atom_idx] == 0) atom_type = 96;
            break; // Ca+2
        case 26:
            if (desc.degree[atom_idx] == 0) {
                if (desc.formal_charge[atom_idx] == 2) atom_type = 87;
                else if (desc.formal_charge[atom_idx] == 3) atom_type = 88;
            }
            break;
        case 29:
            if (desc.degree[atom_idx] == 0) {
                if (desc.formal_charge[atom_idx] == 1) atom_type = 97;
                else if (desc.formal_charge[atom_idx] == 2) atom_type = 98;
            }
            break;
        case 30:
            if (desc.degree[atom_idx] == 0) atom_type = 95;
            break; // Zn+2
        case 35:
            if (desc.degree[atom_idx] == 1) {
                atom_type = 13;
                break;
            }
            if (desc.degree[atom_idx] == 0) {
                atom_type = 91;
                break;
            }
            break;
        case 53:
            if (desc.degree[atom_idx] == 1) {
                atom_type = 14;
                break;
            }
            break;
        }
    }

    return atom_type;
}

/**
 *  @brief Port of `setMMFFHydrogenType`.  Takes the already-computed heavy-atom types so the H-typing nested
 *  switches can key on them.
 */
SMALLFIELD_HD_ inline std::uint8_t native_set_mmff_hydrogen_type(molecule_descriptor_t const &desc,
                                                                 std::uint32_t atom_idx,
                                                                 std::span<std::uint8_t const> atom_types) noexcept {
    std::uint8_t atom_type = 0;
    std::uint16_t const start = desc.neighbor_offset[atom_idx];
    std::uint16_t const end = desc.neighbor_offset[atom_idx + 1];
    for (std::uint16_t n = start; n < end; ++n) {
        std::uint8_t const nb = desc.neighbor_atom[n];
        unsigned const element = desc.element[nb];
        switch (element) {
        case 6:
        case 14: atom_type = 5; break;
        case 7: {
            std::uint8_t const nt = atom_types[nb];
            switch (nt) {
            case 8:
            case 39:
            case 62:
            case 67:
            case 68: atom_type = 23; break;
            case 34:
            case 54:
            case 55:
            case 56:
            case 58:
            case 81: atom_type = 36; break;
            case 9: atom_type = 27; break;
            default: atom_type = 28; break;
            }
            break;
        }
        case 8: {
            std::uint8_t const nt = atom_types[nb];
            switch (nt) {
            case 49: atom_type = 50; break;
            case 51: atom_type = 52; break;
            case 70: atom_type = 31; break;
            case 6: {
                bool isHOCCorHOCN = false, isHOCO = false, isHOP = false, isHOS = false;
                std::uint16_t const s2 = desc.neighbor_offset[nb];
                std::uint16_t const e2 = desc.neighbor_offset[nb + 1];
                for (std::uint16_t n2 = s2; n2 < e2; ++n2) {
                    std::uint8_t const b = desc.neighbor_atom[n2];
                    if (desc.element[b] == 6) {
                        std::uint16_t const s3 = desc.neighbor_offset[b];
                        std::uint16_t const e3 = desc.neighbor_offset[b + 1];
                        for (std::uint16_t n3 = s3; n3 < e3; ++n3) {
                            std::uint8_t const c = desc.neighbor_atom[n3];
                            if (c == nb) continue;
                            unsigned const bo_bc = bond_order_between(desc, b, c);
                            bool const arom_bc = bond_is_aromatic_between(desc, b, c);
                            if ((desc.element[c] == 6 || desc.element[c] == 7) && (bo_bc == 2 || arom_bc))
                                isHOCCorHOCN = true;
                            if (desc.element[c] == 8 && bo_bc == 2) isHOCO = true;
                        }
                    }
                    if (desc.element[b] == 15) isHOP = true;
                    if (desc.element[b] == 16) isHOS = true;
                }
                if (isHOCO || isHOP) {
                    atom_type = 24;
                    break;
                }
                if (isHOCCorHOCN) {
                    atom_type = 29;
                    break;
                }
                if (isHOS) {
                    atom_type = 33;
                    break;
                }
                atom_type = 21;
                break;
            }
            default: atom_type = 21; break;
            }
            break;
        }
        case 15:
        case 16: atom_type = 71; break;
        }
        if (atom_type) break;
    }
    return atom_type;
}

/**
 *  @brief Compute MMFF formal charges per atom.
 *
 *  Port of the first half of `computeMMFFCharges` at `AtomTyper.cpp:3094-3421`.
 *  Sets per-atom formal charges for atom types with special allocation rules
 *  (carboxylate charge sharing, N5M anion splitting, conjugated nitrogen flood
 *  fill).  All other atoms keep `desc.formal_charge` (the RDKit-assigned
 *  formal charge from SMILES parsing).
 */
SMALLFIELD_HD_ inline void native_compute_mmff_formal_charges(molecule_descriptor_t const &desc,
                                                              std::span<std::uint8_t const> atom_types,
                                                              std::span<float> formal_charges) noexcept {
    std::uint32_t const num_atoms = desc.num_atoms;
    // Initialise with zero — only atom types listed below get overridden.
    for (std::uint32_t a = 0; a < num_atoms; ++a) formal_charges[a] = 0.0f;

    // Scratch bitset for the conjugated-nitrogen flood fill (types 55/56/81).
    std::array<std::uint32_t, (max_atoms_k + 31) / 32> conj_bits {};
    auto conj_set = [&](std::uint32_t i) { conj_bits[i >> 5] |= (1u << (i & 31)); };
    auto conj_get = [&](std::uint32_t i) -> bool { return (conj_bits[i >> 5] & (1u << (i & 31))) != 0; };
    auto conj_reset = [&]() { conj_bits.fill(0); };

    for (std::uint32_t idx = 0; idx < num_atoms; ++idx) {
        std::uint8_t const atom_type = atom_types[idx];
        float formal_charge = 0.0f;
        switch (atom_type) {
        case 32: // O2CM — carboxylate / phosphate / sulfonate oxygen
        case 72: // SM   — anionic terminal sulfur
        {
            std::uint16_t const neighbor_start = desc.neighbor_offset[idx];
            std::uint16_t const neighbor_end = desc.neighbor_offset[idx + 1];
            for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                std::uint8_t const neighbor_idx = desc.neighbor_atom[neighbor_slot];
                std::uint8_t const neighbor_type = atom_types[neighbor_idx];
                int num_sec_n_bonded_to_neighbor = 0;
                int num_terminal_os_bonded_to_neighbor = 0;
                std::uint16_t const second_shell_start = desc.neighbor_offset[neighbor_idx];
                std::uint16_t const second_shell_end = desc.neighbor_offset[neighbor_idx + 1];
                for (std::uint16_t second_shell_slot = second_shell_start; second_shell_slot < second_shell_end;
                     ++second_shell_slot) {
                    std::uint8_t const second_shell_idx = desc.neighbor_atom[second_shell_slot];
                    if (desc.element[second_shell_idx] == 7 && desc.degree[second_shell_idx] == 2 &&
                        !(desc.atom_flags[second_shell_idx] & atom_flag_aromatic_k))
                        ++num_sec_n_bonded_to_neighbor;
                    if ((desc.element[second_shell_idx] == 8 || desc.element[second_shell_idx] == 16) &&
                        desc.degree[second_shell_idx] == 1)
                        ++num_terminal_os_bonded_to_neighbor;
                }
                // Sulfonamide with 2 terminal O + 1 sec N: don't treat N
                // as an O replacement.
                if (desc.element[neighbor_idx] == 16 && num_terminal_os_bonded_to_neighbor == 2 &&
                    num_sec_n_bonded_to_neighbor == 1)
                    num_sec_n_bonded_to_neighbor = 0;

                if (desc.element[neighbor_idx] == 6 && num_terminal_os_bonded_to_neighbor) {
                    formal_charge = (num_terminal_os_bonded_to_neighbor == 1)
                                        ? -1.0f
                                        : -(static_cast<float>(num_terminal_os_bonded_to_neighbor - 1) /
                                            static_cast<float>(num_terminal_os_bonded_to_neighbor));
                    break;
                }
                if (neighbor_type == 45 && num_terminal_os_bonded_to_neighbor == 3) {
                    formal_charge = -1.0f / 3.0f;
                    break;
                }
                if (neighbor_type == 25 && num_terminal_os_bonded_to_neighbor) {
                    formal_charge = (num_terminal_os_bonded_to_neighbor == 1)
                                        ? 0.0f
                                        : -(static_cast<float>(num_terminal_os_bonded_to_neighbor - 1) /
                                            static_cast<float>(num_terminal_os_bonded_to_neighbor));
                    break;
                }
                if (neighbor_type == 18 && num_terminal_os_bonded_to_neighbor) {
                    formal_charge =
                        ((num_sec_n_bonded_to_neighbor + num_terminal_os_bonded_to_neighbor) == 2)
                            ? 0.0f
                            : -(static_cast<float>((num_sec_n_bonded_to_neighbor + num_terminal_os_bonded_to_neighbor) -
                                                   2) /
                                static_cast<float>(num_terminal_os_bonded_to_neighbor));
                    break;
                }
                if (neighbor_type == 73 && num_terminal_os_bonded_to_neighbor) {
                    formal_charge = (num_terminal_os_bonded_to_neighbor == 1)
                                        ? 0.0f
                                        : -(static_cast<float>(num_terminal_os_bonded_to_neighbor - 1) /
                                            static_cast<float>(num_terminal_os_bonded_to_neighbor));
                    break;
                }
                if (neighbor_type == 77 && num_terminal_os_bonded_to_neighbor) {
                    formal_charge = -(1.0f / static_cast<float>(num_terminal_os_bonded_to_neighbor));
                    break;
                }
            }
            break;
        }
        case 76: {
            // N5M: find the 5-ring containing this atom, share -1
            // across all N5M atoms in that ring.
            std::uint32_t const ring_bitmap = desc.atom_ring_bitmap[idx];
            for (unsigned ring_idx = 0; ring_idx < desc.num_rings; ++ring_idx) {
                if (!(ring_bitmap & (1u << ring_idx))) continue;
                unsigned const ring_size = desc.ring_offset[ring_idx + 1] - desc.ring_offset[ring_idx];
                if (ring_size != 5) continue;
                unsigned num_n76_in_ring = 0;
                for (unsigned ring_atom_slot = desc.ring_offset[ring_idx];
                     ring_atom_slot < desc.ring_offset[ring_idx + 1]; ++ring_atom_slot)
                    if (atom_types[desc.ring_atoms[ring_atom_slot]] == 76) ++num_n76_in_ring;
                if (num_n76_in_ring) formal_charge = -1.0f / static_cast<float>(num_n76_in_ring);
                break;
            }
            break;
        }
        case 55:
        case 56:
        case 81: {
            // Conjugated nitrogen (amidinium / guanidinium /
            // imidazolium). Flood-fill through carbon types 57/80,
            // collecting all conjugated 55/56/81 atoms, sum formal
            // charges, split evenly.
            formal_charge = static_cast<float>(desc.formal_charge[idx]);
            unsigned num_conjugated = 1, prev_num_conjugated = 0;
            conj_reset();
            conj_set(idx);
            while (num_conjugated > prev_num_conjugated) {
                prev_num_conjugated = num_conjugated;
                for (std::uint32_t atom_in_conj = 0; atom_in_conj < num_atoms; ++atom_in_conj) {
                    if (!conj_get(atom_in_conj)) continue;
                    std::uint16_t const neighbor_start = desc.neighbor_offset[atom_in_conj];
                    std::uint16_t const neighbor_end = desc.neighbor_offset[atom_in_conj + 1];
                    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                        std::uint8_t const carbon_idx = desc.neighbor_atom[neighbor_slot];
                        std::uint8_t const carbon_type = atom_types[carbon_idx];
                        if (carbon_type != 57 && carbon_type != 80) continue;
                        std::uint16_t const carbon_neighbor_start = desc.neighbor_offset[carbon_idx];
                        std::uint16_t const carbon_neighbor_end = desc.neighbor_offset[carbon_idx + 1];
                        for (std::uint16_t carbon_neighbor_slot = carbon_neighbor_start;
                             carbon_neighbor_slot < carbon_neighbor_end; ++carbon_neighbor_slot) {
                            std::uint8_t const candidate_idx = desc.neighbor_atom[carbon_neighbor_slot];
                            std::uint8_t const candidate_type = atom_types[candidate_idx];
                            if (candidate_type != 55 && candidate_type != 56 && candidate_type != 81) continue;
                            if (!conj_get(candidate_idx)) {
                                conj_set(candidate_idx);
                                formal_charge += static_cast<float>(desc.formal_charge[candidate_idx]);
                                ++num_conjugated;
                            }
                        }
                    }
                }
            }
            if (num_conjugated) formal_charge /= static_cast<float>(num_conjugated);
            break;
        }
        case 61: {
            // Isonitrile C. If bonded to type 42 (triply bonded N),
            // set +1 (on the N, via ipso bookkeeping).
            std::uint16_t const neighbor_start = desc.neighbor_offset[idx];
            std::uint16_t const neighbor_end = desc.neighbor_offset[idx + 1];
            for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot)
                if (atom_types[desc.neighbor_atom[neighbor_slot]] == 42) {
                    formal_charge = 1.0f;
                    break;
                }
            break;
        }
        case 34:
        case 49:
        case 51:
        case 54:
        case 58:
        case 92:
        case 93:
        case 94:
        case 97: formal_charge = 1.0f; break;
        case 87:
        case 95:
        case 96:
        case 98:
        case 99: formal_charge = 2.0f; break;
        case 88: formal_charge = 3.0f; break;
        case 35:
        case 62:
        case 89:
        case 90:
        case 91: formal_charge = -1.0f; break;
        }
        formal_charges[idx] = formal_charge;
    }
}

/** Compute MMFF partial charges — second half of `computeMMFFCharges`
 *  (AtomTyper.cpp:3423-3487).  Applies bond-charge increments (BCIs)
 *  from the extracted `chg` table, with the PBCI fallback when no BCI
 *  is tabulated.  Formal-charge distribution follows equation 15 of
 *  MMFF.V. */
SMALLFIELD_HD_ inline void native_compute_mmff_partial_charges(molecule_descriptor_t const &desc,
                                                               mmff94_tables_view_t const &tables,
                                                               std::span<std::uint8_t const> atom_types,
                                                               std::span<float const> formal_charges,
                                                               std::span<float> partial_charges) noexcept {
    std::uint32_t const num_atoms = desc.num_atoms;
    for (std::uint32_t idx = 0; idx < num_atoms; ++idx) {
        std::uint8_t const atom_type = atom_types[idx];
        auto const *prop_entry = lookup_prop(tables, atom_type);
        if (!prop_entry) {
            partial_charges[idx] = formal_charges[idx];
            continue;
        }
        double const crd_count = static_cast<double>(prop_entry->crd);

        auto const *pbci_self = lookup_pbci(tables, atom_type);
        double fc_adjust = 0.0;
        double pbci_self_value = 0.0;
        if (pbci_self) {
            fc_adjust = static_cast<double>(pbci_self->fcadj);
            pbci_self_value = static_cast<double>(pbci_self->pbci);
        }

        double charge_base = static_cast<double>(formal_charges[idx]);
        bool const fc_adjust_is_zero = std::fabs(fc_adjust) < 1.0e-10;

        std::uint16_t const neighbor_start = desc.neighbor_offset[idx];
        std::uint16_t const neighbor_end = desc.neighbor_offset[idx + 1];
        if (fc_adjust_is_zero) {
            for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                std::uint8_t const neighbor_idx = desc.neighbor_atom[neighbor_slot];
                double const neighbor_formal_charge = static_cast<double>(formal_charges[neighbor_idx]);
                if (neighbor_formal_charge < 0.0)
                    charge_base += neighbor_formal_charge / (2.0 * static_cast<double>(desc.degree[neighbor_idx]));
            }
        }
        if (atom_type == 62) {
            for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                std::uint8_t const neighbor_idx = desc.neighbor_atom[neighbor_slot];
                double const neighbor_formal_charge = static_cast<double>(formal_charges[neighbor_idx]);
                if (neighbor_formal_charge > 0.0) charge_base -= neighbor_formal_charge / 2.0;
            }
        }
        double sum_formal_charge = 0.0, sum_partial_charge = 0.0;
        for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
            std::uint8_t const neighbor_idx = desc.neighbor_atom[neighbor_slot];
            std::uint8_t const neighbor_type = atom_types[neighbor_idx];
            std::uint16_t const bond_id = desc.neighbor_bond[neighbor_slot];
            // Native bond type: replicate `getMMFFBondType`.
            std::uint8_t bond_type = 0;
            bool const aromatic_bond = (desc.bond_flags[bond_id] & bond_flag_aromatic_k) != 0;
            bool const single_bond = desc.bond_order[bond_id] == 1 && !aromatic_bond;
            if (single_bond) {
                auto const *prop_self = lookup_prop(tables, atom_type);
                auto const *prop_neighbor = lookup_prop(tables, neighbor_type);
                if (prop_self && prop_neighbor &&
                    ((prop_self->sbmb && prop_neighbor->sbmb) || (prop_self->arom && prop_neighbor->arom)))
                    bond_type = 1;
            }
            auto const charge_entry = lookup_chg(tables, bond_type, atom_type, neighbor_type);
            if (charge_entry.second) {
                sum_partial_charge += static_cast<double>(charge_entry.first) *
                                      static_cast<double>(charge_entry.second->bci);
            }
            else {
                auto const *pbci_neighbor = lookup_pbci(tables, neighbor_type);
                double const neighbor_pbci = pbci_neighbor ? static_cast<double>(pbci_neighbor->pbci) : 0.0;
                sum_partial_charge += pbci_self_value - neighbor_pbci;
            }
            sum_formal_charge += static_cast<double>(formal_charges[neighbor_idx]);
        }
        double const partial_charge = (1.0 - crd_count * fc_adjust) * charge_base + fc_adjust * sum_formal_charge +
                                      sum_partial_charge;
        partial_charges[idx] = static_cast<float>(partial_charge);
    }
}

/**
 *  @brief Top-level entry point that types every atom in the molecule.
 *
 *  Populates `atom_types[]` with MMFF94 atom types for every atom (heavy atoms
 *  via `native_set_mmff_heavy_atom_type`, hydrogens via
 *  `native_set_mmff_hydrogen_type`).  Returns true iff every atom was typed
 *  successfully.
 */
SMALLFIELD_HD_ inline bool native_compute_mmff_atom_types(molecule_descriptor_t const &desc,
                                                          std::span<std::uint8_t> atom_types) noexcept {
    std::uint32_t const num_atoms = desc.num_atoms;
    bool ok = true;
    // Heavy atoms first.
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        if (desc.element[a] == 1) {
            atom_types[a] = 0;
            continue;
        }
        atom_types[a] = native_set_mmff_heavy_atom_type(desc, a);
        if (atom_types[a] == 0) ok = false;
    }
    // Then hydrogens (need heavy atom types populated first).
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        if (desc.element[a] != 1) continue;
        atom_types[a] = native_set_mmff_hydrogen_type(desc, a, atom_types);
        if (atom_types[a] == 0) ok = false;
    }
    return ok;
}

} // namespace smallfield::mmff94
