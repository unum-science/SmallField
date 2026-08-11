/**
 *  @file src/smallfield/mmff94/mmff94_typer.hpp
 *  @brief Native C++ port of RDKit's MMFF atom typer.
 *
 *  Reads `molecule_descriptor_t` and the MMFFProp table and produces the RDKit-equivalent MMFF atom type, 1
 *  through 99, for each atom. This eliminates the one remaining runtime RDKit dependency in the CPU prep path:
 *  `MMFFMolProperties::MMFFMolProperties()` and its per-atom `setMMFFHeavyAtomType` / `setMMFFHydrogenType`
 *  decision trees.
 *
 *  Ports:
 *   - `setMMFFHeavyAtomType` — atom types 1-99 for heavy atoms
 *   - `setMMFFHydrogenType`  — hydrogen atom types (5, 21-33, 36, 50,
 *                              52, 71)
 *   - helpers: `is_atom_n_oxide`, `is_atom_in_aromatic_ring_of_size`,
 *              `atoms_in_same_ring_of_size`, ring-info queries
 *
 *  Implementation faithfully mirrors `rdkit/.../AtomTyper.cpp` lines 504-2302. Structural decisions such as case
 *  branches and break sequences are preserved so future merges from RDKit's typer are mechanical; local variable
 *  names are not — they follow this project's snake_case convention and spell out the chemistry instead of
 *  RDKit's abbreviations, so a merge still needs the renames re-applied by hand.
 *
 *  Aromaticity input consumes the descriptor's aromatic flags, which are set by RDKit's default aromaticity
 *  perception during `SmilesToMol(...sanitize=true...)`. That is @b not always the perception RDKit's own
 *  typer sees. `MMFFMolProperties` guards only its Kekulize behind `_MMFFSanitized`; it calls
 *  `MolOps::setMMFFAromaticity` unconditionally, so presetting that property suppresses the Kekulize and
 *  nothing else. MMFF perceives aromaticity with its own pi-electron count, and it only ever adds.
 *
 *  So a caller that snapshots the descriptor before constructing `MMFFMolProperties` types from RDKit's
 *  default perception while RDKit types from MMFF's, and the two disagree on roughly 400 molecules per
 *  million — N-oxides and fused heterocycles, where the extra perception changes an atom type outright.
 *  Applying MMFF's perception before the snapshot removes the difference, since a second application is a
 *  no-op.
 */

#pragma once

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <array> // `std::array`
#include <span>  // `std::span`

#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::mmff94 {
using smallfield::parameters::lookup_prop;
using smallfield::parameters::mmff94_tables_view_t;
using smallfield::types::atom_flag_aromatic_k;
using smallfield::types::bond_flag_order_aromatic_k;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::molecule_descriptor_t;

/**
 *  @defgroup ring_queries Ring-info queries on the descriptor
 *  Helpers that read `descriptor.atom_ring_bitmap` + `descriptor.ring_offset` /
 *  `descriptor.ring_atoms` to answer the "is atom X in a ring of size Y" /
 *  "do atoms X and Y share a ring" questions the MMFF typer needs.
 *  @{
 */

/** Number of SSSR rings that contain atom. */
constexpr unsigned atom_num_rings(molecule_descriptor_t const &descriptor, std::uint32_t atom_index) noexcept {
    std::uint32_t const bitmap = descriptor.atom_ring_bitmap[atom_index];
    unsigned ring_count = 0;
    for (unsigned ring_index = 0; ring_index < descriptor.num_rings; ++ring_index)
        if (bitmap & (1u << ring_index)) ++ring_count;
    return ring_count;
}

/** True iff atom is in at least one SSSR ring of exactly `size` atoms. */
constexpr bool atom_in_ring_of_size(molecule_descriptor_t const &descriptor, std::uint32_t atom_index,
                                    unsigned size) noexcept {
    std::uint32_t const bitmap = descriptor.atom_ring_bitmap[atom_index];
    for (unsigned ring_index = 0; ring_index < descriptor.num_rings; ++ring_index) {
        if (!(bitmap & (1u << ring_index))) continue;
        unsigned const ring_size = descriptor.ring_offset[ring_index + 1] - descriptor.ring_offset[ring_index];
        if (ring_size == size) return true;
    }
    return false;
}

/** The bond order of the bond between two atoms (0 if not adjacent). An aromatic bond reports 1; ask
 *  `bond_order_is_aromatic_between` to tell the two apart. */
constexpr std::uint8_t bond_order_between(molecule_descriptor_t const &descriptor, std::uint32_t atom_i,
                                          std::uint32_t atom_j) noexcept {
    std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_i];
    std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_i + 1];
    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot)
        if (descriptor.neighbor_atom[neighbor_slot] == atom_j)
            return descriptor.bond_order[descriptor.neighbor_bond[neighbor_slot]];
    return 0;
}

/** True iff the bond between two atoms is typed `Bond::AROMATIC`, which is the only bond-aromaticity test
 *  RDKit's typer ever makes — never `Bond::getIsAromatic()`, which an in-ring triple bond also carries.
 *  Returns false when the atoms are not adjacent. */
constexpr bool bond_order_is_aromatic_between(molecule_descriptor_t const &descriptor, std::uint32_t atom_i,
                                              std::uint32_t atom_j) noexcept {
    std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_i];
    std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_i + 1];
    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot)
        if (descriptor.neighbor_atom[neighbor_slot] == atom_j)
            return (descriptor.bond_flags[descriptor.neighbor_bond[neighbor_slot]] & bond_flag_order_aromatic_k) != 0;
    return false;
}

/**
 *  @brief True iff atom sits in a size-`size` SSSR ring whose bonds are all typed `Bond::AROMATIC`.
 *
 *  Matches `RDKit::MMFF::isRingAromatic`, which asks the bonds rather than the atoms: an in-ring triple
 *  bond and a quinoid single bond both carry aromatic *atoms* at both ends yet leave the ring non-aromatic
 *  for MMFF. Consecutive slots of `ring_atoms` are bonded because `RingInfo::atomRings()` stores each ring
 *  in cyclic walk order, and the ring-closure bond is deliberately left untested — RDKit's loop stops one
 *  short of the end, so a ring whose only non-aromatic bond is the closure still counts as aromatic.
 */
constexpr bool atom_in_aromatic_ring_of_size(molecule_descriptor_t const &descriptor, std::uint32_t atom_index,
                                             unsigned size) noexcept {
    std::uint32_t const bitmap = descriptor.atom_ring_bitmap[atom_index];
    for (unsigned ring_index = 0; ring_index < descriptor.num_rings; ++ring_index) {
        if (!(bitmap & (1u << ring_index))) continue;
        unsigned const first_slot = descriptor.ring_offset[ring_index];
        unsigned const last_slot = descriptor.ring_offset[ring_index + 1];
        if (last_slot - first_slot != size) continue;
        bool all_aromatic = true;
        for (unsigned ring_atom_slot = first_slot; ring_atom_slot + 1 < last_slot; ++ring_atom_slot) {
            std::uint8_t const ring_atom = descriptor.ring_atoms[ring_atom_slot];
            std::uint8_t const next_ring_atom = descriptor.ring_atoms[ring_atom_slot + 1];
            if (!bond_order_is_aromatic_between(descriptor, ring_atom, next_ring_atom)) {
                all_aromatic = false;
                break;
            }
        }
        if (all_aromatic) return true;
    }
    return false;
}

/** True iff both atoms appear in the same SSSR ring of exactly `size`. */
constexpr bool atoms_in_same_ring_of_size(molecule_descriptor_t const &descriptor, std::uint32_t first_atom,
                                          std::uint32_t second_atom, unsigned size) noexcept {
    std::uint32_t const bits = descriptor.atom_ring_bitmap[first_atom] & descriptor.atom_ring_bitmap[second_atom];
    for (unsigned ring_index = 0; ring_index < descriptor.num_rings; ++ring_index) {
        if (!(bits & (1u << ring_index))) continue;
        unsigned const ring_size = descriptor.ring_offset[ring_index + 1] - descriptor.ring_offset[ring_index];
        if (ring_size == size) return true;
    }
    return false;
}

/** Total valence, RDKit's `atom->getValence(ExplicitType) + atom->getNumImplicitHs()`. Read straight from the
 *  descriptor: summing `bond_order` cannot reproduce it, since RDKit weighs an aromatic bond 1.5 and then
 *  clamps an aromatic atom's total down to the element's nearest permitted valence. */
constexpr unsigned atom_total_valence(molecule_descriptor_t const &descriptor, std::uint32_t atom_index) noexcept {
    return descriptor.total_valence[atom_index];
}

/** Total degree = explicit neighbors, including explicit Hs after `addHs`, plus any still-implicit Hs. Since the
 *  pipeline calls `MolOps::addHs` before `build_descriptor`, every H is explicit and `descriptor.degree` already
 *  counts them. `descriptor.num_hydrogens` also counts them, since RDKit's `getTotalNumHs(true)` includes
 *  neighbor Hs, so it must @b not be added a second time — that would double-count H neighbors. */
constexpr unsigned atom_total_degree(molecule_descriptor_t const &descriptor, std::uint32_t atom_index) noexcept {
    return descriptor.degree[atom_index];
}

/** N-oxide detector. Matches `RDKit::MMFF::isAtomNOxide`: nitrogen with total degree ≥ 3 bonded to a
 *  singly-coordinated oxygen. */
constexpr bool is_atom_n_oxide(molecule_descriptor_t const &descriptor, std::uint32_t atom_index) noexcept {
    if (descriptor.element[atom_index] != 7) return false;
    if (atom_total_degree(descriptor, atom_index) < 3) return false;
    std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_index];
    std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_index + 1];
    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
        std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
        if (descriptor.element[neighbor_atom] == 8 && atom_total_degree(descriptor, neighbor_atom) == 1) return true;
    }
    return false;
}

/** Count terminal oxygens single-bonded to `atom_index`. */
constexpr unsigned count_terminal_o_neighbors(molecule_descriptor_t const &descriptor,
                                              std::uint32_t atom_index) noexcept {
    unsigned oxygen_count = 0;
    std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_index];
    std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_index + 1];
    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
        std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
        if (descriptor.element[neighbor_atom] == 8 && atom_total_degree(descriptor, neighbor_atom) == 1) ++oxygen_count;
    }
    return oxygen_count;
}

/** @} */ // end of ring_queries

/**
 *  @brief Match of `RDKit::MMFF::setMMFFHeavyAtomType`. Returns an MMFF atom type in [1, 99], or 0 on failure.
 *
 *  Faithful line-by-line port of `AtomTyper.cpp:504-2078` with element-keyed dispatch, the same branch
 *  order, and identical break semantics. Hydrogen atoms should not be passed here; use
 *  `native_set_mmff_hydrogen_type` for those.
 */
constexpr std::uint8_t native_set_mmff_heavy_atom_type(molecule_descriptor_t const &descriptor,
                                                       std::uint32_t atom_index) noexcept {
    std::uint8_t atom_type = 0;
    unsigned const element = descriptor.element[atom_index];
    bool const aromatic = (descriptor.atom_flags[atom_index] & atom_flag_aromatic_k) != 0;

    // Aromatic typing handles 5-ring and 6-ring cases.
    if (aromatic) {
        bool const in_aromatic_5_ring = atom_in_aromatic_ring_of_size(descriptor, atom_index, 5);
        bool const in_aromatic_6_ring = atom_in_aromatic_ring_of_size(descriptor, atom_index, 6);

        // Collect alpha/beta heteroatoms in 5-ring context (C, N only).
        std::array<std::uint8_t, 16> alpha_het {};
        std::array<std::uint8_t, 16> beta_het {};
        unsigned alpha_count = 0, beta_count = 0;
        bool is_alpha_os = false, is_beta_os = false;
        bool alpha_or_beta_in_same_ring = false;

        if (in_aromatic_5_ring && (element == 6 || element == 7)) {
            std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_index];
            std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_index + 1];
            for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
                if (!atom_in_aromatic_ring_of_size(descriptor, neighbor_atom, 5)) continue;
                bool const is_heteroatom =
                    (descriptor.element[neighbor_atom] == 8 || descriptor.element[neighbor_atom] == 16 ||
                     (descriptor.element[neighbor_atom] == 7 && atom_total_degree(descriptor, neighbor_atom) == 3 &&
                      !is_atom_n_oxide(descriptor, neighbor_atom)));
                if (is_heteroatom && atoms_in_same_ring_of_size(descriptor, atom_index, neighbor_atom, 5) &&
                    alpha_count < alpha_het.size())
                    alpha_het[alpha_count++] = neighbor_atom;
                std::uint16_t const second_neighbor_start = descriptor.neighbor_offset[neighbor_atom];
                std::uint16_t const second_neighbor_end = descriptor.neighbor_offset[neighbor_atom + 1];
                for (std::uint16_t second_neighbor_slot = second_neighbor_start;
                     second_neighbor_slot < second_neighbor_end; ++second_neighbor_slot) {
                    std::uint8_t const second_neighbor = descriptor.neighbor_atom[second_neighbor_slot];
                    if (second_neighbor == atom_index) continue;
                    if (!atom_in_aromatic_ring_of_size(descriptor, second_neighbor, 5)) continue;
                    bool const is_second_neighbor_heteroatom = (descriptor.element[second_neighbor] == 8 ||
                                                                descriptor.element[second_neighbor] == 16 ||
                                                                (descriptor.element[second_neighbor] == 7 &&
                                                                 atom_total_degree(descriptor, second_neighbor) == 3 &&
                                                                 !is_atom_n_oxide(descriptor, second_neighbor)));
                    if (is_second_neighbor_heteroatom &&
                        atoms_in_same_ring_of_size(descriptor, atom_index, second_neighbor, 5) &&
                        beta_count < beta_het.size())
                        beta_het[beta_count++] = second_neighbor;
                }
            }
            for (unsigned alpha_index = 0; alpha_index < alpha_count && !is_alpha_os; ++alpha_index)
                is_alpha_os = (descriptor.element[alpha_het[alpha_index]] == 8 ||
                               descriptor.element[alpha_het[alpha_index]] == 16);
            for (unsigned beta_index = 0; beta_index < beta_count && !is_beta_os; ++beta_index)
                is_beta_os = (descriptor.element[beta_het[beta_index]] == 8 ||
                              descriptor.element[beta_het[beta_index]] == 16);
            if (alpha_count && beta_count) {
                for (unsigned alpha_index = 0; alpha_index < alpha_count && !alpha_or_beta_in_same_ring; ++alpha_index)
                    for (unsigned beta_index = 0; beta_index < beta_count && !alpha_or_beta_in_same_ring; ++beta_index)
                        alpha_or_beta_in_same_ring = atoms_in_same_ring_of_size(descriptor, alpha_het[alpha_index],
                                                                                beta_het[beta_index], 5);
            }
        }

        if (in_aromatic_5_ring) {
            switch (element) {
            case 6: {
                if (beta_count == 0) {
                    unsigned num_nitrogen_neighbors = 0;
                    unsigned num_charged_nitrogen_neighbors = 0;
                    unsigned num_nitrogen_in_aromatic_5ring = 0;
                    unsigned num_nitrogen_in_aromatic_6ring = 0;
                    std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_index];
                    std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_index + 1];
                    for (std::uint16_t neighbor_offset_i = neighbor_start; neighbor_offset_i < neighbor_end;
                         ++neighbor_offset_i) {
                        std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_offset_i];
                        if (descriptor.element[neighbor_atom] == 7 &&
                            atom_total_degree(descriptor, neighbor_atom) == 3) {
                            ++num_nitrogen_neighbors;
                            if (descriptor.formal_charge[neighbor_atom] > 0 &&
                                !is_atom_n_oxide(descriptor, neighbor_atom))
                                ++num_charged_nitrogen_neighbors;
                            if (atom_in_aromatic_ring_of_size(descriptor, neighbor_atom, 5))
                                ++num_nitrogen_in_aromatic_5ring;
                            if (atom_in_aromatic_ring_of_size(descriptor, neighbor_atom, 6))
                                ++num_nitrogen_in_aromatic_6ring;
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
                    bool surrounded_by_aromatic = true;
                    std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_index];
                    std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_index + 1];
                    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                        std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
                        if (descriptor.element[neighbor_atom] != 6 ||
                            !atom_in_ring_of_size(descriptor, neighbor_atom, 6))
                            surrounded_by_benzene_c = false;
                        if (atoms_in_same_ring_of_size(descriptor, atom_index, neighbor_atom, 5) &&
                            !(descriptor.atom_flags[neighbor_atom] & atom_flag_aromatic_k))
                            surrounded_by_aromatic = false;
                    }
                    if ((alpha_count == 0 && beta_count == 0 && !surrounded_by_benzene_c && surrounded_by_aromatic) ||
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
                if (is_atom_n_oxide(descriptor, atom_index)) {
                    atom_type = 82;
                    break;
                }
                if (alpha_count == 0 && beta_count == 0) {
                    if (atom_total_degree(descriptor, atom_index) == 3) {
                        atom_type = 39;
                        break;
                    }
                    atom_type = 76;
                    break;
                }
                if (atom_total_degree(descriptor, atom_index) == 3 && alpha_count != beta_count) {
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

        if (!atom_type && in_aromatic_6_ring) {
            switch (element) {
            case 6: atom_type = 37; break; // CB
            case 7:
                if (is_atom_n_oxide(descriptor, atom_index)) {
                    atom_type = 69;
                    break;
                }
                if (atom_total_degree(descriptor, atom_index) == 3) {
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
        std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_index];
        std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_index + 1];
        unsigned const total_degree = atom_total_degree(descriptor, atom_index);
        unsigned const explicit_degree = descriptor.degree[atom_index];

        // `RDKit::Bond::BondType` as an integer: 1, 2, 3, or 4 for `Bond::AROMATIC`. Only a bond RDKit
        // types aromatic reports 4 — an in-ring triple bond keeps its 3 despite `getIsAromatic()`, which
        // is what makes the isonitrile nitrogen of a dehydrobenzene ring type 61 rather than 53.
        auto bond_type_to = [&](std::uint8_t other) -> unsigned {
            for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot)
                if (descriptor.neighbor_atom[neighbor_slot] == other) {
                    std::uint16_t const bond_index = descriptor.neighbor_bond[neighbor_slot];
                    if (descriptor.bond_flags[bond_index] & bond_flag_order_aromatic_k) return 4;
                    return descriptor.bond_order[bond_index];
                }
            return 0;
        };

        // Count terminal O neighbors for later reuse (N element path).
        unsigned n_term_o_bonded_to_n = count_terminal_o_neighbors(descriptor, atom_index);

        switch (element) {
        case 3:
            if (explicit_degree == 0) atom_type = 92;
            break; // Li+
        case 6: {
            if (total_degree == 4) {
                if (atom_in_ring_of_size(descriptor, atom_index, 3)) {
                    atom_type = 22;
                    break;
                }
                if (atom_in_ring_of_size(descriptor, atom_index, 4)) {
                    atom_type = 20;
                    break;
                }
                atom_type = 1;
                break;
            }
            if (total_degree == 3) {
                unsigned num_n2_nitrogens = 0, num_n3_nitrogens = 0, num_terminal_oxygens = 0, num_terminal_sulfurs = 0;
                unsigned double_bonded_elem = 0;
                for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                    std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
                    if (bond_type_to(neighbor_atom) == 2) double_bonded_elem = descriptor.element[neighbor_atom];
                    if (atom_total_degree(descriptor, neighbor_atom) == 1) {
                        if (descriptor.element[neighbor_atom] == 8) ++num_terminal_oxygens;
                        else if (descriptor.element[neighbor_atom] == 16) ++num_terminal_sulfurs;
                    }
                    else if (descriptor.element[neighbor_atom] == 7) {
                        if (atom_total_degree(descriptor, neighbor_atom) == 3) ++num_n3_nitrogens;
                        else if (atom_total_degree(descriptor, neighbor_atom) == 2 && bond_type_to(neighbor_atom) == 2)
                            ++num_n2_nitrogens;
                    }
                }
                if (num_n3_nitrogens >= 2 && !num_n2_nitrogens && double_bonded_elem == 7) {
                    atom_type = 57;
                    break;
                }
                if (num_terminal_oxygens == 2 || num_terminal_sulfurs == 2) {
                    atom_type = 41;
                    break;
                }
                if (atom_in_ring_of_size(descriptor, atom_index, 4) && double_bonded_elem == 6) {
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
            if (total_degree == 2) {
                atom_type = 4;
                break;
            }
            if (total_degree == 1) {
                atom_type = 60;
                break;
            }
            break;
        }
        case 7: {
            // Precompute sulfonamide/sulfate flag from S or P neighbors.
            bool is_nso2_or_nso3_or_ncn = false;
            unsigned total_val_n = atom_total_valence(descriptor, atom_index);
            for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
                if (total_val_n >= 3 &&
                    (descriptor.element[neighbor_atom] == 15 || descriptor.element[neighbor_atom] == 16)) {
                    unsigned n_o_bonded_to_sp = 0;
                    std::uint16_t const second_neighbor_start = descriptor.neighbor_offset[neighbor_atom];
                    std::uint16_t const second_neighbor_end = descriptor.neighbor_offset[neighbor_atom + 1];
                    for (std::uint16_t second_neighbor_slot = second_neighbor_start;
                         second_neighbor_slot < second_neighbor_end; ++second_neighbor_slot) {
                        std::uint8_t const second_neighbor = descriptor.neighbor_atom[second_neighbor_slot];
                        if (descriptor.element[second_neighbor] == 8 &&
                            atom_total_degree(descriptor, second_neighbor) == 1)
                            ++n_o_bonded_to_sp;
                    }
                    if (n_o_bonded_to_sp >= 2) is_nso2_or_nso3_or_ncn = true;
                }
            }
            if (total_degree == 4) {
                if (is_atom_n_oxide(descriptor, atom_index)) {
                    atom_type = 68;
                    break;
                }
                atom_type = 34;
                break;
            }
            if (total_degree == 3) {
                if (total_val_n >= 4) {
                    bool double_bonded_cn = false;
                    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                        std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
                        if (bond_type_to(neighbor_atom) == 2) {
                            double_bonded_cn = (descriptor.element[neighbor_atom] == 7 ||
                                                descriptor.element[neighbor_atom] == 6);
                            if (descriptor.element[neighbor_atom] == 6) {
                                std::uint16_t const second_neighbor_start = descriptor.neighbor_offset[neighbor_atom];
                                std::uint16_t const second_neighbor_end = descriptor.neighbor_offset[neighbor_atom + 1];
                                for (std::uint16_t second_neighbor_slot = second_neighbor_start;
                                     second_neighbor_slot < second_neighbor_end && double_bonded_cn;
                                     ++second_neighbor_slot) {
                                    std::uint8_t const second_neighbor = descriptor.neighbor_atom[second_neighbor_slot];
                                    if (second_neighbor == atom_index) continue;
                                    double_bonded_cn = !(descriptor.element[second_neighbor] == 7 &&
                                                         atom_total_degree(descriptor, second_neighbor) == 3);
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
                    bool is_n_c_o_or_n_c_s = false, is_amidinium_nitrogen = false, is_guanidinium_nitrogen = false;
                    bool is_n_n_n_or_n_n_c = false, is_neighbor_carbon = false, is_neighbor_benzene_carbon = false;
                    unsigned element_double_bonded_to_carbon = 0, element_triple_bonded_to_carbon = 0;
                    unsigned num_n2_nitrogens_bonded_to_carbon = 0, num_n3_nitrogens_bonded_to_carbon = 0;
                    unsigned num_oxygens_bonded_to_carbon = 0, num_sulfurs_bonded_to_carbon = 0;
                    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                        std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
                        if (descriptor.element[neighbor_atom] == 6) {
                            is_neighbor_carbon = true;
                            if ((descriptor.atom_flags[neighbor_atom] & atom_flag_aromatic_k) &&
                                atom_in_ring_of_size(descriptor, neighbor_atom, 6))
                                is_neighbor_benzene_carbon = true;
                            num_n2_nitrogens_bonded_to_carbon = num_n3_nitrogens_bonded_to_carbon =
                                num_oxygens_bonded_to_carbon = num_sulfurs_bonded_to_carbon = 0;
                            unsigned num_formal_charged_nitrogen_neighbors = 0, num_in_aromatic_six_ring = 0;
                            std::uint16_t const second_neighbor_start = descriptor.neighbor_offset[neighbor_atom];
                            std::uint16_t const second_neighbor_end = descriptor.neighbor_offset[neighbor_atom + 1];
                            for (std::uint16_t second_neighbor_slot = second_neighbor_start;
                                 second_neighbor_slot < second_neighbor_end; ++second_neighbor_slot) {
                                std::uint8_t const second_neighbor = descriptor.neighbor_atom[second_neighbor_slot];
                                unsigned const bond_order = bond_order_between(descriptor, neighbor_atom,
                                                                               second_neighbor);
                                bool const is_aromatic_bond =
                                    bond_order_is_aromatic_between(descriptor, neighbor_atom, second_neighbor);
                                if (bond_order == 2 && (descriptor.element[second_neighbor] == 8 ||
                                                        descriptor.element[second_neighbor] == 16))
                                    is_n_c_o_or_n_c_s = true;
                                if (bond_order == 2 ||
                                    (is_aromatic_bond && (descriptor.element[second_neighbor] == 6 ||
                                                          (descriptor.element[second_neighbor] == 7 &&
                                                           atom_num_rings(descriptor, second_neighbor) == 1))))
                                    element_double_bonded_to_carbon = descriptor.element[second_neighbor];
                                if (bond_order == 3)
                                    element_triple_bonded_to_carbon = descriptor.element[second_neighbor];
                                if (descriptor.element[second_neighbor] == 7 &&
                                    atom_total_degree(descriptor, second_neighbor) == 3) {
                                    if (descriptor.formal_charge[second_neighbor] == 1)
                                        ++num_formal_charged_nitrogen_neighbors;
                                    if (atom_in_aromatic_ring_of_size(descriptor, neighbor_atom, 6))
                                        ++num_in_aromatic_six_ring;
                                    unsigned num_oxygens_bonded_to_trivalent_nitrogen = 0;
                                    std::uint16_t const third_neighbor_start =
                                        descriptor.neighbor_offset[second_neighbor];
                                    std::uint16_t const third_neighbor_end =
                                        descriptor.neighbor_offset[second_neighbor + 1];
                                    for (std::uint16_t third_neighbor_slot = third_neighbor_start;
                                         third_neighbor_slot < third_neighbor_end; ++third_neighbor_slot)
                                        if (descriptor.element[descriptor.neighbor_atom[third_neighbor_slot]] == 8)
                                            ++num_oxygens_bonded_to_trivalent_nitrogen;
                                    if (num_oxygens_bonded_to_trivalent_nitrogen < 2)
                                        ++num_n3_nitrogens_bonded_to_carbon;
                                }
                                if (descriptor.element[second_neighbor] == 7 &&
                                    atom_total_degree(descriptor, second_neighbor) == 2 &&
                                    (bond_order == 2 || is_aromatic_bond))
                                    ++num_n2_nitrogens_bonded_to_carbon;
                                if (descriptor.atom_flags[second_neighbor] & atom_flag_aromatic_k) {
                                    if (descriptor.element[second_neighbor] == 8) ++num_oxygens_bonded_to_carbon;
                                    if (descriptor.element[second_neighbor] == 16) ++num_sulfurs_bonded_to_carbon;
                                }
                            }
                            if (element_double_bonded_to_carbon == 7) {
                                if (num_n3_nitrogens_bonded_to_carbon == 2 && !num_n2_nitrogens_bonded_to_carbon &&
                                    num_formal_charged_nitrogen_neighbors && !num_in_aromatic_six_ring &&
                                    atom_total_degree(descriptor, neighbor_atom) < 4)
                                    is_amidinium_nitrogen = true;
                                if (num_n3_nitrogens_bonded_to_carbon == 3) is_guanidinium_nitrogen = true;
                            }
                        }
                        if (descriptor.element[neighbor_atom] == 7) {
                            unsigned num_nitrogens_bonded_to_carbon_neighbor = 0,
                                     num_oxygens_bonded_to_carbon_neighbor = 0,
                                     num_sulfurs_bonded_to_carbon_neighbor = 0;
                            std::uint16_t const second_neighbor_start = descriptor.neighbor_offset[neighbor_atom];
                            std::uint16_t const second_neighbor_end = descriptor.neighbor_offset[neighbor_atom + 1];
                            for (std::uint16_t second_neighbor_slot = second_neighbor_start;
                                 second_neighbor_slot < second_neighbor_end; ++second_neighbor_slot) {
                                std::uint8_t const second_neighbor = descriptor.neighbor_atom[second_neighbor_slot];
                                unsigned const bond_order = bond_order_between(descriptor, neighbor_atom,
                                                                               second_neighbor);
                                if (bond_order != 2) continue;
                                if (descriptor.element[second_neighbor] == 6) {
                                    std::uint16_t const third_neighbor_start =
                                        descriptor.neighbor_offset[second_neighbor];
                                    std::uint16_t const third_neighbor_end =
                                        descriptor.neighbor_offset[second_neighbor + 1];
                                    for (std::uint16_t third_neighbor_slot = third_neighbor_start;
                                         third_neighbor_slot < third_neighbor_end; ++third_neighbor_slot) {
                                        std::uint8_t const third_neighbor =
                                            descriptor.neighbor_atom[third_neighbor_slot];
                                        if (third_neighbor == neighbor_atom) continue;
                                        switch (descriptor.element[third_neighbor]) {
                                        case 7: ++num_nitrogens_bonded_to_carbon_neighbor; break;
                                        case 8: ++num_oxygens_bonded_to_carbon_neighbor; break;
                                        case 16: ++num_sulfurs_bonded_to_carbon_neighbor; break;
                                        }
                                    }
                                    if (!num_oxygens_bonded_to_carbon_neighbor &&
                                        !num_sulfurs_bonded_to_carbon_neighbor &&
                                        !num_nitrogens_bonded_to_carbon_neighbor && !is_neighbor_benzene_carbon)
                                        is_n_n_n_or_n_n_c = true;
                                }
                                if (descriptor.element[second_neighbor] == 7 && !is_neighbor_benzene_carbon)
                                    is_n_n_n_or_n_n_c = true;
                            }
                        }
                    }
                    if (is_neighbor_carbon) {
                        if (element_triple_bonded_to_carbon == 7) is_nso2_or_nso3_or_ncn = true;
                        if (is_amidinium_nitrogen) {
                            atom_type = 55;
                            break;
                        }
                        if (is_guanidinium_nitrogen) {
                            atom_type = 56;
                            break;
                        }
                        if (!is_n_c_o_or_n_c_s && !is_nso2_or_nso3_or_ncn &&
                            (((!num_oxygens_bonded_to_carbon) && (!num_sulfurs_bonded_to_carbon) &&
                              is_neighbor_benzene_carbon) ||
                             (element_double_bonded_to_carbon == 6 || element_double_bonded_to_carbon == 7 ||
                              element_double_bonded_to_carbon == 15 || element_triple_bonded_to_carbon == 6))) {
                            atom_type = 40;
                            break;
                        }
                    }
                    if (!is_nso2_or_nso3_or_ncn && (is_n_c_o_or_n_c_s || is_n_n_n_or_n_n_c)) {
                        atom_type = 10;
                        break;
                    }
                }
            }
            if (total_degree == 2) {
                if (total_val_n == 4) {
                    bool is_isonitrile = false;
                    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end && !is_isonitrile;
                         ++neighbor_slot)
                        is_isonitrile = bond_type_to(descriptor.neighbor_atom[neighbor_slot]) == 3;
                    if (is_isonitrile) {
                        atom_type = 61;
                        break;
                    }
                    atom_type = 53;
                    break;
                }
                if (total_val_n == 3) {
                    bool is_nitroso = false, is_imine_or_azo = false;
                    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                        std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
                        if (bond_type_to(neighbor_atom) == 2) {
                            is_nitroso = (descriptor.element[neighbor_atom] == 8 && n_term_o_bonded_to_n == 1);
                            is_imine_or_azo = (descriptor.element[neighbor_atom] == 6 ||
                                               descriptor.element[neighbor_atom] == 7);
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
                    bool is_bonded_to_sulfinyl_sulfur = false;
                    for (std::uint16_t neighbor_slot = neighbor_start;
                         neighbor_slot < neighbor_end && !is_bonded_to_sulfinyl_sulfur; ++neighbor_slot) {
                        std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
                        if (descriptor.element[neighbor_atom] == 16) {
                            unsigned num_terminal_oxygens_bonded_to_sulfur = 0;
                            std::uint16_t const second_neighbor_start = descriptor.neighbor_offset[neighbor_atom];
                            std::uint16_t const second_neighbor_end = descriptor.neighbor_offset[neighbor_atom + 1];
                            for (std::uint16_t second_neighbor_slot = second_neighbor_start;
                                 second_neighbor_slot < second_neighbor_end; ++second_neighbor_slot) {
                                std::uint8_t const second_neighbor = descriptor.neighbor_atom[second_neighbor_slot];
                                if (descriptor.element[second_neighbor] == 8 &&
                                    atom_total_degree(descriptor, second_neighbor) == 1)
                                    ++num_terminal_oxygens_bonded_to_sulfur;
                            }
                            is_bonded_to_sulfinyl_sulfur = (num_terminal_oxygens_bonded_to_sulfur == 1);
                        }
                    }
                    if (is_bonded_to_sulfinyl_sulfur) {
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
            if (total_degree == 1) {
                bool is_triple_bonded_nitrogen = false, is_nazt = false;
                for (std::uint16_t neighbor_slot = neighbor_start;
                     neighbor_slot < neighbor_end && !is_triple_bonded_nitrogen && !is_nazt; ++neighbor_slot) {
                    std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
                    is_triple_bonded_nitrogen = (bond_type_to(neighbor_atom) == 3);
                    if (descriptor.element[neighbor_atom] == 7 && atom_total_degree(descriptor, neighbor_atom) == 2) {
                        std::uint16_t const second_neighbor_start = descriptor.neighbor_offset[neighbor_atom];
                        std::uint16_t const second_neighbor_end = descriptor.neighbor_offset[neighbor_atom + 1];
                        for (std::uint16_t second_neighbor_slot = second_neighbor_start;
                             second_neighbor_slot < second_neighbor_end && !is_nazt; ++second_neighbor_slot) {
                            std::uint8_t const second_neighbor = descriptor.neighbor_atom[second_neighbor_slot];
                            is_nazt = (descriptor.element[second_neighbor] == 7 &&
                                       atom_total_degree(descriptor, second_neighbor) == 2) ||
                                      (descriptor.element[second_neighbor] == 6 &&
                                       atom_total_degree(descriptor, second_neighbor) == 3);
                        }
                    }
                }
                if (is_triple_bonded_nitrogen) {
                    atom_type = 42;
                    break;
                }
                if (is_nazt) {
                    atom_type = 47;
                    break;
                }
            }
            atom_type = 8; // NR — fallback
            break;
        }
        case 8: {
            if (total_degree == 3) {
                atom_type = 49;
                break;
            }
            if (total_degree == 2) {
                if (atom_total_valence(descriptor, atom_index) == 3) {
                    atom_type = 51;
                    break;
                }
                // OH2 (water): two H neighbors.
                if (descriptor.num_hydrogens[atom_index] == 2) {
                    atom_type = 70;
                    break;
                }
                atom_type = 6;
                break;
            }
            if (explicit_degree <= 1) {
                unsigned num_nitrogens_bonded_to_carbon_nitrogen_or_sulfur = 0,
                         num_oxygens_bonded_to_carbon_nitrogen_or_sulfur = 0,
                         num_sulfurs_bonded_to_carbon_nitrogen_or_sulfur = 0;
                bool is_oxide_oxygen_bonded_to_hydrogen = (descriptor.num_hydrogens[atom_index] > 0);
                bool is_carboxylate_oxygen = false, is_carbonyl_oxygen = false,
                     is_oxide_oxygen_bonded_to_carbon = false;
                bool is_nitroso_oxygen = false, is_oxide_oxygen_bonded_to_nitrogen = false, is_n_oxide_oxygen = false;
                bool is_nitro_oxygen = false, is_thiosulfinate_oxygen = false, is_sulfate_oxygen = false;
                bool is_sulfoxide_oxygen = false, is_phosphate_or_perchlorate_oxygen = false;
                for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                    std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
                    if (descriptor.element[neighbor_atom] == 6 || descriptor.element[neighbor_atom] == 7 ||
                        descriptor.element[neighbor_atom] == 16) {
                        std::uint16_t const second_neighbor_start = descriptor.neighbor_offset[neighbor_atom];
                        std::uint16_t const second_neighbor_end = descriptor.neighbor_offset[neighbor_atom + 1];
                        for (std::uint16_t second_neighbor_slot = second_neighbor_start;
                             second_neighbor_slot < second_neighbor_end; ++second_neighbor_slot) {
                            std::uint8_t const second_neighbor = descriptor.neighbor_atom[second_neighbor_slot];
                            if (descriptor.element[second_neighbor] == 7 &&
                                atom_total_degree(descriptor, second_neighbor) == 2)
                                ++num_nitrogens_bonded_to_carbon_nitrogen_or_sulfur;
                            if (descriptor.element[second_neighbor] == 8 &&
                                atom_total_degree(descriptor, second_neighbor) == 1)
                                ++num_oxygens_bonded_to_carbon_nitrogen_or_sulfur;
                            if (descriptor.element[second_neighbor] == 16 &&
                                atom_total_degree(descriptor, second_neighbor) == 1)
                                ++num_sulfurs_bonded_to_carbon_nitrogen_or_sulfur;
                        }
                    }
                    if (descriptor.element[neighbor_atom] == 1) is_oxide_oxygen_bonded_to_hydrogen = true;
                    unsigned const bond_type = bond_type_to(neighbor_atom);
                    if (descriptor.element[neighbor_atom] == 6) {
                        is_carboxylate_oxygen = (num_oxygens_bonded_to_carbon_nitrogen_or_sulfur == 2);
                        is_carbonyl_oxygen = (bond_type == 2);
                        is_oxide_oxygen_bonded_to_carbon = (bond_type == 1 &&
                                                            num_oxygens_bonded_to_carbon_nitrogen_or_sulfur == 1);
                    }
                    if (descriptor.element[neighbor_atom] == 7) {
                        is_nitroso_oxygen = (bond_type == 2);
                        if (bond_type == 1 && num_oxygens_bonded_to_carbon_nitrogen_or_sulfur == 1) {
                            // RDKit adds `getNumImplicitHs()` to the valence here, which `MolOps::addHs` has
                            // already driven to zero; `num_hydrogens` counts neighbor Hs the valence sum
                            // covers, so adding it would count every hydrogen twice.
                            is_oxide_oxygen_bonded_to_nitrogen = (atom_total_degree(descriptor, neighbor_atom) == 2 ||
                                                                  atom_total_valence(descriptor, neighbor_atom) == 3);
                            is_n_oxide_oxygen = (atom_total_valence(descriptor, neighbor_atom) == 4);
                        }
                        is_nitro_oxygen = (num_oxygens_bonded_to_carbon_nitrogen_or_sulfur >= 2);
                    }
                    if (descriptor.element[neighbor_atom] == 16) {
                        is_thiosulfinate_oxygen = (num_sulfurs_bonded_to_carbon_nitrogen_or_sulfur == 1);
                        is_sulfate_oxygen = (bond_type == 1 ||
                                             (bond_type == 2 &&
                                              (num_oxygens_bonded_to_carbon_nitrogen_or_sulfur +
                                               num_nitrogens_bonded_to_carbon_nitrogen_or_sulfur) > 1));
                        is_sulfoxide_oxygen = (bond_type == 2 &&
                                               (num_oxygens_bonded_to_carbon_nitrogen_or_sulfur +
                                                num_nitrogens_bonded_to_carbon_nitrogen_or_sulfur) == 1);
                    }
                    is_phosphate_or_perchlorate_oxygen = (descriptor.element[neighbor_atom] == 15 ||
                                                          descriptor.element[neighbor_atom] == 17);
                    if (is_oxide_oxygen_bonded_to_carbon || is_oxide_oxygen_bonded_to_nitrogen ||
                        is_oxide_oxygen_bonded_to_hydrogen || is_carboxylate_oxygen || is_nitro_oxygen ||
                        is_n_oxide_oxygen || is_thiosulfinate_oxygen || is_sulfate_oxygen ||
                        is_phosphate_or_perchlorate_oxygen || is_carbonyl_oxygen || is_nitroso_oxygen ||
                        is_sulfoxide_oxygen)
                        break;
                }
                if (is_oxide_oxygen_bonded_to_carbon || is_oxide_oxygen_bonded_to_nitrogen ||
                    is_oxide_oxygen_bonded_to_hydrogen) {
                    atom_type = 35;
                    break;
                }
                if (is_carboxylate_oxygen || is_nitro_oxygen || is_n_oxide_oxygen || is_thiosulfinate_oxygen ||
                    is_sulfate_oxygen || is_phosphate_or_perchlorate_oxygen) {
                    atom_type = 32;
                    break;
                }
                if (is_carbonyl_oxygen || is_nitroso_oxygen || is_sulfoxide_oxygen) {
                    atom_type = 7;
                    break;
                }
            }
            break;
        }
        case 9: {
            if (descriptor.degree[atom_index] == 1) {
                atom_type = 11;
                break;
            }
            if (descriptor.degree[atom_index] == 0) {
                atom_type = 89;
                break;
            }
            break;
        }
        case 11:
            if (descriptor.degree[atom_index] == 0) atom_type = 93;
            break; // Na+
        case 12:
            if (descriptor.degree[atom_index] == 0) atom_type = 99;
            break;                      // Mg+2
        case 14: atom_type = 19; break; // Si
        case 15: {
            if (total_degree == 4) {
                atom_type = 25;
                break;
            }
            if (total_degree == 3) {
                atom_type = 26;
                break;
            }
            if (total_degree == 2) {
                atom_type = 75;
                break;
            }
            break;
        }
        case 16: {
            if (total_degree == 3 || total_degree == 4) {
                unsigned num_oxygens_or_nitrogens_bonded_to_sulfur = 0, num_sulfurs_bonded_to_sulfur = 0;
                bool is_carbon_double_bonded_to_sulfur = false;
                for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                    std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
                    if (descriptor.element[neighbor_atom] == 6 && bond_type_to(neighbor_atom) == 2)
                        is_carbon_double_bonded_to_sulfur = true;
                    if ((descriptor.degree[neighbor_atom] == 1 && descriptor.element[neighbor_atom] == 8) ||
                        (atom_total_degree(descriptor, neighbor_atom) == 2 && descriptor.element[neighbor_atom] == 7))
                        ++num_oxygens_or_nitrogens_bonded_to_sulfur;
                    if (descriptor.degree[neighbor_atom] == 1 && descriptor.element[neighbor_atom] == 16)
                        ++num_sulfurs_bonded_to_sulfur;
                }
                if ((total_degree == 3 && num_oxygens_or_nitrogens_bonded_to_sulfur == 2 &&
                     is_carbon_double_bonded_to_sulfur) ||
                    total_degree == 4) {
                    atom_type = 18;
                    break;
                }
                if ((num_oxygens_or_nitrogens_bonded_to_sulfur && num_sulfurs_bonded_to_sulfur) ||
                    (num_oxygens_or_nitrogens_bonded_to_sulfur == 2 && !is_carbon_double_bonded_to_sulfur)) {
                    atom_type = 73;
                    break;
                }
                atom_type = 17;
                break;
            }
            if (total_degree == 2) {
                bool is_oxygen_double_bonded_to_sulfur = false;
                for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                    std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
                    if (descriptor.element[neighbor_atom] == 8 && bond_type_to(neighbor_atom) == 2)
                        is_oxygen_double_bonded_to_sulfur = true;
                }
                if (is_oxygen_double_bonded_to_sulfur) {
                    atom_type = 74;
                    break;
                }
                atom_type = 15;
                break;
            }
            if (descriptor.degree[atom_index] == 1) {
                unsigned num_terminal_sulfurs_bonded_to_neighbor = 0;
                bool is_carbon_double_bonded_to_sulfur = false;
                for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                    std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
                    std::uint16_t const second_neighbor_start = descriptor.neighbor_offset[neighbor_atom];
                    std::uint16_t const second_neighbor_end = descriptor.neighbor_offset[neighbor_atom + 1];
                    for (std::uint16_t second_neighbor_slot = second_neighbor_start;
                         second_neighbor_slot < second_neighbor_end; ++second_neighbor_slot) {
                        std::uint8_t const second_neighbor = descriptor.neighbor_atom[second_neighbor_slot];
                        if (descriptor.element[second_neighbor] == 16 &&
                            atom_total_degree(descriptor, second_neighbor) == 1)
                            ++num_terminal_sulfurs_bonded_to_neighbor;
                    }
                    if (descriptor.element[neighbor_atom] == 6 && bond_type_to(neighbor_atom) == 2)
                        is_carbon_double_bonded_to_sulfur = true;
                }
                if (is_carbon_double_bonded_to_sulfur && num_terminal_sulfurs_bonded_to_neighbor != 2) {
                    atom_type = 16;
                    break;
                }
                atom_type = 72;
                break;
            }
            break;
        }
        case 17: {
            if (total_degree == 4) {
                unsigned num_oxygens_bonded_to_chlorine = 0;
                for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot)
                    if (descriptor.element[descriptor.neighbor_atom[neighbor_slot]] == 8)
                        ++num_oxygens_bonded_to_chlorine;
                if (num_oxygens_bonded_to_chlorine == 4) {
                    atom_type = 77;
                    break;
                }
            }
            if (total_degree == 1) {
                atom_type = 12;
                break;
            }
            if (descriptor.degree[atom_index] == 0) {
                atom_type = 90;
                break;
            }
            break;
        }
        case 19:
            if (descriptor.degree[atom_index] == 0) atom_type = 94;
            break; // K+
        case 20:
            if (descriptor.degree[atom_index] == 0) atom_type = 96;
            break; // Ca+2
        case 26:
            if (descriptor.degree[atom_index] == 0) {
                if (descriptor.formal_charge[atom_index] == 2) atom_type = 87;
                else if (descriptor.formal_charge[atom_index] == 3) atom_type = 88;
            }
            break;
        case 29:
            if (descriptor.degree[atom_index] == 0) {
                if (descriptor.formal_charge[atom_index] == 1) atom_type = 97;
                else if (descriptor.formal_charge[atom_index] == 2) atom_type = 98;
            }
            break;
        case 30:
            if (descriptor.degree[atom_index] == 0) atom_type = 95;
            break; // Zn+2
        case 35:
            if (descriptor.degree[atom_index] == 1) {
                atom_type = 13;
                break;
            }
            if (descriptor.degree[atom_index] == 0) {
                atom_type = 91;
                break;
            }
            break;
        case 53:
            if (descriptor.degree[atom_index] == 1) {
                atom_type = 14;
                break;
            }
            break;
        }
    }

    return atom_type;
}

/**
 *  @brief Port of `setMMFFHydrogenType`. Takes the already-computed heavy-atom types so the H-typing nested
 *  switches can key on them.
 */
constexpr std::uint8_t native_set_mmff_hydrogen_type(molecule_descriptor_t const &descriptor, std::uint32_t atom_index,
                                                     std::span<std::uint8_t const> atom_types) noexcept {
    std::uint8_t atom_type = 0;
    std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_index];
    std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_index + 1];
    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
        std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
        unsigned const element = descriptor.element[neighbor_atom];
        switch (element) {
        case 6:
        case 14: atom_type = 5; break;
        case 7: {
            std::uint8_t const neighbor_atom_type = atom_types[neighbor_atom];
            switch (neighbor_atom_type) {
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
            std::uint8_t const neighbor_atom_type = atom_types[neighbor_atom];
            switch (neighbor_atom_type) {
            case 49: atom_type = 50; break;
            case 51: atom_type = 52; break;
            case 70: atom_type = 31; break;
            case 6: {
                bool is_h_o_c_c_or_h_o_c_n = false, is_h_o_c_o = false, is_h_o_p = false, is_h_o_s = false;
                std::uint16_t const second_neighbor_start = descriptor.neighbor_offset[neighbor_atom];
                std::uint16_t const second_neighbor_end = descriptor.neighbor_offset[neighbor_atom + 1];
                for (std::uint16_t second_neighbor_slot = second_neighbor_start;
                     second_neighbor_slot < second_neighbor_end; ++second_neighbor_slot) {
                    std::uint8_t const second_neighbor = descriptor.neighbor_atom[second_neighbor_slot];
                    if (descriptor.element[second_neighbor] == 6) {
                        std::uint16_t const third_neighbor_start = descriptor.neighbor_offset[second_neighbor];
                        std::uint16_t const third_neighbor_end = descriptor.neighbor_offset[second_neighbor + 1];
                        for (std::uint16_t third_neighbor_slot = third_neighbor_start;
                             third_neighbor_slot < third_neighbor_end; ++third_neighbor_slot) {
                            std::uint8_t const third_neighbor = descriptor.neighbor_atom[third_neighbor_slot];
                            if (third_neighbor == neighbor_atom) continue;
                            unsigned const bond_order_second_third = bond_order_between(descriptor, second_neighbor,
                                                                                        third_neighbor);
                            bool const is_aromatic_bond_second_third = bond_order_is_aromatic_between(
                                descriptor, second_neighbor, third_neighbor);
                            if ((descriptor.element[third_neighbor] == 6 || descriptor.element[third_neighbor] == 7) &&
                                (bond_order_second_third == 2 || is_aromatic_bond_second_third))
                                is_h_o_c_c_or_h_o_c_n = true;
                            if (descriptor.element[third_neighbor] == 8 && bond_order_second_third == 2)
                                is_h_o_c_o = true;
                        }
                    }
                    if (descriptor.element[second_neighbor] == 15) is_h_o_p = true;
                    if (descriptor.element[second_neighbor] == 16) is_h_o_s = true;
                }
                if (is_h_o_c_o || is_h_o_p) {
                    atom_type = 24;
                    break;
                }
                if (is_h_o_c_c_or_h_o_c_n) {
                    atom_type = 29;
                    break;
                }
                if (is_h_o_s) {
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
 *  fill). All other atoms keep `descriptor.formal_charge` (the RDKit-assigned
 *  formal charge from SMILES parsing).
 */
constexpr void native_compute_mmff_formal_charges(molecule_descriptor_t const &descriptor,
                                                  std::span<std::uint8_t const> atom_types,
                                                  std::span<float> formal_charges) noexcept {
    std::uint32_t const num_atoms = descriptor.num_atoms;
    // Initialise with zero — only atom types listed below get overridden.
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) formal_charges[atom_index] = 0.0f;

    // Scratch bitset for the conjugated-nitrogen flood fill (types 55/56/81).
    std::array<std::uint32_t, (max_atoms_per_molecule_k + 31) / 32> conj_bits {};
    auto conj_set = [&](std::uint32_t atom_index) { conj_bits[atom_index >> 5] |= (1u << (atom_index & 31)); };
    auto conj_get = [&](std::uint32_t atom_index) -> bool {
        return (conj_bits[atom_index >> 5] & (1u << (atom_index & 31))) != 0;
    };
    auto conj_reset = [&]() { conj_bits.fill(0); };

    for (std::uint32_t index = 0; index < num_atoms; ++index) {
        std::uint8_t const atom_type = atom_types[index];
        float formal_charge = 0.0f;
        switch (atom_type) {
        case 32: // O2CM — carboxylate / phosphate / sulfonate oxygen
        case 72: // SM   — anionic terminal sulfur
        {
            std::uint16_t const neighbor_start = descriptor.neighbor_offset[index];
            std::uint16_t const neighbor_end = descriptor.neighbor_offset[index + 1];
            for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                std::uint8_t const neighbor_index = descriptor.neighbor_atom[neighbor_slot];
                std::uint8_t const neighbor_type = atom_types[neighbor_index];
                int num_sec_n_bonded_to_neighbor = 0;
                int num_terminal_os_bonded_to_neighbor = 0;
                std::uint16_t const second_shell_start = descriptor.neighbor_offset[neighbor_index];
                std::uint16_t const second_shell_end = descriptor.neighbor_offset[neighbor_index + 1];
                for (std::uint16_t second_shell_slot = second_shell_start; second_shell_slot < second_shell_end;
                     ++second_shell_slot) {
                    std::uint8_t const second_shell_index = descriptor.neighbor_atom[second_shell_slot];
                    if (descriptor.element[second_shell_index] == 7 && descriptor.degree[second_shell_index] == 2 &&
                        !(descriptor.atom_flags[second_shell_index] & atom_flag_aromatic_k))
                        ++num_sec_n_bonded_to_neighbor;
                    if ((descriptor.element[second_shell_index] == 8 || descriptor.element[second_shell_index] == 16) &&
                        descriptor.degree[second_shell_index] == 1)
                        ++num_terminal_os_bonded_to_neighbor;
                }
                // Sulfonamide with 2 terminal O + 1 sec N: don't treat N
                // as an O replacement.
                if (descriptor.element[neighbor_index] == 16 && num_terminal_os_bonded_to_neighbor == 2 &&
                    num_sec_n_bonded_to_neighbor == 1)
                    num_sec_n_bonded_to_neighbor = 0;

                if (descriptor.element[neighbor_index] == 6 && num_terminal_os_bonded_to_neighbor) {
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
            std::uint32_t const ring_bitmap = descriptor.atom_ring_bitmap[index];
            for (unsigned ring_index = 0; ring_index < descriptor.num_rings; ++ring_index) {
                if (!(ring_bitmap & (1u << ring_index))) continue;
                unsigned const ring_size = descriptor.ring_offset[ring_index + 1] - descriptor.ring_offset[ring_index];
                if (ring_size != 5) continue;
                unsigned num_n76_in_ring = 0;
                for (unsigned ring_atom_slot = descriptor.ring_offset[ring_index];
                     ring_atom_slot < descriptor.ring_offset[ring_index + 1]; ++ring_atom_slot)
                    if (atom_types[descriptor.ring_atoms[ring_atom_slot]] == 76) ++num_n76_in_ring;
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
            formal_charge = static_cast<float>(descriptor.formal_charge[index]);
            unsigned num_conjugated = 1, prev_num_conjugated = 0;
            conj_reset();
            conj_set(index);
            while (num_conjugated > prev_num_conjugated) {
                prev_num_conjugated = num_conjugated;
                for (std::uint32_t atom_in_conj = 0; atom_in_conj < num_atoms; ++atom_in_conj) {
                    if (!conj_get(atom_in_conj)) continue;
                    std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_in_conj];
                    std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_in_conj + 1];
                    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                        std::uint8_t const carbon_index = descriptor.neighbor_atom[neighbor_slot];
                        std::uint8_t const carbon_type = atom_types[carbon_index];
                        if (carbon_type != 57 && carbon_type != 80) continue;
                        std::uint16_t const carbon_neighbor_start = descriptor.neighbor_offset[carbon_index];
                        std::uint16_t const carbon_neighbor_end = descriptor.neighbor_offset[carbon_index + 1];
                        for (std::uint16_t carbon_neighbor_slot = carbon_neighbor_start;
                             carbon_neighbor_slot < carbon_neighbor_end; ++carbon_neighbor_slot) {
                            std::uint8_t const candidate_index = descriptor.neighbor_atom[carbon_neighbor_slot];
                            std::uint8_t const candidate_type = atom_types[candidate_index];
                            if (candidate_type != 55 && candidate_type != 56 && candidate_type != 81) continue;
                            if (!conj_get(candidate_index)) {
                                conj_set(candidate_index);
                                formal_charge += static_cast<float>(descriptor.formal_charge[candidate_index]);
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
            std::uint16_t const neighbor_start = descriptor.neighbor_offset[index];
            std::uint16_t const neighbor_end = descriptor.neighbor_offset[index + 1];
            for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot)
                if (atom_types[descriptor.neighbor_atom[neighbor_slot]] == 42) {
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
        formal_charges[index] = formal_charge;
    }
}

/** Compute MMFF partial charges — second half of `computeMMFFCharges`
 *  (AtomTyper.cpp:3423-3487). Applies bond-charge increments (BCIs)
 *  from the extracted `chg` table, with the PBCI fallback when no BCI
 *  is tabulated. Formal-charge distribution follows equation 15 of
 *  MMFF.V. */
SF_HOST_DEVICE_ inline void native_compute_mmff_partial_charges(molecule_descriptor_t const &descriptor,
                                                                mmff94_tables_view_t const &tables,
                                                                std::span<std::uint8_t const> atom_types,
                                                                std::span<float const> formal_charges,
                                                                std::span<float> partial_charges) noexcept {
    std::uint32_t const num_atoms = descriptor.num_atoms;
    for (std::uint32_t index = 0; index < num_atoms; ++index) {
        std::uint8_t const atom_type = atom_types[index];
        auto const *prop_entry = lookup_prop(tables, atom_type);
        if (!prop_entry) {
            partial_charges[index] = formal_charges[index];
            continue;
        }
        double const crd_count = static_cast<double>(prop_entry->coordination_number);

        auto const *pbci_self = lookup_pbci(tables, atom_type);
        double fc_adjust = 0.0;
        double pbci_self_value = 0.0;
        if (pbci_self) {
            fc_adjust = static_cast<double>(pbci_self->fcadj);
            pbci_self_value = static_cast<double>(pbci_self->pbci);
        }

        double charge_base = static_cast<double>(formal_charges[index]);
        bool const fc_adjust_is_zero = std::fabs(fc_adjust) < 1.0e-10;

        std::uint16_t const neighbor_start = descriptor.neighbor_offset[index];
        std::uint16_t const neighbor_end = descriptor.neighbor_offset[index + 1];
        if (fc_adjust_is_zero) {
            for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                std::uint8_t const neighbor_index = descriptor.neighbor_atom[neighbor_slot];
                double const neighbor_formal_charge = static_cast<double>(formal_charges[neighbor_index]);
                if (neighbor_formal_charge < 0.0)
                    charge_base += neighbor_formal_charge /
                                   (2.0 * static_cast<double>(descriptor.degree[neighbor_index]));
            }
        }
        if (atom_type == 62) {
            for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
                std::uint8_t const neighbor_index = descriptor.neighbor_atom[neighbor_slot];
                double const neighbor_formal_charge = static_cast<double>(formal_charges[neighbor_index]);
                if (neighbor_formal_charge > 0.0) charge_base -= neighbor_formal_charge / 2.0;
            }
        }
        double sum_formal_charge = 0.0, sum_partial_charge = 0.0;
        for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
            std::uint8_t const neighbor_index = descriptor.neighbor_atom[neighbor_slot];
            std::uint8_t const neighbor_type = atom_types[neighbor_index];
            std::uint16_t const bond_id = descriptor.neighbor_bond[neighbor_slot];
            // Native bond type: replicate `getMMFFBondType`, whose one gate is `getBondType() == SINGLE`.
            // An aromatic bond also reports order 1, so the aromatic flag has to exclude it explicitly.
            std::uint8_t bond_type = 0;
            bool const single_bond = descriptor.bond_order[bond_id] == 1 &&
                                     !(descriptor.bond_flags[bond_id] & bond_flag_order_aromatic_k);
            if (single_bond) {
                auto const *prop_self = lookup_prop(tables, atom_type);
                auto const *prop_neighbor = lookup_prop(tables, neighbor_type);
                if (prop_self && prop_neighbor &&
                    ((prop_self->single_bond_multiple_bond && prop_neighbor->single_bond_multiple_bond) ||
                     (prop_self->aromatic && prop_neighbor->aromatic)))
                    bond_type = 1;
            }
            auto const charge_entry = lookup_chg(tables, bond_type, atom_type, neighbor_type);
            if (charge_entry.second) {
                sum_partial_charge += static_cast<double>(charge_entry.first) *
                                      static_cast<double>(charge_entry.second->bond_charge_increment);
            }
            else {
                auto const *pbci_neighbor = lookup_pbci(tables, neighbor_type);
                double const neighbor_pbci = pbci_neighbor ? static_cast<double>(pbci_neighbor->pbci) : 0.0;
                sum_partial_charge += pbci_self_value - neighbor_pbci;
            }
            sum_formal_charge += static_cast<double>(formal_charges[neighbor_index]);
        }
        double const partial_charge = (1.0 - crd_count * fc_adjust) * charge_base + fc_adjust * sum_formal_charge +
                                      sum_partial_charge;
        partial_charges[index] = static_cast<float>(partial_charge);
    }
}

/**
 *  @brief Top-level entry point that types every atom in the molecule.
 *
 *  Returns true iff every atom was typed successfully.
 */
constexpr bool native_compute_mmff_atom_types(molecule_descriptor_t const &descriptor,
                                              std::span<std::uint8_t> atom_types) noexcept {
    std::uint32_t const num_atoms = descriptor.num_atoms;
    bool ok = true;
    // Heavy atoms first.
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        if (descriptor.element[atom_index] == 1) {
            atom_types[atom_index] = 0;
            continue;
        }
        atom_types[atom_index] = native_set_mmff_heavy_atom_type(descriptor, atom_index);
        if (atom_types[atom_index] == 0) ok = false;
    }
    // Then hydrogens (need heavy atom types populated first).
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        if (descriptor.element[atom_index] != 1) continue;
        atom_types[atom_index] = native_set_mmff_hydrogen_type(descriptor, atom_index, atom_types);
        if (atom_types[atom_index] == 0) ok = false;
    }
    return ok;
}

} // namespace smallfield::mmff94
