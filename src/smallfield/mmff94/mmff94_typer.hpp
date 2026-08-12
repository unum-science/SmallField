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

#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/views.hpp"

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

/**
 *  @brief The bond between two atoms as RDKit's typer sees it.
 *
 *  Both facts come off one CSR walk because they describe one bond, and the typer's decision trees ask
 *  for them together at every site but one. `aromatic` is `Bond::getBondType() == Bond::AROMATIC`, the
 *  only bond-aromaticity test RDKit's typer ever makes — never `getIsAromatic()`, which an in-ring triple
 *  bond also carries.
 */
struct bond_typing_t {
    /** 1, 2 or 3, and 0 when the atoms are not adjacent. An aromatic bond has no Kekulé order to report,
     *  so it reports 1. */
    std::uint8_t order = 0;
    bool aromatic = false;
};

/**
 *  @brief The typing of the bond one neighbour slot of the CSR reaches.
 *
 *  Every decision tree below that asks about a bond is already walking the slot that holds it, so this is
 *  the form nearly all of them want: the search is the exception, not the rule.
 */
constexpr bond_typing_t bond_typing_at_slot(molecule_descriptor_t const &descriptor,
                                            std::uint16_t neighbor_slot) noexcept {
    std::uint16_t const bond_index = descriptor.neighbor_bond[neighbor_slot];
    return {.order = descriptor.bond_order[bond_index],
            .aromatic = (descriptor.bond_flags[bond_index] & bond_flag_order_aromatic_k) != 0};
}

/** The typing of the bond between two atoms; `order` is 0 when they are not adjacent. */
constexpr bond_typing_t bond_typing_between(molecule_descriptor_t const &descriptor, std::uint32_t atom_i,
                                            std::uint32_t atom_j) noexcept {
    std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_i];
    std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_i + 1];
    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot)
        if (descriptor.neighbor_atom[neighbor_slot] == atom_j) return bond_typing_at_slot(descriptor, neighbor_slot);
    return {};
}

/**
 *  @brief `RDKit::Bond::BondType` as the integer the typer's decision trees compare against.
 *
 *  1, 2 or 3 for a Kekulé order, 4 for `Bond::AROMATIC`, and 0 for an absent bond. Only a bond RDKit
 *  types aromatic reports 4 — an in-ring triple bond keeps its 3 despite `getIsAromatic()`, which is what
 *  makes the isonitrile nitrogen of a dehydrobenzene ring type 61 rather than 53.
 */
constexpr unsigned rdkit_bond_type(bond_typing_t typing) noexcept { return typing.aromatic ? 4u : typing.order; }

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
            if (!bond_typing_between(descriptor, ring_atom, next_ring_atom).aromatic) {
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

/*
 *  RDKit's typer asks two per-atom questions this port answers by reading the descriptor directly.
 *
 *  Its `getTotalDegree()` is `descriptor.degree`: the pipeline runs `MolOps::addHs` before
 *  `build_descriptor`, so every hydrogen is already an explicit neighbour and `descriptor.num_hydrogens`
 *  counts those same neighbours — adding it would count every hydrogen twice.
 *
 *  Its `getValence(ExplicitType) + getNumImplicitHs()` is `descriptor.total_valence`. Summing `bond_order`
 *  cannot reproduce that, since RDKit weighs an aromatic bond 1.5 and then clamps an aromatic atom's total
 *  down to the element's nearest permitted valence.
 */

/** N-oxide detector. Matches `RDKit::MMFF::isAtomNOxide`: nitrogen with total degree ≥ 3 bonded to a
 *  singly-coordinated oxygen. */
constexpr bool is_atom_n_oxide(molecule_descriptor_t const &descriptor, std::uint32_t atom_index) noexcept {
    if (descriptor.element[atom_index] != 7) return false;
    if (descriptor.degree[atom_index] < 3) return false;
    std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_index];
    std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_index + 1];
    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
        std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
        if (descriptor.element[neighbor_atom] == 8 && descriptor.degree[neighbor_atom] == 1) return true;
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
        if (descriptor.element[neighbor_atom] == 8 && descriptor.degree[neighbor_atom] == 1) ++oxygen_count;
    }
    return oxygen_count;
}

/** @} */ // end of ring_queries

/**
 *  @brief What the aromatic five-ring branches of the heavy-atom typer ask about the heteroatoms around
 *  one atom.
 *
 *  Alpha heteroatoms are ring neighbours of the atom, beta heteroatoms ring neighbours of those. Every
 *  five-ring case of elements 6 and 7 keys on some combination of these five answers, and nothing
 *  downstream needs the atom indices themselves — which is why they stay inside the census rather than
 *  travelling as two 16-slot arrays through the eight hundred lines that follow.
 */
struct aromatic_five_ring_census_t {
    unsigned alpha_count = 0;
    unsigned beta_count = 0;
    /** An alpha heteroatom is oxygen or sulfur. */
    bool alpha_is_oxygen_or_sulfur = false;
    /** A beta heteroatom is oxygen or sulfur. */
    bool beta_is_oxygen_or_sulfur = false;
    /** Some alpha and some beta heteroatom sit in one five-ring together. */
    bool alpha_and_beta_share_ring = false;
};

/** True for the elements MMFF counts as a five-ring heteroatom: O, S, and trivalent non-N-oxide N. */
constexpr bool is_aromatic_five_ring_heteroatom_(molecule_descriptor_t const &descriptor,
                                                 std::uint8_t atom_index) noexcept {
    return descriptor.element[atom_index] == 8 || descriptor.element[atom_index] == 16 ||
           (descriptor.element[atom_index] == 7 && descriptor.degree[atom_index] == 3 &&
            !is_atom_n_oxide(descriptor, atom_index));
}

/**
 *  @brief Census the alpha and beta heteroatoms of the aromatic five-ring around @p atom_index.
 *
 *  @param[in] descriptor Molecule being typed.
 *  @param[in] atom_index Ring atom whose neighbourhood is walked; only carbon and nitrogen reach here.
 *  @return The five answers the five-ring type cases key on, all zero when the walk finds nothing.
 */
constexpr aromatic_five_ring_census_t census_aromatic_five_ring_(molecule_descriptor_t const &descriptor,
                                                                 std::uint32_t atom_index) noexcept {
    std::array<std::uint8_t, 16> alpha_heteroatoms {};
    std::array<std::uint8_t, 16> beta_heteroatoms {};
    aromatic_five_ring_census_t census;

    std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_index];
    std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_index + 1];
    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
        std::uint8_t const neighbor_atom = descriptor.neighbor_atom[neighbor_slot];
        if (!atom_in_aromatic_ring_of_size(descriptor, neighbor_atom, 5)) continue;
        if (is_aromatic_five_ring_heteroatom_(descriptor, neighbor_atom) &&
            atoms_in_same_ring_of_size(descriptor, atom_index, neighbor_atom, 5) &&
            census.alpha_count < alpha_heteroatoms.size())
            alpha_heteroatoms[census.alpha_count++] = neighbor_atom;
        std::uint16_t const second_neighbor_start = descriptor.neighbor_offset[neighbor_atom];
        std::uint16_t const second_neighbor_end = descriptor.neighbor_offset[neighbor_atom + 1];
        for (std::uint16_t second_neighbor_slot = second_neighbor_start; second_neighbor_slot < second_neighbor_end;
             ++second_neighbor_slot) {
            std::uint8_t const second_neighbor = descriptor.neighbor_atom[second_neighbor_slot];
            if (second_neighbor == atom_index) continue;
            if (!atom_in_aromatic_ring_of_size(descriptor, second_neighbor, 5)) continue;
            if (is_aromatic_five_ring_heteroatom_(descriptor, second_neighbor) &&
                atoms_in_same_ring_of_size(descriptor, atom_index, second_neighbor, 5) &&
                census.beta_count < beta_heteroatoms.size())
                beta_heteroatoms[census.beta_count++] = second_neighbor;
        }
    }

    for (unsigned alpha_index = 0; alpha_index < census.alpha_count && !census.alpha_is_oxygen_or_sulfur; ++alpha_index)
        census.alpha_is_oxygen_or_sulfur = descriptor.element[alpha_heteroatoms[alpha_index]] == 8 ||
                                           descriptor.element[alpha_heteroatoms[alpha_index]] == 16;
    for (unsigned beta_index = 0; beta_index < census.beta_count && !census.beta_is_oxygen_or_sulfur; ++beta_index)
        census.beta_is_oxygen_or_sulfur = descriptor.element[beta_heteroatoms[beta_index]] == 8 ||
                                          descriptor.element[beta_heteroatoms[beta_index]] == 16;
    for (unsigned alpha_index = 0; alpha_index < census.alpha_count && !census.alpha_and_beta_share_ring; ++alpha_index)
        for (unsigned beta_index = 0; beta_index < census.beta_count && !census.alpha_and_beta_share_ring; ++beta_index)
            census.alpha_and_beta_share_ring = atoms_in_same_ring_of_size(descriptor, alpha_heteroatoms[alpha_index],
                                                                          beta_heteroatoms[beta_index], 5);
    return census;
}

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

        // Only carbon and nitrogen have five-ring cases that key on the heteroatom census.
        aromatic_five_ring_census_t const census = (in_aromatic_5_ring && (element == 6 || element == 7))
                                                       ? census_aromatic_five_ring_(descriptor, atom_index)
                                                       : aromatic_five_ring_census_t {};
        unsigned const alpha_count = census.alpha_count;
        unsigned const beta_count = census.beta_count;

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
                        if (descriptor.element[neighbor_atom] == 7 && descriptor.degree[neighbor_atom] == 3) {
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
                        (alpha_count && beta_count &&
                         (!census.alpha_and_beta_share_ring ||
                          (!census.alpha_is_oxygen_or_sulfur && !census.beta_is_oxygen_or_sulfur)))) {
                        atom_type = 78;
                        break; // C5
                    }
                }
                if (alpha_count && (beta_count == 0 || census.alpha_is_oxygen_or_sulfur)) {
                    atom_type = 63;
                    break;
                }
                if (beta_count && (alpha_count == 0 || census.beta_is_oxygen_or_sulfur)) {
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
                    if (descriptor.degree[atom_index] == 3) {
                        atom_type = 39;
                        break;
                    }
                    atom_type = 76;
                    break;
                }
                if (descriptor.degree[atom_index] == 3 && alpha_count != beta_count) {
                    atom_type = 81;
                    break;
                }
                if (alpha_count && (beta_count == 0 || census.alpha_is_oxygen_or_sulfur)) {
                    atom_type = 65;
                    break;
                }
                if (beta_count && (alpha_count == 0 || census.beta_is_oxygen_or_sulfur)) {
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
                if (descriptor.degree[atom_index] == 3) {
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
        unsigned const total_degree = descriptor.degree[atom_index];

        // Terminal oxygens, read by the N and O element paths.
        unsigned const terminal_oxygen_count = count_terminal_o_neighbors(descriptor, atom_index);

        switch (element) {
        case 3:
            if (total_degree == 0) atom_type = 92;
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
                    unsigned const bond_type = rdkit_bond_type(bond_typing_at_slot(descriptor, neighbor_slot));
                    if (bond_type == 2) double_bonded_elem = descriptor.element[neighbor_atom];
                    if (descriptor.degree[neighbor_atom] == 1) {
                        if (descriptor.element[neighbor_atom] == 8) ++num_terminal_oxygens;
                        else if (descriptor.element[neighbor_atom] == 16) ++num_terminal_sulfurs;
                    }
                    else if (descriptor.element[neighbor_atom] == 7) {
                        if (descriptor.degree[neighbor_atom] == 3) ++num_n3_nitrogens;
                        else if (descriptor.degree[neighbor_atom] == 2 && bond_type == 2) ++num_n2_nitrogens;
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
            unsigned total_val_n = descriptor.total_valence[atom_index];
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
                        if (descriptor.element[second_neighbor] == 8 && descriptor.degree[second_neighbor] == 1)
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
                        if (rdkit_bond_type(bond_typing_at_slot(descriptor, neighbor_slot)) == 2) {
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
                                                         descriptor.degree[second_neighbor] == 3);
                                }
                            }
                        }
                    }
                    if (terminal_oxygen_count == 1) {
                        atom_type = 67;
                        break;
                    }
                    if (terminal_oxygen_count >= 2) {
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
                                auto const [bond_order, is_aromatic_bond] = bond_typing_at_slot(descriptor,
                                                                                                second_neighbor_slot);
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
                                    descriptor.degree[second_neighbor] == 3) {
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
                                    descriptor.degree[second_neighbor] == 2 && (bond_order == 2 || is_aromatic_bond))
                                    ++num_n2_nitrogens_bonded_to_carbon;
                                if (descriptor.atom_flags[second_neighbor] & atom_flag_aromatic_k) {
                                    if (descriptor.element[second_neighbor] == 8) ++num_oxygens_bonded_to_carbon;
                                    if (descriptor.element[second_neighbor] == 16) ++num_sulfurs_bonded_to_carbon;
                                }
                            }
                            if (element_double_bonded_to_carbon == 7) {
                                if (num_n3_nitrogens_bonded_to_carbon == 2 && !num_n2_nitrogens_bonded_to_carbon &&
                                    num_formal_charged_nitrogen_neighbors && !num_in_aromatic_six_ring &&
                                    descriptor.degree[neighbor_atom] < 4)
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
                                if (bond_typing_at_slot(descriptor, second_neighbor_slot).order != 2) continue;
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
                        is_isonitrile = rdkit_bond_type(bond_typing_at_slot(descriptor, neighbor_slot)) == 3;
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
                        if (rdkit_bond_type(bond_typing_at_slot(descriptor, neighbor_slot)) == 2) {
                            is_nitroso = (descriptor.element[neighbor_atom] == 8 && terminal_oxygen_count == 1);
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
                                if (descriptor.element[second_neighbor] == 8 && descriptor.degree[second_neighbor] == 1)
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
                    is_triple_bonded_nitrogen = rdkit_bond_type(bond_typing_at_slot(descriptor, neighbor_slot)) == 3;
                    if (descriptor.element[neighbor_atom] == 7 && descriptor.degree[neighbor_atom] == 2) {
                        std::uint16_t const second_neighbor_start = descriptor.neighbor_offset[neighbor_atom];
                        std::uint16_t const second_neighbor_end = descriptor.neighbor_offset[neighbor_atom + 1];
                        for (std::uint16_t second_neighbor_slot = second_neighbor_start;
                             second_neighbor_slot < second_neighbor_end && !is_nazt; ++second_neighbor_slot) {
                            std::uint8_t const second_neighbor = descriptor.neighbor_atom[second_neighbor_slot];
                            is_nazt =
                                (descriptor.element[second_neighbor] == 7 && descriptor.degree[second_neighbor] == 2) ||
                                (descriptor.element[second_neighbor] == 6 && descriptor.degree[second_neighbor] == 3);
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
                if (descriptor.total_valence[atom_index] == 3) {
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
            if (total_degree <= 1) {
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
                            if (descriptor.element[second_neighbor] == 7 && descriptor.degree[second_neighbor] == 2)
                                ++num_nitrogens_bonded_to_carbon_nitrogen_or_sulfur;
                            if (descriptor.element[second_neighbor] == 8 && descriptor.degree[second_neighbor] == 1)
                                ++num_oxygens_bonded_to_carbon_nitrogen_or_sulfur;
                            if (descriptor.element[second_neighbor] == 16 && descriptor.degree[second_neighbor] == 1)
                                ++num_sulfurs_bonded_to_carbon_nitrogen_or_sulfur;
                        }
                    }
                    if (descriptor.element[neighbor_atom] == 1) is_oxide_oxygen_bonded_to_hydrogen = true;
                    unsigned const bond_type = rdkit_bond_type(bond_typing_at_slot(descriptor, neighbor_slot));
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
                            is_oxide_oxygen_bonded_to_nitrogen = (descriptor.degree[neighbor_atom] == 2 ||
                                                                  descriptor.total_valence[neighbor_atom] == 3);
                            is_n_oxide_oxygen = (descriptor.total_valence[neighbor_atom] == 4);
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
                    if (descriptor.element[neighbor_atom] == 6 &&
                        rdkit_bond_type(bond_typing_at_slot(descriptor, neighbor_slot)) == 2)
                        is_carbon_double_bonded_to_sulfur = true;
                    if ((descriptor.degree[neighbor_atom] == 1 && descriptor.element[neighbor_atom] == 8) ||
                        (descriptor.degree[neighbor_atom] == 2 && descriptor.element[neighbor_atom] == 7))
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
                    if (descriptor.element[neighbor_atom] == 8 &&
                        rdkit_bond_type(bond_typing_at_slot(descriptor, neighbor_slot)) == 2)
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
                        if (descriptor.element[second_neighbor] == 16 && descriptor.degree[second_neighbor] == 1)
                            ++num_terminal_sulfurs_bonded_to_neighbor;
                    }
                    if (descriptor.element[neighbor_atom] == 6 &&
                        rdkit_bond_type(bond_typing_at_slot(descriptor, neighbor_slot)) == 2)
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
                                                     custd::span<std::uint8_t const> atom_types) noexcept {
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
                            auto const [bond_order, is_aromatic_bond] = bond_typing_at_slot(descriptor,
                                                                                            third_neighbor_slot);
                            if ((descriptor.element[third_neighbor] == 6 || descriptor.element[third_neighbor] == 7) &&
                                (bond_order == 2 || is_aromatic_bond))
                                is_h_o_c_c_or_h_o_c_n = true;
                            if (descriptor.element[third_neighbor] == 8 && bond_order == 2) is_h_o_c_o = true;
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

#pragma region Formal Charge Rules

/**
 *  @brief Fixed-capacity set of atom indices, backing the conjugated-nitrogen flood fill.
 *
 *  Constructed empty per fill rather than reset per fill, which is what removes the third of the three
 *  capture-everything lambdas this rule used to carry.
 */
struct atom_bitset_t {
    std::array<std::uint32_t, (max_atoms_per_molecule_k + 31) / 32> words {};

    constexpr void insert(std::uint32_t atom_index) noexcept { words[atom_index >> 5] |= 1u << (atom_index & 31); }
    [[nodiscard]] constexpr bool contains(std::uint32_t atom_index) const noexcept {
        return (words[atom_index >> 5] & (1u << (atom_index & 31))) != 0;
    }
};

/**
 *  @brief Charge on a terminal oxygen or sulfur, shared across the equivalent terminals of its centre.
 *
 *  The first neighbour that matches a rule decides, which is why each arm returns rather than accumulating.
 *
 *  @param[in] descriptor Molecule being charged.
 *  @param[in] atom_types MMFF types of every atom, already computed.
 *  @param[in] atom_index The terminal oxygen or sulfur, of MMFF type 32 or 72.
 *  @return Its share of the centre's negative charge, or zero when no rule applies.
 */
constexpr float shared_anion_formal_charge_(molecule_descriptor_t const &descriptor,
                                            custd::span<std::uint8_t const> atom_types,
                                            std::uint32_t atom_index) noexcept {
    std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_index];
    std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_index + 1];
    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot) {
        std::uint8_t const center_index = descriptor.neighbor_atom[neighbor_slot];
        std::uint8_t const center_type = atom_types[center_index];
        int num_secondary_nitrogens = 0;
        int num_terminal_chalcogens = 0;
        std::uint16_t const second_shell_start = descriptor.neighbor_offset[center_index];
        std::uint16_t const second_shell_end = descriptor.neighbor_offset[center_index + 1];
        for (std::uint16_t second_shell_slot = second_shell_start; second_shell_slot < second_shell_end;
             ++second_shell_slot) {
            std::uint8_t const second_shell_index = descriptor.neighbor_atom[second_shell_slot];
            if (descriptor.element[second_shell_index] == 7 && descriptor.degree[second_shell_index] == 2 &&
                !(descriptor.atom_flags[second_shell_index] & atom_flag_aromatic_k))
                ++num_secondary_nitrogens;
            if ((descriptor.element[second_shell_index] == 8 || descriptor.element[second_shell_index] == 16) &&
                descriptor.degree[second_shell_index] == 1)
                ++num_terminal_chalcogens;
        }
        // Every rule below divides the shared charge by this count, so a centre with no terminal
        // chalcogen has no rule and is skipped before the division exists.
        if (num_terminal_chalcogens == 0) continue;

        // A sulfonamide carrying two terminal oxygens and one secondary nitrogen does not count that
        // nitrogen as an oxygen replacement.
        if (descriptor.element[center_index] == 16 && num_terminal_chalcogens == 2 && num_secondary_nitrogens == 1)
            num_secondary_nitrogens = 0;

        float const terminal_count = static_cast<float>(num_terminal_chalcogens);
        // Most arms spread one unit over the terminals that share it.
        float const even_share = -(static_cast<float>(num_terminal_chalcogens - 1) / terminal_count);

        if (descriptor.element[center_index] == 6) return (num_terminal_chalcogens == 1) ? -1.0f : even_share;
        if (center_type == 45 && num_terminal_chalcogens == 3) return -1.0f / 3.0f;
        if (center_type == 25) return (num_terminal_chalcogens == 1) ? 0.0f : even_share;
        if (center_type == 18) {
            int const shared = num_secondary_nitrogens + num_terminal_chalcogens;
            return (shared == 2) ? 0.0f : -(static_cast<float>(shared - 2) / terminal_count);
        }
        if (center_type == 73) return (num_terminal_chalcogens == 1) ? 0.0f : even_share;
        if (center_type == 77) return -(1.0f / terminal_count);
    }
    return 0.0f;
}

/**
 *  @brief Charge on an N5M nitrogen, sharing one unit across the N5M atoms of its five-ring.
 *
 *  Only the first five-ring the atom belongs to is consulted, matching RDKit.
 *
 *  @param[in] descriptor Molecule being charged.
 *  @param[in] atom_types MMFF types of every atom, already computed.
 *  @param[in] atom_index The nitrogen, of MMFF type 76.
 *  @return Its share of the ring's unit negative charge, or zero when the ring holds no N5M.
 */
constexpr float five_ring_anion_formal_charge_(molecule_descriptor_t const &descriptor,
                                               custd::span<std::uint8_t const> atom_types,
                                               std::uint32_t atom_index) noexcept {
    std::uint32_t const ring_bitmap = descriptor.atom_ring_bitmap[atom_index];
    for (unsigned ring_index = 0; ring_index < descriptor.num_rings; ++ring_index) {
        if (!(ring_bitmap & (1u << ring_index))) continue;
        unsigned const first_slot = descriptor.ring_offset[ring_index];
        unsigned const last_slot = descriptor.ring_offset[ring_index + 1];
        if (last_slot - first_slot != 5) continue;
        unsigned num_n5m_in_ring = 0;
        for (unsigned ring_atom_slot = first_slot; ring_atom_slot < last_slot; ++ring_atom_slot)
            if (atom_types[descriptor.ring_atoms[ring_atom_slot]] == 76) ++num_n5m_in_ring;
        return num_n5m_in_ring ? -1.0f / static_cast<float>(num_n5m_in_ring) : 0.0f;
    }
    return 0.0f;
}

/**
 *  @brief Charge on a conjugated nitrogen, split evenly across the conjugated system it belongs to.
 *
 *  Flood-fills from @p atom_index through carbon types 57 and 80 to every other 55, 56 or 81 nitrogen,
 *  sums their RDKit formal charges, and divides. The fill repeats until a sweep adds nothing, because a
 *  newly reached nitrogen can open a further carbon.
 *
 *  @param[in] descriptor Molecule being charged.
 *  @param[in] atom_types MMFF types of every atom, already computed.
 *  @param[in] atom_index The nitrogen, of MMFF type 55, 56 or 81.
 *  @return Its equal share of the system's total formal charge.
 */
constexpr float conjugated_nitrogen_formal_charge_(molecule_descriptor_t const &descriptor,
                                                   custd::span<std::uint8_t const> atom_types,
                                                   std::uint32_t atom_index) noexcept {
    atom_bitset_t conjugated;
    conjugated.insert(atom_index);
    float charge_sum = static_cast<float>(descriptor.formal_charge[atom_index]);
    unsigned num_conjugated = 1, previous_num_conjugated = 0;

    while (num_conjugated > previous_num_conjugated) {
        previous_num_conjugated = num_conjugated;
        for (std::uint32_t nitrogen = 0; nitrogen < descriptor.num_atoms; ++nitrogen) {
            if (!conjugated.contains(nitrogen)) continue;
            std::uint16_t const neighbor_start = descriptor.neighbor_offset[nitrogen];
            std::uint16_t const neighbor_end = descriptor.neighbor_offset[nitrogen + 1];
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
                    if (conjugated.contains(candidate_index)) continue;
                    conjugated.insert(candidate_index);
                    charge_sum += static_cast<float>(descriptor.formal_charge[candidate_index]);
                    ++num_conjugated;
                }
            }
        }
    }
    return charge_sum / static_cast<float>(num_conjugated);
}

/**
 *  @brief Charge on an isonitrile carbon, which is +1 when it carries the triply bonded nitrogen.
 *
 *  @param[in] descriptor Molecule being charged.
 *  @param[in] atom_types MMFF types of every atom, already computed.
 *  @param[in] atom_index The carbon, of MMFF type 61.
 *  @return 1 when a neighbour is MMFF type 42, else zero.
 */
constexpr float isonitrile_carbon_formal_charge_(molecule_descriptor_t const &descriptor,
                                                 custd::span<std::uint8_t const> atom_types,
                                                 std::uint32_t atom_index) noexcept {
    std::uint16_t const neighbor_start = descriptor.neighbor_offset[atom_index];
    std::uint16_t const neighbor_end = descriptor.neighbor_offset[atom_index + 1];
    for (std::uint16_t neighbor_slot = neighbor_start; neighbor_slot < neighbor_end; ++neighbor_slot)
        if (atom_types[descriptor.neighbor_atom[neighbor_slot]] == 42) return 1.0f;
    return 0.0f;
}

#pragma endregion Formal Charge Rules

/**
 *  @brief Compute MMFF formal charges per atom.
 *
 *  Port of the first half of `computeMMFFCharges` at `AtomTyper.cpp:3094-3421`. Four MMFF types allocate
 *  charge by a rule of their own — carboxylate sharing, N5M anion splitting, the conjugated-nitrogen flood
 *  fill and the isonitrile carbon — and the rest read straight off the type.
 *
 *  @param[in] descriptor Molecule being charged.
 *  @param[in] atom_types MMFF type of every atom, already computed.
 *  @param[out] formal_charges One per atom; every slot is written.
 */
constexpr void native_compute_mmff_formal_charges(molecule_descriptor_t const &descriptor,
                                                  custd::span<std::uint8_t const> atom_types,
                                                  custd::span<float> formal_charges) noexcept {
    for (std::uint32_t atom_index = 0; atom_index < descriptor.num_atoms; ++atom_index) {
        float formal_charge = 0.0f;
        switch (atom_types[atom_index]) {
        // O2CM carboxylate / phosphate / sulfonate oxygen, and SM anionic terminal sulfur.
        case 32:
        case 72: formal_charge = shared_anion_formal_charge_(descriptor, atom_types, atom_index); break;
        case 76: formal_charge = five_ring_anion_formal_charge_(descriptor, atom_types, atom_index); break;
        // Amidinium, guanidinium and imidazolium nitrogen.
        case 55:
        case 56:
        case 81: formal_charge = conjugated_nitrogen_formal_charge_(descriptor, atom_types, atom_index); break;
        case 61: formal_charge = isonitrile_carbon_formal_charge_(descriptor, atom_types, atom_index); break;
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
        formal_charges[atom_index] = formal_charge;
    }
}

/**
 *  @brief Native port of `MMFFMolProperties::getMMFFBondType`.
 *
 *  Returns 1 iff the bond is SINGLE and its two endpoints either both carry the MMFFProp
 *  single-bond-multiple-bond designator or are both aromatic atoms. Otherwise 0.
 *
 *  RDKit's one gate is `getBondType() == Bond::SINGLE`, so the test asks `bond_flag_order_aromatic_k`
 *  rather than `bond_flag_aromatic_k`: the latter is `getIsAromatic()`, which an in-ring triple bond and
 *  a Kekulé single bond both carry, and it is not the question RDKit asks.
 *
 *  @param[in] descriptor Molecule being typed.
 *  @param[in] tables Extracted MMFF94 tables, read for the two endpoints' MMFFProp rows.
 *  @param[in] atom_types MMFF type of every atom, already computed.
 *  @param[in] bond_index Bond to classify.
 *  @return MMFF bond type, 0 or 1.
 */
constexpr std::uint8_t native_get_mmff_bond_type(molecule_descriptor_t const &descriptor,
                                                 mmff94_tables_view_t const &tables,
                                                 custd::span<std::uint8_t const> atom_types,
                                                 std::uint32_t bond_index) noexcept {
    /*  The pair below is `getBondType() == Bond::SINGLE` and nothing else, which is RDKit's whole gate
     *  at `AtomTyper.cpp:2470`. `bond_order` reports 1 for a single bond and also 1 for an aromatic
     *  one, which carries no integer order, so the aromatic case is excluded by the flag that means
     *  exactly `getBondType() == Bond::AROMATIC`.
     *
     *  Not `bond_flag_aromatic_k`, which is `getIsAromatic()` and is set on every bond in an aromatic
     *  ring including one that kept a Kekulé single order. Asking that flag would drop a bond RDKit
     *  types as SINGLE, so the two are not interchangeable and only this one is parity. Two of the
     *  three copies this function replaces asked the wider flag, so the majority was the defect. */
    bool const single_bond = descriptor.bond_order[bond_index] == 1 &&
                             !(descriptor.bond_flags[bond_index] & bond_flag_order_aromatic_k);
    if (!single_bond) return 0;
    auto const *prop_i = lookup_prop(tables, atom_types[descriptor.bond_atom_i[bond_index]]);
    auto const *prop_j = lookup_prop(tables, atom_types[descriptor.bond_atom_j[bond_index]]);
    if (!prop_i || !prop_j) return 0;
    return ((prop_i->single_bond_multiple_bond && prop_j->single_bond_multiple_bond) ||
            (prop_i->aromatic && prop_j->aromatic))
               ? 1u
               : 0u;
}

/** Compute MMFF partial charges — second half of `computeMMFFCharges`
 *  (AtomTyper.cpp:3423-3487). Applies bond-charge increments (BCIs)
 *  from the extracted `chg` table, with the PBCI fallback when no BCI
 *  is tabulated. Formal-charge distribution follows equation 15 of
 *  MMFF.V. */
SF_HOST_DEVICE_ inline void native_compute_mmff_partial_charges(molecule_descriptor_t const &descriptor,
                                                                mmff94_tables_view_t const &tables,
                                                                custd::span<std::uint8_t const> atom_types,
                                                                custd::span<float const> formal_charges,
                                                                custd::span<float> partial_charges) noexcept {
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
            std::uint8_t const bond_type = native_get_mmff_bond_type(descriptor, tables, atom_types,
                                                                     descriptor.neighbor_bond[neighbor_slot]);
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
                                              custd::span<std::uint8_t> atom_types) noexcept {
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
