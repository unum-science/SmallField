/**
 *  @file src/smallfield/preprocess/device_capacities.hpp
 *  @brief Per-molecule capacities for the regions the device preprocessing kernels fill.
 *
 *  The MMFF extractor, the CSD torsion matcher and the ETK CSR builder all write into batch-global
 *  arrays at a base the host assigns, so the host has to know how much room each molecule claims
 *  before any kernel runs. Every count below is an enumeration count over the descriptor's own
 *  topology, and every kernel emits a subset of that enumeration — a torsion whose three barrier
 *  heights all vanish is dropped, an out-of-plane centre with no table entry is dropped — so the
 *  numbers here are upper bounds that hold molecule by molecule.
 *
 *  Sizing this way is what lets the pipeline skip `mmff94_scan_counts_kernel`: the scan's totals
 *  are exact but live in device memory, and reading them back to size the packed arrays would put
 *  a synchronize in the middle of an otherwise asynchronous chain. The price is the slack between
 *  bound and actual, which leaves gaps in the packed arrays that no consumer reads. Measured over
 *  a 5000-SMILES stratified PubChem sample the device-only regions reserve 34.5 KB per molecule
 *  against the 17.2 KB the packed arrays actually hold, and over a 5000-SMILES Enamine REAL sample
 *  35.1 KB against 19.4 KB — so the slack costs roughly twice the packed footprint, and buys an
 *  entirely asynchronous preprocessing chain.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

#include <algorithm> // `std::min`

#include "smallfield/types/mmff94_bounds.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::preprocess {

using smallfield::types::molecule_descriptor_t;

/** Elements one molecule claims in each device-filled region. */
struct device_capacities_t {
    std::uint32_t etk_torsions;
    std::uint32_t etk_interactions;
    std::uint32_t mmff_bonds;
    std::uint32_t mmff_angles;
    std::uint32_t mmff_torsions;
    std::uint32_t mmff_oops;
    std::uint32_t mmff_stretchbends;
    std::uint32_t mmff_interactions;
};

/**
 *  @brief Upper-bound the elements every device preprocessing kernel can emit for one molecule.
 *
 *  @param[in] descriptor Flat descriptor the kernels will read.
 *  @param[in] num_distance_constraints 1-2 and 1-3 constraints the host already emitted, which
 *             enter the ETK per-atom list alongside the torsions.
 *  @return Per-region capacities, each clamped to the matching `max_mmff_*_per_mol_k` ceiling
 *          the extractor itself saturates at.
 */
inline device_capacities_t compute_device_capacities(molecule_descriptor_t const &descriptor,
                                                     std::uint32_t num_distance_constraints) noexcept {
    std::uint32_t const num_atoms = descriptor.num_atoms;
    std::uint32_t const num_bonds = descriptor.num_bonds;

    // Angles and stretch-bends both walk the neighbour pairs of every centre.
    std::uint32_t angle_pairs = 0;
    std::uint32_t out_of_plane_slots = 0;
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint32_t const degree = descriptor.degree[center];
        angle_pairs += degree * (degree > 0 ? degree - 1 : 0) / 2;
        // The extractor caches at most eight neighbours and emits one term per cached neighbour.
        if (degree >= 3) out_of_plane_slots += std::min<std::uint32_t>(degree, 8);
    }

    // Torsions walk every bond as the central pair and pair off the two end atoms' other neighbours.
    std::uint32_t torsion_quadruples = 0;
    for (std::uint32_t bond = 0; bond < num_bonds; ++bond) {
        std::uint32_t const degree_j = descriptor.degree[descriptor.bond_atom_i[bond]];
        std::uint32_t const degree_k = descriptor.degree[descriptor.bond_atom_j[bond]];
        torsion_quadruples += (degree_j > 0 ? degree_j - 1 : 0) * (degree_k > 0 ? degree_k - 1 : 0);
    }

    device_capacities_t capacities {};
    // A pattern claims a bond at most once and the ring pass only visits bonds no pattern claimed,
    // so the torsion set of a molecule never outgrows its bond list.
    capacities.etk_torsions = num_bonds;
    capacities.etk_interactions = 4u * num_bonds + 2u * num_distance_constraints;
    capacities.mmff_bonds = std::min(num_bonds, smallfield::types::max_mmff_bonds_per_mol_k);
    capacities.mmff_angles = std::min(angle_pairs, smallfield::types::max_mmff_angles_per_mol_k);
    capacities.mmff_torsions = std::min(torsion_quadruples, smallfield::types::max_mmff_torsions_per_mol_k);
    capacities.mmff_oops = std::min(out_of_plane_slots, smallfield::types::max_mmff_oops_per_mol_k);
    capacities.mmff_stretchbends = std::min(angle_pairs, smallfield::types::max_mmff_stretchbends_per_mol_k);
    capacities.mmff_interactions = std::min(2u * capacities.mmff_bonds + 3u * capacities.mmff_angles +
                                                4u * capacities.mmff_torsions + 4u * capacities.mmff_oops +
                                                3u * capacities.mmff_stretchbends,
                                            smallfield::types::max_mmff_interactions_per_mol_k);
    return capacities;
}

} // namespace smallfield::preprocess
