/**
 *  @file molecule_descriptor.hpp
 *  @brief Flat stack-allocated molecule representation for zero-allocation preprocessing.
 *
 *  A single pass through RDKit's slow accessor API fills this descriptor.
 *  Downstream preprocessing (torsion extraction, bounds matrix, MMFF parameter lookup,
 *  interaction list construction) reads only this flat data — no virtual dispatch,
 *  no heap allocations, no RDKit access.
 *
 *  Layout is SoA for cache locality and so the same struct can be memcpy'd straight
 *  to device memory when the GPU upload format converges with the CPU working format.
 *
 *  Size: ~22 KB for N=128. Stack-allocates cleanly.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <cstdint>
#include <cstring>

#include <GraphMol/GraphMol.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/MolOps.h>

namespace smallfield::types {

/** Maximum atom count supported by the single-warp GPU pipeline. */
inline constexpr std::uint32_t max_atoms_k = 128;

/** Maximum bond count. Drug-like molecules with N≤128 rarely exceed 160 bonds. */
inline constexpr std::uint32_t max_bonds_k = 256;

/** Maximum number of neighbors stored across all atoms (adjacency CSR size). */
inline constexpr std::uint32_t max_neighbor_entries_k = max_bonds_k * 2;

/** Maximum SSSR rings supported. Drug-like molecules rarely exceed 8. */
inline constexpr std::uint32_t max_rings_k = 32;

/** Flat bit representation of hybridization. Matches RDKit::Atom::HybridizationType ordering. */
enum hybridization_t : std::uint8_t {
    hybridization_unspecified_k = 0,
    hybridization_s_k = 1,
    hybridization_sp_k = 2,
    hybridization_sp2_k = 3,
    hybridization_sp3_k = 4,
    hybridization_sp3d_k = 5,
    hybridization_sp3d2_k = 6,
};

/** Per-atom boolean flags packed into a single byte. */
enum atom_flag_t : std::uint8_t {
    atom_flag_aromatic_k = 1u << 0,
    atom_flag_in_ring_k = 1u << 1,
    atom_flag_hbond_donor_h_k = 1u << 2,  ///< H covalently bound to N or O
    atom_flag_hbond_acceptor_k = 1u << 3, ///< N or O with a lone pair
    atom_flag_has_chirality_k = 1u << 4,
};

/** Per-bond boolean flags packed into a single byte. */
enum bond_flag_t : std::uint8_t {
    bond_flag_aromatic_k = 1u << 0,
    bond_flag_in_ring_k = 1u << 1,
    bond_flag_conjugated_k = 1u << 2,
    bond_flag_stereo_e_k = 1u << 3, ///< E / trans
    bond_flag_stereo_z_k = 1u << 4, ///< Z / cis
};

/** @brief Sentinel in `bond_stereo_atom_i` / `_j` for a bond with no E/Z assignment. */
inline constexpr std::uint8_t no_stereo_atom_k = 0xFFu;

/**
 *  Flat molecule descriptor. All fields SoA, stack-allocated.
 *
 *  Filled by build_descriptor() in one pass through RDKit's API. Used by every
 *  downstream preprocessing stage. No RDKit calls once this is built.
 */
struct alignas(8) molecule_descriptor_t {
    std::uint8_t num_atoms;
    std::uint16_t num_bonds;
    std::uint8_t num_rings;

    std::uint8_t element[max_atoms_k];       ///< atomic number (0 for padding)
    std::uint8_t hybridization[max_atoms_k]; ///< hybridization_t
    std::uint8_t degree[max_atoms_k];        ///< total explicit degree
    std::uint8_t num_hydrogens[max_atoms_k]; ///< total H count (implicit + explicit)
    std::int8_t formal_charge[max_atoms_k];
    std::uint8_t atom_flags[max_atoms_k];    ///< bitwise OR of atom_flag_t
    std::uint8_t smallest_ring[max_atoms_k]; ///< smallest SSSR ring size containing atom (0 if none)
    std::uint8_t cip_rank[max_atoms_k];      ///< CIP priority rank (0 = lowest)
    std::uint8_t chiral_tag[max_atoms_k];    ///< 0=none, 1=CW, 2=CCW
    std::uint8_t isotope[max_atoms_k];       ///< isotope mass number (0 = natural / unspecified)

    std::uint8_t bond_atom_i[max_bonds_k];
    std::uint8_t bond_atom_j[max_bonds_k];
    std::uint8_t bond_order[max_bonds_k]; ///< 1, 2, 3 (aromatic reported as 4; check flags)
    std::uint8_t bond_flags[max_bonds_k]; ///< bitwise OR of bond_flag_t
    /** @brief The two substituents the E/Z flag is stated relative to — one neighbour of
     *  each double-bond end, `no_stereo_atom_k` when the bond carries no E/Z.  The flag
     *  alone is not enough to place an arbitrary 1-4 pair: it fixes the relationship of
     *  *these* two atoms, and any other substituent sits on the opposite side of its end. */
    std::uint8_t bond_stereo_atom_i[max_bonds_k];
    std::uint8_t bond_stereo_atom_j[max_bonds_k];

    std::uint16_t neighbor_offset[max_atoms_k + 1]; ///< prefix sum; neighbor_offset[N] = total
    std::uint8_t neighbor_atom[max_neighbor_entries_k];
    std::uint16_t neighbor_bond[max_neighbor_entries_k];

    std::uint8_t bond_distance[max_atoms_k * max_atoms_k]; ///< BFS shortest path (clamped to 255)

    std::uint16_t ring_offset[max_rings_k + 1];  ///< atoms per ring, prefix sum
    std::uint8_t ring_atoms[max_atoms_k * 2];    ///< concatenated ring atoms (SSSR is sparse)
    std::uint32_t atom_ring_bitmap[max_atoms_k]; ///< bit r of atom a = atom a is in ring r
};

/**
 *  Build a flat descriptor from an RDKit molecule in a single pass.
 *
 *  Reads every field we need for downstream preprocessing. After this call the
 *  ROMol can be discarded for preprocessing purposes.
 *
 *  Preconditions:
 *   - Hydrogens added (MolOps::addHs)
 *   - Sanitized (SmilesToMol with sanitize=true, default)
 *   - num_atoms ≤ max_atoms_k
 *
 *  @param[in] molecule RDKit molecule.
 *  @param[out] desc Descriptor to fill. Any prior contents overwritten.
 *  @return true on success, false if the molecule exceeds capacity.
 */
inline bool build_descriptor(RDKit::ROMol const &molecule, molecule_descriptor_t &desc) {
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    std::uint32_t const num_bonds = molecule.getNumBonds();
    if (num_atoms > max_atoms_k || num_bonds > max_bonds_k) return false;

    // Zero counts and prefix arrays. Per-atom/per-bond arrays don't need zeroing
    // since we fully populate entries [0, num_atoms/num_bonds).
    std::memset(desc.atom_flags, 0, sizeof(desc.atom_flags));
    std::memset(desc.atom_ring_bitmap, 0, sizeof(desc.atom_ring_bitmap));
    std::memset(desc.bond_flags, 0, sizeof(desc.bond_flags));

    desc.num_atoms = static_cast<std::uint8_t>(num_atoms);
    desc.num_bonds = static_cast<std::uint16_t>(num_bonds);

    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        auto const *atom = molecule.getAtomWithIdx(a);
        desc.element[a] = static_cast<std::uint8_t>(atom->getAtomicNum());
        desc.hybridization[a] = static_cast<std::uint8_t>(atom->getHybridization());
        desc.degree[a] = static_cast<std::uint8_t>(atom->getDegree());
        desc.num_hydrogens[a] = static_cast<std::uint8_t>(atom->getTotalNumHs(true));
        desc.formal_charge[a] = static_cast<std::int8_t>(atom->getFormalCharge());
        desc.isotope[a] = static_cast<std::uint8_t>(std::min<unsigned>(atom->getIsotope(), 255));
        desc.smallest_ring[a] = 0;
        desc.cip_rank[a] = 0;
        desc.chiral_tag[a] = 0;

        std::uint8_t flags = 0;
        if (atom->getIsAromatic()) flags |= atom_flag_aromatic_k;

        // H-bond acceptor: N or O (simplified; RDKit's full definition also checks
        // lone pair availability via charge/hybridization, but this covers drug-like cases).
        int const element = atom->getAtomicNum();
        if (element == 7 || element == 8) flags |= atom_flag_hbond_acceptor_k;
        desc.atom_flags[a] = flags;

        // Chiral tag
        auto chiral = atom->getChiralTag();
        if (chiral == RDKit::Atom::CHI_TETRAHEDRAL_CW) {
            desc.chiral_tag[a] = 1;
            desc.atom_flags[a] |= atom_flag_has_chirality_k;
        }
        else if (chiral == RDKit::Atom::CHI_TETRAHEDRAL_CCW) {
            desc.chiral_tag[a] = 2;
            desc.atom_flags[a] |= atom_flag_has_chirality_k;
        }
    }

    for (std::uint32_t b = 0; b < num_bonds; ++b) {
        auto const *bond = molecule.getBondWithIdx(b);
        desc.bond_atom_i[b] = static_cast<std::uint8_t>(bond->getBeginAtomIdx());
        desc.bond_atom_j[b] = static_cast<std::uint8_t>(bond->getEndAtomIdx());
        // Aromatic bonds report order 1.5 — we encode the integer representation here
        // and set the aromatic flag separately.
        auto order = bond->getBondType();
        std::uint8_t bond_order_u8 = 1;
        if (order == RDKit::Bond::SINGLE) bond_order_u8 = 1;
        else if (order == RDKit::Bond::DOUBLE) bond_order_u8 = 2;
        else if (order == RDKit::Bond::TRIPLE) bond_order_u8 = 3;
        else if (order == RDKit::Bond::AROMATIC) bond_order_u8 = 1; // aromatic → treat as single for topology
        desc.bond_order[b] = bond_order_u8;

        std::uint8_t flags = 0;
        if (bond->getIsAromatic()) flags |= bond_flag_aromatic_k;
        if (bond->getIsConjugated()) flags |= bond_flag_conjugated_k;
        auto stereo = bond->getStereo();
        if (stereo == RDKit::Bond::STEREOE || stereo == RDKit::Bond::STEREOTRANS) flags |= bond_flag_stereo_e_k;
        if (stereo == RDKit::Bond::STEREOZ || stereo == RDKit::Bond::STEREOCIS) flags |= bond_flag_stereo_z_k;
        desc.bond_flags[b] = flags;

        // The E/Z assignment is stated relative to a specific pair of substituents, one
        // per double-bond end.  Record them so the bounds builder can tell which side any
        // given 1-4 partner lands on.
        desc.bond_stereo_atom_i[b] = no_stereo_atom_k;
        desc.bond_stereo_atom_j[b] = no_stereo_atom_k;
        if (flags & (bond_flag_stereo_e_k | bond_flag_stereo_z_k)) {
            auto const &stereo_atoms = bond->getStereoAtoms();
            if (stereo_atoms.size() == 2) {
                std::uint32_t const first = static_cast<std::uint32_t>(stereo_atoms[0]);
                std::uint32_t const second = static_cast<std::uint32_t>(stereo_atoms[1]);
                if (first < num_atoms && second < num_atoms) {
                    desc.bond_stereo_atom_i[b] = static_cast<std::uint8_t>(first);
                    desc.bond_stereo_atom_j[b] = static_cast<std::uint8_t>(second);
                }
            }
        }
    }

    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        if (desc.element[a] != 1) continue;
        auto const *atom = molecule.getAtomWithIdx(a);
        for (auto const *neighbor : molecule.atomNeighbors(atom)) {
            int const n_element = neighbor->getAtomicNum();
            if (n_element == 7 || n_element == 8) {
                desc.atom_flags[a] |= atom_flag_hbond_donor_h_k;
                break;
            }
        }
    }

    // First pass: count neighbors per atom.
    std::uint16_t cursor[max_atoms_k + 1] = {};
    for (std::uint32_t b = 0; b < num_bonds; ++b) {
        cursor[desc.bond_atom_i[b] + 1]++;
        cursor[desc.bond_atom_j[b] + 1]++;
    }
    // Prefix sum into neighbor_offset.
    desc.neighbor_offset[0] = 0;
    for (std::uint32_t a = 0; a < num_atoms; ++a) desc.neighbor_offset[a + 1] = desc.neighbor_offset[a] + cursor[a + 1];
    // Reset cursor for second pass (used as write position).
    std::memcpy(cursor, desc.neighbor_offset, sizeof(std::uint16_t) * (num_atoms + 1));
    // Second pass: fill neighbor_atom + neighbor_bond.
    for (std::uint32_t b = 0; b < num_bonds; ++b) {
        std::uint8_t const ai = desc.bond_atom_i[b];
        std::uint8_t const aj = desc.bond_atom_j[b];
        desc.neighbor_atom[cursor[ai]] = aj;
        desc.neighbor_bond[cursor[ai]] = static_cast<std::uint16_t>(b);
        cursor[ai]++;
        desc.neighbor_atom[cursor[aj]] = ai;
        desc.neighbor_bond[cursor[aj]] = static_cast<std::uint16_t>(b);
        cursor[aj]++;
    }

    // distance clamped to 255 (plenty for N≤128). Unreachable pairs stay 255.
    std::memset(desc.bond_distance, 255, sizeof(desc.bond_distance));
    std::uint8_t bfs_queue[max_atoms_k];
    for (std::uint32_t source = 0; source < num_atoms; ++source) {
        desc.bond_distance[source * max_atoms_k + source] = 0;
        std::uint32_t head = 0;
        std::uint32_t tail = 0;
        bfs_queue[tail++] = static_cast<std::uint8_t>(source);
        while (head < tail) {
            std::uint8_t current = bfs_queue[head++];
            std::uint8_t next_dist = desc.bond_distance[source * max_atoms_k + current] + 1;
            std::uint16_t const start = desc.neighbor_offset[current];
            std::uint16_t const end = desc.neighbor_offset[current + 1];
            for (std::uint16_t n = start; n < end; ++n) {
                std::uint8_t neighbor = desc.neighbor_atom[n];
                if (desc.bond_distance[source * max_atoms_k + neighbor] > next_dist) {
                    desc.bond_distance[source * max_atoms_k + neighbor] = next_dist;
                    bfs_queue[tail++] = neighbor;
                }
            }
        }
    }

    // A bond is in a ring iff there's a path from atom_i to atom_j not using that bond.
    // Cheap test: check RDKit's bond->getIsRingBond() via the RingInfo.
    auto const *ring_info = molecule.getRingInfo();
    if (ring_info != nullptr && ring_info->isInitialized()) {
        for (std::uint32_t b = 0; b < num_bonds; ++b) {
            if (ring_info->numBondRings(b) > 0) desc.bond_flags[b] |= bond_flag_in_ring_k;
        }

        auto const &atom_rings = ring_info->atomRings();
        std::uint32_t const n_rings = std::min(static_cast<std::uint32_t>(atom_rings.size()), max_rings_k);
        desc.num_rings = static_cast<std::uint8_t>(n_rings);
        desc.ring_offset[0] = 0;
        std::uint16_t ring_cursor = 0;
        for (std::uint32_t r = 0; r < n_rings; ++r) {
            auto const &ring = atom_rings[r];
            for (int atom_idx : ring) {
                if (ring_cursor >= max_atoms_k * 2) break;
                desc.ring_atoms[ring_cursor++] = static_cast<std::uint8_t>(atom_idx);
                desc.atom_ring_bitmap[atom_idx] |= (1u << r);
                desc.atom_flags[atom_idx] |= atom_flag_in_ring_k;
                // Track smallest ring size per atom.
                std::uint8_t const rs = static_cast<std::uint8_t>(ring.size());
                if (desc.smallest_ring[atom_idx] == 0 || rs < desc.smallest_ring[atom_idx])
                    desc.smallest_ring[atom_idx] = rs;
            }
            desc.ring_offset[r + 1] = ring_cursor;
        }
    }
    else {
        desc.num_rings = 0;
        desc.ring_offset[0] = 0;
    }

    return true;
}

/**
 *  Convenience accessors for the descriptor. All inline, no allocations.
 */
inline bool atom_has_flag(molecule_descriptor_t const &desc, std::uint32_t atom_idx, atom_flag_t flag) {
    return (desc.atom_flags[atom_idx] & flag) != 0;
}

inline bool bond_has_flag(molecule_descriptor_t const &desc, std::uint32_t bond_idx, bond_flag_t flag) {
    return (desc.bond_flags[bond_idx] & flag) != 0;
}

/** Bond distance between atoms i and j (in bonds). 255 if unreachable. */
inline std::uint8_t bonds_between(molecule_descriptor_t const &desc, std::uint32_t i, std::uint32_t j) {
    return desc.bond_distance[i * max_atoms_k + j];
}

/** True iff atom i and atom j are in the same SSSR ring. */
inline bool share_ring(molecule_descriptor_t const &desc, std::uint32_t i, std::uint32_t j) {
    return (desc.atom_ring_bitmap[i] & desc.atom_ring_bitmap[j]) != 0;
}

} // namespace smallfield::types
