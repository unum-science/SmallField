/**
 *  @file src/smallfield/types/molecule_descriptor.hpp
 *  @brief Flat stack-allocated molecule representation for zero-allocation preprocessing.
 *
 *  A single pass through RDKit's slow accessor API fills this descriptor. Downstream preprocessing (torsion
 *  extraction, bounds matrix, MMFF parameter lookup, interaction list construction) reads only this flat data —
 *  no virtual dispatch, no heap allocations, no RDKit access.
 *
 *  Layout is SoA for cache locality and so the same struct can be memcpy'd straight to device memory when the
 *  GPU upload format converges with the CPU working format.
 *
 *  Stack-allocates cleanly. The all-pairs bond-hop matrix is not a member: it is a pure function of the CSR
 *  adjacency, so every consumer derives it — `fill_bond_distances` on the host, the warp BFS in
 *  `smallfield/cuda/bond_distance_kernel.cuh` on the device.
 */

#pragma once

#include <cstdint> // `std::uint32_t`, `std::uint8_t`
#include <cstring> // `std::memset`, `std::memcpy`

#include <GraphMol/GraphMol.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/MolOps.h>

namespace smallfield::types {

/** Maximum atom count supported by the single-warp GPU pipeline. */
inline constexpr std::uint32_t max_atoms_per_molecule_k = 192;

/** Maximum bond count. A connected molecule at the 192-atom cap carries 191 bonds plus one per ring
 *  closure, so 384 covers the widest molecule the 32-ring cap admits. */
inline constexpr std::uint32_t max_bonds_k = 384;

/** Maximum number of neighbors stored across all atoms (adjacency CSR size). */
inline constexpr std::uint32_t max_neighbor_entries_k = max_bonds_k * 2;

/** Maximum SSSR rings supported. Drug-like molecules rarely exceed 8. */
inline constexpr std::uint32_t max_rings_k = 32;

/** Flat bit representation of hybridization. Matches `RDKit::Atom::HybridizationType` ordering. */
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
    /** H covalently bound to N or O. */
    atom_flag_hbond_donor_h_k = 1u << 2,
    /** N or O with a lone pair. */
    atom_flag_hbond_acceptor_k = 1u << 3,
    atom_flag_has_chirality_k = 1u << 4,
};

/** Per-bond boolean flags packed into a single byte. */
enum bond_flag_t : std::uint8_t {
    bond_flag_aromatic_k = 1u << 0,
    bond_flag_in_ring_k = 1u << 1,
    bond_flag_conjugated_k = 1u << 2,
    /** E / trans. */
    bond_flag_stereo_e_k = 1u << 3,
    /** Z / cis. */
    bond_flag_stereo_z_k = 1u << 4,
};

/** Sentinel in `bond_stereo_atom_i` / `_j` for a bond with no E/Z assignment. */
inline constexpr std::uint8_t no_stereo_atom_k = 0xFFu;

/**
 *  @brief Flat molecule descriptor. All fields SoA, stack-allocated.
 *
 *  Used by every downstream preprocessing stage. No RDKit calls once this is built.
 */
struct alignas(8) molecule_descriptor_t {
    std::uint8_t num_atoms;
    std::uint16_t num_bonds;
    std::uint8_t num_rings;

    /** Atomic number; 0 for padding. */
    std::uint8_t element[max_atoms_per_molecule_k];
    /** `hybridization_t` value. */
    std::uint8_t hybridization[max_atoms_per_molecule_k];
    /** Total explicit degree. */
    std::uint8_t degree[max_atoms_per_molecule_k];
    /** Total H count, implicit plus explicit. */
    std::uint8_t num_hydrogens[max_atoms_per_molecule_k];
    std::int8_t formal_charge[max_atoms_per_molecule_k];
    /** Bitwise OR of `atom_flag_t`. */
    std::uint8_t atom_flags[max_atoms_per_molecule_k];
    /** Smallest SSSR ring size containing the atom; 0 if none. */
    std::uint8_t smallest_ring[max_atoms_per_molecule_k];
    /** Ring-bond connectivity, the SMARTS `x<n>` primitive: how many of the atom's bonds
     *  carry `bond_flag_in_ring_k`. Zero for an acyclic atom and for a ring atom whose every
     *  bond leaves the ring, which cannot happen, so zero reads as "not in a ring". */
    std::uint8_t ring_bond_degree[max_atoms_per_molecule_k];
    /** CIP priority rank; 0 is lowest. */
    std::uint8_t cip_rank[max_atoms_per_molecule_k];
    /** 0=none, 1=CW, 2=CCW. */
    std::uint8_t chiral_tag[max_atoms_per_molecule_k];
    /** Isotope mass number; 0 is natural or unspecified. */
    std::uint8_t isotope[max_atoms_per_molecule_k];

    std::uint8_t bond_atom_i[max_bonds_k];
    std::uint8_t bond_atom_j[max_bonds_k];
    /** 1, 2, or 3. An aromatic bond reports 1 and carries `bond_flag_aromatic_k` instead. */
    std::uint8_t bond_order[max_bonds_k];
    /** Bitwise OR of `bond_flag_t`. */
    std::uint8_t bond_flags[max_bonds_k];
    /** The two substituents the E/Z flag is stated relative to — one neighbour of
     *  each double-bond end, `no_stereo_atom_k` when the bond carries no E/Z. The flag
     *  alone is not enough to place an arbitrary 1-4 pair: it fixes the relationship of
     *  *these* two atoms, and any other substituent sits on the opposite side of its end. */
    std::uint8_t bond_stereo_atom_i[max_bonds_k];
    std::uint8_t bond_stereo_atom_j[max_bonds_k];

    /** Prefix sum; `neighbor_offset[N]` is the total. */
    std::uint16_t neighbor_offset[max_atoms_per_molecule_k + 1];
    std::uint8_t neighbor_atom[max_neighbor_entries_k];
    std::uint16_t neighbor_bond[max_neighbor_entries_k];

    /** Atoms per ring, prefix sum. */
    std::uint16_t ring_offset[max_rings_k + 1];
    /** Concatenated ring atoms; SSSR is sparse. */
    std::uint8_t ring_atoms[max_atoms_per_molecule_k * 2];
    /** Bit `ring_index` of an atom is set when that atom belongs to that ring. */
    std::uint32_t atom_ring_bitmap[max_atoms_per_molecule_k];
};

/**
 *  @brief Build a flat descriptor from an RDKit molecule in a single pass.
 *
 *  Reads every field needed for downstream preprocessing. After this call the
 *  ROMol can be discarded for preprocessing purposes.
 *
 *  Preconditions:
 *   - Hydrogens added through `MolOps::addHs`
 *   - Sanitized — `SmilesToMol` with `sanitize=true`, its default
 *   - `num_atoms` ≤ `max_atoms_per_molecule_k`
 *
 *  @param[in] molecule RDKit molecule.
 *  @param[out] descriptor Descriptor to fill. Any prior contents overwritten.
 *  @return true on success, false if the molecule exceeds capacity.
 */
inline bool build_descriptor(RDKit::ROMol const &molecule, molecule_descriptor_t &descriptor) {
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    std::uint32_t const num_bonds = molecule.getNumBonds();
    if (num_atoms > max_atoms_per_molecule_k || num_bonds > max_bonds_k) return false;

    // Zero counts and prefix arrays. Per-atom/per-bond arrays don't need zeroing since the loops
    // below fully populate entries [0, num_atoms/num_bonds).
    std::memset(descriptor.atom_flags, 0, sizeof(descriptor.atom_flags));
    std::memset(descriptor.atom_ring_bitmap, 0, sizeof(descriptor.atom_ring_bitmap));
    std::memset(descriptor.bond_flags, 0, sizeof(descriptor.bond_flags));

    descriptor.num_atoms = static_cast<std::uint8_t>(num_atoms);
    descriptor.num_bonds = static_cast<std::uint16_t>(num_bonds);

    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        auto const *atom = molecule.getAtomWithIdx(atom_index);
        descriptor.element[atom_index] = static_cast<std::uint8_t>(atom->getAtomicNum());
        descriptor.hybridization[atom_index] = static_cast<std::uint8_t>(atom->getHybridization());
        descriptor.degree[atom_index] = static_cast<std::uint8_t>(atom->getDegree());
        descriptor.num_hydrogens[atom_index] = static_cast<std::uint8_t>(atom->getTotalNumHs(true));
        descriptor.formal_charge[atom_index] = static_cast<std::int8_t>(atom->getFormalCharge());
        descriptor.isotope[atom_index] = static_cast<std::uint8_t>(std::min<unsigned>(atom->getIsotope(), 255));
        descriptor.smallest_ring[atom_index] = 0;
        descriptor.cip_rank[atom_index] = 0;
        descriptor.chiral_tag[atom_index] = 0;

        std::uint8_t flags = 0;
        if (atom->getIsAromatic()) flags |= atom_flag_aromatic_k;

        // H-bond acceptor: N or O. Simplified — RDKit's full definition also checks lone pair
        // availability via charge and hybridization — but this covers drug-like cases.
        int const element = atom->getAtomicNum();
        if (element == 7 || element == 8) flags |= atom_flag_hbond_acceptor_k;
        descriptor.atom_flags[atom_index] = flags;

        // Chiral tag
        auto chiral = atom->getChiralTag();
        if (chiral == RDKit::Atom::CHI_TETRAHEDRAL_CW) {
            descriptor.chiral_tag[atom_index] = 1;
            descriptor.atom_flags[atom_index] |= atom_flag_has_chirality_k;
        }
        else if (chiral == RDKit::Atom::CHI_TETRAHEDRAL_CCW) {
            descriptor.chiral_tag[atom_index] = 2;
            descriptor.atom_flags[atom_index] |= atom_flag_has_chirality_k;
        }
    }

    for (std::uint32_t bond_index = 0; bond_index < num_bonds; ++bond_index) {
        auto const *bond = molecule.getBondWithIdx(bond_index);
        descriptor.bond_atom_i[bond_index] = static_cast<std::uint8_t>(bond->getBeginAtomIdx());
        descriptor.bond_atom_j[bond_index] = static_cast<std::uint8_t>(bond->getEndAtomIdx());
        // Aromatic bonds report order 1.5 — the integer representation is encoded here and the
        // aromatic flag is set separately.
        auto order = bond->getBondType();
        std::uint8_t bond_order_u8 = 1;
        if (order == RDKit::Bond::SINGLE) bond_order_u8 = 1;
        else if (order == RDKit::Bond::DOUBLE) bond_order_u8 = 2;
        else if (order == RDKit::Bond::TRIPLE) bond_order_u8 = 3;
        else if (order == RDKit::Bond::AROMATIC) bond_order_u8 = 1; // aromatic → treat as single for topology
        descriptor.bond_order[bond_index] = bond_order_u8;

        std::uint8_t flags = 0;
        if (bond->getIsAromatic()) flags |= bond_flag_aromatic_k;
        if (bond->getIsConjugated()) flags |= bond_flag_conjugated_k;
        auto stereo = bond->getStereo();
        if (stereo == RDKit::Bond::STEREOE || stereo == RDKit::Bond::STEREOTRANS) flags |= bond_flag_stereo_e_k;
        if (stereo == RDKit::Bond::STEREOZ || stereo == RDKit::Bond::STEREOCIS) flags |= bond_flag_stereo_z_k;
        descriptor.bond_flags[bond_index] = flags;

        // The E/Z assignment is stated relative to a specific pair of substituents, one
        // per double-bond end. Record them so the bounds builder can tell which side any
        // given 1-4 partner lands on.
        descriptor.bond_stereo_atom_i[bond_index] = no_stereo_atom_k;
        descriptor.bond_stereo_atom_j[bond_index] = no_stereo_atom_k;
        if (flags & (bond_flag_stereo_e_k | bond_flag_stereo_z_k)) {
            auto const &stereo_atoms = bond->getStereoAtoms();
            if (stereo_atoms.size() == 2) {
                std::uint32_t const first = static_cast<std::uint32_t>(stereo_atoms[0]);
                std::uint32_t const second = static_cast<std::uint32_t>(stereo_atoms[1]);
                if (first < num_atoms && second < num_atoms) {
                    descriptor.bond_stereo_atom_i[bond_index] = static_cast<std::uint8_t>(first);
                    descriptor.bond_stereo_atom_j[bond_index] = static_cast<std::uint8_t>(second);
                }
            }
        }
    }

    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        if (descriptor.element[atom_index] != 1) continue;
        auto const *atom = molecule.getAtomWithIdx(atom_index);
        for (auto const *neighbor : molecule.atomNeighbors(atom)) {
            int const neighbor_element = neighbor->getAtomicNum();
            if (neighbor_element == 7 || neighbor_element == 8) {
                descriptor.atom_flags[atom_index] |= atom_flag_hbond_donor_h_k;
                break;
            }
        }
    }

    // First pass: count neighbors per atom.
    std::uint16_t cursor[max_atoms_per_molecule_k + 1] = {};
    for (std::uint32_t bond_index = 0; bond_index < num_bonds; ++bond_index) {
        cursor[descriptor.bond_atom_i[bond_index] + 1]++;
        cursor[descriptor.bond_atom_j[bond_index] + 1]++;
    }
    // Prefix sum into neighbor_offset.
    descriptor.neighbor_offset[0] = 0;
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index)
        descriptor.neighbor_offset[atom_index + 1] = descriptor.neighbor_offset[atom_index] + cursor[atom_index + 1];
    // Reset cursor for the second pass, where it holds the write position.
    std::memcpy(cursor, descriptor.neighbor_offset, sizeof(std::uint16_t) * (num_atoms + 1));
    // Second pass: fill neighbor_atom + neighbor_bond.
    for (std::uint32_t bond_index = 0; bond_index < num_bonds; ++bond_index) {
        std::uint8_t const atom_i = descriptor.bond_atom_i[bond_index];
        std::uint8_t const atom_j = descriptor.bond_atom_j[bond_index];
        descriptor.neighbor_atom[cursor[atom_i]] = atom_j;
        descriptor.neighbor_bond[cursor[atom_i]] = static_cast<std::uint16_t>(bond_index);
        cursor[atom_i]++;
        descriptor.neighbor_atom[cursor[atom_j]] = atom_i;
        descriptor.neighbor_bond[cursor[atom_j]] = static_cast<std::uint16_t>(bond_index);
        cursor[atom_j]++;
    }

    // A bond is in a ring iff there's a path from atom_i to atom_j not using that bond.
    // Cheap test: check RDKit's `bond->getIsRingBond()` via the `RingInfo`.
    std::memset(descriptor.ring_bond_degree, 0, sizeof(descriptor.ring_bond_degree));
    auto const *ring_info = molecule.getRingInfo();
    if (ring_info != nullptr && ring_info->isInitialized()) {
        for (std::uint32_t bond_index = 0; bond_index < num_bonds; ++bond_index) {
            if (ring_info->numBondRings(bond_index) == 0) continue;
            descriptor.bond_flags[bond_index] |= bond_flag_in_ring_k;
            descriptor.ring_bond_degree[descriptor.bond_atom_i[bond_index]]++;
            descriptor.ring_bond_degree[descriptor.bond_atom_j[bond_index]]++;
        }

        auto const &atom_rings = ring_info->atomRings();
        std::uint32_t const num_rings = std::min(static_cast<std::uint32_t>(atom_rings.size()), max_rings_k);
        descriptor.num_rings = static_cast<std::uint8_t>(num_rings);
        descriptor.ring_offset[0] = 0;
        std::uint16_t ring_cursor = 0;
        for (std::uint32_t ring_index = 0; ring_index < num_rings; ++ring_index) {
            auto const &ring = atom_rings[ring_index];
            for (int atom_index : ring) {
                if (ring_cursor >= max_atoms_per_molecule_k * 2) break;
                descriptor.ring_atoms[ring_cursor++] = static_cast<std::uint8_t>(atom_index);
                descriptor.atom_ring_bitmap[atom_index] |= (1u << ring_index);
                descriptor.atom_flags[atom_index] |= atom_flag_in_ring_k;
                // Track smallest ring size per atom.
                std::uint8_t const ring_size = static_cast<std::uint8_t>(ring.size());
                if (descriptor.smallest_ring[atom_index] == 0 || ring_size < descriptor.smallest_ring[atom_index])
                    descriptor.smallest_ring[atom_index] = ring_size;
            }
            descriptor.ring_offset[ring_index + 1] = ring_cursor;
        }
    }
    else {
        descriptor.num_rings = 0;
        descriptor.ring_offset[0] = 0;
    }

    return true;
}

/**
 *  @brief Convenience accessors for the descriptor. All inline, no allocations.
 */
inline bool atom_has_flag(molecule_descriptor_t const &descriptor, std::uint32_t atom_index, atom_flag_t flag) {
    return (descriptor.atom_flags[atom_index] & flag) != 0;
}

inline bool bond_has_flag(molecule_descriptor_t const &descriptor, std::uint32_t bond_index, bond_flag_t flag) {
    return (descriptor.bond_flags[bond_index] & flag) != 0;
}

/**
 *  @brief Fill @p matrix with the all-pairs bond-hop counts of @p descriptor.
 *
 *  One breadth-first search per source atom over the CSR adjacency, writing row `source`
 *  at `source*row_stride`. Atoms with no bonded path between them — the two halves of a
 *  salt, a solvate's water — keep the 255 sentinel that the MMFF nonbonded sweep reads as
 *  "unreachable", and 255 is also the saturation value, unreachable for N <= 192 where the
 *  longest simple path is 191 hops.
 *
 *  @param[in] descriptor Descriptor supplying the atom count and the CSR adjacency.
 *  @param[out] matrix Caller-owned buffer of at least @p row_stride times `descriptor.num_atoms` bytes.
 *  @param[in] row_stride Element distance between consecutive rows; at least `descriptor.num_atoms`.
 */
inline void fill_bond_distances(molecule_descriptor_t const &descriptor, std::uint8_t *matrix,
                                std::uint32_t row_stride) {
    std::uint32_t const num_atoms = descriptor.num_atoms;
    std::memset(matrix, 255, static_cast<std::size_t>(row_stride) * num_atoms);
    std::uint8_t queue[max_atoms_per_molecule_k];
    for (std::uint32_t source = 0; source < num_atoms; ++source) {
        std::uint8_t *const row = matrix + static_cast<std::size_t>(source) * row_stride;
        row[source] = 0;
        std::uint32_t head = 0;
        std::uint32_t tail = 0;
        queue[tail++] = static_cast<std::uint8_t>(source);
        while (head < tail) {
            std::uint8_t const current = queue[head++];
            std::uint8_t const next_distance = static_cast<std::uint8_t>(row[current] + 1);
            std::uint16_t const start = descriptor.neighbor_offset[current];
            std::uint16_t const end = descriptor.neighbor_offset[current + 1];
            for (std::uint16_t entry = start; entry < end; ++entry) {
                std::uint8_t const neighbor = descriptor.neighbor_atom[entry];
                if (row[neighbor] > next_distance) {
                    row[neighbor] = next_distance;
                    queue[tail++] = neighbor;
                }
            }
        }
    }
}

/** True iff atom @p atom_i and atom @p atom_j are in the same SSSR ring. */
inline bool share_ring(molecule_descriptor_t const &descriptor, std::uint32_t atom_i, std::uint32_t atom_j) {
    return (descriptor.atom_ring_bitmap[atom_i] & descriptor.atom_ring_bitmap[atom_j]) != 0;
}

} // namespace smallfield::types
