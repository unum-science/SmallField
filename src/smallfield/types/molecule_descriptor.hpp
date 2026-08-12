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

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`
#include <cstring> // `std::memset`, `std::memcpy`

#include <algorithm> // `std::min`

#include <GraphMol/GraphMol.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/MolOps.h>

#include "smallfield/api/conformer_status.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::types {

/** Maximum atom count supported by the single-warp GPU pipeline. */
inline constexpr std::uint32_t max_atoms_per_molecule_k = 192;

/** Maximum bond count. A connected molecule at the 192-atom cap carries 191 bonds plus one per ring
 *  closure, so 384 covers the widest molecule the 32-ring cap admits. */
inline constexpr std::uint32_t max_bonds_k = 384;

/** Maximum number of neighbors stored across all atoms (adjacency CSR size). */
inline constexpr std::uint32_t max_neighbor_entries_k = max_bonds_k * 2;

/** Maximum SSSR rings supported. Drug-like molecules rarely exceed 8, and `atom_ring_bitmap` is
 *  one `std::uint32_t` per atom, so this is the width of that word rather than a tunable. */
inline constexpr std::uint32_t max_rings_k = 32;

/** Maximum ring-membership entries across all rings, the size of the concatenated `ring_atoms`. */
inline constexpr std::uint32_t max_ring_atom_entries_k = max_atoms_per_molecule_k * 2;

/**
 *  @brief Bond-hop count for a pair with no bonded path — the two halves of a salt, a solvate's water.
 *
 *  Also the saturation value, and unreachable as a real distance because the longest simple path across
 *  `max_atoms_per_molecule_k` atoms is 191 hops.
 */
inline constexpr std::uint8_t unreachable_hops_k = 255u;

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
    /** `Bond::getIsAromatic()`. A bond that sits in an aromatic ring carries this even when it keeps a
     *  Kekulé order, which is how an in-ring triple bond reaches the descriptor as order 3 and aromatic. */
    bond_flag_aromatic_k = 1u << 0,
    bond_flag_in_ring_k = 1u << 1,
    bond_flag_conjugated_k = 1u << 2,
    /** E / trans. */
    bond_flag_stereo_e_k = 1u << 3,
    /** Z / cis. */
    bond_flag_stereo_z_k = 1u << 4,
    /** `Bond::getBondType() == Bond::AROMATIC`: the bond has no Kekulé order at all. Strictly narrower
     *  than `bond_flag_aromatic_k`, and the flag RDKit's MMFF ring queries test. */
    bond_flag_order_aromatic_k = 1u << 5,
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
    /** Explicit bond-order sum plus still-implicit hydrogens, exactly RDKit's
     *  `getValence(EXPLICIT) + getNumImplicitHs()`. Not recoverable from `bond_order`: RDKit counts an
     *  aromatic bond as 1.5 and then clamps an aromatic atom's sum down to the element's nearest
     *  permitted valence, so pyrrole's nitrogen reads 3 rather than 4. */
    std::uint8_t total_valence[max_atoms_per_molecule_k];
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
    /** 1, 2, or 3. A bond RDKit types `Bond::AROMATIC` has no Kekulé order to report, so it reports 1 and
     *  carries `bond_flag_order_aromatic_k`; ask for that flag, never for `bond_order == 1` alone, when the
     *  question is whether the bond is aromatic. */
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
    std::uint8_t ring_atoms[max_ring_atom_entries_k];
    /** Bit `ring_index` of an atom is set when that atom belongs to that ring. */
    std::uint32_t atom_ring_bitmap[max_atoms_per_molecule_k];
};

#pragma region Construction

/**
 *  @brief Whether @p molecule fits every extent the flat descriptor has, and which one it trips.
 *
 *  Every refusal is checked before a single field is written, so a descriptor is either complete or
 *  untouched. The ring extents are part of that: past either one a genuinely in-ring atom would
 *  carry no ring label, and MMFF typing and the CSD torsion match would read the molecule as
 *  acyclic where it is not.
 *
 *  @param[in] molecule Sanitized, hydrogen-explicit molecule.
 *  @return `success_k`, or the extent the molecule exceeds.
 */
inline api::conformer_status_t descriptor_capacity_status_(RDKit::ROMol const &molecule) {
    if (molecule.getNumAtoms() > max_atoms_per_molecule_k) return api::conformer_status_t::too_many_atoms_k;
    if (molecule.getNumBonds() > max_bonds_k) return api::conformer_status_t::too_many_bonds_k;

    RDKit::RingInfo const *const ring_info = molecule.getRingInfo();
    if (ring_info == nullptr || !ring_info->isInitialized()) return api::conformer_status_t::success_k;
    auto const &atom_rings = ring_info->atomRings();
    if (atom_rings.size() > max_rings_k) return api::conformer_status_t::too_many_rings_k;
    std::size_t ring_atom_entries = 0;
    for (auto const &ring : atom_rings) ring_atom_entries += ring.size();
    if (ring_atom_entries > max_ring_atom_entries_k) return api::conformer_status_t::too_many_rings_k;
    return api::conformer_status_t::success_k;
}

/**
 *  @brief Fill every per-atom array of @p descriptor from @p molecule's atoms.
 *
 *  @param[in] molecule Source molecule, already known to fit `max_atoms_per_molecule_k`.
 *  @param[out] descriptor Descriptor whose `num_atoms` is already set.
 */
inline void fill_atoms_(RDKit::ROMol const &molecule, molecule_descriptor_t &descriptor) {
    // Padding entries past `num_atoms` are zeroed here rather than left to the loop, so a device
    // read of a whole warp's worth of lanes sees a stable value.
    std::memset(descriptor.atom_flags, 0, sizeof(descriptor.atom_flags));

    std::uint32_t const num_atoms = descriptor.num_atoms;
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        RDKit::Atom const *const atom = molecule.getAtomWithIdx(atom_index);
        int const element = atom->getAtomicNum();
        descriptor.element[atom_index] = static_cast<std::uint8_t>(element);
        descriptor.hybridization[atom_index] = static_cast<std::uint8_t>(atom->getHybridization());
        descriptor.degree[atom_index] = static_cast<std::uint8_t>(atom->getDegree());
        descriptor.num_hydrogens[atom_index] = static_cast<std::uint8_t>(atom->getTotalNumHs(true));
        descriptor.total_valence[atom_index] = static_cast<std::uint8_t>(
            atom->getValence(RDKit::Atom::ValenceType::EXPLICIT) + atom->getNumImplicitHs());
        descriptor.formal_charge[atom_index] = static_cast<std::int8_t>(atom->getFormalCharge());
        descriptor.isotope[atom_index] = static_cast<std::uint8_t>(std::min<unsigned>(atom->getIsotope(), 255));
        descriptor.smallest_ring[atom_index] = 0;
        descriptor.cip_rank[atom_index] = 0;

        std::uint8_t flags = 0;
        if (atom->getIsAromatic()) flags |= atom_flag_aromatic_k;

        // H-bond acceptor: N or O. Simplified — RDKit's full definition also checks lone pair
        // availability via charge and hybridization — but this covers drug-like cases.
        if (element == 7 || element == 8) flags |= atom_flag_hbond_acceptor_k;

        // An H covalently bound to N or O donates. RDKit's own neighbour iteration answers this
        // without the CSR adjacency, which is not built yet.
        if (element == 1)
            for (RDKit::Atom const *const neighbor : molecule.atomNeighbors(atom)) {
                int const neighbor_element = neighbor->getAtomicNum();
                if (neighbor_element != 7 && neighbor_element != 8) continue;
                flags |= atom_flag_hbond_donor_h_k;
                break;
            }

        RDKit::Atom::ChiralType const chiral = atom->getChiralTag();
        std::uint8_t const chiral_tag = chiral == RDKit::Atom::CHI_TETRAHEDRAL_CW    ? std::uint8_t {1}
                                        : chiral == RDKit::Atom::CHI_TETRAHEDRAL_CCW ? std::uint8_t {2}
                                                                                     : std::uint8_t {0};
        descriptor.chiral_tag[atom_index] = chiral_tag;
        if (chiral_tag != 0) flags |= atom_flag_has_chirality_k;
        descriptor.atom_flags[atom_index] = flags;
    }
}

/**
 *  @brief Fill every per-bond array of @p descriptor from @p molecule's bonds.
 *
 *  @param[in] molecule Source molecule, already known to fit `max_bonds_k`.
 *  @param[out] descriptor Descriptor whose `num_atoms` and `num_bonds` are already set.
 */
inline void fill_bonds_(RDKit::ROMol const &molecule, molecule_descriptor_t &descriptor) {
    std::memset(descriptor.bond_flags, 0, sizeof(descriptor.bond_flags));

    std::uint32_t const num_atoms = descriptor.num_atoms;
    std::uint32_t const num_bonds = descriptor.num_bonds;
    for (std::uint32_t bond_index = 0; bond_index < num_bonds; ++bond_index) {
        RDKit::Bond const *const bond = molecule.getBondWithIdx(bond_index);
        descriptor.bond_atom_i[bond_index] = static_cast<std::uint8_t>(bond->getBeginAtomIdx());
        descriptor.bond_atom_j[bond_index] = static_cast<std::uint8_t>(bond->getEndAtomIdx());

        // An aromatic bond has no integer order; it reports 1 for topology and announces itself
        // through `bond_flag_order_aromatic_k`.
        RDKit::Bond::BondType const order = bond->getBondType();
        descriptor.bond_order[bond_index] = order == RDKit::Bond::DOUBLE   ? std::uint8_t {2}
                                            : order == RDKit::Bond::TRIPLE ? std::uint8_t {3}
                                                                           : std::uint8_t {1};

        std::uint8_t flags = 0;
        if (bond->getIsAromatic()) flags |= bond_flag_aromatic_k;
        if (order == RDKit::Bond::AROMATIC) flags |= bond_flag_order_aromatic_k;
        if (bond->getIsConjugated()) flags |= bond_flag_conjugated_k;
        RDKit::Bond::BondStereo const stereo = bond->getStereo();
        if (stereo == RDKit::Bond::STEREOE || stereo == RDKit::Bond::STEREOTRANS) flags |= bond_flag_stereo_e_k;
        if (stereo == RDKit::Bond::STEREOZ || stereo == RDKit::Bond::STEREOCIS) flags |= bond_flag_stereo_z_k;
        descriptor.bond_flags[bond_index] = flags;

        // The E/Z assignment is stated relative to a specific pair of substituents, one per
        // double-bond end. Record them so the bounds builder can tell which side any given 1-4
        // partner lands on.
        descriptor.bond_stereo_atom_i[bond_index] = no_stereo_atom_k;
        descriptor.bond_stereo_atom_j[bond_index] = no_stereo_atom_k;
        if ((flags & (bond_flag_stereo_e_k | bond_flag_stereo_z_k)) == 0) continue;
        auto const &stereo_atoms = bond->getStereoAtoms();
        if (stereo_atoms.size() != 2) continue;
        std::uint32_t const first = static_cast<std::uint32_t>(stereo_atoms[0]);
        std::uint32_t const second = static_cast<std::uint32_t>(stereo_atoms[1]);
        if (first >= num_atoms || second >= num_atoms) continue;
        descriptor.bond_stereo_atom_i[bond_index] = static_cast<std::uint8_t>(first);
        descriptor.bond_stereo_atom_j[bond_index] = static_cast<std::uint8_t>(second);
    }
}

/**
 *  @brief Build the CSR adjacency of @p descriptor from its own bond arrays.
 *
 *  A pure function of `bond_atom_i` and `bond_atom_j`, so it reads no RDKit state: count the
 *  neighbours per atom, prefix-sum them into `neighbor_offset`, then walk the bonds again writing
 *  both endpoints.
 *
 *  @param[inout] descriptor Descriptor whose bond arrays are filled and whose CSR is not.
 */
inline void fill_adjacency_(molecule_descriptor_t &descriptor) {
    std::uint32_t const num_atoms = descriptor.num_atoms;
    std::uint32_t const num_bonds = descriptor.num_bonds;

    std::uint16_t cursor[max_atoms_per_molecule_k + 1] = {};
    for (std::uint32_t bond_index = 0; bond_index < num_bonds; ++bond_index) {
        cursor[descriptor.bond_atom_i[bond_index] + 1]++;
        cursor[descriptor.bond_atom_j[bond_index] + 1]++;
    }
    descriptor.neighbor_offset[0] = 0;
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index)
        descriptor.neighbor_offset[atom_index + 1] = descriptor.neighbor_offset[atom_index] + cursor[atom_index + 1];

    // The cursor now holds each atom's write position rather than its count.
    std::memcpy(cursor, descriptor.neighbor_offset, sizeof(std::uint16_t) * (num_atoms + 1));
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
}

/**
 *  @brief Label the rings of @p molecule into @p descriptor, on the atoms and on the bonds.
 *
 *  `descriptor_capacity_status_` has already refused anything that would not fit, so every ring
 *  RDKit perceived is written whole. A molecule whose ring information was never initialized
 *  carries no rings at all rather than a partial set.
 *
 *  @param[in] molecule Source molecule.
 *  @param[inout] descriptor Descriptor whose atom and bond arrays are filled.
 */
inline void fill_rings_(RDKit::ROMol const &molecule, molecule_descriptor_t &descriptor) {
    std::memset(descriptor.ring_bond_degree, 0, sizeof(descriptor.ring_bond_degree));
    std::memset(descriptor.atom_ring_bitmap, 0, sizeof(descriptor.atom_ring_bitmap));
    descriptor.num_rings = 0;
    descriptor.ring_offset[0] = 0;

    RDKit::RingInfo const *const ring_info = molecule.getRingInfo();
    if (ring_info == nullptr || !ring_info->isInitialized()) return;

    std::uint32_t const num_bonds = descriptor.num_bonds;
    for (std::uint32_t bond_index = 0; bond_index < num_bonds; ++bond_index) {
        if (ring_info->numBondRings(bond_index) == 0) continue;
        descriptor.bond_flags[bond_index] |= bond_flag_in_ring_k;
        descriptor.ring_bond_degree[descriptor.bond_atom_i[bond_index]]++;
        descriptor.ring_bond_degree[descriptor.bond_atom_j[bond_index]]++;
    }

    auto const &atom_rings = ring_info->atomRings();
    std::uint32_t const num_rings = static_cast<std::uint32_t>(atom_rings.size());
    descriptor.num_rings = static_cast<std::uint8_t>(num_rings);
    std::uint16_t ring_cursor = 0;
    for (std::uint32_t ring_index = 0; ring_index < num_rings; ++ring_index) {
        auto const &ring = atom_rings[ring_index];
        std::uint8_t const ring_size = static_cast<std::uint8_t>(ring.size());
        for (int const member : ring) {
            std::uint32_t const atom_index = static_cast<std::uint32_t>(member);
            descriptor.ring_atoms[ring_cursor++] = static_cast<std::uint8_t>(atom_index);
            descriptor.atom_ring_bitmap[atom_index] |= 1u << ring_index;
            descriptor.atom_flags[atom_index] |= atom_flag_in_ring_k;
            if (descriptor.smallest_ring[atom_index] == 0 || ring_size < descriptor.smallest_ring[atom_index])
                descriptor.smallest_ring[atom_index] = ring_size;
        }
        descriptor.ring_offset[ring_index + 1] = ring_cursor;
    }
}

/**
 *  @brief Build a flat descriptor from an RDKit molecule.
 *
 *  Reads every field needed for downstream preprocessing, so the `ROMol` can be discarded
 *  afterwards. Either every extent fits and the descriptor is complete, or nothing is written and
 *  the refused extent is named — a caller never holds a partially populated descriptor.
 *
 *  Preconditions:
 *   - Hydrogens added through `MolOps::addHs`
 *   - Sanitized — `SmilesToMol` with `sanitize=true`, its default
 *
 *  @param[in] molecule RDKit molecule.
 *  @param[out] descriptor Descriptor to fill. Any prior contents overwritten.
 *  @return `success_k`, or `too_many_atoms_k`, `too_many_bonds_k` or `too_many_rings_k` for the
 *          extent the molecule exceeds. The reason is the same vocabulary the pipeline reports
 *          per conformer, so a caller forwards it rather than translating it.
 */
inline api::conformer_status_t build_descriptor(RDKit::ROMol const &molecule, molecule_descriptor_t &descriptor) {
    api::conformer_status_t const capacity = descriptor_capacity_status_(molecule);
    if (capacity != api::conformer_status_t::success_k) return capacity;

    descriptor.num_atoms = static_cast<std::uint8_t>(molecule.getNumAtoms());
    descriptor.num_bonds = static_cast<std::uint16_t>(molecule.getNumBonds());
    fill_atoms_(molecule, descriptor);
    fill_bonds_(molecule, descriptor);
    fill_adjacency_(descriptor);
    fill_rings_(molecule, descriptor);
    return api::conformer_status_t::success_k;
}

#pragma endregion Construction

#pragma region Accessors

/** True iff atom @p atom_index carries @p flag. */
inline bool atom_has_flag(molecule_descriptor_t const &descriptor, std::uint32_t atom_index, atom_flag_t flag) {
    return (descriptor.atom_flags[atom_index] & flag) != 0;
}

/** True iff bond @p bond_index carries @p flag. */
inline bool bond_has_flag(molecule_descriptor_t const &descriptor, std::uint32_t bond_index, bond_flag_t flag) {
    return (descriptor.bond_flags[bond_index] & flag) != 0;
}

/** True iff atom @p atom_i and atom @p atom_j are in the same SSSR ring. */
inline bool share_ring(molecule_descriptor_t const &descriptor, std::uint32_t atom_i, std::uint32_t atom_j) {
    return (descriptor.atom_ring_bitmap[atom_i] & descriptor.atom_ring_bitmap[atom_j]) != 0;
}

/**
 *  @brief Fill @p matrix with the all-pairs bond-hop counts of @p descriptor.
 *
 *  One breadth-first search per source atom over the CSR adjacency. Atoms with no bonded path
 *  between them — the two halves of a salt, a solvate's water — keep `unreachable_hops_k`, which
 *  the MMFF nonbonded sweep reads as "unreachable" and which no real distance reaches, since the
 *  longest simple path across `max_atoms_per_molecule_k` atoms is 191 hops.
 *
 *  @param[in] descriptor Descriptor supplying the atom count and the CSR adjacency.
 *  @param[out] matrix Square view over caller-owned storage, `descriptor.num_atoms` on both extents.
 */
inline void fill_bond_distances(molecule_descriptor_t const &descriptor, hop_matrix_t matrix) {
    std::uint32_t const num_atoms = descriptor.num_atoms;
    sf_assert_(matrix.extent(0) == num_atoms && matrix.extent(1) == num_atoms);
    std::memset(matrix.data_handle(), unreachable_hops_k, static_cast<std::size_t>(num_atoms) * num_atoms);

    std::uint8_t queue[max_atoms_per_molecule_k];
    for (std::uint32_t source = 0; source < num_atoms; ++source) {
        matrix(source, source) = 0;
        std::uint32_t head = 0;
        std::uint32_t tail = 0;
        queue[tail++] = static_cast<std::uint8_t>(source);
        while (head < tail) {
            std::uint8_t const current = queue[head++];
            std::uint8_t const next_distance = static_cast<std::uint8_t>(matrix(source, current) + 1);
            std::uint16_t const start = descriptor.neighbor_offset[current];
            std::uint16_t const end = descriptor.neighbor_offset[current + 1];
            for (std::uint16_t entry = start; entry < end; ++entry) {
                std::uint8_t const neighbor = descriptor.neighbor_atom[entry];
                if (matrix(source, neighbor) <= next_distance) continue;
                matrix(source, neighbor) = next_distance;
                queue[tail++] = neighbor;
            }
        }
    }
}

#pragma endregion Accessors

} // namespace smallfield::types
