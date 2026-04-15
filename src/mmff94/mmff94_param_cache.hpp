/**
 *  @file mmff94_param_cache.hpp
 *  @brief Global cache for MMFF94 per-term parameter lookups.
 *
 *  Once the native MMFF port replaced the RDKit parameter calls, `extract_mmff94_params` still
 *  accounted for 23 % of prep CPU. The bulk is in
 *  `getMMFFTorsionParams` (6.2 %), `getMMFFStretchBendParams` (5.3 %),
 *  `getMMFFAngleBendParams` (3.7 %), and ancillary type-determination
 *  (`getMMFFBondType`, `getMMFFAngleType`, `getMMFFTorsionType`).  The
 *  `MMFFMolProperties` constructor itself is only 2.6 %.
 *
 *  Key observation: these per-term calls are pure functions of the atom
 *  types + bond classifications.  MMFF94 has ~92 atom types and a handful
 *  of bond-type codes, so the effective tuple space is small (a few
 *  thousand unique keys across drug-like ChEMBL) and warms up within the
 *  first hundred molecules.  From then on, each call becomes an O(1) hash
 *  lookup instead of an O(log N) std::map search through RDKit's global
 *  parameter tables plus the type-determination work.
 *
 *  The cache is keyed on the RDKit-assigned MMFF atom types (obtained
 *  cheaply via `mmff_props.getMMFFAtomType(atom_idx)` — array access) plus
 *  bond order / aromaticity / ring flags of the relevant bonds.  This is
 *  the same information RDKit's internal classification uses, so a cache
 *  hit is byte-identical to calling the RDKit function.
 *
 *  Thread-safety: 64 shards, each guarded by `std::shared_mutex`.  Lookups
 *  take shared locks (no contention between readers); inserts take
 *  exclusive locks (rare after warmup).
 */

#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>

namespace smallfield {

/** Per-bond stretch parameter (MMFFBond in RDKit). */
struct mmff_bond_cached_t {
    float kb;
    float r0;
};

/** Per-angle bend parameter (MMFFAngle.ka + theta0 pre-converted to radians). */
struct mmff_angle_cached_t {
    float ka;
    float theta0_rad;
};

/** Per-torsion parameter (MMFFTor V1/V2/V3). */
struct mmff_torsion_cached_t {
    float V1;
    float V2;
    float V3;
};

/** Per-OOP parameter (MMFFOop koop). */
struct mmff_oop_cached_t {
    float koop;
};

/** Per-stretch-bend parameter: couples a 3-atom angle with its two bonds.
 *  The four floats we actually consume downstream (see `mmff_stretchbends`
 *  assembly in `extract_mmff94_params`). */
struct mmff_stretchbend_cached_t {
    float kbaIJK;     ///< stretch-bend force const (bond IJ side)
    float kbaKJI;     ///< stretch-bend force const (bond KJ side)
    float theta0_rad; ///< equilibrium angle (from inner MMFFAngle)
    float r0_ij;      ///< equilibrium bond length IJ (from inner MMFFBond[0])
    float r0_jk;      ///< equilibrium bond length JK (from inner MMFFBond[1])
};

/**
 *  Thread-local cache.  Each OpenMP thread owns its own `std::unordered_map`;
 *  no mutexes.  An initial attempt used sharded `std::shared_mutex` for
 *  cross-thread sharing, but with 192 threads × ~30 lookups per molecule
 *  × 241 K molecules, `pthread_rwlock_rdlock`/`unlock` alone consumed 46 %
 *  of CPU — the atomic CAS overhead dominated the work being protected.
 *
 *  Thread-local eliminates lock contention entirely.  Memory cost: each
 *  thread holds its own copy of the entries it has seen (~5 K keys for
 *  drug-like ChEMBL × 192 threads × ~24 bytes ≈ 23 MB total).  On a 1.7 TB
 *  RAM machine this is negligible; the throughput win dwarfs the duplication.
 */
template <typename Value>
class mmff_cache_t {
public:
    Value const *find(std::uint64_t key) const noexcept {
        auto it = map_.find(key);
        return (it == map_.end()) ? nullptr : &it->second;
    }

    void insert(std::uint64_t key, Value const &value) { map_.try_emplace(key, value); }

    std::size_t size() const noexcept { return map_.size(); }

private:
    std::unordered_map<std::uint64_t, Value> map_;
};

// Per-thread cache instances. All MMFF94 param tables are globally fixed, so deriving once per thread
// is safe. Caches are process-local and thread-local and persist for the thread's
// lifetime, so warmup cost amortises across every molecule that thread
// handles via OpenMP's persistent pool.

inline mmff_cache_t<mmff_bond_cached_t> &mmff_bond_cache() {
    thread_local mmff_cache_t<mmff_bond_cached_t> c;
    return c;
}
inline mmff_cache_t<mmff_angle_cached_t> &mmff_angle_cache() {
    thread_local mmff_cache_t<mmff_angle_cached_t> c;
    return c;
}
inline mmff_cache_t<mmff_torsion_cached_t> &mmff_torsion_cache() {
    thread_local mmff_cache_t<mmff_torsion_cached_t> c;
    return c;
}
inline mmff_cache_t<mmff_oop_cached_t> &mmff_oop_cache() {
    thread_local mmff_cache_t<mmff_oop_cached_t> c;
    return c;
}
inline mmff_cache_t<mmff_stretchbend_cached_t> &mmff_stretchbend_cache() {
    thread_local mmff_cache_t<mmff_stretchbend_cached_t> c;
    return c;
}

// Key packing. MMFF atom types fit in 7 bits — the maximum type number is 99. Bond signatures pack
// into 4 bits: 2 bits for bond_order, 1 bit for aromatic, 1 bit for in_ring. All keys
// fit comfortably in 64 bits with room for a 4-bit "kind" tag.

enum : std::uint64_t {
    mmff_key_kind_bond_stretch_k = 0x1,
    mmff_key_kind_angle_bend_k   = 0x2,
    mmff_key_kind_torsion_k      = 0x3,
    mmff_key_kind_oop_k          = 0x4,
    mmff_key_kind_stretchbend_k  = 0x5,
};

inline std::uint8_t pack_bond_signature_(std::uint8_t bond_order, bool aromatic, bool in_ring) noexcept {
    return static_cast<std::uint8_t>((bond_order & 0x3) | ((aromatic ? 1u : 0u) << 2) | ((in_ring ? 1u : 0u) << 3));
}

inline std::uint64_t make_bond_stretch_key_(std::uint8_t at_i, std::uint8_t at_j,
                                            std::uint8_t bond_sig) noexcept {
    // Commutative under (i, j) swap.
    if (at_i > at_j) std::swap(at_i, at_j);
    return (mmff_key_kind_bond_stretch_k << 60) | (static_cast<std::uint64_t>(at_i) << 16) |
           (static_cast<std::uint64_t>(at_j) << 8) | bond_sig;
}

inline std::uint64_t make_angle_key_(std::uint8_t at_i, std::uint8_t at_center, std::uint8_t at_j,
                                     std::uint8_t bsig_ic, std::uint8_t bsig_cj) noexcept {
    // Commutative under (i, j)↔(j, i) with paired bond-signature swap.
    if (at_i > at_j || (at_i == at_j && bsig_ic > bsig_cj)) {
        std::swap(at_i, at_j);
        std::swap(bsig_ic, bsig_cj);
    }
    return (mmff_key_kind_angle_bend_k << 60) | (static_cast<std::uint64_t>(at_i) << 32) |
           (static_cast<std::uint64_t>(at_center) << 24) | (static_cast<std::uint64_t>(at_j) << 16) |
           (static_cast<std::uint64_t>(bsig_ic) << 8) | bsig_cj;
}

inline std::uint64_t make_torsion_key_(std::uint8_t at_i, std::uint8_t at_j, std::uint8_t at_k,
                                       std::uint8_t at_l, std::uint8_t bsig_ij, std::uint8_t bsig_jk,
                                       std::uint8_t bsig_kl) noexcept {
    // Torsion params are symmetric under (i,j,k,l) ↔ (l,k,j,i). Canonicalize
    // by comparing the two endpoint-type pairs; flip the whole thing if the
    // reverse ordering is smaller.
    bool flip = false;
    if (at_j > at_k) flip = true;
    else if (at_j == at_k && at_i > at_l) flip = true;
    else if (at_j == at_k && at_i == at_l && bsig_ij > bsig_kl) flip = true;
    if (flip) {
        std::swap(at_i, at_l);
        std::swap(at_j, at_k);
        std::swap(bsig_ij, bsig_kl);
    }
    return (mmff_key_kind_torsion_k << 60) | (static_cast<std::uint64_t>(at_i) << 48) |
           (static_cast<std::uint64_t>(at_j) << 40) | (static_cast<std::uint64_t>(at_k) << 32) |
           (static_cast<std::uint64_t>(at_l) << 24) | (static_cast<std::uint64_t>(bsig_ij) << 16) |
           (static_cast<std::uint64_t>(bsig_jk) << 8) | bsig_kl;
}

inline std::uint64_t make_oop_key_(std::uint8_t at_center, std::uint8_t at_a, std::uint8_t at_b,
                                   std::uint8_t at_c) noexcept {
    // Out-of-plane: center is fixed, the three planar neighbors are
    // interchangeable. Sort (a, b, c) ascending.
    std::uint8_t nbrs[3] = {at_a, at_b, at_c};
    if (nbrs[0] > nbrs[1]) std::swap(nbrs[0], nbrs[1]);
    if (nbrs[1] > nbrs[2]) std::swap(nbrs[1], nbrs[2]);
    if (nbrs[0] > nbrs[1]) std::swap(nbrs[0], nbrs[1]);
    return (mmff_key_kind_oop_k << 60) | (static_cast<std::uint64_t>(at_center) << 32) |
           (static_cast<std::uint64_t>(nbrs[0]) << 24) | (static_cast<std::uint64_t>(nbrs[1]) << 16) |
           (static_cast<std::uint64_t>(nbrs[2]) << 8);
}

inline std::uint64_t make_stretchbend_key_(std::uint8_t at_i, std::uint8_t at_center, std::uint8_t at_j,
                                           std::uint8_t bsig_ic, std::uint8_t bsig_cj) noexcept {
    // Stretch-bend is directional (kbaIJK ≠ kbaKJI in general). Do NOT
    // canonicalize endpoint order.
    return (mmff_key_kind_stretchbend_k << 60) | (static_cast<std::uint64_t>(at_i) << 32) |
           (static_cast<std::uint64_t>(at_center) << 24) | (static_cast<std::uint64_t>(at_j) << 16) |
           (static_cast<std::uint64_t>(bsig_ic) << 8) | bsig_cj;
}

/** Snapshot of current cache occupancy across all five per-term caches. */
struct mmff_cache_stats_t {
    std::size_t bond_entries = 0;
    std::size_t angle_entries = 0;
    std::size_t torsion_entries = 0;
    std::size_t oop_entries = 0;
    std::size_t stretchbend_entries = 0;
};

inline mmff_cache_stats_t get_mmff_cache_stats() {
    mmff_cache_stats_t out;
    out.bond_entries = mmff_bond_cache().size();
    out.angle_entries = mmff_angle_cache().size();
    out.torsion_entries = mmff_torsion_cache().size();
    out.oop_entries = mmff_oop_cache().size();
    out.stretchbend_entries = mmff_stretchbend_cache().size();
    return out;
}

} // namespace smallfield
