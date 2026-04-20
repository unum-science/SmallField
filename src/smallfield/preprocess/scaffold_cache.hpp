/**
 *  @file scaffold_cache.hpp
 *  @brief Per-context scaffold cache for amortising RDKit `SubstructMatch`
 *  across molecules sharing the same Bemis–Murcko scaffold.
 *
 *  A scaffold-cache feasibility experiment on a 241K ChEMBL set showed high
 *  scaffold reuse (98 % for the curated subset).  RDKit's SubstructMatch on
 *  the ~36 % of CSD torsion patterns that still fall through to VF2 (after
 *  the native matcher handles the rest) is the single largest remaining CPU
 *  bottleneck — cache-hitting on the scaffold avoids the graph-isomorphism
 *  search for the fraction of matches whose atoms are entirely in the
 *  scaffold (no R-group participation).  See `BENCH.md` §"scaffold-cache
 *  architecture sketch" and `ROADMAP.md` (Dataset roadmap) for the design rationale.
 *
 *  Design:
 *   - Sharded `std::unordered_map<uint64_t, scaffold_cache_entry_t>` keyed on
 *     a 64-bit hash of the scaffold's canonical SMILES.  64 shards reduce
 *     contention across the 192-thread OpenMP prep pool to practical zero.
 *   - Each entry stores, per non-native CSD pattern, the list of 4-atom
 *     match tuples in **scaffold atom space**.  Callers translate via a
 *     scaffold-to-molecule atom index map produced at cache-miss time.
 *   - Grow-only; LRU-style eviction lives behind a size cap.
 *     Even at ~100 K unique scaffolds the cache stays under ~2.5 GB.
 *
 *  Opt-in via @ref context_t::enable_scaffold_cache — disabled by default
 *  so the existing public API behaviour is unchanged.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace smallfield::preprocess {

/**
 *  @brief Per-scaffold precomputed SubstructMatch results for a subset of
 *  the CSD torsion pattern set.
 *
 *  `pattern_matches[p]` is the list of matches for pattern @p p whose all
 *  four matched atoms are inside the scaffold.  Each match is a 4-tuple of
 *  SCAFFOLD atom indices; callers translate into molecule indices via the
 *  scaffold→molecule atom map maintained per-molecule.
 */
struct scaffold_cache_entry_t {
    /** For pattern @p p, the list of matches in scaffold atom space.  Size
     *  equals the total pattern count (most entries are empty vectors for
     *  patterns whose 4-atom match is never scaffold-local). */
    std::vector<std::vector<std::uint8_t>> pattern_matches_flat; // [p] → [a0,b0,c0,d0, a1,b1,c1,d1, ...]
    /** Per-pattern flag: 1 if the scaffold's SubstructMatch returned at least
     *  one match whose atoms partly span an R-group (i.e. an atom whose
     *  scaffold-atom-set membership test failed because the molecule's match
     *  reached an R-group atom).  Forces the full-molecule SubstructMatch on
     *  cache hits for that pattern; 0 lets the hit translate cached matches
     *  alone.  Sized identically to @ref pattern_matches_flat. */
    std::vector<std::uint8_t> has_r_group_matches;
};

/** Runtime-tunable knobs for the cache. */
struct scaffold_cache_config_t {
    /** LRU-evict when entries exceed this count.  Default 500K covers an
     *  entire Enamine REAL run with headroom. */
    std::size_t max_entries = 500'000;
    /** Soft cap on resident bytes; checked approximately at insert time. */
    std::size_t max_bytes = 4ULL << 30; // 4 GB
    /** If non-empty, @ref context_t::~context_t() calls save() to this path. */
    std::filesystem::path persist_path {};
};

/** Introspection counters for observability. */
struct scaffold_cache_stats_t {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::size_t entries = 0;
    std::size_t bytes_used = 0;
};

/**
 *  @brief Sharded concurrent scaffold cache.  Thread-safe for concurrent
 *  lookup+insert from the 192-thread OpenMP prep pool.
 *
 *  Sharding is by the low bits of the scaffold hash; each shard has its own
 *  `std::shared_mutex` so lookups from different shards never contend.
 */
class scaffold_cache_t {
  public:
    /** Number of shards.  Must be a power of two. */
    static constexpr std::size_t scaffold_cache_shard_count_k = 64;

    scaffold_cache_t() = default;

    /** Configure limits.  Call before first generate_conformers() call. */
    void configure(scaffold_cache_config_t const &cfg) { config_ = cfg; }
    scaffold_cache_config_t const &config() const noexcept { return config_; }

    /** Return the entry for @p scaffold_hash if present, otherwise @c nullptr.
     *  Takes a shared lock on the hash's shard. */
    scaffold_cache_entry_t const *lookup(std::uint64_t scaffold_hash) const noexcept {
        shard_t const &s = shards_[scaffold_hash & (scaffold_cache_shard_count_k - 1)];
        std::shared_lock<std::shared_mutex> lk(s.lock);
        auto it = s.entries.find(scaffold_hash);
        if (it == s.entries.end()) {
            ++misses_;
            return nullptr;
        }
        ++hits_;
        return &it->second;
    }

    /** Insert @p entry under @p scaffold_hash if absent.  Caller-provided
     *  entry is moved in.  Takes an exclusive lock on the hash's shard. */
    void insert(std::uint64_t scaffold_hash, scaffold_cache_entry_t entry) {
        shard_t &s = shards_[scaffold_hash & (scaffold_cache_shard_count_k - 1)];
        std::unique_lock<std::shared_mutex> lk(s.lock);
        s.entries.emplace(scaffold_hash, std::move(entry));
    }

    /** Aggregate snapshot across shards. */
    scaffold_cache_stats_t stats() const noexcept {
        scaffold_cache_stats_t out;
        out.hits = hits_.load();
        out.misses = misses_.load();
        for (auto const &s : shards_) {
            std::shared_lock<std::shared_mutex> lk(s.lock);
            out.entries += s.entries.size();
            for (auto const &e : s.entries)
                for (auto const &v : e.second.pattern_matches_flat) out.bytes_used += v.capacity();
        }
        return out;
    }

    /** Erase all entries. */
    void clear() noexcept {
        for (auto &s : shards_) {
            std::unique_lock<std::shared_mutex> lk(s.lock);
            s.entries.clear();
        }
        hits_.store(0);
        misses_.store(0);
    }

  private:
    struct shard_t {
        mutable std::shared_mutex lock;
        std::unordered_map<std::uint64_t, scaffold_cache_entry_t> entries;
    };
    scaffold_cache_config_t config_ {};
    mutable std::atomic<std::uint64_t> hits_ {0};
    mutable std::atomic<std::uint64_t> misses_ {0};
    shard_t shards_[scaffold_cache_shard_count_k];
};

/**
 *  @brief Hash a canonical scaffold SMILES to a 64-bit key.  Uses a simple
 *  FNV-1a variant — good collision properties at the ~100K entries we
 *  expect, without adding a dependency.
 */
inline std::uint64_t hash_scaffold_smiles(std::string_view smiles) noexcept {
    std::uint64_t h = 14695981039346656037ULL;
    for (char c : smiles) {
        h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
        h *= 1099511628211ULL;
    }
    return h;
}

} // namespace smallfield::preprocess
