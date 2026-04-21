/**
 *  @file csd_torsion_dispatcher.hpp
 *  @brief CSD torsion extraction with element prefilter, native matcher, scaffold cache.
 *
 *  Replaces `ForceFields::CrystalFF::getExperimentalTorsions()` with an owned
 *  matching loop.  The CSD SMARTS patterns are loaded once at static-init, each
 *  is compiled to an `ROMol` via `SmartsToMol`, and a parallel native form is
 *  extracted by @ref parse_linear_torsion (see `csd_torsion_native.hpp`).  For
 *  every molecule the loop:
 *
 *   1. Element prefilter — skip patterns whose required elements are absent.
 *   2. Native path — for `native == true` patterns, run the stack-only
 *      @ref match_linear_pattern.  Zero heap allocations on the hot path; handles
 *      ~84 % of the loaded pattern corpus (v2 + smallrings).
 *   3. Scaffold cache — for patterns that don't parse natively, look up the
 *      molecule's canonical Bemis-Murcko scaffold hash.  Hits reuse cached
 *      scaffold-local 4-tuples via a scaffold-to-molecule atom map, avoiding
 *      `SubstructMatch`.
 *   4. RDKit fallback — only for the residual patterns (nested recursive
 *      SMARTS, ring-closure topology the linear matcher cannot represent).
 *
 *  Correctness: identical torsion set to RDKit's `getExperimentalTorsions()`.
 *  The native matcher is byte-equivalence validated against RDKit across
 *  100K ChEMBL and 1M USearch samples in `test_csd_torsion_predicates.cpp`.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include <GraphMol/GraphMol.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/SmilesParse/SmartsWrite.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <GraphMol/ChemTransforms/ChemTransforms.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/preprocess/csd_torsion_native.hpp"
#include "smallfield/preprocess/native_murcko.hpp"
#include "smallfield/preprocess/scaffold_cache.hpp"

namespace smallfield::preprocess {
using smallfield::types::atom_flag_aromatic_k;
using smallfield::types::atom_has_flag;
using smallfield::types::bond_flag_aromatic_k;
using smallfield::types::etk_torsion_descriptor_t;
using smallfield::types::hybridization_sp2_k;
using smallfield::types::max_atoms_k;
using smallfield::types::max_bonds_k;
using smallfield::types::molecule_descriptor_t;

/** Compiled CSD torsion pattern. */
struct csd_compiled_pattern_t {
    std::string smarts;
    std::unique_ptr<RDKit::ROMol const> pattern;
    std::vector<double> V;
    std::vector<std::int8_t> signs;
    std::uint32_t idx[4];                 ///< atom indices in the pattern for labels :1 through :4
    std::uint64_t required_elements;      ///< bitset: bit N = pattern requires element N
    std::uint8_t required_hybridizations; ///< bitset: bit H = pattern requires hybridization H (see @ref hybridization_t)
    bool requires_aromatic;               ///< true if any pattern atom is aromatic
    bool native;                          ///< true if the pattern parsed cleanly into native form
    simple_torsion_pattern_t native_form; ///< valid only when native == true
};

/**
 *  Hybridizations that every match of the pattern must contain.  Per-atom,
 *  a hybridization H is required iff EVERY disjunct of that atom pins
 *  hybridization to H; if any disjunct is unconstrained or pins a different
 *  value, the atom can still match molecules without H.  The pattern's
 *  overall required mask is the union of these per-atom required sets
 *  across the 4 labelled atoms (branches excluded — they're soft).
 */
inline std::uint8_t compute_required_hybridizations(simple_torsion_pattern_t const &pat) {
    std::uint8_t mask = 0;
    for (auto const &a : pat.atom) {
        if (a.num_disjuncts == 0) continue;
        std::uint8_t const h0 = a.disjuncts[0].hybridization;
        if (h0 == 0) continue;
        bool unanimous = true;
        for (std::uint8_t i = 1; i < a.num_disjuncts; ++i) {
            if (a.disjuncts[i].hybridization != h0) {
                unanimous = false;
                break;
            }
        }
        if (unanimous && h0 < 8) mask |= static_cast<std::uint8_t>(1u << h0);
    }
    return mask;
}

/** Embedded contents of the RDKit .in files for torsion preferences. */
inline std::string const &csd_torsionPreferencesV2() {
    static std::string const data =
#include "csd_data/torsionPreferences_v2.in_str"
        ;
    return data;
}
inline std::string const &csd_torsionPreferencesSmallRings() {
    static std::string const data =
#include "csd_data/torsionPreferences_smallrings.in_str"
        ;
    return data;
}

/** Parse one line from a .in block: "SMARTS s1 V1 s2 V2 s3 V3 s4 V4 s5 V5 s6 V6". */
inline bool parse_pattern_line(std::string const &line, csd_compiled_pattern_t &out) {
    std::istringstream iss(line);
    if (!(iss >> out.smarts)) return false;
    out.V.clear();
    out.signs.clear();
    out.V.reserve(6);
    out.signs.reserve(6);
    for (std::uint32_t i = 0; i < 6; ++i) {
        int s;
        double V;
        if (!(iss >> s >> V)) return false;
        out.signs.push_back(static_cast<std::int8_t>(s));
        out.V.push_back(V);
    }
    return true;
}

/**
 *  Parse the embedded .in block (one SMARTS per non-comment line) and compile each
 *  SMARTS to an ROMol. Fills required_elements bitset and the atom-label idx[4].
 */
inline std::vector<csd_compiled_pattern_t> compile_patterns(std::string const &block) {
    std::vector<csd_compiled_pattern_t> out;
    out.reserve(400);
    std::istringstream ss(block);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty() || line[0] == '#') continue;

        csd_compiled_pattern_t pat;
        if (!parse_pattern_line(line, pat)) continue;

        pat.pattern.reset(RDKit::SmartsToMol(pat.smarts));
        if (!pat.pattern) continue;

        // Find the atoms labeled :1 through :4 via their molAtomMapNumber property.
        pat.idx[0] = pat.idx[1] = pat.idx[2] = pat.idx[3] = 0xFFFFFFFFu;
        pat.required_elements = 0;
        pat.required_hybridizations = 0;
        pat.requires_aromatic = false;
        for (std::uint32_t i = 0; i < pat.pattern->getNumAtoms(); ++i) {
            auto const *patom = pat.pattern->getAtomWithIdx(i);
            int map_num;
            if (patom->getPropIfPresent("molAtomMapNumber", map_num))
                if (map_num >= 1 && map_num <= 4) pat.idx[map_num - 1] = i;
            int const anum = patom->getAtomicNum();
            if (anum > 0 && anum < 64) pat.required_elements |= (1ULL << anum);
            if (patom->getIsAromatic()) pat.requires_aromatic = true;
        }

        // Valid pattern must have all 4 atom labels mapped.
        if (pat.idx[0] == 0xFFFFFFFFu || pat.idx[1] == 0xFFFFFFFFu || pat.idx[2] == 0xFFFFFFFFu ||
            pat.idx[3] == 0xFFFFFFFFu)
            continue;

        // Try to parse into a simple linear native form. Falls through silently on
        // any unsupported feature (branches, recursive SMARTS, logical ops).
        pat.native = parse_linear_torsion(pat.smarts, pat.native_form);
        if (pat.native) {
            for (std::uint32_t i = 0; i < 6; ++i) {
                pat.native_form.signs[i] = (i < pat.signs.size()) ? pat.signs[i] : static_cast<std::int8_t>(1);
                pat.native_form.V[i] = (i < pat.V.size()) ? static_cast<float>(pat.V[i]) : 0.0f;
            }
            pat.required_hybridizations = compute_required_hybridizations(pat.native_form);
        }

        out.push_back(std::move(pat));
    }
    return out;
}

/** Singleton: lazy-load and compile all CSD patterns once per process. */
inline std::vector<csd_compiled_pattern_t> const &csd_patterns() {
    static std::vector<csd_compiled_pattern_t> const patterns = [] {
        std::string combined = csd_torsionPreferencesV2();
        combined += "\n";
        combined += csd_torsionPreferencesSmallRings();
        return compile_patterns(combined);
    }();
    return patterns;
}

/**
 *  @brief Extract CSD torsions with element prefiltering, replacing
 *  ForceFields::CrystalFF::getExperimentalTorsions().
 *
 *  Native matching loop: for each pattern, check element bitset against molecule;
 *  skip patterns whose required elements are absent. Call RDKit's SubstructMatch
 *  only for patterns that could possibly match.
 *
 *  Also handles the "basic knowledge" torsions (flat aromatic rings, sp2 centers
 *  with 3 neighbors) that getExperimentalTorsions produces.
 */

inline void extract_etk_torsions_native(RDKit::ROMol const &molecule, molecule_descriptor_t const &desc,
                                        std::vector<etk_torsion_descriptor_t> &torsions,
                                        scaffold_cache_t *cache = nullptr) {
    torsions.clear();

    std::uint32_t const num_atoms = desc.num_atoms;
    std::uint32_t const num_bonds = desc.num_bonds;

    std::uint64_t molecule_elements = 0;
    std::uint8_t molecule_hybridizations = 0;
    bool has_aromatic = false;
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        std::uint8_t e = desc.element[a];
        if (e > 0 && e < 64) molecule_elements |= (1ULL << e);
        if (atom_has_flag(desc, a, atom_flag_aromatic_k)) has_aromatic = true;
        std::uint8_t h = desc.hybridization[a];
        if (h < 8) molecule_hybridizations |= static_cast<std::uint8_t>(1u << h);
    }

    boost::dynamic_bitset<> excluded_bonds(num_bonds);
    {
        auto const *ring_info = molecule.getRingInfo();
        if (ring_info != nullptr && ring_info->isInitialized()) {
            constexpr std::uint32_t min_macrocycle_size = 9;
            auto const &bond_rings = ring_info->bondRings();
            for (auto rii = bond_rings.begin(); rii != bond_rings.end(); ++rii) {
                boost::dynamic_bitset<> rs1(num_bonds);
                for (auto bi : *rii) rs1.set(bi);
                for (auto rjj = rii + 1; rjj != bond_rings.end(); ++rjj) {
                    if (rii->size() >= min_macrocycle_size && rjj->size() >= min_macrocycle_size) continue;
                    std::uint32_t n_common = 0;
                    for (auto bj : *rjj) {
                        if (rs1[bj]) {
                            if (++n_common > 1) break;
                        }
                    }
                    if (n_common > 1) {
                        if (rii->size() < min_macrocycle_size)
                            for (auto bi : *rii) excluded_bonds.set(bi);
                        if (rjj->size() < min_macrocycle_size)
                            for (auto bj : *rjj) excluded_bonds.set(bj);
                    }
                }
            }
        }
    }

    boost::dynamic_bitset<> done_bonds(num_bonds);
    auto const *ring_info = molecule.getRingInfo();

    // Done-bonds as a raw uint8_t array on the stack, avoiding a dynamic_bitset heap allocation on the
    // native matcher's hot path. Kept in sync with the dynamic_bitset used by the fallback path.
    std::uint8_t done_bonds_u8[max_bonds_k] = {};
    for (std::uint32_t b = 0; b < num_bonds; ++b) {
        if (done_bonds[b]) done_bonds_u8[b] = 1;
    }

    auto const &patterns = csd_patterns();

    // Native Murcko scaffold extraction. This replaces `RDKit::MurckoDecompose`, which is O(N³) via
    // Floyd-Warshall, with a native O(N+M) peel on the pre-computed ring info. `compute_native_murcko`
    // also canonicalizes the scaffold's atom order via Weisfeiler-Lehman
    // refinement, so `scaffold_to_mol` / `mol_to_scaffold` are defined without
    // a second SubstructMatch call. Symmetric scaffolds (WL-tied atoms) skip
    // the cache — rare in drug-like molecules.
    std::uint64_t scaffold_hash = 0;
    std::int8_t scaffold_to_mol[max_atoms_k];
    std::int8_t mol_to_scaffold[max_atoms_k];
    bool mol_in_scaffold[max_atoms_k] = {};
    std::memset(scaffold_to_mol, -1, sizeof(scaffold_to_mol));
    std::memset(mol_to_scaffold, -1, sizeof(mol_to_scaffold));
    bool have_scaffold_path = false;
    scaffold_cache_entry_t const *cached_entry = nullptr;
    scaffold_cache_entry_t fresh_entry; // populated on miss; inserted at end
    bool will_populate = false;

    if (cache != nullptr) {
        native_murcko_result_t nm;
        if (compute_native_murcko(desc, nm) && nm.valid && nm.canonicalizable) {
            scaffold_hash = nm.scaffold_hash;
            std::memcpy(scaffold_to_mol, nm.scaffold_to_mol, sizeof(scaffold_to_mol));
            std::memcpy(mol_to_scaffold, nm.mol_to_scaffold, sizeof(mol_to_scaffold));
            for (std::uint32_t a = 0; a < num_atoms; ++a) mol_in_scaffold[a] = nm.mol_in_scaffold[a];
            have_scaffold_path = true;
            cached_entry = cache->lookup(scaffold_hash);
            if (cached_entry == nullptr) {
                will_populate = true;
                fresh_entry.pattern_matches_flat.assign(patterns.size(), {});
                fresh_entry.has_r_group_matches.assign(patterns.size(), 0);
            }
        }
    }

    auto emit_torsion_for = [&](std::uint8_t a1, std::uint8_t a2, std::uint8_t a3, std::uint8_t a4,
                                csd_compiled_pattern_t const &pat) {
        auto const *bnd = molecule.getBondBetweenAtoms(a2, a3);
        if (!bnd) return;
        unsigned int const bid = bnd->getIdx();
        if (excluded_bonds[bid] || ring_info->numBondRings(bid) > 3) {
            done_bonds.set(bid);
            done_bonds_u8[bid] = 1;
            return;
        }
        if (done_bonds[bid]) return;
        done_bonds.set(bid);
        done_bonds_u8[bid] = 1;
        etk_torsion_descriptor_t torsion {};
        torsion.atom_indices[0] = a1;
        torsion.atom_indices[1] = a2;
        torsion.atom_indices[2] = a3;
        torsion.atom_indices[3] = a4;
        for (std::uint32_t i = 0; i < 6; ++i) {
            double const sign = (i < pat.signs.size()) ? static_cast<double>(pat.signs[i]) : 1.0;
            double const V = (i < pat.V.size()) ? pat.V[i] : 0.0;
            torsion.fourier_coefficients[i] = static_cast<float>(sign * V);
        }
        torsions.push_back(torsion);
    };

    for (std::size_t pattern_index = 0; pattern_index < patterns.size(); ++pattern_index) {
        auto const &pat = patterns[pattern_index];
        // Element prefilter: skip patterns whose required elements are absent.
        if ((pat.required_elements & ~molecule_elements) != 0) continue;
        if (pat.requires_aromatic && !has_aromatic) continue;
        if ((pat.required_hybridizations & ~molecule_hybridizations) != 0) continue;

        if (pat.native) {
            // Stack-only native matcher. Zero heap allocations in this branch.
            match_linear_pattern(
                desc, pat.native_form, done_bonds_u8,
                [&](std::uint8_t a1, std::uint8_t a2, std::uint8_t a3, std::uint8_t a4, std::uint32_t bid) -> bool {
                    if (excluded_bonds[bid] || ring_info->numBondRings(bid) > 3) {
                        done_bonds.set(bid);
                        done_bonds_u8[bid] = 1;
                        return true;
                    }
                    done_bonds.set(bid);
                    done_bonds_u8[bid] = 1;

                    etk_torsion_descriptor_t torsion {};
                    torsion.atom_indices[0] = a1;
                    torsion.atom_indices[1] = a2;
                    torsion.atom_indices[2] = a3;
                    torsion.atom_indices[3] = a4;
                    for (std::uint32_t i = 0; i < 6; ++i)
                        torsion.fourier_coefficients[i] = static_cast<float>(pat.native_form.signs[i]) *
                                                          pat.native_form.V[i];
                    torsions.push_back(torsion);
                    return true;
                });
            continue;
        }

        // Scaffold-cache hit path.
        // If we have a cached entry and this pattern was recorded as fully
        // scaffold-local (no R-group-spanning matches were observed when the
        // entry was populated), we can skip the full-molecule SubstructMatch
        // and translate the cached scaffold-atom-index 4-tuples into molecule
        // indices via scaffold_to_mol[].
        if (cached_entry != nullptr && have_scaffold_path && pattern_index < cached_entry->has_r_group_matches.size() &&
            cached_entry->has_r_group_matches[pattern_index] == 0) {
            auto const &flat = cached_entry->pattern_matches_flat[pattern_index];
            for (std::size_t k = 0; k + 3 < flat.size(); k += 4) {
                std::int8_t const m1 = scaffold_to_mol[flat[k + 0]];
                std::int8_t const m2 = scaffold_to_mol[flat[k + 1]];
                std::int8_t const m3 = scaffold_to_mol[flat[k + 2]];
                std::int8_t const m4 = scaffold_to_mol[flat[k + 3]];
                if (m1 < 0 || m2 < 0 || m3 < 0 || m4 < 0) continue;
                emit_torsion_for(static_cast<std::uint8_t>(m1), static_cast<std::uint8_t>(m2),
                                 static_cast<std::uint8_t>(m3), static_cast<std::uint8_t>(m4), pat);
            }
            continue; // skip full-molecule SubstructMatch
        }

        // Fallback: run RDKit SubstructMatch on the full molecule.
        // Thread-local: capacity persists across calls on the same thread so
        // repeated fallbacks don't re-allocate. `.clear()` keeps the
        // reserved storage.
        thread_local std::vector<RDKit::MatchVectType> matches;
        matches.clear();
        RDKit::SubstructMatch(molecule, *pat.pattern, matches, /*uniquify=*/false, /*recursionPossible=*/true);

        bool any_r_group = false;
        for (auto const &match : matches) {
            unsigned int const a1 = match[pat.idx[0]].second;
            unsigned int const a2 = match[pat.idx[1]].second;
            unsigned int const a3 = match[pat.idx[2]].second;
            unsigned int const a4 = match[pat.idx[3]].second;

            // Classify match for cache population: scaffold-local (all 4 atoms
            // in scaffold) or R-group-spanning. Only scaffold-local matches
            // are cached; R-group matches force re-running SubstructMatch on
            // future hits (we record the bit but otherwise skip caching them).
            bool const all_in_scaffold = have_scaffold_path && a1 < num_atoms && a2 < num_atoms && a3 < num_atoms &&
                                         a4 < num_atoms && mol_in_scaffold[a1] && mol_in_scaffold[a2] &&
                                         mol_in_scaffold[a3] && mol_in_scaffold[a4];
            if (!all_in_scaffold) any_r_group = true;
            if (will_populate && all_in_scaffold) {
                std::int8_t const s1 = mol_to_scaffold[a1];
                std::int8_t const s2 = mol_to_scaffold[a2];
                std::int8_t const s3 = mol_to_scaffold[a3];
                std::int8_t const s4 = mol_to_scaffold[a4];
                if (s1 >= 0 && s2 >= 0 && s3 >= 0 && s4 >= 0) {
                    auto &flat = fresh_entry.pattern_matches_flat[pattern_index];
                    flat.push_back(static_cast<std::uint8_t>(s1));
                    flat.push_back(static_cast<std::uint8_t>(s2));
                    flat.push_back(static_cast<std::uint8_t>(s3));
                    flat.push_back(static_cast<std::uint8_t>(s4));
                }
            }

            emit_torsion_for(static_cast<std::uint8_t>(a1), static_cast<std::uint8_t>(a2),
                             static_cast<std::uint8_t>(a3), static_cast<std::uint8_t>(a4), pat);
        }
        if (will_populate) fresh_entry.has_r_group_matches[pattern_index] = any_r_group ? 1 : 0;
    }
    if (will_populate) cache->insert_if_missing(scaffold_hash, std::move(fresh_entry));

    // Matches RDKit getExperimentalTorsions' useBasicKnowledge=true behavior.
    if (ring_info != nullptr && ring_info->isInitialized()) {
        for (auto const &atom_ring : ring_info->atomRings()) {
            std::uint32_t const rsize = static_cast<std::uint32_t>(atom_ring.size());
            if (rsize < 4 || rsize > 6) continue;
            for (std::uint32_t i = 0; i < rsize; ++i) {
                std::uint32_t const a1 = atom_ring[i];
                std::uint32_t const a2 = atom_ring[(i + 1) % rsize];
                std::uint32_t const a3 = atom_ring[(i + 2) % rsize];
                std::uint32_t const a4 = atom_ring[(i + 3) % rsize];
                auto const *bnd = molecule.getBondBetweenAtoms(a2, a3);
                if (!bnd) continue;
                std::uint32_t const bid = bnd->getIdx();
                if (done_bonds[bid]) continue;
                constexpr std::uint8_t sp2 = hybridization_sp2_k;
                if (desc.hybridization[a1] != sp2 || desc.hybridization[a2] != sp2 || desc.hybridization[a3] != sp2 ||
                    desc.hybridization[a4] != sp2)
                    continue;

                done_bonds.set(bid);
                done_bonds_u8[bid] = 1;
                etk_torsion_descriptor_t torsion {};
                torsion.atom_indices[0] = static_cast<std::uint8_t>(a1);
                torsion.atom_indices[1] = static_cast<std::uint8_t>(a2);
                torsion.atom_indices[2] = static_cast<std::uint8_t>(a3);
                torsion.atom_indices[3] = static_cast<std::uint8_t>(a4);
                // Flat-ring signature: signs[1]=-1, V[1]=100.0 (from RDKit's hardcoded values).
                for (std::uint32_t k = 0; k < 6; ++k) torsion.fourier_coefficients[k] = 0.0f;
                torsion.fourier_coefficients[1] = -100.0f;
                torsions.push_back(torsion);
            }
        }
    }
}

} // namespace smallfield::preprocess
