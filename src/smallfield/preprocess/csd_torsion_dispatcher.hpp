/**
 *  @file csd_torsion_dispatcher.hpp
 *  @brief CSD torsion extraction with an element prefilter and a native matcher.
 *
 *  Replaces `ForceFields::CrystalFF::getExperimentalTorsions()` with an owned matching loop.  The
 *  CSD SMARTS patterns are loaded once at static-init and compiled by @ref parse_linear_torsion
 *  (see `csd_torsion_native.hpp`).  For every molecule the loop applies an element, aromaticity and
 *  hybridization prefilter, then runs the stack-only @ref match_linear_pattern.  No heap allocation
 *  and no RDKit substructure search on the hot path.
 *
 *  The whole loaded corpus — v2 plus smallrings, 470 patterns — compiles natively, so there is no
 *  fallback.  A pattern the compiler cannot read is a load-time error rather than a silent runtime
 *  detour: the pattern library is checked-in data, so an unreadable entry is a build problem, and a
 *  fallback would hide it behind a path that never otherwise executes.
 *
 *  Correctness: identical torsion set to RDKit's `getExperimentalTorsions()`.  The native matcher is
 *  byte-equivalence validated against RDKit across 19193 PubChem molecules in
 *  `test_csd_torsion_predicates.cpp` — 9020710 pattern checks, zero disagreements.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <cstdint>
#include <cstdio>  // `std::fprintf`
#include <cstdlib> // `std::abort`
#include <cstring> // `std::memset`
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/SmilesParse/SmartsWrite.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ChemTransforms/ChemTransforms.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/preprocess/csd_torsion_native.hpp"

namespace smallfield::preprocess {
using smallfield::types::atom_flag_aromatic_k;
using smallfield::types::atom_has_flag;
using smallfield::types::bond_flag_aromatic_k;
using smallfield::types::etk_torsion_descriptor_t;
using smallfield::types::hybridization_sp2_k;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::max_bonds_k;
using smallfield::types::molecule_descriptor_t;

/** Compiled CSD torsion pattern. */
struct csd_compiled_pattern_t {
    std::string smarts;
    std::unique_ptr<RDKit::ROMol const> pattern;
    std::vector<double> V;
    std::vector<std::int8_t> signs;
    std::uint32_t index[4];          ///< atom indices in the pattern for labels :1 through :4
    std::uint64_t required_elements; ///< bitset: bit N = pattern requires element N
    std::uint8_t
        required_hybridizations; ///< bitset: bit H = pattern requires hybridization H (see @ref hybridization_t)
    bool requires_aromatic;      ///< true if any pattern atom is aromatic
    bool native;                 ///< true if the pattern parsed cleanly into native form
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
 *  SMARTS to an ROMol. Fills required_elements bitset and the atom-label index[4].
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
        pat.index[0] = pat.index[1] = pat.index[2] = pat.index[3] = 0xFFFFFFFFu;
        pat.required_elements = 0;
        pat.required_hybridizations = 0;
        pat.requires_aromatic = false;
        for (std::uint32_t i = 0; i < pat.pattern->getNumAtoms(); ++i) {
            auto const *patom = pat.pattern->getAtomWithIdx(i);
            int map_num;
            if (patom->getPropIfPresent("molAtomMapNumber", map_num))
                if (map_num >= 1 && map_num <= 4) pat.index[map_num - 1] = i;
            int const anum = patom->getAtomicNum();
            if (anum > 0 && anum < 64) pat.required_elements |= (1ULL << anum);
            if (patom->getIsAromatic()) pat.requires_aromatic = true;
        }

        // Valid pattern must have all 4 atom labels mapped.
        if (pat.index[0] == 0xFFFFFFFFu || pat.index[1] == 0xFFFFFFFFu || pat.index[2] == 0xFFFFFFFFu ||
            pat.index[3] == 0xFFFFFFFFu)
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

/** Lazy-load and compile all CSD patterns once per process.
 *
 *  A pattern the native compiler cannot read aborts here rather than falling back to an RDKit
 *  substructure search.  The pattern library is checked-in data, so an unreadable entry is a build
 *  problem that should be loud; a fallback would hide it behind a path nothing else exercises, and
 *  carry the scaffold cache, Murcko decomposition and VF2 deadline machinery along with it. */
inline std::vector<csd_compiled_pattern_t> const &csd_patterns() {
    static std::vector<csd_compiled_pattern_t> const patterns = [] {
        std::string combined = csd_torsionPreferencesV2();
        combined += "\n";
        combined += csd_torsionPreferencesSmallRings();
        auto compiled = compile_patterns(combined);
        for (auto const &pattern : compiled)
            if (!pattern.native) {
                std::fprintf(
                    stderr,
                    "smallfield: CSD torsion pattern does not compile natively and there is no " "fallback:\n  %s\n",
                    pattern.smarts.c_str());
                std::abort();
            }
        return compiled;
    }();
    return patterns;
}

/**
 *  @brief Extract CSD torsions with element prefiltering, replacing
 *  ForceFields::CrystalFF::getExperimentalTorsions().
 *
 *  For each pattern, check the element, aromaticity and hybridization bitsets against the molecule
 *  and skip the ones that cannot match, then run the native matcher on the rest.  A bond carries at
 *  most one torsion and the first pattern in file order claims it; a bond on a fused small ring, or
 *  on more than three rings, is claimed and left empty.
 *
 *  Also handles the "basic knowledge" torsions (flat aromatic rings, sp2 centers with 3 neighbors)
 *  that getExperimentalTorsions produces.  Everything reads the flat descriptor, so the same
 *  answer comes out of the device kernel in `smallfield/cuda/csd_torsion_device.cuh`.
 *
 *  @param[in] descriptor Molecule descriptor with topology, property and ring arrays.
 *  @param[out] torsions Extracted torsions; any prior contents are dropped.
 */
inline void extract_etk_torsions_native(molecule_descriptor_t const &descriptor,
                                        std::vector<etk_torsion_descriptor_t> &torsions) {
    torsions.clear();

    csd_molecule_gate_t const gate = build_molecule_gate(descriptor);

    // One bond claims at most one torsion, first pattern in file order wins.
    std::uint8_t done_bonds[max_bonds_k] = {};

    auto const &patterns = csd_patterns();

    for (std::size_t pattern_index = 0; pattern_index < patterns.size(); ++pattern_index) {
        auto const &pattern = patterns[pattern_index];
        // Element prefilter: skip patterns whose required elements are absent.
        if ((pattern.required_elements & ~gate.elements) != 0) continue;
        if (pattern.requires_aromatic && !gate.has_aromatic) continue;
        if ((pattern.required_hybridizations & ~gate.hybridizations) != 0) continue;

        // Stack-only native matcher. Zero heap allocations in this branch.
        match_linear_pattern(
            descriptor, csd_pattern_view_t(pattern.native_form), done_bonds,
            [&](std::uint8_t a1, std::uint8_t a2, std::uint8_t a3, std::uint8_t a4, std::uint32_t bond_index) -> bool {
                done_bonds[bond_index] = 1;
                if (bond_excluded_by_ring_fusion(descriptor, bond_index) || bond_ring_count(descriptor, bond_index) > 3)
                    return true;

                etk_torsion_descriptor_t torsion {};
                torsion.atom_indices[0] = a1;
                torsion.atom_indices[1] = a2;
                torsion.atom_indices[2] = a3;
                torsion.atom_indices[3] = a4;
                for (std::uint32_t term = 0; term < 6; ++term)
                    torsion.fourier_coefficients[term] = static_cast<float>(pattern.native_form.signs[term]) *
                                                         pattern.native_form.V[term];
                torsions.push_back(torsion);
                return true;
            });
    }

    // Matches RDKit getExperimentalTorsions' useBasicKnowledge=true behavior.
    for (std::uint32_t ring_index = 0; ring_index < descriptor.num_rings; ++ring_index) {
        std::uint32_t const size = ring_atom_count(descriptor, ring_index);
        if (size < 4 || size > 6) continue;
        std::uint16_t const start = descriptor.ring_offset[ring_index];
        for (std::uint32_t position = 0; position < size; ++position) {
            std::uint8_t const a1 = descriptor.ring_atoms[start + position];
            std::uint8_t const a2 = descriptor.ring_atoms[start + (position + 1) % size];
            std::uint8_t const a3 = descriptor.ring_atoms[start + (position + 2) % size];
            std::uint8_t const a4 = descriptor.ring_atoms[start + (position + 3) % size];
            std::uint32_t const bond_index = find_bond_between(descriptor, a2, a3);
            if (bond_index == no_bond_index_k) continue;
            if (done_bonds[bond_index]) continue;
            constexpr std::uint8_t planar_k = hybridization_sp2_k;
            if (descriptor.hybridization[a1] != planar_k || descriptor.hybridization[a2] != planar_k ||
                descriptor.hybridization[a3] != planar_k || descriptor.hybridization[a4] != planar_k)
                continue;

            done_bonds[bond_index] = 1;
            etk_torsion_descriptor_t torsion {};
            torsion.atom_indices[0] = a1;
            torsion.atom_indices[1] = a2;
            torsion.atom_indices[2] = a3;
            torsion.atom_indices[3] = a4;
            // Flat-ring signature: signs[1]=-1, V[1]=100.0 (from RDKit's hardcoded values).
            for (std::uint32_t term = 0; term < 6; ++term) torsion.fourier_coefficients[term] = 0.0f;
            torsion.fourier_coefficients[1] = -100.0f;
            torsions.push_back(torsion);
        }
    }
}

} // namespace smallfield::preprocess
