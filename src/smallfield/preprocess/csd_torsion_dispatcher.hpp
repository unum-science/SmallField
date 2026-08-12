/**
 *  @file src/smallfield/preprocess/csd_torsion_dispatcher.hpp
 *  @brief CSD torsion extraction with an element prefilter and a native matcher.
 *
 *  Produces the same torsion set as `ForceFields::CrystalFF::getExperimentalTorsions()` with no heap
 *  allocation and no RDKit substructure search on the hot path: the 470 patterns of the v2 and small-ring
 *  corpora are compiled once at first use, and every molecule passes an element, aromaticity and
 *  hybridization prefilter before the stack-only `match_linear_pattern` runs.
 *
 *  There is no runtime fallback to RDKit. The whole corpus compiles natively, and a pattern that stops
 *  doing so is a build problem a fallback would hide behind a path nothing else executes.
 */

#pragma once

#include <cstdint> // `std::int8_t`, `std::uint32_t`
#include <cstdlib> // `std::abort`

#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <GraphMol/GraphMol.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/SmilesParse/SmartsWrite.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ChemTransforms/ChemTransforms.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

#include "smallfield/log.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/views.hpp"
#include "smallfield/preprocess/csd_torsion_native.hpp"
#include "smallfield/preprocess/csd_torsion_table.hpp"

namespace smallfield::preprocess {
using smallfield::types::atom_flag_aromatic_k;
using smallfield::types::bond_flag_aromatic_k;
using smallfield::types::etk_torsion_descriptor_t;
using smallfield::types::hybridization_sp2_k;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::max_bonds_k;
using smallfield::types::molecule_descriptor_t;

/** One CSD torsion pattern, compiled. Every pattern `csd_patterns` hands out has a usable `native_form`,
 *  so nothing here records whether it compiled — a row that did not is dropped at load. */
struct csd_compiled_pattern_t {
    /** Borrowed from the static table, which outlives every caller. */
    std::string_view smarts;
    /** RDKit's own reading of the same SMARTS, which only the cross-validation test consults. */
    std::unique_ptr<RDKit::ROMol const> pattern;
    /** Atom indices in `pattern` for labels :1 through :4. */
    std::uint32_t index[4];
    /** Bit N set when the pattern requires element N. */
    std::uint64_t required_elements;
    /** Bit H set when the pattern requires hybridization H, see `hybridization_t`. */
    std::uint8_t required_hybridizations;
    bool requires_aromatic;
    /** What the matcher reads. */
    simple_torsion_pattern_t native_form;
    /** V₁ through V₆ in kcal/mol with the corpus row's signs already applied. */
    float fourier_coefficients[6];
};

/**
 *  @brief Hybridizations every match of the pattern must contain.
 *
 *  An atom requires hybridization H only when @b every disjunct of it pins H; one unconstrained disjunct,
 *  or one pinning something else, lets the atom match a molecule without H. The pattern's mask is the
 *  union over the four labelled atoms, branches excluded because they are soft.
 */
inline std::uint8_t compute_required_hybridizations(simple_torsion_pattern_t const &pattern) {
    std::uint8_t mask = 0;
    for (auto const &atom : pattern.atom) {
        if (atom.num_disjuncts == 0) continue;
        std::uint8_t const first_hybridization = atom.disjuncts[0].hybridization;
        if (first_hybridization == 0) continue;
        bool unanimous = true;
        for (std::uint8_t disjunct_index = 1; disjunct_index < atom.num_disjuncts; ++disjunct_index) {
            if (atom.disjuncts[disjunct_index].hybridization != first_hybridization) {
                unanimous = false;
                break;
            }
        }
        if (unanimous && first_hybridization < 8) mask |= static_cast<std::uint8_t>(1u << first_hybridization);
    }
    return mask;
}

/**
 *  @brief Compile every row of a corpus, append the ones the matcher can carry to @p out in row order,
 *  and return the number dropped.
 *
 *  Dropping an unreadable row rather than keeping it inert is what makes "every pattern in this vector
 *  matches" an invariant of the container rather than a flag each consumer must remember to test — and
 *  the device table's flattener does not test one. Each drop names its SMARTS on @p logger.
 */
inline std::size_t compile_patterns_into(custd::span<csd_torsion_row_t const> rows,
                                         std::vector<csd_compiled_pattern_t> &out, logger_t const &logger) {
    out.reserve(out.size() + rows.size());
    std::size_t dropped = 0;
    for (csd_torsion_row_t const &row : rows) {
        csd_compiled_pattern_t compiled_pattern;
        compiled_pattern.smarts = row.smarts;

        compiled_pattern.pattern.reset(RDKit::SmartsToMol(std::string(compiled_pattern.smarts)));
        if (!compiled_pattern.pattern) {
            ++dropped;
            logger.error("smallfield: RDKit cannot read a CSD torsion pattern, dropping it:\n  {}", row.smarts);
            continue;
        }

        compiled_pattern.index[0] = compiled_pattern.index[1] = compiled_pattern.index[2] = compiled_pattern.index[3] =
            0xFFFFFFFFu;
        compiled_pattern.required_elements = 0;
        compiled_pattern.required_hybridizations = 0;
        compiled_pattern.requires_aromatic = false;
        for (std::uint32_t atom_index = 0; atom_index < compiled_pattern.pattern->getNumAtoms(); ++atom_index) {
            auto const *pattern_atom = compiled_pattern.pattern->getAtomWithIdx(atom_index);
            int map_number;
            if (pattern_atom->getPropIfPresent("molAtomMapNumber", map_number))
                if (map_number >= 1 && map_number <= 4) compiled_pattern.index[map_number - 1] = atom_index;
            int const atomic_number = pattern_atom->getAtomicNum();
            if (atomic_number > 0 && atomic_number < 64) compiled_pattern.required_elements |= (1ULL << atomic_number);
            if (pattern_atom->getIsAromatic()) compiled_pattern.requires_aromatic = true;
        }

        // All four labels have to be mapped, or there is no torsion for the row to name.
        if (compiled_pattern.index[0] == 0xFFFFFFFFu || compiled_pattern.index[1] == 0xFFFFFFFFu ||
            compiled_pattern.index[2] == 0xFFFFFFFFu || compiled_pattern.index[3] == 0xFFFFFFFFu) {
            ++dropped;
            logger.error("smallfield: a CSD torsion pattern does not label all four atoms, dropping it:\n  {}",
                         row.smarts);
            continue;
        }

        if (!parse_linear_torsion(compiled_pattern.smarts, compiled_pattern.native_form)) {
            ++dropped;
            logger.error("smallfield: a CSD torsion pattern is outside the matcher's vocabulary, dropping it:\n  {}",
                         row.smarts);
            continue;
        }

        // The corpus spells a term as a sign beside a magnitude, and nothing downstream wants the two
        // apart.
        for (std::uint32_t term = 0; term < 6; ++term)
            compiled_pattern.fourier_coefficients[term] = static_cast<float>(row.signs[term]) * row.magnitudes[term];
        compiled_pattern.required_hybridizations = compute_required_hybridizations(compiled_pattern.native_form);

        out.push_back(std::move(compiled_pattern));
    }
    return dropped;
}

/**
 *  @brief Lazy-load and compile all CSD patterns once per process.
 *
 *  The checked-in corpus compiles whole, so a non-zero drop count means a row was edited into syntax the
 *  matcher cannot represent, and the conformers generated for the bonds that row claims will change. It
 *  is reported rather than fatal: this runs at first use inside a library, under a host application with
 *  its own idea of how to fail, and losing one pattern out of 470 degrades the torsion set rather than
 *  invalidating it.
 */
inline std::vector<csd_compiled_pattern_t> const &csd_patterns() {
    static std::vector<csd_compiled_pattern_t> const patterns = [] {
        // First use has no caller to borrow a logger from, so this one owns a default-configured one.
        logger_t const logger;
        // v2 then small rings, in that order: a bond goes to the first pattern that claims it.
        std::vector<csd_compiled_pattern_t> compiled;
        std::size_t const dropped = compile_patterns_into(csd_torsion_rows_v2(), compiled, logger) +
                                    compile_patterns_into(csd_torsion_rows_small_rings(), compiled, logger);
        if (dropped != 0)
            logger.error("smallfield: {} of {} CSD torsion patterns did not compile and are not in the table", dropped,
                         dropped + compiled.size());
        return compiled;
    }();
    return patterns;
}

/**
 *  @brief Extract every CSD torsion of one molecule into @p torsions, whose prior contents are dropped.
 *
 *  The corpus pass is followed by the flat-ring torsions `getExperimentalTorsions` emits under
 *  `useBasicKnowledge`. Everything reads the flat descriptor, so the device kernel in
 *  `smallfield/cuda/csd_torsion_kernel.cuh` produces the same answer.
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
        if ((pattern.required_elements & ~gate.elements) != 0) continue;
        if (pattern.requires_aromatic && !gate.has_aromatic) continue;
        if ((pattern.required_hybridizations & ~gate.hybridizations) != 0) continue;

        match_linear_pattern(
            descriptor, csd_pattern_view_t(pattern.native_form), done_bonds,
            [&](std::uint8_t atom_1, std::uint8_t atom_2, std::uint8_t atom_3, std::uint8_t atom_4,
                std::uint32_t bond_index) -> bool {
                done_bonds[bond_index] = 1;
                if (bond_excluded_by_ring_fusion(descriptor, bond_index) || bond_ring_count(descriptor, bond_index) > 3)
                    return true;

                etk_torsion_descriptor_t torsion {};
                torsion.atom_indices[0] = atom_1;
                torsion.atom_indices[1] = atom_2;
                torsion.atom_indices[2] = atom_3;
                torsion.atom_indices[3] = atom_4;
                for (std::uint32_t term = 0; term < 6; ++term)
                    torsion.fourier_coefficients[term] = pattern.fourier_coefficients[term];
                torsions.push_back(torsion);
                return true;
            });
    }

    // Flat-ring torsions, matching `getExperimentalTorsions` under `useBasicKnowledge`.
    for (std::uint32_t ring_index = 0; ring_index < descriptor.num_rings; ++ring_index) {
        std::uint32_t const size = ring_atom_count(descriptor, ring_index);
        if (size < 4 || size > 6) continue;
        std::uint16_t const start = descriptor.ring_offset[ring_index];
        for (std::uint32_t position = 0; position < size; ++position) {
            std::uint8_t const atom_1 = descriptor.ring_atoms[start + position];
            std::uint8_t const atom_2 = descriptor.ring_atoms[start + (position + 1) % size];
            std::uint8_t const atom_3 = descriptor.ring_atoms[start + (position + 2) % size];
            std::uint8_t const atom_4 = descriptor.ring_atoms[start + (position + 3) % size];
            std::uint32_t const bond_index = find_bond_between(descriptor, atom_2, atom_3);
            if (bond_index == no_bond_index_k) continue;
            if (done_bonds[bond_index]) continue;
            constexpr std::uint8_t planar_k = hybridization_sp2_k;
            if (descriptor.hybridization[atom_1] != planar_k || descriptor.hybridization[atom_2] != planar_k ||
                descriptor.hybridization[atom_3] != planar_k || descriptor.hybridization[atom_4] != planar_k)
                continue;

            done_bonds[bond_index] = 1;
            etk_torsion_descriptor_t torsion {};
            torsion.atom_indices[0] = atom_1;
            torsion.atom_indices[1] = atom_2;
            torsion.atom_indices[2] = atom_3;
            torsion.atom_indices[3] = atom_4;
            // The signature RDKit hardcodes for a flat ring: a sign of -1 on a V of 100.
            for (std::uint32_t term = 0; term < 6; ++term) torsion.fourier_coefficients[term] = 0.0f;
            torsion.fourier_coefficients[1] = -100.0f;
            torsions.push_back(torsion);
        }
    }
}

} // namespace smallfield::preprocess
