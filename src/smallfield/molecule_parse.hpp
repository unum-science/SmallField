/**
 *  @file src/smallfield/molecule_parse.hpp
 *  @brief The one place a SMILES string becomes an `RWMol`.
 *
 *  Four call sites need a SMILES parsed — the device preprocess path, the host preprocess path, the
 *  shape-descriptor postprocess, and the CPU fallback — and independent parsing lets them drift:
 *  one site gating the atom count against a hardcoded 128 where another uses
 *  `max_atoms_per_molecule_k`, or `AddHs` failure semantics differing between sites.
 *
 *  Drift here is not cosmetic. `compute_shape_descriptors` accepts a molecule's descriptors only
 *  when a fresh parse reproduces the atom count the GPU path recorded, so two sites that disagree
 *  about how a SMILES becomes a molecule silently cost that molecule its shape descriptors.
 *
 *  Deliberately free of CUDA and of the batch pool, so `cli/` and `postprocess/` can include
 *  it without pulling in `cuda_runtime.h`.
 */

#pragma once

#include <cstdint> // `std::uint32_t`

#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`

#include <GraphMol/Descriptors/MolDescriptors.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolStandardize/Fragment.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/types/conformer_status.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield {

/** What to do with a SMILES that names more than one disconnected fragment. */
enum salt_policy_t : std::uint8_t {
    /** Keep every fragment, counterions included. The bond-hop tests need this to reach the
     *  unreachable-hop sentinel, which a single connected component never produces. */
    keep_all_fragments_k = 0,
    /** Keep the largest fragment and drop the rest. A counterion has no meaningful
     *  conformer, and MMFF94 cannot type an isolated `HCl`, so keeping one rejects the whole
     *  molecule — parent included. */
    keep_largest_fragment_k = 1,
};

/** Whether to spend the extra RDKit calls that fill the shard's descriptive columns. */
enum metadata_policy_t : std::uint8_t {
    skip_metadata_k = 0,
    compute_metadata_k = 1,
};

/** Knobs the four call sites genuinely disagree on. */
struct parse_options_t {
    /** Reject past this many atoms, counting hydrogens; zero disables the check.
     *  The preprocess path passes the compile-time warp-pipeline cap, the CPU fallback passes the
     *  operator's runtime `--max-atoms`, and the postprocess path passes zero because it only ever
     *  sees molecules the pipeline already accepted. */
    std::uint32_t max_atoms = types::max_atoms_per_molecule_k;
    salt_policy_t salt_policy = keep_largest_fragment_k;
    metadata_policy_t metadata_policy = skip_metadata_k;
};

/** A parsed molecule, or the reason there isn't one. */
struct parse_result_t {
    /** Null unless `status` is `success_k`. */
    std::unique_ptr<RDKit::RWMol> molecule;
    types::conformer_status_t status = types::conformer_status_t::invalid_smiles_k;
    std::uint32_t num_atoms = 0;
    /** Zero unless `parse_options_t::metadata_policy` asked for them. */
    std::uint32_t num_heavy_atoms = 0;
    std::uint32_t num_bonds = 0;
    float molecular_weight = 0.0f;
};

/**
 *  @brief Whether a SMILES names more than one fragment.
 *
 *  A `.` inside brackets is part of an atom specification rather than a fragment separator, so the
 *  scan tracks bracket depth. Worth the pass: the chooser walks the whole graph, and on every
 *  corpus measured here the overwhelming majority of molecules are a single fragment.
 */
inline bool names_multiple_fragments_(std::string_view smiles) noexcept {
    std::uint32_t bracket_depth = 0;
    for (char const character : smiles) {
        if (character == '[') ++bracket_depth;
        else if (character == ']') {
            if (bracket_depth > 0) --bracket_depth;
        }
        else if (character == '.' && bracket_depth == 0) return true;
    }
    return false;
}

/**
 *  @brief Parse a SMILES into a hydrogen-explicit molecule, applying one shared contract.
 *
 *  Order is the contract: parse, drop counterions, add hydrogens, gate on atom count, then
 *  optionally measure. Dropping fragments has to precede `AddHs` so the hydrogen count belongs to
 *  the parent, and precede the atom-count gate so a bulky counterion stops pushing an otherwise
 *  ordinary molecule past the cap. The gate is two-sided: a molecule of no atoms is refused as
 *  `invalid_smiles_k`, so `success_k` and a non-zero `num_atoms` always travel together.
 *
 *  @param[in] smiles Input string; not required to be null-terminated.
 *  @param[in] options Per-call-site knobs.
 *  @return A molecule and `success_k`, or a null molecule and the reason.
 */
inline parse_result_t parse_smiles(std::string_view smiles, parse_options_t const &options) {
    parse_result_t result;

    try {
        // RDKit::SmilesToMol takes std::string const& — one unavoidable construction per molecule,
        // kept scoped to this call. removeHs=false because addHs follows.
        std::string const smiles_string {smiles};
        RDKit::SmilesParserParams parameters;
        parameters.sanitize = true;
        parameters.removeHs = false;
        parameters.strictCXSMILES = false;
        result.molecule.reset(RDKit::SmilesToMol(smiles_string, parameters));
    }
    catch (...) {
        return result;
    }
    if (!result.molecule) return result;

    if (options.salt_policy == keep_largest_fragment_k && names_multiple_fragments_(smiles)) {
        try {
            // Stateless and const, so one instance serves every worker thread. `preferOrganic`
            // keeps the organic parent even when a metal cluster carries more atoms, which is the
            // fragment the force field can actually type.
            static RDKit::MolStandardize::LargestFragmentChooser const chooser {/*preferOrganic=*/true};
            chooser.chooseInPlace(*result.molecule);
            // Removing atoms leaves the ring information describing the molecule that was parsed,
            // not the one that survived, and anything that consults it afterwards — aromatic SMARTS
            // matching, MMFF typing — throws `RingInfo not initialized` rather than reading stale
            // rings. Re-sanitising is the honest repair and only the multi-fragment minority pays.
            RDKit::MolOps::sanitizeMol(*result.molecule);
        }
        catch (...) {
            result.molecule.reset();
            return result;
        }
    }

    try {
        RDKit::MolOps::addHs(*result.molecule);
    }
    catch (...) {
        result.molecule.reset();
        return result;
    }

    result.num_atoms = static_cast<std::uint32_t>(result.molecule->getNumAtoms());
    // An empty or whitespace-only string parses into a valid `RWMol` of no atoms, which is how a null
    // cell in an input shard arrives here. It is not a molecule: there is no geometry to embed and no
    // atom to type, and admitting it lets a row nothing was generated for count as a success. The
    // status stays `invalid_smiles_k`, the reason `GLOSSARY.md` names for an empty molecule.
    if (result.num_atoms == 0) {
        result.molecule.reset();
        return result;
    }
    if (options.max_atoms != 0 && result.num_atoms > options.max_atoms) {
        result.molecule.reset();
        result.status = types::conformer_status_t::too_many_atoms_k;
        return result;
    }

    if (options.metadata_policy == compute_metadata_k) {
        result.num_heavy_atoms = static_cast<std::uint32_t>(result.molecule->getNumHeavyAtoms());
        result.num_bonds = static_cast<std::uint32_t>(result.molecule->getNumBonds());
        try {
            result.molecular_weight = static_cast<float>(RDKit::Descriptors::calcExactMW(*result.molecule));
        }
        catch (...) {
            result.molecular_weight = 0.0f;
        }
    }

    result.status = types::conformer_status_t::success_k;
    return result;
}

} // namespace smallfield
