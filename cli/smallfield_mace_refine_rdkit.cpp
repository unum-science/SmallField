/**
 *  @file smallfield_mace_refine_rdkit.cpp
 *  @brief RDKit-backed atomic-number resolver for the standalone refiner.
 *
 *  Split out of `smallfield_mace_refine.cpp` so the main TU stays RDKit-free
 *  (mirrors the `cli/rdkit_fallback.cpp` split established by the existing
 *  `smallfield_batch` binary).  Building both TUs into the final binary
 *  links RDKit; building only the main TU lets a unit test stub the
 *  `resolve_atomic_numbers_from_smiles` symbol.
 */

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/MolOps.h>

namespace {

/// Resolve atomic numbers in the order RDKit emits them (post-`AddHs`).  When
/// `expected_num_atoms != 0`, it should match the SmallField parquet's
/// `num_atoms` column; `0` means "derive the count from RDKit and don't enforce
/// a parquet-side expectation".  On any RDKit failure or count mismatch we
/// return an empty vector to signal "skip this row".
std::vector<int> resolve_(std::string const &smiles, std::uint8_t expected_num_atoms) {
    std::unique_ptr<RDKit::ROMol> molecule;
    try {
        molecule.reset(RDKit::SmilesToMol(smiles));
    }
    catch (...) {
        return {};
    }
    if (!molecule) return {};
    auto rwmol = std::unique_ptr<RDKit::RWMol>(new RDKit::RWMol(*molecule));
    try {
        RDKit::MolOps::addHs(*rwmol);
    }
    catch (...) {
        return {};
    }
    auto const n = static_cast<std::uint32_t>(rwmol->getNumAtoms());
    if (expected_num_atoms != 0u && n != expected_num_atoms) {
        std::fprintf(
            stderr,
            "[smallfield_mace_refine] atom-count mismatch: SMILES %s has %u atoms, " "parquet says %u — skipping row\n",
            smiles.c_str(), n, static_cast<unsigned>(expected_num_atoms));
        return {};
    }
    std::vector<int> out(n);
    for (std::uint32_t i = 0; i < n; ++i) { out[i] = static_cast<int>(rwmol->getAtomWithIdx(i)->getAtomicNum()); }
    return out;
}

} // namespace

extern std::vector<int> resolve_atomic_numbers_from_smiles(std::string const &smiles, std::uint8_t expected_num_atoms);
std::vector<int> resolve_atomic_numbers_from_smiles(std::string const &smiles, std::uint8_t expected_num_atoms) {
    return resolve_(smiles, expected_num_atoms);
}
