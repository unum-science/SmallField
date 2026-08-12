/**
 *  @file test/test_rdkit_reference_helpers.hpp
 *  @brief Shared utilities for C++ ETKDG tests.
 *
 *  Provides molecule loading, coordinate extraction/injection, and RDKit MMFF94 energy evaluation for
 *  cross-referencing GPU-generated conformers against RDKit's reference implementation.
 */

#pragma once

#include <cmath>   // `std::fabs`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`
#include <cstdlib> // `std::getenv`

#include <algorithm>    // `std::max`
#include <charconv>     // `std::from_chars`
#include <fstream>      // `std::ifstream`
#include <limits>       // `std::numeric_limits`
#include <memory>       // `std::unique_ptr`
#include <string>       // `std::string`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`
#include <vector>       // `std::vector`

#include <fmt/format.h> // `fmt::println`

#include <ForceField/ForceField.h>
#include <Geometry/point.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/views.hpp"

#include "test/test_helpers.hpp"

namespace smallfield::tests {

using smallfield::types::conformer_aos_t;
using smallfield::types::hop_matrix_t;
using smallfield::types::mmff94_bond_t;
using smallfield::types::unreachable_hops_k;

/**
 *  @brief Create a molecule from SMILES with explicit hydrogens added.
 *
 *  @param[in] smiles SMILES string to parse.
 *  @return Owning pointer to the molecule, or nullptr on parse failure.
 */
inline std::unique_ptr<RDKit::RWMol> mol_from_smiles(std::string const &smiles) {
    std::unique_ptr<RDKit::RWMol> molecule(RDKit::SmilesToMol(smiles));
    if (!molecule) {
        fmt::println(stderr, "NOTE mol_from_smiles smiles={} parsed=0", smiles);
        return nullptr;
    }
    RDKit::MolOps::addHs(*molecule);
    return molecule;
}

/**
 *  @brief Embed a molecule with ETKDGv3.
 *
 *  @param[inout] molecule Molecule to embed (modified in place with 3D coordinates).
 *  @param[in] seed Random seed for the embedding.
 *  @return True on success.
 */
inline bool embed_molecule(RDKit::RWMol &molecule, int seed = 42) {
    RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
    params.randomSeed = seed;
    int conformer_id = RDKit::DGeomHelpers::EmbedMolecule(molecule, params);
    return conformer_id >= 0;
}

/**
 *  @brief Attach an all-zero 3D conformer sized to the molecule, without embedding it.
 *
 *  `set_coords_interleaved` needs a conformer to overwrite, and a caller that is about to supply its own
 *  coordinates has no use for ETKDG's. Embedding to obtain the slot also discards whichever molecules ETKDG
 *  refuses, and those are not drawn uniformly: over a 4,788-molecule drug-like corpus ETKDGv3 at seed 42
 *  refuses 0.46 % overall but 5.2 % of the 100-128 atom bin, so the discard thins the very bin a size-binned
 *  comparison judges.
 *
 *  @param[inout] molecule Molecule to receive the conformer.
 *  @return Identifier of the conformer just attached.
 */
inline unsigned int attach_empty_conformer(RDKit::RWMol &molecule) {
    auto conformer = std::make_unique<RDKit::Conformer>(molecule.getNumAtoms());
    conformer->set3D(true);
    return molecule.addConformer(conformer.release(), true);
}

/**
 *  @brief Read a molecule's conformer straight into caller-owned atom-major storage.
 *
 *  The atom count is not a parameter: @p atom_major carries its own extent, so it cannot disagree with the
 *  storage behind it.
 *
 *  @param[in] molecule Molecule carrying the conformer to read.
 *  @param[in] conformer_id Conformer index (-1 for the default conformer).
 *  @param[out] atom_major View spanning exactly the molecule's atoms.
 */
inline void read_conformer_into(RDKit::ROMol const &molecule, conformer_aos_t atom_major, int conformer_id = -1) {
    auto const &conformer = molecule.getConformer(conformer_id);
    for (std::uint32_t atom = 0; atom < atom_major.extent(0); ++atom) {
        auto const &position = conformer.getAtomPos(atom);
        atom_major(atom, 0) = static_cast<float>(position.x);
        atom_major(atom, 1) = static_cast<float>(position.y);
        atom_major(atom, 2) = static_cast<float>(position.z);
    }
}

/**
 *  @brief Set coordinates on a molecule from a flat interleaved array.
 *
 *  Used to inject GPU-generated coordinates back into RDKit for cross-validation.
 *
 *  @param[inout] molecule Molecule whose conformer coordinates will be overwritten.
 *  @param[in] coords Flat interleaved xyz coordinate array.
 *  @param[in] conformer_id Conformer index (-1 for the default conformer).
 */
inline void set_coords_interleaved(RDKit::RWMol &molecule, float const *coords, int conformer_id = -1) {
    auto &conformer = molecule.getConformer(conformer_id);
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        conformer.setAtomPos(atom_index, RDGeom::Point3D(static_cast<double>(coords[atom_index * 3 + 0]),
                                                         static_cast<double>(coords[atom_index * 3 + 1]),
                                                         static_cast<double>(coords[atom_index * 3 + 2])));
    }
}

/**
 *  @brief Evaluate MMFF94 energy at current coordinates without optimizing.
 *
 *  @param[inout] molecule Molecule with coordinates to evaluate.
 *  @param[in] conformer_id Conformer index (-1 for the default conformer).
 *  @return Energy in kcal/mol, or NaN if MMFF94 properties cannot be computed.
 */
inline double rdkit_mmff94_energy(RDKit::RWMol &molecule, int conformer_id = -1) {
    std::unique_ptr<RDKit::MMFF::MMFFMolProperties> mmff_properties(new RDKit::MMFF::MMFFMolProperties(molecule));
    if (!mmff_properties || !mmff_properties->isValid()) { return std::numeric_limits<double>::quiet_NaN(); }
    std::unique_ptr<ForceFields::ForceField> force_field(
        RDKit::MMFF::constructForceField(molecule, mmff_properties.get(), 100, conformer_id));
    if (!force_field) { return std::numeric_limits<double>::quiet_NaN(); }
    force_field->initialize();
    return force_field->calcEnergy();
}

/** How a `load_smiles_into` call ended; a named path that does not open is never an empty run. */
enum class smiles_load_t {
    /** The file opened and yielded at least one SMILES. */
    loaded_k,
    /** The path could not be opened at all. */
    missing_file_k,
    /** The file opened but held no SMILES line. */
    empty_file_k,
};

/** The sentence a `smiles_load_t` reads as, for the caller's own `FAIL <test_name>: <reason>`. */
inline std::string_view smiles_load_reason(smiles_load_t outcome) noexcept {
    switch (outcome) {
    case smiles_load_t::missing_file_k: return "cannot open corpus";
    case smiles_load_t::empty_file_k: return "corpus opened but yielded no SMILES";
    default: return "corpus loaded";
    }
}

/**
 *  @brief Load up to @p max_count SMILES strings from @p path into @p out, one per line, first field only.
 *
 *  Lines starting with `#` are skipped, and each kept line is truncated at its first run of
 *  whitespace, which drops a trailing name field along with any `\r`.
 *
 *  The outcome is returned rather than exited on, so the failure is reported by the caller under its
 *  own `test_name_k` and answers `grep '^FAIL test_'`; a helper that exits reports under whatever
 *  literal it was written with and is invisible to that filter.
 *
 *  @param[in] path File to read.
 *  @param[in] max_count Upper bound on how many lines to keep.
 *  @param[out] out Cleared, then filled with the loaded SMILES.
 *  @return Whether the file opened, and whether it held anything.
 */
inline smiles_load_t load_smiles_into(std::string const &path, std::size_t max_count, std::vector<std::string> &out) {
    out.clear();
    std::ifstream file(path);
    if (!file) return smiles_load_t::missing_file_k;
    std::string line;
    while (out.size() < max_count && std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::size_t const end = line.find_first_of(" \t\r");
        if (end != std::string::npos) line.resize(end);
        if (!line.empty()) out.push_back(std::move(line));
    }
    return out.empty() ? smiles_load_t::empty_file_k : smiles_load_t::loaded_k;
}

/** Whether a test was pointed at a corpus, and whether its command line made sense. */
enum class corpus_knob_t {
    /** Neither knob was given, so the built-in molecules are the whole corpus. */
    unset_k,
    /** A path was named, by `argv[1]` or by `SMALLFIELD_TEST_CORPUS`. */
    named_k,
    /** An argument could not be understood; the caller prints a usage line and exits 2. */
    refused_k,
};

/** The corpus a test walks, as resolved from its command line and environment. */
struct corpus_selection_t {
    corpus_knob_t knob = corpus_knob_t::unset_k;
    std::string path;
    std::size_t max_molecules = 0;
};

/**
 *  @brief Resolve the corpus knobs every corpus-driven test shares.
 *
 *  The path comes from `argv[1]`, falling back to `SMALLFIELD_TEST_CORPUS`; the cap comes from
 *  `argv[2]`, falling back to `SMALLFIELD_TEST_MAX_MOLS`, and finally to @p default_max_molecules.
 *  No corpus path is ever a literal in a test, so a binary that names one names it here.
 *
 *  @param[in] argc Argument count as `main` received it.
 *  @param[in] argv Argument vector as `main` received it.
 *  @param[in] default_max_molecules Cap applied when neither cap knob is set.
 *  @return The resolved corpus, or `refused_k` when a cap argument is not a number.
 */
inline corpus_selection_t select_corpus(int argc, char const *const *argv, std::size_t default_max_molecules) {
    corpus_selection_t selection;
    selection.max_molecules = default_max_molecules;

    char const *const path_from_environment = std::getenv("SMALLFIELD_TEST_CORPUS");
    if (argc > 1 && argv[1][0] != '\0') selection.path = argv[1];
    else if (path_from_environment != nullptr && path_from_environment[0] != '\0')
        selection.path = path_from_environment;
    if (!selection.path.empty()) selection.knob = corpus_knob_t::named_k;

    char const *const cap_from_environment = std::getenv("SMALLFIELD_TEST_MAX_MOLS");
    char const *cap_text = nullptr;
    if (argc > 2) cap_text = argv[2];
    else if (cap_from_environment != nullptr && cap_from_environment[0] != '\0') cap_text = cap_from_environment;
    if (cap_text != nullptr) {
        std::string_view const text {cap_text};
        std::size_t ceiling = 0;
        auto const [unparsed, error] = std::from_chars(text.data(), text.data() + text.size(), ceiling);
        if (error != std::errc {} || unparsed != text.data() + text.size() || ceiling == 0)
            selection.knob = corpus_knob_t::refused_k;
        else selection.max_molecules = ceiling;
    }
    return selection;
}

/**
 *  @brief Compare two floats within a tolerance that scales with the larger operand's magnitude.
 *
 *  @param[in] lhs First value.
 *  @param[in] rhs Second value.
 *  @param[in] tolerance Absolute tolerance below unit magnitude; scales with magnitude above it.
 *  @return True when the values agree within tolerance.
 */
inline bool approximately_equal(float lhs, float rhs, float tolerance = 1e-3f) {
    if (std::fabs(lhs - rhs) <= tolerance) return true;
    float const scale = std::max(1.0f, std::max(std::fabs(lhs), std::fabs(rhs)));
    return std::fabs(lhs - rhs) <= tolerance * scale;
}

/**
 *  @brief Standard set of drug-like test molecules covering a range of sizes.
 *
 *  @return Vector of SMILES strings for common test molecules.
 */
inline std::vector<std::string> standard_test_smiles() {
    return {
        "CCO",                                    // ethanol (9 atoms w/ H)
        "CC(=O)O",                                // acetic acid (8 atoms)
        "c1ccccc1",                               // benzene (12 atoms)
        "CC(=O)Oc1ccccc1C(=O)O",                  // aspirin (21 atoms)
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",             // caffeine (24 atoms)
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",        // ibuprofen (33 atoms)
        "CC(=O)Nc1ccc(O)cc1",                     // acetaminophen (20 atoms)
        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", // glucose (24 atoms)
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",    // testosterone (49 atoms)
        "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",       // pyrene (26 atoms)
    };
}

/**
 *  @brief Rebuild the packed N × N bond-hop matrix a single-molecule energy call expects.
 *
 *  The production pool carries no such matrix on the host — a warp BFS fills the device-only
 *  region from the bond list — so the harnesses that drive one molecule through the energy
 *  kernel derive it here from the same bond list, breadth-first from every source atom.
 *  Pairs with no bonded path keep 255, which the nonbonded sweep reads as unreachable.
 *
 *  The atom count is not a parameter: @p matrix carries its own extent, so it cannot disagree with the
 *  storage behind it.
 *
 *  @param[in] bonds Bond terms, each exposing `atom_i` and `atom_j`.
 *  @param[out] matrix Square view over caller-owned storage, filled row-major.
 */
inline void fill_bond_hops_into(custd::span<mmff94_bond_t const> bonds, hop_matrix_t matrix) {
    std::uint32_t const num_atoms = matrix.extent(0);
    for (std::uint32_t source = 0; source < num_atoms; ++source)
        for (std::uint32_t target = 0; target < num_atoms; ++target) matrix(source, target) = unreachable_hops_k;

    std::vector<std::uint8_t> queue(num_atoms, std::uint8_t {0});
    for (std::uint32_t source = 0; source < num_atoms; ++source) {
        matrix(source, source) = 0;
        std::uint32_t head = 0, tail = 0;
        queue[tail++] = static_cast<std::uint8_t>(source);
        while (head < tail) {
            std::uint8_t const current = queue[head++];
            auto const next_distance = static_cast<std::uint8_t>(matrix(source, current) + 1);
            for (auto const &bond : bonds) {
                std::uint8_t neighbor = unreachable_hops_k;
                if (bond.atom_i == current) neighbor = bond.atom_j;
                else if (bond.atom_j == current) neighbor = bond.atom_i;
                else continue;
                if (matrix(source, neighbor) > next_distance) {
                    matrix(source, neighbor) = next_distance;
                    queue[tail++] = neighbor;
                }
            }
        }
    }
}

} // namespace smallfield::tests
