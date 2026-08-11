/**
 *  @file test/test_rdkit_reference_helpers.hpp
 *  @brief Shared utilities for C++ ETKDG tests.
 *
 *  Provides molecule loading, coordinate extraction/injection, and RDKit MMFF94 energy evaluation for
 *  cross-referencing GPU-generated conformers against RDKit's reference implementation.
 */

#pragma once

#include <algorithm> // `std::max`
#include <cmath>     // `std::fabs`
#include <cstdint>   // `std::uint32_t`, `std::uint8_t`
#include <cstdio>    // `fprintf`, `stderr`
#include <cstdlib>   // `std::exit`

#include <fstream> // `std::ifstream`
#include <limits>  // `std::numeric_limits`
#include <memory>  // `std::unique_ptr`
#include <string>  // `std::string`
#include <vector>  // `std::vector`

#include <DistGeom/BoundsMatrix.h>
#include <DistGeom/TriangleSmooth.h>
#include <ForceField/ForceField.h>
#include <Geometry/point.h>
#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

/**
 *  @brief Fail the test with a message and the source location when @p condition is false.
 *
 *  Carries the `sf_` prefix so it cannot collide with RDKit's own `TEST_ASSERT`, which keeps the
 *  tests free of any framework dependency.
 */
#define sf_verify_(condition, message)                                                     \
    do {                                                                                   \
        if (!(condition)) {                                                                \
            std::fprintf(stderr, "FAIL: %s\n  at %s:%d\n", (message), __FILE__, __LINE__); \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

#define sf_verify_near_(actual, expected, tolerance, message)                                                    \
    sf_verify_(                                                                                                  \
        std::fabs(static_cast<double>(actual) - static_cast<double>(expected)) < static_cast<double>(tolerance), \
        (message))

namespace smallfield::tests {

/**
 *  @brief Create a molecule from SMILES with explicit hydrogens added.
 *
 *  @param[in] smiles SMILES string to parse.
 *  @return Owning pointer to the molecule, or nullptr on parse failure.
 */
inline std::unique_ptr<RDKit::RWMol> mol_from_smiles(std::string const &smiles) {
    std::unique_ptr<RDKit::RWMol> molecule(RDKit::SmilesToMol(smiles));
    if (!molecule) {
        std::fprintf(stderr, "Failed to parse SMILES: %s\n", smiles.c_str());
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
 *  @brief Extract 3D coordinates as a flat interleaved array [x0,y0,z0, x1,y1,z1, ...].
 *
 *  @param[in] molecule Molecule with at least one conformer.
 *  @param[in] conformer_id Conformer index (-1 for the default conformer).
 *  @return Flat float vector of interleaved xyz coordinates.
 */
inline std::vector<float> extract_coords_interleaved(RDKit::ROMol const &molecule, int conformer_id = -1) {
    auto const &conformer = molecule.getConformer(conformer_id);
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    std::vector<float> coords(num_atoms * 3);
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        auto const &position = conformer.getAtomPos(atom_index);
        coords[atom_index * 3 + 0] = static_cast<float>(position.x);
        coords[atom_index * 3 + 1] = static_cast<float>(position.y);
        coords[atom_index * 3 + 2] = static_cast<float>(position.z);
    }
    return coords;
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

/**
 *  @brief Get the smoothed bounds matrix from RDKit as a flat N × N double array.
 *
 *  @param[inout] molecule Molecule to compute bounds for.
 *  @return Row-major N × N flat double vector (upper triangle = upper bounds, lower = lower bounds).
 */
inline std::vector<double> rdkit_bounds_matrix(RDKit::RWMol &molecule) {
    std::uint32_t const num_atoms = molecule.getNumAtoms();
    auto *bounds_raw = new DistGeom::BoundsMatrix(num_atoms);
    DistGeom::BoundsMatPtr bounds_matrix(bounds_raw);
    RDKit::DGeomHelpers::initBoundsMat(bounds_raw);
    RDKit::DGeomHelpers::setTopolBounds(molecule, bounds_matrix);
    DistGeom::triangleSmoothBounds(bounds_raw);

    // flatten to row-major N × N
    std::vector<double> flat(num_atoms * num_atoms, 0.0);
    for (std::uint32_t row = 0; row < num_atoms; ++row) {
        for (std::uint32_t column = row + 1; column < num_atoms; ++column) {
            flat[row * num_atoms + column] = bounds_matrix->getUpperBound(row, column); // upper
            flat[column * num_atoms + row] = bounds_matrix->getLowerBound(row, column); // lower
        }
    }
    return flat;
}

/** What happens when a SMILES corpus file passed to `load_smiles` cannot be opened. */
enum class smiles_missing_file_t {
    returns_empty_k,
    aborts_k,
};

/**
 *  @brief Load up to @p max_count SMILES strings from @p path, one per line, first field only.
 *
 *  Lines starting with `#` are skipped, and each kept line is truncated at its first run of
 *  whitespace, which drops a trailing name field along with any `\r`.
 *
 *  @param[in] path File to read.
 *  @param[in] max_count Upper bound on how many lines to keep.
 *  @param[in] on_missing Whether a missing @p path aborts the process or yields an empty list.
 *  @return The loaded SMILES; empty when the file cannot be opened and @p on_missing allows it.
 */
inline std::vector<std::string> load_smiles(std::string const &path, std::size_t max_count,
                                            smiles_missing_file_t on_missing = smiles_missing_file_t::returns_empty_k) {
    std::vector<std::string> smiles_list;
    std::ifstream file(path);
    if (!file) {
        if (on_missing == smiles_missing_file_t::aborts_k) {
            std::fprintf(stderr, "FAIL: cannot open corpus %s\n", path.c_str());
            std::exit(1);
        }
        return smiles_list;
    }
    std::string line;
    while (smiles_list.size() < max_count && std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::size_t const end = line.find_first_of(" \t\r");
        if (end != std::string::npos) line.resize(end);
        if (!line.empty()) smiles_list.push_back(std::move(line));
    }
    return smiles_list;
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
 *  @param[in] bonds Bond terms, each exposing `atom_i` and `atom_j`.
 *  @param[in] bond_count Number of bonds.
 *  @param[in] num_atoms Atom count; the matrix is @p num_atoms squared entries.
 *  @param[out] matrix Resized to @p num_atoms squared and filled row-major.
 */
template <typename bond_>
inline void fill_bond_hops(bond_ const *bonds, std::uint32_t bond_count, std::uint32_t num_atoms,
                           std::vector<std::uint8_t> &matrix) {
    matrix.assign(static_cast<std::size_t>(num_atoms) * num_atoms, std::uint8_t {255});
    std::vector<std::uint8_t> queue(num_atoms, std::uint8_t {0});
    for (std::uint32_t source = 0; source < num_atoms; ++source) {
        std::uint8_t *const row = matrix.data() + static_cast<std::size_t>(source) * num_atoms;
        row[source] = 0;
        std::uint32_t head = 0, tail = 0;
        queue[tail++] = static_cast<std::uint8_t>(source);
        while (head < tail) {
            std::uint8_t const current = queue[head++];
            std::uint8_t const next_distance = static_cast<std::uint8_t>(row[current] + 1);
            for (std::uint32_t bond = 0; bond < bond_count; ++bond) {
                std::uint8_t neighbor = 255;
                if (bonds[bond].atom_i == current) neighbor = bonds[bond].atom_j;
                else if (bonds[bond].atom_j == current) neighbor = bonds[bond].atom_i;
                else continue;
                if (row[neighbor] > next_distance) {
                    row[neighbor] = next_distance;
                    queue[tail++] = neighbor;
                }
            }
        }
    }
}

} // namespace smallfield::tests
