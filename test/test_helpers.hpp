/**
 *  @file test_helpers.hpp
 *  @brief Shared utilities for C++ ETKDG tests.
 *
 *  Provides molecule loading, coordinate extraction/injection,
 *  and RDKit MMFF94 energy evaluation for cross-referencing
 *  GPU-generated conformers against RDKit's reference implementation.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

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

namespace smallfield {
namespace test {

// simple assertion macros (no test framework dependency)
// prefixed with SF_ to avoid collision with RDKit's TEST_ASSERT
#define SF_TEST_ASSERT(condition, message)                                                 \
    do {                                                                                   \
        if (!(condition)) {                                                                \
            std::fprintf(stderr, "FAIL: %s\n  at %s:%d\n", (message), __FILE__, __LINE__); \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

#define SF_TEST_NEAR(actual, expected, tolerance, message)                                                       \
    SF_TEST_ASSERT(                                                                                              \
        std::fabs(static_cast<double>(actual) - static_cast<double>(expected)) < static_cast<double>(tolerance), \
        (message))

/**
 *  @brief Create a molecule from SMILES with explicit hydrogens added.
 *
 *  @param[in] smiles SMILES string to parse.
 *  @return Owning pointer to the molecule, or nullptr on parse failure.
 */
inline std::unique_ptr<RDKit::RWMol> mol_from_smiles(std::string const &smiles) {
    std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smiles));
    if (!mol) {
        std::fprintf(stderr, "Failed to parse SMILES: %s\n", smiles.c_str());
        return nullptr;
    }
    RDKit::MolOps::addHs(*mol);
    return mol;
}

/**
 *  @brief Embed a molecule with ETKDGv3.
 *
 *  @param[in,out] mol Molecule to embed (modified in place with 3D coordinates).
 *  @param[in] seed Random seed for the embedding.
 *  @return True on success.
 */
inline bool embed_molecule(RDKit::RWMol &mol, int seed = 42) {
    RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
    params.randomSeed = seed;
    int conformer_id = RDKit::DGeomHelpers::EmbedMolecule(mol, params);
    return conformer_id >= 0;
}

/**
 *  @brief Extract 3D coordinates as a flat interleaved array [x0,y0,z0, x1,y1,z1, ...].
 *
 *  @param[in] mol Molecule with at least one conformer.
 *  @param[in] conformer_id Conformer index (-1 for the default conformer).
 *  @return Flat float vector of interleaved xyz coordinates.
 */
inline std::vector<float> extract_coords_interleaved(RDKit::ROMol const &mol, int conformer_id = -1) {
    auto const &conformer = mol.getConformer(conformer_id);
    std::uint32_t const num_atoms = mol.getNumAtoms();
    std::vector<float> coords(num_atoms * 3);
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        auto const &position = conformer.getAtomPos(i);
        coords[i * 3 + 0] = static_cast<float>(position.x);
        coords[i * 3 + 1] = static_cast<float>(position.y);
        coords[i * 3 + 2] = static_cast<float>(position.z);
    }
    return coords;
}

/**
 *  @brief Set coordinates on a molecule from a flat interleaved array.
 *
 *  Used to inject GPU-generated coordinates back into RDKit for cross-validation.
 *
 *  @param[in,out] mol Molecule whose conformer coordinates will be overwritten.
 *  @param[in] coords Flat interleaved xyz coordinate array.
 *  @param[in] conformer_id Conformer index (-1 for the default conformer).
 */
inline void set_coords_interleaved(RDKit::RWMol &mol, float const *coords, int conformer_id = -1) {
    auto &conformer = mol.getConformer(conformer_id);
    std::uint32_t const num_atoms = mol.getNumAtoms();
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        conformer.setAtomPos(
            i, RDGeom::Point3D(static_cast<double>(coords[i * 3 + 0]), static_cast<double>(coords[i * 3 + 1]),
                               static_cast<double>(coords[i * 3 + 2])));
    }
}

/**
 *  @brief Evaluate MMFF94 energy at current coordinates without optimizing.
 *
 *  @param[in,out] mol Molecule with coordinates to evaluate.
 *  @param[in] conformer_id Conformer index (-1 for the default conformer).
 *  @return Energy in kcal/mol, or NaN if MMFF94 properties cannot be computed.
 */
inline double rdkit_mmff94_energy(RDKit::RWMol &mol, int conformer_id = -1) {
    std::unique_ptr<RDKit::MMFF::MMFFMolProperties> mmff_properties(new RDKit::MMFF::MMFFMolProperties(mol));
    if (!mmff_properties || !mmff_properties->isValid()) { return std::numeric_limits<double>::quiet_NaN(); }
    std::unique_ptr<ForceFields::ForceField> force_field(
        RDKit::MMFF::constructForceField(mol, mmff_properties.get(), 100, conformer_id));
    if (!force_field) { return std::numeric_limits<double>::quiet_NaN(); }
    force_field->initialize();
    return force_field->calcEnergy();
}

/**
 *  @brief Get the smoothed bounds matrix from RDKit as a flat N x N double array.
 *
 *  @param[in,out] mol Molecule to compute bounds for.
 *  @return Row-major N x N flat double vector (upper triangle = upper bounds, lower = lower bounds).
 */
inline std::vector<double> rdkit_bounds_matrix(RDKit::RWMol &mol) {
    std::uint32_t const num_atoms = mol.getNumAtoms();
    auto *bounds_raw = new DistGeom::BoundsMatrix(num_atoms);
    DistGeom::BoundsMatPtr bounds_matrix(bounds_raw);
    RDKit::DGeomHelpers::initBoundsMat(bounds_raw);
    RDKit::DGeomHelpers::setTopolBounds(mol, bounds_matrix);
    DistGeom::triangleSmoothBounds(bounds_raw);

    // flatten to row-major N x N
    std::vector<double> flat(num_atoms * num_atoms, 0.0);
    for (std::uint32_t i = 0; i < num_atoms; ++i) {
        for (std::uint32_t j = i + 1; j < num_atoms; ++j) {
            flat[i * num_atoms + j] = bounds_matrix->getUpperBound(i, j); // upper
            flat[j * num_atoms + i] = bounds_matrix->getLowerBound(i, j); // lower
        }
    }
    return flat;
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

} // namespace test
} // namespace smallfield
