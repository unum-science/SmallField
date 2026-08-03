/**
 *  @file test_shape_descriptors.cpp
 *  @brief Golden test: native USR / USRCAT match RDKit within tolerance, and
 *         fragment separation is a non-overlapping rigid layout.
 *
 *  For a spread of molecules we parse with the library's parse contract, embed
 *  one ETKDGv3 conformer (fixed seed), MMFF-optimize, then feed that conformer's
 *  float coordinates to both the native descriptors and RDKit's reference
 *  `Descriptors::USR` / `USRCAT`.  To compare like with like, the RDKit conformer
 *  is overwritten with the same float-rounded coordinates the native path sees,
 *  so any residual difference is accumulation order only.  Gate: max abs
 *  difference < 1e-4 on all 12 (USR) and 60 (USRCAT) elements.
 *
 *  Also checks that the acetate/sodium salt is two components and that
 *  `separate_fragments` pushes the two fragment centroids apart while leaving
 *  intra-fragment distances unchanged (rigid translation).
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <GraphMol/Descriptors/USRDescriptor.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/postprocess.hpp"
#include "test_helpers.hpp"

namespace sfp = smallfield::postprocess;

namespace {

// Parse with the same contract build_shape_context uses, add Hs, embed one
// ETKDGv3 conformer with a fixed seed, and MMFF-optimize it.
std::unique_ptr<RDKit::RWMol> prepare_mol(std::string const &smiles, int seed = 0xC0FFEE) {
    RDKit::SmilesParserParams params;
    params.sanitize = true;
    params.removeHs = false;
    params.strictCXSMILES = false;
    std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smiles, params));
    if (!mol) return nullptr;
    RDKit::MolOps::addHs(*mol);
    RDKit::DGeomHelpers::EmbedParameters embed_params = RDKit::DGeomHelpers::ETKDGv3;
    embed_params.randomSeed = seed;
    int const conformer_id = RDKit::DGeomHelpers::EmbedMolecule(*mol, embed_params);
    if (conformer_id < 0) return nullptr;
    RDKit::MMFF::MMFFOptimizeMolecule(*mol);
    return mol;
}

double max_abs_diff(std::vector<double> const &reference, double const *native, std::size_t count) {
    double worst = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        double const d = std::fabs(reference[i] - native[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

void validate_descriptors(std::string const &smiles) {
    std::unique_ptr<RDKit::RWMol> mol = prepare_mol(smiles);
    SF_TEST_ASSERT(mol != nullptr, ("prepare_mol failed for: " + smiles).c_str());

    std::uint32_t const num_atoms = mol->getNumAtoms();
    std::vector<float> coords = smallfield::tests::extract_coords_interleaved(*mol);

    // Overwrite the RDKit conformer with the float-rounded coordinates so the
    // reference computation and the native one operate on identical inputs.
    smallfield::tests::set_coords_interleaved(*mol, coords.data());

    sfp::molecule_shape_context_t const context = sfp::build_shape_context(smiles);
    SF_TEST_ASSERT(context.ok, ("build_shape_context failed for: " + smiles).c_str());
    SF_TEST_ASSERT(context.num_atoms == num_atoms, "num_atoms mismatch between context and RDKit mol");

    // USR.
    std::array<double, 12> native_usr {};
    sfp::usr_descriptor(std::span<float const>(coords.data(), coords.size()), num_atoms, native_usr);
    std::vector<double> rdkit_usr(12, 0.0);
    RDKit::Descriptors::USR(*mol, rdkit_usr);
    double const usr_deviation = max_abs_diff(rdkit_usr, native_usr.data(), 12);

    // USRCAT.
    std::array<double, 60> native_usrcat {};
    sfp::usrcat_descriptor(std::span<float const>(coords.data(), coords.size()), context, native_usrcat);
    std::vector<double> rdkit_usrcat(60, 0.0);
    std::vector<std::vector<unsigned int>> atom_ids;
    RDKit::Descriptors::USRCAT(*mol, rdkit_usrcat, atom_ids);
    double const usrcat_deviation = max_abs_diff(rdkit_usrcat, native_usrcat.data(), 60);

    double const tolerance = 1e-4;
    bool const pass = usr_deviation < tolerance && usrcat_deviation < tolerance;
    std::printf("  [%s] %-34s USR maxdev=%.3e  USRCAT maxdev=%.3e\n", pass ? "PASS" : "FAIL", smiles.c_str(),
                usr_deviation, usrcat_deviation);
    SF_TEST_ASSERT(usr_deviation < tolerance, ("USR deviation exceeds tolerance for: " + smiles).c_str());
    SF_TEST_ASSERT(usrcat_deviation < tolerance, ("USRCAT deviation exceeds tolerance for: " + smiles).c_str());
}

// Sum of pairwise distances within one component -- rigid translation invariant.
double intra_component_distance_sum(std::vector<float> const &coords, std::vector<std::uint8_t> const &label,
                                    std::uint8_t component) {
    std::vector<std::uint32_t> atoms;
    for (std::uint32_t atom_index = 0; atom_index < label.size(); ++atom_index)
        if (label[atom_index] == component) atoms.push_back(atom_index);
    double total = 0.0;
    for (std::size_t a = 0; a < atoms.size(); ++a)
        for (std::size_t b = a + 1; b < atoms.size(); ++b) {
            std::size_t const ia = static_cast<std::size_t>(atoms[a]) * 3u;
            std::size_t const ib = static_cast<std::size_t>(atoms[b]) * 3u;
            double const dx = static_cast<double>(coords[ia + 0]) - coords[ib + 0];
            double const dy = static_cast<double>(coords[ia + 1]) - coords[ib + 1];
            double const dz = static_cast<double>(coords[ia + 2]) - coords[ib + 2];
            total += std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    return total;
}

std::array<double, 3> component_centroid(std::vector<float> const &coords, std::vector<std::uint8_t> const &label,
                                         std::uint8_t component) {
    std::array<double, 3> centroid {0.0, 0.0, 0.0};
    std::uint32_t count = 0;
    for (std::uint32_t atom_index = 0; atom_index < label.size(); ++atom_index) {
        if (label[atom_index] != component) continue;
        std::size_t const base = static_cast<std::size_t>(atom_index) * 3u;
        centroid[0] += coords[base + 0];
        centroid[1] += coords[base + 1];
        centroid[2] += coords[base + 2];
        ++count;
    }
    if (count > 0) {
        centroid[0] /= count;
        centroid[1] /= count;
        centroid[2] /= count;
    }
    return centroid;
}

double centroid_distance(std::array<double, 3> const &a, std::array<double, 3> const &b) {
    double const dx = a[0] - b[0];
    double const dy = a[1] - b[1];
    double const dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void validate_fragment_separation() {
    std::string const salt = "CC(=O)[O-].[Na+]";
    std::unique_ptr<RDKit::RWMol> mol = prepare_mol(salt);
    SF_TEST_ASSERT(mol != nullptr, "prepare_mol failed for salt");

    // `separate_fragments` is a geometric utility over a multi-component context; the production
    // path strips counterions before it can ever see one, so the test builds the context itself.
    sfp::molecule_shape_context_t const context = sfp::build_shape_context(salt, smallfield::keep_all_fragments_k);
    SF_TEST_ASSERT(context.ok, "build_shape_context failed for salt");
    SF_TEST_ASSERT(context.num_components == 2, "salt should have exactly two components");

    std::vector<float> coords = smallfield::tests::extract_coords_interleaved(*mol);

    double const intra0_before = intra_component_distance_sum(coords, context.component_label, 0);
    double const intra1_before = intra_component_distance_sum(coords, context.component_label, 1);
    double const centroid_gap_before =
        centroid_distance(component_centroid(coords, context.component_label, 0),
                          component_centroid(coords, context.component_label, 1));

    sfp::separate_fragments(std::span<float>(coords.data(), coords.size()), context);

    double const intra0_after = intra_component_distance_sum(coords, context.component_label, 0);
    double const intra1_after = intra_component_distance_sum(coords, context.component_label, 1);
    double const centroid_gap_after =
        centroid_distance(component_centroid(coords, context.component_label, 0),
                          component_centroid(coords, context.component_label, 1));

    std::printf("  [salt] centroid gap %.3f -> %.3f  intra sums (%.4f,%.4f)->(%.4f,%.4f)\n", centroid_gap_before,
                centroid_gap_after, intra0_before, intra1_before, intra0_after, intra1_after);

    SF_TEST_ASSERT(centroid_gap_after > centroid_gap_before, "separation did not push fragment centroids apart");
    SF_TEST_NEAR(intra0_after, intra0_before, 1e-3, "component 0 intra-fragment distances changed");
    SF_TEST_NEAR(intra1_after, intra1_before, 1e-3, "component 1 intra-fragment distances changed");
}

} // namespace

int main() {
    std::printf("Validating native USR / USRCAT against RDKit...\n");

    validate_descriptors("CCO");
    validate_descriptors("c1ccccc1");
    validate_descriptors("CC(=O)Oc1ccccc1C(=O)O");        // aspirin
    validate_descriptors("Nc1ccc(cc1)S(=O)(=O)N");        // sulfanilamide: donors + acceptors
    validate_descriptors("OC(=O)CCc1cc(O)c(O)cc1");       // multiple donors/acceptors/aromatic
    validate_descriptors("CC(=O)Nc1ccc(O)cc1");           // paracetamol

    std::printf("Validating fragment separation...\n");
    validate_fragment_separation();

    std::printf("\nAll shape descriptor tests passed.\n");
    return 0;
}
