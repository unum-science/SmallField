/**
 *  @file test/postprocess/test_shape_descriptors.cpp
 *  @brief Confirm native USR / USRCAT match RDKit within tolerance, the suite's golden test.
 *
 *  Fragment separation is a non-overlapping rigid layout. For a spread of molecules, each is parsed with the
 *  library's parse contract, embedded as one ETKDGv3 conformer with a fixed seed, MMFF-optimized, then that
 *  conformer's float coordinates feed both the native descriptors and RDKit's reference `Descriptors::USR` /
 *  `USRCAT`. To compare like with like, the RDKit conformer is overwritten with the same float-rounded coordinates
 *  the native path sees, so any residual difference is accumulation order only. Gate: max abs difference < 1e-4 on
 *  all 12 USR elements and all 60 USRCAT elements.
 *
 *  Also checks that the acetate/sodium salt is two components and that `separate_fragments` pushes the two fragment
 *  centroids apart while leaving intra-fragment distances unchanged — a rigid translation.
 */

#include <cmath>   // `std::fabs`, `std::sqrt`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <memory>      // `std::unique_ptr`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`

#include <GraphMol/Descriptors/USRDescriptor.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/postprocess/fragment_separation.hpp"
#include "smallfield/postprocess/shape_descriptors.hpp"

#include "test/test_rdkit_reference_helpers.hpp"

namespace sfp = smallfield::postprocess;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_shape_descriptors";

namespace smallfield::tests {

/**
 *  @brief Parse with the same contract `build_shape_context` uses, add Hs, embed one
 *  ETKDGv3 conformer with a fixed seed, and MMFF-optimize it.
 */
static std::unique_ptr<RDKit::RWMol> prepare_mol_(std::string const &smiles, int seed = 0xC0FFEE) {
    RDKit::SmilesParserParams params;
    params.sanitize = true;
    params.removeHs = false;
    params.strictCXSMILES = false;
    std::unique_ptr<RDKit::RWMol> molecule(RDKit::SmilesToMol(smiles, params));
    if (!molecule) return nullptr;
    RDKit::MolOps::addHs(*molecule);
    RDKit::DGeomHelpers::EmbedParameters embed_params = RDKit::DGeomHelpers::ETKDGv3;
    embed_params.randomSeed = seed;
    int const conformer_id = RDKit::DGeomHelpers::EmbedMolecule(*molecule, embed_params);
    if (conformer_id < 0) return nullptr;
    RDKit::MMFF::MMFFOptimizeMolecule(*molecule);
    return molecule;
}

static double max_abs_diff_(std::span<double const> reference, std::span<double const> native) {
    double worst = 0.0;
    for (std::size_t element = 0; element < native.size(); ++element) {
        double const difference = std::fabs(reference[element] - native[element]);
        if (difference > worst) worst = difference;
    }
    return worst;
}

static void validate_descriptors_(std::string const &smiles) {
    std::unique_ptr<RDKit::RWMol> molecule = prepare_mol_(smiles);
    sf_verify_(molecule != nullptr, ("prepare_mol failed for: " + smiles).c_str());

    std::uint32_t const num_atoms = molecule->getNumAtoms();
    std::vector<float> coords = smallfield::tests::extract_coords_interleaved(*molecule);

    // Overwrite the RDKit conformer with the float-rounded coordinates so the
    // reference computation and the native one operate on identical inputs.
    smallfield::tests::set_coords_interleaved(*molecule, coords.data());

    sfp::molecule_shape_context_t const context = sfp::build_shape_context(smiles);
    sf_verify_(context.ok, ("build_shape_context failed for: " + smiles).c_str());
    sf_verify_(context.num_atoms == num_atoms, "num_atoms mismatch between context and RDKit mol");

    // USR.
    std::array<double, 12> native_usr {};
    sfp::usr_descriptor(std::span<float const>(coords.data(), coords.size()), num_atoms, native_usr);
    std::vector<double> rdkit_usr(12, 0.0);
    RDKit::Descriptors::USR(*molecule, rdkit_usr);
    double const usr_deviation = max_abs_diff_(rdkit_usr, native_usr);

    // USRCAT.
    std::array<double, 60> native_usrcat {};
    sfp::usrcat_descriptor(std::span<float const>(coords.data(), coords.size()), context, native_usrcat);
    std::vector<double> rdkit_usrcat(60, 0.0);
    std::vector<std::vector<unsigned int>> atom_ids;
    RDKit::Descriptors::USRCAT(*molecule, rdkit_usrcat, atom_ids);
    double const usrcat_deviation = max_abs_diff_(rdkit_usrcat, native_usrcat);

    double const tolerance = 1e-4;
    fmt::println("  NOTE shape_descriptors smiles={:<34} usr_max_deviation={:.3g} usrcat_max_deviation={:.3g}", smiles,
                 usr_deviation, usrcat_deviation);
    sf_verify_(usr_deviation < tolerance, ("USR deviation exceeds tolerance for: " + smiles).c_str());
    sf_verify_(usrcat_deviation < tolerance, ("USRCAT deviation exceeds tolerance for: " + smiles).c_str());
}

/** Sum of pairwise distances within one component — rigid translation invariant. */
static double intra_component_distance_sum_(std::vector<float> const &coords, std::vector<std::uint8_t> const &label,
                                            std::uint8_t component) {
    std::vector<std::uint32_t> atoms;
    for (std::uint32_t atom_index = 0; atom_index < label.size(); ++atom_index)
        if (label[atom_index] == component) atoms.push_back(atom_index);
    double total = 0.0;
    for (std::size_t first = 0; first < atoms.size(); ++first)
        for (std::size_t second = first + 1; second < atoms.size(); ++second) {
            std::size_t const first_base = static_cast<std::size_t>(atoms[first]) * 3u;
            std::size_t const second_base = static_cast<std::size_t>(atoms[second]) * 3u;
            double const delta_x = static_cast<double>(coords[first_base + 0]) - coords[second_base + 0];
            double const delta_y = static_cast<double>(coords[first_base + 1]) - coords[second_base + 1];
            double const delta_z = static_cast<double>(coords[first_base + 2]) - coords[second_base + 2];
            total += std::sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
        }
    return total;
}

static std::array<double, 3> component_centroid_(std::vector<float> const &coords,
                                                 std::vector<std::uint8_t> const &label, std::uint8_t component) {
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

static double centroid_distance_(std::array<double, 3> const &first, std::array<double, 3> const &second) {
    double const delta_x = first[0] - second[0];
    double const delta_y = first[1] - second[1];
    double const delta_z = first[2] - second[2];
    return std::sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
}

static void validate_fragment_separation_() {
    std::string const salt = "CC(=O)[O-].[Na+]";
    std::unique_ptr<RDKit::RWMol> molecule = prepare_mol_(salt);
    sf_verify_(molecule != nullptr, "prepare_mol failed for salt");

    // `separate_fragments` is a geometric utility over a multi-component context; the production
    // path strips counterions before it can ever see one, so the test builds the context itself.
    sfp::molecule_shape_context_t const context = sfp::build_shape_context(salt, smallfield::keep_all_fragments_k);
    sf_verify_(context.ok, "build_shape_context failed for salt");
    sf_verify_(context.num_components == 2, "salt should have exactly two components");

    std::vector<float> coords = smallfield::tests::extract_coords_interleaved(*molecule);

    double const intra0_before = intra_component_distance_sum_(coords, context.component_label, 0);
    double const intra1_before = intra_component_distance_sum_(coords, context.component_label, 1);
    double const centroid_gap_before = centroid_distance_(component_centroid_(coords, context.component_label, 0),
                                                          component_centroid_(coords, context.component_label, 1));

    sfp::separate_fragments(std::span<float>(coords.data(), coords.size()), context);

    double const intra0_after = intra_component_distance_sum_(coords, context.component_label, 0);
    double const intra1_after = intra_component_distance_sum_(coords, context.component_label, 1);
    double const centroid_gap_after = centroid_distance_(component_centroid_(coords, context.component_label, 0),
                                                         component_centroid_(coords, context.component_label, 1));

    fmt::println("  NOTE fragment_separation subject=salt centroid_gap_before={:.3f} centroid_gap_after={:.3f} "
                 "intra_sums_before=({:.4f},{:.4f}) intra_sums_after=({:.4f},{:.4f})",
                 centroid_gap_before, centroid_gap_after, intra0_before, intra1_before, intra0_after, intra1_after);

    sf_verify_(centroid_gap_after > centroid_gap_before, "separation did not push fragment centroids apart");
    sf_verify_near_(intra0_after, intra0_before, 1e-3, "component 0 intra-fragment distances changed");
    sf_verify_near_(intra1_after, intra1_before, 1e-3, "component 1 intra-fragment distances changed");
}

} // namespace smallfield::tests

int main() {
    fmt::println("test_shape_descriptors: native USR / USRCAT vs RDKit, then fragment separation");

    smallfield::tests::validate_descriptors_("CCO");
    smallfield::tests::validate_descriptors_("c1ccccc1");
    smallfield::tests::validate_descriptors_("CC(=O)Oc1ccccc1C(=O)O");  // aspirin
    smallfield::tests::validate_descriptors_("Nc1ccc(cc1)S(=O)(=O)N");  // sulfanilamide: donors + acceptors
    smallfield::tests::validate_descriptors_("OC(=O)CCc1cc(O)c(O)cc1"); // multiple donors/acceptors/aromatic
    smallfield::tests::validate_descriptors_("CC(=O)Nc1ccc(O)cc1");     // paracetamol

    fmt::println("Validating fragment separation...");
    smallfield::tests::validate_fragment_separation_();

    return sftest::report_pass(test_name_k);
}
