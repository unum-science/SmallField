/**
 *  @file test/preprocess/test_element_gate.cpp
 *  @brief Validate the host-side element-validation gate in `prepare_molecule_into`.
 *
 *  Molecules whose atoms fall outside the MMFF94 parameter set (Y, lanthanides, transition metals, etc.) must be
 *  rejected here: left unchecked they reach `etkdg_pipeline_warp_kernel` and go out of bounds, poisoning every
 *  subsequent CUDA call on every device. These tests assert each known rejection path returns the right
 *  `conformer_status_t` so the caller can route to CPU fallback or surface a precise error to the user.
 */

#include <cstdio>  // `fprintf`, `stderr`
#include <cstdlib> // `std::exit`

#include <string> // `std::string`

#include "smallfield/api/conformer_status.hpp"
#include "smallfield/preprocess/etkdg_prepare_batch.hpp"

#include "test/test_rdkit_reference_helpers.hpp"

namespace sfapi = smallfield::api;
namespace sfprep = smallfield::preprocess;

static void expect_status(std::string const &smiles, sfapi::conformer_status_t expected, char const *label) {
    sfprep::etkdg_prepared_molecule_t scratch {};
    auto const status = sfprep::prepare_molecule_into(smiles, scratch);
    if (status != expected) {
        std::fprintf(stderr, "FAIL [%s]: SMILES=%s expected status=%u got=%u\n", label, smiles.c_str(),
                     static_cast<unsigned>(expected), static_cast<unsigned>(status));
        std::exit(1);
    }
    bool const expect_drop = (expected != sfapi::conformer_status_t::success_k);
    bool const did_drop = (scratch.num_atoms == 0);
    if (expect_drop != did_drop) {
        std::fprintf(stderr, "FAIL [%s]: SMILES=%s num_atoms=%u (expected drop=%d)\n", label, smiles.c_str(),
                     scratch.num_atoms, static_cast<int>(expect_drop));
        std::exit(1);
    }
}

int main() {
    // Happy path — drug-like organic, all MMFF94-supported elements.
    expect_status("CCO", sfapi::conformer_status_t::success_k, "ethanol");
    expect_status("c1ccccc1", sfapi::conformer_status_t::success_k, "benzene");
    expect_status("CC(=O)Oc1ccccc1C(=O)O", sfapi::conformer_status_t::success_k, "aspirin");

    // Bug-report repro — yttrium (Z=39) is not in the MMFF94 atom-type table.
    expect_status("CC[Y]CC", sfapi::conformer_status_t::unsupported_element_k, "yttrium-bracketed");

    // Other transition metals and lanthanides — no MMFF94 type.
    expect_status("[Fe]", sfapi::conformer_status_t::unsupported_element_k, "lone-iron");
    expect_status("[La]", sfapi::conformer_status_t::unsupported_element_k, "lanthanum");

    // Atom-count cap — single-warp pipeline only handles ≤128 atoms.
    expect_status(std::string(200, 'C'), sfapi::conformer_status_t::too_many_atoms_k, "long-alkane");

    // SMILES parse failure — clearly malformed input.
    expect_status("not-a-smiles-string-((", sfapi::conformer_status_t::invalid_smiles_k, "garbage");

    std::printf("test_element_gate: PASS\n");
    return 0;
}
