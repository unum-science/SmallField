/**
 *  @file test/preprocess/test_element_gate.cpp
 *  @brief Validate the host-side element-validation gate in `prepare_molecule_into`.
 *
 *  Molecules whose atoms fall outside the MMFF94 parameter set (Y, lanthanides, transition metals, etc.) must be
 *  rejected here: left unchecked they reach `etkdg_pipeline_warp_kernel` and go out of bounds, poisoning every
 *  subsequent CUDA call on every device. These tests assert each known rejection path returns the right
 *  `conformer_status_t` so the caller can route to CPU fallback or surface a precise error to the user.
 */

#include <cstdlib> // `std::exit`

#include <string>      // `std::string`
#include <string_view> // `std::string_view`

#include "smallfield/api/conformer_status.hpp"
#include "smallfield/preprocess/etkdg_prepare_batch.hpp"

#include "test/test_helpers.hpp"

namespace sfapi = smallfield::api;
namespace sfprep = smallfield::preprocess;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_element_gate";

static void expect_status(std::string const &smiles, sfapi::conformer_status_t expected, std::string_view label) {
    sfprep::etkdg_prepared_molecule_t scratch {};
    auto const status = sfprep::prepare_molecule_into(smiles, scratch);
    if (status != expected) {
        [[maybe_unused]] int const exit_code = sftest::report_fail(
            test_name_k, "{} smiles={} expected_status={} observed_status={}", label, smiles,
            static_cast<unsigned>(expected), static_cast<unsigned>(status));
        std::exit(1);
    }
    unsigned const expected_atoms_dropped = expected != sfapi::conformer_status_t::success_k ? 1u : 0u;
    unsigned const observed_atoms_dropped = scratch.num_atoms == 0 ? 1u : 0u;
    if (expected_atoms_dropped != observed_atoms_dropped) {
        [[maybe_unused]] int const exit_code = sftest::report_fail(
            test_name_k, "{} smiles={} num_atoms={} expected_dropped={} observed_dropped={}", label, smiles,
            scratch.num_atoms, expected_atoms_dropped, observed_atoms_dropped);
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

    return sftest::report_pass(test_name_k);
}
