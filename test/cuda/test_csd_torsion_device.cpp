/**
 *  @file test/cuda/test_csd_torsion_device.cpp
 *  @brief Gate the device CSD torsion matcher against the host extractor, molecule for molecule.
 *
 *  The device kernel and `extract_etk_torsions_native` read the same flat descriptor and the same compiled
 *  pattern corpus, one through a device arena and one through the host pattern objects, so their torsion sets
 *  must agree exactly — same four atom indices in the same order, same six Fourier coefficients bit for bit.
 *  Anything less means the POD mirror of a pattern lost something on the way to the device.
 *
 *  Corpus: the built-in standards by default, so a fresh clone stays green; name a `.smi` file in `argv[1]` or in
 *  `SMALLFIELD_TEST_CORPUS`, colon-separated for several, for a wide sweep. A named path that does not open is a
 *  failure, never an empty run.
 */

#include <cstdint> // `std::uint32_t`, `std::uint8_t`
#include <cstdlib> // `std::exit`
#include <cstring> // `std::memcpy`

#include <chrono>      // `std::chrono::steady_clock`, `std::chrono::duration`
#include <memory>      // `std::unique_ptr`
#include <set>         // `std::set`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <tuple>       // `std::tuple`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/preprocess/csd_torsion_dispatcher.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

#include "test/cuda/test_csd_torsion_device_kernels.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_csd_torsion_device";

namespace smallfield::tests {

/** Molecules uploaded in one launch; keeps the device staging buffers comfortably small. */
constexpr std::uint32_t batch_capacity_k = 1024;

/** Output slots per molecule. A bond carries at most one torsion, so the bond capacity suffices. */
constexpr std::uint32_t torsion_stride_k = sft::max_bonds_k;

/** One torsion reduced to a comparable key: the four atoms in order plus the six coefficients. */
using torsion_key_t = std::tuple<std::uint8_t, std::uint8_t, std::uint8_t, std::uint8_t, std::uint32_t, std::uint32_t,
                                 std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>;

static torsion_key_t torsion_key_(sft::etk_torsion_descriptor_t const &torsion) {
    std::uint32_t bits[6];
    for (std::uint32_t term = 0; term < 6; ++term) std::memcpy(&bits[term], &torsion.fourier_coefficients[term], 4);
    return torsion_key_t {torsion.atom_indices[0],
                          torsion.atom_indices[1],
                          torsion.atom_indices[2],
                          torsion.atom_indices[3],
                          bits[0],
                          bits[1],
                          bits[2],
                          bits[3],
                          bits[4],
                          bits[5]};
}

/** Split a colon-separated corpus list into its paths.
 *  @param[in] value Colon-separated list.
 *  @return The individual paths. */
static std::vector<std::string> split_paths_(std::string_view value) {
    std::vector<std::string> paths;
    std::string current;
    for (char const character : value) {
        if (character == ':') {
            if (!current.empty()) paths.push_back(current);
            current.clear();
        }
        else { current.push_back(character); }
    }
    if (!current.empty()) paths.push_back(current);
    return paths;
}

/** Parse and hydrogenate one SMILES into a descriptor.
 *  @param[in] smiles Molecule to build.
 *  @param[out] descriptor Filled on success.
 *  @return True when the molecule parsed and fits the descriptor's capacity. */
static bool build_descriptor_from_smiles_(std::string const &smiles, sft::molecule_descriptor_t &descriptor) {
    std::unique_ptr<RDKit::RWMol> molecule;
    try {
        molecule.reset(RDKit::SmilesToMol(smiles));
    }
    catch (...) {
        return false;
    }
    if (!molecule) return false;
    try {
        RDKit::MolOps::addHs(*molecule);
    }
    catch (...) {
        return false;
    }
    if (molecule->getNumAtoms() > sft::max_atoms_per_molecule_k || molecule->getNumBonds() > sft::max_bonds_k)
        return false;
    return sft::build_descriptor(*molecule, descriptor);
}

} // namespace smallfield::tests

using sftest::batch_capacity_k;
using sftest::build_descriptor_from_smiles_;
using sftest::split_paths_;
using sftest::torsion_key_;
using sftest::torsion_key_t;
using sftest::torsion_stride_k;

int main(int argc, char **argv) {
    fmt::println("{}: device CSD torsion matcher vs host extractor", test_name_k);

    std::vector<std::string> smiles_list = sftest::standard_test_smiles();
    auto const selection = sftest::select_corpus(argc, argv, 100000);
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        fmt::println(stderr, "usage: {} [corpus.smi[:corpus.smi...]] [max_molecules]", argv[0]);
        return 2;
    }
    if (selection.knob == sftest::corpus_knob_t::named_k)
        for (auto const &path : split_paths_(selection.path)) {
            std::vector<std::string> loaded;
            auto const outcome = sftest::load_smiles_into(path, selection.max_molecules, loaded);
            if (outcome != sftest::smiles_load_t::loaded_k)
                return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), path);
            fmt::println("Corpus: {} ({} molecules)", path, fmt::group_digits(loaded.size()));
            for (auto &smiles : loaded) smiles_list.push_back(std::move(smiles));
        }
    else fmt::println("Corpus: builtin standards only; name one on the command line or in SMALLFIELD_TEST_CORPUS");

    std::size_t table_bytes = 0;
    void *table_handle = sftest::create_csd_torsion_device_table(table_bytes);
    sf_verify_(table_handle != nullptr, "device torsion table upload failed");
    fmt::println("Device pattern table: {} bytes", fmt::group_digits(table_bytes));

    std::vector<sft::molecule_descriptor_t> descriptors;
    std::vector<std::string> descriptor_smiles;
    std::vector<sft::etk_torsion_descriptor_t> device_torsions(static_cast<std::size_t>(batch_capacity_k) *
                                                               torsion_stride_k);
    std::vector<std::uint32_t> device_counts(batch_capacity_k);
    std::vector<sft::etk_torsion_descriptor_t> host_torsions;
    descriptors.reserve(batch_capacity_k);
    descriptor_smiles.reserve(batch_capacity_k);

    std::size_t compared = 0;
    std::size_t mismatches = 0;
    std::size_t total_torsions = 0;
    // Correctness is the gate, but the port only earns its place if the device is faster than the
    // host it replaces, so both sides are timed over the same molecules. The device figure covers
    // the launch and the synchronize; the host figure is one thread, which is what the preprocess
    // fan-out gives each molecule.
    double host_microseconds = 0.0;
    double device_microseconds = 0.0;

    auto flush_batch = [&]() {
        if (descriptors.empty()) return;
        std::uint32_t const count = static_cast<std::uint32_t>(descriptors.size());
        auto const device_start = std::chrono::steady_clock::now();
        bool const launched = sftest::run_csd_torsion_device_batch(
            table_handle, descriptors.data(), count, torsion_stride_k, device_torsions.data(), device_counts.data());
        device_microseconds +=
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - device_start).count();
        sf_verify_(launched, "device torsion kernel failed");

        for (std::uint32_t molecule = 0; molecule < count; ++molecule) {
            auto const host_start = std::chrono::steady_clock::now();
            sfp::extract_etk_torsions_native(descriptors[molecule], host_torsions);
            host_microseconds +=
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - host_start).count();
            std::set<torsion_key_t> expected;
            for (auto const &torsion : host_torsions) expected.insert(torsion_key_(torsion));

            std::uint32_t const produced_count = device_counts[molecule];
            sf_verify_(produced_count <= torsion_stride_k, "device torsion count overflowed the output slots");
            std::set<torsion_key_t> produced;
            sft::etk_torsion_descriptor_t const *slots = device_torsions.data() +
                                                         static_cast<std::size_t>(molecule) * torsion_stride_k;
            for (std::uint32_t entry = 0; entry < produced_count; ++entry) produced.insert(torsion_key_(slots[entry]));

            ++compared;
            total_torsions += expected.size();
            if (expected == produced) continue;
            ++mismatches;
            if (mismatches <= 5) {
                std::fflush(stdout);
                fmt::println(stderr, "MISMATCH csd_torsion smiles={} field=torsion_count host={} device={}",
                             descriptor_smiles[molecule], expected.size(), produced.size());
            }
        }
        descriptors.clear();
        descriptor_smiles.clear();
    };

    for (auto const &smiles : smiles_list) {
        sft::molecule_descriptor_t descriptor;
        if (!build_descriptor_from_smiles_(smiles, descriptor)) continue;
        descriptors.push_back(descriptor);
        descriptor_smiles.push_back(smiles);
        if (descriptors.size() == batch_capacity_k) flush_batch();
    }
    flush_batch();

    sftest::destroy_csd_torsion_device_table(table_handle);

    fmt::println("Molecules compared: {}, host torsions: {}, mismatches: {}", fmt::group_digits(compared),
                 fmt::group_digits(total_torsions), fmt::group_digits(mismatches));
    if (compared > 0) {
        double const host_per = host_microseconds / static_cast<double>(compared);
        double const device_per = device_microseconds / static_cast<double>(compared);
        fmt::println("Matching cost per molecule: host {:.2f} us (1 thread), device {:.2f} us (launch + sync), {:.2f}x",
                     host_per, device_per, device_per > 0.0 ? host_per / device_per : 0.0);
    }
    sf_verify_(compared > 0, "no molecules were compared");
    sf_verify_(mismatches == 0, "device torsion sets diverge from the host extractor");
    return sftest::report_pass(test_name_k);
}
