/**
 *  @file test/cuda/test_bond_distance.cpp
 *  @brief Gate byte-for-byte parity between the device hop-count BFS and the host all-pairs BFS.
 *
 *  A warp BFS rebuilds the N × N hop matrix the MMFF nonbonded sweep reads, on the device, from the batch's bond
 *  list; the host builds the same matrix from the descriptor's CSR adjacency. Two different traversals over two
 *  different encodings of the same graph, so the check is exact equality over every byte of every molecule — a
 *  single differing entry silently changes which pairs the sweep treats as 1-4 or as excluded.
 *
 *  Corpus: `SMALLFIELD_TEST_CORPUS=/path/to.smi` streams real molecules, capped by `SMALLFIELD_TEST_MAX_MOLS`
 *  (default 300). Absent it the built-in list runs, which carries its own multi-fragment salts so the 255
 *  unreachable sentinel is covered either way.
 */

#include <cstdio>  // `fprintf`, `stderr`
#include <cstdlib> // `std::getenv`

#include <fstream>     // `std::ifstream`
#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`, `std::getline`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

#include "test/cuda/test_bond_distance_kernels.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

namespace smallfield::tests {

/** Load up to @p max_count SMILES strings from @p path, one per line, first field only. */
static std::vector<std::string> load_smiles_(char const *path, std::size_t max_count) {
    std::vector<std::string> smiles_list;
    std::ifstream file(path);
    if (!file) return smiles_list;
    std::string line;
    while (smiles_list.size() < max_count && std::getline(file, line)) {
        std::size_t const end = line.find_first_of(" \t\r");
        if (end != std::string::npos) line.resize(end);
        if (!line.empty()) smiles_list.push_back(line);
    }
    return smiles_list;
}

/**
 *  @brief Multi-fragment molecules whose halves have no bonded path between them.
 *
 *  A salt is the only way the 255 sentinel appears, and the sweep's exclusion logic hinges on
 *  it, so the corpus always carries some regardless of what the caller streams in.
 */
static std::vector<std::string> builtin_multi_fragment_smiles_() {
    return {
        "CC(=O)Oc1ccccc1C(=O)O.CCO",               // aspirin with an ethanol solvate
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O.OCCO",    // ibuprofen with ethylene glycol
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O.OC(=O)C(=O)O", // caffeine oxalate
        "c1ccccc1.c1ccccc1.CCO",                   // three disjoint fragments
        "CCO.CCO",                                 // two copies of one fragment
    };
}

/**
 *  @brief Host reference: parse @p smiles the way preprocessing does, then BFS the CSR.
 *
 *  @param[in] smiles Molecule to rebuild, as it entered the batch.
 *  @param[out] reference Resized to N squared and filled with the packed hop matrix.
 *  @return Atom count, or 0 if the molecule could not be rebuilt.
 */
static std::uint32_t host_bond_hops_(std::string const &smiles, std::vector<std::uint8_t> &reference) {
    RDKit::SmilesParserParams params;
    params.sanitize = true;
    params.removeHs = false;
    params.strictCXSMILES = false;
    std::unique_ptr<RDKit::RWMol> molecule(RDKit::SmilesToMol(smiles, params));
    if (!molecule) return 0;
    RDKit::MolOps::addHs(*molecule);

    sft::molecule_descriptor_t descriptor;
    if (!sft::build_descriptor(*molecule, descriptor)) return 0;
    std::uint32_t const num_atoms = descriptor.num_atoms;
    reference.assign(static_cast<std::size_t>(num_atoms) * num_atoms, std::uint8_t {255});
    sft::fill_bond_distances(descriptor, reference.data(), num_atoms);
    return num_atoms;
}

static void test_device_matches_host_(std::vector<std::string> const &smiles_list) {
    std::vector<std::string_view> views;
    views.reserve(smiles_list.size());
    for (auto const &smiles : smiles_list) views.emplace_back(smiles);

    // Counterions are kept deliberately: a disconnected fragment is the only way to reach the
    // unreachable-hop sentinel, and the production path strips them.
    auto batch = sfp::prepare_molecule_batch(views, smallfield::keep_all_fragments_k);
    sf_verify_(batch.num_molecules > 0, "no molecule survived preprocessing");

    std::vector<std::uint8_t> device_matrix(batch.layout.num_bond_distance, std::uint8_t {0});
    sftest::run_bond_distance_bfs_on_gpu(batch.host.data(), batch.layout, batch.num_molecules, device_matrix.data());

    auto const metadata = batch.molecule_metadata();
    auto const input_indices = batch.input_indices();
    std::vector<std::uint8_t> reference;
    std::size_t compared_bytes = 0;
    std::size_t unreachable_pairs = 0;
    std::size_t molecules_with_fragments = 0;

    for (std::uint32_t molecule = 0; molecule < batch.num_molecules; ++molecule) {
        auto const &meta = metadata[molecule];
        std::string const &smiles = smiles_list[input_indices[molecule]];
        std::uint32_t const num_atoms = host_bond_hops_(smiles, reference);
        sf_verify_(num_atoms == meta.num_atoms, smiles.c_str());

        std::uint8_t const *device_block = device_matrix.data() + meta.bond_distance_offset;
        std::size_t local_unreachable = 0;
        for (std::uint32_t row = 0; row < num_atoms; ++row) {
            for (std::uint32_t column = 0; column < num_atoms; ++column) {
                std::size_t const entry = static_cast<std::size_t>(row) * num_atoms + column;
                if (device_block[entry] != reference[entry]) {
                    std::fprintf(stderr, "FAIL: %s pair (%u,%u): device %u, host %u\n", smiles.c_str(), row, column,
                                 device_block[entry], reference[entry]);
                    sf_verify_(false, "device hop count differs from the host BFS");
                }
                if (reference[entry] == 255) ++local_unreachable;
            }
        }
        compared_bytes += static_cast<std::size_t>(num_atoms) * num_atoms;
        unreachable_pairs += local_unreachable;
        if (local_unreachable > 0) ++molecules_with_fragments;
    }

    sf_verify_(molecules_with_fragments > 0, "corpus covered no disconnected fragment");
    std::printf("  %u molecules, %zu matrix bytes, %zu unreachable pairs across %zu multi-fragment molecules\n",
                batch.num_molecules, compared_bytes, unreachable_pairs, molecules_with_fragments);
}

} // namespace smallfield::tests

int main() {
    std::printf("test_bond_distance\n");

    char const *corpus_path = std::getenv("SMALLFIELD_TEST_CORPUS");
    char const *max_molecules_env = std::getenv("SMALLFIELD_TEST_MAX_MOLS");
    std::size_t const max_molecules = max_molecules_env ? static_cast<std::size_t>(std::atoll(max_molecules_env)) : 300;

    std::vector<std::string> smiles_list = sftest::builtin_multi_fragment_smiles_();
    for (auto &smiles : sftest::standard_test_smiles()) smiles_list.push_back(std::move(smiles));
    if (corpus_path) {
        auto streamed = sftest::load_smiles_(corpus_path, max_molecules);
        sf_verify_(!streamed.empty(), corpus_path);
        std::printf("Corpus: %s (up to %zu molecules) plus built-ins\n", corpus_path, max_molecules);
        for (auto &smiles : streamed) smiles_list.push_back(std::move(smiles));
    }
    else std::printf("Corpus: built-ins only; set SMALLFIELD_TEST_CORPUS for a wider sweep\n");

    sftest::test_device_matches_host_(smiles_list);
    std::printf("test_bond_distance: PASS\n");
    return 0;
}
