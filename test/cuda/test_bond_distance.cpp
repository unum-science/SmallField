/**
 *  @file test/cuda/test_bond_distance.cpp
 *  @brief Gate byte-for-byte parity between the device hop-count BFS and the host all-pairs BFS.
 *
 *  A warp BFS rebuilds the N × N hop matrix the MMFF nonbonded sweep reads, on the device, from the batch's bond
 *  list; the host builds the same matrix from the descriptor's CSR adjacency. Two different traversals over two
 *  different encodings of the same graph, so the check is exact equality over every byte of every molecule — a
 *  single differing entry silently changes which pairs the sweep treats as 1-4 or as excluded.
 *
 *  Corpus: a `.smi` path in `argv[1]` or in `SMALLFIELD_TEST_CORPUS` streams real molecules, capped by `argv[2]`
 *  or `SMALLFIELD_TEST_MAX_MOLS` (default 300). Absent both the built-in list runs, which carries its own
 *  multi-fragment salts so the 255 unreachable sentinel is covered either way.
 */

#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

#include "test/cuda/test_bond_distance_kernels.hpp"
#include "test/test_rdkit_reference_helpers.hpp"

namespace sfapi = smallfield::api;
namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_bond_distance";

namespace smallfield::tests {

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
    if (sft::build_descriptor(*molecule, descriptor) != sfapi::conformer_status_t::success_k) return 0;
    std::uint32_t const num_atoms = descriptor.num_atoms;
    reference.assign(static_cast<std::size_t>(num_atoms) * num_atoms, std::uint8_t {255});
    sft::fill_bond_distances(descriptor, sft::hop_matrix_t {reference.data(), num_atoms, num_atoms});
    return num_atoms;
}

static void test_device_matches_host_(std::vector<std::string> const &smiles_list) {
    std::vector<std::string_view> views;
    views.reserve(smiles_list.size());
    for (auto const &smiles : smiles_list) views.emplace_back(smiles);

    // Counterions are kept deliberately: a disconnected fragment is the only way to reach the
    // unreachable-hop sentinel, and the production path strips them.
    sfp::etkdg_molecule_batch_t batch {};
    sfp::prepare_molecule_batch_into(views, sft::max_atoms_per_molecule_k, batch, smallfield::keep_all_fragments_k);
    sf_verify_(batch.num_molecules > 0, "no molecule survived preprocessing");

    std::vector<std::uint8_t> device_matrix(batch.layout.bond_distance.count, std::uint8_t {0});
    sftest::run_bond_distance_bfs_on_gpu(batch.host.view(), batch.layout, batch.num_molecules, device_matrix);

    sfapi::batch_pool_view_t const view = batch.host_view();
    auto const metadata = view.molecule_metadata();
    auto const input_indices = view.input_indices();
    std::vector<std::uint8_t> reference;
    std::size_t compared_bytes = 0;
    std::size_t unreachable_pairs = 0;
    std::size_t molecules_with_fragments = 0;

    for (std::uint32_t molecule = 0; molecule < batch.num_molecules; ++molecule) {
        auto const &meta = metadata[molecule];
        std::string const &smiles = smiles_list[input_indices[molecule]];
        std::uint32_t const num_atoms = host_bond_hops_(smiles, reference);
        sf_verify_(num_atoms == meta.num_atoms,
                   fmt::format("the host rebuild of {} carries {} atoms against the batch's {}", smiles, num_atoms,
                               meta.num_atoms));

        std::uint8_t const *device_block = device_matrix.data() + meta.bond_distance_offset;
        std::size_t local_unreachable = 0;
        for (std::uint32_t row = 0; row < num_atoms; ++row) {
            for (std::uint32_t column = 0; column < num_atoms; ++column) {
                std::size_t const entry = static_cast<std::size_t>(row) * num_atoms + column;
                if (device_block[entry] != reference[entry]) {
                    std::fflush(stdout);
                    fmt::println(stderr, "MISMATCH bond_distance smiles={} row={} column={} field=hops host={} device={}",
                                 smiles, row, column, reference[entry], device_block[entry]);
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
    fmt::println("  {} molecules, {} matrix bytes, {} unreachable pairs across {} multi-fragment molecules",
                 fmt::group_digits(batch.num_molecules), fmt::group_digits(compared_bytes),
                 fmt::group_digits(unreachable_pairs), fmt::group_digits(molecules_with_fragments));
}

} // namespace smallfield::tests

int main(int argc, char **argv) {
    fmt::println("{}: device warp BFS vs host CSR BFS over the packed hop matrix", test_name_k);

    auto const selection = sftest::select_corpus(argc, argv, 300);
    if (selection.knob == sftest::corpus_knob_t::refused_k) {
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", argv[0]);
        return 2;
    }

    std::vector<std::string> smiles_list = sftest::builtin_multi_fragment_smiles_();
    for (auto &smiles : sftest::standard_test_smiles()) smiles_list.push_back(std::move(smiles));
    if (selection.knob == sftest::corpus_knob_t::named_k) {
        std::vector<std::string> streamed;
        auto const outcome = sftest::load_smiles_into(selection.path, selection.max_molecules, streamed);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), selection.path);
        fmt::println("Corpus: {} ({} molecules) plus built-ins", selection.path, fmt::group_digits(streamed.size()));
        for (auto &smiles : streamed) smiles_list.push_back(std::move(smiles));
    }
    else fmt::println("Corpus: built-ins only; name one on the command line or in SMALLFIELD_TEST_CORPUS");

    sftest::test_device_matches_host_(smiles_list);
    return sftest::report_pass(test_name_k);
}
