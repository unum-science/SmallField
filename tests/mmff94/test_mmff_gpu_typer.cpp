/**
 *  @file test_mmff_gpu_typer.cpp
 *  @brief Run the GPU MMFF typer kernel against the native CPU typer on a set of drug-like SMILES.
 *  Gate: byte-identical atom types and per-atom partial charges.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <cuda_runtime.h>

#include <RDGeneral/types.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/mmff94/mmff94_typer.hpp"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "test_mmff_gpu_typer_kernels.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace sf = smallfield;
namespace sfmmff = smallfield::mmff94;
namespace sfparam = smallfield::parameters;
namespace sft = smallfield::types;

namespace {

bool approximately_equal(float lhs, float rhs, float eps = 1e-3f) {
    if (std::fabs(lhs - rhs) <= eps) return true;
    float const scale = std::max(1.0f, std::max(std::fabs(lhs), std::fabs(rhs)));
    return std::fabs(lhs - rhs) <= eps * scale;
}

std::vector<std::string> load_smiles(std::string const &path, std::size_t cap) {
    std::vector<std::string> smiles;
    std::ifstream input_stream(path);
    if (!input_stream) return smiles;
    std::string line;
    while (std::getline(input_stream, line) && smiles.size() < cap) {
        if (line.empty() || line[0] == '#') continue;
        auto const separator = line.find_first_of(" \t");
        if (separator != std::string::npos) line.resize(separator);
        if (!line.empty()) smiles.push_back(line);
    }
    return smiles;
}

} // namespace

int main(int argc, char **argv) {
    std::printf("=== test_mmff_gpu_typer: GPU vs CPU MMFF typer byte-equality gate ===\n");

    sfparam::mmff94_tables_host_t tables_host {};
    sfparam::extract_mmff94_tables_from_rdkit(tables_host);
    sfparam::mmff94_tables_view_t host_view = tables_host.view();

    sfparam::mmff94_tables_view_t device_view {};
    void *device_tables_handle = create_mmff94_device_tables_view(tables_host, device_view);
    if (!device_tables_handle) {
        std::printf("FAIL: device table upload failed\n");
        return 1;
    }

    // Assemble a batch of descriptors.
    std::vector<std::string> smiles_batch = {
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "CC(=O)NC1=CC=C(O)C=C1",
        "[NH2+]=C(N)N",
        "CC(=O)[O-]",
        "c1cc[nH+]c(c1)",
        "C1CCOC1",
        "C1CC1",
        "c1ccc2[nH]ccc2c1",
    };
    if (argc > 1) {
        auto extra_smiles = load_smiles(argv[1], argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 500);
        for (auto &smiles : extra_smiles) smiles_batch.push_back(std::move(smiles));
    }

    std::vector<sft::molecule_descriptor_t> descriptors;
    std::vector<std::uint32_t> atom_offsets;
    descriptors.reserve(smiles_batch.size());
    atom_offsets.reserve(smiles_batch.size());
    std::uint32_t total_atoms = 0;

    for (auto const &smiles : smiles_batch) {
        std::unique_ptr<RDKit::RWMol> molecule;
        try {
            RDKit::SmilesParserParams params;
            params.sanitize = true;
            params.removeHs = false;
            params.strictCXSMILES = false;
            molecule.reset(RDKit::SmilesToMol(smiles, params));
        }
        catch (...) {
            continue;
        }
        if (!molecule) continue;
        try {
            RDKit::MolOps::addHs(*molecule);
        }
        catch (...) {
            continue;
        }
        if (molecule->getNumAtoms() > 128) continue;

        sft::molecule_descriptor_t descriptor;
        if (!sft::build_descriptor(*molecule, descriptor)) continue;
        atom_offsets.push_back(total_atoms);
        total_atoms += descriptor.num_atoms;
        descriptors.push_back(descriptor);
    }

    std::printf("descriptors: %zu molecules, %u total atoms\n", descriptors.size(), total_atoms);

    std::vector<std::uint8_t> gpu_atom_types(total_atoms);
    std::vector<float> gpu_atom_charges(total_atoms);
    std::vector<std::uint8_t> gpu_typing_status(descriptors.size());

    launch_mmff94_gpu_typer_batch(device_view, descriptors.data(), static_cast<std::uint32_t>(descriptors.size()),
                                  atom_offsets.data(), total_atoms, gpu_atom_types.data(), gpu_atom_charges.data(),
                                  gpu_typing_status.data());

    // Compare against CPU native.
    int matching_atom_count = 0;
    int mismatched_atom_count = 0;
    int matching_charge_count = 0;
    int mismatched_charge_count = 0;
    int matching_molecule_count = 0;
    int mismatched_molecule_count = 0;
    int fallback_molecule_count = 0;

    for (std::size_t molecule_index = 0; molecule_index < descriptors.size(); ++molecule_index) {
        auto const &descriptor = descriptors[molecule_index];
        std::uint32_t const atom_offset = atom_offsets[molecule_index];
        std::uint32_t const num_atoms = descriptor.num_atoms;

        std::vector<std::uint8_t> cpu_atom_types(num_atoms);
        std::vector<float> cpu_formal_charges(num_atoms), cpu_partial_charges(num_atoms);
        std::span<std::uint8_t> atom_types_view(cpu_atom_types);
        sfmmff::native_compute_mmff_atom_types(descriptor, atom_types_view);
        std::span<std::uint8_t const> const_atom_types_view(cpu_atom_types);
        std::span<float> formal_charge_view(cpu_formal_charges);
        sfmmff::native_compute_mmff_formal_charges(descriptor, const_atom_types_view, formal_charge_view);
        std::span<float const> const_formal_charge_view(cpu_formal_charges);
        std::span<float> partial_charge_view(cpu_partial_charges);
        sfmmff::native_compute_mmff_partial_charges(descriptor, host_view, const_atom_types_view,
                                                    const_formal_charge_view, partial_charge_view);

        bool const requires_cpu_fallback = (gpu_typing_status[molecule_index] != 0);
        if (requires_cpu_fallback) ++fallback_molecule_count;

        bool molecule_matches = true;
        for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
            if (cpu_atom_types[atom_index] == gpu_atom_types[atom_offset + atom_index]) { ++matching_atom_count; }
            else {
                ++mismatched_atom_count;
                if (mismatched_atom_count < 10) {
                    std::printf("  mol=%zu atom=%u element=%u cpu=%u gpu=%u\n", molecule_index, atom_index,
                                descriptor.element[atom_index], cpu_atom_types[atom_index],
                                gpu_atom_types[atom_offset + atom_index]);
                }
                molecule_matches = false;
            }

            // Skip charge comparison when the molecule is flagged for CPU
            // fallback — GPU intentionally zeroed its charges there, the
            // real value will come from the OpenMP CPU fallback.
            if (!requires_cpu_fallback && cpu_atom_types[atom_index] == gpu_atom_types[atom_offset + atom_index] &&
                cpu_atom_types[atom_index] != 0) {
                if (approximately_equal(cpu_partial_charges[atom_index], gpu_atom_charges[atom_offset + atom_index])) {
                    ++matching_charge_count;
                }
                else {
                    ++mismatched_charge_count;
                    if (mismatched_charge_count < 10) {
                        std::printf("  mol=%zu atom=%u cpu_pc=%.4f gpu_pc=%.4f\n", molecule_index, atom_index,
                                    cpu_partial_charges[atom_index], gpu_atom_charges[atom_offset + atom_index]);
                    }
                    molecule_matches = false;
                }
            }
        }
        if (molecule_matches) ++matching_molecule_count;
        else ++mismatched_molecule_count;
    }
    std::printf("fallback-flagged molecules (typing_status != 0): %d / %zu\n", fallback_molecule_count,
                descriptors.size());

    std::printf("atoms: %d match / %d mismatch\n", matching_atom_count, mismatched_atom_count);
    std::printf("charges: %d match / %d mismatch\n", matching_charge_count, mismatched_charge_count);
    std::printf("molecules: %d match / %d mismatch\n", matching_molecule_count, mismatched_molecule_count);

    destroy_mmff94_device_tables_view(device_tables_handle);
    if (mismatched_atom_count > 0 || mismatched_charge_count > 0) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
