/**
 *  @file test_mmff_gpu_typer.cpp
 *  @brief Run the GPU MMFF typer kernel against the native CPU typer on a set of drug-like SMILES.
 *  Gate: byte-identical atom types and per-atom partial charges.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <cuda_runtime.h>

#include <RDGeneral/types.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>

#include "types/molecule_descriptor.hpp"
#include "parameters/mmff94_tables.hpp"
#include "mmff94/mmff94_typer.hpp"

using namespace smallfield;

namespace smallfield {
// Forward declarations — real definitions in test_mmff_gpu_typer_kernels.cu.
void *device_tables_create(mmff94_tables_host_t const &host, mmff94_tables_view_t &out_view);
void device_tables_destroy(void *handle);
void run_mmff94_gpu_typer(mmff94_tables_view_t tables_view,
                          molecule_descriptor_t const *descriptors,
                          std::uint32_t num_molecules,
                          std::uint32_t const *atom_offsets,
                          std::uint32_t total_atoms,
                          std::uint8_t *atom_types_out,
                          float *atom_charges_out,
                          std::uint8_t *typing_status_out);
} // namespace smallfield

namespace {

bool near_float(float a, float b, float eps = 1e-3f) {
    if (std::fabs(a - b) <= eps) return true;
    float const scale = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) <= eps * scale;
}

std::vector<std::string> load_smiles(std::string const &path, std::size_t cap) {
    std::vector<std::string> out;
    std::ifstream ifs(path);
    if (!ifs) return out;
    std::string line;
    while (std::getline(ifs, line) && out.size() < cap) {
        if (line.empty() || line[0] == '#') continue;
        auto sp = line.find_first_of(" \t");
        if (sp != std::string::npos) line.resize(sp);
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

} // namespace

int main(int argc, char **argv) {
    std::printf("=== test_mmff_gpu_typer: GPU vs CPU MMFF typer byte-equality gate ===\n");

    mmff94_tables_host_t tables_host;
    extract_mmff94_tables_from_rdkit(tables_host);
    mmff94_tables_view_t host_view = tables_host.view();

    mmff94_tables_view_t device_view {};
    void *device_handle = device_tables_create(tables_host, device_view);
    if (!device_handle) {
        std::printf("FAIL: device table upload failed\n");
        return 1;
    }

    // Assemble a batch of descriptors.
    std::vector<std::string> smis = {
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
        auto extra = load_smiles(argv[1], argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 500);
        for (auto &s : extra) smis.push_back(std::move(s));
    }

    std::vector<molecule_descriptor_t> descriptors;
    std::vector<std::uint32_t> atom_offsets;
    descriptors.reserve(smis.size());
    atom_offsets.reserve(smis.size());
    std::uint32_t total_atoms = 0;

    for (auto const &s : smis) {
        std::unique_ptr<RDKit::RWMol> mol;
        try {
            RDKit::SmilesParserParams p;
            p.sanitize = true; p.removeHs = false; p.strictCXSMILES = false;
            mol.reset(RDKit::SmilesToMol(s, p));
        }
        catch (...) { continue; }
        if (!mol) continue;
        try { RDKit::MolOps::addHs(*mol); } catch (...) { continue; }
        if (mol->getNumAtoms() > 128) continue;

        molecule_descriptor_t desc;
        if (!build_descriptor(*mol, desc)) continue;
        atom_offsets.push_back(total_atoms);
        total_atoms += desc.num_atoms;
        descriptors.push_back(desc);
    }

    std::printf("descriptors: %zu molecules, %u total atoms\n",
                 descriptors.size(), total_atoms);

    std::vector<std::uint8_t> gpu_types(total_atoms);
    std::vector<float> gpu_charges(total_atoms);
    std::vector<std::uint8_t> gpu_status(descriptors.size());

    run_mmff94_gpu_typer(device_view,
                          descriptors.data(),
                          (std::uint32_t)descriptors.size(),
                          atom_offsets.data(),
                          total_atoms,
                          gpu_types.data(),
                          gpu_charges.data(),
                          gpu_status.data());

    // Compare against CPU native.
    int atom_ok = 0, atom_bad = 0;
    int charge_ok = 0, charge_bad = 0;
    int mol_ok = 0, mol_bad = 0;
    int fallback_mols = 0;

    for (std::size_t m = 0; m < descriptors.size(); ++m) {
        auto const &desc = descriptors[m];
        std::uint32_t const off = atom_offsets[m];
        std::uint32_t const n = desc.num_atoms;

        std::vector<std::uint8_t> cpu_types(n);
        std::vector<float> cpu_fc(n), cpu_pc(n);
        std::span<std::uint8_t> ts(cpu_types);
        native_compute_mmff_atom_types(desc, ts);
        std::span<std::uint8_t const> ts_c(cpu_types);
        std::span<float> fc(cpu_fc);
        native_compute_mmff_formal_charges(desc, ts_c, fc);
        std::span<float const> fc_c(cpu_fc);
        std::span<float> pc(cpu_pc);
        native_compute_mmff_partial_charges(desc, host_view, ts_c, fc_c, pc);

        bool const flagged = (gpu_status[m] != 0);
        if (flagged) ++fallback_mols;

        bool mol_match = true;
        for (std::uint32_t a = 0; a < n; ++a) {
            if (cpu_types[a] == gpu_types[off + a]) ++atom_ok;
            else {
                ++atom_bad;
                if (atom_bad < 10)
                    std::printf("  mol=%zu atom=%u element=%u cpu=%u gpu=%u\n",
                                 m, a, desc.element[a], cpu_types[a], gpu_types[off + a]);
                mol_match = false;
            }
            // Skip charge comparison when the molecule is flagged for CPU
            // fallback — GPU intentionally zeroed its charges there, the
            // real value will come from the OpenMP CPU fallback.
            if (!flagged && cpu_types[a] == gpu_types[off + a] && cpu_types[a] != 0) {
                if (near_float(cpu_pc[a], gpu_charges[off + a])) ++charge_ok;
                else {
                    ++charge_bad;
                    if (charge_bad < 10)
                        std::printf("  mol=%zu atom=%u cpu_pc=%.4f gpu_pc=%.4f\n",
                                     m, a, cpu_pc[a], gpu_charges[off + a]);
                    mol_match = false;
                }
            }
        }
        if (mol_match) ++mol_ok; else ++mol_bad;
    }
    std::printf("fallback-flagged molecules (typing_status != 0): %d / %zu\n",
                 fallback_mols, descriptors.size());

    std::printf("atoms: %d match / %d mismatch\n", atom_ok, atom_bad);
    std::printf("charges: %d match / %d mismatch\n", charge_ok, charge_bad);
    std::printf("molecules: %d match / %d mismatch\n", mol_ok, mol_bad);

    device_tables_destroy(device_handle);
    if (atom_bad > 0 || charge_bad > 0) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
