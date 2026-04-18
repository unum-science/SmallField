/**
 *  @file bench_mmff_gpu.cpp
 *  @brief GPU MMFF kernel-trio scaling benchmark: measure end-to-end wall time
 *  of the GPU MMFF kernel trio (type+charge, term extraction, CSR
 *  build) vs the CPU native path across large batches.  Reports
 *  mol/s throughput + per-stage timing breakdown.
 *
 *  Usage:
 *    bench_mmff_gpu /path/to/chembl.smi [max_mols] [--warmup N]
 *
 *  Default max_mols = 10K.  ChEMBL-scale inputs (240K) use ~12 GB
 *  of GPU memory for the per-molecule fixed-stride output arrays;
 *  chunk internally if you need more.
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <omp.h>
#include <boost/shared_ptr.hpp>
#include <cuda_runtime.h>

#include <RDGeneral/types.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/mmff94/mmff94_cpu.hpp"
#include "smallfield/mmff94/mmff94_cpu_extract.hpp"
#include "smallfield/mmff94/mmff94_extract_device.cuh"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_rdkit.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

#include "bench_mmff_kernels_device.hpp"

namespace sf = smallfield;
namespace sfmmff = smallfield::mmff94;
namespace sfparam = smallfield::parameters;
namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;

namespace {

using clock_t_ = std::chrono::high_resolution_clock;
double elapsed_ms(clock_t_::time_point a, clock_t_::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::vector<std::string> load_smiles_batch(std::string const &path, std::size_t cap) {
    std::vector<std::string> smiles_batch;
    std::ifstream input_stream(path);
    if (!input_stream) return smiles_batch;
    smiles_batch.reserve(std::min<std::size_t>(cap, 1'000'000));
    std::string line;
    while (std::getline(input_stream, line) && smiles_batch.size() < cap) {
        if (line.empty() || line[0] == '#') continue;
        auto const separator_index = line.find_first_of(" \t");
        if (separator_index != std::string::npos) line.resize(separator_index);
        if (!line.empty()) smiles_batch.push_back(line);
    }
    return smiles_batch;
}

struct parsed_molecule_t {
    std::unique_ptr<RDKit::RWMol> rdkit_molecule;
    sft::molecule_descriptor_t descriptor;
    bool is_valid = false;
};

/** Parse SMILES + build descriptor.  OpenMP-parallel. */
std::vector<parsed_molecule_t> parse_smiles_batch(std::vector<std::string> const &smiles_batch) {
    std::vector<parsed_molecule_t> parsed_molecules(smiles_batch.size());
#pragma omp parallel for schedule(dynamic, 64)
    for (std::size_t molecule_index = 0; molecule_index < smiles_batch.size(); ++molecule_index) {
        auto &parsed = parsed_molecules[molecule_index];
        try {
            RDKit::SmilesParserParams parser_params;
            parser_params.sanitize = true;
            parser_params.removeHs = false;
            parser_params.strictCXSMILES = false;
            parsed.rdkit_molecule.reset(RDKit::SmilesToMol(smiles_batch[molecule_index], parser_params));
        }
        catch (...) {
            continue;
        }
        if (!parsed.rdkit_molecule) continue;
        try {
            RDKit::MolOps::addHs(*parsed.rdkit_molecule);
        }
        catch (...) {
            parsed.rdkit_molecule.reset();
            continue;
        }
        if (parsed.rdkit_molecule->getNumAtoms() > 128) {
            parsed.rdkit_molecule.reset();
            continue;
        }
        if (!sft::build_descriptor(*parsed.rdkit_molecule, parsed.descriptor)) {
            parsed.rdkit_molecule.reset();
            continue;
        }
        parsed.is_valid = true;
    }
    return parsed_molecules;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: bench_mmff_gpu <smi_file> [max_mols=10000]\n");
        return 2;
    }
    std::string const smi_path = argv[1];
    std::size_t const cap = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 10'000;

    std::printf("=== bench_mmff_kernels: GPU MMFF scaling benchmark ===\n");
    std::printf("input: %s (cap %zu)\n", smi_path.c_str(), cap);

    // Table extraction once.
    sfparam::mmff94_tables_host_t tables_host;
    sfparam::extract_mmff94_tables_from_rdkit(tables_host);
    auto const tables_view = tables_host.view();
    std::printf("tables extracted: vdw=%zu bond=%zu angle=%zu tor=%zu\n", tables_host.vdw.size(),
                tables_host.bond.size(), tables_host.angle.size(), tables_host.torsion.size());

    auto const load_start = clock_t_::now();
    auto const smiles_batch = load_smiles_batch(smi_path, cap);
    auto const load_end = clock_t_::now();
    std::printf("SMILES loaded: %zu lines in %.1f ms\n", smiles_batch.size(), elapsed_ms(load_start, load_end));

    auto const parse_start = clock_t_::now();
    auto const parsed_molecules = parse_smiles_batch(smiles_batch);
    auto const parse_end = clock_t_::now();

    std::vector<sft::molecule_descriptor_t> descriptors;
    std::vector<std::uint32_t> atom_offsets;
    descriptors.reserve(parsed_molecules.size());
    atom_offsets.reserve(parsed_molecules.size());
    std::uint32_t total_atom_count = 0;
    for (auto const &parsed_molecule : parsed_molecules) {
        if (!parsed_molecule.is_valid) continue;
        atom_offsets.push_back(total_atom_count);
        total_atom_count += parsed_molecule.descriptor.num_atoms;
        descriptors.push_back(parsed_molecule.descriptor);
    }
    std::uint32_t const num_molecules = static_cast<std::uint32_t>(descriptors.size());
    std::printf("parsed: %u valid molecules / %u total atoms in %.1f ms\n", num_molecules, total_atom_count,
                elapsed_ms(parse_start, parse_end));

    if (num_molecules == 0) {
        std::printf("no molecules — aborting\n");
        return 1;
    }

    // CPU timing. Run extract_mmff94_params_fully_native on every molecule with OpenMP parallelism,
    // matching the prep path's production settings. Sink a cheap derived quantity into a shared atomic
    // so `-O3` does not DCE the call body when it sees the result go out of scope.
    std::size_t checksum = 0;
    auto const cpu_start = clock_t_::now();
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : checksum)
    for (std::uint32_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        sfp::etkdg_prepared_molecule_t extracted_molecule;
        extracted_molecule.num_atoms = descriptors[molecule_index].num_atoms;
        sfmmff::extract_mmff94_params_fully_native(descriptors[molecule_index], tables_view, extracted_molecule);
        checksum += extracted_molecule.mmff_bonds.size() + extracted_molecule.mmff_angles.size() +
                    extracted_molecule.mmff_torsions.size() + extracted_molecule.mmff_oops.size() +
                    extracted_molecule.mmff_stretchbends.size() + extracted_molecule.mmff_interactions.size();
    }
    auto const cpu_end = clock_t_::now();
    double const cpu_ms = elapsed_ms(cpu_start, cpu_end);
    std::printf("CPU sink (checksum): %zu\n", checksum);
    double const cpu_throughput = static_cast<double>(num_molecules) * 1000.0 / cpu_ms;
    std::printf("\nCPU native path:\n");
    std::printf("  wall = %.1f ms  → %.1f mol/s  (%d OMP threads)\n", cpu_ms, cpu_throughput, omp_get_max_threads());

    // GPU timing.
    float type_and_charge_ms = 0.0f;
    float extract_terms_ms = 0.0f;
    float build_csr_ms = 0.0f;
    float total_gpu_ms = 0.0f;
    bench_mmff::launch_mmff94_kernel_trio_benchmark(tables_host, descriptors.data(), num_molecules, atom_offsets.data(),
                                                    total_atom_count, &type_and_charge_ms, &extract_terms_ms,
                                                    &build_csr_ms, &total_gpu_ms);
    double const gpu_throughput = static_cast<double>(num_molecules) * 1000.0 / total_gpu_ms;
    std::printf("\nGPU kernel trio:\n");
    std::printf("  type+charge   = %.1f ms\n", type_and_charge_ms);
    std::printf("  extract_terms = %.1f ms\n", extract_terms_ms);
    std::printf("  build_csr     = %.1f ms\n", build_csr_ms);
    std::printf("  total         = %.1f ms  → %.1f mol/s\n", total_gpu_ms, gpu_throughput);
    std::printf("\nspeedup (CPU→GPU): %.2f ×\n", cpu_throughput > 0.0 ? gpu_throughput / cpu_throughput : 0.0);

    return 0;
}
