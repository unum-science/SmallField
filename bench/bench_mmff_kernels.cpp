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

#include "preprocess/etkdg_prepare_batch.hpp"
#include "mmff94/mmff94_cpu.hpp"
#include "mmff94/mmff94_extract_device.cuh"
#include "parameters/mmff94_tables.hpp"
#include "types/molecule_descriptor.hpp"

using namespace smallfield;

namespace smallfield {
// External launchers defined in the bench's .cu companion.
void bench_run_gpu_mmff(mmff94_tables_host_t const &tables,
                         molecule_descriptor_t const *descriptors,
                         std::uint32_t num_mols,
                         std::uint32_t const *atom_offsets,
                         std::uint32_t total_atoms,
                         float *type_ms, float *extract_ms, float *csr_ms,
                         float *total_ms);
} // namespace smallfield

namespace {

using clock_t_ = std::chrono::high_resolution_clock;
double elapsed_ms(clock_t_::time_point a, clock_t_::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::vector<std::string> load_smiles(std::string const &path, std::size_t cap) {
    std::vector<std::string> out;
    std::ifstream ifs(path);
    if (!ifs) return out;
    out.reserve(std::min<std::size_t>(cap, 1'000'000));
    std::string line;
    while (std::getline(ifs, line) && out.size() < cap) {
        if (line.empty() || line[0] == '#') continue;
        auto sp = line.find_first_of(" \t");
        if (sp != std::string::npos) line.resize(sp);
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

struct molecule_t {
    std::unique_ptr<RDKit::RWMol> mol;
    molecule_descriptor_t desc;
    bool valid = false;
};

/** Parse SMILES + build descriptor.  OpenMP-parallel. */
std::vector<molecule_t> parse_batch(std::vector<std::string> const &smis) {
    std::vector<molecule_t> out(smis.size());
#pragma omp parallel for schedule(dynamic, 64)
    for (std::size_t i = 0; i < smis.size(); ++i) {
        auto &m = out[i];
        try {
            RDKit::SmilesParserParams p;
            p.sanitize = true; p.removeHs = false; p.strictCXSMILES = false;
            m.mol.reset(RDKit::SmilesToMol(smis[i], p));
        } catch (...) { continue; }
        if (!m.mol) continue;
        try { RDKit::MolOps::addHs(*m.mol); } catch (...) { m.mol.reset(); continue; }
        if (m.mol->getNumAtoms() > 128) { m.mol.reset(); continue; }
        if (!build_descriptor(*m.mol, m.desc)) { m.mol.reset(); continue; }
        m.valid = true;
    }
    return out;
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
    mmff94_tables_host_t tables_host;
    extract_mmff94_tables_from_rdkit(tables_host);
    auto host_view = tables_host.view();
    std::printf("tables extracted: vdw=%zu bond=%zu angle=%zu tor=%zu\n",
                 tables_host.vdw.size(), tables_host.bond.size(),
                 tables_host.angle.size(), tables_host.torsion.size());

    auto const t_load_start = clock_t_::now();
    auto const smis = load_smiles(smi_path, cap);
    auto const t_load_end = clock_t_::now();
    std::printf("SMILES loaded: %zu lines in %.1f ms\n", smis.size(),
                 elapsed_ms(t_load_start, t_load_end));

    auto const t_parse_start = clock_t_::now();
    auto const mols = parse_batch(smis);
    auto const t_parse_end = clock_t_::now();

    std::vector<molecule_descriptor_t> descs;
    std::vector<std::uint32_t> atom_offsets;
    descs.reserve(mols.size());
    atom_offsets.reserve(mols.size());
    std::uint32_t total_atoms = 0;
    for (auto const &m : mols) {
        if (!m.valid) continue;
        atom_offsets.push_back(total_atoms);
        total_atoms += m.desc.num_atoms;
        descs.push_back(m.desc);
    }
    std::uint32_t const N = (std::uint32_t)descs.size();
    std::printf("parsed: %u valid molecules / %u total atoms in %.1f ms\n",
                 N, total_atoms, elapsed_ms(t_parse_start, t_parse_end));

    if (N == 0) { std::printf("no molecules — aborting\n"); return 1; }

    // CPU timing. Run extract_mmff94_params_fully_native on every molecule with OpenMP parallelism,
    // matching the prep path's production settings. Sink a cheap derived quantity into a shared atomic
    // so `-O3` does not DCE the call body when it sees the result go out of scope.
    std::size_t sink = 0;
    auto const t_cpu_start = clock_t_::now();
#pragma omp parallel for schedule(dynamic, 64) reduction(+:sink)
    for (std::uint32_t i = 0; i < N; ++i) {
        etkdg_prepared_molecule_t result;
        result.num_atoms = descs[i].num_atoms;
        extract_mmff94_params_fully_native(descs[i], host_view, result);
        sink += result.mmff_bonds.size() + result.mmff_angles.size() +
                result.mmff_torsions.size() + result.mmff_oops.size() +
                result.mmff_stretchbends.size() + result.mmff_interactions.size();
    }
    auto const t_cpu_end = clock_t_::now();
    double const cpu_ms = elapsed_ms(t_cpu_start, t_cpu_end);
    std::printf("CPU sink (checksum): %zu\n", sink);
    double const cpu_throughput = (double)N * 1000.0 / cpu_ms;
    std::printf("\nCPU native path:\n");
    std::printf("  wall = %.1f ms  → %.1f mol/s  (%d OMP threads)\n",
                 cpu_ms, cpu_throughput, omp_get_max_threads());

    // GPU timing.
    float type_ms = 0.0f, extract_ms = 0.0f, csr_ms = 0.0f, total_ms = 0.0f;
    bench_run_gpu_mmff(tables_host, descs.data(), N, atom_offsets.data(),
                        total_atoms,
                        &type_ms, &extract_ms, &csr_ms, &total_ms);
    double const gpu_throughput = (double)N * 1000.0 / total_ms;
    std::printf("\nGPU kernel trio:\n");
    std::printf("  type+charge   = %.1f ms\n", type_ms);
    std::printf("  extract_terms = %.1f ms\n", extract_ms);
    std::printf("  build_csr     = %.1f ms\n", csr_ms);
    std::printf("  total         = %.1f ms  → %.1f mol/s\n", total_ms, gpu_throughput);
    std::printf("\nspeedup (CPU→GPU): %.2f ×\n", cpu_throughput > 0.0 ? gpu_throughput / cpu_throughput : 0.0);

    return 0;
}
