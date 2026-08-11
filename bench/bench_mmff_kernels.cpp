/**
 *  @file bench/bench_mmff_kernels.cpp
 *  @brief GPU MMFF kernel-trio scaling benchmark.
 *
 *  Measures end-to-end wall time of the GPU MMFF kernel trio — type+charge, term extraction, CSR build — vs the CPU
 *  native path across large batches. Reports mol/s throughput + per-stage timing breakdown.
 *
 *  Usage:
 *  @code
 *    bench_mmff_kernels /path/to/chembl.smi [max_mols] [--warmup N]
 *  @endcode
 *
 *  Default max_mols = 10K. ChEMBL-scale inputs, 240K, use ~12 GB of GPU memory for the per-molecule fixed-stride
 *  output arrays; chunk internally if you need more.
 */

#include <algorithm>   // `std::min`, `std::unique`
#include <chrono>      // `std::chrono::high_resolution_clock`, `std::chrono::duration`
#include <fstream>     // `std::ifstream`
#include <istream>     // `std::istream`
#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`, `std::getline`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <omp.h>
#include <boost/shared_ptr.hpp>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <RDGeneral/types.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/mmff94/mmff94_extract.hpp"
#include "smallfield/mmff94/mmff94_extract_device.cuh"
#include "smallfield/mmff94/mmff94_term_equations.hpp"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_rdkit.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

#include "bench/bench_arguments.hpp"
#include "bench/bench_mmff_kernels_device.hpp"

namespace sfbench = smallfield::bench;
namespace sfmmff = smallfield::mmff94;
namespace sfparam = smallfield::parameters;
namespace sfp = smallfield::preprocess;
namespace sft = smallfield::types;

namespace smallfield::bench {

using clock_type_t = std::chrono::high_resolution_clock;
static double elapsed_ms_(clock_type_t::time_point start, clock_type_t::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/** Read up to @p capacity SMILES from an already-open @p input_stream, so a failed open cannot pass as an empty file. */
static std::vector<std::string> load_smiles_batch_(std::istream &input_stream, std::size_t capacity) {
    std::vector<std::string> smiles_batch;
    smiles_batch.reserve(std::min<std::size_t>(capacity, 1'000'000));
    std::string line;
    while (std::getline(input_stream, line) && smiles_batch.size() < capacity) {
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

/** Parse SMILES + build descriptor. OpenMP-parallel. */
static std::vector<parsed_molecule_t> parse_smiles_batch_(std::vector<std::string> const &smiles_batch) {
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

} // namespace smallfield::bench

using sfbench::clock_type_t;
using sfbench::elapsed_ms_;
using sfbench::load_smiles_batch_;
using sfbench::parse_smiles_batch_;
using sfbench::parsed_molecule_t;

/** Name printed on every throughput line so a concatenated CI log stays attributable. */
static constexpr char const *executable_name_k = "bench_mmff_kernels";

/**
 *  @brief Ceiling on `[max_mols]`.
 *
 *  Every term array is allocated at the per-molecule maximum stride, so device memory grows at roughly 50 kB per
 *  molecule — 240 K already takes ~12 GB. The cap turns a mistyped count into a diagnostic instead of a
 *  `cudaMalloc` failure several seconds into the run.
 */
static constexpr std::size_t max_molecules_per_run_k = 1'000'000;

int main(int argc, char **argv) {
    if (argc < 2) {
        fmt::println(stderr, "Usage: {} <smi_file> [max_mols=10000]", executable_name_k);
        return 2;
    }
    std::string const smiles_path = argv[1];
    std::size_t capacity = 10'000;
    if (argc > 2) {
        auto const status = sfbench::parse_number(std::string_view {argv[2]}, std::size_t {1},
                                                  max_molecules_per_run_k, capacity);
        if (status != sfbench::number_parse_t::parsed_k) {
            fmt::println(stderr, "{}: max_mols '{}' {} (accepted range 1..{})", executable_name_k, argv[2],
                         sfbench::describe_number_parse(status), fmt::group_digits(max_molecules_per_run_k));
            return 2;
        }
    }

    fmt::println("=== bench_mmff_kernels: GPU MMFF scaling benchmark ===");
    fmt::println("input: {} (cap {})", smiles_path, fmt::group_digits(capacity));

    // Table extraction once.
    sfparam::mmff94_tables_host_t tables_host;
    sfparam::extract_mmff94_tables_from_rdkit(tables_host);
    auto const tables_view = tables_host.view();
    fmt::println("tables extracted: vdw={} bond={} angle={} tor={}", tables_host.vdw.size(), tables_host.bond.size(),
                 tables_host.angle.size(), tables_host.torsion.size());

    // A corpus that never opened must not read as an empty corpus.
    std::ifstream input_stream(smiles_path);
    if (!input_stream) {
        fmt::println(stderr, "{}: cannot open corpus '{}'", executable_name_k, smiles_path);
        return 2;
    }
    auto const load_start = clock_type_t::now();
    auto const smiles_batch = load_smiles_batch_(input_stream, capacity);
    auto const load_end = clock_type_t::now();
    fmt::println("SMILES loaded: {} lines in {:.1f} ms", fmt::group_digits(smiles_batch.size()),
                 elapsed_ms_(load_start, load_end));

    auto const parse_start = clock_type_t::now();
    auto const parsed_molecules = parse_smiles_batch_(smiles_batch);
    auto const parse_end = clock_type_t::now();

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
    fmt::println("parsed: {} valid molecules / {} total atoms in {:.1f} ms", fmt::group_digits(num_molecules),
                 fmt::group_digits(total_atom_count), elapsed_ms_(parse_start, parse_end));

    if (num_molecules == 0) {
        fmt::println(stderr, "{}: no molecules — aborting", executable_name_k);
        return 1;
    }

    // CPU timing. Run extract_mmff94_params_fully_native on every molecule with OpenMP parallelism,
    // matching the prep path's production settings. Sink a cheap derived quantity into a shared atomic
    // so `-O3` does not DCE the call body when it sees the result go out of scope.
    std::size_t checksum = 0;
    auto const cpu_start = clock_type_t::now();
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : checksum)
    for (std::uint32_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        sfp::etkdg_prepared_molecule_t extracted_molecule;
        extracted_molecule.num_atoms = descriptors[molecule_index].num_atoms;
        sfmmff::extract_mmff94_params_fully_native(descriptors[molecule_index], tables_view, extracted_molecule);
        checksum += extracted_molecule.mmff_bonds.size() + extracted_molecule.mmff_angles.size() +
                    extracted_molecule.mmff_torsions.size() + extracted_molecule.mmff_oops.size() +
                    extracted_molecule.mmff_stretchbends.size() + extracted_molecule.mmff_interactions.size();
    }
    auto const cpu_end = clock_type_t::now();
    double const cpu_ms = elapsed_ms_(cpu_start, cpu_end);
    fmt::println("CPU sink (checksum): {}", fmt::group_digits(checksum));
    double const cpu_throughput = static_cast<double>(num_molecules) * 1000.0 / cpu_ms;
    fmt::println("\nCPU native path:");
    fmt::println("  wall = {:.1f} ms  ({} OMP threads)", cpu_ms, omp_get_max_threads());
    fmt::println("  {} cpu mol/s: {:.1f}", executable_name_k, cpu_throughput);

    // GPU timing.
    float type_and_charge_ms = 0.0f;
    float extract_terms_ms = 0.0f;
    float build_csr_ms = 0.0f;
    float total_gpu_ms = 0.0f;
    sfbench::launch_mmff94_kernel_trio_benchmark(tables_host, descriptors.data(), num_molecules, atom_offsets.data(),
                                                 total_atom_count, &type_and_charge_ms, &extract_terms_ms,
                                                 &build_csr_ms, &total_gpu_ms);
    double const gpu_throughput = static_cast<double>(num_molecules) * 1000.0 / total_gpu_ms;
    fmt::println("\nGPU kernel trio:");
    fmt::println("  type+charge   = {:.1f} ms", type_and_charge_ms);
    fmt::println("  extract_terms = {:.1f} ms", extract_terms_ms);
    fmt::println("  build_csr     = {:.1f} ms", build_csr_ms);
    fmt::println("  total         = {:.1f} ms", total_gpu_ms);
    fmt::println("  {} gpu mol/s: {:.1f}", executable_name_k, gpu_throughput);
    fmt::println("\nspeedup (CPU→GPU): {:.2f} ×", cpu_throughput > 0.0 ? gpu_throughput / cpu_throughput : 0.0);

    return 0;
}
