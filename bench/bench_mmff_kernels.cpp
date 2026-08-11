/**
 *  @file bench/bench_mmff_kernels.cpp
 *  @brief GPU MMFF kernel-trio scaling benchmark.
 *
 *  Measures end-to-end wall time of the GPU MMFF kernel trio — type+charge, term extraction, CSR build — vs the CPU
 *  native path across large batches. Reports mol/s throughput + per-stage timing breakdown.
 *
 *  Usage:
 *  @code
 *    bench_mmff_kernels --smi /path/to/corpus.smi [--max-molecules N]
 *  @endcode
 *
 *  Default `--max-molecules` is 10K. ChEMBL-scale inputs, 240K, use ~12 GB of GPU memory for the per-molecule
 *  fixed-stride output arrays; chunk internally if you need more. The device launcher warms the driver with one
 *  untimed launch of the first kernel, so there is no warm-up count to pass.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `stderr`

#include <array>       // `std::array`
#include <chrono>      // `std::chrono::high_resolution_clock`, `std::chrono::duration`
#include <memory>      // `std::unique_ptr`
#include <string>      // `std::string`
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

/**
 *  @brief Atom ceiling this benchmark admits.
 *
 *  Held at 128 rather than at `sft::max_atoms_per_molecule_k` so the batch composition — and therefore every
 *  published mol/s figure — stays comparable across runs when the library's dispatch tiers move.
 */
static constexpr std::uint32_t benchmark_atom_ceiling_k = 128;

/** How far a SMILES got before the timed section, so the batch can report what it dropped and why. */
enum class parse_outcome_t {
    /** Sanitized, hydrogenated, within the atom ceiling, and described. */
    parsed_k,
    /** `SmilesToMol` returned nothing, or threw. */
    unparseable_k,
    /** `MolOps::addHs` threw. */
    hydrogenation_failed_k,
    /** Above `benchmark_atom_ceiling_k` once hydrogens are explicit. */
    too_many_atoms_k,
    /** `build_descriptor` refused the molecule. */
    no_descriptor_k,
};

/** Column label naming the outcome in the parse histogram. */
static char const *describe_parse_outcome_(parse_outcome_t outcome) {
    switch (outcome) {
    case parse_outcome_t::parsed_k: return "parsed";
    case parse_outcome_t::unparseable_k: return "unparseable";
    case parse_outcome_t::hydrogenation_failed_k: return "hydrogenation_failed";
    case parse_outcome_t::too_many_atoms_k: return "too_many_atoms";
    case parse_outcome_t::no_descriptor_k: return "no_descriptor";
    }
    return "other";
}

struct parsed_molecule_t {
    std::unique_ptr<RDKit::RWMol> rdkit_molecule;
    sft::molecule_descriptor_t descriptor;
    /** Only `parsed_k` carries a usable `descriptor`; the other four name the reason it is absent. */
    parse_outcome_t outcome = parse_outcome_t::unparseable_k;
};

/** Parse SMILES and build descriptors, recording per-molecule outcomes. OpenMP-parallel. */
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
            continue; // The default outcome already reads `unparseable_k`.
        }
        if (!parsed.rdkit_molecule) continue;
        try {
            RDKit::MolOps::addHs(*parsed.rdkit_molecule);
        }
        catch (...) {
            parsed.rdkit_molecule.reset();
            parsed.outcome = parse_outcome_t::hydrogenation_failed_k;
            continue;
        }
        if (parsed.rdkit_molecule->getNumAtoms() > benchmark_atom_ceiling_k) {
            parsed.rdkit_molecule.reset();
            parsed.outcome = parse_outcome_t::too_many_atoms_k;
            continue;
        }
        if (!sft::build_descriptor(*parsed.rdkit_molecule, parsed.descriptor)) {
            parsed.rdkit_molecule.reset();
            parsed.outcome = parse_outcome_t::no_descriptor_k;
            continue;
        }
        parsed.outcome = parse_outcome_t::parsed_k;
    }
    return parsed_molecules;
}

/** Molecules per parse outcome, indexed by the enumerator's own value. */
using parse_outcome_histogram_t = std::array<std::size_t, 5>;

/** Tally @p parsed_molecules by outcome in one pass. */
static parse_outcome_histogram_t parse_outcome_histogram_of_(std::vector<parsed_molecule_t> const &parsed_molecules) {
    parse_outcome_histogram_t histogram {};
    for (auto const &parsed_molecule : parsed_molecules) ++histogram[static_cast<std::size_t>(parsed_molecule.outcome)];
    return histogram;
}

/** Print one line per non-empty parse outcome, so a rejection class cannot grow behind the mol/s number. */
static void print_parse_outcomes_(parse_outcome_histogram_t const &histogram, std::size_t total_molecules) {
    if (total_molecules == 0) return;
    for (std::size_t code = 0; code < histogram.size(); ++code) {
        if (histogram[code] == 0) continue;
        fmt::println("  {:<22} {:>12}  {:.3f} %", describe_parse_outcome_(static_cast<parse_outcome_t>(code)),
                     fmt::group_digits(histogram[code]),
                     100.0 * static_cast<double>(histogram[code]) / static_cast<double>(total_molecules));
    }
}

} // namespace smallfield::bench

using sfbench::clock_type_t;
using sfbench::elapsed_ms_;
using sfbench::parse_outcome_t;
using sfbench::parse_smiles_batch_;
using sfbench::parsed_molecule_t;
using sfbench::print_parse_outcomes_;

/** Name printed on every throughput line so a concatenated CI log stays attributable. */
static constexpr char const *executable_name_k = "bench_mmff_kernels";

/**
 *  @brief Ceiling on `--max-molecules`.
 *
 *  Every term array is allocated at the per-molecule maximum stride, so device memory grows at roughly 50 kB per
 *  molecule — 240 K already takes ~12 GB. The cap turns a mistyped count into a diagnostic instead of a
 *  `cudaMalloc` failure several seconds into the run.
 */
static constexpr std::size_t max_molecules_per_run_k = 1'000'000;

#pragma region Command Line

/** Everything this driver takes from the command line. */
struct kernel_bench_arguments_t {
    /** Corpus to time over, from `--smi` or from `SMALLFIELD_TEST_CORPUS`. There is no default. */
    std::string_view smiles_path;
    /** Upper bound on molecules read from the corpus. */
    std::size_t max_molecules = 10'000;
};

/** Whether the command line was accepted or refused. */
enum class arguments_parse_t { accepted_k, refused_k };

/** Print the one usage line, naming every flag this driver understands. */
static void print_usage_(char const *invocation) {
    fmt::println(stderr, "usage: {} --smi PATH [--max-molecules N]", invocation);
}

/** Fill @p arguments from `argv`, or explain on `stderr` why the command line was refused. */
static arguments_parse_t parse_arguments_(int argc, char **argv, kernel_bench_arguments_t &arguments) {
    for (int argument_index = 1; argument_index < argc; ++argument_index) {
        std::string_view const flag = argv[argument_index];
        if (flag == "--smi") {
            if (argument_index + 1 >= argc) {
                fmt::println(stderr, "{}: --smi takes a value, and none followed it", executable_name_k);
                return arguments_parse_t::refused_k;
            }
            arguments.smiles_path = argv[++argument_index];
        }
        else if (flag == "--max-molecules") {
            if (argument_index + 1 >= argc) {
                fmt::println(stderr, "{}: --max-molecules takes a value, and none followed it", executable_name_k);
                return arguments_parse_t::refused_k;
            }
            std::string_view const text = argv[++argument_index];
            auto const status = sfbench::parse_number(text, std::size_t {1}, max_molecules_per_run_k,
                                                      arguments.max_molecules);
            if (status != sfbench::number_parse_t::parsed_k) {
                fmt::println(stderr, "{}: --max-molecules value '{}' {} (accepted range 1..{})", executable_name_k,
                             text, sfbench::describe_number_parse(status), fmt::group_digits(max_molecules_per_run_k));
                return arguments_parse_t::refused_k;
            }
        }
        else {
            fmt::println(stderr, "{}: unknown argument '{}'", executable_name_k, flag);
            print_usage_(argv[0]);
            return arguments_parse_t::refused_k;
        }
    }
    if (arguments.smiles_path.empty()) arguments.smiles_path = sfbench::corpus_path_from_environment();
    return arguments_parse_t::accepted_k;
}

#pragma endregion Command Line

int main(int argc, char **argv) {
    kernel_bench_arguments_t arguments;
    if (parse_arguments_(argc, argv, arguments) != arguments_parse_t::accepted_k) return 2;

    fmt::println("{}: GPU MMFF kernel trio against the CPU native path", executable_name_k);
    fmt::println("input: {} (cap {})", arguments.smiles_path, fmt::group_digits(arguments.max_molecules));

    // A corpus that never opened must not read as an empty corpus. It is read before the parameter tables are
    // extracted, so a refused path costs nothing.
    std::vector<std::string> smiles_batch;
    auto const load_start = clock_type_t::now();
    auto const load_status = sfbench::load_corpus_into(arguments.smiles_path, arguments.max_molecules, smiles_batch);
    auto const load_end = clock_type_t::now();
    if (load_status != sfbench::corpus_load_t::loaded_k) {
        fmt::println(stderr, "{}: corpus '{}' {}", executable_name_k, arguments.smiles_path,
                     sfbench::describe_corpus_load(load_status));
        print_usage_(argv[0]);
        return 2;
    }
    fmt::println("SMILES loaded: {} lines in {:.1f} ms", fmt::group_digits(smiles_batch.size()),
                 elapsed_ms_(load_start, load_end));

    // Table extraction once.
    sfparam::mmff94_tables_host_t tables_host;
    sfparam::extract_mmff94_tables_from_rdkit(tables_host);
    auto const tables_view = tables_host.view();
    fmt::println("tables extracted: vdw={} bond={} angle={} tor={}", tables_host.vdw.size(), tables_host.bond.size(),
                 tables_host.angle.size(), tables_host.torsion.size());

    auto const parse_start = clock_type_t::now();
    auto const parsed_molecules = parse_smiles_batch_(smiles_batch);
    auto const parse_end = clock_type_t::now();

    std::vector<sft::molecule_descriptor_t> descriptors;
    std::vector<std::uint32_t> atom_offsets;
    descriptors.reserve(parsed_molecules.size());
    atom_offsets.reserve(parsed_molecules.size());
    std::uint32_t total_atom_count = 0;
    for (auto const &parsed_molecule : parsed_molecules) {
        if (parsed_molecule.outcome != parse_outcome_t::parsed_k) continue; // no descriptor was built
        atom_offsets.push_back(total_atom_count);
        total_atom_count += parsed_molecule.descriptor.num_atoms;
        descriptors.push_back(parsed_molecule.descriptor);
    }
    std::uint32_t const num_molecules = static_cast<std::uint32_t>(descriptors.size());
    fmt::println("parsed: {} valid molecules / {} total atoms in {:.1f} ms", fmt::group_digits(num_molecules),
                 fmt::group_digits(total_atom_count), elapsed_ms_(parse_start, parse_end));
    print_parse_outcomes_(sfbench::parse_outcome_histogram_of_(parsed_molecules), parsed_molecules.size());

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
