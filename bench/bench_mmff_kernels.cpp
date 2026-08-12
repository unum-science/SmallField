/**
 *  @file bench/bench_mmff_kernels.cpp
 *  @brief GPU MMFF kernel-trio scaling benchmark: type-and-charge, term extraction and CSR build, timed
 *  against the CPU native path over the same batch.
 *
 *  The per-molecule fixed-stride output arrays are what bound the batch: ChEMBL scale, 240 K molecules,
 *  already takes about 12 GB of device memory.
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
    above_atom_ceiling_k,
    too_many_bonds_k,
    too_many_rings_k,
    /** `build_descriptor` refused the molecule for a reason this table does not tabulate. */
    no_descriptor_k,
};

/** Column label naming the outcome in the parse histogram. */
static char const *describe_parse_outcome_(parse_outcome_t outcome) {
    switch (outcome) {
    case parse_outcome_t::parsed_k: return "parsed";
    case parse_outcome_t::unparseable_k: return "unparseable";
    case parse_outcome_t::hydrogenation_failed_k: return "hydrogenation_failed";
    case parse_outcome_t::above_atom_ceiling_k: return "above_atom_ceiling";
    case parse_outcome_t::too_many_bonds_k: return "too_many_bonds";
    case parse_outcome_t::too_many_rings_k: return "too_many_rings";
    case parse_outcome_t::no_descriptor_k: return "no_descriptor";
    }
    return "other";
}

/** Carry `build_descriptor`'s refusal across, so the extent a corpus outgrew stays named in the histogram. */
static parse_outcome_t outcome_of_descriptor_(smallfield::api::conformer_status_t status) {
    switch (status) {
    case smallfield::api::conformer_status_t::success_k: return parse_outcome_t::parsed_k;
    case smallfield::api::conformer_status_t::too_many_atoms_k: return parse_outcome_t::above_atom_ceiling_k;
    case smallfield::api::conformer_status_t::too_many_bonds_k: return parse_outcome_t::too_many_bonds_k;
    case smallfield::api::conformer_status_t::too_many_rings_k: return parse_outcome_t::too_many_rings_k;
    default: return parse_outcome_t::no_descriptor_k;
    }
}

struct parsed_molecule_t {
    std::unique_ptr<RDKit::RWMol> rdkit_molecule;
    sft::molecule_descriptor_t descriptor;
    /** Only `parsed_k` carries a usable `descriptor`; every other outcome names why it is absent. */
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
            parsed.outcome = parse_outcome_t::above_atom_ceiling_k;
            continue;
        }
        parsed.outcome = outcome_of_descriptor_(sft::build_descriptor(*parsed.rdkit_molecule, parsed.descriptor));
        if (parsed.outcome != parse_outcome_t::parsed_k) parsed.rdkit_molecule.reset();
    }
    return parsed_molecules;
}

/** Molecules per parse outcome, indexed by the enumerator's own value. */
using parse_outcome_histogram_t = std::array<std::size_t, 7>;
static_assert(static_cast<std::size_t>(parse_outcome_t::no_descriptor_k) + 1 ==
                  std::tuple_size_v<parse_outcome_histogram_t>,
              "one bucket per outcome");

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

/** Ceiling on `--max-molecules`, so a mistyped count is a diagnostic rather than a late `cudaMalloc` failure. */
static constexpr std::size_t max_molecules_per_run_k = 1'000'000;

#pragma region Command Line

/** Everything this driver takes from the command line. */
struct kernel_bench_arguments_t {
    /** From `--smi` or from `SMALLFIELD_TEST_CORPUS`; there is no default. */
    std::string_view smiles_path;
    std::size_t max_molecules = 10'000;
};

using sfbench::arguments_parse_t;

static void print_usage_(char const *invocation) {
    fmt::println(stderr, "usage: {} --smi PATH [--max-molecules N]", invocation);
}

/** Fill @p arguments from `argv`, answer a help request, or explain on `stderr` why the line was refused. */
static arguments_parse_t parse_arguments_(int argc, char **argv, kernel_bench_arguments_t &arguments) {
    for (sfbench::command_line_t line {executable_name_k, argc, argv}; line.advance();) {
        arguments_parse_t status = arguments_parse_t::accepted_k;
        if (line.flag() == "--smi") status = line.take_text(arguments.smiles_path);
        else if (line.flag() == "--max-molecules")
            status = line.take_number(max_molecules_per_run_k, arguments.max_molecules);
        else if (line.flag() == "--help") {
            print_usage_(argv[0]);
            return arguments_parse_t::help_requested_k;
        }
        else {
            print_usage_(argv[0]);
            status = line.refuse_unknown();
        }
        if (status != arguments_parse_t::accepted_k) return arguments_parse_t::refused_k;
    }
    if (arguments.smiles_path.empty()) arguments.smiles_path = sfbench::corpus_path_from_environment();
    return arguments_parse_t::accepted_k;
}

#pragma endregion Command Line

#pragma region Timed Sections

/**
 *  @brief The batch the kernels are timed over: one descriptor per molecule that parsed, with the atom
 *  offsets addressing the shared atom arena.
 *
 *  Both arrays are walked by the same molecule index, and an offset is the running sum of its
 *  predecessors' atom counts.
 */
struct descriptor_batch_t {
    std::vector<sft::molecule_descriptor_t> descriptors;
    std::vector<std::uint32_t> atom_offsets;
    std::uint32_t total_atom_count = 0;

    [[nodiscard]] std::uint32_t num_molecules() const noexcept {
        return static_cast<std::uint32_t>(descriptors.size());
    }
};

/** Pack every molecule that parsed into @p batch, which is cleared first. */
static void fill_descriptor_batch_into_(std::vector<parsed_molecule_t> const &parsed_molecules,
                                        descriptor_batch_t &batch) {
    batch.descriptors.clear();
    batch.atom_offsets.clear();
    batch.total_atom_count = 0;
    batch.descriptors.reserve(parsed_molecules.size());
    batch.atom_offsets.reserve(parsed_molecules.size());
    for (auto const &parsed_molecule : parsed_molecules) {
        if (parsed_molecule.outcome != parse_outcome_t::parsed_k) continue; // no descriptor was built
        batch.atom_offsets.push_back(batch.total_atom_count);
        batch.total_atom_count += parsed_molecule.descriptor.num_atoms;
        batch.descriptors.push_back(parsed_molecule.descriptor);
    }
}

/**
 *  @brief Time the CPU native extraction over the whole batch and report its throughput.
 *
 *  The term counts are summed into a reduction, and the sum is printed, so `-O3` cannot dead-code the call
 *  body it sees go out of scope.
 *
 *  @return Molecules per second on the CPU path.
 */
static double time_cpu_native_path_(descriptor_batch_t const &batch, sfparam::mmff94_tables_view_t const &tables) {
    std::size_t checksum = 0;
    std::uint32_t const num_molecules = batch.num_molecules();
    auto const start = clock_type_t::now();
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : checksum)
    for (std::uint32_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        sfp::etkdg_prepared_molecule_t extracted_molecule;
        extracted_molecule.num_atoms = batch.descriptors[molecule_index].num_atoms;
        sfmmff::extract_mmff94_params_fully_native(batch.descriptors[molecule_index], tables, extracted_molecule);
        checksum += extracted_molecule.mmff_bonds.size() + extracted_molecule.mmff_angles.size() +
                    extracted_molecule.mmff_torsions.size() + extracted_molecule.mmff_oops.size() +
                    extracted_molecule.mmff_stretchbends.size() + extracted_molecule.mmff_interactions.size();
    }
    double const wall_ms = elapsed_ms_(start, clock_type_t::now());
    double const throughput = static_cast<double>(num_molecules) * 1000.0 / wall_ms;
    fmt::println("CPU sink (checksum): {}", fmt::group_digits(checksum));
    fmt::println("\nCPU native path:");
    fmt::println("  wall = {:.1f} ms  ({} OMP threads)", wall_ms, omp_get_max_threads());
    fmt::println("  {} cpu mol/s: {:.1f}", executable_name_k, throughput);
    return throughput;
}

/**
 *  @brief Time the GPU kernel trio over the same batch and report its per-stage breakdown.
 *
 *  @return Molecules per second, or zero when the launch named a refusal instead.
 */
static double time_gpu_kernel_trio_(descriptor_batch_t const &batch, sfparam::mmff94_tables_host_t const &tables) {
    sfbench::kernel_trio_timings_t timings;
    if (sfbench::trio_launch_t const status = sfbench::launch_mmff94_kernel_trio_benchmark(
            tables, batch.descriptors, batch.atom_offsets, batch.total_atom_count, timings);
        status != sfbench::trio_launch_t::timed_k) {
        fmt::println(stderr, "{}: the GPU kernel trio {}", executable_name_k, sfbench::describe_trio_launch(status));
        return 0.0;
    }
    double const throughput = static_cast<double>(batch.num_molecules()) * 1000.0 / timings.total_ms;
    fmt::println("\nGPU kernel trio:");
    fmt::println("  type+charge   = {:.1f} ms", timings.type_and_charge_ms);
    fmt::println("  extract_terms = {:.1f} ms", timings.extract_terms_ms);
    fmt::println("  build_csr     = {:.1f} ms", timings.build_csr_ms);
    fmt::println("  total         = {:.1f} ms", timings.total_ms);
    fmt::println("  {} gpu mol/s: {:.1f}", executable_name_k, throughput);
    return throughput;
}

#pragma endregion Timed Sections

int main(int argc, char **argv) {
    kernel_bench_arguments_t arguments;
    switch (parse_arguments_(argc, argv, arguments)) {
    case arguments_parse_t::accepted_k: break;
    case arguments_parse_t::help_requested_k: return 0;
    case arguments_parse_t::refused_k: return 2;
    }

    fmt::println("{}: GPU MMFF kernel trio against the CPU native path", executable_name_k);
    fmt::println("input: {} (cap {})", arguments.smiles_path, fmt::group_digits(arguments.max_molecules));

    // Read before the parameter tables are extracted, so a refused corpus path costs nothing.
    std::vector<std::string> smiles_batch;
    auto const load_start = clock_type_t::now();
    auto const load_status = sfbench::load_corpus_into(arguments.smiles_path, arguments.max_molecules, smiles_batch);
    if (load_status != sfbench::corpus_load_t::loaded_k) {
        fmt::println(stderr, "{}: corpus '{}' {}", executable_name_k, arguments.smiles_path,
                     sfbench::describe_corpus_load(load_status));
        print_usage_(argv[0]);
        return 2;
    }
    fmt::println("SMILES loaded: {} lines in {:.1f} ms", fmt::group_digits(smiles_batch.size()),
                 elapsed_ms_(load_start, clock_type_t::now()));

    sfparam::mmff94_tables_host_t tables_host;
    sfparam::extract_mmff94_tables_from_rdkit(tables_host);
    fmt::println("tables extracted: vdw={} bond={} angle={} tor={}", tables_host.vdw.size(), tables_host.bond.size(),
                 tables_host.angle.size(), tables_host.torsion.size());

    auto const parse_start = clock_type_t::now();
    auto const parsed_molecules = parse_smiles_batch_(smiles_batch);
    double const parse_ms = elapsed_ms_(parse_start, clock_type_t::now());

    descriptor_batch_t batch;
    fill_descriptor_batch_into_(parsed_molecules, batch);
    fmt::println("parsed: {} valid molecules / {} total atoms in {:.1f} ms", fmt::group_digits(batch.num_molecules()),
                 fmt::group_digits(batch.total_atom_count), parse_ms);
    print_parse_outcomes_(sfbench::parse_outcome_histogram_of_(parsed_molecules), parsed_molecules.size());
    if (batch.num_molecules() == 0) {
        fmt::println(stderr, "{}: no molecules — aborting", executable_name_k);
        return 1;
    }

    double const cpu_throughput = time_cpu_native_path_(batch, tables_host.view());
    double const gpu_throughput = time_gpu_kernel_trio_(batch, tables_host);
    fmt::println("\nspeedup (CPU→GPU): {:.2f} ×", cpu_throughput > 0.0 ? gpu_throughput / cpu_throughput : 0.0);
    return 0;
}
