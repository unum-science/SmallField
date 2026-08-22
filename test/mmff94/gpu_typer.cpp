/**
 *  @file test/mmff94/gpu_typer.cpp
 *  @brief Run the GPU MMFF typer kernel against the native CPU typer on a set of drug-like SMILES.
 *
 *  Gate: byte-identical atom types and per-atom partial charges.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`
#include <cstdio>  // `std::fflush`

#include <iostream>    // RDKit's `AtomTyper.h` needs it but does not include it
#include <memory>      // `std::unique_ptr`
#include <sstream>     // RDKit's `AtomTyper.h` needs it but does not include it
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <boost/shared_ptr.hpp>
#include <cuda_runtime.h>
#include <RDGeneral/types.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/mmff94/mmff94_typer.hpp"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_rdkit.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

#include "test/helpers.hpp"
#include "test/mmff94/gpu_typer_kernels.hpp"
#include "test/rdkit_reference.hpp"

namespace sfmmff = smallfield::mmff94;
namespace sfparam = smallfield::parameters;
namespace sft = smallfield::types;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "smallfield_test_gpu_typer";

namespace smallfield::tests {

/** How many mismatching fields either section prints before the rest are only counted. */
static constexpr std::size_t max_printed_lines_k = 32;

/** Widest molecule this sweep samples, which is not the descriptor's capacity. `sft::build_descriptor` refuses
 *  anything above `sft::max_atoms_per_molecule_k` on its own, so this guards no storage — it only keeps the
 *  compared population drug-like, and raising it to the capacity would widen that population, not fix an overflow. */
static constexpr std::uint32_t sampled_atom_ceiling_k = 128;
static_assert(sampled_atom_ceiling_k <= sft::max_atoms_per_molecule_k, "the sample ceiling sits under capacity");

/** Whether this molecule's GPU charges are meaningful, or were zeroed for the CPU fallback path. */
enum class charge_comparison_t : std::uint8_t {
    compared_k,
    skipped_for_fallback_k,
};

/** One reporter per compared section, so each closes with its own `compared=` count; the two counts are
 *  deliberately not derivable from one another, as `compare_molecule_` explains. */
struct typer_reporters_t {
    diagnostic_reporter_t atom_type {"cpu", "gpu"};
    diagnostic_reporter_t partial_charge {"cpu", "gpu"};

    /** Cap what each section prints, so one pathological molecule cannot bury the log. */
    void limit_printed_lines(std::size_t maximum) noexcept {
        atom_type.limit_printed_lines(maximum);
        partial_charge.limit_printed_lines(maximum);
    }

    /** Close both sections; true when each compared something and every field agreed. */
    bool summarize_sections() {
        atom_type.summarize("atom_type");
        partial_charge.summarize("partial_charge");
        return atom_type.passed() && partial_charge.passed();
    }
};

/** Per-molecule outcomes of the sweep. `matched` plus `mismatched` closes over the descriptors alone, not over
 *  `smiles_batch.size()`: `build_descriptor_batch_` drops what it cannot parse, hydrogenate, fit under
 *  `sampled_atom_ceiling_k` or describe, so the gap is what the sweep never compared and deriving either counter
 *  from the input size would report molecules as matching that were never typed. `fallback_flagged` counts those
 *  the GPU flagged for CPU fallback; their atom types are still compared, so it overlaps both buckets rather than
 *  partitioning the batch with them. */
struct typer_tally_t {
    std::uint32_t matched = 0;
    std::uint32_t mismatched = 0;
    std::uint32_t fallback_flagged = 0;
};

/** Reusable per-molecule scratch for the CPU typer, so the sweep allocates once rather than per molecule. */
struct cpu_typing_scratch_t {
    std::vector<std::uint8_t> atom_types;
    std::vector<float> formal_charges;
    std::vector<float> partial_charges;

    void resize(std::uint32_t num_atoms) {
        atom_types.assign(num_atoms, std::uint8_t {0});
        formal_charges.assign(num_atoms, 0.0f);
        partial_charges.assign(num_atoms, 0.0f);
    }
};

/** Parse the batch into descriptors, filling caller-owned outputs, and answer the atom total. A SMILES that will
 *  not parse, will not take hydrogens, exceeds `sampled_atom_ceiling_k`, or carries an element or bond order the
 *  descriptor does not model is dropped, so the descriptor count is not the SMILES count. */
static std::uint32_t build_descriptor_batch_(std::vector<std::string> const &smiles_batch,
                                             std::vector<sft::molecule_descriptor_t> &descriptors,
                                             std::vector<std::uint32_t> &atom_offsets) {
    descriptors.clear();
    atom_offsets.clear();
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
        if (!molecule) continue; // the parser refused it
        try {
            RDKit::MolOps::addHs(*molecule);
        }
        catch (...) {
            continue;
        }
        if (molecule->getNumAtoms() > sampled_atom_ceiling_k) continue; // wider than this sweep samples

        sft::molecule_descriptor_t descriptor;
        // an element or a bond order we do not carry
        if (sft::build_descriptor(*molecule, descriptor) != sft::conformer_status_t::success_k) continue;
        atom_offsets.push_back(total_atoms);
        total_atoms += descriptor.num_atoms;
        descriptors.push_back(descriptor);
    }
    return total_atoms;
}

/** Compare one molecule atom by atom, answering how many of its atoms disagree. The charge section's denominator is
 *  deliberately smaller than the atom section's and neither is recoverable from the other: a molecule flagged for
 *  CPU fallback has its GPU charges zeroed on purpose, and an atom whose type differs, or is zero, has no charge
 *  worth comparing. `compared` minus `mismatches` on the atom section is therefore not the charge section's count,
 *  and assuming it is would silently compare charges that mean nothing. */
static std::size_t compare_molecule_(typer_reporters_t &reporters, cpu_typing_scratch_t &scratch,
                                     std::size_t molecule_index, sft::molecule_descriptor_t const &descriptor,
                                     sfparam::mmff94_tables_view_t const &host_view, std::uint32_t atom_offset,
                                     std::uint8_t const *gpu_atom_types, float const *gpu_atom_charges,
                                     charge_comparison_t charges) {
    std::uint32_t const num_atoms = descriptor.num_atoms;
    scratch.resize(num_atoms);
    sfmmff::native_compute_mmff_atom_types(descriptor, scratch.atom_types);
    sfmmff::native_compute_mmff_formal_charges(descriptor, scratch.atom_types, scratch.formal_charges);
    sfmmff::native_compute_mmff_partial_charges(descriptor, host_view, scratch.atom_types, scratch.formal_charges,
                                                scratch.partial_charges);

    std::size_t mismatching_atoms = 0;
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        auto atom = reporters.atom_type.mismatch("mmff_typer", "molecule={} atom={} element={}", molecule_index,
                                                 atom_index, descriptor.element[atom_index]);
        atom.field("atom_type", scratch.atom_types[atom_index], gpu_atom_types[atom_offset + atom_index],
                   exactly_equal);
        bool const types_agree = atom.reported_field_count() == 0;
        mismatching_atoms += !types_agree;

        if (charges == charge_comparison_t::skipped_for_fallback_k) continue; // GPU zeroed these charges
        if (!types_agree) continue;                                           // a wrong type gives a wrong charge
        if (scratch.atom_types[atom_index] == 0) continue;                    // untyped atom, no charge to compare
        auto charge = reporters.partial_charge.mismatch("mmff_typer", "molecule={} atom={}", molecule_index,
                                                        atom_index);
        charge.field("partial_charge", scratch.partial_charges[atom_index], gpu_atom_charges[atom_offset + atom_index],
                     equal_within(mmff94_partial_charge_tolerance_k));
        mismatching_atoms += charge.reported_field_count() != 0;
    }
    return mismatching_atoms;
}

} // namespace smallfield::tests

int main(int argc, char **argv) {
    fmt::println("{}: GPU vs CPU MMFF typer byte-equality gate", test_name_k);

    auto const corpus = sftest::select_corpus(argc, argv, 500);
    if (corpus.knob == sftest::corpus_knob_t::refused_k) {
        std::fflush(stdout);
        fmt::println(stderr, "usage: {} [corpus.smi] [max_molecules]", test_name_k);
        return 2;
    }

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
    if (corpus.knob == sftest::corpus_knob_t::named_k) {
        std::vector<std::string> extra_smiles;
        auto const outcome = sftest::load_smiles_into(corpus.path, corpus.max_molecules, extra_smiles);
        if (outcome != sftest::smiles_load_t::loaded_k)
            return sftest::report_fail(test_name_k, "{} {}", sftest::smiles_load_reason(outcome), corpus.path);
        for (auto &smiles : extra_smiles) smiles_batch.push_back(std::move(smiles));
    }

    sfparam::mmff94_tables_host_t tables_host {};
    sfparam::extract_mmff94_tables_from_rdkit(tables_host);
    sfparam::mmff94_tables_view_t host_view = tables_host.view();

    sfparam::mmff94_tables_view_t device_view {};
    void *device_tables_handle = sftest::create_mmff94_device_tables_view(tables_host, device_view);
    if (!device_tables_handle) return sftest::report_fail(test_name_k, "device MMFF94 table upload failed");

    std::vector<sft::molecule_descriptor_t> descriptors;
    std::vector<std::uint32_t> atom_offsets;
    std::uint32_t const total_atoms = sftest::build_descriptor_batch_(smiles_batch, descriptors, atom_offsets);
    std::uint32_t const num_molecules = static_cast<std::uint32_t>(descriptors.size());

    fmt::println("descriptors: {} molecules, {} total atoms", fmt::group_digits(num_molecules),
                 fmt::group_digits(total_atoms));

    std::vector<std::uint8_t> gpu_atom_types(total_atoms);
    std::vector<float> gpu_atom_charges(total_atoms);
    std::vector<std::uint8_t> gpu_typing_status(num_molecules);

    // Zero is the typer's verdict for a molecule it accepted, so a refused launch would leave the status array
    // claiming the whole batch was typed and the sweep would compare uninitialised types.
    auto const probe = sftest::launch_mmff94_gpu_typer_batch(device_view, descriptors, atom_offsets, gpu_atom_types,
                                                             gpu_atom_charges, gpu_typing_status);
    if (probe != sftest::typer_probe_t::completed_k) {
        sftest::destroy_mmff94_device_tables_view(device_tables_handle);
        return sftest::report_fail(test_name_k, "the device refused the typer batch: {}",
                                   sftest::typer_probe_reason(probe));
    }

    sftest::typer_reporters_t reporters;
    reporters.limit_printed_lines(sftest::max_printed_lines_k);
    sftest::typer_tally_t tally;
    sftest::cpu_typing_scratch_t scratch;

    for (std::uint32_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        bool const flagged_for_fallback = gpu_typing_status[molecule_index] != 0;
        if (flagged_for_fallback) ++tally.fallback_flagged;
        auto const charges = flagged_for_fallback ? sftest::charge_comparison_t::skipped_for_fallback_k
                                                  : sftest::charge_comparison_t::compared_k;
        std::size_t const mismatching_atoms = sftest::compare_molecule_(
            reporters, scratch, molecule_index, descriptors[molecule_index], host_view, atom_offsets[molecule_index],
            gpu_atom_types.data(), gpu_atom_charges.data(), charges);
        if (mismatching_atoms == 0) ++tally.matched;
        else ++tally.mismatched;
    }

    fmt::println("fallback-flagged molecules (typing_status != 0): {} / {}", fmt::group_digits(tally.fallback_flagged),
                 fmt::group_digits(num_molecules));
    fmt::println("molecules: {} match / {} mismatch", fmt::group_digits(tally.matched),
                 fmt::group_digits(tally.mismatched));
    bool const sections_passed = reporters.summarize_sections();

    sftest::destroy_mmff94_device_tables_view(device_tables_handle);
    if (tally.mismatched > 0)
        return sftest::report_fail(test_name_k, "{} of {} molecules diverge from the CPU typer", tally.mismatched,
                                   num_molecules);
    if (!sections_passed)
        return sftest::report_fail(test_name_k, "a comparison section mismatched or compared nothing at all");
    return sftest::report_pass(test_name_k);
}
