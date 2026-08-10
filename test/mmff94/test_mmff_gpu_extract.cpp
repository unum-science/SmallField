/**
 *  @file test_mmff_gpu_extract.cpp
 *  @brief Compare the GPU MMFF term extractor against the CPU native extractor.
 *
 *  Runs a batch of drug-like SMILES through both paths. Byte identity is required for every bonded
 *  term of every molecule where the GPU typer flagged success, meaning `typing_status == 0`.
 */

#include <cmath>   // `std::fabs`
#include <cstdio>  // `printf`
#include <cstring> // `std::memcmp`

#include <algorithm> // `std::max`, `std::unique`
#include <fstream>   // `std::ifstream`
#include <memory>    // `std::unique_ptr`
#include <string>    // `std::string`, `std::getline`
#include <vector>    // `std::vector`

#include <boost/shared_ptr.hpp>
#include <cuda_runtime.h>
#include <RDGeneral/types.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/mmff94/mmff94_cpu.hpp"
#include "smallfield/mmff94/mmff94_cpu_extract.hpp"
#include "smallfield/mmff94/mmff94_extract_device.cuh"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_rdkit.hpp"
#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

#include "test_mmff_gpu_extract_kernels.hpp"

namespace sf = smallfield;
namespace sfmmff = smallfield::mmff94;
namespace sfparam = smallfield::parameters;
namespace sfp = smallfield::preprocess;
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
    std::printf("=== test_mmff_gpu_extract: GPU vs CPU MMFF extractor byte-equality gate ===\n");

    sfparam::mmff94_tables_host_t tables_host {};
    sfparam::extract_mmff94_tables_from_rdkit(tables_host);
    sfparam::mmff94_tables_view_t host_view = tables_host.view();

    std::vector<std::string> smiles_batch = {
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "CC(=O)NC1=CC=C(O)C=C1",
        "CC(=O)[O-]",
        "c1cc[nH+]c(c1)",
        "C1CCOC1",
        "C1CC1",
        "c1ccc2[nH]ccc2c1",
    };
    if (argc > 1) {
        auto extra_smiles = load_smiles(argv[1], argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 200);
        for (auto &smiles : extra_smiles) smiles_batch.push_back(std::move(smiles));
    }

    std::vector<sft::molecule_descriptor_t> descriptors;
    std::vector<std::uint32_t> atom_offsets;
    std::vector<std::unique_ptr<RDKit::RWMol>> molecules;
    descriptors.reserve(smiles_batch.size());
    atom_offsets.reserve(smiles_batch.size());
    molecules.reserve(smiles_batch.size());
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
        molecules.push_back(std::move(molecule));
    }
    std::uint32_t const num_molecules = static_cast<std::uint32_t>(descriptors.size());

    std::printf("batch: %u molecules, %u total atoms\n", num_molecules, total_atoms);

    std::vector<sfmmff::mmff94_per_mol_counts_t> counts(num_molecules);
    std::vector<std::uint8_t> typing_status(num_molecules);
    std::vector<sft::mmff94_bond_t> bonds(static_cast<std::size_t>(num_molecules) * sft::max_mmff_bonds_per_mol_k);
    std::vector<sft::mmff94_angle_t> angles(static_cast<std::size_t>(num_molecules) * sft::max_mmff_angles_per_mol_k);
    std::vector<sft::mmff94_torsion_t> torsions(static_cast<std::size_t>(num_molecules) *
                                                sft::max_mmff_torsions_per_mol_k);
    std::vector<sft::mmff94_oop_t> oops(static_cast<std::size_t>(num_molecules) * sft::max_mmff_oops_per_mol_k);
    std::vector<sft::mmff94_stretchbend_t> stretch_bend_terms(static_cast<std::size_t>(num_molecules) *
                                                              sft::max_mmff_stretchbends_per_mol_k);
    std::vector<sft::mmff94_atom_interaction_t> interactions(static_cast<std::size_t>(num_molecules) *
                                                             sft::max_mmff_interactions_per_mol_k);
    std::vector<sft::mmff94_atom_offsets_t> atom_offsets_out(total_atoms);

    mmff94_packed_host_result_t packed;
    launch_mmff94_gpu_extract_pipeline(tables_host, descriptors.data(), num_molecules, atom_offsets.data(), total_atoms,
                                       counts.data(), typing_status.data(), bonds.data(), sft::max_mmff_bonds_per_mol_k,
                                       angles.data(), sft::max_mmff_angles_per_mol_k, torsions.data(),
                                       sft::max_mmff_torsions_per_mol_k, oops.data(), sft::max_mmff_oops_per_mol_k,
                                       stretch_bend_terms.data(), sft::max_mmff_stretchbends_per_mol_k,
                                       interactions.data(), atom_offsets_out.data(), packed);

    // CPU reference via extract_mmff94_params_fully_native.
    int matching_molecule_count = 0;
    int skipped_molecule_count = 0;
    int mismatched_molecule_count = 0;
    int matching_bond_count = 0;
    int mismatched_bond_count = 0;
    int matching_angle_count = 0;
    int mismatched_angle_count = 0;
    int matching_torsion_count = 0;
    int mismatched_torsion_count = 0;
    int matching_oop_count = 0;
    int mismatched_oop_count = 0;
    int matching_stretch_bend_count = 0;
    int mismatched_stretch_bend_count = 0;
    int matching_csr_count = 0;
    int mismatched_csr_count = 0;
    int matching_packed_count = 0;
    int mismatched_packed_count = 0;
    int typer_fallback_count = 0;
    int extractor_fallback_count = 0;

    for (std::uint32_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        if (typing_status[molecule_index] != 0) {
            ++skipped_molecule_count;
            ++typer_fallback_count;
            continue;
        }

        auto const &descriptor = descriptors[molecule_index];
        sfp::etkdg_prepared_molecule_t cpu_reference;
        cpu_reference.num_atoms = descriptor.num_atoms;
        if (!sfmmff::extract_mmff94_params_fully_native(descriptor, host_view, cpu_reference)) {
            ++skipped_molecule_count;
            ++extractor_fallback_count;
            continue;
        }

        auto const &gpu_counts = counts[molecule_index];
        bool molecule_matches = true;
        auto const require_count_match = [&](char const *tag, std::size_t cpu_count, std::size_t gpu_count) {
            if (cpu_count != gpu_count) {
                std::printf("  mol=%u %s count diff: cpu=%zu gpu=%zu\n", molecule_index, tag, cpu_count, gpu_count);
                molecule_matches = false;
                return false;
            }
            return true;
        };
        if (!require_count_match("bond", cpu_reference.mmff_bonds.size(), gpu_counts.bonds)) {
            ++mismatched_molecule_count;
            continue;
        }
        if (!require_count_match("angle", cpu_reference.mmff_angles.size(), gpu_counts.angles)) {
            ++mismatched_molecule_count;
            continue;
        }
        if (!require_count_match("torsion", cpu_reference.mmff_torsions.size(), gpu_counts.torsions)) {
            ++mismatched_molecule_count;
            continue;
        }
        if (!require_count_match("oop", cpu_reference.mmff_oops.size(), gpu_counts.oops)) {
            ++mismatched_molecule_count;
            continue;
        }
        if (!require_count_match("stbn", cpu_reference.mmff_stretchbends.size(), gpu_counts.stretchbends)) {
            ++mismatched_molecule_count;
            continue;
        }

        auto const bond_terms = bonds.data() + static_cast<std::size_t>(molecule_index) * sft::max_mmff_bonds_per_mol_k;
        auto const angle_terms = angles.data() +
                                 static_cast<std::size_t>(molecule_index) * sft::max_mmff_angles_per_mol_k;
        auto const torsion_terms = torsions.data() +
                                   static_cast<std::size_t>(molecule_index) * sft::max_mmff_torsions_per_mol_k;
        auto const oop_terms = oops.data() + static_cast<std::size_t>(molecule_index) * sft::max_mmff_oops_per_mol_k;
        auto const stretch_bend_terms_slot = stretch_bend_terms.data() + static_cast<std::size_t>(molecule_index) *
                                                                             sft::max_mmff_stretchbends_per_mol_k;

        for (std::size_t term_index = 0; term_index < cpu_reference.mmff_bonds.size(); ++term_index) {
            auto const &cpu_term = cpu_reference.mmff_bonds[term_index];
            auto const &gpu_term = bond_terms[term_index];
            if (cpu_term.atom_i == gpu_term.atom_i && cpu_term.atom_j == gpu_term.atom_j &&
                approximately_equal(cpu_term.force_constant, gpu_term.force_constant) &&
                approximately_equal(cpu_term.equilibrium_distance, gpu_term.equilibrium_distance)) {
                ++matching_bond_count;
            }
            else {
                ++mismatched_bond_count;
                molecule_matches = false;
            }
        }
        for (std::size_t term_index = 0; term_index < cpu_reference.mmff_angles.size(); ++term_index) {
            auto const &cpu_term = cpu_reference.mmff_angles[term_index];
            auto const &gpu_term = angle_terms[term_index];
            if (cpu_term.atom_i == gpu_term.atom_i && cpu_term.atom_j == gpu_term.atom_j &&
                cpu_term.atom_k == gpu_term.atom_k &&
                approximately_equal(cpu_term.force_constant, gpu_term.force_constant) &&
                approximately_equal(cpu_term.equilibrium_angle, gpu_term.equilibrium_angle)) {
                ++matching_angle_count;
            }
            else {
                ++mismatched_angle_count;
                molecule_matches = false;
            }
        }
        for (std::size_t term_index = 0; term_index < cpu_reference.mmff_torsions.size(); ++term_index) {
            auto const &cpu_term = cpu_reference.mmff_torsions[term_index];
            auto const &gpu_term = torsion_terms[term_index];
            if (cpu_term.atom_i == gpu_term.atom_i && cpu_term.atom_j == gpu_term.atom_j &&
                cpu_term.atom_k == gpu_term.atom_k && cpu_term.atom_l == gpu_term.atom_l &&
                approximately_equal(cpu_term.v1, gpu_term.v1) && approximately_equal(cpu_term.v2, gpu_term.v2) &&
                approximately_equal(cpu_term.v3, gpu_term.v3)) {
                ++matching_torsion_count;
            }
            else {
                ++mismatched_torsion_count;
                molecule_matches = false;
            }
        }
        for (std::size_t term_index = 0; term_index < cpu_reference.mmff_oops.size(); ++term_index) {
            auto const &cpu_term = cpu_reference.mmff_oops[term_index];
            auto const &gpu_term = oop_terms[term_index];
            if (cpu_term.atom_i == gpu_term.atom_i && cpu_term.atom_j == gpu_term.atom_j &&
                cpu_term.atom_k == gpu_term.atom_k && cpu_term.atom_l == gpu_term.atom_l &&
                approximately_equal(cpu_term.force_constant, gpu_term.force_constant)) {
                ++matching_oop_count;
            }
            else {
                ++mismatched_oop_count;
                molecule_matches = false;
            }
        }
        for (std::size_t term_index = 0; term_index < cpu_reference.mmff_stretchbends.size(); ++term_index) {
            auto const &cpu_term = cpu_reference.mmff_stretchbends[term_index];
            auto const &gpu_term = stretch_bend_terms_slot[term_index];
            if (cpu_term.atom_i == gpu_term.atom_i && cpu_term.atom_j == gpu_term.atom_j &&
                cpu_term.atom_k == gpu_term.atom_k && approximately_equal(cpu_term.kba_ijk, gpu_term.kba_ijk) &&
                approximately_equal(cpu_term.kba_kji, gpu_term.kba_kji) &&
                approximately_equal(cpu_term.equilibrium_angle, gpu_term.equilibrium_angle) &&
                approximately_equal(cpu_term.equilibrium_distance_ij, gpu_term.equilibrium_distance_ij) &&
                approximately_equal(cpu_term.equilibrium_distance_kj, gpu_term.equilibrium_distance_kj)) {
                ++matching_stretch_bend_count;
            }
            else {
                ++mismatched_stretch_bend_count;
                molecule_matches = false;
            }
        }

        // CSR comparison. The CPU's `extract_mmff94_params_fully_native`
        // produces `mmff_interactions` and `mmff_atom_offsets`; we diff
        // the GPU CSR against it. Per-atom ordering is deterministic
        // on both sides (atoms walked in index order, term types in
        // bond/angle/torsion/oop/sb order).
        auto const atom_offset_base = atom_offsets[molecule_index];
        for (std::uint32_t atom_index = 0; atom_index < descriptor.num_atoms; ++atom_index) {
            auto const &cpu_offset = cpu_reference.mmff_atom_offsets[atom_index];
            auto const &gpu_offset = atom_offsets_out[atom_offset_base + atom_index];
            // GPU offsets are per-molecule (relative to this mol's
            // `max_mmff_interactions_per_mol_k` slot); CPU offsets are
            // per-molecule as well (into the mol's own interactions
            // vector). Both start at 0 per molecule, so the counts are
            // the comparison metric.
            if (cpu_offset.bond_count != gpu_offset.bond_count || cpu_offset.angle_count != gpu_offset.angle_count ||
                cpu_offset.torsion_count != gpu_offset.torsion_count || cpu_offset.oop_count != gpu_offset.oop_count ||
                cpu_offset.stretchbend_count != gpu_offset.stretchbend_count) {
                ++mismatched_csr_count;
                molecule_matches = false;
                if (mismatched_csr_count < 5) {
                    std::printf(
                        "  mol=%u atom=%u CSR count diff: cpu=(b%u a%u t%u o%u s%u) gpu=(b%u a%u t%u o%u s%u)\n",
                        molecule_index, atom_index, cpu_offset.bond_count, cpu_offset.angle_count,
                        cpu_offset.torsion_count, cpu_offset.oop_count, cpu_offset.stretchbend_count,
                        gpu_offset.bond_count, gpu_offset.angle_count, gpu_offset.torsion_count, gpu_offset.oop_count,
                        gpu_offset.stretchbend_count);
                }
            }
            else { ++matching_csr_count; }
        }

        // Packed-layout check. The staged slices above are already known good
        // against the CPU, so the compaction pass is correct exactly when the
        // packed arrays reproduce them byte for byte at the metadata-declared
        // offsets, with every CSR start shifted into the batch-global
        // interaction array.
        auto const &meta = packed.metadata[molecule_index];
        bool packed_matches = meta.num_mmff_bonds == gpu_counts.bonds && meta.num_mmff_angles == gpu_counts.angles &&
                              meta.num_mmff_torsions == gpu_counts.torsions && meta.num_mmff_oops == gpu_counts.oops &&
                              meta.num_mmff_stretchbends == gpu_counts.stretchbends &&
                              meta.mmff_atom_csr_offset == atom_offset_base;
        auto const slices_equal = [&](void const *staged, void const *packed_slice, std::size_t bytes) {
            return std::memcmp(staged, packed_slice, bytes) == 0;
        };
        if (packed_matches) {
            packed_matches = slices_equal(bond_terms, packed.bonds.data() + meta.mmff_bond_offset,
                                          gpu_counts.bonds * sizeof(sft::mmff94_bond_t)) &&
                             slices_equal(angle_terms, packed.angles.data() + meta.mmff_angle_offset,
                                          gpu_counts.angles * sizeof(sft::mmff94_angle_t)) &&
                             slices_equal(torsion_terms, packed.torsions.data() + meta.mmff_torsion_offset,
                                          gpu_counts.torsions * sizeof(sft::mmff94_torsion_t)) &&
                             slices_equal(oop_terms, packed.oops.data() + meta.mmff_oop_offset,
                                          gpu_counts.oops * sizeof(sft::mmff94_oop_t)) &&
                             slices_equal(stretch_bend_terms_slot,
                                          packed.stretchbends.data() + meta.mmff_stretchbend_offset,
                                          gpu_counts.stretchbends * sizeof(sft::mmff94_stretchbend_t));
        }
        if (packed_matches && descriptor.num_atoms > 0) {
            std::uint32_t const interaction_base = packed.atom_offsets[atom_offset_base].bond_offset;
            packed_matches = slices_equal(
                interactions.data() + static_cast<std::size_t>(molecule_index) * sft::max_mmff_interactions_per_mol_k,
                packed.interactions.data() + interaction_base,
                gpu_counts.interactions * sizeof(sft::mmff94_atom_interaction_t));
            for (std::uint32_t atom_index = 0; atom_index < descriptor.num_atoms && packed_matches; ++atom_index) {
                auto const &staged_offset = atom_offsets_out[atom_offset_base + atom_index];
                auto const &packed_offset = packed.atom_offsets[atom_offset_base + atom_index];
                packed_matches = packed_offset.bond_offset == staged_offset.bond_offset + interaction_base &&
                                 packed_offset.angle_offset == staged_offset.angle_offset + interaction_base &&
                                 packed_offset.torsion_offset == staged_offset.torsion_offset + interaction_base &&
                                 packed_offset.oop_offset == staged_offset.oop_offset + interaction_base &&
                                 packed_offset.stretchbend_offset ==
                                     staged_offset.stretchbend_offset + interaction_base;
            }
        }
        if (packed_matches) { ++matching_packed_count; }
        else {
            ++mismatched_packed_count;
            molecule_matches = false;
            if (mismatched_packed_count < 5) std::printf("  mol=%u packed layout diff\n", molecule_index);
        }

        if (molecule_matches) ++matching_molecule_count;
        else ++mismatched_molecule_count;
    }

    std::printf("molecules: %d match / %d bad / %d skipped-to-fallback\n", matching_molecule_count,
                mismatched_molecule_count, skipped_molecule_count);
    std::printf("bonds:    %d / %d\n", matching_bond_count, mismatched_bond_count);
    std::printf("angles:   %d / %d\n", matching_angle_count, mismatched_angle_count);
    std::printf("torsions: %d / %d\n", matching_torsion_count, mismatched_torsion_count);
    std::printf("oops:     %d / %d\n", matching_oop_count, mismatched_oop_count);
    std::printf("stbns:    %d / %d\n", matching_stretch_bend_count, mismatched_stretch_bend_count);
    std::printf("csr atoms: %d / %d\n", matching_csr_count, mismatched_csr_count);
    std::printf("packed:   %d / %d\n", matching_packed_count, mismatched_packed_count);

    // Footprint accounting: the staging arrays are fixed-stride, so their cost is
    // the same for every molecule; the packed arrays cost what the batch actually
    // holds. Both numbers are per molecule so the ratio is the compaction win.
    std::size_t const staged_bytes_per_molecule =
        sft::max_mmff_bonds_per_mol_k * sizeof(sft::mmff94_bond_t) +
        sft::max_mmff_angles_per_mol_k * sizeof(sft::mmff94_angle_t) +
        sft::max_mmff_torsions_per_mol_k * sizeof(sft::mmff94_torsion_t) +
        sft::max_mmff_oops_per_mol_k * sizeof(sft::mmff94_oop_t) +
        sft::max_mmff_stretchbends_per_mol_k * sizeof(sft::mmff94_stretchbend_t) +
        sft::max_mmff_interactions_per_mol_k * sizeof(sft::mmff94_atom_interaction_t);
    std::size_t const packed_bytes = packed.bonds.size() * sizeof(sft::mmff94_bond_t) +
                                     packed.angles.size() * sizeof(sft::mmff94_angle_t) +
                                     packed.torsions.size() * sizeof(sft::mmff94_torsion_t) +
                                     packed.oops.size() * sizeof(sft::mmff94_oop_t) +
                                     packed.stretchbends.size() * sizeof(sft::mmff94_stretchbend_t) +
                                     packed.interactions.size() * sizeof(sft::mmff94_atom_interaction_t);
    std::printf("bytes/mol: staged %zu, packed %.1f (%.1fx)\n", staged_bytes_per_molecule,
                static_cast<double>(packed_bytes) / num_molecules,
                static_cast<double>(staged_bytes_per_molecule) * num_molecules / static_cast<double>(packed_bytes));
    std::printf("fallback: typer %.2f %% (%d), extractor %.2f %% (%d), combined %.2f %% of %u\n",
                100.0 * typer_fallback_count / num_molecules, typer_fallback_count,
                100.0 * extractor_fallback_count / num_molecules, extractor_fallback_count,
                100.0 * skipped_molecule_count / num_molecules, num_molecules);
    if (mismatched_molecule_count > 0) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
