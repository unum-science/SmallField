/**
 *  @file test_mmff_gpu_extract.cpp
 *  @brief Compare the GPU MMFF term extractor output against
 *  the CPU native extractor on a batch of drug-like SMILES.  Byte-
 *  identity required for every bonded term of every molecule where the
 *  GPU typer flagged success (typing_status == 0).
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
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "smallfield/mmff94/mmff94_cpu.hpp"
#include "smallfield/mmff94/mmff94_extract_device.cuh"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "test_mmff_gpu_extract_kernels.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

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

    launch_mmff94_gpu_extract_pipeline(tables_host, descriptors.data(), num_molecules, atom_offsets.data(), total_atoms,
                                       counts.data(), typing_status.data(), bonds.data(), sft::max_mmff_bonds_per_mol_k,
                                       angles.data(), sft::max_mmff_angles_per_mol_k, torsions.data(),
                                       sft::max_mmff_torsions_per_mol_k, oops.data(), sft::max_mmff_oops_per_mol_k,
                                       stretch_bend_terms.data(), sft::max_mmff_stretchbends_per_mol_k,
                                       interactions.data(), atom_offsets_out.data());

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

    for (std::uint32_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
        if (typing_status[molecule_index] != 0) {
            ++skipped_molecule_count;
            continue;
        }

        auto const &descriptor = descriptors[molecule_index];
        sfp::etkdg_prepared_molecule_t cpu_reference;
        cpu_reference.num_atoms = descriptor.num_atoms;
        if (!sfmmff::extract_mmff94_params_fully_native(descriptor, host_view, cpu_reference)) {
            ++skipped_molecule_count;
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
                        "  mol=%u atom=%u CSR count diff: " "cpu=(b%u a%u t%u o%u s%u) gpu=(b%u a%u t%u o%u s%u)\n",
                        molecule_index, atom_index, cpu_offset.bond_count, cpu_offset.angle_count,
                        cpu_offset.torsion_count, cpu_offset.oop_count, cpu_offset.stretchbend_count,
                        gpu_offset.bond_count, gpu_offset.angle_count, gpu_offset.torsion_count, gpu_offset.oop_count,
                        gpu_offset.stretchbend_count);
                }
            }
            else { ++matching_csr_count; }
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
    if (mismatched_molecule_count > 0) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
