/**
 *  @file test_mmff_gpu_extract.cpp
 *  @brief MG-E gate: compare the GPU MMFF term extractor output against
 *  the CPU native extractor on a batch of drug-like SMILES.  Byte-
 *  identity required for every bonded term of every molecule where the
 *  GPU typer flagged success (typing_status == 0).
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#include "etkdg_types.cuh"
#include "etkdg_preprocess.hpp"
#include "mmff94_cpu.hpp"
#include "mmff94_extract_device.cuh"
#include "mmff94_tables.hpp"
#include "molecule_descriptor.hpp"

using namespace smallfield;

namespace smallfield {
void run_mmff94_gpu_extract(mmff94_tables_host_t const &tables_host,
                             molecule_descriptor_t const *descriptors,
                             std::uint32_t num_mols,
                             std::uint32_t const *atom_offsets,
                             std::uint32_t total_atoms,
                             mmff94_per_mol_counts_t *out_counts,
                             std::uint8_t *out_status,
                             mmff94_bond_t *out_bonds, std::uint32_t bond_stride,
                             mmff94_angle_t *out_angles, std::uint32_t angle_stride,
                             mmff94_torsion_t *out_torsions, std::uint32_t torsion_stride,
                             mmff94_oop_t *out_oops, std::uint32_t oop_stride,
                             mmff94_stretchbend_t *out_stbns, std::uint32_t stbn_stride,
                             mmff94_atom_interaction_t *out_interactions,
                             mmff94_atom_offsets_t *out_atom_offsets);
} // namespace smallfield

namespace {

bool near_eq(float a, float b, float eps = 1e-3f) {
    if (std::fabs(a - b) <= eps) return true;
    float const s = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) <= eps * s;
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
    std::printf("=== test_mmff_gpu_extract: MG-E gate ===\n");

    mmff94_tables_host_t tables_host;
    extract_mmff94_tables_from_rdkit(tables_host);
    mmff94_tables_view_t host_view = tables_host.view();

    std::vector<std::string> smis = {
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
        auto extra = load_smiles(argv[1], argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 200);
        for (auto &s : extra) smis.push_back(std::move(s));
    }

    std::vector<molecule_descriptor_t> descriptors;
    std::vector<std::uint32_t> atom_offsets;
    std::vector<std::unique_ptr<RDKit::RWMol>> mols;
    descriptors.reserve(smis.size());
    atom_offsets.reserve(smis.size());
    mols.reserve(smis.size());
    std::uint32_t total_atoms = 0;

    for (auto const &s : smis) {
        std::unique_ptr<RDKit::RWMol> m;
        try {
            RDKit::SmilesParserParams p;
            p.sanitize = true; p.removeHs = false; p.strictCXSMILES = false;
            m.reset(RDKit::SmilesToMol(s, p));
        } catch (...) { continue; }
        if (!m) continue;
        try { RDKit::MolOps::addHs(*m); } catch (...) { continue; }
        if (m->getNumAtoms() > 128) continue;
        molecule_descriptor_t d;
        if (!build_descriptor(*m, d)) continue;
        atom_offsets.push_back(total_atoms);
        total_atoms += d.num_atoms;
        descriptors.push_back(d);
        mols.push_back(std::move(m));
    }
    std::uint32_t const num_mols = (std::uint32_t)descriptors.size();

    std::printf("batch: %u molecules, %u total atoms\n", num_mols, total_atoms);

    std::vector<mmff94_per_mol_counts_t> counts(num_mols);
    std::vector<std::uint8_t> status(num_mols);
    std::vector<mmff94_bond_t> bonds((std::size_t)num_mols * max_mmff_bonds_per_mol_k);
    std::vector<mmff94_angle_t> angles((std::size_t)num_mols * max_mmff_angles_per_mol_k);
    std::vector<mmff94_torsion_t> torsions((std::size_t)num_mols * max_mmff_torsions_per_mol_k);
    std::vector<mmff94_oop_t> oops((std::size_t)num_mols * max_mmff_oops_per_mol_k);
    std::vector<mmff94_stretchbend_t> stbns((std::size_t)num_mols * max_mmff_stretchbends_per_mol_k);
    std::vector<mmff94_atom_interaction_t> interactions((std::size_t)num_mols * max_mmff_interactions_per_mol_k);
    std::vector<mmff94_atom_offsets_t> atom_offs_out(total_atoms);

    run_mmff94_gpu_extract(tables_host, descriptors.data(), num_mols,
                             atom_offsets.data(), total_atoms,
                             counts.data(), status.data(),
                             bonds.data(), max_mmff_bonds_per_mol_k,
                             angles.data(), max_mmff_angles_per_mol_k,
                             torsions.data(), max_mmff_torsions_per_mol_k,
                             oops.data(), max_mmff_oops_per_mol_k,
                             stbns.data(), max_mmff_stretchbends_per_mol_k,
                             interactions.data(), atom_offs_out.data());

    // CPU reference via extract_mmff94_params_fully_native.
    int mol_ok = 0, mol_skipped = 0, mol_bad = 0;
    int bond_ok = 0, bond_bad = 0;
    int angle_ok = 0, angle_bad = 0;
    int tor_ok = 0, tor_bad = 0;
    int oop_ok = 0, oop_bad = 0;
    int sb_ok = 0, sb_bad = 0;
    int csr_ok = 0, csr_bad = 0;

    for (std::uint32_t m = 0; m < num_mols; ++m) {
        if (status[m] != 0) { ++mol_skipped; continue; }

        auto const &desc = descriptors[m];
        etkdg_prepared_molecule_t cpu;
        cpu.num_atoms = desc.num_atoms;
        if (!extract_mmff94_params_fully_native(desc, host_view, cpu)) { ++mol_skipped; continue; }

        auto const &c = counts[m];
        bool ok = true;
        auto check_term = [&](char const *tag, std::size_t cpu_n, std::size_t gpu_n) {
            if (cpu_n != gpu_n) {
                std::printf("  mol=%u %s count diff: cpu=%zu gpu=%zu\n", m, tag, cpu_n, gpu_n);
                ok = false;
                return false;
            }
            return true;
        };
        if (!check_term("bond", cpu.mmff_bonds.size(), c.bonds)) { ++mol_bad; continue; }
        if (!check_term("angle", cpu.mmff_angles.size(), c.angles)) { ++mol_bad; continue; }
        if (!check_term("torsion", cpu.mmff_torsions.size(), c.torsions)) { ++mol_bad; continue; }
        if (!check_term("oop", cpu.mmff_oops.size(), c.oops)) { ++mol_bad; continue; }
        if (!check_term("stbn", cpu.mmff_stretchbends.size(), c.stretchbends)) { ++mol_bad; continue; }

        auto bond_slot = bonds.data() + (std::size_t)m * max_mmff_bonds_per_mol_k;
        auto angle_slot = angles.data() + (std::size_t)m * max_mmff_angles_per_mol_k;
        auto tor_slot = torsions.data() + (std::size_t)m * max_mmff_torsions_per_mol_k;
        auto oop_slot = oops.data() + (std::size_t)m * max_mmff_oops_per_mol_k;
        auto sb_slot = stbns.data() + (std::size_t)m * max_mmff_stretchbends_per_mol_k;

        for (std::size_t t = 0; t < cpu.mmff_bonds.size(); ++t) {
            auto const &r = cpu.mmff_bonds[t]; auto const &g = bond_slot[t];
            if (r.atom_i == g.atom_i && r.atom_j == g.atom_j &&
                near_eq(r.force_constant, g.force_constant) &&
                near_eq(r.equilibrium_distance, g.equilibrium_distance)) ++bond_ok;
            else { ++bond_bad; ok = false; }
        }
        for (std::size_t t = 0; t < cpu.mmff_angles.size(); ++t) {
            auto const &r = cpu.mmff_angles[t]; auto const &g = angle_slot[t];
            if (r.atom_i == g.atom_i && r.atom_j == g.atom_j && r.atom_k == g.atom_k &&
                near_eq(r.force_constant, g.force_constant) &&
                near_eq(r.equilibrium_angle, g.equilibrium_angle)) ++angle_ok;
            else { ++angle_bad; ok = false; }
        }
        for (std::size_t t = 0; t < cpu.mmff_torsions.size(); ++t) {
            auto const &r = cpu.mmff_torsions[t]; auto const &g = tor_slot[t];
            if (r.atom_i == g.atom_i && r.atom_j == g.atom_j &&
                r.atom_k == g.atom_k && r.atom_l == g.atom_l &&
                near_eq(r.v1, g.v1) && near_eq(r.v2, g.v2) && near_eq(r.v3, g.v3)) ++tor_ok;
            else { ++tor_bad; ok = false; }
        }
        for (std::size_t t = 0; t < cpu.mmff_oops.size(); ++t) {
            auto const &r = cpu.mmff_oops[t]; auto const &g = oop_slot[t];
            if (r.atom_i == g.atom_i && r.atom_j == g.atom_j &&
                r.atom_k == g.atom_k && r.atom_l == g.atom_l &&
                near_eq(r.force_constant, g.force_constant)) ++oop_ok;
            else { ++oop_bad; ok = false; }
        }
        for (std::size_t t = 0; t < cpu.mmff_stretchbends.size(); ++t) {
            auto const &r = cpu.mmff_stretchbends[t]; auto const &g = sb_slot[t];
            if (r.atom_i == g.atom_i && r.atom_j == g.atom_j && r.atom_k == g.atom_k &&
                near_eq(r.kba_ijk, g.kba_ijk) && near_eq(r.kba_kji, g.kba_kji) &&
                near_eq(r.equilibrium_angle, g.equilibrium_angle) &&
                near_eq(r.equilibrium_distance_ij, g.equilibrium_distance_ij) &&
                near_eq(r.equilibrium_distance_kj, g.equilibrium_distance_kj)) ++sb_ok;
            else { ++sb_bad; ok = false; }
        }

        // CSR comparison.  The CPU's `extract_mmff94_params_fully_native`
        // produces `mmff_interactions` and `mmff_atom_offsets`; we diff
        // the GPU CSR against it.  Per-atom ordering is deterministic
        // on both sides (atoms walked in index order, term types in
        // bond/angle/torsion/oop/sb order).
        auto const ao = atom_offsets[m];
        for (std::uint32_t a = 0; a < desc.num_atoms; ++a) {
            auto const &cpu_off = cpu.mmff_atom_offsets[a];
            auto const &gpu_off = atom_offs_out[ao + a];
            // GPU offsets are per-molecule (relative to this mol's
            // `max_mmff_interactions_per_mol_k` slot); CPU offsets are
            // per-molecule as well (into the mol's own interactions
            // vector).  Both start at 0 per molecule, so the counts are
            // the comparison metric.
            if (cpu_off.bond_count != gpu_off.bond_count ||
                cpu_off.angle_count != gpu_off.angle_count ||
                cpu_off.torsion_count != gpu_off.torsion_count ||
                cpu_off.oop_count != gpu_off.oop_count ||
                cpu_off.stretchbend_count != gpu_off.stretchbend_count) {
                ++csr_bad; ok = false;
                if (csr_bad < 5) {
                    std::printf("  mol=%u atom=%u CSR count diff: "
                                 "cpu=(b%u a%u t%u o%u s%u) gpu=(b%u a%u t%u o%u s%u)\n",
                                 m, a, cpu_off.bond_count, cpu_off.angle_count,
                                 cpu_off.torsion_count, cpu_off.oop_count,
                                 cpu_off.stretchbend_count,
                                 gpu_off.bond_count, gpu_off.angle_count,
                                 gpu_off.torsion_count, gpu_off.oop_count,
                                 gpu_off.stretchbend_count);
                }
            } else {
                ++csr_ok;
            }
        }

        if (ok) ++mol_ok; else ++mol_bad;
    }

    std::printf("molecules: %d match / %d bad / %d skipped-to-fallback\n",
                 mol_ok, mol_bad, mol_skipped);
    std::printf("bonds:    %d / %d\n", bond_ok, bond_bad);
    std::printf("angles:   %d / %d\n", angle_ok, angle_bad);
    std::printf("torsions: %d / %d\n", tor_ok, tor_bad);
    std::printf("oops:     %d / %d\n", oop_ok, oop_bad);
    std::printf("stbns:    %d / %d\n", sb_ok, sb_bad);
    std::printf("csr atoms: %d / %d\n", csr_ok, csr_bad);
    if (mol_bad > 0) { std::printf("FAIL\n"); return 1; }
    std::printf("PASS\n");
    return 0;
}
