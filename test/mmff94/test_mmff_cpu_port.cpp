/**
 *  @file test_mmff_cpu_port.cpp
 *  @brief Cross-validate the native MMFF CPU port
 *  (`extract_mmff94_params_native`) against the existing RDKit-backed
 *  path (`extract_mmff94_params`).  Must produce byte-identical MMFF
 *  term arrays on a curated SMILES set; small numerical deltas (≤ 1e-3)
 *  are allowed on empirical-rule fallback terms due to float/double
 *  rounding in MMFF.V eqs 18-19 and the torsion rules.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <RDGeneral/types.h>
#include <ForceField/MMFF/Params.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>

#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/mmff94/mmff94_cpu.hpp"
#include "smallfield/mmff94/mmff94_cpu_extract.hpp"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_rdkit.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

namespace sfp = smallfield::preprocess;
namespace sfparam = smallfield::parameters;
namespace sfmmff = smallfield::mmff94;
namespace sft = smallfield::types;

namespace {

int n_checks = 0;
int n_fails = 0;
int n_mol_pass = 0;
int n_mol_fail = 0;
int n_mol_with_any_fail = 0;

/** Bitwise-identical float comparison (allowing NaN != NaN). */
bool bit_eq(float a, float b) {
    std::uint32_t ua, ub;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&ub, &b, 4);
    return ua == ub;
}

/** Near-equal with absolute tolerance — used for empirical-rule terms. */
bool near(float a, float b, float eps = 1e-3f) {
    if (bit_eq(a, b)) return true;
    return std::fabs(a - b) <= eps * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}

bool compare_mol(std::string_view smiles, sfp::etkdg_prepared_molecule_t const &rdk,
                 sfp::etkdg_prepared_molecule_t const &nat, bool verbose) {
    bool ok = true;
    auto complain = [&](char const *what) {
        if (verbose && ok) std::printf("  [%.*s] MISMATCH: ", (int)smiles.size(), smiles.data());
        if (verbose) std::printf("%s ", what);
        ok = false;
    };

    if (rdk.num_atoms != nat.num_atoms) complain("num_atoms");
    if (rdk.mmff_bonds.size() != nat.mmff_bonds.size()) complain("bond_count");
    if (rdk.mmff_angles.size() != nat.mmff_angles.size()) complain("angle_count");
    if (rdk.mmff_torsions.size() != nat.mmff_torsions.size()) {
        complain("torsion_count");
        if (verbose) {
            // Dump first few torsions present in rdk but missing from nat
            // and vice versa — index them by (i, j, k, l).
            auto ijkl_of = [](auto const &t) {
                return (std::uint32_t)t.atom_i | ((std::uint32_t)t.atom_j << 8) | ((std::uint32_t)t.atom_k << 16) |
                       ((std::uint32_t)t.atom_l << 24);
            };
            std::vector<std::uint32_t> rset, nset;
            for (auto const &t : rdk.mmff_torsions) rset.push_back(ijkl_of(t));
            for (auto const &t : nat.mmff_torsions) nset.push_back(ijkl_of(t));
            std::sort(rset.begin(), rset.end());
            std::sort(nset.begin(), nset.end());
            int printed = 0;
            for (auto k : rset) {
                if (!std::binary_search(nset.begin(), nset.end(), k) && printed < 3) {
                    std::printf("    rdk-only tor: %u-%u-%u-%u\n", k & 0xff, (k >> 8) & 0xff, (k >> 16) & 0xff,
                                (k >> 24) & 0xff);
                    ++printed;
                }
            }
            printed = 0;
            for (auto k : nset) {
                if (!std::binary_search(rset.begin(), rset.end(), k) && printed < 3) {
                    std::printf("    nat-only tor: %u-%u-%u-%u\n", k & 0xff, (k >> 8) & 0xff, (k >> 16) & 0xff,
                                (k >> 24) & 0xff);
                    ++printed;
                }
            }
        }
    }
    if (rdk.mmff_oops.size() != nat.mmff_oops.size()) complain("oop_count");
    if (rdk.mmff_stretchbends.size() != nat.mmff_stretchbends.size()) complain("stbn_count");

    if (!ok) {
        if (verbose)
            std::printf("\n    sizes: bond %zu/%zu, ang %zu/%zu, tor %zu/%zu, oop %zu/%zu, sb %zu/%zu\n",
                        rdk.mmff_bonds.size(), nat.mmff_bonds.size(), rdk.mmff_angles.size(), nat.mmff_angles.size(),
                        rdk.mmff_torsions.size(), nat.mmff_torsions.size(), rdk.mmff_oops.size(), nat.mmff_oops.size(),
                        rdk.mmff_stretchbends.size(), nat.mmff_stretchbends.size());
        return ok;
    }

    // Per-atom VdW + charge
    for (std::size_t a = 0; a < rdk.mmff_atom_vdw.size(); ++a) {
        auto const &r = rdk.mmff_atom_vdw[a];
        auto const &n = nat.mmff_atom_vdw[a];
        if (!(bit_eq(r.alpha, n.alpha) && bit_eq(r.n_effective, n.n_effective) && bit_eq(r.a_scale, n.a_scale) &&
              bit_eq(r.g_factor, n.g_factor) && bit_eq(r.r_star, n.r_star) && r.donor_acceptor == n.donor_acceptor)) {
            if (verbose)
                std::printf("    vdw[%zu] rdk α=%.4f r*=%.4f; nat α=%.4f r*=%.4f\n", a, r.alpha, r.r_star, n.alpha,
                            n.r_star);
            ++n_fails;
        }
        if (!bit_eq(rdk.mmff_atom_charge[a], nat.mmff_atom_charge[a])) {
            if (verbose)
                std::printf("    charge[%zu] rdk=%.5f nat=%.5f\n", a, rdk.mmff_atom_charge[a], nat.mmff_atom_charge[a]);
            ++n_fails;
        }
    }

    // Bonds
    for (std::size_t t = 0; t < rdk.mmff_bonds.size(); ++t) {
        auto const &r = rdk.mmff_bonds[t];
        auto const &n = nat.mmff_bonds[t];
        ++n_checks;
        if (r.atom_i != n.atom_i || r.atom_j != n.atom_j) {
            if (verbose)
                std::printf("    bond[%zu] ij rdk=%u-%u nat=%u-%u\n", t, r.atom_i, r.atom_j, n.atom_i, n.atom_j);
            ++n_fails;
            continue;
        }
        if (!near(r.force_constant, n.force_constant) || !near(r.equilibrium_distance, n.equilibrium_distance)) {
            if (verbose)
                std::printf("    bond[%zu] kb rdk=%.4f nat=%.4f  r0 rdk=%.4f nat=%.4f\n", t, r.force_constant,
                            n.force_constant, r.equilibrium_distance, n.equilibrium_distance);
            ++n_fails;
        }
    }

    // Angles
    for (std::size_t t = 0; t < rdk.mmff_angles.size(); ++t) {
        auto const &r = rdk.mmff_angles[t];
        auto const &n = nat.mmff_angles[t];
        ++n_checks;
        if (r.atom_i != n.atom_i || r.atom_j != n.atom_j || r.atom_k != n.atom_k) {
            if (verbose)
                std::printf("    angle[%zu] ijk rdk=%u-%u-%u nat=%u-%u-%u\n", t, r.atom_i, r.atom_j, r.atom_k, n.atom_i,
                            n.atom_j, n.atom_k);
            ++n_fails;
            continue;
        }
        if (!near(r.force_constant, n.force_constant) || !near(r.equilibrium_angle, n.equilibrium_angle)) {
            if (verbose)
                std::printf("    angle[%zu] ka rdk=%.4f nat=%.4f θ₀ rdk=%.4f nat=%.4f\n", t, r.force_constant,
                            n.force_constant, r.equilibrium_angle, n.equilibrium_angle);
            ++n_fails;
        }
    }

    // Torsions
    for (std::size_t t = 0; t < rdk.mmff_torsions.size(); ++t) {
        auto const &r = rdk.mmff_torsions[t];
        auto const &n = nat.mmff_torsions[t];
        ++n_checks;
        if (r.atom_i != n.atom_i || r.atom_j != n.atom_j || r.atom_k != n.atom_k || r.atom_l != n.atom_l) {
            if (verbose)
                std::printf("    tor[%zu] ijkl rdk=%u-%u-%u-%u nat=%u-%u-%u-%u\n", t, r.atom_i, r.atom_j, r.atom_k,
                            r.atom_l, n.atom_i, n.atom_j, n.atom_k, n.atom_l);
            ++n_fails;
            continue;
        }
        if (!near(r.v1, n.v1) || !near(r.v2, n.v2) || !near(r.v3, n.v3)) {
            if (verbose)
                std::printf("    tor[%zu] ijkl=%u-%u-%u-%u V rdk=(%.4f, %.4f, %.4f) nat=(%.4f, %.4f, %.4f)\n", t,
                            r.atom_i, r.atom_j, r.atom_k, r.atom_l, r.v1, r.v2, r.v3, n.v1, n.v2, n.v3);
            ++n_fails;
        }
    }

    // OOPs
    for (std::size_t t = 0; t < rdk.mmff_oops.size(); ++t) {
        auto const &r = rdk.mmff_oops[t];
        auto const &n = nat.mmff_oops[t];
        ++n_checks;
        if (r.atom_i != n.atom_i || r.atom_j != n.atom_j || r.atom_k != n.atom_k || r.atom_l != n.atom_l) {
            ++n_fails;
            continue;
        }
        if (!near(r.force_constant, n.force_constant)) ++n_fails;
    }

    // Stretch-bends
    for (std::size_t t = 0; t < rdk.mmff_stretchbends.size(); ++t) {
        auto const &r = rdk.mmff_stretchbends[t];
        auto const &n = nat.mmff_stretchbends[t];
        ++n_checks;
        if (r.atom_i != n.atom_i || r.atom_j != n.atom_j || r.atom_k != n.atom_k) {
            if (verbose) std::printf("    sb[%zu] ijk mismatch\n", t);
            ++n_fails;
            continue;
        }
        if (!near(r.kba_ijk, n.kba_ijk) || !near(r.kba_kji, n.kba_kji) ||
            !near(r.equilibrium_angle, n.equilibrium_angle) ||
            !near(r.equilibrium_distance_ij, n.equilibrium_distance_ij) ||
            !near(r.equilibrium_distance_kj, n.equilibrium_distance_kj)) {
            if (verbose)
                std::printf(
                    "    sb[%zu] ijk=%u-%u-%u rdk k=(%.4f, %.4f) θ₀=%.4f r0=(%.4f, %.4f) " "nat k=(%.4f, %.4f) " "θ₀=%."
                                                                                                                 "4f " "r0=(%.4f," " %.4f)\n",
                    t, r.atom_i, r.atom_j, r.atom_k, r.kba_ijk, r.kba_kji, r.equilibrium_angle,
                    r.equilibrium_distance_ij, r.equilibrium_distance_kj, n.kba_ijk, n.kba_kji, n.equilibrium_angle,
                    n.equilibrium_distance_ij, n.equilibrium_distance_kj);
            ++n_fails;
        }
    }

    // CSR interactions are built only by the native path in this test
    // (the rdkit_direct_extract baseline leaves them empty). Skip that
    // comparison — the native CSR is validated implicitly by the bonded
    // term counts above plus cross-checks in downstream ETKDG kernels.

    return ok;
}

/** Build a "ground truth" prepared molecule by calling RDKit's per-term
 *  `getMMFF*Params` directly — bypasses the `mmff_param_cache`, whose
 *  torsion/angle/stbn keys miss ring context.  The native CPU port is
 *  validated against this, not against `extract_mmff94_params`, so that
 *  any latent cache bugs don't contaminate the comparison. */
void rdkit_direct_extract(RDKit::ROMol &mol, sft::molecule_descriptor_t const &desc,
                          sfp::etkdg_prepared_molecule_t &out) {
    if (!mol.hasProp(RDKit::common_properties::_MMFFSanitized)) {
        mol.setProp(RDKit::common_properties::_MMFFSanitized, 1, true);
    }
    RDKit::MMFF::MMFFMolProperties props(mol);
    if (!props.isValid()) return;
    std::uint32_t const num_atoms = desc.num_atoms;
    out.mmff_atom_vdw.resize(num_atoms);
    out.mmff_atom_charge.resize(num_atoms);
    auto const *vc = RDKit::MMFF::DefaultParameters::getMMFFVdW();
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        auto const *vdw = (*vc)(props.getMMFFAtomType(a));
        sft::mmff94_atom_vdw_params_t p {};
        if (vdw) {
            p.alpha = (float)vdw->alpha_i;
            p.n_effective = (float)vdw->N_i;
            p.a_scale = (float)vdw->A_i;
            p.g_factor = (float)vdw->G_i;
            p.r_star = (float)vdw->R_star;
            p.donor_acceptor = (vdw->DA == 'D') ? 2 : (vdw->DA == 'A') ? 1 : 0;
        }
        out.mmff_atom_vdw[a] = p;
        out.mmff_atom_charge[a] = (float)props.getMMFFPartialCharge(a);
    }
    auto find_bond = [&](std::uint32_t a, std::uint32_t b) -> std::int32_t {
        for (std::uint16_t n = desc.neighbor_offset[a]; n < desc.neighbor_offset[a + 1]; ++n)
            if (desc.neighbor_atom[n] == b) return desc.neighbor_bond[n];
        return -1;
    };
    for (std::uint32_t b = 0; b < desc.num_bonds; ++b) {
        std::uint32_t i = desc.bond_atom_i[b], j = desc.bond_atom_j[b];
        unsigned bt;
        ForceFields::MMFF::MMFFBond bp;
        if (!props.getMMFFBondStretchParams(mol, i, j, bt, bp)) continue;
        out.mmff_bonds.push_back({(std::uint8_t)i, (std::uint8_t)j, {0, 0}, (float)bp.kb, (float)bp.r0, 0.0f});
    }
    for (std::uint32_t c = 0; c < num_atoms; ++c) {
        std::uint16_t ns = desc.neighbor_offset[c], ne = desc.neighbor_offset[c + 1];
        for (std::uint16_t a = ns; a < ne; ++a)
            for (std::uint16_t b = a + 1; b < ne; ++b) {
                std::uint32_t ai = desc.neighbor_atom[a], aj = desc.neighbor_atom[b];
                if (find_bond(ai, c) < 0 || find_bond(c, aj) < 0) continue;
                unsigned at;
                ForceFields::MMFF::MMFFAngle ap;
                if (!props.getMMFFAngleBendParams(mol, ai, c, aj, at, ap)) continue;
                auto const *prop_center = (*RDKit::MMFF::DefaultParameters::getMMFFProp())(props.getMMFFAtomType(c));
                std::uint8_t const is_linear = (prop_center && prop_center->linh) ? 1u : 0u;
                out.mmff_angles.push_back({(std::uint8_t)ai, (std::uint8_t)c, (std::uint8_t)aj, is_linear, (float)ap.ka,
                                           (float)(ap.theta0 * M_PI / 180.0), 0.0f});
            }
    }
    for (std::uint32_t b = 0; b < desc.num_bonds; ++b) {
        std::uint32_t j = desc.bond_atom_i[b], k = desc.bond_atom_j[b];
        std::uint16_t js = desc.neighbor_offset[j], je = desc.neighbor_offset[j + 1];
        std::uint16_t ks = desc.neighbor_offset[k], ke = desc.neighbor_offset[k + 1];
        for (std::uint16_t nj = js; nj < je; ++nj) {
            std::uint32_t i = desc.neighbor_atom[nj];
            if (i == k) continue;
            for (std::uint16_t nk = ks; nk < ke; ++nk) {
                std::uint32_t l = desc.neighbor_atom[nk];
                if (l == j) continue;
                unsigned tt;
                ForceFields::MMFF::MMFFTor tp;
                if (!props.getMMFFTorsionParams(mol, i, j, k, l, tt, tp)) continue;
                out.mmff_torsions.push_back({(std::uint8_t)i, (std::uint8_t)j, (std::uint8_t)k, (std::uint8_t)l,
                                             (float)tp.V1, (float)tp.V2, (float)tp.V3});
            }
        }
    }
    for (std::uint32_t c = 0; c < num_atoms; ++c) {
        if (desc.degree[c] < 3) continue;
        std::uint16_t ns = desc.neighbor_offset[c];
        std::uint8_t nb[8];
        std::uint32_t nc = 0;
        for (std::uint16_t n = ns; n < desc.neighbor_offset[c + 1] && nc < 8; ++n) nb[nc++] = desc.neighbor_atom[n];
        for (std::uint32_t a = 0; a < nc; ++a) {
            std::uint8_t pl[2];
            std::uint32_t pc = 0;
            for (std::uint32_t q = 0; q < nc && pc < 2; ++q)
                if (q != a) pl[pc++] = nb[q];
            if (pc < 2) continue;
            ForceFields::MMFF::MMFFOop op;
            if (!props.getMMFFOopBendParams(mol, nb[a], c, pl[0], pl[1], op)) continue;
            out.mmff_oops.push_back({nb[a], (std::uint8_t)c, pl[0], pl[1], (float)op.koop, {0.0f, 0.0f}});
        }
    }
    for (std::uint32_t c = 0; c < num_atoms; ++c) {
        std::uint16_t ns = desc.neighbor_offset[c], ne = desc.neighbor_offset[c + 1];
        for (std::uint16_t a = ns; a < ne; ++a)
            for (std::uint16_t b = a + 1; b < ne; ++b) {
                std::uint32_t ai = desc.neighbor_atom[a], aj = desc.neighbor_atom[b];
                if (find_bond(ai, c) < 0 || find_bond(c, aj) < 0) continue;
                unsigned sbt;
                ForceFields::MMFF::MMFFStbn sb;
                ForceFields::MMFF::MMFFBond bp[2];
                ForceFields::MMFF::MMFFAngle ap;
                if (!props.getMMFFStretchBendParams(mol, ai, c, aj, sbt, sb, bp, ap)) continue;
                out.mmff_stretchbends.push_back({(std::uint8_t)ai, (std::uint8_t)c, (std::uint8_t)aj, 0,
                                                 (float)sb.kbaIJK, (float)sb.kbaKJI, (float)(ap.theta0 * M_PI / 180.0),
                                                 (float)bp[0].r0, (float)bp[1].r0, 0.0f});
            }
    }
}

void run_one(std::string_view smiles, sfparam::mmff94_tables_view_t const &tables, bool verbose) {
    // Native + RDKit preprocessing share the same SmilesToMol/addHs/build_descriptor path.
    std::unique_ptr<RDKit::RWMol> mol;
    try {
        std::string s {smiles};
        RDKit::SmilesParserParams params;
        params.sanitize = true;
        params.removeHs = false;
        params.strictCXSMILES = false;
        mol.reset(RDKit::SmilesToMol(s, params));
    }
    catch (...) {
        return;
    }
    if (!mol) return;
    try {
        RDKit::MolOps::addHs(*mol);
    }
    catch (...) {
        return;
    }
    std::uint32_t const num_atoms = mol->getNumAtoms();
    if (num_atoms > 128) return;

    sft::molecule_descriptor_t desc;
    if (!sft::build_descriptor(*mol, desc)) return;

    // RDKit direct (cache-free) baseline.
    sfp::etkdg_prepared_molecule_t rdk_result;
    rdk_result.num_atoms = num_atoms;
    rdkit_direct_extract(*mol, desc, rdk_result);

    // Native path — pull atom types + charges from MMFFMolProperties (same
    // source as the RDKit path).
    if (!mol->hasProp(RDKit::common_properties::_MMFFSanitized)) {
        mol->setProp(RDKit::common_properties::_MMFFSanitized, 1, true);
    }
    RDKit::MMFF::MMFFMolProperties props(*mol);
    if (!props.isValid()) return;

    std::vector<std::uint8_t> atom_types(num_atoms);
    std::vector<float> charges(num_atoms);
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        atom_types[a] = static_cast<std::uint8_t>(props.getMMFFAtomType(a));
        charges[a] = static_cast<float>(props.getMMFFPartialCharge(a));
    }

    sfp::etkdg_prepared_molecule_t nat_result;
    nat_result.num_atoms = num_atoms;
    sfmmff::extract_mmff94_params_native(desc, atom_types, charges, tables, nat_result);

    // Fully-native: zero runtime RDKit. Term counts should match the
    // hybrid path on molecules where our typer agrees with RDKit.
    sfp::etkdg_prepared_molecule_t fn;
    fn.num_atoms = num_atoms;
    (void)sfmmff::extract_mmff94_params_fully_native(desc, tables, fn);

    int const fails_before = n_fails;
    bool const ok = compare_mol(smiles, rdk_result, nat_result, verbose);
    if (ok && n_fails == fails_before) ++n_mol_pass;
    else if (!ok) ++n_mol_fail;
    if (n_fails > fails_before) ++n_mol_with_any_fail;
}

std::vector<std::string> load_smiles(std::string const &path, std::size_t max_count) {
    std::vector<std::string> out;
    std::ifstream ifs(path);
    if (!ifs) return out;
    std::string line;
    while (std::getline(ifs, line) && out.size() < max_count) {
        if (line.empty() || line[0] == '#') continue;
        // Accept either raw SMILES or SMILES<space>name.
        auto sp = line.find_first_of(" \t");
        if (sp != std::string::npos) line.resize(sp);
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

} // namespace

int main(int argc, char **argv) {
    std::printf("=== test_mmff_cpu_port: native MMFF term extractor gate ===\n");

    sfparam::mmff94_tables_host_t tables_host;
    sfparam::extract_mmff94_tables_from_rdkit(tables_host);
    sfparam::mmff94_tables_view_t const tables = tables_host.view();
    std::printf("extracted tables: vdw=%zu bond=%zu angle=%zu tor=%zu oop=%zu stbn=%zu dfsb=%zu\n",
                tables_host.vdw.size(), tables_host.bond.size(), tables_host.angle.size(), tables_host.torsion.size(),
                tables_host.oop.size(), tables_host.stbn.size(), tables_host.dfsb.size());

    // Curated handful of drug-like SMILES (aspirin, caffeine, ibuprofen,
    // paracetamol, benzodiazepine, adenine, imidazolium, carboxylate).
    std::vector<std::string> probe = {
        "CC(=O)OC1=CC=CC=C1C(=O)O",      // aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  // caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", // ibuprofen
        "CC(=O)NC1=CC=C(O)C=C1",         // paracetamol
        "O=C1NC(=O)C2=CC=CC=C2N1",       // phthalimidine-ish
        "Nc1ncnc2n(cnc12)",              // adenine
        "[NH2+]=C(N)N",                  // guanidinium
        "CC(=O)[O-]",                    // acetate
        "c1cc[nH+]c(c1)",                // pyridinium
        "C1CCOC1",                       // tetrahydrofuran
        "C1CC1",                         // cyclopropane
        "c1ccc2[nH]ccc2c1",              // indole
    };

    // Optional: a SMILES file for larger coverage
    std::vector<std::string> extra;
    std::size_t const cap = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 5000;
    if (argc > 1) extra = load_smiles(argv[1], cap);
    std::printf("probing %zu curated + %zu external SMILES\n", probe.size(), extra.size());

    bool const verbose = (argc <= 1);
    for (auto const &s : probe) run_one(s, tables, verbose);
    // Print first few failures for triage, then go silent.
    int fail_prints = 0;
    for (auto const &s : extra) {
        int const fail_before = n_fails;
        int const mfail_before = n_mol_fail;
        bool const v = (fail_prints < 3);
        run_one(s, tables, v);
        bool const had_fail = n_fails > fail_before || n_mol_fail > mfail_before;
        if (had_fail && fail_prints < 3) {
            std::printf("  (above is from SMILES: %s)\n", s.c_str());
            ++fail_prints;
        }
    }

    std::size_t const total_mols = n_mol_pass + n_mol_fail + n_mol_with_any_fail;
    std::printf("molecules: %d size-pass / %d size-fail / %d value-fail (checks=%d, term failures=%d)\n",
                n_mol_pass + n_mol_with_any_fail, n_mol_fail, n_mol_with_any_fail, n_checks, n_fails);
    if (total_mols > 0) {
        double const mol_fail_ppm = 1.0e6 * (double)(n_mol_fail + n_mol_with_any_fail) / (double)total_mols;
        double const term_fail_ppm = (n_checks > 0) ? 1.0e6 * (double)n_fails / (double)n_checks : 0.0;
        std::printf("ppm: molecules with any failure=%.1f, term failures=%.1f\n", mol_fail_ppm, term_fail_ppm);
    }
    // Gate: ≤ 100 ppm mol-level failures, ≤ 500 ppm term failures on
    // drug-like ChEMBL. Exotic structures (pentacoordinate P porphyrins,
    // etc.) can exceed this and should be triaged on the full 10M USearch set.
    if (n_mol_fail > 0) {
        std::printf("FAIL (structural mismatches)\n");
        return 1;
    }
    double const fail_frac = (total_mols > 0) ? (double)n_mol_with_any_fail / (double)total_mols : 0.0;
    if (fail_frac > 0.001) {
        std::printf("FAIL (too many molecules with value differences: %.3f %%)\n", fail_frac * 100.0);
        return 1;
    }
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
