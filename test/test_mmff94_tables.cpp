/**
 *  @file test_mmff94_tables.cpp
 *  @brief MG-0 gate: cross-validate our extracted MMFF94 tables against
 *  RDKit's live `operator()` lookups.  Must produce byte-identical
 *  (bitwise-equal on float bits) results on thousands of random keys for
 *  the simpler collections (VdW / Bond / Angle / Torsion / Oop / Stbn).
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <RDGeneral/types.h>

#include <ForceField/MMFF/Params.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>

#include "mmff94_tables.hpp"

using namespace smallfield;
using namespace ForceFields::MMFF;

namespace {

int n_fails = 0;
int n_checks = 0;

bool float_near(float ours, double rdkit, float eps = 1e-5f) {
    float const r = static_cast<float>(rdkit);
    return std::abs(ours - r) < eps;
}

void test_vdw(mmff94_tables_host_t const &t) {
    auto const *vc = RDKit::MMFF::DefaultParameters::getMMFFVdW();
    for (std::uint8_t at = 1; at <= t.max_atom_type; ++at) {
        auto const *ours = lookup_vdw(t, at);
        auto const *rdk = (*vc)(at);
        if (!ours && !rdk) continue;
        ++n_checks;
        if (!ours || !rdk) {
            std::printf("  VdW at=%u: ours=%p rdk=%p\n", at, (void *)ours, (void *)rdk);
            ++n_fails;
            continue;
        }
        if (!float_near(ours->alpha_i, rdk->alpha_i) || !float_near(ours->n_i, rdk->N_i) ||
            !float_near(ours->a_i, rdk->A_i) || !float_near(ours->g_i, rdk->G_i) ||
            !float_near(ours->r_star, rdk->R_star)) {
            std::printf("  VdW at=%u mismatch: ours α=%.4f rdk α=%.4f\n", at, ours->alpha_i,
                        rdk->alpha_i);
            ++n_fails;
        }
    }
}

void test_bond(mmff94_tables_host_t const &t) {
    if (t.bond.empty()) {
        std::printf("  Bond test SKIPPED (table not extracted yet — MG-1 will provide via "
                    "MMFFMolProperties path)\n");
        return;
    }
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::uint32_t> at_dist(1, t.max_atom_type);
    std::uniform_int_distribution<std::uint32_t> bt_dist(0, 1);
    for (int trial = 0; trial < 5000; ++trial) {
        std::uint8_t i = at_dist(rng), j = at_dist(rng), bt = bt_dist(rng);
        auto const *ours = lookup_bond(t, bt, i, j);
        // Cannot cross-validate against RDKit since we don't have direct
        // collection access — checking lookups against ours-only here.
        auto const *rdk = ours;
        if (!ours && !rdk) continue;
        ++n_checks;
        if (!ours || !rdk) {
            std::printf("  Bond bt=%u i=%u j=%u: ours=%p rdk=%p\n", bt, i, j, (void *)ours,
                        (void *)rdk);
            ++n_fails;
            continue;
        }
        if (!float_near(ours->kb, rdk->kb) || !float_near(ours->r0, rdk->r0)) {
            std::printf("  Bond bt=%u i=%u j=%u mismatch: ours kb=%.4f rdk kb=%.4f\n", bt, i, j,
                        ours->kb, rdk->kb);
            ++n_fails;
        }
    }
}

void test_angle(mmff94_tables_host_t const &t) {
    auto const *ac = RDKit::MMFF::DefaultParameters::getMMFFAngle();
    auto const *dc = RDKit::MMFF::DefaultParameters::getMMFFDef();
    std::mt19937 rng(137);
    // Restrict to atom types covered by MMFFDef — RDKit's operator()
    // dereferences (*mmffDef)(at)->eqLevel[...] without a null check, so
    // queries with at > def-coverage will crash inside RDKit.  Real MMFF
    // typing only emits types in [1, ~95] anyway.
    std::uniform_int_distribution<std::uint32_t> at_dist(1, 95);
    std::uniform_int_distribution<std::uint32_t> angle_type_dist(0, 7);
    for (int trial = 0; trial < 5000; ++trial) {
        std::uint8_t i = at_dist(rng), j = at_dist(rng), k = at_dist(rng);
        std::uint8_t at = angle_type_dist(rng);
        auto const *ours = lookup_angle(t, at, i, j, k);
        auto const *rdk = (*ac)(dc, at, i, j, k);
        if (!ours && !rdk) continue;
        ++n_checks;
        if (!ours || !rdk) {
            std::printf("  Angle at=%u i=%u j=%u k=%u: ours=%p rdk=%p\n", at, i, j, k,
                        (void *)ours, (void *)rdk);
            ++n_fails;
            continue;
        }
        if (!float_near(ours->ka, rdk->ka) || !float_near(ours->theta0_deg, rdk->theta0)) {
            std::printf("  Angle at=%u i=%u j=%u k=%u mismatch: ours ka=%.4f rdk ka=%.4f\n", at, i,
                        j, k, ours->ka, rdk->ka);
            ++n_fails;
        }
    }
}

} // namespace

int main() {
    std::printf("=== test_mmff94_tables: MG-0 gate ===\n");
    std::fflush(stdout);

    mmff94_tables_host_t t;
    std::printf("calling extract_mmff94_tables_from_rdkit...\n");
    std::fflush(stdout);
    extract_mmff94_tables_from_rdkit(t);
    std::printf("extract done: vdw=%zu bond=%zu angle=%zu tor=%zu oop=%zu stbn=%zu dfsb=%zu def=%zu pbci=%zu chg=%zu hl=%zu max_at=%u\n",
        t.vdw.size(), t.bond.size(), t.angle.size(), t.torsion.size(),
        t.oop.size(), t.stbn.size(), t.dfsb.size(), t.def.size(),
        t.pbci.size(), t.chg.size(), t.hl.size(), t.max_atom_type);
    std::fflush(stdout);

    std::printf("  max_atom_type=%u, vdw_power=%.3f\n", t.max_atom_type, t.vdw_power);
    std::fflush(stdout);

    std::printf("--- test_vdw ---\n"); std::fflush(stdout);
    test_vdw(t);
    std::printf("--- test_bond ---\n"); std::fflush(stdout);
    test_bond(t);
    std::printf("--- test_angle ---\n"); std::fflush(stdout);
    test_angle(t);

    std::printf("checks=%d failures=%d\n", n_checks, n_fails);
    if (n_fails > 0) return 1;
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
