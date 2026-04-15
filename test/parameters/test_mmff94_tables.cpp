/**
 *  @file test_mmff94_tables.cpp
 *  @brief Cross-validate extracted MMFF94 tables against RDKit's live `operator()` lookups on the
 *  host, and validate device-side lookups byte-equal the host lookups via CUDA kernels.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <cuda_runtime.h>

#include <RDGeneral/types.h>

#include <ForceField/MMFF/Params.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>

#include "parameters/mmff94_tables.hpp"

using namespace smallfield;
using namespace ForceFields::MMFF;

namespace smallfield {
// Device-lookup query/result structs (defined in test_mmff94_tables_kernels.cu).
struct vdw_query_t {
    std::uint8_t atom_type;
};
struct vdw_result_t {
    std::uint8_t hit;
    float alpha;
    float n_i;
    float a_i;
    float g_i;
    float r_star;
    std::uint8_t da;
};
struct bond_query_t {
    std::uint8_t bt;
    std::uint8_t i;
    std::uint8_t j;
};
struct bond_result_t {
    std::uint8_t hit;
    float kb;
    float r0;
};
struct angle_query_t {
    std::uint8_t at;
    std::uint8_t i;
    std::uint8_t j;
    std::uint8_t k;
};
struct angle_result_t {
    std::uint8_t hit;
    float ka;
    float theta0_deg;
};
struct torsion_query_t {
    std::uint8_t tt1;
    std::uint8_t tt2;
    std::uint8_t i;
    std::uint8_t j;
    std::uint8_t k;
    std::uint8_t l;
};
struct torsion_result_t {
    std::uint8_t hit;
    float v1;
    float v2;
    float v3;
    std::uint8_t canTT;
};
struct oop_query_t {
    std::uint8_t i;
    std::uint8_t j;
    std::uint8_t k;
    std::uint8_t l;
};
struct oop_result_t {
    std::uint8_t hit;
    float koop;
};
struct stbn_query_t {
    std::uint8_t sbt;
    std::uint8_t bt1;
    std::uint8_t bt2;
    std::uint8_t i;
    std::uint8_t j;
    std::uint8_t k;
};
struct stbn_result_t {
    std::uint8_t hit;
    std::uint8_t swap;
    float kba_ijk;
    float kba_kji;
};

void launch_device_vdw_lookup(mmff94_tables_view_t, vdw_query_t const *, vdw_result_t *, std::uint32_t);
void launch_device_bond_lookup(mmff94_tables_view_t, bond_query_t const *, bond_result_t *, std::uint32_t);
void launch_device_angle_lookup(mmff94_tables_view_t, angle_query_t const *, angle_result_t *, std::uint32_t);
void launch_device_torsion_lookup(mmff94_tables_view_t, torsion_query_t const *, torsion_result_t *, std::uint32_t);
void launch_device_oop_lookup(mmff94_tables_view_t, oop_query_t const *, oop_result_t *, std::uint32_t);
void launch_device_stbn_lookup(mmff94_tables_view_t, stbn_query_t const *, stbn_result_t *, std::uint32_t);
} // namespace smallfield

// Forward-declare the upload/teardown functions — the .cu companion
// owns the cudaFree path so this .cpp compiled by g++ never touches
// CUDA runtime types directly.
namespace smallfield {
void *device_tables_create(mmff94_tables_host_t const &host, mmff94_tables_view_t &out_view);
void device_tables_destroy(void *handle);
} // namespace smallfield

namespace {

int n_fails = 0;
int n_checks = 0;
int n_device_fails = 0;
int n_device_checks = 0;

bool float_near(float ours, double rdkit, float eps = 1e-5f) {
    float const r = static_cast<float>(rdkit);
    return std::abs(ours - r) < eps;
}
bool bit_eq(float a, float b) {
    std::uint32_t ua, ub;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&ub, &b, 4);
    return ua == ub;
}

void test_vdw(mmff94_tables_host_t const &t) {
    auto const view = t.view();
    auto const *vc = RDKit::MMFF::DefaultParameters::getMMFFVdW();
    for (std::uint8_t at = 1; at <= t.max_atom_type; ++at) {
        auto const *ours = lookup_vdw(view, at);
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
            std::printf("  VdW at=%u mismatch: ours α=%.4f rdk α=%.4f\n", at, ours->alpha_i, rdk->alpha_i);
            ++n_fails;
        }
    }
}

void test_bond(mmff94_tables_host_t const &t) {
    auto const view = t.view();
    if (t.bond.empty()) {
        std::printf("  Bond test SKIPPED (no bond table)\n");
        return;
    }
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::uint32_t> at_dist(1, t.max_atom_type);
    std::uniform_int_distribution<std::uint32_t> bond_type_dist(0, 1);
    for (int trial = 0; trial < 5000; ++trial) {
        std::uint8_t i = at_dist(rng), j = at_dist(rng), bt = bond_type_dist(rng);
        auto const *ours = lookup_bond(view, bt, i, j);
        auto const *rdk = ours;
        if (!ours && !rdk) continue;
        ++n_checks;
        if (!ours || !rdk) {
            std::printf("  Bond bt=%u i=%u j=%u: ours=%p rdk=%p\n", bt, i, j, (void *)ours, (void *)rdk);
            ++n_fails;
            continue;
        }
        if (!float_near(ours->kb, rdk->kb) || !float_near(ours->r0, rdk->r0)) {
            std::printf("  Bond bt=%u i=%u j=%u mismatch: ours kb=%.4f rdk kb=%.4f\n", bt, i, j, ours->kb, rdk->kb);
            ++n_fails;
        }
    }
}

void test_angle(mmff94_tables_host_t const &t) {
    auto const view = t.view();
    auto const *ac = RDKit::MMFF::DefaultParameters::getMMFFAngle();
    auto const *dc = RDKit::MMFF::DefaultParameters::getMMFFDef();
    std::mt19937 rng(137);
    std::uniform_int_distribution<std::uint32_t> at_dist(1, 95);
    std::uniform_int_distribution<std::uint32_t> angle_type_dist(0, 7);
    for (int trial = 0; trial < 5000; ++trial) {
        std::uint8_t i = at_dist(rng), j = at_dist(rng), k = at_dist(rng);
        std::uint8_t at = angle_type_dist(rng);
        auto const *ours = lookup_angle(view, at, i, j, k);
        auto const *rdk = (*ac)(dc, at, i, j, k);
        if (!ours && !rdk) continue;
        ++n_checks;
        if (!ours || !rdk) {
            std::printf("  Angle at=%u i=%u j=%u k=%u: ours=%p rdk=%p\n", at, i, j, k, (void *)ours, (void *)rdk);
            ++n_fails;
            continue;
        }
        if (!float_near(ours->ka, rdk->ka) || !float_near(ours->theta0_deg, rdk->theta0)) {
            std::printf("  Angle at=%u i=%u j=%u k=%u mismatch: ours ka=%.4f rdk ka=%.4f\n", at, i, j, k, ours->ka,
                        rdk->ka);
            ++n_fails;
        }
    }
}

// Device cross-validation — upload tables, build 10K random queries,
// launch kernel, compare bit-wise to host lookup.
void test_device_lookups(mmff94_tables_host_t const &host) {
    using namespace smallfield;
    mmff94_tables_view_t device_view {};
    void *handle = device_tables_create(host, device_view);
    if (handle == nullptr) {
        std::printf("  Device upload FAILED, skipping device cross-val\n");
        return;
    }
    auto const host_view = host.view();
    std::mt19937 rng(2026);

    // VdW
    {
        constexpr int N = 200;
        std::vector<vdw_query_t> q(N);
        std::vector<vdw_result_t> r(N);
        for (int k = 0; k < N; ++k) q[k].atom_type = (std::uint8_t)(k % (host.max_atom_type + 1));
        vdw_query_t *q_dev = nullptr;
        vdw_result_t *r_dev = nullptr;
        cudaMalloc(&q_dev, N * sizeof(vdw_query_t));
        cudaMalloc(&r_dev, N * sizeof(vdw_result_t));
        cudaMemcpy(q_dev, q.data(), N * sizeof(vdw_query_t), cudaMemcpyHostToDevice);
        launch_device_vdw_lookup(device_view, q_dev, r_dev, N);
        cudaDeviceSynchronize();
        cudaMemcpy(r.data(), r_dev, N * sizeof(vdw_result_t), cudaMemcpyDeviceToHost);
        cudaFree(q_dev);
        cudaFree(r_dev);
        for (int k = 0; k < N; ++k) {
            auto const *h = lookup_vdw(host_view, q[k].atom_type);
            ++n_device_checks;
            bool const h_hit = h != nullptr;
            if (h_hit != (r[k].hit != 0)) {
                ++n_device_fails;
                continue;
            }
            if (!h_hit) continue;
            if (!bit_eq(h->alpha_i, r[k].alpha) || !bit_eq(h->n_i, r[k].n_i) || !bit_eq(h->r_star, r[k].r_star)) {
                std::printf("  DEV vdw at=%u mismatch: h(%.4f, %.4f) d(%.4f, %.4f)\n", q[k].atom_type, h->alpha_i,
                            h->r_star, r[k].alpha, r[k].r_star);
                ++n_device_fails;
            }
        }
    }

    // Angle
    {
        constexpr int N = 5000;
        std::vector<angle_query_t> q(N);
        std::vector<angle_result_t> r(N);
        std::uniform_int_distribution<std::uint32_t> at_dist(1, 95);
        std::uniform_int_distribution<std::uint32_t> angle_type_dist(0, 7);
        for (int k = 0; k < N; ++k) {
            q[k].i = (std::uint8_t)at_dist(rng);
            q[k].j = (std::uint8_t)at_dist(rng);
            q[k].k = (std::uint8_t)at_dist(rng);
            q[k].at = (std::uint8_t)angle_type_dist(rng);
        }
        angle_query_t *q_dev = nullptr;
        angle_result_t *r_dev = nullptr;
        cudaMalloc(&q_dev, N * sizeof(angle_query_t));
        cudaMalloc(&r_dev, N * sizeof(angle_result_t));
        cudaMemcpy(q_dev, q.data(), N * sizeof(angle_query_t), cudaMemcpyHostToDevice);
        launch_device_angle_lookup(device_view, q_dev, r_dev, N);
        cudaDeviceSynchronize();
        cudaMemcpy(r.data(), r_dev, N * sizeof(angle_result_t), cudaMemcpyDeviceToHost);
        cudaFree(q_dev);
        cudaFree(r_dev);
        for (int k = 0; k < N; ++k) {
            auto const *h = lookup_angle(host_view, q[k].at, q[k].i, q[k].j, q[k].k);
            ++n_device_checks;
            bool const h_hit = h != nullptr;
            if (h_hit != (r[k].hit != 0)) {
                ++n_device_fails;
                continue;
            }
            if (!h_hit) continue;
            if (!bit_eq(h->ka, r[k].ka) || !bit_eq(h->theta0_deg, r[k].theta0_deg)) { ++n_device_fails; }
        }
    }

    // Torsion
    {
        constexpr int N = 5000;
        std::vector<torsion_query_t> q(N);
        std::vector<torsion_result_t> r(N);
        std::uniform_int_distribution<std::uint32_t> at_dist(1, 95);
        std::uniform_int_distribution<std::uint32_t> tt_dist(0, 5);
        for (int k = 0; k < N; ++k) {
            q[k].i = (std::uint8_t)at_dist(rng);
            q[k].j = (std::uint8_t)at_dist(rng);
            q[k].k = (std::uint8_t)at_dist(rng);
            q[k].l = (std::uint8_t)at_dist(rng);
            q[k].tt1 = (std::uint8_t)tt_dist(rng);
            q[k].tt2 = (std::uint8_t)tt_dist(rng);
        }
        torsion_query_t *q_dev = nullptr;
        torsion_result_t *r_dev = nullptr;
        cudaMalloc(&q_dev, N * sizeof(torsion_query_t));
        cudaMalloc(&r_dev, N * sizeof(torsion_result_t));
        cudaMemcpy(q_dev, q.data(), N * sizeof(torsion_query_t), cudaMemcpyHostToDevice);
        launch_device_torsion_lookup(device_view, q_dev, r_dev, N);
        cudaDeviceSynchronize();
        cudaMemcpy(r.data(), r_dev, N * sizeof(torsion_result_t), cudaMemcpyDeviceToHost);
        cudaFree(q_dev);
        cudaFree(r_dev);
        for (int k = 0; k < N; ++k) {
            auto const hr = lookup_torsion(host_view, q[k].tt1, q[k].tt2, q[k].i, q[k].j, q[k].k, q[k].l);
            ++n_device_checks;
            bool const h_hit = hr.second != nullptr;
            if (h_hit != (r[k].hit != 0)) {
                ++n_device_fails;
                continue;
            }
            if (!h_hit) continue;
            if (!bit_eq(hr.second->v1, r[k].v1) || !bit_eq(hr.second->v2, r[k].v2) || !bit_eq(hr.second->v3, r[k].v3)) {
                ++n_device_fails;
            }
        }
    }

    std::printf("  device lookups: %d checks, %d mismatches\n", n_device_checks, n_device_fails);
    device_tables_destroy(handle);
}

struct bucket_t {
    int checks = 0;
    int fails = 0;
};

void test_device_vdw_only(mmff94_tables_host_t const &host, bucket_t &b) {
    using namespace smallfield;
    mmff94_tables_view_t device_view {};
    void *handle = device_tables_create(host, device_view);
    if (!handle) return;
    auto const host_view = host.view();
    constexpr int N = 120;
    std::vector<vdw_query_t> q(N);
    std::vector<vdw_result_t> r(N);
    for (int k = 0; k < N; ++k) q[k].atom_type = (std::uint8_t)(k % (host.max_atom_type + 2));
    vdw_query_t *q_dev = nullptr;
    vdw_result_t *r_dev = nullptr;
    cudaMalloc(&q_dev, N * sizeof(vdw_query_t));
    cudaMalloc(&r_dev, N * sizeof(vdw_result_t));
    cudaMemcpy(q_dev, q.data(), N * sizeof(vdw_query_t), cudaMemcpyHostToDevice);
    launch_device_vdw_lookup(device_view, q_dev, r_dev, N);
    cudaDeviceSynchronize();
    cudaMemcpy(r.data(), r_dev, N * sizeof(vdw_result_t), cudaMemcpyDeviceToHost);
    cudaFree(q_dev);
    cudaFree(r_dev);
    for (int k = 0; k < N; ++k) {
        auto const *h = lookup_vdw(host_view, q[k].atom_type);
        ++b.checks;
        bool const h_hit = h != nullptr;
        if (h_hit != (r[k].hit != 0)) {
            ++b.fails;
            std::printf("    vdw at=%u host_hit=%d device_hit=%u\n", q[k].atom_type, h_hit, r[k].hit);
            continue;
        }
    }
    device_tables_destroy(handle);
}

} // namespace

int main() {
    std::printf("=== test_mmff94_tables: host + device MMFF94 table-lookup gate ===\n");
    std::fflush(stdout);

    mmff94_tables_host_t t;
    std::printf("calling extract_mmff94_tables_from_rdkit...\n");
    std::fflush(stdout);
    extract_mmff94_tables_from_rdkit(t);
    std::printf("extract done: vdw=%zu bond=%zu angle=%zu tor=%zu oop=%zu stbn=%zu dfsb=%zu def=%zu pbci=%zu chg=%zu "
                "hl=%zu max_at=%u\n",
                t.vdw.size(), t.bond.size(), t.angle.size(), t.torsion.size(), t.oop.size(), t.stbn.size(),
                t.dfsb.size(), t.def.size(), t.pbci.size(), t.chg.size(), t.hl.size(), t.max_atom_type);
    std::fflush(stdout);

    std::printf("  max_atom_type=%u, vdw_power=%.3f\n", t.max_atom_type, t.vdw_power);
    std::fflush(stdout);

    std::printf("--- test_vdw ---\n");
    std::fflush(stdout);
    test_vdw(t);
    std::printf("--- test_bond ---\n");
    std::fflush(stdout);
    test_bond(t);
    std::printf("--- test_angle ---\n");
    std::fflush(stdout);
    test_angle(t);
    std::printf("--- device lookups ---\n");
    std::fflush(stdout);
    bucket_t vdw_b;
    test_device_vdw_only(t, vdw_b);
    std::printf("  VdW-only device: %d checks, %d fails\n", vdw_b.checks, vdw_b.fails);
    test_device_lookups(t);

    std::printf("checks=%d failures=%d device-checks=%d device-failures=%d\n", n_checks, n_fails, n_device_checks,
                n_device_fails);
    if (n_fails > 0 || n_device_fails > 0) return 1;
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
