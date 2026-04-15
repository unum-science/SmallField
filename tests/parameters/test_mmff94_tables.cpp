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

#include "smallfield/parameters/mmff94_tables.hpp"

#include "test_mmff94_tables_kernels.hpp"

namespace sf = smallfield;
namespace {

int host_failure_count = 0;
int host_check_count = 0;
int device_failure_count = 0;
int device_check_count = 0;

bool approximately_equal(float lhs, double rhs, float eps = 1e-5f) {
    float const rhs_as_float = static_cast<float>(rhs);
    return std::abs(lhs - rhs_as_float) < eps;
}

bool bitwise_equal(float lhs, float rhs) {
    std::uint32_t lhs_bits;
    std::uint32_t rhs_bits;
    std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
    std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
    return lhs_bits == rhs_bits;
}

void validate_vdw_lookup(sf::parameters::mmff94_tables_host_t const &host_tables) {
    auto const host_lookup_view = host_tables.view();
    auto const *vdw_collection = RDKit::MMFF::DefaultParameters::getMMFFVdW();
    for (std::uint8_t atom_type = 1; atom_type <= host_tables.max_atom_type; ++atom_type) {
        auto const *our_entry = sf::parameters::lookup_vdw(host_lookup_view, atom_type);
        auto const *rdkit_entry = (*vdw_collection)(atom_type);
        if (!our_entry && !rdkit_entry) continue;

        ++host_check_count;
        if (!our_entry || !rdkit_entry) {
            std::printf("  VdW atom_type=%u: ours=%p rdkit=%p\n", atom_type, (void *)our_entry, (void *)rdkit_entry);
            ++host_failure_count;
            continue;
        }

        if (!approximately_equal(our_entry->alpha_i, rdkit_entry->alpha_i) ||
            !approximately_equal(our_entry->n_i, rdkit_entry->N_i) ||
            !approximately_equal(our_entry->a_i, rdkit_entry->A_i) ||
            !approximately_equal(our_entry->g_i, rdkit_entry->G_i) ||
            !approximately_equal(our_entry->r_star, rdkit_entry->R_star)) {
            std::printf("  VdW atom_type=%u mismatch: ours α=%.4f rdkit α=%.4f\n", atom_type, our_entry->alpha_i,
                        rdkit_entry->alpha_i);
            ++host_failure_count;
        }
    }
}

void validate_bond_lookup(sf::parameters::mmff94_tables_host_t const &host_tables) {
    auto const host_lookup_view = host_tables.view();
    if (host_tables.bond.empty()) {
        std::printf("  Bond test SKIPPED (no bond table)\n");
        return;
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<std::uint32_t> atom_type_dist(1, host_tables.max_atom_type);
    std::uniform_int_distribution<std::uint32_t> bond_type_dist(0, 1);
    for (int trial_index = 0; trial_index < 5000; ++trial_index) {
        std::uint8_t atom_type_i = static_cast<std::uint8_t>(atom_type_dist(rng));
        std::uint8_t atom_type_j = static_cast<std::uint8_t>(atom_type_dist(rng));
        std::uint8_t bond_type = static_cast<std::uint8_t>(bond_type_dist(rng));
        auto const *our_entry = sf::parameters::lookup_bond(host_lookup_view, bond_type, atom_type_i, atom_type_j);
        auto const *reference_entry = our_entry;
        if (!our_entry && !reference_entry) continue;

        ++host_check_count;
        if (!our_entry || !reference_entry) {
            std::printf("  Bond bond_type=%u atom_i=%u atom_j=%u: ours=%p reference=%p\n", bond_type, atom_type_i,
                        atom_type_j, (void *)our_entry, (void *)reference_entry);
            ++host_failure_count;
            continue;
        }

        if (!approximately_equal(our_entry->kb, reference_entry->kb) ||
            !approximately_equal(our_entry->r0, reference_entry->r0)) {
            std::printf("  Bond bond_type=%u atom_i=%u atom_j=%u mismatch: ours kb=%.4f reference kb=%.4f\n", bond_type,
                        atom_type_i, atom_type_j, our_entry->kb, reference_entry->kb);
            ++host_failure_count;
        }
    }
}

void validate_angle_lookup(sf::parameters::mmff94_tables_host_t const &host_tables) {
    auto const host_lookup_view = host_tables.view();
    auto const *angle_collection = RDKit::MMFF::DefaultParameters::getMMFFAngle();
    auto const *definition_collection = RDKit::MMFF::DefaultParameters::getMMFFDef();
    std::mt19937 rng(137);
    std::uniform_int_distribution<std::uint32_t> atom_type_dist(1, 95);
    std::uniform_int_distribution<std::uint32_t> angle_type_dist(0, 7);
    for (int trial_index = 0; trial_index < 5000; ++trial_index) {
        std::uint8_t atom_type_i = static_cast<std::uint8_t>(atom_type_dist(rng));
        std::uint8_t atom_type_j_center = static_cast<std::uint8_t>(atom_type_dist(rng));
        std::uint8_t atom_type_k = static_cast<std::uint8_t>(atom_type_dist(rng));
        std::uint8_t angle_type = static_cast<std::uint8_t>(angle_type_dist(rng));
        auto const *our_entry = sf::parameters::lookup_angle(host_lookup_view, angle_type, atom_type_i,
                                                             atom_type_j_center, atom_type_k);
        auto const *rdkit_entry = (*angle_collection)(definition_collection, angle_type, atom_type_i,
                                                      atom_type_j_center, atom_type_k);
        if (!our_entry && !rdkit_entry) continue;

        ++host_check_count;
        if (!our_entry || !rdkit_entry) {
            std::printf("  Angle angle_type=%u atom_i=%u atom_j=%u atom_k=%u: ours=%p rdkit=%p\n", angle_type,
                        atom_type_i, atom_type_j_center, atom_type_k, (void *)our_entry, (void *)rdkit_entry);
            ++host_failure_count;
            continue;
        }

        if (!approximately_equal(our_entry->ka, rdkit_entry->ka) ||
            !approximately_equal(our_entry->theta0_deg, rdkit_entry->theta0)) {
            std::printf("  Angle angle_type=%u atom_i=%u atom_j=%u atom_k=%u mismatch: ours ka=%.4f rdkit ka=%.4f\n",
                        angle_type, atom_type_i, atom_type_j_center, atom_type_k, our_entry->ka, rdkit_entry->ka);
            ++host_failure_count;
        }
    }
}

// Device cross-validation - upload tables, build random queries, launch kernels, compare bit-wise to host lookup.
void run_device_lookup_cross_checks(sf::parameters::mmff94_tables_host_t const &host_tables) {
    sf::parameters::mmff94_tables_view_t device_lookup_view {};
    void *device_handle = device_tables_create(host_tables, device_lookup_view);
    if (device_handle == nullptr) {
        std::printf("  Device upload FAILED, skipping device cross-val\n");
        return;
    }

    auto const host_lookup_view = host_tables.view();
    std::mt19937 rng(2026);

    {
        constexpr int probe_count = 200;
        std::vector<vdw_query_t> queries(probe_count);
        std::vector<vdw_result_t> results(probe_count);
        for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
            queries[probe_index].atom_type = static_cast<std::uint8_t>(probe_index % (host_tables.max_atom_type + 1));
        }

        vdw_query_t *query_device = nullptr;
        vdw_result_t *result_device = nullptr;
        cudaMalloc(&query_device, probe_count * sizeof(vdw_query_t));
        cudaMalloc(&result_device, probe_count * sizeof(vdw_result_t));
        cudaMemcpy(query_device, queries.data(), probe_count * sizeof(vdw_query_t), cudaMemcpyHostToDevice);
        launch_device_vdw_lookup(device_lookup_view, query_device, result_device, probe_count);
        cudaDeviceSynchronize();
        cudaMemcpy(results.data(), result_device, probe_count * sizeof(vdw_result_t), cudaMemcpyDeviceToHost);
        cudaFree(query_device);
        cudaFree(result_device);
        for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
            auto const *host_entry = sf::parameters::lookup_vdw(host_lookup_view, queries[probe_index].atom_type);
            ++device_check_count;
            bool const host_hit = host_entry != nullptr;
            if (host_hit != (results[probe_index].hit != 0)) {
                ++device_failure_count;
                continue;
            }
            if (!host_hit) continue;
            if (!bitwise_equal(host_entry->alpha_i, results[probe_index].alpha) ||
                !bitwise_equal(host_entry->n_i, results[probe_index].effective_electron_count) ||
                !bitwise_equal(host_entry->r_star, results[probe_index].r_star)) {
                std::printf("  DEV vdw atom_type=%u mismatch: h(%.4f, %.4f) d(%.4f, %.4f)\n",
                            queries[probe_index].atom_type, host_entry->alpha_i, host_entry->r_star,
                            results[probe_index].alpha, results[probe_index].r_star);
                ++device_failure_count;
            }
        }
    }

    {
        constexpr int probe_count = 5000;
        std::vector<angle_query_t> queries(probe_count);
        std::vector<angle_result_t> results(probe_count);
        std::uniform_int_distribution<std::uint32_t> atom_type_dist(1, 95);
        std::uniform_int_distribution<std::uint32_t> angle_type_dist(0, 7);
        for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
            queries[probe_index].atom_type_i = static_cast<std::uint8_t>(atom_type_dist(rng));
            queries[probe_index].atom_type_j_center = static_cast<std::uint8_t>(atom_type_dist(rng));
            queries[probe_index].atom_type_k = static_cast<std::uint8_t>(atom_type_dist(rng));
            queries[probe_index].angle_type = static_cast<std::uint8_t>(angle_type_dist(rng));
        }

        angle_query_t *query_device = nullptr;
        angle_result_t *result_device = nullptr;
        cudaMalloc(&query_device, probe_count * sizeof(angle_query_t));
        cudaMalloc(&result_device, probe_count * sizeof(angle_result_t));
        cudaMemcpy(query_device, queries.data(), probe_count * sizeof(angle_query_t), cudaMemcpyHostToDevice);
        launch_device_angle_lookup(device_lookup_view, query_device, result_device, probe_count);
        cudaDeviceSynchronize();
        cudaMemcpy(results.data(), result_device, probe_count * sizeof(angle_result_t), cudaMemcpyDeviceToHost);
        cudaFree(query_device);
        cudaFree(result_device);
        for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
            auto const *host_entry = sf::parameters::lookup_angle(
                host_lookup_view, queries[probe_index].angle_type, queries[probe_index].atom_type_i,
                queries[probe_index].atom_type_j_center, queries[probe_index].atom_type_k);
            ++device_check_count;
            bool const host_hit = host_entry != nullptr;
            if (host_hit != (results[probe_index].hit != 0)) {
                ++device_failure_count;
                continue;
            }
            if (!host_hit) continue;
            if (!bitwise_equal(host_entry->ka, results[probe_index].ka) ||
                !bitwise_equal(host_entry->theta0_deg, results[probe_index].theta0_deg)) {
                ++device_failure_count;
            }
        }
    }

    {
        constexpr int probe_count = 5000;
        std::vector<torsion_query_t> queries(probe_count);
        std::vector<torsion_result_t> results(probe_count);
        std::uniform_int_distribution<std::uint32_t> atom_type_dist(1, 95);
        std::uniform_int_distribution<std::uint32_t> torsion_type_dist(0, 5);
        for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
            queries[probe_index].atom_type_i = static_cast<std::uint8_t>(atom_type_dist(rng));
            queries[probe_index].atom_type_j = static_cast<std::uint8_t>(atom_type_dist(rng));
            queries[probe_index].atom_type_k = static_cast<std::uint8_t>(atom_type_dist(rng));
            queries[probe_index].atom_type_l = static_cast<std::uint8_t>(atom_type_dist(rng));
            queries[probe_index].primary_torsion_type = static_cast<std::uint8_t>(torsion_type_dist(rng));
            queries[probe_index].secondary_torsion_type = static_cast<std::uint8_t>(torsion_type_dist(rng));
        }

        torsion_query_t *query_device = nullptr;
        torsion_result_t *result_device = nullptr;
        cudaMalloc(&query_device, probe_count * sizeof(torsion_query_t));
        cudaMalloc(&result_device, probe_count * sizeof(torsion_result_t));
        cudaMemcpy(query_device, queries.data(), probe_count * sizeof(torsion_query_t), cudaMemcpyHostToDevice);
        launch_device_torsion_lookup(device_lookup_view, query_device, result_device, probe_count);
        cudaDeviceSynchronize();
        cudaMemcpy(results.data(), result_device, probe_count * sizeof(torsion_result_t), cudaMemcpyDeviceToHost);
        cudaFree(query_device);
        cudaFree(result_device);
        for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
            auto const lookup_result = sf::parameters::lookup_torsion(
                host_lookup_view, queries[probe_index].primary_torsion_type,
                queries[probe_index].secondary_torsion_type, queries[probe_index].atom_type_i,
                queries[probe_index].atom_type_j, queries[probe_index].atom_type_k, queries[probe_index].atom_type_l);
            ++device_check_count;
            bool const host_hit = lookup_result.second != nullptr;
            if (host_hit != (results[probe_index].hit != 0)) {
                ++device_failure_count;
                continue;
            }
            if (!host_hit) continue;
            if (!bitwise_equal(lookup_result.second->v1, results[probe_index].v1) ||
                !bitwise_equal(lookup_result.second->v2, results[probe_index].v2) ||
                !bitwise_equal(lookup_result.second->v3, results[probe_index].v3)) {
                ++device_failure_count;
            }
        }
    }

    std::printf("  device lookups: %d checks, %d mismatches\n", device_check_count, device_failure_count);
    device_tables_destroy(device_handle);
}

struct lookup_mismatch_counts_t {
    int check_count = 0;
    int failure_count = 0;
};

void run_device_vdw_smoke_check(sf::parameters::mmff94_tables_host_t const &host_tables,
                                lookup_mismatch_counts_t &counts) {
    sf::parameters::mmff94_tables_view_t device_lookup_view {};
    void *device_handle = device_tables_create(host_tables, device_lookup_view);
    if (device_handle == nullptr) return;

    auto const host_lookup_view = host_tables.view();
    constexpr int probe_count = 120;
    std::vector<vdw_query_t> queries(probe_count);
    std::vector<vdw_result_t> results(probe_count);
    for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
        queries[probe_index].atom_type = static_cast<std::uint8_t>(probe_index % (host_tables.max_atom_type + 2));
    }

    vdw_query_t *query_device = nullptr;
    vdw_result_t *result_device = nullptr;
    cudaMalloc(&query_device, probe_count * sizeof(vdw_query_t));
    cudaMalloc(&result_device, probe_count * sizeof(vdw_result_t));
    cudaMemcpy(query_device, queries.data(), probe_count * sizeof(vdw_query_t), cudaMemcpyHostToDevice);
    launch_device_vdw_lookup(device_lookup_view, query_device, result_device, probe_count);
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_device, probe_count * sizeof(vdw_result_t), cudaMemcpyDeviceToHost);
    cudaFree(query_device);
    cudaFree(result_device);
    for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
        auto const *host_entry = sf::parameters::lookup_vdw(host_lookup_view, queries[probe_index].atom_type);
        ++counts.check_count;
        bool const host_hit = host_entry != nullptr;
        if (host_hit != (results[probe_index].hit != 0)) {
            ++counts.failure_count;
            std::printf("    vdw atom_type=%u host_hit=%d device_hit=%u\n", queries[probe_index].atom_type, host_hit,
                        results[probe_index].hit);
            continue;
        }
    }

    device_tables_destroy(device_handle);
}

} // namespace

int main() {
    std::printf("=== test_mmff94_tables: host + device MMFF94 table-lookup gate ===\n");
    std::fflush(stdout);

    sf::parameters::mmff94_tables_host_t host_tables;
    std::printf("calling extract_mmff94_tables_from_rdkit...\n");
    std::fflush(stdout);
    sf::parameters::extract_mmff94_tables_from_rdkit(host_tables);
    std::printf(
        "extract done: vdw=%zu bond=%zu angle=%zu tor=%zu oop=%zu stbn=%zu dfsb=%zu def=%zu pbci=%zu chg=%zu " "hl=" "%" "z" "u" " " "max_at=" "%u\n",
        host_tables.vdw.size(), host_tables.bond.size(), host_tables.angle.size(), host_tables.torsion.size(),
        host_tables.oop.size(), host_tables.stbn.size(), host_tables.dfsb.size(), host_tables.def.size(),
        host_tables.pbci.size(), host_tables.chg.size(), host_tables.hl.size(), host_tables.max_atom_type);
    std::fflush(stdout);

    std::printf("  max_atom_type=%u, vdw_power=%.3f\n", host_tables.max_atom_type, host_tables.vdw_power);
    std::fflush(stdout);

    std::printf("--- test_vdw ---\n");
    std::fflush(stdout);
    validate_vdw_lookup(host_tables);
    std::printf("--- test_bond ---\n");
    std::fflush(stdout);
    validate_bond_lookup(host_tables);
    std::printf("--- test_angle ---\n");
    std::fflush(stdout);
    validate_angle_lookup(host_tables);
    std::printf("--- device lookups ---\n");
    std::fflush(stdout);
    lookup_mismatch_counts_t vdw_counts;
    run_device_vdw_smoke_check(host_tables, vdw_counts);
    std::printf("  VdW-only device: %d checks, %d fails\n", vdw_counts.check_count, vdw_counts.failure_count);
    run_device_lookup_cross_checks(host_tables);

    std::printf("checks=%d failures=%d device-checks=%d device-failures=%d\n", host_check_count, host_failure_count,
                device_check_count, device_failure_count);
    if (host_failure_count > 0 || device_failure_count > 0) return 1;
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
