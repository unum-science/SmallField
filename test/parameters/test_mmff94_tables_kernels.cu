/**
 *  @file test_mmff94_tables_kernels.cu
 *  @brief Device-side validation kernels for test_mmff94_tables.
 *
 *  Launches one device lookup per probe key (vdw / bond / angle / torsion
 *  / oop / stbn) and writes the resolved parameters into a device output
 *  buffer.  The host then bit-equals the device result against the
 *  host-side lookup over the same inputs.
 */

#include <cstdint>

#include <cuda_runtime.h>

#include "test_mmff94_tables_kernels.hpp"

namespace sf = smallfield;

__global__ void vdw_lookup_kernel(sf::parameters::mmff94_tables_view_t view, vdw_query_t const *queries,
                                  vdw_result_t *results, std::uint32_t query_count) {
    std::uint32_t const probe_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (probe_index >= query_count) return;

    auto const *lookup_entry = sf::parameters::lookup_vdw(view, queries[probe_index].atom_type);
    vdw_result_t result {};
    if (lookup_entry) {
        result.hit = 1;
        result.alpha = lookup_entry->alpha_i;
        result.effective_electron_count = lookup_entry->n_i;
        result.polarizability_scale = lookup_entry->a_i;
        result.slater_kirkwood_factor = lookup_entry->g_i;
        result.r_star = lookup_entry->r_star;
        result.donor_acceptor = lookup_entry->donor_acceptor;
    }
    results[probe_index] = result;
}

__global__ void bond_lookup_kernel(sf::parameters::mmff94_tables_view_t view, bond_query_t const *queries,
                                   bond_result_t *results, std::uint32_t query_count) {
    std::uint32_t const probe_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (probe_index >= query_count) return;

    auto const *lookup_entry = sf::parameters::lookup_bond(
        view, queries[probe_index].bond_type, queries[probe_index].atom_type_i, queries[probe_index].atom_type_j);
    bond_result_t result {};
    if (lookup_entry) {
        result.hit = 1;
        result.kb = lookup_entry->kb;
        result.r0 = lookup_entry->r0;
    }
    results[probe_index] = result;
}

__global__ void angle_lookup_kernel(sf::parameters::mmff94_tables_view_t view, angle_query_t const *queries,
                                    angle_result_t *results, std::uint32_t query_count) {
    std::uint32_t const probe_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (probe_index >= query_count) return;

    auto const *lookup_entry = sf::parameters::lookup_angle(
        view, queries[probe_index].angle_type, queries[probe_index].atom_type_i,
        queries[probe_index].atom_type_j_center, queries[probe_index].atom_type_k);
    angle_result_t result {};
    if (lookup_entry) {
        result.hit = 1;
        result.ka = lookup_entry->ka;
        result.theta0_deg = lookup_entry->theta0_deg;
    }
    results[probe_index] = result;
}

__global__ void torsion_lookup_kernel(sf::parameters::mmff94_tables_view_t view, torsion_query_t const *queries,
                                      torsion_result_t *results, std::uint32_t query_count) {
    std::uint32_t const probe_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (probe_index >= query_count) return;

    auto const lookup_result = sf::parameters::lookup_torsion(
        view, queries[probe_index].primary_torsion_type, queries[probe_index].secondary_torsion_type,
        queries[probe_index].atom_type_i, queries[probe_index].atom_type_j, queries[probe_index].atom_type_k,
        queries[probe_index].atom_type_l);
    torsion_result_t result {};
    if (lookup_result.second) {
        result.hit = 1;
        result.v1 = lookup_result.second->v1;
        result.v2 = lookup_result.second->v2;
        result.v3 = lookup_result.second->v3;
        result.canonical_torsion_type = static_cast<std::uint8_t>(lookup_result.first);
    }
    results[probe_index] = result;
}

__global__ void oop_lookup_kernel(sf::parameters::mmff94_tables_view_t view, oop_query_t const *queries,
                                  oop_result_t *results, std::uint32_t query_count) {
    std::uint32_t const probe_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (probe_index >= query_count) return;

    auto const *lookup_entry = sf::parameters::lookup_oop(
        view, queries[probe_index].atom_type_i, queries[probe_index].atom_type_j_center,
        queries[probe_index].atom_type_k, queries[probe_index].atom_type_l);
    oop_result_t result {};
    if (lookup_entry) {
        result.hit = 1;
        result.koop = lookup_entry->koop;
    }
    results[probe_index] = result;
}

__global__ void stbn_lookup_kernel(sf::parameters::mmff94_tables_view_t view, stbn_query_t const *queries,
                                   stbn_result_t *results, std::uint32_t query_count) {
    std::uint32_t const probe_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (probe_index >= query_count) return;

    auto const lookup_result = sf::parameters::lookup_stbn(
        view, queries[probe_index].stretch_bend_type, queries[probe_index].bond_type_ij,
        queries[probe_index].bond_type_kj, queries[probe_index].atom_type_i, queries[probe_index].atom_type_j_center,
        queries[probe_index].atom_type_k);
    stbn_result_t result {};
    if (lookup_result.second) {
        result.hit = 1;
        result.canonical_order_swapped = lookup_result.first ? 1 : 0;
        result.kba_ijk = lookup_result.second->kba_ijk;
        result.kba_kji = lookup_result.second->kba_kji;
    }
    results[probe_index] = result;
}

// Host-callable launchers.

void launch_device_vdw_lookup(sf::parameters::mmff94_tables_view_t view, vdw_query_t const *queries,
                              vdw_result_t *results, std::uint32_t query_count) {
    vdw_lookup_kernel<<<(query_count + 127) / 128, 128>>>(view, queries, results, query_count);
}

void launch_device_bond_lookup(sf::parameters::mmff94_tables_view_t view, bond_query_t const *queries,
                               bond_result_t *results, std::uint32_t query_count) {
    bond_lookup_kernel<<<(query_count + 127) / 128, 128>>>(view, queries, results, query_count);
}

void launch_device_angle_lookup(sf::parameters::mmff94_tables_view_t view, angle_query_t const *queries,
                                angle_result_t *results, std::uint32_t query_count) {
    angle_lookup_kernel<<<(query_count + 127) / 128, 128>>>(view, queries, results, query_count);
}

void launch_device_torsion_lookup(sf::parameters::mmff94_tables_view_t view, torsion_query_t const *queries,
                                  torsion_result_t *results, std::uint32_t query_count) {
    torsion_lookup_kernel<<<(query_count + 127) / 128, 128>>>(view, queries, results, query_count);
}

void launch_device_oop_lookup(sf::parameters::mmff94_tables_view_t view, oop_query_t const *queries,
                              oop_result_t *results, std::uint32_t query_count) {
    oop_lookup_kernel<<<(query_count + 127) / 128, 128>>>(view, queries, results, query_count);
}

void launch_device_stbn_lookup(sf::parameters::mmff94_tables_view_t view, stbn_query_t const *queries,
                               stbn_result_t *results, std::uint32_t query_count) {
    stbn_lookup_kernel<<<(query_count + 127) / 128, 128>>>(view, queries, results, query_count);
}

void *device_tables_create(sf::parameters::mmff94_tables_host_t const &host_tables,
                           sf::parameters::mmff94_tables_view_t &out_view) {
    auto *device_tables = new sf::parameters::mmff94_tables_device_t;
    sf::parameters::upload_mmff94_tables_to_device(host_tables, *device_tables);
    if (device_tables->arena.data() == nullptr) {
        delete device_tables;
        return nullptr;
    }
    out_view = device_tables->view;
    return device_tables;
}

void device_tables_destroy(void *handle) { delete static_cast<sf::parameters::mmff94_tables_device_t *>(handle); }
