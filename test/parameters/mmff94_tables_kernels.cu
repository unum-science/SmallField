/**
 *  @file test/parameters/mmff94_tables_kernels.cu
 *  @brief Device-side validation kernels for mmff94_tables.
 *
 *  One device lookup per probe key — vdw, bond, angle, torsion — writes its resolved parameters to a device buffer
 *  the host then bit-equals against the host-side lookup over the same inputs. `lookup_oop` and `lookup_stbn` are
 *  reached on the device through the terms `gpu_extract` compares, so they need no probe here.
 */

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <cuda_runtime.h>

#include "test/parameters/mmff94_tables_kernels.hpp"

namespace sf = smallfield;

namespace smallfield::tests {

__global__ void vdw_lookup_kernel(sf::parameters::mmff94_tables_view_t view, vdw_query_t const *queries,
                                  vdw_result_t *results, std::uint32_t query_count) {
    std::uint32_t const probe_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (probe_index >= query_count) return;

    auto const *lookup_entry = sf::parameters::lookup_vdw(view, queries[probe_index].atom_type);
    vdw_result_t result {};
    if (lookup_entry) {
        result.hit = 1;
        result.alpha = lookup_entry->alpha_i;
        result.effective_electron_count = lookup_entry->n_effective;
        result.polarizability_scale = lookup_entry->a_scale;
        result.slater_kirkwood_factor = lookup_entry->g_factor;
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
        result.force_constant = lookup_entry->force_constant;
        result.equilibrium_distance = lookup_entry->equilibrium_distance;
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
        result.force_constant = lookup_entry->force_constant;
        result.equilibrium_angle_degrees = lookup_entry->theta0_deg;
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
        result.torsion_v1 = lookup_result.second->torsion_v1;
        result.torsion_v2 = lookup_result.second->torsion_v2;
        result.torsion_v3 = lookup_result.second->torsion_v3;
        result.canonical_torsion_type = static_cast<std::uint8_t>(lookup_result.first);
    }
    results[probe_index] = result;
}

#pragma region Host Callable Launchers

/** Threads per block for every probe launch, and with it the grid each launcher rounds up to. */
static constexpr std::uint32_t probe_block_threads_k = 128;

static std::uint32_t probe_blocks_(std::uint32_t query_count) noexcept {
    return (query_count + probe_block_threads_k - 1) / probe_block_threads_k;
}

void launch_device_vdw_lookup(sf::parameters::mmff94_tables_view_t view, vdw_query_t const *queries,
                              vdw_result_t *results, std::uint32_t query_count, cudaStream_t stream) {
    vdw_lookup_kernel<<<probe_blocks_(query_count), probe_block_threads_k, 0, stream>>>(view, queries, results,
                                                                                        query_count);
}

void launch_device_bond_lookup(sf::parameters::mmff94_tables_view_t view, bond_query_t const *queries,
                               bond_result_t *results, std::uint32_t query_count, cudaStream_t stream) {
    bond_lookup_kernel<<<probe_blocks_(query_count), probe_block_threads_k, 0, stream>>>(view, queries, results,
                                                                                         query_count);
}

void launch_device_angle_lookup(sf::parameters::mmff94_tables_view_t view, angle_query_t const *queries,
                                angle_result_t *results, std::uint32_t query_count, cudaStream_t stream) {
    angle_lookup_kernel<<<probe_blocks_(query_count), probe_block_threads_k, 0, stream>>>(view, queries, results,
                                                                                          query_count);
}

void launch_device_torsion_lookup(sf::parameters::mmff94_tables_view_t view, torsion_query_t const *queries,
                                  torsion_result_t *results, std::uint32_t query_count, cudaStream_t stream) {
    torsion_lookup_kernel<<<probe_blocks_(query_count), probe_block_threads_k, 0, stream>>>(view, queries, results,
                                                                                            query_count);
}

#pragma endregion Host Callable Launchers

} // namespace smallfield::tests
