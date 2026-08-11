/**
 *  @file src/smallfield/mmff94/mmff94_extract_device.cuh
 *  @brief `mmff94_extract_terms_kernel` runs the native MMFF term extractor on the GPU.
 *
 *  Layout:
 *   - One block per molecule, with lane 0 doing all the work inside it. Atom typing and lookup
 *     enumeration is an inherently graph-walking algorithm carrying short-lived per-molecule state
 *     such as `mmff94_angle_scratch_t`, so SIMT parallelism across atoms and bonds is possible but
 *     adds substantial complexity for a modest per-molecule speedup — drug-like molecules average
 *     under 30 atoms and 30 bonds. The win is in the cross-molecule dimension instead: 50 K
 *     molecules over the 132 SMs of an H100, at 8 blocks per SM, keeps roughly 1000 molecules in
 *     flight at steady state.
 *   - The extractor and the CSR builder write into per-molecule slots of fixed stride, sized by the
 *     `max_mmff_*_per_mol_k` constants, which costs 116 KB of staging per molecule. Each reports its
 *     actual per-term-type counts to the `per_mol_counts` array.
 *   - `mmff94_scan_counts_kernel` and `mmff94_compact_terms_kernel` turn that staging into the packed
 *     variable-length layout `wave_slice_pool_t` and `mmff_polish_warp_kernel` consume: one contiguous
 *     array per term type, per-molecule bases stamped into `etkdg_molecule_metadata_t`, and per-atom
 *     CSR starts rebased from molecule-relative to batch-global. `test_mmff_gpu_extract` prints both
 *     footprints per molecule and gates the packed arrays byte for byte against the staging; it
 *     measures 17.2 KB per molecule over 19 167 PubChem molecules and 19.4 KB over 20 009 Enamine REAL
 *     molecules, so compaction buys 6.1x to 6.9x on those corpora, and the packed arrays are the ones
 *     a 50 K-molecule wave allocates.
 */

#pragma once

#include <cstdint> // `std::uint16_t`, `std::uint8_t`

#include <cuda_runtime.h>

#include "smallfield/mmff94/mmff94_term_equations.hpp"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/mmff94/mmff94_typer.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/mmff94_bounds.hpp"
#include "smallfield/types/views.hpp"
#include "smallfield/utilities.hpp"

namespace smallfield::mmff94 {
using smallfield::parameters::lookup_angle;
using smallfield::parameters::lookup_bndk;
using smallfield::parameters::lookup_bond;
using smallfield::parameters::lookup_covrad;
using smallfield::parameters::lookup_dfsb;
using smallfield::parameters::lookup_hl;
using smallfield::parameters::lookup_oop;
using smallfield::parameters::lookup_prop;
using smallfield::parameters::lookup_stbn;
using smallfield::parameters::lookup_torsion;
using smallfield::parameters::lookup_vdw;
using smallfield::parameters::mmff94_tables_view_t;
using smallfield::types::atom_role_first_k;
using smallfield::types::atom_role_fourth_k;
using smallfield::types::atom_role_second_k;
using smallfield::types::atom_role_third_k;
using smallfield::types::bond_flag_aromatic_k;
using smallfield::types::etkdg_molecule_metadata_t;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::max_bonds_k;
using smallfield::types::max_mmff_angles_per_mol_k;
using smallfield::types::max_mmff_bonds_per_mol_k;
using smallfield::types::max_mmff_interactions_per_mol_k;
using smallfield::types::max_mmff_oops_per_mol_k;
using smallfield::types::max_mmff_stretchbends_per_mol_k;
using smallfield::types::max_mmff_torsions_per_mol_k;
using smallfield::types::mmff94_angle_center_k;
using smallfield::types::mmff94_angle_end_k;
using smallfield::types::mmff94_angle_t;
using smallfield::types::mmff94_atom_interaction_t;
using smallfield::types::mmff94_atom_offsets_t;
using smallfield::types::mmff94_atom_vdw_params_t;
using smallfield::types::mmff94_bond_k;
using smallfield::types::mmff94_bond_t;
using smallfield::types::mmff94_oop_t;
using smallfield::types::mmff94_out_of_plane_k;
using smallfield::types::mmff94_stretch_bend_k;
using smallfield::types::mmff94_stretchbend_t;
using smallfield::types::mmff94_torsion_k;
using smallfield::types::mmff94_torsion_t;
using smallfield::types::molecule_descriptor_t;

/** Per-molecule term counts. Written by the emit kernel; consumed by
 *  the CSR build kernel and by downstream pipeline stages that need
 *  per-molecule offsets into the term arrays. */
struct mmff94_per_mol_counts_t {
    std::uint16_t bonds;
    std::uint16_t angles;
    std::uint16_t torsions;
    std::uint16_t oops;
    std::uint16_t stretchbends;
    std::uint16_t interactions;
    /** Non-zero if any fixed stride was saturated. */
    std::uint8_t overflow;
    std::uint8_t padding;
};

/**
 *  @brief Batch-global base offset of one molecule inside each packed term array.
 *
 *  `mmff94_scan_counts_kernel` produces one entry per molecule as the exclusive prefix sum
 *  of `mmff94_per_mol_counts_t`, and its one-element companion output carries the batch
 *  totals a caller reads back to size the packed arrays. The conformer pipeline instead fills
 *  the array on the host from per-molecule capacity bounds and skips the scan entirely, so the
 *  sizing never crosses the bus.
 */
using smallfield::types::mmff94_packed_offsets_t;

/** Number of independently scanned components in `mmff94_packed_offsets_t`. */
inline constexpr std::uint32_t mmff94_packed_components_k = 6;

/** Thread count of the single-block scan that turns per-molecule counts into packed offsets. */
inline constexpr std::uint32_t mmff94_scan_block_k = 512;

/** Angle scratch — populated during the angle pass and consumed by the
 *  stretch-bend pass. Sized at `max_mmff_angles_per_mol_k` slots per
 *  molecule and held in shared memory, one array per block rather than
 *  one per thread: 16 B × 512 = 8 KB per block, which the static
 *  `__shared__` assertion below pins to the struct layout. */
struct mmff94_angle_scratch_t {
    std::uint8_t atom_i, atom_j, atom_k;
    std::uint8_t angle_type;
    std::uint8_t bond_type_ij, bond_type_kj;
    std::int16_t bond_index_ij, bond_index_kj;
    float theta0_deg;
};

static_assert(sizeof(mmff94_angle_scratch_t) == 16, "angle scratch must stay at 8 KB per block");

#if defined(__CUDACC__)
/** Single-kernel MMFF94 term extractor. One block per molecule; lane 0
 *  of the block does all enumeration + classification + lookup. Outputs
 *  are written into the per-molecule fixed-stride slots of the output
 *  arrays. Blockers on this kernel are avoided by keeping per-molecule
 *  state in shared memory: `shared_angle_scratch` plus the per-bond
 *  `shared_bond_type` and `shared_bond_equilibrium_distance`. */
static __launch_bounds__(32, 4) __global__ void mmff94_extract_terms_kernel(
    // Inputs
    molecule_descriptor_t const *__restrict__ descriptors, std::uint8_t const *__restrict__ atom_types_all,
    float const *__restrict__ atom_charges_all, std::uint32_t const *__restrict__ atom_offsets,
    std::uint32_t num_molecules, mmff94_tables_view_t tables,
    // Outputs — fixed-stride per-molecule slices
    mmff94_atom_vdw_params_t *__restrict__ out_atom_vdw, float *__restrict__ out_atom_charge,
    mmff94_bond_t *__restrict__ out_bonds, mmff94_angle_t *__restrict__ out_angles,
    mmff94_torsion_t *__restrict__ out_torsions, mmff94_oop_t *__restrict__ out_oops,
    mmff94_stretchbend_t *__restrict__ out_stbns, mmff94_per_mol_counts_t *__restrict__ per_mol_counts,
    std::uint8_t const *__restrict__ typing_status) {

    std::uint32_t const molecule = blockIdx.x;
    if (molecule >= num_molecules) return;

    // Early-exit status check runs on all 32 lanes so they all
    // agree to fall out together (no warp divergence afterwards).
    if (typing_status[molecule] != 0) {
        if (threadIdx.x == 0) per_mol_counts[molecule] = {};
        return;
    }

    std::uint32_t const lane = threadIdx.x;
    molecule_descriptor_t const &descriptor = descriptors[molecule];
    std::uint32_t const atom_off = atom_offsets[molecule];
    std::uint8_t const *atom_types = atom_types_all + atom_off;
    float const *charges = atom_charges_all + atom_off;
    std::uint32_t const num_atoms = descriptor.num_atoms;
    std::uint32_t const num_bonds = descriptor.num_bonds;

    // Per-molecule output bases.
    mmff94_atom_vdw_params_t *vdw_out = out_atom_vdw + atom_off;
    float *charge_out = out_atom_charge + atom_off;
    mmff94_bond_t *bonds_out = out_bonds + (std::size_t)molecule * max_mmff_bonds_per_mol_k;
    mmff94_angle_t *angles_out = out_angles + (std::size_t)molecule * max_mmff_angles_per_mol_k;
    mmff94_torsion_t *torsions_out = out_torsions + (std::size_t)molecule * max_mmff_torsions_per_mol_k;
    mmff94_oop_t *oops_out = out_oops + (std::size_t)molecule * max_mmff_oops_per_mol_k;
    mmff94_stretchbend_t *stbns_out = out_stbns + (std::size_t)molecule * max_mmff_stretchbends_per_mol_k;

    // Atom VdW + charge — lane-parallel over atoms (each atom's work
    // is independent and written to its own slot).
    for (std::uint32_t atom = lane; atom < num_atoms; atom += 32) {
        auto const *vdw_entry = lookup_vdw(tables, atom_types[atom]);
        mmff94_atom_vdw_params_t vdw_params {};
        if (vdw_entry) {
            vdw_params.alpha = vdw_entry->alpha_i;
            vdw_params.n_effective = vdw_entry->n_effective;
            vdw_params.a_scale = vdw_entry->a_scale;
            vdw_params.g_factor = vdw_entry->g_factor;
            vdw_params.r_star = vdw_entry->r_star;
            vdw_params.donor_acceptor = vdw_entry->donor_acceptor;
        }
        vdw_out[atom] = vdw_params;
        charge_out[atom] = charges[atom];
    }

    // Precompute per-bond bond_type + bond-stretch params — lane-parallel
    // over bonds. Shared-memory scratch for downstream angle/stbn reuse.
    __shared__ std::uint8_t shared_bond_type[max_bonds_k];
    __shared__ float shared_bond_equilibrium_distance[max_bonds_k];
    for (std::uint32_t bond = lane; bond < num_bonds; bond += 32) {
        std::uint8_t const atom_i = descriptor.bond_atom_i[bond];
        std::uint8_t const atom_j = descriptor.bond_atom_j[bond];
        std::uint8_t bond_type = 0;
        bool const arom_bond = (descriptor.bond_flags[bond] & bond_flag_aromatic_k) != 0;
        if (descriptor.bond_order[bond] == 1 && !arom_bond) {
            auto const *prop_i = lookup_prop(tables, atom_types[atom_i]);
            auto const *prop_j = lookup_prop(tables, atom_types[atom_j]);
            if (prop_i && prop_j &&
                ((prop_i->single_bond_multiple_bond && prop_j->single_bond_multiple_bond) ||
                 (prop_i->aromatic && prop_j->aromatic)))
                bond_type = 1;
        }
        shared_bond_type[bond] = bond_type;
        auto const *bond_parameters = lookup_bond(tables, bond_type, atom_types[atom_i], atom_types[atom_j]);
        float force_constant = 0.0f, equilibrium_distance = 0.0f;
        if (bond_parameters) {
            force_constant = bond_parameters->force_constant;
            equilibrium_distance = bond_parameters->equilibrium_distance;
        }
        else {
            auto const [empirical_force_constant, empirical_equilibrium_distance] = native_bond_empirical(
                tables, descriptor.element[atom_i], descriptor.element[atom_j]);
            force_constant = empirical_force_constant;
            equilibrium_distance = empirical_equilibrium_distance;
        }
        shared_bond_equilibrium_distance[bond] = equilibrium_distance;
        // Emit bond at deterministic position = original bond index `bond`
        // so output ordering matches the CPU (which also iterates bonds
        // in index order and push_backs in order).
        if (bond < max_mmff_bonds_per_mol_k) {
            bonds_out[bond] = {.atom_i = atom_i,
                               .atom_j = atom_j,
                               .padding = {0, 0},
                               .force_constant = force_constant,
                               .equilibrium_distance = equilibrium_distance,
                               .padding_float = 0.0f};
        }
    }
    __syncwarp();

    // Angle scratch — populated by the angle pass, consumed by the stretch-bend
    // pass. Block-scoped, so the cost is 8 KB per warp rather than per thread.
    __shared__ mmff94_angle_scratch_t shared_angle_scratch[max_mmff_angles_per_mol_k];

    // From here on the passes with atomic cursors must run on lane-0
    // only to preserve deterministic ordering for byte-equality with CPU.
    if (lane != 0) return;

    // Both arrays are indexed by original bond index: the MMFF bond type of every bond, and the
    // equilibrium bond length the angle and stretch-bend passes read back.
    std::uint8_t const *bond_type_arr = shared_bond_type;
    float const *bond_equilibrium_distance = shared_bond_equilibrium_distance;

    std::uint32_t bond_count = num_bonds < max_mmff_bonds_per_mol_k ? num_bonds : max_mmff_bonds_per_mol_k;
    std::uint32_t angle_count = 0, torsion_count = 0, oop_count = 0, stbn_count = 0;
    std::uint8_t overflow = (num_bonds > max_mmff_bonds_per_mol_k) ? 1 : 0;

    // Angle terms (and angle-scratch for stretch-bend reuse).
    std::uint32_t angle_scratch_count = 0;

    custd::span<std::uint8_t const> bond_types {bond_type_arr, num_bonds};
    custd::span<std::uint8_t const> at_span {atom_types, num_atoms};
    for (std::uint32_t atom_j = 0; atom_j < num_atoms; ++atom_j) {
        std::uint16_t const nstart = descriptor.neighbor_offset[atom_j];
        std::uint16_t const nend = descriptor.neighbor_offset[atom_j + 1];
        for (std::uint16_t first_neighbor_slot = nstart; first_neighbor_slot < nend; ++first_neighbor_slot) {
            for (std::uint16_t second_neighbor_slot = first_neighbor_slot + 1; second_neighbor_slot < nend;
                 ++second_neighbor_slot) {
                std::uint32_t const atom_i = descriptor.neighbor_atom[first_neighbor_slot];
                std::uint32_t const atom_k = descriptor.neighbor_atom[second_neighbor_slot];
                std::int32_t const bond_index_ij = find_bond_index(descriptor, atom_i, atom_j);
                std::int32_t const bond_index_kj = find_bond_index(descriptor, atom_j, atom_k);
                if (bond_index_ij < 0 || bond_index_kj < 0) continue;
                std::uint8_t const at_code = native_get_mmff_angle_type(descriptor, bond_types, atom_i, atom_j, atom_k);
                auto const *angle_params = lookup_angle(tables, at_code, atom_types[atom_i], atom_types[atom_j],
                                                        atom_types[atom_k]);
                float force_constant = 0.0f, theta0_deg = 0.0f;
                bool use_empirical = !angle_params || native_is_zero(angle_params->force_constant);
                if (!use_empirical) {
                    force_constant = angle_params->force_constant;
                    theta0_deg = angle_params->theta0_deg;
                }
                if (use_empirical) {
                    float const equilibrium_distance_ij = bond_equilibrium_distance[bond_index_ij];
                    float const equilibrium_distance_kj = bond_equilibrium_distance[bond_index_kj];
                    float const override_deg = angle_params ? angle_params->theta0_deg : -1.0f;
                    auto const [empirical_force_constant, empirical_theta0_deg] = native_angle_empirical(
                        descriptor, tables, at_span, atom_i, atom_j, atom_k, equilibrium_distance_ij,
                        equilibrium_distance_kj, override_deg);
                    force_constant = empirical_force_constant;
                    theta0_deg = empirical_theta0_deg;
                }
                float const theta0_rad = theta0_deg * (float)M_PI / 180.0f;
                auto const *prop_center = lookup_prop(tables, at_span[atom_j]);
                std::uint8_t const is_linear = (prop_center && prop_center->linear) ? 1u : 0u;
                if (angle_count < max_mmff_angles_per_mol_k) {
                    angles_out[angle_count++] = {.atom_i = (std::uint8_t)atom_i,
                                                 .atom_j = (std::uint8_t)atom_j,
                                                 .atom_k = (std::uint8_t)atom_k,
                                                 .is_linear = is_linear,
                                                 .force_constant = force_constant,
                                                 .equilibrium_angle = theta0_rad,
                                                 .padding_float = 0.0f};
                }
                else { overflow = 1; }
                if (angle_scratch_count < max_mmff_angles_per_mol_k) {
                    shared_angle_scratch[angle_scratch_count++] = {.atom_i = (std::uint8_t)atom_i,
                                                                   .atom_j = (std::uint8_t)atom_j,
                                                                   .atom_k = (std::uint8_t)atom_k,
                                                                   .angle_type = at_code,
                                                                   .bond_type_ij = bond_type_arr[bond_index_ij],
                                                                   .bond_type_kj = bond_type_arr[bond_index_kj],
                                                                   .bond_index_ij = (std::int16_t)bond_index_ij,
                                                                   .bond_index_kj = (std::int16_t)bond_index_kj,
                                                                   .theta0_deg = theta0_deg};
                }
            }
        }
    }

    // Torsion terms.
    for (std::uint32_t bond = 0; bond < num_bonds; ++bond) {
        std::uint32_t const atom_j = descriptor.bond_atom_i[bond];
        std::uint32_t const atom_k = descriptor.bond_atom_j[bond];
        std::uint16_t const jstart = descriptor.neighbor_offset[atom_j];
        std::uint16_t const jend = descriptor.neighbor_offset[atom_j + 1];
        std::uint16_t const kstart = descriptor.neighbor_offset[atom_k];
        std::uint16_t const kend = descriptor.neighbor_offset[atom_k + 1];
        for (std::uint16_t j_neighbor_slot = jstart; j_neighbor_slot < jend; ++j_neighbor_slot) {
            std::uint32_t const atom_i = descriptor.neighbor_atom[j_neighbor_slot];
            if (atom_i == atom_k) continue;
            for (std::uint16_t k_neighbor_slot = kstart; k_neighbor_slot < kend; ++k_neighbor_slot) {
                std::uint32_t const atom_l = descriptor.neighbor_atom[k_neighbor_slot];
                if (atom_l == atom_j) continue;
                auto const [torsion_type_primary, torsion_type_secondary] = native_get_mmff_torsion_type(
                    descriptor, at_span, bond_types, atom_i, atom_j, atom_k, atom_l);
                [[maybe_unused]] auto const [canonical_torsion_type, torsion_entry] = lookup_torsion(
                    tables, torsion_type_primary, torsion_type_secondary, atom_types[atom_i], atom_types[atom_j],
                    atom_types[atom_k], atom_types[atom_l]);
                float torsion_v1 = 0.0f, torsion_v2 = 0.0f, torsion_v3 = 0.0f;
                if (torsion_entry) {
                    torsion_v1 = torsion_entry->torsion_v1;
                    torsion_v2 = torsion_entry->torsion_v2;
                    torsion_v3 = torsion_entry->torsion_v3;
                }
                else {
                    auto const [empirical_v1, empirical_v2, empirical_v3] = native_torsion_empirical(
                        descriptor, tables, at_span, atom_j, atom_k, (std::int32_t)bond);
                    torsion_v1 = empirical_v1;
                    torsion_v2 = empirical_v2;
                    torsion_v3 = empirical_v3;
                }
                if (native_is_zero(torsion_v1) && native_is_zero(torsion_v2) && native_is_zero(torsion_v3)) continue;
                if (torsion_count < max_mmff_torsions_per_mol_k) {
                    torsions_out[torsion_count++] = {.atom_i = (std::uint8_t)atom_i,
                                                     .atom_j = (std::uint8_t)atom_j,
                                                     .atom_k = (std::uint8_t)atom_k,
                                                     .atom_l = (std::uint8_t)atom_l,
                                                     .torsion_v1 = torsion_v1,
                                                     .torsion_v2 = torsion_v2,
                                                     .torsion_v3 = torsion_v3};
                }
                else { overflow = 1; }
            }
        }
    }

    // OOP terms.
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint32_t const degree = descriptor.degree[center];
        if (degree < 3) continue;
        std::uint16_t const nstart = descriptor.neighbor_offset[center];
        std::uint8_t neighbor_cache[8];
        std::uint32_t neighbor_count = 0;
        for (std::uint16_t neighbor_slot = nstart;
             neighbor_slot < descriptor.neighbor_offset[center + 1] && neighbor_count < 8; ++neighbor_slot)
            neighbor_cache[neighbor_count++] = descriptor.neighbor_atom[neighbor_slot];
        for (std::uint32_t apex_neighbor = 0; apex_neighbor < neighbor_count; ++apex_neighbor) {
            std::uint8_t plane[2];
            std::uint32_t plane_count = 0;
            for (std::uint32_t plane_candidate = 0; plane_candidate < neighbor_count && plane_count < 2;
                 ++plane_candidate)
                if (plane_candidate != apex_neighbor) plane[plane_count++] = neighbor_cache[plane_candidate];
            if (plane_count < 2) continue;
            auto const *oop_params = lookup_oop(tables, atom_types[neighbor_cache[apex_neighbor]], atom_types[center],
                                                atom_types[plane[0]], atom_types[plane[1]]);
            if (!oop_params) continue;
            if (oop_count < max_mmff_oops_per_mol_k) {
                oops_out[oop_count++] = {.atom_i = neighbor_cache[apex_neighbor],
                                         .atom_j = (std::uint8_t)center,
                                         .atom_k = plane[0],
                                         .atom_l = plane[1],
                                         .force_constant = oop_params->force_constant,
                                         .padding = {0.0f, 0.0f}};
            }
            else { overflow = 1; }
        }
    }

    // Stretch-bend terms (reuse angle scratch).
    for (std::uint32_t angle_index = 0; angle_index < angle_scratch_count; ++angle_index) {
        auto const &angle = shared_angle_scratch[angle_index];
        auto const *prop_j = lookup_prop(tables, atom_types[angle.atom_j]);
        if (prop_j && prop_j->linear) continue;
        std::uint8_t const stretch_bend_type = native_get_mmff_stretch_bend_type(
            angle.angle_type,
            (atom_types[angle.atom_i] <= atom_types[angle.atom_k]) ? angle.bond_type_ij : angle.bond_type_kj,
            (atom_types[angle.atom_i] < atom_types[angle.atom_k]) ? angle.bond_type_kj : angle.bond_type_ij);
        auto stbn = lookup_stbn(tables, stretch_bend_type, angle.bond_type_ij, angle.bond_type_kj,
                                atom_types[angle.atom_i], atom_types[angle.atom_j], atom_types[angle.atom_k]);
        float kba_ijk = 0.0f, kba_kji = 0.0f;
        bool swap_flag = false;
        if (stbn.second) {
            kba_ijk = stbn.second->kba_ijk;
            kba_kji = stbn.second->kba_kji;
            swap_flag = stbn.first;
        }
        else {
            auto dfsb = lookup_dfsb(tables, periodic_table_row(descriptor.element[angle.atom_i]),
                                    periodic_table_row(descriptor.element[angle.atom_j]),
                                    periodic_table_row(descriptor.element[angle.atom_k]));
            if (!dfsb.second) continue;
            kba_ijk = dfsb.second->kba_ijk;
            kba_kji = dfsb.second->kba_kji;
            swap_flag = dfsb.first;
        }
        if (native_is_zero(kba_ijk) && native_is_zero(kba_kji)) continue;
        if (swap_flag) swap_(kba_ijk, kba_kji);
        float const theta0_rad = angle.theta0_deg * (float)M_PI / 180.0f;
        float const equilibrium_distance_ij = bond_equilibrium_distance[angle.bond_index_ij];
        float const equilibrium_distance_kj = bond_equilibrium_distance[angle.bond_index_kj];
        if (stbn_count < max_mmff_stretchbends_per_mol_k) {
            stbns_out[stbn_count++] = {.atom_i = angle.atom_i,
                                       .atom_j = angle.atom_j,
                                       .atom_k = angle.atom_k,
                                       .padding = 0,
                                       .kba_ijk = kba_ijk,
                                       .kba_kji = kba_kji,
                                       .equilibrium_angle = theta0_rad,
                                       .equilibrium_distance_ij = equilibrium_distance_ij,
                                       .equilibrium_distance_kj = equilibrium_distance_kj,
                                       .padding_float = 0.0f};
        }
        else { overflow = 1; }
    }

    per_mol_counts[molecule] = {.bonds = (std::uint16_t)bond_count,
                                .angles = (std::uint16_t)angle_count,
                                .torsions = (std::uint16_t)torsion_count,
                                .oops = (std::uint16_t)oop_count,
                                .stretchbends = (std::uint16_t)stbn_count,
                                .interactions = 0, // Stamped by `mmff94_build_csr_kernel`.
                                .overflow = overflow,
                                .padding = 0};
}

/** Build the MMFF per-atom CSR interaction list from emitted terms.
 *  Mirrors the host CSR build in `mmff94_extract.hpp` — two-pass count + write. One
 *  block per molecule, lane-0-only (the CSR-build cost is tiny
 *  relative to the term extractor and cross-molecule parallelism
 *  dominates). */
static __launch_bounds__(32, 8) __global__ void mmff94_build_csr_kernel(
    molecule_descriptor_t const *__restrict__ descriptors, std::uint32_t const *__restrict__ atom_offsets,
    std::uint32_t num_molecules, mmff94_per_mol_counts_t *__restrict__ per_mol_counts,
    mmff94_bond_t const *__restrict__ bonds_all, mmff94_angle_t const *__restrict__ angles_all,
    mmff94_torsion_t const *__restrict__ torsions_all, mmff94_oop_t const *__restrict__ oops_all,
    mmff94_stretchbend_t const *__restrict__ stbns_all, mmff94_atom_interaction_t *__restrict__ interactions_all,
    mmff94_atom_offsets_t *__restrict__ atom_offsets_all, std::uint8_t const *__restrict__ typing_status) {

    // Per-atom per-type write cursors. Block-scoped so the five arrays cost
    // 2.5 KB per warp instead of per thread; only lane 0 ever touches them.
    __shared__ std::uint32_t cursors_bond[max_atoms_per_molecule_k];
    __shared__ std::uint32_t cursors_angle[max_atoms_per_molecule_k];
    __shared__ std::uint32_t cursors_torsion[max_atoms_per_molecule_k];
    __shared__ std::uint32_t cursors_oop[max_atoms_per_molecule_k];
    __shared__ std::uint32_t cursors_stretchbend[max_atoms_per_molecule_k];

    std::uint32_t const molecule = blockIdx.x;
    if (molecule >= num_molecules) return;
    if (threadIdx.x != 0) return;
    if (typing_status[molecule] != 0) return;

    auto const &descriptor = descriptors[molecule];
    std::uint32_t const num_atoms = descriptor.num_atoms;
    std::uint32_t const atom_base = atom_offsets[molecule];
    auto const &counts = per_mol_counts[molecule];

    mmff94_bond_t const *bonds = bonds_all + (std::size_t)molecule * max_mmff_bonds_per_mol_k;
    mmff94_angle_t const *angles = angles_all + (std::size_t)molecule * max_mmff_angles_per_mol_k;
    mmff94_torsion_t const *torsions = torsions_all + (std::size_t)molecule * max_mmff_torsions_per_mol_k;
    mmff94_oop_t const *oops = oops_all + (std::size_t)molecule * max_mmff_oops_per_mol_k;
    mmff94_stretchbend_t const *stbns = stbns_all + (std::size_t)molecule * max_mmff_stretchbends_per_mol_k;

    std::uint32_t const ints_base = (std::uint32_t)molecule * max_mmff_interactions_per_mol_k;
    mmff94_atom_interaction_t *interactions = interactions_all + ints_base;
    mmff94_atom_offsets_t *offsets = atom_offsets_all + atom_base;

    // Pass 1 — count per-atom per-term-type.
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) offsets[atom] = {};
    for (std::uint32_t bond = 0; bond < counts.bonds; ++bond) {
        offsets[bonds[bond].atom_i].bond_count++;
        offsets[bonds[bond].atom_j].bond_count++;
    }
    for (std::uint32_t angle = 0; angle < counts.angles; ++angle) {
        offsets[angles[angle].atom_i].angle_count++;
        offsets[angles[angle].atom_j].angle_count++;
        offsets[angles[angle].atom_k].angle_count++;
    }
    for (std::uint32_t torsion = 0; torsion < counts.torsions; ++torsion) {
        offsets[torsions[torsion].atom_i].torsion_count++;
        offsets[torsions[torsion].atom_j].torsion_count++;
        offsets[torsions[torsion].atom_k].torsion_count++;
        offsets[torsions[torsion].atom_l].torsion_count++;
    }
    for (std::uint32_t oop_term = 0; oop_term < counts.oops; ++oop_term) {
        offsets[oops[oop_term].atom_i].oop_count++;
        offsets[oops[oop_term].atom_j].oop_count++;
        offsets[oops[oop_term].atom_k].oop_count++;
        offsets[oops[oop_term].atom_l].oop_count++;
    }
    for (std::uint32_t stretch_bend = 0; stretch_bend < counts.stretchbends; ++stretch_bend) {
        offsets[stbns[stretch_bend].atom_i].stretchbend_count++;
        offsets[stbns[stretch_bend].atom_j].stretchbend_count++;
        offsets[stbns[stretch_bend].atom_k].stretchbend_count++;
    }

    // Compute per-type offsets + running total.
    std::uint32_t running = 0;
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        auto &atom_offset_entry = offsets[atom];
        atom_offset_entry.bond_offset = running;
        running += atom_offset_entry.bond_count;
        atom_offset_entry.angle_offset = running;
        running += atom_offset_entry.angle_count;
        atom_offset_entry.torsion_offset = running;
        running += atom_offset_entry.torsion_count;
        atom_offset_entry.oop_offset = running;
        running += atom_offset_entry.oop_count;
        atom_offset_entry.stretchbend_offset = running;
        running += atom_offset_entry.stretchbend_count;
    }
    // Running is the total interactions for this molecule. Store in
    // per_mol_counts for downstream CSR consumers.
    per_mol_counts[molecule].interactions = (std::uint16_t)running;

    // Seed the write cursors from the `_offset` fields just computed.
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom) {
        cursors_bond[atom] = offsets[atom].bond_offset;
        cursors_angle[atom] = offsets[atom].angle_offset;
        cursors_torsion[atom] = offsets[atom].torsion_offset;
        cursors_oop[atom] = offsets[atom].oop_offset;
        cursors_stretchbend[atom] = offsets[atom].stretchbend_offset;
    }

    // Pass 2 — write interactions.
    for (std::uint32_t bond = 0; bond < counts.bonds; ++bond) {
        interactions[cursors_bond[bonds[bond].atom_i]++] = {.term_index = bond,
                                                            .interaction_type = mmff94_bond_k,
                                                            .role = atom_role_first_k};
        interactions[cursors_bond[bonds[bond].atom_j]++] = {.term_index = bond,
                                                            .interaction_type = mmff94_bond_k,
                                                            .role = atom_role_second_k};
    }
    for (std::uint32_t angle = 0; angle < counts.angles; ++angle) {
        interactions[cursors_angle[angles[angle].atom_i]++] = {.term_index = angle,
                                                               .interaction_type = mmff94_angle_end_k,
                                                               .role = atom_role_first_k};
        interactions[cursors_angle[angles[angle].atom_j]++] = {.term_index = angle,
                                                               .interaction_type = mmff94_angle_center_k,
                                                               .role = atom_role_second_k};
        interactions[cursors_angle[angles[angle].atom_k]++] = {.term_index = angle,
                                                               .interaction_type = mmff94_angle_end_k,
                                                               .role = atom_role_third_k};
    }
    for (std::uint32_t torsion = 0; torsion < counts.torsions; ++torsion) {
        interactions[cursors_torsion[torsions[torsion].atom_i]++] = {.term_index = torsion,
                                                                     .interaction_type = mmff94_torsion_k,
                                                                     .role = atom_role_first_k};
        interactions[cursors_torsion[torsions[torsion].atom_j]++] = {.term_index = torsion,
                                                                     .interaction_type = mmff94_torsion_k,
                                                                     .role = atom_role_second_k};
        interactions[cursors_torsion[torsions[torsion].atom_k]++] = {.term_index = torsion,
                                                                     .interaction_type = mmff94_torsion_k,
                                                                     .role = atom_role_third_k};
        interactions[cursors_torsion[torsions[torsion].atom_l]++] = {.term_index = torsion,
                                                                     .interaction_type = mmff94_torsion_k,
                                                                     .role = atom_role_fourth_k};
    }
    for (std::uint32_t oop_term = 0; oop_term < counts.oops; ++oop_term) {
        interactions[cursors_oop[oops[oop_term].atom_i]++] = {.term_index = oop_term,
                                                              .interaction_type = mmff94_out_of_plane_k,
                                                              .role = atom_role_first_k};
        interactions[cursors_oop[oops[oop_term].atom_j]++] = {.term_index = oop_term,
                                                              .interaction_type = mmff94_out_of_plane_k,
                                                              .role = atom_role_second_k};
        interactions[cursors_oop[oops[oop_term].atom_k]++] = {.term_index = oop_term,
                                                              .interaction_type = mmff94_out_of_plane_k,
                                                              .role = atom_role_third_k};
        interactions[cursors_oop[oops[oop_term].atom_l]++] = {.term_index = oop_term,
                                                              .interaction_type = mmff94_out_of_plane_k,
                                                              .role = atom_role_fourth_k};
    }
    for (std::uint32_t stretch_bend = 0; stretch_bend < counts.stretchbends; ++stretch_bend) {
        interactions[cursors_stretchbend[stbns[stretch_bend].atom_i]++] = {.term_index = stretch_bend,
                                                                           .interaction_type = mmff94_stretch_bend_k,
                                                                           .role = atom_role_first_k};
        interactions[cursors_stretchbend[stbns[stretch_bend].atom_j]++] = {.term_index = stretch_bend,
                                                                           .interaction_type = mmff94_stretch_bend_k,
                                                                           .role = atom_role_second_k};
        interactions[cursors_stretchbend[stbns[stretch_bend].atom_k]++] = {.term_index = stretch_bend,
                                                                           .interaction_type = mmff94_stretch_bend_k,
                                                                           .role = atom_role_third_k};
    }
}

/**
 *  @brief Exclusive prefix sum of the per-molecule term counts, plus the batch totals.
 *
 *  Launched as a single block of `mmff94_scan_block_k` threads that sweeps the molecule
 *  axis one tile at a time, carrying the running sums in registers. A 50 K-molecule wave is
 *  98 tiles of a six-component Hillis-Steele scan, which costs a few microseconds against the
 *  millisecond-scale extractor it feeds, and it needs no temporary block-sum array.
 *
 *  @param[in] per_mol_counts one entry per molecule, as written by the extractor and the CSR builder
 *  @param[in] num_molecules number of entries in @p per_mol_counts
 *  @param[out] per_mol_offsets one entry per molecule, the base of that molecule in each packed array
 *  @param[out] totals single entry holding the batch-wide element count of each packed array
 */
static __launch_bounds__(mmff94_scan_block_k, 1) __global__
    void mmff94_scan_counts_kernel(mmff94_per_mol_counts_t const *__restrict__ per_mol_counts,
                                   std::uint32_t num_molecules, mmff94_packed_offsets_t *__restrict__ per_mol_offsets,
                                   mmff94_packed_offsets_t *__restrict__ totals) {

    __shared__ std::uint32_t partial_sums[mmff94_packed_components_k][mmff94_scan_block_k];
    std::uint32_t const slot = threadIdx.x;
    std::uint32_t carry[mmff94_packed_components_k] = {0, 0, 0, 0, 0, 0};

    for (std::uint32_t tile_base = 0; tile_base < num_molecules; tile_base += mmff94_scan_block_k) {
        std::uint32_t const molecule = tile_base + slot;
        std::uint32_t own_counts[mmff94_packed_components_k] = {0, 0, 0, 0, 0, 0};
        if (molecule < num_molecules) {
            auto const counts = per_mol_counts[molecule];
            own_counts[0] = counts.bonds;
            own_counts[1] = counts.angles;
            own_counts[2] = counts.torsions;
            own_counts[3] = counts.oops;
            own_counts[4] = counts.stretchbends;
            own_counts[5] = counts.interactions;
        }
        for (std::uint32_t component = 0; component < mmff94_packed_components_k; ++component)
            partial_sums[component][slot] = own_counts[component];
        __syncthreads();

        for (std::uint32_t stride = 1; stride < mmff94_scan_block_k; stride <<= 1) {
            std::uint32_t addend[mmff94_packed_components_k];
            for (std::uint32_t component = 0; component < mmff94_packed_components_k; ++component)
                addend[component] = slot >= stride ? partial_sums[component][slot - stride] : 0u;
            __syncthreads();
            for (std::uint32_t component = 0; component < mmff94_packed_components_k; ++component)
                partial_sums[component][slot] += addend[component];
            __syncthreads();
        }

        if (molecule < num_molecules)
            per_mol_offsets[molecule] = {.bonds = carry[0] + partial_sums[0][slot] - own_counts[0],
                                         .angles = carry[1] + partial_sums[1][slot] - own_counts[1],
                                         .torsions = carry[2] + partial_sums[2][slot] - own_counts[2],
                                         .oops = carry[3] + partial_sums[3][slot] - own_counts[3],
                                         .stretchbends = carry[4] + partial_sums[4][slot] - own_counts[4],
                                         .interactions = carry[5] + partial_sums[5][slot] - own_counts[5]};
        for (std::uint32_t component = 0; component < mmff94_packed_components_k; ++component)
            carry[component] += partial_sums[component][mmff94_scan_block_k - 1];
        __syncthreads();
    }

    if (slot == 0)
        *totals = {.bonds = carry[0],
                   .angles = carry[1],
                   .torsions = carry[2],
                   .oops = carry[3],
                   .stretchbends = carry[4],
                   .interactions = carry[5]};
}

/** Lane-strided copy of @p count elements from the staging slot into the packed array. */
template <typename element_>
__device__ inline void copy_packed_terms_(element_ const *__restrict__ source, element_ *__restrict__ destination,
                                          std::uint32_t count, std::uint32_t lane) {
    for (std::uint32_t element_index = lane; element_index < count; element_index += 32)
        destination[element_index] = source[element_index];
}

/**
 *  @brief Compact the fixed-stride staging arrays into the packed layout `wave_slice_pool_t` uses.
 *
 *  One warp per molecule. Every term type is copied from its 116 KB per-molecule staging slot
 *  to the batch-global base the scan assigned, the per-atom CSR starts are rebased from
 *  molecule-relative to batch-global, and the molecule's metadata entry gains the offsets and
 *  counts that `mmff_polish_warp_kernel` and `bond_distance_kernel` read. A molecule the typer
 *  routed to the CPU fallback carries zero counts, so it contributes an empty slice and its
 *  metadata offsets stay well-formed.
 *
 *  @param[in] descriptors device-resident molecule descriptors, read for the atom count
 *  @param[in] atom_offsets batch-global first-atom index of each molecule
 *  @param[in] num_molecules number of molecules in the batch
 *  @param[in] per_mol_counts per-molecule term counts from the extractor and the CSR builder
 *  @param[in] per_mol_offsets per-molecule packed bases from `mmff94_scan_counts_kernel`
 *  @param[in] staged_bonds fixed-stride bond staging, `max_mmff_bonds_per_mol_k` per molecule
 *  @param[in] staged_angles fixed-stride angle staging
 *  @param[in] staged_torsions fixed-stride torsion staging
 *  @param[in] staged_oops fixed-stride out-of-plane staging
 *  @param[in] staged_stbns fixed-stride stretch-bend staging
 *  @param[in] staged_interactions fixed-stride CSR staging, `max_mmff_interactions_per_mol_k` per molecule
 *  @param[out] packed_bonds packed bond array, sized from the scan totals
 *  @param[out] packed_angles packed angle array
 *  @param[out] packed_torsions packed torsion array
 *  @param[out] packed_oops packed out-of-plane array
 *  @param[out] packed_stbns packed stretch-bend array
 *  @param[out] packed_interactions packed CSR interaction array
 *  @param[inout] atom_interaction_offsets per-atom CSR starts, rebased in place
 *  @param[out] metadata per-molecule metadata whose MMFF offset and count fields are stamped here
 *  @param[in] typing_status per-molecule typer status; non-zero routes the molecule to the CPU fallback
 */
static __launch_bounds__(32, 8) __global__ void mmff94_compact_terms_kernel(
    molecule_descriptor_t const *__restrict__ descriptors, std::uint32_t const *__restrict__ atom_offsets,
    std::uint32_t num_molecules, mmff94_per_mol_counts_t const *__restrict__ per_mol_counts,
    mmff94_packed_offsets_t const *__restrict__ per_mol_offsets, mmff94_bond_t const *__restrict__ staged_bonds,
    mmff94_angle_t const *__restrict__ staged_angles, mmff94_torsion_t const *__restrict__ staged_torsions,
    mmff94_oop_t const *__restrict__ staged_oops, mmff94_stretchbend_t const *__restrict__ staged_stbns,
    mmff94_atom_interaction_t const *__restrict__ staged_interactions, mmff94_bond_t *__restrict__ packed_bonds,
    mmff94_angle_t *__restrict__ packed_angles, mmff94_torsion_t *__restrict__ packed_torsions,
    mmff94_oop_t *__restrict__ packed_oops, mmff94_stretchbend_t *__restrict__ packed_stbns,
    mmff94_atom_interaction_t *__restrict__ packed_interactions,
    mmff94_atom_offsets_t *__restrict__ atom_interaction_offsets, etkdg_molecule_metadata_t *__restrict__ metadata,
    std::uint8_t const *__restrict__ typing_status) {

    std::uint32_t const molecule = blockIdx.x;
    if (molecule >= num_molecules) return;
    std::uint32_t const lane = threadIdx.x;

    auto const counts = per_mol_counts[molecule];
    auto const bases = per_mol_offsets[molecule];
    std::uint32_t const atom_base = atom_offsets[molecule];

    copy_packed_terms_(staged_bonds + (std::size_t)molecule * max_mmff_bonds_per_mol_k, packed_bonds + bases.bonds,
                       counts.bonds, lane);
    copy_packed_terms_(staged_angles + (std::size_t)molecule * max_mmff_angles_per_mol_k, packed_angles + bases.angles,
                       counts.angles, lane);
    copy_packed_terms_(staged_torsions + (std::size_t)molecule * max_mmff_torsions_per_mol_k,
                       packed_torsions + bases.torsions, counts.torsions, lane);
    copy_packed_terms_(staged_oops + (std::size_t)molecule * max_mmff_oops_per_mol_k, packed_oops + bases.oops,
                       counts.oops, lane);
    copy_packed_terms_(staged_stbns + (std::size_t)molecule * max_mmff_stretchbends_per_mol_k,
                       packed_stbns + bases.stretchbends, counts.stretchbends, lane);
    copy_packed_terms_(staged_interactions + (std::size_t)molecule * max_mmff_interactions_per_mol_k,
                       packed_interactions + bases.interactions, counts.interactions, lane);

    // The CSR builder wrote molecule-relative starts; the consumers index one
    // batch-global interaction array, so shift every start by this molecule's base.
    // A fallback molecule never entered the CSR builder, so its slots start empty.
    bool const typed = typing_status[molecule] == 0;
    std::uint32_t const num_atoms = descriptors[molecule].num_atoms;
    for (std::uint32_t atom = lane; atom < num_atoms; atom += 32) {
        auto entry = typed ? atom_interaction_offsets[atom_base + atom] : mmff94_atom_offsets_t {};
        entry.bond_offset += bases.interactions;
        entry.angle_offset += bases.interactions;
        entry.torsion_offset += bases.interactions;
        entry.oop_offset += bases.interactions;
        entry.stretchbend_offset += bases.interactions;
        atom_interaction_offsets[atom_base + atom] = entry;
    }

    if (lane != 0) return;
    auto &entry = metadata[molecule];
    entry.num_mmff_bonds = counts.bonds;
    entry.num_mmff_angles = counts.angles;
    entry.num_mmff_torsions = counts.torsions;
    entry.num_mmff_oops = counts.oops;
    entry.num_mmff_stretchbends = counts.stretchbends;
    entry.mmff_bond_offset = bases.bonds;
    entry.mmff_angle_offset = bases.angles;
    entry.mmff_torsion_offset = bases.torsions;
    entry.mmff_oop_offset = bases.oops;
    entry.mmff_stretchbend_offset = bases.stretchbends;
    entry.mmff_atom_csr_offset = atom_base;
    entry.mmff_atom_vdw_offset = atom_base;
    entry.mmff_atom_charge_offset = atom_base;
}
#endif // __CUDACC__

#if defined(__CUDACC__)
inline void launch_mmff94_build_csr(molecule_descriptor_t const *descriptors, std::uint32_t const *atom_offsets,
                                    std::uint32_t num_molecules, mmff94_per_mol_counts_t *per_mol_counts,
                                    mmff94_bond_t const *bonds, mmff94_angle_t const *angles,
                                    mmff94_torsion_t const *torsions, mmff94_oop_t const *oops,
                                    mmff94_stretchbend_t const *stbns, mmff94_atom_interaction_t *interactions,
                                    mmff94_atom_offsets_t *atom_offsets_out, std::uint8_t const *typing_status,
                                    cudaStream_t stream = nullptr) {
    if (num_molecules == 0) return;
    mmff94_build_csr_kernel<<<num_molecules, 32, 0, stream>>>(descriptors, atom_offsets, num_molecules, per_mol_counts,
                                                              bonds, angles, torsions, oops, stbns, interactions,
                                                              atom_offsets_out, typing_status);
}

/** Host launcher for the count scan. The caller reads @p totals back to size the packed arrays. */
inline void launch_mmff94_scan_counts(mmff94_per_mol_counts_t const *per_mol_counts, std::uint32_t num_molecules,
                                      mmff94_packed_offsets_t *per_mol_offsets, mmff94_packed_offsets_t *totals,
                                      cudaStream_t stream = nullptr) {
    if (num_molecules == 0) return;
    mmff94_scan_counts_kernel<<<1, mmff94_scan_block_k, 0, stream>>>(per_mol_counts, num_molecules, per_mol_offsets,
                                                                     totals);
}

/** Host launcher for the compaction pass. */
inline void launch_mmff94_compact_terms(
    molecule_descriptor_t const *descriptors, std::uint32_t const *atom_offsets, std::uint32_t num_molecules,
    mmff94_per_mol_counts_t const *per_mol_counts, mmff94_packed_offsets_t const *per_mol_offsets,
    mmff94_bond_t const *staged_bonds, mmff94_angle_t const *staged_angles, mmff94_torsion_t const *staged_torsions,
    mmff94_oop_t const *staged_oops, mmff94_stretchbend_t const *staged_stbns,
    mmff94_atom_interaction_t const *staged_interactions, mmff94_bond_t *packed_bonds, mmff94_angle_t *packed_angles,
    mmff94_torsion_t *packed_torsions, mmff94_oop_t *packed_oops, mmff94_stretchbend_t *packed_stbns,
    mmff94_atom_interaction_t *packed_interactions, mmff94_atom_offsets_t *atom_interaction_offsets,
    etkdg_molecule_metadata_t *metadata, std::uint8_t const *typing_status, cudaStream_t stream = nullptr) {
    if (num_molecules == 0) return;
    mmff94_compact_terms_kernel<<<num_molecules, 32, 0, stream>>>(
        descriptors, atom_offsets, num_molecules, per_mol_counts, per_mol_offsets, staged_bonds, staged_angles,
        staged_torsions, staged_oops, staged_stbns, staged_interactions, packed_bonds, packed_angles, packed_torsions,
        packed_oops, packed_stbns, packed_interactions, atom_interaction_offsets, metadata, typing_status);
}
#endif

#if defined(__CUDACC__)
inline void launch_mmff94_extract_terms(molecule_descriptor_t const *descriptors, std::uint8_t const *atom_types_all,
                                        float const *atom_charges_all, std::uint32_t const *atom_offsets,
                                        std::uint32_t num_molecules, mmff94_tables_view_t tables,
                                        mmff94_atom_vdw_params_t *out_atom_vdw, float *out_atom_charge,
                                        mmff94_bond_t *out_bonds, mmff94_angle_t *out_angles,
                                        mmff94_torsion_t *out_torsions, mmff94_oop_t *out_oops,
                                        mmff94_stretchbend_t *out_stbns, mmff94_per_mol_counts_t *per_mol_counts,
                                        std::uint8_t const *typing_status, cudaStream_t stream = nullptr) {
    if (num_molecules == 0) return;
    mmff94_extract_terms_kernel<<<num_molecules, 32, 0, stream>>>(
        descriptors, atom_types_all, atom_charges_all, atom_offsets, num_molecules, tables, out_atom_vdw,
        out_atom_charge, out_bonds, out_angles, out_torsions, out_oops, out_stbns, per_mol_counts, typing_status);
}
#endif // __CUDACC__

} // namespace smallfield::mmff94
