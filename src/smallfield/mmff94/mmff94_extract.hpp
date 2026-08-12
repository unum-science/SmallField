/**
 *  @file src/smallfield/mmff94/mmff94_extract.hpp
 *  @brief Native CPU MMFF94 term extraction.
 *
 *  Host-only, because it reaches `etkdg_prepare_batch.hpp`. The classification and empirical rules it
 *  shares with the device extractor live in `mmff94_term_equations.hpp`.
 */

#pragma once

#include "smallfield/mmff94/mmff94_term_equations.hpp"
#include "smallfield/preprocess/etkdg_prepare_batch.hpp"
#include "smallfield/types/views.hpp"
#include "smallfield/utilities.hpp"

namespace smallfield::mmff94 {
using smallfield::parameters::mmff94_tables_view_t;
using smallfield::preprocess::etkdg_prepared_molecule_t;
using smallfield::types::atom_role_first_k;
using smallfield::types::atom_role_fourth_k;
using smallfield::types::atom_role_second_k;
using smallfield::types::atom_role_third_k;
using smallfield::types::max_atoms_per_molecule_k;
using smallfield::types::max_bonds_k;
using smallfield::types::max_mmff_angles_per_mol_k;
using smallfield::types::mmff94_angle_center_k;
using smallfield::types::mmff94_angle_end_k;
using smallfield::types::mmff94_atom_vdw_params_t;
using smallfield::types::mmff94_bond_k;
using smallfield::types::mmff94_out_of_plane_k;
using smallfield::types::mmff94_stretch_bend_k;
using smallfield::types::mmff94_torsion_k;
using smallfield::types::molecule_descriptor_t;

inline void extract_mmff94_params_native(molecule_descriptor_t const &descriptor,
                                         custd::span<std::uint8_t const> atom_types, custd::span<float const> charges,
                                         mmff94_tables_view_t const &tables, etkdg_prepared_molecule_t &result) {
    result.mmff_atom_vdw.clear();
    result.mmff_atom_charge.clear();
    result.mmff_bonds.clear();
    result.mmff_angles.clear();
    result.mmff_torsions.clear();
    result.mmff_oops.clear();
    result.mmff_stretchbends.clear();
    result.mmff_interactions.clear();
    result.mmff_atom_offsets.clear();

    std::uint32_t const num_atoms = descriptor.num_atoms;
    std::uint32_t const num_bonds = descriptor.num_bonds;

    result.mmff_atom_vdw.resize(num_atoms);
    result.mmff_atom_charge.resize(num_atoms);
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        auto const *vdw_entry = lookup_vdw(tables, atom_types[atom_index]);
        mmff94_atom_vdw_params_t params {};
        if (vdw_entry) {
            params.alpha = vdw_entry->alpha_i;
            params.n_effective = vdw_entry->n_effective;
            params.a_scale = vdw_entry->a_scale;
            params.g_factor = vdw_entry->g_factor;
            params.r_star = vdw_entry->r_star;
            params.donor_acceptor = vdw_entry->donor_acceptor;
        }
        result.mmff_atom_vdw[atom_index] = params;
        result.mmff_atom_charge[atom_index] = charges[atom_index];
    }

    std::array<std::uint8_t, max_bonds_k> bond_type_arr {};
    for (std::uint32_t bond = 0; bond < num_bonds; ++bond) {
        bond_type_arr[bond] = native_get_mmff_bond_type(descriptor, tables, atom_types, bond);
    }
    custd::span<std::uint8_t const> bond_types {bond_type_arr.data(), num_bonds};

    std::array<float, max_bonds_k> bond_r0 {};
    for (std::uint32_t bond = 0; bond < num_bonds; ++bond) {
        std::uint8_t const atom_i = descriptor.bond_atom_i[bond];
        std::uint8_t const atom_j = descriptor.bond_atom_j[bond];
        std::uint8_t const bond_type = bond_type_arr[bond];
        auto const *bond_parameters = lookup_bond(tables, bond_type, atom_types[atom_i], atom_types[atom_j]);
        float force_constant = 0.0f, equilibrium_distance = 0.0f;
        if (bond_parameters) {
            force_constant = bond_parameters->force_constant;
            equilibrium_distance = bond_parameters->equilibrium_distance;
        }
        else {
            auto const empirical_bond = native_bond_empirical(tables, descriptor.element[atom_i],
                                                              descriptor.element[atom_j]);
            force_constant = empirical_bond.first;
            equilibrium_distance = empirical_bond.second;
        }
        bond_r0[bond] = equilibrium_distance;
        result.mmff_bonds.push_back({.atom_i = static_cast<std::uint8_t>(atom_i),
                                     .atom_j = static_cast<std::uint8_t>(atom_j),
                                     .padding = {0, 0},
                                     .force_constant = force_constant,
                                     .equilibrium_distance = equilibrium_distance,
                                     .padding_float = 0.0f});
    }

    struct angle_scratch_t {
        std::uint8_t atom_i, atom_j, atom_k;
        std::uint8_t angle_type;
        std::uint8_t bond_type_ij, bond_type_kj;
        std::int32_t bond_index_ij, bond_index_kj;
        float theta0_deg;
    };
    std::array<angle_scratch_t, max_mmff_angles_per_mol_k> angles_scratch;
    std::uint32_t num_angles_scratch = 0;

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
                std::uint8_t const angle_type = native_get_mmff_angle_type(descriptor, bond_types, atom_i, atom_j,
                                                                           atom_k);
                auto const *angle_parameters = lookup_angle(tables, angle_type, atom_types[atom_i], atom_types[atom_j],
                                                            atom_types[atom_k]);
                float force_constant = 0.0f, theta0_deg = 0.0f;
                bool const use_empirical = !angle_parameters || native_is_zero(angle_parameters->force_constant);
                if (!use_empirical) {
                    force_constant = angle_parameters->force_constant;
                    theta0_deg = angle_parameters->theta0_deg;
                }
                else {
                    float const r0_ij = bond_r0[bond_index_ij];
                    float const r0_kj = bond_r0[bond_index_kj];
                    float const theta_override = angle_parameters ? angle_parameters->theta0_deg : -1.0f;
                    auto const empirical_angle = native_angle_empirical(descriptor, tables, atom_types, atom_i, atom_j,
                                                                        atom_k, r0_ij, r0_kj, theta_override);
                    force_constant = empirical_angle.first;
                    theta0_deg = empirical_angle.second;
                }
                float const theta0_rad = theta0_deg * static_cast<float>(M_PI) / 180.0f;
                auto const *prop_j = lookup_prop(tables, atom_types[atom_j]);
                std::uint8_t const is_linear = (prop_j && prop_j->linear) ? 1u : 0u;
                result.mmff_angles.push_back({.atom_i = static_cast<std::uint8_t>(atom_i),
                                              .atom_j = static_cast<std::uint8_t>(atom_j),
                                              .atom_k = static_cast<std::uint8_t>(atom_k),
                                              .is_linear = is_linear,
                                              .force_constant = force_constant,
                                              .equilibrium_angle = theta0_rad,
                                              .padding_float = 0.0f});
                angles_scratch[num_angles_scratch++] = {.atom_i = static_cast<std::uint8_t>(atom_i),
                                                        .atom_j = static_cast<std::uint8_t>(atom_j),
                                                        .atom_k = static_cast<std::uint8_t>(atom_k),
                                                        .angle_type = angle_type,
                                                        .bond_type_ij = bond_types[bond_index_ij],
                                                        .bond_type_kj = bond_types[bond_index_kj],
                                                        .bond_index_ij = bond_index_ij,
                                                        .bond_index_kj = bond_index_kj,
                                                        .theta0_deg = theta0_deg};
            }
        }
    }

    for (std::uint32_t bond = 0; bond < num_bonds; ++bond) {
        std::uint32_t const atom_j = descriptor.bond_atom_i[bond];
        std::uint32_t const atom_k = descriptor.bond_atom_j[bond];
        std::uint16_t const jstart = descriptor.neighbor_offset[atom_j];
        std::uint16_t const jend = descriptor.neighbor_offset[atom_j + 1];
        std::uint16_t const kstart = descriptor.neighbor_offset[atom_k];
        std::uint16_t const kend = descriptor.neighbor_offset[atom_k + 1];
        for (std::uint16_t neighbor_slot_j = jstart; neighbor_slot_j < jend; ++neighbor_slot_j) {
            std::uint32_t const atom_i = descriptor.neighbor_atom[neighbor_slot_j];
            if (atom_i == atom_k) continue;
            for (std::uint16_t neighbor_slot_k = kstart; neighbor_slot_k < kend; ++neighbor_slot_k) {
                std::uint32_t const atom_l = descriptor.neighbor_atom[neighbor_slot_k];
                if (atom_l == atom_j) continue;
                auto const [torsion_type, torsion_type_secondary] = native_get_mmff_torsion_type(
                    descriptor, atom_types, bond_types, atom_i, atom_j, atom_k, atom_l);
                auto const lookup = lookup_torsion(tables, torsion_type, torsion_type_secondary, atom_types[atom_i],
                                                   atom_types[atom_j], atom_types[atom_k], atom_types[atom_l]);
                float torsion_v1 = 0.0f, torsion_v2 = 0.0f, torsion_v3 = 0.0f;
                if (lookup.second) {
                    torsion_v1 = lookup.second->torsion_v1;
                    torsion_v2 = lookup.second->torsion_v2;
                    torsion_v3 = lookup.second->torsion_v3;
                }
                else {
                    auto const empirical_torsion = native_torsion_empirical(descriptor, tables, atom_types, atom_j,
                                                                            atom_k, static_cast<std::int32_t>(bond));
                    torsion_v1 = empirical_torsion[0];
                    torsion_v2 = empirical_torsion[1];
                    torsion_v3 = empirical_torsion[2];
                }
                bool const all_zero = native_is_zero(torsion_v1) && native_is_zero(torsion_v2) &&
                                      native_is_zero(torsion_v3);
                if (all_zero) continue;
                result.mmff_torsions.push_back({.atom_i = static_cast<std::uint8_t>(atom_i),
                                                .atom_j = static_cast<std::uint8_t>(atom_j),
                                                .atom_k = static_cast<std::uint8_t>(atom_k),
                                                .atom_l = static_cast<std::uint8_t>(atom_l),
                                                .torsion_v1 = torsion_v1,
                                                .torsion_v2 = torsion_v2,
                                                .torsion_v3 = torsion_v3});
            }
        }
    }

    for (std::uint32_t atom_j = 0; atom_j < num_atoms; ++atom_j) {
        std::uint32_t const degree = descriptor.degree[atom_j];
        if (degree < 3) continue;
        std::uint16_t const nstart = descriptor.neighbor_offset[atom_j];
        std::uint8_t neighbor_cache[8];
        std::uint32_t neighbor_count = 0;
        for (std::uint16_t neighbor_slot = nstart;
             neighbor_slot < descriptor.neighbor_offset[atom_j + 1] && neighbor_count < 8; ++neighbor_slot)
            neighbor_cache[neighbor_count++] = descriptor.neighbor_atom[neighbor_slot];

        for (std::uint32_t out_of_plane_index = 0; out_of_plane_index < neighbor_count; ++out_of_plane_index) {
            std::uint8_t plane[2];
            std::uint32_t plane_count = 0;
            for (std::uint32_t plane_candidate_index = 0; plane_candidate_index < neighbor_count && plane_count < 2;
                 ++plane_candidate_index)
                if (plane_candidate_index != out_of_plane_index)
                    plane[plane_count++] = neighbor_cache[plane_candidate_index];
            if (plane_count < 2) continue;
            auto const *out_of_plane_parameters = lookup_oop(tables, atom_types[neighbor_cache[out_of_plane_index]],
                                                             atom_types[atom_j], atom_types[plane[0]],
                                                             atom_types[plane[1]]);
            if (!out_of_plane_parameters) continue;
            result.mmff_oops.push_back({.atom_i = neighbor_cache[out_of_plane_index],
                                        .atom_j = static_cast<std::uint8_t>(atom_j),
                                        .atom_k = plane[0],
                                        .atom_l = plane[1],
                                        .force_constant = out_of_plane_parameters->force_constant,
                                        .padding = {0.0f, 0.0f}});
        }
    }

    for (std::uint32_t angle_index = 0; angle_index < num_angles_scratch; ++angle_index) {
        auto const &scratch_angle = angles_scratch[angle_index];
        auto const *prop_j = lookup_prop(tables, atom_types[scratch_angle.atom_j]);
        if (prop_j && prop_j->linear) continue;

        // `<=` versus `<` is verbatim RDKit `getMMFFStretchBendType`: equal atom types pick the same bond type twice.
        std::uint8_t const stretch_bend_type = native_get_mmff_stretch_bend_type(
            scratch_angle.angle_type,
            (atom_types[scratch_angle.atom_i] <= atom_types[scratch_angle.atom_k]) ? scratch_angle.bond_type_ij
                                                                                   : scratch_angle.bond_type_kj,
            (atom_types[scratch_angle.atom_i] < atom_types[scratch_angle.atom_k]) ? scratch_angle.bond_type_kj
                                                                                  : scratch_angle.bond_type_ij);
        auto const [stretch_bend_swap, stretch_bend_entry] = lookup_stbn(
            tables, stretch_bend_type, scratch_angle.bond_type_ij, scratch_angle.bond_type_kj,
            atom_types[scratch_angle.atom_i], atom_types[scratch_angle.atom_j], atom_types[scratch_angle.atom_k]);
        float kba_ijk = 0.0f, kba_kji = 0.0f;
        bool swap_flag = false;
        if (stretch_bend_entry) {
            kba_ijk = stretch_bend_entry->kba_ijk;
            kba_kji = stretch_bend_entry->kba_kji;
            swap_flag = stretch_bend_swap;
        }
        else {
            auto const [default_stretch_bend_swap, default_stretch_bend_entry] = lookup_dfsb(
                tables, periodic_table_row(descriptor.element[scratch_angle.atom_i]),
                periodic_table_row(descriptor.element[scratch_angle.atom_j]),
                periodic_table_row(descriptor.element[scratch_angle.atom_k]));
            if (!default_stretch_bend_entry) continue;
            kba_ijk = default_stretch_bend_entry->kba_ijk;
            kba_kji = default_stretch_bend_entry->kba_kji;
            swap_flag = default_stretch_bend_swap;
        }
        if (native_is_zero(kba_ijk) && native_is_zero(kba_kji)) continue;
        if (swap_flag) swap_(kba_ijk, kba_kji);
        float const theta0_rad = scratch_angle.theta0_deg * static_cast<float>(M_PI) / 180.0f;
        float const r0_ij = bond_r0[scratch_angle.bond_index_ij];
        float const r0_kj = bond_r0[scratch_angle.bond_index_kj];
        result.mmff_stretchbends.push_back({.atom_i = scratch_angle.atom_i,
                                            .atom_j = scratch_angle.atom_j,
                                            .atom_k = scratch_angle.atom_k,
                                            .padding = 0,
                                            .kba_ijk = kba_ijk,
                                            .kba_kji = kba_kji,
                                            .equilibrium_angle = theta0_rad,
                                            .equilibrium_distance_ij = r0_ij,
                                            .equilibrium_distance_kj = r0_kj,
                                            .padding_float = 0.0f});
    }

    auto const &bonds = result.mmff_bonds;
    auto const &ang_v = result.mmff_angles;
    auto const &tor_v = result.mmff_torsions;
    auto const &oop_v = result.mmff_oops;
    auto const &sb_v = result.mmff_stretchbends;

    struct type_counts_t {
        std::uint8_t bond, angle, torsion, oop, stretchbend;
    };
    std::array<type_counts_t, max_atoms_per_molecule_k> type_counts {};
    for (auto const &bond : bonds) {
        type_counts[bond.atom_i].bond++;
        type_counts[bond.atom_j].bond++;
    }
    for (auto const &angle : ang_v) {
        type_counts[angle.atom_i].angle++;
        type_counts[angle.atom_j].angle++;
        type_counts[angle.atom_k].angle++;
    }
    for (auto const &torsion : tor_v) {
        type_counts[torsion.atom_i].torsion++;
        type_counts[torsion.atom_j].torsion++;
        type_counts[torsion.atom_k].torsion++;
        type_counts[torsion.atom_l].torsion++;
    }
    for (auto const &oop_term : oop_v) {
        type_counts[oop_term.atom_i].oop++;
        type_counts[oop_term.atom_j].oop++;
        type_counts[oop_term.atom_k].oop++;
        type_counts[oop_term.atom_l].oop++;
    }
    for (auto const &stretch_bend : sb_v) {
        type_counts[stretch_bend.atom_i].stretchbend++;
        type_counts[stretch_bend.atom_j].stretchbend++;
        type_counts[stretch_bend.atom_k].stretchbend++;
    }

    result.mmff_atom_offsets.resize(num_atoms);
    std::uint32_t running = 0;
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        auto const &counts = type_counts[atom_index];
        auto &atom_offsets = result.mmff_atom_offsets[atom_index];
        atom_offsets.bond_offset = running;
        atom_offsets.bond_count = counts.bond;
        running += counts.bond;
        atom_offsets.angle_offset = running;
        atom_offsets.angle_count = counts.angle;
        running += counts.angle;
        atom_offsets.torsion_offset = running;
        atom_offsets.torsion_count = counts.torsion;
        running += counts.torsion;
        atom_offsets.oop_offset = running;
        atom_offsets.oop_count = counts.oop;
        running += counts.oop;
        atom_offsets.stretchbend_offset = running;
        atom_offsets.stretchbend_count = counts.stretchbend;
        running += counts.stretchbend;
    }
    result.mmff_interactions.resize(running);

    struct cursors_t {
        std::uint32_t bond, angle, torsion, oop, stretchbend;
    };
    std::array<cursors_t, max_atoms_per_molecule_k> cursors {};
    for (std::uint32_t atom_index = 0; atom_index < num_atoms; ++atom_index) {
        auto const &atom_offsets = result.mmff_atom_offsets[atom_index];
        cursors[atom_index] = {.bond = atom_offsets.bond_offset,
                               .angle = atom_offsets.angle_offset,
                               .torsion = atom_offsets.torsion_offset,
                               .oop = atom_offsets.oop_offset,
                               .stretchbend = atom_offsets.stretchbend_offset};
    }
    for (std::uint32_t bond_index = 0; bond_index < bonds.size(); ++bond_index) {
        result.mmff_interactions[cursors[bonds[bond_index].atom_i].bond++] = {.term_index = bond_index,
                                                                              .interaction_type = mmff94_bond_k,
                                                                              .role = atom_role_first_k};
        result.mmff_interactions[cursors[bonds[bond_index].atom_j].bond++] = {.term_index = bond_index,
                                                                              .interaction_type = mmff94_bond_k,
                                                                              .role = atom_role_second_k};
    }
    for (std::uint32_t angle_index = 0; angle_index < ang_v.size(); ++angle_index) {
        result.mmff_interactions[cursors[ang_v[angle_index].atom_i].angle++] = {.term_index = angle_index,
                                                                                .interaction_type = mmff94_angle_end_k,
                                                                                .role = atom_role_first_k};
        result.mmff_interactions[cursors[ang_v[angle_index].atom_j].angle++] = {
            .term_index = angle_index,
            .interaction_type = mmff94_angle_center_k,
            .role = atom_role_second_k};
        result.mmff_interactions[cursors[ang_v[angle_index].atom_k].angle++] = {.term_index = angle_index,
                                                                                .interaction_type = mmff94_angle_end_k,
                                                                                .role = atom_role_third_k};
    }
    for (std::uint32_t torsion_index = 0; torsion_index < tor_v.size(); ++torsion_index) {
        result.mmff_interactions[cursors[tor_v[torsion_index].atom_i].torsion++] = {
            .term_index = torsion_index,
            .interaction_type = mmff94_torsion_k,
            .role = atom_role_first_k};
        result.mmff_interactions[cursors[tor_v[torsion_index].atom_j].torsion++] = {
            .term_index = torsion_index,
            .interaction_type = mmff94_torsion_k,
            .role = atom_role_second_k};
        result.mmff_interactions[cursors[tor_v[torsion_index].atom_k].torsion++] = {
            .term_index = torsion_index,
            .interaction_type = mmff94_torsion_k,
            .role = atom_role_third_k};
        result.mmff_interactions[cursors[tor_v[torsion_index].atom_l].torsion++] = {
            .term_index = torsion_index,
            .interaction_type = mmff94_torsion_k,
            .role = atom_role_fourth_k};
    }
    for (std::uint32_t oop_index = 0; oop_index < oop_v.size(); ++oop_index) {
        result.mmff_interactions[cursors[oop_v[oop_index].atom_i].oop++] = {.term_index = oop_index,
                                                                            .interaction_type = mmff94_out_of_plane_k,
                                                                            .role = atom_role_first_k};
        result.mmff_interactions[cursors[oop_v[oop_index].atom_j].oop++] = {.term_index = oop_index,
                                                                            .interaction_type = mmff94_out_of_plane_k,
                                                                            .role = atom_role_second_k};
        result.mmff_interactions[cursors[oop_v[oop_index].atom_k].oop++] = {.term_index = oop_index,
                                                                            .interaction_type = mmff94_out_of_plane_k,
                                                                            .role = atom_role_third_k};
        result.mmff_interactions[cursors[oop_v[oop_index].atom_l].oop++] = {.term_index = oop_index,
                                                                            .interaction_type = mmff94_out_of_plane_k,
                                                                            .role = atom_role_fourth_k};
    }
    for (std::uint32_t stretch_bend_index = 0; stretch_bend_index < sb_v.size(); ++stretch_bend_index) {
        result.mmff_interactions[cursors[sb_v[stretch_bend_index].atom_i].stretchbend++] = {
            .term_index = stretch_bend_index,
            .interaction_type = mmff94_stretch_bend_k,
            .role = atom_role_first_k};
        result.mmff_interactions[cursors[sb_v[stretch_bend_index].atom_j].stretchbend++] = {
            .term_index = stretch_bend_index,
            .interaction_type = mmff94_stretch_bend_k,
            .role = atom_role_second_k};
        result.mmff_interactions[cursors[sb_v[stretch_bend_index].atom_k].stretchbend++] = {
            .term_index = stretch_bend_index,
            .interaction_type = mmff94_stretch_bend_k,
            .role = atom_role_third_k};
    }
}

inline bool extract_mmff94_params_fully_native(molecule_descriptor_t const &descriptor,
                                               mmff94_tables_view_t const &tables, etkdg_prepared_molecule_t &result) {
    std::uint32_t const num_atoms = descriptor.num_atoms;
    std::array<std::uint8_t, max_atoms_per_molecule_k> atom_types_buf {};
    std::array<float, max_atoms_per_molecule_k> formal_charges_buf {};
    std::array<float, max_atoms_per_molecule_k> partial_charges_buf {};
    custd::span<std::uint8_t> atom_types {atom_types_buf.data(), num_atoms};
    custd::span<float> formal_charges {formal_charges_buf.data(), num_atoms};
    custd::span<float> partial_charges {partial_charges_buf.data(), num_atoms};

    if (!native_compute_mmff_atom_types(descriptor, atom_types)) return false;
    native_compute_mmff_formal_charges(descriptor, atom_types, formal_charges);
    native_compute_mmff_partial_charges(descriptor, tables, atom_types, formal_charges, partial_charges);

    extract_mmff94_params_native(descriptor, atom_types, partial_charges, tables, result);
    return true;
}

} // namespace smallfield::mmff94
