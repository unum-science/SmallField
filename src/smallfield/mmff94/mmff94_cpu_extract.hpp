/**
 *  @file mmff94_cpu_extract.hpp
 *  @brief Native CPU MMFF94 term extraction (host-only).
 *
 *  Separated from mmff94_cpu.hpp so that CUDA translation units can
 *  include the helpers (periodic-table rows, ring queries, empirical
 *  rules) without pulling in etkdg_prepare_batch.hpp.
 */

#pragma once

#include "smallfield/mmff94/mmff94_cpu.hpp"
#include "smallfield/preprocess/etkdg_prepare_batch.hpp"

namespace smallfield::mmff94 {
using smallfield::parameters::mmff94_tables_view_t;
using smallfield::preprocess::etkdg_prepared_molecule_t;
using smallfield::types::atom_role_first_k;
using smallfield::types::atom_role_fourth_k;
using smallfield::types::atom_role_second_k;
using smallfield::types::atom_role_third_k;
using smallfield::types::max_atoms_k;
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

inline void extract_mmff94_params_native(molecule_descriptor_t const &desc, std::span<std::uint8_t const> atom_types,
                                         std::span<float const> charges, mmff94_tables_view_t const &tables,
                                         etkdg_prepared_molecule_t &result) {
    result.mmff_atom_vdw.clear();
    result.mmff_atom_charge.clear();
    result.bond_distance.clear();
    result.mmff_bonds.clear();
    result.mmff_angles.clear();
    result.mmff_torsions.clear();
    result.mmff_oops.clear();
    result.mmff_stretchbends.clear();
    result.mmff_interactions.clear();
    result.mmff_atom_offsets.clear();

    std::uint32_t const num_atoms = desc.num_atoms;
    std::uint32_t const num_bonds = desc.num_bonds;

    result.mmff_atom_vdw.resize(num_atoms);
    result.mmff_atom_charge.resize(num_atoms);
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        auto const *vdw = lookup_vdw(tables, atom_types[a]);
        mmff94_atom_vdw_params_t params {};
        if (vdw) {
            params.alpha = vdw->alpha_i;
            params.n_effective = vdw->n_i;
            params.a_scale = vdw->a_i;
            params.g_factor = vdw->g_i;
            params.r_star = vdw->r_star;
            params.donor_acceptor = vdw->donor_acceptor;
        }
        result.mmff_atom_vdw[a] = params;
        result.mmff_atom_charge[a] = charges[a];
    }

    result.bond_distance.resize(static_cast<std::size_t>(num_atoms) * num_atoms);
    for (std::uint32_t i = 0; i < num_atoms; ++i)
        for (std::uint32_t j = 0; j < num_atoms; ++j)
            result.bond_distance[i * num_atoms + j] = desc.bond_distance[i * max_atoms_k + j];

    std::array<std::uint8_t, max_bonds_k> bond_type_arr {};
    for (std::uint32_t b = 0; b < num_bonds; ++b) {
        bond_type_arr[b] = native_get_mmff_bond_type(desc, tables, atom_types, b);
    }
    std::span<std::uint8_t const> bond_types {bond_type_arr.data(), num_bonds};

    std::array<float, max_bonds_k> bond_r0 {};
    std::array<float, max_bonds_k> bond_kb {};
    for (std::uint32_t b = 0; b < num_bonds; ++b) {
        std::uint8_t const i = desc.bond_atom_i[b];
        std::uint8_t const j = desc.bond_atom_j[b];
        std::uint8_t const bt = bond_type_arr[b];
        auto const *bp = lookup_bond(tables, bt, atom_types[i], atom_types[j]);
        float kb = 0.0f, r0 = 0.0f;
        if (bp) {
            kb = bp->kb;
            r0 = bp->r0;
        }
        else {
            auto const emp = native_bond_empirical(tables, desc.element[i], desc.element[j]);
            kb = emp.first;
            r0 = emp.second;
        }
        bond_r0[b] = r0;
        bond_kb[b] = kb;
        result.mmff_bonds.push_back({static_cast<std::uint8_t>(i), static_cast<std::uint8_t>(j), {0, 0}, kb, r0, 0.0f});
    }

    struct angle_scratch_t {
        std::uint8_t i, j, k;
        std::uint8_t angle_type;
        std::uint8_t bond_type_ij, bond_type_jk;
        std::int32_t bond_index_ij, bond_index_jk;
        float theta0_deg;
        float ka;
    };
    std::array<angle_scratch_t, max_mmff_angles_per_mol_k> angles_scratch;
    std::uint32_t num_angles_scratch = 0;

    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint16_t const nstart = desc.neighbor_offset[center];
        std::uint16_t const nend = desc.neighbor_offset[center + 1];
        for (std::uint16_t a = nstart; a < nend; ++a) {
            for (std::uint16_t b = a + 1; b < nend; ++b) {
                std::uint32_t const ai = desc.neighbor_atom[a];
                std::uint32_t const aj = desc.neighbor_atom[b];
                std::int32_t const bond_index_ic = find_bond_idx(desc, ai, center);
                std::int32_t const bond_index_cj = find_bond_idx(desc, center, aj);
                if (bond_index_ic < 0 || bond_index_cj < 0) continue;
                std::uint8_t const at = native_get_mmff_angle_type(desc, bond_types, ai, center, aj);
                auto const *ap = lookup_angle(tables, at, atom_types[ai], atom_types[center], atom_types[aj]);
                float ka = 0.0f, theta0_deg = 0.0f;
                bool use_empirical = (!ap || native_is_zero(ap->ka));
                if (ap && !native_is_zero(ap->ka)) {
                    ka = ap->ka;
                    theta0_deg = ap->theta0_deg;
                }
                if (use_empirical) {
                    float const r0_ic = bond_r0[bond_index_ic];
                    float const r0_cj = bond_r0[bond_index_cj];
                    float const theta_override = ap ? ap->theta0_deg : -1.0f;
                    auto const emp = native_angle_empirical(desc, tables, atom_types, ai, center, aj, r0_ic, r0_cj,
                                                            theta_override);
                    ka = emp.first;
                    theta0_deg = emp.second;
                }
                float const theta0_rad = theta0_deg * static_cast<float>(M_PI) / 180.0f;
                result.mmff_angles.push_back({static_cast<std::uint8_t>(ai), static_cast<std::uint8_t>(center),
                                              static_cast<std::uint8_t>(aj), 0, ka, theta0_rad, 0.0f});
                angles_scratch[num_angles_scratch++] = {static_cast<std::uint8_t>(ai),
                                                        static_cast<std::uint8_t>(center),
                                                        static_cast<std::uint8_t>(aj),
                                                        at,
                                                        bond_types[bond_index_ic],
                                                        bond_types[bond_index_cj],
                                                        bond_index_ic,
                                                        bond_index_cj,
                                                        theta0_deg,
                                                        ka};
            }
        }
    }

    for (std::uint32_t b = 0; b < num_bonds; ++b) {
        std::uint32_t const j = desc.bond_atom_i[b];
        std::uint32_t const k = desc.bond_atom_j[b];
        std::uint16_t const jstart = desc.neighbor_offset[j];
        std::uint16_t const jend = desc.neighbor_offset[j + 1];
        std::uint16_t const kstart = desc.neighbor_offset[k];
        std::uint16_t const kend = desc.neighbor_offset[k + 1];
        for (std::uint16_t nj = jstart; nj < jend; ++nj) {
            std::uint32_t const i = desc.neighbor_atom[nj];
            if (i == k) continue;
            for (std::uint16_t nk = kstart; nk < kend; ++nk) {
                std::uint32_t const l = desc.neighbor_atom[nk];
                if (l == j) continue;
                auto const tt = native_get_mmff_torsion_type(desc, atom_types, bond_types, i, j, k, l);
                auto const lookup = lookup_torsion(tables, tt.first, tt.second, atom_types[i], atom_types[j],
                                                   atom_types[k], atom_types[l]);
                float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
                if (lookup.second) {
                    v1 = lookup.second->v1;
                    v2 = lookup.second->v2;
                    v3 = lookup.second->v3;
                }
                else {
                    auto const emp = native_torsion_empirical(desc, tables, atom_types, j, k,
                                                              static_cast<std::int32_t>(b));
                    v1 = emp[0];
                    v2 = emp[1];
                    v3 = emp[2];
                }
                bool const all_zero = native_is_zero(v1) && native_is_zero(v2) && native_is_zero(v3);
                if (all_zero) continue;
                result.mmff_torsions.push_back({static_cast<std::uint8_t>(i), static_cast<std::uint8_t>(j),
                                                static_cast<std::uint8_t>(k), static_cast<std::uint8_t>(l), v1, v2,
                                                v3});
            }
        }
    }

    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint32_t const degree = desc.degree[center];
        if (degree < 3) continue;
        std::uint16_t const nstart = desc.neighbor_offset[center];
        std::uint8_t neighbor_cache[8];
        std::uint32_t neighbor_count = 0;
        for (std::uint16_t n = nstart; n < desc.neighbor_offset[center + 1] && neighbor_count < 8; ++n)
            neighbor_cache[neighbor_count++] = desc.neighbor_atom[n];

        for (std::uint32_t a = 0; a < neighbor_count; ++a) {
            std::uint8_t plane[2];
            std::uint32_t plane_count = 0;
            for (std::uint32_t c = 0; c < neighbor_count && plane_count < 2; ++c)
                if (c != a) plane[plane_count++] = neighbor_cache[c];
            if (plane_count < 2) continue;
            auto const *op = lookup_oop(tables, atom_types[neighbor_cache[a]], atom_types[center], atom_types[plane[0]],
                                        atom_types[plane[1]]);
            if (!op) continue;
            result.mmff_oops.push_back(
                {neighbor_cache[a], static_cast<std::uint8_t>(center), plane[0], plane[1], op->koop, {0.0f, 0.0f}});
        }
    }

    for (std::uint32_t angle_index = 0; angle_index < num_angles_scratch; ++angle_index) {
        auto const &as = angles_scratch[angle_index];
        auto const *prop_j = lookup_prop(tables, atom_types[as.j]);
        if (prop_j && prop_j->linh) continue;

        std::uint8_t const sbt = native_get_mmff_stretch_bend_type(
            as.angle_type, (atom_types[as.i] <= atom_types[as.k]) ? as.bond_type_ij : as.bond_type_jk,
            (atom_types[as.i] < atom_types[as.k]) ? as.bond_type_jk : as.bond_type_ij);
        auto stbn = lookup_stbn(tables, sbt, as.bond_type_ij, as.bond_type_jk, atom_types[as.i], atom_types[as.j],
                                atom_types[as.k]);
        float kba_ijk = 0.0f, kba_kji = 0.0f;
        bool swap_flag = false;
        if (stbn.second) {
            kba_ijk = stbn.second->kba_ijk;
            kba_kji = stbn.second->kba_kji;
            swap_flag = stbn.first;
        }
        else {
            auto dfsb = lookup_dfsb(tables, periodic_table_row(desc.element[as.i]),
                                    periodic_table_row(desc.element[as.j]), periodic_table_row(desc.element[as.k]));
            if (!dfsb.second) continue;
            kba_ijk = dfsb.second->kba_ijk;
            kba_kji = dfsb.second->kba_kji;
            swap_flag = dfsb.first;
        }
        if (native_is_zero(kba_ijk) && native_is_zero(kba_kji)) continue;
        if (swap_flag) std::swap(kba_ijk, kba_kji);
        float const theta0_rad = as.theta0_deg * static_cast<float>(M_PI) / 180.0f;
        float const r0_ij = bond_r0[as.bond_index_ij];
        float const r0_jk = bond_r0[as.bond_index_jk];
        result.mmff_stretchbends.push_back({as.i, as.j, as.k, 0, kba_ijk, kba_kji, theta0_rad, r0_ij, r0_jk, 0.0f});
    }

    auto const &bonds = result.mmff_bonds;
    auto const &ang_v = result.mmff_angles;
    auto const &tor_v = result.mmff_torsions;
    auto const &oop_v = result.mmff_oops;
    auto const &sb_v = result.mmff_stretchbends;

    struct type_counts_t {
        std::uint8_t bond, angle, torsion, oop, stretchbend;
    };
    std::array<type_counts_t, max_atoms_k> type_counts {};
    for (auto const &e : bonds) {
        type_counts[e.atom_i].bond++;
        type_counts[e.atom_j].bond++;
    }
    for (auto const &e : ang_v) {
        type_counts[e.atom_i].angle++;
        type_counts[e.atom_j].angle++;
        type_counts[e.atom_k].angle++;
    }
    for (auto const &e : tor_v) {
        type_counts[e.atom_i].torsion++;
        type_counts[e.atom_j].torsion++;
        type_counts[e.atom_k].torsion++;
        type_counts[e.atom_l].torsion++;
    }
    for (auto const &e : oop_v) {
        type_counts[e.atom_i].oop++;
        type_counts[e.atom_j].oop++;
        type_counts[e.atom_k].oop++;
        type_counts[e.atom_l].oop++;
    }
    for (auto const &e : sb_v) {
        type_counts[e.atom_i].stretchbend++;
        type_counts[e.atom_j].stretchbend++;
        type_counts[e.atom_k].stretchbend++;
    }

    result.mmff_atom_offsets.resize(num_atoms);
    std::uint32_t running = 0;
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        auto const &c = type_counts[a];
        auto &off = result.mmff_atom_offsets[a];
        off.bond_start = running;
        off.bond_count = c.bond;
        running += c.bond;
        off.angle_start = running;
        off.angle_count = c.angle;
        running += c.angle;
        off.torsion_start = running;
        off.torsion_count = c.torsion;
        running += c.torsion;
        off.oop_start = running;
        off.oop_count = c.oop;
        running += c.oop;
        off.stretchbend_start = running;
        off.stretchbend_count = c.stretchbend;
        running += c.stretchbend;
    }
    result.mmff_interactions.resize(running);

    struct cursors_t {
        std::uint32_t bond, angle, torsion, oop, stretchbend;
    };
    std::array<cursors_t, max_atoms_k> cursors {};
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        auto const &off = result.mmff_atom_offsets[a];
        cursors[a] = {off.bond_start, off.angle_start, off.torsion_start, off.oop_start, off.stretchbend_start};
    }
    for (std::uint32_t t = 0; t < bonds.size(); ++t) {
        result.mmff_interactions[cursors[bonds[t].atom_i].bond++] = {t, mmff94_bond_k, atom_role_first_k};
        result.mmff_interactions[cursors[bonds[t].atom_j].bond++] = {t, mmff94_bond_k, atom_role_second_k};
    }
    for (std::uint32_t t = 0; t < ang_v.size(); ++t) {
        result.mmff_interactions[cursors[ang_v[t].atom_i].angle++] = {t, mmff94_angle_end_k, atom_role_first_k};
        result.mmff_interactions[cursors[ang_v[t].atom_j].angle++] = {t, mmff94_angle_center_k, atom_role_second_k};
        result.mmff_interactions[cursors[ang_v[t].atom_k].angle++] = {t, mmff94_angle_end_k, atom_role_third_k};
    }
    for (std::uint32_t t = 0; t < tor_v.size(); ++t) {
        result.mmff_interactions[cursors[tor_v[t].atom_i].torsion++] = {t, mmff94_torsion_k, atom_role_first_k};
        result.mmff_interactions[cursors[tor_v[t].atom_j].torsion++] = {t, mmff94_torsion_k, atom_role_second_k};
        result.mmff_interactions[cursors[tor_v[t].atom_k].torsion++] = {t, mmff94_torsion_k, atom_role_third_k};
        result.mmff_interactions[cursors[tor_v[t].atom_l].torsion++] = {t, mmff94_torsion_k, atom_role_fourth_k};
    }
    for (std::uint32_t t = 0; t < oop_v.size(); ++t) {
        result.mmff_interactions[cursors[oop_v[t].atom_i].oop++] = {t, mmff94_out_of_plane_k, atom_role_first_k};
        result.mmff_interactions[cursors[oop_v[t].atom_j].oop++] = {t, mmff94_out_of_plane_k, atom_role_second_k};
        result.mmff_interactions[cursors[oop_v[t].atom_k].oop++] = {t, mmff94_out_of_plane_k, atom_role_third_k};
        result.mmff_interactions[cursors[oop_v[t].atom_l].oop++] = {t, mmff94_out_of_plane_k, atom_role_fourth_k};
    }
    for (std::uint32_t t = 0; t < sb_v.size(); ++t) {
        result.mmff_interactions[cursors[sb_v[t].atom_i].stretchbend++] = {t, mmff94_stretch_bend_k, atom_role_first_k};
        result.mmff_interactions[cursors[sb_v[t].atom_j].stretchbend++] = {t, mmff94_stretch_bend_k,
                                                                           atom_role_second_k};
        result.mmff_interactions[cursors[sb_v[t].atom_k].stretchbend++] = {t, mmff94_stretch_bend_k, atom_role_third_k};
    }
}

inline bool extract_mmff94_params_fully_native(molecule_descriptor_t const &desc, mmff94_tables_view_t const &tables,
                                               etkdg_prepared_molecule_t &result) {
    std::uint32_t const num_atoms = desc.num_atoms;
    std::array<std::uint8_t, max_atoms_k> atom_types_buf {};
    std::array<float, max_atoms_k> formal_charges_buf {};
    std::array<float, max_atoms_k> partial_charges_buf {};
    std::span<std::uint8_t> atom_types {atom_types_buf.data(), num_atoms};
    std::span<float> formal_charges {formal_charges_buf.data(), num_atoms};
    std::span<float> partial_charges {partial_charges_buf.data(), num_atoms};

    if (!native_compute_mmff_atom_types(desc, atom_types)) return false;
    native_compute_mmff_formal_charges(desc, atom_types, formal_charges);
    native_compute_mmff_partial_charges(desc, tables, atom_types, formal_charges, partial_charges);

    extract_mmff94_params_native(desc, atom_types, partial_charges, tables, result);
    return true;
}

} // namespace smallfield::mmff94
