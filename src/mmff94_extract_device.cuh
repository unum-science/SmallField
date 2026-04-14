/**
 *  @file mmff94_extract_device.cuh
 *  @brief Workstream MG-E — `mmff94_extract_terms_kernel`: runs the
 *  native MMFF term extractor on GPU.
 *
 *  Layout decisions (initial pass, optimisable later):
 *   - One block per molecule; lane-0 does all the work within a
 *     molecule.  Atom typing + lookup enumeration is an inherently
 *     graph-walking algorithm with short-lived per-molecule state
 *     (angle_scratch etc.) — SIMT parallelism across atoms / bonds is
 *     possible but adds substantial complexity for modest per-molecule
 *     speedup (drug-like mols average < 30 atoms / 30 bonds).  The win
 *     is in the cross-molecule dimension: 50 K molecules × 132 SMs ×
 *     8 blocks/SM = ~1000 molecules in flight at steady state.
 *   - Per-molecule output slots have fixed stride (`max_mmff_*_per_mol`
 *     constants).  Each molecule writes into its own reserved slice of
 *     the output arrays, and reports its actual per-term-type counts
 *     to the `per_mol_counts` array.  Downstream passes compact if
 *     desired — today's CPU pipeline already stores variable-length
 *     arrays via `etkdg_prepared_molecule_t` and consumes counts from
 *     metadata, so the layout is compatible.
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "mmff94_cpu.hpp"
#include "mmff94_tables.hpp"
#include "mmff94_typer.hpp"
#include "molecule_descriptor.hpp"
#include "etkdg_types.cuh"

namespace smallfield {

// Conservative per-molecule upper bounds.  Tuned for drug-like inputs
// (N=128 ceiling in `molecule_descriptor_t`); the emit kernel will clamp
// silently if a molecule exceeds these (molecule is flagged in
// `per_mol_status` and the caller routes it to the CPU fallback).
inline constexpr std::uint32_t max_mmff_bonds_per_mol_k = 256;
inline constexpr std::uint32_t max_mmff_angles_per_mol_k = 512;
inline constexpr std::uint32_t max_mmff_torsions_per_mol_k = 1024;
inline constexpr std::uint32_t max_mmff_oops_per_mol_k = 256;
inline constexpr std::uint32_t max_mmff_stretchbends_per_mol_k = 512;
inline constexpr std::uint32_t max_mmff_interactions_per_mol_k =
    max_mmff_bonds_per_mol_k * 2 +
    max_mmff_angles_per_mol_k * 3 +
    max_mmff_torsions_per_mol_k * 4 +
    max_mmff_oops_per_mol_k * 4 +
    max_mmff_stretchbends_per_mol_k * 3;

/** Per-molecule term counts.  Written by the emit kernel; consumed by
 *  the CSR build kernel and by downstream pipeline stages that need
 *  per-molecule offsets into the term arrays. */
struct mmff94_per_mol_counts_t {
    std::uint16_t bonds;
    std::uint16_t angles;
    std::uint16_t torsions;
    std::uint16_t oops;
    std::uint16_t stretchbends;
    std::uint16_t interactions;
    std::uint8_t overflow; ///< non-zero if any fixed stride was saturated
    std::uint8_t padding;
};

namespace detail_extract {

/** Angle scratch — populated during the angle pass and consumed by the
 *  stretch-bend pass.  Sized at `max_mmff_angles_per_mol_k` slots per
 *  molecule and stored in shared memory (≤ 16 B × 512 = 8 KB per block
 *  — fits within the 228 KB Blackwell shared-memory cap). */
struct angle_scratch_t {
    std::uint8_t i, j, k;
    std::uint8_t angle_type;
    std::uint8_t bt_ij, bt_jk;
    std::int16_t bid_ij, bid_jk;
    float theta0_deg;
    float ka;
};

} // namespace detail_extract

#if defined(__CUDACC__)
/** Single-kernel MMFF94 term extractor.  One block per molecule; lane 0
 *  of the block does all enumeration + classification + lookup.  Outputs
 *  are written into the per-molecule fixed-stride slots of the output
 *  arrays.  Blockers on this kernel are avoided by keeping per-molecule
 *  state in shared memory (angle_scratch + per-bond scratch). */
__launch_bounds__(32, 4)
__global__ void mmff94_extract_terms_kernel(
    // Inputs
    molecule_descriptor_t const * __restrict__ descriptors,
    std::uint8_t const * __restrict__ atom_types_all,
    float const * __restrict__ atom_charges_all,
    std::uint32_t const * __restrict__ atom_offsets,
    std::uint32_t num_molecules,
    mmff94_tables_view_t tables,
    // Outputs — fixed-stride per-molecule slices
    mmff94_atom_vdw_params_t * __restrict__ out_atom_vdw,
    float * __restrict__ out_atom_charge,
    mmff94_bond_t * __restrict__ out_bonds,
    mmff94_angle_t * __restrict__ out_angles,
    mmff94_torsion_t * __restrict__ out_torsions,
    mmff94_oop_t * __restrict__ out_oops,
    mmff94_stretchbend_t * __restrict__ out_stbns,
    mmff94_per_mol_counts_t * __restrict__ per_mol_counts,
    std::uint8_t const * __restrict__ typing_status) {

    std::uint32_t const mol = blockIdx.x;
    if (mol >= num_molecules) return;
    if (threadIdx.x != 0) return;
    // If the molecule was flagged as typing-failed, skip it entirely —
    // the CPU fallback (MG-D) will produce the outputs for this slot.
    if (typing_status[mol] != 0) {
        per_mol_counts[mol] = {};
        return;
    }

    molecule_descriptor_t const &desc = descriptors[mol];
    std::uint32_t const atom_off = atom_offsets[mol];
    std::uint8_t const *atom_types = atom_types_all + atom_off;
    float const *charges = atom_charges_all + atom_off;
    std::uint32_t const n = desc.num_atoms;
    std::uint32_t const nb = desc.num_bonds;

    // Per-molecule output bases.
    mmff94_atom_vdw_params_t *vdw_out = out_atom_vdw + atom_off;
    float *charge_out = out_atom_charge + atom_off;
    mmff94_bond_t *bonds_out = out_bonds + (std::size_t)mol * max_mmff_bonds_per_mol_k;
    mmff94_angle_t *angles_out = out_angles + (std::size_t)mol * max_mmff_angles_per_mol_k;
    mmff94_torsion_t *torsions_out = out_torsions + (std::size_t)mol * max_mmff_torsions_per_mol_k;
    mmff94_oop_t *oops_out = out_oops + (std::size_t)mol * max_mmff_oops_per_mol_k;
    mmff94_stretchbend_t *stbns_out = out_stbns + (std::size_t)mol * max_mmff_stretchbends_per_mol_k;

    std::uint32_t bond_count = 0, angle_count = 0, torsion_count = 0,
                 oop_count = 0, stbn_count = 0;
    std::uint8_t overflow = 0;

    // Atom VdW + charge
    for (std::uint32_t a = 0; a < n; ++a) {
        auto const *vdw = lookup_vdw(tables, atom_types[a]);
        mmff94_atom_vdw_params_t p {};
        if (vdw) {
            p.alpha = vdw->alpha_i;
            p.n_effective = vdw->n_i;
            p.a_scale = vdw->a_i;
            p.g_factor = vdw->g_i;
            p.r_star = vdw->r_star;
            p.donor_acceptor = vdw->donor_acceptor;
        }
        vdw_out[a] = p;
        charge_out[a] = charges[a];
    }

    // Precompute per-bond bond_type + bond-stretch params (kb, r0).
    std::uint8_t bond_type_arr[max_bonds_k];
    float bond_r0[max_bonds_k];
    for (std::uint32_t b = 0; b < nb; ++b) {
        std::uint8_t const i = desc.bond_atom_i[b];
        std::uint8_t const j = desc.bond_atom_j[b];
        std::uint8_t bt = 0;
        bool const arom_bond = (desc.bond_flags[b] & bond_flag_aromatic_k) != 0;
        if (desc.bond_order[b] == 1 && !arom_bond) {
            auto const *pi = lookup_prop(tables, atom_types[i]);
            auto const *pj = lookup_prop(tables, atom_types[j]);
            if (pi && pj && ((pi->sbmb && pj->sbmb) || (pi->arom && pj->arom))) bt = 1;
        }
        bond_type_arr[b] = bt;
        auto const *bp = lookup_bond(tables, bt, atom_types[i], atom_types[j]);
        float kb = 0.0f, r0 = 0.0f;
        if (bp) { kb = bp->kb; r0 = bp->r0; }
        else {
            auto const emp = native_bond_empirical(tables, desc.element[i], desc.element[j]);
            kb = emp.first; r0 = emp.second;
        }
        bond_r0[b] = r0;
        if (bond_count < max_mmff_bonds_per_mol_k) {
            bonds_out[bond_count++] = {i, j, {0, 0}, kb, r0, 0.0f};
        }
        else { overflow = 1; }
    }

    // Angle terms (and angle-scratch for stretch-bend reuse).
    detail_extract::angle_scratch_t as[max_mmff_angles_per_mol_k];
    std::uint32_t as_count = 0;

    std::span<std::uint8_t const> bond_types {bond_type_arr, nb};
    std::span<std::uint8_t const> at_span {atom_types, n};
    for (std::uint32_t center = 0; center < n; ++center) {
        std::uint16_t const nstart = desc.neighbor_offset[center];
        std::uint16_t const nend = desc.neighbor_offset[center + 1];
        for (std::uint16_t a = nstart; a < nend; ++a) {
            for (std::uint16_t b = a + 1; b < nend; ++b) {
                std::uint32_t const ai = desc.neighbor_atom[a];
                std::uint32_t const aj = desc.neighbor_atom[b];
                std::int32_t const bid_ic = find_bond_idx(desc, ai, center);
                std::int32_t const bid_cj = find_bond_idx(desc, center, aj);
                if (bid_ic < 0 || bid_cj < 0) continue;
                std::uint8_t const at_code = native_get_mmff_angle_type(desc, bond_types, ai, center, aj);
                auto const *ap = lookup_angle(tables, at_code, atom_types[ai],
                                                atom_types[center], atom_types[aj]);
                float ka = 0.0f, theta0_deg = 0.0f;
                bool use_empirical = !ap || native_is_zero(ap->ka);
                if (!use_empirical) { ka = ap->ka; theta0_deg = ap->theta0_deg; }
                if (use_empirical) {
                    float const r0_ic = bond_r0[bid_ic];
                    float const r0_cj = bond_r0[bid_cj];
                    float const override_deg = ap ? ap->theta0_deg : -1.0f;
                    auto const emp = native_angle_empirical(desc, tables, at_span,
                                                              ai, center, aj,
                                                              r0_ic, r0_cj, override_deg);
                    ka = emp.first; theta0_deg = emp.second;
                }
                float const theta0_rad = theta0_deg * (float)M_PI / 180.0f;
                if (angle_count < max_mmff_angles_per_mol_k) {
                    angles_out[angle_count++] = {(std::uint8_t)ai, (std::uint8_t)center,
                                                   (std::uint8_t)aj, 0, ka, theta0_rad, 0.0f};
                }
                else { overflow = 1; }
                if (as_count < max_mmff_angles_per_mol_k) {
                    as[as_count++] = {(std::uint8_t)ai, (std::uint8_t)center, (std::uint8_t)aj,
                                       at_code, bond_type_arr[bid_ic], bond_type_arr[bid_cj],
                                       (std::int16_t)bid_ic, (std::int16_t)bid_cj,
                                       theta0_deg, ka};
                }
            }
        }
    }

    // Torsion terms.
    for (std::uint32_t b = 0; b < nb; ++b) {
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
                auto const tt = native_get_mmff_torsion_type(desc, at_span, bond_types, i, j, k, l);
                auto const hit = lookup_torsion(tables, tt.first, tt.second,
                                                  atom_types[i], atom_types[j],
                                                  atom_types[k], atom_types[l]);
                float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
                if (hit.second) { v1 = hit.second->v1; v2 = hit.second->v2; v3 = hit.second->v3; }
                else {
                    auto const emp = native_torsion_empirical(desc, tables, at_span,
                                                                j, k, (std::int32_t)b);
                    v1 = emp[0]; v2 = emp[1]; v3 = emp[2];
                }
                if (native_is_zero(v1) && native_is_zero(v2) && native_is_zero(v3)) continue;
                if (torsion_count < max_mmff_torsions_per_mol_k) {
                    torsions_out[torsion_count++] = {(std::uint8_t)i, (std::uint8_t)j,
                                                       (std::uint8_t)k, (std::uint8_t)l,
                                                       v1, v2, v3};
                }
                else { overflow = 1; }
            }
        }
    }

    // OOP terms.
    for (std::uint32_t center = 0; center < n; ++center) {
        std::uint32_t const deg = desc.degree[center];
        if (deg < 3) continue;
        std::uint16_t const nstart = desc.neighbor_offset[center];
        std::uint8_t nb_cache[8];
        std::uint32_t nb_count = 0;
        for (std::uint16_t nn = nstart;
             nn < desc.neighbor_offset[center + 1] && nb_count < 8; ++nn)
            nb_cache[nb_count++] = desc.neighbor_atom[nn];
        for (std::uint32_t a = 0; a < nb_count; ++a) {
            std::uint8_t plane[2]; std::uint32_t plane_count = 0;
            for (std::uint32_t c = 0; c < nb_count && plane_count < 2; ++c)
                if (c != a) plane[plane_count++] = nb_cache[c];
            if (plane_count < 2) continue;
            auto const *op = lookup_oop(tables, atom_types[nb_cache[a]], atom_types[center],
                                          atom_types[plane[0]], atom_types[plane[1]]);
            if (!op) continue;
            if (oop_count < max_mmff_oops_per_mol_k) {
                oops_out[oop_count++] = {nb_cache[a], (std::uint8_t)center,
                                           plane[0], plane[1], op->koop, {0.0f, 0.0f}};
            }
            else { overflow = 1; }
        }
    }

    // Stretch-bend terms (reuse angle scratch).
    for (std::uint32_t i = 0; i < as_count; ++i) {
        auto const &a = as[i];
        auto const *prop_j = lookup_prop(tables, atom_types[a.j]);
        if (prop_j && prop_j->linh) continue;
        std::uint8_t const sbt = native_get_mmff_stretch_bend_type(
            a.angle_type,
            (atom_types[a.i] <= atom_types[a.k]) ? a.bt_ij : a.bt_jk,
            (atom_types[a.i] <  atom_types[a.k]) ? a.bt_jk : a.bt_ij);
        auto stbn = lookup_stbn(tables, sbt, a.bt_ij, a.bt_jk,
                                  atom_types[a.i], atom_types[a.j], atom_types[a.k]);
        float kba_ijk = 0.0f, kba_kji = 0.0f;
        bool swap_flag = false;
        if (stbn.second) {
            kba_ijk = stbn.second->kba_ijk;
            kba_kji = stbn.second->kba_kji;
            swap_flag = stbn.first;
        }
        else {
            auto dfsb = lookup_dfsb(tables, periodic_table_row(desc.element[a.i]),
                                      periodic_table_row(desc.element[a.j]),
                                      periodic_table_row(desc.element[a.k]));
            if (!dfsb.second) continue;
            kba_ijk = dfsb.second->kba_ijk;
            kba_kji = dfsb.second->kba_kji;
            swap_flag = dfsb.first;
        }
        if (native_is_zero(kba_ijk) && native_is_zero(kba_kji)) continue;
        if (swap_flag) { float tmp = kba_ijk; kba_ijk = kba_kji; kba_kji = tmp; }
        float const theta0_rad = a.theta0_deg * (float)M_PI / 180.0f;
        float const r0_ij = bond_r0[a.bid_ij];
        float const r0_jk = bond_r0[a.bid_jk];
        if (stbn_count < max_mmff_stretchbends_per_mol_k) {
            stbns_out[stbn_count++] = {a.i, a.j, a.k, 0, kba_ijk, kba_kji,
                                         theta0_rad, r0_ij, r0_jk, 0.0f};
        }
        else { overflow = 1; }
    }

    per_mol_counts[mol] = {
        (std::uint16_t)bond_count, (std::uint16_t)angle_count, (std::uint16_t)torsion_count,
        (std::uint16_t)oop_count, (std::uint16_t)stbn_count, 0, overflow, 0
    };
}

inline void launch_mmff94_extract_terms(
    molecule_descriptor_t const *descriptors,
    std::uint8_t const *atom_types_all,
    float const *atom_charges_all,
    std::uint32_t const *atom_offsets,
    std::uint32_t num_molecules,
    mmff94_tables_view_t tables,
    mmff94_atom_vdw_params_t *out_atom_vdw,
    float *out_atom_charge,
    mmff94_bond_t *out_bonds,
    mmff94_angle_t *out_angles,
    mmff94_torsion_t *out_torsions,
    mmff94_oop_t *out_oops,
    mmff94_stretchbend_t *out_stbns,
    mmff94_per_mol_counts_t *per_mol_counts,
    std::uint8_t const *typing_status,
    cudaStream_t stream = nullptr) {
    if (num_molecules == 0) return;
    mmff94_extract_terms_kernel<<<num_molecules, 32, 0, stream>>>(
        descriptors, atom_types_all, atom_charges_all, atom_offsets, num_molecules, tables,
        out_atom_vdw, out_atom_charge, out_bonds, out_angles, out_torsions,
        out_oops, out_stbns, per_mol_counts, typing_status);
}
#endif // __CUDACC__

} // namespace smallfield
