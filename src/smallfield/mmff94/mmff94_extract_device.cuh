/**
 *  @file mmff94_extract_device.cuh
 *  @brief `mmff94_extract_terms_kernel` runs the native MMFF term extractor on the GPU.
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

#include "smallfield/module_fwd.hpp"

#include <cstdint>
#include <cuda_runtime.h>

#include "smallfield/mmff94/mmff94_cpu.hpp"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/mmff94/mmff94_typer.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/mmff94_bounds.hpp"

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
using smallfield::types::max_atoms_k;
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
 *  — fits comfortably within the SM shared-memory cap. */
struct angle_scratch_t {
    std::uint8_t i, j, k;
    std::uint8_t angle_type;
    std::uint8_t bond_type_ij, bond_type_jk;
    std::int16_t bond_index_ij, bond_index_jk;
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
__launch_bounds__(32, 4) __global__ void mmff94_extract_terms_kernel(
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

    std::uint32_t const mol = blockIdx.x;
    if (mol >= num_molecules) return;

    // Early-exit status check runs on all 32 lanes so they all
    // agree to fall out together (no warp divergence afterwards).
    if (typing_status[mol] != 0) {
        if (threadIdx.x == 0) per_mol_counts[mol] = {};
        return;
    }

    std::uint32_t const lane = threadIdx.x;
    molecule_descriptor_t const &desc = descriptors[mol];
    std::uint32_t const atom_off = atom_offsets[mol];
    std::uint8_t const *atom_types = atom_types_all + atom_off;
    float const *charges = atom_charges_all + atom_off;
    std::uint32_t const num_atoms = desc.num_atoms;
    std::uint32_t const num_bonds = desc.num_bonds;

    // Per-molecule output bases.
    mmff94_atom_vdw_params_t *vdw_out = out_atom_vdw + atom_off;
    float *charge_out = out_atom_charge + atom_off;
    mmff94_bond_t *bonds_out = out_bonds + (std::size_t)mol * max_mmff_bonds_per_mol_k;
    mmff94_angle_t *angles_out = out_angles + (std::size_t)mol * max_mmff_angles_per_mol_k;
    mmff94_torsion_t *torsions_out = out_torsions + (std::size_t)mol * max_mmff_torsions_per_mol_k;
    mmff94_oop_t *oops_out = out_oops + (std::size_t)mol * max_mmff_oops_per_mol_k;
    mmff94_stretchbend_t *stbns_out = out_stbns + (std::size_t)mol * max_mmff_stretchbends_per_mol_k;

    // Atom VdW + charge — lane-parallel over atoms (each atom's work
    // is independent and written to its own slot).
    for (std::uint32_t a = lane; a < num_atoms; a += 32) {
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

    // Precompute per-bond bond_type + bond-stretch params — lane-parallel
    // over bonds. Shared-memory scratch for downstream angle/stbn reuse.
    __shared__ std::uint8_t s_bond_type[max_bonds_k];
    __shared__ float s_bond_r0[max_bonds_k];
    for (std::uint32_t b = lane; b < num_bonds; b += 32) {
        std::uint8_t const i = desc.bond_atom_i[b];
        std::uint8_t const j = desc.bond_atom_j[b];
        std::uint8_t bt = 0;
        bool const arom_bond = (desc.bond_flags[b] & bond_flag_aromatic_k) != 0;
        if (desc.bond_order[b] == 1 && !arom_bond) {
            auto const *pi = lookup_prop(tables, atom_types[i]);
            auto const *pj = lookup_prop(tables, atom_types[j]);
            if (pi && pj && ((pi->sbmb && pj->sbmb) || (pi->arom && pj->arom))) bt = 1;
        }
        s_bond_type[b] = bt;
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
        s_bond_r0[b] = r0;
        // Emit bond at deterministic position = original bond index `b`
        // so output ordering matches the CPU (which also iterates bonds
        // in index order and push_backs in order).
        if (b < max_mmff_bonds_per_mol_k) { bonds_out[b] = {i, j, {0, 0}, kb, r0, 0.0f}; }
    }
    __syncwarp();

    // From here on the passes with atomic cursors must run on lane-0
    // only to preserve deterministic ordering for byte-equality with CPU.
    if (lane != 0) return;

    // Wire the shared scratch back into the rest of the kernel body via
    // the existing names — avoid renaming the remaining 200+ LOC.
    std::uint8_t const *bond_type_arr = s_bond_type;
    float const *bond_r0 = s_bond_r0;

    std::uint32_t bond_count = num_bonds < max_mmff_bonds_per_mol_k ? num_bonds : max_mmff_bonds_per_mol_k;
    std::uint32_t angle_count = 0, torsion_count = 0, oop_count = 0, stbn_count = 0;
    std::uint8_t overflow = (num_bonds > max_mmff_bonds_per_mol_k) ? 1 : 0;

    // Angle terms (and angle-scratch for stretch-bend reuse).
    detail_extract::angle_scratch_t as[max_mmff_angles_per_mol_k];
    std::uint32_t as_count = 0;

    std::span<std::uint8_t const> bond_types {bond_type_arr, num_bonds};
    std::span<std::uint8_t const> at_span {atom_types, num_atoms};
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
                std::uint8_t const at_code = native_get_mmff_angle_type(desc, bond_types, ai, center, aj);
                auto const *ap = lookup_angle(tables, at_code, atom_types[ai], atom_types[center], atom_types[aj]);
                float ka = 0.0f, theta0_deg = 0.0f;
                bool use_empirical = !ap || native_is_zero(ap->ka);
                if (!use_empirical) {
                    ka = ap->ka;
                    theta0_deg = ap->theta0_deg;
                }
                if (use_empirical) {
                    float const r0_ic = bond_r0[bond_index_ic];
                    float const r0_cj = bond_r0[bond_index_cj];
                    float const override_deg = ap ? ap->theta0_deg : -1.0f;
                    auto const emp = native_angle_empirical(desc, tables, at_span, ai, center, aj, r0_ic, r0_cj,
                                                            override_deg);
                    ka = emp.first;
                    theta0_deg = emp.second;
                }
                float const theta0_rad = theta0_deg * (float)M_PI / 180.0f;
                if (angle_count < max_mmff_angles_per_mol_k) {
                    angles_out[angle_count++] = {
                        (std::uint8_t)ai, (std::uint8_t)center, (std::uint8_t)aj, 0, ka, theta0_rad, 0.0f};
                }
                else { overflow = 1; }
                if (as_count < max_mmff_angles_per_mol_k) {
                    as[as_count++] = {(std::uint8_t)ai,
                                      (std::uint8_t)center,
                                      (std::uint8_t)aj,
                                      at_code,
                                      bond_type_arr[bond_index_ic],
                                      bond_type_arr[bond_index_cj],
                                      (std::int16_t)bond_index_ic,
                                      (std::int16_t)bond_index_cj,
                                      theta0_deg,
                                      ka};
                }
            }
        }
    }

    // Torsion terms.
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
                auto const tt = native_get_mmff_torsion_type(desc, at_span, bond_types, i, j, k, l);
                auto const hit = lookup_torsion(tables, tt.first, tt.second, atom_types[i], atom_types[j],
                                                atom_types[k], atom_types[l]);
                float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
                if (hit.second) {
                    v1 = hit.second->v1;
                    v2 = hit.second->v2;
                    v3 = hit.second->v3;
                }
                else {
                    auto const emp = native_torsion_empirical(desc, tables, at_span, j, k, (std::int32_t)b);
                    v1 = emp[0];
                    v2 = emp[1];
                    v3 = emp[2];
                }
                if (native_is_zero(v1) && native_is_zero(v2) && native_is_zero(v3)) continue;
                if (torsion_count < max_mmff_torsions_per_mol_k) {
                    torsions_out[torsion_count++] = {
                        (std::uint8_t)i, (std::uint8_t)j, (std::uint8_t)k, (std::uint8_t)l, v1, v2, v3};
                }
                else { overflow = 1; }
            }
        }
    }

    // OOP terms.
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint32_t const deg = desc.degree[center];
        if (deg < 3) continue;
        std::uint16_t const nstart = desc.neighbor_offset[center];
        std::uint8_t neighbor_cache[8];
        std::uint32_t neighbor_count = 0;
        for (std::uint16_t nn = nstart; nn < desc.neighbor_offset[center + 1] && neighbor_count < 8; ++nn)
            neighbor_cache[neighbor_count++] = desc.neighbor_atom[nn];
        for (std::uint32_t a = 0; a < neighbor_count; ++a) {
            std::uint8_t plane[2];
            std::uint32_t plane_count = 0;
            for (std::uint32_t c = 0; c < neighbor_count && plane_count < 2; ++c)
                if (c != a) plane[plane_count++] = neighbor_cache[c];
            if (plane_count < 2) continue;
            auto const *op = lookup_oop(tables, atom_types[neighbor_cache[a]], atom_types[center], atom_types[plane[0]],
                                        atom_types[plane[1]]);
            if (!op) continue;
            if (oop_count < max_mmff_oops_per_mol_k) {
                oops_out[oop_count++] = {neighbor_cache[a], (std::uint8_t)center, plane[0], plane[1],
                                         op->koop,          {0.0f, 0.0f}};
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
            a.angle_type, (atom_types[a.i] <= atom_types[a.k]) ? a.bond_type_ij : a.bond_type_jk,
            (atom_types[a.i] < atom_types[a.k]) ? a.bond_type_jk : a.bond_type_ij);
        auto stbn = lookup_stbn(tables, sbt, a.bond_type_ij, a.bond_type_jk, atom_types[a.i], atom_types[a.j],
                                atom_types[a.k]);
        float kba_ijk = 0.0f, kba_kji = 0.0f;
        bool swap_flag = false;
        if (stbn.second) {
            kba_ijk = stbn.second->kba_ijk;
            kba_kji = stbn.second->kba_kji;
            swap_flag = stbn.first;
        }
        else {
            auto dfsb = lookup_dfsb(tables, periodic_table_row(desc.element[a.i]),
                                    periodic_table_row(desc.element[a.j]), periodic_table_row(desc.element[a.k]));
            if (!dfsb.second) continue;
            kba_ijk = dfsb.second->kba_ijk;
            kba_kji = dfsb.second->kba_kji;
            swap_flag = dfsb.first;
        }
        if (native_is_zero(kba_ijk) && native_is_zero(kba_kji)) continue;
        if (swap_flag) {
            float tmp = kba_ijk;
            kba_ijk = kba_kji;
            kba_kji = tmp;
        }
        float const theta0_rad = a.theta0_deg * (float)M_PI / 180.0f;
        float const r0_ij = bond_r0[a.bond_index_ij];
        float const r0_jk = bond_r0[a.bond_index_jk];
        if (stbn_count < max_mmff_stretchbends_per_mol_k) {
            stbns_out[stbn_count++] = {a.i, a.j, a.k, 0, kba_ijk, kba_kji, theta0_rad, r0_ij, r0_jk, 0.0f};
        }
        else { overflow = 1; }
    }

    per_mol_counts[mol] = {(std::uint16_t)bond_count,
                           (std::uint16_t)angle_count,
                           (std::uint16_t)torsion_count,
                           (std::uint16_t)oop_count,
                           (std::uint16_t)stbn_count,
                           0,
                           overflow,
                           0};
}

/** Build the MMFF per-atom CSR interaction list from emitted terms.
 *  Mirrors `mmff94_cpu.hpp:731-802` — two-pass count + write.  One
 *  block per molecule, lane-0-only (the CSR-build cost is tiny
 *  relative to the term extractor and cross-molecule parallelism
 *  dominates). */
__launch_bounds__(32, 8) __global__ void mmff94_build_csr_kernel(
    molecule_descriptor_t const *__restrict__ descriptors, std::uint32_t const *__restrict__ atom_offsets,
    std::uint32_t num_molecules, mmff94_per_mol_counts_t *__restrict__ per_mol_counts,
    mmff94_bond_t const *__restrict__ bonds_all, mmff94_angle_t const *__restrict__ angles_all,
    mmff94_torsion_t const *__restrict__ torsions_all, mmff94_oop_t const *__restrict__ oops_all,
    mmff94_stretchbend_t const *__restrict__ stbns_all, mmff94_atom_interaction_t *__restrict__ interactions_all,
    mmff94_atom_offsets_t *__restrict__ atom_offsets_all, std::uint8_t const *__restrict__ typing_status) {

    std::uint32_t const mol = blockIdx.x;
    if (mol >= num_molecules) return;
    if (threadIdx.x != 0) return;
    if (typing_status[mol] != 0) return;

    auto const &desc = descriptors[mol];
    std::uint32_t const num_atoms = desc.num_atoms;
    std::uint32_t const ao = atom_offsets[mol];
    auto const &c = per_mol_counts[mol];

    mmff94_bond_t const *bonds = bonds_all + (std::size_t)mol * max_mmff_bonds_per_mol_k;
    mmff94_angle_t const *angles = angles_all + (std::size_t)mol * max_mmff_angles_per_mol_k;
    mmff94_torsion_t const *torsions = torsions_all + (std::size_t)mol * max_mmff_torsions_per_mol_k;
    mmff94_oop_t const *oops = oops_all + (std::size_t)mol * max_mmff_oops_per_mol_k;
    mmff94_stretchbend_t const *stbns = stbns_all + (std::size_t)mol * max_mmff_stretchbends_per_mol_k;

    std::uint32_t const ints_base = (std::uint32_t)mol * max_mmff_interactions_per_mol_k;
    mmff94_atom_interaction_t *interactions = interactions_all + ints_base;
    mmff94_atom_offsets_t *offsets = atom_offsets_all + ao;

    // Pass 1 — count per-atom per-term-type.
    for (std::uint32_t a = 0; a < num_atoms; ++a) offsets[a] = {};
    for (std::uint32_t t = 0; t < c.bonds; ++t) {
        offsets[bonds[t].atom_i].bond_count++;
        offsets[bonds[t].atom_j].bond_count++;
    }
    for (std::uint32_t t = 0; t < c.angles; ++t) {
        offsets[angles[t].atom_i].angle_count++;
        offsets[angles[t].atom_j].angle_count++;
        offsets[angles[t].atom_k].angle_count++;
    }
    for (std::uint32_t t = 0; t < c.torsions; ++t) {
        offsets[torsions[t].atom_i].torsion_count++;
        offsets[torsions[t].atom_j].torsion_count++;
        offsets[torsions[t].atom_k].torsion_count++;
        offsets[torsions[t].atom_l].torsion_count++;
    }
    for (std::uint32_t t = 0; t < c.oops; ++t) {
        offsets[oops[t].atom_i].oop_count++;
        offsets[oops[t].atom_j].oop_count++;
        offsets[oops[t].atom_k].oop_count++;
        offsets[oops[t].atom_l].oop_count++;
    }
    for (std::uint32_t t = 0; t < c.stretchbends; ++t) {
        offsets[stbns[t].atom_i].stretchbend_count++;
        offsets[stbns[t].atom_j].stretchbend_count++;
        offsets[stbns[t].atom_k].stretchbend_count++;
    }

    // Compute per-type offsets + running total.
    std::uint32_t running = 0;
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        auto &off = offsets[a];
        off.bond_start = running;
        running += off.bond_count;
        off.angle_start = running;
        running += off.angle_count;
        off.torsion_start = running;
        running += off.torsion_count;
        off.oop_start = running;
        running += off.oop_count;
        off.stretchbend_start = running;
        running += off.stretchbend_count;
    }
    // Running is the total interactions for this molecule. Store in
    // per_mol_counts for downstream CSR consumers.
    per_mol_counts[mol].interactions = (std::uint16_t)running;

    // Per-atom per-type cursors (reuse `_start` as running cursor).
    std::uint32_t cursors_bond[max_atoms_k];
    std::uint32_t cursors_angle[max_atoms_k];
    std::uint32_t cursors_torsion[max_atoms_k];
    std::uint32_t cursors_oop[max_atoms_k];
    std::uint32_t cursors_sb[max_atoms_k];
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        cursors_bond[a] = offsets[a].bond_start;
        cursors_angle[a] = offsets[a].angle_start;
        cursors_torsion[a] = offsets[a].torsion_start;
        cursors_oop[a] = offsets[a].oop_start;
        cursors_sb[a] = offsets[a].stretchbend_start;
    }

    // Pass 2 — write interactions.
    for (std::uint32_t t = 0; t < c.bonds; ++t) {
        interactions[cursors_bond[bonds[t].atom_i]++] = {t, mmff94_bond_k, atom_role_first_k};
        interactions[cursors_bond[bonds[t].atom_j]++] = {t, mmff94_bond_k, atom_role_second_k};
    }
    for (std::uint32_t t = 0; t < c.angles; ++t) {
        interactions[cursors_angle[angles[t].atom_i]++] = {t, mmff94_angle_end_k, atom_role_first_k};
        interactions[cursors_angle[angles[t].atom_j]++] = {t, mmff94_angle_center_k, atom_role_second_k};
        interactions[cursors_angle[angles[t].atom_k]++] = {t, mmff94_angle_end_k, atom_role_third_k};
    }
    for (std::uint32_t t = 0; t < c.torsions; ++t) {
        interactions[cursors_torsion[torsions[t].atom_i]++] = {t, mmff94_torsion_k, atom_role_first_k};
        interactions[cursors_torsion[torsions[t].atom_j]++] = {t, mmff94_torsion_k, atom_role_second_k};
        interactions[cursors_torsion[torsions[t].atom_k]++] = {t, mmff94_torsion_k, atom_role_third_k};
        interactions[cursors_torsion[torsions[t].atom_l]++] = {t, mmff94_torsion_k, atom_role_fourth_k};
    }
    for (std::uint32_t t = 0; t < c.oops; ++t) {
        interactions[cursors_oop[oops[t].atom_i]++] = {t, mmff94_out_of_plane_k, atom_role_first_k};
        interactions[cursors_oop[oops[t].atom_j]++] = {t, mmff94_out_of_plane_k, atom_role_second_k};
        interactions[cursors_oop[oops[t].atom_k]++] = {t, mmff94_out_of_plane_k, atom_role_third_k};
        interactions[cursors_oop[oops[t].atom_l]++] = {t, mmff94_out_of_plane_k, atom_role_fourth_k};
    }
    for (std::uint32_t t = 0; t < c.stretchbends; ++t) {
        interactions[cursors_sb[stbns[t].atom_i]++] = {t, mmff94_stretch_bend_k, atom_role_first_k};
        interactions[cursors_sb[stbns[t].atom_j]++] = {t, mmff94_stretch_bend_k, atom_role_second_k};
        interactions[cursors_sb[stbns[t].atom_k]++] = {t, mmff94_stretch_bend_k, atom_role_third_k};
    }
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
