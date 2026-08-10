/**
 *  @file mmff94_extract_device.cuh
 *  @brief `mmff94_extract_terms_kernel` runs the native MMFF term extractor on the GPU.
 *
 *  Layout decisions (initial pass, optimisable later):
 *   - One block per molecule; lane-0 does all the work within a
 *     molecule. Atom typing + lookup enumeration is an inherently
 *     graph-walking algorithm with short-lived per-molecule state
 *     (angle_scratch etc.) — SIMT parallelism across atoms / bonds is
 *     possible but adds substantial complexity for modest per-molecule
 *     speedup (drug-like mols average < 30 atoms / 30 bonds). The win
 *     is in the cross-molecule dimension: 50 K molecules × 132 SMs ×
 *     8 blocks/SM = ~1000 molecules in flight at steady state.
 *   - The extractor and the CSR builder write into per-molecule slots of
 *     fixed stride (`max_mmff_*_per_mol` constants), which costs 116 KB of
 *     staging per molecule, and report their actual per-term-type counts
 *     to the `per_mol_counts` array.
 *   - `mmff94_scan_counts_kernel` + `mmff94_compact_terms_kernel` turn that
 *     staging into the packed variable-length layout `wave_slice_pool_t` and
 *     `mmff_polish_warp_kernel` consume: one contiguous array per term type,
 *     per-molecule bases stamped into `etkdg_molecule_metadata_t`, and the
 *     per-atom CSR starts rebased from molecule-relative to batch-global.
 *     `test_mmff_gpu_extract` prints both footprints per molecule and gates
 *     the packed arrays byte for byte against the staging; it measures 17.2 KB
 *     per molecule over 19 167 PubChem molecules and 19.4 KB over 20 009
 *     Enamine REAL molecules, so compaction buys 6.1x to 6.9x on those corpora
 *     and the packed arrays are the ones a 50 K-molecule wave allocates.
 */

#pragma once

#include <cstdint> // `std::uint16_t`, `std::uint8_t`

#include <cuda_runtime.h>

#include "smallfield/module_fwd.hpp"
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

namespace detail_extract {

/** Angle scratch — populated during the angle pass and consumed by the
 *  stretch-bend pass. Sized at `max_mmff_angles_per_mol_k` slots per
 *  molecule and held in shared memory, one array per block rather than
 *  one per thread: 16 B × 512 = 8 KB per block, which the static
 *  `__shared__` assertion below pins to the struct layout. */
struct angle_scratch_t {
    std::uint8_t i, j, k;
    std::uint8_t angle_type;
    std::uint8_t bond_type_ij, bond_type_jk;
    std::int16_t bond_index_ij, bond_index_jk;
    float theta0_deg;
};

static_assert(sizeof(angle_scratch_t) == 16, "angle scratch must stay at 8 KB per block");

} // namespace detail_extract

#if defined(__CUDACC__)
/** Single-kernel MMFF94 term extractor. One block per molecule; lane 0
 *  of the block does all enumeration + classification + lookup. Outputs
 *  are written into the per-molecule fixed-stride slots of the output
 *  arrays. Blockers on this kernel are avoided by keeping per-molecule
 *  state in shared memory (angle_scratch + per-bond scratch). */
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

    std::uint32_t const mol = blockIdx.x;
    if (mol >= num_molecules) return;

    // Early-exit status check runs on all 32 lanes so they all
    // agree to fall out together (no warp divergence afterwards).
    if (typing_status[mol] != 0) {
        if (threadIdx.x == 0) per_mol_counts[mol] = {};
        return;
    }

    std::uint32_t const lane = threadIdx.x;
    molecule_descriptor_t const &descriptor = descriptors[mol];
    std::uint32_t const atom_off = atom_offsets[mol];
    std::uint8_t const *atom_types = atom_types_all + atom_off;
    float const *charges = atom_charges_all + atom_off;
    std::uint32_t const num_atoms = descriptor.num_atoms;
    std::uint32_t const num_bonds = descriptor.num_bonds;

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
        std::uint8_t const i = descriptor.bond_atom_i[b];
        std::uint8_t const j = descriptor.bond_atom_j[b];
        std::uint8_t bt = 0;
        bool const arom_bond = (descriptor.bond_flags[b] & bond_flag_aromatic_k) != 0;
        if (descriptor.bond_order[b] == 1 && !arom_bond) {
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
            auto const emp = native_bond_empirical(tables, descriptor.element[i], descriptor.element[j]);
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

    // Angle scratch — populated by the angle pass, consumed by the stretch-bend
    // pass. Block-scoped, so the cost is 8 KB per warp rather than per thread.
    __shared__ detail_extract::angle_scratch_t shared_angle_scratch[max_mmff_angles_per_mol_k];

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
    std::uint32_t angle_scratch_count = 0;

    std::span<std::uint8_t const> bond_types {bond_type_arr, num_bonds};
    std::span<std::uint8_t const> at_span {atom_types, num_atoms};
    for (std::uint32_t center = 0; center < num_atoms; ++center) {
        std::uint16_t const nstart = descriptor.neighbor_offset[center];
        std::uint16_t const nend = descriptor.neighbor_offset[center + 1];
        for (std::uint16_t a = nstart; a < nend; ++a) {
            for (std::uint16_t b = a + 1; b < nend; ++b) {
                std::uint32_t const ai = descriptor.neighbor_atom[a];
                std::uint32_t const aj = descriptor.neighbor_atom[b];
                std::int32_t const bond_index_ic = find_bond_index(descriptor, ai, center);
                std::int32_t const bond_index_cj = find_bond_index(descriptor, center, aj);
                if (bond_index_ic < 0 || bond_index_cj < 0) continue;
                std::uint8_t const at_code = native_get_mmff_angle_type(descriptor, bond_types, ai, center, aj);
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
                    auto const emp = native_angle_empirical(descriptor, tables, at_span, ai, center, aj, r0_ic, r0_cj,
                                                            override_deg);
                    ka = emp.first;
                    theta0_deg = emp.second;
                }
                float const theta0_rad = theta0_deg * (float)M_PI / 180.0f;
                auto const *prop_center = lookup_prop(tables, at_span[center]);
                std::uint8_t const is_linear = (prop_center && prop_center->linh) ? 1u : 0u;
                if (angle_count < max_mmff_angles_per_mol_k) {
                    angles_out[angle_count++] = {
                        (std::uint8_t)ai, (std::uint8_t)center, (std::uint8_t)aj, is_linear, ka, theta0_rad, 0.0f};
                }
                else { overflow = 1; }
                if (angle_scratch_count < max_mmff_angles_per_mol_k) {
                    shared_angle_scratch[angle_scratch_count++] = {(std::uint8_t)ai,
                                                                   (std::uint8_t)center,
                                                                   (std::uint8_t)aj,
                                                                   at_code,
                                                                   bond_type_arr[bond_index_ic],
                                                                   bond_type_arr[bond_index_cj],
                                                                   (std::int16_t)bond_index_ic,
                                                                   (std::int16_t)bond_index_cj,
                                                                   theta0_deg};
                }
            }
        }
    }

    // Torsion terms.
    for (std::uint32_t b = 0; b < num_bonds; ++b) {
        std::uint32_t const j = descriptor.bond_atom_i[b];
        std::uint32_t const k = descriptor.bond_atom_j[b];
        std::uint16_t const jstart = descriptor.neighbor_offset[j];
        std::uint16_t const jend = descriptor.neighbor_offset[j + 1];
        std::uint16_t const kstart = descriptor.neighbor_offset[k];
        std::uint16_t const kend = descriptor.neighbor_offset[k + 1];
        for (std::uint16_t nj = jstart; nj < jend; ++nj) {
            std::uint32_t const i = descriptor.neighbor_atom[nj];
            if (i == k) continue;
            for (std::uint16_t nk = kstart; nk < kend; ++nk) {
                std::uint32_t const l = descriptor.neighbor_atom[nk];
                if (l == j) continue;
                auto const tt = native_get_mmff_torsion_type(descriptor, at_span, bond_types, i, j, k, l);
                auto const hit = lookup_torsion(tables, tt.first, tt.second, atom_types[i], atom_types[j],
                                                atom_types[k], atom_types[l]);
                float v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
                if (hit.second) {
                    v1 = hit.second->v1;
                    v2 = hit.second->v2;
                    v3 = hit.second->v3;
                }
                else {
                    auto const emp = native_torsion_empirical(descriptor, tables, at_span, j, k, (std::int32_t)b);
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
        std::uint32_t const deg = descriptor.degree[center];
        if (deg < 3) continue;
        std::uint16_t const nstart = descriptor.neighbor_offset[center];
        std::uint8_t neighbor_cache[8];
        std::uint32_t neighbor_count = 0;
        for (std::uint16_t nn = nstart; nn < descriptor.neighbor_offset[center + 1] && neighbor_count < 8; ++nn)
            neighbor_cache[neighbor_count++] = descriptor.neighbor_atom[nn];
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
    for (std::uint32_t i = 0; i < angle_scratch_count; ++i) {
        auto const &a = shared_angle_scratch[i];
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
            auto dfsb = lookup_dfsb(tables, periodic_table_row(descriptor.element[a.i]),
                                    periodic_table_row(descriptor.element[a.j]),
                                    periodic_table_row(descriptor.element[a.k]));
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
 *  Mirrors `mmff94_cpu.hpp:731-802` — two-pass count + write. One
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

    std::uint32_t const mol = blockIdx.x;
    if (mol >= num_molecules) return;
    if (threadIdx.x != 0) return;
    if (typing_status[mol] != 0) return;

    auto const &descriptor = descriptors[mol];
    std::uint32_t const num_atoms = descriptor.num_atoms;
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
        off.bond_offset = running;
        running += off.bond_count;
        off.angle_offset = running;
        running += off.angle_count;
        off.torsion_offset = running;
        running += off.torsion_count;
        off.oop_offset = running;
        running += off.oop_count;
        off.stretchbend_offset = running;
        running += off.stretchbend_count;
    }
    // Running is the total interactions for this molecule. Store in
    // per_mol_counts for downstream CSR consumers.
    per_mol_counts[mol].interactions = (std::uint16_t)running;

    // Seed the write cursors from the `_start` fields just computed.
    for (std::uint32_t a = 0; a < num_atoms; ++a) {
        cursors_bond[a] = offsets[a].bond_offset;
        cursors_angle[a] = offsets[a].angle_offset;
        cursors_torsion[a] = offsets[a].torsion_offset;
        cursors_oop[a] = offsets[a].oop_offset;
        cursors_stretchbend[a] = offsets[a].stretchbend_offset;
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
        interactions[cursors_stretchbend[stbns[t].atom_i]++] = {t, mmff94_stretch_bend_k, atom_role_first_k};
        interactions[cursors_stretchbend[stbns[t].atom_j]++] = {t, mmff94_stretch_bend_k, atom_role_second_k};
        interactions[cursors_stretchbend[stbns[t].atom_k]++] = {t, mmff94_stretch_bend_k, atom_role_third_k};
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
        std::uint32_t own[mmff94_packed_components_k] = {0, 0, 0, 0, 0, 0};
        if (molecule < num_molecules) {
            auto const counts = per_mol_counts[molecule];
            own[0] = counts.bonds;
            own[1] = counts.angles;
            own[2] = counts.torsions;
            own[3] = counts.oops;
            own[4] = counts.stretchbends;
            own[5] = counts.interactions;
        }
        for (std::uint32_t component = 0; component < mmff94_packed_components_k; ++component)
            partial_sums[component][slot] = own[component];
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
            per_mol_offsets[molecule] = {
                carry[0] + partial_sums[0][slot] - own[0], carry[1] + partial_sums[1][slot] - own[1],
                carry[2] + partial_sums[2][slot] - own[2], carry[3] + partial_sums[3][slot] - own[3],
                carry[4] + partial_sums[4][slot] - own[4], carry[5] + partial_sums[5][slot] - own[5]};
        for (std::uint32_t component = 0; component < mmff94_packed_components_k; ++component)
            carry[component] += partial_sums[component][mmff94_scan_block_k - 1];
        __syncthreads();
    }

    if (slot == 0) *totals = {carry[0], carry[1], carry[2], carry[3], carry[4], carry[5]};
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

    std::uint32_t const mol = blockIdx.x;
    if (mol >= num_molecules) return;
    std::uint32_t const lane = threadIdx.x;

    auto const counts = per_mol_counts[mol];
    auto const bases = per_mol_offsets[mol];
    std::uint32_t const atom_base = atom_offsets[mol];

    copy_packed_terms_(staged_bonds + (std::size_t)mol * max_mmff_bonds_per_mol_k, packed_bonds + bases.bonds,
                       counts.bonds, lane);
    copy_packed_terms_(staged_angles + (std::size_t)mol * max_mmff_angles_per_mol_k, packed_angles + bases.angles,
                       counts.angles, lane);
    copy_packed_terms_(staged_torsions + (std::size_t)mol * max_mmff_torsions_per_mol_k,
                       packed_torsions + bases.torsions, counts.torsions, lane);
    copy_packed_terms_(staged_oops + (std::size_t)mol * max_mmff_oops_per_mol_k, packed_oops + bases.oops, counts.oops,
                       lane);
    copy_packed_terms_(staged_stbns + (std::size_t)mol * max_mmff_stretchbends_per_mol_k,
                       packed_stbns + bases.stretchbends, counts.stretchbends, lane);
    copy_packed_terms_(staged_interactions + (std::size_t)mol * max_mmff_interactions_per_mol_k,
                       packed_interactions + bases.interactions, counts.interactions, lane);

    // The CSR builder wrote molecule-relative starts; the consumers index one
    // batch-global interaction array, so shift every start by this molecule's base.
    // A fallback molecule never entered the CSR builder, so its slots start empty.
    bool const typed = typing_status[mol] == 0;
    std::uint32_t const num_atoms = descriptors[mol].num_atoms;
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
    auto &entry = metadata[mol];
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
