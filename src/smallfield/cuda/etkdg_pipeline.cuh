/**
 *  @file etkdg_pipeline.cuh
 *  @brief Persistent warp-loop CUDA kernels for the ETKDG conformer generation pipeline.
 *
 *  The pipeline lives in two sibling kernels that share the `warp_per_molecule`
 *  pattern, split to keep register pressure per kernel under the launch-bounds
 *  cap.  The monolithic predecessor hit 145 regs/thread because DG/ETK/MMFF
 *  state was held simultaneously across branch boundaries; each sub-kernel
 *  here lands at 72 regs/thread on sm_90.
 *
 *  @c etkdg_pipeline_warp_kernel — bounds quantization through final validation
 *  (bounds quantize → DG FIRE → stereo retry → fourth-dim crush → ETK torsion
 *  → final validation).
 *  Writes ETK-refined 3D positions to @c conformer_coordinates_out.
 *
 *  @c mmff_polish_warp_kernel — MMFF94 polish (MMFF94 relaxation + optional heavy
 *  rescue).  Reads positions from @c conformer_coordinates_out (written by
 *  the first kernel), overwrites them with the MMFF-polished coordinates,
 *  replaces the ETK energy with the MMFF energy in @c conformer_energies_out.
 *  Skips slots whose status is non-zero (failed stereo) or whose molecule
 *  has no MMFF terms.
 *
 *  Stages per molecule (across both kernels):
 *   bounds quantization: load pre-smoothed bounds from global memory, int8-quantize to shared
 *   metric embedding: native metric embedding with mirrored retries (separate prepass kernel)
 *   distance geometry minimization: 4D FIRE, atom-centric
 *   stereochemistry validation: chiral volume signs
 *   fourth-dimension crush: FIRE with w₄ penalty
 *   torsion refinement: ETK 3D FIRE, atom-centric
 *   final validation: post-refinement stereochemistry check
 *   MMFF94 polish: in the @c mmff_polish_warp_kernel sibling
 *
 *  Architecture:
 *   - warp-loop: gridDim warps, each loops over (molecule, conformer) pairs with stride
 *   - atom-centric: no atomicAdd, no pair index scan
 *   - int8 bounds: per-molecule linear quantization, 16 KB at N=128
 *   - Philox RNG: stateless counter-based, ~100 cycles init
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cfloat>
#include <cuda_fp16.h>
#include <curand_kernel.h>

#include "smallfield/cuda/dg_kernel.cuh"
#include "smallfield/cuda/etk_kernel.cuh"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/cuda/fire_warp.cuh"
#include "smallfield/cuda/mmff94_energy_kernel.cuh"

namespace smallfield::cuda {
using namespace smallfield::types;

/** @brief Per-molecule FIRE iteration budgets for ETK torsion refinement and MMFF94 polish. */
struct refinement_policy_t {
    std::int32_t etk_initial_iterations;
    std::int32_t etk_continuation_iterations;
    float etk_gradient_threshold;
    std::int32_t mmff_initial_iterations;
    std::int32_t mmff_continuation_iterations;
    float mmff_gradient_threshold;
};

__host__ __device__ __forceinline__ std::uint32_t align_up_(std::uint32_t value, std::uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

__host__ __device__ __forceinline__ std::uint32_t total_mmff_terms_(etkdg_molecule_metadata_t const &meta) {
    // Nonbonded pairs are enumerated on the fly — approximate their contribution as
    // N²/2 to keep the "is this a heavy case" heuristic calibrated similarly.
    std::uint32_t const approx_nonbonded = meta.num_atoms * (meta.num_atoms - 1u) / 2u;
    return approx_nonbonded + meta.num_mmff_bonds + meta.num_mmff_angles + meta.num_mmff_torsions + meta.num_mmff_oops +
           meta.num_mmff_stretchbends;
}

__host__ __device__ __forceinline__ bool is_heavy_refinement_case_(etkdg_molecule_metadata_t const &meta) {
    std::uint32_t const total_mmff_terms = total_mmff_terms_(meta);
    std::uint32_t const torsion_complexity = meta.num_etk_torsions + meta.num_mmff_torsions;
    return meta.num_atoms >= 96 || total_mmff_terms >= 900 || torsion_complexity >= 64;
}

__host__ __device__ __forceinline__ float heavy_mmff_rescue_threshold_(etkdg_molecule_metadata_t const &meta) {
    return fmaxf(220.0f, 2.0f * static_cast<float>(meta.num_atoms));
}

__host__ __device__ __forceinline__ refinement_policy_t
select_refinement_policy_(etkdg_molecule_metadata_t const &meta) {
    std::uint32_t const total_mmff_terms = total_mmff_terms_(meta);
    std::uint32_t const torsion_complexity = meta.num_etk_torsions + meta.num_mmff_torsions;

    if (meta.num_atoms <= 20 && total_mmff_terms <= 96 && torsion_complexity <= 12)
        return {60, 0, 1.5e-4f, 120, 0, 1.5e-4f};

    if (is_heavy_refinement_case_(meta)) return {160, 80, 7.5e-5f, 320, 160, 5.0e-5f};

    return {100, 40, 1.0e-4f, 200, 80, 1.0e-4f};
}

template <typename energy_gradient_functor_>
__device__ __forceinline__ fire_result_t run_fire_stage_(float *positions, float *velocities, float *gradient,
                                                         std::uint32_t num_elements,
                                                         std::int32_t initial_iterations,
                                                         std::int32_t continuation_iterations,
                                                         float gradient_threshold, std::uint32_t lane,
                                                         energy_gradient_functor_ &&compute_energy_gradient) {
    fire_result_t result = fire_minimize_(positions, velocities, gradient, num_elements, initial_iterations,
                                          gradient_threshold, lane, compute_energy_gradient);

    bool const needs_continuation = continuation_iterations > 0 &&
                                    (result.status != 0 || result.rms_gradient > gradient_threshold);
    if (needs_continuation) {
        std::int32_t const initial_iterations_done = result.iterations;
        result = fire_minimize_(positions, velocities, gradient, num_elements, continuation_iterations,
                                gradient_threshold, lane, compute_energy_gradient);
        result.iterations += initial_iterations_done;
    }
    return result;
}

/** Compute shared memory bytes for the DG+ETK pipeline kernel.
 *
 *  Layout: [bounds matrix: N² × uint8_t][positions, velocities, gradient: 3 × 4N × float].
 *  Bounds are int8-quantized with a per-molecule min/inv_scale pair kept in
 *  registers — halves the bounds footprint vs the previous `__half` layout,
 *  32 KB → 16 KB at N=128.
 *
 *  @see compute_mmff_polish_shared_memory_bytes
 */
__host__ __device__ __forceinline__ std::uint32_t compute_pipeline_shared_memory_bytes(std::uint32_t max_atoms) {
    std::uint32_t const dimensions_4d = 4 * max_atoms;
    std::uint32_t const bounds_bytes = max_atoms * compute_bounds_row_stride(max_atoms) * sizeof(std::uint8_t);
    std::uint32_t const aligned_bounds_bytes = align_up_(bounds_bytes, alignof(float));
    std::uint32_t const fire_arrays = 3 * dimensions_4d * sizeof(float);
    return aligned_bounds_bytes + fire_arrays;
}

/** Compute shared memory bytes for the MMFF polish kernel.
 *
 *  Layout: [positions, velocities, gradient: 3 × 3N × float]
 *          [atom_vdw cache: N × mmff94_atom_vdw_params_t]
 *          [atom_charge cache: N × float].
 *  No bounds matrix — MMFF94 uses explicit bonded + nonbonded term lists rather
 *  than a pair-distance matrix.
 *  At N=128: 4.5 KB FIRE + 3.5 KB caches = ~8 KB; still leaves comfortable
 *  occupancy headroom on Hopper / Blackwell (typically reg-cap-bound rather
 *  than shmem-bound).  The caches eliminate ~2N² duplicate global reads of
 *  atom_vdw_params[j] / atom_charge[j] per FIRE step.
 *
 *  @see compute_pipeline_shared_memory_bytes
 */
__host__ __device__ __forceinline__ std::uint32_t compute_mmff_polish_shared_memory_bytes(std::uint32_t max_atoms) {
    std::uint32_t const dimensions_3d = 3 * max_atoms;
    std::uint32_t const fire_bytes = 3 * dimensions_3d * sizeof(float);
    std::uint32_t const atom_vdw_bytes = max_atoms * static_cast<std::uint32_t>(
                                                          sizeof(smallfield::types::mmff94_atom_vdw_params_t));
    std::uint32_t const atom_charge_bytes = max_atoms * static_cast<std::uint32_t>(sizeof(float));
    return fire_bytes + atom_vdw_bytes + atom_charge_bytes;
}

/**
 *  @brief Load 4D metric-embedding coordinates into shared-memory positions.
 *
 *  Adds per-conformer jitter and optionally reflects the x-axis for stereo retries.
 *  Called once at the start of each stereo attempt inside the DG+ETK pipeline kernel.
 *
 *  @param[in] initial_coordinates Global 4D embedding, SoA at @p initial_coordinate_stride.
 *  @param[out] positions Shared-memory SoA positions to fill — 4 × @p num_atoms floats.
 *  @param[in] initial_coordinate_stride Atom stride in the global embedding array.
 *  @param[in] num_atoms Number of atoms in this molecule.
 *  @param[in] conf Conformer index — affects jitter magnitude.
 *  @param[in] attempt Stereo-retry attempt — odd attempts reflect x.
 *  @param[in] lane Warp lane index, 0–31.
 *  @param[inout] rng_state Philox RNG state for uniform jitter draws.
 *
 *  @see etkdg_pipeline_warp_kernel
 */
__device__ __forceinline__ void initialize_from_metric_coordinates_(float const *initial_coordinates, float *positions,
                                                                    std::uint32_t initial_coordinate_stride,
                                                                    std::uint32_t num_atoms, std::uint32_t conf,
                                                                    std::uint32_t attempt, std::uint32_t lane,
                                                                    curandStatePhilox4_32_10_t *rng_state) {
    float const xyz_jitter = 0.02f + 0.01f * static_cast<float>((conf + attempt) & 3u);
    float const w_jitter = 0.02f + 0.01f * static_cast<float>(attempt & 3u);
    float const x_reflection = (attempt & 1u) ? -1.0f : 1.0f;
    for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k) {
        for (std::uint32_t dim = 0; dim < 4; ++dim) {
            float base = initial_coordinates[dim * initial_coordinate_stride + atom];
            if (dim == 0) base *= x_reflection;
            float const jitter = (dim < 3) ? xyz_jitter : w_jitter;
            positions[dim * num_atoms + atom] = base + (curand_uniform(rng_state) - 0.5f) * jitter;
        }
    }
    __syncwarp(full_mask_k);
}

/**
 *  @brief Copy one conformer's 3D positions from shared memory to the global output array.
 *
 *  Writes energy and status for the conformer slot.
 *  The output uses @p output_stride between atoms — which may differ from @p num_atoms —
 *  while shared-memory positions are packed at @p num_atoms stride.
 *
 *  @param[out] coordinates_out Global output coordinate buffer, SoA at @p output_stride.
 *  @param[out] energies_out Global output energy buffer, one float per conformer slot.
 *  @param[out] status_out Global output status buffer, 0 = success, 1 = failed.
 *  @param[in] positions Shared-memory SoA positions — 3 × @p num_atoms floats.
 *  @param[in] mol_index Molecule index within this wave's dispatch.
 *  @param[in] conformer_index Conformer index within this molecule.
 *  @param[in] num_atoms Number of atoms in this molecule.
 *  @param[in] output_stride Atom stride in the global output arrays.
 *  @param[in] num_conformers Conformers per molecule — used to compute the flat slot.
 *  @param[in] energy Final energy to write for this conformer.
 *  @param[in] succeeded Whether this conformer passed stereo validation.
 *  @param[in] lane Warp lane index, 0–31.
 *
 *  @see etkdg_pipeline_warp_kernel
 *  @see mmff_polish_warp_kernel
 */
__device__ __forceinline__ void write_conformer_output_(float *coordinates_out, float *energies_out,
                                                        std::uint8_t *status_out, float const *positions,
                                                        std::uint32_t mol_index, std::uint32_t conformer_index,
                                                        std::uint32_t num_atoms, std::uint32_t output_stride,
                                                        std::uint32_t num_conformers, float energy, bool succeeded,
                                                        std::uint32_t lane) {
    std::uint32_t const slot = mol_index * num_conformers + conformer_index;
    std::uint32_t const coord_offset = slot * output_stride * 3;
    // Positions use SoA layout with num_atoms stride, while output uses output_stride.
    for (std::uint32_t d = 0; d < 3; ++d)
        for (std::uint32_t i = lane; i < num_atoms; i += warp_size_k)
            coordinates_out[coord_offset + d * output_stride + i] = positions[d * num_atoms + i];
    if (lane == 0) {
        energies_out[slot] = energy;
        status_out[slot] = succeeded ? 0 : 1;
    }
    __syncwarp(full_mask_k);
}

/**
 *  Persistent warp-loop ETKDG kernel covering bounds quantization through final validation.
 *
 *  Launch: <<<min(total_molecules × num_conformers, max_concurrent_warps), 32, shared_bytes>>>
 *  Each warp loops over (molecule, conformer) pairs with stride = gridDim.x.
 *
 *  The unit of work is a pair rather than a molecule because the atom-count bins hold wildly
 *  different molecule counts: at a 50 K wave over the PubChem distribution, four of the eight
 *  bins hold fewer molecules than the resident-warp cap, so their launches cannot fill the
 *  device.  Multiplying the index space by the conformer count fills them.  The price is that
 *  the bounds quantization is paid per conformer instead of per molecule.
 *
 *  `__launch_bounds__(32, 4)` is a floor, not a cap: it asks nvcc to keep the kernel fitting at
 *  least four blocks per SM, and at 32 threads that permits far more registers than the hardware
 *  allows, so it constrains nothing here.  What actually bounds occupancy is the register count
 *  against the SM register file, and below the largest bins, the 32-blocks-per-SM slot limit.
 *
 *  @b Register-pressure-audit — nvcc 12.8, `cuobjdump --dump-resource-usage`:
 *    sm_80=72, sm_90=72, sm_100=118, sm_120=123 regs/thread.  The two Hopper-and-earlier
 *    targets spill nothing (STACK=0, LOCAL=0); both Blackwell targets carry a 120-byte frame.
 *    On sm_90 that is 65536 / (72 × 32) = 28 blocks/SM = 28 warps, against a hard ceiling of
 *    32 warps — a 32-thread block can never fill more than 32 of the SM's 64 warp slots, so
 *    50 % is the most this shape can reach at any register count.  Reaching 64 regs/thread
 *    would buy the last 1.14× of that; nothing below 32 regs/thread buys more.
 *    The remaining eight registers are not in the energy evaluators any more: stubbing the DG
 *    chiral loop out of the sm_90 build drops it to 71 and stubbing the ETK torsion term drops
 *    it not at all, while stubbing the whole ETK stage drops it to 64.  What is left is the
 *    state the stereo-retry loop and the FIRE driver carry across every stage.
 *
 *  Writes ETK-refined 3D positions into @c conformer_coordinates_out;
 *  @c mmff_polish_warp_kernel runs next on the same buffer.
 *
 *  @see mmff_polish_warp_kernel
 */
__global__ __launch_bounds__(32, 4) void etkdg_pipeline_warp_kernel(
    float const *__restrict__ bounds_matrices, float const *__restrict__ initial_coordinates,
    chiral_center_descriptor_t const *__restrict__ chiral_centers,
    etk_torsion_descriptor_t const *__restrict__ torsion_terms,
    distance_constraint_t const *__restrict__ distance_constraints,
    etk_atom_interaction_t const *__restrict__ etk_interactions,
    etk_atom_offsets_t const *__restrict__ etk_atom_offsets,
    etkdg_molecule_metadata_t const *__restrict__ molecule_metadata,
    std::uint32_t const *__restrict__ molecule_input_indices, float *__restrict__ conformer_coordinates_out,
    float *__restrict__ conformer_energies_out, std::uint8_t *__restrict__ conformer_status_out,
    std::uint32_t total_molecules, std::uint32_t num_conformers_per_molecule, std::uint32_t max_stereo_attempts,
    std::uint32_t max_atoms_per_molecule, std::uint32_t initial_coordinate_stride, std::uint32_t output_atoms_stride,
    std::uint32_t wave_input_base, std::uint64_t random_seed) {

    std::uint32_t const warp_id = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const grid_stride = gridDim.x;

    // Layout shared memory as [bounds matrix: N² uint8_t] [positions, velocities, gradient: 3 * 4N floats].
    extern __shared__ char shared_raw[];
    std::uint32_t const max_n = max_atoms_per_molecule;
    std::uint8_t *shared_bounds = reinterpret_cast<std::uint8_t *>(shared_raw);
    std::uint32_t const bounds_bytes = max_n * compute_bounds_row_stride(max_n) * sizeof(std::uint8_t);
    std::uint32_t const aligned_bounds_bytes = align_up_(bounds_bytes, alignof(float));
    float *shared_float = reinterpret_cast<float *>(shared_raw + aligned_bounds_bytes);

    // Process (molecule, conformer) pairs in a warp loop with gridDim.x stride.  The flat pair
    // index decomposes into the bin-relative molecule and the conformer slot within it.
    std::uint32_t const total_pairs = total_molecules * num_conformers_per_molecule;
    for (std::uint32_t pair_index = warp_id; pair_index < total_pairs; pair_index += grid_stride) {
        std::uint32_t const mol = pair_index / num_conformers_per_molecule;
        std::uint32_t const conf = pair_index - mol * num_conformers_per_molecule;

        auto const meta = molecule_metadata[mol];
        auto const refinement_policy = select_refinement_policy_(meta);
        std::uint32_t const num_atoms = meta.num_atoms;
        std::uint32_t const dimensions_4d = 4 * num_atoms;
        std::uint32_t const dimensions_3d = 3 * num_atoms;

        // FIRE shared memory: positions + velocities + gradient (3 arrays).
        std::uint32_t off = 0;
        float *positions = shared_float + off;
        off += dimensions_4d;
        float *velocities = shared_float + off;
        off += dimensions_4d;
        float *gradient = shared_float + off;

        // Bounds quantization: Load post-smoothed bounds (the metric prepass
        // wrote smoothed bounds back into global memory at meta.bounds_offset),
        // find per-molecule [min, max] across the warp, and int8-quantize into
        // shared memory.  Linear quantization:
        //   q = round((x − mol_min) · scale_quantize),
        //   x ≈ q · inv_scale + mol_min at read.
        // The min/max scan must run on the *post-smoothed* values — host-side
        // precompute in prepare_molecule_into would be over the un-smoothed
        // matrix and produces a different range, breaking the dequantisation.
        std::uint32_t const bounds_size = num_atoms * num_atoms;
        float const *global_bounds = bounds_matrices + meta.bounds_offset;

        float lane_min = FLT_MAX;
        float lane_max = -FLT_MAX;
        for (std::uint32_t i = lane; i < bounds_size; i += warp_size_k) {
            float const v = global_bounds[i];
            lane_min = fminf(lane_min, v);
            lane_max = fmaxf(lane_max, v);
        }
        float const mol_min = warp_reduce_min_f32_(lane_min);
        float const mol_max = warp_reduce_max_f32_(lane_max);
        float const range = fmaxf(mol_max - mol_min, 1e-6f);
        float const scale_quantize = 255.0f / range;
        float const inv_scale = range * (1.0f / 255.0f);

        // Global keeps the packed n² stride the metric prepass wrote back; shared uses the
        // bank-conflict-free padded stride, so the flat index has to be decomposed.
        std::uint32_t const bounds_row_stride = compute_bounds_row_stride(num_atoms);
        for (std::uint32_t entry_index = lane; entry_index < bounds_size; entry_index += warp_size_k) {
            float const value = global_bounds[entry_index];
            int quantized = __float2int_rn((value - mol_min) * scale_quantize);
            quantized = max(0, min(255, quantized));
            std::uint32_t const row = entry_index / num_atoms;
            std::uint32_t const column = entry_index - row * num_atoms;
            shared_bounds[row * bounds_row_stride + column] = static_cast<std::uint8_t>(quantized);
        }
        __syncwarp(full_mask_k);

        // Build molecule-level pointers into the batched arrays.
        float const *mol_initial_coordinates = initial_coordinates + mol * initial_coordinate_stride * 4;
        chiral_center_descriptor_t const *mol_chirals = chiral_centers + meta.chiral_offset;
        etk_torsion_descriptor_t const *mol_torsions = torsion_terms + meta.torsion_offset;
        distance_constraint_t const *mol_constraints = distance_constraints + meta.constraint_offset;

        // Seed Philox from the molecule's position in the caller's input, not from its offset
        // inside the device batch.  Preprocess distributes molecules across host threads with a
        // dynamic schedule and then atom-count-sorts them, so the batch offset of a given
        // molecule varies between runs while its input position does not; keying off the input
        // position is what makes a fixed `random_seed` reproduce the same conformers.  The
        // conformer index enters the stream too, so the conformers of one molecule stay
        // independent of each other.
        std::uint64_t const global_input_index = wave_input_base + molecule_input_indices[mol];
        std::uint64_t const sequence = (global_input_index * num_conformers_per_molecule + conf) * warp_size_k + lane;

        // A Philox state is a dozen registers wide, and it is touched only while the attempt
        // seeds its coordinates.  Rebuilding it at the head of every attempt from an explicit
        // absolute offset keeps it out of the DG, crush, and ETK stages entirely; two Philox
        // rounds per attempt are far cheaper than carrying the state across all of them.
        // The offset is the exact position the carried state would have reached: each attempt
        // first skips 100 elements per attempt index, then draws four uniforms for every atom
        // this lane owns.
        std::uint32_t const lane_owned_atoms = (lane < num_atoms)
                                                   ? ((num_atoms - lane + warp_size_k - 1u) / warp_size_k)
                                                   : 0u;

        bool succeeded = false;
        float final_energy = 0.0f;

        for (std::uint32_t attempt = 0; attempt < max_stereo_attempts; ++attempt) {
            unsigned long long const attempt_offset = 50ull * attempt * (attempt + 1ull) +
                                                      4ull * attempt * lane_owned_atoms;
            curandStatePhilox4_32_10_t rng_state;
            curand_init(random_seed, sequence, attempt_offset, &rng_state);
            initialize_from_metric_coordinates_(mol_initial_coordinates, positions, initial_coordinate_stride,
                                                num_atoms, conf, attempt, lane, &rng_state);

            // Distance geometry minimization: Run DG minimization in 4D with up to 100 FIRE iterations.
            distance_geometry_energy_functor_t dg_functor {shared_bounds,
                                                           mol_chirals,
                                                           mol_min,
                                                           inv_scale,
                                                           bounds_row_stride,
                                                           num_atoms,
                                                           meta.num_chiral_centers,
                                                           4,
                                                           1.0f,
                                                           1.0f,
                                                           100.0f,
                                                           0.0f};

            [[maybe_unused]] fire_result_t const dg_result = fire_minimize_(
                positions, velocities, gradient, dimensions_4d, 100, 1e-4f, lane, dg_functor);

            // Stereochemistry validation: Check stereochemistry.
            if (meta.num_chiral_centers > 0) {
                if (!validate_chirality_(positions, num_atoms, mol_chirals, meta.num_chiral_centers, lane)) continue;
            }

            // Fourth-dimension crush: Crush the fourth dimension with 50 FIRE iterations.
            dg_functor.fourth_dimension_penalty_weight = 10.0f;
            fire_result_t const crush_result = fire_minimize_(positions, velocities, gradient, dimensions_4d, 50, 1e-4f,
                                                              lane, dg_functor);

            // Torsion refinement: Run ETK torsion refinement in 3D with 100 FIRE iterations.
            if (meta.num_etk_torsions > 0) {
                etk_torsion_energy_functor_t etk_functor {mol_torsions, mol_constraints, etk_interactions,
                                                          etk_atom_offsets + meta.etk_atom_offsets_start, num_atoms};

                fire_result_t const etk_result = run_fire_stage_(
                    positions, velocities, gradient, dimensions_3d, refinement_policy.etk_initial_iterations,
                    refinement_policy.etk_continuation_iterations, refinement_policy.etk_gradient_threshold, lane,
                    etk_functor);
                final_energy = etk_result.energy;
            }
            else { final_energy = crush_result.energy; }

            // Final validation: Run final validation.
            bool valid = true;
            if (meta.num_chiral_centers > 0) {
                valid = validate_chirality_(positions, num_atoms, mol_chirals, meta.num_chiral_centers, lane);
            }

            if (valid) {
                // ETK-refined 3D positions are the final output of this kernel.
                // The sibling mmff_polish_warp_kernel picks them up from
                // conformer_coordinates_out and runs MMFF94 polish in-place.
                succeeded = true;
                break;
            }
        }

        // Write the conformer output.
        write_conformer_output_(conformer_coordinates_out, conformer_energies_out, conformer_status_out, positions, mol,
                                conf, num_atoms, output_atoms_stride, num_conformers_per_molecule, final_energy,
                                succeeded, lane);
    }
}

/**
 *  Persistent warp-loop MMFF94 polish kernel.
 *
 *  Reads ETK-refined 3D positions from @c conformer_coordinates_out (produced
 *  by @c etkdg_pipeline_warp_kernel on the same stream + wave), runs MMFF94
 *  FIRE with the per-molecule refinement policy, and overwrites the positions
 *  and energy in place.  Skips slots that failed stereo (status != 0) and
 *  molecules with zero MMFF terms.
 *
 *  Shared memory at bin 128 is ~4.5 KB, FIRE scratch only, since no bounds matrix is needed here.
 *  Registers bind before shared memory at every bin.
 *
 *  @b Register-pressure-audit — nvcc 12.8, `cuobjdump --dump-resource-usage`:
 *    sm_80=64, sm_90=72, sm_100=94, sm_120=103 regs/thread, no spills (STACK=0, LOCAL=0).
 *    On sm_90 that is 28 blocks/SM, against the same 32-warp ceiling a 32-thread block imposes,
 *    which is exactly where its sibling sits.
 *
 *    @c bond_distance is read from global memory on every FIRE iteration, N² per iteration.  It
 *    is only ever tested as `< 3`, `== 3`, and `>= 4`, which is two bits per pair, so an N²/4
 *    packed copy staged into shared memory alongside the FIRE arrays would remove that traffic
 *    without displacing anything.
 *
 *  @see etkdg_pipeline_warp_kernel
 */
__global__ __launch_bounds__(32, 4) void mmff_polish_warp_kernel(
    mmff94_atom_vdw_params_t const *__restrict__ mmff_atom_vdw, float const *__restrict__ mmff_atom_charge,
    std::uint8_t const *__restrict__ bond_distance, mmff94_bond_t const *__restrict__ mmff_bond_terms,
    mmff94_angle_t const *__restrict__ mmff_angle_terms, mmff94_torsion_t const *__restrict__ mmff_torsion_terms,
    mmff94_oop_t const *__restrict__ mmff_oop_terms, mmff94_stretchbend_t const *__restrict__ mmff_stretchbend_terms,
    mmff94_atom_interaction_t const *__restrict__ mmff_interactions,
    mmff94_atom_offsets_t const *__restrict__ mmff_atom_offsets,
    etkdg_molecule_metadata_t const *__restrict__ molecule_metadata, float *__restrict__ conformer_coordinates_out,
    float *__restrict__ conformer_energies_out, std::uint8_t const *__restrict__ conformer_status_in,
    std::uint32_t total_molecules, std::uint32_t num_conformers_per_molecule, std::uint32_t output_atoms_stride,
    std::uint32_t bin_max_atoms) {

    std::uint32_t const warp_id = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const grid_stride = gridDim.x;

    // Shared layout (sized at launch by compute_mmff_polish_shared_memory_bytes(bin_max_atoms)):
    //   [positions, velocities, gradient: 3 × 3·bin_max_atoms × float]
    //   [atom_vdw cache: bin_max_atoms × mmff94_atom_vdw_params_t]
    //   [atom_charge cache: bin_max_atoms × float]
    // The caches are loaded once per warp-loop iteration and reused across every
    // FIRE step, eliminating ~2N² duplicate global reads of
    // atom_vdw_params[j] / atom_charge[j] inside the nonbonded inner loop.
    extern __shared__ float shared_float[];
    auto *shared_atom_vdw =
        reinterpret_cast<mmff94_atom_vdw_params_t *>(shared_float + 9u * bin_max_atoms);
    float *shared_atom_charge = reinterpret_cast<float *>(shared_atom_vdw + bin_max_atoms);

    // One warp per (molecule, conformer) pair, matching the sibling kernel's index space so
    // both launches size their grid off the same pair count.
    std::uint32_t const total_pairs = total_molecules * num_conformers_per_molecule;
    for (std::uint32_t pair_index = warp_id; pair_index < total_pairs; pair_index += grid_stride) {
        std::uint32_t const mol = pair_index / num_conformers_per_molecule;
        std::uint32_t const conf = pair_index - mol * num_conformers_per_molecule;

        std::uint32_t const slot = mol * num_conformers_per_molecule + conf;
        if (conformer_status_in[slot] != 0) continue;

        auto const meta = molecule_metadata[mol];
        std::uint32_t const num_atoms = meta.num_atoms;
        std::uint32_t const dimensions_3d = 3 * num_atoms;

        std::uint32_t const total_mmff_terms = total_mmff_terms_(meta);
        if (total_mmff_terms == 0 || mmff_interactions == nullptr) continue;

        auto const refinement_policy = select_refinement_policy_(meta);

        // Stage per-atom params into shmem for this pair.  6 float-words
        // per atom_vdw entry (24 bytes, aligned 8) + 1 word per atom_charge.
        // Iterate as a flat float-words sweep so all lanes participate.
        {
            mmff94_atom_vdw_params_t const *global_vdw = mmff_atom_vdw + meta.mmff_atom_vdw_offset;
            float const *global_vdw_words = reinterpret_cast<float const *>(global_vdw);
            float *shared_vdw_words = reinterpret_cast<float *>(shared_atom_vdw);
            std::uint32_t const vdw_words = num_atoms * (sizeof(mmff94_atom_vdw_params_t) / sizeof(float));
            for (std::uint32_t i = lane; i < vdw_words; i += warp_size_k)
                shared_vdw_words[i] = global_vdw_words[i];
            float const *global_charge = mmff_atom_charge + meta.mmff_atom_charge_offset;
            for (std::uint32_t i = lane; i < num_atoms; i += warp_size_k)
                shared_atom_charge[i] = global_charge[i];
        }
        __syncwarp(full_mask_k);

        mmff94_energy_functor_t mmff_functor {shared_atom_vdw,
                                              shared_atom_charge,
                                              bond_distance + meta.bond_distance_offset,
                                              mmff_bond_terms + meta.mmff_bond_offset,
                                              mmff_angle_terms + meta.mmff_angle_offset,
                                              mmff_torsion_terms + meta.mmff_torsion_offset,
                                              mmff_oop_terms + meta.mmff_oop_offset,
                                              mmff_stretchbend_terms + meta.mmff_stretchbend_offset,
                                              mmff_interactions,
                                              mmff_atom_offsets + meta.mmff_atom_offsets_start,
                                              num_atoms};

        float *positions = shared_float;
        float *velocities = shared_float + dimensions_3d;
        float *gradient = shared_float + 2 * dimensions_3d;

        std::uint32_t const coord_offset = slot * output_atoms_stride * 3;

        // Stage in ETK-refined positions from global memory (SoA layout).
        for (std::uint32_t d = 0; d < 3; ++d)
            for (std::uint32_t i = lane; i < num_atoms; i += warp_size_k)
                positions[d * num_atoms + i] = conformer_coordinates_out[coord_offset + d * output_atoms_stride + i];
        __syncwarp(full_mask_k);

        fire_result_t mmff_result = run_fire_stage_(positions, velocities, gradient, dimensions_3d,
                                                    refinement_policy.mmff_initial_iterations,
                                                    refinement_policy.mmff_continuation_iterations,
                                                    refinement_policy.mmff_gradient_threshold, lane, mmff_functor);

        bool const run_heavy_rescue = is_heavy_refinement_case_(meta) &&
                                      (mmff_result.status != 0 ||
                                       mmff_result.energy > heavy_mmff_rescue_threshold_(meta));
        if (run_heavy_rescue) {
            mmff_result = run_fire_stage_(positions, velocities, gradient, dimensions_3d, 160, 160, 2.5e-5f, lane,
                                          mmff_functor);
        }
        float const mmff_energy = mmff_result.energy;

        // Write polished positions + updated energy back.  Status is left
        // untouched: MMFF non-convergence does not mark the conformer failed.
        for (std::uint32_t d = 0; d < 3; ++d)
            for (std::uint32_t i = lane; i < num_atoms; i += warp_size_k)
                conformer_coordinates_out[coord_offset + d * output_atoms_stride + i] = positions[d * num_atoms + i];
        if (lane == 0) conformer_energies_out[slot] = mmff_energy;
        __syncwarp(full_mask_k);
    }
}

/**
 *  @brief Element-wise fp32 → fp16 conversion of the coordinate buffer.
 *
 *  Launched after @c mmff_polish_warp_kernel as the final stream step before D2H.
 *  Halves the device→host bandwidth and removes the host-side @c __float2half loop
 *  that previously ran on the calling thread inside @c generate_conformers.
 *
 *  One thread per element, vector-friendly grid:
 *  @code{.cpp}
 *  std::uint32_t const block = 256;
 *  std::uint32_t const grid = (num_elements + block - 1) / block;
 *  convert_coords_fp32_to_fp16_kernel<<<grid, block, 0, stream>>>(src, dst, num_elements);
 *  @endcode
 */
__global__ void convert_coords_fp32_to_fp16_kernel(float const *src, __half *dst, std::uint32_t num_elements) {
    std::uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) dst[idx] = __float2half_rn(src[idx]);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
