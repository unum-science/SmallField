/**
 *  @file src/smallfield/cuda/etkdg_pipeline.cuh
 *  @brief Persistent warp-loop CUDA kernels for the ETKDG conformer generation pipeline.
 *
 *  The pipeline lives in two sibling kernels that share the `warp_per_molecule` pattern, split to keep register
 *  pressure per kernel under the launch-bounds cap. What each costs is a measurement of today's binary rather
 *  than a rule, so it is not restated here — read it from `cuobjdump --dump-resource-usage`.
 *
 *  `etkdg_pipeline_warp_kernel` — bounds quantization through final validation, writing ETK-refined 3D
 *  positions to `conformer_coordinates_out`.
 *
 *  `mmff_polish_warp_kernel` — MMFF94 relaxation under one Barzilai-Borwein call. Reads what the first kernel
 *  wrote to `conformer_coordinates_out`, overwrites it with polished coordinates, and replaces the ETK energy
 *  in `conformer_energies_out`. Skips slots with a non-zero status — a failed stereo check — or no MMFF terms.
 *
 *  `stamp_refused_molecules_kernel` — one thread per molecule, last of the three on the status buffer and the
 *  only one speaking for the metric prepass: it restamps every slot whose bounds no coordinates satisfy.
 *
 *  Stages per molecule, across both kernels:
 *   - bounds quantization: load pre-smoothed bounds from global memory, int8-quantize to shared
 *   - metric embedding: native metric embedding with mirrored retries (separate prepass kernel)
 *   - distance geometry minimization: 4D FIRE, atom-centric
 *   - stereochemistry validation: chiral volume signs
 *   - fourth-dimension crush: FIRE with w₄ penalty
 *   - torsion refinement: ETK 3D FIRE, atom-centric
 *   - final validation: post-refinement stereochemistry check
 *   - MMFF94 polish: in the `mmff_polish_warp_kernel` sibling
 *
 *  Architecture: a warp loop of gridDim warps, each striding over (molecule, conformer) pairs, over
 *  atom-centric kernels with no atomicAdd and no pair index scan, reading per-molecule linearly-quantized
 *  int8 bounds at one byte per pair, seeded by stateless counter-based Philox at ~100 cycles per init.
 */

#pragma once

#if defined(__CUDACC__)

#include <cfloat>
#include <cuda_fp16.h>
#include <curand_kernel.h>

#include "smallfield/cuda/dg_kernel.cuh"
#include "smallfield/cuda/etk_kernel.cuh"
#include "smallfield/types/conformer_status.hpp"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/views.hpp"
#include "smallfield/cuda/fire_warp.cuh"
#include "smallfield/cuda/mmff94_energy_kernel.cuh"
#include "smallfield/preprocess/molecule_complexity.hpp"
#include "smallfield/utilities.hpp"

namespace smallfield::cuda {
using smallfield::preprocess::is_heavy_refinement_case;
using smallfield::preprocess::total_mmff_terms;
using smallfield::types::chiral_center_descriptor_t;
using smallfield::types::conformer_status_t;
using smallfield::types::distance_constraint_t;
using smallfield::types::etk_atom_interaction_t;
using smallfield::types::etk_atom_offsets_t;
using smallfield::types::etk_torsion_descriptor_t;
using smallfield::types::etkdg_molecule_metadata_t;
using smallfield::types::mmff94_angle_t;
using smallfield::types::mmff94_atom_interaction_t;
using smallfield::types::mmff94_atom_offsets_t;
using smallfield::types::mmff94_atom_vdw_folded_t;
using smallfield::types::mmff94_atom_vdw_params_t;
using smallfield::types::mmff94_bond_t;
using smallfield::types::mmff94_oop_t;
using smallfield::types::mmff94_stretchbend_t;
using smallfield::types::mmff94_torsion_t;

/**
 *  FIRE iterations the two four-dimensional stages get. Flat rather than per-molecule: what they buy is a
 *  better starting geometry for the reflection-blind torsion stage, not a function of molecule size the way
 *  the polish budget next door is.
 *
 *  Named rather than inlined because the trade reverses with the arm: they pay for themselves wherever the CPU
 *  fallback is on the critical path, a refusal costing the CPU embedder hundreds of milliseconds against
 *  microseconds here, and the arithmetic goes the other way once the GPU is critical and refusals are rare.
 *  `BENCH.md`, _The Distance-Geometry Budget Is Not Free, And Still Worth Raising_, carries what doubling
 *  them costs and buys.
 */
constexpr std::int32_t distance_geometry_iterations_k = 200;
constexpr std::int32_t fourth_dimension_crush_iterations_k = 100;

/** Per-molecule FIRE iteration budget for the ETK torsion refinement stage. */
struct refinement_policy_t {
    std::int32_t etk_initial_iterations;
    std::int32_t etk_continuation_iterations;
    float etk_gradient_threshold;
};

/**
 *  Energy evaluations one conformer's MMFF94 polish may spend, and the RMS gradient that ends it early.
 *
 *  One cap and one threshold rather than a tier ladder: `total_mmff_terms` counts the N(N-1)/2
 *  nonbonded pairs, so any boundary drawn on it is a pure atom count, and no atom-count split
 *  separates this corpus, whose medians run 27 to 52 atoms with hydrogens.
 *
 *  The cap is explicit rather than a `while (!converged)` because the grid the polish launches under exceeds
 *  the slot count at every captured bin: each block draws exactly one slot, so the kernel's wall time is the
 *  single worst slot's, and a per-slot stopping rule prices the maximum rather than the mean.
 *
 *  The threshold is an absolute RMS over all 3N components, which is not RDKit's scaled test and does not
 *  convert into it. Its floor is what this evaluator can resolve: against RDKit's double gradient at a
 *  converged geometry over 227 drug-like molecules the fp32 noise sits at 1.7e-5 median and 4.7e-5 at p90, so
 *  anything tighter reports `exhausted_k` about the constant rather than the geometry. RDKit's own minimizer
 *  lands at 3.7e-4 on the same molecules, so 5.0e-4 stays an order of magnitude above the floor and still
 *  tighter than the reference reaches.
 */
constexpr std::int32_t mmff_polish_iteration_cap_k = 800;
constexpr float mmff_polish_gradient_threshold_k = 5.0e-4f;

constexpr refinement_policy_t select_refinement_policy_(etkdg_molecule_metadata_t const &meta) {
    std::uint32_t const mmff_terms = total_mmff_terms(meta);
    std::uint32_t const torsion_complexity = meta.num_etk_torsions + meta.num_mmff_torsions;

    if (meta.num_atoms <= 20 && mmff_terms <= 96 && torsion_complexity <= 12) return {60, 0, 1.5e-4f};

    if (is_heavy_refinement_case(meta)) return {160, 80, 7.5e-5f};

    return {100, 40, 1.0e-4f};
}

template <std::size_t dimensions_, typename energy_gradient_functor_>
__device__ __forceinline__ fire_result_t run_fire_stage_(fire_scratch<dimensions_> scratch,
                                                         std::int32_t initial_iterations,
                                                         std::int32_t continuation_iterations, float gradient_threshold,
                                                         std::uint32_t lane,
                                                         energy_gradient_functor_ &&compute_energy_gradient) {
    fire_result_t result = fire_minimize_(scratch, initial_iterations, gradient_threshold, lane,
                                          compute_energy_gradient);

    // The status already answers this: it is `exhausted_k` exactly when the RMS gradient at the returned
    // positions is still above the threshold, so testing the float beside it asks one fact twice, off a stale norm.
    bool const needs_continuation = continuation_iterations > 0 && result.status == fire_status_t::exhausted_k;
    if (needs_continuation) {
        std::int32_t const initial_iterations_done = result.iterations;
        result = fire_minimize_(scratch, continuation_iterations, gradient_threshold, lane, compute_energy_gradient);
        result.iterations += initial_iterations_done;
    }
    return result;
}

/** Compute shared memory bytes for the DG+ETK pipeline kernel.
 *
 *  Layout: [bounds matrix: N × `compute_bounds_row_stride(N)` × uint8_t]
 *          [positions, velocities, gradient: 3 × 4N × float].
 *  Bounds are int8-quantized with a per-molecule min/inv_scale pair in registers, so the matrix costs half
 *  what a `__half` layout would.
 *
 *  @see compute_mmff_polish_shared_memory_bytes
 */
constexpr std::uint32_t compute_pipeline_shared_memory_bytes(std::uint32_t max_atoms) {
    std::uint32_t const dimensions_4d = 4 * max_atoms;
    std::uint32_t const bounds_bytes = max_atoms * compute_bounds_row_stride(max_atoms) * sizeof(std::uint8_t);
    std::uint32_t const aligned_bounds_bytes = align_up_(bounds_bytes, alignof(float));
    std::uint32_t const fire_arrays = 3 * dimensions_4d * sizeof(float);
    return aligned_bounds_bytes + fire_arrays;
}

/** Compute shared memory bytes for the MMFF polish kernel.
 *
 *  Layout: [positions, velocities, gradient: 3 × 3N × float]
 *          [atom_vdw cache: N × mmff94_atom_vdw_folded_t]
 *          [atom_charge cache: N × float].
 *  No bounds matrix — MMFF94 uses explicit bonded and nonbonded term lists rather than a pair-distance
 *  matrix. At N=128 that is 4.5 KB of minimizer scratch and 2.5 KB of caches, and the caches spare ~2N²
 *  duplicate global reads of `atom_vdw_params[j]` / `atom_charge[j]` per minimizer step.
 *
 *  @see compute_pipeline_shared_memory_bytes
 */
constexpr std::uint32_t compute_mmff_polish_shared_memory_bytes(std::uint32_t max_atoms) {
    std::uint32_t const dimensions_3d = 3 * max_atoms;
    std::uint32_t const fire_bytes = 3 * dimensions_3d * sizeof(float);
    std::uint32_t const atom_vdw_bytes = max_atoms * static_cast<std::uint32_t>(
                                                         sizeof(smallfield::types::mmff94_atom_vdw_folded_t));
    std::uint32_t const atom_charge_bytes = max_atoms * static_cast<std::uint32_t>(sizeof(float));
    return fire_bytes + atom_vdw_bytes + atom_charge_bytes;
}

/**
 *  @brief Load 4D metric-embedding coordinates into shared-memory positions.
 *
 *  Called once at the start of each stereo attempt inside the DG+ETK pipeline kernel. It adds jitter whose
 *  magnitude follows @p conformer and reflects the x-axis on odd @p attempt, filling @p positions — 4 ×
 *  @p num_atoms shared floats — from the global embedding held SoA at @p initial_coordinate_stride, with
 *  @p rng_state supplying the uniform draws.
 *
 *  @see etkdg_pipeline_warp_kernel
 */
__device__ __forceinline__ void initialize_from_metric_coordinates_(float const *initial_coordinates, float *positions,
                                                                    std::uint32_t initial_coordinate_stride,
                                                                    std::uint32_t num_atoms, std::uint32_t conformer,
                                                                    std::uint32_t attempt, std::uint32_t lane,
                                                                    curandStatePhilox4_32_10_t *rng_state) {
    float const xyz_jitter = 0.02f + 0.01f * static_cast<float>((conformer + attempt) & 3u);
    float const w_jitter = 0.02f + 0.01f * static_cast<float>(attempt & 3u);
    float const x_reflection = (attempt & 1u) ? -1.0f : 1.0f;
    for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k) {
        for (std::uint32_t dimension = 0; dimension < 4; ++dimension) {
            float base = initial_coordinates[dimension * initial_coordinate_stride + atom];
            if (dimension == 0) base *= x_reflection;
            float const jitter = (dimension < 3) ? xyz_jitter : w_jitter;
            positions[dimension * num_atoms + atom] = base + (curand_uniform(rng_state) - 0.5f) * jitter;
        }
    }
    __syncwarp(full_mask_k);
}

/**
 *  @brief Copy one conformer's 3D positions from shared memory to the global output array.
 *
 *  Writes the slot's @p energy and @p verdict too, the verdict reaching the host unchanged. @p positions is
 *  3 × @p num_atoms shared floats packed at @p num_atoms, while the global arrays run at @p output_stride,
 *  which may differ; the flat slot is @p molecule_index × @p num_conformers plus the conformer index.
 *
 *  @see etkdg_pipeline_warp_kernel
 *  @see mmff_polish_warp_kernel
 */
__device__ __forceinline__ void write_conformer_output_(float *coordinates_out, float *energies_out,
                                                        std::uint8_t *status_out, float const *positions,
                                                        std::uint32_t molecule_index, std::uint32_t conformer_index,
                                                        std::uint32_t num_atoms, std::uint32_t output_stride,
                                                        std::uint32_t num_conformers, float energy,
                                                        conformer_status_t verdict, std::uint32_t lane) {
    std::uint32_t const slot = molecule_index * num_conformers + conformer_index;
    std::uint32_t const coord_offset = slot * output_stride * 3;
    // Positions use SoA layout with num_atoms stride, while output uses output_stride.
    for (std::uint32_t dimension = 0; dimension < 3; ++dimension)
        for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k)
            coordinates_out[coord_offset + dimension * output_stride + atom] = positions[dimension * num_atoms + atom];
    if (lane == 0) {
        energies_out[slot] = energy;
        status_out[slot] = static_cast<std::uint8_t>(verdict);
    }
    __syncwarp(full_mask_k);
}

/**
 *  @brief Persistent warp-loop ETKDG kernel covering bounds quantization through final validation.
 *
 *  Launch: <<<min(total_molecules × num_conformers, max_concurrent_warps), 32, shared_bytes>>>
 *  Each warp loops over (molecule, conformer) pairs with stride = gridDim.x.
 *
 *  The unit of work is a pair rather than a molecule because the atom-count bins hold wildly different
 *  molecule counts: at a 50 K wave over the PubChem distribution, four of the eight bins hold fewer molecules
 *  than the resident-warp cap, so their launches cannot fill the device. Multiplying the index space by the
 *  conformer count fills them, at the price of quantizing bounds per conformer instead of per molecule.
 *
 *  `__launch_bounds__(32, 4)` is a floor, not a cap: it asks nvcc to keep the kernel fitting at least four
 *  blocks per SM, and at 32 threads that permits far more registers than the hardware allows, so it constrains
 *  nothing here. What bounds occupancy is the register count against the SM register file, and below the
 *  largest bins the 32-blocks-per-SM slot limit. A 32-thread block can never fill more than 32 of the SM's 64
 *  warp slots either way, so 50 % is this shape's ceiling at any register count and a lower count buys little.
 *
 *  Reading a verdict here rather than writing one is what that budget cannot afford. A molecule whose bounds
 *  the metric prepass refused is minimised from its zeroed embedding like any other and lands here as a stereo
 *  failure or a success, because a guard skipping it at the head of the pair loop costs four registers on
 *  `sm_120`. `stamp_refused_molecules_kernel` restamps those slots afterwards, where the fact costs nothing.
 *
 *  Writes ETK-refined 3D positions into `conformer_coordinates_out`, where `mmff_polish_warp_kernel` runs next.
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
    std::uint64_t random_seed) {

    std::uint32_t const warp_id = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const grid_stride = gridDim.x;

    // Layout shared memory as [bounds matrix: N² uint8_t] [positions, velocities, gradient: 3 × 4N floats].
    extern __shared__ char shared_raw[];
    std::uint32_t const max_n = max_atoms_per_molecule;
    std::uint8_t *shared_bounds = reinterpret_cast<std::uint8_t *>(shared_raw);
    std::uint32_t const bounds_bytes = max_n * compute_bounds_row_stride(max_n) * sizeof(std::uint8_t);
    std::uint32_t const aligned_bounds_bytes = align_up_(bounds_bytes, alignof(float));
    float *shared_float = reinterpret_cast<float *>(shared_raw + aligned_bounds_bytes);

    // Process (molecule, conformer) pairs in a warp loop with gridDim.x stride. The flat pair
    // index decomposes into the bin-relative molecule and the conformer slot within it.
    std::uint32_t const total_pairs = total_molecules * num_conformers_per_molecule;
    for (std::uint32_t pair_index = warp_id; pair_index < total_pairs; pair_index += grid_stride) {
        std::uint32_t const molecule = pair_index / num_conformers_per_molecule;
        std::uint32_t const conformer = pair_index - molecule * num_conformers_per_molecule;

        auto const meta = molecule_metadata[molecule];
        auto const refinement_policy = select_refinement_policy_(meta);
        std::uint32_t const num_atoms = meta.num_atoms;
        std::uint32_t const dimensions_4d = 4 * num_atoms;

        // FIRE shared memory: positions, velocities, gradient. All three are carved at the four-dimensional
        // pitch and viewed twice — four rows while the fourth dimension is live, the same first three after.
        std::uint32_t shared_offset = 0;
        float *positions = shared_float + shared_offset;
        shared_offset += dimensions_4d;
        float *velocities = shared_float + shared_offset;
        shared_offset += dimensions_4d;
        float *gradient = shared_float + shared_offset;

        fire_scratch<4> const scratch_4d {.positions = fire_positions<4> {positions, num_atoms},
                                          .velocities = fire_velocities<4> {velocities, num_atoms},
                                          .gradient = fire_gradient<4> {gradient, num_atoms}};
        fire_scratch<3> const scratch_3d {.positions = fire_positions<3> {positions, num_atoms},
                                          .velocities = fire_velocities<3> {velocities, num_atoms},
                                          .gradient = fire_gradient<3> {gradient, num_atoms}};

        // The min/max scan must run on the post-smoothed bounds the metric prepass wrote back at
        // `meta.bounds_offset`; a host-side precompute would scan the un-smoothed matrix and never invert.
        std::uint32_t const bounds_size = num_atoms * num_atoms;
        float const *global_bounds = bounds_matrices + meta.bounds_offset;

        float lane_min = FLT_MAX;
        float lane_max = -FLT_MAX;
        for (std::uint32_t entry_index = lane; entry_index < bounds_size; entry_index += warp_size_k) {
            float const value = global_bounds[entry_index];
            lane_min = fminf(lane_min, value);
            lane_max = fmaxf(lane_max, value);
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
        float const *mol_initial_coordinates = initial_coordinates + molecule * initial_coordinate_stride * 4;
        chiral_center_descriptor_t const *mol_chirals = chiral_centers + meta.chiral_offset;
        etk_torsion_descriptor_t const *mol_torsions = torsion_terms + meta.torsion_offset;
        distance_constraint_t const *mol_constraints = distance_constraints + meta.constraint_offset;

        // Seed Philox from the caller's own row, which `molecule_input_indices` holds. Everything between the
        // caller and here reorders molecules — preprocess distributes across host threads on a dynamic schedule
        // and atom-count-sorts, the dispatcher length-sorts the input and splits it into waves — so a fixed
        // `random_seed` reproduces the same conformers only if the stream is keyed to something none of that
        // moves. The conformer index enters the stream too, keeping one molecule's conformers independent.
        std::uint64_t const caller_row = molecule_input_indices[molecule];
        std::uint64_t const sequence = (caller_row * num_conformers_per_molecule + conformer) * warp_size_k + lane;

        // Rebuilding it at the head of every attempt from an explicit absolute offset keeps it out of the DG,
        // crush and ETK stages entirely. The offset is where the carried state would have reached: 100 elements
        // per attempt index, then four uniforms for every atom this lane owns.
        std::uint32_t const lane_owned_atoms = (lane < num_atoms) ? divide_round_up_(num_atoms - lane, warp_size_k)
                                                                  : 0u;

        conformer_status_t verdict = conformer_status_t::stereo_failure_k;
        float final_energy = 0.0f;

        for (std::uint32_t attempt = 0; attempt < max_stereo_attempts; ++attempt) {
            unsigned long long const attempt_offset = 50ull * attempt * (attempt + 1ull) +
                                                      4ull * attempt * lane_owned_atoms;
            curandStatePhilox4_32_10_t rng_state;
            curand_init(random_seed, sequence, attempt_offset, &rng_state);
            initialize_from_metric_coordinates_(mol_initial_coordinates, positions, initial_coordinate_stride,
                                                num_atoms, conformer, attempt, lane, &rng_state);

            // Distance geometry minimization, in four dimensions.
            distance_geometry_energy_functor_t dg_functor {.bounds_matrix = shared_bounds,
                                                           .chiral_centers = mol_chirals,
                                                           .mol_min = mol_min,
                                                           .inv_scale = inv_scale,
                                                           .bounds_row_stride = bounds_row_stride,
                                                           .num_atoms = num_atoms,
                                                           .num_chiral_centers = meta.num_chiral_centers,
                                                           .spatial_dimension = 4,
                                                           .upper_violation_weight = 1.0f,
                                                           .lower_violation_weight = 1.0f,
                                                           .chiral_volume_weight = 100.0f,
                                                           .fourth_dimension_penalty_weight = 0.0f};

            [[maybe_unused]] fire_result_t const dg_result = fire_minimize_(scratch_4d, distance_geometry_iterations_k,
                                                                            1e-4f, lane, dg_functor);

            if (count_invalid_chiral_centers_(positions, num_atoms, mol_chirals, meta.num_chiral_centers, lane) != 0)
                continue;

            // Fourth-dimension crush, under the w₄ penalty.
            dg_functor.fourth_dimension_penalty_weight = 10.0f;
            fire_result_t const crush_result = fire_minimize_(scratch_4d, fourth_dimension_crush_iterations_k, 1e-4f,
                                                              lane, dg_functor);

            // ETK torsion refinement, in three dimensions, on the policy's budget.
            if (meta.num_etk_torsions > 0) {
                etk_torsion_energy_functor_t etk_functor {mol_torsions, mol_constraints, etk_interactions,
                                                          etk_atom_offsets + meta.atom_base_offset, num_atoms};

                fire_result_t const etk_result = run_fire_stage_(
                    scratch_3d, refinement_policy.etk_initial_iterations, refinement_policy.etk_continuation_iterations,
                    refinement_policy.etk_gradient_threshold, lane, etk_functor);
                final_energy = etk_result.energy;
            }
            else { final_energy = crush_result.energy; }

            // Final validation. The torsion functor carries only 1-2 and 1-3 distances, a set exactly invariant
            // under reflection at a stereocenter, so the refinement above is free to invert a center the crush
            // had correct — which is what this catches. Re-entering the distance geometry functor in three
            // dimensions repairs most of the rest, for eight more bytes of stack on both Blackwell targets
            // against two centers in twenty-five hundred; `validate_polished_stereo_kernel` refuses instead.
            if (count_invalid_chiral_centers_(positions, num_atoms, mol_chirals, meta.num_chiral_centers, lane) != 0)
                continue;

            verdict = conformer_status_t::success_k;
            break;
        }

        write_conformer_output_(conformer_coordinates_out, conformer_energies_out, conformer_status_out, positions,
                                molecule, conformer, num_atoms, output_atoms_stride, num_conformers_per_molecule,
                                final_energy, verdict, lane);
    }
}

#pragma region Polish Diagnostics

/**
 *  @brief Which of `mmff_polish_warp_kernel`'s four exits a conformer slot left through.
 *
 *  Three of the four ship a geometry the MMFF94 minimizer never improved, and nothing in the status
 *  byte or the energy tells them apart — the byte is reserved for verdicts the host classifies and
 *  the energy is written on every path.
 */
enum class polish_path_t : std::int32_t {
    /** The grid never drew this slot, or the diagnostics array was not primed. */
    unreached_k = 0,
    /** Skipped at the status gate: an earlier stage had already failed the conformer. */
    status_skipped_k = 1,
    /** Skipped for an empty term list: the molecule has no MMFF94 energy to minimize. */
    no_terms_k = 2,
    /** Relaxed, then declined — the minimizer finished above what it was handed, so ETK coordinates ship. */
    write_back_declined_k = 3,
    /** Relaxed and kept. */
    write_back_accepted_k = 4,
};

/** Exits `polish_path_t` names, counted off its last enumerator so a histogram declares its width once. */
constexpr std::size_t polish_path_count_k = static_cast<std::size_t>(polish_path_t::write_back_accepted_k) + 1u;

/** What each of the floats `mmff_polish_warp_kernel` writes per slot into its diagnostics array holds. */
enum class polish_diagnostic_t : std::uint32_t {
    /** A `polish_path_t`, widened to float. */
    path_k = 0,
    /** RMS gradient at the geometry the minimizer returned, whether or not that geometry was kept. */
    rms_gradient_k = 1,
    /** The step length the minimizer last used — the stall detector. */
    final_timestep_k = 2,
    /** Energy evaluations the polish spent on this slot. */
    iterations_k = 3,
    /** A `fire_status_t`, widened to float. */
    fire_status_k = 4,
    /** MMFF94 energy of the ETK-refined geometry as staged, before the minimizer moved anything. */
    staged_energy_k = 5,
};

/** Floats per conformer slot in `mmff_polish_warp_kernel`'s diagnostics array, counted off the enum. */
constexpr std::uint32_t polish_diagnostic_floats_k = static_cast<std::uint32_t>(polish_diagnostic_t::staged_energy_k) +
                                                     1u;

/** One conformer slot's row of `polish_diagnostic_floats_k` floats, unpacked. */
struct polish_row_t {
    polish_path_t path = polish_path_t::unreached_k;
    float rms_gradient = 0.0f;
    float final_timestep = 0.0f;
    float iterations = 0.0f;
    fire_status_t status = fire_status_t::converged_k;
    float staged_energy = 0.0f;
};

/**
 *  @brief Read slot @p slot out of a staged diagnostics array, host-side.
 *
 *  Lives beside the enums it indexes so no reader re-derives the row layout; @p diagnostics must span
 *  `polish_diagnostic_floats_k` floats per slot, which is what the kernel's own writes assume.
 */
inline polish_row_t decode_polish_row(custd::span<float const> diagnostics, std::size_t slot) noexcept {
    float const *const row = diagnostics.data() + slot * polish_diagnostic_floats_k;
    auto const field = [row](polish_diagnostic_t which) { return row[static_cast<std::uint32_t>(which)]; };
    return {.path = static_cast<polish_path_t>(static_cast<int>(field(polish_diagnostic_t::path_k))),
            .rms_gradient = field(polish_diagnostic_t::rms_gradient_k),
            .final_timestep = field(polish_diagnostic_t::final_timestep_k),
            .iterations = field(polish_diagnostic_t::iterations_k),
            .status = static_cast<fire_status_t>(static_cast<int>(field(polish_diagnostic_t::fire_status_k))),
            .staged_energy = field(polish_diagnostic_t::staged_energy_k)};
}

/** Write one diagnostic field for @p slot, from lane 0, doing nothing when no array was handed in. */
__device__ __forceinline__ void store_polish_diagnostic_(float *diagnostics_out, std::uint32_t slot,
                                                         polish_diagnostic_t field, float value, std::uint32_t lane) {
    if (diagnostics_out == nullptr || lane != 0) return;
    diagnostics_out[slot * polish_diagnostic_floats_k + static_cast<std::uint32_t>(field)] = value;
}

/** Record the four fields a finished minimization stage reports, for @p slot, from lane 0. */
__device__ __forceinline__ void store_polish_stage_(float *diagnostics_out, std::uint32_t slot,
                                                    fire_result_t const &result, std::uint32_t lane) {
    if (diagnostics_out == nullptr || lane != 0) return;
    float *const row = diagnostics_out + slot * polish_diagnostic_floats_k;
    row[static_cast<std::uint32_t>(polish_diagnostic_t::rms_gradient_k)] = result.rms_gradient;
    row[static_cast<std::uint32_t>(polish_diagnostic_t::final_timestep_k)] = result.final_timestep;
    row[static_cast<std::uint32_t>(polish_diagnostic_t::iterations_k)] = static_cast<float>(result.iterations);
    row[static_cast<std::uint32_t>(polish_diagnostic_t::fire_status_k)] = static_cast<float>(result.status);
}

#pragma endregion Polish Diagnostics

/**
 *  @brief Persistent warp-loop MMFF94 polish kernel.
 *
 *  Runs on the same stream and wave as `etkdg_pipeline_warp_kernel`, over the buffer that kernel wrote.
 *
 *  `chiral_centers` keeps the declared stereochemistry. Published MMFF94 has no chiral term, so its
 *  energy cannot tell a declared center from its mirror and the minimizer descends into the reflected
 *  basin, often the lower one. `mmff94_energy_functor_t` carries a flat-bottomed signed-volume restraint
 *  so wrong handedness is a hill rather than a verdict read off the geometry afterwards; refusing the
 *  write-back instead ships the raw embedding, an order of magnitude worse at p90. It holds both signed
 *  volumes a center carries, since `validate_polished_stereo_kernel` reads both and they are reachable
 *  apart — `BENCH.md`, _Restraining Only One Of The Two Volumes_.
 *
 *  Barzilai-Borwein rather than FIRE: one timestep shared by every degree of freedom settles at the
 *  stiffest bond stretch's stability limit. `barzilai_borwein_minimize_` carries the numbers.
 *
 *  Shared memory at bin 128 is 7 KB — minimizer scratch and two per-atom caches, no bounds matrix. On
 *  `sm_90` neither it nor the register count binds, leaving the 32-block slot limit of a 32-thread
 *  block; on `sm_120`'s 100 KB shared memory binds first, so scratch added here spends occupancy from
 *  its first byte, and `ROADMAP.md` measures an ablation that does. The second launch bound binds too,
 *  unlike its sibling's: 20 blocks at 32 threads asks `ptxas` for at most 102 registers, above what any
 *  architecture reaches alone, so it states the Blackwell budget and costs `sm_90` nothing. For today's
 *  per-architecture cost read `cuobjdump --dump-resource-usage` on an all-`-real` build.
 *
 *  `polish_diagnostics_out` spans `polish_diagnostic_floats_k` floats per slot and all four exits write
 *  their `polish_path_t`; `unreached_k` is what a primed array keeps where no exit ran. A null array
 *  declines the telemetry and carries no default, so each launch states which it wants.
 *
 *  @see etkdg_pipeline_warp_kernel
 */
__global__ __launch_bounds__(32, 20) void mmff_polish_warp_kernel(
    mmff94_atom_vdw_params_t const *__restrict__ mmff_atom_vdw, float const *__restrict__ mmff_atom_charge,
    std::uint8_t const *__restrict__ bond_distance, mmff94_bond_t const *__restrict__ mmff_bond_terms,
    mmff94_angle_t const *__restrict__ mmff_angle_terms, mmff94_torsion_t const *__restrict__ mmff_torsion_terms,
    mmff94_oop_t const *__restrict__ mmff_oop_terms, mmff94_stretchbend_t const *__restrict__ mmff_stretchbend_terms,
    mmff94_atom_interaction_t const *__restrict__ mmff_interactions,
    mmff94_atom_offsets_t const *__restrict__ mmff_atom_offsets,
    chiral_center_descriptor_t const *__restrict__ chiral_centers,
    etkdg_molecule_metadata_t const *__restrict__ molecule_metadata, float *__restrict__ conformer_coordinates_out,
    float *__restrict__ conformer_energies_out, std::uint8_t const *__restrict__ conformer_status_in,
    std::uint32_t total_molecules, std::uint32_t num_conformers_per_molecule, std::uint32_t output_atoms_stride,
    std::uint32_t bin_max_atoms, float *__restrict__ polish_diagnostics_out) {

    std::uint32_t const warp_id = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const grid_stride = gridDim.x;

    // `compute_mmff_polish_shared_memory_bytes` sizes this block at launch from `bin_max_atoms`, and
    // the three pointers below carve it in the order that function assumes.
    extern __shared__ float shared_float[];
    auto *shared_atom_vdw = reinterpret_cast<mmff94_atom_vdw_folded_t *>(shared_float + 9u * bin_max_atoms);
    float *shared_atom_charge = reinterpret_cast<float *>(shared_atom_vdw + bin_max_atoms);

    // One warp per (molecule, conformer) pair, matching the sibling kernel's index space so
    // both launches size their grid off the same pair count.
    std::uint32_t const total_pairs = total_molecules * num_conformers_per_molecule;
    for (std::uint32_t pair_index = warp_id; pair_index < total_pairs; pair_index += grid_stride) {
        std::uint32_t const molecule = pair_index / num_conformers_per_molecule;
        std::uint32_t const conformer = pair_index - molecule * num_conformers_per_molecule;

        std::uint32_t const slot = molecule * num_conformers_per_molecule + conformer;
        if (conformer_status_in[slot] != 0) {
            store_polish_diagnostic_(polish_diagnostics_out, slot, polish_diagnostic_t::path_k,
                                     static_cast<float>(polish_path_t::status_skipped_k), lane);
            continue;
        }

        auto const meta = molecule_metadata[molecule];
        std::uint32_t const num_atoms = meta.num_atoms;
        std::uint32_t const dimensions_3d = 3 * num_atoms;

        std::uint32_t const mmff_terms = total_mmff_terms(meta);
        if (mmff_terms == 0 || mmff_interactions == nullptr) {
            store_polish_diagnostic_(polish_diagnostics_out, slot, polish_diagnostic_t::path_k,
                                     static_cast<float>(polish_path_t::no_terms_k), lane);
            continue;
        }

        // Stage per-atom params into shmem for this pair, folding the pair-invariant half of the
        // combining rules on the way in so the minimizer never re-derives it.
        {
            mmff94_atom_vdw_params_t const *global_vdw = mmff_atom_vdw + meta.atom_base_offset;
            for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k)
                shared_atom_vdw[atom] = fold_atom_vdw_params_(global_vdw[atom]);
            float const *global_charge = mmff_atom_charge + meta.atom_base_offset;
            for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k)
                shared_atom_charge[atom] = global_charge[atom];
        }
        __syncwarp(full_mask_k);

        mmff94_energy_functor_t const mmff_functor {
            .atom_vdw_params = shared_atom_vdw,
            .atom_charge = shared_atom_charge,
            .bond_distance = bond_distance + meta.bond_distance_offset,
            .bond_params = mmff_bond_terms + meta.mmff_bond_offset,
            .angle_params = mmff_angle_terms + meta.mmff_angle_offset,
            .torsion_params = mmff_torsion_terms + meta.mmff_torsion_offset,
            .oop_params = mmff_oop_terms + meta.mmff_oop_offset,
            .stretchbend_params = mmff_stretchbend_terms + meta.mmff_stretchbend_offset,
            .interactions = mmff_interactions,
            .atom_offsets = mmff_atom_offsets + meta.atom_base_offset,
            .chiral_centers = chiral_centers + meta.chiral_offset,
            .num_atoms = num_atoms,
            .num_chiral_centers = meta.num_chiral_centers};

        float *positions = shared_float;
        fire_scratch<3> const scratch {.positions = fire_positions<3> {positions, num_atoms},
                                       .velocities = fire_velocities<3> {shared_float + dimensions_3d, num_atoms},
                                       .gradient = fire_gradient<3> {shared_float + 2 * dimensions_3d, num_atoms}};

        std::uint32_t const coord_offset = slot * output_atoms_stride * 3;

        // Stage in ETK-refined positions from global memory (SoA layout).
        for (std::uint32_t dimension = 0; dimension < 3; ++dimension)
            for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k)
                positions[dimension * num_atoms + atom] =
                    conformer_coordinates_out[coord_offset + dimension * output_atoms_stride + atom];
        __syncwarp(full_mask_k);

        // The MMFF energy of the geometry as staged, before anything moves — what decides below whether the
        // polish is worth keeping. Parked in the output slot rather than a register: holding it across the
        // minimizer costs `sm_90` five registers, 32 resident blocks against 29, and the slot is written anyway.
        {
            float const staged_energy = mmff_functor(positions, shared_float + 2 * dimensions_3d, dimensions_3d, lane);
            if (lane == 0) conformer_energies_out[slot] = staged_energy;
            store_polish_diagnostic_(polish_diagnostics_out, slot, polish_diagnostic_t::staged_energy_k, staged_energy,
                                     lane);
        }
        __syncwarp(full_mask_k);

        fire_result_t const mmff_result = barzilai_borwein_minimize_(
            scratch, mmff_polish_iteration_cap_k, mmff_polish_gradient_threshold_k, lane, mmff_functor);
        store_polish_stage_(polish_diagnostics_out, slot, mmff_result, lane);
        float const mmff_energy = mmff_result.energy;

        // Write polished positions and the updated energy back, but only if the polish improved on what it was
        // handed: Barzilai-Borwein is deliberately non-monotone — the secant step is read off the last
        // displacement, not off a line search — so a call can finish above where it started, and shipping that
        // would be a regression nothing else would see. Handedness is not tested here, because the restraint
        // above already carries it: an inverted center costs energy, so this compares two geometries scored on
        // the same terms, and `validate_polished_stereo_kernel` re-reads the geometry afterwards and reports a
        // center the restraint failed to hold. The fallback geometry needs no buffer, since
        // `conformer_coordinates_out` is what this kernel staged from and has not touched, so declining leaves
        // the ETK-refined coordinates in place. Status is untouched either way: MMFF non-convergence does not
        // mark the conformer failed.
        bool const keep_polish = mmff_energy < conformer_energies_out[slot];
        if (keep_polish) {
            for (std::uint32_t dimension = 0; dimension < 3; ++dimension)
                for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size_k)
                    conformer_coordinates_out[coord_offset + dimension * output_atoms_stride + atom] =
                        positions[dimension * num_atoms + atom];
            if (lane == 0) conformer_energies_out[slot] = mmff_energy;
        }
        store_polish_diagnostic_(polish_diagnostics_out, slot, polish_diagnostic_t::path_k,
                                 static_cast<float>(keep_polish ? polish_path_t::write_back_accepted_k
                                                                : polish_path_t::write_back_declined_k),
                                 lane);
        __syncwarp(full_mask_k);
    }
}

/**
 *  @brief Replace the status of every conformer slot whose molecule the metric prepass refused.
 *
 *  One thread per molecule, launched after `mmff_polish_warp_kernel` so it is the last word on the status
 *  buffer. `etkdg_pipeline_warp_kernel` cannot leave that word itself: knowing costs registers it lacks.
 *
 *  @param[in] molecule_verdicts What the prepass decided per molecule, `success_k` where it has nothing to say.
 *  @param[out] conformer_status_out Per-slot statuses the two warp kernels have finished writing.
 *  @param[in] total_molecules Molecules the two arrays cover.
 *  @param[in] num_conformers_per_molecule Slots each molecule owns in @p conformer_status_out.
 *
 *  @see etkdg_pipeline_warp_kernel
 */
__global__ void stamp_refused_molecules_kernel(conformer_status_t const *__restrict__ molecule_verdicts,
                                               std::uint8_t *__restrict__ conformer_status_out,
                                               std::uint32_t total_molecules,
                                               std::uint32_t num_conformers_per_molecule) {
    std::uint32_t const molecule = blockIdx.x * blockDim.x + threadIdx.x;
    if (molecule >= total_molecules) return;
    conformer_status_t const verdict = molecule_verdicts[molecule];
    if (verdict == conformer_status_t::success_k) return;
    for (std::uint32_t conformer = 0; conformer < num_conformers_per_molecule; ++conformer)
        conformer_status_out[molecule * num_conformers_per_molecule + conformer] = static_cast<std::uint8_t>(verdict);
}

/**
 *  @brief Element-wise fp32 → fp16 conversion of the coordinate buffer.
 *
 *  Launched after `mmff_polish_warp_kernel` as the final stream step before D2H, halving the device→host
 *  bandwidth against shipping fp32 coordinates. One thread per element, vector-friendly grid:
 *  @code{.cpp}
 *  std::uint32_t const block = 256;
 *  std::uint32_t const grid = divide_round_up_(num_elements, block);
 *  convert_coords_fp32_to_fp16_kernel<<<grid, block, 0, stream>>>(source, destination, num_elements);
 *  @endcode
 */
__global__ void convert_coords_fp32_to_fp16_kernel(float const *source, __half *destination,
                                                   std::uint32_t num_elements) {
    std::uint32_t const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) destination[index] = __float2half_rn(source[index]);
}

} // namespace smallfield::cuda

#endif // __CUDACC__
