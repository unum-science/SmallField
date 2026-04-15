/**
 *  @file etkdg_pipeline.cuh
 *  @brief Persistent warp-loop CUDA kernel for the full ETKDG conformer generation pipeline.
 *
 *  Stages per molecule:
 *   2: load pre-smoothed bounds from global memory
 *   3: native metric embedding with mirrored retries
 *   4: distance geometry minimization (4D FIRE, atom-centric)
 *   5: stereochemistry check (chiral volume signs)
 *   6: fourth-dimension crush (FIRE with w₄ penalty)
 *   7: ETK torsion refinement (3D FIRE, atom-centric)
 *   8: final validation
 *   9: MMFF94 polish
 *
 *  Architecture:
 *   - warp-loop: gridDim warps, each loops over molecules with stride
 *   - atom-centric: no atomicAdd, no pair index scan
 *   - half-precision bounds: halves shared memory, enables 2 blocks/SM at N=128
 *   - Philox RNG: stateless counter-based, ~100 cycles init
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#if defined(__CUDACC__)

#include <cuda_fp16.h>
#include <curand_kernel.h>

#include "smallfield/cuda/dg_kernel.cuh"
#include "smallfield/cuda/etk_kernel.cuh"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/cuda/fire_warp.cuh"
#include "smallfield/cuda/mmff94_energy_kernel.cuh"

namespace smallfield::cuda {
using namespace smallfield::api;
using namespace smallfield::mmff94;
using namespace smallfield::parameters;
using namespace smallfield::preprocess;
using namespace smallfield::types;

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

template <typename EnergyGradientFunctor>
__device__ __forceinline__ void run_fire_stage_(float *positions, float *velocities, float *gradient,
                                                std::uint32_t num_elements, std::int32_t initial_iterations,
                                                std::int32_t continuation_iterations, float gradient_threshold,
                                                std::uint32_t lane, EnergyGradientFunctor &&compute_energy_gradient,
                                                float *out_final_energy, std::int32_t *out_iterations,
                                                std::int32_t *out_status, float *out_final_rms_gradient) {
    float stage_energy = 0.0f;
    float stage_rms_gradient = 0.0f;
    std::int32_t stage_iterations = 0;
    std::int32_t stage_status = 0;

    fire_minimize_(positions, velocities, gradient, num_elements, initial_iterations, gradient_threshold, lane,
                   compute_energy_gradient, &stage_energy, &stage_iterations, &stage_status, &stage_rms_gradient);

    std::int32_t total_iterations = stage_iterations;

    bool needs_continuation = false;
    if (lane == 0)
        needs_continuation = continuation_iterations > 0 &&
                             (stage_status != 0 || stage_rms_gradient > gradient_threshold);
    needs_continuation = __shfl_sync(full_mask, needs_continuation ? 1 : 0, 0) != 0;

    if (needs_continuation) {
        float continuation_energy = 0.0f;
        float continuation_rms_gradient = 0.0f;
        std::int32_t continuation_iterations_used = 0;
        std::int32_t continuation_status = 0;

        fire_minimize_(positions, velocities, gradient, num_elements, continuation_iterations, gradient_threshold, lane,
                       compute_energy_gradient, &continuation_energy, &continuation_iterations_used,
                       &continuation_status, &continuation_rms_gradient);

        stage_energy = continuation_energy;
        stage_rms_gradient = continuation_rms_gradient;
        stage_status = continuation_status;
        total_iterations += continuation_iterations_used;
    }

    if (lane == 0) {
        *out_final_energy = stage_energy;
        *out_iterations = total_iterations;
        *out_status = stage_status;
        if (out_final_rms_gradient != nullptr) *out_final_rms_gradient = stage_rms_gradient;
    }
}

/** Compute shared memory bytes for the pipeline kernel. */
__host__ __device__ __forceinline__ std::uint32_t compute_pipeline_shared_memory_bytes(std::uint32_t max_atoms,
                                                                                       std::uint32_t /*unused*/) {
    std::uint32_t const dim4 = 4 * max_atoms;
    std::uint32_t const bounds_bytes = max_atoms * max_atoms * sizeof(__half);
    std::uint32_t const aligned_bounds_bytes = align_up_(bounds_bytes, alignof(float));
    std::uint32_t const fire_arrays = 3 * dim4 * sizeof(float);
    return aligned_bounds_bytes + fire_arrays;
}

__device__ __forceinline__ void initialize_from_metric_coordinates_(float const *initial_coordinates, float *positions,
                                                                    std::uint32_t initial_coordinate_stride,
                                                                    std::uint32_t num_atoms, std::uint32_t conf,
                                                                    std::uint32_t attempt, std::uint32_t lane,
                                                                    curandStatePhilox4_32_10_t *rng_state) {
    float const xyz_jitter = 0.02f + 0.01f * static_cast<float>((conf + attempt) & 3u);
    float const w_jitter = 0.02f + 0.01f * static_cast<float>(attempt & 3u);
    float const x_reflection = (attempt & 1u) ? -1.0f : 1.0f;
    for (std::uint32_t atom = lane; atom < num_atoms; atom += warp_size) {
        for (std::uint32_t dim = 0; dim < 4; ++dim) {
            float base = initial_coordinates[dim * initial_coordinate_stride + atom];
            if (dim == 0) base *= x_reflection;
            float const jitter = (dim < 3) ? xyz_jitter : w_jitter;
            positions[dim * num_atoms + atom] = base + (curand_uniform(rng_state) - 0.5f) * jitter;
        }
    }
    __syncwarp(full_mask);
}

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
        for (std::uint32_t i = lane; i < num_atoms; i += warp_size)
            coordinates_out[coord_offset + d * output_stride + i] = positions[d * num_atoms + i];
    if (lane == 0) {
        energies_out[slot] = energy;
        status_out[slot] = succeeded ? 0 : 1;
    }
    __syncwarp(full_mask);
}

/**
 *  Persistent warp-loop ETKDG pipeline kernel.
 *
 *  Launch: <<<min(total_molecules, max_concurrent_warps), 32, shared_bytes>>>
 *  Each warp loops over molecules with stride = gridDim.x.
 *
 *  `__launch_bounds__(32, 4)` hints the compiler to fit at least 4 blocks per
 *  SM — 1 warp × 32 threads × 4 blocks = 128 warps/SM of occupancy — which
 *  improves latency hiding on the long_scoreboard / wait stalls that dominate
 *  this kernel per ncu.  Without the hint, nvcc picks register count first
 *  (~128 regs/thread here) which caps concurrency at 1 block/SM.  The cap at
 *  4 blocks/SM — raised from an earlier 2 blocks/SM setting — lets nvcc spill
 *  more registers to recover 4x occupancy. The trade-off is extra spill
 *  traffic to local memory; on Blackwell's large register file and L1 this
 *  has consistently been a net win on drug-like ChEMBL molecules.
 */
__global__ __launch_bounds__(32, 4) void etkdg_pipeline_warp_kernel(
    float const *__restrict__ bounds_matrices, float const *__restrict__ initial_coordinates,
    chiral_center_descriptor_t const *__restrict__ chiral_centers,
    etk_torsion_descriptor_t const *__restrict__ torsion_terms,
    distance_constraint_t const *__restrict__ distance_constraints,
    etk_atom_interaction_t const *__restrict__ etk_interactions,
    etk_atom_offsets_t const *__restrict__ etk_atom_offsets, mmff94_atom_vdw_params_t const *__restrict__ mmff_atom_vdw,
    float const *__restrict__ mmff_atom_charge, std::uint8_t const *__restrict__ bond_distance,
    mmff94_bond_t const *__restrict__ mmff_bond_terms, mmff94_angle_t const *__restrict__ mmff_angle_terms,
    mmff94_torsion_t const *__restrict__ mmff_torsion_terms, mmff94_oop_t const *__restrict__ mmff_oop_terms,
    mmff94_stretchbend_t const *__restrict__ mmff_stretchbend_terms,
    mmff94_atom_interaction_t const *__restrict__ mmff_interactions,
    mmff94_atom_offsets_t const *__restrict__ mmff_atom_offsets,
    etkdg_molecule_metadata_t const *__restrict__ molecule_metadata, float *__restrict__ conformer_coordinates_out,
    float *__restrict__ conformer_energies_out, std::uint8_t *__restrict__ conformer_status_out,
    std::uint32_t total_molecules, std::uint32_t num_conformers_per_molecule, std::uint32_t max_stereo_attempts,
    std::uint32_t max_atoms_per_molecule, std::uint32_t initial_coordinate_stride, std::uint32_t output_atoms_stride,
    std::uint64_t random_seed) {

    std::uint32_t const warp_id = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const grid_stride = gridDim.x;

    // Layout shared memory as [bounds matrix: N² halves] [positions, velocities, gradient: 3 * 4N floats].
    extern __shared__ char shared_raw[];
    std::uint32_t const max_n = max_atoms_per_molecule;
    __half *shared_bounds = reinterpret_cast<__half *>(shared_raw);
    std::uint32_t const bounds_bytes = max_n * max_n * sizeof(__half);
    std::uint32_t const aligned_bounds_bytes = align_up_(bounds_bytes, alignof(float));
    float *shared_float = reinterpret_cast<float *>(shared_raw + aligned_bounds_bytes);

    // Process molecules in a warp loop with gridDim.x stride.
    for (std::uint32_t mol = warp_id; mol < total_molecules; mol += grid_stride) {
        auto const meta = molecule_metadata[mol];
        auto const refinement_policy = select_refinement_policy_(meta);
        std::uint32_t const num_atoms = meta.num_atoms;
        std::uint32_t const dim4 = 4 * num_atoms;
        std::uint32_t const dim3 = 3 * num_atoms;

        // FIRE shared memory: positions + velocities + gradient (3 arrays).
        std::uint32_t off = 0;
        float *positions = shared_float + off;
        off += dim4;
        float *velocities = shared_float + off;
        off += dim4;
        float *gradient = shared_float + off;

        // Stage 2: Load pre-smoothed bounds and convert to half (vectorized half2 stores).
        std::uint32_t const bounds_size = num_atoms * num_atoms;
        float const *global_bounds = bounds_matrices + meta.bounds_offset;
        std::uint32_t const bounds_pairs = bounds_size / 2;
        for (std::uint32_t i = lane; i < bounds_pairs; i += warp_size) {
            __half2 packed = __floats2half2_rn(global_bounds[2 * i], global_bounds[2 * i + 1]);
            reinterpret_cast<__half2 *>(shared_bounds)[i] = packed;
        }
        if (lane == 0 && (bounds_size & 1))
            shared_bounds[bounds_size - 1] = __float2half(global_bounds[bounds_size - 1]);
        __syncwarp(full_mask);

        // Build molecule-level pointers into the batched arrays.
        float const *mol_initial_coordinates = initial_coordinates + mol * initial_coordinate_stride * 4;
        chiral_center_descriptor_t const *mol_chirals = chiral_centers + meta.chiral_offset;
        etk_torsion_descriptor_t const *mol_torsions = torsion_terms + meta.torsion_offset;
        distance_constraint_t const *mol_constraints = distance_constraints + meta.constraint_offset;

        // Initialize the Philox RNG once per molecule.
        curandStatePhilox4_32_10_t rng_state;
        curand_init(random_seed, mol * 32 + lane, 0, &rng_state);

        // Generate conformers for this molecule.
        for (std::uint32_t conf = 0; conf < num_conformers_per_molecule; ++conf) {
            bool succeeded = false;
            float final_energy = 0.0f;

            for (std::uint32_t attempt = 0; attempt < max_stereo_attempts; ++attempt) {
                skipahead(static_cast<unsigned long long>(conf * 1000 + attempt * 100), &rng_state);
                initialize_from_metric_coordinates_(mol_initial_coordinates, positions, initial_coordinate_stride,
                                                    num_atoms, conf, attempt, lane, &rng_state);

                // Stage 4: Run DG minimization in 4D with up to 100 FIRE iterations.
                distance_geometry_energy_functor_t dg_functor {
                    shared_bounds, mol_chirals, num_atoms, meta.num_chiral_centers, 4, 1.0f, 1.0f, 100.0f, 0.0f};

                float dg_energy = 0.0f;
                float dg_rms_gradient = 0.0f;
                std::int32_t dg_iterations = 0, dg_status = 0;
                fire_minimize_(positions, velocities, gradient, dim4, 100, 1e-4f, lane, dg_functor, &dg_energy,
                               &dg_iterations, &dg_status, &dg_rms_gradient);

                // Stage 5: Check stereochemistry.
                if (meta.num_chiral_centers > 0) {
                    if (!validate_chirality_(positions, num_atoms, mol_chirals, meta.num_chiral_centers, lane))
                        continue;
                }

                // Stage 6: Crush the fourth dimension with 50 FIRE iterations.
                dg_functor.fourth_dimension_penalty_weight = 10.0f;
                float crush_energy = 0.0f;
                float crush_rms_gradient = 0.0f;
                std::int32_t crush_iterations = 0, crush_status = 0;
                fire_minimize_(positions, velocities, gradient, dim4, 50, 1e-4f, lane, dg_functor, &crush_energy,
                               &crush_iterations, &crush_status, &crush_rms_gradient);

                // Stage 7: Run ETK torsion refinement in 3D with 100 FIRE iterations.
                if (meta.num_etk_torsions > 0) {
                    etk_torsion_energy_functor_t etk_functor {mol_torsions, mol_constraints, etk_interactions,
                                                              etk_atom_offsets + meta.etk_atom_offsets_start,
                                                              num_atoms};

                    float etk_energy = 0.0f;
                    float etk_rms_gradient = 0.0f;
                    std::int32_t etk_iterations = 0, etk_status = 0;
                    run_fire_stage_(positions, velocities, gradient, dim3, refinement_policy.etk_initial_iterations,
                                    refinement_policy.etk_continuation_iterations,
                                    refinement_policy.etk_gradient_threshold, lane, etk_functor, &etk_energy,
                                    &etk_iterations, &etk_status, &etk_rms_gradient);
                    final_energy = etk_energy;
                }
                else { final_energy = crush_energy; }

                // Stage 8: Run final validation.
                bool valid = true;
                if (meta.num_chiral_centers > 0) {
                    valid = validate_chirality_(positions, num_atoms, mol_chirals, meta.num_chiral_centers, lane);
                }

                if (valid) {
                    // Stage 9: Run MMFF94 polishing in 3D.
                    std::uint32_t const total_mmff_terms = total_mmff_terms_(meta);
                    if (total_mmff_terms > 0 && mmff_interactions != nullptr) {
                        mmff94_energy_functor_t mmff_functor {mmff_atom_vdw + meta.mmff_atom_vdw_offset,
                                                              mmff_atom_charge + meta.mmff_atom_charge_offset,
                                                              bond_distance + meta.bond_distance_offset,
                                                              mmff_bond_terms + meta.mmff_bond_offset,
                                                              mmff_angle_terms + meta.mmff_angle_offset,
                                                              mmff_torsion_terms + meta.mmff_torsion_offset,
                                                              mmff_oop_terms + meta.mmff_oop_offset,
                                                              mmff_stretchbend_terms + meta.mmff_stretchbend_offset,
                                                              mmff_interactions,
                                                              mmff_atom_offsets + meta.mmff_atom_offsets_start,
                                                              num_atoms};

                        float mmff_energy = 0.0f;
                        float mmff_rms_gradient = 0.0f;
                        std::int32_t mmff_iterations = 0, mmff_status = 0;
                        run_fire_stage_(
                            positions, velocities, gradient, dim3, refinement_policy.mmff_initial_iterations,
                            refinement_policy.mmff_continuation_iterations, refinement_policy.mmff_gradient_threshold,
                            lane, mmff_functor, &mmff_energy, &mmff_iterations, &mmff_status, &mmff_rms_gradient);

                        bool run_heavy_rescue = false;
                        if (lane == 0) {
                            run_heavy_rescue = is_heavy_refinement_case_(meta) &&
                                               (mmff_status != 0 || mmff_energy > heavy_mmff_rescue_threshold_(meta));
                        }
                        run_heavy_rescue = __shfl_sync(full_mask, run_heavy_rescue ? 1 : 0, 0) != 0;

                        if (run_heavy_rescue) {
                            float rescue_energy = 0.0f;
                            float rescue_rms_gradient = 0.0f;
                            std::int32_t rescue_iterations = 0;
                            std::int32_t rescue_status = 0;
                            run_fire_stage_(positions, velocities, gradient, dim3, 160, 160, 2.5e-5f, lane,
                                            mmff_functor, &rescue_energy, &rescue_iterations, &rescue_status,
                                            &rescue_rms_gradient);
                            mmff_energy = rescue_energy;
                            mmff_iterations += rescue_iterations;
                            mmff_status = rescue_status;
                            mmff_rms_gradient = rescue_rms_gradient;
                        }

                        final_energy = mmff_energy;
                    }

                    succeeded = true;
                    break;
                }
            }

            // Write the conformer output.
            write_conformer_output_(conformer_coordinates_out, conformer_energies_out, conformer_status_out, positions,
                                    mol, conf, num_atoms, output_atoms_stride, num_conformers_per_molecule,
                                    final_energy, succeeded, lane);
        }
    }
}

} // namespace smallfield::cuda

#endif // __CUDACC__
