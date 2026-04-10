/**
 * @file etkdg_pipeline.cuh
 * @brief Persistent warp-loop CUDA kernel for the full ETKDG conformer generation pipeline.
 *
 * Stages per molecule:
 *   2: triangle smoothing (Floyd-Warshall on half-precision bounds)
 *   3: random 4D coordinate generation (Philox RNG)
 *   4: distance geometry minimization (4D L-BFGS, atom-centric)
 *   5: stereochemistry check (chiral volume signs)
 *   6: fourth-dimension crush (L-BFGS with w₄ penalty)
 *   7: ETK torsion refinement (3D L-BFGS, atom-centric)
 *   8: final validation
 *
 * Architecture:
 *   - warp-loop: gridDim warps, each loops over molecules with stride
 *   - atom-centric: no atomicAdd, no pair index scan
 *   - half-precision bounds: halves shared memory, enables 2 blocks/SM at N=128
 *   - Philox RNG: stateless counter-based, ~100 cycles init
 */

#pragma once

#if defined(__CUDACC__)

#include <cuda_fp16.h>
#include <curand_kernel.h>

#include "dg_forcefield.cuh"
#include "etk_forcefield.cuh"
#include "etkdg_types.cuh"
#include "fire_warp.cuh"
#include "mmff94_functor.cuh"
#include "triangle_smooth.cuh"

namespace smallfield {

/** Compute shared memory bytes for the pipeline kernel (FIRE optimizer). */
__host__ __device__ __forceinline__ std::uint32_t compute_pipeline_shared_memory_bytes(std::uint32_t max_atoms,
                                                                                       std::uint32_t /*unused*/) {
    std::uint32_t const dim4 = 4 * max_atoms;
    // bounds in half precision: N² × 2 bytes
    std::uint32_t const bounds_bytes = max_atoms * max_atoms * sizeof(__half);
    // FIRE arrays in float: positions + velocities + gradient (3 arrays, no history)
    std::uint32_t const fire_arrays = 3 * dim4 * sizeof(float);
    return bounds_bytes + fire_arrays;
}

// stage 3: generate random 4D coordinates scaled by molecular diameter

__device__ __forceinline__ void generate_random_coordinates_(__half const* bounds_half, float* positions,
                                                             std::uint32_t num_atoms, std::uint32_t lane,
                                                             curandStatePhilox4_32_10_t* rng_state) {
    // estimate diameter from max upper bound
    float max_upper = 0.0f;
    for (std::uint32_t idx = lane; idx < num_atoms * num_atoms; idx += warp_size) {
        std::uint32_t const i = idx / num_atoms;
        std::uint32_t const j = idx % num_atoms;
        if (i < j) {
            float const val = __half2float(bounds_half[i * num_atoms + j]);
            max_upper = fmaxf(max_upper, val);
        }
    }
    // warp max reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        max_upper = fmaxf(max_upper, __shfl_xor_sync(full_mask, max_upper, offset));

    float const scale = max_upper * 0.5f;

    for (std::uint32_t i = lane; i < num_atoms; i += warp_size) {
        for (std::uint32_t d = 0; d < 4; ++d)
            positions[d * num_atoms + i] = (curand_uniform(rng_state) - 0.5f) * scale;
    }
    __syncwarp(full_mask);
}

// stage 9: write conformer output to global memory

__device__ __forceinline__ void write_conformer_output_(float* coordinates_out, float* energies_out,
                                                        std::uint8_t* status_out, float const* positions,
                                                        std::uint32_t mol_index, std::uint32_t conformer_index,
                                                        std::uint32_t num_atoms, std::uint32_t output_stride,
                                                        std::uint32_t num_conformers, float energy, bool succeeded,
                                                        std::uint32_t lane) {
    std::uint32_t const slot = mol_index * num_conformers + conformer_index;
    std::uint32_t const coord_offset = slot * output_stride * 3;
    // positions are SoA with num_atoms stride, output uses output_stride
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
 * Persistent warp-loop ETKDG pipeline kernel.
 *
 * Launch: <<<min(total_molecules, max_concurrent_warps), 32, shared_bytes>>>
 * Each warp loops over molecules with stride = gridDim.x.
 */
__global__ void etkdg_pipeline_warp_kernel(
    float const* __restrict__ bounds_matrices, chiral_center_descriptor_t const* __restrict__ chiral_centers,
    etk_torsion_descriptor_t const* __restrict__ torsion_terms,
    distance_constraint_t const* __restrict__ distance_constraints,
    // MMFF94 parameter arrays + per-atom interaction lists (stage 9, global memory)
    mmff94_nonbonded_pair_t const* __restrict__ mmff_nonbonded_pairs, mmff94_bond_t const* __restrict__ mmff_bond_terms,
    mmff94_angle_t const* __restrict__ mmff_angle_terms, mmff94_torsion_t const* __restrict__ mmff_torsion_terms,
    mmff94_atom_interaction_t const* __restrict__ mmff_interactions,
    mmff94_atom_offsets_t const* __restrict__ mmff_atom_offsets,
    etkdg_molecule_metadata_t const* __restrict__ molecule_metadata, float* __restrict__ conformer_coordinates_out,
    float* __restrict__ conformer_energies_out, std::uint8_t* __restrict__ conformer_status_out,
    std::uint32_t total_molecules, std::uint32_t num_conformers_per_molecule, std::uint32_t max_stereo_attempts,
    std::uint32_t max_atoms_per_molecule, std::uint32_t output_atoms_stride, std::uint64_t random_seed) {

    std::uint32_t const warp_id = blockIdx.x;
    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const grid_stride = gridDim.x;

    // shared memory partitioning
    extern __shared__ char shared_raw[];
    std::uint32_t const max_n = max_atoms_per_molecule;

    __half* shared_bounds = reinterpret_cast<__half*>(shared_raw);
    std::uint32_t float_offset = (max_n * max_n * sizeof(__half) + sizeof(float) - 1) / sizeof(float);
    float* shared_float = reinterpret_cast<float*>(shared_raw) + float_offset;

    // warp-loop: each warp processes molecules with stride
    for (std::uint32_t mol = warp_id; mol < total_molecules; mol += grid_stride) {
        auto const meta = molecule_metadata[mol];
        std::uint32_t const num_atoms = meta.num_atoms;
        std::uint32_t const dim4 = 4 * num_atoms;
        std::uint32_t const dim3 = 3 * num_atoms;

        // FIRE shared memory: positions + velocities + gradient (3 arrays)
        std::uint32_t off = 0;
        float* positions = shared_float + off;
        off += dim4;
        float* velocities = shared_float + off;
        off += dim4;
        float* gradient = shared_float + off;

        // stage 2: load bounds (float → half) and smooth
        std::uint32_t const bounds_size = num_atoms * num_atoms;
        float const* global_bounds = bounds_matrices + meta.bounds_offset;
        for (std::uint32_t i = lane; i < bounds_size; i += warp_size)
            shared_bounds[i] = __float2half(global_bounds[i]);
        __syncwarp(full_mask);

        triangle_smooth_warp_half_(shared_bounds, num_atoms, lane);

        // molecule-level pointers into batched arrays
        chiral_center_descriptor_t const* mol_chirals = chiral_centers + meta.chiral_offset;
        etk_torsion_descriptor_t const* mol_torsions = torsion_terms + meta.torsion_offset;
        distance_constraint_t const* mol_constraints = distance_constraints + meta.constraint_offset;

        // Philox RNG: init once per molecule
        curandStatePhilox4_32_10_t rng_state;
        curand_init(random_seed, mol * 32 + lane, 0, &rng_state);

        // conformer generation loop
        for (std::uint32_t conf = 0; conf < num_conformers_per_molecule; ++conf) {
            bool succeeded = false;
            float final_energy = 0.0f;

            for (std::uint32_t attempt = 0; attempt < max_stereo_attempts; ++attempt) {
                // stage 3: random 4D coordinates
                skipahead(static_cast<unsigned long long>(conf * 1000 + attempt * 100), &rng_state);
                generate_random_coordinates_(shared_bounds, positions, num_atoms, lane, &rng_state);

                // stage 4: DG minimization (4D, FIRE, 100 iterations max)
                distance_geometry_energy_functor_t dg_functor{
                    shared_bounds, mol_chirals, num_atoms, meta.num_chiral_centers, 4, 1.0f, 1.0f, 100.0f, 0.0f};

                float dg_energy = 0.0f;
                std::int32_t dg_iterations = 0, dg_status = 0;
                fire_minimize_(positions, velocities, gradient, dim4, 100, 1e-4f, lane, dg_functor, &dg_energy,
                               &dg_iterations, &dg_status);

                // stage 5: stereochemistry check
                if (meta.num_chiral_centers > 0) {
                    if (!validate_chirality_(positions, num_atoms, mol_chirals, meta.num_chiral_centers, lane))
                        continue;
                }

                // stage 6: fourth-dimension crush (50 iterations)
                dg_functor.fourth_dimension_penalty_weight = 10.0f;
                float crush_energy = 0.0f;
                std::int32_t crush_iterations = 0, crush_status = 0;
                fire_minimize_(positions, velocities, gradient, dim4, 50, 1e-4f, lane, dg_functor, &crush_energy,
                               &crush_iterations, &crush_status);

                // stage 7: ETK torsion refinement (3D, 100 iterations)
                if (meta.num_etk_torsions > 0) {
                    etk_torsion_energy_functor_t etk_functor{mol_torsions, mol_constraints, meta.num_etk_torsions,
                                                             meta.num_distance_constraints, num_atoms};

                    float etk_energy = 0.0f;
                    std::int32_t etk_iterations = 0, etk_status = 0;
                    fire_minimize_(positions, velocities, gradient, dim3, 100, 1e-4f, lane, etk_functor, &etk_energy,
                                   &etk_iterations, &etk_status);
                    final_energy = etk_energy;
                } else {
                    final_energy = crush_energy;
                }

                // stage 8: final validation
                bool valid = true;
                if (meta.num_chiral_centers > 0) {
                    valid = validate_chirality_(positions, num_atoms, mol_chirals, meta.num_chiral_centers, lane);
                }

                if (valid) {
                    // stage 9: MMFF94 polish (3D, FIRE, 100 iterations)
                    if (meta.num_mmff_nonbonded > 0 && mmff_interactions != nullptr) {
                        // debug: verify stage 9 data (remove after debugging)
                        // if (lane == 0 && mol == 0 && conf == 0) { ... }
                        mmff94_energy_functor_t mmff_functor{mmff_nonbonded_pairs + meta.mmff_nonbonded_offset,
                                                             mmff_bond_terms + meta.mmff_bond_offset,
                                                             mmff_angle_terms + meta.mmff_angle_offset,
                                                             mmff_torsion_terms + meta.mmff_torsion_offset,
                                                             mmff_interactions,
                                                             mmff_atom_offsets + meta.mmff_atom_offsets_start,
                                                             num_atoms};

                        float mmff_energy = 0.0f;
                        std::int32_t mmff_iterations = 0, mmff_status = 0;
                        fire_minimize_(positions, velocities, gradient, dim3, 200, 1e-4f, lane, mmff_functor,
                                       &mmff_energy, &mmff_iterations, &mmff_status);
                        final_energy = mmff_energy;
                    }

                    succeeded = true;
                    break;
                }
            }

            // write output
            write_conformer_output_(conformer_coordinates_out, conformer_energies_out, conformer_status_out, positions,
                                    mol, conf, num_atoms, output_atoms_stride, num_conformers_per_molecule,
                                    final_energy, succeeded, lane);
        }
    }
}

} // namespace smallfield

#endif // __CUDACC__
