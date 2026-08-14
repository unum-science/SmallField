/**
 *  @file test/cuda/test_mmff_gradient_kernels.cu
 *  @brief CUDA kernel wrapper for MMFF94 energy and gradient tests.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::uint32_t`

#include <string_view> // `std::string_view`

#include <cuda_runtime.h>

#include "smallfield/cuda/mmff94_energy_kernel.cuh"
#include "smallfield/types/storage.hpp"

#include "test/cuda/test_mmff_gradient_kernels.hpp"

namespace sfc = smallfield::cuda;
namespace sft = smallfield::types;

namespace {

__global__ void mmff_energy_eval_kernel(
    float const *positions, float *gradient, sft::mmff94_atom_vdw_params_t const *atom_vdw, float const *atom_charge,
    std::uint8_t const *bond_distance, sft::mmff94_bond_t const *bond_params, sft::mmff94_angle_t const *angle_params,
    sft::mmff94_torsion_t const *torsion_params, sft::mmff94_oop_t const *oop_params,
    sft::mmff94_stretchbend_t const *stretchbend_params, sft::mmff94_atom_interaction_t const *interactions,
    sft::mmff94_atom_offsets_t const *atom_offsets, std::uint32_t num_atoms, float *out_energy) {

    std::uint32_t const lane = threadIdx.x;

    // The pair loop reads the folded table, and the polish kernel folds while it stages, so the
    // harness stages the same way rather than folding on the host under different rounding.
    extern __shared__ sft::mmff94_atom_vdw_folded_t shared_atom_vdw[];
    for (std::uint32_t atom = lane; atom < num_atoms; atom += sfc::warp_size_k)
        shared_atom_vdw[atom] = sfc::fold_atom_vdw_params_(atom_vdw[atom]);
    __syncwarp(sfc::full_mask_k);

    sfc::mmff94_energy_functor_t functor {shared_atom_vdw, atom_charge,    bond_distance, bond_params,
                                          angle_params,    torsion_params, oop_params,    stretchbend_params,
                                          interactions,    atom_offsets,   num_atoms};

    float const energy = functor(positions, gradient, 3 * num_atoms, lane);
    if (lane == 0) *out_energy = energy;
}

/**
 *  @brief Floats the three minimizer arrays occupy before the folded van der Waals table starts.
 *
 *  `mmff94_atom_vdw_folded_t` is sixteen bytes and eight-byte aligned, and `9 * num_atoms` floats is a
 *  multiple of eight bytes only for an even atom count -- the polish kernel carves at the bin's atom cap,
 *  which is even, while this harness carves at the molecule's own count and would otherwise hand the pair
 *  sweep a misaligned load on every odd molecule.
 */
constexpr std::uint32_t minimizer_array_floats_(std::uint32_t num_atoms) { return (9u * num_atoms + 3u) & ~3u; }

/**
 *  One warp relaxing one molecule, staged as `mmff_polish_warp_kernel` stages it: the three minimizer
 *  arrays first in the dynamic block, the folded van der Waals table behind them.
 */
__global__ void mmff_minimize_kernel(float *positions_inout, sft::mmff94_atom_vdw_params_t const *atom_vdw,
                                     float const *atom_charge, std::uint8_t const *bond_distance,
                                     sft::mmff94_bond_t const *bond_params, sft::mmff94_angle_t const *angle_params,
                                     sft::mmff94_torsion_t const *torsion_params, sft::mmff94_oop_t const *oop_params,
                                     sft::mmff94_stretchbend_t const *stretchbend_params,
                                     sft::mmff94_atom_interaction_t const *interactions,
                                     sft::mmff94_atom_offsets_t const *atom_offsets, std::uint32_t num_atoms,
                                     std::int32_t max_iterations, float gradient_threshold,
                                     smallfield::tests::mmff_minimize_result_t *out_result) {

    std::uint32_t const lane = threadIdx.x;
    std::uint32_t const num_elements = 3 * num_atoms;

    extern __shared__ float shared_float[];
    auto *shared_atom_vdw = reinterpret_cast<sft::mmff94_atom_vdw_folded_t *>(shared_float +
                                                                              minimizer_array_floats_(num_atoms));
    for (std::uint32_t atom = lane; atom < num_atoms; atom += sfc::warp_size_k)
        shared_atom_vdw[atom] = sfc::fold_atom_vdw_params_(atom_vdw[atom]);
    for (std::uint32_t element = lane; element < num_elements; element += sfc::warp_size_k)
        shared_float[element] = positions_inout[element];
    __syncwarp(sfc::full_mask_k);

    sfc::mmff94_energy_functor_t functor {shared_atom_vdw, atom_charge,    bond_distance, bond_params,
                                          angle_params,    torsion_params, oop_params,    stretchbend_params,
                                          interactions,    atom_offsets,   num_atoms};

    sfc::fire_scratch<3> const scratch {.positions = sfc::fire_positions<3> {shared_float, num_atoms},
                                        .velocities = sfc::fire_velocities<3> {shared_float + num_elements, num_atoms},
                                        .gradient = sfc::fire_gradient<3> {shared_float + 2 * num_elements, num_atoms}};

    sfc::fire_result_t const result = sfc::barzilai_borwein_minimize_(scratch, max_iterations, gradient_threshold, lane,
                                                                      functor);

    for (std::uint32_t element = lane; element < num_elements; element += sfc::warp_size_k)
        positions_inout[element] = shared_float[element];
    if (lane == 0)
        *out_result = {.energy = result.energy,
                       .rms_gradient = result.rms_gradient,
                       .final_step = result.final_timestep,
                       .iterations = result.iterations,
                       .status = result.status};
}

} // namespace

namespace smallfield::tests {

/** Grow-only device storage over the legacy default stream, which is the one this evaluation rides. */
template <typename value_type_>
using device_array = sft::buffer<value_type_, sft::device_allocator_t>;

/** Every device array one energy evaluation stages, released in stream order however the call leaves. */
struct gradient_arena_t {
    device_array<float> positions;
    device_array<float> gradient;
    device_array<float> energy;
    device_array<mmff_minimize_result_t> minimize_result;
    device_array<sft::mmff94_atom_vdw_params_t> atom_vdw;
    device_array<float> atom_charge;
    device_array<std::uint8_t> bond_distance;
    device_array<sft::mmff94_bond_t> bonds;
    device_array<sft::mmff94_angle_t> angles;
    device_array<sft::mmff94_torsion_t> torsions;
    device_array<sft::mmff94_oop_t> oops;
    device_array<sft::mmff94_stretchbend_t> stretchbends;
    device_array<sft::mmff94_atom_interaction_t> interactions;
    device_array<sft::mmff94_atom_offsets_t> atom_offsets;

    /** Reserve every array one molecule of @p num_atoms needs, at the extents @p terms carries. */
    [[nodiscard]] bool resize_for(std::uint32_t num_atoms, mmff_energy_terms_t const &terms) noexcept {
        bool reserved = true;
        auto reserve_ = [&](auto &array, std::size_t count) {
            reserved &= array.try_reserve(count) != sft::reserve_outcome_t::failed_k;
        };
        // A term array may be empty beside a non-zero atom count -- the synthetic single-term fixtures build
        // exactly that -- and the functor is handed every pointer regardless, so each term keeps one slot.
        auto reserve_term_ = [&](auto &array, std::size_t count) { reserve_(array, count == 0 ? 1 : count); };

        std::size_t const num_elements = std::size_t {3} * num_atoms;
        reserve_(positions, num_elements);
        reserve_(gradient, num_elements);
        reserve_(energy, 1);
        reserve_(minimize_result, 1);
        reserve_term_(atom_vdw, terms.atom_vdw.size());
        reserve_term_(atom_charge, terms.atom_charge.size());
        reserve_term_(bond_distance, std::size_t {num_atoms} * num_atoms);
        reserve_term_(bonds, terms.bonds.size());
        reserve_term_(angles, terms.angles.size());
        reserve_term_(torsions, terms.torsions.size());
        reserve_term_(oops, terms.oops.size());
        reserve_term_(stretchbends, terms.stretchbends.size());
        reserve_term_(interactions, terms.interactions.size());
        reserve_term_(atom_offsets, terms.atom_offsets.size());
        return reserved;
    }

    /** Upload every host array onto its device counterpart; an empty one copies nothing. */
    [[nodiscard]] bool upload(custd::span<float const> host_positions, mmff_energy_terms_t const &terms) noexcept {
        custd::span<std::uint8_t const> const hops {terms.bond_distance.data_handle(), terms.bond_distance.size()};
        return sft::upload_into(positions.view(), host_positions, nullptr) &&
               sft::upload_into(atom_vdw.view(), terms.atom_vdw, nullptr) &&
               sft::upload_into(atom_charge.view(), terms.atom_charge, nullptr) &&
               sft::upload_into(bond_distance.view(), hops, nullptr) &&
               sft::upload_into(bonds.view(), terms.bonds, nullptr) &&
               sft::upload_into(angles.view(), terms.angles, nullptr) &&
               sft::upload_into(torsions.view(), terms.torsions, nullptr) &&
               sft::upload_into(oops.view(), terms.oops, nullptr) &&
               sft::upload_into(stretchbends.view(), terms.stretchbends, nullptr) &&
               sft::upload_into(interactions.view(), terms.interactions, nullptr) &&
               sft::upload_into(atom_offsets.view(), terms.atom_offsets, nullptr);
    }
};

std::string_view mmff_energy_probe_reason(mmff_energy_probe_t outcome) noexcept {
    switch (outcome) {
    case mmff_energy_probe_t::allocation_failed_k: return "device_allocation_failed";
    case mmff_energy_probe_t::transfer_failed_k: return "device_transfer_failed";
    case mmff_energy_probe_t::launch_failed_k: return "kernel_launch_failed";
    default: return "completed";
    }
}

mmff_energy_probe_t gpu_mmff_energy(custd::span<float const> host_positions, std::uint32_t num_atoms,
                                    mmff_energy_terms_t const &terms, custd::span<float> host_gradient,
                                    float *out_energy) {

    gradient_arena_t arena;
    if (!arena.resize_for(num_atoms, terms)) return mmff_energy_probe_t::allocation_failed_k;
    if (!arena.upload(host_positions, terms)) return mmff_energy_probe_t::transfer_failed_k;

    std::uint32_t const shared_bytes = num_atoms * static_cast<std::uint32_t>(sizeof(sft::mmff94_atom_vdw_folded_t));
    mmff_energy_eval_kernel<<<1, sfc::warp_size_k, shared_bytes>>>(
        arena.positions.data(), arena.gradient.data(), arena.atom_vdw.data(), arena.atom_charge.data(),
        arena.bond_distance.data(), arena.bonds.data(), arena.angles.data(), arena.torsions.data(), arena.oops.data(),
        arena.stretchbends.data(), arena.interactions.data(), arena.atom_offsets.data(), num_atoms,
        arena.energy.data());
    // A launch the driver refuses to enqueue reports here and nowhere else. The kernel never runs, so the
    // drain below has nothing to fail on, and the gradient still holds whatever the allocation found --
    // which a caller differencing it against its own energies would read as a wrong analytic gradient.
    if (cudaGetLastError() != cudaSuccess) return mmff_energy_probe_t::launch_failed_k;

    custd::span<float const> const device_gradient = arena.gradient.view();
    custd::span<float const> const device_energy = arena.energy.view();
    bool const copied = sft::download_from(host_gradient, device_gradient, nullptr) &&
                        sft::download_from(custd::span<float> {out_energy, 1}, device_energy, nullptr);
    cudaError_t const drained = cudaStreamSynchronize(nullptr);
    if (!copied) return mmff_energy_probe_t::transfer_failed_k;
    if (drained != cudaSuccess) return mmff_energy_probe_t::launch_failed_k;
    return mmff_energy_probe_t::completed_k;
}

mmff_energy_probe_t gpu_mmff_minimize(custd::span<float> host_positions, std::uint32_t num_atoms,
                                      mmff_energy_terms_t const &terms, std::int32_t max_iterations,
                                      float gradient_threshold, mmff_minimize_result_t &out_result) {

    gradient_arena_t arena;
    if (!arena.resize_for(num_atoms, terms)) return mmff_energy_probe_t::allocation_failed_k;
    if (!arena.upload(custd::span<float const> {host_positions.data(), host_positions.size()}, terms))
        return mmff_energy_probe_t::transfer_failed_k;

    // The three minimizer arrays, then the folded per-atom table behind them, which is the carve
    // `mmff_minimize_kernel` performs.
    std::uint32_t const shared_bytes = minimizer_array_floats_(num_atoms) * static_cast<std::uint32_t>(sizeof(float)) +
                                       num_atoms * static_cast<std::uint32_t>(sizeof(sft::mmff94_atom_vdw_folded_t));
    mmff_minimize_kernel<<<1, sfc::warp_size_k, shared_bytes>>>(
        arena.positions.data(), arena.atom_vdw.data(), arena.atom_charge.data(), arena.bond_distance.data(),
        arena.bonds.data(), arena.angles.data(), arena.torsions.data(), arena.oops.data(), arena.stretchbends.data(),
        arena.interactions.data(), arena.atom_offsets.data(), num_atoms, max_iterations, gradient_threshold,
        arena.minimize_result.data());
    if (cudaGetLastError() != cudaSuccess) return mmff_energy_probe_t::launch_failed_k;

    custd::span<float const> const device_positions = arena.positions.view();
    custd::span<mmff_minimize_result_t const> const device_result = arena.minimize_result.view();
    bool const copied = sft::download_from(host_positions, device_positions, nullptr) &&
                        sft::download_from(custd::span<mmff_minimize_result_t> {&out_result, 1}, device_result,
                                           nullptr);
    cudaError_t const drained = cudaStreamSynchronize(nullptr);
    if (!copied) return mmff_energy_probe_t::transfer_failed_k;
    if (drained != cudaSuccess) return mmff_energy_probe_t::launch_failed_k;
    return mmff_energy_probe_t::completed_k;
}

} // namespace smallfield::tests
