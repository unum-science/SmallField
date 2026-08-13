/**
 *  @file test/cuda/test_mmff_gradient_kernels.cu
 *  @brief CUDA kernel wrapper for MMFF94 energy and gradient tests.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::uint32_t`

#include <fmt/format.h> // `fmt::println` -- host-side only; nothing below a `__global__` uses it

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

void gpu_mmff_energy(custd::span<float const> host_positions, std::uint32_t num_atoms, mmff_energy_terms_t const &terms,
                     custd::span<float> host_gradient, float *out_energy) {

    gradient_arena_t arena;
    if (!arena.resize_for(num_atoms, terms)) {
        fmt::println(stderr, "NOTE mmff_gradient_kernels field=device_arena atoms={} succeeded=0", num_atoms);
        return;
    }
    if (!arena.upload(host_positions, terms)) {
        fmt::println(stderr, "NOTE mmff_gradient_kernels field=upload atoms={} succeeded=0", num_atoms);
        return;
    }

    std::uint32_t const shared_bytes = num_atoms * static_cast<std::uint32_t>(sizeof(sft::mmff94_atom_vdw_folded_t));
    mmff_energy_eval_kernel<<<1, sfc::warp_size_k, shared_bytes>>>(
        arena.positions.data(), arena.gradient.data(), arena.atom_vdw.data(), arena.atom_charge.data(),
        arena.bond_distance.data(), arena.bonds.data(), arena.angles.data(), arena.torsions.data(), arena.oops.data(),
        arena.stretchbends.data(), arena.interactions.data(), arena.atom_offsets.data(), num_atoms,
        arena.energy.data());
    cudaDeviceSynchronize();

    custd::span<float const> const device_gradient = arena.gradient.view();
    custd::span<float const> const device_energy = arena.energy.view();
    bool copied = sft::download_from(host_gradient, device_gradient, nullptr);
    copied &= sft::download_from(custd::span<float> {out_energy, 1}, device_energy, nullptr);
    if (!copied) fmt::println(stderr, "NOTE mmff_gradient_kernels field=download atoms={} succeeded=0", num_atoms);
    cudaStreamSynchronize(nullptr);
}

} // namespace smallfield::tests
