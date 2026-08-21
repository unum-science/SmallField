/**
 *  @file test/cuda/test_mmff_gradient_kernels.hpp
 *  @brief Device harness for the single-molecule MMFF94 energy and gradient kernel, for use by its unit tests.
 */

#pragma once

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <string_view> // `std::string_view`

#include "smallfield/cuda/fire_warp.cuh"
#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::tests {

/**
 *  @brief Which step of the device round trip refused, so a caller can say why it compared nothing. A
 *  declined launch never writes the gradient or the energy and neither buffer is zeroed first, so a caller
 *  reading them anyway differentiates whatever the allocation found.
 */
enum class mmff_energy_probe_t {
    completed_k,
    allocation_failed_k,
    transfer_failed_k,
    launch_failed_k,
};

/** The sentence an `mmff_energy_probe_t` reads as, for the `NOTE` line above a skipped comparison. */
std::string_view mmff_energy_probe_reason(mmff_energy_probe_t outcome) noexcept;

/**
 *  @brief One molecule's MMFF94 terms, as the device stages them.
 *
 *  Ten arrays describing the same molecule, so they travel as one named aggregate rather than as ten
 *  positional pointers a call site can transpose between neighbours of the same type. Each carries its own
 *  extent, so an empty array stages nothing and needs no guard beside it. `bond_distance` is the square
 *  bond-hop matrix that excludes the bonded pairs, and `atom_offsets` the per-atom CSR offsets into the
 *  term arrays.
 */
struct mmff_energy_terms_t {
    custd::span<smallfield::types::mmff94_atom_vdw_params_t const> atom_vdw;
    custd::span<float const> atom_charge;
    smallfield::types::packed_matrix<std::uint8_t const> bond_distance;
    custd::span<smallfield::types::mmff94_bond_t const> bonds;
    custd::span<smallfield::types::mmff94_angle_t const> angles;
    custd::span<smallfield::types::mmff94_torsion_t const> torsions;
    custd::span<smallfield::types::mmff94_oop_t const> oops;
    custd::span<smallfield::types::mmff94_stretchbend_t const> stretchbends;
    custd::span<smallfield::types::mmff94_atom_interaction_t const> interactions;
    custd::span<smallfield::types::mmff94_atom_offsets_t const> atom_offsets;
};

/**
 *  @brief Evaluate MMFF94 energy and its gradient on the device for one molecule. A device array it cannot
 *  reserve leaves @p host_gradient and @p out_energy untouched and answers with the refusal, rather than
 *  launching the functor on a null pointer.
 *  @param[in] host_positions Coordinates as `(axis, atom)`, `3 * num_atoms` entries, which is the layout
 *      the kernel indexes; an interleaved block passed here scores a different molecule.
 *  @param[out] host_gradient Receives the gradient, same extent and layout as @p host_positions.
 *  @return Whether the round trip completed, and which step refused it.
 */
[[nodiscard]] mmff_energy_probe_t gpu_mmff_energy(custd::span<float const> host_positions, std::uint32_t num_atoms,
                                                  mmff_energy_terms_t const &terms, custd::span<float> host_gradient,
                                                  float *out_energy);

/**
 *  @brief What one `gpu_mmff_minimize` call measured, at the geometry it hands back.
 *
 *  `smallfield::cuda::fire_result_t` sits inside that header's `__CUDACC__` guard, so a host driver reading
 *  the outcome needs a spelling of the same five fields it can name from a `.cpp`. `rms_gradient` is over
 *  all `3 * num_atoms` components in kcal per mol per angstrom, `final_step` in Å² per kcal per mol.
 */
struct mmff_minimize_result_t {
    float energy = 0.0f;
    float rms_gradient = 0.0f;
    float final_step = 0.0f;
    std::int32_t iterations = 0;
    smallfield::cuda::fire_status_t status = smallfield::cuda::fire_status_t::exhausted_k;
};

/**
 *  @brief Relax one molecule on the device with `barzilai_borwein_minimize_`, the production minimizer.
 *
 *  Budget and threshold are parameters rather than `mmff_polish_iteration_cap_k` and
 *  `mmff_polish_gradient_threshold_k`, so a caller can sweep either without moving a production constant.
 *  Everything else matches `mmff_polish_warp_kernel`: one warp, the same folded van der Waals staging, the
 *  same three shared arrays, the same functor.
 *  @param[inout] host_positions Coordinates as `(axis, atom)`, `3 * num_atoms` entries; overwritten with the
 *      relaxed geometry the returned energy and gradient norm were measured at.
 *  @param[in] max_iterations Iteration budget; the last evaluation is measured, not stepped from.
 *  @param[in] gradient_threshold Convergence when the RMS gradient falls below it.
 *  @param[out] out_result Receives the outcome; untouched on a refusal.
 *  @return Whether the round trip completed, and which step refused it.
 */
[[nodiscard]] mmff_energy_probe_t gpu_mmff_minimize(custd::span<float> host_positions, std::uint32_t num_atoms,
                                                    mmff_energy_terms_t const &terms, std::int32_t max_iterations,
                                                    float gradient_threshold, mmff_minimize_result_t &out_result);

/**
 *  @brief Evaluate the chiral restraint alone on the device, as `mmff94_energy_functor_t` walks it.
 *
 *  One warp, one lane per atom, every declared center walked per atom -- the production sweep with every
 *  published MMFF94 term removed, so a central difference of the returned energy measures the restraint's
 *  own derivative and nothing else. That difference is the restraint's only reference, published MMFF94
 *  carrying no chiral term to compare against.
 *  @param[in] host_positions Coordinates as `(axis, atom)`, `3 * num_atoms` entries.
 *  @param[in] host_centers Declared centers, in the layout the polish kernel hands the functor.
 *  @param[out] host_gradient Receives the gradient, same extent and layout as @p host_positions.
 *  @return Whether the round trip completed, and which step refused it.
 */
[[nodiscard]] mmff_energy_probe_t gpu_chiral_restraint(
    custd::span<float const> host_positions, std::uint32_t num_atoms,
    custd::span<smallfield::types::chiral_center_descriptor_t const> host_centers, custd::span<float> host_gradient,
    float *out_energy);

} // namespace smallfield::tests
