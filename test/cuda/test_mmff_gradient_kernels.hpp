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
 *  @brief Which step of the device round trip refused, so a caller can say why it compared nothing.
 *
 *  A launch the driver declines never writes the gradient or the energy, and neither buffer is zeroed
 *  first, so a caller that reads them anyway differentiates whatever the allocation found.
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
 *  extent, so an empty array stages nothing and needs no guard beside it.
 */
struct mmff_energy_terms_t {
    /** Per-atom Van der Waals parameters. */
    custd::span<smallfield::types::mmff94_atom_vdw_params_t const> atom_vdw;
    /** Per-atom partial charges. */
    custd::span<float const> atom_charge;
    /** Square bond-hop matrix over the same atoms, which is what excludes the bonded pairs. */
    smallfield::types::packed_matrix<std::uint8_t const> bond_distance;
    /** Bond-stretch terms. */
    custd::span<smallfield::types::mmff94_bond_t const> bonds;
    /** Angle-bend terms. */
    custd::span<smallfield::types::mmff94_angle_t const> angles;
    /** Torsion terms. */
    custd::span<smallfield::types::mmff94_torsion_t const> torsions;
    /** Out-of-plane bend terms. */
    custd::span<smallfield::types::mmff94_oop_t const> oops;
    /** Stretch-bend terms. */
    custd::span<smallfield::types::mmff94_stretchbend_t const> stretchbends;
    /** Nonbonded interaction terms. */
    custd::span<smallfield::types::mmff94_atom_interaction_t const> interactions;
    /** Per-atom CSR offsets into the term arrays above. */
    custd::span<smallfield::types::mmff94_atom_offsets_t const> atom_offsets;
};

/**
 *  @brief Evaluate MMFF94 energy and its gradient on the device for one molecule.
 *
 *  A device array it cannot reserve leaves @p host_gradient and @p out_energy untouched and answers with
 *  the refusal, rather than launching the functor on a null pointer.
 *
 *  @param[in] host_positions Coordinates as `(axis, atom)`, `3 * num_atoms` entries, which is the layout
 *      the kernel indexes; an interleaved block passed here scores a different molecule.
 *  @param[in] num_atoms Atom count.
 *  @param[in] terms Every parameter array the energy functor reads.
 *  @param[out] host_gradient Receives the gradient, same extent and layout as @p host_positions.
 *  @param[out] out_energy Receives the total energy.
 *  @return Whether the round trip completed, and which step refused it.
 */
[[nodiscard]] mmff_energy_probe_t gpu_mmff_energy(custd::span<float const> host_positions, std::uint32_t num_atoms,
                                                  mmff_energy_terms_t const &terms, custd::span<float> host_gradient,
                                                  float *out_energy);

/**
 *  @brief What one `gpu_mmff_minimize` call measured, at the geometry it hands back.
 *
 *  `smallfield::cuda::fire_result_t` sits inside that header's `__CUDACC__` guard, so a host driver
 *  reading the outcome needs a spelling of the same five fields it can name from a `.cpp`.
 */
struct mmff_minimize_result_t {
    float energy = 0.0f;
    /** Root mean square of the gradient over all `3 * num_atoms` components, in kcal per mol per angstrom. */
    float rms_gradient = 0.0f;
    /** Length of the last Barzilai-Borwein step, in Å² per kcal per mol. */
    float final_step = 0.0f;
    std::int32_t iterations = 0;
    smallfield::cuda::fire_status_t status = smallfield::cuda::fire_status_t::exhausted_k;
};

/**
 *  @brief Relax one molecule on the device with `barzilai_borwein_minimize_`, the production minimizer.
 *
 *  The budget and the threshold are parameters rather than `mmff_polish_iteration_cap_k` and
 *  `mmff_polish_gradient_threshold_k`, so a caller can sweep either without moving a production constant.
 *  Everything else matches `mmff_polish_warp_kernel`: one warp, the same folded van der Waals staging, the
 *  same three shared arrays, and the same functor.
 *
 *  @param[inout] host_positions Coordinates as `(axis, atom)`, `3 * num_atoms` entries; overwritten with the
 *      relaxed geometry the returned energy and gradient norm were measured at.
 *  @param[in] num_atoms Atom count.
 *  @param[in] terms Every parameter array the energy functor reads.
 *  @param[in] max_iterations Iteration budget; the last evaluation is measured, not stepped from.
 *  @param[in] gradient_threshold Convergence when the RMS gradient falls below it.
 *  @param[out] out_result Receives the outcome; untouched on a refusal.
 *  @return Whether the round trip completed, and which step refused it.
 */
[[nodiscard]] mmff_energy_probe_t gpu_mmff_minimize(custd::span<float> host_positions, std::uint32_t num_atoms,
                                                    mmff_energy_terms_t const &terms, std::int32_t max_iterations,
                                                    float gradient_threshold, mmff_minimize_result_t &out_result);

} // namespace smallfield::tests
