/**
 *  @file test/cuda/test_mmff_gradient_kernels.hpp
 *  @brief Device harness for the single-molecule MMFF94 energy and gradient kernel, for use by its unit tests.
 */

#pragma once

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include "smallfield/types/etkdg_types.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::tests {

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
 *  A device array it cannot reserve leaves @p host_gradient and @p out_energy untouched and says so on
 *  `stderr`, rather than launching the functor on a null pointer.
 *
 *  @param[in] host_positions Coordinates as `(axis, atom)`, `3 * num_atoms` entries, which is the layout
 *      the kernel indexes; an interleaved block passed here scores a different molecule.
 *  @param[in] num_atoms Atom count.
 *  @param[in] terms Every parameter array the energy functor reads.
 *  @param[out] host_gradient Receives the gradient, same extent and layout as @p host_positions.
 *  @param[out] out_energy Receives the total energy.
 */
void gpu_mmff_energy(custd::span<float const> host_positions, std::uint32_t num_atoms, mmff_energy_terms_t const &terms,
                     custd::span<float> host_gradient, float *out_energy);

} // namespace smallfield::tests
