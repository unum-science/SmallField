/**
 *  @file test/cuda/test_mmff_gradient_kernels.hpp
 *  @brief Device harness for the single-molecule MMFF94 energy and gradient kernel, for use by its unit tests.
 */

#pragma once

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include "smallfield/types/etkdg_types.hpp"

namespace smallfield::tests {

/**
 *  @brief Evaluate MMFF94 energy and its gradient on the device for one molecule.
 *
 *  @param[in] host_positions Flat interleaved xyz coordinates, `3 * num_atoms` entries.
 *  @param[in] num_atoms Atom count.
 *  @param[in] host_atom_vdw Per-atom Van der Waals parameters.
 *  @param[in] host_atom_charge Per-atom partial charges.
 *  @param[in] host_bond_distance Flat `num_atoms` squared bond-hop matrix.
 *  @param[in] host_bond_params Bond-stretch terms.
 *  @param[in] num_bond_params Number of entries in @p host_bond_params.
 *  @param[in] host_angle_params Angle-bend terms.
 *  @param[in] num_angle_params Number of entries in @p host_angle_params.
 *  @param[in] host_torsion_params Torsion terms.
 *  @param[in] num_torsion_params Number of entries in @p host_torsion_params.
 *  @param[in] host_oop_params Out-of-plane bend terms.
 *  @param[in] num_oop_params Number of entries in @p host_oop_params.
 *  @param[in] host_stretchbend_params Stretch-bend terms.
 *  @param[in] num_stretchbend_params Number of entries in @p host_stretchbend_params.
 *  @param[in] host_interactions Nonbonded interaction terms.
 *  @param[in] num_interactions Number of entries in @p host_interactions.
 *  @param[in] host_atom_offsets Per-atom term offsets.
 *  @param[in] num_atom_offsets Number of entries in @p host_atom_offsets.
 *  @param[out] host_gradient Receives the gradient, same layout as @p host_positions.
 *  @param[out] out_energy Receives the total energy.
 */
void gpu_mmff_energy(float const *host_positions, std::uint32_t num_atoms,
                     smallfield::types::mmff94_atom_vdw_params_t const *host_atom_vdw, float const *host_atom_charge,
                     std::uint8_t const *host_bond_distance, smallfield::types::mmff94_bond_t const *host_bond_params,
                     std::uint32_t num_bond_params, smallfield::types::mmff94_angle_t const *host_angle_params,
                     std::uint32_t num_angle_params, smallfield::types::mmff94_torsion_t const *host_torsion_params,
                     std::uint32_t num_torsion_params, smallfield::types::mmff94_oop_t const *host_oop_params,
                     std::uint32_t num_oop_params,
                     smallfield::types::mmff94_stretchbend_t const *host_stretchbend_params,
                     std::uint32_t num_stretchbend_params,
                     smallfield::types::mmff94_atom_interaction_t const *host_interactions,
                     std::uint32_t num_interactions, smallfield::types::mmff94_atom_offsets_t const *host_atom_offsets,
                     std::uint32_t num_atom_offsets, float *host_gradient, float *out_energy);

} // namespace smallfield::tests
