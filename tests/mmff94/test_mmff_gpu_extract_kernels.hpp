#pragma once

#include <cstdint>

#include "smallfield/mmff94/mmff94_extract_device.cuh"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

void launch_mmff94_gpu_extract_pipeline(smallfield::parameters::mmff94_tables_host_t const &tables_host,
                                        smallfield::types::molecule_descriptor_t const *descriptors,
                                        std::uint32_t num_mols, std::uint32_t const *atom_offsets,
                                        std::uint32_t total_atoms,
                                        smallfield::mmff94::mmff94_per_mol_counts_t *out_counts,
                                        std::uint8_t *out_status, smallfield::types::mmff94_bond_t *out_bonds,
                                        std::uint32_t bond_stride, smallfield::types::mmff94_angle_t *out_angles,
                                        std::uint32_t angle_stride, smallfield::types::mmff94_torsion_t *out_torsions,
                                        std::uint32_t torsion_stride, smallfield::types::mmff94_oop_t *out_oops,
                                        std::uint32_t oop_stride, smallfield::types::mmff94_stretchbend_t *out_stbns,
                                        std::uint32_t stbn_stride,
                                        smallfield::types::mmff94_atom_interaction_t *out_interactions,
                                        smallfield::types::mmff94_atom_offsets_t *out_atom_offsets);
