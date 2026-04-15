#pragma once

#include <cstdint>

#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/types/molecule_descriptor.hpp"

void *create_mmff94_device_tables_view(smallfield::parameters::mmff94_tables_host_t const &host,
                                       smallfield::parameters::mmff94_tables_view_t &out_view);

void destroy_mmff94_device_tables_view(void *handle);

void launch_mmff94_gpu_typer_batch(smallfield::parameters::mmff94_tables_view_t tables_view,
                                   smallfield::types::molecule_descriptor_t const *descriptors,
                                   std::uint32_t num_molecules, std::uint32_t const *atom_offsets,
                                   std::uint32_t total_atoms, std::uint8_t *atom_types_out, float *atom_charges_out,
                                   std::uint8_t *typing_status_out);
