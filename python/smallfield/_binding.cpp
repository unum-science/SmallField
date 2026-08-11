/**
 *  @file python/smallfield/_binding.cpp
 *  @brief pybind11 bindings exposing the SmallField C++ API as the `_smallfield` Python module.
 */

#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstring> // `std::memcpy`

#include <algorithm>   // `std::stable_sort`
#include <limits>      // `std::numeric_limits`
#include <memory>      // `std::make_unique`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <cuda_fp16.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "smallfield/api/smallfield.hpp"
#include "smallfield/types/views.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_smallfield, smallfield_module) {
    smallfield_module.doc() = "SmallField Python bindings";

    smallfield_module.def("supports_cuda", []() { return true; });

    py::enum_<smallfield::api::conformer_status_t>(smallfield_module, "ConformerStatus")
        .value("SUCCESS", smallfield::api::conformer_status_t::success_k)
        .value("STEREO_FAILURE", smallfield::api::conformer_status_t::stereo_failure_k)
        .value("MMFF_UNCONVERGED", smallfield::api::conformer_status_t::mmff_unconverged_k)
        .value("INVALID_SMILES", smallfield::api::conformer_status_t::invalid_smiles_k)
        .value("TOO_MANY_ATOMS", smallfield::api::conformer_status_t::too_many_atoms_k)
        .value("PREPROCESS_TIMEOUT", smallfield::api::conformer_status_t::preprocess_timeout_k)
        .value("UNSUPPORTED_ELEMENT", smallfield::api::conformer_status_t::unsupported_element_k);

    py::class_<smallfield::api::context_t>(smallfield_module, "Context", "GPU context managing CUDA resources.")
        .def(py::init([](std::vector<int> device_ids) {
                 custd::span<int const> const device_span(device_ids);
                 return std::make_unique<smallfield::api::context_t>(device_span);
             }),
             py::arg("device_ids") = std::vector<int> {})
        .def(
            "generate_conformers",
            [](smallfield::api::context_t &gpu_context, std::vector<std::string> const &smiles_list,
               std::uint32_t num_conformers, std::uint32_t max_atoms_per_molecule, std::uint32_t max_stereo_attempts,
               std::uint64_t random_seed) {
                smallfield::api::config_t config;
                config.num_conformers = num_conformers;
                config.max_stereo_attempts = max_stereo_attempts;
                config.random_seed = random_seed;

                std::uint32_t const num_molecules = static_cast<std::uint32_t>(smiles_list.size());
                std::uint32_t const total_conformers = num_molecules * num_conformers;
                std::uint32_t const coordinate_stride = max_atoms_per_molecule * 3;
                std::size_t const total_coordinates = static_cast<std::size_t>(total_conformers) * coordinate_stride;

                auto coordinates_array = py::array_t<float>(static_cast<py::ssize_t>(total_coordinates));
                auto energies_array = py::array_t<float>(static_cast<py::ssize_t>(total_conformers));
                auto status_array = py::array_t<std::uint8_t>(static_cast<py::ssize_t>(total_conformers));
                auto num_atoms_array = py::array_t<std::uint8_t>(static_cast<py::ssize_t>(num_molecules));
                auto num_heavy_atoms_array = py::array_t<std::uint16_t>(static_cast<py::ssize_t>(num_molecules));
                auto num_bonds_array = py::array_t<std::uint16_t>(static_cast<py::ssize_t>(num_molecules));
                auto molecular_weight_array = py::array_t<float>(static_cast<py::ssize_t>(num_molecules));

                std::vector<std::string_view> smiles_views(smiles_list.begin(), smiles_list.end());
                std::vector<__half> half_coordinates(total_coordinates);
                std::vector<smallfield::api::conformer_status_t> status_buffer(total_conformers);

                // The NumPy-backed outputs are the only ones needing an explicit extent: `mutable_data`
                // hands back a bare pointer, so the count comes from the shape we allocated above.
                custd::span<float> const energies_span(energies_array.mutable_data(), total_conformers);
                custd::span<std::uint8_t> const num_atoms_span(num_atoms_array.mutable_data(), num_molecules);
                custd::span<std::uint16_t> const num_heavy_atoms_span(num_heavy_atoms_array.mutable_data(),
                                                                      num_molecules);
                custd::span<std::uint16_t> const num_bonds_span(num_bonds_array.mutable_data(), num_molecules);
                custd::span<float> const molecular_weight_span(molecular_weight_array.mutable_data(), num_molecules);

                // The whole wave runs without the GIL, so other interpreter threads keep running for
                // however many minutes it takes. Nothing inside touches a Python object: the NumPy
                // buffers are sized above and reached through raw spans.
                {
                    py::gil_scoped_release const unlocked;
                    [[maybe_unused]] smallfield::api::conformer_status_t const batch_status =
                        gpu_context.generate_conformers(smiles_views, config, max_atoms_per_molecule, half_coordinates,
                                                        energies_span, status_buffer, num_atoms_span,
                                                        num_heavy_atoms_span, num_bonds_span, molecular_weight_span);
                }

                // Per-molecule sort by energy ascending. Failed conformers (status != success) sort last so
                // that successful conformers fill the lowest indices. Tie-break on original index for
                // stability. K is small (3 typical, 16 max) so an N²-style permutation pass is cheap; coords
                // are temp-buffered to avoid in-place swap hazards with the SoA layout.
                float *energies_ptr = energies_array.mutable_data();
                if (num_conformers > 1 && num_molecules > 0) {
                    std::vector<std::uint32_t> permutation(num_conformers);
                    std::vector<__half> coord_temp(static_cast<std::size_t>(num_conformers) * coordinate_stride);
                    std::vector<float> energy_temp(num_conformers);
                    std::vector<smallfield::api::conformer_status_t> status_temp(num_conformers);
                    constexpr float failed_sentinel_k = std::numeric_limits<float>::infinity();

                    for (std::uint32_t molecule_index = 0; molecule_index < num_molecules; ++molecule_index) {
                        std::uint32_t const base_conf = molecule_index * num_conformers;
                        std::size_t const base_coord = static_cast<std::size_t>(base_conf) * coordinate_stride;
                        for (std::uint32_t conformer_rank = 0; conformer_rank < num_conformers; ++conformer_rank)
                            permutation[conformer_rank] = conformer_rank;
                        std::stable_sort(permutation.begin(), permutation.end(),
                                         [&](std::uint32_t left, std::uint32_t right) {
                                             float const left_energy = (status_buffer[base_conf + left] ==
                                                                        smallfield::api::conformer_status_t::success_k)
                                                                           ? energies_ptr[base_conf + left]
                                                                           : failed_sentinel_k;
                                             float const right_energy = (status_buffer[base_conf + right] ==
                                                                         smallfield::api::conformer_status_t::success_k)
                                                                            ? energies_ptr[base_conf + right]
                                                                            : failed_sentinel_k;
                                             return left_energy < right_energy;
                                         });

                        bool identity = true;
                        for (std::uint32_t conformer_rank = 0; conformer_rank < num_conformers; ++conformer_rank) {
                            if (permutation[conformer_rank] != conformer_rank) {
                                identity = false;
                                break;
                            }
                        }
                        if (identity) continue;

                        for (std::uint32_t conformer_rank = 0; conformer_rank < num_conformers; ++conformer_rank) {
                            std::uint32_t const source = permutation[conformer_rank];
                            std::memcpy(
                                &coord_temp[static_cast<std::size_t>(conformer_rank) * coordinate_stride],
                                &half_coordinates[base_coord + static_cast<std::size_t>(source) * coordinate_stride],
                                coordinate_stride * sizeof(__half));
                            energy_temp[conformer_rank] = energies_ptr[base_conf + source];
                            status_temp[conformer_rank] = status_buffer[base_conf + source];
                        }
                        std::memcpy(&half_coordinates[base_coord], coord_temp.data(),
                                    static_cast<std::size_t>(num_conformers) * coordinate_stride * sizeof(__half));
                        for (std::uint32_t conformer_rank = 0; conformer_rank < num_conformers; ++conformer_rank) {
                            energies_ptr[base_conf + conformer_rank] = energy_temp[conformer_rank];
                            status_buffer[base_conf + conformer_rank] = status_temp[conformer_rank];
                        }
                    }
                }

                float *coordinates_ptr = coordinates_array.mutable_data();
                for (std::size_t index = 0; index < total_coordinates; ++index) {
                    coordinates_ptr[index] = __half2float(half_coordinates[index]);
                }

                std::uint8_t *status_ptr = status_array.mutable_data();
                for (std::uint32_t index = 0; index < total_conformers; ++index) {
                    status_ptr[index] = static_cast<std::uint8_t>(status_buffer[index]);
                }

                return py::make_tuple(coordinates_array, energies_array, status_array, num_atoms_array,
                                      num_heavy_atoms_array, num_bonds_array, molecular_weight_array);
            },
            py::arg("smiles"), py::arg("num_conformers") = 3, py::arg("max_atoms_per_molecule") = 128,
            py::arg("max_stereo_attempts") = 10, py::arg("random_seed") = 42);
}
