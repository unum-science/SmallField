#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_fp16.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "smallfield/api/smallfield.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_smallfield, m) {
    m.doc() = "SmallField Python bindings";

    m.def("supports_cuda", []() { return true; });

    py::enum_<smallfield::api::conformer_status_t>(m, "ConformerStatus")
        .value("success", smallfield::api::conformer_status_t::success)
        .value("stereo_failure", smallfield::api::conformer_status_t::stereo_failure)
        .value("mmff_unconverged", smallfield::api::conformer_status_t::mmff_unconverged)
        .value("invalid_smiles", smallfield::api::conformer_status_t::invalid_smiles)
        .value("too_many_atoms", smallfield::api::conformer_status_t::too_many_atoms)
        .value("preprocess_timeout", smallfield::api::conformer_status_t::preprocess_timeout);

    py::class_<smallfield::api::context_t>(m, "Context", "GPU context managing CUDA resources.")
        .def(py::init([](std::vector<int> device_ids) {
                 std::span<int const> device_span(device_ids.data(), device_ids.size());
                 return std::make_unique<smallfield::api::context_t>(device_span);
             }),
             py::arg("device_ids") = std::vector<int> {})
        .def(
            "generate_conformers",
            [](smallfield::api::context_t &gpu_context, std::vector<std::string> const &smiles_list,
               std::uint32_t num_conformers, std::uint32_t max_atoms_per_molecule,
               std::uint32_t max_stereo_attempts, std::uint64_t random_seed) {
                smallfield::api::config_t config;
                config.num_conformers = num_conformers;
                config.max_stereo_attempts = max_stereo_attempts;
                config.random_seed = random_seed;

                std::uint32_t const num_molecules = static_cast<std::uint32_t>(smiles_list.size());
                std::uint32_t const total_conformers = num_molecules * num_conformers;
                std::uint32_t const coordinate_stride = max_atoms_per_molecule * 3;
                std::size_t const total_coordinates =
                    static_cast<std::size_t>(total_conformers) * coordinate_stride;

                auto coordinates_array = py::array_t<float>(static_cast<py::ssize_t>(total_coordinates));
                auto energies_array = py::array_t<float>(static_cast<py::ssize_t>(total_conformers));
                auto status_array = py::array_t<std::uint8_t>(static_cast<py::ssize_t>(total_conformers));

                std::vector<std::string_view> smiles_views(smiles_list.begin(), smiles_list.end());
                std::span<std::string_view const> smiles_span(smiles_views);

                std::vector<__half> half_coordinates(total_coordinates);
                std::span<__half> coordinates_span(half_coordinates);
                std::span<float> energies_span(energies_array.mutable_data(), total_conformers);
                std::vector<smallfield::api::conformer_status_t> status_buffer(total_conformers);
                std::span<smallfield::api::conformer_status_t> status_span(status_buffer);

                gpu_context.generate_conformers(smiles_span, config, max_atoms_per_molecule, coordinates_span,
                                                energies_span, status_span);

                float *coordinates_ptr = coordinates_array.mutable_data();
                for (std::size_t index = 0; index < total_coordinates; ++index) {
                    coordinates_ptr[index] = __half2float(half_coordinates[index]);
                }

                std::uint8_t *status_ptr = status_array.mutable_data();
                for (std::uint32_t index = 0; index < total_conformers; ++index) {
                    status_ptr[index] = static_cast<std::uint8_t>(status_buffer[index]);
                }

                return py::make_tuple(coordinates_array, energies_array, status_array);
            },
            py::arg("smiles"),
            py::arg("num_conformers") = 3,
            py::arg("max_atoms_per_molecule") = 128,
            py::arg("max_stereo_attempts") = 10,
            py::arg("random_seed") = 42);
}
