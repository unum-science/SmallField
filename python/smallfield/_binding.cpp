#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "smallfield/api/smallfield.hpp"
#include "smallfield/api/smallfield_mace.hpp"
#include "smallfield/types/mace_types.hpp"

namespace py = pybind11;

namespace {

/// Copy a host numpy contiguous array to a freshly cudaMalloc'd device buffer.
/// Caller takes ownership; free with `cudaFree`.
template <typename T>
T *device_alloc_copy_from_host_(py::array_t<T, py::array::c_style | py::array::forcecast> const &host_array) {
    std::size_t const n = static_cast<std::size_t>(host_array.size());
    T *device_ptr = nullptr;
    if (n > 0) {
        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&device_ptr), n * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
        err = cudaMemcpy(device_ptr, host_array.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(device_ptr);
            throw std::runtime_error(std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(err));
        }
    }
    return device_ptr;
}

template <typename T>
T *device_alloc_zeroed_(std::size_t n) {
    if (n == 0) return nullptr;
    T *device_ptr = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&device_ptr), n * sizeof(T));
    if (err != cudaSuccess) { throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err)); }
    cudaMemset(device_ptr, 0, n * sizeof(T));
    return device_ptr;
}

template <typename T>
void device_to_host_copy_(py::array_t<T> &host_dst, T const *device_src) {
    std::size_t const n = static_cast<std::size_t>(host_dst.size());
    if (n == 0) return;
    cudaError_t const err = cudaMemcpy(host_dst.mutable_data(), device_src, n * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(err));
    }
}

/// Build a `batch_view_t` from numpy arrays staged to device, used by both
/// `infer_*` and `refine` bindings.  The struct returned holds raw cudaMalloc'd
/// pointers; the caller (a small RAII guard below) frees them on scope exit.
struct mace_device_batch_t {
    float *positions = nullptr;
    std::int32_t *atom_element = nullptr;
    std::int32_t *edge_source = nullptr;
    std::int32_t *edge_target = nullptr;
    std::int32_t *graph_of_atom = nullptr;
    float *energies = nullptr;
    float *forces = nullptr;
    std::uint32_t n_atoms = 0;
    std::uint32_t n_edges = 0;
    std::uint32_t n_graphs = 0;

    ~mace_device_batch_t() {
        if (positions) cudaFree(positions);
        if (atom_element) cudaFree(atom_element);
        if (edge_source) cudaFree(edge_source);
        if (edge_target) cudaFree(edge_target);
        if (graph_of_atom) cudaFree(graph_of_atom);
        if (energies) cudaFree(energies);
        if (forces) cudaFree(forces);
    }
};

} // namespace

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
            py::arg("smiles"), py::arg("num_conformers") = 3, py::arg("max_atoms_per_molecule") = 128,
            py::arg("max_stereo_attempts") = 10, py::arg("random_seed") = 42);

    // ===================================================================
    // MACE-OFF inference + refinement bindings.
    //
    // Mirrors the C++ surface in `smallfield::mace::*` (smallfield_mace.hpp).
    // Three classes:
    //   MaceArch      — runtime metadata (n_channels, r_max, ...) read from the weights.
    //   MaceWeights   — RAII handle around device-resident MACE weights blob.
    //   MaceContext   — pre-allocated scratch + stream + capture cache.
    //
    // Numpy I/O contract:
    //   positions      : float32, shape (n_atoms, 3)
    //   atom_element   : int32,   shape (n_atoms,)        — index into arch.n_elements
    //   edge_source    : int32,   shape (n_edges,)
    //   edge_target    : int32,   shape (n_edges,)
    //   graph_of_atom  : int32,   shape (n_atoms,)        — graph membership
    //
    // Edge construction (atomic-number → element-index mapping, r_max neighbour
    // search) is the caller's responsibility.  The high-level Python wrapper
    // in smallfield/api.py handles this from RDKit.
    // ===================================================================

    py::enum_<smallfield::mace::status_t>(m, "MaceStatus")
        .value("success", smallfield::mace::status_t::success)
        .value("invalid_input", smallfield::mace::status_t::invalid_input)
        .value("weights_load_failed", smallfield::mace::status_t::weights_load_failed)
        .value("cuda_error", smallfield::mace::status_t::cuda_error)
        .value("not_supported", smallfield::mace::status_t::not_supported);

    py::class_<smallfield::types::mace_arch_t>(m, "MaceArch",
                                               "Architecture metadata derived from the weights blob (read-only).")
        .def_readonly("n_elements", &smallfield::types::mace_arch_t::n_elements)
        .def_readonly("n_radial", &smallfield::types::mace_arch_t::n_radial)
        .def_readonly("n_channels", &smallfield::types::mace_arch_t::n_channels)
        .def_readonly("l_max", &smallfield::types::mace_arch_t::l_max)
        .def_readonly("num_interactions", &smallfield::types::mace_arch_t::num_interactions)
        .def_readonly("r_max", &smallfield::types::mace_arch_t::r_max)
        .def_readonly("avg_num_neighbors", &smallfield::types::mace_arch_t::avg_num_neighbors)
        .def_readonly("scale", &smallfield::types::mace_arch_t::scale)
        .def_readonly("shift", &smallfield::types::mace_arch_t::shift)
        .def("__repr__", [](smallfield::types::mace_arch_t const &a) {
            return "<MaceArch n_elements=" + std::to_string(a.n_elements) +
                   " n_channels=" + std::to_string(a.n_channels) + " r_max=" + std::to_string(a.r_max) + ">";
        });

    py::class_<smallfield::mace::weights_t>(m, "MaceWeights",
                                            "RAII handle around a device-resident MACE-OFF weights blob.")
        .def_static(
            "load", &smallfield::mace::weights_t::load, py::arg("path"),
            "Load MACE-OFF weights from a SmallField .bin file produced by " "`python/smallfield/mace_export.py`.")
        .def("arch", &smallfield::mace::weights_t::arch, py::return_value_policy::reference_internal,
             "Architecture metadata read from the weights blob.")
        .def("is_loaded", &smallfield::mace::weights_t::is_loaded);

    py::
        class_<smallfield::mace::context_t>(m, "MaceContext", "Pre-allocated MACE-OFF inference + refinement context.")
            .def(
                py::init([](int device_id, smallfield::mace::weights_t const &weights, std::uint32_t max_atoms,
                            std::uint32_t max_edges, std::uint32_t max_graphs) {
                    return std::make_unique<smallfield::mace::context_t>(device_id, weights.arch(), max_atoms,
                                                                         max_edges, max_graphs);
                }),
                py::arg("device_id"), py::arg("weights"), py::arg("max_atoms"), py::arg("max_edges"),
                py::arg("max_graphs"),
                "Construct a context dimensioned for `(max_atoms, max_edges, max_graphs)`.  " "Pass `weights` for the "
                                                                                              "architecture metadata "
                                                                                              "only — the context does "
                                                                                              "NOT " "retain a "
                                                                                                     "reference; you "
                                                                                                     "still pass "
                                                                                                     "`weights` per "
                                                                                                     "call.")
            .def(
                "infer_energy",
                [](smallfield::mace::context_t &self, smallfield::mace::weights_t const &weights,
                   py::array_t<float, py::array::c_style | py::array::forcecast> positions,
                   py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> atom_element,
                   py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> edge_source,
                   py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> edge_target,
                   py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> graph_of_atom,
                   std::uint32_t n_graphs) {
                    mace_device_batch_t batch;
                    batch.n_atoms = static_cast<std::uint32_t>(atom_element.size());
                    batch.n_edges = static_cast<std::uint32_t>(edge_source.size());
                    batch.n_graphs = n_graphs;
                    if (positions.shape(0) != static_cast<py::ssize_t>(batch.n_atoms) || positions.shape(1) != 3) {
                        throw std::invalid_argument("positions must be shape (n_atoms, 3)");
                    }
                    batch.positions = device_alloc_copy_from_host_<float>(positions);
                    batch.atom_element = device_alloc_copy_from_host_<std::int32_t>(atom_element);
                    batch.edge_source = device_alloc_copy_from_host_<std::int32_t>(edge_source);
                    batch.edge_target = device_alloc_copy_from_host_<std::int32_t>(edge_target);
                    batch.graph_of_atom = device_alloc_copy_from_host_<std::int32_t>(graph_of_atom);
                    batch.energies = device_alloc_zeroed_<float>(n_graphs);

                    smallfield::mace::batch_view_t view {};
                    view.n_atoms = batch.n_atoms;
                    view.n_edges = batch.n_edges;
                    view.n_graphs = batch.n_graphs;
                    view.positions_xyz = batch.positions;
                    view.atom_element = batch.atom_element;
                    view.edge_source_atom = batch.edge_source;
                    view.edge_target_atom = batch.edge_target;
                    view.graph_of_atom = batch.graph_of_atom;

                    [[maybe_unused]] auto const status = self.infer_energy(weights, view, batch.energies);
                    if (status != smallfield::mace::status_t::success) {
                        throw std::runtime_error("MaceContext.infer_energy failed");
                    }
                    auto energies_array = py::array_t<float>(static_cast<py::ssize_t>(n_graphs));
                    device_to_host_copy_<float>(energies_array, batch.energies);
                    return energies_array;
                },
                py::arg("weights"), py::arg("positions"), py::arg("atom_element"), py::arg("edge_source"),
                py::arg("edge_target"), py::arg("graph_of_atom"), py::arg("n_graphs"),
                "Forward pass — returns per-graph energies in eV (numpy float32, shape (n_graphs,)).")
            .def(
                "infer_energy_and_forces",
                [](smallfield::mace::context_t &self, smallfield::mace::weights_t const &weights,
                   py::array_t<float, py::array::c_style | py::array::forcecast> positions,
                   py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> atom_element,
                   py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> edge_source,
                   py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> edge_target,
                   py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> graph_of_atom,
                   std::uint32_t n_graphs) {
                    mace_device_batch_t batch;
                    batch.n_atoms = static_cast<std::uint32_t>(atom_element.size());
                    batch.n_edges = static_cast<std::uint32_t>(edge_source.size());
                    batch.n_graphs = n_graphs;
                    if (positions.shape(0) != static_cast<py::ssize_t>(batch.n_atoms) || positions.shape(1) != 3) {
                        throw std::invalid_argument("positions must be shape (n_atoms, 3)");
                    }
                    batch.positions = device_alloc_copy_from_host_<float>(positions);
                    batch.atom_element = device_alloc_copy_from_host_<std::int32_t>(atom_element);
                    batch.edge_source = device_alloc_copy_from_host_<std::int32_t>(edge_source);
                    batch.edge_target = device_alloc_copy_from_host_<std::int32_t>(edge_target);
                    batch.graph_of_atom = device_alloc_copy_from_host_<std::int32_t>(graph_of_atom);
                    batch.energies = device_alloc_zeroed_<float>(n_graphs);
                    batch.forces = device_alloc_zeroed_<float>(static_cast<std::size_t>(batch.n_atoms) * 3u);

                    smallfield::mace::batch_view_t view {};
                    view.n_atoms = batch.n_atoms;
                    view.n_edges = batch.n_edges;
                    view.n_graphs = batch.n_graphs;
                    view.positions_xyz = batch.positions;
                    view.atom_element = batch.atom_element;
                    view.edge_source_atom = batch.edge_source;
                    view.edge_target_atom = batch.edge_target;
                    view.graph_of_atom = batch.graph_of_atom;

                    [[maybe_unused]] auto const status = self.infer_energy_and_forces(weights, view, batch.energies,
                                                                                      batch.forces);
                    if (status != smallfield::mace::status_t::success) {
                        throw std::runtime_error("MaceContext.infer_energy_and_forces failed");
                    }

                    auto energies_array = py::array_t<float>(static_cast<py::ssize_t>(n_graphs));
                    auto forces_array = py::array_t<float>(
                        {static_cast<py::ssize_t>(batch.n_atoms), static_cast<py::ssize_t>(3)});
                    device_to_host_copy_<float>(energies_array, batch.energies);
                    device_to_host_copy_<float>(forces_array, batch.forces);
                    return py::make_tuple(energies_array, forces_array);
                },
                py::arg("weights"), py::arg("positions"), py::arg("atom_element"), py::arg("edge_source"),
                py::arg("edge_target"), py::arg("graph_of_atom"), py::arg("n_graphs"),
                "Forward + adjoint backward — returns `(energies, forces)` numpy arrays.  " "`energies` is shape "
                                                                                            "`(n_graphs,)` in eV; "
                                                                                            "`forces` is shape "
                                                                                            "`(n_atoms, 3)` in eV/Å.")
            .def(
                "refine",
                [](smallfield::mace::context_t &self, smallfield::mace::weights_t const &weights,
                   py::array_t<float, py::array::c_style | py::array::forcecast> positions,
                   py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> atom_element,
                   py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> edge_source,
                   py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> edge_target,
                   py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> graph_of_atom,
                   std::uint32_t n_graphs, std::uint32_t num_steps, float fmax, float step_size) {
                    mace_device_batch_t batch;
                    batch.n_atoms = static_cast<std::uint32_t>(atom_element.size());
                    batch.n_edges = static_cast<std::uint32_t>(edge_source.size());
                    batch.n_graphs = n_graphs;
                    if (positions.shape(0) != static_cast<py::ssize_t>(batch.n_atoms) || positions.shape(1) != 3) {
                        throw std::invalid_argument("positions must be shape (n_atoms, 3)");
                    }
                    batch.positions = device_alloc_copy_from_host_<float>(positions);
                    batch.atom_element = device_alloc_copy_from_host_<std::int32_t>(atom_element);
                    batch.edge_source = device_alloc_copy_from_host_<std::int32_t>(edge_source);
                    batch.edge_target = device_alloc_copy_from_host_<std::int32_t>(edge_target);
                    batch.graph_of_atom = device_alloc_copy_from_host_<std::int32_t>(graph_of_atom);
                    batch.energies = device_alloc_zeroed_<float>(n_graphs);

                    smallfield::mace::batch_view_t view {};
                    view.n_atoms = batch.n_atoms;
                    view.n_edges = batch.n_edges;
                    view.n_graphs = batch.n_graphs;
                    view.positions_xyz = batch.positions;
                    view.atom_element = batch.atom_element;
                    view.edge_source_atom = batch.edge_source;
                    view.edge_target_atom = batch.edge_target;
                    view.graph_of_atom = batch.graph_of_atom;

                    smallfield::mace::relax_config_t config {};
                    config.num_steps = num_steps;
                    config.fmax_eV_per_A = fmax;
                    config.step_size = step_size;

                    [[maybe_unused]] auto const status = self.refine_batch(weights, view, config, batch.positions,
                                                                           batch.energies);
                    if (status != smallfield::mace::status_t::success) {
                        throw std::runtime_error("MaceContext.refine failed");
                    }

                    auto refined_positions = py::array_t<float>(
                        {static_cast<py::ssize_t>(batch.n_atoms), static_cast<py::ssize_t>(3)});
                    auto energies_array = py::array_t<float>(static_cast<py::ssize_t>(n_graphs));
                    device_to_host_copy_<float>(refined_positions, batch.positions);
                    device_to_host_copy_<float>(energies_array, batch.energies);
                    return py::make_tuple(refined_positions, energies_array);
                },
                py::arg("weights"), py::arg("positions"), py::arg("atom_element"), py::arg("edge_source"),
                py::arg("edge_target"), py::arg("graph_of_atom"), py::arg("n_graphs"), py::arg("num_steps") = 20,
                py::arg("fmax") = 0.05f, py::arg("step_size") = 0.01f,
                "Batched LBFGS-style relaxation.  Returns `(refined_positions, energies)` " "numpy arrays.  "
                                                                                            "`num_steps` caps the "
                                                                                            "iteration count, `fmax` "
                                                                                            "is the " "per-atom "
                                                                                                      "force-magnitude "
                                                                                                      "convergence "
                                                                                                      "threshold "
                                                                                                      "(eV/Å).");
}
