/**
 *  @file python/smallfield/_binding.cpp
 *  @brief pybind11 bindings exposing the SmallField C++ API as the `_smallfield` Python module.
 *
 *  This file is a boundary between two vocabularies, exactly as `parameters/mmff94_tables_rdkit.hpp` is:
 *  `PascalCase` classes and `UPPER` enumerators are what a Python caller expects, so they keep Python's
 *  spelling here while the C++ they bind to keeps ours.
 *
 *  Coordinates cross as `float16`, which is the storage the pipeline produces, rather than being widened on
 *  the way out. Widening here would cost a second buffer three times the size of the first plus a pass over
 *  every element, for a precision the fp16 slots never held; `smallfield.api` widens for the callers that
 *  want `float32`.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <algorithm>   // `std::copy_n`, `std::is_sorted`, `std::stable_sort`
#include <limits>      // `std::numeric_limits`
#include <memory>      // `std::make_unique`
#include <numeric>     // `std::iota`
#include <stdexcept>   // `std::invalid_argument`, `std::runtime_error`
#include <string>      // `std::string`, `std::to_string`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "smallfield/api/smallfield.hpp"
#include "smallfield/types/views.hpp"

namespace py = pybind11;
namespace api = smallfield::api;

namespace {

#pragma region Outputs

/**
 *  @brief A NumPy output array and the span the pipeline writes through it.
 *
 *  The span takes its extent from the array's own shape, so what C++ writes and what Python reads cannot
 *  disagree. `dtype_name` is NumPy's spelling of `element_` — `float16` for `__half`, `uint8` for
 *  `conformer_status_t`, neither of which pybind11 has a descriptor for — and the item size is checked
 *  against the C++ type, so a mistyped name is a refusal rather than a silent reinterpretation.
 */
template <typename element_>
struct numpy_output {
    py::array array;
    custd::span<element_> view;

    numpy_output(char const *dtype_name, std::size_t count)
        : array(py::dtype(dtype_name), static_cast<py::ssize_t>(count)),
          view(static_cast<element_ *>(array.mutable_data()), static_cast<std::size_t>(array.size())) {
        if (array.itemsize() != static_cast<py::ssize_t>(sizeof(element_)))
            throw std::invalid_argument(std::string("NumPy dtype ") + dtype_name +
                                        " is not the size of the C++ element it stores");
    }
};

/**
 *  @brief One `generate_conformers` call's outputs, named rather than counted.
 *
 *  Seven parallel arrays behind a tuple is seven positions a caller has to get right, and one that unpacks
 *  the first three binds `status` to what is actually `energies`, with nothing to say so until the numbers
 *  come out wrong. A field name cannot be miscounted, and a field added later shifts nothing.
 */
struct conformer_batch_t {
    py::object coordinates;
    py::object energies;
    py::object status;
    py::object num_atoms;
    py::object num_heavy_atoms;
    py::object num_bonds;
    py::object molecular_weight;
};

#pragma endregion Outputs

#pragma region Conformer Ordering

/** Energy the ordering sees: a failed conformer carries infinity, so every success precedes it. */
float ordering_energy_(float energy, api::conformer_status_t status) noexcept {
    return status == api::conformer_status_t::success_k ? energy : std::numeric_limits<float>::infinity();
}

/**
 *  @brief Reorders one molecule's conformer slots into ascending MMFF94 energy, failures last.
 *
 *  Scratch is owned here and sized once for the whole batch, so no molecule allocates. Every span the sort
 *  takes is one molecule's slice of the outputs, so slot zero means the same conformer in all three and no
 *  base offset is recomputed per array.
 */
class conformer_energy_sorter_t {
    std::vector<std::uint32_t> permutation_;
    std::vector<__half> coordinates_scratch_;
    std::vector<float> energies_scratch_;
    std::vector<api::conformer_status_t> status_scratch_;

  public:
    conformer_energy_sorter_t(std::uint32_t num_conformers, std::size_t coordinate_stride)
        : permutation_(num_conformers), coordinates_scratch_(num_conformers * coordinate_stride),
          energies_scratch_(num_conformers), status_scratch_(num_conformers) {}

    /**
     *  @brief Permute one molecule's slots in place.
     *
     *  @param[inout] coordinates This molecule's conformer slots, `num_conformers` strides of halves.
     *  @param[inout] energies This molecule's per-conformer MMFF94 energies, in kcal/mol.
     *  @param[inout] status This molecule's per-conformer status codes.
     */
    void sort_molecule(custd::span<__half> coordinates, custd::span<float> energies,
                       custd::span<api::conformer_status_t> status) {
        std::size_t const num_conformers = energies.size();
        std::size_t const coordinate_stride = coordinates.size() / num_conformers;
        sf_assert_(status.size() == num_conformers);
        sf_assert_(permutation_.size() == num_conformers);

        std::iota(permutation_.begin(), permutation_.end(), std::uint32_t {0});
        std::stable_sort(permutation_.begin(), permutation_.end(),
                         [energies, status](std::uint32_t left, std::uint32_t right) {
                             return ordering_energy_(energies[left], status[left]) <
                                    ordering_energy_(energies[right], status[right]);
                         });

        // A permutation of the conformer indices is the identity exactly when it is ascending, and that is
        // the common case, so the copies below are skipped for most molecules.
        if (std::is_sorted(permutation_.begin(), permutation_.end())) return;

        for (std::size_t rank = 0; rank < num_conformers; ++rank) {
            std::size_t const source = permutation_[rank];
            std::copy_n(coordinates.data() + source * coordinate_stride, coordinate_stride,
                        coordinates_scratch_.data() + rank * coordinate_stride);
            energies_scratch_[rank] = energies[source];
            status_scratch_[rank] = status[source];
        }
        std::copy_n(coordinates_scratch_.data(), coordinates.size(), coordinates.data());
        std::copy_n(energies_scratch_.data(), num_conformers, energies.data());
        std::copy_n(status_scratch_.data(), num_conformers, status.data());
    }
};

#pragma endregion Conformer Ordering

} // namespace

PYBIND11_MODULE(_smallfield, smallfield_module) {
    smallfield_module.doc() = "SmallField Python bindings";

    /*  The atom ceiling is a module constant so `smallfield.api` reads it rather than restating it. The same
     *  number is the stride every conformer slot is padded to, so two layers holding separate copies means
     *  one flat buffer addressed two ways, with nothing comparing them. */
    smallfield_module.attr("MAX_ATOMS_PER_MOLECULE") = api::max_atoms_per_molecule_k;

    smallfield_module.def(
        "supports_cuda",
        []() {
            int device_count = 0;
            return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
        },
        py::call_guard<py::gil_scoped_release>(), "Whether this process can see a CUDA device.");

    py::enum_<api::conformer_status_t>(smallfield_module, "ConformerStatus")
        .value("SUCCESS", api::conformer_status_t::success_k)
        .value("STEREO_FAILURE", api::conformer_status_t::stereo_failure_k)
        .value("MMFF_UNCONVERGED", api::conformer_status_t::mmff_unconverged_k)
        .value("INVALID_SMILES", api::conformer_status_t::invalid_smiles_k)
        .value("TOO_MANY_ATOMS", api::conformer_status_t::too_many_atoms_k)
        .value("PREPROCESS_TIMEOUT", api::conformer_status_t::preprocess_timeout_k)
        .value("UNSUPPORTED_ELEMENT", api::conformer_status_t::unsupported_element_k);

    py::class_<conformer_batch_t>(smallfield_module, "ConformerBatch",
                                  "Named outputs of one `Context.generate_conformers` call.")
        .def_readonly("coordinates", &conformer_batch_t::coordinates,
                      "float16, flat, `molecules * conformers * max_atoms_per_molecule * 3`, " // pinned split
                      "structure-of-arrays within each conformer slot.")
        .def_readonly("energies", &conformer_batch_t::energies, "float32, `molecules * conformers`, kcal/mol.")
        .def_readonly("status", &conformer_batch_t::status, "uint8 `ConformerStatus` codes, `molecules * conformers`.")
        .def_readonly("num_atoms", &conformer_batch_t::num_atoms,
                      "uint8 per molecule, hydrogens included, 0 where preprocess rejected it.")
        .def_readonly("num_heavy_atoms", &conformer_batch_t::num_heavy_atoms, "uint16 per molecule.")
        .def_readonly("num_bonds", &conformer_batch_t::num_bonds, "uint16 per molecule, hydrogen bonds included.")
        .def_readonly("molecular_weight", &conformer_batch_t::molecular_weight, "float32 per molecule, in daltons.");

    py::class_<api::context_t>(smallfield_module, "Context",
                               "GPU context owning CUDA streams and per-device scratch. Construction " // split
                               "initializes every device, so reuse one across calls.")
        .def(py::init([](std::vector<int> const &device_ids) {
                 return std::make_unique<api::context_t>(custd::span<int const>(device_ids));
             }),
             py::arg("device_ids") = std::vector<int> {}, py::call_guard<py::gil_scoped_release>())
        .def(
            "generate_conformers",
            [](api::context_t &gpu_context, std::vector<std::string> const &smiles_list, std::uint32_t num_conformers,
               std::uint32_t max_atoms_per_molecule, std::uint32_t max_stereo_attempts, std::uint64_t random_seed) {
                api::config_t config;
                config.num_conformers = num_conformers;
                config.max_stereo_attempts = max_stereo_attempts;
                config.random_seed = random_seed;

                std::size_t const num_molecules = smiles_list.size();
                std::size_t const num_slots = num_molecules * num_conformers;
                std::size_t const coordinate_stride = static_cast<std::size_t>(max_atoms_per_molecule) * 3;

                numpy_output<__half> coordinates {"float16", num_slots * coordinate_stride};
                numpy_output<float> energies {"float32", num_slots};
                numpy_output<api::conformer_status_t> status {"uint8", num_slots};
                numpy_output<std::uint8_t> num_atoms {"uint8", num_molecules};
                numpy_output<std::uint16_t> num_heavy_atoms {"uint16", num_molecules};
                numpy_output<std::uint16_t> num_bonds {"uint16", num_molecules};
                numpy_output<float> molecular_weight {"float32", num_molecules};

                std::vector<std::string_view> const smiles_views(smiles_list.begin(), smiles_list.end());
                std::uint64_t const errors_before = gpu_context.cuda_error_count.load();
                {
                    // The wave and the reorder both run without the GIL, so other interpreter threads keep
                    // running for however many minutes they take. Neither touches a Python object: the
                    // arrays are allocated above and reached only through their spans.
                    py::gil_scoped_release const unlocked;

                    // The batch status is `success_k` exactly when some slot of `status` is, so returning it
                    // as well would be one fact in two encodings. A caller reads the array.
                    [[maybe_unused]] api::conformer_status_t const batch_status = gpu_context.generate_conformers(
                        smiles_views, config, max_atoms_per_molecule, coordinates.view, energies.view, status.view,
                        num_atoms.view, num_heavy_atoms.view, num_bonds.view, molecular_weight.view);

                    if (num_conformers > 1) {
                        conformer_energy_sorter_t sorter(num_conformers, coordinate_stride);
                        for (std::size_t molecule = 0; molecule < num_molecules; ++molecule) {
                            std::size_t const first_slot = molecule * num_conformers;
                            sorter.sort_molecule(coordinates.view.subspan(first_slot * coordinate_stride,
                                                                          num_conformers * coordinate_stride),
                                                 energies.view.subspan(first_slot, num_conformers),
                                                 status.view.subspan(first_slot, num_conformers));
                        }
                    }
                }

                // A CUDA failure leaves the outputs at the zero and `invalid_smiles_k` fill that
                // `generate_conformers` starts from, which reads exactly like a batch of unparseable input.
                // This counter is the only thing separating the two, and `smallfield.hpp` says production
                // callers exit non-zero on it — so it raises here rather than becoming an eighth array
                // nobody reads.
                std::uint64_t const cuda_errors = gpu_context.cuda_error_count.load() - errors_before;
                if (cuda_errors != 0)
                    throw std::runtime_error(std::to_string(cuda_errors) +
                                             " CUDA errors during generate_conformers; " // pinned split
                                             "the context logger names each one");

                return conformer_batch_t {
                    .coordinates = coordinates.array,
                    .energies = energies.array,
                    .status = status.array,
                    .num_atoms = num_atoms.array,
                    .num_heavy_atoms = num_heavy_atoms.array,
                    .num_bonds = num_bonds.array,
                    .molecular_weight = molecular_weight.array,
                };
            },
            py::arg("smiles"), py::arg("num_conformers") = api::config_t {}.num_conformers,
            py::arg("max_atoms_per_molecule") = api::max_atoms_per_molecule_k,
            py::arg("max_stereo_attempts") = api::config_t {}.max_stereo_attempts,
            py::arg("random_seed") = api::config_t {}.random_seed,
            "Embed and minimize a batch, returning a `ConformerBatch`. Conformers within each " // pinned split
            "molecule come back sorted by ascending energy with failures last. Raises "         // pinned split
            "`RuntimeError` if the run hit a CUDA error, because the outputs are then "         // pinned split
            "indistinguishable from a batch of unparseable input.");
}
