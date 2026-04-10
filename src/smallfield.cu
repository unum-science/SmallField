/**
 * @file smallfield.cu
 * @brief Python bindings for SmallField ETKDG conformer generation pipeline.
 *
 * Provides GPU-accelerated ETKDG + MMFF94 conformer generation via pybind11.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstdlib>
#include <utility>

#if defined(__CUDACC__)
#include "etkdg_pipeline.cuh"
#endif

namespace py = pybind11;

namespace smallfield {

#if defined(__CUDACC__)

/** CUDA unified memory allocator. */
struct cuda_allocator_t {
    static void* allocate(std::size_t bytes, std::size_t /*alignment*/) {
        void* ptr = nullptr;
        cudaMallocManaged(&ptr, bytes);
        return ptr;
    }
    static void deallocate(void* ptr) noexcept { cudaFree(ptr); }
};

/** Aligned buffer with allocator policy. Non-copyable, movable. */
template <typename scalar_type_, typename allocator_type_ = cuda_allocator_t, std::size_t alignment_ = 64>
struct aligned_buffer {
    using scalar_t = scalar_type_;
    using allocator_t = allocator_type_;

    scalar_t* data_ = nullptr;
    std::size_t capacity_ = 0;

    aligned_buffer() = default;
    ~aligned_buffer() {
        if (data_)
            allocator_t::deallocate(data_);
    }

    aligned_buffer(aligned_buffer const&) = delete;
    aligned_buffer& operator=(aligned_buffer const&) = delete;
    aligned_buffer(aligned_buffer&& o) noexcept : data_(o.data_), capacity_(o.capacity_) {
        o.data_ = nullptr;
        o.capacity_ = 0;
    }
    aligned_buffer& operator=(aligned_buffer&& o) noexcept {
        std::swap(data_, o.data_);
        std::swap(capacity_, o.capacity_);
        return *this;
    }

    void allocate(std::size_t count) {
        if (data_)
            allocator_t::deallocate(data_);
        data_ = static_cast<scalar_t*>(allocator_t::allocate(count * sizeof(scalar_t), alignment_));
        capacity_ = count;
    }

    scalar_t* data() noexcept { return data_; }
    scalar_t const* data() const noexcept { return data_; }
};

/** GPU ETKDG conformer generation context with pre-allocated device buffers. */
class etkdg_context_t {
    aligned_buffer<float> bounds_buffer_;
    aligned_buffer<std::uint8_t> chirals_buffer_;
    aligned_buffer<std::uint8_t> torsions_buffer_;
    aligned_buffer<std::uint8_t> constraints_buffer_;
    aligned_buffer<std::uint8_t> metadata_buffer_;
    aligned_buffer<float> coordinates_out_buffer_;
    aligned_buffer<float> energies_out_buffer_;
    aligned_buffer<std::uint8_t> status_out_buffer_;

    std::uint32_t max_atoms_;
    std::uint32_t max_molecules_;
    std::uint32_t max_conformers_;

  public:
    etkdg_context_t(std::uint32_t max_atoms, std::uint32_t max_molecules, std::uint32_t max_conformers)
        : max_atoms_(max_atoms), max_molecules_(max_molecules), max_conformers_(max_conformers) {

        std::uint32_t const max_total_conformers = max_molecules * max_conformers;
        bounds_buffer_.allocate(max_molecules * max_atoms * max_atoms);
        chirals_buffer_.allocate(max_molecules * 16 * sizeof(chiral_center_descriptor_t));
        torsions_buffer_.allocate(max_molecules * 64 * sizeof(etk_torsion_descriptor_t));
        constraints_buffer_.allocate(max_molecules * 256 * sizeof(distance_constraint_t));
        metadata_buffer_.allocate(max_molecules * sizeof(etkdg_molecule_metadata_t));
        coordinates_out_buffer_.allocate(max_total_conformers * max_atoms * 3);
        energies_out_buffer_.allocate(max_total_conformers);
        status_out_buffer_.allocate(max_total_conformers);
    }

    std::uint32_t max_atoms() const { return max_atoms_; }
    std::uint32_t max_molecules() const { return max_molecules_; }
    std::uint32_t max_conformers() const { return max_conformers_; }

    py::tuple generate_conformers(py::array_t<float> bounds_matrices, py::array_t<std::uint8_t> chiral_centers,
                                  py::array_t<std::uint8_t> torsion_terms,
                                  py::array_t<std::uint8_t> distance_constraints,
                                  py::array_t<std::uint8_t> molecule_metadata, std::uint32_t num_conformers,
                                  std::uint32_t max_stereo_attempts, std::uint64_t random_seed) {

        auto bounds_info = bounds_matrices.request();
        auto chirals_info = chiral_centers.request();
        auto torsions_info = torsion_terms.request();
        auto constraints_info = distance_constraints.request();
        auto metadata_info = molecule_metadata.request();

        std::uint32_t const num_molecules =
            static_cast<std::uint32_t>(metadata_info.size / sizeof(etkdg_molecule_metadata_t));

        // copy input to device
        cudaMemcpy(bounds_buffer_.data(), bounds_info.ptr, bounds_info.size * sizeof(float), cudaMemcpyHostToDevice);
        if (chirals_info.size > 0)
            cudaMemcpy(chirals_buffer_.data(), chirals_info.ptr, chirals_info.size, cudaMemcpyHostToDevice);
        if (torsions_info.size > 0)
            cudaMemcpy(torsions_buffer_.data(), torsions_info.ptr, torsions_info.size, cudaMemcpyHostToDevice);
        if (constraints_info.size > 0)
            cudaMemcpy(constraints_buffer_.data(), constraints_info.ptr, constraints_info.size, cudaMemcpyHostToDevice);
        cudaMemcpy(metadata_buffer_.data(), metadata_info.ptr, metadata_info.size, cudaMemcpyHostToDevice);

        // compute grid size from occupancy
        std::uint32_t const shared_bytes = compute_pipeline_shared_memory_bytes(max_atoms_, 0);
        cudaFuncSetAttribute(etkdg_pipeline_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shared_bytes));

        int max_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, etkdg_pipeline_warp_kernel, 32, shared_bytes);
        int num_sms = 0;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        std::uint32_t const grid_size =
            std::min(num_molecules, static_cast<std::uint32_t>(max_blocks_per_sm * num_sms));

        etkdg_pipeline_warp_kernel<<<grid_size, 32, shared_bytes>>>(
            bounds_buffer_.data(), reinterpret_cast<chiral_center_descriptor_t const*>(chirals_buffer_.data()),
            reinterpret_cast<etk_torsion_descriptor_t const*>(torsions_buffer_.data()),
            reinterpret_cast<distance_constraint_t const*>(constraints_buffer_.data()), nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, reinterpret_cast<etkdg_molecule_metadata_t const*>(metadata_buffer_.data()),
            coordinates_out_buffer_.data(), energies_out_buffer_.data(), status_out_buffer_.data(), num_molecules,
            num_conformers, max_stereo_attempts, max_atoms_, max_atoms_, random_seed);
        cudaDeviceSynchronize();

        // copy results back
        std::uint32_t const total_conformers = num_molecules * num_conformers;
        std::uint32_t const total_coords = total_conformers * max_atoms_ * 3;

        auto coords_result = py::array_t<float>(total_coords);
        auto energies_result = py::array_t<float>(total_conformers);
        auto status_result = py::array_t<std::uint8_t>(total_conformers);

        cudaMemcpy(coords_result.mutable_data(), coordinates_out_buffer_.data(), total_coords * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(energies_result.mutable_data(), energies_out_buffer_.data(), total_conformers * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(status_result.mutable_data(), status_out_buffer_.data(), total_conformers * sizeof(std::uint8_t),
                   cudaMemcpyDeviceToHost);

        return py::make_tuple(coords_result, energies_result, status_result);
    }
};

#endif // __CUDACC__

} // namespace smallfield

PYBIND11_MODULE(_smallfield, m) {
    m.doc() = "SmallField: GPU ETKDG + MMFF94 conformer generation";

    m.def("supports_cuda", []() {
#if defined(__CUDACC__)
        return true;
#else
        return false;
#endif
    });

#if defined(__CUDACC__)
    py::class_<smallfield::etkdg_context_t>(m, "EtkdgContext", "GPU ETKDG conformer generation pipeline context.")
        .def(py::init<std::uint32_t, std::uint32_t, std::uint32_t>(), py::arg("max_atoms"), py::arg("max_molecules"),
             py::arg("max_conformers"))
        .def_property_readonly("max_atoms", &smallfield::etkdg_context_t::max_atoms)
        .def_property_readonly("max_molecules", &smallfield::etkdg_context_t::max_molecules)
        .def_property_readonly("max_conformers", &smallfield::etkdg_context_t::max_conformers)
        .def("generate_conformers", &smallfield::etkdg_context_t::generate_conformers,
             "Run ETKDG pipeline on preprocessed molecular data. "
             "Returns (coordinates, energies, status).",
             py::arg("bounds_matrices"), py::arg("chiral_centers"), py::arg("torsion_terms"),
             py::arg("distance_constraints"), py::arg("molecule_metadata"), py::arg("num_conformers") = 10,
             py::arg("max_stereo_attempts") = 20, py::arg("random_seed") = 42);
#endif
}
