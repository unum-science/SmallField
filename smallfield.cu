/**
 * @file smallfield.cu
 * @brief Python bindings for SmallField MMFF94 force field calculations
 * @author Ash Vardanian
 * @date 2024
 *
 * Provides CPU (serial, AVX-512) and GPU (CUDA) implementations for:
 * - Van der Waals energy (buffered 14-7 potential)
 * - Electrostatic energy (Coulomb with buffer)
 *
 * Bindings use pybind11 for clean Python interface.
 *
 * Module name: _smallfield
 * Namespace: smallfield
 * Internal functions end with underscore (_)
 * Uses std::size_t for positive indexes
 * East Const style
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "fork_union.hpp"
#include "mmff94.hpp"

#if defined(__CUDACC__)
#include "mmff94.cuh"
#endif

namespace py = pybind11;
namespace fu = ashvardanian::fork_union;

namespace smallfield {

#pragma region Allocators and Aligned Buffers

/**
 * @brief CPU memory allocator using std::aligned_alloc.
 *
 * Allocates 64-byte aligned memory for AVX-512 SIMD operations.
 */
struct cpu_allocator_t {
    static void* allocate(std::size_t bytes, std::size_t alignment) {
        // Round up to alignment boundary (required by std::aligned_alloc)
        std::size_t const aligned_bytes = (bytes + alignment - 1) & ~(alignment - 1);
        return std::aligned_alloc(alignment, aligned_bytes);
    }
    static void deallocate(void* ptr) noexcept { std::free(ptr); }
};

#if defined(__CUDACC__)
/**
 * @brief CUDA unified memory allocator using cudaMallocManaged.
 *
 * Memory is accessible from both CPU and GPU with automatic migration.
 */
struct cuda_allocator_t {
    static void* allocate(std::size_t bytes, std::size_t /*alignment*/) {
        void* ptr = nullptr;
        cudaMallocManaged(&ptr, bytes);
        return ptr;
    }
    static void deallocate(void* ptr) noexcept { cudaFree(ptr); }
};

// Default allocator is CUDA when compiling with nvcc
using default_allocator_t = cuda_allocator_t;
#else
// Default allocator is CPU when compiling without nvcc
using default_allocator_t = cpu_allocator_t;
#endif

/**
 * @brief Aligned buffer with allocator policy.
 *
 * @tparam scalar_type_ Element type
 * @tparam allocator_type_ Allocator type (cpu_allocator_t or cuda_allocator_t)
 * @tparam alignment_ Memory alignment in bytes (default 64 for AVX-512)
 *
 * Properly aligns heap data (not just the object) for SIMD operations.
 * Non-copyable, movable.
 */
template <typename scalar_type_, typename allocator_type_ = default_allocator_t, std::size_t alignment_ = 64>
struct aligned_buffer {
    using scalar_t = scalar_type_;
    using allocator_t = allocator_type_;
    static constexpr std::size_t alignment = alignment_;

    scalar_t* data_ = nullptr;
    std::size_t capacity_ = 0;

    aligned_buffer() = default;

    explicit aligned_buffer(std::size_t count) { allocate(count); }

    ~aligned_buffer() {
        if (data_)
            allocator_t::deallocate(data_);
    }

    // Non-copyable
    aligned_buffer(aligned_buffer const&) = delete;
    aligned_buffer& operator=(aligned_buffer const&) = delete;

    // Movable
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
        std::size_t const bytes = count * sizeof(scalar_t);
        data_ = static_cast<scalar_t*>(allocator_t::allocate(bytes, alignment));
        capacity_ = count;
    }

    scalar_t* data() noexcept { return data_; }
    scalar_t const* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return capacity_; }

    scalar_t& operator[](std::size_t i) noexcept { return data_[i]; }
    scalar_t const& operator[](std::size_t i) const noexcept { return data_[i]; }
};

#pragma endregion

#pragma region Structured Buffer Types

/**
 * @brief Coordinate buffers for a batch of atoms (x, y, z).
 *
 * Uses byte storage that can be cast to any scalar type.
 */
struct coord_buffers_t {
    aligned_buffer<std::uint8_t> x, y, z;
    std::size_t n_atoms_ = 0;
    std::size_t scalar_size_ = 4; // default float

    void allocate(std::size_t n_atoms, std::size_t scalar_size = 4) {
        n_atoms_ = n_atoms;
        scalar_size_ = scalar_size;
        std::size_t bytes = n_atoms * scalar_size;
        x.allocate(bytes);
        y.allocate(bytes);
        z.allocate(bytes);
    }

    template <typename T> T* x_data() noexcept { return reinterpret_cast<T*>(x.data()); }
    template <typename T> T* y_data() noexcept { return reinterpret_cast<T*>(y.data()); }
    template <typename T> T* z_data() noexcept { return reinterpret_cast<T*>(z.data()); }
};

/**
 * @brief Non-bonded pair buffers for VdW/electrostatic calculations.
 *
 * Uses byte storage for params that can be cast to any scalar type.
 */
struct pair_buffers_t {
    aligned_buffer<std::int32_t> atom_i, atom_j;
    aligned_buffer<std::uint8_t> param1, param2; // epsilon/r_star or qi/qj
    aligned_buffer<std::int32_t> is_1_4;
    std::size_t n_pairs_ = 0;
    std::size_t scalar_size_ = 4; // default float

    void allocate(std::size_t n_pairs, std::size_t scalar_size = 4) {
        n_pairs_ = n_pairs;
        scalar_size_ = scalar_size;
        atom_i.allocate(n_pairs);
        atom_j.allocate(n_pairs);
        std::size_t bytes = n_pairs * scalar_size;
        param1.allocate(bytes);
        param2.allocate(bytes);
        is_1_4.allocate(n_pairs);
    }

    std::int32_t* i_data() noexcept { return atom_i.data(); }
    std::int32_t* j_data() noexcept { return atom_j.data(); }
    template <typename T> T* param1_data() noexcept { return reinterpret_cast<T*>(param1.data()); }
    template <typename T> T* param2_data() noexcept { return reinterpret_cast<T*>(param2.data()); }
    std::int32_t* is_1_4_data() noexcept { return is_1_4.data(); }
};

/**
 * @brief Batch indexing buffers for per-molecule processing.
 */
struct batch_buffers_t {
    aligned_buffer<std::int32_t> mol_start, mol_end;
    aligned_buffer<float> out_energy;

    void allocate(std::size_t n_molecules) {
        mol_start.allocate(n_molecules);
        mol_end.allocate(n_molecules);
        out_energy.allocate(n_molecules);
    }

    std::int32_t* start_data() noexcept { return mol_start.data(); }
    std::int32_t* end_data() noexcept { return mol_end.data(); }
    float* energy_data() noexcept { return out_energy.data(); }
};

#pragma endregion

#pragma region Atom-Centric Buffers

#if defined(__CUDACC__)
/**
 * @brief Buffers for atom-centric kernel data layout.
 *
 * Converts pair-based layout to symmetric adjacency list for the
 * atom-centric kernel that eliminates atomics and gathers.
 */
struct atom_centric_buffers_t {
    aligned_buffer<std::uint8_t> neighbors;        // neighbor_t array
    aligned_buffer<std::uint32_t> neighbor_offsets; // Per-atom neighbor ranges
    aligned_buffer<std::uint8_t> metas;            // mol_meta_t array
    aligned_buffer<float> coords_aos;              // AoS coordinates: x0,y0,z0,x1,y1,z1,...

    std::size_t capacity_neighbors_ = 0;
    std::size_t capacity_atoms_ = 0;
    std::size_t capacity_molecules_ = 0;

    void allocate(std::size_t max_neighbors, std::size_t max_atoms, std::size_t max_molecules) {
        capacity_neighbors_ = max_neighbors;
        capacity_atoms_ = max_atoms;
        capacity_molecules_ = max_molecules;

        neighbors.allocate(max_neighbors * sizeof(neighbor_t));
        neighbor_offsets.allocate(max_atoms + max_molecules); // +1 per molecule
        metas.allocate(max_molecules * sizeof(mol_meta_t));
        coords_aos.allocate(max_atoms * 3);
    }

    neighbor_t* neighbors_data() noexcept { return reinterpret_cast<neighbor_t*>(neighbors.data()); }
    std::uint32_t* offsets_data() noexcept { return neighbor_offsets.data(); }
    mol_meta_t* metas_data() noexcept { return reinterpret_cast<mol_meta_t*>(metas.data()); }
    float* coords_data() noexcept { return coords_aos.data(); }
};

/**
 * @brief Build symmetric adjacency list from pair arrays.
 *
 * Converts (pair_i, pair_j, epsilon, r_star, qi, qj, is_1_4) pairs
 * into a symmetric adjacency list where each pair is stored twice:
 * - Once in atom i's neighbor list (with sign = -1)
 * - Once in atom j's neighbor list (with sign = +1)
 *
 * This enables the atom-centric kernel to process each atom independently
 * without atomic operations or cross-lane communication for gradients.
 *
 * @param pair_i Atom i indices for each pair
 * @param pair_j Atom j indices for each pair
 * @param epsilon VdW epsilon for each pair
 * @param r_star VdW equilibrium distance for each pair
 * @param qi Charge of atom i for each pair
 * @param qj Charge of atom j for each pair
 * @param is_1_4 1-4 interaction flag for each pair
 * @param mol_pair_start Start index in pairs for each molecule
 * @param mol_pair_end End index in pairs for each molecule
 * @param mol_atom_start Start index in atoms for each molecule
 * @param mol_atom_end End index in atoms for each molecule
 * @param n_molecules Number of molecules
 * @param coords_x X coordinates (SoA)
 * @param coords_y Y coordinates (SoA)
 * @param coords_z Z coordinates (SoA)
 * @param bufs Output buffers
 */
inline void build_symmetric_adjacency_list_(
    std::int32_t const* pair_i, std::int32_t const* pair_j,
    float const* epsilon, float const* r_star,
    float const* qi, float const* qj,
    std::int32_t const* is_1_4,
    std::int32_t const* mol_pair_start, std::int32_t const* mol_pair_end,
    std::int32_t const* mol_atom_start, std::int32_t const* mol_atom_end,
    std::uint32_t n_molecules,
    float const* coords_x, float const* coords_y, float const* coords_z,
    atom_centric_buffers_t& bufs) {

    neighbor_t* neighbors = bufs.neighbors_data();
    std::uint32_t* offsets = bufs.offsets_data();
    mol_meta_t* metas = bufs.metas_data();
    float* coords_aos = bufs.coords_data();

    std::uint32_t neighbor_idx = 0;
    std::uint32_t offset_idx = 0;

    for (std::uint32_t mol = 0; mol < n_molecules; ++mol) {
        std::uint32_t const atom_start = static_cast<std::uint32_t>(mol_atom_start[mol]);
        std::uint32_t const atom_end = static_cast<std::uint32_t>(mol_atom_end[mol]);
        std::uint32_t const n_atoms = atom_end - atom_start;
        std::uint32_t const pair_start = static_cast<std::uint32_t>(mol_pair_start[mol]);
        std::uint32_t const pair_end = static_cast<std::uint32_t>(mol_pair_end[mol]);

        // Convert SoA coordinates to AoS for this molecule
        for (std::uint32_t a = atom_start; a < atom_end; ++a) {
            coords_aos[a * 3 + 0] = coords_x[a];
            coords_aos[a * 3 + 1] = coords_y[a];
            coords_aos[a * 3 + 2] = coords_z[a];
        }

        // First pass: count neighbors per atom
        std::vector<std::uint32_t> counts(n_atoms, 0);
        for (std::uint32_t p = pair_start; p < pair_end; ++p) {
            std::uint32_t const i = static_cast<std::uint32_t>(pair_i[p]) - atom_start;
            std::uint32_t const j = static_cast<std::uint32_t>(pair_j[p]) - atom_start;
            counts[i]++;
            counts[j]++;
        }

        // Compute offsets for this molecule
        std::uint32_t const mol_neighbor_start = neighbor_idx;
        std::vector<std::uint32_t> atom_offsets(n_atoms + 1);
        atom_offsets[0] = 0;
        for (std::uint32_t a = 0; a < n_atoms; ++a) {
            atom_offsets[a + 1] = atom_offsets[a] + counts[a];
        }

        // Write offsets to global array
        for (std::uint32_t a = 0; a <= n_atoms; ++a) {
            offsets[offset_idx++] = atom_offsets[a];
        }

        // Second pass: fill neighbors
        std::vector<std::uint32_t> current(n_atoms, 0);
        for (std::uint32_t p = pair_start; p < pair_end; ++p) {
            std::uint32_t const i = static_cast<std::uint32_t>(pair_i[p]) - atom_start;
            std::uint32_t const j = static_cast<std::uint32_t>(pair_j[p]) - atom_start;
            float const eps = epsilon[p];
            float const rs = r_star[p];
            float const q_prod = qi[p] * qj[p];
            std::uint16_t const flags = static_cast<std::uint16_t>(is_1_4[p]);

            // Entry for atom i: neighbor is j, sign = -1
            std::uint32_t const idx_i = mol_neighbor_start + atom_offsets[i] + current[i]++;
            neighbors[idx_i].j = static_cast<std::uint8_t>(j);
            neighbors[idx_i].sign = -1;
            neighbors[idx_i].flags = flags;
            neighbors[idx_i].epsilon = eps;
            neighbors[idx_i].r_star = rs;
            neighbors[idx_i].q_product = q_prod;

            // Entry for atom j: neighbor is i, sign = +1
            std::uint32_t const idx_j = mol_neighbor_start + atom_offsets[j] + current[j]++;
            neighbors[idx_j].j = static_cast<std::uint8_t>(i);
            neighbors[idx_j].sign = +1;
            neighbors[idx_j].flags = flags;
            neighbors[idx_j].epsilon = eps;
            neighbors[idx_j].r_star = rs;
            neighbors[idx_j].q_product = q_prod;
        }

        // Fill molecule metadata
        metas[mol].atom_start = atom_start;
        metas[mol].n_atoms = n_atoms;
        metas[mol].neighbor_start = mol_neighbor_start;
        metas[mol].n_neighbors = atom_offsets[n_atoms];

        neighbor_idx += atom_offsets[n_atoms];
    }
}
#endif // __CUDACC__

#pragma endregion

#pragma region context_t Class

/**
 * @brief Context for SmallField MMFF94 force field calculations.
 *
 * Pre-allocates buffers with total capacity limits.
 * Uses default_allocator_t which is CUDA unified memory when compiling with nvcc,
 * or CPU aligned memory otherwise.
 *
 * Python API receives and returns f64 (double) arrays.
 * All internal conversion to f32/f16/bf16 happens inside the native library.
 */
class context_t {
  private:
    std::uint32_t capacity_atoms_;
    std::uint32_t capacity_pairs_;
    std::uint32_t capacity_molecules_;

    // ForkUnion thread pool for CPU parallelism
    fu::basic_pool_t pool_;

    // Single set of buffers - use max scalar size (4 bytes for float)
    coord_buffers_t coords_;
    pair_buffers_t pairs_;
    batch_buffers_t batch_;

    // L-BFGS workspace for optimization
    // Workspace per molecule: m*(6*n_atoms + 2) floats where m=10 (memory parameter)
    aligned_buffer<std::uint8_t> lbfgs_workspace_;

#if defined(__CUDACC__)
    // Atom-centric kernel buffers (symmetric adjacency list)
    atom_centric_buffers_t atom_centric_bufs_;
#endif

    /**
     * @brief Copy f64 array to buffer as f32.
     */
    void copy_f64_to_f32_(py::array_t<double> const& src, float* dst, std::uint32_t count) {
        auto const buf = src.request();
        double const* src_ptr = static_cast<double const*>(buf.ptr);
        for (std::uint32_t i = 0; i < count; ++i) {
            dst[i] = static_cast<float>(src_ptr[i]);
        }
    }

    /**
     * @brief Copy f64 array to buffer as bf16.
     */
    void copy_f64_to_bf16_(py::array_t<double> const& src, std::uint16_t* dst, std::uint32_t count) {
        auto const buf = src.request();
        double const* src_ptr = static_cast<double const*>(buf.ptr);
        for (std::uint32_t i = 0; i < count; ++i) {
            float const f32_val = static_cast<float>(src_ptr[i]);
            std::uint32_t f32_bits;
            std::memcpy(&f32_bits, &f32_val, sizeof(float));
            dst[i] = static_cast<std::uint16_t>(f32_bits >> 16);
        }
    }

    /**
     * @brief Copy f64 array to buffer as fp16.
     */
    void copy_f64_to_fp16_(py::array_t<double> const& src, std::uint16_t* dst, std::uint32_t count) {
        auto const buf = src.request();
        double const* src_ptr = static_cast<double const*>(buf.ptr);
        for (std::uint32_t i = 0; i < count; ++i) {
            float const f32_val = static_cast<float>(src_ptr[i]);
#if defined(__AVX512FP16__)
            _Float16 const fp16_val = static_cast<_Float16>(f32_val);
            std::memcpy(&dst[i], &fp16_val, sizeof(std::uint16_t));
#else
            // Fallback software conversion
            std::uint32_t f32_bits;
            std::memcpy(&f32_bits, &f32_val, sizeof(float));
            std::uint32_t sign = (f32_bits >> 16) & 0x8000;
            std::int32_t exp = ((f32_bits >> 23) & 0xFF) - 127 + 15;
            std::uint32_t mant = (f32_bits >> 13) & 0x3FF;
            if (exp <= 0) {
                dst[i] = static_cast<std::uint16_t>(sign);
            } else if (exp >= 31) {
                dst[i] = static_cast<std::uint16_t>(sign | 0x7C00);
            } else {
                dst[i] = static_cast<std::uint16_t>(sign | (exp << 10) | mant);
            }
#endif
        }
    }

    // Round up to SIMD register size (32 for AVX-512 with half precision)
    static constexpr std::uint32_t simd_align(std::uint32_t n) { return (n + 31) & ~31u; }

  public:
    context_t(std::uint32_t total_atoms, std::uint32_t total_pairs, std::uint32_t max_molecules)
        : capacity_atoms_(simd_align(total_atoms)), capacity_pairs_(simd_align(total_pairs)),
          capacity_molecules_(max_molecules) {

        pool_.try_spawn(std::thread::hardware_concurrency());

        // Allocate with max scalar size (4 bytes for float), rounded up for SIMD
        coords_.allocate(capacity_atoms_, 4);
        pairs_.allocate(capacity_pairs_, 4);
        batch_.allocate(max_molecules);

        // Allocate L-BFGS workspace
        // m=10 (L-BFGS memory), n_atoms per molecule = capacity_atoms_/max_molecules
        // Workspace per molecule (floats):
        //   s_history:    m * 3 * n_atoms
        //   y_history:    m * 3 * n_atoms
        //   alpha:        m
        //   rho:          m
        //   q:            3 * n_atoms
        //   grad:         3 * n_atoms
        //   grad_prev:    3 * n_atoms
        //   direction:    3 * n_atoms
        //   coord_backup: 3 * n_atoms
        // Total: 6*m*n_atoms + 2*m + 15*n_atoms
        std::uint32_t const atoms_per_mol = capacity_atoms_ / max_molecules;
        std::uint32_t const m = 10; // L-BFGS memory parameter
        std::size_t const floats_per_mol = 6 * m * atoms_per_mol + 2 * m + 15 * atoms_per_mol;
        std::size_t const workspace_bytes = max_molecules * floats_per_mol * sizeof(float);
        lbfgs_workspace_.allocate(workspace_bytes);

#if defined(__CUDACC__)
        // Allocate atom-centric buffers (2x pairs for symmetric adjacency list)
        atom_centric_bufs_.allocate(capacity_pairs_ * 2, capacity_atoms_, max_molecules);
#endif
    }

    ~context_t() = default;

    context_t(context_t const&) = delete;
    context_t& operator=(context_t const&) = delete;
    context_t(context_t&&) = delete;
    context_t& operator=(context_t&&) = delete;

    std::uint32_t capacity_atoms() const { return capacity_atoms_; }
    std::uint32_t capacity_pairs() const { return capacity_pairs_; }
    std::uint32_t capacity_molecules() const { return capacity_molecules_; }

    /**
     * @brief VdW batch using Skylake AVX-512 with ForkUnion parallelization.
     */
    double vdw_skylake(py::array_t<double> coords_x, py::array_t<double> coords_y, py::array_t<double> coords_z,
                       py::array_t<std::int32_t> pair_atom_i, py::array_t<std::int32_t> pair_atom_j,
                       py::array_t<double> pair_epsilon, py::array_t<double> pair_r_star,
                       py::array_t<std::int32_t> mol_pair_start, py::array_t<std::int32_t> mol_pair_end,
                       std::uint32_t n_molecules) {
#if defined(__AVX512F__)
        std::uint32_t const num_atoms = static_cast<std::uint32_t>(coords_x.size());
        std::uint32_t const num_pairs = static_cast<std::uint32_t>(pair_atom_i.size());

        if (num_pairs > capacity_pairs_ || num_atoms > capacity_atoms_ || n_molecules > capacity_molecules_) {
            throw std::runtime_error("Input exceeds context capacity");
        }

        copy_f64_to_f32_(coords_x, coords_.x_data<float>(), num_atoms);
        copy_f64_to_f32_(coords_y, coords_.y_data<float>(), num_atoms);
        copy_f64_to_f32_(coords_z, coords_.z_data<float>(), num_atoms);
        copy_f64_to_f32_(pair_epsilon, pairs_.param1_data<float>(), num_pairs);
        copy_f64_to_f32_(pair_r_star, pairs_.param2_data<float>(), num_pairs);

        auto const buf_i = pair_atom_i.request();
        auto const buf_j = pair_atom_j.request();
        auto const buf_start = mol_pair_start.request();
        auto const buf_end = mol_pair_end.request();

        std::int32_t const* pair_i = static_cast<std::int32_t const*>(buf_i.ptr);
        std::int32_t const* pair_j = static_cast<std::int32_t const*>(buf_j.ptr);
        std::int32_t const* mol_start = static_cast<std::int32_t const*>(buf_start.ptr);
        std::int32_t const* mol_end = static_cast<std::int32_t const*>(buf_end.ptr);

        float const* cx = coords_.x_data<float>();
        float const* cy = coords_.y_data<float>();
        float const* cz = coords_.z_data<float>();
        float const* eps = pairs_.param1_data<float>();
        float const* r_star = pairs_.param2_data<float>();
        float* out_energy = batch_.energy_data();

        pool_.for_n(n_molecules, [=](auto prong) noexcept {
            std::uint32_t const mol = static_cast<std::uint32_t>(prong.task);
            std::int32_t const start = mol_start[mol];
            std::int32_t const end = mol_end[mol];
            std::int32_t const count = end - start;
            out_energy[mol] = compute_vdw_energy_skylake(cx, cy, cz, pair_i, pair_j, eps, r_star, start, count);
        });

        double total_energy = 0.0;
        for (std::uint32_t i = 0; i < n_molecules; ++i) {
            total_energy += static_cast<double>(out_energy[i]);
        }
        return total_energy;
#else
        throw std::runtime_error("AVX-512 F32 support not available");
#endif
    }

    /**
     * @brief VdW energy using CUDA F32 kernel.
     *
     * Uses CUDA unified memory - data written to buffers is automatically
     * migrated to GPU when kernel runs, results readable after sync.
     */
    double vdw_cuda_f32(py::array_t<double> coords_x, py::array_t<double> coords_y, py::array_t<double> coords_z,
                        py::array_t<std::int32_t> nb_i, py::array_t<std::int32_t> nb_j, py::array_t<double> nb_epsilon,
                        py::array_t<double> nb_r_star, py::array_t<std::int32_t> mol_nb_start,
                        py::array_t<std::int32_t> mol_nb_end, std::uint32_t n_molecules) {
#if defined(__CUDACC__)
        std::uint32_t const num_atoms = static_cast<std::uint32_t>(coords_x.size());
        std::uint32_t const num_pairs = static_cast<std::uint32_t>(nb_i.size());

        if (num_pairs > capacity_pairs_ || num_atoms > capacity_atoms_ || n_molecules > capacity_molecules_) {
            throw std::runtime_error("Input exceeds context capacity");
        }

        // Convert f64 to f32 directly into unified memory buffers
        copy_f64_to_f32_(coords_x, coords_.x_data<float>(), num_atoms);
        copy_f64_to_f32_(coords_y, coords_.y_data<float>(), num_atoms);
        copy_f64_to_f32_(coords_z, coords_.z_data<float>(), num_atoms);
        copy_f64_to_f32_(nb_epsilon, pairs_.param1_data<float>(), num_pairs);
        copy_f64_to_f32_(nb_r_star, pairs_.param2_data<float>(), num_pairs);

        // Copy pair indices directly to unified memory
        auto const buf_i = nb_i.request();
        auto const buf_j = nb_j.request();
        auto const buf_start = mol_nb_start.request();
        auto const buf_end = mol_nb_end.request();

        std::memcpy(pairs_.i_data(), buf_i.ptr, num_pairs * sizeof(std::int32_t));
        std::memcpy(pairs_.j_data(), buf_j.ptr, num_pairs * sizeof(std::int32_t));
        std::memcpy(batch_.start_data(), buf_start.ptr, n_molecules * sizeof(std::int32_t));
        std::memcpy(batch_.end_data(), buf_end.ptr, n_molecules * sizeof(std::int32_t));

        // Launch kernel (unified memory migrates automatically)
        mmff94_vdw_energy_kernel_f32<<<n_molecules, warp_size>>>(
            coords_.x_data<float>(), coords_.y_data<float>(), coords_.z_data<float>(), pairs_.i_data(), pairs_.j_data(),
            pairs_.param1_data<float>(), pairs_.param2_data<float>(), batch_.start_data(), batch_.end_data(),
            batch_.energy_data(), n_molecules);

        // Sync and sum results (unified memory accessible after sync)
        cudaDeviceSynchronize();

        double total_energy = 0.0;
        float const* out_energy = batch_.energy_data();
        for (std::uint32_t i = 0; i < n_molecules; ++i) {
            total_energy += static_cast<double>(out_energy[i]);
        }

        return total_energy;
#else
        throw std::runtime_error("CUDA support not available");
#endif
    }

    /**
     * @brief VdW energy using CUDA F16 kernel.
     *
     * Uses CUDA unified memory with F16 buffers for reduced bandwidth.
     */
    double vdw_cuda_f16(py::array_t<double> coords_x, py::array_t<double> coords_y, py::array_t<double> coords_z,
                        py::array_t<std::int32_t> nb_i, py::array_t<std::int32_t> nb_j, py::array_t<double> nb_epsilon,
                        py::array_t<double> nb_r_star, py::array_t<std::int32_t> mol_nb_start,
                        py::array_t<std::int32_t> mol_nb_end, std::uint32_t n_molecules) {
#if defined(__CUDACC__)
        std::uint32_t const num_atoms = static_cast<std::uint32_t>(coords_x.size());
        std::uint32_t const num_pairs = static_cast<std::uint32_t>(nb_i.size());

        if (num_pairs > capacity_pairs_ || num_atoms > capacity_atoms_ || n_molecules > capacity_molecules_) {
            throw std::runtime_error("Input exceeds context capacity");
        }

        // Convert f64 directly to fp16 using copy_f64_to_fp16_
        copy_f64_to_fp16_(coords_x, reinterpret_cast<std::uint16_t*>(coords_.x_data<float>()), num_atoms);
        copy_f64_to_fp16_(coords_y, reinterpret_cast<std::uint16_t*>(coords_.y_data<float>()), num_atoms);
        copy_f64_to_fp16_(coords_z, reinterpret_cast<std::uint16_t*>(coords_.z_data<float>()), num_atoms);
        copy_f64_to_fp16_(nb_epsilon, reinterpret_cast<std::uint16_t*>(pairs_.param1_data<float>()), num_pairs);
        copy_f64_to_fp16_(nb_r_star, reinterpret_cast<std::uint16_t*>(pairs_.param2_data<float>()), num_pairs);

        half* cx_f16 = reinterpret_cast<half*>(coords_.x_data<float>());
        half* cy_f16 = reinterpret_cast<half*>(coords_.y_data<float>());
        half* cz_f16 = reinterpret_cast<half*>(coords_.z_data<float>());
        half* eps_f16 = reinterpret_cast<half*>(pairs_.param1_data<float>());
        half* r_f16 = reinterpret_cast<half*>(pairs_.param2_data<float>());

        // Copy pair indices directly to unified memory
        auto const buf_i = nb_i.request();
        auto const buf_j = nb_j.request();
        auto const buf_start = mol_nb_start.request();
        auto const buf_end = mol_nb_end.request();

        std::memcpy(pairs_.i_data(), buf_i.ptr, num_pairs * sizeof(std::int32_t));
        std::memcpy(pairs_.j_data(), buf_j.ptr, num_pairs * sizeof(std::int32_t));
        std::memcpy(batch_.start_data(), buf_start.ptr, n_molecules * sizeof(std::int32_t));
        std::memcpy(batch_.end_data(), buf_end.ptr, n_molecules * sizeof(std::int32_t));

        // Launch kernel (unified memory migrates automatically)
        mmff94_vdw_energy_kernel_f16<<<n_molecules, warp_size>>>(cx_f16, cy_f16, cz_f16, pairs_.i_data(),
                                                                 pairs_.j_data(), eps_f16, r_f16, batch_.start_data(),
                                                                 batch_.end_data(), batch_.energy_data(), n_molecules);

        // Sync and sum results
        cudaDeviceSynchronize();

        double total_energy = 0.0;
        float const* out_energy = batch_.energy_data();
        for (std::uint32_t i = 0; i < n_molecules; ++i) {
            total_energy += static_cast<double>(out_energy[i]);
        }

        return total_energy;
#else
        throw std::runtime_error("CUDA support not available");
#endif
    }

    /**
     * @brief Optimize geometry using serial F64 L-BFGS.
     * @return tuple: (final_energy, iterations, converged, rms_gradient)
     */
    py::tuple optimize_serial(py::array_t<double> coords_x, py::array_t<double> coords_y, py::array_t<double> coords_z,
                               py::array_t<std::int32_t> pair_atom_i, py::array_t<std::int32_t> pair_atom_j,
                               py::array_t<double> pair_epsilon, py::array_t<double> pair_r_star, int max_iterations = 200,
                               double gradient_threshold = 1e-4) {
        // Placeholder implementation - will be completed with actual L-BFGS integration
        throw std::runtime_error("optimize_serial not yet implemented - requires L-BFGS integration");
    }

    /**
     * @brief Optimize geometry using Skylake AVX-512 F32 L-BFGS.
     * @return tuple: (final_energy, iterations, converged, rms_gradient)
     */
    py::tuple optimize_skylake(py::array_t<double> coords_x, py::array_t<double> coords_y, py::array_t<double> coords_z,
                                py::array_t<std::int32_t> pair_atom_i, py::array_t<std::int32_t> pair_atom_j,
                                py::array_t<double> pair_epsilon, py::array_t<double> pair_r_star, int max_iterations = 200,
                                double gradient_threshold = 1e-4) {
#if defined(__AVX512F__)
        // Placeholder implementation - will be completed with actual L-BFGS integration
        throw std::runtime_error("optimize_skylake not yet implemented - requires L-BFGS integration");
#else
        throw std::runtime_error("AVX-512 F32 support not available");
#endif
    }

    /**
     * @brief Optimize geometry using CUDA F32 batch L-BFGS.
     * @return tuple: (final_energy, iterations, converged, rms_gradient)
     */
    py::tuple optimize_cuda_f32(py::array_t<double> coords_x, py::array_t<double> coords_y, py::array_t<double> coords_z,
                                 py::array_t<std::int32_t> pair_atom_i, py::array_t<std::int32_t> pair_atom_j,
                                 py::array_t<double> pair_epsilon, py::array_t<double> pair_r_star,
                                 py::array_t<std::int32_t> mol_pair_start, py::array_t<std::int32_t> mol_pair_end,
                                 std::uint32_t n_molecules, int max_iterations = 200, double gradient_threshold = 1e-4) {
#if defined(__CUDACC__)
        // Placeholder implementation - will be completed with actual L-BFGS integration
        throw std::runtime_error("optimize_cuda_f32 not yet implemented - requires L-BFGS integration");
#else
        throw std::runtime_error("CUDA support not available");
#endif
    }

    /**
     * @brief Optimize geometry using CUDA F16 batch L-BFGS.
     * @return tuple: (final_energy, iterations, converged, rms_gradient)
     */
    py::tuple optimize_cuda_f16(py::array_t<double> coords_x, py::array_t<double> coords_y, py::array_t<double> coords_z,
                                 py::array_t<std::int32_t> pair_atom_i, py::array_t<std::int32_t> pair_atom_j,
                                 py::array_t<double> pair_epsilon, py::array_t<double> pair_r_star,
                                 py::array_t<std::int32_t> mol_pair_start, py::array_t<std::int32_t> mol_pair_end,
                                 std::uint32_t n_molecules, int max_iterations = 200, double gradient_threshold = 1e-4) {
#if defined(__CUDACC__)
        // Placeholder implementation - will be completed with actual L-BFGS integration
        throw std::runtime_error("optimize_cuda_f16 not yet implemented - requires L-BFGS integration");
#else
        throw std::runtime_error("CUDA support not available");
#endif
    }

    /**
     * @brief Compute energy and gradient using atom-centric CUDA F32 kernel.
     *
     * This kernel eliminates atomics and gathers by using:
     * - Register-based coordinate storage
     * - Shuffle-based coordinate access
     * - Symmetric adjacency list (pairs stored twice)
     * - Local gradient accumulation
     *
     * Requires additional charge arrays for electrostatics.
     *
     * @return tuple: (total_energy, gradient_x, gradient_y, gradient_z)
     */
    py::tuple energy_gradient_atom_centric_cuda_f32(
        py::array_t<double> coords_x, py::array_t<double> coords_y, py::array_t<double> coords_z,
        py::array_t<std::int32_t> pair_atom_i, py::array_t<std::int32_t> pair_atom_j,
        py::array_t<double> pair_epsilon, py::array_t<double> pair_r_star,
        py::array_t<double> pair_qi, py::array_t<double> pair_qj,
        py::array_t<std::int32_t> pair_is_1_4,
        py::array_t<std::int32_t> mol_pair_start, py::array_t<std::int32_t> mol_pair_end,
        py::array_t<std::int32_t> mol_atom_start, py::array_t<std::int32_t> mol_atom_end,
        std::uint32_t n_molecules) {
#if defined(__CUDACC__)
        std::uint32_t const num_atoms = static_cast<std::uint32_t>(coords_x.size());
        std::uint32_t const num_pairs = static_cast<std::uint32_t>(pair_atom_i.size());

        if (num_pairs > capacity_pairs_ || num_atoms > capacity_atoms_ || n_molecules > capacity_molecules_) {
            throw std::runtime_error("Input exceeds context capacity");
        }

        // Convert f64 to f32
        copy_f64_to_f32_(coords_x, coords_.x_data<float>(), num_atoms);
        copy_f64_to_f32_(coords_y, coords_.y_data<float>(), num_atoms);
        copy_f64_to_f32_(coords_z, coords_.z_data<float>(), num_atoms);
        copy_f64_to_f32_(pair_epsilon, pairs_.param1_data<float>(), num_pairs);
        copy_f64_to_f32_(pair_r_star, pairs_.param2_data<float>(), num_pairs);

        // Allocate temporary buffers for charges
        aligned_buffer<float> qi_buf(num_pairs);
        aligned_buffer<float> qj_buf(num_pairs);
        {
            auto const buf_qi = pair_qi.request();
            auto const buf_qj = pair_qj.request();
            double const* qi_ptr = static_cast<double const*>(buf_qi.ptr);
            double const* qj_ptr = static_cast<double const*>(buf_qj.ptr);
            for (std::uint32_t i = 0; i < num_pairs; ++i) {
                qi_buf[i] = static_cast<float>(qi_ptr[i]);
                qj_buf[i] = static_cast<float>(qj_ptr[i]);
            }
        }

        // Copy indices
        auto const buf_i = pair_atom_i.request();
        auto const buf_j = pair_atom_j.request();
        auto const buf_is_1_4 = pair_is_1_4.request();
        auto const buf_pair_start = mol_pair_start.request();
        auto const buf_pair_end = mol_pair_end.request();
        auto const buf_atom_start = mol_atom_start.request();
        auto const buf_atom_end = mol_atom_end.request();

        std::memcpy(pairs_.i_data(), buf_i.ptr, num_pairs * sizeof(std::int32_t));
        std::memcpy(pairs_.j_data(), buf_j.ptr, num_pairs * sizeof(std::int32_t));
        std::memcpy(pairs_.is_1_4_data(), buf_is_1_4.ptr, num_pairs * sizeof(std::int32_t));
        std::memcpy(batch_.start_data(), buf_pair_start.ptr, n_molecules * sizeof(std::int32_t));
        std::memcpy(batch_.end_data(), buf_pair_end.ptr, n_molecules * sizeof(std::int32_t));

        // Build symmetric adjacency list
        build_symmetric_adjacency_list_(
            pairs_.i_data(), pairs_.j_data(),
            pairs_.param1_data<float>(), pairs_.param2_data<float>(),
            qi_buf.data(), qj_buf.data(),
            pairs_.is_1_4_data(),
            batch_.start_data(), batch_.end_data(),
            static_cast<std::int32_t const*>(buf_atom_start.ptr),
            static_cast<std::int32_t const*>(buf_atom_end.ptr),
            n_molecules,
            coords_.x_data<float>(), coords_.y_data<float>(), coords_.z_data<float>(),
            atom_centric_bufs_);

        // Allocate output gradient buffer
        aligned_buffer<float> grad_out(num_atoms * 3);

        // Launch atom-centric kernel
        mmff94_energy_gradient_atom_centric_f32<<<n_molecules, warp_size>>>(
            atom_centric_bufs_.coords_data(),
            atom_centric_bufs_.neighbors_data(),
            atom_centric_bufs_.offsets_data(),
            atom_centric_bufs_.metas_data(),
            grad_out.data(),
            batch_.energy_data(),
            n_molecules);

        cudaDeviceSynchronize();

        // Sum energies
        double total_energy = 0.0;
        for (std::uint32_t i = 0; i < n_molecules; ++i) {
            total_energy += static_cast<double>(batch_.energy_data()[i]);
        }

        // Convert gradients back to f64 numpy arrays
        py::array_t<double> grad_x({static_cast<py::ssize_t>(num_atoms)});
        py::array_t<double> grad_y({static_cast<py::ssize_t>(num_atoms)});
        py::array_t<double> grad_z({static_cast<py::ssize_t>(num_atoms)});

        auto gx = grad_x.mutable_unchecked<1>();
        auto gy = grad_y.mutable_unchecked<1>();
        auto gz = grad_z.mutable_unchecked<1>();

        for (std::uint32_t i = 0; i < num_atoms; ++i) {
            gx(i) = static_cast<double>(grad_out[i * 3 + 0]);
            gy(i) = static_cast<double>(grad_out[i * 3 + 1]);
            gz(i) = static_cast<double>(grad_out[i * 3 + 2]);
        }

        return py::make_tuple(total_energy, grad_x, grad_y, grad_z);
#else
        throw std::runtime_error("CUDA support not available");
#endif
    }
};

#pragma endregion

#pragma region Internal Helpers

/**
 * @brief Check if CUDA support is available at compile time.
 */
inline bool supports_cuda_() noexcept {
#if defined(__CUDACC__)
    return true;
#else
    return false;
#endif
}

/**
 * @brief Check if AVX-512 F32 support is available at compile time.
 */
inline bool supports_avx512_f32_() noexcept {
#if defined(__AVX512F__)
    return true;
#else
    return false;
#endif
}

/**
 * @brief Check if AVX-512 BF16 support is available at compile time.
 */
inline bool supports_avx512_bf16_() noexcept {
#if defined(__AVX512BF16__)
    return true;
#else
    return false;
#endif
}

/**
 * @brief Check if AVX-512 FP16 support is available at compile time.
 */
inline bool supports_avx512_fp16_() noexcept {
#if defined(__AVX512FP16__)
    return true;
#else
    return false;
#endif
}

#pragma endregion

#pragma region Serial CPU Kernels

/**
 * @brief Compute van der Waals energy using serial CPU kernel (F32).
 */
float compute_vdw_energy_serial_f32_(py::array_t<float> coords_x, py::array_t<float> coords_y,
                                     py::array_t<float> coords_z, py::array_t<std::int32_t> pair_atom_i,
                                     py::array_t<std::int32_t> pair_atom_j, py::array_t<float> pair_epsilon,
                                     py::array_t<float> pair_equilibrium_distance) {
    auto const buf_x = coords_x.request();
    auto const buf_y = coords_y.request();
    auto const buf_z = coords_z.request();
    auto const buf_i = pair_atom_i.request();
    auto const buf_j = pair_atom_j.request();
    auto const buf_eps = pair_epsilon.request();
    auto const buf_r0 = pair_equilibrium_distance.request();

    std::int32_t const num_pairs = static_cast<std::int32_t>(buf_i.size);

    return compute_vdw_energy<float>(static_cast<float const*>(buf_x.ptr), static_cast<float const*>(buf_y.ptr),
                                     static_cast<float const*>(buf_z.ptr), static_cast<std::int32_t const*>(buf_i.ptr),
                                     static_cast<std::int32_t const*>(buf_j.ptr),
                                     static_cast<float const*>(buf_eps.ptr), static_cast<float const*>(buf_r0.ptr), 0,
                                     num_pairs);
}

/**
 * @brief Compute van der Waals energy using serial CPU kernel (F64).
 */
double compute_vdw_energy_serial_f64_(py::array_t<double> coords_x, py::array_t<double> coords_y,
                                      py::array_t<double> coords_z, py::array_t<std::int32_t> pair_atom_i,
                                      py::array_t<std::int32_t> pair_atom_j, py::array_t<double> pair_epsilon,
                                      py::array_t<double> pair_equilibrium_distance) {
    auto const buf_x = coords_x.request();
    auto const buf_y = coords_y.request();
    auto const buf_z = coords_z.request();
    auto const buf_i = pair_atom_i.request();
    auto const buf_j = pair_atom_j.request();
    auto const buf_eps = pair_epsilon.request();
    auto const buf_r0 = pair_equilibrium_distance.request();

    std::int32_t const num_pairs = static_cast<std::int32_t>(buf_i.size);

    return compute_vdw_energy<double>(
        static_cast<double const*>(buf_x.ptr), static_cast<double const*>(buf_y.ptr),
        static_cast<double const*>(buf_z.ptr), static_cast<std::int32_t const*>(buf_i.ptr),
        static_cast<std::int32_t const*>(buf_j.ptr), static_cast<double const*>(buf_eps.ptr),
        static_cast<double const*>(buf_r0.ptr), 0, num_pairs);
}

/**
 * @brief Compute electrostatic energy using serial CPU kernel (F32).
 */
float compute_electrostatic_energy_serial_f32_(py::array_t<float> coords_x, py::array_t<float> coords_y,
                                               py::array_t<float> coords_z, py::array_t<std::int32_t> pair_atom_i,
                                               py::array_t<std::int32_t> pair_atom_j, py::array_t<float> pair_charge_i,
                                               py::array_t<float> pair_charge_j, py::array_t<std::int32_t> pair_is_1_4,
                                               float scaling_1_4) {
    auto const buf_x = coords_x.request();
    auto const buf_y = coords_y.request();
    auto const buf_z = coords_z.request();
    auto const buf_i = pair_atom_i.request();
    auto const buf_j = pair_atom_j.request();
    auto const buf_qi = pair_charge_i.request();
    auto const buf_qj = pair_charge_j.request();
    auto const buf_is_1_4 = pair_is_1_4.request();

    std::int32_t const num_pairs = static_cast<std::int32_t>(buf_i.size);

    return compute_electrostatic_energy<float>(
        static_cast<float const*>(buf_x.ptr), static_cast<float const*>(buf_y.ptr),
        static_cast<float const*>(buf_z.ptr), static_cast<std::int32_t const*>(buf_i.ptr),
        static_cast<std::int32_t const*>(buf_j.ptr), static_cast<float const*>(buf_qi.ptr),
        static_cast<float const*>(buf_qj.ptr), static_cast<std::int32_t const*>(buf_is_1_4.ptr), 0, num_pairs,
        scaling_1_4);
}

/**
 * @brief Compute electrostatic energy using serial CPU kernel (F64).
 */
double compute_electrostatic_energy_serial_f64_(py::array_t<double> coords_x, py::array_t<double> coords_y,
                                                py::array_t<double> coords_z, py::array_t<std::int32_t> pair_atom_i,
                                                py::array_t<std::int32_t> pair_atom_j,
                                                py::array_t<double> pair_charge_i, py::array_t<double> pair_charge_j,
                                                py::array_t<std::int32_t> pair_is_1_4, double scaling_1_4) {
    auto const buf_x = coords_x.request();
    auto const buf_y = coords_y.request();
    auto const buf_z = coords_z.request();
    auto const buf_i = pair_atom_i.request();
    auto const buf_j = pair_atom_j.request();
    auto const buf_qi = pair_charge_i.request();
    auto const buf_qj = pair_charge_j.request();
    auto const buf_is_1_4 = pair_is_1_4.request();

    std::int32_t const num_pairs = static_cast<std::int32_t>(buf_i.size);

    return compute_electrostatic_energy<double>(
        static_cast<double const*>(buf_x.ptr), static_cast<double const*>(buf_y.ptr),
        static_cast<double const*>(buf_z.ptr), static_cast<std::int32_t const*>(buf_i.ptr),
        static_cast<std::int32_t const*>(buf_j.ptr), static_cast<double const*>(buf_qi.ptr),
        static_cast<double const*>(buf_qj.ptr), static_cast<std::int32_t const*>(buf_is_1_4.ptr), 0, num_pairs,
        scaling_1_4);
}

#pragma endregion

#pragma region AVX-512 Skylake F32 Kernels

#if defined(__AVX512F__)

/**
 * @brief Compute van der Waals energy using AVX-512 F32 (Skylake).
 */
float compute_vdw_energy_skylake_f32_(py::array_t<float> coords_x, py::array_t<float> coords_y,
                                      py::array_t<float> coords_z, py::array_t<std::int32_t> pair_atom_i,
                                      py::array_t<std::int32_t> pair_atom_j, py::array_t<float> pair_epsilon,
                                      py::array_t<float> pair_equilibrium_distance) {
    auto const buf_x = coords_x.request();
    auto const buf_y = coords_y.request();
    auto const buf_z = coords_z.request();
    auto const buf_i = pair_atom_i.request();
    auto const buf_j = pair_atom_j.request();
    auto const buf_eps = pair_epsilon.request();
    auto const buf_r0 = pair_equilibrium_distance.request();

    std::int32_t const num_pairs = static_cast<std::int32_t>(buf_i.size);

    return compute_vdw_energy_skylake(static_cast<float const*>(buf_x.ptr), static_cast<float const*>(buf_y.ptr),
                                      static_cast<float const*>(buf_z.ptr), static_cast<std::int32_t const*>(buf_i.ptr),
                                      static_cast<std::int32_t const*>(buf_j.ptr),
                                      static_cast<float const*>(buf_eps.ptr), static_cast<float const*>(buf_r0.ptr), 0,
                                      num_pairs);
}

/**
 * @brief Compute electrostatic energy using AVX-512 F32 (Skylake).
 */
float compute_electrostatic_energy_skylake_f32_(py::array_t<float> coords_x, py::array_t<float> coords_y,
                                                py::array_t<float> coords_z, py::array_t<std::int32_t> pair_atom_i,
                                                py::array_t<std::int32_t> pair_atom_j, py::array_t<float> pair_charge_i,
                                                py::array_t<float> pair_charge_j, py::array_t<std::int32_t> pair_is_1_4,
                                                float scaling_1_4) {
    auto const buf_x = coords_x.request();
    auto const buf_y = coords_y.request();
    auto const buf_z = coords_z.request();
    auto const buf_i = pair_atom_i.request();
    auto const buf_j = pair_atom_j.request();
    auto const buf_qi = pair_charge_i.request();
    auto const buf_qj = pair_charge_j.request();
    auto const buf_is_1_4 = pair_is_1_4.request();

    std::int32_t const num_pairs = static_cast<std::int32_t>(buf_i.size);

    return compute_electrostatic_energy_skylake(
        static_cast<float const*>(buf_x.ptr), static_cast<float const*>(buf_y.ptr),
        static_cast<float const*>(buf_z.ptr), static_cast<std::int32_t const*>(buf_i.ptr),
        static_cast<std::int32_t const*>(buf_j.ptr), static_cast<float const*>(buf_qi.ptr),
        static_cast<float const*>(buf_qj.ptr), static_cast<std::int32_t const*>(buf_is_1_4.ptr), 0, num_pairs,
        scaling_1_4);
}

#endif // __AVX512F__

#pragma endregion

#pragma region AVX-512 Genoa BF16 Kernels

#if defined(__AVX512BF16__)

/**
 * @brief Compute van der Waals energy using AVX-512 BF16 (Genoa).
 */
float compute_vdw_energy_genoa_bf16_(py::array_t<std::uint16_t> coords_x, py::array_t<std::uint16_t> coords_y,
                                     py::array_t<std::uint16_t> coords_z, py::array_t<std::int32_t> pair_atom_i,
                                     py::array_t<std::int32_t> pair_atom_j, py::array_t<std::uint16_t> pair_epsilon,
                                     py::array_t<std::uint16_t> pair_equilibrium_distance) {
    auto const buf_x = coords_x.request();
    auto const buf_y = coords_y.request();
    auto const buf_z = coords_z.request();
    auto const buf_i = pair_atom_i.request();
    auto const buf_j = pair_atom_j.request();
    auto const buf_eps = pair_epsilon.request();
    auto const buf_r0 = pair_equilibrium_distance.request();

    std::int32_t const num_pairs = static_cast<std::int32_t>(buf_i.size);

    return compute_vdw_energy_genoa(
        static_cast<std::uint16_t const*>(buf_x.ptr), static_cast<std::uint16_t const*>(buf_y.ptr),
        static_cast<std::uint16_t const*>(buf_z.ptr), static_cast<std::int32_t const*>(buf_i.ptr),
        static_cast<std::int32_t const*>(buf_j.ptr), static_cast<std::uint16_t const*>(buf_eps.ptr),
        static_cast<std::uint16_t const*>(buf_r0.ptr), 0, num_pairs);
}

#endif // __AVX512BF16__

#pragma endregion

#pragma region AVX-512 Turin FP16 Kernels

#if defined(__AVX512FP16__)

/**
 * @brief Compute van der Waals energy using AVX-512 FP16 (Turin).
 *
 * Uses std::uint16_t for Python binding compatibility (NumPy doesn't know _Float16).
 * IEEE 754 FP16 and _Float16 have identical binary representation.
 */
float compute_vdw_energy_turin_fp16_(py::array_t<std::uint16_t> coords_x, py::array_t<std::uint16_t> coords_y,
                                     py::array_t<std::uint16_t> coords_z, py::array_t<std::int32_t> pair_atom_i,
                                     py::array_t<std::int32_t> pair_atom_j, py::array_t<std::uint16_t> pair_epsilon,
                                     py::array_t<std::uint16_t> pair_equilibrium_distance) {
    auto const buf_x = coords_x.request();
    auto const buf_y = coords_y.request();
    auto const buf_z = coords_z.request();
    auto const buf_i = pair_atom_i.request();
    auto const buf_j = pair_atom_j.request();
    auto const buf_eps = pair_epsilon.request();
    auto const buf_r0 = pair_equilibrium_distance.request();

    std::int32_t const num_pairs = static_cast<std::int32_t>(buf_i.size);

    // Cast uint16_t* to _Float16* - same binary representation
    return compute_vdw_energy_turin(
        reinterpret_cast<_Float16 const*>(buf_x.ptr), reinterpret_cast<_Float16 const*>(buf_y.ptr),
        reinterpret_cast<_Float16 const*>(buf_z.ptr), static_cast<std::int32_t const*>(buf_i.ptr),
        static_cast<std::int32_t const*>(buf_j.ptr), reinterpret_cast<_Float16 const*>(buf_eps.ptr),
        reinterpret_cast<_Float16 const*>(buf_r0.ptr), 0, num_pairs);
}

#endif // __AVX512FP16__

#pragma endregion

#pragma region CUDA GPU Kernels

#if defined(__CUDACC__)

/**
 * @brief Compute van der Waals energy using CUDA F16.
 */
float compute_vdw_energy_cuda_f16_(py::array_t<std::uint16_t> coords_x, py::array_t<std::uint16_t> coords_y,
                                   py::array_t<std::uint16_t> coords_z, py::array_t<std::int32_t> nb_i,
                                   py::array_t<std::int32_t> nb_j, py::array_t<std::uint16_t> nb_epsilon,
                                   py::array_t<std::uint16_t> nb_r_star, py::array_t<std::int32_t> mol_nb_start,
                                   py::array_t<std::int32_t> mol_nb_end, std::size_t n_molecules) {
    auto const buf_x = coords_x.request();
    auto const buf_y = coords_y.request();
    auto const buf_z = coords_z.request();
    auto const buf_nb_i = nb_i.request();
    auto const buf_nb_j = nb_j.request();
    auto const buf_eps = nb_epsilon.request();
    auto const buf_rstar = nb_r_star.request();
    auto const buf_start = mol_nb_start.request();
    auto const buf_end = mol_nb_end.request();

    std::size_t const num_atoms = buf_x.size;
    std::size_t const num_pairs = buf_nb_i.size;

    // Allocate device memory
    half* d_coords_x = nullptr;
    half* d_coords_y = nullptr;
    half* d_coords_z = nullptr;
    std::int32_t* d_nb_i = nullptr;
    std::int32_t* d_nb_j = nullptr;
    half* d_nb_epsilon = nullptr;
    half* d_nb_r_star = nullptr;
    std::int32_t* d_mol_nb_start = nullptr;
    std::int32_t* d_mol_nb_end = nullptr;
    float* d_out_vdw = nullptr;

    cudaMalloc(&d_coords_x, num_atoms * sizeof(half));
    cudaMalloc(&d_coords_y, num_atoms * sizeof(half));
    cudaMalloc(&d_coords_z, num_atoms * sizeof(half));
    cudaMalloc(&d_nb_i, num_pairs * sizeof(std::int32_t));
    cudaMalloc(&d_nb_j, num_pairs * sizeof(std::int32_t));
    cudaMalloc(&d_nb_epsilon, num_pairs * sizeof(half));
    cudaMalloc(&d_nb_r_star, num_pairs * sizeof(half));
    cudaMalloc(&d_mol_nb_start, n_molecules * sizeof(std::int32_t));
    cudaMalloc(&d_mol_nb_end, n_molecules * sizeof(std::int32_t));
    cudaMalloc(&d_out_vdw, n_molecules * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_coords_x, buf_x.ptr, num_atoms * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords_y, buf_y.ptr, num_atoms * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords_z, buf_z.ptr, num_atoms * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_i, buf_nb_i.ptr, num_pairs * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_j, buf_nb_j.ptr, num_pairs * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_epsilon, buf_eps.ptr, num_pairs * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_r_star, buf_rstar.ptr, num_pairs * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mol_nb_start, buf_start.ptr, n_molecules * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mol_nb_end, buf_end.ptr, n_molecules * sizeof(std::int32_t), cudaMemcpyHostToDevice);

    // Launch kernel (1 block per molecule, 32 threads per block)
    mmff94_vdw_energy_kernel_f16<<<n_molecules, warp_size>>>(d_coords_x, d_coords_y, d_coords_z, d_nb_i, d_nb_j,
                                                             d_nb_epsilon, d_nb_r_star, d_mol_nb_start, d_mol_nb_end,
                                                             d_out_vdw, static_cast<std::uint32_t>(n_molecules));

    // Copy result back (sum all molecule energies)
    float* h_out_vdw = new float[n_molecules];
    cudaMemcpy(h_out_vdw, d_out_vdw, n_molecules * sizeof(float), cudaMemcpyDeviceToHost);

    float total_energy = 0.0f;
    for (std::size_t i = 0; i < n_molecules; ++i) {
        total_energy += h_out_vdw[i];
    }
    delete[] h_out_vdw;

    // Free device memory
    cudaFree(d_coords_x);
    cudaFree(d_coords_y);
    cudaFree(d_coords_z);
    cudaFree(d_nb_i);
    cudaFree(d_nb_j);
    cudaFree(d_nb_epsilon);
    cudaFree(d_nb_r_star);
    cudaFree(d_mol_nb_start);
    cudaFree(d_mol_nb_end);
    cudaFree(d_out_vdw);

    return total_energy;
}

/**
 * @brief Compute van der Waals energy using CUDA F32.
 */
float compute_vdw_energy_cuda_f32_(py::array_t<float> coords_x, py::array_t<float> coords_y,
                                   py::array_t<float> coords_z, py::array_t<std::int32_t> nb_i,
                                   py::array_t<std::int32_t> nb_j, py::array_t<float> nb_epsilon,
                                   py::array_t<float> nb_r_star, py::array_t<std::int32_t> mol_nb_start,
                                   py::array_t<std::int32_t> mol_nb_end, std::size_t n_molecules) {
    auto const buf_x = coords_x.request();
    auto const buf_y = coords_y.request();
    auto const buf_z = coords_z.request();
    auto const buf_nb_i = nb_i.request();
    auto const buf_nb_j = nb_j.request();
    auto const buf_eps = nb_epsilon.request();
    auto const buf_rstar = nb_r_star.request();
    auto const buf_start = mol_nb_start.request();
    auto const buf_end = mol_nb_end.request();

    std::size_t const num_atoms = buf_x.size;
    std::size_t const num_pairs = buf_nb_i.size;

    // Allocate device memory
    float* d_coords_x = nullptr;
    float* d_coords_y = nullptr;
    float* d_coords_z = nullptr;
    std::int32_t* d_nb_i = nullptr;
    std::int32_t* d_nb_j = nullptr;
    float* d_nb_epsilon = nullptr;
    float* d_nb_r_star = nullptr;
    std::int32_t* d_mol_nb_start = nullptr;
    std::int32_t* d_mol_nb_end = nullptr;
    float* d_out_vdw = nullptr;

    cudaMalloc(&d_coords_x, num_atoms * sizeof(float));
    cudaMalloc(&d_coords_y, num_atoms * sizeof(float));
    cudaMalloc(&d_coords_z, num_atoms * sizeof(float));
    cudaMalloc(&d_nb_i, num_pairs * sizeof(std::int32_t));
    cudaMalloc(&d_nb_j, num_pairs * sizeof(std::int32_t));
    cudaMalloc(&d_nb_epsilon, num_pairs * sizeof(float));
    cudaMalloc(&d_nb_r_star, num_pairs * sizeof(float));
    cudaMalloc(&d_mol_nb_start, n_molecules * sizeof(std::int32_t));
    cudaMalloc(&d_mol_nb_end, n_molecules * sizeof(std::int32_t));
    cudaMalloc(&d_out_vdw, n_molecules * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_coords_x, buf_x.ptr, num_atoms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords_y, buf_y.ptr, num_atoms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords_z, buf_z.ptr, num_atoms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_i, buf_nb_i.ptr, num_pairs * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_j, buf_nb_j.ptr, num_pairs * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_epsilon, buf_eps.ptr, num_pairs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_r_star, buf_rstar.ptr, num_pairs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mol_nb_start, buf_start.ptr, n_molecules * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mol_nb_end, buf_end.ptr, n_molecules * sizeof(std::int32_t), cudaMemcpyHostToDevice);

    // Launch kernel (1 block per molecule, 32 threads per block)
    mmff94_vdw_energy_kernel_f32<<<n_molecules, warp_size>>>(d_coords_x, d_coords_y, d_coords_z, d_nb_i, d_nb_j,
                                                             d_nb_epsilon, d_nb_r_star, d_mol_nb_start, d_mol_nb_end,
                                                             d_out_vdw, static_cast<std::uint32_t>(n_molecules));

    // Copy result back (sum all molecule energies)
    float* h_out_vdw = new float[n_molecules];
    cudaMemcpy(h_out_vdw, d_out_vdw, n_molecules * sizeof(float), cudaMemcpyDeviceToHost);

    float total_energy = 0.0f;
    for (std::size_t i = 0; i < n_molecules; ++i) {
        total_energy += h_out_vdw[i];
    }
    delete[] h_out_vdw;

    // Free device memory
    cudaFree(d_coords_x);
    cudaFree(d_coords_y);
    cudaFree(d_coords_z);
    cudaFree(d_nb_i);
    cudaFree(d_nb_j);
    cudaFree(d_nb_epsilon);
    cudaFree(d_nb_r_star);
    cudaFree(d_mol_nb_start);
    cudaFree(d_mol_nb_end);
    cudaFree(d_out_vdw);

    return total_energy;
}

/**
 * @brief Compute electrostatic energy using CUDA F16.
 */
float compute_electrostatic_energy_cuda_f16_(py::array_t<std::uint16_t> coords_x, py::array_t<std::uint16_t> coords_y,
                                             py::array_t<std::uint16_t> coords_z, py::array_t<std::int32_t> nb_i,
                                             py::array_t<std::int32_t> nb_j, py::array_t<std::uint16_t> nb_qi,
                                             py::array_t<std::uint16_t> nb_qj, py::array_t<std::int32_t> nb_is_1_4,
                                             py::array_t<std::int32_t> mol_nb_start,
                                             py::array_t<std::int32_t> mol_nb_end, std::size_t n_molecules) {
    auto const buf_x = coords_x.request();
    auto const buf_y = coords_y.request();
    auto const buf_z = coords_z.request();
    auto const buf_nb_i = nb_i.request();
    auto const buf_nb_j = nb_j.request();
    auto const buf_qi = nb_qi.request();
    auto const buf_qj = nb_qj.request();
    auto const buf_is_1_4 = nb_is_1_4.request();
    auto const buf_start = mol_nb_start.request();
    auto const buf_end = mol_nb_end.request();

    std::size_t const num_atoms = buf_x.size;
    std::size_t const num_pairs = buf_nb_i.size;

    // Allocate device memory
    half* d_coords_x = nullptr;
    half* d_coords_y = nullptr;
    half* d_coords_z = nullptr;
    std::int32_t* d_nb_i = nullptr;
    std::int32_t* d_nb_j = nullptr;
    half* d_nb_qi = nullptr;
    half* d_nb_qj = nullptr;
    std::int32_t* d_nb_is_1_4 = nullptr;
    std::int32_t* d_mol_nb_start = nullptr;
    std::int32_t* d_mol_nb_end = nullptr;
    float* d_out_elec = nullptr;

    cudaMalloc(&d_coords_x, num_atoms * sizeof(half));
    cudaMalloc(&d_coords_y, num_atoms * sizeof(half));
    cudaMalloc(&d_coords_z, num_atoms * sizeof(half));
    cudaMalloc(&d_nb_i, num_pairs * sizeof(std::int32_t));
    cudaMalloc(&d_nb_j, num_pairs * sizeof(std::int32_t));
    cudaMalloc(&d_nb_qi, num_pairs * sizeof(half));
    cudaMalloc(&d_nb_qj, num_pairs * sizeof(half));
    cudaMalloc(&d_nb_is_1_4, num_pairs * sizeof(std::int32_t));
    cudaMalloc(&d_mol_nb_start, n_molecules * sizeof(std::int32_t));
    cudaMalloc(&d_mol_nb_end, n_molecules * sizeof(std::int32_t));
    cudaMalloc(&d_out_elec, n_molecules * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_coords_x, buf_x.ptr, num_atoms * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords_y, buf_y.ptr, num_atoms * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords_z, buf_z.ptr, num_atoms * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_i, buf_nb_i.ptr, num_pairs * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_j, buf_nb_j.ptr, num_pairs * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_qi, buf_qi.ptr, num_pairs * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_qj, buf_qj.ptr, num_pairs * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_is_1_4, buf_is_1_4.ptr, num_pairs * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mol_nb_start, buf_start.ptr, n_molecules * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mol_nb_end, buf_end.ptr, n_molecules * sizeof(std::int32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    mmff94_elec_energy_kernel_f16<<<n_molecules, warp_size>>>(
        d_coords_x, d_coords_y, d_coords_z, d_nb_i, d_nb_j, d_nb_qi, d_nb_qj, d_nb_is_1_4, d_mol_nb_start, d_mol_nb_end,
        d_out_elec, static_cast<std::uint32_t>(n_molecules));

    // Copy result back
    float* h_out_elec = new float[n_molecules];
    cudaMemcpy(h_out_elec, d_out_elec, n_molecules * sizeof(float), cudaMemcpyDeviceToHost);

    float total_energy = 0.0f;
    for (std::size_t i = 0; i < n_molecules; ++i) {
        total_energy += h_out_elec[i];
    }
    delete[] h_out_elec;

    // Free device memory
    cudaFree(d_coords_x);
    cudaFree(d_coords_y);
    cudaFree(d_coords_z);
    cudaFree(d_nb_i);
    cudaFree(d_nb_j);
    cudaFree(d_nb_qi);
    cudaFree(d_nb_qj);
    cudaFree(d_nb_is_1_4);
    cudaFree(d_mol_nb_start);
    cudaFree(d_mol_nb_end);
    cudaFree(d_out_elec);

    return total_energy;
}

/**
 * @brief Compute electrostatic energy using CUDA F32.
 */
float compute_electrostatic_energy_cuda_f32_(py::array_t<float> coords_x, py::array_t<float> coords_y,
                                             py::array_t<float> coords_z, py::array_t<std::int32_t> nb_i,
                                             py::array_t<std::int32_t> nb_j, py::array_t<float> nb_qi,
                                             py::array_t<float> nb_qj, py::array_t<std::int32_t> nb_is_1_4,
                                             py::array_t<std::int32_t> mol_nb_start,
                                             py::array_t<std::int32_t> mol_nb_end, std::size_t n_molecules) {
    auto const buf_x = coords_x.request();
    auto const buf_y = coords_y.request();
    auto const buf_z = coords_z.request();
    auto const buf_nb_i = nb_i.request();
    auto const buf_nb_j = nb_j.request();
    auto const buf_qi = nb_qi.request();
    auto const buf_qj = nb_qj.request();
    auto const buf_is_1_4 = nb_is_1_4.request();
    auto const buf_start = mol_nb_start.request();
    auto const buf_end = mol_nb_end.request();

    std::size_t const num_atoms = buf_x.size;
    std::size_t const num_pairs = buf_nb_i.size;

    // Allocate device memory
    float* d_coords_x = nullptr;
    float* d_coords_y = nullptr;
    float* d_coords_z = nullptr;
    std::int32_t* d_nb_i = nullptr;
    std::int32_t* d_nb_j = nullptr;
    float* d_nb_qi = nullptr;
    float* d_nb_qj = nullptr;
    std::int32_t* d_nb_is_1_4 = nullptr;
    std::int32_t* d_mol_nb_start = nullptr;
    std::int32_t* d_mol_nb_end = nullptr;
    float* d_out_elec = nullptr;

    cudaMalloc(&d_coords_x, num_atoms * sizeof(float));
    cudaMalloc(&d_coords_y, num_atoms * sizeof(float));
    cudaMalloc(&d_coords_z, num_atoms * sizeof(float));
    cudaMalloc(&d_nb_i, num_pairs * sizeof(std::int32_t));
    cudaMalloc(&d_nb_j, num_pairs * sizeof(std::int32_t));
    cudaMalloc(&d_nb_qi, num_pairs * sizeof(float));
    cudaMalloc(&d_nb_qj, num_pairs * sizeof(float));
    cudaMalloc(&d_nb_is_1_4, num_pairs * sizeof(std::int32_t));
    cudaMalloc(&d_mol_nb_start, n_molecules * sizeof(std::int32_t));
    cudaMalloc(&d_mol_nb_end, n_molecules * sizeof(std::int32_t));
    cudaMalloc(&d_out_elec, n_molecules * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_coords_x, buf_x.ptr, num_atoms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords_y, buf_y.ptr, num_atoms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords_z, buf_z.ptr, num_atoms * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_i, buf_nb_i.ptr, num_pairs * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_j, buf_nb_j.ptr, num_pairs * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_qi, buf_qi.ptr, num_pairs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_qj, buf_qj.ptr, num_pairs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb_is_1_4, buf_is_1_4.ptr, num_pairs * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mol_nb_start, buf_start.ptr, n_molecules * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mol_nb_end, buf_end.ptr, n_molecules * sizeof(std::int32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    mmff94_elec_energy_kernel_f32<<<n_molecules, warp_size>>>(
        d_coords_x, d_coords_y, d_coords_z, d_nb_i, d_nb_j, d_nb_qi, d_nb_qj, d_nb_is_1_4, d_mol_nb_start, d_mol_nb_end,
        d_out_elec, static_cast<std::uint32_t>(n_molecules));

    // Copy result back
    float* h_out_elec = new float[n_molecules];
    cudaMemcpy(h_out_elec, d_out_elec, n_molecules * sizeof(float), cudaMemcpyDeviceToHost);

    float total_energy = 0.0f;
    for (std::size_t i = 0; i < n_molecules; ++i) {
        total_energy += h_out_elec[i];
    }
    delete[] h_out_elec;

    // Free device memory
    cudaFree(d_coords_x);
    cudaFree(d_coords_y);
    cudaFree(d_coords_z);
    cudaFree(d_nb_i);
    cudaFree(d_nb_j);
    cudaFree(d_nb_qi);
    cudaFree(d_nb_qj);
    cudaFree(d_nb_is_1_4);
    cudaFree(d_mol_nb_start);
    cudaFree(d_mol_nb_end);
    cudaFree(d_out_elec);

    return total_energy;
}

#endif // __CUDACC__

#pragma endregion

} // namespace smallfield

#pragma region Pybind11 Module Definition

PYBIND11_MODULE(_smallfield, m) {
    m.doc() = "SmallField MMFF94 force field implementation with CPU and GPU support";

    // Capability detection
    m.def("supports_cuda", &smallfield::supports_cuda_, "Check if CUDA support is available");
    m.def("supports_avx512_f32", &smallfield::supports_avx512_f32_, "Check if AVX-512 F32 support is available");
    m.def("supports_avx512_bf16", &smallfield::supports_avx512_bf16_, "Check if AVX-512 BF16 support is available");
    m.def("supports_avx512_fp16", &smallfield::supports_avx512_fp16_, "Check if AVX-512 FP16 support is available");

    // Convenience aliases for capability detection
    m.def("has_cuda", &smallfield::supports_cuda_, "Check if CUDA support is available");
    m.def("has_skylake", &smallfield::supports_avx512_f32_, "Check if Skylake AVX-512 support is available");
    m.def("has_genoa", &smallfield::supports_avx512_bf16_, "Check if Genoa AVX-512 BF16 support is available");
    m.def("has_turin", &smallfield::supports_avx512_fp16_, "Check if Turin AVX-512 FP16 support is available");

    // Serial F32 and F64
    m.def("compute_vdw_energy_serial_f32", &smallfield::compute_vdw_energy_serial_f32_,
          "Compute van der Waals energy using serial CPU kernel (F32)", py::arg("coords_x"), py::arg("coords_y"),
          py::arg("coords_z"), py::arg("pair_atom_i"), py::arg("pair_atom_j"), py::arg("pair_epsilon"),
          py::arg("pair_equilibrium_distance"));

    m.def("compute_vdw_energy_serial_f64", &smallfield::compute_vdw_energy_serial_f64_,
          "Compute van der Waals energy using serial CPU kernel (F64)", py::arg("coords_x"), py::arg("coords_y"),
          py::arg("coords_z"), py::arg("pair_atom_i"), py::arg("pair_atom_j"), py::arg("pair_epsilon"),
          py::arg("pair_equilibrium_distance"));

    m.def("compute_electrostatic_energy_serial_f32", &smallfield::compute_electrostatic_energy_serial_f32_,
          "Compute electrostatic energy using serial CPU kernel (F32)", py::arg("coords_x"), py::arg("coords_y"),
          py::arg("coords_z"), py::arg("pair_atom_i"), py::arg("pair_atom_j"), py::arg("pair_charge_i"),
          py::arg("pair_charge_j"), py::arg("pair_is_1_4"), py::arg("scaling_1_4") = 0.75f);

    m.def("compute_electrostatic_energy_serial_f64", &smallfield::compute_electrostatic_energy_serial_f64_,
          "Compute electrostatic energy using serial CPU kernel (F64)", py::arg("coords_x"), py::arg("coords_y"),
          py::arg("coords_z"), py::arg("pair_atom_i"), py::arg("pair_atom_j"), py::arg("pair_charge_i"),
          py::arg("pair_charge_j"), py::arg("pair_is_1_4"), py::arg("scaling_1_4") = 0.75);

#if defined(__AVX512F__)
    // Skylake AVX-512 F32
    m.def("compute_vdw_energy_skylake_f32", &smallfield::compute_vdw_energy_skylake_f32_,
          "Compute van der Waals energy using AVX-512 F32 (Skylake)", py::arg("coords_x"), py::arg("coords_y"),
          py::arg("coords_z"), py::arg("pair_atom_i"), py::arg("pair_atom_j"), py::arg("pair_epsilon"),
          py::arg("pair_equilibrium_distance"));

    m.def("compute_electrostatic_energy_skylake_f32", &smallfield::compute_electrostatic_energy_skylake_f32_,
          "Compute electrostatic energy using AVX-512 F32 (Skylake)", py::arg("coords_x"), py::arg("coords_y"),
          py::arg("coords_z"), py::arg("pair_atom_i"), py::arg("pair_atom_j"), py::arg("pair_charge_i"),
          py::arg("pair_charge_j"), py::arg("pair_is_1_4"), py::arg("scaling_1_4") = 0.75f);
#endif

#if defined(__AVX512BF16__)
    // Genoa AVX-512 BF16
    m.def("compute_vdw_energy_genoa_bf16", &smallfield::compute_vdw_energy_genoa_bf16_,
          "Compute van der Waals energy using AVX-512 BF16 (Genoa)", py::arg("coords_x"), py::arg("coords_y"),
          py::arg("coords_z"), py::arg("pair_atom_i"), py::arg("pair_atom_j"), py::arg("pair_epsilon"),
          py::arg("pair_equilibrium_distance"));
#endif

#if defined(__AVX512FP16__)
    // Turin AVX-512 FP16
    m.def("compute_vdw_energy_turin_fp16", &smallfield::compute_vdw_energy_turin_fp16_,
          "Compute van der Waals energy using AVX-512 FP16 (Turin)", py::arg("coords_x"), py::arg("coords_y"),
          py::arg("coords_z"), py::arg("pair_atom_i"), py::arg("pair_atom_j"), py::arg("pair_epsilon"),
          py::arg("pair_equilibrium_distance"));
#endif

#if defined(__CUDACC__)
    // CUDA F16 and F32
    m.def("compute_vdw_energy_cuda_f16", &smallfield::compute_vdw_energy_cuda_f16_,
          "Compute van der Waals energy using CUDA F16", py::arg("coords_x"), py::arg("coords_y"), py::arg("coords_z"),
          py::arg("nb_i"), py::arg("nb_j"), py::arg("nb_epsilon"), py::arg("nb_r_star"), py::arg("mol_nb_start"),
          py::arg("mol_nb_end"), py::arg("n_molecules"));

    m.def("compute_vdw_energy_cuda_f32", &smallfield::compute_vdw_energy_cuda_f32_,
          "Compute van der Waals energy using CUDA F32", py::arg("coords_x"), py::arg("coords_y"), py::arg("coords_z"),
          py::arg("nb_i"), py::arg("nb_j"), py::arg("nb_epsilon"), py::arg("nb_r_star"), py::arg("mol_nb_start"),
          py::arg("mol_nb_end"), py::arg("n_molecules"));

    m.def("compute_electrostatic_energy_cuda_f16", &smallfield::compute_electrostatic_energy_cuda_f16_,
          "Compute electrostatic energy using CUDA F16", py::arg("coords_x"), py::arg("coords_y"), py::arg("coords_z"),
          py::arg("nb_i"), py::arg("nb_j"), py::arg("nb_qi"), py::arg("nb_qj"), py::arg("nb_is_1_4"),
          py::arg("mol_nb_start"), py::arg("mol_nb_end"), py::arg("n_molecules"));

    m.def("compute_electrostatic_energy_cuda_f32", &smallfield::compute_electrostatic_energy_cuda_f32_,
          "Compute electrostatic energy using CUDA F32", py::arg("coords_x"), py::arg("coords_y"), py::arg("coords_z"),
          py::arg("nb_i"), py::arg("nb_j"), py::arg("nb_qi"), py::arg("nb_qj"), py::arg("nb_is_1_4"),
          py::arg("mol_nb_start"), py::arg("mol_nb_end"), py::arg("n_molecules"));
#endif

    // context_t class
    py::class_<smallfield::context_t>(m, "Context", "Pre-allocated context for SmallField MMFF94 force field")
        .def(py::init<std::uint32_t, std::uint32_t, std::uint32_t>(), py::arg("total_atoms"), py::arg("total_pairs"),
             py::arg("max_molecules"), "Create a new context with pre-allocated buffers (rounded up for SIMD).")
        .def_property_readonly("capacity_atoms", &smallfield::context_t::capacity_atoms)
        .def_property_readonly("capacity_pairs", &smallfield::context_t::capacity_pairs)
        .def_property_readonly("capacity_molecules", &smallfield::context_t::capacity_molecules)
        // Batch VdW energy methods
        .def("vdw_skylake", &smallfield::context_t::vdw_skylake,
             "VdW batch energy using Skylake AVX-512 F32 with ForkUnion parallelization.", py::arg("coords_x"),
             py::arg("coords_y"), py::arg("coords_z"), py::arg("pair_atom_i"), py::arg("pair_atom_j"),
             py::arg("pair_epsilon"), py::arg("pair_r_star"), py::arg("mol_pair_start"), py::arg("mol_pair_end"),
             py::arg("n_molecules"))
        .def("vdw_cuda_f32", &smallfield::context_t::vdw_cuda_f32, "VdW batch energy using CUDA F32 kernel.",
             py::arg("coords_x"), py::arg("coords_y"), py::arg("coords_z"), py::arg("pair_atom_i"),
             py::arg("pair_atom_j"), py::arg("pair_epsilon"), py::arg("pair_r_star"), py::arg("mol_pair_start"),
             py::arg("mol_pair_end"), py::arg("n_molecules"))
        .def("vdw_cuda_f16", &smallfield::context_t::vdw_cuda_f16, "VdW batch energy using CUDA F16 kernel.",
             py::arg("coords_x"), py::arg("coords_y"), py::arg("coords_z"), py::arg("pair_atom_i"),
             py::arg("pair_atom_j"), py::arg("pair_epsilon"), py::arg("pair_r_star"), py::arg("mol_pair_start"),
             py::arg("mol_pair_end"), py::arg("n_molecules"))
        // Optimization methods
        .def("optimize_serial", &smallfield::context_t::optimize_serial,
             "Optimize geometry using serial F64 L-BFGS. Returns (final_energy, iterations, converged, rms_gradient).",
             py::arg("coords_x"), py::arg("coords_y"), py::arg("coords_z"), py::arg("pair_atom_i"),
             py::arg("pair_atom_j"), py::arg("pair_epsilon"), py::arg("pair_r_star"), py::arg("max_iterations") = 200,
             py::arg("gradient_threshold") = 1e-4)
        .def("optimize_skylake", &smallfield::context_t::optimize_skylake,
             "Optimize geometry using Skylake AVX-512 F32 L-BFGS. Returns (final_energy, iterations, converged, "
             "rms_gradient).",
             py::arg("coords_x"), py::arg("coords_y"), py::arg("coords_z"), py::arg("pair_atom_i"),
             py::arg("pair_atom_j"), py::arg("pair_epsilon"), py::arg("pair_r_star"), py::arg("max_iterations") = 200,
             py::arg("gradient_threshold") = 1e-4)
        .def("optimize_cuda_f32", &smallfield::context_t::optimize_cuda_f32,
             "Optimize geometry using CUDA F32 batch L-BFGS. Returns (final_energy, iterations, converged, "
             "rms_gradient).",
             py::arg("coords_x"), py::arg("coords_y"), py::arg("coords_z"), py::arg("pair_atom_i"),
             py::arg("pair_atom_j"), py::arg("pair_epsilon"), py::arg("pair_r_star"), py::arg("mol_pair_start"),
             py::arg("mol_pair_end"), py::arg("n_molecules"), py::arg("max_iterations") = 200,
             py::arg("gradient_threshold") = 1e-4)
        .def("optimize_cuda_f16", &smallfield::context_t::optimize_cuda_f16,
             "Optimize geometry using CUDA F16 batch L-BFGS. Returns (final_energy, iterations, converged, "
             "rms_gradient).",
             py::arg("coords_x"), py::arg("coords_y"), py::arg("coords_z"), py::arg("pair_atom_i"),
             py::arg("pair_atom_j"), py::arg("pair_epsilon"), py::arg("pair_r_star"), py::arg("mol_pair_start"),
             py::arg("mol_pair_end"), py::arg("n_molecules"), py::arg("max_iterations") = 200,
             py::arg("gradient_threshold") = 1e-4)
        // Atom-centric kernel (eliminates atomics and gathers)
        .def("energy_gradient_atom_centric_cuda_f32", &smallfield::context_t::energy_gradient_atom_centric_cuda_f32,
             "Compute energy and gradient using atom-centric CUDA F32 kernel. "
             "Eliminates atomics and gathers via register-based coordinate storage "
             "and symmetric adjacency list. Returns (total_energy, grad_x, grad_y, grad_z).",
             py::arg("coords_x"), py::arg("coords_y"), py::arg("coords_z"),
             py::arg("pair_atom_i"), py::arg("pair_atom_j"),
             py::arg("pair_epsilon"), py::arg("pair_r_star"),
             py::arg("pair_qi"), py::arg("pair_qj"), py::arg("pair_is_1_4"),
             py::arg("mol_pair_start"), py::arg("mol_pair_end"),
             py::arg("mol_atom_start"), py::arg("mol_atom_end"),
             py::arg("n_molecules"));
}

#pragma endregion
