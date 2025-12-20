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
#include <stdexcept>
#include <string>

#include "mmff94.hpp"

#if defined(__CUDACC__)
#include "mmff94.cuh"
#endif

namespace py = pybind11;

namespace smallfield {

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

    return compute_vdw_energy_skylake(
        static_cast<float const*>(buf_x.ptr), static_cast<float const*>(buf_y.ptr),
        static_cast<float const*>(buf_z.ptr), static_cast<std::int32_t const*>(buf_i.ptr),
        static_cast<std::int32_t const*>(buf_j.ptr), static_cast<float const*>(buf_eps.ptr),
        static_cast<float const*>(buf_r0.ptr), 0, num_pairs);
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
 */
float compute_vdw_energy_turin_fp16_(py::array_t<_Float16> coords_x, py::array_t<_Float16> coords_y,
                                     py::array_t<_Float16> coords_z, py::array_t<std::int32_t> pair_atom_i,
                                     py::array_t<std::int32_t> pair_atom_j, py::array_t<_Float16> pair_epsilon,
                                     py::array_t<_Float16> pair_equilibrium_distance) {
    auto const buf_x = coords_x.request();
    auto const buf_y = coords_y.request();
    auto const buf_z = coords_z.request();
    auto const buf_i = pair_atom_i.request();
    auto const buf_j = pair_atom_j.request();
    auto const buf_eps = pair_epsilon.request();
    auto const buf_r0 = pair_equilibrium_distance.request();

    std::int32_t const num_pairs = static_cast<std::int32_t>(buf_i.size);

    return compute_vdw_energy_turin(
        static_cast<_Float16 const*>(buf_x.ptr), static_cast<_Float16 const*>(buf_y.ptr),
        static_cast<_Float16 const*>(buf_z.ptr), static_cast<std::int32_t const*>(buf_i.ptr),
        static_cast<std::int32_t const*>(buf_j.ptr), static_cast<_Float16 const*>(buf_eps.ptr),
        static_cast<_Float16 const*>(buf_r0.ptr), 0, num_pairs);
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
                                                             d_out_vdw, static_cast<std::int32_t>(n_molecules));

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
                                                             d_out_vdw, static_cast<std::int32_t>(n_molecules));

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
        d_out_elec, static_cast<std::int32_t>(n_molecules));

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
        d_out_elec, static_cast<std::int32_t>(n_molecules));

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
}

#pragma endregion
