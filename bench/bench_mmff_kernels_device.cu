/**
 *  @file bench/bench_mmff_kernels_device.cu
 *  @brief Device-side launcher for the MMFF GPU kernel trio.
 *
 *  Nothing is copied back: the outputs run to 10 GB or more on a large batch, and the correctness of the
 *  terms is the parity suite's job rather than this one's.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`
#include <cstdio>  // `stderr`

#include <cuda_runtime.h>
#include <fmt/format.h>

#include "smallfield/mmff94/mmff94_extract_device.cuh"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_device.cuh"
#include "smallfield/mmff94/mmff94_typer_device.cuh"
#include "smallfield/types/storage.hpp"

#include "bench/bench_mmff_kernels_device.hpp"

namespace sfmmff = smallfield::mmff94;
namespace sfparam = smallfield::parameters;
namespace sft = smallfield::types;

namespace smallfield::bench {

char const *describe_trio_launch(trio_launch_t status) {
    switch (status) {
    case trio_launch_t::timed_k: return "timed";
    case trio_launch_t::table_upload_failed_k: return "could not upload the MMFF94 parameter tables";
    case trio_launch_t::allocation_failed_k: return "could not allocate the batch's device arrays";
    case trio_launch_t::kernel_failed_k: return "was stopped by a kernel error";
    }
    return "was refused";
}

/** Grow-only device storage over the benchmark's stream, which is the one the kernels and events ride. */
template <typename value_type_>
using device_array = types::buffer<value_type_, types::device_allocator_t>;

/** Every device array one batch needs, each released in stream order on every path out of the launch. */
struct device_batch_t {
    device_array<sft::molecule_descriptor_t> descriptors;
    device_array<std::uint32_t> atom_offsets;
    device_array<std::uint8_t> atom_types;
    device_array<float> partial_charges;
    device_array<std::uint8_t> status_codes;
    device_array<sft::mmff94_atom_vdw_params_t> vdw_params;
    device_array<float> output_charges;
    device_array<sft::mmff94_bond_t> bond_terms;
    device_array<sft::mmff94_angle_t> angle_terms;
    device_array<sft::mmff94_torsion_t> torsion_terms;
    device_array<sft::mmff94_oop_t> out_of_plane_terms;
    device_array<sft::mmff94_stretchbend_t> stretch_bend_terms;
    device_array<sfmmff::mmff94_per_mol_counts_t> counts_per_molecule;
    device_array<sft::mmff94_atom_interaction_t> interactions;
    device_array<sft::mmff94_atom_offsets_t> atom_offsets_lookup;

    /**
     *  @brief Reserve every array for @p num_molecules molecules over @p total_atom_count atoms.
     *
     *  Every term array is sized at the per-molecule maximum stride, so the batch costs about 50 kB a
     *  molecule whatever the molecules are.
     */
    [[nodiscard]] bool reserve(std::uint32_t num_molecules, std::uint32_t total_atom_count) {
        bool reserved = true;
        auto reserve_ = [&](auto &array, std::size_t count) {
            reserved &= array.try_reserve(count) != types::reserve_outcome_t::failed_k;
        };
        reserve_(descriptors, num_molecules);
        reserve_(atom_offsets, num_molecules);
        reserve_(atom_types, total_atom_count);
        reserve_(partial_charges, total_atom_count);
        reserve_(status_codes, num_molecules);
        reserve_(vdw_params, total_atom_count);
        reserve_(output_charges, total_atom_count);
        reserve_(bond_terms, static_cast<std::size_t>(num_molecules) * sft::max_mmff_bonds_per_mol_k);
        reserve_(angle_terms, static_cast<std::size_t>(num_molecules) * sft::max_mmff_angles_per_mol_k);
        reserve_(torsion_terms, static_cast<std::size_t>(num_molecules) * sft::max_mmff_torsions_per_mol_k);
        reserve_(out_of_plane_terms, static_cast<std::size_t>(num_molecules) * sft::max_mmff_oops_per_mol_k);
        reserve_(stretch_bend_terms, static_cast<std::size_t>(num_molecules) * sft::max_mmff_stretchbends_per_mol_k);
        reserve_(counts_per_molecule, num_molecules);
        reserve_(interactions, static_cast<std::size_t>(num_molecules) * sft::max_mmff_interactions_per_mol_k);
        reserve_(atom_offsets_lookup, total_atom_count);
        return reserved;
    }
};

/** The four events one timed run records between, destroyed together whichever way the launch leaves. */
struct trio_events_t {
    cudaEvent_t start {};
    cudaEvent_t after_type_and_charge {};
    cudaEvent_t after_extract_terms {};
    cudaEvent_t after_build_csr {};

    trio_events_t() {
        cudaEventCreate(&start);
        cudaEventCreate(&after_type_and_charge);
        cudaEventCreate(&after_extract_terms);
        cudaEventCreate(&after_build_csr);
    }
    ~trio_events_t() {
        cudaEventDestroy(start);
        cudaEventDestroy(after_type_and_charge);
        cudaEventDestroy(after_extract_terms);
        cudaEventDestroy(after_build_csr);
    }

    trio_events_t(trio_events_t const &) = delete;
    trio_events_t &operator=(trio_events_t const &) = delete;

    /** Read the three intervals and the span across them into @p timings. */
    void read_into(kernel_trio_timings_t &timings) const {
        cudaEventElapsedTime(&timings.type_and_charge_ms, start, after_type_and_charge);
        cudaEventElapsedTime(&timings.extract_terms_ms, after_type_and_charge, after_extract_terms);
        cudaEventElapsedTime(&timings.build_csr_ms, after_extract_terms, after_build_csr);
        cudaEventElapsedTime(&timings.total_ms, start, after_build_csr);
    }
};

/** Report what the device has left, so an allocation refusal names the headroom it was refused against. */
static void print_gpu_memory_usage_(char const *label) {
    std::size_t free_memory = 0, total_memory = 0;
    cudaMemGetInfo(&free_memory, &total_memory);
    fmt::println(stderr, "{}: GPU free={:.1f} GB / total={:.1f} GB", label, static_cast<double>(free_memory) / 1.0e9,
                 static_cast<double>(total_memory) / 1.0e9);
}

trio_launch_t launch_mmff94_kernel_trio_benchmark(sfparam::mmff94_tables_host_t const &tables_host,
                                                  custd::span<sft::molecule_descriptor_t const> descriptors,
                                                  custd::span<std::uint32_t const> atom_offsets,
                                                  std::uint32_t total_atom_count, kernel_trio_timings_t &timings) {
    sfparam::mmff94_tables_device_t tables_device;
    sfparam::upload_mmff94_tables_to_device(tables_host, tables_device, nullptr);
    if (tables_device.view.vdw.empty()) return trio_launch_t::table_upload_failed_k;

    std::uint32_t const num_molecules = static_cast<std::uint32_t>(descriptors.size());
    print_gpu_memory_usage_("before alloc");
    device_batch_t batch;
    if (!batch.reserve(num_molecules, total_atom_count)) {
        print_gpu_memory_usage_("after refused alloc");
        return trio_launch_t::allocation_failed_k;
    }

    // The allocations above are stream-ordered against the stream the kernels and events below ride, so the
    // uploads see them without a synchronize of their own.
    if (!types::upload_into(batch.descriptors.view(), descriptors, nullptr)) return trio_launch_t::allocation_failed_k;
    if (!types::upload_into(batch.atom_offsets.view(), atom_offsets, nullptr))
        return trio_launch_t::allocation_failed_k;

    trio_events_t events;
    auto launch_type_and_charge_ = [&] {
        sfmmff::mmff94_type_and_charge_kernel<<<num_molecules, sfmmff::warp_size_k, 0, nullptr>>>(
            batch.descriptors.data(), batch.atom_offsets.data(), num_molecules, tables_device.view,
            batch.atom_types.data(), batch.partial_charges.data(), batch.status_codes.data());
    };

    // Warm the driver, so the one-shot setup of a first launch is not charged to the steady-state cost.
    launch_type_and_charge_();
    cudaDeviceSynchronize();

    cudaEventRecord(events.start);
    launch_type_and_charge_();
    cudaEventRecord(events.after_type_and_charge);
    sfmmff::mmff94_extract_terms_kernel<<<num_molecules, sfmmff::warp_size_k, 0, nullptr>>>(
        batch.descriptors.data(), batch.atom_types.data(), batch.partial_charges.data(), batch.atom_offsets.data(),
        num_molecules, tables_device.view, batch.vdw_params.data(), batch.output_charges.data(),
        batch.bond_terms.data(), batch.angle_terms.data(), batch.torsion_terms.data(), batch.out_of_plane_terms.data(),
        batch.stretch_bend_terms.data(), batch.counts_per_molecule.data(), batch.status_codes.data());
    cudaEventRecord(events.after_extract_terms);
    sfmmff::mmff94_build_csr_kernel<<<num_molecules, sfmmff::warp_size_k, 0, nullptr>>>(
        batch.descriptors.data(), batch.atom_offsets.data(), num_molecules, batch.counts_per_molecule.data(),
        batch.bond_terms.data(), batch.angle_terms.data(), batch.torsion_terms.data(), batch.out_of_plane_terms.data(),
        batch.stretch_bend_terms.data(), batch.interactions.data(), batch.atom_offsets_lookup.data(),
        batch.status_codes.data());
    cudaEventRecord(events.after_build_csr);
    cudaEventSynchronize(events.after_build_csr);

    if (cudaError_t const kernel_error = cudaGetLastError(); kernel_error != cudaSuccess) {
        fmt::println(stderr, "kernel error: {}", cudaGetErrorString(kernel_error));
        return trio_launch_t::kernel_failed_k;
    }
    events.read_into(timings);
    return trio_launch_t::timed_k;
}

} // namespace smallfield::bench
