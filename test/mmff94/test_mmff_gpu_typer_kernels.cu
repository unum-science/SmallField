/**
 *  @file test/mmff94/test_mmff_gpu_typer_kernels.cu
 *  @brief Validate the GPU MMFF typer kernel.
 *
 *  Runs it on a batch of descriptors and hands results back to the host .cpp for byte-level comparison vs the
 *  CPU native typer.
 */

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <string_view> // `std::string_view`

#include <cuda_runtime.h>

#include "smallfield/mmff94/mmff94_typer_device.cuh"
#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_device.hpp"
#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/storage.hpp"

#include "test/mmff94/test_mmff_gpu_typer_kernels.hpp"

namespace sfmmff = smallfield::mmff94;
namespace sfparam = smallfield::parameters;
namespace sft = smallfield::types;

namespace smallfield::tests {

/** Grow-only device storage over the legacy default stream, which is the one the typer launches on. */
template <typename value_type_>
using device_array = sft::buffer<value_type_, sft::device_allocator_t>;

/** Every device array one typed batch needs, released in stream order on every path out of the launch. */
struct typer_arena_t {
    device_array<sft::molecule_descriptor_t> descriptors;
    device_array<std::uint32_t> atom_offsets;
    device_array<std::uint8_t> atom_types;
    device_array<float> atom_charges;
    device_array<std::uint8_t> typing_status;

    /** Reserve every array for @p num_molecules molecules over @p total_atoms atoms. */
    [[nodiscard]] bool resize_for(std::uint32_t num_molecules, std::size_t total_atoms) noexcept {
        bool reserved = true;
        auto reserve_ = [&](auto &array, std::size_t count) {
            reserved &= array.try_reserve(count) != sft::reserve_outcome_t::failed_k;
        };
        reserve_(descriptors, num_molecules);
        reserve_(atom_offsets, num_molecules);
        reserve_(atom_types, total_atoms);
        reserve_(atom_charges, total_atoms);
        reserve_(typing_status, num_molecules);
        return reserved;
    }
};

std::string_view typer_probe_reason(typer_probe_t outcome) noexcept {
    switch (outcome) {
    case typer_probe_t::empty_batch_k: return "empty_batch";
    case typer_probe_t::allocation_failed_k: return "device_allocation_failed";
    case typer_probe_t::transfer_failed_k: return "device_transfer_failed";
    case typer_probe_t::launch_failed_k: return "kernel_launch_failed";
    default: return "completed";
    }
}

typer_probe_t launch_mmff94_gpu_typer_batch(sfparam::mmff94_tables_view_t tables_view,
                                            custd::span<sft::molecule_descriptor_t const> descriptors,
                                            custd::span<std::uint32_t const> atom_offsets,
                                            custd::span<std::uint8_t> atom_types_out,
                                            custd::span<float> atom_charges_out,
                                            custd::span<std::uint8_t> typing_status_out) {
    // The caller builds descriptors from SMILES and none of them has to parse, and a zero-block
    // launch is a configuration error rather than a no-op.
    std::uint32_t const num_molecules = static_cast<std::uint32_t>(descriptors.size());
    if (num_molecules == 0) return typer_probe_t::empty_batch_k;

    typer_arena_t arena;
    if (!arena.resize_for(num_molecules, atom_types_out.size())) return typer_probe_t::allocation_failed_k;
    if (!sft::upload_into(arena.descriptors.view(), descriptors, nullptr) ||
        !sft::upload_into(arena.atom_offsets.view(), atom_offsets, nullptr))
        return typer_probe_t::transfer_failed_k;

    sfmmff::mmff94_type_and_charge_kernel<<<num_molecules, sfmmff::warp_size_k, 0, nullptr>>>(
        arena.descriptors.data(), arena.atom_offsets.data(), num_molecules, tables_view, arena.atom_types.data(),
        arena.atom_charges.data(), arena.typing_status.data());
    // A launch the driver refuses to enqueue reports here and nowhere else. The kernel never runs, so the
    // status array still holds whatever the allocation found -- and zero, which it is full of after a fresh
    // reservation more often than not, is the verdict for a molecule the typer accepted.
    if (cudaGetLastError() != cudaSuccess) return typer_probe_t::launch_failed_k;

    custd::span<std::uint8_t const> const device_types = arena.atom_types.view();
    custd::span<float const> const device_charges = arena.atom_charges.view();
    custd::span<std::uint8_t const> const device_status = arena.typing_status.view();
    bool const copied = sft::download_from(atom_types_out, device_types, nullptr) &&
                        sft::download_from(atom_charges_out, device_charges, nullptr) &&
                        sft::download_from(typing_status_out, device_status, nullptr);
    cudaError_t const drained = cudaStreamSynchronize(nullptr);
    if (!copied) return typer_probe_t::transfer_failed_k;
    if (drained != cudaSuccess) return typer_probe_t::launch_failed_k;
    return typer_probe_t::completed_k;
}

void *create_mmff94_device_tables_view(sfparam::mmff94_tables_host_t const &host,
                                       sfparam::mmff94_tables_view_t &out_view) {
    auto *device_tables = new sfparam::mmff94_tables_device_t;
    sfparam::upload_mmff94_tables_to_device(host, *device_tables, nullptr);
    if (device_tables->view.vdw.empty()) {
        delete device_tables;
        return nullptr;
    }
    out_view = device_tables->view;
    return device_tables;
}

void destroy_mmff94_device_tables_view(void *handle) { delete static_cast<sfparam::mmff94_tables_device_t *>(handle); }

} // namespace smallfield::tests
