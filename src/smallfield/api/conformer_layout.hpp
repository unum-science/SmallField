/**
 *  @file conformer_layout.hpp
 *  @brief Addressing contract for conformer coordinate slots.
 *
 *  Deliberately free of CUDA and of the rest of the public API: the three sites that
 *  must agree on this layout -- the GPU producer, the RDKit `fallback-role` producer,
 *  and the Parquet writer -- plus its regression test, should be able to include the
 *  contract without dragging in `context_t` or the CUDA runtime headers.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "smallfield/types/molecule_descriptor.hpp"

namespace smallfield::api {

/** @brief Maximum atom count the kernels support per molecule.
 *  One definition, re-exported here because the coordinate layout is stated in these terms; the
 *  descriptor's array bounds and this stride have to be the same number or the two disagree about
 *  what a molecule can hold. */
using types::max_atoms_per_molecule_k;

/**
 *  @brief Addressing for one conformer slot inside a coordinates buffer.
 *
 *  A slot is structure-of-arrays with a fixed @p max_atoms stride per component:
 *  `[x_0 … x_{max_atoms-1}][y_0 … y_{max_atoms-1}][z_0 … z_{max_atoms-1}]`.
 *  Atoms past a molecule's own count are padding and carry no meaning.
 *
 *  Two independent producers fill these slots -- the GPU pipeline via
 *  `finalize_gpu_pipeline`, and the RDKit `fallback-role` for molecules the GPU
 *  could not complete -- while the Parquet writer consumes them with no way to tell
 *  which producer wrote a given slot.  Every one of those sites must therefore agree
 *  on the stride exactly, so they address slots through this type rather than
 *  open-coding the arithmetic.  A single contiguous `memcpy` of `3 * num_atoms`
 *  halves from a slot base is *not* a valid read: it yields this molecule's x values
 *  followed by inter-component padding, silently dropping y and z for every molecule
 *  with `3 * num_atoms <= max_atoms`.
 */
struct conformer_slot_layout_t {
    std::uint32_t max_atoms = max_atoms_per_molecule_k;
    std::uint32_t num_conformers = 1;

    /** @brief Halves spanned by one conformer slot, padding included. */
    [[nodiscard]] constexpr std::size_t slot_stride() const noexcept {
        return static_cast<std::size_t>(max_atoms) * 3u;
    }

    /** @brief Offset of molecule @p molecule 's conformer @p conformer from the buffer base. */
    [[nodiscard]] constexpr std::size_t slot_offset(std::uint64_t molecule, std::uint32_t conformer) const noexcept {
        return (static_cast<std::size_t>(molecule) * num_conformers + conformer) * slot_stride();
    }

    /** @brief Offset of component @p axis (0 = x, 1 = y, 2 = z) of atom @p atom within a slot. */
    [[nodiscard]] constexpr std::size_t component_offset(std::uint32_t axis, std::uint32_t atom) const noexcept {
        return static_cast<std::size_t>(axis) * max_atoms + atom;
    }
};


} // namespace smallfield::api
