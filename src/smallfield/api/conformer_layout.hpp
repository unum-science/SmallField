/**
 *  @file src/smallfield/api/conformer_layout.hpp
 *  @brief Addressing contract for conformer coordinate slots, and the transposition against it.
 *
 *  Deliberately free of the rest of the public API: the three sites that must agree on this layout — the GPU
 *  producer, the RDKit `fallback-role` producer, and the Parquet writer — plus its regression test, should be
 *  able to include the contract without dragging in `context_t` or the CUDA runtime. `types/views.hpp` is the
 *  one dependency, and it names shapes rather than owning memory, so it costs none of that. The transposition
 *  below is templated on the slot's element type for the same reason: the pool holds `__half`, and naming that
 *  type here would put `<cuda_fp16.h>` in front of every one of those sites.
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <type_traits> // `std::is_integral_v`, `std::is_same_v`

#include <cuda/std/array> // `custd::array`

#include "smallfield/types/molecule_descriptor.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::api {

/** Maximum atom count the kernels support per molecule.
 *  One definition, re-exported here because the coordinate layout is stated in these terms; the
 *  descriptor's array bounds and this stride have to be the same number or the two disagree about
 *  what a molecule can hold. */
using types::max_atoms_per_molecule_k;

/**
 *  @brief Addressing for one conformer slot inside a coordinates buffer.
 *
 *  A slot is structure-of-arrays with a fixed @p max_atoms stride per component:
 *  `[x_0 .. x_{max_atoms-1}][y_0 .. y_{max_atoms-1}][z_0 .. z_{max_atoms-1}]`.
 *  Atoms past a molecule's own count are padding and carry no meaning.
 *
 *  Two independent producers fill these slots — the GPU pipeline via `finalize_gpu_pipeline`, and the RDKit
 *  `fallback-role` for molecules the GPU could not complete — while the Parquet writer consumes them with no way
 *  to tell which producer wrote a given slot. Every one of those sites must therefore agree on the stride
 *  exactly, so they address slots through this type rather than open-coding the arithmetic. A single contiguous
 *  `memcpy` of `3 * num_atoms` halves from a slot base is *not* a valid read: it yields this molecule's x values
 *  followed by inter-component padding, silently dropping y and z for every molecule with
 *  `3 * num_atoms <= max_atoms`.
 */
struct conformer_slot_layout_t {
    std::uint32_t max_atoms = max_atoms_per_molecule_k;
    std::uint32_t num_conformers = 1;

    /** Halves spanned by one conformer slot, padding included. */
    [[nodiscard]] constexpr std::size_t slot_stride() const noexcept {
        return static_cast<std::size_t>(max_atoms) * types::conformer_dimensions_k;
    }

    /** Position of molecule @p molecule 's conformer @p conformer among all slots, which is what the
     *  per-conformer status, energy and descriptor arrays are indexed by. */
    [[nodiscard]] constexpr std::size_t flat_slot(std::uint64_t molecule, std::uint32_t conformer) const noexcept {
        return static_cast<std::size_t>(molecule) * num_conformers + conformer;
    }

    /** Offset of molecule @p molecule 's conformer @p conformer from the buffer base. */
    [[nodiscard]] constexpr std::size_t slot_offset(std::uint64_t molecule, std::uint32_t conformer) const noexcept {
        return flat_slot(molecule, conformer) * slot_stride();
    }

    /** Offset of component @p axis (0 = x, 1 = y, 2 = z) of atom @p atom within a slot. */
    [[nodiscard]] constexpr std::size_t component_offset(std::uint32_t axis, std::uint32_t atom) const noexcept {
        return static_cast<std::size_t>(axis) * max_atoms + atom;
    }

    /**
     *  @brief One slot's first @p num_atoms atoms as a `(axis, atom)` view, padded at this layout's stride.
     *
     *  The padding is what keeps this off `coordinates_soa`: that view's `custd::layout_left` packs its axis
     *  planes at the atom count, while a slot pads every plane to `max_atoms` whatever the molecule holds.
     *
     *  @param[in] slot Base of the slot inside the coordinate pool, `pool.data() + slot_offset(...)`.
     *  @param[in] num_atoms Atoms the molecule actually has; the slot's padding atoms fall outside the view.
     */
    template <typename value_type_>
    [[nodiscard]] constexpr types::strided_matrix<value_type_> slot_view(value_type_ *slot,
                                                                         std::uint32_t num_atoms) const noexcept {
        return {
            slot,
            {custd::dextents<std::uint32_t, 2> {static_cast<std::uint32_t>(types::conformer_dimensions_k), num_atoms},
             custd::array<std::size_t, 2> {max_atoms, 1u}}};
    }
};

#pragma region Slot Transposition

/**
 *  @brief Whether a slot element and an interleaved element name the same coordinate under plain assignment.
 *
 *  Identical types copy the stored pattern verbatim, and two arithmetic-but-not-integer types convert through
 *  their own operators — which is the whole of the `__half` against `float` case, without this header having to
 *  know that `__half` is what the pool holds. An integer on one side only is refused: a raw `std::uint16_t`
 *  half pattern assigned to a `float` converts numerically, turning a 1 Å coordinate into 15360.
 */
template <typename slot_type_, typename interleaved_type_>
constexpr bool coordinates_assign_losslessly_k = std::is_same_v<slot_type_, interleaved_type_> ||
                                                 (!std::is_integral_v<slot_type_> &&
                                                  !std::is_integral_v<interleaved_type_>);

/**
 *  @brief Read one conformer slot's structure-of-arrays components into a tight interleaved `(atom, 3)` block.
 *
 *  The transposition every reader of the pool performs, stated once beside the layout it addresses: the Parquet
 *  writer, the descriptor tool, and the tests and benchmarks that hand a conformer to RDKit all go through
 *  `slot_view`, so no two of them can disagree about where a component sits.
 *
 *  @param[in] slot Base of the slot inside the coordinate pool, `pool.data() + layout.slot_offset(...)`.
 *  @param[in] layout Addressing contract the producers and the consumers all agree on.
 *  @param[in] num_atoms Atoms the molecule actually has; the slot's padding atoms are not read.
 *  @param[out] interleaved Destination for `3 * num_atoms` components, atom-major.
 */
template <typename slot_type_, typename interleaved_type_>
inline void gather_slot_into_interleaved(slot_type_ const *slot, conformer_slot_layout_t const &layout,
                                         std::uint32_t num_atoms, interleaved_type_ *interleaved) noexcept {
    static_assert(coordinates_assign_losslessly_k<slot_type_, interleaved_type_>,
                  "an interleaved element is either the slot's own storage or a type it converts to numerically");
    types::strided_matrix<slot_type_ const> const source = layout.slot_view(slot, num_atoms);
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
        for (std::uint32_t axis = 0; axis < types::conformer_dimensions_k; ++axis)
            interleaved[atom * types::conformer_dimensions_k + axis] = source(axis, atom);
}

/**
 *  @brief Write a tight interleaved `(atom, 3)` block back into one conformer slot.
 *
 *  The inverse of `gather_slot_into_interleaved`, for the producers that build a geometry atom-major and the
 *  tools that correct one in place before it is written out.
 *
 *  @param[in] interleaved Source of `3 * num_atoms` components, atom-major.
 *  @param[in] layout Addressing contract the producers and the consumers all agree on.
 *  @param[in] num_atoms Atoms the molecule actually has; the slot's padding atoms are left alone.
 *  @param[out] slot Base of the slot inside the coordinate pool, `pool.data() + layout.slot_offset(...)`.
 */
template <typename interleaved_type_, typename slot_type_>
inline void scatter_interleaved_into_slot(interleaved_type_ const *interleaved, conformer_slot_layout_t const &layout,
                                          std::uint32_t num_atoms, slot_type_ *slot) noexcept {
    static_assert(coordinates_assign_losslessly_k<slot_type_, interleaved_type_>,
                  "an interleaved element is either the slot's own storage or a type it converts to numerically");
    types::strided_matrix<slot_type_> const destination = layout.slot_view(slot, num_atoms);
    for (std::uint32_t atom = 0; atom < num_atoms; ++atom)
        for (std::uint32_t axis = 0; axis < types::conformer_dimensions_k; ++axis)
            destination(axis, atom) = interleaved[atom * types::conformer_dimensions_k + axis];
}

#pragma endregion Slot Transposition

} // namespace smallfield::api
