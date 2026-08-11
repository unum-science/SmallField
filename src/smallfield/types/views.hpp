/**
 *  @file   src/smallfield/types/views.hpp
 *  @brief  Non-owning multi-dimensional views over caller-owned molecule buffers.
 *
 *  Every matrix in this project is a flat buffer plus an extent the caller carries separately, which is
 *  three parameters where the data is one thing and lets an extent disagree with its storage undetected.
 *  These aliases put the extent inside the view.
 *
 *  `std::mdspan` is the type we want and cannot have yet: it needs C++23, and `nvcc` 12.8 and 12.9 both
 *  reject `-std=c++23` outright, so it cannot enter any translation unit `nvcc` touches. libcu++ ships the
 *  same interface, works at C++20 in host and device code alike, and arrives with `CUDA::cudart`, which is
 *  already a `PUBLIC` dependency of `smallfield_objects`. `mdspan_ns` names the namespace once so the swap
 *  is one `#if` when `nvcc` 13 and libstdc++ 16 land together.
 *
 *  @section indexing Index width
 *
 *  Extents are `std::uint32_t`, which is a claim about scope rather than a micro-optimization: these view a
 *  single molecule, capped at `max_atoms_per_molecule_k` atoms and `max_bonds_k` bonds. The width is not
 *  free — a packed view is 16 bytes at `std::uint32_t` against 24 at `std::size_t`, and a strided one 24
 *  against 40, per kernel argument.
 *
 *  @b Never view the conformer pool through these. That pool is indexed in `std::size_t` on purpose: a
 *  10 M molecule run at three conformers each passes `UINT32_MAX` elements around conformer 11.2 M, and a
 *  32-bit offset wraps into another molecule's coordinates. `mmff94_atom_offsets_t` documents the same
 *  reasoning one size down, where a `std::uint8_t` count wraps on an atom carrying over 255 torsions.
 *
 *  @section bounds Bounds checking
 *
 *  `cuda::std::mdspan` performs none, even with `CCCL_ENABLE_ASSERTIONS` set — a `(9, 9)` access on a 4 × 4
 *  view reads out of bounds and returns whatever was there. Raw index arithmetic still wants `sf_assert_`.
 */

#pragma once

#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <vector> // `std::vector`

#include <cuda/std/mdspan>

namespace smallfield::types {

namespace mdspan_ns = cuda::std;

#pragma region Aliases

/**
 *  @brief Row-major matrix whose rows are packed at the column extent.
 *
 *  The host layout: `build_bounds_matrix` and `fill_bond_distances` both write `matrix(row, column)` with
 *  no padding between rows.
 */
template <typename value_type_>
using packed_matrix = mdspan_ns::mdspan<value_type_, mdspan_ns::dextents<std::uint32_t, 2>>;

/** All-pairs bond-hop counts, one row per source atom, `unreachable_hops_k` where no path exists. */
using hop_matrix_t = packed_matrix<std::uint8_t>;

/* The device bounds matrix wants a third alias over `mdspan_ns::layout_stride`, because
   `compute_bounds_row_stride` pads rows to an odd multiple of four so all 32 lanes of a warp land on
   distinct shared-memory banks. It arrives with its first consumer rather than ahead of it: the two view
   types this header replaced sat unused long enough for one of them to acquire a docblock that misstated
   its own size. */

#pragma endregion Aliases

#pragma region Builders

/**
 *  @brief Grow @p storage to hold a square matrix of @p extent, fill it, and view it.
 *
 *  The view does not own, so sizing stays with the caller's buffer; doing it here rather than at each call
 *  site is what keeps an extent from disagreeing with the storage behind it.
 *
 *  @param[inout] storage Caller-owned buffer, reused across molecules so growth happens once.
 *  @param[in] extent Row and column count.
 *  @param[in] fill Value written to every element.
 *  @return View over @p storage.
 */
template <typename value_type_>
inline packed_matrix<value_type_> reset_square_matrix(std::vector<value_type_> &storage, std::uint32_t extent,
                                                      value_type_ fill) {
    storage.assign(static_cast<std::size_t>(extent) * extent, fill);
    return packed_matrix<value_type_> {storage.data(), extent, extent};
}

#pragma endregion Builders

} // namespace smallfield::types
