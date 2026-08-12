/**
 *  @file   src/smallfield/types/views.hpp
 *  @brief  The libcu++ namespace alias, and non-owning views over caller-owned molecule buffers.
 *
 *  Every matrix in this project is a flat buffer plus an extent the caller carries separately, which is
 *  three parameters where the data is one thing and lets an extent disagree with its storage undetected.
 *  These aliases put the extent inside the view.
 *
 *  The standard library's own types cannot serve here. `std::mdspan` needs C++23, and `nvcc` 12.8 and 12.9
 *  both reject `-std=c++23` outright, so it cannot enter any translation unit `nvcc` touches; and
 *  `std::span`'s members are not `__host__ __device__`, so a device helper cannot take one. libcu++ ships
 *  both interfaces at C++20 for host and device alike and arrives with `CUDA::cudart`, which is already a
 *  `PUBLIC` dependency of `smallfield_objects`. `custd` names the namespace once — the namespace rather
 *  than one type in it, because it carries the span as well as the views — so the swap is one `#if` here
 *  when `nvcc` 13 and libstdc++ 16 land together.
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

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint8_t`

#include <concepts>    // `std::convertible_to`
#include <type_traits> // `std::conditional_t`, `std::is_const_v`
#include <vector>      // `std::vector`

#include <cuda/std/mdspan>
#include <cuda/std/span>

#include "smallfield/utilities.hpp"

/*
 *  At global scope on purpose. Every subsystem spells `custd::span`, and so does the pybind module, which
 *  sits outside `smallfield` entirely — an alias inside the project namespace would force that translation
 *  unit and every other global-scope consumer to write the prefix back out. One short, distinctive name in
 *  the global namespace is the price of the type being spellable everywhere it is used.
 */
namespace custd = cuda::std;

namespace smallfield::types {

#pragma region Aliases

/**
 *  @brief Row-major matrix whose rows are packed at the column extent.
 *
 *  The host layout: `build_bounds_matrix` and `fill_bond_distances` both write `matrix(row, column)` with
 *  no padding between rows.
 */
template <typename value_type_>
using packed_matrix = custd::mdspan<value_type_, custd::dextents<std::uint32_t, 2>>;

/**
 *  @brief Row-major matrix whose rows are padded to a stride the caller states.
 *
 *  The one thing `packed_matrix` cannot say is a row stride above the column extent, and two layouts here
 *  need exactly that: the device bounds matrix, whose rows `compute_bounds_row_stride` pads to an odd
 *  multiple of four so all 32 lanes of a warp land on distinct shared-memory banks, and a conformer slot,
 *  whose component planes are padded to `conformer_slot_layout_t::max_atoms` rather than to the molecule's
 *  own atom count. A strided view is 24 bytes against a packed view's 16, per kernel argument.
 */
template <typename value_type_>
using strided_matrix = custd::mdspan<value_type_, custd::dextents<std::uint32_t, 2>, custd::layout_stride>;

/** All-pairs bond-hop counts, one row per source atom, `unreachable_hops_k` where no path exists. */
using hop_matrix_t = packed_matrix<std::uint8_t>;

/** Rows a conformer carries: x, y and z. */
inline constexpr std::size_t conformer_dimensions_k = 3;

/** Rows a metric embedding carries: the three spatial ones plus the w the crush penalty flattens. */
inline constexpr std::size_t embedding_dimensions_k = 4;

/**
 *  @brief What a coordinate view holds.
 *
 *  Positions, velocities and a gradient are identical in shape and layout and ride adjacent through the
 *  FIRE driver, so nothing but this separates them and any two of them are mutually assignable without it.
 */
enum class coordinate_role_t : std::uint8_t { position_k, velocity_k, gradient_k, embedding_k };

/**
 *  @brief Empty accessor carrying the role, so two roles are distinct types.
 *
 *  `custd::mdspan`'s converting constructor requires the accessors be constructible from one another, and
 *  the accessor is already one of its four template parameters, so the role costs no bytes and needs no
 *  wrapper class. It also closes what the layout policy alone cannot: an atom-major view is `layout_right`
 *  like `packed_matrix`, and a static extent always converts to a dynamic one.
 *
 *  It restates `custd::default_accessor`'s three members rather than deriving from it, and the difference
 *  is load-bearing. A derived empty struct is still an aggregate, so `role_accessor<gradient_k, float>` is
 *  aggregate-initializable from any sibling's base subobject, `is_constructible` reports true, and every
 *  role converts to every other — which is the whole guarantee, lost. Standing alone, only the one
 *  conversion below exists.
 */
template <coordinate_role_t role_, typename value_type_>
struct role_accessor {
    using offset_policy = role_accessor;
    using element_type = value_type_;
    using reference = value_type_ &;
    using data_handle_type = value_type_ *;

    constexpr role_accessor() noexcept = default;

    /** Adding `const` to the element type keeps the role, so a mutable view still passes as a reader. */
    template <typename other_type_>
        requires std::convertible_to<other_type_ (*)[], value_type_ (*)[]>
    constexpr role_accessor(role_accessor<role_, other_type_> const &) noexcept {}

    constexpr reference access(data_handle_type handle, std::size_t index) const noexcept { return handle[index]; }
    constexpr data_handle_type offset(data_handle_type handle, std::size_t index) const noexcept {
        return handle + index;
    }
};

/**
 *  @brief Coordinates over one molecule, always addressed `(atom, axis)`.
 *
 *  The layout policy says where the elements sit, so a call site reads the same whichever storage it holds
 *  and the compiler performs the stride arithmetic that would otherwise be written by hand. Distinguishing
 *  the two layouts by which extent is static does not work: a static extent converts implicitly to a
 *  dynamic one, so both would pass where a `packed_matrix` was expected. `custd::layout_left` against
 *  `custd::layout_right` blocks that in both directions, because the cross-layout mapping constructor is
 *  constrained to rank one or lower and these are rank two.
 *
 *  The dimension count is a template parameter because it is a property of the algorithm rather than of the
 *  molecule: distance geometry works at four rows and everything downstream at three, and the two must not
 *  be interchangeable at a call site.
 */
template <coordinate_role_t role_, typename value_type_, std::size_t dimensions_, typename layout_type_>
using coordinates = custd::mdspan<value_type_, custd::extents<std::uint32_t, custd::dynamic_extent, dimensions_>,
                                  layout_type_, role_accessor<role_, value_type_>>;

/** Contiguous axis planes, so `stride(atom)` is one and `stride(axis)` is the atom count. */
template <coordinate_role_t role_, typename value_type_, std::size_t dimensions_>
using coordinates_soa = coordinates<role_, value_type_, dimensions_, custd::layout_left>;

/** One xyz triple per atom, the layout RDKit reads and writes. */
template <coordinate_role_t role_, typename value_type_>
using coordinates_aos = coordinates<role_, value_type_, conformer_dimensions_k, custd::layout_right>;

/** A conformer: three spatial rows over one molecule's atoms. */
using conformer_soa_t = coordinates_soa<coordinate_role_t::position_k, float, conformer_dimensions_k>;

/** A metric embedding, which distance geometry works in before projecting down to a conformer. */
using embedding_soa_t = coordinates_soa<coordinate_role_t::embedding_k, float, embedding_dimensions_k>;

/** A conformer in RDKit's order. */
using conformer_aos_t = coordinates_aos<coordinate_role_t::position_k, float>;

#pragma endregion Aliases

#pragma region Distance Bounds

/*
 *  A bounds matrix folds both ends of every atom pair into one square: the upper triangle holds the pair's
 *  upper bound and the lower triangle its lower bound, so reaching a bound means ordering the two atom
 *  indices and choosing which of them lands in the row. Written out at a call site that is a two-line dance
 *  whose correct and inverted forms differ in nothing but that choice, and both read plausible. Naming it
 *  takes the pair unordered instead, so a caller never states an order at all. Equal indices land on the
 *  diagonal, which the convention holds at zero in both roles.
 *
 *  They are `constexpr` rather than `__host__ __device__` because this header is compiled as plain C++ by
 *  `test/parameters/test_mmff94_tables.cpp`, where a CUDA attribute is not a keyword the host compiler
 *  knows. Device code reaches them through `--expt-relaxed-constexpr`, the same route `utilities.hpp` takes.
 *
 *  They read one bound each rather than returning both, and that is measured rather than assumed. Both
 *  smoothing loops branch on which inequality applies before it is known whether the lower bound is wanted,
 *  so a pair-returning form holds a value one branch never reads live across that branch, and `cuobjdump`
 *  charges `triangle_smooth_kernel` the extra registers for it. The packed view itself is free:
 *  `triangle_smooth_kernel` compiles to the same SASS through these as through raw index arithmetic.
 *
 *  Each name carries a second form over a mutable matrix, returning the cell rather than its value, so the
 *  sweeps that build a square name the triangle they land in exactly as the sweeps that read it do. A
 *  mutable matrix is an exact match for that form and a user-defined conversion away from the reading one,
 *  so it is chosen without ambiguity and no reader of a `packed_matrix<float const>` is affected.
 */

/**
 *  @brief Upper distance bound of an atom pair, from the packed square's upper triangle.
 *
 *  @param[in] matrix Square bounds matrix over one molecule's atoms.
 *  @param[in] first_atom One end of the pair; the two ends are interchangeable.
 *  @param[in] second_atom The other end.
 *  @return Furthest apart the pair may go, in angstrom.
 */
constexpr float upper_distance_bound_at(packed_matrix<float const> matrix, std::uint32_t first_atom,
                                        std::uint32_t second_atom) noexcept {
    std::uint32_t const row = first_atom < second_atom ? first_atom : second_atom;
    std::uint32_t const column = first_atom < second_atom ? second_atom : first_atom;
    return matrix(row, column);
}

/**
 *  @brief The upper triangle's cell for an atom pair, for a caller that writes it.
 *
 *  @param[inout] matrix Square bounds matrix over one molecule's atoms.
 *  @param[in] first_atom One end of the pair; the two ends are interchangeable.
 *  @param[in] second_atom The other end.
 *  @return The cell holding how far apart the pair may go, in angstrom.
 */
constexpr float &upper_distance_bound_at(packed_matrix<float> matrix, std::uint32_t first_atom,
                                         std::uint32_t second_atom) noexcept {
    std::uint32_t const row = first_atom < second_atom ? first_atom : second_atom;
    std::uint32_t const column = first_atom < second_atom ? second_atom : first_atom;
    return matrix(row, column);
}

/**
 *  @brief Lower distance bound of an atom pair, from the packed square's lower triangle.
 *
 *  @param[in] matrix Square bounds matrix over one molecule's atoms.
 *  @param[in] first_atom One end of the pair; the two ends are interchangeable.
 *  @param[in] second_atom The other end.
 *  @return Closest the pair may come, in angstrom.
 */
constexpr float lower_distance_bound_at(packed_matrix<float const> matrix, std::uint32_t first_atom,
                                        std::uint32_t second_atom) noexcept {
    std::uint32_t const row = first_atom < second_atom ? second_atom : first_atom;
    std::uint32_t const column = first_atom < second_atom ? first_atom : second_atom;
    return matrix(row, column);
}

/**
 *  @brief The lower triangle's cell for an atom pair, for a caller that writes it.
 *
 *  @param[inout] matrix Square bounds matrix over one molecule's atoms.
 *  @param[in] first_atom One end of the pair; the two ends are interchangeable.
 *  @param[in] second_atom The other end.
 *  @return The cell holding how close the pair may come, in angstrom.
 */
constexpr float &lower_distance_bound_at(packed_matrix<float> matrix, std::uint32_t first_atom,
                                         std::uint32_t second_atom) noexcept {
    std::uint32_t const row = first_atom < second_atom ? second_atom : first_atom;
    std::uint32_t const column = first_atom < second_atom ? first_atom : second_atom;
    return matrix(row, column);
}

#pragma endregion Distance Bounds

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

/**
 *  @brief Grow @p storage to hold @p num_atoms axis-major coordinates, zero it, and view it.
 *
 *  @param[inout] storage Caller-owned buffer, reused across molecules so growth happens once.
 *  @param[in] num_atoms Atoms the view spans.
 *  @return View over @p storage.
 */
template <coordinate_role_t role_, std::size_t dimensions_, typename value_type_>
inline coordinates_soa<role_, value_type_, dimensions_> reset_coordinates_soa(std::vector<value_type_> &storage,
                                                                              std::uint32_t num_atoms) {
    storage.assign(dimensions_ * num_atoms, value_type_ {});
    return coordinates_soa<role_, value_type_, dimensions_> {storage.data(), num_atoms};
}

/**
 *  @brief Grow @p storage to hold @p num_atoms atom-major coordinates, zero it, and view it.
 *
 *  @param[inout] storage Caller-owned buffer, reused across molecules so growth happens once.
 *  @param[in] num_atoms Atoms the view spans.
 *  @return View over @p storage.
 */
template <coordinate_role_t role_, typename value_type_>
inline coordinates_aos<role_, value_type_> reset_coordinates_aos(std::vector<value_type_> &storage,
                                                                 std::uint32_t num_atoms) {
    storage.assign(static_cast<std::size_t>(num_atoms) * conformer_dimensions_k, value_type_ {});
    return coordinates_aos<role_, value_type_> {storage.data(), num_atoms};
}

#pragma endregion Builders

#pragma region Transposition

/**
 *  @brief Scatter atom-major triples into the three spatial columns of an axis-major view.
 *
 *  Columns past the third keep whatever they already held, so a caller that fills the fourth first and the
 *  spatial three second gets the same buffer as one that does it the other way round.
 *
 *  Both views are addressed `(atom, axis)` and the layout supplies the stride, so the body says what it
 *  does rather than which storage order it assumes.
 *
 *  @param[in] atom_major Source, one xyz triple per atom.
 *  @param[out] axis_major Destination, at least three columns over the same atoms.
 */
template <coordinate_role_t source_role_, typename source_type_, coordinate_role_t destination_role_,
          typename destination_type_, std::size_t dimensions_>
inline void transpose_into(coordinates_aos<source_role_, source_type_> atom_major,
                           coordinates_soa<destination_role_, destination_type_, dimensions_> axis_major) {
    static_assert(dimensions_ >= conformer_dimensions_k,
                  "an axis-major coordinate view carries at least the three spatial columns");
    sf_assert_(atom_major.extent(0) == axis_major.extent(0));
    for (std::uint32_t atom = 0; atom < atom_major.extent(0); ++atom)
        for (std::uint32_t axis = 0; axis < conformer_dimensions_k; ++axis)
            axis_major(atom, axis) = atom_major(atom, axis);
}

/**
 *  @brief Gather the three spatial columns of an axis-major view back into atom-major triples.
 *
 *  Columns past the third have no atom-major counterpart and are not read; dropping the fourth is the
 *  projection that turns a metric embedding into a conformer.
 *
 *  @param[in] axis_major Source, at least three columns over the atoms.
 *  @param[out] atom_major Destination, one xyz triple per atom.
 */
template <coordinate_role_t source_role_, typename source_type_, std::size_t dimensions_,
          coordinate_role_t destination_role_, typename destination_type_>
inline void transpose_into(coordinates_soa<source_role_, source_type_, dimensions_> axis_major,
                           coordinates_aos<destination_role_, destination_type_> atom_major) {
    static_assert(dimensions_ >= conformer_dimensions_k,
                  "an axis-major coordinate view carries at least the three spatial columns");
    sf_assert_(atom_major.extent(0) == axis_major.extent(0));
    for (std::uint32_t atom = 0; atom < atom_major.extent(0); ++atom)
        for (std::uint32_t axis = 0; axis < conformer_dimensions_k; ++axis)
            atom_major(atom, axis) = axis_major(atom, axis);
}

#pragma endregion Transposition

#pragma region Pool Carving

/** Alignment every carved subspan starts at, matching the CUDA convention the wave pool carves with. */
inline constexpr std::size_t pool_alignment_k = 256;

/**
 *  @brief Cursor handing out typed subspans of one byte pool, each at `pool_alignment_k`.
 *
 *  One sequence of carves answers both questions a sub-allocated pool asks. Over a cursor holding no
 *  pool every carve hands back an empty span while `bytes` still advances, so the walk ends holding the
 *  size the allocation needs; over a cursor holding that allocation the same sequence hands back the
 *  real subspans. There is no second sequence to keep in step, so a carve added to the walk sizes the
 *  pool and fills the view in one edit.
 *
 *  Constness travels from the pool to the spans, so a cursor over `std::byte const` cannot hand out a
 *  writable view of it.
 *
 *  Here rather than in `utilities.hpp` because a carve hands out a `custd::span`, and that header's
 *  admission rule asks for nothing outside the standard ones — it includes no header at all today.
 */
template <typename byte_type_>
struct pool_cursor {
    /** The allocation being carved, empty while the walk is only sizing one. */
    custd::span<byte_type_> pool {};
    /** Bytes the walk has placed so far, and the offset the next aligned carve starts from. */
    std::size_t bytes = 0;
};

/** An element viewed through bytes of the same constness, so `std::byte const` yields `element_ const`. */
template <typename byte_type_, typename element_>
using pool_element_t = std::conditional_t<std::is_const_v<byte_type_>, element_ const, element_>;

/**
 *  @brief Place @p count elements of @p element_ at @p cursor's next aligned offset and advance past them.
 *
 *  A free function rather than a member so a walk written against a dependent cursor type spells the
 *  element once, without a `template` disambiguator at every carve.
 */
template <typename element_, typename byte_type_>
[[nodiscard]] constexpr custd::span<pool_element_t<byte_type_, element_>> carve(pool_cursor<byte_type_> &cursor,
                                                                                std::size_t count) noexcept {
    cursor.bytes = align_up_(cursor.bytes, pool_alignment_k);
    std::size_t const byte_offset = cursor.bytes;
    cursor.bytes += count * sizeof(element_);
    if (cursor.pool.empty()) return {};
    // The sizing walk and the carving walk are one sequence, so a pool too small for the walk that
    // measured it cannot happen by editing one and not the other. It can still happen by allocating
    // against a different shape than the one carved here, and that reads as valid device memory.
    sf_assert_(cursor.bytes <= cursor.pool.size());
    return {reinterpret_cast<pool_element_t<byte_type_, element_> *>(cursor.pool.data() + byte_offset), count};
}

#pragma endregion Pool Carving

} // namespace smallfield::types
