/**
 *  @file   src/smallfield/types/views.hpp
 *  @brief  The libcu++ namespace alias, and non-owning views over caller-owned molecule buffers.
 *
 *  Every matrix is a flat buffer plus an extent carried as one thing, so the two cannot disagree. The
 *  standard types cannot serve: `std::mdspan` needs C++23, which `nvcc` 12.8, 12.9 and 13.2 all reject, and
 *  `std::span`'s members are not `__host__ __device__`. libcu++ ships both at C++20 for host and device
 *  and arrives with `CUDA::cudart`, already a `PUBLIC` dependency of `smallfield_objects`; `custd` names
 *  it once, so a swap stays one `#if` — though CUDA 13 arrived without C++23, so this is the spelling to
 *  keep rather than one to wait out.
 *
 *  `std::uint32_t` extents are a claim about scope: these view one molecule, capped at
 *  `max_atoms_per_molecule_k` atoms and `max_bonds_k` bonds. The width is not free — a packed view is
 *  16 bytes at `std::uint32_t` against 24 at `std::size_t`, and a strided one 24 against 40, per kernel
 *  argument. @b Never view the conformer pool through them — it is indexed in `std::size_t` because 10 M
 *  molecules at three conformers each pass `UINT32_MAX` elements around conformer 11.2 M, where a 32-bit
 *  offset wraps into another molecule; `mmff94_atom_offsets_t` documents the same reasoning one size down,
 *  where a `std::uint8_t` count wraps on an atom carrying over 255 torsions.
 *
 *  `cuda::std::mdspan` bounds-checks nothing, even with `CCCL_ENABLE_ASSERTIONS` set — a `(9, 9)` access on
 *  a 4 × 4 view reads out of bounds and returns whatever was there — so raw index arithmetic still wants
 *  `sf_assert_`.
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
 *  At global scope on purpose: the pybind module spells `custd::span` from outside `smallfield`, and an
 *  alias inside the project namespace would make every global-scope consumer write the prefix back out.
 */
namespace custd = cuda::std;

namespace smallfield::types {

#pragma region Aliases

/** Row-major matrix packed at the column extent, no padding between rows; the host layout
 *  `build_bounds_matrix` and `fill_bond_distances` write as `matrix(row, column)`. */
template <typename value_type_>
using packed_matrix = custd::mdspan<value_type_, custd::dextents<std::uint32_t, 2>>;

/**
 *  @brief Row-major matrix whose rows are padded to a stride the caller states.
 *
 *  Two layouts need a row stride above the column extent, which `packed_matrix` cannot say: the device
 *  bounds matrix, padded by `compute_bounds_row_stride` to an odd multiple of four so a warp's 32 lanes
 *  land on distinct shared-memory banks, and a conformer slot, whose component planes pad to
 *  `conformer_slot_layout_t::max_atoms` rather than the molecule's own atom count. A strided view is
 *  24 bytes against a packed view's 16, per kernel argument.
 */
template <typename value_type_>
using strided_matrix = custd::mdspan<value_type_, custd::dextents<std::uint32_t, 2>, custd::layout_stride>;

/** All-pairs bond-hop counts, one row per source atom, `unreachable_hops_k` where no path exists. */
using hop_matrix_t = packed_matrix<std::uint8_t>;

/** Rows a conformer carries: x, y and z. */
inline constexpr std::size_t conformer_dimensions_k = 3;

/** Rows a metric embedding carries: the three spatial ones plus the w the crush penalty flattens. */
inline constexpr std::size_t embedding_dimensions_k = 4;

/** What a coordinate view holds. Positions, velocities and gradients share one shape and ride adjacent
 *  through the FIRE driver, so without this tag any two of them would be mutually assignable. */
enum class coordinate_role_t : std::uint8_t { position_k, velocity_k, gradient_k, embedding_k };

/** Empty accessor carrying the role, so two roles are distinct types. `custd::mdspan`'s converting
 *  constructor requires accessors constructible from one another, and the accessor is one of its four
 *  template parameters, so the role costs no bytes. It closes what the layout policy cannot: an atom-major
 *  view is `layout_right` like `packed_matrix`, and a static extent always converts to a dynamic one.
 *  @b Do not derive this from `custd::default_accessor` — a derived empty struct is still an aggregate, so
 *  every role becomes aggregate-initializable from a sibling's base subobject and converts to every other,
 *  losing the whole guarantee. Standing alone, only the one conversion below exists. */
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

/** Coordinates over one molecule, always addressed `(atom, axis)`. The layout policy says where the
 *  elements sit, so a call site reads the same whichever storage it holds, and `custd::layout_left` against
 *  `custd::layout_right` keeps the two orders from converting into one another: the cross-layout mapping
 *  constructor is constrained to rank one or lower and these are rank two, whereas a static extent would
 *  convert implicitly to a dynamic one. The dimension count is a template parameter because it belongs to
 *  the algorithm, not the molecule — distance geometry works at four rows and everything downstream at
 *  three, and the two must not be interchangeable at a call site. */
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
 *  A bounds matrix folds both ends of every atom pair into one square: upper triangle for the pair's upper
 *  bound, lower triangle for its lower, and equal indices on the diagonal, held at zero in both roles.
 *  These take the pair unordered, so a caller never states which index lands in the row — a choice whose
 *  correct and inverted forms differ in nothing else and both read plausible. `constexpr` rather than
 *  `__host__ __device__` because `test/parameters/mmff94_tables.cpp` compiles this header as plain
 *  C++; device code reaches them through `--expt-relaxed-constexpr`. Each reads one bound rather than
 *  returning both: the smoothing loops branch on which inequality applies before it is known whether the
 *  lower bound is wanted, and a pair-returning form costs `triangle_smooth_kernel` registers for a value
 *  one branch never reads.
 */

/** Upper distance bound of an atom pair, from the packed square's upper triangle, in angstrom. */
constexpr float upper_distance_bound_at(packed_matrix<float const> matrix, std::uint32_t first_atom,
                                        std::uint32_t second_atom) noexcept {
    std::uint32_t const row = first_atom < second_atom ? first_atom : second_atom;
    std::uint32_t const column = first_atom < second_atom ? second_atom : first_atom;
    return matrix(row, column);
}

/** The upper triangle's cell for an atom pair, in angstrom, for a caller that writes it. @p matrix is the
 *  square bounds matrix over one molecule's atoms. */
constexpr float &upper_distance_bound_at(packed_matrix<float> matrix, std::uint32_t first_atom,
                                         std::uint32_t second_atom) noexcept {
    std::uint32_t const row = first_atom < second_atom ? first_atom : second_atom;
    std::uint32_t const column = first_atom < second_atom ? second_atom : first_atom;
    return matrix(row, column);
}

/** Lower distance bound of an atom pair, from the packed square's lower triangle, in angstrom. */
constexpr float lower_distance_bound_at(packed_matrix<float const> matrix, std::uint32_t first_atom,
                                        std::uint32_t second_atom) noexcept {
    std::uint32_t const row = first_atom < second_atom ? second_atom : first_atom;
    std::uint32_t const column = first_atom < second_atom ? first_atom : second_atom;
    return matrix(row, column);
}

/** The lower triangle's cell for an atom pair, in angstrom, for a caller that writes it. @p matrix is the
 *  square bounds matrix over one molecule's atoms. */
constexpr float &lower_distance_bound_at(packed_matrix<float> matrix, std::uint32_t first_atom,
                                         std::uint32_t second_atom) noexcept {
    std::uint32_t const row = first_atom < second_atom ? second_atom : first_atom;
    std::uint32_t const column = first_atom < second_atom ? first_atom : second_atom;
    return matrix(row, column);
}

#pragma endregion Distance Bounds

#pragma region Builders

/**
 *  @brief Grow @p storage to a square matrix of @p extent and write @p fill to every element, grown or not.
 *
 *  Sizing here rather than at each call site keeps an extent from disagreeing with the storage behind it.
 *  @p storage is caller-owned and reused across molecules, so growth happens once.
 */
template <typename value_type_>
inline packed_matrix<value_type_> reset_square_matrix(std::vector<value_type_> &storage, std::uint32_t extent,
                                                      value_type_ fill) {
    storage.assign(static_cast<std::size_t>(extent) * extent, fill);
    return packed_matrix<value_type_> {storage.data(), extent, extent};
}

/** Grow @p storage to @p num_atoms axis-major coordinates, zero it, and view it. @p storage is
 *  caller-owned and reused across molecules, so growth happens once. */
template <coordinate_role_t role_, std::size_t dimensions_, typename value_type_>
inline coordinates_soa<role_, value_type_, dimensions_> reset_coordinates_soa(std::vector<value_type_> &storage,
                                                                              std::uint32_t num_atoms) {
    storage.assign(dimensions_ * num_atoms, value_type_ {});
    return coordinates_soa<role_, value_type_, dimensions_> {storage.data(), num_atoms};
}

/** Grow @p storage to @p num_atoms atom-major coordinates, zero it, and view it. @p storage is
 *  caller-owned and reused across molecules, so growth happens once. */
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
 *  Columns past the third keep what they held, so filling the fourth before or after the spatial three
 *  yields the same buffer. Both views are addressed `(atom, axis)` and the layout supplies the stride.
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
 *  Columns past the third have no counterpart and are not read; dropping the fourth is the projection that
 *  turns a metric embedding into a conformer.
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
 *  One sequence of carves answers both questions a sub-allocated pool asks: over a cursor holding no pool
 *  every carve hands back an empty span while `bytes` still advances, so the walk ends holding the size the
 *  allocation needs, and over a cursor holding that allocation the same sequence hands back the real
 *  subspans. A carve added to the walk therefore sizes the pool and fills the view in one edit. Constness
 *  travels from the pool to the spans, so a cursor over `std::byte const` cannot hand out a writable view.
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

/** Place @p count elements of @p element_ at @p cursor's next aligned offset and advance past them. Free
 *  rather than a member, so a walk over a dependent cursor type needs no `template` disambiguator. */
template <typename element_, typename byte_type_>
[[nodiscard]] constexpr custd::span<pool_element_t<byte_type_, element_>> carve(pool_cursor<byte_type_> &cursor,
                                                                                std::size_t count) noexcept {
    cursor.bytes = align_up_(cursor.bytes, pool_alignment_k);
    std::size_t const byte_offset = cursor.bytes;
    cursor.bytes += count * sizeof(element_);
    if (cursor.pool.empty()) return {};
    // A pool allocated against a different shape overruns, and the overrun reads as valid device memory.
    sf_assert_(cursor.bytes <= cursor.pool.size());
    return {reinterpret_cast<pool_element_t<byte_type_, element_> *>(cursor.pool.data() + byte_offset), count};
}

#pragma endregion Pool Carving

} // namespace smallfield::types
