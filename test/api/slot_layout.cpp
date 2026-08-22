/**
 *  @file test/api/slot_layout.cpp
 *  @brief Guard the structure-of-arrays conformer slot contract.
 *
 *  Reading a slot as `3 * num_atoms` contiguous halves from its base gets x and then inter-component
 *  padding, dropping y and z wherever `3 * num_atoms <= max_atoms` and collapsing the geometry onto the x
 *  axis; bond lengths and energies stay plausible under that, so only the degeneracy check exposes it.
 */

#include <cmath>   // `std::abs`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <array>       // `std::array`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`

#include "smallfield/api/conformer_layout.hpp"

#include "test/helpers.hpp"

namespace sftest = smallfield::tests;

using smallfield::api::conformer_slot_layout_t;

static constexpr std::string_view test_name_k = "smallfield_test_slot_layout";

namespace smallfield::tests {

/**
 *  @brief Whether `slot_view` addresses every component of a full slot where the contract says it does.
 *  The tetrahedron below covers four atoms; this walks all `3 * max_atoms` and alone pins the strides to
 *  `axis * max_atoms + atom`, which no call site spells out.
 */
static constexpr bool slot_view_matches_contract_() {
    conformer_slot_layout_t const layout {128u, 3u};
    custd::layout_stride::mapping<custd::dextents<std::uint32_t, 2>> const mapping =
        layout.slot_view<float>(nullptr, layout.max_atoms).mapping();
    for (std::uint32_t axis = 0; axis < 3u; ++axis)
        for (std::uint32_t atom = 0; atom < layout.max_atoms; ++atom)
            if (mapping(axis, atom) != static_cast<std::size_t>(axis) * layout.max_atoms + atom) return false;
    return true;
}

static_assert(slot_view_matches_contract_(), "a slot view must land every component where the contract puts it");

/** Rank of the centered coordinate cloud; a real conformer spans 3 dimensions. */
static std::uint32_t coordinate_rank_(std::vector<float> const &row, std::uint32_t atom_count) {
    float center[3] = {0.0f, 0.0f, 0.0f};
    for (std::uint32_t atom = 0; atom < atom_count; ++atom)
        for (std::uint32_t axis = 0; axis < 3u; ++axis) center[axis] += row[atom * 3u + axis];
    for (float &value : center) value /= static_cast<float>(atom_count);

    // Gaussian elimination with partial pivoting over the centered 3-column cloud.
    std::vector<std::array<float, 3>> rows(atom_count);
    for (std::uint32_t atom = 0; atom < atom_count; ++atom)
        for (std::uint32_t axis = 0; axis < 3u; ++axis) rows[atom][axis] = row[atom * 3u + axis] - center[axis];

    std::uint32_t rank = 0;
    for (std::uint32_t column = 0; column < 3u && rank < atom_count; ++column) {
        std::uint32_t pivot = rank;
        for (std::uint32_t candidate = rank; candidate < atom_count; ++candidate)
            if (std::abs(rows[candidate][column]) > std::abs(rows[pivot][column])) pivot = candidate;
        if (std::abs(rows[pivot][column]) < 1e-4f) continue;
        std::swap(rows[rank], rows[pivot]);
        for (std::uint32_t other = rank + 1; other < atom_count; ++other) {
            float const factor = rows[other][column] / rows[rank][column];
            for (std::uint32_t axis = 0; axis < 3u; ++axis) rows[other][axis] -= factor * rows[rank][axis];
        }
        ++rank;
    }
    return rank;
}

} // namespace smallfield::tests

using smallfield::tests::coordinate_rank_;

int main() {
    fmt::println("{}: structure-of-arrays slot transpose and slot offset arithmetic", test_name_k);

    // A tetrahedron: genuinely three-dimensional, so any layout error collapses its rank.
    constexpr std::uint32_t atom_count = 4;
    float const reference[atom_count][3] = {{0.000f, 0.000f, 0.000f},
                                            {1.520f, 0.000f, 0.000f},
                                            {0.760f, 1.316f, 0.000f},
                                            {0.760f, 0.438f, 1.240f}};

    // `max_atoms` far exceeds `atom_count`: 3 * 4 = 12 halves never reach the y component at offset 128.
    conformer_slot_layout_t const layout {128u, 3u};

    // Filled through the contract spelled out rather than the type implementing it: three planes of
    // `max_atoms`, component `axis` of `atom` at `axis * max_atoms + atom`, which is how a shipped shard's
    // `conformer_coords` is packed. Asking the layout would make the round trip agree with itself regardless.
    std::vector<float> buffer(layout.slot_offset(1u, 0u), 0.0f);
    for (std::uint32_t atom = 0; atom < atom_count; ++atom)
        for (std::uint32_t axis = 0; axis < 3u; ++axis)
            buffer[static_cast<std::size_t>(axis) * layout.max_atoms + atom] = reference[atom][axis];

    // The gather the Parquet writer, the descriptor tool and the benchmarks all reach the pool through.
    std::vector<float> row(static_cast<std::size_t>(atom_count) * 3u, 0.0f);
    smallfield::api::gather_slot_into_interleaved(buffer.data(), layout, atom_count, row.data());

    // Round-trip fidelity.
    for (std::uint32_t atom = 0; atom < atom_count; ++atom)
        for (std::uint32_t axis = 0; axis < 3u; ++axis)
            sf_verify_(row[atom * 3u + axis] == reference[atom][axis], "the slot transpose lost a component");

    // A contiguous read collapses every point onto x for rank 1; a correct transpose gives rank 3.
    std::uint32_t const rank = coordinate_rank_(row, atom_count);
    if (rank != 3u)
        return sftest::report_fail(test_name_k, "coordinate rank {}, expected 3 -- the geometry is degenerate", rank);

    // Slot offsets must not overlap, or conformers would alias each other.
    sf_verify_(layout.slot_offset(0u, 1u) - layout.slot_offset(0u, 0u) == layout.slot_stride(),
               "two conformers of one molecule must sit one slot stride apart");
    sf_verify_(layout.slot_offset(1u, 0u) == layout.slot_offset(0u, 0u) + 3u * layout.slot_stride(),
               "the next molecule must start num_conformers slot strides on");

    fmt::println("  rank={} atoms={}", rank, atom_count);
    return sftest::report_pass(test_name_k);
}
