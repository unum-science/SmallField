/**
 *  @file test/api/test_conformer_slot_layout.cpp
 *  @brief Guard the structure-of-arrays conformer slot contract.
 *
 *  A shipped writer once copied `3 * num_atoms` contiguous halves from a slot base, which reads x for the real
 *  atoms and then inter-component padding -- so y and z were dropped for every molecule with
 *  `3 * num_atoms <= max_atoms`, and the stored geometry collapsed onto the x axis. Bond lengths and energies both
 *  still looked plausible; only a degeneracy check exposes it, so that is what this test asserts.
 */

#include <cassert> // `assert`
#include <cmath>   // `std::abs`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `fprintf`, `stderr`

#include <array>  // `std::array`
#include <vector> // `std::vector`

#include "smallfield/api/conformer_layout.hpp"

using smallfield::api::conformer_slot_layout_t;

namespace smallfield::tests {

/** Emulates the writer's slot-to-row transpose without the Arrow machinery. */
static std::vector<float> pack_row_(std::vector<float> const &slot, conformer_slot_layout_t const &layout,
                                    std::uint32_t atom_count) {
    std::vector<float> row(static_cast<std::size_t>(atom_count) * 3u);
    for (std::uint32_t atom = 0; atom < atom_count; ++atom)
        for (std::uint32_t axis = 0; axis < 3u; ++axis)
            row[atom * 3u + axis] = slot[layout.component_offset(axis, atom)];
    return row;
}

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
using smallfield::tests::pack_row_;

int main() {
    // A tetrahedron: genuinely three-dimensional, so any layout error collapses its rank.
    constexpr std::uint32_t atom_count = 4;
    float const reference[atom_count][3] = {{0.000f, 0.000f, 0.000f},
                                            {1.520f, 0.000f, 0.000f},
                                            {0.760f, 1.316f, 0.000f},
                                            {0.760f, 0.438f, 1.240f}};

    // `max_atoms` far exceeds `atom_count`, which is exactly the regime the bug hit:
    // 3 * 4 = 12 halves never reach the y component at offset 128.
    conformer_slot_layout_t const layout {128u, 3u};
    std::vector<float> buffer(layout.slot_offset(1u, 0u), 0.0f);
    for (std::uint32_t atom = 0; atom < atom_count; ++atom)
        for (std::uint32_t axis = 0; axis < 3u; ++axis)
            buffer[layout.component_offset(axis, atom)] = reference[atom][axis];

    std::vector<float> const row = pack_row_(buffer, layout, atom_count);

    // Round-trip fidelity.
    for (std::uint32_t atom = 0; atom < atom_count; ++atom)
        for (std::uint32_t axis = 0; axis < 3u; ++axis)
            assert(row[atom * 3u + axis] == reference[atom][axis] && "slot transpose lost a component");

    // A contiguous read collapses every point onto the x axis and so yields rank 1, while a correct
    // transpose yields rank 3. This is the assertion that separates the two.
    std::uint32_t const rank = coordinate_rank_(row, atom_count);
    if (rank != 3u) {
        std::fprintf(stderr, "FAIL: coordinate rank %u, expected 3 (geometry is degenerate)\n", rank);
        return 1;
    }

    // Slot offsets must not overlap, or conformers would alias each other.
    assert(layout.slot_offset(0u, 1u) - layout.slot_offset(0u, 0u) == layout.slot_stride());
    assert(layout.slot_offset(1u, 0u) == layout.slot_offset(0u, 0u) + 3u * layout.slot_stride());

    std::printf("test_conformer_slot_layout: OK (rank=%u, %u atoms)\n", rank, atom_count);
    return 0;
}
