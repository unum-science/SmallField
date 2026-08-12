/**
 *  @file src/smallfield/preprocess/csd_torsion_table.hpp
 *  @brief The CSD torsion-preference corpus, as a flat table of plain data.
 *
 *  The rows themselves live in a host-only `.cpp`, so nothing here costs a device translation unit
 *  41 KB of string literals it must tokenize once per architecture for data no kernel reads.
 */

#pragma once

#include <cstdint> // `std::int8_t`

#include "smallfield/types/views.hpp"

namespace smallfield::preprocess {

/** One torsion-preference row: a SMARTS pattern and its six Fourier terms. */
struct csd_torsion_row_t {
    /** Points into static storage in the generated table, so it outlives every caller. */
    char const *smarts;
    /** Each is -1 or +1; the corpus carries no other value. */
    std::int8_t signs[6];
    /** In kcal/mol. Stored as `float` because that is what the device descriptor holds. */
    float magnitudes[6];
};

/** The general corpus, from RDKit's `torsionPreferences_v2.in`. */
custd::span<csd_torsion_row_t const> csd_torsion_rows_v2();

/** The small-ring corpus, from RDKit's `torsionPreferences_smallrings.in`, consulted after v2. */
custd::span<csd_torsion_row_t const> csd_torsion_rows_small_rings();

} // namespace smallfield::preprocess
