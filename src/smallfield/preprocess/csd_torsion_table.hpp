/**
 *  @file src/smallfield/preprocess/csd_torsion_table.hpp
 *  @brief The CSD torsion-preference corpus, as a flat table of plain data.
 *
 *  RDKit ships this corpus as escaped C++ string literals that a consumer concatenates into one
 *  `std::string` and re-parses at load. Reproducing that here put roughly 41 KB of literals inside a
 *  header that every CUDA translation unit compiles once per architecture, for data no kernel reads,
 *  and then spent one `std::istringstream` and two heap allocations per row turning it back into
 *  numbers. The rows are a table with fixed arity, so they are stored as one.
 *
 *  The table itself lives in a host-only `.cpp`. Only this declaration is safe for a device
 *  translation unit to see, and there is nothing here for nvcc to tokenize.
 *
 *  Row order is semantic - `extract_etk_torsions_native` gives a bond to the first pattern that
 *  claims it, and the corpus is ordered most-specific first, so sorting or regrouping these rows
 *  changes the conformers the pipeline generates while everything still compiles. Anything that
 *  wants to group them has to preserve the sequence. `csd_torsion_table.cpp` carries the snippet
 *  that regenerates it from an RDKit checkout.
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
