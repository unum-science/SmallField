/**
 *  @file bench/reporting.hpp
 *  @brief The per-status breakdown every conformer benchmark prints beneath its throughput line.
 *
 *  A `conf/s` figure already divides by produced conformers, so only this breakdown makes a rejection class visible —
 *  the atom-count cap alone drops about 4 % of PubChem before any molecule reaches the GPU, and a class that grows
 *  quietly looks like a speed-up in every number except this table.
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`

#include <array>       // `std::array`
#include <string_view> // `std::string_view`

#include <fmt/format.h>

#include "smallfield/types/conformer_status.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::bench {

using types::conformer_status_name;
using types::unnamed_status_k;

/** Conformers per status code, indexed by the enumerator's own value; the last bucket collects unclaimed codes. */
using status_histogram_t = std::array<std::uint64_t, types::conformer_status_count_k + 1>;

/** Tally @p status by code in one pass; a code no enumerator claims lands in the last bucket. */
inline status_histogram_t status_histogram_of(custd::span<types::conformer_status_t const> status) {
    status_histogram_t histogram {};
    for (types::conformer_status_t const status_code : status) {
        std::size_t const code = static_cast<std::size_t>(status_code);
        ++histogram[code < histogram.size() ? code : histogram.size() - 1];
    }
    return histogram;
}

/** Conformers that survived. */
inline std::uint64_t successful_conformers_of(status_histogram_t const &histogram) {
    return histogram[static_cast<std::size_t>(types::conformer_status_t::success_k)];
}

/** Print every non-empty failure class in @p histogram as a count and a share of @p total_conformers. */
inline void print_status_histogram(status_histogram_t const &histogram, std::size_t total_conformers) {
    if (total_conformers == 0) return;
    for (std::size_t code = 1; code < histogram.size(); ++code)
        if (histogram[code] > 0)
            fmt::println("  {:<20} {:>12}  {:.3f} %",
                         conformer_status_name(static_cast<types::conformer_status_t>(code)),
                         fmt::group_digits(histogram[code]),
                         100.0 * static_cast<double>(histogram[code]) / static_cast<double>(total_conformers));
}

} // namespace smallfield::bench
