/**
 *  @file bench/bench_reporting.hpp
 *  @brief The per-status breakdown every conformer benchmark prints beneath its throughput line.
 *
 *  A `conf/s` figure already divides by produced conformers, so it is comparable across corpora with
 *  different rejection rates — but only the breakdown makes a rejection class visible. The atom-count cap
 *  alone drops about 4 % of PubChem before any molecule reaches the GPU, and a class that grows quietly
 *  looks like a speed-up in every number except this table.
 *
 *  `bench/` is read as a report, so it prints through bare `fmt::println` rather than the logger: a
 *  34-character varying timestamp in front of every row destroys the column alignment the table exists for.
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`

#include <array> // `std::array`

#include <fmt/format.h>

#include "smallfield/api/conformer_status.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::bench {

/** Conformers per status code, indexed by the enumerator's own value. */
using status_histogram_t = std::array<std::uint64_t, 8>;

/** Label naming each status code, in enumerator order; the last slot covers a code no enumerator claims. */
inline constexpr std::array<char const *, 8> status_names_k {
    "success",        "stereo_failure",     "mmff_unconverged",    "invalid_smiles",
    "too_many_atoms", "preprocess_timeout", "unsupported_element", "other"};

/** Tally @p status by code in one pass, so a rejection class cannot grow behind a throughput number. */
inline status_histogram_t status_histogram_of(custd::span<api::conformer_status_t const> status) {
    status_histogram_t histogram {};
    for (api::conformer_status_t const status_code : status) {
        std::size_t const code = static_cast<std::size_t>(status_code);
        ++histogram[code < histogram.size() ? code : histogram.size() - 1];
    }
    return histogram;
}

/** Conformers that survived, which is the success bucket rather than a counter kept beside it. */
inline std::uint64_t successful_conformers_of(status_histogram_t const &histogram) {
    return histogram[static_cast<std::size_t>(api::conformer_status_t::success_k)];
}

/** Print every non-empty failure class in @p histogram as a count and a share of @p total_conformers. */
inline void print_status_histogram(status_histogram_t const &histogram, std::size_t total_conformers) {
    if (total_conformers == 0) return;
    for (std::size_t code = 1; code < histogram.size(); ++code)
        if (histogram[code] > 0)
            fmt::println("  {:<20} {:>12}  {:.3f} %", status_names_k[code], fmt::group_digits(histogram[code]),
                         100.0 * static_cast<double>(histogram[code]) / static_cast<double>(total_conformers));
}

} // namespace smallfield::bench
