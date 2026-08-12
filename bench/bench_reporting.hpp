/**
 *  @file bench/bench_reporting.hpp
 *  @brief The per-status breakdown every conformer benchmark prints beneath its throughput line.
 *
 *  A `conf/s` figure already divides by produced conformers, so only this breakdown makes a rejection class
 *  visible: the atom-count cap alone drops about 4 % of PubChem before any molecule reaches the GPU, and a
 *  class that grows quietly looks like a speed-up in every number except this table.
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`

#include <array>       // `std::array`
#include <string_view> // `std::string_view`

#include <fmt/format.h>

#include "smallfield/api/conformer_status.hpp"
#include "smallfield/types/views.hpp"

namespace smallfield::bench {

/** Label for a status byte no enumerator claims, which a shard from a newer producer can still carry. */
inline constexpr std::string_view unnamed_status_k = "unnamed_status";

/**
 *  @brief The label this breakdown prints for @p status, which is the enumerator without its suffix.
 *
 *  Exhaustive over `conformer_status_t` and carrying no `default:`, so a status appended to that
 *  append-only enum is a `-Wswitch` warning here rather than a row printed under a neighbour's name.
 */
inline constexpr std::string_view conformer_status_name(api::conformer_status_t status) noexcept {
    switch (status) {
    case api::conformer_status_t::success_k: return "success";
    case api::conformer_status_t::stereo_failure_k: return "stereo_failure";
    case api::conformer_status_t::mmff_unconverged_k: return "mmff_unconverged";
    case api::conformer_status_t::invalid_smiles_k: return "invalid_smiles";
    case api::conformer_status_t::too_many_atoms_k: return "too_many_atoms";
    case api::conformer_status_t::preprocess_timeout_k: return "preprocess_timeout";
    case api::conformer_status_t::unsupported_element_k: return "unsupported_element";
    case api::conformer_status_t::too_many_bonds_k: return "too_many_bonds";
    case api::conformer_status_t::too_many_rings_k: return "too_many_rings";
    case api::conformer_status_t::bounds_infeasible_k: return "bounds_infeasible";
    }
    return unnamed_status_k;
}

/** Conformers per status code, indexed by the enumerator's own value; the trailing bucket collects the
 *  codes no enumerator claims. */
using status_histogram_t = std::array<std::uint64_t, api::conformer_status_count_k + 1>;

static_assert(conformer_status_name(static_cast<api::conformer_status_t>(api::conformer_status_count_k - 1)) !=
                      unnamed_status_k &&
                  conformer_status_name(static_cast<api::conformer_status_t>(api::conformer_status_count_k)) ==
                      unnamed_status_k,
              "every bucket but the trailing one carries a name, and the trailing one collects only the rest");

/** Tally @p status by code in one pass; a code no enumerator claims lands in the last bucket. */
inline status_histogram_t status_histogram_of(custd::span<api::conformer_status_t const> status) {
    status_histogram_t histogram {};
    for (api::conformer_status_t const status_code : status) {
        std::size_t const code = static_cast<std::size_t>(status_code);
        ++histogram[code < histogram.size() ? code : histogram.size() - 1];
    }
    return histogram;
}

/** Conformers that survived. */
inline std::uint64_t successful_conformers_of(status_histogram_t const &histogram) {
    return histogram[static_cast<std::size_t>(api::conformer_status_t::success_k)];
}

/** Print every non-empty failure class in @p histogram as a count and a share of @p total_conformers. */
inline void print_status_histogram(status_histogram_t const &histogram, std::size_t total_conformers) {
    if (total_conformers == 0) return;
    for (std::size_t code = 1; code < histogram.size(); ++code)
        if (histogram[code] > 0)
            fmt::println("  {:<20} {:>12}  {:.3f} %", conformer_status_name(static_cast<api::conformer_status_t>(code)),
                         fmt::group_digits(histogram[code]),
                         100.0 * static_cast<double>(histogram[code]) / static_cast<double>(total_conformers));
}

} // namespace smallfield::bench
