/**
 *  @file bench/bench_arguments.hpp
 *  @brief Whole-token numeric parsing shared by the benchmark drivers.
 *
 *  `std::atoi` and `std::strtoull` both answer a malformed token with zero, so `--conformers abc` reads as
 *  `--conformers 0` and the run continues on a number nobody typed. Every benchmark flag goes through
 *  @ref parse_number instead, which reports why a token was refused and enforces the caller's bounds in the
 *  same pass — the ceiling is what stops a four-billion conformer count from reaching `std::vector::resize`.
 *
 *  @code{.cpp}
 *  std::uint32_t num_conformers = 3;
 *  auto const status = smallfield::bench::parse_number(argv[index], 1u, 1024u, num_conformers);
 *  if (status != smallfield::bench::number_parse_t::parsed_k)
 *      fmt::println(stderr, "--conformers {}", smallfield::bench::describe_number_parse(status));
 *  @endcode
 */

#pragma once

#include <charconv>     // `std::from_chars`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`

namespace smallfield::bench {

/** @brief Why a command-line token was accepted or refused as a number. */
enum class number_parse_t {
    parsed_k,        ///< The whole token is a number inside the accepted range.
    malformed_k,     ///< The token is not a number, or carries trailing characters.
    out_of_range_k,  ///< The token is a number the destination type cannot hold.
    below_minimum_k, ///< The token parses but falls under the caller's floor.
    above_maximum_k, ///< The token parses but exceeds the caller's ceiling.
};

/**
 *  @brief Parse @p text whole as a number in `[minimum, maximum]`, leaving @p number untouched on refusal.
 *
 *  Partial parses are refusals: `12x` is malformed, not twelve. The destination keeps its prior value unless
 *  the token is accepted, so a caller may seed it with the default and read it back either way.
 */
template <typename number_type_>
number_parse_t parse_number(std::string_view text, number_type_ minimum, number_type_ maximum,
                            number_type_ &number) {
    number_type_ parsed {};
    auto const [unparsed, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error == std::errc::result_out_of_range) return number_parse_t::out_of_range_k;
    if (error != std::errc {} || unparsed != text.data() + text.size()) return number_parse_t::malformed_k;
    if (parsed < minimum) return number_parse_t::below_minimum_k;
    if (parsed > maximum) return number_parse_t::above_maximum_k;
    number = parsed;
    return number_parse_t::parsed_k;
}

/** @brief Sentence fragment naming the refusal, meant to follow the flag name in a diagnostic. */
inline char const *describe_number_parse(number_parse_t status) {
    switch (status) {
    case number_parse_t::parsed_k: return "parsed";
    case number_parse_t::malformed_k: return "is not a whole number";
    case number_parse_t::out_of_range_k: return "does not fit the destination type";
    case number_parse_t::below_minimum_k: return "is below the accepted minimum";
    case number_parse_t::above_maximum_k: return "is above the accepted maximum";
    }
    return "was refused";
}

} // namespace smallfield::bench
