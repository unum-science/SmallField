/**
 *  @file src/smallfield/number_parse.hpp
 *  @brief Bounded whole-token number parsing, shared by the `cli/` and `bench/` command lines.
 *
 *  `std::atoi` and `std::strtoull` answer a malformed token with zero, and `argparse`'s own `scan` answers it
 *  without naming the flag that carried it, so a run continues on a number nobody typed. Every value-taking
 *  flag goes through `parse_number` instead, which names the refusal and enforces the caller's bounds in the
 *  same pass.
 *
 *  A root header rather than `utilities.hpp`, because that one includes nothing at all: three `.cuh` headers
 *  pull it in, and `<charconv>` is host-only weight no kernel can reach.
 */

#pragma once

#include <charconv>     // `std::from_chars`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`

namespace smallfield {

/** Why a command-line token was accepted or refused as a number. */
enum class number_parse_t {
    /** The whole token is a number inside the accepted range. */
    parsed_k,
    /** The token is not a number, or carries trailing characters. */
    malformed_k,
    /** The token is a number the destination type cannot hold. */
    out_of_range_k,
    /** The token parses but falls under the caller's floor. */
    below_minimum_k,
    /** The token parses but exceeds the caller's ceiling. */
    above_maximum_k,
};

/**
 *  @brief Parse @p text whole as a number in `[minimum, maximum]`, leaving @p number untouched on refusal.
 *
 *  Partial parses are refusals: `12x` is malformed, not twelve. The destination keeps its prior value unless
 *  the token is accepted, so a caller may seed it with the default and read it back either way.
 *
 *  @param[in] text The token as typed, parsed in full or not at all.
 *  @param[in] minimum Lowest accepted value, inclusive.
 *  @param[in] maximum Highest accepted value, inclusive.
 *  @param[out] number Written only when the token is accepted.
 */
template <typename number_type_>
number_parse_t parse_number(std::string_view text, number_type_ minimum, number_type_ maximum, number_type_ &number) {
    number_type_ parsed {};
    auto const [unparsed, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error == std::errc::result_out_of_range) return number_parse_t::out_of_range_k;
    if (error != std::errc {} || unparsed != text.data() + text.size()) return number_parse_t::malformed_k;
    if (parsed < minimum) return number_parse_t::below_minimum_k;
    if (parsed > maximum) return number_parse_t::above_maximum_k;
    number = parsed;
    return number_parse_t::parsed_k;
}

/** Sentence fragment naming the refusal, meant to follow the flag name in a diagnostic. */
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

} // namespace smallfield
