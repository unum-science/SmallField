/**
 *  @file cli/cli_arguments.hpp
 *  @brief Command-line plumbing shared by the `cli/` binaries: bounded number parsing and the `argparse`
 *  bindings that write a parsed value straight into its destination.
 *
 *  `std::atoi` and `std::strtoull` both answer a malformed token with zero, so `--num-conformers abc` reads
 *  as zero and the run continues on a number nobody typed — which for that flag is a wave of nothing the
 *  library reports as a CUDA fault. Every value-taking flag goes through `parse_number` instead, which
 *  reports @b why a token was refused and enforces the caller's bounds in the same pass.
 *
 *  The conversion stays ours rather than `argparse`'s `scan`, whose diagnostic is `pattern 'abc' does not
 *  match to the end` — it never names the flag the operator actually typed, and it has no bounds to enforce.
 */

#pragma once

#include <charconv>     // `std::from_chars`
#include <stdexcept>    // `std::runtime_error`
#include <string>       // `std::string`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`

#include <argparse/argparse.hpp>
#include <fmt/format.h>

namespace smallfield::cli {

/** Outcome of a command line: a usable configuration, a help request, or a rejected command line. */
enum class cli_parse_t {
    parsed_k,
    help_requested_k,
    invalid_k,
};

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
 *  @param[in] text Token as it appeared on the command line.
 *  @param[in] minimum Smallest value the flag accepts.
 *  @param[in] maximum Largest value the flag accepts.
 *  @param[out] number Value the token denotes; untouched unless the whole token parses inside the bounds.
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

/**
 *  @brief Bind a string flag straight to its destination, so the parsed value never sits anywhere but the config.
 *
 *  `append` marks the flag repeatable, which is what keeps the last occurrence winning: without it `argparse`
 *  rejects a second occurrence outright, and a wrapper that appends an override to a fixed prefix stops working.
 */
template <typename string_type_>
void add_string(argparse::ArgumentParser &parser, char const *flag, string_type_ &destination) {
    parser.add_argument(flag).append().action([&destination](std::string const &text) { destination = text; });
}

/** Bind a numeric flag bounded by `[minimum, maximum]`, naming both the flag and the reason on refusal. */
template <typename number_type_>
void add_number(argparse::ArgumentParser &parser, char const *flag, number_type_ minimum, number_type_ maximum,
                number_type_ &number) {
    parser.add_argument(flag).append().action([flag, minimum, maximum, &number](std::string const &text) {
        number_parse_t const status = parse_number(text, minimum, maximum, number);
        if (status != number_parse_t::parsed_k)
            throw std::runtime_error(fmt::format("{} {}, got: {}", flag, describe_number_parse(status), text));
    });
}

} // namespace smallfield::cli
