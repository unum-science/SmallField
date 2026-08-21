/**
 *  @file cli/cli_arguments.hpp
 *  @brief `argparse` bindings shared by the `cli/` binaries, writing a parsed value into its destination.
 *
 *  `argparse`'s own `scan` answers a malformed token without naming the flag, so every value-taking flag goes
 *  through `parse_number` in `smallfield/number_parse.hpp`, which names the refusal and bounds in one pass.
 */

#pragma once

#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`

#include <argparse/argparse.hpp>
#include <fmt/format.h>

#include "smallfield/number_parse.hpp"

namespace smallfield::cli {

/** Outcome of a command line: a usable configuration, a help request, or a rejected command line. */
enum class cli_parse_t {
    parsed_k,
    help_requested_k,
    invalid_k,
};

/** Bind a string flag straight to its destination. `append` marks it repeatable so the last occurrence wins;
 *  without it `argparse` refuses a second occurrence outright. */
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
