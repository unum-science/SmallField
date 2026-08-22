/**
 *  @file bench/bench_arguments.hpp
 *  @brief Command-line and corpus plumbing shared by the benchmark drivers.
 *
 *  Every flag goes through `parse_number` from `smallfield/number_parse.hpp`, whose ceiling stops a four-billion
 *  conformer count from reaching `std::vector::resize`.
 */

#pragma once

#include <cstddef> // `std::size_t`
#include <cstdio>  // `stderr`
#include <cstdlib> // `std::getenv`

#include <algorithm>   // `std::min`
#include <fstream>     // `std::ifstream`
#include <string>      // `std::string`, `std::getline`
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h>

#include "smallfield/number_parse.hpp"

namespace smallfield::bench {

/** Environment variable naming a corpus when no `--smi` flag is given. */
inline constexpr char const *corpus_environment_variable_k = "SMALLFIELD_TEST_CORPUS";

/** Why a SMILES corpus was accepted or refused. A path that never opened stays apart from one that opened and held
 *  nothing — and `unnamed_k` from both, being neither a flag nor the environment naming one — since a zero-molecule
 *  corpus measures nothing and still exits clean. */
enum class corpus_load_t {
    loaded_k,
    unnamed_k,
    cannot_open_k,
    empty_k,
};

/** Read up to @p capacity SMILES from @p path into a cleared @p smiles, one per line. Empty and `#` lines are
 *  skipped; each kept line truncates at its first whitespace run, dropping a trailing name field along with any
 *  `\r`. */
inline corpus_load_t load_corpus_into(std::string_view path, std::size_t capacity, std::vector<std::string> &smiles) {
    smiles.clear();
    if (path.empty()) return corpus_load_t::unnamed_k;
    std::ifstream input_stream {std::string {path}};
    if (!input_stream) return corpus_load_t::cannot_open_k;
    smiles.reserve(std::min<std::size_t>(capacity, 1'000'000));
    std::string line;
    while (smiles.size() < capacity && std::getline(input_stream, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::size_t const separator_index = line.find_first_of(" \t\r");
        if (separator_index != std::string::npos) line.resize(separator_index);
        if (!line.empty()) smiles.push_back(std::move(line));
    }
    return smiles.empty() ? corpus_load_t::empty_k : corpus_load_t::loaded_k;
}

/** Sentence fragment naming the refusal, meant to follow the corpus path in a diagnostic. */
inline char const *describe_corpus_load(corpus_load_t status) {
    switch (status) {
    case corpus_load_t::loaded_k: return "loaded";
    case corpus_load_t::unnamed_k: return "was never named — pass --smi PATH or set SMALLFIELD_TEST_CORPUS";
    case corpus_load_t::cannot_open_k: return "cannot be opened";
    case corpus_load_t::empty_k: return "holds no SMILES";
    }
    return "was refused";
}

/** Corpus path named by the environment, or an empty view when the variable is unset. */
inline std::string_view corpus_path_from_environment() {
    char const *const named = std::getenv(corpus_environment_variable_k);
    return named != nullptr ? std::string_view {named} : std::string_view {};
}

#pragma region Command Line

/** How a command line was answered: usable arguments, a help request, or a refusal. */
enum class arguments_parse_t { accepted_k, help_requested_k, refused_k };

/** The command line as one cursor: the tokens, the position in them, and the name to blame. Three things are
 *  refusals rather than silent defaults: a non-numeric token, one outside the caller's bounds, and a value-taking
 *  flag appearing last with nothing after it. */
class command_line_t {
    char const *executable_name_;
    int argc_;
    char **argv_;
    int index_ = 0;

  public:
    command_line_t(char const *executable_name, int argc, char **argv) noexcept
        : executable_name_(executable_name), argc_(argc), argv_(argv) {}

    /** Step onto the next flag, or report that the line is exhausted. */
    [[nodiscard]] bool advance() noexcept { return ++index_ < argc_; }

    /** The flag the cursor stands on. */
    [[nodiscard]] std::string_view flag() const noexcept { return argv_[index_]; }

    /** Consume the token after the current flag as text. */
    [[nodiscard]] arguments_parse_t take_text(std::string_view &destination) {
        if (index_ + 1 >= argc_) {
            fmt::println(stderr, "{}: {} takes a value, and none followed it", executable_name_, flag());
            return arguments_parse_t::refused_k;
        }
        destination = argv_[++index_];
        return arguments_parse_t::accepted_k;
    }

    /** Consume the token after the current flag as a number in `[1, maximum]`. */
    template <typename number_type_>
    [[nodiscard]] arguments_parse_t take_number(number_type_ maximum, number_type_ &number) {
        // Named before the cursor moves off it, because after `take_text` the cursor stands on the value.
        std::string_view const flag_text = flag();
        std::string_view text;
        if (take_text(text) != arguments_parse_t::accepted_k) return arguments_parse_t::refused_k;
        number_parse_t const status = parse_number(text, number_type_ {1}, maximum, number);
        if (status == number_parse_t::parsed_k) return arguments_parse_t::accepted_k;
        fmt::println(stderr, "{}: {} value '{}' {} (accepted range 1..{})", executable_name_, flag_text, text,
                     describe_number_parse(status), fmt::group_digits(maximum));
        return arguments_parse_t::refused_k;
    }

    /** Name the flag nobody understands, so a run under a typo never reports the defaults as what was asked. */
    [[nodiscard]] arguments_parse_t refuse_unknown() const {
        fmt::println(stderr, "{}: unknown argument '{}'", executable_name_, flag());
        return arguments_parse_t::refused_k;
    }
};

#pragma endregion Command Line

} // namespace smallfield::bench
