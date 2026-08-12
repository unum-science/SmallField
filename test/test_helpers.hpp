/**
 *  @file test/test_helpers.hpp
 *  @brief One machine-readable grammar for every diagnostic the C++ tests print.
 *
 *  A diagnostic is read twice: once by a person scanning a CI log, and once by `grep`, `awk`, or a
 *  diff between two runs.  Prose serves the first reader and defeats the second, so every line here
 *  opens with a verb at column zero, carries ASCII `key=value` pairs separated by exactly one space,
 *  and names the single field that disagreed rather than dumping every field of the record it
 *  belongs to:
 *
 *      MISMATCH vdw atom_type=12 field=alpha rdkit=1.5 ours=1.5000001 delta=+1.19e-07
 *      MISMATCH bond bond_type=0 atom_i=3 atom_j=7 field=force_constant rdkit=4.928 ours=6.947 delta=+2.019
 *      DIVERGENCE bounds_1_4 smiles=C/C=C/C pair=0,3 upstream=BoundsMatrixBuilder.cpp:_getAtomStereo …
 *      PASS vdw compared=1245 mismatches=0 divergences=0 suppressed=0
 *      PASS test_mmff94_tables
 *      FAIL test_mmff94_tables: 3 mismatches over 4838 comparisons
 *
 *  `^MISMATCH` filters, `awk -F'[ =]'` parses, and the failing field names itself instead of leaving
 *  the reader to spot which of five printed numbers moved.  Nothing carries a timestamp: ctest stamps
 *  its own, and a varying prefix would destroy the column alignment and make two runs undiffable.
 *
 *  `MISMATCH` and `DIVERGENCE` are counted apart because they mean different things. A mismatch is an
 *  unexplained difference and fails its section; a divergence is one the suite asserts on purpose,
 *  because upstream is wrong and we are right, and it carries the citation that says so. Folding the
 *  second into the first would leave a suite only able to widen a tolerance until the disagreement
 *  stopped failing, which is what destroys the tolerance's meaning.
 *
 *  Precision belongs to a value's role, never to the call site, so no call site writes a format
 *  specifier.  The roles and their specs:
 *
 *      compared value  `{}`      shortest round-trip, so two distinct floats can never render alike
 *      delta           `{:+.3g}` floating, or `{:+}` when both operands are integral
 *      count           `{}`
 *      tolerance       `{:.3g}`  reserved -- see `comparison_group_t`, which never owns a tolerance
 *      duration in ms  `{:.1f}`  reserved until a timing line is migrated
 *      percentage      `{:.3f}`  reserved until a rate line is migrated
 *
 *  Mismatches and failures go to stderr and passes and narration to stdout, which keeps the readable
 *  story on one stream and the machine-readable complaints on the other.
 */
#pragma once

#include <cmath>   // `std::fabs`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`
#include <cstdio>  // `std::FILE`, `std::fwrite`, `stdout`, `stderr`
#include <cstdlib> // `std::exit`

#include <algorithm>   // `std::max`
#include <bit>         // `std::bit_cast`
#include <iterator>    // `std::back_inserter`
#include <limits>      // `std::numeric_limits`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::common_type_t`, `std::is_floating_point_v`
#include <utility>     // `std::forward`, `std::move`

#include <fmt/format.h> // `fmt::memory_buffer`, `fmt::format_to`, `fmt::println`

namespace smallfield::tests {

#pragma region Grammar

/** The verb that opens a diagnostic line, at column zero so `^MISMATCH` filters a CI log. */
enum class diagnostic_verb_t : std::uint8_t {
    /** Two operands expected to agree do not. */
    mismatch_k,
    /** A section closed with every comparison satisfied. */
    pass_k,
    /** A section closed with mismatches, or with nothing compared at all. */
    fail_k,
    /** Narration: an operand pair worth printing whether or not it differs. */
    note_k,
    /** A difference the suite asserts on purpose, because the reference is the one that is wrong. */
    divergence_k,
};

/** The literal a verb prints as. */
inline std::string_view diagnostic_verb_name(diagnostic_verb_t verb) noexcept {
    switch (verb) {
    case diagnostic_verb_t::mismatch_k: return "MISMATCH";
    case diagnostic_verb_t::pass_k: return "PASS";
    case diagnostic_verb_t::fail_k: return "FAIL";
    case diagnostic_verb_t::divergence_k: return "DIVERGENCE";
    default: return "NOTE";
    }
}

/** Complaints on stderr, narration on stdout, so a passing run's stdout stays the readable story. */
inline std::FILE *diagnostic_stream(diagnostic_verb_t verb) noexcept {
    switch (verb) {
    case diagnostic_verb_t::mismatch_k:
    case diagnostic_verb_t::fail_k: return stderr;
    /*  `DIVERGENCE` joins `PASS` and `NOTE` on stdout. It is an asserted and cited difference rather
     *  than a complaint, so a reader watching stderr sees only the differences nobody has explained. */
    default: return stdout;
    }
}

/** Append a compared operand at shortest round-trip; a flag prints as `1`/`0` to keep its delta integral. */
template <typename value_type_>
inline void append_compared_value(fmt::memory_buffer &out, value_type_ const &value) {
    if constexpr (std::is_same_v<value_type_, bool>) fmt::format_to(std::back_inserter(out), "{}", value ? 1 : 0);
    else fmt::format_to(std::back_inserter(out), "{}", value);
}

/** Append `observed - reference` at `{:+.3g}` when either operand is floating, at `{:+}` when both are integral. */
template <typename reference_type_, typename observed_type_>
inline void append_delta(fmt::memory_buffer &out, reference_type_ const &reference, observed_type_ const &observed) {
    using promoted_type_ = std::common_type_t<reference_type_, observed_type_>;
    if constexpr (std::is_floating_point_v<promoted_type_>) {
        promoted_type_ const delta = static_cast<promoted_type_>(observed) - static_cast<promoted_type_>(reference);
        fmt::format_to(std::back_inserter(out), "{:+.3g}", delta);
    }
    else {
        long long const delta = static_cast<long long>(observed) - static_cast<long long>(reference);
        fmt::format_to(std::back_inserter(out), "{:+}", delta);
    }
}

#pragma endregion Grammar

#pragma region Predicates

/*  The predicates `comparison_group_t::field` is handed, which is why each is a `constexpr` callable rather
 *  than a function carrying a default argument: a function name decays to a pointer whose type does not
 *  include the default, so it cannot be passed where a two-argument callable is wanted and every call site
 *  grows a lambda to bind what the declaration already said.
 *
 *  The two tolerance predicates narrow to `float` before subtracting. A reference tabulated in `double` is
 *  being asked whether our `float` reproduces it at our own precision, and subtracting in `double` would
 *  charge us for the narrowing itself -- at `1e-5` over the MMFF94 angle table, half an ULP of `theta0` is
 *  most of that budget. */

/** Exact equality, promoted so a signed operand and an unsigned one still compare as numbers. */
inline constexpr auto exactly_equal = [](auto const &reference, auto const &observed) {
    using promoted_type_ = std::common_type_t<std::decay_t<decltype(reference)>, std::decay_t<decltype(observed)>>;
    return static_cast<promoted_type_>(reference) == static_cast<promoted_type_>(observed);
};

/** Bit-pattern identity, for a comparison written to catch the one-ULP divergence any tolerance would hide. */
inline constexpr auto bitwise_equal = [](auto const &reference, auto const &observed) {
    static_assert(std::is_same_v<std::decay_t<decltype(reference)>, float> &&
                      std::is_same_v<std::decay_t<decltype(observed)>, float>,
                  "bitwise equality is written for `float`, and a wider operand would narrow into it silently");
    return std::bit_cast<std::uint32_t>(reference) == std::bit_cast<std::uint32_t>(observed);
};

/**
 *  @brief Agreement at `1e-3`, scaled by the larger operand once that passes unit magnitude.
 *
 *  For a quantity whose error grows with itself, and which spans orders of magnitude -- a force constant, an
 *  equilibrium distance, a torsion amplitude. A quantity whose error is instead a fixed accumulation of
 *  rounding, such as an MMFF94 partial charge, takes `equal_within` and names its threshold, so that a large
 *  operand cannot quietly buy slack the arithmetic behind it never spent.
 */
inline constexpr auto approximately_equal = [](auto const &reference, auto const &observed) {
    float const expected = static_cast<float>(reference), measured = static_cast<float>(observed);
    return std::fabs(expected - measured) <= 1e-3f * std::max(1.0f, std::max(std::fabs(expected), std::fabs(measured)));
};

/** Agreement at an absolute @p tolerance, which the call site therefore names rather than inherits. */
inline constexpr auto equal_within = [](float tolerance) {
    return [tolerance](auto const &reference, auto const &observed) {
        return std::fabs(static_cast<float>(reference) - static_cast<float>(observed)) <= tolerance;
    };
};

#pragma endregion Predicates

#pragma region Reporter

class diagnostic_reporter_t;

/**
 *  @brief One group of diagnostic lines sharing a verb, a subject, and the caller's key columns.
 *
 *  A group is opened per record under test and costs one short string; its fields decide, one by one,
 *  whether a line appears.  Opening a group that emits nothing is the normal case and is free of any
 *  output, which is what lets a call site drop the `if` that used to guard a hand-rolled `println`.
 */
class diagnostic_group_t {
  public:
    /** Fields that produced a line, counted even when the reporter's line budget suppressed the print. */
    std::size_t reported_field_count() const noexcept { return reported_field_count_; }

  protected:
    diagnostic_group_t(diagnostic_reporter_t &owner, diagnostic_verb_t verb, std::string prefix) noexcept
        : owner_(&owner), verb_(verb), prefix_(std::move(prefix)) {}

    /** Print `field=<name>` with both operands and their delta, subject to the reporter's line budget. */
    template <typename reference_type_, typename observed_type_>
    void report_(std::string_view name, reference_type_ const &reference, observed_type_ const &observed);

    void count_comparison_() noexcept;
    void count_mismatch_() noexcept;
    void count_divergence_() noexcept;

  private:
    diagnostic_reporter_t *owner_;
    diagnostic_verb_t verb_;
    std::string prefix_;
    std::size_t reported_field_count_ = 0;
};

/**
 *  @brief A `MISMATCH` group whose fields print only where the caller's predicate says they disagree.
 *
 *  The predicate is the caller's on purpose.  The suites compare bit patterns, absolute error under a
 *  strict `<` at 1e-5, and absolute error rejected by a `>` at 1e-2 -- three boundaries no single
 *  built-in tolerance reproduces without silently moving one of them.  It is invoked as
 *  `equals(reference, observed)` and must answer "these agree".
 */
class comparison_group_t final : public diagnostic_group_t {
  public:
    template <typename reference_type_, typename observed_type_, typename equals_type_>
    comparison_group_t &field(std::string_view name, reference_type_ const &reference, observed_type_ const &observed,
                              equals_type_ equals) {
        count_comparison_();
        if (equals(reference, observed)) return *this;
        count_mismatch_();
        report_(name, reference, observed);
        return *this;
    }

  private:
    friend class diagnostic_reporter_t;
    comparison_group_t(diagnostic_reporter_t &owner, std::string prefix) noexcept
        : diagnostic_group_t(owner, diagnostic_verb_t::mismatch_k, std::move(prefix)) {}
};

/**
 *  @brief A `NOTE` group whose fields print whether they agree or not.
 *
 *  For the operand pairs a passing run still wants to see -- a reference energy beside ours -- where the
 *  number itself is the point and nothing is being asserted about it. Nothing here counts as a
 *  comparison, so narration can never make a validator look busier than it is. A difference that @b is
 *  asserted, and cited, is a `divergence_group_t` instead.
 */
class narration_group_t final : public diagnostic_group_t {
  public:
    template <typename reference_type_, typename observed_type_>
    narration_group_t &field(std::string_view name, reference_type_ const &reference, observed_type_ const &observed) {
        report_(name, reference, observed);
        return *this;
    }

  private:
    friend class diagnostic_reporter_t;
    narration_group_t(diagnostic_reporter_t &owner, std::string prefix) noexcept
        : diagnostic_group_t(owner, diagnostic_verb_t::note_k, std::move(prefix)) {}
};

/**
 *  @brief A `DIVERGENCE` group whose fields print unconditionally and are counted apart from mismatches.
 *
 *  Where a `NOTE` is narration nobody asserts, a divergence is an assertion the caller has already
 *  made: it has established that the two operands differ, that the reference is the one that is wrong,
 *  and that a citation says why. The caller therefore supplies no predicate, exactly as with a note,
 *  but the line is counted -- as a comparison, so `compared=` stays the number of comparisons actually
 *  performed, and as a divergence, so `mismatches=` keeps meaning "differences nobody has explained".
 */
class divergence_group_t final : public diagnostic_group_t {
  public:
    template <typename reference_type_, typename observed_type_>
    divergence_group_t &field(std::string_view name, reference_type_ const &reference, observed_type_ const &observed) {
        count_comparison_();
        count_divergence_();
        report_(name, reference, observed);
        return *this;
    }

  private:
    friend class diagnostic_reporter_t;
    divergence_group_t(diagnostic_reporter_t &owner, std::string prefix) noexcept
        : diagnostic_group_t(owner, diagnostic_verb_t::divergence_k, std::move(prefix)) {}
};

/** What one section closed with. Zero `compared` is a failure, so a caller can gate on it directly. */
struct section_counts_t {
    std::size_t compared = 0, mismatched = 0, diverged = 0, suppressed = 0;
};

/**
 *  @brief Formats diagnostics for one pair of operand roles; it never compares anything itself.
 *
 *  Constructed with the two labels its lines carry -- the reference first, then what is under test --
 *  because a delta is only readable when its sign has a fixed meaning: `delta = observed - reference`.
 *
 *  The counters are the point as much as the lines are.  A validator that silently checks nothing
 *  passes every run, so `summarize` prints what it compared and fails a section that compared zero.
 */
class diagnostic_reporter_t {
  public:
    diagnostic_reporter_t(std::string_view reference_label, std::string_view observed_label)
        : reference_label_(reference_label), observed_label_(observed_label) {}

    /** Open a group that prints one line per field the caller's predicate rejects. */
    template <typename... key_types_>
    comparison_group_t mismatch(std::string_view subject, fmt::format_string<key_types_...> keys,
                                key_types_ &&...key_values) {
        return comparison_group_t(*this, prefix_(subject, keys, std::forward<key_types_>(key_values)...));
    }

    /** Open a group that prints one line per field, differing or not. */
    template <typename... key_types_>
    narration_group_t note(std::string_view subject, fmt::format_string<key_types_...> keys,
                           key_types_ &&...key_values) {
        return narration_group_t(*this, prefix_(subject, keys, std::forward<key_types_>(key_values)...));
    }

    /** Open a group whose every field is a difference the caller has already established and cited. */
    template <typename... key_types_>
    divergence_group_t divergence(std::string_view subject, fmt::format_string<key_types_...> keys,
                                  key_types_ &&...key_values) {
        return divergence_group_t(*this, prefix_(subject, keys, std::forward<key_types_>(key_values)...));
    }

    /**
     *  @brief Close a section with `PASS <subject> compared=… mismatches=… divergences=… suppressed=…`.
     *
     *  Counts run from the previous `summarize`, so one reporter can close several sections in order.
     *  A section that compared nothing fails: it is indistinguishable from a passing one otherwise.
     *  They are returned as well as printed, so a caller gating on its own section does not recompute them;
     *  most callers ignore the result, so this is deliberately not `[[nodiscard]]`.
     */
    section_counts_t summarize(std::string_view subject) {
        std::size_t const compared = comparison_count_ - section_comparison_mark_;
        std::size_t const mismatched = mismatch_count_ - section_mismatch_mark_;
        std::size_t const diverged = divergence_count_ - section_divergence_mark_;
        std::size_t const suppressed = suppressed_line_count_ - section_suppressed_mark_;
        if (compared == 0) ++vacuous_section_count_;
        diagnostic_verb_t const verb = (mismatched != 0 || compared == 0) ? diagnostic_verb_t::fail_k
                                                                          : diagnostic_verb_t::pass_k;
        std::FILE *const stream = diagnostic_stream(verb);
        if (stream == stderr) std::fflush(stdout);
        fmt::println(stream, "{} {} compared={} mismatches={} divergences={} suppressed={}", diagnostic_verb_name(verb),
                     subject, compared, mismatched, diverged, suppressed);
        section_comparison_mark_ = comparison_count_;
        section_mismatch_mark_ = mismatch_count_;
        section_divergence_mark_ = divergence_count_;
        section_suppressed_mark_ = suppressed_line_count_;
        return {.compared = compared, .mismatched = mismatched, .diverged = diverged, .suppressed = suppressed};
    }

    /** Cap printed lines so one pathological molecule cannot bury the log; the rest are still counted. */
    void limit_printed_lines(std::size_t maximum) noexcept { printed_line_budget_ = maximum; }

    std::size_t comparison_count() const noexcept { return comparison_count_; }
    std::size_t mismatch_count() const noexcept { return mismatch_count_; }
    std::size_t divergence_count() const noexcept { return divergence_count_; }
    std::size_t suppressed_line_count() const noexcept { return suppressed_line_count_; }

    /** True when nothing mismatched and every summarized section actually compared something. */
    bool passed() const noexcept { return mismatch_count_ == 0 && vacuous_section_count_ == 0; }

  private:
    friend class diagnostic_group_t;

    template <typename... key_types_>
    std::string prefix_(std::string_view subject, fmt::format_string<key_types_...> keys,
                        key_types_ &&...key_values) const {
        fmt::memory_buffer buffer;
        fmt::format_to(std::back_inserter(buffer), "{} ", subject);
        fmt::format_to(std::back_inserter(buffer), keys, std::forward<key_types_>(key_values)...);
        return std::string(buffer.data(), buffer.size());
    }

    /** Claim one printed line against the budget, counting the line as suppressed when it is spent. */
    bool claim_printed_line_() noexcept {
        if (printed_line_count_ >= printed_line_budget_) {
            ++suppressed_line_count_;
            return false;
        }
        ++printed_line_count_;
        return true;
    }

    std::string reference_label_;
    std::string observed_label_;
    std::size_t comparison_count_ = 0;
    std::size_t mismatch_count_ = 0;
    std::size_t divergence_count_ = 0;
    std::size_t printed_line_count_ = 0;
    std::size_t suppressed_line_count_ = 0;
    std::size_t vacuous_section_count_ = 0;
    std::size_t printed_line_budget_ = std::numeric_limits<std::size_t>::max();
    std::size_t section_comparison_mark_ = 0;
    std::size_t section_mismatch_mark_ = 0;
    std::size_t section_divergence_mark_ = 0;
    std::size_t section_suppressed_mark_ = 0;
};

inline void diagnostic_group_t::count_comparison_() noexcept { ++owner_->comparison_count_; }
inline void diagnostic_group_t::count_mismatch_() noexcept { ++owner_->mismatch_count_; }
inline void diagnostic_group_t::count_divergence_() noexcept { ++owner_->divergence_count_; }

template <typename reference_type_, typename observed_type_>
void diagnostic_group_t::report_(std::string_view name, reference_type_ const &reference,
                                 observed_type_ const &observed) {
    ++reported_field_count_;
    if (!owner_->claim_printed_line_()) return;

    fmt::memory_buffer line;
    auto out = std::back_inserter(line);
    fmt::format_to(out, "{} {} field={} {}=", diagnostic_verb_name(verb_), prefix_, name, owner_->reference_label_);
    append_compared_value(line, reference);
    fmt::format_to(out, " {}=", owner_->observed_label_);
    append_compared_value(line, observed);
    fmt::format_to(out, " delta=");
    append_delta(line, reference, observed);
    fmt::format_to(out, "\n");

    std::FILE *const stream = diagnostic_stream(verb_);
    if (stream == stderr) std::fflush(stdout);
    std::fwrite(line.data(), 1, line.size(), stream);
}

#pragma endregion Reporter

#pragma region Verdict

/**
 *  @brief Close a test with `PASS <test_name>` and answer `main` with its exit code.
 *
 *  Every test prints exactly one verdict, as its last line, and every verdict names the test.  A
 *  thirty-test CI log then answers `grep '^PASS test_'` with the roster that passed and
 *  `grep '^FAIL test_'` with the roster that did not, which a bare `PASS` or an `ALL TESTS PASSED`
 *  cannot do: they say that something passed without saying what.  Anchoring at column zero is the
 *  other half of it, because it is what stops a narration line that merely contains the word --
 *  `Device upload unavailable`, a `[PASS]` table cell -- from answering the same grep.
 *
 *  Section summaries share the verb but never the `test_` prefix: `PASS vdw compared=…` closes a
 *  section, `PASS test_mmff94_tables` closes the process.
 */
inline int report_pass(std::string_view test_name) {
    fmt::println(diagnostic_stream(diagnostic_verb_t::pass_k), "{} {}", diagnostic_verb_name(diagnostic_verb_t::pass_k),
                 test_name);
    return 0;
}

/**
 *  @brief Close a test with `FAIL <test_name>: <reason>` on stderr, and answer `main` with its exit code.
 *
 *  The reason is the one thing the exit code cannot carry, so it states what was expected against what
 *  was seen rather than repeating that the test failed.
 */
template <typename... reason_types_>
inline int report_fail(std::string_view test_name, fmt::format_string<reason_types_...> reason,
                       reason_types_ &&...reason_values) {
    fmt::memory_buffer line;
    auto out = std::back_inserter(line);
    fmt::format_to(out, "{} {}: ", diagnostic_verb_name(diagnostic_verb_t::fail_k), test_name);
    fmt::format_to(out, reason, std::forward<reason_types_>(reason_values)...);
    fmt::format_to(out, "\n");

    std::FILE *const stream = diagnostic_stream(diagnostic_verb_t::fail_k);
    std::fflush(stdout);
    std::fwrite(line.data(), 1, line.size(), stream);
    return 1;
}

#pragma endregion Verdict

} // namespace smallfield::tests

#pragma region Assertions

/**
 *  @brief Fail the test with a message and the source location when @p condition is false.
 *
 *  Carries the `sf_` prefix so it cannot collide with RDKit's own `TEST_ASSERT`, which keeps the
 *  tests free of any framework dependency.
 *
 *  Aborting still spends the test's one verdict, so this prints the same `FAIL <test_name>: <reason>`
 *  a `main` would have returned -- otherwise `grep '^FAIL test_'` would list every test that returned
 *  a failure and silently omit every test that asserted its way out. The name comes from the
 *  `test_name_k` each test declares beside its `main`.
 *
 *  @p message is a reason, not an operand: it says what the code expected, so a path, a SMILES or a
 *  SMARTS belongs inside a sentence rather than standing in for one. It is evaluated only on the
 *  failing branch, so composing it with `fmt::format` costs a passing run nothing.
 */
#define sf_verify_(condition, message)                                                                            \
    do {                                                                                                          \
        if (!(condition)) {                                                                                       \
            [[maybe_unused]] int const exit_code = smallfield::tests::report_fail(test_name_k, "{} at {}:{}",     \
                                                                                  (message), __FILE__, __LINE__); \
            std::exit(1);                                                                                         \
        }                                                                                                         \
    } while (0)

#pragma endregion Assertions
