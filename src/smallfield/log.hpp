/**
 *  @file src/smallfield/log.hpp
 *  @brief Leveled, colored, timestamped logger used across library, CLI, tests, benches.
 *
 *  `logger_t` is a plain struct owned by the object that needs it — `context_t` carries one, `smallfield_batch`
 *  owns its own for shard progress, each bench constructs a local one. No process-wide singletons, no
 *  thread_locals.
 *
 *  Formatting is layered on `{fmt}`: `fmt::format`, `fmt::print`, and `fmt::text_style` for color. The built-in
 *  `fmt::print(stream, style, ...)` auto-disables ANSI escapes when the stream is not a terminal, and the logger
 *  further gates on `NO_COLOR` via `detect_color_enabled`.
 *
 *  @code{.cpp}
 *  smallfield::logger_t log;
 *  log.info("loaded {} smiles ({} mols/s)", count, rate);
 *  log.verbose_level = 1;
 *  log.trace("wave {}/{}", wave_index, num_waves);   // emits only when verbose_level >= 1
 *  @endcode
 */

#pragma once

#include <cstdio>  // `stdout`, `stderr`
#include <cstdlib> // `std::getenv`
#include <ctime>

#include <chrono> // `std::chrono::system_clock`, `std::chrono::steady_clock`
#include <concepts>
#include <locale>
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`
#include <utility>   // `std::forward`

#include <unistd.h>
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/format.h>

namespace smallfield {

/**
 *  @brief Returns a locale with thousands grouping for use with `"{:L}"`.
 *
 *  Constructed lazily on first call; defaults to `en_US.UTF-8` with comma grouping, falling back to
 *  `std::locale::classic()` if that locale is unavailable.
 */
inline std::locale const &grouping_locale() {
    static std::locale const locale = []() {
        try {
            return std::locale("en_US.UTF-8");
        }
        catch (std::runtime_error const &) {
            return std::locale::classic();
        }
    }();
    return locale;
}

/** Formats `1234567` as `"1,234,567"`. Uses `grouping_locale`. */
template <typename integer_>
    requires std::integral<integer_>
inline std::string thousands(integer_ value) {
    if constexpr (std::is_signed_v<integer_>)
        return fmt::format(grouping_locale(), "{:L}", static_cast<std::int64_t>(value));
    else return fmt::format(grouping_locale(), "{:L}", static_cast<std::uint64_t>(value));
}

/** Wall-clock `time_point` formatted as "YYYY-MM-DD HH:MM:SS" in local time. */
inline std::string format_wall_time(std::chrono::system_clock::time_point const &time_point) {
    std::time_t const seconds_since_epoch = std::chrono::system_clock::to_time_t(time_point);
    std::tm tm_local {};
    ::localtime_r(&seconds_since_epoch, &tm_local);
    return fmt::format("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", tm_local.tm_year + 1900, tm_local.tm_mon + 1,
                       tm_local.tm_mday, tm_local.tm_hour, tm_local.tm_min, tm_local.tm_sec);
}

/** Elapsed `duration` formatted as "HH:MM:SS" for pairing with wall clock. */
inline std::string format_hms(std::chrono::steady_clock::duration const &duration) {
    auto const total_seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    return fmt::format("{:02}:{:02}:{:02}", total_seconds / 3600, (total_seconds / 60) % 60, total_seconds % 60);
}

/**
 *  @brief True when ANSI styling should be emitted to stdout.
 *
 *  Requires stdout to be a terminal and `NO_COLOR` to be unset, per https://no-color.org/. Cached after the
 *  first call.
 */
inline bool detect_color_enabled() {
    static bool const enabled = []() {
        char const *no_color = std::getenv("NO_COLOR");
        if (no_color != nullptr && no_color[0] != '\0') return false;
        return ::isatty(fileno(stdout)) != 0;
    }();
    return enabled;
}

/**
 *  @brief Owned logger: one instance per logical caller, be it a context, a CLI, or a bench.
 *
 *  Every field is public so callers can tune the logger after construction, such as
 *  `context.logger.verbose_level = 2` before `generate_conformers`.
 */
struct logger_t {
    /** 0 silent, 1 summary, 2 per-device detail. */
    int verbose_level = 0;
    std::FILE *info_stream = stdout;
    std::FILE *warn_stream = stderr;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    bool color_enabled = detect_color_enabled();

    /** `"[2026-04-18 10:17:15 | +00:13:27]"` — wall clock plus elapsed. */
    std::string timestamp_prefix() const {
        return fmt::format("[{} | +{}]", format_wall_time(std::chrono::system_clock::now()),
                           format_hms(std::chrono::steady_clock::now() - start_time));
    }

    template <typename... args_types_>
    void info(fmt::format_string<args_types_...> fmt_string, args_types_ &&...args) const {
        write_(info_stream, fmt::text_style {}, fmt_string, std::forward<args_types_>(args)...);
    }

    template <typename... args_types_>
    void warn(fmt::format_string<args_types_...> fmt_string, args_types_ &&...args) const {
        write_(warn_stream, color_enabled ? fmt::fg(fmt::color::yellow) : fmt::text_style {}, fmt_string,
               std::forward<args_types_>(args)...);
    }

    template <typename... args_types_>
    void error(fmt::format_string<args_types_...> fmt_string, args_types_ &&...args) const {
        write_(warn_stream, color_enabled ? (fmt::fg(fmt::color::red) | fmt::emphasis::bold) : fmt::text_style {},
               fmt_string, std::forward<args_types_>(args)...);
    }

    /** Emit only when `verbose_level` >= 1, the wave-level summary tier. */
    template <typename... args_types_>
    void trace(fmt::format_string<args_types_...> fmt_string, args_types_ &&...args) const {
        if (verbose_level < 1) return;
        write_(info_stream, color_enabled ? fmt::fg(fmt::color::cyan) : fmt::text_style {}, fmt_string,
               std::forward<args_types_>(args)...);
    }

    /** Emit only when `verbose_level` >= 2, the per-device detail tier. */
    template <typename... args_types_>
    void trace2(fmt::format_string<args_types_...> fmt_string, args_types_ &&...args) const {
        if (verbose_level < 2) return;
        write_(info_stream, color_enabled ? fmt::fg(fmt::color::dark_gray) : fmt::text_style {}, fmt_string,
               std::forward<args_types_>(args)...);
    }

  private:
    template <typename... args_types_>
    void write_(std::FILE *stream, fmt::text_style style, fmt::format_string<args_types_...> fmt_string,
                args_types_ &&...args) const {
        // `fmt::print(stream, style, fmt, args...)` with an empty style emits no ANSI escapes; no need to branch.
        fmt::print(stream, "{} ", timestamp_prefix());
        fmt::print(stream, style, fmt_string, std::forward<args_types_>(args)...);
        std::fputc('\n', stream);
        // `fflush` here so redirected-to-file log tails are live and long-running jobs expose progress
        // mid-shard. Cost is one syscall per line; with hundreds of lines over a multi-hour run this is
        // negligible.
        std::fflush(stream);
    }
};

} // namespace smallfield
