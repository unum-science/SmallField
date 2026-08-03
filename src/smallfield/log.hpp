/**
 *  @file log.hpp
 *  @brief Leveled, colored, timestamped logger used across library, CLI, tests, benches.
 *
 *  `logger_t` is a plain struct owned by the object that needs it — `context_t`
 *  carries one, `smallfield_batch` owns its own for shard progress, each bench
 *  constructs a local one.  No process-wide singletons, no thread_locals.
 *
 *  Formatting is layered on `{fmt}`: `fmt::format`, `fmt::print`, and
 *  `fmt::text_style` for color.  The built-in `fmt::print(stream, style, …)`
 *  auto-disables ANSI escapes when the stream is not a terminal, and the logger
 *  further gates on `NO_COLOR` via @ref detect_color_enabled.
 *
 *  @code{.cpp}
 *  smallfield::logger_t log;
 *  log.info("loaded {} smiles ({} mols/s)", count, rate);
 *  log.verbose_level = 1;
 *  log.trace("wave {}/{}", i, n);           // emits only when verbose_level >= 1
 *  @endcode
 */

#pragma once

#include <chrono>
#include <concepts>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <locale>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>

#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/format.h>

namespace smallfield {

/**
 *  @brief Returns a locale with thousands grouping for use with `"{:L}"`.
 *
 *  Constructed lazily on first call; defaults to `en_US.UTF-8` (comma grouping),
 *  falling back to `std::locale::classic()` if that locale is unavailable.
 */
inline std::locale const &grouping_locale() {
    static std::locale const loc = []() {
        try {
            return std::locale("en_US.UTF-8");
        }
        catch (std::runtime_error const &) {
            return std::locale::classic();
        }
    }();
    return loc;
}

/** @brief `1234567 → "1,234,567"`. Uses @ref grouping_locale. */
template <typename integer_>
    requires std::integral<integer_>
inline std::string thousands(integer_ v) {
    if constexpr (std::is_signed_v<integer_>)
        return fmt::format(grouping_locale(), "{:L}", static_cast<std::int64_t>(v));
    else return fmt::format(grouping_locale(), "{:L}", static_cast<std::uint64_t>(v));
}

/** @brief Wall-clock `time_point` formatted as "YYYY-MM-DD HH:MM:SS" in local time. */
inline std::string format_wall_time(std::chrono::system_clock::time_point const &tp) {
    std::time_t const seconds_since_epoch = std::chrono::system_clock::to_time_t(tp);
    std::tm tm_local {};
    ::localtime_r(&seconds_since_epoch, &tm_local);
    return fmt::format("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", tm_local.tm_year + 1900, tm_local.tm_mon + 1,
                       tm_local.tm_mday, tm_local.tm_hour, tm_local.tm_min, tm_local.tm_sec);
}

/** @brief Elapsed `duration` formatted as "HH:MM:SS" for pairing with wall clock. */
inline std::string format_hms(std::chrono::steady_clock::duration const &d) {
    auto const total_seconds = std::chrono::duration_cast<std::chrono::seconds>(d).count();
    return fmt::format("{:02}:{:02}:{:02}", total_seconds / 3600, (total_seconds / 60) % 60, total_seconds % 60);
}

/**
 *  @brief True when ANSI styling should be emitted to stdout.
 *
 *  Requires stdout to be a terminal and `NO_COLOR` to be unset (cf.
 *  https://no-color.org/).  Cached after the first call.
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
 *  @brief Owned logger: one instance per logical caller (context, CLI, bench).
 *
 *  Every field is public so callers can tune the logger after construction —
 *  e.g. `ctx.logger.verbose_level = 2` before `generate_conformers`.
 */
struct logger_t {
    int verbose_level = 0; ///< 0 silent, 1 summary, 2 per-device detail
    std::FILE *info_stream = stdout;
    std::FILE *warn_stream = stderr;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    bool color_enabled = detect_color_enabled();

    /** @brief `"[2026-04-18 10:17:15 | +00:13:27]"` — wall clock plus elapsed. */
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

    /** @brief Emit only when @ref verbose_level >= 1 (wave-level summary tier). */
    template <typename... args_types_>
    void trace(fmt::format_string<args_types_...> fmt_string, args_types_ &&...args) const {
        if (verbose_level < 1) return;
        write_(info_stream, color_enabled ? fmt::fg(fmt::color::cyan) : fmt::text_style {}, fmt_string,
               std::forward<args_types_>(args)...);
    }

    /** @brief Emit only when @ref verbose_level >= 2 (per-device detail tier). */
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
        // `fmt::print(stream, style, fmt, args…)` with an empty style emits no
        // ANSI escapes; no need to branch.
        fmt::print(stream, "{} ", timestamp_prefix());
        fmt::print(stream, style, fmt_string, std::forward<args_types_>(args)...);
        std::fputc('\n', stream);
        // `fflush` here so redirected-to-file log tails are live and long-running
        // jobs expose progress mid-shard.  Cost is one syscall per line; with
        // hundreds of lines over a multi-hour run this is negligible.
        std::fflush(stream);
    }
};

} // namespace smallfield
