/**
 * @file log.hpp
 * @brief Minimal thread-safe logger built on `std::format` (no external dependency).
 *
 * Usage:
 * @code
 * ft::log::info("Loaded {} with {} classes", path, n);
 * @endcode
 * Each line is prefixed with a wall-clock timestamp and the level, and written atomically
 * to `std::clog` so lines from different threads never interleave.
 */
#pragma once

#include <chrono>
#include <format>
#include <iostream>
#include <mutex>
#include <string_view>

namespace ft::log {

/// Severity of a log line.
enum class Level { Debug, Info, Warning, Error };

/// @brief Global minimum level; lines below it are discarded.
inline Level& minLevel() {
    static Level level = Level::Info;
    return level;
}

/**
 * @brief Writes one formatted line (implementation detail of the helpers below).
 * @param level severity of the message.
 * @param text  already formatted message.
 */
inline void write(Level level, std::string_view text) {
    if (level < minLevel()) return;
    static std::mutex mutex;
    static constexpr std::string_view names[] = {"DEBUG", "INFO ", "WARN ", "ERROR"};
    const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    const auto local = std::chrono::current_zone()->to_local(now);
    std::scoped_lock lock(mutex);
    std::clog << std::format("{:%H:%M:%S} {} {}\n", local, names[static_cast<int>(level)], text);
}

/// @brief Debug-level message with `std::format` syntax.
template <class... Args>
void debug(std::format_string<Args...> fmt, Args&&... args) {
    write(Level::Debug, std::format(fmt, std::forward<Args>(args)...));
}
/// @brief Info-level message with `std::format` syntax.
template <class... Args>
void info(std::format_string<Args...> fmt, Args&&... args) {
    write(Level::Info, std::format(fmt, std::forward<Args>(args)...));
}
/// @brief Warning-level message with `std::format` syntax.
template <class... Args>
void warn(std::format_string<Args...> fmt, Args&&... args) {
    write(Level::Warning, std::format(fmt, std::forward<Args>(args)...));
}
/// @brief Error-level message with `std::format` syntax.
template <class... Args>
void error(std::format_string<Args...> fmt, Args&&... args) {
    write(Level::Error, std::format(fmt, std::forward<Args>(args)...));
}

}  // namespace ft::log
