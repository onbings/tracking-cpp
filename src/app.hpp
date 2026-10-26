/**
 * @file app.hpp
 * @brief TrackingApp: wires every module together and runs the processing loop.
 */
#pragma once

#include "config.hpp"

#include <memory>

namespace ft {

/**
 * @brief Runs the full football-tracking pipeline for one configuration.
 */
class TrackingApp {
public:
    /// @brief Builds every module from the configuration (loads the detector, opens the video).
    explicit TrackingApp(Config config);
    ~TrackingApp();

    /// @brief Runs until end of stream, `--max-frames`, or the user quits. @return process exit code.
    int run();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ft
