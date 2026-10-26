/**
 * @file main.cpp
 * @brief Entry point: parse arguments, prepare the CUDA runtime, run the app.
 */
#include "app.hpp"
#include "config.hpp"
#include "log.hpp"
#include "runtime_env.hpp"

#include <exception>

int main(int argc, char** argv) {
    try {
        auto config = ft::parseArgs(argc, argv);
        if (!config) return 0;  // --help or a parse error already reported
        if (config->verbose) ft::log::minLevel() = ft::log::Level::Debug;

        // Must run before ONNX Runtime touches CUDA (sets CUDA_MODULE_LOADING and PATH).
        ft::prepareCudaRuntime(config->cudaDir);

        ft::TrackingApp app(std::move(*config));
        return app.run();
    } catch (const std::exception& e) {
        ft::log::error("fatal: {}", e.what());
        return 1;
    }
}
