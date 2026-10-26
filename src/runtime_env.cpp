/**
 * @file runtime_env.cpp
 * @brief Implementation of the CUDA runtime discovery (Windows-specific parts are guarded).
 */
#include "runtime_env.hpp"

#include "log.hpp"

#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#ifndef TRACKER_DEFAULT_CUDA_DIR
#define TRACKER_DEFAULT_CUDA_DIR ""
#endif

namespace ft {

std::filesystem::path executableDir() {
#ifdef _WIN32
    std::wstring buffer(MAX_PATH, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    buffer.resize(n);
    return std::filesystem::path(buffer).parent_path();
#else
    return std::filesystem::current_path();
#endif
}

/// @brief Reads an environment variable portably (empty string if missing).
static std::string getEnv(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t len = 0;
    std::string out;
    if (_dupenv_s(&value, &len, name) == 0 && value != nullptr) {
        out = value;
        std::free(value);
    }
    return out;
#else
    const char* v = std::getenv(name);
    return v ? v : "";
#endif
}

/// @brief Sets an environment variable for this process (inherited by DLLs loaded later).
static void setEnv(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

std::optional<std::filesystem::path> prepareCudaRuntime(const std::filesystem::path& explicitDir) {
    // Must happen before the first CUDA call of the process: keep a user-provided value.
    if (getEnv("CUDA_MODULE_LOADING").empty()) setEnv("CUDA_MODULE_LOADING", "LAZY");

    std::vector<std::filesystem::path> candidates;
    if (!explicitDir.empty()) candidates.push_back(explicitDir);
    if (auto env = getEnv("TRACKER_CUDA_DIR"); !env.empty()) candidates.emplace_back(env);
    candidates.push_back(executableDir() / "cuda");
    if (std::string def = TRACKER_DEFAULT_CUDA_DIR; !def.empty()) candidates.emplace_back(def);
    candidates.emplace_back("C:/pro/tracking/.venv/Lib/site-packages/torch/lib");  // PoC venv (CUDA 12.6: may be too old)

    for (const auto& dir : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(dir / "cudart64_12.dll", ec) && std::filesystem::exists(dir / "cudnn64_9.dll", ec)) {
            const auto absolute = std::filesystem::absolute(dir);
            setEnv("PATH", absolute.string() + ";" + getEnv("PATH"));
            log::info("CUDA runtime DLLs: {}", absolute.string());
            return absolute;
        }
    }
    log::warn("No CUDA 12/cuDNN 9 runtime folder found (run tools/fetch_cuda_runtime.ps1 or use --cuda-dir); "
              "the CUDA execution provider will probably be unavailable");
    return std::nullopt;
}

}  // namespace ft
