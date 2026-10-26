/**
 * @file runtime_env.hpp
 * @brief Process-level setup needed before ONNX Runtime touches CUDA.
 *
 * Two problems found with the Python proof of concept are handled here (see DOCUMENTATION.md §7):
 *  1. With an old driver (e.g. 538.xx = CUDA 12.2) and a newer CUDA runtime, *eager* module loading
 *     fails with `cudaErrorDevicesUnavailable`. Setting `CUDA_MODULE_LOADING=LAZY` before the first
 *     CUDA call fixes it.
 *  2. The CUDA execution provider DLL depends on cudart/cuBLAS/cuFFT/cuDNN, which are not installed
 *     system-wide. We locate a folder holding them and prepend it to `PATH` so the Windows loader
 *     finds them when ONNX Runtime loads `onnxruntime_providers_cuda.dll`.
 */
#pragma once

#include <filesystem>
#include <optional>

namespace ft {

/**
 * @brief Prepares environment variables for the CUDA execution provider.
 * @param explicitDir folder given by `--cuda-dir` (may be empty).
 * @return the CUDA runtime folder that was added to PATH, if one was found.
 *
 * Search order: @p explicitDir, `%TRACKER_CUDA_DIR%`, `<exe dir>/cuda`,
 * the compile-time default (third_party/cuda-runtime/bin), the Python PoC's torch/lib folder.
 */
std::optional<std::filesystem::path> prepareCudaRuntime(const std::filesystem::path& explicitDir);

/// @return the folder containing the running executable.
std::filesystem::path executableDir();

}  // namespace ft
