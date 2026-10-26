# CMake package config for the prebuilt ONNX Runtime GPU binaries.
# Provides the imported target onnxruntime::onnxruntime and the variable
# ONNXRUNTIME_RUNTIME_DLLS (all DLLs to copy next to the executable).

get_filename_component(_ort_prefix "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_ort_prefix}/include/onnxruntime"
        IMPORTED_IMPLIB "${_ort_prefix}/lib/onnxruntime.lib"
        IMPORTED_LOCATION "${_ort_prefix}/bin/onnxruntime.dll"
        IMPORTED_IMPLIB_DEBUG "${_ort_prefix}/debug/lib/onnxruntime.lib"
        IMPORTED_LOCATION_DEBUG "${_ort_prefix}/debug/bin/onnxruntime.dll"
        IMPORTED_CONFIGURATIONS "RELEASE;DEBUG")
endif()

set(ONNXRUNTIME_RUNTIME_DLLS
    "${_ort_prefix}/bin/onnxruntime.dll"
    "${_ort_prefix}/bin/onnxruntime_providers_shared.dll"
    "${_ort_prefix}/bin/onnxruntime_providers_cuda.dll")
set(onnxruntime-gpu-bin_FOUND TRUE)
unset(_ort_prefix)
