# Overlay port: installs Microsoft's prebuilt ONNX Runtime GPU (CUDA 12) package.
#
# The registry port "onnxruntime[cuda]" builds ONNX Runtime from source and requires the CUDA
# toolkit (nvcc) + cuDNN SDK: 1-2 hours of build and an admin install. The official release
# archive already contains the C/C++ headers, the import library and the DLLs (CPU EP built-in,
# CUDA EP as a plug-in DLL), so we simply repackage it.

set(VCPKG_POLICY_DLLS_WITHOUT_LIBS enabled)          # providers_cuda.dll is loaded dynamically
set(VCPKG_POLICY_SKIP_ARCHITECTURE_CHECK enabled)
set(VCPKG_POLICY_ALLOW_RESTRICTED_HEADERS enabled)

set(ORT_PACKAGE "onnxruntime-win-x64-gpu_cuda12-${VERSION}")
vcpkg_download_distfile(ARCHIVE
    URLS "https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${ORT_PACKAGE}.zip"
    FILENAME "${ORT_PACKAGE}.zip"
    SHA512 4dab17566706aef3df9e69a6e6bfb5bf72d05521da9ad351f95f552cc923c8ab423b7f412bbaecc5103b8ab574197be320c04bbb5f4fab5beec3f08b927af68b
)
vcpkg_extract_source_archive(SOURCE_PATH ARCHIVE "${ARCHIVE}" NO_REMOVE_ONE_LEVEL)
set(SRC "${SOURCE_PATH}/${ORT_PACKAGE}")

file(INSTALL "${SRC}/include/" DESTINATION "${CURRENT_PACKAGES_DIR}/include/onnxruntime")

# Release binaries are used for both configurations: the C API is ABI-stable across CRT flavours.
foreach(dir IN ITEMS "" "/debug")
    file(INSTALL "${SRC}/lib/onnxruntime.lib" DESTINATION "${CURRENT_PACKAGES_DIR}${dir}/lib")
    file(INSTALL
        "${SRC}/lib/onnxruntime.dll"
        "${SRC}/lib/onnxruntime_providers_shared.dll"
        "${SRC}/lib/onnxruntime_providers_cuda.dll"
        DESTINATION "${CURRENT_PACKAGES_DIR}${dir}/bin")
endforeach()

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/onnxruntime-gpu-bin-config.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SRC}/LICENSE" "${SRC}/ThirdPartyNotices.txt")
