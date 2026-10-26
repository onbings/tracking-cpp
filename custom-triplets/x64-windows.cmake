# Custom x64-windows triplet that pins the Visual Studio 2022 toolset.
#
# vcpkg otherwise auto-selects the newest installed Visual Studio (e.g. VS 2026), whose STL emits
# symbols (such as __std_find_first_not_of_trivial_pos_1) that the VS 2022 runtime we link the
# application against does not provide, causing LNK2019. Pinning both the toolset and the VS path
# here keeps every dependency ABI-compatible with the application, even when CMake re-triggers
# vcpkg during a build. To build everything with VS 2026 instead, delete this triplet and build the
# app from a VS 2026 developer prompt.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_PLATFORM_TOOLSET v143)
set(VCPKG_VISUAL_STUDIO_PATH "C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional")
