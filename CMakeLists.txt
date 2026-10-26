cmake_minimum_required(VERSION 3.25)
project(tracking_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8 /W4 /permissive- /Zc:__cplusplus /EHsc)
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui video calib3d)
find_package(onnxruntime-gpu-bin CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Paths baked into the binary so it finds its models/output without a working-directory assumption.
set(TRACKER_DEFAULT_VIDEO "C:/media/14369070-uhd_3840_2160_30fps.mp4" CACHE STRING "Default input video")
set(TRACKER_DEFAULT_CUDA_DIR "${CMAKE_SOURCE_DIR}/third_party/cuda-runtime/bin" CACHE PATH "CUDA runtime DLL folder")

# --- core library (everything except main, so the tests can link it) ---
add_library(tracker_core STATIC
    src/config.cpp
    src/runtime_env.cpp
    src/video_reader.cpp
    src/detector.cpp
    src/camera_motion.cpp
    src/geometry.cpp
    src/tracker.cpp
    src/kinematics.cpp
    src/team.cpp
    src/ball.cpp
    src/space.cpp
    src/analytics.cpp
    src/perf.cpp
    src/overlay.cpp
    src/app.cpp)
target_include_directories(tracker_core PUBLIC src)
target_compile_definitions(tracker_core PUBLIC
    TRACKER_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    TRACKER_DEFAULT_VIDEO="${TRACKER_DEFAULT_VIDEO}"
    TRACKER_DEFAULT_CUDA_DIR="${TRACKER_DEFAULT_CUDA_DIR}")
target_link_libraries(tracker_core PUBLIC
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
    CLI11::CLI11
    nlohmann_json::nlohmann_json)

# --- executable ---
add_executable(tracker src/main.cpp)
target_link_libraries(tracker PRIVATE tracker_core)

# --- detector micro-benchmark (optional) ---
add_executable(bench_detect tools/bench_detect.cpp)
target_link_libraries(bench_detect PRIVATE tracker_core)

# Copy the ONNX Runtime DLLs next to the executable (the CUDA provider DLL is loaded at runtime).
add_custom_command(TARGET tracker POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_RUNTIME_DLLS} $<TARGET_FILE_DIR:tracker>
    COMMAND_EXPAND_LISTS)

# --- tests (feature "tests") ---
find_package(GTest CONFIG QUIET)
if(GTest_FOUND)
    enable_testing()
    add_executable(tracker_tests
        tests/test_geometry.cpp
        tests/test_tracker.cpp
        tests/test_kinematics.cpp
        tests/test_team.cpp
        tests/test_ball.cpp
        tests/test_space.cpp
        tests/test_misc.cpp
        tests/test_robustness.cpp)
    target_link_libraries(tracker_tests PRIVATE tracker_core GTest::gtest GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(tracker_tests)
endif()
