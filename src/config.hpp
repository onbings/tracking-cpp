/**
 * @file config.hpp
 * @brief Run configuration (all command-line options) of the football tracker.
 *
 * The options mirror the Python proof of concept, minus the back-ends that the benchmarks showed
 * to be useless or too slow on this footage (DeepSORT, YOLOv5, ReID, ByteTrack):
 * see docs/PORTING.md.
 */
#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ft {

/**
 * @brief Every tunable of a run. Defaults are the values validated on the 4K drone footage.
 */
struct Config {
    // ---------------------------------------------------------------- input / detection
    std::string source;                 ///< Video file path or webcam index ("0").
    std::string model = "yolo11s";      ///< Detector name; resolved to models/<model>[_fp16].onnx.
    std::filesystem::path modelsDir;    ///< Folder containing the exported ONNX models.
    int imgSize = 1280;                 ///< Inference size (long side, pixels).
    float conf = 0.3f;                  ///< Player confidence (tracker "high" threshold).
    float iou = 0.5f;                   ///< NMS IoU threshold.
    std::string classes = "person";     ///< Tracked COCO classes: names/ids, comma separated, or "all".
    std::string device = "auto";        ///< "auto", "cpu" or "cuda[:N]".
    bool half = true;                   ///< Use the FP16 model on CUDA.
    int startFrame = 0;                 ///< Seek to this frame before starting.
    double fps = 0.0;                   ///< Frame-rate override (0 = use the file's metadata).
    std::filesystem::path cudaDir;      ///< Folder with the CUDA/cuDNN runtime DLLs (auto-detected if empty).

    // ---------------------------------------------------------------- tracking
    int maxAge = 30;                    ///< Frames a lost track is kept before deletion.

    // ---------------------------------------------------------------- football analytics
    bool ball = true;                   ///< Track the ball.
    bool ballRoi = true;                ///< High-resolution ROI search when the ball is lost.
    bool teams = true;                  ///< Team classification by jersey colour.
    double hfovDeg = 70.0;              ///< Horizontal field of view of the camera (degrees).
    double personHeight = 1.4;          ///< Typical person height (m) used for self-calibration.
    double maxDepth = 60.0;             ///< Ground points farther than this (m) are ignored.
    bool voronoi = false;               ///< Start with the space-control overlay enabled.

    // ---------------------------------------------------------------- output / UI
    bool show = true;                   ///< Open a window (false = headless).
    int displayWidth = 1600;            ///< Width of the rendered canvas (window and MP4).
    std::filesystem::path save;         ///< Annotated MP4 output (empty = none).
    std::filesystem::path tracksCsv;    ///< Per-frame tracks CSV (empty = none).
    std::filesystem::path perfJson;     ///< Performance report JSON (empty = none).
    std::filesystem::path statsJson;    ///< Football statistics JSON (empty = none).
    std::filesystem::path outputDir;    ///< Folder for snapshots.
    int maxFrames = 0;                  ///< Stop after N frames (0 = whole video).
    std::optional<std::array<double, 4>> line;  ///< Counting line x1,y1,x2,y2 (normalised), off by default.
    int trailLength = 40;               ///< Points kept per trail.
    int queueSize = 8;                  ///< Decoder read-ahead.
    bool verbose = false;               ///< Debug logging.
};

/**
 * @brief Parses the command line into a Config.
 * @param argc,argv the arguments of main().
 * @return the configuration, or std::nullopt when the program should exit (e.g. `--help`).
 * @throws std::runtime_error on invalid values.
 */
std::optional<Config> parseArgs(int argc, char** argv);

/**
 * @brief Splits "a,b,c" into trimmed, non-empty tokens.
 * @param text comma-separated list.
 * @return the tokens in order.
 */
std::vector<std::string> splitCsv(const std::string& text);

}  // namespace ft
