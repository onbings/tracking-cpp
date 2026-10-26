/**
 * @file config.cpp
 * @brief Command-line parsing with CLI11.
 */
#include "config.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>

#ifndef TRACKER_SOURCE_DIR
#define TRACKER_SOURCE_DIR "."
#endif
#ifndef TRACKER_DEFAULT_VIDEO
#define TRACKER_DEFAULT_VIDEO "C:/media/14369070-uhd_3840_2160_30fps.mp4"
#endif

namespace ft {

std::vector<std::string> splitCsv(const std::string& text) {
    std::vector<std::string> out;
    std::string token;
    auto flush = [&] {
        auto first = token.find_first_not_of(" \t");
        auto last = token.find_last_not_of(" \t");
        if (first != std::string::npos) out.push_back(token.substr(first, last - first + 1));
        token.clear();
    };
    for (char c : text) {
        if (c == ',') flush(); else token += c;
    }
    flush();
    return out;
}

/// @brief Parses "x1,y1,x2,y2" (normalised, 0..1) or "off".
static std::optional<std::array<double, 4>> parseLine(const std::string& text) {
    std::string lower = text;
    std::ranges::transform(lower, lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower.empty() || lower == "off" || lower == "none") return std::nullopt;
    const auto parts = splitCsv(text);
    if (parts.size() != 4) throw std::runtime_error("--line expects x1,y1,x2,y2 (normalised) or 'off'");
    std::array<double, 4> v{};
    for (std::size_t i = 0; i < 4; ++i) {
        v[i] = std::stod(parts[i]);
        if (v[i] < 0.0 || v[i] > 1.0) throw std::runtime_error("--line values must be in [0, 1]");
    }
    return v;
}

std::optional<Config> parseArgs(int argc, char** argv) {
    Config c;
    c.source = TRACKER_DEFAULT_VIDEO;
    c.modelsDir = std::filesystem::path(TRACKER_SOURCE_DIR) / "models";
    c.outputDir = std::filesystem::path(TRACKER_SOURCE_DIR) / "output";

    CLI::App app{"Football tracking: OpenCV -> YOLO11 (ONNX Runtime) -> BoT-SORT + teams, speed, ball, space control."};
    app.get_formatter()->column_width(34);

    bool noHalf = false, noBall = false, noBallRoi = false, noTeams = false, noShow = false;
    std::string line = "off", save, tracksCsv, perfJson, statsJson, modelsDir, cudaDir;

    auto* io = "input / detection";
    app.add_option("--source", c.source, "Video file path or webcam index (e.g. 0)")->group(io)->capture_default_str();
    app.add_option("--model", c.model, "Detector: yolo11n, yolo11s, yolo11m (models/<name>[_fp16].onnx)")->group(io)->capture_default_str();
    app.add_option("--models-dir", modelsDir, "Folder with the exported ONNX models")->group(io);
    app.add_option("--img-size", c.imgSize, "Inference size (long side, pixels)")->group(io)->capture_default_str()->check(CLI::Range(160, 4096));
    app.add_option("--conf", c.conf, "Player confidence threshold (tracker high threshold)")->group(io)->capture_default_str()->check(CLI::Range(0.01, 0.99));
    app.add_option("--iou", c.iou, "NMS IoU threshold")->group(io)->capture_default_str()->check(CLI::Range(0.05, 0.95));
    app.add_option("--classes", c.classes, "Tracked COCO classes (names or ids, comma separated) or 'all'")->group(io)->capture_default_str();
    app.add_option("--device", c.device, "'auto', 'cpu' or 'cuda[:N]'")->group(io)->capture_default_str();
    app.add_flag("--no-half", noHalf, "Use the FP32 model on CUDA")->group(io);
    app.add_option("--start-frame", c.startFrame, "Seek to this frame before starting")->group(io)->capture_default_str();
    app.add_option("--fps", c.fps, "Override the source frame rate (0 = read it from the file); all speeds scale with it")
        ->group(io)->capture_default_str()->check(CLI::Range(0.0, 240.0));
    app.add_option("--cuda-dir", cudaDir, "Folder with the CUDA 12.8+/cuDNN 9 runtime DLLs (auto-detected if omitted)")->group(io);

    auto* tr = "tracking";
    app.add_option("--max-age", c.maxAge, "Frames before a lost track is deleted")->group(tr)->capture_default_str();

    auto* fb = "football analytics";
    app.add_flag("--no-ball", noBall, "Disable ball tracking")->group(fb);
    app.add_flag("--no-ball-roi", noBallRoi, "Disable the high-resolution ROI search for a lost ball")->group(fb);
    app.add_flag("--no-teams", noTeams, "Disable team classification")->group(fb);
    app.add_option("--hfov", c.hfovDeg, "Camera horizontal field of view (deg)")->group(fb)->capture_default_str()->check(CLI::Range(20.0, 150.0));
    app.add_option("--person-height", c.personHeight, "Typical person height (m) for self-calibration")->group(fb)->capture_default_str()->check(CLI::Range(0.8, 2.2));
    app.add_option("--max-depth", c.maxDepth, "Ignore ground points farther than this (m)")->group(fb)->capture_default_str();
    app.add_flag("--voronoi", c.voronoi, "Start with the space-control (Voronoi) overlay on")->group(fb);

    auto* out = "output / UI";
    app.add_flag("--no-show", noShow, "Headless mode (no window)")->group(out);
    app.add_option("--display-width", c.displayWidth, "Width of the rendered/saved canvas")->group(out)->capture_default_str();
    app.add_option("--save", save, "Save the annotated video to this .mp4 path")->group(out);
    app.add_option("--tracks-csv", tracksCsv, "Export per-frame tracks to CSV")->group(out);
    app.add_option("--perf-json", perfJson, "Write a JSON performance report")->group(out);
    app.add_option("--stats-json", statsJson, "Write player/team/ball statistics to JSON")->group(out);
    app.add_option("--max-frames", c.maxFrames, "Stop after N frames (0 = all)")->group(out)->capture_default_str();
    app.add_option("--line", line, "Counting line x1,y1,x2,y2 (normalised) or 'off'")->group(out)->capture_default_str();
    app.add_option("--trail-len", c.trailLength, "Points kept per track trail")->group(out)->capture_default_str();
    app.add_option("--queue-size", c.queueSize, "Decoder read-ahead queue size")->group(out)->capture_default_str();
    app.add_flag("-v,--verbose", c.verbose, "Debug logging")->group(out);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        app.exit(e);
        return std::nullopt;
    }

    // Relative output paths are resolved against the project folder, like the Python version.
    auto resolve = [](const std::string& p) -> std::filesystem::path {
        if (p.empty()) return {};
        std::filesystem::path path(p);
        return path.is_absolute() ? path : std::filesystem::path(TRACKER_SOURCE_DIR) / path;
    };
    c.half = !noHalf;
    c.ball = !noBall;
    c.ballRoi = !noBallRoi;
    c.teams = !noTeams;
    c.show = !noShow;
    c.save = resolve(save);
    c.tracksCsv = resolve(tracksCsv);
    c.perfJson = resolve(perfJson);
    c.statsJson = resolve(statsJson);
    if (!modelsDir.empty()) c.modelsDir = modelsDir;
    if (!cudaDir.empty()) c.cudaDir = cudaDir;
    c.line = parseLine(line);
    return c;
}

}  // namespace ft
