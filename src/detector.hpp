/**
 * @file detector.hpp
 * @brief YOLO11 object detector running an exported ONNX model with ONNX Runtime.
 *
 * Replaces the ultralytics/torch.hub detectors of the Python PoC. Lessons carried over
 * (DOCUMENTATION.md §12), with C++-specific tuning:
 *  - the frame is downscaled by us (INTER_AREA) before inference: letterboxing a 4K frame inside
 *    the inference library was a hidden ~20 ms CPU cost;
 *  - static-shape ONNX models (one session per input size: 1280 full frame, 640 ball ROI) let
 *    ONNX Runtime fully plan the CUDA graph; dynamic shapes were ~40% slower here;
 *  - cuDNN convolution algorithm search is EXHAUSTIVE: with static shapes the fastest kernel is
 *    found once during warm-up and cached (the ONNX Runtime analogue of torch's cudnn.benchmark).
 */
#pragma once

#include "common.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ft {

/// @return the 80 COCO class names, indexed by class id.
const std::vector<std::string>& cocoNames();

/**
 * @brief Converts "person,sports ball" / "0,32" / "all" into class ids.
 * @return the sorted ids, or std::nullopt for "all classes".
 * @throws std::runtime_error for an unknown class name.
 */
std::optional<std::vector<int>> parseClasses(const std::string& spec);

/**
 * @brief Single-image YOLO11 detector (thread-safe: detect() may be called concurrently).
 */
class YoloDetector {
public:
    /**
     * @brief Loads `<modelsDir>/<model>[_fp16].onnx` on the requested device.
     * @param modelsDir folder of the exported models (tools/export_models.py).
     * @param model     model name, e.g. "yolo11s".
     * @param device    "auto", "cpu" or "cuda[:N]"; "auto" falls back to CPU if CUDA fails.
     * @param half      prefer the FP16 model when running on CUDA.
     * @param imgSize   default inference size (long side).
     * @param conf      minimum confidence kept by the decoder.
     * @param iou       NMS IoU threshold.
     * @throws std::runtime_error / Ort::Exception if the model cannot be loaded.
     */
    YoloDetector(const std::filesystem::path& modelsDir, const std::string& model, const std::string& device,
                 bool half, int imgSize, float conf, float iou);

    /**
     * @brief Detects objects in a BGR image.
     * @param bgr     image of any size (full frame or crop).
     * @param classes class ids to keep (empty = all classes).
     * @param imgSize inference size override (0 = default).
     * @return detections in the pixel coordinates of @p bgr, after class-aware NMS.
     */
    [[nodiscard]] std::vector<Detection> detect(const cv::Mat& bgr, std::span<const int> classes, int imgSize = 0) const;

    /**
     * @brief Runs a few inferences so that CUDA/cuDNN initialisation does not hit the first frames.
     * @param frameSize size of the video frames.
     * @param roiSize   extra square input size to warm up (0 = none), e.g. the ball ROI.
     */
    void warmup(cv::Size frameSize, int roiSize);

    [[nodiscard]] const std::string& provider() const { return provider_; }  ///< "CUDA" or "CPU".
    [[nodiscard]] bool fp16() const { return fp16_; }                          ///< FP16 model loaded.
    [[nodiscard]] int imgSize() const { return imgSize_; }                     ///< Default input size.
    [[nodiscard]] const std::filesystem::path& modelPath() const { return modelPath_; }  ///< Loaded file.
    [[nodiscard]] double lastRunMs() const { return lastRunMs_; }  ///< Pure session.Run() time of the last detect().

private:
    /// @brief A loaded static-shape session plus its I/O names.
    struct Net {
        Ort::Session session{nullptr};
        std::string inputName, outputName;
        std::filesystem::path path;
    };

    /// @brief Creates one static session for @p size on the chosen device; returns false on failure.
    bool createSession(int size, bool useCuda, int deviceId, Net& out) const;
    /// @brief Returns (creating if needed) the session for @p size.
    Net& netFor(int size) const;

    mutable Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "tracker"};
    std::filesystem::path modelsDir_;
    std::string model_;
    std::filesystem::path modelPath_;
    std::string provider_ = "CPU";
    mutable bool fp16_ = false;
    bool preferHalf_ = true;
    bool useCuda_ = false;
    int deviceId_ = 0;
    int imgSize_;
    float conf_, iou_;
    mutable std::mutex netsMutex_;
    mutable std::map<int, Net> nets_;  ///< One static session per input size (1280 main, 640 ROI).
    mutable double lastRunMs_ = 0.0;   ///< Timing probe (last session.Run duration).
};

/**
 * @brief Greedy non-maximum suppression (class-aware when boxes carry different classes).
 * @param dets     candidate detections (any order).
 * @param iouThr   boxes overlapping a better one by more than this are removed.
 * @param maxKeep  maximum number of results.
 * @return the kept detections, best score first.
 */
std::vector<Detection> nonMaxSuppression(std::vector<Detection> dets, float iouThr, std::size_t maxKeep = 300);

}  // namespace ft
