/**
 * @file detector.cpp
 * @brief ONNX Runtime YOLO11 inference, letterboxing and output decoding.
 */
#include "detector.hpp"

#include "log.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace ft {

const std::vector<std::string>& cocoNames() {
    static const std::vector<std::string> names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    return names;
}

std::optional<std::vector<int>> parseClasses(const std::string& spec) {
    std::string lower = spec;
    std::ranges::transform(lower, lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower.empty() || lower == "all" || lower == "*") return std::nullopt;

    const auto& names = cocoNames();
    std::vector<int> ids;
    std::string token;
    auto handle = [&] {
        auto a = token.find_first_not_of(" \t");
        if (a == std::string::npos) { token.clear(); return; }
        auto b = token.find_last_not_of(" \t");
        std::string t = token.substr(a, b - a + 1);
        token.clear();
        if (std::ranges::all_of(t, [](unsigned char c) { return std::isdigit(c); })) {
            ids.push_back(std::stoi(t));
            return;
        }
        std::ranges::transform(t, t.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        auto it = std::ranges::find(names, t);
        if (it == names.end()) throw std::runtime_error("Unknown class '" + t + "'");
        ids.push_back(static_cast<int>(std::distance(names.begin(), it)));
    };
    for (char c : spec) { if (c == ',') handle(); else token += c; }
    handle();
    std::ranges::sort(ids);
    ids.erase(std::ranges::unique(ids).begin(), ids.end());
    return ids;
}

std::vector<Detection> nonMaxSuppression(std::vector<Detection> dets, float iouThr, std::size_t maxKeep) {
    std::ranges::sort(dets, std::ranges::greater{}, &Detection::score);
    std::vector<Detection> kept;
    std::vector<char> removed(dets.size(), 0);
    for (std::size_t i = 0; i < dets.size() && kept.size() < maxKeep; ++i) {
        if (removed[i]) continue;
        kept.push_back(dets[i]);
        for (std::size_t j = i + 1; j < dets.size(); ++j) {
            if (!removed[j] && dets[j].classId == dets[i].classId &&
                iou(dets[i].box, dets[j].box) > iouThr) {
                removed[j] = 1;
            }
        }
    }
    return kept;
}

/// @brief Letterbox parameters: how a resized image maps back to the original.
namespace {
struct Letterbox {
    cv::Mat image;  ///< Padded square (or imgSize x imgSize) input.
    double scale;   ///< Resize ratio applied to the original.
    int padX;       ///< Left padding in input pixels.
    int padY;       ///< Top padding in input pixels.
};

/// @brief Converts an IEEE-754 half (as uint16 bits) to float; used to decode the FP16 model output.
inline float halfToFloat(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h & 0x7C00u) >> 10;
    const uint32_t mant = h & 0x03FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // +/- zero
        } else {
            int e = -1;    // subnormal: normalise
            uint32_t m = mant;
            do { m <<= 1; ++e; } while ((m & 0x0400u) == 0);
            m &= 0x03FFu;
            bits = sign | ((127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / NaN
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

/// @brief Resizes keeping the aspect ratio and pads to (netW, netH) with grey borders.
Letterbox letterbox(const cv::Mat& bgr, int netW, int netH) {
    const double scale = std::min(netW / static_cast<double>(bgr.cols), netH / static_cast<double>(bgr.rows));
    const int newW = static_cast<int>(std::round(bgr.cols * scale));
    const int newH = static_cast<int>(std::round(bgr.rows * scale));
    cv::Mat resized;
    // INTER_AREA preserves small objects (a far ball) when shrinking; see DOCUMENTATION.md §12.3.
    cv::resize(bgr, resized, {newW, newH}, 0, 0, scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
    const int padX = (netW - newW) / 2;
    const int padY = (netH - newH) / 2;
    cv::Mat out(netH, netW, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(padX, padY, newW, newH)));
    return {out, scale, padX, padY};
}
}  // namespace

YoloDetector::YoloDetector(const std::filesystem::path& modelsDir, const std::string& model, const std::string& device,
                           bool half, int imgSize, float conf, float iou)
    : modelsDir_(modelsDir), model_(model), preferHalf_(half), imgSize_(imgSize), conf_(conf), iou_(iou) {
    std::string dev = device;
    std::ranges::transform(dev, dev.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    int deviceId = 0;
    if (auto pos = dev.find(':'); pos != std::string::npos) {
        deviceId = std::stoi(dev.substr(pos + 1));
        dev = dev.substr(0, pos);
    }

    const bool wantCuda = (dev == "auto" || dev == "cuda");
    useCuda_ = false;
    deviceId_ = deviceId;
    Net main;
    if (wantCuda && createSession(imgSize_, true, deviceId, main)) {
        useCuda_ = true;
    } else {
        if (dev == "cuda") throw std::runtime_error("CUDA requested but the CUDA execution provider is unavailable");
        if (wantCuda) log::warn("CUDA execution provider unavailable -> running on CPU");
        if (!createSession(imgSize_, false, 0, main)) throw std::runtime_error("Failed to create the ONNX Runtime session");
    }
    provider_ = useCuda_ ? "CUDA" : "CPU";
    modelPath_ = main.path;
    log::info("Loaded {} on {} ({})", main.path.filename().string(), provider_, fp16_ ? "FP16" : "FP32");
    nets_.emplace(imgSize_, std::move(main));
}

bool YoloDetector::createSession(int size, bool useCuda, int deviceId, Net& out) const {
    const bool half = useCuda && preferHalf_;
    // Static models are named <model>[_<size>][_fp16].onnx; the main size has no size suffix.
    const std::string sizeTag = size == imgSize_ ? "" : "_" + std::to_string(size);
    auto candidate = [&](bool fp16) { return modelsDir_ / (model_ + sizeTag + (fp16 ? "_fp16" : "") + ".onnx"); };
    std::filesystem::path path = candidate(half);
    bool fp16 = half;
    if (!std::filesystem::exists(path)) { path = candidate(false); fp16 = false; }  // fall back to FP32
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Model not found: " + path.string() + " (run tools/export_static.py)");
    }

    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(0);  // let ORT pick

    if (useCuda) {
        try {
            OrtCUDAProviderOptionsV2* cuda = nullptr;
            Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda));
            const std::string id = std::to_string(deviceId);
            // Static input shapes + EXHAUSTIVE search let ORT pick the fastest conv kernels once
            // (cached per shape), the ONNX Runtime equivalent of PyTorch's cudnn.benchmark.
            const std::vector<const char*> keys{"device_id", "cudnn_conv_algo_search", "arena_extend_strategy",
                                                "cudnn_conv_use_max_workspace"};
            const std::vector<const char*> vals{id.c_str(), "EXHAUSTIVE", "kSameAsRequested", "1"};
            Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cuda, keys.data(), vals.data(), keys.size()));
            options.AppendExecutionProvider_CUDA_V2(*cuda);
            Ort::GetApi().ReleaseCUDAProviderOptions(cuda);
        } catch (const Ort::Exception& e) {
            log::warn("CUDA provider setup failed: {}", e.what());
            return false;
        }
    }

    try {
        out.session = Ort::Session(env_, path.c_str(), options);
    } catch (const Ort::Exception& e) {
        if (useCuda) { log::warn("CUDA session creation failed: {}", e.what()); return false; }
        throw;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    out.inputName = out.session.GetInputNameAllocated(0, allocator).get();
    out.outputName = out.session.GetOutputNameAllocated(0, allocator).get();
    out.path = path;
    fp16_ = fp16;
    return true;
}

YoloDetector::Net& YoloDetector::netFor(int size) const {
    std::scoped_lock lock(netsMutex_);
    auto it = nets_.find(size);
    if (it != nets_.end()) return it->second;
    Net net;
    if (!createSession(size, useCuda_, deviceId_, net)) {
        if (!createSession(size, false, 0, net)) throw std::runtime_error("Cannot create session for size " + std::to_string(size));
    }
    log::info("Loaded {} ({}x{})", net.path.filename().string(), size, size);
    return nets_.emplace(size, std::move(net)).first->second;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& bgr, std::span<const int> classes, int imgSize) const {
    const int size = imgSize > 0 ? imgSize : imgSize_;
    // YOLO11 accepts a square input; letterbox keeps the aspect ratio.
    const Letterbox lb = letterbox(bgr, size, size);

    // Letterboxed BGR uint8 -> RGB, then straight to a planar (CHW) normalised tensor. Thread-local
    // scratch buffers are reused across calls (the detector thread always runs 1280, the main thread
    // 640), so there is no per-frame allocation.
    thread_local cv::Mat rgb;
    cv::cvtColor(lb.image, rgb, cv::COLOR_BGR2RGB);
    const std::array<int64_t, 4> shape{1, 3, size, size};
    const std::size_t count = static_cast<std::size_t>(3) * size * size;
    Net& net = netFor(size);

    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* inNames[] = {net.inputName.c_str()};
    const char* outNames[] = {net.outputName.c_str()};

    // FP16 models have FP16 I/O (ONNX Runtime does not auto-cast). For FP16 we normalise and convert
    // to half in a single OpenCV pass straight into the planar tensor; for FP32 we split to planes.
    thread_local std::vector<Ort::Float16_t> halfBlob;
    thread_local std::vector<float> blob;
    Ort::Value input{nullptr};
    if (fp16_) {
        halfBlob.resize(count);
        std::array<cv::Mat, 3> planes{
            cv::Mat(size, size, CV_16F, halfBlob.data()),
            cv::Mat(size, size, CV_16F, halfBlob.data() + static_cast<std::size_t>(size) * size),
            cv::Mat(size, size, CV_16F, halfBlob.data() + static_cast<std::size_t>(2) * size * size)};
        cv::Mat halfHwc;
        rgb.convertTo(halfHwc, CV_16FC3, 1.0 / 255.0);  // uint8 HWC -> fp16 HWC, normalised
        cv::split(halfHwc, planes.data());               // -> fp16 CHW directly in halfBlob
        input = Ort::Value::CreateTensor(memory, halfBlob.data(), count * sizeof(Ort::Float16_t),
                                         shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    } else {
        blob.resize(count);
        std::array<cv::Mat, 3> planes{
            cv::Mat(size, size, CV_32F, blob.data()),
            cv::Mat(size, size, CV_32F, blob.data() + static_cast<std::size_t>(size) * size),
            cv::Mat(size, size, CV_32F, blob.data() + static_cast<std::size_t>(2) * size * size)};
        cv::Mat floatHwc;
        rgb.convertTo(floatHwc, CV_32FC3, 1.0 / 255.0);
        cv::split(floatHwc, planes.data());
        input = Ort::Value::CreateTensor<float>(memory, blob.data(), count, shape.data(), shape.size());
    }
    const auto _runT0 = std::chrono::steady_clock::now();
    auto outputs = net.session.Run(Ort::RunOptions{nullptr}, inNames, &input, 1, outNames, 1);
    lastRunMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _runT0).count();

    // YOLO11 output: (1, 4 + nClasses, nAnchors). Row layout is [cx, cy, w, h, score_0..score_{C-1}].
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto dims = info.GetShape();
    const int channels = static_cast<int>(dims[1]);
    const int anchors = static_cast<int>(dims[2]);
    const int numClasses = channels - 4;

    // Decode directly from the model output, converting only the values actually read. For FP16 this
    // avoids materialising the full ~2.8M-value output as float (which cost ~8 ms/frame). Scores are
    // scanned first; box coordinates are converted only for anchors that pass the threshold.
    const float* dataF = fp16_ ? nullptr : outputs[0].GetTensorData<float>();
    const uint16_t* dataH = fp16_ ? reinterpret_cast<const uint16_t*>(outputs[0].GetTensorData<Ort::Float16_t>()) : nullptr;
    auto at = [&](int channel, int anchor) -> float {
        const std::size_t i = static_cast<std::size_t>(channel) * anchors + anchor;
        return fp16_ ? halfToFloat(dataH[i]) : dataF[i];
    };

    // Class ids to consider (empty span = all classes).
    std::vector<int> wanted(classes.begin(), classes.end());
    std::vector<Detection> candidates;
    candidates.reserve(256);
    for (int a = 0; a < anchors; ++a) {
        int best = -1;
        float bestScore = conf_;
        if (wanted.empty()) {
            for (int c = 0; c < numClasses; ++c) {
                const float s = at(4 + c, a);
                if (s > bestScore) { bestScore = s; best = c; }
            }
        } else {
            for (int c : wanted) {
                const float s = at(4 + c, a);
                if (s > bestScore) { bestScore = s; best = c; }
            }
        }
        if (best < 0) continue;
        const float cx = at(0, a), cy = at(1, a), w = at(2, a), h = at(3, a);
        // Undo the letterbox: input pixels -> original pixels.
        const float x = static_cast<float>((cx - w * 0.5f - lb.padX) / lb.scale);
        const float y = static_cast<float>((cy - h * 0.5f - lb.padY) / lb.scale);
        candidates.push_back({cv::Rect2f(x, y, static_cast<float>(w / lb.scale), static_cast<float>(h / lb.scale)),
                              bestScore, best});
    }
    return nonMaxSuppression(std::move(candidates), iou_);
}

void YoloDetector::warmup(cv::Size frameSize, int roiSize) {
    // Run each size a few times so EXHAUSTIVE conv-algo selection settles before the first frame.
    const cv::Mat frame(frameSize, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < 3; ++i) detect(frame, {}, imgSize_);
    if (roiSize > 0) {
        const cv::Mat roi(roiSize, roiSize, CV_8UC3, cv::Scalar(0, 0, 0));
        for (int i = 0; i < 3; ++i) detect(roi, {}, roiSize);
    }
}

}  // namespace ft
