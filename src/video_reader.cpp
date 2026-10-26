/**
 * @file video_reader.cpp
 * @brief Implementation of the threaded decoder.
 */
#include "video_reader.hpp"

#include "log.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ft {

VideoReader::VideoReader(const std::string& source, int queueSize, int startFrame) : queue_(static_cast<std::size_t>(queueSize)) {
    const bool isCamera = !source.empty() && source.find_first_not_of("0123456789") == std::string::npos;
    if (isCamera) {
        capture_.open(std::stoi(source));
    } else {
        // Ask Media Foundation for GPU decoding first; fall back to whatever OpenCV has.
        const std::vector<int> params{cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY};
        if (!capture_.open(source, cv::CAP_MSMF, params)) capture_.open(source, cv::CAP_ANY);
    }
    if (!capture_.isOpened()) throw std::runtime_error("Cannot open video source: " + source);

    width_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps = capture_.get(cv::CAP_PROP_FPS);
    if (std::isfinite(fps) && fps >= kMinFps && fps <= kMaxFps) {
        fps_ = fps;
    } else {
        // Webcams and some remuxed/streamed files report 0, 1000 or NaN; speeds would be nonsense.
        fps_ = 30.0;
        fpsSource_ = "fallback";
        log::warn("Source reports an implausible frame rate ({}); assuming 30 fps. Use --fps to set it.", fps);
    }
    frameCount_ = std::max(0, static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_COUNT)));
    backend_ = capture_.getBackendName();
    hwAccel_ = capture_.get(cv::CAP_PROP_HW_ACCELERATION) > 0;
    if (!isCamera && startFrame > 0 && capture_.set(cv::CAP_PROP_POS_FRAMES, startFrame)) startFrame_ = startFrame;
}

VideoReader::~VideoReader() { stop(); }

void VideoReader::start() {
    thread_ = std::jthread([this](std::stop_token st) { run(st); });
}

void VideoReader::stop() {
    if (thread_.joinable()) {
        thread_.request_stop();
        queue_.close();  // unblocks a producer waiting for room and any consumer
        thread_.join();
    }
}

std::optional<FramePacket> VideoReader::read() { return queue_.pop(); }

void VideoReader::run(std::stop_token stop) {
    using clock = std::chrono::steady_clock;
    int index = startFrame_;
    while (!stop.stop_requested()) {
        FramePacket packet;
        const auto t0 = clock::now();
        if (!capture_.read(packet.frame) || packet.frame.empty()) break;
        packet.decodeMs = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        packet.index = index++;
        if (!queue_.push(std::move(packet))) return;
    }
    queue_.close();  // end of stream
}

}  // namespace ft
