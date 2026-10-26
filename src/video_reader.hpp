/**
 * @file video_reader.hpp
 * @brief Decodes the video on a background thread into a bounded queue of frames.
 */
#pragma once

#include "bounded_queue.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <optional>
#include <string>
#include <thread>

namespace ft {

/**
 * @brief One decoded frame plus bookkeeping.
 */
struct FramePacket {
    int index = 0;          ///< Frame index in the source (starts at the requested start frame).
    cv::Mat frame;          ///< BGR image at source resolution.
    double decodeMs = 0.0;  ///< Time spent decoding this frame (concurrent "decode" stage).
};

/**
 * @brief Threaded video decoder.
 *
 * On Windows the Media Foundation backend is tried first with hardware acceleration requested
 * (`CAP_PROP_HW_ACCELERATION = ANY`), which offloads 4K H.264 decoding to the GPU when available;
 * any other backend is used as a fallback. Decoding then overlaps inference and analytics.
 */
class VideoReader {
public:
    /**
     * @brief Opens the source (file path or webcam index such as "0").
     * @param source     path or camera index.
     * @param queueSize  read-ahead depth.
     * @param startFrame frame to seek to before decoding (ignored for cameras).
     * @throws std::runtime_error if the source cannot be opened.
     */
    VideoReader(const std::string& source, int queueSize, int startFrame);
    /// @brief Stops the thread and releases the capture.
    ~VideoReader();
    VideoReader(const VideoReader&) = delete;
    VideoReader& operator=(const VideoReader&) = delete;

    /// @brief Starts the decoding thread.
    void start();
    /// @brief Blocks for the next frame; std::nullopt at end of stream.
    std::optional<FramePacket> read();
    /// @brief Stops decoding early (unblocks read()).
    void stop();

    [[nodiscard]] int width() const { return width_; }        ///< Frame width in pixels.
    [[nodiscard]] int height() const { return height_; }      ///< Frame height in pixels.
    [[nodiscard]] double fps() const { return fps_; }         ///< Frames per second used for timing (see fpsSource()).
    /**
     * @brief Where fps() comes from: "stream" (container metadata), "fallback" (metadata missing or
     *        implausible, 30 assumed) or "override" (--fps).
     *
     * Every speed is distance × fps, so a wrong frame rate scales all km/h values proportionally.
     */
    [[nodiscard]] const std::string& fpsSource() const { return fpsSource_; }
    /// @brief Forces the frame rate (the --fps option), e.g. for files with broken metadata.
    void setFps(double fps) { fps_ = fps; fpsSource_ = "override"; }
    /// @brief Plausible frame-rate range for container metadata; values outside it are rejected.
    static constexpr double kMinFps = 5.0, kMaxFps = 240.0;
    [[nodiscard]] int frameCount() const { return frameCount_; }  ///< Total frames (0 if unknown).
    [[nodiscard]] int startFrame() const { return startFrame_; }  ///< Effective first frame index.
    [[nodiscard]] const std::string& backend() const { return backend_; }  ///< OpenCV backend name.
    [[nodiscard]] bool hardwareAccelerated() const { return hwAccel_; }    ///< True if HW decode is active.
    [[nodiscard]] std::size_t queueDepth() const { return queue_.size(); } ///< Frames waiting.
    [[nodiscard]] std::size_t queueCapacity() const { return queue_.capacity(); }  ///< Queue bound.

private:
    void run(std::stop_token stop);

    cv::VideoCapture capture_;
    BoundedQueue<FramePacket> queue_;
    std::jthread thread_;
    int width_ = 0, height_ = 0, frameCount_ = 0, startFrame_ = 0;
    double fps_ = 30.0;
    std::string fpsSource_ = "stream";
    std::string backend_;
    bool hwAccel_ = false;
};

}  // namespace ft
