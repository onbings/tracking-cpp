/**
 * @file analytics.hpp
 * @brief Per-id motion trails (image space), a decaying occupancy heatmap, and a line counter.
 *
 * Trails and the heatmap are stored in image/canvas space and warped every frame with the camera
 * homography, so they stay attached to the ground under a moving camera. See DOCUMENTATION.md §20, §28.
 */
#pragma once

#include <opencv2/core.hpp>

#include <deque>
#include <unordered_map>
#include <vector>

namespace ft {

/**
 * @brief Keeps a short trail of foot points per track id, in canvas pixels.
 */
class Trails {
public:
    /// @param maxLength number of points kept per track.
    explicit Trails(std::size_t maxLength) : maxLength_(maxLength) {}

    /// @brief Appends a canvas-space point for a track.
    void add(int id, const cv::Point2f& canvasPt);
    /// @brief Warps all trails with a canvas-space homography (follows camera motion).
    void warp(const cv::Matx33d& canvasHomography);
    /// @brief Drops trails of ids not in @p alive.
    void keepOnly(const std::vector<int>& alive);
    /// @return the trail of a track (empty if none).
    [[nodiscard]] const std::deque<cv::Point2f>& get(int id) const;

private:
    std::size_t maxLength_;
    std::unordered_map<int, std::deque<cv::Point2f>> trails_;
    std::deque<cv::Point2f> empty_;
};

/**
 * @brief Decaying occupancy heatmap accumulated at reduced resolution.
 */
class Heatmap {
public:
    /// @param canvasSize size of the display canvas.
    explicit Heatmap(cv::Size canvasSize, int downscale = 4);

    /// @brief Fades the accumulator and adds the current foot points (canvas pixels).
    void add(const std::vector<cv::Point2f>& canvasPts);
    /// @brief Warps the accumulator with a canvas-space homography.
    void warp(const cv::Matx33d& canvasHomography);
    /// @brief Blends the coloured heatmap onto the canvas.
    void render(cv::Mat& canvas, double alpha = 0.5) const;

private:
    int downscale_;
    double decay_ = 0.997;
    int radius_ = 4;
    cv::Mat accumulator_;  ///< CV_32F occupancy.
};

/**
 * @brief Counts tracks crossing a virtual line (image space).
 */
class LineCounter {
public:
    /// @param a,b endpoints of the line in image pixels.
    LineCounter(cv::Point2d a, cv::Point2d b) : a_(a), b_(b) {}

    /// @brief Updates with a track's foot point; returns +1 (in), -1 (out) or 0 (no crossing).
    int update(int id, const cv::Point2d& foot, int frameIndex);
    /// @brief Forgets the last position of removed ids.
    void forget(const std::vector<int>& ids);

    [[nodiscard]] cv::Point2d a() const { return a_; }        ///< First endpoint.
    [[nodiscard]] cv::Point2d b() const { return b_; }        ///< Second endpoint.
    [[nodiscard]] int countIn() const { return countIn_; }    ///< Crossings towards the positive side.
    [[nodiscard]] int countOut() const { return countOut_; }  ///< Crossings the other way.
    [[nodiscard]] int lastEventFrame() const { return lastEventFrame_; }  ///< For the flash effect.

private:
    cv::Point2d a_, b_;
    int countIn_ = 0, countOut_ = 0, lastEventFrame_ = -10000;
    std::unordered_map<int, cv::Point2d> last_;
};

}  // namespace ft
