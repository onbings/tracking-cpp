/**
 * @file ball.hpp
 * @brief Ball candidate filtering and a single-target ball tracker.
 *
 * COCO "sports ball" detections on a training pitch include disc cones, balls printed on banners and
 * several real balls. Three filters (shape, colour saturation, physical size) keep the real ball; a
 * gated nearest-neighbour tracker follows it, flags large jumps as re-acquisitions, and exposes a
 * predicted ROI for a high-resolution search when the ball is lost. See DOCUMENTATION.md §18.
 */
#pragma once

#include "common.hpp"
#include "geometry.hpp"

#include <opencv2/core.hpp>

#include <deque>
#include <optional>
#include <vector>

namespace ft {

/**
 * @brief Keeps only detections shaped and coloured like a real ball.
 * @param dets  COCO class-32 detections.
 * @param frame the current BGR frame (for the saturation test).
 * @return the plausible candidates.
 */
std::vector<Detection> ballCandidates(const std::vector<Detection>& dets, const cv::Mat& frame);

/**
 * @brief Keeps candidates whose pixel size matches a real ball at their ground point.
 * @param cands  candidates (already shape/colour filtered).
 * @param ground the calibrated ground model (pass-through if not ready).
 */
std::vector<Detection> plausibleBallSize(const std::vector<Detection>& cands, const GroundModel& ground);

/**
 * @brief Single-ball tracker: gated association + constant-velocity prediction + ROI support.
 */
class BallTracker {
public:
    /// @param fps video frame rate (sets the "lost" timeout and trail length scaling).
    explicit BallTracker(double fps, int trailLength = 45);

    /// @brief Warps the position and trail by the camera homography (call once per frame).
    void warp(const cv::Matx33d& h);

    /**
     * @brief Associates this frame's candidates with the ball.
     * @return true if the ball was observed.
     *
     * Sets reacquired() when the match is a plausible but large jump (re-acquisition), so the caller
     * can restart the ball kinematics instead of integrating a teleport.
     */
    bool update(const std::vector<Detection>& candidates);

    [[nodiscard]] bool active() const { return pos_.has_value(); }   ///< Ball currently tracked.
    [[nodiscard]] bool visible() const;                             ///< Seen recently and confirmed.
    [[nodiscard]] bool reacquired() const { return reacquired_; }    ///< Last match was a large jump.
    [[nodiscard]] cv::Point2d position() const { return pos_.value_or(cv::Point2d{}); }  ///< Centre (px).
    [[nodiscard]] double size() const { return size_; }              ///< Diameter estimate (px).
    [[nodiscard]] const std::deque<cv::Point2d>& trail() const { return trail_; }  ///< Recent centres.
    /// @return bottom-of-ball ground-contact point, or std::nullopt if inactive.
    [[nodiscard]] std::optional<cv::Point2d> groundPoint() const;

    /**
     * @brief Region to search at full resolution when the ball is lost.
     * @param frameSize source frame size.
     * @param size      square ROI side.
     * @return the ROI, or std::nullopt if there is no prediction.
     */
    [[nodiscard]] std::optional<cv::Rect> roi(cv::Size frameSize, int size = 640) const;

    [[nodiscard]] int detectedFrames() const { return detectedFrames_; }  ///< Frames the ball was seen.

    /// @brief Forgets the ball position, velocity and trail (lost for too long, or a shot cut).
    void reset();

private:
    [[nodiscard]] std::optional<cv::Point2d> predicted() const;

    double acquireConf_ = 0.3;
    double gatePx_ = 150.0;
    int maxLost_;
    std::optional<cv::Point2d> pos_;
    cv::Point2d vel_{};
    double size_ = 20.0;
    double conf_ = 0.0;
    int lost_ = 0;
    int hits_ = 0;
    int detectedFrames_ = 0;
    bool reacquired_ = false;
    std::deque<cv::Point2d> trail_;
    std::size_t trailLength_;
};

}  // namespace ft
