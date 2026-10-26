/**
 * @file camera_motion.hpp
 * @brief Frame-to-frame camera (drone) motion estimated from features on the pitch only.
 *
 * Points on the planar pitch, seen from two camera poses, are related by a homography. By tracking
 * KLT features that lie on the turf (and masking out the players), the estimated homography is the
 * **ground-plane** motion between frames. It is shared by the tracker (motion compensation), the
 * kinematics (to cancel camera motion), the ball, the trails and the heatmap.
 */
#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <vector>

namespace ft {

/// @return binary mask (255 = turf) from the artificial/natural grass colour in HSV.
cv::Mat grassMask(const cv::Mat& bgr);

/**
 * @brief Estimates the pitch homography between consecutive frames.
 */
class CameraMotion {
public:
    /**
     * @param fullSize  source frame size (for scaling the homography back to full resolution).
     * @param procWidth width at which features are tracked (smaller = faster, less precise).
     */
    explicit CameraMotion(cv::Size fullSize, int procWidth = 640);

    /**
     * @brief Processes a frame and returns the previous-to-current homography (full resolution).
     * @param image        current frame at any resolution (internally downscaled to procWidth).
     * @param excludeBoxes player boxes (full-resolution pixels) to exclude from feature detection.
     * @return homography H such that `current ≈ H * previous`; identity on the first frame or on failure.
     */
    const cv::Matx33d& update(const cv::Mat& image, const std::vector<cv::Rect2f>& excludeBoxes);

    [[nodiscard]] bool ok() const { return ok_; }               ///< True if the last estimate succeeded.
    /**
     * @brief True if the last frame starts a new shot (editing cut, replay, scene change).
     *
     * A cut is declared when the colour histogram changes abruptly (correlation below
     * kCutCorrelation), or when the turf features cannot be followed at all while the histogram
     * also moved noticeably. Continuous drone/tactical footage never triggers it; broadcast
     * footage triggers it at every camera switch.
     */
    [[nodiscard]] bool cut() const { return cut_; }
    /// @brief Histogram correlation (−1..1) between the last two frames (1 = identical colours).
    [[nodiscard]] double histCorrelation() const { return histCorr_; }
    /// @brief Fraction (0..1) of the frame classified as turf.
    [[nodiscard]] double turfRatio() const { return turfRatio_; }
    /// @brief Fraction (0..1) of the lower half of the frame classified as turf (the pitch-view signal).
    [[nodiscard]] double lowerTurfRatio() const { return lowerTurfRatio_; }
    /**
     * @brief True if the camera is looking at the pitch, so the ground-plane analytics are meaningful.
     *
     * Uses the lower half of the frame: green banners or seats can fill the top of a close-up and fool
     * a whole-frame turf ratio, but only a pitch view has grass under the camera.
     */
    [[nodiscard]] bool pitchView() const { return lowerTurfRatio_ >= kMinTurfRatio; }

    static constexpr double kCutCorrelation = 0.5;   ///< Below this histogram correlation: hard cut.
    static constexpr double kSoftCutCorrelation = 0.85;  ///< Below this and KLT failure: cut.
    static constexpr double kMinTurfRatio = 0.70;    ///< Minimum lower-half turf fraction for a pitch view.
    [[nodiscard]] int inliers() const { return inliers_; }       ///< RANSAC inlier count.
    [[nodiscard]] const cv::Matx33d& homography() const { return homography_; }  ///< Last H (full res).
    [[nodiscard]] const cv::Mat& grass() const { return grass_; } ///< Turf mask at processing scale.
    [[nodiscard]] double scale() const { return scale_; }         ///< procWidth / fullWidth.

    /// @brief Applies a homography to image points.
    static std::vector<cv::Point2d> warpPoints(const cv::Matx33d& h, const std::vector<cv::Point2d>& pts);
    /// @brief Applies a homography to a single point.
    static cv::Point2d warpPoint(const cv::Matx33d& h, const cv::Point2d& p);

private:
    cv::Size fullSize_;
    cv::Size procSize_;
    double scale_;
    int maxCorners_ = 300;
    cv::Mat prevGray_;
    cv::Mat prevMask_;
    cv::Mat grass_;
    cv::Matx33d homography_ = cv::Matx33d::eye();
    int inliers_ = 0;
    bool ok_ = false;
    cv::Mat prevHist_;
    double histCorr_ = 1.0;
    double turfRatio_ = 1.0;
    double lowerTurfRatio_ = 1.0;
    bool cut_ = false;
};

}  // namespace ft
