/**
 * @file common.hpp
 * @brief Small value types and constants shared by every module of the football tracker.
 *
 * Coordinates convention used throughout the project:
 *  - **image** coordinates are full-resolution pixels of the source video (x right, y down);
 *  - **canvas** coordinates are pixels of the downscaled display image (`Config::displayWidth`);
 *  - **ground** coordinates are camera-centric metres on the pitch plane (X right, Y forward),
 *    produced by GroundModel.
 */
#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace ft {

/// COCO class id of "person".
inline constexpr int kPersonClass = 0;
/// COCO class id of "sports ball".
inline constexpr int kBallClass = 32;
/// Team id meaning "not a player of either team" (referee, coach, staff, spectator).
inline constexpr int kTeamOther = -1;

/**
 * @brief One object detection in full-resolution image pixels.
 */
struct Detection {
    cv::Rect2f box;      ///< Axis-aligned box (x, y, width, height) in image pixels.
    float score = 0.f;   ///< Detector confidence in [0, 1].
    int classId = 0;     ///< COCO class id.

    /// @return Bottom-centre of the box: the ground-contact ("foot") point of a standing person.
    [[nodiscard]] cv::Point2d foot() const {
        return {box.x + box.width * 0.5, box.y + box.height};
    }
    /// @return Centre of the box.
    [[nodiscard]] cv::Point2d center() const {
        return {box.x + box.width * 0.5, box.y + box.height * 0.5};
    }
};

/**
 * @brief A tracked object returned by the multi-object tracker for the current frame.
 */
struct Track {
    int id = 0;          ///< Persistent track identifier (unique for the lifetime of the tracker).
    cv::Rect2f box;      ///< Kalman-smoothed box in image pixels.
    float score = 0.f;   ///< Confidence of the detection matched this frame.
    int classId = 0;     ///< COCO class id.

    /// @return Bottom-centre of the box (foot point).
    [[nodiscard]] cv::Point2d foot() const {
        return {box.x + box.width * 0.5, box.y + box.height};
    }
};

/// @return Intersection-over-union of two boxes (0 when they don't overlap).
[[nodiscard]] inline double iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const double inter = (a & b).area();
    const double uni = a.area() + b.area() - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

}  // namespace ft
