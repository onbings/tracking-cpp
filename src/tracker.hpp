/**
 * @file tracker.hpp
 * @brief BoT-SORT-style multi-object tracker with external camera-motion compensation.
 *
 * A compact re-implementation of the parts of BoT-SORT that mattered on this footage (the Python
 * PoC delegated to ultralytics). Each track runs an 8-state Kalman filter on the box; association is
 * IoU-based with the two-stage BYTE scheme (match high-confidence detections first, then recover
 * low-confidence ones). Before matching, every track prediction is warped by the pitch homography
 * from CameraMotion, so a moving drone does not break the association. Appearance/ReID is omitted:
 * the benchmarks showed it only hurt on identical-kit footage (DOCUMENTATION.md §15, §25).
 */
#pragma once

#include "common.hpp"
#include "kalman_box.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace ft {

/**
 * @brief Multi-object tracker producing stable ids.
 */
class BotSortTracker {
public:
    /**
     * @param highThresh detections at/above this score start and sustain tracks (the "--conf" value).
     * @param maxAge     frames a track survives unmatched before deletion.
     * @param nInit      consecutive hits before a new track is confirmed (reported).
     */
    explicit BotSortTracker(float highThresh, int maxAge, int nInit = 3);

    /**
     * @brief Advances the tracker by one frame.
     * @param detections all detections for the class(es) being tracked (any score).
     * @param warp       homography from the previous to the current frame (camera motion).
     * @return confirmed tracks visible this frame.
     */
    std::vector<Track> update(const std::vector<Detection>& detections, const cv::Matx33d& warp);

    /**
     * @brief Drops every track after a shot cut (positions from another shot are meaningless).
     *
     * Track ids keep increasing, so players seen after the cut never reuse a pre-cut id.
     */
    void reset() { tracks_.clear(); }

private:
    /// @brief Internal per-track state.
    struct Trk {
        int id;
        KalmanBox kf;
        int classId;
        float score;
        int hits = 1;         ///< Total matched frames.
        int age = 0;          ///< Frames since the last match.
        bool confirmed = false;
    };

    float highThresh_;
    float lowThresh_ = 0.1f;
    int maxAge_;
    int nInit_;
    int nextId_ = 1;
    std::vector<Trk> tracks_;
};

}  // namespace ft
