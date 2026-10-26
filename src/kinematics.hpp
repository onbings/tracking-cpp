/**
 * @file kinematics.hpp
 * @brief Per-object speed and distance in metres, robust to a moving camera.
 *
 * Displacement is measured in the *current* ground frame as ground(foot_t) - ground(H*foot_{t-1}),
 * which cancels drone motion, then integrated into a dead-reckoned position. A constant-velocity
 * Kalman filter in metres smooths it. Gross innovations reset the filter (so an id switch or a bad
 * geometry does not pump the velocity), and the top speed must be sustained over a short window so a
 * single-frame glitch cannot set a record. See DOCUMENTATION.md §17 and Appendix B.
 */
#pragma once

#include "geometry.hpp"

#include <opencv2/core.hpp>

#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ft {

/**
 * @brief Motion state of one tracked object (player or ball).
 */
struct MotionState {
    int firstFrame = 0;         ///< First frame the object was observed.
    int lastFrame = 0;          ///< Last frame the object was observed.
    double distance = 0.0;      ///< Cumulative distance travelled (m).
    double maxSpeedKmh = 0.0;   ///< Sustained top speed (km/h).
    int updates = 0;            ///< Number of Kalman updates.
    bool hasGround = false;     ///< Whether `ground` is valid this frame.
    cv::Point2d ground{};       ///< Current ground position (m), if hasGround.

    /// @return current smoothed speed in km/h (0 before the filter is initialised).
    [[nodiscard]] double speedKmh() const { return hasFilter ? std::hypot(vx, vy) * 3.6 : 0.0; }

    // --- filter internals (public so the estimator can manipulate them cheaply) ---
    bool hasFilter = false;            ///< Kalman filter initialised.
    cv::Point2d prevImg{};             ///< Previous foot point (image px), warped each frame.
    bool hasPrev = false;              ///< Whether prevImg is valid.
    cv::Point2d z{};                   ///< Dead-reckoned position (m).
    double px = 0, py = 0, vx = 0, vy = 0;  ///< Kalman state [pos, velocity] in metres / (m/s).
    cv::Matx<double, 4, 4> cov = cv::Matx<double, 4, 4>::eye();  ///< State covariance.
    std::deque<double> recent;         ///< Recent speeds (km/h) for the sustained-maximum rule.
};

/**
 * @brief Estimates speed and distance for a set of keyed objects.
 */
class Kinematics {
public:
    /**
     * @param fps          video frame rate (for time steps and km/h).
     * @param accelSigma   process noise (white acceleration, m/s^2).
     * @param maxSpeedKmh  clip and jump-rejection threshold.
     * @param maxGap       frames of gap after which the filter restarts.
     * @param deadBandKmh  speeds below this don't accumulate distance.
     * @param sustainS     window (seconds) over which the top speed must be sustained.
     */
    Kinematics(double fps, double accelSigma, double maxSpeedKmh, int maxGap, double deadBandKmh, double sustainS);

    /// @brief Warps every stored previous point with the camera homography (call once per frame).
    void warp(const cv::Matx33d& h);
    /// @brief Marks camera motion as unknown this frame: no displacement is measured.
    void invalidate();
    /// @brief Forgets the filter of a key after a known discontinuity (e.g. a ball re-acquisition).
    void reset(const std::string& key);

    /**
     * @brief Integrates one observation of an object.
     * @param key        stable object key (track id as string, or "ball").
     * @param footImg    foot/contact point in image pixels.
     * @param ground     the current ground model.
     * @param frameIndex current frame index.
     * @param pixelSigma measurement noise in pixels (scaled by box/ball size by the caller).
     * @return reference to the (updated) motion state.
     */
    MotionState& observe(const std::string& key, const cv::Point2d& footImg, const GroundModel& ground,
                         int frameIndex, double pixelSigma);

    /// @brief Moves states not seen for `maxAge` frames to the finished list; returns the removed keys.
    std::vector<std::string> prune(int frameIndex, int maxAge);

    /// @return the state of a key, or nullptr.
    [[nodiscard]] const MotionState* find(const std::string& key) const;
    /// @return all states (active and finished), for the end-of-run report.
    [[nodiscard]] std::vector<std::pair<std::string, MotionState>> all() const;

private:
    void kfInit(MotionState& s, double sigmaM) const;

    double fps_, accelSigma_, maxSpeed_, deadBand_, pixelSigma_ = 2.0;
    int maxGap_, sustain_, minUpdatesForMax_;
    std::unordered_map<std::string, MotionState> states_;
    std::vector<std::pair<std::string, MotionState>> finished_;
};

}  // namespace ft
