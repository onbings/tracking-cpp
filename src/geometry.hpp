/**
 * @file geometry.hpp
 * @brief Self-calibrating pinhole ground model: the core of every metric output.
 *
 * Assumes a flat pitch (Z = 0), a pinhole camera with known horizontal FOV, principal point at the
 * image centre, and **zero roll** (gimbal-stabilised drone). Two unknowns are fitted from the people
 * in view: the camera tilt θ and height h_c. A person of height `personHeight` standing at a detected
 * foot point has a predictable image height; fitting θ and h_c so predicted matches observed box
 * heights (robust least squares over a sliding window) calibrates the camera with no manual input.
 *
 * Ground coordinates are camera-centric metres: X to the right, Y forward (depth). See
 * DOCUMENTATION.md §14 and Appendix A for the derivation.
 */
#pragma once

#include <opencv2/core.hpp>

#include <deque>
#include <limits>
#include <optional>
#include <vector>

namespace ft {

/**
 * @brief Monocular ground model with online self-calibration.
 */
class GroundModel {
public:
    /**
     * @param width,height source frame size in pixels.
     * @param hfovDeg      horizontal field of view in degrees.
     * @param personHeight assumed person height in metres (scales every metric output linearly).
     * @param maxDepth     ground points beyond this depth (m) are treated as invalid.
     */
    GroundModel(int width, int height, double hfovDeg, double personHeight, double maxDepth);

    /// @brief Overrides the calibration (used by tests); marks the model ready.
    void setParams(double tiltRad, double camHeight);

    /**
     * @brief Forgets the calibration after a shot cut: the new shot may come from another camera.
     *
     * The current tilt/height stay as the starting guess, but the model reports not-ready (so no
     * metric output) until the next fit, which is taken as-is rather than blended with the old one.
     */
    void reset();

    /**
     * @brief Adds this frame's person boxes to the fitting window (border/short/implausible ones dropped).
     * @param boxes person detections in image pixels.
     */
    void addObservations(const std::vector<cv::Rect2f>& boxes);

    /**
     * @brief Refits θ and h_c every few frames from the windowed observations.
     * @param frameIndex current frame index (controls the refit cadence).
     * @return true if the parameters changed this call.
     */
    bool update(int frameIndex);

    [[nodiscard]] bool ready() const { return ready_; }           ///< True once calibrated.
    [[nodiscard]] int width() const { return width_; }            ///< Source frame width (px).
    [[nodiscard]] int height() const { return height_; }          ///< Source frame height (px).
    [[nodiscard]] double tiltDeg() const;                         ///< Camera tilt in degrees.
    [[nodiscard]] double camHeight() const { return camHeight_; } ///< Camera height in metres.
    [[nodiscard]] double residual() const { return residual_; }   ///< Median |log(h_pred/h_obs)| of the last fit.
    [[nodiscard]] double horizonRow() const;                      ///< Image row of the horizon.

    /**
     * @brief Maps an image point to ground metres.
     * @param[out] valid set to false if the point is above the horizon or beyond maxDepth.
     * @return (X, Y) in metres (meaningless when !valid).
     */
    [[nodiscard]] cv::Point2d toGround(const cv::Point2d& image, bool& valid) const;

    /**
     * @brief Maps a ground point (metres) to an image point.
     * @param[out] valid set to false if the point is behind the camera.
     */
    [[nodiscard]] cv::Point2d toImage(const cv::Point2d& ground, bool& valid) const;

    /// @return ground distance (m) of a one-pixel vertical step at @p image (worst-conditioned direction).
    [[nodiscard]] double metersPerPixel(const cv::Point2d& image) const;
    /// @return image width (px) of an object @p sizeM metres wide lying on the ground under @p image.
    [[nodiscard]] double expectedSizePx(const cv::Point2d& image, double sizeM) const;

    /// @return true if the box height is plausible for a person standing at its foot point.
    [[nodiscard]] bool plausiblePerson(const cv::Rect2f& box, double lo = 0.45, double hi = 2.2) const;

    /// @return predicted image height (px) of a `personHeight`-tall person whose feet are at @p foot.
    [[nodiscard]] double personBoxHeightPx(const cv::Point2d& foot) const {
        return predictedHeight(foot.x, foot.y, tilt_, camHeight_);
    }

    /// @brief Metric bounding box (xmin, ymin, xmax, ymax) of the visible ground; empty if not ready.
    [[nodiscard]] std::optional<cv::Vec4d> visibleBounds() const;

private:
    /// @brief One windowed observation: box centre column, foot row and box height (pixels).
    struct Obs { double u, vFoot, height; };

    [[nodiscard]] cv::Point2d groundRaw(double u, double v, double tilt, double camH, bool& valid) const;
    [[nodiscard]] double predictedHeight(double u, double vFoot, double tilt, double camH) const;

    int width_, height_;
    double cx_, cy_, f_;
    double personHeight_, maxDepth_;
    double tilt_;      ///< Current tilt (radians).
    double camHeight_; ///< Current camera height (metres).
    bool ready_ = false;
    double residual_ = std::numeric_limits<double>::quiet_NaN();
    std::deque<std::vector<Obs>> window_;  ///< Per-frame observations (last ~30 frames).
    static constexpr std::size_t kWindow = 30;
    static constexpr int kRefitEvery = 5;
    static constexpr double kSmoothing = 0.35;  ///< EMA factor for successive fits.
};

}  // namespace ft
