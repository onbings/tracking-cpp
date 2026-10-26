/**
 * @file kalman_box.hpp
 * @brief 8-state constant-velocity Kalman filter tracking an axis-aligned box.
 *
 * State: (cx, cy, aspect, height, vcx, vcy, vaspect, vheight). This is the standard SORT/BoT-SORT
 * parameterisation (centre, width/height aspect ratio and height, plus their velocities). Keeping the
 * header-only keeps the tracker self-contained.
 */
#pragma once

#include <opencv2/core.hpp>

namespace ft {

/**
 * @brief Minimal constant-velocity Kalman filter for a bounding box.
 */
class KalmanBox {
public:
    /// @brief Initialises the state from a box (zero velocity).
    explicit KalmanBox(const cv::Rect2f& box);

    /// @brief Predicts the next state (call once per frame before matching).
    void predict();
    /// @brief Corrects the state with a measured box.
    void update(const cv::Rect2f& box);
    /// @brief Applies a camera-motion homography to the predicted centre and size.
    void applyWarp(const cv::Matx33d& warp);

    /// @return the current box estimate (x, y, width, height).
    [[nodiscard]] cv::Rect2f box() const;

private:
    /// @return (cx, cy, aspect, height) measurement vector of a box.
    static cv::Vec4d toMeasurement(const cv::Rect2f& box);

    cv::Matx<double, 8, 1> x_;   ///< State mean.
    cv::Matx<double, 8, 8> p_;   ///< State covariance.
    double posStd_ = 1.0 / 20;   ///< Position noise weight (relative to height).
    double velStd_ = 1.0 / 160;  ///< Velocity noise weight.
};

inline cv::Vec4d KalmanBox::toMeasurement(const cv::Rect2f& box) {
    const double h = std::max(1.0f, box.height);
    return {box.x + box.width * 0.5, box.y + box.height * 0.5, box.width / h, h};
}

inline KalmanBox::KalmanBox(const cv::Rect2f& box) {
    const cv::Vec4d m = toMeasurement(box);
    x_ = cv::Matx<double, 8, 1>::zeros();
    for (int i = 0; i < 4; ++i) x_(i, 0) = m[i];
    p_ = cv::Matx<double, 8, 8>::eye();
    const double h = m[3];
    const double p[8] = {2 * posStd_ * h, 2 * posStd_ * h, 1e-2, 2 * posStd_ * h,
                         10 * velStd_ * h, 10 * velStd_ * h, 1e-5, 10 * velStd_ * h};
    for (int i = 0; i < 8; ++i) p_(i, i) = p[i] * p[i];
}

inline void KalmanBox::predict() {
    // Constant velocity: position += velocity.
    for (int i = 0; i < 4; ++i) x_(i, 0) += x_(i + 4, 0);
    const double h = std::max(1.0, x_(3, 0));
    cv::Matx<double, 8, 8> f = cv::Matx<double, 8, 8>::eye();
    for (int i = 0; i < 4; ++i) f(i, i + 4) = 1.0;
    p_ = f * p_ * f.t();
    const double q[8] = {posStd_ * h, posStd_ * h, 1e-2, posStd_ * h,
                         velStd_ * h, velStd_ * h, 1e-5, velStd_ * h};
    for (int i = 0; i < 8; ++i) p_(i, i) += q[i] * q[i];
}

inline void KalmanBox::update(const cv::Rect2f& box) {
    const cv::Vec4d z = toMeasurement(box);
    const double h = std::max(1.0, x_(3, 0));
    const double r[4] = {posStd_ * h, posStd_ * h, 1e-1, posStd_ * h};
    // Standard Kalman update, exploiting H = [I4 | 0].
    for (int i = 0; i < 4; ++i) {
        const double s = p_(i, i) + r[i] * r[i];
        const double k = p_(i, i) / s;  // diagonal gain (cross terms are small for this model)
        const double innov = z[i] - x_(i, 0);
        x_(i, 0) += k * innov;
        x_(i + 4, 0) += (p_(i + 4, i) / s) * innov;
        p_(i, i) *= (1 - k);
    }
}

inline void KalmanBox::applyWarp(const cv::Matx33d& warp) {
    const cv::Vec3d c = warp * cv::Vec3d(x_(0, 0), x_(1, 0), 1.0);
    x_(0, 0) = c[0] / c[2];
    x_(1, 0) = c[1] / c[2];
    // Scale the height and velocities by the homography's local scale factor.
    const double scale = std::sqrt(std::abs(warp(0, 0) * warp(1, 1) - warp(0, 1) * warp(1, 0)));
    x_(3, 0) *= scale;
    for (int i = 4; i < 8; ++i) x_(i, 0) *= scale;
}

inline cv::Rect2f KalmanBox::box() const {
    const double aspect = x_(2, 0), h = x_(3, 0), w = aspect * h;
    return {static_cast<float>(x_(0, 0) - w * 0.5), static_cast<float>(x_(1, 0) - h * 0.5),
            static_cast<float>(w), static_cast<float>(h)};
}

}  // namespace ft
