/**
 * @file geometry.cpp
 * @brief Implementation of the self-calibrating ground model.
 *
 * The fit minimises Σ ρ( log(h_pred_i / h_obs_i) ) over (θ, h_c) with ρ = soft_l1 (robust to adults,
 * bending kids and merged boxes). It uses iteratively reweighted Gauss-Newton with numerical
 * derivatives: at each iteration a residual gets the soft_l1 weight w = 1/sqrt(1 + (r/s)^2), so the
 * normal equations approximate the robust optimum. Two unknowns and a short window make this cheap
 * (well under 1 ms).
 */
#include "geometry.hpp"

#include <algorithm>
#include <cmath>

namespace ft {

GroundModel::GroundModel(int width, int height, double hfovDeg, double personHeight, double maxDepth)
    : width_(width),
      height_(height),
      cx_(width / 2.0),
      cy_(height / 2.0),
      f_((width / 2.0) / std::tan(hfovDeg * CV_PI / 180.0 / 2.0)),
      personHeight_(personHeight),
      maxDepth_(maxDepth),
      tilt_(25.0 * CV_PI / 180.0),
      camHeight_(8.0) {}

void GroundModel::setParams(double tiltRad, double camHeight) {
    tilt_ = tiltRad;
    camHeight_ = camHeight;
    ready_ = true;
}

void GroundModel::reset() {
    window_.clear();
    ready_ = false;
    residual_ = std::numeric_limits<double>::quiet_NaN();
}

double GroundModel::tiltDeg() const { return tilt_ * 180.0 / CV_PI; }
double GroundModel::horizonRow() const { return cy_ - f_ * std::tan(tilt_); }

cv::Point2d GroundModel::groundRaw(double u, double v, double tilt, double camH, bool& valid) const {
    const double x = (u - cx_) / f_;
    const double y = (v - cy_) / f_;
    const double st = std::sin(tilt), ct = std::cos(tilt);
    const double down = st + y * ct;  // -Z component of the ray; > 0 means it hits the ground
    valid = down > 1e-6;
    if (!valid) return {0, 0};
    const double t = camH / down;
    return {t * x, t * (ct - y * st)};
}

cv::Point2d GroundModel::toGround(const cv::Point2d& image, bool& valid) const {
    cv::Point2d g = groundRaw(image.x, image.y, tilt_, camHeight_, valid);
    valid = valid && ready_ && g.y > 0 && g.y <= maxDepth_;
    return g;
}

cv::Point2d GroundModel::toImage(const cv::Point2d& ground, bool& valid) const {
    const double st = std::sin(tilt_), ct = std::cos(tilt_);
    const double z = ground.y * ct + camHeight_ * st;
    const double yCam = -ground.y * st + camHeight_ * ct;
    valid = z > 1e-6;
    if (!valid) return {0, 0};
    return {cx_ + f_ * ground.x / z, cy_ + f_ * yCam / z};
}

double GroundModel::metersPerPixel(const cv::Point2d& image) const {
    bool ok0 = false, ok1 = false;
    const cv::Point2d g0 = groundRaw(image.x, image.y, tilt_, camHeight_, ok0);
    const cv::Point2d g1 = groundRaw(image.x, image.y + 1.0, tilt_, camHeight_, ok1);
    if (!ok0 || !ok1) return std::numeric_limits<double>::infinity();
    return cv::norm(g1 - g0);
}

double GroundModel::expectedSizePx(const cv::Point2d& image, double sizeM) const {
    bool ok0 = false, ok1 = false;
    const cv::Point2d g0 = groundRaw(image.x, image.y, tilt_, camHeight_, ok0);
    const cv::Point2d g1 = groundRaw(image.x + 1.0, image.y, tilt_, camHeight_, ok1);
    if (!ok0 || !ok1) return 0.0;
    const double mpp = cv::norm(g1 - g0);
    return mpp > 0 ? sizeM / mpp : 0.0;
}

double GroundModel::predictedHeight(double u, double vFoot, double tilt, double camH) const {
    bool valid = false;
    const cv::Point2d g = groundRaw(u, vFoot, tilt, camH, valid);
    if (!valid) return 1e-3;
    const double st = std::sin(tilt), ct = std::cos(tilt);
    const double relZ = personHeight_ - camH;
    const double z = g.y * ct - relZ * st;
    const double yCam = -g.y * st - relZ * ct;
    if (z <= 0) return 1e-3;
    const double vTop = cy_ + f_ * yCam / z;
    return vFoot - vTop;
}

bool GroundModel::plausiblePerson(const cv::Rect2f& box, double lo, double hi) const {
    if (!ready_) return true;
    const double expected = predictedHeight(box.x + box.width * 0.5, box.y + box.height, tilt_, camHeight_);
    if (expected <= 1.0) return false;
    const double ratio = box.height / expected;
    return ratio >= lo && ratio <= hi;
}

void GroundModel::addObservations(const std::vector<cv::Rect2f>& boxes) {
    constexpr double margin = 4.0;
    std::vector<Obs> frame;
    for (const auto& b : boxes) {
        const bool onBorder = b.x <= margin || b.y <= margin ||
                              b.x + b.width >= width_ - margin || b.y + b.height >= height_ - margin;
        if (onBorder || b.height < 24.0) continue;
        if (ready_ && !plausiblePerson(b)) continue;  // don't feed the model banners/false positives
        frame.push_back({b.x + b.width * 0.5, b.y + b.height, b.height});
    }
    window_.push_back(std::move(frame));
    if (window_.size() > kWindow) window_.pop_front();
}

bool GroundModel::update(int frameIndex) {
    if (frameIndex % kRefitEvery != 0) return false;

    std::vector<Obs> obs;
    for (const auto& frame : window_) obs.insert(obs.end(), frame.begin(), frame.end());
    if (obs.size() < 12) return false;

    // Iteratively reweighted Gauss-Newton on p = (tilt, camHeight).
    constexpr double fScale = 0.08;  // soft_l1 scale ~8% relative error
    double tilt = tilt_, camH = camHeight_;
    const auto residual = [&](const Obs& o, double th, double h) {
        return std::log(std::max(1e-3, predictedHeight(o.u, o.vFoot, th, h)) / o.height);
    };

    for (int iter = 0; iter < 20; ++iter) {
        double jtj[2][2] = {{0, 0}, {0, 0}};
        double jtr[2] = {0, 0};
        const double dT = 1e-4, dH = 1e-3;
        for (const auto& o : obs) {
            const double r = residual(o, tilt, camH);
            const double gT = (residual(o, tilt + dT, camH) - r) / dT;
            const double gH = (residual(o, tilt, camH + dH) - r) / dH;
            const double w = 1.0 / std::sqrt(1.0 + (r / fScale) * (r / fScale));  // soft_l1 weight
            jtj[0][0] += w * gT * gT;
            jtj[0][1] += w * gT * gH;
            jtj[1][1] += w * gH * gH;
            jtr[0] += w * gT * r;
            jtr[1] += w * gH * r;
        }
        jtj[1][0] = jtj[0][1];
        const double lambda = 1e-6;  // tiny damping for conditioning
        const double a = jtj[0][0] + lambda, b = jtj[0][1], c = jtj[1][0], d = jtj[1][1] + lambda;
        const double det = a * d - b * c;
        if (std::abs(det) < 1e-12) break;
        const double stepT = -(d * jtr[0] - b * jtr[1]) / det;
        const double stepH = -(-c * jtr[0] + a * jtr[1]) / det;
        tilt = std::clamp(tilt + stepT, 1.0 * CV_PI / 180.0, 85.0 * CV_PI / 180.0);
        camH = std::clamp(camH + stepH, 1.5, 300.0);
        if (std::abs(stepT) < 1e-5 && std::abs(stepH) < 1e-4) break;
    }

    std::vector<double> absRes;
    absRes.reserve(obs.size());
    for (const auto& o : obs) absRes.push_back(std::abs(residual(o, tilt, camH)));
    std::ranges::nth_element(absRes, absRes.begin() + absRes.size() / 2);
    residual_ = absRes[absRes.size() / 2];

    const double alpha = ready_ ? kSmoothing : 1.0;  // first fit taken as-is, then EMA
    tilt_ = (1 - alpha) * tilt_ + alpha * tilt;
    camHeight_ = (1 - alpha) * camHeight_ + alpha * camH;
    ready_ = true;
    return true;
}

std::optional<cv::Vec4d> GroundModel::visibleBounds() const {
    if (!ready_) return std::nullopt;
    const int samples = 24;
    double xmin = 1e9, ymin = 1e9, xmax = -1e9, ymax = -1e9;
    bool any = false;
    const double halfW = maxDepth_ * (width_ / 2.0) / f_ * 1.2;
    auto consider = [&](double u, double v) {
        bool valid = false;
        cv::Point2d g = groundRaw(u, v, tilt_, camHeight_, valid);
        if (!valid || g.y <= 0) return;
        g.y = std::min(g.y, maxDepth_);
        const double x = std::clamp(g.x, -halfW, halfW);
        xmin = std::min(xmin, x); xmax = std::max(xmax, x);
        ymin = std::min(ymin, g.y); ymax = std::max(ymax, g.y);
        any = true;
    };
    for (int i = 0; i <= samples; ++i) {
        const double t = static_cast<double>(i) / samples;
        consider(t * (width_ - 1), height_ - 1);
        consider(t * (width_ - 1), 0);
        consider(0, t * (height_ - 1));
        consider(width_ - 1, t * (height_ - 1));
    }
    if (!any) return std::nullopt;
    return cv::Vec4d(xmin, ymin, xmax, ymax);
}

}  // namespace ft
