/**
 * @file kinematics.cpp
 * @brief Implementation of the camera-compensated metric Kalman filter.
 */
#include "kinematics.hpp"

#include "camera_motion.hpp"

#include <algorithm>
#include <cmath>

namespace ft {

Kinematics::Kinematics(double fps, double accelSigma, double maxSpeedKmh, int maxGap, double deadBandKmh, double sustainS)
    : fps_(fps),
      accelSigma_(accelSigma),
      maxSpeed_(maxSpeedKmh),
      deadBand_(deadBandKmh / 3.6),
      maxGap_(maxGap),
      sustain_(std::max(1, static_cast<int>(std::lround(fps * sustainS)))),
      minUpdatesForMax_(std::max(1, static_cast<int>(fps * 0.75))) {}

void Kinematics::warp(const cv::Matx33d& h) {
    for (auto& [key, s] : states_) {
        if (s.hasPrev) s.prevImg = CameraMotion::warpPoint(h, s.prevImg);
    }
}

void Kinematics::invalidate() {
    for (auto& [key, s] : states_) s.hasPrev = false;
}

void Kinematics::reset(const std::string& key) {
    if (auto it = states_.find(key); it != states_.end()) {
        it->second.hasPrev = false;
        it->second.hasFilter = false;
        it->second.recent.clear();
    }
}

void Kinematics::kfInit(MotionState& s, double sigmaM) const {
    // Start at rest with a wide velocity prior: a single frame-to-frame delta is mostly noise.
    const double vSigma = maxSpeed_ / 3.6 / 2.0;
    s.px = s.z.x; s.py = s.z.y; s.vx = 0; s.vy = 0;
    s.cov = cv::Matx<double, 4, 4>::zeros();
    s.cov(0, 0) = sigmaM * sigmaM; s.cov(1, 1) = sigmaM * sigmaM;
    s.cov(2, 2) = vSigma * vSigma; s.cov(3, 3) = vSigma * vSigma;
    s.hasFilter = true;
    s.recent.clear();
}

MotionState& Kinematics::observe(const std::string& key, const cv::Point2d& footImg, const GroundModel& ground,
                                 int frameIndex, double pixelSigma) {
    MotionState& s = states_[key];
    if (s.lastFrame == 0 && s.firstFrame == 0 && s.updates == 0 && !s.hasFilter && !s.hasPrev) {
        s.firstFrame = frameIndex;
    }

    bool okNow = false;
    const cv::Point2d gNow = ground.toGround(footImg, okNow);
    s.hasGround = okNow;
    if (okNow) s.ground = gNow;

    const int gap = frameIndex - s.lastFrame;
    if (!s.hasPrev || gap <= 0 || gap > maxGap_ || !okNow) {
        s.prevImg = footImg;
        s.hasPrev = true;
        s.lastFrame = frameIndex;
        if (gap > maxGap_) s.hasFilter = false;  // long occlusion: restart, keep accumulated stats
        return s;
    }

    bool okPrev = false;
    const cv::Point2d gPrev = ground.toGround(s.prevImg, okPrev);
    s.prevImg = footImg;
    s.lastFrame = frameIndex;
    if (!okPrev) return s;

    const double dt = gap / fps_;
    const cv::Point2d delta = gNow - gPrev;
    if (cv::norm(delta) / dt > 2.0 * maxSpeed_ / 3.6) {  // implausible jump (id switch / bad geometry)
        s.hasFilter = false;
        return s;
    }
    s.z += delta;

    const double sigmaM = pixelSigma * ground.metersPerPixel(footImg);
    if (!s.hasFilter) { kfInit(s, sigmaM); return s; }

    // --- full-matrix constant-velocity Kalman (identical to the Python reference) ---
    // Keeping the complete 4x4 covariance (not a diagonal approximation) matters: the position/
    // velocity cross-covariance is what lets the velocity estimate react to real accelerations, so
    // a diagonal shortcut over-smooths and under-reports peak speeds.
    using Vec4 = cv::Matx<double, 4, 1>;
    using Mat4 = cv::Matx<double, 4, 4>;
    using Mat2 = cv::Matx<double, 2, 2>;
    using Mat42 = cv::Matx<double, 4, 2>;
    using Mat24 = cv::Matx<double, 2, 4>;
    using Vec2 = cv::Matx<double, 2, 1>;

    const cv::Point2d prevPos(s.px, s.py);  // pre-predict position, for the distance increment
    Vec4 x(s.px, s.py, s.vx, s.vy);
    Mat4& P = s.cov;

    // predict: x = F x,  P = F P F^T + G G^T q
    Mat4 F = Mat4::eye();
    F(0, 2) = dt; F(1, 3) = dt;
    const Mat42 G(dt * dt / 2, 0, 0, dt * dt / 2, dt, 0, 0, dt);
    const double q = accelSigma_ * accelSigma_;
    x = F * x;
    P = F * P * F.t() + (G * G.t()) * q;

    // update: innovation y = z - H x, gated by its Mahalanobis distance
    const Mat24 Hm(1, 0, 0, 0, 0, 1, 0, 0);
    const double r = std::max(sigmaM, 0.02);
    Mat2 R = Mat2::eye() * (r * r);
    const Vec2 z(s.z.x, s.z.y);
    const Vec2 y = z - Hm * x;
    Mat2 S = Hm * P * Hm.t() + R;
    double d2 = (y.t() * S.inv() * y)(0, 0);
    if (d2 > 36.0) {  // > 6 sigma: a discontinuity (merged boxes, id switch) -> restart the filter
        kfInit(s, sigmaM);
        return s;
    }
    if (d2 > 9.0) {  // 3-6 sigma: occlusion jitter, trust the measurement less
        R *= d2 / 9.0;
        S = Hm * P * Hm.t() + R;
    }
    const Mat42 K = P * Hm.t() * S.inv();
    x = x + K * y;
    P = (Mat4::eye() - K * Hm) * P;
    s.px = x(0); s.py = x(1); s.vx = x(2); s.vy = x(3);

    s.updates += 1;
    const double speed = std::hypot(s.vx, s.vy);
    if (speed > deadBand_) s.distance += std::hypot(s.px - prevPos.x, s.py - prevPos.y);
    s.recent.push_back(speed * 3.6);
    if (static_cast<int>(s.recent.size()) > sustain_) s.recent.pop_front();
    if (s.updates >= minUpdatesForMax_ && static_cast<int>(s.recent.size()) == sustain_) {
        // Sustained maximum: a single-frame glitch cannot set a record.
        const double sustained = *std::ranges::min_element(s.recent);
        s.maxSpeedKmh = std::max(s.maxSpeedKmh, std::min(sustained, maxSpeed_));
    }
    return s;
}

std::vector<std::string> Kinematics::prune(int frameIndex, int maxAge) {
    std::vector<std::string> removed;
    for (auto it = states_.begin(); it != states_.end();) {
        if (frameIndex - it->second.lastFrame > maxAge) {
            finished_.emplace_back(it->first, it->second);
            removed.push_back(it->first);
            it = states_.erase(it);
        } else {
            ++it;
        }
    }
    return removed;
}

const MotionState* Kinematics::find(const std::string& key) const {
    auto it = states_.find(key);
    return it == states_.end() ? nullptr : &it->second;
}

std::vector<std::pair<std::string, MotionState>> Kinematics::all() const {
    auto out = finished_;
    for (const auto& kv : states_) out.push_back(kv);
    return out;
}

}  // namespace ft
