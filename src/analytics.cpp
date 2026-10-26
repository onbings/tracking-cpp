/**
 * @file analytics.cpp
 * @brief Implementation of trails, heatmap and line counter.
 */
#include "analytics.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace ft {

void Trails::add(int id, const cv::Point2f& canvasPt) {
    auto& t = trails_[id];
    t.push_back(canvasPt);
    while (t.size() > maxLength_) t.pop_front();
}

void Trails::warp(const cv::Matx33d& h) {
    for (auto& [id, trail] : trails_) {
        for (auto& p : trail) {
            const cv::Vec3d q = h * cv::Vec3d(p.x, p.y, 1.0);
            p = {static_cast<float>(q[0] / q[2]), static_cast<float>(q[1] / q[2])};
        }
    }
}

void Trails::keepOnly(const std::vector<int>& alive) {
    for (auto it = trails_.begin(); it != trails_.end();) {
        it = (std::ranges::find(alive, it->first) == alive.end()) ? trails_.erase(it) : std::next(it);
    }
}

const std::deque<cv::Point2f>& Trails::get(int id) const {
    auto it = trails_.find(id);
    return it == trails_.end() ? empty_ : it->second;
}

Heatmap::Heatmap(cv::Size canvasSize, int downscale)
    : downscale_(downscale),
      accumulator_(std::max(1, canvasSize.height / downscale), std::max(1, canvasSize.width / downscale), CV_32F,
                   cv::Scalar(0)) {}

void Heatmap::add(const std::vector<cv::Point2f>& canvasPts) {
    accumulator_ *= decay_;
    for (const auto& p : canvasPts) {
        cv::circle(accumulator_, {static_cast<int>(p.x / downscale_), static_cast<int>(p.y / downscale_)},
                   radius_, cv::Scalar(1.0), cv::FILLED);
    }
}

void Heatmap::warp(const cv::Matx33d& h) {
    // Scale the canvas homography to the reduced accumulator resolution: S * H * S^-1.
    const double s = 1.0 / downscale_;
    const cv::Matx33d scale(s, 0, 0, 0, s, 0, 0, 0, 1);
    const cv::Matx33d scaleInv(1 / s, 0, 0, 0, 1 / s, 0, 0, 0, 1);
    cv::Mat warpMat;
    cv::Mat(scale * h * scaleInv).convertTo(warpMat, CV_64F);
    cv::warpPerspective(accumulator_, accumulator_, warpMat, accumulator_.size(), cv::INTER_LINEAR);
}

void Heatmap::render(cv::Mat& canvas, double alpha) const {
    double peak = 0;
    cv::minMaxLoc(accumulator_, nullptr, &peak);
    if (peak <= 0) return;
    cv::Mat blurred, norm, colored;
    cv::GaussianBlur(accumulator_, blurred, {}, radius_);
    cv::normalize(blurred, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::resize(norm, norm, canvas.size(), 0, 0, cv::INTER_LINEAR);
    cv::applyColorMap(norm, colored, cv::COLORMAP_INFERNO);
    cv::Mat mask = norm > 12;
    cv::Mat blended;
    cv::addWeighted(canvas, 1 - alpha, colored, alpha, 0, blended);
    blended.copyTo(canvas, mask);
}

namespace {
/// @return 2D cross product: > 0 if p is left of a->b.
double sideOfLine(const cv::Point2d& a, const cv::Point2d& b, const cv::Point2d& p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}
/// @return true if segment p1->p2 crosses segment q1-q2.
bool segmentsIntersect(const cv::Point2d& p1, const cv::Point2d& p2, const cv::Point2d& q1, const cv::Point2d& q2) {
    const double d1 = sideOfLine(q1, q2, p1), d2 = sideOfLine(q1, q2, p2);
    const double d3 = sideOfLine(p1, p2, q1), d4 = sideOfLine(p1, p2, q2);
    return (d1 >= 0) != (d2 >= 0) && d3 * d4 <= 0;
}
}  // namespace

int LineCounter::update(int id, const cv::Point2d& foot, int frameIndex) {
    auto it = last_.find(id);
    const bool hadPrev = it != last_.end();
    const cv::Point2d prev = hadPrev ? it->second : foot;
    last_[id] = foot;
    if (!hadPrev || !segmentsIntersect(prev, foot, a_, b_)) return 0;
    lastEventFrame_ = frameIndex;
    if (sideOfLine(a_, b_, foot) >= 0) { ++countIn_; return 1; }
    ++countOut_;
    return -1;
}

void LineCounter::forget(const std::vector<int>& ids) {
    for (int id : ids) last_.erase(id);
}

}  // namespace ft
